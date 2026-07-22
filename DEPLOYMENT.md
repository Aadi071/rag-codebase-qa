# Deploying CodeSage

The local build runs fully offline (bge-base embeddings + Ollama). **That build
cannot go to a free cloud tier**: PyTorch alone needs ~1GB and Ollama is a
desktop app. So the deployed build swaps both for hosted APIs via env vars --
no code changes, which is exactly why the backends are pluggable.

| | Local (default) | Deployed |
|---|---|---|
| Embeddings | bge-base (local, free, 768-dim) | OpenAI `text-embedding-3-small` (1536-dim) |
| LLM | Ollama (local, free) | OpenAI / Anthropic |
| Postgres | Docker pgvector | Managed Postgres with pgvector |
| Image | n/a | `Dockerfile` (~200MB, no torch) |

> **The one hard gotcha:** the two backends produce different vector sizes
> (768 vs 1536). A database indexed with one **cannot** be queried with the
> other. The deployed database is therefore a *separate* index, built with
> OpenAI embeddings. Never point the deployed app at your local DB.

---

## A. Run the whole stack in Docker locally (no accounts, no cloud)

Proves the container works before paying for anything.

```bash
export OPENAI_API_KEY=sk-...          # PowerShell: $env:OPENAI_API_KEY="sk-..."
docker compose -f docker-compose.prod.yml up --build
# open http://localhost:8000  (health: http://localhost:8000/healthz)
```

---

## B. Deploy to a public URL

### 1. Managed Postgres with pgvector
Use **Neon** or **Supabase** (both have pgvector on their free tier). Railway's
built-in Postgres does *not* reliably have it.

- Create a project, copy the connection string.
- Neon requires SSL — make sure the URL ends with `?sslmode=require`.
- The app runs `CREATE EXTENSION IF NOT EXISTS vector` on boot, and creates all
  tables itself, so there is no migration step. (On Supabase you may need to
  enable the `vector` extension once in the dashboard.)

### 2. Deploy the app
Push this repo to GitHub, then on **Railway** or **Render**: create a new service
from the repo. Both auto-detect the `Dockerfile` — no build config needed.

### 3. Environment variables
Set these on the service:

| Variable | Value | Notes |
|---|---|---|
| `DATABASE_URL` | from step 1 | include `?sslmode=require` on Neon |
| `EMBEDDING_BACKEND` | `openai` | **must not** be `local` in the cloud |
| `LLM_BACKEND` | `openai` | or `anthropic` |
| `OPENAI_API_KEY` | your key | needs billing credit |
| `SECRET_KEY` | long random string | signs session tokens |
| `ALLOW_INDEXING` | `false` (after seeding) | see step 5 |

Generate a secret: `python -c "import secrets;print(secrets.token_urlsafe(32))"`

Do **not** commit `.env` — it is in `.dockerignore` and `.gitignore`.

### 4. Verify
Open `https://<your-app>/healthz`. Expect:
```json
{"ok": true, "embeddings": "openai", "llm": "openai", "dim": 1536}
```
`ok: false` means the app is up but cannot reach Postgres — check `DATABASE_URL`
and SSL.

### 5. Seed the index (do this from your machine)
The deployed database starts empty. Indexing a repo takes minutes and would tie
up a small web dyno, so run the indexer **locally against the production DB**:

```bash
# temporarily, in your shell only:
export DATABASE_URL="<production connection string>"
export EMBEDDING_BACKEND=openai
export OPENAI_API_KEY=sk-...
python build_index.py https://github.com/pallets/click
```
This writes 1536-dim vectors into the production DB. Then set
`ALLOW_INDEXING=false` on the service so visitors can ask questions about the
pre-indexed repo but cannot trigger new (costly) indexing jobs.

To allow a *specific* set of repos instead:
`INDEX_ALLOWLIST=https://github.com/pallets/click,https://github.com/psf/requests`

### 6. Cost control
Embeddings are charged per token indexed (one-off per repo); answers are charged
per question and dominate ongoing cost. A demo is comfortably a few dollars, but
check current provider pricing. The protections that matter:
- `ALLOW_INDEXING=false` — the expensive path stays closed.
- The **answer cache** — repeat/near-duplicate questions never reach the LLM.
- Set a hard spend limit in your provider dashboard.

---

## Operational notes

- **Ephemeral disk.** Repos are shallow-cloned to a temp dir during indexing and
  are not needed afterwards; all durable state is in Postgres.
- **Cold starts.** Free tiers sleep. The first request may take ~30s.
- **Background indexing** runs in an in-process thread. Fine for one box; at real
  scale move it to a durable job queue (see README "What I'd change at 10x").
- **Rollback.** Redeploy the previous commit; the schema is additive
  (`CREATE TABLE IF NOT EXISTS` / `ADD COLUMN IF NOT EXISTS`), so older code runs
  against a newer database.
- **Teardown.** Delete the service and the Postgres project. No other state.
