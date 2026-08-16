# Deploying CodeSage

The local build runs fully offline (bge-base embeddings + Ollama). **That build
cannot go to a free cloud tier**: PyTorch alone needs ~1GB and Ollama is a
desktop app. So the deployed build swaps both for hosted APIs via env vars --
no code changes, which is exactly why the backends are pluggable.

The `openai` backend talks to **any OpenAI-compatible API**, not just OpenAI.
That's the trick that makes a **$0/month** deploy possible: point it at Google
**Gemini's** free-tier compatibility endpoint, which serves *both* the chat LLM
and the embeddings from a single API key.

| | Local (default) | Deployed — free (recommended) | Deployed — paid |
|---|---|---|---|
| Embeddings | bge-base (local, 768-dim) | Gemini `gemini-embedding-001` (768-dim) | OpenAI `text-embedding-3-small` (1536-dim) |
| LLM | Ollama (local) | Gemini `gemini-2.0-flash` | OpenAI / Anthropic |
| Postgres | Docker pgvector | Managed pgvector (Neon/Supabase) | same |
| Cost | free | **free** (Gemini free tier) | usage-based |
| Image | n/a | `Dockerfile` (~200MB, no torch) | same |

> **The one hard gotcha:** different embedding models produce different, *non-
> interchangeable* vectors. A database indexed with one embedder **cannot** be
> queried with another — even at the same dimension (local bge and Gemini are
> both 768-dim but live in different vector spaces). The deployed database is a
> *separate* index, built with whichever hosted embedder you deploy. Never point
> the deployed app at your local DB, and re-seed if you switch embedders.

---

## A. Get a free Gemini API key

1. Go to **https://aistudio.google.com/apikey** (sign in with a Google account).
2. Click **Create API key**. Copy it — this single key covers both chat and
   embeddings. No billing setup, no card.

Free-tier limits (per Google, subject to change) are generous for a portfolio
demo: flash models allow thousands of requests/day; embeddings allow ~100
requests/min. Seeding a repo (~700 chunks ≈ 7 batched embedding calls) and
answering demo questions sit comfortably inside them.

---

## B. Run the whole stack in Docker locally (no cloud, proves the container)

Do this once before deploying — it's the real smoke test of the production image.

```bash
# macOS / Linux
export GEMINI_KEY=AIza...            # PowerShell: $env:GEMINI_KEY="AIza..."
export OPENAI_API_KEY=$GEMINI_KEY
export OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
export EMBEDDING_BACKEND=openai
export EMBEDDING_MODEL=gemini-embedding-001
export EMBEDDING_DIM=768
export EMBEDDING_DIMENSIONS=768
export LLM_BACKEND=openai
export OPENAI_CHAT_MODEL=gemini-2.0-flash
docker compose -f docker-compose.prod.yml up --build
# open http://localhost:8000   (health: http://localhost:8000/healthz)
```

`docker-compose.prod.yml` also starts a pgvector Postgres, so this is a complete
self-contained run. Seed it (Section D) against `postgresql://rag:rag@localhost:5432/rag`.

---

## C. Deploy to a public URL

### 1. Managed Postgres with pgvector
Use **Neon** or **Supabase** (both have pgvector on their free tier). Railway's
built-in Postgres does *not* reliably have it.

- Create a project, copy the connection string.
- Neon requires SSL — make sure the URL ends with `?sslmode=require`.
- The app runs `CREATE EXTENSION IF NOT EXISTS vector` on boot and creates all
  tables itself — no migration step. (On Supabase you may need to enable the
  `vector` extension once in the dashboard.)

### 2. Deploy the app
Push this repo to GitHub, then on **Render** or **Railway**: create a new service
from the repo. Both auto-detect the `Dockerfile` — no build config needed. (Free
tiers sleep when idle; the first request after a nap takes ~30s.)

### 3. Environment variables (the free Gemini recipe)
Set these on the service:

| Variable | Value | Notes |
|---|---|---|
| `DATABASE_URL` | from step 1 | include `?sslmode=require` on Neon |
| `EMBEDDING_BACKEND` | `openai` | the generic OpenAI-compatible client |
| `OPENAI_BASE_URL` | `https://generativelanguage.googleapis.com/v1beta/openai/` | Gemini's compat endpoint (used for chat **and** embeddings) |
| `OPENAI_API_KEY` | your Gemini key | the single key from Section A |
| `EMBEDDING_MODEL` | `gemini-embedding-001` | |
| `EMBEDDING_DIM` | `768` | stored vector size |
| `EMBEDDING_DIMENSIONS` | `768` | requests 768-dim output (Matryoshka truncation) |
| `LLM_BACKEND` | `openai` | |
| `OPENAI_CHAT_MODEL` | `gemini-2.0-flash` | any current Gemini flash model works |
| `SECRET_KEY` | long random string | signs session tokens |
| `ALLOW_INDEXING` | `false` (after seeding) | see step 5 |

Generate a secret: `python -c "import secrets;print(secrets.token_urlsafe(32))"`

Do **not** commit `.env` — it is in `.dockerignore` and `.gitignore`.

> **Splitting providers (optional).** Groq has a faster free LLM tier but no
> embeddings API. To use Groq for chat + Gemini for embeddings, set the chat
> vars to Groq (`OPENAI_BASE_URL=https://api.groq.com/openai/v1`,
> `OPENAI_CHAT_MODEL=llama-3.3-70b-versatile`) and override the embedder
> separately with `EMBEDDING_BASE_URL` + `EMBEDDING_API_KEY` pointing at Gemini.

### 4. Verify
Open `https://<your-app>/healthz`. Expect:
```json
{"ok": true, "embeddings": "openai", "llm": "openai", "dim": 768}
```
`ok: false` means the app is up but cannot reach Postgres — check `DATABASE_URL`
and SSL.

### 5. Seed the index (do this from your machine)
The deployed database starts empty. Indexing a repo takes minutes and would tie
up a small web dyno, so run the indexer **locally against the production DB**
(Section D), then set `ALLOW_INDEXING=false` on the service so visitors can ask
questions about the pre-indexed repo but cannot trigger new indexing jobs.

To allow a *specific* set of repos instead of closing it entirely:
`INDEX_ALLOWLIST=https://github.com/pallets/click,https://github.com/psf/requests`

### 6. Cost control
On the Gemini free tier the ongoing cost is **$0** — the ceiling is rate limits,
not dollars. The protections that still matter:
- `ALLOW_INDEXING=false` — keeps the heavy path closed to the public.
- The **answer cache** — repeat/near-duplicate questions never reach the LLM,
  which also keeps you under the per-minute request cap.
- If you later switch to a paid provider, set a hard spend limit in its dashboard.

---

## D. Seeding command (local shell → production DB)

```bash
export DATABASE_URL="<production connection string, ?sslmode=require on Neon>"
export EMBEDDING_BACKEND=openai
export OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
export OPENAI_API_KEY=AIza...            # your Gemini key
export EMBEDDING_MODEL=gemini-embedding-001
export EMBEDDING_DIM=768
export EMBEDDING_DIMENSIONS=768
python build_index.py https://github.com/pallets/click
```
This writes 768-dim Gemini vectors into the production DB. Verify with
`python search.py "where is command aliasing handled?" --repo https://github.com/pallets/click`.

> If seeding errors with a batch-size / too-many-inputs message from Gemini, set
> `EMBEDDING_BATCH_SIZE=1` for the seed run (slower, but one input per request).

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
