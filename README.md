# RAG-Powered Codebase Q&A Assistant

Ask natural-language questions about any public GitHub repository and get
answers **grounded in the real code**, with `file:line` citations, instead of
hallucinated generalities. Point it at a repo, wait for indexing, then ask
"how are aliases resolved for a command?" and get a cited answer.

Runs **fully locally and free** by default: local embeddings (bge-base) and a
local LLM (Ollama), with an optional switch to hosted APIs (OpenAI / Anthropic).

---

**Supported languages:** Python, JavaScript, TypeScript/TSX, Go, Java, Rust (adding another is: install its tree-sitter wheel + 2 lines).

## What it does

1. **Ingests** a repo: shallow-clones it, filters out non-code files.
2. **Chunks** the code with tree-sitter, on **function/class boundaries** (not
   fixed line windows), keeping each chunk semantically whole and tagged with
   its file, line range, and symbol name.
3. **Embeds** each chunk and stores it in **pgvector** (Postgres).
4. **Retrieves** with a **hybrid** of dense vector search + BM25 keyword search,
   fused via **Reciprocal Rank Fusion (RRF)**.
5. **Answers** by feeding the retrieved chunks to an LLM under a strict grounded
   prompt: cite `path:line`, or say "I couldn't find this in the indexed code."
6. **Measures** retrieval quality against a hand-labeled question set.

## Architecture

```mermaid
flowchart TD
    U[GitHub repo URL] --> C[Clone + filter files]
    C --> K[AST chunking tree-sitter<br/>function/class boundaries + metadata]
    K --> E[Embed chunks<br/>bge-base local or OpenAI]
    E --> P[(pgvector<br/>chunk text + vector + file/line/symbol)]

    Q[User question] --> EQ[Embed query]
    EQ --> V[Vector search top-k]
    Q --> B[BM25 keyword search top-k]
    P --> V
    P --> B
    V --> F[RRF fusion]
    B --> F
    F --> G[Grounded prompt with citations]
    G --> L[LLM Ollama / API]
    L --> A[Cited answer + sources]
```

Two flows share one database: **index-time** (slow, once per repo, runs in a
background thread) and **query-time** (fast, every question).

## Tech stack

| Layer | Choice |
|-------|--------|
| Chunking | tree-sitter AST (Python, JavaScript, TypeScript/TSX, Go, Java, Rust) |
| Embeddings | bge-base via sentence-transformers (local) / OpenAI text-embedding-3-small |
| Vector store | pgvector (Postgres extension) + HNSW cosine index |
| Retrieval | Vector + BM25 (`rank-bm25`), fused with RRF |
| LLM | Ollama (local) / Anthropic / OpenAI |
| Backend | FastAPI (background-thread indexing + status polling) |
| Frontend | Single-page vanilla JS |

## Hardest decisions

**1. Chunk on AST boundaries, and never split a function.** Fixed-size chunks
slice functions in half and orphan them from context. We parse to an AST and
split on function/class boundaries. A function is kept whole even when it exceeds
the size budget; an oversized *class* is split into its methods, and **each
method chunk is prefixed with the class signature** and named `ClassName.method`
so it never loses its parent context. This directly improves what the LLM sees.

**2. Hybrid retrieval — built it, measured it, then turned it off.** I added BM25
alongside dense retrieval on the standard reasoning that embeddings miss exact
identifiers. A 28-question labeled eval showed the opposite on this codebase:
vector-only beat BM25 even on identifier lookups, because the AST chunks already
carry symbol names and class signatures. I swept fusion weights and fixed a
tokenizer bug before accepting the result. Ships vector-only by default, with
hybrid retained behind config and the eval kept as the evidence. (Full writeup in
Evaluation.)

**3. pgvector over a dedicated vector DB.** One fewer moving part: the chunk text,
its metadata, and its vector live in the same Postgres row and transaction. No
separate service to run or keep in sync.

**4. Local-first, but pluggable.** Embedding and LLM backends are chosen via
`.env` (`EMBEDDING_BACKEND`, `LLM_BACKEND`), so "swap the model" is a one-line
change. Default is zero-cost local (bge + Ollama).

## Evaluation

`evaluate.py` scores retrieval on a 28-question hand-labeled set
(`eval/eval_set.json`) — 18 conceptual ("How does Click parse options?") and 10
identifier lookups ("Where is `resolve_command` defined?"), each mapped to the
file that holds the answer. Measured on `pallets/click` with bge-base:

```
mode                        R@1      R@3      R@5      MRR
----------------------------------------------------------
vector                    75.0%    96.4%   100.0%   0.852
bm25                      42.9%    75.0%    82.1%   0.604
hybrid(bm25=0.25)         75.0%    89.3%    96.4%   0.832
hybrid(bm25=0.5)          71.4%    89.3%    96.4%   0.821
hybrid(bm25=1.0)          75.0%    89.3%    92.9%   0.829

Recall@3 by question type   conceptual    identifier
vector                           94.4%        100.0%
bm25                             83.3%         60.0%
hybrid(bm25>0)                   88.9%         90.0%
```

**Headline: 96.4% top-3 retrieval accuracy, MRR 0.852.**

### The result that surprised me

I built hybrid retrieval on the standard premise: *embeddings are weak on exact
identifiers, so BM25 is needed to catch them.* **The measurement falsified that
here.** Vector-only beat BM25 even on identifier lookups (100% vs 60%), and
fusing BM25 in at any weight made things strictly worse.

I chased it down in two steps:

1. **Suspected the fusion weighting.** Added weighted RRF and swept BM25 weights
   0 → 1. Every non-zero weight underperformed. Not the cause.
2. **Found a real bug in my tokenizer.** It split on underscores *before*
   recording the whole word, so `resolve_command` never existed as a token —
   destroying BM25's exact-match advantage on snake_case code. Fixed it to emit
   the full identifier plus its parts. BM25 *still* didn't improve.

The actual explanation: **BM25 finds occurrences, not definitions.** A test file
mentioning `resolve_command` five times outranks the one chunk that defines it.
Meanwhile the AST chunker already puts the symbol name and `class X:` signature
into every chunk, so the embedding matches the definition directly. **The
chunking work made BM25 redundant** — two components solving the same problem.

So the shipped default is `RRF_WEIGHT_BM25=0.0` (vector-only), and `hybrid_search`
skips the BM25 pass entirely at weight 0 to save the per-query index build. The
hybrid path stays in the codebase behind config, because with a weaker embedder
or a codebase with unusual identifiers the trade-off could flip — and now there's
a harness to re-check that in one command.

> `selftest.py` additionally verifies the full stack end-to-end (chunking,
> pgvector + real embeddings, retrieval, grounded Ollama answer).

## Learning from feedback

Every answer is rated (1-5 stars) and logged. That signal is used two ways:

**1. In-context learning (live, no training).** When a new question comes in, the
most similar *highly-rated* past answer is retrieved (by question embedding) and
injected into the prompt as a few-shot exemplar. The assistant imitates answers
users liked, immediately -- a real feedback loop that reuses the embedding +
pgvector infra.

**2. Fine-tuning dataset export.** `export_training_data.py` dumps the feedback
log as `sft.jsonl` (highly-rated question->answer pairs in chat format, ready for
SFT via a provider API or local LoRA with unsloth/axolotl) and `preferences.jsonl`
(all rated interactions). True DPO pairs need two answers to the same prompt, so
those are collected by re-asking + re-rating.

Design note: in a RAG system most quality comes from *retrieval*, so low-rated
answers are also mined to grow the eval set and tune retrieval -- often a bigger
win than touching model weights.

## Efficiency: answer caching

Re-running retrieval + the LLM for a question that was already answered wastes
time and (for hosted models) money. So answers are cached per repo:

- **Exact + semantic cache.** On each question we check `answer_cache` for the
  same repo: an exact question match, or a stored question whose embedding is
  >= 0.93 cosine-similar. A hit returns the stored answer instantly and **skips
  retrieval and the LLM entirely**.
- **Invalidation.** Re-indexing a repo (its code changed) drops that repo's
  cached answers, so cached results never go stale.
- The analytics dashboard shows the **cache hit rate**, and the app labels
  cached answers as "served from cache."

## What I'd change at 10x scale

Already implemented beyond the core spec: incremental indexing (per-file
hashing), token streaming, a cross-encoder re-ranker, auth + accounts, an answer
cache, and a feedback-driven few-shot loop.

Still ahead at real scale:
- A durable job queue (not an in-process thread) so many repos index concurrently.
- Per-tenant isolation + rate limiting.
- JS symbol coverage is structurally lower than Python's (~8% vs ~58% of chunks
  named): CommonJS/arrow assignments now resolve, but much JS top-level code is
  `require` blocks and `describe()` wrappers that aren't definitions at all.
- True DPO preference pairs (two answers per prompt) for fine-tuning.
- A live public URL: container, prod config and guide are done (DEPLOYMENT.md);
  it only needs an account + a few dollars of API credit.

## Deployment

The local build is fully offline (bge-base + Ollama). The **deployed** build swaps
both for hosted APIs via env vars -- no code changes, because the backends are
pluggable. PyTorch and Ollama don't fit free cloud tiers, so the production image
deliberately omits them (`requirements-prod.txt`).

```bash
# whole stack in containers, locally:
export OPENAI_API_KEY=sk-...
docker compose -f docker-compose.prod.yml up --build   # -> http://localhost:8000
```

Public deploy (managed Postgres w/ pgvector + a container host), env vars, index
seeding and cost controls are documented step-by-step in **[DEPLOYMENT.md](DEPLOYMENT.md)**.

> Gotcha: the two embedding backends produce different vector sizes (768 vs
> 1536), so a database indexed with one cannot be queried with the other. The
> deployed DB is a separate index built with hosted embeddings.

## Run it locally

Prereqs: Docker Desktop, Python 3.10+, and (for local mode) Ollama.

```bash
# 1. Database
docker compose up -d

# 2. Python deps
python -m venv .venv && .venv\Scripts\activate      # Windows
pip install -r requirements.txt

# 3. Config
copy .env.example .env      # local bge + Ollama by default

# 4. (local LLM) pull a model
ollama pull llama3.2

# 5. Index a repo
python build_index.py https://github.com/pallets/click

# 6a. CLI
python search.py "how are options parsed?" --repo https://github.com/pallets/click
python answer.py "how are aliases resolved for a command?" --repo https://github.com/pallets/click

# 6b. Web app
uvicorn app:app --port 8000      # open http://localhost:8000

# 7. Evaluate retrieval
python evaluate.py

# Health check (all of the above at once)
python selftest.py --repo https://github.com/pallets/click

# Run the test suite (no torch/Ollama needed — stubs the model)
pip install -r requirements-dev.txt
pytest
```

## Configuration (.env)

| Var | Default | Notes |
|-----|---------|-------|
| `EMBEDDING_BACKEND` | `local` | `local` (bge-base, 768d) or `openai` (1536d) |
| `LLM_BACKEND` | `ollama` | `ollama`, `anthropic`, or `openai` |
| `OLLAMA_MODEL` | `llama3.1` | e.g. `llama3.2` for faster CPU inference |
| `DATABASE_URL` | `postgresql://rag:rag@localhost:5432/rag` | matches docker-compose |
| `RERANK` | `false` | set `true` to add a cross-encoder re-ranker (sharper, slower) |

> Switching `EMBEDDING_BACKEND` changes the vector dimension, so reset the DB
> volume (`docker compose down -v && docker compose up -d`) and re-index.

## Project layout

```
ingest/repo.py       clone + walk/filter files
ingest/chunker.py    AST-aware chunking
embed/embeddings.py  local/OpenAI embeddings
store/db.py          pgvector store + search
retrieve.py          BM25 + RRF hybrid retrieval
answer.py            grounded, cited answer
llm.py               pluggable LLM client
app.py + web/        FastAPI backend + chat UI
build_index.py       index a repo (CLI)
search.py            retrieval CLI
evaluate.py          retrieval accuracy metrics
selftest.py          end-to-end health check
```
