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

**2. Hybrid retrieval, because embeddings are weak on exact identifiers.** Dense
embeddings capture meaning but miss exact symbols like `resolve_command`. BM25
catches those. We run both and fuse with RRF, which combines by **rank** (not raw
score), so cosine-similarity and BM25 scores don't need to be normalized against
each other. The eval below quantifies the lift.

**3. pgvector over a dedicated vector DB.** One fewer moving part: the chunk text,
its metadata, and its vector live in the same Postgres row and transaction. No
separate service to run or keep in sync.

**4. Local-first, but pluggable.** Embedding and LLM backends are chosen via
`.env` (`EMBEDDING_BACKEND`, `LLM_BACKEND`), so "swap the model" is a one-line
change. Default is zero-cost local (bge + Ollama).

## Evaluation

`evaluate.py` scores retrieval on an 18-question hand-labeled set
(`eval/eval_set.json`), comparing vector-only, BM25-only, and hybrid:

```
mode        Recall@1  Recall@3  Recall@5
----------------------------------------
vector        [ ... fill in from `python evaluate.py` ... ]
bm25
hybrid
Hybrid MRR: [ ... ]
```

Headline: **hybrid top-3 retrieval accuracy = [X]%.** The vector-vs-hybrid gap is
the evidence that hybrid retrieval was worth building.

> `selftest.py` additionally verifies the full stack end-to-end (chunking,
> pgvector + real embeddings, hybrid retrieval, grounded Ollama answer).

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
- Better JS symbol extraction (CommonJS / assigned function expressions).
- True DPO preference pairs (two answers per prompt) for fine-tuning.
- A live cloud deployment (needs hosted models -- bge + Ollama don't fit free tiers).

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
