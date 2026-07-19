# RAG-Powered Codebase Q&A Assistant — Progress Notes

Running log for Claude (and Anty) to pick this project back up in a future
session without losing context. If you're Claude reading this at the start of a
new session: read this whole file first, then READING.md, before doing anything.

## Project

Source spec: PROJECT 03 from the "10 Software Engineering Projects That Get You
Interviews" PDF Anty uploaded (RAG-Powered Codebase Q&A Assistant — AI/LLM
Engineering, Advanced, 7–10 days).

Goal: paste a public GitHub repo URL → app clones it, AST-chunks the code,
embeds + stores chunks, and answers natural-language questions ("where is auth
handled?") with accurate, cited answers grounded in real code (`auth.py:42`
style). The differentiator vs. a toy: hybrid retrieval (vector + BM25) and a
measured retrieval-accuracy eval.

## Current status: PREP PHASE (not building yet)

Anty is doing self-study reading BEFORE we write code. Plan:
- Anty picks up **one reading session per day** from `READING.md`, ticks the
  box, logs one line in that file's Progress Log.
- Starting date: **the day after 2026-07-13** (i.e., ~2026-07-14 onward).
- A daily reminder is scheduled to nudge them to pick up a session.
- When all 8 sessions in READING.md are checked, we scaffold the project and
  start Milestone 1.

**Nothing is built yet.** Only `READING.md` and this file exist in the folder.

## How Anty wants to work (carried over from realtime-doc-editor)

- Learning-first, hands-on, beginner level — explain concepts before/while
  writing code. Anty explicitly wants to understand nuances deeply, not just
  ship.
- Step-by-step, one milestone at a time, interactive.
- Direct file edits (not chat-displayed code to hand-type).
- Concise, direct communication preference (remove-words-and-same-meaning test).
- Wants reference reading suggested when new concepts show up.
- Update THIS file at the end of any meaningful session.

## Environment / project location

- Working folder: `Desktop/projects/rag-codebase-qa/` (cowork-connected folder
  on Anty's real Windows machine — NOT the sandbox).
- Anty just finished `realtime-doc-editor/` (Node/React/TS/WebSockets/CRDT/
  Postgres/Redis/Docker/JWT — deployed). Full-stack + Postgres + Docker muscle
  carries over.
- **Open decision:** build language/stack not yet chosen. Spec suggests
  Python/FastAPI. Anty is more comfortable in Node/TypeScript after last
  project. Decide before writing any code — it shapes everything. (pgvector =
  Postgres extension, already familiar with Docker Postgres.)

## Tech stack (from spec, to confirm)

- Ingestion: Python + tree-sitter (AST-aware chunking by function/class), OR a
  Node equivalent if we go Node.
- Embeddings: OpenAI `text-embedding-3-small` or open-source `bge-base`.
- Vector store: **pgvector** (better resume signal than Chroma/Qdrant — one
  fewer moving part).
- Retrieval: hybrid — vector similarity + BM25 keyword, merged via Reciprocal
  Rank Fusion (RRF).
- LLM: Anthropic Claude or OpenAI, grounded prompting with source citations.
- Backend: FastAPI (or Node). Frontend: simple React or plain HTML chat.
- Deploy: backend on Railway/Render, frontend on Vercel; index any public repo
  on demand (async/background indexing with status polling).

## The hard parts (the nuances Anty wants to master) — reference

1. AST chunking (tree-sitter): split on function/class boundaries, carry
   metadata (file, line range, symbol name) = the citation.
2. Embeddings are weak on exact code identifiers → why hybrid/BM25 matters.
3. Context-window budgeting (top-k tuning, dedup, ordering).
4. Grounded prompting (cite file:line, say "I don't know", no hallucination).
5. Evaluation: 15–20 hand-labeled Q→expected-chunk set; Recall@k, Precision@k,
   MRR. This is the resume-bullet `[X]% top-3 accuracy` number.
6. Operational: repo clone + file filtering (skip node_modules/binaries/lock
   files), async indexing with progress status.

## Milestones (from spec, adapt to learning-first style)

1. Day 1–2: repo clone + file walk/filter + AST chunking; verify chunk quality by eye.
2. Day 3–4: pgvector setup, embed + store chunks, basic vector similarity search.
3. Day 5: BM25 hybrid retrieval + RRF re-ranking; grounded prompt template with citations.
4. Day 6: FastAPI (or Node) endpoints + simple chat frontend; wire end-to-end.
5. Day 7–8: 15–20 question eval set, measure retrieval accuracy, deploy, write README.

## Reading list already given (see READING.md for full links)

8 sessions: (1) RAG mental model, (2) embeddings + pgvector, (3) tree-sitter AST
part 1, (4) tree-sitter AST part 2 + cAST paper, (5) hybrid search BM25+RRF,
(6) grounded prompting + citations, (7) evaluation metrics, (8) FastAPI.

## Resume instructions for next session

If Anty comes back mid-prep: ask which READING.md sessions they've ticked, answer
any concept questions, don't start building until the reading is done (unless
Anty asks). When reading is done: confirm the stack decision (Python vs Node),
then scaffold and start Milestone 1.

---

## Session 2026-07-14 — Milestone 1 COMPLETE (repo clone + AST chunking)

Decision locked: **Python + FastAPI** (Anty chose it over Node).

Built the ingestion pipeline (all verified in sandbox against pallets/click):
- `requirements.txt` — tree-sitter + per-language wheels (python/javascript/
  typescript). NOTE: we deliberately use the per-language wheels, NOT
  tree-sitter-language-pack, because the language-pack downloads grammars from
  GitHub at runtime (fragile / fails offline). The wheels ship the grammar
  compiled in.
- `ingest/repo.py` — `clone_repo(url)` shallow-clones; `iter_source_files(root)`
  walks + filters (skips .git/node_modules/build dirs, >1MB files, non-code
  extensions), returns SourceFile(path, rel_path, language).
- `ingest/chunker.py` — AST chunking. Key design (this was the real learning):
  greedy "divide and combine" over AST nodes; **recurse into classes to split
  them into methods, but keep functions WHOLE even when oversized** — never
  split a function's signature from its body. Decorators stay attached to their
  function. Every Chunk carries rel_path + start/end line + symbol name (the
  citation metadata).
- `index.py` — CLI driver: clone -> walk -> chunk -> print sample chunks.

Bug caught + fixed during verification: first version recursed into oversized
functions and produced 68 signature-only broken chunks + detached decorators.
Fixed with the descend-only-into-containers rule. Result on click: 76 files ->
686 chunks, 0 broken, classes split into named methods, decorated funcs intact.

GOTCHA for future sessions: writing code files to the Windows-mounted folder via
the file-write tool truncated chunker.py twice (silently). Writing via sandbox
bash heredoc + verifying with py_compile was reliable. Always py_compile after
writing Python here.

Anty has NOT run it on their own machine yet — next time, confirm it ran, then
start Milestone 2 (pgvector + embeddings).

### Next: Milestone 2 — pgvector + embeddings
Set up Postgres+pgvector (Docker, like last project), embed each chunk, store
vector + metadata, build basic vector similarity search.

### Milestone 1 polish (same session, 2026-07-14) — class-context chunking

Fixed the orphaned-class-signature issue Anty spotted in their own run
(aliases.py: a bare `class AliasedGroup(click.Group):` chunk + a nameless body
chunk). New behavior in chunker.py:
- Added `_is_class()` and `_split_class()`. When a class is too big to keep
  whole, we split it into its members and PREPEND the class signature
  (`class X(...):`) to each member chunk, and name it `X.method`. The line range
  still points at the member's real lines.
- Small classes still stay whole. Functions still kept whole. Decorators intact.
Verified on click (LF): 0 orphan class-signature chunks, 0 broken function
chunks, 77 members carrying `ClassName.method` symbols + class-signature prefix
(e.g. UsageError.show, HelpFormatter.write_usage). NOTE: on Windows (CRLF) files
are larger in bytes, so classes cross the 1500-char budget and split more often
than on LF — now those splits are clean.

Milestone 1 is fully done. Next session: Milestone 2 (pgvector + embeddings).

---

## Session 2026-07-14 (cont.) — Milestone 2 CODE-COMPLETE (embeddings + pgvector)

Embedding model chosen: **OpenAI text-embedding-3-small** (1536 dims) over local
bge-base, to avoid a heavy PyTorch install on Windows; costs ~1 cent to index click.

New files:
- `docker-compose.yml` — pgvector/pgvector:pg16, db `rag`/user `rag`/pw `rag` on
  localhost:5432; mounts `db/init` so schema auto-runs on a fresh volume (same
  down -v-wipes-data gotcha as last project).
- `db/init/001_schema.sql` — `chunks` table (repo, rel_path, language, symbol,
  start_line, end_line, content, embedding vector(1536)) + HNSW cosine index
  (`vector_cosine_ops`) + repo btree index.
- `config.py` — loads .env (DATABASE_URL, OPENAI_API_KEY, EMBEDDING_MODEL,
  EMBEDDING_DIM=1536).
- `store/db.py` — connect() (ensures `vector` ext then register_vector),
  init_schema, clear_repo (idempotent re-index), insert_chunks (executemany),
  search (top-k, `<=>` cosine distance, score = 1 - distance). Query vector MUST
  be wrapped in pgvector `Vector` or Postgres errors `vector <=> double
  precision[]`.
- `embed/embeddings.py` — embed_texts(): batched OpenAI embeddings + retry.
- `build_index.py` — clone -> chunk -> embed -> store (idempotent per repo).
- `search.py` — embed a question, print top-k chunks + citations.
- `requirements.txt` — added openai, psycopg[binary], pgvector, python-dotenv.

VERIFIED in sandbox with `pgserver` (bundled postgres+pgvector, no Docker/root):
DB layer nearest-neighbour ordering + repo filter + clear all correct; full
clone->chunk->embed->store->search wired end-to-end on click (655 chunks) using a
stub embedder. Real semantic quality pending Anty's run with an OpenAI key.

Anty still needs to (on their machine): `docker compose up -d`, copy .env.example
-> .env and add OPENAI_API_KEY, `pip install -r requirements.txt`, then
`python build_index.py <url>` and `python search.py "..." --repo <url>`.

### Next: Milestone 3 — hybrid retrieval (BM25 + vector via RRF) + grounded prompting

### Milestone 2 addendum (2026-07-14) — switched to LOCAL embeddings (bge-base)

Anty's OpenAI account hit insufficient_quota (429) — key valid but no billing
credit. Anty chose free local embeddings over adding credit. Changes:
- Embedding backend now switchable via `.env` EMBEDDING_BACKEND = local | openai.
- local = BAAI/bge-base-en-v1.5 via sentence-transformers, 768 dims (default).
  openai = text-embedding-3-small, 1536 dims (kept as an option; key still in .env).
- `config.py` sets EMBEDDING_DIM from the backend (768 vs 1536).
- `embed/embeddings.py`: embed_texts() for documents, embed_query() for queries
  (bge wants the "Represent this sentence..." instruction prepended to queries).
  Local model lazy-loaded + cached; first run downloads ~400MB.
- `db/init/001_schema.sql` now ONLY creates the extension; the app's init_schema()
  creates the chunks table so vector(dim) always matches the backend.
- Rewrote Anty's `.env` to EMBEDDING_BACKEND=local (kept their OPENAI key for later).
- requirements.txt: added sentence-transformers (pulls torch).
Verified full pipeline at 768 dims in sandbox (pgserver + stub embedder): 655
chunks stored, vector_dims=768, search returns top-k.

ACTION FOR ANTY (dim changed 1536->768, so the old table must go):
  docker compose down -v && docker compose up -d
  pip install -r requirements.txt      # installs sentence-transformers + torch (big, one-time)
  python build_index.py https://github.com/pallets/click   # first run downloads bge-base
  python search.py "where is authentication handled?" --repo https://github.com/pallets/click

---

## Session 2026-07-14 (cont.) — Milestone 3a DONE (hybrid retrieval: BM25 + vector via RRF)

New:
- `retrieve.py` — code-aware tokenize() (splits snake_case + camelCase, keeps
  whole identifier), bm25_rank() (rank-bm25 BM25Okapi over a repo's chunks),
  rrf_fuse() (Reciprocal Rank Fusion, k=60, fuse by rank not raw score),
  hybrid_search() (vector top-N + BM25 top-N -> RRF -> top-k; results carry rrf
  score + vec_rank + bm25_rank).
- `store/db.py` — added fetch_chunks() and vector_search() returning dict rows
  WITH the chunk id (needed to fuse the two rankings by id).
- `search.py` — now runs hybrid_search (was pure vector). Prints rrf + per-source
  ranks so the fusion is visible.
- requirements: rank-bm25.

VERIFIED in sandbox (pgserver + stub embedder + real click, 660 chunks):
tokenizer splits identifiers correctly; BM25 surfaces exact-identifier chunks
(resolve_command); hybrid fuses well — e.g. a chunk at vec#28 but bm25#2 gets
pulled into the top-5 by RRF (the exact hybrid win). ANTY still needs to
`pip install rank-bm25` (or -r requirements.txt) then re-run search.py (no
re-index needed; retrieval only).

### Next: Milestone 3b — grounded prompt + cited LLM answer
Blocker/decision: needs an LLM. Anty has NO OpenAI credit. Options: Anthropic API
(needs key+credit), local LLM via Ollama (free, ~4GB model download), or defer.
Retrieval already returns the right chunks; 3b is the last mile (synthesize a
cited answer + "I don't find this in the code" grounding).

## Session 2026-07-14 (cont.) — Milestone 3b DONE (grounded, cited answers via Ollama)

Answer LLM: LOCAL Ollama (Anty chose it; consistent with local embeddings, free).
New:
- `config.py` — LLM_BACKEND (ollama|anthropic|openai, default ollama), OLLAMA_HOST,
  OLLAMA_MODEL (default llama3.1), ANTHROPIC_* placeholders.
- `llm.py` — complete(system,user) dispatch. Ollama via stdlib urllib (no new dep),
  friendly error if server/model missing. anthropic/openai lazy-imported.
- `answer.py` — question -> hybrid_search -> grounded prompt -> llm -> cited answer
  + printed sources. SYSTEM_PROMPT forces: answer ONLY from context, cite
  (path:start-end), else say exactly "I couldn't find this in the indexed code",
  no outside knowledge.
- `.env.example` — LLM settings added.
VERIFIED in sandbox (stub LLM + real click): prompt carries rules + context
headers (path:lines + symbol, incl. class-signature prefix from M1), answer +
sources printed. Real Ollama call runs on Anty's machine.

ANTY TO DO (local, one-time): install Ollama (ollama.com), `ollama pull llama3.1`
(or llama3.2 for speed / qwen2.5-coder:7b for code quality; set OLLAMA_MODEL in
.env), ensure Ollama running, `pip install -r requirements.txt`, then:
  python answer.py "how are aliases resolved for a command?" --repo https://github.com/pallets/click

MILESTONE 3 COMPLETE (3a hybrid retrieval + 3b grounded answers).

### Next: Milestone 4 — FastAPI backend + simple chat frontend (async indexing + status)
### Still pending housekeeping: update tracker (M2+M3 done, check topics); retire
### redundant rag-daily-reading-nudge (duplicates the 8:35pm daily-grind).

## Session 2026-07-14 (cont.) — added selftest.py (one-command health check)

Anty asked to "run the codes." Constraints: I can't type into their real
terminal (computer-use terminal tier blocks keystrokes), and my sandbox can't
run real bge (HuggingFace blocked 403) or Ollama or their Docker pg. So instead:
- Re-verified everything runnable in sandbox: M1 real (667 chunks, 81 qualified
  members, 0 broken/orphan) + the earlier 15/15 integration checks.
- Wrote `selftest.py`: one command that runs the REAL stack on Anty's machine and
  prints PASS/FAIL per milestone (M1 chunking, M2 pgvector+real bge, M3a hybrid,
  M3b Ollama answer), with actionable hints on each failure. Anty runs:
  `python selftest.py --repo https://github.com/pallets/click` (expect 5/5).
Still awaiting Anty's local run of selftest.py / answer.py to confirm bge+Ollama.

## Session 2026-07-15 — FULL STACK VERIFIED 5/5 ON ANTY'S MACHINE

selftest.py -> 5/5 PASS: M1 chunking (676 chunks), M2 pgvector + real bge (665
indexed), M3a hybrid (top = aliases.py AliasedGroup), M3b grounded Ollama answer
with real line-level citations (aliases.py:132-143 / 76-129). Fully local, no API.

Ollama setup notes (did this via computer-use / guidance):
- Ollama installed per-user at %LOCALAPPDATA%\Programs\Ollama\. Launching the
  Ollama app starts the server on :11434. GUI model catalog is a curated 2026
  list (gemma4:12b, qwen3.6, cloud models) with no llama and no sizes; GUI pull
  flow was flaky, so pulled via CLI instead.
- `ollama` not on PATH in an already-open terminal -> use full path
  `& "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" pull <model>` or reopen shell.
- Set .env OLLAMA_MODEL=llama3.2 (3B, fast on CPU). Pulled llama3.2. Works.

MILESTONES 1-3 COMPLETE AND VERIFIED ON HARDWARE.
### Next: Milestone 4 (FastAPI /index + /ask + chat UI, async indexing) then M5.
### Housekeeping still pending: sync tracker (M2+M3 done, check topics); retire
### redundant rag-daily-reading-nudge.

## Session 2026-07-15 (cont.) — Milestone 4 DONE (FastAPI backend + chat UI)

Housekeeping first: synced tracker (Project 3 M1-M3 done + topics checked via a
3rd guarded migration p3_m3_v1); DISABLED rag-daily-reading-nudge (redundant with
the 8:35pm daily-grind; reversible).

New:
- `app.py` — FastAPI. GET / serves the chat page. POST /api/index starts a
  BACKGROUND THREAD (clone->chunk->embed->store) so the request doesn't block;
  GET /api/status?repo= reports stage (cloning/chunking/embedding/storing/ready/
  error/unindexed) and falls back to a DB count if not tracked in memory (e.g.
  indexed via CLI). POST /api/ask reuses answer.answer() -> {answer, sources}.
- `web/index.html` — single-page vanilla-JS UI: index a repo (polls status every
  1.5s), ask questions, renders the grounded answer + a sources list. No build step.
- requirements: fastapi, uvicorn enabled.
VERIFIED in sandbox (TestClient + pgserver + stub embed/llm): GET / 200, index ->
ready (667 chunks), /api/ask returns answer + 6 sources. Async indexing works.

ANTY TO RUN: `pip install -r requirements.txt` then `uvicorn app:app --port 8000`,
open http://localhost:8000 (Docker pg + Ollama already running; click already
indexed so page shows Ready). First ask is slow on CPU (LLM).

MILESTONES 1-4 COMPLETE. Next: Milestone 5 — eval set (Recall@k/MRR), deploy, README.

## Session 2026-07-15 (cont.) — Milestone 5a DONE (evaluation harness)

New:
- `eval/eval_set.json` — 18 hand-labeled questions for pallets/click, each mapped
  to the file(s) that contain the answer (ground truth verified against real click
  source: parser.py, termui.py, _termui_impl.py, core.py, testing.py, types.py,
  exceptions.py, decorators.py, formatting.py, shell_completion.py, _winconsole.py,
  examples/aliases/aliases.py).
- `evaluate.py` — for each question runs vector-only, bm25-only, and hybrid(RRF)
  retrieval; reports Recall@1/3/5 per mode + hybrid MRR + a headline
  "hybrid top-3 retrieval accuracy = X%". File-level ground truth; rel_path
  normalized so Windows backslashes match. Metric math unit-checked.
VERIFIED runs in sandbox (stub vector, real BM25): harness executes, table prints.
NOTE: stub vector drags hybrid below bm25 in sandbox (61% vs 83% R@3) — on Anty's
real bge the vector + hybrid rows lift and hybrid should top both.

ANTY TO RUN (Docker up, click indexed): `python evaluate.py` -> real numbers.
The hybrid Recall@3 is the resume-bullet accuracy figure.

MILESTONES 1-4 + 5a done. Remaining M5: README (with the eval number) + deploy.
### Deploy caveat: bge (torch/RAM) + Ollama don't run on free cloud tiers; need a
### decision (hosted LLM for prod / retrieval-only demo / demo video).

## Session 2026-07-15 (cont.) — Milestone 5b DONE (README)

Wrote `README.md`: problem statement, mermaid architecture diagram, tech-stack
table, the 4 hardest decisions (AST/class-context chunking, hybrid+RRF, pgvector,
pluggable local-first backends), evaluation section (table has [X]% placeholders
for Anty to paste real `python evaluate.py` numbers), "what I'd change at 10x",
full run instructions (CLI + web + eval + selftest), .env config table, project
layout. GitHub renders the mermaid diagram.

TODO for Anty: run `python evaluate.py`, paste the 3-row table + MRR into the
README's Evaluation section (replace the [X]% placeholders).

Remaining: Milestone 5c DEPLOY (decision pending — local bge+Ollama don't run on
free cloud tiers). Also: record a 60-90s demo video (spec strongly recommends).

## Session 2026-07-15 (cont.) — Generalization test on other repos (+ TS/JS fix)

Anty asked to skip deploy for now and test if it works on OTHER repos. Ran the
ingestion/chunker (real, in sandbox) across diverse repos:
- flask (Python): 494 chunks, 281 named, 92 ClassName.method. Excellent.
- express (JS): 276 chunks. JS naming weak (CommonJS module.exports / assigned
  function expressions have no name field) -> only ~11 named. Chunks still fine;
  file:line citations always present, symbol label sometimes empty. KNOWN LIMIT.
- ky (TypeScript): found+fixed a real bug -> now 399 chunks, 24 named incl.
  Ky.create + split class methods.
- gin (Go): 0 chunks (no Go grammar shipped) -> expected; adding a language is
  install the tree-sitter wheel + 2 lines (EXT_TO_LANG + _LANGUAGES).

FIX in ingest/chunker.py: TS/JS wrap `export function`/`export class` in an
`export_statement` node. _definition_node and _is_class now UNWRAP
export_statement (and recurse), so exported defs get named and exported classes
split into methods. Python (click) regression: unchanged (667 chunks, 0 broken/
orphan, 81 qualified).

VERDICT: pipeline generalizes to any Python/JS/TS repo. TS good, Python great, JS
ok (weaker symbol labels for CommonJS). Other languages need a grammar added.

ANTY TO TRY on own machine: index another repo via the web app or
`python build_index.py <url>` then ask, e.g. a TS repo (sindresorhus/ky) or
another Python repo (pallets/flask). Deploy (M5c) intentionally deferred.
