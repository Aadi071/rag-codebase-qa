# RAG Codebase Q&A — Prep Reading Checklist

Come here each day, pick **one** unchecked session, watch/read it, then tick the
box and jot one line in the Progress Log at the bottom. One session per day is
enough — the tree-sitter days are the heavy ones, so don't rush them.

To check a box: change `[ ]` to `[x]`.

---

## Session 1 — Get the RAG mental model (~½ day)
- [ ] Watch [RAG in 10 minutes (beginner-friendly)](https://www.youtube.com/watch?v=gweRh5Xtkq0)
- [ ] Watch [RAG Explained in 20 Minutes + hands-on project](https://www.youtube.com/watch?v=RosLeHGBLoY)

**Get this out of it:** RAG = "search first, then let the LLM answer using what
you found." Everything else is just making that search good.

---

## Session 2 — Embeddings + vector search + pgvector (~1 day)
- [ ] Read [pgvector Tutorial (DataCamp)](https://www.datacamp.com/tutorial/pgvector-tutorial)
- [ ] Skim [pgvector README (official)](https://github.com/pgvector/pgvector) — focus on the `<->` / `<=>` operators and the HNSW index section
- [ ] Read ["You probably don't need a vector database" (Encore)](https://encore.dev/blog/you-probably-dont-need-a-vector-database)

**Get this out of it:** an embedding is ~1,500 numbers capturing meaning; similar
text → nearby vectors. pgvector stores them and finds the nearest ones. Know
*why* pgvector beats Chroma/Qdrant here — that's an interview answer.

---

## Session 3 — Code chunking with tree-sitter / AST, part 1 (~1 day)
- [ ] Read [Semantic Code Indexing with AST and Tree-sitter](https://medium.com/@email2dineshkuppan/semantic-code-indexing-with-ast-and-tree-sitter-for-ai-agents-part-1-of-3-eb5237ba687a)

**Get this out of it:** how code becomes an AST, and how you walk that tree to
pull out functions and classes as whole units.

---

## Session 4 — Code chunking with tree-sitter / AST, part 2 (~1 day)
- [ ] Read [code-chunk: AST-aware chunking, explained (Supermemory)](https://supermemory.ai/blog/building-code-chunk-ast-aware-code-chunking/)
- [ ] Browse the [code-chunk repo](https://github.com/supermemoryai/code-chunk) — read the actual chunking code
- [ ] (Optional/advanced) Skim the [cAST paper](https://arxiv.org/html/2506.15655v1) — recursively splitting the AST so functions stay whole; great to cite in your README

**Get this out of it:** split on function/class boundaries, not line counts, and
carry metadata (file, line range, symbol name) with every chunk — that metadata
becomes your `auth.py:42` citation.

---

## Session 5 — Hybrid search: BM25 + vectors + RRF (~½ day)
- [ ] Read [Why Vector Search Alone Isn't Enough (InfoQ)](https://www.infoq.com/articles/vector-search-hybrid-retrieval-rag/)
- [ ] Read [Hybrid Search + Reranking Playbook (OptyxStack)](https://optyxstack.com/rag-reliability/hybrid-search-reranking-playbook)

**Get this out of it:** embeddings miss exact identifiers like `getUserByIdV2`;
BM25 keyword search catches them; RRF merges the two ranked lists without needing
their scores on the same scale. This is your single best interview story.

---

## Session 6 — Grounded prompting + citations (~½ day)
- [ ] Read [Anthropic prompt engineering docs](https://docs.claude.com/en/docs/build-with-claude/prompt-engineering/overview)
- [ ] Skim [OpenAI embeddings guide](https://platform.openai.com/docs/guides/embeddings) (if going the OpenAI route for embeddings)

**Get this out of it:** how to structure a prompt that answers *only* from
retrieved code, cites file:line, and says "I don't find this" instead of
hallucinating.

---

## Session 7 — Evaluation, the differentiator (~½ day)
- [ ] Reread the evaluation sections of the [Hybrid Search Playbook](https://optyxstack.com/rag-reliability/hybrid-search-reranking-playbook) — Recall@k, Precision@k, MRR

**Get this out of it:** how you get the `[X]% top-3 accuracy` number for your
resume bullet, by scoring retrieval against a 15–20 question test set.

---

## Session 8 — FastAPI backend (~½ day)
- [ ] Read [FastAPI official tutorial](https://fastapi.tiangolo.com/tutorial/) — "First Steps" through "Request Body"

**Get this out of it:** enough to build the `/index` and `/ask` endpoints.

---

## When all boxes are ticked
You're ready to build. Come back and we'll scaffold the project and start
Milestone 1 (repo cloning + AST chunking).

---

## Progress Log
_One line per session: date + what stuck._

-
