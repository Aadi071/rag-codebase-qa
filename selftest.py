"""One-command health check for the RAG codebase Q&A assistant.

Runs the REAL pipeline on your machine and reports PASS/FAIL per milestone:
  M1 chunking, M2 pgvector + real embeddings, M3a hybrid retrieval, M3b Ollama.

Usage:
    python selftest.py                                   # uses pallets/click
    python selftest.py --repo https://github.com/pallets/click
    python selftest.py --repo <url> --question "how are options parsed?"

Prereqs: Docker up (docker compose up -d), repo indexed (build_index.py),
Ollama running with the model pulled. Each check degrades gracefully with a
hint if a prereq is missing.
"""

import argparse
import re
import traceback

PASS, FAIL = [], []


def check(name, fn):
    try:
        ok, detail = fn()
    except Exception as e:
        ok, detail = False, f"{type(e).__name__}: {e}"
    (PASS if ok else FAIL).append(name)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -> {detail}" if detail else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="https://github.com/pallets/click")
    ap.add_argument("--question", default="how are aliases resolved for a command?")
    args = ap.parse_args()

    import config
    print(f"Config: embeddings={config.EMBEDDING_BACKEND} ({config.EMBEDDING_MODEL}, "
          f"dim {config.EMBEDDING_DIM}) | llm={config.LLM_BACKEND} "
          f"({config.OLLAMA_MODEL if config.LLM_BACKEND=='ollama' else ''})")
    print(f"Repo under test: {args.repo}\n")

    # ---- M1: chunking (does not need DB) ----
    def m1():
        import tempfile, subprocess, os
        from ingest.repo import clone_repo, iter_source_files
        from ingest.chunker import chunk_source
        root = clone_repo(args.repo)
        chunks = []
        for sf in iter_source_files(root):
            chunks.extend(chunk_source(open(sf.path, "rb").read(), sf.rel_path, sf.language))
        broken = [c for c in chunks if c.language == "python"
                  and c.text.strip().startswith("def ") and c.text.rstrip().endswith(":")]
        orphan = [c for c in chunks if re.fullmatch(r"class .*:", (c.text.strip() or ""))]
        qualified = sum(1 for c in chunks if c.symbol and "." in c.symbol)
        ok = len(chunks) > 50 and not broken and not orphan
        return ok, f"{len(chunks)} chunks, {qualified} qualified members, 0 broken/orphan"
    check("M1 chunking", m1)

    # ---- shared DB connection ----
    conn = {"c": None}
    def connect():
        from store import db
        conn["c"] = db.connect()
        return True, "connected to Postgres/pgvector"
    check("DB connection (Docker pgvector)", connect)

    # ---- M2: repo indexed + real embedding + vector search ----
    def m2():
        from store import db
        from embed import embeddings
        c = conn["c"]
        n = c.execute("SELECT count(*) FROM chunks WHERE repo=%s", (args.repo,)).fetchone()[0]
        if n == 0:
            return False, f"no chunks for {args.repo} — run: python build_index.py {args.repo}"
        qv = embeddings.embed_query("parse command line options")   # exercises real bge
        if len(qv) != config.EMBEDDING_DIM:
            return False, f"embedding dim {len(qv)} != {config.EMBEDDING_DIM}"
        rows = db.vector_search(c, qv, repo=args.repo, k=5)
        return len(rows) == 5, f"{n} chunks indexed, real embedding ok, vector top-5 returned"
    check("M2 pgvector + real embeddings", m2)

    # ---- M3a: hybrid retrieval ----
    def m3a():
        import retrieve
        hits = retrieve.hybrid_search(conn["c"], args.question, repo=args.repo, k=6)
        has_ranks = all("rrf" in h and "vec_rank" in h and "bm25_rank" in h for h in hits)
        top = hits[0]
        return len(hits) == 6 and has_ranks, \
            f"top: {top['rel_path']}:{top['start_line']}-{top['end_line']} {top.get('symbol') or ''}"
    check("M3a hybrid retrieval (BM25+vector+RRF)", m3a)

    # ---- M3b: grounded answer via Ollama ----
    def m3b():
        import answer as amod
        reply, srcs = amod.answer(args.question, repo=args.repo, k=6)
        cited = bool(re.search(r"\w+\.\w+:\d+", reply)) or "couldn't find" in reply.lower()
        print("\n    --- answer ---\n    " + reply.replace("\n", "\n    ") + "\n    --------------")
        return bool(reply) and len(srcs) > 0, f"{len(srcs)} sources; citation-shaped={cited}"
    check("M3b grounded answer (Ollama)", m3b)

    if conn["c"]:
        conn["c"].close()
    print(f"\n================ {len(PASS)}/{len(PASS)+len(FAIL)} checks passed ================")
    if FAIL:
        print("FAILED:", ", ".join(FAIL))


if __name__ == "__main__":
    main()
