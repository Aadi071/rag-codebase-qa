"""Milestone 5: measure retrieval accuracy on a hand-labeled question set.

For each question we know which file(s) contain the answer (eval/eval_set.json).
We run three retrieval strategies and check whether a correct file appears in the
top-k results:
  - vector-only  (dense embeddings)
  - bm25-only    (keyword)
  - hybrid       (vector + bm25 fused with RRF)  <- what the app uses

Metrics:
  Recall@k = fraction of questions with a correct file in the top-k.
  MRR      = mean of 1/(rank of first correct file).  (reported for hybrid)

The vector-only-vs-hybrid gap is the evidence that hybrid retrieval was worth it,
and Recall@3 for hybrid is the "top-3 retrieval accuracy" number for your resume.

Usage:  python evaluate.py            (uses eval/eval_set.json)
"""

import json
import os

from store import db
from embed import embeddings
import retrieve

KS = [1, 3, 5]
CANDIDATES = 20


def _norm(p):
    return p.replace("\\", "/")


def _first_hit_rank(ranked_paths, expected):
    for i, p in enumerate(ranked_paths, start=1):
        if _norm(p) in expected:
            return i
    return None


def main():
    spec = json.load(open(os.path.join("eval", "eval_set.json"), encoding="utf-8"))
    repo = spec["repo"]
    questions = spec["questions"]

    conn = db.connect()
    all_chunks = db.fetch_chunks(conn, repo=repo)
    if not all_chunks:
        print(f"No chunks for {repo}. Run: python build_index.py {repo}")
        return
    by_id = {c["id"]: c for c in all_chunks}

    # accumulators: mode -> {k -> hits}, plus reciprocal ranks for hybrid
    hits = {m: {k: 0 for k in KS} for m in ("vector", "bm25", "hybrid")}
    rr_hybrid = 0.0

    for item in questions:
        q, expected = item["q"], set(_norm(f) for f in item["files"])

        qv = embeddings.embed_query(q)
        vec_ids = [r["id"] for r in db.vector_search(conn, qv, repo=repo, k=CANDIDATES)]
        bm_ids = retrieve.bm25_rank(q, all_chunks)[:CANDIDATES]
        hyb_ids, _ = retrieve.rrf_fuse([vec_ids, bm_ids])

        ranked = {
            "vector": [by_id[i]["rel_path"] for i in vec_ids],
            "bm25": [by_id[i]["rel_path"] for i in bm_ids],
            "hybrid": [by_id[i]["rel_path"] for i in hyb_ids],
        }
        for mode, paths in ranked.items():
            rank = _first_hit_rank(paths, expected)
            for k in KS:
                if rank is not None and rank <= k:
                    hits[mode][k] += 1
        hr = _first_hit_rank(ranked["hybrid"], expected)
        rr_hybrid += (1.0 / hr) if hr else 0.0

    conn.close()
    n = len(questions)

    print(f"\nRetrieval accuracy on {n} hand-labeled questions ({repo})\n")
    header = "mode      " + "".join(f"  Recall@{k}" for k in KS)
    print(header)
    print("-" * len(header))
    for mode in ("vector", "bm25", "hybrid"):
        row = f"{mode:<9}" + "".join(f"   {hits[mode][k]/n*100:5.1f}%" for k in KS)
        print(row)
    print(f"\nHybrid MRR: {rr_hybrid/n:.3f}")
    print(f"\n==> Headline: hybrid top-3 retrieval accuracy = {hits['hybrid'][3]/n*100:.1f}%")


if __name__ == "__main__":
    main()
