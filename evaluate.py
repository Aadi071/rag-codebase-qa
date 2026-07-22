"""Measure retrieval accuracy on a hand-labeled question set.

Compares retrieval strategies on questions where we know which file holds the
answer (eval/eval_set.json). Questions are tagged by kind:
  - conceptual  ("How does Click parse options?")  -> embeddings usually win
  - identifier  ("Where is resolve_command defined?") -> BM25 usually wins
Hybrid should win overall by covering BOTH. Reporting them separately is the
only way to see whether hybrid is actually earning its keep.

Metrics: Recall@k (a correct file in the top-k) and MRR (1/rank of first hit).

Usage:
    python evaluate.py                # current configured fusion weights
    python evaluate.py --sweep        # try several BM25 weights in ONE pass
    python evaluate.py --sweep --rerank
"""

import argparse
import json
import os

import config
import retrieve
from embed import embeddings
from store import db

KS = [1, 3, 5]
CANDIDATES = 20
SWEEP_BM25_WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0]


def _norm(p):
    return p.replace("\\", "/")


def _first_hit_rank(ranked_paths, expected):
    for i, p in enumerate(ranked_paths, start=1):
        if _norm(p) in expected:
            return i
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true",
                    help="evaluate several BM25 fusion weights in one pass")
    ap.add_argument("--rerank", action="store_true",
                    help="also score a cross-encoder reranked row")
    args = ap.parse_args()

    spec = json.load(open(os.path.join("eval", "eval_set.json"), encoding="utf-8"))
    repo, questions = spec["repo"], spec["questions"]

    conn = db.connect()
    all_chunks = db.fetch_chunks(conn, repo=repo)
    if not all_chunks:
        print(f"No chunks for {repo}. Run: python build_index.py {repo}")
        return
    by_id = {c["id"]: c for c in all_chunks}

    weights = SWEEP_BM25_WEIGHTS if args.sweep else [config.RRF_WEIGHT_BM25]
    modes = ["vector", "bm25"] + [f"hybrid(bm25={w})" for w in weights]
    if args.rerank:
        modes.append("hybrid+rerank")

    kinds = sorted({q.get("kind", "conceptual") for q in questions})
    hits = {m: {k: 0 for k in KS} for m in modes}
    rr = {m: 0.0 for m in modes}
    kind_hits = {m: {kd: [0, 0] for kd in kinds} for m in modes}   # [hit@3, total]

    for item in questions:
        q = item["q"]
        expected = {_norm(f) for f in item["files"]}
        kind = item.get("kind", "conceptual")

        qv = embeddings.embed_query(q)                       # embed ONCE per question
        vec_ids = [r["id"] for r in db.vector_search(conn, qv, repo=repo, k=CANDIDATES)]
        bm_ids = retrieve.bm25_rank(q, all_chunks)[:CANDIDATES]

        ranked = {
            "vector": [by_id[i]["rel_path"] for i in vec_ids],
            "bm25": [by_id[i]["rel_path"] for i in bm_ids],
        }
        for w in weights:
            ids, _ = retrieve.rrf_fuse([vec_ids, bm_ids],
                                       weights=[config.RRF_WEIGHT_VECTOR, w])
            ranked[f"hybrid(bm25={w})"] = [by_id[i]["rel_path"] for i in ids]
        if args.rerank:
            base, _ = retrieve.rrf_fuse(
                [vec_ids, bm_ids],
                weights=[config.RRF_WEIGHT_VECTOR, config.RRF_WEIGHT_BM25])
            cand = [dict(by_id[i]) for i in base[:CANDIDATES]]
            ranked["hybrid+rerank"] = [r["rel_path"] for r in retrieve.rerank(q, cand)]

        for mode in modes:
            rank = _first_hit_rank(ranked[mode], expected)
            for k in KS:
                if rank is not None and rank <= k:
                    hits[mode][k] += 1
            rr[mode] += (1.0 / rank) if rank else 0.0
            kind_hits[mode][kind][1] += 1
            if rank is not None and rank <= 3:
                kind_hits[mode][kind][0] += 1

    conn.close()
    n = len(questions)

    counts = {kd: sum(1 for q in questions if q.get("kind", "conceptual") == kd)
              for kd in kinds}
    breakdown = ", ".join(f"{kd}: {c}" for kd, c in counts.items())
    print(f"\nRetrieval accuracy on {n} hand-labeled questions ({repo})")
    print(f"({breakdown})\n")
    header = f"{'mode':<22}" + "".join(f"  R@{k}" .rjust(9) for k in KS) + "      MRR"
    print(header)
    print("-" * len(header))
    for m in modes:
        row = f"{m:<22}" + "".join(f"{hits[m][k]/n*100:8.1f}%" for k in KS)
        row += f"   {rr[m]/n:.3f}"
        print(row)

    print(f"\nRecall@3 by question type")
    kh = f"{'mode':<22}" + "".join(f"{kd:>14}" for kd in kinds)
    print(kh)
    print("-" * len(kh))
    for m in modes:
        row = f"{m:<22}"
        for kd in kinds:
            hit, tot = kind_hits[m][kd]
            row += f"{(hit/tot*100 if tot else 0):13.1f}%"
        print(row)

    best = max(modes, key=lambda m: (hits[m][3], rr[m]))
    print(f"\n==> Best overall at Recall@3: {best} = {hits[best][3]/n*100:.1f}% "
          f"(MRR {rr[best]/n:.3f})")


if __name__ == "__main__":
    main()
