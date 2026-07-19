"""Ask a question: hybrid-retrieve (vector + BM25) and print top-k chunks.

Usage:
    python search.py "how does click parse options?" --repo https://github.com/pallets/click -k 5
"""

import argparse

from store import db
from retrieve import hybrid_search


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--repo", help="restrict to one indexed repo")
    ap.add_argument("-k", type=int, default=5)
    args = ap.parse_args()

    conn = db.connect()
    results = hybrid_search(conn, args.question, repo=args.repo, k=args.k)
    conn.close()

    for r in results:
        ranks = f"vec#{r['vec_rank']} bm25#{r['bm25_rank']}"
        print(f"[rrf {r['rrf']:.4f}  {ranks}] {r['rel_path']}:{r['start_line']}-{r['end_line']}  {r['symbol'] or ''}")
        preview = r["content"] if len(r["content"]) <= 300 else r["content"][:300] + " ..."
        print(preview)
        print("-" * 64)


if __name__ == "__main__":
    main()
