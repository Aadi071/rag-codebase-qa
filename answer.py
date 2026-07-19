"""Milestone 3b: answer a question about an indexed codebase, with citations.

Pipeline:  question -> hybrid_search (vector + BM25) -> grounded prompt -> LLM.

The prompt forces the model to answer ONLY from the retrieved code and to cite
sources as (path:start-end), or to say it couldn't find the answer. This is what
turns retrieval into a trustworthy, hallucination-resistant assistant.

Usage:
    python answer.py "how are aliases resolved for a command?" --repo https://github.com/pallets/click
"""

import argparse

from store import db
from retrieve import hybrid_search
import llm

SYSTEM_PROMPT = (
    "You are a codebase Q&A assistant. Answer the user's question using ONLY the "
    "code context provided. Rules:\n"
    "1. Ground every claim in the context and cite sources inline as "
    "(path:start-end), copying the path and line range from the context headers.\n"
    "2. If the answer is not present in the context, reply exactly: "
    "\"I couldn't find this in the indexed code.\" Do not guess.\n"
    "3. Do not use outside knowledge or invent files, functions, or code.\n"
    "4. Be concise and concrete; quote short snippets when helpful."
)


def build_user_prompt(question, chunks):
    parts = [f"Question: {question}", "", "Code context:"]
    for c in chunks:
        sym = f"  ({c['symbol']})" if c.get("symbol") else ""
        parts.append(f"\n--- {c['rel_path']}:{c['start_line']}-{c['end_line']}{sym}")
        parts.append(c["content"])
    parts.append("\nAnswer the question using only the context above, with (path:line) citations.")
    return "\n".join(parts)


def answer(question, repo=None, k=6):
    conn = db.connect()
    chunks = hybrid_search(conn, question, repo=repo, k=k)
    conn.close()
    if not chunks:
        return "No indexed chunks found. Did you run build_index.py for this repo?", []
    user_prompt = build_user_prompt(question, chunks)
    reply = llm.complete(SYSTEM_PROMPT, user_prompt)
    return reply, chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--repo", help="restrict to one indexed repo")
    ap.add_argument("-k", type=int, default=6, help="chunks to retrieve for context")
    args = ap.parse_args()

    reply, chunks = answer(args.question, repo=args.repo, k=args.k)
    print("\n" + reply + "\n")
    if chunks:
        print("Sources retrieved:")
        for c in chunks:
            print(f"  - {c['rel_path']}:{c['start_line']}-{c['end_line']}  {c.get('symbol') or ''}")


if __name__ == "__main__":
    main()
