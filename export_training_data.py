"""Export the feedback log as fine-tuning datasets (the 'train the LLM' artifact).

Turns the interactions table (question, answer, user rating) into:
  - sft.jsonl        : highly-rated (>= min-rating) question->answer pairs in chat
                       format, ready for supervised fine-tuning (OpenAI/Together
                       fine-tune APIs, or local LoRA via unsloth / axolotl).
  - preferences.jsonl: every rated interaction with its rating (reward-model /
                       data-filtering signal).

NOTE on DPO: true preference tuning needs TWO responses to the SAME prompt (a
"chosen" and a "rejected"). Collect those by re-asking a question and re-rating
the new answer; then pair same-prompt high vs low into dpo.jsonl. This export
gives you the SFT set now (the practical, high-impact path) plus the rated pool.

Usage:
    python export_training_data.py --out train --min-rating 4
"""

import argparse
import json
import os

from store import db

SYSTEM = ("You are a codebase Q&A assistant. Answer only from the provided code, "
          "cite sources as path:line, and say when you can't find the answer.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="train")
    ap.add_argument("--min-rating", type=int, default=4)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    conn = db.connect()
    rows = conn.execute(
        "SELECT question, answer, rating, repo FROM interactions "
        "WHERE rating IS NOT NULL ORDER BY id"
    ).fetchall()
    conn.close()

    sft_path = os.path.join(args.out, "sft.jsonl")
    pref_path = os.path.join(args.out, "preferences.jsonl")
    n_sft = 0

    with open(sft_path, "w", encoding="utf-8") as sft, \
         open(pref_path, "w", encoding="utf-8") as pref:
        for question, ans, rating, repo in rows:
            pref.write(json.dumps({"repo": repo, "prompt": question,
                                   "response": ans, "rating": rating}) + "\n")
            if rating >= args.min_rating:
                sft.write(json.dumps({"messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": ans},
                ]}) + "\n")
                n_sft += 1

    print(f"Wrote {n_sft} SFT examples (rating >= {args.min_rating}) -> {sft_path}")
    print(f"Wrote {len(rows)} rated examples -> {pref_path}")
    if n_sft == 0:
        print("(No highly-rated answers yet -- rate some answers in the app first.)")


if __name__ == "__main__":
    main()
