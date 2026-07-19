"""Milestone 1 driver: clone a repo, walk its code, AST-chunk it, show samples.

Run it:
    python index.py https://github.com/pallets/click
    python index.py https://github.com/pallets/click --limit 5
    python index.py ./some/local/folder --local     # skip cloning

The goal of this milestone is to LOOK at the output and confirm the chunks are
sensible: functions and classes stay whole, and each chunk knows its file, line
range, and symbol name. That metadata is what powers citations later.
"""

import argparse

from ingest.repo import clone_repo, iter_source_files
from ingest.chunker import chunk_source


def main():
    ap = argparse.ArgumentParser(description="Clone a repo and AST-chunk its code.")
    ap.add_argument("target", help="GitHub URL, or a local folder path with --local")
    ap.add_argument("--local", action="store_true", help="treat target as a local folder")
    ap.add_argument("--limit", type=int, default=8, help="how many sample chunks to print")
    args = ap.parse_args()

    if args.local:
        root = args.target
        print(f"Walking local folder: {root}")
    else:
        print(f"Cloning {args.target} ...")
        root = clone_repo(args.target)
        print(f"Cloned to: {root}")

    files = list(iter_source_files(root))
    print(f"Found {len(files)} source files")

    all_chunks = []
    for sf in files:
        with open(sf.path, "rb") as f:
            source = f.read()
        all_chunks.extend(chunk_source(source, sf.rel_path, sf.language))

    print(f"Produced {len(all_chunks)} chunks\n")

    print("=" * 72)
    for c in all_chunks[: args.limit]:
        symbol = c.symbol or "(module-level)"
        print(f"{c.rel_path}:{c.start_line}-{c.end_line}  [{c.language}]  {symbol}")
        print("-" * 72)
        preview = c.text if len(c.text) <= 600 else c.text[:600] + "\n... (truncated)"
        print(preview)
        print("=" * 72)


if __name__ == "__main__":
    main()
