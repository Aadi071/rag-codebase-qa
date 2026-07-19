"""Milestone 2 driver: clone -> chunk -> embed -> store into pgvector.

Usage:
    python build_index.py https://github.com/pallets/click
    python build_index.py ./some/local/folder --local --name my-repo
"""

import argparse

from ingest.repo import clone_repo
import config
from embed import embeddings          # imported as module so tests can stub it
from store import db
import indexer


def build(target, local=False, repo_name=None):
    root = target if local else clone_repo(target)
    repo_name = repo_name or target

    print(f"Indexing {repo_name} (incremental; backend: {config.EMBEDDING_BACKEND}, "
          f"model: {config.EMBEDDING_MODEL})...")
    conn = db.connect()
    stats = indexer.index_repo(conn, root, repo_name, embeddings.embed_texts)
    conn.close()
    print(f"Done: {stats['changed_files']} changed, {stats['reused_files']} reused, "
          f"{stats['removed_files']} removed; embedded {stats['embedded_chunks']} chunks.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target", help="GitHub URL, or local path with --local")
    ap.add_argument("--local", action="store_true")
    ap.add_argument("--name", help="repo name to store under (defaults to target)")
    args = ap.parse_args()
    build(args.target, local=args.local, repo_name=args.name)


if __name__ == "__main__":
    main()
