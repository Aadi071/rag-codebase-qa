"""Milestone 2 driver: clone -> chunk -> embed -> store into pgvector.

Usage:
    python build_index.py https://github.com/pallets/click
    python build_index.py ./some/local/folder --local --name my-repo
"""

import argparse

from ingest.repo import clone_repo, iter_source_files
from ingest.chunker import chunk_source
import config
from embed import embeddings          # imported as module so tests can stub it
from store import db


def build(target, local=False, repo_name=None):
    root = target if local else clone_repo(target)
    repo_name = repo_name or target

    chunks = []
    for sf in iter_source_files(root):
        with open(sf.path, "rb") as f:
            chunks.extend(chunk_source(f.read(), sf.rel_path, sf.language))
    print(f"Chunked {repo_name}: {len(chunks)} chunks")

    print(f"Embedding {len(chunks)} chunks (backend: {config.EMBEDDING_BACKEND}, model: {config.EMBEDDING_MODEL})...")
    vectors = embeddings.embed_texts([c.text for c in chunks])

    conn = db.connect()
    db.init_schema(conn)
    db.clear_repo(conn, repo_name)                # idempotent re-index
    n = db.insert_chunks(conn, repo_name, chunks, vectors)
    conn.close()
    print(f"Stored {n} chunks for {repo_name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target", help="GitHub URL, or local path with --local")
    ap.add_argument("--local", action="store_true")
    ap.add_argument("--name", help="repo name to store under (defaults to target)")
    args = ap.parse_args()
    build(args.target, local=args.local, repo_name=args.name)


if __name__ == "__main__":
    main()
