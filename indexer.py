"""Incremental indexing: only (re-)embed files whose content changed.

Each chunk stores the SHA-256 of its source file. On re-index we compare the
current files' hashes to what's already stored:
  - unchanged file  -> keep its existing chunks (no re-chunk, no re-embed)
  - new/changed file -> drop its old chunks, re-chunk + re-embed
  - deleted file     -> drop its chunks
Embedding is the expensive step, so this makes re-indexing a large repo after a
small change nearly free.
"""

import hashlib

from ingest.repo import iter_source_files
from ingest.chunker import chunk_source
from store import db


def index_repo(conn, root, repo, embed_fn):
    db.init_schema(conn)
    existing = db.get_repo_file_hashes(conn, repo)   # {rel_path: file_hash}

    seen = set()
    changed_chunks, changed_hashes, changed_files = [], [], []
    for sf in iter_source_files(root):
        content = open(sf.path, "rb").read()
        h = hashlib.sha256(content).hexdigest()
        seen.add(sf.rel_path)
        if existing.get(sf.rel_path) == h:
            continue                                  # unchanged -> reuse chunks
        db.delete_file_chunks(conn, repo, sf.rel_path)
        for c in chunk_source(content, sf.rel_path, sf.language):
            changed_chunks.append(c)
            changed_hashes.append(h)
        changed_files.append(sf.rel_path)

    removed = set(existing) - seen
    for rel in removed:
        db.delete_file_chunks(conn, repo, rel)

    embedded = 0
    if changed_chunks:
        vectors = embed_fn([c.text for c in changed_chunks])
        db.insert_chunks(conn, repo, changed_chunks, vectors, file_hashes=changed_hashes)
        embedded = len(changed_chunks)

    if changed_files or removed:                      # code changed -> drop stale cache
        db.cache_invalidate(conn, repo)

    return {
        "changed_files": len(changed_files),
        "removed_files": len(removed),
        "reused_files": len(seen) - len(changed_files),
        "embedded_chunks": embedded,
    }
