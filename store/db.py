"""Database layer: connect to Postgres/pgvector, store chunks, search by vector.

The two operations that matter:
  - insert_chunks(): write a batch of chunks + their embeddings.
  - search():        given a query embedding, return the top-k nearest chunks.

pgvector adds a `vector` column type and distance operators. We use `<=>`, which
is COSINE DISTANCE (0 = identical direction, 2 = opposite). Cosine similarity is
just `1 - distance`, which we return as a human-friendly score.
"""

import psycopg
from pgvector.psycopg import register_vector
from pgvector import Vector

import config


def connect(dsn=None):
    """Open a connection and teach psycopg how to talk to pgvector `vector`s."""
    conn = psycopg.connect(dsn or config.DATABASE_URL)
    # The `vector` type must exist before we can register its adapter. Docker's
    # init SQL already creates it, but ensure it here so a fresh DB works too.
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn.commit()
    register_vector(conn)  # enables passing/reading Python lists as vectors
    return conn


# Schema mirrors db/init/001_schema.sql. Docker runs that SQL automatically on a
# fresh volume; this function lets non-Docker setups (or tests) build it too.
_SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS chunks (
    id          BIGSERIAL PRIMARY KEY,
    repo        TEXT         NOT NULL,
    rel_path    TEXT         NOT NULL,
    language    TEXT         NOT NULL,
    symbol      TEXT,
    start_line  INT          NOT NULL,
    end_line    INT          NOT NULL,
    content     TEXT         NOT NULL,
    embedding   vector(%d)   NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_repo_idx ON chunks (repo);
""" % config.EMBEDDING_DIM


def init_schema(conn):
    conn.execute(_SCHEMA_SQL)
    conn.commit()


def clear_repo(conn, repo):
    """Remove a repo's chunks so re-indexing doesn't create duplicates."""
    conn.execute("DELETE FROM chunks WHERE repo = %s", (repo,))
    conn.commit()


def insert_chunks(conn, repo, chunks, embeddings):
    """Batch-insert chunks with their embeddings (parallel lists)."""
    rows = [
        (repo, c.rel_path, c.language, c.symbol,
         c.start_line, c.end_line, c.text, emb)
        for c, emb in zip(chunks, embeddings)
    ]
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO chunks "
            "(repo, rel_path, language, symbol, start_line, end_line, content, embedding) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
            rows,
        )
    conn.commit()
    return len(rows)


def search(conn, query_embedding, repo=None, k=5):
    """Return the top-k most similar chunks to `query_embedding`.

    Each result: (rel_path, language, symbol, start_line, end_line, content, score).
    `score` is cosine similarity in [-1, 1]; higher is better.
    """
    sql = (
        "SELECT rel_path, language, symbol, start_line, end_line, content, "
        "1 - (embedding <=> %s) AS score FROM chunks"
    )
    qv = Vector(query_embedding)
    params = [qv]
    if repo is not None:
        sql += " WHERE repo = %s"
        params.append(repo)
    # ORDER BY the raw distance operator so the HNSW index can be used.
    sql += " ORDER BY embedding <=> %s LIMIT %s"
    params += [qv, k]

    with conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchall()


# ---- helpers for hybrid retrieval (Milestone 3) ---------------------------
_CHUNK_COLS = ["id", "rel_path", "language", "symbol", "start_line", "end_line", "content"]


def fetch_chunks(conn, repo=None):
    """Return all chunks (as dicts) for BM25 keyword search."""
    sql = "SELECT id, rel_path, language, symbol, start_line, end_line, content FROM chunks"
    params = []
    if repo is not None:
        sql += " WHERE repo = %s"
        params.append(repo)
    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
    return [dict(zip(_CHUNK_COLS, r)) for r in rows]


def vector_search(conn, query_embedding, repo=None, k=40):
    """Top-k by cosine similarity, returned as dicts including the chunk id."""
    qv = Vector(query_embedding)
    sql = ("SELECT id, rel_path, language, symbol, start_line, end_line, content, "
           "1 - (embedding <=> %s) AS score FROM chunks")
    params = [qv]
    if repo is not None:
        sql += " WHERE repo = %s"
        params.append(repo)
    sql += " ORDER BY embedding <=> %s LIMIT %s"
    params += [qv, k]
    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
    cols = _CHUNK_COLS + ["score"]
    return [dict(zip(cols, r)) for r in rows]
