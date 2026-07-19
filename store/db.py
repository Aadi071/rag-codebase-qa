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
    embedding   vector(%d)   NOT NULL,
    file_hash   TEXT
);
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_repo_idx ON chunks (repo);
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS file_hash TEXT;
""" % config.EMBEDDING_DIM


def init_schema(conn):
    conn.execute(_SCHEMA_SQL)
    conn.commit()


def clear_repo(conn, repo):
    """Remove a repo's chunks so re-indexing doesn't create duplicates."""
    conn.execute("DELETE FROM chunks WHERE repo = %s", (repo,))
    conn.commit()


def insert_chunks(conn, repo, chunks, embeddings, file_hashes=None):
    """Batch-insert chunks with their embeddings (parallel lists).

    `file_hashes` (parallel to chunks) records the hash of the source file each
    chunk came from, enabling incremental re-indexing.
    """
    if file_hashes is None:
        file_hashes = [None] * len(chunks)
    rows = [
        (repo, c.rel_path, c.language, c.symbol,
         c.start_line, c.end_line, c.text, emb, fh)
        for c, emb, fh in zip(chunks, embeddings, file_hashes)
    ]
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO chunks "
            "(repo, rel_path, language, symbol, start_line, end_line, content, embedding, file_hash) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
            rows,
        )
    conn.commit()
    return len(rows)


def get_repo_file_hashes(conn, repo):
    """Return {rel_path: file_hash} for a repo's currently-indexed files."""
    rows = conn.execute(
        "SELECT DISTINCT rel_path, file_hash FROM chunks WHERE repo=%s", (repo,)
    ).fetchall()
    return {r[0]: r[1] for r in rows}


def delete_file_chunks(conn, repo, rel_path):
    conn.execute("DELETE FROM chunks WHERE repo=%s AND rel_path=%s", (repo, rel_path))
    conn.commit()


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


# ==== App tables: users + interaction/feedback log (Milestone 6) ============
from psycopg.types.json import Json  # noqa: E402

_APP_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS users (
    id            BIGSERIAL PRIMARY KEY,
    email         TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS interactions (
    id                 BIGSERIAL PRIMARY KEY,
    user_id            BIGINT REFERENCES users(id),
    repo               TEXT NOT NULL,
    question           TEXT NOT NULL,
    answer             TEXT NOT NULL,
    sources            JSONB,
    rating             INT,        -- 1..5 satisfaction; NULL until feedback given
    comment            TEXT,
    question_embedding vector({config.EMBEDDING_DIM}),  -- for few-shot exemplar lookup
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);
-- upgrade older tables that predate the embedding column:
ALTER TABLE interactions
    ADD COLUMN IF NOT EXISTS question_embedding vector({config.EMBEDDING_DIM});
ALTER TABLE interactions
    ADD COLUMN IF NOT EXISTS cached BOOL NOT NULL DEFAULT false;
CREATE TABLE IF NOT EXISTS answer_cache (
    id                 BIGSERIAL PRIMARY KEY,
    repo               TEXT NOT NULL,
    question           TEXT NOT NULL,
    question_embedding vector({config.EMBEDDING_DIM}),
    answer             TEXT NOT NULL,
    sources            JSONB,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS answer_cache_repo_idx ON answer_cache(repo);
"""


def init_app_schema(conn):
    conn.execute(_APP_SCHEMA)
    conn.commit()


def create_user(conn, email, password_hash):
    row = conn.execute(
        "INSERT INTO users (email, password_hash) VALUES (%s, %s) RETURNING id",
        (email, password_hash),
    ).fetchone()
    conn.commit()
    return row[0]


def get_user_by_email(conn, email):
    r = conn.execute(
        "SELECT id, email, password_hash FROM users WHERE email = %s", (email,)
    ).fetchone()
    return {"id": r[0], "email": r[1], "password_hash": r[2]} if r else None


def log_interaction(conn, user_id, repo, question, answer, sources, embedding=None, cached=False):
    emb = Vector(embedding) if embedding is not None else None
    row = conn.execute(
        "INSERT INTO interactions "
        "(user_id, repo, question, answer, sources, question_embedding, cached) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id",
        (user_id, repo, question, answer, Json(sources), emb, cached),
    ).fetchone()
    conn.commit()
    return row[0]


def find_similar_good_answers(conn, repo, query_embedding, k=1, min_rating=4):
    """Nearest highly-rated past answers to `query_embedding` (few-shot exemplars).

    If `repo` is given, restrict to that repo; otherwise search across all repos.
    """
    if query_embedding is None:
        return []
    qv = Vector(query_embedding)
    rows = conn.execute(
        "SELECT question, answer, rating FROM interactions "
        "WHERE rating >= %s AND question_embedding IS NOT NULL "
        "AND (%s::text IS NULL OR repo = %s) "
        "ORDER BY question_embedding <=> %s LIMIT %s",
        (min_rating, repo, repo, qv, k),
    ).fetchall()
    return [{"question": r[0], "answer": r[1], "rating": r[2]} for r in rows]


def set_feedback(conn, interaction_id, user_id, rating, comment):
    cur = conn.execute(
        "UPDATE interactions SET rating = %s, comment = %s "
        "WHERE id = %s AND user_id = %s",
        (rating, comment, interaction_id, user_id),
    )
    conn.commit()
    return cur.rowcount > 0


def analytics_summary(conn, user_id):
    total = conn.execute(
        "SELECT count(*) FROM interactions WHERE user_id=%s", (user_id,)
    ).fetchone()[0]
    rated, avg = conn.execute(
        "SELECT count(rating), avg(rating) FROM interactions WHERE user_id=%s",
        (user_id,),
    ).fetchone()
    repos = conn.execute(
        "SELECT count(DISTINCT repo) FROM interactions WHERE user_id=%s", (user_id,)
    ).fetchone()[0]
    cache_hits = conn.execute(
        "SELECT count(*) FROM interactions WHERE user_id=%s AND cached", (user_id,)
    ).fetchone()[0]
    dist = {str(i): 0 for i in range(1, 6)}
    for r, c in conn.execute(
        "SELECT rating, count(*) FROM interactions "
        "WHERE user_id=%s AND rating IS NOT NULL GROUP BY rating", (user_id,)
    ).fetchall():
        dist[str(r)] = c
    over_time = [
        {"day": d.isoformat(), "count": c}
        for d, c in conn.execute(
            "SELECT date_trunc('day', created_at)::date AS d, count(*) "
            "FROM interactions WHERE user_id=%s GROUP BY d ORDER BY d", (user_id,)
        ).fetchall()
    ]
    recent = [
        {"id": r[0], "repo": r[1], "question": r[2], "rating": r[3],
         "created_at": r[4].isoformat()}
        for r in conn.execute(
            "SELECT id, repo, question, rating, created_at FROM interactions "
            "WHERE user_id=%s ORDER BY created_at DESC LIMIT 10", (user_id,)
        ).fetchall()
    ]
    return {
        "total_questions": total,
        "rated_count": rated or 0,
        "avg_rating": round(float(avg), 2) if avg is not None else None,
        "distinct_repos": repos,
        "cache_hits": cache_hits,
        "cache_hit_rate": round(cache_hits / total * 100, 1) if total else 0.0,
        "rating_distribution": dist,
        "questions_over_time": over_time,
        "recent": recent,
    }


# ==== Answer cache: reuse answers for identical / near-identical questions ====
def cache_lookup(conn, repo, question, query_embedding, threshold=0.93):
    """Return a cached answer for this repo if the question is an exact match or
    semantically very close (cosine >= threshold). Skips retrieval + the LLM."""
    r = conn.execute(
        "SELECT answer, sources FROM answer_cache "
        "WHERE repo=%s AND question=%s ORDER BY created_at DESC LIMIT 1",
        (repo, question),
    ).fetchone()
    if r:
        return {"answer": r[0], "sources": r[1] or [], "match": "exact", "similarity": 1.0}
    if query_embedding is None:
        return None
    qv = Vector(query_embedding)
    r = conn.execute(
        "SELECT answer, sources, 1 - (question_embedding <=> %s) AS sim "
        "FROM answer_cache WHERE repo=%s AND question_embedding IS NOT NULL "
        "ORDER BY question_embedding <=> %s LIMIT 1",
        (qv, repo, qv),
    ).fetchone()
    if r and r[2] is not None and float(r[2]) >= threshold:
        return {"answer": r[0], "sources": r[1] or [], "match": "semantic",
                "similarity": round(float(r[2]), 3)}
    return None


def cache_store(conn, repo, question, query_embedding, answer, sources):
    emb = Vector(query_embedding) if query_embedding is not None else None
    conn.execute(
        "INSERT INTO answer_cache (repo, question, question_embedding, answer, sources) "
        "VALUES (%s, %s, %s, %s, %s)",
        (repo, question, emb, answer, Json(sources)),
    )
    conn.commit()


def cache_invalidate(conn, repo):
    """Drop cached answers for a repo (call on re-index so answers never go stale).

    Tolerant of a missing cache table (e.g. CLI indexing before the app has
    created the app-schema tables).
    """
    try:
        conn.execute("DELETE FROM answer_cache WHERE repo=%s", (repo,))
        conn.commit()
    except psycopg.errors.UndefinedTable:
        conn.rollback()
