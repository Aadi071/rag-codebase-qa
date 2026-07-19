"""Hybrid retrieval: dense vector search fused with BM25 keyword search.

Why both? Embeddings capture MEANING but are weak on exact identifiers
(getUserByIdV2). BM25 is exact lexical matching: great for identifiers, weak on
paraphrase. We run both and combine their RANKINGS with Reciprocal Rank Fusion
(RRF). RRF fuses by rank position, not raw score, so the two incompatible score
scales (cosine similarity vs BM25) don't need to be normalised against each other.
"""

import re

from rank_bm25 import BM25Okapi

from embed import embeddings
from store import db

# Split on non-alphanumerics (handles snake_case), then split camelCase runs.
_WORD = re.compile(r"[A-Za-z0-9]+")
_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+")


def tokenize(text):
    """Lowercased tokens; identifiers split on snake_case AND camelCase.

    "getUserById" -> ["getuserbyid", "get", "user", "by", "id"]
    "read_config" -> ["read", "config"]  (underscore already splits _WORD)
    Keeping the whole identifier too lets exact-name queries score highest.
    """
    tokens = []
    for word in _WORD.findall(text):
        tokens.append(word.lower())
        for part in _CAMEL.findall(word):
            low = part.lower()
            if low != word.lower():
                tokens.append(low)
    return tokens


def bm25_rank(query, chunks):
    """Return chunk ids ordered best-first by BM25 score."""
    corpus = [tokenize(c["content"]) for c in chunks]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(tokenize(query))
    order = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
    return [chunks[i]["id"] for i in order]


def rrf_fuse(rankings, k=60):
    """Reciprocal Rank Fusion. score(d) = sum 1/(k + rank(d)) across rankings.

    rank is 1-based. Higher fused score = better. Returns (ordered_ids, scores).
    """
    scores = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    ordered = sorted(scores, key=scores.get, reverse=True)
    return ordered, scores


def hybrid_search(conn, query, repo=None, k=5, candidates=40):
    """Vector search + BM25, fused with RRF. Returns top-k chunk dicts."""
    qv = embeddings.embed_query(query)
    vec_rows = db.vector_search(conn, qv, repo=repo, k=candidates)
    all_chunks = db.fetch_chunks(conn, repo=repo)

    vec_ids = [r["id"] for r in vec_rows]
    bm_ids = bm25_rank(query, all_chunks)[:candidates]

    fused_ids, fused_scores = rrf_fuse([vec_ids, bm_ids])

    by_id = {r["id"]: r for r in all_chunks}
    vec_rank = {doc_id: i + 1 for i, doc_id in enumerate(vec_ids)}
    bm_rank = {doc_id: i + 1 for i, doc_id in enumerate(bm_ids)}

    results = []
    for doc_id in fused_ids[:k]:
        row = dict(by_id[doc_id])
        row["rrf"] = fused_scores[doc_id]
        row["vec_rank"] = vec_rank.get(doc_id)     # None if not in that list
        row["bm25_rank"] = bm_rank.get(doc_id)
        results.append(row)
    return results
