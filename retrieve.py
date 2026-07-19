"""Hybrid retrieval: dense vector search fused with BM25 keyword search.

Why both? Embeddings capture MEANING but are weak on exact identifiers
(getUserByIdV2). BM25 is exact lexical matching: great for identifiers, weak on
paraphrase. We run both and combine their RANKINGS with Reciprocal Rank Fusion
(RRF). RRF fuses by rank position, not raw score, so the two incompatible score
scales (cosine similarity vs BM25) don't need to be normalised against each other.
"""

import re

from rank_bm25 import BM25Okapi

import config
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


_cross_encoder = None


def _score_pairs(pairs):
    """Cross-encoder relevance scores for (query, chunk_text) pairs.

    Lazy-loads the model on first use. Tests monkeypatch this function.
    """
    global _cross_encoder
    from sentence_transformers import CrossEncoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder(config.RERANK_MODEL)
    return _cross_encoder.predict(pairs)


def rerank(query, results):
    """Reorder candidate chunk dicts by a cross-encoder relevance score.

    A cross-encoder reads the query and chunk TOGETHER (unlike bi-encoder
    embeddings), giving sharper relevance -- at the cost of running the model on
    each candidate, which is why we only rerank the top fused candidates.
    """
    if not results:
        return results
    scores = _score_pairs([(query, r["content"]) for r in results])
    for r, sc in zip(results, scores):
        r["rerank_score"] = float(sc)
    return sorted(results, key=lambda r: r["rerank_score"], reverse=True)


def hybrid_search(conn, query, repo=None, k=5, candidates=40, query_embedding=None):
    """Vector search + BM25, fused with RRF. Returns top-k chunk dicts.

    Pass `query_embedding` to reuse an already-computed query vector (avoids a
    second embedding call when the caller also needs the vector).
    """
    qv = query_embedding if query_embedding is not None else embeddings.embed_query(query)
    vec_rows = db.vector_search(conn, qv, repo=repo, k=candidates)
    all_chunks = db.fetch_chunks(conn, repo=repo)

    vec_ids = [r["id"] for r in vec_rows]
    bm_ids = bm25_rank(query, all_chunks)[:candidates]

    fused_ids, fused_scores = rrf_fuse([vec_ids, bm_ids])

    by_id = {r["id"]: r for r in all_chunks}
    vec_rank = {doc_id: i + 1 for i, doc_id in enumerate(vec_ids)}
    bm_rank = {doc_id: i + 1 for i, doc_id in enumerate(bm_ids)}

    # If reranking, keep more candidates for the cross-encoder to reorder.
    n_cand = max(k, config.RERANK_CANDIDATES) if config.RERANK else k
    results = []
    for doc_id in fused_ids[:n_cand]:
        row = dict(by_id[doc_id])
        row["rrf"] = fused_scores[doc_id]
        row["vec_rank"] = vec_rank.get(doc_id)     # None if not in that list
        row["bm25_rank"] = bm_rank.get(doc_id)
        results.append(row)
    if config.RERANK:
        results = rerank(query, results)
    return results[:k]
