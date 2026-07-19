import config
import indexer
import retrieve
from embed import embeddings
from store import db


def test_index_hybrid_and_bm25(conn, sample_repo):
    indexer.index_repo(conn, sample_repo, "r", embeddings.embed_texts)
    chunks = db.fetch_chunks(conn, repo="r")
    assert chunks
    by = {c["id"]: c for c in chunks}
    bm = retrieve.bm25_rank("resolve_alias", chunks)
    assert "resolve_alias" in by[bm[0]]["content"]          # exact identifier match
    res = retrieve.hybrid_search(conn, "resolve alias", repo="r", k=3)
    assert res and all("rrf" in r and "vec_rank" in r for r in res)


def test_incremental_reuse(conn, sample_repo, monkeypatch):
    indexer.index_repo(conn, sample_repo, "r", embeddings.embed_texts)
    calls = {"n": 0}
    base = embeddings.embed_texts
    monkeypatch.setattr(embeddings, "embed_texts",
                        lambda ts: (calls.__setitem__("n", calls["n"] + len(ts)) or base(ts)))
    stats = indexer.index_repo(conn, sample_repo, "r", embeddings.embed_texts)
    assert stats["embedded_chunks"] == 0 and calls["n"] == 0 and stats["reused_files"] >= 2


def test_answer_cache(conn):
    qv = embeddings.embed_query("hello world")
    assert db.cache_lookup(conn, "r", "hello world", qv) is None
    db.cache_store(conn, "r", "hello world", qv, "cached answer",
                   [{"path": "a", "start": 1, "end": 2}])
    hit = db.cache_lookup(conn, "r", "hello world", qv)
    assert hit and hit["answer"] == "cached answer"
    db.cache_invalidate(conn, "r")
    assert db.cache_lookup(conn, "r", "hello world", qv) is None


def test_rerank_reorders(conn, sample_repo, monkeypatch):
    indexer.index_repo(conn, sample_repo, "r", embeddings.embed_texts)
    monkeypatch.setattr(retrieve, "_score_pairs",
                        lambda pairs: [p[1].count("resolve") for p in pairs])
    monkeypatch.setattr(config, "RERANK", True)
    res = retrieve.hybrid_search(conn, "resolve", repo="r", k=3)
    scores = [r["rerank_score"] for r in res]
    assert scores == sorted(scores, reverse=True)
