"""Milestone 4: FastAPI backend + chat UI for the codebase Q&A assistant.

Endpoints:
  GET  /                      -> the chat web page
  POST /api/index {repo_url}  -> start background indexing (clone/chunk/embed/store)
  GET  /api/status?repo=...   -> indexing progress for a repo
  POST /api/ask {repo_url, question} -> grounded, cited answer

Indexing runs in a BACKGROUND THREAD because it is slow (embedding hundreds of
chunks on CPU). The UI polls /api/status until the repo is 'ready', so a single
HTTP request never blocks on the whole index.
"""

import os
import threading

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from ingest.repo import clone_repo, iter_source_files
from ingest.chunker import chunk_source
from embed import embeddings
from store import db
import answer as answer_mod

app = FastAPI(title="Codebase Q&A")

_JOBS = {}                       # repo_url -> {stage, chunks, error}
_LOCK = threading.Lock()
_ACTIVE = ("cloning", "chunking", "embedding", "storing")
WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")


def _set(repo, **kw):
    with _LOCK:
        _JOBS.setdefault(repo, {}).update(kw)


def _index_job(repo_url):
    """Runs in a background thread: clone -> chunk -> embed -> store."""
    try:
        _set(repo_url, stage="cloning", error=None)
        root = clone_repo(repo_url)

        _set(repo_url, stage="chunking")
        chunks = []
        for sf in iter_source_files(root):
            with open(sf.path, "rb") as f:
                chunks.extend(chunk_source(f.read(), sf.rel_path, sf.language))

        _set(repo_url, stage="embedding", chunks=len(chunks))
        vectors = embeddings.embed_texts([c.text for c in chunks])

        _set(repo_url, stage="storing")
        conn = db.connect()
        db.init_schema(conn)
        db.clear_repo(conn, repo_url)
        db.insert_chunks(conn, repo_url, chunks, vectors)
        conn.close()

        _set(repo_url, stage="ready", chunks=len(chunks))
    except Exception as e:
        _set(repo_url, stage="error", error=str(e))


class IndexReq(BaseModel):
    repo_url: str


class AskReq(BaseModel):
    repo_url: str
    question: str
    k: int = 6


@app.get("/", response_class=HTMLResponse)
def home():
    with open(os.path.join(WEB_DIR, "index.html"), encoding="utf-8") as f:
        return f.read()


@app.post("/api/index")
def start_index(req: IndexReq):
    repo = req.repo_url.strip()
    with _LOCK:
        already = _JOBS.get(repo, {}).get("stage") in _ACTIVE
    if not already:
        threading.Thread(target=_index_job, args=(repo,), daemon=True).start()
    return {"repo": repo, "stage": _JOBS.get(repo, {}).get("stage", "cloning")}


@app.get("/api/status")
def status(repo: str):
    repo = repo.strip()
    job = _JOBS.get(repo)
    if job:
        return {"repo": repo, **job}
    # Not tracked in memory -> maybe it was indexed earlier (e.g. via the CLI).
    try:
        conn = db.connect()
        n = conn.execute("SELECT count(*) FROM chunks WHERE repo=%s", (repo,)).fetchone()[0]
        conn.close()
    except Exception:
        n = 0
    return {"repo": repo, "stage": "ready" if n else "unindexed", "chunks": n}


@app.post("/api/ask")
def ask(req: AskReq):
    reply, chunks = answer_mod.answer(req.question, repo=req.repo_url.strip(), k=req.k)
    sources = [
        {"path": c["rel_path"], "start": c["start_line"], "end": c["end_line"],
         "symbol": c.get("symbol")}
        for c in chunks
    ]
    return {"answer": reply, "sources": sources}
