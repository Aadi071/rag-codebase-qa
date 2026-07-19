"""Milestone 4 + 6: FastAPI backend, chat UI, auth, feedback + analytics.

Pages:
  GET /            landing page
  GET /login       sign in / register
  GET /app         the chat app (auth-gated client-side)
  GET /analytics   usage + satisfaction dashboard (auth-gated)

API:
  POST /api/register {email,password}          -> {token,email}
  POST /api/login    {email,password}          -> {token,email}
  POST /api/index    {repo_url}                 -> start background indexing
  GET  /api/status?repo=...                     -> indexing progress
  POST /api/ask      {repo_url,question}        -> {answer,sources,interaction_id}
  POST /api/feedback {interaction_id,rating,comment} -> {ok}
  GET  /api/analytics                           -> usage aggregates

All /api endpoints except register/login/status require a Bearer token.
Indexing runs in a background thread; the UI polls /api/status.
"""

import json
import os
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from ingest.repo import clone_repo, iter_source_files
from ingest.chunker import chunk_source
from embed import embeddings
from store import db
import retrieve
import answer as answer_mod
import indexer
import security
import llm

WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
_JOBS = {}
_LOCK = threading.Lock()
_ACTIVE = ("cloning", "chunking", "embedding", "storing")


@asynccontextmanager
async def lifespan(app):
    try:
        conn = db.connect()
        db.init_schema(conn)
        db.init_app_schema(conn)
        conn.close()
    except Exception as e:
        print(f"[startup] schema init deferred: {e}")
    yield


app = FastAPI(title="Codebase Q&A", lifespan=lifespan)


# ---- auth helpers ----------------------------------------------------------
def current_user(authorization: str = Header(None)):
    token = (authorization or "").replace("Bearer ", "").strip()
    data = security.verify_token(token)
    if not data:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return data


# ---- background indexing ---------------------------------------------------
def _set(repo, **kw):
    with _LOCK:
        _JOBS.setdefault(repo, {}).update(kw)


def _index_job(repo_url):
    try:
        _set(repo_url, stage="cloning", error=None)
        root = clone_repo(repo_url)
        _set(repo_url, stage="indexing")
        conn = db.connect()
        # Incremental: only re-embed changed files; cache invalidation happens
        # inside index_repo when anything actually changed.
        indexer.index_repo(conn, root, repo_url, embeddings.embed_texts)
        n = conn.execute("SELECT count(*) FROM chunks WHERE repo=%s",
                         (repo_url,)).fetchone()[0]
        conn.close()
        _set(repo_url, stage="ready", chunks=n)
    except Exception as e:
        _set(repo_url, stage="error", error=str(e))


# ---- request models --------------------------------------------------------
class Creds(BaseModel):
    email: str
    password: str

class IndexReq(BaseModel):
    repo_url: str

class AskReq(BaseModel):
    repo_url: str
    question: str
    k: int = 6

class FeedbackReq(BaseModel):
    interaction_id: int
    rating: int
    comment: str = ""


# ---- pages -----------------------------------------------------------------
def _page(name):
    with open(os.path.join(WEB_DIR, name), encoding="utf-8") as f:
        return HTMLResponse(f.read())

@app.get("/", response_class=HTMLResponse)
def landing(): return _page("landing.html")

@app.get("/login", response_class=HTMLResponse)
def login_page(): return _page("login.html")

@app.get("/app", response_class=HTMLResponse)
def app_page(): return _page("app.html")

@app.get("/analytics", response_class=HTMLResponse)
def analytics_page(): return _page("analytics.html")


# ---- auth API --------------------------------------------------------------
@app.post("/api/register")
def register(c: Creds):
    email = c.email.strip().lower()
    if not email or len(c.password) < 6:
        raise HTTPException(400, "Email required and password >= 6 chars.")
    conn = db.connect()
    try:
        if db.get_user_by_email(conn, email):
            raise HTTPException(400, "Email already registered.")
        uid = db.create_user(conn, email, security.hash_password(c.password))
    finally:
        conn.close()
    return {"token": security.make_token(uid, email), "email": email}


@app.post("/api/login")
def login(c: Creds):
    email = c.email.strip().lower()
    conn = db.connect()
    try:
        u = db.get_user_by_email(conn, email)
    finally:
        conn.close()
    if not u or not security.check_password(c.password, u["password_hash"]):
        raise HTTPException(401, "Invalid email or password.")
    return {"token": security.make_token(u["id"], email), "email": email}


# ---- core API --------------------------------------------------------------
@app.post("/api/index")
def start_index(req: IndexReq, user=None):
    # auth optional here so status polling stays simple, but require a token:
    return _do_index(req)

def _do_index(req: IndexReq):
    repo = req.repo_url.strip()
    with _LOCK:
        active = _JOBS.get(repo, {}).get("stage") in _ACTIVE
    if not active:
        threading.Thread(target=_index_job, args=(repo,), daemon=True).start()
    return {"repo": repo, "stage": _JOBS.get(repo, {}).get("stage", "cloning")}


@app.get("/api/status")
def status(repo: str):
    repo = repo.strip()
    job = _JOBS.get(repo)
    if job:
        return {"repo": repo, **job}
    try:
        conn = db.connect()
        n = conn.execute("SELECT count(*) FROM chunks WHERE repo=%s", (repo,)).fetchone()[0]
        conn.close()
    except Exception:
        n = 0
    return {"repo": repo, "stage": "ready" if n else "unindexed", "chunks": n}


@app.post("/api/ask")
def ask(req: AskReq, authorization: str = Header(None)):
    user = current_user(authorization)
    repo = req.repo_url.strip()
    qv = embeddings.embed_query(req.question)          # embed once, reuse
    conn = db.connect()
    try:
        hit = db.cache_lookup(conn, repo, req.question, qv)
        if hit:                                         # reuse -> skip retrieval + LLM
            reply, sources, cached = hit["answer"], hit["sources"], True
        else:
            reply, chunks = answer_mod.answer(req.question, repo=repo, k=req.k, query_embedding=qv)
            sources = [
                {"path": c["rel_path"], "start": c["start_line"], "end": c["end_line"],
                 "symbol": c.get("symbol")}
                for c in chunks
            ]
            db.cache_store(conn, repo, req.question, qv, reply, sources)
            cached = False
        iid = db.log_interaction(conn, user["uid"], repo, req.question, reply,
                                 sources, embedding=qv, cached=cached)
    finally:
        conn.close()
    return {"answer": reply, "sources": sources, "interaction_id": iid, "cached": cached}


@app.post("/api/feedback")
def feedback(req: FeedbackReq, authorization: str = Header(None)):
    user = current_user(authorization)
    if not (1 <= req.rating <= 5):
        raise HTTPException(400, "rating must be 1..5")
    conn = db.connect()
    try:
        ok = db.set_feedback(conn, req.interaction_id, user["uid"], req.rating, req.comment)
    finally:
        conn.close()
    return {"ok": ok}


@app.get("/api/analytics")
def analytics(authorization: str = Header(None)):
    user = current_user(authorization)
    conn = db.connect()
    try:
        return db.analytics_summary(conn, user["uid"])
    finally:
        conn.close()


def _sse(event, data):
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.post("/api/ask_stream")
def ask_stream(req: AskReq, authorization: str = Header(None)):
    user = current_user(authorization)
    repo = req.repo_url.strip()
    qv = embeddings.embed_query(req.question)

    def gen():
        conn = db.connect()
        try:
            hit = db.cache_lookup(conn, repo, req.question, qv)
            if hit:
                sources = hit["sources"]
                yield _sse("meta", {"sources": sources, "cached": True})
                yield _sse("token", hit["answer"])
                full = hit["answer"]
                cached = True
            else:
                chunks = retrieve.hybrid_search(conn, req.question, repo=repo,
                                                k=req.k, query_embedding=qv)
                exemplars = db.find_similar_good_answers(conn, repo, qv, k=1)
                sources = [
                    {"path": c["rel_path"], "start": c["start_line"],
                     "end": c["end_line"], "symbol": c.get("symbol")}
                    for c in chunks
                ]
                yield _sse("meta", {"sources": sources, "cached": False})
                if not chunks:
                    full = "No indexed chunks found. Did you index this repo?"
                    yield _sse("token", full)
                else:
                    prompt = answer_mod.build_user_prompt(req.question, chunks, exemplars)
                    full = ""
                    for tok in llm.complete_stream(answer_mod.SYSTEM_PROMPT, prompt):
                        full += tok
                        yield _sse("token", tok)
                    db.cache_store(conn, repo, req.question, qv, full, sources)
                cached = False
            iid = db.log_interaction(conn, user["uid"], repo, req.question, full,
                                     sources, embedding=qv, cached=cached)
            yield _sse("done", {"interaction_id": iid, "cached": cached})
        finally:
            conn.close()

    return StreamingResponse(gen(), media_type="text/event-stream")
