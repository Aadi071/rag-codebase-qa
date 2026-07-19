import time

import pytest
from fastapi.testclient import TestClient

import app as appmod


@pytest.fixture
def client(conn, sample_repo, monkeypatch):
    monkeypatch.setattr(appmod, "clone_repo", lambda url: sample_repo)
    with TestClient(appmod.app) as c:
        yield c


def _auth(client):
    t = client.post("/api/register", json={"email": "a@b.com", "password": "secret1"}).json()["token"]
    return {"Authorization": "Bearer " + t}


def test_auth_required(client):
    assert client.post("/api/ask", json={"repo_url": "r", "question": "x"}).status_code == 401


def _index_and_wait(client, H):
    client.post("/api/index", json={"repo_url": "r"}, headers=H)
    for _ in range(60):
        if client.get("/api/status", params={"repo": "r"}).json().get("stage") in ("ready", "error"):
            return
        time.sleep(0.25)


def test_full_flow(client):
    H = _auth(client)
    _index_and_wait(client, H)
    d = client.post("/api/ask", json={"repo_url": "r", "question": "how to resolve an alias"},
                    headers=H).json()
    assert d["answer"] and d["interaction_id"]
    fb = client.post("/api/feedback",
                     json={"interaction_id": d["interaction_id"], "rating": 5}, headers=H).json()
    assert fb["ok"]
    a = client.get("/api/analytics", headers=H).json()
    assert a["total_questions"] >= 1 and a["avg_rating"] == 5.0


def test_cache_hit_second_time(client):
    H = _auth(client)
    _index_and_wait(client, H)
    q = {"repo_url": "r", "question": "resolve alias question"}
    r1 = client.post("/api/ask", json=q, headers=H).json()
    r2 = client.post("/api/ask", json=q, headers=H).json()
    assert r1["cached"] is False and r2["cached"] is True
