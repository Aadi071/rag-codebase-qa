"""Shared test fixtures. Uses pgserver (bundled Postgres + pgvector, no root) and
stubs the embedder + LLM so tests run with no torch, no Ollama, no network."""

import hashlib
import os
import sys
import tempfile

import numpy as np
import pytest
import pgserver

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="session")
def _server():
    srv = pgserver.get_server(tempfile.mkdtemp())
    import config
    config.DATABASE_URL = srv.get_uri()
    yield srv
    srv.cleanup()


@pytest.fixture(autouse=True)
def _stubs(_server, monkeypatch):
    import retrieve
    from embed import embeddings
    import llm

    def stub(texts):
        out = []
        for t in texts:
            v = np.zeros(768)
            for tok in retrieve.tokenize(t):
                v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % 768] += 1.0
            nrm = np.linalg.norm(v)
            out.append((v / nrm if nrm else v).tolist())
        return out

    monkeypatch.setattr(embeddings, "embed_texts", lambda ts: stub(ts))
    monkeypatch.setattr(embeddings, "embed_query", lambda t: stub([t])[0])
    monkeypatch.setattr(llm, "complete", lambda s, u: "Grounded answer (pkg/core.py:1-3).")
    monkeypatch.setattr(llm, "complete_stream",
                        lambda s, u: iter(["Grounded ", "answer ", "(pkg/core.py:1-3)."]))


@pytest.fixture
def conn(_server):
    from store import db
    c = db.connect()
    db.init_schema(c)
    db.init_app_schema(c)
    for t in ("answer_cache", "interactions", "chunks", "users"):
        c.execute(f"TRUNCATE {t} RESTART IDENTITY CASCADE")
    c.commit()
    yield c
    c.close()


@pytest.fixture
def sample_repo(tmp_path):
    pkg = tmp_path / "repo" / "pkg"
    pkg.mkdir(parents=True)
    files = {
        "core.py": ("class Resolver:\n"
                    "    def resolve_alias(self, name):\n"
                    "        return name.upper()\n\n"
                    "def top_level_helper(x):\n"
                    "    return x + 1\n"),
        "util.py": "def slugify(text):\n    return text.lower().replace(' ', '-')\n",
        "math_ops.py": "def add(a, b):\n    return a + b\n\ndef subtract(a, b):\n    return a - b\n",
        "strings.py": "def upper_it(s):\n    return s.upper()\n\ndef lower_it(s):\n    return s.lower()\n",
        "io_ops.py": "def read_file(p):\n    with open(p) as f:\n        return f.read()\n",
        "net.py": "def fetch(url):\n    return 'GET ' + url\n",
    }
    for name, content in files.items():
        (pkg / name).write_text(content)
    return str(tmp_path / "repo")
