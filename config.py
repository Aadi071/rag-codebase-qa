"""Central config: reads .env so every module shares the same settings."""

import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://rag:rag@localhost:5432/rag")

# --- Embeddings ------------------------------------------------------------
# "local" = free bge-base (needs torch). "openai" = ANY OpenAI-COMPATIBLE API:
# real OpenAI, or a free provider via OPENAI_BASE_URL — e.g. Google Gemini's
# compat endpoint https://generativelanguage.googleapis.com/v1beta/openai/ .
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL") or None  # None = real OpenAI
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "local").lower()
if EMBEDDING_BACKEND == "openai":
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1536"))
else:
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
    EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "768"))

# Embeddings may use a DIFFERENT provider than the chat LLM (e.g. Groq for chat,
# which has no embeddings API, + Gemini for embeddings). Default to chat creds.
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY") or OPENAI_API_KEY
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL") or OPENAI_BASE_URL
# If set, request this output dimensionality. Providers like gemini-embedding-001
# support Matryoshka truncation; must equal EMBEDDING_DIM (the stored vector size).
_emb_dims = os.getenv("EMBEDDING_DIMENSIONS")
EMBEDDING_DIMENSIONS = int(_emb_dims) if _emb_dims else None
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "100"))

# --- Answer LLM ------------------------------------------------------------
# "ollama" (local, free), "anthropic", or "openai". The "openai" backend also
# talks to any OpenAI-compatible API via OPENAI_BASE_URL (Gemini, Groq, etc.).
LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama").lower()
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

# --- Optional cross-encoder re-ranker (higher precision, extra compute) -------
RERANK = os.getenv("RERANK", "false").lower() in ("1", "true", "yes")
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "20"))

# --- Hybrid fusion weights (how much each retriever counts in RRF) -----------
# MEASURED on a 28-question labeled set: BM25 hurt at every weight, because
# AST chunks already carry symbol names + class signatures, so the embedding
# handles exact-identifier lookup. Default 0.0 = vector-only. Set >0 to
# re-enable hybrid (may help with a weaker embedder or unusual codebases).
RRF_WEIGHT_VECTOR = float(os.getenv("RRF_WEIGHT_VECTOR", "1.0"))
RRF_WEIGHT_BM25 = float(os.getenv("RRF_WEIGHT_BM25", "0.0"))

# --- Deployment guardrails ---------------------------------------------------
# A public /api/index clones + embeds arbitrary repos, which is unbounded cost.
# On a public demo set ALLOW_INDEXING=false (serve pre-indexed repos only), or
# restrict with INDEX_ALLOWLIST="https://github.com/a/b,https://github.com/c/d".
ALLOW_INDEXING = os.getenv("ALLOW_INDEXING", "true").lower() in ("1", "true", "yes")
INDEX_ALLOWLIST = [u.strip() for u in os.getenv("INDEX_ALLOWLIST", "").split(",") if u.strip()]


def indexing_allowed(repo_url: str) -> bool:
    if not ALLOW_INDEXING:
        return False
    return (not INDEX_ALLOWLIST) or repo_url.strip() in INDEX_ALLOWLIST
