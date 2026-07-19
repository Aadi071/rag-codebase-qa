"""Central config: reads .env so every module shares the same settings."""

import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://rag:rag@localhost:5432/rag")

# --- Embeddings ------------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "local").lower()
if EMBEDDING_BACKEND == "openai":
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    EMBEDDING_DIM = 1536
else:
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
    EMBEDDING_DIM = 768

# --- Answer LLM ------------------------------------------------------------
# "ollama" (local, free), "anthropic", or "openai".
LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama").lower()
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")

# --- Optional cross-encoder re-ranker (higher precision, extra compute) -------
RERANK = os.getenv("RERANK", "false").lower() in ("1", "true", "yes")
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "20"))
