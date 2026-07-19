"""Turn text into embedding vectors, via a local model or the OpenAI API.

Two public functions:
  - embed_texts(texts): embed DOCUMENTS (code chunks). Returns vectors in order.
  - embed_query(text):  embed a QUERY. For the bge model we prepend a short
    retrieval instruction, which the model was trained to expect and which
    noticeably improves search quality (asymmetric query/document embedding).

The heavy local model is loaded lazily on first use and cached.
"""

import time

import config

_openai_client = None
_local_model = None

# bge-* models want this exact instruction prepended to QUERIES only.
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


# ---- OpenAI backend -------------------------------------------------------
def _embed_openai(texts, batch_size=100):
    from openai import OpenAI
    global _openai_client
    if _openai_client is None:
        if not config.OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is not set (add it to your .env).")
        _openai_client = OpenAI(api_key=config.OPENAI_API_KEY)

    vectors = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        for attempt in range(3):
            try:
                resp = _openai_client.embeddings.create(
                    model=config.EMBEDDING_MODEL, input=batch)
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2 * (attempt + 1))
        vectors.extend(item.embedding for item in resp.data)
    return vectors


# ---- Local (sentence-transformers) backend --------------------------------
def _get_local_model():
    global _local_model
    if _local_model is None:
        from sentence_transformers import SentenceTransformer
        # First call downloads the model (a few hundred MB), then it's cached.
        _local_model = SentenceTransformer(config.EMBEDDING_MODEL)
    return _local_model


def _embed_local(texts):
    model = _get_local_model()
    vecs = model.encode(list(texts), normalize_embeddings=True,
                        batch_size=64, show_progress_bar=False)
    return [v.tolist() for v in vecs]


# ---- Public API -----------------------------------------------------------
def embed_texts(texts):
    """Embed documents (code chunks)."""
    if config.EMBEDDING_BACKEND == "openai":
        return _embed_openai(texts)
    return _embed_local(texts)


def embed_query(text):
    """Embed a single search query."""
    if config.EMBEDDING_BACKEND == "openai":
        return _embed_openai([text])[0]
    return _embed_local([_BGE_QUERY_PREFIX + text])[0]
