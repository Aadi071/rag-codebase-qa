"""LLM client: generate a chat completion from a chosen backend.

Default backend is Ollama (local, free). The others are lazy-imported so you
only need the provider you actually use.
"""

import json
import os
import urllib.error
import urllib.request

import config


def _ollama_chat(system, user):
    url = config.OLLAMA_HOST.rstrip("/") + "/api/chat"
    body = json.dumps({
        "model": config.OLLAMA_MODEL,
        "stream": False,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }).encode("utf-8")
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            data = json.loads(resp.read())
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Could not reach Ollama at {config.OLLAMA_HOST} ({e}). "
            f"Is it running (`ollama serve`) and is the model pulled "
            f"(`ollama pull {config.OLLAMA_MODEL}`)?"
        )
    return data["message"]["content"]


def _anthropic_chat(system, user):
    import anthropic
    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    msg = client.messages.create(
        model=config.ANTHROPIC_MODEL, max_tokens=1024,
        system=system, messages=[{"role": "user", "content": user}],
    )
    return msg.content[0].text


def _openai_chat(system, user):
    # Works with real OpenAI or any OpenAI-compatible API (Gemini, Groq, ...)
    # via config.OPENAI_BASE_URL. base_url=None falls back to real OpenAI.
    from openai import OpenAI
    client = OpenAI(api_key=config.OPENAI_API_KEY, base_url=config.OPENAI_BASE_URL)
    resp = client.chat.completions.create(
        model=config.OPENAI_CHAT_MODEL,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
    )
    return resp.choices[0].message.content


def complete(system, user):
    if config.LLM_BACKEND == "anthropic":
        return _anthropic_chat(system, user)
    if config.LLM_BACKEND == "openai":
        return _openai_chat(system, user)
    return _ollama_chat(system, user)


def _ollama_stream(system, user):
    url = config.OLLAMA_HOST.rstrip("/") + "/api/chat"
    body = json.dumps({
        "model": config.OLLAMA_MODEL, "stream": True,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
    }).encode("utf-8")
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        resp = urllib.request.urlopen(req, timeout=600)
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Could not reach Ollama at {config.OLLAMA_HOST} ({e}). "
            f"Is it running and is the model pulled (ollama pull {config.OLLAMA_MODEL})?"
        )
    for line in resp:                      # Ollama streams JSONL, one obj per line
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        piece = obj.get("message", {}).get("content", "")
        if piece:
            yield piece
        if obj.get("done"):
            break


def complete_stream(system, user):
    """Yield answer tokens as they are generated (Ollama). Other backends yield
    the whole answer once (no token streaming), so callers work either way."""
    if config.LLM_BACKEND == "ollama":
        yield from _ollama_stream(system, user)
    else:
        yield complete(system, user)
