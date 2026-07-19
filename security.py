"""Auth helpers: password hashing (bcrypt) + stateless signed session tokens.

Tokens are a compact HMAC-signed JSON payload (no external JWT dependency):
    base64url(payload) + "." + base64url(HMAC-SHA256(payload))
The server can verify a token without any session store.
"""

import base64
import hashlib
import hmac
import json
import os
import time

import bcrypt

SECRET = os.getenv("SECRET_KEY", "dev-insecure-change-me").encode()


def hash_password(pw: str) -> str:
    return bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode()


def check_password(pw: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(pw.encode(), hashed.encode())
    except Exception:
        return False


def _b64e(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode().rstrip("=")


def _b64d(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def make_token(user_id: int, email: str) -> str:
    payload = _b64e(json.dumps({"uid": user_id, "email": email,
                                "iat": int(time.time())}).encode())
    sig = _b64e(hmac.new(SECRET, payload.encode(), hashlib.sha256).digest())
    return f"{payload}.{sig}"


def verify_token(token: str):
    try:
        payload, sig = token.split(".")
        expected = _b64e(hmac.new(SECRET, payload.encode(), hashlib.sha256).digest())
        if not hmac.compare_digest(sig, expected):
            return None
        return json.loads(_b64d(payload))
    except Exception:
        return None
