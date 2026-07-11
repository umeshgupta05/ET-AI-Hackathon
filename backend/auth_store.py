"""SQLite-backed auth, profile, and case history helpers."""

import hashlib
import hmac
import json
import os
import secrets
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from jose import JWTError, jwt

DB_PATH = Path(__file__).resolve().parent / "data" / "app.db"
JWT_SECRET = os.getenv("JWT_SECRET", "dev-change-me-citizen-fraud-shield")
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_MINUTES = int(os.getenv("ACCESS_TOKEN_MINUTES", "480"))


SUPPORTED_LANGUAGES = {
    "en": "English",
    "hi": "हिन्दी",
    "te": "తెలుగు",
    "ta": "தமிழ்",
    "kn": "ಕನ್ನಡ",
    "bn": "বাংলা",
    "mr": "मराठी",
    "gu": "ગુજરાતી",
    "ml": "മലയാളം",
    "pa": "ਪੰਜਾਬੀ",
    "or": "ଓଡ଼ିଆ",
    "ur": "اردو",
}


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                hashed_password TEXT NOT NULL,
                preferred_language TEXT NOT NULL DEFAULT 'en',
                token_version INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS case_history (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                case_type TEXT NOT NULL,
                verdict TEXT,
                risk_level TEXT,
                confidence REAL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
            """
        )


def _hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 150_000)
    return f"pbkdf2_sha256${salt}${digest.hex()}"


def _verify_password(password: str, stored: str) -> bool:
    try:
        method, salt, expected = stored.split("$", 2)
    except ValueError:
        return False
    if method != "pbkdf2_sha256":
        return False
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 150_000)
    return hmac.compare_digest(digest.hex(), expected)


def _user_from_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "name": row["name"],
        "email": row["email"],
        "preferred_language": row["preferred_language"],
        "created_at": row["created_at"],
    }


def create_user(name: str, email: str, password: str, preferred_language: str) -> dict[str, Any]:
    language = preferred_language if preferred_language in SUPPORTED_LANGUAGES else "en"
    user_id = secrets.token_urlsafe(16)
    now = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO users (id, name, email, hashed_password, preferred_language, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (user_id, name.strip(), email.lower().strip(), _hash_password(password), language, now),
        )
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    return _user_from_row(row)


def authenticate_user(email: str, password: str) -> Optional[dict[str, Any]]:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM users WHERE email = ?", (email.lower().strip(),)).fetchone()
    if not row or not _verify_password(password, row["hashed_password"]):
        return None
    return _user_from_row(row)


def get_user(user_id: str) -> Optional[dict[str, Any]]:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    return _user_from_row(row) if row else None


def update_user(user_id: str, name: Optional[str] = None, preferred_language: Optional[str] = None) -> Optional[dict[str, Any]]:
    updates = []
    values: list[Any] = []
    if name is not None:
        updates.append("name = ?")
        values.append(name.strip())
    if preferred_language is not None:
        updates.append("preferred_language = ?")
        values.append(preferred_language if preferred_language in SUPPORTED_LANGUAGES else "en")
    if updates:
        values.append(user_id)
        with _connect() as conn:
            conn.execute(f"UPDATE users SET {', '.join(updates)} WHERE id = ?", values)
    return get_user(user_id)


def create_access_token(user: dict[str, Any]) -> str:
    expires = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_MINUTES)
    with _connect() as conn:
        row = conn.execute("SELECT token_version FROM users WHERE id = ?", (user["id"],)).fetchone()
    token_version = int(row["token_version"]) if row else 0
    payload = {"sub": user["id"], "ver": token_version, "exp": expires}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> Optional[dict[str, Any]]:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        user_id = payload.get("sub")
        if not user_id:
            return None
        with _connect() as conn:
            row = conn.execute("SELECT token_version FROM users WHERE id = ?", (user_id,)).fetchone()
        if not row or int(payload.get("ver", -1)) != int(row["token_version"]):
            return None
        return get_user(user_id)
    except JWTError:
        return None


def revoke_tokens(user_id: str) -> None:
    """Invalidate every access token previously issued to a user."""
    with _connect() as conn:
        conn.execute("UPDATE users SET token_version = token_version + 1 WHERE id = ?", (user_id,))


def save_case(user_id: str, case_type: str, result: dict[str, Any]) -> dict[str, Any]:
    case_id = result.get("case_id") or secrets.token_urlsafe(12)
    result["case_id"] = case_id
    now = datetime.now(timezone.utc).isoformat()
    evidence_payload = json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    result["evidence_integrity"] = {
        "algorithm": "SHA-256",
        "hash": hashlib.sha256(evidence_payload.encode("utf-8")).hexdigest(),
        "captured_at": now,
        "schema_version": "1.0",
    }
    with _connect() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO case_history
            (id, user_id, case_type, verdict, risk_level, confidence, payload_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                case_id,
                user_id,
                case_type,
                result.get("verdict") or result.get("final_verdict"),
                result.get("risk_level"),
                float(result.get("confidence", 0.0) or 0.0),
                json.dumps(result),
                now,
            ),
        )
    return result


def get_case(user_id: str, case_id: str) -> Optional[dict[str, Any]]:
    """Return a case only when it belongs to the authenticated user."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT payload_json, case_type, created_at FROM case_history WHERE id = ? AND user_id = ?",
            (case_id, user_id),
        ).fetchone()
    if not row:
        return None
    return {
        "id": case_id,
        "case_type": row["case_type"],
        "created_at": row["created_at"],
        "result": json.loads(row["payload_json"]),
    }


def get_history(user_id: str) -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT id, case_type, verdict, risk_level, confidence, payload_json, created_at
            FROM case_history
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT 100
            """,
            (user_id,),
        ).fetchall()
    return [
        {
            "id": row["id"],
            "case_type": row["case_type"],
            "verdict": row["verdict"],
            "risk_level": row["risk_level"],
            "confidence": row["confidence"],
            "created_at": row["created_at"],
            "result": json.loads(row["payload_json"]),
        }
        for row in rows
    ]
