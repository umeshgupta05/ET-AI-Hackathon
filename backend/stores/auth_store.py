"""SQLAlchemy-backed auth, profile, and case history helpers."""

import hashlib
import hmac
import json
import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from jose import JWTError, jwt
from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import insert

from .database import engine, SessionLocal
from .models import Base, User, CaseHistory

JWT_SECRET = os.getenv("JWT_SECRET") or "dev-change-me-citizen-fraud-shield"
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_MINUTES = int(os.getenv("ACCESS_TOKEN_MINUTES", "480"))
PBKDF2_ITERATIONS = 600_000

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

def init_db() -> None:
    Base.metadata.create_all(bind=engine)

def _hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        PBKDF2_ITERATIONS,
    )
    return f"pbkdf2_sha256${PBKDF2_ITERATIONS}${salt}${digest.hex()}"

def _verify_password(password: str, stored: str) -> bool:
    parts = stored.split("$")
    if len(parts) == 3:
        method, salt, expected = parts
        iterations = 150_000
    elif len(parts) == 4:
        method, raw_iterations, salt, expected = parts
        try:
            iterations = int(raw_iterations)
        except ValueError:
            return False
    else:
        return False
    if method != "pbkdf2_sha256":
        return False
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        iterations,
    )
    return hmac.compare_digest(digest.hex(), expected)

def _user_from_orm(user: User) -> dict[str, Any]:
    return {
        "id": user.id,
        "name": user.name,
        "email": user.email,
        "preferred_language": user.preferred_language,
        "created_at": user.created_at,
    }

def create_user(name: str, email: str, password: str, preferred_language: str) -> dict[str, Any]:
    language = preferred_language if preferred_language in SUPPORTED_LANGUAGES else "en"
    user_id = secrets.token_urlsafe(16)
    now = datetime.now(timezone.utc).isoformat()
    
    with SessionLocal() as db:
        new_user = User(
            id=user_id,
            name=name.strip(),
            email=email.lower().strip(),
            hashed_password=_hash_password(password),
            preferred_language=language,
            created_at=now
        )
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        return _user_from_orm(new_user)

def authenticate_user(email: str, password: str) -> Optional[dict[str, Any]]:
    with SessionLocal() as db:
        user = db.query(User).filter(User.email == email.lower().strip()).first()
    if not user or not _verify_password(password, user.hashed_password):
        return None
    return _user_from_orm(user)

def get_user(user_id: str) -> Optional[dict[str, Any]]:
    with SessionLocal() as db:
        user = db.query(User).filter(User.id == user_id).first()
    return _user_from_orm(user) if user else None

def update_user(user_id: str, name: Optional[str] = None, preferred_language: Optional[str] = None) -> Optional[dict[str, Any]]:
    with SessionLocal() as db:
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            return None
        if name is not None:
            user.name = name.strip()
        if preferred_language is not None:
            user.preferred_language = preferred_language if preferred_language in SUPPORTED_LANGUAGES else "en"
        db.commit()
        db.refresh(user)
        return _user_from_orm(user)

def create_access_token(user: dict[str, Any]) -> str:
    expires = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_MINUTES)
    with SessionLocal() as db:
        u = db.query(User).filter(User.id == user["id"]).first()
        token_version = u.token_version if u else 0
    payload = {"sub": user["id"], "ver": token_version, "exp": expires}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def decode_token(token: str) -> Optional[dict[str, Any]]:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        user_id = payload.get("sub")
        if not user_id:
            return None
        with SessionLocal() as db:
            user = db.query(User).filter(User.id == user_id).first()
        if not user or int(payload.get("ver", -1)) != user.token_version:
            return None
        return _user_from_orm(user)
    except JWTError:
        return None

def revoke_tokens(user_id: str) -> None:
    with SessionLocal() as db:
        user = db.query(User).filter(User.id == user_id).first()
        if user:
            user.token_version += 1
            db.commit()

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
    
    with SessionLocal() as db:
        stmt = insert(CaseHistory).values(
            id=case_id,
            user_id=user_id,
            case_type=case_type,
            verdict=result.get("verdict") or result.get("final_verdict"),
            risk_level=result.get("risk_level"),
            confidence=float(result.get("confidence", 0.0) or 0.0),
            payload_json=result,
            created_at=now
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=['id'],
            set_={
                'user_id': stmt.excluded.user_id,
                'case_type': stmt.excluded.case_type,
                'verdict': stmt.excluded.verdict,
                'risk_level': stmt.excluded.risk_level,
                'confidence': stmt.excluded.confidence,
                'payload_json': stmt.excluded.payload_json,
                'created_at': stmt.excluded.created_at,
            }
        )
        db.execute(stmt)
        db.commit()
        
    return result

def get_case(user_id: str, case_id: str) -> Optional[dict[str, Any]]:
    with SessionLocal() as db:
        case = db.query(CaseHistory).filter(CaseHistory.id == case_id, CaseHistory.user_id == user_id).first()
    if not case:
        return None
    return {
        "id": case.id,
        "case_type": case.case_type,
        "created_at": case.created_at,
        "result": case.payload_json,
    }

from sqlalchemy import cast, String, func

def get_history(user_id: str, search_query: Optional[str] = None) -> list[dict[str, Any]]:
    with SessionLocal() as db:
        query = db.query(CaseHistory).filter(CaseHistory.user_id == user_id)
        if search_query:
            # PostgreSQL full-text search on the JSON payload casted to text
            query = query.filter(
                func.to_tsvector('english', cast(CaseHistory.payload_json, String))
                .op('@@')(func.websearch_to_tsquery('english', search_query))
            )
        cases = query.order_by(CaseHistory.created_at.desc()).limit(100).all()
    return [
        {
            "id": c.id,
            "case_type": c.case_type,
            "verdict": c.verdict,
            "risk_level": c.risk_level,
            "confidence": c.confidence,
            "created_at": c.created_at,
            "result": c.payload_json,
        }
        for c in cases
    ]
