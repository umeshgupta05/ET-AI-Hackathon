from sqlalchemy import Column, String, Integer, Float, ForeignKey, JSON, Index, UniqueConstraint
from .database import Base

class User(Base):
    __tablename__ = "users"
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    email = Column(String, nullable=False, unique=True)
    hashed_password = Column(String, nullable=False)
    preferred_language = Column(String, nullable=False, default="en")
    token_version = Column(Integer, nullable=False, default=0)
    created_at = Column(String, nullable=False)

class CaseHistory(Base):
    __tablename__ = "case_history"
    id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    case_type = Column(String, nullable=False)
    verdict = Column(String)
    risk_level = Column(String)
    confidence = Column(Float)
    payload_json = Column(JSON, nullable=False)
    created_at = Column(String, nullable=False)
    
    __table_args__ = (
        Index("idx_case_history_user", "user_id"),
    )

class AnalysisJob(Base):
    __tablename__ = "analysis_jobs"
    id = Column(String, primary_key=True)
    user_id = Column(String, nullable=False)
    status = Column(String, nullable=False)
    input_json = Column(JSON)
    result_json = Column(JSON)
    error = Column(String)
    attempts = Column(Integer, nullable=False, default=0)
    lease_expires_at = Column(String)
    created_at = Column(String, nullable=False)
    updated_at = Column(String, nullable=False)
    
    __table_args__ = (
        Index("idx_jobs_status", "status"),
    )

class GeospatialIncident(Base):
    __tablename__ = "geospatial_incidents"
    id = Column(String, primary_key=True)
    district = Column(String, nullable=False)
    lat = Column(Float, nullable=False)
    lon = Column(Float, nullable=False)
    type = Column(String, nullable=False)
    reports = Column(Integer, nullable=False)
    severity = Column(Float, nullable=False)
    source = Column(String, nullable=False)
    source_reference = Column(String)
    occurred_at = Column(String)
    received_at = Column(String, nullable=False)
    metadata_json = Column(JSON, nullable=False)

    __table_args__ = (
        Index("idx_geospatial_time", "received_at"),
    )

class GraphEntity(Base):
    __tablename__ = "graph_entities"
    id = Column(String, primary_key=True)
    entity_type = Column(String, nullable=False)
    label = Column(String, nullable=False)
    source = Column(String, nullable=False)
    source_reference = Column(String)
    updated_at = Column(String, nullable=False)
    attrs_json = Column(JSON, nullable=False)

class GraphEdge(Base):
    __tablename__ = "graph_edges"
    id = Column(String, primary_key=True)
    source_id = Column(String, nullable=False)
    target_id = Column(String, nullable=False)
    edge_type = Column(String, nullable=False)
    weight = Column(Float, nullable=False)
    source = Column(String, nullable=False)
    source_reference = Column(String)
    observed_at = Column(String)
    updated_at = Column(String, nullable=False)
    attrs_json = Column(JSON, nullable=False)

class ReportingDraft(Base):
    __tablename__ = "reporting_drafts"
    id = Column(String, primary_key=True)
    case_id = Column(String)
    user_id = Column(String)
    destination = Column(String, nullable=False)
    status = Column(String, nullable=False)
    payload_json = Column(JSON, nullable=False)
    integrity_hash = Column(String, nullable=False)
    created_at = Column(String, nullable=False)

class CurrencyCertifiedSpecimen(Base):
    __tablename__ = "currency_certified_specimens"
    id = Column(String, primary_key=True)
    denomination = Column(String, nullable=False)
    label = Column(String, nullable=False)
    issuer = Column(String, nullable=False)
    image_sha256 = Column(String, nullable=False)
    certification_reference = Column(String, nullable=False)
    captured_device = Column(String)
    captured_at = Column(String)
    received_at = Column(String, nullable=False)
    metadata_json = Column(JSON, nullable=False)

class ModelFeedback(Base):
    __tablename__ = "model_feedback"
    id = Column(String, primary_key=True)
    case_id = Column(String, nullable=False)
    user_id = Column(String, nullable=False)
    predicted_verdict = Column(String, nullable=False)
    predicted_confidence = Column(Float, nullable=False)
    observed_outcome = Column(String, nullable=False)
    reporting_reference = Column(String)
    notes = Column(String)
    review_status = Column(String, nullable=False)
    created_at = Column(String, nullable=False)

class RealtimeSession(Base):
    __tablename__ = "realtime_sessions"
    id = Column(String, primary_key=True)
    channel = Column(String, nullable=False)
    language = Column(String, nullable=False)
    status = Column(String, nullable=False)
    caller_hash = Column(String)
    participant_hash = Column(String)
    risk_score = Column(Float, nullable=False)
    risk_level = Column(String, nullable=False)
    event_count = Column(Integer, nullable=False)
    started_at = Column(String, nullable=False)
    updated_at = Column(String, nullable=False)
    closed_at = Column(String)
    metadata_json = Column(JSON, nullable=False)

class RealtimeEvent(Base):
    __tablename__ = "realtime_events"
    id = Column(String, primary_key=True)
    session_id = Column(String, nullable=False)
    sequence = Column(Integer, nullable=False)
    transcript = Column(String, nullable=False)
    metadata_json = Column(JSON, nullable=False)
    signal_score = Column(Float, nullable=False)
    model_score = Column(Float, nullable=False)
    combined_score = Column(Float, nullable=False)
    reasons_json = Column(JSON, nullable=False)
    occurred_at = Column(String, nullable=False)
    evidence_hash = Column(String, nullable=False)
    
    __table_args__ = (
        UniqueConstraint("session_id", "sequence", name="uq_events_session_seq"),
        Index("idx_events_session", "session_id", "sequence"),
    )

class AlertOutbox(Base):
    __tablename__ = "alert_outbox"
    id = Column(String, primary_key=True)
    session_id = Column(String, nullable=False)
    destination = Column(String, nullable=False)
    idempotency_key = Column(String, nullable=False, unique=True)
    status = Column(String, nullable=False)
    attempt_count = Column(Integer, nullable=False)
    payload_json = Column(JSON, nullable=False)
    payload_hash = Column(String, nullable=False)
    last_error = Column(String)
    created_at = Column(String, nullable=False)
    updated_at = Column(String, nullable=False)
    
    __table_args__ = (
        Index("idx_alerts_session", "session_id", "created_at"),
    )
