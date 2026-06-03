"""
SQLAlchemy ORM models — portable across PostgreSQL and SQLite.

These mirror backend/schema.sql. JSON columns use SQLAlchemy's portable
JSON type (stored as JSONB on PostgreSQL, TEXT-JSON on SQLite). Every
tenant-scoped table carries user_id for multi-tenant isolation.
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    BigInteger, Boolean, Date, DateTime, Float, ForeignKey, Integer,
    JSON, SmallInteger, String, Text, UniqueConstraint, Index,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from config.database import Base

# Portable big-int primary key: BIGINT on PostgreSQL, INTEGER on SQLite
# (SQLite only autoincrements INTEGER PRIMARY KEY columns).
PK = BigInteger().with_variant(Integer, "sqlite")
FK = BigInteger().with_variant(Integer, "sqlite")


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(PK, primary_key=True, autoincrement=True)
    auth_uid: Mapped[str | None] = mapped_column(String, unique=True, nullable=True)
    email: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    display_name: Mapped[str | None] = mapped_column(String)
    company: Mapped[str | None] = mapped_column(String)
    plan: Mapped[str] = mapped_column(String, default="pilot")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    sites = relationship("Site", back_populates="user", cascade="all, delete-orphan")
    turbines = relationship("Turbine", back_populates="user", cascade="all, delete-orphan")


class Site(Base):
    __tablename__ = "sites"
    id: Mapped[int] = mapped_column(PK, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(FK, ForeignKey("users.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    country: Mapped[str | None] = mapped_column(String)
    latitude: Mapped[float | None] = mapped_column(Float)
    longitude: Mapped[float | None] = mapped_column(Float)
    data_source: Mapped[str] = mapped_column(String, default="CSV_UPLOAD")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="sites")
    turbines = relationship("Turbine", back_populates="site", cascade="all, delete-orphan")


class Turbine(Base):
    __tablename__ = "turbines"
    id: Mapped[int] = mapped_column(PK, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(FK, ForeignKey("users.id", ondelete="CASCADE"), index=True)
    site_id: Mapped[int] = mapped_column(FK, ForeignKey("sites.id", ondelete="CASCADE"), index=True)
    external_ref: Mapped[str | None] = mapped_column(String)
    name: Mapped[str] = mapped_column(String, nullable=False)
    oem: Mapped[str | None] = mapped_column(String)
    model: Mapped[str | None] = mapped_column(String)
    rated_power_kw: Mapped[float | None] = mapped_column(Float)
    commissioned_on: Mapped[Date | None] = mapped_column(Date)
    data_source: Mapped[str] = mapped_column(String, default="CSV_UPLOAD")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (UniqueConstraint("user_id", "site_id", "external_ref", name="uq_turbine_ref"),)

    user = relationship("User", back_populates="turbines")
    site = relationship("Site", back_populates="turbines")


class ScadaTimeseries(Base):
    __tablename__ = "scada_timeseries"
    id: Mapped[int] = mapped_column(PK, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(BigInteger, index=True, nullable=False)
    turbine_id: Mapped[int] = mapped_column(FK, ForeignKey("turbines.id", ondelete="CASCADE"), nullable=False)
    ts: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    status_type_id: Mapped[int | None] = mapped_column(SmallInteger)
    data_source: Mapped[str] = mapped_column(String, default="CSV_UPLOAD")
    # hot signals (fast plotting/filtering)
    wind_speed: Mapped[float | None] = mapped_column(Float)
    power_kw: Mapped[float | None] = mapped_column(Float)
    rotor_rpm: Mapped[float | None] = mapped_column(Float)
    gearbox_oil_temp: Mapped[float | None] = mapped_column(Float)
    gen_bearing_de_temp: Mapped[float | None] = mapped_column(Float)
    nacelle_temp: Mapped[float | None] = mapped_column(Float)
    ambient_temp: Mapped[float | None] = mapped_column(Float)
    # full canonical signal set
    signals: Mapped[dict | None] = mapped_column(JSON)

    __table_args__ = (
        Index("ix_scada_turbine_ts", "turbine_id", "ts"),
        Index("ix_scada_user_ts", "user_id", "ts"),
    )


class Event(Base):
    __tablename__ = "events"
    id: Mapped[int] = mapped_column(PK, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(FK, ForeignKey("users.id", ondelete="CASCADE"), index=True)
    turbine_id: Mapped[int] = mapped_column(FK, ForeignKey("turbines.id", ondelete="CASCADE"), index=True)
    external_event_id: Mapped[str | None] = mapped_column(String)
    label: Mapped[str] = mapped_column(String, default="normal")
    description: Mapped[str | None] = mapped_column(Text)
    component: Mapped[str | None] = mapped_column(String)
    start_time: Mapped[datetime | None] = mapped_column(DateTime)
    end_time: Mapped[datetime | None] = mapped_column(DateTime)
    data_source: Mapped[str] = mapped_column(String, default="CSV_UPLOAD")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class MLPrediction(Base):
    __tablename__ = "ml_predictions"
    id: Mapped[int] = mapped_column(PK, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(FK, ForeignKey("users.id", ondelete="CASCADE"), index=True)
    turbine_id: Mapped[int] = mapped_column(FK, ForeignKey("turbines.id", ondelete="CASCADE"), index=True)
    ts: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    model_version: Mapped[str] = mapped_column(String, nullable=False)
    fault_probability: Mapped[float | None] = mapped_column(Float)
    predicted_class: Mapped[int | None] = mapped_column(Integer)
    predicted_component: Mapped[str | None] = mapped_column(String)
    class_probabilities: Mapped[dict | None] = mapped_column(JSON)
    rul_days: Mapped[float | None] = mapped_column(Float)
    confidence: Mapped[float | None] = mapped_column(Float)
    shap_top_features: Mapped[list | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (Index("ix_pred_turbine_ts", "turbine_id", "ts"),)


class RiskRanking(Base):
    __tablename__ = "risk_rankings"
    id: Mapped[int] = mapped_column(PK, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(FK, ForeignKey("users.id", ondelete="CASCADE"), index=True)
    turbine_id: Mapped[int] = mapped_column(FK, ForeignKey("turbines.id", ondelete="CASCADE"), index=True)
    computed_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    fault_probability: Mapped[float | None] = mapped_column(Float)
    consequence_weight: Mapped[float | None] = mapped_column(Float)
    detectability: Mapped[float | None] = mapped_column(Float)
    maintenance_coverage: Mapped[float | None] = mapped_column(Float, default=0.0)
    risk_score: Mapped[float | None] = mapped_column(Float)
    rpn: Mapped[float | None] = mapped_column(Float)
    risk_class: Mapped[str | None] = mapped_column(String)
    rank: Mapped[int | None] = mapped_column(Integer)
    rul_days: Mapped[float | None] = mapped_column(Float)
    component: Mapped[str | None] = mapped_column(String)
    financial_exposure: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class AssetHealthCertificate(Base):
    __tablename__ = "asset_health_certificates"
    id: Mapped[int] = mapped_column(PK, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(FK, ForeignKey("users.id", ondelete="CASCADE"), index=True)
    turbine_id: Mapped[int] = mapped_column(FK, ForeignKey("turbines.id", ondelete="CASCADE"), index=True)
    certificate_ref: Mapped[str | None] = mapped_column(String, unique=True)
    issued_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    valid_until: Mapped[datetime | None] = mapped_column(DateTime)
    overall_health_score: Mapped[float | None] = mapped_column(Float)
    risk_class: Mapped[str | None] = mapped_column(String)
    classification: Mapped[str | None] = mapped_column(String)
    component_scores: Mapped[dict | None] = mapped_column(JSON)
    rul_estimates: Mapped[dict | None] = mapped_column(JSON)
    narrative: Mapped[str | None] = mapped_column(Text)
    model_version: Mapped[str | None] = mapped_column(String)
    data_source: Mapped[str] = mapped_column(String, default="CSV_UPLOAD")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    # forensic / tamper-evidence fields
    content_hash: Mapped[str | None] = mapped_column(String)            # SHA-256 seal
    data_coverage_pct: Mapped[float | None] = mapped_column(Float)
    model_training_window: Mapped[str | None] = mapped_column(String)
    calibration_status: Mapped[str | None] = mapped_column(String)
    certifying_engineer: Mapped[str | None] = mapped_column(String)
    limitations: Mapped[str | None] = mapped_column(Text)


class AuditLog(Base):
    """Append-only audit trail (score overrides, sign-offs, data exclusions, cert issuance)."""
    __tablename__ = "audit_log"
    id: Mapped[int] = mapped_column(PK, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(FK, ForeignKey("users.id", ondelete="CASCADE"), index=True)
    ts: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    event_type: Mapped[str] = mapped_column(String, nullable=False)     # CERT_ISSUED | SCORE_OVERRIDE | SIGN_OFF | DATA_EXCLUDED
    turbine_id: Mapped[int | None] = mapped_column(BigInteger)
    certificate_ref: Mapped[str | None] = mapped_column(String)
    actor: Mapped[str | None] = mapped_column(String)
    detail: Mapped[dict | None] = mapped_column(JSON)
    content_hash: Mapped[str | None] = mapped_column(String)


class ModelRegistry(Base):
    __tablename__ = "model_registry"
    id: Mapped[int] = mapped_column(PK, primary_key=True, autoincrement=True)
    user_id: Mapped[int | None] = mapped_column(FK, ForeignKey("users.id", ondelete="CASCADE"), nullable=True)
    version: Mapped[str] = mapped_column(String, nullable=False)
    algorithm: Mapped[str] = mapped_column(String, nullable=False)
    trained_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    n_train_rows: Mapped[int | None] = mapped_column(Integer)
    feature_list: Mapped[list | None] = mapped_column(JSON)
    metrics: Mapped[dict | None] = mapped_column(JSON)
    artifact_path: Mapped[str | None] = mapped_column(String)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
