"""
Database connection management.

Portable across PostgreSQL (production / Supabase) and SQLite (local dev),
so a non-technical user can run the whole stack with zero database install,
then switch to PostgreSQL by changing one line in .env.
"""
from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from .settings import settings

# SQLite needs a special flag for multi-threaded FastAPI access.
connect_args = {"check_same_thread": False} if settings.is_sqlite else {}

engine = create_engine(
    settings.DATABASE_URL,
    connect_args=connect_args,
    pool_pre_ping=True,          # recover dropped Postgres connections
    future=True,
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)

Base = declarative_base()


def get_db():
    """FastAPI dependency that yields a scoped DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def run_light_migrations():
    """
    Idempotent ADD COLUMN migrations for columns introduced after a DB was
    first created (create_all does not alter existing tables). Safe to run on
    every startup — each statement is wrapped and ignored if the column exists.
    """
    from sqlalchemy import text
    stmts = [
        "ALTER TABLE asset_health_certificates ADD COLUMN content_hash TEXT",
        "ALTER TABLE asset_health_certificates ADD COLUMN data_coverage_pct FLOAT",
        "ALTER TABLE asset_health_certificates ADD COLUMN model_training_window TEXT",
        "ALTER TABLE asset_health_certificates ADD COLUMN calibration_status TEXT",
        "ALTER TABLE asset_health_certificates ADD COLUMN certifying_engineer TEXT",
        "ALTER TABLE asset_health_certificates ADD COLUMN limitations TEXT",
    ]
    with engine.begin() as conn:
        for s in stmts:
            try:
                conn.execute(text(s))
            except Exception:
                pass  # column already exists
