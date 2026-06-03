"""
Central application settings, loaded from environment variables / .env file.

Uses pydantic-settings so every value is typed and validated at startup.
"""
from __future__ import annotations

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/ package directory (config/ -> backend/)
BACKEND_DIR = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Database ──────────────────────────────────────────────────
    DATABASE_URL: str = "sqlite:///./windsense.db"

    # ── Data ingestion ────────────────────────────────────────────
    CSV_PATH: str = ""
    MAX_SCADA_ROWS: int = 120_000           # 0 = ingest everything

    # ── First user / tenant ───────────────────────────────────────
    FIRST_USER_EMAIL: str = "operator@windsense.ai"
    FIRST_USER_NAME: str = "North Sea Operations"
    FIRST_SITE_NAME: str = "Wind Farm A"
    FIRST_SITE_COUNTRY: str = "Ireland"

    # ── ML ────────────────────────────────────────────────────────
    MODEL_DIR: str = ""                     # blank → anchored to backend/ml/artifacts
    MODEL_ALGORITHM: str = "xgboost"        # xgboost | random_forest
    ROWS_PER_CLASS: int = 1500

    # ── Auth ──────────────────────────────────────────────────────
    SUPABASE_URL: str = ""                  # https://<ref>.supabase.co (for JWKS)
    SUPABASE_JWT_SECRET: str = ""           # only needed if project uses HS256
    AUTH_MODE: str = "dev"                  # dev | email_header | jwt

    # ── API ───────────────────────────────────────────────────────
    CORS_ORIGINS: str = "*"

    @property
    def is_sqlite(self) -> bool:
        return self.DATABASE_URL.startswith("sqlite")

    @property
    def model_path(self) -> Path:
        p = Path(self.MODEL_DIR) if self.MODEL_DIR else (BACKEND_DIR / "ml" / "artifacts")
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def cors_list(self) -> list[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]


settings = Settings()
