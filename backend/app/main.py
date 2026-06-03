"""
WindSense AI — FastAPI application entrypoint.

Run from the backend/ directory:
    uvicorn app.main:app --reload
Then open the interactive API docs:
    http://localhost:8000/docs
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings, Base, engine
from config.database import run_light_migrations
from .routers import dashboard, turbines, reports, ml_ops, validation, onboarding

# Create tables on startup (portable; for Postgres prefer schema.sql / migrations)
Base.metadata.create_all(bind=engine)
run_light_migrations()

app = FastAPI(
    title="WindSense AI — Reliability Intelligence API",
    description="OEM-agnostic fault prediction, FMECA risk ranking, RUL & "
                "Asset Health Certificates for wind turbine fleets.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(dashboard.router)
app.include_router(turbines.router)
app.include_router(reports.router)
app.include_router(ml_ops.router)
app.include_router(validation.router)
app.include_router(onboarding.router)


@app.get("/", tags=["health"])
def root():
    return {
        "service": "WindSense AI API",
        "version": "0.1.0",
        "status": "ok",
        "docs": "/docs",
        "screens": {
            "1_fleet_overview": "/api/dashboard/fleet_overview",
            "2_turbine_health": "/api/turbines/{id}/health_summary",
            "3_timeseries": "/api/turbines/{id}/timeseries",
            "3_risk_queue": "/api/fleet/risk_ranking",
            "4_asset_health_certificate": "/api/reports/asset_health_certificate",
        },
    }


@app.get("/api/health", tags=["health"])
def health():
    return {"status": "healthy", "auth_mode": settings.AUTH_MODE,
            "database": "sqlite" if settings.is_sqlite else "postgresql"}
