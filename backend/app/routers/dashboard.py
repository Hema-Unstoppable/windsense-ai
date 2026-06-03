"""
SCREEN 1 — Fleet Overview / Reliability Dashboard
SCREEN 3 (queue) — Fleet risk ranking
"""
from __future__ import annotations

from datetime import datetime
from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from config import get_db
from app.deps import get_current_user
from models.schemas import (
    FleetOverviewResponse, FleetKPIs, RiskRow, AnomalyAlert, DataSourceInfo,
)
from services import kpis
from services.risk_ranking import get_risk_ranking

router = APIRouter(prefix="/api", tags=["screen 1 · fleet overview"])


def _data_source(db, user) -> DataSourceInfo:
    from models import Site
    site = db.execute(select(Site).where(Site.user_id == user.id)).scalars().first()
    src = site.data_source if site else "CSV_UPLOAD"
    labels = {
        "SCADA_LIVE": ("SCADA live feed", "scada-live"),
        "CSV_UPLOAD": ("Historical CSV import", "csv"),
        "HISTORIAN_PI": ("OSIsoft PI historian", "historian"),
        "API_PULL": ("API pull", "api"),
        "SIMULATED": ("Simulated data", "sim"),
    }
    label, icon = labels.get(src, ("Data feed", "data"))
    return DataSourceInfo(type=src, label=label, icon=icon, last_updated=datetime.utcnow())


@router.get("/dashboard/fleet_overview", response_model=FleetOverviewResponse)
def fleet_overview(top: int = Query(10, ge=1, le=100),
                   db: Session = Depends(get_db), user=Depends(get_current_user)):
    from models import Turbine, Site

    kpi = kpis.compute_fleet_kpis(db, user.id)

    turbines = {t.id: t for t in db.execute(
        select(Turbine).where(Turbine.user_id == user.id)).scalars().all()}
    sites = {s.id: s for s in db.execute(
        select(Site).where(Site.user_id == user.id)).scalars().all()}

    rows = []
    for r in get_risk_ranking(db, user.id, limit=top):
        t = turbines.get(r.turbine_id)
        site = sites.get(t.site_id) if t else None
        rows.append(RiskRow(
            turbine_id=r.turbine_id,
            turbine_name=t.name if t else f"T-{r.turbine_id}",
            site_name=site.name if site else "—",
            risk_score=round(r.risk_score or 0, 3), rpn=r.rpn or 0,
            risk_class=r.risk_class or "LOW", rank=r.rank or 0,
            health_score=round(100 * (1 - (r.fault_probability or 0)), 0),
            rul_days=r.rul_days, component=r.component,
            fault_probability=round(r.fault_probability or 0, 3),
            financial_exposure_eur=r.financial_exposure or 0,
            status=_status(r.risk_class),
            **_eel_detail(r, t),
        ))

    alerts = [AnomalyAlert(**a) for a in kpis.build_anomaly_alerts(db, user.id, limit=5)]

    return FleetOverviewResponse(
        kpis=FleetKPIs(**kpi), risk_ranking=rows, anomaly_alerts=alerts,
        data_source=_data_source(db, user), generated_at=datetime.utcnow(),
    )


@router.get("/fleet/risk_ranking", response_model=list[RiskRow],
            tags=["screen 3 · risk queue"])
def fleet_risk_ranking(limit: int = Query(50, ge=1, le=200),
                       db: Session = Depends(get_db), user=Depends(get_current_user)):
    from models import Turbine, Site
    turbines = {t.id: t for t in db.execute(
        select(Turbine).where(Turbine.user_id == user.id)).scalars().all()}
    sites = {s.id: s for s in db.execute(
        select(Site).where(Site.user_id == user.id)).scalars().all()}
    out = []
    for r in get_risk_ranking(db, user.id, limit=limit):
        t = turbines.get(r.turbine_id)
        site = sites.get(t.site_id) if t else None
        out.append(RiskRow(
            turbine_id=r.turbine_id, turbine_name=t.name if t else f"T-{r.turbine_id}",
            site_name=site.name if site else "—",
            risk_score=round(r.risk_score or 0, 3), rpn=r.rpn or 0,
            risk_class=r.risk_class or "LOW", rank=r.rank or 0,
            health_score=round(100 * (1 - (r.fault_probability or 0)), 0),
            rul_days=r.rul_days, component=r.component,
            fault_probability=round(r.fault_probability or 0, 3),
            financial_exposure_eur=r.financial_exposure or 0,
            status=_status(r.risk_class),
            **_eel_detail(r, t),
        ))
    return out


def _status(rc: str | None) -> str:
    return {"CRITICAL": "Critical", "HIGH": "Warning",
            "MEDIUM": "Warning", "LOW": "Operational"}.get(rc or "LOW", "Operational")


def _eel_detail(r, turbine) -> dict:
    """Auditable Expected-Economic-Loss breakdown for a risk row, computed on the fly."""
    from ml import fmeca
    comp = r.component or "Hydraulic"
    rated = (turbine.rated_power_kw if turbine and turbine.rated_power_kw else 2000.0)
    fp = r.fault_probability or 0.0
    cost = fmeca.consequence_cost(comp, rated_power_kw=rated)
    eel = round(fp * cost["total"])
    from services.recommendations import build_recommendation
    rec = build_recommendation(fp, r.rul_days, comp, cost["severity"], eel, explain=None)
    return {
        "expected_loss_eur": eel,
        "cost_breakdown": cost,
        "action_priority": fmeca.action_priority(fp, comp, r.rul_days),
        "iso14224_code": fmeca.ISO14224_TAXONOMY.get(comp, {}).get("code"),
        "severity_criteria": fmeca.SEVERITY_CRITERIA.get(comp),
        "recommended_action": rec["action_label"],
    }
