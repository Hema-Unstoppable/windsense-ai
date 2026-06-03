"""
═══════════════════════════════════════════════════════════════════════
 FLEET KPI SERVICE   (powers Screen 1 header cards)
═══════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

from datetime import datetime, timedelta

from sqlalchemy import select, func
from sqlalchemy.orm import Session

from .risk_ranking import get_risk_ranking


def compute_fleet_kpis(db: Session, user_id: int) -> dict:
    from models import Turbine, RiskRanking, ScadaTimeseries, MLPrediction

    turbines = db.execute(
        select(Turbine).where(Turbine.user_id == user_id)
    ).scalars().all()
    n_turbines = len(turbines)

    rankings = get_risk_ranking(db, user_id)
    critical = sum(1 for r in rankings if r.risk_class == "CRITICAL")
    high = sum(1 for r in rankings if r.risk_class == "HIGH")
    active_alerts = sum(1 for r in rankings if r.risk_class in ("CRITICAL", "HIGH", "MEDIUM"))
    exposure = sum((r.financial_exposure or 0) for r in rankings)

    # fleet health = mean of (100 − fault%) across turbines with predictions
    preds = _latest_predictions(db, user_id)
    if preds:
        health = sum(100 * (1 - (p.fault_probability or 0)) for p in preds) / len(preds)
    else:
        health = 100.0

    # turbines offline = those whose most recent status_type_id indicates downtime (4)
    offline = _count_offline(db, user_id, turbines)

    # predicted failures next 30 days = predictions with RUL <= 30 and fault prob high
    pred_30d = sum(1 for p in preds
                   if (p.rul_days or 999) <= 30 and (p.fault_probability or 0) >= 0.5)

    availability = round(100.0 * (n_turbines - offline) / n_turbines, 1) if n_turbines else 0.0

    return {
        "turbines_monitored": n_turbines,
        "turbines_operating": n_turbines - offline,
        "turbines_offline": offline,
        "fleet_health_score": round(health, 0),
        "critical_alerts": critical,
        "total_active_alerts": active_alerts,
        "predicted_failures_30d": pred_30d,
        "fleet_availability": availability,
        "financial_exposure_eur": round(exposure, 0),
    }


def _latest_predictions(db: Session, user_id: int) -> list:
    from models import MLPrediction
    rows = db.execute(
        select(MLPrediction).where(MLPrediction.user_id == user_id)
        .order_by(MLPrediction.turbine_id, MLPrediction.ts.desc(), MLPrediction.id.desc())
    ).scalars().all()
    seen, out = set(), []
    for p in rows:
        if p.turbine_id not in seen:
            seen.add(p.turbine_id)
            out.append(p)
    return out


def _count_offline(db: Session, user_id: int, turbines: list) -> int:
    from models import ScadaTimeseries
    offline = 0
    for t in turbines:
        last = db.execute(
            select(ScadaTimeseries.status_type_id)
            .where(ScadaTimeseries.turbine_id == t.id)
            .order_by(ScadaTimeseries.ts.desc()).limit(1)
        ).scalar()
        if last in (4,):                          # 4 = downtime
            offline += 1
    return offline


def build_anomaly_alerts(db: Session, user_id: int, limit: int = 5) -> list[dict]:
    """Most recent high-severity predictions, formatted for the alerts panel."""
    from models import Turbine
    rankings = get_risk_ranking(db, user_id, limit=limit)
    turbines = {t.id: t for t in db.execute(
        select(Turbine).where(Turbine.user_id == user_id)
    ).scalars().all()}
    alerts = []
    for r in rankings:
        if r.risk_class not in ("CRITICAL", "HIGH", "MEDIUM"):
            continue
        t = turbines.get(r.turbine_id)
        sev = "CRITICAL" if r.risk_class == "CRITICAL" else "WARNING"
        msg = f"{r.component} fault risk — RUL {int(r.rul_days or 0)}d"
        alerts.append({
            "turbine_id": r.turbine_id,
            "turbine_name": t.name if t else f"T-{r.turbine_id}",
            "severity": sev, "component": r.component, "message": msg,
            "ml_confidence": round((r.fault_probability or 0) * 100, 0),
            "detected_ago": "live",
        })
    return alerts
