"""
═══════════════════════════════════════════════════════════════════════
 RISK RANKING SERVICE   (powers Screen 1 + Screen 3)
═══════════════════════════════════════════════════════════════════════
Takes the latest ML prediction per turbine and computes the FMECA-weighted
fleet risk ranking:

    risk_score = fault_probability × consequence_weight × (1 − maintenance_coverage)
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from ml import fmeca


def _latest_prediction_per_turbine(db: Session, user_id: int) -> dict[int, object]:
    from models import MLPrediction
    rows = db.execute(
        select(MLPrediction)
        .where(MLPrediction.user_id == user_id)
        .order_by(MLPrediction.turbine_id, MLPrediction.ts.desc(), MLPrediction.id.desc())
    ).scalars().all()
    latest: dict[int, object] = {}
    for p in rows:
        if p.turbine_id not in latest:          # first seen = newest (sorted desc)
            latest[p.turbine_id] = p
    return latest


def recompute_risk_rankings(db: Session, user_id: int) -> int:
    """Rebuild the risk_rankings table for a user. Returns rows written."""
    from models import RiskRanking, Turbine

    latest = _latest_prediction_per_turbine(db, user_id)
    turbines = {t.id: t for t in db.execute(
        select(Turbine).where(Turbine.user_id == user_id)
    ).scalars().all()}

    computed = []
    for tid, pred in latest.items():
        turbine = turbines.get(tid)
        component = pred.predicted_component or "Normal"
        fp = pred.fault_probability or 0.0
        score = fmeca.risk_score(fp, component, maintenance_coverage=0.0)
        rpn = fmeca.rpn(fp, component, pred.rul_days)
        rc = fmeca.risk_class(score)
        rated = (turbine.rated_power_kw if turbine and turbine.rated_power_kw else 2000.0)
        exposure = fmeca.financial_exposure(component, fp, rated_power_kw=rated)
        computed.append({
            "turbine_id": tid, "fault_probability": fp,
            "consequence_weight": fmeca.consequence_weight(component),
            "detectability": fmeca.detectability_factor(component, pred.rul_days),
            "risk_score": round(score, 4), "rpn": rpn, "risk_class": rc,
            "rul_days": pred.rul_days, "component": component,
            "financial_exposure": exposure,
        })

    # Rank by Expected Economic Loss (€) — the auditable, money-based ranking.
    # (Previously ranked by the 0..1 risk_score; EEL is defensible line-by-line.)
    computed.sort(key=lambda x: x["financial_exposure"], reverse=True)
    for rank, row in enumerate(computed, start=1):
        row["rank"] = rank

    # replace previous snapshot
    db.query(RiskRanking).filter(RiskRanking.user_id == user_id).delete()
    now = datetime.utcnow()
    for row in computed:
        db.add(RiskRanking(user_id=user_id, computed_at=now, **row))
    db.commit()
    return len(computed)


def get_risk_ranking(db: Session, user_id: int, limit: int | None = None) -> list:
    from models import RiskRanking
    q = (select(RiskRanking)
         .where(RiskRanking.user_id == user_id)
         .order_by(RiskRanking.rank))
    if limit:
        q = q.limit(limit)
    return db.execute(q).scalars().all()
