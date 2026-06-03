"""
SCREEN 2 — Turbine Health & Predictions   (/health_summary, /predictions, /rul)
SCREEN 3 — Time-Series & Anomaly Explorer (/timeseries)
"""
from __future__ import annotations

from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from config import get_db
from app.deps import get_current_user
from app.routers.dashboard import _data_source
from models.schemas import (
    TurbineHealthSummary, FailureModePrediction, ComponentHealth, ShapFeature,
    ExplainabilityIntegrity, TimeseriesResponse, TimeseriesPoint, AnomalyMarker,
)
from ml import fmeca
from ml.explainability import annotate_shap
from services.asset_health import _component_scores, _narrative

router = APIRouter(prefix="/api/turbines", tags=["screen 2 · turbine health"])


@router.get("")
def list_turbines(db: Session = Depends(get_db), user=Depends(get_current_user)):
    """All turbines for the tenant, with their latest health snapshot."""
    from models import Turbine, Site, RiskRanking
    turbines = db.execute(
        select(Turbine).where(Turbine.user_id == user.id).order_by(Turbine.name)
    ).scalars().all()
    sites = {s.id: s for s in db.execute(
        select(Site).where(Site.user_id == user.id)).scalars().all()}
    ranks = {r.turbine_id: r for r in db.execute(
        select(RiskRanking).where(RiskRanking.user_id == user.id)).scalars().all()}
    out = []
    for t in turbines:
        r = ranks.get(t.id)
        out.append({
            "id": t.id, "name": t.name, "oem": t.oem, "model": t.model,
            "external_ref": t.external_ref,
            "site_name": sites[t.site_id].name if t.site_id in sites else None,
            "rated_power_kw": t.rated_power_kw,
            "health_score": round(100 * (1 - (r.fault_probability or 0)), 0) if r else None,
            "risk_class": r.risk_class if r else None,
            "component": r.component if r else None,
            "rul_days": r.rul_days if r else None,
        })
    return out


def _latest_pred(db, user_id, turbine_id):
    from models import MLPrediction
    return db.execute(
        select(MLPrediction).where(MLPrediction.user_id == user_id,
                                   MLPrediction.turbine_id == turbine_id)
        .order_by(MLPrediction.ts.desc(), MLPrediction.id.desc()).limit(1)
    ).scalar()


@router.get("/{turbine_id}/health_summary", response_model=TurbineHealthSummary)
def health_summary(turbine_id: int, db: Session = Depends(get_db),
                   user=Depends(get_current_user)):
    from models import Turbine, Site
    t = db.execute(select(Turbine).where(Turbine.id == turbine_id,
                                         Turbine.user_id == user.id)).scalar()
    if not t:
        raise HTTPException(404, "Turbine not found")
    site = db.execute(select(Site).where(Site.id == t.site_id)).scalar()
    pred = _latest_pred(db, user.id, turbine_id)
    if not pred:
        raise HTTPException(409, "No prediction yet. POST /api/ml/run_inference first.")

    fp = pred.fault_probability or 0.0
    health = fmeca.health_score_from_probability(fp)
    classification = fmeca.classification_from_health(health)

    # raw (pre-calibration) fault prob = 1 − P(normal) from stored class probabilities
    cp = pred.class_probabilities or {}
    p0 = cp.get("0", cp.get(0))
    raw_fp = round(1 - float(p0), 3) if p0 is not None else None
    is_calibrated = raw_fp is not None and abs(raw_fp - fp) > 0.005

    # failure-mode predictions from class probabilities
    fm = []
    for cls_str, p in (pred.class_probabilities or {}).items():
        cls = int(cls_str)
        if cls == 0:
            continue
        comp = fmeca.CLASS_TO_COMPONENT.get(cls, "Unknown")
        fm.append(FailureModePrediction(
            component=comp, predicted_class=cls, probability=round(float(p), 3),
            severity=_severity(float(p)),
        ))
    fm.sort(key=lambda x: x.probability, reverse=True)

    comp_scores = _component_scores(pred)
    comp_health = [
        ComponentHealth(component=c, health_score=s,
                        rul_days=fmeca.rul_from_probability(
                            1 - s / 100.0, c if c in fmeca.DETECTION_LEAD_DAYS else "Gearbox"))
        for c, s in comp_scores.items()
    ]

    # Explainability integrity: tag drivers physical vs cross-component + collinearity
    ann = annotate_shap(pred.shap_top_features or [], pred.predicted_component or "Normal")
    shap = [ShapFeature(**f) for f in ann["features"]]
    explainability = ExplainabilityIntegrity(
        alignment_score=ann["alignment_score"], caveat=ann["caveat"],
        unexpected_top=ann["unexpected_top"], verdict=ann["verdict"],
    )

    # Traceable maintenance decision-support recommendation (edge-case aware)
    from services.recommendations import build_recommendation
    from models.schemas import MaintenanceRecommendation
    comp = pred.predicted_component or "Hydraulic"
    cost = fmeca.consequence_cost(comp, rated_power_kw=(t.rated_power_kw or 2000.0))
    eel = round(fp * cost["total"])
    rec = build_recommendation(fp, pred.rul_days, comp, cost["severity"], eel, ann)
    recommendation = MaintenanceRecommendation(**rec)

    return TurbineHealthSummary(
        turbine_id=t.id, turbine_name=t.name, oem=t.oem, model=t.model,
        site_name=site.name if site else "—",
        overall_health_score=health, classification=classification,
        fault_probability=round(fp, 3), raw_fault_probability=raw_fp, calibrated=is_calibrated,
        predicted_component=pred.predicted_component,
        confidence=round(pred.confidence or 0, 3), rul_days=pred.rul_days,
        failure_mode_predictions=fm, component_health=comp_health,
        shap_explanation=shap, explainability=explainability, recommendation=recommendation,
        narrative=_narrative(t, pred, health, classification),
        model_version=pred.model_version, data_source=_data_source(db, user),
        last_prediction_at=pred.ts,
    )


@router.get("/{turbine_id}/predictions")
def turbine_predictions(turbine_id: int, limit: int = Query(50, ge=1, le=500),
                        db: Session = Depends(get_db), user=Depends(get_current_user)):
    from models import MLPrediction
    rows = db.execute(
        select(MLPrediction).where(MLPrediction.user_id == user.id,
                                   MLPrediction.turbine_id == turbine_id)
        .order_by(MLPrediction.ts.desc(), MLPrediction.id.desc()).limit(limit)
    ).scalars().all()
    return [{
        "ts": p.ts, "fault_probability": p.fault_probability,
        "predicted_class": p.predicted_class, "predicted_component": p.predicted_component,
        "class_probabilities": p.class_probabilities, "rul_days": p.rul_days,
        "confidence": p.confidence, "shap_top_features": p.shap_top_features,
        "model_version": p.model_version,
    } for p in rows]


@router.get("/{turbine_id}/rul")
def turbine_rul(turbine_id: int, db: Session = Depends(get_db),
                user=Depends(get_current_user)):
    pred = _latest_pred(db, user.id, turbine_id)
    if not pred:
        raise HTTPException(409, "No prediction yet.")
    comp_scores = _component_scores(pred)
    return {
        "turbine_id": turbine_id,
        "overall_rul_days": pred.rul_days,
        "predicted_component": pred.predicted_component,
        "component_rul_days": {
            c: fmeca.rul_from_probability(1 - s / 100.0,
                                          c if c in fmeca.DETECTION_LEAD_DAYS else "Gearbox")
            for c, s in comp_scores.items()
        },
        "model_version": pred.model_version,
    }


@router.get("/{turbine_id}/timeseries", response_model=TimeseriesResponse,
            tags=["screen 3 · scada explorer"])
def turbine_timeseries(
    turbine_id: int,
    from_ts: datetime | None = Query(None, alias="from"),
    to_ts: datetime | None = Query(None, alias="to"),
    signals: str = Query("power_kw,wind_speed,gearbox_oil_temp,rotor_rpm"),
    limit: int = Query(2000, ge=1, le=20000),
    db: Session = Depends(get_db), user=Depends(get_current_user),
):
    from models import Turbine, ScadaTimeseries, Event
    t = db.execute(select(Turbine).where(Turbine.id == turbine_id,
                                         Turbine.user_id == user.id)).scalar()
    if not t:
        raise HTTPException(404, "Turbine not found")

    wanted = [s.strip() for s in signals.split(",") if s.strip()]
    hot = {"power_kw", "wind_speed", "rotor_rpm", "gearbox_oil_temp",
           "gen_bearing_de_temp", "nacelle_temp", "ambient_temp"}

    q = select(ScadaTimeseries).where(ScadaTimeseries.turbine_id == turbine_id)
    if from_ts:
        q = q.where(ScadaTimeseries.ts >= from_ts)
    if to_ts:
        q = q.where(ScadaTimeseries.ts <= to_ts)
    q = q.order_by(ScadaTimeseries.ts.desc()).limit(limit)
    rows = list(reversed(db.execute(q).scalars().all()))

    # anomaly windows for this turbine
    events = db.execute(
        select(Event).where(Event.turbine_id == turbine_id,
                            Event.label == "anomaly")
    ).scalars().all()
    markers = [AnomalyMarker(start=e.start_time, end=e.end_time,
                             label=e.description or "anomaly", component=e.component,
                             source="actual_event") for e in events if e.start_time]

    def in_anomaly(ts):
        return any(e.start_time and e.end_time and e.start_time <= ts <= e.end_time
                   for e in events)

    points = []
    for r in rows:
        vals = {}
        for s in wanted:
            if s in hot:
                vals[s] = getattr(r, s, None)
            elif r.signals and s in r.signals:
                vals[s] = r.signals[s]
            else:
                vals[s] = None
        points.append(TimeseriesPoint(ts=r.ts, values=vals,
                                      status_type_id=r.status_type_id,
                                      is_anomaly=in_anomaly(r.ts)))

    return TimeseriesResponse(
        turbine_id=turbine_id, turbine_name=t.name, signals=wanted,
        points=points, anomaly_markers=markers, resolution_minutes=10,
        data_source=_data_source(db, user),
    )


def _severity(p: float) -> str:
    if p >= 0.6:
        return "Critical"
    if p >= 0.4:
        return "High"
    if p >= 0.2:
        return "Medium"
    return "Low"
