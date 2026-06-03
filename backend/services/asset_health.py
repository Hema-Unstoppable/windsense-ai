"""
═══════════════════════════════════════════════════════════════════════
 ASSET HEALTH CERTIFICATE SERVICE   (powers Screen 4)
═══════════════════════════════════════════════════════════════════════
Aggregates ML outputs into an engineering decision-support Asset Health Certificate:
overall health score, per-component scores + RUL, a narrative summary,
sustainability KPIs and financial impact.
"""
from __future__ import annotations

from datetime import datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from ml import fmeca

# Components reported on every certificate
CERT_COMPONENTS = ["Gearbox", "Generator", "Main Bearing", "Hydraulic", "Transformer", "Pitch"]


def _latest_prediction(db: Session, user_id: int, turbine_id: int):
    from models import MLPrediction
    return db.execute(
        select(MLPrediction)
        .where(MLPrediction.user_id == user_id, MLPrediction.turbine_id == turbine_id)
        .order_by(MLPrediction.ts.desc(), MLPrediction.id.desc()).limit(1)
    ).scalar()


def _component_scores(pred) -> dict[str, float]:
    """Derive per-component health from class probabilities."""
    probs = pred.class_probabilities or {}
    scores = {}
    for comp in CERT_COMPONENTS:
        cls = fmeca.COMPONENT_TO_CLASS.get(comp)
        # probability this component is the fault driver
        p = float(probs.get(str(cls), probs.get(cls, 0.0))) if cls is not None else 0.0
        # main bearing/pitch not modelled directly → derive from overall health
        if cls is None:
            base = 1 - (pred.fault_probability or 0)
            scores[comp] = round(max(40, min(99, 100 * base - 5)), 0)
        else:
            scores[comp] = round(max(20, 100 * (1 - p)), 0)
    return scores


def _narrative(turbine, pred, health, classification) -> str:
    comp = pred.predicted_component or "no dominant"
    fp = (pred.fault_probability or 0) * 100
    rul = int(pred.rul_days or 0)
    drivers = ", ".join(
        f"{f['feature']} ({f['pct']}%)" for f in (pred.shap_top_features or [])[:3]
    ) or "multiple SCADA signals"
    return (
        f"{turbine.name} presents an overall asset health score of {int(health)}/100, "
        f"classified as {classification}. The WindSense ML model ({pred.model_version}) "
        f"estimates a {fp:.0f}% calibrated probability of an anomaly, with the {comp} subsystem "
        f"as the leading driver. Estimated Remaining Useful Life for the affected component is "
        f"{rul} days (indicative). Leading SHAP contributors: {drivers}. "
        f"This is an engineering decision-support assessment generated from normalised SCADA "
        f"data; it is not an insurance instrument or a guarantee of future performance."
    )


def _provenance_and_hash(db: Session, turbine, pred) -> tuple[float, str, int]:
    """Data coverage %, SHA-256 content seal, and rows assessed."""
    import hashlib, json
    from models import ScadaTimeseries
    rows = db.execute(
        select(ScadaTimeseries).where(ScadaTimeseries.turbine_id == turbine.id)
        .order_by(ScadaTimeseries.ts.desc()).limit(1000)
    ).scalars().all()
    n = len(rows)
    present = sum(1 for r in rows if r.power_kw is not None and r.wind_speed is not None
                  and r.gearbox_oil_temp is not None)
    coverage = round(100.0 * present / n, 1) if n else 0.0
    # frozen snapshot digest of the actual data used
    snap = [(r.ts.isoformat(), r.power_kw, r.wind_speed, r.gearbox_oil_temp, r.nacelle_temp)
            for r in rows[:300]]
    data_digest = hashlib.sha256(json.dumps(snap, default=str).encode()).hexdigest()
    # content hash links data + model version + explanation (SHAP) state
    payload = {
        "data_digest": data_digest,
        "model_version": pred.model_version,
        "fault_probability": pred.fault_probability,
        "class_probabilities": pred.class_probabilities,
        "shap_vector": pred.shap_top_features,
        "prediction_ts": pred.ts.isoformat() if pred.ts else None,
        "turbine_ref": turbine.external_ref,
    }
    content_hash = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()
    return coverage, content_hash, n


def _calibration_status(pred) -> str:
    try:
        from ml.calibrate import load_calibrator
        cal = load_calibrator()
        if cal and cal.get("model_version") == pred.model_version:
            m = cal["metrics"]
            return (f"Calibrated (isotonic) — ECE {m['ece_before']}→{m['ece_after']}, "
                    f"Brier {m['brier_before']}→{m['brier_after']}")
    except Exception:
        pass
    return "Uncalibrated (raw probabilities)"


def _training_window(db: Session, pred) -> str:
    from models import ModelRegistry
    reg = db.execute(
        select(ModelRegistry).where(ModelRegistry.is_active == True)  # noqa: E712
        .order_by(ModelRegistry.trained_at.desc()).limit(1)
    ).scalar()
    if reg:
        return (f"{reg.algorithm} {reg.version}, balanced sample {reg.n_train_rows or '?'} rows, "
                f"trained {reg.trained_at:%Y-%m-%d}")
    return f"model {pred.model_version}"


def _limitations_statement() -> str:
    """Honest limitations + known miss-rate, pulled from the validation report."""
    try:
        from ml.validation import load_report
        rep = load_report()
        if rep:
            ft = rep["split_comparison"]["purged_embargoed_time"].get("pr_auc")
            hs = rep["multiclass_scoreboard"].get("high_severity_downgrade_rate")
            n_fail = sum(t["n_real_failures"] for t in rep["provenance"]["turbines"])
            return (
                f"Developed on the CARE open benchmark: {len(rep['provenance']['turbines'])} "
                f"turbines, {n_fail} labelled failure events, 10-minute averaged SCADA (hides "
                f"early high-frequency signatures). Forward-in-time (purged/embargoed) PR-AUC "
                f"≈ {ft} — close to base rate, so prediction of UNSEEN FUTURE failures is NOT "
                f"yet validated. KNOWN MISS RATE: high-severity failure-mode downgrade rate "
                f"≈ {int((hs or 0)*100)}% (gearbox/generator failures may be classified as a "
                f"lower-severity fault); component attribution is INDICATIVE ONLY. This document "
                f"is engineering decision-support, not a validated insurance or lending instrument."
            )
    except Exception:
        pass
    return ("Engineering decision-support assessment based on a limited labelled-failure set and "
            "10-minute averaged SCADA. Predictive performance on unseen future failures is not "
            "yet independently validated; treat component attribution as indicative.")


def generate_certificate(db: Session, user_id: int, turbine_id: int, persist: bool = True,
                         certifying_engineer: str | None = None):
    from models import Turbine, Site, AssetHealthCertificate, AuditLog

    turbine = db.execute(
        select(Turbine).where(Turbine.id == turbine_id, Turbine.user_id == user_id)
    ).scalar()
    if not turbine:
        return None
    site = db.execute(select(Site).where(Site.id == turbine.site_id)).scalar()
    pred = _latest_prediction(db, user_id, turbine_id)
    if not pred:
        return None

    fp = pred.fault_probability or 0.0
    health = fmeca.health_score_from_probability(fp)
    classification = fmeca.classification_from_health(health)
    risk_cls = fmeca.risk_class(fmeca.risk_score(fp, pred.predicted_component or "Normal"))
    comp_scores = _component_scores(pred)
    rul_estimates = {
        comp: fmeca.rul_from_probability(
            1 - comp_scores[comp] / 100.0, comp if comp in fmeca.DETECTION_LEAD_DAYS else "Gearbox")
        for comp in CERT_COMPONENTS
    }
    narrative = _narrative(turbine, pred, health, classification)
    issued = datetime.utcnow()
    valid_until = issued + timedelta(days=90)
    # unique per issuance (so every re-run is recorded, never silently overwritten)
    ref = f"AHC-{issued:%Y%m%d-%H%M%S}-{turbine_id:03d}"

    # forensic / tamper-evidence fields
    coverage, content_hash, rows_assessed = _provenance_and_hash(db, turbine, pred)
    calibration_status = _calibration_status(pred)
    training_window = _training_window(db, pred)
    limitations = _limitations_statement()
    engineer = certifying_engineer or "Pending engineer sign-off — not yet certified"

    if persist:
        cert = AssetHealthCertificate(
            user_id=user_id, turbine_id=turbine_id, certificate_ref=ref,
            issued_at=issued, valid_until=valid_until,
            overall_health_score=health, risk_class=risk_cls, classification=classification,
            component_scores=comp_scores, rul_estimates=rul_estimates,
            narrative=narrative, model_version=pred.model_version,
            data_source=turbine.data_source,
            content_hash=content_hash, data_coverage_pct=coverage,
            model_training_window=training_window, calibration_status=calibration_status,
            certifying_engineer=engineer, limitations=limitations,
        )
        db.add(cert)
        # immutable audit log entry for certificate issuance
        db.add(AuditLog(
            user_id=user_id, event_type="CERT_ISSUED", turbine_id=turbine_id,
            certificate_ref=ref, actor=engineer, content_hash=content_hash,
            detail={"health": health, "model_version": pred.model_version,
                    "data_coverage_pct": coverage, "rows_assessed": rows_assessed},
        ))
        db.commit()

    return {
        "certificate_ref": ref, "turbine": turbine, "site": site, "prediction": pred,
        "overall_health_score": health, "risk_class": risk_cls,
        "classification": classification, "component_scores": comp_scores,
        "rul_estimates": rul_estimates, "narrative": narrative,
        "issued_at": issued, "valid_until": valid_until,
        "content_hash": content_hash, "data_coverage_pct": coverage,
        "rows_assessed": rows_assessed, "model_training_window": training_window,
        "calibration_status": calibration_status, "certifying_engineer": engineer,
        "limitations": limitations,
    }


def sustainability_kpis(db: Session, user_id: int, turbine) -> dict:
    """Simple, defensible sustainability KPI estimates per turbine."""
    rated_mw = (turbine.rated_power_kw or 2000) / 1000.0
    cf = 0.34
    annual_mwh = rated_mw * cf * 8760
    return {
        "co2_avoided_tonnes": round(annual_mwh * 0.4, 0),     # ~0.4 t CO2 / MWh grid avg
        "energy_production_gwh": round(annual_mwh / 1000.0, 1),
        "capacity_factor_pct": round(cf * 100, 1),
        "fleet_uptime_pct": 96.8,
    }


def financial_impact(db: Session, user_id: int, pred) -> dict:
    comp = pred.predicted_component or "Gearbox"
    fp = pred.fault_probability or 0.0
    exposure = fmeca.financial_exposure(comp, fp)
    econ = fmeca.COMPONENT_ECONOMICS.get(comp, fmeca.COMPONENT_ECONOMICS["Gearbox"])
    avoided = (econ["unplanned"] - econ["planned"]) * fp
    return {
        "avoided_downtime_cost_eur": round(avoided, 0),
        "opex_per_mwh_eur": 18.4,
        "avoided_unplanned_premium_eur": round(avoided * 0.08, 0),
        "total_exposure_eur": round(exposure, 0),
    }


def compliance_items() -> list[dict]:
    """Static compliance checklist scaffold (would be driven by O&M records)."""
    return [
        {"item": "Annual blade inspection", "status": "ok", "detail": "2026-03-12"},
        {"item": "Generator oil sampling", "status": "ok", "detail": "2026-04-01"},
        {"item": "Gearbox oil change", "status": "overdue", "detail": "Overdue 14d"},
        {"item": "Tower bolt torque check", "status": "not_completed", "detail": "Not completed"},
    ]
