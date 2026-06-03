"""
═══════════════════════════════════════════════════════════════════════
 INFERENCE ENGINE
═══════════════════════════════════════════════════════════════════════
Loads the active model bundle, scores turbines from SCADA data stored in
the database, computes SHAP explanations, writes ml_predictions, and
triggers the fleet risk-ranking recompute.

Used by:  POST /api/ml/run_inference   and   the seed script.
"""
from __future__ import annotations

import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from config import settings
from .feature_engineering import engineer_features
from .fmeca import (
    CLASS_TO_COMPONENT, rul_from_probability, health_score_from_probability,
)

_BUNDLE_CACHE: dict | None = None


def load_model() -> dict:
    """Load (and cache) the active model bundle from disk."""
    global _BUNDLE_CACHE
    if _BUNDLE_CACHE is not None:
        return _BUNDLE_CACHE
    path = settings.model_path / "latest.joblib"
    if not path.exists():
        raise FileNotFoundError(
            f"No trained model found at {path}. Run (from backend/):  python -m ml.train_model"
        )
    _BUNDLE_CACHE = joblib.load(path)
    return _BUNDLE_CACHE


def reset_model_cache():
    global _BUNDLE_CACHE
    _BUNDLE_CACHE = None
    try:
        from .calibrate import reset_calibrator_cache
        reset_calibrator_cache()
    except Exception:
        pass


def _window_to_features(rows: list, features: list[str]) -> pd.DataFrame:
    """Rebuild a canonical feature frame from stored SCADA signal JSON."""
    records = []
    for r in rows:
        rec = dict(r.signals or {})
        # ensure hot signals present under canonical names
        rec.setdefault("Wind_speed_m/s_avg", r.wind_speed)
        rec.setdefault("Grid_power_kW_avg", r.power_kw)
        rec.setdefault("Rot_rpm_rpm_avg", r.rotor_rpm)
        rec.setdefault("Temp_oil_gearbox_C_avg", r.gearbox_oil_temp)
        rec.setdefault("Temp_gen_bearing_Drive_End_C_avg", r.gen_bearing_de_temp)
        rec.setdefault("Nac_temperature_C_avg", r.nacelle_temp)
        rec.setdefault("Amb_temp_C_avg", r.ambient_temp)
        records.append(rec)
    df = pd.DataFrame.from_records(records)
    df = engineer_features(df)
    # guarantee every model feature exists
    for f in features:
        if f not in df.columns:
            df[f] = np.nan
    return df[features]


def _shap_top(bundle, X_repr: pd.DataFrame, target_class: int, k: int = 5) -> list[dict]:
    """Top-k SHAP contributions for a single representative row."""
    features = bundle["features"]
    pipe = bundle["pipeline"]
    try:
        import shap
        model = pipe.named_steps["clf"]
        X_proc = pipe[:-1].transform(X_repr)
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_proc)
        arr = np.array(sv)
        if arr.ndim == 3:                       # (n, features, classes)
            vals = arr[0, :, target_class]
        elif isinstance(sv, list):
            vals = sv[target_class][0]
        else:
            vals = arr[0]
        order = np.argsort(np.abs(vals))[::-1][:k]
        total = float(np.sum(np.abs(vals))) or 1.0
        return [
            {"feature": features[i], "value": float(X_repr.iloc[0, i]),
             "contribution": round(float(vals[i]), 4),
             "pct": round(100 * abs(float(vals[i])) / total, 1)}
            for i in order
        ]
    except Exception as e:
        # fallback: global feature importance
        try:
            imp = pipe.named_steps["clf"].feature_importances_
            order = np.argsort(imp)[::-1][:k]
            total = float(np.sum(imp)) or 1.0
            return [
                {"feature": features[i], "value": float(X_repr.iloc[0, i]),
                 "contribution": round(float(imp[i]), 4),
                 "pct": round(100 * float(imp[i]) / total, 1)}
                for i in order
            ]
        except Exception:
            return []


def score_turbine(bundle: dict, rows: list) -> dict | None:
    """Score one turbine's recent SCADA window → aggregated prediction."""
    if not rows:
        return None
    features = bundle["features"]
    pipe = bundle["pipeline"]
    X = _window_to_features(rows, features)
    if X.empty:
        return None

    proba = pipe.predict_proba(X)                  # (n, n_classes)
    classes = list(pipe.named_steps["clf"].classes_)
    mean_proba = proba.mean(axis=0)                # window-averaged
    # raw fault probability = 1 − P(normal class 0)
    normal_idx = classes.index(0) if 0 in classes else None
    raw_fault_prob = float(1.0 - mean_proba[normal_idx]) if normal_idx is not None else float(mean_proba.max())

    # CALIBRATION: map the over-confident raw probability to a calibrated one
    # BEFORE it feeds health score, RUL and the Expected-Loss risk ranking.
    from .calibrate import apply_calibration
    fault_prob, calibrated = apply_calibration(raw_fault_prob, bundle["version"])

    # Dominant fault mode = argmax over non-normal classes. We ALWAYS attribute
    # the dominant mode so Expected Loss reflects the real consequence even at
    # moderate (calibrated) probability — the risk queue must still surface the
    # top exposure. Whether the turbine is "in anomaly" is a separate status.
    class_probs = {int(c): round(float(mean_proba[i]), 4) for i, c in enumerate(classes)}
    fault_classes = {c: p for c, p in class_probs.items() if c != 0}
    pred_class = max(fault_classes, key=fault_classes.get) if fault_classes else 0
    component = CLASS_TO_COMPONENT.get(pred_class, "Hydraulic")
    is_anomaly = fault_prob >= 0.4                 # calibrated status threshold

    confidence = float(max(mean_proba))
    rul = rul_from_probability(fault_prob, component)

    # SHAP on the window-mean representative row
    X_repr = X.mean(axis=0).to_frame().T
    target_for_shap = pred_class if pred_class in classes else classes[0]
    shap_top = _shap_top(bundle, X_repr, classes.index(target_for_shap))

    return {
        "fault_probability": round(fault_prob, 4),
        "raw_fault_probability": round(raw_fault_prob, 4),
        "calibrated": calibrated,
        "is_anomaly": is_anomaly,
        "predicted_class": int(pred_class),
        "predicted_component": component,          # dominant fault mode (drives EEL)
        "class_probabilities": class_probs,
        "rul_days": rul,
        "confidence": round(confidence, 4),
        "shap_top_features": shap_top,
        "health_score": health_score_from_probability(fault_prob),
        "ts": rows[-1].ts,
    }


def run_inference(db: Session, user_id: int, turbine_id: int | None = None,
                  window: int = 144, store: bool = True) -> dict:
    """
    Score one turbine or the whole fleet using the latest `window` SCADA
    readings per turbine (144 × 10-min = 24h). Writes predictions + risk.
    """
    from models import Turbine, ScadaTimeseries, MLPrediction
    from services.risk_ranking import recompute_risk_rankings

    t0 = time.time()
    bundle = load_model()
    version = bundle["version"]

    q = select(Turbine).where(Turbine.user_id == user_id)
    if turbine_id:
        q = q.where(Turbine.id == turbine_id)
    turbines = db.execute(q).scalars().all()

    # idempotent: clear prior predictions for the turbines we are about to score
    if store and turbines:
        ids = [t.id for t in turbines]
        db.query(MLPrediction).filter(
            MLPrediction.user_id == user_id, MLPrediction.turbine_id.in_(ids)
        ).delete(synchronize_session=False)
        db.commit()

    written = 0
    for t in turbines:
        rows = db.execute(
            select(ScadaTimeseries)
            .where(ScadaTimeseries.turbine_id == t.id)
            .order_by(ScadaTimeseries.ts.desc())
            .limit(window)
        ).scalars().all()
        rows = list(reversed(rows))               # chronological
        result = score_turbine(bundle, rows)
        if result is None:
            continue
        if store:
            db.add(MLPrediction(
                user_id=user_id, turbine_id=t.id, ts=result["ts"],
                model_version=version,
                fault_probability=result["fault_probability"],
                predicted_class=result["predicted_class"],
                predicted_component=result["predicted_component"],
                class_probabilities=result["class_probabilities"],
                rul_days=result["rul_days"], confidence=result["confidence"],
                shap_top_features=result["shap_top_features"],
            ))
            written += 1
    if store:
        db.commit()

    ranked = recompute_risk_rankings(db, user_id) if store else 0

    return {
        "status": "ok",
        "model_version": version,
        "turbines_scored": len(turbines),
        "predictions_written": written,
        "risk_rankings_written": ranked,
        "elapsed_seconds": round(time.time() - t0, 2),
    }


if __name__ == "__main__":
    from config import SessionLocal
    from models import User
    db = SessionLocal()
    user = db.query(User).first()
    if not user:
        print("No user found. Run the seed script first.")
    else:
        print(run_inference(db, user.id))
    db.close()
