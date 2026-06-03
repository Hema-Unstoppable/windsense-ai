"""
═══════════════════════════════════════════════════════════════════════
 PROBABILITY CALIBRATION  (production)
═══════════════════════════════════════════════════════════════════════
The fault classifier is trained on a class-balanced sample, so its raw
probabilities are over-confident (we measured ECE ≈ 0.43). Every Expected
Economic Loss € figure multiplies this probability — so we calibrate it
BEFORE it feeds the risk score.

This module fits an isotonic-regression calibrator that maps the raw fault
probability (1 − P(normal)) to a calibrated probability, evaluated on a
leakage-safe (group-by-event) hold-out, then saves it next to the model.
Inference loads and applies it automatically.

Run:  python -m ml.calibrate
"""
from __future__ import annotations

from datetime import datetime

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

from config import settings

_CAL_CACHE: dict | None = None
_CAL_LOADED = False


def _ece(p, y, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    n, ece = len(y), 0.0
    for i in range(bins):
        m = (p >= edges[i]) & (p <= edges[i + 1] if i == bins - 1 else p < edges[i + 1])
        if m.sum():
            ece += (m.sum() / n) * abs(float(p[m].mean()) - float(y[m].mean()))
    return round(float(ece), 4)


def _raw_fault_prob(bundle, X):
    pipe = bundle["pipeline"]
    classes = list(pipe.named_steps["clf"].classes_)
    proba = pipe.predict_proba(X)
    if 0 in classes:
        return 1.0 - proba[:, classes.index(0)]
    return proba.max(axis=1)


def fit_calibrator(per_event_cap: int = 2500) -> dict:
    """Fit + evaluate + save the production isotonic calibrator."""
    from sklearn.model_selection import GroupShuffleSplit
    from .validation import load_eval_frame, _load_bundle

    bundle = _load_bundle()
    features = bundle["features"]
    df, _ = load_eval_frame(per_event_cap=per_event_cap)
    X = df[features].apply(lambda c: c.fillna(c.median()))
    y = df["y_binary"].values
    raw = _raw_fault_prob(bundle, X)

    # leakage-safe evaluation of the calibration gain
    tr, te = next(GroupShuffleSplit(1, test_size=0.4, random_state=11)
                  .split(X, y, df["event_id"].values))
    iso_eval = IsotonicRegression(out_of_bounds="clip")
    iso_eval.fit(raw[tr], y[tr])
    cal_te = iso_eval.predict(raw[te])
    metrics = {
        "brier_before": round(float(brier_score_loss(y[te], raw[te])), 4),
        "brier_after": round(float(brier_score_loss(y[te], cal_te)), 4),
        "ece_before": _ece(raw[te], y[te]),
        "ece_after": _ece(cal_te, y[te]),
        "n_fit_rows": int(len(df)),
    }

    # production calibrator: fit on ALL available data for the richest map
    iso_full = IsotonicRegression(out_of_bounds="clip")
    iso_full.fit(raw, y)

    payload = {"calibrator": iso_full, "model_version": bundle["version"],
               "method": "isotonic", "metrics": metrics,
               "fitted_at": datetime.utcnow().isoformat()}
    out = settings.model_path / "calibrator.joblib"
    joblib.dump(payload, out)
    reset_calibrator_cache()
    print(f"[calibrate] saved -> {out}")
    print(f"[calibrate] Brier {metrics['brier_before']} -> {metrics['brier_after']} | "
          f"ECE {metrics['ece_before']} -> {metrics['ece_after']}")
    return metrics


def load_calibrator() -> dict | None:
    """Load (and cache) the saved calibrator payload, or None if absent."""
    global _CAL_CACHE, _CAL_LOADED
    if _CAL_LOADED:
        return _CAL_CACHE
    _CAL_LOADED = True
    path = settings.model_path / "calibrator.joblib"
    _CAL_CACHE = joblib.load(path) if path.exists() else None
    return _CAL_CACHE


def reset_calibrator_cache():
    global _CAL_CACHE, _CAL_LOADED
    _CAL_CACHE, _CAL_LOADED = None, False


def apply_calibration(raw_fault_prob: float, model_version: str) -> tuple[float, bool]:
    """
    Map a raw fault probability to a calibrated one. Returns (prob, applied).
    Only applies if a calibrator exists for the SAME model version.
    """
    cal = load_calibrator()
    if not cal or cal.get("model_version") != model_version:
        return float(raw_fault_prob), False
    val = float(cal["calibrator"].predict([raw_fault_prob])[0])
    return max(0.0, min(1.0, val)), True


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    fit_calibrator()
