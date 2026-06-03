"""
═══════════════════════════════════════════════════════════════════════
 MODEL EVALUATION
═══════════════════════════════════════════════════════════════════════
Metric helpers used during training and for monitoring model drift.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)


def evaluate_classifier(pipe, X_test, y_test) -> dict:
    """Return a compact, JSON-serialisable metrics dict."""
    y_pred = pipe.predict(X_test)
    try:
        y_proba = pipe.predict_proba(X_test)
    except Exception:
        y_proba = None

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "f1_macro": round(float(f1_score(y_test, y_pred, average="macro")), 4),
        "precision_macro": round(float(precision_score(y_test, y_pred, average="macro", zero_division=0)), 4),
        "recall_macro": round(float(recall_score(y_test, y_pred, average="macro", zero_division=0)), 4),
    }
    if y_proba is not None:
        try:
            metrics["roc_auc_macro_ovr"] = round(
                float(roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")), 4
            )
        except Exception:
            metrics["roc_auc_macro_ovr"] = None
    return metrics


def population_stability_index(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """
    PSI for drift monitoring: compare a feature's training distribution
    (expected) to live data (actual). PSI > 0.2 → significant drift.
    """
    breakpoints = np.quantile(expected, np.linspace(0, 1, bins + 1))
    breakpoints[0], breakpoints[-1] = -np.inf, np.inf
    e_perc = np.histogram(expected, breakpoints)[0] / max(len(expected), 1)
    a_perc = np.histogram(actual, breakpoints)[0] / max(len(actual), 1)
    e_perc = np.clip(e_perc, 1e-4, None)
    a_perc = np.clip(a_perc, 1e-4, None)
    return float(np.sum((a_perc - e_perc) * np.log(a_perc / e_perc)))
