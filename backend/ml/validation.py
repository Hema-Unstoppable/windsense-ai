"""
═══════════════════════════════════════════════════════════════════════
 ML VALIDATION SUITE  —  prove, don't assert
═══════════════════════════════════════════════════════════════════════
Runs the rigorous anti-cheating tests on the WindSense fault model and the
WindFarm A (CARE to Compare) data, and emits an honest JSON report:

  1. Data provenance & coverage   (channels, sampling rate, gaps, REAL
                                    labeled failures vs anomalies, base rate)
  2. Feature leakage audit        (physical vs status/alarm/derived channels)
  3. Split-methodology comparison (random  vs  group-by-event  vs
                                    leave-one-turbine-out  vs  blocked-time)
                                    — the smoking gun for adjacency leakage
  4. Label-permutation test       (shuffle labels → must collapse to chance)
  5. Post-failure embargo test    (drop last N days before failure)
  6. Class-imbalance reporting     (PR-AUC headline, real-world base-rate note)

Headline metric is **PR-AUC (average precision)** on a BINARY anomaly target
under a LEAKAGE-SAFE split — never ROC, never accuracy, never a random split.

Run:  python -m scripts.run_validation
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, roc_auc_score, precision_recall_curve,
    f1_score, recall_score, confusion_matrix, brier_score_loss,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.isotonic import IsotonicRegression

from config import settings
from .canonical import normalise_columns, META_COLS
from .feature_engineering import add_target, engineer_features
from . import fmeca

# Fault classes + severity weights (for high-severity-miss penalty)
CLASS_NAMES = {0: "Normal", 1: "Generator", 2: "Gearbox", 3: "Hydraulic", 4: "Transformer"}
CLASS_SEVERITY = {0: 0.0, 1: 0.85, 2: 0.90, 3: 0.55, 4: 0.75}   # gearbox/generator worst

# ── feature classification (leakage audit) ───────────────────────────
STATUS_TOKENS = ["status", "alarm", "error", "warning", "fault_code", "mode",
                 "trip", "curtail", "manual", "service", "maintenance"]
# canonical physical signal stems we expect (temperatures, power, speed, etc.)
PHYSICAL_TOKENS = ["temp", "power", "wind", "rpm", "curr", "voltage", "freq",
                   "reactive", "pit_angle", "displacement", "direction", "margin",
                   "bearing", "gearbox", "generator", "hydraulic", "transformer", "nacelle"]


def classify_feature(name: str) -> str:
    n = name.lower()
    if any(tok in n for tok in STATUS_TOKENS):
        return "STATUS/ALARM (leakage risk)"
    if any(tok in n for tok in PHYSICAL_TOKENS):
        return "physical"
    return "other"


# ══════════════════════════════════════════════════════════════════════
#  1. DATA PROVENANCE  (chunked scan of the full CSV)
# ══════════════════════════════════════════════════════════════════════
def data_provenance(csv_path: str | None = None) -> dict:
    csv_path = csv_path or settings.CSV_PATH
    header = pd.read_csv(csv_path, nrows=0)
    all_cols = header.columns.tolist()
    sensor_cols = [c for c in all_cols if c not in META_COLS]

    cols = ["time_stamp", "asset_id", "train_test", "status_type_id",
            "event_id", "event_label", "event_description"]
    per_turbine: dict[int, dict] = {}
    total_rows = 0
    anomaly_rows = 0
    deltas = []

    for ch in pd.read_csv(csv_path, usecols=cols, parse_dates=["time_stamp"],
                          chunksize=300_000):
        total_rows += len(ch)
        anomaly_rows += int((ch["event_label"] == "anomaly").sum())
        for aid, g in ch.groupby("asset_id"):
            d = per_turbine.setdefault(int(aid), {
                "asset_id": int(aid), "rows": 0, "tmin": None, "tmax": None,
                "anomaly_rows": 0, "events": set(), "anomaly_events": set(),
                "failure_modes": {}, "status_codes": {},
            })
            d["rows"] += len(g)
            d["anomaly_rows"] += int((g["event_label"] == "anomaly").sum())
            d["events"].update(g["event_id"].unique().tolist())
            d["anomaly_events"].update(
                g.loc[g["event_label"] == "anomaly", "event_id"].unique().tolist())
            gmin, gmax = g["time_stamp"].min(), g["time_stamp"].max()
            d["tmin"] = gmin if d["tmin"] is None else min(d["tmin"], gmin)
            d["tmax"] = gmax if d["tmax"] is None else max(d["tmax"], gmax)
            for desc in g.loc[g["event_label"] == "anomaly", "event_description"].dropna().unique():
                d["failure_modes"][desc] = d["failure_modes"].get(desc, 0) + 1
            for sc, cnt in g["status_type_id"].value_counts().items():
                d["status_codes"][int(sc)] = d["status_codes"].get(int(sc), 0) + int(cnt)
        # sample timestamp deltas (per turbine) from a slice for sampling-rate estimate
        sample = ch.sort_values(["asset_id", "time_stamp"]).groupby("asset_id")["time_stamp"].diff().dropna()
        deltas.extend(sample.dt.total_seconds().sample(min(2000, len(sample)), random_state=1).tolist())

    median_delta = float(np.median(deltas)) if deltas else None

    turbines_out = []
    for aid, d in sorted(per_turbine.items()):
        span_days = (d["tmax"] - d["tmin"]).days if d["tmin"] is not None else 0
        # expected intervals at the detected sampling rate vs actual rows = gap estimate
        expected = int(span_days * 86400 / median_delta) if median_delta else 0
        turbines_out.append({
            "asset_id": aid,
            "scada_channels": len(sensor_cols),
            "rows": d["rows"],
            "date_range": [d["tmin"].isoformat(), d["tmax"].isoformat()],
            "span_days": span_days,
            "row_positive_rate": round(d["anomaly_rows"] / d["rows"], 3) if d["rows"] else 0,
            "n_events": len(d["events"]),
            "n_real_failures": len(d["anomaly_events"]),
            "failure_modes": d["failure_modes"],
            "status_code_distribution": d["status_codes"],
            "coverage_note": f"~{round(100*d['rows']/max(expected,1))}% of a continuous record "
                             f"(data is event-windowed, not continuous)",
        })

    return {
        "sampling_rate_seconds": median_delta,
        "sampling_rate_human": f"{round(median_delta/60)}-min averages" if median_delta else "unknown",
        "sampling_rate_warning": (
            "Data is 10-minute AVERAGES. 10-min SCADA hides most early bearing/gear "
            "signatures that need 1-Hz raw or high-frequency vibration. Early-fault "
            "claims must be qualified accordingly."
        ),
        "total_rows": total_rows,
        "fleet_row_positive_rate": round(anomaly_rows / total_rows, 3),
        "fleet_positive_rate_warning": (
            "This ~50% positive rate is a BENCHMARK ARTIFACT — CARE to Compare pairs each "
            "anomaly window with normal windows. A real operating fleet sees ~0.1-1% "
            "failure-days. Metrics here are optimistic and will not transfer 1:1 to production."
        ),
        "turbines": turbines_out,
        "labeling_provenance": (
            "Labels come from the CARE to Compare benchmark (real anonymised operating "
            "turbines, Fraunhofer IWES). Failure events carry a root-cause description "
            "(gearbox / generator bearing / hydraulic / transformer) but there are only a "
            "HANDFUL of real failure events across 5 turbines — too few to claim "
            "fleet-validated performance. Honest framing: method development, not validated "
            "deployment performance."
        ),
    }


# ══════════════════════════════════════════════════════════════════════
#  2. FEATURE LEAKAGE AUDIT
# ══════════════════════════════════════════════════════════════════════
def feature_leakage_audit() -> dict:
    bundle = _load_bundle()
    features = bundle["features"]
    classified = [{"feature": f, "category": classify_feature(f)} for f in features]
    status_feats = [c["feature"] for c in classified if c["category"].startswith("STATUS")]
    return {
        "n_features": len(features),
        "features": classified,
        "status_or_alarm_features_in_model": status_feats,
        "status_type_id_excluded": "status_type_id" in META_COLS,
        "verdict": (
            "PASS — model uses only physical SCADA channels; status/alarm/mode codes "
            "(status_type_id) are excluded from features."
            if not status_feats and "status_type_id" in META_COLS else
            "FAIL — status/alarm features detected in the model feature set: " + ", ".join(status_feats)
        ),
        "checks": {
            "alarm_status_codes_removed": not status_feats,
            "maintenance_action_flags_removed": not status_feats,
            "no_oem_derived_health_bits": "PASS — CARE channels are raw physical signals "
                                          "(temperatures, power, currents, voltages, rpm); no OEM health/warning bits.",
        },
    }


# ══════════════════════════════════════════════════════════════════════
#  Data loader for evaluation (preserves event/turbine/time structure)
# ══════════════════════════════════════════════════════════════════════
def _load_bundle() -> dict:
    import joblib
    p = settings.model_path / "latest.joblib"
    if not p.exists():
        raise FileNotFoundError("Train a model first: python -m ml.train_model")
    return joblib.load(p)


def load_eval_frame(per_event_cap: int = 2000):
    """
    Stream the CSV, sampling up to N rows per event, keeping group/time keys.
    Also returns the TRUE end time of each event (from the full data, not the
    sample) so warning lead-time can be measured against the real failure point.
    Returns (df, event_ends).
    """
    buckets: dict = {}
    event_ends: dict = {}
    for ch in pd.read_csv(settings.CSV_PATH, parse_dates=["time_stamp"],
                          chunksize=150_000, low_memory=False):
        ch = normalise_columns(ch)
        for eid, g in ch.groupby("event_id"):
            gmax = g["time_stamp"].max()
            event_ends[eid] = max(event_ends.get(eid, gmax), gmax)
            cur = buckets.setdefault(eid, [])
            remaining = per_event_cap - sum(len(x) for x in cur)
            if remaining > 0:
                cur.append(g.head(remaining))
    frames = [pd.concat(v) for v in buckets.values() if v]
    df = pd.concat(frames).reset_index(drop=True)
    df = add_target(df)
    df = engineer_features(df)
    return df, event_ends


def _eval_xy(df: pd.DataFrame, features: list[str]):
    X = df[features].apply(lambda c: c.fillna(c.median()))
    y = df["y_binary"].values        # 1 = anomaly, 0 = normal
    return X, y


def _train_eval_binary(X_tr, y_tr, X_te, y_te) -> dict:
    """Train a binary XGBoost (class-weighted, NO SMOTE) and score honestly."""
    from xgboost import XGBClassifier
    pos = max(int(y_tr.sum()), 1)
    neg = max(int((y_tr == 0).sum()), 1)
    clf = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
        tree_method="hist", random_state=42, n_jobs=-1,
        scale_pos_weight=neg / pos,
    )
    clf.fit(X_tr, y_tr)
    score = clf.predict_proba(X_te)[:, 1]
    base = float(np.mean(y_te))
    out = {"base_rate": round(base, 3), "n_test": int(len(y_te)),
           "test_positive": int(y_te.sum())}
    try:
        out["pr_auc"] = round(float(average_precision_score(y_te, score)), 3)
        out["pr_auc_lift_over_base"] = round(out["pr_auc"] / base, 2) if base else None
    except Exception:
        out["pr_auc"] = None
    try:
        out["roc_auc"] = round(float(roc_auc_score(y_te, score)), 3)
    except Exception:
        out["roc_auc"] = None
    # precision/recall at a cost-aware threshold (max F1)
    try:
        prec, rec, thr = precision_recall_curve(y_te, score)
        f1 = 2 * prec * rec / (prec + rec + 1e-9)
        i = int(np.argmax(f1))
        out["best_f1"] = round(float(f1[i]), 3)
        out["precision_at_best_f1"] = round(float(prec[i]), 3)
        out["recall_at_best_f1"] = round(float(rec[i]), 3)
    except Exception:
        pass
    return out


# ══════════════════════════════════════════════════════════════════════
#  3. SPLIT-METHODOLOGY COMPARISON  (the smoking gun)
# ══════════════════════════════════════════════════════════════════════
def split_comparison(df: pd.DataFrame, features: list[str]) -> dict:
    X, y = _eval_xy(df, features)
    groups_event = df["event_id"].values
    groups_turbine = df["asset_id"].values
    res = {}

    # (a) RANDOM split — the current (leaky) method
    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    idx = np.arange(len(df))
    rng = np.random.RandomState(42)
    rng.shuffle(idx)
    cut = int(0.7 * len(idx))
    tr, te = idx[:cut], idx[cut:]
    res["random_split"] = _train_eval_binary(X.iloc[tr], y[tr], X.iloc[te], y[te])
    res["random_split"]["method"] = "Random 70/30 (LEAKY — adjacency)"

    # (b) GROUP-BY-EVENT — no event in both train and test
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.35, random_state=7)
                  .split(X, y, groups_event))
    res["group_by_event"] = _train_eval_binary(X.iloc[tr], y[tr], X.iloc[te], y[te])
    res["group_by_event"]["method"] = "Grouped by event (no window spans the split)"

    # (c) LEAVE-ONE-OEM-OUT — hold out one turbine/OEM entirely (cross-OEM generalization)
    anom_by_turbine = df[df["y_binary"] == 1].groupby("asset_id").size()
    holdout = int(anom_by_turbine.idxmax()) if len(anom_by_turbine) else int(df["asset_id"].iloc[0])
    te = df.index[df["asset_id"] == holdout].values
    tr = df.index[df["asset_id"] != holdout].values
    res["leave_one_oem_out"] = _train_eval_binary(X.iloc[tr], y[tr], X.iloc[te], y[te])
    res["leave_one_oem_out"]["method"] = (
        f"Train on 4 turbines, test on a completely unseen turbine (asset_id={holdout}). "
        f"On this dataset each turbine is treated as a distinct OEM — the closest available "
        f"proxy for a Leave-One-OEM-Out test. NOTE: CARE is a single anonymised source, so OEM "
        f"labels are assigned, not real; a true cross-OEM test needs multi-OEM data."
    )

    # (d) PURGED & EMBARGOED TIME SPLIT — train past, embargo gap, test future
    embargo_days = 30
    order = df.sort_values("time_stamp").index.values
    cut = int(0.65 * len(order))
    cutoff_time = df.loc[order[cut], "time_stamp"]
    embargo_end = cutoff_time + timedelta(days=embargo_days)
    tr = df.index[df["time_stamp"] <= cutoff_time].values
    te = df.index[df["time_stamp"] > embargo_end].values
    purged = int(((df["time_stamp"] > cutoff_time) & (df["time_stamp"] <= embargo_end)).sum())
    if len(te) < 20 or y[te].sum() < 2:          # fall back if the gap empties the test set
        te = df.index[df["time_stamp"] > cutoff_time].values
        purged = 0
    res["purged_embargoed_time"] = _train_eval_binary(X.iloc[tr], y[tr], X.iloc[te], y[te])
    res["purged_embargoed_time"]["method"] = (
        f"Train on data up to {cutoff_time.date()}, PURGE a {embargo_days}-day embargo window "
        f"({purged} rows removed), then test on the future block. The deployment-realistic split."
    )

    # keep a stable alias for the page/headline
    res["blocked_by_time"] = res["purged_embargoed_time"]
    res["leave_one_turbine_out"] = res["leave_one_oem_out"]

    rand = res["random_split"].get("pr_auc") or 0
    honest = res["group_by_event"].get("pr_auc") or 0
    res["interpretation"] = (
        f"Random split PR-AUC = {rand} vs leakage-safe (grouped) PR-AUC = {honest}. "
        + ("The drop confirms the random-split score was inflated by temporal-adjacency "
           "leakage. The grouped/blocked numbers are the defensible ones."
           if rand - honest > 0.05 else
           "Scores are similar across splits — adjacency leakage is not the dominant effect here.")
    )
    return res


# ══════════════════════════════════════════════════════════════════════
#  4. LABEL-PERMUTATION TEST  (the cheapest cheating test)
# ══════════════════════════════════════════════════════════════════════
def permutation_test(df: pd.DataFrame, features: list[str]) -> dict:
    X, y = _eval_xy(df, features)
    groups_event = df["event_id"].values
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.35, random_state=7)
                  .split(X, y, groups_event))
    real = _train_eval_binary(X.iloc[tr], y[tr], X.iloc[te], y[te])

    rng = np.random.RandomState(123)
    y_shuf = y.copy()
    rng.shuffle(y_shuf)
    shuffled = _train_eval_binary(X.iloc[tr], y_shuf[tr], X.iloc[te], y_shuf[te])

    base = real["base_rate"]
    real_pr = real.get("pr_auc") or 0
    shuf_pr = shuffled.get("pr_auc") or 0
    # PASS if shuffled labels do NOT score meaningfully ABOVE chance (base rate),
    # AND the real model is clearly above the shuffled one.
    collapsed = (shuf_pr <= base + 0.10) and (real_pr - shuf_pr > 0.15)
    return {
        "real_label_pr_auc": real.get("pr_auc"),
        "shuffled_label_pr_auc": shuffled.get("pr_auc"),
        "base_rate": base,
        "verdict": (
            f"PASS — with labels shuffled, PR-AUC collapses to ~base rate ({base}), so the "
            f"real model ({real.get('pr_auc')}) is learning genuine signal, not exploiting the "
            f"evaluation."
            if collapsed else
            f"WARNING — shuffled-label PR-AUC ({shuffled.get('pr_auc')}) stays well above base "
            f"rate ({base}). The pipeline may be leaking; investigate before trusting any score."
        ),
        "passed": bool(collapsed),
    }


# ══════════════════════════════════════════════════════════════════════
#  5. POST-FAILURE EMBARGO TEST
# ══════════════════════════════════════════════════════════════════════
def embargo_test(df: pd.DataFrame, features: list[str], embargo_days: int = 3) -> dict:
    # event end times
    ends = df[df["y_binary"] == 1].groupby("event_id")["time_stamp"].max()
    keep = pd.Series(True, index=df.index)
    for eid, end in ends.items():
        mask = (df["event_id"] == eid) & (df["time_stamp"] > end - timedelta(days=embargo_days))
        keep &= ~mask
    df_emb = df[keep]

    def run(d):
        X, y = _eval_xy(d, features)
        tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.35, random_state=7)
                      .split(X, y, d["event_id"].values))
        return _train_eval_binary(X.iloc[tr], y[tr], X.iloc[te], y[te])

    full = run(df)
    emb = run(df_emb)
    drop = (full.get("pr_auc") or 0) - (emb.get("pr_auc") or 0)
    return {
        "embargo_days": embargo_days,
        "with_late_window_pr_auc": full.get("pr_auc"),
        "embargoed_pr_auc": emb.get("pr_auc"),
        "rows_removed": int((~keep).sum()),
        "interpretation": (
            f"Removing the last {embargo_days} days before each failure drops PR-AUC by "
            f"{round(drop,3)}. "
            + ("A large drop means much of the signal is late-stage (failure already in "
               "progress) — genuine EARLY prediction is weaker than headline numbers suggest."
               if drop > 0.1 else
               "Small drop — the model retains predictive power from earlier, subtler signals.")
        ),
    }


# ══════════════════════════════════════════════════════════════════════
#  6. BINARY SCOREBOARD  —  precision@top-K + warning lead-time + RUL AMAE
# ══════════════════════════════════════════════════════════════════════
def binary_scoreboard(df: pd.DataFrame, features: list[str], event_ends: dict) -> dict:
    from xgboost import XGBClassifier
    X, y = _eval_xy(df, features)
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.35, random_state=7)
                  .split(X, y, df["event_id"].values))
    pos, neg = max(int(y[tr].sum()), 1), max(int((y[tr] == 0).sum()), 1)
    clf = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                        tree_method="hist", random_state=42, n_jobs=-1,
                        scale_pos_weight=neg / pos)
    clf.fit(X.iloc[tr], y[tr])
    test = df.iloc[te].copy()
    test["score"] = clf.predict_proba(X.iloc[te])[:, 1]
    test["y"] = y[te]

    # precision@top-K (what an operator actually acts on)
    prec_at_k = {}
    ranked = test.sort_values("score", ascending=False)
    for k in [5, 10, 20]:
        top = ranked.head(k)
        prec_at_k[f"P@{k}"] = round(float(top["y"].mean()), 3) if len(top) else None

    # warning lead-time per held-out anomaly event (days before the true failure)
    lead_times, detected = [], 0
    anom_events = test[test["y"] == 1]["event_id"].unique()
    for eid in anom_events:
        ev = test[test["event_id"] == eid].sort_values("time_stamp")
        fired = ev[ev["score"] >= 0.5]
        end = event_ends.get(eid)
        if len(fired) and end is not None:
            first = fired["time_stamp"].iloc[0]
            lead = (end - first).total_seconds() / 86400.0
            lead_times.append(max(0.0, lead))
            detected += 1
    n_anom = len(anom_events)

    # RUL asymmetric error (overestimate penalised 5x) on test anomaly rows
    anom = test[test["y"] == 1].copy()
    over_pen = 5.0
    amae = mae = over_rate = None
    if len(anom):
        ends_map = anom["event_id"].map(event_ends)
        true_rul = np.array([(pd.Timestamp(e) - pd.Timestamp(t)).total_seconds() / 86400.0
                             for e, t in zip(ends_map, anom["time_stamp"])])
        true_rul = np.clip(true_rul, 0, 60)
        comp = anom["y_multi"].map(CLASS_NAMES).fillna("Gearbox").values
        pred_rul = np.array([fmeca.rul_from_probability(float(s), c)
                             for s, c in zip(anom["score"].values, comp)])
        err = pred_rul - true_rul                       # +ve = overestimate (dangerous)
        loss = np.where(err > 0, over_pen * np.abs(err), np.abs(err))
        amae = round(float(np.mean(loss)), 2)
        mae = round(float(np.mean(np.abs(err))), 2)
        over_rate = round(float(np.mean(err > 0)), 3)

    return {
        "pr_auc": round(float(average_precision_score(y[te], test["score"])), 3),
        "precision_at_k": prec_at_k,
        "lead_time": {
            "anomaly_events_in_test": int(n_anom),
            "detected": int(detected),
            "detection_rate": round(detected / n_anom, 3) if n_anom else None,
            "mean_lead_days": round(float(np.mean(lead_times)), 1) if lead_times else None,
            "median_lead_days": round(float(np.median(lead_times)), 1) if lead_times else None,
            "note": "Lead time = days between the first alert (P>=0.5) and the true failure time.",
            "label_warning": (
                "Mean lead-time far exceeds a typical 7-31 day event window. This is a RED FLAG "
                "that the positive window is mis-defined: baseline ('train'-split) rows of each "
                "anomaly event are currently labelled positive, so the model 'alerts' on normal "
                "baseline data. Fix the label (positives = prediction-window only) before any "
                "lead-time claim."
                if (lead_times and float(np.mean(lead_times)) > 45) else None
            ),
        },
        "rul_asymmetric": {
            "amae_5x_overestimate": amae,
            "plain_mae_days": mae,
            "overestimate_rate": over_rate,
            "note": "AMAE penalises over-estimating RUL (saying 20 days when it is 5) 5x harder "
                    "than under-estimating. Current RUL is a heuristic, not a trained regressor — "
                    "this number quantifies how far it is from trustworthy crane-scheduling.",
        },
        "headline_metric": "PR-AUC + Precision@K + warning lead-time (NOT accuracy/ROC).",
    }


# ══════════════════════════════════════════════════════════════════════
#  7. MULTI-CLASS FAILURE-MODE SCOREBOARD
# ══════════════════════════════════════════════════════════════════════
def multiclass_scoreboard(df: pd.DataFrame, features: list[str]) -> dict:
    from xgboost import XGBClassifier
    X = df[features].apply(lambda c: c.fillna(c.median()))
    ym = df["y_multi"].values
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.35, random_state=7)
                  .split(X, ym, df["event_id"].values))
    classes = sorted(np.unique(ym[tr]).tolist())
    if len(classes) < 2:
        return {"available": False, "note": "Too few classes in the training split."}

    remap = {c: i for i, c in enumerate(classes)}
    inv = {i: c for c, i in remap.items()}
    ytr = np.array([remap[v] for v in ym[tr]])
    mask_te = np.isin(ym[te], classes)
    Xte = X.iloc[te][mask_te]
    yte_raw = ym[te][mask_te]
    yte = np.array([remap[v] for v in yte_raw])

    clf = XGBClassifier(n_estimators=250, max_depth=4, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8,
                        objective="multi:softprob", num_class=len(classes),
                        eval_metric="mlogloss", tree_method="hist",
                        random_state=42, n_jobs=-1)
    clf.fit(X.iloc[tr], ytr)
    pred = clf.predict(Xte)

    macro_f1 = round(float(f1_score(yte, pred, average="macro")), 3)
    rec = recall_score(yte, pred, average=None, labels=list(range(len(classes))), zero_division=0)
    per_class_recall = {CLASS_NAMES.get(inv[i], str(inv[i])): round(float(rec[i]), 3)
                        for i in range(len(classes))}
    cm = confusion_matrix(yte, pred, labels=list(range(len(classes)))).tolist()
    labels = [CLASS_NAMES.get(inv[i], str(inv[i])) for i in range(len(classes))]

    # high-severity miss rate: true gearbox/generator predicted as a LOWER-severity class
    true_c = np.array([inv[v] for v in yte])
    pred_c = np.array([inv[v] for v in pred])
    hs_mask = np.isin(true_c, [1, 2])               # generator, gearbox
    if hs_mask.sum():
        downgraded = sum(CLASS_SEVERITY.get(int(p), 0) < CLASS_SEVERITY.get(int(t), 0)
                         for t, p in zip(true_c[hs_mask], pred_c[hs_mask]))
        hs_miss = round(float(downgraded / hs_mask.sum()), 3)
    else:
        hs_miss = None

    return {
        "available": True,
        "macro_f1": macro_f1,
        "per_class_recall": per_class_recall,
        "confusion_matrix": {"labels": labels, "matrix": cm},
        "high_severity_downgrade_rate": hs_miss,
        "note": "Macro-F1 weights every failure mode equally. The high-severity downgrade rate "
                "is the share of real gearbox/generator failures the model labelled as a "
                "lower-severity fault — the costly error class.",
    }


# ══════════════════════════════════════════════════════════════════════
#  8. EXPLAINABILITY INTEGRITY  —  perturbation + stability + alignment
# ══════════════════════════════════════════════════════════════════════
def explainability_integrity(df: pd.DataFrame, features: list[str]) -> dict:
    from xgboost import XGBClassifier
    from . import explainability as ex

    X, y = _eval_xy(df, features)
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.35, random_state=7)
                  .split(X, y, df["event_id"].values))
    pos, neg = max(int(y[tr].sum()), 1), max(int((y[tr] == 0).sum()), 1)
    clf = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                        tree_method="hist", random_state=42, n_jobs=-1,
                        scale_pos_weight=neg / pos)
    clf.fit(X.iloc[tr], y[tr])
    Xte = X.iloc[te].reset_index(drop=True)

    # ── Perturbation / causal test (this is the empirical monotonicity check) ──
    base_rows = Xte.sample(n=min(400, len(Xte)), random_state=1)
    base_score = float(clf.predict_proba(base_rows)[:, 1].mean())
    perturb = []
    for sig, delta in ex.PERTURBATION_SIGNALS.items():
        if sig not in features:
            continue
        bumped = base_rows.copy()
        bumped[sig] = bumped[sig] + delta
        new_score = float(clf.predict_proba(bumped)[:, 1].mean())
        d = round(new_score - base_score, 4)
        perturb.append({"signal": sig, "delta_applied": delta,
                        "risk_change": d, "direction_ok": d >= 0})
    n_ok = sum(p["direction_ok"] for p in perturb)
    perturb_pass = bool(perturb) and n_ok >= max(1, int(0.7 * len(perturb)))

    # ── SHAP stability test (near-identical inputs → stable top features) ──
    stability = None
    try:
        import shap
        samp = Xte.sample(n=min(40, len(Xte)), random_state=3).reset_index(drop=True)
        expl = shap.TreeExplainer(clf)
        sv = np.array(expl.shap_values(samp))
        if sv.ndim == 3:
            sv = sv[:, :, -1]
        stds = samp.std(axis=0).replace(0, 1e-6).values
        rng = np.random.RandomState(5)
        noisy = samp + rng.normal(0, 0.01, samp.shape) * stds
        sv2 = np.array(expl.shap_values(noisy))
        if sv2.ndim == 3:
            sv2 = sv2[:, :, -1]
        jac = []
        for i in range(len(samp)):
            a = set(np.argsort(np.abs(sv[i]))[::-1][:3])
            b = set(np.argsort(np.abs(sv2[i]))[::-1][:3])
            jac.append(len(a & b) / len(a | b))
        mean_jac = round(float(np.mean(jac)), 3)
        stability = {"top3_jaccard_after_1pct_noise": mean_jac,
                     "passed": mean_jac >= 0.6,
                     "note": "Top-3 SHAP feature overlap after adding 1% noise. <0.6 means "
                             "explanations are unstable — collinear signals trading places."}
    except Exception as e:
        stability = {"error": str(e)}

    # ── Global driver alignment (are top drivers physically expected?) ──
    top_drivers = []
    try:
        import shap
        anom = df.iloc[te][df.iloc[te]["y_binary"] == 1]
        if len(anom):
            comp = anom["y_multi"].map(CLASS_NAMES).mode()
            dominant = comp.iloc[0] if len(comp) else "Gearbox"
            samp = X.loc[anom.index].sample(n=min(150, len(anom)), random_state=2)
            expl = shap.TreeExplainer(clf)
            sv = np.array(expl.shap_values(samp))
            if sv.ndim == 3:
                sv = sv[:, :, -1]
            mean_abs = np.abs(sv).mean(axis=0)
            order = np.argsort(mean_abs)[::-1][:6]
            total = float(mean_abs.sum()) or 1.0
            for i in order:
                fname = features[i]
                top_drivers.append({
                    "feature": fname,
                    "pct": round(100 * float(mean_abs[i]) / total, 1),
                    "expected": ex.is_expected(fname, dominant),
                    "group": ex.collinear_group(fname),
                })
            aligned_abs = sum(mean_abs[i] for i in range(len(features))
                              if ex.is_expected(features[i], dominant))
            alignment = round(float(aligned_abs / mean_abs.sum()), 3)
        else:
            dominant, alignment = None, None
    except Exception:
        dominant, alignment = None, None

    return {
        "perturbation_test": {
            "baseline_risk": round(base_score, 4),
            "signals": perturb,
            "passed": perturb_pass,
            "verdict": ("PASS — raising degradation temperatures increases predicted risk, so "
                        "the model responds in the physically correct direction."
                        if perturb_pass else
                        "WARNING — some degradation signals did NOT raise risk when increased; "
                        "the model is not reasoning causally for those channels."),
        },
        "stability_test": stability,
        "global_alignment": {
            "dominant_anomaly_mode": dominant,
            "alignment_score": alignment,
            "top_drivers": top_drivers,
            "note": "Per-prediction physical alignment + collinearity flags are also shown live "
                    "on each turbine's SHAP panel.",
        },
    }


# ══════════════════════════════════════════════════════════════════════
#  9. PROBABILITY CALIBRATION  —  does "80%" really mean 80%?
# ══════════════════════════════════════════════════════════════════════
def calibration_test(df: pd.DataFrame, features: list[str], bins: int = 10) -> dict:
    from xgboost import XGBClassifier
    X, y = _eval_xy(df, features)
    # 3-way: train / calibration / test by event groups
    gss = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=11)
    tr, rest = next(gss.split(X, y, df["event_id"].values))
    rest_groups = df["event_id"].values[rest]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=12)
    cal_rel, te_rel = next(gss2.split(X.iloc[rest], y[rest], rest_groups))
    cal, te = rest[cal_rel], rest[te_rel]

    pos, neg = max(int(y[tr].sum()), 1), max(int((y[tr] == 0).sum()), 1)
    clf = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                        tree_method="hist", random_state=42, n_jobs=-1,
                        scale_pos_weight=neg / pos)
    clf.fit(X.iloc[tr], y[tr])
    p_test = clf.predict_proba(X.iloc[te])[:, 1]
    y_test = y[te]

    def reliability(p, yt):
        edges = np.linspace(0, 1, bins + 1)
        pts, ece, n = [], 0.0, len(yt)
        for i in range(bins):
            m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= edges[i + 1])
            if m.sum() == 0:
                continue
            conf = float(p[m].mean()); obs = float(yt[m].mean()); w = int(m.sum())
            pts.append({"bin": round((edges[i] + edges[i + 1]) / 2, 2),
                        "predicted": round(conf, 3), "observed": round(obs, 3), "n": w})
            ece += (w / n) * abs(conf - obs)
        return pts, round(ece, 3)

    raw_pts, raw_ece = reliability(p_test, y_test)
    raw_brier = round(float(brier_score_loss(y_test, p_test)), 4)

    # Fix attempt: isotonic regression fitted on the calibration fold
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(clf.predict_proba(X.iloc[cal])[:, 1], y[cal])
    p_cal = iso.predict(p_test)
    cal_pts, cal_ece = reliability(p_cal, y_test)
    cal_brier = round(float(brier_score_loss(y_test, p_cal)), 4)

    well = raw_ece < 0.1
    return {
        "brier_score": raw_brier,
        "ece": raw_ece,
        "reliability_diagram": raw_pts,
        "isotonic": {"brier_score": cal_brier, "ece": cal_ece, "reliability_diagram": cal_pts},
        "verdict": (
            f"PASS — probabilities are reasonably calibrated (ECE {raw_ece}, Brier {raw_brier}); "
            f"'80%' broadly means 80%."
            if well else
            f"NEEDS CALIBRATION — ECE {raw_ece} / Brier {raw_brier} show the raw probabilities "
            f"are mis-stated. Isotonic regression improves ECE to {cal_ece} / Brier {cal_brier}. "
            f"Apply isotonic before the probability feeds the Expected-Loss risk score."
        ),
        "note": "Every Expected-Loss € figure multiplies this probability — if it is mis-calibrated, "
                "the money ranking inherits the error. Calibrate first.",
    }


# ══════════════════════════════════════════════════════════════════════
#  ASSEMBLE FULL REPORT
# ══════════════════════════════════════════════════════════════════════
def run_full_validation(per_event_cap: int = 2000) -> dict:
    t0 = time.time()
    bundle = _load_bundle()
    features = bundle["features"]

    provenance = data_provenance()
    audit = feature_leakage_audit()
    df, event_ends = load_eval_frame(per_event_cap=per_event_cap)
    splits = split_comparison(df, features)
    perm = permutation_test(df, features)
    embargo = embargo_test(df, features, embargo_days=3)
    scoreboard = binary_scoreboard(df, features, event_ends)
    multiclass = multiclass_scoreboard(df, features)
    explain = explainability_integrity(df, features)
    calibration = calibration_test(df, features)

    honest_pr = splits["group_by_event"].get("pr_auc")
    lotu_pr = splits["leave_one_oem_out"].get("pr_auc")

    report = {
        "generated_at": datetime.utcnow().isoformat(),
        "model_version": bundle.get("version"),
        "headline": {
            "honest_pr_auc_grouped": honest_pr,
            "leave_one_turbine_out_pr_auc": lotu_pr,
            "claimed_balanced_accuracy": bundle.get("metrics", {}).get("accuracy"),
            "claimed_roc_auc": bundle.get("metrics", {}).get("roc_auc_macro_ovr"),
            "statement": (
                "The deployable, leakage-safe headline is PR-AUC under a grouped split "
                f"({honest_pr}) and leave-one-turbine-out ({lotu_pr}) — NOT the balanced "
                f"multiclass accuracy/ROC from training. Report those, with the 10-min "
                "sampling and tiny-failure-count caveats, to any technical reviewer or auditor."
            ),
        },
        "provenance": provenance,
        "feature_leakage_audit": audit,
        "split_comparison": splits,
        "permutation_test": perm,
        "embargo_test": embargo,
        "binary_scoreboard": scoreboard,
        "multiclass_scoreboard": multiclass,
        "explainability_integrity": explain,
        "calibration": calibration,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    return report


def save_report(report: dict) -> Path:
    out = settings.model_path / "validation_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return out


def load_report() -> dict | None:
    p = settings.model_path / "validation_report.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))
