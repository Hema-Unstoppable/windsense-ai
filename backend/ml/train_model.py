"""
═══════════════════════════════════════════════════════════════════════
 TRAINING PIPELINE
═══════════════════════════════════════════════════════════════════════
Loads SCADA data, normalises it, engineers features, trains a multi-class
fault classifier (XGBoost or Random Forest), evaluates it, and saves a
versioned model bundle to disk + registers it in model_registry.

Memory-safe: balanced class sampling is done by streaming the 824 MB CSV
in chunks, so it runs on a normal laptop.

Run (from the backend/ directory):
    python -m ml.train_model
    python -m ml.train_model --algorithm random_forest --rows-per-class 2000
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from config import settings, SessionLocal
from .canonical import normalise_columns
from .feature_engineering import add_target, engineer_features, select_features
from .fmeca import CLASS_TO_COMPONENT
from .evaluation import evaluate_classifier


def load_balanced_from_csv(csv_path: str, rows_per_class: int,
                           profile: str = "windfarm_a",
                           chunksize: int = 200_000) -> pd.DataFrame:
    """
    Stream the CSV in chunks and collect a class-balanced sample.
    Returns a normalised + feature-engineered DataFrame.
    """
    buckets: dict[int, list[pd.DataFrame]] = {0: [], 1: [], 2: [], 3: [], 4: []}
    counts = {k: 0 for k in buckets}
    target = rows_per_class

    print(f"[train] streaming {csv_path} in {chunksize:,}-row chunks ...")
    for ch in pd.read_csv(csv_path, chunksize=chunksize, low_memory=False):
        ch = normalise_columns(ch, profile)
        ch = add_target(ch)
        for cls in buckets:
            if counts[cls] >= target:
                continue
            sub = ch[ch["y_multi"] == cls]
            if len(sub):
                need = target - counts[cls]
                take = sub.sample(n=min(need, len(sub)), random_state=42)
                buckets[cls].append(take)
                counts[cls] += len(take)
        if all(counts[c] >= target for c in buckets):
            break
    print(f"[train] collected per-class rows: {counts}")

    frames = [pd.concat(v) for v in buckets.values() if v]
    df = pd.concat(frames).sample(frac=1.0, random_state=42).reset_index(drop=True)
    df = engineer_features(df)
    return df


def build_classifier(algorithm: str, n_classes: int):
    if algorithm == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=300, max_features="sqrt",
            class_weight="balanced_subsample", n_jobs=-1, random_state=42,
        )
    else:  # xgboost
        from xgboost import XGBClassifier
        clf = XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            objective="multi:softprob", num_class=n_classes,
            eval_metric="mlogloss", tree_method="hist",
            random_state=42, n_jobs=-1,
        )
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("clf", clf)])


def train(algorithm: str | None = None, rows_per_class: int | None = None) -> dict:
    algorithm = algorithm or settings.MODEL_ALGORITHM
    rows_per_class = rows_per_class or settings.ROWS_PER_CLASS
    t0 = time.time()

    df = load_balanced_from_csv(settings.CSV_PATH, rows_per_class)

    # only keep classes that actually have data
    present = sorted(df["y_multi"].unique().tolist())
    df = df[df["y_multi"].isin(present)].copy()

    features = select_features(df, top_k=30)
    print(f"[train] selected {len(features)} features")

    X = df[features]
    y = df["y_multi"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    pipe = build_classifier(algorithm, n_classes=len(present))
    print(f"[train] fitting {algorithm} ...")
    pipe.fit(X_tr, y_tr)

    metrics = evaluate_classifier(pipe, X_te, y_te)
    print(f"[train] metrics: {metrics}")

    version = datetime.utcnow().strftime("v%Y.%m.%d.%H%M")
    bundle = {
        "pipeline": pipe,
        "features": features,
        "algorithm": algorithm,
        "version": version,
        "classes": present,
        "class_to_component": CLASS_TO_COMPONENT,
        "metrics": metrics,
        "trained_at": datetime.utcnow().isoformat(),
    }
    out = settings.model_path / f"windsense_{algorithm}_{version}.joblib"
    joblib.dump(bundle, out)
    # stable "latest" pointer used by inference
    joblib.dump(bundle, settings.model_path / "latest.joblib")
    print(f"[train] saved model bundle -> {out}")

    _register_model(version, algorithm, len(X_tr), features, metrics, str(out))

    # fit the production probability calibrator for this model version
    try:
        from .calibrate import fit_calibrator
        cal_metrics = print("[train] fitting probability calibrator ...") or fit_calibrator()
        metrics["calibration"] = cal_metrics
    except Exception as e:
        print(f"[train] calibrator skipped: {e}")

    bundle_summary = {
        "version": version, "algorithm": algorithm,
        "n_train_rows": len(X_tr), "n_features": len(features),
        "metrics": metrics, "artifact": str(out),
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    return bundle_summary


def _register_model(version, algorithm, n_rows, features, metrics, path):
    """Insert into model_registry and deactivate previous versions."""
    from models import ModelRegistry
    db = SessionLocal()
    try:
        db.query(ModelRegistry).update({ModelRegistry.is_active: False})
        db.add(ModelRegistry(
            user_id=None, version=version, algorithm=algorithm,
            n_train_rows=n_rows, feature_list=features,
            metrics=metrics, artifact_path=path, is_active=True,
        ))
        db.commit()
    except Exception as e:   # registry is best-effort (table may not exist yet)
        print(f"[train] registry skipped: {e}")
        db.rollback()
    finally:
        db.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train the WindSense fault classifier")
    ap.add_argument("--algorithm", choices=["xgboost", "random_forest"], default=None)
    ap.add_argument("--rows-per-class", type=int, default=None)
    args = ap.parse_args()
    summary = train(args.algorithm, args.rows_per_class)
    print("\n=== TRAINING COMPLETE ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
