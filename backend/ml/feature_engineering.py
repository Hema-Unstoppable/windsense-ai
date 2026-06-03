"""
═══════════════════════════════════════════════════════════════════════
 FEATURE ENGINEERING
═══════════════════════════════════════════════════════════════════════
Refactor of the notebook's feature pipeline into reusable, production
functions. Works on the CANONICAL schema (post-normalisation), so it is
OEM-agnostic and applies to any client feed.

Pipeline:
  1. map raw event_description → fault_class (0..4)
  2. engineer domain features (temperature margins, rolling stats)
  3. programmatic feature selection (variance → correlation → mutual info)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .canonical import META_COLS

# event_description text → fault class
FAULT_KEYWORDS = {
    1: ["generator", "stator", "rotor winding", "gen bearing", "generator bearing"],
    2: ["gearbox", "gear box", "gear tooth", "high speed shaft"],
    3: ["hydraulic", "hydraulic group", "pitch"],
    4: ["transformer", "hv transformer"],
}


def map_fault_class(row) -> int:
    """0 normal, 1 generator, 2 gearbox, 3 hydraulic, 4 transformer."""
    if str(row.get("event_label", "normal")).lower() == "normal":
        return 0
    desc = str(row.get("event_description", "") or "").lower()
    for cls, keywords in FAULT_KEYWORDS.items():
        if any(k in desc for k in keywords):
            return cls
    return 0  # unlabeled anomaly treated conservatively


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["y_multi"] = df.apply(map_fault_class, axis=1)
    df["y_binary"] = (df.get("event_label", "normal")
                      .astype(str).str.lower().eq("anomaly").astype(int))
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Domain-engineered features. Uses canonical names; guards every column
    so missing signals (different OEM) never crash the pipeline.
    """
    df = df.copy()

    def col(name):
        return df[name] if name in df.columns else pd.Series(np.nan, index=df.index)

    nacelle = col("Nac_temperature_C_avg")
    df["gearbox_temp_margin"]    = col("Temp_oil_gearbox_C_avg") - nacelle
    df["hydraulic_temp_margin"]  = col("Temp_hydraulic_oil_group_C") - nacelle
    df["gen_bearing_DE_margin"]  = col("Temp_gen_bearing_Drive_End_C_avg") - nacelle
    df["gen_bearing_NDE_margin"] = col("Temp_gen_bearing_Non_Drive_End_C_avg") - nacelle

    # power coefficient proxy (power vs wind^3) — efficiency anomalies
    ws = col("Wind_speed_m/s_avg").clip(lower=0.1)
    df["power_per_wind"] = col("Grid_power_kW_avg") / (ws ** 3)
    df["power_per_wind"] = df["power_per_wind"].replace([np.inf, -np.inf], np.nan)

    return df


ENGINEERED = [
    "gearbox_temp_margin", "hydraulic_temp_margin",
    "gen_bearing_DE_margin", "gen_bearing_NDE_margin", "power_per_wind",
]


def select_features(df: pd.DataFrame, target_col: str = "y_multi",
                    top_k: int = 30, mi_sample: int = 40_000) -> list[str]:
    """
    Programmatic feature selection (mirrors the notebook but automated):
      variance filter → drop high-correlation redundancy → mutual information.
    Returns the top_k feature names + always keeps engineered features.
    """
    from sklearn.feature_selection import VarianceThreshold, mutual_info_classif

    candidates = [c for c in df.select_dtypes(include=[np.number]).columns
                  if c not in META_COLS and c not in ("y_multi", "y_binary")]
    X = df[candidates].apply(lambda c: c.fillna(c.median()))
    y = df[target_col]

    # 1) variance
    vt = VarianceThreshold(threshold=1e-4)
    vt.fit(X)
    X = X.loc[:, vt.get_support()]

    # 2) redundancy (|r| > 0.95)
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop = {col for col in upper.columns if any(upper[col] > 0.95)}
    X = X.drop(columns=list(drop), errors="ignore")

    # 3) mutual information (subsampled for speed)
    n = min(mi_sample, len(X))
    Xs = X.sample(n=n, random_state=42)
    ys = y.loc[Xs.index]
    mi = mutual_info_classif(Xs, ys, random_state=42)
    ranked = pd.Series(mi, index=Xs.columns).sort_values(ascending=False)

    selected = ranked.head(top_k).index.tolist()
    for f in ENGINEERED:
        if f in df.columns and f not in selected:
            selected.append(f)
    return selected


def build_training_frame(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Full chain: target → engineered features (assumes already normalised)."""
    df = add_target(df_raw)
    df = engineer_features(df)
    return df
