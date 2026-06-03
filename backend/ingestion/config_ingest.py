"""
═══════════════════════════════════════════════════════════════════════
 CONFIG-DRIVEN INGESTION
═══════════════════════════════════════════════════════════════════════
Onboard a new site with ONLY a YAML/JSON mapping file — no per-site Python.

  load_mapping()        read the YAML/JSON site config
  validate_mapping()    flag unmapped / unknown / missing-required channels
  apply_mapping()       rename source->canonical + unit conversion (generic)
  data_quality_report() missingness, frozen sensors, timestamp gaps, timezone,
                        out-of-range — the report generated on every ingestion
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .canonical_model import (
    CANONICAL_FIELDS, REQUIRED_FIELDS, is_known, field_meta, convert_units,
)

META_DEFAULTS = {"timestamp_column": "time_stamp", "asset_id_column": "asset_id",
                 "timezone": "UTC", "sample_rate_seconds": 600}


def load_mapping(path: str | Path) -> dict:
    """Load a site mapping config from YAML or JSON. No code, just config."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        import yaml
        cfg = yaml.safe_load(text)
    else:
        cfg = json.loads(text)
    for k, v in META_DEFAULTS.items():
        cfg.setdefault(k, v)
    # normalise mappings list -> also build index
    cfg["_index"] = {m["source"]: m for m in cfg.get("mappings", [])}
    return cfg


def validate_mapping(cfg: dict, df_columns: list[str]) -> dict:
    """Flag mapping problems before ingestion."""
    idx = cfg["_index"]
    meta_cols = {cfg["timestamp_column"], cfg["asset_id_column"],
                 "id", "train_test", "status_type_id", "event_id",
                 "event_label", "event_description", "source_file"}
    df_cols = set(df_columns)

    mapped_sources = set(idx.keys())
    refers_missing = sorted(mapped_sources - df_cols)                 # mapping points at absent columns
    unmapped = sorted(c for c in df_cols - mapped_sources if c not in meta_cols)
    canon_targets = {m["canonical"] for m in cfg.get("mappings", [])}
    unknown_canonical = sorted(c for c in canon_targets if not is_known(c))
    missing_required = sorted(set(REQUIRED_FIELDS) - canon_targets)
    bad_conversions = []
    for m in cfg.get("mappings", []):
        cm = field_meta(m["canonical"])
        su = m.get("unit")
        if cm and su and su != cm["unit"]:
            from .canonical_model import CONVERSIONS
            if (su, cm["unit"]) not in CONVERSIONS:
                bad_conversions.append(f"{m['source']}: {su}->{cm['unit']} (no converter)")

    n_issues = len(refers_missing) + len(unknown_canonical) + len(missing_required) + len(bad_conversions)
    return {
        "mapped_channels": len(mapped_sources & df_cols),
        "mapping_refers_missing_source": refers_missing,
        "unmapped_source_channels": unmapped,
        "unknown_canonical_targets": unknown_canonical,
        "missing_required_fields": missing_required,
        "unconvertible_units": bad_conversions,
        "passed": n_issues == 0,
        "verdict": ("PASS — every required field is mapped and all targets/units are valid."
                    if n_issues == 0 else
                    f"{n_issues} mapping issue(s) to resolve before ingestion."),
    }


def apply_mapping(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, list[str]]:
    """Rename source->canonical and convert units. Fully generic (config-driven)."""
    idx = cfg["_index"]
    rename = {src: m["canonical"] for src, m in idx.items() if src in df.columns}
    out = df.rename(columns=rename)
    conversions = []
    for src, m in idx.items():
        canon = m["canonical"]
        cm = field_meta(canon)
        su = m.get("unit")
        if canon in out.columns and cm and su and su != cm["unit"]:
            out[canon], applied = convert_units(out[canon], su, cm["unit"])
            if applied:
                conversions.append(f"{canon}: {su}->{cm['unit']}")
    return out, conversions


def data_quality_report(df: pd.DataFrame, cfg: dict) -> dict:
    """Full DQ report: timestamps, missingness, frozen sensors, ranges, tz."""
    ts_col = cfg["timestamp_column"]
    asset_col = cfg["asset_id_column"]
    rate = float(cfg["sample_rate_seconds"])
    n = len(df)

    # ── timestamps ──
    ts = pd.to_datetime(df[ts_col], errors="coerce")
    n_bad_ts = int(ts.isna().sum())
    duplicates = int(df.duplicated(subset=[c for c in (asset_col, ts_col) if c in df.columns]).sum())
    diffs = []
    if asset_col in df.columns:
        for _, g in df.assign(_ts=ts).dropna(subset=["_ts"]).sort_values([asset_col, "_ts"]).groupby(asset_col):
            diffs.extend(g["_ts"].diff().dropna().dt.total_seconds().tolist())
    else:
        diffs = ts.dropna().sort_values().diff().dropna().dt.total_seconds().tolist()
    diffs = np.array(diffs) if diffs else np.array([rate])
    median_rate = float(np.median(diffs))
    gaps = int((diffs > 1.5 * rate).sum())
    largest_gap_min = round(float(diffs.max()) / 60.0, 1) if len(diffs) else 0
    rate_mismatch = abs(median_rate - rate) > 0.2 * rate

    tz_note = (f"Declared timezone: {cfg['timezone']}. Timestamps are timezone-naive in the feed; "
               f"they will be interpreted as {cfg['timezone']} and stored UTC.")

    # ── per-canonical-field stats ──
    canon_df, conversions = apply_mapping(df, cfg)
    fields = []
    frozen, oor_fields, high_missing = [], [], []
    for m in cfg.get("mappings", []):
        f = m["canonical"]
        if f not in canon_df.columns:
            continue
        col = pd.to_numeric(canon_df[f], errors="coerce")
        miss = round(100.0 * col.isna().mean(), 1)
        std = float(col.std()) if col.notna().sum() > 1 else 0.0
        meta = field_meta(f) or {}
        oor = None
        if "min" in meta:
            outside = ((col < meta["min"]) | (col > meta["max"]))
            oor = round(100.0 * outside.mean(), 1)
        is_frozen = bool(meta.get("frozen_check") and col.notna().sum() > 100 and std < 1e-9)
        if is_frozen:
            frozen.append(f)
        if oor is not None and oor > 1.0:
            oor_fields.append(f"{f} ({oor}%)")
        if miss > 20:
            high_missing.append(f"{f} ({miss}%)")
        fields.append({"field": f, "unit": (meta.get("unit")), "missing_pct": miss,
                       "frozen": is_frozen, "out_of_range_pct": oor})

    # ── readiness score ──
    score = 100
    score -= min(30, len(high_missing) * 6)
    score -= min(25, len(frozen) * 8)
    score -= min(20, gaps and 10 or 0)
    score -= min(15, len(oor_fields) * 4)
    if rate_mismatch:
        score -= 10
    score = max(0, score)
    verdict = ("READY — data quality sufficient for onboarding."
               if score >= 80 else
               "NEEDS ATTENTION — resolve flagged channels before relying on outputs."
               if score >= 55 else
               "BLOCKED — data quality too low for reliable inference.")

    return {
        "rows_assessed": n,
        "timestamps": {
            "invalid": n_bad_ts, "duplicates": duplicates,
            "declared_rate_seconds": rate, "observed_median_rate_seconds": round(median_rate, 1),
            "rate_mismatch": rate_mismatch, "gaps_over_1_5x_rate": gaps,
            "largest_gap_minutes": largest_gap_min,
        },
        "timezone_note": tz_note,
        "unit_conversions_applied": conversions,
        "fields": fields,
        "flags": {
            "frozen_sensors": frozen,
            "out_of_range_fields": oor_fields,
            "high_missingness_fields": high_missing,
        },
        "readiness_score": score,
        "verdict": verdict,
    }
