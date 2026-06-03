"""
Config-driven site onboarding — validate a mapping file + produce a data-quality
report from a data sample. No per-site code.

Run (from backend/):
  python -m scripts.onboard_site --mapping ingestion/mappings/windfarm_a.yaml --csv "<path>" --rows 60000
"""
from __future__ import annotations

import argparse
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import pandas as pd
from config import settings
from ingestion.config_ingest import load_mapping, validate_mapping, data_quality_report


def main():
    ap = argparse.ArgumentParser(description="Validate a site mapping + run a data-quality report")
    ap.add_argument("--mapping", required=True)
    ap.add_argument("--csv", default=None, help="defaults to CSV_PATH in .env")
    ap.add_argument("--rows", type=int, default=60000)
    args = ap.parse_args()

    csv = args.csv or settings.CSV_PATH
    cfg = load_mapping(args.mapping)
    print(f"\n=== Onboarding: {cfg.get('site')} ({cfg.get('country','')}) ===")
    print(f"Sampling {args.rows:,} rows from {csv}\n")
    df = pd.read_csv(csv, nrows=args.rows, low_memory=False)

    val = validate_mapping(cfg, df.columns.tolist())
    print("-- Mapping validation --")
    print(f"  {val['verdict']}")
    print(f"  mapped channels      : {val['mapped_channels']}")
    print(f"  unmapped (info only) : {len(val['unmapped_source_channels'])} channels not used")
    print(f"  missing required     : {val['missing_required_fields'] or 'none'}")
    print(f"  unknown canonical    : {val['unknown_canonical_targets'] or 'none'}")
    print(f"  bad unit conversions : {val['unconvertible_units'] or 'none'}")

    dq = data_quality_report(df, cfg)
    print("\n-- Data-quality report --")
    print(f"  readiness score : {dq['readiness_score']}/100  ->  {dq['verdict']}")
    t = dq["timestamps"]
    print(f"  timestamps      : {t['gaps_over_1_5x_rate']} gaps, largest {t['largest_gap_minutes']} min, "
          f"{t['duplicates']} dupes, rate {t['observed_median_rate_seconds']}s "
          f"(declared {t['declared_rate_seconds']}s)")
    print(f"  unit conversions: {dq['unit_conversions_applied'] or 'none'}")
    print(f"  frozen sensors  : {dq['flags']['frozen_sensors'] or 'none'}")
    print(f"  out-of-range    : {dq['flags']['out_of_range_fields'] or 'none'}")
    print(f"  high missingness: {dq['flags']['high_missingness_fields'] or 'none'}")
    print(f"\n  {len(dq['fields'])} canonical fields validated.")


if __name__ == "__main__":
    main()
