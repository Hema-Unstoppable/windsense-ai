"""
Run the full ML validation suite and save the report.

Run (from backend/):  python -m scripts.run_validation
"""
from __future__ import annotations

import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from ml.validation import run_full_validation, save_report


def main():
    print("\n=== WindSense AI — ML Validation Suite ===\n")
    report = run_full_validation(per_event_cap=2000)
    path = save_report(report)

    h = report["headline"]
    print(f"Model: {report['model_version']}")
    print(f"Sampling rate: {report['provenance']['sampling_rate_human']}")
    print(f"Fleet row positive rate: {report['provenance']['fleet_row_positive_rate']} "
          f"(benchmark artifact)")
    print(f"Real failure events: "
          f"{sum(t['n_real_failures'] for t in report['provenance']['turbines'])} across "
          f"{len(report['provenance']['turbines'])} turbines\n")
    print("--- Split comparison (PR-AUC) ---")
    for k in ["random_split", "group_by_event", "leave_one_turbine_out", "blocked_by_time"]:
        s = report["split_comparison"][k]
        print(f"  {k:24} PR-AUC={s.get('pr_auc')}  ROC={s.get('roc_auc')}  "
              f"base={s.get('base_rate')}")
    print(f"\n  {report['split_comparison']['interpretation']}\n")
    print("--- Label-permutation (cheating) test ---")
    print(f"  real={report['permutation_test']['real_label_pr_auc']}  "
          f"shuffled={report['permutation_test']['shuffled_label_pr_auc']}  "
          f"base={report['permutation_test']['base_rate']}")
    print(f"  {report['permutation_test']['verdict']}\n")
    print("--- Post-failure embargo test ---")
    print(f"  {report['embargo_test']['interpretation']}\n")
    print("--- Feature leakage audit ---")
    print(f"  {report['feature_leakage_audit']['verdict']}\n")
    print(f"HONEST HEADLINE: {h['statement']}\n")
    print(f"Saved → {path}")


if __name__ == "__main__":
    main()
