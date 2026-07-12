"""Stage 2 offline calibration: run this once to produce the frozen risk model.

    python scripts/run_stage2_calibration.py

Reads data/raw/mid_long_distance_runners_injuries/timeseries_weekly.csv
(see scripts/download_dataset1.py), reports athlete-level cross-validated
AUC/recall against the Lövdal et al. (2021) benchmark, then refits on the
full dataset and freezes the logistic coefficients to
models/stage2_logistic_coeffs.json.
"""

from __future__ import annotations

import json
from pathlib import Path

from overstride.risk.calibration import (
    LABEL_COL,
    cross_validated_auc,
    fit_final_model,
    load_dataset,
)

DATA_PATH = (
    Path(__file__).resolve().parent.parent
    / "data" / "raw" / "mid_long_distance_runners_injuries" / "timeseries_weekly.csv"
)
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "stage2_logistic_coeffs.json"
REPORT_PATH = Path(__file__).resolve().parent.parent / "models" / "stage2_validation_report.md"

# Lövdal, Den Hartigh & Azzopardi (2021) — bagged XGBoost, same dataset.
BENCHMARK_AUC_DAY = 0.724
BENCHMARK_AUC_WEEK = 0.678


def main() -> None:
    df = load_dataset(DATA_PATH)
    print(f"Loaded {len(df)} athlete-weeks, {df['Athlete ID'].nunique()} athletes, "
          f"{int(df[LABEL_COL].sum())} injury-labeled rows")

    print("\nRunning athlete-level 5-fold cross-validation (GroupKFold)...")
    metrics = cross_validated_auc(df, n_splits=5)
    print(f"  AUC:    {metrics['auc']:.3f}")
    print(f"  Recall: {metrics['recall']:.3f}")
    print(f"  n={metrics['n']} (positives={metrics['n_positive']})")
    print(f"\nBenchmark to beat (Lövdal et al. 2021, bagged XGBoost): "
          f"day-level {BENCHMARK_AUC_DAY}, week-level {BENCHMARK_AUC_WEEK}")

    print("\nRefitting on full dataset for deployment...")
    frozen = fit_final_model(df)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    print(f"Frozen coefficients written to {MODEL_PATH}")

    report = f"""# Stage 2 offline calibration report

Dataset: Mid-Long Distance Runners Injuries ({df['Athlete ID'].nunique()} athletes,
{len(df)} athlete-weeks, {int(df[LABEL_COL].sum())} injury-labeled rows)

## Athlete-level 5-fold cross-validated holdout

| Metric | Value |
|---|---|
| AUC | {metrics['auc']:.3f} |
| Recall (threshold 0.5) | {metrics['recall']:.3f} |

Population covariance and logistic coefficients are refit per fold on
training-fold athletes only; reported metrics are pooled out-of-fold
predictions across all athletes, so no athlete's own data leaks into its
own held-out prediction.

## Benchmark (Lövdal et al. 2021, bagged XGBoost, same dataset)

- Day-level AUC: {BENCHMARK_AUC_DAY}
- Week-level AUC: {BENCHMARK_AUC_WEEK}

## Deployed model

Frozen coefficients (refit on the full 74-athlete dataset) saved to
`models/stage2_logistic_coeffs.json`: intercept={frozen['intercept']:.4f},
coef_d2={frozen['coef_d2']:.4f}.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"Report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
