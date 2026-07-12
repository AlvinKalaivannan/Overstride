# Stage 2 offline calibration report

Dataset: Mid-Long Distance Runners Injuries (74 athletes,
42798 athlete-weeks, 575 injury-labeled rows)

## Athlete-level 5-fold cross-validated holdout

| Metric | Value |
|---|---|
| AUC | 0.692 |
| Recall (threshold 0.5) | 0.449 |

Population covariance and logistic coefficients are refit per fold on
training-fold athletes only; reported metrics are pooled out-of-fold
predictions across all athletes, so no athlete's own data leaks into its
own held-out prediction.

## Benchmark (Lövdal et al. 2021, bagged XGBoost, same dataset)

- Day-level AUC: 0.724
- Week-level AUC: 0.678

## Deployed model

Frozen coefficients (refit on the full 74-athlete dataset) saved to
`models/stage2_logistic_coeffs.json`: intercept=-0.7058,
coef_d2=0.0257.
