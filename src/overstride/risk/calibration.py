"""Stage 2 offline calibration: per-athlete Mahalanobis baselines -> frozen logistic regression.

Dataset #1 (Mid-Long Distance Runners Injuries) already ships each row as an
athlete-week observation: a 22-feature "week 0" block, plus lagged week-1 /
week-2 blocks, plus three week-over-week relative km ratios. We use the week-0
block + the three relative ratios as the feature vector — the 22-feature
weekly vector "plus week-over-week relative features" the README describes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from overstride.baseline.mahalanobis import fit_baseline, mahalanobis_sq

WEEK0_FEATURE_COLUMNS = [
    "nr. sessions",
    "nr. rest days",
    "total kms",
    "max km one day",
    "total km Z3-Z4-Z5-T1-T2",
    "nr. tough sessions (effort in Z5, T1 or T2)",
    "nr. days with interval session",
    "total km Z3-4",
    "max km Z3-4 one day",
    "total km Z5-T1-T2",
    "max km Z5-T1-T2 one day",
    "total hours alternative training",
    "nr. strength trainings",
    "avg exertion",
    "min exertion",
    "max exertion",
    "avg training success",
    "min training success",
    "max training success",
    "avg recovery",
    "min recovery",
    "max recovery",
]

RELATIVE_FEATURE_COLUMNS = [
    "rel total kms week 0_1",
    "rel total kms week 0_2",
    "rel total kms week 1_2",
]

FEATURE_COLUMNS = WEEK0_FEATURE_COLUMNS + RELATIVE_FEATURE_COLUMNS

ATHLETE_COL = "Athlete ID"
LABEL_COL = "injury"


def load_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def feature_matrix(df: pd.DataFrame) -> np.ndarray:
    return df[FEATURE_COLUMNS].to_numpy(dtype=float)


def population_covariance(df: pd.DataFrame) -> np.ndarray:
    """Pooled covariance of clean (non-injury) weeks across the given athletes only.

    Callers must restrict `df` to training-fold athletes to avoid leaking
    held-out athletes' distribution into the shrinkage prior.
    """
    clean = df[df[LABEL_COL] == 0]
    return np.cov(feature_matrix(clean), rowvar=False)


def compute_d2(
    df: pd.DataFrame,
    population_cov: np.ndarray,
    prior_strength: float = 10.0,
) -> pd.Series:
    """Per-athlete baseline (from that athlete's own clean weeks) -> D^2 for every row.

    `population_cov` is passed in (not recomputed here) so callers control
    exactly which athletes contributed to the shrinkage prior.
    """
    d2 = np.empty(len(df))
    for _, idx in df.groupby(ATHLETE_COL).groups.items():
        athlete_df = df.loc[idx]
        clean = athlete_df[athlete_df[LABEL_COL] == 0]
        mean, cov = fit_baseline(feature_matrix(clean), population_cov, prior_strength)
        d2[df.index.get_indexer(idx)] = mahalanobis_sq(feature_matrix(athlete_df), mean, cov)
    return pd.Series(d2, index=df.index)


def fit_logistic(d2: np.ndarray, y: np.ndarray) -> LogisticRegression:
    model = LogisticRegression(class_weight="balanced")
    model.fit(d2.reshape(-1, 1), y)
    return model


def evaluate(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "auc": roc_auc_score(y_true, y_prob),
        "recall": recall_score(y_true, y_pred),
        "n": len(y_true),
        "n_positive": int(y_true.sum()),
    }


def cross_validated_auc(
    df: pd.DataFrame,
    n_splits: int = 5,
    prior_strength: float = 10.0,
    random_state: int = 0,
) -> dict:
    """Athlete-level GroupKFold: population covariance and logistic fit are
    trained on the train-fold athletes only; out-of-fold D^2 for test-fold
    athletes are scored and pooled for a single overall AUC/recall.
    """
    groups = df[ATHLETE_COL].to_numpy()
    gkf = GroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    oof_prob = np.empty(len(df))
    oof_prob[:] = np.nan

    for train_idx, test_idx in gkf.split(df, groups=groups):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

        pop_cov = population_covariance(train_df)
        train_d2 = compute_d2(train_df, pop_cov, prior_strength).to_numpy()
        test_d2 = compute_d2(test_df, pop_cov, prior_strength).to_numpy()

        model = fit_logistic(train_d2, train_df[LABEL_COL].to_numpy())
        oof_prob[test_idx] = model.predict_proba(test_d2.reshape(-1, 1))[:, 1]

    y = df[LABEL_COL].to_numpy()
    return evaluate(y, oof_prob)


def fit_final_model(df: pd.DataFrame, prior_strength: float = 10.0) -> dict:
    """Refit on the full dataset for deployment: population covariance and
    logistic coefficients computed over all 74 athletes. This is the frozen
    artifact — not used for the AUC report, which comes from cross_validated_auc.
    """
    pop_cov = population_covariance(df)
    d2 = compute_d2(df, pop_cov, prior_strength).to_numpy()
    model = fit_logistic(d2, df[LABEL_COL].to_numpy())
    return {
        "feature_columns": FEATURE_COLUMNS,
        "prior_strength": prior_strength,
        "population_cov": pop_cov.tolist(),
        "intercept": float(model.intercept_[0]),
        "coef_d2": float(model.coef_[0][0]),
    }
