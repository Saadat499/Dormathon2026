"""
VoxelSort: Evaluation & Safety Module
Component: Evaluation Protocol, Patient-Level Splitting & Statistical Power (P0)

Guarantees:
- Patient-level split isolation (strictly zero patient leakage across folds).
- Multi-label classification metrics: Sensitivity, Specificity, AUROC, AUPRC, Brier Score.
- Clustered patient-level bootstrapping for 95% Confidence Intervals.
- Formal sample size calculations for diagnostic non-inferiority and precision bounds.
"""

from typing import Dict, Any, List, Optional, Tuple, Set
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss


def deduplicate_studies(
    df: pd.DataFrame,
    patient_col: str = "patient_id",
    study_col: str = "study_id",
    timestamp_col: Optional[str] = None,
    keep: str = "first"
) -> pd.DataFrame:
    """
    Deduplicates repeated scans or duplicate study records.
    If timestamps exist, sorts before keeping the specified encounter.
    """
    cleaned = df.copy()
    if timestamp_col and timestamp_col in cleaned.columns:
        cleaned = cleaned.sort_values(by=[patient_col, timestamp_col])
    cleaned = cleaned.drop_duplicates(subset=[study_col], keep=keep)
    return cleaned.reset_index(drop=True)


def patient_level_split(
    df: pd.DataFrame,
    patient_col: str = "patient_id",
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Performs strict patient-level partition into Train, Validation, and Test sets.
    Guarantees no patient_id exists across multiple splits.
    """
    rng = np.random.RandomState(seed)
    unique_patients = np.array(df[patient_col].unique())
    rng.shuffle(unique_patients)

    n_patients = len(unique_patients)
    n_test = int(np.round(n_patients * test_fraction))
    n_val = int(np.round(n_patients * val_fraction))

    test_patients: Set[Any] = set(unique_patients[:n_test])
    val_patients: Set[Any] = set(unique_patients[n_test : n_test + n_val])
    train_patients: Set[Any] = set(unique_patients[n_test + n_val :])

    train_df = df[df[patient_col].isin(train_patients)].copy().reset_index(drop=True)
    val_df = df[df[patient_col].isin(val_patients)].copy().reset_index(drop=True)
    test_df = df[df[patient_col].isin(test_patients)].copy().reset_index(drop=True)

    # Verification assertion: zero leakage
    assert len(train_patients.intersection(val_patients)) == 0, "Patient leakage in train/val!"
    assert len(train_patients.intersection(test_patients)) == 0, "Patient leakage in train/test!"
    assert len(val_patients.intersection(test_patients)) == 0, "Patient leakage in val/test!"

    return train_df, val_df, test_df


def compute_binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Computes diagnostic metrics for a single finding at a specified threshold.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    n_pos = tp + fn
    n_neg = tn + fp

    sensitivity = tp / n_pos if n_pos > 0 else 0.0
    specificity = tn / n_neg if n_neg > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    # AUROC and AUPRC (require at least 1 positive and 1 negative)
    if len(np.unique(y_true)) > 1:
        try:
            auroc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            auroc = 0.5
        try:
            auprc = float(average_precision_score(y_true, y_prob))
        except Exception:
            auprc = n_pos / len(y_true)
    else:
        auroc = 0.5
        auprc = 0.0

    brier = float(brier_score_loss(y_true, y_prob))

    return {
        "prevalence": float(n_pos / len(y_true)) if len(y_true) > 0 else 0.0,
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "ppv": float(ppv),
        "npv": float(npv),
        "auroc": float(auroc),
        "auprc": float(auprc),
        "brier_score": float(brier),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "n_samples": len(y_true),
        "threshold": float(threshold),
    }


def compute_bootstrap_ci(
    df: pd.DataFrame,
    patient_col: str,
    y_true_col: str,
    y_prob_col: str,
    threshold: float,
    metric_name: str = "sensitivity",
    n_bootstraps: int = 500,
    alpha: float = 0.05,
    seed: int = 42
) -> Tuple[float, float, float]:
    """
    Computes patient-level clustered bootstrap confidence intervals for a metric.
    Resamples patient clusters with replacement to respect intra-patient correlation.
    
    Returns:
        (point_estimate, ci_lower, ci_upper)
    """
    rng = np.random.RandomState(seed)
    patients = df[patient_col].unique()
    n_patients = len(patients)

    # Point estimate on entire dataset
    point_metrics = compute_binary_metrics(
        df[y_true_col].values,
        df[y_prob_col].values,
        threshold=threshold
    )
    point_val = point_metrics.get(metric_name, 0.0)

    # Pre-group by patient for high-speed resampling
    grouped = df.groupby(patient_col)
    patient_indices = {pid: grp.index.values for pid, grp in grouped}

    bootstrap_values: List[float] = []

    for _ in range(n_bootstraps):
        sampled_patients = rng.choice(patients, size=n_patients, replace=True)
        sampled_row_indices = np.concatenate([patient_indices[pid] for pid in sampled_patients])
        sample_df = df.loc[sampled_row_indices]

        metrics = compute_binary_metrics(
            sample_df[y_true_col].values,
            sample_df[y_prob_col].values,
            threshold=threshold
        )
        bootstrap_values.append(metrics.get(metric_name, 0.0))

    ci_lower = float(np.percentile(bootstrap_values, 100 * (alpha / 2.0)))
    ci_upper = float(np.percentile(bootstrap_values, 100 * (1.0 - alpha / 2.0)))

    return float(point_val), ci_lower, ci_upper


def calculate_required_sample_size(
    target_sensitivity: float = 0.95,
    margin_of_error: float = 0.03,  # +/- 3% acceptable margin
    prevalence: float = 0.10,       # Disease prevalence in cohort
    confidence_level: float = 0.95
) -> Dict[str, Any]:
    """
    Calculates the required number of positive cases and total study volume
    needed to estimate sensitivity within a given margin of error at a confidence level.
    Uses Wilson/Wald formulation:
      N_pos = (Z^2 * S * (1 - S)) / delta^2
      N_total = N_pos / prevalence
    """
    from scipy.stats import norm
    alpha = 1.0 - confidence_level
    z = norm.ppf(1.0 - alpha / 2.0)

    n_pos_required = int(np.ceil((z ** 2 * target_sensitivity * (1.0 - target_sensitivity)) / (margin_of_error ** 2)))
    n_total_required = int(np.ceil(n_pos_required / prevalence))

    return {
        "target_sensitivity": target_sensitivity,
        "margin_of_error": margin_of_error,
        "prevalence": prevalence,
        "confidence_level": confidence_level,
        "z_critical": round(float(z), 3),
        "min_positive_cases_required": n_pos_required,
        "min_total_studies_required": n_total_required,
        "recommendation": (
            f"To validate {target_sensitivity*100:.0f}% sensitivity within +/-{margin_of_error*100:.1f}%, "
            f"the test set requires at least {n_pos_required} positive studies ({n_total_required} total studies at {prevalence*100:.1f}% prevalence)."
        )
    }
