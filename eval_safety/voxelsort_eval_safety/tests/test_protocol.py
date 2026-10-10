"""
Tests for Protocol, Patient-Level Splitting, Deduplication, Bootstrap CI, and Sample Size.
"""

import pytest
import numpy as np
import pandas as pd
from protocol import (
    deduplicate_studies,
    patient_level_split,
    compute_binary_metrics,
    compute_bootstrap_ci,
    calculate_required_sample_size,
)


def test_patient_level_split_isolation():
    """Verify zero patient overlap across train, val, and test partitions."""
    rng = np.random.RandomState(42)
    pids = [f"P_{i:03d}" for i in rng.randint(0, 100, size=300)]
    sids = [f"S_{j:04d}" for j in range(300)]
    df = pd.DataFrame({"patient_id": pids, "study_id": sids})

    train_df, val_df, test_df = patient_level_split(df, val_fraction=0.2, test_fraction=0.2, seed=42)

    train_patients = set(train_df["patient_id"].unique())
    val_patients = set(val_df["patient_id"].unique())
    test_patients = set(test_df["patient_id"].unique())

    assert len(train_patients.intersection(val_patients)) == 0
    assert len(train_patients.intersection(test_patients)) == 0
    assert len(val_patients.intersection(test_patients)) == 0
    assert len(train_df) + len(val_df) + len(test_df) == len(df)


def test_deduplicate_studies():
    """Verify duplicate study records are cleanly filtered."""
    df = pd.DataFrame({
        "patient_id": ["P1", "P1", "P2"],
        "study_id": ["S100", "S100", "S101"],
        "score": [0.8, 0.85, 0.4]
    })
    deduped = deduplicate_studies(df, study_col="study_id")
    assert len(deduped) == 2
    assert list(deduped["study_id"]) == ["S100", "S101"]


def test_compute_binary_metrics():
    """Verify sensitivity, specificity, and AUROC calculations."""
    y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    y_prob = np.array([0.9, 0.8, 0.4, 0.1, 0.2, 0.3, 0.6, 0.05])
    metrics = compute_binary_metrics(y_true, y_prob, threshold=0.5)

    # At thresh 0.5: TP = 2 (0.9, 0.8), FN = 1 (0.4), FP = 1 (0.6), TN = 4
    assert metrics["tp"] == 2
    assert metrics["fn"] == 1
    assert metrics["fp"] == 1
    assert metrics["tn"] == 4
    assert metrics["sensitivity"] == pytest.approx(2 / 3, 0.01)
    assert metrics["specificity"] == pytest.approx(4 / 5, 0.01)
    assert metrics["auroc"] > 0.8


def test_compute_bootstrap_ci():
    """Verify clustered bootstrap produces valid confidence bounds."""
    rng = np.random.RandomState(42)
    n = 200
    df = pd.DataFrame({
        "patient_id": [f"P_{i:02d}" for i in rng.randint(0, 40, size=n)],
        "y_true": rng.choice([0, 1], size=n, p=[0.8, 0.2]),
        "y_prob": rng.rand(n)
    })
    point, lower, upper = compute_bootstrap_ci(
        df,
        patient_col="patient_id",
        y_true_col="y_true",
        y_prob_col="y_prob",
        threshold=0.5,
        metric_name="auroc",
        n_bootstraps=100
    )
    assert 0.0 <= lower <= point <= upper <= 1.0


def test_calculate_required_sample_size():
    """Verify sample size calculation for target sensitivity non-inferiority."""
    res = calculate_required_sample_size(target_sensitivity=0.95, margin_of_error=0.03, prevalence=0.10)
    assert res["min_positive_cases_required"] > 100
    assert res["min_total_studies_required"] > 1000
    assert "recommendation" in res
