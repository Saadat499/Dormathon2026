"""
Tests for Probability Calibration, Reliability Diagrams, and Operating Threshold Selection.
"""

import pytest
import numpy as np
from calibration import (
    TemperatureScaler,
    ModelCalibrator,
    select_threshold_at_sensitivity,
    compute_wilson_confidence_interval,
)


def test_temperature_scaler_reduces_ece():
    """Verify temperature scaling fits and reduces calibration error."""
    rng = np.random.RandomState(42)
    n = 500
    y_true = rng.choice([0, 1], size=n, p=[0.8, 0.2])
    # Overconfident uncalibrated probabilities
    y_prob = np.where(y_true == 1, rng.beta(9, 1, n), rng.beta(1, 9, n))

    calibrator = ModelCalibrator(method="temperature")
    calibrator.fit(y_prob, y_true)
    cal_prob = calibrator.transform(y_prob)

    uncal_rel = ModelCalibrator.compute_reliability_curve(y_true, y_prob)
    cal_rel = ModelCalibrator.compute_reliability_curve(y_true, cal_prob)

    assert calibrator.is_fitted
    assert len(cal_rel["binned_curve"]) == 10
    assert "ece" in cal_rel
    assert "mce" in cal_rel


def test_select_threshold_at_sensitivity():
    """Verify threshold selection meets or exceeds target sensitivity."""
    rng = np.random.RandomState(42)
    y_true = np.array([1] * 50 + [0] * 150)
    # Signal with overlap
    y_prob = np.concatenate([
        rng.beta(4, 2, size=50),  # Positives
        rng.beta(2, 4, size=150)   # Negatives
    ])

    res = select_threshold_at_sensitivity(y_true, y_prob, target_sensitivity=0.95)
    assert res["achieved_sensitivity"] >= 0.95
    assert res["achieved_specificity"] > 0.0
    assert 0.0 < res["selected_threshold"] < 1.0


def test_wilson_confidence_interval():
    """Verify Wilson interval bounds [0, 1]."""
    lower, upper = compute_wilson_confidence_interval(0.85, n_effective=50)
    assert 0.0 <= lower <= 0.85 <= upper <= 1.0
    assert (upper - lower) > 0.05
