"""
Tests for Site Readiness Drift Assessment, Verdict Rules, Fallback Switch, and Shortcut Auditing.
"""

import pytest
import numpy as np
import pandas as pd
from site_readiness import (
    calculate_psi,
    SiteReadinessAssessor,
    SiteReadinessVerdict,
    audit_ood_detector_performance
)
from shortcut_audit import ShortcutAuditor


def test_psi_zero_for_identical_distributions():
    """Verify PSI is near 0 for identical cohorts."""
    rng = np.random.RandomState(42)
    sample = rng.normal(0.5, 0.15, size=500)
    psi = calculate_psi(sample, sample)
    assert psi < 0.05


def test_site_readiness_verdicts():
    """Verify GO, WARNING, and NO-GO verdict logic."""
    assessor = SiteReadinessAssessor()
    rng = np.random.RandomState(42)
    ref = {"Pneumothorax": rng.beta(2, 5, size=400)}

    # Same distribution -> GO
    target_clean = {"Pneumothorax": rng.beta(2, 5, size=400)}
    ood_clean = np.zeros(400)
    rep_go = assessor.evaluate_site("Site_Safe", ref, target_clean, ood_clean)
    assert rep_go["verdict"] == SiteReadinessVerdict.GO.value
    assert rep_go["fallback_active"] is False

    # Severe OOD rate (e.g. 10%) -> NO-GO with automated fallback
    ood_bad = (rng.rand(400) < 0.10).astype(int)
    rep_nogo = assessor.evaluate_site("Site_Unsafe", ref, target_clean, ood_bad)
    assert rep_nogo["verdict"] == SiteReadinessVerdict.NO_GO.value
    assert rep_nogo["fallback_active"] is True
    assert "AUTOMATED FALLBACK ACTIVATED" in rep_nogo["action_guidance"]


def test_ood_detector_performance_audit():
    """Verify OOD detector metrics and safety decision rule."""
    y_true_ood = np.array([1]*20 + [0]*80)
    scores_ood = np.array([0.9]*19 + [0.1]*1 + [0.05]*78 + [0.85]*2)

    audit = audit_ood_detector_performance(y_true_ood, scores_ood, threshold=0.5)
    assert audit["ood_detection_recall"] >= 0.90
    assert audit["clinical_safety_verdict"] == "PASS"


def test_shortcut_audit_chest_tube():
    """Verify shortcut auditor computes disparity gap on acute untreated pneumothoraces."""
    df = pd.DataFrame({
        "true_pneumothorax": [1]*20 + [0]*80,
        "pred_pneumothorax": [0.85]*18 + [0.20]*2 + [0.10]*75 + [0.70]*5,
        "has_chest_tube": [1]*10 + [0]*10 + [1]*5 + [0]*75
    })
    res = ShortcutAuditor.audit_chest_tube_shortcut_on_pneumothorax(df, threshold=0.5)
    assert "overall_sensitivity" in res
    assert "without_chest_tube_sensitivity" in res
    assert "sensitivity_disparity_gap" in res
    assert "audit_passed" in res
