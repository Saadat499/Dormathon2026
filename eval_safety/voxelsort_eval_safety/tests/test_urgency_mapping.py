"""
Tests for Urgency Mapping, Priority Assignment, and Harm-Bounded Safety Overrides.
"""

import pytest
from urgency_mapping import PriorityLevel, UrgencyClassifier, CLINICAL_FINDING_RULES


def test_clinical_finding_rules_presence():
    """Verify core CheXpert high-stakes findings are defined."""
    assert "Pneumothorax" in CLINICAL_FINDING_RULES
    assert CLINICAL_FINDING_RULES["Pneumothorax"].target_priority == PriorityLevel.STAT
    assert CLINICAL_FINDING_RULES["Pneumothorax"].is_life_threatening is True
    assert CLINICAL_FINDING_RULES["Pneumonia"].target_priority == PriorityLevel.URGENT
    assert CLINICAL_FINDING_RULES["Atelectasis"].target_priority == PriorityLevel.ROUTINE


def test_stat_priority_triggered():
    """Verify high Pneumothorax probability triggers STAT priority."""
    classifier = UrgencyClassifier()
    findings = {
        "Pneumothorax": 0.85,
        "Pneumonia": 0.10,
        "Atelectasis": 0.20,
    }
    result = classifier.classify_study("ST_001", findings)
    assert result["priority"] == PriorityLevel.STAT.value
    assert result["study_id"] == "ST_001"
    assert result["ood_flag"] is False
    assert result["urgency_score"] > 0.5


def test_ood_safety_override():
    """Verify OOD detection automatically escalates study to STAT fallback (harm-bounded design)."""
    classifier = UrgencyClassifier()
    # Low probability for all diseases, but OOD flag is True
    findings = {
        "Pneumothorax": 0.05,
        "Pneumonia": 0.05,
        "Atelectasis": 0.10,
    }
    result = classifier.classify_study("ST_OOD_999", findings, ood_flag=True)
    assert result["priority"] == PriorityLevel.STAT.value
    assert result["ood_flag"] is True
    assert result["safety_override_triggered"] is True
    assert "OOD flag active" in result["override_reason"]


def test_data_contract_format():
    """Verify output strictly matches the agreed team data contract:
    {study_id, findings: {name: probability}, priority, confidence_band, ood_flag, heatmap_path}
    """
    classifier = UrgencyClassifier()
    findings = {"Pneumothorax": 0.12, "Pneumonia": 0.75}
    cbands = {"Pneumothorax": (0.08, 0.18), "Pneumonia": (0.68, 0.82)}
    res = classifier.classify_study(
        study_id="ST_CONTRACT_1",
        findings_prob=findings,
        ood_flag=False,
        heatmap_path="/heatmaps/ST_CONTRACT_1.png",
        confidence_bands=cbands
    )

    required_keys = {"study_id", "findings", "priority", "confidence_band", "ood_flag", "heatmap_path"}
    assert required_keys.issubset(res.keys())
    assert res["findings"] == findings
    assert res["confidence_band"] == cbands
    assert res["priority"] == PriorityLevel.URGENT.value
