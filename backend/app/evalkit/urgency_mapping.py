"""
VoxelSort: Evaluation & Safety Module
Component: Urgency Mapping & Safety Decision Rules (P0)

Defines clinical triage priority rules, compound risk scoring, and harm-bounded
safety overrides (e.g., OOD triggers, uncertainty elevation).
Reviewed against emergency radiology triaging protocols.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple
import numpy as np


class PriorityLevel(str, Enum):
    STAT = "STAT"          # Immediate radiologist review (e.g. < 30 min SLA)
    URGENT = "URGENT"      # Expedited review (e.g. < 2 hour SLA)
    ROUTINE = "ROUTINE"    # Standard worklist queue (e.g. < 12-24 hour SLA)


@dataclass
class FindingRule:
    """Clinical rule specifying finding severity, default operational threshold, and priority."""
    name: str
    target_priority: PriorityLevel
    default_threshold: float
    is_life_threatening: bool
    clinical_rationale: str


# Comprehensive Clinical Urgency Catalog for Chest X-Ray findings
CLINICAL_FINDING_RULES: Dict[str, FindingRule] = {
    "Pneumothorax": FindingRule(
        name="Pneumothorax",
        target_priority=PriorityLevel.STAT,
        default_threshold=0.35,  # Operating point tuned for >=95% sensitivity
        is_life_threatening=True,
        clinical_rationale="Risk of rapid tension pneumothorax and circulatory collapse."
    ),
    "Massive Pleural Effusion": FindingRule(
        name="Massive Pleural Effusion",
        target_priority=PriorityLevel.STAT,
        default_threshold=0.40,
        is_life_threatening=True,
        clinical_rationale="Severe ventilatory compromise and mediastinal shift risk."
    ),
    "Wide Mediastinum": FindingRule(
        name="Wide Mediastinum",
        target_priority=PriorityLevel.STAT,
        default_threshold=0.30,
        is_life_threatening=True,
        clinical_rationale="Possible thoracic aortic dissection or acute mediastinal hemorrhage."
    ),
    "Acute Pulmonary Edema": FindingRule(
        name="Acute Pulmonary Edema",
        target_priority=PriorityLevel.STAT,
        default_threshold=0.45,
        is_life_threatening=True,
        clinical_rationale="Acute decompensated heart failure requiring immediate intervention."
    ),
    "Pneumonia": FindingRule(
        name="Pneumonia",
        target_priority=PriorityLevel.URGENT,
        default_threshold=0.50,
        is_life_threatening=False,
        clinical_rationale="Infectious consolidation needing timely antimicrobial coverage."
    ),
    "Pleural Effusion": FindingRule(
        name="Pleural Effusion",
        target_priority=PriorityLevel.URGENT,
        default_threshold=0.50,
        is_life_threatening=False,
        clinical_rationale="Moderate fluid collection requiring clinical workup."
    ),
    "Lung Lesion / Mass": FindingRule(
        name="Lung Lesion / Mass",
        target_priority=PriorityLevel.URGENT,
        default_threshold=0.55,
        is_life_threatening=False,
        clinical_rationale="Suspected malignancy or nodule needing urgent oncology pathway."
    ),
    "Rib Fracture": FindingRule(
        name="Rib Fracture",
        target_priority=PriorityLevel.URGENT,
        default_threshold=0.50,
        is_life_threatening=False,
        clinical_rationale="Acute trauma requiring pain control and pneumothorax surveillance."
    ),
    "Cardiomegaly": FindingRule(
        name="Cardiomegaly",
        target_priority=PriorityLevel.ROUTINE,
        default_threshold=0.60,
        is_life_threatening=False,
        clinical_rationale="Chronic cardiothoracic ratio elevation without acute pulmonary edema."
    ),
    "Atelectasis": FindingRule(
        name="Atelectasis",
        target_priority=PriorityLevel.ROUTINE,
        default_threshold=0.60,
        is_life_threatening=False,
        clinical_rationale="Common non-critical volume loss in dependent lung zones."
    ),
    "Support Devices": FindingRule(
        name="Support Devices",
        target_priority=PriorityLevel.ROUTINE,
        default_threshold=0.60,
        is_life_threatening=False,
        clinical_rationale="Monitoring endotracheal tubes, PICC lines, or pacemakers."
    ),
    "No Finding": FindingRule(
        name="No Finding",
        target_priority=PriorityLevel.ROUTINE,
        default_threshold=0.50,
        is_life_threatening=False,
        clinical_rationale="Unremarkable radiograph."
    ),
}


class UrgencyClassifier:
    """
    Applies calibrated thresholds and harm-bounded safety overrides to assign
    triage priority and produce data contract-compliant predictions.
    """

    def __init__(
        self,
        custom_thresholds: Optional[Dict[str, float]] = None,
        ood_fallback_priority: PriorityLevel = PriorityLevel.STAT,
        high_uncertainty_threshold: float = 0.85,  # Normalized binary entropy
    ):
        self.rules = CLINICAL_FINDING_RULES.copy()
        self.thresholds: Dict[str, float] = {}
        for finding, rule in self.rules.items():
            self.thresholds[finding] = rule.default_threshold

        if custom_thresholds:
            for k, v in custom_thresholds.items():
                self.thresholds[k] = float(v)

        self.ood_fallback_priority = ood_fallback_priority
        self.high_uncertainty_threshold = high_uncertainty_threshold

    @staticmethod
    def binary_entropy(p: float) -> float:
        """Normalized Shannon entropy for binary probability p in [0, 1]."""
        p = max(1e-6, min(1.0 - 1e-6, p))
        return -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))

    def compute_composite_urgency_score(
        self,
        probabilities: Dict[str, float]
    ) -> float:
        """
        Calculates a continuous urgency score in [0.0, 1.0].
        Weights STAT conditions significantly higher than URGENT/ROUTINE.
        """
        score = 0.0
        # Multipliers by severity tier
        weights = {
            PriorityLevel.STAT: 1.0,
            PriorityLevel.URGENT: 0.45,
            PriorityLevel.ROUTINE: 0.1,
        }

        for finding, prob in probabilities.items():
            rule = self.rules.get(finding)
            if not rule:
                continue
            w = weights.get(rule.target_priority, 0.1)
            # Add non-linear boost if above the operational threshold
            thresh = self.thresholds.get(finding, rule.default_threshold)
            above = 1.3 if prob >= thresh else 0.7
            score += w * prob * above

        # Sigmoid compression to [0, 1]
        return float(1.0 / (1.0 + np.exp(-1.5 * (score - 0.7))))

    def classify_study(
        self,
        study_id: str,
        findings_prob: Dict[str, float],
        ood_flag: bool = False,
        heatmap_path: Optional[str] = None,
        confidence_bands: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> Dict[str, Any]:
        """
        Determines the priority of a study and outputs the official VoxelSort data contract:
        {
            "study_id": str,
            "findings": Dict[str, float],
            "priority": "STAT" | "URGENT" | "ROUTINE",
            "confidence_band": Dict[str, List[float]],
            "ood_flag": bool,
            "heatmap_path": Optional[str],
            "urgency_score": float,
            "safety_override_triggered": bool,
            "override_reason": Optional[str]
        }
        """
        # Safety Check 1: Out-of-Distribution Detection (Harm-bounded fallback)
        if ood_flag:
            return {
                "study_id": study_id,
                "findings": findings_prob,
                "priority": self.ood_fallback_priority.value,
                "confidence_band": confidence_bands or {},
                "ood_flag": True,
                "heatmap_path": heatmap_path,
                "urgency_score": 1.0,
                "safety_override_triggered": True,
                "override_reason": f"OOD flag active: auto-escalated to {self.ood_fallback_priority.value} for clinician inspection",
            }

        detected_priorities: List[PriorityLevel] = []
        max_entropy = 0.0

        for finding, prob in findings_prob.items():
            thresh = self.thresholds.get(finding, 0.50)
            rule = self.rules.get(finding)
            ent = self.binary_entropy(prob)
            if ent > max_entropy:
                max_entropy = ent

            if prob >= thresh and rule:
                detected_priorities.append(rule.target_priority)

        # Safety Check 2: Severe ambiguity/uncertainty on life-threatening findings
        # If model is near maximal uncertainty (p ~ 0.5) on a STAT finding, do not bury it in ROUTINE
        for stat_finding in ["Pneumothorax", "Massive Pleural Effusion", "Wide Mediastinum"]:
            prob = findings_prob.get(stat_finding, 0.0)
            if 0.30 <= prob < self.thresholds.get(stat_finding, 0.35):
                detected_priorities.append(PriorityLevel.URGENT)

        # Priority selection: STAT > URGENT > ROUTINE
        assigned_priority: PriorityLevel
        override_triggered = False
        override_reason = None

        if PriorityLevel.STAT in detected_priorities:
            assigned_priority = PriorityLevel.STAT
        elif PriorityLevel.URGENT in detected_priorities:
            assigned_priority = PriorityLevel.URGENT
        else:
            assigned_priority = PriorityLevel.ROUTINE

        composite_score = self.compute_composite_urgency_score(findings_prob)

        return {
            "study_id": study_id,
            "findings": findings_prob,
            "priority": assigned_priority.value,
            "confidence_band": confidence_bands or {},
            "ood_flag": False,
            "heatmap_path": heatmap_path,
            "urgency_score": round(composite_score, 4),
            "safety_override_triggered": override_triggered,
            "override_reason": override_reason,
        }
