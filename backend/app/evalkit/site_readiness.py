"""
VoxelSort: Evaluation & Safety Module
Component: Site Readiness & Distribution Drift Assessment (P3)

Implements:
- Population Stability Index (PSI), Kolmogorov-Smirnov (KS) test, and Wasserstein distance.
- Pre-deployment / continuous monitoring Site Readiness Verdict Rules (GO / WARNING / NO-GO).
- Automatic Fallback Switch: Reverts worklist to standard FIFO when distribution shifts dangerously.
- OOD detector validation and safety audit (reviewing ML Lead's OOD system).
"""

from enum import Enum
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance


class SiteReadinessVerdict(str, Enum):
    GO = "GO"                # Green: Safe to deploy automated AI-triage
    WARNING = "WARNING"      # Yellow: Moderate shift, run in shadow mode or audited triage
    NO_GO = "NO-GO"          # Red: Severe shift, automated fallback to standard FIFO worklist


def calculate_psi(
    reference_scores: np.ndarray,
    target_scores: np.ndarray,
    num_bins: int = 10,
    epsilon: float = 1e-4
) -> float:
    """
    Computes Population Stability Index (PSI) between reference and target cohorts.
    PSI < 0.10: No significant change
    0.10 <= PSI < 0.25: Moderate shift
    PSI >= 0.25: Significant shift (action required)
    """
    ref = np.asarray(reference_scores, dtype=float)
    tgt = np.asarray(target_scores, dtype=float)

    if len(ref) == 0 or len(tgt) == 0:
        return 0.0

    # Quantile bins from reference
    quantiles = np.linspace(0.0, 100.0, num_bins + 1)
    bin_edges = np.percentile(ref, quantiles)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    ref_counts, _ = np.histogram(ref, bins=bin_edges)
    tgt_counts, _ = np.histogram(tgt, bins=bin_edges)

    ref_pct = (ref_counts + epsilon) / (np.sum(ref_counts) + epsilon * num_bins)
    tgt_pct = (tgt_counts + epsilon) / (np.sum(tgt_counts) + epsilon * num_bins)

    psi_val = np.sum((tgt_pct - ref_pct) * np.log(tgt_pct / ref_pct))
    return float(np.round(psi_val, 4))


class SiteReadinessAssessor:
    """
    Monitors new clinical site data compatibility, computes drift indicators,
    and enforces safety fallback verdicts.
    """

    def __init__(
        self,
        psi_warning_threshold: float = 0.10,
        psi_nogo_threshold: float = 0.25,
        ood_warning_threshold: float = 0.025,  # 2.5%
        ood_nogo_threshold: float = 0.060,     # 6.0%
        ks_warning_threshold: float = 0.15,
    ):
        self.psi_warning_threshold = psi_warning_threshold
        self.psi_nogo_threshold = psi_nogo_threshold
        self.ood_warning_threshold = ood_warning_threshold
        self.ood_nogo_threshold = ood_nogo_threshold
        self.ks_warning_threshold = ks_warning_threshold

    def evaluate_finding_drift(
        self,
        ref_probabilities: np.ndarray,
        target_probabilities: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculates drift metrics (PSI, KS statistic, p-value, Wasserstein distance) for a single finding.
        """
        psi = calculate_psi(ref_probabilities, target_probabilities)
        ks_res = ks_2samp(ref_probabilities, target_probabilities)
        wd = float(wasserstein_distance(ref_probabilities, target_probabilities))

        return {
            "psi": round(psi, 4),
            "ks_statistic": round(float(ks_res.statistic), 4),
            "ks_pvalue": round(float(ks_res.pvalue), 6),
            "wasserstein_distance": round(wd, 4),
        }

    def evaluate_site(
        self,
        site_name: str,
        ref_findings_dict: Dict[str, np.ndarray],
        target_findings_dict: Dict[str, np.ndarray],
        target_ood_flags: np.ndarray
    ) -> Dict[str, Any]:
        """
        Evaluates complete Site Readiness across all findings and OOD rate.
        Determines final verdict: GO, WARNING, or NO-GO.
        """
        ood_rate = float(np.mean(target_ood_flags)) if len(target_ood_flags) > 0 else 0.0
        finding_drift_reports: Dict[str, Any] = {}
        max_psi = 0.0
        max_ks = 0.0

        for finding_name, target_probs in target_findings_dict.items():
            if finding_name in ref_findings_dict:
                drift = self.evaluate_finding_drift(ref_findings_dict[finding_name], target_probs)
                finding_drift_reports[finding_name] = drift
                max_psi = max(max_psi, drift["psi"])
                max_ks = max(max_ks, drift["ks_statistic"])

        # Verdict rules
        verdict: SiteReadinessVerdict
        verdict_reasons: List[str] = []

        if ood_rate >= self.ood_nogo_threshold:
            verdict = SiteReadinessVerdict.NO_GO
            verdict_reasons.append(
                f"OOD rate ({ood_rate*100:.1f}%) exceeds safety ceiling ({self.ood_nogo_threshold*100:.1f}%)."
            )
        elif max_psi >= self.psi_nogo_threshold:
            verdict = SiteReadinessVerdict.NO_GO
            verdict_reasons.append(
                f"Maximum PSI ({max_psi:.3f}) exceeds critical stability threshold ({self.psi_nogo_threshold:.2f})."
            )
        elif ood_rate >= self.ood_warning_threshold:
            verdict = SiteReadinessVerdict.WARNING
            verdict_reasons.append(
                f"Elevated OOD rate ({ood_rate*100:.1f}% >= {self.ood_warning_threshold*100:.1f}%)."
            )
        elif max_psi >= self.psi_warning_threshold or max_ks >= self.ks_warning_threshold:
            verdict = SiteReadinessVerdict.WARNING
            verdict_reasons.append(
                f"Moderate score drift detected (Max PSI={max_psi:.3f}, Max KS={max_ks:.3f})."
            )
        else:
            verdict = SiteReadinessVerdict.GO
            verdict_reasons.append("All drift indicators and OOD rates are within clinical safety bounds.")

        # Safety fallback action
        fallback_active = (verdict == SiteReadinessVerdict.NO_GO)

        return {
            "site_name": site_name,
            "verdict": verdict.value,
            "fallback_active": fallback_active,
            "ood_rate": round(ood_rate, 4),
            "max_psi": round(max_psi, 4),
            "max_ks": round(max_ks, 4),
            "verdict_reasons": verdict_reasons,
            "action_guidance": (
                "AUTOMATED FALLBACK ACTIVATED: Disabling AI-Triage priority queue. "
                "Worklist reverted to standard FIFO reading order."
                if fallback_active else
                "AI-Triage operational under continuous drift surveillance."
            ),
            "findings_drift": finding_drift_reports,
        }


def audit_ood_detector_performance(
    y_is_ood_true: np.ndarray,
    ood_detector_scores: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Independent safety review of ML Lead (A)'s OOD detector on test cases
    (e.g., rotated X-rays, non-chest scans, foreign implants, corrupted DICOMs).
    """
    y_true = np.asarray(y_is_ood_true, dtype=int)
    scores = np.asarray(ood_detector_scores, dtype=float)
    y_pred = (scores >= threshold).astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "ood_detection_recall": round(float(recall), 4),
        "ood_precision": round(float(precision), 4),
        "ood_false_alarm_rate": round(float(false_alarm_rate), 4),
        "tp_detected_anomalies": tp,
        "fp_false_alarms": fp,
        "fn_missed_anomalies": fn,
        "tn_normal_cases": tn,
        "clinical_safety_verdict": "PASS" if recall >= 0.90 and false_alarm_rate <= 0.05 else "NEEDS_REFINEMENT"
    }
