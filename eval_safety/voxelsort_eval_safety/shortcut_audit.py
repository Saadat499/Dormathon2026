"""
VoxelSort: Evaluation & Safety Module
Component: Shortcut & Confounder Audit (P3)

Audits model robustness against known medical imaging shortcuts and spurious correlations:
1. Chest tube / drain presence as a shortcut for Pneumothorax.
2. Radiograph projection (AP portable bedside vs PA ambulatory) as an illness surrogate.
3. Support devices / ICU monitoring hardware.
4. Acquisition site / manufacturer tokens.

Calculates subgroup disparities to ensure the model detects true pathology in acute,
untreated patients rather than relying on medical interventions.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from protocol import compute_binary_metrics


class ShortcutAuditor:
    """
    Evaluates whether the triage model relies on confounding shortcuts
    by auditing performance across clinically stratified cohorts.
    """

    @staticmethod
    def audit_chest_tube_shortcut_on_pneumothorax(
        df: pd.DataFrame,
        y_true_col: str = "true_pneumothorax",
        y_prob_col: str = "pred_pneumothorax",
        confounder_col: str = "has_chest_tube",
        threshold: float = 0.35
    ) -> Dict[str, Any]:
        """
        Critical safety audit: Pneumothorax sensitivity MUST remain high in patients
        WITHOUT a chest tube (acute, untreated cases requiring immediate emergency triage).
        """
        # Overall metrics
        overall = compute_binary_metrics(df[y_true_col].values, df[y_prob_col].values, threshold=threshold)

        # Subgroup with chest tube
        sub_with_tube = df[df[confounder_col] == 1]
        metrics_with_tube = compute_binary_metrics(
            sub_with_tube[y_true_col].values,
            sub_with_tube[y_prob_col].values,
            threshold=threshold
        ) if len(sub_with_tube) > 0 else {}

        # Subgroup WITHOUT chest tube (Crucial cohort!)
        sub_no_tube = df[df[confounder_col] == 0]
        metrics_no_tube = compute_binary_metrics(
            sub_no_tube[y_true_col].values,
            sub_no_tube[y_prob_col].values,
            threshold=threshold
        ) if len(sub_no_tube) > 0 else {}

        sens_with = metrics_with_tube.get("sensitivity", 0.0)
        sens_without = metrics_no_tube.get("sensitivity", 0.0)
        sens_gap = abs(sens_with - sens_without)

        # Safety rule: Untreated sensitivity must not drop below 90%
        audit_passed = (sens_without >= 0.90) and (sens_gap <= 0.12)

        return {
            "audit_target": "Pneumothorax without Chest Tube (Acute Untreated)",
            "overall_sensitivity": round(overall.get("sensitivity", 0.0), 4),
            "with_chest_tube_sensitivity": round(sens_with, 4),
            "without_chest_tube_sensitivity": round(sens_without, 4),
            "sensitivity_disparity_gap": round(sens_gap, 4),
            "sample_counts": {
                "total": len(df),
                "with_tube": len(sub_with_tube),
                "without_tube": len(sub_no_tube),
            },
            "audit_passed": audit_passed,
            "clinical_interpretation": (
                "PASS: Model maintains high sensitivity on acute patients without prior chest tube drainage."
                if audit_passed else
                "WARNING: Model exhibits significant performance drop when chest tube is absent, suggesting shortcut reliance."
            )
        }

    @staticmethod
    def audit_view_position_confounder(
        df: pd.DataFrame,
        finding_col: str,
        prob_col: str,
        view_col: str = "view_position",  # 'AP' vs 'PA'
        threshold: float = 0.50
    ) -> Dict[str, Any]:
        """
        Audits performance consistency across AP (anteroposterior/bedside) and PA (posteroanterior) views.
        """
        views = df[view_col].unique()
        view_metrics: Dict[str, Any] = {}

        for v in views:
            sub = df[df[view_col] == v]
            if len(sub) > 0:
                view_metrics[str(v)] = compute_binary_metrics(
                    sub[finding_col].values,
                    sub[prob_col].values,
                    threshold=threshold
                )

        # Check gap between AP and PA if both present
        auroc_ap = view_metrics.get("AP", {}).get("auroc", 0.0)
        auroc_pa = view_metrics.get("PA", {}).get("auroc", 0.0)
        gap = abs(auroc_ap - auroc_pa)

        return {
            "finding": finding_col,
            "view_metrics": view_metrics,
            "auroc_disparity_gap": round(gap, 4),
            "audit_passed": gap <= 0.10,
            "interpretation": (
                "Robust: Performance consistent across portable bedside (AP) and ambulatory (PA) projections."
                if gap <= 0.10 else
                "Potential projection bias: Substantial discrepancy between AP and PA views."
            )
        }
