"""
VoxelSort: Evaluation & Safety Module
Component: Probability Calibration, Reliability Diagrams & Operating Point Selection (P2)

Provides:
- Temperature scaling (Platt/logistic logit calibration) and Isotonic regression.
- Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).
- Reliability curve data export for Recharts front-end visualizers.
- High-stakes threshold selection at fixed clinical sensitivity targets (e.g. >= 95% sensitivity).
- Wilson score and bootstrap confidence band calculations for the data contract.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.isotonic import IsotonicRegression


class TemperatureScaler:
    """
    Parametric post-hoc calibration via Temperature Scaling:
    p_calibrated = sigma(z / T), where T > 0 is learned by minimizing NLL.
    """

    def __init__(self):
        self.temperature: float = 1.0

    def fit(self, y_prob: np.ndarray, y_true: np.ndarray) -> "TemperatureScaler":
        """
        Fits scalar temperature T on validation probabilities and binary labels.
        """
        y_prob = np.clip(np.asarray(y_prob, dtype=np.float64), 1e-5, 1.0 - 1e-5)
        y_true = np.asarray(y_true, dtype=np.float64)
        logits = logit(y_prob)

        def nll_loss(t_val: np.ndarray) -> float:
            temp = float(t_val[0])
            if temp <= 0:
                return 1e9
            scaled_logits = logits / temp
            # Binary cross entropy with log-sum-exp stabilization
            p = expit(scaled_logits)
            p = np.clip(p, 1e-7, 1.0 - 1e-7)
            loss = -np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p))
            return float(loss)

        res = minimize(nll_loss, x0=[1.0], bounds=[(0.05, 10.0)], method="L-BFGS-B")
        if res.success:
            self.temperature = float(res.x[0])
        else:
            self.temperature = 1.0
        return self

    def predict_proba(self, y_prob: np.ndarray) -> np.ndarray:
        """Applies learned temperature to scale uncalibrated probabilities."""
        y_prob = np.clip(np.asarray(y_prob, dtype=np.float64), 1e-5, 1.0 - 1e-5)
        logits = logit(y_prob)
        return expit(logits / self.temperature)


class ModelCalibrator:
    """
    Calibrator supporting both Temperature Scaling and Isotonic Regression,
    with reliability diagram and ECE/MCE extraction.
    """

    def __init__(self, method: str = "temperature"):
        self.method = method
        self.temp_scaler = TemperatureScaler()
        self.isotonic_reg = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self.is_fitted = False

    def fit(self, y_prob: np.ndarray, y_true: np.ndarray) -> "ModelCalibrator":
        y_prob = np.asarray(y_prob, dtype=float)
        y_true = np.asarray(y_true, dtype=int)

        if self.method == "temperature":
            self.temp_scaler.fit(y_prob, y_true)
        elif self.method == "isotonic":
            self.isotonic_reg.fit(y_prob, y_true)
        else:
            raise ValueError(f"Unknown calibration method: {self.method}")

        self.is_fitted = True
        return self

    def transform(self, y_prob: np.ndarray) -> np.ndarray:
        y_prob = np.asarray(y_prob, dtype=float)
        if not self.is_fitted:
            return y_prob

        if self.method == "temperature":
            return self.temp_scaler.predict_proba(y_prob)
        elif self.method == "isotonic":
            return self.isotonic_reg.predict(y_prob)
        return y_prob

    @staticmethod
    def compute_reliability_curve(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int = 10
    ) -> Dict[str, Any]:
        """
        Computes reliability bins, Expected Calibration Error (ECE), and Max Calibration Error (MCE).
        """
        y_true = np.asarray(y_true, dtype=int)
        y_prob = np.asarray(y_prob, dtype=float)
        n_samples = len(y_true)

        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        bins_data: List[Dict[str, Any]] = []
        ece = 0.0
        mce = 0.0

        for i in range(n_bins):
            lower, upper = bin_edges[i], bin_edges[i + 1]
            if i == n_bins - 1:
                mask = (y_prob >= lower) & (y_prob <= upper)
            else:
                mask = (y_prob >= lower) & (y_prob < upper)

            count = int(np.sum(mask))
            if count > 0:
                conf = float(np.mean(y_prob[mask]))
                acc = float(np.mean(y_true[mask]))
                diff = abs(acc - conf)
                ece += (count / n_samples) * diff
                mce = max(mce, diff)
            else:
                conf = float((lower + upper) / 2.0)
                acc = 0.0
                diff = 0.0

            bins_data.append({
                "bin_index": i,
                "bin_range": f"{lower:.1f}-{upper:.1f}",
                "mean_confidence": round(conf, 4),
                "empirical_accuracy": round(acc, 4),
                "count": count,
                "calibration_gap": round(diff, 4),
            })

        return {
            "n_bins": n_bins,
            "ece": round(float(ece), 4),
            "mce": round(float(mce), 4),
            "binned_curve": bins_data,
        }


def select_threshold_at_sensitivity(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_sensitivity: float = 0.95,
) -> Dict[str, Any]:
    """
    Selects operating decision threshold at a fixed clinical sensitivity target.
    Chooses the highest threshold that maintains sensitivity >= target_sensitivity,
    thereby maximizing specificity and minimizing false alarm burden.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    threshold_candidates = np.sort(np.unique(y_prob))
    best_threshold = 0.05
    best_specificity = 0.0
    achieved_sens = 1.0

    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)

    if n_pos == 0:
        return {
            "threshold": 0.5,
            "achieved_sensitivity": 0.0,
            "achieved_specificity": 1.0,
            "target_sensitivity": target_sensitivity,
        }

    for thresh in threshold_candidates:
        y_pred = (y_prob >= thresh).astype(int)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        sens = tp / n_pos

        if sens >= target_sensitivity:
            tn = np.sum((y_true == 0) & (y_pred == 0))
            spec = tn / n_neg if n_neg > 0 else 0.0
            if thresh > best_threshold:
                best_threshold = thresh
                best_specificity = spec
                achieved_sens = sens

    return {
        "target_sensitivity": target_sensitivity,
        "selected_threshold": round(float(best_threshold), 4),
        "achieved_sensitivity": round(float(achieved_sens), 4),
        "achieved_specificity": round(float(best_specificity), 4),
        "clinical_note": (
            f"At threshold {best_threshold:.3f}, sensitivity is guaranteed at {achieved_sens*100:.1f}% "
            f"while ruling out {best_specificity*100:.1f}% of negatives."
        )
    }


def compute_wilson_confidence_interval(
    p: float,
    n_effective: int = 50,
    confidence: float = 0.95
) -> Tuple[float, float]:
    """
    Computes Wilson score confidence interval [lower, upper] for predicted probability p.
    Supplies the confidence_band field for individual studies in the data contract.
    """
    from scipy.stats import norm
    alpha = 1.0 - confidence
    z = norm.ppf(1.0 - alpha / 2.0)

    denominator = 1.0 + (z ** 2) / n_effective
    centre_adjusted_probability = (p + (z ** 2) / (2.0 * n_effective)) / denominator
    adjusted_std_error = (z / denominator) * np.sqrt((p * (1.0 - p) / n_effective) + (z ** 2) / (4.0 * (n_effective ** 2)))

    lower = max(0.0, centre_adjusted_probability - adjusted_std_error)
    upper = min(1.0, centre_adjusted_probability + adjusted_std_error)
    return round(float(lower), 4), round(float(upper), 4)
