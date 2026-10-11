"""Decision policy: probabilities -> Raised / Normal / AI unsure.

Rules (matching the project's safety design):
  composite score = max probability over the urgent findings
  score >= t_raise                 -> Raised        (confidence 'high')
  t_unsure <= score < t_raise      -> AI unsure     (confidence 'low')  [never ranked below Normal]
  score < t_unsure                 -> Normal        (confidence 'high')  [keeps its FCFS place, never removed]
  out-of-distribution image        -> AI unsure     (confidence 'none') [not scored]

Two ways to set the thresholds on NIH validation data (Member B / the clinical lead own the final choice):
  mode='budget' (default): Raised = the top `raise_budget` share of scans (e.g. 20%), AI unsure = the next
      `unsure_budget` share. The sensitivity this captures is measured and stored, and checked against
      `min_sensitivity` for the Raised + AI unsure lanes together (a warning, not a silent pass).
  mode='sensitivity': Raised at `target_sens_raise`, AI unsure down to `target_sens_unsure`.
  queue_rank(): rank-based alternative applied to the current queue (top X% of what is waiting now).
A weak model at 90% sensitivity raises most of the queue, so the queue barely changes; a capacity
budget plus ordering by score is where the ranking (AUROC) actually helps.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

from .metrics import operating_point, sigmoid, threshold_for_sensitivity


@dataclass
class Policy:
    findings: list
    urgent: list
    mode: str = "budget"
    raise_budget: float = 0.20
    unsure_budget: float = 0.10
    min_sensitivity: float = 0.60
    target_sens_raise: float = 0.90
    target_sens_unsure: float = 0.97
    t_raise: float = 0.5
    t_unsure: float = 0.25
    fitted: dict = field(default_factory=dict)          # measured on NIH validation when fitted
    temperature: dict = field(default_factory=dict)    # per-finding T from Member B (default 1.0)
    composite: str = "max_of_urgent_probabilities"
    status: str = "PROVISIONAL: fitted by Part A on NIH validation. Urgent list needs clinician review."

    def urgent_idx(self):
        return [self.findings.index(f) for f in self.urgent]

    def probs(self, logits: np.ndarray) -> np.ndarray:
        T = np.array([self.temperature.get(f, 1.0) for f in self.findings], dtype=np.float64)
        return sigmoid(np.asarray(logits, dtype=np.float64) / T)

    def score(self, P: np.ndarray) -> np.ndarray:
        return P[:, self.urgent_idx()].max(axis=1)

    def fit_thresholds(self, y_comp: np.ndarray, s_comp: np.ndarray):
        y, s = np.asarray(y_comp, float), np.asarray(s_comp, float)
        if self.mode == "budget":
            self.t_raise = float(np.quantile(s, 1.0 - self.raise_budget))
            self.t_unsure = float(np.quantile(s, 1.0 - self.raise_budget - self.unsure_budget))
        elif self.mode == "sensitivity":
            self.t_raise = threshold_for_sensitivity(y, s, self.target_sens_raise)
            self.t_unsure = threshold_for_sensitivity(y, s, self.target_sens_unsure)
        else:
            raise ValueError("mode must be 'budget' or 'sensitivity'")
        self.t_unsure = min(self.t_unsure, self.t_raise)
        r, u = operating_point(y, s, self.t_raise), operating_point(y, s, self.t_unsure)
        self.fitted = {"on": "NIH validation", "raised_share": r["flag_rate"], "raised_sensitivity": r["sensitivity"],
                       "raised_ppv": r["ppv"], "raised_or_unsure_share": u["flag_rate"],
                       "raised_or_unsure_sensitivity": u["sensitivity"], "prevalence": float(np.nanmean(y))}
        return self

    def warnings(self) -> list:
        out = []
        if self.t_unsure >= self.t_raise:
            out.append("t_unsure == t_raise: no 'AI unsure' score band.")
        sens = self.fitted.get("raised_or_unsure_sensitivity")
        if sens is not None and sens < self.min_sensitivity:
            out.append(f"Raised + AI unsure lanes capture {sens:.0%} of urgent scans on validation, below "
                       f"min_sensitivity {self.min_sensitivity:.0%}. Widen the budget or get clinical sign-off.")
        return out

    def decide(self, P: np.ndarray, ood_flag=None) -> dict:
        s = self.score(P)
        pr = np.where(s >= self.t_raise, "Raised", np.where(s >= self.t_unsure, "AI unsure", "Normal")).astype(object)
        band = np.where((s >= self.t_unsure) & (s < self.t_raise), "low", "high").astype(object)
        if ood_flag is not None:
            o = np.asarray(ood_flag, bool)
            pr[o], band[o] = "AI unsure", "none"
        return {"priority": pr, "confidence_band": band, "composite_score": s}

    def queue_rank(self, scores, ood_flag=None) -> np.ndarray:
        """Rank-based lanes for the CURRENT queue (Member C calls this whenever the queue changes):
        Raised = top `raise_budget` share of the scored scans waiting now, AI unsure = the next
        `unsure_budget` share, plus every OOD scan. Unlike the fixed cut-off, the Raised share stays at
        the budget at any site (a fixed NIH cut-off raised 89% of CheXpert, whose scores run higher).
        Ties are broken by arrival order, so pass scores in arrival order."""
        s = np.asarray(scores, float)
        o = np.zeros(len(s), bool) if ood_flag is None else np.asarray(ood_flag, bool)
        pr = np.full(len(s), "Normal", dtype=object)
        idx = np.flatnonzero(~o)
        order = idx[np.argsort(-s[idx], kind="stable")]
        n_r = int(np.ceil(self.raise_budget * len(idx)))
        n_u = int(np.ceil((self.raise_budget + self.unsure_budget) * len(idx))) - n_r
        pr[order[:n_r]] = "Raised"
        pr[order[n_r:n_r + n_u]] = "AI unsure"
        pr[o] = "AI unsure"
        return pr

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path) -> "Policy":
        d = json.loads(Path(path).read_text())
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})
