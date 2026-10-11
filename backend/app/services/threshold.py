"""Safety vs Speed: preview and guarded change of the decision threshold.

Rules (contract + safety rule 6):
  * a preview changes nothing;
  * a change needs `confirm: true`;
  * a change is refused if the sensitivity it gives on the REFERENCE (validation) split is below
    `min_sensitivity_target`. The held-out test split is reported, never used to choose a threshold;
  * every change is written to the audit log.
"Clinical lead only" cannot be enforced without a login: the confirmation and the log are the only controls.
"""
from __future__ import annotations

from typing import Optional

from ..context import SCALE_NOTE, AppContext
from ..errors import ApiError
from . import audit

MIN_VALUE, MAX_VALUE = 0.01, 0.99


def _need_predictions(ctx: AppContext) -> None:
    if not ctx.predictions.available:
        raise ApiError(503, "Saved predictions are not available on this server, so thresholds cannot be previewed or changed. "
                            + " ".join(ctx.predictions.problems))


def _check_value(value: float) -> float:
    if value != value or not MIN_VALUE <= value <= MAX_VALUE:
        raise ApiError(400, f"value must be between {MIN_VALUE} and {MAX_VALUE}")
    return float(value)


def info(ctx: AppContext) -> dict:
    lanes = ctx.lanes()
    out = {
        "value": ctx.threshold,
        "min_sensitivity_target": ctx.target,
        "scale": SCALE_NOTE,
        "score_cutoff": lanes.t_raise,
        "unsure_score_cutoff": lanes.t_unsure,
        "unsure_budget": ctx.unsure_budget,
        "cutoff_source": ctx.lanes_source(),
        "data_source": ctx.predictions.source,
        "data_is_real": ctx.predictions.is_real,
    }
    if ctx.predictions.available:
        out["max_allowed_value"] = ctx.predictions.highest_value_meeting(ctx.target)
    return out


def preview(ctx: AppContext, value: float) -> dict:
    _need_predictions(ctx)
    value = _check_value(value)
    op = ctx.predictions.operating_point(value, ctx.unsure_budget)
    op.update(
        min_sensitivity_target=ctx.target,
        meets_target=op["sensitivity"] is not None and op["sensitivity"] >= ctx.target,
        scale=SCALE_NOTE, unsure_budget=ctx.unsure_budget,
        data_source=ctx.predictions.source, data_is_real=ctx.predictions.is_real,
        model_version=ctx.predictions.model_version,
        note=("Measured counts on saved predictions, not a guarantee: error is never zero in practice, "
              "and public-dataset prevalence differs from a hospital worklist."
              + ("" if ctx.predictions.is_real else " NOT REAL MODEL OUTPUT: the saved predictions on this server are a test fixture.")),
    )
    return op


def update(ctx: AppContext, value: float, confirm: bool, actor: Optional[str]) -> dict:
    if confirm is not True:
        raise ApiError(400, "confirm must be true to change the threshold.")
    _need_predictions(ctx)
    value = _check_value(value)
    op = ctx.predictions.operating_point(value, ctx.unsure_budget)
    sens = op["sensitivity"]
    if sens is None:
        raise ApiError(400, "The saved predictions have no urgent scans, so sensitivity at this threshold cannot be checked.")
    if sens < ctx.target:
        raise ApiError(400, f"Sensitivity {sens * 100:.1f}% at threshold {value:.2f} is below the minimum target of "
                            f"{ctx.target * 100:.0f}% on the reference split. The highest allowed value is "
                            f"{ctx.predictions.highest_value_meeting(ctx.target)}.")
    old = ctx.threshold
    with ctx.db.session() as s:
        ctx.set(s, "threshold", value)
        audit.record(
            s, actor=actor, action="threshold_change", study_id=None,
            text=(f"Threshold changed from {'none' if old is None else format(old, '.2f')} to {value:.2f} (score cut-off {op['score_cutoff']:.4f}). "
                  f"On the reference split: sensitivity {sens * 100:.1f}%, false alarms {op['false_alarm_rate'] * 100:.1f}%, "
                  f"AI unsure {op['abstention_rate'] * 100:.1f}% ({ctx.predictions.source} saved predictions)."),
            old=old, new=value, score_cutoff=op["score_cutoff"], sensitivity=sens, false_alarm_rate=op["false_alarm_rate"],
            data_source=ctx.predictions.source, confirmed=True)
    return info(ctx)
