"""Site Readiness: is there enough evidence to use VoxelSort at a new site?

What is measured
  drift        Member B's `SiteReadinessAssessor.evaluate_site`: PSI / KS of the site's per-finding
               probabilities against the reference (NIH validation) predictions, plus the share of
               images Member A's detector flags as out of distribution. B's verdict is GO / WARNING /
               NO-GO. Member A's own drift score (domain AUROC on backbone features) is reported next
               to it when at least 20 images were scored. Drift is a WARNING signal: A showed it reacts
               to file format alone (NIH re-saved as JPEG scores 0.998). It never proves accuracy dropped.
  sensitivity  on the composite label "any urgent finding", from the labels CSV: the share of labelled
               urgent scans whose score is at or above the current cut-off, with an exact
               (Clopper-Pearson) two-sided 95% interval.

How the target and the interval are used (contract mapping, stated limits)
  target = `min_sensitivity_target`. S is the sensitivity evidence, D is B's drift verdict.
    not_ready            the interval's UPPER end is below the target (demonstrated shortfall), or
                         D = NO-GO that is not explainable by sample size (see below)
    not_enough_evidence  no labels, fewer than 30 labelled urgent scans, or the interval's LOWER end is
                         below the target. A small sample is not evidence of poor performance.
    ready                lower end >= target, at least 30 urgent scans, and D = GO
    use_with_caution     lower end >= target, at least 30 urgent scans, and D = WARNING or the drift
                         check could not run
  A site is never `ready` without both a passing interval and a clean drift check.

  Sample size and B's drift rule. PSI computed on a small sample is large by chance alone (its expected
  value with 10 bins is about 9 / n), so B's fixed cut-offs (0.10 warning, 0.25 no-go) fire on small
  samples that come from the SAME distribution as the reference. A NO-GO therefore counts towards
  `not_ready` only when it is beyond chance for the sample size:
    * PSI trigger: the site's max PSI must exceed the 95th percentile of the max PSI of random samples of
      the same size drawn from the reference predictions themselves (computed live, `chance_level_95`);
    * OOD-rate trigger: at least 50 images must have been checked.
  Otherwise the NO-GO is reported as B returned it but counted as a WARNING. A WARNING is never upgraded
  to GO: a small sample cannot show the absence of drift, so it can reach `use_with_caution` at best.
  With B's cut-offs a clean GO needs a few hundred images.

Scores come from the model when it runs on this host (real mode), from a saved output when the image
is byte-identical to a saved demo image (replay mode), or from a `score` column the uploader puts in
the CSV (labelled as uploader-supplied). Images are never matched to saved outputs by file name.
"""
from __future__ import annotations

import csv
import io
import logging
import math
import uuid
from pathlib import PurePath
from typing import Optional

import numpy as np
from scipy.special import betaincinv

from .. import config
from ..context import AppContext
from ..db import now_utc
from ..errors import ApiError
from ..evalkit.protocol import calculate_required_sample_size
from ..evalkit.site_readiness import SiteReadinessAssessor, calculate_psi
from ..models import SiteJob
from . import audit, images
from .predictor import sha256_hex

log = logging.getLogger("voxelsort.site")
NOTE_DRIFT = "Warning signal only, not a proven predictor of performance at a new site."
VERDICT_MAP = {"GO": "ready", "WARNING": "use_with_caution", "NO-GO": "not_ready"}
ID_COLUMNS = ("filename", "file", "image", "image_path", "path", "study_id", "id")
SCORE_COLUMNS = ("score", "composite_score")


# ------------------------------------------------------------------------------------ statistics
def clopper_pearson(x: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Exact two-sided binomial interval for x successes out of n."""
    if n <= 0:
        return 0.0, 1.0
    a = (1.0 - confidence) / 2.0
    lo = 0.0 if x == 0 else float(betaincinv(x, n - x + 1, a))
    hi = 1.0 if x == n else float(betaincinv(x + 1, n - x, 1.0 - a))
    return lo, hi


def exact_positives_needed(target: float, expected: float, confidence: float, max_n: int = 20000) -> Optional[int]:
    """Smallest number of urgent scans for which the exact lower bound would reach `target`
    if the observed sensitivity equalled `expected`."""
    n = np.arange(1, max_n + 1)
    x = np.rint(expected * n).astype(int)
    a = (1.0 - confidence) / 2.0
    lower = np.where(x == 0, 0.0, betaincinv(np.maximum(x, 1), n - x + 1, a))
    ok = np.nonzero(lower >= target)[0]
    return int(n[ok[0]]) if ok.size else None


def sample_size(target: float, expected: float, confidence: float = 0.95, prevalence: float = 0.2) -> dict:
    """`positives_needed` is the larger of two live calculations:
      * Member B's `calculate_required_sample_size` (normal approximation), asked for a margin of error
        equal to the gap between the expected sensitivity and the target;
      * the exact Clopper-Pearson search, which is the bound the verdict actually applies.
    A planning estimate that assumes the true sensitivity equals the expected value."""
    for name, v in (("target_sensitivity", target), ("expected_sensitivity", expected)):
        if not (isinstance(v, (int, float)) and 0 < v < 1):
            raise ApiError(400, f"{name} must be between 0 and 1.")
    if not 0.5 < confidence < 1:
        raise ApiError(400, "confidence must be between 0.5 and 1.")
    if not 0 < prevalence <= 1:
        raise ApiError(400, "prevalence must be above 0 and at most 1.")
    if expected <= target:
        return {"positives_needed": None, "scans_needed": None, "method": "none",
                "note": "The sensitivity you expect must be higher than the target, otherwise no sample size can confirm the target."}
    b = calculate_required_sample_size(target_sensitivity=expected, margin_of_error=expected - target,
                                       prevalence=prevalence, confidence_level=confidence)
    wald = int(b["min_positive_cases_required"])
    exact = exact_positives_needed(target, expected, confidence)
    if exact is None:
        return {"positives_needed": None, "scans_needed": None, "method": "clopper-pearson", "normal_approximation_positives": wald,
                "note": "More than 20,000 urgent scans would be needed for the exact lower bound to reach the target."}
    pos = max(wald, exact)
    return {
        "positives_needed": pos,
        "scans_needed": math.ceil(pos / prevalence),
        "method": "max(normal approximation, exact Clopper-Pearson)",
        "normal_approximation_positives": wald,
        "exact_positives": exact,
        "note": (f"Planning estimate, not a guarantee. It assumes the true sensitivity equals the expected value. "
                 f"The normal-approximation formula gives {wald} urgent scans; the exact (Clopper-Pearson) lower bound that the "
                 f"verdict uses needs {exact}. The larger number is shown. Fewer than {config.MIN_POSITIVES_FOR_VERDICT} urgent scans "
                 "always gives 'not enough evidence'."),
    }


def psi_chance_level(ref_probs: np.ndarray, n_site: int, draws: int = 100, seed: int = 0) -> float:
    """95th percentile of B's max-PSI statistic for samples of `n_site` rows drawn from the reference
    itself: what "no drift at all" looks like at this sample size."""
    rng = np.random.default_rng(seed)
    n_ref, k = ref_probs.shape
    vals = []
    for _ in range(draws):
        idx = rng.choice(n_ref, size=n_site, replace=n_site > n_ref)
        vals.append(max(calculate_psi(ref_probs[:, j], ref_probs[idx, j]) for j in range(k)))
    return float(np.quantile(vals, 0.95))


# ------------------------------------------------------------------------------------ labels CSV
def parse_labels(text: str) -> list[dict]:
    """CSV with a header row. `label` (1 = urgent finding, 0 = not) is required. Optional: a file-name
    column (filename / file / image / study_id) to link a row to an uploaded image, and `score`."""
    rows = [r for r in csv.reader(io.StringIO(text.lstrip("﻿"))) if any(c.strip() for c in r)]
    if not rows:
        return []
    header = [c.strip().lower() for c in rows[0]]
    if "label" not in header:
        raise ValueError("The labels CSV needs a header row with a 'label' column (1 = urgent finding, 0 = not).")
    li = header.index("label")
    si = next((header.index(c) for c in SCORE_COLUMNS if c in header), None)
    ii = next((header.index(c) for c in ID_COLUMNS if c in header), None)
    out = []
    for n, r in enumerate(rows[1:], start=2):
        if len(r) <= li or not r[li].strip():
            continue
        lab = r[li].strip().lower()
        if lab not in ("0", "1", "0.0", "1.0"):
            raise ValueError(f"Row {n}: label must be 0 or 1, got '{r[li].strip()[:20]}'.")
        score = None
        if si is not None and len(r) > si and r[si].strip():
            try:
                score = float(r[si])
            except ValueError:
                raise ValueError(f"Row {n}: score must be a number.") from None
            if score != score or not 0.0 <= score <= 1.0:
                raise ValueError(f"Row {n}: score must be between 0 and 1.")
        key = r[ii].strip() if ii is not None and len(r) > ii and r[ii].strip() else None
        out.append({"label": int(float(lab)), "score": score, "key": key})
    if len(out) > 20000:
        raise ValueError("The labels CSV has more than 20,000 rows.")
    return out


def _key(name: Optional[str]) -> Optional[str]:
    """File names are compared by base name without extension, case-insensitively."""
    if not name:
        return None
    return PurePath(name.replace("\\", "/")).stem.lower()


# ------------------------------------------------------------------------------------ the check
def score_images(ctx: AppContext, files: list[tuple[str, bytes]]) -> dict:
    """Runs what this host can run on each image. Returns per-image rows plus counts."""
    rows, unreadable, embeddings = [], 0, []
    for name, data in files:
        try:
            pil = images.load_image(data)
        except images.ImageError:
            unreadable += 1
            continue
        row = {"key": _key(name), "name": name, "probs": None, "score": None, "ood_flag": None, "score_source": None, "ood_source": None}
        if ctx.model.active == "real" and ctx.model.real is not None:
            try:
                a = ctx.model.real.analyse(pil)
                row.update(probs=a["probs"], score=a["composite_score"], ood_flag=a["ood_flag"], score_source="model", ood_source="trained detector")
                embeddings.append(a["embedding"])
            except Exception:  # noqa: BLE001
                ctx.model.runtime_errors += 1
                log.exception("site image could not be scored")
        elif ctx.model.active == "replay":
            hit = ctx.model.by_hash.get(sha256_hex(data))
            if hit is not None:
                row.update(probs=hit.get("findings") or None, score=hit.get("composite_score"), ood_flag=bool(hit.get("ood_flag")),
                           score_source="saved output (exact image match)", ood_source="trained detector (saved result)")
        if row["ood_flag"] is None:
            b = images.basic_check(pil)
            row.update(ood_flag=b.flag, ood_source="basic image check")
        rows.append(row)
    return {"rows": rows, "unreadable": unreadable, "embeddings": embeddings}


def evaluate(ctx: AppContext, files: list[tuple[str, bytes]], labels_text: Optional[str]) -> dict:
    target = ctx.target
    notes: list[str] = []
    scored = score_images(ctx, files)
    rows = scored["rows"]
    n_img = len(rows)
    with_probs = [r for r in rows if r["probs"]]
    n_ood = sum(1 for r in rows if r["ood_flag"])
    if scored["unreadable"]:
        notes.append(f"{scored['unreadable']} file(s) could not be read as PNG or JPG and were skipped.")

    # --- drift (Member B's rule) ---------------------------------------------------------------
    b_verdict, b_reasons, b_report, drift_kind = None, [], None, "none"
    drift_score: Optional[float] = None
    trained_ood = bool(rows) and all((r["ood_source"] or "").startswith("trained detector") for r in rows)
    if with_probs and ctx.predictions.available:
        ref = {f: ctx.predictions.val.probs[:, k] for k, f in enumerate(ctx.predictions.findings)}
        tgt = {f: np.array([r["probs"][f] for r in with_probs if f in r["probs"]], dtype=float) for f in ctx.predictions.findings}
        tgt = {f: v for f, v in tgt.items() if len(v)}
        ood_flags = np.array([bool(r["ood_flag"]) for r in rows], dtype=bool)
        b_report = SiteReadinessAssessor().evaluate_site("uploaded site sample", ref, tgt, ood_flags)
        b_verdict, b_reasons = b_report["verdict"], list(b_report["verdict_reasons"])
        drift_score, drift_kind = float(b_report["max_psi"]), "max_psi_of_finding_probabilities"
    elif n_img:
        drift_score, drift_kind = round(n_ood / n_img, 4), "ood_flag_share"
        notes.append("No model scores were available for the uploaded images on this host, so the drift score is only the share of "
                     f"images flagged ({n_ood} of {n_img}) by the {'trained detector' if trained_ood else 'basic image check'}; Member B's score-drift rule could not run.")
    counted, chance = b_verdict, None
    if b_verdict == "NO-GO" and b_report is not None:
        rule = SiteReadinessAssessor()
        ood_beyond_chance = b_report["ood_rate"] >= rule.ood_nogo_threshold and n_img >= config.MIN_DRIFT_SAMPLE_FOR_NOT_READY
        psi_beyond_chance = False
        if b_report["max_psi"] >= rule.psi_nogo_threshold:
            chance = psi_chance_level(ctx.predictions.val.probs, len(with_probs))
            psi_beyond_chance = b_report["max_psi"] > chance
        if not (ood_beyond_chance or psi_beyond_chance):
            counted = "WARNING"
            notes.append(f"Member B's drift rule returned NO-GO on {len(with_probs)} scored images, but that is within what sample size alone "
                         "produces" + (f" (max PSI {b_report['max_psi']:.2f}; samples of this size drawn from the reference data itself reach "
                                       f"{chance:.2f})" if chance is not None else "")
                         + ". It is treated as a warning, not as evidence the site is not ready.")
    if b_verdict in ("WARNING", "NO-GO") and len(with_probs) < 300:
        notes.append("With Member B's cut-offs a clean drift result needs a few hundred images; a smaller sample can reach 'use with caution' at best.")
    domain = None
    if ctx.model.active == "real" and ctx.model.real is not None and ctx.model.real.has_drift_reference:
        if len(scored["embeddings"]) >= config.MIN_DRIFT_SAMPLE:
            try:
                domain = ctx.model.real.site_drift(np.stack(scored["embeddings"]))
            except Exception as e:  # noqa: BLE001
                notes.append(f"Member A's drift score could not be computed ({type(e).__name__}).")
        elif n_img:
            notes.append(f"Member A's drift score needs at least {config.MIN_DRIFT_SAMPLE} images; {len(scored['embeddings'])} were scored.")
    drift_note = NOTE_DRIFT
    if b_reasons:
        drift_note += " Member B's rule: " + " ".join(b_reasons)
    if domain is not None and domain.get("domain_auroc") == domain.get("domain_auroc"):
        drift_note += (f" Member A's drift score (domain AUROC) is {domain['domain_auroc']:.2f}: about 0.5 means the images look like the "
                       "training data; it also reaches about 1.0 for training images merely re-saved as JPEG, so read it as 'these images differ, check them'.")
    drift = {
        "score": None if drift_score is None else round(drift_score, 4),
        "warning": counted in ("WARNING", "NO-GO") or (b_verdict is None and n_img > 0 and n_ood / n_img >= 0.025),
        "note": drift_note if n_img else "No images were supplied, so no drift check could run. " + NOTE_DRIFT,
        "kind": drift_kind, "b_verdict": b_verdict, "b_verdict_counted_as": counted,
        "ood_rate": round(n_ood / n_img, 4) if n_img else None,
        "max_psi": None if b_report is None else b_report["max_psi"], "max_ks": None if b_report is None else b_report["max_ks"],
        "domain_auroc": None if domain is None else domain.get("domain_auroc"),
        "n_images_scored": len(with_probs), "chance_level_95": None if chance is None else round(chance, 4),
    }

    # --- sensitivity ---------------------------------------------------------------------------
    label_rows = parse_labels(labels_text) if labels_text and labels_text.strip() else []
    by_key = {r["key"]: r for r in rows if r["key"]}
    lanes = ctx.lanes()
    supplied = 0
    for lr in label_rows:
        img = by_key.get(_key(lr["key"])) if lr["key"] else None
        if img is not None and img["score"] is not None:
            lr["used_score"], lr["score_from"] = img["score"], img["score_source"]
        elif lr["score"] is not None:
            lr["used_score"], lr["score_from"] = lr["score"], "uploader"
            supplied += 1
        else:
            lr["used_score"], lr["score_from"] = None, None
    positives = [r for r in label_rows if r["label"] == 1]
    usable = [r for r in positives if r["used_score"] is not None]
    sensitivity, s_state = None, "none"
    if not label_rows:
        notes.append("No labels CSV was supplied, so sensitivity could not be measured.")
    elif not positives:
        notes.append("The labels CSV has no urgent (label = 1) scans, so sensitivity could not be measured.")
    elif lanes.t_raise is None:
        notes.append("No score cut-off is available on this server, so sensitivity could not be measured.")
    elif not usable:
        if not any(r["key"] for r in label_rows):
            notes.append("The labels CSV has no file-name column (filename), so its rows cannot be linked to the uploaded images.")
        elif ctx.model.active != "real":
            notes.append("This host does not run the model, so the labelled images have no scores. Sensitivity could not be measured.")
        else:
            notes.append("None of the labelled urgent scans matched an uploaded image by file name.")
    else:
        if len(usable) < len(positives):
            notes.append(f"{len(positives) - len(usable)} of {len(positives)} labelled urgent scans had no score and were left out.")
        hit = sum(1 for r in usable if r["used_score"] >= lanes.t_raise)
        n = len(usable)
        lo, hi = clopper_pearson(hit, n, 0.95)
        sensitivity = {"point": hit / n, "ci_low": lo, "ci_high": hi, "width": hi - lo, "method": "clopper-pearson",
                       "n_positive": n, "n_caught": hit, "confidence": 0.95}
        if hi < target:
            s_state = "below"
        elif n < config.MIN_POSITIVES_FOR_VERDICT or lo < target:
            s_state = "insufficient"
            if n < config.MIN_POSITIVES_FOR_VERDICT:
                notes.append(f"Only {n} labelled urgent scans: at least {config.MIN_POSITIVES_FOR_VERDICT} are needed before any verdict "
                             "other than 'not enough evidence'. A small sample is not evidence of poor performance.")
            else:
                notes.append("The lower end of the interval is below the target, so the target is not confirmed. That is not the same as showing the model falls short.")
        else:
            s_state = "meets"
        notes.append(f"An urgent scan counts as caught when its score is at or above the current cut-off ({lanes.t_raise:.4f}, threshold {ctx.threshold}).")
        if supplied:
            notes.append(f"{supplied} score(s) came from the CSV's score column: supplied by the uploader, not computed by this server.")

    # --- verdict -------------------------------------------------------------------------------
    if s_state == "below":
        verdict = "not_ready"
        notes.append("Verdict: even the upper end of the sensitivity interval is below the target.")
    elif counted == "NO-GO":
        verdict = VERDICT_MAP["NO-GO"]
        notes.append("Verdict: Member B's drift rule returned NO-GO, beyond what sample size alone explains. This reflects image drift, not a measured drop in sensitivity.")
    elif s_state == "meets":
        if counted == "GO":
            verdict = VERDICT_MAP["GO"]
        else:
            verdict = VERDICT_MAP["WARNING"]
            if counted is None:
                notes.append("The drift check could not run, so the verdict cannot be better than 'use with caution'.")
    else:
        verdict = "not_enough_evidence"

    return {
        "status": "done", "drift": drift, "sensitivity": sensitivity, "target": target, "verdict": verdict,
        "note": " ".join(notes) if notes else None,
        "n_images": n_img, "n_images_flagged_ood": n_ood, "n_label_rows": len(label_rows),
        "model_mode": ctx.model.active, "model_version": ctx.model.version, "threshold": ctx.threshold,
        "verdict_rules": "not_ready: interval upper end < target, or a drift NO-GO beyond chance for the sample size. ready: lower end >= target with >= 30 urgent scans and drift GO. use_with_caution: lower end >= target but drift WARNING or not run. Otherwise not_enough_evidence.",
    }


# ------------------------------------------------------------------------------------ jobs
def start_job(ctx: AppContext, n_files: int, has_labels: bool, actor: Optional[str]) -> str:
    job_id = f"sr-{uuid.uuid4().hex[:12]}"
    with ctx.db.session() as s:
        s.add(SiteJob(job_id=job_id, status="processing", result=None, created_at=now_utc()))
        audit.record(s, actor=actor, action="site_readiness_run", study_id=None,
                     text=f"Site readiness check started: {n_files} file(s), labels {'provided' if has_labels else 'not provided'}. Job {job_id}.",
                     job_id=job_id, n_files=n_files, labels=has_labels, model_mode=ctx.model.active)
    return job_id


def run_job(ctx: AppContext, job_id: str, files: list[tuple[str, bytes]], labels_text: Optional[str]) -> None:
    try:
        result = evaluate(ctx, files, labels_text)
    except Exception as e:  # noqa: BLE001
        log.exception("site readiness job %s failed", job_id)
        result = {"status": "done", "drift": {"score": None, "warning": False, "note": "The check failed before a drift score was computed."},
                  "sensitivity": None, "target": ctx.target, "verdict": "not_enough_evidence",
                  "note": f"The check could not be completed ({type(e).__name__}). No verdict can be drawn from it."}
    with ctx.db.session() as s:
        job = s.get(SiteJob, job_id)
        if job is not None:
            job.status, job.result = "done", result


def get_job(ctx: AppContext, job_id: str) -> dict:
    with ctx.db.session() as s:
        job = s.get(SiteJob, job_id)
        if job is None:
            raise ApiError(404, "Job not found. The server may have restarted; run the check again.")
        if job.status != "done" or job.result is None:
            return {"status": "processing"}
        return dict(job.result)
