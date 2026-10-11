"""Where model output comes from. One interface, four honest sources.

    predict(image_bytes) -> PredictionRecord

  real    Member A's frozen model runs on the upload: ONNX DenseNet121 backbone (ImageNet weights,
          224x224) -> linear head (head.npz) -> policy.json findings -> ood.joblib detector.
          All maths is A's code (backend/chesttriage_ml, a byte-identical copy of ml/chesttriage_ml).
  replay  The model is NOT run. An upload whose SHA-256 equals a saved demo image returns that
          image's saved model output. Anything else gets NO model output (it stays visible as
          `unsure`). Images are never matched by file name.
  fake    Development only, never the default. Deterministic numbers from the image hash, labelled
          FAKE everywhere.
  none    No model output at all. Every scan is `unsure`.

MODEL_MODE=auto picks real when the model files are present, verified and loadable, else replay when
saved demo outputs exist, else none. If a requested mode cannot be honoured the service says so on
/health (`degraded: true`) instead of quietly producing numbers some other way.
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from . import images
from .saved_predictions import DEFAULT_FINDINGS, DEFAULT_URGENT

log = logging.getLogger("voxelsort.model")
RUNTIME_FILES = ("backbone.json", "backbone_densenet121-onnx.onnx", "head.npz", "policy.json", "ood.joblib")
DRIFT_FILE = "drift_reference.npy"
DEMO_INDEX = "demo/demo_studies.json"
TRAINED_OOD_METHOD = "Member A's trained detector (Mahalanobis distance + mistake probe on backbone features)"
NOT_FLAGGED = "no out-of-distribution signal (this does not prove the image is like the training data)"


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class OodResult:
    ood_flag: bool
    reason: str
    score: Optional[float]
    method: str
    source: str                                   # real | replay | basic
    threshold: Optional[float] = None
    probe_probability: Optional[float] = None


@dataclass
class PredictionRecord:
    findings: list = field(default_factory=list)  # [{"name","probability"}] highest first; [] when not scored
    composite_score: Optional[float] = None
    ood: Optional[OodResult] = None
    source: str = "none"                          # real | replay | fake | none
    model_version: Optional[str] = None
    note: Optional[str] = None                    # why there is no model output, when there is none
    cam: Optional[np.ndarray] = None              # saliency map in [0,1], or None
    cam_finding: Optional[str] = None

    @property
    def ood_flag(self) -> bool:
        return bool(self.ood and self.ood.ood_flag)


def sorted_findings(probs: dict) -> list:
    return sorted(({"name": k, "probability": round(float(v), 4)} for k, v in probs.items()),
                  key=lambda f: (-f["probability"], f["name"]))


def ood_reason(flag_maha: bool, flag_probe: bool) -> str:
    """Fixed wording for the two signals of A's detector."""
    if flag_probe and flag_maha:
        return "not a usable chest X-ray (looks rotated, inverted, blurred, noisy or not a chest image) and far from the training distribution"
    if flag_probe:
        return "not a usable chest X-ray (looks rotated, inverted, blurred, noisy or not a chest image)"
    if flag_maha:
        return "different distribution from the training images"
    return NOT_FLAGGED


# ------------------------------------------------------------------------------------------ real
class RealModel:
    """Member A's engine, loaded once. Inference is serialised with a lock (one small CPU)."""

    def __init__(self, art_dir: Path, threads: int = 1):
        import joblib
        import onnxruntime as ort
        from chesttriage_ml.backbone import OnnxDenseNet121
        from chesttriage_ml.engine import ChestTriageEngine
        from chesttriage_ml.headmodel import LinearHead

        info = json.loads((art_dir / "backbone.json").read_text())
        if info.get("name") != "densenet121-onnx":
            raise RuntimeError(f"unsupported backbone {info.get('name')!r}: this host only has ONNX Runtime")

        class SmallHostBackbone(OnnxDenseNet121):
            """A's backbone class with a memory-lean session (one thread, no arena). Same graph, same weights."""

            def __init__(self, onnx_path: Path, n_threads: int):  # noqa: D401
                so = ort.SessionOptions()
                so.log_severity_level = 3
                so.intra_op_num_threads = max(1, n_threads)
                so.inter_op_num_threads = 1
                so.enable_cpu_mem_arena = False
                so.enable_mem_pattern = False
                self._bytes = b""                                   # save() is not needed when serving
                self.sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
                self.provider = self.sess.get_providers()[0]

        # Same steps as ChestTriageEngine.__init__, with the lean backbone session.
        eng = ChestTriageEngine.__new__(ChestTriageEngine)
        eng.art_dir = Path(art_dir)
        eng.backbone = SmallHostBackbone(art_dir / info["weights"], threads)
        eng.head = LinearHead.load(art_dir / "head.npz")
        eng.ood = joblib.load(art_dir / "ood.joblib") if (art_dir / "ood.joblib").exists() else None
        ref = art_dir / DRIFT_FILE
        eng.drift_ref = np.load(ref).astype(np.float32) if ref.exists() else None
        eng.reload()
        self.engine = eng
        self.lock = threading.Lock()
        self.version = eng.version
        self.findings = list(eng.policy.findings)
        self.urgent = list(eng.policy.urgent)
        self.has_ood = eng.ood is not None
        self.has_drift_reference = eng.drift_ref is not None
        self.provider = eng.backbone.provider

    def forward(self, pil: Image.Image) -> dict:
        """One pass through A's building blocks, in the order `ChestTriageEngine.predict_batch` uses them."""
        from chesttriage_ml.preprocess import to_array
        eng = self.engine
        with self.lock:
            fmap = eng.backbone.fmap_np(np.stack([to_array(pil)]))
        emb = fmap.mean(axis=(2, 3))
        P = eng.policy.probs(eng.head.logits(emb))
        o = eng.ood.predict(emb) if eng.ood is not None else None
        return {"fmap": fmap[0], "emb": emb[0], "P": P[0], "o": None if o is None else {k: v[0] for k, v in o.items()}}

    def ood_from(self, o: Optional[dict]) -> Optional[OodResult]:
        if o is None:
            return None
        return OodResult(
            ood_flag=bool(o["flag"]), reason=ood_reason(bool(o["flag_maha"]), bool(o["flag_probe"])),
            score=round(float(o["maha"]), 3), method=TRAINED_OOD_METHOD, source="real",
            threshold=round(float(self.engine.ood.threshold), 3), probe_probability=round(float(o["probe"]), 4))

    def predict(self, pil: Image.Image) -> PredictionRecord:
        from chesttriage_ml.saliency import cam_for
        f = self.forward(pil)
        ood = self.ood_from(f["o"])
        if ood is not None and ood.ood_flag:      # A's rule: an out-of-distribution image is not scored
            return PredictionRecord(ood=ood, source="real", model_version=self.version)
        pol = self.engine.policy
        P = f["P"]
        urg = pol.urgent_idx()
        k = urg[int(np.argmax(P[urg]))]
        return PredictionRecord(
            findings=sorted_findings(dict(zip(pol.findings, P))), composite_score=round(float(P[urg].max()), 4),
            ood=ood, source="real", model_version=self.version,
            cam=cam_for(self.engine.head, f["fmap"], k), cam_finding=pol.findings[k])

    def analyse(self, pil: Image.Image) -> dict:
        """For Site Readiness: scores even when the image is flagged, plus the embedding for the drift score."""
        f = self.forward(pil)
        pol = self.engine.policy
        P = f["P"]
        return {"probs": {n: float(p) for n, p in zip(pol.findings, P)}, "composite_score": float(P[pol.urgent_idx()].max()),
                "ood_flag": bool(f["o"]["flag"]) if f["o"] is not None else False, "embedding": f["emb"]}

    def site_drift(self, embeddings: np.ndarray) -> dict:
        from chesttriage_ml.drift import drift_summary
        if self.engine.drift_ref is None:
            raise RuntimeError("drift_reference.npy is not installed")
        return drift_summary(self.engine.drift_ref, np.asarray(embeddings, dtype=np.float32), self.engine.ood)


# ---------------------------------------------------------------------------------------- service
class ModelService:
    def __init__(self, artifacts_dir: Path, requested: str = "auto", threads: int = 1):
        self.artifacts_dir = Path(artifacts_dir)
        self.requested = requested
        self.active = "none"
        self.problems: list[str] = []
        self.real: Optional[RealModel] = None
        self.demo: dict[str, dict] = {}            # study_id -> saved demo record
        self.by_hash: dict[str, dict] = {}         # sha256 of the image file -> saved demo record
        self.manifest: dict = {}
        self.version: Optional[str] = None
        self.findings = list(DEFAULT_FINDINGS)
        self.urgent = list(DEFAULT_URGENT)
        self.policy: dict = {}
        self.runtime_errors = 0
        self._load_demo()
        self._load_policy()
        self._activate(threads)

    # --- loading ---------------------------------------------------------------------------
    def _load_demo(self) -> None:
        p = self.artifacts_dir / DEMO_INDEX
        if not p.is_file():
            return
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            for r in data.get("studies", []):
                self.demo[r["study_id"]] = r
                if r.get("sha256"):
                    self.by_hash[r["sha256"]] = r
            self.version = data.get("model_version")
        except Exception as e:  # noqa: BLE001
            self.problems.append(f"{DEMO_INDEX} could not be read: {e}")

    def _load_policy(self) -> None:
        for name, target in (("policy.json", "policy"), ("MANIFEST.json", "manifest")):
            p = self.artifacts_dir / name
            if p.is_file():
                try:
                    setattr(self, target, json.loads(p.read_text(encoding="utf-8")))
                except Exception as e:  # noqa: BLE001
                    self.problems.append(f"{name} could not be read: {e}")
        pol = self.policy or self.manifest.get("policy") or {}
        if pol.get("findings"):
            self.findings, self.urgent = list(pol["findings"]), list(pol.get("urgent", DEFAULT_URGENT))
            self.policy = pol
        self.version = self.manifest.get("model_version") or self.version

    def verify_runtime_files(self) -> list[str]:
        """Every model file must be present and match the SHA-256 in Member A's MANIFEST.json."""
        want = (self.manifest.get("deploy_files") or {})
        problems = []
        for name in RUNTIME_FILES:
            f = self.artifacts_dir / name
            if not f.is_file():
                problems.append(f"{name} is missing")
            elif name not in want:
                problems.append(f"{name} has no checksum in MANIFEST.json")
            elif file_sha256(f) != want[name]:
                problems.append(f"{name} does not match the checksum in MANIFEST.json")
        return problems

    def _activate(self, threads: int) -> None:
        want = self.requested
        if want == "fake":
            self.active = "fake"
            self.problems.append("MODEL_MODE=fake: outputs are fabricated for development and labelled FAKE. Never use this in a demo of results.")
            return
        if want == "none":
            self.active = "none"
            return
        if want in ("auto", "real"):
            file_problems = self.verify_runtime_files()
            if not file_problems:
                try:
                    self.real = RealModel(self.artifacts_dir, threads)
                    if self.manifest.get("model_version") and self.real.version != self.manifest["model_version"]:
                        raise RuntimeError(f"loaded model version {self.real.version} differs from MANIFEST.json {self.manifest['model_version']}")
                    self.active, self.version = "real", self.real.version
                    self.findings, self.urgent = self.real.findings, self.real.urgent
                    return
                except Exception as e:  # noqa: BLE001
                    self.real = None
                    file_problems = [f"the model could not be loaded: {e}"]
                    log.exception("real model failed to load")
            if want == "real":
                self.problems.append("MODEL_MODE=real was requested but real inference is unavailable: " + "; ".join(file_problems))
            else:
                self.problems.append("Real inference is not available on this host (" + "; ".join(file_problems) + ").")
        if self.by_hash:
            self.active = "replay"
        else:
            self.active = "none"
            self.problems.append("No saved demo outputs are installed either, so uploads get no model output.")

    # --- status ----------------------------------------------------------------------------
    @property
    def degraded(self) -> bool:
        """True when what was asked for is not what is running, or inference has failed at run time."""
        if self.runtime_errors:
            return True
        if self.requested in ("auto", "none"):
            return False
        return self.requested != self.active

    def describe(self) -> dict:
        text = {
            "real": "Member A's frozen model runs on every upload (ONNX DenseNet121 backbone + linear head + trained OOD detector).",
            "replay": "The model is not run on this host. An upload that is byte-identical to a saved demo image returns its saved model output; any other upload gets no model output and stays in the queue as 'AI unsure'.",
            "fake": "FAKE development outputs. Not a model.",
            "none": "No model output is available. Every scan is 'AI unsure' and stays visible for a clinician.",
        }[self.active]
        return {
            "requested": self.requested, "active": self.active, "runs_a_trained_model": self.active == "real",
            "degraded": self.degraded, "version": self.version, "description": text, "problems": self.problems,
            "runtime_errors": self.runtime_errors, "saved_demo_outputs": len(self.demo),
            "ood_check": TRAINED_OOD_METHOD if (self.real and self.real.has_ood) else
                         ("saved results of the trained detector for demo images; " + images.BASIC_METHOD if self.active == "replay" else images.BASIC_METHOD),
            "findings": self.findings, "urgent_findings": self.urgent,
        }

    # --- interface -------------------------------------------------------------------------
    def demo_record(self, r: dict, source: str) -> PredictionRecord:
        ood = OodResult(
            ood_flag=bool(r.get("ood_flag")),
            reason=(r.get("ood_reason") or "flagged by the trained detector when this image was scored") if r.get("ood_flag") else NOT_FLAGGED,
            score=r.get("ood_score"), method=TRAINED_OOD_METHOD + " (saved result)", source="replay")
        return PredictionRecord(
            findings=sorted_findings(r.get("findings") or {}), composite_score=r.get("composite_score"), ood=ood,
            source=source, model_version=r.get("model_version") or self.version)

    def predict(self, data: bytes, pil: Optional[Image.Image] = None) -> PredictionRecord:
        pil = pil or images.load_image(data)
        if self.active == "real" and self.real is not None:
            try:
                return self.real.predict(pil)
            except Exception as e:  # noqa: BLE001
                self.runtime_errors += 1
                log.exception("real inference failed")
                return PredictionRecord(source="none", model_version=self.version,
                                        note=f"the model failed on this image ({type(e).__name__})")
        if self.active == "fake":
            return self._fake(data, pil)
        basic = images.basic_check(pil)
        basic_ood = OodResult(basic.flag, basic.reason, basic.score, basic.method, "basic")
        if self.active == "replay":
            hit = self.by_hash.get(sha256_hex(data))
            if hit is not None:
                return self.demo_record(hit, "replay")
            return PredictionRecord(ood=basic_ood, source="none", model_version=None,
                                    note="this host replays saved model outputs only, and this image does not match a saved study")
        return PredictionRecord(ood=basic_ood, source="none", note="no model is installed on this host")

    def check_ood(self, data: bytes, pil: Optional[Image.Image] = None) -> OodResult:
        pil = pil or images.load_image(data)
        if self.active == "real" and self.real is not None and self.real.has_ood:
            try:
                return self.real.ood_from(self.real.forward(pil)["o"])  # type: ignore[return-value]
            except Exception:  # noqa: BLE001
                self.runtime_errors += 1
                log.exception("OOD check failed; falling back to the basic check")
        if self.active == "replay":
            hit = self.by_hash.get(sha256_hex(data))
            if hit is not None:
                return self.demo_record(hit, "replay").ood  # type: ignore[return-value]
        b = images.basic_check(pil)
        return OodResult(b.flag, b.reason, b.score, b.method, "basic")

    def _fake(self, data: bytes, pil: Image.Image) -> PredictionRecord:
        """Deterministic, made-up numbers. Only reachable with MODEL_MODE=fake."""
        basic = images.basic_check(pil)
        ood = OodResult(basic.flag, basic.reason, basic.score, "FAKE: " + basic.method, "basic")
        if basic.flag:
            return PredictionRecord(ood=ood, source="fake", model_version="fake-dev")
        rng = np.random.default_rng(int(sha256_hex(data)[:16], 16))
        lead = self.findings[int(rng.integers(len(self.findings)))]
        probs = {f: float(rng.beta(1.2, 12)) for f in self.findings}
        probs[lead] = float(rng.beta(2, 3))
        comp = max(probs[f] for f in self.urgent)
        return PredictionRecord(findings=sorted_findings(probs), composite_score=round(comp, 4), ood=ood,
                                source="fake", model_version="fake-dev")
