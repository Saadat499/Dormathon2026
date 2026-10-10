"""Live inference engine for Member C's API.

    engine = ChestTriageEngine("artifacts")
    rec = engine.predict("scan.png", study_id="S123", heatmap_dir="heatmaps")
    recs = engine.predict_batch([img1, img2])
    rows = engine.analyse_batch([img1, img2])      # Site Readiness: scores + OOD, no decisions
    drift = engine.site_drift([img1, ...])          # Site Readiness: domain AUROC vs training data
    engine.reload()                                 # after an approved edit of policy.json

Everything it needs is in the artifacts folder (works offline): backbone.json (+ weights),
head.npz, policy.json, and optionally ood.joblib and drift_reference.npy.
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image

from .backbone import load_backbone_from_artifacts
from .contract import make_record, model_version
from .headmodel import LinearHead
from .policy import Policy
from .preprocess import load_image, to_array
from .saliency import cam_for, save_panel


class ChestTriageEngine:
    def __init__(self, art_dir="artifacts", device: str | None = None):
        self.art_dir = Path(art_dir)
        self.backbone = load_backbone_from_artifacts(self.art_dir, device)
        self.head = LinearHead.load(self.art_dir / "head.npz")
        self.ood = None
        if (self.art_dir / "ood.joblib").exists():
            import joblib
            self.ood = joblib.load(self.art_dir / "ood.joblib")
        ref = self.art_dir / "drift_reference.npy"
        self.drift_ref = np.load(ref).astype(np.float32) if ref.exists() else None
        self.reload()

    def reload(self) -> None:
        """Re-read policy.json (thresholds, urgent list, temperatures). Call after a
        clinical lead approves a threshold change; model_version changes with it."""
        self.policy = Policy.load(self.art_dir / "policy.json")
        self.version = model_version(self.art_dir)

    @staticmethod
    def _to_pil(x) -> Image.Image:
        if isinstance(x, Image.Image):
            return x
        if isinstance(x, (bytes, bytearray)):
            im = Image.open(io.BytesIO(x)); im.load(); return im
        return load_image(x)

    def _forward(self, chunk):
        fmap = self.backbone.fmap_np(np.stack([to_array(p) for p in chunk]))
        emb = fmap.mean(axis=(2, 3))
        P = self.policy.probs(self.head.logits(emb))
        o = self.ood.predict(emb) if self.ood is not None else None
        return fmap, P, o

    def predict_batch(self, images, study_ids=None, heatmap_dir=None, batch_size: int = 32,
                      ignore_ood: bool = False) -> list:
        """ignore_ood=True is for offline analysis only (decide and explain as if the image were in-distribution)."""
        pils = [self._to_pil(x) for x in images]
        ids = list(study_ids) if study_ids is not None else [f"upload_{i}" for i in range(len(pils))]
        if heatmap_dir:
            Path(heatmap_dir).mkdir(parents=True, exist_ok=True)
        records = []
        for b in range(0, len(pils), batch_size):
            chunk = pils[b:b + batch_size]
            fmap, P, o = self._forward(chunk)
            flag = o["flag"] if (o is not None and not ignore_ood) else np.zeros(len(chunk), bool)
            dec = self.policy.decide(P, flag)
            for i, p in enumerate(chunk):
                sid, hp = ids[b + i], None
                if heatmap_dir and not flag[i]:
                    urg = self.policy.urgent_idx()
                    k = urg[int(np.argmax(P[i, urg]))]
                    hp = str(Path(heatmap_dir) / f"{sid}.png")
                    save_panel(p, cam_for(self.head, fmap[i], k), hp)
                records.append(make_record(
                    sid, self.policy, P[i], dec["priority"][i], dec["confidence_band"][i],
                    dec["composite_score"][i], bool(flag[i]),
                    None if o is None else o["maha"][i], hp, self.version))
        return records

    def predict(self, image, study_id: str = "upload_0", heatmap_dir=None) -> dict:
        return self.predict_batch([image], [study_id], heatmap_dir)[0]

    def cam(self, image, finding: str | None = None):
        """Raw saliency map (224x224, values 0-1) for one image and one finding (default: the top
        urgent finding). Returns (cam, finding). For audits; the app uses the saved panel."""
        fmap, P, _ = self._forward([self._to_pil(image)])
        if finding is None:
            urg = self.policy.urgent_idx()
            k = urg[int(np.argmax(P[0, urg]))]
        else:
            k = self.policy.findings.index(finding)
        return cam_for(self.head, fmap[0], k), self.policy.findings[k]

    def analyse_batch(self, images, study_ids=None, batch_size: int = 32) -> list:
        """For Site Readiness (not for the queue). One flat dict per image with the same
        columns as predictions/*_full.csv: p_<finding>, composite_score, priority_score_only
        (the decision if OOD were ignored), ood_flag, ood_score. Scores are returned even for
        OOD images so a labelled site sample can be evaluated; the OOD rate is the drift signal."""
        pils = [self._to_pil(x) for x in images]
        ids = list(study_ids) if study_ids is not None else [f"upload_{i}" for i in range(len(pils))]
        rows = []
        for b in range(0, len(pils), batch_size):
            _, P, o = self._forward(pils[b:b + batch_size])
            dec = self.policy.decide(P, None)
            for i in range(len(P)):
                r = {"study_id": ids[b + i]}
                r.update({f"p_{f}": float(P[i, k]) for k, f in enumerate(self.policy.findings)})
                r.update({"composite_score": float(dec["composite_score"][i]),
                          "priority_score_only": str(dec["priority"][i]),
                          "ood_flag": bool(o["flag"][i]) if o is not None else False,
                          "ood_score": float(o["maha"][i]) if o is not None else None,
                          "model_version": self.version})
                rows.append(r)
        return rows

    def site_drift(self, images, batch_size: int = 32) -> dict:
        """Drift warning for an uploaded site sample (aim for 50+ images): domain AUROC vs the training
        reference (0.5 = looks like training data, 1.0 = clearly different), OOD flag rate and median
        OOD score. A warning only; only labelled site data shows whether accuracy dropped."""
        from .drift import drift_summary
        if self.drift_ref is None:
            raise RuntimeError("drift_reference.npy missing: re-run scripts/04_build_ood.py")
        pils = [self._to_pil(x) for x in images]
        emb = np.concatenate([self.backbone.embed_np(np.stack([to_array(p) for p in pils[b:b + batch_size]]))
                              for b in range(0, len(pils), batch_size)])
        return {**drift_summary(self.drift_ref, emb, self.ood), "model_version": self.version}
