"""Phase 4 freeze: write artifacts/MANIFEST.json and check the deploy set is complete and consistent.

The manifest records model_version, the policy (thresholds, urgent list, temperatures), SHA-256 of
every file Member C deploys, and library versions. ood.joblib is a pickle: deploy it with the same
scikit-learn version listed here. After freezing, only an approved policy.json edit changes
model_version (and the manifest must be re-run).
"""
import _common  # noqa: F401
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone

from chesttriage_ml.config import art
from chesttriage_ml.contract import model_version


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    A = art()
    bb = json.loads((A / "backbone.json").read_text())
    deploy = ["backbone.json", "head.npz", "policy.json", "ood.joblib"] + ([bb["weights"]] if bb.get("weights") else [])
    demo = ["predictions/demo_contract.jsonl", "predictions/demo_manifest.csv"]
    missing = [f for f in deploy + demo if not (A / f).exists()]
    if missing:
        sys.exit(f"FREEZE FAILED, missing: {missing}. Run scripts 00-06 first.")

    version = model_version(A)
    stale = []
    for name in ("nih_contract.jsonl", "chexpert_contract.jsonl", "demo_contract.jsonl"):
        f = A / "predictions" / name
        if f.exists():
            first = json.loads(f.open().readline() or "{}")
            if first.get("model_version") != version:
                stale.append(name)
    if stale:
        sys.exit(f"FREEZE FAILED: {stale} were made with a different model/policy. Re-run 05 and 06.")
    demo_recs = [json.loads(l) for l in (A / demo[0]).open()]
    for r in demo_recs:
        for k in ("heatmap_path", "image_path"):
            if r.get(k) and not (A / r[k]).exists():
                sys.exit(f"FREEZE FAILED: {r['study_id']} {k} {r[k]} does not exist")

    import numpy, pandas, sklearn, PIL
    libs = {"python": platform.python_version(), "numpy": numpy.__version__, "pandas": pandas.__version__,
            "scikit-learn": sklearn.__version__, "pillow": PIL.__version__}
    if bb["name"].endswith("-onnx"):                      # record the runtime the backbone actually needs
        import onnxruntime
        libs["onnxruntime"] = onnxruntime.__version__
    elif bb["name"] != "fake":
        import torch, torchvision
        libs.update(torch=torch.__version__, torchvision=torchvision.__version__)
    man = {
        "model_version": version,
        "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "backbone": bb["name"],
        "policy": json.loads((A / "policy.json").read_text()),
        "deploy_files": {f: sha(A / f) for f in deploy},
        "demo_files": {f: sha(A / f) for f in demo},
        "demo_studies": len(demo_recs),
        "libraries": libs,
    }
    (A / "MANIFEST.json").write_text(json.dumps(man, indent=2))
    print(json.dumps({k: man[k] for k in ("model_version", "backbone", "demo_studies", "libraries")}, indent=2))
    print(f"t_raise={man['policy']['t_raise']:.4f}  t_unsure={man['policy']['t_unsure']:.4f}  "
          f"urgent={man['policy']['urgent']}")
    print("FROZEN. Deploy:", ", ".join(deploy), "+ heatmaps/ demo_images/ predictions/demo_contract.jsonl")


if __name__ == "__main__":
    main()
