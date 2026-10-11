"""Put Member A's model files into backend/model_artifacts/ and verify every checksum.

    python scripts/fetch_model.py            # copy from ../ml/<folder>/ if present, else download
    python scripts/fetch_model.py --check    # verify only; exit 1 if anything is missing or wrong

The files are already in the repository under ml/artifacts_v2_imagenet_fullnih/ (Member A committed
them there), so they are not committed a second time under backend/. The Docker build context is
backend/ only, so the image downloads them from the repository's own raw URL, pinned to the commit
that added them, and refuses to continue if a SHA-256 differs from model_artifacts/SOURCE.json.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import urllib.request
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
ART = BACKEND / "model_artifacts"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    src = json.loads((ART / "SOURCE.json").read_text(encoding="utf-8"))
    files, base = src["runtime_files"], src["runtime_base_url"]
    local = BACKEND.parent / "ml" / src["artifacts_folder"]
    check_only = "--check" in sys.argv
    bad = 0
    for name, want in files.items():
        dst = ART / name
        if dst.is_file() and sha256(dst) == want:
            print(f"ok       {name}")
            continue
        if check_only:
            print(f"MISSING  {name}" if not dst.is_file() else f"WRONG    {name}")
            bad += 1
            continue
        if (local / name).is_file():
            shutil.copy2(local / name, dst)
            how = f"copied from ml/{src['artifacts_folder']}/"
        else:
            with urllib.request.urlopen(base + name, timeout=120) as resp, dst.open("wb") as out:   # noqa: S310 (fixed https URL)
                shutil.copyfileobj(resp, out)
            how = "downloaded"
        if sha256(dst) != want:
            dst.unlink(missing_ok=True)
            print(f"CHECKSUM MISMATCH {name}: removed")
            bad += 1
        else:
            print(f"fetched  {name} ({how})")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
