"""Extract and cache frozen-backbone features for the saved subsets (resumable)."""
import _common  # noqa: F401
import argparse

from chesttriage_ml.backbone import load_backbone_from_artifacts
from chesttriage_ml.config import art
from chesttriage_ml.features import extract
from _common import backbone_name, feat_dir, get_subset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["nih", "chexpert"])
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    bb = load_backbone_from_artifacts(art())
    for name in a.datasets:
        df = get_subset(name)
        print(f"[{name}] {len(df)} images with {bb.name}")
        extract(df, bb, feat_dir(name, backbone_name()), a.batch_size, a.workers)
    print("done. Features cached in", art("features"))


if __name__ == "__main__":
    main()
