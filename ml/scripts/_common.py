"""Shared helpers for the numbered scripts (run them from the project root)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import numpy as np
import pandas as pd

from chesttriage_ml.config import FINDINGS, DEFAULT_URGENT, art, load_dataset_paths
from chesttriage_ml.data import LABEL_COLS
from chesttriage_ml.features import load_features
from chesttriage_ml.policy import Policy


def subset_csv(name):
    return art("subsets", f"{name}.csv")


def feat_dir(name, backbone):
    return art("features", f"{name}_{backbone}")


def get_subset(name) -> pd.DataFrame:
    f = subset_csv(name)
    if not f.exists():
        sys.exit(f"Missing {f}. Run scripts/00_check_setup.py first.")
    return pd.read_csv(f)


def backbone_name() -> str:
    """Feature-cache key: the backbone's 'tag' (changes with fine-tuned weights), else its name."""
    f = art("backbone.json")
    if not f.exists():
        return "densenet121"
    info = json.loads(f.read_text())
    return info.get("tag", info["name"])


def load_xy(name, backbone):
    df = get_subset(name)
    X, d = load_features(feat_dir(name, backbone), df)
    return X, d[LABEL_COLS].values.astype(float), d


def get_policy() -> Policy:
    """policy.json if it exists, else a default policy (with Member B's calibration.json if present)."""
    f = art("policy.json")
    if f.exists():
        return Policy.load(f)
    temps = {}
    c = art("calibration.json")
    if c.exists():
        temps = json.loads(c.read_text()).get("temperature", {})
    return Policy(findings=list(FINDINGS), urgent=list(DEFAULT_URGENT), temperature=temps)


def run_info() -> dict:
    """Which model folder and version produced a report (every report states this)."""
    from chesttriage_ml.contract import model_version
    return {"artifacts_folder": art().name, "model_version": model_version(art())}


def stamp(df: pd.DataFrame) -> pd.DataFrame:
    info = run_info()
    df = df.copy()
    for k, v in reversed(list(info.items())):
        df.insert(0, k, v)
    return df


def stamp_line() -> str:
    i = run_info()
    return f"Artifacts folder: `{i['artifacts_folder']}`, model_version: `{i['model_version']}`"


def md_table(df: pd.DataFrame, nd: int = 3) -> str:
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else f"{v:.{nd}f}"
        return str(v)
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt(v) for v in r) + " |")
    return "\n".join(lines)


def urgent_idx(policy=None):
    p = policy or get_policy()
    return p.urgent_idx()
