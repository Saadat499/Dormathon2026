"""Dataset discovery, label parsing, patient-level subsampling and splits.

Standard columns produced for both datasets:
  study_id, dataset, path, patient_id, split, age, sex, view, y_<finding> ...
Label values: 1.0 positive, 0.0 negative, NaN = masked (CheXpert 'uncertain').
We cannot know the exact folder layout of each Kaggle copy in advance, so files are
located by content (CSV columns) and images by name, not by hard-coded paths.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import FINDINGS, NIH_NAME, CHEXPERT_NAME

IMG_EXT = {".jpg", ".jpeg", ".png"}
LABEL_COLS = [f"y_{f}" for f in FINDINGS]


# --------------------------------------------------------------------- discovery
def _walk(root, max_depth=4):
    root = Path(root)
    base = len(root.parts)
    for dp, dn, fn in os.walk(root):
        if len(Path(dp).parts) - base >= max_depth:
            dn[:] = []
        for f in fn:
            yield Path(dp) / f


def find_csvs(root, required_cols, max_depth=4):
    out = []
    for p in _walk(root, max_depth):
        if p.suffix.lower() == ".csv":
            try:
                cols = set(pd.read_csv(p, nrows=0).columns)
            except Exception:
                continue
            if set(required_cols) <= cols:
                out.append(p)
    return out


def index_images(root, key_parts: int = 1) -> dict:
    """Map the last `key_parts` path components (posix) -> absolute path.
    Each image is indexed twice: with its extension and without (so .png vs .jpg copies still match)."""
    idx = {}
    for p in _walk(root, max_depth=12):
        if p.suffix.lower() in IMG_EXT:
            key = "/".join(p.parts[-key_parts:])
            idx.setdefault(key, str(p))
            idx.setdefault(key.rsplit(".", 1)[0], str(p))
    return idx


def _lookup(keys: pd.Series, idx: dict) -> pd.Series:
    """Exact key first, then the key without its extension."""
    hit = keys.map(idx)
    miss = hit.isna()
    if miss.any():
        hit[miss] = keys[miss].map(lambda k: idx.get(k.rsplit(".", 1)[0]))
    return hit


def _need_images(df, what, root):
    if len(df) == 0:
        raise FileNotFoundError(f"[{what}] none of the CSV rows matched an image file under {root}. "
                                "Check the dataset folder layout / file names.")


# --------------------------------------------------------------------- helpers
def subsample_by_group(df: pd.DataFrame, group_col: str, n: int | None, seed: int) -> pd.DataFrame:
    if n is None or n >= len(df):
        return df
    rng = np.random.default_rng(seed)
    groups = df[group_col].unique()
    rng.shuffle(groups)
    sizes = df.groupby(group_col).size().reindex(groups).values
    k = int(np.searchsorted(np.cumsum(sizes), n)) + 1
    return df[df[group_col].isin(set(groups[:k]))]


def split_by_group(df: pd.DataFrame, group_col: str, fracs=(0.7, 0.1, 0.2), seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    groups = df[group_col].unique()
    rng.shuffle(groups)
    n = len(groups)
    a, b = int(fracs[0] * n), int((fracs[0] + fracs[1]) * n)
    m = {g: "train" for g in groups[:a]}
    m.update({g: "val" for g in groups[a:b]})
    m.update({g: "test" for g in groups[b:]})
    return df[group_col].map(m)


def _dedupe(df: pd.DataFrame, key: str, what: str) -> pd.DataFrame:
    """One row per image, so no study is counted twice or lands in two splits."""
    n = len(df)
    df = df.drop_duplicates(subset=[key]).copy()
    if len(df) < n:
        print(f"[{what}] dropped {n - len(df)} duplicate rows")
    return df


def _safe_id(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", s)


# --------------------------------------------------------------------- NIH
def load_nih(root, n_images: int | None = 30000, seed: int = 42, fracs=(0.7, 0.1, 0.2)) -> pd.DataFrame:
    tables = find_csvs(root, ["Image Index", "Finding Labels"])
    if not tables:
        raise FileNotFoundError(f"No NIH label CSV (needs 'Image Index' and 'Finding Labels') under {root}")
    tables.sort(key=lambda p: (0 if "data_entry" in p.name.lower() else 1, -p.stat().st_size))
    df = pd.read_csv(tables[0])
    img_idx = index_images(root, 1)
    df["path"] = _lookup(df["Image Index"], img_idx)
    n_missing = int(df["path"].isna().sum())
    if n_missing:
        print(f"[nih] {n_missing} of {len(df)} CSV rows have no image file; dropped")
    df = _dedupe(df.dropna(subset=["path"]), "Image Index", "nih")
    _need_images(df, "nih", root)
    if "Patient ID" not in df.columns:
        df["Patient ID"] = df["Image Index"].str.split("_").str[0]
    sets = df["Finding Labels"].fillna("").str.split("|")
    for f in FINDINGS:
        df[f"y_{f}"] = sets.apply(lambda l, name=NIH_NAME[f]: float(name in l))
    df = subsample_by_group(df, "Patient ID", n_images, seed).reset_index(drop=True)
    out = pd.DataFrame({
        "study_id": "nih_" + df["Image Index"].map(lambda s: _safe_id(Path(s).stem)),
        "dataset": "nih",
        "path": df["path"].values,
        "patient_id": "nih_" + df["Patient ID"].astype(str),
        # some copies store age as "058Y": keep the digits
        "age": pd.to_numeric(df["Patient Age"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
               if "Patient Age" in df else np.nan,
        "sex": df["Patient Gender"] if "Patient Gender" in df else None,
        "view": df["View Position"] if "View Position" in df else None,
    }).reset_index(drop=True)
    for c in LABEL_COLS:
        out[c] = df[c].values
    out["split"] = split_by_group(out, "patient_id", fracs, seed).values
    return out


# --------------------------------------------------------------------- CheXpert
def _chexpert_frame(csv: Path, img_idx: dict, split: str, blank_as: float) -> pd.DataFrame:
    df = pd.read_csv(csv)
    if "Frontal/Lateral" in df.columns:
        df = df[df["Frontal/Lateral"] == "Frontal"]
    key = df["Path"].map(lambda s: "/".join(Path(s).parts[-3:]))
    df = df.assign(_path=_lookup(key, img_idx))
    miss = int(df["_path"].isna().sum())
    if miss:
        print(f"[chexpert:{csv.name}] {miss} of {len(df)} rows have no image file; dropped")
    df = _dedupe(df.dropna(subset=["_path"]), "Path", f"chexpert:{csv.name}").reset_index(drop=True)
    _need_images(df, f"chexpert:{csv.name}", csv.parent)
    out = pd.DataFrame({
        "study_id": "cx_" + df["Path"].map(lambda s: _safe_id("_".join(Path(s).parts[-3:]).rsplit(".", 1)[0])),
        "dataset": "chexpert",
        "path": df["_path"].values,
        "patient_id": "cx_" + df["Path"].str.extract(r"(patient\d+)")[0].fillna("unknown"),
        "age": pd.to_numeric(df["Age"], errors="coerce") if "Age" in df else np.nan,
        "sex": df["Sex"] if "Sex" in df else None,
        "view": df["AP/PA"] if "AP/PA" in df else None,
    }).reset_index(drop=True)
    for f in FINDINGS:
        col = df[CHEXPERT_NAME[f]].astype(float)
        col = col.fillna(blank_as)                 # blank = not mentioned -> negative
        col = col.where(col != -1.0, np.nan)       # uncertain -> masked (ignored)
        out[f"y_{f}"] = col.values
    out["split"] = split
    return out


def load_chexpert(root, n_images: int | None = 15000, seed: int = 42, blank_as: float = 0.0) -> pd.DataFrame:
    tables = find_csvs(root, ["Path", "Pleural Effusion"])
    if not tables:
        raise FileNotFoundError(f"No CheXpert CSV (needs 'Path' and 'Pleural Effusion') under {root}")
    train = [p for p in tables if "train" in p.name.lower()]
    valid = [p for p in tables if "valid" in p.name.lower()]
    if not train:
        train = tables[:1]
    img_idx = index_images(root, 3)
    tr = _chexpert_frame(train[0], img_idx, "external", blank_as)
    tr = subsample_by_group(tr, "patient_id", n_images, seed)
    parts = [tr]
    if valid:
        va = _chexpert_frame(valid[0], img_idx, "gold_valid", blank_as)
        parts.append(va[~va["study_id"].isin(set(tr["study_id"]))])
    return pd.concat(parts, ignore_index=True)


# --------------------------------------------------------------------- reporting
def label_overlap_table(nih: pd.DataFrame, cx: pd.DataFrame, urgent) -> pd.DataFrame:
    rows = []
    cx_ext = cx[cx["split"] == "external"]
    for f in FINDINGS:
        c = f"y_{f}"
        rows.append({
            "finding": f, "in_nih_as": NIH_NAME[f], "in_chexpert_as": CHEXPERT_NAME[f],
            "nih_pos_rate": nih[c].mean(), "nih_n_pos": int((nih[c] == 1).sum()),
            "chexpert_pos_rate": cx_ext[c].mean(), "chexpert_n_pos": int((cx_ext[c] == 1).sum()),
            "chexpert_uncertain_rate": cx_ext[c].isna().mean(),
            "provisional_urgent": f in urgent,
        })
    return pd.DataFrame(rows)
