"""Cached feature extraction (resumable, shard by shard). Image I/O uses threads, no torch."""
from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .config import IMG_SIZE
from .preprocess import load_image, to_array

try:
    from tqdm import tqdm
except Exception:                                       # tqdm optional
    def tqdm(x, **k):
        return x


def _load_one(path):
    try:
        return to_array(load_image(path)), 1
    except Exception:
        return np.zeros((3, IMG_SIZE, IMG_SIZE), np.float32), 0


def _stamp(ids) -> str:
    return hashlib.md5("|".join(ids).encode()).hexdigest()


def extract(df: pd.DataFrame, backbone, cache_dir, batch_size: int = 64,
            workers: int = 8, shard_size: int = 2048) -> None:
    """Write feats.npy / ok.npy into cache_dir. Re-running resumes from finished shards."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    stamp = _stamp(df["study_id"].tolist())
    sf = cache_dir / "ids.md5"
    if sf.exists() and sf.read_text() != stamp:
        raise RuntimeError(f"{cache_dir} was built for a different subset. Delete it and re-run.")
    sf.write_text(stamp)
    paths = df["path"].tolist()
    n_shards = (len(paths) + shard_size - 1) // shard_size
    with ThreadPoolExecutor(workers) as ex:
        for s in range(n_shards):
            f = cache_dir / f"shard_{s:04d}.npz"
            if f.exists():
                continue
            chunk = paths[s * shard_size:(s + 1) * shard_size]
            feats, oks = [], []
            for b in tqdm(range(0, len(chunk), batch_size), desc=f"shard {s + 1}/{n_shards}"):
                res = list(ex.map(_load_one, chunk[b:b + batch_size]))
                x = np.stack([r[0] for r in res])
                feats.append(backbone.embed_np(x).astype(np.float32))
                oks.append(np.array([r[1] for r in res], dtype=np.int8))
            np.savez(f, feats=np.concatenate(feats), ok=np.concatenate(oks))
    parts = [np.load(cache_dir / f"shard_{s:04d}.npz") for s in range(n_shards)]
    np.save(cache_dir / "feats.npy", np.concatenate([p["feats"] for p in parts]))
    np.save(cache_dir / "ok.npy", np.concatenate([p["ok"] for p in parts]))


def load_features(cache_dir, df: pd.DataFrame):
    """Return (X, df_ok): features aligned to df, with unreadable images dropped."""
    cache_dir = Path(cache_dir)
    if (cache_dir / "ids.md5").read_text() != _stamp(df["study_id"].tolist()):
        raise RuntimeError(f"{cache_dir} does not match this subset. Delete it and re-run extraction.")
    X, ok = np.load(cache_dir / "feats.npy"), np.load(cache_dir / "ok.npy").astype(bool)
    if len(X) != len(df):
        raise RuntimeError("feature count differs from subset size")
    if (~ok).sum():
        print(f"[features] dropping {int((~ok).sum())} unreadable images")
    return X[ok], df[ok].reset_index(drop=True)


def embed_pils(backbone, pils, batch_size: int = 64) -> np.ndarray:
    out = []
    for b in range(0, len(pils), batch_size):
        x = np.stack([to_array(p) for p in pils[b:b + batch_size]])
        out.append(backbone.embed_np(x).astype(np.float32))
    return np.concatenate(out) if out else np.zeros((0, backbone.dim), np.float32)
