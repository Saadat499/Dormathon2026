"""Build tiny NIH-like and CheXpert-like folders (random images) to dry-run the whole pipeline.
The layouts imitate the real datasets; the images are synthetic, so metrics mean nothing."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

FIND = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion", "Pneumonia", "Pneumothorax"]
CX_COL = {**{f: f for f in FIND}, "Effusion": "Pleural Effusion"}


def make_image(pos, size, rng, shift=False):
    h, w = size
    yy, xx = np.mgrid[0:h, 0:w]
    img = 90 + 60 * (yy / h) + rng.normal(0, 12, (h, w))
    for k in pos:                                         # one blob per positive finding
        cy, cx = (k % 3 + 1) * h / 4, (k // 3 + 1) * w / 4
        img += 70 * np.exp(-(((yy - cy) / (h / 8)) ** 2 + ((xx - cx) / (w / 8)) ** 2))
    img = np.clip(img, 0, 255)
    if shift:                                             # a 'different scanner'
        img = 255 * (img / 255) ** 1.6 * 0.85
    return Image.fromarray(img.astype(np.uint8))


def build(root, n_nih=420, n_cx=300, seed=0):
    root, rng = Path(root), np.random.default_rng(seed)
    # ---- NIH-like
    d = root / "nih" / "versions" / "1"
    (d / "images_001" / "images").mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n_nih):
        pid = i // 3 + 1
        pos = [k for k in range(7) if rng.random() < [.1, .03, .04, .03, .12, .015, .05][k]]
        name = f"{pid:08d}_{i % 3:03d}.png"
        make_image(pos, (224, 224), rng).save(d / "images_001" / "images" / name)
        rows.append({"Image Index": name, "Finding Labels": "|".join(FIND[k] for k in pos) or "No Finding",
                     "Follow-up #": i % 3, "Patient ID": pid, "Patient Age": int(rng.integers(20, 80)),
                     "Patient Gender": rng.choice(["M", "F"]), "View Position": rng.choice(["PA", "AP"])})
    pd.DataFrame(rows).to_csv(d / "Data_Entry_2017.csv", index=False)
    # a decoy CSV that must NOT be picked
    pd.DataFrame({"Image Index": ["x.png"], "Finding Label": ["Mass"]}).to_csv(d / "BBox_List_2017.csv", index=False)
    # ---- CheXpert-like
    d = root / "chexpert" / "versions" / "1"
    for split, n in (("train", n_cx), ("valid", 30)):
        rows = []
        for i in range(n):
            pid = (i // 2 + 1) + (0 if split == "train" else 90000)
            rel = f"CheXpert-v1.0-small/{split}/patient{pid:05d}/study{i % 2 + 1}/view1_frontal.jpg"
            pos, row = [], {"Path": rel, "Sex": rng.choice(["Male", "Female"]), "Age": int(rng.integers(20, 90)),
                            "Frontal/Lateral": "Frontal", "AP/PA": rng.choice(["AP", "PA"])}
            for k, f in enumerate(FIND):
                u = rng.random()
                v = 1.0 if u < .12 else (-1.0 if u < .20 else (0.0 if u < .5 else np.nan))
                row[CX_COL[f]] = v
                if v == 1.0:
                    pos.append(k)
            p = d / rel; p.parent.mkdir(parents=True, exist_ok=True)
            make_image(pos, (320, 390), rng, shift=True).save(p)
            rows.append(row)
            if i % 7 == 0:                                   # a lateral view that must be filtered out
                rel2 = rel.replace("view1_frontal", "view2_lateral")
                (d / rel2).parent.mkdir(parents=True, exist_ok=True)
                make_image([], (320, 390), rng, True).save(d / rel2)
                rows.append({**row, "Path": rel2, "Frontal/Lateral": "Lateral"})
        pd.DataFrame(rows).to_csv(d / f"{split}.csv", index=False)
    return root


if __name__ == "__main__":
    import tempfile
    build(sys.argv[1] if len(sys.argv) > 1 else Path(tempfile.gettempdir()) / "chesttriage_synth")
