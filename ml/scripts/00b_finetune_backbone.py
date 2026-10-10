"""Optional, timeboxed: fine-tune the LAST block of the ImageNet DenseNet121 on NIH chest X-rays.

Run after 00_check_setup.py and before 01_extract_features.py. Needs torch (training only).
  - trains features.denseblock4 + features.norm5 (+ a temporary 7-finding classifier); the rest stays frozen
  - data: NIH 'train' rows only; NIH 'val' for early stopping. NIH 'test' and CheXpert are never read,
    so the internal test and the cross-dataset test stay clean
  - stops at --time_budget_min or when validation AUROC stops improving (keeps the best epoch)
  - exports the fine-tuned backbone to ONNX (same graph as densenet121-onnx), so serving stays torch-free
  - writes backbone.json with tag 'densenet121-ft' (new feature cache); the linear head, OOD detector and
    policy are then refit on the new features by scripts 01-05, exactly as before
Fallback: keep a copy of the previous artifacts folder; switch back by restoring its backbone.json/.onnx.
"""
import _common  # noqa: F401
import argparse
import json
import time

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance

from chesttriage_ml.backbone import OnnxDenseNet121, read_safetensors
from chesttriage_ml.config import DEFAULT_URGENT, FINDINGS, IMG_SIZE, art
from chesttriage_ml.data import LABEL_COLS
from chesttriage_ml.metrics import auroc, composite_label
from chesttriage_ml.preprocess import load_image, to_array, to_gray
from _common import get_subset

TRAINABLE = ("features.denseblock4", "features.norm5", "classifier")


class XrayDataset:
    """Same preprocessing as feature extraction (to_array); mild augmentation for training only.
    No horizontal flips: a mirrored chest X-ray is anatomically wrong (heart on the right)."""

    def __init__(self, paths, labels, augment, seed=0):
        self.paths, self.labels, self.augment, self.seed = list(paths), np.asarray(labels, np.float32), augment, seed

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = to_gray(load_image(self.paths[i]))
        if self.augment:
            rng = np.random.default_rng()
            img = img.resize((IMG_SIZE, IMG_SIZE)).rotate(rng.uniform(-7, 7), resample=Image.BILINEAR, fillcolor=0)
            s = rng.uniform(0.85, 1.0); w = int(IMG_SIZE * s)
            x0, y0 = rng.integers(0, IMG_SIZE - w + 1, 2)
            img = img.crop((x0, y0, x0 + w, y0 + w))
            img = ImageEnhance.Contrast(ImageEnhance.Brightness(img).enhance(rng.uniform(0.9, 1.1))).enhance(rng.uniform(0.9, 1.1))
        return to_array(img), self.labels[i]


def build_model(torch, tvm):
    m = tvm.densenet121(weights=None)
    W = OnnxDenseNet121._fetch_weights()                     # torchvision ImageNet weights (cached safetensors)
    sd = {k: torch.from_numpy(np.array(v)) for k, v in W.items() if not k.startswith("classifier")}
    missing, unexpected = m.load_state_dict(sd, strict=False)
    assert not unexpected and all(k.startswith("classifier") for k in missing), (missing, unexpected)
    m.classifier = torch.nn.Linear(1024, len(FINDINGS))
    for n, p in m.named_parameters():
        p.requires_grad_(n.startswith(TRAINABLE))
    return m


def set_train_mode(m):
    m.train()
    for n, mod in m.named_modules():                         # frozen part keeps ImageNet BN statistics
        if n and not n.startswith(TRAINABLE):
            mod.eval()


def evaluate(torch, m, loader, device):
    m.eval()
    out, ys = [], []
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        for x, y in loader:
            out.append(torch.sigmoid(m(x.to(device, non_blocking=True).to(memory_format=torch.channels_last)).float()).cpu().numpy())
            ys.append(y.numpy())
    P, Y = np.concatenate(out), np.concatenate(ys)
    per = [auroc(Y[:, k], P[:, k]) for k in range(len(FINDINGS))]
    ui = [FINDINGS.index(f) for f in DEFAULT_URGENT]
    return float(np.nanmean(per)), auroc(composite_label(Y, ui), P[:, ui].max(1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--time_budget_min", type=float, default=45)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4, help="for the fine-tuned block")
    ap.add_argument("--head_lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=2, choices=[0, 1, 2],
                    help="0-2 only: more workers ran Windows out of paging-file memory")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--export_even_if_worse", action="store_true", help="testing only")
    a = ap.parse_args()

    import torch
    import torchvision.models as tvm
    torch.manual_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = get_subset("nih")
    tr, va = df[df.split == "train"], df[df.split == "val"]
    assert not set(tr.patient_id) & set(df[df.split == "test"].patient_id), "patient leakage"
    print(f"fine-tuning on {len(tr)} NIH train images, early stopping on {len(va)} NIH val images, device {device}")
    dl = lambda d, aug, sh: torch.utils.data.DataLoader(
        XrayDataset(d.path, d[LABEL_COLS].values, aug), batch_size=a.batch_size, shuffle=sh, num_workers=a.workers,
        pin_memory=True, persistent_workers=a.workers > 0, drop_last=sh)
    train_dl, val_dl = dl(tr, True, True), dl(va, False, False)

    m = build_model(torch, tvm).to(device).to(memory_format=torch.channels_last)
    params = [{"params": [p for n, p in m.named_parameters() if p.requires_grad and not n.startswith("classifier")], "lr": a.lr},
              {"params": m.classifier.parameters(), "lr": a.head_lr}]
    opt = torch.optim.AdamW(params, weight_decay=1e-4)
    steps = a.epochs * len(train_dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.lr, a.head_lr], total_steps=steps, pct_start=0.1)
    lossf = torch.nn.BCEWithLogitsLoss(reduction="none")

    base = evaluate(torch, m, val_dl, device)
    print(f"epoch 0 (ImageNet block, new classifier): val mean AUROC {base[0]:.3f}")
    t0, best, best_state, bad, log = time.time(), -1.0, None, 0, []
    for ep in range(1, a.epochs + 1):
        set_train_mode(m)
        tl, nb, te = 0.0, 0, time.time()
        for x, y in train_dl:
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last); y = y.to(device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logit = m(x)
            mask = ~torch.isnan(y)
            loss = (lossf(logit.float(), torch.nan_to_num(y)) * mask).sum() / mask.sum()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            tl += loss.item(); nb += 1
            if (time.time() - t0) / 60 > a.time_budget_min:
                break
        v_mean, v_urg = evaluate(torch, m, val_dl, device)
        log.append({"epoch": ep, "train_loss": tl / max(nb, 1), "val_mean_auroc": v_mean, "val_any_urgent_auroc": v_urg,
                    "minutes": (time.time() - te) / 60})
        print(f"epoch {ep}: loss {tl / max(nb, 1):.4f}  val mean AUROC {v_mean:.3f}  any-urgent {v_urg:.3f}  "
              f"({(time.time() - te) / 60:.1f} min)", flush=True)
        if v_mean > best:
            best, bad = v_mean, 0
            best_state = {k: v.detach().float().cpu().clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
        if bad >= a.patience or (time.time() - t0) / 60 > a.time_budget_min:
            print("stopping:", "no improvement" if bad >= a.patience else "time budget reached")
            break
    pd.DataFrame(log).to_csv(art("reports", "finetune_log.csv"), index=False)
    if best <= base[0] and not a.export_even_if_worse:
        raise SystemExit("Fine-tuning did not beat the ImageNet block on validation; backbone.json left unchanged.")

    # ---- export the best fine-tuned backbone to ONNX and check it against torch
    m.load_state_dict(best_state); m = m.cpu().eval().float().to(memory_format=torch.contiguous_format)
    W = {k: v.numpy() for k, v in best_state.items() if k.startswith("features.") and "num_batches" not in k}
    onnx_bytes = OnnxDenseNet121.build_onnx(W)
    wfile = "backbone_densenet121-ft.onnx"
    (art() / wfile).write_bytes(onnx_bytes)
    onx = OnnxDenseNet121(str(art() / wfile), device="cpu")
    x = np.stack([to_array(load_image(p)) for p in va.path.iloc[:8]])
    with torch.no_grad():
        ref = torch.relu(m.features(torch.from_numpy(x))).numpy()
    diff = float(np.abs(onx.fmap_np(x) - ref).max())
    print(f"ONNX vs torch feature-map max abs diff: {diff:.2e}")
    assert diff < 1e-3, "ONNX export does not match the fine-tuned torch model"
    old = art("backbone.json")
    if old.exists():
        art("backbone_previous.json").write_text(old.read_text())
    best_ep = max(log, key=lambda r: r["val_mean_auroc"])
    old.write_text(json.dumps({
        "name": "densenet121-onnx", "weights": wfile, "tag": "densenet121-ft",
        "source": "torchvision ImageNet DenseNet121; denseblock4+norm5 fine-tuned on NIH train only "
                  "(NIH val for early stopping). Never saw NIH test or CheXpert.",
        "val_mean_auroc_imagenet_block": base[0], "val_mean_auroc_finetuned": best, "best_epoch": best_ep["epoch"]}, indent=2))
    print(f"saved {wfile}; backbone.json now points to it (previous saved as backbone_previous.json). "
          "Next: 01_extract_features.py and the rest.")


if __name__ == "__main__":
    main()
