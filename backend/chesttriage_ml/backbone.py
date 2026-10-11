"""Frozen pretrained backbones. torch is imported lazily, only here.

Interface used everywhere else (numpy in, numpy out):
    fmap_np(x) : (N,3,224,224) float32 -> (N,C,h,w) last feature map (post-ReLU)
    embed_np(x): (N,3,224,224) float32 -> (N,C)     global-average-pooled embedding

Default weights are ImageNet. Do NOT use chest X-ray weights trained on NIH or CheXpert
(e.g. TorchXRayVision 'all' models): that contaminates the cross-dataset test.

name='fake' is a tiny numpy stand-in for dry runs and tests (no torch needed).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SUPPORTED = ("densenet121", "resnet50", "resnet18", "densenet121-onnx", "fake")


class FakeBackbone:
    """Deterministic random-projection 'backbone' for pipeline tests. Not a real model."""
    name = "fake"
    dim = 64

    def __init__(self, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.proj = rng.normal(size=(self.dim, 3 * 4)).astype(np.float32)

    def fmap_np(self, x: np.ndarray) -> np.ndarray:
        n = x.shape[0]
        g = x.mean(axis=1)                                        # (N,224,224)
        p = g.reshape(n, 7, 32, 7, 32).mean(axis=(2, 4))          # (N,7,7) block means
        # 12 simple per-cell features: value, squares, shifts
        feats = np.stack([p, p ** 2, np.roll(p, 1, 1), np.roll(p, 1, 2)] * 3, axis=1)  # (N,12,7,7)
        f = np.einsum("dk,nkhw->ndhw", self.proj, feats)
        return np.maximum(f, 0).astype(np.float32)

    def embed_np(self, x: np.ndarray) -> np.ndarray:
        return self.fmap_np(x).mean(axis=(2, 3))

    def save(self, path) -> None:
        pass


class TorchBackbone:
    def __init__(self, name: str = "densenet121", pretrained: bool = True,
                 device: str | None = None, weights_path: str | None = None):
        import torch
        import torch.nn as nn
        import torchvision.models as tvm

        self.torch, self.name = torch, name
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.relu_after = False
        use_pre = pretrained and weights_path is None
        if name == "densenet121":
            m = tvm.densenet121(weights=tvm.DenseNet121_Weights.IMAGENET1K_V1 if use_pre else None)
            self.body, self.dim, self.relu_after = m.features, 1024, True
        elif name == "resnet50":
            m = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2 if use_pre else None)
            self.body, self.dim = nn.Sequential(*list(m.children())[:-2]), 2048
        elif name == "resnet18":
            m = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1 if use_pre else None)
            self.body, self.dim = nn.Sequential(*list(m.children())[:-2]), 512
        else:
            raise ValueError(f"unknown backbone {name}")
        if weights_path:
            self.body.load_state_dict(torch.load(weights_path, map_location="cpu"))
        self.body.to(self.device).eval()
        for p in self.body.parameters():
            p.requires_grad_(False)

    def fmap_np(self, x: np.ndarray) -> np.ndarray:
        torch = self.torch
        with torch.no_grad():
            t = torch.from_numpy(np.ascontiguousarray(x)).to(self.device)
            with torch.autocast(device_type=self.device.type, dtype=torch.float16,
                                enabled=self.device.type == "cuda"):
                f = self.body(t)
                if self.relu_after:
                    f = torch.relu(f)
            return f.float().cpu().numpy()

    def embed_np(self, x: np.ndarray) -> np.ndarray:
        return self.fmap_np(x).mean(axis=(2, 3))

    def save(self, path) -> None:
        self.torch.save(self.body.state_dict(), path)


class OnnxDenseNet121:
    """torchvision's ImageNet DenseNet121 (same weights, same maths) run by ONNX Runtime, no torch.

    Use it where torch cannot load (e.g. Windows Smart App Control blocks torch's unsigned DLLs)
    or for a light deploy (onnxruntime is far smaller than torch). The graph is rebuilt from
    torchvision's weights as published by timm ('densenet121.tv_in1k', safetensors), so features
    match TorchBackbone('densenet121'). GPU via DirectML or CUDA when available, else CPU.
    """
    name = "densenet121-onnx"
    dim = 1024
    WEIGHTS_URL = "https://huggingface.co/timm/densenet121.tv_in1k/resolve/main/model.safetensors"
    BLOCKS, GROWTH, BN_SIZE = (6, 12, 24, 16), 32, 4

    def __init__(self, onnx_path: str | None = None, device: str | None = None):
        import onnxruntime as ort
        self._bytes = Path(onnx_path).read_bytes() if onnx_path else self.build_onnx(self._fetch_weights())
        avail = ort.get_available_providers()
        if device == "cpu":
            prov = ["CPUExecutionProvider"]
        else:
            prov = [p for p in ("CUDAExecutionProvider", "DmlExecutionProvider") if p in avail][:1] + ["CPUExecutionProvider"]
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.sess = ort.InferenceSession(self._bytes, so, providers=prov)
        self.provider = self.sess.get_providers()[0]

    @classmethod
    def _fetch_weights(cls) -> dict:
        import urllib.request
        cache = Path.home() / ".cache" / "chesttriage" / "densenet121_tv_in1k.safetensors"
        if not cache.exists():
            cache.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(cls.WEIGHTS_URL, cache)
        return read_safetensors(cache)

    @classmethod
    def build_onnx(cls, W: dict) -> bytes:
        """ONNX graph for torchvision densenet121 `features` + final ReLU -> (N,1024,7,7)."""
        import onnx
        from onnx import TensorProto, helper, numpy_helper
        nodes, inits, n = [], [], [0]

        def name(p):
            n[0] += 1
            return f"{p}_{n[0]}"

        def init(key):
            inits.append(numpy_helper.from_array(np.ascontiguousarray(W[key], np.float32), key))
            return key

        def op(kind, inputs, **attrs):
            out = name(kind)
            nodes.append(helper.make_node(kind, inputs, [out], **attrs))
            return out

        def bn(x, p):
            return op("BatchNormalization", [x, init(p + ".weight"), init(p + ".bias"),
                                             init(p + ".running_mean"), init(p + ".running_var")], epsilon=1e-5)

        def conv(x, p, k, s=1, pad=0):
            return op("Conv", [x, init(p + ".weight")], kernel_shape=[k, k], strides=[s, s], pads=[pad] * 4)

        f = "features."
        x = conv("input", f + "conv0", 7, 2, 3)
        x = op("Relu", [bn(x, f + "norm0")])
        x = op("MaxPool", [x], kernel_shape=[3, 3], strides=[2, 2], pads=[1, 1, 1, 1])
        for bi, nl in enumerate(cls.BLOCKS, 1):
            feats = [x]
            for li in range(1, nl + 1):
                p = f"{f}denseblock{bi}.denselayer{li}."
                cat = feats[0] if len(feats) == 1 else op("Concat", feats, axis=1)
                h = conv(op("Relu", [bn(cat, p + "norm1")]), p + "conv1", 1)
                h = conv(op("Relu", [bn(h, p + "norm2")]), p + "conv2", 3, 1, 1)
                feats.append(h)
            x = op("Concat", feats, axis=1)
            if bi < len(cls.BLOCKS):
                p = f"{f}transition{bi}."
                x = conv(op("Relu", [bn(x, p + "norm")]), p + "conv", 1)
                x = op("AveragePool", [x], kernel_shape=[2, 2], strides=[2, 2])
        x = op("Relu", [bn(x, f + "norm5")])
        nodes[-1].output[0] = "fmap"
        g = helper.make_graph(nodes, "densenet121_features",
                              [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", 3, 224, 224])],
                              [helper.make_tensor_value_info("fmap", TensorProto.FLOAT, ["N", 1024, 7, 7])], inits)
        m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)], producer_name="chesttriage")
        m.ir_version = 8
        onnx.checker.check_model(m)
        return m.SerializeToString()

    def fmap_np(self, x: np.ndarray) -> np.ndarray:
        return self.sess.run(["fmap"], {"input": np.ascontiguousarray(x, np.float32)})[0]

    def embed_np(self, x: np.ndarray) -> np.ndarray:
        return self.fmap_np(x).mean(axis=(2, 3))

    def save(self, path) -> None:
        Path(path).write_bytes(self._bytes)


def read_safetensors(path) -> dict:
    """Minimal numpy reader for .safetensors (avoids another compiled dependency)."""
    import struct
    b = Path(path).read_bytes()
    n = struct.unpack("<Q", b[:8])[0]
    head = json.loads(b[8:8 + n])
    dt = {"F32": np.float32, "F16": np.float16, "I64": np.int64}
    out = {}
    for k, v in head.items():
        if k == "__metadata__" or v["dtype"] not in dt:
            continue
        s, e = v["data_offsets"]
        out[k] = np.frombuffer(b, dt[v["dtype"]], count=(e - s) // np.dtype(dt[v["dtype"]]).itemsize,
                               offset=8 + n + s).reshape(v["shape"])
    return out


def get_backbone(name: str, pretrained: bool = True, weights_path: str | None = None,
                 device: str | None = None):
    if name not in SUPPORTED:
        raise ValueError(f"backbone must be one of {SUPPORTED}")
    if name == "fake":
        return FakeBackbone()
    if name == OnnxDenseNet121.name:
        return OnnxDenseNet121(weights_path, device)
    return TorchBackbone(name, pretrained, device, weights_path)


def save_backbone_info(art_dir: Path, bb) -> None:
    """Persist weights so the live engine runs offline (no internet at deploy time)."""
    wfile = None
    if bb.name != "fake":
        wfile = f"backbone_{bb.name}.{'onnx' if bb.name.endswith('-onnx') else 'pt'}"
        bb.save(art_dir / wfile)
    (art_dir / "backbone.json").write_text(json.dumps({"name": bb.name, "weights": wfile}))


def load_backbone_from_artifacts(art_dir: Path, device: str | None = None):
    info = json.loads((art_dir / "backbone.json").read_text())
    wp = str(art_dir / info["weights"]) if info.get("weights") else None
    return get_backbone(info["name"], pretrained=False, weights_path=wp, device=device)
