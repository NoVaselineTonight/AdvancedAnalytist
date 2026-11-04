"""RF-DETR through ONNX Runtime: GPU on any DirectX 12 card (AMD/Intel/NVIDIA) via DirectML, or CPU.

The ONNX file is exported once from the same Apache-2.0 RF-DETR COCO weights (``rfdetr`` does the export)
and cached in ``<project>/models``. Pre/post-processing mirrors ``RFDETR.predict``: bilinear resize without
antialiasing, ImageNet normalisation, per-class sigmoid, global top-k over query x class pairs.
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

from replay.detection.base import BALL, PERSON, Detections, Detector

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
COCO_PERSON, COCO_SPORTS_BALL = 1, 37
NUM_SELECT = 300
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)
_VARIANT_CLASS = {"nano": "RFDETRNano", "small": "RFDETRSmall", "medium": "RFDETRMedium", "large": "RFDETRLarge"}


def onnx_path(variant: str) -> Path:
    return MODELS_DIR / f"rfdetr-{variant}" / f"rfdetr-{variant}.onnx"


def ensure_onnx(variant: str) -> Path:
    """Export the ONNX model on first use (needs torch + rfdetr once; takes ~10 s)."""
    path = onnx_path(variant)
    if path.exists():
        return path
    import rfdetr

    model = getattr(rfdetr, _VARIANT_CLASS[variant])()
    out = Path(model.export(output_dir=str(path.parent), format="onnx", verbose=False))
    if out != path:
        out.replace(path)
    return path


def available_providers() -> list[str]:
    import onnxruntime as ort

    return ort.get_available_providers()


class ONNXDetector(Detector):
    license_note = "RF-DETR COCO weights: Apache-2.0 (Roboflow); ONNX Runtime: MIT"

    def __init__(self, variant: str = "medium", threshold: float = 0.3, device: str = "auto"):
        import onnxruntime as ort

        path = ensure_onnx(variant)
        avail = ort.get_available_providers()
        gpu_ep = next((p for p in ("CUDAExecutionProvider", "DmlExecutionProvider") if p in avail), None)
        if device == "gpu" and gpu_ep is None:
            raise RuntimeError("GPU requested but no GPU runtime found (Windows: pip install onnxruntime-directml; "
                               "NVIDIA: onnxruntime-gpu)")
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if device in ("auto", "gpu") and gpu_ep == "DmlExecutionProvider":
            # DirectML requires sequential execution and no memory-pattern optimisation
            so.enable_mem_pattern = False
            so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            providers = [("DmlExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
        elif device in ("auto", "gpu") and gpu_ep == "CUDAExecutionProvider":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            so.intra_op_num_threads = max(1, (os.cpu_count() or 2) // 2)  # physical cores
            providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(path), sess_options=so, providers=providers)
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.h, self.w = int(inp.shape[2]), int(inp.shape[3])
        outs = [o.name for o in self.session.get_outputs()]
        self.box_name = next(n for n in outs if "box" in n.lower() or n == "dets")
        self.logit_name = next(n for n in outs if n != self.box_name)
        self.threshold = threshold
        ep = self.session.get_providers()[0]
        self.device = {"DmlExecutionProvider": "GPU (DirectML)", "CUDAExecutionProvider": "GPU (CUDA)"}.get(ep, "CPU")
        self.name = f"rfdetr-{variant} on {self.device}"

    def preprocess(self, image_bgr: np.ndarray) -> np.ndarray:
        """Resize, BGR->RGB, ImageNet normalisation, NCHW float32. Computed per channel straight into the
        output (no float HWC intermediates or transposes): ~3 ms instead of ~9-14 ms at 576x576. Same float32
        operations in the same order as ``(rgb / 255 - mean) / std``, so bit-identical: RF-DETR's query
        selection makes the output sensitive to even 1e-7 input changes."""
        img = cv2.resize(image_bgr, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        out = np.empty((1, 3, self.h, self.w), np.float32)
        for c in range(3):  # RGB channel c = BGR channel 2 - c
            o = out[0, c]
            np.multiply(img[:, :, 2 - c], np.float32(1.0 / 255.0), out=o, dtype=np.float32)
            np.subtract(o, _MEAN[c], out=o)
            np.divide(o, _STD[c], out=o)
        return out

    def detect(self, image_bgr: np.ndarray) -> Detections:
        return self.detect_prepared(self.preprocess(image_bgr), image_bgr)

    def prepare(self, image_bgr: np.ndarray) -> np.ndarray:
        return self.preprocess(image_bgr)

    def detect_prepared(self, prepared: np.ndarray, image_bgr: np.ndarray) -> Detections:
        H, W = image_bgr.shape[:2]
        boxes, logits = self.session.run([self.box_name, self.logit_name], {self.input_name: prepared})
        return self.decode(boxes[0], logits[0], W, H)

    def decode(self, boxes: np.ndarray, logits: np.ndarray, W: int, H: int) -> Detections:
        # Only the two classes we use are scored; ranking within them is identical to the full top-k.
        cols = [c for c in (COCO_PERSON, COCO_SPORTS_BALL) if c < logits.shape[1]]
        scores = 1.0 / (1.0 + np.exp(-np.clip(logits[:, cols], -88, 88)))
        flat = scores.ravel()
        k = min(NUM_SELECT, flat.size)
        idx = np.argpartition(-flat, k - 1)[:k]
        idx = idx[flat[idx] > self.threshold]
        if idx.size == 0:
            return Detections.empty()
        q, c = np.divmod(idx, len(cols))
        cx, cy, bw, bh = boxes[q].T
        xyxy = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1) * np.array([W, H, W, H], np.float32)
        xyxy = np.clip(xyxy, 0, [W, H, W, H]).astype(np.float32)
        cls = np.where(np.array(cols)[c] == COCO_PERSON, PERSON, BALL).astype(np.int32)
        return Detections(xyxy, flat[idx].astype(np.float32), cls)
