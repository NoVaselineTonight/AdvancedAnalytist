from replay.config import DetectorConfig
from replay.detection.base import BALL, PERSON, Detections, Detector, TiledDetector


def build_detector(cfg: DetectorConfig) -> Detector:
    """Create the configured detector. Add new backends here (e.g. an ONNX Runtime one)."""
    backend = cfg.backend
    if backend == "auto":
        backend = "onnx" if cfg.device != "cpu" and _gpu_available() else "torch"
    if backend == "onnx":
        from replay.detection.onnx_backend import ONNXDetector

        det: Detector = ONNXDetector(cfg.variant, cfg.threshold, device=cfg.device)
    elif backend in ("torch", "rfdetr"):
        from replay.detection.rfdetr_backend import RFDETRDetector

        det = RFDETRDetector(cfg.variant, cfg.threshold, cfg.resolution)
    else:
        raise ValueError(f"Unknown detector backend: {cfg.backend!r}")
    rows, cols = cfg.tiles
    if rows * cols > 1:
        det = TiledDetector(det, rows, cols, cfg.tile_overlap)
    store = getattr(cfg, "store_threshold", None)
    if store is not None and store < cfg.threshold and hasattr(det, "threshold"):
        det = StoreLowDetector(det, cfg.threshold, store)
    return det


class StoreLowDetector(Detector):
    """Task 9: runs the inner detector at ``store`` and keeps person boxes >= store, balls only >= threshold."""

    def __init__(self, inner: Detector, threshold: float, store: float):
        self.inner, self.activation, self.store = inner, float(threshold), float(store)
        inner.threshold = self.store
        self.name = f"{inner.name} [persons >= {self.store:g}]"
        self.license_note = inner.license_note

    def _filter(self, d: Detections) -> Detections:
        return d.subset((d.cls == PERSON) | (d.conf > self.activation))

    def detect(self, image_bgr):
        return self._filter(self.inner.detect(image_bgr))

    def prepare(self, image_bgr):
        return self.inner.prepare(image_bgr)

    def detect_prepared(self, prepared, image_bgr):
        return self._filter(self.inner.detect_prepared(prepared, image_bgr))


def _gpu_available() -> bool:
    try:
        from replay.detection.onnx_backend import available_providers

        return "DmlExecutionProvider" in available_providers() or "CUDAExecutionProvider" in available_providers()
    except ImportError:
        return False


def activation_threshold(signature: dict | None) -> float:
    """Detector threshold of a pass-1 run: rows scoring below it are Task-9 low-score crumbs (0 = none stored)."""
    det = (signature or {}).get("detector") or {}
    return float(det.get("threshold", 0.0)) if det.get("store_threshold") is not None else 0.0


__all__ = ["BALL", "PERSON", "Detections", "Detector", "StoreLowDetector", "TiledDetector", "activation_threshold",
           "build_detector"]
