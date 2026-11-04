"""Detector interface. Every backend returns people and balls in our own class ids."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

PERSON = 0
BALL = 1


@dataclass
class Detections:
    xyxy: np.ndarray  # (N, 4) float32, pixels in the full frame
    conf: np.ndarray  # (N,) float32
    cls: np.ndarray  # (N,) int32, PERSON or BALL

    @staticmethod
    def empty() -> "Detections":
        return Detections(np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int32))

    def __len__(self) -> int:
        return len(self.conf)

    def subset(self, mask: np.ndarray) -> "Detections":
        return Detections(self.xyxy[mask], self.conf[mask], self.cls[mask])

    @staticmethod
    def concat(items: list["Detections"]) -> "Detections":
        items = [d for d in items if len(d)]
        if not items:
            return Detections.empty()
        return Detections(np.concatenate([d.xyxy for d in items]), np.concatenate([d.conf for d in items]),
                          np.concatenate([d.cls for d in items]))


class Detector(ABC):
    name: str = "base"
    license_note: str = ""

    @abstractmethod
    def detect(self, image_bgr: np.ndarray) -> Detections:
        ...

    # Split detection for pipelining: ``prepare`` (CPU pre-processing, may run in another thread) and
    # ``detect_prepared`` (inference + decoding). detect_prepared(prepare(img), img) == detect(img).
    def prepare(self, image_bgr: np.ndarray) -> object:
        return None

    def detect_prepared(self, prepared: object, image_bgr: np.ndarray) -> Detections:
        return self.detect(image_bgr)


class TiledDetector(Detector):
    """Runs an inner detector on overlapping tiles and merges the results.

    Useful when people are small relative to the frame: the detector resizes
    its input to a fixed resolution, so tiling raises the effective resolution.
    """

    def __init__(self, inner: Detector, rows: int, cols: int, overlap: float = 0.15):
        self.inner, self.rows, self.cols, self.overlap = inner, rows, cols, overlap
        self.name = f"{inner.name}[tiled {rows}x{cols}]"
        self.license_note = inner.license_note

    def tiles(self, w: int, h: int) -> list[tuple[int, int, int, int]]:
        tw = w / (self.cols - (self.cols - 1) * self.overlap)
        th = h / (self.rows - (self.rows - 1) * self.overlap)
        out = []
        for r in range(self.rows):
            for c in range(self.cols):
                x0 = int(round(c * tw * (1 - self.overlap)))
                y0 = int(round(r * th * (1 - self.overlap)))
                out.append((x0, y0, min(w, int(round(x0 + tw))), min(h, int(round(y0 + th)))))
        return out

    def detect(self, image_bgr: np.ndarray) -> Detections:
        h, w = image_bgr.shape[:2]
        parts = []
        for x0, y0, x1, y1 in self.tiles(w, h):
            d = self.inner.detect(image_bgr[y0:y1, x0:x1])
            if len(d):
                d.xyxy = d.xyxy + np.array([x0, y0, x0, y0], np.float32)
                parts.append(d)
        return merge_overlapping(Detections.concat(parts))


def merge_overlapping(d: Detections, ios_threshold: float = 0.7) -> Detections:
    """Greedy suppression by intersection-over-smaller-box, per class.

    IoU-NMS keeps a box cut in half at a tile border next to the full box
    (IoU ~0.5); intersection-over-smaller removes it.
    """
    if len(d) == 0:
        return d
    area = (d.xyxy[:, 2] - d.xyxy[:, 0]) * (d.xyxy[:, 3] - d.xyxy[:, 1])
    order = np.lexsort((-d.conf, -area))  # prefer larger, then more confident
    keep: list[int] = []
    for i in order:
        ok = True
        for j in keep:
            if d.cls[i] != d.cls[j]:
                continue
            ix = max(0.0, min(d.xyxy[i, 2], d.xyxy[j, 2]) - max(d.xyxy[i, 0], d.xyxy[j, 0]))
            iy = max(0.0, min(d.xyxy[i, 3], d.xyxy[j, 3]) - max(d.xyxy[i, 1], d.xyxy[j, 1]))
            if ix * iy / max(1e-6, min(area[i], area[j])) > ios_threshold:
                ok = False
                break
        if ok:
            keep.append(int(i))
    return d.subset(np.array(sorted(keep), int))
