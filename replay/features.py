"""Appearance features for team classification.

Per person box we store the median CIE-Lab colour of the torso (jersey) and
of the shorts region. Medians are robust to arms, numbers and a bit of
background. Lab is used so that Euclidean distance roughly matches perceived
colour difference.
"""

from __future__ import annotations

import cv2
import numpy as np

FEATURE_DIM = 6
TORSO = (0.18, 0.48, 0.28, 0.72)  # (y0, y1, x0, x1) as fractions of the box
SHORTS = (0.50, 0.66, 0.30, 0.70)


def _region_median_lab(image_bgr: np.ndarray, box: np.ndarray, frac: tuple[float, float, float, float]) -> np.ndarray:
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    ya, yb = int(y0 + frac[0] * h), int(y0 + frac[1] * h)
    xa, xb = int(x0 + frac[2] * w), int(x0 + frac[3] * w)
    H, W = image_bgr.shape[:2]
    ya, yb, xa, xb = max(0, ya), min(H, yb), max(0, xa), min(W, xb)
    if yb - ya < 2 or xb - xa < 2:
        return np.full(3, np.nan, np.float32)
    # the colour conversion is per pixel, so converting only the patch equals slicing a converted frame
    lab = cv2.cvtColor(np.ascontiguousarray(image_bgr[ya:yb, xa:xb]), cv2.COLOR_BGR2LAB)
    return np.median(lab.reshape(-1, 3).astype(np.float32), axis=0)


def bgr_to_lab_float(lab_u8: np.ndarray) -> np.ndarray:
    """OpenCV 8-bit Lab -> real Lab (L 0..100, a/b roughly -128..127)."""
    lab = lab_u8.astype(np.float32)
    return np.stack([lab[..., 0] * 100.0 / 255.0, lab[..., 1] - 128.0, lab[..., 2] - 128.0], axis=-1)


def rgb_to_lab(rgb: tuple[int, int, int]) -> np.ndarray:
    px = np.array([[rgb[::-1]]], np.uint8)  # to BGR
    return bgr_to_lab_float(cv2.cvtColor(px, cv2.COLOR_BGR2LAB))[0, 0]


def jersey_features(image_bgr: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    boxes = np.asarray(boxes).reshape(-1, 4)
    out = np.full((len(boxes), FEATURE_DIM), np.nan, np.float32)
    if len(boxes) == 0:
        return out
    for i, b in enumerate(boxes):
        t = _region_median_lab(image_bgr, b, TORSO)
        s = _region_median_lab(image_bgr, b, SHORTS)
        out[i, :3] = bgr_to_lab_float(t[None, None, :])[0, 0] if np.isfinite(t).all() else np.nan
        out[i, 3:] = bgr_to_lab_float(s[None, None, :])[0, 0] if np.isfinite(s).all() else np.nan
    return out
