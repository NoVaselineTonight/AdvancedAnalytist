"""Frame-to-frame camera motion (pan / tilt / zoom) estimation.

For a camera that rotates and zooms about a fixed point (tripod pan, most
broadcast rigs) consecutive frames are related *exactly* by a homography for
the whole scene, so we can track background features (walls, floor lines)
while masking out people and on-screen graphics. A handheld camera that also
translates breaks this assumption; the RANSAC inlier count then drops and the
chain is flagged as broken rather than trusted.

Measured on the koristv test footage: after 20+ s of panning back and forth the
chained estimate is off by ~2 px (median, 640-px frame) when the camera returns
to an earlier view. Anchor-frame matching was tried and rejected: gym walls
(wall bars, corrugated panels) are so repetitive that direct matching
occasionally locks onto a copy of the pattern, which is far worse than the
chain's slow drift.
"""

from __future__ import annotations

import cv2
import numpy as np

from replay.config import MotionConfig


class MotionEstimator:
    def __init__(self, cfg: MotionConfig, frame_size: tuple[int, int], overlay_regions: list[list[float]]):
        self.cfg = cfg
        w, h = frame_size
        self.scale = min(1.0, cfg.work_width / float(w))
        self.size = (int(round(w * self.scale)), int(round(h * self.scale)))
        self.static_mask = np.full((self.size[1], self.size[0]), 255, np.uint8)
        for x0, y0, x1, y1 in overlay_regions:
            self.static_mask[int(y0 * self.size[1]):int(np.ceil(y1 * self.size[1])),
                             int(x0 * self.size[0]):int(np.ceil(x1 * self.size[0]))] = 0
        self.prev_gray: np.ndarray | None = None
        self.prev_mask: np.ndarray | None = None
        self.last_good: np.ndarray | None = None  # last successful M (Task 6 fallback)
        self.fail_count = 0
        self._was_break = True

    def _mask(self, boxes: np.ndarray) -> np.ndarray:
        m = self.static_mask.copy()
        for x0, y0, x1, y1 in np.asarray(boxes).reshape(-1, 4) * self.scale:
            pad = 0.1 * (x1 - x0)
            m[max(0, int(y0 - pad)):int(y1 + pad), max(0, int(x0 - pad)):int(x1 + pad)] = 0
        return m

    def update(self, image_bgr: np.ndarray, person_boxes: np.ndarray) -> tuple[np.ndarray | None, int]:
        """Returns (M, inliers): M maps previous-frame pixels to current-frame pixels (full resolution).

        M is None for the first frame and whenever the estimate is unreliable.
        """
        gray = cv2.cvtColor(cv2.resize(image_bgr, self.size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        mask = self._mask(person_boxes)
        M, inliers = None, 0
        self._was_break = self.prev_gray is None  # first frame / chain reset by the caller
        if self.prev_gray is not None:
            M, inliers = self._estimate(self.prev_gray, self.prev_mask, gray)
        self.prev_gray, self.prev_mask = gray, mask
        return M, inliers

    def fallback(self, M: np.ndarray | None) -> np.ndarray | None:
        """Approximate motion for the TRACKER when the estimate ``M`` of this sample failed: the last good M (the
        per-sample pan rate), re-used for at most ``cfg.fallback_samples`` consecutive failures, else None. Never
        across a chain reset. Call once per sample right after ``update``. Off when fallback_samples == 0."""
        n = int(getattr(self.cfg, "fallback_samples", 0) or 0)
        if M is not None:
            self.last_good, self.fail_count = M, 0
            return None
        if self._was_break:
            self.last_good, self.fail_count = None, 0
            return None
        self.fail_count += 1
        if n <= 0 or self.last_good is None or self.fail_count > n:
            return None
        return self.last_good

    def _estimate(self, prev: np.ndarray, prev_mask: np.ndarray, cur: np.ndarray) -> tuple[np.ndarray | None, int]:
        p0 = cv2.goodFeaturesToTrack(prev, self.cfg.max_corners, 0.01, 8, mask=prev_mask, blockSize=7)
        if p0 is None or len(p0) < self.cfg.min_inliers:
            return None, 0
        lk = dict(winSize=(21, 21), maxLevel=4, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, cur, p0, None, **lk)
        p0r, st_back, _ = cv2.calcOpticalFlowPyrLK(cur, prev, p1, None, **lk)
        good = (st.ravel() == 1) & (st_back.ravel() == 1) & (np.linalg.norm(p0 - p0r, axis=2).ravel() < 1.0)
        if good.sum() < self.cfg.min_inliers:
            return None, int(good.sum())
        M, inl = cv2.findHomography(p0[good], p1[good], cv2.RANSAC, self.cfg.ransac_px)
        n = int(inl.sum()) if inl is not None else 0
        if M is None or n < self.cfg.min_inliers:
            return None, n
        S = np.diag([self.scale, self.scale, 1.0])
        return np.linalg.inv(S) @ M @ S, n
