"""Second look at the rims (pass-2 step 7a): ball detection on upscaled crops around a rim after shot attempts.

Most missed made baskets (full game vs the official scoresheet: 13 of 15 misses in analysed spans) had NO ball
candidate near the rim at all in pass 1 - the ball is small there, partly hidden by the net / backboard, and blurred.
Pass 1 runs the detector once on the whole 720p frame; here the same detector looks again at a square crop around the
rim (the detector resizes its input, so a ~300 px crop is a 2x upscale) with a lower threshold, only where a shot
is plausible:

* windows: samples where a pass-1 ball candidate is in the air near a rim (within ``win_dx_r`` rim radii sideways,
  between ``win_dy_r`` above and a few radii below the ring), extended ``after_s`` forward (the make itself);
* each sample of a window: crop centred on the rim, side = ``crop_r`` rim radii (min ``min_crop_px``); ball boxes with
  conf >= ``threshold`` inside the crop become extra ball candidates of that sample (``rows`` like pass 1, used by
  the made-basket test only - the ball track is not changed).

Budget ``max_crops`` per game (windows nearest to a rim first). Cached in ``work/rimlook.npz`` keyed by the windows
and settings. Off when no video / no detector (then nothing changes).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

ENABLED = os.environ.get("REPLAY_RIMLOOK", "0") == "1"  # OFF by default (full game: +3 makes for +4 false, 211 s)
CACHE = "rimlook.npz"


@dataclass
class RimLookConfig:
    win_dx_r: float = 12.0
    win_dy_r: tuple[float, float] = (-30.0, 4.0)  # ball centre relative to the ring, rim radii (negative = above)
    min_conf: float = 0.2  # pass-1 candidates that open a window
    after_s: float = 1.5
    before_s: float = 0.2
    crop_r: float = 16.0  # crop side in rim radii
    min_crop_px: float = 160.0
    max_crop_px: float = 320.0
    up_shift_r: float = 3.0  # crop centre this many radii above the ring (ball comes from above)
    threshold: float = 0.15
    max_crops: int = 900
    stride: int = 2  # look at every k-th sample of a window (runtime; ~70 ms per crop on the GPU)


def attempt_windows(ball_frames: Sequence[Any], rims: dict, t: np.ndarray, rc: RimLookConfig
                    ) -> list[tuple[int, str]]:
    """(sample, side) after each detected shot attempt: the ball track is at a rim (ball.py "rim" anchors);
    one window per attempt (anchors closer than ``after_s`` merge), every ``stride``-th sample."""
    from replay import court
    n = len(t)
    bk = dict(zip("LR", court.baskets()))
    seeds: list[tuple[int, str]] = []
    for bf in ball_frames:
        if getattr(bf, "source", "") != "rim" or bf.x is None:
            continue
        side = min("LR", key=lambda s: abs(bk[s][0] - bf.x))
        if seeds and seeds[-1][1] == side and t[bf.i] - t[seeds[-1][0]] < rc.after_s:
            continue
        seeds.append((bf.i, side))
    want = []
    for i, side in seeds:
        rr = (rims or {}).get(side)
        if rr is None:
            continue
        j0 = int(np.searchsorted(t, t[i] - rc.before_s))
        j1 = int(np.searchsorted(t, t[i] + rc.after_s, "right"))
        want += [(j, side) for j in range(j0, min(j1, n), max(1, rc.stride)) if j < len(rr) and rr[j] is not None]
    return sorted(set(want))[: rc.max_crops]


def windows(ball_rows: Sequence[np.ndarray], rims: dict, t: np.ndarray, rc: RimLookConfig) -> list[tuple[int, str]]:
    """(sample index, side) pairs to look at."""
    n = len(t)
    want: dict[tuple[int, str], float] = {}
    for side, rr in (rims or {}).items():
        for i in range(n):
            r = rr[i] if i < len(rr) else None
            rows = ball_rows[i] if i < len(ball_rows) else None
            if r is None or rows is None or not len(rows):
                continue
            u, v, rad = r
            best = None
            for x1, y1, x2, y2, cf in np.asarray(rows, float)[:, :5]:
                if cf < rc.min_conf:
                    continue
                dx, dy = ((x1 + x2) / 2 - u) / rad, ((y1 + y2) / 2 - v) / rad
                if abs(dx) <= rc.win_dx_r and rc.win_dy_r[0] <= dy <= rc.win_dy_r[1]:
                    d = abs(dx) + 0.3 * abs(dy)
                    best = d if best is None else min(best, d)
            if best is None:
                continue
            j0 = int(np.searchsorted(t, t[i] - rc.before_s))
            j1 = int(np.searchsorted(t, t[i] + rc.after_s, "right"))
            for j in range(j0, j1, max(1, rc.stride)):
                if rr[j] is not None:
                    want[(j, side)] = min(want.get((j, side), 1e9), best)
    order = sorted(want, key=lambda k: want[k])[: rc.max_crops]
    return sorted(order)


def crop_region(r: tuple[float, float, float], size: tuple[int, int], rc: RimLookConfig) -> tuple[int, int, int, int]:
    u, v, rad = r
    side = float(np.clip(rc.crop_r * rad, rc.min_crop_px, rc.max_crop_px))
    cx, cy = u, v - rc.up_shift_r * rad
    W, H = size
    x0 = int(np.clip(cx - side / 2, 0, max(0, W - side)))
    y0 = int(np.clip(cy - side / 2, 0, max(0, H - side)))
    return x0, y0, int(min(W, x0 + side)), int(min(H, y0 + side))


def augment(work: Path, data: Any, ball_rows: list[np.ndarray], rims: dict, video: Any, cfg: Any,
            size: tuple[int, int], rc: RimLookConfig | None = None, detector: Any = None,
            ball_frames: Sequence[Any] | None = None) -> list[np.ndarray]:
    """Pass-2 hook: ball_rows with second-look rim candidates appended (unchanged when disabled / no video).
    With ``ball_frames`` the windows follow the detected shot attempts (``attempt_windows``)."""
    rc = rc or RimLookConfig()
    if not ENABLED:
        return ball_rows
    t = np.asarray(data.t, float)
    want = attempt_windows(ball_frames, rims, t, rc) if ball_frames is not None else windows(ball_rows, rims, t, rc)
    if not want:
        return ball_rows
    regions = {(i, s): crop_region(rims[s][i], size, rc) for i, s in want}
    det_cfg = getattr(cfg, "detector", None)
    key = hashlib.sha1(json.dumps([[int(i), s, list(regions[(i, s)])] for i, s in want] +
                                  [asdict(rc), getattr(det_cfg, "variant", "medium")]).encode()).hexdigest()
    path = Path(work) / CACHE
    found = None
    if path.exists():
        try:
            with np.load(path) as z:
                if str(z["key"]) == key:
                    found = z["rows"]
        except Exception:  # noqa: BLE001
            found = None
    if found is None:
        from replay.redetect import iter_frames
        from replay.team_appearance import resolve_video
        vid = resolve_video(data, video)
        if vid is None:
            return ball_rows
        t0 = time.perf_counter()
        if detector is None:
            from replay.config import DetectorConfig
            from replay.detection import build_detector
            dc = DetectorConfig(**{**(asdict(det_cfg) if det_cfg is not None else {}), "tiles": (1, 1)})
            detector = build_detector(dc)
        by_i: dict[int, list[str]] = {}
        for i, s in want:
            by_i.setdefault(i, []).append(s)
        out = []
        from replay.detection import BALL
        from replay.frame_tap import GPU_LOCK
        for i, img in iter_frames(vid, data, sorted(by_i)):
            for s in by_i[i]:
                x0, y0, x1, y1 = regions[(i, s)]
                crop = np.ascontiguousarray(img[y0:y1, x0:x1])
                if crop.size == 0:
                    continue
                old = getattr(detector, "threshold", None)
                if old is not None:
                    detector.threshold = rc.threshold
                try:
                    with GPU_LOCK:
                        d = detector.detect(crop)
                finally:
                    if old is not None:
                        detector.threshold = old
                m = (d.cls == BALL) & (d.conf >= rc.threshold)
                for b, c in zip(d.xyxy[m], d.conf[m]):
                    out.append([i, b[0] + x0, b[1] + y0, b[2] + x0, b[3] + y0, float(c)])
        found = np.array(out, float).reshape(-1, 6)
        print(f"rimlook: {len(want)} crops, {len(found)} ball candidates, {time.perf_counter() - t0:.1f} s")
        try:
            tmp = Path(work) / "rimlook.tmp.npz"
            np.savez_compressed(tmp, key=np.array(key), rows=found)
            os.replace(tmp, path)
        except OSError:
            pass
    else:
        print(f"rimlook: {len(found)} ball candidates (cached)")
    out_rows = [np.asarray(r, float).reshape(-1, 5) if r is not None else np.zeros((0, 5)) for r in ball_rows]
    for i, x1, y1, x2, y2, c in found:
        i = int(i)
        new = np.array([[x1, y1, x2, y2, c]])
        cur = out_rows[i]
        if len(cur):  # not a duplicate of a pass-1 candidate
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            ccx, ccy = (cur[:, 0] + cur[:, 2]) / 2, (cur[:, 1] + cur[:, 3]) / 2
            if np.any(np.hypot(ccx - cx, ccy - cy) < max(4.0, 0.5 * (x2 - x1))):
                continue
        out_rows[i] = np.vstack([cur, new])
    return out_rows
