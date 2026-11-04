"""Debug/QA video: projected court lines, tracked boxes and a mini-map on every processed frame.

Use it to check calibration propagation (the yellow court lines must stay glued to
the real floor lines while the camera pans) and tracking quality (IDs on boxes).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
from tqdm import tqdm

from replay import calibration as calib_mod
from replay import court, homography
from replay.config import Config
from replay.detection import PERSON
from replay.pipeline import load_pass1
from replay.video import VideoSampler

MAP_S = 9  # minimap px per metre


def _palette(i: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(i * 7919)
    return tuple(int(c) for c in rng.integers(60, 255, 3))


def render_check_video(out: Path, calib_path: Path, cfg: Config, dst: Path, max_seconds: float | None = None) -> Path:
    data = load_pass1(out / "work")
    calib = calib_mod.CalibrationFile.load(calib_path)
    res = calib_mod.resolve(data.t, data.motion, calib, cfg.calibration)
    sig = data.signature
    sampler = VideoSampler(sig["video"], sig["process_fps"], sig["start_s"], sig["end_s"])
    size = sampler.info.size
    # H.264 via the bundled ffmpeg so the file plays in any browser/player
    writer = imageio_ffmpeg.write_frames(str(dst), size, fps=sig["process_fps"], codec="libx264", quality=7,
                                         macro_block_size=1, pix_fmt_in="bgr24", output_params=["-pix_fmt", "yuv420p"])
    writer.send(None)
    rows = data.rows[data.rows[:, 6] == PERSON]
    by_k: dict[int, np.ndarray] = {}
    for k in np.unique(rows[:, 0]).astype(int):
        by_k[k] = rows[rows[:, 0] == k]
    idx_of_k = {int(k): i for i, k in enumerate(data.k)}
    lines = court.court_polylines()
    mw, mh = int((court.LENGTH + 2) * MAP_S), int((court.WIDTH + 2) * MAP_S)

    def mp(p: np.ndarray) -> tuple[int, int]:
        return int((p[0] + 1) * MAP_S), int((court.WIDTH + 1 - p[1]) * MAP_S)

    n = len(data.k) if max_seconds is None else min(len(data.k), int(max_seconds * sig["process_fps"]))
    for s in tqdm(sampler.iter_samples(int(data.k[0])), total=n, unit="frame", desc="check video"):
        i = idx_of_k.get(s.k)
        if i is None:  # not analysed (pass 1 skipped a break in play): skip, don't stop
            if not n or s.k > int(data.k[n - 1]):
                break
            continue
        if i >= n:
            break
        img = s.image
        H = res.H[i]
        if H is not None:
            for line in lines:
                pts = homography.court_to_image(H, line)
                for a, b in zip(pts[:-1], pts[1:]):
                    if np.isfinite(a).all() and np.isfinite(b).all() and np.abs(np.r_[a, b]).max() < 1e5:
                        cv2.line(img, tuple(np.int32(a)), tuple(np.int32(b)), (0, 255, 255), 1, cv2.LINE_AA)
        mini = np.full((mh, mw, 3), 40, np.uint8)
        for line in lines:
            cv2.polylines(mini, [np.array([mp(p) for p in line], np.int32)], False, (180, 180, 180), 1)
        for r in by_k.get(s.k, []):
            x1, y1, x2, y2 = r[1:5].astype(int)
            tid = int(r[7])
            col = _palette(tid) if tid >= 0 else (90, 90, 90)
            cv2.rectangle(img, (x1, y1), (x2, y2), col, 2 if tid >= 0 else 1)
            if tid >= 0:
                cv2.putText(img, str(tid), (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
            if H is not None and tid >= 0:
                p = homography.project(H, np.array([[(x1 + x2) / 2, y2]]))[0]
                if np.isfinite(p).all() and court.in_court(p, 3.0)[0]:
                    cv2.circle(mini, mp(p), 4, col, -1)
        mi = data.inliers[i]
        status = f"t={s.t:7.2f}s  calib={res.quality[i]}  motion_inliers={mi if mi >= 0 else 'fixed'}"
        cv2.rectangle(img, (0, 0), (len(status) * 9 + 10, 24), (0, 0, 0), -1)
        cv2.putText(img, status, (5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        h0 = size[1] - mh - 8
        img[h0:h0 + mh, 8:8 + mw] = cv2.addWeighted(img[h0:h0 + mh, 8:8 + mw], 0.2, mini, 0.8, 0)
        writer.send(np.ascontiguousarray(img))
    writer.close()
    return dst
