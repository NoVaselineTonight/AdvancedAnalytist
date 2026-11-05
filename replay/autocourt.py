"""Fully automatic court calibration - no clicks needed, for any gym and camera.

Why single-frame matching fails: other sports' lines, wall bars and a camera that sees little of the court
give many look-alikes, and many gyms have a non-standard court (this project's test gym is ~26.5 m long).

How it works
  1. Rotation + zoom per frame from vanishing points. A gym is a "Manhattan world": floor lines of every
     sport, wall bars, doors and posts run along three orthogonal directions, which gives tilt, roll, focal
     length and the pan angle modulo 90 degrees - independently in every frame (no drift).
  2. The 90-degree ambiguity is removed with the relative camera motion measured in pass 1, and the rule
     that a sideline camera looks across the court.
  3. Line pixels of every colour are found (thin strokes that differ from the floor around them) and
     labelled by colour class (black, white, hues), so the basketball lines can be told apart from
     other sports' lines in a multi-sport hall.
  4. Floor mosaic: with rotation and zoom known, each frame's line pixels are projected onto the floor
     around the camera (in camera heights); a cell counts as a line when most frames seeing it found one,
     so people and moving clutter drop out.
  5. Court fit: court size (length, width), marking standard (FIBA, NBA, NCAA, old FIBA), camera height and
     position are searched to explain as much reliable line evidence as possible. The court colour is the
     one whose lines explain the basketball-only geometry (3-point arcs, free-throw and centre circles)
     much better than chance.
  6. The camera position is re-checked per time window (moved tripod, carried camera, TV cuts).
  7. Joint alignment (bundle adjustment) per camera-motion segment: rotations follow the measured motion
     with a slowly varying correction, zoom varies smoothly, and all frames' line evidence is matched at
     once, so clutter in one frame cannot drag it away. The court width is settled last.
  8. resolve_auto(): per analysed frame, motion chain between anchors + interpolated corrections.

Stored in calibration.json ("auto" + "court"); calibration.resolve() uses it.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp

from replay import court
from replay.camera_model import PTZ, fit_ptz_to_homography, rotation

WORK_W = 640  # line evidence is kept at this width
TRUNC = 10.0  # px at WORK_W: distances beyond count the same
MAX_LINE_PX = 7000  # line pixels kept per anchor (subsampled)
AUTO_VERSION = 1


# ------------------------------------------------------------------------------------------- basics
def K_of(f: float, size: tuple[int, int]) -> np.ndarray:
    w, h = size
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])


def court_to_image_R(C: np.ndarray, R: np.ndarray, f: float, size: tuple[int, int]) -> np.ndarray:
    """court(x, y, 1) -> pixels for world->camera rotation R."""
    return K_of(f, size) @ np.column_stack([R[:, 0], R[:, 1], -R @ C])


def ptz_from_R(R: np.ndarray, f: float) -> PTZ:
    """Inverse of camera_model.rotation (rows = right, down, forward)."""
    fwd = R[2]
    tilt = float(np.arcsin(np.clip(-fwd[2], -1, 1)))
    pan = float(np.arctan2(fwd[1], fwd[0]))
    R0 = rotation(pan, tilt, 0.0)
    # roll: rotation about forward between R0 and R
    right, down = R0[0], R0[1]
    roll = float(np.arctan2(np.dot(R[0], down), np.dot(R[0], right)))
    return PTZ(pan, tilt, roll, float(f))


def model_points_dirs(step: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    """Points along every line of the current court (court.LENGTH/WIDTH) and their directions."""
    pts, dirs = [], []
    for line in court.court_polylines(n_arc=96):
        for a, b in zip(line[:-1], line[1:]):
            seg = b - a
            L = float(np.linalg.norm(seg))
            if L < 1e-6:
                continue
            n = max(1, int(np.ceil(L / step)))
            u = np.linspace(0, 1, n, endpoint=False)[:, None]
            pts.append(a + seg * u)
            dirs.append(np.repeat((seg / L)[None], n, axis=0))
    return np.vstack(pts), np.vstack(dirs)


# ------------------------------------------------------------------------------------ line evidence
N_CLASSES = 8  # 0 dark neutral (black), 1 bright neutral (white), 2..7 colour by hue
WEAK = 8  # class + WEAK = the same colour, low contrast


def with_weak(classes) -> tuple[int, ...]:
    return tuple(classes) + tuple(int(c) + WEAK for c in classes)


LINE_MIN_EXTENT = 12  # px at 1280 wide: shorter isolated specks are not line pixels (grainy floors)
GRAINY_LEVEL = 6.0  # median |line contrast| of the floor (lower image half) above which it is 'grainy'
NOISE_K = 2.0  # line contrast must exceed NOISE_K x the local median contrast of the floor texture


def _local_level(a: np.ndarray, step: int = 4, k: int = 11) -> np.ndarray:
    """Local median of a non-negative contrast map (robust texture/noise level), full size, float32."""
    h, w = a.shape
    small = np.clip(a[::step, ::step], 0, 255).astype(np.uint8)
    med = cv2.medianBlur(small, k)
    return cv2.resize(med, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)


def line_pixels(img: np.ndarray, person_boxes: np.ndarray | None = None,
                static_mask: np.ndarray | None = None) -> np.ndarray:
    """Thin painted lines of ANY colour: (N, 3) int16 [x, y, colour class] at WORK_W scale.

    A line pixel differs from the floor around it (median background) and belongs to a thin structure
    (removed by a morphological opening). Computed at full resolution - far lines are only 1-2 px wide.
    The colour class lets the calibration find which lines are the basketball court (black lines on a
    multi-sport floor, white lines on wood, coloured lines, ...).
    """
    h, w = img.shape[:2]
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    k = max(9, int(round(17 * w / 1280)) | 1)
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    L, A, B = cv2.split(lab)
    dark = cv2.morphologyEx(L, cv2.MORPH_BLACKHAT, se).astype(np.int16)
    bright = cv2.morphologyEx(L, cv2.MORPH_TOPHAT, se).astype(np.int16)
    da = np.maximum(cv2.morphologyEx(A, cv2.MORPH_BLACKHAT, se), cv2.morphologyEx(A, cv2.MORPH_TOPHAT, se))
    db = np.maximum(cv2.morphologyEx(B, cv2.MORPH_BLACKHAT, se), cv2.morphologyEx(B, cv2.MORPH_TOPHAT, se))
    dL = np.where(dark >= bright, -dark, bright)
    dc = np.hypot(da.astype(np.float32), db.astype(np.float32))
    ko = max(7, int(round(13 * w / 1280)) | 1)
    open_se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ko, ko))
    absL = np.abs(dL)
    grainy = float(np.median(absL[h // 2::4, ::4])) > GRAINY_LEVEL
    if not grainy:
        # smooth floor (most gyms): fixed thresholds, so faint far lines count
        tL = tC = None
        cand = ((absL > 11) | (dc > 11)).astype(np.uint8)  # faint far lines too (marked 'weak' below)
        thick = cv2.morphologyEx(cand, cv2.MORPH_OPEN, open_se)
        thin = cand & (1 - thick)
    else:
        # grainy / noisy floor (wood grain, heavy compression): with fixed thresholds the whole floor would
        # count as 'lines'. Thresholds follow the local texture level (median |contrast| over ~45 px), blobs
        # are found on a median-filtered image, and isolated specks are dropped.
        tL = np.maximum(11.0, NOISE_K * _local_level(absL))
        tC = np.maximum(11.0, NOISE_K * _local_level(dc))
        cand = ((absL > tL) | (dc > tC)).astype(np.uint8)
        Lm = cv2.medianBlur(L, 5)
        blob = ((cv2.morphologyEx(Lm, cv2.MORPH_BLACKHAT, se) > 11) | (cv2.morphologyEx(Lm, cv2.MORPH_TOPHAT, se) > 11)
                | (dc > 11)).astype(np.uint8) | cand
        thick = cv2.morphologyEx(blob, cv2.MORPH_OPEN, open_se)
        thin = cand & (1 - thick)
        n_cc, lab_cc, st, _c = cv2.connectedComponentsWithStats(thin, connectivity=8)
        ext_ = np.maximum(st[:, cv2.CC_STAT_WIDTH], st[:, cv2.CC_STAT_HEIGHT])
        ok_cc = ext_ >= max(4, int(round(LINE_MIN_EXTENT * w / 1280)))
        ok_cc[0] = False
        thin = ok_cc[lab_cc].astype(np.uint8)
    if person_boxes is not None:
        for x1, y1, x2, y2 in np.asarray(person_boxes).reshape(-1, 4):
            pad = 0.1 * (x2 - x1)
            thin[max(0, int(y1 - pad)):int(y2 + pad), max(0, int(x1 - pad)):int(x2 + pad)] = 0
    ys, xs = np.nonzero(thin)
    if not len(xs):
        return np.zeros((0, 3), np.int16)
    px = lab[ys, xs].astype(np.int16)
    a_, b_ = px[:, 1] - 128, px[:, 2] - 128
    chroma = np.hypot(a_, b_)
    # floor colour around the line (lines are thin, so a wide mean is ~ the floor): a white line on a wooden
    # floor blends into orange at its edges - it is still "less colourful than the floor", i.e. neutral
    bga = cv2.blur(A, (31, 31))[ys, xs].astype(np.float32) - 128
    bgb = cv2.blur(B, (31, 31))[ys, xs].astype(np.float32) - 128
    bg_chroma = np.hypot(bga, bgb)
    neutral = chroma < np.maximum(14.0, 0.7 * bg_chroma)
    hue_bin = (((np.arctan2(b_, a_) + np.pi) / (2 * np.pi) * 6).astype(int) % 6) + 2
    cls = np.where(neutral, np.where(dL[ys, xs] < 0, 0, 1), hue_bin)
    # weak = low contrast (faint far lines, but also noise): class + WEAK. The court search uses strong
    # pixels only; frame alignment, which starts from a nearly right pose, also uses the weak ones.
    if tL is None:
        strong = (np.abs(dL[ys, xs]) > 22) | (dc[ys, xs] > 16)
    else:
        strong = (np.abs(dL[ys, xs]) > np.maximum(22.0, 1.6 * tL[ys, xs])) | (dc[ys, xs] > np.maximum(16.0, 1.6 * tC[ys, xs]))
    cls = np.where(strong, cls, cls + WEAK)
    s = WORK_W / float(w)
    # unique (x, y, class) at WORK_W scale, as one integer key (much faster than unique rows)
    key = np.unique(((xs * s).astype(np.int64) * 4096 + (ys * s).astype(np.int64)) * 32 + cls)
    out = np.column_stack([key // (32 * 4096), (key // 32) % 4096, key % 32])
    return _mask_lines(out, static_mask).astype(np.int16)


def _mask_lines(lines: np.ndarray, static_mask: np.ndarray | None) -> np.ndarray:
    if static_mask is None or not len(lines):
        return lines
    return lines[~static_mask[np.clip(lines[:, 1], 0, static_mask.shape[0] - 1), lines[:, 0]]]


def static_overlay_mask(frames: list[np.ndarray]) -> np.ndarray | None:
    """Pixels that never change while the camera pans = broadcast overlays (score bar, logo)."""
    if len(frames) < 8:
        return None
    hh = int(round(frames[0].shape[0] * WORK_W / frames[0].shape[1]))
    g = np.stack([cv2.cvtColor(cv2.resize(f, (WORK_W, hh), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
                  for f in frames]).astype(np.float32)
    std = g.std(axis=0)
    med = np.median(g, axis=0)
    grad = np.hypot(cv2.Sobel(med, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(med, cv2.CV_32F, 0, 1, ksize=3))
    # an overlay never changes AND has sharp edges (text, boxes); a plain floor also looks the same in
    # every frame, but has no edges - it must not be masked (the court lines on it move with the camera)
    stat = ((std < 7.0) & (grad > 40)).astype(np.uint8)
    stat = cv2.morphologyEx(stat, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    stat = cv2.dilate(stat, np.ones((5, 5), np.uint8))
    return stat.astype(bool)


# -------------------------------------------------------------------------------- vanishing points
def line_segments(img: np.ndarray, static_mask: np.ndarray | None = None, min_len: float = 25.0):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    det = lsd.detect(g)[0]
    if det is None:
        return np.zeros((0, 4)), np.zeros(0)
    Ls = det.reshape(-1, 4).astype(np.float64)
    ln = np.hypot(Ls[:, 2] - Ls[:, 0], Ls[:, 3] - Ls[:, 1])
    keep = ln >= min_len
    if static_mask is not None:
        h, w = g.shape
        s = static_mask.shape[1] / w
        mx = np.clip(((Ls[:, 0] + Ls[:, 2]) / 2 * s).astype(int), 0, static_mask.shape[1] - 1)
        my = np.clip(((Ls[:, 1] + Ls[:, 3]) / 2 * s).astype(int), 0, static_mask.shape[0] - 1)
        keep &= ~static_mask[my, mx]
    return Ls[keep], ln[keep]


def _vp_resid(v, Ls, ln, size, trunc=0.05):
    R = rotation(v[0], v[1], v[2])
    VP = K_of(float(np.exp(v[3])), size) @ R  # columns: vanishing points of the world axes (K R e_k)
    a, b = Ls[:, :2], Ls[:, 2:]
    m = (a + b) / 2
    d = (b - a) / ln[:, None]
    best = np.full(len(Ls), 1.0)
    for k in range(3):
        vp = VP[:, k]
        dir_ = vp[:2] - m * vp[2]
        n = np.linalg.norm(dir_, axis=1) + 1e-9
        best = np.minimum(best, np.abs(d[:, 0] * dir_[:, 1] - d[:, 1] * dir_[:, 0]) / n)
    return np.minimum(best, trunc) * np.sqrt(ln / ln.mean())


def _rotations(pan, tilt, roll=0.0):
    """Vectorised camera_model.rotation: arrays of angles -> (..., 3, 3)."""
    pan, tilt = np.broadcast_arrays(np.asarray(pan, float), np.asarray(tilt, float))
    roll = np.broadcast_to(np.asarray(roll, float), pan.shape)
    fwd = np.stack([np.cos(tilt) * np.cos(pan), np.cos(tilt) * np.sin(pan), -np.sin(tilt)], -1)
    right = np.stack([np.sin(pan), -np.cos(pan), np.zeros_like(pan)], -1)
    down = np.cross(fwd, right)
    cr, sr = np.cos(roll)[..., None], np.sin(roll)[..., None]
    return np.stack([cr * right + sr * down, -sr * right + cr * down, fwd], -2)


def _vp_grid(Ls, ln, size):
    """Coarse exhaustive search (roll 0), all candidates at once."""
    w, h = size
    pans = np.radians(np.arange(0, 90, 3.0))
    tilts = np.radians(np.arange(-6, 36, 3.0))
    fs = np.exp(np.linspace(np.log(0.35 * w), np.log(3.0 * w), 9))
    sel = np.argsort(-ln)[:300]
    Ls, ln = Ls[sel], ln[sel]
    P, T = np.meshgrid(pans, tilts, indexing="ij")
    R = _rotations(P.ravel(), T.ravel())  # (M, 3, 3)
    m = (Ls[:, :2] + Ls[:, 2:]) / 2
    d = (Ls[:, 2:] - Ls[:, :2]) / ln[:, None]
    wts = ln / ln.mean()
    best = (np.inf, None)
    for f in fs:
        K = K_of(f, size)
        VP = K[None] @ R  # (M, 3, 3) columns = vanishing points
        res = np.full((len(R), len(Ls)), 0.05)
        for k in range(3):
            vp = VP[:, :, k]  # (M, 3)
            dx = vp[:, None, 0] - m[None, :, 0] * vp[:, None, 2]
            dy = vp[:, None, 1] - m[None, :, 1] * vp[:, None, 2]
            nrm = np.hypot(dx, dy) + 1e-9
            sn = np.abs(d[None, :, 0] * dy - d[None, :, 1] * dx) / nrm
            res = np.minimum(res, sn)
        cost = (res ** 2 * wts[None]).sum(1)
        j = int(np.argmin(cost))
        if cost[j] < best[0]:
            best = (cost[j], np.array([P.ravel()[j], T.ravel()[j], 0.0, np.log(f)]))
    return best[1]


def fit_rotation(Ls, ln, size, init: np.ndarray | None = None,
                 grid_fallback: bool = True) -> tuple[np.ndarray | None, float]:
    """(pan mod 90deg, tilt, roll, log f) and the inlier fraction of line length."""
    if len(Ls) < 15:
        return None, 0.0
    lo = [-10, -0.6, -0.3, np.log(0.2 * size[0])]
    hi = [10, 1.4, 0.3, np.log(8.0 * size[0])]
    lo_a, hi_a = np.array(lo) + 1e-6, np.array(hi) - 1e-6

    def run(x0):
        try:
            return least_squares(_vp_resid, np.clip(x0, lo_a, hi_a), args=(Ls, ln, size), bounds=(lo, hi),
                                 max_nfev=150)
        except ValueError:
            return None

    best = run(init) if init is not None else None
    if best is None or (grid_fallback and _inliers(best.x, Ls, ln, size) < 0.55):
        r = run(_vp_grid(Ls, ln, size))
        if r is not None and (best is None or r.cost < best.cost):
            best = r
    if best is None:
        return None, 0.0
    x = best.x.copy()
    x[0] = x[0] % (np.pi / 2)
    return x, _inliers(x, Ls, ln, size)


def _inliers(x, Ls, ln, size) -> float:
    r = _vp_resid(x, Ls, ln, size, trunc=1.0) / np.sqrt(ln / ln.mean())
    return float(np.sum(ln * (r < 0.012)) / max(1e-9, np.sum(ln)))


# ------------------------------------------------------------------------------- per-anchor features
@dataclass
class Anchor:
    i: int  # pass-1 sample index
    t: float
    frame: int
    vp: np.ndarray | None  # pan mod 90, tilt, roll, log f
    vp_inliers: float
    lines: np.ndarray  # (N, 3) int16 line pixels at WORK_W scale: x, y, colour class
    work_size: tuple[int, int]
    seg: int = -1
    relpan: float = 0.0  # degrees, from the motion chain
    pan: float | None = None  # absolute pan (radians) once disambiguated
    R: np.ndarray | None = None  # refined world->camera rotation
    f: float | None = None
    cost: float = np.inf  # refined line-matching cost (px at WORK_W)
    inliers: float = 0.0
    good: bool = False


def _anchor_features(args):
    i, t, frame, img, boxes, static_mask, init = args
    Ls, ln = line_segments(img, static_mask)
    vp, inl = fit_rotation(Ls, ln, (img.shape[1], img.shape[0]), init)
    lines = line_pixels(img, boxes, static_mask)
    keep = []  # cap per colour class, so a dominant texture (wood grain) cannot crowd out the court lines
    rng = np.random.default_rng(i)
    for c in np.unique(lines[:, 2]) if len(lines) else []:
        sel = np.where(lines[:, 2] == c)[0]
        if len(sel) > MAX_LINE_PX:
            sel = rng.choice(sel, MAX_LINE_PX, replace=False)
        keep.append(sel)
    if keep:
        lines = lines[np.concatenate(keep)]
    hh = int(round(img.shape[0] * WORK_W / img.shape[1]))
    return Anchor(i, t, frame, vp, inl, lines, (WORK_W, hh))


# ------------------------------------------------------------- features measured during pass 1 (frame tap)
FEATURES_FILE = "autocal_features.pkl"  # in the pass-1 work folder, written by replay.autocal_tap.AutocalTap
FEATURES_VERSION = 1
FEATURES_DIR = "autocal_features"  # incremental shards (frame tap); FEATURES_FILE = single-file form


def _cap_lines(lines: np.ndarray, i: int, cap: int = MAX_LINE_PX) -> np.ndarray:
    """At most `cap` pixels per colour class (same rule as _anchor_features)."""
    keep = []
    rng = np.random.default_rng(i)
    for c in np.unique(lines[:, 2]) if len(lines) else []:
        sel = np.where(lines[:, 2] == c)[0]
        if len(sel) > cap:
            sel = rng.choice(sel, cap, replace=False)
        keep.append(sel)
    return lines[np.sort(np.concatenate(keep))] if keep else lines


def anchor_raw(args):
    """Measurements of one anchor frame BEFORE the broadcast-overlay mask is known (pass-1 frame tap):
    line segments, their vanishing-point rotation, line pixels. The mask is applied in _anchor_finish."""
    i, t, frame, img, boxes = args
    Ls, ln = line_segments(img, None)
    size = (img.shape[1], img.shape[0])
    vp, inl = fit_rotation(Ls, ln, size, None)
    lines = _cap_lines(line_pixels(img, boxes, None), i, 2 * MAX_LINE_PX)
    hh = int(round(img.shape[0] * WORK_W / img.shape[1]))
    return (int(i), float(t), int(frame), Ls.astype(np.float32), ln.astype(np.float32), vp, float(inl),
            lines.astype(np.int16), size, (WORK_W, hh))


def _anchor_finish(args):
    """Overlay mask applied to anchor_raw output: segments on overlays dropped (rotation refitted from the first
    fit when any were), line pixels on overlays dropped, then the usual per-class cap."""
    (i, t, frame, Ls, ln, vp, inl, lines, size, work_size), static_mask = args[:2]
    known = args[2] if len(args) > 2 else None  # (vp, inl) already refitted with this same mask (streaming)
    Ls, ln = Ls.astype(np.float64), ln.astype(np.float64)
    if known is not None:
        vp, inl = known
    elif static_mask is not None and len(Ls):
        s = static_mask.shape[1] / size[0]
        mx = np.clip(((Ls[:, 0] + Ls[:, 2]) / 2 * s).astype(int), 0, static_mask.shape[1] - 1)
        my = np.clip(((Ls[:, 1] + Ls[:, 3]) / 2 * s).astype(int), 0, static_mask.shape[0] - 1)
        keep = ~static_mask[my, mx]
        if not keep.all():
            vp, inl = fit_rotation(Ls[keep], ln[keep], size, vp, grid_fallback=vp is None)
    if static_mask is not None and len(lines):
        lines = lines[~static_mask[np.clip(lines[:, 1], 0, static_mask.shape[0] - 1),
                                   np.clip(lines[:, 0], 0, static_mask.shape[1] - 1)]]
    return Anchor(i, t, frame, vp, inl, _cap_lines(lines, i), work_size)


def load_features(work: Path | None, data) -> dict | None:
    """Features cached by the pass-1 frame tap, if they belong to this pass-1 data (else None)."""
    if work is None:
        return None
    import pickle

    f = Path(work) / FEATURES_FILE
    shards = sorted((Path(work) / FEATURES_DIR).glob("shard_*.pkl"))
    try:
        if shards:  # written incrementally by the frame tap: shard_NNNN.pkl (new anchors) + small.pkl
            raws = []
            for sh in shards:
                raws += pickle.loads(sh.read_bytes())
            sm = Path(work) / FEATURES_DIR / "small.pkl"
            d = {"version": FEATURES_VERSION, "raws": raws, "small": pickle.loads(sm.read_bytes()) if sm.exists() else []}
        elif f.exists():
            d = pickle.loads(f.read_bytes())
        else:
            return None
    except Exception:  # noqa: BLE001 - a broken cache only costs the slow path
        return None
    raws = d.get("raws") or []
    if d.get("version") != FEATURES_VERSION or len(raws) < 3:
        return None
    n = len(data.t)
    raws = [r for r in raws if 0 <= r[0] < n]  # pass 1 may still be running (streaming): its data ends earlier
    for r in raws:  # every anchor must be the sample it claims to be
        if abs(float(data.t[r[0]]) - r[1]) > 1e-3 or int(data.frame[r[0]]) != r[2]:
            return None
    if len(raws) < 3:
        return None
    return dict(d, raws=raws)


STATE_FILE = "autocal_state.pkl"  # early court lock + streamed alignment (written during pass 1, see stream_job)


def load_state(work: Path | None) -> dict | None:
    if work is None or not (Path(work) / STATE_FILE).exists():
        return None
    try:
        import pickle

        d = pickle.loads((Path(work) / STATE_FILE).read_bytes())
        return d if d.get("version") == FEATURES_VERSION else None
    except Exception:  # noqa: BLE001
        return None


def stream_job(video: str, work: str, size: tuple[int, int], n_workers: int = 3) -> dict:
    """One round of calibration WHILE pass 1 runs (called by AutocalTap in its own process): calibrate on the
    pass-1 data written so far, reusing the previous round's court (early court lock) and alignments, and save
    the new state. Returns a short summary."""
    import pickle

    from replay.pipeline import load_pass1

    w = Path(work)
    data = load_pass1(w)
    feats = load_features(w, data)
    if feats is None or len(feats["raws"]) < STREAM_MIN_LOCK:  # not enough of the video analysed yet
        return {"ok": False, "n": len(data.t), "anchors": 0 if feats is None else len(feats["raws"])}
    dbg: dict = {}
    res = calibrate(Path(video), data, tuple(size), work=w, debug=dbg, n_workers=n_workers)
    st = dbg.get("state")
    n_anchor = len(dbg.get("anchors") or [])
    if res is None or st is None or n_anchor < STREAM_MIN_LOCK or res.good_frac < 0.4:
        # too few frames to lock the court on, or a court that does not fit: the next round searches again
        return {"ok": False, "n": len(data.t), "anchors": n_anchor}
    tmp = w / (STATE_FILE + ".tmp")
    tmp.write_bytes(pickle.dumps(st, protocol=pickle.HIGHEST_PROTOCOL))
    tmp.replace(w / STATE_FILE)
    return {"ok": True, "n": len(data.t), "msg": res.message, "reused": dbg.get("reused", 0)}


def read_frames(video: Path, frames: list[int], cancel=None):
    """Sequential decode (reliable on every container); yields (frame_no, image)."""
    cap = cv2.VideoCapture(str(video))
    try:
        want = sorted(set(int(f) for f in frames))
        pos = 0
        for fno in want:
            if cancel is not None and cancel.is_set():
                return
            if fno - pos > 300:  # long jump: seek (cheap on mp4), then verify
                cap.set(cv2.CAP_PROP_POS_FRAMES, fno)
                pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            while pos < fno:
                if not cap.grab():
                    return
                pos += 1
            ok, img = cap.read()
            pos += 1
            if not ok:
                return
            yield fno, img
    finally:
        cap.release()


# --------------------------------------------------------------------------------- pan disambiguation
def chain(motions: list[np.ndarray | None], n: int):
    from replay.calibration import _segment_transforms

    return _segment_transforms(motions, n)


def relative_pan_deg(G: np.ndarray, f: float, size: tuple[int, int]) -> float:
    """Pan of a frame relative to its segment reference (degrees, positive = turned right).

    Taken from the relative rotation (not from where the image centre lands), so it stays correct beyond
    90 degrees, where a homography no longer tells in front from behind."""
    A = chain_rotation(G, f, f, size)
    v = A[2, :]  # this frame's optical axis in reference-camera coordinates
    return float(np.degrees(np.arctan2(v[0], v[2])))


def disambiguate(anchors: list[Anchor]) -> dict[int, list[float]]:
    """Unwrap each motion segment's pan; returns per segment the plausible offsets (degrees) to try.

    pan_abs = S(t) - relpan (camera_model: turning right decreases pan). The vanishing points give
    pan mod 90, so S is known mod 90; S drifts slowly (motion-chain drift), so it is estimated locally
    and unwrapped in time. Of the 4 possible offsets, keep those that make the camera look at the court
    side (+y) for nearly all frames.
    """
    hyps: dict[int, list[float]] = {}
    for s in sorted({a.seg for a in anchors}):
        mem = [a for a in anchors if a.seg == s and a.vp is not None and a.vp_inliers > 0.25]
        if not mem:
            continue
        tt = np.array([a.t for a in mem])
        u = np.radians((np.degrees(np.array([a.vp[0] for a in mem])) + np.array([a.relpan for a in mem])) * 4.0)
        Sm = np.zeros(len(mem))
        for j in range(len(mem)):
            w = np.abs(tt - tt[j]) <= 45.0
            Sm[j] = np.degrees(np.angle(np.mean(np.exp(1j * u[w])))) / 4.0  # circular mean mod 90
        S = np.degrees(np.unwrap(np.radians(Sm * 4.0))) / 4.0
        Sfull = np.interp([a.t for a in anchors if a.seg == s], tt, S)
        for a, Sa in zip([a for a in anchors if a.seg == s], Sfull):
            a._S = Sa  # type: ignore[attr-defined]
        opts = []
        for off in range(-4, 8):
            pans = S + 90.0 * off - np.array([a.relpan for a in mem])
            frac = float(np.mean((pans > 3) & (pans < 177)))
            if frac > 0.85:
                opts.append((frac, 90.0 * off))
        if not opts:  # camera turned outside (0, 180) at times: take the most plausible offsets anyway
            for off in range(-4, 8):
                pans = S + 90.0 * off - np.array([a.relpan for a in mem])
                opts.append((float(np.mean((pans > -20) & (pans < 200))), 90.0 * off))
        opts.sort(key=lambda o: -o[0])
        hyps[s] = [o[1] for o in opts[:2]]
    return hyps


def set_pan(a: Anchor, offset_deg: float) -> None:
    if a.vp is None:
        a.pan = None
        return
    pred = getattr(a, "_S", np.degrees(a.vp[0]) + a.relpan) + offset_deg - a.relpan
    pm = np.degrees(a.vp[0]) % 90.0
    k = np.round((pred - pm) / 90.0)
    a.pan = float(np.radians(pm + 90.0 * k))


# ------------------------------------------------------------------------------------------ mosaic
@dataclass
class Mosaic:
    lines: np.ndarray  # float32 accumulated line evidence
    cover: np.ndarray  # float32 how often each cell was in view
    res: float  # mosaic cell size in camera heights
    ext: float  # half extent in camera heights (camera at the centre)


def build_mosaic(anchors: list[Anchor], size: tuple[int, int], res: float = 0.02, ext: float = 16.0,
                 classes: tuple[int, ...] | None = None) -> Mosaic:
    """Floor mosaic in camera-height units. ``lines`` = fraction of the frames seeing a cell that found a
    line there: painted lines are there every time, while streaks left by people near the camera or by
    vertical objects move around and stay rare."""
    N = int(round(2 * ext / res))
    acc = np.zeros(N * N, np.float32)
    cov = np.zeros((N, N), np.float32)
    w, h = size
    yy, xx = np.mgrid[0:h:6, 0:w:6]
    grid = np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float64)
    offs = (np.arange(-1, 2)[:, None] * N + np.arange(-1, 2)[None, :]).ravel()  # 3x3 neighbourhood

    def to_cells(pts, R, Kinv):
        d = (R.T @ Kinv @ np.column_stack([pts, np.ones(len(pts))]).T).T
        ok = d[:, 2] < -0.03
        g = -d[ok, :2] / d[ok, 2:3]
        g = g[np.linalg.norm(g, axis=1) < ext * 0.98]
        return ((g + ext) / res).astype(int)

    k9 = np.ones((9, 9), np.uint8)
    for a in anchors:
        if a.pan is None or a.vp is None:
            continue
        R = rotation(a.pan, a.vp[1], a.vp[2])
        Kinv = np.linalg.inv(K_of(float(np.exp(a.vp[3])), size))
        s_ = a.work_size[0] / size[0]
        lp = a.lines if classes is None else a.lines[np.isin(a.lines[:, 2], classes)]
        if len(lp):
            gi = to_cells(lp[:, :2].astype(np.float64) / s_, R, Kinv)
            if len(gi):  # each frame counts once per cell (3x3 dilated); cells are >= 1 away from the border
                lin = np.unique(gi[:, 1] * N + gi[:, 0])
                acc[np.unique((lin[:, None] + offs[None]).ravel())] += 1
        # the frame's footprint on the floor: cells near the projected sample grid (closing fills the gaps
        # between grid points; far away, where they are sparse, so are the line pixels). On the bounding box.
        gi = to_cells(grid, R, Kinv)
        if len(gi):
            x0, y0 = np.maximum(0, gi.min(axis=0) - 6)
            x1, y1 = np.minimum(N, gi.max(axis=0) + 7)
            one = np.zeros((y1 - y0, x1 - x0), np.uint8)
            one[gi[:, 1] - y0, gi[:, 0] - x0] = 1
            one = cv2.morphologyEx(one, cv2.MORPH_CLOSE, k9, borderType=cv2.BORDER_CONSTANT, borderValue=0)
            cov[y0:y1, x0:x1] += one
    frac = acc.reshape(N, N) / np.maximum(cov, 3.0)
    return Mosaic(frac.astype(np.float32), cov, res, ext)


@dataclass
class CourtFit:
    h: float  # camera height, metres
    C: np.ndarray  # camera position (x, y, z=h)
    length: float
    width: float
    score: float  # line evidence explained by the court (weighted cells)
    spec: str = "FIBA"


@dataclass
class _Prepared:
    weight: np.ndarray  # reliable thin-line evidence per cell (0 = none)
    total: float
    n_cov: int  # cells seen often enough
    res: float  # cell size, camera heights
    ext: float


def _prepare(mos: Mosaic, factor: int) -> _Prepared | None:
    """Thin, consistently seen lines of the mosaic, weighted by how consistently they were seen."""
    res = mos.res * factor
    a = cv2.resize(mos.lines, None, fx=1 / factor, fy=1 / factor, interpolation=cv2.INTER_AREA)
    c = cv2.resize(mos.cover, None, fx=1 / factor, fy=1 / factor, interpolation=cv2.INTER_AREA)
    ridge = np.maximum(0.0, a - cv2.blur(a, (7, 7)))  # thin lines only: blotches/texture cancel out
    lines = (ridge > 0.12) & (a > 0.2) & (c >= 4)
    wgt = np.where(lines, a, 0.0).astype(np.float32)
    tot = float(wgt.sum())
    if np.count_nonzero(lines) < 50 or tot <= 0:
        return None
    return _Prepared(wgt, tot, int(np.count_nonzero(c >= 4)), res, mos.ext)


_POLY_CACHE: dict = {}


def _polylines(n_arc: int = 64) -> list[tuple[np.ndarray, bool]]:
    """court.court_polylines() of the current court (cached) with an is-curve flag (arcs, circles)."""
    key = (court.LENGTH, court.WIDTH, court.SPEC, n_arc)
    v = _POLY_CACHE.get(key)
    if v is None:
        if len(_POLY_CACHE) > 5000:
            _POLY_CACHE.clear()
        v = [(l, len(l) > 6) for l in court.court_polylines(n_arc=n_arc)]
        _POLY_CACHE[key] = v
    return v


def _court_mask(h: float, res: float, tol: int):
    """Court lines drawn (2*tol+1 cells thick) at camera height h; returns (mask, pad)."""
    s = 1.0 / (h * res)
    pad = tol + 2
    shape = (int(court.WIDTH * s) + 2 * pad + 1, int(court.LENGTH * s) + 2 * pad + 1)
    k = np.zeros(shape, np.uint8)
    polys = _polylines(64)
    for curves in (False, True):  # arcs/circles drawn last with weight 2 (see explained())
        for l, c in polys:
            if c == curves:
                cv2.polylines(k, [np.int32(np.round(l * s + pad)).reshape(-1, 1, 2)], False, 2 if curves else 1,
                              2 * tol + 1)
    return k.astype(np.float32), pad


def explained(prep: _Prepared, h: float, cx: float, cy: float, tol: int = 2, curves_only: bool = False,
              enrichment: bool = False) -> float:
    """Line evidence lying on the current court (court.LENGTH/WIDTH/SPEC) seen from camera (cx, cy, h).

    Measured on the data side: every reliable line cell counts once if a court line passes within `tol`
    cells. A court of the wrong size can match some lines, but it cannot explain both 3-point arcs, the
    keys and the centre circle at once; a tiny court explains little; extra lines of other sports are
    simply left unexplained by every candidate."""
    N = prep.weight.shape[0]
    polys = []
    for l, c in _polylines(64):
        if curves_only and not c:
            continue
        p = ((l - np.array([cx, cy])) / h + prep.ext) / prep.res
        if np.abs(p).max() > 4 * N:
            continue
        polys.append((np.round(p), c))
    if not polys:
        return 0.0
    allp = np.vstack([p for p, _c in polys])
    x0, y0 = np.maximum(0, np.floor(allp.min(axis=0)).astype(int) - tol - 1)
    x1, y1 = np.minimum(N, np.ceil(allp.max(axis=0)).astype(int) + tol + 2)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    # drawn on the court's bounding box only (same result as on the whole mosaic, much faster)
    mk = np.zeros((y1 - y0, x1 - x0), np.uint8)
    off = np.array([x0, y0])
    # straight lines weigh 1, arcs and circles 2: other sports' straight lines can mimic a wrong-sized court's
    # sidelines/baselines, but nothing mimics 3-point arcs and circles
    for curves in (False, True):
        for p, c in polys:
            if c == curves:
                cv2.polylines(mk, [np.int32(p - off).reshape(-1, 1, 2)], False, 2 if curves else 1, 2 * tol + 1)
    wt = prep.weight[y0:y1, x0:x1]
    got = float((wt * mk).sum())
    if not enrichment:
        return got
    mk = (mk > 0).astype(np.uint8)
    got = float((wt * mk).sum())
    # how much denser the evidence is on the court lines than on the seen floor overall: random texture
    # (wood grain, noise) gives ~1, painted court lines give much more
    area = float(np.count_nonzero(mk))
    if area == 0:
        return 0.0
    return (got / area) / max(1e-9, prep.total / max(1, prep.n_cov))


def _fit_job(args):
    """fit_court_to_mosaic for one marking standard and court length (all widths and camera heights)."""
    prep, spec, L, widths, heights = args
    res, ext = prep.res, prep.ext
    out: list[CourtFit] = []
    court.configure(L, widths[0], spec)
    for W in widths:
        court.configure(L, W, spec)
        for h in heights:
            k, pad = _court_mask(h, res, 2)
            if k.shape[0] >= prep.weight.shape[0] or k.shape[1] >= prep.weight.shape[1]:
                continue
            sc = cv2.matchTemplate(prep.weight, k, cv2.TM_CCORR)
            jy = np.arange(sc.shape[0])
            cy = -((jy + pad) * res - ext) * h
            sc[(cy > 0.5) | (cy < -40)] = -1  # camera on the near side, outside the court
            j = np.unravel_index(int(np.argmax(sc)), sc.shape)
            cx = -((j[1] + pad) * res - ext) * h
            out.append(CourtFit(float(h), np.array([cx, cy[j[0]], h]), float(L), float(W),
                                float(sc[j]) * size_prior(float(L)), spec))
    return out


def fit_court_to_mosaic(mos: Mosaic, lengths=None, widths=None, heights=None, factor: int = 3,
                        cancel=None, specs=("FIBA",), keep: list | None = None, pool=None) -> CourtFit | None:
    """Brute force over court size and camera height; camera x/y by template matching (all at once).
    ``pool``: optional process pool - the court lengths are then tried in parallel."""
    prep = _prepare(mos, factor)
    if prep is None:
        return None
    lengths = np.arange(20.0, 29.01, 1.0) if lengths is None else lengths
    widths = np.array([15.0]) if widths is None else widths  # width is settled later in image space
    heights = np.exp(np.arange(np.log(0.8), np.log(15.0), 0.03)) if heights is None else heights
    L0, W0, S0 = court.LENGTH, court.WIDTH, court.SPEC
    jobs = [(prep, spec, float(L), list(widths), list(heights)) for spec in specs for L in lengths]
    best: CourtFit | None = None
    try:
        if pool is not None and len(jobs) > 1:
            outs = pool.map(_fit_job, jobs)
        else:
            outs = (_fit_job(j) for j in jobs)
        for out in outs:
            if cancel is not None and cancel.is_set():
                return best
            for cf in out:
                if keep is not None:
                    keep.append(cf)
                if best is None or cf.score > best.score:
                    best = cf
    finally:
        court.configure(L0, W0, S0)
    return best


def _rescore_one(prep: _Prepared, c: CourtFit, spec: str, fix_length: bool, fix_width: bool) -> CourtFit:
    """Coordinate ascent of one candidate under one marking standard (score = explained line evidence)."""
    # x = [length, log height, camera x from the court centre, camera y from the centre line, width]
    x = np.array([c.length, np.log(c.h), c.C[0] - c.length / 2, c.C[1] - c.width / 2, c.width])

    def f(y):
        if not (18.0 <= y[0] <= 30.0) or not (11.0 <= y[4] <= 16.0) or y[3] + y[4] / 2 > 0.5:
            return -1.0
        court.configure(y[0], y[4], spec)
        return explained(prep, float(np.exp(y[1])), y[2] + y[0] / 2, y[3] + y[4] / 2)

    cur = f(x)
    steps = np.array([0.0 if fix_length else 0.5, 0.02, 0.06 * c.h, 0.06 * c.h, 0.0 if fix_width else 0.5])
    for _round in range(4):  # coordinate ascent with shrinking steps
        for _ in range(8):
            improved = False
            for d in range(5):
                if steps[d] == 0:
                    continue
                for sgn in (-1, 1):
                    y = x.copy()
                    y[d] += sgn * steps[d]
                    v = f(y)
                    if v > cur + 1e-6:
                        x, cur, improved = y, v, True
            if not improved:
                break
        steps = steps / 2
    h = float(np.exp(x[1]))
    return CourtFit(h, np.array([x[2] + x[0] / 2, x[3] + x[4] / 2, h]), float(x[0]), float(x[4]), float(cur), spec)


def _rescore_job(args):
    prep, pairs, fix_length, fix_width = args
    return [_rescore_one(prep, c, spec, fix_length, fix_width) for c, spec in pairs]


def rescore(mos: Mosaic, cands: list[CourtFit], specs, cancel=None, factor: int = 2,
            fix_length: bool = False, fix_width: bool = False, pool=None) -> CourtFit | None:
    """Local refinement of the best coarse candidates (length, width, camera height and position) on a finer
    mosaic, trying every marking standard; keeps the court that explains the most line evidence. FIBA is
    preferred unless another standard explains clearly more (they differ only in a few lines)."""
    prep = _prepare(mos, factor)
    if prep is None:
        return None
    L0, W0, S0 = court.LENGTH, court.WIDTH, court.SPEC
    pairs = [(c, spec) for c in cands for spec in specs]
    try:
        fits = []
        if pool is not None and len(pairs) > 1:
            n = max(1, min(len(pairs), _pool_size()))
            idx_groups = [list(range(k, len(pairs), n)) for k in range(n)]
            got = pool.map(_rescore_job, [(prep, [pairs[i] for i in ig], fix_length, fix_width)
                                          for ig in idx_groups])
            fits = [None] * len(pairs)
            for ig, out in zip(idx_groups, got):
                for i, cf in zip(ig, out):
                    fits[i] = cf
        else:
            for c, spec in pairs:
                if cancel is not None and cancel.is_set():
                    break
                fits.append(_rescore_one(prep, c, spec, fix_length, fix_width))
    finally:
        court.configure(L0, W0, S0)
    best, best_rank = None, -np.inf
    for cf in fits:
        rank = cf.score * (1.0 if cf.spec == "FIBA" else 0.97) * size_prior(cf.length)
        if rank > best_rank:
            best, best_rank = cf, rank
    return best


COMMON_LENGTHS = (28.0, 26.0, 28.65)  # FIBA, old FIBA / small halls, NBA/NCAA


def size_prior(length: float) -> float:
    """Mild preference for common court lengths: an unusual size must be clearly better supported.
    Short clips that see only part of the court otherwise fall for other sports' lines."""
    d = min(abs(length - c) for c in COMMON_LENGTHS)
    return float(max(0.75, 1.0 - 0.03 * d))


def _diverse(cands: list[CourtFit], n_per_length: int = 2, n_top: int = 8) -> list[CourtFit]:
    cands = sorted(cands, key=lambda c: -c.score)
    out, per = list(cands[:n_top]), {}
    for c in cands:
        if per.get(c.length, 0) < n_per_length:
            per[c.length] = per.get(c.length, 0) + 1
            if not any(o is c for o in out):
                out.append(c)
    return out


# ------------------------------------------------------------------------------- per-frame refinement
def _dist_maps(a: Anchor, classes: tuple[int, ...] | None = None):
    w, h = a.work_size
    m = np.zeros((h, w), np.uint8)
    lp = a.lines if classes is None else a.lines[np.isin(a.lines[:, 2], classes)]
    if len(lp):
        m[np.clip(lp[:, 1], 0, h - 1), np.clip(lp[:, 0], 0, w - 1)] = 1
    dist = cv2.distanceTransform((1 - m).astype(np.uint8), cv2.DIST_L2, 3)
    return np.minimum(dist, TRUNC).astype(np.float32)


def _lookup(dist: np.ndarray, px: np.ndarray) -> np.ndarray:
    h, w = dist.shape
    x, y = px[:, 0], px[:, 1]
    ok = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (y >= 0) & (x < w - 1) & (y < h - 1)
    out = np.full(len(px), np.nan, np.float32)
    xi, yi = x[ok].astype(int), y[ok].astype(int)
    fx, fy = x[ok] - xi, y[ok] - yi
    out[ok] = (dist[yi, xi] * (1 - fx) * (1 - fy) + dist[yi, xi + 1] * fx * (1 - fy)
               + dist[yi + 1, xi] * (1 - fx) * fy + dist[yi + 1, xi + 1] * fx * fy)
    return out


def _project(C, R, f, size, pts2d, s):
    H = court_to_image_R(C, R, f, size)
    q = np.column_stack([pts2d, np.ones(len(pts2d))]) @ H.T
    ok = q[:, 2] > 1e-6
    p = np.full((len(pts2d), 2), np.nan)
    p[ok] = q[ok, :2] / q[ok, 2:3] * s
    return p


def anchor_cost(a: Anchor, C, size, pts, dist=None) -> tuple[float, float]:
    if a.R is None:
        return TRUNC, 0.0
    dist = _dist_maps(a) if dist is None else dist
    s = a.work_size[0] / size[0]
    p = _project(C, a.R, a.f, size, pts, s)
    d = _lookup(dist, p)
    vis = np.isfinite(d) & (p[:, 1] > 0.25 * dist.shape[0])
    if vis.sum() < 30:
        return TRUNC, 0.0
    return float(np.mean(d[vis])), float(np.mean(d[vis] < 2.0))


# --------------------------------------------------------------------------------------- the driver
def refine_pose(a: Anchor, C0: np.ndarray, size, pts, dist, sigma_c: float = 1.0, max_deg: float = 6.0) -> None:
    """Joint camera position + rotation + zoom refinement on this frame's line pixels, with priors."""
    if a.pan is None or a.vp is None:
        return
    s = a.work_size[0] / size[0]
    R0 = rotation(a.pan, a.vp[1], a.vp[2])
    f0 = float(np.exp(a.vp[3]))
    top = 0.2 * dist.shape[0]

    def make(z):
        return Rotation.from_rotvec(z[:3]).as_matrix() @ R0, f0 * np.exp(z[3]), C0 + z[4:7]

    def resid(z, dmap):
        R, f, C = make(z)
        if C[2] < 0.3:
            return np.full(len(pts) + 7, 50.0)
        p = _project(C, R, f, size, pts, s)
        d = _lookup(dmap, p)
        vis = np.isfinite(d) & (p[:, 1] > top)
        r = np.where(vis, d, 0.0) / np.sqrt(max(1, vis.sum())) * 10
        lim = np.radians(max_deg)
        prior = np.array([z[0] / lim, z[1] / lim, z[2] / lim, z[3] / 0.15,
                          z[4] / sigma_c, z[5] / sigma_c, z[6] / (0.5 * sigma_c)]) * 0.6
        return np.concatenate([r, prior])

    z = np.zeros(7)
    for blur in (4.0, 1.5, 0.0):
        dmap = cv2.GaussianBlur(dist, (0, 0), blur) if blur else dist
        try:
            z = least_squares(resid, z, args=(dmap,), diff_step=1e-3, max_nfev=80,
                              x_scale=[0.01, 0.01, 0.01, 0.02, 0.2, 0.2, 0.1]).x
        except ValueError:
            break
    R, f, C = make(z)
    a.R, a.f = R, float(f)
    a.C = C  # type: ignore[attr-defined]
    p = _project(C, R, f, size, pts, s)
    d = _lookup(dist, p)
    vis = np.isfinite(d) & (p[:, 1] > top)
    if vis.sum() >= 30:
        a.cost, a.inliers = float(np.mean(d[vis])), float(np.mean(d[vis] < 2.0))
    else:
        a.cost, a.inliers = TRUNC, 0.0
    a.good = bool(vis.sum() >= 60 and a.inliers >= 0.35 and a.cost <= 4.0)


def chain_rotation(G: np.ndarray, f: float, f_ref: float, size: tuple[int, int]) -> np.ndarray:
    """Relative camera rotation A with R_frame = A @ R_ref, from a frame -> segment-reference homography."""
    M = np.linalg.inv(K_of(f, size)) @ np.linalg.inv(G) @ K_of(f_ref, size)
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    return -R if np.linalg.det(R) < 0 else R


def bundle_refine(mem: list[Anchor], A: dict[int, np.ndarray], C0: dict[int, np.ndarray], size, pts: np.ndarray,
                  dists: dict[int, np.ndarray], knot_s: float = 8.0, n_pts: int = 160, fit_C: bool = True) -> None:
    """Joint alignment of all anchors of one camera-motion segment.

    Each anchor's rotation is R_i = A_i @ R_ref(t_i): A_i is the measured camera motion (precise frame to
    frame, but drifting), R_ref(t) a slowly varying correction (knots every `knot_s` seconds, linear in
    between) that absorbs the drift. Zoom likewise varies smoothly. All anchors' line evidence is fitted at
    once, with the camera position shared (plus per-window offsets from C0), so clutter in a single frame
    cannot drag that frame away. Updates a.R, a.f, a.C, a.cost, a.inliers, a.good.
    """
    mem = [a for a in mem if a.R is not None and a.i in dists]
    if len(mem) < 2:
        return
    t = np.array([a.t for a in mem])
    knots = np.arange(t[0], t[-1] + knot_s, knot_s)
    if len(knots) < 2:
        knots = np.array([t[0], t[0] + knot_s])
    K = len(knots)
    # initial R_ref per anchor (from its own vanishing-point rotation) and a robust smooth version
    refs = Rotation.from_matrix(np.stack([A[a.i].T @ a.R for a in mem]))
    base = refs[len(refs) // 2]
    rel = (base.inv() * refs).as_rotvec()
    logf = np.log(np.array([a.f for a in mem]))
    rv_k, lf_k = np.zeros((K, 3)), np.zeros(K)
    for k in range(K):
        w = np.abs(t - knots[k]) <= max(knot_s, 15.0)
        if not w.any():
            w = np.abs(t - knots[k]) <= np.min(np.abs(t - knots[k])) + 1e-6
        rv_k[k] = np.median(rel[w], axis=0)
        lf_k[k] = np.median(logf[w])
    s = mem[0].work_size[0] / size[0]
    # fixed subset of model points per anchor: those visible under the initial pose
    sel_pts = []
    for a in mem:
        p = _project(C0[a.i], a.R, a.f, size, pts, s)
        h_, w_ = dists[a.i].shape
        vis = np.where(np.isfinite(p).all(axis=1) & (p[:, 0] > -20) & (p[:, 0] < w_ + 20)
                       & (p[:, 1] > 0.2 * h_) & (p[:, 1] < h_ + 20))[0]
        if len(vis) > n_pts:
            vis = vis[np.linspace(0, len(vis) - 1, n_pts).astype(int)]
        sel_pts.append(pts[vis])
    seg_of = np.clip(np.searchsorted(knots, t, side="right") - 1, 0, K - 2)
    u_of = (t - knots[seg_of]) / (knots[seg_of + 1] - knots[seg_of])
    nC = 3 if fit_C else 0

    def unpack(z):
        dC = z[:nC] if fit_C else np.zeros(3)
        kv = z[nC:].reshape(K, 4)
        return dC, kv

    def poses(z):
        dC, kv = unpack(z)
        out = []
        for j, a in enumerate(mem):
            k, u = seg_of[j], u_of[j]
            r = (1 - u) * (rv_k[k] + kv[k, :3]) + u * (rv_k[k + 1] + kv[k + 1, :3])
            lf = (1 - u) * (lf_k[k] + kv[k, 3]) + u * (lf_k[k + 1] + kv[k + 1, 3])
            Rr = (base * Rotation.from_rotvec(r)).as_matrix()
            out.append((A[a.i] @ Rr, float(np.exp(lf)), C0[a.i] + dC))
        return out

    def resid(z, dmaps):
        res = []
        for (R, f, C), a, P in zip(poses(z), mem, sel_pts):
            if not len(P):
                continue
            p = _project(C, R, f, size, P, s)
            d = _lookup(dmaps[a.i], p)
            res.append(np.where(np.isfinite(d), d, TRUNC) / np.sqrt(len(P)))
        dC, kv = unpack(z)
        prior = [kv[:, :3].ravel() / np.radians(4.0) * 0.3, kv[:, 3] / 0.1 * 0.3,
                 np.diff(kv[:, :3], axis=0).ravel() / np.radians(1.0) * 0.3]
        if fit_C:
            prior.append(dC / 0.5 * 0.5)
        return np.concatenate(res + [np.ravel(x) for x in prior])

    # sparsity: anchor residuals depend on C, and on the two knots around it
    n_par = nC + 4 * K
    rows_n = [len(P) for P in sel_pts if len(P)]
    n_res = sum(rows_n) + 3 * K + K + 3 * (K - 1) + nC
    from scipy.sparse import lil_matrix

    J = lil_matrix((n_res, n_par), dtype=int)
    r0 = 0
    for j, P in enumerate(sel_pts):
        if not len(P):
            continue
        k = seg_of[j]
        J[r0:r0 + len(P), :nC] = 1
        J[r0:r0 + len(P), nC + 4 * k:nC + 4 * k + 8] = 1
        r0 += len(P)
    for k in range(K):  # knot priors
        J[r0 + 3 * k:r0 + 3 * k + 3, nC + 4 * k:nC + 4 * k + 3] = 1
    r0 += 3 * K
    for k in range(K):
        J[r0 + k, nC + 4 * k + 3] = 1
    r0 += K
    for k in range(K - 1):
        J[r0 + 3 * k:r0 + 3 * k + 3, nC + 4 * k:nC + 4 * k + 7] = 1
    r0 += 3 * (K - 1)
    if fit_C:
        J[r0:r0 + 3, :3] = 1
    z = np.zeros(n_par)
    for blur in (3.0, 1.0, 0.0):
        dm = {i: (cv2.GaussianBlur(d, (0, 0), blur) if blur else d) for i, d in dists.items()}
        try:
            z = least_squares(resid, z, args=(dm,), jac_sparsity=J, diff_step=1e-3, max_nfev=40,
                              x_scale=np.r_[np.full(nC, 0.2), np.tile([0.01, 0.01, 0.01, 0.02], K)],
                              loss="soft_l1", f_scale=0.5).x
        except ValueError:
            break
    for (R, f, C), a in zip(poses(z), mem):
        a.R, a.f = R, f
        a.C = C  # type: ignore[attr-defined]
        P, _ = pts, None
        p = _project(C, R, f, size, pts, s)
        d = _lookup(dists[a.i], p)
        vis = np.isfinite(d) & (p[:, 1] > 0.2 * dists[a.i].shape[0])
        if vis.sum() >= 30:
            a.cost, a.inliers = float(np.mean(d[vis])), float(np.mean(d[vis] < 2.0))
        else:
            a.cost, a.inliers = TRUNC, 0.0
        a.good = bool(vis.sum() >= 40 and a.inliers >= 0.18 and a.cost <= 6.5)
        if a.good and a.vp is not None and a.pan is not None:
            # independent check: this frame's own vanishing-point rotation never drifts; a joint solution that
            # disagrees with it by more than a few degrees has latched onto the wrong lines
            Rv = rotation(a.pan, a.vp[1], a.vp[2])
            ang = float(np.degrees(np.linalg.norm(Rotation.from_matrix(R @ Rv.T).as_rotvec())))
            a.good = ang <= 5.0


def pan_options(a: Anchor, C: np.ndarray, size, pts: np.ndarray, dist: np.ndarray):
    """The 4 possible pans of a frame (vanishing points give pan modulo 90 degrees) and how well the court
    matches this frame's lines under each: lower = better. Returns (pans, scores, (cost, inliers) of best)."""
    if a.vp is None:
        return None
    s_ = a.work_size[0] / size[0]
    pm = float(a.vp[0]) % (np.pi / 2)
    f = float(np.exp(a.vp[3]))
    pans, scores, stats = [], [], []
    for k in range(4):
        pan = pm + k * np.pi / 2
        R = rotation(pan, a.vp[1], a.vp[2])
        p = _project(C, R, f, size, pts, s_)
        d = _lookup(dist, p)
        vis = np.isfinite(d) & (p[:, 1] > 0.2 * dist.shape[0])
        if vis.sum() < 30:
            pans.append(pan)
            scores.append(50.0)
            stats.append((TRUNC, 0.0))
            continue
        c, inl = float(np.mean(d[vis])), float(np.mean(d[vis] < 2.0))
        pans.append(pan)
        scores.append(c - 4.0 * inl)
        stats.append((c, inl))
    return np.array(pans), np.array(scores), stats


def _pan_job(args):
    a, C, size, L, W, spec, classes = args
    court.configure(L, W, spec)
    pts, _ = model_points_dirs(0.35)
    return a.i, pan_options(a, C, size, pts, _dist_maps(a, with_weak(classes)))


def choose_pans(anchors: list[Anchor], opts: dict) -> None:
    """Pick each frame's pan (one of 4) jointly over time: each frame's own line match, plus the camera
    motion measured between frames (a 90-degree jump is only allowed if the camera really turned that far).
    Viterbi per camera-motion segment."""
    for s_ in sorted({a.seg for a in anchors}):
        mem = [a for a in anchors if a.seg == s_ and opts.get(a.i) is not None]
        if not mem:
            continue
        U = np.array([opts[a.i][1] for a in mem])  # (n, 4)
        P = np.array([opts[a.i][0] for a in mem])
        n = len(mem)
        D = U[0].copy()
        back = np.zeros((n, 4), int)
        for j in range(1, n):
            pred = -np.radians(mem[j].relpan - mem[j - 1].relpan)  # turning right lowers the pan
            act = P[j][None, :] - P[j - 1][:, None]  # (prev k, cur k)
            err = np.abs((act - pred + np.pi) % (2 * np.pi) - np.pi)
            trans = np.where(err < np.radians(20.0), 0.0, 12.0)
            tot = D[:, None] + trans
            back[j] = np.argmin(tot, axis=0)
            D = tot[back[j], np.arange(4)] + U[j]
        k = int(np.argmin(D))
        for j in range(n - 1, -1, -1):
            mem[j].pan = float(P[j][k])
            k = back[j][k] if j > 0 else k


def _probe_job(args):
    """Can this frame be aligned under the given court/camera? (used to pick between court candidates)"""
    a, C, size, L, W, spec, classes = args
    court.configure(L, W, spec)
    pts, _ = model_points_dirs(0.3)
    dist = _dist_maps(a, with_weak(classes))
    o = pan_options(a, C, size, pts, dist)
    if o is None:
        return False
    k = int(np.argmin(o[1]))
    R_vp = rotation(float(o[0][k]), a.vp[1], a.vp[2])
    R, f, c, inl, nv = refine_frame(R_vp, float(np.exp(a.vp[3])), C, size, pts, dist, a.work_size[0] / size[0])
    agree = float(np.degrees(np.linalg.norm(Rotation.from_matrix(R @ R_vp.T).as_rotvec()))) <= 5.0
    return bool(nv >= 40 and inl >= 0.2 and c <= 6.0 and agree)


def _window_job(args):
    """Does this time window clearly fit another camera position than the main one? Returns it, or None."""
    win, size, classes, L, W, spec, C_glob, mkw = args
    court.configure(L, W, spec)
    m_w = build_mosaic(win, size, classes=classes, **mkw)
    prep = _prepare(m_w, 2)
    if prep is None:
        return None
    e_glob = explained(prep, C_glob[2], C_glob[0], C_glob[1])
    kw: list[CourtFit] = []
    heights = C_glob[2] * np.exp(np.linspace(-0.4, 0.4, 21))  # a moved camera stays at a similar height
    fit_court_to_mosaic(m_w, lengths=[L], widths=[W], specs=(spec,), keep=kw, heights=heights)
    fw = rescore(m_w, _diverse(kw, 1, 3), (spec,), fix_length=True, fix_width=True) if kw else None
    if fw is not None and fw.score > 1.2 * e_glob + 1e-6 and np.linalg.norm(fw.C - C_glob) > 1.0:
        return fw.C
    return None


def refine_frame(R0: np.ndarray, f0: float, C: np.ndarray, size, pts: np.ndarray, dist: np.ndarray, s: float,
                 max_deg: float = 6.0, zoom_sigma: float = 0.15):
    """Rotation + zoom refinement of one frame on its line pixels (camera position fixed).

    The model points visible under the starting pose are fixed; a point that leaves the image counts as a
    full miss, so the optimiser cannot 'win' by pushing the court out of view.
    Returns (R, f, cost, inliers, n_visible)."""
    h_, w_ = dist.shape
    top = 0.2 * h_
    p0 = _project(C, R0, f0, size, pts, s)
    vis0 = np.isfinite(p0).all(axis=1) & (p0[:, 0] >= 0) & (p0[:, 0] < w_ - 1) & (p0[:, 1] > top) & (p0[:, 1] < h_ - 1)
    P = pts[vis0]
    if len(P) < 30:
        return R0, f0, TRUNC, 0.0, int(len(P))
    if len(P) > 400:
        P = P[np.linspace(0, len(P) - 1, 400).astype(int)]

    def make(z):
        return Rotation.from_rotvec(z[:3]).as_matrix() @ R0, f0 * np.exp(z[3])

    def resid(z, dmap):
        R, f = make(z)
        d = _lookup(dmap, _project(C, R, f, size, P, s))
        r = np.where(np.isfinite(d), d, TRUNC) / np.sqrt(len(P)) * 10
        lim = np.radians(max_deg)
        return np.concatenate([r, np.array([z[0] / lim, z[1] / lim, z[2] / lim, z[3] / zoom_sigma]) * 0.6])

    z = np.zeros(4)
    for blur in (4.0, 1.5, 0.0):
        dmap = cv2.GaussianBlur(dist, (0, 0), blur) if blur else dist
        try:
            z = least_squares(resid, z, args=(dmap,), diff_step=1e-3, max_nfev=60, loss="soft_l1", f_scale=1.0,
                              x_scale=[0.01, 0.01, 0.01, 0.02]).x
        except ValueError:
            break
    R, f = make(z)
    p = _project(C, R, f, size, pts, s)
    d = _lookup(dist, p)
    vis = np.isfinite(d) & (p[:, 1] > top)
    if vis.sum() < 30:
        return R, f, TRUNC, 0.0, int(vis.sum())
    return R, f, float(np.mean(d[vis])), float(np.mean(d[vis] < 2.0)), int(vis.sum())


def search_frame(R0: np.ndarray, f0: float, C: np.ndarray, size, pts: np.ndarray, dist: np.ndarray, s: float,
                 max_shift: int = 60, step: int = 2):
    """Brute-force alignment for a frame whose starting pose may be a few degrees off.

    Small camera rotations/zooms move the projected court roughly by a shift, a scale about the image centre
    and a roll. All combinations within a window are tried at once (vectorised); the one putting the most
    court points on detected lines wins, and the exact camera angles are then fitted to that placement.
    Returns (R, f) or None."""
    h_, w_ = dist.shape
    top = 0.2 * h_
    p = _project(C, R0, f0, size, pts, s)
    ok = np.isfinite(p).all(axis=1) & (p[:, 1] > top - 40) & (p[:, 0] > -80) & (p[:, 0] < w_ + 80) & (p[:, 1] < h_ + 60)
    P3, p = pts[ok], p[ok]
    if len(p) < 40:
        return None
    if len(p) > 300:
        sel = np.linspace(0, len(p) - 1, 300).astype(int)
        P3, p = P3[sel], p[sel]
    c = np.array([w_ / 2.0, h_ / 2.0])
    shifts = np.arange(-max_shift, max_shift + 1, step)
    DX, DY = np.meshgrid(shifts, shifts, indexing="xy")
    best = (-1.0, None)
    for roll in np.radians([-1.5, -0.75, 0.0, 0.75, 1.5]):
        cr, sr = np.cos(roll), np.sin(roll)
        Rm = np.array([[cr, -sr], [sr, cr]])
        for sc in np.exp(np.linspace(-0.12, 0.12, 9)):
            q = c + sc * (p - c) @ Rm.T  # (N, 2)
            X = np.round(q[:, 0][:, None, None] + DX[None]).astype(int)
            Y = np.round(q[:, 1][:, None, None] + DY[None]).astype(int)
            inside = (X >= 0) & (X < w_) & (Y > top) & (Y < h_)
            d = np.where(inside, dist[np.clip(Y, 0, h_ - 1), np.clip(X, 0, w_ - 1)], TRUNC)
            score = np.mean(d < 2.5, axis=0) - 0.02 * np.mean(d, axis=0) / TRUNC
            j = np.unravel_index(int(np.argmax(score)), score.shape)
            if score[j] > best[0]:
                best = (float(score[j]), q + np.array([DX[j], DY[j]]))
    if best[1] is None:
        return None
    target = best[1] / s  # full-resolution pixels

    def resid(z):
        R = Rotation.from_rotvec(z[:3]).as_matrix() @ R0
        return (_project(C, R, f0 * np.exp(z[3]), size, P3, 1.0) - target).ravel()

    try:
        z = least_squares(resid, np.zeros(4), x_scale=[0.01, 0.01, 0.01, 0.02], max_nfev=100).x
    except ValueError:
        return None
    return Rotation.from_rotvec(z[:3]).as_matrix() @ R0, f0 * float(np.exp(z[3]))


def track_chunk(mem: list[Anchor], G: dict, C_of: dict, size, pts: np.ndarray, pts_coarse: np.ndarray,
                dists: dict) -> None:
    """Follow the camera through one stretch of anchors (time order, one motion segment).

    For each frame two starting poses are refined on its lines: (1) the previous frame's pose carried over
    by the measured camera motion - exact frame to frame, including zoom - and (2) the frame's own
    vanishing-point pose (best of its 4 pans). The better one wins. A frame is 'good' when it matches its
    lines well AND agrees with its own vanishing points; otherwise the prediction is kept for continuity,
    but never a bad refinement."""
    prev = None  # (anchor, court->image H) of the last frame we trust enough to predict from
    for a in mem:
        C = C_of[a.i]
        dist = dists[a.i]
        s_ = a.work_size[0] / size[0]
        cands = []
        R_vp = None
        o = pan_options(a, C, size, pts_coarse, dist)
        if o is not None:
            k = int(np.argmin(o[1]))
            a.pan = float(o[0][k])
            R_vp = rotation(a.pan, a.vp[1], a.vp[2])
            cands.append((R_vp, float(np.exp(a.vp[3]))))
        if prev is not None:
            Hp = np.linalg.inv(G[a.i]) @ G[prev[0].i] @ prev[1]
            init = ptz_from_R(prev[0].R, prev[0].f)
            v = fit_ptz_to_homography(C, Hp, size, init)
            if v is not None:
                cands.append((rotation(v.pan, v.tilt, v.roll), float(v.f)))
        best = None

        def consider(R0, f0):
            nonlocal best
            R, f, c, inl, nv = refine_frame(R0, f0, C, size, pts, dist, s_)
            score = c - 4.0 * inl if nv >= 40 else 50.0
            if best is None or score < best[0]:
                best = (score, R, f, c, inl, nv)

        for R0, f0 in cands:
            consider(R0, f0)
        if best is not None and not (best[4] >= 0.2 and best[3] <= 6.5):
            # not clearly aligned: brute-force search around each start, then polish
            for R0, f0 in cands:
                r = search_frame(R0, f0, C, size, pts_coarse, dist, s_)
                if r is not None:
                    consider(*r)
        a.C = C  # type: ignore[attr-defined]
        if best is None:
            a.R, a.good = None, False
            continue
        _sc, R, f, c, inl, nv = best
        agree = True
        if R_vp is not None:
            agree = float(np.degrees(np.linalg.norm(Rotation.from_matrix(R @ R_vp.T).as_rotvec()))) <= 5.0
        a.R, a.f, a.cost, a.inliers = R, float(f), c, inl
        # usable = reasonably aligned and consistent with its own vanishing points; single wrong frames that
        # slip through are removed later by comparing each frame with its neighbours in time (resolve_auto)
        a.good = bool(nv >= 40 and inl >= 0.15 and c <= 7.0 and agree)
        if a.good:
            prev = (a, court_to_image_R(C, R, f, size))
        elif prev is not None and len(cands) == 2:
            # keep following the camera motion from the last trusted frame (continuity), unrefined
            prev = (prev[0], prev[1])


def _track_job(args):
    mem, G, C_of, size, L, W, spec, classes = args
    court.configure(L, W, spec)
    pts, _ = model_points_dirs(0.25)
    pts_c, _ = model_points_dirs(0.35)
    track_chunk(mem, G, C_of, size, pts, pts_c, {a.i: _dist_maps(a, with_weak(classes)) for a in mem})
    return mem


def _bundle_job(args):
    mem, A, C0, size, L, W, spec, classes = args
    court.configure(L, W, spec)
    pts, _ = model_points_dirs(0.25)
    bundle_refine(mem, A, C0, size, pts, {a.i: _dist_maps(a, with_weak(classes)) for a in mem})
    return mem


def _pose_job(args):
    a, C0, size, L, W, spec, classes = args
    court.configure(L, W, spec)
    pts, _ = model_points_dirs(0.25)
    refine_pose(a, C0, size, pts, _dist_maps(a, with_weak(classes)))
    return a



@dataclass
class AutoResult:
    C: np.ndarray
    length: float
    width: float
    spec: str = "FIBA"
    line_classes: tuple[int, ...] = (0,)
    anchors: list[dict] = field(default_factory=list)  # t, i, pan, tilt, roll, f, cost, good, C
    good_frac: float = 0.0
    message: str = ""
    frames: list[dict] = field(default_factory=list)  # court detector: {"t", "i", "H"} court->image per frame
    max_gap_s: float | None = None  # frames: how far a located frame's H is carried by the motion chain

    def to_json(self) -> dict:
        d = {"version": AUTO_VERSION, "C": [round(float(v), 4) for v in self.C], "spec": self.spec,
             "line_classes": [int(c) for c in self.line_classes], "anchors": self.anchors,
             "good_frac": round(self.good_frac, 3), "message": self.message}
        if self.frames:
            d["frames"] = self.frames
            if self.max_gap_s is not None:
                d["max_gap_s"] = float(self.max_gap_s)
        return d


CLASS_NAMES = ["black", "white", "cyan", "blue", "purple/magenta", "red/orange", "yellow", "green"]
WINDOW_S = 90.0  # camera position is re-checked per window: moved tripods, walking operators, TV cuts
CHUNK_S = 60.0  # frame-to-frame tracking runs on chunks of this length (in parallel)
MAX_EXTRAP_S = 4.0  # frames this far outside the verified anchors still get a calibration
STREAM_MIN_LOCK = 150  # anchors (~2.5 min of video) before a streaming round may lock the court
STREAM_TOL_M = 0.15  # a court refit closer than this to the streamed one keeps the streamed alignments
MAX_BRIDGE_S = 20.0  # longer gaps between verified anchors are only bridged near their ends


def _pool_size() -> int:
    return max(1, min(8, (os.cpu_count() or 2) - 1))


def calibrate(video: Path, data, size: tuple[int, int], every_s: float = 1.0,
              progress: Callable[[float, str], None] | None = None, cancel=None,
              debug: dict | None = None, work: Path | None = None, n_workers: int | None = None,
              use_state: bool = True) -> AutoResult | None:
    """Automatic calibration for pass-1 data (load_pass1). Returns None when it cannot find the court.
    ``work``: the pass-1 work folder; if pass 1 ran with replay.autocal_tap.AutocalTap, the per-anchor image
    measurements are already there (autocal_features.pkl) and the video is not read again."""
    # ONE process pool for every step (starting workers costs seconds each, much more on a busy PC); short clips
    # get fewer workers (a 30-s clip has ~30 anchors)
    span = float(data.t[-1] - data.t[0]) if len(data.t) else 0.0
    n_workers = n_workers or min(_pool_size(), max(2, int(span / every_s) // 10))
    with ProcessPoolExecutor(n_workers) as fitpool:
        return _calibrate_impl(video, data, size, every_s, progress, cancel, debug, work, n_workers, use_state,
                               fitpool)


def _calibrate_impl(video, data, size, every_s, progress, cancel, debug, work, n_workers, use_state,
                    fitpool) -> AutoResult | None:
    dbg = debug if debug is not None else {}
    prog = progress or (lambda x, m: None)
    t0 = time.time()
    nw = n_workers or _pool_size()
    n = len(data.t)
    if n == 0:
        return None
    G, seg = chain(data.motion, n)
    idx, last = [], -1e9
    for i in range(n):
        if data.t[i] - last >= every_s - 1e-6 or (i > 0 and seg[i] != seg[i - 1]):
            idx.append(i)
            last = data.t[i]
    rows = data.rows
    persons = rows[rows[:, 6] == 0] if len(rows) else rows
    if len(persons):  # sorted by sample index (stable: same boxes, same order) -> binary search per anchor
        persons = persons[np.argsort(persons[:, 0], kind="stable")]
    pk = persons[:, 0] if len(persons) else np.zeros(0)

    feats = load_features(work, data)
    state = load_state(work) if (use_state and feats is not None) else None
    if feats is not None:
        # measured during pass 1: only the overlay mask is left to apply (the streaming rounds fix the mask
        # once and remember each frame's refitted rotation)
        prog(0.0, "Using the camera measurements made during the analysis")
        if state is not None and state.get("mask") is not None:
            static = state["mask"]
        else:
            static = static_overlay_mask(feats.get("small") or [])
        known = (state or {}).get("finished", {}) if state is not None and state.get("mask") is not None else {}
        with nullcontext(fitpool) as ex:
            anchors = list(ex.map(_anchor_finish, [(r, static, known.get(r[0])) for r in feats["raws"]],
                                  chunksize=4))
        dbg["features"] = "tap"
        dbg["mask"] = static
        dbg["finished"] = {a.i: (a.vp, a.vp_inliers) for a in anchors}
    else:
        prog(0.0, "Looking for broadcast overlays")
        probe = [int(data.frame[i]) for i in idx[:: max(1, len(idx) // 40)]][:40]
        static = static_overlay_mask([img for _f, img in read_frames(video, probe, cancel)])

        frame_to_i = {int(data.frame[i]): i for i in idx}
        anchors: list[Anchor] = []
        with nullcontext(fitpool) as ex:
            # 1. per-anchor camera rotation (vanishing points) + line pixels
            pending = []
            for fno, img in read_frames(video, list(frame_to_i), cancel):
                i = frame_to_i[fno]
                if len(persons):
                    lo, hi = np.searchsorted(pk, data.k[i], "left"), np.searchsorted(pk, data.k[i], "right")
                    boxes = persons[lo:hi, 1:5]
                else:
                    boxes = None
                pending.append(ex.submit(_anchor_features, (i, float(data.t[i]), fno, img, boxes, static, None)))
                while len(pending) >= 3 * _pool_size():
                    anchors.append(pending.pop(0).result())
                    prog(0.5 * len(anchors) / len(idx), f"Measuring camera angles: frame {len(anchors)} / {len(idx)}")
            for pf in pending:
                anchors.append(pf.result())
            if cancel is not None and cancel.is_set():
                return None
    anchors.sort(key=lambda a: a.t)
    fs = [np.exp(a.vp[3]) for a in anchors if a.vp is not None and a.vp_inliers > 0.4]
    dbg["n_anchors"] = len(anchors)
    dbg["n_vp_good"] = len(fs)
    if len(fs) < 3:
        dbg["fail"] = "fewer than 3 frames with a vanishing-point rotation"
        return None
    f_med = float(np.median(fs))
    for a in anchors:
        a.seg = int(seg[a.i])
        a.relpan = relative_pan_deg(G[a.i], f_med, size)

    # 2. pan disambiguation per camera-motion segment
    hyps = disambiguate(anchors)
    if not hyps:
        dbg["fail"] = "no pan hypothesis"
        return None
    segs = sorted(hyps, key=lambda s_: -sum(1 for a in anchors if a.seg == s_))
    main = segs[0]
    use = [a for a in anchors if a.seg == main and a.vp is not None and a.vp_inliers > 0.3]
    dbg["n_segments"] = len(set(a.seg for a in anchors))
    dbg["n_use"] = len(use)
    if len(use) < 3:
        dbg["fail"] = f"main motion segment has {len(use)} usable frames"
        return None
    use = use[:: max(1, len(use) // 300)]
    # mosaic extent/resolution (camera-height units) from how steeply the camera looks down: a high broadcast
    # camera sees the court within a few camera heights, a low gym camera out to ~15 - the court must always
    # span many mosaic cells (a low camera keeps the old 16 / 0.02)
    tilt_med = float(np.median([a.vp[1] for a in use]))
    ext_m = float(np.clip(2.5 / np.tan(max(tilt_med, 0.02)), 3.0, 16.0))
    mos_kw = {"ext": ext_m, "res": ext_m / 800.0}
    dbg["mosaic_ext"] = ext_m

    def _finish(fit0, off0, classes, keep, mos, prev=None):
        keep = keep + [fit0]
        for a in use:
            set_pan(a, off0)
        prog(0.56, f"Court lines are {'+'.join(CLASS_NAMES[c] for c in classes)}; measuring the court")
        fit = rescore(mos, _diverse(keep), tuple(court.STANDARDS), cancel=cancel, pool=fitpool) or fit0
        L, W, spec = fit.length, fit.width, fit.spec
        court.configure(L, W, spec)
        C_glob = fit.C.copy()
        dbg.update(anchors=anchors, use=use, mosaic=mos, fit0=fit0, fit=fit, hyps=hyps, main=main, classes=classes)
        # streaming: frames aligned in an earlier round are kept if the court has not changed meanwhile
        reuse: dict = {}
        if prev is not None and prev.get("spec") == spec and abs(prev["L"] - L) < STREAM_TOL_M                 and abs(prev["W"] - W) < STREAM_TOL_M and np.linalg.norm(np.asarray(prev["C"]) - C_glob) < STREAM_TOL_M:
            have = {a.i for a in anchors}
            reuse = {i: v for i, v in prev.get("tracked", {}).items() if i in have}
            L, W, C_glob = prev["L"], prev["W"], np.asarray(prev["C"], float)  # identical court -> identical frames
            court.configure(L, W, spec)
        dbg["reused"] = len(reuse)

        # 4. every frame's pan (vanishing points give it modulo 90 degrees): each frame tries all 4 against its
        #    own lines, and the camera motion between frames keeps the choice consistent over time
        with nullcontext(fitpool) as ex:
            jobs = [a for a in anchors if a.vp is not None and a.i not in reuse]
            opts = {}
            for j, (i, o) in enumerate(ex.map(_pan_job, [(a, C_glob, size, L, W, spec, classes) for a in jobs],
                                             chunksize=8)):
                opts[i] = o
                if j % 50 == 0:
                    prog(0.56 + 0.02 * j / max(1, len(jobs)), "Checking each frame's viewing direction")
        for a in anchors:  # reused frames: their pan is known; one sure option keeps the time chain consistent
            if a.i in reuse and a.vp is not None:
                pan = float(reuse[a.i][6])
                opts[a.i] = (np.array([pan] * 4), np.array([0.0, 50.0, 50.0, 50.0]), [(0.0, 1.0)] * 4)
        choose_pans(anchors, opts)

        # 5. camera position per time window: the same position unless a window clearly fits another one
        #    (tripod moved, camera carried along the sideline, TV broadcast cutting between cameras)
        C_of: dict[int, np.ndarray] = {}
        windows = []
        for s_ in sorted({a.seg for a in anchors}):
            mem = [a for a in anchors if a.seg == s_ and a.pan is not None]
            if not mem:
                continue
            t_s = mem[0].t
            cur: list[Anchor] = []
            for a in mem:
                if a.t - t_s > WINDOW_S and len(cur) >= 10:
                    windows.append(cur)
                    cur, t_s = [], a.t
                cur.append(a)
            if cur:
                windows.append(cur)
        moved = 0
        todo_w = [w for w in windows if len(w) >= 10 and not all(a.i in reuse for a in w)] if len(windows) > 1 else []
        results_w: dict[int, np.ndarray | None] = {}
        if todo_w:
            with nullcontext(fitpool) as ex:
                futs = {ex.submit(_window_job, ([a for a in w if a.vp_inliers > 0.3], size, classes, L, W, spec,
                                                C_glob, mos_kw)): k for k, w in enumerate(todo_w)}
                for j, fu in enumerate(as_completed(futs)):
                    if cancel is not None and cancel.is_set():
                        return None
                    results_w[futs[fu]] = fu.result()
                    prog(0.58 + 0.07 * (j + 1) / len(todo_w),
                         f"Checking the camera position over time: {j + 1} / {len(todo_w)}")
        for k, w in enumerate(todo_w):
            if results_w.get(k) is not None:
                moved += 1
        moved_of = {id(w): results_w.get(k) for k, w in enumerate(todo_w)}
        for win in windows:
            C_w = moved_of.get(id(win))
            for a in win:
                C_of[a.i] = np.asarray(reuse[a.i][5], float) if a.i in reuse else (C_glob if C_w is None else C_w)
        court.configure(L, W, spec)

        # 6. joint alignment (bundle adjustment) per camera-motion segment, in chunks of a few minutes.
        #    Zoom-aware: the motion chain is turned into rotations with each frame's own focal length (the
        #    operator may zoom) and the focal length of the segment's first frame (always an anchor).
        def f_of(a):
            return float(np.exp(a.vp[3])) if a.vp is not None and a.vp_inliers > 0.3 else f_med

        f_ref = {}
        for a in anchors:
            if a.seg not in f_ref:
                f_ref[a.seg] = f_of(a)
        for a in anchors:
            if a.pan is not None and a.vp is not None:
                a.R, a.f = rotation(a.pan, a.vp[1], a.vp[2]), float(np.exp(a.vp[3]))
        chunks = []
        for s_ in sorted({a.seg for a in anchors}):
            mem = [a for a in anchors if a.seg == s_ and a.R is not None and a.i in C_of and a.i not in reuse]
            cur: list[Anchor] = []
            for a in mem:
                if cur and a.t - cur[0].t > CHUNK_S:
                    chunks.append(cur)
                    cur = []
                cur.append(a)
            if cur:
                chunks.append(cur)
        with nullcontext(fitpool) as ex:
            futs = [ex.submit(_track_job, ([a for a in ch], {a.i: G[a.i] for a in ch},
                                           {a.i: C_of[a.i] for a in ch}, size, L, W, spec, classes)) for ch in chunks]
            done = {}
            for j, fu in enumerate(futs):
                if cancel is not None and cancel.is_set():
                    return None
                for r in fu.result():
                    done[r.i] = r
                prog(0.62 + 0.33 * (j + 1) / max(1, len(futs)), f"Aligning the court over time: part {j + 1} / {len(futs)}")
        for a in anchors:
            if a.i in reuse:
                R_, f_, c_, inl_, g_, C_, _p = reuse[a.i]
                a.R, a.f, a.cost, a.inliers, a.good, a.C = np.asarray(R_), f_, c_, inl_, g_, np.asarray(C_, float)
                continue
            r = done.get(a.i)
            if r is not None and r.R is not None and hasattr(r, "C"):
                a.R, a.f, a.cost, a.inliers, a.good = r.R, r.f, r.cost, r.inliers, r.good
                a.C = r.C  # type: ignore[attr-defined]
            else:  # not aligned (e.g. too little line evidence in its chunk): not used as an anchor
                a.R, a.good = None, False
                if hasattr(a, "C"):
                    del a.C

        # state for the next streaming round / the final run (before the width fine-tune, which is redone)
        dbg["state"] = {"version": FEATURES_VERSION, "mask": dbg.get("mask"), "finished": dbg.get("finished", {}),
                        "classes": [int(c) for c in classes], "off": off0, "spec": spec, "L": L, "W": W,
                        "C": np.asarray(C_glob, float).tolist(),
                        "fit": (fit.h, np.asarray(fit.C, float).tolist(), fit.length, fit.width, fit.score, fit.spec),
                        "tracked": {a.i: (a.R, a.f, a.cost, a.inliers, a.good, np.asarray(a.C, float).tolist(), a.pan)
                                    for a in anchors if a.R is not None and hasattr(a, "C") and a.pan is not None}}

        # 7. court width: the far sideline is thin - settle it on the refined frames (key stays where seen)
        good = [a for a in anchors if a.good]
        width = W
        if len(good) >= 3:
            smp = good[:: max(1, len(good) // 30)]
            dists = [_dist_maps(a, with_weak(classes)) for a in smp]
            best_w = (np.inf, W)
            for Wc in np.arange(W - 0.5, W + 0.51, 0.25):  # fine-tune only; the mosaic fit measured W
                court.configure(L, Wc, spec)
                dy = np.array([0.0, (Wc - W) / 2, 0.0])
                p_w, _ = model_points_dirs(0.25)
                tot = float(np.mean([anchor_cost(a, a.C + dy, size, p_w, d)[0] for a, d in zip(smp, dists)]))
                if tot < best_w[0] - 1e-6:
                    best_w = (tot, float(Wc))
            width = best_w[1]
            dy = np.array([0.0, (width - W) / 2, 0.0])
            for a in anchors:
                if hasattr(a, "C"):
                    a.C = a.C + dy  # type: ignore[attr-defined]
            C_glob = C_glob + dy
        court.configure(L, width, spec)

        good = [a for a in anchors if a.good]
        res = AutoResult(C=C_glob, length=L, width=width, spec=spec, line_classes=classes)
        for a in anchors:
            if a.R is None or not hasattr(a, "C"):
                continue
            v = ptz_from_R(a.R, a.f)
            res.anchors.append({"t": round(a.t, 4), "i": int(a.i), "seg": int(a.seg), "pan": round(v.pan, 6),
                                "tilt": round(v.tilt, 6), "roll": round(v.roll, 6), "f": round(v.f, 2),
                                "cost": round(a.cost, 3), "inliers": round(a.inliers, 3), "good": bool(a.good),
                                "C": [round(float(x), 4) for x in a.C]})
        res.good_frac = len(good) / max(1, len(anchors))
        cam = f"camera moved in {moved} of {len(windows)} time windows" if moved else "static camera"
        res.message = (f"{spec} markings, court {L:.2f} x {width:.2f} m, {'+'.join(CLASS_NAMES[c] for c in classes)} lines; {cam} at "
                       f"({C_glob[0]:.1f}, {C_glob[1]:.1f}) m, {C_glob[2]:.2f} m high; {len(good)} / {len(anchors)} "
                       f"frames aligned to the lines ({time.time() - t0:.0f} s)")
        prog(1.0, res.message)
        return res

    if state is not None and state.get("off") in hyps[main] and state.get("fit") is not None:
        # early court lock: the court colour, pan offset and court were found on the first minutes of pass 1;
        # only refine the court on all frames (a full re-search costs minutes on a long video)
        prog(0.5, "Refining the court found during the analysis")
        cls_s, off_s = tuple(state["classes"]), state["off"]
        for a in use:
            set_pan(a, off_s)
        mos_s = build_mosaic(use, size, classes=cls_s, **mos_kw)
        h_, C_, L_, W_, sc_, sp_ = state["fit"]
        r = _finish(CourtFit(h_, np.asarray(C_, float), L_, W_, sc_, sp_), off_s, cls_s, [], mos_s, prev=state)
        if r is not None and r.good_frac >= 0.4:
            court.configure(r.length, r.width, r.spec)
            return r
        dbg["state_rejected"] = True
        state = None  # the early court does not fit the whole video: full search

    # 3. which painted lines are the court + court size + camera: floor mosaic of the main segment
    counts = np.bincount(np.concatenate([a.lines[:, 2] for a in use]).astype(int), minlength=N_CLASSES)
    counts = counts[:N_CLASSES]  # strong pixels only for finding the court
    cand = [int(c) for c in np.argsort(-counts) if counts[c] >= 0.04 * counts.sum()][:4]
    if dbg.get("force_classes"):
        cand = list(dbg["force_classes"])
    # The court's colour is the one whose line evidence lies mostly ON the fitted court: clutter colours
    # (wood grain, wall bars, shadows) may have more pixels, but only a small share of them fits a court.
    trials = []
    for off in hyps[main]:
        for a in use:
            set_pan(a, off)
        sets = [(c,) for c in cand]
        if 0 in cand and 1 in cand:
            sets.append((0, 1))  # black lines often come with a bright edge (and vice versa): try both together
        for cs in sets:
            prog(0.52, f"Fitting the court to the floor lines ({'+'.join(CLASS_NAMES[c] for c in cs)} lines)")
            mos = build_mosaic(use, size, classes=cs, **mos_kw)
            prep = _prepare(mos, 3)
            keep: list[CourtFit] = []
            fit = fit_court_to_mosaic(mos, cancel=cancel, keep=keep, pool=fitpool)
            if fit is not None and prep is not None:
                trials.append((fit.score / prep.total, fit, off, cs, keep, mos))
    if not trials:
        dbg["fail"] = "no court fits the floor mosaic"
        return None
    # refine each trial a little, then keep the colour (and pan hypothesis) whose lines explain the most
    # basketball-only geometry: the 3-point arcs, free-throw circles and the centre circle
    scored = []
    for frac, fit, off, cls, keep, mos in trials:
        for a in use:
            set_pan(a, off)
        rf = rescore(mos, _diverse(keep, 1, 4), ("FIBA",), cancel=cancel, pool=fitpool) or fit
        prep2 = _prepare(mos, 2)
        court.configure(rf.length, rf.width, rf.spec)
        if prep2 is None:
            continue
        curve = explained(prep2, rf.h, rf.C[0], rf.C[1], curves_only=True)
        enrich = explained(prep2, rf.h, rf.C[0], rf.C[1], curves_only=True, enrichment=True)
        scored.append((enrich, curve, rf, off, cls, keep, mos))
    court.configure(28.0, 15.0, "FIBA")
    if not scored:
        return None
    # The decisive test: under which colour's court do the most video frames actually line up with their
    # own lines (each frame tries its 4 possible pans)? Enrichment x arc evidence only breaks ties.
    probe = [a for a in use if a.vp is not None][:: max(1, len(use) // 30)][:30]
    ranked = []
    with nullcontext(fitpool) as ex:
        for t in scored:
            rf, cls = t[2], t[4]
            outs = list(ex.map(_pan_job, [(a, rf.C, size, rf.length, rf.width, rf.spec, cls) for a in probe]))
            # mean best match per frame (before fine alignment, so absolute values are modest; what matters is
            # which colour's court fits the frames better)
            vals = [float(np.min(o[1])) for _i, o in outs if o is not None]
            quality = -float(np.mean(vals)) if vals else -50.0
            ranked.append((quality, t))
    ranked.sort(key=lambda r: -r[0])
    dbg["class_trials"] = [(round(r, 2), round(t[0], 2), round(t[1]), t[2].length, t[4], t[3]) for r, t in ranked]
    ranked = [t for _r, t in ranked]
    # quick real test: align 40 sample frames under each candidate (own vanishing-point start, best of 4 pans,
    # refined); the candidate that aligns most of them goes first
    probe2 = [a for a in anchors if a.vp is not None][:: max(1, len(anchors) // 40)][:40]
    rates = []
    with nullcontext(fitpool) as ex:
        for t in ranked[:4]:
            rf, cls = t[2], t[4]
            prog(0.555, f"Testing {'+'.join(CLASS_NAMES[c] for c in cls)} court lines on sample frames")
            oks = list(ex.map(_probe_job, [(a, rf.C, size, rf.length, rf.width, rf.spec, cls) for a in probe2]))
            rates.append(float(np.mean(oks)) if oks else 0.0)
    order = sorted(range(len(rates)), key=lambda k: -rates[k])
    dbg["probe_rates"] = [("+".join(CLASS_NAMES[c] for c in ranked[k][4]), round(rates[k], 2)) for k in range(len(rates))]
    ranked = [ranked[k] for k in order]

    # the colour vote can be close (e.g. bright gaps beside black lines); the real court colour is the one
    # whose fit then aligns the frames - try the runner-up when the first choice does not
    best_res = None
    for _e, _c, fit0, off0, classes, keep, mos in ranked[:2]:
        r = _finish(fit0, off0, classes, keep, mos)
        if cancel is not None and cancel.is_set():
            return None
        if r is not None and (best_res is None or r.good_frac > best_res.good_frac):
            best_res = r
        if best_res is not None and best_res.good_frac >= 0.4:
            break
    if best_res is not None:
        court.configure(best_res.length, best_res.width, best_res.spec)
    else:
        dbg["fail"] = "no frames aligned to the fitted court"
    return best_res


# ------------------------------------------------------------------------------ per-frame resolution
# ------------------------------------------------------------------------- court detector (replay/courtnet.py)
NET_EVERY_S = 0.5  # the detector runs on one sample per this many seconds; the motion chain fills between
NET_MIN_INLIERS = 5
NET_MIN_VISIBLE = 30  # projected court-line samples inside the frame
NET_MIN_ON_LINES = 0.45  # share of them on the frame's own line pixels
NET_PEAK_THR = 0.15  # keypoint heatmap peak threshold (the RANSAC removes wrong peaks)
NET_MAX_GAP_S = 1.0  # frames this far from a located frame (same camera segment) get its H via the motion chain


def calibrate_net(video: Path, data, size: tuple[int, int], progress=None, cancel=None, model: Path | None = None,
                  providers: list[str] | None = None, every_s: float = NET_EVERY_S) -> AutoResult | None:
    """Per-frame calibration from the trained court detector: keypoint heatmaps -> court choice over all frames ->
    per-frame homography (RANSAC on the keypoints). No vanishing points or long motion chains needed (zooming,
    cutting broadcast cameras). Returns None when the model is missing or finds too little."""
    from replay import courtnet

    prog = progress or (lambda x, m: None)
    t0 = time.time()
    try:
        net = courtnet.CourtNet(model, providers)
    except Exception:  # noqa: BLE001 - no model: the classic method runs
        return None
    names = list(court.KEYPOINTS)
    n = len(data.t)
    idx, last = [], -1e9
    for i in range(n):
        if data.t[i] - last >= every_s - 1e-6:
            idx.append(i)
            last = data.t[i]
    f2i = {int(data.frame[i]): i for i in idx}
    pks: dict[int, dict] = {}
    dmaps: dict[int, np.ndarray] = {}  # the frame's own painted-line pixels (classic detector) for the check
    for j, (fno, img) in enumerate(read_frames(video, list(f2i), cancel)):
        _lp, hm = net.predict(img)
        pks[f2i[fno]] = courtnet.peaks(hm, size, NET_PEAK_THR)
        if len(pks[f2i[fno]]) >= NET_MIN_INLIERS:
            lp = line_pixels(img)
            hh = int(round(img.shape[0] * WORK_W / img.shape[1]))
            m = np.zeros((hh, WORK_W), np.uint8)
            m[np.clip(lp[:, 1], 0, hh - 1), np.clip(lp[:, 0], 0, WORK_W - 1)] = 1
            dmaps[f2i[fno]] = cv2.distanceTransform(1 - m, cv2.DIST_L2, 3)
        if j % 50 == 0:
            prog(0.8 * j / max(1, len(idx)), f"Finding the court lines: frame {j} / {len(idx)}")
    if cancel is not None and cancel.is_set():
        return None
    prog(0.8, "Measuring the court")
    smp = [pks[i] for i in sorted(pks)][:: max(1, len(pks) // 150)]
    ch = courtnet.choose_court(smp, names, size)
    if ch is None:
        return None
    spec, L, W, _score = ch
    court.configure(L, W, spec)
    cxy = courtnet.court_points(names)
    lpts, _ = model_points_dirs(0.3)
    frames = []
    for i in sorted(pks):
        r = courtnet.frame_H(pks[i], cxy, size)
        if r is None or len(r[1]) < NET_MIN_INLIERS or i not in dmaps:
            continue
        H = r[0] / r[0][2, 2]
        # the court must lie on the frame's own painted lines (junk peaks can agree on a wrong H)
        q = np.column_stack([lpts, np.ones(len(lpts))]) @ H.T
        if np.median(q[:, 2]) <= 0:
            continue
        uv = q[:, :2] / q[:, 2:3] * (WORK_W / size[0])
        dm = dmaps[i]
        vis = (q[:, 2] > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < dm.shape[1] - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < dm.shape[0] - 1)
        if vis.sum() < NET_MIN_VISIBLE or float(np.mean(dm[uv[vis, 1].astype(int), uv[vis, 0].astype(int)] < 2.5)) < NET_MIN_ON_LINES:
            continue
        frames.append({"t": round(float(data.t[i]), 4), "i": int(i), "H": [round(float(v), 8) for v in H.ravel()]})
    res = AutoResult(C=np.zeros(3), length=L, width=W, spec=spec, line_classes=(), frames=frames)
    res.good_frac = len(frames) / max(1, len(pks))
    res.message = (f"court detector: {spec} markings, court {L:.2f} x {W:.2f} m; {len(frames)} / {len(pks)} frames "
                   f"located ({time.time() - t0:.0f} s)")
    prog(1.0, res.message)
    return res


# ------------------------------------------------- per-frame solver (replay/court_solver.py): broadcast fallback
SOLVER_EVERY_S = 0.5  # one solved sample per this many seconds; the motion chain carries H in between
SOLVER_TRACK_S = 3.0  # prior from the last located sample through the motion chain (same segment) up to this gap
SOLVER_CARRY_S = 1.5  # across a chain break (fast pan, missed motion): the previous H is still a free start
SOLVER_PROBE = 20  # frames per court candidate when choosing the marking standard
SOLVER_COURTS = (("FIBA", 28.0, 15.0), ("NCAA", 28.65, 15.24), ("NBA", 28.65, 15.24))  # TV courts are standard
SOLVER_MIN_EVIDENCE = 300  # evidence pixels (WORK_W) below which a frame shows no court worth a coarse search
SOLVER_MIN_FRAC = 0.05  # located samples / samples for the solver result to count at all
SOLVER_FALLBACK_FRAC = 0.05  # classic good_frac below this = "could not find the court" -> the solver runs
SOLVER_CAM_MIN = 8  # located frames before the camera position is trusted to narrow the coarse search


def calibrate_solver(video: Path, data, size: tuple[int, int], progress=None, cancel=None,
                     every_s: float = SOLVER_EVERY_S, model: Path | None = None, courts=SOLVER_COURTS,
                     debug: dict | None = None, n_threads: int = 2) -> AutoResult | None:
    """Per-frame court registration that needs neither vanishing points nor a long motion chain (TV broadcasts
    that pan, zoom and cut; floors with painted keys, aprons and logos).

    Per sample (every ``every_s``): court-line evidence from court_solver.floor_evidence (floor-region boundary +
    thin lines inside it) -> a court->image H initialised from the last located sample through the pass-1 motion
    chain (zoom included: G is a full homography), from the previous H across a chain break, from the court
    detector's keypoints when a model exists, else from a coarse search over a grid of plausible cameras; refined
    on the distance map and accepted by its line score (court_solver.solve_frame). Once a few frames are located,
    the camera position (TV cameras do not move) narrows the coarse search. The marking standard / size is chosen
    among ``courts`` by the summed score on probe frames. A backward pass gives unlocated samples a second chance
    from the next located one. Output as calibrate_net: frames=[{t, i, H, score, source}] -> resolve_auto."""
    from concurrent.futures import ThreadPoolExecutor

    from replay import court_solver as cs

    prog = progress or (lambda x, m: None)
    dbg = debug if debug is not None else {}
    t0 = time.time()
    n = len(data.t)
    if n == 0:
        return None
    G, seg = chain(data.motion, n)
    idx, last = [], -1e9
    for i in range(n):
        if data.t[i] - last >= every_s - 1e-6:
            idx.append(i)
            last = data.t[i]
    rows = data.rows
    persons = rows[rows[:, 6] == 0] if len(rows) else rows
    if len(persons):
        persons = persons[np.argsort(persons[:, 0], kind="stable")]
    pk = persons[:, 0] if len(persons) else np.zeros(0)

    def boxes_of(i):
        if not len(persons):
            return None
        lo, hi = np.searchsorted(pk, data.k[i], "left"), np.searchsorted(pk, data.k[i], "right")
        return persons[lo:hi, 1:5]

    # optional: the court detector's keypoints as one more initialisation
    net = None
    try:
        from replay import courtnet

        if model is not None or (courtnet.MODEL_DIR / "courtnet.onnx").exists():
            net = courtnet.CourtNet(model, threads=n_threads)
    except Exception:  # noqa: BLE001 - no model / no runtime: evidence only
        net = None

    prog(0.0, "Looking for broadcast overlays")
    probe = [int(data.frame[i]) for i in idx[:: max(1, len(idx) // 40)]][:40]
    static = static_overlay_mask([img for _f, img in read_frames(video, probe, cancel)])
    f2i = {int(data.frame[i]): i for i in idx}
    ev: dict[int, np.ndarray] = {}
    fl4: dict[int, np.ndarray | None] = {}  # floor region at the coarse-search resolution (region consistency)
    peaks: dict[int, dict] = {}
    with ThreadPoolExecutor(max(1, n_threads)) as ex:
        pending = []
        for j, (fno, img) in enumerate(read_frames(video, list(f2i), cancel)):
            i = f2i[fno]
            pending.append((i, ex.submit(cs.floor_evidence, img, boxes_of(i), static)))
            if net is not None:
                try:
                    _lp, hm = net.predict(img)
                    peaks[i] = courtnet.peaks(hm, size, 0.3)
                except Exception:  # noqa: BLE001
                    net = None
            while len(pending) > 2 * n_threads:
                i0, fu = pending.pop(0)
                ev[i0], fl4[i0] = fu.result()[0], fu.result()[1].get("floor4")
            if j % 40 == 0:
                prog(0.45 * j / max(1, len(idx)), f"Finding the floor and its lines: frame {j} / {len(idx)}")
        for i0, fu in pending:
            ev[i0], fl4[i0] = fu.result()[0], fu.result()[1].get("floor4")
    if cancel is not None and cancel.is_set():
        return None
    dbg["evidence_px"] = {i: int(len(e)) for i, e in ev.items()}
    useful = [i for i in idx if len(ev.get(i, ())) >= SOLVER_MIN_EVIDENCE]
    if len(useful) < 3:
        dbg["fail"] = "no frames with floor evidence"
        return None

    def maps(i):
        return cs.evidence_maps(ev[i], size)  # fine, wide, coarse-search map

    # 1. marking standard / size: coarse-solve probe frames under each candidate court
    prog(0.46, "Measuring the court")
    probe_i = [useful[k] for k in np.linspace(0, len(useful) - 1, min(SOLVER_PROBE, len(useful))).astype(int)]
    probe_maps = {i: maps(i) for i in probe_i}
    table = []
    for spec, L, W in courts:
        court.configure(L, W, spec)
        pts, cls = cs.class_samples(0.3)
        hyps = cs.template_hypotheses(size)
        tot, loc = 0.0, 0
        for i in probe_i:
            if cancel is not None and cancel.is_set():
                return None
            d, dc, dc4, e4 = probe_maps[i]
            r = cs.solve_frame(d, size, pts, cls, [], hyps=hyps, dist_c=dc, dist_c4=dc4, floor4=fl4.get(i), ev4=e4)
            if r is not None:
                tot += r["score"]
                loc += 1
        table.append((tot, loc, spec, L, W))
        prog(0.46 + 0.14 * len(table) / len(courts), f"Measuring the court: {spec} {loc} / {len(probe_i)} probe frames")
    dbg["courts"] = [(round(t, 2), l_, s_, L_, W_) for t, l_, s_, L_, W_ in table]
    table.sort(key=lambda r: -r[0])
    if table[0][1] == 0:
        dbg["fail"] = "no probe frame could be located"
        return None
    _tot, _loc, spec, L, W = table[0]
    court.configure(L, W, spec)
    pts, cls = cs.class_samples(0.3)
    hyps_full = cs.template_hypotheses(size)
    cxy = None
    if peaks:
        from replay import courtnet

        cxy = courtnet.court_points(list(court.KEYPOINTS))

    # 2. forward pass
    located: dict[int, dict] = {}
    cams: list[np.ndarray] = []
    order = [i for i in idx if i in ev]
    last_i = None
    counts = {"tracked": 0, "carry": 0, "peaks": 0, "coarse": 0}
    for j, i in enumerate(order):
        if cancel is not None and cancel.is_set():
            return None
        if j % 20 == 0:
            prog(0.6 + 0.3 * j / max(1, len(order)), f"Fitting the court frame by frame: {len(located)} located of {j} / {len(order)}")
        if len(ev[i]) < 10:
            continue
        inits = []
        if last_i is not None:
            Hp = located[last_i]["H"]
            gap = float(data.t[i] - data.t[last_i])
            if seg[i] == seg[last_i] and gap <= SOLVER_TRACK_S:
                inits.append(("tracked", np.linalg.inv(G[i]) @ G[last_i] @ Hp, {}))
            elif gap <= SOLVER_CARRY_S:
                inits.append(("carry", Hp, {}))
        if cxy is not None and len(peaks.get(i, ())) >= 6:
            from replay import courtnet

            r0 = courtnet.frame_H(peaks[i], cxy, size, 0.006, 6, 5)
            if r0 is not None:
                inits.append(("peaks", r0[0] / r0[0][2, 2], {}))
        d, dc, dc4, e4 = maps(i)
        hy = hyps_full if len(ev[i]) >= SOLVER_MIN_EVIDENCE else None
        r = cs.solve_frame(d, size, pts, cls, inits, hyps=hy, dist_c=dc, dist_c4=dc4, floor4=fl4.get(i), ev4=e4)
        if r is None:
            continue
        located[i] = r
        counts[r["source"]] += 1
        last_i = i
        cam = cs.camera_from_H(r["H"], size)
        if cam is not None:
            cams.append(np.asarray(cam[0], float))
    # 3. backward pass: a second chance from the next located sample
    nxt = None
    for i in reversed(order):
        if i in located:
            nxt = i
            continue
        if nxt is None or len(ev[i]) < 10:
            continue
        gap = float(data.t[nxt] - data.t[i])
        Hn = located[nxt]["H"]
        inits = []
        if seg[i] == seg[nxt] and gap <= SOLVER_TRACK_S:
            inits.append(("tracked", np.linalg.inv(G[i]) @ G[nxt] @ Hn, {}))
        elif gap <= SOLVER_CARRY_S:
            inits.append(("carry", Hn, {}))
        if not inits:
            continue
        d = maps(i)[0]
        r = cs.solve_frame(d, size, pts, cls, inits, hyps=None)
        if r is not None:
            located[i] = r
            counts["back"] = counts.get("back", 0) + 1
    dbg["counts"] = counts
    if len(located) < 3:
        dbg["fail"] = "fewer than 3 frames located"
        return None
    frames = [{"t": round(float(data.t[i]), 4), "i": int(i), "H": [round(float(v), 8) for v in located[i]["H"].ravel()],
               "score": round(float(located[i]["score"]), 3), "source": located[i]["source"]} for i in sorted(located)]
    C = np.median(np.array(cams), axis=0) if cams else np.zeros(3)
    res = AutoResult(C=C, length=L, width=W, spec=spec, line_classes=(), frames=frames, max_gap_s=2 * every_s + 0.1)
    res.good_frac = len(frames) / max(1, len(idx))
    sc = float(np.mean([located[i]["score"] for i in located]))
    res.message = (f"frame-by-frame court fit: {spec} markings, court {L:.2f} x {W:.2f} m; {len(frames)} / {len(idx)} "
                   f"frames located (mean line score {sc:.2f}; camera about ({C[0]:.0f}, {C[1]:.0f}) m, {C[2]:.0f} m high;"
                   f" {time.time() - t0:.0f} s)")
    prog(1.0, res.message)
    return res


def calibrate_best(video: Path, data, size: tuple[int, int], progress=None, cancel=None, work: Path | None = None,
                   detector: str = "off") -> AutoResult | None:
    """Classic calibration and/or the trained court detector, per ``detector`` ("off" | "fallback" | "first";
    env REPLAY_COURT_DETECTOR overrides). The result that locates more frames wins when both run.
    When the classic method fails outright (nothing found or < SOLVER_FALLBACK_FRAC of the frames aligned, which
    the GUI / CLI report as "could not find the court"), the per-frame solver (calibrate_solver, the broadcast path)
    runs; clips the classic method calibrates are untouched. ``detector="none"`` disables that fallback (A/B)."""
    detector = os.environ.get("REPLAY_COURT_DETECTOR", detector or "off")
    if detector == "first":
        r = calibrate_net(video, data, size, progress, cancel)
        if r is not None and r.good_frac >= 0.5:
            return r
        c = calibrate(video, data, size, progress=progress, cancel=cancel, work=work)
        return max([x for x in (r, c) if x is not None], key=lambda x: x.good_frac, default=None)
    c = calibrate(video, data, size, progress=progress, cancel=cancel, work=work)
    if detector == "fallback" and (c is None or c.good_frac < 0.4):
        r = calibrate_net(video, data, size, progress, cancel)
        c = max([x for x in (r, c) if x is not None], key=lambda x: x.good_frac, default=None)
    if detector != "none" and (c is None or c.good_frac < SOLVER_FALLBACK_FRAC):
        if cancel is not None and cancel.is_set():
            return c
        s = calibrate_solver(video, data, size, progress, cancel)
        if s is not None and s.good_frac >= SOLVER_MIN_FRAC and (c is None or s.good_frac > c.good_frac):
            return s
    return c


def _resolve_frames(times, G, seg, frames, max_gap_s: float = NET_MAX_GAP_S):
    """Per-sample court->image H from the court detector's located frames: own H, else the nearest located frame
    of the same motion segment within max_gap_s (NET_MAX_GAP_S) carried by the measured camera motion."""
    n = len(times)
    Hs: list[np.ndarray | None] = [None] * n
    dist_s = np.full(n, np.inf)
    loc = {}
    for fr in frames:
        i = int(fr.get("i", -1))
        if not (0 <= i < n) or abs(times[i] - fr["t"]) > 0.25:
            i = int(np.argmin(np.abs(times - fr["t"])))
            if abs(times[i] - fr["t"]) > 0.25:
                continue
        loc[i] = np.asarray(fr["H"], float).reshape(3, 3)
    if not loc:
        return Hs, dist_s
    li = np.array(sorted(loc))
    lt = times[li]
    for i in range(n):
        k = int(np.searchsorted(lt, times[i]))
        best = None
        for j in (k - 1, k):
            if 0 <= j < len(li) and seg[li[j]] == seg[i]:
                d = abs(lt[j] - times[i])
                if d <= max_gap_s and (best is None or d < best[0]):
                    best = (d, int(li[j]))
        if best is None:
            continue
        d, j = best
        Hs[i] = loc[j] if j == i else np.linalg.inv(G[i]) @ G[j] @ loc[j]
        dist_s[i] = d
    return Hs, dist_s


def resolve_auto(times: np.ndarray, motions: list[np.ndarray | None], auto: dict, size: tuple[int, int],
                 extra: list[tuple[float, np.ndarray, float]] | None = None, max_gap_s: float = 120.0):
    """Per-sample court->image homographies (or None) from automatic anchors + the motion chain.

    For a rotating (and zooming) camera, G_i (frame i -> first frame of its segment) =
    K_ref R_ref R_i^T K_i^-1, so R_i = A_i R_ref with A_i = K_i^-1 G_i^-1 K_ref. Every good anchor gives
    an estimate of R_ref; those drift slowly (motion-chain drift) and are interpolated in time (slerp),
    with outlier anchors rejected. Zoom (focal length) is interpolated between anchors.
    Each segment uses its own camera position (anchors' "C"; TV broadcasts cut between cameras).
    ``extra``: additional anchors (t, R, f), e.g. from manual keyframes - trusted over automatic ones.
    """
    n = len(times)
    C_glob = np.asarray(auto["C"], float)
    G, seg = chain(motions, n)
    if auto.get("frames"):
        return _resolve_frames(times, G, seg, auto["frames"], float(auto.get("max_gap_s") or NET_MAX_GAP_S))
    anc = [a for a in auto.get("anchors", []) if a.get("good")]
    Hs: list[np.ndarray | None] = [None] * n
    dist_s = np.full(n, np.inf)
    if not anc and not extra:
        return Hs, dist_s

    items = []  # (t, i, R_abs, f, C or None, weight)
    for a in anc:
        i = int(a.get("i", -1))
        if not (0 <= i < n) or abs(times[i] - a["t"]) > 0.25:
            i = int(np.argmin(np.abs(times - a["t"])))
            if abs(times[i] - a["t"]) > 0.25:
                continue
        Ca = np.asarray(a["C"], float) if "C" in a else None
        items.append((float(times[i]), i, rotation(a["pan"], a["tilt"], a["roll"]), float(a["f"]), Ca, 1.0))
    for t, R, f in extra or []:
        i = int(np.argmin(np.abs(times - t)))
        if abs(times[i] - t) <= 0.25:
            items.append((float(times[i]), i, R, float(f), None, 3.0))

    for s in np.unique(seg):
        mem = np.where(seg == s)[0]
        its = sorted([it for it in items if seg[it[1]] == s], key=lambda it: it[0])
        if not its:
            continue
        ct = np.array([it[0] for it in its if it[4] is not None])
        cv_ = np.array([it[4] for it in its if it[4] is not None]).reshape(-1, 3)

        def C_at(t, ct=ct, cv_=cv_):
            if not len(ct):
                return C_glob
            if len(ct) == 1:
                return cv_[0]
            return np.array([np.interp(t, ct, cv_[:, k]) for k in range(3)])
        tt = np.array([it[0] for it in its])
        ff = np.array([it[3] for it in its])

        def f_at(t, tt=tt, ff=ff):
            return float(np.interp(t, tt, ff)) if len(tt) > 1 else float(ff[0])

        K_ref = K_of(f_at(times[mem[0]]), size)

        def A(i, K_ref=K_ref, f_at=f_at):
            M = np.linalg.inv(K_of(f_at(times[i]), size)) @ np.linalg.inv(G[i]) @ K_ref
            U, _, Vt = np.linalg.svd(M)
            R = U @ Vt
            return -R if np.linalg.det(R) < 0 else R

        rv = Rotation.from_matrix(np.stack([A(i).T @ R for _t, i, R, _f, _c, _w in its]))
        wts = np.array([it[5] for it in its])
        if len(its) >= 5:  # outliers: anchors disagreeing with their neighbours by > 1.5 degrees
            rel = (rv[0].inv() * rv).as_rotvec()
            # "the court cannot teleport": each frame's camera (after removing the measured motion) must agree
            # with the median of its neighbours over ~15 frames; runs of up to ~7 wrong frames are rejected
            sm = np.array([np.median(rel[max(0, j - 7):j + 8], axis=0) for j in range(len(rel))])
            keep = (np.linalg.norm(rel - sm, axis=1) <= np.radians(2.5)) | (wts > 1)
            if keep.sum() >= 2:
                sel = np.where(keep)[0]
                tt, rv = tt[sel], rv[sel]
        tt_u, ui = np.unique(tt, return_index=True)  # Slerp needs strictly increasing times
        rv_u = rv[ui]
        sl = Slerp(tt_u, rv_u) if len(tt_u) >= 2 else None
        ok_i, ds = [], []
        for i in mem:  # which frames get a calibration; then all of them are evaluated at once
            d = float(np.min(np.abs(tt_u - times[i])))
            if d > max_gap_s:
                continue
            # never trust the motion chain far from a verified anchor: a wrong court is worse than none
            if times[i] < tt_u[0] - MAX_EXTRAP_S or times[i] > tt_u[-1] + MAX_EXTRAP_S:
                continue
            k = int(np.searchsorted(tt_u, times[i]))
            if 0 < k < len(tt_u) and tt_u[k] - tt_u[k - 1] > MAX_BRIDGE_S and d > MAX_EXTRAP_S:
                continue
            ok_i.append(i)
            ds.append(d)
        if not ok_i:
            continue
        ti = times[ok_i]
        Rr = (np.repeat(rv_u[0].as_matrix()[None], len(ti), 0) if sl is None
              else sl(np.clip(ti, tt_u[0], tt_u[-1])).as_matrix())
        Cs = (np.repeat(np.asarray(C_glob, float)[None], len(ti), 0) if not len(ct) else
              np.repeat(cv_[:1], len(ti), 0) if len(ct) == 1 else
              np.stack([np.interp(ti, ct, cv_[:, c]) for c in range(3)], 1))
        tt0, ff0 = f_at.__defaults__  # anchors' focal lengths before the outlier filter (what f_at uses)
        fs = np.interp(ti, tt0, ff0) if len(tt0) > 1 else np.full(len(ti), float(ff0[0]))
        # A(i) for all frames at once (batched linear algebra)
        Ks = np.zeros((len(ti), 3, 3))
        Ks[:, 0, 0] = Ks[:, 1, 1] = fs
        Ks[:, 0, 2], Ks[:, 1, 2], Ks[:, 2, 2] = size[0] / 2, size[1] / 2, 1.0
        M = np.linalg.inv(Ks) @ np.linalg.inv(np.stack([G[i] for i in ok_i])) @ K_ref
        U, _, Vt = np.linalg.svd(M)
        Ai = U @ Vt
        Ai[np.linalg.det(Ai) < 0] *= -1
        Rf = Ai @ Rr  # world -> camera per frame
        P = np.concatenate([Rf[:, :, :2], -(Rf @ Cs[:, :, None])], axis=2)
        Hb = Ks @ P
        for j, i in enumerate(ok_i):
            Hs[i] = Hb[j]
            dist_s[i] = ds[j]
    return Hs, dist_s
