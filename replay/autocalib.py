"""Automatic court calibration by matching the FIBA court model to the lines on the floor.

1. Line pixels: basketball lines are thin, dark and unsaturated on a light floor. A black-hat
   filter finds thin dark strokes; saturation/value thresholds drop the coloured lines of other
   sports (green/red/blue) and players are masked out using the detection boxes.
2. A distance transform turns the mask into a smooth cost: every pixel knows how far the nearest
   line pixel is.
3. The court model (all FIBA lines, sampled every 20 cm) is projected through a physical camera
   (camera_model.PTZ) and the camera is moved until the projected lines sit on the detected ones
   (chamfer matching; truncated so clutter can't dominate).
4. Accepted only when enough of the projected model lies on line pixels.

Uses: snap a rough calibration onto the real lines; add automatic keyframes along the video;
or find the camera from scratch (global search over camera positions).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


def minimize(*a, **k):  # scipy imported on first use: importing it costs seconds at GUI start-up
    from scipy.optimize import minimize as _mn

    return _mn(*a, **k)

from replay import court
from replay.camera_model import PTZ, court_to_image, project_points

WORK_W = 640
TRUNC = 12.0  # px (at WORK_W): distances beyond this count the same - clutter / missing lines
INLIER_PX = 2.5


def model_points(step: float = 0.2) -> np.ndarray:
    """Points along every court line (N, 3), z = 0."""
    pts = []
    for line in court.court_polylines(n_arc=96):
        for a, b in zip(line[:-1], line[1:]):
            n = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
            u = np.linspace(0, 1, n, endpoint=False)[:, None]
            pts.append(a + (b - a) * u)
    p = np.vstack(pts)
    return np.column_stack([p, np.zeros(len(p))])


MODEL = model_points()


@dataclass
class LineImage:
    dist: np.ndarray  # distance (px, work scale) to the nearest line pixel, truncated
    mask: np.ndarray
    scale: float  # work / full resolution
    size: tuple[int, int]  # full resolution (w, h)

    @property
    def work_size(self) -> tuple[int, int]:
        return self.dist.shape[1], self.dist.shape[0]


def line_image(img_bgr: np.ndarray, person_boxes: np.ndarray | None = None,
               overlay_regions: list[list[float]] | None = None) -> LineImage:
    h, w = img_bgr.shape[:2]
    s = WORK_W / float(w)
    small = cv2.resize(img_bgr, (WORK_W, int(round(h * s))), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    floor_v = np.median(hsv[..., 2])
    mask = (bh > 22) & (hsv[..., 1] < 75) & (hsv[..., 2] < floor_v - 25)
    m = mask.astype(np.uint8)
    hh, ww = m.shape
    for x1, y1, x2, y2 in (np.asarray(person_boxes).reshape(-1, 4) * s if person_boxes is not None else []):
        pad = 0.08 * (x2 - x1)
        m[max(0, int(y1 - pad)):int(y2 + pad), max(0, int(x1 - pad)):int(x2 + pad)] = 0
    for x0, y0, x1, y1 in overlay_regions or []:
        m[int(y0 * hh):int(np.ceil(y1 * hh)), int(x0 * ww):int(np.ceil(x1 * ww))] = 0
    dist = cv2.distanceTransform((1 - m).astype(np.uint8), cv2.DIST_L2, 3)
    return LineImage(np.minimum(dist, TRUNC).astype(np.float32), m, s, (w, h))


def _lookup(dist: np.ndarray, px: np.ndarray) -> np.ndarray:
    """Bilinear lookup of the distance map; NaN outside."""
    h, w = dist.shape
    x, y = px[:, 0], px[:, 1]
    ok = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (y >= 0) & (x < w - 1) & (y < h - 1)
    out = np.full(len(px), np.nan, np.float32)
    xi, yi = x[ok].astype(int), y[ok].astype(int)
    fx, fy = x[ok] - xi, y[ok] - yi
    d = dist
    out[ok] = (d[yi, xi] * (1 - fx) * (1 - fy) + d[yi, xi + 1] * fx * (1 - fy) + d[yi + 1, xi] * (1 - fx) * fy
               + d[yi + 1, xi + 1] * fx * fy)
    return out


@dataclass
class MatchScore:
    cost: float  # mean truncated distance of visible model points (px, work scale)
    visible: int
    inlier_frac: float


def score(C: np.ndarray, ptz: PTZ, li: LineImage, min_visible: int = 60) -> MatchScore:
    full = project_points(C, ptz, li.size, MODEL)
    d = _lookup(li.dist, full * li.scale)
    vis = np.isfinite(d)
    n = int(vis.sum())
    if n < min_visible:
        return MatchScore(TRUNC * 2, n, 0.0)
    dv = d[vis]
    return MatchScore(float(np.mean(dv)), n, float(np.mean(dv < INLIER_PX)))


def refine_view(C: np.ndarray, ptz: PTZ, li: LineImage, iters: int = 2) -> tuple[PTZ, MatchScore]:
    """Move the view (C fixed) until the court lines sit on the detected lines. Coarse-to-fine."""
    best = ptz
    for blur in [7, 3, 0][-(iters + 1):]:
        dist = li.dist if blur == 0 else cv2.GaussianBlur(li.dist, (0, 0), blur)
        lb = LineImage(dist, li.mask, li.scale, li.size)

        def f(v):
            return score(C, PTZ.from_vec(v), lb).cost

        x0 = best.vec()
        steps = np.array([0.02, 0.02, 0.01, 0.03 * x0[3]])
        r = minimize(lambda z: f(x0 + z * steps), np.zeros(4), method="Powell",
                     options={"xtol": 0.05, "ftol": 1e-3, "maxfev": 400})
        best = PTZ.from_vec(x0 + r.x * steps)
    return best, score(C, best, li)


def good_match(m: MatchScore) -> bool:
    return m.visible >= 150 and m.inlier_frac >= 0.45 and m.cost <= 4.0


# ------------------------------------------------------------------------------------- oriented matching
def model_points_dirs(step: float = 0.35) -> tuple[np.ndarray, np.ndarray]:
    """Court line points (N, 3) and their unit tangent directions in the court plane (N, 3)."""
    pts, dirs = [], []
    for line in court.court_polylines(n_arc=96):
        for a, b in zip(line[:-1], line[1:]):
            seg = b - a
            L = np.linalg.norm(seg)
            if L < 1e-6:
                continue
            n = max(1, int(np.ceil(L / step)))
            u = np.linspace(0, 1, n, endpoint=False)[:, None]
            pts.append(a + seg * u)
            dirs.append(np.repeat((seg / L)[None], n, axis=0))
    p, d = np.vstack(pts), np.vstack(dirs)
    z = np.zeros((len(p), 1))
    return np.hstack([p, z]), np.hstack([d, z])


SEARCH_PTS, SEARCH_DIRS = model_points_dirs()


@dataclass
class OrientedLines:
    dist: np.ndarray  # distance to nearest line pixel (work px), truncated
    angle: np.ndarray  # line direction (radians, mod pi) of that nearest line pixel
    scale: float
    size: tuple[int, int]
    top_ignore: float  # rows above this fraction of the height are ignored (walls)


def oriented_lines(img_bgr: np.ndarray, person_boxes=None, overlay_regions=None, top_ignore: float = 0.3) -> OrientedLines:
    li = line_image(img_bgr, person_boxes, overlay_regions)
    m = li.mask.copy()
    m[: int(top_ignore * m.shape[0])] = 0
    dist, labels = cv2.distanceTransformWithLabels((1 - m).astype(np.uint8), cv2.DIST_L2, 3,
                                                   labelType=cv2.DIST_LABEL_PIXEL)
    # direction of the line at every line pixel: perpendicular to the intensity gradient of the blurred mask
    mb = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 1.5)
    gx = cv2.Sobel(mb, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(mb, cv2.CV_32F, 0, 1, ksize=3)
    # structure tensor smoothing for a stable orientation on thin lines
    jxx = cv2.GaussianBlur(gx * gx, (0, 0), 2)
    jyy = cv2.GaussianBlur(gy * gy, (0, 0), 2)
    jxy = cv2.GaussianBlur(gx * gy, (0, 0), 2)
    grad_ang = 0.5 * np.arctan2(2 * jxy, jxx - jyy)
    line_ang = (grad_ang + np.pi / 2) % np.pi
    ys, xs = np.nonzero(m)
    lut = np.zeros(labels.max() + 1, np.float32)
    lut[labels[ys, xs]] = line_ang[ys, xs]
    ang = lut[labels]
    return OrientedLines(np.minimum(dist, TRUNC).astype(np.float32), ang.astype(np.float32), li.scale, li.size,
                         top_ignore)


def oriented_cost(H_c2i: np.ndarray, ol: OrientedLines, angle_weight: float = 0.6,
                  pts=None, dirs=None) -> tuple[float, int, float]:
    """(cost, visible points, inlier fraction) of the court model under court->image homography H_c2i."""
    pts = SEARCH_PTS if pts is None else pts
    dirs = SEARCH_DIRS if dirs is None else dirs
    S = np.diag([ol.scale, ol.scale, 1.0]) @ H_c2i
    a = np.column_stack([pts[:, 0], pts[:, 1], np.ones(len(pts))]) @ S.T
    b = np.column_stack([pts[:, 0] + 0.2 * dirs[:, 0], pts[:, 1] + 0.2 * dirs[:, 1], np.ones(len(pts))]) @ S.T
    ok = (a[:, 2] > 1e-6) & (b[:, 2] > 1e-6)
    pa = a[:, :2] / np.where(ok, a[:, 2], 1)[:, None]
    pb = b[:, :2] / np.where(ok, b[:, 2], 1)[:, None]
    h, w = ol.dist.shape
    x, y = pa[:, 0], pa[:, 1]
    ok &= (x >= 0) & (x < w - 1) & (y >= ol.top_ignore * h) & (y < h - 1)
    n = int(ok.sum())
    if n < 40:
        return TRUNC * 2, n, 0.0
    xi, yi = x[ok].astype(int), y[ok].astype(int)
    d = ol.dist[yi, xi]
    seg_ang = np.arctan2(pb[ok, 1] - pa[ok, 1], pb[ok, 0] - pa[ok, 0]) % np.pi
    dang = np.abs(seg_ang - ol.angle[yi, xi])
    dang = np.minimum(dang, np.pi - dang)
    near = d < TRUNC
    c = d + angle_weight * TRUNC * np.sin(dang) * near
    return float(np.mean(np.minimum(c, TRUNC))), n, float(np.mean((d < INLIER_PX) & (np.sin(dang) < 0.3)))


# ------------------------------------------------------------------------- image -> model ("explained lines")
_CRES = 0.05  # metres per court-map pixel
_CPAD = 4.0


def _court_distance_map() -> np.ndarray:
    w = int((court.LENGTH + 2 * _CPAD) / _CRES)
    h = int((court.WIDTH + 2 * _CPAD) / _CRES)
    canvas = np.full((h, w), 255, np.uint8)
    for line in court.court_polylines(n_arc=128):
        pts = np.int32(np.round((line + _CPAD) / _CRES))
        cv2.polylines(canvas, [pts.reshape(-1, 1, 2)], False, 0, 1)
    return cv2.distanceTransform(canvas, cv2.DIST_L2, 3) * _CRES  # metres


COURT_DIST = _court_distance_map()


def line_samples(ol: OrientedLines, n: int = 800, seed: int = 0) -> np.ndarray:
    """Up to n detected line pixels (full-resolution coordinates) below the ignored wall band."""
    ys, xs = np.nonzero(ol.dist == 0)
    keep = ys >= ol.top_ignore * ol.dist.shape[0]
    xs, ys = xs[keep], ys[keep]
    if not len(xs):
        return np.zeros((0, 2))
    sel = np.random.default_rng(seed).choice(len(xs), min(n, len(xs)), replace=False)
    return np.column_stack([xs[sel], ys[sel]]) / ol.scale


def explain_cost(H_c2i: np.ndarray, samples_px: np.ndarray, trunc_px: float = TRUNC, scale: float = 0.5) -> float:
    """Mean distance (work px, truncated) from detected line pixels to the nearest court line.

    Each pixel is mapped onto the court plane; the distance to the nearest court line (metres, from a
    precomputed map) is converted back to pixels with the local image scale.
    """
    if not len(samples_px):
        return trunc_px
    Hinv = np.linalg.inv(H_c2i)
    q = np.column_stack([samples_px, np.ones(len(samples_px))]) @ Hinv.T
    ok = q[:, 2] > 1e-9
    xy = q[:, :2] / np.where(ok, q[:, 2], 1)[:, None]
    cx = np.round((xy[:, 0] + _CPAD) / _CRES).astype(int)
    cy = np.round((xy[:, 1] + _CPAD) / _CRES).astype(int)
    h, w = COURT_DIST.shape
    ok &= (cx >= 0) & (cx < w) & (cy >= 0) & (cy < h)
    d_m = np.full(len(samples_px), np.inf)
    d_m[ok] = COURT_DIST[cy[ok], cx[ok]]
    # local metres -> pixels: project a 10 cm step back into the image
    q2 = np.column_stack([xy[:, 0] + 0.1, xy[:, 1], np.ones(len(xy))]) @ H_c2i.T
    p2 = q2[:, :2] / np.where(np.abs(q2[:, 2]) > 1e-9, q2[:, 2], 1)[:, None]
    px_per_m = np.linalg.norm(p2 - samples_px, axis=1) * 10 * scale
    d_px = np.where(ok & np.isfinite(px_per_m), d_m * px_per_m, trunc_px)
    return float(np.mean(np.minimum(d_px, trunc_px)))


# ---------------------------------------------------------------------------- limited (safe) snapping
@dataclass
class SnapResult:
    ptz: PTZ
    accepted: bool
    before: float  # combined cost before / after (work px)
    after: float
    inliers: float
    reason: str


def _court_region_samples(ol: OrientedLines, C, ptz: PTZ, margin_m: float = 0.8) -> np.ndarray:
    """Detected line pixels that fall on the court (plus margin) under the starting camera: walls, benches
    and the stands can then never attract the fit."""
    samp = line_samples(ol, n=1500)
    if not len(samp):
        return samp
    H = court_to_image(C, ptz, ol.size)
    q = np.column_stack([samp, np.ones(len(samp))]) @ np.linalg.inv(H).T
    ok = q[:, 2] > 1e-9
    xy = q[:, :2] / np.where(ok, q[:, 2], 1)[:, None]
    ok &= (xy[:, 0] > -margin_m) & (xy[:, 0] < court.LENGTH + margin_m)
    ok &= (xy[:, 1] > -margin_m) & (xy[:, 1] < court.WIDTH + margin_m)
    return samp[ok]


def snap_view(C: np.ndarray, ptz: PTZ, ol: OrientedLines, max_deg: float = 4.0, max_zoom: float = 0.12,
              min_inliers: float = 0.22) -> SnapResult:
    """Refine a view so the court lines sit on the detected floor lines - only small corrections allowed.

    Combined cost = court lines -> detected lines (with direction) + detected court-area lines -> court lines,
    plus a penalty for moving far from the start. Accepted only if clearly better and well supported.
    """
    samples = _court_region_samples(ol, C, ptz)
    x0 = ptz.vec()

    def total(z):
        v = PTZ.from_vec([x0[0] + z[0], x0[1] + z[1], x0[2] + z[2], x0[3] * np.exp(z[3])])
        H = court_to_image(C, v, ol.size)
        c1, n, inl = oriented_cost(H, ol)
        c2 = explain_cost(H, samples) if len(samples) >= 30 else c1
        lim = np.radians(max_deg)
        reg = (z[0] / lim) ** 2 + (z[1] / lim) ** 2 + (z[2] / (lim / 2)) ** 2 + (z[3] / max_zoom) ** 2
        return 0.5 * c1 + 0.5 * c2 + 0.5 * reg, inl

    before, inl0 = total(np.zeros(4))
    r = minimize(lambda z: total(z)[0], np.zeros(4), method="Powell",
                 options={"maxfev": 800, "xtol": 1e-3, "ftol": 1e-4})
    z = r.x
    after, inl1 = total(z)
    v = PTZ.from_vec([x0[0] + z[0], x0[1] + z[1], x0[2] + z[2], x0[3] * np.exp(z[3])])
    lim = np.radians(max_deg)
    if abs(z[0]) > lim * 1.5 or abs(z[1]) > lim * 1.5 or abs(z[3]) > max_zoom * 1.5:
        return SnapResult(ptz, False, before, after, inl1, "correction too large - not trusted")
    if inl1 < min_inliers:
        return SnapResult(ptz, False, before, after, inl1, "too few court lines visible / matched")
    if after > 0.93 * before and inl1 < inl0 + 0.05:
        return SnapResult(ptz, False, before, after, inl1, "no clear improvement")
    return SnapResult(v, True, before, after, inl1, "ok")


def keypoints_from_view(C: np.ndarray, ptz: PTZ, size: tuple[int, int], margin: float = 0.03) -> dict:
    """Court landmarks visible in the frame under this camera, as keyframe points (name -> pixel)."""
    names = list(court.KEYPOINTS)
    xyz = np.array([[*court.KEYPOINTS[n], 0.0] for n in names])
    px = project_points(C, ptz, size, xyz)
    w, h = size
    out = {}
    for n, (u, v) in zip(names, px):
        if np.isfinite(u) and margin * w < u < (1 - margin) * w and margin * h < v < (1 - margin) * h:
            out[n] = (round(float(u), 1), round(float(v), 1))
    return out


def swap_left_right(points: dict) -> dict:
    """Relabel clicked points as if the two baskets had been mixed up.

    Seen from a camera on one sideline, mixing up the baskets is a 180 degree rotation of the court:
    left <-> right AND near <-> far (otherwise the camera would have to be on the opposite sideline).
    """
    def rot(name: str) -> str:
        if name.startswith("L_"):
            name = "R_" + name[2:]
        elif name.startswith("R_"):
            name = "L_" + name[2:]
        if name.endswith("_near"):
            return name[:-5] + "_far"
        if name.endswith("_far"):
            return name[:-4] + "_near"
        return name

    return {rot(n): p for n, p in points.items()}
