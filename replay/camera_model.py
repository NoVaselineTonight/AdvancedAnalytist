"""Physical pan-tilt-zoom camera model for court calibration.

A free 8-parameter homography can bend the court into shapes no camera can
produce (shears, impossible perspective) - with few or slightly misplaced clicks
it happily does. A real broadcast/phone camera on a tripod has:

    one fixed position C = (x, y, z) in court metres     (shared by the whole video)
    per frame: pan, tilt, roll, focal length              (4 numbers)

with square pixels and the principal point at the image centre. Every court
calibration is fitted *within* this model, so lines can only be viewed from
somewhere, never distorted. Three well-spread points already constrain a frame,
and all keyframes jointly constrain the camera position.

Coordinates: court x along the sideline (0..28), y along the baseline (0..15),
z up. Camera axes: x right, y down, z forward.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def least_squares(*a, **k):  # scipy imported on first use: importing it costs seconds at GUI start-up
    from scipy.optimize import least_squares as _ls

    return _ls(*a, **k)


@dataclass
class PTZ:
    pan: float  # radians, direction of view in the court plane (0 = +x)
    tilt: float  # radians, >0 looks down
    roll: float  # radians
    f: float  # focal length in pixels

    def vec(self) -> np.ndarray:
        return np.array([self.pan, self.tilt, self.roll, self.f])

    @staticmethod
    def from_vec(v) -> "PTZ":
        return PTZ(float(v[0]), float(v[1]), float(v[2]), float(v[3]))


def rotation(pan: float, tilt: float, roll: float) -> np.ndarray:
    """World -> camera rotation (rows = camera right, down, forward in world coordinates)."""
    fwd = np.array([np.cos(tilt) * np.cos(pan), np.cos(tilt) * np.sin(pan), -np.sin(tilt)])
    right = np.array([np.sin(pan), -np.cos(pan), 0.0])
    down = np.cross(fwd, right)
    cr, sr = np.cos(roll), np.sin(roll)
    return np.stack([cr * right + sr * down, -sr * right + cr * down, fwd])


def court_to_image(C: np.ndarray, ptz: PTZ, size: tuple[int, int]) -> np.ndarray:
    """3x3 homography court(x, y, 1) -> image pixels."""
    w, h = size
    K = np.array([[ptz.f, 0, w / 2], [0, ptz.f, h / 2], [0, 0, 1]])
    R = rotation(ptz.pan, ptz.tilt, ptz.roll)
    return K @ np.column_stack([R[:, 0], R[:, 1], -R @ C])


def project_points(C, ptz: PTZ, size, xyz: np.ndarray) -> np.ndarray:
    """Project 3D court points (N, 3) -> pixels (N, 2); points behind the camera are NaN."""
    w, h = size
    R = rotation(ptz.pan, ptz.tilt, ptz.roll)
    cam = (np.asarray(xyz, float) - C) @ R.T
    out = np.full((len(cam), 2), np.nan)
    ok = cam[:, 2] > 1e-3
    out[ok, 0] = w / 2 + ptz.f * cam[ok, 0] / cam[ok, 2]
    out[ok, 1] = h / 2 + ptz.f * cam[ok, 1] / cam[ok, 2]
    return out


def look_at(C: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    d = np.asarray(target, float) - C
    pan = float(np.arctan2(d[1], d[0]))
    tilt = float(np.arctan2(-d[2], np.hypot(d[0], d[1])))
    return pan, tilt


# ------------------------------------------------------------------------------------------- joint fit
@dataclass
class CameraFit:
    C: np.ndarray  # camera position, metres
    views: list[PTZ]  # one per observation group (keyframe or reference frame)
    rms_px: float
    per_point_px: list[np.ndarray]  # residual magnitude per point, per group


def _apply(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    q = np.hstack([pts, np.ones((len(pts), 1))]) @ H.T
    return q[:, :2] / q[:, 2:3]


def fit_camera(groups: list[tuple[np.ndarray, np.ndarray]], size: tuple[int, int],
               C_fixed: np.ndarray | None = None, robust_px: float = 6.0,
               init_views: list[PTZ] | None = None) -> CameraFit | None:
    """Fit one camera position + one PTZ per group.

    groups: [(court_xy (N,2), image_px (N,2))] - each group is one view (e.g. one keyframe, or all keyframes
    of a motion segment mapped into that segment's reference frame).
    """
    groups = [(np.asarray(c, float), np.asarray(p, float)) for c, p in groups if len(c)]
    n_pts = sum(len(c) for c, _ in groups)
    if not groups or n_pts < 3:
        return None
    w, h = size
    diag = float(np.hypot(w, h))

    def unpack(v):
        if C_fixed is None:
            C = v[:3]
            rest = v[3:]
        else:
            C, rest = C_fixed, v
        views = [PTZ.from_vec(rest[4 * g:4 * g + 4]) for g in range(len(groups))]
        return C, views

    def residuals(v):
        C, views = unpack(v)
        res = []
        for (court, img), ptz in zip(groups, views):
            xyz = np.column_stack([court, np.zeros(len(court))])
            proj = project_points(C, ptz, size, xyz)
            r = (proj - img) / 1.0
            r[~np.isfinite(r)] = diag  # behind the camera: large penalty
            res.append(r.ravel())
            # weak priors keep the camera physical when points are few
            res.append([np.degrees(ptz.roll) / 3.0, np.log(max(ptz.f, 1.0) / diag) * 2.0])
        if C_fixed is None:
            res.append([(C[2] - 5.0) / 6.0])  # camera height: a few metres, very weak
        return np.concatenate([np.ravel(r) for r in res])

    # multi-start over plausible camera positions around the court
    starts = []
    if C_fixed is not None:
        starts.append(np.asarray(C_fixed, float))
    else:
        for z in (3.0, 7.0):
            for x, y in [(14, -7), (7, -7), (21, -7), (14, -15), (14, 22), (-6, 7.5), (34, 7.5), (3, -6), (25, -6),
                         (14, 3), (14, -2)]:
                starts.append(np.array([x, y, z], float))
    best = None
    for C0 in starts:
        v0 = [] if C_fixed is not None else list(C0)
        for g_idx, (court, _img) in enumerate(groups):
            if init_views is not None and C_fixed is not None:
                v0 += list(init_views[g_idx].vec())
                continue
            pan, tilt = look_at(C0, np.array([*court.mean(axis=0), 0.0]))
            v0 += [pan, tilt, 0.0, 0.9 * w]
        lo = ([-60, -60, 0.3] if C_fixed is None else []) + [-np.inf, -0.2, -0.35, 0.15 * w] * len(groups)
        hi = ([90, 75, 40] if C_fixed is None else []) + [np.inf, 1.4, 0.35, 12.0 * w] * len(groups)
        v0 = np.clip(v0, np.array(lo) + 1e-6, np.array(hi) - 1e-6)
        try:
            r = least_squares(residuals, v0, bounds=(lo, hi), loss="soft_l1", f_scale=robust_px, max_nfev=400)
        except ValueError:
            continue
        if best is None or r.cost < best.cost:
            best = r
    if best is None:
        return None
    C, views = unpack(best.x)
    per_point, sq = [], []
    for (court, img), ptz in zip(groups, views):
        proj = project_points(C, ptz, size, np.column_stack([court, np.zeros(len(court))]))
        e = np.linalg.norm(proj - img, axis=1)
        per_point.append(e)
        sq.append(e ** 2)
    rms = float(np.sqrt(np.nanmean(np.concatenate(sq))))
    return CameraFit(np.array(C, float), views, rms, per_point)


def init_from_homography(C: np.ndarray, H_c2i: np.ndarray, size: tuple[int, int], f_guess: float) -> PTZ | None:
    """Rough view looking where H_c2i says the image centre lands on the floor."""
    w, h = size
    q = np.linalg.inv(H_c2i) @ np.array([w / 2, h / 2, 1.0])
    if abs(q[2]) < 1e-12:
        return None
    x, y = q[0] / q[2], q[1] / q[2]
    if not (np.isfinite(x) and np.isfinite(y)) or abs(x) > 200 or abs(y) > 200:
        return None
    pan, tilt = look_at(C, np.array([x, y, 0.0]))
    # focal: match the image distance of a 1 m step at the centre point
    p0 = _apply(H_c2i, np.array([[x, y]]))[0]
    p1 = _apply(H_c2i, np.array([[x + 1.0, y]]))[0]
    dist = float(np.linalg.norm(np.array([x, y, 0.0]) - C))
    f = f_guess
    if np.isfinite(p1).all() and np.linalg.norm(p1 - p0) > 1:
        f = float(np.clip(np.linalg.norm(p1 - p0) * dist, 0.2 * w, 10 * w))
    return PTZ(pan, tilt, 0.0, f)


def fit_ptz_to_homography(C: np.ndarray, H_c2i: np.ndarray, size: tuple[int, int], init: PTZ) -> PTZ | None:
    """Nearest physically valid view (C fixed) to an arbitrary court->image homography.

    Used to keep frames reached through the camera-motion chain on the camera model: whatever small
    non-rotational errors the chain accumulates are projected away. Tries the given starting view and one
    derived from where the image centre lands, and keeps the better fit.
    """
    w, h = size
    gx, gy = np.meshgrid(np.linspace(0, 28, 15), np.linspace(0, 15, 9))
    grid = np.column_stack([gx.ravel(), gy.ravel()])
    target = _apply(H_c2i, grid)
    q = np.hstack([grid, np.ones((len(grid), 1))]) @ H_c2i.T
    vis = (q[:, 2] > 0) & np.isfinite(target).all(axis=1)
    vis &= (target[:, 0] > -0.5 * w) & (target[:, 0] < 1.5 * w) & (target[:, 1] > -0.5 * h) & (target[:, 1] < 1.5 * h)
    if vis.sum() < 4:
        return None
    grid, target = grid[vis], target[vis]
    xyz = np.column_stack([grid, np.zeros(len(grid))])

    def residuals(v):
        p = project_points(C, PTZ.from_vec(v), size, xyz) - target
        p[~np.isfinite(p)] = 1e4
        return p.ravel()

    best = None
    for start in (init, init_from_homography(C, H_c2i, size, init.f)):
        if start is None:
            continue
        try:
            r = least_squares(residuals, start.vec(), loss="soft_l1", f_scale=10.0, max_nfev=80,
                              x_scale=[0.05, 0.05, 0.02, max(50.0, start.f * 0.05)])
        except ValueError:
            continue
        if best is None or r.cost < best.cost:
            best = r
        if best.cost < 1.0 * len(target):  # already a good fit: skip the second start
            break
    return None if best is None else PTZ.from_vec(best.x)
