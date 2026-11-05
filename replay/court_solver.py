"""Per-frame court solver (court_design.md stage 0): homography refinement on line evidence + acceptance.

Used by autocourt.calibrate_net (only when CalibrationConfig.court_detector is not "off"). Works on the EXISTING
court detector output (one line-probability channel, no semantic classes yet) plus the classic line_pixels:
  dist_map(line_prob, lp, size)   -> truncated distance transform of the line evidence at WORK_W
  class_samples(step)             -> court-line samples (metres) + their marking class
  refine_H(dist, H0, size, ...)   -> H minimising the Huber distance of projected samples to the evidence (8 dof,
                                     4-point parameterisation, blur schedule 4 / 1.5 / 0 px like refine_pose)
  score_H(dist, H, size, ...)     -> mean exp(-d / 4 px) overall and per class
  accept(...)                     -> (ok, reason): enough samples from >= 2 classes, score, plausible camera,
                                     no jump against the tracked prior
Own code (OpenCV + SciPy).
"""

from __future__ import annotations

import math

import cv2
import numpy as np

WORK_W = 640
TRUNC = 10.0  # px at WORK_W
DT_SCALE = 16.0  # distance maps are stored as uint8 (d * DT_SCALE) to keep hundreds of frames in memory
SCORE_PX = 4.0  # score = exp(-d / SCORE_PX)
HUBER_PX = 3.0
JAC_STEP_PX = 0.25  # refine_H: finite-difference step of the corner offsets (WORK_W px), see _fd_jac
ACCEPT_SCORE = 0.55
JUMP_SCORE = 0.65
JUMP_FRAC = 0.08
MIN_VISIBLE = 40
MIN_CLASS_N = 8  # visible samples for a class to count as "seen"
BAD_CLASS_N, BAD_CLASS_SCORE = 15, 0.3

CLASSES = ("sideline", "baseline", "centre", "lane", "ft_line", "ft_circle", "three")


def class_samples(step: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    """Points along the current court's markings (court.configure) and their class index in CLASSES.
    The no-charge arcs are left out (often not painted)."""
    from replay import court

    lines = court.court_polylines(n_arc=96)
    # order (court.court_polylines): rectangle, centre line, centre circle, then pairs (left, right) of
    # paint, FT circle, 3-pt line, no-charge arc
    kinds = ["rect", "centre", "centre", "paint", "paint", "ft_circle", "ft_circle", "three", "three", None, None]
    pts, cls = [], []
    for line, kind in zip(lines, kinds):
        if kind is None:
            continue
        for a, b in zip(line[:-1], line[1:]):
            seg = b - a
            L = float(np.linalg.norm(seg))
            if L < 1e-6:
                continue
            horiz = abs(seg[0]) >= abs(seg[1])
            if kind == "rect":
                c = "sideline" if horiz else "baseline"
            elif kind == "paint":
                c = "lane" if horiz else "ft_line"
            else:
                c = kind
            n = max(1, int(math.ceil(L / step)))
            u = np.linspace(0, 1, n, endpoint=False)[:, None]
            pts.append(a + seg * u)
            cls.append(np.full(n, CLASSES.index(c), np.int8))
    return np.vstack(pts), np.concatenate(cls)


def dist_map(line_prob: np.ndarray | None, lp: np.ndarray | None, size: tuple[int, int], thr: float = 0.5) -> np.ndarray:
    """uint8 distance map (d * DT_SCALE, truncated at TRUNC) at WORK_W of the union of the net's line mask and the
    classic line pixels (lp: autocourt.line_pixels output at WORK_W)."""
    hh = int(round(size[1] * WORK_W / size[0]))
    m = np.zeros((hh, WORK_W), np.uint8)
    if line_prob is not None:
        mk = (line_prob > thr).astype(np.uint8)
        m |= cv2.resize(mk, (WORK_W, hh), interpolation=cv2.INTER_NEAREST)
    if lp is not None and len(lp):
        m[np.clip(lp[:, 1], 0, hh - 1), np.clip(lp[:, 0], 0, WORK_W - 1)] = 1
    d = cv2.distanceTransform((1 - m).astype(np.uint8), cv2.DIST_L2, 3)
    return np.round(np.minimum(d, TRUNC) * DT_SCALE).astype(np.uint8)


def _as_float(dist: np.ndarray) -> np.ndarray:
    return dist.astype(np.float32) / DT_SCALE if dist.dtype == np.uint8 else dist.astype(np.float32)


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


def project(H: np.ndarray, pts: np.ndarray, s: float = 1.0) -> np.ndarray:
    """court (N,2) -> pixels * s; NaN for points behind the camera."""
    q = np.column_stack([pts, np.ones(len(pts))]) @ np.asarray(H, float).T
    ok = q[:, 2] > 1e-9
    p = np.full((len(pts), 2), np.nan)
    p[ok] = q[ok, :2] / q[ok, 2:3] * s
    return p


def score_H(dist: np.ndarray, H: np.ndarray, size: tuple[int, int], pts: np.ndarray, cls: np.ndarray) -> dict:
    """Mean exp(-d / SCORE_PX) over projected samples inside the frame, overall and per class."""
    d_f = _as_float(dist)
    s = d_f.shape[1] / size[0]
    d = _lookup(d_f, project(H, pts, s))
    vis = np.isfinite(d)
    sc = np.exp(-d[vis] / SCORE_PX)
    per = {}
    cv = cls[vis]
    for c in np.unique(cv):
        k = cv == c
        per[CLASSES[int(c)]] = (int(k.sum()), float(sc[k].mean()))
    return {"score": float(sc.mean()) if len(sc) else 0.0, "n": int(vis.sum()), "per_class": per}


def camera_from_H(H: np.ndarray, size: tuple[int, int]):
    """(C (3,), tilt_deg, f) of a pinhole camera (principal point at the centre, square pixels) that gives this
    court->image H, or None when f cannot be recovered (degenerate)."""
    w, h = size
    T = np.array([[1, 0, -w / 2], [0, 1, -h / 2], [0, 0, 1.0]])
    M = T @ np.asarray(H, float)
    M = M / np.linalg.norm(M[:, :2])
    h1, h2 = M[:, 0], M[:, 1]
    cands = []
    den = h1[2] * h2[2]
    if abs(den) > 1e-12:
        cands.append(-(h1[0] * h2[0] + h1[1] * h2[1]) / den)
    den2 = h1[2] ** 2 - h2[2] ** 2
    if abs(den2) > 1e-12:
        cands.append(-(h1[0] ** 2 + h1[1] ** 2 - h2[0] ** 2 - h2[1] ** 2) / den2)
    f2 = [c for c in cands if np.isfinite(c) and c > 0]
    if not f2:
        return None
    f = float(math.sqrt(f2[0]))
    Kinv = np.diag([1 / f, 1 / f, 1.0])
    A = Kinv @ M
    lam = 2.0 / (np.linalg.norm(A[:, 0]) + np.linalg.norm(A[:, 1]))
    best = None
    for sg in (1.0, -1.0):
        r1, r2, t = sg * lam * A[:, 0], sg * lam * A[:, 1], sg * lam * A[:, 2]
        r1 = r1 / np.linalg.norm(r1)
        r2 = r2 - r1 * np.dot(r1, r2)
        r2 = r2 / np.linalg.norm(r2)
        r3 = np.cross(r1, r2)
        R = np.column_stack([r1, r2, r3])
        C = -R.T @ t
        if C[2] > 0:
            best = (C, R)
            break
    if best is None:
        return None
    C, R = best
    fwd = R[2]  # rows of R are the camera axes in world coordinates (right, down, forward)
    tilt = float(np.degrees(np.arcsin(np.clip(-fwd[2], -1, 1))))
    return C, tilt, f


def _corners_px(H, size):
    from replay import court

    c = np.array([[0, 0], [court.LENGTH, 0], [court.LENGTH, court.WIDTH], [0, court.WIDTH]], float)
    return project(H, c)


def plausible(H: np.ndarray, size: tuple[int, int], pts: np.ndarray) -> tuple[bool, str]:
    """Geometry checks that need no line evidence: in front of the camera, camera on the near side (far sideline
    above the near one, not mirrored), camera height / tilt / focal length in range."""
    from replay import court

    q = np.column_stack([pts, np.ones(len(pts))]) @ np.asarray(H, float).T
    p = project(H, pts)
    w, h = size
    inside = np.isfinite(p[:, 0]) & (p[:, 0] >= 0) & (p[:, 0] < w) & (p[:, 1] >= 0) & (p[:, 1] < h)
    if inside.sum() and (q[inside, 2] <= 0).any():
        return False, "behind camera"
    # local Jacobian at the centroid of the visible samples: court +y (far) must go up, not mirrored
    c = pts[inside].mean(0) if inside.sum() else np.array([court.LENGTH / 2, court.WIDTH / 2])
    e = 0.5
    a, bx, by = project(H, np.array([c, c + [e, 0], c + [0, e]]))
    if not (np.isfinite(a).all() and np.isfinite(bx).all() and np.isfinite(by).all()):
        return False, "degenerate"
    J = np.column_stack([bx - a, by - a]) / e
    if J[1, 1] >= 0:
        return False, "far side below near side"
    if np.linalg.det(J) >= 0:
        return False, "mirrored"
    cam = camera_from_H(H, size)
    if cam is not None:
        C, tilt, f = cam
        if not (1.5 <= C[2] <= 25.0):
            return False, f"camera height {C[2]:.1f} m"
        if not (3.0 <= tilt <= 60.0):
            return False, f"tilt {tilt:.0f} deg"
        if not (0.5 * w <= f <= 12.0 * w):
            return False, f"f {f:.0f} px"
    return True, ""


def _fd_jac(fun, x: np.ndarray, f0: np.ndarray, step: float = JAC_STEP_PX) -> np.ndarray:
    """Forward-difference Jacobian with an ABSOLUTE step. least_squares' ``diff_step`` is relative to |x|; at the
    start x = 0, so SciPy fell back to a ~1e-8 px step, which vanishes in the float32 corner transform and the
    bilinear lookup: J was 0 and the optimiser stopped at once (gtol), leaving H = H0."""
    J = np.empty((len(f0), len(x)))
    for j in range(len(x)):
        z = x.copy()
        z[j] += step
        J[:, j] = (fun(z) - f0) / step
    return J


def refine_H(dist: np.ndarray, H0: np.ndarray, size: tuple[int, int], pts: np.ndarray,
             prior_H: np.ndarray | None = None, prior_px: float = 8.0, free_px: float = 40.0,
             blurs: tuple[float, ...] = (4.0, 1.5, 0.0), max_nfev: int = 60) -> np.ndarray:
    """Refine a court->image H on the distance map. 8 dof = image offsets (WORK_W px) of 4 court reference points
    (the corners of the visible samples' court bounding box). Points visible under the START H are fixed; a point
    leaving the image counts as TRUNC (the optimiser cannot push the court out of view). Prior: offsets stay near
    the prior H's positions (sigma prior_px) or, without a prior, near the start (sigma free_px, weak)."""
    from scipy.optimize import least_squares

    d_f = _as_float(dist)
    s = d_f.shape[1] / size[0]
    S = np.diag([s, s, 1.0])
    Sinv = np.diag([1 / s, 1 / s, 1.0])
    Hw = S @ np.asarray(H0, float)
    p0 = project(Hw, pts)
    hh, ww = d_f.shape
    vis = np.isfinite(p0[:, 0]) & (p0[:, 0] >= 0) & (p0[:, 0] < ww - 1) & (p0[:, 1] >= 0) & (p0[:, 1] < hh - 1)
    if vis.sum() < 10:
        return np.asarray(H0, float)
    P = pts[vis]
    lo, hi = P.min(0), P.max(0)
    ctr = (lo + hi) / 2
    half = np.maximum((hi - lo) / 2, 1.0)
    ref = np.array([[ctr[0] - half[0], ctr[1] - half[1]], [ctr[0] + half[0], ctr[1] - half[1]],
                    [ctr[0] + half[0], ctr[1] + half[1]], [ctr[0] - half[0], ctr[1] + half[1]]])
    r0 = project(Hw, ref)
    if not np.isfinite(r0).all():
        return np.asarray(H0, float)
    if prior_H is not None:
        rp = project(S @ np.asarray(prior_H, float), ref)
        if not np.isfinite(rp).all():
            rp, sig = r0, free_px
        else:
            sig = prior_px
    else:
        rp, sig = r0, free_px
    src = ref.astype(np.float32)
    front = np.array([ctr[0], ctr[1], 1.0])  # a court point in front of the camera (centre of the visible samples)

    def H_of(x):
        M = cv2.getPerspectiveTransform(src, (r0 + x.reshape(4, 2)).astype(np.float32)).astype(float)
        # getPerspectiveTransform scales h33 to 1: when the court origin is behind the camera (h33 < 0 in the
        # true H) that flips the sign and every point would project "behind the camera" - keep the visible side
        return -M if M[2] @ front < 0 else M

    x = np.zeros(8)
    for b in blurs:
        dm = cv2.GaussianBlur(d_f, (0, 0), b) if b > 0 else d_f

        def res(z):
            p = project(H_of(z), P)
            d = _lookup(dm, p)
            d = np.where(np.isfinite(d), d, TRUNC)
            pr = ((r0 + z.reshape(4, 2)) - rp).ravel() / sig * 0.6
            return np.concatenate([d, pr * np.sqrt(len(P) / 8.0) * 0.25]).astype(float)

        try:
            x = least_squares(res, x, jac=lambda z: _fd_jac(res, z, res(z)), loss="huber", f_scale=HUBER_PX,
                              max_nfev=max_nfev, x_scale=2.0).x
        except Exception:  # noqa: BLE001 - singular quad: keep the last good x
            break
    Hr = Sinv @ H_of(x)
    return Hr / abs(Hr[2, 2])  # (abs: dividing by a negative h33 would flip the visible side, see H_of)


def corner_jump(H: np.ndarray, H_ref: np.ndarray, size: tuple[int, int]) -> float:
    """Median movement of the court corners (and centre) between two Hs, as a fraction of the frame width."""
    from replay import court

    c = np.array([[0, 0], [court.LENGTH, 0], [court.LENGTH, court.WIDTH], [0, court.WIDTH],
                  [court.LENGTH / 2, court.WIDTH / 2]], float)
    a, b = project(H, c), project(H_ref, c)
    d = np.linalg.norm(a - b, axis=1)
    d = d[np.isfinite(d)]
    return float(np.median(d) / size[0]) if len(d) else 1.0


def accept(sc: dict, H: np.ndarray, size: tuple[int, int], pts: np.ndarray,
           prior_H: np.ndarray | None = None) -> tuple[bool, str]:
    """Section 2.3 acceptance on the score dict of score_H."""
    if sc["n"] < MIN_VISIBLE:
        return False, f"{sc['n']} visible samples"
    seen = [c for c, (n, _) in sc["per_class"].items() if n >= MIN_CLASS_N]
    if len(seen) < 2:
        return False, "one line class only"
    need = ACCEPT_SCORE
    if prior_H is not None and corner_jump(H, prior_H, size) > JUMP_FRAC:
        need = JUMP_SCORE
    if sc["score"] < need:
        return False, f"score {sc['score']:.2f} < {need:.2f}"
    for c, (n, v) in sc["per_class"].items():
        if n >= BAD_CLASS_N and v < BAD_CLASS_SCORE:
            return False, f"class {c} score {v:.2f}"
    ok, why = plausible(H, size, pts)
    return ok, why
