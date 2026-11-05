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
             blurs: tuple[float, ...] = (4.0, 1.5, 0.0), max_nfev: int = 60, scale_sigma: float = 0.25) -> np.ndarray:
    """Refine a court->image H on the distance map. 8 dof = image offsets (WORK_W px) of 4 court reference points
    (the corners of the visible samples' court bounding box). Points visible under the START H are fixed; a point
    leaving the image counts as TRUNC (the optimiser cannot push the court out of view). Prior: offsets stay near
    the prior H's positions (sigma prior_px) or, without a prior, near the start (sigma free_px, weak).
    ``scale_sigma``: the log change of the reference quad's edge lengths is penalised with this sigma (a free
    start could otherwise shrink the whole court onto one dense blob of evidence, e.g. a logo); None = off."""
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
    e0 = np.linalg.norm(np.roll(r0, -1, axis=0) - r0, axis=1)  # reference quad edge lengths at the start
    wq = np.sqrt(len(P) / 8.0) * 0.25

    def spread(p):  # sqrt of the covariance determinant of the (finite) projected samples: their 2-D extent
        f = p[np.isfinite(p[:, 0])]
        if len(f) < 8:
            return 1e-3
        return float(np.sqrt(max(np.linalg.det(np.cov(f.T)), 1e-6)))

    s0 = spread(p0[vis])

    def res_on(z, dm):
        p = project(H_of(z), P)
        d = _lookup(dm, p)
        d = np.where(np.isfinite(d), d, TRUNC)
        q = r0 + z.reshape(4, 2)
        pr = (q - rp).ravel() / sig * 0.6
        parts = [d, pr * wq]
        if scale_sigma:  # soft barriers: ~free within one sigma, prohibitive at collapse scale (log ratio ~2)
            e = np.linalg.norm(np.roll(q, -1, axis=0) - q, axis=1)
            u = np.log(np.maximum(e, 1e-3) / e0) / scale_sigma
            # the samples' own extent: extreme perspective squeezes them onto a band while the quad keeps its
            # edges (a court "seen" edge-on along one thick line of evidence)
            us = np.log(spread(p) / s0) / (2 * scale_sigma)
            # dead zone (+-15 % extent is free), then weight ~ n (the data term it must beat is a sum over n
            # samples): a 2x change in extent costs more than any fit can gain, one zoom grid step almost nothing
            us = np.sign(us) * max(abs(us) - 0.3, 0.0)
            parts.append(np.append(u * (1.0 + u * u) * wq, us * (1.0 + us * us) * 3.0 * np.sqrt(len(P))))
        return np.concatenate(parts).astype(float)

    for b in blurs:
        dm = cv2.GaussianBlur(d_f, (0, 0), b) if b > 0 else d_f

        def res(z, dm=dm):  # bound to this blur level
            return res_on(z, dm)

        try:
            x = least_squares(res, x, jac=lambda z, r=res: _fd_jac(r, z, r(z)), loss="huber", f_scale=HUBER_PX,
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
           prior_H: np.ndarray | None = None, need: float | None = None,
           min_area_frac: float = 0.0) -> tuple[bool, str]:
    """Section 2.3 acceptance on the score dict of score_H. ``need`` overrides the score threshold (a frame
    initialised without a tracked prior must score JUMP_SCORE); ``min_area_frac``: the visible court samples must
    span at least this share of the frame (a court collapsed onto a textured patch scores well on nothing)."""
    if sc["n"] < MIN_VISIBLE:
        return False, f"{sc['n']} visible samples"
    seen = [c for c, (n, _) in sc["per_class"].items() if n >= MIN_CLASS_N]
    if len(seen) < 2:
        return False, "one line class only"
    if need is None:
        need = ACCEPT_SCORE
        if prior_H is not None and corner_jump(H, prior_H, size) > JUMP_FRAC:
            need = JUMP_SCORE
    if sc["score"] < need:
        return False, f"score {sc['score']:.2f} < {need:.2f}"
    for c, (n, v) in sc["per_class"].items():
        if n >= BAD_CLASS_N and v < BAD_CLASS_SCORE:
            return False, f"class {c} score {v:.2f}"
    if min_area_frac > 0:
        p = project(H, pts)
        w, h = size
        vis = np.isfinite(p[:, 0]) & (p[:, 0] >= 0) & (p[:, 0] < w) & (p[:, 1] >= 0) & (p[:, 1] < h)
        if vis.sum() >= 3:
            area = cv2.contourArea(cv2.convexHull(p[vis].astype(np.float32)))
            if area < min_area_frac * w * h:
                return False, f"court covers {area / (w * h):.2f} of the frame"
    ok, why = plausible(H, size, pts)
    return ok, why


# ------------------------------------------------------------ evidence that survives paint, aprons and logos
FLOOR_BAND = 0.45  # without person boxes the floor colour is seeded from the lower (1 - FLOOR_BAND) of the frame
FLOOR_M2 = 12.0  # Mahalanobis gate (squared) of the floor colour model
FLOOR_MIN_FRAC = 0.03  # a "floor" region smaller than this share of the frame is not a court view
HOLE_FRAC = 0.004  # holes in the floor region smaller than this share of the frame are glare / grain, not paint
FLOOR_TEX = 20.0  # candidate floor regions are weighted by exp(-mean |gradient| / this): crowds are textured


def _boxes_at(shape: tuple[int, int], boxes, s: float, pad_frac: float = 0.1) -> np.ndarray:
    """bool mask (True inside the person boxes, padded) at the evidence resolution."""
    m = np.zeros(shape, bool)
    if boxes is None:
        return m
    for x1, y1, x2, y2 in np.asarray(boxes, float).reshape(-1, 4) * s:
        pad = pad_frac * (x2 - x1)
        m[max(0, int(y1 - pad)):int(y2 + pad) + 1, max(0, int(x1 - pad)):int(x2 + pad) + 1] = True
    return m


def floor_evidence(img: np.ndarray, person_boxes=None, static_mask: np.ndarray | None = None,
                   lp: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """Court-line evidence for floors with painted keys / aprons and big logos: (N, 2) int16 pixels at WORK_W.

    TV floors mark the court as much by colour as by stripes: a dark painted key, a painted apron around a wood
    court (the boundary IS the sideline), lettering on the apron, a logo over the centre circle. The classic
    line_pixels sees thin stripes only and is flooded by wood grain, crowd and LED boards. Here:
      1. floor colour model (Lab mean + covariance): seeded by the pixels under the players' feet (pass-1 person
         boxes - the court is where the players stand), else by the dominant colour of the lower part of the frame;
      2. floor region = pixels inside the colour gate, closed over thin lines, opened, large components only, small
         holes (glare, grain) filled - the painted key and a logo are large holes and stay;
      3. evidence = the region's boundary (court outline, lane, FT line, logo rim) + thin lines INSIDE the region
         (line_pixels of any colour class) - crowd, LED boards and apron lettering lie outside and are dropped.
    Returns the pixels and {"floor_frac", "lab"}; an empty array when no floor region was found."""
    h0, w0 = img.shape[:2]
    hh = int(round(h0 * WORK_W / w0))
    s = WORK_W / w0
    im = cv2.resize(img, (WORK_W, hh), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB).astype(np.float32)
    boxes = _boxes_at((hh, WORK_W), person_boxes, s)
    valid = ~boxes
    if static_mask is not None and static_mask.shape == valid.shape:
        valid &= ~static_mask
    # candidate floor colours: the most common coarse Lab bins under the players' feet and in the lower band;
    # the winner is the colour whose region covers most of the lower part of the frame (the court is the big
    # region at the bottom; crowd, bench and apron colours lose)
    cands: list[np.ndarray] = []
    band = valid.copy()
    band[: int(hh * FLOOR_BAND)] = False
    seed = np.zeros((hh, WORK_W), bool)
    if person_boxes is not None:
        for x1, y1, x2, y2 in np.asarray(person_boxes, float).reshape(-1, 4) * s:
            d = max(3, int(0.08 * (y2 - y1)))
            seed[int(y2) + 1:int(y2) + 1 + d, max(0, int(x1)):int(x2) + 1] = True
        seed &= valid
    for sel, top in ((seed, 2), (band, 3)):
        px = lab[sel]
        if len(px) < 200:
            continue
        q = np.floor(px / np.array([32.0, 6.0, 6.0], np.float32)).astype(np.int64)
        key = (q[:, 0] * 64 + q[:, 1]) * 64 + q[:, 2]
        u, cnt = np.unique(key, return_counts=True)
        for k in u[np.argsort(-cnt)[:top]]:
            k = int(k)
            cands.append(np.array([(k // 4096) * 32 + 16, ((k // 64) % 64) * 6 + 3, (k % 64) * 6 + 3], np.float32))
    if not cands:
        return np.zeros((0, 2), np.int16), {"floor_frac": 0.0}
    px_all = lab[band] if band.sum() >= 200 else lab[valid]
    # texture: a floor is smooth at this scale, a crowd / bench row is not (mean |gradient| of L inside the region)
    gL = cv2.magnitude(cv2.Sobel(lab[:, :, 0], cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(lab[:, :, 0], cv2.CV_32F, 0, 1, ksize=3))
    best = None
    seen: list[np.ndarray] = []
    for centre in cands:
        if any(np.linalg.norm(centre - c) < 10 for c in seen):
            continue
        seen.append(centre)
        sub = px_all
        for _ in range(2):  # grain and glare vary mostly along L and b: model them with the covariance
            near = np.linalg.norm((px_all - centre) * np.array([0.35, 1.0, 1.0], np.float32), axis=1) < 14
            if near.sum() < 100:
                break
            sub = px_all[near]
            centre = sub.mean(0)
        cov = np.cov(sub.T) + np.diag([9.0, 2.0, 2.0])
        try:
            icov = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            continue
        dl = (lab - centre).reshape(-1, 3)
        m2 = np.einsum("ij,jk,ik->i", dl, icov, dl).reshape(hh, WORK_W)
        fl = (m2 < FLOOR_M2).astype(np.uint8)
        fl[~valid] = 0
        fl = cv2.morphologyEx(fl, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        fl = cv2.morphologyEx(fl, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        n, cc, st, _ = cv2.connectedComponentsWithStats(fl, connectivity=4)
        if n <= 1:
            continue
        areas = st[1:, cv2.CC_STAT_AREA]
        bottoms = st[1:, cv2.CC_STAT_TOP] + st[1:, cv2.CC_STAT_HEIGHT]
        # large components that reach the lower half of the frame (the camera looks down at the court; a crowd
        # band of similar colour sits above it)
        big = np.where((areas >= max(0.2 * areas.max(), 0.002 * fl.size)) & (bottoms >= FLOOR_BAND * hh))[0] + 1
        if not len(big):
            continue
        fl = np.isin(cc, big).astype(np.uint8)
        low = float(fl[int(hh * FLOOR_BAND):].mean())
        tex = float(gL[fl > 0].mean()) if fl.any() else 1e9
        val = low * float(np.exp(-tex / FLOOR_TEX))
        if best is None or val > best[0]:
            best = (val, fl, centre, tex)
    if best is None:
        return np.zeros((0, 2), np.int16), {"floor_frac": 0.0}
    _val, floor, centre, tex = best
    frac = float(floor.mean())
    if frac < FLOOR_MIN_FRAC:
        return np.zeros((0, 2), np.int16), {"floor_frac": frac}
    inv = (1 - floor).astype(np.uint8)
    n_h, hc, st_h, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    small = np.zeros(n_h, bool)
    for j in range(1, n_h):
        x, y, w_, h_, a = st_h[j]
        small[j] = a < HOLE_FRAC * floor.size and x > 0 and y > 0 and x + w_ < WORK_W and y + h_ < hh
    floor[small[hc]] = 1
    # 1. boundary of the floor region (not along masked boxes / overlays / the frame edge)
    grad = cv2.morphologyEx(floor, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    grad[cv2.dilate((~valid).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0] = 0
    # 2. thin lines inside the region
    if lp is None:
        from replay.autocourt import line_pixels

        lp = line_pixels(img, person_boxes, static_mask)
    if lp is not None and len(lp):
        inside = cv2.dilate(floor, np.ones((5, 5), np.uint8))
        xs, ys = np.clip(lp[:, 0], 0, WORK_W - 1), np.clip(lp[:, 1], 0, hh - 1)
        keep = inside[ys, xs] > 0
        grad[ys[keep], xs[keep]] = 1
    ys, xs = np.nonzero(grad)
    floor4 = cv2.resize(floor, (WORK_W // COARSE_DS, hh // COARSE_DS), interpolation=cv2.INTER_AREA) > 0
    return np.column_stack([xs, ys]).astype(np.int16), {"floor_frac": frac, "lab": [round(float(v), 1) for v in centre],
                                                        "tex": round(tex, 1), "floor4": floor4}


def dist_from_points(pts_px: np.ndarray, size: tuple[int, int], trunc: float = TRUNC) -> np.ndarray:
    """uint8 distance map (d * DT_SCALE, truncated) at WORK_W from evidence pixels (x, y) at WORK_W."""
    hh = int(round(size[1] * WORK_W / size[0]))
    m = np.zeros((hh, WORK_W), np.uint8)
    if pts_px is not None and len(pts_px):
        m[np.clip(pts_px[:, 1], 0, hh - 1), np.clip(pts_px[:, 0], 0, WORK_W - 1)] = 1
    d = cv2.distanceTransform((1 - m).astype(np.uint8), cv2.DIST_L2, 3)
    return np.round(np.minimum(d, trunc) * DT_SCALE).astype(np.uint8)


# ----------------------------------------------------------------------------- coarse camera search
COARSE_TRUNC = 40.0  # px at WORK_W: the wide map used to refine coarse hits sees evidence this far
COARSE_DS = 4  # the coarse search scores on a 1/COARSE_DS map ...
COARSE_FAR = 40.0  # ... truncated at this many of its pixels (160 px at WORK_W): a wide basin for a coarse grid
COARSE_PX = 8.0  # coarse score = exp(-d / COARSE_PX), d in 1/COARSE_DS pixels
COARSE_RANK = 12  # coarse candidates given a short wide-map refine and re-ranked by the fine score ...
COARSE_TOP = 3  # ... of which this many get the full refine
COARSE_STEP = 0.15  # target grid spacing = this share of the frame width in the image (adaptive to distance / f)
COARSE_SPREAD = 0.04  # projected samples must spread (smallest std) at least this share of the frame width
FAMILIES = {
    # (dx from the centre line, distance from the near sideline, height) in metres; f in frame widths
    "tv": dict(dx=(-6.0, 0.0, 6.0), dy=(12.0, 22.0), z=(8.0, 15.0), f=(0.7, 1.1, 1.7, 2.6, 4.0, 6.0)),
    "gym": dict(dx=(-8.0, 0.0, 8.0), dy=(3.0, 7.0), z=(1.8, 3.5), f=(0.45, 0.7, 1.1, 1.7)),
}


def _H_cam(C: np.ndarray, target: np.ndarray, f: float, size: tuple[int, int]) -> np.ndarray:
    from replay.camera_model import rotation

    d = np.asarray(target, float) - C
    pan = float(np.arctan2(d[1], d[0]))
    tilt = float(np.arctan2(-d[2], np.hypot(d[0], d[1])))
    R = rotation(pan, tilt, 0.0)
    K = np.array([[f, 0, size[0] / 2], [0, f, size[1] / 2], [0, 0, 1.0]])
    return K @ np.column_stack([R[:, 0], R[:, 1], -R @ C])


def camera_hypotheses(size: tuple[int, int], families=("tv", "gym"), C_near: np.ndarray | None = None) -> np.ndarray:
    """(M, 3, 3) court->image homographies of a grid of plausible cameras for the current court: camera on the
    near side, looking at points spread over the court. ``C_near``: a known camera position (TV cameras do not
    move) -> a small grid of positions around it with a finer zoom grid."""
    from replay import court

    L, W = court.LENGTH, court.WIDTH
    w = size[0]
    out = []

    def add(C, f):
        # targets spaced so that neighbouring views differ by ~COARSE_STEP of the frame width in the image
        dist = float(np.linalg.norm(np.array([L / 2, W / 2, 0.0]) - C))
        step = max(0.8, dist * COARSE_STEP * w / f)
        for x in np.arange(1.0, L - 1.0 + 1e-6, step):
            for y in np.arange(1.0, W - 1.0 + 1e-6, step):
                out.append(_H_cam(C, np.array([x, y, 0.0]), f, size))

    if C_near is not None:
        C0 = np.asarray(C_near, float)
        offs = [(0, 0, 0), (3, 0, 0), (-3, 0, 0), (0, 4, 0), (0, -4, 0), (0, 0, 3), (0, 0, -3)]
        for o in offs:
            C = C0 + np.array(o, float)
            if C[2] < 1.2:
                continue
            for f in np.exp(np.linspace(np.log(0.5), np.log(7.0), 10)) * w:
                add(C, f)
        return np.array(out)
    for fam in families:
        g = FAMILIES[fam]
        for dx in g["dx"]:
            for dy in g["dy"]:
                for z in g["z"]:
                    for f in g["f"]:
                        add(np.array([L / 2 + dx, -dy, z]), f * w)
    return np.array(out)


def evidence_maps(ev: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """From evidence pixels (x, y at WORK_W): (dist uint8 at WORK_W truncated at TRUNC - the fine map,
    dist_c float32 at WORK_W truncated at COARSE_TRUNC - the wide map, dist_c4 float32 at WORK_W / COARSE_DS
    truncated at COARSE_FAR of its pixels - the coarse-search map, ev4 bool evidence mask at WORK_W / COARSE_DS)."""
    hh = int(round(size[1] * WORK_W / size[0]))
    m = np.zeros((hh, WORK_W), np.uint8)
    if ev is not None and len(ev):
        m[np.clip(ev[:, 1], 0, hh - 1), np.clip(ev[:, 0], 0, WORK_W - 1)] = 1
    d = cv2.distanceTransform((1 - m).astype(np.uint8), cv2.DIST_L2, 3)
    dist = np.round(np.minimum(d, TRUNC) * DT_SCALE).astype(np.uint8)
    dist_c = np.minimum(d, COARSE_TRUNC).astype(np.float32)
    m4 = (cv2.resize(m.astype(np.float32), (WORK_W // COARSE_DS, hh // COARSE_DS), interpolation=cv2.INTER_AREA) > 0)
    d4 = cv2.distanceTransform((~m4).astype(np.uint8), cv2.DIST_L2, 3)
    dist_c4 = np.minimum(d4, COARSE_FAR).astype(np.float32)
    return dist, dist_c, dist_c4, m4


# -------------------------------------------------------- correlation search (pan / tilt by image translation)
TEMPLATE_FAMILIES = {
    # camera positions (distance from the near sideline, height) in metres and zooms (f in frame widths): the
    # viewing direction is NOT gridded - it comes from the correlation (an image translation of the template)
    "tv": dict(dy=(10.0, 18.0, 30.0, 45.0), z=(6.0, 10.0, 16.0, 24.0), f=(0.6, 0.85, 1.2, 1.7, 2.4, 3.4, 4.8, 6.5)),
    "gym": dict(dy=(2.5, 5.0, 8.0), z=(1.6, 3.0, 5.0), f=(0.45, 0.65, 0.9, 1.3, 1.8)),
}
CORR_SIGMA = 2.0  # px at WORK_W / COARSE_DS: blur of the evidence for the correlation
CORR_PER = 2  # translations kept per template (local maxima)
CORR_MIN_VIS = 60  # template samples inside the frame


def template_hypotheses(size: tuple[int, int], families=("tv", "gym")) -> np.ndarray:
    """(M, 3, 3) court->image homographies whose viewing DIRECTION is only roughly right: per camera position and
    zoom, targets spaced by half the visible court width (so every part of the court is the template's content
    once) and two tilts. correlate_candidates finds the pan / tilt as an image translation."""
    from replay import court

    L, W = court.LENGTH, court.WIDTH
    w = size[0]
    out = []
    for fam in families:
        g = TEMPLATE_FAMILIES[fam]
        for dy in g["dy"]:
            for z in g["z"]:
                C = np.array([L / 2, -dy, z])
                for fw in g["f"]:
                    f = fw * w
                    D = float(np.hypot(dy + W / 2, z))
                    step = max(1.5, 0.5 * D / fw)  # half the visible width (metres) at the court centre
                    for x in np.arange(1.5, L - 1.5 + 1e-6, step):
                        for y in (W / 2 - 3.0, W / 2 + 3.0):
                            out.append(_H_cam(C, np.array([x, y, 0.0]), f, size))
    return np.array(out)


def correlate_candidates(ev4: np.ndarray, size: tuple[int, int], pts: np.ndarray, Hs: np.ndarray,
                         per: int = CORR_PER) -> np.ndarray:
    """For each template H: rasterise its visible court samples at the ev4 resolution, correlate with the blurred
    evidence (FFT, all translations at once), keep the best ``per`` translations -> candidate Hs (translated in
    full-frame pixels). Returns (K, 3, 3); score them with coarse_scores."""
    hh, ww = ev4.shape
    s = ww / size[0]
    E = cv2.GaussianBlur(ev4.astype(np.float32), (0, 0), CORR_SIGMA)
    if E.max() <= 0:
        return np.zeros((0, 3, 3))
    E /= E.max()
    Ep = np.zeros((3 * hh, 3 * ww), np.float32)
    Ep[hh:2 * hh, ww:2 * ww] = E
    P = np.column_stack([pts, np.ones(len(pts))]).astype(np.float32)
    out = []
    kern = np.ones((5, 5), np.uint8)
    for H in Hs:
        q = P @ np.asarray(H, np.float32).T
        wgt = q[:, 2]
        ok = wgt > 1e-6
        x = np.where(ok, q[:, 0] / np.where(ok, wgt, 1), -1) * s
        y = np.where(ok, q[:, 1] / np.where(ok, wgt, 1), -1) * s
        vis = ok & (x >= 0) & (x < ww) & (y >= 0) & (y < hh)
        n = int(vis.sum())
        if n < CORR_MIN_VIS:
            continue
        xi, yi = x[vis].astype(np.int32), y[vis].astype(np.int32)
        x0, y0, x1, y1 = xi.min(), yi.min(), xi.max() + 1, yi.max() + 1
        if x1 - x0 < 3 or y1 - y0 < 3:
            continue
        tmpl = np.zeros((y1 - y0, x1 - x0), np.float32)
        tmpl[yi - y0, xi - x0] = 1.0
        npx = float(tmpl.sum())
        R = cv2.matchTemplate(Ep, tmpl, cv2.TM_CCORR) / max(npx, 60.0)
        # local maxima (5x5), best `per`
        mx = cv2.dilate(R, kern)
        cand = np.argwhere((R >= mx) & (R > 0.15))
        if not len(cand):
            continue
        vals = R[cand[:, 0], cand[:, 1]]
        for k in np.argsort(-vals)[:per]:
            j, i = cand[k]
            dx = (i - (x0 + ww)) / s
            dy = (j - (y0 + hh)) / s
            T = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]])
            out.append(T @ H)
    return np.array(out) if out else np.zeros((0, 3, 3))


def coarse_scores(dist_c4: np.ndarray, Hs: np.ndarray, size: tuple[int, int], pts: np.ndarray,
                  min_vis: int = MIN_VISIBLE, chunk: int = 1500, floor4: np.ndarray | None = None) -> np.ndarray:
    """Coarse score per hypothesis on the coarse-search map (evidence_maps): mean exp(-d / COARSE_PX) over the
    visible samples, scaled down when fewer than 120 are visible (a court seen in full on the evidence beats a
    corner that happens to lie on a line); 0 when the projected samples do not spread in both directions
    (a court squashed onto one thick band of evidence - a sliver - scores perfectly and means nothing).
    With ``floor4`` (the floor region at the same resolution, floor_evidence) the score is also weighted by region
    consistency: the court's interior must land on the floor region and the floor region must map back into the
    court (flat views that squeeze the court into a band of evidence fail both)."""
    from replay import court

    hh, ww = dist_c4.shape
    s = ww / size[0]
    P = np.column_stack([pts, np.ones(len(pts))]).astype(np.float32)
    out = np.zeros(len(Hs), np.float32)
    if floor4 is not None and floor4.shape == dist_c4.shape and floor4.sum() >= 50:
        gx, gy = np.meshgrid(np.linspace(0.6, court.LENGTH - 0.6, 14), np.linspace(0.6, court.WIDTH - 0.6, 7))
        Q = np.column_stack([gx.ravel(), gy.ravel(), np.ones(gx.size)]).astype(np.float32)  # court interior
        fy, fx = np.nonzero(floor4)
        rng = np.random.default_rng(0)
        sel = rng.choice(len(fx), min(300, len(fx)), replace=False)
        F = np.column_stack([(fx[sel] + 0.5) / s, (fy[sel] + 0.5) / s, np.ones(len(sel))]).astype(np.float32)
    else:
        floor4 = None
    for a in range(0, len(Hs), chunk):
        Hc = np.asarray(Hs[a:a + chunk], np.float32)
        reg = 1.0
        if floor4 is not None:
            qi = np.einsum("mij,nj->mni", Hc, Q)
            wi = qi[:, :, 2]
            with np.errstate(divide="ignore", invalid="ignore"):
                xi_ = qi[:, :, 0] / wi * s
                yi_ = qi[:, :, 1] / wi * s
            vi = (wi > 1e-6) & (xi_ >= 0) & (xi_ < ww) & (yi_ >= 0) & (yi_ < hh)
            on = floor4[np.where(vi, yi_, 0).astype(np.int32), np.where(vi, xi_, 0).astype(np.int32)] & vi
            nvi = vi.sum(1)
            inside = np.where(nvi >= 6, on.sum(1) / np.maximum(nvi, 1), 0.0)
            try:
                Hinv = np.linalg.inv(Hc.astype(np.float64)).astype(np.float32)
                qf = np.einsum("mij,nj->mni", Hinv, F)
                wf = qf[:, :, 2]
                with np.errstate(divide="ignore", invalid="ignore"):
                    cx = qf[:, :, 0] / wf
                    cy = qf[:, :, 1] / wf
                inc = (np.abs(wf) > 1e-9) & (cx >= -1.0) & (cx <= court.LENGTH + 1.0) & (cy >= -1.0) & (cy <= court.WIDTH + 1.0)
                covered = inc.mean(1)
            except np.linalg.LinAlgError:
                covered = np.zeros(len(Hc))
            reg = (0.3 + 0.7 * inside) * (0.3 + 0.7 * covered)
        q = np.einsum("mij,nj->mni", Hc, P)
        wgt = q[:, :, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            x = q[:, :, 0] / wgt * s
            y = q[:, :, 1] / wgt * s
        vis = (wgt > 1e-6) & (x >= 0) & (x < ww - 1) & (y >= 0) & (y < hh - 1)
        xi = np.where(vis, x, 0).astype(np.int32)
        yi = np.where(vis, y, 0).astype(np.int32)
        d = dist_c4[yi, xi]
        e = np.where(vis, np.exp(-d / COARSE_PX), 0.0).sum(1)
        n = vis.sum(1)
        nn = np.maximum(n, 1)
        # spread of the visible samples: smallest eigenvalue of their 2x2 covariance
        mx = np.where(vis, x, 0).sum(1) / nn
        my = np.where(vis, y, 0).sum(1) / nn
        dx = np.where(vis, x - mx[:, None], 0)
        dy = np.where(vis, y - my[:, None], 0)
        sxx, syy, sxy = (dx * dx).sum(1) / nn, (dy * dy).sum(1) / nn, (dx * dy).sum(1) / nn
        tr, det = sxx + syy, sxx * syy - sxy * sxy
        lam_min = tr / 2 - np.sqrt(np.maximum(tr * tr / 4 - det, 0))
        sc = e / np.maximum(n, 120) * reg
        sc[(n < min_vis) | (lam_min < (COARSE_SPREAD * ww) ** 2)] = 0.0
        out[a:a + chunk] = sc
    return out


def _diverse_top(Hs: np.ndarray, scores: np.ndarray, size: tuple[int, int], top: int) -> list[int]:
    """Indices of the best hypotheses that are not near-duplicates (court corners > 6 % of the width apart)."""
    order = np.argsort(-scores)
    keep: list[int] = []
    for j in order:
        if scores[j] <= 0:
            break
        if all(corner_jump(Hs[j], Hs[k], size) > 0.06 for k in keep):
            keep.append(int(j))
        if len(keep) >= top:
            break
    return keep


def solve_frame(dist: np.ndarray, size: tuple[int, int], pts: np.ndarray, cls: np.ndarray,
                inits: list[tuple[str, np.ndarray, dict]], hyps: np.ndarray | None = None,
                need_free: float = JUMP_SCORE, need_tracked: float = ACCEPT_SCORE,
                min_area_frac: float = 0.08, dist_c: np.ndarray | None = None,
                dist_c4: np.ndarray | None = None, floor4: np.ndarray | None = None,
                ev4: np.ndarray | None = None, debug: list | None = None) -> dict | None:
    """One frame: try the initialisations in order (("tracked", H0, {prior}) | ("carry", H0, {}) |
    ("peaks", H0, {})), each refined on the distance map and accepted with the matching threshold; when none
    passes and ``hyps`` is given, a coarse search: with ``ev4`` the hyps are templates (template_hypotheses)
    whose pan / tilt is found by correlation (correlate_candidates), else a plain grid (camera_hypotheses);
    candidates are scored on ``dist_c4`` (+ region consistency with ``floor4``), the best COARSE_TOP refined on
    the wide map ``dist_c`` and then on ``dist``. Returns {"H", "score", "source", "sc"} or None."""
    for name, H0, kw in inits:
        if H0 is None:
            continue
        if name == "tracked":
            H = refine_H(dist, H0, size, pts, prior_H=H0, prior_px=kw.get("prior_px", 8.0))
            need = need_tracked
        else:
            H = refine_H(dist, H0, size, pts, free_px=kw.get("free_px", 60.0), blurs=(6.0, 2.0, 0.0))
            need = need_free
        sc = score_H(dist, H, size, pts, cls)
        ok, why = accept(sc, H, size, pts, need=need, min_area_frac=min_area_frac)
        if debug is not None:
            debug.append({"H": H, "score": sc["score"], "source": name, "sc": sc, "ok": ok, "why": why})
        if ok:
            return {"H": H, "score": sc["score"], "source": name, "sc": sc}
    if hyps is None or not len(hyps):
        return None
    if dist_c is None:
        dist_c = _as_float(dist)
    if dist_c4 is None:
        hh, ww = dist_c.shape
        dist_c4 = cv2.resize(dist_c, (ww // COARSE_DS, hh // COARSE_DS), interpolation=cv2.INTER_AREA) / COARSE_DS
    if ev4 is not None:
        hyps = correlate_candidates(ev4, size, pts, hyps)
        if not len(hyps):
            return None
    scores = coarse_scores(dist_c4, hyps, size, pts, floor4=floor4)
    # the coarse score cannot rank the near-truth candidate reliably (its zoom / tilt is off by a grid step):
    # a short refine on the wide map for the best COARSE_RANK, ranked by the fine score, then the full refine
    stage1 = []
    for j in _diverse_top(hyps, scores, size, COARSE_RANK):
        H1 = refine_H(dist_c, hyps[j], size, pts, free_px=150.0, blurs=(10.0, 4.0), max_nfev=25)
        sc1 = score_H(dist, H1, size, pts, cls)
        if sc1["n"] >= MIN_VISIBLE:
            stage1.append((sc1["score"] * min(1.0, sc1["n"] / 120.0), H1))
    stage1.sort(key=lambda t: -t[0])
    best = None
    for _s1, H1 in stage1[:COARSE_TOP]:
        H = refine_H(dist, H1, size, pts, free_px=40.0, blurs=(3.0, 1.0, 0.0), max_nfev=60)
        sc = score_H(dist, H, size, pts, cls)
        ok, why = accept(sc, H, size, pts, need=need_free, min_area_frac=min_area_frac)
        if debug is not None:
            debug.append({"H": H, "score": sc["score"], "source": "coarse", "sc": sc, "ok": ok, "why": why})
        if ok and (best is None or sc["score"] > best["score"]):
            best = {"H": H, "score": sc["score"], "source": "coarse", "sc": sc}
    return best
