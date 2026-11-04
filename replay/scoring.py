"""Automatic made-basket detection from the ball (no scoreboard / graphics).

Pipeline
  1. **Camera from the floor homography.** ``H`` (image -> court) is inverted to the court -> image map
     ``G ~ K [r1 r2 t]``. With square pixels, zero skew and the principal point at the image centre the
     only unknown in ``K`` is the focal length ``f``; the two constraints r1.r2 = 0 and |r1| = |r2|
     (Zhang's method for a single plane) are each linear in ``1/f^2`` and are solved together by least
     squares. ``r3`` completes the rotation (its sign is chosen so the camera is above the floor) and
     ``P = K [r1 r2 r3 t]`` projects the 3D rim, 3.05 m above the floor point under each ring.
  2. **Rim track.** Per-frame estimates from propagated calibrations jitter, so the rim centre and radius
     are moved into camera-stabilised pixels (the camera-motion chain; a rim is static in the gym), a
     running median is taken over +-``rim_smooth_s`` and the result mapped back to every frame.
  3. **Made baskets.** All ball *candidates* of a frame are used (detections near the rim are weak and the
     single tracked ball often jumps elsewhere). Positions are measured relative to the rim in units of
     the rim's image radius. A basket = a candidate ABOVE the rim near it horizontally, followed within
     ``max_fall_s`` by a candidate BELOW the rim within ~1.5 rim widths (frames in between may have no
     detection: net, backboard, hands). Candidates that stay fixed relative to the rim (logos, the ring,
     a spare ball) are dropped first.
  4. **Ballistic check** (optional, when the approach has >= 4 tracked points). A 3D parabola with known
     gravity is fitted to the approach (linear least squares in the start point and velocity, through the
     recovered camera). Gravity fixes the metric depth, so a shot that passes clearly in front of or
     behind the ring - which looks like a make in 2D - is vetoed, and a fit through the ring raises the
     confidence.
  5. Debounce per rim, credit the team of the last possessor before the shot.

Trade-offs (precision over recall): the 2D test cannot see depth, so without enough approach detections
for the ballistic fit a ball dropping just in front of/behind the ring can still pass; tight horizontal
limits and the "bounced up again" / "re-appears above the rim" penalties remove most rim-outs, at the
cost of missing makes where the ball is only detected far from the rim. Tip-ins and dunks with the ball
hidden in hands are usually missed. Accuracy depends on the calibration: a rough keyframe gives a
rough focal length and therefore a rim that may be several pixels off.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from replay import court
from replay.config import ScoringConfig

RIM_HEIGHT = 3.05
RIM_R = 0.225  # inner radius of the ring
BALL_R = 0.12  # size 7 ball (size 6: 0.116)
GRAVITY = 9.81
BALL_TO_RIM_WIDTH = BALL_R / RIM_R  # ball diameter / rim diameter
# A ball that went THROUGH the ring is, just below it, horizontally inside the ring: |dx| <= ~1 rim radius
# (+ rim projection error). Measured against the broadcast score bar (full game): made baskets had |dx| <= 1.25
# below the rim; most false "makes" (ball dropping beside the ring, heads/shoes/hands under it) had more.
THROUGH_DX_R = 1.35
THROUGH_MIN_DY_R = 1.3  # ... and clearly below the ring (0.9-1.3 radii = ball resting on the front of the ring)
# The 3D ballistic veto works on exact synthetic geometry but NOT with real automatic calibrations: on the score-bar
# ground truth (full game + 9 clips) the fitted miss distance of made and missed shots overlapped completely
# (makes up to 5 m "off"), and the veto removed 25 made baskets for 10 false ones. The fit is still computed
# (details.ballistic, small confidence bonus); the veto is only applied when this flag is set.
BALLISTIC_VETO = False


# ---------------------------------------------------------------------------------------------- camera
@dataclass
class Camera:
    f: float
    c: tuple[float, float]  # principal point
    R: np.ndarray  # columns r1 r2 r3: court axes in camera coordinates
    t: np.ndarray
    f_pair: tuple[float, float]  # focal from each constraint alone (nan if degenerate): a consistency check

    @property
    def P(self) -> np.ndarray:
        K = np.array([[self.f, 0, self.c[0]], [0, self.f, self.c[1]], [0, 0, 1.0]])
        return K @ np.column_stack([self.R, self.t])

    def project(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """X (N,3) court metres -> (uv (N,2), depth (N,)). Points behind the camera get NaN."""
        Xc = np.asarray(X, float).reshape(-1, 3) @ self.R.T + self.t
        z = Xc[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            uv = self.f * Xc[:, :2] / z[:, None] + np.asarray(self.c)
        uv[z <= 1e-6] = np.nan
        return uv, z

    @property
    def centre(self) -> np.ndarray:
        return -self.R.T @ self.t


def camera_from_homography(H: np.ndarray | None, frame_size: tuple[int, int],
                           focal_range_rel: tuple[float, float] = (0.3, 15.0)) -> Camera | None:
    """Full pinhole camera from an image->court floor homography (see module docstring); None on failure."""
    if H is None or not np.isfinite(H).all():
        return None
    try:
        Gm = np.linalg.inv(np.asarray(H, float))
    except np.linalg.LinAlgError:
        return None
    w, h = frame_size
    cx, cy = w / 2.0, h / 2.0
    T = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1.0]])
    g = T @ Gm
    g = g / np.linalg.norm(g)
    s = float(w)  # work in (f / w) units for conditioning
    g1, g2, g3 = g[:, 0], g[:, 1], g[:, 2]
    # unknown x = (w / f)^2.  a * x + b = 0 for each constraint
    rows = [
        ((g1[0] * g2[0] + g1[1] * g2[1]) / s ** 2, g1[2] * g2[2]),
        ((g1[0] ** 2 + g1[1] ** 2 - g2[0] ** 2 - g2[1] ** 2) / s ** 2, g1[2] ** 2 - g2[2] ** 2),
    ]
    A = np.array(rows)
    norms = np.linalg.norm(A, axis=1)
    if norms.max() < 1e-12:
        return None
    single = []
    for a, b in rows:
        single.append(s / math.sqrt(-b / a) if abs(a) > 1e-15 and -b / a > 0 else float("nan"))
    An = A[norms > 1e-3 * norms.max()] / norms[norms > 1e-3 * norms.max(), None]
    x = -float(An[:, 0] @ An[:, 1]) / max(1e-18, float(An[:, 0] @ An[:, 0]))
    if not np.isfinite(x) or x <= 0:
        return None
    f = s / math.sqrt(x)
    if not (focal_range_rel[0] * w <= f <= focal_range_rel[1] * w):
        return None
    Kinv = np.diag([1 / f, 1 / f, 1.0])
    k1, k2, k3 = Kinv @ g1, Kinv @ g2, Kinv @ g3
    lam = 2.0 / (np.linalg.norm(k1) + np.linalg.norm(k2))
    r1, r2, t = lam * k1, lam * k2, lam * k3
    if (r1 * court.LENGTH / 2 + r2 * court.MID_Y + t)[2] < 0:  # court centre must be in front of the camera
        r1, r2, t = -r1, -r2, -t
    r3 = np.cross(r1, r2)
    U, _, Vt = np.linalg.svd(np.column_stack([r1, r2, r3]))
    R = U @ Vt
    t = t / ((np.linalg.norm(r1) + np.linalg.norm(r2)) / 2)
    if float(-R[:, 2] @ t) < 0:  # camera centre below the floor: mirrored z axis
        R[:, 2] = -R[:, 2]
    return Camera(f, (cx, cy), R, t, (single[0], single[1]))


def rim_in_image(cam: Camera, basket_xy: tuple[float, float], n: int = 16) -> tuple[float, float, float] | None:
    """(u, v, r_px) of the rim centre and the half width of its image ellipse; None if behind the camera."""
    bx, by = basket_xy
    a = np.linspace(0, 2 * math.pi, n, endpoint=False)
    pts = np.vstack([[bx, by, RIM_HEIGHT],
                     np.stack([bx + RIM_R * np.cos(a), by + RIM_R * np.sin(a), np.full(n, RIM_HEIGHT)], 1)])
    uv, z = cam.project(pts)
    if not np.isfinite(uv).all() or (z <= 0).any():
        return None
    r = 0.5 * float(uv[1:, 0].max() - uv[1:, 0].min())
    return float(uv[0, 0]), float(uv[0, 1]), r


def _apply(M: np.ndarray, p: Sequence[float]) -> np.ndarray:
    q = M @ np.array([p[0], p[1], 1.0])
    return q[:2] / q[2]


def rim_tracks(Hs: Sequence[np.ndarray | None], t: np.ndarray, frame_size: tuple[int, int], cfg: ScoringConfig,
               G: Sequence[np.ndarray] | None = None, seg: np.ndarray | None = None
               ) -> tuple[dict[str, list[tuple[float, float, float] | None]], list[Camera | None]]:
    """Smoothed per-frame rim (u, v, r) for "L" and "R" (None = unknown / off frame), plus per-frame cameras."""
    n = len(Hs)
    # H is piecewise constant between key frames: one camera (and one rim projection) per distinct H; the shared
    # Camera objects are read-only downstream, so the result is identical to one call per frame
    keys = [None if H is None else (np.asarray(H).dtype.str, np.shape(H), np.asarray(H).tobytes()) for H in Hs]
    cam_of: dict = {}
    for H, key in zip(Hs, keys):
        if key is not None and key not in cam_of:
            cam_of[key] = camera_from_homography(H, frame_size, cfg.focal_range_rel)
    cams = [None if key is None else cam_of[key] for key in keys]
    if G is None or seg is None:
        G, seg = [np.eye(3)] * n, np.arange(n)  # no camera-motion chain: no temporal smoothing
    w, h = frame_size
    out: dict[str, list[tuple[float, float, float] | None]] = {}
    for side, bxy in zip("LR", court.baskets()):
        rim_of = {key: (None if c is None else rim_in_image(c, bxy)) for key, c in cam_of.items()}
        raw = [None if key is None else rim_of[key] for key in keys]
        stab = np.full((n, 3), np.nan)
        for i, rr in enumerate(raw):
            if rr is None:
                continue
            p = _apply(G[i], rr[:2])
            q = _apply(G[i], (rr[0] + rr[2], rr[1]))
            stab[i] = [p[0], p[1], np.linalg.norm(q - p)]
        res: list[tuple[float, float, float] | None] = [None] * n
        for i in range(n):
            if Hs[i] is None:
                continue
            lo = int(np.searchsorted(t, t[i] - cfg.rim_smooth_s, "left"))
            hi = int(np.searchsorted(t, t[i] + cfg.rim_smooth_s, "right"))
            sel = np.arange(lo, hi)
            sel = sel[(np.asarray(seg)[sel] == seg[i]) & np.isfinite(stab[sel, 0])]
            if not len(sel):
                continue
            m = np.median(stab[sel], axis=0)
            Gi = np.linalg.inv(G[i])
            p = _apply(Gi, m[:2])
            q = _apply(Gi, (m[0] + m[2], m[1]))
            r = float(np.linalg.norm(q - p))
            mg = cfg.rim_margin_px
            if not (np.isfinite(p).all() and mg <= p[0] <= w - mg and mg <= p[1] <= h - mg and r >= cfg.min_rim_px):
                continue
            res[i] = (float(p[0]), float(p[1]), r)
        out[side] = res
    return out, cams


# ---------------------------------------------------------------------------------------------- ballistics
def fit_ballistic(P: np.ndarray, times: np.ndarray, uv: np.ndarray) -> tuple[np.ndarray, np.ndarray, float] | None:
    """3D parabola X(tau) = p0 + v0 tau - g tau^2/2 z through image points (one camera P, 3x4).

    Linear least squares in (p0, v0). Returns (p0, v0, rms reprojection px) or None.
    """
    A, b = [], []
    for tau, (u, v) in zip(times, uv):
        gz = np.array([0, 0, -0.5 * GRAVITY * tau * tau])
        for row, m in ((P[0], u), (P[1], v)):
            a = m * P[2, :3] - row[:3]
            c = m * P[2, 3] - row[3]
            A.append(np.concatenate([a, tau * a]))
            b.append(-c - a @ gz)
    A, b = np.array(A), np.array(b)
    if len(A) < 6:
        return None
    scale = np.linalg.norm(A, axis=1)
    A, b = A / scale[:, None], b / scale
    sv = np.linalg.svd(A, compute_uv=False)
    if sv[-1] < 1e-9 * sv[0]:
        return None
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    p0, v0 = x[:3], x[3:]
    X = p0 + np.outer(times, v0) + np.outer(-0.5 * GRAVITY * times ** 2, [0, 0, 1])
    q = np.hstack([X, np.ones((len(X), 1))]) @ P.T
    if (q[:, 2] <= 0).any():
        return None
    err = np.linalg.norm(q[:, :2] / q[:, 2:3] - uv, axis=1)
    return p0, v0, float(np.sqrt(np.mean(err ** 2)))


def rim_plane_crossing(p0: np.ndarray, v0: np.ndarray) -> np.ndarray | None:
    """Court (x, y) where the parabola descends through the rim height; None if it never gets that high."""
    a, bq, c = -0.5 * GRAVITY, v0[2], p0[2] - RIM_HEIGHT
    disc = bq * bq - 4 * a * c
    if disc < 0:
        return None
    tau = (-bq - math.sqrt(disc)) / (2 * a)  # later root (a < 0)
    return (p0 + v0 * tau)[:2]


# ---------------------------------------------------------------------------------------------- detection
@dataclass
class _Obs:
    dx: float  # (ball centre - rim centre) / rim radius px
    dy: float  # positive = below
    ratio: float  # ball width / expected width at the rim
    conf: float
    uv: tuple[float, float]


def _observations(ball_rows: Sequence[np.ndarray], rim: Sequence[tuple[float, float, float] | None],
                  cfg: ScoringConfig) -> list[list[_Obs]]:
    out: list[list[_Obs]] = []
    lo_dy, hi_dy = cfg.above_dy_r[0], cfg.below_dy_r[1]
    max_dx = max(cfg.above_dx_r, cfg.below_dx_r)
    for i, rr in enumerate(rim):
        cur: list[_Obs] = []
        rows = ball_rows[i] if i < len(ball_rows) else np.zeros((0, 5))
        if rr is not None and len(rows):
            u, v, r = rr
            for x1, y1, x2, y2, conf in np.asarray(rows, float)[:, :5]:
                if conf < cfg.min_ball_conf:
                    continue
                cx, cy, bw = (x1 + x2) / 2, (y1 + y2) / 2, 0.5 * ((x2 - x1) + (y2 - y1))
                o = _Obs(float((cx - u) / r), float((cy - v) / r), float(bw / (2 * BALL_TO_RIM_WIDTH * r)),
                         float(conf), (float(cx), float(cy)))
                if abs(o.dx) <= max_dx and lo_dy <= o.dy <= hi_dy and \
                        cfg.ball_size_ratio[0] <= o.ratio <= cfg.ball_size_ratio[1]:
                    cur.append(o)
        out.append(cur)
    return out


def _drop_static(obs: list[list[_Obs]], t: np.ndarray, cfg: ScoringConfig) -> None:
    """Remove candidates that sit at the same rim-relative place for most of +-static_window_s."""
    n = len(obs)
    keep_all = []
    for i in range(n):
        if not obs[i]:
            keep_all.append([])
            continue
        lo = int(np.searchsorted(t, t[i] - cfg.static_window_s))
        hi = int(np.searchsorted(t, t[i] + cfg.static_window_s, "right"))
        span = hi - lo - 1
        keep = []
        for o in obs[i]:
            same = sum(1 for j in range(lo, hi) if j != i and
                       any(math.hypot(o.dx - q.dx, o.dy - q.dy) < 0.6 for q in obs[j]))
            keep.append(o if span < 4 or same < cfg.static_frac * span else None)
        keep_all.append([o for o in keep if o is not None])
    obs[:] = keep_all


def _is_above(o: _Obs, cfg: ScoringConfig) -> bool:
    return cfg.above_dy_r[0] <= o.dy <= cfg.above_dy_r[1] and abs(o.dx) <= cfg.above_dx_r


def _is_below(o: _Obs, cfg: ScoringConfig) -> bool:
    return cfg.below_dy_r[0] <= o.dy <= cfg.below_dy_r[1] and abs(o.dx) <= cfg.below_dx_r


def _at_rim(o: _Obs) -> bool:
    return abs(o.dx) <= 1.5 and -1.5 <= o.dy <= 1.5


def _ballistic_check(a: int, oa: _Obs, ball_frames: Sequence[Any], t: np.ndarray, cams: Sequence[Camera | None],
                     G: Sequence[np.ndarray] | None, seg: np.ndarray | None, basket_xy: tuple[float, float],
                     cfg: ScoringConfig) -> dict | None:
    cam = cams[a]
    if cam is None:
        return None
    pts, taus = [list(oa.uv)], [0.0]
    Ga_inv = np.linalg.inv(G[a]) if G is not None else None
    for j in range(a - 1, -1, -1):
        if t[a] - t[j] > cfg.ballistic_window_s:
            break
        if j >= len(ball_frames):
            continue
        bf = ball_frames[j]
        if bf.box is None or bf.state == "possessed":
            if bf.state == "possessed":
                break  # still in the shooter's hands: the flight starts after this
            continue
        if G is not None and seg is not None and seg[j] != seg[a]:
            break
        p = ((bf.box[0] + bf.box[2]) / 2, (bf.box[1] + bf.box[3]) / 2)
        if Ga_inv is not None:
            p = tuple(_apply(Ga_inv @ G[j], p))
        pts.append(list(p))
        taus.append(float(t[j] - t[a]))
    if len(pts) < 4 or (max(taus) - min(taus)) < 0.25:
        return None
    pts_a, taus_a = np.array(pts), np.array(taus)
    use = np.ones(len(pts_a), bool)
    fit = None
    for _ in range(3):  # drop up to 2 outliers (tracker hopping to another object)
        fit = fit_ballistic(cam.P, taus_a[use], pts_a[use])
        if fit is None:
            return None
        p0, v0, rms = fit
        if rms <= cfg.ballistic_max_rms_px or use.sum() <= 4:
            break
        X = p0 + np.outer(taus_a, v0) + np.outer(-0.5 * GRAVITY * taus_a ** 2, [0, 0, 1])
        q = np.hstack([X, np.ones((len(X), 1))]) @ cam.P.T
        err = np.linalg.norm(q[:, :2] / q[:, 2:3] - pts_a, axis=1)
        err[~use] = -1
        err[0] = -1  # never drop the "above the rim" observation itself
        use[int(np.argmax(err))] = False
    assert fit is not None
    p0, v0, rms = fit
    if rms > cfg.ballistic_max_rms_px or np.linalg.norm(v0) > 25:
        return {"ok": False, "rms_px": round(rms, 2), "n": int(use.sum())}
    cross = rim_plane_crossing(p0, v0)
    dist = None if cross is None else float(np.hypot(cross[0] - basket_xy[0], cross[1] - basket_xy[1]))
    return {"ok": True, "rms_px": round(rms, 2), "n": int(use.sum()),
            "crossing_m": None if cross is None else [round(float(cross[0]), 2), round(float(cross[1]), 2)],
            "miss_m": None if dist is None else round(dist, 2),
            "apex_m": round(float(p0[2] + v0[2] ** 2 / (2 * GRAVITY)) if v0[2] > 0 else float(p0[2]), 2)}


def _shooter(a: int, ball_frames: Sequence[Any], t: np.ndarray, rim: Sequence[tuple[float, float, float] | None],
             cfg: ScoringConfig) -> int | None:
    for j in range(min(a, len(ball_frames) - 1), -1, -1):
        if t[a] - t[j] > cfg.shooter_window_s:
            break
        bf = ball_frames[j]
        if bf.state != "possessed" or bf.owner is None:
            continue
        rr = rim[j]
        if rr is not None and bf.box is not None:  # a "holder" whose box overlaps the rim is often a defender
            cx, cy = (bf.box[0] + bf.box[2]) / 2, (bf.box[1] + bf.box[3]) / 2
            if math.hypot(cx - rr[0], cy - rr[1]) < 2.5 * rr[2]:
                continue
        return int(bf.owner)
    return None


def detect_baskets(H: Sequence[np.ndarray | None], t: np.ndarray, frame: np.ndarray,
                   ball_rows: Sequence[np.ndarray], ball_frames: Sequence[Any], track_team: dict[int, str],
                   frame_size: tuple[int, int], fps: float, cfg: ScoringConfig | None = None,
                   G: Sequence[np.ndarray] | None = None, seg: np.ndarray | None = None,
                   quality: Sequence[str] | None = None, rims: dict | None = None,
                   cams: list | None = None) -> list[dict]:
    """Made-basket events (see module docstring for the schema and method).

    H[i]: image->court homography of analysed frame i (None = uncalibrated); t[i] video seconds; frame[i]
    video frame index; ball_rows[i]: (m, 5) [x1, y1, x2, y2, conf]; ball_frames: replay.ball.BallFrame per
    frame (owner / state / tracked box); track_team: track id -> "A" | "B" | ...; G/seg: camera-motion chain
    (calibration._segment_transforms) used to smooth the rim and to stabilise the ballistic fit.
    """
    cfg = cfg or ScoringConfig()
    if not cfg.enabled or not len(H):
        return []
    t = np.asarray(t, float)
    if rims is None or cams is None:
        rims, cams = rim_tracks(H, t, frame_size, cfg, G, seg)
    candidates: list[dict] = []
    for side, bxy in zip("LR", court.baskets()):
        rim = rims[side]
        if not any(r is not None for r in rim):
            continue
        obs = _observations(ball_rows, rim, cfg)
        _drop_static(obs, t, cfg)
        n = len(obs)
        for b in range(n):
            belows = [o for o in obs[b] if _is_below(o, cfg) and abs(o.dx) <= min(cfg.below_dx_r, THROUGH_DX_R)
                      and o.dy >= max(cfg.below_dy_r[0], THROUGH_MIN_DY_R)]
            if not belows:
                continue
            # most recent frame with the ball above the rim, within max_fall_s
            a, oa = None, None
            for j in range(b - 1, -1, -1):
                if t[b] - t[j] > cfg.max_fall_s:
                    break
                ab = [o for o in obs[j] if _is_above(o, cfg)]
                if ab:
                    a, oa = j, min(ab, key=lambda o: abs(o.dx) + 0.2 * abs(o.dy + 2))
                    break
            if a is None or oa is None:
                continue
            ob = min(belows, key=lambda o: abs(o.dx - oa.dx * 0.3) + 0.1 * abs(o.dy))
            if not (0.5 <= ob.ratio / oa.ratio <= 2.0):
                continue  # different objects
            between = [o for j in range(a + 1, b) for o in obs[j]]
            hit_rim = _at_rim(oa) or _at_rim(ob) or any(_at_rim(o) for o in between)
            # bounced up off the rim before coming down: rim-out risk
            bounced = False
            lo = int(np.searchsorted(t, t[a] - cfg.max_fall_s))
            seq = [(j, o) for j in range(lo, a + 1) for o in obs[j]]
            for j1, o1 in seq:
                if _at_rim(o1) and any(j2 > j1 and o2.dy < o1.dy - 2.0 for j2, o2 in seq):
                    bounced = True
                    break
            # re-appearing above the rim soon after = the "below" was not the ball going through
            hi = int(np.searchsorted(t, t[b] + 0.5, "right"))
            reappears = any(o.dy < -0.5 and abs(o.dx) < 3 for j in range(b + 1, min(hi, n)) for o in obs[j])

            c = 0.45
            c += 0.20 * float(np.clip(1 - abs(ob.dx) / cfg.below_dx_r, 0, 1))
            c += 0.10 * float(np.clip(1 - abs(oa.dx) / cfg.above_dx_r, 0, 1))
            c += 0.10 if hit_rim else 0.0
            c += 0.10 * 0.5 * (oa.conf + ob.conf)
            if bounced:
                c *= 0.75
            if reappears:
                c *= 0.6
            if ball_frames and a < len(ball_frames) and b < len(ball_frames) and \
                    ball_frames[a].state == "possessed" and ball_frames[b].state == "possessed":
                c *= 0.5  # looks like a player holding the ball near the rim
            bal = None
            if cfg.ballistic and ball_frames:
                bal = _ballistic_check(a, oa, ball_frames, t, cams, G, seg, bxy, cfg)
                if bal is not None and bal.get("ok"):
                    if BALLISTIC_VETO and (bal["miss_m"] is None or bal["miss_m"] > cfg.ballistic_veto_m):
                        continue  # clean 3D fit says the ball missed the ring
                    miss = bal["miss_m"] if bal["miss_m"] is not None else math.inf  # None: arc never reaches the rim
                    c += 0.10 if miss < 0.5 else 0.0
                    c *= 1.0 if miss < 0.8 else 0.85
            c = float(np.clip(c, 0, 1))
            if c < cfg.min_confidence:
                continue
            u, v, r = rim[b]  # type: ignore[misc]
            candidates.append({"side": side, "a": a, "b": b, "c": c, "oa": oa, "ob": ob, "rim": (u, v, r),
                               "hit_rim": hit_rim, "bounced": bounced, "reappears": reappears, "ballistic": bal})

    # debounce: best first, at most one per rim per debounce_s
    candidates.sort(key=lambda e: -e["c"])
    chosen: list[dict] = []
    for e in candidates:
        if any(o["side"] == e["side"] and abs(t[o["b"]] - t[e["b"]]) < cfg.debounce_s for o in chosen):
            continue
        chosen.append(e)
    chosen.sort(key=lambda e: t[e["b"]])

    events = []
    for e in chosen:
        a, b = e["a"], e["b"]
        rim = rims[e["side"]]
        shooter = _shooter(a, ball_frames, t, rim, cfg) if ball_frames else None
        team = track_team.get(shooter) if shooter is not None else None
        team = team if team in ("A", "B") else None
        cam = cams[b] or cams[a]
        u, v, r = e["rim"]
        fr = int(frame[b])
        events.append({
            "id": f"basket-{fr}-{e['side']}", "type": "basket", "t": round(float(t[b]), 3), "frame": fr,
            "basket": e["side"], "team": team, "confidence": round(e["c"], 3), "source": "auto",
            "details": {
                "shooter_track": shooter,
                "rim_px": [round(u, 1), round(v, 1), round(r, 2)],
                "t_above": round(float(t[a]), 3), "frame_above": int(frame[a]),
                "above_rel": [round(e["oa"].dx, 2), round(e["oa"].dy, 2)],
                "below_rel": [round(e["ob"].dx, 2), round(e["ob"].dy, 2)],
                "touched_rim_zone": bool(e["hit_rim"]), "bounced": bool(e["bounced"]),
                "reappeared_above": bool(e["reappears"]),
                "ballistic": e["ballistic"],
                "focal_px": None if cam is None else round(cam.f, 1),
                "calibration": None if quality is None else quality[b],
            },
        })
    return events
