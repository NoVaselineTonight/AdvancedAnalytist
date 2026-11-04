"""Court-space smoothing: robust Rauch-Tung-Striebel smoother with a physically plausible motion model.

We process offline, so a forward-backward (RTS) smoother is used instead of a plain Kalman filter: each
position uses past *and* future measurements, with no lag.

Motion model (per court axis): position + velocity driven by continuous-time white-noise acceleration,
with the velocity slowly relaxing towards zero (integrated Ornstein-Uhlenbeck / Singer model):

    dv = -v / velocity_tau_s dt + dW,   E[dW^2] = q dt,   q = accel_sigma^2 * 0.1 s

Being continuous-time, the result does not depend on the processing frame rate (the same settings give the
same motion at 10 fps and 30 fps). The velocity relaxation stops the smoother from shooting off in long
gaps: a player who is unseen for 2 s is bridged with a gentle curve, not a big overshooting arc.

Measurement noise comes from the homography Jacobian (see postprocess.py), so far-court positions (where
1 px is many cm) are trusted less than near ones.

Outliers (foot points that jump when a box is cut off, merged with another player or only covers the upper
body) are handled in two layers:
  1. forward pass: Student-t down-weighting of large innovations, hard rejection beyond ``gate_chi2``;
  2. a few iteratively-reweighted passes against the *smoothed* track (which sees the future too, so a
     short run of bad measurements is recognised even right at the start of a track).
Measurements finally rejected are reported as not ``measured``.

Gaps up to ``max_gap_s`` are filled by the smoother and flagged ``interpolated``. Longer gaps split the
track into independent segments and nothing is output inside them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from replay.config import SmoothingConfig

_Q_REF_S = 0.1  # accel_sigma is specified per 0.1 s (the historical 10 fps sample interval)


@dataclass
class SmoothedSegment:
    k: np.ndarray  # sample indices (contiguous)
    xy: np.ndarray  # (n, 2)
    measured: np.ndarray  # (n,) bool - False = interpolated or rejected outlier
    vel: np.ndarray | None = None  # (n, 2) smoothed velocity, m/s
    # Per-sample confidence for the UI / lineup (see data/benchmark/notes_motion.md):
    #   state: STATE_SEEN (measurement used), STATE_PREDICTED (gap filled by the motion model),
    #          STATE_REJECTED (a foot point existed but was an outlier; position from the model),
    #          STATE_OCCLUDED / STATE_OFFSCREEN (predicted, and the spot is behind a nearer person / out of view;
    #          only with smooth_track(..., hidden=...))
    #   sigma: 1-sigma position uncertainty in metres (sqrt of the largest eigenvalue of the smoothed
    #          position covariance; grows inside gaps, largest in the middle of a gap)
    state: np.ndarray | None = None  # (n,) uint8
    sigma: np.ndarray | None = None  # (n,) float, metres
    gap_sigma: np.ndarray | None = None  # (n,) the gap-spread part of sigma (rescaled by the game phase)


STATE_SEEN, STATE_PREDICTED, STATE_REJECTED, STATE_OCCLUDED, STATE_OFFSCREEN = 1, 2, 3, 4, 5


def _sigma_of(P: np.ndarray) -> np.ndarray:
    """Largest 1-sigma axis (m) of 2x2 position covariances (n, 2, 2)."""
    a, b, d = P[:, 0, 0], P[:, 0, 1], P[:, 1, 1]
    lam = 0.5 * (a + d) + np.sqrt(np.maximum(0.25 * (a - d) ** 2 + b * b, 0.0))
    return np.sqrt(np.maximum(lam, 0.0))


def _gap_sigma(used: np.ndarray, dt: float, cfg: SmoothingConfig) -> np.ndarray:
    """Realistic extra uncertainty inside gaps (the RTS covariance of the smooth motion model is far too
    confident about where a hidden player went): Brownian-bridge spread  s^2 = g^2 * a*b/(a+b),  a/b = seconds
    since the last / until the next used measurement (one-sided at segment ends)."""
    g = float(getattr(cfg, "gap_sigma_m_per_sqrt_s", 1.5))
    n = len(used)
    if g <= 0 or used.all() or not used.any():
        return np.zeros(n)
    idx = np.arange(n)
    last = np.maximum.accumulate(np.where(used, idx, -1))
    nxt = np.minimum.accumulate(np.where(used, idx, n)[::-1])[::-1]
    a = np.where(last >= 0, (idx - last) * dt, np.inf)
    b = np.where(nxt < n, (nxt - idx) * dt, np.inf)
    both = np.isfinite(a) & np.isfinite(b)
    with np.errstate(invalid="ignore"):
        var = np.where(both, a * b / np.maximum(a + b, 1e-9), np.minimum(a, b))
    return g * np.sqrt(np.where(used, 0.0, var))


def _states(used: np.ndarray, has: np.ndarray) -> np.ndarray:
    st = np.full(len(used), STATE_PREDICTED, np.uint8)
    st[has & ~used] = STATE_REJECTED
    st[used] = STATE_SEEN
    return st


def motion_model(dt: float, cfg: SmoothingConfig) -> tuple[np.ndarray, np.ndarray]:
    """Transition F and process noise Q for state [x, y, vx, vy] over dt seconds."""
    q = cfg.accel_sigma ** 2 * _Q_REF_S
    tau = cfg.velocity_tau_s
    if tau and tau > 0:
        a = np.exp(-dt / tau)
        f01 = tau * (1 - a)
        q11 = q * tau ** 2 * (dt - 2 * tau * (1 - a) + tau / 2 * (1 - a * a))
        q12 = q * tau ** 2 * ((1 - a) - (1 - a * a) / 2)
        q22 = q * tau / 2 * (1 - a * a)
    else:  # pure constant velocity (white-noise acceleration)
        a, f01 = 1.0, dt
        q11, q12, q22 = q * dt ** 3 / 3, q * dt ** 2 / 2, q * dt
    F = np.eye(4)
    F[0, 2] = F[1, 3] = f01
    F[2, 2] = F[3, 3] = a
    Q = np.zeros((4, 4))
    for p, v in ((0, 2), (1, 3)):
        Q[p, p], Q[p, v], Q[v, p], Q[v, v] = q11, q12, q12, q22
    return F, Q


def _mahal(y0: float, y1: float, s00: float, s01: float, s11: float):
    """Squared Mahalanobis distance of (y0, y1) under the 2x2 covariance S, and S^-1 as (a, b, d)."""
    det = s00 * s11 - s01 * s01
    a, b, d = s11 / det, -s01 / det, s00 / det
    return y0 * (a * y0 + b * y1) + y1 * (b * y0 + d * y1), (a, b, d)


def _rts(grid: np.ndarray, meas_at: dict[int, int], z: np.ndarray, R: np.ndarray, F: np.ndarray, Q: np.ndarray,
         v0_var: float, gate_chi2: float, nu: float, forward_robust: bool, reset_n: int = 0):
    """One forward Kalman pass + RTS backward pass. R may be inflated per measurement (robust weights).

    reset_n > 0: after this many consecutive gated-out measurements the filter is assumed to have lost
    lock (e.g. a real move it did not believe, or an ID switch upstream) and its position uncertainty is
    reopened so it can re-acquire the player instead of rejecting everything from then on.
    Returns smoothed states, smoothed covariances and which measurements were used.
    """
    n = len(grid)
    xs_f = np.zeros((n, 4))
    Ps_f = np.zeros((n, 4, 4))
    xs_p = np.zeros((n, 4))
    Ps_p = np.zeros((n, 4, 4))
    used = np.zeros(n, bool)
    j0 = meas_at[int(grid[0])]
    x = np.array([z[j0, 0], z[j0, 1], 0.0, 0.0])
    P = np.zeros((4, 4))
    P[:2, :2] = R[j0]
    P[2, 2] = P[3, 3] = v0_var
    FT = F.T
    n_rej = 0
    zl = z.tolist()
    Rl = R.reshape(-1, 4).tolist()
    for i in range(n):
        if i > 0:
            x = F @ x
            P = F @ P @ FT + Q
        xs_p[i], Ps_p[i] = x, P
        j = meas_at.get(int(grid[i]))
        if j is not None and Rl[j][0] < np.inf:
            r00, r01, _, r11 = Rl[j]
            y0, y1 = zl[j][0] - x[0], zl[j][1] - x[1]
            p00, p01, p11 = P[0, 0], P[0, 1], P[1, 1]
            m2, (s00, s01, s11) = _mahal(y0, y1, p00 + r00, p01 + r01, p11 + r11)
            if m2 > gate_chi2 and reset_n and n_rej + 1 >= reset_n:
                P = P.copy()
                P[0, 0] += 25.0
                P[1, 1] += 25.0
                P[2, 2] += v0_var
                P[3, 3] += v0_var
                p00, p01, p11 = P[0, 0], P[0, 1], P[1, 1]
                m2, (s00, s01, s11) = _mahal(y0, y1, p00 + r00, p01 + r01, p11 + r11)
            if i == 0 or m2 <= gate_chi2:
                n_rej = 0
                if forward_robust and i > 0 and m2 > 2.0:
                    # Student-t: shrink the influence of surprising measurements instead of all-or-nothing
                    w = (nu + 2.0) / (nu + m2)
                    _, (s00, s01, s11) = _mahal(y0, y1, p00 + r00 / w, p01 + r01 / w, p11 + r11 / w)
                Sinv = np.array([[s00, s01], [s01, s11]])
                K = P[:, :2] @ Sinv
                x = x + K @ np.array([y0, y1])
                P = P - K @ P[:2, :]
                P = (P + P.T) * 0.5
                used[i] = True
            else:
                n_rej += 1
        xs_f[i], Ps_f[i] = x, P

    # backward (RTS) pass; the gains are computed for all samples at once
    C = np.matmul(Ps_f[:-1] @ FT, np.linalg.inv(Ps_p[1:]))
    xs = xs_f.copy()
    Ps = Ps_f.copy()
    for i in range(n - 2, -1, -1):
        Ci = C[i]
        xs[i] = xs_f[i] + Ci @ (xs[i + 1] - xs_p[i + 1])
        Ps[i] = Ps_f[i] + Ci @ (Ps[i + 1] - Ps_p[i + 1]) @ Ci.T
    return xs, Ps, used


def smooth_segment(k: np.ndarray, z: np.ndarray, R: np.ndarray, dt: float, cfg: SmoothingConfig) -> SmoothedSegment:
    """k: sorted unique sample indices with measurements z (n,2) and covariances R (n,2,2); dt = s per index."""
    grid = np.arange(k[0], k[-1] + 1)
    if len(k) == 1:
        return SmoothedSegment(grid, z.copy(), np.ones(1, bool), np.zeros((1, 2)),
                               np.full(1, STATE_SEEN, np.uint8), _sigma_of(R[:1]))
    meas_at = {int(kk): i for i, kk in enumerate(k)}
    F, Q = motion_model(dt, cfg)
    nu = cfg.robust_nu
    v0_var = 4.0
    reset_n = max(3, int(round(cfg.relock_s / dt)))
    xs, Ps, used = _rts(grid, meas_at, z, R, F, Q, v0_var, cfg.gate_chi2, nu, True, reset_n)
    gi = np.array([meas_at.get(int(g), -1) for g in grid])
    has = gi >= 0
    for _ in range(max(0, cfg.robust_iters)):
        # reweight every measurement against the smoothed track (past and future both vote)
        r = z[gi[has]] - xs[has, :2]
        S = R[gi[has]] + Ps[has, :2, :2]
        m2 = np.einsum("ni,nij,nj->n", r, np.linalg.inv(S), r)
        w = (nu + 2.0) / (nu + m2)
        Rw = R.copy()
        idx = gi[has]
        Rw[idx] = R[idx] / np.maximum(w, 1e-4)[:, None, None]
        xs_new, Ps, _ = _rts(grid, meas_at, z, Rw, F, Q, v0_var, np.inf, nu, False)
        used = np.zeros(len(grid), bool)
        used[np.flatnonzero(has)[m2 <= cfg.gate_chi2]] = True
        done = np.abs(xs_new[:, :2] - xs[:, :2]).max() < 1e-3
        xs = xs_new
        if done:
            break
    gs = _gap_sigma(used, dt, cfg)
    sig = np.hypot(_sigma_of(Ps[:, :2, :2]), gs)
    return SmoothedSegment(grid, xs[:, :2].copy(), used, xs[:, 2:].copy(), _states(used, has), sig, gs)


def _hidden_gap_ok(ka: int, kb: int, za: np.ndarray, zb: np.ndarray, fps: float, cfg: SmoothingConfig,
                   hidden) -> bool:
    """A gap longer than max_gap_s is still bridged when the player was plausibly hidden all along: reachable at
    running speed and, along the straight line between the two sightings, behind someone / off-screen for at
    least ``occluded_gap_share`` of the samples (a visible empty spot means the player was not there)."""
    lim = float(getattr(cfg, "occluded_max_gap_s", 0.0) or 0.0)
    if hidden is None or lim <= 0 or (kb - ka - 1) > lim * fps:
        return False
    dt = (kb - ka) / fps
    if np.hypot(*(zb - za)) > float(getattr(cfg, "max_speed_mps", 7.0)) * dt + 1.0:
        return False
    ks = np.arange(ka + 1, kb)
    f = ((ks - ka) / (kb - ka))[:, None]
    codes = np.asarray(hidden(ks, za + f * (zb - za)))
    known = codes != 3  # UNKNOWN_VIS (no calibration) neither for nor against
    if not known.any():
        return False
    return float(np.mean(codes[known] != 0)) >= float(getattr(cfg, "occluded_gap_share", 0.7))


def smooth_track(k: np.ndarray, z: np.ndarray, R: np.ndarray, fps: float, cfg: SmoothingConfig,
                 hidden=None, conf: np.ndarray | None = None) -> list[SmoothedSegment]:
    """Smooth one identity. ``hidden(ks, xy) -> codes`` (court_tracker.OcclusionContext.classify_many: 0 visible,
    1 occluded, 2 off-screen, 3 unknown) is optional: with it, longer gaps where the player was hidden are
    bridged (``occluded_max_gap_s``) and filled samples are labelled STATE_OCCLUDED / STATE_OFFSCREEN."""
    if conf is not None and len(k) > 1:  # one measurement per sample: the most confident (dedup.py)
        from replay.dedup import best_per_sample

        sel = best_per_sample(k, conf, R)
        k, z, R = k[sel], z[sel], R[sel]
    else:
        order = np.argsort(k, kind="stable")
        k, z, R = k[order], z[order], R[order]
        if len(k) > 1:  # one measurement per sample (keep the first)
            keep = np.r_[True, np.diff(k) > 0]
            k, z, R = k[keep], z[keep], R[keep]
    max_gap = max(1, int(round(cfg.max_gap_s * fps)))
    # A gap of g missing samples means consecutive indices differ by g + 1.
    breaks = np.where(np.diff(k) > max_gap + 1)[0] + 1
    if hidden is not None and len(breaks):
        breaks = np.array([b for b in breaks if not _hidden_gap_ok(int(k[b - 1]), int(k[b]), z[b - 1], z[b], fps,
                                                                      cfg, hidden)], int)
    capped = []  # long gaps bridged only where the position stays sure enough (sigma_cap_m)
    if len(breaks) and float(getattr(cfg, "capped_gap_max_s", 0.0) or 0.0) > 0:
        lim = float(cfg.capped_gap_max_s) * fps
        vmax = float(getattr(cfg, "max_speed_mps", 7.0))
        ok = [int(b) for b in breaks if (k[b] - k[b - 1] - 1) <= lim
              and np.hypot(*(z[b] - z[b - 1])) <= vmax * (k[b] - k[b - 1]) / fps + 1.0]
        capped = [(int(k[b - 1]), int(k[b])) for b in ok]
        breaks = np.array([b for b in breaks if int(b) not in set(ok)], int)
    ctx = getattr(hidden, "__self__", None)
    prior = getattr(ctx, "prior", None)
    segs = []
    for idx in np.split(np.arange(len(k)), breaks):
        if len(idx):
            seg = smooth_segment(k[idx], z[idx], R[idx], 1.0 / fps, cfg)
            if ctx is not None and getattr(cfg, "snap_behind_occluder", False):
                seg = _snap_behind(seg, k[idx], z[idx], R[idx], fps, cfg, ctx)
            if prior is not None and seg.gap_sigma is not None:
                _phase_sigma(seg, prior)
            if hidden is not None:
                _label_hidden(seg, hidden)
            if capped:
                segs += _cap_gaps(seg, capped, float(getattr(cfg, "sigma_cap_m", 1.5)))
            else:
                segs.append(seg)
    return segs


def _phase_sigma(seg: SmoothedSegment, prior) -> None:
    """Rescale the gap part of sigma by the game phase (court_tracker.MotionPrior.sigma_scale)."""
    sc = prior.sigma_scale(seg.k)
    gs = seg.gap_sigma
    seg.sigma = np.sqrt(np.maximum(seg.sigma ** 2 - gs ** 2, 0.0) + (gs * sc) ** 2)
    seg.gap_sigma = gs * sc


def _cap_gaps(seg: SmoothedSegment, gaps: list, cap: float) -> list:
    """Inside the given long gaps keep only samples whose sigma <= cap (near a sighting); the unsure middle is
    dropped, splitting the segment: nothing is shown without recent evidence."""
    drop = np.zeros(len(seg.k), bool)
    for ka, kb in gaps:
        drop |= (seg.k > ka) & (seg.k < kb) & (seg.sigma > cap)
    if not drop.any():
        return [seg]
    out = []
    keep_idx = np.flatnonzero(~drop)
    for run in np.split(keep_idx, np.flatnonzero(np.diff(keep_idx) > 1) + 1):
        if len(run):
            sl = slice(run[0], run[-1] + 1)
            out.append(SmoothedSegment(seg.k[sl], seg.xy[sl], seg.measured[sl],
                                       None if seg.vel is None else seg.vel[sl],
                                       None if seg.state is None else seg.state[sl],
                                       None if seg.sigma is None else seg.sigma[sl],
                                       None if seg.gap_sigma is None else seg.gap_sigma[sl]))
    return out


def _label_hidden(seg: SmoothedSegment, hidden) -> None:
    pred = np.flatnonzero(seg.state == STATE_PREDICTED)
    if not len(pred):
        return
    codes = np.asarray(hidden(seg.k[pred], seg.xy[pred]))
    seg.state[pred[codes == 1]] = STATE_OCCLUDED
    seg.state[pred[codes == 2]] = STATE_OFFSCREEN


def annotate_row(row, seg: SmoothedSegment, k: int) -> None:
    """Attach the per-sample confidence of ``seg`` at sample ``k`` to an output row (export.TrackRow) as
    ``row.pos_state`` (1 seen, 2 predicted on a visible spot, 3 rejected outlier, 4 predicted behind a nearer
    person, 5 predicted out of view) and
    ``row.pos_sigma`` (metres). Plain attributes so export.py can pick them up with getattr(row, ..., None)."""
    if seg.state is None:
        return
    j = int(np.searchsorted(seg.k, int(k)))
    if j < len(seg.k) and int(seg.k[j]) == int(k):
        row.pos_state = int(seg.state[j])
        row.pos_sigma = round(float(seg.sigma[j]), 3)


def hidden_classifier(H: list, rows: np.ndarray, idx_of_k: dict[int, int], size: tuple[int, int],
                      overlay: np.ndarray | None, cfg: SmoothingConfig, t: np.ndarray | None = None):
    """The ``hidden`` callable for smooth_track (None when occlusion awareness is off). With sample times ``t``
    and ``game_aware_sigma`` its context also carries a court_tracker.MotionPrior (``.prior``)."""
    if not getattr(cfg, "occlusion_aware", False):
        return None
    from replay.court_tracker import MotionPrior, OcclusionContext

    ctx = OcclusionContext.from_rows(H, rows, idx_of_k, size, overlay)
    if t is not None and getattr(cfg, "game_aware_sigma", False):
        ctx.prior = MotionPrior.from_rows(H, rows, idx_of_k, np.asarray(t, float))
    return ctx.classify_many


def _snap_behind(seg: SmoothedSegment, k: np.ndarray, z: np.ndarray, R: np.ndarray, fps: float,
                 cfg: SmoothingConfig, ctx) -> SmoothedSegment:
    """Undetected but on a visible spot = probably hidden behind someone nearby: add soft pseudo-measurements
    at the nearest spot behind a nearer person (court_tracker.OcclusionContext.behind_point) for gap samples
    whose filled position lies on a visible, empty floor spot, then smooth again. Seen samples are unchanged
    as measurements; the snapped samples stay 'predicted' (labelled occluded afterwards)."""
    pred = np.flatnonzero(seg.state == STATE_PREDICTED)
    if not len(pred):
        return seg
    codes = ctx.classify_many(seg.k[pred], seg.xy[pred])
    sig_r = float(getattr(cfg, "snap_sigma_m", 0.7))
    add_k, add_z = [], []
    for j, c in zip(pred, codes):
        if c != 0:
            continue
        q = ctx.behind_point(int(seg.k[j]), seg.xy[j], max_m=min(2.0, 2.0 * float(seg.sigma[j]) + 0.5))
        if q is not None:
            add_k.append(int(seg.k[j]))
            add_z.append(q)
    if not add_k:
        return seg
    k2 = np.r_[k, add_k]
    z2 = np.r_[z, np.array(add_z)]
    R2 = np.r_[R, np.tile(np.eye(2) * sig_r ** 2, (len(add_k), 1, 1))]
    o = np.argsort(k2, kind="stable")
    new = smooth_segment(k2[o], z2[o], R2[o], 1.0 / fps, cfg)
    # states / sigma from the real measurements only; the pseudo-measurements narrow nothing
    pseudo = np.isin(new.k, add_k)
    new.state[pseudo] = STATE_PREDICTED
    new.measured = new.measured & ~pseudo
    new.sigma = np.maximum(new.sigma, seg.sigma)
    return new
