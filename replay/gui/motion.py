"""Continuous, plausible motion for the 2D replay view (display only - exported data is unaffected).

The replay data has one (already smoothed) court position per analysed sample (~10 per second) with gaps where
a player was not seen. Drawing it directly is choppy; this module turns every track into a continuous trajectory
once (per smoothing setting), so the view can be drawn at the display rate (~60 fps) and evaluated at any time t
in O(log n) per track:

* between samples: cubic Hermite curve with velocities from central differences (clamped to a running speed of
  ``VMAX``; tangents are limited against the chord so the curve never loops or overshoots);
* short gaps (``<= bridge``): bridged along the same smooth curve, matching the velocities at both ends, and
  slightly faded (flagged interpolated);
* implausible jumps (faster than a person can run) or long gaps: never slid across - the dot coasts briefly
  (decelerating), fades out, and the next sighting fades in (no popping, no invented players);
* long runs the smoother filled in without measurements are faded towards ``INTERP_MIN_ALPHA``.

The ball follows its holder's trajectory while possessed, travels along a smooth path between holders (a pass,
or a shot towards the rim, with a small "height" bump for the arc), stays with the last holder / last spot while
its position is unknown and only fades after a long gap.

``PlaybackClock`` turns the media player's coarse position (it updates only ~15 times per second) into a smooth,
monotonic display clock.
"""

from __future__ import annotations

import bisect
import math

import numpy as np

from replay import court

VMAX = 8.0  # m/s, fastest plausible running speed
COAST_TAU = 0.35  # s, how quickly an unseen player's extrapolated motion dies out
COAST_MAX = 1.0  # m, never extrapolate further than this
SHORT_GAP = 0.35  # s, always bridged (a couple of missed samples)
BRIDGE_MAX = 1.5  # s, longest gap bridged with a curve (if "hold unseen" allows)
INTERP_FULL_S = 0.75  # s from the nearest measurement before an interpolated position starts to fade
INTERP_MIN_ALPHA = 0.35
BALL_GAP = 3.0  # s of unknown ball position before it fades
BALL_PASS_SPEED = 10.0  # m/s (plan view) to time a pass across an unknown stretch
BALL_MAX_SPEED = 15.0  # m/s, fastest plausible plan-view ball movement between samples
HAND_RAMP = 0.2  # s, the ball moves from the holder's hand to the flight path over this time
# position uncertainty (1 sigma, metres) shown as a halo around predicted / hidden players. Used when the data has
# no per-sample uncertainty (track "u", centimetres): grows with the time since the nearest measurement.
SIG_SEEN = 0.25
SIG_RATE = 1.2  # m per s without a measurement
SIG_MAX = 3.0


def _hermite(p0, p1, m0, m1, dt, u):
    """Cubic Hermite position and velocity at u in [0, 1] (m0, m1 are velocities per second)."""
    u2, u3 = u * u, u * u * u
    h00, h10, h01, h11 = 2 * u3 - 3 * u2 + 1, u3 - 2 * u2 + u, -2 * u3 + 3 * u2, u3 - u2
    d00, d10, d01, d11 = 6 * u2 - 6 * u, 3 * u2 - 4 * u + 1, -6 * u2 + 6 * u, 3 * u2 - 2 * u
    x = h00 * p0[0] + h10 * m0[0] * dt + h01 * p1[0] + h11 * m1[0] * dt
    y = h00 * p0[1] + h10 * m0[1] * dt + h01 * p1[1] + h11 * m1[1] * dt
    vx = (d00 * p0[0] + d01 * p1[0]) / dt + d10 * m0[0] + d11 * m1[0]
    vy = (d00 * p0[1] + d01 * p1[1]) / dt + d10 * m0[1] + d11 * m1[1]
    return x, y, vx, vy


def _limit(vx: float, vy: float, vmax: float) -> tuple[float, float]:
    s = math.hypot(vx, vy)
    return (vx, vy) if s <= vmax or s == 0 else (vx * vmax / s, vy * vmax / s)


def sample_dt(ts) -> float:
    """Typical time between analysed samples."""
    if len(ts) < 2:
        return 0.1
    d = np.diff(np.asarray(ts, float))
    d = np.sort(d[d > 0])
    if not len(d):
        return 0.1
    m = len(d) // 2
    return float(d[m]) if len(d) % 2 else float((d[m - 1] + d[m]) / 2)  # median (np.median: slow first call)


class TrackPath:
    """Continuous trajectory of one track (positions in metres, times in seconds)."""

    __slots__ = ("t", "x", "y", "vx", "vy", "f", "dm", "link", "gap", "run_start", "t0", "t1", "step", "_dense",
                 "sig", "ext")

    def __init__(self, t: list[float], x: list[float], y: list[float], f: list[int], step: float,
                 bridge: float = BRIDGE_MAX, sig: list[float] | None = None):
        n = len(t)
        self.t, self.x, self.y, self.f, self.step = t, x, y, f, step
        self.sig = sig  # per-sample 1-sigma position uncertainty (m) from the data, or None (estimated)
        self.ext = None  # optional joint extrapolator after a run end: ext(k, dt) -> (x, y, sigma) (JointDisplay)
        self._dense = None
        self.t0, self.t1 = (t[0], t[-1]) if n else (math.inf, -math.inf)
        consec = 1.6 * step
        if n == 0:
            self.link = self.gap = self.vx = self.vy = self.run_start = self.dm = []
            return
        ta, xa, ya = np.asarray(t, float), np.asarray(x, float), np.asarray(y, float)
        # links: which consecutive samples are joined by a curve (never faster than a person can run)
        dt = np.diff(ta)
        d = np.hypot(np.diff(xa), np.diff(ya))
        ok_speed = d <= VMAX * dt * 1.25 + 0.3  # + slack for position noise
        gap = dt > consec
        link = ok_speed & (~gap | (dt <= max(SHORT_GAP, bridge)))
        # velocities: central differences inside a linked run, one-sided at the run ends
        idx = np.arange(n)
        lprev = np.r_[False, link]
        lnext = np.r_[link, False]
        a = np.where(lprev, idx - 1, idx)
        b = np.where(lnext, idx + 1, idx)
        span = ta[b] - ta[a]
        has = b > a
        span[~has] = 1.0
        vx = np.where(has, (xa[b] - xa[a]) / span, 0.0)
        vy = np.where(has, (ya[b] - ya[a]) / span, 0.0)
        sp = np.hypot(vx, vy)
        scale = np.where(sp > VMAX, VMAX / np.maximum(sp, 1e-9), 1.0)
        vx, vy = vx * scale, vy * scale
        # first sample of each run (for the fade-in)
        starts = ~lprev
        rs = ta[np.maximum.accumulate(np.where(starts, idx, 0))]
        self.link, self.gap = link.tolist(), gap.tolist()
        self.vx, self.vy, self.run_start = vx.tolist(), vy.tolist(), rs.tolist()
        # time to the nearest measured sample within the run (interpolated stretches fade with it)
        meas = np.asarray(f) == 1
        if meas.all():
            self.dm = [0.0] * n
            return
        rs_i = np.maximum.accumulate(np.where(starts, idx, 0))  # first sample of k's run
        re_i = np.minimum.accumulate(np.where(~lnext, idx, n)[::-1])[::-1]  # last sample of k's run
        last_m = np.maximum.accumulate(np.where(meas, idx, -1))
        next_m = np.minimum.accumulate(np.where(meas, idx, n)[::-1])[::-1]
        dp = np.where(last_m >= rs_i, ta - ta[np.maximum(last_m, 0)], np.inf)
        dn = np.where(next_m <= re_i, ta[np.minimum(next_m, n - 1)] - ta, np.inf)
        self.dm = np.minimum(dp, dn).tolist()

    def _seg(self, k: int, t: float):
        """(x, y, vx, vy, flag, alpha) on the linked segment k -> k+1 at time t."""
        tk, tn = self.t[k], self.t[k + 1]
        dt = tn - tk
        u = (t - tk) / dt
        p0, p1 = (self.x[k], self.y[k]), (self.x[k + 1], self.y[k + 1])
        cx, cy = (p1[0] - p0[0]) / dt, (p1[1] - p0[1]) / dt
        lim = max(2.0 * math.hypot(cx, cy), 0.5)  # tangents never more than 2x the chord speed: no loops
        m0 = _limit(self.vx[k], self.vy[k], lim)
        m1 = _limit(self.vx[k + 1], self.vy[k + 1], lim)
        x, y, vx, vy = _hermite(p0, p1, m0, m1, dt, u)
        flag = 2 if self.gap[k] else (self.f[k] if u < 0.5 else self.f[k + 1])
        dmt = min(self.dm[k] + t - tk, self.dm[k + 1] + tn - t)  # time to the nearest measurement
        alpha = 1.0
        if dmt > INTERP_FULL_S:
            alpha = max(INTERP_MIN_ALPHA, 1.0 - (dmt - INTERP_FULL_S) / 1.5 * (1.0 - INTERP_MIN_ALPHA))
        return x, y, vx, vy, flag, alpha

    def at(self, t: float, hold_s: float = 4.0, fade_s: float = 0.25) -> list[tuple]:
        """Drawable states at time t: [(x, y, flag, alpha, vx, vy)] - empty, one, or two (cross-fade between the
        end of one run and the start of the next). flag: 1 measured, 2 interpolated / held."""
        ts = self.t
        n = len(ts)
        if n == 0 or t < self.t0 - fade_s or t > self.t1 + max(hold_s, fade_s):
            return []
        k = bisect.bisect_right(ts, t) - 1
        out = []
        if 0 <= k < n - 1 and self.link[k]:
            x, y, vx, vy, flag, alpha = self._seg(k, t)
            rs = self.run_start[k]
            alpha *= min(1.0, (t - rs) / fade_s + 0.5) if fade_s > 0 else 1.0
            out.append((x, y, flag, alpha, vx, vy))
            return out
        if k >= 0:
            # after the end of a run (or exactly at a lone sample): coast to a stop and fade out
            dt = max(0.0, t - ts[k])
            vx, vy = self.vx[k], self.vy[k]
            span = hold_s if hold_s > 0 else fade_s
            if dt <= 1e-6 or (span > 0 and dt < span):
                alpha = 1.0 if dt <= 1e-6 else ((1.0 - dt / span) ** 2 if hold_s > 0 else 1.0 - dt / span)
                if self.dm[k] > INTERP_FULL_S:
                    alpha *= max(INTERP_MIN_ALPHA, 1.0 - (self.dm[k] - INTERP_FULL_S) / 1.5 * (1 - INTERP_MIN_ALPHA))
                if fade_s > 0:  # continuing the fade-in of a run that only just started
                    alpha *= min(1.0, (t - self.run_start[k]) / fade_s + 0.5)
                sp = math.hypot(vx, vy)
                c = COAST_TAU * (1.0 - math.exp(-dt / COAST_TAU))
                if sp * c > COAST_MAX:
                    c = COAST_MAX / sp
                decay = math.exp(-dt / COAST_TAU)
                ex = self.ext(k, dt) if (self.ext is not None and dt > 1e-6) else None
                if ex is not None:  # joint extrapolation: moves with the play, fades as it gets unsure
                    alpha = honest_alpha(ex[2]) * min(1.0, max(0.0, (span - dt) / 0.5) if span > 0 else 1.0)
                    if fade_s > 0:
                        alpha *= min(1.0, (t - self.run_start[k]) / fade_s + 0.5)
                    if alpha > 0.01:
                        e2 = self.ext(k, dt + 0.1) or ex
                        out.append((ex[0], ex[1], 2, alpha, (e2[0] - ex[0]) / 0.1, (e2[1] - ex[1]) / 0.1))
                elif alpha > 0.01:
                    out.append((self.x[k] + vx * c, self.y[k] + vy * c, self.f[k] if dt <= 1e-6 else 2, alpha,
                                vx * decay, vy * decay))
        if k + 1 < n and (k < 0 or not self.link[k]):
            # before the start of the next run: fade in at its first position
            lead = ts[k + 1] - t
            if fade_s > 0 and lead < fade_s * 0.5:
                alpha = 0.5 - lead / fade_s
                if alpha > 0.01:
                    out.append((self.x[k + 1], self.y[k + 1], self.f[k + 1], alpha, self.vx[k + 1], self.vy[k + 1]))
        return out

    def unseen(self, t: float) -> float:
        """Seconds from t to the nearest measured (seen) sample of the drawn stretch: 0 while seen, growing across
        a bridged gap (largest in its middle) and after the last sighting. inf without samples."""
        ts = self.t
        n = len(ts)
        if n == 0:
            return math.inf
        k = bisect.bisect_right(ts, t) - 1
        if 0 <= k < n - 1 and self.link[k]:
            return max(0.0, min(self.dm[k] + t - ts[k], self.dm[k + 1] + ts[k + 1] - t))
        if k >= 0:
            return self.dm[k] + max(0.0, t - ts[k])
        return self.dm[0] + max(0.0, ts[0] - t)

    def _sig_k(self, k: int) -> float:
        if self.sig is not None and math.isfinite(self.sig[k]):
            return self.sig[k]
        return min(SIG_MAX, SIG_SEEN + SIG_RATE * self.dm[k]) if self.f[k] != 1 else SIG_SEEN

    def uncertainty(self, t: float) -> float:
        """1-sigma position uncertainty (m) at time t: the data's value ("u") between samples, else an estimate
        growing with the time since / until the nearest measurement (bridged gaps grow towards their middle,
        coasting after the last sighting grows with the time since it)."""
        ts = self.t
        n = len(ts)
        if n == 0:
            return SIG_MAX
        k = bisect.bisect_right(ts, t) - 1
        if k < 0:
            return self._sig_k(0)
        if k < n - 1 and self.link[k]:
            tk, tn = ts[k], ts[k + 1]
            a, b = self._sig_k(k), self._sig_k(k + 1)
            if self.gap[k]:  # bridged gap: unknown in between
                return min(SIG_MAX, max(a, b) + SIG_RATE * min(t - tk, tn - t))
            u = (t - tk) / (tn - tk)
            return a + (b - a) * u
        return min(SIG_MAX, self._sig_k(k) + SIG_RATE * max(0.0, t - ts[k]))

    def _dense_points(self):
        """(times, xs, ys, piece starts) of the whole trajectory for tracers: the samples plus extra points inside
        bridged gaps so they are drawn curved; built once."""
        d = self._dense
        if d is None:
            ts = self.t
            T, X, Y, starts = [], [], [], []
            for k in range(len(ts)):
                if k == 0 or not self.link[k - 1]:
                    starts.append(len(T))
                T.append(ts[k])
                X.append(self.x[k])
                Y.append(self.y[k])
                if k < len(ts) - 1 and self.link[k] and self.gap[k]:
                    m = max(2, int((ts[k + 1] - ts[k]) / self.step))
                    for j in range(1, m):
                        tm = ts[k] + (ts[k + 1] - ts[k]) * j / m
                        p = self._seg(k, tm)
                        T.append(tm)
                        X.append(p[0])
                        Y.append(p[1])
            starts.append(len(T))
            d = self._dense = (T, X, Y, starts)
        return d

    def points(self, ta: float, tb: float) -> list[tuple[list[float], list[float], list[float]]]:
        """Polyline pieces (times, xs, ys) of the trajectory between ta and tb (for tracers), split where runs
        break."""
        if not self.t or tb < self.t0 or ta > self.t1:
            return []
        T, X, Y, starts = self._dense_points()
        j0, j1 = bisect.bisect_left(T, ta), bisect.bisect_right(T, tb)
        if j1 - j0 < 1:
            return []
        out = []
        r = bisect.bisect_right(starts, j0) - 1
        a = j0
        while a < j1:
            b = min(j1, starts[r + 1])
            if b - a >= 1:
                out.append((T[a:b], X[a:b], Y[a:b]))
            a, r = b, r + 1
        return out


def _box_smooth(xs: np.ndarray, ys: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Average over the available samples within +/- n (same as the old display smoothing)."""
    ok = np.isfinite(xs) & np.isfinite(ys)
    m = len(xs)
    cx = np.concatenate([[0.0], np.cumsum(np.where(ok, xs, 0.0))])
    cy = np.concatenate([[0.0], np.cumsum(np.where(ok, ys, 0.0))])
    cn = np.concatenate([[0], np.cumsum(ok)])
    idx = np.arange(m)
    lo, hi = np.clip(idx - n, 0, m), np.clip(idx + n + 1, 0, m)
    cnt = np.maximum(cn[hi] - cn[lo], 1)
    sx, sy = (cx[hi] - cx[lo]) / cnt, (cy[hi] - cy[lo]) / cnt
    return np.where(ok, sx, np.nan), np.where(ok, sy, np.nan)


_NAN = float("nan")


def _floats(v: list) -> np.ndarray:
    """float array of a JSON list with nulls (None -> NaN); 2-3x faster than np.array(v, dtype=float), same values."""
    if None not in v:
        return np.fromiter(v, float, len(v))
    return np.array([_NAN if x is None else x for x in v], float)


def _flags(tr: dict) -> np.ndarray | None:
    """Per-sample flag: 1 seen, 2 not seen but positioned (predicted / interpolated / rejected / occluded), 0 no
    position. From the per-sample state "st" (smoothing.py: 1 seen, 2 predicted, 3 rejected, 4 occluded) when
    present, else "f" (1 measured, 2 interpolated); None if neither."""
    fr = tr.get("f")
    st = tr.get("st")
    if isinstance(st, list) and st:
        sv = np.nan_to_num(_floats(st))
        out = np.where(sv == 1, 1.0, np.where(sv > 1, 2.0, 0.0))
        if fr:  # samples without a state keep their "f"
            fv = np.nan_to_num(_floats(fr))
            res = np.zeros(max(len(fv), len(out)))
            res[:len(fv)] = fv
            res[:len(out)] = np.where(sv > 0, out, res[:len(out)])
            return res
        return out
    if fr:
        return np.nan_to_num(_floats(fr))
    return None


def _sigmas(tr: dict) -> np.ndarray | None:
    """Per-sample 1-sigma position uncertainty in metres from the track's "u" (centimetres, like positions) or
    "s" / "sigma" (centimetres; values that all look like metres - none above 5 - are taken as metres)."""
    for key in ("u", "s", "sigma"):
        v = tr.get(key)
        if isinstance(v, list) and v:
            a = _floats(v)  # None -> nan
            fin = a[np.isfinite(a)]
            if key != "u" and len(fin) and fin.max() <= 5.0:
                return a
            return a / 100.0
    return None


def build_path(tr: dict, ts, step: float, smooth_n: int = 0, bridge: float = BRIDGE_MAX) -> TrackPath:
    """TrackPath of a replay_data track (positions in cm, sample j at global sample i0 + j); ts: sample times
    (list or array)."""
    i0 = int(tr.get("i0", 0))
    ts = np.asarray(ts, float)
    m = min(len(tr["x"]), max(0, len(ts) - i0))
    xs = _floats(tr["x"][:m]) / 100.0  # None -> nan
    ys = _floats(tr["y"][:m]) / 100.0
    if smooth_n > 0:
        xs, ys = _box_smooth(xs, ys, smooth_n)
    fl = np.zeros(m, float)
    fr = _flags(tr)
    if fr is not None:
        fv = fr[:m]
        fl[:len(fv)] = fv
    else:
        fl[:] = 1
    ok = np.flatnonzero(np.isfinite(xs) & np.isfinite(ys))
    t = ts[i0 + ok]
    if len(t) > 1:  # strictly increasing times (duplicate timestamps would divide by zero)
        keep = np.r_[True, np.diff(t) > 1e-6]
        ok, t = ok[keep], t[keep]
    f = np.where(fl[ok] == 1, 1, 2).astype(int)
    sig = None
    uv = _sigmas(tr)
    if uv is not None:
        u = np.full(m, np.nan)
        uv = uv[:m]
        u[:len(uv)] = uv
        sig = np.clip(u[ok], 0.05, SIG_MAX).tolist()
    return TrackPath(t.tolist(), xs[ok].tolist(), ys[ok].tolist(), f.tolist(), step, bridge, sig)


def _retime_fast_hops(t, x, y, h, f, step, vmax: float = BALL_MAX_SPEED, max_drop_s: float = 0.6):
    """The ball data sometimes jumps several metres between two consecutive samples (a pass whose release and
    catch were detected late / early). Drop knots next to such a hop - preferring estimated ones, then the end
    of the passer's possession or the start of the receiver's - until the flight has time to happen at a
    plausible speed; the unknown stretch is then drawn as "held, then a pass" (BallPath.at)."""
    t, x, y, h, f = list(t), list(x), list(y), list(h), list(f)
    k = 0
    while k < len(t) - 1:
        dt = t[k + 1] - t[k]
        d = math.hypot(x[k + 1] - x[k], y[k + 1] - y[k])
        if d <= vmax * dt + 0.3:
            k += 1
            continue
        # candidates: the knot before k (making k's predecessor the release) or after k+1
        def lone(j):  # a one-sample possession between others: a holder blip, not a real catch
            return h[j] is not None and (j == 0 or h[j - 1] != h[j]) and (j == len(t) - 1 or h[j + 1] != h[j])

        drop = None
        lo_ok = k > 0 and t[k + 1] - t[k - 1] <= dt + max_drop_s and (h[k] is None or h[k - 1] == h[k] or lone(k))
        hi_ok = k + 2 < len(t) and t[k + 2] - t[k] <= dt + max_drop_s and (
            h[k + 1] is None or h[k + 2] == h[k + 1] or lone(k + 1))
        if hi_ok and lone(k + 1) and not (lo_ok and lone(k)):
            drop = k + 1
        elif lo_ok and (h[k] is None or f[k] != 1 or lone(k) or not hi_ok):
            drop = k
        elif hi_ok:
            drop = k + 1
        elif lo_ok:
            drop = k
        if drop is None:
            k += 1
            continue
        for a in (t, x, y, h, f):
            del a[drop]
        k = max(0, drop - 1)
    return t, x, y, h, f


class BallPath:
    """Continuous ball position: with its holder while possessed, smooth flights between, kept (then faded)
    while unknown."""

    def __init__(self, ball: dict, ts: list[float], step: float, raw_paths: dict[str, TrackPath],
                 sample_xy=None):
        """sample_xy(holder id, sample index) -> (x, y) | None: the holder's stored position at a sample (used
        instead of building the holder's whole trajectory - much faster on a full game)."""
        bx, by, bf = ball.get("x") or [], ball.get("y") or [], ball.get("f") or []
        bo = ball.get("o") or [None] * len(bx)
        t, x, y, h, f, si = [], [], [], [], [], []
        for i in range(min(len(bx), len(ts))):
            if bx[i] is None or by[i] is None or not (bf[i] if i < len(bf) else 0):
                continue
            if t and ts[i] <= t[-1] + 1e-6:
                continue
            t.append(float(ts[i]))
            si.append(i)
            x.append(bx[i] / 100.0)
            y.append(by[i] / 100.0)
            f.append(int(bf[i]))
            o = bo[i] if i < len(bo) and bf[i] == 1 else None
            h.append(None if o is None else str(o))
        n = len(t)
        consec = 1.6 * step

        # estimated samples that sit on the holder's position are still "with the holder" (ball.py follows the
        # holder while the ball is hidden in their hands); infer that forwards and backwards
        def on(hid, k):
            if sample_xy is not None:
                q = sample_xy(hid, si[k])
                if q is not None:
                    return math.hypot(q[0] - x[k], q[1] - y[k]) < 0.25
            p = raw_paths.get(hid)  # no stored sample there (a bridged gap): ask the trajectory
            if p is None:
                return False
            s = p.at(t[k], 0.0, 0.0)
            return bool(s) and math.hypot(s[0][0] - x[k], s[0][1] - y[k]) < 0.25

        for rng, nb in ((range(1, n), -1), (range(n - 2, -1, -1), 1)):
            for k in rng:
                if h[k] is None and h[k + nb] is not None and abs(t[k] - t[k + nb]) <= consec and on(h[k + nb], k):
                    h[k] = h[k + nb]
        t, x, y, h, f = _retime_fast_hops(t, x, y, h, f, step)
        n = len(t)
        self.t, self.x, self.y, self.h, self.f, self.step = t, x, y, h, f, step
        self.t0, self.t1 = (t[0], t[-1]) if n else (math.inf, -math.inf)
        # flights: stretches without a holder between two anchors, with a small "height" bump (arc)
        baskets = court.baskets()
        rim = [any(math.hypot(x[k] - b[0], y[k] - b[1]) < 0.4 for b in baskets) for k in range(n)]
        self.flights: list[tuple[float, float, float]] = []
        k = 0
        while k < n:
            if h[k] is None and not rim[k]:
                a = k
                while k < n and h[k] is None and not rim[k]:
                    k += 1
                ta = t[a - 1] if a > 0 else t[a]
                tb = t[k] if k < n else t[k - 1]
                dur = tb - ta
                if 0.2 <= dur <= 3.5 and a > 0 and k < n and tb - ta < 4.0:
                    amp = 0.55 if rim[k] else min(0.3, 0.08 + 0.2 * dur)
                    self.flights.append((ta, tb, amp))
            else:
                k += 1
        self.flight_t0 = [fl[0] for fl in self.flights]

    def _point(self, k: int, pos_of) -> tuple[float, float]:
        """Knot k's position: the holder's current trajectory position when held (so the ball moves with the
        drawn dot), else the stored ball position."""
        hid = self.h[k]
        if hid is not None:
            p = pos_of(hid, self.t[k])
            if p is not None:
                return p
        return self.x[k], self.y[k]

    def _held(self, hid, t: float, pos_of, fallback):
        p = pos_of(hid, t)
        return p if p is not None else fallback

    def at(self, t: float, pos_of) -> tuple | None:
        """(x, y, alpha, est, holder, hand, lift) at time t; pos_of(holder_id, t) -> (x, y) | None.

        est: position estimated (not seen); hand: 0..1 how much the ball sits in the holder's hand (draw it on the
        dot's rim), lift: 0..~0.55 extra apparent size during a pass / shot arc."""
        ts = self.t
        n = len(ts)
        if n == 0 or t < self.t0 - 0.3 or t > self.t1 + BALL_GAP:
            return None
        k = bisect.bisect_right(ts, t) - 1
        if k < 0:  # before the first knot: fade in at it
            px, py = self._point(0, pos_of)
            return px, py, 1.0 - (self.t0 - t) / 0.3, True, self.h[0], 1.0 if self.h[0] else 0.0, 0.0
        if k == n - 1:  # after the last knot: stay with the holder / spot, then fade
            dt = t - ts[k]
            hid = self.h[k]
            p = self._held(hid, t, pos_of, (self.x[k], self.y[k])) if hid else (self.x[k], self.y[k])
            alpha = 1.0 if dt < BALL_GAP - 1.0 else max(0.0, BALL_GAP - dt)
            return p[0], p[1], alpha, dt > self.step, hid, 1.0 if hid else 0.0, 0.0
        tk, tn = ts[k], ts[k + 1]
        dt = tn - tk
        ha, hb = self.h[k], self.h[k + 1]
        lift = self._lift(t)
        if ha is not None and ha == hb:  # possessed (also across an unknown stretch: it stays in their hands)
            u = (t - tk) / dt
            fb = (self.x[k] + (self.x[k + 1] - self.x[k]) * u, self.y[k] + (self.y[k + 1] - self.y[k]) * u)
            p = self._held(ha, t, pos_of, fb)
            return p[0], p[1], 1.0, self.f[k] != 1 or self.f[k + 1] != 1, ha, 1.0, 0.0
        if dt <= 1.6 * self.step:
            # consecutive knots: smooth curve through the neighbouring knots (Catmull-Rom)
            p0, p1 = self._point(k, pos_of), self._point(k + 1, pos_of)
            pm = self._point(k - 1, pos_of) if k > 0 and tk - ts[k - 1] <= 1.6 * self.step else None
            pp = self._point(k + 2, pos_of) if k + 2 < n and ts[k + 2] - tn <= 1.6 * self.step else None
            m0 = ((p1[0] - pm[0]) / (tn - ts[k - 1]), (p1[1] - pm[1]) / (tn - ts[k - 1])) if pm else \
                ((p1[0] - p0[0]) / dt, (p1[1] - p0[1]) / dt)
            m1 = ((pp[0] - p0[0]) / (ts[k + 2] - tk), (pp[1] - p0[1]) / (ts[k + 2] - tk)) if pp else \
                ((p1[0] - p0[0]) / dt, (p1[1] - p0[1]) / dt)
            x, y, _, _ = _hermite(p0, p1, m0, m1, dt, (t - tk) / dt)
            hand = self._hand(t, k)
            return x, y, 1.0, self.f[k] != 1 and self.f[k + 1] != 1, ha if hand > 0.5 else None, hand, lift
        # an unknown stretch between two knots: stay with the holder / at the spot, then travel to the next knot
        a_fb = (self.x[k], self.y[k])
        b = self._point(k + 1, pos_of)
        if dt > BALL_GAP:
            out_end = min(tk + BALL_GAP, tn - 0.3)  # faded out before the next sighting fades in
            alpha_out = min(1.0, max(0.0, out_end - t))
            lead = tn - t
            if lead < 0.3:
                return b[0], b[1], 1.0 - lead / 0.3, True, hb, 1.0 if hb else 0.0, 0.0
            p = self._held(ha, t, pos_of, a_fb) if ha else a_fb
            return p[0], p[1], alpha_out, True, ha, 1.0 if ha else 0.0, 0.0
        a_rel = self._held(ha, tk, pos_of, a_fb) if ha else a_fb
        flight = min(dt, max(0.25, math.hypot(b[0] - a_rel[0], b[1] - a_rel[1]) / BALL_PASS_SPEED))
        tr = tn - flight
        if t <= tr:
            p = self._held(ha, t, pos_of, a_fb) if ha else a_fb
            return p[0], p[1], 1.0, True, ha, 1.0 if ha else 0.0, 0.0
        a = self._held(ha, tr, pos_of, a_fb) if ha else a_fb
        u = (t - tr) / flight
        e = 0.5 * u + 0.5 * u * u * (3 - 2 * u)  # mostly constant speed, softened at the ends
        hand = max(0.0, 1.0 - (t - tr) / HAND_RAMP) if ha else 0.0
        hand = max(hand, max(0.0, 1.0 - (tn - t) / HAND_RAMP) if hb else 0.0)
        bump = 0.25 * math.sin(math.pi * u) if flight >= 0.4 else 0.0
        return (a[0] + (b[0] - a[0]) * e, a[1] + (b[1] - a[1]) * e, 1.0, True, None, hand, max(lift, bump))

    def _hand(self, t: float, k: int) -> float:
        """1 with a holder, ramping to 0 within HAND_RAMP of the release / catch."""
        ha, hb = self.h[k], self.h[k + 1]
        v = 0.0
        if ha is not None:
            v = max(v, 1.0 - (t - self.t[k]) / HAND_RAMP)
        if hb is not None:
            v = max(v, 1.0 - (self.t[k + 1] - t) / HAND_RAMP)
        return min(1.0, max(0.0, v))

    def _lift(self, t: float) -> float:
        j = bisect.bisect_right(self.flight_t0, t) - 1
        if j < 0:
            return 0.0
        ta, tb, amp = self.flights[j]
        if not ta <= t <= tb:
            return 0.0
        return amp * math.sin(math.pi * (t - ta) / (tb - ta))


BALL_CHUNK_S = 60.0  # long replays: the ball path is built per minute ...
BALL_CHUNK_PAD_S = 8.0  # ... from the samples within this margin around it (> BALL_GAP: same result inside)
BALL_CHUNK_MIN_SAMPLES = 6000  # (~10 min at 10 samples/s)


class ChunkedBall:
    """Ball path of a long replay, built per BALL_CHUNK_S window on first use (each window from the ball
    samples within BALL_CHUNK_PAD_S around it), so opening a full game does not build the whole ball path."""

    def __init__(self, model: "ReplayModel"):
        self.m = model
        self._chunks: dict[int, BallPath | None] = {}

    def chunk(self, c: int):
        if c in self._chunks:
            return self._chunks[c]
        m = self.m
        ts = m._ts_arr
        a = int(np.searchsorted(ts, c * BALL_CHUNK_S - BALL_CHUNK_PAD_S))
        b = int(np.searchsorted(ts, (c + 1) * BALL_CHUNK_S + BALL_CHUNK_PAD_S, side="right"))
        ball = m.data.get("ball") or {}
        part = {k: v[a:b] for k, v in ball.items() if isinstance(v, list)}
        bp = None
        if b > a and any(v is not None for v in part.get("x") or []):
            bp = BallPath(part, m.ts[a:b], m.step, m.paths(0), lambda hid, i: m.sample_xy(hid, i + a))
        if len(self._chunks) > 64:
            self._chunks.clear()
        self._chunks[c] = bp
        return bp

    def at(self, t: float, pos_of):
        bp = self.chunk(int(math.floor(t / BALL_CHUNK_S)))
        return None if bp is None else bp.at(t, pos_of)


class LazyPaths:
    """TrackPaths of all tracks (one smoothing window), each built on first use: a full game has thousands of
    tracks but only a few dozen are alive at any time (building all took 4-15 s on the GUI thread).
    Indexable like a list; ``get(track_id)`` like a dict (used by the ball); ``build_some`` warms it up in the
    background in small time slices."""

    def __init__(self, model: "ReplayModel", smooth_n: int, bridge: float):
        self.m, self.smooth_n, self.bridge = model, smooth_n, bridge
        self._p: list[TrackPath | None] = [None] * len(model.tracks)
        self._cursor = 0

    def __len__(self) -> int:
        return len(self._p)

    def __getitem__(self, i: int) -> TrackPath:
        p = self._p[i]
        if p is None:
            m = self.m
            p = self._p[i] = build_path(m.tracks[i], m._ts_arr, m.step, self.smooth_n, self.bridge)
            if p.t and joint_2d_enabled():
                p.ext = _lazy_ext(m, i, p)
        return p

    def __iter__(self):
        for i in range(len(self._p)):
            yield self[i]

    def get(self, track_id, default=None):
        i = self.m.index_of(track_id)
        return default if i is None else self[i]

    @property
    def complete(self) -> bool:
        return self._cursor >= len(self._p)

    def build_some(self, budget_s: float = 0.008, near: float | None = None, dense: bool = False) -> bool:
        """Build paths for about budget_s seconds (those alive around time ``near`` first; with dense also their
        tracer polylines); True when all are built."""
        import time as _time
        t_end = _time.perf_counter() + budget_s
        if near is not None:
            for i in self.m.alive(near, 5.0, 20.0):
                p = self._p[i]
                if p is None or (dense and p._dense is None):
                    p = self[i]
                    if dense:
                        p._dense_points()
                    if _time.perf_counter() > t_end:
                        return False
        n = len(self._p)
        while self._cursor < n:
            p = self._p[self._cursor]
            if p is None:
                p = self[self._cursor]
            if dense and p._dense is None:
                p._dense_points()
            self._cursor += 1
            if _time.perf_counter() > t_end:
                break
        return self._cursor >= n


class ReplayModel:
    """All trajectories of one replay_data dict; paths are built lazily (per track, per smoothing window) and
    cached; ``alive(t)`` finds the tracks that can be visible at t without building anything."""

    def __init__(self, data: dict):
        self.data = data
        self.ts = [float(v) for v in data.get("t") or []]
        self._ts_arr = np.asarray(self.ts, float)
        self.step = sample_dt(self.ts)
        self.tracks = data.get("tracks") or []
        self._paths: dict[tuple[int, float], LazyPaths] = {}
        self._ball: BallPath | None = None
        self._ball_built = False
        self._span = None
        self._index: dict[str, int] | None = None

    def paths(self, smooth_n: int = 0, bridge: float = BRIDGE_MAX) -> LazyPaths:
        key = (smooth_n, round(bridge, 2))
        ps = self._paths.get(key)
        if ps is None:
            if len(self._paths) > 8:
                self._paths.clear()
            ps = LazyPaths(self, smooth_n, bridge)
            self._paths[key] = ps
        return ps

    def spans(self) -> tuple[np.ndarray, np.ndarray]:
        """(first, last) time with a position of every track (inf / -inf if none); computed once, cheap."""
        if self._span is None:
            n = len(self.tracks)
            a, b = np.full(n, np.inf), np.full(n, -np.inf)
            ts, nt = self.ts, len(self.ts)
            for k, tr in enumerate(self.tracks):
                xs, i0 = tr.get("x") or [], int(tr.get("i0", 0))
                m = min(len(xs), max(0, nt - i0))
                j0 = next((j for j in range(m) if xs[j] is not None), None)
                if j0 is None:
                    continue
                j1 = next(j for j in range(m - 1, -1, -1) if xs[j] is not None)
                a[k], b[k] = ts[i0 + j0], ts[i0 + j1]
            self._span = (a, b)
        return self._span

    def alive(self, t: float, before: float, after: float) -> list[int]:
        """Indices (in track order) of the tracks with data between t - after and t + before, i.e. that may be
        drawn at t when they fade in ``before`` s ahead of their first and out ``after`` s after their last
        position."""
        a, b = self.spans()
        return np.flatnonzero((a - before <= t) & (t <= b + after)).tolist()

    def index_of(self, track_id) -> int | None:
        if self._index is None:
            self._index = {str(tr["id"]): k for k, tr in enumerate(self.tracks)}
        return self._index.get(str(track_id))

    def ball(self):
        """BallPath of the whole replay, or for long replays (a full game: ~1-2 s to build at once) a
        ChunkedBall that builds the ball path per minute on first use."""
        if not self._ball_built:
            b = self.data.get("ball")
            if b and b.get("x"):
                if len(self.ts) > BALL_CHUNK_MIN_SAMPLES:
                    self._ball = ChunkedBall(self)
                else:
                    self._ball = BallPath(b, self.ts, self.step, self.paths(0), self.sample_xy)
            self._ball_built = True
        return self._ball

    def ball_ready(self) -> bool:
        return self._ball_built

    def joint_ready(self) -> "JointDisplay | None":
        """The joint index if it is already built (never builds it: the GUI thread must not stall)."""
        return getattr(self, "_joint", None) if joint_2d_enabled() else None

    def joint(self) -> "JointDisplay | None":
        """Joint extrapolation of unseen players for the display (None when switched off, see JOINT_2D).
        Builds the index (~0.8 s for a full game): call from a worker thread (ReplayCourt._prepare)."""
        if not joint_2d_enabled():
            return None
        jd = getattr(self, "_joint", None)
        if jd is None:
            jd = self._joint = JointDisplay(self)
        return jd

    def sample_xy(self, track_id, i: int) -> tuple[float, float] | None:
        """Stored position (m) of a track at global sample i, or None."""
        k = self.index_of(track_id)
        if k is None:
            return None
        tr = self.tracks[k]
        j = i - int(tr.get("i0", 0))
        xs, ys = tr.get("x") or [], tr.get("y") or []
        if 0 <= j < len(xs) and xs[j] is not None and ys[j] is not None:
            return xs[j] / 100.0, ys[j] / 100.0
        return None

    def quality_ready(self) -> bool:
        return getattr(self, "_quality", None) is not None


    # ---------------------------------------------------------------- confidence (display)
    def meta(self) -> dict[str, dict]:
        """Per track id: {"label": text on the dot, "uncertain": team / identity doubtful, "slot", "number",
        "name", "numbered"} from track_info (jersey numbers, lineup slots), computed once."""
        m = getattr(self, "_meta", None)
        if m is None:
            m = self._meta = track_meta(self.data)
        return m

    def quality(self) -> dict:
        q = getattr(self, "_quality", None)
        if q is None:
            q = self._quality = tracking_quality(self.data)
        return q


TEAM_UNCERTAIN = 0.5  # team_confidence below this: "team uncertain" marker
ID_UNCERTAIN = 0.5  # identity confidence (track_info "id_confidence" / "identity_confidence") below this


def _slot_index(slot) -> int | None:
    """Slot number from a slot value like 3, "3", "A3", "B-3"; None if not recognisable."""
    if slot is None or isinstance(slot, bool):
        return None
    digits = "".join(ch for ch in str(slot) if ch.isdigit())
    return int(digits) if digits else None


def _slot_of(data: dict) -> tuple[dict, dict]:
    """({track id: slot}, {slot id: slot doc}) from track_info "slot" and / or data["lineup"]["slots"]."""
    info = data.get("track_info") or {}
    lineup = data.get("lineup") if isinstance(data.get("lineup"), dict) else {}
    docs, slot_of = {}, {}
    for s in lineup.get("slots") or []:
        if isinstance(s, dict) and s.get("id") is not None:
            docs[str(s["id"])] = s
            for tid in s.get("tracks") or []:
                slot_of[str(tid)] = s["id"]
    for tid, ti in info.items():
        if isinstance(ti, dict) and ti.get("slot") is not None:
            slot_of[str(tid)] = ti["slot"]
    return slot_of, docs


def track_meta(data: dict) -> dict[str, dict]:
    info = data.get("track_info") or {}
    slot_of, docs = _slot_of(data)
    out = {}
    for tr in data.get("tracks") or []:
        tid = str(tr["id"])
        ti = info.get(tid) or {}
        slot = slot_of.get(tid)
        sd = docs.get(str(slot), {}) if slot is not None else {}
        number = ti.get("number") or sd.get("number")
        name = ti.get("name") or sd.get("name")
        conf = ti.get("team_confidence")
        idc = ti.get("id_confidence", ti.get("identity_confidence"))
        uncertain = tr.get("team") in ("A", "B") and (
            (isinstance(conf, (int, float)) and conf < TEAM_UNCERTAIN) or
            (isinstance(idc, (int, float)) and idc < ID_UNCERTAIN) or bool(ti.get("team_uncertain")))
        numbered = number not in (None, "")
        pid = tr.get("pid", ti.get("player_id"))
        if numbered:
            label = str(number)
        elif pid not in (None, ""):
            label = f"P{pid}"  # continuous player id (export.py / lineup.player_id_of): survives track-id breaks
        elif slot is not None and _slot_index(slot) is not None:
            label = f"P{_slot_index(slot)}"  # lineup slot without a known number
        else:
            label = tid
        # jersey number per time span of a lineup slot (players change on substitutions): [(t0, t1, text)]
        spans = []
        for pc in sd.get("pieces") or []:
            try:
                a, b, num = float(pc[0]), float(pc[1]), pc[2]
            except (TypeError, ValueError, IndexError):
                continue
            if num not in (None, "", -1):
                spans.append((a, b, str(num)))
        spans.sort()
        out[tid] = {"label": label, "uncertain": uncertain, "slot": slot, "number": number, "name": name,
                    "numbered": numbered, "spans": spans, "span_t0": [sp[0] for sp in spans], "pid": pid}
    return out


def label_at(mt: dict, t: float, tol: float = 2.0) -> tuple[str, bool]:
    """(text, is_jersey_number) for a track's dot at time t: a lineup slot's number for the piece at t (or the
    nearest piece within tol seconds), else the track's own label."""
    spans = mt.get("spans")
    if spans:
        j = bisect.bisect_right(mt["span_t0"], t) - 1
        best, bd = None, tol
        for k in (j, j + 1):
            if 0 <= k < len(spans):
                a, b, num = spans[k]
                d = 0.0 if a <= t <= b else min(abs(t - a), abs(t - b))
                if d <= bd:
                    best, bd = num, d
        if best is not None:
            return best, True
    return mt["label"], mt["numbered"]


def lineup_from_file(doc: dict, data: dict) -> dict | None:
    """replay_data "lineup" block from results/lineup.json (lineup.py: {"slots": {"A1": {"id": track id, "team",
    "numbers", "pieces": [[piece tid, t0, t1, number]]}}}); None when it does not match the data's tracks."""
    slots = (doc or {}).get("slots")
    if not isinstance(slots, dict):
        return None
    ids = {str(tr["id"]) for tr in data.get("tracks") or []}
    out = []
    for label, sd in slots.items():
        if not isinstance(sd, dict) or sd.get("id") is None or str(sd["id"]) not in ids:
            continue
        nums = [n for n in sd.get("numbers") or [] if n is not None and n != -1]
        pieces = [[pc[1], pc[2], (str(pc[3]) if pc[3] is not None and pc[3] >= 0 else None)]
                  for pc in sd.get("pieces") or [] if isinstance(pc, list) and len(pc) >= 4]
        out.append({"id": label, "team": sd.get("team") or label[:1], "tracks": [sd["id"]],
                    "number": str(nums[0]) if len(nums) == 1 else None, "pieces": pieces})
    return {"slots": out} if out else None


# tracking quality levels (timeline lane), worst last
Q_GOOD, Q_LINEUP, Q_FEW, Q_CALIB_POOR, Q_CALIB_NONE = 0, 1, 2, 3, 4
Q_NAMES = {Q_GOOD: "good", Q_LINEUP: "lineup incomplete", Q_FEW: "few players seen",
           Q_CALIB_POOR: "calibration poor", Q_CALIB_NONE: "no calibration"}
FEW_SEEN = 6  # fewer players (both teams) measured in a sample than this: "few players seen"
LOCATED_MIN = 8  # without a lineup: fewer players located (seen or predicted) than this = "lineup incomplete"
CALIB_POOR_Q = 0.5  # optional per-sample calibration quality "calib_q" (0..1) below this: "calibration poor"


def tracking_quality(data: dict) -> dict:
    """Per-sample tracking quality for the timeline lane: {"level", "seen", "located" (arrays), "stretches":
    [(t0, t1, level)], "lineup": bool}. Uses data["calib"] (0 = none), optional data["calib_q"] (0..1), the
    players measured (f == 1) / located (f > 0) per sample, and lineup slots when present (then "complete"
    means all 10 slots located)."""
    ts = np.asarray(data.get("t") or [], float)
    n = len(ts)
    if n == 0:
        z = np.zeros(0, int)
        return {"level": z, "seen": z, "located": z, "stretches": [], "lineup": False}
    seen = np.zeros(n, int)
    located = np.zeros(n, int)
    slot_of, _docs = _slot_of(data)
    slots_on: dict = {}
    for tr in data.get("tracks") or []:
        if tr.get("team") not in ("A", "B"):
            continue
        i0 = int(tr.get("i0", 0))
        f = _flags(tr)
        if f is None:
            f = np.array([0 if v is None else 1 for v in tr.get("x") or []], float)
        m = min(len(f), max(0, n - i0))
        if m <= 0:
            continue
        f = f[:m]
        seen[i0:i0 + m] += f == 1
        located[i0:i0 + m] += f > 0
        slot = slot_of.get(str(tr["id"]))
        if slot is not None:
            arr = slots_on.setdefault((tr["team"], str(slot)), np.zeros(n, bool))
            arr[i0:i0 + m] |= f > 0
    has_lineup = bool(slots_on)
    if has_lineup:
        filled = np.zeros(n, int)
        for arr in slots_on.values():
            filled += arr
        lineup_bad = filled < 10
    else:
        lineup_bad = located < LOCATED_MIN
    level = np.where(lineup_bad, Q_LINEUP, Q_GOOD)
    level = np.where(seen < FEW_SEEN, Q_FEW, level)
    cq = data.get("calib_q")
    if isinstance(cq, list) and len(cq) >= n:
        q = np.array([np.nan if v is None else v for v in cq[:n]], float)
        level = np.where(np.isfinite(q) & (q < CALIB_POOR_Q), Q_CALIB_POOR, level)
    cal = data.get("calib")
    if isinstance(cal, list) and len(cal) >= n:
        level = np.where(np.asarray(cal[:n]) == 0, Q_CALIB_NONE, level)
    # stretches of equal level (a sample covers [t_i, t_i+1)); breaks in the analysis stay empty
    step = sample_dt(ts)
    ends = np.minimum(np.r_[ts[1:], ts[-1] + step], ts + 1.6 * step)
    cut = np.flatnonzero((np.diff(level) != 0) | (np.diff(ts) > 1.6 * step)) + 1
    starts, stops = np.r_[0, cut], np.r_[cut, n]
    stretches = [(float(ts[a]), float(ends[b - 1]), int(level[a])) for a, b in zip(starts, stops)]
    return {"level": level, "seen": seen, "located": located, "stretches": stretches, "lineup": has_lineup}


_MODELS: dict[int, tuple[dict, ReplayModel]] = {}


def model_for(data: dict) -> ReplayModel:
    """Cached model for a replay_data dict (rebuilt when a different dict / track list is passed)."""
    key = id(data)
    hit = _MODELS.get(key)
    if hit is not None and hit[0] is data and hit[1].tracks is data.get("tracks"):
        return hit[1]
    if len(_MODELS) > 4:
        _MODELS.clear()
    m = ReplayModel(data)
    _MODELS[key] = (data, m)
    return m


def separate(pos: list[list[float]], weights: list[float], dmin: float, iters: int = 2) -> None:
    """Push overlapping dots apart in place (pos: [[x, y], ...] in metres) so they never slide through each
    other; displacement is shared according to weights (a faded dot gives way to a solid one)."""
    n = len(pos)
    if n < 2 or dmin <= 0:
        return
    d2min = dmin * dmin
    for _ in range(iters):
        moved = False
        for a in range(n):
            pa = pos[a]
            for b in range(a + 1, n):
                pb = pos[b]
                dx, dy = pb[0] - pa[0], pb[1] - pa[1]
                d2 = dx * dx + dy * dy
                if d2 >= d2min:
                    continue
                d = math.sqrt(d2)
                if d < 1e-6:
                    dx, dy, d = 1.0, 0.0, 1.0  # identical positions: split them sideways
                push = dmin - d
                wa, wb = weights[b], weights[a]
                s = wa + wb
                wa, wb = (0.5, 0.5) if s <= 0 else (wa / s, wb / s)
                ux, uy = dx / d, dy / d
                pa[0] -= ux * push * wa
                pa[1] -= uy * push * wa
                pb[0] += ux * push * wb
                pb[1] += uy * push * wb
                moved = True
        if not moved:
            break


class PlaybackClock:
    """Smooth display time from a media player position that only updates every ~50-70 ms.

    While playing, the time is extrapolated from the last position change with the playback rate and eased
    towards the player's clock (never backwards); paused / seeking it follows the player exactly.
    """

    def __init__(self, max_extrapolate: float = 0.3, snap: float = 0.25, gain: float = 0.12):
        self.max_extrapolate, self.snap, self.gain = max_extrapolate, snap, gain
        self.reset()

    def reset(self) -> None:
        self.anchor_pos: float | None = None
        self.anchor_wall = 0.0
        self.raw: float | None = None
        self.t: float | None = None
        self.wall = 0.0

    def update(self, pos_s: float, playing: bool, rate: float, now: float) -> float:
        if not playing:
            self.reset()
            self.raw = pos_s
            self.t = pos_s
            return pos_s
        if self.anchor_pos is None or self.raw is None or pos_s != self.raw:  # resume after pause at same pos
            self.anchor_pos, self.anchor_wall, self.raw = pos_s, now, pos_s
        pred = self.anchor_pos + min(self.max_extrapolate, max(0.0, now - self.anchor_wall)) * rate
        if self.t is None:
            self.t, self.wall = pred, now
            return pred
        run = self.t + max(0.0, now - self.wall) * rate
        err = pred - run
        if abs(err) > self.snap:
            t = pred  # a seek or a stall: jump
        else:
            t = max(self.t, run + err * self.gain)  # ease towards the player clock, never backwards
        self.t, self.wall = t, now
        return t


# ====================================================================================== coach navigation helpers
KEY_EVENT_WORDS = ("shot", "basket", "made", "miss", "free throw", "turnover", "steal", "lost ball", "bad pass",
                   "interception", "out of bounds", "violation")


def is_key_event(event_id: str | None, title: str = "") -> bool:
    """Shots / baskets / turnovers - the events the "next event" key jumps between (auto marks only)."""
    if not event_id:
        return False
    s = f"{event_id} {title}".lower().replace("_", " ")
    return any(w in s for w in KEY_EVENT_WORDS)


def next_event_time(times, t: float, direction: int = 1, eps: float = 0.3) -> float | None:
    """The first time after t (direction > 0) or before t (< 0) in ``times``, more than eps away; None if none."""
    ts = sorted(float(x) for x in times)
    if direction > 0:
        j = bisect.bisect_right(ts, t + eps)
        return ts[j] if j < len(ts) else None
    j = bisect.bisect_left(ts, t - eps) - 1
    return ts[j] if j >= 0 else None


def speed_kmh(vx: float, vy: float) -> float:
    return math.hypot(vx, vy) * 3.6


class TrailBuffer:
    """Recent positions of one moving thing (the ball) for a fading trail: cleared on seeks (time going back or
    jumping ahead by more than ``max_jump`` s)."""

    def __init__(self, keep_s: float = 2.0, max_jump: float = 0.6):
        self.keep_s, self.max_jump = keep_s, max_jump
        self.pts: list[tuple[float, float, float]] = []

    def add(self, t: float, x: float | None, y: float | None) -> None:
        if self.pts and (t < self.pts[-1][0] - 1e-6 or t - self.pts[-1][0] > self.max_jump):
            self.pts = []
        if x is not None and y is not None and (not self.pts or t > self.pts[-1][0]):
            self.pts.append((t, float(x), float(y)))
        while self.pts and self.pts[0][0] < t - self.keep_s:
            self.pts.pop(0)



# ====================================================================================== joint 2D extrapolation
# A player who is no longer seen does not stand still: he moves with the play (the players around him and the
# ball), slows down, and our knowledge of where he is fades. Display only (exported data are unaffected).
JOINT_2D = True  # ON in dev 20.10 (per-venue hide-and-predict, see data/resume/motion.md); env REPLAY_JOINT_2D=0/1 overrides
JOINT_L_M = 3.0  # neighbour weight exp(-distance at the last sighting / L)
JOINT_ALPHA = 0.6  # gain of the neighbours' common motion
JOINT_RAMP_S = 3.0  # own velocity first, the play takes over within this time
JOINT_TAU_S = 0.7  # own velocity decays with this time constant
JOINT_SIG0 = 0.3
JOINT_SIG_RATE = 1.5  # m per sqrt(s) without a sighting (hide-and-predict: 2 sigma covers ~95 % up to 10 s)
JOINT_HORIZON_S = 6.0
FADE_SIG0, FADE_SIG1 = 1.0, 2.6  # fully drawn below FADE_SIG0 m uncertainty, gone at FADE_SIG1


JOINT_BALL_GAP_S = 4.0  # unknown ball stretches up to this long are bridged linearly (passes, shots, transition)


def _fill_gaps(xy: np.ndarray, max_gap: int) -> np.ndarray:
    """Linear interpolation over nan runs of at most max_gap samples between two known samples."""
    out = xy.copy()
    ok = np.flatnonzero(np.isfinite(xy[:, 0]))
    if len(ok) < 2:
        return out
    for a, b in zip(ok[:-1], ok[1:]):
        if 1 < b - a <= max_gap + 1:
            f = ((np.arange(a + 1, b) - a) / (b - a))[:, None]
            out[a + 1:b] = xy[a] + f * (xy[b] - xy[a])
    return out


def joint_2d_enabled() -> bool:
    import os
    v = os.environ.get("REPLAY_JOINT_2D")
    return JOINT_2D if v is None else v not in ("0", "", "false", "False")


def honest_alpha(sigma: float) -> float:
    """Opacity for a position known to +-sigma m: solid while sure, gone once it could be anywhere nearby."""
    if sigma <= FADE_SIG0:
        return 1.0
    return max(0.0, 1.0 - (sigma - FADE_SIG0) / (FADE_SIG1 - FADE_SIG0)) ** 1.5


class JointDisplay:
    """Per-sample seen positions of every track; extrapolates an unseen player from his last sighting with the
    neighbours' common motion (replay/extrapolate.py, same model as the pass-2 API), court-clamped, with a
    sigma that grows honestly. Results are cached per (track, run end)."""

    def __init__(self, model: "ReplayModel"):
        self.m = model
        n = len(model.ts)
        self.n = n
        ks, iis, xys = [], [], []
        for k, tr in enumerate(model.tracks):
            if tr.get("team") not in ("A", "B"):
                continue
            i0 = int(tr.get("i0", 0))
            m_ = min(len(tr.get("x") or []), max(0, n - i0))
            if m_ <= 0:
                continue
            xs = _floats(tr["x"][:m_])
            ys = _floats(tr["y"][:m_])
            ok = np.isfinite(xs) & np.isfinite(ys)
            fl = _flags(tr)
            if fl is not None:
                f = np.zeros(m_)
                f[:min(m_, len(fl))] = fl[:m_]
                ok &= f == 1
            j = np.flatnonzero(ok)
            ks.append(np.full(len(j), k))
            iis.append(i0 + j)
            xys.append(np.c_[xs[j], ys[j]] / 100.0)
        if ks:
            kk, ii, xy = np.concatenate(ks), np.concatenate(iis), np.concatenate(xys)
            o = np.lexsort((kk, ii))
            self.k, self.i, self.xy = kk[o], ii[o], xy[o]
        else:
            self.k, self.i, self.xy = np.zeros(0, int), np.zeros(0, int), np.zeros((0, 2))
        self.start = np.searchsorted(self.i, np.arange(n + 1))  # rows of sample i: start[i]:start[i+1]
        self._cache: dict = {}
        self.ball = None
        b = model.data.get("ball")
        if b and b.get("x"):
            bx = _floats(b["x"][:n]) / 100.0
            by = _floats(b["y"][:n]) / 100.0
            self.ball = _fill_gaps(np.c_[bx, by], int(round(JOINT_BALL_GAP_S / max(model.step, 1e-3))))

    def _at(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        a, b = self.start[i], self.start[i + 1]
        return self.k[a:b], self.xy[a:b]

    def residual(self, k_self: int, a: int, H: int, p0) -> np.ndarray:
        """(H, 2) mean displacement since sample a of the players seen at a (and the ball), weighted by distance."""
        ka, xa = self._at(a)
        sel = ka != k_self
        ka, xa = ka[sel], xa[sel]
        num = np.zeros((H, 2))
        den = np.zeros(H)
        w = np.exp(-np.hypot(xa[:, 0] - p0[0], xa[:, 1] - p0[1]) / JOINT_L_M)
        keep = w > 0.05
        for kk, x0, ww in zip(ka[keep], xa[keep], w[keep]):
            pos = self._track_xy(int(kk), a + 1, a + 1 + H)
            ok = np.isfinite(pos[:, 0])
            if ok.mean() < 0.5:
                continue
            num[ok] += ww * (pos[ok] - x0)
            den[ok] += ww
        if self.ball is not None and np.isfinite(self.ball[a]).all():
            bb = self.ball[a + 1:a + 1 + H]
            ok = np.isfinite(bb[:, 0])
            if len(ok) == H and ok.mean() >= 0.5:
                wb = 0.5 * math.exp(-math.hypot(*(self.ball[a] - p0)) / JOINT_L_M)
                num[ok] += wb * (bb[ok] - self.ball[a])
                den[ok] += wb
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(den[:, None] > 0, num / np.maximum(den, 1e-9)[:, None], 0.0)

    def _track_xy(self, k: int, i0: int, i1: int) -> np.ndarray:
        tr = self.m.tracks[k]
        o = int(tr.get("i0", 0))
        xs, ys = tr.get("x") or [], tr.get("y") or []
        out = np.full((i1 - i0, 2), np.nan)
        a, b = max(i0, o), min(i1, o + len(xs))
        if b > a:
            sx = _floats(xs[a - o:b - o]) / 100.0
            sy = _floats(ys[a - o:b - o]) / 100.0
            out[a - i0:b - i0] = np.c_[sx, sy]
        return out

    def extrapolator(self, k_track: int, path: "TrackPath"):
        ts = self.m._ts_arr
        H = max(2, int(round(JOINT_HORIZON_S / max(self.m.step, 1e-3))))

        def ext(j: int, dt: float):
            key = (k_track, j)
            r = self._cache.get(key)
            if r is None:
                if len(self._cache) > 4000:
                    self._cache.clear()
                t_end = path.t[j]
                a = int(np.searchsorted(ts, t_end - 1e-6))
                if a >= self.n - 1:
                    return None
                h = min(H, self.n - 1 - a)
                tt = ts[a + 1:a + 1 + h] - ts[a]
                p0 = np.array([path.x[j], path.y[j]])
                v0 = np.array([path.vx[j], path.vy[j]])
                own = p0 + v0[None] * JOINT_TAU_S * (1 - np.exp(-tt / JOINT_TAU_S))[:, None]
                ramp = np.minimum(1.0, tt / JOINT_RAMP_S)[:, None]
                xy = own + JOINT_ALPHA * ramp * self.residual(k_track, a, h, p0)
                c = self.m.data.get("court") or {}
                L, W = float(c.get("length") or court.LENGTH), float(c.get("width") or court.WIDTH)
                full = np.r_[p0[None], xy]  # physically plausible: never faster than VMAX between samples
                step = np.diff(full, axis=0)
                lim = VMAX * np.maximum(np.diff(np.r_[0.0, tt]), 1e-3)
                sp = np.hypot(step[:, 0], step[:, 1])
                step *= np.minimum(1.0, lim / np.maximum(sp, 1e-9))[:, None]
                xy = p0[None] + np.cumsum(step, axis=0)
                xy[:, 0] = np.clip(xy[:, 0], -1.0, L + 1.0)
                xy[:, 1] = np.clip(xy[:, 1], -1.0, W + 1.0)
                sig = JOINT_SIG0 + JOINT_SIG_RATE * np.sqrt(tt)
                r = self._cache[key] = (np.r_[0.0, tt], np.r_[p0[None], xy], np.r_[JOINT_SIG0, sig])
            tt, xy, sig = r
            if dt > tt[-1]:
                return None
            x = float(np.interp(dt, tt, xy[:, 0]))
            y = float(np.interp(dt, tt, xy[:, 1]))
            return x, y, float(np.interp(dt, tt, sig))
        return ext


def _lazy_ext(model: "ReplayModel", k: int, path: "TrackPath"):
    """Extrapolator that uses the joint index once it exists (built in the background); before that the path
    falls back to the old coast (ext returns None). Small replays build the index right away."""
    cell: list = []

    def ext(j: int, dt: float):
        if not cell:
            jd = model.joint_ready()
            if jd is None and len(model.ts) <= JOINT_SYNC_MAX_SAMPLES:
                jd = model.joint()
            if jd is None:
                return None
            cell.append(jd.extrapolator(k, path))
        return cell[0](j, dt)
    return ext


JOINT_SYNC_MAX_SAMPLES = 3000  # replays up to ~5 min build the joint index on first use (well under 0.1 s)
