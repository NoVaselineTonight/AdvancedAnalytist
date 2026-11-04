"""Who is a player? Separates players + referees from bench, coaches, table officials and spectators.

Works per detection and per time window (substitutions: a person can sit on the bench for minutes and then
play), from several independent cues so that it still works when the court calibration is poor or missing:

  static     how often the spot under this person's feet is occupied by *someone* over +-45 s, measured in
             camera-stabilised image space (frame-to-frame camera motion from pass 1 chained into frame-to-frame
             homographies, re-registered on the people themselves because the chain drifts across fast
             pans). Bench rows and people along the walls stay put for minutes; players don't, and a player
             who has just come on court is immediately "non-static" because their new spot was empty.
             Track-free, so it survives broken IDs.
  activity   how far the same track wanders over +-20 s (in box heights, camera-stabilised).
  posture    seated: low box aspect (height/width) and a box that is short compared with standing people at
             the same image row (robust fit of box height vs. foot row over the surrounding second).
  court      distance of the projected foot point outside the court lines (only when calibrated; soft, so a
             15-40 px calibration error doesn't decide alone).
  count      at most ``max_on_court`` (10 players + 3 referees) are kept per frame, most player-like first
             (``cap_rank``: score + a bonus for standing metres inside the lines; pass 2's later per-frame
             limits use the same priority).
  Inside the court lines the negative "static"/"activity" evidence is dropped: standing still at a free
  throw or during a dead ball is what players and referees do there.

Each cue gives evidence in [-1, 1] (+ = player); a weighted sum (log-odds-like score) is smoothed along each
track over a sliding window and thresholded at 0. Team colour is deliberately not used: bench players wear the
same jerseys as the players on court.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from replay import court
from replay.config import FilterConfig

CUES = ("static", "activity", "posture", "court", "crowd", "benchlife")
REASON = {"static": "static (bench/wall)", "activity": "not moving", "posture": "seated",
          "court": "off court", "crowd": "next to the bench", "count": "over per-frame limit", "short": "too short",
          "benchlife": "bench life (sideline strip)"}
# benchlife: whole-life bench-likeness of the track (replay/bench_guard.py: court feet in the strip next to the bench,
# never deep in the court, no travel, slow). Evidence -score for flagged tracks, 0 otherwise; folded into the score
# like the other cues so flagged tracks lose the per-frame count contests (cap_rank). Weight from env
# REPLAY_BENCH_GUARD_ROSTER (0 = off, default; data/plans/bench_guard.md (b)).
W_BENCHLIFE = float(os.environ.get("REPLAY_BENCH_GUARD_ROSTER", "0") or 0)
# bench crowd: someone standing/walking just outside the lines right next to seated or static people is a coach,
# a substitute or a table official (players stand there only for a moment, e.g. at an inbound)
CROWD_X_REL = 1.5  # neighbour within this many own box heights sideways ...
CROWD_Y_REL = 0.25  # ... and with the feet within this share of the box height (same depth)
CROWD_STATIC_OCC = 0.8  # neighbour counts as "bench" when its spot is occupied this much of the time ...
CROWD_SEATED_EV = -0.3  # ... or its posture evidence is this seated
CROWD_FULL = 2  # this many bench neighbours = full evidence
CROWD_OFF_M = 0.2  # only outside the lines: evidence grows from this many metres off court ...
CROWD_OFF_FULL_M = 0.7  # ... to full here
# The crowd cue decides RosterResult.hide (display only), not keep: dropping those tracks before team classification
# changed the team clustering enough to turn a long player track "unknown" on clip-4400. REPLAY_BENCH_CROWD=0: off
# calibration sanity: a standing person is ~1.1-2.6 m tall; box height x ground scale at the feet far outside that band
# (per frame, median over everyone, +-2 s) means the court calibration is off (wrong scale/place) -> court
# positions are treated as unknown for the player/bench decision (no off-court / inside-court evidence)
CALIB_HEIGHT_M = (1.1, 2.6)
CALIB_SMOOTH_S = 2.0
W_CROWD = 0.0 if os.environ.get("REPLAY_BENCH_CROWD", "").strip() in ("0", "false", "no") else 1.5


@dataclass
class RosterResult:
    keep: np.ndarray  # bool per detection: player or referee
    score: np.ndarray  # smoothed score per detection (> 0 = player-like)
    evidence: dict[str, np.ndarray]  # cue -> per-detection evidence in [-1, 1], NaN = not measurable
    raw: dict[str, np.ndarray]  # occupancy, spread (box heights), aspect, rel. height, metres off court
    reason: np.ndarray  # object array: why a detection was dropped ("" if kept)
    dropped: dict[str, int] = field(default_factory=dict)  # reason -> number of tracks (mostly) removed
    rank: np.ndarray | None = None  # per-frame cap priority: score + bonus for being deep inside the court
    # kept for team classification but never shown: standing next to the seated bench just outside the lines
    # (hiding them only after the team colours are learnt keeps the team clustering unchanged)
    hide: np.ndarray | None = None


def relative_transforms(G: list[np.ndarray] | None, n: int) -> tuple[np.ndarray, np.ndarray]:
    """(G, G^-1) as (n, 3, 3) arrays; frame-i pixels -> frame-j pixels is ``Ginv[j] @ G[i]``."""
    if G is None:
        Gs = np.repeat(np.eye(3)[None], n, axis=0)
    else:
        Gs = np.asarray(G, float).reshape(n, 3, 3)
    return Gs, np.linalg.inv(Gs)


def _map(T: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """T (..., 3, 3) applied to points p (m, 2) -> (..., m, 2) and the homogeneous w (..., m)."""
    ph = np.concatenate([p, np.ones((len(p), 1))], axis=1)
    q = ph @ np.swapaxes(T, -1, -2)  # (..., m, 3); matmul (BLAS) is several times faster than einsum here
    w = q[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        return q[..., :2] / w[..., None], w


def _by_frame(i: np.ndarray, n: int) -> list[np.ndarray]:
    order = np.argsort(i, kind="stable")
    bounds = np.searchsorted(i[order], np.arange(n + 1))
    return [order[bounds[f]:bounds[f + 1]] for f in range(n)]


def _feet(box: np.ndarray) -> np.ndarray:
    return np.stack([(box[:, 0] + box[:, 2]) / 2, box[:, 3]], axis=1)


class _Scene:
    """All person detections per frame + camera-stabilised frame-to-frame transforms."""

    def __init__(self, ref_i: np.ndarray, ref_box: np.ndarray, G: list[np.ndarray] | None, seg: np.ndarray,
                 n: int, size: tuple[int, int], cfg: FilterConfig, ref_feat: np.ndarray | None = None):
        self.n, self.size, self.cfg, self.seg = n, size, cfg, seg
        self.Gs, self.Ginv = relative_transforms(G, n)
        by = _by_frame(ref_i, n)
        m = max(1, max((len(r) for r in by), default=0))
        self.foot = np.full((n, m, 2), np.nan)
        self.h = np.full((n, m), np.nan)
        self.col = np.full((n, m, 3), np.nan)  # torso Lab colour (pass-1 jersey feature), NaN = unknown
        for f, r in enumerate(by):
            self.foot[f, :len(r)] = _feet(ref_box[r])
            self.h[f, :len(r)] = ref_box[r, 3] - ref_box[r, 1]
            if ref_feat is not None:
                self.col[f, :len(r)] = ref_feat[r, :3]
        # how far the camera travelled (image centre, pixels, cumulative) - bounds the possible drift
        c = np.array([[size[0] / 2, size[1] / 2]])
        step = np.zeros(n)
        if n > 1:
            q, _ = _map(self.Ginv[1:] @ self.Gs[:-1], c)
            step[1:] = np.nan_to_num(np.linalg.norm(q[:, 0] - c, axis=1), nan=0.0, posinf=0.0)
            step[1:][seg[1:] != seg[:-1]] = 0.0
        self.path = np.cumsum(step)

    def _sources(self, f: int):
        """Frame f's people for re-registration: (feet, heads, torso colours, histogram bin px) or None (cached)."""
        cache = self.__dict__.setdefault("_src_cache", {})
        if f not in cache:
            ok = np.isfinite(self.h[f])
            if self.cfg.realign_min_support <= 0 or ok.sum() < self.cfg.realign_min_support:
                cache[f] = None
            else:
                src, hs = self.foot[f][ok], self.h[f][ok]
                head = src - np.stack([np.zeros_like(hs), hs], axis=1)
                cache[f] = (src, head, self.col[f][ok], max(6.0, 0.15 * float(np.median(hs))))
        return cache[f]

    def transforms(self, f: int, J: np.ndarray) -> np.ndarray:
        """(len(J), 3, 3): frame-f pixels -> frame-j pixels, re-registered on the people in both frames.

        The chained camera motion drifts (~100+ px after a fast pan away and back). The residual is mostly a
        shift, found as the densest offset between frame-f people (mapped) and frame-j people of similar
        size and torso colour: everyone who stood still votes for the same offset, moving players scatter.
        Colour matters for rows of seated people, where "shifted by one seat" would otherwise also agree.
        """
        T = self.Ginv[J] @ self.Gs[f]
        cfg = self.cfg
        src = self._sources(f)
        if src is None or not len(J):
            return T
        src, head, cs, b = src
        p, w = _map(T, src)  # (nJ, ns, 2)
        ph, _ = _map(T, head)
        hm = np.linalg.norm(p - ph, axis=2)  # source heights as seen in frame j
        cap = 20.0 + cfg.realign_drift_rel * np.abs(self.path[J] - self.path[f])
        # candidate pairs (j, source, target): the cheap offset test first, the other tests only on survivors
        fJ = self.foot[J]
        ox = fJ[:, None, :, 0] - p[:, :, None, 0]  # (nJ, ns, m)
        oy = fJ[:, None, :, 1] - p[:, :, None, 1]
        with np.errstate(invalid="ignore", divide="ignore"):
            jj, si, ti = np.nonzero((np.maximum(np.abs(ox), np.abs(oy)) <= cap[:, None, None]) & (w[:, :, None] > 0))
            ratio = self.h[J[jj], ti] / hm[jj, si]
            dc = self.col[J[jj], ti] - cs[si]
            dcol = np.sqrt(dc[:, 0] * dc[:, 0] + dc[:, 1] * dc[:, 1] + dc[:, 2] * dc[:, 2])
            valid = (ratio > 0.7) & (ratio < 1.45) & ~(dcol > cfg.realign_max_color)
        jj, si, ti = jj[valid], si[valid], ti[valid]
        if not len(jj):
            return T
        o = np.stack([ox[jj, si, ti], oy[jj, si, ti]], axis=1)
        K = int(np.ceil(cap.max() / b)) + 1
        gx = np.clip(np.round(o[:, 0] / b).astype(int) + K, 0, 2 * K)
        gy = np.clip(np.round(o[:, 1] / b).astype(int) + K, 0, 2 * K)
        hist = np.zeros((len(J), 2 * K + 3, 2 * K + 3))
        np.add.at(hist, (jj, gx + 1, gy + 1), 1.0)
        sm = sum(hist[:, 1 + dx:2 * K + 2 + dx, 1 + dy:2 * K + 2 + dy] for dx in (-1, 0, 1) for dy in (-1, 0, 1))
        sm[:, K, K] += 0.5  # prefer "no correction" on ties
        flat = sm.reshape(len(J), -1).argmax(axis=1)
        peak = sm.reshape(len(J), -1)[np.arange(len(J)), flat]
        cx, cy = (flat // (2 * K + 1) - K) * b, (flat % (2 * K + 1) - K) * b
        fix = (peak >= cfg.realign_min_support) & ~((cx == 0) & (cy == 0) & (peak < cfg.realign_min_support + 0.5))
        if not fix.any():
            return T
        # shift = mean of the votes within 1.5 bins of the peak (the peak itself if none)
        near = fix[jj] & (np.maximum(np.abs(o[:, 0] - cx[jj]), np.abs(o[:, 1] - cy[jj])) <= 1.5 * b)
        cnt = np.bincount(jj[near], minlength=len(J))
        sx = np.bincount(jj[near], weights=o[near, 0], minlength=len(J))
        sy = np.bincount(jj[near], weights=o[near, 1], minlength=len(J))
        with np.errstate(invalid="ignore", divide="ignore"):
            dx = np.where(cnt > 0, sx / cnt, cx)
            dy = np.where(cnt > 0, sy / cnt, cy)
        A = np.flatnonzero(fix)
        S = np.repeat(np.eye(3)[None], len(A), axis=0)
        S[:, 0, 2], S[:, 1, 2] = dx[A], dy[A]
        T[A] = S @ T[A]
        return T


def occupancy(q_i: np.ndarray, q_box: np.ndarray, scene: _Scene, fps: float,
              q_col: np.ndarray | None = None, only=None) -> tuple[np.ndarray, np.ndarray]:
    """Share of (sampled) frames within +-static_window_s in which the query's foot spot, while in view, has
    a similar-sized (and, with colours, similar-looking) person standing on it. Returns (occupancy, visible
    seconds); occupancy NaN when seen too briefly."""
    cfg, n, seg = scene.cfg, scene.n, scene.seg
    W, H = scene.size
    occ = np.full(len(q_i), np.nan)
    vis_s = np.zeros(len(q_i))
    step = max(1, int(round(cfg.static_sample_s * fps)))
    half = int(round(cfg.static_window_s * fps))
    gap = int(round(cfg.static_min_dt_s * fps))
    margin = 8.0
    by = _by_frame(q_i, n)
    for f in (range(n) if only is None else only):  # only: a subset of frames (process pool, see _heavy)
        qs = by[f]
        if not len(qs):
            continue
        J = np.arange(max(0, f - half), min(n, f + half + 1), step)
        J = J[(np.abs(J - f) >= gap) & (seg[J] == seg[f])
              & (np.abs(scene.path[J] - scene.path[f]) <= cfg.static_max_travel_px)]
        if not len(J):
            continue
        b = q_box[qs]
        foot = _feet(b)
        head = foot - np.stack([np.zeros(len(b)), b[:, 3] - b[:, 1]], axis=1)
        T = scene.transforms(f, J)  # (nJ, 3, 3): frame f -> frame j
        fj, w1 = _map(T, foot)  # (nJ, nq, 2)
        hj, w2 = _map(T, head)
        hgt = np.linalg.norm(fj - hj, axis=2)  # query height in frame j
        with np.errstate(invalid="ignore"):
            visible = ((w1 > 0) & (w2 > 0) & (fj[..., 0] > margin) & (fj[..., 0] < W - margin)
                       & (fj[..., 1] > margin) & (fj[..., 1] < H - margin) & np.isfinite(hgt))
            fJ = scene.foot[J]
            dx = fj[:, :, None, 0] - fJ[:, None, :, 0]
            dy = fj[:, :, None, 1] - fJ[:, None, :, 1]
            d = np.sqrt(dx * dx + dy * dy)  # (nJ, nq, m)
            ratio = scene.h[J][:, None, :] / hgt[..., None]
            same = (d < cfg.static_radius_rel * hgt[..., None]) & (ratio > 1 / cfg.static_height_ratio) & (ratio < cfg.static_height_ratio)
            if q_col is not None and cfg.static_color_veto:
                a, qi, ti = np.nonzero(same)  # colour only where everything else matched
                dc = scene.col[J[a], ti] - q_col[qs[qi]]
                dcol = np.sqrt(dc[:, 0] * dc[:, 0] + dc[:, 1] * dc[:, 1] + dc[:, 2] * dc[:, 2])
                same[a, qi, ti] = ~(dcol > cfg.realign_max_color)  # unknown colour = no veto
            hit = same.any(axis=2)
        nv = visible.sum(axis=0)
        vis_s[qs] = nv * step / fps
        ok = nv * step / fps >= cfg.static_min_visible_s
        occ[qs[ok]] = (hit & visible).sum(axis=0)[ok] / nv[ok]
    return occ, vis_s


def activity(i: np.ndarray, box: np.ndarray, track: np.ndarray, scene: _Scene,
             fps: float, only=None) -> tuple[np.ndarray, np.ndarray]:
    """How far each track moves around each detection: 90th percentile distance (in box heights, camera-
    stabilised into the detection's frame) of the track's foot points within +-activity_window_s from their
    median. NaN when the track is seen for less than activity_min_s around the detection.
    Returns (spread, observed seconds)."""
    cfg, seg = scene.cfg, scene.seg
    out = np.full(len(i), np.nan)
    span = np.zeros(len(i))
    half = cfg.activity_window_s * fps
    foot = _feet(box)
    hgt = np.maximum(1.0, box[:, 3] - box[:, 1])
    qstep = max(1, int(round(1.0 * fps)))  # evaluate ~1x per second and reuse in between (0.5 s: same benchmark, 1.4x slower)
    for t in (np.unique(track[track > 0]) if only is None else only):  # only: a subset of track ids
        idx = np.flatnonzero(track == t)
        idx = idx[np.argsort(i[idx], kind="stable")]
        ii = i[idx]
        if len(idx) < 2:
            continue
        for a in range(0, len(idx), qstep):
            f = ii[a]
            lo, hi = np.searchsorted(ii, f - half), np.searchsorted(ii, f + half, side="right")
            win = np.arange(lo, hi)
            win = win[(seg[ii[win]] == seg[f])
                      & (np.abs(scene.path[ii[win]] - scene.path[f]) <= cfg.static_max_travel_px)]
            if len(win) > 40:
                win = win[np.linspace(0, len(win) - 1, 40).round().astype(int)]
            if len(win) < 2 or (ii[win].max() - ii[win].min()) / fps < cfg.activity_min_s:
                continue
            T = np.linalg.inv(scene.transforms(f, ii[win]))  # each window frame -> frame f
            ph = np.concatenate([foot[idx[win]], np.ones((len(win), 1))], axis=1)
            q = np.einsum("kab,kb->ka", T, ph)
            if (q[:, 2] <= 0).any():
                continue
            p = q[:, :2] / q[:, 2:]
            spread = np.percentile(np.linalg.norm(p - np.median(p, axis=0), axis=1), 90) / hgt[idx[a]]
            out[idx[a:a + qstep]] = spread
            span[idx[a:a + qstep]] = (ii[win].max() - ii[win].min()) / fps
    return out, span


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    """Least-squares slope of y = a x + b (closed form; np.polyfit(x, y, 1)[0] up to rounding)."""
    xm = x - x.mean()
    d = float(xm @ xm)
    return float(xm @ (y - y.mean())) / d if d > 0 else 0.0


def relative_height(q_i: np.ndarray, q_box: np.ndarray, scene: _Scene, fps: float, only=None) -> np.ndarray:
    """Box height / height expected for a standing person at the same foot row (robust upper fit of
    height vs. foot y over the surrounding +-1 s of raw detections). NaN when the fit is not possible."""
    out = np.full(len(q_i), np.nan)
    n = scene.n
    half = max(1, int(round(1.0 * fps)))
    by = _by_frame(q_i, n)
    for f in (range(n) if only is None else only):
        qs = by[f]
        if not len(qs):
            continue
        sl = slice(max(0, f - half), f + half + 1)
        h = scene.h[sl].ravel()
        y = scene.foot[sl][..., 1].ravel()
        ok = np.isfinite(h)
        h, y = h[ok], y[ok]
        if len(h) < 8 or np.ptp(y) < 30:
            continue
        # h = a * y_foot + b. For a camera near head height a ~ 1 (standing heads share an image row); the
        # slope is kept in a plausible range so a few near/cut boxes can't tilt it. Refit on the upper part
        # (standing people) so rows of seated people don't drag the line down.
        keep = np.ones(len(h), bool)
        a = b = None
        for _ in range(4):
            if keep.sum() < 5:
                break
            a = float(np.clip(_slope(y[keep], h[keep]), 0.4, 1.6))
            b = float(np.median(h[keep] - a * y[keep]))
            keep = h > 0.85 * (a * y + b)
        if a is None:
            continue
        pred = a * q_box[qs, 3] + b
        good = pred > 5
        out[qs[good]] = (q_box[qs, 3] - q_box[qs, 1])[good] / pred[good]
    return out


def distance_off_court(xy: np.ndarray) -> np.ndarray:
    """Metres outside the court rectangle (0 inside, NaN for unknown positions)."""
    xy = np.asarray(xy, float).reshape(-1, 2)
    dx = np.maximum(np.maximum(-xy[:, 0], xy[:, 0] - court.LENGTH), 0)
    dy = np.maximum(np.maximum(-xy[:, 1], xy[:, 1] - court.WIDTH), 0)
    return np.hypot(dx, dy)


def distance_inside_court(xy: np.ndarray) -> np.ndarray:
    """Metres to the nearest court line for positions inside the court (0 outside, NaN for unknown)."""
    xy = np.asarray(xy, float).reshape(-1, 2)
    d = np.minimum(np.minimum(xy[:, 0], court.LENGTH - xy[:, 0]), np.minimum(xy[:, 1], court.WIDTH - xy[:, 1]))
    return np.maximum(d, 0.0)


def bench_crowd(i: np.ndarray, box: np.ndarray, occ: np.ndarray, posture_ev: np.ndarray, n: int) -> np.ndarray:
    """Number of seated/static people next to each detection in the same frame (same depth, beside it)."""
    out = np.zeros(len(i))
    foot = _feet(box)
    h = np.maximum(1.0, box[:, 3] - box[:, 1])
    benchy = (np.nan_to_num(occ, nan=0.0) >= CROWD_STATIC_OCC) | (np.nan_to_num(posture_ev, nan=0.0) <= CROWD_SEATED_EV)
    for ds in _by_frame(i, n):
        if len(ds) < 2:
            continue
        bs = ds[benchy[ds]]
        if not len(bs):
            continue
        dx = np.abs(foot[ds, None, 0] - foot[None, bs, 0])
        dy = np.abs(foot[ds, None, 1] - foot[None, bs, 1])
        near = (dx < CROWD_X_REL * h[ds, None]) & (dy < CROWD_Y_REL * h[ds, None]) & (ds[:, None] != bs[None, :])
        out[ds] = near.sum(axis=1)
    return out


def calibration_plausible(i: np.ndarray, box: np.ndarray, H: list | None, n: int, fps: float) -> np.ndarray:
    """bool per detection: the court calibration of its frame gives people a plausible height (True without H or
    when nothing can be measured - the old behaviour)."""
    ok = np.ones(len(i), bool)
    if H is None or not len(i) or os.environ.get("REPLAY_CALIB_GATE", "").strip() in ("0", "false", "no"):
        return ok
    from replay import homography

    med = np.full(n, np.nan)
    for f, ds in enumerate(_by_frame(i, n)):
        if not len(ds) or f >= len(H) or H[f] is None or not np.isfinite(H[f]).all():
            continue
        b = box[ds]
        J = homography.jacobian_each(np.repeat(np.asarray(H[f], float)[None], len(ds), axis=0),
                                     np.c_[(b[:, 0] + b[:, 2]) / 2, b[:, 3]])
        hm = (b[:, 3] - b[:, 1]) * np.hypot(J[:, 0, 0], J[:, 1, 0])
        hm = hm[np.isfinite(hm)]
        if len(hm):
            med[f] = float(np.median(hm))
    half = max(1, int(round(CALIB_SMOOTH_S * fps)))
    good = np.ones(n, bool)
    for f in range(n):
        w = med[max(0, f - half):f + half + 1]
        w = w[np.isfinite(w)]
        if len(w):
            m = float(np.median(w))
            good[f] = CALIB_HEIGHT_M[0] <= m <= CALIB_HEIGHT_M[1]
    return good[i]


def cap_rank(score: np.ndarray, inside_m: np.ndarray, cfg: FilterConfig) -> np.ndarray:
    """Priority for the per-frame people limits: the player score plus a bonus for standing deep inside the
    court. The score alone can't tell a bench player who keeps getting up (moving, spot not always occupied)
    from a player; being several metres inside the lines can (unknown position = no bonus)."""
    return np.asarray(score, float) + cfg.cap_inside_bonus * np.clip(np.nan_to_num(inside_m, nan=0.0), 0.0,
                                                                     cfg.cap_inside_full_m)


def limit_shown_rows(rows, caps, rank: dict[int, float], strength: dict) -> set[int]:
    """Final per-frame limits of pass 2 (e.g. 10 players, 3 referees), applied to output rows (export.TrackRow:
    .k, .team, .interpolated, .track_id). Returns the id() of the rows to drop.

    Rows with a detection go first, most player-like first (``rank``: id(row) -> cap_rank of its detection),
    then filled-in rows in a fixed track order (``strength``: track_id -> number). Ranking everything by track
    strength alone let filled-in gaps and long bench/sideline tracks push out players who were seen right then.
    """
    by_frame: dict[int, list] = {}
    for r in rows:
        by_frame.setdefault(r.k, []).append(r)
    from replay.consolidate import partial_rows

    drop: set[int] = set()
    for rs in by_frame.values():
        part = None
        for labels, cap in caps:
            mine = [r for r in rs if r.team in labels]
            if len(mine) <= cap:
                continue
            if part is None:  # a partial second box of someone (legs hidden) gives way first
                part = partial_rows(rs)
            mine.sort(key=lambda r: (bool(r.interpolated), id(r) in part, -rank.get(id(r), 0.0),
                                     -strength.get(r.track_id, 0.0)))
            drop.update(id(r) for r in mine[cap:])
    return drop


def _track_smooth(z: np.ndarray, i: np.ndarray, track: np.ndarray, half: float) -> np.ndarray:
    """Mean of z over +-half frames along each track (detections without a track keep their own z)."""
    out = z.copy()
    for t in np.unique(track[track > 0]):
        idx = np.flatnonzero(track == t)
        idx = idx[np.argsort(i[idx], kind="stable")]
        ii = i[idx].astype(float)
        c = np.concatenate([[0.0], np.cumsum(z[idx])])
        lo = np.searchsorted(ii, ii - half)
        hi = np.searchsorted(ii, ii + half, side="right")
        out[idx] = (c[hi] - c[lo]) / (hi - lo)
    return out


def short_but_continuing(track: np.ndarray, i: np.ndarray, court_xy: np.ndarray, feat: np.ndarray | None, fps: float,
                         cfg: FilterConfig, kept: np.ndarray | None = None) -> np.ndarray:
    """Per detection: True when its identity is shorter than ``min_track_frames`` AND within ``short_keep_window_s``
    before/after it a kept identity (``kept``; default: every identity that is not short) of the same kit (torso Lab
    distance < ``realign_max_color``; unknown colour = no veto) ends/starts at a position reachable at
    ``short_keep_reach_mps`` (+ 1 m; NaN court positions = colour test only). Such a piece is a player re-appearing
    for a moment between occlusions, not noise."""
    track = np.asarray(track, int)
    i = np.asarray(i, int)
    xy = np.asarray(court_xy, float).reshape(-1, 2)
    n = len(track)
    out = np.zeros(n, bool)
    ids, counts = np.unique(track[track > 0], return_counts=True)
    if not len(ids):
        return out
    short_ids = set(ids[counts < cfg.min_track_frames].tolist())
    if kept is None:
        kept = (track > 0) & ~np.isin(track, list(short_ids))
    kept = np.asarray(kept, bool)

    def summary(m: np.ndarray):
        o = m[np.argsort(i[m], kind="stable")]
        f = None
        if feat is not None:
            ff = np.asarray(feat, float)[o, :3]
            ff = ff[np.isfinite(ff).all(axis=1)]
            f = np.median(ff, axis=0) if len(ff) else None
        return int(i[o[0]]), int(i[o[-1]]), xy[o[0]], xy[o[-1]], f

    longs = [summary(np.flatnonzero(kept & (track == g))) for g in np.unique(track[kept & (track > 0)])]
    if not longs:
        return out
    L0 = np.array([s[0] for s in longs])
    L1 = np.array([s[1] for s in longs])
    win = cfg.short_keep_window_s * fps
    for g in short_ids:
        m = np.flatnonzero(track == g)
        a0, a1, p0, p1, fa = summary(m)
        for j in np.flatnonzero(((L1 <= a0) & (L1 >= a0 - win)) | ((L0 >= a1) & (L0 <= a1 + win))):
            b0, b1, q0, q1, fb = longs[j]
            if fa is not None and fb is not None and np.linalg.norm(fa - fb) > cfg.realign_max_color:
                continue
            before = b1 <= a0
            dt = (a0 - b1 if before else b0 - a1) / fps
            pa, pb = (p0, q1) if before else (p1, q0)
            if np.isfinite(pa).all() and np.isfinite(pb).all() and                     np.linalg.norm(pa - pb) > cfg.short_keep_reach_mps * dt + 1.0:
                continue
            out[m] = True
            break
    return out


def benchlife(i: np.ndarray, track: np.ndarray, court_xy: np.ndarray, fps: float) -> np.ndarray:
    """Per detection: bench_guard score of its track over the track's whole life (0 for tracks not flagged
    is_bench, for detections without a track, or without enough court positions)."""
    from replay.bench_guard import score_track

    N = len(i)
    out = np.zeros(N)
    xy = np.asarray(court_xy, float).reshape(N, 2)
    track = np.asarray(track, int)
    ok = np.isfinite(xy).all(axis=1) & (track > 0)
    for t in np.unique(track[ok]):
        m = np.flatnonzero(ok & (track == t))
        rows = [(float(i[d]) / fps, float(xy[d, 0]), float(xy[d, 1]), True) for d in m]
        s = score_track(rows, court.LENGTH, court.WIDTH)
        if s.is_bench:
            out[track == t] = s.score
    return out


# ---- process pool for the three per-frame / per-track cues (identical output: every element is computed by the same
# code from the same inputs, in a worker instead of the parent). Falls back to serial for short clips (pool start-up
# costs ~3 s), REPLAY_PARALLEL=0, or when already inside a worker process.
PAR_MIN_FRAMES = 5000  # sampled frames below this: serial (REPLAY_PARALLEL=1 forces the pool)
_PAR: dict = {}


def _par_init(scene, i, box, track, fps, q_col):
    _PAR.update(scene=scene, i=i, box=box, track=track, fps=fps, q_col=q_col, by=_by_frame(i, scene.n))


def _par_job(job):
    kind, items = job
    P = _PAR
    sc, i = P["scene"], P["i"]
    if kind == "act":
        out, span = activity(i, P["box"], P["track"], sc, P["fps"], only=items)
        idx = np.flatnonzero(np.isin(P["track"], items))
        return kind, idx, out[idx], span[idx]
    idx = np.concatenate([P["by"][f] for f in items]) if len(items) else np.zeros(0, int)
    if kind == "occ":
        occ, vis = occupancy(i, P["box"], sc, P["fps"], P["q_col"], only=items)
        return kind, idx, occ[idx], vis[idx]
    rel = relative_height(i, P["box"], sc, P["fps"], only=items)
    return kind, idx, rel[idx], None


def _use_pool(n: int) -> int:
    import multiprocessing as mp
    env = os.environ.get("REPLAY_PARALLEL", "").strip().lower()
    if env in ("0", "false", "no") or mp.parent_process() is not None:
        return 0
    if n < PAR_MIN_FRAMES and env not in ("1", "true", "yes"):
        return 0
    return max(1, min(4, (os.cpu_count() or 2) - 2))  # at most 4 worker processes


def _heavy(i, box, track, scene, fps, q_col):
    """(static, visible_s, activity, observed_s, rel_height) - serial or pooled, same values."""
    n = scene.n
    workers = _use_pool(n)
    if not workers:
        static, vis_s = occupancy(i, box, scene, fps, q_col)
        act, act_s = activity(i, box, track, scene, fps)
        return static, vis_s, act, act_s, relative_height(i, box, scene, fps)
    from concurrent.futures import ProcessPoolExecutor
    nb = workers * 6
    frames = np.flatnonzero(np.bincount(i, minlength=n) > 0)
    tracks = np.unique(track[track > 0])
    jobs = []
    for k in range(nb):  # interleaved blocks: even load
        jobs.append(("act", tracks[k::nb].tolist()))
        jobs.append(("occ", frames[k::nb].tolist()))
        jobs.append(("rel", frames[k::nb].tolist()))
    static = np.full(len(i), np.nan)
    vis_s = np.zeros(len(i))
    act = np.full(len(i), np.nan)
    act_s = np.zeros(len(i))
    rel = np.full(len(i), np.nan)
    with ProcessPoolExecutor(workers, initializer=_par_init,
                             initargs=(scene, i, box, track, fps, q_col)) as ex:
        for kind, idx, a, b in ex.map(_par_job, jobs):
            if kind == "occ":
                static[idx], vis_s[idx] = a, b
            elif kind == "rel":
                rel[idx] = a
            else:
                act[idx], act_s[idx] = a, b
    return static, vis_s, act, act_s, rel


def classify_roster(i: np.ndarray, box: np.ndarray, track: np.ndarray, fps: float, size: tuple[int, int],
                    cfg: FilterConfig, *, n_frames: int | None = None, G: list[np.ndarray] | None = None,
                    seg: np.ndarray | None = None, court_xy: np.ndarray | None = None,
                    ref_i: np.ndarray | None = None, ref_box: np.ndarray | None = None,
                    ref_feat: np.ndarray | None = None, feat: np.ndarray | None = None, candidate: np.ndarray | None = None,
                    H: list | None = None) -> RosterResult:
    """Decide per detection whether it is a player/referee.

    i: frame (sample) index per detection; box: (N, 4) x1 y1 x2 y2 pixels; track: track id per detection
    (<= 0 = none). G/seg: camera-motion mosaic transforms from ``calibration._segment_transforms`` (None =
    fixed camera). court_xy: (N, 2) court metres or NaN (None = no calibration). ref_i/ref_box: *all* person
    detections of the video (also ones dropped earlier) - used for spot occupancy, re-registration and the
    standing-height model; default = the queries. ref_feat/feat: pass-1 jersey features (N, 6) of the
    references / queries (optional; torso colour sharpens the camera re-registration). candidate: detections to consider (default all); the
    others stay dropped and don't count against the per-frame limit.
    """
    i = np.asarray(i, int)
    box = np.asarray(box, float).reshape(-1, 4)
    track = np.asarray(track, int)
    N = len(i)
    n = int(n_frames if n_frames is not None else (i.max() + 1 if N else 0))
    cand = np.ones(N, bool) if candidate is None else np.asarray(candidate, bool).copy()
    seg = np.zeros(n, int) if seg is None else np.asarray(seg, int)
    if ref_i is None:
        ref_i, ref_box, ref_feat = i, box, feat

    raw: dict[str, np.ndarray] = {}
    ev: dict[str, np.ndarray] = {}
    calib_ok = calibration_plausible(i, box, H, n, fps)
    if court_xy is not None and not calib_ok.all():
        court_xy = np.asarray(court_xy, float).reshape(-1, 2).copy()
        court_xy[~calib_ok] = np.nan  # implausible calibration: position unknown for this decision
    if N:
        scene = _Scene(np.asarray(ref_i, int), np.asarray(ref_box, float).reshape(-1, 4), G, seg, n, size, cfg,
                       None if ref_feat is None else np.asarray(ref_feat, float).reshape(len(ref_i), -1))
        raw["static"], vis_s, raw["activity"], act_s, rel_h = _heavy(
            i, box, track, scene, fps, None if feat is None else np.asarray(feat, float).reshape(N, -1)[:, :3])
        raw["aspect"] = (box[:, 3] - box[:, 1]) / np.maximum(1.0, box[:, 2] - box[:, 0])
        raw["rel_height"] = rel_h
        raw["off_court_m"] = distance_off_court(court_xy) if court_xy is not None else np.full(N, np.nan)
    else:
        for k in ("static", "activity", "aspect", "rel_height", "off_court_m", "crowd", "benchlife"):
            raw[k] = np.zeros(0)
        vis_s = act_s = np.zeros(0)

    with np.errstate(invalid="ignore"):
        ev["static"] = (np.clip((cfg.static_occupancy - raw["static"]) / cfg.static_occupancy_scale, -1, 1)
                        * np.clip(vis_s / cfg.static_full_s, 0, 1))  # seen briefly = weaker evidence
        ev["activity"] = np.clip((raw["activity"] - cfg.active_spread) / cfg.active_spread_scale, -1, 1)
        # moving is evidence at once; standing still only counts fully after a long look (dead balls)
        ev["activity"] = np.where(ev["activity"] < 0, ev["activity"] * np.clip(act_s / cfg.activity_full_s, 0, 1),
                                  ev["activity"])
        # the person's own track moving overrides "someone stands on this spot" (neighbours, crowded areas)
        moving = np.nan_to_num(np.maximum(ev["activity"], 0))
        ev["static"] = np.where(np.isnan(ev["static"]), np.nan, np.maximum(ev["static"], -1 + moving))
        e_asp = np.clip((raw["aspect"] - cfg.seated_aspect) / 0.4, -1, 1)
        e_rel = np.clip((raw["rel_height"] - cfg.seated_rel_height) / 0.12, -1, 1)
        post = np.where(np.isnan(e_rel), e_asp, 0.35 * e_asp + 0.65 * e_rel)
        # standing tall is weak evidence (coaches stand too); sitting is strong
        ev["posture"] = np.where(post > 0, 0.3 * post, post)
        ev["court"] = np.clip((cfg.off_court_m - raw["off_court_m"]) / cfg.off_court_scale_m, -1, 1)
        # inside the lines is weak evidence: a 15-40 px calibration error puts the bench "on court" too
        ev["court"] = np.where(ev["court"] > 0, cfg.court_inside_weight * ev["court"], ev["court"])
        # ... but inside the lines, standing still is what players do at free throws / dead balls: "static" and
        # "not moving" count less there (they exist to catch the bench and people along the walls)
        raw["inside_m"] = distance_inside_court(court_xy) if court_xy is not None and N else np.full(N, np.nan)
        span = max(1e-6, cfg.deep_inside_full_m - cfg.deep_inside_m)
        w_deep = 1.0 - (1.0 - cfg.deep_inside_min_weight) * np.clip((raw["inside_m"] - cfg.deep_inside_m) / span, 0, 1)
        w_deep = np.nan_to_num(w_deep, nan=1.0)
        for c in ("static", "activity"):
            ev[c] = np.where(ev[c] < 0, ev[c] * w_deep, ev[c])
        raw["crowd"] = bench_crowd(i, box, raw["static"], ev["posture"], n) if N and W_CROWD > 0 else np.zeros(N)
        outside = np.clip((np.nan_to_num(raw["off_court_m"], nan=0.0) - CROWD_OFF_M)
                          / max(1e-6, CROWD_OFF_FULL_M - CROWD_OFF_M), 0, 1)
        ev["crowd"] = -np.clip(raw["crowd"] / CROWD_FULL, 0, 1) * outside
        raw["benchlife"] = (benchlife(i, track, court_xy, fps) if N and W_BENCHLIFE > 0 and court_xy is not None
                            else np.zeros(N))
        ev["benchlife"] = -np.clip(raw["benchlife"], 0, 1)
    # benchlife, like crowd, decides hide / display rank only (weight 0 in keep): dropping bench-life tracks before
    # the team classification broke the p4400 kit clustering (team acc 96 -> 65 % on one seed, bg2-roster-s3)
    weights = {"static": cfg.w_static, "activity": cfg.w_activity, "posture": cfg.w_posture, "court": cfg.w_court,
               "crowd": 0.0, "benchlife": 0.0}
    contrib = {c: np.nan_to_num(weights[c] * ev[c]) for c in CUES}
    z = cfg.prior + sum(contrib.values()) if N else np.zeros(0)
    zs = _track_smooth(np.asarray(z, float), i, track, cfg.decision_smooth_s * fps)
    crowd_c = np.nan_to_num(W_CROWD * ev["crowd"] + W_BENCHLIFE * ev["benchlife"]) if N else np.zeros(0)
    zs_crowd = _track_smooth(np.asarray(z + crowd_c, float), i, track, cfg.decision_smooth_s * fps) if N else zs

    reason = np.full(N, "", dtype=object)
    worst = np.argmin(np.stack([contrib[c] for c in CUES]), axis=0) if N else np.zeros(0, int)
    keep = cand & (zs > 0)
    for d in np.flatnonzero(cand & ~keep):
        reason[d] = REASON[CUES[worst[d]]]

    # tracks too short to judge
    if cfg.min_track_frames > 1:
        ids, counts = np.unique(track[cand & (track > 0)], return_counts=True)
        short = np.isin(track, ids[counts < cfg.min_track_frames]) & keep
        if short.any() and cfg.short_keep_window_s > 0:  # short re-appearances that continue a kept player stay
            xy = court_xy if court_xy is not None else np.full((N, 2), np.nan)
            short &= ~short_but_continuing(track, i, xy, feat, fps, cfg, kept=keep & ~short)
        reason[short] = REASON["short"]
        keep &= ~short

    hide = keep & (zs_crowd <= 0)

    # per-frame count constraint: the most player-like first
    rank0 = cap_rank(zs, raw["inside_m"], cfg) if N else np.zeros(0)  # (the bench crowd stays in for the teams)
    rank = cap_rank(zs_crowd, raw["inside_m"], cfg) if N else np.zeros(0)  # pass 2's display limits
    if cfg.max_on_court > 0:
        for ds in _by_frame(i, n):
            ds = ds[keep[ds]]
            if len(ds) > cfg.max_on_court:
                over = ds[np.argsort(-rank0[ds], kind="stable")][cfg.max_on_court:]
                keep[over] = False
                reason[over] = REASON["count"]

    dropped: dict[str, int] = {}
    for t in np.unique(track[cand & (track > 0)]):
        m = cand & (track == t)
        if keep[m].mean() >= 0.5:
            continue
        rs, cs = np.unique(reason[m & ~keep].astype(str), return_counts=True)
        r = str(rs[np.argmax(cs)]) if len(rs) else "?"
        dropped[r] = dropped.get(r, 0) + 1
    hide &= keep
    if hide.any():
        dropped[REASON["crowd"]] = int(len(np.unique(track[hide & (track > 0)])))
    raw["calib_ok"] = calib_ok
    return RosterResult(keep=keep, score=zs, evidence=ev, raw=raw, reason=reason, dropped=dropped, rank=rank, hide=hide)


def filter_players(data, det_i: np.ndarray, det_box: np.ndarray, det_feat: np.ndarray, tid: np.ndarray,
                   keep: np.ndarray, det_xy: np.ndarray | None, fps: float, size: tuple[int, int],
                   cfg: FilterConfig, H: list | None = None) -> RosterResult:
    """Pass-2 entry point (postprocess step 4). ``data`` is pipeline.Pass1Data; all of its person rows serve
    as the reference crowd (also the ones step 2 dropped), the court-tracked detections are the queries."""
    from replay.calibration import _segment_transforms  # local: calibration is owned/edited elsewhere

    n = len(data.k)
    idx_of_k = {int(k): j for j, k in enumerate(data.k)}
    from replay.detection import activation_threshold

    persons = data.rows[(data.rows[:, 6] == 0) & (data.rows[:, 5] >= activation_threshold(data.signature))]
    ref_i = np.array([idx_of_k[int(k)] for k in persons[:, 0]], int)
    G, seg = _segment_transforms(data.motion, n)
    return classify_roster(det_i, det_box, tid, fps, size, cfg, n_frames=n, G=G, seg=seg, court_xy=det_xy,
                           ref_i=ref_i, ref_box=persons[:, 1:5], ref_feat=persons[:, 8:], feat=det_feat,
                           candidate=keep, H=H)
