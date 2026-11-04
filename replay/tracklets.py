"""Offline identity tracking: reliable short tracklets, then global linking.

The whole video is available, so instead of deciding identities frame by frame
(online) we do it in two stages, the way offline multi-object trackers do:

1. **Tracklets** - short pieces we are confident belong to one person:
   ByteTrack IDs from pass 1, cut wherever the piece is physically impossible
   (the head jumps further than a sprint allows, in camera-stabilised image
   space; or the court position does) or where the jersey colour changes for
   good (ByteTrack swapped two players while their boxes overlapped).
   Detections ByteTrack left without an ID (low score - often exactly the
   half-hidden players we need during an occlusion) are chained into their own
   small tracklets.

2. **Partial boxes** - when a player stands behind another, the detector box
   often ends at the knees, so the "foot point" is too high and lands metres
   too far away on the court. Within a tracklet the full body height is known
   (a robust upper percentile of the box height, camera-zoom corrected), so a
   box clearly shorter than that, with its top in place, gets its foot
   re-estimated as top + full height, reprojected, and a larger uncertainty.

3. **Linking** - tracklet ends are joined to later tracklet starts by a global
   minimum-cost one-to-one assignment (sparse bipartite matching; every end may
   also stay unlinked at a fixed cost). Costs use, symmetrically from both
   sides of the gap (forward-backward):
     * motion on the court: the gap must be reachable at running speed (hard
       gate); the residual after constant-velocity bridging is scored against
       an uncertainty that grows with the gap,
     * the same in camera-stabilised image space, measured in body heights,
       so linking still works where the court calibration is poor or missing,
     * appearance: torso + shorts colour (robust medians over full boxes) and
       body height (zoom corrected),
     * a small cost per second of gap (prefer the short explanation).
   Linking runs in rounds of growing maximum gap; after each round linked
   pieces are merged, so later rounds compare longer, better-estimated pieces.

Positions are never invented here; the smoother fills short gaps later.
"""

from __future__ import annotations

import copy
import math
import os
from dataclasses import dataclass, field, fields, replace

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching

from replay import corrections, homography
from replay.config import CourtTrackConfig

# appearance: per-dimension detection noise (Lab units) for [L, a, b] torso and shorts, measured on real footage
# (1-s medians of one person differ by ~2 units typically but with a heavy tail: turning, lighting, motion blur)
_APP_SIGMA = np.array([12.0, 7.0, 7.0, 14.0, 8.0, 8.0])
_APP_FLOOR = np.array([6.0, 4.5, 4.5, 6.0, 4.5, 4.5])
_SPLIT_W = np.array([0.6, 1.0, 1.0, 0.6, 1.0, 1.0])  # lightness counts less than colour when cutting
_APP_CAP = 9.0  # per-dimension cap of the squared z-score (robust to the heavy tail)
_MIN_MEAS_SIGMA_M = 0.15
CHAIN_APPEARANCE_CHECK = True  # anti-chimera chain check (cuts chains joining two clearly different kits)
_CHAIN_MIN_HIST = 4  # ... and with at least this many kit histograms when those exist
_CHAIN_MIN_APP = 8  # a piece's kit counts in the chain check with at least this many clean boxes

# kit colour histograms (team_appearance descriptors, sum 1): L1 distance between two pieces' means. Measured on
# clip c300 (pieces >= 3 s): halves of one piece median 0.22 / 95 % 0.62 / 99 % 0.88; two people seen at the same
# time median 1.39 (team-mates from 0.45) -> the histogram tells kits apart, not team-mates.
_HIST_MAX_OCC = 0.3  # descriptor of a box whose shirt is more than this covered by someone else: not used
_HIST_SIGMA_FLOOR = 0.15
_HIST_SIGMA_ONE = 0.5  # noise of a single-box histogram
_HIST_GATE = 0.9  # L1 above this (and above _HIST_GATE_Z sigma) = different kit: never the same person
_HIST_GATE_Z = 3.0
_HIST_CAP = 9.0
HIST_WEIGHT = 1.0
KIT_SPLIT_L1 = 0.0  # > 0: veto when both sides have >= KIT_SPLIT_MIN_N histograms and differ by more than this
KIT_SPLIT_MIN_N = 3
# body height in metres (box height x ground metres-per-pixel at the foot; ~true height for a low side camera).
# c300: halves of one piece median |diff| 0.04 m (heavy tail: bending, jumping), different people median 0.33 m
_HEIGHT_SIGMA_FLOOR = 0.06
_HEIGHT_SIGMA_ONE = 0.25
_HEIGHT_CAP = 4.0
HEIGHT_M_WEIGHT = 0.5
# re-ID embeddings (reid.py, cosine similarity of mean unit vectors over >= EMB_MIN_SAMPLES clean crops). Measured with
# the interim model (iter 900) on pieces >= 1.5 s, c4400 + b3min + c900: same player (same jersey number, separate
# pieces) p5 0.93 / median 0.97; different players of the same kit seen at the same time median 0.55, p90 0.83, p95 0.90
USE_REID = False  # re-ID cue (reid.py cache / ReidConfig.enabled). Off: interim model (it900) gave fewer, longer IDs
# (p4400+c300: IDs/min 79.9->76.8, ID life 14.4->15.9 s) but ID transfers 9->11 and ref recall 65.8->62.9 % - re-test with the final model
EMB_MIN_SAMPLES = 5
_EMB_INERT = os.environ.get("AA_IDENTITY_REID_INERT") == "1"  # diagnosis: embeddings loaded but never used
EMB_MIN_GAP_S = 4.0  # re-ID used only for links over at least this long a gap (long rounds, joint stage)
EMB_TEAM_KITS_ONLY = True  # no re-ID opinion for grey/black kits (referees)
EMB_ACHROMATIC_SHARE = 0.6
EMB_STRONG_MAX_HIST_L1 = 0.6  # kit agreement required for a re-ID-backed long link (same-person halves: 95 % < 0.62)
EMB_STRONG_MIN_HIST = 5
EMB_TEAM_KIT_MAX_ACH = 0.4  # re-ID only between pieces that are clearly team kits (p4400: a referee piece linked
# by re-ID to an unknown-kit piece lost its referee rows)
_N_ACH_BINS = 4  # team_appearance: the last 4 descriptor bins are black / dark grey / light grey / white
EMB_SAME_SIM = 0.9  # similarity at which the cost is 0
EMB_SCALE = 0.1
EMB_WEIGHT = 1.0
EMB_GATE_SIM = 0.6  # below: never the same person
EMB_STRONG_SIM = 0.97  # above: strong enough evidence for a long-gap link (0.95 raised team switches on c4400)
# long-gap rounds after the configured ones (CourtTrackConfig.link_rounds_s): player left the picture / long occlusion
LONG_ROUNDS_S: tuple[float, ...] = (10.0, 30.0, 120.0)
LONG_THRESHOLDS: tuple[float, ...] = (26.0, 28.0, 30.0)
LONG_MIN_DETS = 10  # pieces shorter than this are not linked over long gaps
LONG_MIN_CONF = 0.5  # ... nor pieces never seen with at least this detector score (crumbs)
LONG_WEAK_MAX_GAP_S = 0.0  # gaps up to this may be linked on kit + height + motion alone (0 = strong evidence only)
LONG_WEAK_MIN_HIST = 6
PILEUP_GUARD = False  # see _pileup_cuts (A/B: AA_TL_SET="PILEUP_GUARD=1")
PILEUP_MAX_DETS = 9
PILEUP_MIN_PIECES = 3
PILEUP_SPAN_S = 1.5
PILEUP_INNER_ONLY = False
PILEUP_MIN_HIST = 2  # kit histograms among the tiny pieces below this = no kit evidence
END_CRUMBS = os.environ.get("REPLAY_END_CRUMBS", "0") == "1"  # short re-appearances join the identity they continue
END_CRUMB_WINDOW_S = 1.0
END_CRUMB_MARGIN = 2.0
END_CRUMB_MAX_COST = 15.0
ANCHOR_MIN_READS = 2
ANCHOR_MAJORITY = 0.8
ANCHOR_MAX_HEIGHT_DIFF_M = 0.3
JOINT_AFTER_ROUNDS = os.environ.get("REPLAY_JOINT_AFTER", "1") == "1"  # joint stage after (not instead of) long rounds
JOINT_TRACKER = None  # None = CourtTrackConfig.joint_tracker; env REPLAY_JOINT=0/1 overrides both (A/B)


def reid_enabled(cfg) -> bool:
    """Re-ID cue on? env AA_IDENTITY_REID (0/1) first, then USE_REID (module A/B), then
    CourtTrackConfig.use_reid (``cfg`` = Config or CourtTrackConfig)."""
    env = os.environ.get("AA_IDENTITY_REID")
    if env:
        return env.strip() not in ("0", "false", "no")
    if USE_REID:
        return True
    return bool(getattr(getattr(cfg, "court_tracking", cfg), "use_reid", False))


def joint_enabled(cfg) -> bool:
    env = os.environ.get("REPLAY_JOINT")
    if env in ("0", "1"):
        return env == "1"
    if JOINT_TRACKER is not None:
        return bool(JOINT_TRACKER)
    return bool(getattr(cfg, "joint_tracker", False))
NUMBER_ANCHORS = True  # identities with the same well-read two-digit number (and compatible kit) are merged


def _cfg_has(cfg, name: str) -> bool:
    return any(f.name == name for f in fields(cfg))


class _LegacyCfg:
    """CourtTrackConfig view with the identity-round additions switched off (AA_IDENTITY_LEGACY)."""

    long_rounds_s: tuple = ()
    long_thresholds: tuple = ()
    chain_appearance_check = False
    number_anchors = False
    height_cue = False

    def __init__(self, cfg):
        self._cfg = cfg

    def __getattr__(self, name):
        return getattr(self._cfg, name)


@dataclass
class IdentityCues:
    """Optional per-detection identity cues for :func:`link_tracks` (rows aligned with its detections).

    hist: (n, DESC_DIM) kit colour histograms (team_appearance.Appearance.desc rows; NaN = not sampled);
    hist_occ: (n,) share of the shirt hidden by someone else (team_appearance occ; NaN = unknown);
    embeddings: (n, D) re-ID vectors (reid.ReidEmbeddings.emb rows; NaN = not embedded).
    Any field may be None; every cue is used only where present.
    """

    hist: np.ndarray | None = None
    hist_occ: np.ndarray | None = None
    embeddings: np.ndarray | None = None


# ----------------------------------------------------------------------------------------------- geometry
def _warp(T: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float).reshape(-1, 2)
    q = p @ T[:2, :2].T + T[:2, 2]
    w = p @ T[2, :2] + T[2, 2]
    return q / w[:, None]


class CameraFrames:
    """Pixel transforms between any two samples of an unbroken camera-motion chain.

    motion[i] maps sample i-1 pixels to sample i pixels (None = chain broken). Chaining hundreds of
    frame-to-frame homographies into one reference frame degenerates for a panning camera (perspective blows
    up), so transforms are kept local: each sample maps to an anchor at most ``block`` samples back, and
    anchors map to the previous anchor.
    """

    def __init__(self, motion: list | None, n: int, block: int = 32):
        self.seg = np.zeros(n, int)
        self.anchor = np.zeros(n, int)
        self.local: list[np.ndarray] = []
        self.up: dict[int, np.ndarray] = {}  # anchor sample -> previous anchor's pixels
        self._cache: dict[tuple[int, int], np.ndarray | None] = {}
        s = -1
        for i in range(n):
            m = None if motion is None or i >= len(motion) else motion[i]
            inv = None
            if i > 0 and m is not None and np.isfinite(m).all():
                try:
                    inv = np.linalg.inv(m)
                except np.linalg.LinAlgError:
                    inv = None
            if inv is None:  # chain broken: new segment, new anchor
                s += 1
                self.anchor[i] = i
                self.local.append(np.eye(3))
            elif i - self.anchor[i - 1] >= block:  # new anchor within the segment
                self.anchor[i] = i
                g = self.local[i - 1] @ inv
                self.up[i] = g / g[2, 2]
                self.local.append(np.eye(3))
            else:
                self.anchor[i] = self.anchor[i - 1]
                g = self.local[i - 1] @ inv
                self.local.append(g / g[2, 2])
            self.seg[i] = s

    def _to_anchor(self, i: int, target_anchor: int) -> np.ndarray:
        T = self.local[i]
        a = int(self.anchor[i])
        while a > target_anchor:
            T = self.up[a] @ T
            a = int(self.anchor[a - 1])
        return T

    def transform(self, i: int, j: int) -> np.ndarray | None:
        """3x3 mapping sample-i pixels to sample-j pixels, or None across a broken chain."""
        i, j = int(i), int(j)
        key = (i, j)
        if key in self._cache:
            return self._cache[key]
        if self.seg[i] != self.seg[j]:
            T = None
        elif i == j:
            T = np.eye(3)
        else:
            lo = min(int(self.anchor[i]), int(self.anchor[j]))
            T = np.linalg.solve(self._to_anchor(j, lo), self._to_anchor(i, lo))
            T = T / T[2, 2]
        if len(self._cache) > 200_000:
            self._cache.clear()
        self._cache[key] = T
        return T


# ----------------------------------------------------------------------------------------------- data
@dataclass
class _Piece:
    det: np.ndarray  # detection indices, sorted by sample
    k0: int = 0
    k1: int = 0
    # court end states: position, velocity, position variance (per axis)
    c0: np.ndarray | None = None
    c1: np.ndarray | None = None
    cv0: np.ndarray | None = None
    cv1: np.ndarray | None = None
    cvar0: float = 1.0
    cvar1: float = 1.0
    # image end states in the pixels of sample k0 / k1: head position, velocity (px/s), full body height (px)
    p0: np.ndarray | None = None
    p1: np.ndarray | None = None
    pv0: np.ndarray | None = None
    pv1: np.ndarray | None = None
    h0: float = np.nan
    h1: float = np.nan
    app: np.ndarray | None = None
    app_var: np.ndarray | None = None
    numbers: dict | None = None  # jersey number code -> reads, numbers read at least twice (jersey_numbers.py)
    n_app: int = 0  # detections behind ``app`` (clean, unoccluded boxes)
    # optional richer cues (None / 0 when the input is missing)
    hist: np.ndarray | None = None  # mean kit colour histogram (team_appearance descriptors) of clean boxes
    n_hist: int = 0
    height_m: float = np.nan  # body height in metres (box height x ground scale at the foot), clean boxes only
    n_height: int = 0
    emb: np.ndarray | None = None  # mean re-ID embedding (unit vector), reid.py
    n_emb: int = 0
    conf_max: float = 0.0
    link_log: list = field(default_factory=list)  # (cost, gap s, threshold, strong) of the links inside this piece


def _median(a: np.ndarray):
    """``np.median(a, axis=0)`` for small float arrays without its wrapper overhead (bit-identical: odd n picks the
    same element, even n gives (a+b)/2 like np.mean of the two middles; a NaN in a column gives NaN)."""
    if a.dtype.kind != "f" or not len(a):
        return np.median(a, axis=0)
    s = np.sort(a, axis=0)
    n = len(s)
    m = s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2
    if s.ndim == 1:
        return s[-1] if np.isnan(s[-1]) else m
    return np.where(np.isnan(s[-1]), s[-1], m)


def _norm2(v: np.ndarray) -> float:
    """``float(np.linalg.norm(v))`` for a float64 2-vector as scalar math (bit-identical: sqrt(x*x + y*y))."""
    if v.dtype != np.float64 or v.shape != (2,):
        return float(np.linalg.norm(v))
    x, y = float(v[0]), float(v[1])
    return math.sqrt(x * x + y * y)


def _line_fit(t: np.ndarray, z: np.ndarray, vmax: float) -> tuple[np.ndarray, np.ndarray]:
    """Position at t=0 and velocity from a straight-line fit (velocity capped at vmax)."""
    if len(t) < 3 or np.ptp(t) == 0:
        return z[np.argmin(np.abs(t))].copy(), np.zeros(z.shape[1])
    coef, *_ = np.linalg.lstsq(np.c_[np.ones_like(t), t], z, rcond=None)
    v = coef[1]
    sp = _norm2(v)
    if np.isfinite(vmax) and sp > vmax:
        v = v * vmax / sp
    return coef[0], v


def _rolling_upper(kk: np.ndarray, v: np.ndarray, half: int, q: float) -> np.ndarray:
    """q-th percentile of v over samples within +-half of each sample.

    Vectorised; reproduces ``np.percentile(v[lo:hi], q)`` (method "linear") exactly, NaN windows give NaN.
    """
    n = len(kk)
    if not n:
        return np.empty(0)
    v = np.asarray(v, float)
    lo = np.searchsorted(kk, kk - half, "left")
    hi = np.searchsorted(kk, kk + half, "right")
    cnt = hi - lo
    cols = np.arange(int(cnt.max()))
    inside = cols[None, :] < cnt[:, None]
    M = np.where(inside, v[np.minimum(lo[:, None] + cols[None, :], n - 1)], np.inf)
    M.sort(axis=1)
    virt = (cnt - 1) * (q / 100)  # numpy: (n - 1) * quantile, floor/next index, lerp with gamma
    prev = np.floor(virt)
    above = virt >= cnt - 1
    pi = np.where(above, cnt - 1, prev).astype(np.intp)
    ni = np.where(above, cnt - 1, prev + 1).astype(np.intp)
    gamma = virt - np.where(above, -1.0, prev)
    r = np.arange(n)
    a, b = M[r, pi], M[r, ni]
    with np.errstate(invalid="ignore"):  # NaN / inf only in windows that become NaN below
        diff = b - a
        out = np.where(gamma >= 0.5, b - diff * (1 - gamma), a + diff * gamma)
    nan = np.concatenate([[0], np.cumsum(np.isnan(v))])
    out[(nan[hi] - nan[lo]) > 0] = np.nan
    return out


class _Data:
    def __init__(self, k, xy, R, feats, conf, fps, cfg, boxes=None, motion=None, n_samples=None, H=None,
                 numbers=None, cues: "IdentityCues | None" = None):
        n = len(k)
        self.k, self.feats, self.conf, self.fps, self.cfg = k, feats, conf, fps, cfg
        self.numbers = None if numbers is None else np.asarray(numbers, int).reshape(n)  # jersey number reads
        self.xy, self.R = xy.copy(), R.copy()
        self.boxes, self.H = boxes, H
        self.has_court = np.isfinite(self.xy).all(axis=1)
        self.has_img = boxes is not None
        self.partial = np.zeros(n, bool)
        self.stats: dict = {}
        self.link_info: dict = {}
        self._summary_cache: dict = {}
        self.head = self.h = self.full_h = self.cam = None
        self._head_cache: dict[tuple[int, int], tuple[float, np.ndarray | None]] = {}
        if self.has_img:
            ns = int(n_samples if n_samples is not None else (k.max() + 1 if n else 0))
            self.cam = CameraFrames(motion, ns)
            self.head = np.c_[(boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 1]]  # own-frame pixels
            self.h = np.maximum(1.0, boxes[:, 3] - boxes[:, 1])
            self.full_h = self.h.copy()
        self.rvar = np.full(n, np.nan)
        self._update_rvar(np.arange(n))
        self.feet_hidden = np.zeros(n, bool)  # someone in front covers the bottom of the box
        self.occluded = np.zeros(n, bool)  # box overlaps another person's box (where ByteTrack may swap IDs)
        if boxes is not None and n:
            order = np.argsort(k, kind="stable")
            bounds = np.flatnonzero(np.diff(k[order])) + 1
            for grp in np.split(order, bounds):
                if len(grp) < 2:
                    continue
                b = boxes[grp]
                ix = np.clip(np.minimum(b[:, None, 2], b[None, :, 2]) - np.maximum(b[:, None, 0], b[None, :, 0]), 0, None)
                iy = np.clip(np.minimum(b[:, None, 3], b[None, :, 3]) - np.maximum(b[:, None, 1], b[None, :, 1]), 0, None)
                area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
                frac = ix * iy / np.maximum(1.0, np.minimum(area[:, None], area[None, :]))
                np.fill_diagonal(frac, 0.0)
                self.occluded[grp] = frac.max(axis=1) > cfg.occlusion_overlap
                # bottom of the box hidden by someone standing in front (their box reaches lower in the image)
                covers = (ix > 0.3 * (b[:, 2] - b[:, 0])[:, None]) & (b[None, :, 3] > b[:, None, 3] + 2)                     & (b[None, :, 1] < b[:, None, 3])
                np.fill_diagonal(covers, False)
                self.feet_hidden[grp] = covers.any(axis=1)
        # optional cues (IdentityCues): kit histograms, re-ID embeddings; body height in metres from calibration
        self.hist = self.hist_ok = self.emb = self.height_m = None
        if cues is not None and cues.hist is not None:
            self.hist = np.asarray(cues.hist, float).reshape(n, -1)
            ok = np.isfinite(self.hist).all(axis=1)
            if cues.hist_occ is not None:
                ok &= np.nan_to_num(np.asarray(cues.hist_occ, float), nan=1.0) < _HIST_MAX_OCC
            else:
                ok &= ~self.occluded
            self.hist_ok = ok
            if not ok.any():
                self.hist = self.hist_ok = None
        if cues is not None and cues.embeddings is not None:
            e = np.asarray(cues.embeddings, float).reshape(n, -1)
            if np.isfinite(e).all(axis=1).any():
                self.emb = e
        if boxes is not None and H is not None and n and getattr(cfg, "height_cue", True):
            b = boxes
            sc = self._ground_scales(np.arange(n), np.c_[(b[:, 0] + b[:, 2]) / 2, b[:, 3]])
            hm = (b[:, 3] - b[:, 1]) * sc
            self.height_m = np.where((hm > 0.5) & (hm < 3.0), hm, np.nan)

    def _update_rvar(self, idx):
        if len(idx):
            self.rvar[idx] = np.where(self.has_court[idx], np.trace(self.R[idx], axis1=1, axis2=2) / 2, np.nan)

    def head_in(self, d: int, j: int) -> np.ndarray | None:
        """[x, y, full height] of detection d's head point expressed in the pixels of sample j."""
        key = (int(d), int(j))
        hit = self._head_cache.get(key)
        if hit is not None and hit[0] == self.full_h[d]:  # full_h changes when partial boxes are corrected
            return None if hit[1] is None else hit[1].copy()
        T = self.cam.transform(int(self.k[d]), j)
        if T is None:
            out = None
        else:
            x, y = self.head[d]
            q = _warp(T, np.array([[x, y], [x, y + self.full_h[d]]]))
            out = np.array([q[0, 0], q[0, 1], np.hypot(*(q[1] - q[0]))])
        if len(self._head_cache) > 500_000:
            self._head_cache.clear()
        self._head_cache[key] = (float(self.full_h[d]), out)
        return None if out is None else out.copy()

    def heads_in(self, dets: np.ndarray, j: int) -> tuple[np.ndarray, np.ndarray]:
        """Vectorised ``head_in`` for many detections: ((m, 3) [x, y, full height] in sample j's pixels, valid)."""
        m = len(dets)
        uk, inv = np.unique(self.k[dets], return_inverse=True)  # one transform per sample, not per detection
        Tu = [self.cam.transform(int(kd), j) for kd in uk]
        oku = np.array([T is not None for T in Tu], bool)
        ok = oku[inv]
        out = np.full((m, 3), np.nan)
        if not ok.any():
            return out, ok
        Tstack = np.stack([t if t is not None else np.eye(3) for t in Tu])
        T = Tstack[inv[ok]]
        d = dets[ok]
        x, y = self.head[d, 0], self.head[d, 1]
        y2 = y + self.full_h[d]
        out_pts = []
        for py in (y, y2):  # same 2-term sums as _warp / einsum
            qx = x * T[:, 0, 0] + py * T[:, 0, 1] + T[:, 0, 2]
            qy = x * T[:, 1, 0] + py * T[:, 1, 1] + T[:, 1, 2]
            w = x * T[:, 2, 0] + py * T[:, 2, 1] + T[:, 2, 2]
            out_pts.append((qx / w, qy / w))
        (ax, ay), (bx, by) = out_pts
        out[ok, 0], out[ok, 1] = ax, ay
        out[ok, 2] = np.hypot(bx - ax, by - ay)
        return out, ok

    def _ground_scale(self, d: int, foot: tuple[float, float]) -> float:
        """Court metres per pixel along the image x axis at a foot pixel (NaN without calibration)."""
        Hs = None if self.H is None else self.H[int(self.k[d])]
        if Hs is None:
            return np.nan
        s = _norm2(homography.jacobian(Hs, foot)[:, 0])
        return s if np.isfinite(s) and s > 0 else np.nan

    def _ground_scales(self, det: np.ndarray, feet: np.ndarray) -> np.ndarray:
        """Vectorised ``_ground_scale`` for many detections (same arithmetic as homography.jacobian)."""
        out = np.full(len(det), np.nan)
        if self.H is None or not len(det):
            return out
        Hl = [self.H[int(k)] for k in self.k[det]]
        ok = np.array([h is not None for h in Hl])
        if not ok.any():
            return out
        Hs = np.stack([h for h in Hl if h is not None]).astype(float)
        step = 0.5
        u, v = feet[ok, 0], feet[ok, 1]
        pts = np.stack([np.c_[u + step, v], np.c_[u - step, v], np.c_[u, v + step], np.c_[u, v - step]], axis=1)
        q = np.concatenate([pts, np.ones(pts.shape[:2] + (1,))], axis=2) @ np.swapaxes(Hs, 1, 2)
        w = q[..., 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            p = q[..., :2] / w[..., None]
            p[w <= 1e-9] = np.nan  # homography.project: behind the horizon / degenerate
            c0 = (p[:, 0] - p[:, 1]) / (2 * step)
            s = np.sqrt(c0[:, 0] * c0[:, 0] + c0[:, 1] * c0[:, 1])
        out[ok] = np.where(np.isfinite(s) & (s > 0), s, np.nan)
        return out

    def correct_partial(self, det: np.ndarray) -> None:
        """Re-estimate the foot point of boxes cut short (legs hidden) from the piece's full body height.

        A player walking away from the camera shrinks in the image, so heights are compared after scaling by
        the ground scale at the foot (pixel height x metres-per-pixel is roughly constant for one person).
        """
        if not self.has_img or len(det) < 3:
            return
        cfg = self.cfg
        kk = self.k[det]
        h = self.h[det]
        b = self.boxes[det]
        scale = self._ground_scales(det, np.c_[(b[:, 0] + b[:, 2]) / 2, b[:, 3]]) if cfg.partial_depth_scale             else np.full(len(det), np.nan)
        fin = np.isfinite(scale)
        scale = np.where(fin, scale, _median(scale[fin]) if fin.any() else 1.0)
        # boxes cut at the knees put the "foot" higher (= further away): a rolling median keeps the real feet's scale
        scale = _rolling_upper(kk, scale, max(1, int(round(0.5 * self.fps))), 50)
        hn = h * scale
        full_n = _rolling_upper(kk, hn, max(2, int(round(cfg.height_window_s * self.fps))), cfg.full_height_percentile)
        full = full_n / scale
        part = hn < cfg.partial_height_ratio * full_n
        self.full_h[det] = np.where(part, full, np.maximum(full, h))
        self.partial[det] = part
        if self.H is None or not cfg.correct_partial_feet:
            return
        for j in np.where(part & self.feet_hidden[det])[0]:  # only where an occluder explains the short box
            d = det[j]
            Hs = self.H[int(self.k[d])]
            if Hs is None:
                continue
            x1, y1, x2, y2 = self.boxes[d]
            cx = (x1 + x2) / 2
            full_px = full[j]
            for _ in range(2):  # the scale at the corrected foot differs a little from the one at the knees
                sc = self._ground_scale(d, (cx, y1 + full_px))
                if np.isfinite(sc):
                    full_px = full_n[j] / sc
            foot = (cx, y1 + full_px)
            p = homography.project(Hs, np.array([foot]))[0]
            if not np.isfinite(p).all():
                continue
            J = homography.jacobian(Hs, foot)
            s_px = cfg.partial_foot_sigma_rel * full_px
            R = J @ J.T * s_px ** 2 + np.eye(2) * _MIN_MEAS_SIGMA_M ** 2
            if not np.isfinite(R).all():
                continue
            self.xy[d], self.R[d] = p, R
            self.full_h[d] = full_px
            self.has_court[d] = True
        self._update_rvar(det)

    def _img_end(self, det: np.ndarray, at_start: bool):
        kk = self.k[det]
        w = max(1, int(round(self.cfg.end_window_s * self.fps)))
        ref = int(kk[0] if at_start else kk[-1])
        sel = det[kk <= kk[0] + w] if at_start else det[kk >= kk[-1] - w]
        z, ok = self.heads_in(sel, ref)
        if not ok.any():
            return None, None, np.nan
        z = z[ok]
        ts = (self.k[sel[ok]] - ref) / self.fps
        hh = float(_median(z[:, 2]))
        p, v = _line_fit(np.array(ts), z[:, :2], self.cfg.max_speed_bh_s * hh)
        return p, v, hh

    def summarise(self, det: np.ndarray) -> _Piece:
        det = det[np.argsort(self.k[det], kind="stable")]
        key = det.tobytes()
        hit = self._summary_cache.get(key)
        fh = None if self.full_h is None else self.full_h[det]  # (no boxes: no full heights, court cues only)
        if hit is not None and (fh is None or np.array_equal(hit[0], fh)):
            p = copy.copy(hit[1])
            p.link_log = []
            return p
        p = self._summarise(det)
        self._summary_cache[key] = (None if fh is None else fh.copy(), p)
        out = copy.copy(p)
        out.link_log = []
        return out

    def _summarise(self, det: np.ndarray) -> _Piece:
        p = _Piece(det=det, k0=int(self.k[det[0]]), k1=int(self.k[det[-1]]))
        cfg, fps = self.cfg, self.fps
        kk = self.k[det]
        w = max(1, int(round(cfg.end_window_s * fps)))
        c = self.has_court[det]
        if c.any():
            dc, kc = det[c], kk[c]
            s0, s1 = kc <= kc[0] + w, kc >= kc[-1] - w
            p.c0, p.cv0 = _line_fit((kc[s0] - kc[0]) / fps, self.xy[dc[s0]], cfg.max_speed_mps)
            p.c1, p.cv1 = _line_fit((kc[s1] - kc[-1]) / fps, self.xy[dc[s1]], cfg.max_speed_mps)
            # end position only as good as the measurements; extrapolating to an end without court position adds error
            e0, e1 = (kc[0] - kk[0]) / fps, (kk[-1] - kc[-1]) / fps
            p.cvar0 = float(_median(self.rvar[dc[s0]])) + (cfg.vel_sigma_mps * e0) ** 2
            p.cvar1 = float(_median(self.rvar[dc[s1]])) + (cfg.vel_sigma_mps * e1) ** 2
            p.c0 = p.c0 - p.cv0 * e0
            p.c1 = p.c1 + p.cv1 * e1
        if self.has_img:
            p.p0, p.pv0, p.h0 = self._img_end(det, True)
            p.p1, p.pv1, p.h1 = self._img_end(det, False)
        f = self.feats[det]
        good = np.isfinite(f).all(axis=1)
        for bad in (self.partial[det], self.occluded[det]):  # boxes with someone else's colours in them
            if (good & ~bad).sum() >= 3:
                good &= ~bad
        f = f[good]
        p.n_app = int(len(f))
        if len(f):
            p.app = _median(f)
            p.app_var = _APP_SIGMA ** 2 / min(len(f), 15) + _APP_FLOOR ** 2
        if self.numbers is not None:
            from replay.jersey_numbers import piece_numbers

            p.numbers = piece_numbers(self.numbers[det])
        p.conf_max = float(self.conf[det].max())
        clean = ~self.partial[det] & ~self.occluded[det]
        if self.hist is not None:
            ok = self.hist_ok[det] & ~self.partial[det]
            if ok.sum() >= 1:
                p.hist = self.hist[det[ok]].mean(axis=0)
                p.n_hist = int(ok.sum())
        if self.height_m is not None:
            hm = self.height_m[det]
            ok = np.isfinite(hm) & clean & ~self.feet_hidden[det]
            if ok.sum() >= 3:
                p.height_m = float(_median(hm[ok]))
                p.n_height = int(ok.sum())
        if self.emb is not None:
            e = self.emb[det]
            ok = np.isfinite(e).all(axis=1) & clean
            if ok.sum() >= 1:
                m = e[ok].mean(axis=0)
                nm = _norm2(m)
                if nm > 1e-6:
                    p.emb = m / nm
                    p.n_emb = int(ok.sum())
        return p


# ----------------------------------------------------------------------------------------------- tracklets
def _img_dist(D: _Data, a: int, b: int) -> float | None:
    """Head distance between detections a and b (in b's frame), in body heights; None across a broken chain."""
    q = D.head_in(a, int(D.k[b]))
    if q is None:
        return None
    dx, dy = D.head[b, 0] - q[0], D.head[b, 1] - q[1]
    return math.sqrt(dx * dx + dy * dy) / max(1.0, 0.5 * (q[2] + D.full_h[b]))


def _jump_cuts(D: _Data, det: np.ndarray, use_image: bool, use_court: bool) -> list[int]:
    """Positions (index into det) where consecutive detections are physically impossible -> cut before them."""
    cfg, fps = D.cfg, D.fps
    kk = D.k[det]
    cuts = []
    for j in range(1, len(det)):
        a, b = det[j - 1], det[j]
        dt = (kk[j] - kk[j - 1]) / fps
        if use_image:
            d = _img_dist(D, a, b)
            if d is not None and d > cfg.max_speed_bh_s * dt + cfg.reach_slack_bh:
                cuts.append(j)
                continue
        if use_court and D.has_court[a] and D.has_court[b]:
            d = _norm2(D.xy[b] - D.xy[a])
            if d > cfg.max_speed_mps * dt + cfg.reach_slack_m + 3.0 * np.sqrt(D.rvar[a] + D.rvar[b]):
                cuts.append(j)
    return cuts


def _window_medians(f: np.ndarray, fin: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Column medians of the finite rows of f[lo[r]:hi[r]] for every r (as np.median), and their counts."""
    width = int((hi - lo).max())
    idx = lo[:, None] + np.arange(width)[None, :]
    ok = idx < hi[:, None]
    idx = np.minimum(idx, len(f) - 1)
    ok &= fin[idx]
    c = ok.sum(axis=1)
    G = np.where(ok[:, :, None], f[idx], np.inf)
    G.sort(axis=1)
    r = np.arange(len(lo))
    a = G[r, np.maximum(c - 1, 0) // 2]
    b = G[r, c // 2]
    with np.errstate(invalid="ignore"):
        med = np.where((c % 2 == 1)[:, None], a, (a + b) / 2)
    return med, c


def _appearance_cuts(f: np.ndarray, w: int, thr: float, min_len: int, allowed: np.ndarray | None = None) -> list[int]:
    """Change points of the jersey colour (ByteTrack swapped identities during an overlap). f may hold NaN rows.

    allowed[j]: a cut before j is possible (ID swaps only happen where boxes overlap).
    """
    n = len(f)
    if n < 2 * min_len:
        return []
    fin = np.isfinite(f).all(axis=1)
    best, best_j = 0.0, -1
    js = np.arange(min_len, n - min_len + 1)
    if allowed is not None:
        js = js[np.asarray(allowed, bool)[js]]
    if len(js):
        ma, ca = _window_medians(f, fin, np.maximum(0, js - w), js)
        mb, cb = _window_medians(f, fin, js, np.minimum(n, js + w))
        for r in np.flatnonzero((ca >= min_len) & (cb >= min_len)):
            d = float(np.linalg.norm((ma[r] - mb[r]) * _SPLIT_W))
            if d > best:
                best, best_j = d, int(js[r])
    if best < thr:
        return []
    al, ar = (None, None) if allowed is None else (allowed[:best_j], allowed[best_j:])
    left = _appearance_cuts(f[:best_j], w, thr, min_len, al)
    right = _appearance_cuts(f[best_j:], w, thr, min_len, ar)
    return left + [best_j] + [best_j + r for r in right]


def _chain_unassigned(D: _Data, idx: np.ndarray) -> list[np.ndarray]:
    """Optimal frame-to-frame chaining of the detections ByteTrack left without an ID."""
    if not len(idx):
        return []
    cfg = D.cfg
    idx = idx[np.argsort(D.k[idx], kind="stable")]
    chains: list[list[int]] = []
    active: list[int] = []
    max_skip = max(1, int(round(cfg.chain_max_gap_s * D.fps)))
    frames, starts = np.unique(D.k[idx], return_index=True)  # idx sorted by k: each frame is one slice
    ends = np.r_[starts[1:], len(idx)]
    for f, s0, s1 in zip(frames, starts, ends):
        cur = idx[s0:s1]
        active = [c for c in active if f - D.k[chains[c][-1]] <= max_skip]
        used: set[int] = set()
        if active:
            C = np.full((len(active), len(cur)), 1e6)
            for a, c in enumerate(active):
                last = chains[c][-1]
                dt = (f - D.k[last]) / D.fps
                for b, d in enumerate(cur):
                    di = _img_dist(D, last, d) if D.has_img else None
                    if di is not None:
                        if di > 0.3 + 2.0 * dt or abs(np.log(D.h[d] / D.h[last])) > 0.3:
                            continue
                        C[a, b] = di
                    elif D.has_court[d] and D.has_court[last]:
                        dc = _norm2(D.xy[d] - D.xy[last])
                        if dc <= cfg.max_speed_mps * dt + cfg.reach_slack_m:
                            C[a, b] = dc
            rows, cols = linear_sum_assignment(C)
            for a, b in zip(rows, cols):
                if C[a, b] < 1e6:
                    chains[active[a]].append(int(cur[b]))
                    used.add(int(b))
        for b, d in enumerate(cur):
            if b not in used:
                chains.append([int(d)])
                active.append(len(chains) - 1)
    return [np.array(c) for c in chains]


def build_tracklets(D: _Data, bt_ids: np.ndarray) -> list[np.ndarray]:
    cfg = D.cfg
    order = np.argsort(D.k, kind="stable")
    bt_sorted = bt_ids[order]
    raw: list[np.ndarray] = []
    o2 = np.argsort(bt_sorted, kind="stable")
    s2 = bt_sorted[o2]
    for b in np.unique(bt_ids[bt_ids >= 0]):
        det = order[o2[np.searchsorted(s2, b):np.searchsorted(s2, b, side="right")]]
        _, first = np.unique(D.k[det], return_index=True)  # ByteTrack: one box per frame and ID
        det = det[np.sort(first)]
        # the head is visible even when the legs are hidden: image continuity first, the court after correction
        cuts = _jump_cuts(D, det, use_image=D.has_img, use_court=not D.has_img)
        D.stats["cuts_image"] = D.stats.get("cuts_image", 0) + len(cuts)
        raw.extend(p for p in np.split(det, cuts) if len(p))
    chained = _chain_unassigned(D, order[bt_sorted < 0])
    for p in raw + chained:
        D.correct_partial(p)
    w = max(3, int(round(cfg.appearance_window_s * D.fps)))
    cut_at: list[set[int]] = []  # per raw piece: samples where a new piece starts
    for p in raw:
        cuts: set[int] = set()
        parts = [p]
        if D.has_img and D.H is not None:
            jc = _jump_cuts(D, p, use_image=False, use_court=True)
            D.stats["cuts_court"] = D.stats.get("cuts_court", 0) + len(jc)
            cuts.update(int(D.k[p[j]]) for j in jc)
            parts = [q for q in np.split(p, jc) if len(q)]
        if cfg.appearance_split:
            for q in parts:
                f = D.feats[q].copy()
                f[D.partial[q]] = np.nan  # a partial box's colour regions are misplaced
                near = np.convolve(D.occluded[q].astype(float), np.ones(5), "same") > 0  # overlap within +-2 samples
                ac = _appearance_cuts(f, w, cfg.appearance_split_lab, max(3, w // 2), near if D.has_img else None)
                D.stats["cuts_colour"] = D.stats.get("cuts_colour", 0) + len(ac)
                cuts.update(int(D.k[q[j]]) for j in ac)
        cut_at.append(cuts)
    if D.has_img and cfg.appearance_split:
        before = sum(map(len, cut_at))
        _partner_cuts(D, raw, cut_at)
        D.stats["cuts_partner"] = sum(map(len, cut_at)) - before
    D.stats["unassigned_chains"] = len(chained)
    out = list(chained)
    for p, cuts in zip(raw, cut_at):
        idx = [j for j in range(1, len(p)) if int(D.k[p[j]]) in cuts]
        out.extend(q for q in np.split(p, idx) if len(q))
    return out


def _partner_cuts(D: _Data, raw: list[np.ndarray], cut_at: list[set[int]]) -> None:
    """A colour change in one ByteTrack ID at an overlap means the overlapping person's ID was swapped too:
    cut that one at the same moment, so the linker can re-pair both sides of the crossing."""
    at_sample: dict[int, list[tuple[int, int]]] = {}
    for pi, p in enumerate(raw):
        for d in p:
            at_sample.setdefault(int(D.k[d]), []).append((pi, int(d)))
    det_of = [{int(D.k[d]): int(d) for d in p} for p in raw]
    for pi, cuts in enumerate(cut_at):
        for s in list(cuts):
            d = det_of[pi].get(s)
            if d is None:
                continue
            bx = D.boxes[d]
            for qi, e in at_sample.get(s, []):
                if qi == pi or s in cut_at[qi]:
                    continue
                be = D.boxes[e]
                ix = min(bx[2], be[2]) - max(bx[0], be[0])
                iy = min(bx[3], be[3]) - max(bx[1], be[1])
                small = min((bx[2] - bx[0]) * (bx[3] - bx[1]), (be[2] - be[0]) * (be[3] - be[1]))
                if ix > 0 and iy > 0 and ix * iy / max(small, 1.0) > D.cfg.occlusion_overlap:
                    cut_at[qi].add(s)


# ----------------------------------------------------------------------------------------------- linking
def _link_cost(D: _Data, a: _Piece, b: _Piece, max_gap: int) -> float | None:
    """Cost of 'b continues a' (lower is better) or None when impossible."""
    cfg, fps = D.cfg, D.fps
    gap_k = b.k0 - a.k1
    if gap_k > max_gap or b.k0 <= a.k0 or b.k1 <= a.k1:
        return None
    if gap_k <= 0:  # overlapping pieces: allowed only for a few samples and only if they coincide there
        if -gap_k + 1 > cfg.max_overlap_samples:
            return None
        return _overlap_cost(D, a, b)
    g = gap_k / fps
    tau = min(g, cfg.bridge_velocity_s)
    cost = cfg.gap_cost_per_s * g
    evidence = False
    if a.c1 is not None and b.c0 is not None:
        d = b.c0 - a.c1
        if _norm2(d) > cfg.max_speed_mps * g + cfg.reach_slack_m + 2.0 * np.sqrt(a.cvar1 + b.cvar0):
            return None
        r = d - 0.5 * (a.cv1 + b.cv0) * tau
        var = a.cvar1 + b.cvar0 + cfg.pos_sigma_m ** 2 + (cfg.vel_sigma_mps * g) ** 2
        cost += 0.5 * float(r @ r) / var + np.log(var / cfg.pos_sigma_m ** 2)
        evidence = True
    T = D.cam.transform(a.k1, b.k0) if D.has_img and a.p1 is not None and b.p0 is not None else None
    if T is not None:
        q = _warp(T, np.array([a.p1, a.p1 + a.pv1, a.p1 + (0.0, a.h1)]))
        pa, va, ha = q[0], q[1] - q[0], _norm2(q[2] - q[0])
        hh = max(1.0, 0.5 * (ha + b.h0))
        d = (b.p0 - pa) / hh
        if _norm2(d) > cfg.max_speed_bh_s * g + cfg.reach_slack_bh:
            return None
        r = d - 0.5 * (va + b.pv0) / hh * tau
        var = cfg.img_sigma_bh ** 2 + (cfg.img_vel_sigma_bh * g) ** 2
        wimg = cfg.image_weight if evidence else 1.0
        cost += wimg * (0.5 * float(r @ r) / var + np.log(var / cfg.img_sigma_bh ** 2))
        if not evidence:  # image space alone is weaker evidence (depth is ambiguous, pans): be more careful
            cost += cfg.image_only_cost_per_s * g
        hr = abs(np.log(max(b.h0, 1.0) / max(ha, 1.0)))
        h_lim = cfg.max_height_log_ratio + 0.05 * g
        if hr > h_lim:
            # a box cut short (partial / feet hidden / overlapped by someone) at either end makes the end
            # height unreliable: soft cost only (scoreboard 19.10: most missed short-gap handovers failed here)
            if not (HEIGHT_SOFT_AT_PARTIAL and _both_on_court(a, b)
                    and (_end_unreliable(D, a, False) or _end_unreliable(D, b, True))
                    and _same_player_evidence(D, a, b) and not _bench_guarded(D, a, b)):
                return None
            hr = h_lim
        cost += cfg.height_weight * hr / 0.15
        evidence = True
    if not evidence:
        return None
    app = _app_cost(D, a, b, g)
    return None if app is None else cost + app


HEIGHT_SOFT_AT_PARTIAL = False  # off until gated (recall -1.1 pt, phantoms +0.2 in hg-on-s3); A/B: AA_TL_SET="HEIGHT_SOFT_AT_PARTIAL=1"
HEIGHT_SOFT_OCCLUDED = True  # also when an end box merely overlaps another person's box

# bench guard (replay/bench_guard.py, data/plans/bench_guard.md (a)): a piece whose WHOLE life looks like a bench
# person (court feet in the sideline strip next to the bench, never deep in the court, no travel, slow) is never
# soft-height-linked into or out of (height-gate rounds 1-3: same-kit bench kids became chimeras of real players).
# Default OFF: env REPLAY_BENCH_GUARD=1 or AA_TL_SET="BENCH_GUARD=1".
BENCH_GUARD = False
BENCH_GUARD_SCORE_MIN = 0.6  # besides is_bench: score >= this on >= BENCH_GUARD_MIN_ROWS court rows also counts
BENCH_GUARD_MIN_ROWS = 20


def bench_guard_enabled() -> bool:
    v = os.environ.get("REPLAY_BENCH_GUARD", "").strip().lower()
    return v in ("1", "true", "yes") if v else BENCH_GUARD


def _bench_piece(D: _Data, p: _Piece) -> bool:
    """Whole-piece bench-likeness (bench_guard.score_track on the court feet of the piece), cached per piece."""
    cache = getattr(D, "_bench_cache", None)
    if cache is None:
        cache = D._bench_cache = {}
    key = (int(p.k0), int(p.k1), len(p.det), int(p.det[0]), int(p.det[-1]))
    v = cache.get(key)
    if v is None:
        from replay import bench_guard, court

        xy = D.xy[p.det]
        ok = np.isfinite(xy).all(axis=1)
        if ok.sum() < bench_guard.MIN_ROWS:
            v = False
        else:
            tt = D.k[p.det][ok] / D.fps
            rows = [(float(t), float(x), float(y), True) for t, (x, y) in zip(tt, xy[ok])]
            s = bench_guard.score_track(rows, court.LENGTH, court.WIDTH)
            v = bool(s.is_bench or (s.score >= BENCH_GUARD_SCORE_MIN and s.rows >= BENCH_GUARD_MIN_ROWS))
        cache[key] = v
    return v


def _bench_guarded(D: _Data, a: _Piece, b: _Piece) -> bool:
    """Guard on and either piece bench-like: the soft height gate must not join them."""
    return bench_guard_enabled() and (_bench_piece(D, a) or _bench_piece(D, b))


def _both_on_court(a: _Piece, b: _Piece) -> bool:
    """Both ends inside the court lines: seated bench people / coaches along the sideline (often cut short behind
    the bench) keep the hard height gate - soft gating there linked referees and players to them."""
    if a.c1 is None or b.c0 is None:
        return False
    from replay import court

    return bool(court.in_court(np.array([a.c1, b.c0], float), 0.0).all())


HEIGHT_SOFT_MAX_KIT_L1 = 0.6  # "well measured kit match" (same-person halves: 95 % below 0.62)
HEIGHT_SOFT_MIN_SPEED = 1.0  # m/s at both ends = moving like a player
HEIGHT_SOFT_EVIDENCE = True  # A/B: 0 = court guard only (the first version)
HEIGHT_SOFT_MOTION_EVIDENCE = True  # A/B: 0 = strong identity evidence only (kit match / same number), no motion-only


def _stationary_or_bench(D: _Data, p: _Piece, end) -> bool:
    """Long-stationary piece (> 5 s within 1.5 m) or an end standing (< 1 m/s) within 0.75 m of a sideline."""
    from replay import court

    xy = D.xy[p.det]
    xy = xy[np.isfinite(xy).all(axis=1)]
    if (p.k1 - p.k0) / D.fps > 5.0 and len(xy) and float(np.ptp(xy, axis=0).max()) < 1.5:
        return True
    c, v = end
    if c is None:
        return True
    slow = v is None or _norm2(np.asarray(v)) < HEIGHT_SOFT_MIN_SPEED
    return bool(slow and (c[1] < 0.75 or c[1] > court.WIDTH - 0.75))


def _same_player_evidence(D: _Data, a: _Piece, b: _Piece) -> bool:
    """Positive evidence that b continues a, required before a cut-short end may skip the height gate: a well
    measured kit match (>= 3 histograms each), the same dominant jersey number, or both ends moving like a player;
    and neither piece long-stationary or standing at a sideline (bench zone)."""
    if not HEIGHT_SOFT_EVIDENCE:
        return True
    if _stationary_or_bench(D, a, (a.c1, a.cv1)) or _stationary_or_bench(D, b, (b.c0, b.cv0)):
        return False
    hd = _hist_dist(a, b)
    if hd is not None and min(a.n_hist, b.n_hist) >= 3 and hd[0] <= HEIGHT_SOFT_MAX_KIT_L1:
        return True
    na, nb = _dominant_number(a), _dominant_number(b)
    if na is not None and na == nb:
        return True
    if not HEIGHT_SOFT_MOTION_EVIDENCE:  # variant 3: motion alone is not identity (bench kids in team kit move too)
        return False
    return (a.cv1 is not None and b.cv0 is not None and _norm2(a.cv1) > HEIGHT_SOFT_MIN_SPEED
            and _norm2(b.cv0) > HEIGHT_SOFT_MIN_SPEED)


def _end_unreliable(D: _Data, p: _Piece, at_start: bool) -> bool:
    """Any box in the piece's end window cut short: partial, feet hidden or overlapped by another person's box.
    (Boxes cut by the bottom image edge are dropped before tracking: filter.drop_cut_feet.)"""
    kk = D.k[p.det]
    w = max(1, int(round(D.cfg.end_window_s * D.fps)))
    sel = p.det[kk <= kk[0] + w] if at_start else p.det[kk >= kk[-1] - w]
    cut = D.partial[sel] | D.feet_hidden[sel]
    if HEIGHT_SOFT_OCCLUDED:
        cut = cut | D.occluded[sel]
    return bool(cut.any())


def _app_cost(D: _Data, a: _Piece, b: _Piece, gap_s: float = 0.0) -> float | None:
    num = _number_cost(D, a, b)
    if num is None:
        return None
    cue = _cue_cost(D, a, b, gap_s)
    if cue is None:
        return None
    if a.app is None or b.app is None:
        return num + cue
    chi = float(np.sum(np.minimum((a.app - b.app) ** 2 / (a.app_var + b.app_var), _APP_CAP)))
    if chi > D.cfg.appearance_gate_chi2:
        return None
    return D.cfg.link_appearance_weight * 0.5 * chi + num + cue


def _hist_dist(a: _Piece, b: _Piece) -> tuple[float, float] | None:
    """(L1 distance of the kit histograms, its noise sigma) or None without histograms."""
    if a.hist is None or b.hist is None:
        return None
    d = float(np.abs(a.hist - b.hist).sum())
    sig = float(np.sqrt(_HIST_SIGMA_FLOOR ** 2 + _HIST_SIGMA_ONE ** 2 * (1.0 / a.n_hist + 1.0 / b.n_hist)))
    return d, sig


def _hist_conflict(a: _Piece, b: _Piece) -> bool:
    """Clearly different kits (team / referee / bench clothes), both well measured."""
    hd = _hist_dist(a, b)
    return hd is not None and hd[0] > max(_HIST_GATE, _HIST_GATE_Z * hd[1])


def _achromatic(p: _Piece) -> bool:
    """Grey / black / white kit (referees, staff): most of the kit histogram in the achromatic bins."""
    return p.hist is not None and float(p.hist[-_N_ACH_BINS:].sum()) > EMB_ACHROMATIC_SHARE


def _team_kit(p: _Piece) -> bool:
    """Clearly a coloured team kit: histogram measured and at most EMB_TEAM_KIT_MAX_ACH of it grey/black/white."""
    return p.hist is not None and p.n_hist >= 3 and float(p.hist[-_N_ACH_BINS:].sum()) <= EMB_TEAM_KIT_MAX_ACH


def _emb_sim(a: _Piece, b: _Piece) -> float | None:
    if _EMB_INERT:
        return None
    if a.emb is None or b.emb is None or min(a.n_emb, b.n_emb) < EMB_MIN_SAMPLES:
        return None
    if EMB_TEAM_KITS_ONLY and not (_team_kit(a) and _team_kit(b)):
        return None  # referees: re-ID was trained on players; it cost referee recall (gate 13.10 / 18.10)
    return float(a.emb @ b.emb)


def _cue_cost(D: _Data, a: _Piece, b: _Piece, gap_s: float = 0.0) -> float | None:
    """Optional cues (IdentityCues + calibration): kit histogram, body height in metres, re-ID embedding.
    Each adds a cost (0 when the cue is missing on either side); the histogram and the embedding can veto."""
    cost = 0.0
    hd = _hist_dist(a, b)
    if hd is not None:
        if hd[0] > max(_HIST_GATE, _HIST_GATE_Z * hd[1]):
            return None
        if KIT_SPLIT_L1 > 0 and min(a.n_hist, b.n_hist) >= KIT_SPLIT_MIN_N and hd[0] > KIT_SPLIT_L1:
            return None  # the merged identity's own histograms would split into two kits
        cost += HIST_WEIGHT * 0.5 * min((hd[0] / hd[1]) ** 2, _HIST_CAP)
    if a.n_height and b.n_height:
        sig = float(np.sqrt(_HEIGHT_SIGMA_FLOOR ** 2 + _HEIGHT_SIGMA_ONE ** 2 * (1.0 / a.n_height + 1.0 / b.n_height)))
        cost += HEIGHT_M_WEIGHT * 0.5 * min(((a.height_m - b.height_m) / sig) ** 2, _HEIGHT_CAP)
    s = _emb_sim(a, b) if gap_s >= EMB_MIN_GAP_S else None  # short gaps: motion decides (re-ID vetoes there cut players)
    if s is not None:
        if s < EMB_GATE_SIM:
            return None
        cost += EMB_WEIGHT * (EMB_SAME_SIM - s) / EMB_SCALE
    return cost


def _strong_same(D: _Data, a: _Piece, b: _Piece) -> bool:
    """Positive identity evidence strong enough for a long gap: the same jersey number, or a confident re-ID match."""
    if D.numbers is not None and a.numbers and b.numbers:
        from replay.jersey_numbers import link_cost

        c = link_cost(a.numbers, b.numbers)
        if c is not None and c < 0:
            return True
    s = _emb_sim(a, b)
    if s is None or s < EMB_STRONG_SIM:
        return False
    hd = _hist_dist(a, b)  # re-ID alone may not join two kits: the merged identity must keep one team majority
    return hd is not None and min(a.n_hist, b.n_hist) >= EMB_STRONG_MIN_HIST and hd[0] <= EMB_STRONG_MAX_HIST_L1


def _number_cost(D: _Data, a: _Piece, b: _Piece) -> float | None:
    """Jersey numbers (optional input, jersey_numbers.link_cost): the same number on both pieces is a bonus,
    different numbers a penalty, impossible when both are well supported."""
    if D.numbers is None or not a.numbers or not b.numbers:
        return 0.0
    from replay.jersey_numbers import link_cost

    return link_cost(a.numbers, b.numbers)


def _overlap_cost(D: _Data, a: _Piece, b: _Piece) -> float | None:
    ka = {int(D.k[d]): d for d in a.det}
    pairs = [(ka[int(D.k[d])], d) for d in b.det if int(D.k[d]) in ka]
    if not pairs:
        return None
    dist = []
    for da, db in pairs:
        if D.has_img:  # duplicate boxes of one person: same head, similar height
            hh = max(1.0, 0.5 * (D.full_h[da] + D.full_h[db]))
            if abs(np.log(D.full_h[da] / D.full_h[db])) > 0.25:
                return None
            dist.append(_norm2(D.head[da] - D.head[db]) / hh / 0.2)
        elif D.has_court[da] and D.has_court[db]:
            dist.append(_norm2(D.xy[da] - D.xy[db]) / 0.7)
    if not dist or max(dist) > 1.0:
        return None
    app = _app_cost(D, a, b)
    return None if app is None else D.cfg.overlap_cost + float(np.mean(dist)) + app


def _merge(D: _Data, pieces: list[_Piece], links: list[tuple[int, int]]) -> list[_Piece]:
    if getattr(D, "lab", None) and links:  # corrections ledger: never join two sides of a split / two players
        links = _ledger_chain_cuts(D, pieces, links)
    nxt = dict(links)
    has_prev = {b for _, b in links}
    out = []
    for s in range(len(pieces)):
        if s in has_prev:
            continue
        det = [pieces[s].det]
        log = list(pieces[s].link_log)
        c = s
        while c in nxt:
            log.append(D.link_info.get((c, nxt[c]), (np.nan, 0.0, np.nan, False)))
            c = nxt[c]
            det.append(pieces[c].det)
            log.extend(pieces[c].link_log)
        if len(det) == 1:
            out.append(pieces[s])
            continue
        det = np.concatenate(det)
        # overlapping pieces: one detection per sample, the higher-confidence one
        det = det[np.lexsort((-D.conf[det], D.k[det]))]
        _, first = np.unique(D.k[det], return_index=True)
        p = D.summarise(det[first])
        p.link_log = log
        out.append(p)
    return out


# ----------------------------------------------------------------------------------------------- corrections ledger
def _labels(D: _Data, p: _Piece) -> frozenset:
    return corrections.piece_labels(D.lab, p.det)


def _ledger_candidates(D: _Data, pieces: list[_Piece], rows: list, cols: list, vals: list, max_gap: int
                       ) -> tuple[list, list, list, set]:
    """Hard constraints of the corrections ledger on one round's link candidates: a candidate joining labels that
    conflict (two sides of a split, two different players) is dropped (infinite cost); a piece's forced successor
    (the next piece in time sharing a merge / player label, within this round's gap) is linked at zero cost and
    vetoes every other candidate of that end and of that start."""
    labs = [_labels(D, p) for p in pieces]
    keep = [(a, b, v) for a, b, v in zip(rows, cols, vals) if not corrections.conflict(labs[a], labs[b])]
    lab_idx = [i for i, lab in enumerate(labs) if lab]
    succ: dict[int, int] = {}
    for a in lab_idx:
        best = None
        for b in lab_idx:
            if b != a and pieces[b].k0 > pieces[a].k1 and corrections.forced(labs[a], labs[b]) and                     (best is None or pieces[b].k0 < pieces[best].k0):
                best = b
        if best is not None and pieces[best].k0 - pieces[a].k1 <= max_gap:
            succ[a] = best
    pred: dict[int, int] = {}
    for a, b in succ.items():  # one forced predecessor per start: the latest-ending one
        if b not in pred or pieces[a].k1 > pieces[pred[b]].k1:
            pred[b] = a
    forced_links = {(a, b) for b, a in pred.items()}
    if forced_links:
        fa, fb = {a for a, _ in forced_links}, {b for _, b in forced_links}
        keep = [(a, b, v) for a, b, v in keep if a not in fa and b not in fb]
        keep += [(a, b, 0.0) for a, b in sorted(forced_links)]
    D.stats["ledger_forced_links"] = D.stats.get("ledger_forced_links", 0) + len(forced_links)
    return [a for a, _, _ in keep], [b for _, b, _ in keep], [v for _, _, v in keep], forced_links


def _ledger_chain_cuts(D: _Data, pieces: list[_Piece], links: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """A chain (A-B, B-C linked at once) must not join conflicting ledger labels through unlabelled pieces in
    between: walk each chain and cut the most expensive link between the last labelled piece and the conflicting
    one (forced links cost 0, so they are cut last)."""
    nxt = dict(links)
    has_prev = {b for _, b in links}
    drop: set[tuple[int, int]] = set()
    for s in nxt:
        if s in has_prev:
            continue
        chain = [s]
        while chain[-1] in nxt:
            chain.append(nxt[chain[-1]])
        labs = [_labels(D, pieces[c]) for c in chain]
        acc: frozenset = labs[0]
        last = 0 if labs[0] else None
        start = 0
        for j in range(1, len(chain)):
            if labs[j] and corrections.conflict(acc, labs[j]):
                i0 = last if last is not None and last >= start else start
                cand = [(chain[i], chain[i + 1]) for i in range(i0, j)]
                costs = []
                for lk in cand:
                    c_ = D.link_info.get(lk, (np.nan,))[0]
                    costs.append(np.inf if not np.isfinite(c_) else c_)
                i = int(np.argmax(costs))
                drop.add(cand[i])
                start = i0 + i + 1
                acc = frozenset().union(*labs[start:j + 1])
            else:
                acc = acc | labs[j]
            if labs[j]:
                last = j
    D.stats["ledger_cuts"] = D.stats.get("ledger_cuts", 0) + len(drop)
    return [lk for lk in links if lk not in drop]


def _ledger_enforce(D: _Data, pieces: list[_Piece]) -> tuple[list[_Piece], dict]:
    """After all linking: pieces that must be one identity (same merge / player label) but are still apart (gap
    longer than any round) are joined, in time order, when they do not overlap in time and no conflict is
    created. Returns (pieces, stats {forced_joins, unsatisfied})."""
    labs = [_labels(D, p) for p in pieces]
    idx = sorted((i for i, lab in enumerate(labs) if any(x[:2] in ("p:", "m:") for x in lab)),
                 key=lambda i: (pieces[i].k0, pieces[i].k1))
    chains: list[list[int]] = []
    chain_lab: list[frozenset] = []
    links: list[tuple[int, int]] = []
    unsatisfied = 0
    for i in idx:
        best = None
        for c, ch in enumerate(chains):
            if corrections.forced(chain_lab[c], labs[i]):
                if pieces[i].k0 > pieces[ch[-1]].k1 and not corrections.conflict(chain_lab[c], labs[i]):
                    best = c
                else:
                    unsatisfied += 1
                break
        if best is None:
            chains.append([i])
            chain_lab.append(labs[i])
        else:
            links.append((chains[best][-1], i))
            chains[best].append(i)
            chain_lab[best] = chain_lab[best] | labs[i]
    st = {"forced_joins": len(links), "unsatisfied": unsatisfied, "cuts": D.stats.get("ledger_cuts", 0),
          "forced_links": D.stats.get("ledger_forced_links", 0)}
    if not links:
        return pieces, st
    D.link_info = {(a, b): (0.0, max(0.0, (pieces[b].k0 - pieces[a].k1) / D.fps), np.nan, True) for a, b in links}
    return _merge(D, pieces, links), st


def _link_round(D: _Data, pieces: list[_Piece], max_gap_s: float, no_link_cost: float, long: bool = False
                ) -> list[tuple[int, int]]:
    """One global assignment round: every piece end may continue at most one later piece start (min-cost
    bipartite matching, each end may stay unlinked at ``no_link_cost``).

    long: a long-gap round (beyond the configured rounds) - only substantial pieces, and only with strong
    positive identity evidence (same jersey number or confident re-ID match, see _strong_same); a gap up to
    LONG_WEAK_MAX_GAP_S may instead rest on kit + height + motion when the kit histogram is well measured."""
    n = len(pieces)
    D.link_info = {}
    if n < 2:
        return []
    cfg = D.cfg
    max_gap = max(1, int(round(max_gap_s * D.fps)))
    starts = np.array([p.k0 for p in pieces])
    order = np.argsort(starts, kind="stable")
    s_sorted = starts[order]
    rows, cols, vals = [], [], []
    short = max_gap_s > cfg.link_rounds_s[0]  # after the first round, leftover crumbs are not linked over gaps
    min_dets = LONG_MIN_DETS if long else cfg.min_link_dets
    weak_gap = int(round(LONG_WEAK_MAX_GAP_S * D.fps))

    def big(p: _Piece) -> bool:
        return len(p.det) >= min_dets and (not long or p.conf_max >= LONG_MIN_CONF)

    for a, pa in enumerate(pieces):
        if short and not big(pa):
            continue
        lo = np.searchsorted(s_sorted, pa.k1 - cfg.max_overlap_samples + 1, "left")
        hi = np.searchsorted(s_sorted, pa.k1 + max_gap, "right")
        for b in order[lo:hi]:
            pb = pieces[b]
            if b == a or (short and not big(pb)):
                continue
            if long:
                if pb.k0 <= pa.k1:
                    continue
                if not _strong_same(D, pa, pb):
                    if pb.k0 - pa.k1 > weak_gap or min(pa.n_hist, pb.n_hist) < LONG_WEAK_MIN_HIST:
                        continue
            c = _link_cost(D, pa, pb, max_gap)
            if c is not None and c < no_link_cost:
                rows.append(a)
                cols.append(int(b))
                vals.append(c)
    forced_links: set[tuple[int, int]] = set()
    if getattr(D, "lab", None):  # corrections ledger: cuts at infinite cost, forced links at zero cost
        rows, cols, vals, forced_links = _ledger_candidates(D, pieces, rows, cols, vals, max_gap)
    if not rows:
        return []
    # bipartite graph: ends (rows) x [starts | one private "stay unlinked" column per end]
    off = 1.0 - min(0.0, min(vals))  # strictly positive weights (explicit zeros would vanish from the matrix)
    rows_all = np.r_[rows, np.arange(n)]
    cols_all = np.r_[cols, n + np.arange(n)]
    vals_all = np.r_[np.array(vals) + off, np.full(n, no_link_cost + off)]
    M = csr_matrix((vals_all, (rows_all, cols_all)), shape=(n, 2 * n))
    r_ind, c_ind = min_weight_full_bipartite_matching(M)
    out = [(int(a), int(b)) for a, b in zip(r_ind, c_ind) if b < n]
    cost_of = {(a, b): v for a, b, v in zip(rows, cols, vals)}
    for a, b in out:  # (cost, gap s, threshold, strong evidence) per accepted link -> identity confidence
        D.link_info[(a, b)] = (float(cost_of[(a, b)]), max(0.0, (pieces[b].k0 - pieces[a].k1) / D.fps),
                               float(no_link_cost), bool((long and _strong_same(D, pieces[a], pieces[b]))
                                                         or (a, b) in forced_links))
    return out


def _app_conflict(D: _Data, a: _Piece, b: _Piece) -> bool:
    """Two pieces whose own appearance is well measured and clearly different (different kit / role)."""
    if a.hist is not None and b.hist is not None and min(a.n_hist, b.n_hist) >= _CHAIN_MIN_HIST:
        return _hist_conflict(a, b)
    s = _emb_sim(a, b)
    if s is not None and s < EMB_GATE_SIM:
        return True
    if D.hist is not None:  # histograms exist but not enough on these pieces: the Lab medians are too noisy
        return False
    if a.app is None or b.app is None or a.n_app < _CHAIN_MIN_APP or b.n_app < _CHAIN_MIN_APP:
        return False
    chi = float(np.sum(np.minimum((a.app - b.app) ** 2 / (a.app_var + b.app_var), _APP_CAP)))
    return chi > D.cfg.appearance_gate_chi2


def _chain_appearance_cuts(D: _Data, pieces: list[_Piece], links: list[tuple[int, int]], max_gap_s: float
                           ) -> list[tuple[int, int]]:
    """Anti-chimera: one round links whole chains at once (A-B, B-C), so A and C are never compared. When B is
    a crumb (few low-score boxes, appearance barely measured) it is compatible with anything, and a chain can
    run orange player -> crumb -> referee -> crumb -> purple player. Walk each chain; where a well-measured
    piece's kit conflicts with any well-measured piece since the last cut, cut the weakest (most expensive)
    link between the two."""
    if not links:
        return links
    nxt = dict(links)
    has_prev = {b for _, b in links}
    max_gap = max(1, int(round(max_gap_s * D.fps)))
    drop: set[tuple[int, int]] = set()
    for s in nxt:
        if s in has_prev:
            continue
        chain = [s]
        while chain[-1] in nxt:
            chain.append(nxt[chain[-1]])
        start = 0
        for j in range(1, len(chain)):
            pj = pieces[chain[j]]
            if pj.app is None or pj.n_app < _CHAIN_MIN_APP:
                continue
            bad = [i for i in range(start, j) if _app_conflict(D, pieces[chain[i]], pj)]
            if not bad:
                continue
            i0 = max(bad)  # the closest conflicting piece: cut between it and this one
            cand = [(chain[i], chain[i + 1]) for i in range(i0, j)]
            costs = [_link_cost(D, pieces[a], pieces[b], max_gap) for a, b in cand]
            costs = [np.inf if c_ is None else c_ for c_ in costs]
            i = int(np.argmax(costs))
            drop.add(cand[i])
            start = i0 + i + 1
    D.stats["appearance_chain_cuts"] = D.stats.get("appearance_chain_cuts", 0) + len(drop)
    return [lk for lk in links if lk not in drop]


def _pileup_cuts(D: _Data, pieces: list[_Piece], links: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Pile-up guard (shortest round): a chain running through PILEUP_MIN_PIECES or more tiny pieces (<=
    PILEUP_MAX_DETS boxes) within PILEUP_SPAN_S, with (almost) no kit histogram among them, is a guess through a
    cluster of players - c300: orange -> purple inside 1.5 s. Drop every link touching those tiny pieces; the bigger
    pieces around them can still be linked directly in the next round, with their own kit evidence."""
    if not links:
        return links
    nxt = dict(links)
    has_prev = {b for _, b in links}
    span = PILEUP_SPAN_S * D.fps
    tiny = lambda i: len(pieces[i].det) <= PILEUP_MAX_DETS  # noqa: E731
    bad: set[int] = set()
    for s0 in nxt:
        if s0 in has_prev:
            continue
        chain = [s0]
        while chain[-1] in nxt:
            chain.append(nxt[chain[-1]])
        run: list[int] = []
        for c in chain + [None]:
            if c is not None and tiny(c):
                run.append(c)
                continue
            if len(run) >= PILEUP_MIN_PIECES and pieces[run[-1]].k1 - pieces[run[0]].k0 <= span                     and sum(pieces[i].n_hist for i in run) < PILEUP_MIN_HIST:
                bad.update(run)
            run = []
    if not bad:
        return links
    D.stats["pileup_cuts"] = D.stats.get("pileup_cuts", 0) + sum(1 for a, b in links if a in bad or b in bad)
    if PILEUP_INNER_ONLY:  # keep each tiny piece's link to a bigger neighbour, drop only tiny -> tiny guesses
        return [(a, b) for a, b in links if not (a in bad and b in bad)]
    return [(a, b) for a, b in links if a not in bad and b not in bad]


def _number_chain_cuts(D: _Data, pieces: list[_Piece], links: list[tuple[int, int]], max_gap_s: float
                       ) -> list[tuple[int, int]]:
    """One round links whole chains at once (A-B and B-C), so A and C are never compared: when B shows no
    number, a chain can join two players with different jersey numbers. Walk each chain and, where a piece's
    numbers conflict with the numbers collected since the last cut, cut the weakest (most expensive) link
    between the two."""
    from replay.jersey_numbers import link_cost

    if not links:
        return links
    nxt = dict(links)
    has_prev = {b for _, b in links}
    max_gap = max(1, int(round(max_gap_s * D.fps)))
    drop: set[tuple[int, int]] = set()
    for s in nxt:
        if s in has_prev:
            continue
        chain = [s]
        while chain[-1] in nxt:
            chain.append(nxt[chain[-1]])
        acc: dict[int, int] = {}
        last_num = None  # position in chain of the last piece with numbers since the last cut
        start = 0
        for j, c in enumerate(chain):
            nums = pieces[c].numbers or {}
            if nums and acc and (lc := link_cost(acc, nums)) is not None and lc <= 0:
                pass
            elif nums and acc:  # conflict: cut the weakest link between the last numbered piece and this one
                cand = [(chain[i], chain[i + 1]) for i in range(max(last_num, start), j)]
                costs = [_link_cost(D, pieces[a], pieces[b], max_gap) for a, b in cand]
                costs = [np.inf if c_ is None else c_ for c_ in costs]
                i = int(np.argmax(costs))
                drop.add(cand[i])
                start = max(last_num, start) + i + 1
                acc = {}
                for q in chain[start:j]:
                    for v, m in (pieces[q].numbers or {}).items():
                        acc[v] = acc.get(v, 0) + m
            for v, m in nums.items():
                acc[v] = acc.get(v, 0) + m
            if nums:
                last_num = j
    D.stats["number_cuts"] = D.stats.get("number_cuts", 0) + len(drop)
    return [lk for lk in links if lk not in drop]


def _absorb_crumbs(D: _Data, pieces: list[_Piece]) -> tuple[list[_Piece], int]:
    """Give the few leftover detections (low-score boxes during an occlusion) to the identity whose gap they fill.

    A crumb detection joins an identity that has no detection in that sample, has detections on both sides
    within ``absorb_window_s``, and whose interpolated position (court and image) it matches.
    """
    cfg, fps = D.cfg, D.fps
    big = [p for p in pieces if len(p.det) > cfg.absorb_max_dets]
    small = [p for p in pieces if len(p.det) <= cfg.absorb_max_dets]
    if not big or not small:
        return pieces, 0
    w = max(1, int(round(cfg.absorb_window_s * fps)))
    kk_big = [D.k[p.det] for p in big]
    k0 = np.array([p.k0 for p in big])
    k1 = np.array([p.k1 for p in big])
    add: dict[int, list[int]] = {}
    keep_small = []
    absorbed = 0
    for c in small:
        target = None
        for d in c.det:
            kd = int(D.k[d])
            best = None
            for bi in np.flatnonzero((k0 < kd) & (k1 > kd)):
                kb = kk_big[bi]
                j = int(np.searchsorted(kb, kd))
                if kb[j] == kd or kd - kb[j - 1] > w or kb[j] - kd > w:
                    continue
                da, db = big[bi].det[j - 1], big[bi].det[j]
                f = (kd - kb[j - 1]) / (kb[j] - kb[j - 1])
                cost, ok = 0.0, True
                if D.has_court[d] and D.has_court[da] and D.has_court[db]:
                    e = _norm2(D.xy[d] - ((1 - f) * D.xy[da] + f * D.xy[db]))
                    ok &= e < cfg.absorb_court_m + 1.5 * np.sqrt(D.rvar[d])
                    cost += e
                if D.has_img:
                    qa, qb = D.head_in(da, kd), D.head_in(db, kd)
                    if qa is not None and qb is not None:
                        q = (1 - f) * qa + f * qb
                        e = _norm2(D.head[d] - q[:2]) / max(1.0, q[2])
                        ok &= e < cfg.absorb_image_bh and abs(np.log(max(D.full_h[d], 1.0) / max(q[2], 1.0))) < 0.35
                        cost += e
                if ok and (best is None or cost < best[0]):
                    best = (cost, bi)
            if best is None or (target is not None and best[1] != target):
                target = None
                break
            target = best[1]
        if target is not None and getattr(D, "lab", None) and \
                corrections.conflict(_labels(D, c), _labels(D, big[target])):
            target = None
        if target is None:
            keep_small.append(c)
        else:
            add.setdefault(target, []).extend(int(d) for d in c.det)
            absorbed += len(c.det)
    out = list(keep_small)
    for bi, p in enumerate(big):
        if bi in add:
            det = np.concatenate([p.det, np.array(add[bi], int)])
            det = det[np.lexsort((-D.conf[det], D.k[det]))]
            _, first = np.unique(D.k[det], return_index=True)
            out.append(D.summarise(det[first]))
        else:
            out.append(p)
    return out, absorbed


def _attach_end_crumbs(D: _Data, pieces: list[_Piece]) -> tuple[list[_Piece], int]:
    """Short re-appearances (<= absorb_max_dets detections) just before an identity starts or just after it ends
    (within END_CRUMB_WINDOW_S) join that identity when the link is feasible (tracklets._link_cost) and unambiguous
    (the best candidate beats the second by END_CRUMB_MARGIN). Otherwise they became their own short output ids."""
    cfg = D.cfg
    lim = cfg.absorb_max_dets
    big = [i for i, p in enumerate(pieces) if len(p.det) > lim]
    small = [i for i, p in enumerate(pieces) if len(p.det) <= lim]
    if not big or not small:
        return pieces, 0
    w = max(1, int(round(END_CRUMB_WINDOW_S * D.fps)))
    k0 = np.array([pieces[i].k0 for i in big])
    k1 = np.array([pieces[i].k1 for i in big])
    links: list[tuple[int, int]] = []
    used_end, used_start = set(), set()
    for c in small:
        pc = pieces[c]
        cands = []
        for j in np.flatnonzero((k1 < pc.k0) & (k1 >= pc.k0 - w)):  # crumb after an identity ends
            bi = big[j]
            cost = _link_cost(D, pieces[bi], pc, w)
            if cost is not None:
                cands.append((cost, bi, c))
        for j in np.flatnonzero((k0 > pc.k1) & (k0 <= pc.k1 + w)):  # crumb before an identity starts
            bi = big[j]
            cost = _link_cost(D, pc, pieces[bi], w)
            if cost is not None:
                cands.append((cost, c, bi))
        if not cands:
            continue
        cands.sort()
        if len(cands) > 1 and cands[1][0] - cands[0][0] < END_CRUMB_MARGIN:
            continue
        _, a, b = cands[0]
        if cands[0][0] > END_CRUMB_MAX_COST or a in used_end or b in used_start:
            continue
        links.append((a, b))
        used_end.add(a)
        used_start.add(b)
    if not links:
        return pieces, 0
    D.link_info = {}
    return _merge(D, pieces, links), len(links)


def _dominant_number(p: _Piece) -> int | None:
    """The piece's two-digit jersey number when it is read at least twice and clearly dominates (else None)."""
    if not p.numbers:
        return None
    two = {v: m for v, m in p.numbers.items() if v >= 10}
    if not two:
        return None
    v, m = max(two.items(), key=lambda it: it[1])
    if m >= ANCHOR_MIN_READS and m >= ANCHOR_MAJORITY * sum(two.values()):
        return int(v)
    return None


def _anchor_compatible(D: _Data, a: _Piece, b: _Piece) -> bool:
    """b (later) may continue a: no time overlap, same kit, plausible height, reachable on the court."""
    if b.k0 <= a.k1:
        return False
    if _hist_conflict(a, b) or _app_conflict(D, a, b):
        return False
    s = _emb_sim(a, b)
    if s is not None and s < EMB_GATE_SIM:
        return False
    if a.n_height >= 10 and b.n_height >= 10 and abs(a.height_m - b.height_m) > ANCHOR_MAX_HEIGHT_DIFF_M:
        return False
    if D.numbers is not None and a.numbers and b.numbers:
        from replay.jersey_numbers import link_cost

        if link_cost(a.numbers, b.numbers) is None:
            return False
    if a.c1 is not None and b.c0 is not None:
        g = (b.k0 - a.k1) / D.fps
        d = _norm2(b.c0 - a.c1)
        if d > D.cfg.max_speed_mps * g + D.cfg.reach_slack_m + 2.0 * np.sqrt(a.cvar1 + b.cvar0):
            return False
    return True


def _number_anchor_merge(D: _Data, pieces: list[_Piece]) -> tuple[list[_Piece], int]:
    """Hard anchors: identities wearing the same well-read two-digit number in the same kit are one person.
    Pieces of one number are taken in time order; each joins the latest-ending compatible chain (no time
    overlap, compatible kit/height/numbers, reachable) or starts a new chain (e.g. the other team's #12)."""
    groups: dict[int, list[int]] = {}
    for i, p in enumerate(pieces):
        v = _dominant_number(p)
        if v is not None and len(p.det) >= LONG_MIN_DETS:
            groups.setdefault(v, []).append(i)
    links: list[tuple[int, int]] = []
    for v, idx in groups.items():
        if len(idx) < 2:
            continue
        idx.sort(key=lambda i: pieces[i].k0)
        chains: list[list[int]] = []
        for i in idx:
            best = None
            for ch in chains:
                last = pieces[ch[-1]]
                if _anchor_compatible(D, last, pieces[i]) and (best is None or last.k1 > pieces[best[-1]].k1):
                    best = ch
            if best is None:
                chains.append([i])
            else:
                links.append((best[-1], i))
                best.append(i)
    if not links:
        return pieces, 0
    D.link_info = {(a, b): (np.nan, max(0.0, (pieces[b].k0 - pieces[a].k1) / D.fps), np.nan, True)
                   for a, b in links}
    return _merge(D, pieces, links), len(links)


def _identity_confidence(D: _Data, p: _Piece) -> float:
    """0..1: how sure we are that this identity is one person throughout.

    Each link contributes a reliability from its cost margin under its round's threshold, discounted for long
    gaps unless strong evidence (number / re-ID) backed it; the identity's confidence mixes the weakest link with
    the average one. A piece without links is as reliable as ByteTrack + the cut checks (0.9); very short
    identities get less (often crumbs / false detections)."""
    rel = []
    for cost, gap, thr, strong in p.link_log:
        if strong:
            r = 0.9
        elif np.isfinite(cost) and np.isfinite(thr) and thr > 0:
            r = float(np.clip(0.55 + 0.45 * (thr - cost) / thr, 0.3, 0.97))
            r *= float(np.exp(-gap / 20.0))
        else:
            r = 0.6
        rel.append(r)
    base = 0.9 if not rel else 0.5 * min(rel) + 0.5 * float(np.mean(rel))
    dur = (p.k1 - p.k0 + 1) / D.fps
    base *= float(np.clip(0.4 + 0.6 * dur / 3.0, 0.4, 1.0))
    if _dominant_number(p) is not None:
        base = max(base, 0.85)
    if p.numbers and sum(1 for v, m in p.numbers.items() if v >= 10 and m >= 2) >= 2:
        base = min(base, 0.4)  # two different well-read numbers: probably two people
    return round(float(np.clip(base, 0.0, 1.0)), 3)


@dataclass
class LinkResult:
    ids: np.ndarray  # identity per detection (>= 1), 0 = left out
    xy: np.ndarray  # court positions, partial boxes' feet re-estimated
    R: np.ndarray
    partial: np.ndarray  # bool per detection: box cut short, foot re-estimated
    stats: dict
    # per identity id: {"confidence" 0..1, "links", "strong_links", "max_gap_s", "number" (two-digit code or None),
    # "height_m", "t0_sample", "t1_sample", "detections"} - see _identity_confidence
    identities: dict = field(default_factory=dict)
    labels: dict = field(default_factory=dict)  # corrections ledger labels per detection (empty without a ledger)


def link_tracks(k: np.ndarray, xy: np.ndarray, R: np.ndarray, feats: np.ndarray, bt_ids: np.ndarray,
                conf: np.ndarray, fps: float, cfg: CourtTrackConfig, boxes: np.ndarray | None = None,
                motion: list | None = None, n_samples: int | None = None, H: list | None = None,
                numbers: np.ndarray | None = None, cues: IdentityCues | None = None, view=None,
                ledger: "corrections.IdentityConstraints | None" = None) -> LinkResult:
    """k: sample index per detection; xy: court metres (NaN rows = no court position, image cues only);
    R: (n,2,2) court covariance; feats: (n,6) jersey Lab; bt_ids: ByteTrack ID (-1 = none); conf: score;
    boxes: (n,4) image boxes + motion: per-sample camera motion -> image-space association;
    H: per-sample image->court homography (None entries allowed) -> foot re-estimation for partial boxes;
    numbers: optional jersey number read per detection (jersey_numbers.py codes, -1 = none) -> link cue and
    hard anchors (same two-digit number + kit = same person; different well-read numbers = never);
    cues: optional IdentityCues (kit histograms, re-ID embeddings) - each used only where present.
    ledger: optional corrections.IdentityConstraints (the coach's ledger, resolved to these detections):
    hard constraints - split cuts are never re-linked, forced links (merge / same assigned player) win at zero
    cost, different assigned players are never one identity. None / empty: no effect.
    Returns LinkResult: ids per detection (stable API: >= 1, 0 = left out) + per-identity info/confidence.
    """
    if os.environ.get("AA_IDENTITY_LEGACY"):  # A/B switch for the benchmark: the linker before the identity round
        cues = None
        cfg = replace(cfg, long_rounds_s=(), chain_appearance_check=False, number_anchors=False, height_cue=False)             if _cfg_has(cfg, "long_rounds_s") else _LegacyCfg(cfg)
    k = np.asarray(k, int)
    n = len(k)
    xy = np.asarray(xy, float).reshape(n, 2)
    R = np.asarray(R, float).reshape(n, 2, 2)
    if n == 0:
        return LinkResult(np.zeros(0, int), xy, R, np.zeros(0, bool), {})
    D = _Data(k, xy, R, np.asarray(feats, float).reshape(n, -1), np.asarray(conf, float), fps, cfg,
              None if boxes is None else np.asarray(boxes, float).reshape(n, 4), motion, n_samples, H,
              numbers=numbers, cues=cues)
    raw = build_tracklets(D, np.asarray(bt_ids, int))
    D.lab = {}
    if ledger:
        raw, D.lab = corrections.apply_identity(ledger, raw, D.k, D.xy)
    pieces = [D.summarise(p) for p in raw]
    stats = dict(D.stats, tracklets=len(pieces), partial_boxes=int(D.partial.sum()), links=[])
    if D.numbers is not None:
        stats["tracklets_with_number"] = sum(bool(p.numbers) for p in pieces)
    rounds = [(g, t, False) for g, t in zip(cfg.link_rounds_s, cfg.link_thresholds)]
    joint = joint_enabled(cfg) and not os.environ.get("AA_IDENTITY_LEGACY")
    if not joint or JOINT_AFTER_ROUNDS:
        rounds += [(g, t, True) for g, t in zip(getattr(cfg, "long_rounds_s", LONG_ROUNDS_S),
                                                getattr(cfg, "long_thresholds", LONG_THRESHOLDS))]
    stats["long_links"] = []
    for max_gap_s, thr, long in rounds:
        links = _link_round(D, pieces, max_gap_s, thr, long=long)
        info = D.link_info
        if PILEUP_GUARD and max_gap_s == cfg.link_rounds_s[0]:
            links = _pileup_cuts(D, pieces, links)
        if getattr(cfg, "chain_appearance_check", CHAIN_APPEARANCE_CHECK):
            links = _chain_appearance_cuts(D, pieces, links, max_gap_s)
        if D.numbers is not None:
            links = _number_chain_cuts(D, pieces, links, max_gap_s)
            stats["number_cuts"] = D.stats.get("number_cuts", 0)
        D.link_info = info
        (stats["long_links"] if long else stats["links"]).append(len(links))
        if links:
            pieces = _merge(D, pieces, links)
    if joint:  # joint long-gap association of all people (joint_tracker.py)
        from replay import joint_tracker

        D.link_info = {}
        links, js = joint_tracker.joint_link(D, pieces, view)
        info = D.link_info
        if getattr(cfg, "chain_appearance_check", CHAIN_APPEARANCE_CHECK):
            links = _chain_appearance_cuts(D, pieces, links, joint_tracker.MAX_GAP_S)
        if D.numbers is not None:
            links = _number_chain_cuts(D, pieces, links, joint_tracker.MAX_GAP_S)
        D.link_info = info
        stats.update(js, joint_kept=len(links))
        if links:
            pieces = _merge(D, pieces, links)
    if D.numbers is not None and getattr(cfg, "number_anchors", NUMBER_ANCHORS):
        pieces, stats["anchor_merges"] = _number_anchor_merge(D, pieces)
    if cfg.absorb_max_dets > 0:
        pieces, stats["absorbed"] = _absorb_crumbs(D, pieces)
        if END_CRUMBS:
            pieces, stats["end_crumbs"] = _attach_end_crumbs(D, pieces)
    if D.lab:
        pieces, stats["corrections"] = _ledger_enforce(D, pieces)
        stats["corrections"]["unresolved_ops"] = len(ledger.unresolved)
    pieces.sort(key=lambda p: (p.k0, p.k1))
    ids = np.zeros(n, int)
    identities: dict[int, dict] = {}
    next_id = 1
    for p in pieces:
        if float(D.conf[p.det].max()) < cfg.new_track_min_conf:
            continue
        ids[p.det] = next_id
        identities[next_id] = {
            "confidence": _identity_confidence(D, p), "links": len(p.link_log),
            "strong_links": sum(1 for lk in p.link_log if lk[3]),
            "max_gap_s": round(max([lk[1] for lk in p.link_log], default=0.0), 2),
            "number": _dominant_number(p), "height_m": None if not p.n_height else round(p.height_m, 2),
            "t0_sample": p.k0, "t1_sample": p.k1, "detections": len(p.det)}
        next_id += 1
    stats["identities"] = next_id - 1
    return LinkResult(ids, D.xy, D.R, D.partial, stats, identities, D.lab)


# ----------------------------------------------------------------------------------------------- pass-2 glue
def cues_for(work, data, video, cfg, det_row: np.ndarray, meta=None) -> IdentityCues | None:
    """IdentityCues for the pass-2 detections (pass-1 row indices ``det_row``), from the caches other modules
    keep in the work folder: kit histograms (team_appearance, via teams.appearance_for - computed once and reused
    by step 5) and re-ID embeddings (USE_REID): a cache matching the current model + pass-1 rows, computed from the
    video only when reid.ReidConfig.enabled.
    Never raises: a failing cue is simply left out."""
    det_row = np.asarray(det_row, int)
    hist = occ = emb = None
    try:
        from replay.teams import appearance_for

        app = appearance_for(work, data, video, cfg.teams, det_row, meta, clip=False)
        if app is not None:
            hist, occ = np.asarray(app.desc, float), np.asarray(app.occ, float)
            from replay import team_appearance as ta

            if ta.DENSE_CROWDS:  # dense crowd histograms are for teams/roster only: the linker keeps the grid rows
                grid = set(ta.sample_frames(data.t, cfg.teams.resample_every_s).tolist())
                idx_of_k = {int(k): i for i, k in enumerate(data.k)}
                on = np.array([idx_of_k.get(int(k), -1) in grid for k in data.rows[det_row, 0]], bool)
                hist = hist.copy()
                hist[~on] = np.nan
    except Exception as e:  # noqa: BLE001
        print("identity cues: kit histograms unavailable:", e)
    try:
        from replay import reid

        rc = getattr(cfg, "reid", None) or reid.ReidConfig()
        if reid_enabled(cfg):  # field CourtTrackConfig.use_reid, env AA_IDENTITY_REID overrides; computed only when ReidConfig.enabled, otherwise a matching cache is used
            got = reid.load_or_compute(work, data, video if rc.enabled else None, rc)
            if got is not None and len(got.emb) == len(data.rows):
                emb = got.get(det_row)
    except Exception as e:  # noqa: BLE001
        print("identity cues: re-ID embeddings unavailable:", e)
    if hist is None and emb is None:
        return None
    return IdentityCues(hist=hist, hist_occ=occ, embeddings=emb)


def detection_confidence(ids: np.ndarray, identities: dict) -> np.ndarray:
    """Per-detection identity confidence (the confidence of the identity it belongs to; NaN for id 0).
    Survives later renumbering (re-attach, lineup slots): aggregate it per final track id."""
    out = np.full(len(ids), np.nan)
    for i, info in identities.items():
        out[ids == i] = info.get("confidence", np.nan)
    return out


def add_identity_confidence(track_info: dict, id_info: dict, tid: np.ndarray, keep: np.ndarray, new_id: dict) -> None:
    """track_info[out id]["identity_confidence"]: median per-detection identity confidence of the final track
    (after re-attach / team split / lineup renumbering, which keep detections aligned)."""
    dc = id_info.get("det_confidence") if id_info else None
    if dc is None or len(dc) != len(tid):
        return
    for t, o in new_id.items():
        if o in track_info:
            v = dc[keep & (tid == t)]
            v = v[np.isfinite(v)]
            if len(v):
                track_info[o]["identity_confidence"] = round(float(np.median(v)), 3)


def _env_overrides() -> None:
    """A/B knobs for the benchmark (subprocess workers inherit the env): AA_TL_SET="NAME=value;NAME2=value"."""
    for kv in os.environ.get("AA_TL_SET", "").split(";"):
        if "=" in kv:
            name, val = (x.strip() for x in kv.split("=", 1))
            if name.isupper() and name in globals():
                globals()[name] = type(globals()[name])(float(val) if isinstance(globals()[name], float) else
                                                         (val == "1" if isinstance(globals()[name], bool) else val))


_env_overrides()
