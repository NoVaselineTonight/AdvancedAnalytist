"""Offline ball trajectory linking in camera-stabilised image coordinates.

Input: every ball candidate of every analysed frame (pass 1, class "ball"), and the camera-motion chain
``G[i]`` (frame-i pixels -> pixels of the first frame of the unbroken motion segment). Positions are compared
*pairwise*: a candidate of frame j is moved into frame i's pixels with ``inv(G[i]) @ G[j]``, so a static
object in the gym keeps its place while the camera pans. (Mapping everything into the segment's first frame
does not work: after a pan of more than ~half a field of view those coordinates run off to infinity.)

1. **Static candidates** (spare balls by the bench, a ball rack, a poster, the orange rim) are removed: a
   candidate that has a near-identical stabilised position in most frames of +-``static_window_s`` is not the
   game ball.
2. **One best path through time** (dynamic programming over all candidates, like a Viterbi decoder with skips).
   A path visits at most one candidate per frame. Each candidate earns ``conf - conf_offset``; a link between
   two candidates ``g`` frames apart costs a small per-skipped-frame penalty plus a motion term, and is only
   allowed if the stabilised displacement is physically possible: at most ``max_speed_mps`` (converted to
   pixels with the ball's own image size, a 0.24 m ruler that scales with zoom and depth) plus detector
   jitter. A sudden jump to an unrelated candidate is possible but costs ``jump_penalty`` (more than a single
   confident detection earns), so isolated false positives are skipped while a ball that really reappears
   elsewhere after a long gap (``> link_s``) costs only ``restart_penalty``.
3. **Gap filling in the image**: between two consecutive path nodes of the same tracklet (no jump, gap
   <= ``link_s``) the stabilised position is interpolated (quadratic through the neighbouring nodes when they
   exist, i.e. along a parabolic arc; otherwise linear). These frames are flagged ``observed = False``.

Everything is in image space, so it works without (or with an imperfect) court calibration.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from replay.config import BallConfig

BALL_DIAMETER_M = 0.24


@dataclass
class BallTrack:
    boxes: list[tuple[float, float, float, float, float] | None]  # chosen detection per frame (x1,y1,x2,y2,conf)
    img: np.ndarray  # (n, 2) ball centre in frame pixels (observed or interpolated), nan = unknown
    size: np.ndarray  # (n,) ball image size in pixels (nan = unknown)
    observed: np.ndarray  # (n,) bool, a detection was chosen in this frame
    tracklet: np.ndarray  # (n,) int, tracklet id (-1 = none); a new id after every jump / restart


def _apply_many(M: np.ndarray, pts: np.ndarray) -> np.ndarray:
    if not len(pts):
        return np.zeros((0, 2))
    q = np.c_[pts, np.ones(len(pts))] @ M.T
    return q[:, :2] / q[:, 2:3]


def _to_frame(Ginv_i: np.ndarray, G: list[np.ndarray], frames: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Points pts[k] of frame frames[k] -> pixels of the frame whose inverse chain transform is Ginv_i."""
    if not len(pts):
        return np.zeros((0, 2))
    M = Ginv_i[None] @ np.stack([G[int(f)] for f in frames])
    q = M @ np.c_[pts, np.ones(len(pts))][:, :, None]
    return q[:, :2, 0] / q[:, 2:3, 0]


def _static_mask(fi: np.ndarray, seg: np.ndarray, xy: np.ndarray, G: list[np.ndarray], Ginv: list[np.ndarray],
                 fps: float, cfg: BallConfig) -> np.ndarray:
    """True for candidates that stay at the same place in the gym for most of +-static_window_s."""
    win = max(1, int(round(cfg.static_window_s * fps)))
    static = np.zeros(len(fi), bool)
    for c in range(len(fi)):
        i = fi[c]
        lo, hi = np.searchsorted(fi, i - win, "left"), np.searchsorted(fi, i + win, "right")
        j = np.arange(lo, hi)
        j = j[(fi[j] != i) & (seg[fi[j]] == seg[i])]
        frames_in_seg = int(np.sum(seg[max(0, i - win):i + win + 1] == seg[i])) - 1
        if frames_in_seg < win or len(j) < cfg.static_frac * frames_in_seg:
            continue  # too little context (segment start/end) or too few candidates around: not static
        pj = _to_frame(Ginv[i], G, fi[j], xy[j])
        close = np.linalg.norm(pj - xy[c], axis=1) < cfg.static_px
        static[c] = len(np.unique(fi[j][close])) >= cfg.static_frac * frames_in_seg
    return static


def link_ball(n: int, ball_rows: list[np.ndarray], G: list[np.ndarray], seg: np.ndarray, fps: float,
              cfg: BallConfig) -> BallTrack:
    """ball_rows[i]: (m, 5) [x1, y1, x2, y2, conf] candidates of analysed frame i."""
    seg = np.asarray(seg)
    fi_l, xy_l, sz_l, cf_l, row_l = [], [], [], [], []
    for i in range(n):
        r = ball_rows[i]
        if not len(r):
            continue
        c = np.stack([(r[:, 0] + r[:, 2]) / 2, (r[:, 1] + r[:, 3]) / 2], 1)
        fi_l.append(np.full(len(r), i))
        xy_l.append(c)
        sz_l.append(np.sqrt(np.maximum(1.0, (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1]))))
        cf_l.append(r[:, 4])
        row_l.append(np.arange(len(r)))
    empty = BallTrack([None] * n, np.full((n, 2), np.nan), np.full(n, np.nan), np.zeros(n, bool), np.full(n, -1))
    if not fi_l:
        return empty
    fi, xy, sz, cf, ri = (np.concatenate(a) for a in (fi_l, xy_l, sz_l, cf_l, row_l))
    Ginv = [np.linalg.inv(g) for g in G]
    keep = cf >= cfg.min_conf
    fi, xy, sz, cf, ri = fi[keep], xy[keep], sz[keep], cf[keep], ri[keep]
    keep = ~_static_mask(fi, seg, xy, G, Ginv, fps, cfg)
    fi, xy, sz, cf, ri = fi[keep], xy[keep], sz[keep], cf[keep], ri[keep]
    m = len(fi)
    if not m:
        return empty

    # ---- dynamic programming: best single path (one candidate per frame at most)
    link = max(1, int(round(cfg.link_s * fps)))
    reward = cf - cfg.conf_offset
    S = np.full(m, -np.inf)
    back = np.full(m, -1)
    jump = np.zeros(m, bool)  # this node starts a new tracklet
    best_upto = np.full(m, -np.inf)  # prefix max of S over nodes 0..c
    best_arg = np.full(m, -1)
    px_per_m = np.maximum(sz, 4.0) / BALL_DIAMETER_M
    for c in range(m):
        i = fi[c]
        # restart after a long gap (or at the start): cheap
        lo = int(np.searchsorted(fi, i - link, "left"))
        cand_s, cand_b, cand_j = -cfg.restart_penalty, -1, True
        if lo > 0 and best_upto[lo - 1] - cfg.restart_penalty > cand_s:
            cand_s, cand_b = best_upto[lo - 1] - cfg.restart_penalty, int(best_arg[lo - 1])
        hi = int(np.searchsorted(fi, i, "left"))  # nodes of earlier frames within the link window
        if hi > lo:
            u = np.arange(lo, hi)
            g = (i - fi[u]).astype(float)
            d = np.linalg.norm(_to_frame(Ginv[i], G, fi[u], xy[u]) - xy[c], axis=1)
            scale = np.sqrt(px_per_m[u] * px_per_m[c])
            allowed = scale * cfg.max_speed_mps * g / fps + cfg.jitter_diam * scale * BALL_DIAMETER_M + 4.0
            ratio = np.maximum(sz[u], sz[c]) / np.maximum(1.0, np.minimum(sz[u], sz[c]))
            ok = (d <= allowed) & (ratio <= cfg.max_size_ratio) & (seg[fi[u]] == seg[i])
            cost = cfg.gap_penalty * (g - 1) + cfg.motion_weight * (d / allowed) ** 2 + 0.3 * np.log(ratio)
            val = np.where(ok, S[u] - cost, -np.inf)
            k = int(np.argmax(val))
            if val[k] > cand_s:
                cand_s, cand_b, cand_j = float(val[k]), int(u[k]), False
            # teleport within the window: allowed but expensive
            k2 = int(np.argmax(S[u]))
            if S[u][k2] - cfg.jump_penalty > cand_s:
                cand_s, cand_b, cand_j = float(S[u][k2] - cfg.jump_penalty), int(u[k2]), True
        S[c], back[c], jump[c] = reward[c] + cand_s, cand_b, cand_j
        if c > 0 and best_upto[c - 1] >= S[c]:
            best_upto[c], best_arg[c] = best_upto[c - 1], best_arg[c - 1]
        else:
            best_upto[c], best_arg[c] = S[c], c
    path = []
    c = int(best_arg[m - 1])
    while c >= 0:
        path.append(c)
        c = int(back[c])
    path.reverse()

    # ---- per-frame output
    out = BallTrack([None] * n, np.full((n, 2), np.nan), np.full(n, np.nan), np.zeros(n, bool), np.full(n, -1))
    tid = -1
    for c in path:
        if jump[c]:
            tid += 1
        i = int(fi[c])
        out.boxes[i] = tuple(float(v) for v in ball_rows[i][ri[c], :5])  # type: ignore[assignment]
        out.img[i] = xy[c]
        out.size[i] = sz[c]
        out.observed[i] = True
        out.tracklet[i] = max(tid, 0)
    _fill_gaps(out, path, fi, jump, link, G, Ginv)
    return out


def _fill_gaps(out: BallTrack, path: list[int], fi: np.ndarray, jump: np.ndarray, link: int,
               G: list[np.ndarray], Ginv: list[np.ndarray]) -> None:
    """Interpolate the ball between consecutive nodes of one tracklet (along a parabola when possible).

    Done in the pixels of the gap's first frame (camera motion removed), then mapped into each frame."""
    frames = [int(fi[c]) for c in path]
    for a in range(len(path) - 1):
        ia, ib = frames[a], frames[a + 1]
        if jump[path[a + 1]] or ib - ia <= 1 or ib - ia > link:
            continue
        # neighbours on both sides in the same tracklet for a quadratic (ballistic arc) fit
        sel = [a, a + 1]
        if a > 0 and not jump[path[a]] and ia - frames[a - 1] <= link:
            sel.insert(0, a - 1)
        if a + 2 < len(path) and not jump[path[a + 2]] and frames[a + 2] - ib <= link:
            sel.append(a + 2)
        sf = np.array([frames[s] for s in sel])
        ts = sf.astype(float)
        P = _to_frame(Ginv[ia], G, sf, np.array([out.img[f] for f in sf]))
        pa, pb = P[sel.index(a)], P[sel.index(a + 1)]
        q = np.arange(ia + 1, ib, dtype=float)
        lin = pa + (pb - pa) * ((q - ia) / (ib - ia))[:, None]
        est = lin
        if len(sel) >= 3:
            co = [np.polyfit(ts, P[:, d], 2) for d in range(2)]
            arc = np.stack([np.polyval(co[d], q) for d in range(2)], 1)
            # sanity: the arc must stay near the straight line (a bad fit would swing far out)
            span = np.linalg.norm(pb - pa) + 2 * float(np.nanmean(out.size[sf]))
            if np.max(np.linalg.norm(arc - lin, axis=1)) <= 0.5 * span:
                est = arc
        for j, i in enumerate(range(ia + 1, ib)):
            out.img[i] = _to_frame(Ginv[i], G, np.array([ia]), est[j:j + 1])[0]
            out.size[i] = out.size[ia] + (out.size[ib] - out.size[ia]) * (i - ia) / (ib - ia)
            out.tracklet[i] = out.tracklet[ia]
