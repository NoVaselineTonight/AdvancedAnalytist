"""Joint association of all on-court people (J1, TRACKING_PLAN.md) - the long-gap stage of the identity linker.

tracklets.py first builds reliable pieces and links them over short gaps (<= 4 s, motion is informative there).
Long gaps (a player leaves the picture, stays hidden in a cluster) used to need strong evidence (same jersey number /
re-ID) because kit + motion alone cannot tell team-mates apart. This stage links them JOINTLY, using constraints
that only make sense for all people together:

* exclusivity - every piece end continues at most one later piece start and vice versa (one detection, one
  person; pieces never overlap inside one identity);
* cardinality - at most ``CAP`` on-court identities of one kit group exist at any moment (5 per team, referees
  fewer): a link keeps the person alive through the gap, so it is refused when that would exceed the cap;
* camera-view exclusion + occlusion - while unseen, the person must have been somewhere hidden: outside the image
  footprint of the court (calibration homography) or behind another person's box. For each gap sample the set of
  places reachable from both ends (running speed) is probed; if every reachable place was in plain view for a
  sustained part of the gap, the person would have been detected -> the link is impossible;
* costs - motion (tracklets._link_cost: court + camera-stabilised image), kit histogram, height, jersey numbers,
  re-ID when cached (tracklets cues), plus a cost per in-view sample.

Solved as a greedy min-cost assignment over all candidate (end, start) pairs of the whole clip in cost order
(windows of up to ``MAX_GAP_S``); the cardinality bookkeeping is global, so each accepted link constrains all later
ones. Chains are re-checked for jersey-number / kit conflicts afterwards (tracklets chain checks).

Entry: ``joint_link(D, pieces, view)`` from tracklets.link_tracks when tracklets.JOINT_TRACKER is on
(env REPLAY_JOINT=1); ``view`` is a court_tracker.OcclusionContext. Off -> the old long-gap rounds.
"""

from __future__ import annotations

import numpy as np

from replay import court

MIN_GAP_S = 4.0  # shorter gaps are the short rounds' business
MAX_GAP_S = 60.0
MIN_DETS = 10  # pieces taking part (crumbs / false detections stay out)
MIN_CONF = 0.5
MIN_HIST = 3  # kit histograms needed to put a piece in a kit group
N_GROUPS = 4  # k-means kit groups (two teams, referees, other clothes)
CAP = 5  # on-court identities per kit group at once
ON_COURT_MARGIN_M = 0.5
TEAM_KITS_ONLY = True  # grey/black kits (referees) are not joint-linked: it cost referee recall
ON_COURT_ONLY = True  # only on-court pieces are linked (long bench / sideline chains became shown phantoms)
SPEED_MPS = 7.0  # reach while unseen
SLACK_M = 1.5
PROBE_DT_S = 0.5  # gap sampled this often for the view check
VIEW_ALLOW = 0.2  # share of probed samples that may be "in plain view" (missed detections happen)
VIEW_MIN_ALLOW = 2
VIEW_COST = 1.0  # per in-view probe sample
MAX_COST = 28.0  # link cost above which a long gap stays unlinked


def _kmeans(X: np.ndarray, w: np.ndarray, k: int, iters: int = 30, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(X)
    k = min(k, n)
    C = [X[int(np.argmax(w))]]
    for _ in range(1, k):  # k-means++ (deterministic seed)
        d = np.min([np.abs(X - c).sum(axis=1) for c in C], axis=0) * w
        C.append(X[int(rng.choice(n, p=d / d.sum()))] if d.sum() > 0 else X[int(rng.integers(n))])
    C = np.array(C)
    lab = np.zeros(n, int)
    for _ in range(iters):
        D = np.abs(X[:, None, :] - C[None]).sum(axis=2)
        new = D.argmin(axis=1)
        if np.array_equal(new, lab) and _ > 0:
            break
        lab = new
        for j in range(k):
            m = lab == j
            if m.any():
                C[j] = (X[m] * w[m, None]).sum(0) / w[m].sum()
    return lab


def kit_groups(pieces: list) -> np.ndarray:
    """Kit group per piece (-1 = no histogram / too short)."""
    g = np.full(len(pieces), -1)
    idx = [i for i, p in enumerate(pieces) if p.hist is not None and p.n_hist >= MIN_HIST]
    if len(idx) < N_GROUPS:
        return g
    X = np.array([pieces[i].hist for i in idx])
    w = np.array([min(pieces[i].n_hist, 30) for i in idx], float)
    g[idx] = _kmeans(X, w, N_GROUPS)
    return g


def _on_court(D, p) -> bool:
    xy = D.xy[p.det][D.has_court[p.det]]
    if not len(xy):
        return False
    m = np.median(xy, axis=0)
    return bool(court.in_court(m[None], ON_COURT_MARGIN_M)[0])


def _all_in_view(view, i: int, pts: np.ndarray) -> bool | None:
    """True when every probe point is in plain view at sample i (inside the image, not behind a person box, not under
    an overlay) - the OcclusionContext.classify rules, vectorised. None = no calibration at i."""
    h = view.Hinv[i] if 0 <= i < view.n else None
    if h is None:
        return None
    q = pts @ h[:2, :2].T + h[:2, 2]
    w = pts @ h[2, :2] + h[2, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = q / w[:, None]
    ok = np.isfinite(u).all(axis=1) & (np.abs(w) > 1e-12)
    W, Hh = view.size
    vis = ok & (u[:, 0] >= 0) & (u[:, 0] <= W) & (u[:, 1] >= 0) & (u[:, 1] <= Hh - 3)
    if not vis.all():
        return False
    for ox1, oy1, ox2, oy2 in view.overlay:
        if ((u[:, 0] >= ox1) & (u[:, 0] <= ox2) & (u[:, 1] >= oy1 - 4) & (u[:, 1] <= oy2)).any():
            return False
    b = view.boxes[i]
    if len(b):
        bh = b[:, 3] - b[:, 1]
        sx = view.slack_rel * 0.5 * bh
        cover = ((b[None, :, 0] - sx <= u[:, None, 0]) & (u[:, None, 0] <= b[None, :, 2] + sx)
                 & (b[None, :, 1] <= u[:, None, 1]) & (b[None, :, 3] >= u[:, None, 1] - view.slack_rel * bh))
        if cover.any():
            return False
    return True


def view_violations(D, view, a, b, limit: int | None = None) -> tuple[int, int]:
    """(probe samples where the person could not have been hidden, probed samples) for 'b continues a'."""
    if view is None or a.c1 is None or b.c0 is None:
        return 0, 0
    step = max(1, int(round(PROBE_DT_S * D.fps)))
    ks = np.arange(a.k1 + step, b.k0, step)
    if not len(ks):
        return 0, 0
    ta = (ks - a.k1) / D.fps
    tb = (b.k0 - ks) / D.fps
    f = ta / (ta + tb)
    cen = a.c1[None] * (1 - f[:, None]) + b.c0[None] * f[:, None]
    r = np.minimum(np.minimum(SPEED_MPS * ta, SPEED_MPS * tb) + SLACK_M, 8.0)
    ang = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    ring = np.c_[np.cos(ang), np.sin(ang)]
    offs = np.concatenate([np.zeros((1, 2)), 0.5 * ring, ring])  # 17 probe points per sample (scaled by r)
    seen = 0
    for j, kk in enumerate(ks):
        if _all_in_view(view, int(kk), cen[j] + r[j] * offs):
            seen += 1
            if limit is not None and seen > limit:
                break
    return seen, len(ks)


def joint_link(D, pieces: list, view=None) -> tuple[list, dict]:
    """Long-gap joint linking (see module doc). Returns (links [(a, b)], stats); a/b index ``pieces``."""
    from replay import tracklets as tl

    fps = D.fps
    n = len(pieces)
    stats = {"joint_candidates": 0, "joint_view_refused": 0, "joint_cap_refused": 0, "joint_links": 0}
    if n < 2:
        return [], stats
    grp = kit_groups(pieces)
    big = np.array([len(p.det) >= MIN_DETS and p.conf_max >= MIN_CONF and grp[i] >= 0
                    for i, p in enumerate(pieces)])
    onc = np.array([_on_court(D, p) if big[i] else False for i, p in enumerate(pieces)])
    ns = int(D.k.max()) + 2
    count = np.zeros((N_GROUPS, ns), np.int16)  # on-court identities alive per kit group and sample
    for i, p in enumerate(pieces):
        if onc[i]:
            count[grp[i], p.k0:p.k1 + 1] += 1
    stats["joint_group_sizes"] = [int((grp == g).sum()) for g in range(N_GROUPS)]
    stats["joint_group_p90_alive"] = [int(np.percentile(count[g], 90)) for g in range(N_GROUPS)]
    lo_k, hi_k = int(round(MIN_GAP_S * fps)), int(round(MAX_GAP_S * fps))
    starts = np.array([p.k0 for p in pieces])
    order = np.argsort(starts, kind="stable")
    s_sorted = starts[order]
    cand = []
    for a in np.flatnonzero(big):
        pa = pieces[a]
        lo = np.searchsorted(s_sorted, pa.k1 + lo_k, "left")
        hi = np.searchsorted(s_sorted, pa.k1 + hi_k, "right")
        for b in order[lo:hi]:
            if not big[b] or grp[b] != grp[a] or (ON_COURT_ONLY and not (onc[a] and onc[b])):
                continue
            if TEAM_KITS_ONLY and (tl._achromatic(pa) or tl._achromatic(pieces[b])):
                continue
            pb = pieces[b]
            c = tl._link_cost(D, pa, pb, hi_k)
            if c is None or c >= MAX_COST:
                continue
            cand.append((c, int(a), int(b)))
    stats["joint_candidates"] = len(cand)
    cand.sort()
    has_next, has_prev = set(), set()
    first_of = list(range(n))  # chain bookkeeping: first / last piece of each piece's current chain
    last_of = list(range(n))
    nxt: dict[int, int] = {}
    links = []
    for c, a, b in cand:
        if a in has_next or b in has_prev or first_of[a] == first_of[b]:
            continue
        pa, pb = pieces[a], pieces[b]
        gap = slice(pa.k1 + 1, pb.k0)
        both_on = onc[a] and onc[b]
        if both_on and count[grp[a], gap].size and int(count[grp[a], gap].max()) + 1 > CAP:
            stats["joint_cap_refused"] += 1
            continue
        n_probe = max(0, (pb.k0 - pa.k1 - 1) // max(1, int(round(PROBE_DT_S * fps))))
        allow = int(max(VIEW_MIN_ALLOW, VIEW_ALLOW * n_probe))
        v, m = view_violations(D, view, pa, pb, limit=allow)
        if m and v > allow:
            stats["joint_view_refused"] += 1
            continue
        c = c + VIEW_COST * v
        if c >= MAX_COST:
            continue
        links.append((a, b))
        has_next.add(a)
        has_prev.add(b)
        if both_on:
            count[grp[a], gap] += 1
        nxt[a] = b
        fa, lb = first_of[a], last_of[b]  # merge chains
        j = fa
        while True:
            first_of[j], last_of[j] = fa, lb
            if j not in nxt:
                break
            j = nxt[j]
        D.link_info[(a, b)] = (float(c), (pb.k0 - pa.k1) / fps, MAX_COST, tl._strong_same(D, pa, pb))
    stats["joint_links"] = len(links)
    return links, stats
