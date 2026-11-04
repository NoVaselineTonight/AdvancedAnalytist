"""Reconcile detected made baskets with an attached official scoresheet (running score).

The running score of an SKL / FIBA scoresheet lists every scoring event in order: team, points (1/2/3), scorer
number and period - but no game-clock or video time. The video detections have times but miss baskets, add false
ones and sometimes credit the wrong team. :func:`align` finds the best order-preserving correspondence
(dynamic programming, like a sequence alignment) between the two lists:

* a pair is allowed only when the detection lies in a video segment that can hold the sheet event's period
  (segments are cut at the game state's quarter breaks / half time; a missing break merges periods);
* a pair earns more when the detected team (and free throw / field goal kind, when known) agree;
* skipping a sheet event (missed basket) is cheap, skipping a detection (likely false basket) costs a little.

Result per detection: the sheet event it is (official team, points, scorer number) or None (not on the
sheet = probably not a basket). Sheet events without a detection get an estimated time between their aligned
neighbours. Nothing is aligned without a sheet; the sheet's totals are always exact, the per-event mapping is
an estimate (``confidence``).
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

MATCH = 2.0
TEAM_OK, TEAM_BAD = 1.5, -3.0
KIND_OK, KIND_BAD = 0.5, -0.5
SKIP_DET = -0.6
SKIP_SHEET = 0.0
POS_W = 3.0  # penalty per unit of |position in segment (sheet rank) - position in segment (video time)|


def segments(breaks: Sequence[dict], t_end: float) -> list[tuple[float, float, set[int]]]:
    """Video segments with the periods they can contain, from quarter_break / half_time breaks."""
    brk = sorted((b for b in breaks if b.get("kind") in ("quarter_break", "half_time")), key=lambda b: b["t0"])
    half = next((b for b in brk if b["kind"] == "half_time"), None)
    out, t0 = [], 0.0
    if half is None:  # no half time known: periods by count of breaks only if 3 breaks were seen
        if len(brk) == 3:
            for k, b in enumerate(brk):
                out.append((t0, float(b["t0"]), {k + 1}))
                t0 = float(b["t1"])
            out.append((t0, t_end, {4}))
            return out
        return [(0.0, t_end, {1, 2, 3, 4})]
    first = [b for b in brk if b["t0"] < half["t0"]]
    second = [b for b in brk if b["t0"] > half["t0"]]
    if first:
        out.append((0.0, float(first[0]["t0"]), {1}))
        out.append((float(first[0]["t1"]), float(half["t0"]), {2}))
    else:
        out.append((0.0, float(half["t0"]), {1, 2}))
    if second:
        out.append((float(half["t1"]), float(second[0]["t0"]), {3}))
        out.append((float(second[0]["t1"]), t_end, {4}))
    else:
        out.append((float(half["t1"]), t_end, {3, 4}))
    return out


def _periods_at(t: float, segs: list[tuple[float, float, set[int]]]) -> set[int]:
    for a, b, ps in segs:
        if a - 30 <= t <= b + 30:
            return ps
    return {1, 2, 3, 4}


def align(sheet_events: Sequence[dict], dets: Sequence[dict], segs: list[tuple[float, float, set[int]]]
          ) -> dict[str, Any]:
    """sheet_events: running score in order [{team, points, scorer, kind ('ft'|'fg2'|'fg3'), period}];
    dets: detections in time order [{t, team, kind ('ft'|'fg'|None)}]. Returns {"det_to_sheet": [k or None],
    "sheet_to_det": [j or None], "score": float}."""
    S, D = list(sheet_events), list(dets)
    n, m = len(S), len(D)
    per = [_periods_at(float(d["t"]), segs) for d in D]
    # where in its segment is each event? (sheet: rank among the segment's events; detection: time fraction)
    seg_of_p = {p: k for k, (_, _, ps) in enumerate(segs) for p in ps}
    cnt: dict[int, int] = {}
    s_frac = []
    for s in S:
        k = seg_of_p.get(s.get("period"), -1)
        cnt[k] = cnt.get(k, 0) + 1
    seen: dict[int, int] = {}
    for s in S:
        k = seg_of_p.get(s.get("period"), -1)
        seen[k] = seen.get(k, 0) + 1
        s_frac.append((seen[k] - 0.5) / cnt[k])
    d_frac = []
    for d in D:
        t = float(d["t"])
        k = next((kk for kk, (a, b, _) in enumerate(segs) if a - 30 <= t <= b + 30), None)
        d_frac.append(None if k is None else float(np.clip((t - segs[k][0]) / max(1.0, segs[k][1] - segs[k][0]), 0, 1)))

    def gain(s: dict, d: dict, ps: set[int], sf: float, df: float | None) -> float | None:
        if s.get("period") not in ps:
            return None
        g = MATCH + (TEAM_OK if d.get("team") == s["team"] else TEAM_BAD if d.get("team") in ("A", "B") else 0.0)
        if df is not None:
            g -= POS_W * abs(sf - df)
        dk = d.get("kind")
        if dk:
            g += KIND_OK if (dk == "ft") == (s.get("kind") == "ft") else KIND_BAD
        return g

    V = np.full((n + 1, m + 1), -np.inf)
    V[0, :] = SKIP_DET * np.arange(m + 1)
    V[:, 0] = SKIP_SHEET * np.arange(n + 1)
    B = np.zeros((n + 1, m + 1), np.int8)  # 1 = match, 2 = skip sheet, 3 = skip det
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            best, arg = V[i - 1, j] + SKIP_SHEET, 2
            v = V[i, j - 1] + SKIP_DET
            if v > best:
                best, arg = v, 3
            g = gain(S[i - 1], D[j - 1], per[j - 1], s_frac[i - 1], d_frac[j - 1])
            if g is not None and V[i - 1, j - 1] + g > best:
                best, arg = V[i - 1, j - 1] + g, 1
            V[i, j], B[i, j] = best, arg
    d2s: list[int | None] = [None] * m
    s2d: list[int | None] = [None] * n
    i, j = n, m
    while i > 0 and j > 0:
        a = B[i, j]
        if a == 1:
            d2s[j - 1], s2d[i - 1] = i - 1, j - 1
            i, j = i - 1, j - 1
        elif a == 2:
            i -= 1
        else:
            j -= 1
    return {"det_to_sheet": d2s, "sheet_to_det": s2d, "score": float(V[n, m])}


def estimate_times(n_sheet: int, s2d: Sequence[int | None], det_t: Sequence[float],
                   sheet_periods: Sequence[int], segs: list[tuple[float, float, set[int]]]) -> list[float]:
    """Video time per sheet event: the aligned detection's time, else interpolated between aligned neighbours
    (by event index) inside the period's segment."""
    known = {k: float(det_t[j]) for k, j in enumerate(s2d) if j is not None}
    out = []
    for k in range(n_sheet):
        if k in known:
            out.append(known[k])
            continue
        seg = next(((a, b) for a, b, ps in segs if sheet_periods[k] in ps), (0.0, max(det_t or [0.0])))
        lo = max((kk for kk in known if kk < k and seg[0] - 30 <= known[kk] <= seg[1] + 30), default=None)
        hi = min((kk for kk in known if kk > k and seg[0] - 30 <= known[kk] <= seg[1] + 30), default=None)
        a_t, a_k = (known[lo], lo) if lo is not None else (seg[0], -1)
        b_t, b_k = (known[hi], hi) if hi is not None else (seg[1], n_sheet)
        out.append(a_t + (b_t - a_t) * (k - a_k) / max(1, b_k - a_k))
    return out
