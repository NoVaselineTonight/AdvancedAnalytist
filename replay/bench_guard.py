"""Bench guard: whole-life "bench-likeness" of an output identity (prototype, pure functions, no pipeline hooks).

Motivation (data/resume/benchmark.md, height-gate round 3): the remaining phantom rows in c300 / p4400 are bench
people in team kit whose feet sit 0-0.5 m beyond the far sideline next to the bench (c300 x 19-21 m, y 15.0-15.5 m;
p4400 x 21-23 m, y 13-15 m). Per-row guards (roster static/posture cues, limits caps) miss them because the kids
move a little and look like players frame by frame. Scored over the WHOLE identity life they are obvious: median
position in the bench strip, nearly all rows there, no court-crossing travel, never deep in the court.

Usage::

    scores = score_identities({tid: [(t, x, y, measured), ...]}, length=26.0, width=14.5)
    bench = {tid for tid, s in scores.items() if s.is_bench}

Positions are court metres (x along the sideline 0..length, y along the baseline 0..width).  ``measured`` marks
rows with a detected box ("seen"); predicted / interpolated rows are ignored when enough measured rows exist.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

STRIP_IN_M = 1.0  # bench strip reaches this far INSIDE the court from the sideline ...
STRIP_OUT_M = 2.5  # ... and this far outside it (beyond: spectators, not drawn anyway)
BASELINE_GUARD_M = 4.0  # inside-court rows closer than this to a baseline are never bench rows
DEEP_M = 2.5  # rows further than this from every sideline are "on court" evidence
STRIP_SHARE_MIN = 0.7  # bench identity: at least this share of rows in the strip ...
DEEP_SHARE_MAX = 0.10  # ... and at most this share deep in the court
TRAVEL_MAX_M = 4.0  # ... and the position cloud radius (95th percentile from the median) is at most this
SPEED_MAX_MPS = 1.2  # ... and median speed (displacement over SPEED_WIN_S windows) below this (sitting / standing)
SPEED_WIN_S = 1.5
MIN_ROWS = 5  # fewer rows: no decision (score reported, is_bench False)
SCORE_MIN = 0.75  # combined score needed for is_bench


@dataclass
class BenchScore:
    rows: int
    median_xy: tuple[float, float]
    median_off_m: float  # signed metres outside the nearest sideline (negative = inside the court)
    strip_share: float  # share of rows in the bench strip (within STRIP_IN_M inside .. STRIP_OUT_M outside)
    beyond_share: float  # share of rows beyond the sideline (off court)
    deep_share: float  # share of rows deeper than DEEP_M inside the court
    travel_m: float  # 95th percentile distance from the median position
    speed_mps: float  # median speed
    entered_from_court: bool  # first 20 % of rows were deep in the court (substituted player, not a phantom)
    score: float  # 0..1 bench-likeness
    is_bench: bool
    side: str = ""  # "far" (y > width/2) or "near"
    reasons: list[str] = field(default_factory=list)


def _pct(v: list[float], q: float) -> float:
    if not v:
        return 0.0
    s = sorted(v)
    k = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[k]


def _median(v: list[float]) -> float:
    return _pct(v, 0.5)


def signed_off_sideline(y: float, width: float) -> float:
    """Signed distance outside the NEAREST sideline (negative inside the court)."""
    return max(-y, y - width)


def score_track(rows, length: float, width: float, bench_x: tuple[float, float] | None = None) -> BenchScore:
    """``rows``: iterable of (t, x, y, measured[, ...]) for one identity. ``bench_x``: optional x range (m) of the
    bench area along the sideline; rows outside the range count as strip rows only when beyond the sideline."""
    rows = sorted(rows, key=lambda r: r[0])
    meas = [r for r in rows if bool(r[3])]
    use = meas if len(meas) >= MIN_ROWS else rows
    n = len(use)
    if n == 0:
        return BenchScore(0, (math.nan, math.nan), math.nan, 0, 0, 0, 0, 0, False, 0.0, False)
    xs = [float(r[1]) for r in use]
    ys = [float(r[2]) for r in use]
    mx, my = _median(xs), _median(ys)
    off = [signed_off_sideline(y, width) for y in ys]
    in_strip = []
    for x, y, o in zip(xs, ys, off):
        ok = -STRIP_IN_M <= o <= STRIP_OUT_M
        if ok and o < 0 and (x < BASELINE_GUARD_M or x > length - BASELINE_GUARD_M):
            ok = False  # inside the court near a baseline corner (inbounder, corner shooter): never the bench
        if ok and bench_x is not None and o < 0 and not (bench_x[0] <= x <= bench_x[1]):
            ok = False  # inside the court but not along the bench: a player on the sideline
        in_strip.append(ok)
    strip_share = sum(in_strip) / n
    beyond_share = sum(o > 0 for o in off) / n
    deep_share = sum(o < -DEEP_M for o in off) / n
    dist = [math.hypot(x - mx, y - my) for x, y in zip(xs, ys)]
    travel = _pct(dist, 0.95)
    speeds = []  # displacement over >= SPEED_WIN_S windows (robust to detection jitter)
    j = 0
    for k, a in enumerate(use):
        while j < n and float(use[j][0]) - float(a[0]) < SPEED_WIN_S:
            j += 1
        if j >= n:
            break
        b = use[j]
        dt = float(b[0]) - float(a[0])
        speeds.append(math.hypot(float(b[1]) - float(a[1]), float(b[2]) - float(a[2])) / dt)
    speed = _median(speeds) if speeds else 0.0
    head = use[: max(1, n // 5)]
    entered = sum(signed_off_sideline(float(r[2]), width) < -DEEP_M for r in head) / len(head) > 0.5
    side = "far" if my > width / 2 else "near"

    # combined score: each cue 0..1, product-ish so one strong contradiction kills it
    c_strip = min(1.0, strip_share / STRIP_SHARE_MIN)
    c_deep = max(0.0, 1.0 - deep_share / max(DEEP_SHARE_MAX, 1e-9) * 0.5)  # 2x the allowed share -> 0
    c_travel = max(0.0, 1.0 - max(0.0, travel - TRAVEL_MAX_M) / TRAVEL_MAX_M)
    c_speed = max(0.0, 1.0 - max(0.0, speed - SPEED_MAX_MPS) / SPEED_MAX_MPS)
    c_off = 0.5 + 0.5 * max(-1.0, min(1.0, _median(off) / STRIP_IN_M))  # median at the line = 0.5, beyond -> 1
    score = c_strip * c_deep * c_travel * c_speed * (0.75 + 0.25 * c_off)
    reasons = []
    if strip_share >= STRIP_SHARE_MIN:
        reasons.append(f"strip {strip_share:.0%}")
    if deep_share <= DEEP_SHARE_MAX:
        reasons.append(f"deep {deep_share:.0%}")
    if travel <= TRAVEL_MAX_M:
        reasons.append(f"travel {travel:.1f} m")
    if speed <= SPEED_MAX_MPS:
        reasons.append(f"speed {speed:.1f} m/s")
    is_bench = (n >= MIN_ROWS and score >= SCORE_MIN and strip_share >= STRIP_SHARE_MIN
                and deep_share <= DEEP_SHARE_MAX and travel <= TRAVEL_MAX_M and speed <= SPEED_MAX_MPS
                and not entered)
    return BenchScore(n, (mx, my), _median(off), strip_share, beyond_share, deep_share, travel, speed, entered,
                      round(score, 3), is_bench, side, reasons)


def score_identities(tracks: dict, length: float, width: float, team: dict | None = None,
                     bench_x: tuple[float, float] | None = None, skip_referees: bool = True) -> dict:
    """``tracks``: id -> rows (t, x, y, measured). ``team``: optional id -> team label; "referee" ids are never
    flagged (they stand on the sideline for long stretches). Returns id -> BenchScore."""
    out = {}
    for tid, rows in tracks.items():
        s = score_track(rows, length, width, bench_x)
        if skip_referees and team is not None and str(team.get(tid, "")).lower() == "referee":
            s.is_bench = False
            s.reasons.append("referee: never flagged")
        out[tid] = s
    return out


def bench_identities(tracks: dict, length: float, width: float, **kw) -> set:
    return {tid for tid, s in score_identities(tracks, length, width, **kw).items() if s.is_bench}
