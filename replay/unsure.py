"""Unsure-segment detector: where should the coach look first when checking identities?

Pure functions on the existing pass-2 outputs (tracking.json rows + its "tracks" info); no GUI, no pipeline hook.
Each reason marks a time segment of one output id (track_id):

  * ``low_confidence``  rows whose ``id_confidence`` is below CONF_THR (tracklets._identity_confidence)
  * ``long_gap``        the id was re-attached over a gap of more than GAP_S without a measured ("seen") row
  * ``cross_team``      one id (or one continuous ``player_id``) drawn in both team colours: the minority team's rows
  * ``bench_zone``      the id lives in the bench strip (bench_guard.score_identities says bench)
  * ``number_conflict`` two ids of one team with the same jersey number at the same time, or one player_id joining
                        ids with different numbers

Overlapping segments of one id are merged; the result is ranked by score (sum of the reasons' weights), highest
first. Usage::

    segs = unsure_segments(rows_from_tracking(doc), doc.get("tracks"), length=28, width=15)
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable

CONF_THR = 0.6
GAP_S = 4.0
MERGE_S = 0.5  # segments of one id closer than this are one segment
W_LOW = 2.0  # x (CONF_THR - confidence) / CONF_THR
W_GAP = 1.0  # + min(gap / 10 s, 2)
W_CROSS = 3.0
W_BENCH = 2.5  # x bench score
W_NUMBER = 3.0
MEASURED = ("seen", "")  # row sources that are a detection (old outputs have no source column: "")
PEOPLE = ("player", "referee")


def rows_from_tracking(doc: dict) -> list[dict]:
    """tracking.json {"columns", "rows"} -> list of row dicts."""
    cols = doc.get("columns") or []
    return [dict(zip(cols, r)) for r in doc.get("rows") or []]


def _f(v: Any) -> float | None:
    if v is None or v == "":
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _runs(ts: list[float], flags: list[bool], max_step: float) -> list[tuple[float, float, list[int]]]:
    """Runs of consecutive flagged rows (ts sorted); a time step above max_step ends a run."""
    out, cur = [], []
    for i, (t, on) in enumerate(zip(ts, flags)):
        if on and cur and t - ts[cur[-1]] > max_step:
            out.append(cur)
            cur = []
        if on:
            cur.append(i)
        elif cur:
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return [(ts[r[0]], ts[r[-1]], r) for r in out]


def _seg(tid: str, t0: float, t1: float, kind: str, weight: float, **detail) -> dict:
    return {"id": tid, "t0": round(t0, 2), "t1": round(t1, 2),
            "reasons": [{"kind": kind, "weight": round(weight, 3), **detail}]}


def _by_id(rows: Iterable[dict]) -> dict[str, list[dict]]:
    by: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r.get("object_type", "player") not in PEOPLE or r.get("track_id") in (None, ""):
            continue
        if _f(r.get("timestamp")) is None:
            continue
        by[str(r["track_id"])].append(r)
    for v in by.values():
        v.sort(key=lambda r: float(r["timestamp"]))
    return by


def low_confidence(by: dict[str, list[dict]], thr: float = CONF_THR) -> list[dict]:
    out = []
    for tid, rs in by.items():
        ts = [float(r["timestamp"]) for r in rs]
        conf = [_f(r.get("id_confidence")) for r in rs]
        step = _median_step(ts)
        for t0, t1, idx in _runs(ts, [c is not None and c < thr for c in conf], 3 * step):
            c = sorted(conf[i] for i in idx)[len(idx) // 2]
            out.append(_seg(tid, t0, t1, "low_confidence", W_LOW * (thr - c) / thr, confidence=round(c, 3)))
    return out


def long_gaps(by: dict[str, list[dict]], gap_s: float = GAP_S) -> list[dict]:
    out = []
    for tid, rs in by.items():
        ts = [float(r["timestamp"]) for r in rs if str(r.get("source", "") or "") in MEASURED]
        for a, b in zip(ts, ts[1:]):
            if b - a > gap_s:
                out.append(_seg(tid, a, b, "long_gap", W_GAP + min((b - a) / 10.0, 2.0), gap_s=round(b - a, 2)))
    return out


def _median_step(ts: list[float]) -> float:
    d = sorted(b - a for a, b in zip(ts, ts[1:]) if b > a)
    return d[len(d) // 2] if d else 0.1


def cross_team(by: dict[str, list[dict]]) -> list[dict]:
    """Ids drawn in both team colours, and player_ids spanning ids of both teams: the minority team's rows."""
    out = []
    for tid, rs in by.items():
        teams = [r.get("team") for r in rs]
        n = {k: teams.count(k) for k in ("A", "B")}
        if n["A"] and n["B"]:
            minority = "A" if n["A"] < n["B"] else "B"
            ts = [float(r["timestamp"]) for r in rs]
            for t0, t1, _ in _runs(ts, [t == minority for t in teams], 3 * _median_step(ts)):
                out.append(_seg(tid, t0, t1, "cross_team", W_CROSS, team=minority))
    by_pid: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for tid, rs in by.items():
        for r in rs:
            pid = r.get("player_id")
            if pid not in (None, "") and r.get("team") in ("A", "B"):
                by_pid[str(pid)][tid].append(r)
    for pid, ids in by_pid.items():
        team_rows = {k: sum(1 for rs in ids.values() for r in rs if r["team"] == k) for k in ("A", "B")}
        if not (team_rows["A"] and team_rows["B"]):
            continue
        minority = "A" if team_rows["A"] < team_rows["B"] else "B"
        for tid, rs in ids.items():
            mine = [r for r in rs if r["team"] == minority]
            if mine and len(mine) * 2 > len(rs):  # this id belongs to the minority team
                out.append(_seg(tid, float(mine[0]["timestamp"]), float(mine[-1]["timestamp"]), "cross_team",
                                W_CROSS, team=minority, player_id=pid))
    return out


def bench_zone(by: dict[str, list[dict]], tracks: dict | None, length: float, width: float) -> list[dict]:
    from replay import bench_guard

    pts = {}
    for tid, rs in by.items():
        p = [(float(r["timestamp"]), _f(r.get("x_m")), _f(r.get("y_m")), str(r.get("source", "") or "") in MEASURED)
             for r in rs]
        p = [q for q in p if q[1] is not None and q[2] is not None]
        if p:
            pts[tid] = p
    team = {str(k): (v or {}).get("team") for k, v in (tracks or {}).items()}
    for tid, rs in by.items():
        team.setdefault(tid, rs[0].get("team"))
    out = []
    for tid, s in bench_guard.score_identities(pts, length, width, team=team).items():
        if s.is_bench:
            ts = [q[0] for q in pts[tid]]
            out.append(_seg(tid, min(ts), max(ts), "bench_zone", W_BENCH * float(s.score)))
    return out


def number_conflicts(by: dict[str, list[dict]], tracks: dict | None) -> list[dict]:
    out = []
    info = {str(k): v or {} for k, v in (tracks or {}).items()}
    span = {tid: (float(rs[0]["timestamp"]), float(rs[-1]["timestamp"])) for tid, rs in by.items()}
    num = {tid: str(info[tid]["number"]) for tid in by if info.get(tid, {}).get("number") not in (None, "")}
    team = {tid: info.get(tid, {}).get("team") or by[tid][0].get("team") for tid in by}
    ids = sorted(num)
    for i, a in enumerate(ids):  # same team + same number at the same time: one of them is someone else
        for b in ids[i + 1:]:
            if num[a] != num[b] or team[a] != team[b] or team[a] not in ("A", "B"):
                continue
            t0, t1 = max(span[a][0], span[b][0]), min(span[a][1], span[b][1])
            if t1 > t0:
                for tid, other in ((a, b), (b, a)):
                    out.append(_seg(tid, t0, t1, "number_conflict", W_NUMBER, number=num[a], other=other))
    by_pid: dict[str, set] = defaultdict(set)  # one continuous player id over ids with different numbers
    for tid, rs in by.items():
        for r in rs:
            if r.get("player_id") not in (None, ""):
                by_pid[str(r["player_id"])].add(tid)
    for pid, tids in by_pid.items():
        nums = defaultdict(list)
        for tid in tids:
            if tid in num:
                nums[num[tid]].append(tid)
        if len(nums) < 2:
            continue
        main = max(nums, key=lambda k: (sum(len(by[t]) for t in nums[k]), k))
        for n_, ts_ in nums.items():
            if n_ != main:
                for tid in ts_:
                    out.append(_seg(tid, *span[tid], "number_conflict", W_NUMBER, number=n_, expected=main,
                                    player_id=pid))
    return out


def merge(segs: list[dict], gap: float = MERGE_S) -> list[dict]:
    """One segment per id and overlapping time span; reasons concatenated, score = sum of their weights."""
    by: dict[str, list[dict]] = defaultdict(list)
    for s in segs:
        by[s["id"]].append(s)
    out = []
    for tid, ss in by.items():
        ss.sort(key=lambda s: (s["t0"], s["t1"]))
        cur = None
        for s in ss:
            if cur is not None and s["t0"] <= cur["t1"] + gap:
                cur["t1"] = max(cur["t1"], s["t1"])
                cur["reasons"] += s["reasons"]
                continue
            cur = {"id": tid, "t0": s["t0"], "t1": s["t1"], "reasons": list(s["reasons"])}
            out.append(cur)
    for s in out:
        s["score"] = round(sum(r["weight"] for r in s["reasons"]), 3)
        s["kinds"] = sorted({r["kind"] for r in s["reasons"]})
    out.sort(key=lambda s: (-s["score"], s["t0"], s["id"]))
    return out


def unsure_segments(rows: Iterable[dict], tracks: dict | None = None, *, length: float = 28.0, width: float = 15.0,
                    conf_thr: float = CONF_THR, gap_s: float = GAP_S) -> list[dict]:
    """Ranked unsure segments [{id, t0, t1, score, kinds, reasons: [{kind, weight, ...}]}] (see module doc)."""
    by = _by_id(rows)
    if not by:
        return []
    segs = (low_confidence(by, conf_thr) + long_gaps(by, gap_s) + cross_team(by) + bench_zone(by, tracks, length, width)
            + number_conflicts(by, tracks))
    return merge(segs)
