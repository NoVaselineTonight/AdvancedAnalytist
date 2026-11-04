"""Output-id handover stitching (pass 2, last step before the export).

Scoreboard 19.10 (data/benchmark/SCOREBOARD.md): many output ids end and a same-team id starts within 1.5 s a few
metres away. Causes upstream are varied (tracklet link gates on a partial box's height, kit histograms of tiny
pieces, two simultaneous tracker identities on one person of which the caps/limits show one and then the other),
so the shown output churns ids although no two of them are ever on screen together.

Here, after every other step, an output id A that ends and an output id B that starts shortly after are given ONE
id when the handover is unambiguous:
  * same team (A, B or referee), B starts <= ``max_gap_s`` after A's last row (never overlapping in the output),
  * B's first position is reachable from A's last one (<= 1 m + ``speed_mps`` x gap, at most ``max_dist_m``),
  * no conflicting identity evidence: different accepted jersey numbers, lineup slots or player ids keep them apart,
  * mutual uniqueness: A has exactly one such successor and B exactly one such predecessor (crowds stay split).
Only ids change (rows, ball owners, basket shooters, track_info); positions and rows are untouched.
Off/on: ``ENABLED`` (env REPLAY_HANDOVER=0/1 overrides).
"""

from __future__ import annotations

import math
import os

ENABLED = False
MAX_GAP_S = 1.5
MAX_DIST_M = 3.0
SPEED_MPS = 6.0
TEAMS = ("A", "B", "referee")


def enabled() -> bool:
    env = os.environ.get("REPLAY_HANDOVER")
    if env in ("0", "1"):
        return env == "1"
    return ENABLED


def _conflict(ia: dict, ib: dict, pa: set, pb: set) -> bool:
    for key in ("number", "slot"):
        if ia.get(key) is not None and ib.get(key) is not None and str(ia[key]) != str(ib[key]):
            return True
    return bool(pa and pb and not (pa & pb))


def plan(rows: list, track_info: dict, max_gap_s: float = MAX_GAP_S, max_dist_m: float = MAX_DIST_M,
         speed_mps: float = SPEED_MPS) -> dict:
    """{old id: new id} for unambiguous handovers (chains map to their first id). ``rows``: export.TrackRow-like
    objects (track_id, object_type, team, t, x, y, player_id)."""
    span: dict = {}
    for r in rows:
        if r.object_type == "ball":
            continue
        s = span.get(r.track_id)
        if s is None:
            span[r.track_id] = s = {"t0": r.t, "t1": r.t, "xy0": (r.x, r.y), "xy1": (r.x, r.y), "teams": {},
                                    "pids": set()}
        if r.t < s["t0"]:
            s["t0"], s["xy0"] = r.t, (r.x, r.y)
        if r.t > s["t1"]:
            s["t1"], s["xy1"] = r.t, (r.x, r.y)
        s["teams"][r.team] = s["teams"].get(r.team, 0) + 1
        if getattr(r, "player_id", None) is not None:
            s["pids"].add(r.player_id)
    for s in span.values():
        s["team"] = max(s["teams"], key=s["teams"].get)
    ids = [i for i in span if span[i]["team"] in TEAMS]
    by_start = sorted(ids, key=lambda i: span[i]["t0"])
    succ: dict = {}
    pred: dict = {}
    for a in ids:
        sa = span[a]
        for b in by_start:
            sb = span[b]
            gap = sb["t0"] - sa["t1"]
            if gap <= 0:
                continue
            if gap > max_gap_s:
                break
            if b == a or sb["team"] != sa["team"]:
                continue
            if math.dist(sa["xy1"], sb["xy0"]) > min(max_dist_m, 1.0 + speed_mps * gap):
                continue
            if _conflict(track_info.get(a, {}), track_info.get(b, {}), sa["pids"], sb["pids"]):
                continue
            succ.setdefault(a, []).append(b)
            pred.setdefault(b, []).append(a)
    nxt = {a: bs[0] for a, bs in succ.items() if len(bs) == 1 and len(pred.get(bs[0], [])) == 1}
    has_pred = set(nxt.values())
    out: dict = {}
    for a in nxt:
        if a in has_pred:
            continue  # not a chain head
        head, cur = a, a
        while cur in nxt:
            cur = nxt[cur]
            out[cur] = head
    return out


def stitch(rows: list, track_info: dict, events: list | None = None) -> tuple[dict, int]:
    """Apply plan() in place to rows (track_id, ball_owner), events (details.shooter_track) and track_info.
    Returns (track_info, number of ids merged). No-op unless enabled()."""
    if not enabled():
        return track_info, 0
    m = plan(rows, track_info)
    if not m:
        return track_info, 0
    for r in rows:
        if r.track_id in m:
            r.track_id = m[r.track_id]
        if getattr(r, "ball_owner", None) in m:
            r.ball_owner = m[r.ball_owner]
    for e in events or []:
        d = e.get("details") or {}
        if d.get("shooter_track") in m:
            d["shooter_track"] = m[d["shooter_track"]]
    for old, new in m.items():
        info = track_info.pop(old, None)
        if info and new in track_info:
            for k, v in info.items():
                track_info[new].setdefault(k, v)
        elif info:
            track_info[new] = info
    return track_info, len(m)
