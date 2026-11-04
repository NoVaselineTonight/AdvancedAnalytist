"""Ball events: per-frame ball state, passes, shots (with location/outcome), bad passes, steals, lost balls.

Works on top of the 2D replay (``replay_data.json``: ball court position / state / owner / image box per frame,
player tracks with team, jersey numbers in ``track_info``) and the game state (``game_state.py``: possessions,
shots with release / rim / outcome, dead balls). Deterministic, no Qt, no video.

* **Ball state per frame** (``states`` run list): ``held`` (by whom), ``air`` (between two holders or after a
  release: pass / shot / loose), ``loose`` (not held, nobody near, slow or unseen), ``dead`` (game-state dead
  ball windows: breaks, free-throw set-ups, referee holding the ball), ``unknown``.
* **Holder runs**: consecutive frames held by the same player track (short unseen gaps bridged).
* **Passes**: run of player P (team X) followed by a run of player Q != P. Same team, the two players at least
  ``min_pass_m`` apart, the ball off hands for at most ``max_pass_s`` -> completed pass P -> Q (type from the
  image trajectory when the ball was seen: bounce / lob / direct). An ID switch of one player (Q appears where
  P was) is not a pass. Other team (and the game state confirms a change of possession):
  * Q took the ball from P at close range (<= ``steal_max_m``) quickly -> **steal** (ball taken from the
    handler), P loses the ball ("Lost ball (stolen)");
  * the ball travelled further -> **bad pass (stolen)** by P, interception (steal) by Q.
  A team change through a shot (release ... rim) or a dead ball is not a pass/turnover.
* **Shots**: the game-state attempts (rim evidence + release + made basket match) are enriched: shooter (jersey
  number), release time, court location (shot chart), distance, outcome (made / missed -> rebound team), arc
  (image apex above the release point in ball diameters when the ball was seen) and titled
  "Shot (missed) from 6.1 m".

Output: :func:`analyse` -> dict (``ball_events`` key of ``game_state.json``); :func:`attach` adds its events to
the game-state events (timeline Auto detection track, data export, AI prompts) and replaces the game state's
coarser steal / turnover events at the same moment.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

VERSION = 13
SOURCE = "ball_events"
TEAMS = ("A", "B")
STATES = ("held", "dribble", "air", "floor", "loose", "dead", "unknown")


@dataclass
class BEConfig:
    hold_gap_s: float = 0.35  # unseen frames inside one holder run
    min_run_frames: int = 2  # a holder run needs this many held frames ...
    min_run_s: float = 0.15  # ... or this duration
    max_pass_s: float = 2.2  # ball off hands at most this long for a pass
    min_pass_m: float = 2.0  # passer -> receiver distance (court) for a pass
    max_pass_m: float = 26.0
    id_switch_m: float = 1.2  # Q appears this close to where P was (and P is gone): same player, new track id
    steal_max_m: float = 1.8  # other-team player gains the ball this close to the handler = steal
    steal_max_gap_s: float = 1.0
    change_confirm_s: float = 1.2  # the new team must keep the ball this long (game-state possession)
    shot_guard_s: float = 0.6  # a release this close before a shot release belongs to the shot
    min_event_conf: float = 0.35
    unconfirmed_steal_factor: float = 0.6  # game-state steal with no ball-level hand-over: confidence x this
    dribble_rel_y: float = 0.6  # held ball below this fraction of the holder's box height = dribble
    air_mps: float = 3.0  # ball off hands moving at least this fast (camera motion removed) = in the air
    use_flights: bool = True  # passes from image ball flights between two players (see _flight_events)
    flight_min_frames: int = 2  # ball seen flying (outside every box, >= air_mps) this many frames
    flight_box_mx: float = 0.2  # thrower / catcher box tolerance: width share each side ...
    flight_box_my: float = 0.1  # ... and height share above the head (ball held overhead)
    flight_search: int = 4  # frames before / after a flight to find the thrower / catcher
    flight_max_rel_y: float = 0.42  # highest ball point in the flight below this share of the thrower's box: dribble
    flight_after_shot_s: float = 1.0  # flights this long after a shot reached the rim are rebounds / tips
    flight_dup_s: float = 0.8  # a flight pass this close to a holder-run pass is the same pass
    oob_after_catch_s: float = 0.6  # ball gone out this soon after a catch: the pass was the bad one
    oob_window_s: float = 3.5  # flight without a catch, then an inbound / out-of-bounds call within this: bad pass out
    flight_air_frames: int = 99  # ball seen flying (image speed) in >= this many frames between: 1 frame less needed
    min_pass_run_frames: int = 3  # passer and receiver must each hold the ball >= this many frames
    dup_iou: float = 0.3  # passer and receiver boxes overlap this much: one person with two tracks
    dup_iou_other_team: float = 0.6  # (other team: overlapping boxes are mostly body contact = a steal)
    contact_iou: float = 0.1  # handler and defender boxes touch at the change: ball taken, not intercepted
    still_mps: float = 0.9  # median player speed below this around a hand-over: dead ball
    pass_speed_mps: float = 8.0  # typical; slower = lob / loose, used for confidence


# ====================================================================================== helpers
def _f(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def _name(gi: Any, tid: str | None) -> str:
    if tid is None:
        return "?"
    num = gi.track_number.get(str(tid))
    return f"#{num}" if num else f"T{tid}"


def holder_runs(gi: Any, cfg: BEConfig) -> list[dict]:
    """Runs of frames held by one player (players of team A/B only; referees end runs)."""
    runs: list[dict] = []
    cur: dict | None = None
    for i in range(gi.n):
        o = gi.ball_o[i]
        if gi.ball_f[i] != 1 or o is None:
            continue
        team = gi.track_team.get(o, "U")
        if cur is not None and cur["owner"] == o and gi.t[i] - gi.t[cur["i1"]] <= cfg.hold_gap_s:
            cur["i1"] = i
            cur["k"] += 1
            continue
        cur = {"owner": o, "team": team, "i0": i, "i1": i, "k": 1}
        runs.append(cur)
    return [r for r in runs if r["k"] >= cfg.min_run_frames or gi.t[r["i1"]] - gi.t[r["i0"]] >= cfg.min_run_s]


def _xy(gi: Any, tid: str | None, i: int, max_frames: int = 3) -> np.ndarray | None:
    from replay.game_state import _track_xy_near
    return _track_xy_near(gi, tid, i, max_frames)


def _track_span(gi: Any, tid: str) -> tuple[int, int]:
    rows = gi.track_rows.get(str(tid))
    if rows is None or not len(rows):
        return (-1, -1)
    fi = gi.r_i[rows]
    return int(fi[0]), int(fi[-1])


def _in(t: float, windows: list[tuple[float, float]]) -> bool:
    return any(a <= t <= b for a, b in windows)


def _overlaps(a: float, b: float, windows: list[tuple[float, float]]) -> bool:
    return any(wa <= b and wb >= a for wa, wb in windows)


MOTION_FILE = "ball_motion.npz"


def save_motion(results_dir: Any, G: Any, seg: Any) -> None:
    """Pass-2 hook: keep the camera-motion chain next to the outputs so rebuilds (GUI, export) can compensate the
    pan without pass 1. Never raises."""
    try:
        from pathlib import Path
        np.savez_compressed(Path(results_dir) / MOTION_FILE, G=np.asarray(G, np.float64).reshape(-1, 3, 3),
                            seg=np.asarray(seg, np.int32))
    except Exception as e:  # noqa: BLE001
        print("ball events: motion not saved:", type(e).__name__, e)


def load_motion(results_dir: Any, n: int) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        from pathlib import Path
        z = np.load(Path(results_dir) / MOTION_FILE)
        G, seg = z["G"], z["seg"]
        return (G, seg) if len(G) == n == len(seg) else None
    except Exception:  # noqa: BLE001 - missing / old file
        return None


def _ball_image(replay: dict, n: int) -> np.ndarray:
    """(n, 3) ball centre x, y and diameter in pixels from the stored ball boxes (nan = not seen)."""
    out = np.full((n, 3), np.nan)
    for i, b in enumerate(((replay.get("ball") or {}).get("b") or [])[:n]):
        if b:
            x1, y1, x2, y2 = (float(v) for v in b[:4])
            out[i] = ((x1 + x2) / 2, (y1 + y2) / 2, max(2.0, 0.5 * ((x2 - x1) + (y2 - y1))))
    return out


def ball_speed_mps(img: np.ndarray, t: np.ndarray, motion_chain: tuple | None = None, max_gap: int = 2) -> np.ndarray:
    """Ball speed per frame (m/s, image plane, the ball's own diameter as the 0.24 m ruler) from the previous
    sighting at most ``max_gap`` frames back; with the camera chain the previous position is moved into the
    current frame's pixels first (camera pan removed)."""
    n = len(img)
    out = np.full(n, np.nan)
    last = -1
    for i in range(n):
        if not np.isfinite(img[i, 0]):
            continue
        j = last
        last = i
        if j < 0 or i - j > max_gap:
            continue
        p = img[j, :2]
        if motion_chain is not None:
            G, seg = motion_chain
            if seg[i] != seg[j]:
                continue
            M = np.linalg.inv(G[i]) @ G[j]
            q = M @ np.array([p[0], p[1], 1.0])
            p = q[:2] / q[2]
        d = float(np.hypot(*(img[i, :2] - p)))
        dia = 0.5 * (img[i, 2] + img[j, 2])
        dt = float(t[i] - t[j])
        if dt > 0:
            out[i] = d / dia * 0.24 / dt
    return out


def _person_at(boxes_i: list, cx: float, cy: float, team_of: dict, mx: float = 0.2, my: float = 0.1
               ) -> tuple | None:
    """The team-A/B player whose image box holds the ball point (box widened by ``mx`` of its width each side and
    raised by ``my`` of its height for a ball held overhead), most central first."""
    best, bs = None, np.inf
    for tid, bx in boxes_i:
        if team_of.get(tid) not in TEAMS:
            continue
        w, h = bx[2] - bx[0], bx[3] - bx[1]
        if bx[0] - mx * w <= cx <= bx[2] + mx * w and bx[1] - my * h <= cy <= bx[3]:
            sc = abs(cx - (bx[0] + bx[2]) / 2) / max(w, 1.0)
            if sc < bs:
                best, bs = (tid, bx), sc
    return best


def _flight_events(gi: Any, cfg: "BEConfig", img: np.ndarray, spd: np.ndarray, boxes: dict, owner: list,
                   shot_win: list, ref_dead: list, oob_t: list, poss_team_at: Any, doc: dict, add: Any,
                   replay: dict, doc_events: list) -> None:
    """Ball flights seen in the image (observed, outside every player box, camera-compensated speed >= air_mps,
    one unseen frame bridged): the thrower is the holder / the player whose box holds the ball just before, the
    catcher the one just after. Same team, >= min_pass_m apart -> pass (unless a holder-run pass already covers
    it). No catcher, then an inbound / out-of-bounds call and the other team's ball -> bad pass out of bounds."""
    from replay.game_state import _runs
    n = gi.n
    by_i: dict = {}
    for tid, per in boxes.items():
        for i, bx in per.items():
            by_i.setdefault(i, []).append((tid, bx))
    team_of = gi.track_team
    obs = np.isfinite(img[:, 0])
    air = np.zeros(n, bool)
    for i in np.flatnonzero(obs):
        cx, cy = img[i, 0], img[i, 1]
        inside = any(bx[0] <= cx <= bx[2] and bx[1] <= cy <= bx[3] for _, bx in by_i.get(i, ()))
        air[i] = (not inside) and bool(np.isfinite(spd[i])) and spd[i] >= cfg.air_mps
    bridged = air.copy()
    for i in range(1, n - 1):
        if not air[i] and not obs[i] and air[i - 1] and air[i + 1]:
            bridged[i] = True

    def who(i: int) -> tuple | None:
        if not (0 <= i < n):
            return None
        if owner[i] is not None and team_of.get(owner[i]) in TEAMS:
            return owner[i], i
        if obs[i]:
            p = _person_at(by_i.get(i, []), img[i, 0], img[i, 1], team_of, cfg.flight_box_mx, cfg.flight_box_my)
            if p is not None:
                return p[0], i
        return None

    have = [p["t_release"] for p in doc["passes"]]
    for f0, f1 in _runs(bridged):
        n_air = int(np.sum(air[f0:f1]))
        if n_air < cfg.flight_min_frames:
            continue
        t0, t1 = float(gi.t[f0]), float(gi.t[f1 - 1])
        if _overlaps(t0 - 0.3, t1, [(a, b + cfg.flight_after_shot_s) for a, b in shot_win]) or                 _overlaps(t0, t1, ref_dead):
            continue
        a = next((w for w in (who(f0 - k) for k in range(1, cfg.flight_search + 1)) if w is not None), None)
        b = next((w for w in (who(f1 - 1 + k) for k in range(1, cfg.flight_search + 1)) if w is not None), None)
        diag = {"t": round(t0, 2), "flight": [round(t0, 2), round(t1, 2)], "P": a and a[0], "Q": b and b[0],
                "source": "flight", "air": n_air}
        doc["candidates"].append(diag)
        if a is None:
            diag["reject"] = "no thrower"
            continue
        P, ia = a
        tP = team_of.get(P)
        if b is None:
            nxt = [x for x in oob_t if t1 <= x <= t1 + cfg.oob_window_s]
            other = poss_team_at(t1 + cfg.oob_window_s)
            if nxt and other is not None and other != tP:
                # just caught from a pass and gone again: the pass itself was bad (overthrown / not catchable)
                prev = next((x for x in reversed(doc["passes"]) if x["receiver"] == P and x.get("completed")
                             and t0 - cfg.oob_after_catch_s <= x["t_catch"] <= t0 + 0.05), None)
                if prev is not None:
                    prev["completed"] = False
                    prev["outcome"] = "out of bounds"
                    drop = {e_["id"] for e_ in doc_events if e_["type"] == "pass"
                            and abs(e_["t"] - prev["t_release"]) < 0.06 and e_["details"].get("receiver_track") == P}
                    doc_events[:] = [e_ for e_ in doc_events if e_["id"] not in drop]
                    P, ia = prev["passer"], gi.idx(prev["t_release"])
                    tP = team_of.get(P)
                pP = _xy(gi, P, ia)
                conf = 0.45 + 0.1 * min(1.0, n_air / 3)
                doc["turnovers"].append({
                    "t": round(t0, 2), "kind": "bad_pass_out_of_bounds", "team_lost": tP, "lost_by": P,
                    "gained_by": None, "lost_by_name": _name(gi, P), "gained_by_name": None,
                    "from": None if pP is None else [round(float(v), 2) for v in pP], "position": None,
                    "air_s": round(t1 - t0, 2), "confidence": round(conf, 2)})
                add("bad_pass", ia, tP, conf, f"Bad pass (out of bounds) by {_name(gi, P)}",
                    f"(meant for {_name(gi, prev['receiver'])})" if prev is not None else "",
                    passer_track=P, kind="out_of_bounds", flight=[round(t0, 2), round(t1, 2)])
                diag["accept"] = "bad pass out of bounds"
            else:
                diag["reject"] = "no catcher"
            continue
        Q, ib = b
        if P == Q or team_of.get(Q) != tP:
            diag["reject"] = "same player" if P == Q else "other team (holder-run logic decides)"
            continue
        t_rel = float(gi.t[ia])
        if any(abs(t_rel - h) <= cfg.flight_dup_s for h in have):
            diag["reject"] = "already a pass"
            continue
        pP, pQ = _xy(gi, P, ia), _xy(gi, Q, ib)
        if pP is None or pQ is None:
            diag["reject"] = "no court position"
            continue
        dist = float(np.linalg.norm(pQ - pP))
        bP, bQ = boxes.get(P, {}), boxes.get(Q, {})
        iou = max((_iou(bP.get(i), bQ.get(j)) for i in range(ia - 2, ib + 3) for j in range(i - 2, i + 3)),
                  default=0.0)
        diag.update(dist=round(dist, 2), iou=round(iou, 2))
        if iou >= cfg.dup_iou or dist < cfg.min_pass_m or dist > cfg.max_pass_m:
            diag["reject"] = "duplicate boxes" if iou >= cfg.dup_iou else "distance"
            continue
        # a dribble bounce also leaves the box fast: a pass is released at chest height or higher
        ks = sorted(bP, key=lambda k: abs(k - f0))
        if ks:
            bx = bP[ks[0]]
            hgt = max(1.0, bx[3] - bx[1])
            rel = [(img[i, 1] - bx[1]) / hgt for i in range(f0, f1) if obs[i]]
            diag["rel_y_min"] = round(float(min(rel)), 2) if rel else None
            if rel and min(rel) > cfg.flight_max_rel_y:
                diag["reject"] = "low ball (dribble)"
                continue
        conf = 0.45 + 0.1 * min(1.0, n_air / 3) + (0.1 if owner[ia] == P else 0.0) + \
            (0.1 if owner[ib] == Q else 0.0) + 0.1 * min(1.0, dist / 5.0)
        if conf < cfg.min_event_conf:
            diag["reject"] = "low confidence"
            continue
        shape = flight_shape(replay, ia, ib)
        ptype = shape["type"] if shape["type"] != "unseen" else "direct"
        t_catch = float(gi.t[ib])
        doc["passes"].append({"t_release": round(t_rel, 2), "t_catch": round(t_catch, 2), "team": tP, "passer": P,
                              "receiver": Q, "passer_name": _name(gi, P), "receiver_name": _name(gi, Q),
                              "from": [round(float(v), 2) for v in pP], "to": [round(float(v), 2) for v in pQ],
                              "length_m": round(dist, 1), "air_s": round(t_catch - t_rel, 2), "type": ptype,
                              "completed": True, "ball_seen_frames": n_air, "shape": shape,
                              "confidence": round(conf, 2), "source": "flight"})
        have.append(t_rel)
        add("pass", ia, tP, conf, f"Pass {_name(gi, P)} -> {_name(gi, Q)}",
            f"({dist:.0f} m{', ' + ptype if ptype not in ('unseen', 'direct') else ''})",
            passer_track=P, receiver_track=Q, length_m=round(dist, 1), pass_type=ptype, t_catch=round(t_catch, 2),
            detected_by="flight")
        diag["accept"] = "pass"


def _boxes(replay: dict) -> dict[str, dict[int, list]]:
    """track id -> {frame index: image box} (measured boxes only)."""
    out: dict[str, dict[int, list]] = {}
    for tr in replay.get("tracks") or []:
        i0 = int(tr.get("i0") or 0)
        out[str(tr.get("id"))] = {i0 + j: b for j, b in enumerate(tr.get("b") or []) if b}
    return out


def _iou(a: Any, b: Any) -> float:
    if not a or not b:
        return 0.0
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / ua) if ua > 0 else 0.0


def _motion(gi: Any, lo: np.ndarray, hi: np.ndarray, i0: int, i1: int, pad_s: float = 1.5) -> float | None:
    """Median speed (m/s) of the visible players (no referees) in the ``pad_s`` before frame i0 up to i1 (before
    a live steal the players move; before a dead-ball hand-over they stand)."""
    a = max(0, gi.idx(float(gi.t[i0]) - pad_s))
    b = min(gi.n - 1, i1)
    sl = slice(int(lo[a]), int(hi[b]))
    v = gi.r_speed[sl][(gi.r_team[sl] != "R") & np.isfinite(gi.r_speed[sl])]
    return float(np.median(v)) if len(v) >= 4 else None


def flight_shape(replay: dict, i0: int, i1: int) -> dict:
    """Image-space shape of the ball between frames i0..i1 (exclusive ends: the holders' frames) from the
    stored ball boxes: number of sightings, apex above the start (in ball diameters, + = up), lowest point
    below the straight line (bounce), and the type guess bounce / lob / direct / unseen."""
    bb = ((replay.get("ball") or {}).get("b") or [])
    pts = []
    for i in range(max(0, i0 + 1), min(len(bb), i1)):
        b = bb[i]
        if b:
            x1, y1, x2, y2 = (float(v) for v in b[:4])
            pts.append((i, (x1 + x2) / 2, (y1 + y2) / 2, max(2.0, 0.5 * ((x2 - x1) + (y2 - y1)))))
    out = {"seen": len(pts), "type": "unseen"}
    if len(pts) < 3:
        return out
    a = np.array(pts, float)
    d = float(np.median(a[:, 3]))
    # straight line between the first and last sighting (the camera pans slowly compared with a pass)
    u = (a[:, 0] - a[0, 0]) / max(1.0, a[-1, 0] - a[0, 0])
    line_y = a[0, 2] + (a[-1, 2] - a[0, 2]) * u
    dev = (a[:, 2] - line_y) / d  # + = below the line (image y down)
    apex = float((a[0, 2] - a[:, 2].min()) / d)
    out.update({"apex_diam": round(apex, 1), "max_below_line_diam": round(float(dev.max()), 1),
                "max_above_line_diam": round(float(-dev.min()), 1)})
    inner = dev[1:-1] if len(dev) > 2 else dev
    if len(inner) and inner.max() > 2.5:
        out["type"] = "bounce"
    elif -dev.min() > 3.0:
        out["type"] = "lob"
    else:
        out["type"] = "direct"
    return out


# ====================================================================================== analysis
def analyse(replay: dict, gs: dict | None, cfg: BEConfig | None = None,
            motion_chain: tuple[np.ndarray, np.ndarray] | None = None) -> dict:
    """``motion_chain`` = (G (n, 3, 3), seg (n,)) camera-motion chain of pass 1 (see :func:`save_motion`), for
    ball speeds without the camera pan; without it image speeds include the pan."""
    from replay.game_state import load_input
    cfg = cfg or BEConfig()
    gi = load_input(replay)
    doc: dict = {"version": VERSION, "states": [], "passes": [], "shots": [], "turnovers": [], "events": [], "candidates": [],
                 "stats": {}, "config": asdict(cfg)}
    if gi.n == 0:
        return doc
    gs = gs or {}
    names = {k: str((gi.teams.get(k) or {}).get("name") or k) for k in TEAMS}
    dead = [(float(d["t0"]), float(d["t1"])) for d in gs.get("dead_balls") or [] if d.get("kind") != "referee_has_ball"]
    ref_dead = [(float(d["t0"]), float(d["t1"])) for d in gs.get("dead_balls") or []]
    shots_gs = list(gs.get("shots") or [])
    shot_win = [(float(s["t_release"]) - cfg.shot_guard_s, float(s["t_rim"]) + 0.5) for s in shots_gs]
    poss = list(gs.get("possessions") or [])
    oob_t = [float(e["t"]) for e in gs.get("events") or [] if e.get("type") == "inbound"
             or (e.get("type") == "turnover" and str((e.get("details") or {}).get("kind", "")).startswith("out"))]
    runs = holder_runs(gi, cfg)
    boxes = _boxes(replay)
    boxes_at: dict[int, list] = {}
    for per in boxes.values():
        for i, bx in per.items():
            boxes_at.setdefault(i, []).append(bx)
    from replay.game_state import _frame_slices
    lo, hi = _frame_slices(gi)
    n_ft = sum(1 for s in shots_gs if s.get("free_throw"))

    # ---- per-frame state
    state = np.array(["unknown"] * gi.n, dtype=object)
    owner: list[str | None] = [None] * gi.n
    for r in runs:
        for i in range(r["i0"], r["i1"] + 1):
            state[i], owner[i] = "held", r["owner"]
    for a, b in zip(runs, runs[1:]):
        if a["owner"] != b["owner"] and gi.t[b["i0"]] - gi.t[a["i1"]] <= cfg.max_pass_s:
            state[a["i1"] + 1:b["i0"]] = "air"
    for s in shots_gs:
        i0, i1 = gi.idx(float(s["t_release"])) + 1, gi.idx(float(s["t_rim"]))
        state[i0:i1 + 1] = "air"
    # image evidence (ball box vs the holder's / everybody's boxes, camera-motion-compensated ball speed)
    img = _ball_image(replay, gi.n)
    spd = ball_speed_mps(img, gi.t, motion_chain)
    for i in range(gi.n):
        if not np.isfinite(img[i, 0]):
            continue
        cx, cy, dia = img[i]
        if state[i] == "held":
            bx = boxes.get(owner[i], {}).get(i)
            if bx and cy > bx[1] + cfg.dribble_rel_y * (bx[3] - bx[1]):
                state[i] = "dribble"
            continue
        if state[i] != "unknown":
            continue
        inside = any(bx[0] <= cx <= bx[2] and bx[1] <= cy <= bx[3] for bx in boxes_at.get(i, ()))
        if not inside and np.isfinite(spd[i]) and spd[i] >= cfg.air_mps:
            state[i] = "air"
        elif not inside and np.isfinite(spd[i]):
            state[i] = "floor"  # slow and away from everybody: on the floor / rolling
    loose = (state == "unknown") & np.isfinite(gi.ball_xy[:, 0])
    state[loose] = "loose"
    for a, b in ref_dead:
        state[gi.idx(a):gi.idx(b, "right")] = "dead"
    for i in range(gi.n):
        if not gi.calib[i]:
            state[i] = "unknown"
    st_runs: list[list] = []
    for i in range(gi.n):
        key = (state[i], owner[i] if state[i] in ("held", "dribble") else None)
        if st_runs and (st_runs[-1][2], st_runs[-1][3]) == key and st_runs[-1][4] == i - 1:
            st_runs[-1][1] = round(float(gi.t[i]), 2)
            st_runs[-1][4] = i
        else:
            st_runs.append([round(float(gi.t[i]), 2), round(float(gi.t[i]), 2), key[0], key[1], i])
    doc["states"] = [r[:4] for r in st_runs]

    events: list[dict] = []

    def add(typ: str, i: int, team: str | None, conf: float, label: str, suffix: str = "", **details: Any) -> dict:
        ev = {"id": f"be-{typ}-{int(gi.frame[i])}", "type": typ, "t": round(float(gi.t[i]), 3),
              "frame": int(gi.frame[i]), "duration": 0.0, "team": team, "basket": details.pop("basket", None),
              "confidence": round(float(np.clip(conf, 0, 1)), 2), "source": SOURCE, "label": label, "suffix": suffix,
              "details": {k: v for k, v in details.items() if v is not None}}
        from replay.game_state import event_title
        ev["title"] = event_title(ev, names)
        events.append(ev)
        return ev

    def poss_team_at(t: float) -> str | None:
        p = next((p for p in poss if p["t0"] - 0.3 <= t <= p["t1"] + 0.3), None)
        return None if p is None else p.get("team")

    # ---- passes, bad passes, steals
    for a, b in zip(runs, runs[1:]):
        P, Q = a["owner"], b["owner"]
        if P == Q or a["team"] not in TEAMS or b["team"] not in TEAMS:
            continue
        t_rel, t_catch = float(gi.t[a["i1"]]), float(gi.t[b["i0"]])
        gap = t_catch - t_rel
        if gap > cfg.max_pass_s or _overlaps(t_rel, t_catch, ref_dead) or _overlaps(t_rel + 0.05, t_catch, shot_win):
            continue
        pP, pQ = _xy(gi, P, a["i1"]), _xy(gi, Q, b["i0"])
        if pP is None or pQ is None:
            continue
        dist = float(np.linalg.norm(pQ - pP))
        # id switch: P's track ends around the catch and Q starts where P was
        p0, p1 = _track_span(gi, P)
        q0, _ = _track_span(gi, Q)
        if dist <= cfg.id_switch_m * 2 and p1 <= b["i0"] + 2 and q0 >= a["i1"] - 2:
            continue
        len_a = gi.t[a["i1"]] - gi.t[a["i0"]]
        len_b = gi.t[b["i1"]] - gi.t[b["i0"]]
        run_q = min(1.0, 0.5 + 0.5 * min(a["k"], b["k"]) / 4)
        seen = int(np.sum(gi.ball_img[a["i1"] + 1:b["i0"]]))
        shape = flight_shape(replay, a["i1"], b["i0"])
        bP, bQ = boxes.get(P, {}), boxes.get(Q, {})
        iou = max((_iou(bP.get(i), bQ.get(j)) for i in range(a["i1"] - 2, b["i0"] + 3)
                   for j in range(i - 2, i + 3)), default=0.0)
        mot = _motion(gi, lo, hi, a["i1"], b["i0"])
        diag = {"t": round(t_rel, 2), "P": P, "Q": Q, "same_team": a["team"] == b["team"], "dist": round(dist, 2),
                "gap": round(gap, 2), "ka": a["k"], "kb": b["k"], "seen": seen, "iou": round(iou, 2),
                "motion": None if mot is None else round(mot, 2), "shape": shape["type"]}
        doc["candidates"].append(diag)
        if iou >= (cfg.dup_iou if a["team"] == b["team"] else cfg.dup_iou_other_team):
            diag["reject"] = "duplicate boxes"
            continue  # two tracks on one person (the ball "moves" between them without leaving the hands)
        still = mot is not None and mot < cfg.still_mps
        n_air = int(np.sum(state[a["i1"] + 1:b["i0"]] == "air"))
        diag["air_frames"] = n_air
        need = cfg.min_pass_run_frames - (1 if n_air >= cfg.flight_air_frames else 0)
        if min(a["k"], b["k"]) < need:
            diag["reject"] = "short holder run"
            continue  # one- or two-frame "possessions" are mostly holder mix-ups (overlapping players)
        if a["team"] == b["team"]:
            if dist < cfg.min_pass_m or dist > cfg.max_pass_m:
                continue
            speed = dist / max(gap, 0.1)
            conf = 0.45 * run_q + 0.15 * min(1.0, dist / 5.0) + (0.15 if seen else 0.0) + \
                (0.1 if speed <= 15.0 else -0.15) + (0.05 if len_a > 0.4 and len_b > 0.4 else 0.0)
            if conf < cfg.min_event_conf:
                diag["reject"] = "low confidence"
                continue
            ptype = shape["type"] if shape["type"] != "unseen" else ("long" if dist > 12 else "unseen")
            rec = {"t_release": round(t_rel, 2), "t_catch": round(t_catch, 2), "team": a["team"], "passer": P,
                   "receiver": Q, "passer_name": _name(gi, P), "receiver_name": _name(gi, Q),
                   "from": [round(float(v), 2) for v in pP], "to": [round(float(v), 2) for v in pQ],
                   "length_m": round(dist, 1), "air_s": round(gap, 2), "type": ptype, "completed": True,
                   "ball_seen_frames": seen, "shape": shape, "confidence": round(conf, 2)}
            doc["passes"].append(rec)
            add("pass", a["i1"], a["team"], conf, f"Pass {_name(gi, P)} -> {_name(gi, Q)}",
                f"({dist:.0f} m{', ' + ptype if ptype not in ('unseen', 'direct') else ''})",
                passer_track=P, receiver_track=Q, length_m=round(dist, 1), pass_type=ptype, t_catch=round(t_catch, 2))
            continue
        # other team gained the ball: confirmed by the game state's possessions (blips are not turnovers)
        new_team = poss_team_at(t_catch + cfg.change_confirm_s)
        old_team = poss_team_at(t_rel - 0.3)
        if poss and (new_team != b["team"] or old_team != a["team"]):
            continue
        if len_b < 0.3 and b["k"] < 3:
            continue
        if still:
            diag["reject"] = "players still (dead ball)"
            continue  # ball handed over in a dead ball (after a whistle / for an inbound)
        if any(t_rel - 0.5 <= x <= t_catch + 1.5 for x in oob_t):
            continue  # an inbound / out-of-bounds call right there: the ball went out, not a live steal
        conf = 0.5 * run_q + 0.2 + (0.1 if seen else 0.0) + (0.1 if len_a > 0.5 else 0.0)
        contact = dist <= cfg.steal_max_m or iou >= cfg.contact_iou or (seen == 0 and gap <= 0.6)
        if contact and gap <= cfg.steal_max_gap_s:  # taken from the handler, no flight seen
            kind = "steal"
            rec = {"t": round(t_catch, 2), "kind": "stolen_from_handler", "team_lost": a["team"], "lost_by": P,
                   "gained_by": Q, "lost_by_name": _name(gi, P), "gained_by_name": _name(gi, Q),
                   "position": [round(float(v), 2) for v in pQ], "distance_m": round(dist, 1),
                   "confidence": round(conf, 2)}
            doc["turnovers"].append(rec)
            add("steal", b["i0"], b["team"], conf, f"Steal by {_name(gi, Q)}", f"(from {_name(gi, P)})",
                stealer_track=Q, lost_by_track=P, lost_by=a["team"], distance_m=round(dist, 1), kind=kind)
            add("lost_ball", a["i1"], a["team"], conf * 0.9, f"Lost ball (stolen) by {_name(gi, P)}",
                player_track=P, stealer_track=Q)
        else:
            if dist < cfg.min_pass_m:
                continue  # unclear scramble
            rec = {"t": round(t_rel, 2), "kind": "bad_pass_intercepted", "team_lost": a["team"], "lost_by": P,
                   "gained_by": Q, "lost_by_name": _name(gi, P), "gained_by_name": _name(gi, Q),
                   "from": [round(float(v), 2) for v in pP], "position": [round(float(v), 2) for v in pQ],
                   "distance_m": round(dist, 1), "air_s": round(gap, 2), "shape": shape,
                   "confidence": round(conf * 0.9, 2)}
            doc["turnovers"].append(rec)
            add("bad_pass", a["i1"], a["team"], conf * 0.9, f"Bad pass (stolen) by {_name(gi, P)}",
                f"(to {_name(gi, Q)})", passer_track=P, stealer_track=Q, length_m=round(dist, 1))
            add("steal", b["i0"], b["team"], conf * 0.9, f"Steal by {_name(gi, Q)}", "(interception)",
                stealer_track=Q, lost_by_track=P, lost_by=a["team"], kind="interception")

    # ---- passes / bad passes out of bounds from ball flights seen in the image (holders missed by ball.py)
    if cfg.use_flights:
        _flight_events(gi, cfg, img, spd, boxes, owner, shot_win, ref_dead, oob_t, poss_team_at, doc, add, replay,
                       events)

    # ---- shots (enrich the game-state attempts)
    for s in shots_gs:
        if s.get("free_throw"):
            continue
        t_rel = float(s["t_release"])
        if any(x["basket"] == s.get("basket") and abs(x["t_release"] - t_rel) < 0.3 for x in doc["shots"]):
            continue  # the same release twice (two rim contacts of one attempt)
        i_rel = min(gi.n - 1, gi.idx(t_rel))
        i_rim = min(gi.n - 1, gi.idx(float(s["t_rim"])))
        shape = flight_shape(replay, i_rel, i_rim)
        pos = s.get("pos") or None
        if pos is None and s.get("shooter_track") is not None:
            p = _xy(gi, str(s["shooter_track"]), i_rel)
            pos = None if p is None else [round(float(v), 2) for v in p]
        dist = s.get("distance_m")
        outcome = "made" if s.get("made") else "missed" if s.get("missed") else "unknown"
        rebound = None
        if outcome == "missed":
            rebound = poss_team_at(float(s["t_rim"]) + 1.5)
        doc["shots"].append({"t_release": round(t_rel, 2), "t_rim": s["t_rim"], "basket": s.get("basket"),
                             "team": s.get("team"), "shooter": s.get("shooter_track"),
                             "shooter_name": _name(gi, s.get("shooter_track")) if s.get("shooter_track") else None,
                             "position": pos, "distance_m": dist, "value_guess": s.get("value_guess"),
                             "outcome": outcome, "rebound_team": rebound, "arc": shape,
                             "confidence": s.get("confidence")})
    doc["stats"] = {"holder_runs": len(runs), "passes": len(doc["passes"]),
                    "bad_passes": sum(1 for x in doc["turnovers"] if x["kind"].startswith("bad_pass")),
                    "steals": sum(1 for e in events if e["type"] == "steal"),
                    "shots": len(doc["shots"]), "free_throw_attempts": n_ft,
                    "frames": {k: int(np.sum(state == k)) for k in STATES}}
    events.sort(key=lambda e: e["t"])
    doc["events"] = events
    return doc


def _shot_label(sh: dict) -> str:
    out = {"made": "made", "missed": "missed"}.get(sh.get("outcome"), "")
    lab = "Shot" + (f" ({out})" if out else "")
    if sh.get("distance_m") is not None:
        lab += f" from {float(sh['distance_m']):.1f} m"
    if sh.get("shooter_name"):
        lab += f" by {sh['shooter_name']}"
    return lab


def attach(gs_doc: dict, replay: dict, cfg: BEConfig | None = None, results_dir: Any = None) -> dict:
    """Run :func:`analyse` and merge into a game-state document (in place): ``gs_doc["ball_events"]``, events
    added to ``gs_doc["events"]``; the game state's own steal / turnover events at the same moment are replaced
    and its shot events get the richer title. Never raises."""
    try:
        n = len(replay.get("t") or [])
        chain = load_motion(results_dir, n) if results_dir is not None else None
        be = analyse(replay, gs_doc, cfg, chain)
        be["camera_compensated"] = chain is not None
    except Exception as e:  # noqa: BLE001 - never break the game state on a problem here
        print("ball events failed:", type(e).__name__, e)
        gs_doc["ball_events"] = {"version": VERSION, "error": f"{type(e).__name__}: {e}"}
        return gs_doc
    evs = list(gs_doc.get("events") or [])
    mine = be["events"]
    to_t = [e["t"] for e in mine if e["type"] in ("steal", "bad_pass", "lost_ball")]
    replaced: list = []
    demoted: list = []
    cfg_ = cfg or BEConfig()
    keep = []
    for e in evs:
        t = _f(e.get("t"))
        if e.get("type") in ("steal", "turnover") and t is not None and any(abs(t - x) <= 1.5 for x in to_t):
            replaced.append(e.get("id"))
            continue
        if e.get("type") == "steal" and t is not None:
            # a possession-level steal without a ball-level hand-over to the other team: in the checked clips
            # these were mostly team mix-ups / passes -> keep, but less sure
            e = {**e, "confidence": round(float(_f(e.get("confidence")) or 0.0) * cfg_.unconfirmed_steal_factor, 2),
                 "suffix": "(not confirmed by the ball track)",
                 "details": {**(e.get("details") or {}), "ball_events": "unconfirmed"}}
            from replay.game_state import event_title
            e["title"] = event_title(e, gs_doc.get("teams") or {})
            demoted.append(e.get("id"))
        if e.get("type") == "shot" and t is not None:
            if any(k.get("type") == "shot" and k.get("basket") == e.get("basket") and abs((_f(k.get("t")) or -9) - t) < 0.3
                   for k in keep):
                replaced.append(e.get("id"))
                continue  # duplicate attempt (same release, second rim contact)
            sh = next((s for s in be["shots"] if abs(s["t_release"] - t) <= 0.15), None)
            if sh is not None:
                e = {**e, "label": _shot_label(sh), "details": {**(e.get("details") or {}), "outcome": sh["outcome"],
                                                                  "arc": sh["arc"], "rebound_team": sh["rebound_team"]}}
                from replay.game_state import event_title
                names = gs_doc.get("teams") or {}
                e["title"] = event_title(e, names)
        keep.append(e)
    be["replaced_game_state_events"] = replaced
    be["demoted_game_state_events"] = demoted
    gs_doc["events"] = sorted(keep + mine, key=lambda e: _f(e.get("t")) or 0.0)
    gs_doc["ball_events"] = be
    if MAKE_CHECK:
        try:
            check_makes(gs_doc, replay)
        except Exception as e:  # noqa: BLE001
            print("make check failed:", type(e).__name__, e)
    if SHEET_ALIGN and results_dir is not None:
        try:
            reconcile_with_sheet(gs_doc, replay, Path(results_dir))
        except Exception as e:  # noqa: BLE001 - never break the game state
            print("scoresheet alignment failed:", type(e).__name__, e)
    if SCOREBAR_ALIGN and results_dir is not None and not gs_doc.get("sheet_alignment"):
        try:
            from replay import scorebar as sbm
            reconcile_with_scorebar(gs_doc, replay, sbm.load_result(Path(results_dir) / "work"))
        except Exception as e:  # noqa: BLE001
            print("score-bar alignment failed:", type(e).__name__, e)
    return gs_doc


SCOREBAR_ALIGN = True  # without a scoresheet: confirm detected makes by the broadcast score bar's changes
BAR_WINDOW_S = (1.0, 11.0)  # the bar changes this long after the ball went through the rim
BAR_MIN_CONFIRM = 0.4  # at least this share of the detected makes must sit before a bar change (else bar unreliable)


def bar_events(sb: dict | None) -> list[dict]:
    """Score-bar timeline [(t, s0, s1)] -> scoring events [{t, side (0/1), points}] (+1..+3 on one side)."""
    tl = (sb or {}).get("timeline") or []
    out = []
    for (t0, a0, b0), (t1, a1, b1) in zip(tl, tl[1:]):
        if None in (a0, b0, a1, b1):
            continue
        da, db = a1 - a0, b1 - b0
        if (da > 0) == (db > 0) or max(da, db) > 3 or min(da, db) < 0:
            continue
        out.append({"t": float(t1), "side": 0 if da > 0 else 1, "points": int(max(da, db))})
    return out


def match_bar(evs: list[dict], bk: list[dict]) -> dict[int, int]:
    """Greedy matching of detected makes ``bk`` (sorted by t) to bar events ``evs``: {j: k} where the bar changed
    ``BAR_WINDOW_S`` after the detection, closest to 4 s first."""
    lo, hi = BAR_WINDOW_S
    pairs = sorted((abs(b["t"] - float(e["t"]) - 4.0), j, k) for j, e in enumerate(bk) for k, b in enumerate(evs)
                   if lo <= b["t"] - float(e["t"]) <= hi)
    used_d, used_b, m = set(), set(), {}
    for _, j, k in pairs:
        if j not in used_d and k not in used_b:
            used_d.add(j), used_b.add(k)
            m[j] = k
    return m


def bar_side_team(evs: list[dict], bk: list[dict], m: dict[int, int], team_of: Any) -> dict[int, str]:
    """Bar side (0/1) -> video team: majority of the matched makes' detected teams (``team_of(basket event)``)."""
    votes: dict[tuple[int, str], int] = {}
    for j, k in m.items():
        team = team_of(bk[j])
        if team in ("A", "B"):
            votes[(evs[k]["side"], team)] = votes.get((evs[k]["side"], team), 0) + 1
    side0 = "A" if votes.get((0, "A"), 0) + votes.get((1, "B"), 0) >= votes.get((0, "B"), 0) + votes.get((1, "A"), 0) else "B"
    return {0: side0, 1: "B" if side0 == "A" else "A"}


def bar_map_votes(evs: list[dict], bk: list[dict], m: dict[int, int], team_of: Any) -> int:
    """How many matched makes had a detected team (the side->team map is unsure below 2)."""
    return sum(1 for j in m if team_of(bk[j]) in ("A", "B"))


def reconcile_with_scorebar(gs_doc: dict, replay: dict, sb: dict | None) -> dict | None:
    """Like :func:`reconcile_with_sheet`, with the score bar's changes (known times) instead of the running score:
    a detected make 1-11 s before a change of the bar is confirmed (team = the side that scored, points = the
    step); detected makes before no change are marked ``not_on_sheet`` (0 points, "not on the score bar?"); bar
    changes without a detected make are listed in ``missing`` with the time ~4 s before the change and the likely
    shooter (the last shot attempt at that team's basket). Skipped when the bar does not explain at least
    ``BAR_MIN_CONFIRM`` of the detected makes (a misread bar)."""
    evs = bar_events(sb)
    bk = sorted((e for e in replay.get("events") or [] if e.get("type") == "basket" and e.get("id") is not None),
                key=lambda e: float(e.get("t") or 0.0))
    if not evs or not bk:
        return None
    ann = gs_doc.setdefault("basket_annotations", {})
    lo, hi = BAR_WINDOW_S
    m = match_bar(evs, bk)
    used_b = set(m.values())
    if len(m) < BAR_MIN_CONFIRM * len(bk):
        gs_doc["scorebar_alignment"] = {"used": False, "confirmed": len(m), "detections": len(bk),
                                        "reason": "the score bar explains too few detected makes (misread bar?)"}
        return None
    # bar side -> video team: majority of the confirmed makes' detected teams
    side_team = bar_side_team(evs, bk, m, lambda e: (ann.get(str(e["id"])) or {}).get("team") or e.get("team"))
    for j, e in enumerate(bk):
        a = dict(ann.get(str(e["id"])) or {})
        if j in m:
            b = evs[m[j]]
            kind = "free_throw" if b["points"] == 1 else "field_goal"
            lab = {1: "Made free throw", 2: "Made 2", 3: "Made 3"}[b["points"]]
            a.pop("counted", None)
            a.update({"label": lab, "points_guess": b["points"], "kind": kind, "team": side_team[b["side"]],
                      "team_source": "score bar", "on_sheet": True, "title": lab})
        else:
            a.update({"label": "Not on the score bar?", "points_guess": 0, "kind": "not_on_sheet", "on_sheet": False,
                      "title": f"Basket not on the score bar? ({e.get('basket') or ''})"})
        ann[str(e["id"])] = a
    shots = [s for s in gs_doc.get("shots") or [] if not s.get("free_throw")]
    missing = []
    for k, b in enumerate(evs):
        if k in used_b:
            continue
        team = side_team[b["side"]]
        cand = [s for s in shots if s.get("team") == team and b["t"] - hi <= float(s.get("t_rim") or -1e9) <= b["t"] - lo]
        sh = max(cand, key=lambda s: float(s["t_rim"]), default=None) if b["points"] > 1 else None
        missing.append({"seq": f"bar{k}", "team": team, "points": b["points"],
                        "kind": "ft" if b["points"] == 1 else f"fg{b['points']}",
                        "t_estimated": round((float(sh["t_rim"]) if sh else b["t"] - 4.0), 1),
                        "scorer_number": None, "shooter_track": (sh or {}).get("shooter_track"),
                        "label": f"Scored (from score bar, +{b['points']})"})
    doc = {"source": "scorebar", "aligned": len(m), "detections": len(bk), "sheet_events": len(evs),
           "not_on_sheet": len(bk) - len(m), "missing": missing, "side_team": side_team,
           "note": "Detected makes confirmed by the broadcast score bar (bar changes 1-11 s after the make); bar "
                   "changes without a detected make are added with an estimated time."}
    gs_doc["sheet_alignment"] = doc
    gs_doc["scorebar_alignment"] = {"used": True, "confirmed": len(m), "detections": len(bk), "bar_events": len(evs)}
    return doc


MAKE_CHECK = False  # (OFF: see check_makes trade-off) field-goal "makes" after which the play does not move away from the basket: not counted
MOVE_AWAY_M = 1.0  # mean player distance from that basket must grow by this much from [-2, 0] s to [6, 10] s


def _mean_player_x(replay: dict) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(replay.get("t") or [], float)
    s = np.zeros(len(t))
    c = np.zeros(len(t))
    for tr in replay.get("tracks") or []:
        if tr.get("type") != "player":
            continue
        i0, fs = int(tr.get("i0") or 0), tr.get("f") or []
        for j, x in enumerate(tr.get("x") or []):
            i = i0 + j
            if x is None or i >= len(t) or (fs[j] if j < len(fs) else 1) != 1:
                continue
            s[i] += x / 100.0
            c[i] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        return t, np.where(c >= 3, s / np.maximum(c, 1), np.nan)


def play_move_m(t: np.ndarray, mx: np.ndarray, t_basket: float, basket_x: float) -> float | None:
    """How much further from the basket the seen players are 6-10 s after it than just before (m)."""
    def mean_d(a: float, b: float) -> float | None:
        i0, i1 = np.searchsorted(t, t_basket + a), np.searchsorted(t, t_basket + b)
        v = mx[i0:i1]
        v = v[np.isfinite(v)]
        return float(np.mean(np.abs(v - basket_x))) if len(v) else None
    a, b = mean_d(-2.0, 0.0), mean_d(6.0, 10.0)
    return None if a is None or b is None else b - a


def check_makes(gs_doc: dict, replay: dict) -> list[str]:
    """A made field goal is followed by the other team bringing the ball up the court: the players move away from
    that basket. Detected "makes" after which the play stays at the basket (missed shot, rebound, tip) are marked
    ``counted: False`` in ``basket_annotations`` (stats leave them out of the points and list them as hints). Free
    throws and baskets right before a break are not checked. Measured on the full game (scoresheet + score bar):
    9 of 27 false makes removed for 3 of 43 true ones."""
    from replay import court
    t, mx = _mean_player_x(replay)
    if not len(t):
        return []
    L = float((replay.get("court") or {}).get("length") or court.LENGTH)
    breaks = [(float(b["t0"]), float(b["t1"])) for b in gs_doc.get("breaks") or []]
    ann = gs_doc.setdefault("basket_annotations", {})
    out = []
    for e in replay.get("events") or []:
        if e.get("type") != "basket" or e.get("id") is None:
            continue
        a = ann.get(str(e["id"])) or {}
        if a.get("kind") == "free_throw" or e.get("basket") not in ("L", "R"):
            continue
        te = float(e["t"])
        if any(b0 - 12 <= te <= b1 for b0, b1 in breaks) or te > float(t[-1]) - 10:
            continue
        mv = play_move_m(t, mx, te, 0.0 if e["basket"] == "L" else L)
        if mv is None or mv >= MOVE_AWAY_M:
            continue
        a = dict(a)
        a.update({"counted": False, "play_move_m": round(mv, 1),
                  "doubt": "the play did not move away from the basket afterwards (missed shot / rebound?)"})
        ann[str(e["id"])] = a
        out.append(str(e["id"]))
    return out


SHEET_ALIGN = True  # with an attached scoresheet (Replay folder scoresheet.json): reconcile detected baskets


def reconcile_with_sheet(gs_doc: dict, replay: dict, results_dir: Path) -> dict | None:
    """Detected made baskets <-> the attached scoresheet's running score (replay/sheet_align.py). Aligned baskets
    get the official team, points and scorer number (``basket_annotations``); detections that are not on the sheet
    are marked ``not_on_sheet`` (0 points); sheet events without a detection are listed with an estimated time in
    ``gs_doc["sheet_alignment"]["missing"]``. Without a scoresheet nothing changes."""
    from replay import scoresheet as ssm, sheet_align as sa
    sheet = None
    for root in (results_dir.parent, results_dir):
        sheet = ssm.load(root)
        if sheet is not None:
            break
    if sheet is None:
        return None
    run = sheet.to_dict().get("running_score") or []
    t_all = replay.get("t") or [0.0]
    if not run or float(t_all[-1]) - float(t_all[0]) < ssm.FULL_GAME_MIN_S:
        return None  # only a part of the game: the sheet's running score cannot be aligned as a whole
    tmap = ssm.team_map(sheet, replay)  # sheet side -> video team label
    inv = {v: k for k, v in tmap.items()}
    ann = gs_doc.setdefault("basket_annotations", {})
    bk = sorted((e for e in replay.get("events") or [] if e.get("type") == "basket" and e.get("id") is not None),
                key=lambda e: float(e.get("t") or 0.0))
    dets = []
    for e in bk:
        a = ann.get(str(e["id"])) or {}
        team = a.get("team") or e.get("team")
        dets.append({"t": float(e["t"]), "team": inv.get(team), "kind": {"free_throw": "ft",
                                                                          "field_goal": "fg"}.get(a.get("kind"))})
    t_end = float((replay.get("t") or [0.0])[-1])
    segs = sa.segments(gs_doc.get("breaks") or [], t_end)
    al = sa.align(run, dets, segs)
    for j, e in enumerate(bk):
        k = al["det_to_sheet"][j]
        a = dict(ann.get(str(e["id"])) or {})
        if k is None:
            a.update({"label": "Not on the scoresheet?", "suffix": f"({e.get('basket') or ''})", "points_guess": 0,
                      "kind": "not_on_sheet", "on_sheet": False,
                      "title": f"Basket not on the scoresheet? ({e.get('basket') or ''})"})
        else:
            r = run[k]
            team = tmap.get(r["team"], r["team"])
            kind = "free_throw" if r.get("kind") == "ft" else "field_goal"
            lab = {"ft": "Made free throw", "fg2": "Made 2", "fg3": "Made 3"}.get(r.get("kind"), "Made basket")
            num = r.get("scorer")
            a.pop("counted", None)
            a.update({"label": lab, "suffix": f"#{num}" if num else "", "points_guess": int(r["points"]), "kind": kind,
                      "team": team, "team_source": "scoresheet", "scorer_number": num, "sheet_seq": r.get("seq"),
                      "on_sheet": True, "title": f"{lab} – #{num}" if num else lab})
        ann[str(e["id"])] = a
    est = sa.estimate_times(len(run), al["sheet_to_det"], [d["t"] for d in dets], [r["period"] for r in run], segs)
    missing = [{"seq": r.get("seq"), "team": tmap.get(r["team"], r["team"]), "points": r["points"],
                "scorer_number": r.get("scorer"), "kind": r.get("kind"), "period": r.get("period"),
                "t_estimated": round(est[k], 1)} for k, r in enumerate(run) if al["sheet_to_det"][k] is None]
    doc = {"aligned": sum(x is not None for x in al["det_to_sheet"]), "detections": len(bk),
           "sheet_events": len(run), "not_on_sheet": sum(x is None for x in al["det_to_sheet"]),
           "missing": missing, "segments": [[round(a, 1), round(b, 1), sorted(p)] for a, b, p in segs],
           "note": "Detected baskets matched to the official running score in order (period, team, free throw). "
                   "Team, points and scorer of matched baskets are the official ones; the match itself is an "
                   "estimate. Missing = on the sheet but not detected (time estimated)."}
    gs_doc["sheet_alignment"] = doc
    return doc


def is_current(doc: Any) -> bool:
    return isinstance(doc, dict) and isinstance(doc.get("ball_events"), dict) and \
        doc["ball_events"].get("version") == VERSION


# ====================================================================================== export / text
PASS_COLUMNS = ["t_release", "t_catch", "team", "passer", "receiver", "passer_name", "receiver_name", "from_x",
                "from_y", "to_x", "to_y", "length_m", "air_s", "type", "confidence"]
SHOT_COLUMNS = ["t_release", "t_rim", "team", "basket", "shooter", "shooter_name", "x", "y", "distance_m",
                "value_guess", "outcome", "rebound_team", "confidence"]


def tables(be: dict | None) -> dict[str, list[list]]:
    """passes.csv / shots.csv rows (header first) for the data export (shot chart: x, y in court metres)."""
    be = be or {}
    passes = [PASS_COLUMNS] + [[p["t_release"], p["t_catch"], p["team"], p["passer"], p["receiver"], p["passer_name"],
                                p["receiver_name"], *p["from"], *p["to"], p["length_m"], p["air_s"], p["type"],
                                p["confidence"]] for p in be.get("passes") or []]
    shots = [SHOT_COLUMNS] + [[s["t_release"], s["t_rim"], s.get("team"), s.get("basket"), s.get("shooter"),
                               s.get("shooter_name"), *((s.get("position") or [None, None])[:2]), s.get("distance_m"),
                               s.get("value_guess"), s.get("outcome"), s.get("rebound_team"), s.get("confidence")]
                              for s in be.get("shots") or []]
    return {"passes.csv": passes, "shots.csv": shots}


def summary_lines(be: dict | None, names: dict[str, str] | None = None) -> list[str]:
    """Short per-team totals for people / AI prompts."""
    be = be or {}
    names = names or {}
    out = []
    for team in TEAMS:
        p = [x for x in be.get("passes") or [] if x["team"] == team]
        bad = [x for x in be.get("turnovers") or [] if x["team_lost"] == team and x["kind"].startswith("bad_pass")]
        st = [x for x in be.get("turnovers") or [] if x["team_lost"] != team]
        sh = [x for x in be.get("shots") or [] if x.get("team") == team]
        made = sum(1 for x in sh if x["outcome"] == "made")
        out.append(f"- {names.get(team, team)} ({team}): {len(p)} completed passes detected, {len(bad)} bad passes "
                   f"(intercepted), {len(st)} steals/interceptions won, {len(sh)} field-goal attempts ({made} made "
                   "detected)")
    return out
