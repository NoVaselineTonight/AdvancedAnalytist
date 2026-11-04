"""Basketball game state and events from the 2D tracking (deterministic, no AI, no Qt).

Raw positions are not enough context for a person or a language model: this module turns the replay data
(``results/replay_data.json``: per-frame player tracks, ball position/state/owner, made baskets) plus the
preliminary analysis (``results/inplay.json``: play intervals and breaks) into basketball structure:

* **Possession** - which team controls the ball over time. Control runs come from the ball owner
  (``possessed`` ball frames); a run must last ``min_run_s``; very short runs of the other team in the
  middle of a team's control are overlap mistakes (blips) and are ignored (hysteresis). Possessions never
  span a dead ball (free throws, breaks, referee holding the ball). Each has a confidence from how much of it
  the holder was actually seen.
* **Attack direction** per half (teams switch baskets at half time; the half-time break is the longest break
  in ``inplay.json`` >= ``halftime_min_s``). Votes: free throws (strongest), made baskets, shot attempts,
  and half-court possessions (ball and most players in one half for several seconds). Halves are coupled:
  evidence from one half counts (mirrored) for the other.
* **Events** with time, team, confidence and a readable title (shown on the timeline's auto track): shot
  attempts (ball at a rim after a release), made baskets (annotated: free throw / 2 / 3?), offensive and
  defensive rebounds, turnovers (steal / out of bounds or violation / lost ball), free throws k/n (dead ball,
  one player at the free-throw line, others along the lane, ball to the rim - never a pass), inbounds, jump
  balls, time-outs / stoppages / quarter and half-time breaks, fast breaks and half-court sets.
* **Lineup** - 5 players per team are always on court in live play. Tracks are assigned to 5 slots per team
  (a new track continues the slot whose player was last seen nearest, given the time since; a jersey number read
  on the track - replay track_info "number" - joins the slot of that number and never one with another number);
  a slot whose player is not visible is "on court, position unknown, last seen at ...", never absent.

Inputs come only from files every analysed Replay has, so the engine also runs for Replays built before it
existed. Pass 2 (postprocess) adds ``rim_ball`` evidence (all ball candidates near the projected rims), which
finds more shot attempts than the ball track alone.

Output: :func:`analyse` -> dict (``advanced-analytist/game-state@1``), saved as ``results/game_state.json``.
:func:`merge_events` combines it with the detected baskets (and light-model events) for the timeline,
exports and the AI.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

FORMAT = "advanced-analytist/game-state@1"
ENGINE_VERSION = 5  # 5: half time relative, attacking-team credit, brief-spell merge, no TO after own shot
ENGINE_VERSION_V6 = 6  # 6: possession model v6 (boundaries + evidence vote; env REPLAY_POSSESSION_V6=1)
FILENAME = "game_state.json"


USER_WEIGHT = 20.0  # corrections ledger (corrections.py): a coach's make / anchor outweighs all detected evidence
USER_MAKE_MATCH_S = 2.0  # a coach's make replaces the detected make this close in time


def possession_v6_enabled() -> bool:
    """The possession model v6 (data/plans/possession_design.md) is the default since 4.11; REPLAY_POSSESSION_V6=0
    switches back to the holder-run model (engine 5)."""
    return os.environ.get("REPLAY_POSSESSION_V6", "1") == "1"


def engine_version() -> int:
    return ENGINE_VERSION_V6 if possession_v6_enabled() else ENGINE_VERSION
SOURCE = "game_state"
TEAMS = ("A", "B")
TIMELINE_MIN_CONF = 0.3  # game events below this confidence are not put on the timeline

# event types -> short timeline titles (team / basket are added by event_title)
TITLES = {
    "shot": "Shot", "free_throw": "Free throw", "free_throws": "Free throws", "off_rebound": "Off. rebound",
    "def_rebound": "Def. rebound", "rebound": "Rebound", "turnover": "Turnover", "steal": "Steal",
    "fast_break": "Fast break", "half_court": "Half-court set", "inbound": "Inbound", "jump_ball": "Jump ball",
    "timeout": "Time-out / stoppage", "quarter_break": "Quarter break", "half_time": "Half time",
    "stoppage": "Stoppage",
}


@dataclass
class GSConfig:
    hold_gap_s: float = 0.45  # unseen/loose frames inside one control run
    min_run_s: float = 0.25  # a control run must last this long (or have 3 holder frames)
    blip_s: float = 0.8  # shorter runs of the other team between two runs of one team are mistakes
    blip_gap_s: float = 1.5
    min_poss_s: float = 0.6
    ref_hold_min_s: float = 1.2
    sandwich_s: float = 3.0  # X, then Y shorter than this without a shot, then X again = mix-up
    sandwich_gap_s: float = 20.0  # ... when X's two spells are at most this far apart (< a shot clock; not across
    # a break or right after free throws). Longer unseen gaps may hide real changes of possession.
    after_shot_s: float = 8.0  # a change of possession this soon after the team's own shot is not a turnover
    steal_gap_s: float = 1.6  # control lost and won by the other team within this time in live play = steal
    dead_gap_s: float = 2.5  # nobody controls the ball this long + players still = dead ball
    rebound_window_s: float = 5.0
    rebound_max_dist_m: float = 5.5
    release_window_s: float = 3.0  # last holder within this time before the ball reaches the rim = shooter
    rim_group_s: float = 1.2
    max_ball_speed_mps: float = 14.0  # release -> rim faster than this is not a shot
    max_shot_dist_m: float = 13.0  # shooter further from the basket: not a shot (ball only near the rim in 2D)
    ft_dist: float = 5.8  # baseline -> free-throw line
    ft_spot_dx: tuple[float, float] = (-1.0, 0.9)  # shooter x offset from the FT line (+ = towards baseline)
    ft_spot_dy: float = 1.4
    ft_lane_y: tuple[float, float] = (1.5, 4.4)  # lane line players: |y - mid| range
    ft_lane_x: tuple[float, float] = (0.6, 5.4)  # ... and distance from the baseline
    ft_lane_outside_m: float = 0.6  # lane players from this far beyond the baseline (calibration error) to ft_lane_x[1]
    ft_min_lane: int = 2
    ft_still_mps: float = 1.2
    ft_min_s: float = 2.5
    ft_window_motion: float = 0.45  # median speed of the visible players during a free throw
    ft_merge_s: float = 14.0
    ft_pad_s: float = 1.5
    ft_setup_max_s: float = 25.0
    ft_setup_motion: float = 0.75
    still_mps: float = 0.7
    stoppage_min_s: float = 20.0
    fast_break_m: float = 9.0  # ball advance towards the attacked basket ...
    fast_break_s: float = 5.0  # ... within this time from the start of the possession
    half_court_min_s: float = 6.0
    timeout_max_s: float = 100.0  # breaks (inplay) shorter than this = time-out / stoppage
    halftime_min_s: float = 420.0  # longest break at least this long = half time
    halftime_rel_min_s: float = 240.0  # ... or at least this long and clearly the longest of several breaks
    halftime_rel_factor: float = 2.5  # (youth games: short half time, inplay may also cut it short)
    halftime_mid_min_span_s: float = 2400.0  # else, in a video this long: the break >= halftime_mid_min_s
    halftime_mid_min_s: float = 90.0  # nearest the middle of the game (in its middle third) = half time
    direction_min_conf: float = 0.3  # attack direction this sure overrides the holder's team for shots
    basket_direction_min_conf: float = 0.15  # ... for crediting shots/baskets/free throws (score-bar check:
    # direction right on 7/7 clips incl. confidence 0.18/0.22; the last holder was right on only 3/6 makes)
    track_flip_min_frames: int = 8
    track_flip_share: float = 0.75
    track_flip_spread_s: float = 4.0
    relabel_stay_s: float = 10.0  # ball this long in one half (8-s back-court rule): holder is an attacker
    lineup_evict_s: float = 5.0  # a track this long may take the slot of a much shorter one
    reach_mps: float = 6.0  # lineup: a player off screen moves at most this fast (slot matching)


@dataclass
class PossessionV6Config:
    """Possession model v6 (data/plans/possession_design.md): possessions are segments between hard boundaries
    (score-bar / detected makes, shots reaching the rim, dead balls, confirmed holder changes); the team of each
    segment is a weighted vote of all evidence (log-odds, A positive). Written as doc["possession_model"]."""
    min_change_s: float = 1.0  # the other team must hold this long for a holder change to be a boundary
    change_gap_max_s: float = 1.0  # the boundary lies at most this long before the new team's first held frame
    w_bar: float = 6.0  # a make confirmed by the score bar (team = bar side)
    w_sheet: float = 6.0  # ... by the scoresheet
    w_rim_make: float = 5.0  # a detected make credited by the attack direction
    w_det_make: float = 2.0  # ... credited only by the detector / last holder
    w_shot: float = 4.0  # a shot at basket X: attacker of X (x direction confidence)
    w_holder: float = 1.0  # per second a track holds the ball ...
    holder_track_cap: float = 3.0  # ... capped per track (3 s of holding = full weight)
    holder_cap: float = 4.0  # ... and in total (so a confirmed make always beats mislabelled holders)
    w_loc: float = 0.3  # per second the ball is in half X (after loc_after_s): attacker of X
    w_paint: float = 0.6  # per second the ball is within 6.75 m of basket X
    loc_after_s: float = 5.0  # (8-s back-court rule; a defensive rebounder may hold the ball under the rim a while)
    loc_cap: float = 4.0
    w_after_make: float = 3.0  # after a make by T the other team inbounds
    w_after_ft: float = 1.5  # after a made last free throw the other team inbounds
    w_inbound: float = 1.5  # first holder at the boundary line after a dead ball
    unclear_score: float = 1.0  # |score| below this: team None ("unclear") instead of a wrong team
    ft_setup_guard_s: float = 1.0  # a free-throw set-up never starts before the last live shot's rim end + this
    bar_before_s: float = 4.0  # a bar change without a detected make: the make was ~this long before the change
    synthetic_make_window_s: float = 8.0  # ... or at the last shot at that team's basket within this window
    min_segment_s: float = 0.2


# ====================================================================================== input
@dataclass
class GameInput:
    t: np.ndarray
    calib: np.ndarray  # bool
    fps: float
    L: float
    W: float
    baskets: np.ndarray  # (2, 2) L, R floor points
    spec: dict
    # flat person rows
    r_i: np.ndarray  # frame index
    r_tid: np.ndarray  # track id (object array of str)
    r_team: np.ndarray  # 'A' 'B' 'R' 'U'
    r_xy: np.ndarray  # (m, 2)
    r_meas: np.ndarray  # bool
    r_speed: np.ndarray  # m/s (nan unknown)
    track_team: dict
    track_rows: dict  # tid -> (row indices sorted by frame)
    ball_xy: np.ndarray  # (n, 2) nan
    ball_f: np.ndarray  # 0 none, 1 possessed, 2 estimated
    ball_o: list  # owner track id (str) or None
    ball_img: np.ndarray  # bool: ball detected in the image
    teams: dict
    frame: np.ndarray
    track_number: dict = field(default_factory=dict)  # tid -> jersey number text (replay track_info "number")

    @property
    def n(self) -> int:
        return len(self.t)

    @property
    def mid(self) -> float:
        return self.W / 2

    def idx(self, t: float, side: str = "left") -> int:
        return int(np.searchsorted(self.t, t, side))

    def ft_x(self, side: str) -> float:
        return self.spec["ft_x"] if side == "L" else self.L - self.spec["ft_x"]

    def basket(self, side: str) -> np.ndarray:
        return self.baskets[0 if side == "L" else 1]


def _spec(game: dict) -> dict:
    from replay import court
    name = str((game or {}).get("court_standard") or "FIBA")
    return dict(court.STANDARDS.get(name, court.STANDARDS["FIBA"]))


def load_input(replay: dict) -> GameInput:
    t = np.asarray(replay.get("t") or [], float)
    n = len(t)
    fps = float(replay.get("fps") or (1.0 / np.median(np.diff(t)) if n > 2 else 10.0))
    c = replay.get("court") or {}
    L, W = float(c.get("length") or 28.0), float(c.get("width") or 15.0)
    spec = _spec(replay.get("game") or {})
    bk = c.get("baskets") or [[spec["basket_x"], W / 2], [L - spec["basket_x"], W / 2]]
    calib = np.asarray(replay.get("calib") or [1] * n, int) > 0
    ri, rt, rteam, rx, ry, rm = [], [], [], [], [], []
    track_team = {}
    for tr in replay.get("tracks") or []:
        tid = str(tr.get("id"))
        typ = tr.get("type")
        team = tr.get("team")
        code = "R" if typ == "referee" or team == "referee" else team if team in TEAMS else "U"
        track_team[tid] = code
        i0 = int(tr.get("i0") or 0)
        xs, ys, fs = tr.get("x") or [], tr.get("y") or [], tr.get("f") or []
        for j, (x, y) in enumerate(zip(xs, ys)):
            if x is None or y is None or i0 + j >= n:  # (a track list may run past the stored frames)
                continue
            ri.append(i0 + j)
            rt.append(tid)
            rteam.append(code)
            rx.append(x / 100.0)
            ry.append(y / 100.0)
            rm.append((fs[j] if j < len(fs) else 1) == 1)
    r_i = np.asarray(ri, int)
    order = np.lexsort((np.asarray(rt, object).astype(str), r_i)) if len(r_i) else np.zeros(0, int)
    r_i = r_i[order]
    r_tid = np.asarray(rt, object)[order] if len(rt) else np.zeros(0, object)
    r_team = np.asarray(rteam, object)[order] if len(rteam) else np.zeros(0, object)
    r_xy = np.c_[np.asarray(rx, float), np.asarray(ry, float)][order] if len(rx) else np.zeros((0, 2))
    r_meas = np.asarray(rm, bool)[order] if len(rm) else np.zeros(0, bool)
    track_rows: dict[str, np.ndarray] = {}
    if len(r_i):
        tid_str = r_tid.astype(str)
        by_tid = np.argsort(tid_str, kind="stable")  # stable: rows of a track stay in frame order
        uniq, starts = np.unique(tid_str[by_tid], return_index=True)
        bounds = list(starts) + [len(by_tid)]
        for k, tid in enumerate(uniq):
            track_rows[str(tid)] = by_tid[bounds[k]:bounds[k + 1]]
    # speed per row: displacement over ~+-0.5 s within the track
    speed = np.full(len(r_i), np.nan)
    half = max(1, int(round(0.5 * fps)))
    for tid, rows in track_rows.items():
        fi = r_i[rows]
        xy = r_xy[rows]
        if len(rows) < 2:
            continue
        lo = np.searchsorted(fi, fi - half)
        hi = np.searchsorted(fi, fi + half, "right") - 1
        dt = (t[fi[hi]] - t[fi[lo]])
        d = np.linalg.norm(xy[hi] - xy[lo], axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            v = np.where(dt > 0.2, d / dt, np.nan)
        speed[rows] = v
    b = replay.get("ball") or {}
    bx = b.get("x") or [None] * n
    by = b.get("y") or [None] * n
    ball_xy = np.full((n, 2), np.nan)
    for i in range(min(n, len(bx))):
        if bx[i] is not None and by[i] is not None:
            ball_xy[i] = (bx[i] / 100.0, by[i] / 100.0)
    def _fit(seq, fill):  # exactly n entries, whatever the stored list length (also when it is empty)
        seq = list(seq or [])[:n]
        return seq + [fill] * (n - len(seq))

    ball_f = np.asarray(_fit(b.get("f"), 0), int)
    ball_xy[ball_f == 3] = np.nan  # guessed (ball.py: display only, never seen there) = unknown for the analysis
    ball_f[ball_f == 3] = 0
    ball_o = [None if o in (None, "") else str(o) for o in (b.get("o") or [None] * n)][:n]
    ball_o += [None] * (n - len(ball_o))
    ball_img = np.array([bb is not None for bb in _fit(b.get("b"), None)], bool)
    teams = ((replay.get("game") or {}).get("teams") or {})
    frame = np.asarray(replay.get("frame") or np.round(t * 30).astype(int), int)
    numbers = {str(k): str(v["number"]) for k, v in (replay.get("track_info") or {}).items()
               if isinstance(v, dict) and v.get("number") not in (None, "")}
    return GameInput(t, calib, fps, L, W, np.asarray(bk, float).reshape(2, 2), spec, r_i, r_tid, r_team, r_xy,
                     r_meas, speed, track_team, track_rows, ball_xy, ball_f, ball_o, ball_img, teams, frame,
                     track_number=numbers)


def _frame_slices(gi: GameInput) -> tuple[np.ndarray, np.ndarray]:
    lo = np.searchsorted(gi.r_i, np.arange(gi.n), "left")
    hi = np.searchsorted(gi.r_i, np.arange(gi.n), "right")
    return lo, hi


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) runs where mask is True."""
    m = np.r_[False, np.asarray(mask, bool), False]
    d = np.flatnonzero(m[1:] != m[:-1])
    return list(zip(d[::2].tolist(), d[1::2].tolist()))


def _close_gaps(mask: np.ndarray, t: np.ndarray, gap_s: float) -> np.ndarray:
    out = mask.copy()
    runs = _runs(mask)
    for (a0, a1), (b0, b1) in zip(runs, runs[1:]):
        if t[b0] - t[a1 - 1] <= gap_s:
            out[a1:b0] = True
    return out


# ====================================================================================== per-frame signals
def holder_codes(gi: GameInput) -> list[str | None]:
    """'A'/'B' = a player of that team holds the ball, 'R' referee, 'U' a person of unknown team, None loose."""
    out: list[str | None] = []
    for i in range(gi.n):
        o = gi.ball_o[i]
        if gi.ball_f[i] != 1 or o is None:
            out.append(None)
            continue
        out.append(gi.track_team.get(o, "U"))
    return out


def motion_level(gi: GameInput, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Median speed (m/s) of the visible players per frame (nan = nobody)."""
    out = np.full(gi.n, np.nan)
    for i in range(gi.n):
        a, b = lo[i], hi[i]
        if b - a < 3:
            continue
        sel = slice(a, b)
        v = gi.r_speed[sel][(gi.r_team[sel] != "R") & np.isfinite(gi.r_speed[sel])]
        if len(v) >= 3:
            out[i] = float(np.median(v))
    # smooth over ~1 s
    k = max(1, int(round(gi.fps)))
    ok = np.isfinite(out)
    if ok.sum() > k:
        vals = np.where(ok, out, 0.0)
        c = np.convolve(vals, np.ones(k), "same")
        w = np.convolve(ok.astype(float), np.ones(k), "same")
        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.where(w > 0, c / np.maximum(w, 1e-9), np.nan)
    return out


# ====================================================================================== free throws
def ft_formation(gi: GameInput, lo: np.ndarray, hi: np.ndarray, cfg: GSConfig) -> dict[str, np.ndarray]:
    """Per frame and basket side: (shooter row index or -1). A frame qualifies when one player stands at the
    free-throw line, at least ``ft_min_lane`` players stand along the lane lines and those players are still."""
    res = {}
    mid = gi.mid
    for s in "LR":
        ftx = gi.ft_x(s)
        sign = -1.0 if s == "L" else 1.0  # direction from the FT line towards the baseline (x)
        base_x = 0.0 if s == "L" else gi.L
        shooter = np.full(gi.n, -1, int)
        for i in range(gi.n):
            a, b = lo[i], hi[i]
            if b - a < 3:
                continue
            xy = gi.r_xy[a:b]
            team = gi.r_team[a:b]
            sp = gi.r_speed[a:b]
            players = team != "R"
            dx = (xy[:, 0] - ftx) * sign  # >0 = between the FT line and the baseline
            dy = np.abs(xy[:, 1] - mid)
            # the shooter stands just behind the line (towards the centre = negative dx)
            at_spot = players & (dx >= cfg.ft_spot_dx[0]) & (dx <= cfg.ft_spot_dx[1]) & (dy <= cfg.ft_spot_dy)
            dbase = np.abs(xy[:, 0] - base_x)
            din = (xy[:, 0] - base_x) * -sign  # signed distance into the court from the baseline
            # lane players: calibration error near the far baseline can put them just beyond it
            lane = players & (din >= -cfg.ft_lane_outside_m) & (din <= cfg.ft_lane_x[1]) & \
                (dy >= cfg.ft_lane_y[0]) & (dy <= cfg.ft_lane_y[1])
            if not at_spot.any() or lane.sum() < cfg.ft_min_lane:
                continue
            still_l = sp[lane]
            still_l = still_l[np.isfinite(still_l)]
            if len(still_l) and np.median(still_l) > cfg.ft_still_mps:
                continue
            cand = np.flatnonzero(at_spot)
            # nobody else inside the key between the shooter and the basket (live play has people in the paint)
            inside = players & (dbase > 0.3) & (dbase < cfg.ft_lane_x[1] - 0.2) & (dy < cfg.ft_lane_y[0] - 0.3)
            if inside.sum() > 1:
                continue
            j = cand[np.argmin(np.abs((xy[cand, 0] - ftx)) + np.abs(xy[cand, 1] - mid))]
            if np.isfinite(sp[j]) and sp[j] > 1.5:
                continue
            shooter[i] = a + j
        res[s] = shooter
    return res


def ft_windows(gi: GameInput, shooter: dict[str, np.ndarray], cfg: GSConfig, motion: np.ndarray | None = None,
               hits: list | None = None) -> list[dict]:
    """Free-throw situations: formation frames (:func:`ft_formation`) for at least ``ft_min_s`` while everybody
    is still, with the ball at the free-throw spot or going to that rim; nearby windows are merged."""
    out = []
    for s in "LR":
        mask = shooter[s] >= 0
        mask = _close_gaps(mask, gi.t, 2.0)
        spot = np.array([gi.ft_x(s), gi.mid])
        near_spot = np.linalg.norm(gi.ball_xy - spot, axis=1) < 1.6
        for a, b in _runs(mask):
            if gi.t[b - 1] - gi.t[a] < cfg.ft_min_s:
                continue
            if motion is not None:
                m = motion[a:b]
                if np.isfinite(m).any() and np.nanmedian(m) > cfg.ft_window_motion:
                    continue
            rim = any(side == s and gi.t[a] - 1.0 <= gi.t[i] <= gi.t[b - 1] + 3.0 for i, side, _ in (hits or []))
            if not rim and near_spot[a:b].mean() < 0.1:
                continue
            rows = shooter[s][a:b]
            rows = rows[rows >= 0]
            teams = [gi.r_team[r] for r in rows]
            tids = [str(gi.r_tid[r]) for r in rows]
            out.append({"side": s, "i0": a, "i1": b - 1, "teams": teams, "tids": tids,
                        "frames": int(len(rows))})
    out.sort(key=lambda w: gi.t[w["i0"]])
    # merge windows of the same side separated by less than ft_merge_s (players relax between attempts)
    merged: list[dict] = []
    for w in out:
        p = merged[-1] if merged else None
        if p is not None and p["side"] == w["side"] and gi.t[w["i0"]] - gi.t[p["i1"]] <= cfg.ft_merge_s:
            p["i1"] = max(p["i1"], w["i1"])
            p["teams"] += w["teams"]
            p["tids"] += w["tids"]
            p["frames"] += w["frames"]
        else:
            merged.append(dict(w))
    for w in merged:
        tc = {k: w["teams"].count(k) for k in TEAMS}
        team = max(tc, key=tc.get) if max(tc.values()) > 0 else None
        if team is not None and tc[team] < 0.6 * sum(tc.values()):
            team = None
        w["team"] = team
        w["shooter_tracks"] = sorted(set(w["tids"]), key=lambda x: -w["tids"].count(x))[:3]
        w["t0"], w["t1"] = float(gi.t[w["i0"]]), float(gi.t[w["i1"]])
        w["confidence"] = round(float(min(1.0, 0.5 + 0.1 * (w["t1"] - w["t0"]))) * (0.9 if team else 0.7), 2)
        del w["teams"], w["tids"]
    return merged


# ====================================================================================== rims / shots
def rim_proximity(ball_rows: Sequence[np.ndarray], rims: dict, t: np.ndarray, min_conf: float = 0.3,
                  static_window_s: float = 2.0, static_frac: float = 0.5) -> list[list]:
    """Pass-2 helper: every frame where a ball candidate is close to (or above) a projected rim.

    ball_rows[i]: (m, 5) [x1, y1, x2, y2, conf]; rims: scoring.rim_tracks output {"L"/"R": [(u, v, r) | None]}.
    Returns sparse [[i, side, dx, dy, conf], ...] in rim radii (dy < 0 = above). Candidates that stay at the same
    rim-relative place (logos, the ring itself, a spare ball) are dropped like in scoring.py."""
    from replay import scoring
    from replay.config import ScoringConfig
    scfg = ScoringConfig(static_window_s=static_window_s, static_frac=static_frac)
    out: list[list] = []
    t = np.asarray(t, float)
    for side, rim in (rims or {}).items():
        obs: list[list] = []
        for i, rr in enumerate(rim):
            cur = []
            rows = ball_rows[i] if i < len(ball_rows) else np.zeros((0, 5))
            if rr is not None and len(rows):
                u, v, r = rr
                for x1, y1, x2, y2, conf in np.asarray(rows, float)[:, :5]:
                    if conf < min_conf:
                        continue
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    bw = 0.5 * ((x2 - x1) + (y2 - y1))
                    dx, dy = (cx - u) / r, (cy - v) / r
                    ratio = bw / (2 * scoring.BALL_TO_RIM_WIDTH * r)
                    if abs(dx) <= 3.0 and -8.0 <= dy <= 2.5 and 0.4 <= ratio <= 2.6:
                        cur.append(scoring._Obs(float(dx), float(dy), float(ratio), float(conf), (cx, cy)))
            obs.append(cur)
        scoring._drop_static(obs, t, scfg)
        for i, cur in enumerate(obs):
            if cur:
                o = min(cur, key=lambda o: abs(o.dx) + 0.3 * abs(o.dy))
                out.append([i, side, round(o.dx, 2), round(o.dy, 2), round(o.conf, 2)])
    out.sort(key=lambda r: r[0])
    return out


def _rim_hits(gi: GameInput, rim_ball: list | None, hold: list[str | None]) -> list[tuple[int, str, float]]:
    """(frame, side, strength) where the ball is at a rim, from pass-2 candidates and/or rim anchors of the
    ball track (ball exactly at the basket floor point while not held)."""
    hits: list[tuple[int, str, float]] = []
    for s, k in (("L", 0), ("R", 1)):
        d = np.linalg.norm(gi.ball_xy - gi.baskets[k], axis=1)
        for i in np.flatnonzero((d < 0.03) & (gi.ball_f == 2)):
            hits.append((int(i), s, 0.8))
    for row in rim_ball or []:
        try:
            i, s, dx, dy, conf = int(row[0]), str(row[1]), float(row[2]), float(row[3]), float(row[4])
        except (TypeError, ValueError, IndexError):
            continue
        if not (0 <= i < gi.n) or s not in "LR":
            continue
        if hold[i] in TEAMS:
            o = gi.ball_o[i]
            p = _track_xy(gi, o, i)
            if p is not None and np.linalg.norm(p - gi.basket(s)) > 3.0:
                continue  # held by a player far from this basket: the ball is only near the rim in the image
        strength = 0.5 + 0.3 * conf + (0.15 if abs(dx) <= 1.5 and dy <= 1.0 else 0.0)
        hits.append((i, s, min(1.0, strength)))
    hits.sort()
    return hits


def _track_xy(gi: GameInput, tid: str | None, i: int) -> np.ndarray | None:
    if tid is None:
        return None
    rows = gi.track_rows.get(str(tid))
    if rows is None or not len(rows):
        return None
    fi = gi.r_i[rows]
    k = int(np.searchsorted(fi, i))
    if k < len(fi) and fi[k] == i:
        return gi.r_xy[rows[k]]
    return None


def _track_xy_near(gi: GameInput, tid: str | None, i: int, max_frames: int) -> np.ndarray | None:
    if tid is None:
        return None
    rows = gi.track_rows.get(str(tid))
    if rows is None or not len(rows):
        return None
    fi = gi.r_i[rows]
    k = int(np.clip(np.searchsorted(fi, i), 0, len(fi) - 1))
    best = min((kk for kk in (k - 1, k) if 0 <= kk < len(fi)), key=lambda kk: abs(fi[kk] - i))
    return gi.r_xy[rows[best]] if abs(fi[best] - i) <= max_frames else None


def shot_attempts(gi: GameInput, hits: list[tuple[int, str, float]], hold: list[str | None],
                  baskets: list[dict], cfg: GSConfig) -> list[dict]:
    """Group rim hits into attempts and find the release (last holder) for each."""
    groups: list[dict] = []
    for i, s, w in hits:
        g = groups[-1] if groups and groups[-1]["side"] == s else None
        g = next((g for g in reversed(groups[-3:]) if g["side"] == s and gi.t[i] - gi.t[g["i1"]] <= cfg.rim_group_s),
                 None)
        if g is not None:
            g["i1"] = i
            g["w"] = max(g["w"], w)
            g["n"] += 1
        else:
            groups.append({"side": s, "i0": i, "i1": i, "w": w, "n": 1})
    # made baskets without rim hits still are attempts
    for e in baskets:
        et = _f(e.get("t"))
        s = e.get("basket")
        if et is None or s not in ("L", "R"):
            continue
        i = gi.idx(et)
        i = min(max(i, 0), gi.n - 1)
        if not any(g["side"] == s and gi.t[g["i0"]] - 2.0 <= et <= gi.t[g["i1"]] + 2.5 for g in groups):
            ta = _f((e.get("details") or {}).get("t_above"))
            i0 = gi.idx(ta) if ta is not None else i
            groups.append({"side": s, "i0": min(i0, i), "i1": i, "w": 0.9, "n": 1})
    groups.sort(key=lambda g: g["i0"])
    shots = []
    max_back = int(round(cfg.release_window_s * gi.fps))
    for g in groups:
        i0 = g["i0"]
        rel, shooter, team = None, None, None
        for j in range(i0, max(-1, i0 - max_back), -1):
            if hold[j] in TEAMS:
                p = _track_xy(gi, gi.ball_o[j], j)
                if p is not None and np.linalg.norm(p - gi.basket(g["side"])) < 0.9 and j > i0 - 3:
                    continue  # a player under the rim (rebounder / defender) is not the shooter
                rel, shooter, team = j, gi.ball_o[j], hold[j]
                break
        pos = _track_xy(gi, shooter, rel) if rel is not None else None
        dist = float(np.linalg.norm(pos - gi.basket(g["side"]))) if pos is not None else None
        if dist is not None and dist > cfg.max_shot_dist_m:
            continue
        if dist is not None and rel is not None and dist > 2.5 and \
                gi.t[i0] - gi.t[rel] < max(0.25, dist / cfg.max_ball_speed_mps):
            continue  # the ball cannot fly that far that fast: only near the rim in the image (dribble, pass)
        conf = g["w"] * (1.0 if rel is not None else 0.75)
        shots.append({"side": g["side"], "i_rim": i0, "i_rim_end": g["i1"],
                      "i_rel": rel if rel is not None else max(0, i0 - int(round(0.8 * gi.fps))),
                      "shooter": shooter, "team": team, "pos": None if pos is None else [round(float(v), 2) for v in pos],
                      "dist": None if dist is None else round(dist, 2), "confidence": round(min(1.0, conf), 2),
                      "made": None, "basket_id": None})
    # made?
    for e in baskets:
        et, s = _f(e.get("t")), e.get("basket")
        if et is None:
            continue
        cand = [sh for sh in shots if sh["side"] == s and gi.t[sh["i_rim"]] - 2.0 <= et <= gi.t[sh["i_rim_end"]] + 2.5]
        if cand:
            sh = min(cand, key=lambda sh: abs(gi.t[sh["i_rim_end"]] - et))
            sh["made"] = True
            sh["basket_id"] = str(e.get("id"))
            if sh["team"] is None and e.get("team") in TEAMS:
                sh["team"] = e["team"]
    # one attempt can't follow another within 1 s at the same rim (rebound tip-ins are kept if > 1 s)
    out: list[dict] = []
    for sh in shots:
        if out and out[-1]["side"] == sh["side"] and gi.t[sh["i_rim"]] - gi.t[out[-1]["i_rim_end"]] < 1.0:
            out[-1]["i_rim_end"] = sh["i_rim_end"]
            out[-1]["made"] = out[-1]["made"] or sh["made"]
            continue
        out.append(sh)
    return out


def _f(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


# ====================================================================================== control / possessions
def control_runs(gi: GameInput, hold: list[str | None], cfg: GSConfig) -> list[dict]:
    runs: list[dict] = []
    cur = None
    for i, h in enumerate(hold):
        if h is None:
            continue
        if cur is not None and cur["code"] == h and gi.t[i] - gi.t[cur["i1"]] <= cfg.hold_gap_s:
            cur["i1"] = i
            cur["k"] += 1
            cur["owners"].append(gi.ball_o[i])
        else:
            cur = {"code": h, "i0": i, "i1": i, "k": 1, "owners": [gi.ball_o[i]]}
            runs.append(cur)
    keep = [r for r in runs if r["k"] >= 3 or gi.t[r["i1"]] - gi.t[r["i0"]] >= cfg.min_run_s]
    # blips: short run of team Y between two runs of team X close in time
    team_runs = [r for r in keep if r["code"] in TEAMS]
    drop = set()
    for k in range(1, len(team_runs) - 1):
        p, c, nx = team_runs[k - 1], team_runs[k], team_runs[k + 1]
        if c["code"] != p["code"] and p["code"] == nx["code"] and \
                gi.t[c["i1"]] - gi.t[c["i0"]] < cfg.blip_s and \
                gi.t[c["i0"]] - gi.t[p["i1"]] <= cfg.blip_gap_s and gi.t[nx["i0"]] - gi.t[c["i1"]] <= cfg.blip_gap_s:
            drop.add(id(c))
    return [r for r in keep if id(r) not in drop]


def _in_windows(t: float, windows: Iterable[tuple[float, float]]) -> bool:
    return any(a <= t <= b for a, b in windows)


def possessions(gi: GameInput, runs: list[dict], dead: list[tuple[float, float, str]], cfg: GSConfig,
                shot_times: Sequence[float] | None = None) -> list[dict]:
    """Team possessions: consecutive control runs of one team, split at dead balls. A "possession" shorter than
    ``min_poss_s`` with few holder frames is a touch (deflection, scramble, holder mix-up) and is dropped, which
    lets the neighbouring spells of the other team join up."""
    runs = [r for r in runs if r["code"] in TEAMS]
    for _ in range(4):
        out = _group_runs(gi, runs, dead)
        weak = [p for p in out if gi.t[p["i1"]] - gi.t[p["i0"]] < cfg.min_poss_s and p["held"] < 6]
        if not weak:
            break
        drop = {id(r) for p in weak for r in p["run_objs"]}
        runs = [r for r in runs if id(r) not in drop]
    for _ in range(8):
        sandwiched = [q for p, q, r in zip(out, out[1:], out[2:])
                      if p["team"] == r["team"] != q["team"] and gi.t[q["i1"]] - gi.t[q["i0"]] < cfg.sandwich_s
                      and gi.t[r["i0"]] - gi.t[p["i1"]] < cfg.sandwich_gap_s
                      and not any(a <= gi.t[r["i0"]] and b >= gi.t[p["i1"]] for a, b, kind in dead
                                  if kind in ("timeout", "quarter_break", "half_time"))
                      and not any(kind == "free_throws" and b - 0.5 <= gi.t[q["i0"]] <= b + 3.0 for a, b, kind in dead)
                      and not any(gi.t[q["i0"]] - 0.5 <= t <= gi.t[r["i0"]] for t in shot_times or [])]
        if not sandwiched:
            break
        drop = {id(r) for q in sandwiched for r in q["run_objs"]}
        runs = [r for r in runs if id(r) not in drop]
        out = _group_runs(gi, runs, dead)
    return _finish_possessions(gi, out)


def _group_runs(gi: GameInput, runs: list[dict], dead: list[tuple[float, float, str]]) -> list[dict]:
    dead_w = [(a, b) for a, b, _ in dead]
    out: list[dict] = []
    for r in runs:
        if r["code"] not in TEAMS:
            continue
        ta, tb = float(gi.t[r["i0"]]), float(gi.t[r["i1"]])
        if any(a <= ta and tb <= b for a, b in dead_w):
            continue  # inside a dead ball (e.g. the shooter holding the ball at the free-throw line)
        p = out[-1] if out else None
        split = p is None or p["team"] != r["code"] or any(gi.t[p["i1"]] <= b and a <= ta for a, b in dead_w
                                                           if a >= gi.t[p["i1"]] - 0.5)
        if not split and ta - gi.t[p["i1"]] > 6.0:
            split = True  # long loose period (likely missed shot + rebound out of view)
        if split:
            out.append({"team": r["code"], "i0": r["i0"], "i1": r["i1"], "held": r["k"], "runs": 1,
                        "owners": list(r["owners"]), "run_objs": [r]})
        else:
            p["i1"] = r["i1"]
            p["held"] += r["k"]
            p["runs"] += 1
            p["owners"] += r["owners"]
            p["run_objs"].append(r)
    return out


def _finish_possessions(gi: GameInput, out: list[dict]) -> list[dict]:
    for k, p in enumerate(out):
        dur = gi.t[p["i1"]] - gi.t[p["i0"]] + 1.0 / gi.fps
        span = p["i1"] - p["i0"] + 1
        p["t0"], p["t1"] = float(gi.t[p["i0"]]), float(gi.t[p["i1"]])
        seen = p["held"] / max(1, span)
        other = sum(1 for i in range(p["i0"], p["i1"] + 1) if gi.ball_f[i] == 1 and
                    gi.track_team.get(gi.ball_o[i] or "", "U") in TEAMS and
                    gi.track_team.get(gi.ball_o[i] or "", "U") != p["team"])
        conf = 0.45 + 0.35 * min(1.0, seen / 0.5) + 0.2 * min(1.0, dur / 3.0) - 0.6 * other / max(1, span)
        p["confidence"] = round(float(np.clip(conf, 0.05, 1.0)), 2)
        p["passes"] = sum(1 for a, b in zip(p["owners"], p["owners"][1:]) if a != b and a and b)
        p["duration_s"] = round(float(dur), 2)
    return out


# ====================================================================================== possession model v6
def _other(team: str | None) -> str | None:
    return None if team not in TEAMS else ("B" if team == "A" else "A")


def v6_makes(gi: GameInput, baskets: list[dict], shots: list[dict], direction: list[dict], cfg: GSConfig,
             v6: PossessionV6Config, scorebar: dict | None = None, make_teams: dict | None = None,
             extra_makes: list | None = None, off_sheet: set | None = None) -> tuple[list[dict], dict]:
    """Made baskets with the team that scored them, one source each (strongest first): the scoresheet
    (``make_teams`` = {basket id: (team, points)}, ``extra_makes`` = sheet events without a detection), the score
    bar (``scorebar`` = work/scorebar.json result: matched by ball_events.match_bar, bar side -> team by the
    majority of the direction-credited makes), the attack direction, the detector. Detected makes the bar / sheet
    does not have (``off_sheet``) are no makes (they end a segment like a missed shot).
    -> ([{t, side, team, points, source, basket_id, ft, shot}], info)."""
    from replay import ball_events as be
    by_id = {str(sh.get("basket_id")): sh for sh in shots if sh.get("basket_id")}
    bk = sorted((e for e in baskets if e.get("id") is not None and _f(e.get("t")) is not None),
                key=lambda e: float(e["t"]))
    info: dict[str, Any] = {"bar_used": False, "bar_votes": 0, "synthetic": 0}
    off = set(off_sheet or ())
    make_teams = make_teams or {}

    def dir_team(e: dict) -> tuple[str | None, str]:
        sh = by_id.get(str(e["id"]))
        att = attacker_of(direction, e.get("basket"), float(e["t"]), cfg.basket_direction_min_conf)
        if att is not None:
            return att, "direction"
        t = (sh or {}).get("team") or e.get("team")
        return (t if t in TEAMS else None), "detector"

    bar_team: dict[str, tuple[str | None, int]] = {}
    synthetic: list[dict] = []
    evs = be.bar_events(scorebar) if scorebar and not make_teams else []
    if evs and bk:
        m = be.match_bar(evs, bk)
        if len(m) >= be.BAR_MIN_CONFIRM * len(bk):
            votes = be.bar_map_votes(evs, bk, m, lambda e: dir_team(e)[0])
            side_team = be.bar_side_team(evs, bk, m, lambda e: dir_team(e)[0]) if votes >= 2 else {0: None, 1: None}
            info.update({"bar_used": True, "bar_votes": votes, "bar_matched": len(m), "bar_events": len(evs),
                         "side_team": side_team})
            for j, k in m.items():
                bar_team[str(bk[j]["id"])] = (side_team[evs[k]["side"]], int(evs[k]["points"]))
            off |= {str(e["id"]) for j, e in enumerate(bk) if j not in m}
            used = set(m.values())
            lo, hi = be.BAR_WINDOW_S
            for k, b in enumerate(evs):
                if k in used:
                    continue
                team = side_team[b["side"]]
                cand = [sh for sh in shots if not sh.get("ft") and not sh.get("made")
                        and (team is None or attacker_of(direction, sh["side"], float(gi.t[sh["i_rim"]]),
                                                         cfg.basket_direction_min_conf) in (team, None))
                        and b["t"] - hi <= float(gi.t[sh["i_rim_end"]]) <= b["t"] - lo]
                sh = max(cand, key=lambda s: float(gi.t[s["i_rim_end"]]), default=None)
                t = float(gi.t[sh["i_rim_end"]]) if sh is not None else b["t"] - v6.bar_before_s
                synthetic.append({"t": round(t, 2), "side": sh["side"] if sh else None, "team": team,
                                  "points": int(b["points"]), "source": "bar", "basket_id": None,
                                  "ft": b["points"] == 1, "shot": sh, "bar_t": b["t"]})
    makes: list[dict] = []
    for e in bk:
        bid = str(e["id"])
        sh = by_id.get(bid)
        ft = bool((sh or {}).get("ft"))
        if bid in make_teams:
            team, pts = make_teams[bid]
            src = "sheet"
        elif bid in off:
            continue
        elif bid in bar_team:
            team, pts = bar_team[bid]
            src = "bar" if team in TEAMS else "direction"
            if team not in TEAMS:
                team = dir_team(e)[0]
        else:
            team, src = dir_team(e)
            pts = 1 if ft else 3 if (sh or {}).get("value") == "3" or (sh or {}).get("dist", 0) and \
                sh["dist"] >= gi.spec["three_r"] + 0.25 else 2
        makes.append({"t": float(e["t"]), "side": e.get("basket"), "team": team if team in TEAMS else None,
                      "points": int(pts), "source": src, "basket_id": bid, "ft": ft, "shot": sh})
    for m in extra_makes or []:  # scoresheet events without a detection: time estimated -> only with a shot nearby
        t, team = _f(m.get("t_estimated", m.get("t"))), m.get("team")
        if t is None or team not in TEAMS:
            continue
        side = attacked_side(direction, team, t)
        cand = [sh for sh in shots if not sh.get("ft") and not sh.get("made") and sh["side"] == side
                and abs(float(gi.t[sh["i_rim_end"]]) - t) <= v6.synthetic_make_window_s]
        sh = min(cand, key=lambda s: abs(float(gi.t[s["i_rim_end"]]) - t), default=None)
        if sh is None:
            continue
        synthetic.append({"t": round(float(gi.t[sh["i_rim_end"]]), 2), "side": side, "team": team,
                          "points": int(m.get("points") or 2), "source": "sheet", "basket_id": None,
                          "ft": m.get("kind") == "ft", "shot": sh})
    info["synthetic"] = len(synthetic)
    makes += synthetic
    makes.sort(key=lambda m: m["t"])
    return makes, info


def v6_user_makes(makes: list[dict], user: dict | None) -> list[dict]:
    """The corrections ledger's makes (corrections.apply_events) applied to the v6 makes: a coach's make takes over
    the detected make nearest in time (within USER_MAKE_MATCH_S, same basket when both are known) - team, points
    and source "user" - or is added as a make of its own. No user makes: ``makes`` unchanged."""
    if not user or not user.get("makes"):
        return makes
    out = [dict(m) for m in makes]
    taken: set[int] = set()
    for um in user["makes"]:
        best = None
        for j, m in enumerate(out):
            if j in taken or abs(m["t"] - um["t"]) > USER_MAKE_MATCH_S:
                continue
            if um.get("side") and m.get("side") and um["side"] != m["side"]:
                continue
            if best is None or abs(m["t"] - um["t"]) < abs(out[best]["t"] - um["t"]):
                best = j
        if best is not None:
            m = out[best]
            m.update(team=um["team"], points=int(um["points"]), source="user", ft=um["points"] == 1,
                     side=um.get("side") or m.get("side"))
            taken.add(best)
        else:
            out.append({"t": float(um["t"]), "side": um.get("side"), "team": um["team"], "points": int(um["points"]),
                        "source": "user", "basket_id": None, "ft": um["points"] == 1, "shot": None})
            taken.add(len(out) - 1)
    out.sort(key=lambda m: m["t"])
    return out


def v6_boundaries(gi: GameInput, makes: list[dict], shots: list[dict], dead: list[tuple[float, float, str]],
                  fts: list[dict], runs: list[dict], v6: PossessionV6Config, user: dict | None = None
                  ) -> tuple[list[dict], list[dict]]:
    """Dead-ball blocks and live spans with their split points. ``user`` (corrections.apply_events): the coach's
    possession boundaries become split points of kind "user".
    -> (blocks [{t0, t1, kind, ft}], spans [{t0, t1, splits: [{t, kind, make|shot|None}], after: block|None}])."""
    blocks: list[dict] = []
    ft_by_setup = {w.get("t_setup"): w for w in fts}
    for a, b, kind in sorted(dead):
        blk = {"t0": float(a), "t1": float(b), "kind": kind, "ft": ft_by_setup.get(a) if kind == "free_throws" else None}
        if blocks and blk["t0"] <= blocks[-1]["t1"] + 1e-6:
            p = blocks[-1]
            p["t1"] = max(p["t1"], blk["t1"])
            if blk["kind"] == "free_throws" and p["kind"] != "free_throws":
                p["kind"], p["ft"] = "free_throws", blk["ft"]
            elif p["kind"] not in ("free_throws", "half_time", "quarter_break") and \
                    blk["kind"] in ("half_time", "quarter_break", "timeout"):
                p["kind"] = blk["kind"]
            continue
        blocks.append(blk)
    t0, t1 = float(gi.t[0]), float(gi.t[-1])
    spans: list[dict] = []
    cursor, after = t0, None
    for blk in blocks:
        if blk["t0"] - cursor > 0.3:
            spans.append({"t0": cursor, "t1": blk["t0"], "after": after, "splits": []})
        cursor, after = max(cursor, blk["t1"]), blk
    if t1 - cursor > 0.3:
        spans.append({"t0": cursor, "t1": t1, "after": after, "splits": []})
    make_shots = {id(m["shot"]) for m in makes if m.get("shot") is not None}
    cands = [{"t": m["t"], "kind": "make", "make": m} for m in makes if not m["ft"]]
    for sh in shots:
        if sh.get("ft") or id(sh) in make_shots:
            continue
        cands.append({"t": float(gi.t[sh["i_rim_end"]]), "kind": "shot", "shot": sh})
    for t in (user or {}).get("boundaries", []):
        if not any(c["kind"] == "make" and abs(c["t"] - t) < 0.2 for c in cands):
            cands.append({"t": float(t), "kind": "user"})
    cands.sort(key=lambda c: c["t"])
    team_runs = [r for r in runs if r["code"] in TEAMS]
    for sp in spans:
        a, b = sp["t0"], sp["t1"]
        sel: list[dict] = []
        for c in cands:
            if not (a + v6.min_segment_s <= c["t"] <= b - v6.min_segment_s):
                continue
            if c["kind"] == "shot" and any(abs(x["t"] - c["t"]) <= 1.0 for x in cands if x["kind"] == "make"):
                continue  # the rim contact of a make
            if sel and abs(sel[-1]["t"] - c["t"]) < 0.2 and sel[-1]["kind"] == c["kind"]:
                continue
            sel.append(c)
        # holder changes confirmed by >= min_change_s of the other team's holding
        spells: list[dict] = []
        for r in team_runs:
            ta, tb = float(gi.t[r["i0"]]), float(gi.t[r["i1"]])
            if tb < a or ta > b:
                continue
            if spells and spells[-1]["team"] == r["code"]:
                spells[-1]["t1"], spells[-1]["held"] = tb, spells[-1]["held"] + r["k"]
            else:
                spells.append({"team": r["code"], "t0": ta, "t1": tb, "held": r["k"]})
        strong = [s for s in spells if s["t1"] - s["t0"] >= v6.min_change_s]
        merged: list[dict] = []
        for s in strong:
            if merged and merged[-1]["team"] == s["team"]:
                merged[-1]["t1"] = s["t1"]
            else:
                merged.append(dict(s))
        for p, q in zip(merged, merged[1:]):
            if any(p["t1"] - 0.3 <= c["t"] <= q["t0"] + 0.3 for c in sel):
                continue  # a make / shot between them already explains the change
            gap = max(0.0, q["t0"] - p["t1"])
            t = q["t0"] - min(v6.change_gap_max_s, gap / 2)
            if a + v6.min_segment_s <= t <= b - v6.min_segment_s:
                sel.append({"t": t, "kind": "holder"})
        sel.sort(key=lambda c: c["t"])
        sp["splits"] = sel
    return blocks, spans


def v6_vote(gi: GameInput, i0: int, i1: int, t_a: float, t_b: float, hold: list, dirsign: np.ndarray,
            att_of: Any, shots: list[dict], end: dict | None, start: dict | None, v6: PossessionV6Config,
            make_shots: set | None = None, user: dict | None = None) -> dict:
    """Evidence vote for the segment [t_a, t_b] (frames i0..i1): log-odds, A positive. ``end`` / ``start`` are the
    split points (or blocks) ending / starting the segment. ``user`` (corrections.apply_events): a coach's anchor
    covering most of the segment adds USER_WEIGHT for its team (evidence "user")."""
    ev = {"make": 0.0, "shot": 0.0, "holder": 0.0, "location": 0.0, "rule": 0.0}
    if user is not None:
        ev["user"] = 0.0
        for a in user.get("anchors", []):
            cover = min(t_b, a["t1"]) - max(t_a, a["t0"])
            if cover > 0 and cover >= 0.5 * max(t_b - t_a, 1e-6):
                ev["user"] += {"A": 1.0, "B": -1.0}[a["team"]] * USER_WEIGHT
    sign = {"A": 1.0, "B": -1.0}
    make_shots = make_shots or set()
    # holder seconds (w_holder per second, capped per track and in total: a brief mix-up counts little, and
    # mislabelled tracks must not beat a confirmed make)
    per_track: dict[str, float] = {}
    for i in range(i0, i1 + 1):
        h = hold[i]
        if h in TEAMS:
            tid = gi.ball_o[i] or "?"
            per_track[tid] = per_track.get(tid, 0.0) + sign[h] * v6.w_holder / gi.fps
    hv = sum(float(np.clip(v, -v6.holder_track_cap, v6.holder_track_cap)) for v in per_track.values())
    ev["holder"] = float(np.clip(hv, -v6.holder_cap, v6.holder_cap))
    # ball location (live, after loc_after_s since the segment start): the half / paint the ball is in belongs
    # to the team attacking that basket (8-s back-court rule)
    j0 = max(i0, gi.idx(t_a + v6.loc_after_s))
    if j0 <= i1:
        idx = np.arange(j0, i1 + 1)
        bx = gi.ball_xy[idx]
        ok = np.isfinite(bx[:, 0]) & (dirsign[idx] != 0)
        if ok.any():
            dt = 1.0 / gi.fps
            half_r = bx[:, 0] > gi.L / 2
            # dirsign = +1 * conf when A attacks R: the ball in R -> A (+), in L -> B (-)
            lv = np.where(half_r, dirsign[idx], -dirsign[idx]) * ok
            d_l = np.linalg.norm(bx - gi.baskets[0], axis=1)
            d_r = np.linalg.norm(bx - gi.baskets[1], axis=1)
            paint = np.where(half_r, d_r, d_l) < 6.75
            loc = float(np.nansum(lv) * v6.w_loc * dt + np.nansum(lv * paint) * v6.w_paint * dt)
            ev["location"] = float(np.clip(loc, -v6.loc_cap, v6.loc_cap))
    # shots at basket X inside the segment: the attacker of X
    for sh in shots:
        if sh.get("ft") or id(sh) in make_shots:  # (the shot of a make is counted as the make)
            continue
        tr = float(gi.t[sh["i_rim_end"]])
        if not (t_a + 0.01 < tr <= t_b + 0.3):  # (a shot split lies at the rim end: it belongs to the segment before)
            continue
        att, conf = att_of(sh["side"], float(gi.t[sh["i_rim"]]))
        if att in TEAMS:
            ev["shot"] += sign[att] * v6.w_shot * conf
    # the make ending the segment
    if end is not None and end.get("kind") == "make":
        m = end["make"]
        if m.get("team") in TEAMS:
            w = {"bar": v6.w_bar, "sheet": v6.w_sheet, "direction": v6.w_rim_make,
                 "user": USER_WEIGHT}.get(m["source"], v6.w_det_make)
            ev["make"] += sign[m["team"]] * w
    # start rules
    if start is not None:
        k = start.get("kind")
        if k == "make" and start["make"].get("team") in TEAMS:
            ev["rule"] += sign[_other(start["make"]["team"])] * (
                USER_WEIGHT if start["make"].get("source") == "user" else v6.w_after_make)
        elif k == "free_throws" and start.get("ft") and start["ft"].get("last_made") and start["ft"].get("team") in TEAMS:
            ev["rule"] += sign[_other(start["ft"]["team"])] * v6.w_after_ft
        elif k in ("timeout", "quarter_break", "half_time", "stoppage", "referee_has_ball", "free_throws"):
            lim = gi.idx(t_a + 3.0)
            for i in range(i0, min(i1, lim) + 1):
                if hold[i] in TEAMS:
                    p = _track_xy(gi, gi.ball_o[i], i)
                    if p is not None and not (0.6 < p[0] < gi.L - 0.6 and 0.6 < p[1] < gi.W - 0.6):
                        ev["rule"] += sign[hold[i]] * v6.w_inbound
                    break
    score = sum(ev.values())
    team = None if abs(score) < v6.unclear_score else ("A" if score > 0 else "B")
    conf = 2.0 / (1.0 + math.exp(-abs(score))) - 1.0
    return {"team": team, "score": round(score, 2), "confidence": round(conf, 2),
            "evidence": {k: round(v, 2) for k, v in ev.items()}}


def possessions_v6(gi: GameInput, hold: list, runs: list[dict], shots: list[dict], makes: list[dict],
                   dead: list[tuple[float, float, str]], fts: list[dict], direction: list[dict], cfg: GSConfig,
                   v6: PossessionV6Config, user: dict | None = None) -> list[dict]:
    """Possessions tiling the live play: segments between boundaries (:func:`v6_boundaries`), team by
    :func:`v6_vote`, same-team neighbours merged unless a make lies between them (an offensive rebound continues
    the possession), free-throw situations attached to the fouled team's possession. Each carries the fields the
    event loops of :func:`analyse` need (i0, i1, owners, run_objs, held, t_held0/1) plus outcome, points, shots,
    off_rebounds, evidence, confidence, kind. ``user``: the corrections ledger's possession constraints
    (corrections.apply_events; None = none)."""
    for w in fts:  # last attempt made? (the other team inbounds after a made last free throw)
        att = [sh for sh in shots if sh.get("ft") and sh.get("ft_window") == fts.index(w)]
        w["last_made"] = bool(att[-1].get("made")) if att else None
        w["ft_points"] = sum(1 for sh in att if sh.get("made"))
    blocks, spans = v6_boundaries(gi, makes, shots, dead, fts, runs, v6, user)
    # per-frame attack sign: +conf when A attacks R
    dirsign = np.zeros(gi.n)
    for d in direction:
        if d.get("A") and d["confidence"] >= cfg.basket_direction_min_conf:
            a, b = gi.idx(d["t0"] - 1), gi.idx(d["t1"] + 1, "right")
            dirsign[a:b] = (1.0 if d["A"] == "R" else -1.0) * d["confidence"]

    def att_of(side: str, t: float) -> tuple[str | None, float]:
        d = next((d for d in direction if d["t0"] - 1 <= t <= d["t1"] + 1), None)
        if d is None or not d.get("A") or d["confidence"] < cfg.basket_direction_min_conf:
            return None, 0.0
        return ("A" if d["A"] == side else "B"), float(d["confidence"])

    segs: list[dict] = []
    make_shots = {id(m["shot"]) for m in makes if m.get("shot") is not None}
    for sp in spans:
        edges = [{"t": sp["t0"], "kind": (sp["after"] or {}).get("kind", "start"), "ft": (sp["after"] or {}).get("ft")}]
        edges += sp["splits"]
        edges.append({"t": sp["t1"], "kind": "end"})
        for s, e in zip(edges, edges[1:]):
            t_a, t_b = float(s["t"]), float(e["t"])
            i0, i1 = gi.idx(t_a), max(gi.idx(t_a), gi.idx(t_b, "right") - 1)
            vote = v6_vote(gi, i0, i1, t_a, t_b, hold, dirsign, att_of, shots, e if e["kind"] != "end" else None,
                           s, v6, make_shots, user)
            segs.append({**vote, "t0": t_a, "t1": t_b, "i0": i0, "i1": i1, "start": s, "end": e, "span": id(sp)})
    # next block after each span (for outcomes)
    span_next: dict[int, dict | None] = {}
    for k, sp in enumerate(spans):
        nb = next((b for b in blocks if b["t0"] >= sp["t1"] - 1e-6), None)
        span_next[id(sp)] = nb

    def new_poss(sg: dict) -> dict:
        return {"team": sg["team"], "t0": sg["t0"], "t1": sg["t1"], "i0": sg["i0"], "i1": sg["i1"], "kind": "live",
                "segments": [sg], "off_rebounds": 0, "points": 0, "shots": [], "outcome": None,
                "start": _start_label(sg["start"]), "span": sg["span"]}

    out: list[dict] = []
    for sg in segs:
        p = out[-1] if out else None
        if p is not None and p["span"] == sg["span"] and p["team"] == sg["team"] and sg["team"] in TEAMS and \
                sg["start"]["kind"] != "make":
            if sg["start"]["kind"] == "shot":
                p["off_rebounds"] += 1
            p["t1"], p["i1"] = sg["t1"], sg["i1"]
            p["segments"].append(sg)
            continue
        out.append(new_poss(sg))
    # X, unclear, X without a make between: one possession of X
    k = 1
    while k + 1 < len(out):
        a, b, c = out[k - 1], out[k], out[k + 1]
        if b["team"] is None and a["team"] == c["team"] and a["team"] in TEAMS and a["span"] == c["span"] and \
                not any(sg["start"]["kind"] == "make" for sg in b["segments"] + c["segments"]):
            a["t1"], a["i1"] = c["t1"], c["i1"]
            a["segments"] += b["segments"] + c["segments"]
            a["off_rebounds"] += c["off_rebounds"] + sum(1 for sg in (b["segments"] + c["segments"])
                                                         if sg["start"]["kind"] == "shot")
            del out[k:k + 2]
            continue
        k += 1
    # free-throw blocks: part of the fouled team's possession when it precedes them, else their own possession
    ft_blocks = [b for b in blocks if b["kind"] == "free_throws" and b.get("ft")]
    for b in ft_blocks:
        w = b["ft"]
        team = attacker_of(direction, w["side"], w["t0"], cfg.basket_direction_min_conf) or w.get("team")
        prev = max((p for p in out if p["t1"] <= b["t0"] + 1e-6), key=lambda p: p["t1"], default=None)
        between_dead = any(x["t0"] >= (prev["t1"] if prev else -1) - 1e-6 and x["t1"] <= b["t0"] + 1e-6
                           and x["kind"] in ("quarter_break", "half_time") for x in blocks)
        fp = {"team": team, "t0": b["t0"], "t1": b["t1"], "i0": gi.idx(b["t0"]), "i1": max(0, gi.idx(b["t1"], "right") - 1),
              "kind": "free_throws", "segments": [], "off_rebounds": 0, "points": int(w.get("ft_points") or 0),
              "shots": [], "outcome": "ft", "start": "foul", "span": None, "confidence": w.get("confidence", 0.7),
              "evidence": {"rule": 0.0}, "score": 0.0, "ft_window": w}
        if prev is not None and prev["kind"] == "live" and prev["team"] == team and team in TEAMS and \
                not between_dead and prev["outcome"] is None and b["t0"] - prev["t1"] <= 15.0:
            prev["t1"], prev["i1"] = b["t1"], fp["i1"]
            prev["points"] += fp["points"]
            prev["outcome"] = "ft"
            prev["ft_window"] = w
        else:
            out.append(fp)
    out.sort(key=lambda p: p["t0"])
    # outcomes, points, shots, provenance
    for k, p in enumerate(out):
        nxt = out[k + 1] if k + 1 < len(out) else None
        if p["kind"] == "live":
            last = p["segments"][-1]
            end = last["end"]
            if p["outcome"] is None:
                if end["kind"] == "make":
                    p["outcome"] = "made"
                    p["points"] += int(end["make"].get("points") or 0)
                    p["make_source"] = end["make"].get("source")
                elif end["kind"] == "shot":
                    p["outcome"] = "missed"
                elif end["kind"] in ("holder", "user"):
                    p["outcome"] = "turnover"
                else:  # the span ends: a dead-ball block or the end of the data
                    nb = span_next.get(p["span"])
                    kind = (nb or {}).get("kind")
                    p["outcome"] = "period_end" if kind in ("quarter_break", "half_time") or nb is None else \
                        "dead_ball" if kind != "free_throws" else "foul"
            p["shots"] = [{"t": round(float(gi.t[sh["i_rel"]]), 2), "basket": sh["side"], "made": bool(sh.get("made")),
                           "basket_event": sh.get("basket_id")}
                          for sh in shots if not sh.get("ft") and p["t0"] + 0.01 < float(gi.t[sh["i_rim_end"]]) <= p["t1"] + 0.3]
            for sg in p["segments"]:
                if sg["end"]["kind"] == "make" and sg["end"]["make"].get("basket_id") is None:
                    p["shots"].append({"t": sg["end"]["make"]["t"], "basket": sg["end"]["make"].get("side"),
                                       "made": True, "basket_event": None, "synthetic": True})
            dur = sum(sg["t1"] - sg["t0"] for sg in p["segments"])
            p["confidence"] = round(sum(sg["confidence"] * (sg["t1"] - sg["t0"]) for sg in p["segments"]) / max(dur, 1e-6), 2)
            ev: dict[str, float] = {}
            for sg in p["segments"]:
                for key, v in sg["evidence"].items():
                    ev[key] = round(ev.get(key, 0.0) + v, 2)
            p["evidence"] = ev
            p["score"] = round(sum(sg["score"] for sg in p["segments"]), 2)
            if p["team"] is None:
                p["outcome"] = "unknown"
        # fields the event loops need
        idx = range(p["i0"], p["i1"] + 1)
        mine = [i for i in idx if hold[i] == p["team"]] if p["team"] in TEAMS else []
        p["owners"] = [gi.ball_o[i] for i in mine]
        p["held"] = len(mine)
        p["t_held0"] = float(gi.t[mine[0]]) if mine else p["t0"]
        p["t_held1"] = float(gi.t[mine[-1]]) if mine else p["t1"]
        p["run_objs"] = [{**r, "code": p["team"]} for r in runs if r["code"] in TEAMS and r["i0"] <= p["i1"]
                         and r["i1"] >= p["i0"]] if p["team"] in TEAMS else []
        p["passes"] = sum(1 for a, b in zip(p["owners"], p["owners"][1:]) if a != b and a and b)
        p["duration_s"] = round(p["t1"] - p["t0"], 2)
        p["seen_s"] = round(p["held"] / gi.fps, 2)
        p["coverage"] = round(p["held"] / max(1, p["i1"] - p["i0"] + 1), 2)
        p["n_segments"] = len(p["segments"])
    return out


def _start_label(s: dict) -> str:
    k = s.get("kind")
    return {"make": "after made basket", "shot": "after shot", "free_throws": "after free throws", "start": "start",
            "holder": "change"}.get(k, "after dead ball")


# ====================================================================================== breaks / periods
def breaks_from_inplay(inplay: dict | None, t0: float, t1: float, cfg: GSConfig) -> list[dict]:
    if not isinstance(inplay, dict):
        return []
    iv = sorted([(float(a), float(b)) for a, b in (inplay.get("intervals") or [])])
    out = []
    for (a0, a1), (b0, b1) in zip(iv, iv[1:]):
        if b0 - a1 <= 1.0:
            continue
        dur = b0 - a1
        kind = "timeout" if dur < cfg.timeout_max_s else "quarter_break"
        out.append({"t0": round(a1, 2), "t1": round(b0, 2), "duration_s": round(dur, 1), "kind": kind})
    if out:
        longest = max(out, key=lambda b: b["duration_s"])
        others = sorted((b["duration_s"] for b in out if b is not longest), reverse=True)
        if longest["duration_s"] >= cfg.halftime_min_s or (
                others and longest["duration_s"] >= cfg.halftime_rel_min_s
                and longest["duration_s"] >= cfg.halftime_rel_factor * others[0]):
            longest["kind"] = "half_time"
        elif iv[-1][1] - iv[0][0] >= cfg.halftime_mid_min_span_s:
            # long video, no clearly longest break (inplay may also split the half-time break or cut it short):
            # the quarter-break-like break nearest the middle of the game, if it lies in the middle third
            a0, a1 = iv[0][0], iv[-1][1]
            mid = 0.5 * (a0 + a1)
            cand = [b for b in out if b["duration_s"] >= cfg.halftime_mid_min_s
                    and a0 + (a1 - a0) / 3 <= 0.5 * (b["t0"] + b["t1"]) <= a1 - (a1 - a0) / 3]
            if cand:
                min(cand, key=lambda b: abs(0.5 * (b["t0"] + b["t1"]) - mid))["kind"] = "half_time"
    return out


def halves(t0: float, t1: float, brks: list[dict]) -> list[dict]:
    ht = next((b for b in brks if b["kind"] == "half_time"), None)
    if ht is None:
        return [{"t0": t0, "t1": t1, "label": "whole analysed range (no half-time break found)", "index": 0}]
    return [{"t0": t0, "t1": ht["t0"], "label": "before half time", "index": 0},
            {"t0": ht["t1"], "t1": max(t1, ht["t1"]), "label": "after half time", "index": 1}]


# ====================================================================================== attack direction
def attack_direction(gi: GameInput, parts: list[dict], fts: list[dict], shots: list[dict], baskets: list[dict],
                     poss: list[dict]) -> list[dict]:
    """Votes for "team A attacks R" (+) / "A attacks L" (-) per half; halves are coupled (teams switch).

    Strong evidence (free throws, made baskets) dominates; weak evidence (missed-shot holders, half-court
    possessions - both depend on who the tracker thinks holds the ball) only decides when strong evidence is
    missing or tied."""
    sgn = {"R": 1.0, "L": -1.0}
    strong_keys = ("free_throws", "made_baskets")
    votes = [{"strong": [0.0, 0.0], "weak": [0.0, 0.0],
              "evidence": {"free_throws": 0, "made_baskets": 0, "shots": 0, "half_court_possessions": 0}}
             for _ in parts]

    def part_of(t: float) -> int | None:
        for k, p in enumerate(parts):
            if p["t0"] - 1 <= t <= p["t1"] + 1:
                return k
        return None

    def vote(t: float, team: str | None, side: str | None, w: float, key: str) -> None:
        if team not in TEAMS or side not in sgn:
            return
        k = part_of(t)
        if k is None:
            return
        s = sgn[side] * (1.0 if team == "A" else -1.0)
        acc = votes[k]["strong" if key in strong_keys else "weak"]
        acc[0] += w * s
        acc[1] += w
        votes[k]["evidence"][key] += 1

    for w in fts:
        vote(w["t0"], w.get("team"), w["side"], 6.0 * w.get("confidence", 0.8), "free_throws")
    for e in baskets:
        vote(_f(e.get("t")) or 0.0, e.get("team"), e.get("basket"), 2.0 * (_f(e.get("confidence")) or 0.5),
             "made_baskets")
    for sh in shots:
        if sh.get("made") or sh.get("ft"):
            continue
        vote(float(gi.t[sh["i_rim"]]), sh.get("team"), sh["side"], 0.5 * sh["confidence"], "shots")
    half = gi.L / 2
    for p in poss:
        if p["duration_s"] < 4.0:
            continue
        idx = range(p["i0"], p["i1"] + 1)
        bx = gi.ball_xy[list(idx), 0]
        bx = bx[np.isfinite(bx)]
        if len(bx) < 5:
            continue
        in_r = float(np.mean(bx > half))
        if 0.25 < in_r < 0.75:
            continue
        side = "R" if in_r >= 0.75 else "L"
        # most visible players in that half too (a set offence, not a ball held in the back court)
        lo = np.searchsorted(gi.r_i, p["i0"])
        hi = np.searchsorted(gi.r_i, p["i1"], "right")
        xs = gi.r_xy[lo:hi, 0][np.isin(gi.r_team[lo:hi], TEAMS)]
        if len(xs) < 10:
            continue
        share = float(np.mean(xs > half)) if side == "R" else float(np.mean(xs <= half))
        if share < 0.65:
            continue
        w = 0.5 * min(3.0, p["duration_s"] / 5.0) * p["confidence"]
        vote(p["t0"], p["team"], side, w, "half_court_possessions")
    out = []
    for k, p in enumerate(parts):
        def comb(kind: str) -> tuple[float, float]:
            own = votes[k][kind]
            sc = own[0] - 0.8 * sum(v[kind][0] for j, v in enumerate(votes) if j != k)  # teams switch at half time
            tot = own[1] + 0.8 * sum(v[kind][1] for j, v in enumerate(votes) if j != k)
            return sc, tot
        ss, st = comb("strong")
        ws, wt = comb("weak")
        wk = 0.25 if st >= 2.0 and abs(ss) >= 0.3 * st else 1.0
        score, total = ss + wk * ws, st + wk * wt
        if total <= 0 or abs(score) < 1e-9:
            a_side, conf = None, 0.0
        else:
            a_side = "R" if score > 0 else "L"
            conf = float(min(0.99, (abs(score) / total) * min(1.0, total / 3.0)))
        out.append({"t0": round(p["t0"], 2), "t1": round(p["t1"], 2), "label": p["label"],
                    "A": a_side, "B": None if a_side is None else ("L" if a_side == "R" else "R"),
                    "confidence": round(conf, 2), "evidence": votes[k]["evidence"]})
    return out


def attacked_side(direction: list[dict], team: str | None, t: float) -> str | None:
    if team not in TEAMS:
        return None
    for d in direction:
        if d["t0"] - 1 <= t <= d["t1"] + 1 and d.get(team):
            return d[team]
    return None


def attacker_of(direction: list[dict], side: str | None, t: float, min_conf: float) -> str | None:
    """The team attacking basket ``side`` at time t (None when the direction is unknown or too unsure)."""
    if side not in ("L", "R"):
        return None
    d = next((d for d in direction if d["t0"] - 1 <= t <= d["t1"] + 1), None)
    if d is None or not d.get("A") or d["confidence"] < min_conf:
        return None
    return "A" if d["A"] == side else "B"


def ball_stays(gi: GameInput, dead: list[tuple[float, float, str]], breaks: Sequence[int] = (),
               max_unknown_s: float = 1.5) -> np.ndarray:
    """Per frame: seconds since the ball's continuous live stay in its current half began (0 = unknown).

    Brief excursions (< 1 s) and short unknown stretches do not break a stay; dead balls, unknown stretches
    longer than ``max_unknown_s`` and the frames in ``breaks`` (shots reaching the rim: after a shot either team
    may get the ball) do."""
    half = gi.L / 2
    side = np.where(np.isfinite(gi.ball_xy[:, 0]), np.where(gi.ball_xy[:, 0] > half, 1, -1), 0)
    for a, b, _ in dead:
        side[gi.idx(a):gi.idx(b, "right")] = 9
    for i in breaks:
        if 0 <= i < gi.n:
            side[i] = 9
    out = np.zeros(gi.n)
    i = 0
    while i < gi.n:
        if side[i] not in (1, -1):
            i += 1
            continue
        s, j, last = side[i], i, i
        while j + 1 < gi.n:
            nx = side[j + 1]
            if nx == s:
                last = j + 1
                j += 1
                continue
            if nx == 9:
                break
            # unknown or the other half: does the ball come back to this half soon (and within the allowed gap)?
            k = j + 1
            while k < gi.n and side[k] in (-s, 0):
                k += 1
            if k < gi.n and side[k] == s:
                other = np.any(side[j + 1:k] == -s)
                gap = gi.t[k] - gi.t[last]
                if (other and gap < 1.0) or (not other and gap <= max_unknown_s):
                    j = k
                    last = k
                    continue
            break
        el = gi.t[i:last + 1] - gi.t[i]
        out[i:last + 1] = np.where(side[i:last + 1] == s, np.maximum(el, 1e-3), 0.0)
        i = last + 1
    return out


def _run_votes(gi: GameInput, r: dict, direction: list[dict], stays: np.ndarray, cfg: GSConfig,
               shots: list[dict] | None) -> list[tuple[int, str]]:
    """(frame, attacker team) for the frames of control run ``r`` where the ball location decides the team."""
    out = []
    rim_t = {s: np.sort([gi.t[sh["i_rim"]] for sh in shots or [] if sh["side"] == s]) for s in "LR"}
    for i in range(r["i0"], r["i1"] + 1):
        if not np.isfinite(gi.ball_xy[i, 0]) or gi.ball_f[i] != 1:
            continue
        side = "R" if gi.ball_xy[i, 0] > gi.L / 2 else "L"
        rt = rim_t[side]
        k = int(np.searchsorted(rt, gi.t[i] - 0.3))
        shot_next = k < len(rt) and rt[k] <= gi.t[i] + 2.5
        if stays[i] < cfg.relabel_stay_s and not shot_next:
            continue
        d = next((d for d in direction if d["t0"] - 1 <= gi.t[i] <= d["t1"] + 1), None)
        if d is None or not d.get("A") or d["confidence"] < cfg.direction_min_conf:
            continue
        out.append((i, "A" if d["A"] == side else "B"))
    return out


def track_team_corrections(gi: GameInput, runs: list[dict], direction: list[dict],
                           dead: list[tuple[float, float, str]], cfg: GSConfig,
                           shots: list[dict] | None = None) -> dict[str, str]:
    """Tracks whose team label is contradicted by where they hold the ball: a player who keeps holding the ball
    in the half that the OTHER team attacks (after the ball has been there for more than 8 s, or right before
    a shot at that basket) is on the other team. Similar jersey colours make such mistakes common."""
    stays = ball_stays(gi, dead, [sh["i_rim_end"] for sh in shots or []])
    votes: dict[str, list[int]] = {}
    against_t: dict[str, list[float]] = {}
    holds: dict[str, int] = {}
    for i in range(gi.n):
        if gi.ball_f[i] == 1 and gi.ball_o[i] is not None:
            holds[gi.ball_o[i]] = holds.get(gi.ball_o[i], 0) + 1
    for r in runs:
        if r["code"] not in TEAMS:
            continue
        for i, att in _run_votes(gi, r, direction, stays, cfg, shots):
            tid = gi.ball_o[i]
            if tid is None:
                continue
            v = votes.setdefault(tid, [0, 0])
            ok = att == gi.track_team.get(tid)
            v[0 if ok else 1] += 1
            if not ok:
                against_t.setdefault(tid, []).append(float(gi.t[i]))
    out = {}
    for tid, (agree, against) in votes.items():
        ts = against_t.get(tid) or [0.0]
        spread = max(ts) - min(ts)  # evidence from more than one moment, not one ambiguous change of possession
        if against >= cfg.track_flip_min_frames and against >= cfg.track_flip_share * (agree + against) and \
                against >= 0.25 * holds.get(tid, 0) and spread >= cfg.track_flip_spread_s:
            out[tid] = "B" if gi.track_team.get(tid) == "A" else "A"
    return out


def relabel_runs(gi: GameInput, runs: list[dict], direction: list[dict], dead: list[tuple[float, float, str]],
                 cfg: GSConfig, shots: list[dict] | None = None) -> int:
    """Correct control runs whose holder team contradicts where the ball is: a ball that stays in one half in
    live play belongs to the team attacking that half (the back court may be held for at most 8 s). Team labels
    of tracks are imperfect (similar jerseys), so this beats a single holder's label when the stay is long."""
    stays = ball_stays(gi, dead, [sh["i_rim_end"] for sh in shots or []])
    n = 0
    for r in runs:
        if r["code"] not in TEAMS:
            continue
        t_mid = float(gi.t[(r["i0"] + r["i1"]) // 2])
        d = next((d for d in direction if d["t0"] - 1 <= t_mid <= d["t1"] + 1), None)
        if d is None or not d.get("A") or d["confidence"] < cfg.direction_min_conf:
            continue
        idx = np.arange(r["i0"], r["i1"] + 1)
        bx = gi.ball_xy[idx, 0]
        ok = np.isfinite(bx) & (stays[idx] > 0)
        if ok.sum() < max(2, 0.5 * len(idx)):
            continue
        right = float(np.mean(bx[ok] > gi.L / 2))
        if 0.2 < right < 0.8:
            continue
        side = "R" if right >= 0.8 else "L"
        att = "A" if d["A"] == side else "B"
        stay = float(np.median(stays[idx][ok]))  # seconds the ball has already been in this half
        shot_next = any(sh["side"] == side and gi.t[r["i1"]] - 0.5 <= gi.t[sh["i_rim"]] <= gi.t[r["i1"]] + 3.0
                        for sh in shots or [])
        if att != r["code"] and (stay >= cfg.relabel_stay_s or shot_next):
            r["code_from_holder"] = r["code"]
            r["code"] = att
            n += 1
    return n


# ====================================================================================== lineup
def lineup(gi: GameInput, cfg: GSConfig) -> dict:
    """Assign player tracks to 5 on-court slots per team (see module docstring)."""
    slots: dict[str, list[dict]] = {k: [{"label": f"{k}{j + 1}", "tracks": [], "last_i": None, "last_xy": None,
                                         "number": None} for j in range(5)] for k in TEAMS}
    extras: list[str] = []
    slot_of: dict[str, str] = {}
    info = []
    for tid, rows in gi.track_rows.items():
        team = gi.track_team.get(tid)
        if team not in TEAMS or not len(rows):
            continue
        meas = rows[gi.r_meas[rows]]
        if not len(meas):
            continue
        info.append((int(gi.r_i[rows[0]]), int(gi.r_i[rows[-1]]), tid, team, rows))
    info.sort(key=lambda r: (r[0], -len(r[4])))
    for i0, i1, tid, team, rows in info:
        best, bcost = None, math.inf
        p0 = gi.r_xy[rows[0]]
        num = gi.track_number.get(tid)
        for s in slots[team]:
            if num is not None and s["number"] is not None and s["number"] != num:
                continue  # jersey numbers differ: another player
            if s["last_i"] is not None and s["last_i"] >= i0:
                # occupied: allow a short overlap with a near position (same person, new ID)
                ov = gi.t[s["last_i"]] - gi.t[i0]
                q = _track_xy(gi, s["tracks"][-1][0], i0)
                if ov <= 0.6 and q is not None and np.linalg.norm(q - p0) < 1.2:
                    cost = 0.0
                else:
                    continue
            elif s["last_i"] is None:
                cost = 6.0
            else:
                dt = gi.t[i0] - gi.t[s["last_i"]]
                d = float(np.linalg.norm(p0 - s["last_xy"]))
                cost = max(0.0, d - (1.0 + cfg.reach_mps * dt)) + 0.05 * d + 0.02 * dt
            if num is not None and s["number"] == num:
                cost -= 10.0  # same jersey number: the same player (strongest cue)
            if cost < bcost:
                best, bcost = s, cost
        if best is None:
            busy = [s for s in slots[team] if s["tracks"] and s["last_i"] is not None and s["last_i"] >= i0]
            short = min(busy, key=lambda s: len(gi.track_rows[s["tracks"][-1][0]]), default=None)
            n_short = len(gi.track_rows[short["tracks"][-1][0]]) if short is not None else 0
            if short is not None and len(rows) >= 2 * n_short and gi.t[i1] - gi.t[i0] >= cfg.lineup_evict_s:
                old_tid = short["tracks"].pop()[0]  # the short track becomes the extra one
                extras.append(old_tid)
                slot_of.pop(old_tid, None)
                prev = short["tracks"][-1] if short["tracks"] else None
                short["last_i"] = prev[2] if prev else None
                short["last_xy"] = gi.r_xy[gi.track_rows[prev[0]][-1]] if prev else None
                best = short
            else:
                extras.append(tid)
                continue
        best["tracks"].append((tid, i0, i1))
        best["last_i"], best["last_xy"] = i1, gi.r_xy[rows[-1]]
        if num is not None:
            best["number"] = num
        slot_of[tid] = best["label"]
    out = {"slots": {s["label"]: [[tid, round(float(gi.t[a]), 2), round(float(gi.t[b]), 2)]
                                 for tid, a, b in s["tracks"]] for k in TEAMS for s in slots[k]},
           "slot_of_track": slot_of, "extra_tracks": sorted(extras, key=_id_key)}
    # jersey number per slot (jersey_numbers.py): the number of the slot's player once any of its tracks was read
    out["numbers"] = {s["label"]: s["number"] for k in TEAMS for s in slots[k] if s["number"] is not None}
    return out


def _id_key(tid: str) -> tuple[int, str]:
    return (int(tid), "") if str(tid).isdigit() else (10 ** 9, str(tid))


def lineup_at(gi: GameInput, lu: dict, i: int) -> dict[str, list[dict]]:
    """Per team: the 5 slots at frame i: visible (with x, y) or off screen (last seen x, y, seconds ago)."""
    out: dict[str, list[dict]] = {k: [] for k in TEAMS}
    for label, tracks in lu["slots"].items():
        team = label[0]
        cur, last = None, None
        for tid, a, b in tracks:
            p = _track_xy(gi, tid, i)
            if p is not None:
                cur = (tid, p)
                break
            if b <= gi.t[i]:
                q = _track_xy(gi, tid, int(np.searchsorted(gi.t, b - 1e-6)))
                if q is not None:
                    last = (tid, q, b)
        num = (lu.get("numbers") or {}).get(label)
        if cur is not None:
            out[team].append({"slot": label, "visible": True, "x": round(float(cur[1][0]), 1),
                              "y": round(float(cur[1][1]), 1), "track": cur[0]})
        elif last is not None:
            out[team].append({"slot": label, "visible": False, "last_x": round(float(last[1][0]), 1),
                              "last_y": round(float(last[1][1]), 1), "seen_s_ago": round(float(gi.t[i] - last[2]), 1)})
        else:
            out[team].append({"slot": label, "visible": False, "last_x": None, "last_y": None, "seen_s_ago": None})
        if num is not None:
            out[team][-1]["number"] = num
    return out


# ====================================================================================== main analysis
def analyse(replay: dict, inplay: dict | None = None, rim_ball: list | None = None,
            cfg: GSConfig | None = None, names: dict[str, str] | None = None, scorebar: dict | None = None,
            make_teams: dict | None = None, extra_makes: list | None = None, off_sheet: set | None = None,
            v6: bool | None = None, corrections: dict | None = None) -> dict:
    """Game state of a Replay (see module docstring). ``replay`` = replay_data.json content.

    ``v6`` (default: env REPLAY_POSSESSION_V6) switches to the possession model v6; it then uses the score bar
    (``scorebar`` = work/scorebar.json result) or the scoresheet alignment of a previous pass (``make_teams``
    = {basket id: (team, points)}, ``extra_makes`` = sheet events without a detection, ``off_sheet`` = detected
    makes the sheet does not have) for the possession boundaries. ``corrections`` = the coach's corrections
    ledger (corrections.py; v6 only): makes and possession anchors as hard constraints."""
    cfg = cfg or GSConfig()
    use_v6 = possession_v6_enabled() if v6 is None else bool(v6)
    gi = load_input(replay)
    if gi.n == 0:
        return _empty(gi)
    team_names = names or {k: str((gi.teams.get(k) or {}).get("name") or k) for k in TEAMS}
    lo, hi = _frame_slices(gi)
    hold = holder_codes(gi)
    motion = motion_level(gi, lo, hi)
    baskets = [e for e in (replay.get("events") or []) if isinstance(e, dict) and e.get("type") == "basket"]

    # --- dead balls: breaks, free throws, referee holding, stoppages
    t0, t1 = float(gi.t[0]), float(gi.t[-1])
    brks = [b for b in breaks_from_inplay(inplay, t0, t1, cfg) if b["t1"] > t0 and b["t0"] < t1]
    shooter = ft_formation(gi, lo, hi, cfg)
    hits = _rim_hits(gi, rim_ball, hold)
    fts = ft_windows(gi, shooter, cfg, motion, hits)
    dead: list[tuple[float, float, str]] = [(b["t0"], b["t1"], b["kind"]) for b in brks]
    v6cfg = PossessionV6Config() if use_v6 else None
    shots = shot_attempts(gi, hits, hold, baskets, cfg) if use_v6 else None

    def ft_guard(w: dict) -> None:
        """v6 (C5): the set-up never reaches back over a live shot / make that is not part of the free throws."""
        if not use_v6:
            return
        live = [float(gi.t[sh["i_rim_end"]]) for sh in shots
                if not any(x["side"] == sh["side"] and x["t0"] - 1.0 <= gi.t[sh["i_rim"]] <= x["t1"] + 4.0 for x in fts)
                and gi.t[sh["i_rim"]] < w["t0"]]
        live += [float(e["t"]) for e in baskets if _f(e.get("t")) is not None and float(e["t"]) < w["t0"] - 1.0
                 and not any(x["side"] == e.get("basket") and x["t0"] - 1.0 <= float(e["t"]) <= x["t1"] + 4.0 for x in fts)]
        if live:
            w["t_setup"] = round(max(w["t_setup"], min(max(live) + v6cfg.ft_setup_guard_s, w["t0"])), 2)

    for w in fts:  # the set-up before the first attempt (walking to the lane, ball to the shooter) is dead too
        i = w["i0"]
        while i > 0 and gi.t[w["i0"]] - gi.t[i - 1] <= cfg.ft_setup_max_s and                 not (np.isfinite(motion[i - 1]) and motion[i - 1] > cfg.ft_setup_motion):
            i -= 1
        w["t_setup"] = round(float(min(gi.t[i], w["t0"] - cfg.ft_pad_s)), 2)
        ft_guard(w)
    ref_runs = [r for r in control_runs(gi, hold, cfg) if r["code"] == "R"
                and gi.t[r["i1"]] - gi.t[r["i0"]] >= cfg.ref_hold_min_s]  # brief = referee only next to the ball
    for r in ref_runs:
        dead.append((float(gi.t[r["i0"]]), float(gi.t[r["i1"]]), "referee_has_ball"))
    # long still periods with nobody holding the ball (time-outs inside play intervals, injuries, substitutions)
    still = np.isfinite(motion) & (motion < cfg.still_mps) & np.array([h not in TEAMS for h in hold])
    still = _close_gaps(still, gi.t, 2.0)
    stoppages = []
    for a, b in _runs(still):
        ta, tb = float(gi.t[a]), float(gi.t[b - 1])
        for w in fts:  # a stoppage running into free throws is their set-up (foul, substitutions, time-out)
            if ta <= w["t1"] + 3 and tb >= w["t_setup"]:
                if ta < w["t_setup"] and tb - ta >= cfg.stoppage_min_s:
                    w["t_setup"] = round(ta, 2)
                    tb = ta
                elif ta < w["t_setup"]:
                    tb = min(tb, w["t_setup"])
                else:
                    ta = max(ta, w["t1"] + 3)
        if tb - ta >= cfg.stoppage_min_s:
            stoppages.append((ta, tb))
            dead.append((ta, tb, "stoppage"))
    for w in fts:
        ft_guard(w)  # (a long stoppage may have moved the set-up back again)
        w["t_end"] = w["t1"] + 0.2
        if use_v6:  # v6: the dead ball lasts until the last attempt's rim contact (a made last free throw too)
            last = [float(gi.t[sh["i_rim_end"]]) for sh in shots
                    if sh["side"] == w["side"] and w["t0"] - 1.0 <= gi.t[sh["i_rim"]] <= w["t1"] + 4.0]
            last += [float(e["t"]) for e in baskets if e.get("basket") == w["side"] and _f(e.get("t")) is not None
                     and w["t0"] - 1.0 <= float(e["t"]) <= w["t1"] + 4.0]
            if last:
                w["t_end"] = round(max(w["t_end"], max(last) + 0.5), 2)
    dead += [(w["t_setup"], w["t_end"], "free_throws") for w in fts]
    dead.sort()

    # --- shots (incl. free-throw attempts)
    if shots is None:
        shots = shot_attempts(gi, hits, hold, baskets, cfg)
    for sh in shots:
        tr = float(gi.t[sh["i_rim"]])
        w = next((w for w in fts if w["side"] == sh["side"] and w["t0"] - 1.0 <= tr <= w["t1"] + 4.0), None)
        sh["ft"] = w is not None
        if w is not None:
            sh["ft_window"] = fts.index(w)
            if w.get("team"):
                sh["team"] = w["team"]
            if w.get("shooter_tracks"):
                sh["shooter"] = w["shooter_tracks"][0]

    # --- possessions (pass 1: holder teams), attack direction, then pass 2: location-corrected teams
    runs = control_runs(gi, hold, cfg)
    poss = possessions(gi, runs, dead, cfg, [float(gi.t[sh['i_rel']]) for sh in shots])
    parts = halves(t0, t1, brks)
    direction = attack_direction(gi, parts, fts, shots, baskets, poss)
    flips = track_team_corrections(gi, runs, direction, dead, cfg, shots)
    flip_log = {tid: (gi.track_team.get(tid), team) for tid, team in flips.items()}
    if flips:
        for tid, team in flips.items():
            gi.track_team[tid] = team
            rows = gi.track_rows.get(tid)
            if rows is not None:
                gi.r_team[rows] = team
        hold = holder_codes(gi)
        runs = control_runs(gi, hold, cfg)
    n_relabelled = relabel_runs(gi, runs, direction, dead, cfg, shots)
    if n_relabelled or flips:
        poss = possessions(gi, runs, dead, cfg, [float(gi.t[sh['i_rel']]) for sh in shots])
    for p in poss:
        p["attacks"] = attacked_side(direction, p["team"], p["t0"])
    # a shot at basket X is taken by the team attacking X: once the direction is known it beats the holder
    # assignment (the "last holder" before a shot is often a defender next to the shooter)
    for sh in shots:
        att = attacker_of(direction, sh["side"], float(gi.t[sh["i_rim"]]), cfg.basket_direction_min_conf)
        if att is None:
            continue
        if sh.get("team") != att:
            sh["team_from_holder"] = sh.get("team")
            sh["team"] = att
            sh["team_source"] = "attack direction"
        # shooter = the last holder of the ATTACKING team before the ball reached the rim (the last holder of
        # any team is often a defender next to the shooter or a duplicate track)
        if not sh.get("ft") and (sh.get("shooter") is None or gi.track_team.get(str(sh["shooter"])) != att):
            i0 = sh["i_rim"]
            for j in range(i0, max(-1, i0 - int(round(cfg.release_window_s * gi.fps))), -1):
                if hold[j] != att:
                    continue
                q = _track_xy(gi, gi.ball_o[j], j)
                if q is None or (np.linalg.norm(q - gi.basket(sh["side"])) < 0.9 and j > i0 - 3):
                    continue
                sh["shooter_from_holder"] = sh.get("shooter")
                sh["shooter"] = gi.ball_o[j]
                if sh.get("pos") is None or j >= sh["i_rel"]:  # an earlier holder may be the passer: keep the
                    sh["i_rel"] = j  # later release position for the shot distance / 2-or-3 guess
                    sh["pos"] = [round(float(v), 2) for v in q]
                    sh["dist"] = round(float(np.linalg.norm(q - gi.basket(sh["side"]))), 2)
                break

    # --- possession model v6: boundaries (bar / makes / shots / dead balls / holder changes) + evidence vote;
    # one model owns the team of possessions, shots and makes (data/plans/possession_design.md)
    make_info: dict = {}
    if use_v6:
        makes, make_info = v6_makes(gi, baskets, shots, direction, cfg, v6cfg, scorebar, make_teams, extra_makes,
                                    off_sheet)
        from replay import corrections as _corr
        user = _corr.apply_events(corrections)
        if user:
            makes = v6_user_makes(makes, user)
            make_info["user_makes"] = len(user["makes"])
            make_info["user_anchors"] = len(user["anchors"])
        poss = possessions_v6(gi, hold, runs, shots, makes, dead, fts, direction, cfg, v6cfg, user)
        src_name = {"bar": "score bar", "sheet": "scoresheet", "direction": "attack direction", "user": "coach"}
        for m in makes:
            sh = m.get("shot")
            if sh is None or m["basket_id"] is None or m["team"] not in TEAMS:
                continue
            if sh.get("team") != m["team"]:
                sh["team_from_holder"] = sh.get("team")
                sh["team"] = m["team"]
            sh["team_source"] = src_name.get(m["source"], sh.get("team_source", "last holder"))
            if m["source"] in ("bar", "sheet"):
                sh["points_known"] = m["points"]
        for p in poss:
            p["attacks"] = attacked_side(direction, p["team"], p["t0"])
        for sh in shots:  # shots the direction could not credit: the team of the possession they lie in
            if sh.get("ft") or sh.get("team_source") in ("attack direction", "score bar", "scoresheet"):
                continue
            tr = float(gi.t[sh["i_rim"]])
            p = next((p for p in poss if p["kind"] == "live" and p["t0"] - 0.05 <= tr <= p["t1"] + 0.3), None)
            if p is not None and p["team"] in TEAMS and p["team"] != sh.get("team") and p["confidence"] >= 0.5:
                sh["team_from_holder"] = sh.get("team")
                sh["team"] = p["team"]
                sh["team_source"] = "possession"

    events: list[dict] = []
    annotations: dict[str, dict] = {}

    def add(typ: str, i: int, team: str | None = None, conf: float = 0.5, duration: float = 0.0,
            basket: str | None = None, label: str | None = None, suffix: str = "", **details: Any) -> dict:
        ev = {"id": f"gs-{typ}-{int(gi.frame[i])}", "type": typ, "t": round(float(gi.t[i]), 3),
              "frame": int(gi.frame[i]), "duration": round(max(0.0, duration), 2), "team": team,
              "basket": basket, "confidence": round(float(np.clip(conf, 0, 1)), 2), "source": SOURCE,
              "details": {k: v for k, v in details.items() if v is not None}}
        ev["label"] = label or TITLES.get(typ, typ.replace("_", " ").capitalize())
        ev["suffix"] = suffix if suffix else (f"({basket})" if basket else "")
        ev["title"] = event_title(ev, team_names)
        events.append(ev)
        return ev

    for k, w in enumerate(fts):
        att = [sh for sh in shots if sh.get("ft_window") == k]
        for j, sh in enumerate(att, 1):
            sh["ft_k"], sh["ft_n"] = j, len(att)

    # shots, made-basket annotations, rebounds
    for sh in shots:
        if sh.get("ft"):
            continue
        team = sh.get("team")
        val = None if sh["dist"] is None else ("3" if sh["dist"] >= gi.spec["three_r"] + 0.25 else
                                               "2" if sh["dist"] <= gi.spec["three_r"] - 0.4 else "2/3")
        sh["value"] = val
        if sh.get("made") and sh.get("basket_id"):
            lab = {"3": "Made 3?", "2": "Made 2", "2/3": "Made 2/3?", None: "Made basket"}[val]
            annotations[sh["basket_id"]] = {"label": lab, "suffix": f"({sh['side']})",
                                            "title": f"{lab} – {_tn(team, team_names)} ({sh['side']})",
                                            "points_guess": sh.get("points_known") or (3 if val == "3" else 2),
                                            "kind": "field_goal",
                                            "team": team, "team_source": sh.get("team_source", "last holder"),
                                            "shot_distance_m": sh["dist"], "shooter_track": sh["shooter"]}
            continue
        add("shot", sh["i_rel"], team, sh["confidence"], basket=sh["side"],
            suffix=f"({sh['side']}" + (f", {val}pt?" if val and val != "2/3" else "") + ")",
            shooter_track=sh.get("shooter"), distance_m=sh["dist"], position=sh["pos"],
            t_at_rim=round(float(gi.t[sh["i_rim"]]), 2))

    # rebounds: first control after a missed attempt (last FT of a sequence or a field goal)
    poss_runs = sorted((r for p in poss for r in p.get("run_objs", [])), key=lambda r: r["i0"])
    for sh in shots:
        if sh.get("made") or (sh.get("ft") and sh.get("ft_k") != sh.get("ft_n")):
            continue
        t_r = float(gi.t[sh["i_rim_end"]])
        nxt = next((r for r in poss_runs if gi.t[r["i0"]] > t_r - 0.1
                    and gi.t[r["i0"]] - t_r <= cfg.rebound_window_s), None)
        sh["missed"] = None
        if nxt is None:
            continue
        # a made basket (undetected) is followed by an inbound from the baseline, not a rebound
        p = _track_xy(gi, nxt["owners"][0], nxt["i0"])
        if p is not None and (p[0] < 0.4 or p[0] > gi.L - 0.4) and gi.t[nxt["i0"]] - t_r > 2.0:
            continue
        team = sh.get("team")
        typ = "rebound" if team is None else "off_rebound" if nxt["code"] == team else "def_rebound"
        conf = 0.4 + 0.4 * sh["confidence"]
        suffix = f"({sh['side']})"
        if typ == "def_rebound" and p is not None and np.linalg.norm(p - gi.basket(sh["side"])) > \
                cfg.rebound_max_dist_m and gi.t[nxt["i0"]] - t_r > 1.0:
            # first control far from the basket, a while after the shot: as likely the inbound pass after a
            # made basket that the basket detector missed
            suffix = f"({sh['side']}, or inbound after a made basket?)"
            conf *= 0.6
        sh["missed"] = True
        sh["rebound_team"] = nxt["code"]
        if any(e["type"] in ("off_rebound", "def_rebound", "rebound") and e["frame"] == int(gi.frame[nxt["i0"]])
               for e in events):
            continue
        add(typ, nxt["i0"], nxt["code"], conf, basket=sh["side"], suffix=suffix,
            rebounder_track=nxt["owners"][0], after_shot_t=round(float(gi.t[sh["i_rel"]]), 2))

    # free throws
    for k, w in enumerate(fts):
        att = [sh for sh in shots if sh.get("ft_window") == k]
        team = w.get("team") or next((sh["team"] for sh in att if sh.get("team")), None)
        # free throws are shot at the basket the shooter's team attacks: the direction beats the label of the
        # player standing at the line (similar jerseys, track mix-ups)
        team = attacker_of(direction, w["side"], w["t0"], cfg.basket_direction_min_conf) or team
        i0, i1 = w["i0"], w["i1"]
        n_att = len(att)
        add("free_throws", i0, team, w["confidence"], duration=w["t1"] - w["t0"], basket=w["side"],
            suffix=f"({w['side']}, " + (f"{n_att} attempt(s) seen)" if n_att else "attempts not seen)"),
            shooter_track=w["shooter_tracks"][0] if w.get("shooter_tracks") else None, attempts_seen=n_att,
            note="Dead ball: one player at the free-throw line, the others along the lane. Ball moves between "
                 "referee and shooter are NOT passes.")
        for j, sh in enumerate(att, 1):
            made = bool(sh.get("made"))
            sh["ft_k"], sh["ft_n"] = j, n_att
            if made and sh.get("basket_id"):
                annotations[sh["basket_id"]] = {"label": f"Made free throw {j}/{n_att}", "suffix": "",
                                                "title": f"Made free throw {j}/{n_att} – {_tn(team, team_names)}",
                                                "points_guess": 1, "kind": "free_throw", "team": team}
            add("free_throw", sh["i_rel"], team, min(sh["confidence"], w["confidence"]) + 0.1, basket=sh["side"],
                label=f"Free throw {j}/{n_att}",
                suffix="(made)" if made else "(missed)" if sh.get("missed") else "(result not seen)",
                made=made, shooter_track=sh.get("shooter"))

    # made baskets that no shot attempt explains: credit the team attacking that basket
    for e in baskets:
        bid, s, et = str(e.get("id")), e.get("basket"), _f(e.get("t"))
        if bid in annotations or s not in ("L", "R") or et is None:
            continue
        att = attacker_of(direction, s, et, cfg.basket_direction_min_conf)
        if att is not None:
            annotations[bid] = {"label": "Made basket", "suffix": f"({s})",
                                "title": f"Made basket – {_tn(att, team_names)} ({s})", "points_guess": 2,
                                "kind": "field_goal", "team": att, "team_source": "attack direction"}

    # turnovers / possession changes
    shot_t = [(float(gi.t[sh["i_rel"]]) - 0.3, float(gi.t[sh["i_rim_end"]]) + 0.3, sh) for sh in shots]
    made_t = [_f(e.get("t")) for e in baskets]
    for p, q in zip(poss, poss[1:]):
        if p["team"] == q["team"] or p["team"] not in TEAMS or q["team"] not in TEAMS or not q.get("owners"):
            continue
        if p.get("kind") == "free_throws" or q.get("kind") == "free_throws":
            continue
        if "outcome" in q and q.get("start") != "change":
            continue  # v6: the boundary (make / shot / dead ball) already explains the change of possession
        # v6 possessions tile the live play: the gap is between the last / first held frames
        a, b = p.get("t_held1", p["t1"]), q.get("t_held0", q["t0"])
        explained = None
        if any(sa <= b and sb >= a - 1.0 and sh.get("ft") for sa, sb, sh in shot_t):
            explained = "free throws"
        elif any(sa <= b and sb >= a - 1.0 for sa, sb, _ in shot_t):
            explained = "shot"
        elif any(sh.get("team") == p["team"] and a - cfg.after_shot_s <= sb - 0.3 <= b for _, sb, sh in shot_t):
            explained = "shot"  # rebound / inbound after the team's own shot or (undetected) basket
        elif any(mt is not None and a - 1.0 <= mt <= b + 0.5 for mt in made_t):
            explained = "made basket"
        elif any(da <= b and db >= a and kind in ("free_throws", "timeout", "quarter_break", "half_time")
                 for da, db, kind in dead):
            explained = "dead ball"
        q["start"] = {"shot": "after shot", "free throws": "after free throws", "made basket": "after made basket",
                      "dead ball": "after dead ball"}.get(explained or "", "change")
        if explained:
            continue
        gap = b - a
        ref_between = any(da <= b and db >= a and kind == "referee_has_ball" for da, db, kind in dead)
        pos_q = _track_xy(gi, q["owners"][0], q["i0"])
        inside = pos_q is not None and 0.8 < pos_q[0] < gi.L - 0.8 and 0.8 < pos_q[1] < gi.W - 0.8
        boundary = pos_q is not None and not (0.5 < pos_q[0] < gi.L - 0.5 and 0.5 < pos_q[1] < gi.W - 0.5)
        mot = motion[gi.idx(a):gi.idx(b) + 1]
        quiet = bool(np.isfinite(mot).any() and np.nanmedian(mot) < cfg.still_mps)
        conf = min(p["confidence"], q["confidence"])
        if gap <= cfg.steal_gap_s and inside and not ref_between:
            kind, conf = "steal", conf * 0.95
        elif ref_between or boundary or (gap >= cfg.dead_gap_s and quiet):
            kind, conf = "out_of_bounds_or_violation", conf * 0.8
        else:
            kind, conf = "lost_ball", conf * 0.7
        if q["duration_s"] < 0.8 or p["duration_s"] < 0.8:
            conf *= 0.6
        q["start"] = kind
        i_ev = p["i1"] if kind != "steal" else q["i0"]
        if kind == "steal":
            add("steal", q["i0"], q["team"], conf, suffix="(turnover by the other team)", lost_by=p["team"], stealer_track=q["owners"][0],
                gap_s=round(gap, 2))
        else:
            label = "out of bounds / violation?" if kind.startswith("out") else "lost ball"
            add("turnover", i_ev, p["team"], conf, suffix=f"({label})",
                kind=kind, next_team=q["team"], gap_s=round(gap, 2))

    # inbounds, fast breaks, half-court sets
    for k, q in enumerate(poss):
        prev = poss[k - 1] if k else None
        if q.get("kind") == "free_throws" or q["team"] not in TEAMS:
            continue
        pos_q = _track_xy(gi, q["owners"][0], q["i0"]) if q.get("owners") else None
        side = q.get("attacks")
        after_dead = prev is None or q.get("start") in ("after made basket", "after dead ball", "after free throws",
                                                         "out_of_bounds_or_violation") or \
            any(da <= q["t0"] <= db + 2.0 for da, db, kind in dead if kind != "referee_has_ball")
        if pos_q is not None and after_dead and (pos_q[0] < 0.6 or pos_q[0] > gi.L - 0.6 or pos_q[1] < 0.6
                                                  or pos_q[1] > gi.W - 0.6):
            add("inbound", q["i0"], q["team"], 0.5 * q["confidence"] + 0.2, position=[round(float(v), 1) for v in pos_q])
        if side is None:
            continue
        bx_side = gi.basket(side)[0]
        idx = np.arange(q["i0"], q["i1"] + 1)
        bx = gi.ball_xy[idx, 0]
        ok = np.isfinite(bx)
        if ok.sum() < 3:
            continue
        dist = np.abs(bx - bx_side)
        start_d = float(dist[ok][0])
        # fast break: starts in the back court after a change of possession, advances quickly
        if prev is not None and prev["team"] != q["team"] and start_d > gi.L / 2 - 1.0 and \
                q.get("start") in ("after shot", "steal", "lost_ball", "change", "after made basket"):
            lim = gi.t[q["i0"]] + cfg.fast_break_s
            sel = ok & (gi.t[idx] <= lim)
            if sel.any():
                adv = start_d - float(np.min(dist[sel]))
                if adv >= cfg.fast_break_m and float(np.min(dist[sel])) < 8.0:
                    j = idx[sel][int(np.argmin(dist[sel]))]
                    add("fast_break", q["i0"], q["team"], 0.45 + 0.35 * q["confidence"] * min(1.0, adv / 14.0),
                        duration=float(gi.t[j] - gi.t[q["i0"]]), basket=side, advance_m=round(adv, 1))
                    q["fast_break"] = True
        # half-court set: ball in the front court for a while
        front = ok & (dist < gi.L / 2)
        if front.any():
            j0 = int(np.argmax(front))
            t_cross = float(gi.t[idx[j0]])
            if q["t1"] - t_cross >= cfg.half_court_min_s and not q.get("fast_break"):
                add("half_court", idx[j0], q["team"], 0.3 + 0.5 * q["confidence"], duration=q["t1"] - t_cross,
                    basket=side, passes=q["passes"])
                q["crossed_half_s"] = round(t_cross - q["t0"], 1)

    # breaks and jump balls
    for bk in brks:
        i = min(gi.n - 1, gi.idx(bk["t0"]))
        add(bk["kind"], i, None, 0.6 if bk["kind"] == "timeout" else 0.5, duration=bk["duration_s"],
            label=TITLES[bk["kind"]] + ("?" if bk["kind"] != "timeout" else ""), suffix=f"({bk['duration_s']:.0f} s)")
    for a, b in stoppages:
        add("stoppage", gi.idx(a), None, 0.45, duration=b - a, suffix=f"({b - a:.0f} s, players still)")
    starts = [bk["t1"] for bk in brks if bk["kind"] in ("quarter_break", "half_time")]
    if isinstance(inplay, dict) and (inplay.get("intervals") or [[0]])[0][0] > 30.0:
        starts.insert(0, t0)  # the game starts after a pre-game break
    for ts in starts:
        i0 = gi.idx(ts)
        i1 = gi.idx(ts + 6.0)
        for i in range(i0, min(i1, gi.n)):
            a, b = lo[i], hi[i]
            xy = gi.r_xy[a:b][np.isin(gi.r_team[a:b], TEAMS)]
            if len(xy) < 6:
                continue
            d = np.linalg.norm(xy - np.array([gi.L / 2, gi.mid]), axis=1)
            if (d < 1.6).sum() >= 2 and (d < 6.0).sum() >= 6:
                add("jump_ball", i, None, 0.5, label="Jump ball?")
                break

    events.sort(key=lambda e: e["t"])
    _dedupe_ids(events)
    lu = lineup(gi, cfg)
    segs = segments(gi, poss, fts, brks, stoppages, shots, events)
    doc = {
        "format": FORMAT, "engine_version": ENGINE_VERSION_V6 if use_v6 else ENGINE_VERSION,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "court": {"length": gi.L, "width": gi.W, "baskets": {"L": gi.baskets[0].round(3).tolist(),
                                                             "R": gi.baskets[1].round(3).tolist()},
                  "free_throw_line_x": {"L": round(gi.ft_x("L"), 2), "R": round(gi.ft_x("R"), 2)},
                  "three_point_radius_m": gi.spec["three_r"]},
        "teams": {k: team_names[k] for k in TEAMS},
        "time_range_s": [round(t0, 2), round(t1, 2)],
        "breaks": brks,
        "halves": parts,
        "attack_direction": direction,
        "possessions": [_poss_out(gi, k, p) for k, p in enumerate(poss)],
        "free_throw_windows": [{k: w.get(k) for k in ("side", "team", "t_setup", "t0", "t1", "confidence", "shooter_tracks")}
                               for w in fts],
        "dead_balls": [{"t0": round(a, 2), "t1": round(b, 2), "kind": kind} for a, b, kind in dead],
        "shots": [_shot_out(gi, sh) for sh in shots],
        "events": events,
        "basket_annotations": annotations,
        "segments": segs,
        "lineup": lu,
        "team_corrections": {tid: {"from": old, "to": new} for tid, (old, new) in
                             sorted(flip_log.items(), key=lambda kv: _id_key(kv[0]))},
        "inputs": {"rim_ball": rim_ball or [], "has_inplay": bool(inplay)},
        "config": asdict(cfg),
    }
    if use_v6:
        doc["possession_model"] = {"version": ENGINE_VERSION_V6, **asdict(v6cfg), **make_info,
                                   "n_unclear": sum(1 for p in poss if p["team"] is None)}
    return doc


def _empty(gi: GameInput) -> dict:
    return {"format": FORMAT, "engine_version": ENGINE_VERSION, "events": [], "possessions": [], "segments": [],
            "attack_direction": [], "breaks": [], "halves": [], "dead_balls": [], "shots": [],
            "free_throw_windows": [], "basket_annotations": {},
            "lineup": {"slots": {}, "slot_of_track": {}, "extra_tracks": []}, "inputs": {"rim_ball": []}}


def _dedupe_ids(events: list[dict]) -> None:
    seen: dict[str, int] = {}
    for e in events:
        k = e["id"]
        if k in seen:
            seen[k] += 1
            e["id"] = f"{k}-{seen[k]}"
        else:
            seen[k] = 0


def _tn(team: str | None, names: dict[str, str]) -> str:
    return names.get(team, team) if team else "unknown team"


def _poss_out(gi: GameInput, k: int, p: dict) -> dict:
    out = {"id": f"p{k}", "team": p["team"], "t0": round(p["t0"], 2), "t1": round(p["t1"], 2),
           "duration_s": p["duration_s"], "attacks": p.get("attacks"), "start": p.get("start", "start"),
           "passes": p["passes"], "confidence": p["confidence"], "crossed_half_s": p.get("crossed_half_s"),
           "fast_break": bool(p.get("fast_break"))}
    if "outcome" in p:  # possession model v6
        out.update({"kind": p.get("kind", "live"), "outcome": p.get("outcome"), "points": int(p.get("points") or 0),
                    "shots": p.get("shots") or [], "off_rebounds": int(p.get("off_rebounds") or 0),
                    "evidence": p.get("evidence") or {}, "score": p.get("score"), "seen_s": p.get("seen_s"),
                    "coverage": p.get("coverage"), "segments": p.get("n_segments"),
                    "make_source": p.get("make_source")})
    return out


def _shot_out(gi: GameInput, sh: dict) -> dict:
    return {"t_release": round(float(gi.t[sh["i_rel"]]), 2), "t_rim": round(float(gi.t[sh["i_rim"]]), 2),
            "basket": sh["side"], "team": sh.get("team"), "shooter_track": sh.get("shooter"),
            "distance_m": sh.get("dist"), "value_guess": sh.get("value"), "free_throw": bool(sh.get("ft")),
            "made": bool(sh.get("made")), "missed": sh.get("missed"), "confidence": sh["confidence"],
            "basket_event": sh.get("basket_id")}


def segments(gi: GameInput, poss: list[dict], fts: list[dict], brks: list[dict], stoppages: list,
             shots: list[dict], events: list[dict]) -> list[dict]:
    """Game segments for the AI: possessions, free-throw sequences, breaks/stoppages and loose/unclear gaps in
    between, in time order, non-overlapping."""
    items: list[dict] = []
    for b in brks:
        items.append({"kind": b["kind"], "t0": b["t0"], "t1": b["t1"], "team": None})
    for a, b in stoppages:
        items.append({"kind": "stoppage", "t0": a, "t1": b, "team": None})
    for w in fts:
        items.append({"kind": "free_throws", "t0": w.get("t_setup", w["t0"] - 1.0), "t1": w["t1"] + 1.5, "team": w.get("team"),
                      "basket": w["side"]})
    fixed = sorted(items, key=lambda s: s["t0"])
    for k, p in enumerate(poss):
        a, b = p["t0"], p["t1"]
        # a possession ends at the shot it ends with (the ball flight belongs to it)
        sh = [s for s in shots if p["t0"] <= gi.t[s["i_rel"]] <= p["t1"] + 1.0 and not s.get("ft")]
        if sh:
            b = max(b, float(gi.t[sh[-1]["i_rim_end"]]))
        for f in fixed:
            if f["t0"] <= a <= f["t1"]:
                a = f["t1"]
            if f["t0"] <= b <= f["t1"]:
                b = f["t0"]
        if b - a < 0.3:
            continue
        items.append({"kind": "possession", "t0": a, "t1": b, "team": p["team"], "basket": p.get("attacks"),
                      "possession": f"p{k}", "start": p.get("start", "start"), "confidence": p["confidence"],
                      "passes": p["passes"], "fast_break": bool(p.get("fast_break"))})
    items.sort(key=lambda s: (s["t0"], s["t1"]))
    out: list[dict] = []
    cursor = float(gi.t[0])
    for s in items:
        if s["t0"] < cursor - 0.05:
            if s["t1"] <= cursor + 0.3:
                continue
            s = {**s, "t0": cursor}
        if s["t0"] - cursor >= 1.0:
            out.append({"kind": "loose_or_unclear", "t0": cursor, "t1": s["t0"], "team": None})
        out.append(s)
        cursor = s["t1"]
    if gi.t[-1] - cursor >= 1.0:
        out.append({"kind": "loose_or_unclear", "t0": cursor, "t1": float(gi.t[-1]), "team": None})
    for k, s in enumerate(out):
        s["id"] = f"g{k}"
        s["t0"], s["t1"] = round(float(s["t0"]), 2), round(float(s["t1"]), 2)
        s["events"] = [e["id"] for e in events if s["t0"] - 0.05 <= e["t"] < s["t1"] + 0.05]
        s["event_titles"] = [e["title"] for e in events if s["t0"] - 0.05 <= e["t"] < s["t1"] + 0.05]
        s["end"] = _segment_end(s, shots, events, gi)
    return out


def _segment_end(s: dict, shots: list[dict], events: list[dict], gi: GameInput) -> str:
    if s["kind"] != "possession":
        return s["kind"]
    inside = [e for e in events if s["t0"] - 0.05 <= e["t"] <= s["t1"] + 0.6]
    types = [e["type"] for e in inside]
    if any(sh.get("made") and s["t0"] <= gi.t[sh["i_rel"]] <= s["t1"] + 0.6 and not sh.get("ft") for sh in shots):
        return "made basket"
    if "shot" in types:
        return "missed shot" if any(e["type"] == "shot" for e in inside) else "shot"
    if "turnover" in types:
        return "turnover"
    after = [e for e in events if s["t1"] - 0.1 <= e["t"] <= s["t1"] + 2.0 and e["type"] == "steal"]
    if after:
        return "turnover (steal)"
    return "unclear"


# ====================================================================================== titles / merging
def event_title(ev: dict, names: dict[str, str] | None = None) -> str:
    """"<label> – <team name> <suffix>" (the team name is looked up at display time, so renamed teams show)."""
    names = names or {}
    label = ev.get("label") or TITLES.get(str(ev.get("type")), str(ev.get("type") or "event").replace("_", " ").capitalize())
    title = str(label)
    if ev.get("team"):
        title += f" – {names.get(ev['team'], ev['team'])}"
    suffix = ev.get("suffix") if "label" in ev else (f"({ev['basket']})" if ev.get("basket") else "")
    if suffix:
        title += f" {suffix}"
    return title


def merge_events(replay_events: Sequence[dict] | None, gs: dict | None, extra: Sequence[dict] | None = None
                 ) -> list[dict]:
    """Detected baskets (replay_data events) annotated by the game state (free throw / 2 / 3?), plus the game
    state's own events and ``extra`` events (e.g. light-model labels). Each event has a ``title``."""
    ann = (gs or {}).get("basket_annotations") or {}
    out: list[dict] = []
    for e in replay_events or []:
        if not isinstance(e, dict) or e.get("id") is None:
            continue
        a = ann.get(str(e.get("id")))
        if a:
            e = {**e, "title": a.get("title"), "label": a.get("label"), "suffix": a.get("suffix", ""),
                 "points_guess": a.get("points_guess"),
                 "details": {**(e.get("details") or {}), "game_state": a}}
            if a.get("team") in TEAMS and a.get("team") != e.get("team"):
                e["team_detected"] = e.get("team")
                e["team"] = a["team"]  # credited by the attack direction / free-throw shooter
        out.append(e)
    have = {str(e.get("id")) for e in out}
    for e in list((gs or {}).get("events") or []) + list(extra or []):
        if isinstance(e, dict) and e.get("id") is not None and str(e["id"]) not in have:
            out.append(e)
            have.add(str(e["id"]))
    out.sort(key=lambda e: _f(e.get("t")) or 0.0)
    return out


# ====================================================================================== files
def path_for(results_dir: Path) -> Path:
    return Path(results_dir) / FILENAME


def _read(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def save(results_dir: Path, doc: dict) -> Path:
    p = path_for(results_dir)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, p)
    return p


def build_for_results(results_dir: Path, rim_ball: list | None = None, replay: dict | None = None,
                      cfg: GSConfig | None = None, corrections: dict | None = None) -> dict | None:
    """Analyse ``results_dir/replay_data.json`` (+ inplay.json) and write ``game_state.json``. ``rim_ball``
    (from pass 2) is kept in the file so later rebuilds can reuse it. ``corrections``: the corrections ledger
    (default: the project's ledger when env REPLAY_CORRECTIONS=1, corrections.load_if_enabled)."""
    results_dir = Path(results_dir)
    if corrections is None:
        from replay import corrections as _corr
        corrections = _corr.load_if_enabled(results_dir.parent)
    if replay is None:
        replay = _read(results_dir / "replay_data.json")
    if not isinstance(replay, dict) or not replay.get("t"):
        return None
    if rim_ball is None:
        old = _read(path_for(results_dir))
        rim_ball = ((old or {}).get("inputs") or {}).get("rim_ball") if isinstance(old, dict) else None
    inplay = _read(results_dir / "inplay.json")
    inplay = inplay if isinstance(inplay, dict) else None
    sb = __import__("replay.scorebar", fromlist=["x"]).load_result(results_dir / "work")  # broadcast score bar
    use_v6 = possession_v6_enabled()
    doc = analyse(replay, inplay, rim_ball, cfg, scorebar=sb if use_v6 else None, v6=use_v6, corrections=corrections)
    from replay import ball_events  # passes / bad passes / steals / shot details (ball_events.py)
    ball_events.attach(doc, replay, results_dir=results_dir)
    sa = doc.get("sheet_alignment")
    if use_v6 and isinstance(sa, dict) and sa.get("source") != "scorebar":
        # v6 with an attached scoresheet: second pass with the sheet's teams / points as possession boundaries
        ann = doc.get("basket_annotations") or {}
        make_teams = {bid: (a["team"], int(a.get("points_guess") or 2)) for bid, a in ann.items()
                      if a.get("team_source") == "scoresheet" and a.get("team") in TEAMS}
        off_sheet = {bid for bid, a in ann.items() if a.get("kind") == "not_on_sheet"}
        doc = analyse(replay, inplay, rim_ball, cfg, make_teams=make_teams, extra_makes=list(sa.get("missing") or []),
                      off_sheet=off_sheet, v6=True, corrections=corrections)
        ball_events.attach(doc, replay, results_dir=results_dir)
    if sb:
        doc["scorebar"] = {k: sb.get(k) for k in ("summary", "timeline", "sides", "clock")}
    try:
        st = (results_dir / "replay_data.json").stat()
        doc["source"] = {"replay_data_mtime": st.st_mtime, "replay_data_size": st.st_size}
    except OSError:
        doc["source"] = {}
    save(results_dir, doc)
    return doc


def is_current(results_dir: Path, doc: Any) -> bool:
    if not isinstance(doc, dict) or doc.get("format") != FORMAT or doc.get("engine_version") != engine_version():
        return False
    from replay import ball_events
    if not ball_events.is_current(doc):  # ball-event layer missing / older version: rebuild
        return False
    try:
        st = (Path(results_dir) / "replay_data.json").stat()
    except OSError:
        return True
    src = doc.get("source") or {}
    return src.get("replay_data_size") == st.st_size and abs((src.get("replay_data_mtime") or 0) - st.st_mtime) < 1e-3


def load_or_build(results_dir: Path, replay: dict | None = None, max_build_frames: int | None = None
                  ) -> dict | None:
    """The current game state of a Replay's results folder, rebuilt when missing or stale (a few seconds per
    10 min of video). With ``max_build_frames`` a longer Replay is not rebuilt here (GUI thread): the stale
    file (or None) is returned and pass 2 / the AI run rebuild it."""
    doc = _read(path_for(results_dir))
    if is_current(results_dir, doc):
        return doc
    if max_build_frames is not None:
        n = len((replay or {}).get("t") or []) if replay else None
        if n is None or n > max_build_frames:
            return doc if isinstance(doc, dict) and doc.get("format") == FORMAT else None
    try:
        return build_for_results(results_dir, replay=replay)
    except Exception as e:  # never break the caller (GUI timeline, export) on an analysis problem
        print("game state:", type(e).__name__, e)
        return doc if isinstance(doc, dict) else None


def light_events(project_root: Path) -> list[dict]:
    """Events from the last AI run's light model (TypeSafe Jev), e.g. turnovers ("losing the ball")."""
    d = _read(Path(project_root) / "results" / "ai_analysis.json")
    ev = (d or {}).get("light_events") if isinstance(d, dict) else None
    return [e for e in (ev or []) if isinstance(e, dict) and e.get("id") is not None]


def auto_events(project_root: Path, replay: dict | None, max_build_frames: int | None = 6000) -> list[dict]:
    """Everything for the timeline's "Auto detection" track of a Replay folder: detected baskets (annotated),
    game-state events and the light model's events of the last AI run."""
    res = Path(project_root) / "results"
    base = list((replay or {}).get("events") or [])
    gs = load_or_build(res, replay, max_build_frames) if replay else None
    out = merge_events(base, gs, light_events(project_root))
    # very uncertain rule/light-model events stay in game_state.json but would only clutter the timeline
    return [e for e in out if e.get("source") not in (SOURCE, "jev")
            or (_f(e.get("confidence")) or 0.0) >= TIMELINE_MIN_CONF]


def after_pass2(out: Path, ball_rows: Sequence[np.ndarray], rims: dict, t: np.ndarray,
                corrections: dict | None = None) -> dict | None:
    """Pass-2 hook (postprocess): rim evidence from all ball candidates + the game state of the outputs just
    written to ``out``. Never raises (a problem here must not fail the build)."""
    try:
        rim_ball = rim_proximity(ball_rows, rims, t)
        doc = build_for_results(Path(out), rim_ball=rim_ball, corrections=corrections)
        if doc:
            print(f"game state: {len(doc.get('possessions', []))} possessions, {len(doc.get('events', []))} events")
        return doc
    except Exception as e:  # noqa: BLE001
        print("game state failed:", type(e).__name__, e)
        return None


# ====================================================================================== text for people / AIs
SEMANTICS = [
    "The single panning camera shows only part of the court. Players who are not visible are still ON THE "
    "COURT: during live play each team ALWAYS has 5 players on court. Never conclude that a team plays with "
    "fewer players, that a player left, or that a side of the court is empty just because nobody is tracked there.",
    "Substitutions are rare and only happen in dead balls (free throws, time-outs, after fouls/out of bounds).",
    "Free throws are DEAD-BALL situations: the shooter stands alone at the free-throw line (5.8 m from the "
    "baseline), the other players stand along the lane lines and the ball goes from the shooter to the rim. "
    "Moving the ball between referee and shooter is NOT a pass; there is no offence/defence/spacing to judge "
    "during them.",
    "Each team attacks ONE basket per half and the teams switch baskets at half time. Use the attack direction "
    "given below; do not guess it from single frames.",
    "A possession belongs to the team that controls the ball; it ends with a shot (made or missed + rebound), "
    "a turnover (steal, out of bounds, violation, lost ball), a free-throw situation or a break.",
    "A referee holding the ball = dead ball (foul, out of bounds, violation or time-out; an inbound follows).",
    "Track IDs are not identities. On-court slots (A1-A5, B1-B5) carry a player across tracker ID changes "
    "heuristically: good for counting and positions, not for naming individuals.",
    "All events and labels are detected automatically and carry a confidence 0..1; below ~0.5 treat them as "
    "uncertain.",
]


def _clock(t: float) -> str:
    t = max(0.0, float(t))
    return f"{int(t // 60)}:{t % 60:04.1f}"


def direction_lines(doc: dict | None, names: dict[str, str]) -> list[str]:
    out = []
    for d in (doc or {}).get("attack_direction") or []:
        span = f"{d['label']}, video {d['t0']:.0f}-{d['t1']:.0f} s"
        ev = ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in (d.get("evidence") or {}).items() if v)
        if d.get("A"):
            out.append(f"- {span}: {names.get('A', 'A')} (A) attacks basket {d['A']}, {names.get('B', 'B')} (B) "
                       f"attacks basket {d['B']} - confidence {d['confidence']:.2f}" + (f" (evidence: {ev})" if ev else ""))
        else:
            out.append(f"- {span}: attack direction UNKNOWN (no free throws, baskets or clear half-court play)")
    return out or ["- unknown (no game state)"]


def event_lines(events: Sequence[dict], names: dict[str, str], limit: int = 400,
                t_range: tuple[float, float] | None = None) -> list[str]:
    out = []
    for e in events:
        t = _f(e.get("t"))
        if t is None or (t_range and not (t_range[0] - 1e-6 <= t <= t_range[1] + 1e-6)):
            continue
        title = e.get("title") or event_title(e, names)
        if e.get("label"):
            title = event_title(e, names)
        dur = _f(e.get("duration")) or 0.0
        src = e.get("source") or "auto"
        conf = _f(e.get("confidence"))
        out.append(f"- t={t:.1f}" + (f"-{t + dur:.1f}" if dur > 0.5 else "") + f" ({_clock(t)}) {title}"
                   + (f" [conf {conf:.2f}" if conf is not None else " [") + (f", {src}]" if src != "auto" else "]"))
        if len(out) >= limit:
            out.append(f"- ... {len(events) - limit} more")
            break
    return out


SEGMENT_COLUMNS = ["id", "t0", "t1", "kind", "team", "attacks", "start", "end", "passes", "conf", "events"]


def segment_table(segs: Sequence[dict], names: dict[str, str] | None = None) -> str:
    """Pipe table of the game segments (possessions, free throws, breaks, loose/unclear spells)."""
    lines = ["|".join(SEGMENT_COLUMNS)]
    for s in segs:
        lines.append("|".join("" if v is None else str(v) for v in [
            s["id"], f"{s['t0']:.1f}", f"{s['t1']:.1f}", s["kind"], s.get("team") or "", s.get("basket") or "",
            s.get("start") or "", s.get("end") or "", s.get("passes"), s.get("confidence"),
            "; ".join(s.get("event_titles") or [])]))
    return "\n".join(lines)
