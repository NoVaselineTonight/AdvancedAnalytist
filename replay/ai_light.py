"""Light tier of the AI analysis: data cleaning, game segmentation and per-segment facts, plus optional labelling
with TypeSafe AI's Jev model.

Division of work (see ``replay/ai_analysis.py`` for the Gemini tier):

* **Python (deterministic, free):** impossible jumps, duplicate/ghost tracks, bench/sideline people, calibration
  gaps, segmentation into possessions, per-segment statistics (duration, ball path, passes, spacing, players
  behind the ball, ...) and rule-based phase labels.
* **Jev (cheap typed decisions, optional):** for every segment, which phase it is from the coached team's point of
  view, the defensive scheme, whether a coach would want to review it and whether its data is reliable; and the
  role of tracks the rules could not decide (player / bench person / duplicate / referee). Jev is a "System One"
  decision model: it answers typed questions (noul = yes/no probability, choice, score) and never writes text, so
  it cannot write summaries - the factual summaries are computed in Python.
* **Gemini (expensive, complex):** receives the cleaned data, the segment table with the labels and the raw
  tracking only where it is needed (segments Jev marks as worth reviewing) and writes the tactical analysis.

Jev API (official docs https://docs.typesafe.ai/api, https://docs.typesafe.ai/models, checked 2026-10):
``POST {base}/v1/systemone`` with ``Authorization: Bearer <key>``, body ``{"state", "model", "questions": {id:
{"type": "noul"|"choice"|"score", "instructions", "criteria"}}}``, response ``{"model", "answers": {id: {...}},
"usage": {"input_tokens", "output_tokens"}}``. Models ``jev-latest`` (alias of ``jev-1.13.0``), ``jev-preview``.
64k-token context per request; errors 401 (key), 422 (validation), 429 (rate limit), 529 (overloaded).

No Qt here.
"""

from __future__ import annotations

import json
import math
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from replay import data_export as dx

PROVIDER_NAME = "TypeSafe AI"
DEFAULT_BASE_URL = "https://api.typesafe.ai"
DEFAULT_MODEL = "jev-latest"
MODELS = ["jev-latest", "jev-1.13.0", "jev-preview"]
REQUEST_TIMEOUT_S = 60.0
QUESTIONS_PER_REQUEST = 120  # well inside the 64k-token budget (each question carries ~100-200 tokens of facts)

MAX_SPEED = 11.0  # m/s, faster steps are tracking glitches
GHOST_DIST = 0.5  # m, two same-team tracks closer than this most of their common time = one person
GHOST_SHARE = 0.7
GHOST_AMBIGUOUS = 0.45
OUTSIDE_MARGIN = 0.3  # m outside the court lines
BENCH_SHARE = 0.8
BENCH_AMBIGUOUS = 0.45
CALIB_GAP_S = 1.0
SEG_MAX_S = 30.0  # long possessions are split so a timestamp points to a reviewable window
SEG_MIN_S = 1.0  # shorter holder runs (deflections, mis-assigned holders) are absorbed
DEAD_BALL_S = 4.0  # nobody holding the ball this long ends a possession
TRANSITION_S = 8.0  # ball crossing half court within this time of the start of a possession = transition

PHASES = {
    "offense_half_court": "The coached team has the ball and attacks a set defence in its front court.",
    "defense_half_court": "The opponent has the ball in its front court against the coached team's set defence.",
    "transition_offense": "The coached team pushes the ball up the court after gaining possession (fast break "
                          "or early offence before the defence is set).",
    "transition_defense": "The opponent pushes the ball up the court after gaining possession and the coached "
                          "team is getting back.",
    "dead_ball": "Stoppage, free throws, inbound set-up, time-out or nobody controls the ball.",
    "unclear": "The data is too incomplete to tell.",
}
SCHEMES = {"man_to_man": "Defenders follow individual opponents.",
           "zone": "Defenders guard areas and shift with the ball.",
           "press": "Defenders pressure the ball in the back court / full court.",
           "unclear": "Cannot be told from the data."}
TRACK_ROLES = {"player_on_court": "A player taking part in the game on the court.",
               "bench_or_sideline": "A substitute, coach, staff member or spectator next to the court.",
               "duplicate": "A second track of a person who already has another track at the same place.",
               "referee": "A referee."}


class LightError(Exception):
    """Readable error from the light-work model (never contains the key)."""


# ====================================================================== data
@dataclass
class GameData:
    d: Any  # data_export._Data
    players: list[dict]
    ball: list[dict]
    team_of_track: dict[str, str]
    length: float
    width: float
    baskets: list[list[float]]
    gs: dict | None = None  # basketball game state (replay/game_state.py): possessions, direction, events, slots


def load_game(project_root: Path) -> GameData:
    from replay.ai_analysis import _court_size
    d = dx._Data(Path(project_root))
    players, ball = dx._players_and_ball(d)
    team_of_track = {tid: str(v.get("team")) for tid, v in d.track_info.items()}
    for r in players:
        team_of_track.setdefault(r["track"], r["team"])
    length, width, baskets = _court_size(d)
    return GameData(d, players, ball, team_of_track, length, width, baskets, getattr(d, "game_state", None))


# ====================================================================== cleaning (Python)
@dataclass
class CleanReport:
    glitch_rows: set = field(default_factory=set)  # {(track, t)}
    ghost_tracks: dict = field(default_factory=dict)  # track -> track it duplicates
    bench_tracks: set = field(default_factory=set)
    ambiguous_tracks: dict = field(default_factory=dict)  # track -> facts (for Jev)
    calib_gaps: list = field(default_factory=list)  # [(t0, t1)]
    tracking_gaps: list = field(default_factory=list)  # calibrated but nobody tracked
    decided_by: dict = field(default_factory=dict)  # track -> "python" | "jev"

    def canonical(self, tid: str | None) -> str | None:
        """The kept track for a duplicate track id (follows chains), else ``tid``."""
        seen = set()
        while tid in self.ghost_tracks and tid not in seen:
            seen.add(tid)
            tid = self.ghost_tracks[tid]
        return tid

    @property
    def excluded_tracks(self) -> set:
        return set(self.ghost_tracks) | self.bench_tracks

    def to_dict(self) -> dict:
        return {"glitch_rows_removed": len(self.glitch_rows),
                "duplicate_tracks": {k: v for k, v in sorted(self.ghost_tracks.items())},
                "bench_or_sideline_tracks": sorted(self.bench_tracks, key=dx._id_key),
                "undecided_tracks_kept": sorted(self.ambiguous_tracks, key=dx._id_key),
                "calibration_gaps_s": [[round(a, 1), round(b, 1)] for a, b in self.calib_gaps],
                "tracking_gaps_s": [[round(a, 1), round(b, 1)] for a, b in self.tracking_gaps],
                "decided_by": dict(sorted(self.decided_by.items()))}


def _ranges(ts: list[float], flags: list[bool], min_len: float) -> list[tuple[float, float]]:
    out, start = [], None
    for t, bad in zip(ts, flags):
        if bad and start is None:
            start = t
        elif not bad and start is not None:
            if t - start >= min_len:
                out.append((start, t))
            start = None
    if start is not None and ts and ts[-1] - start >= min_len:
        out.append((start, ts[-1]))
    return out


def clean_data(g: GameData) -> CleanReport:
    rep = CleanReport()
    L, W = g.length, g.width
    by_track: dict[str, list[dict]] = defaultdict(list)
    for r in g.players:
        by_track[r["track"]].append(r)

    # --- impossible jumps
    for tid, rows in by_track.items():
        last = None
        for r in rows:
            if r["interp"]:
                continue
            if last is not None:
                dt = r["t"] - last["t"]
                if 0 < dt <= 1.0 and math.hypot(r["x"] - last["x"], r["y"] - last["y"]) / dt > MAX_SPEED:
                    rep.glitch_rows.add((tid, r["t"]))
                    continue  # compare the next row with the last good one
            last = r

    # --- bench / sideline people (mostly outside the court lines)
    for tid, rows in by_track.items():
        if not rows or rows[0]["role"] == "referee":
            continue
        out = sum(1 for r in rows if r["x"] < -OUTSIDE_MARGIN or r["x"] > L + OUTSIDE_MARGIN
                  or r["y"] < -OUTSIDE_MARGIN or r["y"] > W + OUTSIDE_MARGIN)
        share = out / len(rows)
        if share >= BENCH_SHARE:
            rep.bench_tracks.add(tid)
            rep.decided_by[tid] = "python"
        elif share >= BENCH_AMBIGUOUS and rows[0]["team"] in ("A", "B"):
            rep.ambiguous_tracks[tid] = _track_facts(g, tid, rows, outside_share=share)

    # --- duplicate / ghost tracks (same team, same place, same time); checked at <= 5 Hz
    step = max(1, int(round(g.d.fps / 5)))
    frames: dict[float, list[dict]] = defaultdict(list)
    for r in g.players:
        if r["team"] in ("A", "B") and not r["interp"]:
            frames[r["t"]].append(r)
    times = sorted(frames)[::step]
    together: dict[tuple[str, str], int] = defaultdict(int)
    common: dict[tuple[str, str], int] = defaultdict(int)
    for t in times:
        rows = frames[t]
        for i, a in enumerate(rows):
            for b in rows[i + 1:]:
                if a["team"] != b["team"] or a["track"] == b["track"]:
                    continue
                key = tuple(sorted((a["track"], b["track"]), key=dx._id_key))
                common[key] += 1
                if math.hypot(a["x"] - b["x"], a["y"] - b["y"]) < GHOST_DIST:
                    together[key] += 1
    min_common = 5  # >= ~1 s of common time at 5 Hz
    owned: dict[str, int] = defaultdict(int)
    for b in g.ball:
        if b["owner"]:
            owned[b["owner"]] += 1
    # the duplicate is the track with fewer rows, then fewer ball possessions, then the newer (higher) ID
    rank = lambda tid: (len(by_track[tid]), owned[tid], -dx._id_key(tid)[0])  # noqa: E731
    for key, n in common.items():
        if n < min_common:
            continue
        share = together[key] / n
        a, b = key
        shorter, longer = (a, b) if rank(a) < rank(b) else (b, a)
        if shorter in rep.bench_tracks or shorter in rep.ghost_tracks:
            continue
        if share >= GHOST_SHARE:
            rep.ghost_tracks[shorter] = longer
            rep.decided_by[shorter] = "python"
            rep.ambiguous_tracks.pop(shorter, None)
        elif share >= GHOST_AMBIGUOUS and shorter not in rep.ambiguous_tracks:
            rep.ambiguous_tracks[shorter] = _track_facts(g, shorter, by_track[shorter], near_track=longer,
                                                         near_share=share)

    # --- calibration and tracking gaps
    rt = [float(t) for t in (g.d.replay.get("t") or [])]
    rc = g.d.replay.get("calib") or []
    if rt and len(rc) == len(rt):
        rep.calib_gaps = _ranges(rt, [not c for c in rc], CALIB_GAP_S)
    else:
        rep.calib_gaps = [tuple(r) for r in ((g.d.report.get("failed_ranges") or {}).get("no_calibration") or [])]
    seen = {r["t"] for r in g.players}
    if rt:
        calibrated = [bool(rc[i]) if i < len(rc) else True for i in range(len(rt))]
        rep.tracking_gaps = _ranges(rt, [c and round(t, 3) not in seen and t not in seen
                                         for t, c in zip(rt, calibrated)], 2.0)
    return rep


def _track_facts(g: GameData, tid: str, rows: list[dict], **extra: Any) -> dict:
    xs, ys = [r["x"] for r in rows], [r["y"] for r in rows]
    f = {"track_id": tid, "team": g.d.team_name(rows[0]["team"]), "seconds_tracked": round(len(rows) / g.d.fps, 1),
         "first_t": round(rows[0]["t"], 1), "last_t": round(rows[-1]["t"], 1),
         "mean_x": round(sum(xs) / len(xs), 1), "mean_y": round(sum(ys) / len(ys), 1),
         "x_range": [round(min(xs), 1), round(max(xs), 1)], "y_range": [round(min(ys), 1), round(max(ys), 1)],
         "distance_moved_m": round(sum(math.hypot(b["x"] - a["x"], b["y"] - a["y"])
                                       for a, b in zip(rows, rows[1:])), 1)}
    for k, v in extra.items():
        f[k] = round(v, 2) if isinstance(v, float) else v
    return f


# ====================================================================== segmentation (Python)
def _basket_side(g: GameData, team: str, t: float) -> tuple[str | None, str]:
    """Which basket ('L'/'R') ``team`` attacks around time ``t``: nearest detected basket of either team (teams
    switch at half time, so the nearest one in time is used), else None."""
    best = None
    for e in g.d.events:
        if e.get("type") != "basket":
            continue
        et, side, eteam = dx._f(e.get("t")), e.get("basket"), e.get("team")
        if et is None or side not in ("L", "R") or eteam not in ("A", "B"):
            continue
        s = side if eteam == team else ("R" if side == "L" else "L")
        if best is None or abs(et - t) < best[0]:
            best = (abs(et - t), s)
    return (best[1], "detected baskets") if best else (None, "")


DEAD_KINDS = ("free_throws", "timeout", "quarter_break", "half_time", "stoppage")


def segment_game(g: GameData, clean: CleanReport, coached: str) -> list[dict]:
    """Split the game into possessions / dead-ball spells and compute the facts of each segment. Uses the
    basketball game state (possessions with hysteresis, free throws, breaks, attack direction) when available,
    else a simple split by the ball holder's team."""
    gsd = g.gs if isinstance(g.gs, dict) else None
    if gsd and gsd.get("segments") and (gsd.get("possessions") or gsd.get("free_throw_windows")):
        return _segments_from_game_state(g, clean, coached, gsd)
    return _segment_game_raw(g, clean, coached)


def _segments_from_game_state(g: GameData, clean: CleanReport, coached: str, gsd: dict) -> list[dict]:
    excluded = clean.excluded_tracks
    ball_t = {r["t"]: r for r in g.ball}
    rt = [float(t) for t in (g.d.replay.get("t") or [])] or sorted({r["t"] for r in g.players} | set(ball_t))
    by_t: dict[float, list[dict]] = defaultdict(list)
    for r in g.players:
        if r["track"] not in excluded and (r["track"], r["t"]) not in clean.glitch_rows and r["team"] in ("A", "B"):
            by_t[r["t"]].append(r)
    out = []
    prev_team = None
    for n, gs_seg in enumerate(gsd.get("segments") or []):
        t0, t1 = float(gs_seg["t0"]), float(gs_seg["t1"])
        ts = [t for t in rt if t0 - 1e-6 <= t <= t1 + 1e-6] or [t0, t1]
        kind = gs_seg.get("kind")
        team = gs_seg.get("team") if kind == "possession" else None
        seg = _segment_facts(g, n, team, prev_team, ts, [], by_t, ball_t, coached, clean,
                             side_given=(gs_seg.get("basket"), "game state") if team else (None, ""))
        seg["kind"] = kind
        seg["start"] = gs_seg.get("start")
        seg["end"] = gs_seg.get("end")
        seg["events"] = list(gs_seg.get("event_titles") or [])
        if gs_seg.get("passes") is not None:
            seg["passes"] = gs_seg["passes"]
        if gs_seg.get("confidence") is not None:
            seg["possession_confidence"] = gs_seg["confidence"]
        seg["possession_change"] = bool(team and prev_team and team != prev_team)
        if kind in DEAD_KINDS:
            seg["phase"] = "dead_ball"
            seg["dead_ball_kind"] = kind
            if kind == "free_throws":
                ft_team = gs_seg.get("team")
                seg["free_throws_by"] = "coached" if ft_team == coached else "opponent" if ft_team else "unknown"
                seg["attacks"] = gs_seg.get("basket")
        elif kind == "loose_or_unclear" or (kind == "possession" and team is None):
            # possession model v6: a live-play possession whose team the evidence vote could not decide is
            # "unclear", never a dead ball
            seg["phase"] = "unclear"
        elif kind == "possession" and gs_seg.get("fast_break"):
            seg["phase"] = "transition_offense" if team == coached else "transition_defense"
        out.append(seg)
        if team is not None:
            prev_team = team
    return out


def _segment_game_raw(g: GameData, clean: CleanReport, coached: str) -> list[dict]:
    """Fallback segmentation (no game state): runs of the ball holder's team."""
    excluded = clean.excluded_tracks
    ball_t = {r["t"]: r for r in g.ball}
    rt = [float(t) for t in (g.d.replay.get("t") or [])] or sorted({r["t"] for r in g.players} | set(ball_t))
    if not rt:
        return []
    # holder team per frame
    holder: list[str | None] = []
    for t in rt:
        b = ball_t.get(t) or ball_t.get(round(t, 3))
        team = None
        owner = clean.canonical(b["owner"]) if b is not None else None
        if b is not None and b["state"] == "possessed" and owner and owner not in excluded:
            tm_ = g.team_of_track.get(owner)
            team = tm_ if tm_ in ("A", "B") else None
        holder.append(team)
    # raw runs: (team or None, i0, i1)
    runs: list[list] = []
    cur, i0, last_held = None, 0, 0
    for i, (t, h) in enumerate(zip(rt, holder)):
        if h is not None:
            if h != cur:
                if i > i0:
                    runs.append([cur, i0, i - 1])
                cur, i0 = h, i
            last_held = i
        elif cur is not None and t - rt[last_held] > DEAD_BALL_S:
            runs.append([cur, i0, last_held])
            cur, i0 = None, last_held + 1
    runs.append([cur, i0, len(rt) - 1])
    runs = [r for r in runs if r[2] >= r[1]]
    # absorb very short holder runs into the previous segment
    merged: list[list] = []
    for r in runs:
        dur = rt[r[2]] - rt[r[1]]
        if merged and (dur < SEG_MIN_S or r[0] == merged[-1][0]):
            merged[-1][2] = r[2]
        else:
            merged.append(r)
    # split long ones
    segs: list[tuple[str | None, int, int]] = []
    for team, a, b in merged:
        start = a
        for i in range(a, b + 1):
            if rt[i] - rt[start] >= SEG_MAX_S:
                segs.append((team, start, i - 1))
                start = i
        segs.append((team, start, b))

    by_t: dict[float, list[dict]] = defaultdict(list)
    for r in g.players:
        if r["track"] not in excluded and (r["track"], r["t"]) not in clean.glitch_rows and r["team"] in ("A", "B"):
            by_t[r["t"]].append(r)
    out = []
    prev_team = None
    for n, (team, a, b) in enumerate(segs):
        out.append(_segment_facts(g, n, team, prev_team, rt[a:b + 1], holder[a:b + 1], by_t, ball_t, coached,
                                  clean))
        if team is not None:
            prev_team = team
    return out


def _segment_facts(g: GameData, n: int, team: str | None, prev_team: str | None, ts: list[float],
                   holders: list, by_t: dict, ball_t: dict, coached: str, clean: CleanReport,
                   side_given: tuple[str | None, str] | None = None) -> dict:
    L = g.length
    other = "B" if coached == "A" else "A"
    t0, t1 = ts[0], ts[-1]
    seg: dict[str, Any] = {"id": f"s{n}", "t0": round(t0, 1), "t1": round(t1, 1), "duration_s": round(t1 - t0, 1)}
    seg["ball_team"] = "coached" if team == coached else "opponent" if team == other else "none"
    balls = [ball_t.get(t) for t in ts if ball_t.get(t) is not None]
    seg["ball_seen_pct"] = round(100 * len(balls) / max(1, len(ts)))
    cal_gap = sum(max(0.0, min(t1, b) - max(t0, a)) for a, b in clean.calib_gaps)
    seg["no_calibration_s"] = round(cal_gap, 1)
    side, src = side_given if side_given is not None else (_basket_side(g, team, t0) if team else (None, ""))
    if team and side is None and balls:  # fall back: where the holding team takes the ball
        late = balls[len(balls) // 2:]
        side, src = ("R" if sum(b["x"] for b in late) / len(late) > L / 2 else "L"), "ball position"
    seg["attacks"] = side
    seg["attack_basket_source"] = src
    if balls:
        seg["ball_start"] = [round(balls[0]["x"], 1), round(balls[0]["y"], 1)]
        seg["ball_end"] = [round(balls[-1]["x"], 1), round(balls[-1]["y"], 1)]
    # crossing half court towards the attacked basket
    crossed = None
    if side and balls:
        own_half = (lambda x: x > L / 2) if side == "L" else (lambda x: x < L / 2)
        if own_half(balls[0]["x"]):
            for bb in balls:
                if not own_half(bb["x"]):
                    crossed = round(bb["t"] - t0, 1)
                    break
    seg["started_in_back_court"] = bool(side and balls and ((balls[0]["x"] > L / 2) if side == "L"
                                                            else (balls[0]["x"] < L / 2)))
    seg["crossed_half_after_s"] = crossed
    # passes: holder changes within the team
    owners = [clean.canonical(b["owner"]) for b in balls if b["state"] == "possessed" and b["owner"]]
    seg["passes"] = sum(1 for a, b in zip(owners, owners[1:]) if a != b and g.team_of_track.get(a) == team
                        and g.team_of_track.get(b) == team)
    seg["possession_change"] = bool(team and prev_team and team != prev_team)
    # team shape
    stats = {coached: defaultdict(list), other: defaultdict(list)}
    for t in ts[::max(1, int(round(g.d.fps / 2)))] or ts[:1]:
        rows = by_t.get(t, [])
        bb = ball_t.get(t)
        for k in (coached, other):
            pts = [(r["x"], r["y"]) for r in rows if r["team"] == k]
            stats[k]["n"].append(len(pts))
            if len(pts) >= 3:
                cx, cy = sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)
                stats[k]["spread"].append(sum(math.hypot(p[0] - cx, p[1] - cy) for p in pts) / len(pts))
                stats[k]["cx"].append(cx)
            if bb is not None and team and k != team and side and pts:
                # defenders between the ball and the basket they protect (= the attacked basket)
                bx = bb["x"]
                stats[k]["behind"].append(sum(1 for p in pts if (p[0] < bx) == (side == "L")))
    avg = lambda v: round(sum(v) / len(v), 1) if v else None  # noqa: E731
    for k, label in ((coached, "coached"), (other, "opponent")):
        s = stats[k]
        seg[f"{label}_visible"] = avg(s["n"])
        seg[f"{label}_spread_m"] = avg(s["spread"])
        seg[f"{label}_centroid_x"] = avg(s["cx"])
        if s["behind"]:
            seg[f"{label}_defenders_behind_ball"] = avg(s["behind"])
    ev = [e for e in g.d.events if e.get("type") == "basket" and t0 - 0.5 <= (dx._f(e.get("t")) or -1) <= t1 + 0.5]
    if ev:
        seg["baskets"] = [{"t": round(dx._f(e.get("t")) or 0, 1),
                           "team": "coached" if e.get("team") == coached else "opponent" if e.get("team") == other
                           else "unknown"} for e in ev]
    # rule-based labels
    transition = crossed is not None and crossed <= TRANSITION_S and seg["started_in_back_court"]
    if team == coached:
        phase = "transition_offense" if transition else "offense_half_court"
    elif team == other:
        phase = "transition_defense" if transition else "defense_half_court"
    else:
        phase = "dead_ball"
    if seg["ball_seen_pct"] < 20 and team is None or cal_gap > 0.6 * max(0.1, t1 - t0):
        phase = "unclear"
    seg["phase"] = phase
    seg["phase_source"] = "rules"
    vis = (seg["coached_visible"] or 0) + (seg["opponent_visible"] or 0)
    seg["reliable"] = round(min(1.0, vis / 10) * (0.5 + 0.5 * seg["ball_seen_pct"] / 100)
                            * (1 - min(1.0, cal_gap / max(0.1, t1 - t0))), 2)
    seg["review"] = None  # filled by Jev; None = not decided (send everything)
    return seg


# ====================================================================== Jev client
class JevClient:
    """TypeSafe AI ``/v1/systemone``. The key lives only in this object."""

    def __init__(self, api_key: str, base_url: str = DEFAULT_BASE_URL, model: str = DEFAULT_MODEL,
                 timeout_s: float = REQUEST_TIMEOUT_S):
        key = (api_key or "").strip()
        if not key:
            raise LightError("No TypeSafe AI API key.")
        self._key = key
        self.base_url = (base_url or DEFAULT_BASE_URL).strip().rstrip("/")
        if not self.base_url.startswith(("https://", "http://localhost", "http://127.0.0.1")):
            raise LightError("The TypeSafe AI base URL must start with https://")
        self.model = (model or DEFAULT_MODEL).strip()
        self.timeout_s = timeout_s
        self.usage = {"input_tokens": 0, "output_tokens": 0, "requests": 0}
        self.model_used = ""

    def __repr__(self) -> str:
        return f"JevClient({self.base_url}, {self.model})"

    def evaluate(self, state: Any, questions: dict) -> dict:
        import urllib.error
        import urllib.request
        body = json.dumps({"state": state, "model": self.model, "questions": questions}).encode("utf-8")
        req = urllib.request.Request(f"{self.base_url}/v1/systemone", data=body, method="POST",
                                     headers={"Content-Type": "application/json",
                                              "Authorization": f"Bearer {self._key}"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
                data = json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise LightError(_jev_http_error(e, self._key)) from None
        except urllib.error.URLError as e:
            raise LightError(f"Network error: could not reach TypeSafe AI ({e.reason}).".replace(self._key, "***")) \
                from None
        except (TimeoutError, OSError) as e:
            raise LightError(f"TypeSafe AI did not answer ({type(e).__name__}).") from None
        except ValueError:
            raise LightError("TypeSafe AI returned an unreadable answer.") from None
        if not isinstance(data, dict) or not isinstance(data.get("answers"), dict):
            raise LightError("TypeSafe AI returned an answer without 'answers'.")
        u = data.get("usage") or {}
        self.usage["input_tokens"] += int(u.get("input_tokens") or 0)
        self.usage["output_tokens"] += int(u.get("output_tokens") or 0)
        self.usage["requests"] += 1
        self.model_used = str(data.get("model") or self.model)
        return data["answers"]


def _jev_http_error(e: Any, key: str) -> str:
    try:
        detail = e.read().decode("utf-8", "replace")[:300]
    except Exception:
        detail = ""
    detail = detail.replace(key, "***")
    code = e.code
    if code == 401:
        return "TypeSafe AI rejected the API key (401)."
    if code == 422:
        return f"TypeSafe AI rejected the request (422 validation): {detail}"
    if code == 429:
        return "TypeSafe AI rate limit reached (429). Try again in a moment."
    if code == 529:
        return "TypeSafe AI is overloaded (529). Try again later."
    return f"TypeSafe AI error {code}: {detail}"


# ====================================================================== Jev labelling
def _jersey(g: GameData, k: str) -> str:
    """Jersey colour word of a team (from the colour binding), '' when unknown."""
    return str(getattr(g.d, "team_jersey", {}).get(k) or "")


def _state(g: GameData, coached: str) -> dict:
    other = "B" if coached == "A" else "A"
    return {"context": "Basketball game measured by automatic 2D player tracking from one panning video camera. "
                       "Facts below were computed from that tracking and can be incomplete.",
            "court_m": {"length": g.length, "width": g.width, "half_court_x": g.length / 2,
                        "baskets": {"L": g.baskets[0], "R": g.baskets[1]}},
            "coached_team": g.d.team_names[coached], "opponent": g.d.team_names[other],
            "coached_team_jersey": _jersey(g, coached), "opponent_jersey": _jersey(g, other),
            "fields": {"ball_team": "who holds the ball (coached / opponent / none)",
                       "attacks": "basket the ball-holding team attacks",
                       "crossed_half_after_s": "seconds from the start until the ball crossed half court",
                       "*_spread_m": "mean distance of a team's visible players to their centroid",
                       "*_defenders_behind_ball": "defenders between the ball and the basket they protect",
                       "*_visible": "players visible on average (10 are on court; the camera shows part of it)"}}


LOST_BALL_Q = ("Did the team with the ball LOSE it during this possession without a shot attempt - a "
               "turnover (steal, bad pass, stepping out of bounds, travelling or another violation, offensive foul)? "
               "A missed or made shot followed by the other team getting the ball is NOT a turnover; free throws "
               "and breaks are not possessions.")


def segment_questions(seg: dict) -> dict:
    facts = {k: v for k, v in seg.items() if k not in ("phase", "phase_source", "review", "reliable", "id",
                                                         "lost_ball")}
    sid = seg["id"]
    if seg.get("kind") in DEAD_KINDS:
        return {}  # dead balls (free throws, breaks) are labelled by the rules; nothing to ask
    extra = {}
    if seg.get("kind") == "possession":
        extra[f"{sid}__lost_ball"] = {"type": "noul", "instructions": {"question": LOST_BALL_Q, "segment": facts},
                                      "criteria": {"true": "The team lost the ball (turnover)",
                                                   "false": "Ended with a shot, foul, free throws, break or still "
                                                            "in possession"}}
    return {**extra, **{
        f"{sid}__phase": {"type": "choice", "instructions": {
            "question": "Which game phase is this segment, from the coached team's point of view?", "segment": facts},
            "criteria": PHASES},
        f"{sid}__scheme": {"type": "choice", "instructions": {
            "question": "Which defensive scheme does the defending team most likely use in this segment?",
            "segment": facts}, "criteria": SCHEMES},
        f"{sid}__review": {"type": "noul", "instructions": {
            "question": "Would the coached team's coach want to review this segment on video for TEAM tactics "
                        "(a breakdown, poor spacing, slow transition defence, a good set, a basket, a turnover)?",
            "segment": facts},
            "criteria": {"true": "Tactically notable, worth showing", "false": "Routine or no usable information"}},
        f"{sid}__reliable": {"type": "noul", "instructions": {
            "question": "Is the tracking data of this segment complete enough to judge team tactics?",
            "segment": facts},
            "criteria": {"true": "Most players and the ball are seen", "false": "Too few players, no ball or "
                                                                                "no calibration"}},
    }}


def track_question(facts: dict) -> dict:
    return {f"trk_{facts['track_id']}": {"type": "choice", "instructions": {
        "question": "What is this tracked person? Court is 0..length x 0..width metres; positions outside it are "
                    "next to the court.", "track": facts}, "criteria": TRACK_ROLES}}


def label_with_jev(client: JevClient, g: GameData, coached: str, segments: list[dict], clean: CleanReport,
                   progress: Callable[[float, str], None] | None = None,
                   cancel: threading.Event | None = None) -> dict:
    """Ask Jev the typed questions; update ``segments`` and ``clean`` in place. Returns counts."""
    state = _state(g, coached)
    questions: dict[str, dict] = {}
    for seg in segments:
        questions.update(segment_questions(seg))
    for facts in clean.ambiguous_tracks.values():
        questions.update(track_question(facts))
    ids = list(questions)
    answers: dict[str, dict] = {}
    batches = [ids[i:i + QUESTIONS_PER_REQUEST] for i in range(0, len(ids), QUESTIONS_PER_REQUEST)]
    for k, batch in enumerate(batches, 1):
        if cancel is not None and cancel.is_set():
            from replay.ai_analysis import Cancelled
            raise Cancelled()
        if progress:
            progress((k - 1) / max(1, len(batches)), f"TypeSafe Jev: labelling segments ({k}/{len(batches)})")
        for attempt in range(3):
            try:
                answers.update(client.evaluate(state, {q: questions[q] for q in batch}))
                break
            except LightError as e:
                if attempt == 2 or not any(s in str(e) for s in ("(429)", "(529)")):
                    raise
                time.sleep(2.0 * (attempt + 1))
    n_seg = n_trk = 0
    for seg in segments:
        sid = seg["id"]
        ph = answers.get(f"{sid}__phase") or {}
        if ph.get("choice") in PHASES:
            seg["phase"], seg["phase_source"] = ph["choice"], "jev"
            seg["phase_confidence"] = _num(ph.get("confidence"))
            n_seg += 1
        sc = answers.get(f"{sid}__scheme") or {}
        if sc.get("choice") in SCHEMES:
            seg["defense_scheme"] = sc["choice"]
            seg["scheme_confidence"] = _num(sc.get("confidence"))
        for q in ("review", "reliable", "lost_ball"):
            v = (answers.get(f"{sid}__{q}") or {}).get("noul")
            if isinstance(v, (int, float)):
                seg[q] = round(float(v), 2)
    for tid, facts in list(clean.ambiguous_tracks.items()):
        a = answers.get(f"trk_{tid}") or {}
        role = a.get("choice")
        if role not in TRACK_ROLES or _num(a.get("confidence")) < 0.5:
            continue
        if role == "bench_or_sideline":
            clean.bench_tracks.add(tid)
        elif role == "duplicate" and facts.get("near_track"):
            clean.ghost_tracks[tid] = facts["near_track"]
        clean.decided_by[tid] = "jev"
        del clean.ambiguous_tracks[tid]
        n_trk += 1
    return {"segments_labelled": n_seg, "tracks_decided": n_trk, "questions": len(ids), "requests": len(batches)}


def _num(v: Any) -> float:
    try:
        return round(float(v), 2)
    except (TypeError, ValueError):
        return 0.0


JEV_TURNOVER_P = 0.6


def light_events(segments: list[dict], coached: str, gs_events: list[dict] | None = None) -> list[dict]:
    """Turnovers ("lost the ball") from the Jev labels of the possessions, for the timeline's auto track. A
    possession whose loss the rules already detected (turnover/steal event near its end) is not repeated."""
    other = "B" if coached == "A" else "A"
    out = []
    for s in segments:
        p = s.get("lost_ball")
        if s.get("kind") != "possession" or not isinstance(p, (int, float)) or p < JEV_TURNOVER_P:
            continue
        team = coached if s.get("ball_team") == "coached" else other if s.get("ball_team") == "opponent" else None
        t1 = float(s["t1"])
        known = any(e.get("type") in ("turnover", "steal") and t1 - 3.0 <= (dx._f(e.get("t")) or -99) <= t1 + 2.5
                    for e in gs_events or [])
        if known:
            s["lost_ball_confirms_rules"] = True
            continue
        out.append({"id": f"jev-turnover-{s['id']}-{int(round(t1 * 10))}", "type": "turnover", "t": round(t1, 2),
                    "duration": 0.0, "team": team, "basket": None, "confidence": round(float(p), 2),
                    "source": "jev", "label": "Turnover? (Jev)", "suffix": "(lost the ball)",
                    "title": "Turnover? (Jev)", "details": {"segment": s["id"], "segment_t0": s["t0"]}})
    return out


# ====================================================================== whole light tier
@dataclass
class LightResult:
    game: GameData
    clean: CleanReport
    segments: list[dict]
    used_jev: bool
    log: list[dict]  # [{"tier", "task", "detail"}]
    notes: list[str]
    jev: dict  # model, usage
    events: list = field(default_factory=list)  # light-model events for the timeline (Jev turnovers)


def run_light(project_root: Path, coached: str, jev: JevClient | None = None,
              progress: Callable[[float, str], None] | None = None,
              cancel: threading.Event | None = None) -> LightResult:
    prog = progress or (lambda f, m: None)
    prog(0.0, "Cleaning the tracking data")
    g = load_game(project_root)
    clean = clean_data(g)
    log = [{"tier": "python", "task": "data cleaning",
            "detail": f"{len(clean.glitch_rows)} impossible jumps removed, {len(clean.ghost_tracks)} duplicate "
                      f"track(s), {len(clean.bench_tracks)} bench/sideline track(s), {len(clean.calib_gaps)} "
                      f"calibration gap(s)"}]
    prog(0.2, "Segmenting the game into possessions")
    segments = segment_game(g, clean, coached)
    log.append({"tier": "python", "task": "segmentation and statistics",
                "detail": f"{len(segments)} segments (possessions / dead balls) with ball path, passes, spacing, "
                          "defenders behind the ball, rule-based phase labels"
                          + (" from the basketball game state (possessions, free throws, attack direction)"
                             if g.gs and any("kind" in s for s in segments) else "")})
    notes: list[str] = []
    info: dict = {}
    used = False
    if jev is None:
        notes.append("Light-work model (TypeSafe AI Jev) not configured: segments were labelled by rules in "
                     "Python, and Gemini received the full tracking detail.")
    elif segments:
        try:
            r = label_with_jev(jev, g, coached, segments, clean,
                               lambda f, m: prog(0.3 + 0.7 * f, m), cancel)
            used = True
            info = {"model": jev.model_used or jev.model, "usage": dict(jev.usage), **r}
            log.append({"tier": "light", "task": "segment labelling",
                        "detail": f"{PROVIDER_NAME} {info['model']}: phase, defensive scheme, worth-reviewing and "
                                  f"reliability for {r['segments_labelled']} segment(s); "
                                  f"{r['tracks_decided']} uncertain track(s) classified; {r['requests']} request(s), "
                                  f"{jev.usage['input_tokens']} input tokens"})
        except LightError as e:
            notes.append(f"Light-work model failed ({e}); continued with the rule-based labels and full detail.")
            for s in segments:
                s["phase_source"], s["review"] = "rules", None
    events = light_events(segments, coached, (g.gs or {}).get("events")) if used else []
    if events:
        log.append({"tier": "light", "task": "turnovers", "detail": f"{len(events)} lost ball(s) labelled by "
                    f"{PROVIDER_NAME} that the rules had not found (shown on the Auto detection track)"})
    prog(1.0, "Light work done")
    return LightResult(g, clean, segments, used, log, notes, info, events)


SEG_COLUMNS = ["id", "t0", "t1", "phase", "src", "ball", "attacks", "cross_s", "passes", "chg", "c_vis", "o_vis",
               "c_spread", "o_spread", "def_behind", "scheme", "review", "reliable", "baskets", "kind", "start", "end",
               "lost", "events"]


def segment_table(segments: list[dict]) -> str:
    """Compact pipe-separated table of the segments for the Gemini prompt."""
    lines = ["|".join(SEG_COLUMNS)]
    for s in segments:
        defending = "coached" if s["ball_team"] == "opponent" else "opponent" if s["ball_team"] == "coached" else ""
        vals = [s["id"], f"{s['t0']:.1f}", f"{s['t1']:.1f}", s["phase"],
                s["phase_source"] + (f"{s['phase_confidence']:.2f}" if s.get("phase_confidence") is not None else ""),
                s["ball_team"], s.get("attacks") or "", s.get("crossed_half_after_s"), s["passes"],
                "y" if s["possession_change"] else "", s.get("coached_visible"), s.get("opponent_visible"),
                s.get("coached_spread_m"), s.get("opponent_spread_m"),
                s.get(f"{defending}_defenders_behind_ball") if defending else None, s.get("defense_scheme") or "",
                s.get("review"), s.get("reliable"),
                ";".join(f"{b['t']}{b['team'][0]}" for b in s.get("baskets", [])),
                s.get("kind") or "", s.get("start") or "", s.get("end") or "", s.get("lost_ball"),
                "; ".join(s.get("events") or [])]
        lines.append("|".join("" if v is None else str(v) for v in vals))
    return "\n".join(lines)
