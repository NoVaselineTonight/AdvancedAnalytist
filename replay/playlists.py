"""Clip playlists from events: "all our turnovers", "all their fast breaks", "every possession of #12", ...

No Qt here (unit-testable). The GUI is ``replay/gui/playlists.py``.

* :func:`collect` gathers every taggable moment of a Replay folder into a :class:`Catalog` of :class:`Item` s:
  detected baskets (``replay_data.json`` events, annotated by the game state), game-state events, possessions and
  free-throw windows (``results/game_state.json``), ball events (passes, bad passes, steals, shots with distance /
  outcome - ``replay/ball_events.py``, part of the game state when present), the light AI model's events and every
  mark of every user timeline track (``timeline.json``: Notes, Highlights, the "AI analysis" track, ...).
  Every item carries ``cats`` = {category: team or None}: one moment can belong to several categories with a
  different team each (a steal is a *steal* of the team that won the ball and a *turnover* of the team that lost it).
* :class:`Filter` selects items (categories, team, player by number / on-court slot / track id, period, time range,
  confidence, shot distance, text); :func:`build_clips` turns them into clips with pre-/post-roll and merges
  overlapping clips.
* :class:`Playlist` / :class:`PlaylistBook` are stored in ``<Replay>/playlists.json`` (so they travel inside the
  ``.aareplay`` package, which contains the whole Replay folder). Clips keep the coach's *reviewed* flag and note
  across rebuilds from the filter.
* Export: one MP4 (ffmpeg from imageio_ffmpeg: every clip re-encoded with identical settings, optional title cards,
  then joined with the concat demuxer without re-encoding) and a CSV of timestamps.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

FORMAT = "advanced-analytist/playlists@1"
FILENAME = "playlists.json"
TEAMS = ("A", "B")
AI_TRACK_ID = "ai-analysis"  # replay/ai_analysis.py AI_TRACK_ID
NOT_IN_PLAY_TRACK_ID = "notinplay"  # replay/gui/timeline_model.py DEAD_TRACK_ID

# category key -> label (coach wording). "track:<id>" categories are added per timeline track.
CATEGORIES: list[tuple[str, str]] = [
    ("possession", "Possessions"),
    ("fast_break", "Fast breaks"),
    ("half_court", "Half-court sets"),
    ("shot", "Shots (all attempts)"),
    ("made", "Made baskets"),
    ("missed", "Missed shots"),
    ("free_throw", "Free throws"),
    ("turnover", "Turnovers"),
    ("steal", "Steals"),
    ("rebound", "Rebounds (all)"),
    ("off_rebound", "Offensive rebounds"),
    ("def_rebound", "Defensive rebounds"),
    ("pass", "Passes"),
    ("inbound", "Inbounds"),
    ("stoppage", "Stoppages, time-outs, breaks"),
    ("ai", "AI analysis issues"),
    ("note", "My timeline marks"),
    ("other", "Other detected events"),
]
CATEGORY_LABELS = dict(CATEGORIES)
# categories whose items never have a team: a team filter does not remove them
TEAMLESS = ("ai", "note")

FIXED_MARK = "(fixed list"  # filter text the Stats tab used for fixed lists before Playlist.fixed existed
DEFAULT_PRE_S = 4.0
DEFAULT_POST_S = 2.0


def new_id() -> str:
    return uuid.uuid4().hex[:12]


def _f(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f and abs(f) != float("inf") else None


def other_team(team: str | None) -> str | None:
    return {"A": "B", "B": "A"}.get(team or "")


def fmt_clock(s: float | None, decimals: int = 1) -> str:
    """Video time as m:ss.s (h:mm:ss.s past an hour)."""
    if s is None:
        return ""
    s = max(0.0, float(s))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    w = 3 + decimals if decimals else 2
    body = f"{sec:0{w}.{decimals}f}"
    return f"{int(h)}:{int(m):02d}:{body}" if h else f"{int(m)}:{body}"


def parse_clock(text: str | None) -> float | None:
    """'1:23.5' / '83.5' / '1:02:03' -> seconds; '' -> None. Raises ValueError for garbage."""
    text = (text or "").strip()
    if not text:
        return None
    parts = text.replace(",", ".").split(":")
    if len(parts) > 3:
        raise ValueError(text)
    total = 0.0
    for p in parts:
        total = total * 60 + float(p)
    if total < 0:
        raise ValueError(text)
    return total


# ====================================================================================== coach wording
def team_names(replay: dict | None, gs: dict | None = None) -> dict[str, str]:
    names = {k: k for k in TEAMS}
    for k, v in (((replay or {}).get("game") or {}).get("teams") or {}).items():
        if k in TEAMS and isinstance(v, dict) and v.get("name"):
            names[k] = str(v["name"])
    for k, v in ((gs or {}).get("teams") or {}).items():
        if k in TEAMS and names[k] == k and v:
            names[k] = str(v)
    return names


def team_label(team: str | None, coached: str | None, names: dict[str, str], short: bool = False) -> str:
    """"Our team (Team orange)" / "Opponent (Team purple)" when the coached team is known, else the team name."""
    if team not in TEAMS:
        return "" if short else "Both teams"
    name = names.get(team, team)
    if coached in TEAMS:
        role = "Our team" if team == coached else "Opponent"
        return role if short else f"{role} ({name})"
    return name


def coached_team(project_settings: dict | None) -> str | None:
    """The team the user coaches: set in the AI dialog (settings["ai"]["coached_team"]) or the playlist panel."""
    ai = (project_settings or {}).get("ai") or {}
    t = ai.get("coached_team") if isinstance(ai, dict) else None
    return t if t in TEAMS else None


# ====================================================================================== catalog
@dataclass
class Item:
    """One taggable moment (or span) of the game."""
    id: str
    t: float
    t_end: float
    title: str
    source: str  # basket | game_state | ball_events | jev | possession | free_throws | user | ai
    cats: dict[str, str | None] = field(default_factory=dict)
    confidence: float | None = None
    players: list[str] = field(default_factory=list)  # track ids named in the event / holding the ball
    distance_m: float | None = None
    track_id: str | None = None  # timeline track of a user mark
    note: str = ""

    def team_for(self, cat: str) -> str | None:
        return self.cats.get(cat)


@dataclass
class Catalog:
    items: list[Item] = field(default_factory=list)
    names: dict[str, str] = field(default_factory=lambda: {k: k for k in TEAMS})
    halves: list[dict] = field(default_factory=list)  # [{"t0", "t1", "label", "index"}]
    tracks: list[dict] = field(default_factory=list)  # timeline tracks [{"id", "name", "kind", "count"}]
    track_team: dict[str, str] = field(default_factory=dict)  # player track id -> "A" | "B"
    track_number: dict[str, str] = field(default_factory=dict)  # track id -> jersey number text
    track_slot: dict[str, str] = field(default_factory=dict)  # track id -> lineup slot "A3"
    track_span: dict[str, tuple[float, float]] = field(default_factory=dict)  # track id -> (t first, t last)
    slot_spans: dict[str, list[tuple[str, float, float]]] = field(default_factory=dict)  # "A3" -> [(tid, t0, t1)]
    held: list[tuple[float, float, str]] = field(default_factory=list)  # ball held: (t0, t1, track id)
    duration: float | None = None

    def category_counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for it in self.items:
            for c in it.cats:
                out[c] = out.get(c, 0) + 1
        return out

    def category_label(self, cat: str) -> str:
        if cat.startswith("track:"):
            tid = cat[6:]
            tr = next((t for t in self.tracks if t["id"] == tid), None)
            return f"Timeline track: {tr['name'] if tr else tid}"
        return CATEGORY_LABELS.get(cat, cat)

    def player_choices(self) -> list[tuple[str, str]]:
        """(key, label) of the players a filter can name: jersey numbers per team, then on-court slots."""
        out = []
        nums = sorted({(self.track_team.get(t, ""), n) for t, n in self.track_number.items()},
                      key=lambda tn: (tn[0], int(tn[1]) if tn[1].isdigit() else 999, tn[1]))
        for team, num in nums:
            if team in TEAMS:
                out.append((f"num:{team}:{num}", f"#{num} – {self.names.get(team, team)}"))
        for slot in sorted(self.slot_spans):
            out.append((f"slot:{slot}", f"Slot {slot} – {self.names.get(slot[:1], slot[:1])} (on-court position "
                                        f"{slot[1:]}, carried across tracker IDs)"))
        if getattr(self, "roster_map", None) is not None:  # roster hook: '#23 Virtanen – Team'
            from replay.roster_names import decorate_choices
            out = decorate_choices(self, out)  # roster hook
        return out

    def player_label(self, key: str) -> str:
        return dict(self.player_choices()).get(key) or describe_player_key(key)

    def tracks_of(self, key: str) -> set[str]:
        """Player track ids matching a player key (num:[team:]N, slot:A3, trk:42)."""
        kind, _, rest = key.partition(":")
        if kind == "trk":
            return {rest}
        if kind == "slot":
            return {tid for tid, s in self.track_slot.items() if s == rest} | \
                {tid for tid, _a, _b in self.slot_spans.get(rest, [])}
        if kind == "num":
            team, _, num = rest.rpartition(":")
            return {tid for tid, n in self.track_number.items()
                    if n == num and (not team or self.track_team.get(tid) == team)}
        return set()

    def on_court(self, key: str, t0: float, t1: float) -> bool:
        """The player (key) is tracked / in the lineup slot at some moment of [t0, t1]."""
        kind, _, rest = key.partition(":")
        if kind == "slot":
            return any(a <= t1 and b >= t0 for _tid, a, b in self.slot_spans.get(rest, []))
        for tid in self.tracks_of(key):
            sp = self.track_span.get(tid)
            if sp and sp[0] <= t1 and sp[1] >= t0:
                return True
        return False

    def holders(self, t0: float, t1: float) -> set[str]:
        return {tid for a, b, tid in self.held if a <= t1 and b >= t0}


def describe_player_key(key: str) -> str:
    kind, _, rest = key.partition(":")
    if kind == "num":
        return f"#{rest.rpartition(':')[2]}"
    if kind == "slot":
        return f"Slot {rest}"
    if kind == "trk":
        return f"Track {rest}"
    return key


def parse_player(text: str) -> str | None:
    """User text -> player key: '#12' / '12' -> num:12 (either team), 'A3' -> slot:A3, 'T42' -> trk:42.
    An existing key (num:A:12 ...) is returned unchanged."""
    s = (text or "").strip()
    if not s:
        return None
    if s.split(":")[0] in ("num", "slot", "trk") and ":" in s:
        return s
    s0 = s.split(" ")[0].strip(",")
    if s0.startswith("#") and s0[1:].isdigit():
        return f"num:{s0[1:]}"
    if s0.isdigit():
        return f"num:{s0}"
    if len(s0) >= 2 and s0[0].upper() in TEAMS and s0[1:].isdigit():
        return f"slot:{s0[0].upper()}{s0[1:]}"
    if len(s0) >= 2 and s0[0] in "Tt" and s0[1:].isdigit():
        return f"trk:{s0[1:]}"
    return None


# --------------------------------------------------------------------------- event -> categories
_PLAYER_KEYS = ("shooter_track", "rebounder_track", "passer_track", "receiver_track", "stealer_track",
                "lost_by_track", "player_track")
_STOPPAGES = ("timeout", "stoppage", "quarter_break", "half_time", "jump_ball")
_TURNOVER_TYPES = ("turnover", "bad_pass", "lost_ball", "losing_the_ball", "lost_possession")


def _event_cats(ev: dict, shots: Sequence[dict]) -> tuple[dict[str, str | None], float | None, float | None]:
    """(categories, shot distance, shot end time) of one detected event."""
    typ = str(ev.get("type") or "")
    team = ev.get("team") if ev.get("team") in TEAMS else None
    d = ev.get("details") or {}
    cats: dict[str, str | None] = {}
    dist = _f(d.get("distance_m"))
    t_end = None
    if typ == "basket":
        ann = d.get("game_state") or {}
        cats.update(made=team, shot=team)
        if ann.get("kind") == "free_throw" or "free throw" in str(ev.get("label") or "").lower():
            cats["free_throw"] = team
        dist = _f(ann.get("shot_distance_m")) if dist is None else dist
    elif typ == "shot":
        cats["shot"] = team
        t = _f(ev.get("t")) or 0.0
        sh = next((s for s in shots if abs((_f(s.get("t_release")) or -99) - t) <= 0.2), None)
        outcome = d.get("outcome")
        if outcome not in ("made", "missed") and sh is not None:
            outcome = "made" if sh.get("made") else "missed" if sh.get("missed") else None
        if outcome == "made":
            cats["made"] = team
        elif outcome == "missed":
            cats["missed"] = team
        if sh is not None and sh.get("free_throw"):
            cats["free_throw"] = team
        if dist is None and sh is not None:
            dist = _f(sh.get("distance_m"))
        t_end = _f(d.get("t_at_rim")) or (_f(sh.get("t_rim")) if sh is not None else None)
    elif typ in ("free_throw", "free_throws"):
        cats["free_throw"] = team
        if typ == "free_throw":
            cats["shot"] = team
    elif typ in _TURNOVER_TYPES:
        cats["turnover"] = team
    elif typ == "steal":
        cats["steal"] = team
        lost = d.get("lost_by") if d.get("lost_by") in TEAMS else other_team(team)
        cats["turnover"] = lost
    elif typ in ("off_rebound", "def_rebound", "rebound"):
        cats["rebound"] = team
        if typ != "rebound":
            cats[typ] = team
    elif typ in ("fast_break", "half_court", "pass", "inbound"):
        cats[typ] = team
    elif typ in _STOPPAGES:
        cats["stoppage"] = team
    else:
        cats["other"] = team
    return cats, dist, t_end


def _event_players(ev: dict) -> list[str]:
    d = ev.get("details") or {}
    ann = d.get("game_state") or {}
    out = []
    for k in _PLAYER_KEYS:
        for v in (d.get(k), ann.get(k)):
            if v not in (None, "") and str(v) not in out:
                out.append(str(v))
    return out


def _read_json(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def collect(project_root: Path | None = None, replay: dict | None = None, gs: dict | None = None,
            timeline: dict | None = None, light: Sequence[dict] | None = None, duration: float | None = None,
            max_build_frames: int | None = 6000) -> Catalog:
    """Everything a playlist can be built from. Missing inputs are read from ``project_root`` (the game state is
    rebuilt like the timeline does when it is stale and the Replay is short enough)."""
    root = Path(project_root) if project_root is not None else None
    if replay is None and root is not None:
        replay = _read_json(root / "results" / "replay_data.json")
    replay = replay if isinstance(replay, dict) else {}
    if gs is None and root is not None and replay.get("t"):
        try:
            from replay import game_state
            gs = game_state.load_or_build(root / "results", replay, max_build_frames)
        except Exception as e:  # noqa: BLE001 - a playlist of user marks still works
            print("playlists: game state unavailable:", type(e).__name__, e)
            gs = None
    gs = gs if isinstance(gs, dict) else {}
    if timeline is None and root is not None:
        timeline = _read_json(root / "timeline.json")
    timeline = timeline if isinstance(timeline, dict) else {}
    if light is None and root is not None:
        ai = _read_json(root / "results" / "ai_analysis.json")
        light = [e for e in ((ai or {}).get("light_events") or []) if isinstance(e, dict)] if isinstance(ai, dict) \
            else []

    cat = Catalog(names=team_names(replay, gs))
    ts = replay.get("t") or []
    cat.duration = duration if duration is not None else (_f(ts[-1]) if ts else None)
    cat.halves = [h for h in gs.get("halves") or [] if isinstance(h, dict)]

    # ---- players
    for tid, info in (replay.get("track_info") or {}).items():
        if not isinstance(info, dict):
            continue
        if info.get("team") in TEAMS:
            cat.track_team[str(tid)] = info["team"]
        if info.get("number") not in (None, ""):
            cat.track_number[str(tid)] = str(info["number"])
    for tr in replay.get("tracks") or []:
        tid = str(tr.get("id"))
        if tr.get("team") in TEAMS:
            cat.track_team.setdefault(tid, tr["team"])
        i0, n = int(tr.get("i0") or 0), len(tr.get("x") or [])
        if n and ts and i0 < len(ts):
            cat.track_span[tid] = (float(ts[i0]), float(ts[min(len(ts) - 1, i0 + n - 1)]))
    lu = gs.get("lineup") or {}
    for slot, runs in (lu.get("slots") or {}).items():
        spans = []
        for r in runs or []:
            if isinstance(r, (list, tuple)) and len(r) >= 3 and _f(r[1]) is not None and _f(r[2]) is not None:
                spans.append((str(r[0]), float(r[1]), float(r[2])))
        cat.slot_spans[str(slot)] = spans
    cat.track_slot = {str(k): str(v) for k, v in (lu.get("slot_of_track") or {}).items()}
    be = gs.get("ball_events") or {}
    for st in be.get("states") or []:
        if isinstance(st, (list, tuple)) and len(st) >= 4 and st[2] == "held" and st[3] not in (None, ""):
            cat.held.append((float(st[0]), float(st[1]), str(st[3])))

    # ---- detected events (same merge as the timeline's Auto detection track)
    from replay.game_state import merge_events
    events = merge_events(replay.get("events") or [], gs, light or [])
    overrides = (timeline.get("auto_overrides") or {}) if isinstance(timeline.get("auto_overrides"), dict) else {}
    shots = [s for s in gs.get("shots") or [] if isinstance(s, dict)]
    from replay.game_state import event_title
    for ev in events:
        eid = str(ev.get("id"))
        ov = overrides.get(eid) or {}
        if ov.get("dismissed"):  # the coach removed this mark from the timeline: it was wrong
            continue
        t = _f(ev.get("t"))
        if t is None:
            continue
        cats, dist, t_end = _event_cats(ev, shots)
        dur = max(0.0, _f(ev.get("duration")) or 0.0)
        end = max(t + dur, t_end or t)
        src = str(ev.get("source") or "auto")
        src = {"auto": "basket"}.get(src, src)
        if ov.get("title"):
            title = str(ov["title"])
        elif ev.get("label"):  # team name looked up now (renamed teams show)
            title = event_title(ev, cat.names)
        else:
            from replay.gui.timeline_model import event_title as tl_title
            title = tl_title(ev, cat.names)
        cat.items.append(Item(id=eid, t=t, t_end=end, title=title, source=src, cats=cats,
                              confidence=_f(ev.get("confidence")), players=_event_players(ev), distance_m=dist,
                              note=str(ov.get("note") or "")))

    # ---- possessions (spans) and free-throw windows
    for k, p in enumerate(gs.get("possessions") or []):
        t0, t1 = _f(p.get("t0")), _f(p.get("t1"))
        if t0 is None or t1 is None:
            continue
        team = p.get("team") if p.get("team") in TEAMS else None
        how = str(p.get("start") or "").replace("_", " ")
        title = f"Possession – {cat.names.get(team, team) if team else '?'}" + (f" (after {how})" if how.startswith(
            ("shot", "made")) else f" ({how})" if how and how != "start" else "") + (" – fast break" if p.get(
                "fast_break") else "")
        players = sorted(cat.holders(t0, t1), key=lambda x: (len(x), x))
        cat.items.append(Item(id=f"poss-{p.get('id', k)}", t=t0, t_end=max(t0, t1), title=title, source="possession",
                              cats={"possession": team}, confidence=_f(p.get("confidence")), players=players))
    if not any("free_throw" in it.cats for it in cat.items):
        for k, w in enumerate(gs.get("free_throw_windows") or []):
            t0, t1 = _f(w.get("t0")), _f(w.get("t1"))
            if t0 is None or t1 is None:
                continue
            team = w.get("team") if w.get("team") in TEAMS else None
            cat.items.append(Item(id=f"ftw-{k}", t=t0, t_end=t1, source="free_throws",
                                  title=f"Free throws – {cat.names.get(team, team) if team else '?'}",
                                  cats={"free_throw": team}, confidence=_f(w.get("confidence")),
                                  players=[str(x) for x in w.get("shooter_tracks") or []]))

    # ---- timeline tracks (user marks, AI analysis marks, ...); the auto track = the detected events above
    auto_cut = None
    try:
        from replay.game_state import SOURCE as GS_SOURCE, TIMELINE_MIN_CONF
        auto_cut = (GS_SOURCE, TIMELINE_MIN_CONF)
    except ImportError:
        pass
    for tr in timeline.get("tracks") or []:
        if not isinstance(tr, dict):
            continue
        tid = str(tr.get("id") or "")
        name = str(tr.get("name") or "Track")
        if tr.get("kind") == "auto":
            n = 0
            for it in cat.items:  # same rule as game_state.auto_events: very uncertain rule events are hidden
                if it.source in ("possession", "free_throws"):
                    continue
                if auto_cut and it.source in (auto_cut[0], "jev") and (it.confidence or 0.0) < auto_cut[1]:
                    continue
                it.cats[f"track:{tid}"] = next(iter(it.cats.values()), None)
                n += 1
            cat.tracks.append({"id": tid, "name": name, "kind": "auto", "count": n})
            continue
        marks = [m for m in tr.get("marks") or [] if isinstance(m, dict) and _f(m.get("t")) is not None]
        cat.tracks.append({"id": tid, "name": name, "kind": "user", "count": len(marks)})
        for m in marks:
            t = float(m["t"])
            dur = max(0.0, _f(m.get("duration")) or 0.0)
            c: dict[str, str | None] = {f"track:{tid}": None}
            if tid == AI_TRACK_ID:
                c["ai"] = None
            elif tid != NOT_IN_PLAY_TRACK_ID:
                c["note"] = None
            cat.items.append(Item(id=f"mark-{m.get('id') or new_id()}", t=t, t_end=t + dur,
                                  title=str(m.get("title") or name), source="ai" if tid == AI_TRACK_ID else "user",
                                  cats=c, track_id=tid, note=str(m.get("note") or "")))
    cat.items.sort(key=lambda it: (it.t, it.id))
    if root is not None:  # roster hook: player names in titles / player choices (only with roster.json)
        from replay.roster_names import decorate_catalog
        decorate_catalog(root, cat)  # roster hook
    return cat


# ====================================================================================== filter
@dataclass
class Filter:
    categories: list[str] = field(default_factory=list)  # empty = every category
    team: str | None = None  # "A" | "B" | None (both)
    players: list[str] = field(default_factory=list)  # player keys (num:A:12, slot:A3, trk:42); any of them
    player_mode: str = "involved"  # "involved" = named in the event / had the ball; "on_court" = on the floor
    period: int | None = None  # index into the game state's halves
    t_from: float | None = None
    t_to: float | None = None
    min_confidence: float = 0.0  # 0..1; items without a confidence (user marks) always pass
    min_distance_m: float | None = None  # shot distance (items without a distance fail when set)
    max_distance_m: float | None = None
    text: str = ""  # case-insensitive substring of the title / note

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Any) -> "Filter":
        d = d if isinstance(d, dict) else {}
        f = cls()
        f.categories = [str(c) for c in d.get("categories") or [] if c]
        f.team = d.get("team") if d.get("team") in TEAMS else None
        f.players = [str(p) for p in d.get("players") or [] if p]
        f.player_mode = "on_court" if d.get("player_mode") == "on_court" else "involved"
        f.period = int(d["period"]) if isinstance(d.get("period"), (int, float)) else None
        f.t_from, f.t_to = _f(d.get("t_from")), _f(d.get("t_to"))
        f.min_confidence = min(1.0, max(0.0, _f(d.get("min_confidence")) or 0.0))
        f.min_distance_m, f.max_distance_m = _f(d.get("min_distance_m")), _f(d.get("max_distance_m"))
        f.text = str(d.get("text") or "")
        return f

    def describe(self, cat: Catalog | None = None, coached: str | None = None) -> str:
        names = cat.names if cat else {k: k for k in TEAMS}
        what = ", ".join((cat.category_label(c) if cat else CATEGORY_LABELS.get(c, c)) for c in self.categories) \
            or "Everything"
        parts = [what]
        if self.team:
            parts.append(team_label(self.team, coached, names))
        if self.players:
            ps = ", ".join(cat.player_label(p) if cat else describe_player_key(p) for p in self.players)
            parts.append(f"{ps} ({'on court' if self.player_mode == 'on_court' else 'involved'})")
        if self.period is not None and cat and 0 <= self.period < len(cat.halves):
            parts.append(str(cat.halves[self.period].get("label") or f"period {self.period + 1}"))
        if self.t_from is not None or self.t_to is not None:
            parts.append(f"{fmt_clock(self.t_from or 0, 0)}–{fmt_clock(self.t_to, 0) if self.t_to else 'end'}")
        if self.min_distance_m:
            parts.append(f"> {self.min_distance_m:g} m")
        if self.max_distance_m:
            parts.append(f"< {self.max_distance_m:g} m")
        if self.min_confidence > 0:
            parts.append(f"confidence ≥ {self.min_confidence * 100:.0f}%")
        if self.text:
            parts.append(f"“{self.text}”")
        return " · ".join(parts)


def _matched_cat(it: Item, flt: Filter) -> str | None:
    cats = flt.categories or list(it.cats)
    for c in cats:
        if c not in it.cats:
            continue
        team = it.cats[c]
        if flt.team and team != flt.team:
            if team is None and (c in TEAMLESS or c.startswith("track:")):
                return c
            continue
        return c
    return None


def apply_filter(cat: Catalog, flt: Filter) -> list[Item]:
    """The catalog items that pass ``flt`` (time order)."""
    lo, hi = flt.t_from, flt.t_to
    if flt.period is not None and 0 <= flt.period < len(cat.halves):
        h = cat.halves[flt.period]
        lo = max(lo if lo is not None else -1e9, _f(h.get("t0")) or 0.0)
        hi = min(hi if hi is not None else 1e18, _f(h.get("t1")) or 1e18)
    keys = list(flt.players)
    key_tracks = {k: cat.tracks_of(k) for k in keys}
    text = flt.text.strip().lower()
    out = []
    for it in cat.items:
        if _matched_cat(it, flt) is None:
            continue
        if lo is not None and it.t_end < lo:
            continue
        if hi is not None and it.t > hi:
            continue
        if flt.min_confidence > 0 and it.confidence is not None and it.confidence < flt.min_confidence - 1e-9:
            continue
        if flt.min_distance_m is not None and flt.min_distance_m > 0:
            if it.distance_m is None or it.distance_m < flt.min_distance_m:
                continue
        if flt.max_distance_m is not None and flt.max_distance_m > 0:
            if it.distance_m is None or it.distance_m > flt.max_distance_m:
                continue
        if keys:
            if flt.player_mode == "on_court":
                ok = any(cat.on_court(k, it.t, it.t_end) for k in keys)
            else:
                ok = any(set(it.players) & key_tracks[k] for k in keys)
            if not ok:
                continue
        if text and text not in it.title.lower() and text not in it.note.lower():
            continue
        out.append(it)
    return out


# ====================================================================================== clips
@dataclass
class Clip:
    t0: float
    t1: float
    title: str
    item_ids: list[str] = field(default_factory=list)
    titles: list[str] = field(default_factory=list)  # every merged item's title
    team: str | None = None  # the team the filter matched (None when mixed / teamless)
    reviewed: bool = False
    note: str = ""
    id: str = field(default_factory=new_id)

    @property
    def duration(self) -> float:
        return max(0.0, self.t1 - self.t0)

    def to_dict(self) -> dict:
        return {"id": self.id, "t0": round(self.t0, 3), "t1": round(self.t1, 3), "title": self.title,
                "item_ids": self.item_ids, "titles": self.titles, "team": self.team, "reviewed": self.reviewed,
                "note": self.note}

    @classmethod
    def from_dict(cls, d: dict) -> "Clip":
        t0 = max(0.0, _f(d.get("t0")) or 0.0)
        return cls(t0=t0, t1=max(t0, _f(d.get("t1")) or t0), title=str(d.get("title") or ""),
                   item_ids=[str(x) for x in d.get("item_ids") or []], titles=[str(x) for x in d.get("titles") or []],
                   team=d.get("team") if d.get("team") in TEAMS else None, reviewed=bool(d.get("reviewed")),
                   note=str(d.get("note") or ""), id=str(d.get("id") or new_id()))


def clip_window(it: Item, pre_s: float, post_s: float, duration: float | None = None) -> tuple[float, float]:
    a = max(0.0, it.t - max(0.0, pre_s))
    b = max(it.t_end, it.t) + max(0.0, post_s)
    if duration is not None and duration > 0:
        b = min(b, duration)
        a = min(a, b)
    return a, b


def build_clips(items: Iterable[Item], pre_s: float = DEFAULT_PRE_S, post_s: float = DEFAULT_POST_S,
                merge_gap_s: float = 0.0, duration: float | None = None, flt: Filter | None = None) -> list[Clip]:
    """One clip per item ([t - pre, end + post]); clips that overlap (or are less than ``merge_gap_s`` apart)
    are merged into one, so nothing is shown twice."""
    wins = sorted(((*clip_window(it, pre_s, post_s, duration), it) for it in items), key=lambda w: (w[0], w[1]))
    clips: list[Clip] = []
    for a, b, it in wins:
        team = it.cats.get(_matched_cat(it, flt)) if flt is not None else next(iter(it.cats.values()), None)
        if clips and a <= clips[-1].t1 + max(0.0, merge_gap_s):
            c = clips[-1]
            c.t1 = max(c.t1, b)
            c.item_ids.append(it.id)
            c.titles.append(it.title)
            if c.team != team:
                c.team = None
            continue
        clips.append(Clip(t0=a, t1=b, title=it.title, item_ids=[it.id], titles=[it.title], team=team))
    for c in clips:
        uniq = list(dict.fromkeys(c.titles))
        c.title = uniq[0] + (f"  (+{len(uniq) - 1} more)" if len(uniq) > 1 else "")
    return clips


# ====================================================================================== playlists
@dataclass
class Playlist:
    name: str
    filter: Filter = field(default_factory=Filter)
    pre_s: float = DEFAULT_PRE_S
    post_s: float = DEFAULT_POST_S
    merge_gap_s: float = 0.0
    clips: list[Clip] = field(default_factory=list)
    created: float = field(default_factory=time.time)
    updated: float = field(default_factory=time.time)
    id: str = field(default_factory=new_id)
    fixed: bool = False  # a hand-made clip list (e.g. from the Stats tab): never rebuilt from the filter
    origin: str = ""  # who made it, e.g. "stats"

    def __post_init__(self) -> None:
        if FIXED_MARK in (self.filter.text or ""):  # Stats tab hand-off before the "fixed" flag existed
            self.fixed = True
            self.filter.text = ""

    @property
    def total_s(self) -> float:
        return sum(c.duration for c in self.clips)

    @property
    def reviewed_count(self) -> int:
        return sum(1 for c in self.clips if c.reviewed)

    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "filter": self.filter.to_dict(), "pre_s": self.pre_s,
                "post_s": self.post_s, "merge_gap_s": self.merge_gap_s, "created": round(self.created, 3),
                "updated": round(self.updated, 3), "fixed": self.fixed, "origin": self.origin,
                "clips": [c.to_dict() for c in self.clips]}

    @classmethod
    def from_dict(cls, d: dict) -> "Playlist":
        return cls(name=str(d.get("name") or "Playlist"), filter=Filter.from_dict(d.get("filter")),
                   pre_s=max(0.0, _f(d.get("pre_s")) if _f(d.get("pre_s")) is not None else DEFAULT_PRE_S),
                   post_s=max(0.0, _f(d.get("post_s")) if _f(d.get("post_s")) is not None else DEFAULT_POST_S),
                   merge_gap_s=max(0.0, _f(d.get("merge_gap_s")) or 0.0),
                   clips=sorted((Clip.from_dict(c) for c in d.get("clips") or [] if isinstance(c, dict)),
                                key=lambda c: c.t0),
                   created=_f(d.get("created")) or time.time(), updated=_f(d.get("updated")) or time.time(),
                   id=str(d.get("id") or new_id()), fixed=bool(d.get("fixed")), origin=str(d.get("origin") or ""))

    def rebuild(self, cat: Catalog) -> tuple[int, int]:
        """Re-run the filter on (new) analysis results. Reviewed flags and notes stay with the clip that contains
        the same events (else with an overlapping clip). Returns (clips before, clips after). A fixed list is
        left as it is."""
        before = len(self.clips)
        if self.fixed:
            return before, before
        new = build_clips(apply_filter(cat, self.filter), self.pre_s, self.post_s, self.merge_gap_s, cat.duration,
                          self.filter)
        old = list(self.clips)
        used: set[str] = set()
        for c in new:
            src = next((o for o in old if o.id not in used and set(o.item_ids) & set(c.item_ids)), None) or \
                next((o for o in old if o.id not in used and o.t0 < c.t1 and o.t1 > c.t0), None)
            if src is not None:
                used.add(src.id)
                c.id, c.reviewed, c.note = src.id, src.reviewed, src.note
        self.clips = new
        self.updated = time.time()
        return before, len(new)

    def index_of(self, clip_id: str) -> int:
        return next((i for i, c in enumerate(self.clips) if c.id == clip_id), -1)


def make_playlist(cat: Catalog, name: str, flt: Filter, pre_s: float = DEFAULT_PRE_S,
                  post_s: float = DEFAULT_POST_S, merge_gap_s: float = 0.0) -> Playlist:
    pl = Playlist(name=name, filter=flt, pre_s=pre_s, post_s=post_s, merge_gap_s=merge_gap_s)
    pl.rebuild(cat)
    return pl


def fixed_playlist(name: str, moments: Iterable[dict], pre_s: float = DEFAULT_PRE_S,
                   post_s: float = DEFAULT_POST_S, duration: float | None = None, origin: str = "") -> Playlist:
    """A fixed clip list around given moments ({"t", optional "t1" (end), "title", "team"}), e.g. the Stats tab's
    lineup stints or shot-zone lists. Overlapping clips are merged like filter playlists."""
    items = []
    for k, m in enumerate(moments):
        t = _f(m.get("t"))
        if t is None:
            continue
        team = m.get("team") if m.get("team") in TEAMS else None
        end = max(t, _f(m.get("t1")) or t)
        items.append(Item(id=str(m.get("id") or f"m{k}"), t=t, t_end=end, title=str(m.get("title") or "Clip"),
                          source=origin or "fixed", cats={"fixed": team}))
    pl = Playlist(name=name, pre_s=pre_s, post_s=post_s, fixed=True, origin=origin)
    pl.clips = build_clips(items, pre_s, post_s, 0.0, duration, Filter(["fixed"]))
    return pl


class PlaylistBook:
    """All playlists of one Replay: ``<Replay>/playlists.json``."""

    def __init__(self, playlists: list[Playlist] | None = None):
        self.playlists: list[Playlist] = playlists or []

    def to_dict(self) -> dict:
        return {"format": FORMAT, "playlists": [p.to_dict() for p in self.playlists]}

    @classmethod
    def from_dict(cls, d: Any) -> "PlaylistBook":
        if not isinstance(d, dict) or not isinstance(d.get("playlists"), list):
            raise ValueError("not a playlists file")
        out, seen = [], set()
        for x in d["playlists"]:
            if isinstance(x, dict):
                p = Playlist.from_dict(x)
                if p.id in seen:
                    p.id = new_id()
                seen.add(p.id)
                out.append(p)
        return cls(out)

    @classmethod
    def load(cls, project_root: Path) -> "PlaylistBook":
        """A missing file = no playlists; a damaged one is kept aside as playlists.json.bad."""
        path = Path(project_root) / FILENAME
        if not path.exists():
            return cls()
        try:
            return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError, AttributeError):
            try:
                os.replace(path, path.with_suffix(path.suffix + ".bad"))
            except OSError:
                pass
            return cls()

    def save(self, project_root: Path) -> Path:
        path = Path(project_root) / FILENAME
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)
        return path

    def get(self, pid: str) -> Playlist | None:
        return next((p for p in self.playlists if p.id == pid), None)

    def add(self, pl: Playlist) -> Playlist:
        self.playlists.append(pl)
        return pl

    def remove(self, pid: str) -> bool:
        p = self.get(pid)
        if p is None:
            return False
        self.playlists.remove(p)
        return True


# ====================================================================================== presets
def presets(coached: str | None, names: dict[str, str]) -> list[tuple[str, Filter, float, float]]:
    """Ready-made filters (name, filter, pre-roll, post-roll) in coach wording."""
    us = coached if coached in TEAMS else "A"
    them = other_team(us)
    if coached in TEAMS:
        our, their = "our", "their"
    else:
        our, their = names.get(us, us), names.get(them, them)
    P = [
        (f"All {our} turnovers", Filter(["turnover"], us), 5.0, 2.0),
        (f"All {their} turnovers", Filter(["turnover"], them), 5.0, 2.0),
        (f"All {our} fast breaks", Filter(["fast_break"], us), 2.0, 2.0),
        (f"All {their} fast breaks", Filter(["fast_break"], them), 2.0, 2.0),
        (f"All made baskets by {names.get(us, us)}", Filter(["made"], us), 5.0, 2.0),
        (f"All made baskets by {names.get(them, them)}", Filter(["made"], them), 5.0, 2.0),
        (f"All {our} shots", Filter(["shot"], us), 4.0, 2.0),
        (f"All {their} shots", Filter(["shot"], them), 4.0, 2.0),
        ("All shots from > 6 m", Filter(["shot"], min_distance_m=6.0), 4.0, 2.0),
        ("All free throws", Filter(["free_throw"]), 2.0, 2.0),
        (f"All {our} possessions", Filter(["possession"], us), 1.0, 1.0),
        (f"All {their} possessions", Filter(["possession"], them), 1.0, 1.0),
        (f"All {our} offensive rebounds", Filter(["off_rebound"], us), 3.0, 3.0),
        (f"All {their} offensive rebounds", Filter(["off_rebound"], them), 3.0, 3.0),
        ("All AI analysis issues", Filter(["ai"]), 1.0, 1.0),
        ("All my timeline marks", Filter(["note"]), 3.0, 3.0),
    ]
    if coached not in TEAMS:  # "our"/"their" make no sense without a coached team: name both teams
        P = [(n[0].upper() + n[1:], f, a, b) for n, f, a, b in P]
    return P


# ====================================================================================== export
def export_csv(pl: Playlist, dst: Path, names: dict[str, str] | None = None, coached: str | None = None) -> Path:
    """One row per clip: times in seconds and m:ss.s (video time), title, events, team, reviewed, note.
    UTF-8 with BOM so Excel shows ä/ö correctly."""
    names = names or {k: k for k in TEAMS}
    dst = Path(dst)
    with open(dst, "w", newline="", encoding="utf-8-sig") as fo:
        w = csv.writer(fo)
        w.writerow(["clip", "start_s", "end_s", "duration_s", "start", "end", "title", "events", "team", "reviewed",
                    "note"])
        for i, c in enumerate(pl.clips, 1):
            w.writerow([i, f"{c.t0:.2f}", f"{c.t1:.2f}", f"{c.duration:.2f}", fmt_clock(c.t0), fmt_clock(c.t1),
                        c.title, " | ".join(dict.fromkeys(c.titles)), team_label(c.team, coached, names, short=False)
                        if c.team else "", "yes" if c.reviewed else "", c.note.replace("\r", " ").replace("\n", " ")])
    return dst


def ffmpeg_exe() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def fps_arg(fps: float | None) -> str:
    """'30000/1001' for 29.97 etc. (ffmpeg rational)."""
    f = float(fps or 30.0)
    for base in (24, 30, 48, 60, 120):  # NTSC rates (29.97 -> 30000/1001)
        if abs(f - base * 1000 / 1001) < 0.006:
            return f"{base * 1000}/1001"
    fr = Fraction(f).limit_denominator(1001)
    return f"{fr.numerator}/{fr.denominator}" if fr.denominator != 1 else str(fr.numerator)


def _even(v: int) -> int:
    return max(2, int(v) // 2 * 2)


# the same encoder settings for every part, so the parts can be joined without re-encoding
def _venc(fps: float, size: tuple[int, int], preset: str = "veryfast") -> list[str]:
    w, h = _even(size[0]), _even(size[1])
    return ["-vf", f"scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1,"
                   f"fps={fps_arg(fps)}",
            "-c:v", "libx264", "-preset", preset, "-crf", "20" if preset == "veryfast" else "22", "-pix_fmt", "yuv420p", "-profile:v", "high",
            "-video_track_timescale", "90000"]


_AENC = ["-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-ac", "2"]


def segment_cmd(ffmpeg: str, video: Path | str, t0: float, t1: float, out: Path | str, *, fps: float,
                size: tuple[int, int], audio: bool) -> list[str]:
    """Cut [t0, t1] of ``video`` (frame-accurate: input seek + re-encode)."""
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-ss", f"{max(0.0, t0):.3f}",
           "-i", str(video), "-t", f"{max(0.05, t1 - t0):.3f}", "-map", "0:v:0"]
    if audio:
        cmd += ["-map", "0:a:0?"] + _AENC
    else:
        cmd += ["-an"]
    return cmd + _venc(fps, size) + ["-sn", "-dn", "-movflags", "+faststart", str(out)]


def card_cmd(ffmpeg: str, image: Path | str, seconds: float, out: Path | str, *, fps: float, size: tuple[int, int],
             audio: bool) -> list[str]:
    """A still title card (PNG) as a video part of ``seconds`` (silent audio when the clips have audio)."""
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-loop", "1", "-framerate",
           fps_arg(fps), "-i", str(image)]
    if audio:
        cmd += ["-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000", "-map", "0:v:0",
                "-map", "1:a:0"] + _AENC
    else:
        cmd += ["-an"]
    return cmd + ["-t", f"{max(0.2, seconds):.3f}"] + _venc(fps, size) + ["-movflags", "+faststart", str(out)]


def concat_list(paths: Sequence[Path | str]) -> str:
    """The concat demuxer's list file (paths quoted; single quotes escaped)."""
    lines = []
    for p in paths:
        s = Path(p).resolve().as_posix().replace("'", "'\\''")
        lines.append(f"file '{s}'")
    return "\n".join(lines) + "\n"


def concat_cmd(ffmpeg: str, list_file: Path | str, out: Path | str) -> list[str]:
    return [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-f", "concat", "-safe", "0",
            "-i", str(list_file), "-c", "copy", "-movflags", "+faststart", str(out)]


def _font(size: int):
    from PIL import ImageFont
    for name in ("segoeuib.ttf", "arialbd.ttf", "seguisb.ttf", "arial.ttf", "DejaVuSans-Bold.ttf",
                 "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def render_card(path: Path, size: tuple[int, int], title: str, lines: Sequence[str] = (),
                color: str = "#0f172a") -> Path:
    """A title card PNG: big centred title, smaller lines below (any Unicode text: ä, ö, –)."""
    from PIL import Image, ImageDraw
    w, h = _even(size[0]), _even(size[1])
    img = Image.new("RGB", (w, h), color)
    d = ImageDraw.Draw(img)
    big, small = _font(max(18, h // 13)), _font(max(12, h // 26))

    def fit(text: str, font, maxw: int) -> str:
        if d.textlength(text, font=font) <= maxw:
            return text
        while text and d.textlength(text + "…", font=font) > maxw:
            text = text[:-1]
        return text + "…"

    rows = [(fit(title, big, int(w * 0.9)), big, "#ffffff")] + \
        [(fit(s, small, int(w * 0.9)), small, "#cbd5e1") for s in lines if s]
    heights = [d.textbbox((0, 0), t or " ", font=f)[3] for t, f, _c in rows]
    gap = h // 40
    y = (h - sum(heights) - gap * (len(rows) - 1)) // 2
    for (t, f, c), hh in zip(rows, heights):
        d.text(((w - d.textlength(t, font=f)) / 2, y), t, font=f, fill=c)
        y += hh + gap
    img.save(path)
    return Path(path)


def _vfilter(fps: float, size: tuple[int, int]) -> str:
    w, h = _even(size[0]), _even(size[1])
    return (f"scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1,"
            f"fps={fps_arg(fps)},format=yuv420p")


BATCH_PARTS = 16  # parts (clips + cards) per ffmpeg run


def batch_cmd(ffmpeg: str, video: Path | str, parts: Sequence[dict], out: Path | str, *, fps: float,
              size: tuple[int, int], audio: bool, preset: str = "veryfast") -> list[str]:
    """ONE ffmpeg run for several parts: every clip is an input of its own (fast input seek, frame-accurate cut),
    every title card a looped PNG (``part["image"]``), joined by the concat filter and encoded once. Starting
    ffmpeg costs seconds on Windows (virus scanner), so this is much faster than one run per part. The encoder
    settings equal :func:`segment_cmd`'s, so several batch outputs can be joined with :func:`concat_cmd`."""
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y"]
    vf = _vfilter(fps, size)
    filt, labels, n_in = [], [], 0
    for k, p in enumerate(parts):
        if p["kind"] == "card":
            sec = f"{max(0.2, float(p['seconds'])):.3f}"
            cmd += ["-loop", "1", "-framerate", fps_arg(fps), "-t", sec, "-i", str(p["image"])]
            vi = n_in
            n_in += 1
            if audio:
                cmd += ["-f", "lavfi", "-t", sec, "-i", "anullsrc=channel_layout=stereo:sample_rate=48000"]
                ai = n_in
                n_in += 1
        else:
            cmd += ["-ss", f"{max(0.0, p['t0']):.3f}", "-t", f"{max(0.05, p['t1'] - p['t0']):.3f}", "-i", str(video)]
            vi = ai = n_in
            n_in += 1
        filt.append(f"[{vi}:v:0]{vf}[v{k}]")
        labels.append(f"[v{k}]")
        if audio:
            filt.append(f"[{ai}:a:0]aformat=sample_rates=48000:channel_layouts=stereo,aresample=async=1[a{k}]")
            labels.append(f"[a{k}]")
    filt.append("".join(labels) + f"concat=n={len(parts)}:v=1:a={1 if audio else 0}[v]" + ("[a]" if audio else ""))
    cmd += ["-filter_complex", ";".join(filt), "-map", "[v]"]
    cmd += (["-map", "[a]"] + _AENC) if audio else ["-an"]
    venc = _venc(fps, size, preset)
    cmd += venc[2:]  # same encoder settings, without -vf (done in the filter graph)
    return cmd + ["-movflags", "+faststart", "-progress", "pipe:1", "-nostats", str(out)]


class ExportCancelled(Exception):
    pass


def _run(cmd: list[str], cancel: threading.Event | None,
         on_time: Callable[[float], None] | None = None) -> None:
    """Run ffmpeg; ``on_time(seconds written)`` from ``-progress pipe:1`` output when given."""
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE if on_time else subprocess.DEVNULL, stderr=subprocess.PIPE,
                            creationflags=flags)
    err: list[bytes] = []
    th = threading.Thread(target=lambda: err.append(proc.stderr.read() if proc.stderr else b""), daemon=True)
    th.start()

    def _progress() -> None:
        for raw in proc.stdout:  # type: ignore[union-attr]
            line = raw.decode("ascii", "replace").strip()
            if line.startswith("out_time_us=") or line.startswith("out_time_ms="):
                try:
                    on_time(int(line.split("=", 1)[1]) / 1e6)  # type: ignore[misc]
                except ValueError:
                    pass

    if on_time:
        threading.Thread(target=_progress, daemon=True).start()
    while proc.poll() is None:
        if cancel is not None and cancel.is_set():
            proc.kill()
            proc.wait()
            raise ExportCancelled()
        time.sleep(0.05)
    th.join(timeout=5)
    if proc.returncode != 0:
        msg = b"".join(err).decode("utf-8", "replace").strip()
        raise RuntimeError(f"ffmpeg failed: {msg[-600:]}")


@dataclass
class ExportOptions:
    title_cards: bool = True
    card_s: float = 2.0
    intro_card: bool = True  # one card with the playlist name first
    only_unreviewed: bool = False
    audio: bool | None = None  # None = keep audio when the video has some
    quick: bool = False  # ultrafast preset: export ~20% faster here, file ~3x bigger


def export_plan(pl: Playlist, opts: ExportOptions, names: dict[str, str] | None = None,
                coached: str | None = None, replay_name: str = "") -> list[dict]:
    """The parts of the exported MP4 in order: {"kind": "card"|"clip", ...}."""
    names = names or {k: k for k in TEAMS}
    clips = [c for c in pl.clips if not (opts.only_unreviewed and c.reviewed)]
    parts: list[dict] = []
    if opts.title_cards and opts.intro_card and clips:
        sub = [f"{len(clips)} clips · {fmt_clock(sum(c.duration for c in clips), 0)}"]
        if replay_name:
            sub.insert(0, replay_name)
        parts.append({"kind": "card", "title": pl.name, "lines": sub, "seconds": opts.card_s})
    for i, c in enumerate(clips, 1):
        if opts.title_cards:
            lines = [f"Clip {i} / {len(clips)} · video {fmt_clock(c.t0)}–{fmt_clock(c.t1)}"]
            if c.team:
                lines.append(team_label(c.team, coached, names))
            if c.note:
                lines.append(c.note.replace("\n", " ")[:140])
            parts.append({"kind": "card", "title": c.title, "lines": lines, "seconds": opts.card_s})
        parts.append({"kind": "clip", "t0": c.t0, "t1": c.t1, "clip_id": c.id})
    return parts


def _part_seconds(p: dict) -> float:
    return float(p["seconds"]) if p["kind"] == "card" else max(0.05, p["t1"] - p["t0"])


def export_mp4(pl: Playlist, video: Path, dst: Path, *, fps: float, size: tuple[int, int], has_audio: bool = False,
               opts: ExportOptions | None = None, names: dict[str, str] | None = None, coached: str | None = None,
               replay_name: str = "", progress: Callable[[float], None] | None = None,
               cancel: threading.Event | None = None, ffmpeg: str | None = None) -> Path:
    """Cut every clip (and title cards) and join them: one ffmpeg run per :data:`BATCH_PARTS` parts (frame-accurate
    cuts, encoded once), batches joined losslessly."""
    opts = opts or ExportOptions()
    ffmpeg = ffmpeg or ffmpeg_exe()
    parts = export_plan(pl, opts, names, coached, replay_name)
    if not any(p["kind"] == "clip" for p in parts):
        raise ValueError("The playlist has no clips to export.")
    audio = has_audio if opts.audio is None else bool(opts.audio and has_audio)
    dst = Path(dst)
    work = Path(tempfile.mkdtemp(prefix="aa-playlist-"))
    total = sum(_part_seconds(p) for p in parts) or 1.0
    done = 0.0
    try:
        for k, p in enumerate(parts):
            if p["kind"] == "card":
                p["image"] = render_card(work / f"card_{k:04d}.png", size, p["title"], p["lines"])
        groups = [parts[i:i + BATCH_PARTS] for i in range(0, len(parts), BATCH_PARTS)]
        tmp = dst.with_name(dst.stem + ".part" + dst.suffix)
        files = []
        for g, grp in enumerate(groups):
            out = tmp if len(groups) == 1 else work / f"group_{g:03d}.mp4"
            base = done

            def on_time(x: float, base: float = base) -> None:
                if progress:
                    progress(min(0.99, (base + x) / total))

            _run(batch_cmd(ffmpeg, video, grp, out, fps=fps, size=size, audio=audio,
                           preset="ultrafast" if opts.quick else "veryfast"), cancel, on_time)
            files.append(out)
            done += sum(_part_seconds(p) for p in grp)
        if len(groups) > 1:
            lst = work / "parts.txt"
            lst.write_text(concat_list(files), encoding="utf-8")
            _run(concat_cmd(ffmpeg, lst, tmp), cancel)
        os.replace(tmp, dst)
        if progress:
            progress(1.0)
        return dst
    finally:
        shutil.rmtree(work, ignore_errors=True)
        try:
            dst.with_name(dst.stem + ".part" + dst.suffix).unlink(missing_ok=True)
        except OSError:
            pass
