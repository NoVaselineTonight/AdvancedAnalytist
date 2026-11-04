"""Timeline data model: tracks of marks, stored in ``<project>/timeline.json``. No Qt here (unit-testable).

Schema (``advanced-analytist/timeline@1``)::

    {"format": "advanced-analytist/timeline@1",
     "tracks": [{"id", "name", "color": "#rrggbb", "visible", "locked", "kind": "user"|"auto",
                 "marks": [{"id", "t", "duration", "title", "note", "color"}]}],
     "auto_overrides": {"<event id>": {"title", "note", "dismissed"}}}

The "auto" track always exists. Its marks are derived from ``replay_data.json["events"]`` (see
:meth:`Timeline.set_events`) and rebuilt whenever that file changes; they are saved too (snapshot, with
``source``/``event_id``/``confidence``/``team`` extras) so other tools can read them, but the user's edits to them
live only in ``auto_overrides`` keyed by event id, so they survive rebuilds.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

FORMAT = "advanced-analytist/timeline@1"
FILENAME = "timeline.json"
AUTO, USER = "auto", "user"
AUTO_PREFIX = "auto:"
TRACK_COLORS = ["#8b5cf6", "#0ea5e9", "#f59e0b", "#10b981", "#ef4444", "#ec4899", "#14b8a6", "#6366f1",
                "#84cc16", "#f97316"]
DEFAULT_TRACKS = [("Auto detection", "#8b5cf6", AUTO), ("Notes", "#0ea5e9", USER), ("Highlights", "#f59e0b", USER)]
EVENT_TITLES = {"basket": "Basket"}
INPLAY_TRACK_ID, INPLAY_TRACK_NAME, INPLAY_COLOR = "inplay", "In play", "#22c55e"  # legacy, replaced below
DEAD_TRACK_ID, DEAD_TRACK_NAME, DEAD_COLOR = "notinplay", "Ball not in play", "#94a3b8"


def new_id() -> str:
    return uuid.uuid4().hex[:12]


def _color(v: Any, default: str | None) -> str | None:
    if isinstance(v, str) and len(v) == 7 and v.startswith("#"):
        try:
            int(v[1:], 16)
            return v.lower()
        except ValueError:
            pass
    return default


def _num(v: Any, default: float = 0.0) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return f if f == f and abs(f) != float("inf") else default


@dataclass
class Mark:
    t: float
    title: str = ""
    note: str = ""
    duration: float = 0.0
    color: str | None = None
    id: str = field(default_factory=new_id)
    # auto marks only
    event_id: str | None = None
    confidence: float | None = None
    team: str | None = None

    @property
    def is_auto(self) -> bool:
        return self.event_id is not None

    @property
    def end(self) -> float:
        return self.t + max(0.0, self.duration)

    def to_dict(self) -> dict:
        d = {"id": self.id, "t": round(self.t, 3), "duration": round(max(0.0, self.duration), 3),
             "title": self.title, "note": self.note, "color": self.color}
        if self.event_id is not None:
            d.update(source="auto", event_id=self.event_id, confidence=self.confidence, team=self.team)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Mark":
        return cls(t=max(0.0, _num(d.get("t"))), title=str(d.get("title") or ""), note=str(d.get("note") or ""),
                   duration=max(0.0, _num(d.get("duration"))), color=_color(d.get("color"), None),
                   id=str(d.get("id") or new_id()))


@dataclass
class Track:
    name: str
    color: str
    kind: str = USER
    visible: bool = True
    locked: bool = False
    marks: list[Mark] = field(default_factory=list)
    id: str = field(default_factory=new_id)

    @property
    def is_auto(self) -> bool:
        return self.kind == AUTO

    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "color": self.color, "visible": self.visible,
                "locked": self.locked, "kind": self.kind,
                "marks": [m.to_dict() for m in sorted(self.marks, key=lambda m: m.t)]}

    @classmethod
    def from_dict(cls, d: dict, i: int = 0) -> "Track":
        kind = AUTO if d.get("kind") == AUTO else USER
        marks = []
        if kind == USER:
            for m in d.get("marks") or []:
                if isinstance(m, dict):
                    marks.append(Mark.from_dict(m))
        return cls(name=str(d.get("name") or "Track"), color=_color(d.get("color"), TRACK_COLORS[i % 10]),
                   kind=kind, visible=bool(d.get("visible", True)), locked=bool(d.get("locked", False)),
                   marks=marks, id=str(d.get("id") or new_id()))


def event_title(ev: dict, team_names: dict[str, str]) -> str:
    if ev.get("label"):  # game-state events (replay/game_state.py): "<label> – <team> <suffix>"
        from replay.game_state import event_title as gs_title
        return gs_title(ev, team_names)
    kind = str(ev.get("type") or "event")
    title = EVENT_TITLES.get(kind, kind.replace("_", " ").capitalize())
    team = ev.get("team")
    if team:
        title += f" – {team_names.get(team, team)}"
    if ev.get("basket"):
        title += f" ({ev['basket']})"
    return title


def event_note(ev: dict) -> str:
    c = ev.get("confidence")
    return f"Detected automatically (confidence {_num(c) * 100:.0f}%)." if c is not None \
        else "Detected automatically."


class Timeline:
    def __init__(self, tracks: list[Track] | None = None, auto_overrides: dict | None = None):
        self.tracks: list[Track] = tracks if tracks is not None else []
        self.auto_overrides: dict[str, dict] = auto_overrides if auto_overrides is not None else {}
        self.events: list[dict] = []
        self.teams: dict[str, dict] = {}
        self._ensure_auto()

    # ------------------------------------------------------------------ construction / io
    @classmethod
    def default(cls) -> "Timeline":
        return cls([Track(name, color, kind) for name, color, kind in DEFAULT_TRACKS])

    @classmethod
    def from_dict(cls, d: Any) -> "Timeline":
        if not isinstance(d, dict) or not isinstance(d.get("tracks"), list):
            raise ValueError("not a timeline")
        tracks, seen = [], set()
        for i, td in enumerate(d["tracks"]):
            if not isinstance(td, dict):
                continue
            tr = Track.from_dict(td, i)
            if tr.id in seen:
                tr.id = new_id()
            seen.add(tr.id)
            if tr.is_auto and any(t.is_auto for t in tracks):
                continue  # only one auto track
            tracks.append(tr)
        ov = {}
        for k, v in (d.get("auto_overrides") or {}).items() if isinstance(d.get("auto_overrides"), dict) else []:
            if isinstance(v, dict):
                ov[str(k)] = {kk: v[kk] for kk in ("title", "note", "dismissed") if kk in v}
        return cls(tracks, ov)

    def to_dict(self) -> dict:
        return {"format": FORMAT, "tracks": [t.to_dict() for t in self.tracks],
                "auto_overrides": self.auto_overrides}

    @classmethod
    def load(cls, path: Path) -> "Timeline":
        """Load ``path``; a missing file gives the defaults. A corrupt file is kept aside as ``*.bad`` (so the next
        save cannot destroy anything recoverable) and the defaults are returned."""
        path = Path(path)
        if not path.exists():
            return cls.default()
        try:
            return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError, AttributeError):
            try:
                os.replace(path, path.with_suffix(path.suffix + ".bad"))
            except OSError:
                pass
            return cls.default()

    def save(self, path: Path) -> None:
        path = Path(path)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)

    def _ensure_auto(self) -> None:
        if not any(t.is_auto for t in self.tracks):
            name, color, kind = DEFAULT_TRACKS[0]
            self.tracks.insert(0, Track(name, color, kind))

    # ------------------------------------------------------------------ lookup
    @property
    def auto_track(self) -> Track:
        return next(t for t in self.tracks if t.is_auto)

    def track(self, track_id: str) -> Track | None:
        return next((t for t in self.tracks if t.id == track_id), None)

    def find_mark(self, mark_id: str) -> tuple[Track, Mark] | tuple[None, None]:
        for t in self.tracks:
            for m in t.marks:
                if m.id == mark_id:
                    return t, m
        return None, None

    def all_marks(self, visible_only: bool = True) -> list[tuple[Track, Mark]]:
        out = [(t, m) for t in self.tracks if t.visible or not visible_only for m in t.marks]
        return sorted(out, key=lambda tm: tm[1].t)

    # ------------------------------------------------------------------ tracks
    def next_color(self) -> str:
        used = {t.color for t in self.tracks}
        return next((c for c in TRACK_COLORS if c not in used), TRACK_COLORS[len(self.tracks) % len(TRACK_COLORS)])

    def add_track(self, name: str, color: str | None = None) -> Track:
        tr = Track(name or "Track", _color(color, None) or self.next_color())
        self.tracks.append(tr)
        return tr

    def delete_track(self, track_id: str) -> bool:
        tr = self.track(track_id)
        if tr is None or tr.is_auto:
            return False
        self.tracks.remove(tr)
        return True

    def move_track(self, track_id: str, delta: int) -> bool:
        tr = self.track(track_id)
        if tr is None:
            return False
        i = self.tracks.index(tr)
        j = max(0, min(len(self.tracks) - 1, i + delta))
        if i == j:
            return False
        self.tracks.insert(j, self.tracks.pop(i))
        return True

    def update_track(self, track_id: str, **fields: Any) -> bool:
        tr = self.track(track_id)
        if tr is None:
            return False
        for k, v in fields.items():
            if k == "color":
                v = _color(v, tr.color)
            if k in ("name", "color", "visible", "locked"):
                setattr(tr, k, v)
        return True

    # ------------------------------------------------------------------ marks
    def add_mark(self, track_id: str, t: float, title: str = "", note: str = "", duration: float = 0.0,
                 color: str | None = None) -> Mark | None:
        tr = self.track(track_id)
        if tr is None or tr.is_auto:
            return None
        m = Mark(t=max(0.0, float(t)), title=title, note=note, duration=max(0.0, float(duration)),
                 color=_color(color, None))
        tr.marks.append(m)
        tr.marks.sort(key=lambda x: x.t)
        return m

    def move_mark(self, mark_id: str, t: float, track_id: str | None = None) -> bool:
        tr, m = self.find_mark(mark_id)
        if m is None or m.is_auto:
            return False
        m.t = max(0.0, float(t))
        if track_id and track_id != tr.id:
            dst = self.track(track_id)
            if dst is not None and not dst.is_auto:
                tr.marks.remove(m)
                dst.marks.append(m)
                tr = dst
        tr.marks.sort(key=lambda x: x.t)
        return True

    def update_mark(self, mark_id: str, **fields: Any) -> bool:
        """Change title/note/color/duration/t (and ``track_id`` to move). For auto marks only title and note can be
        changed; they are stored as overrides keyed by the event id."""
        tr, m = self.find_mark(mark_id)
        if m is None:
            return False
        if m.is_auto:
            ov = self.auto_overrides.setdefault(m.event_id, {})
            for k in ("title", "note"):
                if k in fields and fields[k] is not None:
                    ov[k] = str(fields[k])
                    setattr(m, k, str(fields[k]))
            return True
        for k in ("title", "note"):
            if k in fields and fields[k] is not None:
                setattr(m, k, str(fields[k]))
        if "color" in fields:
            m.color = _color(fields["color"], None)
        if "duration" in fields:
            m.duration = max(0.0, _num(fields["duration"]))
        if "t" in fields or fields.get("track_id"):
            self.move_mark(mark_id, _num(fields.get("t", m.t), m.t), fields.get("track_id"))
        return True

    def delete_mark(self, mark_id: str) -> bool:
        """Delete a user mark, or dismiss an auto mark (remembered in the overrides)."""
        tr, m = self.find_mark(mark_id)
        if m is None:
            return False
        if m.is_auto:
            self.auto_overrides.setdefault(m.event_id, {})["dismissed"] = True
        tr.marks.remove(m)
        return True

    def reset_auto(self, mark_id: str) -> bool:
        """Forget the user's title/note edits of an auto mark (back to the detected text)."""
        _, m = self.find_mark(mark_id)
        if m is None or not m.is_auto or m.event_id not in self.auto_overrides:
            return False
        ov = self.auto_overrides[m.event_id]
        ov.pop("title", None)
        ov.pop("note", None)
        if not ov:
            del self.auto_overrides[m.event_id]
        self._rebuild_auto()
        return True

    def restore_dismissed(self) -> int:
        n = 0
        for ov in self.auto_overrides.values():
            if ov.get("dismissed"):
                ov["dismissed"] = False
                n += 1
        if n:
            self._rebuild_auto()
        return n

    def dismissed_count(self) -> int:
        return sum(1 for ov in self.auto_overrides.values() if ov.get("dismissed"))

    # ------------------------------------------------------------------ auto detection
    def set_events(self, events: list[dict] | None, teams: dict[str, dict] | None = None) -> bool:
        """Rebuild the auto track from replay events. ``teams`` = {"A": {"name", "color"}, ...}.
        Returns True when the auto marks changed."""
        self.events = [e for e in (events or []) if isinstance(e, dict) and e.get("id") is not None]
        self.teams = teams or {}
        return self._rebuild_auto()

    def _rebuild_auto(self) -> bool:
        tr = self.auto_track
        before = [m.to_dict() for m in tr.marks]
        names = {k: str(v.get("name") or k) for k, v in self.teams.items()}
        marks = []
        for ev in self.events:
            eid = str(ev["id"])
            ov = self.auto_overrides.get(eid, {})
            if ov.get("dismissed"):
                continue
            team = ev.get("team")
            conf = ev.get("confidence")
            marks.append(Mark(t=max(0.0, _num(ev.get("t"))), duration=max(0.0, _num(ev.get("duration"))),
                              title=str(ov["title"]) if "title" in ov else event_title(ev, names),
                              note=str(ov["note"]) if "note" in ov else event_note(ev),
                              color=_color((self.teams.get(team) or {}).get("color"), None) if team else None,
                              id=AUTO_PREFIX + eid, event_id=eid,
                              confidence=None if conf is None else _num(conf), team=team))
        marks.sort(key=lambda m: m.t)
        tr.marks = marks
        return [m.to_dict() for m in marks] != before


    # ------------------------------------------------------------------ in play (preliminary analysis)
    def set_inplay(self, intervals: list[tuple[float, float]] | None,
                   span: tuple[float, float] | None = None) -> bool:
        """Show the times the ball is NOT in play (breaks, timeouts, halftime - skipped by the preliminary
        analysis, so nothing is tracked there) as range marks on a locked "Ball not in play" track.
        ``intervals`` are the in-play ranges, ``span`` the analysed range (start, end) of the video.
        Removed when ``intervals`` is None. Returns True when something changed."""
        changed = False
        old = self.track(INPLAY_TRACK_ID)  # the earlier "In play" track is replaced by this one
        if old is not None:
            self.tracks.remove(old)
            changed = True
        tr = self.track(DEAD_TRACK_ID)
        if intervals is None:
            if tr is not None:
                self.tracks.remove(tr)
                changed = True
            return changed
        iv = sorted((float(a), float(b)) for a, b in intervals)
        lo = float(span[0]) if span else (iv[0][0] if iv else 0.0)
        hi = float(span[1]) if span else (iv[-1][1] if iv else 0.0)
        gaps, cur = [], lo
        for a, b in iv:
            if a - cur > 0.5:
                gaps.append((cur, a))
            cur = max(cur, b)
        if hi - cur > 0.5:
            gaps.append((cur, hi))
        if tr is None:
            tr = Track(DEAD_TRACK_NAME, DEAD_COLOR, USER, locked=True, id=DEAD_TRACK_ID)
            self.tracks.insert(1 if self.tracks and self.tracks[0].is_auto else 0, tr)
            changed = True
        before = [(round(m.t, 2), round(m.duration, 2)) for m in tr.marks]
        new = [(round(a, 2), round(b - a, 2)) for a, b in gaps]
        if before == new:
            return changed
        tr.marks = [Mark(t=a, duration=d, title=DEAD_TRACK_NAME,
                         note="Break / timeout / halftime found by the preliminary analysis: skipped, so no "
                              "players or ball are tracked here.", id=f"notinplay:{i}")
                    for i, (a, d) in enumerate(new)]
        return True


def load(project_root: Path) -> Timeline:
    return Timeline.load(Path(project_root) / FILENAME)


def save(tl: Timeline, project_root: Path) -> None:
    tl.save(Path(project_root) / FILENAME)
