"""Team rosters: player names behind jersey numbers. No Qt here (unit-testable).

Two files:

* the roster LIBRARY, per Windows user, reused across Replays: ``%APPDATA%/AdvancedAnalytist/rosters.json``
  (next to the app's other per-user settings; override with env ``ADVANCED_ANALYTIST_ROSTERS`` = file path)::

      {"format": "advanced-analytist/rosters@1",
       "rosters": [{"id", "team": "KlaNMKY", "kit": "purple", "players": [{"number": "23", "name", "position"}]}]}

* the per-Replay ASSIGNMENT, ``projects/<p>/roster.json``::

      {"format": "advanced-analytist/replay-roster@1",
       "home": {"roster_id", "roster": {copy of the library roster}},   # team "A" downstream
       "away": {...},                                                    # team "B"
       "overrides": {"track:12": {"team": "A", "number": "23", "name": "Virtanen"},
                     "slot:A3": {...}, "track:40": {"none": true}}}

  The Replay keeps its own copy of both rosters, so it also works on another computer (``.aareplay`` packages).

Mapping (``RosterMap``): a manual override (track id, then lineup slot) always wins; otherwise an accepted jersey
number (replay_data ``track_info[id]["number"]``, game-state ``lineup.numbers`` / ``lineup_at`` "number") of a
team is looked up in that team's roster. Without ``roster.json`` every hook here returns its input unchanged, so
the program behaves exactly as before.

CSV import / export: ``number,name,position`` (header optional, ``,`` or ``;`` separated, UTF-8 with or without BOM).
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

LIBRARY_FORMAT = "advanced-analytist/rosters@1"
REPLAY_FORMAT = "advanced-analytist/replay-roster@1"
FILENAME = "roster.json"
LIBRARY_ENV = "ADVANCED_ANALYTIST_ROSTERS"
SIDES = ("home", "away")
TEAM_OF_SIDE = {"home": "A", "away": "B"}
SIDE_OF_TEAM = {"A": "home", "B": "away"}
CSV_COLUMNS = ["number", "name", "position"]


# ====================================================================================== data model
def norm_number(v: Any) -> str:
    """Jersey number text: '#7' / ' 7 ' / '07' -> '7'; '00' and '0' stay distinct; non-digits kept as typed."""
    s = str(v if v is not None else "").strip().lstrip("#").strip()
    if isinstance(v, float) and v == int(v):
        s = str(int(v))
    if s.isdigit() and len(s) > 1 and s.startswith("0") and s.strip("0"):
        s = s.lstrip("0")
    return s


@dataclass
class Player:
    number: str
    name: str
    position: str = ""

    def to_dict(self) -> dict:
        return {"number": self.number, "name": self.name, "position": self.position}

    @classmethod
    def from_dict(cls, d: Any) -> "Player | None":
        if not isinstance(d, dict):
            return None
        p = cls(norm_number(d.get("number")), str(d.get("name") or "").strip(), str(d.get("position") or "").strip())
        return p if (p.number or p.name) else None


@dataclass
class Roster:
    team: str = ""
    kit: str = ""  # kit colour note, e.g. "purple shirts, white numbers"
    players: list[Player] = field(default_factory=list)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def to_dict(self) -> dict:
        return {"id": self.id, "team": self.team, "kit": self.kit, "players": [p.to_dict() for p in self.players]}

    @classmethod
    def from_dict(cls, d: Any) -> "Roster | None":
        if not isinstance(d, dict):
            return None
        players = [p for p in (Player.from_dict(x) for x in d.get("players") or []) if p is not None]
        return cls(team=str(d.get("team") or "").strip(), kit=str(d.get("kit") or "").strip(), players=players,
                   id=str(d.get("id") or uuid.uuid4().hex[:12]))

    def by_number(self, number: Any) -> Player | None:
        n = norm_number(number)
        if not n:
            return None
        return next((p for p in self.players if p.number == n), None)

    def label(self) -> str:
        return self.team or "(unnamed roster)"


# ====================================================================================== CSV
def roster_to_csv(r: Roster) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(CSV_COLUMNS)
    for p in r.players:
        w.writerow([p.number, p.name, p.position])
    return buf.getvalue()


def players_from_csv(text: str) -> list[Player]:
    """Parse ``number,name,position`` rows (header optional; ',' ';' or tab separated). Rows without a name and
    number are skipped. Raises ValueError when nothing usable is found."""
    text = (text or "").lstrip("﻿")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise ValueError("the file is empty")
    first = lines[0]
    delim = max((";", ",", "\t"), key=first.count) if any(c in first for c in ";,\t") else ","
    rows = list(csv.reader(lines, delimiter=delim))
    idx = {"number": 0, "name": 1, "position": 2}
    aliases = {"number": ("number", "no", "nr", "num", "jersey", "numero", "#"), "name": ("name", "player", "nimi"),
               "position": ("position", "pos", "pelipaikka")}
    head = [c.strip().lower() for c in rows[0]]
    if not norm_number(rows[0][0] if rows[0] else "").isdigit() and any(h in al for al in aliases.values()
                                                                        for h in head):
        for k, al in aliases.items():
            i = next((i for i, h in enumerate(head) if h in al), None)
            if i is not None:
                idx[k] = i
        rows = rows[1:]
    out: list[Player] = []
    for row in rows:
        get = lambda k: row[idx[k]].strip() if idx[k] < len(row) else ""  # noqa: E731
        p = Player(norm_number(get("number")), get("name"), get("position"))
        if p.number or p.name:
            out.append(p)
    if not out:
        raise ValueError("no players found (expected columns number,name,position)")
    return out


# ====================================================================================== library (per user)
def library_path() -> Path:
    env = os.environ.get(LIBRARY_ENV)
    if env:
        return Path(env)
    base = os.environ.get("APPDATA")
    root = Path(base) / "AdvancedAnalytist" if base else Path.home() / ".advancedanalytist"
    return root / "rosters.json"


def _read_json(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _write_json(path: Path, doc: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def load_library(path: Path | None = None) -> list[Roster]:
    doc = _read_json(path or library_path())
    items = doc.get("rosters") if isinstance(doc, dict) else None
    return [r for r in (Roster.from_dict(x) for x in items or []) if r is not None]


def save_library(rosters: Iterable[Roster], path: Path | None = None) -> Path:
    p = path or library_path()
    _write_json(p, {"format": LIBRARY_FORMAT, "rosters": [r.to_dict() for r in rosters]})
    return p


def upsert_library(roster: Roster, path: Path | None = None) -> None:
    lib = [r for r in load_library(path) if r.id != roster.id]
    lib.append(roster)
    lib.sort(key=lambda r: r.team.lower())
    save_library(lib, path)


# ====================================================================================== per-Replay assignment
@dataclass
class Assignment:
    home: Roster | None = None
    away: Roster | None = None
    overrides: dict[str, dict] = field(default_factory=dict)  # "track:<id>" / "slot:<A3>" -> player ref

    def roster(self, team: str | None) -> Roster | None:
        return self.home if team == "A" else self.away if team == "B" else None

    def set_roster(self, team: str, r: Roster | None) -> None:
        if team == "A":
            self.home = r
        elif team == "B":
            self.away = r

    def to_dict(self) -> dict:
        def side(r: Roster | None) -> dict | None:
            return None if r is None else {"roster_id": r.id, "roster": r.to_dict()}
        return {"format": REPLAY_FORMAT, "home": side(self.home), "away": side(self.away),
                "overrides": self.overrides}

    @classmethod
    def from_dict(cls, d: Any) -> "Assignment":
        if not isinstance(d, dict):
            raise ValueError("not a roster assignment")

        def side(v: Any) -> Roster | None:
            return Roster.from_dict(v.get("roster")) if isinstance(v, dict) else None
        ov = {}
        for k, v in (d.get("overrides") or {}).items() if isinstance(d.get("overrides"), dict) else []:
            if isinstance(v, dict) and str(k).partition(":")[0] in ("track", "slot"):
                ov[str(k)] = {kk: v[kk] for kk in ("team", "number", "name", "none") if kk in v}
        return cls(side(d.get("home")), side(d.get("away")), ov)


def assignment_path(project_root: Path) -> Path:
    return Path(project_root) / FILENAME


def load_assignment(project_root: Path | None) -> Assignment | None:
    """The Replay's roster.json, or None when it does not exist / cannot be read (= no roster, old behaviour)."""
    if project_root is None:
        return None
    f = assignment_path(project_root)
    if not f.exists():
        return None
    try:
        return Assignment.from_dict(_read_json(f))
    except ValueError:
        return None


def save_assignment(project_root: Path, a: Assignment) -> Path:
    f = assignment_path(project_root)
    _write_json(f, a.to_dict())
    return f


# ====================================================================================== mapping
def override_key(kind: str, ident: Any) -> str:
    return f"{kind}:{ident}"


class RosterMap:
    """Resolves track ids / lineup slots / (team, number) to roster players."""

    def __init__(self, a: Assignment, track_info: dict | None = None, track_team: dict | None = None,
                 slot_of_track: dict | None = None, slot_numbers: dict | None = None):
        self.a = a
        self.track_info = {str(k): v for k, v in (track_info or {}).items() if isinstance(v, dict)}
        self.track_team = {str(k): str(v) for k, v in (track_team or {}).items() if v}
        for k, v in self.track_info.items():
            if v.get("team") in ("A", "B"):
                self.track_team.setdefault(k, v["team"])
        self.slot_of_track = {str(k): str(v) for k, v in (slot_of_track or {}).items() if v}
        self.slot_numbers = {str(k): norm_number(v) for k, v in (slot_numbers or {}).items() if v not in (None, "")}

    # ---- primitives
    def player_by_number(self, team: str | None, number: Any) -> Player | None:
        r = self.a.roster(team)
        return r.by_number(number) if r is not None else None

    def _from_override(self, ov: dict) -> Player | None | bool:
        """Player for an override; False = explicitly 'nobody' (suppresses the automatic name)."""
        if ov.get("none"):
            return False
        team, num, name = ov.get("team"), norm_number(ov.get("number")), str(ov.get("name") or "").strip()
        p = self.player_by_number(team, num) if num else None
        if p is not None:
            return p
        if name:
            return Player(num, name)
        return None

    def number_of_track(self, tid: Any) -> str:
        tid = str(tid)
        n = norm_number((self.track_info.get(tid) or {}).get("number"))
        if not n:
            slot = self.slot_of_track.get(tid)
            n = self.slot_numbers.get(slot, "") if slot else ""
        return n

    def team_of_track(self, tid: Any) -> str | None:
        tid = str(tid)
        t = self.track_team.get(tid)
        if t in ("A", "B"):
            return t
        slot = self.slot_of_track.get(tid) or ""
        return slot[:1] if slot[:1] in ("A", "B") else None

    # ---- public
    def player_for_track(self, tid: Any, number: Any = None, team: str | None = None) -> Player | None:
        tid = str(tid)
        for key in (override_key("track", tid),
                    override_key("slot", self.slot_of_track[tid]) if tid in self.slot_of_track else None):
            if key and key in self.a.overrides:
                p = self._from_override(self.a.overrides[key])
                if p is False:
                    return None
                if p is not None:
                    return p
        num = norm_number(number) if number not in (None, "") else self.number_of_track(tid)
        team = team or self.team_of_track(tid)
        return self.player_by_number(team, num) if num else None

    def player_for_slot(self, slot: Any, number: Any = None) -> Player | None:
        slot = str(slot)
        key = override_key("slot", slot)
        if key in self.a.overrides:
            p = self._from_override(self.a.overrides[key])
            if p is False:
                return None
            if p is not None:
                return p
        num = norm_number(number) if number not in (None, "") else self.slot_numbers.get(slot, "")
        return self.player_by_number(slot[:1], num) if num else None

    def name_for_track(self, tid: Any, number: Any = None, team: str | None = None) -> str:
        p = self.player_for_track(tid, number, team)
        return p.name if p is not None else ""

    def number_names(self) -> dict[str, dict[str, str]]:
        out: dict[str, dict[str, str]] = {}
        for team in ("A", "B"):
            r = self.a.roster(team)
            out[team] = {p.number: p.name for p in (r.players if r else []) if p.number and p.name}
        return out


def _lineup(gs: dict | None) -> tuple[dict, dict]:
    lu = (gs or {}).get("lineup") or {}
    return (lu.get("slot_of_track") or {}) if isinstance(lu, dict) else {}, \
        (lu.get("numbers") or {}) if isinstance(lu, dict) else {}


def map_for(project_root: Path | None, replay: dict | None = None, gs: dict | None = None,
            track_info: dict | None = None, track_team: dict | None = None) -> RosterMap | None:
    """RosterMap of a Replay, or None when it has no roster.json. replay / gs are read from results/ when omitted
    and needed (gs only when given or results/game_state.json exists)."""
    a = load_assignment(project_root)
    if a is None:
        return None
    root = Path(project_root)
    if track_info is None:
        if replay is None:
            replay = _read_json(root / "results" / "replay_data.json") or {}
        track_info = replay.get("track_info") or {}
        if track_team is None:
            track_team = {str(t.get("id")): t.get("team") for t in replay.get("tracks") or [] if isinstance(t, dict)}
    if gs is None:
        gs = _read_json(root / "results" / "game_state.json")
    slot_of, nums = _lineup(gs if isinstance(gs, dict) else None)
    return RosterMap(a, track_info, track_team, slot_of, nums)


# ====================================================================================== text decoration
def _decorate_tokens(text: str, pairs: list[tuple[str, str]]) -> str:
    """Insert ' Name' after the first undecorated occurrence of each token ('#7', 'T12')."""
    for token, name in pairs:
        if not token or not name:
            continue
        pat = re.compile(re.escape(token) + r"(?![0-9A-Za-z])(?! " + re.escape(name) + ")")
        text = pat.sub(f"{token} {name}", text, count=1)
    return text


PLAYER_KEYS = ("shooter_track", "rebounder_track", "passer_track", "receiver_track", "stealer_track",
               "lost_by_track", "player_track", "fouler_track", "holder_track")


def event_players(ev: dict) -> list[str]:
    d = ev.get("details") or {}
    ann = d.get("game_state") or {} if isinstance(d, dict) else {}
    out: list[str] = []
    for k in PLAYER_KEYS:
        for v in (d.get(k) if isinstance(d, dict) else None, ann.get(k) if isinstance(ann, dict) else None):
            if v not in (None, "") and str(v) not in out:
                out.append(str(v))
    return out


def decorate_text(text: str, rm: RosterMap, tids: Iterable[str] = (), team: str | None = None) -> str:
    """'Pass #7 -> #12' -> 'Pass #7 Virtanen -> #12 Korhonen'. Track ids named by the event come first (their
    team decides); remaining '#N' tokens use ``team``'s roster."""
    if not text:
        return text
    pairs = []
    for tid in tids:
        p = rm.player_for_track(tid)
        if p is None or not p.name:
            continue
        num = rm.number_of_track(tid)
        pairs.append((f"#{num}" if num else f"T{tid}", p.name))
    text = _decorate_tokens(text, pairs)
    if team in ("A", "B"):
        rest = []
        end = text.find(" (")  # "(from #12)" etc. name the other team: only the main part uses ``team``
        for m in re.finditer(r"#(\d{1,2})(?![0-9A-Za-z])", text):
            if 0 <= end < m.start():
                break
            p = rm.player_by_number(team, m.group(1))
            if p is not None and p.name and not text[m.end():].startswith(" " + p.name):
                rest.append((m.group(0), p.name))
        text = _decorate_tokens(text, rest)
    return text


def decorate_events(project_root: Path | None, data: dict | None, events: list[dict]) -> list[dict]:
    """Events with player names in "label" / "suffix" / "title" (copies); the input list itself when the Replay
    has no roster.json. Used by the timeline's auto track (roster hook in gui/timeline.py)."""
    try:
        rm = map_for(project_root, replay=data) if project_root is not None else None
    except Exception as e:  # noqa: BLE001 - names are decoration, never break the timeline
        print("roster: names unavailable:", type(e).__name__, e)
        return events
    if rm is None:
        return events
    out = []
    for ev in events:
        if not isinstance(ev, dict):
            out.append(ev)
            continue
        tids, team = event_players(ev), ev.get("team")
        e2 = dict(ev)
        for k in ("label", "suffix", "title"):
            if isinstance(ev.get(k), str) and ev.get(k):
                e2[k] = decorate_text(ev[k], rm, tids, team if k != "suffix" else None)
        out.append(e2)
    return out


def annotate_replay_data(project_root: Path | None, data: dict | None) -> dict | None:
    """Add data["roster_names"] = {"tracks": {id: name}, "numbers": {"A": {num: name}, "B": ...}} for the 2D view
    tooltip; removes it when the Replay has no roster.json. Returns data."""
    if not isinstance(data, dict):
        return data
    data.pop("roster_names", None)
    try:
        rm = map_for(project_root, replay=data)
    except Exception as e:  # noqa: BLE001
        print("roster: names unavailable:", type(e).__name__, e)
        return data
    if rm is None:
        return data
    tracks = {}
    ids = {str(t.get("id")) for t in data.get("tracks") or [] if isinstance(t, dict)} | set(rm.track_info)
    for tid in ids:
        n = rm.name_for_track(tid)
        if n:
            tracks[tid] = n
    data["roster_names"] = {"tracks": tracks, "numbers": rm.number_names()}
    return data


def dot_name(data: dict | None, tid: Any, team: str | None, number: str | None) -> str:
    """Name for a 2D dot from data["roster_names"]: a track's (manual or numbered) name, else its number's."""
    rn = (data or {}).get("roster_names") or {}
    n = (rn.get("tracks") or {}).get(str(tid))
    if n:
        return n
    if number and team in ("A", "B"):
        return ((rn.get("numbers") or {}).get(team) or {}).get(norm_number(number), "")
    return ""


# ====================================================================================== exports / AI / playlists
# PRIVACY: players are often minors. Names are written to data exports only when this is True (or when the caller
# passes include_names=True). Licence numbers are never part of a roster and never exported.
EXPORT_PLAYER_NAMES = False


def augment_export(project_root: Path, out: Path, include_names: bool | None = None) -> bool:
    """Data export (data_export.export_simple) roster hook: a ``player_name`` column right after ``jersey_number``
    in players.csv and track_summary.csv, player names in events.csv titles and a roster.csv. No roster.json ->
    nothing is touched (returns False). Also does nothing unless names are enabled (``include_names`` or the
    module default ``EXPORT_PLAYER_NAMES``, False)."""
    if not (EXPORT_PLAYER_NAMES if include_names is None else include_names):
        return False
    try:
        rm = map_for(project_root)
    except Exception as e:  # noqa: BLE001
        print("roster: export names skipped:", type(e).__name__, e)
        return False
    if rm is None:
        return False
    out = Path(out)
    for fname in ("players.csv", "track_summary.csv"):
        f = out / fname
        if not f.exists():
            continue
        rows = list(csv.reader(f.read_text(encoding="utf-8").splitlines()))
        if not rows or "jersey_number" not in rows[0] or "player_name" in rows[0]:
            continue
        head = rows[0]
        ci, ti, ni = head.index("jersey_number"), head.index("track_id"), head.index("jersey_number")
        tm = head.index("team") if "team" in head else None
        cache: dict[tuple, str] = {}
        new = [head[:ci + 1] + ["player_name"] + head[ci + 1:]]
        for r in rows[1:]:
            key = (r[ti] if ti < len(r) else "", r[ni] if ni < len(r) else "",
                   r[tm] if tm is not None and tm < len(r) else "")
            if key not in cache:
                cache[key] = rm.name_for_track(key[0], key[1] or None, key[2] if key[2] in ("A", "B") else None)
            new.append(r[:ci + 1] + [cache[key]] + r[ci + 1:])
        with open(f, "w", encoding="utf-8", newline="") as fh:
            csv.writer(fh, lineterminator="\n").writerows(new)
    f = out / "events.csv"
    if f.exists():
        rows = list(csv.reader(io.StringIO(f.read_text(encoding="utf-8"))))
        if rows and "title" in rows[0]:
            ti, tm = rows[0].index("title"), rows[0].index("team") if "team" in rows[0] else None
            for r in rows[1:]:
                if ti < len(r):
                    r[ti] = decorate_text(r[ti], rm, (), r[tm] if tm is not None and tm < len(r) else None)
            with open(f, "w", encoding="utf-8", newline="") as fh:
                csv.writer(fh, lineterminator="\n").writerows(rows)
    lines = [["team", "team_side", "roster", "number", "name", "position"]]
    for team in ("A", "B"):
        r = rm.a.roster(team)
        for p in (r.players if r else []):
            lines.append([team, SIDE_OF_TEAM[team], r.team, p.number, p.name, p.position])
    with open(out / "roster.csv", "w", encoding="utf-8", newline="") as fh:
        csv.writer(fh, lineterminator="\n").writerows(lines)
    return True


def ai_names(project_root: Path | None, track_info: dict, team_of_track: dict, gs: dict | None = None
             ) -> dict[str, tuple[str, str]]:
    """{track id: (jersey number or "", name)} for the AI prompt labels; {} when the Replay has no roster.json.
    The number is the track's own or its lineup slot's (``lineup.numbers``)."""
    try:
        rm = map_for(project_root, gs=gs, track_info=track_info, track_team=team_of_track)
    except Exception:  # noqa: BLE001
        return {}
    if rm is None:
        return {}
    ids = set(map(str, track_info)) | set(map(str, team_of_track)) | set(rm.slot_of_track)
    out = {}
    for tid in ids:
        p = rm.player_for_track(tid)
        if p is not None and p.name:
            out[tid] = (rm.number_of_track(tid) or p.number, p.name)
    return out


def ai_suffix(entry: tuple[str, str], label_has_number: bool) -> str:
    """Text appended to a prompt label: '#23 Virtanen' (number only when the label has none yet)."""
    num, name = entry
    return ("" if label_has_number or not num else f"#{num}") + f" {ai_label_name(name)}"


def ai_label_name(name: str) -> str:
    """Name as one token inside a frame line ('Matti Virtanen' -> 'Matti_Virtanen')."""
    return re.sub(r"[\s|:,]+", "_", name.strip())


def decorate_catalog(project_root: Path | None, cat: Any) -> Any:
    """Playlists (replay/playlists.py Catalog) roster hook: names in item titles and in the player choices."""
    try:
        a = load_assignment(project_root)
    except Exception:  # noqa: BLE001
        a = None
    if a is None:
        return cat
    slot_of = dict(getattr(cat, "track_slot", {}) or {})
    rm = RosterMap(a, {k: {"number": v} for k, v in (getattr(cat, "track_number", {}) or {}).items()},
                   getattr(cat, "track_team", {}) or {}, slot_of, None)
    for it in getattr(cat, "items", []) or []:
        try:
            team = next((v for v in (it.cats or {}).values() if v in ("A", "B")), None)
            it.title = decorate_text(it.title, rm, list(it.players or []), team)
        except Exception:  # noqa: BLE001
            continue
    cat.roster_map = rm
    return cat


def decorate_choices(cat: Any, choices: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Playlists player_choices labels: '#23 – Team' -> '#23 Virtanen – Team' (num keys and slot keys)."""
    rm = getattr(cat, "roster_map", None)
    if rm is None:
        return choices
    out = []
    for key, label in choices:
        kind, _, rest = key.partition(":")
        p = None
        if kind == "num":
            team, _, num = rest.rpartition(":")
            p = rm.player_by_number(team or None, num)
            if p is not None and p.name:
                label = label.replace(f"#{num}", f"#{num} {p.name}", 1)
        elif kind == "slot":
            p = rm.player_for_slot(rest)
            if p is not None and p.name:
                label = label.replace(f"Slot {rest}", f"Slot {rest} {p.name}", 1)
        out.append((key, label))
    return out


# ====================================================================================== players seen
def players_seen(project_root: Path, replay: dict | None = None, gs: dict | None = None) -> list[dict]:
    """Team players of this Replay for manual assignment: lineup slots (if any) then numbered / long tracks.
    [{"key": "track:12" | "slot:A3", "kind", "id", "team", "number", "seconds", "name", "manual"}]."""
    root = Path(project_root)
    if replay is None:
        replay = _read_json(root / "results" / "replay_data.json") or {}
    if gs is None:
        gs = _read_json(root / "results" / "game_state.json")
    gs = gs if isinstance(gs, dict) else {}
    a = load_assignment(root) or Assignment()
    slot_of, nums = _lineup(gs)
    info = replay.get("track_info") or {}
    team_of = {str(t.get("id")): t.get("team") for t in replay.get("tracks") or [] if isinstance(t, dict)}
    rm = RosterMap(a, info, team_of, slot_of, nums)
    ts = replay.get("t") or []
    step = (float(ts[-1]) - float(ts[0])) / max(1, len(ts) - 1) if len(ts) > 1 else 0.0
    out: list[dict] = []
    for slot in sorted((gs.get("lineup") or {}).get("slots") or {}):
        p = rm.player_for_slot(slot)
        out.append({"key": override_key("slot", slot), "kind": "slot", "id": slot, "team": slot[:1],
                    "number": rm.slot_numbers.get(slot, ""), "seconds": None, "name": p.name if p else "",
                    "manual": override_key("slot", slot) in a.overrides})
    for tr in replay.get("tracks") or []:
        if not isinstance(tr, dict):
            continue
        tid, team = str(tr.get("id")), tr.get("team")
        if team not in ("A", "B"):
            continue
        secs = sum(1 for v in tr.get("x") or [] if v is not None) * step
        num = norm_number((info.get(tid) or {}).get("number"))
        key = override_key("track", tid)
        if not num and key not in a.overrides and secs < 10.0:
            continue  # short anonymous pieces: too many to list
        p = rm.player_for_track(tid)
        out.append({"key": key, "kind": "track", "id": tid, "team": team, "number": num,
                    "seconds": round(secs, 1), "name": p.name if p else "", "manual": key in a.overrides})
    out.sort(key=lambda r: (r["kind"] != "slot", r["team"], not r["number"], -(r["seconds"] or 0), r["id"]))
    return out
