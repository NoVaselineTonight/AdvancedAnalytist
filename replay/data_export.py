"""Simple data export: one new folder of timestamped, documented files meant for an AI (or a person) to analyse.

``export_simple(project_root, dest_parent)`` creates ``<replay name>_data_<YYYYMMDD-HHMMSS>/`` containing

    README_FOR_AI.md        instructions for an AI analyst, generated from the actual data
    game.json               metadata, teams, court geometry, sampling, coverage statistics
    players.csv             one row per person per analysed frame
    ball.csv                one row per analysed frame with a ball position
    events.csv / .json      detected baskets with a running (detected, NOT official) score
    timeline_marks.csv      the user's timeline marks and notes, plus the auto-detected marks
    timeline.json           the same, structured
    track_summary.csv       per track: team, time on court, distance, speeds, mean position
    possession_summary.csv  ball possession time per team
    replay_2d.html          stand-alone 2D replay with the scoring marked
    report.md               the processing report

No Qt here: usable from the GUI, the command line and tests.
"""

from __future__ import annotations

import csv
import json
import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from replay import __version__, court

FORMAT = "advanced-analytist/data-export@1"
POINTS_PER_BASKET = 2  # baskets are detected, the shot value is not: every basket counts 2 points
MAX_PAIR_GAP_S = 1.0  # distance/speed only between measured rows at most this far apart
MAX_PLAUSIBLE_SPEED = 11.0  # m/s; faster steps are tracking glitches (ID switches) and are skipped
SPEED_WINDOW_S = 0.5  # max speed is measured over windows of at least this length (less jitter)
POSSESSION_BREAK_S = 5.0  # no player holding the ball for longer than this ends a team possession

PLAYER_COLUMNS = ["t", "frame", "track_id", "team", "team_name", "role", "x_m", "y_m", "confidence",
                  "interpolated", "calibration", "jersey_number", "source", "player_id", "id_confidence"]
BALL_COLUMNS = ["t", "frame", "x_m", "y_m", "state", "owner_track_id", "owner_team", "owner_team_name"]
EVENT_COLUMNS = ["event_id", "t", "time", "frame", "type", "basket", "team", "team_name", "points", "score_A",
                 "score_B", "confidence", "source", "title", "note"]
MARK_COLUMNS = ["t", "time", "duration_s", "t_end", "track", "track_kind", "title", "note", "color", "mark_id",
                "event_id", "team", "confidence"]
SUMMARY_COLUMNS = ["track_id", "team", "team_name", "role", "team_confidence", "first_seen_t", "last_seen_t",
                   "span_s", "seconds_tracked", "rows", "measured_rows", "pct_interpolated", "distance_m",
                   "mean_speed_m_s", "max_speed_m_s", "mean_x_m", "mean_y_m", "glitch_steps_skipped",
                   "jersey_number", "jersey_number_reads"]
from replay.export_docs import EVENT_TYPE_DOCS  # noqa: E402

POSS_TABLE_COLUMNS = ["possession_id", "team", "team_name", "t0", "t1", "duration_s", "attacks_basket", "start",
                      "passes", "fast_break", "confidence"]
POSSESSION_COLUMNS = ["category", "team", "team_name", "frames", "seconds", "pct_of_analysed_time",
                      "pct_of_possessed_time", "team_possessions", "avg_possession_s"]


# ====================================================================== helpers
def _read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _safe_name(name: str) -> str:
    s = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", name).strip(" ._")
    return s[:60] or "replay"


def fmt_clock(s: float | None) -> str:
    if s is None or not math.isfinite(s):
        return ""
    m, sec = divmod(round(max(0.0, float(s)), 1), 60)  # round first: 59.96 s is "01:00.0", not "00:60.0"
    return f"{int(m):02d}:{sec:04.1f}"


def _f(v: Any) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _bool(v: Any) -> bool:
    return v is True or str(v).lower() in ("true", "1")


def _r(v: float | None, n: int = 3) -> Any:
    return "" if v is None else round(v, n)


def _write_csv(path: Path, columns: list[str], rows: list[list[Any]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(columns)
        w.writerows(rows)


def _pct(a: float, b: float) -> float:
    return round(100.0 * a / b, 1) if b else 0.0


# ====================================================================== loading
class _CourtGeom:
    """This Replay's court (size from its own replay data / calibration, not the program-wide ``court`` state,
    which belongs to whatever Replay was built last and may be in use by a background build)."""

    def __init__(self, replay: dict, tracking: dict, calib: dict | None):
        rc = replay.get("court") if isinstance(replay.get("court"), dict) else {}
        cc = (calib or {}).get("court") if isinstance((calib or {}).get("court"), dict) else {}
        cs = tracking.get("coordinate_system") if isinstance(tracking.get("coordinate_system"), dict) else {}
        self.LENGTH = float(_f(rc.get("length")) or _f(cs.get("length")) or _f(cc.get("length")) or court.LENGTH)
        self.WIDTH = float(_f(rc.get("width")) or _f(cs.get("width")) or _f(cc.get("width")) or court.WIDTH)
        self.MID_Y = self.WIDTH / 2
        self.SPEC = str(cc.get("spec") or court.SPEC)
        sp = court.STANDARDS.get(self.SPEC, court.STANDARDS["FIBA"])
        self.THREE_R, self.THREE_SIDE_Y, self.FT_LINE_X = sp["three_r"], sp["three_side"], sp["ft_x"]
        self.NO_CHARGE_R, self.BASKET_X = sp["no_charge_r"], sp["basket_x"]
        self._lines = rc.get("lines") if isinstance(rc.get("lines"), list) and rc.get("lines") else None
        same = abs(self.LENGTH - court.LENGTH) < 1e-6 and abs(self.WIDTH - court.WIDTH) < 1e-6
        if self._lines is None and same:
            self._lines = [line.tolist() for line in court.court_polylines()]

    def baskets(self) -> list[tuple[float, float]]:
        return [(self.BASKET_X, self.MID_Y), (self.LENGTH - self.BASKET_X, self.MID_Y)]

    def court_polylines(self) -> list[list[list[float]]]:
        if self._lines is not None:
            return self._lines
        # no stored lines and another court is configured: the outline and the half-court line at least
        L, W = self.LENGTH, self.WIDTH
        return [[[0, 0], [L, 0], [L, W], [0, W], [0, 0]], [[L / 2, 0], [L / 2, W]]]


class _Data:
    """Everything the export reads from a project folder."""

    def __init__(self, root: Path):
        self.root = root
        self.info = _read_json(root / "project.json", {}) or {}
        self.meta = _read_json(root / "meta.json", {}) or {}
        res = root / "results"
        self.tracking = _read_json(res / "tracking.json")
        if not isinstance(self.tracking, dict) or "rows" not in self.tracking:
            raise ValueError("This Replay has no tracking data yet: build the 2D replay first.")
        self.replay = _read_json(res / "replay_data.json", {}) or {}
        self.report = _read_json(res / "report.json", {}) or {}
        self.timeline_raw = _read_json(root / "timeline.json")
        self.court = _CourtGeom(self.replay, self.tracking, _read_json(root / "calibration.json"))
        self.name = str(self.info.get("name") or root.name)
        self.columns = self.tracking.get("columns") or []
        self.fps = float(self.tracking.get("process_fps") or self.replay.get("fps") or 10.0)
        game = self.tracking.get("game") or self.replay.get("game") or {}
        teams = game.get("teams") or {}
        self.game = game
        from replay import team_binding as tbind
        bound = tbind.resolved_names(self.meta, self.meta.get(tbind.FIELD))  # stable colour-bound names
        self.team_names = {k: str((teams.get(k) or {}).get("name") or bound[k]) for k in ("A", "B")}
        ov = tbind.team_overview(self.meta)
        # "home" / "away" only when the user confirmed which jersey colour is which team
        self.team_sides = {k: tbind.side_text(teams.get(k) or ov[k], k) for k in ("A", "B")}
        self.team_jersey = {k: str((teams.get(k) or {}).get("colour_name") or
                                   (tbind.colour_name((teams.get(k) or {}).get("primary")) if (teams.get(k) or {}).get("primary")
                                    else ov[k]["colour_name"] if ov[k]["detected"] else "")) for k in ("A", "B")}
        self.team_jersey = {k: "" if v == "unknown" else v for k, v in self.team_jersey.items()}
        disp = (self.info.get("settings") or {}).get("display_colors") or {}
        self.team_colors = {k: disp.get(k) or (teams.get(k) or {}).get("primary") for k in ("A", "B")}
        self.track_info = {str(k): v for k, v in (self.tracking.get("tracks") or {}).items()}
        events = self.tracking.get("events")
        if not events:
            events = self.replay.get("events")
        self.events = [e for e in (events or []) if isinstance(e, dict)]
        # basketball game state (possessions, attack direction, events; replay/game_state.py) + light-model events
        from replay import game_state as gs_mod
        self.game_state = None
        try:
            self.game_state = gs_mod.load_or_build(res, self.replay if self.replay.get("t") else None)
        except Exception as e:  # noqa: BLE001 - the export must work without it
            print("game state unavailable:", type(e).__name__, e)
        self.events = gs_mod.merge_events(self.events, self.game_state, gs_mod.light_events(root))
        self.overrides = {}
        if isinstance(self.timeline_raw, dict) and isinstance(self.timeline_raw.get("auto_overrides"), dict):
            self.overrides = {str(k): v for k, v in self.timeline_raw["auto_overrides"].items() if isinstance(v, dict)}

    def team_name(self, team: str | None) -> str:
        if team in self.team_names:
            return self.team_names[team]  # type: ignore[index]
        return {"referee": "Referee", "unknown": "Unknown", None: "", "": ""}.get(team, str(team))

    def rows(self):
        idx = {c: i for i, c in enumerate(self.columns)}
        need = ["frame", "timestamp", "track_id", "object_type", "team", "x_m", "y_m"]
        if any(c not in idx for c in need):
            raise ValueError(f"tracking.json has unexpected columns: {self.columns}")

        def get(r, c, default=None):
            i = idx.get(c)
            return r[i] if i is not None and i < len(r) else default

        for r in self.tracking["rows"]:
            yield {c: get(r, c) for c in idx}


# ====================================================================== tables
def _players_and_ball(d: _Data) -> tuple[list[dict], list[dict]]:
    players, ball = [], []
    for r in d.rows():
        t, x, y = _f(r.get("timestamp")), _f(r.get("x_m")), _f(r.get("y_m"))
        if t is None or x is None or y is None:
            continue
        row = {"t": t, "frame": int(r["frame"]), "x": x, "y": y, "conf": _f(r.get("confidence")),
               "interp": _bool(r.get("interpolated")), "calib": r.get("calibration") or ""}
        if r.get("object_type") == "ball":
            owner = r.get("ball_owner")
            row.update(state=r.get("ball_state") or ("estimated" if row["interp"] else "possessed"),
                       owner=None if owner in (None, "") else str(owner))
            ball.append(row)
        else:
            team = r.get("team") or "unknown"
            role = "referee" if r.get("object_type") == "referee" or team == "referee" else \
                "unknown" if team == "unknown" else "player"
            row.update(track=str(r["track_id"]), team=team, role=role,
                       source=r.get("source") or ("bridged" if row["interp"] else "seen"),
                       player_id=r.get("player_id"), id_conf=_f(r.get("id_confidence")))
            players.append(row)
    players.sort(key=lambda r: (r["t"], _id_key(r["track"])))
    ball.sort(key=lambda r: r["t"])
    return players, ball


def _id_key(tid: str) -> tuple[int, str]:
    return (int(tid), "") if str(tid).isdigit() else (10 ** 9, str(tid))


def _track_summary(d: _Data, players: list[dict]) -> list[dict]:
    by_track: dict[str, list[dict]] = defaultdict(list)
    for r in players:
        by_track[r["track"]].append(r)
    out = []
    for tid, rows in by_track.items():
        rows.sort(key=lambda r: r["t"])
        meas = [r for r in rows if not r["interp"]]
        dist = moving = 0.0
        glitches = 0
        for a, b in zip(meas, meas[1:]):
            dt = b["t"] - a["t"]
            if dt <= 0 or dt > MAX_PAIR_GAP_S:
                continue
            step = math.hypot(b["x"] - a["x"], b["y"] - a["y"])
            if step / dt > MAX_PLAUSIBLE_SPEED:
                glitches += 1
                continue
            dist += step
            moving += dt
        vmax: float | None = None
        j = 0
        for i, a in enumerate(meas):  # speed over >= SPEED_WINDOW_S windows of consecutive measured rows
            j = max(j, i + 1)
            while j < len(meas) and meas[j]["t"] - a["t"] < SPEED_WINDOW_S:
                j += 1
            if j >= len(meas):
                break
            b = meas[j]
            dt = b["t"] - a["t"]
            if dt > MAX_PAIR_GAP_S + SPEED_WINDOW_S:
                continue
            v = math.hypot(b["x"] - a["x"], b["y"] - a["y"]) / dt
            if v <= MAX_PLAUSIBLE_SPEED:
                vmax = v if vmax is None else max(vmax, v)
        ti = d.track_info.get(tid, {})
        out.append({
            "track_id": tid, "team": rows[0]["team"], "team_name": d.team_name(rows[0]["team"]),
            "role": rows[0]["role"], "team_confidence": ti.get("team_confidence", ""),
            "first_seen_t": rows[0]["t"], "last_seen_t": rows[-1]["t"], "span_s": rows[-1]["t"] - rows[0]["t"],
            "seconds_tracked": len(rows) / d.fps, "rows": len(rows), "measured_rows": len(meas),
            "pct_interpolated": _pct(len(rows) - len(meas), len(rows)), "distance_m": dist,
            "mean_speed_m_s": dist / moving if moving > 0 else None, "max_speed_m_s": vmax,
            "mean_x_m": sum(r["x"] for r in meas) / len(meas) if meas else None,
            "mean_y_m": sum(r["y"] for r in meas) / len(meas) if meas else None,
            "glitch_steps_skipped": glitches,
            "jersey_number": str(ti.get("number") or ""), "jersey_number_reads": ti.get("number_reads", ""),
        })
    out.sort(key=lambda s: _id_key(s["track_id"]))
    return out


def _possession(d: _Data, ball: list[dict], team_of_track: dict[str, str], n_frames: int) -> list[dict]:
    dt = 1.0 / d.fps
    counts: dict[str, int] = defaultdict(int)
    seq: list[tuple[float, str]] = []  # (t, team) of frames where a player of a team holds the ball
    for r in ball:
        team = team_of_track.get(r["owner"] or "", "") if r["state"] == "possessed" else ""
        if r["state"] == "possessed":
            if team in ("A", "B"):
                counts[team] += 1
                seq.append((r["t"], team))
            else:
                counts["other"] += 1
        else:
            counts["estimated"] += 1
    spells: dict[str, list[float]] = {"A": [], "B": []}  # team possessions: runs of one team holding the ball
    cur, start, last = None, 0.0, 0.0
    for t, team in seq:
        if cur is not None and (team != cur or t - last > POSSESSION_BREAK_S):
            spells[cur].append(last - start + dt)
            cur = None
        if cur is None:
            cur, start = team, t
        last = t
    if cur is not None:
        spells[cur].append(last - start + dt)
    possessed = counts["A"] + counts["B"]
    no_ball = max(0, n_frames - len(ball))
    out = []
    for team in ("A", "B"):
        sp = spells[team]
        out.append({"category": "possessed", "team": team, "team_name": d.team_name(team), "frames": counts[team],
                    "seconds": counts[team] * dt, "pct_of_analysed_time": _pct(counts[team], n_frames),
                    "pct_of_possessed_time": _pct(counts[team], possessed), "team_possessions": len(sp),
                    "avg_possession_s": sum(sp) / len(sp) if sp else None})
    for cat, n, label in (("possessed_unknown_team", counts["other"], "held by a person of no team"),
                          ("estimated", counts["estimated"], "in flight / loose (position estimated)"),
                          ("no_ball_position", no_ball, "no ball position (not seen or not calibrated)")):
        out.append({"category": cat, "team": "", "team_name": label, "frames": n, "seconds": n * dt,
                    "pct_of_analysed_time": _pct(n, n_frames), "pct_of_possessed_time": "",
                    "team_possessions": "", "avg_possession_s": None})
    return out


def _scored_events(d: _Data) -> tuple[list[dict], list[dict]]:
    """Events with the running detected score; dismissed (by the user) events are returned separately."""
    kept, dismissed = [], []
    score = {"A": 0, "B": 0}
    for ev in sorted(d.events, key=lambda e: _f(e.get("t")) or 0.0):
        eid = str(ev.get("id", ""))
        ov = d.overrides.get(eid, {})
        if ov.get("dismissed"):
            dismissed.append(ev)
            continue
        team = ev.get("team") if ev.get("team") in ("A", "B") else None
        typ = str(ev.get("type") or "event")
        pts = int(_f(ev.get("points_guess")) or POINTS_PER_BASKET) if typ == "basket" and team else 0
        if pts:
            score[team] += pts  # type: ignore[index]
        kept.append({**ev, "id": eid, "team": team, "team_name": d.team_name(team) if team else "",
                     "points": pts, "score_A": score["A"], "score_B": score["B"],
                     "title": str(ov.get("title") or _event_title(ev, d)), "note": str(ov.get("note") or ""),
                     "user_edited": bool(ov.get("title") or ov.get("note"))})
    return kept, dismissed


def _event_title(ev: dict, d: _Data) -> str:
    if ev.get("label"):
        from replay.game_state import event_title
        return event_title(ev, d.team_names)
    title = "Basket" if ev.get("type") == "basket" else str(ev.get("type") or "Event").replace("_", " ").capitalize()
    if ev.get("team") in ("A", "B"):
        title += f" – {d.team_name(ev['team'])}"
    if ev.get("basket"):
        title += f" ({ev['basket']})"
    return title


def _timeline(d: _Data, events: list[dict]) -> tuple[list[dict], list[dict], dict]:
    """(user tracks, auto marks, timeline doc). Auto marks = detected events with the user's edits applied,
    dismissed ones left out."""
    tl = d.timeline_raw if isinstance(d.timeline_raw, dict) else {}
    tracks = [t for t in (tl.get("tracks") or []) if isinstance(t, dict)]
    user_tracks = []
    for tr in tracks:
        if tr.get("kind") == "auto":
            continue
        marks = sorted((m for m in (tr.get("marks") or []) if isinstance(m, dict)), key=lambda m: _f(m.get("t")) or 0)
        user_tracks.append({"id": tr.get("id"), "name": tr.get("name") or "Track", "color": tr.get("color"),
                            "visible": tr.get("visible", True), "marks": [
                                {"id": m.get("id"), "t": _f(m.get("t")) or 0.0, "duration": _f(m.get("duration")) or 0.0,
                                 "title": m.get("title") or "", "note": m.get("note") or "",
                                 "color": m.get("color") or tr.get("color")} for m in marks]})
    auto_name = next((t.get("name") for t in tracks if t.get("kind") == "auto"), None) or "Auto detection"
    auto = [{"id": f"auto:{e['id']}", "event_id": e["id"], "t": _f(e.get("t")) or 0.0,
             "duration": _f(e.get("duration")) or 0.0, "title": e["title"],
             "note": e["note"] or (f"Detected automatically (confidence {(_f(e.get('confidence')) or 0) * 100:.0f}%)."),
             "team": e.get("team"), "confidence": e.get("confidence"), "track": auto_name}
            for e in events]
    doc = {"format": "advanced-analytist/timeline-export@1", "source_format": tl.get("format"),
           "has_user_timeline": bool(tl),
           "note": "User tracks hold marks and notes written by the person who analysed the video. auto_marks are "
                   "the automatically detected events with the user's title/note edits applied; events the user "
                   "dismissed are left out.",
           "tracks": user_tracks, "auto_track_name": auto_name, "auto_marks": auto,
           "dismissed_event_ids": sorted(k for k, v in d.overrides.items() if v.get("dismissed"))}
    return user_tracks, auto, doc


# ====================================================================== html
def _replay_html(d: _Data, events: list[dict], user_tracks: list[dict]) -> str | None:
    template_path = Path(__file__).parent / "viewer" / "template.html"
    payload = dict(d.replay) if d.replay.get("t") else None
    if payload is None:
        old = d.root / "results" / "replay.html"
        return old.read_text(encoding="utf-8") if old.exists() else None
    payload["video_src"] = ""  # the video is not part of this folder; "Load video" in the page still works
    payload["events"] = [{k: e.get(k) for k in ("id", "type", "t", "frame", "basket", "team", "confidence", "title",
                                                 "points")} for e in events]
    payload["points_per_basket"] = POINTS_PER_BASKET
    payload["score_note"] = "Detected automatically - not the official score"
    payload["marks"] = [{"t": m["t"], "title": m["title"], "note": m["note"], "track": tr["name"]}
                        for tr in user_tracks for m in tr["marks"]]
    for k in ("A", "B"):
        if d.team_colors.get(k) and k in (payload.get("game") or {}).get("teams", {}):
            payload["game"]["teams"][k] = {**payload["game"]["teams"][k], "primary": d.team_colors[k]}
    blob = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    return template_path.read_text(encoding="utf-8").replace("/*__DATA__*/null", blob)


# ====================================================================== main
def export_simple(project_root: Path, dest_parent: Path, include_names: bool | None = None) -> Path:
    """Write a new ``<replay name>_data_<YYYYMMDD-HHMMSS>`` folder in ``dest_parent`` and return it."""
    project_root, dest_parent = Path(project_root), Path(dest_parent)
    d = _Data(project_root)
    court = d.court  # noqa: F811 - this Replay's court, not the program-wide one
    now = time.time()
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(now))
    base = dest_parent / f"{_safe_name(d.name)}_data_{stamp}"
    out, n = base, 2
    while out.exists():
        out = Path(f"{base}-{n}")
        n += 1
    out.mkdir(parents=True)

    players, ball = _players_and_ball(d)
    team_of_track = {tid: str(v.get("team")) for tid, v in d.track_info.items()}
    for r in players:
        team_of_track.setdefault(r["track"], r["team"])
    frames_t = d.replay.get("t") or sorted({r["t"] for r in players} | {r["t"] for r in ball})
    n_frames = len(frames_t)

    # ---- players.csv / ball.csv
    _write_csv(out / "players.csv", PLAYER_COLUMNS, [
        [round(r["t"], 3), r["frame"], r["track"], r["team"], d.team_name(r["team"]), r["role"], round(r["x"], 3),
         round(r["y"], 3), _r(r["conf"]), int(r["interp"]), r["calib"],
         str(d.track_info.get(r["track"], {}).get("number") or ""), r["source"],
         "" if r["player_id"] in (None, "") else r["player_id"], _r(r["id_conf"])] for r in players])
    _write_csv(out / "ball.csv", BALL_COLUMNS, [
        [round(r["t"], 3), r["frame"], round(r["x"], 3), round(r["y"], 3), r["state"], r["owner"] or "",
         team_of_track.get(r["owner"] or "", "") if r["owner"] else "",
         d.team_name(team_of_track.get(r["owner"] or "")) if r["owner"] else ""] for r in ball])

    # ---- summaries
    summary = _track_summary(d, players)
    _write_csv(out / "track_summary.csv", SUMMARY_COLUMNS, [
        [s["track_id"], s["team"], s["team_name"], s["role"], s["team_confidence"], _r(s["first_seen_t"]),
         _r(s["last_seen_t"]), _r(s["span_s"], 2), _r(s["seconds_tracked"], 2), s["rows"], s["measured_rows"],
         s["pct_interpolated"], _r(s["distance_m"], 1), _r(s["mean_speed_m_s"], 2), _r(s["max_speed_m_s"], 2),
         _r(s["mean_x_m"], 2), _r(s["mean_y_m"], 2), s["glitch_steps_skipped"], s["jersey_number"],
         s["jersey_number_reads"]] for s in summary])
    poss = _possession(d, ball, team_of_track, n_frames)
    _write_csv(out / "possession_summary.csv", POSSESSION_COLUMNS, [
        [p["category"], p["team"], p["team_name"], p["frames"], _r(p["seconds"], 1), p["pct_of_analysed_time"],
         p["pct_of_possessed_time"], p["team_possessions"], _r(p["avg_possession_s"], 1)] for p in poss])

    # ---- events
    events, dismissed = _scored_events(d)
    _write_csv(out / "events.csv", EVENT_COLUMNS, [
        [e["id"], _r(_f(e.get("t"))), fmt_clock(_f(e.get("t"))), e.get("frame", ""), e.get("type", ""),
         e.get("basket") or "", e["team"] or "", e["team_name"], e["points"], e["score_A"], e["score_B"],
         _r(_f(e.get("confidence"))), e.get("source", "auto"), e["title"], e["note"]] for e in events])
    final = {"A": events[-1]["score_A"] if events else 0, "B": events[-1]["score_B"] if events else 0}
    (out / "events.json").write_text(json.dumps({
        "format": "advanced-analytist/events-export@1",
        "warning": "DETECTED automatically from video - NOT the official score. Every detected basket counts "
                   f"{POINTS_PER_BASKET} points (2- and 3-pointers and free throws are not distinguished).",
        "points_per_basket": POINTS_PER_BASKET,
        "teams": {k: {"name": d.team_names[k], "color": d.team_colors.get(k)} for k in ("A", "B")},
        "detected_final_score": {d.team_names["A"]: final["A"], d.team_names["B"]: final["B"]},
        "detected_final_score_by_label": final,
        "events": [{**e, "time": fmt_clock(_f(e.get("t")))} for e in events],
        "dismissed_by_user": dismissed,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- game state (possessions, attack direction, game events)
    gsd = d.game_state or {}
    _write_csv(out / "possessions.csv", POSS_TABLE_COLUMNS, [
        [p.get("id"), p.get("team"), d.team_name(p.get("team")), _r(_f(p.get("t0")), 2), _r(_f(p.get("t1")), 2),
         _r(_f(p.get("duration_s")), 2), p.get("attacks") or "", p.get("start") or "", p.get("passes"),
         int(bool(p.get("fast_break"))), _r(_f(p.get("confidence")), 2)] for p in gsd.get("possessions") or []])
    (out / "game_state.json").write_text(json.dumps({k: v for k, v in gsd.items() if k not in ("inputs", "config")},
                                                    indent=1, ensure_ascii=False), encoding="utf-8")
    from replay import ball_events  # passes.csv / shots.csv (shot chart) from the ball-event layer
    for name, rows in ball_events.tables(gsd.get("ball_events")).items():
        _write_csv(out / name, rows[0], rows[1:])
    from replay import stats as stats_mod  # stats.json / stats_*.csv: team, shot-zone, lineup stats (stats.py)
    files = dict(FILE_DESCRIPTIONS)  # per export: never mutate the module-level dict
    files.update(stats_mod.export_into(out, project_root, d.game_state, d.replay))

    # ---- timeline
    user_tracks, auto_marks, tl_doc = _timeline(d, events)
    (out / "timeline.json").write_text(json.dumps(tl_doc, indent=2, ensure_ascii=False), encoding="utf-8")
    mark_rows = []
    for tr in user_tracks:
        for m in tr["marks"]:
            mark_rows.append([round(m["t"], 3), fmt_clock(m["t"]), round(m["duration"], 3),
                              round(m["t"] + m["duration"], 3), tr["name"], "user", m["title"], m["note"],
                              m["color"] or "", m["id"], "", "", ""])
    for m in auto_marks:
        mark_rows.append([round(m["t"], 3), fmt_clock(m["t"]), round(m["duration"], 3),
                          round(m["t"] + m["duration"], 3), m["track"], "auto", m["title"], m["note"], "",
                          m["id"], m["event_id"], m["team"] or "", _r(_f(m["confidence"]))])
    mark_rows.sort(key=lambda r: r[0])
    _write_csv(out / "timeline_marks.csv", MARK_COLUMNS, mark_rows)

    # ---- html + report
    html = _replay_html(d, events, user_tracks)
    if html is not None:
        (out / "replay_2d.html").write_text(html, encoding="utf-8")
    rep_md = project_root / "results" / "report.md"
    if rep_md.exists():
        (out / "report.md").write_text(rep_md.read_text(encoding="utf-8"), encoding="utf-8")

    # ---- game.json
    stats_clock = None
    try:
        stats_clock = ((_read_json(out / "stats.json", {}) or {}).get("lineups") or {}).get("clock")
    except Exception:  # noqa: BLE001
        stats_clock = None
    vi = d.info.get("video_info") or {}
    t0 = frames_t[0] if frames_t else None
    t1 = frames_t[-1] if frames_t else None
    role_counts: dict[str, int] = defaultdict(int)
    team_tracks: dict[str, int] = defaultdict(int)
    for s in summary:
        role_counts[s["role"]] += 1
        team_tracks[s["team"]] += 1
    report = {k: v for k, v in d.report.items() if k != "people"}
    if "people" in d.report:
        report["people"] = {k: v for k, v in d.report["people"].items() if k != "team_centroids_lab"}
    game = {
        "format": FORMAT,
        "exported_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(now)),
        "exported_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
        "app": "AdvancedAnalytist", "app_version": __version__,
        "replay_name": d.name,
        "source_video": (d.info.get("source") or {}).get("filename"),
        "game": {
            "date": d.game.get("game_date") or d.meta.get("game_date"),
            "location": d.game.get("location") or d.meta.get("location"),
            "court_standard": d.game.get("court_standard") or "FIBA",
            "teams": {k: {"label": k, "role": d.team_sides[k], "name": d.team_names[k],
                          "jersey": d.team_jersey[k], "color": d.team_colors.get(k),
                          "secondary_color": ((d.game.get("teams") or {}).get(k) or {}).get("secondary")}
                      for k in ("A", "B")},
            "referee_color": d.game.get("referee_color"),
        },
        "video": {"fps": vi.get("fps"), "duration_s": vi.get("duration_s"), "size": vi.get("size"),
                  "note": "t = seconds in the video (the playback copy); frame = frame number in that video"},
        "sampling": {"analysed_frames_per_second": d.fps,
                     "frame_stride": (d.info.get("settings") or {}).get("frame_stride"),
                     "analysed_frames": n_frames, "time_range_s": [t0, t1]},
        "court": {
            "standard": court.SPEC, "length_m": court.LENGTH, "width_m": court.WIDTH, "units": "metres",
            "origin": "corner where the LEFT baseline meets the NEAR sideline, as seen from the camera",
            "x_axis": f"along the sideline, 0 (left baseline) .. {court.LENGTH:g} (right baseline)",
            "y_axis": f"along the baseline, 0 (near sideline) .. {court.WIDTH:g} (far sideline)",
            "baskets": {"L": list(court.baskets()[0]), "R": list(court.baskets()[1])},
            "basket_note": "floor point under the centre of each ring",
            "lines": [[[round(float(p[0]), 3), round(float(p[1]), 3)] for p in line]
                      for line in court.court_polylines()],
        },
        "counts": {"player_rows": len(players), "ball_rows": len(ball), "tracks_by_role": dict(role_counts),
                   "tracks_by_team": dict(team_tracks), "detected_events": len(events),
                   "events_dismissed_by_user": len(dismissed),
                   "user_marks": sum(len(t["marks"]) for t in user_tracks)},
        "detected_score": {"points_per_basket": POINTS_PER_BASKET, "final": final,
                           "warning": "detected automatically, not official"},
        "coverage": report,
        "files": {k: v for k, v in files.items() if k in ("README_FOR_AI.md", "game.json") or (out / k).exists()},
        "clock": {"t": "video seconds", "game_clock": stats_clock or None},
    }
    (out / "game.json").write_text(json.dumps(game, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- README
    (out / "README_FOR_AI.md").write_text(
        _readme(d, game, players, ball, summary, poss, events, dismissed, user_tracks, auto_marks, out.name,
                html is not None, stats_clock,
                {p.name for p in out.glob("*.csv")}),
        encoding="utf-8")
    from replay.roster_names import augment_export  # roster hook: player_name column (only with roster.json)
    augment_export(project_root, out, include_names)  # roster hook (names only if enabled; default off)
    return out


FILE_DESCRIPTIONS = {
    "README_FOR_AI.md": "how to read and analyse this data (start here)",
    "game.json": "metadata, teams and colours, court geometry, sampling, coverage statistics",
    "players.csv": "one row per tracked person per analysed frame",
    "ball.csv": "one row per analysed frame with a ball position",
    "events.csv": "detected baskets with the running detected score",
    "events.json": "the same events with the detector's evidence",
    "timeline_marks.csv": "timeline marks and notes (user and auto)",
    "timeline.json": "timeline tracks, marks and notes, structured",
    "track_summary.csv": "one row per track: team, time, distance, speeds, mean position",
    "possession_summary.csv": "ball possession time per team",
    "possessions.csv": "one row per team possession (game-state engine): team, time, attacked basket, how it "
                       "started, passes, confidence",
    "game_state.json": "basketball game state: attack direction per half, possessions, free throws, shots, "
                       "events, segments, on-court slots (lineup)",
    "passes.csv": "detected passes (ball events): passer -> receiver, from/to court position, length, type",
    "shots.csv": "detected field-goal attempts for a shot chart: shooter, court x/y, distance, outcome",
    "replay_2d.html": "stand-alone 2D replay with the scoring marked (open in a browser)",
    "report.md": "processing report (calibration and tracking coverage)",
}


# ====================================================================== README
def _readme_game_state(d: _Data, add: Any, events: list[dict]) -> None:
    """Section 6b: the basketball game state (replay/game_state.py)."""
    from replay import game_state as gs_mod
    gsd = d.game_state or {}
    add("## 6b. Game state: possessions, attack direction, events, lineup\n")
    if not gsd.get("possessions") and not gsd.get("events"):
        add("- No game state is available for this Replay (rebuild the replay to create it).\n")
        return
    add("The game-state engine turns the tracking into basketball structure (deterministic rules, no AI). Read "
        "these rules of the game before analysing:\n")
    for ln in gs_mod.SEMANTICS:
        add(f"- {ln}")
    add("\n**Attack direction per half:**\n")
    for ln in gs_mod.direction_lines(gsd, d.team_names):
        add(ln)
    poss = gsd.get("possessions") or []
    by_team = {k: [p for p in poss if p.get("team") == k] for k in ("A", "B")}
    add(f"\n**Possessions** (`possessions.csv`, `game_state.json` → `possessions`): {len(poss)} in total – "
        + ", ".join(f"{d.team_name(k)} {len(v)} (avg {sum(p['duration_s'] for p in v) / len(v):.1f} s)" if v else
                    f"{d.team_name(k)} 0" for k, v in by_team.items())
        + ". A possession is a spell of one team controlling the ball (holder seen by the tracker, short "
          "mix-ups ignored, corrected by where the ball is when the holder's team label contradicts the attack "
          "direction); `start` tells how it began (after shot = rebound, steal, after free throws, inbound ...).")
    fts = gsd.get("free_throw_windows") or []
    if fts:
        add(f"\n**Free-throw situations:** {len(fts)} – " + ", ".join(
            f"{fmt_clock(_f(w.get('t0')))} {d.team_name(w.get('team')) or 'unknown team'} at basket {w.get('side')}"
            for w in fts[:20]) + ". The ball is dead there; do not read passes or team tactics into them.")
    other = [e for e in events if e.get("type") != "basket"]
    kinds: dict[str, int] = defaultdict(int)
    for e in other:
        kinds[str(e.get("type"))] += 1
    add(f"\n**Detected game events** (`events.csv`, type ≠ basket; also on the app's \"Auto detection\" timeline "
        f"track): {len(other)} – " + ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in sorted(kinds.items()))
        + ". Each has a confidence; `source = game_state` (rules) or `jev` (light AI model, e.g. lost balls).\n")
    for ln in gs_mod.event_lines(events, d.team_names, limit=120):
        add(ln)
    lu = gsd.get("lineup") or {}
    add(f"\n**On-court slots (lineup):** `game_state.json` → `lineup.slots` maps each slot A1–A5 / B1–B5 to the "
        "track IDs that filled it over time ([track, first_t, last_t]); tracks that could not be placed (a 6th "
        f"same-team person at once: duplicates, bench) are in `extra_tracks` ({len(lu.get('extra_tracks') or [])}). "
        "A slot whose player is not visible is still on court at an unknown position (last seen where its last "
        "track ended). `lineup.numbers` = the jersey number of a slot's player, where one was read.\n")


def _sanity_warnings(d: _Data, players: list[dict], ball: list[dict], n_frames: int, pct_cal: Any,
                     ppl: dict) -> list[str]:
    """Plain-language warnings about the data, so an AI doesn't build conclusions on a broken input."""
    out = []
    rows_a = sum(1 for r in players if r["team"] == "A")
    rows_b = sum(1 for r in players if r["team"] == "B")
    A, B = d.team_names["A"], d.team_names["B"]
    if rows_a + rows_b and min(rows_a, rows_b) < 0.25 * (rows_a + rows_b):
        few = B if rows_b < rows_a else A
        out.append(f"Team assignment looks unreliable: only {_pct(min(rows_a, rows_b), rows_a + rows_b)} % of the "
                   f"team rows are {few} ({rows_a} {A} rows vs {rows_b} {B} rows). Two teams on court should be "
                   "roughly balanced, so many players were probably put in the wrong team (similar jersey colours). "
                   "Do NOT draw team-vs-team conclusions (possession share, team distance, spacing); report "
                   "per-track and whole-game numbers instead and say why.")
    unknown = sum(1 for r in players if r["team"] == "unknown")
    if players and unknown > 0.2 * len(players):
        out.append(f"{_pct(unknown, len(players))} % of the person rows have no team (`unknown`).")
    if isinstance(pct_cal, (int, float)) and pct_cal < 80:
        out.append(f"The court calibration is valid for only {pct_cal} % of the analysed frames; the rest has no "
                   "positions at all.")
    avg = ppl.get("avg_tracked_per_calibrated_frame")
    if isinstance(avg, (int, float)) and avg < 6:
        out.append(f"Only {avg} people are tracked per frame on average (about 12–13 are on court): coverage is "
                   "low, so team totals are strong underestimates.")
    if n_frames and len(ball) < 0.3 * n_frames:
        out.append(f"The ball has a court position in only {_pct(len(ball), n_frames)} % of the analysed frames; "
                   "possession numbers cover only that part of the video.")
    return out


def _readme(d: _Data, game: dict, players: list[dict], ball: list[dict], summary: list[dict], poss: list[dict],
            events: list[dict], dismissed: list[dict], user_tracks: list[dict], auto_marks: list[dict],
            folder: str, has_html: bool, stats_clock: dict | None = None,
            csv_files_written: Any = ()) -> str:
    court = d.court  # noqa: F811 - this Replay's court, not the program-wide one
    A, B = d.team_names["A"], d.team_names["B"]
    cA, cB = d.team_colors.get("A") or "unknown", d.team_colors.get("B") or "unknown"
    rep = d.report
    cal = rep.get("calibration", {})
    ppl = rep.get("people", {})
    t0, t1 = game["sampling"]["time_range_s"]
    span = (t1 - t0) if t0 is not None and t1 is not None else 0.0
    vdur = game["video"]["duration_s"]
    n_frames = game["sampling"]["analysed_frames"]
    roles = game["counts"]["tracks_by_role"]
    teams_tr = game["counts"]["tracks_by_team"]
    n_interp = sum(1 for r in players if r["interp"])
    n_unknown_rows = sum(1 for r in players if r["team"] == "unknown")
    calib_counts = defaultdict(int)
    for r in players:
        calib_counts[r["calib"]] += 1
    ball_poss = sum(1 for r in ball if r["state"] == "possessed")
    pA = next(p for p in poss if p["team"] == "A")
    pB = next(p for p in poss if p["team"] == "B")
    final = game["detected_score"]["final"]
    n_marks = sum(len(t["marks"]) for t in user_tracks)
    real = [s for s in summary if s["role"] == "player" and s["measured_rows"] >= max(5, d.fps * 3)]
    top = sorted(real, key=lambda s: -s["distance_m"])[:3]
    detail_keys = sorted({k for e in events for k in (e.get("details") or {}).keys()})
    pct_cal = cal.get("valid_pct")
    fps = d.fps
    dt = 1.0 / fps

    def num(v: Any, nd: int = 1) -> str:
        return "n/a" if v is None or v == "" else f"{v:.{nd}f}" if isinstance(v, float) else str(v)

    L = []
    add = L.append
    add(f"# {d.name} – basketball tracking data for AI analysis\n")
    add(f"Exported by AdvancedAnalytist {__version__} on {game['exported_at']} (folder `{folder}`).\n")
    add("You are reading the instructions for an AI analyst. This folder contains the positions of every tracked "
        "person and the ball on the court, measured automatically from ONE broadcast/handheld video of a "
        "basketball game, plus detected baskets and notes a human wrote on a timeline. Read this whole file "
        "before analysing; section 9 lists the mistakes that are easy to make with this data.\n")

    add("## 0. Key facts\n")
    add("| | |\n|---|---|")
    rows = [
        ("Replay", d.name),
        ("Game", f"**{A}** (team label `A`, {d.team_sides['A']}, {d.team_jersey['A']} jerseys, colour {cA}) vs "
                 f"**{B}** (team label `B`, {d.team_sides['B']}, {d.team_jersey['B']} jerseys, colour {cB})"),
        ("Date / location", f"{game['game']['date'] or 'not given'} / {game['game']['location'] or 'not given'}"),
        ("Video length", f"{num(vdur)} s" if vdur else "unknown"),
        ("Analysed time range", f"{fmt_clock(t0)} – {fmt_clock(t1)} of the video ({span:.1f} s)" if t0 is not None
         else "none"),
        ("Sampling", f"{fps:.2f} analysed frames per second (one row every {dt:.3f} s); {n_frames} analysed frames"),
        ("Court calibration valid", f"{num(pct_cal)} % of analysed frames (keyframe {cal.get('keyframe_frames', 'n/a')}, "
                                    f"propagated {cal.get('propagated_frames', 'n/a')}, missing "
                                    f"{cal.get('missing_frames', 'n/a')} frames)"),
        ("People tracked per calibrated frame", f"{num(ppl.get('avg_tracked_per_calibrated_frame'))} on average "
                                                f"({num(ppl.get('avg_measured_per_calibrated_frame'))} measured, the rest "
                                                "interpolated); a full game situation has 10 players + 2–3 referees"),
        ("Tracks", f"{len(summary)} track IDs: {teams_tr.get('A', 0)} {A}, {teams_tr.get('B', 0)} {B}, "
                   f"{roles.get('referee', 0)} referee, {roles.get('unknown', 0)} unknown"),
        ("players.csv", f"{len(players)} rows, {_pct(n_interp, len(players))} % interpolated, "
                        f"{_pct(n_unknown_rows, len(players))} % team unknown"),
        ("Ball", f"{len(ball)} rows with a court position ({_pct(len(ball), n_frames)} % of analysed frames); "
                 f"held by a player in {ball_poss} of them, the rest estimated"),
        ("Possession (ball held)", f"{A} {pA['seconds']:.1f} s ({pA['pct_of_possessed_time']} %), {B} "
                                   f"{pB['seconds']:.1f} s ({pB['pct_of_possessed_time']} %)"),
        ("Detected baskets", f"{sum(1 for e in events if e.get('type') == 'basket')} → detected score {A} {final['A']} – {final['B']} {B} "
                             f"(**not official**, {POINTS_PER_BASKET} points per basket)"
                             + (f"; {len(dismissed)} detection(s) dismissed by the user" if dismissed else "")),
        ("User timeline", f"{n_marks} mark(s) with notes on {len(user_tracks)} track(s)" if user_tracks else
         "no user timeline in this Replay"),
    ]
    L += [f"| {a} | {b} |" for a, b in rows]
    add("")
    warnings = _sanity_warnings(d, players, ball, n_frames, pct_cal, ppl)
    if warnings:
        add("**Automatic sanity checks – this data has known problems; take them into account and mention them "
            "in your report:**\n")
        L += [f"- ⚠ {w}" for w in warnings]
        add("")

    add("## 1. What the game and the data are\n")
    add(f"- One video of a basketball game between **{A}** and **{B}** was analysed by AdvancedAnalytist: a "
        "person detector finds everyone in each analysed frame, a tracker links detections over time into "
        "tracks, a calibration maps the floor in the image to the court (homography), and each person's floor "
        "point (feet) is projected to court metres and smoothed (Kalman/RTS smoother).")
    add(f"- Teams were assigned automatically by jersey colour clustering"
        f"{' (seeded with the known team colours / players picked by the user)' if ppl.get('team_clustering_seeded_by_metadata') else ''}. "
        f"`A` = {A} ({d.team_sides['A']}, {d.team_jersey['A']} jerseys), `B` = {B} ({d.team_sides['B']}, "
        f"{d.team_jersey['B']} jerseys). Colours: {A} {cA}, {B} {cB}.")
    add("- The ball is located by a ball detector; when a player holds it, its court position is the "
        "holder's position (`possessed`); between holders (passes, shots, loose balls) the position is "
        "interpolated (`estimated`). A ball in the air is never projected directly.")
    add("- Baskets were detected automatically (see section 6). A human may have added notes on a timeline "
        "(section 7). Nothing in this folder is official game statistics.\n")

    add("## 2. Coordinate system\n")
    add(f"- {court.SPEC} court markings, **{court.LENGTH:g} m × {court.WIDTH:g} m**, units metres.")
    add("- **Origin (0, 0)** = the corner where the **LEFT baseline meets the NEAR sideline, as seen from the "
        "camera**.")
    bl, br = court.baskets()
    add(f"- **x** runs along the sideline: 0 = left baseline → {court.LENGTH:g} = right baseline. **y** runs along "
        f"the baseline: 0 = near sideline (camera side) → {court.WIDTH:g} = far sideline. Half-court line at "
        f"x = {court.LENGTH / 2:g}.")
    add(f"- Basket floor points (centre of the ring projected on the floor): **L = ({bl[0]:g}, {bl[1]:g})**, "
        f"**R = ({br[0]:g}, {br[1]:g})**. The 3-point line is {court.THREE_R:g} m from the basket point "
        f"({court.THREE_SIDE_Y:g} m from the sideline in "
        f"the corners); the free-throw line is {court.FT_LINE_X:g} m from the baseline (x = {court.FT_LINE_X:g} and "
        f"x = {court.LENGTH - court.FT_LINE_X:g}); the restricted-area arc has radius {court.NO_CHARGE_R:g} m. All "
        "court lines are in `game.json` → `court.lines` as polylines.")
    add(f"- Distance to a basket: `d = sqrt((x - bx)^2 + (y - {court.MID_Y:g})^2)` with bx = {bl[0]:g} or {br[0]:g}.")
    add("- **Which team attacks which basket** (teams switch at half time), inferred by the game-state engine "
        "from free throws, made baskets and half-court possessions (details in section 6b):")
    from replay import game_state as gs_mod
    for ln in gs_mod.direction_lines(d.game_state, d.team_names):
        add(f"  {ln}")
    add("")

    add("## 3. Files and columns\n")
    add("All times are **video seconds** (`t`) measured in the video the analysis used; `frame` is the frame "
        "number in that video (0-based, = round(t × video fps)). Use `t` to join files. CSV files are UTF-8, comma separated, with a "
        "header row. Empty cells mean \"not available\".\n")
    add("### players.csv – one row per tracked person per analysed frame\n")
    add("| column | meaning |\n|---|---|")
    for c, m in [
        ("t", "time in the video, seconds"), ("frame", "video frame number"),
        ("track_id", "tracker ID (integer). NOT a player identity or jersey number, see section 5"),
        ("jersey_number", "jersey number read from the video for this track (text, e.g. `7`, `23`, `00`; empty "
                          "= not read reliably). Same number + same team = same player, also across tracks"),
        ("team", f"`A` ({A}), `B` ({B}), `referee` or `unknown`"), ("team_name", "readable team name"),
        ("role", "`player`, `referee` or `unknown` (unknown = could not be assigned; often bench, coaches, crowd)"),
        ("x_m, y_m", "court position in metres (section 2) of the floor point under the person"),
        ("confidence", "detector confidence 0..1 of this detection (interpolated rows: half the track's mean)"),
        ("interpolated", "1 = position filled in across a short gap (≤ 1 s) by the smoother, not measured"),
        ("calibration", "`keyframe` = a frame calibrated by hand (most accurate); `propagated` = calibration "
                        "carried from keyframes using the measured camera motion"),
        ("source", "how this position was obtained: `seen`, `bridged`, `rejected`, `hidden`, `offscreen` or "
                   "`predicted` or `unsure` (section 5, \"Which rows to trust\")"),
        ("player_id", "continuous player id: stays the same across track_id breaks of one player (same lineup "
                      "slot without a dead ball in between, or same team + jersey number); empty = unknown"),
        ("id_confidence", "0..1 how sure the identity of this row is (track identity confidence x 1.0 seen, "
                          "0.7 bridged/rejected/hidden/offscreen, 0.4 predicted); empty = unknown"),
    ]:
        add(f"| `{c}` | {m} |")
    add("\n### ball.csv – ball position, one row per analysed frame where it is known\n")
    add("| column | meaning |\n|---|---|")
    for c, m in [
        ("t, frame", "as above"), ("x_m, y_m", "court position, metres"),
        ("state", "`possessed` = a player holds the ball, position = that player's position; `estimated` = in "
                  "flight / loose, interpolated between holders (rough)"),
        ("owner_track_id", "track_id of the player holding the ball (possessed rows)"),
        ("owner_team, owner_team_name", "that player's team"),
    ]:
        add(f"| `{c}` | {m} |")
    add("\n### events.csv / events.json – detected baskets\n")
    add("| column | meaning |\n|---|---|")
    for c, m in [
        ("event_id", "stable ID, e.g. `basket-<frame>-L`"), ("t, time, frame", "when the basket was detected"),
        ("type", "event type: `basket` plus the game-event types listed below"),
        ("basket", f"`L` (x = {court.baskets()[0][0]:g}) or `R` (x = {court.baskets()[1][0]:g}): the basket the ball went into"),
        ("team, team_name", "team credited with the event (empty = could not be determined; a basket then scores "
                            "no points)"),
        ("points", "points of this row: baskets only, 2 normally, 1 when recognised as a free throw, 3 when "
                   "estimated a 3-pointer (title `Made 3?`); ALWAYS 0 for every other event type"),
        ("score_A, score_B", "running DETECTED score after this row; only `basket` rows change it, other rows "
                             "repeat the current score"),
        ("confidence", "detector confidence 0..1"),
        ("source", "who produced the row: `auto` = basket detector (default for baskets), `game_state` = "
                   "rules layer (rebounds, steals, turnovers, shots, free throws, breaks ...), `jev` = light AI "
                   "model (e.g. lost balls)"),
        ("title, note", "title (edited by the user if they changed it) and the user's note"),
    ]:
        add(f"| `{c}` | {m} |")
    add("\n`events.json` additionally holds `details` (the detector's evidence for each event"
        + (f"; keys present: {', '.join('`' + k + '`' for k in detail_keys)}" if detail_keys else "")
        + ") and `dismissed_by_user` (detections the user marked as wrong; they are excluded from the score).\n")
    kinds_here: dict[str, int] = defaultdict(int)
    for e in events:
        kinds_here[str(e.get("type"))] += 1
    add("\n**Event types** (every row of events.csv has one; counts in THIS export, glossary of all known types):\n")
    add("| type | rows here | meaning |\n|---|---|---|")
    for k in sorted(set(kinds_here) | {"basket"}):
        add(f"| `{k}` | {kinds_here.get(k, 0)} | {EVENT_TYPE_DOCS.get(k, 'detected game event')} |")
    for k, m in EVENT_TYPE_DOCS.items():
        if k not in kinds_here and k != "basket":
            add(f"| `{k}` | 0 | {m} |")
    add("\n`points` per type: `basket` 2 (1 = free throw, 3 = probable 3-pointer), everything else 0. "
        "`source` per type: `basket` -> `auto`; game events -> `game_state` or `jev`. "
        "events.csv has no `label`/`details` columns (events.json has `details`).\n")
    add("### timeline_marks.csv / timeline.json – the human's notes\n")
    add("| column | meaning |\n|---|---|")
    for c, m in [
        ("t, time, duration_s, t_end", "start (video seconds), length (0 = a single moment) and end"),
        ("track", "the timeline track (a named row of marks, e.g. \"Notes\", \"Highlights\", or the user's own)"),
        ("track_kind", "`user` = written by a person; `auto` = a detected event shown on the timeline"),
        ("title, note", "the mark's title and free-text note"), ("color", "display colour"),
        ("mark_id, event_id", "IDs; `event_id` links auto marks to events.csv"),
        ("team, confidence", "for auto marks: team and detector confidence"),
    ]:
        add(f"| `{c}` | {m} |")
    add("\n### track_summary.csv – one row per track (computed by the exporter)\n")
    add("| column | meaning |\n|---|---|")
    for c, m in [
        ("track_id, team, team_name, role", "as in players.csv"),
        ("team_confidence", "share of the track's detections that voted for its team (0..1)"),
        ("jersey_number, jersey_number_reads", "number read on the shirt (see players.csv) and how many "
                                               "agreeing reads support it (at least 3)"),
        ("first_seen_t, last_seen_t, span_s", "first/last time with a position and the time between them"),
        ("seconds_tracked", f"rows × {dt:.3f} s (time with a position, incl. interpolated)"),
        ("rows, measured_rows, pct_interpolated", "row counts"),
        ("distance_m", f"path length summed over consecutive MEASURED rows at most {MAX_PAIR_GAP_S:g} s apart; "
                       f"steps faster than {MAX_PLAUSIBLE_SPEED:g} m/s are skipped as tracking glitches"),
        ("mean_speed_m_s", "distance_m / time covered by those steps"),
        ("max_speed_m_s", f"highest speed over windows of ≥ {SPEED_WINDOW_S:g} s between measured rows"),
        ("mean_x_m, mean_y_m", "mean position over measured rows"),
        ("glitch_steps_skipped", "number of implausible steps left out"),
    ]:
        add(f"| `{c}` | {m} |")
    add("\n### possession_summary.csv – ball possession\n")
    add(f"Rows `possessed` per team (frames where a player of that team holds the ball, × {dt:.3f} s), then "
        "`possessed_unknown_team`, `estimated` (in flight/loose) and `no_ball_position`. `pct_of_analysed_time` "
        f"is relative to all {n_frames} analysed frames, `pct_of_possessed_time` to frames held by A or B. "
        "`team_possessions` counts runs of consecutive holding by the same team (passes between teammates do "
        f"not break a run; the other team holding the ball or nobody holding it for > {POSSESSION_BREAK_S:g} s "
        "does) – an approximation of the number of possessions.\n")
    from replay.export_docs import csv_column_docs
    for fname, (cols, docs) in csv_column_docs().items():
        if fname not in csv_files_written:
            continue
        add(f"\n### {fname}\n")
        add("| column | meaning |\n|---|---|")
        for c in cols:
            add(f"| `{c}` | {docs.get(c, '(undocumented)')} |")
    add("\nTeam labels `A`/`B` are tied to shirt colour, not home/away. `x`/`x_m`, `y`/`y_m` are court metres "
        "in the camera court system (section 2); they are NOT mirrored to one half.\n")
    add("**Optional columns added after export** (only when the export was made with player names enabled and a "
        "roster exists): `player_name` after `jersey_number` in players.csv and track_summary.csv, and a "
        "`roster.csv` (number, name, position). Without them the files have exactly the columns above.\n")
    add("\nColumns of possession_summary.csv: `category` (`possessed` per team, `possessed_unknown_team`, "
        "`estimated`, `no_ball_position`), `team`, `team_name`, `frames` (analysed frames in the category), "
        "`seconds` (frames x sampling interval, video seconds), `pct_of_analysed_time`, `pct_of_possessed_time` "
        "(both percent 0-100), `team_possessions` (runs of holding, an approximation; NOT the same count as "
        "possessions.csv or stats_team.csv `possessions`) and `avg_possession_s` (seconds per run).\n")
    add("### game.json, replay_2d.html, report.md\n")
    add("- `game.json`: teams and colours, court geometry (`court.lines`, `court.baskets`), sampling, video info, "
        "row counts, the detected score and `coverage` (the processing report: calibration %, people per frame, "
        "ball rates, time ranges where calibration failed or nobody was tracked).")
    if has_html:
        add("- `replay_2d.html`: open in a browser to watch the 2D replay; detected baskets are ticks on the time "
            "slider and a clickable list, with the running detected score. Use it to sanity-check findings.")
    add("- `report.md`: the processing report in readable form.\n")

    add("## 4. Timing and sampling\n")
    add("- Every `t`, `t0`, `t1`, `t_*` column and every `time` string is **video time**: seconds from the start of "
        f"the video. Analysed frames are every {dt:.3f} s ({fps:.2f} per second), so player and ball rows share "
        "the same `t` values.")
    add(f"- The analysed range is {fmt_clock(t0)} – {fmt_clock(t1)}. Video time includes dead balls, time-outs, "
        "replays and breaks.")
    ck = (stats_clock or {}).get("source")
    clock_here = (f" For THIS export: `source = {ck}`, game length {stats_clock.get('game_length_s')} s, played "
                  f"{stats_clock.get('played_s')} s." if ck else
                  " The clock source of this export is not available (no lineup stats), so no game-clock figures "
                  "exist.")
    add("- **Game-clock minutes** exist in exactly one place: `stats_lineups.csv` `minutes` (and `stats.json` -> "
        "`lineups`). They are NOT video time: the clock stops in dead balls, so they are far smaller than the "
        "video time of the same spell. The method is in `game.json` -> `clock.game_clock` and `stats.json` -> "
        "`lineups.clock`: `source = clock` (broadcast score-bar clock read from the video), `live` (live play "
        "= video time minus detected dead balls, scaled to the game length from the scoresheet or the clock "
        "periods) or `video` (plain video time, no better source)." + clock_here)
    add("- Periods: `stats_team.csv` has `period` rows (detected quarter/half breaks) whose `t0`/`t1` are VIDEO "
        "seconds. Other files carry no period or game-clock column; do not compute per-minute rates from video "
        "time. When the source is `live` or `video`, minutes are estimates.")
    add("- To convert a count of rows to seconds multiply by the sampling interval; for velocities use the "
        "difference of `t`, never the row count, because rows can be missing.\n")

    add("### Shot zones and the 3-point margin\n")
    try:
        from replay import stats as stats_mod
        g = stats_mod.court_geometry(d.game_state)
        add(f"Zones (`stats_shots.csv` `zone`, `stats_zones.csv`) are classified from the shot position (x, y) toward "
            f"the attacked basket (`L`/`R`), measured from the attacked baseline / basket centre "
            f"(this Replay's court: {g['length']:g} x {g['width']:g} m, basket {g['basket_x']:g} m from the "
            f"baseline, 3-point radius {g['three_r']:g} m, straight 3-point segments {g['three_side']:g} m from "
            f"the sidelines, free-throw line {g['ft_x']:g} m from the baseline, paint width "
            f"{2 * g['paint_half']:g} m):\n")
        add("- `corner_3`: beyond the 3-point line while level with or behind the point where the arc meets the "
            "straight segments (corner area).")
        add("- `above_break_3`: beyond the 3-point arc (distance to the basket >= radius), further out than that "
            "point.")
        add("- `paint`: inside the 3-point line, no farther from the baseline than the free-throw line and within "
            "half the paint width of the basket axis.")
        add("- `mid_range`: everything else inside the 3-point line.")
        add("- `unknown`: no shot position (e.g. 'made basket only' rows).")
        add(f"- **3-point margin:** a shot closer than **{stats_mod.ZONE_MARGIN_M:g} m** to the 3-point line (either "
            "side) is marked `zone_uncertain = 1`, because position error (about 0.5–1 m) can flip 2 vs 3. Treat 2/3 "
            "splits and FG% by zone as soft; with fewer than ~10 attempts per zone they mean little.")
        add("- The position is the shooter's floor point at the ESTIMATED release time (often the last holder's "
            "position); `value` 3 means the position was beyond the line, not a confirmed 3-pointer.\n")
    except Exception:  # noqa: BLE001
        add("Shot-zone definitions are unavailable in this export.\n")

    add("## 5. Data quality – read before analysing\n")
    src_counts = Counter(r["source"] for r in players)
    add("### Which rows to trust\n")
    add("Every players.csv row says in `source` where its position comes from:\n")
    add("| source | meaning | rows here |\n|---|---|---|")
    for s_, m in [("seen", "measured: the person was detected in this frame"),
                  ("bridged", "filled in by the smoother across a short gap while the spot was visible (a missed "
                              "detection, or the fill is wrong - the least trustworthy fill)"),
                  ("rejected", "a detection existed but was an outlier; the position comes from the motion model"),
                  ("hidden", "filled in while the person was behind a nearer person (plausible)"),
                  ("offscreen", "filled in while the spot was outside the picture or under the score bar"),
                  ("predicted", "an on-court player out of view, moved by a motion prior (rough, uncertainty grows)"),
                  ("unsure", "a 6th person of a team during live play (at most 5 are on court): may be a bench "
                             "player, a duplicate box or a real player; not drawn in the 2D view"),
                  ("clamped", "the measured position was more than 1 m outside the court (court calibration off "
                              "here); moved to the court edge - do not use for distances")]:
        add(f"| `{s_}` | {m} | {src_counts.get(s_, 0)} |")
    add("\nRecommended: positions and distances from `source` in (`seen`, `bridged`, `hidden`); per-player "
        "aggregates by `player_id` (falls back to `track_id` where empty); `track_id` for the strictest "
        "identity (one unbroken track). `id_confidence` < 0.5: treat the identity as a guess.\n")
    add(f"- **interpolated = 1** rows ({_pct(n_interp, len(players))} % here) are filled in by the smoother "
        "across gaps up to 1 s (occlusion, missed detection). They are plausible but NOT measurements.")
    add("- **Ball `estimated`** rows are a straight-ish interpolation between two holders: fine for "
        "\"where roughly\", useless for shot arcs or pass speed. `possessed` rows are only as good as the "
        "holder assignment (nearest player to the ball).")
    add(f"- **calibration**: `keyframe` rows ({calib_counts.get('keyframe', 0)}) are the most accurate; "
        f"`propagated` rows ({calib_counts.get('propagated', 0)}) depend on camera-motion tracking and can drift "
        "by up to a metre or two, mostly far from the camera and near the far sideline. Frames without "
        "calibration have no rows at all.")
    add("- **confidence** is the detector's score; below ~0.4 treat positions with care.")
    add("- **track_id is not an identity.** One real player is often split over several tracks (after "
        "occlusions, leaving the view, or crossing paths) and an ID can occasionally jump to another person. "
        "Never report \"player 7\" as if it were jersey number 7. Per-player totals are per-track totals.")
    add("- **jersey_number** is read automatically from the shirts (text recognition, accepted only after at "
        "least 3 agreeing reads, one number per team at a time). Where it is filled in it names the player the "
        "track mostly follows (a track can still jump to someone else for a moment): combine the tracks of one "
        "team with the same jersey_number for per-player totals. Most "
        "tracks have none (numbers are visible only when a player faces away from or towards the camera); "
        "never guess a number for those.")
    add("- **Coverage limits of a panning camera:** the camera shows only part of the court at a time, so "
        "players outside the view are simply missing (not standing still). Counts per team vary from frame to "
        f"frame; here the average is {num(ppl.get('avg_tracked_per_calibrated_frame'))} tracked people per "
        "calibrated frame versus ~12–13 actually on court. Any per-team total (distance, coverage) is a lower "
        "bound and is biased towards the part of the court the camera follows (usually the ball).")
    ranges = (rep.get("failed_ranges") or {})
    nc = ranges.get("no_calibration") or []
    npl = ranges.get("no_players_tracked") or []
    if nc or npl:
        add(f"- Time ranges with no calibration: {', '.join(f'{fmt_clock(a)}–{fmt_clock(b)}' for a, b in nc) or 'none'}"
            f"; calibrated but nobody tracked: {', '.join(f'{fmt_clock(a)}–{fmt_clock(b)}' for a, b in npl) or 'none'}.")
    for w in (cal.get("warnings") or [])[:5]:
        add(f"- Calibration warning from the processing: {w}")
    add("")

    all_events, events = events, [e for e in events if e.get("type") == "basket"]
    add("## 6. Baskets (scoring) detection\n")
    add(f"- {len(events)} basket(s) detected automatically (`source = auto`), each with a confidence 0..1. The "
        "detector looks at the tracked ball around the two baskets; the evidence it used is in `events.json` → "
        "`details`. Treat events below ~0.5 confidence as uncertain.")
    add(f"- The running score in `events.csv` adds **{POINTS_PER_BASKET} points for every detected basket** "
        "with a known team, except baskets the game-state engine recognised as free throws (1 point) or as "
        "probable 3-pointers (3 points, titled `Made 3?`; the value is estimated from the shooter's position). "
        "It is a "
        "DETECTED score, **not the official score**: baskets can be missed (ball hidden, camera elsewhere, free "
        "throws) or falsely detected (ball passing close to the rim).")
    if events:
        add(f"- Detected final score: **{A} {final['A']} – {final['B']} {B}**. Baskets:")
        for e in events[:60]:
            c = _f(e.get("confidence"))
            add(f"  - {fmt_clock(_f(e.get('t')))} (t = {num(_f(e.get('t')), 1)} s) basket {e.get('basket') or '?'} – "
                f"{e['team_name'] or 'team unknown'} – confidence {num(c, 2)} → {e['score_A']}–{e['score_B']}"
                + (f" – note: {e['note']}" if e["note"] else ""))
        if len(events) > 60:
            add(f"  - … {len(events) - 60} more in events.csv")
    else:
        add("- No baskets were detected in this Replay (or this version of the analysis did not provide basket "
            "events). Do not invent a score.")
    if dismissed:
        add(f"- The user dismissed {len(dismissed)} detection(s) as wrong; they are excluded from the score "
            "and listed under `dismissed_by_user` in events.json.")
    add("")

    _readme_game_state(d, add, all_events)

    add("## 7. Timeline notes written by the user\n")
    if not user_tracks:
        add("No user timeline exists for this Replay. `timeline_marks.csv` contains only the auto-detected "
            "marks (if any).\n")
    else:
        add(f"The user organised {n_marks} mark(s) on {len(user_tracks)} track(s). These are human observations "
            "and are usually the most trustworthy information here – use them as context, as labels for "
            "situations (e.g. \"fast break\", \"zone defence\") and to check your findings. Times are video "
            "seconds.\n")
        shown = 0
        for tr in user_tracks:
            add(f"- **{tr['name']}** ({len(tr['marks'])} mark(s){'' if tr.get('visible', True) else ', hidden in the app'})")
            for m in tr["marks"]:
                if shown >= 80:
                    break
                dur = f" for {m['duration']:.1f} s" if m["duration"] > 0 else ""
                note = " – " + m["note"].replace("\n", " ") if m["note"] else ""
                add(f"  - {fmt_clock(m['t'])}{dur}: {m['title'] or '(no title)'}{note}")
                shown += 1
        if shown >= 80:
            add("  - … more in timeline_marks.csv")
        add("")

    add("## 8. Suggested analyses (with formulas)\n")
    add(f"Notation: Δt = difference of consecutive `t` of the same track (≈ {dt:.3f} s); p = (x, y). Filter "
        "`interpolated == 0` whenever you measure, and `role == player` for team statistics.\n")
    add("1. **Possession / time with the ball per team.** From ball.csv: possession_time(team) = Σ Δt over rows "
        "with `state == possessed` and `owner_team == team`. Share = time(A) / (time(A) + time(B)). Already "
        f"computed in possession_summary.csv: {A} {pA['seconds']:.1f} s, {B} {pB['seconds']:.1f} s. Also "
        "count team possessions (runs) and their mean length.")
    add("2. **Distance covered and speed per player-track.** distance = Σ |p_{i+1} − p_i| over consecutive "
        f"measured rows with Δt ≤ {MAX_PAIR_GAP_S:g} s; speed v_i = |p_{{i+1}} − p_i| / Δt. Discard v > "
        f"{MAX_PLAUSIBLE_SPEED:g} m/s (glitches). Bands: walking < 2 m/s, jogging 2–4, running 4–6, "
        "sprinting > 6 m/s. Team distance per minute = Σ track distances / tracked minutes (it is a lower "
        "bound, see section 5). Ready-made in track_summary.csv"
        + ("; longest tracks here: " + ", ".join(f"track {s['track_id']} ({s['team_name']}, "
                                                    f"{s['distance_m']:.0f} m in {s['seconds_tracked']:.0f} s)"
                                                    for s in top) if top else "") + ".")
    add("3. **Spacing.** Per frame and team (≥ 3 measured players visible): centroid c = mean(p); spread = mean "
        "|p − c|; convex-hull area of the team's positions (m²); mean nearest-teammate distance; mean distance "
        "to the nearest opponent (defensive pressure). Report distributions, not single frames.")
    add("4. **Court coverage heatmaps.** 2D histogram of measured positions per team, e.g. 1 m × 1 m bins on "
        f"{court.LENGTH:g} × {court.WIDTH:g} (x 0..{court.LENGTH:g}, y 0..{court.WIDTH:g}), normalised to seconds "
        "per bin (count × Δt). Compare with/without the "
        f"ball (join ball.csv owner_team on `t`). Mirror x → {court.LENGTH:g} − x for one half if you want both halves "
        "attacking the same direction.")
    add("5. **Shot / basket locations.** For each event, the ball and the scoring team's players shortly "
        "before `t` (e.g. 1–3 s): the last `possessed` ball row of the scoring team before the event gives the "
        "likely shooter position; distance to the basket (section 2) > 6.75 m suggests a 3-point attempt. "
        "State that this is an estimate.")
    add("6. **Pace and transitions.** Possession changes per minute; time from a change of possession to the "
        "ball crossing half court (x = 14); team centroid x velocity dc_x/dt; count of times the ball moves "
        "from one half to the other.")
    add("7. **Fast-break detection.** A possession change followed within ~6 s by the ball (or the ball "
        "holder) moving > 14 m towards the opposite basket, with the attacking team's players outnumbering the "
        "defending team's players in that half (count players with x in the attacked half). Use the speed of "
        "the ball holder (> 5 m/s) as a supporting signal. List candidate times so a human can check them in "
        "replay_2d.html.")
    add("8. **Referee positioning.** For `role == referee` tracks: distance to the ball (ball.csv at the same "
        f"`t`), the share of time on the near vs far side (y < {court.MID_Y:g} / y ≥ {court.MID_Y:g}) and whether they stay behind the "
        "play (trail) or ahead of it (lead) relative to the ball's x direction.")
    add("9. **Link to the human notes.** For each user mark, describe what the data shows in the window "
        "[t − 5 s, t_end + 5 s]: possession, team spacing, speeds, ball path.\n")

    add("## 9. Pitfalls\n")
    for p in [
        "Don't treat interpolated rows (players `interpolated = 1`, ball `estimated`) as measurements.",
        "Positions of `unknown`-team rows are often bench players, coaches, staff or spectators near the court. "
        "Exclude them from team statistics; at most use them as a hint of untracked players.",
        "A track ID is not a player. Never sum per-track numbers into \"per player\" claims; one person can have "
        "several tracks and a track can change person after an occlusion.",
        "Missing rows mean \"not visible / not calibrated\", not \"standing still\" and not \"off court\".",
        "Team totals are lower bounds because a panning camera does not see all 10 players all the time.",
        "The detected score is not the official score; shot values are estimates (free throws 1 when recognised).",
        "Team-to-basket direction switches at half time; use the attack direction of section 2 / 6b.",
        "Players outside the camera view are still on the court (5 per team in live play).",
        "Free throws are dead balls: ball moves between referee and shooter are not passes.",
        "`t` columns are video time and include stoppages; only `stats_lineups.csv` `minutes` is game-clock "
        "time (section 4). Don't mix them, and don't compute per-minute rates from video time.",
        "Positions are the floor point under the feet; jumping players and players at the far sideline are "
        "less accurate (±0.5–1 m is normal, more for `propagated` calibration).",
        "Small samples: this export covers "
        f"{span:.0f} s of video. Don't generalise beyond it.",
    ]:
        add(f"- {p}")
    add("")

    add("## 10. Recommended step-by-step plan\n")
    for i, s in enumerate([
        "Read game.json (teams, sampling, coverage) and this README; note the analysed time range and calibration %.",
        "Load players.csv and ball.csv; check row counts, time range and the share of interpolated/unknown rows.",
        "Read the user's timeline notes (timeline_marks.csv) and the detected events (events.csv).",
        "Build a clean player table: `role == player`, `interpolated == 0`; keep tracks with ≥ 3 s of measured "
        "rows for per-track statistics.",
        "Possession and pace (analyses 1 and 6) from possessions.csv; check the attack direction of section 2.",
        "Movement: distance and speed per track and per team (analysis 2).",
        "Spacing and heatmaps per team, in possession vs defence (analyses 3 and 4).",
        "Scoring: basket locations / likely shooters, runs of baskets (analysis 5).",
        "Fast-break candidates and referee positioning (analyses 7 and 8).",
        "Go through each user note and explain it with the data (analysis 9).",
        "Write the report in the format below; state uncertainty and data limits explicitly.",
    ], 1):
        add(f"{i}. {s}")
    add("")

    add("## 11. Output format for your report\n")
    add("Write Markdown with these sections:\n")
    add(f"1. **Summary** – 5–8 bullet points with the most important findings about {A} vs {B}.")
    add("2. **Data used and its limits** – time range, calibration %, tracked people per frame, what was "
        "excluded (interpolated / unknown rows) and why.")
    add("3. **Possession and pace** – table: team, possession seconds and %, number of possessions, average "
        "possession length, possession changes per minute.")
    add("4. **Scoring** – detected baskets (time, team, basket, confidence), detected score with the "
        "\"not official\" caveat, likely shot locations.")
    add("5. **Movement** – table per team (and top tracks): distance, distance per tracked minute, mean and max "
        "speed, time in speed bands.")
    add("6. **Spacing and court coverage** – spacing metrics per team with/without the ball; heatmap "
        "description (or image if you can plot).")
    add("7. **Transitions and fast breaks** – candidate times (mm:ss) with a one-line description each.")
    add("8. **Referees** – positioning summary.")
    add("9. **User notes explained** – one entry per timeline note: time, note, what the data shows.")
    add("10. **Confidence and open questions** – what a human should verify in replay_2d.html (give times).\n")
    add("Always give times as mm:ss of video time and as seconds, and give the formula or filter you used for "
        "each number.\n")
    return "\n".join(L) + "\n"
