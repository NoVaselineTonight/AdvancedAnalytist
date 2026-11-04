"""Shot chart, team / possession statistics and lineup statistics (deterministic, no Qt, no video).

Everything here is ESTIMATED FROM DETECTED EVENTS: the inputs are the automatic game state
(``results/game_state.json``: possessions, shots, free throws, turnovers, rebounds, fast breaks, breaks,
attack direction; its ``ball_events`` layer adds shot locations / outcomes / shooters and steals), the detected
made baskets (``results/replay_data.json`` events, annotated by the game state as free throw / 2 / 3?), the
user's timeline edits (dismissed auto events are left out) and, when present, the lineup solver's slots
(``results/lineup.json``, replay/lineup.py). Nothing reads the scoreboard: if a ground truth score is supplied
(``score_truth.json`` in the Replay folder or its results folder) the detected points are reconciled with it.

Definitions (also written into the output ``notes``):

* **Points** = sum of the detected made baskets (field goals 2 or 3 points as guessed from the shooter's
  distance by the game state, free throws 1). Baskets whose team is unknown are reported as "unattributed".
* **Possessions** = the game state's possessions merged while the same team keeps the ball (an offensive
  rebound continues a possession). ``possessions_est`` = FGA - ORB + TOV + 0.44 x FTA (box-score estimate) for
  comparison. **PPP** = points / possessions.
* **Turnover rate** = turnovers / possessions. A turnover is a game-state ``turnover`` (team = the team that
  lost the ball) or a ``steal`` (counted against the team it was stolen from).
* **Offensive rebound rate** = ORB / (ORB + opponent DRB).
* **Fast-break points** = field-goal points scored within a detected fast break (its window + 4 s).
* **Time of possession** = summed possession durations (video seconds; dead balls excluded).
* **Shot zones** (court metres, attacked basket): paint (inside the key), mid-range (other 2-point area),
  corner 3 (beyond the straight 3-point segment), above-the-break 3. Shots within 0.5 m of the 3-point line are
  flagged ``zone_uncertain`` (foot-point accuracy).
* **Periods** = video time between detected quarter / half-time breaks (the game clock is not read).
* **Lineups** = 5-player combinations per team from the lineup solver's slots; a player is a jersey number
  read on a slot piece (pieces without a read number inherit the slot's previous number). Minutes are video
  time excluding detected breaks; ratings are points per 100 possessions.

Public API: :func:`load_inputs`, :func:`compute`, :func:`for_project`, :func:`write_exports`,
:func:`export_into` (simple data export hook), :func:`filter_shots`, :func:`zone_table`, :func:`classify_zone`.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

FORMAT = "advanced-analytist/stats@1"
TEAMS = ("A", "B")
MIN_EVENT_CONF = 0.3  # game-state events below this confidence are not counted (same as the timeline)
ZONE_MARGIN_M = 0.5  # shots this close to the 3-point line: 2 or 3 is uncertain
FAST_BREAK_PAD_S = 4.0
BRIEF_SPELL_S = 3.0  # other-team ball control shorter than this between two possessions of one team, unconfirmed
#                      by a shot / basket, is treated as a holder mix-up (not two turnovers + two possessions)
ZONES = ("paint", "mid_range", "corner_3", "above_break_3")
ZONE_TITLES = {"paint": "Paint", "mid_range": "Mid-range", "corner_3": "Corner 3", "above_break_3": "Above-break 3",
               "unknown": "Location unknown"}
BREAK_KINDS = ("quarter_break", "half_time")
TRUTH_FILE = "score_truth.json"

STAT_ROWS = [  # (key, title, kind) - kind: int / pct / float / time
    ("points", "Points", "int"), ("pts_2", "  2-point FG points", "int"), ("pts_3", "  3-point FG points", "int"),
    ("pts_ft", "  Free-throw points", "int"), ("fgm", "FG made", "int"), ("fga", "FG attempts", "int"),
    ("fg_pct", "FG %", "pct"), ("fg3m", "3-pt made", "int"), ("fg3a", "3-pt attempts", "int"),
    ("ftm", "FT made", "int"), ("fta", "FT attempts", "int"), ("possessions", "Possessions", "int"),
    ("possessions_est", "Possessions (box-score estimate)", "float"), ("ppp", "Points per possession", "float"),
    ("turnovers", "Turnovers", "int"), ("tov_rate", "Turnover rate", "pct"), ("steals", "Steals", "int"),
    ("orb", "Off. rebounds", "int"), ("drb", "Def. rebounds", "int"), ("orb_rate", "Off. rebound rate", "pct"),
    ("fast_breaks", "Fast breaks", "int"), ("fast_break_points", "Fast-break points", "int"),
    ("time_of_possession_s", "Time of possession", "time"), ("top_pct", "Possession share", "pct"),
]
# which detected items explain a stat row (clicking the row lists them)
ROW_ITEMS = {"points": "baskets", "pts_2": "baskets_2", "pts_3": "baskets_3", "pts_ft": "baskets_ft",
             "fgm": "fg_made", "fga": "fg_attempts", "fg_pct": "fg_attempts", "fg3m": "fg3_made",
             "fg3a": "fg3_attempts", "ftm": "ft_made", "fta": "ft_attempts", "possessions": "possessions",
             "possessions_est": "possessions", "ppp": "possessions", "turnovers": "turnovers",
             "tov_rate": "turnovers", "steals": "steals", "orb": "orb", "drb": "drb", "orb_rate": "orb",
             "fast_breaks": "fast_breaks", "fast_break_points": "fast_break_baskets",
             "time_of_possession_s": "possessions", "top_pct": "possessions"}


# ====================================================================================== helpers
def _f(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _read(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _other(team: str | None) -> str | None:
    return {"A": "B", "B": "A"}.get(team or "")


def _ratio(a: float, b: float) -> float | None:
    return round(a / b, 3) if b else None


def fmt_clock(t: float | None) -> str:
    if t is None:
        return ""
    t = max(0.0, float(t))
    h, rem = divmod(int(t), 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _item(t: float | None, title: str, t1: float | None = None, **extra: Any) -> dict:
    out = {"t": None if t is None else round(float(t), 2), "title": title}
    if t1 is not None:
        out["t1"] = round(float(t1), 2)
    out.update({k: v for k, v in extra.items() if v is not None})
    return out


# ====================================================================================== court geometry
def court_geometry(gs: dict | None = None) -> dict:
    """Court size and markings for the zones / drawing: size + baskets from the game state (the Replay's own
    court), marking standard from replay.court read at call time."""
    from replay import court
    c = (gs or {}).get("court") or {}
    L = _f(c.get("length")) or court.LENGTH
    W = _f(c.get("width")) or court.WIDTH
    b = c.get("baskets") or {}
    bl = b.get("L") or [court.BASKET_X, W / 2]
    basket_x = _f(bl[0]) or court.BASKET_X
    return {"length": L, "width": W, "basket_x": basket_x, "basket_y": _f(bl[1]) or W / 2,
            "three_r": _f(c.get("three_point_radius_m")) or court.THREE_R, "three_side": court.THREE_SIDE_Y,
            "paint_half": court.PAINT_HALF_WIDTH, "ft_x": court.FT_LINE_X, "circle_r": court.CENTER_CIRCLE_R,
            "ft_circle_r": court.FT_CIRCLE_R, "no_charge_r": court.NO_CHARGE_R}


def court_lines(geo: dict, n_arc: int = 48) -> list[list[tuple[float, float]]]:
    """Court markings (metres) for ``geo`` (same drawing as replay.court.court_polylines, any size)."""
    L, W = geo["length"], geo["width"]
    my = W / 2
    bx, R, side = geo["basket_x"], geo["three_r"], geo["three_side"]

    def arc(cx, cy, r, a0, a1, n):
        return [(cx + r * math.cos(a0 + (a1 - a0) * k / (n - 1)), cy + r * math.sin(a0 + (a1 - a0) * k / (n - 1)))
                for k in range(n)]

    lines = [[(0, 0), (L, 0), (L, W), (0, W), (0, 0)], [(L / 2, 0), (L / 2, W)],
             arc(L / 2, my, geo["circle_r"], 0, 2 * math.pi, n_arc)]
    a3 = math.asin(min(1.0, (my - side) / R))
    ph, fx = geo["paint_half"], geo["ft_x"]
    half = [[(0, my - ph), (fx, my - ph), (fx, my + ph), (0, my + ph)],
            arc(fx, my, geo["ft_circle_r"], 0, 2 * math.pi, n_arc),
            [(0, side)] + arc(bx, my, R, -a3, a3, n_arc) + [(0, W - side)],
            arc(bx, my, geo["no_charge_r"], -math.pi / 2, math.pi / 2, n_arc // 2)]
    for ln in half:
        lines.append(ln)
        lines.append([(L - x, y) for x, y in ln])
    return lines


def classify_zone(x: float | None, y: float | None, side: str | None, geo: dict) -> dict:
    """Zone of a shot taken at court position (x, y) towards basket ``side`` ("L"/"R"; None = nearest).
    Returns {"zone", "three" (bool), "uncertain" (bool: within ZONE_MARGIN_M of the 3-point line),
    "distance_m"}; zone "unknown" when the position is unknown."""
    if x is None or y is None:
        return {"zone": "unknown", "three": None, "uncertain": True, "distance_m": None}
    L, W = geo["length"], geo["width"]
    if side not in ("L", "R"):
        side = "L" if x < L / 2 else "R"
    along = x if side == "L" else L - x  # depth from the attacked baseline
    by = geo.get("basket_y") or W / 2
    dx, dy = along - geo["basket_x"], y - by
    R = geo["three_r"]
    lat = min(R, W / 2 - geo["three_side"])  # lateral distance of the straight 3-point segments
    meet = math.sqrt(max(0.0, R * R - lat * lat))  # where the straight segments meet the arc (along the court)
    d = math.hypot(dx, dy)
    if dx <= meet:  # alongside the straight 3-point segments (corners)
        three = abs(dy) >= lat
        margin = abs(dy) - lat if three else min(lat - abs(dy), R - d)
    else:
        three, margin = d >= R, d - R
    if three:
        zone = "corner_3" if dx <= meet else "above_break_3"
    elif along <= geo["ft_x"] and abs(dy) <= geo["paint_half"]:
        zone = "paint"
    else:
        zone = "mid_range"
    return {"zone": zone, "three": bool(three), "uncertain": abs(margin) < ZONE_MARGIN_M, "distance_m": round(d, 2)}


# ====================================================================================== inputs
def _overrides(timeline: Any) -> dict[str, dict]:
    if isinstance(timeline, dict) and isinstance(timeline.get("auto_overrides"), dict):
        return {str(k): v for k, v in timeline["auto_overrides"].items() if isinstance(v, dict)}
    return {}


def load_inputs(project_root: Path | str, replay: dict | None = None, game_state: dict | None = None,
                max_build_frames: int | None = None) -> dict:
    """Read everything :func:`compute` needs from a Replay folder (``<root>/results/...``). ``replay`` /
    ``game_state`` may be passed when the caller already has them loaded. The game state is (re)built only
    when it is missing or stale and the Replay is short enough (``max_build_frames``, None = always)."""
    root = Path(project_root)
    res = root / "results"
    if replay is None:
        replay = _read(res / "replay_data.json")
    replay = replay if isinstance(replay, dict) else {}
    if game_state is None:
        from replay import game_state as gs_mod
        try:
            game_state = gs_mod.load_or_build(res, replay if replay.get("t") else None, max_build_frames)
        except Exception as e:  # noqa: BLE001 - stats must work with what is there
            print("stats: game state unavailable:", type(e).__name__, e)
            game_state = _read(res / "game_state.json")
    gs = game_state if isinstance(game_state, dict) else {}
    names = {k: str((gs.get("teams") or {}).get(k) or k) for k in TEAMS}
    colors: dict[str, str | None] = {k: None for k in TEAMS}
    colour_names: dict[str, str] = {}
    for k, v in ((replay.get("game") or {}).get("teams") or {}).items():
        if k in TEAMS and isinstance(v, dict):
            names[k] = str(v.get("name") or names[k])
            colors[k] = v.get("primary") or colors[k]
            if v.get("colour_name"):
                colour_names[k] = str(v["colour_name"])
    info = _read(root / "project.json") or {}
    disp = ((info.get("settings") or {}).get("display_colors") or {}) if isinstance(info, dict) else {}
    colors.update({k: disp[k] for k in TEAMS if disp.get(k)})
    truth = _read(root / TRUTH_FILE)
    if not isinstance(truth, dict):
        truth = _read(res / TRUTH_FILE)
    lineup_doc = _read(res / "lineup.json") if lineup_is_current(res, replay) else None
    return {"game_state": gs, "replay_events": [e for e in (replay.get("events") or []) if isinstance(e, dict)],
            "overrides": _overrides(_read(root / "timeline.json")),
            "lineup": lineup_doc if isinstance(lineup_doc, dict) else None,
            "truth": truth if isinstance(truth, dict) else None, "names": names, "colors": colors,
            "colour_names": colour_names, "game_clock": _game_clock(root, gs)}


def _game_clock(root: Path, gs: dict):
    """minutes.GameClock from the score-bar clock (work/scorebar.json) + scoresheet game length (if present)."""
    from replay import minutes as minutes_mod
    try:
        from replay import scorebar
        sb = scorebar.load_result(root / "results" / "work")
    except Exception:  # noqa: BLE001
        sb = None
    sheet = _read(root / "scoresheet.json")
    return minutes_mod.GameClock(gs, sb if isinstance(sb, dict) else None, sheet if isinstance(sheet, dict) else None)


def lineup_is_current(res: Path, replay: dict) -> bool:
    """results/lineup.json belongs to the current build: the lineup solver (replay/lineup.py, off by default) marks
    its on-court slots in replay_data track_info ("slot"); builds from before that marker count when lineup.json
    was written in the same pass 2 (at most an hour before replay_data.json). A lineup.json left over from an
    earlier build with the solver on is ignored."""
    f = res / "lineup.json"
    if not f.exists():
        return False
    info = replay.get("track_info") or {}
    if any(isinstance(v, dict) and v.get("slot") for v in info.values()):
        return True
    try:
        dt = (res / "replay_data.json").stat().st_mtime - f.stat().st_mtime
    except OSError:
        return True  # no replay data to compare with: trust the file
    return -60.0 <= dt <= 3600.0


# ====================================================================================== periods
def periods(gs: dict) -> list[dict]:
    """Video-time periods between detected quarter / half-time breaks."""
    t0, t1 = (gs.get("time_range_s") or [0.0, 0.0])[:2]
    t0, t1 = _f(t0) or 0.0, _f(t1) or 0.0
    brks = sorted((b for b in gs.get("breaks") or [] if b.get("kind") in BREAK_KINDS), key=lambda b: b["t0"])
    out, a = [], t0
    for b in brks:
        if _f(b.get("t0")) is None or b["t0"] <= a:
            a = max(a, _f(b.get("t1")) or a)
            continue
        out.append((a, float(b["t0"])))
        a = float(b.get("t1") or b["t0"])
    if t1 > a or not out:
        out.append((a, max(a, t1)))
    single = len(out) == 1
    return [{"index": i, "label": "Whole analysed range" if single else f"Period {i + 1}",
             "t0": round(p0, 2), "t1": round(p1, 2)} for i, (p0, p1) in enumerate(out)]


# ====================================================================================== core
def _baskets(replay_events: Iterable[dict], gs: dict, overrides: dict[str, dict]) -> tuple[list[dict], list[dict]]:
    """(made baskets counted, dismissed baskets). Each: id, t, team (A/B/None), points, kind, side, confidence."""
    ann = gs.get("basket_annotations") or {}
    kept, dismissed = [], []
    for e in replay_events:
        if e.get("type") != "basket" or e.get("id") is None:
            continue
        eid = str(e["id"])
        a = ann.get(eid) or {}
        ov = overrides.get(eid) or {}
        team = a.get("team") if a.get("team") in TEAMS else e.get("team") if e.get("team") in TEAMS else None
        if ov.get("team") in TEAMS:
            team = ov["team"]
        kind = a.get("kind") or "field_goal"
        pts = int(_f(a.get("points_guess")) or (1 if kind == "free_throw" else 2))
        b = {"id": eid, "t": _f(e.get("t")), "team": team, "points": pts, "kind": kind, "side": e.get("basket"),
             "confidence": _f(e.get("confidence")), "label": a.get("label") or "Made basket",
             "annotated": bool(a), "team_source": a.get("team_source") or ("detector" if team else None)}
        (dismissed if ov.get("dismissed") else kept).append(b)
    # attached scoresheet (ball_events.reconcile_with_sheet): baskets that are not on the sheet do not count (listed as
    # hints), sheet events the video missed count with their estimated time -> totals equal the sheet
    if isinstance(gs.get("sheet_alignment"), dict):
        drop = [b for b in kept if (ann.get(b["id"]) or {}).get("kind") == "not_on_sheet"]
        kept = [b for b in kept if b not in drop]
        dismissed += [{**b, "not_on_sheet": True} for b in drop]
        for m in gs["sheet_alignment"].get("missing") or []:
            kind = "free_throw" if m.get("kind") == "ft" else "field_goal"
            num = m.get("scorer_number")
            kept.append({"id": f"sheet-{m.get('seq')}", "t": _f(m.get("t_estimated")), "team": m.get("team"),
                         "points": int(m.get("points") or 0), "kind": kind, "side": None, "confidence": None,
                         "label": m.get("label") or (f"From the scoresheet (#{num}, time estimated)" if num else
                                                     "From the scoresheet (time estimated)"), "annotated": True,
                         "team_source": gs["sheet_alignment"].get("source") or "scoresheet",
                         "from_sheet": True, "scorer_number": num})
    # "makes" after which the play stayed at the basket (ball_events.check_makes): probably a missed shot / rebound
    doubt = [b for b in kept if (ann.get(b["id"]) or {}).get("counted") is False]
    kept = [b for b in kept if b not in doubt]
    dismissed += [{**b, "doubtful": True} for b in doubt]
    kept.sort(key=lambda b: b["t"] or 0.0)
    return kept, dismissed


def _events(gs: dict, overrides: dict[str, dict], types: Sequence[str]) -> list[dict]:
    out = []
    for e in gs.get("events") or []:
        if e.get("type") not in types or (_f(e.get("confidence")) or 0.0) < MIN_EVENT_CONF:
            continue
        if (overrides.get(str(e.get("id"))) or {}).get("dismissed"):
            continue
        out.append(e)
    return out


def _shots(gs: dict, baskets: list[dict], dismissed: list[dict], geo: dict) -> list[dict]:
    """Field-goal attempts for the shot chart: ball-event shots (location, shooter, outcome) or the game-state
    attempts, plus detected made field goals that no attempt explains (location unknown)."""
    by_id = {b["id"]: b for b in baskets}
    gone = {b["id"] for b in dismissed}
    off_sheet = {b["id"] for b in dismissed if b.get("not_on_sheet")}
    gs_shots = [s for s in gs.get("shots") or [] if not s.get("free_throw")]
    be = gs.get("ball_events") if isinstance(gs.get("ball_events"), dict) else {}
    pos_of_event = {}
    for e in gs.get("events") or []:
        if e.get("type") == "shot" and (e.get("details") or {}).get("position"):
            pos_of_event[round(_f(e.get("t")) or 0.0, 1)] = e["details"]["position"]

    def gs_match(t: float) -> dict | None:
        return min((s for s in gs_shots if abs((_f(s.get("t_release")) or -99) - t) <= 0.2),
                   key=lambda s: abs(s["t_release"] - t), default=None)

    rows: list[dict] = []
    src = be.get("shots") if be.get("shots") else None
    for s in (src if src is not None else gs_shots):
        t = _f(s.get("t_release"))
        if t is None:
            continue
        g = gs_match(t) if src is not None else s
        bid = (g or {}).get("basket_event")
        pos = s.get("position") if src is not None else pos_of_event.get(round(t, 1))
        if src is not None:
            outcome = s.get("outcome") or "unknown"
        else:
            outcome = "made" if s.get("made") else "missed" if s.get("missed") else "unknown"
        if bid in off_sheet:
            continue  # a "made basket" the attached scoresheet does not have: not shown (listed as a hint)
        if bid in gone:
            outcome, bid = "missed", None  # the user dismissed that basket
        if outcome == "made" and bid and bid not in by_id:
            bid = None
        shooter = s.get("shooter") if src is not None else s.get("shooter_track")
        name = s.get("shooter_name") if src is not None else None
        x, y = (pos[0], pos[1]) if isinstance(pos, (list, tuple)) and len(pos) >= 2 else (None, None)
        z = classify_zone(_f(x), _f(y), s.get("basket"), geo)
        b = by_id.get(bid) if bid else None
        team = s.get("team") if s.get("team") in TEAMS else None
        if b and b["team"] in TEAMS:
            team = b["team"]
        value = 3 if z["three"] else 2 if z["three"] is False else (
            3 if s.get("value_guess") == "3" else 2 if s.get("value_guess") == "2" else None)
        if b and outcome == "made" and b["kind"] == "field_goal":
            value = b["points"]  # the counted points decide (same as the timeline / events.csv score)
        rows.append({"t": round(t, 2), "t_rim": _f(s.get("t_rim")), "team": team, "basket": s.get("basket"),
                     "x": _f(x), "y": _f(y), "distance_m": _f(s.get("distance_m")) or z["distance_m"],
                     "zone": z["zone"], "zone_uncertain": z["uncertain"], "value": value, "outcome": outcome,
                     "points": b["points"] if b and outcome == "made" else 0, "basket_event": bid,
                     "shooter": None if shooter is None else str(shooter),
                     "shooter_name": name if name and str(name).startswith("#") else None,
                     "confidence": _f(s.get("confidence")), "source": "ball_events" if src is not None else "game_state"})
    used = {r["basket_event"] for r in rows if r["basket_event"]}
    for b in baskets:
        if b["kind"] != "field_goal" or b["id"] in used:
            continue
        rows.append({"t": round(b["t"] or 0.0, 2), "t_rim": b["t"], "team": b["team"], "basket": b["side"], "x": None,
                     "y": None, "distance_m": None, "zone": "unknown", "zone_uncertain": True,
                     "value": b["points"], "outcome": "made", "points": b["points"], "basket_event": b["id"],
                     "shooter": None, "shooter_name": None, "confidence": b["confidence"],
                     "source": "made basket only (attempt not seen)"})
    rows.sort(key=lambda r: r["t"])
    for k, r in enumerate(rows):
        r["id"] = f"s{k}"
        r["title"] = shot_title(r)
    return rows


def shot_title(s: dict) -> str:
    out = {"made": "Made", "missed": "Missed"}.get(s.get("outcome"), "Shot (result unknown)")
    v = s.get("value")
    t = f"{out} {v}" if v and s.get("outcome") in ("made", "missed") else out
    t += f" – {ZONE_TITLES.get(s.get('zone'), s.get('zone'))}"
    if s.get("zone_uncertain") and s.get("zone") != "unknown":
        t += "?"
    if s.get("distance_m") is not None:
        t += f", {float(s['distance_m']):.1f} m"
    if s.get("shooter_name"):
        t += f" by {s['shooter_name']}"
    return t


def merge_possessions(poss: Sequence[dict], brief_s: float = 0.0,
                      evidence: Sequence[tuple[float, str | None]] = (), absorbed: list | None = None) -> list[dict]:
    """Game-state possessions merged while the same team keeps the ball (offensive rebounds, short dead balls).

    With ``brief_s`` > 0, a spell of the other team shorter than that between two possessions of one team
    (X, brief Y, X) is absorbed into X when nothing confirms it: no shot / made basket by Y in it (``evidence`` =
    [(t, team)]) and it did not start after free throws. Such spells are mostly the holder being mislabelled for a
    moment (a defender next to the ball handler, a track that swapped teams); counting them doubles the
    possessions and adds two turnovers. Absorbed spells are appended to ``absorbed`` as (t0, t1, team)."""
    out = _merge_same(poss)
    if brief_s <= 0:
        return out
    changed = True
    while changed:
        changed = False
        for i in range(1, len(out) - 1):
            a, b, c = out[i - 1], out[i], out[i + 1]
            if a["team"] != c["team"] or b["duration_s"] >= brief_s or b.get("start") == "after free throws":
                continue
            if a.get("outcome") in ENDS_POSSESSION or b.get("outcome") in ENDS_POSSESSION:
                continue  # possession model v6: a made basket / turnover always ends the possession
            if "outcome" in b and b["team"] in TEAMS and (_f(b.get("confidence")) or 0.0) >= 0.5:
                continue  # v6: a confident spell of the other team is real (the vote already dropped mix-ups)
            if any(team in (b["team"], None) and b["t0"] - 0.5 <= t <= b["t1"] + 1.5 for t, team in evidence):
                continue
            if absorbed is not None:
                absorbed.append((b["t0"], b["t1"], b["team"]))
            _join(a, c)
            a["absorbed"] = a.get("absorbed", 0) + 1
            del out[i:i + 2]
            changed = True
            break
    return out


ENDS_POSSESSION = ("made", "turnover", "ft")  # v6 outcomes after which the same team's next spell is a NEW possession
V6_KEYS = ("outcome", "points", "off_rebounds", "shots", "evidence", "kind", "seen_s")


def _join(q: dict, p: dict) -> None:
    q["t1"] = max(q["t1"], float(p["t1"]))
    q["duration_s"] = round(q["duration_s"] + float(p.get("duration_s") or 0.0), 2)
    q["parts"] += int(p.get("parts") or 1)
    q["fast_break"] = q["fast_break"] or bool(p.get("fast_break"))
    if "outcome" in p:
        q["outcome"] = p.get("outcome")
        q["points"] = int(q.get("points") or 0) + int(p.get("points") or 0)
        q["off_rebounds"] = int(q.get("off_rebounds") or 0) + int(p.get("off_rebounds") or 0) + \
            (1 if p.get("start") == "after shot" else 0)
        q["shots"] = list(q.get("shots") or []) + list(p.get("shots") or [])


def _merge_same(poss: Sequence[dict]) -> list[dict]:
    """Consecutive spells of one team joined into one possession. With the possession model v6 (``outcome`` in the
    possession) a spell after a made basket / turnover / free throws of the same team is a new possession, and
    "unclear" spells (team None) are kept as rows of their own."""
    out: list[dict] = []
    for p in sorted((p for p in poss if p.get("team") in TEAMS or ("outcome" in p and p.get("team") is None)),
                    key=lambda p: p["t0"]):
        if out and out[-1]["team"] == p["team"] and p["team"] in TEAMS and out[-1].get("outcome") not in ENDS_POSSESSION:
            _join(out[-1], p)
            continue
        q = {"team": p["team"], "t0": float(p["t0"]), "t1": float(p["t1"]),
             "duration_s": float(p.get("duration_s") or 0.0), "parts": 1,
             "fast_break": bool(p.get("fast_break")), "start": p.get("start"),
             "confidence": _f(p.get("confidence"))}
        if "outcome" in p:
            q.update({k: p.get(k) for k in V6_KEYS})
            q["points"] = int(p.get("points") or 0)
            q["off_rebounds"] = int(p.get("off_rebounds") or 0)
        out.append(q)
    return out


def _team_block(team: str, t0: float, t1: float, d: dict) -> dict:
    """One team's stats inside [t0, t1) (+ item lists explaining every number)."""
    def inside(t):
        return t is not None and t0 <= t < t1

    opp = _other(team)
    bk = [b for b in d["baskets"] if b["team"] == team and inside(b["t"])]
    fg = [s for s in d["shots"] if s["team"] == team and inside(s["t"])]
    fg_made = [s for s in fg if s["outcome"] == "made"]
    fg3 = [s for s in fg if s["value"] == 3]
    ft = [e for e in d["ft_events"] if e.get("team") == team and inside(_f(e.get("t")))]
    ft_made = [e for e in ft if (e.get("details") or {}).get("made")]
    poss = [p for p in d["possessions"] if p["team"] == team and inside(p["t0"])]
    poss_opp = [p for p in d["possessions"] if p["team"] == opp and inside(p["t0"])]
    tov = [x for x in d["turnovers"] if x["team"] == team and inside(x["t"])]
    stl = [x for x in d["turnovers"] if x["team"] == opp and x["kind"] == "steal" and inside(x["t"])]
    orb = [e for e in d["orb_events"] if e.get("team") == team and inside(_f(e.get("t")))]
    drb = [e for e in d["drb_events"] if e.get("team") == team and inside(_f(e.get("t")))]
    opp_drb = [e for e in d["drb_events"] if e.get("team") == opp and inside(_f(e.get("t")))]
    fbs = [e for e in d["fb_events"] if e.get("team") == team and inside(_f(e.get("t")))]
    fb_win = [(float(e["t"]), float(e["t"]) + float(e.get("duration") or 0.0) + FAST_BREAK_PAD_S) for e in fbs]
    fb_win += [(p["t0"], p["t1"] + FAST_BREAK_PAD_S) for p in poss if p["fast_break"]]
    fb_bk = [b for b in bk if b["kind"] == "field_goal" and any(a <= (b["t"] or -1) <= z for a, z in fb_win)]
    points = sum(b["points"] for b in bk)
    pts_ft = sum(b["points"] for b in bk if b["kind"] == "free_throw")
    pts_3 = sum(b["points"] for b in bk if b["kind"] == "field_goal" and b["points"] == 3)
    top = sum(p["duration_s"] for p in poss)
    top_opp = sum(p["duration_s"] for p in poss_opp)
    n_poss = len(poss)
    est = len(fg) - len(orb) + len(tov) + 0.44 * len(ft)
    nm = d["names"]
    stats = {
        "points": points, "pts_2": points - pts_ft - pts_3, "pts_3": pts_3, "pts_ft": pts_ft,
        "fgm": len(fg_made), "fga": len(fg), "fg_pct": _ratio(len(fg_made), len(fg)),
        "fg3m": sum(1 for s in fg3 if s["outcome"] == "made"), "fg3a": len(fg3),
        "ftm": len(ft_made), "fta": len(ft), "ft_pct": _ratio(len(ft_made), len(ft)),
        "possessions": n_poss, "possessions_est": round(est, 1), "ppp": _ratio(points, n_poss),
        "turnovers": len(tov), "tov_rate": _ratio(len(tov), n_poss), "steals": len(stl),
        "orb": len(orb), "drb": len(drb), "orb_rate": _ratio(len(orb), len(orb) + len(opp_drb)),
        "fast_breaks": len(fbs), "fast_break_points": sum(b["points"] for b in fb_bk),
        "time_of_possession_s": round(top, 1), "top_pct": _ratio(top, top + top_opp),
    }

    def bitems(bs):
        return [_item(b["t"], f"{b['label']} ({b['points']} pt) – {nm.get(team, team)}", id=b["id"]) for b in bs]

    def eitems(es):
        return [_item(_f(e.get("t")), str(e.get("title") or e.get("type")), id=e.get("id"),
                      t1=(_f(e.get("t")) or 0.0) + (_f(e.get("duration")) or 0.0) if e.get("duration") else None)
                for e in es]

    stats["items"] = {
        "baskets": bitems(bk), "baskets_ft": bitems([b for b in bk if b["kind"] == "free_throw"]),
        "baskets_3": bitems([b for b in bk if b["kind"] == "field_goal" and b["points"] == 3]),
        "baskets_2": bitems([b for b in bk if b["kind"] == "field_goal" and b["points"] != 3]),
        "fg_made": [_item(s["t"], s["title"], id=s["id"]) for s in fg_made],
        "fg_attempts": [_item(s["t"], s["title"], id=s["id"]) for s in fg],
        "fg3_made": [_item(s["t"], s["title"], id=s["id"]) for s in fg3 if s["outcome"] == "made"],
        "fg3_attempts": [_item(s["t"], s["title"], id=s["id"]) for s in fg3],
        "ft_made": eitems(ft_made), "ft_attempts": eitems(ft),
        "possessions": [_item(p["t0"], f"Possession {fmt_clock(p['t0'])}–{fmt_clock(p['t1'])} "
                                       f"({p['duration_s']:.0f} s{', fast break' if p['fast_break'] else ''})",
                              t1=p["t1"]) for p in poss],
        "turnovers": [_item(x["t"], x["title"], id=x["id"]) for x in tov],
        "steals": [_item(x["t"], x["title"], id=x["id"]) for x in stl],
        "orb": eitems(orb), "drb": eitems(drb), "fast_breaks": eitems(fbs), "fast_break_baskets": bitems(fb_bk),
    }
    return stats


def _turnovers(gs: dict, overrides: dict[str, dict]) -> list[dict]:
    """Turnovers (team = the team that lost the ball); steals counted against the team they were stolen from.
    Two reports of one loss (rules + ball events) within 1.5 s count once."""
    out: list[dict] = []
    for e in sorted(_events(gs, overrides, ("turnover", "steal")), key=lambda e: _f(e.get("t")) or 0.0):
        t = _f(e.get("t"))
        if e["type"] == "steal":
            loser = (e.get("details") or {}).get("lost_by") or _other(e.get("team"))
        else:
            loser = e.get("team")
        if loser not in TEAMS or t is None:
            continue
        if any(x["team"] == loser and abs(x["t"] - t) <= 1.5 for x in out):
            continue
        out.append({"t": t, "team": loser, "kind": e["type"], "id": e.get("id"),
                    "title": str(e.get("title") or e["type"])})
    return out


AFTER_BASKET_S = 8.0  # a "turnover" this soon after the same team's shot / made basket is the undetected
#                       rebound / the inbound after the score


def _clean_turnovers(tov: list[dict], absorbed: Sequence[tuple], evidence: Sequence[tuple]) -> list[dict]:
    """Drop turnovers that are artefacts: around absorbed holder mix-ups (merge_possessions), or a change of
    possession right after the same team's shot / made basket (a missed rebound or the inbound after a score
    that the game state could not explain)."""
    out = []
    for x in tov:
        if any(a0 - 1.5 <= x["t"] <= a1 + 1.5 for a0, a1, _tm in absorbed):
            continue
        if any(tm == x["team"] and 0.0 <= x["t"] - t <= AFTER_BASKET_S for t, tm in evidence):
            continue
        out.append(x)
    return out


def compute(gs: dict | None, replay_events: Iterable[dict] | None = None, overrides: dict | None = None,
            lineup_doc: dict | None = None, truth: dict | None = None, names: dict | None = None,
            colors: dict | None = None, colour_names: dict | None = None, game_clock=None) -> dict:
    """All statistics of one Replay (see module docstring). Never needs Qt or the video. ``game_clock``:
    minutes.GameClock (score-bar clock / scoresheet game length); default = live-play estimate from ``gs``."""
    gs = gs or {}
    overrides = overrides or {}
    names = {k: str((names or {}).get(k) or (gs.get("teams") or {}).get(k) or k) for k in TEAMS}
    geo = court_geometry(gs)
    baskets, dismissed = _baskets(replay_events or [], gs, overrides)
    shots = _shots(gs, baskets, dismissed, geo)
    evidence = [(s["t"], s["team"]) for s in shots] + [(b["t"], b["team"]) for b in baskets if b["t"] is not None]
    absorbed: list[tuple[float, float, str]] = []
    d = {"baskets": baskets, "shots": shots, "names": names,
         "ft_events": _events(gs, overrides, ("free_throw",)),
         "possessions": merge_possessions(gs.get("possessions") or [], BRIEF_SPELL_S, evidence, absorbed),
         "turnovers": _clean_turnovers(_turnovers(gs, overrides), absorbed, evidence),
         "orb_events": _events(gs, overrides, ("off_rebound",)),
         "drb_events": _events(gs, overrides, ("def_rebound",)),
         "fb_events": _events(gs, overrides, ("fast_break",))}
    pers = periods(gs)
    t_lo = min([pers[0]["t0"]] + [b["t"] for b in baskets if b["t"] is not None]) if pers else 0.0
    t_hi = max([pers[-1]["t1"]] + [b["t"] for b in baskets if b["t"] is not None]) + 1.0 if pers else 1e12
    game = {k: _team_block(k, t_lo, t_hi, d) for k in TEAMS}
    per = []
    for i, p in enumerate(pers):
        a = t_lo if i == 0 else p["t0"]
        b = t_hi if i == len(pers) - 1 else pers[i + 1]["t0"]  # breaks belong to the period before them
        per.append({**p, "w0": round(a, 2), "w1": round(b, 2), **{k: _team_block(k, a, b, d) for k in TEAMS}})
    unattributed = [b for b in baskets if b["team"] not in TEAMS]
    score = {"detected": {k: game[k]["points"] for k in TEAMS},
             "unattributed_points": sum(b["points"] for b in unattributed),
             "unattributed_baskets": [_item(b["t"], f"{b['label']} ({b['points']} pt) – team unknown", id=b["id"])
                                      for b in unattributed],
             "dismissed_baskets": len(dismissed), "made_baskets": len(baskets),
             "not_on_sheet": [_item(b["t"], f"Basket not on the scoresheet? ({b.get('side') or ''})", id=b["id"])
                              for b in dismissed if b.get("not_on_sheet")],
             "from_sheet": sum(1 for b in baskets if b.get("from_sheet")),
             "doubtful_makes": [_item(b["t"], f"Made basket? not counted - the play stayed at the basket "
                                              f"({b.get('side') or ''})", id=b["id"]) for b in dismissed
                                if b.get("doubtful")]}
    score.update(reconcile(score["detected"], truth, names, colour_names, pers, baskets))
    lu = lineup_stats(lineup_doc, gs, baskets, d["possessions"], clock=game_clock)
    n_raw = len(gs.get("possessions") or [])
    notes = [
        "All numbers are ESTIMATED FROM AUTOMATICALLY DETECTED EVENTS (ball, rims, players), not an official "
        "box score. Click a row to see and check the detected events behind it.",
        f"Points = {len(baskets)} detected made baskets ({score['unattributed_points']} points without a team"
        + (f"; {len(dismissed)} dismissed on the timeline" if dismissed else "")
        + "). 2 vs 3 points is guessed from the shooter's distance; the scoreboard is not read.",
        f"Possessions: {n_raw} detected ball-control spells merged into {len(d['possessions'])} team possessions "
        f"(an offensive rebound continues a possession; {len(absorbed)} other-team spells shorter than "
        f"{BRIEF_SPELL_S:g} s with no shot were treated as holder mix-ups, and the turnovers around them were dropped).",
        f"Turnovers: a change of possession within {AFTER_BASKET_S:g} s after the same team's shot or made basket "
        "is not counted (missed rebound / inbound after a score).",
        f"Counted events have confidence >= {MIN_EVENT_CONF:.0%}; rebounds and turnovers are often missed when "
        "the ball is not seen.",
        "Periods are video time between detected breaks (" + (", ".join(p["label"] for p in pers) or "none") + ").",
    ]
    located = sum(1 for s in shots if s["zone"] != "unknown")
    notes.append(f"Shot chart: {located} of {len(shots)} field-goal attempts have a court location; zones within "
                 f"{ZONE_MARGIN_M} m of the 3-point line are marked '?'.")
    return {"format": FORMAT, "teams": {k: {"name": names[k], "color": (colors or {}).get(k)} for k in TEAMS},
            "court": geo, "time_range_s": gs.get("time_range_s"), "periods": per, "game": game,
            "shots": shots, "zones": {k: zone_table(shots, team=k) for k in TEAMS}, "score": score,
            "lineups": lu, "notes": notes, "attack_direction": gs.get("attack_direction") or []}


def for_project(project_root: Path | str, replay: dict | None = None, game_state: dict | None = None,
                max_build_frames: int | None = None) -> dict:
    inp = load_inputs(project_root, replay, game_state, max_build_frames)
    return compute(inp["game_state"], inp["replay_events"], inp["overrides"], inp["lineup"], inp["truth"],
                   inp["names"], inp["colors"], inp["colour_names"], game_clock=inp.get("game_clock"))


# ====================================================================================== shot filters / zones
def filter_shots(shots: Iterable[dict], team: str | None = None, period: dict | None = None,
                 shooter: str | None = None, outcome: str | None = None) -> list[dict]:
    out = []
    for s in shots:
        if team and s.get("team") != team:
            continue
        if period is not None and not (period.get("w0", period["t0"]) <= s["t"] < period.get("w1", period["t1"])):
            continue
        if shooter and s.get("shooter_name") != shooter:
            continue
        if outcome and s.get("outcome") != outcome:
            continue
        out.append(s)
    return out


def zone_table(shots: Iterable[dict], team: str | None = None) -> dict[str, dict]:
    """FGM / FGA / FG% per zone (+ "unknown" location and "total")."""
    rows = {z: {"fgm": 0, "fga": 0} for z in ZONES + ("unknown", "total")}
    for s in shots:
        if team and s.get("team") != team:
            continue
        if s.get("outcome") not in ("made", "missed"):
            continue  # result not seen: not an attempt we can grade
        for z in (s.get("zone") or "unknown", "total"):
            rows[z]["fga"] += 1
            rows[z]["fgm"] += s["outcome"] == "made"
    for r in rows.values():
        r["pct"] = _ratio(r["fgm"], r["fga"])
    return rows


# ====================================================================================== score truth
def _truth_team(key: str, names: dict, colour_names: dict | None) -> str | None:
    k = str(key).strip().lower()
    if k.upper() in TEAMS:
        return k.upper()
    for t in TEAMS:
        if k == names.get(t, "").lower() or k == (colour_names or {}).get(t, "").lower():
            return t
    for t in TEAMS:
        if k and (k in names.get(t, "").lower() or k in (colour_names or {}).get(t, "").lower()):
            return t
    return None


def reconcile(detected: dict, truth: dict | None, names: dict, colour_names: dict | None = None,
              pers: Sequence[dict] = (), baskets: Sequence[dict] = ()) -> dict:
    """Compare detected points with a ground truth score: ``{"final": {team: points}}`` (team = A/B, a team
    name or a jersey colour) and/or ``{"events": [{"t", "team", "points"}]}`` (score-bar changes)."""
    if not truth:
        return {"truth": None}
    final: dict[str, float] = {}
    for k, v in (truth.get("final") or {}).items():
        t = _truth_team(k, names, colour_names)
        if t and _f(v) is not None:
            final[t] = float(v)
    evs = []
    for e in truth.get("events") or []:
        t = _truth_team(str(e.get("team", "")), names, colour_names)
        if t and _f(e.get("t")) is not None:
            evs.append({"t": float(e["t"]), "team": t, "points": float(_f(e.get("points")) or 2)})
    if evs and not final:
        final = {k: sum(e["points"] for e in evs if e["team"] == k) for k in TEAMS}
    out: dict[str, Any] = {"truth": {k: final.get(k) for k in TEAMS}, "source": truth.get("source") or "score_truth.json"}
    out["diff"] = {k: (detected.get(k, 0) - final[k]) if k in final else None for k in TEAMS}
    out["pct_of_truth"] = {k: (round(100.0 * detected.get(k, 0) / final[k], 1) if final.get(k) else None)
                           for k in TEAMS}
    tot_t = sum(final.values())
    out["total_abs_error"] = sum(abs(v) for v in out["diff"].values() if v is not None)
    out["total_pct_of_truth"] = round(100.0 * sum(detected.values()) / tot_t, 1) if tot_t else None
    if evs and pers:
        out["periods"] = [{"label": p["label"], **{k: sum(e["points"] for e in evs if e["team"] == k
                                                          and p["t0"] - 15 <= e["t"] < p["t1"] + 15) for k in TEAMS}}
                          for p in pers]
    return out


# ====================================================================================== lineups
def lineup_stats(lineup_doc: dict | None, gs: dict, baskets: Sequence[dict], poss: Sequence[dict],
                 clock=None) -> dict:
    """Per 5-player combination (lineup solver slots, jersey numbers): minutes, +/-, ratings.
    Minutes are GAME-CLOCK minutes (replay/minutes.py GameClock: score-bar clock, else live play scaled to the game
    length). A slot's player label is decided per stretch between dead balls (substitutions happen only there): the
    number read most in that stretch; unread pieces never inherit a number across a dead ball."""
    from replay import minutes as minutes_mod
    clock = clock or minutes_mod.GameClock(gs)
    if not isinstance(lineup_doc, dict) or not isinstance(lineup_doc.get("slots"), dict):
        return {"available": False, "reason": "Lineups unavailable: the lineup solver did not run for this build "
                                              "(it is off by default; turn it on with REPLAY_LINEUP=1 and rebuild "
                                              "the replay to get results/lineup.json)."}
    slots = lineup_doc["slots"]
    tl: dict[str, list[tuple[float, float, str, bool]]] = {}  # slot -> [(t0, t1, player label, number read)]
    n_pieces = n_num = 0
    for label, s in slots.items():
        pieces = sorted((p for p in (s or {}).get("pieces") or [] if len(p) >= 3), key=lambda p: float(p[1]))
        cut = []  # a tracker piece running through a dead ball is cut there (the player may change)
        for p in pieces:
            a_, b_ = float(p[1]), float(p[2])
            for x, y in clock.dead:
                if a_ < x and y < b_:
                    cut.append([p[0], a_, x, *p[3:]])
                    a_ = y
            cut.append([p[0], a_, b_, *p[3:]])
        pieces = cut
        stretches: list[list] = []  # pieces between dead balls (a substitution can only happen in a dead ball)
        for p in pieces:
            if not stretches or clock.dead_between(float(stretches[-1][-1][2]), float(p[1])):
                stretches.append([])
            stretches[-1].append(p)
        seq = []
        reads = [(float(x[0]), int(x[1])) for x in (lineup_doc.get("reads") or {}).get(label) or []]
        for st_ in stretches:
            w: dict[int, float] = {}
            a_, b_ = float(st_[0][1]), max(float(p[2]) for p in st_)
            valid = getattr(clock, "rosters", {}).get(label[0])  # scoresheet numbers of this team, if known
            for tr_, num in reads:  # jersey reads inside this stretch (lineup.json "reads", newer builds)
                if a_ <= tr_ <= b_ and (not valid or num in valid):
                    w[num] = w.get(num, 0.0) + 1.0
            if not reads:  # older lineup.json: one number per piece (a long piece may span substitutions)
                for p in st_:
                    num = int(p[3]) if len(p) > 3 and p[3] is not None else -1
                    if num >= 0:
                        w[num] = w.get(num, 0.0) + float(p[2]) - float(p[1])
            main = f"#{max(w, key=w.get)}" if w else f"{label} (no number)"
            prev_end = seq[-1][1] if seq else None
            for j, p in enumerate(st_):
                num = int(p[3]) if len(p) > 3 and p[3] is not None else -1
                n_pieces += 1
                n_num += num >= 0
                own = f"#{num}" if num >= 0 and not reads else main  # with reads: the stretch's number wins
                a0 = float(p[1])
                if j == 0 and prev_end is not None:  # the new stretch starts right after its dead ball
                    a0 = min(a0, max([y for x, y in clock.dead if prev_end <= y <= a0] or [a0]))
                seq.append((a0, float(p[2]), own, num >= 0))
        if seq:
            tl[label] = seq
    teams_slots = {k: sorted(lbl for lbl in tl if lbl.startswith(k)) for k in TEAMS}
    if not any(teams_slots.values()):
        return {"available": False, "reason": "Lineups unavailable: the lineup solver found no slots."}
    t0, t1 = (gs.get("time_range_s") or [0.0, 0.0])[:2]
    t0 = min([float(t0 or 0.0)] + [seq[0][0] for seq in tl.values()])
    t1 = max([float(t1 or 0.0)] + [seq[-1][1] for seq in tl.values()])
    brk = [(float(b["t0"]), float(b["t1"])) for b in gs.get("breaks") or [] if _f(b.get("t0")) is not None]

    def label_at(seq, t):
        cur = seq[0][2]
        for a, _b, lab, _k in seq:
            if a <= t + 1e-6:
                cur = lab
            else:
                break
        return cur

    def live(a, b):
        return max(0.0, (b - a) - sum(max(0.0, min(b, y) - max(a, x)) for x, y in brk))

    bounds = sorted({t0, t1} | {a for seq in tl.values() for a, *_ in seq if t0 < a < t1})
    rows: dict[tuple, dict] = {}
    for team in TEAMS:
        sl = teams_slots[team]
        if not sl:
            continue
        stints: list[list] = []
        for a, b in zip(bounds, bounds[1:]):
            key = tuple(sorted(label_at(tl[s], (a + b) / 2) for s in sl))
            if stints and stints[-1][0] == key:
                stints[-1][2] = b
            else:
                stints.append([key, a, b])
        for key, a, b in stints:
            r = rows.setdefault((team, key), {"team": team, "players": list(key), "seconds": 0.0, "stints": [],
                                               "pts_for": 0, "pts_against": 0, "poss_for": 0, "poss_against": 0})
            r["seconds"] += clock.seconds(a, b)
            r["stints"].append(_item(a, f"{' '.join(key)} on court", t1=b))
            last = b >= t1 - 1e-6
            for bk in baskets:
                if bk["t"] is not None and (a <= bk["t"] < b or (last and bk["t"] >= b)) and bk["team"] in TEAMS:
                    r["pts_for" if bk["team"] == team else "pts_against"] += bk["points"]
            for p in poss:
                if a <= p["t0"] < b and p.get("team") in TEAMS:  # (v6 "unclear" possessions count for nobody)
                    r["poss_for" if p["team"] == team else "poss_against"] += 1
    out_rows = []
    for r in rows.values():
        r["minutes"] = round(r["seconds"] / 60.0, 1)
        r["plus_minus"] = r["pts_for"] - r["pts_against"]
        r["off_rating"] = round(100.0 * r["pts_for"] / r["poss_for"], 1) if r["poss_for"] else None
        r["def_rating"] = round(100.0 * r["pts_against"] / r["poss_against"], 1) if r["poss_against"] else None
        r["net_rating"] = (round(r["off_rating"] - r["def_rating"], 1)
                           if r["off_rating"] is not None and r["def_rating"] is not None else None)
        r["identified"] = sum(1 for p in r["players"] if p.startswith("#"))
        r["duplicate_numbers"] = len(set(r["players"])) < len(r["players"])  # two slots read the same number
        r["t"] = r["stints"][0]["t"]
        r["seconds"] = round(r["seconds"], 1)
        out_rows.append(r)
    out_rows.sort(key=lambda r: (r["team"], -r["seconds"]))
    player_s: dict[tuple[str, str], float] = {}
    for r in out_rows:
        for p in dict.fromkeys(r["players"]):
            if p.startswith("#"):
                player_s[(r["team"], p)] = player_s.get((r["team"], p), 0.0) + r["seconds"]
    warnings = clock.warnings + minutes_mod.sanity(player_s, clock)
    return {"available": True, "rows": out_rows, "pieces": n_pieces, "pieces_with_number": n_num,
            "clock": clock.info(), "warnings": warnings,
            "note": f"Players = jersey numbers read on {n_num} of {n_pieces} slot pieces (others inherit the slot's "
                    "previous number within the stretch between two dead balls). Minutes = game-clock minutes ("
                    + {"clock": "score-bar clock", "live": "live play, scaled to the game length",
                       "video": "video time"}[clock.source] + "). Lineups with '(no number)' "
                    "players or the same number twice (two slots, one read) are uncertain."}


# ====================================================================================== exports
TEAM_COLUMNS = ["scope", "period", "t0", "t1", "team", "team_name"] + [k for k, _t, _k in STAT_ROWS] + ["ft_pct"]
SHOT_COLUMNS = ["id", "t", "time", "team", "team_name", "basket", "x_m", "y_m", "distance_m", "zone",
                "zone_uncertain", "value", "outcome", "points", "shooter", "shooter_name", "confidence", "source",
                "basket_event"]
ZONE_COLUMNS = ["team", "team_name", "zone", "fgm", "fga", "fg_pct"]
LINEUP_COLUMNS = ["team", "team_name", "players", "identified_players", "duplicate_numbers", "minutes", "stints",
                  "pts_for",
                  "pts_against", "plus_minus", "poss_for", "poss_against", "off_rating_per100", "def_rating_per100",
                  "net_rating", "first_t"]


def _strip_items(x: Any) -> Any:
    if isinstance(x, dict):
        return {k: _strip_items(v) for k, v in x.items() if k not in ("items", "stints")}
    if isinstance(x, list):
        return [_strip_items(v) for v in x]
    return x


def tables(st: dict) -> dict[str, list[list]]:
    """CSV tables (header first): stats_team.csv, stats_shots.csv, stats_zones.csv, stats_lineups.csv."""
    nm = {k: st["teams"][k]["name"] for k in TEAMS}
    team_rows = [TEAM_COLUMNS]
    tr = st.get("time_range_s") or [None, None]
    for k in TEAMS:
        g = st["game"][k]
        team_rows.append(["game", "", tr[0], tr[1], k, nm[k]] + [g.get(c) for c, _t, _k in STAT_ROWS] + [g.get("ft_pct")])
    for p in st["periods"]:
        for k in TEAMS:
            team_rows.append(["period", p["label"], p["t0"], p["t1"], k, nm[k]]
                             + [p[k].get(c) for c, _t, _k in STAT_ROWS] + [p[k].get("ft_pct")])
    shots = [SHOT_COLUMNS] + [[s["id"], s["t"], fmt_clock(s["t"]), s["team"] or "", nm.get(s["team"], ""),
                               s["basket"] or "", s["x"], s["y"], s["distance_m"], s["zone"], int(s["zone_uncertain"]),
                               s["value"], s["outcome"], s["points"], s["shooter"] or "", s["shooter_name"] or "",
                               s["confidence"], s["source"], s["basket_event"] or ""] for s in st["shots"]]
    zones = [ZONE_COLUMNS]
    for k in TEAMS:
        for z, r in st["zones"][k].items():
            zones.append([k, nm[k], z, r["fgm"], r["fga"], r["pct"]])
    lineups = [LINEUP_COLUMNS]
    for r in (st.get("lineups") or {}).get("rows") or []:
        lineups.append([r["team"], nm[r["team"]], " ".join(r["players"]), r["identified"],
                        int(r["duplicate_numbers"]), r["minutes"],
                        len(r["stints"]), r["pts_for"], r["pts_against"], r["plus_minus"], r["poss_for"],
                        r["poss_against"], r["off_rating"], r["def_rating"], r["net_rating"], r["t"]])
    return {"stats_team.csv": team_rows, "stats_shots.csv": shots, "stats_zones.csv": zones,
            "stats_lineups.csv": lineups}


def write_exports(st: dict, out_dir: Path | str, full_json: bool = True) -> list[Path]:
    """Write the CSV tables + ``stats.json`` (with the explaining item lists when ``full_json``)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, rows in tables(st).items():
        p = out_dir / name
        with p.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows([["" if v is None else v for v in r] for r in rows])
        paths.append(p)
    p = out_dir / "stats.json"
    p.write_text(json.dumps(st if full_json else _strip_items(st), indent=1, ensure_ascii=False), encoding="utf-8")
    paths.append(p)
    return paths


EXPORT_DESCRIPTIONS = {
    "stats.json": "team / period / shot-zone / lineup statistics ESTIMATED from the detected events, with the "
                  "events behind every number and the definitions (notes)",
    "stats_team.csv": "team stats per game and per detected period: points, FG/3PT/FT, possessions, points per "
                      "possession, turnover rate, offensive rebound rate, fast-break points, time of possession",
    "stats_shots.csv": "field-goal attempts for a shot chart: court x/y, zone (paint / mid_range / corner_3 / "
                       "above_break_3), outcome, points, shooter",
    "stats_zones.csv": "FG made / attempts / % per zone and team",
    "stats_lineups.csv": "5-player combinations (lineup solver + jersey numbers): minutes, +/-, points for / "
                         "against per 100 possessions (empty when lineups are unavailable)",
}


def export_into(out_dir: Path, project_root: Path, game_state: dict | None = None, replay: dict | None = None
                ) -> dict[str, str]:
    """Simple data export hook: write the stats files into ``out_dir``. Returns {file: description} of what was
    written. Never raises (the export must work without stats)."""
    try:
        st = for_project(project_root, replay, game_state)
        write_exports(st, out_dir, full_json=False)
        return dict(EXPORT_DESCRIPTIONS)
    except Exception as e:  # noqa: BLE001
        print("stats export failed:", type(e).__name__, e)
        return {}


# ====================================================================================== text
def summary_lines(st: dict) -> list[str]:
    nm = {k: st["teams"][k]["name"] for k in TEAMS}
    g = st["game"]
    out = [f"Detected score: {nm['A']} {g['A']['points']} – {g['B']['points']} {nm['B']}"
           + (f" (+{st['score']['unattributed_points']} pts without team)" if st["score"]["unattributed_points"] else "")]
    tr = st["score"].get("truth")
    if tr and all(tr.get(k) is not None for k in TEAMS):
        out.append(f"Ground truth: {nm['A']} {tr['A']:.0f} – {tr['B']:.0f} {nm['B']} "
                   f"(detected {st['score'].get('total_pct_of_truth')} % of the points)")
    for k in TEAMS:
        s = g[k]
        pct = lambda v: "–" if v is None else f"{100 * v:.0f}%"  # noqa: E731
        out.append(f"{nm[k]}: {s['points']} pts, FG {s['fgm']}/{s['fga']} ({pct(s['fg_pct'])}), 3PT "
                   f"{s['fg3m']}/{s['fg3a']}, FT {s['ftm']}/{s['fta']}, {s['possessions']} poss, PPP "
                   f"{s['ppp'] if s['ppp'] is not None else '–'}, TOV {s['turnovers']} ({pct(s['tov_rate'])}), "
                   f"ORB% {pct(s['orb_rate'])}, fast-break pts {s['fast_break_points']}, possession "
                   f"{pct(s['top_pct'])}")
    return out
