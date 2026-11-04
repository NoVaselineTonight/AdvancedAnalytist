"""Post-game coach report data: box score per player, team comparison, lineups, runs / momentum, player pages.

No Qt here (unit-testable). The page layout, PDF and the in-app view are in ``replay/gui/coach_report.py``.
Built on :mod:`replay.stats` (team stats, shots, lineups, periods) and :mod:`replay.playlists` (who is who: jersey
numbers per tracker ID, on-court slots). Nothing new is detected here; every number says where it comes from:

* ``measured``  - counted directly from detected events (made baskets at the rim, steals seen by the ball tracker).
* ``inferred``  - derived from detections plus a rule (shooter = ball holder at release, assist = the pass caught by
  the scorer within ASSIST_S before the shot, minutes / +/- from the lineup solver's on-court slots).
* ``not measured`` - not detected at all (personal fouls): shown as "–", never as 0.

Every stat cell carries the moments behind it ({"t", "t1", "title", "team"}) so a click can open them as a video
playlist (``playlists.fixed_playlist``).
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

from replay import stats as S

FORMAT = "advanced-analytist/coach-report@1"
TEAMS = ("A", "B")
ASSIST_S = 3.0  # a pass caught by the scorer at most this long before the shot counts as an assist
RUN_MIN_POINTS = 6  # unanswered points that make a "run"
HEAT_BIN_M = 1.0

COLUMNS = [  # (key, title, provenance)
    ("min", "MIN", "inferred"), ("pts", "PTS", "inferred"), ("fg", "FG", "inferred"), ("fg3", "3PT", "inferred"),
    ("ft", "FT", "team"), ("oreb", "OREB", "inferred"), ("dreb", "DREB", "inferred"), ("ast", "AST", "inferred"),
    ("stl", "STL", "measured"), ("tov", "TOV", "inferred"), ("pf", "PF", "not measured"),
    ("pm", "+/-", "inferred"),
]
PROVENANCE = {
    "min": "Minutes on court from the lineup solver (tracking + jersey numbers), on the GAME CLOCK (score-bar clock; "
           "else live play scaled to the game length); a player is re-identified after every dead ball.",
    "pts": "Made baskets are measured at the rim; the scorer is the ball holder at release (inferred). 2 vs 3 is "
           "guessed from the shot distance.",
    "fg": "Field-goal attempts: shots detected by the ball tracker; shooter inferred from ball possession.",
    "fg3": "3-point attempts: shot position beyond the 3-point line (foot-point accuracy ~0.5 m).",
    "ft": "Free throws are counted per team only (the shooter is not identified).",
    "oreb": "Rebounds: the first player to control the ball after a missed shot (inferred).",
    "dreb": "Rebounds: the first player to control the ball after a missed shot (inferred).",
    "ast": f"Assists: the passer of a pass caught by the scorer within {ASSIST_S:g} s before a made field goal.",
    "stl": "Steals: ball control changing team between two nearby players (measured by the ball tracker).",
    "tov": "Turnovers: steals against and lost balls, charged to the player who had the ball.",
    "pf": "Personal fouls are not detected.",
    "pm": "+/-: points scored minus points allowed while the player's 5-man unit was on court (lineup solver).",
}


def _f(v: Any) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _moment(t: Any, title: str, team: str | None, t1: Any = None) -> dict | None:
    t = _f(t)
    if t is None:
        return None
    return {"t": t, "t1": _f(t1) if t1 is not None else None, "title": title, "team": team}


class _Who:
    """Tracker ID -> player ("#12" of team A) from the playlist catalog (jersey numbers per track)."""

    def __init__(self, catalog: Any):
        self.num = dict(getattr(catalog, "track_number", {}) or {})
        self.team = dict(getattr(catalog, "track_team", {}) or {})

    def __call__(self, tid: Any, team: str | None = None) -> tuple[str | None, str]:
        """(team, player label). Unnumbered tracks are 'Unidentified' (one row per team)."""
        if tid is None:
            return team, "Unidentified"
        tid = str(tid)
        tm = self.team.get(tid) or team
        n = self.num.get(tid)
        return tm, (f"#{n}" if n else "Unidentified")


def _blank() -> dict:
    return {"min_s": 0.0, "pts": 0, "fgm": 0, "fga": 0, "fg3m": 0, "fg3a": 0, "oreb": 0, "dreb": 0, "ast": 0,
            "stl": 0, "tov": 0, "pm": 0, "pm_known": False, "moments": {k: [] for k, _t, _p in COLUMNS}}


def _sort_key(label: str) -> tuple:
    if label.startswith("#") and label[1:].isdigit():
        return (0, int(label[1:]))
    return (1, label)


def build(project_root: Path | str, st: dict | None = None, catalog: Any = None, replay: dict | None = None) -> dict:
    """Everything the coach report shows, for one Replay folder."""
    root = Path(project_root)
    inp = S.load_inputs(root, replay)
    gs = inp["game_state"]
    if st is None:
        st = S.compute(gs, inp["replay_events"], inp["overrides"], inp["lineup"], inp["truth"], inp["names"],
                       inp["colors"], inp["colour_names"])
    if catalog is None:
        from replay import playlists as pl
        try:
            catalog = pl.collect(root)
        except Exception as e:  # noqa: BLE001 - a report without player names is still a report
            print("coach report: no player catalog:", type(e).__name__, e)
            catalog = None
    who = _Who(catalog)
    players: dict[tuple[str, str], dict] = {}

    def row(team: str | None, label: str) -> dict | None:
        if team not in TEAMS:
            return None
        return players.setdefault((team, label), _blank())

    # ---- shots (FG / 3PT / PTS)
    shots = [s for s in st.get("shots") or [] if s.get("team") in TEAMS]
    for s in shots:
        team, label = who(s.get("shooter"), s["team"])
        if team != s["team"]:
            team = s["team"]
        r = row(team, label)
        m = _moment(s.get("t"), s.get("title") or "Shot", team, s.get("t_rim"))
        r["fga"] += 1
        r["moments"]["fg"].append(m)
        if s.get("value") == 3:
            r["fg3a"] += 1
            r["moments"]["fg3"].append(m)
        if s.get("outcome") == "made":
            r["fgm"] += 1
            r["pts"] += int(s.get("points") or s.get("value") or 2)
            r["moments"]["pts"].append(m)
            if s.get("value") == 3:
                r["fg3m"] += 1
    # ---- events: rebounds, steals, turnovers, passes (assists)
    events = [e for e in gs.get("events") or [] if isinstance(e, dict)
              and (_f(e.get("confidence")) or 0.0) >= S.MIN_EVENT_CONF
              and not (inp["overrides"].get(str(e.get("id"))) or {}).get("dismissed")]
    passes = []
    for e in events:
        d = e.get("details") or {}
        typ = e.get("type")
        title = e.get("title") or typ
        if typ in ("off_rebound", "def_rebound"):
            team, label = who(d.get("rebounder_track"), e.get("team"))
            r = row(team, label)
            if r is not None:
                key = "oreb" if typ == "off_rebound" else "dreb"
                r[key] += 1
                r["moments"][key].append(_moment(e.get("t"), title, team))
        elif typ == "steal":
            team, label = who(d.get("stealer_track"), e.get("team"))
            r = row(team, label)
            if r is not None:
                r["stl"] += 1
                r["moments"]["stl"].append(_moment(e.get("t"), title, team))
            lt = d.get("lost_by") if d.get("lost_by") in TEAMS else S._other(e.get("team"))
            team2, label2 = who(d.get("lost_by_track"), lt)
            r2 = row(team2, label2)
            if r2 is not None:
                r2["tov"] += 1
                r2["moments"]["tov"].append(_moment(e.get("t"), title, team2))
        elif typ in ("lost_ball", "turnover", "bad_pass"):
            team, label = who(d.get("holder_track") or d.get("lost_by_track") or d.get("passer_track"),
                               e.get("team"))
            r = row(team, label)
            if r is not None:
                r["tov"] += 1
                r["moments"]["tov"].append(_moment(e.get("t"), title, team))
        elif typ == "pass":
            passes.append(e)
    for s in shots:
        if s.get("outcome") != "made" or s.get("shooter") is None:
            continue
        st_t = _f(s.get("t")) or 0.0
        best = None
        for e in passes:
            d = e.get("details") or {}
            tc = _f(d.get("t_catch")) or _f(e.get("t"))
            if e.get("team") == s["team"] and str(d.get("receiver_track")) == str(s["shooter"]) and tc is not None \
                    and 0.0 <= st_t - tc <= ASSIST_S and (best is None or tc > best[0]):
                best = (tc, e)
        if best is not None:
            e = best[1]
            team, label = who((e.get("details") or {}).get("passer_track"), s["team"])
            r = row(team, label)
            if r is not None:
                r["ast"] += 1
                r["moments"]["ast"].append(_moment(e.get("t"), f"Assist → {s.get('title') or 'made shot'}", team,
                                                   s.get("t")))
    # ---- minutes and +/- from the lineup solver
    lu = st.get("lineups") or {}
    for lr in lu.get("rows") or []:
        team = lr.get("team")
        for pname in dict.fromkeys(lr.get("players") or []):  # a number read twice in one unit counts once
            label = pname if str(pname).startswith("#") else "Unidentified"
            if label == "Unidentified":
                continue  # an unnumbered slot is not one person
            r = row(team, label)
            if r is None:
                continue
            r["min_s"] += float(lr.get("seconds") or 0.0)
            r["pm"] += int(lr.get("plus_minus") or 0)
            r["pm_known"] = True
            for stn in lr.get("stints") or []:
                r["moments"]["min"].append(_moment(stn.get("t"), stn.get("title") or "On court", team, stn.get("t1")))
                r["moments"]["pm"].append(r["moments"]["min"][-1])
    # ---- tables
    names = {k: (st.get("teams") or {}).get(k, {}).get("name") or k for k in TEAMS}
    colors = {k: (st.get("teams") or {}).get(k, {}).get("color") for k in TEAMS}
    box = {}
    for k in TEAMS:
        rows = []
        for (team, label), r in sorted(players.items(), key=lambda kv: _sort_key(kv[0][1])):
            if team != k:
                continue
            rows.append({"player": label, **{c: r[c] for c in ("min_s", "pts", "fgm", "fga", "fg3m", "fg3a",
                                                                "oreb", "dreb", "ast", "stl", "tov")},
                         "pm": r["pm"] if r["pm_known"] else None,
                         "moments": {c: [m for m in v if m] for c, v in r["moments"].items()}})
        g = (st.get("game") or {}).get(k) or {}
        it = g.get("items") or {}

        def moms(key: str, team: str = k) -> list:
            return [m for m in (_moment(x.get("t"), x.get("title") or key, team, x.get("t1")) for x in it.get(key) or [])
                    if m]
        total = {"player": "Team", "min_s": None, "pts": g.get("points", 0), "fgm": g.get("fgm", 0),
                 "fga": g.get("fga", 0), "fg3m": g.get("fg3m", 0), "fg3a": g.get("fg3a", 0), "ftm": g.get("ftm", 0),
                 "fta": g.get("fta", 0), "oreb": g.get("orb", 0), "dreb": g.get("drb", 0),
                 "ast": sum(r["ast"] for r in rows), "stl": g.get("steals", 0), "tov": g.get("turnovers", 0),
                 "pm": None,
                 "moments": {"pts": moms("baskets"), "fg": moms("fg_attempts"), "fg3": moms("fg3_attempts"),
                             "ft": moms("ft_attempts"), "oreb": moms("orb"), "dreb": moms("drb"),
                             "ast": [m for r in rows for m in r["moments"]["ast"]], "stl": moms("steals"),
                             "tov": moms("turnovers"), "min": [], "pf": [], "pm": []}}
        box[k] = {"rows": rows, "total": total}
    baskets = _team_baskets(inp, gs)
    rep = {"format": FORMAT, "teams": {k: {"name": names[k], "color": colors[k]} for k in TEAMS},
           "score": {k: (st.get("game") or {}).get(k, {}).get("points", 0) for k in TEAMS},
           "score_note": _score_note(st), "box": box, "compare": _compare(st), "shots": shots,
           "court": st.get("court"), "lineups": _lineup_table(st), "runs": runs(baskets),
           "momentum": momentum(baskets, st.get("time_range_s")), "periods": st.get("periods") or [],
           "time_range_s": st.get("time_range_s"), "notes": list(st.get("notes") or []) + list((st.get("lineups") or {}).get("warnings") or []),
           "provenance": PROVENANCE, "columns": COLUMNS, "sheet": None, "stats_score": st.get("score") or {}}
    sheet_info = None
    try:
        sheet_info = merge_scoresheet(rep, root)
    except Exception as e:  # noqa: BLE001 - the report works without the sheet
        print("coach report: scoresheet not merged:", type(e).__name__, e)
    rep["period_table"] = period_table(st, root, sheet_info)
    try:  # possession timeline + shot quality map (report page 2)
        rep["possessions"] = possession_timeline(st, gs, who)
        rep["shot_quality"] = shot_quality(shots, st.get("court"))
    except Exception as e:  # noqa: BLE001 - page 1 must survive
        print("coach report: possessions / shot quality failed:", type(e).__name__, e)
        rep["possessions"], rep["shot_quality"] = [], None
    return rep


def _team_baskets(inp: dict, gs: dict) -> list[dict]:
    kept, _dismissed = S._baskets(inp["replay_events"], gs, inp["overrides"])
    return [b for b in kept if b["team"] in TEAMS and b["t"] is not None]


def _score_note(st: dict) -> str:
    sc = st.get("score") or {}
    tr = sc.get("truth")
    parts = [f"{sc.get('made_baskets', 0)} made baskets detected"]
    if sc.get("unattributed_points"):
        parts.append(f"{sc['unattributed_points']} points without a team")
    if tr:
        parts.append("scoreboard truth supplied")
    return "; ".join(parts)


COMPARE_ROWS = [("points", "Points"), ("fg_pct", "FG %"), ("fg3a", "3-pt attempts"), ("fta", "FT attempts"),
                ("orb", "Off. rebounds"), ("drb", "Def. rebounds"), ("turnovers", "Turnovers"),
                ("steals", "Steals"), ("possessions", "Possessions"), ("ppp", "Points / possession"),
                ("fast_break_points", "Fast-break points"), ("top_pct", "Possession share")]


def _compare(st: dict) -> list[dict]:
    out = []
    g = st.get("game") or {}
    for key, title in COMPARE_ROWS:
        item_key = S.ROW_ITEMS.get(key)
        row = {"key": key, "title": title}
        for k in TEAMS:
            gk = g.get(k) or {}
            row[k] = gk.get(key)
            its = (gk.get("items") or {}).get(item_key) or [] if item_key else []
            row[f"moments_{k}"] = [m for m in (_moment(x.get("t"), x.get("title") or title, k, x.get("t1"))
                                               for x in its) if m]
        out.append(row)
    return out


def _lineup_table(st: dict, top: int = 5) -> dict:
    lu = st.get("lineups") or {}
    out = {"available": bool(lu.get("available")), "note": lu.get("note") or ""}
    for k in TEAMS:
        rows = sorted([r for r in lu.get("rows") or [] if r.get("team") == k], key=lambda r: -(r.get("seconds") or 0))
        out[k] = [{"players": r.get("players") or [], "seconds": r.get("seconds") or 0.0,
                   "pts_for": r.get("pts_for"), "pts_against": r.get("pts_against"), "pm": r.get("plus_minus"),
                   "net": r.get("net_rating"),
                   "moments": [m for m in (_moment(s.get("t"), s.get("title") or "On court", k, s.get("t1"))
                                           for s in r.get("stints") or []) if m]} for r in rows[:top]]
    return out


def runs(baskets: list[dict], min_points: int = RUN_MIN_POINTS) -> list[dict]:
    """Unanswered scoring runs: consecutive made baskets of one team worth >= min_points."""
    out, cur = [], None
    for b in sorted(baskets, key=lambda b: b["t"]):
        if cur is not None and cur["team"] == b["team"]:
            cur["points"] += b["points"]
            cur["t1"] = b["t"]
            cur["baskets"].append(b)
            continue
        if cur is not None and cur["points"] >= min_points:
            out.append(cur)
        cur = {"team": b["team"], "t0": b["t"], "t1": b["t"], "points": b["points"], "baskets": [b]}
    if cur is not None and cur["points"] >= min_points:
        out.append(cur)
    for r in out:
        r["title"] = f"{r['points']}-0 run"
        r["moments"] = [{"t": r["t0"] - 15.0, "t1": r["t1"], "title": r["title"], "team": r["team"]}]
        r.pop("baskets")
    return out


def momentum(baskets: list[dict], time_range: Any = None) -> list[tuple[float, int, int]]:
    """Score after every basket: [(t, points A, points B)], starting at 0-0."""
    t0 = _f((time_range or [0.0])[0]) if isinstance(time_range, (list, tuple)) and time_range else 0.0
    out = [(t0 or 0.0, 0, 0)]
    a = b = 0
    for bk in sorted(baskets, key=lambda x: x["t"]):
        if bk["team"] == "A":
            a += bk["points"]
        else:
            b += bk["points"]
        out.append((bk["t"], a, b))
    if isinstance(time_range, (list, tuple)) and len(time_range) > 1 and _f(time_range[1]):
        out.append((_f(time_range[1]), a, b))
    return out


# ====================================================================================== periods
_PERIOD_RE = re.compile(r"^\s*(q\s*[1-4]|[1-4]\s*(st|nd|rd|th|\.)?\s*(quarter|period|jakso|neljännes|erä)"
                        r"|[12]\s*(st|nd|\.)?\s*(half|puoliaika)|ot\s*\d?|overtime|jatkoaika)\b", re.I)


def _timeline_periods(root: Path) -> list[tuple[float, str]]:
    """(start time, label) of the user's timeline marks titled like a period ("Q2", "2nd quarter", "2. jakso",
    "2nd half", "OT"), sorted."""
    out = []
    tl = S._read(root / "timeline.json")
    for tr in (tl.get("tracks") or []) if isinstance(tl, dict) else []:
        for m in tr.get("marks") or []:
            title = str(m.get("title") or "")
            t = _f(m.get("t"))
            if t is not None and _PERIOD_RE.match(title):
                out.append((t, title.strip()))
    return sorted(out)


def _sheet_period_of(sheet_info: dict | None, t0: float, t1: float) -> int | None:
    """The sheet's quarter of the scoring entries aligned into [t0, t1) (Replays that are part of a game)."""
    if not sheet_info:
        return None
    ps = [e.get("period") for e in sheet_info.get("window") or [] if e.get("t") is not None
          and t0 <= e["t"] < t1 and e.get("period")]
    return max(set(ps), key=ps.count) if ps else None


def period_table(st: dict, root: Path, sheet_info: dict | None = None) -> dict:
    """Per-period team numbers. Periods: the user's timeline period marks if any, else the detected breaks
    (quarter / half-time); labelled with the scoresheet's quarter where the sheet is aligned. Official points per
    period from the sheet: whole game -> its quarter scores; part of a game -> its scoring entries in the window."""
    marks = _timeline_periods(root)
    tr = st.get("time_range_s") or [0.0, 0.0]
    lo, hi = _f(tr[0]) or 0.0, _f(tr[1]) or 0.0
    if marks:
        source = "timeline period marks"
        bounds = [(t, lbl) for t, lbl in marks if t < hi]
        if not bounds or bounds[0][0] > lo + 1.0:
            bounds.insert(0, (lo, "Before " + (bounds[0][1] if bounds else "the marks")))
        spans = [(a, (bounds[i + 1][0] if i + 1 < len(bounds) else hi + 1.0), lbl) for i, (a, lbl) in enumerate(bounds)]
    else:
        pers = st.get("periods") or []
        source = "detected breaks" if len(pers) > 1 else "whole Replay"
        spans = [(p.get("w0", p["t0"]), p.get("w1", p["t1"]), p.get("label") or f"Period {i + 1}")
                 for i, p in enumerate(pers)]
    full = bool(sheet_info and sheet_info.get("full_game"))
    q_scores = (sheet_info or {}).get("period_scores") or {}
    rows = []
    for i, (a, b, label) in enumerate(spans):
        tm = {}
        for k in TEAMS:
            shots = [s for s in st.get("shots") or [] if s["team"] == k and s.get("t") is not None and a <= s["t"] < b]
            made = [s for s in shots if s.get("outcome") == "made"]
            tm[k] = {"pts": sum(int(s.get("points") or s.get("value") or 2) for s in made), "fgm": len(made),
                     "fga": len(shots), "fg3m": sum(1 for s in made if s.get("value") == 3),
                     "fg3a": sum(1 for s in shots if s.get("value") == 3),
                     "moments": [m for m in (_moment(s.get("t"), s.get("title") or "Shot", k, s.get("t_rim"))
                                             for s in shots) if m]}
        official = None
        if full and q_scores and len(spans) == len(q_scores.get("A") or []):
            official = {k: q_scores[sheet_info["map_inv"][k]][i] for k in TEAMS}
        elif sheet_info and not full:
            win = [e for e in sheet_info.get("window") or [] if e.get("t") is not None and a <= e["t"] < b]
            if win:
                official = {k: sum(int(e.get("points") or 0) for e in win if sheet_info["map"].get(e.get("team")) == k)
                            for k in TEAMS}
        q = _sheet_period_of(sheet_info, a, b)
        if q and source != "timeline period marks":
            label = (f"Q{q}" if q <= 4 else f"OT{q - 4}") + " (from the scoresheet)"
        rows.append({"label": label, "t0": a, "t1": b, **tm, "official": official})
    return {"source": source, "rows": rows}


# ====================================================================================== scoresheet
def merge_scoresheet(rep: dict, root: Path) -> dict | None:
    """Official numbers from an attached scoresheet (replay/scoresheet.py) replace the detected ones where the sheet
    knows them, marked "official"; disagreements become data-quality hints. Whole-game Replay: per-player points,
    FT made and fouls of the game. Part of a game: only the sheet's scoring entries aligned into this Replay
    (points / FT per scorer); fouls then stay "not measured" (the sheet has no times for them). Tracking-only
    numbers (minutes, shots, rebounds, locations) are never touched. Returns the sheet info (or None)."""
    from replay import scoresheet as SS
    sheet = SS.load(root)
    if sheet is None:
        return None
    cc = SS.crosscheck(root, sheet)
    mp = dict(cc.get("team_map") or {"A": "A", "B": "B"})  # sheet side -> video team
    inv = {v: k for k, v in mp.items()}
    full = bool(cc.get("full_game"))
    window = [e for e in cc.get("sheet_window") or [] if isinstance(e, dict)]
    info = {"attached": True, "full_game": full, "map": mp, "map_inv": inv,
            "map_source": cc.get("team_map_source") or "", "window": window, "hints": [],
            "period_scores": dict(sheet.period_scores or {}), "final": sheet.final,
            "sheet_names": {k: sheet.teams[k].name for k in TEAMS}, "summary": cc.get("summary") or "",
            "alignment": cc.get("alignment"), "start_source": cc.get("start_source"),
            "pin_label": (cc.get("pin") or {}).get("label")}
    hints = info["hints"]
    off: dict[tuple[str, str], dict] = {}  # (video team, "#n") -> official numbers
    if full:
        for side in TEAMS:
            for p_ in sheet.teams[side].players:
                if p_.number:
                    off[(mp[side], f"#{p_.number}")] = {"pts": p_.points, "ftm": p_.ft_made, "pf": p_.n_fouls,
                                                        "moments": []}
        fin = sheet.final
        if isinstance(fin, dict) and fin.get("A") is not None:
            rep["score_official"] = {mp[s]: fin.get(s) for s in TEAMS}
        elif isinstance(fin, (list, tuple)) and len(fin) == 2:
            rep["score_official"] = {mp["A"]: fin[0], mp["B"]: fin[1]}
    else:
        for e in window:
            k = mp.get(e.get("team"))
            if k not in TEAMS or not e.get("scorer"):
                continue
            o = off.setdefault((k, f"#{e['scorer']}"), {"pts": 0, "ftm": 0, "pf": None, "moments": []})
            o["pts"] += int(e.get("points") or 0)
            if e.get("kind") == "ft":
                o["ftm"] += 1
            m = _moment(e.get("t"), f"Scoresheet: #{e['scorer']} {e.get('points')} pt", k)
            if m:
                o["moments"].append(m)
        if window:
            rep["score_official_window"] = {k: sum(int(e.get("points") or 0) for e in window
                                                   if mp.get(e.get("team")) == k) for k in TEAMS}
    names = {(mp[s], f"#{p_.number}"): p_.name for s in TEAMS for p_ in sheet.teams[s].players if p_.number}
    for k in TEAMS:
        rows = rep["box"][k]["rows"]
        by = {r["player"]: r for r in rows}
        for (team, label), o in off.items():
            if team != k:
                continue
            r = by.get(label)
            if r is None:  # a scorer / player the video did not identify: add the sheet's row
                if not (o.get("pts") or o.get("pf")):
                    continue
                r = {"player": label, "min_s": 0.0, "pts": 0, "fgm": 0, "fga": 0, "fg3m": 0, "fg3a": 0, "oreb": 0,
                     "dreb": 0, "ast": 0, "stl": 0, "tov": 0, "pm": None, "moments": {}, "sheet_only": True}
                rows.append(r)
                by[label] = r
            off_keys = r.setdefault("official", [])
            if o.get("pts") is not None:
                if o["pts"] != r["pts"]:
                    hints.append({"team": k, "player": label, "stat": "PTS", "detected": r["pts"], "sheet": o["pts"],
                                  "moments": list(r.get("moments", {}).get("pts", [])) + o.get("moments", [])})
                r["pts_detected"], r["pts"] = r["pts"], o["pts"]
                off_keys.append("pts")
                if o.get("moments"):
                    r.setdefault("moments", {})["pts"] = o["moments"] + list(r.get("moments", {}).get("pts", []))
            if o.get("ftm") is not None:
                r["ftm"], r["fta"] = o["ftm"], None
                off_keys.append("ft")
            if o.get("pf") is not None:
                r["pf"] = o["pf"]
                off_keys.append("pf")
        if off or (not full and window):  # the sheet lists every point: detected scorers it does not list scored 0
            for r in rows:
                if (k, r["player"]) in off or not r.get("pts"):
                    continue
                hints.append({"team": k, "player": r["player"], "stat": "PTS", "detected": r["pts"], "sheet": 0,
                              "moments": list(r.get("moments", {}).get("pts", []))})
                r["pts_detected"], r["pts"] = r["pts"], 0
                r.setdefault("official", []).append("pts")
        for r in rows:
            if names.get((k, r["player"])):
                r["name"] = names[(k, r["player"])]
        rows.sort(key=lambda r: (r["player"] == "Unidentified", _sort_key(r["player"])))
        tot = rep["box"][k]["total"]
        sheet_pts = (rep.get("score_official") or {}).get(k) if full else (rep.get("score_official_window") or {}).get(k)
        if sheet_pts is not None:
            if sheet_pts != tot["pts"]:
                hints.append({"team": k, "player": "Team", "stat": "PTS" if full else "PTS in this stretch",
                              "detected": tot["pts"], "sheet": sheet_pts, "moments": tot["moments"].get("pts", [])})
            tot["pts_detected"], tot["pts"] = tot["pts"], sheet_pts
            tot.setdefault("official", []).append("pts")
        if full:
            pf = [r.get("pf") for r in rows if r.get("pf") is not None]
            if pf:
                tot["pf"] = sum(pf)
                tot.setdefault("official", []).append("pf")
    official_score = rep.get("score_official") if full else rep.get("score_official_window")
    if official_score:
        rep["score_detected"] = dict(rep["score"])
        rep["score"] = {k: official_score[k] for k in TEAMS}
        for row in rep.get("compare") or []:
            if row["key"] == "points":
                row.update({k: official_score[k] for k in TEAMS}, official=True)
    # whole game: stats.py already reconciled the detected baskets with the sheet's running score
    # (ball_events.reconcile_with_sheet) - its team points then equal the sheet; say how far the detection was off
    sc = rep.get("stats_score") or {}
    off_sheet = sc.get("not_on_sheet") or []
    if full and (off_sheet or sc.get("from_sheet")):
        info["basket_check"] = {"not_on_sheet": len(off_sheet), "from_sheet": int(sc.get("from_sheet") or 0)}
        if off_sheet:
            hints.append({"team": "A", "player": "", "stat": "", "detected": len(off_sheet), "sheet": 0,
                          "moments": [m for m in (_moment(x.get("t"), x.get("title") or "Basket", None)
                                                  for x in off_sheet) if m],
                          "text": f"{len(off_sheet)} detected basket(s) are not on the scoresheet (not counted)"})
        if sc.get("from_sheet"):
            hints.append({"team": "A", "player": "", "stat": "", "detected": 0, "sheet": int(sc["from_sheet"]),
                          "moments": [], "text": f"{int(sc['from_sheet'])} scoresheet basket(s) were not detected in "
                                                 "the video (counted from the sheet, time estimated)"})
    for h in hints:
        if h.get("text"):
            continue
        h["text"] = (f"{rep['teams'][h['team']]['name']} {h['player']}: {h['stat']} detected {h['detected']}, "
                     f"scoresheet {h['sheet']}")
    rep["sheet"] = info
    return info


# ====================================================================================== possessions / shot quality
OUTCOMES = [("made", "Made basket"), ("ft", "Free throws"), ("miss", "Miss (rebound)"), ("turnover", "Turnover"),
            ("dead", "Dead ball"), ("period_end", "End of period"), ("unknown", "Not seen")]
UNCLEAR_MIN_S = 3.0  # v6: shorter "team unclear" spells are not rows of their own
OUTCOME_TITLES = dict(OUTCOMES)
QUALITY_ZONES = [("restricted", "Restricted area"), ("paint", "Paint (non-RA)"), ("mid_range", "Mid-range"),
                 ("corner_3", "Corner 3"), ("above_break_3", "Above-break 3")]
PRIOR_FG, PRIOR_N = 0.40, 5.0  # expected FG% per zone: this game's FG% there, shrunk to 40 % (weight 5 shots)


def possession_timeline(st: dict, gs: dict, who: Any) -> list[dict]:
    """Every (merged) possession of the game state as one row: start, team, duration, outcome, shooter.

    Outcome (first that applies, inside the possession + 2 s): made basket (2 / 3 / FT by the scorer: rim measured,
    shooter inferred), free throws, turnover / steal against the team, missed shot (the rebound decides the next
    possession), the end of a period, else "not seen" (the ball was lost from view). Offensive rebounds inside the
    possession are counted. Each row has the moment for its video clip.

    With the possession model v6 (game-state possessions carry ``outcome``) the rows come straight from the game
    state: outcome, points, shots and provenance (score bar / scoresheet / rim / holders / ball location) are the
    model's own, so a row never shows a team other than the one that scored in it; spells whose team the
    evidence does not decide are "Team unclear" rows (team None) instead of a wrong team."""
    raw = gs.get("possessions") or []
    shots = [s for s in st.get("shots") or [] if s.get("team") in TEAMS and s.get("t") is not None]
    if any("outcome" in p for p in raw):
        return _possession_timeline_v6(st, gs, who, raw, shots)
    poss = S.merge_possessions(raw)
    evs = [e for e in gs.get("events") or [] if isinstance(e, dict) and _f(e.get("t")) is not None
           and (_f(e.get("confidence")) or 0.0) >= S.MIN_EVENT_CONF]
    period_ends = [_f(p.get("t1")) for p in st.get("periods") or [] if _f(p.get("t1")) is not None]
    out = []
    used: set = set()
    for k, p in enumerate(poss):
        team, t0, t1 = p.get("team"), _f(p.get("t0")), _f(p.get("t1"))
        if team not in TEAMS or t0 is None or t1 is None:
            continue
        hi = t1 + 2.0
        mine = [s for s in shots if s["team"] == team and t0 - 0.5 <= s["t"] <= hi]
        made = [s for s in mine if s.get("outcome") == "made"]
        fts = [e for e in evs if e.get("type") in ("free_throw", "free_throws") and e.get("team") == team
               and t0 - 0.5 <= _f(e["t"]) <= hi + 8.0]
        tov = [e for e in evs if t0 <= _f(e["t"]) <= hi and (
            (e.get("type") == "steal" and e.get("team") == S._other(team))
            or (e.get("type") in ("turnover", "lost_ball", "bad_pass") and e.get("team") == team))]
        orb = [e for e in evs if e.get("type") == "off_rebound" and e.get("team") == team and t0 <= _f(e["t"]) <= t1]
        shooter = None
        if made:
            s = made[-1]
            outcome = "made"
            _tm, shooter = who(s.get("shooter"), team)
            detail = f"Made {int(s.get('points') or s.get('value') or 2)}" + (f" by {shooter}" if shooter and shooter != "Unidentified" else "")
            prov = "measured (rim); shooter inferred"
        elif fts:
            outcome, detail, prov = "ft", "Free throws", "inferred (free-throw situation)"
        elif tov:
            e = tov[0]
            outcome = "turnover"
            detail = "Turnover (steal)" if e.get("type") == "steal" else "Turnover (lost ball)"
            prov = "inferred (ball control changed)"
        elif mine:
            s = mine[-1]
            outcome = "miss"
            _tm, shooter = who(s.get("shooter"), team)
            nxt = poss[k + 1] if k + 1 < len(poss) else None
            reb = "defensive rebound" if nxt is not None and nxt.get("team") != team else "rebound"
            detail = f"Missed {s.get('value') or 2} + {reb}"
            prov = "inferred (ball tracking)"
        elif any(abs(pe - t1) < 6.0 for pe in period_ends):
            outcome, detail, prov = "period_end", "End of period", "inferred (detected break)"
        else:
            outcome, detail, prov = "unknown", "Not seen (ball out of view)", "–"
        if orb:
            detail += f" · {len(orb)} off. reb."
        out.append({"t0": round(t0, 2), "t1": round(t1, 2), "team": team, "duration_s": round(t1 - t0, 1),
                    "outcome": outcome, "detail": detail, "shooter": shooter, "points": sum(
                        int(s.get("points") or s.get("value") or 2) for s in made),
                    "fast_break": bool(p.get("fast_break")), "provenance": prov,
                    "moments": [{"t": round(t0, 2), "t1": round(t1, 2), "title": f"{detail}", "team": team}]})
        for x in made:
            used.add(id(x))
    # made baskets outside every possession of the scoring team (the ball control was not seen / mislabelled):
    # still one row each, so the timeline never loses points
    for s in shots:
        if s.get("outcome") != "made" or id(s) in used:
            continue
        team = s["team"]
        _tm, shooter = who(s.get("shooter"), team)
        t = float(s["t"])
        detail = f"Made {int(s.get('points') or s.get('value') or 2)}" + (f" by {shooter}" if shooter and shooter != "Unidentified" else "")
        out.append({"t0": round(max(0.0, t - 6.0), 2), "t1": round(_f(s.get("t_rim")) or t + 1.5, 2), "team": team,
                    "duration_s": None, "outcome": "made", "detail": detail + " (possession not detected)",
                    "shooter": shooter, "points": int(s.get("points") or s.get("value") or 2), "fast_break": False,
                    "provenance": "rim measured; possession unseen",
                    "moments": [{"t": round(max(0.0, t - 6.0), 2), "t1": round(t + 2.0, 2), "title": detail,
                                 "team": team}]})
    out.sort(key=lambda r: r["t0"])
    return out


def _possession_timeline_v6(st: dict, gs: dict, who: Any, raw: list[dict], shots: list[dict]) -> list[dict]:
    poss = S.merge_possessions(raw)
    evs = [e for e in gs.get("events") or [] if isinstance(e, dict) and _f(e.get("t")) is not None
           and (_f(e.get("confidence")) or 0.0) >= S.MIN_EVENT_CONF]
    out: list[dict] = []
    used: set = set()
    for k, p in enumerate(poss):
        team, t0, t1 = p.get("team"), _f(p.get("t0")), _f(p.get("t1"))
        if t0 is None or t1 is None or (team not in TEAMS and t1 - t0 < UNCLEAR_MIN_S):
            continue
        oc = p.get("outcome")
        ev = p.get("evidence") or {}
        mine = [s for s in shots if s["team"] == team and t0 - 0.5 <= s["t"] <= t1 + 0.5] if team in TEAMS else []
        made_ids = {s.get("basket_event") for s in p.get("shots") or [] if s.get("made") and s.get("basket_event")}
        made = [s for s in mine if s.get("outcome") == "made" and (s.get("basket_event") in made_ids or not made_ids)]
        shooter = None
        points = int(p.get("points") or 0)
        orb = int(p.get("off_rebounds") or 0)
        if team not in TEAMS:
            outcome, detail, prov = "unknown", "Team unclear (evidence does not decide)", "no usable evidence"
        elif oc == "made":
            outcome = "made"
            s = made[-1] if made else None
            _tm, shooter = who(s.get("shooter"), team) if s else (None, None)
            pts = points or (int(s.get("points") or s.get("value") or 2) if s else 2)
            detail = f"Made {pts}" + (f" by {shooter}" if shooter and shooter != "Unidentified" else "")
            prov = _provenance(ev, "shooter inferred")
        elif oc == "ft":
            outcome = "ft"
            detail = "Free throws" + (f" ({points} made)" if points else "")
            prov = "free-throw situation (formation)"
        elif oc == "turnover":
            outcome = "turnover"
            steal = any(e.get("type") == "steal" and e.get("team") == S._other(team) and t1 - 2.5 <= _f(e["t"]) <= t1 + 1.5
                        for e in evs)
            detail = "Turnover (steal)" if steal else "Turnover (lost ball / out of bounds)"
            prov = _provenance(ev, "ball control changed")
        elif oc == "missed":
            outcome = "miss"
            s = mine[-1] if mine else None
            if s is not None:
                _tm, shooter = who(s.get("shooter"), team)
            nxt = poss[k + 1] if k + 1 < len(poss) else None
            reb = "defensive rebound" if nxt is not None and nxt.get("team") not in (team, None) else "rebound"
            detail = f"Missed {(s or {}).get('value') or 2} + {reb}"
            prov = _provenance(ev, "rim measured")
        elif oc == "period_end":
            outcome, detail, prov = "period_end", "End of period", _provenance(ev, "detected break")
        elif oc in ("dead_ball", "foul"):
            outcome = "dead"
            detail = "Dead ball (foul / out of bounds / stoppage)" if oc == "dead_ball" else "Fouled (free throws for the other team)"
            prov = _provenance(ev, "detected dead ball")
        else:
            outcome, detail, prov = "unknown", "Not seen (ball out of view)", _provenance(ev, "–")
        if orb:
            detail += f" · {orb} off. reb."
        out.append({"t0": round(t0, 2), "t1": round(t1, 2), "team": team, "duration_s": round(t1 - t0, 1),
                    "outcome": outcome, "detail": detail, "shooter": shooter, "points": points,
                    "fast_break": bool(p.get("fast_break")), "provenance": prov, "confidence": p.get("confidence"),
                    "moments": [{"t": round(t0, 2), "t1": round(t1, 2), "title": f"{detail}", "team": team}]})
        for x in made:
            used.add(id(x))
    # a confirmed make that no possession of its team covers (should not happen with v6 - the make is a boundary):
    # still one row, so the timeline never loses points
    for s in shots:
        if s.get("outcome") != "made" or id(s) in used:
            continue
        team, t = s["team"], float(s["t"])
        if any(p.get("team") == team and _f(p["t0"]) - 0.5 <= t <= _f(p["t1"]) + 2.0 for p in poss):
            continue
        _tm, shooter = who(s.get("shooter"), team)
        detail = f"Made {int(s.get('points') or s.get('value') or 2)}" + (f" by {shooter}" if shooter and shooter != "Unidentified" else "")
        out.append({"t0": round(max(0.0, t - 6.0), 2), "t1": round(_f(s.get("t_rim")) or t + 1.5, 2), "team": team,
                    "duration_s": None, "outcome": "made", "detail": detail + " (possession not detected)",
                    "shooter": shooter, "points": int(s.get("points") or s.get("value") or 2), "fast_break": False,
                    "provenance": "rim measured; possession unseen",
                    "moments": [{"t": round(max(0.0, t - 6.0), 2), "t1": round(t + 2.0, 2), "title": detail,
                                 "team": team}]})
    out.sort(key=lambda r: r["t0"])
    return out


def _provenance(ev: dict, extra: str = "") -> str:
    """Readable evidence summary of a v6 possession: the strongest sources first."""
    names = {"make": "make (score bar / rim)", "shot": "shot at the basket", "holder": "ball holders",
             "location": "ball location", "rule": "inbound rule"}
    parts = [names[k] for k, v in sorted(ev.items(), key=lambda kv: -abs(float(kv[1] or 0.0)))
             if k in names and abs(float(v or 0.0)) >= 1.0]
    s = ", ".join(parts[:3]) if parts else "weak evidence"
    return f"{s}; {extra}" if extra else s


def shot_quality(shots: list[dict], geo: dict | None) -> dict:
    """Per team and zone: attempts, made, points, points per shot, and the expected points per shot (zone value x
    this game's FG % in the zone, both teams, shrunk to PRIOR_FG). Restricted area = within the no-charge radius
    (+0.5 m) of the basket. Located shots only; zones within 0.5 m of the 3-point line are counted as detected."""
    ra = float((geo or {}).get("no_charge_r") or 1.25) + 0.5
    def zone_of(s):
        z = s.get("zone")
        if z == "paint" and s.get("distance_m") is not None and float(s["distance_m"]) <= ra:
            return "restricted"
        return z
    loc = [s for s in shots if s.get("team") in TEAMS and zone_of(s) in dict(QUALITY_ZONES)]
    pooled = {}
    for z, _t in QUALITY_ZONES:
        zs = [s for s in loc if zone_of(s) == z]
        mk = sum(1 for s in zs if s.get("outcome") == "made")
        fg = (mk + PRIOR_FG * PRIOR_N) / (len(zs) + PRIOR_N)
        value = 3 if z.endswith("_3") else 2
        pooled[z] = {"fg_exp": round(fg, 3), "xpps": round(fg * value, 2), "value": value}
    out = {"zones": QUALITY_ZONES, "expected": pooled, "unlocated": {k: sum(
        1 for s in shots if s.get("team") == k and zone_of(s) not in dict(QUALITY_ZONES)) for k in TEAMS}}
    for k in TEAMS:
        rows = []
        for z, title in QUALITY_ZONES:
            zs = [s for s in loc if s["team"] == k and zone_of(s) == z]
            made = [s for s in zs if s.get("outcome") == "made"]
            pts = sum(int(s.get("points") or s.get("value") or 2) for s in made)
            rows.append({"zone": z, "title": title, "fga": len(zs), "fgm": len(made), "pts": pts,
                         "pps": round(pts / len(zs), 2) if zs else None, "xpps": pooled[z]["xpps"],
                         "moments": [m for m in (_moment(s.get("t"), s.get("title") or "Shot", k, s.get("t_rim"))
                                                 for s in zs) if m]})
        out[k] = rows
    return out


# ====================================================================================== player pages
def player_tracks(catalog: Any, team: str, label: str) -> set[str]:
    if catalog is None or not label.startswith("#"):
        return set()
    return set(catalog.tracks_of(f"num:{team}:{label[1:]}"))


def player_page(rep: dict, catalog: Any, replay: dict, team: str, label: str) -> dict:
    """Minutes, heat map (court metres binned, seconds per bin), shots and moments of one player."""
    tids = player_tracks(catalog, team, label)
    ts = replay.get("t") or []
    dt = (ts[-1] - ts[0]) / max(1, len(ts) - 1) if len(ts) > 1 else 0.1
    L = float((rep.get("court") or {}).get("length") or 28.0)
    W = float((rep.get("court") or {}).get("width") or 15.0)
    nx, ny = max(1, int(math.ceil(L / HEAT_BIN_M))), max(1, int(math.ceil(W / HEAT_BIN_M)))
    heat = [[0.0] * nx for _ in range(ny)]
    seen_s = 0.0
    for tr in replay.get("tracks") or []:
        if str(tr.get("id")) not in tids:
            continue
        for x, y in zip(tr.get("x") or [], tr.get("y") or []):
            if x is None or y is None:
                continue
            xm, ym = x / 100.0, y / 100.0
            if not (0 <= xm <= L and 0 <= ym <= W):
                continue
            heat[min(ny - 1, int(ym / HEAT_BIN_M))][min(nx - 1, int(xm / HEAT_BIN_M))] += dt
            seen_s += dt
    row = next((r for r in rep["box"][team]["rows"] if r["player"] == label), None)
    shots = [s for s in rep.get("shots") or [] if s.get("team") == team and str(s.get("shooter")) in tids]
    return {"team": team, "player": label, "tracks": sorted(tids, key=lambda t: int(t) if t.isdigit() else 0),
            "row": row, "heat": heat, "bin_m": HEAT_BIN_M, "court": rep.get("court"), "tracked_s": round(seen_s, 1),
            "shots": shots}


def fmt_min(sec: float | None) -> str:
    if sec is None:
        return "–"
    m, sec_i = divmod(int(round(sec)), 60)
    return f"{m}:{sec_i:02d}"


def cell_text(r: dict, key: str) -> str:
    """Box-score cell text (FG as made-attempted)."""
    if key == "min":
        return fmt_min(r.get("min_s")) if r.get("min_s") else "–"
    if key == "fg":
        return f"{r.get('fgm', 0)}-{r.get('fga', 0)}"
    if key == "fg3":
        return f"{r.get('fg3m', 0)}-{r.get('fg3a', 0)}"
    if key == "ft":
        if r.get("ftm") is None:
            return "–"
        return f"{r['ftm']}" if r.get("fta") is None else f"{r['ftm']}-{r['fta']}"
    if key == "pf":
        return "–" if r.get("pf") is None else str(r["pf"])
    if key == "pm":
        v = r.get("pm")
        return "–" if v is None else f"{v:+d}"
    v = r.get(key)
    return "–" if v is None else str(v)
