"""Column documentation for the simple data export (README_FOR_AI.md).

The README tables for possessions.csv, passes.csv, shots.csv and the stats_*.csv files are generated from the SAME
header lists the CSV writers use (``csv_column_docs``), so they cannot drift. A column without an entry here is a
failure of tests/test_export_docs.py.
"""

from __future__ import annotations

POSS_DOCS = {
    "possession_id": "ID of the possession (also in game_state.json -> possessions)",
    "team": "`A` or `B`: the team that controls the ball",
    "team_name": "readable team name",
    "t0": "start, VIDEO seconds", "t1": "end, VIDEO seconds",
    "duration_s": "t1 - t0 in video seconds (includes dead time inside the possession; NOT game-clock time)",
    "attacks_basket": "`L` or `R`: the basket this team attacks (empty = not known)",
    "start": "how the possession began: `after shot` (rebound), `after made basket`, `after free throws`, "
             "`after dead ball`, `steal`, `lost_ball`, `out_of_bounds_or_violation`, `change` (unexplained change "
             "of control) or `start` (first possession)",
    "passes": "number of passes detected inside the possession (count)",
    "fast_break": "1 = classified as a fast break, 0 = no",
    "confidence": "0..1 reliability of this possession (below ~0.5 treat as uncertain)",
}
PASS_DOCS = {
    "t_release": "VIDEO seconds when the ball left the passer", "t_catch": "VIDEO seconds when the receiver caught it",
    "team": "`A` or `B` (the passing team)", "passer": "track_id of the passer (not a player identity)",
    "receiver": "track_id of the receiver",
    "passer_name": "`#number` of the passer when a jersey number was read, else empty",
    "receiver_name": "`#number` of the receiver when read, else empty",
    "from_x": "court x of the passer when releasing, metres", "from_y": "court y of the passer, metres",
    "to_x": "court x of the receiver when catching, metres", "to_y": "court y of the receiver, metres",
    "length_m": "straight distance from release to catch point, metres",
    "air_s": "time in the air (t_catch - t_release), seconds",
    "type": "flight shape: `direct`, `lob` (high arc), `bounce`, `long` (> 12 m, flight not seen) or `unseen`",
    "confidence": "0..1 reliability of this pass",
}
SHOT_DOCS = {
    "t_release": "VIDEO seconds of the shot release (ESTIMATED from the ball track)",
    "t_rim": "VIDEO seconds when the ball reached the rim (empty if not seen)",
    "team": "`A` or `B` (the shooting team)", "basket": "`L` or `R`: the basket shot at",
    "shooter": "track_id of the shooter (the holder at release; not a player identity)",
    "shooter_name": "`#number` when a jersey number was read, else empty",
    "x": "court x of the shooter at release, metres (camera court system, not mirrored to one half)",
    "y": "court y of the shooter at release, metres",
    "distance_m": "distance from (x, y) to the basket centre, metres",
    "value_guess": "2 or 3: ESTIMATE of the shot value from the position (see 'Shot zones')",
    "outcome": "`made`, `missed` or `unknown`",
    "rebound_team": "team (`A`/`B`) that controlled the ball ~1.5 s after a missed shot (empty = unknown)",
    "confidence": "0..1 reliability of this shot detection",
}
STATS_SHOT_DOCS = {
    "id": "shot id (`s0`, `s1`, ... in time order)",
    "t": "VIDEO seconds of the shot (release; the basket time for 'made basket only' rows)",
    "time": "`mm:ss.s` of t (video time)", "team": "`A` or `B`", "team_name": "readable team name",
    "basket": "`L` or `R`: the basket shot at", "x_m": "court x of the shot, metres (empty = position unknown)",
    "y_m": "court y of the shot, metres", "distance_m": "distance to the basket centre, metres",
    "zone": "`paint`, `mid_range`, `corner_3`, `above_break_3` or `unknown` (definitions in 'Shot zones')",
    "zone_uncertain": "1 = within the 3-point margin of the line (2 or 3 is a guess) or the position is unknown",
    "value": "shot value 2 or 3 (3 = beyond the line, estimated; for made baskets the points of the basket)",
    "outcome": "`made`, `missed` or `unknown`", "points": "points scored by this shot (0 if missed/unknown)",
    "shooter": "track_id of the shooter (empty = unknown)",
    "shooter_name": "`#number` when read, else empty",
    "confidence": "0..1 reliability",
    "source": "`ball_events` (ball-event layer), `game_state` (rules layer) or `made basket only (attempt not "
              "seen)` (a detected basket with no matching attempt; no position, zone `unknown`)",
    "basket_event": "event_id in events.csv of the matching made basket (empty for misses)",
}
STATS_ZONE_DOCS = {
    "team": "`A` or `B`", "team_name": "readable team name", "zone": "shot zone (see 'Shot zones')",
    "fgm": "field goals made in the zone (count)",
    "fga": "field-goal attempts in the zone (count; only attempts with outcome made/missed)",
    "fg_pct": "fgm / fga as a fraction 0..1 (empty if fga = 0); meaningless for small fga (< 10)",
}
STATS_LINEUP_DOCS = {
    "team": "`A` or `B`", "team_name": "readable team name",
    "players": "space-separated jersey numbers (`#7 #9 ...`) of the 5 on court; `(no number)` = slot whose number "
               "was not read",
    "identified_players": "how many of the 5 have a read number (0..5)",
    "duplicate_numbers": "1 = the same number appears twice (two slots, one read): uncertain lineup",
    "minutes": "GAME-CLOCK minutes this 5-man lineup played (NOT video time; method in stats.json -> lineups.clock)",
    "stints": "number of separate spells on court", "pts_for": "points scored by the team while on court",
    "pts_against": "points conceded while on court", "plus_minus": "pts_for - pts_against",
    "poss_for": "team possessions that began while on court", "poss_against": "opponent possessions in that time",
    "off_rating_per100": "100 x pts_for / poss_for", "def_rating_per100": "100 x pts_against / poss_against",
    "net_rating": "off_rating_per100 - def_rating_per100", "first_t": "VIDEO seconds of the first stint",
}
STATS_TEAM_EXTRA = {
    "scope": "`game` (whole analysed range) or `period` (one detected quarter/half)",
    "period": "period label (empty for `game`)", "t0": "start of the range, VIDEO seconds",
    "t1": "end of the range, VIDEO seconds", "team": "`A` or `B`", "team_name": "readable team name",
    "ft_pct": "ftm / fta as a fraction 0..1",
}

EVENT_TYPE_DOCS = {
    "basket": "made basket seen by the basket detector (source `auto`); the only type that moves score_A/score_B",
    "shot": "field-goal attempt", "free_throw": "one free throw", "free_throws": "a free-throw series",
    "off_rebound": "offensive rebound", "def_rebound": "defensive rebound", "rebound": "rebound, team unknown",
    "turnover": "turnover", "steal": "steal (ball won from the other team)",
    "lost_ball": "ball lost without a clear steal", "bad_pass": "intercepted / failed pass",
    "pass": "completed pass", "fast_break": "fast break", "half_court": "half-court set", "inbound": "inbound",
    "jump_ball": "jump ball", "timeout": "time-out / stoppage", "stoppage": "stoppage",
    "quarter_break": "quarter break", "half_time": "half time",
}


def stats_team_docs() -> dict[str, str]:
    """stats_team.csv docs: the stat columns come from replay.stats.STAT_ROWS (title + kind)."""
    from replay import stats as S
    out = dict(STATS_TEAM_EXTRA)
    units = {"int": "count", "pct": "fraction 0..1", "float": "number", "time": "seconds"}
    for key, title, kind in S.STAT_ROWS:
        out[key] = f"{title.strip()} ({units.get(kind, kind)})"
    out["time_of_possession_s"] = "time the team had the ball, seconds of VIDEO time (sum of possession durations)"
    out["possessions"] = "number of team possessions (game-state engine, possessions.csv)"
    out["possessions_est"] = "box-score estimate FGA - OREB + TOV + 0.44 x FTA (a cross-check of `possessions`)"
    return out


def csv_column_docs() -> dict[str, tuple[list[str], dict[str, str]]]:
    """{file: (header columns, {column: meaning})}. The headers are the very lists the writers use."""
    from replay import ball_events as BE
    from replay import data_export as DE
    from replay import stats as S
    return {
        "possessions.csv": (DE.POSS_TABLE_COLUMNS, POSS_DOCS),
        "passes.csv": (BE.PASS_COLUMNS, PASS_DOCS),
        "shots.csv": (BE.SHOT_COLUMNS, SHOT_DOCS),
        "stats_team.csv": (S.TEAM_COLUMNS, stats_team_docs()),
        "stats_shots.csv": (S.SHOT_COLUMNS, STATS_SHOT_DOCS),
        "stats_zones.csv": (S.ZONE_COLUMNS, STATS_ZONE_DOCS),
        "stats_lineups.csv": (S.LINEUP_COLUMNS, STATS_LINEUP_DOCS),
    }
