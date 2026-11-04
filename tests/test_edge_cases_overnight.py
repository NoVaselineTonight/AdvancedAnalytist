"""Edge cases (empty game, no ball, one team, no score bar, sheet with no matching numbers, zero-length periods,
players without jersey numbers) for coach / coach_export / stats / scorebar / sheet_align."""

import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from replay import coach as C  # noqa: E402
from replay import coach_export as E  # noqa: E402
from replay import scorebar as sb  # noqa: E402
from replay import sheet_align as sa  # noqa: E402
from replay import stats as S  # noqa: E402

COURT = {"length": 28.0, "width": 15.0, "baskets": {"L": [1.575, 7.5], "R": [26.425, 7.5]},
         "free_throw_line_x": {"L": 5.8, "R": 22.2}, "three_point_radius_m": 6.75}


def _ev(typ, t, team=None, conf=0.8, **details):
    return {"id": f"gs-{typ}-{int(t * 10)}", "type": typ, "t": t, "team": team, "confidence": conf,
            "duration": 0.0, "details": details, "title": f"{typ} {team}"}


def _gs(**kw):
    base = {"court": COURT, "teams": {"A": "Orange", "B": "Purple"}, "time_range_s": [0.0, 100.0], "breaks": [],
            "possessions": [], "shots": [], "events": [], "basket_annotations": {}}
    base.update(kw)
    return base


def _empty_root(tmp_path):
    root = tmp_path / "empty"
    (root / "results").mkdir(parents=True)
    return root


# ============================================================================== stats
def test_stats_compute_none_and_empty():
    for gs in (None, {}, _gs()):
        st = S.compute(gs)
        assert st["game"]["A"]["points"] == 0 and st["game"]["B"]["points"] == 0
        assert st["shots"] == [] and st["score"]["made_baskets"] == 0
        assert S.summary_lines(st)
        assert isinstance(S.tables(st), dict)


def test_stats_periods_zero_length_and_missing_range():
    p = S.periods({})
    assert len(p) == 1 and p[0]["t0"] == p[0]["t1"] == 0.0
    p = S.periods(_gs(time_range_s=[50.0, 50.0]))
    assert len(p) == 1 and p[0]["t1"] >= p[0]["t0"]
    p = S.periods(_gs(breaks=[{"t0": 0.0, "t1": 10.0, "kind": "quarter_break"}]))
    assert all(x["t1"] >= x["t0"] for x in p)
    p = S.periods(_gs(breaks=[{"t0": 40.0, "kind": "quarter_break"}]))
    assert all(x["t1"] >= x["t0"] for x in p)


def test_stats_compute_zero_length_period_no_crash():
    st = S.compute(_gs(time_range_s=[10.0, 10.0]))
    assert len(st["periods"]) == 1
    S.summary_lines(st)


def test_stats_no_ball_only_events():
    st = S.compute(_gs(possessions=[], events=[_ev("turnover", 5.0, "A")]))
    assert st["game"]["A"]["points"] == 0
    assert S.summary_lines(st)


def test_stats_only_one_team_detected():
    poss = [{"id": "p0", "team": "A", "t0": 1.0, "t1": 9.0, "duration_s": 8.0, "fast_break": False, "confidence": 0.9},
            {"id": "p1", "team": "A", "t0": 20.0, "t1": 30.0, "duration_s": 10.0, "fast_break": False,
             "confidence": 0.9}]
    gs = _gs(possessions=poss, basket_annotations={"b1": {"kind": "field_goal", "points_guess": 2, "team": "A",
                                                          "label": "Made 2"}})
    st = S.compute(gs, [{"id": "b1", "type": "basket", "t": 5.0, "team": "A", "basket": "L", "confidence": 0.9}])
    assert st["game"]["B"]["points"] == 0 and st["game"]["A"]["points"] == 2
    assert st["game"]["B"]["fga"] == 0
    for k in ("A", "B"):
        S.zone_table(st["shots"], team=k)
    assert S.summary_lines(st)


def test_stats_shot_without_team_is_unattributed_not_crash():
    gs = _gs(shots=[{"t_release": 5.0, "t_rim": 5.8, "basket": "L", "team": None, "free_throw": False,
                     "made": True, "basket_event": "b1"}],
             basket_annotations={"b1": {"kind": "field_goal", "points_guess": 2, "team": None, "label": "Made"}})
    st = S.compute(gs)
    assert st["game"]["A"]["points"] + st["game"]["B"]["points"] == 0


def test_stats_zone_and_geometry_edge():
    geo = S.court_geometry(None)
    assert isinstance(S.classify_zone(None, None, None, geo), dict)
    S.classify_zone(0.0, 0.0, "L", geo)
    S.classify_zone(-5.0, 99.0, "R", geo)
    assert S.zone_table([], team="A") is not None
    assert S.filter_shots([], team="A") == []


def test_stats_fmt_clock_edges():
    assert S.fmt_clock(None) is not None
    S.fmt_clock(0.0)
    S.fmt_clock(-3.0)
    S.fmt_clock(36000.0)


def test_stats_reconcile_no_truth():
    r = S.reconcile({"A": 0, "B": 0}, None, {"A": "A", "B": "B"})
    assert isinstance(r, dict)


def test_stats_merge_possessions_empty():
    assert S.merge_possessions([]) == []


def test_stats_load_inputs_empty_folder(tmp_path):
    inp = S.load_inputs(_empty_root(tmp_path))
    assert inp["replay_events"] == [] and inp["truth"] is None
    assert S.compute(inp["game_state"])["game"]["A"]["points"] == 0


def test_stats_write_exports_empty(tmp_path):
    paths = S.write_exports(S.compute(None), tmp_path / "out")
    assert paths and all(p.exists() for p in paths)


# ============================================================================== coach
def test_coach_build_empty_game(tmp_path):
    rep = C.build(_empty_root(tmp_path))
    assert set(rep["box"]) == {"A", "B"}
    for k in "AB":
        assert rep["box"][k]["rows"] == [] and rep["box"][k]["total"]["pts"] == 0
    assert rep["score"] == {"A": 0, "B": 0} and rep["runs"] == []
    assert tuple(rep["momentum"][0][1:]) == (0, 0)
    assert E.summary_text(rep)


def test_coach_runs_momentum_empty_and_single():
    assert C.runs([]) == []
    assert C.momentum([]) == [(0.0, 0, 0)]
    assert len(C.momentum([], [5.0, 5.0])) <= 2
    assert C.runs([{"t": 1.0, "team": "A", "points": 2}]) == []
    assert C.runs([{"t": 1.0, "team": "A", "points": 9}])[0]["points"] == 9
    assert C.momentum([{"t": 3.0, "team": "B", "points": 3}], None)[-1] == (3.0, 0, 3)


def test_coach_runs_simultaneous_baskets_other_team():
    bk = [{"t": 5.0, "team": "A", "points": 3}, {"t": 5.0, "team": "B", "points": 3}]
    assert isinstance(C.runs(bk), list)


def test_coach_unattributed_baskets_make_no_run_or_momentum(tmp_path):
    import json
    root = _empty_root(tmp_path)
    evs = [{"id": f"b{i}", "type": "basket", "t": float(i), "team": None, "basket": "L", "confidence": 0.9}
           for i in range(1, 5)]
    (root / "results" / "replay_data.json").write_text(json.dumps({"events": evs}))
    rep = C.build(root)
    assert rep["runs"] == [] and rep["score"] == {"A": 0, "B": 0}
    assert all(tuple(m[1:]) == (0, 0) for m in rep["momentum"])


def test_coach_helpers_none():
    assert C._f(None) is None and C._f("x") is None and C._f("2.5") == 2.5
    assert C._moment(None, "x", "A") is None
    assert C.fmt_min(0) == "0:00"
    assert C.cell_text({}, "fg") == "0-0" and C.cell_text({}, "pts") == "–"
    assert C.cell_text({"ftm": 2}, "ft") == "2" and C.cell_text({"pm": 0}, "pm") == "+0"
    assert C.cell_text({"min_s": 0.0}, "min") == "–"


def test_coach_period_table_zero_length_and_no_marks(tmp_path):
    root = _empty_root(tmp_path)
    for tr in (None, [0.0, 0.0], [5.0, 5.0]):
        pt = C.period_table({"time_range_s": tr, "periods": [], "shots": []}, root)
        assert all(r["t1"] >= r["t0"] for r in pt["rows"])
    st = {"time_range_s": [0.0, 0.0], "periods": [{"t0": 0.0, "t1": 0.0, "label": "Whole", "w0": 0.0, "w1": 0.0}],
          "shots": [{"team": "A", "t": 0.0, "outcome": "made", "value": 2}]}
    pt = C.period_table(st, root)
    assert pt["rows"][0]["A"]["fga"] == 0  # half-open [0,0) holds nothing


def test_coach_period_table_bad_marks(tmp_path):
    root = _empty_root(tmp_path)
    (root / "timeline.json").write_text(json.dumps({"tracks": [{"marks": [{"t": None, "title": "Q1"},
                                                                          {"t": 3, "title": None}]}]}))
    pt = C.period_table({"time_range_s": [0.0, 10.0], "periods": [], "shots": []}, root)
    assert isinstance(pt["rows"], list)
    (root / "timeline.json").write_text("not json")
    assert isinstance(C.period_table({"time_range_s": [0.0, 10.0], "periods": [], "shots": []}, root), dict)


def test_coach_players_without_jersey_numbers(tmp_path):
    root = _empty_root(tmp_path)
    shots = [{"t": 5.0, "team": "A", "outcome": "made", "value": 2, "shooter": "7", "title": "Made 2", "zone": "paint"},
             {"t": 8.0, "team": "A", "outcome": "missed", "value": 3, "shooter": None, "title": "Miss 3"}]
    st = S.compute(None)
    st["shots"] = shots
    st["game"]["A"].update(points=2, fgm=1, fga=2)
    rep = C.build(root, st=st)
    rows = rep["box"]["A"]["rows"]
    assert rows, "shots by unnumbered players still show in the box score"
    assert sum(r["pts"] for r in rows) == 2 and sum(r["fga"] for r in rows) == 2
    assert E.box_rows(rep, "A")
    assert E.summary_text(rep)


def test_coach_lineup_rows_without_numbers_not_a_person(tmp_path):
    root = _empty_root(tmp_path)
    st = S.compute(None)
    st["lineups"] = {"available": True, "rows": [{"team": "A", "players": ["Unidentified", "Unidentified"],
                                                    "seconds": 30.0, "plus_minus": 2, "stints": []}]}
    rep = C.build(root, st=st)
    assert all(r["player"] != "Unidentified" or r["min_s"] == 0 for r in rep["box"]["A"]["rows"])


def test_coach_event_with_unknown_team(tmp_path):
    root = _empty_root(tmp_path)
    gs = _gs(events=[_ev("steal", 5.0, None), _ev("off_rebound", 6.0, "X"), _ev("pass", 7.0, "A")])
    (root / "results" / "game_state.json").write_text(json.dumps(gs))
    rep = C.build(root, st=S.compute(gs))
    assert rep["score"] == {"A": 0, "B": 0}


def test_coach_only_one_team_has_players(tmp_path):
    root = _empty_root(tmp_path)
    st = S.compute(None)
    st["shots"] = [{"t": 5.0, "team": "A", "outcome": "made", "value": 2, "shooter": None, "title": "Made 2"}]
    st["game"]["A"].update(points=2, fgm=1, fga=1)
    rep = C.build(root, st=st)
    assert rep["box"]["B"]["rows"] == [] and rep["box"]["B"]["total"]["pts"] == 0
    assert "–" in E.summary_text(rep)


def test_coach_merge_scoresheet_without_sheet(tmp_path):
    rep = {"box": {"A": {"rows": [], "total": {}}, "B": {"rows": [], "total": {}}}, "score": {"A": 0, "B": 0}}
    assert C.merge_scoresheet(rep, _empty_root(tmp_path)) is None


def test_coach_shot_quality_edge():
    sq = C.shot_quality([], None)
    assert sq["A"] and all(r["fga"] == 0 and r["pps"] is None for r in sq["A"])
    sq = C.shot_quality([{"team": "A", "outcome": "made"}, {"team": "Z", "zone": "paint"}], {})
    assert all(r["fga"] == 0 for r in sq["A"])


def test_coach_possession_timeline_empty():
    assert C.possession_timeline(S.compute(None), {}, C._Who(None)) == []


def test_coach_player_page_no_tracks():
    rep = {"box": {"A": {"rows": []}, "B": {"rows": []}}, "court": None, "shots": []}
    pg = C.player_page(rep, None, {}, "A", "#7")
    assert pg["tracked_s"] == 0.0 and pg["row"] is None
    pg = C.player_page(rep, None, {"t": [0.0], "tracks": [{"id": 1, "x": [None], "y": [None]}]}, "A", "Unidentified")
    assert pg["tracked_s"] == 0.0


# ============================================================================== coach_export
def _min_rep(**over):
    z = {"player": "Team", "min_s": None, "pts": 0, "fgm": 0, "fga": 0, "fg3m": 0, "fg3a": 0, "ftm": 0, "fta": 0,
         "oreb": 0, "dreb": 0, "ast": 0, "stl": 0, "tov": 0, "pm": None, "moments": {}}
    rep = {"teams": {"A": {"name": "Home", "color": None}, "B": {"name": "Away", "color": None}},
           "score": {"A": 0, "B": 0}, "box": {"A": {"rows": [], "total": dict(z)}, "B": {"rows": [], "total": dict(z)}},
           "compare": [], "runs": [], "period_table": None, "sheet": None}
    rep.update(over)
    return rep


def test_export_empty_report(tmp_path):
    rep = _min_rep()
    for k in "AB":
        rows = E.box_rows(rep, k)
        assert rows[0][0] == "Player" and rows[-1][0] == "Team"
    assert E.compare_rows(rep)[0][0] == "Stat"
    paths = E.write_csv(rep, tmp_path / "g.csv")
    assert paths and all(p.exists() for p in paths)
    txt = E.summary_text(rep, "T")
    assert "Home 0 – 0 Away" in txt and "Key runs" not in txt


def test_export_period_of_edges():
    assert E.period_of({}, None) == "" and E.period_of({}, 5.0) == ""
    rep = {"period_table": {"rows": [{"label": "P1", "t0": 0.0, "t1": 0.0}, {"label": "P2", "t0": 0.0, "t1": 10.0}]}}
    assert E.period_of(rep, 0.0) == "P2"  # zero-length first period never matches
    assert E.period_of(rep, 10.0) == ""
    rep = {"period_table": {"rows": [{"label": "P1", "t0": None, "t1": None}, {"label": "P2", "t0": 0, "t1": 5}]}}
    assert E.period_of(rep, 1.0) == "P2"


def test_export_clock_edges():
    assert E.clock(0) == "0:00" and E.clock(-4) == "0:00" and E.clock(3600) == "1:00:00" and E.clock(59.6) == "1:00"


def test_export_safe_text():
    assert E._safe_text("=1+1").startswith("'")
    assert E._safe_text("") == ""


def test_export_unnumbered_players_summary_and_rows():
    rows = [{"player": "Unidentified", "min_s": 0.0, "pts": 4, "fgm": 2, "fga": 3, "fg3m": 0, "fg3a": 0, "oreb": 0,
             "dreb": 0, "ast": 0, "stl": 0, "tov": 0, "pm": None, "moments": {}}]
    rep = _min_rep()
    rep["box"]["A"]["rows"] = rows
    rep["box"]["A"]["total"]["pts"] = 4
    rep["score"]["A"] = 4
    assert "Unidentified" not in E.summary_text(rep)  # not a person
    assert E.box_rows(rep, "A")[1][0] == "Unidentified players"


def test_export_summary_run_without_period_table():
    rep = _min_rep(runs=[{"team": "A", "t0": 5.0, "t1": 20.0, "points": 8, "title": "8-0 run", "moments": []}])
    txt = E.summary_text(rep)
    assert "8-0" in txt and "0:05" in txt


def test_export_zero_attempts_percent_blank():
    rep = _min_rep()
    rep["box"]["A"]["rows"] = [{"player": "#4", "min_s": 0.0, "pts": 0, "fgm": 0, "fga": 0, "fg3m": 0, "fg3a": 0,
                                "oreb": 0, "dreb": 0, "ast": 0, "stl": 0, "tov": 0, "pm": 0, "moments": {}}]
    rows = E.box_rows(rep, "A")
    h = {n: i for i, n in enumerate(rows[0])}
    assert rows[1][h["FG%"]] == ""


def test_export_write_xlsx_empty(tmp_path):
    if not E.has_xlsx():
        pytest.skip("no xlsx writer")
    p = E.write_xlsx(_min_rep(), tmp_path / "e.xlsx")
    assert p.exists() and p.stat().st_size > 0


# ============================================================================== sheet_align
def test_align_empty_sides():
    segs = sa.segments([], 100.0)
    al = sa.align([], [], segs)
    assert al["det_to_sheet"] == [] and al["sheet_to_det"] == [] and al["score"] == 0.0
    al = sa.align([], [{"t": 5, "team": "A", "kind": "fg"}], segs)
    assert al["det_to_sheet"] == [None]
    al = sa.align([{"team": "A", "points": 2, "scorer": "5", "kind": "fg2", "period": 1}], [], segs)
    assert al["sheet_to_det"] == [None]


def test_align_no_matching_period():
    segs = sa.segments([], 100.0)
    al = sa.align([{"team": "A", "points": 2, "scorer": "5", "kind": "fg2", "period": 9}],
                  [{"t": 5, "team": "A", "kind": "fg"}], segs)
    assert al["sheet_to_det"] == [None] and al["det_to_sheet"] == [None]


def test_align_wrong_team_everywhere_is_not_forced():
    segs = sa.segments([], 100.0)
    sheet = [{"team": "A", "points": 2, "scorer": "5", "kind": "fg2", "period": 1}] * 3
    dets = [{"t": 10, "team": "B", "kind": "fg"}, {"t": 20, "team": "B", "kind": "fg"}]
    assert sa.align(sheet, dets, segs)["det_to_sheet"] == [None, None]


def test_align_detection_team_unknown_and_no_kind():
    segs = sa.segments([], 100.0)
    al = sa.align([{"team": "A", "points": 2, "scorer": "5", "kind": "fg2", "period": 1}],
                  [{"t": 10, "team": None, "kind": None}], segs)
    assert len(al["det_to_sheet"]) == 1


def test_segments_zero_length_and_odd_breaks():
    assert sa.segments([], 0.0) == [(0.0, 0.0, {1, 2, 3, 4})]
    segs = sa.segments([{"kind": "half_time", "t0": 50.0, "t1": 50.0}], 100.0)
    assert [s[2] for s in segs] == [{1, 2}, {3, 4}]
    assert all(b >= a for a, b, _ in segs)
    segs = sa.segments([{"kind": "timeout", "t0": 5, "t1": 9}], 100.0)  # irrelevant kinds ignored
    assert segs == [(0.0, 100.0, {1, 2, 3, 4})]
    segs = sa.segments([{"kind": "quarter_break", "t0": 10, "t1": 20}, {"kind": "quarter_break", "t0": 30, "t1": 40},
                        {"kind": "quarter_break", "t0": 50, "t1": 60}], 100.0)
    assert [s[2] for s in segs] == [{1}, {2}, {3}, {4}]


def test_segments_half_time_after_end():
    assert sa.segments([{"kind": "half_time", "t0": 500.0, "t1": 600.0}], 100.0)


def test_estimate_times_edges():
    segs = sa.segments([], 100.0)
    assert sa.estimate_times(0, [], [], [], segs) == []
    out = sa.estimate_times(2, [None, None], [], [1, 1], segs)  # nothing aligned, no detections at all
    assert len(out) == 2 and all(0.0 <= t <= 100.0 for t in out)
    out = sa.estimate_times(3, [None, 0, None], [40.0], [1, 1, 1], segs)
    assert out[1] == 40.0 and out[0] < 40.0 < out[2]


def test_estimate_times_unknown_period_falls_back():
    assert len(sa.estimate_times(1, [None], [10.0], [7], sa.segments([], 100.0))) == 1


# ============================================================================== scorebar
def test_scorebar_smooth_scores_empty_and_all_none():
    cfg = sb.ScorebarConfig()
    assert sb.smooth_scores([], cfg) == []
    assert sb.smooth_scores([(0, None), (1, None)], cfg) == [(0, None), (1, None)]


def test_scorebar_smooth_periods_clock_edges():
    assert sb.smooth_periods([]) == []
    assert sb.smooth_periods([(0, None), (1, None)]) == [None, None]
    assert sb.smooth_clock([], []) == []
    assert sb.smooth_clock([(0.0, None), (1.0, None)], [None, None]) == [(None, None), (None, None)]
    assert len(sb.smooth_clock([(0.0, 300.0)], [1])) == 1
    sb.smooth_clock([(1.0, 300.0), (1.0, 299.0), (1.0, 298.0)], [1, 1, 1])  # duplicate timestamps


def test_scorebar_period_value_parse():
    assert sb._period_value("Q3") == 3 and sb._period_value("OT") == 5 and sb._period_value("OT2") == 6
    assert sb._period_value("") is None and sb._period_value("0") is None and sb._period_value("xyz") is None


def test_scorebar_clock_at_and_played_seconds_no_clock():
    assert sb.clock_at({}, 5.0) == (None, None, None)
    assert sb.clock_at({"clock": []}, 5.0) == (None, None, None)
    assert sb.clock_at({"clock": [[10.0, 1, 300.0, True]]}, 5.0) == (None, None, None)
    assert sb.played_seconds({}, 0.0, 100.0) is None
    assert sb.played_seconds({"clock": [[1.0, 1, 300.0, None]]}, 0.0, 10.0) is None
    assert sb.played_seconds({"clock": [[1.0, 1, 300.0, True], [2.0, 1, 299.0, True]]}, 5.0, 5.0) is None


def test_scorebar_summary_no_samples():
    assert sb.summary({}) == {"sure": False}
    assert sb.summary({"samples": [[0.0, 1, 2, None, ""]] * 3}) == {"sure": False}


def test_scorebar_summary_only_one_side_read():
    smp = [[float(t), 5, None, None, ""] for t in range(10)]
    assert sb.summary({"samples": smp})["sure"] is False


def test_scorebar_clock_text_edges():
    assert sb.clock_text(0)
    sb.clock_text(0.5)
    sb.clock_text(3599.9)


def test_scorebar_load_result_missing(tmp_path):
    assert sb.load_result(tmp_path) is None
    assert sb.load_result(tmp_path / "does-not-exist") is None
    (tmp_path / sb.CACHE).write_text("{broken")
    assert sb.load_result(tmp_path) is None


def test_scorebar_load_or_compute_no_video(tmp_path):
    assert sb.load_or_compute(tmp_path, None) is None


def test_scorebar_pin_candidate_edges():
    assert sb.pin_candidate([], None) is None
    assert sb.pin_candidate([], {}) is None


def test_scorebar_locate_blocks_flat_frames():
    flat = [np.zeros((120, 200, 3), np.uint8) for _ in range(10)]
    blocks, _ = sb.locate_blocks(flat, sb.ScorebarConfig())
    assert blocks == []


class _NoRec:
    device = "cpu"

    def __call__(self, *a, **k):
        raise AssertionError("recognizer must not run without a bar")


def test_scorebar_reader_no_overlay_gives_no_bar_result():
    rd = sb.ScorebarReader(sb.ScorebarConfig(), _NoRec())
    flat = np.full((120, 200, 3), 90, np.uint8)
    for k in range(600):  # retries later in the video (5 and 10 min after the first miss), then gives up
        rd.add(k * 2.0, flat)
    res = rd.result()
    assert rd.no_bar
    assert res["summary"].get("sure") is False
    assert not (res.get("timeline") or [])


def test_scorebar_reader_result_with_zero_frames():
    assert isinstance(sb.ScorebarReader(sb.ScorebarConfig(), _NoRec()).result(), dict)


def test_scorebar_read_frames_empty():
    res = sb.read_frames([], sb.ScorebarConfig(), _NoRec())
    assert isinstance(res, dict) and res.get("summary", {}).get("sure") is False


def test_scorebar_classify_no_blocks():
    cls = sb.classify([])
    assert cls["score"] == [] and cls["clock"] is None and cls["period"] is None


def test_scorebar_classify_blocks_without_reads():
    cls = sb.classify([sb.Block(0, 0, 40, 20), sb.Block(50, 0, 90, 20)])
    assert cls["score"] == []


# ============================================================================== scoresheet with no matching numbers
def test_coach_scoresheet_numbers_not_in_video(tmp_path, monkeypatch):
    from fake_project import make_project

    from replay import scoresheet as SS
    root = make_project(tmp_path / "p")
    rep0 = C.build(root)
    video_nums = {r["player"] for k in "AB" for r in rep0["box"][k]["rows"]}
    num = next(str(n) for n in range(80, 99) if f"#{n}" not in video_nums)
    csv = tmp_path / "sheet.csv"
    csv.write_text(f"# home: Kouvot\n# away: Honka\n# final: 12-8\nteam,number,name,points,fouls,ft\n"
                   f"A,{num},Nobody Known,12,1,2\nB,{int(num) + 1},Other Nobody,8,2,0\n", encoding="utf-8")
    SS.import_into(root, csv)
    monkeypatch.setattr(SS, "crosscheck", lambda root, sheet=None, **k: {
        "team_map": {"A": "A", "B": "B"}, "full_game": True, "sheet_window": [], "team_map_source": "test",
        "summary": "", "alignment": "high"})
    rep = C.build(root)
    a = {r["player"]: r for r in rep["box"]["A"]["rows"]}
    assert a[f"#{num}"].get("sheet_only") and a[f"#{num}"]["pts"] == 12
    # the video's own tracked players keep their tracking numbers
    for r in rep0["box"]["A"]["rows"]:
        assert a[r["player"]]["fga"] == r["fga"]
    assert E.summary_text(rep)
    assert E.box_rows(rep, "A")


def test_coach_scoresheet_garbage_file_does_not_break_build(tmp_path):
    from fake_project import make_project
    root = make_project(tmp_path / "p")
    for name in ("scoresheet.json", "scoresheet.csv"):
        (root / name).write_text("{{{ not a sheet", encoding="utf-8")
    rep = C.build(root)
    assert set(rep["box"]) == {"A", "B"}
