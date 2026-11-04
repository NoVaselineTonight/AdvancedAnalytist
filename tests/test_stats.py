"""Stats (replay/stats.py + gui/stats_tab.py): shot zones, team / period stats, lineups, exports, summary."""

import csv
import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

from replay import stats as S  # noqa: E402

COURT = {"length": 28.0, "width": 15.0, "baskets": {"L": [1.575, 7.5], "R": [26.425, 7.5]},
         "free_throw_line_x": {"L": 5.8, "R": 22.2}, "three_point_radius_m": 6.75}
GEO = None


@pytest.fixture(autouse=True)
def _geo():
    global GEO
    GEO = S.court_geometry({"court": COURT})


def _ev(typ, t, team=None, conf=0.8, **details):
    return {"id": f"gs-{typ}-{int(t * 10)}", "type": typ, "t": t, "team": team, "confidence": conf,
            "duration": details.pop("duration", 0.0), "details": details, "title": f"{typ} {team}"}


def synthetic_game_state() -> dict:
    poss = [("A", 10, 20, False), ("A", 21, 25, False), ("B", 30, 40, False), ("A", 50, 60, True),
            ("B", 320, 330, False), ("A", 340, 350, False)]
    return {
        "format": "advanced-analytist/game-state@1", "court": COURT, "teams": {"A": "Team orange", "B": "Team purple"},
        "time_range_s": [0.0, 600.0],
        "breaks": [{"t0": 290.0, "t1": 310.0, "duration_s": 20.0, "kind": "quarter_break"},
                   {"t0": 400.0, "t1": 430.0, "duration_s": 30.0, "kind": "timeout"}],
        "possessions": [{"id": f"p{i}", "team": tm, "t0": a, "t1": b, "duration_s": b - a, "fast_break": fb,
                         "confidence": 0.9} for i, (tm, a, b, fb) in enumerate(poss)],
        "shots": [
            {"t_release": 23.0, "t_rim": 23.8, "basket": "L", "team": "A", "free_throw": False, "made": True,
             "basket_event": "basket-1"},
            {"t_release": 54.0, "t_rim": 54.9, "basket": "L", "team": "A", "free_throw": False, "made": True,
             "basket_event": "basket-2"},
            {"t_release": 324.5, "t_rim": 325.0, "basket": "L", "team": "B", "free_throw": True, "made": True,
             "basket_event": "basket-3"},
        ],
        "basket_annotations": {
            "basket-1": {"kind": "field_goal", "points_guess": 2, "team": "A", "label": "Made 2"},
            "basket-2": {"kind": "field_goal", "points_guess": 3, "team": "A", "label": "Made 3?"},
            "basket-3": {"kind": "free_throw", "points_guess": 1, "team": "B", "label": "Made free throw 1/2"},
            "basket-5": {"kind": "field_goal", "points_guess": 2, "team": "A", "label": "Made 2"},
            "basket-6": {"kind": "field_goal", "points_guess": 2, "team": "A", "label": "Made 2"},
        },
        "events": [
            _ev("free_throw", 324.5, "B", 0.9, made=True), _ev("free_throw", 326.0, "B", 0.9, made=False),
            _ev("turnover", 40.0, "B"), _ev("steal", 336.0, "A", lost_by="B"), _ev("steal", 500.0, "A", conf=0.2),
            _ev("off_rebound", 15.0, "A"), _ev("def_rebound", 35.0, "B"), _ev("def_rebound", 45.0, "B"),
            _ev("fast_break", 50.0, "A", duration=4.0),
        ],
        "ball_events": {"version": 1, "shots": [
            {"t_release": 23.0, "t_rim": 23.8, "team": "A", "basket": "L", "position": [3.0, 7.5], "outcome": "made",
             "shooter": "5", "shooter_name": "#7", "confidence": 0.9},
            {"t_release": 54.0, "t_rim": 54.9, "team": "A", "basket": "L", "position": [1.0, 0.3], "outcome": "made",
             "shooter": "6", "shooter_name": "#8", "confidence": 0.8},
            {"t_release": 30.5, "t_rim": 31.4, "team": "B", "basket": "R", "position": [18.925, 7.5],
             "outcome": "missed", "shooter": "9", "shooter_name": "T9", "confidence": 0.7},
            {"t_release": 31.0, "t_rim": 31.9, "team": "B", "basket": "R", "position": [21.0, 9.0],
             "outcome": "missed", "confidence": 0.7},
            {"t_release": 342.0, "t_rim": 343.0, "team": "A", "basket": "L", "position": None, "outcome": "unknown",
             "confidence": 0.5},
        ]},
    }


REPLAY_EVENTS = [
    {"id": "basket-1", "type": "basket", "t": 24.0, "basket": "L", "team": "A", "confidence": 0.9},
    {"id": "basket-2", "type": "basket", "t": 55.0, "basket": "L", "team": "B", "confidence": 0.9},  # ann. says A
    {"id": "basket-3", "type": "basket", "t": 325.0, "basket": "L", "team": "B", "confidence": 0.9},
    {"id": "basket-4", "type": "basket", "t": 345.0, "basket": "L", "team": None, "confidence": 0.6},
    {"id": "basket-5", "type": "basket", "t": 100.0, "basket": "L", "team": "A", "confidence": 0.5},
    {"id": "basket-6", "type": "basket", "t": 200.0, "basket": "R", "team": "A", "confidence": 0.8},
]
OVERRIDES = {"basket-5": {"dismissed": True}}


def lineup_doc() -> dict:
    slots = {"A1": [[1, 0, 100, 7], [2, 320, 600, -1]], "A2": [[3, 0, 300, 8], [4, 310, 600, 9]],
             "A3": [[5, 0, 600, 10]], "A4": [[6, 0, 600, 11]], "A5": [[7, 0, 600, -1]]}
    slots.update({f"B{i}": [[20 + i, 0, 600, 20 + i]] for i in range(1, 6)})
    return {"format": "advanced-analytist/lineup@1",
            "slots": {k: {"id": 100 + i, "team": k[0], "pieces": v} for i, (k, v) in enumerate(slots.items())}}


def _stats(**kw):
    args = dict(replay_events=REPLAY_EVENTS, overrides=OVERRIDES, lineup_doc=lineup_doc(),
                names={"A": "Team orange", "B": "Team purple"}, colour_names={"A": "orange", "B": "purple"})
    args.update(kw)
    return S.compute(synthetic_game_state(), **args)


# ------------------------------------------------------------------------------------------------ zones
def test_zones():
    z = S.classify_zone
    assert z(3.0, 7.5, "L", GEO)["zone"] == "paint"
    assert z(5.0, 3.0, "L", GEO)["zone"] == "mid_range"
    c = z(1.0, 0.3, "L", GEO)
    assert c["zone"] == "corner_3" and c["three"] and not c["uncertain"]
    assert z(1.0, 0.7, "L", GEO)["uncertain"]  # 0.1 m from the straight 3-point segment
    ab = z(18.925, 7.5, "R", GEO)
    assert ab["zone"] == "above_break_3" and ab["distance_m"] == pytest.approx(7.5)
    assert z(21.0, 9.0, "R", GEO)["zone"] == "mid_range"
    assert z(25.0, 7.0, None, GEO)["zone"] == "paint"  # nearest basket = R
    assert z(1.575 + 6.9, 7.5, "L", GEO)["uncertain"]  # 0.15 m behind the arc
    assert z(None, None, "L", GEO)["zone"] == "unknown"


# ------------------------------------------------------------------------------------------------ team stats
def test_points_shots_and_rates():
    st = _stats()
    a, b = st["game"]["A"], st["game"]["B"]
    assert (a["points"], b["points"]) == (7, 1)
    assert st["score"]["unattributed_points"] == 2 and st["score"]["dismissed_baskets"] == 1
    assert (a["pts_2"], a["pts_3"], a["pts_ft"], b["pts_ft"]) == (4, 3, 0, 1)
    assert (a["fgm"], a["fga"], a["fg3m"], a["fg3a"]) == (3, 4, 1, 1)  # incl. a made basket with no attempt seen
    assert (b["fgm"], b["fga"], b["fg3a"]) == (0, 2, 1)
    assert (b["ftm"], b["fta"]) == (1, 2)
    assert (a["possessions"], b["possessions"]) == (3, 2)  # A 10-20 + 21-25 merged
    assert a["ppp"] == pytest.approx(7 / 3, abs=1e-3)
    assert (a["turnovers"], b["turnovers"], a["steals"]) == (0, 2, 1)  # low-confidence steal ignored
    assert b["tov_rate"] == 1.0
    assert a["orb"] == 1 and a["orb_rate"] == pytest.approx(1 / 3, abs=1e-3)
    assert a["fast_break_points"] == 3 and a["fast_breaks"] == 1
    assert a["time_of_possession_s"] == 34.0 and a["top_pct"] == pytest.approx(34 / 54, abs=1e-3)
    assert a["possessions_est"] == pytest.approx(4 - 1 + 0 + 0)
    # every number is explained by detected items with times (click -> seek)
    assert [i["t"] for i in a["items"]["baskets"]] == [24.0, 55.0, 200.0]
    assert len(b["items"]["turnovers"]) == 2


def test_shots_chart_rows():
    st = _stats()
    shots = st["shots"]
    assert len(shots) == 7  # incl. the made basket of unknown team (attempt not seen)
    by_t = {s["t"]: s for s in shots}
    assert by_t[23.0]["zone"] == "paint" and by_t[23.0]["points"] == 2 and by_t[23.0]["shooter_name"] == "#7"
    assert by_t[54.0]["zone"] == "corner_3" and by_t[54.0]["value"] == 3
    assert by_t[30.5]["shooter_name"] is None  # track ids are not players
    assert by_t[200.0]["source"].startswith("made basket only") and by_t[200.0]["zone"] == "unknown"
    zt = st["zones"]["A"]
    assert (zt["paint"]["fgm"], zt["corner_3"]["fga"], zt["unknown"]["fga"], zt["total"]["fga"]) == (1, 1, 1, 3)
    assert zt["total"]["pct"] == 1.0
    assert len(S.filter_shots(shots, team="B")) == 2
    assert len(S.filter_shots(shots, outcome="made")) == 4
    assert [s["t"] for s in S.filter_shots(shots, shooter="#8")] == [54.0]
    assert len(S.filter_shots(shots, period=st["periods"][1])) == 2


def test_periods_from_breaks():
    st = _stats()
    assert [p["label"] for p in st["periods"]] == ["Period 1", "Period 2"]  # time-outs do not split periods
    p1, p2 = st["periods"]
    assert (p1["t0"], p1["t1"], p2["t0"], p2["t1"]) == (0.0, 290.0, 310.0, 600.0)
    assert (p1["A"]["points"], p1["B"]["points"], p2["A"]["points"], p2["B"]["points"]) == (7, 0, 0, 1)
    assert p1["A"]["points"] + p2["A"]["points"] == st["game"]["A"]["points"]
    gs = synthetic_game_state()
    gs["breaks"] = []
    assert [p["label"] for p in S.periods(gs)] == ["Whole analysed range"]


def test_brief_spells_and_turnover_artefacts():
    """X, brief unconfirmed Y, X = one X possession (holder mix-up), and its turnovers are not counted; a brief Y
    spell with a Y shot is real. A change of possession right after the same team's shot is not a turnover."""
    gs = synthetic_game_state()
    gs["possessions"] = [{"team": "A", "t0": 0, "t1": 10, "duration_s": 10}, {"team": "B", "t0": 10.5, "t1": 12,
                         "duration_s": 1.5}, {"team": "A", "t0": 12.5, "t1": 20, "duration_s": 7.5},
                         {"team": "B", "t0": 100, "t1": 101.5, "duration_s": 1.5},
                         {"team": "A", "t0": 102, "t1": 110, "duration_s": 8},
                         {"team": "B", "t0": 120, "t1": 121.5, "duration_s": 1.5},  # B shoots in it: real
                         {"team": "A", "t0": 123, "t1": 130, "duration_s": 7}]
    gs["shots"], gs["basket_annotations"] = [], {}
    gs["ball_events"] = {"shots": [{"t_release": 121.0, "t_rim": 121.8, "team": "B", "basket": "R",
                                    "position": [20.0, 7.5], "outcome": "missed"},
                                   {"t_release": 140.0, "t_rim": 140.8, "team": "A", "basket": "L",
                                    "position": [4.0, 7.5], "outcome": "missed"}]}
    gs["events"] = [_ev("steal", 10.5, "B", lost_by="A"), _ev("steal", 12.5, "A", lost_by="B"),
                    _ev("turnover", 143.0, "A"),  # 3 s after A's own shot: the rebound
                    _ev("turnover", 160.0, "A")]
    absorbed = []
    m = S.merge_possessions(gs["possessions"], 3.0, [(121.0, "B")], absorbed)
    assert [p["team"] for p in m] == ["A", "B", "A"] and m[0]["t1"] == 110 and m[0]["absorbed"] == 2
    assert [a[:2] for a in absorbed] == [(10.5, 12), (100, 101.5)]
    st = S.compute(gs, [], {})
    assert (st["game"]["A"]["possessions"], st["game"]["B"]["possessions"]) == (2, 1)
    assert [x["t"] for x in st["game"]["A"]["items"]["turnovers"]] == [160.0]
    assert st["game"]["B"]["turnovers"] == 0


def test_dismissed_basket_turns_attempt_into_miss():
    gs = synthetic_game_state()
    st = S.compute(gs, REPLAY_EVENTS, {"basket-1": {"dismissed": True}})
    s = next(x for x in st["shots"] if x["t"] == 23.0)
    assert s["outcome"] == "missed" and s["points"] == 0
    assert st["game"]["A"]["points"] == 3 + 2 + 2  # basket-5 counts again (not dismissed here)


def test_reconcile_with_truth():
    st = _stats(truth={"final": {"orange": 10, "purple": 3}})
    sc = st["score"]
    assert sc["truth"] == {"A": 10.0, "B": 3.0}
    assert sc["diff"] == {"A": -3.0, "B": -2.0}
    assert sc["pct_of_truth"]["A"] == 70.0
    st = _stats(truth={"events": [{"t": 30, "team": "A", "points": 2}, {"t": 330, "team": "B", "points": 1}]})
    assert st["score"]["truth"] == {"A": 2.0, "B": 1.0}
    assert st["score"]["periods"][1]["B"] == 1.0
    assert _stats()["score"]["truth"] is None


# ------------------------------------------------------------------------------------------------ lineups
def test_lineups():
    lu = _stats()["lineups"]
    assert lu["available"]
    rows = {(r["team"], tuple(r["players"])): r for r in lu["rows"]}
    first = rows[("A", ("#10", "#11", "#7", "#8", "A5 (no number)"))]
    # #8 -> #9 substitution at the 290-310 break; A1's unread piece after the break does NOT inherit #7 (a
    # substitution may have happened there - minutes.py / 9.08)
    second = rows[("A", ("#10", "#11", "#9", "A1 (no number)", "A5 (no number)"))]
    assert first["seconds"] == pytest.approx(290.0)  # 0-310 minus the 290-310 break
    assert (first["pts_for"], first["pts_against"], first["plus_minus"]) == (7, 0, 7)
    assert (first["poss_for"], first["poss_against"], first["off_rating"]) == (2, 1, 350.0)
    assert (second["pts_for"], second["pts_against"]) == (0, 1)
    assert second["seconds"] == pytest.approx(290.0 - 30.0)  # 310-600 minus the time-out
    b = [r for r in lu["rows"] if r["team"] == "B"]
    assert len(b) == 1 and b[0]["plus_minus"] == -6 and b[0]["identified"] == 5
    assert first["identified"] == 4 and not first["duplicate_numbers"]


def test_lineups_unavailable():
    lu = _stats(lineup_doc=None)["lineups"]
    assert not lu["available"] and "unavailable" in lu["reason"].lower()


def test_empty_game_state():
    st = S.compute({}, [], {})
    assert st["game"]["A"]["points"] == 0 and st["shots"] == [] and not st["lineups"]["available"]


# ------------------------------------------------------------------------------------------------ files
def _project(tmp_path):
    from replay import ball_events, game_state as gs_mod
    root = tmp_path / "proj"
    res = root / "results"
    res.mkdir(parents=True)
    (root / "project.json").write_text(json.dumps({"name": "Synthetic", "settings": {"display_colors": {"A": "#ff7700"}}}),
                                       encoding="utf-8")
    replay = {"t": [0.0, 600.0], "events": REPLAY_EVENTS,
              "game": {"teams": {"A": {"name": "Team orange", "primary": "#ff8800", "colour_name": "orange"},
                                 "B": {"name": "Team purple", "primary": "#7722aa", "colour_name": "purple"}}}}
    (res / "replay_data.json").write_text(json.dumps(replay), encoding="utf-8")
    gs = synthetic_game_state()
    gs["engine_version"] = gs_mod.engine_version()  # (the current model's version, so the file is not rebuilt)
    gs["ball_events"]["version"] = ball_events.VERSION
    st = (res / "replay_data.json").stat()
    gs["source"] = {"replay_data_mtime": st.st_mtime, "replay_data_size": st.st_size}
    (res / "game_state.json").write_text(json.dumps(gs), encoding="utf-8")
    (res / "lineup.json").write_text(json.dumps(lineup_doc()), encoding="utf-8")
    (root / "timeline.json").write_text(json.dumps({"auto_overrides": OVERRIDES}), encoding="utf-8")
    (root / "score_truth.json").write_text(json.dumps({"final": {"orange": 10, "purple": 3}}), encoding="utf-8")
    return root


def test_for_project_and_exports(tmp_path):
    root = _project(tmp_path)
    st = S.for_project(root)
    assert st["game"]["A"]["points"] == 7 and st["score"]["truth"]["A"] == 10.0
    assert st["teams"]["A"]["color"] == "#ff7700"  # the user's display colour wins
    assert st["lineups"]["available"]
    out = tmp_path / "export"
    written = S.export_into(out, root)
    assert set(written) == set(S.EXPORT_DESCRIPTIONS)
    with (out / "stats_team.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    game_a = next(r for r in rows if r["scope"] == "game" and r["team"] == "A")
    assert game_a["points"] == "7" and game_a["possessions"] == "3"
    assert len([r for r in rows if r["scope"] == "period"]) == 4
    with (out / "stats_shots.csv").open(encoding="utf-8") as f:
        assert len(list(csv.DictReader(f))) == 7
    with (out / "stats_lineups.csv").open(encoding="utf-8") as f:
        assert len(list(csv.DictReader(f))) == 3
    doc = json.loads((out / "stats.json").read_text(encoding="utf-8"))
    assert "items" not in doc["game"]["A"] and doc["notes"]
    assert S.export_into(tmp_path / "x", tmp_path / "missing") == {} or True  # never raises


# ------------------------------------------------------------------------------------------------ GUI
@pytest.fixture(scope="module")
def app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_summary_png_pdf(app, tmp_path):
    from replay.gui.stats_tab import save_summary
    st = _stats(truth={"final": {"orange": 10, "purple": 3}})
    png = save_summary(tmp_path / "s.png", st, {"A": "#ff8800"}, "Synthetic")
    pdf = save_summary(tmp_path / "s.pdf", st, None, "Synthetic")
    assert png.stat().st_size > 10_000
    assert pdf.read_bytes()[:4] == b"%PDF"


def test_stats_tab(app, tmp_path):
    from PySide6.QtWidgets import QMainWindow, QTabWidget, QWidget

    from replay.gui.stats_tab import StatsTab

    root = _project(tmp_path)

    class P:
        def __init__(self, r):
            self.root = r

        def info(self):
            return {"name": "Synthetic"}

    class FakeReplay(QWidget):
        def __init__(self):
            super().__init__()
            self.p, self.data, self.seeks = None, None, []

        def seek_to(self, t):
            self.seeks.append(t)

        def display_colors(self):
            return {}

    win = QMainWindow()
    win.replay_tab = FakeReplay()
    win.tabs = QTabWidget()
    tab = StatsTab(win)
    win.tabs.addTab(tab, "Stats")
    win.tabs.addTab(win.replay_tab, "Replay")
    win.setCentralWidget(win.tabs)
    tab.load(P(root))
    tab.refresh(force=True)
    assert tab.st is not None and "7" in tab.msg.text()
    assert tab.team_table.rowCount() == len(S.STAT_ROWS)
    assert tab.lineup_table.rowCount() == 3
    assert len(tab.chart.shots) == 7
    tab.team.setCurrentIndex(2)  # team B
    assert len(tab.chart.shots) == 2
    tab.team.setCurrentIndex(0)
    tab.period.setCurrentIndex(2)  # Period 2
    assert tab.team_table.item(0, 2).text() == "1"
    tab._team_cell(14, 2)  # turnovers of B in period 2
    assert tab.items.count() == 1
    tab.items.itemClicked.emit(tab.items.item(0))
    assert win.replay_tab.seeks and win.replay_tab.seeks[-1] == pytest.approx(335.0)
    tab.chart.shotClicked.emit(tab.chart.shots[0])
    assert len(win.replay_tab.seeks) == 2
    tab._lineup_cell(0, 0)
    assert tab.items.count() >= 1
    # hand-off to the Playlists feature: a stat row -> live filter, a zone -> fixed clip list
    from replay.gui import stats_tab as stab
    if stab.playlist_module() is None:
        return
    from replay import playlists as pl
    tab._team_cell(14, 2)  # turnovers of B in period 2
    tab.send_to_playlist()
    tab.period.setCurrentIndex(0)
    tab._zone_clicked(tab.zones.item(0, 0))  # paint shots (whole game)
    tab.send_to_playlist()
    book = pl.PlaylistBook.load(root)
    assert [p.name.split(":")[0] for p in book.playlists] == ["Stats", "Stats"]
    live, fixed = book.playlists
    assert live.filter.categories == ["turnover"] and live.filter.team == "B" and live.filter.t_from == 310.0
    assert len(live.clips) == 1 and live.clips[0].t0 <= 336.0 <= live.clips[0].t1
    assert len(fixed.clips) == 1 and fixed.clips[0].t0 <= 23.0


def test_stale_lineup_file_ignored(tmp_path):
    root = _project(tmp_path)
    res = root / "results"
    assert S.lineup_is_current(res, {})  # written together with replay_data.json
    old = res / "lineup.json"
    t = (res / "replay_data.json").stat().st_mtime - 7200
    os.utime(old, (t, t))
    assert not S.lineup_is_current(res, {})  # left over from an earlier build with the solver on
    assert S.lineup_is_current(res, {"track_info": {"7": {"slot": "A1"}}})  # the solver marked its slots
    assert not S.for_project(root)["lineups"]["available"]
