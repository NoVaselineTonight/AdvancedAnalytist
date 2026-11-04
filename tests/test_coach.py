"""Coach report: box score / runs / momentum (no Qt) and the page, PDF and click-to-playlist (offscreen)."""

import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402
from fake_project import make_project  # noqa: E402

from replay import coach as C  # noqa: E402


def test_runs_and_momentum():
    bk = [{"t": 10.0, "team": "A", "points": 2}, {"t": 20.0, "team": "A", "points": 3},
          {"t": 30.0, "team": "A", "points": 2}, {"t": 40.0, "team": "B", "points": 2},
          {"t": 50.0, "team": "B", "points": 1}]
    r = C.runs(bk)
    assert len(r) == 1 and r[0]["team"] == "A" and r[0]["points"] == 7 and r[0]["moments"]
    mo = C.momentum(bk, [0.0, 60.0])
    assert mo[0] == (0.0, 0, 0) and mo[-1] == (60.0, 7, 3)


def test_cell_text_and_minutes():
    assert C.fmt_min(59.6) == "1:00" and C.fmt_min(None) == "–"
    r = {"fgm": 2, "fga": 5, "fg3m": 0, "fg3a": 1, "pm": -3, "min_s": 125.0, "pts": 4}
    assert (C.cell_text(r, "fg"), C.cell_text(r, "fg3"), C.cell_text(r, "pm"), C.cell_text(r, "min"),
            C.cell_text(r, "pf"), C.cell_text(r, "ft")) == ("2-5", "0-1", "-3", "2:05", "–", "–")


def test_build_on_a_replay_folder(tmp_path):
    root = make_project(tmp_path / "p")
    rep = C.build(root)
    assert rep["format"] == C.FORMAT and set(rep["box"]) == {"A", "B"}
    for k in "AB":
        tot = rep["box"][k]["total"]
        assert tot["player"] == "Team" and tot["pts"] == rep["score"][k]
    assert rep["provenance"]["pf"].startswith("Personal fouls are not detected")


@pytest.fixture(scope="module")
def app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def _rep():
    m = [{"t": 5.0, "t1": None, "title": "Made 2", "team": "A"}]
    row = {"player": "#12", "min_s": 300.0, "pts": 2, "fgm": 1, "fga": 2, "fg3m": 0, "fg3a": 0, "oreb": 1, "dreb": 0,
           "ast": 0, "stl": 0, "tov": 0, "pm": 2, "moments": {"pts": m, "fg": m}}
    tot = {**row, "player": "Team", "ftm": 0, "fta": 0, "min_s": None, "pm": None}
    box = {"A": {"rows": [row], "total": tot},
           "B": {"rows": [], "total": {**tot, "pts": 0, "moments": {}}}}
    return {"teams": {"A": {"name": "Home", "color": "#ff0000"}, "B": {"name": "Away", "color": "#0000ff"}},
            "score": {"A": 2, "B": 0}, "score_note": "1 made basket", "box": box,
            "compare": [{"key": "points", "title": "Points", "A": 2, "B": 0, "moments_A": m, "moments_B": []}],
            "shots": [{"t": 5.0, "team": "A", "x": 3.0, "y": 7.0, "outcome": "made", "title": "Made 2", "shooter": "1"}],
            "court": {"length": 28.0, "width": 15.0, "basket_x": 1.575, "basket_y": 7.5, "three_r": 6.75,
                      "three_side": 0.9, "paint_half": 2.45, "ft_x": 5.8, "circle_r": 1.8, "ft_circle_r": 1.8,
                      "no_charge_r": 1.25},
            "lineups": {"available": False, "note": "no lineup"}, "runs": [],
            "momentum": [(0.0, 0, 0), (5.0, 2, 0), (60.0, 2, 0)], "periods": [], "provenance": C.PROVENANCE,
            "columns": C.COLUMNS}


def test_page_pdf_and_click_to_playlist(app, tmp_path):
    from PySide6.QtCore import QPointF
    from replay.gui import coach_report as R
    from replay.gui.projects import Project
    rep = _rep()
    img = R.render_image(rep, 800)
    assert img.width() == 800 and not img.isNull()
    pdf = R.save_pdf(rep, tmp_path / "r.pdf", "x")
    assert pdf.read_bytes()[:4] == b"%PDF"
    from PySide6.QtGui import QImage, QPainter
    im = QImage(1600, 1131, QImage.Format.Format_ARGB32)
    qp = QPainter(im)
    hits = R.paint_report(qp, rep)
    qp.end()
    assert any(h[1].endswith("PTS") for h in hits)
    root = make_project(tmp_path / "p")
    dlg = R.CoachReportDialog(None, Project(root), rep)
    h = next(h for h in hits if h[1].endswith("PTS"))
    dlg.open_playlist(h[1], h[2])
    book = json.loads((root / "playlists.json").read_text(encoding="utf-8"))
    assert book["playlists"][-1]["name"].endswith("PTS") and len(book["playlists"][-1]["clips"]) == 1
    assert hits.at(QPointF(h[0].center())) is h


# ---------------------------------------------------------------------- periods + scoresheet merge
def test_period_table_from_timeline_marks(tmp_path):
    root = make_project(tmp_path / "p")
    tl = json.loads((root / "timeline.json").read_text(encoding="utf-8"))
    tl["tracks"].append({"id": "q", "name": "Periods", "kind": "user",
                         "marks": [{"id": "m1", "t": 0.0, "title": "Q1"}, {"id": "m2", "t": 2.0, "title": "2nd quarter"}]})
    (root / "timeline.json").write_text(json.dumps(tl), encoding="utf-8")
    rep = C.build(root)
    pt = rep["period_table"]
    assert pt["source"] == "timeline period marks" and [r["label"] for r in pt["rows"]] == ["Q1", "2nd quarter"]
    assert sum(r["A"]["fga"] for r in pt["rows"]) == rep["box"]["A"]["total"]["fga"]


def test_scoresheet_merge_full_game(tmp_path, monkeypatch):
    from replay import scoresheet as SS
    root = make_project(tmp_path / "p")
    rep0 = C.build(root)
    num = next((r["player"] for r in rep0["box"]["A"]["rows"] if r["player"].startswith("#")), "#7")[1:]
    csv = tmp_path / "sheet.csv"
    csv.write_text(f"# home: Kouvot\n# away: Honka\n# final: 30-20\nteam,number,name,points,fouls,ft\n"
                   f"A,{num},Pelaaja Yksi,30,3,4\nB,9,Toinen Pelaaja,20,5,2\n", encoding="utf-8")
    SS.import_into(root, csv)
    monkeypatch.setattr(SS, "crosscheck", lambda root, sheet=None, **k: {
        "team_map": {"A": "A", "B": "B"}, "full_game": True, "sheet_window": [], "team_map_source": "test",
        "summary": "", "alignment": "high"})
    rep = C.build(root)
    a = {r["player"]: r for r in rep["box"]["A"]["rows"]}
    assert a[f"#{num}"]["pts"] == 30 and "pts" in a[f"#{num}"]["official"] and a[f"#{num}"]["pf"] == 3
    assert C.cell_text(a[f"#{num}"], "ft") == "4" and a[f"#{num}"]["name"] == "Pelaaja Yksi"
    b = {r["player"]: r for r in rep["box"]["B"]["rows"]}
    assert b["#9"]["pts"] == 20 and b["#9"].get("sheet_only")  # a scorer the video did not identify
    assert rep["score"] == {"A": 30, "B": 20} and rep["score_detected"] == {"A": rep0["score"]["A"], "B": rep0["score"]["B"]}
    assert any(h["player"] == "Team" for h in rep["sheet"]["hints"])  # detected team points disagree
    # tracking-only numbers are untouched
    before = {r["player"]: r for r in rep0["box"]["A"]["rows"]}.get(f"#{num}")
    assert a[f"#{num}"]["fga"] == (before["fga"] if before else 0)
    assert rep["box"]["A"]["total"]["fga"] == rep0["box"]["A"]["total"]["fga"]


# ---------------------------------------------------------------------- scoresheet teams vs kits
def _sheet_and_replay():
    from replay import scoresheet as SS
    sh = SS.Scoresheet(kind="csv")
    sh.teams["A"].name, sh.teams["B"].name = "KlaNMKY", "LePy"
    sh.teams["A"].players = [SS.SheetPlayer("A", n) for n in ("6", "36", "43")]
    sh.teams["B"].players = [SS.SheetPlayer("B", n) for n in ("20", "25", "28")]
    # video team A (orange, set as home) shows LePy's numbers, B (purple) shows KlaNMKY's
    info = {"1": {"number": "25", "team": "A", "number_reads": 30}, "2": {"number": "20", "team": "A", "number_reads": 12},
            "3": {"number": "43", "team": "B", "number_reads": 40}, "4": {"number": "36", "team": "B", "number_reads": 9}}
    replay = {"t": [0.0, 0.1], "tracks": [{"id": k, "team": v["team"], "i0": 0, "x": [100, 100]} for k, v in info.items()],
              "track_info": info,
              "game": {"teams": {"A": {"name": "Team orange", "colour_name": "orange", "side": "home", "side_confirmed": True},
                                 "B": {"name": "Team purple", "colour_name": "purple", "side": "away", "side_confirmed": True}}}}
    return sh, replay


def test_scoresheet_teams_follow_jersey_numbers_over_a_wrong_home_away(tmp_path):
    from replay import scoresheet as SS
    sh, replay = _sheet_and_replay()
    sug = SS.team_map_suggestion(sh, replay)
    assert sug["map"] == {"A": "B", "B": "A"} and sug["support"] >= SS.NUMBERS_STRONG
    tm, src = SS.team_map_info(sh, replay)
    assert tm == {"A": "B", "B": "A"} and "contradict" in src  # never silently credit the wrong team
    root = tmp_path / "p"
    (root / "results").mkdir(parents=True)
    (root / "results" / "replay_data.json").write_text(json.dumps(replay), encoding="utf-8")
    SS.save(root, sh)
    c = SS.home_away_conflict(root, replay, {})
    assert c["home_video_team"] == "B" and c["home_name"] == "KlaNMKY" and c["colour"] == "purple"
    # agreeing home/away: no prompt
    replay["game"]["teams"]["A"]["side"], replay["game"]["teams"]["B"]["side"] = "away", "home"
    assert SS.home_away_conflict(root, replay, {}) is None and SS.team_map_info(sh, replay)[0] == {"A": "B", "B": "A"}


def test_kit_colour_words_on_the_sheet():
    from replay import scoresheet as SS
    sh, replay = _sheet_and_replay()
    sh.teams["A"].players, sh.teams["B"].players = [], []
    sh.teams["A"].name = "KlaNMKY Violetti"
    assert SS.kit_evidence(sh, replay) == (0.0, 10.0)
    assert SS.team_map_suggestion(sh, replay)["map"] == {"A": "B", "B": "A"}


# ---------------------------------------------------------------------- possessions + shot quality (page 2)
def test_possession_timeline_outcomes_and_unseen_baskets():
    st = {"shots": [{"t": 12.0, "t_rim": 13.0, "team": "A", "outcome": "made", "value": 3, "points": 3, "shooter": "7"},
                    {"t": 30.0, "team": "B", "outcome": "missed", "value": 2, "points": 0, "shooter": None},
                    {"t": 70.0, "team": "B", "outcome": "made", "value": 2, "points": 2, "shooter": None}],
          "periods": [{"t0": 0.0, "t1": 60.0}, {"t0": 65.0, "t1": 120.0}]}
    gs = {"possessions": [{"team": "A", "t0": 2.0, "t1": 14.0, "duration_s": 12.0},
                          {"team": "B", "t0": 15.0, "t1": 31.0, "duration_s": 16.0},
                          {"team": "A", "t0": 32.0, "t1": 40.0, "duration_s": 8.0},
                          {"team": "B", "t0": 41.0, "t1": 58.0, "duration_s": 17.0}],
          "events": [{"type": "steal", "team": "B", "t": 39.0, "confidence": 0.9}]}
    who = C._Who(type("Cat", (), {"track_number": {"7": "12"}, "track_team": {"7": "A"}})())
    rows = C.possession_timeline(st, gs, who)
    assert [r["outcome"] for r in rows] == ["made", "miss", "turnover", "period_end", "made"]
    assert rows[0]["detail"].startswith("Made 3 by #12") and rows[0]["points"] == 3
    assert "defensive rebound" in rows[1]["detail"]
    assert rows[-1]["duration_s"] is None and "possession not detected" in rows[-1]["detail"]  # never lose points
    assert all(r["moments"] for r in rows)


def test_shot_quality_zones_and_expected():
    shots = [{"team": "A", "zone": "paint", "distance_m": 1.0, "outcome": "made", "value": 2, "points": 2, "t": 1.0},
             {"team": "A", "zone": "paint", "distance_m": 3.0, "outcome": "missed", "value": 2, "points": 0, "t": 2.0},
             {"team": "B", "zone": "corner_3", "distance_m": 6.8, "outcome": "made", "value": 3, "points": 3, "t": 3.0},
             {"team": "B", "zone": "unknown", "outcome": "missed", "value": 2, "points": 0, "t": 4.0}]
    sq = C.shot_quality(shots, {"no_charge_r": 1.25})
    a = {r["zone"]: r for r in sq["A"]}
    assert a["restricted"]["fga"] == 1 and a["restricted"]["pps"] == 2.0 and a["paint"]["fga"] == 1
    b = {r["zone"]: r for r in sq["B"]}
    assert b["corner_3"]["pps"] == 3.0 and sq["unlocated"]["B"] == 1
    # expected = 3 x (1 made + 0.4 x 5) / (1 + 5)
    assert abs(b["corner_3"]["xpps"] - round(3 * 3.0 / 6.0, 2)) < 1e-9


def test_page2_paints_and_filters(app):
    from PySide6.QtGui import QImage, QPainter
    from replay.gui import coach_report as R
    rep = _rep()
    rep["possessions"] = [{"t0": 1.0, "t1": 9.0, "team": "A", "duration_s": 8.0, "outcome": "made", "detail": "Made 2",
                           "shooter": "#12", "points": 2, "provenance": "measured", "moments": [{"t": 1.0, "title": "x"}]},
                          {"t0": 10.0, "t1": 20.0, "team": "B", "duration_s": 10.0, "outcome": "turnover",
                           "detail": "Turnover", "shooter": None, "points": 0, "provenance": "inferred",
                           "moments": [{"t": 10.0, "title": "y"}]}]
    rep["shot_quality"] = C.shot_quality(rep["shots"], rep["court"])
    img = QImage(1600, 1131, QImage.Format.Format_ARGB32)
    qp = QPainter(img)
    hits = R.paint_page2(qp, rep)
    qp.end()
    assert sum(1 for h in hits if h[1].startswith("Possession")) == 2
    assert [p["team"] for p in R.filtered_possessions(rep, team="B")] == ["B"]
    assert R.filtered_possessions(rep, outcome="made")[0]["detail"] == "Made 2"
