"""Coach report exports: box score CSV, printable PDF with player pages, copy-summary text."""

import csv
import os
import re

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402
from fake_project import make_project  # noqa: E402

from replay import coach as C  # noqa: E402
from replay import coach_export as E  # noqa: E402

COURT = {"length": 28.0, "width": 15.0, "basket_x": 1.575, "basket_y": 7.5, "three_r": 6.75, "three_side": 0.9,
         "paint_half": 2.45, "ft_x": 5.8, "circle_r": 1.8, "ft_circle_r": 1.8, "no_charge_r": 1.25}


def _pt(pts: int) -> dict:
    return {"pts": pts, "fgm": 0, "fga": 0, "fg3m": 0, "fg3a": 0, "moments": []}


def _rep(sheet: bool = False) -> dict:
    m = [{"t": 5.0, "t1": None, "title": "Made 3", "team": "A"}]
    r12 = {"player": "#12", "name": "=Virtanen", "min_s": 300.0, "pts": 7, "fgm": 3, "fga": 4, "fg3m": 1, "fg3a": 2,
           "oreb": 1, "dreb": 2, "ast": 1, "stl": 1, "tov": 0, "pm": 4, "moments": {"pts": m}}
    r5 = {"player": "#5", "min_s": 120.0, "pts": 2, "fgm": 1, "fga": 3, "fg3m": 0, "fg3a": 0, "oreb": 0, "dreb": 0,
          "ast": 0, "stl": 0, "tov": 2, "pm": -1, "moments": {}}
    un = {"player": "Unidentified", "min_s": 0.0, "pts": 0, "fgm": 0, "fga": 1, "fg3m": 0, "fg3a": 0, "oreb": 0,
          "dreb": 1, "ast": 0, "stl": 0, "tov": 1, "pm": None, "moments": {}}
    tot_a = {"player": "Team", "min_s": None, "pts": 9, "fgm": 4, "fga": 8, "fg3m": 1, "fg3a": 2, "ftm": 0, "fta": 2,
             "oreb": 1, "dreb": 3, "ast": 1, "stl": 1, "tov": 3, "pm": None, "moments": {}}
    r7 = {"player": "#7", "min_s": 420.0, "pts": 10, "fgm": 5, "fga": 9, "fg3m": 0, "fg3a": 1, "oreb": 0, "dreb": 1,
          "ast": 0, "stl": 0, "tov": 1, "pm": -4, "moments": {}}
    tot_b = {**r7, "player": "Team", "min_s": None, "pm": None, "ftm": 0, "fta": 0}
    rep = {"teams": {"A": {"name": "Kouvot", "color": "#ff8800"}, "B": {"name": "Honka", "color": "#1144aa"}},
           "score": {"A": 9, "B": 10}, "score_note": "", "box": {"A": {"rows": [r5, r12, un], "total": tot_a},
                                                                 "B": {"rows": [r7], "total": tot_b}},
           "compare": [{"key": "points", "title": "Points", "A": 9, "B": 10, "moments_A": [], "moments_B": []},
                       {"key": "fg_pct", "title": "FG %", "A": 0.5, "B": 0.556},
                       {"key": "ppp", "title": "Points / possession", "A": 0.9, "B": 1.25},
                       {"key": "steals", "title": "Steals", "A": 1, "B": None}],
           "shots": [{"t": 5.0, "team": "A", "x": 8.0, "y": 7.0, "outcome": "made", "value": 3, "title": "Made 3",
                      "shooter": "1"}],
           "court": COURT, "lineups": {"available": False, "note": ""}, "momentum": [(0.0, 0, 0), (60.0, 9, 10)],
           "runs": [{"team": "B", "t0": 70.0, "t1": 130.0, "points": 8, "title": "8-0 run", "moments": []},
                    {"team": "A", "t0": 10.0, "t1": 40.0, "points": 6, "title": "6-0 run", "moments": []}],
           "periods": [], "provenance": C.PROVENANCE, "columns": C.COLUMNS, "sheet": None,
           "period_table": {"source": "detected breaks", "rows": [
               {"label": "Period 1", "t0": 0.0, "t1": 60.0, "A": _pt(9), "B": _pt(2), "official": None},
               {"label": "Q2 (from the scoresheet)", "t0": 60.0, "t1": 200.0, "A": _pt(0), "B": _pt(8),
                "official": None}]}}
    if sheet:
        r12.update(official=["pts", "ft", "pf"], pts_detected=7, pts=9, ftm=2, fta=None, pf=3)
        rep["sheet"] = {"full_game": True}
        rep["score_official"] = {"A": 11, "B": 10}
        rep["score"] = {"A": 11, "B": 10}
    return rep


def _cols(rows):
    return {h: i for i, h in enumerate(rows[0])}


def test_box_rows_values_and_sources():
    rows = E.box_rows(_rep(), "A")
    c = _cols(rows)
    assert rows[0][0] == "Player" and rows[0][-1] == "Source"
    r5, r12, un, team = rows[1:]
    assert r12[c["Name"]] == "'=Virtanen"  # not run as a spreadsheet formula
    assert (r12[c["MIN"]], r12[c["PTS"]], r12[c["FGM"]], r12[c["FGA"]], r12[c["FG%"]], r12[c["REB"]]) == \
        ("5.0", "7", "3", "4", "75.0", "3")
    assert r12[c["PF"]] == "" and r12[c["FTM"]] == ""  # not measured / team only: empty, never 0
    assert "not measured: PF" in r12[c["Source"]] and "team only: FT" in r12[c["Source"]]
    assert "measured: STL" in r12[c["Source"]] and "inferred: MIN PTS" in r12[c["Source"]]
    assert un[0] == "Unidentified players" and un[c["MIN"]] == "" and un[c["+/-"]] == ""
    assert team[0] == "Team" and team[c["FTA"]] == "2" and "MIN" not in team[c["Source"]]


def test_box_rows_scoresheet_marks_official():
    rows = E.box_rows(_rep(sheet=True), "A")
    c = _cols(rows)
    r12 = rows[2]
    assert (r12[c["PTS"]], r12[c["FTM"]], r12[c["FTA"]], r12[c["PF"]]) == ("9", "2", "", "3")
    src = r12[c["Source"]]
    assert src.startswith("official: PTS FT PF (scoresheet, video PTS 7)") and "not measured" not in src


def test_compare_rows_and_decimal_comma():
    rows = E.compare_rows(_rep(), decimal=",")
    assert rows[0] == ["Stat", "Kouvot", "Honka", "Source"]
    by = {r[0]: r for r in rows[1:]}
    assert by["FG %"][1:] == ["50,0", "55,6", "inferred"]
    assert by["Points / possession"][1:3] == ["0,90", "1,25"]
    assert by["Steals"][1:] == ["1", "", "measured"]
    assert by["Points"][3] == "measured"
    rep = _rep()
    rep["compare"][0]["official"] = True
    assert E.compare_rows(rep)[1][3] == "official (scoresheet)"


def test_write_csv_three_files(tmp_path):
    paths = E.write_csv(_rep(), tmp_path / "Game 1.csv", ";", ",")
    assert [p.name for p in paths] == ["Game 1 - Kouvot.csv", "Game 1 - Honka.csv", "Game 1 - team comparison.csv"]
    raw = paths[0].read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")  # BOM: Excel reads UTF-8
    with open(paths[0], encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.reader(fh, delimiter=";"))
    assert rows[0][:4] == ["Player", "Name", "MIN", "PTS"] and len(rows) == 5
    assert rows[2][2] == "5,0"
    # same team names -> still two files
    rep = _rep()
    rep["teams"]["B"]["name"] = "Kouvot"
    assert len({p.name for p in E.write_csv(rep, tmp_path / "x")}) == 3


def test_xlsx_only_with_openpyxl(tmp_path):
    if not E.has_xlsx():
        with pytest.raises(ImportError):
            E.write_xlsx(_rep(), tmp_path / "b.xlsx")
        return
    import openpyxl
    out = E.write_xlsx(_rep(), tmp_path / "b.xlsx")
    wb = openpyxl.load_workbook(out)
    assert wb.sheetnames == ["Kouvot", "Honka", "Team comparison"]


def test_summary_text():
    txt = E.summary_text(_rep(), "Kouvot vs Honka")
    lines = txt.splitlines()
    assert lines[0] == "Kouvot vs Honka"
    assert lines[1] == "Kouvot 9 – 10 Honka (detected from the video)"
    assert "Kouvot: #12 =Virtanen 7, #5 2" in txt and "Honka: #7 10" in txt
    assert "Kouvot: FG 4/8 (50%), 3PT 1/2 (50%), FT 0/2" in txt
    i = lines.index("Key runs")
    assert lines[i + 1] == "Kouvot 6-0 (Period 1, video 0:10–0:40)"  # in time order
    assert lines[i + 2] == "Honka 8-0 (Q2, video 1:10–2:10)"
    assert "Unidentified" not in txt and len(txt) < 900
    off = E.summary_text(_rep(sheet=True))
    assert off.splitlines()[0] == "Kouvot 11 – 10 Honka (final, from the scoresheet)"


def test_summary_single_period_has_no_period_label():
    rep = _rep()
    rep["period_table"]["rows"] = rep["period_table"]["rows"][:1]
    assert "Honka 8-0 (video 1:10–2:10)" in E.summary_text(rep)


# ====================================================================================== Qt
@pytest.fixture(scope="module")
def app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def _pdf_pages(path) -> int:
    return len(re.findall(rb"/Type\s*/Page[^s]", path.read_bytes()))


def test_pdf_with_player_pages(app, tmp_path):
    from PySide6.QtGui import QColor, QImage, QPainter
    from replay.gui import coach_report as R
    rep = _rep()
    pg = {"team": "A", "player": "#12", "tracks": ["1"], "row": rep["box"]["A"]["rows"][1],
          "heat": [[1.0 if (i + j) % 5 == 0 else 0.0 for i in range(28)] for j in range(16)], "bin_m": 1.0,
          "court": COURT, "tracked_s": 300.0, "shots": rep["shots"]}
    one = R.save_pdf(rep, tmp_path / "one.pdf", "x")
    many = R.save_pdf(rep, tmp_path / "many.pdf", "x", players=[pg, {**pg, "player": "#5", "shots": []}])
    assert _pdf_pages(one) == 1 and _pdf_pages(many) == 3
    img = QImage(800, 566, QImage.Format.Format_ARGB32)
    img.fill(QColor("white"))
    qp = QPainter(img)
    qp.scale(0.5, 0.5)
    R.paint_player_sheet(qp, pg, rep, "x", "2 / 3")
    qp.end()
    assert img.pixelColor(200, 180) != QColor("white")  # the heat-map court is painted


def test_player_pages_skip_sheet_only_rows():
    from replay.gui import coach_report as R
    rep = _rep()
    rep["box"]["A"]["rows"].append({**rep["box"]["A"]["rows"][0], "player": "#9", "sheet_only": True})
    pages = R.player_pages(rep, None, {"t": [0.0, 0.1], "tracks": []})
    assert [(p["team"], p["player"]) for p in pages] == [("A", "#5"), ("A", "#12"), ("B", "#7")]


def test_dialog_exports(app, tmp_path):
    from PySide6.QtWidgets import QApplication
    from replay.gui import coach_report as R
    from replay.gui.projects import Project
    root = make_project(tmp_path / "p")
    rep = C.build(root)
    dlg = R.CoachReportDialog(None, Project(root), rep)
    acts = [a.text() for a in dlg.export_btn.menu().actions() if a.text()]
    assert any("CSV" in a for a in acts) and any("player" in a for a in acts)
    assert any("Excel" in a for a in acts) == E.has_xlsx()
    txt = dlg.copy_summary()
    assert QApplication.clipboard().text() == txt and txt.startswith("Kouvot vs Honka")
    paths = dlg.export_box_csv(str(tmp_path / "box.csv"))
    assert len(paths) == 3 and all(p.exists() for p in paths)
    out = dlg.save_with_players(str(tmp_path / "all"))
    assert out is not None and out.suffix == ".pdf" and _pdf_pages(out) >= 1
    assert "player page" in dlg.last_note
