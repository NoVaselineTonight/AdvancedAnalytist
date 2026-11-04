"""Scoresheet (poytakirja) import: built-in PDF extractor, SKL sheet parser, roster / score truth, cross-check."""

from __future__ import annotations

import hashlib as _hashlib

import json
import re
import zlib
from pathlib import Path

import pytest

from replay import pdftext
from replay import roster_names as rn
from replay import scoresheet as S

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "scoresheets" / "1D 2026-10-03 KlaNMKY-LePy.pdf"
needs_sample = pytest.mark.skipif(not SAMPLE.exists(), reason="sample scoresheet not available")

# page 2 (OTTELUYHTEENVETO) of the sample: number -> (points, 1P, 2P, 3P, [J1..J4], fouls)
SUMMARY_A = {"2": (1, 1, 0, 0, [0, 0, 0, 1], 3), "4": (4, 0, 2, 0, [0, 2, 0, 2], 2),
             "6": (12, 1, 4, 1, [5, 3, 2, 2], 3), "11": (0, 0, 0, 0, [0, 0, 0, 0], 2),
             "13": (0, 0, 0, 0, [0, 0, 0, 0], 1), "21": (0, 0, 0, 0, [0, 0, 0, 0], 1),
             "27": (0, 0, 0, 0, [0, 0, 0, 0], 3), "36": (4, 2, 1, 0, [0, 3, 1, 0], 1),
             "43": (2, 0, 1, 0, [0, 2, 0, 0], 1)}
SUMMARY_B = {"10": (18, 2, 8, 0, [0, 10, 8, 0], 1), "13": (6, 0, 3, 0, [0, 2, 2, 2], 0),
             "20": (8, 0, 4, 0, [2, 2, 2, 2], 1), "21": (15, 3, 6, 0, [4, 5, 0, 6], 1),
             "25": (2, 0, 1, 0, [2, 0, 0, 0], 2), "28": (28, 0, 14, 0, [12, 12, 0, 4], 1),
             "33": (0, 0, 0, 0, [0, 0, 0, 0], 2), "41": (8, 0, 4, 0, [0, 2, 4, 2], 3),
             "44": (4, 0, 2, 0, [0, 2, 0, 2], 1), "53": (9, 3, 3, 0, [4, 0, 3, 2], 2)}


@pytest.fixture(scope="module")
def sheet() -> S.Scoresheet:
    if not SAMPLE.exists():
        pytest.skip("sample scoresheet not available")
    return S.read(SAMPLE)


# ====================================================================================== the sample PDF
@needs_sample
def test_sample_teams_game_and_final(sheet: S.Scoresheet) -> None:
    assert sheet.kind == "skl-pdf"
    assert sheet.teams["A"].name == "KlaNMKY" and sheet.teams["B"].name == "LePy"
    assert sheet.final == {"A": 23, "B": 98}
    assert sheet.period_scores == {"A": [5, 10, 3, 5], "B": [24, 35, 19, 20]}
    assert sheet.game["date"] == "3.10.2026" and sheet.game["time"] == "12:45"
    assert "13-vuotiaat" in sheet.game["competition"]
    assert sheet.game["venue"].lower().startswith("klaukkalan koulu")
    assert len(sheet.officials["referees"]) == 2
    assert sheet.officials["scorer"]


@needs_sample
def test_sample_rosters_points_fouls_match_summary_page(sheet: S.Scoresheet) -> None:
    for side, exp in (("A", SUMMARY_A), ("B", SUMMARY_B)):
        team = sheet.teams[side]
        assert sorted(p.number for p in team.players) == sorted(exp, key=int) or \
            {p.number for p in team.players} == set(exp)
        for num, (pts, ft, fg2, fg3, quarters, fouls) in exp.items():
            p = team.player(num)
            assert p is not None, num
            assert p.name and not re.search(r"\d", p.name)
            assert (p.points, p.ft_made, p.fg2_made, p.fg3_made) == (pts, ft, fg2, fg3), num
            assert p.points_by_period == quarters, num
            assert p.fouls_total == fouls and len(p.fouls) == fouls, num
            assert p.played
            assert re.fullmatch(r"\d{8}", p.licence)  # read, but private
        assert len(team.coaches) == 2 and {c["role"] for c in team.coaches} == {"head", "assistant"}
    assert _h(sheet.teams["A"].player("6").name) == "b822ee01815cbd6f"  # real name kept out of the repo
    assert sheet.teams["B"].player("53").captain and sheet.teams["A"].player("2").captain
    assert sum(p.starter for p in sheet.teams["A"].players) == 5
    assert sum(p.starter for p in sheet.teams["B"].players) == 5


@needs_sample
def test_sample_running_score_and_fouls_detail(sheet: S.Scoresheet) -> None:
    run = sheet.running
    assert len(run) == 66
    assert (run[-1].score_a, run[-1].score_b) == (23, 98)
    assert (run[0].team, run[0].points, run[0].scorer, run[0].kind) == ("A", 1, "6", "ft")
    threes = [e for e in run if e.points == 3]
    assert [(e.team, e.scorer, e.score_a) for e in threes] == [("A", "6", 12)]
    # the running score credits each player exactly the summary page's points
    for side, exp in (("A", SUMMARY_A), ("B", SUMMARY_B)):
        for num, row in exp.items():
            assert sum(e.points for e in run if e.team == side and e.scorer == num) == row[0]
    # quarters: each team's entries end at the quarter totals
    assert max(e.score_b for e in run if e.period == 1) == 24
    assert max(e.score_a for e in run if e.period <= 2) == 15
    assert [e.period for e in run] == sorted(e.period for e in run)
    assert sheet.teams["A"].team_fouls == {1: 2, 2: 4, 3: 4, 4: 4}
    assert sheet.teams["B"].team_fouls == {1: 3, 2: 3, 3: 3, 4: 4}
    six = sheet.teams["A"].player("6")
    assert [(f.kind, f.free_throws, f.period) for f in six.fouls] == [("P", 2, 1), ("P", None, 1), ("P", 1, 2)]
    assert sheet.teams["A"].timeouts and sheet.teams["A"].timeouts[0]["minute"] == 8
    assert sheet.warnings == []  # the sample is consistent with itself


@needs_sample
def test_sample_json_has_no_licence_numbers(sheet: S.Scoresheet) -> None:
    txt = json.dumps(sheet.to_dict(), ensure_ascii=False)
    assert not re.search(r"\d{8}", txt)
    assert re.search(r"\d{8}", json.dumps(sheet.to_dict(private=True)))
    back = S.Scoresheet.from_dict(json.loads(txt))
    assert back.final == sheet.final and len(back.running) == 66
    assert back.teams["B"].player("28").points == 28


@needs_sample
def test_extractor_reads_sample_text_and_marks() -> None:
    pages = pdftext.extract(SAMPLE)
    assert len(pages) == 2
    assert "OTTELUYHTEENVETO" in pages[1].text()
    assert "Tulos: 23 – 98" in pages[1].text()  # WinAnsi en dash
    assert any(m.kind == "circle" for m in pages[0].marks)
    assert any(s.depth > 0 for s in pages[0].spans)  # template (form XObject) text is read too


# ====================================================================================== extractor (synthetic PDFs)
def _pdf(content: bytes, fonts: str = "/F1 5 0 R", extra_objs: list[bytes] | None = None, compress: bool = True
         ) -> bytes:
    body = zlib.compress(content) if compress else content
    filt = b"/Filter /FlateDecode " if compress else b""
    objs = [b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 /MediaBox [0 0 595 842] >>",
            b"<< /Type /Page /Parent 2 0 R /Resources << /Font << " + fonts.encode() + b" >> >> /Contents 4 0 R >>",
            b"<< " + filt + b"/Length " + str(len(body)).encode() + b" >>\nstream\n" + body + b"\nendstream",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>"]
    objs += extra_objs or []
    out = b"%PDF-1.4\n"
    for i, o in enumerate(objs, 1):
        out += f"{i} 0 obj\n".encode() + o + b"\nendobj\n"
    return out + b"trailer\n<< /Root 1 0 R >>\n%%EOF\n"


def _h(t):
    return _hashlib.sha256(t.encode()).hexdigest()[:16]



def test_extractor_text_operators_and_encodings() -> None:
    content = (b"BT /F1 10 Tf 50 700 Td (Hello \\(world\\)) Tj ET\n"
               b"BT /F1 12 Tf 1 0 0 1 100 600 Tm [(Erkkil) -20 (\xe4)] TJ ET\n"
               b"BT /F2 8 Tf 2 0 0 2 10 10 Tm <0102> Tj ET\n"
               b"q 1 0 0 1 0 0 cm 0.471 0 0 RG 1.7 w 10 10 m 25 24 l S Q\n")
    diff = b"<< /Type /Font /Subtype /Type1 /BaseFont /X /Encoding << /Differences [1 /adieresis /seven] >> >>"
    pdf = _pdf(content, fonts="/F1 5 0 R /F2 6 0 R", extra_objs=[diff])
    page = pdftext.extract(pdf)[0]
    texts = {s.text: s for s in page.spans}
    assert "Hello (world)" in texts and texts["Hello (world)"].x == 50 and texts["Hello (world)"].y == 700
    _k = [k for k in texts if _h(k) == "2895e7ffabccea16"]  # real name kept out of the repo
    assert _k and texts[_k[0]].size == 12
    assert "ä7" in texts and texts["ä7"].size == 16  # Tm scale 2 x 8 pt
    line = [m for m in page.marks if m.kind == "line"][0]
    assert (line.x0, line.y0, line.x1, line.y1) == (10, 10, 25, 24) and line.color[0] == pytest.approx(0.471)
    assert S._color_name(line.color) == "red"


def test_extractor_rejects_non_pdf_and_scanned() -> None:
    with pytest.raises(pdftext.PdfError):
        pdftext.extract(b"hello, not a pdf")
    blank = _pdf(b"q 100 0 0 100 0 0 cm 0 0 m 1 1 l S Q")  # no text at all, like a scanned page
    with pytest.raises(S.ScannedSheetError):
        S.parse_pdf(blank)


def test_scanned_image_is_a_clear_error(tmp_path: Path) -> None:
    img = tmp_path / "sheet.jpg"
    img.write_bytes(b"\xff\xd8\xff\xe0 fake jpeg")
    with pytest.raises(S.ScannedSheetError, match="OCR"):
        S.read(img)
    other = tmp_path / "sheet.docx"
    other.write_bytes(b"PK\x03\x04")
    with pytest.raises(S.ScoresheetError):
        S.read(other)


# ====================================================================================== CSV / JSON
CSV = """# home: Kotijoukkue
# away: Vieraat
# final: 10-6
# quarters: 2-2, 4-0, 0-2, 4-2
team,number,name,points,fouls,ft,fg2,fg3
home,7,Matti Meikäläinen,6,2,0,3,0
A,23,Teppo Testi,4,1,0,2,0
away,5,Ville Vieras,6,4,0,3,0
"""


def test_csv_scoresheet(tmp_path: Path) -> None:
    f = tmp_path / "sheet.csv"
    f.write_text(CSV, encoding="utf-8")
    s = S.read(f)
    assert s.kind == "csv" and s.teams["A"].name == "Kotijoukkue" and s.teams["B"].name == "Vieraat"
    assert s.final == {"A": 10, "B": 6} and s.period_scores["A"] == [2, 4, 0, 4]
    assert s.teams["A"].player("7").points == 6 and s.teams["B"].player("5").n_fouls == 4
    assert s.warnings == []
    j = tmp_path / "sheet.json"
    j.write_text(json.dumps(s.to_dict()), encoding="utf-8")
    s2 = S.read(j)
    assert s2.teams["A"].player("23").name == "Teppo Testi" and s2.final == s.final


def test_csv_inconsistent_points_warn(tmp_path: Path) -> None:
    f = tmp_path / "bad.csv"
    f.write_text(CSV.replace("# final: 10-6", "# final: 12-6"), encoding="utf-8")
    s = S.read(f)
    assert any("add up to 10" in w for w in s.warnings)


# ====================================================================================== synthetic game for the Replay side
def _sheet() -> S.Scoresheet:
    s = S.Scoresheet(kind="json", source="test")
    s.teams["A"].name, s.teams["B"].name = "Home", "Away"
    for num, name, pts in (("4", "Aino", 4), ("7", "Bea", 2), ("12", "Cia", 1)):
        s.teams["A"].players.append(S.SheetPlayer("A", num, name, points=pts, played=True, sources=["summary"]))
    for num, name, pts in (("5", "Dora", 4), ("9", "Zyxa", 2)):
        s.teams["B"].players.append(S.SheetPlayer("B", num, name, points=pts, played=True, sources=["summary"]))
    seq = [("A", 2, "4"), ("B", 2, "5"), ("A", 2, "4"), ("B", 2, "9"), ("A", 2, "7"), ("A", 1, "12"),
           ("B", 2, "5")]
    a = b = 0
    for k, (team, pts, num) in enumerate(seq, 1):
        a, b = (a + pts, b) if team == "A" else (a, b + pts)
        s.running.append(S.ScoreEvent(k, team, pts, a, b, num, "ft" if pts == 1 else "fg2", 1))
    s.final = {"A": a, "B": b}
    return s


def _replay(duration: float = 120.0) -> dict:
    n = int(duration) + 1
    ts = [float(i) for i in range(n)]
    tracks = [{"id": 1, "team": "A", "i0": 0, "x": [1.0] * n}, {"id": 2, "team": "A", "i0": 10, "x": [1.0] * 40},
              {"id": 3, "team": "B", "i0": 0, "x": [1.0] * n}, {"id": 4, "team": "A", "i0": 0, "x": [1.0] * 30}]
    info = {"1": {"team": "A", "number": "4", "number_reads": 30}, "2": {"team": "A", "number": "24",
                                                                         "number_reads": 25},
            "3": {"team": "B", "number": "5", "number_reads": 20}, "4": {"team": "A", "number": "7",
                                                                         "number_reads": 12}}
    ev = [{"id": "b1", "type": "basket", "t": 10.0, "team": "A", "confidence": 0.9},
          {"id": "b2", "type": "basket", "t": 30.0, "team": "B", "confidence": 0.9},
          {"id": "b3", "type": "basket", "t": 50.0, "team": "A", "confidence": 0.9},
          {"id": "b5", "type": "basket", "t": 90.0, "team": "A", "confidence": 0.8},
          {"id": "bx", "type": "basket", "t": 85.0, "team": "B", "confidence": 0.5}]
    return {"t": ts, "tracks": tracks, "track_info": info, "events": ev,
            "game": {"teams": {"A": {"name": "Team purple", "side": "home", "side_confirmed": True,
                                     "colour_name": "purple"},
                               "B": {"name": "Team orange", "side": "away", "side_confirmed": True,
                                     "colour_name": "orange"}}}}


def _gs() -> dict:
    ann = {"b1": {"team": "A", "kind": "field_goal", "points_guess": 2, "shooter_track": "1"},
           "b2": {"team": "B", "kind": "field_goal", "points_guess": 2, "shooter_track": "3"},
           "b3": {"team": "A", "kind": "field_goal", "points_guess": 2, "shooter_track": "4"},  # sheet says #4
           "b5": {"team": "A", "kind": "field_goal", "points_guess": 2, "shooter_track": "4"},
           "bx": {"team": "B", "kind": "field_goal", "points_guess": 2}}
    return {"basket_annotations": ann, "time_range_s": [0.0, 120.0],
            "events": [{"type": "free_throw", "t": 95.0, "team": "A", "details": {"made": True}}]}


def test_numbers_seen_and_missing_player_item() -> None:
    seen = S.numbers_seen(_replay())
    assert seen["A"]["24"]["reads"] == 25 and seen["A"]["24"]["first_t"] == 10.0
    rep = S.crosscheck(None, _sheet(), _replay(), _gs(), {})
    miss = [i for i in rep["items"] if i["kind"] == "missing_player"]
    assert len(miss) == 1 and "#24" in miss[0]["title"] and miss[0]["t"] == 10.0 and miss[0]["severity"] == "check"
    assert any(p["number"] == "24" and p.get("missing") for p in rep["players"])


def test_crosscheck_with_score_bar_times() -> None:
    labels = {"initial_score": [0, 0], "events": [
        {"t_bar": 15.0, "team": "home", "points": 2, "score_after": [2, 0]},
        {"t_bar": 35.0, "team": "away", "points": 2, "score_after": [2, 2]},
        {"t_bar": 55.0, "team": "home", "points": 2, "score_after": [4, 2]},
        {"t_bar": 75.0, "team": "away", "points": 2, "score_after": [4, 4]},
        {"t_bar": 95.0, "team": "home", "points": 2, "score_after": [6, 4]},
        {"t_bar": 105.0, "team": "home", "points": 1, "score_after": [7, 4]},
        {"t_bar": 115.0, "team": "away", "points": 2, "score_after": [7, 6]}]}
    rep = S.crosscheck(None, _sheet(), _replay(), _gs(), {}, labels=labels)
    assert rep["alignment"] == "high"
    kinds = [(i["kind"], i["t"]) for i in rep["items"]]
    scorer = [i for i in rep["items"] if i["kind"] == "scorer"]
    assert len(scorer) == 1 and scorer[0]["t"] == 50.0 and "#4" in scorer[0]["title"] and "#7" in scorer[0]["title"]
    assert ("missed_basket", 70.0) in kinds  # 4-4 by #9: nothing detected
    assert ("extra_basket", 85.0) in kinds and ("missed_basket", 110.0) in kinds
    m = rep["match_stats"]
    assert m["found"] == 4 and m["ft_found"] == 1 and m["scorer_ok"] == 3 and m["scorer_bad"] == 1
    assert all("t" in i and "category" in i for i in rep["items"])


def test_crosscheck_order_alignment_with_start_score() -> None:
    sh = _sheet()
    sh.game["replay_start_score"] = [2, 2]  # this Replay starts after the first two scores
    rp = _replay()
    rp["events"] = [e for e in rp["events"] if e["id"] in ("b3", "b5")]
    rep = S.crosscheck(None, sh, rp, _gs(), {})
    got = [(p["sheet"]["score"], p["video"]["id"]) for p in rep["pairs"] if p["sheet"] and p["video"]]
    assert got[0] == ([4, 2], "b3")
    assert rep["alignment"] in ("medium", "low")


def test_align_handles_missed_and_extra_detections() -> None:
    sheet_seq = [{"team": t, "points": 2} for t in "ABABAA"]
    det = [{"team": "A", "points": 2}, {"team": "A", "points": 2}, {"team": "B", "points": 2},
           {"team": "A", "points": 2}]
    pairs = S.align(sheet_seq, det)
    matched = [(i, j) for i, j in pairs if i is not None and j is not None]
    assert all(sheet_seq[i]["team"] == det[j]["team"] for i, j in matched)
    assert len(matched) == 4
    clip = S.align(sheet_seq, det[:1], free_start=True, free_end=True)
    assert sum(1 for i, j in clip if i is not None and j is not None) == 1


def test_team_map_uses_jersey_numbers_when_unconfirmed() -> None:
    rp = _replay()
    rp["game"]["teams"]["A"].update(side="away", side_confirmed=False)  # wrong automatic guess
    rp["game"]["teams"]["B"].update(side="home", side_confirmed=False)
    tm, src = S.team_map_info(_sheet(), rp, None)
    assert tm == {"A": "A", "B": "B"} and "jersey" in src
    rp["game"]["teams"]["A"].update(side_confirmed=True)
    rp["game"]["teams"]["B"].update(side_confirmed=True)
    # a confirmed home/away that clearly contradicts the jersey numbers no longer wins silently (18.10): the
    # numbers decide and the source says so (the Overview offers the swap: scoresheet.home_away_conflict)
    tm, src = S.team_map_info(_sheet(), rp, None)
    assert tm == {"A": "A", "B": "B"} and "contradict" in src
    sh = _sheet()
    sh.game["team_map"] = {"A": "A", "B": "B"}
    assert S.team_map_info(sh, rp, None) == ({"A": "A", "B": "B"}, "manual")


def test_digits_close() -> None:
    assert S._digits_close("21", "24") and S._digits_close("4", "14") and S._digits_close("6", "76")
    assert not S._digits_close("21", "53") and not S._digits_close("7", "7")


# ====================================================================================== Replay integration
def _project(tmp_path: Path, duration: float) -> Path:
    root = tmp_path / "proj"
    (root / "results").mkdir(parents=True)
    (root / "project.json").write_text(json.dumps({"video_info": {"duration_s": duration}}), encoding="utf-8")
    (root / "results" / "replay_data.json").write_text(json.dumps(_replay()), encoding="utf-8")
    return root


def test_roster_fill_never_overwrites_user_names_without_asking(tmp_path: Path) -> None:
    root = _project(tmp_path, 120.0)
    sh = _sheet()
    # the user already typed a roster for the home team, with a different name for #4 and an override
    a = rn.Assignment(home=rn.Roster(team="Home", players=[rn.Player("4", "Aino Virtanen"), rn.Player("99", "X")]))
    a.overrides["track:1"] = {"team": "A", "number": "4", "name": "Manual"}
    rn.save_assignment(root, a)
    plan = S.roster_plan(root, sh)
    assert plan["conflicts"] == 1 and plan["A"]["conflicts"] == [("4", "Aino Virtanen", "Aino")]
    assert ("7", "Bea") in plan["A"]["new"] and not plan["B"]["has_roster"]
    S.apply_roster(root, sh, replace_conflicts=False)
    b = rn.load_assignment(root)
    assert b.home.by_number("4").name == "Aino Virtanen" and b.home.by_number("7").name == "Bea"
    assert b.home.by_number("99") is not None and b.away.by_number("5").name == "Dora"
    assert b.overrides["track:1"]["name"] == "Manual"
    S.apply_roster(root, sh, replace_conflicts=True)
    assert rn.load_assignment(root).home.by_number("4").name == "Aino"


def test_score_truth_only_for_whole_game(tmp_path: Path) -> None:
    clip = _project(tmp_path, 178.0)
    ok, why = S.write_score_truth(clip, _sheet())
    assert not ok and "not the whole game" in why and not (clip / "score_truth.json").exists()
    game = tmp_path / "game"
    (game / "results").mkdir(parents=True)
    (game / "project.json").write_text(json.dumps({"video_info": {"duration_s": 5100.0}}), encoding="utf-8")
    rp = _replay()
    ok, _ = S.write_score_truth(game, _sheet(), replay=rp)
    doc = json.loads((game / "score_truth.json").read_text(encoding="utf-8"))
    assert ok and doc["final"] == {"A": 7, "B": 6} and doc["generated_by"] == "scoresheet"
    from replay import stats
    rec = stats.reconcile({"A": 6, "B": 6}, doc, {"A": "Team purple", "B": "Team orange"})
    assert rec["truth"] == {"A": 7.0, "B": 6.0} and rec["diff"]["A"] == -1
    (game / "score_truth.json").write_text(json.dumps({"final": {"A": 1, "B": 2}}), encoding="utf-8")
    ok, why = S.write_score_truth(game, _sheet(), replay=rp)
    assert not ok and "already exists" in why


@needs_sample
def test_import_into_replay_copies_file_and_keeps_licences_out(tmp_path: Path) -> None:
    root = _project(tmp_path, 178.0)
    sh = S.import_into(root, SAMPLE)
    assert (root / "scoresheet" / SAMPLE.name).exists()
    txt = (root / "scoresheet.json").read_text(encoding="utf-8")
    assert not re.search(r"\d{8}", txt)
    loaded = S.load(root)
    assert loaded.final == {"A": 23, "B": 98} and len(loaded.running) == 66 and sh.source.startswith("scoresheet/")
    rep = S.crosscheck(root)
    assert rep["summary"] and isinstance(rep["items"], list)
    S.remove(root)
    assert S.load(root) is None and not (root / "scoresheet").exists()


# ====================================================================================== GUI (offscreen)
def test_dialog_fills_tables_roster_and_jumps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QMessageBox
    from replay.gui import scoresheet_dialog as D

    root = _project(tmp_path, 120.0)
    (root / "results" / "game_state.json").write_text(json.dumps(_gs()), encoding="utf-8")
    S.save(root, _sheet())
    monkeypatch.setattr(QMessageBox, "exec", lambda self: 0)  # never block on a modal box

    class P:
        def __init__(self, r: Path):
            self.root = r
            self.id = "p"

        def meta(self) -> dict:
            return {}

    seeks: list[float] = []

    class RT:
        def seek_to(self, t: float) -> None:
            seeks.append(t)

    class Win:
        current = P(root)
        replay_tab = RT()

    dlg = D.ScoresheetDialog(P(root), Win(), None, just_imported=True)
    try:
        assert dlg.players.rowCount() >= 6 and dlg.running.rowCount() == 7
        dlg.filter.setCurrentIndex(dlg.filter.findData("all"))
        assert dlg.checks.rowCount() >= 1
        assert rn.load_assignment(root).home.by_number("7").name == "Bea"  # roster filled (no conflicts)
        assert "Score truth" in dlg.status.text() and not (root / "score_truth.json").exists()  # a clip
        row = next(r for r in range(dlg.checks.rowCount()) if dlg.checks.item(r, 0).text())
        dlg._jump(dlg.checks, row)
        assert seeks and seeks[0] >= 0.0
        dlg.start_known.setChecked(True)
        dlg.start_a.setValue(2)
        dlg.start_b.setValue(2)
        dlg._start_changed()
        assert S.load(root).game["replay_start_score"] == [2, 2]
    finally:
        dlg.deleteLater()

    class Tab:
        p = P(root)
        win = None

    btn = D.scoresheet_button(Tab())
    btn.refresh()
    assert btn.text() == "Scoresheet…"
    S.remove(root)
    btn.refresh()
    assert btn.text() == "Add scoresheet…"
    btn.deleteLater()


def test_scoresheet_travels_in_replay_package(tmp_path: Path) -> None:
    pytest.importorskip("PySide6")
    import zipfile

    from fake_project import make_project
    from replay.gui.projects import ProjectStore

    store = ProjectStore(tmp_path / "projects")
    make_project(store.root / "game-abc123")
    p = store.get("game-abc123")
    src = tmp_path / "sheet.csv"
    src.write_text(CSV, encoding="utf-8")
    S.import_into(p.root, src)
    store.export(p.id, tmp_path / "out.aareplay")
    with zipfile.ZipFile(tmp_path / "out.aareplay") as z:
        assert {"scoresheet.json", "scoresheet/sheet.csv"} <= set(z.namelist())
    q = store.import_package(tmp_path / "out.aareplay")
    assert S.load(q.root).final == {"A": 10, "B": 6}
