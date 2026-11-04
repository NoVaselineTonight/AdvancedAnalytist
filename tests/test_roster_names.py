"""Team rosters (replay/roster_names.py): data model, CSV, mapping, and the guarded hooks."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from replay import roster_names as rn


def _home() -> rn.Roster:
    return rn.Roster(team="KlaNMKY", kit="purple", players=[rn.Player("7", "Virtanen", "G"),
                                                            rn.Player("12", "Korhonen", "F"),
                                                            rn.Player("00", "Testaaja")])


def _away() -> rn.Roster:
    return rn.Roster(team="LePy", kit="orange", players=[rn.Player("7", "Mäkinen"), rn.Player("23", "Laine")])


def _replay() -> dict:
    return {"t": [0.0, 0.1, 0.2],
            "tracks": [{"id": "5", "team": "A", "x": [1, 2, 3]}, {"id": "6", "team": "A", "x": [1, None, 3]},
                       {"id": "9", "team": "B", "x": [1, 2, 3]}, {"id": "40", "team": "A", "x": [1]}],
            "track_info": {"5": {"number": "7", "team": "A"}, "6": {"number": "12", "team": "A"},
                           "9": {"number": "7", "team": "B"}}}


@pytest.fixture()
def proj(tmp_path: Path) -> Path:
    root = tmp_path / "p"
    (root / "results").mkdir(parents=True)
    (root / "results" / "replay_data.json").write_text(json.dumps(_replay()), encoding="utf-8")
    return root


def test_norm_number():
    assert rn.norm_number("#7") == "7"
    assert rn.norm_number(" 07 ") == "7"
    assert rn.norm_number("00") == "00"
    assert rn.norm_number("0") == "0"
    assert rn.norm_number(23) == "23"


def test_csv_roundtrip_and_variants():
    r = _home()
    text = rn.roster_to_csv(r)
    assert text.splitlines()[0] == "number,name,position"
    back = rn.players_from_csv(text)
    assert [(p.number, p.name, p.position) for p in back] == [(p.number, p.name, p.position) for p in r.players]
    # no header, semicolons, BOM, reordered header
    assert [p.name for p in rn.players_from_csv("﻿4;Aho;C\n5;Bäck;\n")] == ["Aho", "Bäck"]
    ps = rn.players_from_csv("name,number\nAho,4\n")
    assert (ps[0].number, ps[0].name) == ("4", "Aho")
    with pytest.raises(ValueError):
        rn.players_from_csv("\n\n")


def test_library_roundtrip(tmp_path: Path, monkeypatch):
    lib = tmp_path / "lib" / "rosters.json"
    monkeypatch.setenv(rn.LIBRARY_ENV, str(lib))
    assert rn.library_path() == lib
    assert rn.load_library() == []
    h = _home()
    rn.upsert_library(h)
    rn.upsert_library(_away())
    h.players.append(rn.Player("4", "Aho"))
    rn.upsert_library(h)  # same id: replaced, not duplicated
    got = rn.load_library()
    assert sorted(r.team for r in got) == ["KlaNMKY", "LePy"]
    assert len(next(r for r in got if r.id == h.id).players) == 4


def test_no_roster_json_changes_nothing(proj: Path):
    assert rn.load_assignment(proj) is None
    assert rn.map_for(proj) is None
    evs = [{"id": "e", "label": "Pass #7 -> #12", "team": "A"}]
    assert rn.decorate_events(proj, _replay(), evs) is evs
    data = _replay()
    before = json.dumps(data, sort_keys=True)
    rn.annotate_replay_data(proj, data)
    assert json.dumps(data, sort_keys=True) == before
    out = proj / "exp"
    out.mkdir()
    (out / "players.csv").write_text("t,track_id,team,jersey_number\n0,5,A,7\n", encoding="utf-8")
    assert rn.augment_export(proj, out) is False
    assert (out / "players.csv").read_text(encoding="utf-8") == "t,track_id,team,jersey_number\n0,5,A,7\n"
    assert rn.ai_names(proj, {"5": {"number": "7"}}, {"5": "A"}) == {}


def test_mapping_numbers_and_manual_override_wins(proj: Path):
    a = rn.Assignment(_home(), _away())
    rn.save_assignment(proj, a)
    rm = rn.map_for(proj)
    assert rm.name_for_track("5") == "Virtanen"
    assert rm.name_for_track("9") == "Mäkinen"  # same number, other team
    assert rm.name_for_track("40") == ""
    a.overrides["track:5"] = {"team": "A", "number": "12"}
    a.overrides["track:40"] = {"team": "A", "number": "00"}
    a.overrides["track:6"] = {"none": True}
    rn.save_assignment(proj, a)
    rm = rn.map_for(proj)
    assert rm.name_for_track("5") == "Korhonen"
    assert rm.name_for_track("40") == "Testaaja"
    assert rm.name_for_track("6") == ""
    # persisted
    assert rn.load_assignment(proj).overrides["track:40"]["number"] == "00"


def test_lineup_slots(proj: Path):
    rn.save_assignment(proj, rn.Assignment(_home(), _away(), {"slot:B2": {"name": "Guest"}}))
    gs = {"lineup": {"slot_of_track": {"77": "A1"}, "numbers": {"A1": "12", "B1": "23"}, "slots": {"A1": [], "B1": []}}}
    (proj / "results" / "game_state.json").write_text(json.dumps(gs), encoding="utf-8")
    rm = rn.map_for(proj)
    assert rm.name_for_track("77") == "Korhonen"  # slot number
    assert rm.player_for_slot("B1").name == "Laine"
    assert rm.player_for_slot("B2").name == "Guest"
    seen = rn.players_seen(proj)
    keys = [s["key"] for s in seen]
    assert "slot:A1" in keys and "track:5" in keys and "track:40" not in keys  # short anonymous piece hidden


def test_event_titles(proj: Path):
    rn.save_assignment(proj, rn.Assignment(_home(), _away()))
    evs = [{"id": "p", "team": "A", "label": "Pass #7 -> #12", "suffix": "(5 m)",
            "details": {"passer_track": "5", "receiver_track": "6"}},
           {"id": "s", "team": "B", "label": "Steal by #7", "suffix": "(from #12)",
            "details": {"stealer_track": "9", "lost_by_track": "6"}},
           {"id": "b", "team": "A", "label": "Basket"}]
    out = rn.decorate_events(proj, _replay(), evs)
    assert out[0]["label"] == "Pass #7 Virtanen -> #12 Korhonen"
    assert out[1]["label"] == "Steal by #7 Mäkinen"
    assert out[1]["suffix"] == "(from #12 Korhonen)"
    assert out[2]["label"] == "Basket"
    assert evs[0]["label"] == "Pass #7 -> #12"  # input untouched
    # idempotent
    again = rn.decorate_events(proj, _replay(), out)
    assert again[0]["label"] == out[0]["label"]
    # team fallback (no track ids) stays out of the bracket naming the other team
    rm = rn.map_for(proj)
    assert rn.decorate_text("Steal by #7 (from #12)", rm, (), "B") == "Steal by #7 Mäkinen (from #12)"


def test_annotate_and_dot_name(proj: Path):
    rn.save_assignment(proj, rn.Assignment(_home(), None, {"track:40": {"name": "Walk-on"}}))
    data = rn.annotate_replay_data(proj, _replay())
    assert data["roster_names"]["tracks"]["5"] == "Virtanen"
    assert data["roster_names"]["tracks"]["40"] == "Walk-on"
    assert "9" not in data["roster_names"]["tracks"]  # away roster not set
    assert rn.dot_name(data, "123", "A", "12") == "Korhonen"  # lineup slot number shown on a dot
    assert rn.dot_name(data, "123", "B", "12") == ""
    assert rn.dot_name({}, "5", "A", "7") == ""


def test_augment_export(proj: Path):
    rn.save_assignment(proj, rn.Assignment(_home(), _away()))
    out = proj / "exp"
    out.mkdir()
    (out / "players.csv").write_text("t,track_id,team,jersey_number,x\n0,5,A,7,1\n0,9,B,7,2\n0,40,A,,3\n",
                                     encoding="utf-8")
    (out / "track_summary.csv").write_text("track_id,team,jersey_number,jersey_number_reads\n6,A,12,4\n",
                                           encoding="utf-8")
    (out / "events.csv").write_text("event_id,team,title\np,A,Pass #7 -> #12\n", encoding="utf-8")
    assert rn.augment_export(proj, out, include_names=True)
    rows = list(csv.reader((out / "players.csv").read_text(encoding="utf-8").splitlines()))
    assert rows[0] == ["t", "track_id", "team", "jersey_number", "player_name", "x"]
    assert [r[4] for r in rows[1:]] == ["Virtanen", "Mäkinen", ""]
    rows = list(csv.reader((out / "track_summary.csv").read_text(encoding="utf-8").splitlines()))
    assert rows[0][3] == "player_name" and rows[1][3] == "Korhonen"
    assert "Pass #7 Virtanen -> #12 Korhonen" in (out / "events.csv").read_text(encoding="utf-8")
    assert "Laine" in (out / "roster.csv").read_text(encoding="utf-8")
    # second call does not add the column twice
    rn.augment_export(proj, out, include_names=True)
    assert (out / "players.csv").read_text(encoding="utf-8").splitlines()[0].count("player_name") == 1


def test_ai_names_and_catalog(proj: Path):
    rn.save_assignment(proj, rn.Assignment(_home(), _away()))
    names = rn.ai_names(proj, _replay()["track_info"], {"5": "A", "6": "A", "9": "B"})
    assert names == {"5": ("7", "Virtanen"), "6": ("12", "Korhonen"), "9": ("7", "Mäkinen")}
    assert rn.ai_label_name("Matti  Virtanen") == "Matti_Virtanen"
    assert rn.ai_suffix(("7", "Matti Virtanen"), True) == " Matti_Virtanen"
    # a lineup slot label without its own track number gets the slot's number: "A1#12 Korhonen"
    gs = {"lineup": {"slot_of_track": {"77": "A1"}, "numbers": {"A1": "12"}}}
    names = rn.ai_names(proj, {}, {"77": "A"}, gs)
    assert names == {"77": ("12", "Korhonen")}
    assert "A1" + rn.ai_suffix(names["77"], False) == "A1#12 Korhonen"

    class It:
        def __init__(self, title, players, cats):
            self.title, self.players, self.cats = title, players, cats

    class Cat:
        track_number = {"5": "7", "6": "12"}
        track_team = {"5": "A", "6": "A"}
        track_slot: dict = {}
        items = [It("Pass #7 -> #12 – KlaNMKY", ["5", "6"], {"pass": "A"})]

    cat = rn.decorate_catalog(proj, Cat())
    assert cat.items[0].title == "Pass #7 Virtanen -> #12 Korhonen – KlaNMKY"
    ch = rn.decorate_choices(cat, [("num:A:7", "#7 – KlaNMKY"), ("slot:A1", "Slot A1 – KlaNMKY (x)")])
    assert ch[0][1] == "#7 Virtanen – KlaNMKY"
    assert ch[1][1] == "Slot A1 – KlaNMKY (x)"
    assert rn.decorate_choices(object(), ch) is ch


# ====================================================================================== GUI (offscreen)
@pytest.fixture(scope="module")
def app():
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_dialog_and_tooltip_offscreen(proj: Path, app, monkeypatch, tmp_path: Path):
    monkeypatch.setenv(rn.LIBRARY_ENV, str(tmp_path / "lib.json"))
    from PySide6.QtCore import QPoint
    from replay.gui.roster_dialog import RosterDialog, dot_at, NOBODY
    from replay.gui.widgets import ReplayCourt

    class P:
        root = proj

        def meta(self):
            return {"home_team": "KlaNMKY", "away_team": "LePy"}

    rn.save_library([_home(), _away()])
    dlg = RosterDialog(P())
    ed = dlg.editors["A"]
    ed.pick.setCurrentIndex(ed.pick.findText("KlaNMKY – purple"))
    assert ed.roster is not None and ed.table.rowCount() == 3
    # track 5 (#7) seen: assign it by hand to nobody, track 6 to #00
    rows = {r["key"]: i for i, r in enumerate(dlg.seen_rows)}
    cb5 = dlg.seen.cellWidget(rows["track:5"], 4)
    cb5.setCurrentIndex(cb5.findData(NOBODY))
    cb6 = dlg.seen.cellWidget(rows["track:6"], 4)
    cb6.setCurrentIndex(cb6.findData("00\tTestaaja"))
    dlg.save()
    a = rn.load_assignment(proj)
    assert a.home.team == "KlaNMKY" and a.away is None
    assert a.overrides == {"track:5": {"none": True}, "track:6": {"team": "A", "number": "00", "name": "Testaaja"}}
    rm = rn.map_for(proj)
    assert rm.name_for_track("6") == "Testaaja" and rm.name_for_track("5") == ""
    # reopening shows the saved choices
    dlg2 = RosterDialog(P())
    rows = {r["key"]: i for i, r in enumerate(dlg2.seen_rows)}
    assert dlg2.seen.cellWidget(rows["track:6"], 4).currentText() == "#00 Testaaja"

    # 2D tooltip: the dot under the mouse
    data = {"t": [0.0, 0.1, 0.2], "court": {"length": 28, "width": 15},
            "tracks": [{"id": "6", "team": "A", "i0": 0, "x": [1400, 1400, 1400], "y": [750, 750, 750],
                        "f": [1, 1, 1]}],
            "track_info": {"6": {"number": "12", "team": "A"}}}
    rn.annotate_replay_data(proj, data)
    court = ReplayCourt()
    court.resize(600, 340)
    court.set_data(data)
    court.set_time(0.1)
    c = court.rect().center()
    hit = dot_at(court, QPoint(c.x(), c.y()))
    assert hit is not None and str(hit[0]["id"]) == "6"
    assert rn.dot_name(data, "6", "A", "12") == "Testaaja"
    assert dot_at(court, QPoint(2, 2)) is None


def test_same_library_roster_for_both_teams_is_not_shared(proj: Path, app, monkeypatch, tmp_path: Path):
    monkeypatch.setenv(rn.LIBRARY_ENV, str(tmp_path / "lib2.json"))
    from replay.gui.roster_dialog import RosterDialog

    class P:
        root = proj

        def meta(self):
            return {}

    rn.save_library([_home()])
    dlg = RosterDialog(P())
    ea, eb = dlg.editors["A"], dlg.editors["B"]
    for ed in (ea, eb):
        ed.pick.setCurrentIndex(ed.pick.findText("KlaNMKY – purple"))
    assert ea.roster is not None and eb.roster is not None and ea.roster is not eb.roster
    eb.table.item(0, 1).setText("Someone Else")
    eb.commit()
    ea.commit()
    assert ea.roster.players[0].name == "Virtanen"
    dlg.save()
    a = rn.load_assignment(proj)
    assert a.home.players[0].name == "Virtanen" and a.away.players[0].name == "Someone Else"
    assert a.home.id != a.away.id
    lib = rn.load_library()
    assert len(lib) == 2 and any(r.id == a.home.id and r.players[0].name == "Virtanen" for r in lib)


def test_dot_at_uses_last_paint(proj: Path, app):
    from PySide6.QtCore import QPoint
    from PySide6.QtGui import QImage
    from replay.gui.roster_dialog import dot_at
    from replay.gui.widgets import ReplayCourt

    data = {"t": [0.0, 0.1, 0.2], "court": {"length": 28, "width": 15},
            "tracks": [{"id": "6", "team": "A", "i0": 0, "x": [1400] * 3, "y": [750] * 3, "f": [1, 1, 1]}],
            "track_info": {"6": {"number": "12", "team": "A"}}}
    court = ReplayCourt()
    court.resize(600, 340)
    court.set_data(data)
    court.set_time(0.1)
    img = QImage(600, 340, QImage.Format.Format_ARGB32)
    court.render(img)  # paint once: dot_at must use the painted dots
    c = court.rect().center()
    if court.dots:
        assert dot_at(court, QPoint(c.x(), c.y())) == court.dot_at(QPoint(c.x(), c.y()))
    hit = dot_at(court, QPoint(c.x(), c.y()))
    assert hit is not None and str(hit[0]["id"]) == "6"
