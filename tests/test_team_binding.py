"""Stable team identity (replay/team_binding.py): teams bound to jersey colours, not to cluster order."""

import json

import cv2
import numpy as np

from replay import team_appearance as ta
from replay import team_binding as tb
from replay.metadata import load_metadata, metadata_from_dict
from replay.teams import TeamResult, swap_ab

ORANGE_LAB, PURPLE_LAB = [33.3, 34.3, 26.2], [14.8, 18.1, -13.6]


def _hist(sector: int) -> list[float]:
    h = np.full(ta.DESC_DIM, 0.01)
    h[sector * ta.N_CHROMA + 2] = 0.5
    return (h / h.sum()).tolist()


def _result(flipped: bool = False, seeded: bool = False) -> TeamResult:
    """Orange players = tracks 1-3, purple = 4-6; flipped = the classifier called purple 'A' this time."""
    r = TeamResult({1: "A", 2: "A", 3: "A", 4: "B", 5: "B", 6: "B", 7: "referee"}, {t: 0.9 for t in range(1, 8)},
                   {"A": ORANGE_LAB, "B": PURPLE_LAB}, seeded, {"A": "#d21c00", "B": "#b603d2"}, [],
                   {"A": _hist(1), "B": _hist(10)}, "video3")
    return swap_ab(r) if flipped else r


def _project(tmp_path, meta: dict | None = None):
    root = tmp_path / "proj"
    (root / "results").mkdir(parents=True)
    (root / "meta.json").write_text(json.dumps(meta or {"home_team": "Home", "away_team": "Away",
                                                       "team_colors_auto": True}), encoding="utf-8")
    return root / "results", root / "meta.json"


def _orange_label(res: TeamResult) -> str:
    return res.track_team[1]


def test_colour_names():
    assert tb.lab_colour_name(ORANGE_LAB) == "orange"
    assert tb.lab_colour_name(PURPLE_LAB) == "purple"
    assert tb.colour_name("#1144aa") == "blue" and tb.colour_name("#f5f5f5") == "white"
    assert tb.colour_name("#141414") == "black"
    assert tb.colour_team_names("#d21c00", "#b603d2", ("orange", "purple")) == {"A": "Team orange",
                                                                               "B": "Team purple"}
    n = tb.colour_team_names("#2040ff", "#101850")
    assert n["A"] != n["B"] and "light" in n["A"] and "dark" in n["B"]


def test_first_build_binds_and_names_by_colour_until_confirmed(tmp_path):
    out, meta_path = _project(tmp_path)
    teams, meta, b = tb.bind(out, metadata_from_dict(json.loads(meta_path.read_text("utf-8"))), _result())
    assert b["confirmed"] is False and b["home"]["colour_name"] == "orange"
    # placeholder "Home"/"Away" are not shown for a guess: colour names
    assert (meta.home.name, meta.away.name) == ("Team orange", "Team purple")
    assert meta.to_public_dict()["teams"]["A"]["side_confirmed"] is False
    stored = json.loads(meta_path.read_text("utf-8"))
    assert stored["home_team"] == "Home" and stored["team_colors_auto"] is True  # user fields untouched
    assert tb.valid(stored["team_binding"])


def test_rebuilds_never_swap_teams(tmp_path):
    out, meta_path = _project(tmp_path)
    tb.bind(out, load_metadata(meta_path), _result())
    for flipped in (True, False, True, True):
        res, meta, _ = tb.bind(out, load_metadata(meta_path), _result(flipped=flipped))
        assert _orange_label(res) == "A" and res.track_team[4] == "B" and res.track_team[7] == "referee"
        assert res.colors["A"] == "#d21c00" and meta.home.name == "Team orange"


def test_matching_works_from_colours_alone(tmp_path):
    """Binding made without video (pass-1 space) still matches a video-space classification by colour."""
    out, meta_path = _project(tmp_path)
    first = _result()
    first.space = "pass1_3"
    tb.bind(out, load_metadata(meta_path), first)
    res, _, b = tb.bind(out, load_metadata(meta_path), _result(flipped=True))
    assert _orange_label(res) == "A"
    assert set(b["home"]["hist"]) == {"pass1_3", "video3"}  # the new space is learnt, sides unchanged


def test_swap_toggle_moves_binding_once(tmp_path):
    out, meta_path = _project(tmp_path)
    tb.bind(out, load_metadata(meta_path), _result())
    res, meta, b = tb.bind(out, load_metadata(meta_path), _result(), swap_teams=True)
    assert _orange_label(res) == "B" and b["confirmed"] and b["swap_flag"] is True
    assert res.colors["A"] == "#b603d2"  # colours follow the players
    for flipped in (False, True):  # the setting stays on: no further swapping
        res, _, _ = tb.bind(out, load_metadata(meta_path), _result(flipped=flipped), swap_teams=True)
        assert _orange_label(res) == "B"
    res, _, _ = tb.bind(out, load_metadata(meta_path), _result(), swap_teams=False)  # toggled back
    assert _orange_label(res) == "A"


def test_user_names_and_sides_from_overview(tmp_path):
    out, meta_path = _project(tmp_path, {"home_team": "KlaNMKY", "away_team": "LePy", "team_colors_auto": True})
    _, meta, _ = tb.bind(out, load_metadata(meta_path), _result())
    assert meta.home.name == "Team orange"  # real names are not attached to a guessed colour
    m = json.loads(meta_path.read_text("utf-8"))
    ov = tb.team_overview(m)
    assert ov["A"]["colour_name"] == "orange" and not ov["A"]["confirmed"]
    # the user: orange = LePy (away), purple = KlaNMKY (home)
    rows = [{"label": "A", "name": "LePy", "side": "away", "color": ov["A"]["color"]},
            {"label": "B", "name": "KlaNMKY", "side": "home", "color": ov["B"]["color"]}]
    new, clear = tb.overview_save(m, rows, dirty=True, confirmed=True, swap_flag=False)
    assert clear and new["home_team"] == "KlaNMKY" and new["away_team"] == "LePy"
    meta_path.write_text(json.dumps(new), encoding="utf-8")
    for flipped in (False, True, False):
        res, meta, b = tb.bind(out, load_metadata(meta_path), _result(flipped=flipped))
        assert _orange_label(res) == "B" and res.track_team[4] == "A"
        assert (meta.home.name, meta.away.name) == ("KlaNMKY", "LePy")
        assert b["confirmed"] and b["source"] == "user"
        pub = meta.to_public_dict()["teams"]
        assert pub["A"]["colour_name"] == "purple" and pub["B"]["side"] == "away" and pub["B"]["side_confirmed"]
    try:
        tb.apply_user_choice(new, [dict(rows[0], side="home"), dict(rows[1], side="home")], False)
        raise AssertionError("two home teams accepted")
    except ValueError:
        pass


def test_overview_save_without_team_edits_changes_nothing(tmp_path):
    out, meta_path = _project(tmp_path)
    tb.bind(out, load_metadata(meta_path), _result())
    m = json.loads(meta_path.read_text("utf-8"))
    rows = [{"label": k, "name": "x", "side": tb.SIDE_OF[k], "color": "#000000"} for k in ("A", "B")]
    new, clear = tb.overview_save(m, rows, dirty=False, confirmed=False, swap_flag=False)
    assert new == m and not clear


def test_picked_players_override_the_binding(tmp_path):
    out, meta_path = _project(tmp_path)
    tb.bind(out, load_metadata(meta_path), _result())
    m = json.loads(meta_path.read_text("utf-8"))
    m["team_samples"] = {"A": [[1, 2, 3, 4, 5, 6]], "B": [[6, 5, 4, 3, 2, 1]]}
    meta_path.write_text(json.dumps(m), encoding="utf-8")
    # the user picked the purple players as home: the seeded classification says purple = A
    res, _, b = tb.bind(out, load_metadata(meta_path), _result(flipped=True, seeded=True))
    assert _orange_label(res) == "B" and b["source"] == "samples" and b["confirmed"]
    res, _, _ = tb.bind(out, load_metadata(meta_path), _result(flipped=False))  # later noisy rebuild
    assert _orange_label(res) == "B"


def test_binding_without_meta_json_goes_to_out_folder(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    tb.bind(out, metadata_from_dict({}), _result())
    assert (out / tb.FILENAME).exists()
    res, _, _ = tb.bind(out, metadata_from_dict({}), _result(flipped=True))
    assert _orange_label(res) == "A"


def test_one_team_found_keeps_its_label(tmp_path):
    out, meta_path = _project(tmp_path)
    tb.bind(out, load_metadata(meta_path), _result())
    one = _result(flipped=True)  # purple = A, and orange not found at all
    one.track_team = {t: v for t, v in one.track_team.items() if v != "B"}
    for d in (one.centroids, one.colors, one.hists):
        d.pop("B")
    res, _, _ = tb.bind(out, load_metadata(meta_path), one)
    assert res.track_team[4] == "B"


def test_scorebar_home_from_box_colours():
    runs = [{"x": 0.30, "w": 0.05, "lab": [50, 40, 45]}, {"x": 0.55, "w": 0.05, "lab": [25, 25, -25]}]
    assert tb.scorebar_home(runs, {"A": ORANGE_LAB, "B": PURPLE_LAB}) == "A"
    assert tb.scorebar_home([dict(runs[0], x=0.7), runs[1]],
                            {"A": ORANGE_LAB, "B": PURPLE_LAB}) == "B"
    assert tb.scorebar_home(runs[:1], {"A": ORANGE_LAB, "B": PURPLE_LAB}) is None  # one coloured box only
    assert tb.scorebar_home(runs, {"A": [80, 0, 0], "B": PURPLE_LAB}) is None  # white kit: no colour cue


def test_scorebar_runs_on_a_synthetic_broadcast(tmp_path):
    """Static bar with a purple box left and an orange box right over a changing scene."""
    path = tmp_path / "bar.avi"
    w, h = 320, 180
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (w, h))
    rng = np.random.default_rng(0)
    for _ in range(60):
        img = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
        img[160:172, 60:250] = 245  # white bar
        img[160:172, 70:110] = (130, 30, 110)  # purple box (BGR)
        img[160:172, 180:220] = (30, 90, 230)  # orange box
        img[162:170, 230:245] = 20  # clock
        vw.write(img)
    vw.release()
    runs = tb.scorebar_runs(path, [230 / w, 162 / h, 245 / w, 170 / h])
    assert len(runs) == 2
    assert tb.scorebar_home(runs, {"A": ORANGE_LAB, "B": PURPLE_LAB}) == "B"


def test_overview_tab_shows_and_saves_teams(tmp_path):
    """Overview: swatches + names + Home/Away of the bound teams; saving flips the binding and rebuilds."""
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from fake_project import make_project
    from replay.gui.app import OverviewTab
    from replay.gui.projects import Project
    _app = QApplication.instance() or QApplication([])  # noqa: F841
    root = make_project(tmp_path / "p")
    meta_path = root / "meta.json"
    meta_path.write_text(json.dumps({"home_team": "Home", "away_team": "Away", "team_colors_auto": True}), "utf-8")
    tb.bind(root / "results", load_metadata(meta_path), _result())

    class Win:
        class jobs:
            submitted = []

            @classmethod
            def submit(cls, pid, kind, **kw):
                cls.submitted.append(kind)

            @classmethod
            def cancel(cls, pid):
                pass

        def refresh_list(self):
            pass

    tab = OverviewTab(Win())
    tab.load(Project(root))
    jersey, sw, nm, side = tab.team_rows["A"]
    assert jersey.text() == "orange jerseys" and nm.text() == "Team orange" and side.currentData() == "home"
    assert sw.color == "#d21c00" and not tab.teams_ok.isChecked() and "Please check" in tab.teams_status.text()
    side.setCurrentIndex(1)  # orange = away -> purple becomes home automatically
    assert tab.team_rows["B"][3].currentData() == "home" and tab.teams_ok.isChecked()
    nm.setText("LePy")
    tab.team_rows["B"][2].setText("KlaNMKY")
    tab.save()
    m = json.loads(meta_path.read_text("utf-8"))
    assert m["home_team"] == "KlaNMKY" and m["away_team"] == "LePy"
    assert m["team_binding"]["home"]["colour_name"] == "purple" and m["team_binding"]["confirmed"]
    res, meta, _ = tb.bind(root / "results", load_metadata(meta_path), _result())
    assert _orange_label(res) == "B" and meta.away.name == "LePy"
    assert tab.team_rows["A"][2].text() == "KlaNMKY"  # rows re-read: row 1 is now the home team
