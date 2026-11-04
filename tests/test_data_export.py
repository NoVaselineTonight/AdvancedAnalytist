import csv
import json
from pathlib import Path

import pytest
from fake_project import make_project

from replay.data_export import (BALL_COLUMNS, EVENT_COLUMNS, MARK_COLUMNS, PLAYER_COLUMNS, POSSESSION_COLUMNS,
                                SUMMARY_COLUMNS, export_simple)

FILES = ["README_FOR_AI.md", "game.json", "players.csv", "ball.csv", "events.csv", "events.json",
         "timeline_marks.csv", "timeline.json", "track_summary.csv", "possession_summary.csv", "replay_2d.html",
         "report.md"]


def _csv(path):
    with open(path, newline="", encoding="utf-8") as fh:
        r = list(csv.reader(fh))
    return r[0], [dict(zip(r[0], row)) for row in r[1:]]


def test_export_simple_files_and_columns(tmp_path):
    proj = make_project(tmp_path / "proj")
    out = export_simple(proj, tmp_path / "out")
    assert out.parent == tmp_path / "out"
    assert out.name.startswith("Kouvot vs Honka_data_")
    for f in FILES:
        assert (out / f).is_file(), f

    for name, cols in [("players.csv", PLAYER_COLUMNS), ("ball.csv", BALL_COLUMNS), ("events.csv", EVENT_COLUMNS),
                       ("timeline_marks.csv", MARK_COLUMNS), ("track_summary.csv", SUMMARY_COLUMNS),
                       ("possession_summary.csv", POSSESSION_COLUMNS)]:
        header, _ = _csv(out / name)
        assert header == cols, name

    _, players = _csv(out / "players.csv")
    assert len(players) == 40 * 3 + 20
    assert {p["role"] for p in players} == {"player", "referee", "unknown"}
    assert {p["team_name"] for p in players if p["team"] == "A"} == {"Kouvot"}
    assert sum(p["interpolated"] == "1" for p in players) == 4

    _, ball = _csv(out / "ball.csv")
    assert len(ball) == 40
    assert ball[0]["state"] == "possessed" and ball[0]["owner_track_id"] == "1" and ball[0]["owner_team"] == "A"
    assert ball[-1]["state"] == "estimated" and ball[-1]["owner_track_id"] == ""

    # running detected score, dismissed event excluded, user's title applied
    _, events = _csv(out / "events.csv")
    assert [e["event_id"] for e in events] == ["basket-30-L", "basket-60-R", "basket-90-L"]
    assert [(e["score_A"], e["score_B"]) for e in events] == [("2", "0"), ("2", "2"), ("4", "2")]
    assert events[1]["title"] == "Honka layup" and events[1]["note"] == "contested"
    ej = json.loads((out / "events.json").read_text(encoding="utf-8"))
    assert "NOT the official score" in ej["warning"]
    assert ej["detected_final_score_by_label"] == {"A": 4, "B": 2}
    assert [e["id"] for e in ej["dismissed_by_user"]] == ["basket-99-R"]

    _, marks = _csv(out / "timeline_marks.csv")
    user = [m for m in marks if m["track_kind"] == "user"]
    auto = [m for m in marks if m["track_kind"] == "auto"]
    assert len(user) == 1 and user[0]["note"] == "Kouvot run the floor after a steal" and user[0]["t_end"] == "3.5"
    assert sorted(m["event_id"] for m in auto) == ["basket-30-L", "basket-60-R", "basket-90-L"]
    tl = json.loads((out / "timeline.json").read_text(encoding="utf-8"))
    assert tl["dismissed_event_ids"] == ["basket-99-R"] and tl["tracks"][0]["name"] == "Notes"

    _, summary = _csv(out / "track_summary.csv")
    t1 = next(s for s in summary if s["track_id"] == "1")
    assert t1["team_name"] == "Kouvot"
    # moves 0.3 m per 0.1 s = 3 m/s; measured-only distance over 3.9 s with interpolated rows skipped
    assert float(t1["mean_speed_m_s"]) == pytest.approx(3.0, abs=0.01)
    assert float(t1["max_speed_m_s"]) == pytest.approx(3.0, abs=0.01)
    assert 9.0 < float(t1["distance_m"]) < 11.8
    assert float(t1["pct_interpolated"]) == 10.0

    _, poss = _csv(out / "possession_summary.csv")
    a = next(p for p in poss if p["team"] == "A")
    assert a["frames"] == "25" and float(a["seconds"]) == 2.5 and a["pct_of_possessed_time"] == "100.0"
    assert a["team_possessions"] == "1"

    game = json.loads((out / "game.json").read_text(encoding="utf-8"))
    assert game["game"]["teams"]["A"]["name"] == "Kouvot"
    assert game["court"]["baskets"] == {"L": [1.575, 7.5], "R": [26.425, 7.5]}
    assert game["coverage"]["calibration"]["valid_pct"] == 97.5
    assert "team_centroids_lab" not in game["coverage"]["people"]
    assert game["sampling"]["analysed_frames_per_second"] == 10.0
    assert game["exported_at"]

    html = (out / "replay_2d.html").read_text(encoding="utf-8")
    assert "/*__DATA__*/null" not in html and '"events":[' in html and "Honka layup" in html

    readme = (out / "README_FOR_AI.md").read_text(encoding="utf-8")
    for s in ["Kouvot", "Tapiolan Honka", "#ff8800", "97.5", "10.00 analysed frames per second",
              "Kouvot 4 – 2 Tapiolan Honka", "Fast break", "(1.575, 7.5)", "(26.425, 7.5)", "interpolated",
              "not official", "Output format"]:
        assert s in readme, s


def test_second_export_creates_new_folder(tmp_path):
    proj = make_project(tmp_path / "proj")
    a = export_simple(proj, tmp_path / "out")
    b = export_simple(proj, tmp_path / "out")
    assert a != b and a.is_dir() and b.is_dir()
    assert (b / "players.csv").is_file()


def test_export_without_events_or_timeline(tmp_path):
    proj = make_project(tmp_path / "proj", events=False, timeline=False)
    out = export_simple(proj, tmp_path / "out")
    _, events = _csv(out / "events.csv")
    assert events == []
    _, marks = _csv(out / "timeline_marks.csv")
    assert marks == []
    readme = (out / "README_FOR_AI.md").read_text(encoding="utf-8")
    assert "No baskets were detected" in readme and "No user timeline" in readme
    html = (out / "replay_2d.html").read_text(encoding="utf-8")
    assert '"events":[]' in html


def test_export_requires_tracking(tmp_path):
    proj = make_project(tmp_path / "proj")
    (proj / "results" / "tracking.json").unlink()
    with pytest.raises(ValueError, match="build the 2D replay"):
        export_simple(proj, tmp_path / "out")
    assert not (tmp_path / "out").exists() or not any((tmp_path / "out").iterdir())


def test_template_still_works_without_events():
    t = (Path(__file__).parents[1] / "replay" / "viewer" / "template.html").read_text(encoding="utf-8")
    assert "Array.isArray(D.events) ? D.events : []" in t
    assert t.count("/*__DATA__*/null") == 1


def test_export_uses_the_replays_own_court_size(tmp_path):
    """The program-wide court (replay.court) belongs to whatever was built last; the export must describe this
    Replay's court (stored in its replay_data.json) without changing the global state."""
    from replay import court

    proj = make_project(tmp_path / "proj")
    rd_path = proj / "results" / "replay_data.json"
    rd = json.loads(rd_path.read_text(encoding="utf-8"))
    rd["court"] = {"length": 26.0, "width": 14.0, "lines": [], "baskets": [[1.575, 7.0], [24.425, 7.0]]}
    rd_path.write_text(json.dumps(rd), encoding="utf-8")
    before = court.dimensions()
    out = export_simple(proj, tmp_path / "out")
    assert court.dimensions() == before
    c = json.loads((out / "game.json").read_text(encoding="utf-8"))["court"]
    assert (c["length_m"], c["width_m"]) == (26.0, 14.0)
    assert c["baskets"]["R"] == [24.425, 7.0]
    assert max(p[0] for line in c["lines"] for p in line) == 26.0
    readme = (out / "README_FOR_AI.md").read_text(encoding="utf-8")
    assert "26 m × 14 m" in readme and "28 m" not in readme


def _export_fixture(tmp_path) -> Path:
    """The fixture project of test_export_simple_files_and_columns, exported."""
    return export_simple(make_project(tmp_path / "proj"), tmp_path / "out")


def test_players_csv_has_source_and_player_id(tmp_path):
    out = _export_fixture(tmp_path)
    header = (out / "players.csv").read_text(encoding="utf-8").splitlines()[0].split(",")
    assert header[-3:] == ["source", "player_id", "id_confidence"]
    _, players = _csv(out / "players.csv")
    assert {p["source"] for p in players} <= {"seen", "bridged", "rejected", "hidden", "offscreen", "predicted"}
    assert sum(p["source"] == "bridged" for p in players) == 4  # old tracking.json: interpolated rows -> bridged
    readme = (out / "README_FOR_AI.md").read_text(encoding="utf-8")
    assert "Which rows to trust" in readme and "offscreen" in readme
