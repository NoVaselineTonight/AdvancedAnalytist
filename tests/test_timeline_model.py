import json

from replay.gui.timeline_model import AUTO, FORMAT, Timeline, load, save

EVENTS = [
    {"id": "basket-369-L", "type": "basket", "t": 12.3, "frame": 369, "basket": "L", "team": "A",
     "confidence": 0.8, "source": "auto", "details": {}},
    {"id": "basket-600-R", "type": "basket", "t": 20.0, "frame": 600, "basket": "R", "team": "B",
     "confidence": 0.55, "source": "auto", "details": {}},
    {"id": "basket-700-R", "type": "basket", "t": 23.5, "frame": 700, "basket": "R", "team": None,
     "confidence": 0.4, "source": "auto", "details": {}},
]
TEAMS = {"A": {"name": "Lions", "color": "#ff0000"}, "B": {"name": "Bears", "color": "#0000ff"}}


def user_track(tl, name="Notes"):
    return next(t for t in tl.tracks if t.name == name)


def test_defaults():
    tl = Timeline.default()
    assert [t.name for t in tl.tracks] == ["Auto detection", "Notes", "Highlights"]
    assert [t.kind for t in tl.tracks] == [AUTO, "user", "user"]
    assert len({t.color for t in tl.tracks}) == 3
    assert all(t.visible and not t.locked and not t.marks for t in tl.tracks)


def test_missing_file_gives_defaults(tmp_path):
    tl = load(tmp_path)
    assert len(tl.tracks) == 3 and tl.auto_track.name == "Auto detection"
    assert not (tmp_path / "timeline.json").exists()


def test_round_trip(tmp_path):
    tl = Timeline.default()
    notes = user_track(tl)
    m = tl.add_mark(notes.id, 5.5, title="Press", note="full court\npress", duration=2.0, color="#123456")
    tr = tl.add_track("Defense", "#abcdef")
    tl.add_mark(tr.id, 1.0, title="Zone")
    tl.update_track(tr.id, locked=True, visible=False)
    tl.set_events(EVENTS, TEAMS)
    tl.update_mark("auto:basket-369-L", note="nice pass")
    save(tl, tmp_path)
    raw = json.loads((tmp_path / "timeline.json").read_text(encoding="utf-8"))
    assert raw["format"] == FORMAT
    assert not (tmp_path / "timeline.json.tmp").exists()

    tl2 = load(tmp_path)
    assert [t.to_dict() for t in tl2.tracks if not t.is_auto] == [t.to_dict() for t in tl.tracks if not t.is_auto]
    _, m2 = tl2.find_mark(m.id)
    assert (m2.t, m2.duration, m2.title, m2.note, m2.color) == (5.5, 2.0, "Press", "full court\npress", "#123456")
    d = tl2.track(tr.id)
    assert d.locked and not d.visible and d.color == "#abcdef"
    # auto marks come back from events + overrides
    assert tl2.auto_track.marks == []
    tl2.set_events(EVENTS, TEAMS)
    _, a = tl2.find_mark("auto:basket-369-L")
    assert a.note == "nice pass"


def test_mark_add_move_delete():
    tl = Timeline.default()
    notes, hi = user_track(tl), user_track(tl, "Highlights")
    a = tl.add_mark(notes.id, 10)
    b = tl.add_mark(notes.id, 3)
    assert [m.id for m in notes.marks] == [b.id, a.id]
    assert tl.move_mark(a.id, 1.0)
    assert [m.id for m in notes.marks] == [a.id, b.id]
    assert tl.move_mark(a.id, -5)
    assert a.t == 0.0
    assert tl.move_mark(a.id, 4.0, hi.id)
    assert a in hi.marks and a not in notes.marks
    assert tl.update_mark(b.id, title="T", note="N", duration=1.5, t=7.0, track_id=hi.id)
    assert b in hi.marks and (b.t, b.duration, b.title, b.note) == (7.0, 1.5, "T", "N")
    assert tl.delete_mark(a.id)
    assert tl.find_mark(a.id) == (None, None)
    assert not tl.delete_mark("nope")
    # auto track takes no user marks
    assert tl.add_mark(tl.auto_track.id, 1.0) is None


def test_track_add_reorder_delete():
    tl = Timeline.default()
    tr = tl.add_track("Offense")
    assert tr.color not in {t.color for t in tl.tracks if t is not tr}
    assert tl.tracks[-1] is tr
    assert tl.move_track(tr.id, -1)
    assert tl.tracks[-2] is tr
    assert not tl.move_track(tl.tracks[0].id, -1)
    assert tl.update_track(tr.id, name="O", color="#00ff00")
    assert tr.name == "O" and tr.color == "#00ff00"
    assert tl.update_track(tr.id, color="garbage") and tr.color == "#00ff00"
    assert not tl.delete_track(tl.auto_track.id)
    assert tl.delete_track(tr.id) and tl.track(tr.id) is None


def test_auto_events_overrides_and_dismissals():
    tl = Timeline.default()
    assert tl.set_events(EVENTS, TEAMS)
    marks = tl.auto_track.marks
    assert [m.title for m in marks] == ["Basket – Lions (L)", "Basket – Bears (R)", "Basket (R)"]
    assert "80%" in marks[0].note and marks[0].confidence == 0.8
    assert [m.color for m in marks] == ["#ff0000", "#0000ff", None]
    assert not tl.set_events(EVENTS, TEAMS)  # unchanged

    tl.update_mark("auto:basket-600-R", title="And one", note="foul too")
    assert not tl.move_mark("auto:basket-600-R", 1.0)
    assert tl.delete_mark("auto:basket-700-R")
    assert tl.dismissed_count() == 1
    # rebuild (replay_data.json changed): overrides survive, dismissed stays gone
    ev2 = [dict(e, t=e["t"] + 0.1) for e in EVENTS] + [
        {"id": "basket-900-L", "type": "basket", "t": 29.0, "basket": "L", "team": "A", "confidence": 0.9}]
    assert tl.set_events(ev2, TEAMS)
    by_id = {m.event_id: m for m in tl.auto_track.marks}
    assert set(by_id) == {"basket-369-L", "basket-600-R", "basket-900-L"}
    assert by_id["basket-600-R"].title == "And one" and by_id["basket-600-R"].note == "foul too"
    assert abs(by_id["basket-600-R"].t - 20.1) < 1e-9
    assert tl.reset_auto("auto:basket-600-R")
    assert tl.find_mark("auto:basket-600-R")[1].title == "Basket – Bears (R)"
    assert "basket-600-R" not in tl.auto_overrides
    assert tl.restore_dismissed() == 1
    assert len(tl.auto_track.marks) == 4
    # no events (older replay data): empty auto track
    tl.set_events(None)
    assert tl.auto_track.marks == []


def test_corrupt_and_odd_files(tmp_path):
    f = tmp_path / "timeline.json"
    f.write_text("{not json", encoding="utf-8")
    tl = load(tmp_path)
    assert [t.name for t in tl.tracks] == ["Auto detection", "Notes", "Highlights"]
    assert (tmp_path / "timeline.json.bad").exists()  # kept aside, not silently overwritten

    f.write_text(json.dumps({"tracks": "x"}), encoding="utf-8")
    assert len(load(tmp_path).tracks) == 3

    # partially broken content is repaired, auto track recreated
    f.write_text(json.dumps({"tracks": [
        {"name": "Mine", "color": "nope", "marks": [{"t": "abc"}, 5, {"t": 3, "title": None}]},
        "junk",
        {"id": "x", "kind": "auto"}, {"id": "y", "kind": "auto"},
    ], "auto_overrides": {"e1": {"dismissed": True, "bogus": 1}, "e2": 7}}), encoding="utf-8")
    tl = load(tmp_path)
    assert [t.kind for t in tl.tracks] == ["user", "auto"]
    mine = tl.tracks[0]
    assert mine.color.startswith("#") and [m.t for m in mine.marks] == [0.0, 3.0]
    assert tl.auto_overrides == {"e1": {"dismissed": True}}

    f.write_text(json.dumps({"tracks": []}), encoding="utf-8")
    assert [t.kind for t in load(tmp_path).tracks] == ["auto"]
