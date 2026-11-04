"""Corrections ledger (replay/corrections.py): schema + canonical file, identity constraints in the tracklet linker
(split / merge / assign), possession constraints in game state v6 (make / anchor), no effect without ops."""

from __future__ import annotations

import json

import numpy as np
import pytest

from replay import corrections as C
from replay import game_state as gs
from replay.court_tracker import _undo_ledger_conflicts
from test_tracklets import FPS, ORANGE, _run, _scene
from tests.test_game_state import Replay, still
from tests.test_possession_model import base, poss_at


def _ids_by_pid(res, s) -> dict[int, list[int]]:
    return {int(p): sorted(set(res.ids[s["pid"] == p].tolist())) for p in np.unique(s["pid"])}


def _ledger(s, ops):
    return C.resolve_identity(C.normalize({"ops": ops}), C.sample_times(s["n"], FPS), s["k"], s["xy"])


# ----------------------------------------------------------------------------------------------- schema / file
def test_schema_validates_every_kind():
    ops = [{"kind": "split", "t0": 1, "at": [3, 4]},
           {"kind": "merge", "t0": 1, "t1": 9, "at": [3, 4], "to": [5, 6]},
           {"kind": "assign", "t0": 1, "t1": 2, "at": [3, 4], "player": "A7"},
           {"kind": "team", "t0": 1, "at": [3, 4], "team": "B"},
           {"kind": "role", "t0": 1, "at": [3, 4], "role": "referee"},
           {"kind": "hide", "t0": 1, "t1": 5, "at": [3, 4]},
           {"kind": "anchor", "t0": 1, "t1": 5, "team": "A"},
           {"kind": "make", "t0": 7, "team": "A", "points": 3, "side": "L"}]
    doc = C.normalize({"ops": ops})
    assert [o["kind"] for o in doc["ops"]] == [o["kind"] for o in ops]  # edit order kept
    assert doc["ops"][0]["t1"] == 1.0 and doc["format"] == C.FORMAT and doc["version"] == C.VERSION
    bad = [{"kind": "nope", "t0": 1}, {"kind": "split", "t0": 1}, {"kind": "split", "t0": "x", "at": [1, 2]},
           {"kind": "merge", "t0": 1, "at": [1, 2]}, {"kind": "assign", "t0": 1, "at": [1, 2], "player": ""},
           {"kind": "make", "t0": 1, "team": "C"}, {"kind": "make", "t0": 1, "team": "A", "points": 4},
           {"kind": "anchor", "t0": 5, "t1": 1, "team": "A"}, {"kind": "role", "t0": 1, "at": [1, 2], "role": "x"},
           {"kind": "split", "t0": float("nan"), "at": [1, 2]}]
    for op in bad:
        with pytest.raises(C.CorrectionsError):
            C.normalize({"ops": [op]})
    with pytest.raises(C.CorrectionsError):
        C.normalize({"version": C.VERSION + 1, "ops": []})


def test_file_round_trip_is_byte_for_byte(tmp_path):
    doc = {"ops": [{"kind": "make", "t0": 7.25, "team": "B", "points": 2, "note": "Zyxa layup ä"},
                   {"kind": "split", "t0": 3, "at": [1.5, 2], "future_key": {"x": 1}}], "editor": "Testaaja"}
    p = C.save(tmp_path, doc)
    text = p.read_bytes()
    again = C.load(tmp_path)
    assert again == C.normalize(doc) and again["ops"][1]["future_key"] == {"x": 1} and again["editor"] == "Testaaja"
    C.save(tmp_path, again)
    assert p.read_bytes() == text
    assert C.dumps(json.loads(text)) == text.decode("utf-8")
    assert C.load(tmp_path / "missing") == {"format": C.FORMAT, "version": C.VERSION, "ops": []}
    (tmp_path / C.FILENAME).write_text("{broken", encoding="utf-8")
    with pytest.raises(C.CorrectionsError):
        C.load(tmp_path)


def test_enabled_flag_and_env(tmp_path, monkeypatch):
    class Cfg:
        corrections = False
    monkeypatch.delenv("REPLAY_CORRECTIONS", raising=False)
    C.save(tmp_path, {"ops": [{"kind": "make", "t0": 1, "team": "A"}]})
    assert not C.enabled(Cfg()) and C.load_if_enabled(tmp_path, Cfg()) is None
    Cfg.corrections = True
    assert C.load_if_enabled(tmp_path, Cfg())["ops"][0]["kind"] == "make"
    monkeypatch.setenv("REPLAY_CORRECTIONS", "0")
    assert C.load_if_enabled(tmp_path, Cfg()) is None
    monkeypatch.setenv("REPLAY_CORRECTIONS", "1")
    assert C.load_if_enabled(tmp_path, None) is not None
    (tmp_path / C.FILENAME).write_text("{broken", encoding="utf-8")
    assert C.load_if_enabled(tmp_path, None) is None  # reported, the analysis carries on
    from replay.config import Config
    assert Config().corrections is False


def test_label_rules():
    a, b = frozenset({"p:4"}), frozenset({"p:7"})
    assert C.conflict(a, b) and not C.forced(a, b)
    assert C.forced(a, frozenset({"p:4", "m:0"})) and not C.conflict(a, frozenset({"p:4"}))
    assert C.conflict(frozenset({"sL:2"}), frozenset({"sR:2"})) and not C.conflict(frozenset({"sL:2"}), frozenset({"sR:3"}))
    assert C.forced(frozenset({"m:1"}), frozenset({"m:1"})) and not C.forced(frozenset(), frozenset({"m:1"}))
    assert C.conflict_any([frozenset({"sL:0"}), frozenset(), frozenset({"sR:0"})])
    assert not C.conflict_any([frozenset({"p:1"}), frozenset({"p:1"})])


# ----------------------------------------------------------------------------------------------- identity
def test_no_ops_is_no_effect():
    n = 60
    a = np.c_[np.linspace(6, 14, n), np.full(n, 6.0)]
    s = _scene({1: a}, {1: ORANGE}, bt_of=lambda p, k: 1)
    assert _ledger(s, []) is None
    assert C.resolve_identity(None, C.sample_times(s["n"], FPS), s["k"], s["xy"]) is None
    only_events = _ledger(s, [{"kind": "make", "t0": 1, "team": "A"}])
    assert only_events is None
    base_ = _run(s)
    same = _run(s, ledger=None)
    assert np.array_equal(base_.ids, same.ids) and base_.stats == same.stats and "corrections" not in same.stats
    pieces, labels = C.apply_identity(None, [np.arange(3)], s["k"], s["xy"])
    assert labels == {} and len(pieces) == 1 and pieces[0].tolist() == [0, 1, 2]


def test_split_cuts_a_tracklet_for_good():
    n = 60
    a = np.c_[np.linspace(6, 14, n), np.full(n, 6.0)]
    s = _scene({1: a}, {1: ORANGE}, bt_of=lambda p, k: 1)
    assert len(np.unique(_run(s).ids)) == 1
    res = _run(s, ledger=_ledger(s, [{"kind": "split", "t0": 3.0, "at": a[30].tolist()}]))
    before, after = set(res.ids[s["k"] < 30]), set(res.ids[s["k"] >= 30])
    assert len(before) == 1 and len(after) == 1 and before != after
    assert res.stats["corrections"]["unresolved_ops"] == 0


def test_split_where_the_tracklet_already_starts():
    """ByteTrack already gave the person a new ID at t0 (the linker would re-join them): the 'before' side is the
    nearest person just before t0."""
    n = 60
    a = np.c_[np.linspace(6, 14, n), np.full(n, 6.0)]
    s = _scene({1: a}, {1: ORANGE}, bt_of=lambda p, k: 1 if k < 30 else 2)
    assert len(np.unique(_run(s).ids)) == 1
    ic = _ledger(s, [{"kind": "split", "t0": 3.0, "at": a[30].tolist()}])
    res = _run(s, ledger=ic)
    assert set(res.ids[s["k"] < 30]) != set(res.ids[s["k"] >= 30])
    assert "before" in ic.picks[0]


def test_merge_is_a_forced_link_over_a_long_gap():
    n = 300
    p = np.c_[np.linspace(4, 24, n), np.full(n, 7.0)]
    s = _scene({1: p}, {1: ORANGE}, bt_of=lambda q, k: 1 if k < 50 else 2, hidden={1: range(50, 250)})
    assert len(np.unique(_run(s).ids)) == 2
    res = _run(s, ledger=_ledger(s, [{"kind": "merge", "t0": 4.9, "at": p[49].tolist(), "t1": 25.0,
                                      "to": p[250].tolist()}]))
    assert len(np.unique(res.ids)) == 1
    info = next(iter(res.identities.values()))
    assert info["strong_links"] >= 1  # a coach-confirmed link counts as strong evidence


def test_merge_beyond_every_linking_round_is_enforced_at_the_end():
    n = 1400  # 130 s hidden: longer than the longest linking round (120 s)
    p = np.c_[np.linspace(4, 24, n), np.full(n, 7.0)]
    s = _scene({1: p}, {1: ORANGE}, bt_of=lambda q, k: 1 if k < 50 else 2, hidden={1: range(50, 1350)})
    assert len(np.unique(_run(s).ids)) == 2
    res = _run(s, ledger=_ledger(s, [{"kind": "merge", "t0": 4.9, "at": p[49].tolist(), "t1": 135.0,
                                      "to": p[1350].tolist()}]))
    assert len(np.unique(res.ids)) == 1
    assert res.stats["corrections"]["forced_joins"] == 1


def test_assign_different_players_are_never_linked_and_same_player_is():
    n = 60
    p1 = np.c_[np.linspace(6, 9, n), np.full(n, 6.0)]
    p2 = np.c_[np.linspace(9.2, 12, n), np.full(n, 6.1)]
    s = _scene({1: p1, 2: p2}, {1: ORANGE, 2: ORANGE}, bt_of=lambda q, k: q,
               hidden={1: range(30, 60), 2: range(0, 32)})
    assert _ids_by_pid(_run(s), s) == {1: [1], 2: [1]}  # same kit, plausible motion: linked as one
    ops = [{"kind": "assign", "t0": 1.0, "t1": 2.9, "at": p1[10].tolist(), "player": "4"},
           {"kind": "assign", "t0": 3.5, "t1": 5.9, "at": p2[40].tolist(), "player": "7"}]
    by = _ids_by_pid(_run(s, ledger=_ledger(s, ops)), s)
    assert len(by[1]) == 1 and len(by[2]) == 1 and by[1] != by[2]
    ops[1]["player"] = "4"
    by = _ids_by_pid(_run(s, ledger=_ledger(s, ops)), s)
    assert by[1] == by[2] and len(by[1]) == 1


def test_unresolved_op_is_counted_not_applied():
    n = 60
    a = np.c_[np.linspace(6, 14, n), np.full(n, 6.0)]
    s = _scene({1: a}, {1: ORANGE}, bt_of=lambda p, k: 1)
    ic = _ledger(s, [{"kind": "split", "t0": 3.0, "at": [25.0, 1.0]},  # nobody there
                     {"kind": "split", "t0": 3.0, "at": a[30].tolist()}])
    assert ic.unresolved == [0]
    res = _run(s, ledger=ic)
    assert len(np.unique(res.ids)) == 2 and res.stats["corrections"]["unresolved_ops"] == 1


def test_corrections_survive_a_rebuild(tmp_path):
    """Rebuilding with the same ledger gives the same identities, and the ledger file is never rewritten."""
    n = 60
    a = np.c_[np.linspace(6, 14, n), np.full(n, 6.0)]
    s = _scene({1: a}, {1: ORANGE}, bt_of=lambda p, k: 1)
    C.save(tmp_path, {"ops": [{"kind": "split", "t0": 3.0, "at": a[30].tolist()}]})
    raw = (tmp_path / C.FILENAME).read_bytes()
    mtime = (tmp_path / C.FILENAME).stat().st_mtime_ns
    runs = []
    for _ in range(2):
        doc = C.load(tmp_path)
        runs.append(_run(s, ledger=C.resolve_identity(doc, C.sample_times(s["n"], FPS), s["k"], s["xy"])).ids)
        C.save(tmp_path, doc)  # e.g. the GUI saving the unchanged ledger
    assert np.array_equal(runs[0], runs[1]) and len(np.unique(runs[0])) == 2
    assert (tmp_path / C.FILENAME).read_bytes() == raw and (tmp_path / C.FILENAME).stat().st_mtime_ns == mtime


def test_reattach_join_that_breaks_the_ledger_is_undone():
    before = np.array([1, 1, 2, 2, 3, 3])
    after = np.array([1, 1, 1, 1, 3, 3])  # re-attach joined 1 and 2
    labels = {0: frozenset({"sL:0"}), 3: frozenset({"sR:0"})}
    out, undone = _undo_ledger_conflicts(before, after, labels)
    assert out.tolist() == before.tolist() and undone == 1
    out, undone = _undo_ledger_conflicts(before, after, {0: frozenset({"p:4"}), 3: frozenset({"p:4"})})
    assert out.tolist() == after.tolist() and undone == 0


# ----------------------------------------------------------------------------------------------- events
def test_apply_events():
    assert C.apply_events(None) is None
    assert C.apply_events(C.normalize({"ops": [{"kind": "split", "t0": 1, "at": [1, 1]}]})) is None
    ev = C.apply_events(C.normalize({"ops": [{"kind": "anchor", "t0": 5, "t1": 9, "team": "B"},
                                             {"kind": "make", "t0": 3, "team": "A", "points": 3}]}))
    assert ev == {"makes": [{"t": 3.0, "team": "A", "points": 3, "side": None}],
                  "anchors": [{"t0": 5.0, "t1": 9.0, "team": "B"}], "boundaries": [5.0, 9.0]}


def test_game_state_without_ledger_is_unchanged():
    r = base(12)
    a = gs.analyse(r.doc(), v6=True)
    b = gs.analyse(r.doc(), v6=True, corrections=None)
    c = gs.analyse(r.doc(), v6=True, corrections=C.normalize(None))
    assert a == b == c


def test_anchor_decides_an_unclear_possession():
    r = Replay(10)
    for k in range(4):
        r.player(str(k + 1), "A" if k < 2 else "B", 0, 10, still(8.0 + 3 * k, 6.0 + k))
    assert [p["team"] for p in gs.analyse(r.doc(), v6=True)["possessions"]] == [None]
    led = C.normalize({"ops": [{"kind": "anchor", "t0": 0.0, "t1": 9.9, "team": "B"}]})
    d = gs.analyse(r.doc(), v6=True, corrections=led)
    assert [p["team"] for p in d["possessions"]] == ["B"]
    assert d["possessions"][0]["evidence"]["user"] < 0 and d["possession_model"]["user_anchors"] == 1


def test_user_make_overrides_the_detected_team():
    r = base(16)
    d0 = gs.analyse(r.doc(), v6=True)
    assert poss_at(d0, 5.6, "A")["outcome"] == "made"
    led = C.normalize({"ops": [{"kind": "make", "t0": 5.8, "team": "B", "points": 3}]})
    d = gs.analyse(r.doc(), v6=True, corrections=led)
    p = poss_at(d, 5.6, "B")
    assert p is not None and p["outcome"] == "made" and p["points"] == 3 and p["make_source"] == "user"
    assert d["possession_model"]["user_makes"] == 1


def test_user_make_without_detection_is_a_boundary():
    r = base(30)
    r.hold("3", 8.0, 14.0)  # B holds, nobody sees the make at ~14.5 s, then A inbounds
    r.hold("2", 16.0, 28.0)
    d0 = gs.analyse(r.doc(), v6=True)
    led = C.normalize({"ops": [{"kind": "make", "t0": 14.5, "team": "B", "points": 2, "side": "R"}]})
    d = gs.analyse(r.doc(), v6=True, corrections=led)
    p = poss_at(d, 12.0, "B")
    assert p is not None and p["outcome"] == "made" and abs(p["t1"] - 14.5) < 0.05
    nxt = poss_at(d, 20.0)
    assert nxt["team"] == "A" and nxt["start"] == "after made basket"
    assert sum(q["points"] for q in d["possessions"] if q["team"] == "B") == \
        sum(q["points"] for q in d0["possessions"] if q["team"] == "B") + 2


def test_build_for_results_reads_the_ledger_only_when_enabled(tmp_path, monkeypatch):
    res = tmp_path / "results"
    res.mkdir()
    r = Replay(10)
    for k in range(4):
        r.player(str(k + 1), "A" if k < 2 else "B", 0, 10, still(8.0 + 3 * k, 6.0 + k))
    (res / "replay_data.json").write_text(json.dumps(r.doc()), encoding="utf-8")
    C.save(tmp_path, {"ops": [{"kind": "anchor", "t0": 0.0, "t1": 9.9, "team": "A"}]})
    raw = (tmp_path / C.FILENAME).read_bytes()
    monkeypatch.setenv("REPLAY_POSSESSION_V6", "1")
    monkeypatch.delenv("REPLAY_CORRECTIONS", raising=False)
    assert [p["team"] for p in gs.build_for_results(res)["possessions"]] == [None]
    monkeypatch.setenv("REPLAY_CORRECTIONS", "1")
    assert [p["team"] for p in gs.build_for_results(res)["possessions"]] == ["A"]
    assert (tmp_path / C.FILENAME).read_bytes() == raw
