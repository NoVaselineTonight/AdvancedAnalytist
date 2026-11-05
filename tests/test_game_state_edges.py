"""Edge cases of replay/game_state.py (and the clock formatters elsewhere) - cloud task 6."""

from __future__ import annotations

import numpy as np
import pytest

from replay import data_export, game_state as gs, playlists, report
from tests.test_game_state import Replay, still


@pytest.mark.parametrize("fmt,t,expected", [
    (gs._clock, 59.96, "1:00.0"), (gs._clock, 59.94, "0:59.9"), (gs._clock, -2.0, "0:00.0"),
    (gs._clock, 119.99, "2:00.0"),
    (data_export.fmt_clock, 59.96, "01:00.0"), (data_export.fmt_clock, 0.04, "00:00.0"),
    (playlists.fmt_clock, 59.96, "1:00.0"), (playlists.fmt_clock, 3599.96, "1:00:00.0"),
    (report._fmt, 59.996, "01:00.00"), (report._fmt, 61.5, "01:01.50"),
])
def test_clocks_never_show_60_seconds(fmt, t, expected):
    assert fmt(t) == expected


def test_playlists_clock_whole_seconds():
    assert playlists.fmt_clock(59.6, 0) == "1:00" and playlists.fmt_clock(59.4, 0) == "0:59"
    assert playlists.fmt_clock(None) == "" and data_export.fmt_clock(float("nan")) == ""


def test_dedupe_ids_never_leaves_a_duplicate():
    ev = [{"id": "x"}, {"id": "x"}, {"id": "x-1"}, {"id": "x"}, {"id": "y"}]
    gs._dedupe_ids(ev)
    ids = [e["id"] for e in ev]
    assert len(set(ids)) == len(ids) and ids[0] == "x" and ids[2] == "x-1" and ids[4] == "y"
    ev = [{"id": "a"}, {"id": "a"}]
    gs._dedupe_ids(ev)
    assert [e["id"] for e in ev] == ["a", "a-1"]  # the usual case is unchanged


def test_runs_and_close_gaps():
    assert gs._runs(np.zeros(0, bool)) == [] and gs._runs(np.zeros(4, bool)) == []
    assert gs._runs(np.ones(3, bool)) == [(0, 3)]
    m = np.array([1, 1, 0, 0, 1, 0, 0, 0, 0, 1], bool)
    t = np.arange(10) / 10.0
    assert gs._close_gaps(m, t, 0.35).tolist() == [1, 1, 1, 1, 1, 0, 0, 0, 0, 1]
    assert gs._close_gaps(m, t, 0.25).tolist() == m.tolist()
    assert gs._close_gaps(m, t, 0.0).tolist() == m.tolist()


def test_direction_lookups_without_data():
    assert gs.attacked_side([], "A", 3.0) is None and gs.attacked_side([{"t0": 0, "t1": 9, "A": "L"}], "X", 3) is None
    d = [{"t0": 0.0, "t1": 10.0, "A": "L", "B": "R", "confidence": 0.4}]
    assert gs.attacker_of(d, "L", 5.0, 0.5) is None  # too unsure
    assert gs.attacker_of(d, "L", 5.0, 0.3) == "A" and gs.attacker_of(d, "R", 10.9, 0.3) == "B"
    assert gs.attacker_of(d, "L", 11.5, 0.3) is None and gs.attacker_of(d, None, 5.0, 0.3) is None


def test_halves_with_and_without_half_time():
    assert gs.halves(0.0, 60.0, [])[0]["index"] == 0 and len(gs.halves(0.0, 60.0, [])) == 1
    h = gs.halves(0.0, 60.0, [{"kind": "half_time", "t0": 20.0, "t1": 70.0}])
    assert [(x["t0"], x["t1"]) for x in h] == [(0.0, 20.0), (70.0, 70.0)]


@pytest.mark.parametrize("v6", [False, True])
def test_tiny_and_empty_replays_do_not_crash(v6):
    empty = Replay(0).doc()
    d = gs.analyse(empty, v6=v6)
    assert d["possessions"] == [] and d["events"] == []
    one = Replay(0.1)  # a single frame
    one.player("1", "A", 0, 1, still(5.0, 5.0))
    d = gs.analyse(one.doc(), v6=v6)
    assert isinstance(d["possessions"], list) and d["format"] == gs.FORMAT
    nobody = Replay(5)  # frames but no people and no ball
    d = gs.analyse(nobody.doc(), v6=v6)
    assert all(p["team"] is None for p in d["possessions"])


def test_v6_vote_without_any_evidence_is_unclear():
    r = Replay(4)
    r.player("1", "A", 0, 4, still(5.0, 5.0))
    gi = gs.load_input(r.doc())
    vote = gs.v6_vote(gi, 0, gi.n - 1, 0.0, 3.9, [None] * gi.n, np.zeros(gi.n), lambda s, t: (None, 0.0), [],
                      None, None, gs.PossessionV6Config())
    assert vote["team"] is None and vote["score"] == 0.0 and vote["confidence"] == 0.0
    assert "user" not in vote["evidence"]
