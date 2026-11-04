import numpy as np
from types import SimpleNamespace as R
from replay import court
from replay.config import FilterConfig
from replay.limits import apply_segment_limits

def row(tid, team, k, x, y, interp=False, state=1, box=None):
    return R(k=k, track_id=tid, team=team, x=x, y=y, interpolated=interp, pos_state=state, pos_sigma=0.2,
             object_type="player", box=box)

def _seq(tid, team, ks, x, y, **kw):
    return [row(tid, team, k, x, y, **kw) for k in ks]

def test_hidden_player_next_to_teammate_is_kept():
    court.configure(28.0, 15.0, "FIBA")
    rows = _seq(1, "A", range(0, 30), 10.0, 5.0)                                  # seen teammate
    rows += _seq(2, "A", range(0, 10), 10.8, 5.0) + _seq(2, "A", range(10, 20), 10.8, 5.0, interp=True, state=4) \
          + _seq(2, "A", range(20, 30), 10.8, 5.0)                                  # hidden behind him for 1 s
    drop = apply_segment_limits(rows, {id(r): 1.0 for r in rows}, {1: 10, 2: 10}, 10.0, FilterConfig(segment_rules=True))
    assert not drop

def test_cap_evicts_whole_run_not_single_samples():
    court.configure(28.0, 15.0, "FIBA")
    rows = []
    for j in range(10):                                    # 10 players for 5 s
        rows += _seq(j, "A" if j < 5 else "B", range(0, 50), 2 + 2.5 * j, 5.0)
    rows += _seq(99, "B", range(20, 25), 20.0, 9.0)        # an 11th person for 0.5 s with the LOWEST rank
    rank = {id(r): (0.1 if r.track_id == 99 else 1.0) for r in rows}
    drop = apply_segment_limits(rows, rank, {t: 1 for t in range(100)}, 10.0, FilterConfig(segment_rules=True))
    gone = {r.track_id for r in rows if id(r) in drop}
    assert gone == {99}
    assert all(id(r) in drop for r in rows if r.track_id == 99)   # whole run, not alternate samples

def test_short_eviction_is_hysteresised_to_next_candidate():
    court.configure(28.0, 15.0, "FIBA")
    rows = []
    for j in range(10):
        rows += _seq(j, "A" if j < 5 else "B", range(0, 50), 2 + 2.5 * j, 5.0)
    rows += _seq(99, "B", range(0, 50), 20.0, 9.0)         # 11th person for the whole 5 s, rank slightly BELOW player 9
    rank = {id(r): (0.9 if r.track_id == 99 else 1.0) for r in rows}
    for r in rows:                                         # player 9 dips in rank for 3 samples only
        if r.track_id == 9 and 20 <= r.k < 23:
            rank[id(r)] = 0.5
    drop = apply_segment_limits(rows, rank, {t: 1 for t in range(100)}, 10.0, FilterConfig(segment_rules=True))
    gone = {r.track_id for r in rows if id(r) in drop}
    assert gone == {99}                                    # not 9 for samples 20-22 then 99 again

def test_true_ghost_of_split_identity_is_dropped():
    court.configure(28.0, 15.0, "FIBA")
    rows = _seq(1, "A", range(0, 10), 10.0, 5.0) + _seq(1, "A", range(10, 30), 10.0, 5.0, interp=True, state=2)
    rows += _seq(2, "A", range(12, 40), 10.3, 5.1)         # same person re-found under a new id 0.2 s later
    drop = apply_segment_limits(rows, {id(r): 1.0 for r in rows}, {1: 5, 2: 10}, 10.0, FilterConfig(segment_rules=True))
    assert {r.k for r in rows if id(r) in drop and r.track_id == 1} == set(range(12, 30))
    assert not any(id(r) in drop for r in rows if r.track_id == 2)

def test_calibration_hole_does_not_split_identity():
    # rows simply absent at k=15..17 (no H): the run logic must treat 14->18 as consecutive for hysteresis
    court.configure(28.0, 15.0, "FIBA")
    ks = [k for k in range(0, 30) if k not in (15, 16, 17)]
    rows = _seq(1, "A", ks, 10.0, 5.0)
    drop = apply_segment_limits(rows, {id(r): 1.0 for r in rows}, {1: 10}, 10.0, FilterConfig(segment_rules=True), present=set(ks))
    assert not drop


def test_default_mode_keeps_hidden_player_next_to_teammate(monkeypatch):
    from replay.limits import final_limits
    monkeypatch.setenv("REPLAY_KEEP_HIDDEN", "1")
    court.configure(28.0, 15.0, "FIBA")
    rows = _seq(1, "A", range(0, 30), 10.0, 5.0)
    rows += _seq(2, "A", range(0, 10), 10.8, 5.0) + _seq(2, "A", range(10, 20), 10.8, 5.0, interp=True, state=4) \
        + _seq(2, "A", range(20, 30), 10.8, 5.0)
    drop = final_limits(rows, {id(r): 1.0 for r in rows}, {1: 10, 2: 10}, 10.0, FilterConfig())
    assert not drop
    # ... but a filled stretch next to a teammate whose id STARTS during it (the tracker handed the person over)
    rows2 = _seq(1, "A", range(0, 10), 10.0, 5.0) + _seq(1, "A", range(10, 20), 10.0, 5.0, interp=True, state=2) \
        + _seq(1, "A", range(20, 30), 10.0, 5.0) + _seq(3, "A", range(12, 19), 10.3, 5.1)
    drop2 = final_limits(rows2, {id(r): 1.0 for r in rows2}, {1: 10, 3: 5}, 10.0, FilterConfig())
    assert any(id(r) in drop2 for r in rows2 if r.track_id == 1 and r.interpolated)


def test_two_ids_on_one_person_are_merged():
    from replay.limits import merge_double_ids
    rows = _seq(1, "A", range(0, 30), 10.0, 5.0) + _seq(2, "A", range(20, 50), 10.3, 5.1)   # 1 s overlap, 0.3 m apart
    rows += _seq(3, "A", range(20, 50), 14.0, 5.0)                                            # a teammate 4 m away
    drop = merge_double_ids(rows, {id(r): 1.0 for r in rows})
    assert {r.track_id for r in rows} == {1, 3}
    assert len([r for r in rows if r.track_id == 1 and id(r) not in drop]) == 50              # one row per sample


def test_sticky_cap_uncertain_fill_never_holds_a_place(monkeypatch):
    # step E (identity_design.md): with sticky caps on, a filled-in row with pos_sigma > FILL_MAX_SIGMA_M gives way
    # to a seen player when the team is over its cap, and its id is not remembered as evicted
    from replay.limits import smoothed_frame_limits
    court.configure(28.0, 15.0, "FIBA")
    monkeypatch.setenv("REPLAY_STICKY_CAP", "1")
    monkeypatch.delenv("REPLAY_STICKY_FILLS", raising=False)
    rows = []
    for j in range(5):                                      # 5 seen A players for 5 s
        rows += _seq(j, "A", range(0, 50), 2 + 2.5 * j, 5.0)
    hidden = _seq(7, "A", range(0, 10), 20.0, 10.0) + _seq(7, "A", range(10, 40), 20.0, 10.0, interp=True, state=4) \
        + _seq(7, "A", range(40, 50), 20.0, 10.0)           # a 6th A: seen, then a 3 s uncertain fill, seen again
    for r in hidden:
        if r.interpolated:
            r.pos_sigma = 2.0
    rows += hidden
    rank = {id(r): 1.0 for r in rows if not r.interpolated}
    for r in rows:
        if r.track_id == 7 and not r.interpolated:
            rank[id(r)] = 3.0                               # the most player-like one when seen
    drop = smoothed_frame_limits(rows, rank, {t: 1 for t in range(10)}, FilterConfig(), fps=10.0)
    gone7 = {r.k for r in rows if id(r) in drop and r.track_id == 7}
    assert gone7 == set(range(10, 40))                      # the uncertain fill goes, the sightings stay
    assert len({r.track_id for r in rows if id(r) in drop}) >= 1


def test_bench_strip_hides_ids_that_never_entered_the_court(monkeypatch):
    from replay.limits import bench_strip_rows, final_limits
    court.configure(28.0, 15.0, "FIBA")
    rows = _seq(1, "A", range(0, 30), 10.0, 5.0, box=(0, 0, 10, 20))                # player
    rows += _seq(2, "B", range(0, 6), 21.0, 15.3, box=(0, 0, 10, 20))               # bench kid beyond the far line
    rows += _seq(3, "B", range(0, 4), 22.0, 14.7, box=(0, 0, 10, 20))               # bench kid 0.3 m inside the line
    rows += _seq(4, "A", range(0, 4), 2.0, 14.8, box=(0, 0, 10, 20)) + _seq(4, "A", range(4, 14), 6.0, 10.0,
                                                                               box=(0, 0, 10, 20))  # inbounder, then play
    rows += _seq(5, "B", range(0, 10), 14.0, 14.9, box=(0, 0, 10, 20)) + _seq(5, "B", range(10, 20), 14.0, 8.0,
                                                                                box=(0, 0, 10, 20))  # entered the court
    rows += _seq(6, "referee", range(0, 10), 14.0, 15.2, box=(0, 0, 10, 20))        # referee on the sideline
    rows += _seq(7, "A", range(0, 5), 20.0, 15.2, interp=True, state=4)             # filled only: no decision
    why = {}
    gone = {r.track_id for r in rows if id(r) in bench_strip_rows(rows, why)}
    assert gone == {2, 3}
    assert all(v == "bench strip" for v in why.values())
    rank = {id(r): 1.0 for r in rows}
    monkeypatch.delenv("REPLAY_BENCH_STRIP", raising=False)
    why = {}
    final_limits(rows, rank, {}, 10.0, FilterConfig(), reasons=why)
    assert "bench strip" not in why.values()                                        # flag off: rule inactive
    monkeypatch.setenv("REPLAY_BENCH_STRIP", "1")
    why = {}
    drop = final_limits(rows, rank, {}, 10.0, FilterConfig(), reasons=why)
    assert {2, 3} <= {r.track_id for r in rows if id(r) in drop}
    assert {r.track_id for r in rows if why.get(id(r)) == "bench strip"} == {2, 3}


def test_bench_strip_share_rule_and_strip_fills(monkeypatch):
    from replay.limits import bench_strip_rows, _weak_fills
    court.configure(28.0, 15.0, "FIBA")
    box = (0, 0, 10, 20)
    walker = _seq(1, "B", range(0, 36), 17.0, 14.8, box=box) + _seq(1, "B", range(36, 40), 17.0, 13.0, box=box)  # 90 % in the strip, 4 rows 2 m inside
    deep = _seq(2, "B", range(0, 36), 17.0, 14.8, box=box) + _seq(2, "B", range(36, 40), 17.0, 8.0, box=box)      # 4 deep rows
    monkeypatch.delenv("REPLAY_BENCH_STRIP_SHARE", raising=False)
    monkeypatch.delenv("REPLAY_BENCH_STRIP_MAX_DEEP", raising=False)
    assert not bench_strip_rows(walker + deep)                                      # default: all rows needed
    monkeypatch.setenv("REPLAY_BENCH_STRIP_SHARE", "0.9")
    monkeypatch.setenv("REPLAY_BENCH_STRIP_MAX_DEEP", "2")
    gone = {r.track_id for r in walker + deep if id(r) in bench_strip_rows(walker + deep)}
    assert gone == {1}
    rows = _seq(3, "A", range(0, 5), 20.0, 14.9, box=box) + _seq(3, "A", range(5, 9), 20.0, 14.9, interp=True, state=4)
    rows += _seq(4, "A", range(0, 5), 10.0, 7.0, box=box) + _seq(4, "A", range(5, 9), 10.0, 7.0, interp=True, state=4)
    pos = {k: k for k in range(9)}
    assert not _weak_fills(rows, pos, 10.0)
    weak = {r.track_id for r in rows if id(r) in _weak_fills(rows, pos, 10.0, strip_in_m=0.5)}
    assert weak == {3}


# ---------------------------------------------------------------------------------------- edge cases (cloud task 6)
import pytest  # noqa: E402

from replay import limits as L  # noqa: E402


def _both(rows, rank, strength, cfg_kw=None, **kw):
    """(segment rules, smoothed per-frame) drops as {(track, k)}."""
    out = []
    for rules in (True, False):
        cfg = FilterConfig(segment_rules=rules, **(cfg_kw or {}))
        d = (apply_segment_limits(rows, rank, strength, 10.0, cfg, **kw) if rules
             else L.smoothed_frame_limits(rows, rank, strength, cfg, fps=10.0, **kw))
        out.append({(r.track_id, r.k) for r in rows if id(r) in d})
    return out


def test_empty_and_ball_only_inputs_drop_nothing():
    court.configure(28.0, 15.0, "FIBA")
    assert _both([], {}, {}) == [set(), set()]
    ball = _seq("ball", "", range(5), 5.0, 5.0)
    for r in ball:
        r.object_type = "ball"
    assert _both(ball, {id(r): 1.0 for r in ball}, {}) == [set(), set()]


def test_fourth_referee_is_capped_in_both_modes():
    court.configure(28.0, 15.0, "FIBA")
    rows = []
    for j in range(4):
        rows += _seq(50 + j, "referee", range(30), 3.0 + 6 * j, 1.0)
    for r in rows:
        r.object_type = "referee"
    rank = {id(r): (0.1 if r.track_id == 53 else 1.0) for r in rows}
    for d in _both(rows, rank, {t: 1 for t in range(100)}):
        assert {t for t, _ in d} == {53} and len(d) == 30


def test_far_off_court_track_is_dropped():
    court.configure(28.0, 15.0, "FIBA")
    rows = _seq(1, "A", range(30), 10.0, 5.0) + _seq(2, "A", range(30), 10.0, 25.0)  # 10 m beyond the sideline
    for d in _both(rows, {id(r): 1.0 for r in rows}, {1: 1, 2: 1}):
        assert {t for t, _ in d} == {2}


def test_drops_do_not_depend_on_row_order():
    court.configure(28.0, 15.0, "FIBA")
    rows = []
    for j in range(12):
        rows += _seq(j, "A" if j < 6 else "B", range(30), 2.0 + 2 * j, 7.0)
    rank = {id(r): 1.0 - 0.01 * r.track_id for r in rows}
    st = {t: 1 for t in range(100)}
    assert _both(rows, rank, st) == _both(list(reversed(rows)), rank, st)


def test_team_caps_before_the_combined_cap(monkeypatch):
    """6 A + 6 B on court (one extra per team). Default smoothed limits: the combined cap of 10 keeps the 10 best of
    A+B first (drops B's 10 and 11), then 5 per team drops A's 6th -> only 4 B drawn. team_cap_first: 5 + 5."""
    monkeypatch.delenv("REPLAY_TEAM_CAP_FIRST", raising=False)
    court.configure(28.0, 15.0, "FIBA")
    rows = []
    for j in range(12):
        rows += _seq(j, "A" if j < 6 else "B", range(30), 2.0 + 2 * j, 7.0)
    rank = {id(r): 1.0 - 0.01 * r.track_id for r in rows}
    st = {t: 1 for t in range(100)}
    seg, old = _both(rows, rank, st)
    assert {t for t, _ in seg} == {5, 11}
    assert {t for t, _ in old} == {5, 10, 11}  # documents the default (unchanged) behaviour
    _, fixed = _both(rows, rank, st, cfg_kw={"team_cap_first": True})
    assert {t for t, _ in fixed} == {5, 11}
    monkeypatch.setenv("REPLAY_TEAM_CAP_FIRST", "1")
    assert {t for t, _ in _both(rows, rank, st)[1]} == {5, 11}


def test_limit_shown_rows_disjoint_caps_unchanged():
    """Earlier caps' drops are excluded from later caps; with disjoint label sets (the default) nothing changes."""
    from replay.roster import limit_shown_rows

    rows = []
    for j in range(7):
        rows += _seq(j, "A", range(3), 2.0 + 2 * j, 7.0)
    rank = {id(r): 1.0 - 0.01 * r.track_id for r in rows}
    d = limit_shown_rows(rows, ((("A",), 5),), rank, {})
    assert {r.track_id for r in rows if id(r) in d} == {5, 6}
    d = limit_shown_rows(rows, ((("A",), 6), (("A", "B"), 5)), rank, {})
    assert {r.track_id for r in rows if id(r) in d} == {5, 6}  # the 2nd cap counts only what the 1st kept


@pytest.mark.parametrize("flags,value,max_len,expected", [
    ([True, False, False, True], False, 2, [True, True, True, True]),  # short hole filled
    ([True, False, False, False, True], False, 2, [True, False, False, False, True]),  # too long
    ([False, False, True], False, 2, [False, False, True]),  # at the start: no neighbour before
    ([True, False, False], False, 2, [True, False, False]),  # at the end: no neighbour after
    ([False, True, False], True, 1, [False, False, False]),  # island removed
])
def test_fill_runs(flags, value, max_len, expected):
    rs = [row(1, "A", k, 0.0, 0.0) for k in range(len(flags))]
    assert L._fill_runs(flags, rs, {k: k for k in range(len(flags))}, value, max_len) == expected


def test_fill_runs_does_not_bridge_a_sample_gap():
    rs = [row(1, "A", k, 0.0, 0.0) for k in (0, 1, 5, 6)]  # samples 2-4 missing (not just calibration-less)
    pos = {0: 0, 1: 1, 5: 5, 6: 6}
    assert L._fill_runs([True, False, False, True], rs, pos, False, 2) == [True, False, False, True]
