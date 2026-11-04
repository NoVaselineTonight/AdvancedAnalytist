"""replay/handover.py: unambiguous output-id handovers get one id."""
from types import SimpleNamespace as R

from replay import handover


def _rows(tid, t0, t1, x0, team="A", step=0.1, pid=None):
    out, t = [], t0
    while t <= t1 + 1e-9:
        out.append(R(track_id=tid, object_type="player", team=team, t=round(t, 3), x=x0 + (t - t0), y=5.0,
                     player_id=pid, ball_owner=None))
        t += step
    return out


def test_handover_joined_and_chain():
    rows = _rows(1, 0, 2, 0) + _rows(2, 2.3, 4, 2.4) + _rows(3, 4.5, 6, 4.6)
    assert handover.plan(rows, {}) == {2: 1, 3: 1}


def test_handover_kept_apart():
    far = _rows(1, 0, 2, 0) + _rows(2, 2.3, 4, 9.0)  # unreachable
    team = _rows(1, 0, 2, 0) + _rows(2, 2.3, 4, 2.4, team="B")
    nums = _rows(1, 0, 2, 0) + _rows(2, 2.3, 4, 2.4)
    amb = _rows(1, 0, 2, 0) + _rows(2, 2.3, 4, 2.4) + _rows(3, 2.4, 4, 2.6)  # two successors
    assert handover.plan(far, {}) == {} and handover.plan(team, {}) == {}
    assert handover.plan(nums, {1: {"number": "7"}, 2: {"number": "9"}}) == {}
    assert handover.plan(amb, {}) == {}


def test_stitch_respects_switch(monkeypatch):
    rows = _rows(1, 0, 2, 0) + _rows(2, 2.3, 4, 2.4)
    monkeypatch.setenv("REPLAY_HANDOVER", "0")
    assert handover.stitch(rows, {1: {}, 2: {}})[1] == 0
    monkeypatch.setenv("REPLAY_HANDOVER", "1")
    info, n = handover.stitch(rows, {1: {"team": "A"}, 2: {"team": "A", "number": "7"}})
    assert n == 1 and set(info) == {1} and info[1]["number"] == "7" and {r.track_id for r in rows} == {1}
