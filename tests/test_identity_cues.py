"""Identity round (replay/tracklets.py): chimera prevention, jersey-number anchors, identity confidence."""

import numpy as np

from replay.config import CourtTrackConfig
from replay.tracklets import IdentityCues, link_tracks

from test_tracklets import FPS, ORANGE, PURPLE, _scene

DIM = 40


def _hist(bin_):
    h = np.zeros(DIM)
    h[bin_] = 1.0
    return h


def _run(s, numbers=None, hist=None):
    cues = None if hist is None else IdentityCues(hist=hist, hist_occ=np.zeros(len(s["k"])))
    return link_tracks(s["k"], s["xy"], s["R"], s["feats"], s["bt"], s["conf"], FPS, CourtTrackConfig(),
                       boxes=s["boxes"], motion=s["motion"], n_samples=s["n"], H=s["H"], numbers=numbers, cues=cues)


def test_no_chimera_through_a_crumb_between_different_kits():
    """Orange player ends, a few low-score boxes without colour (crumb) bridge the gap, a purple player starts
    where the orange one stopped: the chain orange -> crumb -> purple must not become one identity."""
    n = 70
    a = np.c_[np.linspace(6, 10, n), np.full(n, 6.0)]
    s = _scene({1: a, 2: a.copy() + [0.2, 0.0], 3: a.copy() + [0.1, 0.0]}, {1: ORANGE, 2: PURPLE, 3: ORANGE},
               bt_of=lambda p, k: {1: 1, 2: 2, 3: -1}[p],
               hidden={1: range(30, n), 2: range(0, 42), 3: [*range(0, 33), *range(37, n)]})
    crumb = s["pid"] == 3
    s["feats"][crumb] = np.nan
    s["conf"][crumb] = 0.3
    hist = np.array([_hist(0) if p == 1 else _hist(30) for p in s["pid"]])
    hist[crumb] = np.nan
    res = _run(s, hist=hist)
    ids = res.ids
    assert set(ids[s["pid"] == 1]).isdisjoint(set(ids[s["pid"] == 2]))
    for i, info in res.identities.items():
        assert 0.0 <= info["confidence"] <= 1.0


def test_no_chimera_when_only_the_kit_histogram_tells_the_kits_apart():
    """Harder: the pass-1 Lab medians of both players are the same (dark purple side-on vs orange often are), only the
    video kit histograms differ. The old linker joined orange -> crumb -> purple here."""
    n = 70
    a = np.c_[np.linspace(6, 10, n), np.full(n, 6.0)]
    s = _scene({1: a, 2: a.copy() + [0.2, 0.0], 3: a.copy() + [0.1, 0.0]}, {1: ORANGE, 2: ORANGE, 3: ORANGE},
               bt_of=lambda p, k: {1: 1, 2: 2, 3: -1}[p],
               hidden={1: range(30, n), 2: range(0, 42), 3: [*range(0, 33), *range(37, n)]})
    crumb = s["pid"] == 3
    s["conf"][crumb] = 0.3
    hist = np.array([_hist(0) if p == 1 else _hist(30) for p in s["pid"]])
    hist[crumb] = np.nan
    res = _run(s, hist=hist)
    assert set(res.ids[s["pid"] == 1]).isdisjoint(set(res.ids[s["pid"] == 2]))


def _gap_scene(num_a, num_b):
    """One orange player seen, out of the picture for 12 s, seen again elsewhere with a new ByteTrack ID; a
    second orange player is visible all the time (so kit and motion alone cannot decide)."""
    n = 200
    p1 = np.c_[np.linspace(5, 20, n), np.full(n, 4.0)]
    p2 = np.c_[np.linspace(20, 8, n), np.full(n, 10.0)]
    s = _scene({1: p1, 2: p2}, {1: ORANGE, 2: ORANGE}, bt_of=lambda p, k: (p if k < 100 else p + 10) if p == 1 else 2,
               hidden={1: range(40, 160)})
    num = np.full(len(s["k"]), -1)
    first = (s["pid"] == 1) & (s["k"] < 100)
    second = (s["pid"] == 1) & (s["k"] >= 100)
    num[np.flatnonzero(first)[::8]] = num_a
    num[np.flatnonzero(second)[::8]] = num_b
    num[np.flatnonzero(s["pid"] == 2)[::10]] = 31
    return s, num


def test_same_number_and_kit_is_one_person_over_a_long_gap():
    s, num = _gap_scene(23, 23)
    res = _run(s, numbers=num, hist=np.tile(_hist(0), (len(s["k"]), 1)))
    assert len(set(res.ids[s["pid"] == 1])) == 1
    assert set(res.ids[s["pid"] == 1]).isdisjoint(set(res.ids[s["pid"] == 2]))
    info = res.identities[int(res.ids[s["pid"] == 1][0])]
    assert info["number"] == 23 and info["strong_links"] >= 1


def test_different_numbers_are_never_joined():
    s, num = _gap_scene(23, 45)
    res = _run(s, numbers=num, hist=np.tile(_hist(0), (len(s["k"]), 1)))
    a = set(res.ids[(s["pid"] == 1) & (s["k"] < 100)])
    b = set(res.ids[(s["pid"] == 1) & (s["k"] >= 100)])
    assert a.isdisjoint(b)


def test_same_number_other_kit_is_not_joined():
    """Both teams can have a #23: the anchor needs the same kit."""
    s, num = _gap_scene(23, 23)
    hist = np.array([_hist(0) if k < 100 else _hist(30) for k in s["k"]])
    hist[s["pid"] == 2] = _hist(0)
    res = _run(s, numbers=num, hist=hist)
    a = set(res.ids[(s["pid"] == 1) & (s["k"] < 100)])
    b = set(res.ids[(s["pid"] == 1) & (s["k"] >= 100)])
    assert a.isdisjoint(b)


def test_joint_tracker_stage_runs_and_respects_numbers(monkeypatch):
    """joint_tracker (REPLAY_JOINT=1): same scene as the number-anchor test; one identity, different numbers never."""
    import replay.tracklets as tl

    monkeypatch.setattr(tl, "JOINT_TRACKER", True)
    s, num = _gap_scene(23, 23)
    res = _run(s, numbers=num, hist=np.tile(_hist(0), (len(s["k"]), 1)))
    assert len(set(res.ids[s["pid"] == 1])) == 1
    assert "joint_candidates" in res.stats
    s, num = _gap_scene(23, 45)
    res = _run(s, numbers=num, hist=np.tile(_hist(0), (len(s["k"]), 1)))
    assert set(res.ids[(s["pid"] == 1) & (s["k"] < 100)]).isdisjoint(set(res.ids[(s["pid"] == 1) & (s["k"] >= 100)]))
