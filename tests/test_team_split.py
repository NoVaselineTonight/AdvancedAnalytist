"""Splitting court tracks whose jersey colour changes for a sustained time (identity switch between kits)."""
import numpy as np

from replay.config import TeamConfig
from replay.teams import A, B, REF, UNKNOWN, TeamResult, _viterbi, split_mixed_tracks


def _result(track_team, det_label):
    return TeamResult(dict(track_team), {t: 0.9 for t in track_team}, {}, False, {}, [], {}, "", np.array(det_label, object))


def test_viterbi_ignores_short_contrary_runs():
    ev = [A] * 10 + [B] * 3 + [A] * 10
    assert set(_viterbi(ev, [A, B, "neutral"], 10.0)) == {A}


def test_viterbi_switches_on_sustained_change():
    ev = [A] * 24 + [B] * 24
    path = _viterbi(ev, [A, B, "neutral"], 10.0)
    assert path[:24] == [A] * 24 and path[24:] == [B] * 24


def test_split_orange_then_referee():
    n = 40
    tid = np.ones(n, int) * 5
    det_i = np.arange(n)
    lab = [A] * 20 + [REF, REF, UNKNOWN] * 6 + [REF, REF]
    teams = _result({5: A}, lab)
    new = split_mixed_tracks(tid, np.ones(n, bool), det_i, teams, TeamConfig())
    assert len(set(new.tolist())) == 2
    assert (new[:20] == 5).all() and teams.track_team[5] == A
    other = int(new[-1])
    assert other != 5 and (new[22:] == other).all() and teams.track_team[other] == REF


def test_no_split_for_flicker_and_disabled():
    n = 40
    tid = np.ones(n, int) * 3
    lab = ([A] * 4 + [B]) * 8
    teams = _result({3: A}, lab)
    assert (split_mixed_tracks(tid, np.ones(n, bool), np.arange(n), teams, TeamConfig()) == 3).all()
    lab2 = [A] * 20 + [B] * 20
    cfg = TeamConfig(split_mixed_tracks=False)
    assert (split_mixed_tracks(tid, np.ones(n, bool), np.arange(n), _result({3: A}, lab2), cfg) == 3).all()


def test_split_respects_keep_mask_and_missing_evidence():
    n = 60
    tid = np.ones(n, int) * 7
    keep = np.ones(n, bool)
    keep[::10] = False  # dropped rows are not part of the classified input
    sub = np.flatnonzero(keep)
    lab = np.array([A if i < 27 else B for i in range(len(sub))], object)
    lab[1::3] = ""  # no histogram for these sightings
    teams = _result({7: A}, lab)
    new = split_mixed_tracks(tid, keep, np.arange(n), teams, TeamConfig())
    ids = set(new[keep].tolist())
    assert len(ids) == 2 and 7 in ids
    nb = (ids - {7}).pop()
    assert teams.track_team[nb] == B and teams.track_team[7] == A
    assert (new[~keep] == 7).all()  # untouched rows keep their id


def test_classify_then_split_synthetic_identity_switch():
    """A court track that is an orange player first and the referee later is split into the two (real histograms)."""
    from replay.metadata import metadata_from_dict
    from replay.teams import classify
    from tests.test_teams import _dataset
    rng = np.random.default_rng(5)
    tid, fr, feats, app, truth = _dataset(rng, n_per=24)
    orange = next(t for t, k in truth.items() if k == "orange")
    ref = next(t for t, k in truth.items() if k == "ref")
    keep = ~((tid == orange) & (fr >= 12))  # orange player's second half dropped ...
    tid = tid.copy()
    tid[(tid == ref) & (fr >= 12)] = orange  # ... and the referee's second half linked into the orange track
    tid, fr, feats = tid[keep], fr[keep], feats[keep]
    app = ta_subset(app, np.flatnonzero(keep))
    res = classify(tid, fr, feats, metadata_from_dict({}), TeamConfig(), appearance=app)
    new = split_mixed_tracks(tid, np.ones(len(tid), bool), fr, res, TeamConfig())
    m = tid == orange
    first, second = set(new[m & (fr < 12)].tolist()), set(new[m & (fr >= 12)].tolist())
    assert len(first) == 1 and len(second) == 1 and first != second
    assert res.track_team[second.pop()] == REF
    assert res.track_team[first.pop()] in (A, B)


def ta_subset(app, idx):
    from replay import team_appearance as ta
    return ta.Appearance(app.desc[idx], app.occ[idx], np.arange(len(idx)), app.height[idx], None, app.sector_lab[idx])


def test_unknown_evidence_splits_only_when_enabled():
    """Dark kits seen side-on often look like nobody's kit: by default that never splits a team track."""
    n = 40
    tid = np.ones(n, int) * 9
    lab = [B] * 20 + [UNKNOWN] * 20
    teams = _result({9: B}, lab)
    assert (split_mixed_tracks(tid, np.ones(n, bool), np.arange(n), teams, TeamConfig()) == 9).all()
    assert teams.track_team[9] == B
    teams = _result({9: B}, lab)
    new = split_mixed_tracks(tid, np.ones(n, bool), np.arange(n), teams, TeamConfig(split_on_unknown=True))
    assert len(set(new.tolist())) == 2 and teams.track_team[9] in (B, UNKNOWN)
