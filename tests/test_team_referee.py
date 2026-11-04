"""Referee cues: grey-shirt test on a track histogram and the sideline position veto."""
import numpy as np

from replay import court
from replay import team_appearance as ta
from replay.config import TeamConfig
from replay.metadata import metadata_from_dict
from replay.teams import REF, UNKNOWN, _ref_shirt, classify
from tests.test_teams import _dataset


def _hist(black, grey, colour):
    h = np.zeros(ta.DESC_DIM)
    h[ta.N_HUE * ta.N_CHROMA] = black
    h[ta.N_HUE * ta.N_CHROMA + 1] = grey / 2
    h[ta.N_HUE * ta.N_CHROMA + 2] = grey / 2
    h[1 * ta.N_CHROMA + 1] = colour  # skin-ish
    h[ta.N_HUE * ta.N_CHROMA + 3] = max(0.0, 1 - black - grey - colour)
    return h


def test_ref_shirt_grey_over_black_vs_all_black_staff():
    cfg = TeamConfig()
    assert _ref_shirt(_hist(0.35, 0.18, 0.2), cfg)  # referee: grey shirt, black trousers
    assert not _ref_shirt(_hist(0.65, 0.07, 0.15), cfg)  # coach in black
    assert not _ref_shirt(_hist(0.05, 0.12, 0.7), cfg)  # coloured kit
    assert not _ref_shirt(_hist(0.05, 0.45, 0.1), cfg)  # spectator: grey hoodie, no black trousers
    assert not _ref_shirt(_hist(0.35, 0.18, 0.2), TeamConfig(ref_shirt_min_grey=0))  # switched off


def test_referee_beyond_sideline_is_not_a_referee():
    court.set_dimensions(28.0, 15.0)
    rng = np.random.default_rng(1)
    tid, fr, feats, app, truth = _dataset(rng)
    refs = [t for t, k in truth.items() if k == "ref"]
    xy = np.tile([[14.0, 7.0]], (len(tid), 1))
    meta = metadata_from_dict({})
    res = classify(tid, fr, feats, meta, TeamConfig(), appearance=app, xy=xy)
    assert all(res.track_team[t] == REF for t in refs)
    xy[tid == refs[0]] = [14.0, 15.0 + 2.5]  # sits 2.5 m beyond the far sideline the whole time
    res = classify(tid, fr, feats, meta, TeamConfig(), appearance=app, xy=xy)
    assert res.track_team[refs[0]] == UNKNOWN and res.track_team[refs[1]] == REF
    res = classify(tid, fr, feats, meta, TeamConfig(), appearance=app)  # without positions: colour only
    assert res.track_team[refs[0]] == REF


def test_black_kit_team_found_by_presence_not_taken_for_referees():
    """A team in black: by colour alone it looks like referees/staff; five of them on court at once make it a team."""
    rng = np.random.default_rng(7)
    tid, fr, feats, app, truth = _dataset(rng, tracks=["orange"] * 5 + ["bench"] * 5 + ["ref"] * 2)
    res = classify(tid, fr, feats, metadata_from_dict({}), TeamConfig(), appearance=app)
    black = {res.track_team[t] for t, k in truth.items() if k == "bench"}
    orange = {res.track_team[t] for t, k in truth.items() if k == "orange"}
    assert len(black) == 1 and len(orange) == 1 and black | orange == {"A", "B"}
    assert {res.track_team[t] for t, k in truth.items() if k == "ref"} == {REF}


def test_kit_colour_names_maroon_white_black():
    from replay.team_binding import lab_colour_name
    assert lab_colour_name([27.9, 22.7, 11.6]) == "red"  # maroon measured under gym light
    assert lab_colour_name([39.0, 38.8, 26.3]) == "orange"  # saturated orange kit
    assert lab_colour_name([85.0, 0.0, 0.0]) == "white" and lab_colour_name([12.0, 0.0, 0.0]) == "black"
