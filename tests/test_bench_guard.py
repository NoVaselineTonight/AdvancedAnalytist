import math
import random

from replay.bench_guard import bench_identities, score_identities, score_track

L, W = 26.0, 14.5


def _rows(xy, n=60, dt=0.5, jitter=0.1, seed=0, measured=True):
    rng = random.Random(seed)
    return [(k * dt, xy[0] + rng.uniform(-jitter, jitter), xy[1] + rng.uniform(-jitter, jitter), measured)
            for k in range(n)]


def test_bench_kid_beyond_far_sideline_is_flagged():
    # c300 case: team-kit kid at x 20, y 15.2 (0.7 m beyond the far sideline) for 30 s, shuffling a bit
    s = score_track(_rows((20.0, 15.2), jitter=0.3), L, W)
    assert s.is_bench and s.score >= 0.75 and s.side == "far"
    assert s.beyond_share > 0.9 and s.deep_share == 0


def test_seated_kid_just_inside_sideline_is_flagged():
    # p4400 case: sits at x 21, y 13.8-14.4 (inside the court by < 1 m), moves slowly, y range ~1 m
    rng = random.Random(1)
    rows = [(k * 0.5, 21 + rng.uniform(-1, 1), 14.1 + rng.uniform(-0.4, 0.4), True) for k in range(80)]
    s = score_track(rows, L, W)
    assert s.is_bench, s


def test_real_player_crossing_court_is_not_flagged():
    rows = [(k * 0.5, 2 + k * 0.4, 7 + 3 * math.sin(k / 5), True) for k in range(60)]
    s = score_track(rows, L, W)
    assert not s.is_bench and s.deep_share > 0.5 and s.score < 0.3


def test_player_inbounding_on_the_sideline_then_playing_is_not_flagged():
    # 5 s standing on the sideline, then 20 s in play
    rows = _rows((13.0, 14.3), n=10) + [(5 + k * 0.5, 13 + k * 0.2, 14 - k * 0.3, True) for k in range(40)]
    s = score_track(rows, L, W)
    assert not s.is_bench


def test_substituted_player_walking_to_the_bench_is_not_flagged():
    # whole life: 15 s on court, then 15 s standing at the bench -> entered_from_court
    rows = [(k * 0.5, 10 + k * 0.1, 7.0, True) for k in range(30)] + _rows((21.0, 14.8), n=30)
    rows = [(t if i < 30 else 15 + t, x, y, m) for i, (t, x, y, m) in enumerate(rows)]
    s = score_track(rows, L, W)
    assert s.entered_from_court and not s.is_bench


def test_player_standing_at_the_sideline_mid_court_with_bench_x_range():
    # standing still at x 8 (not along the bench, inside the court): not a bench row when bench_x is given
    rows = _rows((8.0, 13.8), n=40)
    assert score_track(rows, L, W, bench_x=(18, 24)).is_bench is False
    assert score_track(rows, L, W).is_bench is True  # without the range it looks like a bench person


def test_short_track_is_undecided():
    assert not score_track(_rows((20.0, 15.2), n=3), L, W).is_bench


def test_referee_never_flagged_and_set_helper():
    tracks = {"1": _rows((20.0, 15.2)), "2": _rows((20.0, 15.2), seed=2), "3": _rows((13.0, 7.0))}
    team = {"1": "A", "2": "referee", "3": "B"}
    sc = score_identities(tracks, L, W, team=team)
    assert sc["1"].is_bench and not sc["2"].is_bench and not sc["3"].is_bench
    assert bench_identities(tracks, L, W, team=team) == {"1"}


def test_predicted_rows_ignored_when_enough_measured():
    rows = _rows((20.0, 15.2), n=40) + [(100 + k, 13.0, 7.0, False) for k in range(40)]
    s = score_track(rows, L, W)
    assert s.rows == 40 and s.is_bench
