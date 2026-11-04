import pytest
import numpy as np

from replay.config import SmoothingConfig
from replay.smoothing import smooth_track

FPS = 10.0


def _track(ks):
    ks = np.asarray(ks)
    z = np.stack([1.0 + 0.2 * ks, np.full(len(ks), 5.0)], axis=1)  # 2 m/s along x
    R = np.tile(np.eye(2) * 0.01, (len(ks), 1, 1))
    return ks, z, R


def test_short_gap_is_interpolated_and_flagged():
    ks = list(range(0, 10)) + list(range(18, 30))  # 8 missing samples = 0.8 s
    segs = smooth_track(*_track(ks), FPS, SmoothingConfig())
    assert len(segs) == 1
    s = segs[0]
    assert s.k.tolist() == list(range(30))
    assert (~s.measured[10:18]).all() and s.measured[:10].all()
    np.testing.assert_allclose(s.xy[10:18, 0], 1.0 + 0.2 * np.arange(10, 18), atol=0.1)


def test_long_gap_is_not_filled():
    ks = list(range(0, 10)) + list(range(25, 35))  # 1.5 s gap
    segs = smooth_track(*_track(ks), FPS, SmoothingConfig(max_gap_s=1.0))
    assert len(segs) == 2
    covered = np.concatenate([s.k for s in segs])
    assert not np.isin(np.arange(10, 25), covered).any()


def test_outlier_is_rejected():
    ks, z, R = _track(range(20))
    z[10] += [4.0, 0.0]  # a 4 m jump in one frame (bad foot point)
    s = smooth_track(ks, z, R, FPS, SmoothingConfig())[0]
    assert not s.measured[10]
    assert abs(s.xy[10, 0] - (1.0 + 0.2 * 10)) < 0.3



# ---------------------------------------------------------------- robust, physically plausible smoother
from replay.smoothing import motion_model  # noqa: E402


def _run(fps, T=12.0, seed=0, noise=0.15):
    """A player jogging, stopping and cutting (|a| <= 4 m/s^2, |v| <= 6 m/s) with noisy foot points."""
    rng = np.random.default_rng(seed)
    n = int(T * fps)
    t = np.arange(n) / fps
    x = 5 + 3.0 * t - 0.6 * np.sin(0.8 * t) * 2
    y = 7 + 2.5 * np.sin(0.5 * t)
    X = np.stack([x, y], 1)
    z = X + rng.normal(0, noise, X.shape)
    R = np.tile(np.eye(2) * noise ** 2, (n, 1, 1))
    return np.arange(n), z, R, X


def test_frame_rate_independent():
    """Same settings give (nearly) the same track at 10 and 30 fps (continuous-time model)."""
    errs = []
    for fps in (10.0, 30.0):
        k, z, R, X = _run(fps)
        s = smooth_track(k, z, R, fps, SmoothingConfig())[0]
        errs.append(np.sqrt(np.mean(np.sum((s.xy - X) ** 2, 1))))
    assert max(errs) < 0.12
    assert abs(errs[0] - errs[1]) < 0.06


def test_smoothing_removes_jitter():
    k, z, R, X = _run(30.0)
    s = smooth_track(k, z, R, 30.0, SmoothingConfig())[0]
    a_raw = np.linalg.norm(np.diff(z, 2, axis=0), axis=1) * 30.0 ** 2
    a_s = np.linalg.norm(np.diff(s.xy, 2, axis=0), axis=1) * 30.0 ** 2
    assert np.median(a_s) < 0.02 * np.median(a_raw)
    assert np.percentile(np.linalg.norm(s.vel, axis=1), 99) < 8.0  # plausible running speeds


def test_outlier_run_from_cut_box_is_rejected():
    """A 0.3 s run of foot points 2 m too far (box cut / merged) must not drag the track."""
    fps = 30.0
    k, z, R, X = _run(fps, seed=1)
    z[150:159] += [0.0, 2.0]
    s = smooth_track(k, z, R, fps, SmoothingConfig())[0]
    assert (~s.measured[150:159]).all()
    assert np.abs(s.xy[150:159] - X[150:159]).max() < 0.4
    assert s.measured[:140].mean() > 0.95


def test_relocks_after_real_jump():
    """A persistent jump (e.g. an ID hand-over upstream) is followed after relock_s instead of lost forever."""
    fps = 10.0
    k = np.arange(60)
    z = np.tile([10.0, 7.0], (60, 1))
    z[30:] += [4.0, 0.0]
    R = np.tile(np.eye(2) * 0.02, (60, 1, 1))
    s = smooth_track(k, z, R, fps, SmoothingConfig())[0]
    assert abs(s.xy[-1, 0] - 14.0) < 0.1
    assert abs(s.xy[20, 0] - 10.0) < 0.3


def test_gap_bridge_does_not_overshoot():
    """Bridging a 2.5 s gap after a sprint: the velocity relaxes, no wild arc past the next sighting."""
    fps = 10.0
    k = np.r_[np.arange(0, 20), np.arange(45, 60)]
    t = k / fps
    x = np.where(k < 20, 5 + 6.0 * t, 21.0)  # sprint at 6 m/s, seen again standing at x = 21
    z = np.stack([x, np.full(len(k), 7.0)], 1)
    R = np.tile(np.eye(2) * 0.01, (len(k), 1, 1))
    s = smooth_track(k, z, R, fps, SmoothingConfig())[0]
    assert s.xy[:, 0].max() < 21.5
    assert np.abs(np.diff(s.xy[:, 0])).max() * fps < 8.0
    assert (~s.measured[20:45]).all()


def test_motion_model_limits():
    cfg = SmoothingConfig()
    F, Q = motion_model(0.1, cfg)
    assert np.allclose(Q, Q.T) and np.all(np.linalg.eigvalsh(Q) > -1e-12)
    F0, Q0 = motion_model(0.1, SmoothingConfig(velocity_tau_s=0))
    assert F0[0, 2] == 0.1 and F0[2, 2] == 1.0
    assert 0 < F[2, 2] < 1


def test_state_and_sigma_for_ui():
    """Per-sample state (seen / predicted / rejected) and an uncertainty that grows inside gaps."""
    from replay.smoothing import STATE_PREDICTED, STATE_REJECTED, STATE_SEEN
    ks = list(range(0, 10)) + list(range(20, 30))
    k, z, R = _track(ks)
    z[3] += [4.0, 0.0]
    s = smooth_track(k, z, R, FPS, SmoothingConfig())[0]
    assert s.state[0] == STATE_SEEN and s.state[3] == STATE_REJECTED
    assert (s.state[10:20] == STATE_PREDICTED).all()
    assert s.sigma[15] > 3 * s.sigma[5]
    assert s.sigma[15] >= s.sigma[11] and s.sigma[15] >= s.sigma[19]


# ---------------------------------------------------------------- occlusion-aware gaps
def _hidden_const(code):
    return lambda ks, xy: np.full(len(ks), code)


def test_hidden_long_gap_is_bridged_and_labelled():
    from replay.smoothing import STATE_OCCLUDED
    ks = list(range(0, 10)) + list(range(50, 60))  # 4 s gap
    cfg = SmoothingConfig(max_gap_s=3.0, occluded_max_gap_s=6.0)
    assert len(smooth_track(*_track(ks), FPS, cfg)) == 2  # without occlusion cues: split
    segs = smooth_track(*_track(ks), FPS, cfg, hidden=_hidden_const(1))
    assert len(segs) == 1
    s = segs[0]
    assert (s.state[10:50] == STATE_OCCLUDED).all()
    np.testing.assert_allclose(s.xy[30, 0], 1.0 + 0.2 * 30, atol=0.5)


def test_visible_empty_long_gap_is_not_bridged():
    ks = list(range(0, 10)) + list(range(50, 60))
    cfg = SmoothingConfig(max_gap_s=3.0, occluded_max_gap_s=6.0)
    assert len(smooth_track(*_track(ks), FPS, cfg, hidden=_hidden_const(0))) == 2


def test_impossible_hidden_gap_is_not_bridged():
    ks = np.array(list(range(0, 10)) + list(range(50, 60)))
    k, z, R = _track(ks)
    z[10:] += [30.0, 0.0]  # 30 m in 4 s
    cfg = SmoothingConfig(max_gap_s=3.0, occluded_max_gap_s=6.0)
    assert len(smooth_track(k, z, R, FPS, cfg, hidden=_hidden_const(2))) == 2


def test_occlusion_context_classifies_behind_offscreen_visible():
    from replay.court_tracker import OCCLUDED, OFFSCREEN, VISIBLE, OcclusionContext
    H = [np.diag([0.02, 0.02, 1.0])]  # 50 px per metre, image = court * 50
    occluder = np.array([[90.0, 100.0, 130.0, 260.0]])  # person whose feet are at y=260 px
    ctx = OcclusionContext(H, [occluder], (1280, 720))
    assert ctx.classify(0, np.array([2.2, 4.0]))[0] == OCCLUDED  # foot pixel (110, 200): behind the occluder
    assert ctx.classify(0, np.array([2.2, 6.0]))[0] == VISIBLE  # (110, 300): in front of the occluder
    assert ctx.classify(0, np.array([10.0, 4.0]))[0] == VISIBLE
    assert ctx.classify(0, np.array([30.0, 4.0]))[0] == OFFSCREEN  # x = 1500 px


def test_capped_long_gap_keeps_only_sure_ends():
    ks = list(range(0, 20)) + list(range(100, 120))  # 8 s gap
    cfg = SmoothingConfig(max_gap_s=3.0, capped_gap_max_s=10.0, sigma_cap_m=1.5)
    segs = smooth_track(*_track(ks), FPS, cfg)
    covered = np.concatenate([s.k for s in segs])
    assert 25 in covered and 95 in covered  # 0.5 s after / before a sighting: shown
    assert 60 not in covered  # the unsure middle stays empty
    assert all(s.sigma.max() <= 1.5 + 1e-9 or s.state.min() == 1 for s in segs)
    assert len(smooth_track(*_track(ks), FPS, SmoothingConfig(max_gap_s=3.0))) == 2


def test_phase_scaled_sigma():
    from replay.smoothing import _phase_sigma

    class Prior:
        def sigma_scale(self, i):
            return np.full(len(i), 0.4)
    ks = list(range(0, 10)) + list(range(20, 30))
    s = smooth_track(*_track(ks), FPS, SmoothingConfig())[0]
    before = s.sigma[15]
    _phase_sigma(s, Prior())
    assert s.sigma[15] < 0.6 * before and s.sigma[5] == pytest.approx(s.sigma[5])



def test_smooth_track_keeps_the_confident_duplicate():
    ks = np.array([0, 1, 2, 2, 3, 4])
    z = np.array([[0, 5], [0.2, 5], [0.4, 5], [3.4, 5], [0.6, 5], [0.8, 5]], float)
    R = np.tile(np.eye(2) * 0.01, (6, 1, 1))
    conf = np.array([0.9, 0.9, 0.9, 0.3, 0.9, 0.9])
    s = smooth_track(ks, z, R, FPS, SmoothingConfig(), conf=conf)[0]
    assert abs(s.xy[2, 0] - 0.4) < 0.05
