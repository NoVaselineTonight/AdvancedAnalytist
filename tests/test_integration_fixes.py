"""Integration fixes: old configs load silently, vectorised helpers match the scalar originals exactly."""

import json

import numpy as np

from replay import homography
from replay.config import Config
from replay.tracklets import _SPLIT_W, _appearance_cuts, _rolling_upper


def test_old_config_used_json_loads_silently(capsys):
    old = {"frame_stride": 1,
           "smoothing": {"accel_noise": 4.0, "max_gap_s": 2.5},
           "filter": {"min_in_court_frac": 0.6, "min_moving_speed_mps": 0.25, "max_players": 11},
           "ball": {"gate_px": 80.0, "max_coast_s": 1.0},
           "teams": {"unknown_distance": 45.0, "min_vote_share": 0.55},
           "court_tracking": {"accel_noise": 3.0}}
    cfg = Config.from_dict(json.loads(json.dumps(old)))
    assert capsys.readouterr().out == ""
    assert cfg.smoothing.max_gap_s == 2.5 and cfg.smoothing.accel_sigma == Config().smoothing.accel_sigma
    assert cfg.filter.max_players == 11
    assert cfg.court_tracking.accel_noise == 3.0  # still a valid key of the online tracker
    Config.from_dict({"filter": {"no_such_key": 1}})
    assert "no_such_key" in capsys.readouterr().out  # genuinely unknown keys are still reported


def test_rolling_upper_matches_numpy_percentile():
    rng = np.random.default_rng(0)
    for it in range(400):
        n = int(rng.integers(1, 50))
        kk = np.sort(rng.choice(150, n, replace=False))
        v = rng.normal(size=n) * rng.uniform(0.1, 50)
        if it % 5 == 0:
            v[rng.integers(n)] = np.nan
        half, q = int(rng.integers(1, 12)), float(rng.choice([50.0, 80.0, 33.3, 0.0, 100.0]))
        lo, hi = np.searchsorted(kk, kk - half, "left"), np.searchsorted(kk, kk + half, "right")
        ref = np.array([np.percentile(v[a:b], q) for a, b in zip(lo, hi)])
        assert np.array_equal(_rolling_upper(kk, v, half, q), ref, equal_nan=True)


def _appearance_cuts_reference(f, w, thr, min_len, allowed=None):
    n = len(f)
    if n < 2 * min_len:
        return []
    fin = np.isfinite(f).all(axis=1)
    best, best_j = 0.0, -1
    for j in range(min_len, n - min_len + 1):
        if allowed is not None and not allowed[j]:
            continue
        a = f[max(0, j - w):j][fin[max(0, j - w):j]]
        b = f[j:j + w][fin[j:j + w]]
        if len(a) < min_len or len(b) < min_len:
            continue
        d = float(np.linalg.norm((np.median(a, 0) - np.median(b, 0)) * _SPLIT_W))
        if d > best:
            best, best_j = d, j
    if best < thr:
        return []
    al, ar = (None, None) if allowed is None else (allowed[:best_j], allowed[best_j:])
    left = _appearance_cuts_reference(f[:best_j], w, thr, min_len, al)
    right = _appearance_cuts_reference(f[best_j:], w, thr, min_len, ar)
    return left + [best_j] + [best_j + r for r in right]


def test_appearance_cuts_match_reference():
    rng = np.random.default_rng(1)
    for it in range(300):
        n = int(rng.integers(1, 100))
        f = rng.normal(size=(n, 6)) * 8
        for _ in range(int(rng.integers(0, 3))):
            f[int(rng.integers(0, max(1, n))):] += rng.normal(size=6) * 20
        f[rng.random(n) < 0.15] = np.nan
        if it % 4 == 0:
            f = np.round(f)  # ties
        w = int(rng.integers(3, 12))
        allowed = rng.random(n) < 0.7 if it % 2 else None
        args = (f, w, 18.0, max(3, w // 2), allowed)
        assert _appearance_cuts(*args) == _appearance_cuts_reference(*args)


def test_batched_projection_matches_scalar():
    rng = np.random.default_rng(2)
    Hs = np.stack([homography.normalize(np.eye(3) + rng.normal(scale=[[0.1, 0.1, 5], [0.1, 0.1, 5], [1e-4, 1e-3, 0.1]]),
                                        (640, 680)) for _ in range(50)])
    px = rng.uniform([0, 300], [1280, 720], size=(50, 2))
    P = homography.project_each(Hs, px[:, None, :])[:, 0]
    J = homography.jacobian_each(Hs, px)
    for H, p, q, j in zip(Hs, px, P, J):
        assert np.array_equal(q, homography.project(H, p[None])[0], equal_nan=True)
        assert np.array_equal(j, homography.jacobian(H, tuple(p)), equal_nan=True)

