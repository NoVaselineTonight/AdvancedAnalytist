"""Automatic court calibration on synthetic videos with a known camera.

The full end-to-end cases take a minute or two each; they run when RUN_SLOW=1 is set
(``RUN_SLOW=1 python -m pytest tests/test_autocourt.py``). The fast tests always run.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from replay import autocourt, court
from replay.camera_model import rotation

slow = pytest.mark.skipif(not os.environ.get("RUN_SLOW"), reason="slow end-to-end test (set RUN_SLOW=1)")


def test_ptz_roundtrip():
    R = rotation(1.9, 0.2, 0.03)
    v = autocourt.ptz_from_R(R, 1000.0)
    assert np.allclose(rotation(v.pan, v.tilt, v.roll), R, atol=1e-9)


def test_vanishing_points_recover_rotation():
    """Line segments along the three world axes -> the camera rotation and focal length (pan mod 90)."""
    size = (1280, 720)
    R = rotation(np.radians(118.0), np.radians(9.0), np.radians(0.8))
    f = 950.0
    C = np.array([13.0, -4.0, 2.5])
    K = autocourt.K_of(f, size)
    rng = np.random.default_rng(1)
    segs = []
    for _ in range(300):
        p = np.array([rng.uniform(0, 28), rng.uniform(0, 15), rng.uniform(0, 4)])
        d = np.eye(3)[rng.integers(0, 3)] * rng.uniform(0.5, 2.0)
        a, b = (K @ (R @ (np.stack([p, p + d]) - C).T)).T[:, [0, 1]] / (K @ (R @ (np.stack([p, p + d]) - C).T)).T[:, 2:3]
        if np.all(np.isfinite([a, b])) and 0 < a[0] < 1280 and 0 < b[0] < 1280 and 0 < a[1] < 720 and 0 < b[1] < 720:
            segs.append([*a, *b])
    Ls = np.array(segs)
    ln = np.hypot(Ls[:, 2] - Ls[:, 0], Ls[:, 3] - Ls[:, 1])
    keep = ln > 20
    x, inl = autocourt.fit_rotation(Ls[keep], ln[keep], size)
    assert x is not None and inl > 0.8
    assert abs(np.degrees(x[0]) % 90 - 118.0 % 90) < 1.0
    assert abs(np.degrees(x[1]) - 9.0) < 1.0
    assert abs(np.exp(x[3]) / f - 1) < 0.05


def test_court_standards_and_size():
    court.configure(26.0, 14.0, "NBA")
    try:
        assert court.LENGTH == 26.0 and court.WIDTH == 14.0
        assert abs(court.KEYPOINTS["L_three_top"][0] - (1.60 + 7.24)) < 1e-9
        assert court.KEYPOINTS["mid_far"] == (13.0, 14.0)
    finally:
        court.configure(28.0, 15.0, "FIBA")
    assert court.describe() == {"length": 28.0, "width": 15.0, "spec": "FIBA"}


def _run_case(tmp_path, scene, cam, max_px=6.0, min_frac=0.8):
    from synthetic_court import render_video, reprojection_error

    video = tmp_path / "syn.mp4"
    data = render_video(video, scene, cam)
    res = autocourt.calibrate(video, data, scene.size)
    assert res is not None, "no calibration found"
    assert abs(res.length - scene.length) <= 0.6, res.message
    Hs, _ = autocourt.resolve_auto(data.t, data.motion, res.to_json(), scene.size)
    err = reprojection_error(Hs, data, scene)
    ok = np.isfinite(err)
    assert ok.mean() >= min_frac, res.message
    assert np.median(err[ok]) <= max_px, (res.message, np.nanpercentile(err, [50, 90]))
    court.configure(28.0, 15.0, "FIBA")
    return res, err


@slow
def test_broadcast_style_white_lines_on_wood(tmp_path):
    from synthetic_court import CameraPath, Scene

    _run_case(tmp_path, Scene(), CameraPath(C0=(14.0, -7.0, 6.0), tilt_deg=16.0))


@slow
def test_small_court_black_lines_low_camera(tmp_path):
    from synthetic_court import CameraPath, Scene

    _run_case(tmp_path, Scene(length=26.0, width=14.0, floor_bgr=(200, 200, 198), line_bgr=(30, 30, 30)),
              CameraPath(C0=(13.0, -5.0, 2.0), tilt_deg=6.0, f0=900.0, pan_range_deg=(40.0, 140.0)))


@slow
def test_moving_camera_along_sideline(tmp_path):
    from synthetic_court import CameraPath, Scene

    _run_case(tmp_path, Scene(), CameraPath(C0=(8.0, -6.0, 3.0), C1=(20.0, -6.0, 3.0), tilt_deg=10.0))


@slow
def test_nba_court_with_zoom(tmp_path):
    from synthetic_court import CameraPath, Scene

    _run_case(tmp_path, Scene(length=28.65, width=15.24, spec="NBA", line_bgr=(40, 40, 160)),
              CameraPath(C0=(14.3, -9.0, 8.0), tilt_deg=20.0, f0=1300.0, f1=1700.0))
