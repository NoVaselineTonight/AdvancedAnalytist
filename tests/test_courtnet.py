"""Court detector geometry (replay/courtnet.py) on exact synthetic labels: keypoint peaks -> homography."""

from __future__ import annotations

import numpy as np

from replay import court, court_synth, courtnet


def test_frame_H_recovers_the_true_homography_from_exact_heatmaps():
    rng = np.random.default_rng(5)
    ok = 0
    for _ in range(6):
        img, line, heat, meta = court_synth.render(rng)
        size = (court_synth.IN_W, court_synth.IN_H)
        pk = courtnet.peaks(heat, size, thr=0.5)
        cxy = courtnet.court_points(list(court.KEYPOINTS))  # render() left the court configured
        r = courtnet.frame_H(pk, cxy, size)
        if r is None:
            continue
        H, _ids = r
        P = np.column_stack([cxy[list(pk)], np.ones(len(pk))])
        a, b = P @ H.T, P @ meta["H"].T
        err = np.linalg.norm(a[:, :2] / a[:, 2:] - b[:, :2] / b[:, 2:], axis=1)
        assert np.median(err) < 3.0
        ok += 1
    court.configure(28.0, 15.0, "FIBA")
    assert ok >= 3
