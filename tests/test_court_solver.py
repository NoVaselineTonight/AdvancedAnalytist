import numpy as np
import cv2
import pytest

from replay import autocourt, court, court_solver as cs
from replay.camera_model import rotation


SIZE = (1280, 720)


def _true_H():
    court.configure(28.0, 15.0, "FIBA")
    C = np.array([14.0, -8.0, 7.0])
    R = rotation(np.radians(90.0), np.radians(25.0), 0.0)
    return autocourt.court_to_image_R(C, R, 1100.0, SIZE)


def _dist(H, pts):
    s = cs.WORK_W / SIZE[0]
    hh = int(round(SIZE[1] * s))
    m = np.zeros((hh, cs.WORK_W), np.uint8)
    p = cs.project(H, pts, s)
    ok = np.isfinite(p[:, 0]) & (p[:, 0] >= 0) & (p[:, 0] < cs.WORK_W) & (p[:, 1] >= 0) & (p[:, 1] < hh)
    m[p[ok, 1].astype(int), p[ok, 0].astype(int)] = 1
    d = cv2.distanceTransform(1 - m, cv2.DIST_L2, 3)
    return np.round(np.minimum(d, cs.TRUNC) * cs.DT_SCALE).astype(np.uint8)


@pytest.mark.xfail(reason="stage 0 WIP: refine_H does not move yet (x stays 0; diff_step/x_scale issue)", strict=False)
def test_refine_recovers_a_perturbed_homography():
    H = _true_H()
    pts, cls = cs.class_samples(0.1)
    dist = _dist(H, pts)
    sc_true = cs.score_H(dist, H, SIZE, pts, cls)
    assert sc_true["score"] > 0.75  # rasterised to whole pixels
    ok, why = cs.accept(sc_true, H, SIZE, pts)
    assert ok, why
    # perturb the image corners of the court by ~20 px
    src = np.array([[0, 0], [28, 0], [28, 15], [0, 15]], np.float32)
    dst = cs.project(H, src).astype(np.float32)
    rng = np.random.default_rng(0)
    H0 = cv2.getPerspectiveTransform(src, dst + rng.uniform(-20, 20, dst.shape).astype(np.float32))
    Hr = cs.refine_H(dist, H0, SIZE, pts)
    err = np.linalg.norm(cs.project(Hr, src) - cs.project(H, src), axis=1)
    err0 = np.linalg.norm(cs.project(H0, src) - cs.project(H, src), axis=1)
    assert np.median(err) < 0.25 * np.median(err0)
    assert np.median(err) < 4.0


def test_accept_rejects_mirrored_court():
    H = _true_H()
    pts, cls = cs.class_samples(0.1)
    Fx = np.array([[-1, 0, 28.0], [0, 1, 0], [0, 0, 1]])  # mirror x: projects onto the same lines, wrong handedness
    Fy = np.array([[1, 0, 0], [0, -1, 15.0], [0, 0, 1]])  # mirror y: far side below the near side
    assert not cs.plausible(H @ Fx, SIZE, pts)[0]
    assert not cs.plausible(H @ Fy, SIZE, pts)[0]
    assert cs.plausible(H, SIZE, pts)[0]


def test_camera_from_H_recovers_height_and_f():
    H = _true_H()
    C, tilt, f = cs.camera_from_H(H, SIZE)
    assert abs(C[2] - 7.0) < 0.05 and abs(f - 1100.0) < 5.0 and abs(tilt - 25.0) < 0.5
