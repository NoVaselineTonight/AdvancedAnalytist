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


@pytest.mark.parametrize("pan_deg,seed", [(60.0, 1), (90.0, 2), (120.0, 3), (75.0, 4)])
def test_refine_moves_a_perturbed_H0_closer_on_synthetic_courts(pan_deg, seed):
    """Bug 4.11: refine_H returned H0 unchanged (relative finite-difference step at x = 0 -> ~1e-8 px -> zero
    Jacobian). Panned partial views of the synthetic_court.py camera: the refined H must be closer to the truth."""
    from synthetic_court import CameraPath, Scene, SynthData, reprojection_error

    scene, cam = Scene(), CameraPath()
    court.configure(scene.length, scene.width, scene.spec)
    C = np.array(cam.C0, float)
    R = rotation(np.radians(pan_deg), np.radians(cam.tilt_deg), np.radians(cam.roll_deg))
    H = autocourt.court_to_image_R(C, R, cam.f0, scene.size)
    data = SynthData(k=np.arange(1), frame=np.arange(1), t=np.zeros(1), motion=[None], inliers=np.zeros(1),
                     rows=np.zeros((0, 14)), signature={}, C=C[None], R=[R], f=np.array([cam.f0]))
    pts, _ = cs.class_samples(0.1)
    dist = _dist(H, pts)
    rng = np.random.default_rng(seed)
    w, h = SIZE  # perturb in the image (the frame corners move up to 12 px): well defined for any pan
    frame = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
    A = cv2.getPerspectiveTransform(frame, frame + rng.uniform(-12, 12, frame.shape).astype(np.float32))
    H0 = A @ H
    Hr = cs.refine_H(dist, H0, SIZE, pts)
    assert not np.allclose(Hr / Hr[2, 2], H0 / H0[2, 2])  # it moves
    e0 = reprojection_error([H0], data, scene)[0]
    e1 = reprojection_error([Hr], data, scene)[0]
    assert e1 < 0.5 * e0, (e0, e1)


def test_fd_jac_uses_an_absolute_step():
    f = lambda z: np.array([3.0 * z[0], z[1] ** 2])  # noqa: E731
    J = cs._fd_jac(f, np.zeros(2), f(np.zeros(2)), step=0.25)
    assert np.allclose(J, [[3.0, 0.0], [0.0, 0.25]])


def test_refine_keeps_the_visible_side_when_the_origin_is_behind_the_camera():
    """Bug 4.11: with the court origin behind the camera (h33 < 0) the h33 normalisation flipped the sign of H and
    every court point projected 'behind the camera'."""
    court.configure(28.0, 15.0, "FIBA")
    R = rotation(np.radians(60.0), np.radians(14.0), np.radians(0.3))
    H = autocourt.court_to_image_R(np.array([14.0, -6.0, 5.0]), R, 1100.0, SIZE)
    assert H[2, 2] < 0
    pts, _ = cs.class_samples(0.1)
    Hr = cs.refine_H(_dist(H, pts), H, SIZE, pts)
    n_true = np.isfinite(cs.project(H, pts)[:, 0]).sum()
    assert np.isfinite(cs.project(Hr, pts)[:, 0]).sum() == n_true
    assert cs.plausible(Hr, SIZE, pts)[0]


def test_coarse_search_finds_a_tv_camera_from_evidence_alone():
    """Broadcast fallback: no prior, no keypoints - the camera grid + refine must land on the court."""
    court.configure(28.0, 15.0, "FIBA")
    C = np.array([16.0, -20.0, 12.0])
    R = rotation(np.radians(100.0), np.radians(28.0), 0.0)
    H = autocourt.court_to_image_R(C, R, 2200.0, SIZE)
    pts, cls = cs.class_samples(0.3)
    s = cs.WORK_W / SIZE[0]
    p = cs.project(H, cs.class_samples(0.1)[0], s)
    ev = p[np.isfinite(p[:, 0])].astype(np.int16)
    dist, dist_c, dist_c4, ev4 = cs.evidence_maps(ev, SIZE)
    hyps = cs.template_hypotheses(SIZE, families=("tv",))
    assert len(hyps) > 200
    r = cs.solve_frame(dist, SIZE, pts, cls, [], hyps=hyps, dist_c=dist_c, dist_c4=dist_c4, ev4=ev4)
    assert r is not None and r["source"] == "coarse"
    a, b = cs.project(r["H"], pts), cs.project(H, pts)
    vis = np.isfinite(b[:, 0]) & (b[:, 0] >= 0) & (b[:, 0] < SIZE[0]) & (b[:, 1] >= 0) & (b[:, 1] < SIZE[1])
    assert vis.sum() >= 40
    assert np.nanmedian(np.linalg.norm(a[vis] - b[vis], axis=1)) < 6.0


def test_floor_evidence_traces_a_painted_key_and_the_apron():
    """A wood court on a dark apron with a dark painted key and a round logo: the evidence must lie on the court
    outline and the key, not on the apron lettering."""
    court.configure(28.0, 15.0, "FIBA")
    H = cs._H_cam(np.array([9.0, -18.0, 11.0]), np.array([6.0, 7.5, 0.0]), 1500.0, SIZE)  # left key in view
    rng = np.random.default_rng(0)
    img = np.zeros((SIZE[1], SIZE[0], 3), np.uint8)
    img[:] = (90, 30, 20)  # dark blue apron
    poly = lambda p: np.int32(np.round(cs.project(H, np.array(p, float)))).reshape(-1, 1, 2)  # noqa: E731
    cv2.fillPoly(img, [poly([[0, 0], [28, 0], [28, 15], [0, 15]])], (120, 170, 210))  # wood
    img[:] = np.clip(img.astype(int) + rng.integers(-8, 9, img.shape), 0, 255)  # grain
    cv2.fillPoly(img, [poly([[0, 5.05], [5.8, 5.05], [5.8, 9.95], [0, 9.95]])], (90, 30, 20))  # painted key
    cv2.fillPoly(img, [poly([[14 + 2.5 * np.cos(a), 7.5 + 2.5 * np.sin(a)] for a in np.linspace(0, 6.28, 40)])],
                 (40, 40, 160))  # logo
    cv2.putText(img, "FINAL FOUR", tuple(poly([[8, -2.5]])[0, 0]), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 6)
    boxes = np.array([[600, 300, 660, 420]], float)
    cv2.rectangle(img, (600, 300), (660, 420), (255, 255, 255), -1)
    ev, info = cs.floor_evidence(img, boxes)
    assert info["floor_frac"] > 0.1 and len(ev) > 200
    pts, cls = cs.class_samples(0.2)
    dist = cs.dist_from_points(ev, SIZE)
    sc = cs.score_H(dist, H, SIZE, pts, cls)
    assert sc["per_class"]["sideline"][1] > 0.5 and sc["per_class"]["lane"][1] > 0.4
    # the lettering on the apron is not evidence: a court shifted onto it scores badly
    T = np.array([[1, 0, 0], [0, 1, 120], [0, 0, 1.0]])
    assert cs.score_H(dist, T @ H, SIZE, pts, cls)["score"] < 0.3
