import numpy as np
import pytest

from replay import court, homography
from replay.calibration import CalibrationFile, Keyframe, resolve
from replay.config import CalibrationConfig

from synthetic import FRAME, court_to_image_matrix, project_court, sideline_camera


def _truth():
    K, R, t = sideline_camera()
    G = court_to_image_matrix(K, R, t)
    return K, R, t, homography.normalize(np.linalg.inv(G), homography.floor_ref_pixel(FRAME))


def test_fit_from_four_points_is_exact():
    K, R, t, _ = _truth()
    names = ["L_corner_near", "R_corner_near", "R_corner_far", "L_corner_far"]
    crt = np.array([court.KEYPOINTS[n] for n in names])
    img = project_court(K, R, t, crt)
    H, mask = homography.fit(img, crt, FRAME)
    assert mask.all()
    probe = np.array([[3.0, 4.0], [14.0, 7.5], [25.0, 12.0], [5.8, 5.05]])
    np.testing.assert_allclose(homography.project(H, project_court(K, R, t, probe)), probe, atol=1e-6)
    np.testing.assert_allclose(homography.reprojection_error_px(H, img, crt), 0.0, atol=1e-3)


def test_fit_with_noise_and_many_points():
    K, R, t, _ = _truth()
    crt = np.array(list(court.KEYPOINTS.values()))
    rng = np.random.default_rng(0)
    img = project_court(K, R, t, crt) + rng.normal(0, 1.0, (len(crt), 2))
    H, _ = homography.fit(img, crt, FRAME)
    err = homography.project(H, project_court(K, R, t, crt)) - crt
    assert np.abs(err).max() < 0.25  # metres, with 1 px click noise
    assert homography.reprojection_error_px(H, img, crt).mean() < 2.0


def test_ransac_rejects_a_misclicked_point():
    K, R, t, _ = _truth()
    crt = np.array(list(court.KEYPOINTS.values()))
    img = project_court(K, R, t, crt)
    img[3] += [60, -40]  # one wrong click
    H, mask = homography.fit(img, crt, FRAME, ransac_px=3.0)
    assert not mask[3] and mask.sum() == len(crt) - 1
    np.testing.assert_allclose(homography.project(H, img[mask]), crt[mask], atol=1e-4)


def test_too_few_points():
    with pytest.raises(ValueError):
        homography.fit(np.zeros((3, 2)), np.zeros((3, 2)), FRAME)


def test_court_to_image_roundtrip():
    K, R, t, H = _truth()
    crt = np.array([[0.0, 0.0], [28.0, 15.0], [14.0, 7.5]])
    np.testing.assert_allclose(homography.court_to_image(H, crt), project_court(K, R, t, crt), atol=1e-6)


def test_points_above_horizon_are_invalid():
    _, _, _, H = _truth()
    # top-left pixel of a sideline camera looking down at 20 degrees sees the wall/sky, not the floor
    sky = homography.project(H, np.array([[640.0, 0.0]]))
    floor = homography.project(H, np.array([[640.0, 700.0]]))
    assert np.isnan(sky).all() or not court.in_court(sky, 50)[0]
    assert np.isfinite(floor).all()


def test_normalize_sign_invariance():
    _, _, _, H = _truth()
    ref = homography.floor_ref_pixel(FRAME)
    np.testing.assert_allclose(homography.normalize(-3.0 * H, ref), homography.normalize(H, ref))


def test_jacobian_matches_finite_difference_scale():
    _, _, _, H = _truth()
    J = homography.jacobian(H, (640.0, 600.0))
    p0 = homography.project(H, np.array([[640.0, 600.0]]))[0]
    p1 = homography.project(H, np.array([[650.0, 600.0]]))[0]
    np.testing.assert_allclose(J @ [10.0, 0.0], p1 - p0, rtol=1e-2, atol=1e-4)
    # far court (higher in the image) must be coarser: more metres per pixel
    J_far = homography.jacobian(H, (640.0, 300.0))
    assert abs(np.linalg.det(J_far)) > abs(np.linalg.det(J))


def test_blend_of_identical_is_identity():
    _, _, _, H = _truth()
    B = homography.blend([H, H], [0.3, 0.7], FRAME)
    pts = np.array([[200.0, 600.0], [1000.0, 500.0]])
    np.testing.assert_allclose(homography.project(B, pts), homography.project(H, pts), atol=1e-6)


def test_keyframe_propagation_through_camera_pan():
    """Pan the synthetic camera; motion homographies must carry a keyframe to later frames."""
    from synthetic import look_at_camera

    targets = [(10.0 + 0.5 * i, 7.5, 0.0) for i in range(9)]
    cams = [look_at_camera((14.0, -9.0, 6.0), tg, f=700.0) for tg in targets]
    G = [court_to_image_matrix(*c) for c in cams]  # court -> image_i
    motions = [None] + [G[i] @ np.linalg.inv(G[i - 1]) for i in range(1, len(G))]  # image_{i-1} -> image_i
    times = np.arange(len(G)) / 10.0

    names = ["L_paint_base_near", "L_paint_base_far", "L_ft_near", "L_ft_far", "mid_near", "center_circle_near"]
    crt = np.array([court.KEYPOINTS[n] for n in names])
    img0 = project_court(*cams[0], crt)
    calib = CalibrationFile(video="synthetic", frame_size=FRAME,
                            keyframes=[Keyframe(0.0, {n: tuple(p) for n, p in zip(names, img0)})])
    calib.keyframes[0].fit(FRAME)
    res = resolve(times, motions, calib, CalibrationConfig())
    assert res.quality[0] == "keyframe" and all(q == "propagated" for q in res.quality[1:])
    probe = np.array([[8.0, 3.0], [12.0, 9.0]])
    for i in range(len(G)):
        np.testing.assert_allclose(homography.project(res.H[i], project_court(*cams[i], probe)), probe, atol=5e-2)

    # a broken motion link must stop propagation instead of guessing
    motions[4] = None
    res = resolve(times, motions, calib, CalibrationConfig())
    assert res.H[4] is None and res.quality[4] == "none"


def test_fixed_camera_uses_single_keyframe_everywhere():
    K, R, t, _ = _truth()
    names = ["L_corner_near", "R_corner_near", "R_corner_far", "L_corner_far"]
    img = project_court(K, R, t, np.array([court.KEYPOINTS[n] for n in names]))
    kf = Keyframe(500.0, {n: tuple(p) for n, p in zip(names, img)})  # outside the processed range
    kf.fit(FRAME)
    calib = CalibrationFile("synthetic", FRAME, fixed_camera=True, keyframes=[kf])
    res = resolve(np.arange(5) / 10.0, [None] * 5, calib, CalibrationConfig())
    assert all(H is not None for H in res.H)


def test_camera_from_homography_recovers_position_and_heights():
    from synthetic import look_at_camera

    for eye, f in (((14.0, -9.0, 6.0), 700.0), ((13.0, -1.8, 2.0), 900.0), ((4.0, -3.0, 3.5), 1200.0)):
        K, R, t = look_at_camera(eye, (14.0, 7.5, 0.0), f=f)
        H = homography.normalize(np.linalg.inv(court_to_image_matrix(K, R, t)), homography.floor_ref_pixel(FRAME))
        cam = homography.camera_from_homography(H, FRAME)
        assert cam is not None
        np.testing.assert_allclose(cam["C"], eye, atol=1e-6)
        assert abs(cam["f"] - f) < 1e-3
        X = np.array([10.0, 5.0, 1.9, 1.0])  # a head 1.9 m above the floor
        q, q_true = cam["P"] @ X, K @ (R @ X[:3] + t)
        np.testing.assert_allclose(q[:2] / q[2], q_true[:2] / q_true[2], atol=1e-6)
