import numpy as np

from replay import footpoint, homography

from synthetic import FRAME, court_to_image_matrix, look_at_camera

FPS = 10.0


def _on(**kw):
    return footpoint.FootConfig(enabled=True, **kw)


def _cam(height):
    K, R, t = look_at_camera(eye=(14.0, -2.0, height), target=(14.0, 7.5, 0.0), f=900.0)
    H = homography.normalize(np.linalg.inv(court_to_image_matrix(K, R, t)), homography.floor_ref_pixel(FRAME))
    return K, R, t, H


def _px(K, R, t, X):
    q = (K @ (R @ np.asarray(X, float).T + t[:, None])).T
    return q[:, :2] / q[:, 2:3]


def _box(K, R, t, xy, z0=0.0, height=1.9, width=0.6):
    """Image box of an upright person standing (feet at z0) at court xy."""
    foot = _px(K, R, t, [[xy[0], xy[1], z0]])[0]
    head = _px(K, R, t, [[xy[0], xy[1], z0 + height]])[0]
    side = _px(K, R, t, [[xy[0] + width / 2, xy[1], z0]])[0]
    hw = abs(side[0] - foot[0])
    return np.array([foot[0] - hw, head[1], foot[0] + hw, foot[1]])


def _walk(cam, n=40, jump=None, start=(10.0, 6.0), vel=(0.8, 0.0)):
    K, R, t, H = cam
    xy = np.array(start) + np.arange(n)[:, None] * np.array(vel) / FPS
    z = np.zeros(n)
    if jump is not None:
        a, b, hmax = jump
        u = np.linspace(0, np.pi, b - a + 2)[1:-1]
        z[a:b] = hmax * np.sin(u)
    boxes = np.array([_box(K, R, t, p, zz) for p, zz in zip(xy, z)])
    return xy, boxes, np.repeat(H[None], n, axis=0)


def test_standing_player_is_unchanged():
    cam = _cam(6.0)
    xy, boxes, Hs = _walk(cam)
    r = footpoint.estimate_feet(boxes, Hs, np.arange(len(xy)), np.ones(len(xy), int), FPS, FRAME, cfg=_on())
    np.testing.assert_allclose(r.xy, xy, atol=1e-6)
    assert (r.flag == footpoint.BOTTOM).all()


def test_jump_is_replaced_by_ground_track_low_camera():
    cam = _cam(2.0)  # camera at head height: a jump throws the projected foot metres away
    xy, boxes, Hs = _walk(cam, jump=(15, 21, 0.45))
    plain = homography.project_each(Hs, np.c_[(boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]][:, None])[:, 0]
    assert np.abs(plain - xy).max() > 1.0
    r = footpoint.estimate_feet(boxes, Hs, np.arange(len(xy)), np.ones(len(xy), int), FPS, FRAME, cfg=_on())
    assert (r.flag[15:21] == footpoint.AIRBORNE).sum() >= 4
    assert np.abs(r.xy - xy).max() < 0.3
    # the replaced samples are less certain than the grounded ones
    assert np.trace(r.R[17]) > np.trace(r.R[5])


def test_cut_box_recovered_only_when_well_conditioned():
    for height, expect in ((8.0, True), (2.0, False)):
        cam = _cam(height)
        xy, boxes, Hs = _walk(cam, start=(9.0, 5.0))
        cut = np.zeros(len(xy), bool)
        cut[20:23] = True
        r = footpoint.estimate_feet(boxes, Hs, np.arange(len(xy)), np.ones(len(xy), int), FPS, FRAME, cut=cut,
                                    cfg=_on(recover_cut=True))
        if expect:
            assert (r.flag[20:23] == footpoint.FROM_HEAD).all()
            assert np.abs(r.xy[20:23] - xy[20:23]).max() < 0.3
        else:  # head at the horizon: no depth information, the foot stays unknown
            assert np.isnan(r.xy[20:23]).all() or np.abs(r.xy[20:23] - xy[20:23]).max() < 0.5


def test_glitch_short_box_fixed_from_head():
    cam = _cam(8.0)
    xy, boxes, Hs = _walk(cam, start=(9.0, 5.0))
    boxes[25, 3] -= 0.35 * (boxes[25, 3] - boxes[25, 1])  # detector cut the legs off in one sample
    r = footpoint.estimate_feet(boxes, Hs, np.arange(len(xy)), np.ones(len(xy), int), FPS, FRAME, cfg=_on())
    assert r.flag[25] == footpoint.GLITCH
    assert np.linalg.norm(r.xy[25] - xy[25]) < 0.3


def test_uncertainty_grows_with_distance():
    K, R, t, H = cam = _cam(6.0)
    near = _box(K, R, t, (14.0, 2.0))
    far = _box(K, R, t, (14.0, 13.0))
    r = footpoint.estimate_feet(np.array([near, far]), np.repeat(H[None], 2, axis=0), np.array([0, 0]),
                                np.array([0, 0]), FPS, FRAME, cfg=_on())
    floor = 2 * 0.15 ** 2  # minimum isotropic variance
    assert np.trace(r.R[1]) - floor > 3 * (np.trace(r.R[0]) - floor)


def test_disabled_is_plain_projection():
    cam = _cam(2.0)
    xy, boxes, Hs = _walk(cam, jump=(15, 21, 0.45))
    r = footpoint.estimate_feet(boxes, Hs, np.arange(len(xy)), np.ones(len(xy), int), FPS, FRAME,
                                cfg=footpoint.FootConfig(enabled=False))
    plain = homography.project_each(Hs, np.c_[(boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]][:, None])[:, 0]
    np.testing.assert_allclose(r.xy, plain)


def test_bottom_glitch_removed_with_camera_motion():
    """Panning camera (pixel shift per sample): a one-sample 'legs cut' box is caught by the stabilised Hampel test,
    a jump (both box ends move) is not."""
    cam = _cam(2.0)
    xy, boxes, Hs = _walk(cam, start=(9.0, 5.0))
    n = len(xy)
    shift = np.array([[1.0, 0, 3.0], [0, 1.0, 1.0], [0, 0, 1.0]])  # sample i-1 -> i pixels
    Hs = Hs.copy()
    for i in range(n):  # the camera pans: pixels move, the homography follows
        T = np.linalg.matrix_power(shift, i)
        boxes[i] = np.r_[(T @ [boxes[i, 0], boxes[i, 1], 1])[:2], (T @ [boxes[i, 2], boxes[i, 3], 1])[:2]]
        Hs[i] = Hs[i] @ np.linalg.inv(T)
    motion = [None] + [shift] * (n - 1)
    boxes[25, 3] -= 0.4 * (boxes[25, 3] - boxes[25, 1])  # legs hidden for one sample
    r = footpoint.estimate_feet(boxes, Hs, np.arange(n), np.ones(n, int), FPS, FRAME, motion=motion, cfg=_on())
    assert r.flag[25] == footpoint.GLITCH
    assert np.linalg.norm(r.xy[25] - xy[25]) < 0.3
    assert (r.flag[np.r_[0:24, 27:n]] == footpoint.BOTTOM).all()
    assert r.R[25].trace() > r.R[24].trace()


def test_project_people_matches_old_behaviour_when_disabled():
    from replay.config import Config

    cam = _cam(6.0)
    xy, boxes, Hs = _walk(cam, n=10)
    rows = np.zeros((10, 14))
    rows[:, 0] = np.arange(10)
    rows[:, 1:5] = boxes
    rows[:, 5] = 0.9
    rows[:, 7] = 1
    cfg = Config()
    cfg.foot.enabled = False
    out = footpoint.project_people(rows, np.arange(10), np.arange(10), list(Hs), FRAME, np.zeros((0, 4)), cfg, FPS)
    np.testing.assert_allclose(out[2], xy, atol=1e-6)
    cfg.foot.enabled = True
    out2 = footpoint.project_people(rows, np.arange(10), np.arange(10), list(Hs), FRAME, np.zeros((0, 4)), cfg, FPS,
                                    [None] + [np.eye(3)] * 9)
    np.testing.assert_allclose(out2[2], xy, atol=1e-6)


def test_membership_identical_when_enabled_with_keep_old_membership():
    """Task 10 invariant: robust feet change positions only - the kept detection set equals the plain one."""
    from replay.config import Config

    cam = _cam(2.0)
    xy, boxes, Hs = _walk(cam, n=30, jump=(10, 16, 0.45))
    boxes[20, 3] -= 0.4 * (boxes[20, 3] - boxes[20, 1])
    boxes[25] = [600, 50, 640, 120]  # a box whose plain foot projects far off court
    rows = np.zeros((30, 14))
    rows[:, 0] = np.arange(30)
    rows[:, 1:5] = boxes
    rows[:, 5] = 0.9
    rows[:, 7] = 1
    cfg = Config()
    out = {}
    for on in (False, True):
        cfg.foot.enabled = on
        out[on] = footpoint.project_people(rows, np.arange(30), np.arange(30), list(Hs), FRAME, np.zeros((0, 4)),
                                           cfg, FPS, [None] + [np.eye(3)] * 29)
    np.testing.assert_array_equal(out[False][0], out[True][0])
    assert (out[True][4] != footpoint.BOTTOM).any()  # but some positions were repaired
