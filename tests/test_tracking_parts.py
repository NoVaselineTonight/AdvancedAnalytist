import numpy as np

from replay import court, homography
from replay.ball import POSSESSED, ESTIMATED, ball_court_positions, track_ball
from replay.calibration import CalibrationFile, Keyframe, resolve
from replay.config import BallConfig, CalibrationConfig, CourtTrackConfig
from replay.court_tracker import track_on_court

from synthetic import FRAME, court_to_image_matrix, look_at_camera, project_court

FPS = 10.0


def _dets(paths, hidden=None):
    """paths: {player: [(x, y) per frame]}; hidden: {player: range of frames not detected}."""
    k, xy, feat, bt = [], [], [], []
    for pid, pts in paths.items():
        for f, p in enumerate(pts):
            if hidden and f in hidden.get(pid, ()):
                continue
            k.append(f)
            xy.append(p)
            feat.append([40, 30 if pid % 2 else -20, 20, 40, 30, 20])
            bt.append(pid * 100 + (0 if f < 30 else 1))  # first-stage ID changes after the occlusion
    n = len(k)
    return (np.array(k), np.array(xy, float), np.tile(np.eye(2) * 0.04, (n, 1, 1)), np.array(feat, float),
            np.array(bt), np.full(n, 0.9))


def test_player_keeps_id_through_occlusion():
    frames = 60
    p1 = [(5 + 0.1 * f, 7.0) for f in range(frames)]  # walks right
    p2 = [(10.0, 4 + 0.05 * f) for f in range(frames)]
    k, xy, R, feat, bt, conf = _dets({1: p1, 2: p2}, hidden={1: range(20, 40)})  # 2 s unseen
    ids = track_on_court(k, xy, R, feat, bt, conf, FPS, CourtTrackConfig())
    is_p1 = feat[:, 1] > 0  # player 1 has the odd (positive a*) kit in _dets
    ids1, ids2 = set(ids[is_p1]), set(ids[~is_p1])
    assert len(ids1) == 1, f"player 1 split into {ids1}"
    assert len(ids2) == 1 and ids1.isdisjoint(ids2)


def test_ball_follows_holder_and_estimates_flight():
    n = 40
    rows = [np.zeros((0, 5)) for _ in range(n)]
    G = [np.eye(3)] * n
    seg = np.zeros(n, int)
    players_at = [[] for _ in range(n)]
    for i in range(n):
        if i < 10 or i >= 20:  # held: ball inside the player's box; frames 10..19 in the air (not detected)
            holder_box = np.array([100.0 + i * 5, 100, 140 + i * 5, 220])
            rows[i] = np.array([[115 + i * 5, 150, 125 + i * 5, 160, 0.8]])
            players_at[i] = [(7 if i < 10 else 9, holder_box, np.array([5.0 if i < 10 else 9.0, 7.0]))]
    boxes = track_ball(n, rows, G, seg, FPS, BallConfig())
    frames = ball_court_positions(boxes, players_at, FPS, BallConfig())
    assert frames[5].state == POSSESSED and frames[5].owner == 7
    assert frames[25].state == POSSESSED and frames[25].owner == 9
    mid = frames[15]
    assert mid.state == ESTIMATED and 5.0 < mid.x < 9.0


def test_static_ball_is_ignored():
    n = 80
    rows = [np.array([[500, 600, 510, 610, 0.9]]) for _ in range(n)]  # a spare ball lying still all game
    boxes = track_ball(n, rows, [np.eye(3)] * n, np.zeros(n, int), FPS, BallConfig())
    assert all(b is None for b in boxes[40:60])


def test_keyframes_combine_across_a_pan():
    """Two keyframes with only 4 noisy points each, on different halves: the joint fit beats either alone."""
    targets = [(6.0 + 2.0 * i, 7.5, 0.0) for i in range(9)]  # pan from left to right half
    cams = [look_at_camera((14.0, -9.0, 6.0), tg, f=700.0) for tg in targets]
    Gc = [court_to_image_matrix(*c) for c in cams]
    motions = [None] + [Gc[i] @ np.linalg.inv(Gc[i - 1]) for i in range(1, len(Gc))]
    times = np.arange(len(Gc)) / 10.0
    rng = np.random.default_rng(1)

    def kf(i, names):
        img = project_court(*cams[i], np.array([court.KEYPOINTS[n] for n in names])) + rng.normal(0, 2.0, (4, 2))
        return Keyframe(times[i], {n: tuple(p) for n, p in zip(names, img)})

    left = kf(0, ["L_paint_base_near", "L_paint_base_far", "L_ft_near", "L_ft_far"])
    right = kf(8, ["R_paint_base_near", "R_paint_base_far", "R_ft_near", "R_ft_far"])
    probe = np.array([[3.0, 2.0], [14.0, 7.5], [25.0, 13.0]])

    def err(keyframes, i):
        res = resolve(times, motions, CalibrationFile("s", FRAME, keyframes=keyframes), CalibrationConfig())
        return np.abs(homography.project(res.H[i], project_court(*cams[i], probe)) - probe).max()

    joint = max(err([left, right], i) for i in (0, 4, 8))
    alone = max(err([left], i) for i in (0, 4, 8))
    assert joint < alone
    assert joint < 0.5  # metres, from 2 px click noise


def test_motion_dropout_uses_last_rate():
    """Task 6: a failed estimate gives the tracker the last good pan rate for <= fallback_samples samples."""
    import cv2

    from replay.camera import MotionEstimator
    from replay.config import MotionConfig

    est = MotionEstimator(MotionConfig(min_inliers=40, fallback_samples=2), (640, 360), [])
    rng = np.random.default_rng(1)
    bg = cv2.GaussianBlur(rng.integers(0, 255, (360, 640, 3), np.uint8), (5, 5), 0)

    def step(img):
        M, _ = est.update(img, np.zeros((0, 4)))
        return M, est.fallback(M)

    assert step(bg) == (None, None)  # first frame: chain start, no fallback
    M1, A1 = step(np.roll(bg, 6, axis=1))
    assert M1 is not None and abs(M1[0, 2] - 6) < 1.5 and A1 is None
    blank = np.full_like(bg, 128)  # no texture: the estimate fails
    M2, A2 = step(blank)
    assert M2 is None and A2 is not None and abs(A2[0, 2] - 6) < 1.5
    assert step(blank)[1] is not None
    assert step(blank)[1] is None  # fallback_samples exhausted
    off = MotionEstimator(MotionConfig(min_inliers=40), (640, 360), [])  # default: off
    for img in (bg, np.roll(bg, 6, axis=1), blank):
        M, _ = off.update(img, np.zeros((0, 4)))
        assert off.fallback(M) is None


def _sprint_ids(cfg):
    from replay.detection import Detections
    from replay.tracking import CMCByteTrack

    tr = CMCByteTrack(cfg, 10.0)
    ids, x = [], 100.0
    for k in range(40):
        x += 0.0 if k < 10 else min(40.0, 20.0 * (k - 9))  # stands 1 s, then sprints: 40 px = 0.8 box widths per sample
        box = np.array([[x, 300.0, x + 50.0, 440.0]], np.float32)
        ids.append(int(tr.update(Detections(box, np.array([0.9], np.float32), np.zeros(1, np.int32)))[0]))
    return ids


def test_sprinting_player_keeps_bytetrack_id_at_10fps():
    """Task 7: 30-fps Kalman noise at 10 fps loses a sprinting player; scaled noise + expanded boxes keep the id."""
    from replay.config import TrackingConfig

    ids = _sprint_ids(TrackingConfig(kalman_fps_scale=True, box_expand=0.2))
    first = next(i for i in ids if i >= 0)
    assert all(i == first for i in ids[3:]), ids
    old = _sprint_ids(TrackingConfig())  # default (off) = previous behaviour
    assert len({i for i in old[3:] if i >= 0}) > 1 or any(i < 0 for i in old[3:])


def test_pass1_switches_off_keep_the_run_signature():
    from replay.config import Config
    from replay.pipeline import _without_off

    cfg = Config()
    assert "kalman_fps_scale" not in _without_off(cfg.to_dict()["tracking"])
    assert "fallback_samples" not in _without_off(cfg.to_dict()["motion"])
    cfg.tracking.kalman_fps_scale = True
    assert _without_off(cfg.to_dict()["tracking"])["kalman_fps_scale"] is True
