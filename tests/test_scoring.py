import numpy as np

from replay import court, homography, scoring
from replay.ball import BallFrame
from replay.config import ScoringConfig
from replay.scoring import BALL_R, GRAVITY, RIM_HEIGHT, camera_from_homography, detect_baskets, rim_in_image

from synthetic import FRAME, court_to_image_matrix, look_at_camera

FPS = 10.0
RIGHT = court.baskets()[1]


def _cam():
    return look_at_camera(eye=(14.0, -9.0, 6.0), target=(20.0, 7.5, 1.0), f=900.0)


def _H(K, R, t):
    return homography.normalize(np.linalg.inv(court_to_image_matrix(K, R, t)), homography.floor_ref_pixel(FRAME))


def _project(K, R, t, X):
    Xc = np.asarray(X, float) @ R.T + t
    uv = (Xc @ K.T)[:, :2] / Xc[:, 2:3]
    return uv, Xc[:, 2]


def test_focal_and_rim_from_floor_homography():
    for eye, target, f in [((14.0, -9.0, 6.0), (20.0, 7.5, 1.0), 900.0), ((10.0, -6.0, 4.0), (5.0, 7.5, 0.0), 1400.0),
                           ((14.0, -12.0, 9.0), (14.0, 7.5, 0.0), 650.0)]:
        K, R, t = look_at_camera(eye, target, f=f)
        cam = camera_from_homography(_H(K, R, t), FRAME)
        assert cam is not None
        assert abs(cam.f - f) / f < 1e-3
        assert np.allclose(cam.centre, eye, atol=1e-2)
        for bxy in court.baskets():
            uv, _ = _project(K, R, t, np.array([[bxy[0], bxy[1], RIM_HEIGHT]]))
            rim = rim_in_image(cam, bxy)
            assert rim is not None and np.allclose(rim[:2], uv[0], atol=0.05)


def test_camera_recovery_fails_cleanly_on_garbage():
    assert camera_from_homography(None, FRAME) is None
    assert camera_from_homography(np.eye(3), FRAME) is None  # orthographic-looking map: no real focal length


def _shot_positions(times, release, target, flight_s, post="through"):
    """3D ball centre: ballistic from ``release`` (time 0) to ``target`` (time flight_s), then ``post``."""
    release, target = np.asarray(release, float), np.asarray(target, float)
    v0 = (target - release + np.array([0, 0, 0.5 * GRAVITY * flight_s ** 2])) / flight_s
    out = []
    for tau in times:
        if tau <= flight_s or post == "free":
            out.append(release + v0 * tau - np.array([0, 0, 0.5 * GRAVITY * tau ** 2]))
        elif post == "through":  # the net kills horizontal speed: straight down under the ring
            d = tau - flight_s
            vz = v0[2] - GRAVITY * flight_s
            out.append(np.array([target[0], target[1], target[2] + vz * d * 0.6 - 0.5 * GRAVITY * d * d]))
        elif post == "bounce":  # off the front of the rim, up and back towards the shooter
            d = tau - flight_s
            out.append(target + np.array([-2.5, -0.3, 3.0]) * d - np.array([0, 0, 0.5 * GRAVITY * d * d]))
    return np.array(out)


def _scene(shots, n=80, owner_team=None):
    """shots: list of (release time, release xyz, target xyz, flight s, post). Returns detect_baskets inputs."""
    K, R, t = _cam()
    H = _H(K, R, t)
    times = np.arange(n) / FPS + 0.03
    rows = [np.zeros((0, 5)) for _ in range(n)]
    frames = [BallFrame(i, None, 0.0) for i in range(n)]
    for t0, release, target, flight, post in shots:
        for i in range(n):
            tau = times[i] - t0
            if tau < -1.0 or tau > flight + 0.6:
                continue
            if tau < 0:  # held by the shooter (track 7)
                X = np.asarray(release, float)
            else:
                X = _shot_positions([tau], release, target, flight, post)[0]
            if X[2] < BALL_R:
                continue
            uv, z = _project(K, R, t, X[None])
            r = 900.0 * BALL_R / z[0]
            box = (uv[0, 0] - r, uv[0, 1] - r, uv[0, 0] + r, uv[0, 1] + r)
            rows[i] = np.vstack([rows[i], [*box, 0.7]])
            frames[i] = BallFrame(i, box, 0.7)
            if tau < 0:
                frames[i].state, frames[i].owner = "possessed", 7
    team = owner_team or {7: "B"}
    return dict(H=[H] * n, t=times, frame=np.arange(n) * 3, ball_rows=rows, ball_frames=frames, track_team=team,
                frame_size=FRAME, fps=FPS)


RIM3 = (RIGHT[0], RIGHT[1], RIM_HEIGHT)


def test_made_shot_gives_one_event():
    sc = _scene([(2.0, (20.5, 6.0, 2.3), RIM3, 1.0, "through")])
    ev = detect_baskets(**sc)
    assert len(ev) == 1, ev
    e = ev[0]
    assert e["basket"] == "R" and e["team"] == "B" and e["type"] == "basket" and e["source"] == "auto"
    assert e["details"]["shooter_track"] == 7
    assert 3.0 <= e["t"] <= 3.3  # rim reached at t = 3.0 s
    assert e["id"] == f"basket-{e['frame']}-R"
    assert 0.5 <= e["confidence"] <= 1.0
    assert e["details"]["ballistic"]["ok"] and e["details"]["ballistic"]["miss_m"] < 0.2


def test_made_shot_without_tracked_flight_still_detected():
    """Only a few detections near the rim (none in the flight): the 2D test alone decides."""
    sc = _scene([(2.0, (20.5, 6.0, 2.3), RIM3, 1.0, "through")])
    for i, tt in enumerate(sc["t"]):
        if not 2.75 <= tt <= 3.4:
            sc["ball_rows"][i] = np.zeros((0, 5))
    sc["ball_frames"] = [BallFrame(i, None, 0.0) for i in range(len(sc["t"]))]
    ev = detect_baskets(**sc)
    assert len(ev) == 1 and ev[0]["team"] is None and ev[0]["details"]["ballistic"] is None


def test_misses_give_no_event():
    beside = (RIGHT[0] - 0.9, RIGHT[1], RIM_HEIGHT)  # short, 0.9 m in front of the ring along the court
    in_front = (RIGHT[0], RIGHT[1] - 2.0, RIM_HEIGHT)  # 2 m nearer the camera: lines up with the ring in 2D
    for target, post in [(beside, "free"), (in_front, "free"), (RIM3[:2] + (RIM_HEIGHT + 0.05,), "bounce")]:
        tgt = target if post != "bounce" else (RIGHT[0] - 0.25, RIGHT[1], RIM_HEIGHT + 0.05)
        sc = _scene([(2.0, (20.5, 6.0, 2.3), tgt, 1.0, post)])
        assert detect_baskets(**sc) == [], (target, post)


def test_steep_miss_in_front_or_behind_is_vetoed_by_the_3d_fit(monkeypatch):
    """A high, slow arc 1 m nearer/further than the ring lines up with it in the image: 2D alone accepts it,
    the ballistic fit (gravity fixes the depth) rejects it, and the same arc through the ring is kept."""
    for dy, expected in [(-1.0, 0), (1.0, 0), (0.0, 1)]:
        sc = _scene([(2.0, (24.5, RIGHT[1] + dy, 2.3), (RIGHT[0], RIGHT[1] + dy, RIM_HEIGHT), 1.4, "free")])
        # 2D only: the arc through the ring is accepted; a 1 m miss may already fail the "through the ring"
        # horizontal check below the rim (scoring.THROUGH_DX_R), else the 3D fit has to reject it
        n2d = len(detect_baskets(**sc, cfg=ScoringConfig(ballistic=False, min_confidence=0.3)))
        assert n2d == 1 if expected else n2d <= 1
        assert len(detect_baskets(**sc)) == expected, dy  # default: 2D 'through the ring' checks
        monkeypatch.setattr(scoring, "BALLISTIC_VETO", True)  # optional 3D veto (exact geometry only)
        assert len(detect_baskets(**sc)) == expected, dy
        monkeypatch.setattr(scoring, "BALLISTIC_VETO", False)


def test_debounce_one_event_per_rim_window():
    shot = ((20.5, 6.0, 2.3), RIM3, 1.0, "through")
    sc = _scene([(2.0, *shot), (3.5, *shot)])  # tip-in style second "make" 1.5 s later
    assert len(detect_baskets(**sc)) == 1
    sc = _scene([(1.0, *shot), (5.0, *shot)])
    ev = detect_baskets(**sc)
    assert len(ev) == 2 and ev[1]["t"] - ev[0]["t"] > 3.0
    assert len(detect_baskets(**sc, cfg=ScoringConfig(debounce_s=5.0))) == 1


def test_disabled_and_uncalibrated():
    sc = _scene([(2.0, (20.5, 6.0, 2.3), RIM3, 1.0, "through")])
    assert detect_baskets(**sc, cfg=ScoringConfig(enabled=False)) == []
    sc["H"] = [None] * len(sc["H"])
    assert detect_baskets(**sc) == []
