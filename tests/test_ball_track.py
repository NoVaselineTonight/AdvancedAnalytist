import numpy as np

from replay.ball import ESTIMATED, POSSESSED, ball_court_positions, estimate_ball
from replay.ball_track import link_ball
from replay.config import BallConfig

FPS = 10.0


def _box(cx, cy, s=12.0, conf=0.8):
    return [cx - s / 2, cy - s / 2, cx + s / 2, cy + s / 2, conf]


def _rows(n, pts, extra=None):
    rows = [np.zeros((0, 5)) for _ in range(n)]
    for i, p in pts.items():
        rows[i] = np.array([_box(*p)])
    for i, b in (extra or {}).items():
        rows[i] = np.vstack([rows[i], [b]]) if len(rows[i]) else np.array([b])
    return rows


def test_isolated_false_positive_is_skipped():
    n = 30
    pts = {i: (200 + 8 * i, 300.0) for i in range(n)}
    del pts[15]
    rows = _rows(n, pts, extra={15: _box(1000, 100, conf=0.9), 16: _box(1000, 100, conf=0.9)})
    tr = link_ball(n, rows, [np.eye(3)] * n, np.zeros(n, int), FPS, BallConfig())
    assert tr.observed.sum() == n - 1
    assert abs(tr.img[16, 0] - (200 + 8 * 16)) < 1  # the real ball, not the far one
    assert abs(tr.img[15, 0] - (200 + 8 * 15)) < 2  # gap filled in the image
    assert not tr.observed[15]


def test_gap_follows_parabolic_arc():
    n = 20
    true = {i: (100 + 20 * i, 400 - 60 * i + 4 * i * i) for i in range(n)}
    pts = {i: p for i, p in true.items() if not 7 <= i <= 10}
    tr = link_ball(n, _rows(n, pts), [np.eye(3)] * n, np.zeros(n, int), FPS, BallConfig())
    for i in range(7, 11):
        assert np.linalg.norm(tr.img[i] - np.array(true[i])) < 3.0


def test_linking_and_static_removal_survive_a_long_pan():
    """Camera pans 25 px/frame (300 frames = 7500 px). A spare ball lying still in the gym must be dropped,
    the moving game ball kept - also far from the segment's first frame."""
    n = 300
    G = [np.array([[1.0, 0, 25.0 * i], [0, 1, 0], [0, 0, 1]]) for i in range(n)]  # frame px -> first-frame px
    rows = [np.zeros((0, 5)) for _ in range(n)]
    for i in range(n):
        world_ball = 600 + 27.0 * i + 30 * np.sin(i / 5)  # moves with (slightly faster than) the camera
        b = [_box(world_ball - 25.0 * i, 350.0)]
        spare_x = 3000.0 - 25.0 * i  # static in the gym: drifts through the image
        if 0 <= spare_x <= 1280:
            b.append(_box(spare_x, 500.0, conf=0.95))
        rows[i] = np.array(b)
    tr = link_ball(n, rows, G, np.zeros(n, int), FPS, BallConfig())
    assert tr.observed.mean() > 0.95
    assert np.all(np.abs(tr.img[tr.observed, 1] - 350.0) < 1)  # never the spare ball (y = 500)


def test_hidden_ball_stays_with_the_same_holder():
    n = 30
    box = np.array([400.0, 200, 460, 400])  # 200 px tall -> ball ~32 px
    pts = {i: (430.0, 300.0) for i in range(n) if not 8 <= i <= 20}
    rows = [np.zeros((0, 5)) for _ in range(n)]
    for i, p in pts.items():
        rows[i] = np.array([_box(*p, s=32)])
    players = [[(5, box, np.array([10.0 + 0.1 * i, 7.0]))] for i in range(n)]
    track_xy = {5: {i: np.array([10.0 + 0.1 * i, 7.0]) for i in range(n)}}
    fr = estimate_ball(n, rows, [np.eye(3)] * n, np.zeros(n, int), FPS, BallConfig(), players, track_xy)
    assert all(f.state == POSSESSED and f.owner == 5 for f in fr)
    assert abs(fr[14].x - 11.4) < 1e-6  # follows the holder while hidden


def test_depth_gate_rejects_far_person():
    """Ball centre inside a small, far person's box, but far too big for that depth: not theirs."""
    n = 5
    rows = [np.array([_box(500, 300, s=40)]) for _ in range(n)]
    far = (8, np.array([480.0, 250, 520, 330]), np.array([10.0, 14.0]))  # 80 px tall: ball would be ~13 px
    fr = ball_court_positions([tuple(r[0]) for r in rows], [[far]] * n, FPS, BallConfig())
    assert all(f.owner is None and f.x is None for f in fr)


def test_one_frame_holder_blip_far_away_is_dropped():
    n = 20
    rows, players = [], []
    near = np.array([400.0, 200, 460, 400])
    other = np.array([700.0, 200, 760, 400])
    for i in range(n):
        bx = 730.0 if i == 10 else 430.0  # one wrong detection on another player 12 m away
        rows.append((bx - 16, 284, bx + 16, 316, 0.8))
        players.append([(1, near, np.array([5.0, 5.0])), (2, other, np.array([17.0, 5.0]))])
    fr = ball_court_positions(rows, players, FPS, BallConfig(), track_xy={1: {i: np.array([5.0, 5.0]) for i in range(n)}})
    assert all(f.x is not None and abs(f.x - 5.0) < 1e-6 for f in fr)
    assert fr[10].owner == 1


def test_rim_anchor_and_flight_from_shooter():
    n = 20
    shooter = np.array([400.0, 200, 460, 400])
    rows, players = [], []
    rims = {"L": [None] * n, "R": [(900.0, 100.0, 10.0)] * n}
    for i in range(n):
        if i < 5:
            rows.append((414, 284, 446, 316, 0.8))
            players.append([(3, shooter, np.array([20.0, 7.5]))])
        elif i == 12:
            rows.append((893, 73, 907, 87, 0.8))  # just above the rim
            players.append([])
        else:
            rows.append(None)
            players.append([])
    fr = ball_court_positions(rows, players, FPS, BallConfig(), rims=rims)
    assert fr[12].source == "rim" and fr[12].state == ESTIMATED
    from replay import court
    assert abs(fr[12].x - court.baskets()[1][0]) < 1e-9
    xs = [fr[i].x for i in range(4, 13)]
    assert all(x is not None for x in xs) and xs == sorted(xs)  # monotone flight towards the basket


def test_ball_never_vanishes_long_gap_is_guessed():
    """Two sightings 20 s apart (beyond max_fill_s): every frame in between has a GUESSED position - with the
    last holder first, then moving to the next one - and nothing before/after the sightings is left empty."""
    from replay.ball import GUESSED
    n = 220
    boxes = [None] * n
    boxes[10] = (500.0, 300.0, 512.0, 312.0, 0.9)
    boxes[210] = (800.0, 300.0, 812.0, 312.0, 0.9)
    pa = (1, np.array([490.0, 250.0, 530.0, 380.0]), np.array([5.0, 7.0]))
    pb = (2, np.array([790.0, 250.0, 830.0, 380.0]), np.array([20.0, 7.0]))
    players = [[] for _ in range(n)]
    players[10], players[210] = [pa], [pb]
    fr = ball_court_positions(boxes, players, FPS, BallConfig())
    assert all(f.x is not None for f in fr)
    assert fr[0].state == GUESSED and abs(fr[0].x - 5.0) < 1e-6  # before the first sighting: where it was seen
    assert fr[100].state == GUESSED and abs(fr[100].x - 5.0) < 1e-6  # held at the last place while unseen
    assert 5.0 < fr[205].x < 20.0  # moving to the next sighting just before it
    assert fr[215].state == GUESSED and abs(fr[215].x - 20.0) < 1e-6
