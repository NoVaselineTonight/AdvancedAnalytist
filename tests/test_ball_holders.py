"""Ball holder decoding and the free-throw prior (replay/ball.py)."""

from __future__ import annotations

import numpy as np

from replay import ball, court
from replay.config import BallConfig

FPS = 10.0
NEW = BallConfig(holder_decode=True, ft_prior=True, person_anchors=False)  # the new tracker (off by default)


def _box(cx: float) -> np.ndarray:
    return np.array([cx - 20, 250.0, cx + 20, 380.0])


def test_single_overlap_does_not_snap_the_ball_across_the_court():
    """A holds the ball for 2 s; in one frame the ball is seen in front of B (12 m away) -> still A's ball, not a
    teleport to B and back."""
    n = 20
    img = np.array([[500.0, 300.0]] * n)
    size = np.full(n, 12.0)
    obs = np.ones(n, bool)
    A = (1, _box(500), np.array([5.0, 7.0]), "A")
    B = (2, _box(800), np.array([17.0, 7.0]), "B")
    players = [[A, B] for _ in range(n)]
    img[10] = (800.0, 300.0)  # the ball overlaps B for one frame
    cfg = NEW
    held = ball.decode_holders(img, size, obs, players, FPS, cfg)
    assert all(held[i][0] == 1 for i in held)
    assert 10 not in held  # the odd sighting is "free" (no holder), the gap logic keeps it with A
    greedy = ball.greedy_holders(img, size, obs, players, FPS, cfg)
    assert greedy[10][0] == 2  # the old per-sighting choice snapped to B


def test_real_pass_is_kept():
    n = 30
    img = np.array([[500.0, 300.0]] * 15 + [[800.0, 300.0]] * 15)
    size = np.full(n, 12.0)
    obs = np.ones(n, bool)
    A = (1, _box(500), np.array([5.0, 7.0]), "A")
    B = (2, _box(800), np.array([11.0, 7.0]), "A")
    players = [[A, B] for _ in range(n)]
    held = ball.decode_holders(img, size, obs, players, FPS, NEW)
    assert held[3][0] == 1 and held[25][0] == 2


def test_free_throw_window_and_shooter_prior():
    """Shooter at the right free-throw line, four players along the lane, everybody still for 10 s: a free-throw
    window; the ball seen in front of a lane player is never given to them."""
    court.set_dimensions(28.0, 15.0)
    n = 100
    L, mid = court.LENGTH, court.MID_Y
    pos = {1: (L - court.FT_LINE_X - 0.3, mid), 2: (L - 2.0, mid - 2.0), 3: (L - 3.0, mid + 2.0),
           4: (L - 2.0, mid + 2.0), 5: (L - 3.0, mid - 2.0), 6: (14.0, 4.0)}
    team = {1: "A", 2: "B", 3: "A", 4: "B", 5: "A", 6: "B"}
    track_xy = {t: {i: np.array(p) for i in range(n)} for t, p in pos.items()}
    players = [[(t, _box(100 + 100 * t), np.array(p), team[t]) for t, p in pos.items()] for _ in range(n)]
    wins = ball.free_throw_windows(players, track_xy, FPS, NEW)
    assert len(wins) == 1 and wins[0]["shooter"] == 1 and wins[0]["side"] == "R"
    assert 2 in wins[0]["others"] and wins[0]["i1"] - wins[0]["i0"] >= 80
    # ball seen inside lane player 3's box in the middle of the window
    boxes = [None] * n
    boxes[50] = (290.0, 300.0, 310.0, 320.0, 0.9)
    fr = ball.ball_court_positions(boxes, players, FPS, NEW, track_xy=track_xy)
    assert fr[50].owner != 3
    assert fr[50].owner == 1 and abs(fr[50].x - pos[1][0]) < 1e-6
    assert all(f.owner in (1, None) for f in fr[wins[0]["i0"]:wins[0]["i1"]])
    # no window without the formation (live play: players moving)
    moving = {t: {i: np.array(p) + [0.15 * i, 0] for i in range(n)} for t, p in pos.items()}
    assert not ball.free_throw_windows(players, moving, FPS, NEW)
