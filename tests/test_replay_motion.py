"""Continuous 2D replay motion (replay/gui/motion.py) and its use in the replay view."""

import math
import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from replay.gui import motion as mo  # noqa: E402

FPS = 10.0


def _track(xs, ys=None, flags=None, i0=0, tid=1, team="A"):
    n = len(xs)
    ys = ys if ys is not None else [500] * n
    return {"id": tid, "team": team, "type": "player", "i0": i0, "x": list(xs), "y": list(ys),
            "f": flags if flags is not None else [0 if x is None else 1 for x in xs]}


def _ts(n):
    return [k / FPS for k in range(n)]


def _path(tr, n, bridge=mo.BRIDGE_MAX, smooth_n=0):
    return mo.build_path(tr, _ts(n), 1 / FPS, smooth_n, bridge)


def _alphas(path, t0, t1, **kw):
    out = []
    for t in np.arange(t0, t1, 1 / 60):
        st = path.at(t, **kw)
        out.append(max((e[3] for e in st), default=0.0))
    return np.array(out)


def test_passes_through_samples_with_continuous_velocity():
    xs = [100 + 10 * k + 0.5 * k * k for k in range(30)]  # accelerating along x (cm), below running speed
    p = _path(_track(xs), 30)
    for k in (3, 10, 20):
        st = p.at(k / FPS)
        assert len(st) == 1 and st[0][0] == pytest.approx(xs[k] / 100, abs=1e-6)
    # velocity is continuous across a sample (no direction change at every sample)
    a, b = p.at(1.0 - 1e-4)[0], p.at(1.0 + 1e-4)[0]
    assert abs(a[4] - b[4]) < 0.05
    # and close to the true speed (1 m/s per 0.1 s sample + acceleration)
    assert p.at(1.05)[0][4] == pytest.approx((xs[11] - xs[10]) / 100 * FPS, rel=0.1)


def test_short_gap_bridged_smoothly_long_gap_not_invented():
    xs = [100 + 30 * k for k in range(40)]
    short = [None if 10 <= k < 15 else x for k, x in enumerate(xs)]  # 0.5 s gap
    p = _path(_track(short), 40)
    st = p.at(1.2)
    assert len(st) == 1 and st[0][2] == 2  # bridged, flagged interpolated
    assert st[0][0] == pytest.approx((100 + 30 * 12) / 100, abs=0.05)  # on the straight-line run
    long_ = [None if 10 <= k < 35 else x for k, x in enumerate(xs)]  # 2.5 s gap: not bridged
    p = _path(_track(long_), 40)
    mid = p.at(2.2, hold_s=4.0)
    assert all(e[3] < 0.5 and e[2] == 2 for e in mid)  # only a fading, coasting ghost of the last sighting
    assert all(abs(e[0] - (100 + 30 * 9) / 100) <= mo.COAST_MAX + 1e-6 for e in mid)
    assert p.at(2.2, hold_s=0.5) == []  # with a short hold: nothing there at all


def test_implausible_jump_is_cross_faded_not_slid():
    xs = [500] * 10 + [1500] * 10  # 10 m in 0.1 s: an ID mix-up, not a person
    p = _path(_track(xs), 20)
    st = p.at(0.95, hold_s=0.0)
    for e in st:
        assert abs(e[0] - 5.0) < 0.3 or abs(e[0] - 15.0) < 0.3  # never half way across the court
    # each drawn dot moves plausibly between frames; a dot at a new place appears faint (fading in)
    for t in np.arange(0.5, 1.5, 1 / 60):
        a, b = p.at(t, hold_s=0.0), p.at(t + 1 / 60, hold_s=0.0)
        for e in b:
            if not a or min(math.hypot(e[0] - q[0], e[1] - q[1]) for q in a) > 0.5:
                assert e[3] <= 0.15


def test_no_popping_fades_in_and_out():
    xs = [None] * 5 + [800] * 20 + [None] * 50
    p = _path(_track(xs), 75)
    al = _alphas(p, 0.0, 7.5, hold_s=2.0, fade_s=0.25)
    assert al[0] == 0 and al[-1] == 0
    assert np.abs(np.diff(al)).max() < 0.15  # smooth at 60 fps
    assert al.max() == pytest.approx(1.0)
    # unseen for longer than the hold: gone (players are not invented)
    assert p.at(2.4 + 2.1, hold_s=2.0) == []


def test_interpolated_runs_fade_with_distance_to_measurements():
    xs = [800] * 60
    flags = [1] * 10 + [2] * 40 + [1] * 10  # 4 s filled in by the smoother
    p = _path(_track(xs, flags=flags), 60)
    near, far = p.at(1.1)[0], p.at(3.0)[0]
    assert near[3] == pytest.approx(1.0) and far[3] < 0.6 and far[2] == 2


def test_speed_clamped_and_headings_from_velocity():
    xs = [100 + 60 * k for k in range(20)]  # 6 m/s
    p = _path(_track(xs, ys=[300 + 0 * k for k in range(20)]), 20)
    e = p.at(1.03)[0]
    assert e[4] == pytest.approx(6.0, rel=0.05) and abs(e[5]) < 0.05
    assert max(abs(v) for v in p.vx) <= mo.VMAX + 1e-9


def test_tracer_points_follow_runs():
    xs = [100 + 30 * k for k in range(30)]
    xs[15:22] = [None] * 7  # 0.7 s gap: bridged (curve points inside it)
    p = _path(_track(xs), 30)
    pieces = p.points(0.5, 2.5)
    assert len(pieces) == 1
    T, X, Y = pieces[0]
    assert T == sorted(T) and T[0] >= 0.5 and T[-1] <= 2.5
    assert any(1.5 < t < 2.1 for t in T)  # inside the gap


def test_separate_pushes_overlapping_dots_apart():
    pos = [[5.0, 5.0], [5.1, 5.0], [9.0, 9.0]]
    mo.separate(pos, [1.0, 1.0, 1.0], 0.6)
    assert math.hypot(pos[0][0] - pos[1][0], pos[0][1] - pos[1][1]) >= 0.6 - 1e-6
    assert pos[2] == [9.0, 9.0]
    assert pos[0][0] + pos[1][0] == pytest.approx(10.1)  # symmetric


def test_playback_clock_is_smooth_and_monotonic():
    c = mo.PlaybackClock()
    out, wall = [], 0.0
    for k in range(240):  # 4 s at 60 Hz; the player reports a new position only every ~66 ms, ~20 ms late
        wall = k / 60
        pos = math.floor(max(0.0, wall - 0.02) / 0.066) * 0.066
        out.append(c.update(pos, True, 1.0, wall))
    steps = np.diff(out[30:])
    assert (steps >= 0).all()
    assert np.percentile(steps, 5) > 0.005 and np.percentile(steps, 95) < 0.03  # no repeated / doubled frames
    assert abs(out[-1] - (wall - 0.02)) < 0.08
    # a seek jumps; pausing follows the player exactly
    assert c.update(100.0, True, 1.0, wall + 1 / 60) == pytest.approx(100.0)
    assert c.update(42.0, False, 1.0, wall + 2 / 60) == 42.0


# ------------------------------------------------------------------ ball
def _ball_setup(n=60):
    a = _track([500 + 10 * k for k in range(n)], [500] * n, tid=1)
    b = _track([1500] * n, [800] * n, tid=2)
    return a, b


def _ball_model(ball, tracks, n):
    data = {"fps": FPS, "t": _ts(n), "tracks": tracks, "ball": ball, "calib": [1] * n}
    return mo.ReplayModel(data)


def _pos_of(model):
    paths = {str(tr["id"]): p for tr, p in zip(model.tracks, model.paths(0))}

    def pos_of(h, t):
        p = paths.get(h)
        st = p.at(t) if p else []
        return (st[0][0], st[0][1]) if st else None
    return pos_of


def test_ball_with_holder_then_pass_across_unknown_stretch():
    n = 60
    a, b = _ball_setup(n)
    bx, by, bf, bo = [None] * n, [None] * n, [0] * n, [None] * n
    for k in range(0, 20):  # held by 1
        bx[k], by[k], bf[k], bo[k] = a["x"][k], 500, 1, 1
    for k in range(35, n):  # held by 2 after 1.5 s unknown
        bx[k], by[k], bf[k], bo[k] = 1500, 800, 1, 2
    m = _ball_model({"x": bx, "y": by, "f": bf, "o": bo}, [a, b], n)
    bp, pos_of = m.ball(), _pos_of(m)
    s = bp.at(1.05, pos_of)  # half a sample after the last sighting: still visible
    assert s is not None and s[2] == 1.0 and s[4] == "1"
    t_seq = np.arange(0.0, 5.9, 1 / 60)
    st = [bp.at(t, pos_of) for t in t_seq]
    assert all(x is not None and x[2] > 0.99 for x in st)  # never disappears
    xy = np.array([(x[0], x[1]) for x in st])
    speed = np.hypot(*np.diff(xy, axis=0).T) * 60
    assert speed.max() < 14.0  # a plausible pass, no teleport
    # first stays with the passer (keeps moving with them), then flies to the receiver
    s = bp.at(2.0, pos_of)
    assert s[4] == "1" and s[0] == pytest.approx((500 + 10 * 20) / 100, abs=0.15)
    assert bp.at(3.5, pos_of)[4] == "2"


def test_ball_fades_only_after_a_long_gap():
    n = 120
    a, b = _ball_setup(n)
    bx, by, bf, bo = [None] * n, [None] * n, [0] * n, [None] * n
    for k in range(0, 10):
        bx[k], by[k], bf[k], bo[k] = a["x"][k], 500, 1, 1
    for k in range(100, n):
        bx[k], by[k], bf[k], bo[k] = 1500, 800, 1, 2
    m = _ball_model({"x": bx, "y": by, "f": bf, "o": bo}, [a, b], n)
    bp, pos_of = m.ball(), _pos_of(m)
    assert bp.at(2.5, pos_of)[2] == 1.0  # 1.6 s unknown: still with the holder
    assert bp.at(6.0, pos_of)[2] == 0.0  # 5 s unknown: faded
    al = [bp.at(t, pos_of)[2] for t in np.arange(0, 11.9, 1 / 60)]
    assert np.abs(np.diff(al)).max() < 0.1  # fades, never pops


def test_ball_fast_hop_retimed():
    n = 40
    a, b = _ball_setup(n)
    bx, by, bf, bo = [None] * n, [None] * n, [0] * n, [None] * n
    for k in range(0, 20):
        bx[k], by[k], bf[k], bo[k] = a["x"][k], 500, 1, 1
    for k in range(20, n):  # caught 10 m away one sample later
        bx[k], by[k], bf[k], bo[k] = 1500, 800, 1, 2
    m = _ball_model({"x": bx, "y": by, "f": bf, "o": bo}, [a, b], n)
    bp, pos_of = m.ball(), _pos_of(m)
    xy = np.array([bp.at(t, pos_of)[:2] for t in np.arange(1.0, 2.5, 1 / 60)])
    assert (np.hypot(*np.diff(xy, axis=0).T) * 60).max() < 20.0


# ------------------------------------------------------------------ drawing
@pytest.fixture(scope="module")
def app():
    from PySide6.QtGui import QGuiApplication
    return QGuiApplication.instance() or QGuiApplication([])


def _render(d, t, **kw):
    from PySide6.QtCore import QRect
    from PySide6.QtGui import QColor, QImage, QPainter

    from replay.gui import widgets as w
    img = QImage(560, 320, QImage.Format.Format_ARGB32)
    img.fill(QColor("#000000"))
    qp = QPainter(img)
    w.paint_court(qp, QRect(0, 0, 560, 320), d, t, colors={"A": "#ff0000", "B": "#0000ff"}, **kw)
    qp.end()
    return img


def _count_near(img, rgb, tol=40):
    n = 0
    for y in range(0, img.height(), 2):
        for x in range(0, img.width(), 2):
            c = img.pixelColor(x, y)
            if abs(c.red() - rgb[0]) + abs(c.green() - rgb[1]) + abs(c.blue() - rgb[2]) < tol:
                n += 1
    return n


def test_ball_drawn_while_position_unknown(app):
    n = 60
    a, b = _ball_setup(n)
    bx, by, bf, bo = [None] * n, [None] * n, [0] * n, [None] * n
    for k in range(0, 20):
        bx[k], by[k], bf[k], bo[k] = a["x"][k], 500, 1, 1
    d = {"fps": FPS, "t": _ts(n), "tracks": [a, b], "ball": {"x": bx, "y": by, "f": bf, "o": bo},
         "calib": [1] * n}
    assert _count_near(_render(d, 3.0), (255, 212, 0)) > 3  # 1 s after the last sighting: with the holder
    assert _count_near(_render(d, 5.6), (255, 212, 0)) == 0  # >3 s: faded out


def test_all_display_options_render(app):
    n = 50
    xs = [500 + 20 * k for k in range(n)]
    xs[20:26] = [None] * 6
    d = {"fps": FPS, "t": _ts(n), "calib": [1] * n, "ball": None,
         "tracks": [_track(xs, tid=1), _track([900] * n, [520] * n, tid=2, team="B"),
                    _track([1200] * n, [300] * n, tid=3, team="referee")]}
    for kw in ({}, {"trails": True, "trail_s": 3.0}, {"smooth_s": 1.0}, {"hollow_interp": True},
               {"show_interp": False}, {"hold_s": 0.0}, {"headings": False, "labels": False},
               {"show_referees": False}):
        img = _render(d, 2.25, **kw)
        assert _count_near(img, (0, 0, 255), 10) > 3  # team B always drawn
    # interpolated (bridged) positions hidden when "Interpolated" is off
    assert _count_near(_render(d, 2.25), (255, 0, 0), 10) > 3
    assert _count_near(_render(d, 2.25, show_interp=False), (255, 0, 0), 10) == 0


def test_clock_resumes_after_pause_at_same_position():
    from replay.gui.motion import PlaybackClock

    c = PlaybackClock()
    assert c.update(5.0, False, 1.0, 0.0) == 5.0
    t1 = c.update(5.0, True, 1.0, 0.1)  # play pressed, player position not moved yet: must not crash
    t2 = c.update(5.0, True, 1.0, 0.2)
    assert 5.0 <= t1 <= t2 <= 5.3
