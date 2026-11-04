"""Occlusion re-attach and the game-aware motion prior (replay/court_tracker.py)."""
import numpy as np

from replay import court
from replay.court_tracker import (DEAD, TRANSITION, MotionPrior, OcclusionContext, reattach_hidden)

FPS = 10.0


def _ctx(n, occluder_box=None):
    H = [np.diag([0.02, 0.02, 1.0])] * n  # image = court * 50 px
    boxes = [np.zeros((0, 4)) if occluder_box is None else np.array([occluder_box]) for _ in range(n)]
    return OcclusionContext(H, boxes, (1280, 720))


def _two_pieces(gap_end_x=6.0):
    # identity 1: samples 0..19 walking to x=5 m (y=4 m); identity 2: samples 40..59 from x=gap_end_x
    k = np.r_[np.arange(20), np.arange(40, 60)]
    xy = np.c_[np.r_[np.linspace(4, 5, 20), np.linspace(gap_end_x, gap_end_x + 1, 20)], np.full(40, 4.0)]
    ids = np.r_[np.ones(20, int), np.full(20, 2)]
    feats = np.tile([50.0, 40.0, 30.0, 40.0, 10.0, 10.0], (40, 1))
    return ids, k, xy, feats


def test_reattach_across_hidden_gap():
    ids, k, xy, feats = _two_pieces()
    ctx = _ctx(60, occluder_box=[200, 120, 340, 300])  # covers foot pixels x 250..300, y 200 (behind it)
    new, n = reattach_hidden(ids, k, xy, feats, FPS, ctx)
    assert n == 1 and len(np.unique(new)) == 1


def test_no_reattach_when_spot_visible_and_empty():
    ids, k, xy, feats = _two_pieces()
    new, n = reattach_hidden(ids, k, xy, feats, FPS, _ctx(60))
    assert n == 0 and len(np.unique(new)) == 2


def test_no_reattach_unreachable_or_other_kit_or_number():
    ctx = _ctx(60, occluder_box=[0, 0, 1280, 719])  # everything hidden
    ids, k, xy, feats = _two_pieces(gap_end_x=25.0)  # 20 m in 2 s
    assert reattach_hidden(ids, k, xy, feats, FPS, ctx)[1] == 0
    ids, k, xy, feats = _two_pieces()
    feats[20:, [1, 2]] = [-20.0, -25.0]  # purple instead of orange
    assert reattach_hidden(ids, k, xy, feats, FPS, ctx)[1] == 0
    ids, k, xy, feats = _two_pieces()
    nums = np.r_[np.full(20, 7), np.full(20, 12)]
    assert reattach_hidden(ids, k, xy, feats, FPS, ctx, numbers=nums)[1] == 0


def _prior_case(vx):
    n = 100
    t = np.arange(n) / FPS
    det_i, xy, tid = [], [], []
    for p in range(8):
        for i in range(n):
            det_i.append(i)
            xy.append([5 + p + vx * t[i], 3 + p])
            tid.append(p + 1)
    return MotionPrior.from_detections(t, np.array(det_i), np.array(xy, float), np.array(tid))


def test_phases_dead_and_transition():
    assert (_prior_case(0.0).phase[10:90] == DEAD).all()
    tr = _prior_case(5.0)
    assert (tr.phase[10:90] == TRANSITION).all() and tr.flow[50] > 4


def test_prior_predictions():
    court.set_dimensions(28.0, 15.0)
    dead = _prior_case(0.0)
    p, s_dead = dead.predict(10, np.array([8.0, 5.0]), np.array([0.0, 0.0]), 40)
    assert np.hypot(*(p - [8.0, 5.0])) < 0.1
    tr = _prior_case(5.0)
    # a player standing still when the fast break starts is predicted to run with it
    p, s_tr = tr.predict(10, np.array([8.0, 5.0]), np.array([0.0, 0.0]), 40)
    assert p[0] > 14.0 and s_tr > s_dead


def test_prior_from_pass1_rows_ignores_off_court_people():
    court.set_dimensions(28.0, 15.0)
    n = 60
    t = np.arange(n) / FPS
    H = [np.diag([0.02, 0.02, 1.0])] * n  # pixel -> court: /50
    rows = []
    for i in range(n):
        for p in range(6):  # players running right at 5 m/s: foot x = (5 + p + 5 t) m
            x = (5 + p + 5 * t[i]) * 50
            rows.append([i, x - 20, 100, x + 20, (3 + p) * 50, 0.9, 0, p, 0, 0, 0, 0, 0, 0])
        for b in range(4):  # static bench people off the court (y = 16 m)
            rows.append([i, 100 * b, 600, 100 * b + 40, 16 * 50, 0.9, 0, 10 + b, 0, 0, 0, 0, 0, 0])
    pr = MotionPrior.from_rows(H, np.array(rows, float), {i: i for i in range(n)}, t)
    assert (pr.phase[10:35] == TRANSITION).all() and 4.5 < pr.flow[25] < 5.5
