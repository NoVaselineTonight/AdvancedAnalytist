"""Joint extrapolation (replay/extrapolate.py)."""
import numpy as np

from replay import court
from replay.court_tracker import OcclusionContext, VISIBLE
from replay.extrapolate import JointField, exclude_view

FPS = 10.0


def _field(n=100, vx=4.0):
    """Six players run right at vx m/s for 5 s and back; player 'h' (with them) is hidden 20..80."""
    t = np.arange(n) / FPS
    x = np.where(t < 5, vx * t, vx * (10 - t))
    P = {}
    for p in range(6):
        P[p] = np.c_[3 + x + p, np.full(n, 3.0 + p)]
    H = np.c_[3 + x + 2.5, np.full(n, 5.0)]
    hidden = H.copy()
    hidden[20:80] = np.nan
    P["h"] = hidden
    return JointField(t, P, {q: "A" for q in P}, np.full((n, 2), np.nan)), H


def test_bridge_follows_the_play():
    court.set_dimensions(28.0, 15.0)
    F, truth = _field()
    ks = np.arange(21, 80)
    base = truth[20] + ((ks - 20) / 60)[:, None] * (truth[80] - truth[20])  # straight line = single-track bridge
    j = F.bridge("h", 20, 80, ks, base, truth[20])
    e0 = np.abs(base - truth[ks]).mean()
    e1 = np.abs(j - truth[ks]).mean()
    assert e1 < 0.8 * e0


def test_open_end_follows_the_play_and_sigma_grows():
    court.set_dimensions(28.0, 15.0)
    F, truth = _field()
    ks = np.arange(21, 50)
    xy, sg = F.extrapolate("h", 20, ks, truth[20], np.zeros(2))
    assert np.abs(xy[-1] - truth[49]).max() < np.abs(truth[20] - truth[49]).max()
    assert (np.diff(sg) > 0).all()


def test_view_exclusion_moves_out_of_the_picture():
    court.set_dimensions(28.0, 15.0)
    H = [np.diag([0.02, 0.02, 1.0])]  # image = court * 50 px, 1280 x 720 -> view x < 25.6 m, y < 14.3 m
    ctx = OcclusionContext(H, [np.zeros((0, 4))], (1280, 720))
    xy, moved = exclude_view(ctx, np.array([0]), np.array([[24.0, 7.0]]))
    assert moved[0] and xy[0, 0] > 25.6
    assert ctx.classify(0, xy[0])[0] != VISIBLE


def test_defender_shadow_follows_his_man():
    court.set_dimensions(28.0, 15.0)
    n = 60
    t = np.arange(n) / FPS
    man = np.c_[5 + 3 * t, np.full(n, 7.0)]  # attacker drives right towards the basket at x = 26.4
    hidden_def = np.full((n, 2), np.nan)
    hidden_def[:11] = man[:11] + [1.0, 0.0]  # defender 1 m on the basket side, then hidden
    P = {"atk": man, "def": hidden_def}
    F = JointField(t, P, {"atk": "B", "def": "A"}, np.full((n, 2), np.nan))
    ks = np.arange(11, 60)
    plain, _ = F.extrapolate("def", 10, ks, hidden_def[10], np.zeros(2))
    sh, _ = F.extrapolate("def", 10, ks, hidden_def[10], np.zeros(2), defending_basket=np.array([26.4, 7.5]))
    truth = man[ks] + [1.0, 0.0]
    assert np.abs(sh - truth).mean() < 0.7 * np.abs(plain - truth).mean()
