"""Rim second look (replay/rimlook.py): windows, crops, merging new candidates."""

from __future__ import annotations

import numpy as np

from replay import rimlook as rl


class _Det:
    threshold = 0.3

    def detect(self, img):
        from replay.detection import BALL, Detections
        h, w = img.shape[:2]
        return Detections(np.array([[w / 2 - 5, h / 2 - 5, w / 2 + 5, h / 2 + 5]], np.float32), np.array([0.2], np.float32),
                          np.array([BALL]))


def test_windows_follow_a_ball_near_the_rim():
    n = 60
    t = np.arange(n) / 10.0
    rims = {"L": [(500.0, 200.0, 10.0)] * n, "R": [None] * n}
    ball_rows = [np.zeros((0, 5)) for _ in range(n)]
    ball_rows[20] = np.array([[497, 50, 503, 60, 0.5]])  # 14.5 radii above the ring
    ball_rows[40] = np.array([[900, 200, 910, 210, 0.5]])  # far to the side: no window
    w = rl.windows(ball_rows, rims, t, rl.RimLookConfig(stride=1))
    idx = sorted(i for i, s in w)
    assert idx[0] == 18 and idx[-1] == 35 and all(s == "L" for _, s in w)
    x0, y0, x1, y1 = rl.crop_region(rims["L"][0], (1280, 720), rl.RimLookConfig())
    assert x0 < 500 < x1 and y0 < 200 < y1 and x1 - x0 == 160


def test_augment_adds_candidates(tmp_path, monkeypatch):
    n = 30
    t = np.arange(n) / 10.0

    class D:
        pass
    data = D()
    data.t = t
    rims = {"L": [(500.0, 200.0, 10.0)] * n}
    ball_rows = [np.zeros((0, 5)) for _ in range(n)]
    ball_rows[5] = np.array([[495, 100, 505, 110, 0.5]])
    import replay.redetect as rd
    import replay.team_appearance as ta
    monkeypatch.setattr(ta, "resolve_video", lambda data, video: "x.mp4")
    monkeypatch.setattr(rd, "iter_frames", lambda vid, data, want: ((i, np.zeros((720, 1280, 3), np.uint8)) for i in want))
    monkeypatch.setattr(rl, "ENABLED", True)
    out = rl.augment(tmp_path, data, ball_rows, rims, None, None, (1280, 720), rc=rl.RimLookConfig(stride=1),
                     detector=_Det())
    added = sum(len(r) for r in out) - 1
    assert added >= 15 and all(len(r) <= 2 for r in out)
    assert (tmp_path / rl.CACHE).exists()


def test_attempt_windows_follow_rim_anchors():
    from replay import court
    court.set_dimensions(28.0, 15.0)

    class F:
        def __init__(self, i, src, x):
            self.i, self.source, self.x = i, src, x
    n = 80
    t = np.arange(n) / 10.0
    bxL = court.baskets()[0][0]
    frames = [F(i, "rim" if i in (20, 21, 22, 60) else "fill", bxL) for i in range(n)]
    rims = {"L": [(500.0, 200.0, 10.0)] * n, "R": [None] * n}
    w = rl.attempt_windows(frames, rims, t, rl.RimLookConfig(stride=2))
    idx = [i for i, s in w]
    assert all(s == "L" for _, s in w) and idx[0] == 18 and 60 in idx and len(idx) <= 2 * 9


def test_balldet_off_by_default_and_decode():
    from replay import balldet
    assert balldet.ENABLED is False or __import__("os").environ.get("REPLAY_BALLDET") == "1"
    out = np.full((1, 2, 10, 20), -10.0, np.float32)
    out[0, 0, 4, 6], out[0, 1, 4, 6] = 5.0, 0.5  # one peak, diameter 16 scaled px
    boxes, sc = balldet.decode(out, 1.0, thr=0.3)
    assert len(boxes) == 1 and abs((boxes[0, 0] + boxes[0, 2]) / 2 - 26) < 1e-6 and abs(boxes[0, 2] - boxes[0, 0] - 16) < 1e-6
