"""Default (clean) 2D view: no far predictions, nothing far off court, no teleporting dots (display only)."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

from test_replay_motion import _count_near, _render, _track, _ts  # noqa: E402


def _tinted(img, ch):
    """Pixels clearly tinted towards one channel (0 red, 2 blue) - also pale / translucent dots."""
    n = 0
    for y in range(0, img.height(), 2):
        for x in range(0, img.width(), 2):
            c = img.pixelColor(x, y)
            v = (c.red(), c.green(), c.blue())
            others = [v[i] for i in range(3) if i != ch]
            if v[ch] - max(others) > 70:
                n += 1
    return n


@pytest.fixture(scope="module")
def app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_clean_view_hides_long_predictions_and_off_court(app):
    n = 60
    # player A: seen until 1.0 s, then "predicted" (f=2) positions for 5 s
    a = _track([500 + 10 * k for k in range(n)], flags=[1] * 10 + [2] * (n - 10), tid=1)
    # player B: seen, but 5 m beyond the far sideline (a projection error)
    b = _track([900] * n, [2000] * n, tid=2, team="B")
    d = {"fps": 10.0, "t": _ts(n), "calib": [1] * n, "ball": None, "tracks": [a, b]}
    assert _count_near(_render(d, 1.2), (255, 0, 0), 10) > 3  # 0.2 s after the last sighting: solid
    rims = _tinted(_render({**d, "tracks": [_track([None] * n, tid=9)]}, 4.0), 0)  # the red rims
    assert _tinted(_render(d, 4.0), 0) == rims  # 3 s unseen: not drawn by default
    assert _tinted(_render(d, 4.0, confidence=True), 0) > rims  # ... but with Confidence
    assert _tinted(_render(d, 1.2), 2) == 0  # far off court: hidden
    assert _tinted(_render(d, 1.2, confidence=True), 2) > 0


def test_drawn_dots_never_teleport(app):
    from PySide6.QtCore import QRect
    from PySide6.QtGui import QColor, QImage, QPainter
    from replay.gui import widgets as w
    n = 40
    xs = [500] * 20 + [2000] * 20  # an identity switch: 15 m jump between two samples
    d = {"fps": 10.0, "t": _ts(n), "calib": [1] * n, "ball": None, "tracks": [_track(xs, tid=7)]}
    prev, seen = {}, []
    img = QImage(560, 320, QImage.Format.Format_ARGB32)
    for k in range(120):
        t = 1.0 + k / 60
        img.fill(QColor("#000000"))
        qp = QPainter(img)
        dots = []
        w.paint_court(qp, QRect(0, 0, 560, 320), d, t, colors={"A": "#ff0000"}, dots=dots, prev=prev)
        qp.end()
        seen.append([(x, y) for tr, x, y, *_ in dots])
    steps = [abs(b[0][0] - a[0][0]) for a, b in zip(seen, seen[1:]) if a and b]
    # either a slow slide (<= DRAW_VMAX) or a fade-in at the new place - never a fast slide across the court
    small = [s for s in steps if s < 30]
    assert len(small) >= len(steps) - 1
