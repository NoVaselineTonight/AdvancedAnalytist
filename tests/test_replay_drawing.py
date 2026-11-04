import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QRect  # noqa: E402
from PySide6.QtGui import QColor, QGuiApplication, QImage, QPainter  # noqa: E402

from replay.gui import widgets as w  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QGuiApplication.instance() or QGuiApplication([])


def _data(n=30, court=None):
    tr = lambda i, team, y: {"id": i, "team": team, "type": "player", "i0": 0,  # noqa: E731
                             "x": [500 + 20 * k for k in range(n)], "y": [y] * n, "f": [1] * n}
    d = {"fps": 10.0, "t": [k / 10 for k in range(n)], "calib": [1] * n,
         "tracks": [tr(1, "A", 300), tr(2, "B", 700), tr(3, "referee", 1100)], "ball": None}
    if court:
        d["court"] = court
    return d


def _render(d, t, **kw):
    img = QImage(560, 320, QImage.Format.Format_ARGB32)
    img.fill(QColor("#000000"))
    qp = QPainter(img)
    w.paint_court(qp, QRect(0, 0, 560, 320), d, t, colors={"A": "#ff0000", "B": "#0000ff", "referee": "#00ff00"},
                  **kw)
    qp.end()
    return img


def _count(img, rgb):
    n = 0
    for y in range(0, img.height(), 2):
        for x in range(0, img.width(), 2):
            c = img.pixelColor(x, y)
            if (c.red(), c.green(), c.blue()) == rgb:
                n += 1
    return n


def test_referees_toggle(app):
    d = _data()
    assert _count(_render(d, 1.5), (0, 255, 0)) > 5
    assert _count(_render(d, 1.5, show_referees=False), (0, 255, 0)) == 0
    assert _count(_render(d, 1.5, show_referees=False), (255, 0, 0)) > 5


def test_court_size_from_data():
    assert w.court_size(_data(court={"length": 26.0, "width": 14.0})) == (26.0, 14.0)
    assert w.court_size(_data()) == (w.court.LENGTH, w.court.WIDTH)


def test_cubic_interpolation_passes_through_samples_and_is_smooth():
    tr = {"i0": 0, "x": [0, 100, 300, 600, 1000], "y": [0] * 5, "f": [1] * 5}
    assert w.track_position(tr, 2, 0.0)[0] == pytest.approx(3.0)
    assert w.track_position(tr, 1, 1.0)[0] == pytest.approx(3.0)
    # accelerating motion: midpoint lies below the linear midpoint (curve follows the acceleration)
    assert w.track_position(tr, 2, 0.5)[0] < 4.5


def test_fade_in_out_instead_of_popping():
    tr = {"i0": 5, "x": [100] * 20, "y": [100] * 20, "f": [1] * 20}
    assert w._run_fade(tr, 5, 0.0, 3) < 0.5
    assert w._run_fade(tr, 12, 0.0, 3) == 1.0
    assert w._run_fade(tr, 24, 0.0, 3) < 0.5
    held = w._held_position(tr, 24 + 30, 0.0, 40, 0)
    assert held is not None and held[3] < 0.3
