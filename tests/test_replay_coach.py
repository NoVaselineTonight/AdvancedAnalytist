"""Coach quality-of-life in the 2D replay: selection + trail + name, tooltip, ball trail, shortcuts, next event."""
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QPointF, Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

import numpy as np  # noqa: E402

from replay.gui import motion  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _data(n=60):
    tr = lambda i, team, y: {"id": i, "team": team, "type": "player", "i0": 0,  # noqa: E731
                             "x": [500 + 30 * k for k in range(n)], "y": [y] * n, "f": [1] * n}
    return {"fps": 10.0, "t": [k / 10 for k in range(n)], "calib": [1] * n,
            "tracks": [tr(1, "A", 300), tr(2, "B", 900)], "ball": None,
            "track_info": {"1": {"number": "23"}},
            "game": {"teams": {"A": {"name": "Orange", "primary": "#ff6600"},
                               "B": {"name": "Purple", "primary": "#7700cc"}}}}


def test_next_event_and_key_events():
    assert motion.next_event_time([1.0, 5.0, 9.0], 5.0, 1) == 9.0
    assert motion.next_event_time([1.0, 5.0, 9.0], 5.0, -1) == 1.0
    assert motion.next_event_time([1.0], 5.0, 1) is None
    assert motion.is_key_event("gs-shot-3") and motion.is_key_event("basket-288-L")
    assert motion.is_key_event("be-7", "Steal by #12") and not motion.is_key_event("gs-inbound-4", "Inbound")
    assert not motion.is_key_event(None, "Shot")  # user marks are not auto events


def test_trail_buffer_resets_on_seek():
    b = motion.TrailBuffer(keep_s=1.0)
    for k in range(20):
        b.add(k * 0.1, k, 0)
    assert len(b.pts) in (10, 11) and b.pts[0][0] >= 0.9 - 1e-9
    b.add(0.5, 1, 1)  # seek back
    assert len(b.pts) == 1


def test_click_selects_tooltip_and_caption(app):
    from replay.gui.widgets import ReplayCourt
    w = ReplayCourt()
    w.set_data(_data())
    w.resize(700, 400)
    w.set_time(2.0)
    w.grab()
    dot = next(d for d in w.dots if d[0]["id"] == 1)
    tip = w.tooltip_text(QPointF(dot[1], dot[2]))
    assert "#23" in tip and "Orange" in tip and "km/h" in tip  # 3 m/s = 11 km/h
    got = []
    w.trackSelected.connect(got.append)

    class Ev:  # minimal mouse event
        def button(self):
            return Qt.MouseButton.LeftButton

        def position(self):
            return QPointF(dot[1], dot[2])

        def accept(self):
            pass
    w.mousePressEvent(Ev())
    assert w.selected == "1" and got == ["1"]
    w.set_time(4.0)
    img = w.grab().toImage()  # trail + ring + caption drawn without errors
    assert not img.isNull()
    w.mousePressEvent(Ev().__class__())  # clicking the same spot again: the dot moved on -> clears
    assert w.selected is None


def test_ball_trail_collects_drawn_ball(app):
    from replay.gui.widgets import ReplayCourt
    w = ReplayCourt()
    w.set_data(_data())
    w.resize(700, 400)
    w.ball_trail = True
    w._ball_hist.add(1.0, 5.0, 5.0)
    w._ball_hist.add(1.1, 5.5, 5.0)
    w.set_time(1.2)
    w.grab()  # no ball in this data: the trail stops (nothing drawn, no error)
    assert w._ball_hist.pts[-1][0] <= 1.1


def test_replay_tab_has_coach_shortcuts(app):
    from PySide6.QtGui import QKeySequence, QShortcut
    from replay.gui.app import ReplayTab

    class Win:
        pass
    tab = ReplayTab(Win())
    keys = {sc.key().toString() for sc in tab.findChildren(QShortcut)}
    for k in ("Left", "Right", "Shift+Left", "Shift+Right", ",", ".", "N", "Shift+N", "1", "5", "Space"):
        assert QKeySequence(k).toString() in keys, k
    assert tab.ball_trail.text() == "Ball trail"
    tab.ball_trail.setChecked(True)
    assert tab.court.ball_trail


def _play(n=80, run=True):
    """8 team players running right at 4 m/s; player 9 (with them) seen only for the first 2 s."""
    xs = lambda x0: [int(100 * (x0 + (4.0 * k / 10 if run else 0.0))) for k in range(n)]  # noqa: E731
    tracks = [{"id": p, "team": "A" if p % 2 else "B", "type": "player", "i0": 0, "x": xs(3 + p),
               "y": [100 * (2 + p)] * n, "f": [1] * n} for p in range(8)]
    seen = 20
    tracks.append({"id": 9, "team": "A", "type": "player", "i0": 0, "x": xs(5.5)[:seen], "y": [700] * seen,
                   "f": [1] * seen})
    return {"fps": 10.0, "t": [k / 10 for k in range(n)], "calib": [1] * n, "tracks": tracks, "ball": None,
            "court": {"length": 28.0, "width": 15.0}}


def test_joint_display_moves_with_the_play_and_fades(monkeypatch):
    monkeypatch.setenv("REPLAY_JOINT_2D", "1")
    d = _play()
    m = motion.ReplayModel(d)
    p = m.paths(0, 1.5).get(9)
    assert p.ext is not None
    st = p.at(1.9 + 1.5, hold_s=4.0)  # 1.5 s after the last sighting: moved on with the others
    truth_x = 5.5 + 4.0 * 3.4
    assert st and abs(st[0][0] - truth_x) < 3.0 and st[0][0] > 5.5 + 4.0 * 1.9 + 2.0
    a_early = p.at(1.9 + 0.3, hold_s=4.0)[0][3]
    late = p.at(1.9 + 3.0, hold_s=4.0)
    assert not late or late[0][3] < 0.3 * a_early  # honest fade: gone / faint once unsure
    for dt in (0.2, 1.0, 2.0):  # never faster than a person can run
        a, b = p.ext(len(p.t) - 1, dt), p.ext(len(p.t) - 1, dt + 0.1)
        assert np.hypot(b[0] - a[0], b[1] - a[1]) <= motion.VMAX * 0.1 + 1e-6


def test_joint_display_off_keeps_old_coast(monkeypatch):
    monkeypatch.setenv("REPLAY_JOINT_2D", "0")
    m = motion.ReplayModel(_play())
    p = m.paths(0, 1.5).get(9)
    assert p.ext is None
    st = p.at(1.9 + 1.5, hold_s=4.0)
    assert st and abs(st[0][0] - (5.5 + 4.0 * 1.9)) <= motion.COAST_MAX + 1e-6
