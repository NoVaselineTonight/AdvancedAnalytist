"""GUI QA regressions: status with an automatic calibration, keyframe editing keeps the automatic calibration,
fast display smoothing, robust replay-data loading, stale AI-job signals."""

import json
import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from fake_project import make_project  # noqa: E402

AUTO = {"version": 1, "C": [13.0, -1.5, 2.0], "spec": "FIBA", "line_classes": [0], "good_frac": 0.9, "message": "",
        "anchors": [{"t": 0.0, "i": 0, "seg": 0, "pan": 0.1, "tilt": 0.2, "roll": 0.0, "f": 900.0, "cost": 1.0,
                     "inliers": 0.5, "good": True, "C": [13.0, -1.5, 2.0]}]}


@pytest.fixture(autouse=True)
def _restore_court():
    """Loading a calibration configures the global court size: put it back for the other tests."""
    from replay import court
    before = (court.LENGTH, court.WIDTH, court.SPEC)
    yield
    court.configure(before[0], before[1], before[2])


@pytest.fixture(scope="module")
def app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def _auto_calibration(root, keyframes=()):
    (root / "calibration.json").write_text(json.dumps({
        "video": "video.mp4", "frame_size": [1280, 720], "fixed_camera": False,
        "court": {"length": 26.5, "width": 15.0, "spec": "FIBA"}, "auto": AUTO, "keyframes": list(keyframes)}),
        encoding="utf-8")


def test_status_counts_automatic_calibration(tmp_path):
    from replay.gui.projects import Project
    root = make_project(tmp_path / "p")
    p = Project(root)
    _auto_calibration(root)
    s = p.status()
    assert s["keyframes"] == 0 and s["auto_calibrated"] and s["calibrated"]
    time.sleep(0.02)
    _auto_calibration(root, [{"time_s": 1.0, "points": {"center": [640, 400]}}])  # summary cache sees the change
    s = p.status()
    assert s["keyframes"] == 1 and s["calibrated"]
    (root / "calibration.json").write_text("{ damaged", encoding="utf-8")
    assert p.status()["calibrated"] is False  # never raises


def test_status_does_not_reconfigure_the_court(tmp_path):
    """Polling a Replay's status must not change the global court size (a build may be running)."""
    from replay import court
    from replay.gui.projects import Project
    root = make_project(tmp_path / "p")
    _auto_calibration(root)
    before = (court.LENGTH, court.WIDTH)
    Project(root).status()
    assert (court.LENGTH, court.WIDTH) == before


def test_status_text_with_automatic_calibration(tmp_path):
    from replay.gui.app import status_text
    from replay.gui.projects import Project
    root = make_project(tmp_path / "p")
    (root / "calibration.json").unlink()
    (root / "results" / "work" / "DONE").write_text("", encoding="utf-8")
    p = Project(root)
    job = {"error": None, "kind": None, "queued": [], "label": "", "progress": 0.0}
    assert status_text(p, job) == "Court not found - fix by hand"
    _auto_calibration(root)
    assert status_text(p, job).startswith("Replay ready")


def test_smoothing_cache_matches_plain_average():
    from replay.gui import widgets as w
    rng = np.random.default_rng(1)
    xs = [None if rng.random() < 0.2 else float(rng.integers(0, 2800)) for _ in range(300)]
    tr = {"i0": 7, "x": xs, "y": [None if x is None else x / 2 for x in xs], "f": [1] * 300}

    def plain(i, n):
        p = w._pos(tr, i)
        if p is None:
            return None
        pts = [q for q in (w._pos(tr, j) for j in range(i - n, i + n + 1)) if q is not None]
        return sum(q[0] for q in pts) / len(pts), sum(q[1] for q in pts) / len(pts), p[2]

    for n in (1, 4, 15):
        for i in range(0, 320):
            a, b = plain(i, n), w._sample(tr, i, n)
            assert (a is None) == (b is None)
            if a:
                assert abs(a[0] - b[0]) < 1e-9 and abs(a[1] - b[1]) < 1e-9


def test_keyframe_save_keeps_automatic_calibration(tmp_path, app):
    """Saving / deleting manual keyframes in the Calibrate tab must keep calibration.json's "auto" and "court"."""
    from replay.gui.app import CalibrateTab
    from replay.gui.projects import Project
    root = make_project(tmp_path / "p")
    _auto_calibration(root)

    class Win:  # the parts of MainWindow the tab uses
        class jobs:
            submitted = []

            @classmethod
            def submit(cls, pid, kind, **kw):
                cls.submitted.append(kind)

        def refresh_project(self):
            pass

    tab = CalibrateTab(Win())
    p = Project(root)
    tab.load(p)
    assert tab._auto_mode() and tab._min_points() == 3
    tab.frame_t = 1.0
    tab.points = {"L_corner_near": (100.0, 600.0), "L_ft_near": (300.0, 500.0), "L_ft_far": (320.0, 420.0)}
    tab.dirty = True
    tab.save_keyframe()
    cal = json.loads((root / "calibration.json").read_text(encoding="utf-8"))
    assert cal["auto"] == AUTO and cal["court"]["length"] == 26.5
    assert len(cal["keyframes"]) == 1 and len(cal["keyframes"][0]["points"]) == 3
    # four points on one line (the baseline) would make every build fail: refused
    tab.frame_t = 2.0
    tab.points = {n: (100.0 + i, 300.0 + 50 * i) for i, n in enumerate(
        ["L_corner_near", "L_three_base_near", "L_paint_base_near", "L_paint_base_far"])}
    tab.dirty = True
    tab.save_keyframe()
    cal = json.loads((root / "calibration.json").read_text(encoding="utf-8"))
    assert len(cal["keyframes"]) == 1 and cal["auto"] == AUTO


def test_replay_tab_survives_damaged_replay_data(tmp_path, app):
    from replay.gui.app import ReplayTab
    from replay.gui.projects import Project
    root = make_project(tmp_path / "p")
    (root / "results" / "replay_data.json").write_text("{ half written", encoding="utf-8")

    class Win:
        pass

    tab = ReplayTab(Win())
    tab.p = Project(root)
    tab.timeline.set_project(root)
    tab.reload_data()  # must not raise
    assert tab.data is None and "could not be read" in tab.msg.text()


def test_big_replay_data_loads_in_the_background(tmp_path, app, monkeypatch):
    """A full-game replay_data.json is parsed off the GUI thread ("Loading ..."), then shown."""
    import time

    from PySide6.QtWidgets import QApplication

    from replay.gui.app import ReplayTab
    from replay.gui.projects import Project
    root = make_project(tmp_path / "p")

    class Win:
        pass

    tab = ReplayTab(Win())
    monkeypatch.setattr(ReplayTab, "ASYNC_LOAD_BYTES", 10)  # treat the fixture as "big"
    tab.p = Project(root)
    tab.timeline.set_project(root)
    tab.reload_data()
    assert tab.data is None and "Loading" in tab.msg.text()
    t_end = time.time() + 10
    while tab.data is None and time.time() < t_end:
        QApplication.processEvents()
        time.sleep(0.01)
    assert tab.data is not None and tab.data.get("t")
    tab.reload_data()  # same file: no second load
    assert getattr(tab, "_loading_mt", None) is None


def test_async_load_worker_failure_reports_error_not_hang(tmp_path, app, monkeypatch):
    """Any exception in the background parse (bad roster.json, MemoryError...) is shown, never 'Loading' forever."""
    from PySide6.QtWidgets import QApplication

    from replay.gui.app import ReplayTab
    from replay.gui.projects import Project
    root = make_project(tmp_path / "p")

    class Win:
        pass

    def boom(p, f):
        raise KeyError("bad roster")

    tab = ReplayTab(Win())
    monkeypatch.setattr(ReplayTab, "ASYNC_LOAD_BYTES", 10)
    monkeypatch.setattr(ReplayTab, "_parse_replay", staticmethod(boom))
    tab.p = Project(root)
    tab.timeline.set_project(root)
    tab.reload_data()
    t_end = time.time() + 10
    while "could not be read" not in tab.msg.text() and time.time() < t_end:
        QApplication.processEvents()
        time.sleep(0.01)
    assert "could not be read" in tab.msg.text()
    assert getattr(tab, "_loading_mt", None) is None


def test_bad_roster_does_not_block_replay_parse(tmp_path, app, monkeypatch):
    from replay.gui.app import ReplayTab
    from replay.gui.projects import Project
    root = make_project(tmp_path / "p")
    import replay.roster_names as rn
    monkeypatch.setattr(rn, "annotate_replay_data", lambda r, d: (_ for _ in ()).throw(AttributeError("x")))
    data, err = ReplayTab._parse_replay(Project(root), Project(root).results / "replay_data.json")
    assert err is None and data.get("t")


def test_main_window_fits_720p_and_na_and_distinct_colours(tmp_path, app):
    from replay.gui import app as A
    from replay.gui import stats_tab
    from replay.team_binding import distinct_colors, hex_to_lab
    import numpy as np
    w = A.MainWindow(tmp_path)
    assert w.minimumSizeHint().height() <= 720  # 1280x720 including menu + status bar (was 814)
    w.close()
    assert stats_tab.fmt_value(None, "int") == "n/a" and stats_tab.fmt_value(0, "int", sparse=True) == "n/a"
    assert stats_tab.fmt_value(0, "int") == "0"
    d = distinct_colors({"A": "#d85a00", "B": "#d94040"})
    assert np.linalg.norm(hex_to_lab(d["A"]) - hex_to_lab(d["B"])) >= 45
