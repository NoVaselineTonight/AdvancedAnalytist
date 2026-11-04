"""Real-use QA regressions (GUI job chain, import, Overview report)."""

import json
import os
import threading
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

from fake_project import make_project  # noqa: E402


@pytest.fixture(scope="module")
def app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def _wait(cond, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


def test_restart_is_not_lost_when_the_running_job_is_cancelled(tmp_path):
    """Overview > Save with new analysis settings: cancel the running analysis + submit a restart. The restart
    used to be dropped because the same kind of job was still (briefly) running."""
    from replay.gui import jobs as J
    from replay.gui.projects import ProjectStore

    store = ProjectStore(tmp_path)
    p = store.create("x", {})
    calls = []
    started = threading.Event()

    def fake_analyze(self, proj, st, restart=False):
        calls.append(restart)
        started.set()
        while not st.cancel.is_set() and not restart:
            time.sleep(0.01)
        if st.cancel.is_set():
            raise J.Cancelled()

    orig = J.JobManager._run_analyze
    J.JobManager._run_analyze = fake_analyze
    try:
        jm = J.JobManager(store)
        jm.submit(p.id, J.ANALYZE)
        assert started.wait(5)
        jm.cancel(p.id)
        jm.submit(p.id, J.ANALYZE, restart=True)
        assert _wait(lambda: calls == [False, True] and jm.state(p.id).kind is None)
        # a plain duplicate submit while running (no cancel) is still ignored
        assert jm.state(p.id).error is None
    finally:
        J.JobManager._run_analyze = orig


def test_overview_report_shows_ball_stats(tmp_path, app):
    from replay.gui.app import OverviewTab
    from replay.gui.projects import Project

    root = make_project(tmp_path / "p")
    rep = {"frames_processed": 40, "process_fps": 10.0,
           "calibration": {"valid_pct": 95.0, "warnings": []},
           "people": {"avg_tracked_per_calibrated_frame": 9.5, "avg_measured_per_calibrated_frame": 9.0,
                      "tracks_output": 12},
           "ball": {"detection_rate_pct": 40.0, "position_rate_pct": 88.5},
           "failed_ranges": {"no_calibration": [], "no_players_tracked": []}}
    (root / "results" / "report.json").write_text(json.dumps(rep), encoding="utf-8")
    (root / "results" / "tracking.json").write_text("{}", encoding="utf-8")

    class Win:
        pass

    tab = OverviewTab(Win())
    tab.p = Project(root)
    tab.refresh()
    html = tab.report.toHtml()
    assert "88.5%" in html and "milestone" not in html


def test_retry_import_offered_after_a_stopped_import(tmp_path, app, monkeypatch):
    from replay.gui import app as A

    from PySide6.QtCore import QSettings
    # never touch the user's real settings (registry): window geometry, last folders, API keys
    monkeypatch.setattr(A, "QSettings", lambda *a: QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat))
    src = tmp_path / "game.mp4"
    src.write_bytes(b"not really a video")
    win = A.MainWindow(tmp_path / "projects")
    try:
        p = win.store.create("game", {"filename": "game.mp4", "kind": "reference", "path": str(src)})
        win.jobs.state(p.id).message = "Import stopped. Press Retry import to start it again."
        win.show_project(p)
        win.refresh_project()
        assert win.btn_action.text() == "Retry import" and not win.btn_action.isHidden()
        submitted = []
        monkeypatch.setattr(win.jobs, "submit", lambda pid, kind, **kw: submitted.append((kind, kw)))
        win.job_action()
        assert submitted == [(A.IMPORT, {"source": str(src)})]
    finally:
        win.timer.stop()
        win.close()


# ---------------------------------------------------------------------- job processes, resume after a kill
def _jm(tmp_path):
    from replay.gui import jobs as J
    from replay.gui.projects import ProjectStore
    store = ProjectStore(tmp_path / "projects")
    p = store.create("x", {})
    p.results.mkdir()
    return J, J.JobManager(store), p


def test_job_process_result_progress_and_errors(tmp_path):
    J, jm, p = _jm(tmp_path)
    st = J.JobState()
    assert jm._in_process("jobs_child_fns:ok", p, st) == {"valid_pct": 90.0, "people": 9.5}
    assert st.progress == 0.5 and st.message == "half way"
    assert not J.marker(p, "jobs_child_fns:ok").exists() and not jm.procs
    with pytest.raises(J.JobError, match="court too small"):
        jm._in_process("jobs_child_fns:boom", p, J.JobState())
    assert "ValueError" in (p.results / "boom.log").read_text(encoding="utf-8")
    with pytest.raises(J.JobError, match="stopped unexpectedly .exit code 3"):
        jm._in_process("jobs_child_fns:crash", p, J.JobState())


def test_job_process_stop_kills_it(tmp_path):
    J, jm, p = _jm(tmp_path)
    st = J.JobState()
    threading.Timer(3.0, st.cancel.set).start()
    t = time.time()
    with pytest.raises(J.Cancelled):
        jm._in_process("jobs_child_fns:slow", p, st)
    assert time.time() - t < 20 and not jm.procs


def test_build_without_players_is_not_silent(tmp_path, monkeypatch):
    J, jm, p = _jm(tmp_path)
    monkeypatch.setattr(jm, "_in_process", lambda kind, p, st: {"valid_pct": 0.0, "people": 0.0})
    st = J.JobState()
    jm._run_build(p, st)
    assert "no players could be placed" in st.message


def test_resume_after_the_app_was_killed(tmp_path):
    from replay.gui import jobs as J
    from replay.gui.projects import ProjectStore
    store = ProjectStore(tmp_path / "projects")
    a = store.create("analysing", {})  # pass 1 half done when the power went off
    (a.root / "video.mp4").write_bytes(b"v")
    (a.results / "work").mkdir(parents=True)
    (a.results / "work" / "run.json").write_text("{}", encoding="utf-8")
    b = store.create("stopped", {})  # the user pressed Stop: stays stopped
    (b.root / "video.mp4").write_bytes(b"v")
    (b.results / "work").mkdir(parents=True)
    (b.results / "work" / "run.json").write_text("{}", encoding="utf-8")
    (b.results / J.PAUSED).write_text("analyze", encoding="utf-8")
    c = Project_with_build_marker(store, J)
    assert c.status()["replay_stale"] and c.status()["build_interrupted"]

    class NoRun(J.JobManager):  # record instead of running
        def _worker(self):
            pass
    jm = NoRun(store)
    got = dict((pid, k) for pid, k in jm.resume_interrupted())
    assert got == {a.id: J.ANALYZE, c.id: J.BUILD}


def Project_with_build_marker(store, J):
    c = store.create("building", {})
    (c.root / "video.mp4").write_bytes(b"v")
    work = c.results / "work"
    work.mkdir(parents=True)
    (work / "run.json").write_text("{}", encoding="utf-8")
    (work / "DONE").write_text("", encoding="utf-8")
    (c.root / "calibration.json").write_text(json.dumps({"video": "video.mp4", "frame_size": [1280, 720],
                                                         "keyframes": [{"time_s": 1.0, "points": {}}]}),
                                             encoding="utf-8")
    (c.results / "tracking.json").write_text("{}", encoding="utf-8")
    J.marker(c, J.BUILD).write_text("1", encoding="utf-8")
    return c


def test_user_stop_is_remembered_but_quitting_is_not(tmp_path):
    from replay.gui import jobs as J
    J_, jm, p = _jm(tmp_path)

    def fake(self, proj, st, restart=False):
        while not st.cancel.is_set():
            time.sleep(0.01)
        raise J.Cancelled()

    orig = J.JobManager._run_analyze
    J.JobManager._run_analyze = fake
    try:
        for quitting, paused in ((False, True), (True, False)):
            (p.results / J.PAUSED).unlink(missing_ok=True)
            jm.quitting = quitting
            jm.submit(p.id, J.ANALYZE)
            assert _wait(lambda: jm.state(p.id).kind == J.ANALYZE)
            jm.cancel(p.id)
            assert _wait(lambda: jm.state(p.id).kind is None)
            assert (p.results / J.PAUSED).exists() == paused
    finally:
        J.JobManager._run_analyze = orig


def test_timeline_playhead_moves_without_repainting_marks(tmp_path, app):
    """Playback moves the playhead 30x/s: the marks layer is cached (a whole game has ~550 marks)."""
    from PySide6.QtGui import QColor
    from replay.gui import timeline as T
    w = T.TimelineWidget()
    w.resize(900, 200)
    w.show()
    w.set_project(tmp_path)
    tr = w._target_track()
    for i in range(300):
        w.tl.add_mark(tr.id, i * 2.0, title=f"m{i}")
    w.set_duration(600)
    w.fit()
    calls = []
    orig = w.canvas._paint_static
    w.canvas._paint_static = lambda *a: (calls.append(1), orig(*a))
    img = w.canvas.grab().toImage()
    assert len(calls) == 1
    for k in range(5):
        w.set_time(100 + k, True)
        img = w.canvas.grab().toImage()
    assert len(calls) == 1  # playhead only: cached layer reused
    x = int(round(w.canvas.x_of(104)))
    assert QColor(img.pixel(x, T.RULER_H + 10)).red() > 200  # the red playhead is drawn where it is now
    w.add_mark(tr, 50.0)  # a real change repaints the layer
    w.canvas.grab()
    assert len(calls) == 2



def test_open_replay_file_lands_on_the_replay_tab(tmp_path, app, monkeypatch, qt_dialogs):
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QFileDialog
    from replay.gui import app as A
    from replay.gui.projects import ProjectStore
    from replay.report import BUILD_VERSION

    monkeypatch.setattr(A, "QSettings", lambda *a: QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat))
    src_store = ProjectStore(tmp_path / "other")
    root = make_project(tmp_path / "other" / "kouvot-abc123")
    (root / "results" / "tracking.json").write_text("{}", encoding="utf-8")
    rep = root / "results" / "report.json"  # current build: opening it must not start a rebuild
    (rep).write_text(json.dumps({"build_version": BUILD_VERSION}), encoding="utf-8")
    pkg = tmp_path / "game.aareplay"
    src_store.export(root.name, pkg)
    win = A.MainWindow(tmp_path / "projects")
    try:
        monkeypatch.setattr(QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (str(pkg), "")))
        win.open_package()
        assert win.current is not None and win.tabs.currentWidget() is win.replay_tab
        assert not win.jobs.state(win.current.id).queued and not win.jobs.state(win.current.id).kind
        assert not [d for d in qt_dialogs if d[0] == "warning"]
    finally:
        win.timer.stop()
        win.jobs.shutdown()
        win.close()


def test_partial_build_says_so(tmp_path, monkeypatch):
    J, jm, p = _jm(tmp_path)
    for rep, want in (({"valid_pct": 56.3, "people": 2.72}, "partial"), ({"valid_pct": 98.0, "people": 9.8}, "Replay ready: ")):
        monkeypatch.setattr(jm, "_in_process", lambda kind, p, st, r=rep: r)
        st = J.JobState()
        jm._run_build(p, st)
        assert want in st.message


def test_job_process_state_and_kwargs(tmp_path):
    J, jm, p = _jm(tmp_path)
    st = J.JobState()
    jm._in_process("jobs_child_fns:stateful", p, st, restart=True)
    assert (st.label, st.message, st.progress, st.started) == ("Preliminary analysis", "restart=True", 0.25, 123.0)


# ---------------------------------------------------------------------- import: Unicode names, ffmpeg failures
CLIP = __import__("pathlib").Path(__file__).resolve().parent.parent / "data" / "clip_30s.mp4"


@pytest.mark.skipif(not CLIP.exists(), reason="needs data/clip_30s.mp4")
def test_import_unicode_filename(tmp_path):
    import shutil
    from replay.gui import media
    src = tmp_path / "Michigan vs. Arizona： 2026 NCAA men's Final Four ｜ FULL REPLAY ä€.mp4"
    shutil.copy2(CLIP, src)
    info = media.make_proxy(src, tmp_path / "video.mp4")
    assert info.duration_s > 25 and (tmp_path / "video.mp4").exists()


def test_ffmpeg_failure_message_is_never_empty(monkeypatch, tmp_path):
    import io
    from replay.gui import media
    info = media.ProbeResult(5481.0, "av1", 1920, 1080, 29.97, True)
    monkeypatch.setattr(media, "probe", lambda src: info)

    class FakeProc:  # ffmpeg dying silently 15 min in (Windows access violation, nothing on stderr)
        def __init__(self, *a, **k):
            self.stdout = io.StringIO("out_time_us=900000000\nprogress=continue\n")
            self.stderr = io.StringIO("")
            self.returncode = None

        def wait(self):
            self.returncode = 0xC0000005
            return self.returncode

        def kill(self):
            pass
    monkeypatch.setattr(media.subprocess, "Popen", FakeProc)
    with pytest.raises(RuntimeError) as e:
        media.make_proxy(tmp_path / "x.mp4", tmp_path / "video.mp4")
    msg = str(e.value)
    assert "0xC0000005" in msg and "access violation" in msg and "0:15:00 of 1:31:21" in msg
    assert "Retry import" in msg
    m2 = media.failure_message(1, "[aac] Too many bits\nConversion failed!\n", 0, info)
    assert "Conversion failed!" in m2 and "before converting anything" in m2


def test_run_busy_returns_and_raises(app):
    from replay.gui.busy import run_busy
    assert run_busy(None, lambda a, b=0: a + b, 2, b=3, spinner_after=0.0) == 5
    with pytest.raises(ValueError, match="bad file"):
        run_busy(None, lambda: (_ for _ in ()).throw(ValueError("bad file")))


def test_clip_candidates_offer_one_fit_per_quarter():
    from replay import scoresheet as S
    sh = S.Scoresheet(kind="csv")
    run, a, b = [], 0, 0
    for i, (team, pts, q) in enumerate([("A", 2, 1), ("B", 2, 1), ("A", 2, 1), ("A", 2, 2), ("B", 3, 2), ("B", 2, 2),
                                         ("A", 2, 3), ("B", 2, 3), ("B", 2, 3)]):
        a, b = a + (pts if team == "A" else 0), b + (pts if team == "B" else 0)
        run.append(S.ScoreEvent(i + 1, team, pts, a, b, None, "fg2", q))
    sh.running = run
    replay = {"events": [{"type": "basket", "id": "x1", "t": 5.0, "team": "B"},
                         {"type": "basket", "id": "x2", "t": 9.0, "team": "B"}], "game": {"teams": {}}}
    cands = S.clip_candidates(sh, replay)
    assert [c["period"] for c in cands] == [1, 2, 3]  # one choice per quarter, in quarter order
    best = next(c for c in cands if c.get("best"))
    assert best["teams_ok"] == 2 and best["start_score"] == [8, 7] and best["period"] == 3  # B 2 + B 2 fits best


def test_scorebar_pin_is_applied_by_the_crosscheck(tmp_path):
    """A sure score-bar reading places a clip on the sheet without asking (the quarter picker stays hidden)."""
    from replay import scoresheet as S
    root = make_project(tmp_path / "p")
    sh = S.Scoresheet(kind="csv")
    run, a, b = [], 0, 0
    for i, (team, pts, q) in enumerate([("A", 2, 1), ("B", 2, 1), ("B", 2, 2), ("A", 1, 2), ("B", 2, 3)]):
        a, b = a + (pts if team == "A" else 0), b + (pts if team == "B" else 0)
        run.append(S.ScoreEvent(i + 1, team, pts, a, b, None, "fg2", q))
    sh.running = run
    replay = json.loads((root / "results" / "replay_data.json").read_text(encoding="utf-8"))
    gs = {"scorebar": {"summary": {"sure": True, "start": [2, 4], "end": [3, 6]}}}
    rep = S.crosscheck(root, sh, replay, gs, {})
    assert rep["start_source"] == "scorebar" and rep["pin"]["period"] == 2 and rep["pin"]["start_score"] == [2, 4]
    sh.game["replay_start_score"] = [0, 0]  # the user's own entry wins
    assert S.crosscheck(root, sh, replay, gs, {})["start_source"] == "user"


def test_startup_banner_from_the_self_check(tmp_path, app, monkeypatch):
    from PySide6.QtCore import QSettings
    from replay import selfcheck
    from replay.gui import app as A
    monkeypatch.setattr(A, "QSettings", lambda *a: QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat))
    monkeypatch.setattr(selfcheck, "startup_check", lambda d, **k: ["x"])
    monkeypatch.setattr(selfcheck, "banner_text", lambda checks: "Detector model file missing")
    win = A.MainWindow(tmp_path / "projects")
    try:
        A._startup_banner(win, tmp_path / "projects")
        assert not win.banner.isHidden() and "model" in win.banner_lbl.text()
        monkeypatch.setattr(selfcheck, "banner_text", lambda checks: "")
        win.banner.hide()
        A._startup_banner(win, tmp_path / "projects")
        assert win.banner.isHidden()  # nothing wrong: no banner
    finally:
        win.timer.stop()
        win.jobs.shutdown()
        win.close()


def test_settings_and_teams_usable_while_importing(tmp_path, app, monkeypatch):
    """9.08 regression: during a (long) import the tabs were hidden, so teams / analysis settings could not be set."""
    from PySide6.QtCore import QSettings
    from replay.gui import app as A
    monkeypatch.setattr(A, "QSettings", lambda *a: QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat))
    win = A.MainWindow(tmp_path / "projects")
    try:
        p = win.store.create("game", {"filename": "game.mp4", "kind": "reference", "path": str(tmp_path / "g.mp4")})
        st = win.jobs.state(p.id)
        st.kind, st.progress, st.message = A.IMPORT, 0.3, "Converting to a playable copy (30%)"
        win.show_project(p, 0)
        win.refresh_project()
        assert not win.tabs.isHidden() and not win.loading.isHidden()
        ov = win.overview_tab
        assert ov.isEnabled() and ov.save_btn.isEnabled()
        ov.name.setText("Final")
        ov.stride.setCurrentIndex(ov.stride.findData(2))
        ov.save()  # nothing analysed yet: just stored
        assert p.info()["name"] == "Final" and p.settings()["frame_stride"] == 2
    finally:
        win.jobs.state(p.id).kind = None
        win.timer.stop()
        win.jobs.shutdown()
        win.close()


def test_window_fits_a_laptop_screen(tmp_path, app, monkeypatch):
    """Every tab must fit 1280 x 720 (option rows wrap, side panels scroll)."""
    from PySide6.QtCore import QSettings
    from replay.gui import app as A
    monkeypatch.setattr(A, "QSettings", lambda *a: QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat))
    root = make_project(tmp_path / "projects" / "kouvot-abc123")
    win = A.MainWindow(tmp_path / "projects")
    try:
        win.show_project(win.store.get(root.name), 0)
        for i in range(win.tabs.count()):  # build every tab (some are built on first show)
            win.tabs.setCurrentIndex(i)
            app.processEvents()
        hint = win.minimumSizeHint()
        assert hint.width() <= 1280 and hint.height() <= 720, hint
    finally:
        win.timer.stop()
        win.jobs.shutdown()
        win.close()
