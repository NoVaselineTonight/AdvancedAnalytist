"""Coach quality-of-life GUI: Replay list (search, sort, status icons, sizes, duplicate), Getting-started page,
tooltips on every Overview / Court-fix control, friendly error texts."""

import json
import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

from fake_project import make_project  # noqa: E402


@pytest.fixture(scope="module")
def app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture
def win(app, tmp_path, monkeypatch):
    from PySide6.QtCore import QSettings

    from replay.gui import app as A
    monkeypatch.setattr(A, "QSettings", lambda *a: QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat))
    projects = tmp_path / "projects"
    projects.mkdir()
    for i, (pid, name, home) in enumerate([("kouvot-honka-aaa111", "Kouvot vs Honka", "Kouvot"),
                                          ("alpha-game-bbb222", "Alpha game", "Alpha"),
                                          ("zeta-cup-ccc333", "Zeta cup final", "Zeta")]):
        root = make_project(projects / pid)
        info = json.loads((root / "project.json").read_text(encoding="utf-8"))
        info.update(name=name, created=1000.0 + i)
        (root / "project.json").write_text(json.dumps(info), encoding="utf-8")
        (root / "meta.json").write_text(json.dumps({"home_team": home, "away_team": "Visitors"}), encoding="utf-8")
    (projects / "zeta-cup-ccc333" / "big.bin").write_bytes(b"x" * 300_000)
    w = A.MainWindow(projects)
    monkeypatch.setattr(w.jobs, "submit", lambda *a, **kw: None)
    yield w
    w.close()


def _names(w):
    from replay.gui.replay_list import NAME_ROLE
    return [w.list.item(i).data(NAME_ROLE) for i in range(w.list.count()) if not w.list.item(i).isHidden()]


def _wait_sizes(w, n):
    for _ in range(100):
        if len(w.replay_list.sizes.snapshot()) >= n:
            return
        time.sleep(0.02)


def test_sort_search_and_pin(win):
    ctl = win.replay_list
    assert _names(win) == ["Zeta cup final", "Alpha game", "Kouvot vs Honka"]  # newest first
    ctl.sort.setCurrentIndex(ctl.sort.findData("name"))
    assert _names(win) == ["Alpha game", "Kouvot vs Honka", "Zeta cup final"]
    ctl.sort.setCurrentIndex(ctl.sort.findData("oldest"))
    assert _names(win)[0] == "Kouvot vs Honka"
    _wait_sizes(win, 3)
    ctl.sort.setCurrentIndex(ctl.sort.findData("size"))
    assert _names(win)[0] == "Zeta cup final"  # has the extra 300 kB file
    assert win.settings.value("replay_sort") == "size"
    win.store.get("alpha-game-bbb222").update_info(pinned=True)
    win.refresh_list()
    assert _names(win)[0] == "Alpha game"  # pinned stays on top whatever the sort
    ctl.search.setText("kouvot")  # by team name / Replay name
    assert _names(win) == ["Kouvot vs Honka"]
    ctl.search.setText("final zeta")  # all words, any order
    assert _names(win) == ["Zeta cup final"]
    ctl.search.setText("")
    assert len(_names(win)) == 3


def test_rows_have_status_icon_size_and_tooltip(win):
    from replay.gui.replay_list import KIND_ROLE, SIZE_ROLE
    _wait_sizes(win, 3)
    win.refresh_list()
    it = win.list.item(0)
    assert it.data(KIND_ROLE) in {"busy", "ready", "attention", "error", "idle"}
    assert it.data(SIZE_ROLE).endswith(("KB", "MB", "B"))
    assert "on disk" in it.toolTip()
    win.jobs.state(it.data(256)).error = "PermissionError: [WinError 32] being used by another process"
    win.refresh_list()
    assert it.data(KIND_ROLE) == "error" and "What to do" not in it.toolTip()
    assert "Close other programs" in it.toolTip()
    win.jobs.state(it.data(256)).error = None


def test_status_kind_mapping():
    from replay.gui.replay_list import fmt_size, status_kind
    idle = {"error": None, "kind": None, "queued": []}
    assert status_kind("Replay ready", idle) == "ready"
    assert status_kind("Replay ready (outdated)", idle) == "attention"
    assert status_kind("Court not found - fix by hand", idle) == "attention"
    assert status_kind("Waiting for analysis", idle) == "idle"
    assert status_kind("Analysing 40%", {**idle, "kind": "analyze"}) == "busy"
    assert status_kind("Error", {**idle, "error": "x"}) == "error"
    assert fmt_size(None) == "…" and fmt_size(1536) == "1.5 KB" and fmt_size(3 << 30) == "3.0 GB"


def test_duplicate_copies_everything_but_job_markers(tmp_path):
    from replay.gui.projects import ProjectStore, folder_size
    store = ProjectStore(tmp_path / "projects")
    make_project(store.root / "kouvot-honka-aaa111")
    src = store.get("kouvot-honka-aaa111")
    src.update_info(pinned=True)
    (src.results / "JOB_build").write_text("{}", encoding="utf-8")
    (src.results / "PAUSED").write_text("", encoding="utf-8")
    (src.root / "timeline.json").write_text("{}", encoding="utf-8")
    seen = []
    new = store.duplicate(src.id, "Copy for testing", progress=seen.append)
    assert new.id != src.id and new.info()["name"] == "Copy for testing"
    assert new.info()["duplicated_from"] == src.id and not new.info().get("pinned")
    assert (new.root / "timeline.json").exists() and new.video.exists()
    assert not (new.results / "JOB_build").exists() and not (new.results / "PAUSED").exists()
    run = json.loads((new.results / "work" / "run.json").read_text(encoding="utf-8"))
    assert run["video"] == str(new.video.resolve())  # analysis points at the copy's own video
    assert seen and seen[-1] == 1.0
    assert folder_size(new.root) > 0 and len(store.list()) == 2


def test_getting_started_page_and_links(win, monkeypatch):
    from replay.gui import help_panel
    assert win.stack.widget(0) is win.help_page and win.stack.currentIndex() == 0
    html = help_panel.STEPS_HTML
    for href in ("action:import", "action:open", "action:scoresheet", "action:replay", "action:report",
                 "action:export"):
        assert href in html
    called = []
    monkeypatch.setattr(win, "import_video", lambda: called.append("import"))
    win.help_page._link("action:import")
    assert called == ["import"]
    help_menu = next(a.menu() for a in win.menuBar().actions() if a.text().replace("&", "") == "Help")
    assert help_menu.actions()[0].text() == "Getting started"
    monkeypatch.setattr(help_panel.QMessageBox, "information", lambda *a, **k: called.append("need"))
    help_panel.do_action(win, "scoresheet")  # nothing selected: explains what to do instead of failing
    assert called[-1] == "need"


def test_every_overview_and_calibrate_control_has_a_tooltip(win):
    from replay.gui.tooltips import missing
    assert missing(win.overview_tab) == []
    assert missing(win.calib_tab) == []


def test_friendly_errors():
    from replay.gui.friendly import explain, job_error_html, message_html
    assert "disk is full" in explain(OSError(28, "No space left on device")).headline
    assert "in use" in explain("PermissionError: [WinError 32] being used by another process").headline
    assert "GPU" in explain("onnxruntime DmlExecutionProvider failed").headline
    assert "different settings" in explain("Analysis was started with different settings").headline
    assert "newer version" in explain(ValueError("made with a newer version of AdvancedAnalytist")).headline
    assert explain("weird").headline == "Something went wrong."
    h = job_error_html("FileNotFoundError: C:/x/game.mp4 <b>")
    assert "What to do:" in h and "put it back" in h and "&lt;b&gt;" in h
    assert "What to do:" in message_html("Export failed.", OSError(28, "No space left on device"))
