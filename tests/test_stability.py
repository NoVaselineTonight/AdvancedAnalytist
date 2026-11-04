"""Stability: crash log + dialog, start-up self-check, log rotation / key redaction, the start .bat."""

from __future__ import annotations

import faulthandler
import multiprocessing as mp
import os
import sys
import threading
import time
from pathlib import Path

import pytest

from replay import logsafe, selfcheck
from replay.gui import crashguard

ROOT = Path(__file__).resolve().parents[1]
FAKE_KEY = "AIzaSyD" + "x" * 32


# ------------------------------------------------------------------------------------------- redaction
@pytest.mark.parametrize("text", [
    f"GET https://generativelanguage.googleapis.com/v1beta/models/x:generateContent?key={FAKE_KEY} 400",
    f"headers={{'x-goog-api-key': '{FAKE_KEY}'}}",
    "Authorization: Bearer abcdefghijklmnop1234",
    "api_key=supersecretvalue123",
    '{"apiKey": "supersecretvalue123"}',
    "token sk-abcdefghijklmnopqrstuvwxyz",
])
def test_redact_hides_keys(text):
    out = logsafe.redact(text)
    for secret in (FAKE_KEY, "abcdefghijklmnop1234", "supersecretvalue123", "sk-abcdefghijklmnopqrstuvwxyz"):
        assert secret not in out
    assert "***" in out


def test_redact_registered_secret_and_plain_text_untouched():
    logsafe.register_secret("jev_custom_secret_42")
    try:
        assert "jev_custom_secret_42" not in logsafe.redact("key was jev_custom_secret_42 !")
    finally:
        logsafe._SECRETS.discard("jev_custom_secret_42")
    plain = "Pass 1: frame 1200 / 54000 (2.2%), sorted(key=lambda r: r[0])"
    assert logsafe.redact(plain) == plain


def test_ai_modules_never_print_or_repr_the_key():
    from replay import ai_analysis, gemini_court

    assert FAKE_KEY not in repr(gemini_court.GeminiVision(FAKE_KEY))
    assert FAKE_KEY not in repr(ai_analysis.GeminiClient(FAKE_KEY, use_sdk=False))
    err = ai_analysis._friendly_error(RuntimeError(f"boom ?key={FAKE_KEY}"), FAKE_KEY, "m")
    assert FAKE_KEY not in err
    # every print() in these modules goes to app.log; none may format a key / client / request
    for mod in (ai_analysis, gemini_court):
        src = Path(mod.__file__).read_text(encoding="utf-8")
        for line in src.splitlines():
            if "print(" in line:
                assert not any(w in line for w in ("_key", "api_key", "headers", "req", "client")), line


# ------------------------------------------------------------------------------------------- rotation
def test_rotate_keeps_last_five(tmp_path):
    log = tmp_path / "app.log"
    for i in range(8):
        log.write_text(f"gen {i}" + "x" * 100, encoding="utf-8")
        assert logsafe.rotate(log, max_bytes=50, backups=5)
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == [f"app.log.{i}" for i in range(1, 6)]
    assert (tmp_path / "app.log.1").read_text(encoding="utf-8").startswith("gen 7")
    assert (tmp_path / "app.log.5").read_text(encoding="utf-8").startswith("gen 3")
    assert not logsafe.rotate(tmp_path / "missing.log")


def test_rotating_stream_rolls_over_and_redacts(tmp_path):
    s = logsafe.RotatingStream(tmp_path / "app.log", max_bytes=1000, backups=5)
    try:
        for i in range(200):
            print(f"line {i:04d} key={FAKE_KEY} " + "." * 40, file=s)
        s.flush()
    finally:
        s.close()
    files = sorted(tmp_path.iterdir())
    assert len(files) == 6  # app.log + 5 backups, older ones dropped
    assert all(f.stat().st_size < 1200 for f in files)
    text = "".join(f.read_text(encoding="utf-8") for f in files)
    assert FAKE_KEY not in text and "line 0199" in text
    s.write("after close is ignored, not an exception\n")


def test_default_limits_are_5_by_5mb():
    assert logsafe.MAX_BYTES == 5 * 1024 * 1024 and logsafe.BACKUPS == 5


# ------------------------------------------------------------------------------------------- crash guard
def test_launch_detection():
    main = str(ROOT / "replay" / "__main__.py")
    assert crashguard.is_gui_launch([main, "gui"])
    assert crashguard.is_gui_launch([main, "gui", "--projects", "D:/p"])
    assert not crashguard.is_gui_launch([main, "process", "--video", "x"])
    assert not crashguard.is_gui_launch(["pytest", "gui"])
    assert crashguard.projects_dir_from_argv([main, "gui", "--projects", "D:/p"]) == Path("D:/p")
    assert crashguard.projects_dir_from_argv([main, "gui", "--projects=E:/q"]) == Path("E:/q")
    assert crashguard.projects_dir_from_argv([main, "gui"]) == ROOT / "projects"


def test_not_installed_under_pytest():
    assert sys.excepthook is not crashguard._excepthook
    assert threading.excepthook is not crashguard._thread_excepthook


@pytest.fixture
def guarded(tmp_path, monkeypatch):
    """crashguard.install() into tmp_path; every global it touches is restored afterwards."""
    monkeypatch.setattr(sys, "excepthook", sys.excepthook)
    monkeypatch.setattr(threading, "excepthook", threading.excepthook)
    monkeypatch.setattr(sys, "stdout", sys.stdout)
    monkeypatch.setattr(sys, "stderr", sys.stderr)
    monkeypatch.setattr(faulthandler, "enable", lambda *a, **k: None)
    monkeypatch.delenv(crashguard.ENV, raising=False)
    monkeypatch.setattr(crashguard, "_state", dict(crashguard._state))
    shown = []
    monkeypatch.setattr(crashguard, "_show_dialog", lambda text, path: shown.append((text, path)))
    crashguard.install(tmp_path, gui=True)
    yield tmp_path, shown
    f = crashguard._state.get("fault_file")
    if f:
        f.close()


def _raise_and_hook(msg: str):
    try:
        raise ValueError(msg)
    except ValueError:
        sys.excepthook(*sys.exc_info())


def test_gui_exception_logged_with_timestamp_and_dialog(guarded):
    tmp, shown = guarded
    assert os.environ[crashguard.ENV] == str(tmp)  # job processes inherit it
    _raise_and_hook(f"bad thing ?key={FAKE_KEY}")
    log = (tmp / "crash.log").read_text(encoding="utf-8")
    assert "uncaught exception in GUI thread" in log and "ValueError: bad thing" in log
    assert time.strftime("%Y-%m-%d") in log
    assert FAKE_KEY not in log
    assert len(shown) == 1 and shown[0][1] == tmp / "crash.log" and "ValueError" in shown[0][0]
    _raise_and_hook(f"bad thing ?key={FAKE_KEY}")  # the same error right again: logged, no second dialog
    assert len(shown) == 1
    assert (tmp / "crash.log").read_text(encoding="utf-8").count("ValueError: bad thing") == 2


def test_thread_exception_logged_and_forwarded_to_gui_thread(guarded):
    tmp, shown = guarded
    emitted = []

    class _B:
        class crashed:  # noqa: N801 - mimics the Qt signal
            @staticmethod
            def emit(text, path):
                emitted.append((text, path))

    crashguard._state["bridge"] = _B()
    t = threading.Thread(target=lambda: 1 / 0, name="worker-x")
    t.start()
    t.join()
    log = (tmp / "crash.log").read_text(encoding="utf-8")
    assert "thread worker-x" in log and "ZeroDivisionError" in log
    assert not shown and len(emitted) == 1  # the dialog opens in the GUI thread, not in the worker


def test_record_for_caught_job_errors(guarded):
    tmp, _ = guarded
    try:
        raise RuntimeError("pass 1 failed")
    except RuntimeError as e:
        text = crashguard.record("job analyze", e)
    assert "job analyze" in text and "RuntimeError: pass 1 failed" in (tmp / "crash.log").read_text("utf-8")


def test_pythonw_streams_go_to_rotating_app_log(tmp_path, monkeypatch, guarded):
    tmp, _ = guarded
    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "stderr", None)
    crashguard.install(tmp, gui=True)
    assert isinstance(sys.stdout, logsafe.RotatingStream) and sys.stderr is sys.stdout
    print(f"hello key={FAKE_KEY}")
    sys.stdout.flush()
    text = (tmp / "app.log").read_text(encoding="utf-8")
    sys.stdout.close()
    assert "hello key=" in text and FAKE_KEY not in text


def test_friendly_dialog_text_and_offscreen_never_blocks(tmp_path):
    msg = crashguard.friendly_message(tmp_path / "crash.log")
    assert "crash.log" in msg and "Copy details" in msg
    t0 = time.time()
    crashguard._show_dialog("Traceback ...", tmp_path / "crash.log")  # QT_QPA_PLATFORM=offscreen in tests
    assert time.time() - t0 < 5


def _child_thread_crash():
    import replay.gui  # noqa: F401 - what a job process imports (replay.gui.jobs); installs the guard
    from replay.gui import crashguard as cg

    assert cg._state["installed"] and not cg._state["gui"]
    t = threading.Thread(target=lambda: [][1], name="child-worker")
    t.start()
    t.join()


def test_job_process_writes_crash_log(tmp_path, monkeypatch):
    monkeypatch.setenv(crashguard.ENV, str(tmp_path))
    ctx = mp.get_context("spawn")
    p = ctx.Process(target=_child_thread_crash)
    p.start()
    p.join(60)
    assert p.exitcode == 0
    log = (tmp_path / "crash.log").read_text(encoding="utf-8")
    assert "job process" in log and "IndexError" in log and "child-worker" in log


# ------------------------------------------------------------------------------------------- self-check
def test_selfcheck_fast_and_clean_on_this_machine():
    t0 = time.perf_counter()
    checks = selfcheck.run_checks()
    assert time.perf_counter() - t0 < 1.0
    assert {c.name for c in checks} >= {"gpu", "models", "ffmpeg", "disk"}
    assert all(isinstance(m, str) for m in selfcheck.messages(checks))


def test_selfcheck_reports_missing_models_and_low_disk(tmp_path, monkeypatch):
    import shutil as sh

    from collections import namedtuple

    du = namedtuple("du", "total used free")
    monkeypatch.setattr(sh, "disk_usage", lambda p: du(100 * 2 ** 30, 95 * 2 ** 30, 5 * 2 ** 30))
    checks = selfcheck.run_checks(tmp_path / "projects", root=tmp_path)
    msgs = selfcheck.messages(checks)
    assert any("rfdetr-medium.onnx" in m and m.startswith("Error") for m in msgs)
    assert any("5.0 GB free" in m for m in msgs)
    banner = selfcheck.banner_text(checks)
    assert banner.startswith("Model file missing: models/rfdetr-medium") and "more start-up warnings" in banner
    monkeypatch.setattr(sh, "disk_usage", lambda p: du(100 * 2 ** 30, 99 * 2 ** 30, 1 * 2 ** 30))
    assert selfcheck.check_disk(tmp_path).level == selfcheck.ERROR


def test_selfcheck_all_ok_gives_empty_banner_and_ok_log_line(tmp_path, monkeypatch):
    for rel, _, _ in selfcheck.model_files():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_bytes(b"x")
    checks = selfcheck.check_models(tmp_path)
    assert [c.level for c in checks] == ["ok"]
    assert selfcheck.banner_text(checks) == ""
    import io

    buf = io.StringIO()
    selfcheck.log_checks(checks, buf)
    assert "[self-check] OK" in buf.getvalue()


def test_selfcheck_gpu_package_detection(tmp_path, monkeypatch):
    (tmp_path / "onnxruntime_directml-1.24.4.dist-info").mkdir()
    assert selfcheck._ort_dists(tmp_path) == {"onnxruntime-directml": "1.24.4"}
    (tmp_path / "onnxruntime-1.20.0.dist-info").mkdir()
    assert set(selfcheck._ort_dists(tmp_path)) == {"onnxruntime-directml", "onnxruntime"}


# ------------------------------------------------------------------------------------------- start .bat
def test_bat_uses_venv_pythonw_and_working_directory():
    bat = (ROOT / "AdvancedAnalytist.bat").read_text(encoding="utf-8")
    line = next(ln for ln in bat.splitlines() if ln.lower().startswith("start "))
    assert '/D "%~dp0."' in line
    assert r'"%~dp0.venv\Scripts\pythonw.exe" -m replay gui' in line
