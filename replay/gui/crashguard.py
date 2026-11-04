"""Crash safety for the desktop app: no silent exits.

Every uncaught exception - GUI thread (``sys.excepthook``, also what PySide6 calls for an exception in a slot),
other threads (``threading.excepthook``) and job processes - is appended to ``projects/crash.log`` with a
timestamp (API keys redacted). In the GUI process a friendly dialog appears with "Copy details"; the app keeps
running when it can. Native crashes (access violations in Qt / ffmpeg / DirectML) are caught by ``faulthandler``
into the same file, also in the job processes.

Under pythonw (no console) stdout/stderr go to ``projects/app.log`` through ``logsafe.RotatingStream``
(5 x 5 MB, keys redacted). ``crash.log`` is rotated the same way on start.

Wiring: ``replay/gui/__init__.py`` calls ``auto_install()``. That is a no-op except
  - in the GUI process (``python -m replay gui ...``): full install, and ``AA_CRASH_DIR`` is put in the
    environment so the job processes (multiprocessing spawn) find the crash log;
  - in a job process started by such a GUI: crash.log + faulthandler, no dialog.
Tests and the CLI commands are untouched. ``record(where, exc)`` lets job code log an exception it caught.
"""

from __future__ import annotations

import datetime as _dt
import os
import sys
import threading
import time
import traceback
from pathlib import Path

from replay import logsafe

ENV = "AA_CRASH_DIR"
CRASH_LOG = "crash.log"
APP_LOG = "app.log"

_state = {"dir": None, "gui": False, "installed": False, "in_dialog": False, "last": ("", 0.0),
          "fault_file": None, "bridge": None}
_lock = threading.Lock()


# ---------------------------------------------------------------------------------------------- paths / setup
def default_projects_dir() -> Path:
    """Same default as ``replay gui --projects`` (replay/cli.py)."""
    return Path(__file__).resolve().parents[2] / "projects"


def projects_dir_from_argv(argv: list[str]) -> Path:
    for i, a in enumerate(argv):
        if a == "--projects" and i + 1 < len(argv):
            return Path(argv[i + 1])
        if a.startswith("--projects="):
            return Path(a.split("=", 1)[1])
    return default_projects_dir()


def is_gui_launch(argv: list[str]) -> bool:
    """``python -m replay gui`` / ``pythonw -m replay gui`` (argv[0] is replay/__main__.py)."""
    if len(argv) < 2 or argv[1] != "gui":
        return False
    a0 = Path(argv[0]).as_posix().lower()
    return a0.endswith("replay/__main__.py") or a0.endswith("/replay") or a0 in ("replay", "-m")


def is_child_process() -> bool:
    """A multiprocessing child. While it unpickles its target (which imports replay.gui.jobs) parent_process()
    is still None and sys.argv is the parent's, so look at the real command line too."""
    import multiprocessing as mp

    return "--multiprocessing-fork" in getattr(sys, "orig_argv", ()) or mp.parent_process() is not None


def crash_log_path() -> Path | None:
    d = _state["dir"] or os.environ.get(ENV)
    return Path(d) / CRASH_LOG if d else None


def auto_install() -> None:
    """Called once on ``import replay.gui``; decides what (if anything) to install. Never raises."""
    try:
        if _state["installed"]:
            return
        if is_child_process():
            if os.environ.get(ENV):
                install(Path(os.environ[ENV]), gui=False)
            return
        if is_gui_launch(sys.argv):
            install(projects_dir_from_argv(sys.argv), gui=True)
    except Exception:  # noqa: BLE001 - the crash guard itself must never stop the app from starting
        pass


def install(projects_dir: Path, gui: bool = True) -> None:
    projects_dir = Path(projects_dir)
    try:
        projects_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    _state.update(dir=str(projects_dir), gui=gui, installed=True)
    os.environ[ENV] = str(projects_dir)
    if gui:
        logsafe.rotate(projects_dir / CRASH_LOG)
        if sys.stdout is None or sys.stderr is None:  # pythonw: app.log, rotated + redacted
            try:
                stream = logsafe.RotatingStream(projects_dir / APP_LOG)
                sys.stdout = sys.stdout or stream
                sys.stderr = sys.stderr or stream
            except OSError:
                pass
    try:  # native crashes: stack of every thread into crash.log (app.py's run() re-enables it on the same file)
        import faulthandler

        f = open(projects_dir / CRASH_LOG, "a", encoding="utf-8")
        _state["fault_file"] = f
        faulthandler.enable(f, all_threads=True)
    except (OSError, ValueError, RuntimeError):
        pass
    sys.excepthook = _excepthook
    threading.excepthook = _thread_excepthook
    if gui:
        _make_bridge()


# ---------------------------------------------------------------------------------------------- recording
def format_report(exc_type, exc, tb, where: str = "GUI") -> str:
    stamp = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    body = "".join(traceback.format_exception(exc_type, exc, tb))
    head = f"===== {stamp}  uncaught exception in {where} (pid {os.getpid()}) ====="
    return logsafe.redact(f"{head}\n{body.rstrip()}\n")


def write_crash_log(text: str) -> Path | None:
    path = crash_log_path()
    if path is None:
        return None
    try:
        with _lock, open(path, "a", encoding="utf-8") as f:
            f.write(text + "\n")
        return path
    except OSError:
        return None


def record(where: str, exc: BaseException) -> str:
    """Log an exception the caller caught (e.g. a job that failed); returns the report text."""
    text = format_report(type(exc), exc, exc.__traceback__, where)
    write_crash_log(text)
    return text


def _excepthook(exc_type, exc, tb) -> None:
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc, tb)
        return
    where = "GUI thread" if _state["gui"] else f"job process {_proc_name()}"
    text = format_report(exc_type, exc, tb, where)
    path = write_crash_log(text)
    _echo(text)
    if _state["gui"]:
        _show(text, path)


def _thread_excepthook(args) -> None:
    if args.exc_type is SystemExit:
        return
    name = getattr(args.thread, "name", "?")
    where = f"thread {name}" if _state["gui"] else f"job process {_proc_name()}, thread {name}"
    text = format_report(args.exc_type, args.exc_value, args.exc_traceback, where)
    path = write_crash_log(text)
    _echo(text)
    if _state["gui"]:
        _show(text, path)


def _proc_name() -> str:
    try:
        import multiprocessing as mp

        return mp.current_process().name
    except Exception:  # noqa: BLE001
        return "?"


def _echo(text: str) -> None:
    try:
        if sys.stderr is not None:
            sys.stderr.write(text)
            sys.stderr.flush()
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------------------------------------- dialog
def _make_bridge() -> None:
    """A QObject living in the GUI thread: other threads emit, the dialog opens in the GUI thread."""
    try:
        from PySide6.QtCore import QObject, Signal

        class _Bridge(QObject):
            crashed = Signal(str, str)

        b = _Bridge()
        b.crashed.connect(lambda t, p: _show_dialog(t, Path(p) if p else None))
        _state["bridge"] = b
    except Exception:  # noqa: BLE001 - no Qt: log only
        _state["bridge"] = None


def _show(text: str, path: Path | None) -> None:
    sig = text.splitlines()[-1] if text else ""
    now = time.monotonic()
    last_sig, last_t = _state["last"]
    if sig == last_sig and now - last_t < 10.0:  # same error again (e.g. every repaint): logged, no new dialog
        return
    _state["last"] = (sig, now)
    if threading.current_thread() is threading.main_thread():
        _show_dialog(text, path)
    elif _state["bridge"] is not None:
        try:
            _state["bridge"].crashed.emit(text, str(path or ""))
        except Exception:  # noqa: BLE001
            pass


def friendly_message(path: Path | None) -> str:
    where = f"The details were saved to:\n{path}" if path else "The details could not be saved to a file."
    return ("Something went wrong inside AdvancedAnalytist.\n\n"
            "Your projects and finished analyses are safe. The app tries to keep running; if something looks "
            "wrong, save your work and restart it.\n\n"
            f"{where}\n\nPress \"Copy details\" and send them to the developer.")


def _show_dialog(text: str, path: Path | None) -> None:
    if _state["in_dialog"]:
        return
    _state["in_dialog"] = True
    try:
        from PySide6.QtWidgets import (QApplication, QDialog, QDialogButtonBox, QLabel, QPlainTextEdit,
                                       QPushButton, QVBoxLayout)

        app = QApplication.instance()
        if app is None:  # crashed before/after the window existed: still say so (pythonw has no console)
            app = QApplication(sys.argv[:1])
        if os.environ.get("QT_QPA_PLATFORM") == "offscreen":  # tests / headless: never block
            return
        dlg = QDialog()
        dlg.setWindowTitle("AdvancedAnalytist - unexpected error")
        lay = QVBoxLayout(dlg)
        msg = QLabel(friendly_message(path))
        msg.setWordWrap(True)
        lay.addWidget(msg)
        details = QPlainTextEdit(text)
        details.setReadOnly(True)
        details.setMinimumSize(640, 220)
        lay.addWidget(details)
        box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        copy = QPushButton("Copy details")
        box.addButton(copy, QDialogButtonBox.ButtonRole.ActionRole)

        def _copy() -> None:
            QApplication.clipboard().setText(text)
            copy.setText("Copied")

        copy.clicked.connect(_copy)
        box.rejected.connect(dlg.reject)
        box.accepted.connect(dlg.accept)
        lay.addWidget(box)
        dlg.exec()
    except Exception:  # noqa: BLE001 - the error is in crash.log already
        pass
    finally:
        _state["in_dialog"] = False
