"""Run a slow call (file probing, parsing, cross-checks) off the GUI thread while the window keeps painting.

:func:`run_busy` shows the busy cursor (and, after ``spinner_after`` seconds, a small "working" dialog with a
moving bar), keeps processing paint / timer events but no user input (no re-entrant clicks), and returns the result
or re-raises the worker's exception in the GUI thread.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable

from PySide6.QtCore import QEventLoop, Qt
from PySide6.QtWidgets import QApplication, QProgressDialog, QWidget


def run_busy(parent: QWidget | None, fn: Callable[..., Any], *args: Any, text: str = "Working…",
             spinner_after: float = 0.4, **kwargs: Any) -> Any:
    state: dict[str, Any] = {"done": False, "result": None, "error": None}

    def work() -> None:
        try:
            state["result"] = fn(*args, **kwargs)
        except BaseException as e:  # noqa: BLE001 - re-raised in the GUI thread
            state["error"] = e
        state["done"] = True

    th = threading.Thread(target=work, daemon=True, name="gui-busy")
    QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
    dlg = None
    t0 = time.perf_counter()
    try:
        th.start()
        while not state["done"]:
            QApplication.processEvents(QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents, 30)
            if dlg is None and time.perf_counter() - t0 > spinner_after:
                dlg = QProgressDialog(text, None, 0, 0, parent)  # 0..0 = moving "busy" bar
                dlg.setWindowModality(Qt.WindowModality.WindowModal)
                dlg.setMinimumDuration(0)
                dlg.setAutoClose(False)
                dlg.show()
            th.join(0.01)
    finally:
        if dlg is not None:
            dlg.close()
            dlg.deleteLater()
        QApplication.restoreOverrideCursor()
    if state["error"] is not None:
        raise state["error"]
    return state["result"]
