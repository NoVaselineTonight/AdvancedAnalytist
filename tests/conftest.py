import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# One QApplication (widgets) for the whole test session, created before any test can create a plain
# QGuiApplication: a QWidget made under a QGuiApplication makes Qt abort the process silently (exit 127, no
# traceback) - that killed the full suite in test_gui_qa.py after test_confidence_ui.py had run.
import os  # noqa: E402

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
try:
    from PySide6.QtCore import QtMsgType, qInstallMessageHandler
    from PySide6.QtWidgets import QApplication

    _QAPP = QApplication.instance() or QApplication([])

    def _qt_messages(mode, context, message):  # Qt's fatal messages otherwise vanish on Windows
        if mode in (QtMsgType.QtCriticalMsg, QtMsgType.QtFatalMsg):
            sys.stderr.write(f"Qt {mode.name}: {message}\n")
            sys.stderr.flush()

    qInstallMessageHandler(_qt_messages)
except ImportError:  # no GUI toolkit: GUI tests skip/fail on their own
    pass

# No modal message box may ever block a test (offscreen nobody can click it: the run hangs for good). Every
# QMessageBox static call is answered at once - Yes / Ok - and recorded in ``qt_dialogs`` for tests to inspect.
try:
    import pytest
    from PySide6.QtWidgets import QMessageBox

    QT_DIALOGS: list[tuple[str, str]] = []

    def _answer(kind):
        def fn(*args, **kwargs):
            text = next((a for a in args[1:] if isinstance(a, str)), "")
            if len(args) > 2 and isinstance(args[2], str):
                text = args[2]
            QT_DIALOGS.append((kind, text))
            return QMessageBox.StandardButton.Yes if kind == "question" else QMessageBox.StandardButton.Ok
        return staticmethod(fn)

    @pytest.fixture(autouse=True)
    def _no_blocking_message_boxes(monkeypatch):
        QT_DIALOGS.clear()
        for kind in ("question", "warning", "information", "critical"):
            monkeypatch.setattr(QMessageBox, kind, _answer(kind))
        monkeypatch.setattr(QMessageBox, "about", _answer("about"))
        yield

    @pytest.fixture
    def qt_dialogs():
        return QT_DIALOGS
except ImportError:
    pass
