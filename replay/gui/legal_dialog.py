"""First-start Terms of Use / Privacy Policy dialog.

``ensure_accepted()`` runs before the main window opens. If the user has not accepted the current
``replay.legal.LEGAL_VERSION`` it shows both texts and the app only starts after "I agree"; "Decline" quits.
The acceptance (version + UTC time) is stored in QSettings. ``show_legal()`` re-opens the texts read-only
(Help -> Terms of Use & Privacy).
"""

from __future__ import annotations

from datetime import datetime, timezone

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import (QCheckBox, QDialog, QDialogButtonBox, QLabel, QPushButton, QTabWidget, QTextBrowser,
                               QVBoxLayout, QWidget)

from replay import legal

VERSION_KEY = "legal/accepted_version"
TIME_KEY = "legal/accepted_utc"


def _settings() -> QSettings:
    return QSettings("AdvancedAnalytist", "AdvancedAnalytist")


def is_accepted(settings: QSettings | None = None) -> bool:
    s = settings or _settings()
    return str(s.value(VERSION_KEY, "") or "") == legal.LEGAL_VERSION


def record_acceptance(settings: QSettings | None = None) -> None:
    s = settings or _settings()
    s.setValue(VERSION_KEY, legal.LEGAL_VERSION)
    s.setValue(TIME_KEY, datetime.now(timezone.utc).isoformat(timespec="seconds"))
    s.sync()


def _browser(md: str) -> QTextBrowser:
    b = QTextBrowser()
    b.setOpenExternalLinks(True)
    b.setMarkdown(md)
    return b


class LegalDialog(QDialog):
    def __init__(self, parent: QWidget | None = None, *, require_accept: bool = True):
        super().__init__(parent)
        self.setWindowTitle("AdvancedAnalytist – Terms of Use and Privacy Policy")
        self.resize(820, 680)
        lay = QVBoxLayout(self)
        if require_accept:
            intro = QLabel("<b>Before you start, please read the Terms of Use and the Privacy Policy.</b><br>"
                           "In short: the app runs on your PC and sends nothing to the author. Tracking, statistics "
                           "and AI output are estimates that can be wrong and must be checked. Optional AI features "
                           "send data to Google / TypeSafe AI under your own key, contract and costs. You are "
                           "responsible for having the right to film and analyse the players (GDPR, minors).")
            intro.setWordWrap(True)
            lay.addWidget(intro)
        self.tabs = QTabWidget()
        self.tabs.addTab(_browser(legal.terms_text()), "Terms of Use")
        self.tabs.addTab(_browser(legal.privacy_text()), "Privacy Policy")
        lay.addWidget(self.tabs, 1)
        if require_accept:
            self.check = QCheckBox("I have read and agree to the Terms of Use and the Privacy Policy, "
                                   "including the AI and limitation-of-liability sections.")
            lay.addWidget(self.check)
            box = QDialogButtonBox()
            self.agree = QPushButton("I agree")
            self.agree.setEnabled(False)
            self.agree.setDefault(True)
            decline = QPushButton("Decline and quit")
            box.addButton(self.agree, QDialogButtonBox.ButtonRole.AcceptRole)
            box.addButton(decline, QDialogButtonBox.ButtonRole.RejectRole)
            self.check.toggled.connect(self.agree.setEnabled)
            self.agree.clicked.connect(self.accept)
            decline.clicked.connect(self.reject)
        else:
            box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
            box.rejected.connect(self.reject)
            s = _settings()
            when = str(s.value(TIME_KEY, "") or "")
            if when:
                lay.addWidget(QLabel(f"<span style='color:gray'>You accepted version "
                                     f"{s.value(VERSION_KEY, '')} on {when[:10]}.</span>"))
        lay.addWidget(box)


def ensure_accepted(parent: QWidget | None = None, settings: QSettings | None = None) -> bool:
    """True if the current terms are accepted (asks once per LEGAL_VERSION); False = user declined."""
    s = settings or _settings()
    if is_accepted(s):
        return True
    dlg = LegalDialog(parent, require_accept=True)
    if dlg.exec() == QDialog.DialogCode.Accepted and dlg.check.isChecked():
        record_acceptance(s)
        return True
    return False


def show_legal(parent: QWidget | None = None) -> None:
    LegalDialog(parent, require_accept=False).exec()
