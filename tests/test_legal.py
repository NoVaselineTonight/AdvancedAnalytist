"""Terms of Use / Privacy Policy: texts ship with the app, first start asks once per version, decline quits."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QSettings, QTimer
from PySide6.QtWidgets import QApplication, QDialog

from replay import legal
from replay.gui import legal_dialog as L


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def settings(tmp_path):
    return QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat)


def test_texts_exist_and_cover_the_key_points():
    t, p = legal.terms_text(), legal.privacy_text()
    for needle in ("AS IS", "Limitation of liability", "AI output can be wrong", "Costs are yours", "controller",
                   "Indemnity", "Finland", "mandatory rights"):
        assert needle in t, needle
    for needle in ("receives none of your data", "Still images from your video", "never sent", "controller",
                   "DPAPI", "no telemetry"):
        assert needle in p, needle


def _run_dialog(qapp, settings, *, tick: bool, agree: bool) -> bool:
    def act():
        dlg = next(w for w in qapp.topLevelWidgets() if isinstance(w, L.LegalDialog) and w.isVisible())
        assert not dlg.agree.isEnabled()  # cannot agree without ticking the box
        if tick:
            dlg.check.setChecked(True)
        (dlg.agree.click if agree and tick else dlg.reject)()
    QTimer.singleShot(50, act)
    return L.ensure_accepted(None, settings)


def test_first_start_asks_and_remembers(qapp, settings):
    assert not L.is_accepted(settings)
    assert _run_dialog(qapp, settings, tick=True, agree=True)
    assert L.is_accepted(settings)
    assert settings.value(L.TIME_KEY)
    assert L.ensure_accepted(None, settings)  # second start: no dialog (it would block here otherwise)


def test_decline_returns_false_and_stores_nothing(qapp, settings):
    assert not _run_dialog(qapp, settings, tick=False, agree=False)
    assert not L.is_accepted(settings)


def test_new_version_asks_again(qapp, settings, monkeypatch):
    L.record_acceptance(settings)
    monkeypatch.setattr(legal, "LEGAL_VERSION", "99.0")
    assert not L.is_accepted(settings)


def test_read_only_view_builds(qapp):
    d = L.LegalDialog(None, require_accept=False)
    assert d.tabs.count() == 2
    assert d.result() == QDialog.DialogCode.Rejected
    d.deleteLater()
