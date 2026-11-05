"""HelpIcon: the small "?" that holds explanations, and proof that every text moved behind one is still reachable."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

from fake_project import make_project  # noqa: E402


@pytest.fixture(scope="module")
def app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_helpicon_basics(app):
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QWidget
    from replay.gui.helpicon import HelpIcon
    host = QWidget()
    ic = HelpIcon("Averages each position <b>over time</b>.", "Smoothing", host)
    host.show()
    assert ic.text() == "?" and ic.width() == ic.height() and ic.focusPolicy() == Qt.FocusPolicy.StrongFocus
    assert ic.accessibleName() == "Help: Smoothing"
    assert ic.accessibleDescription() == "Averages each position over time."  # plain text for screen readers
    assert "over time" in ic.toolTip()  # hover shows the text
    assert ic.help_text().startswith("Averages") and ic.plain_text() == "Averages each position over time."


def test_helpicon_click_and_keyboard_open_and_close(app):
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QKeyEvent
    from PySide6.QtCore import QEvent
    from PySide6.QtWidgets import QWidget
    from replay.gui.helpicon import HelpIcon
    host = QWidget()
    host.resize(300, 200)
    ic = HelpIcon("Some help text", "Topic", host)
    ic.move(10, 10)
    host.show()
    assert not ic.is_open()
    ic.click()
    assert ic.is_open() and ic.popup().label.text() == "Some help text"
    ic.click()  # click again closes
    assert not ic.is_open()
    ic.setFocus()
    ic.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_F1, Qt.KeyboardModifier.NoModifier))
    assert ic.is_open()
    ic.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier))
    assert not ic.is_open()
    ic.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Return, Qt.KeyboardModifier.NoModifier))
    assert ic.is_open()
    ic.hide_popup()


def test_helpicon_empty_text_hides_and_later_shows(app):
    from PySide6.QtWidgets import QWidget
    from replay.gui.helpicon import HelpIcon
    host = QWidget()
    ic = HelpIcon("", "Dynamic", host)
    host.show()
    assert ic.isHidden() and not ic.isVisible()
    ic.show_popup()  # nothing to show: no popup
    assert not ic.is_open()
    ic.set_help("Now there is text")
    assert ic.isVisible() and "Now there is text" in ic.toolTip()
    ic.set_help("")
    assert not ic.isVisible()


def test_helpicon_link_signal_and_helpers(app):
    from PySide6.QtWidgets import QLabel, QWidget
    from replay.gui.helpicon import HelpIcon, help_contains, help_row, help_texts, plain, titled
    host = QWidget()
    from PySide6.QtWidgets import QVBoxLayout
    v = QVBoxLayout(host)
    row = help_row(QLabel("Visible"), "<a href='x:1'>click</a> hidden detail", "Row")
    v.addWidget(row)
    v.addWidget(titled("<b>Title</b>", "Title help", ""))
    host.show()
    got = []
    row.help_icon.linkActivated.connect(got.append)
    row.help_icon.show_popup()
    row.help_icon.popup().label.linkActivated.emit("x:1")
    assert got == ["x:1"]
    assert ("Row", "click hidden detail") in help_texts(host)
    assert help_contains(host, "hidden detail", "title help")
    assert plain("a<br>b &amp; <i>c</i>") == "a b & c"
    row.help_icon.hide_popup()


def test_tooltips_missing_ignores_help_icons(app):
    from PySide6.QtWidgets import QWidget, QVBoxLayout
    from replay.gui.helpicon import HelpIcon
    from replay.gui.tooltips import missing
    w = QWidget()
    QVBoxLayout(w).addWidget(HelpIcon("", "x"))
    assert missing(w) == []


# ------------------------------------------------------------------ everything moved is still reachable
@pytest.fixture()
def win(tmp_path, app):
    from replay.gui.app import MainWindow
    root = tmp_path / "projects"
    make_project(root / "game-aaa111")
    w = MainWindow(root)
    w.list.setCurrentRow(0)
    yield w
    w.close()


def _all_help(win) -> str:
    from replay.gui.helpicon import help_texts
    return " ".join(t for _k, t in help_texts(win)).lower()


def test_moved_texts_are_reachable_in_the_main_window(win):
    """Every grey hint / how-to that used to be on the screen is now in a ? icon of the same tab."""
    from replay.gui.helpicon import help_contains
    ov, cal, rp = win.overview_tab, win.calib_tab, win.replay_tab
    # Overview
    for needle in ("court found", "press details for the technical diagnostics", "automatic (by jersey hue)",
                   "click a few players of each team", "shirts", "directx 12", "every 3rd frame", "bigger finds more",
                   "preliminary analysis", "[x0, y0, x1, y1]", "tick when the names and home/away"):
        assert help_contains(ov, needle), needle
    # Court fix
    for needle in ("left-click: place point", "ctrl+z: undo point", "pick a landmark", "first keyframe needs 4+",
                   "yellow lines must sit on the real court lines", "carried there by the camera motion",
                   "all keyframes show the same view", "not reliable yet", "error"):
        assert help_contains(cal, needle), needle
    # Replay (the key is filled in when the replay data is shown)
    for needle in ("averages each position", "briefly out of view", "hollow circles", "halo",
                   "coaches or spectators", "boxes on video"):
        assert help_contains(rp, needle), needle
    rp.legend.set_help(__import__("replay.gui.tooltips", fromlist=["HELP"]).HELP["replay_key"])
    assert help_contains(rp, "dashed ball = estimated", "no calibration")


def test_teams_status_wording_is_short_and_the_rest_is_behind_the_icon(win):
    from replay import team_binding as tbind
    ov = win.overview_tab
    # no jersey colours found yet: the line is empty, the explanation is behind the ?
    assert ov.teams_status.text() == "" and "found by jersey colour" in ov.teams_help.plain_text()
    # colours found, not confirmed: short visible warning (it keeps the words "Please check"), details behind the ?
    real = tbind.team_overview

    def fake(m):
        o = real(m)
        for k in ("A", "B"):
            o[k] = dict(o[k], detected=True, confirmed=False, colour_name="orange")
        return o

    tbind.team_overview = fake
    try:
        ov._load_teams({})
    finally:
        tbind.team_overview = real
    assert "Please check" in ov.teams_status.text() and len(ov.teams_status.text()) < 120
    txt = ov.teams_help.plain_text().lower()
    assert "tick confirmed and save" in txt and "which colour is home" in txt
    assert not ov.teams_help.isHidden()


def test_stats_notes_and_keys_are_behind_icons(win, tmp_path):
    from replay.gui.helpicon import help_contains
    st = win.stats_tab
    st._ensure_ui()
    assert help_contains(st, "filled = made, ring = missed", "click a number in the stats table")
    st.notes.set_help("<b>About these numbers.</b> estimated")
    assert "about these numbers" in st.notes.plain_text().lower()


def test_dialog_texts_moved_behind_icons(app, tmp_path):
    from replay.gui.app import ImportDialog
    from replay.gui.helpicon import help_contains
    d = ImportDialog(None, "C:/x/game.mp4")
    assert help_contains(d, "always contains the original", "more disk space")
    assert d.copy.isEnabled() and not d.copy.isChecked()


# ------------------------------------------------------------------ timeline, coach report, roster, scoresheet
def test_timeline_key_line_is_behind_an_icon(win):
    from replay.gui.helpicon import help_contains
    tl = win.replay_tab.timeline
    assert help_contains(tl, "m: add a mark", "double-click a track", "shift+wheel", "right-click for options")
    tl._hint()
    assert tl.info.text() == "" and not tl.info_help.isHidden()


def test_coach_report_hint_is_behind_an_icon(win, tmp_path):
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from test_coach import _rep
    from replay.gui import coach_report as CR
    from replay.gui.helpicon import help_contains
    from replay.gui.projects import Project
    root = make_project(tmp_path / "pp")
    d = CR.CoachReportDialog(None, Project(root), _rep())
    assert help_contains(d, "click any blue number to watch those moments")
    assert d.hint.text() == ""  # the line itself is free for feedback such as "Saved ..."
    d._note("Saved x")
    assert d.hint.text() == "Saved x"
    d._reset_note()
    assert d.hint.text() == ""


def test_roster_and_scoresheet_hints_are_behind_icons(win):
    from replay.gui.helpicon import help_contains
    from replay.gui.roster_dialog import RosterDialog
    d = RosterDialog(win.current, win)
    assert help_contains(d, "a manual choice always wins", "matched to names automatically")
    from replay.gui.scoresheet_dialog import ScoresheetDialog
    s = ScoresheetDialog(win.current, win)
    assert help_contains(s, "double-click a row to jump to it in the replay")
    s.notes_help.set_help("· note one")
    assert "note one" in s.notes_help.plain_text()


def test_team_name_field_gets_the_free_width(win):
    ov = win.overview_tab
    ov.resize(900, 700)
    ov.show()
    for k in ("A", "B"):
        _j, sw, nm, side = ov.team_rows[k]
        assert nm.minimumWidth() >= 90 and sw.width() <= 60
