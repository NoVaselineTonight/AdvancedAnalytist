"""First-run help: "Getting started" in 3 steps, shown where the app used to show an empty page.

Step 1 import a video -> step 2 wait (analysis runs by itself) -> step 3 open the Replay tab / coach report, plus how to
attach the game's scoresheet (better names, numbers and scores). Every step is a clickable link that does the action.
Also available any time from Help -> Getting started (F1).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QHBoxLayout, QMessageBox, QTextBrowser, QVBoxLayout,
                               QWidget)

if TYPE_CHECKING:  # pragma: no cover
    from replay.gui.app import MainWindow

STEPS_HTML = """
<div>
<p>&nbsp;</p>
<h2 style='margin-bottom:2px'>Getting started</h2>
<p style='color:gray; margin-top:0'>From a game video to a 2D replay and a coach report in three steps.</p>

<table cellspacing='10'>
<tr><td valign='top'><span style='font-size:22px; font-weight:bold; color:#2563eb'>1</span></td>
<td><b>Import the game video</b><br>
<a href='action:import'>Import video…</a> (Ctrl+I) - any normal video file (MP4 is best).
The original file stays where it is.<br>
<span style='color:gray'>Got a Replay file from someone else? <a href='action:open'>Open Replay file…</a>
(.aareplay) - nothing has to be analysed again.</span></td></tr>

<tr><td valign='top'><span style='font-size:22px; font-weight:bold; color:#2563eb'>2</span></td>
<td><b>Wait - the analysis runs by itself</b><br>
The Replay appears in the list on the left with a <span style='color:#2563eb'>blue</span> icon while it works
(about the length of the game on a normal laptop; faster with a graphics card). You can close the app - it
continues next time. A <span style='color:#16a34a'>green tick</span> means ready;
<span style='color:#d97706'>amber</span> means it needs you (the status line tells what to do).<br>
<span style='color:gray'>Meanwhile, in the <b>Overview</b> tab, type the team names and
<a href='action:scoresheet'>attach the scoresheet</a> (the federation's digital PDF, or a CSV): it gives the players'
names, numbers and the official score.</span></td></tr>

<tr><td valign='top'><span style='font-size:22px; font-weight:bold; color:#2563eb'>3</span></td>
<td><b>Watch and share</b><br>
Open the <a href='action:replay'>Replay tab</a> to see the video and the 2D court side by side, or the
<a href='action:report'>coach report</a> (box score, shot charts, lineups - click any number to see the clips).
Share everything as one file with <a href='action:export'>Export Replay</a> (Ctrl+S).</td></tr>
</table>

<p style='color:gray'>Tips: hover over any button for a short explanation · right-click a Replay to rename,
duplicate or open its folder · if the 2D court looks wrong, use <b>Court fix (advanced)</b>.</p>
<p style='color:gray; font-size:small'>© Onni Kauppinen 2026</p>
</div>
"""


def _need_replay(win: "MainWindow", what: str) -> bool:
    if win.current is not None:
        return True
    QMessageBox.information(win, "Getting started", f"First select a Replay in the list on the left, then {what}.\n\n"
                            "No Replays yet? Start with step 1: Import video…")
    return False


def do_action(win: "MainWindow", name: str) -> None:
    """Run a Getting-started link. Unknown names are ignored."""
    if name == "import":
        win.import_video()
    elif name == "open":
        win.open_package()
    elif name == "scoresheet":
        if _need_replay(win, "attach the scoresheet in the Overview tab"):
            win.tabs.setCurrentWidget(win.overview_tab)
            from replay.gui.scoresheet_dialog import _SheetButton
            btn = next(iter(win.overview_tab.findChildren(_SheetButton)), None)
            if btn is not None:
                btn.click()
    elif name == "replay":
        if _need_replay(win, "open its Replay tab"):
            win.tabs.setCurrentWidget(win.replay_tab)
    elif name == "report":
        if _need_replay(win, "open its coach report"):
            if not win.current.status()["replay_ready"]:
                QMessageBox.information(win, "Coach report", "The coach report is available once the 2D replay is "
                                        "built (green tick in the list).\n\nWhat to do: wait for the analysis to "
                                        "finish, or press the button at the top of the Replay.")
                return
            from replay.gui import coach_report
            coach_report.open_report(win, win.current)
    elif name == "export":
        if _need_replay(win, "export it"):
            win.export_project(win.current)


class GettingStarted(QWidget):
    """The 3-step help page (link clicks run the actions on the main window)."""

    def __init__(self, win: "MainWindow", parent: QWidget | None = None):
        super().__init__(parent)
        self.win = win
        lay = QHBoxLayout(self)
        # QTextBrowser (not QLabel): rich text with a table wraps reliably at any window size, and scrolls
        self.text = QTextBrowser()
        self.text.setOpenLinks(False)
        self.text.setHtml(STEPS_HTML)
        self.text.setFrameShape(QTextBrowser.Shape.NoFrame)
        self.text.setStyleSheet("background: transparent;")
        self.text.setMaximumWidth(720)
        self.text.anchorClicked.connect(lambda url: self._link(url.toString()))
        lay.addStretch(1)
        lay.addWidget(self.text, 4)
        lay.addStretch(1)

    def _link(self, href: str) -> None:
        if href.startswith("action:"):
            do_action(self.win, href.split(":", 1)[1])


class GettingStartedDialog(QDialog):
    def __init__(self, win: "MainWindow"):
        super().__init__(win)
        self.setWindowTitle("Getting started")
        lay = QVBoxLayout(self)
        page = GettingStarted(win, self)
        page.text.anchorClicked.connect(lambda _u: self.accept())  # close, then the action runs on the window
        lay.addWidget(page)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self.resize(760, 640)


def install(win: "MainWindow") -> GettingStarted:
    """Replace the empty start page with the Getting-started page and add Help -> Getting started (F1)."""
    page = GettingStarted(win)
    old = win.stack.widget(0)
    win.stack.insertWidget(0, page)
    if old is not None:
        win.stack.removeWidget(old)
        old.deleteLater()
    win.stack.setCurrentIndex(0 if win.current is None else win.stack.currentIndex())
    win.help_page = page
    mb = win.menuBar()
    help_menu = next((a.menu() for a in mb.actions() if a.text().replace("&", "") == "Help" and a.menu()), None)
    if help_menu is None:
        help_menu = mb.addMenu("&Help")
    act = QAction("Getting started", win)
    act.setShortcut(QKeySequence("F1"))
    act.triggered.connect(lambda: GettingStartedDialog(win).exec())
    first = help_menu.actions()[0] if help_menu.actions() else None
    help_menu.insertAction(first, act)
    return page
