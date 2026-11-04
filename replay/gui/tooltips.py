"""Plain-language tooltips for every control in the Overview and Court fix (Calibrate) tabs.

apply(win) fills in tooltips that are missing (existing, more specific tooltips are kept). Controls are found by
their attribute name on the tab, or - for unnamed buttons - by their text. ``missing(widget)`` lists the
interactive controls that still have no tooltip (used by the tests, so new controls don't slip through).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtWidgets import (QAbstractButton, QAbstractSlider, QAbstractSpinBox, QComboBox, QLineEdit,
                               QListWidget, QWidget)

if TYPE_CHECKING:  # pragma: no cover
    from replay.gui.app import MainWindow

OVERVIEW = {
    "name": "The name of this Replay as shown in the list on the left. Change it any time.",
    "ref_c": "Colour of the referees' shirts - helps the app tell referees from players",
    "ref_c2": "Colour of the referees' trousers",
    "date": "Day of the game (for example 2026-10-18). Shown in the coach report.",
    "location": "Where the game was played (gym or town). Shown in the coach report.",
    "variant": "How big (careful) the player detector is. Bigger finds more players in crowds but is slower. "
               "Medium is right for most games. Changing it means analysing the video again.",
    "save_btn": "Save the game information above. If teams or colours changed, the 2D replay is rebuilt "
                "(seconds, not a new analysis).",
    "pick_status": "",
    "report": "What the analysis found. Click the blue links to jump to the moment in the video.",
    "steps": "",
}
TEAM_ROW = ("Jersey colour found in the video (click to change how the team is drawn)",
            "",
            "Team name as it should appear in the replay and the coach report",
            "Is this the home or the away team? Changing it swaps the sides for both teams.")

CALIBRATE = {
    "play_btn": "Play or pause the video (Space)",
    "slider": "Drag to move through the video. Positions with yellow court lines already have a calibration.",
    "save_btn": "Store the points you clicked on this frame as a keyframe (Enter). The court position of every "
                "frame is then recalculated.",
    "accept_snap_btn": "Keep the fine-tuned (green) lines",
    "reject_snap_btn": "Throw away the fine-tuning and keep the previous lines",
    "fixed_cb": "Tick when the camera stands still the whole game (tripod, no turning or zooming). Then one keyframe "
                "is enough for the whole video.",
    "gemini_game_cap": "Most money the Gemini helper may spend on one game (it stops when this is reached)",
    "gemini_month_cap": "Most money the Gemini helper may spend in one month",
    "kf_list": "Frames where you clicked court points. Double-click one to go there. A large error means a point is "
               "probably misplaced.",
    "diagram": "Court map: click a landmark here first (for example a corner or the free-throw line), then click the "
               "same spot in the video",
    "view": "The video frame. Click to place the selected court point; mouse wheel zooms, right-drag moves the view.",
}
BUTTON_TEXT = {
    "−10s": "Back 10 seconds (↓)", "−1s": "Back 1 second (←)", "−0.1s": "Back one frame (,)",
    "+0.1s": "Forward one frame (.)", "+1s": "Forward 1 second (→)", "+10s": "Forward 10 seconds (↑)",
    "Fit view": "Zoom out so the whole frame is visible again",
    "Undo point": "Remove the last point you clicked (Ctrl+Z)",
    "Clear": "Remove all points on this frame (keyframes already saved are kept)",
    "Go to": "Jump to the selected keyframe",
    "Delete keyframe": "Delete the selected keyframe; the court position is recalculated without it",
}


def _interactive(w: QWidget) -> bool:
    return isinstance(w, (QAbstractButton, QComboBox, QLineEdit, QAbstractSpinBox, QAbstractSlider, QListWidget))


def missing(root: QWidget) -> list[str]:
    """Interactive controls under ``root`` without a tooltip (spin/combo line-edit internals are skipped)."""
    out = []
    for w in root.findChildren(QWidget):
        if not _interactive(w) or w.toolTip():
            continue
        par = w.parentWidget()
        if isinstance(w, QLineEdit) and isinstance(par, (QComboBox, QAbstractSpinBox)):
            continue  # the editor inside a combo/spin box: the box has the tooltip
        if isinstance(w, QAbstractButton) and par is not None and par.inherits("QScrollBar"):
            continue
        if w.inherits("QScrollBar") or (par is not None and par.inherits("QAbstractItemView")):
            continue  # scroll bars of lists and text views
        out.append(f"{type(w).__name__}('{getattr(w, 'text', lambda: '')() if hasattr(w, 'text') else ''}')")
    return out


def _set(w: QWidget | None, tip: str) -> None:
    if w is not None and tip and not w.toolTip():
        w.setToolTip(tip)


def apply_overview(tab: QWidget) -> None:
    for attr, tip in OVERVIEW.items():
        _set(getattr(tab, attr, None), tip)
    for _k, row in getattr(tab, "team_rows", {}).items():
        for w, tip in zip(row, TEAM_ROW):
            _set(w, tip)
    # anything else interactive gets a short tooltip from its text (roster / scoresheet hooks set their own)
    for w in tab.findChildren(QAbstractButton):
        _set(w, BUTTON_TEXT.get(w.text(), ""))


def apply_calibrate(tab: QWidget) -> None:
    for attr, tip in CALIBRATE.items():
        _set(getattr(tab, attr, None), tip)
    for w in tab.findChildren(QAbstractButton):
        _set(w, BUTTON_TEXT.get(w.text(), ""))


def apply(win: "MainWindow") -> None:
    apply_overview(win.overview_tab)
    apply_calibrate(win.calib_tab)
