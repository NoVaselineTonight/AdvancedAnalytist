"""Plain-language tooltips for every control in the Overview and Court fix (Calibrate) tabs.

apply(win) fills in tooltips that are missing (existing, more specific tooltips are kept). Controls are found by
their attribute name on the tab, or - for unnamed buttons - by their text. ``missing(widget)`` lists the
interactive controls that still have no tooltip (used by the tests, so new controls don't slip through).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtWidgets import (QAbstractButton, QAbstractSlider, QAbstractSpinBox, QComboBox, QLineEdit,
                               QListWidget, QWidget)

from replay.gui.helpicon import HelpIcon

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
        if not _interactive(w) or w.toolTip() or isinstance(w, HelpIcon):  # a "?" is itself the description
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


# ---------------------------------------------------------------------------------------------------------------
# Text behind the small "?" icons (replay/gui/helpicon.py). Everything here used to be a grey line or paragraph
# on the screen itself; the screens keep only the main numbers and controls. HTML is allowed.
HELP = {
    # ---- Overview
    "overview_report": "What the analysis found. <b>Court found</b>: how much of the video the app could place on the "
                       "court. <b>Ball</b>: on how many frames the ball's position is known (\"detected\" means it "
                       "was seen directly; the rest is estimated). Click the blue links to jump to that moment in "
                       "the video.",
    "overview_details": "Press <b>Details</b> for the technical diagnostics and all settings: what ran and why "
                        "something did not finish, frame counts, tracks, time ranges, the log files and every setting "
                        "used. Press it again to go back to the short view.",
    "teams_confirmed": "Teams are bound to their jersey colours and stay the same on every rebuild.",
    "teams_not_detected": "The two teams are found by jersey colour when the replay is built; then name them here.",
    "teams_please": "{why} Enter the team names, choose Home/Away, tick Confirmed and Save.",
    "teams_why_scorebar": "Home was guessed from the score bar.",
    "teams_why_unknown": "Which colour is home is not known yet, so teams are named after their jersey colour.",
    "confirmed": "Tick when the names and Home/Away of both jersey colours are right. Teams then stay bound to "
                 "their jersey colours on every rebuild.",
    "referees": "Colour of the referees' shirts (left) and trousers (right) - helps the app tell referees from "
                "players.",
    "pick_teams": "Click a few players of each team on a frame. Teams are then recognised from how their kits really "
                  "look in this video - much more reliable than automatic detection for dark or similar colours.",
    "pick_auto": "Automatic (by jersey hue). If teams come out wrong, pick players in the video.",
    "device": "GPU uses DirectX 12 (any AMD/NVIDIA/Intel card) and is several times faster. It can be changed at any "
              "time without re-analysing.",
    "stride": "Playback always shows every frame; positions are interpolated between analysed frames. Every 3rd frame "
              "(about 10 per second) is plenty for player movement. Changing it means analysing the video again.",
    "variant": "How big (careful) the player detector is. Bigger finds more players in crowds but is slower. Medium "
               "is right for most games. Changing it means analysing the video again.",
    "skip_breaks": "A fast preliminary analysis (about 2-3 minutes for a whole game) finds when the ball is in play "
                   "from the scoreboard clock and camera/player movement. Players are then detected and tracked only "
                   "during play, not in warm-ups, timeouts or breaks. Shown as the 'In play' timeline track.",
    "overlays": "Screen rectangles covered by broadcast graphics (for example the score bar), written as "
                "[x0, y0, x1, y1] with values from 0 to 1. Example: [[0.26, 0.9, 0.74, 0.975]]. Leave as [] when "
                "there are none.",
    # ---- Court fix
    "calib_keys": "<b>Mouse and keys.</b> Left-click: place point · mouse wheel: zoom · right-drag: pan · Space: "
                  "play/pause · ← / → 1 s, ↑ / ↓ 10 s, , / . 0.1 s · Enter: save keyframe · Ctrl+Z: undo point.",
    "calib_steps": "<b>1.</b> Pick a landmark on the court diagram · <b>2.</b> click the same spot on the frame. The "
                   "first keyframe needs 4+ points. After that every keyframe is combined with the others, so 2–3 "
                   "points on a new part of the court already help.",
    "calib_fit": "The yellow lines must sit on the real court lines. On frames without a keyframe they show the "
                 "calibration carried there by the camera motion (once the analysis has reached that point). Where "
                 "they drift off the real lines, add a keyframe there.",
    "calib_fixed": "All keyframes show the same view. Tick when the camera stands still the whole game (tripod, no "
                   "turning or zooming); then one keyframe is enough for the whole video.",
    "calib_gemini": "Experimental: not reliable yet - in a first test it found too few court views.",
    "calib_keyframes": "Frames where you clicked court points. Double-click one to go there. <b>error</b> = how well "
                       "the points fit the combined calibration; a large error means a point is probably misplaced.",
    # ---- Replay
    "smoothing": "Display smoothing: averages each position over this time window (display only; exported data is "
                 "unaffected).",
    "hold": "Keep showing players who are briefly out of view (e.g. the camera panned away) at their last known "
            "position, faded, for this long. Display only.",
    "interp": "Interpolated: show positions filled in across short gaps (up to 3 s, flagged in the data). "
              "Hollow when interpolated: draw those filled-in positions as hollow circles.",
    "confidence": "Off: clean view - seen players solid, short gaps as small faint dots, nothing predicted far from a "
                  "sighting or off the court. On: show how sure the tracking is: solid dot = seen; pale dashed dot "
                  "with a halo = position predicted while the player is hidden (bigger halo = less sure where); "
                  "'?' = team or identity uncertain. Jersey numbers replace track ids where known. "
                  "<b>Legend</b> shows the key to these dot styles in the 2D view.",
    "unknown": "<b>Unclassified people</b>: people that could not be assigned to a team or as referee (often coaches "
               "or spectators near the court). <b>Boxes on video</b>: draw the tracking boxes, IDs and ball on the "
               "video.",
    "replay_key": "<b>Key.</b> Dashed ball = estimated (in flight) · no dots = no court calibration at this moment · "
                  "timeline top strip = tracking quality: "
                  "<span style='color:#4caf7a'>■</span> good "
                  "<span style='color:#e8d36a'>■</span> lineup incomplete "
                  "<span style='color:#f2a33a'>■</span> few players seen "
                  "<span style='color:#e8743b'>■</span> calibration poor "
                  "<span style='color:#d64545'>■</span> no calibration",
    # ---- Stats
    "stats_shots": "Filled = made, ring = missed, dotted = result not seen. Click a shot to see it in the video.",
    "stats_events": "Click a number in the stats table (or a zone or lineup row) to list the detected events behind "
                    "it here. Click an event to watch it in the video.",
    "stats_goto": "Clicking a shot, row or event seeks the video and switches to the Replay tab.",
    "stats_notes": "<b>About these numbers.</b> {notes}",
    # ---- dialogs
    "import_copy": "Off: the Replay keeps a playable copy and remembers where the original is. On: the original is "
                   "copied too (uses more disk space). Either way, an exported Replay (.aareplay) always contains "
                   "the original.",
    "export_how": "Open the file with <i>File → Open Replay file…</i> on this or another computer - no re-analysis "
                  "needed.",
    "picker": "Choose a team below, then click a player of that team on the frame (inside their box). Pick 2–4 "
              "players per team from different moments, plus a referee if there is one. Players are matched by how "
              "their kit actually looks in this video.",
}
