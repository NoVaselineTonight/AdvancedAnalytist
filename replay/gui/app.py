"""Desktop application (Qt / PySide6). Start with ``python -m replay gui``.

Window layout:
    left   Replay list + Import video / Open Replay file / Delete
    right  selected Replay: status + progress, tabs Overview | Calibrate | Replay
    menu   File -> import, open (.aareplay / legacy .aaproj)
           Export -> Replay package (.aareplay), simple data export (for AI), individual files

"Replay" is the user-facing name of a project (internally still Project / projects/ / project.json).
"""

from __future__ import annotations

import json
import os
from html import escape as html_escape
import shutil
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QRectF, QSettings, QSize, QSizeF, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QAction, QColor, QFont, QIcon, QKeySequence, QPainter, QPen, QPixmap, QShortcut
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtMultimediaWidgets import QGraphicsVideoItem
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
                               QFileDialog, QFormLayout, QGridLayout, QGraphicsScene, QGraphicsView, QHBoxLayout, QInputDialog,
                               QLabel, QLineEdit, QListWidget,
                               QListWidgetItem, QMainWindow, QMenu, QMessageBox, QProgressBar, QProgressDialog, QPushButton, QScrollArea,
                               QFrame, QSlider, QSplitter, QStackedWidget, QTabWidget, QTextBrowser, QVBoxLayout, QWidget)

from replay import camera_model, court, homography
from replay.calibration import CalibrationFile, FrameSolver, Keyframe, _joint_fit, resolve
from replay.config import CalibrationConfig
from replay.gui import media
from replay.autocalib import keypoints_from_view, oriented_lines, snap_view, swap_left_right
from replay.camera_model import court_to_image as cam_court_to_image
from replay.gui.jobs import ANALYZE, AUTOCAL, BUILD, IMPORT, JobManager
from replay import __version__
from replay.report import BUILD_VERSION
from replay.gui.projects import LEGACY_PACKAGE_EXT, PACKAGE_EXT, ExportPlan, Project, ProjectStore
from replay.pipeline import load_pass1
from replay.gui.motion import PlaybackClock
from replay.gui.widgets import ColorButton, CourtDiagram, FrameGrabber, FrameView, ReplayCourt, frame_index
from replay.gui.timeline import TimelineWidget
from replay.gui import ai_dialog
from replay.gui import friendly
from replay.gui.helpicon import HelpIcon, help_row, titled
from replay.gui.tooltips import HELP

APP_NAME = "AdvancedAnalytist"
VIDEO_FILTER = "Videos (" + " ".join("*" + e for e in media.VIDEO_EXTENSIONS) + ");;All files (*)"


def fmt_time(s: float | None) -> str:
    if s is None or not np.isfinite(s):
        return "–"
    h, m, sec = int(s // 3600), int(s // 60) % 60, s % 60
    return f"{h}:{m:02d}:{sec:04.1f}" if h else f"{m}:{sec:04.1f}"


def fmt_dur(s: float | None) -> str:
    if s is None:
        return ""
    return f"{s:.0f} s" if s < 90 else f"{s / 60:.0f} min" if s < 5400 else f"{s / 3600:.1f} h"


def status_text(p: Project, job: dict) -> str:
    s = p.status()
    if job["error"]:
        return "Error"
    if job["kind"]:
        return f"{job['label']} {job['progress'] * 100:.0f}%"
    if job["queued"]:
        return "Queued"
    if not s["imported"]:
        return "Not imported"
    if not s["analyzed"]:
        if not s["analysis_started"]:
            return "Waiting for analysis"
        return "Stopped" if s.get("paused") else "Analysis paused"
    if not s["calibrated"]:
        return "Court not found - fix by hand"
    if not s["replay_ready"]:
        return "Replay not built"
    return "Replay ready" + (" (outdated)" if s["replay_stale"] else "")


_ICONS: dict[str, QIcon] = {}


def _pin_icon() -> QIcon:
    """Small pin drawn at runtime (no image files to ship)."""
    if "pin" not in _ICONS:
        pm = QPixmap(16, 16)
        pm.fill(Qt.GlobalColor.transparent)
        qp = QPainter(pm)
        qp.setRenderHint(QPainter.RenderHint.Antialiasing)
        qp.setPen(QPen(QColor("#555"), 1.6))
        qp.drawLine(8, 9, 8, 15)  # needle
        qp.setPen(Qt.PenStyle.NoPen)
        qp.setBrush(QColor("#e11d48"))
        qp.drawEllipse(3, 1, 10, 10)  # head
        qp.end()
        _ICONS["pin"] = QIcon(pm)
    return _ICONS["pin"]


def _blank_icon() -> QIcon:
    if "blank" not in _ICONS:
        pm = QPixmap(16, 16)
        pm.fill(Qt.GlobalColor.transparent)
        _ICONS["blank"] = QIcon(pm)
    return _ICONS["blank"]


def run_with_progress(parent: QWidget, title: str, fn, *args):
    """Run fn(*args, progress=cb) in a thread with a modal progress dialog. Returns result or raises."""
    dlg = QProgressDialog(title, None, 0, 1000, parent)
    dlg.setWindowModality(Qt.WindowModality.WindowModal)
    dlg.setMinimumDuration(0)
    dlg.setAutoClose(False)
    state = {"p": 0.0, "done": False, "result": None, "error": None}

    def work():
        try:
            state["result"] = fn(*args, progress=lambda x: state.__setitem__("p", x))
        except Exception as e:  # reported below
            state["error"] = e
        state["done"] = True

    threading.Thread(target=work, daemon=True).start()
    while not state["done"]:
        dlg.setValue(int(state["p"] * 1000))
        QApplication.processEvents()
        threading.Event().wait(0.05)
    dlg.close()
    if state["error"]:
        raise state["error"]
    return state["result"]


# ====================================================================== overview tab
class OverviewTab(QWidget):
    def __init__(self, win: "MainWindow"):
        super().__init__()
        self.win = win
        self.p: Project | None = None
        lay = QHBoxLayout(self)

        left = QVBoxLayout()
        left.addWidget(QLabel("<b>Steps</b>"))
        self.steps = QLabel()
        self.steps.setTextFormat(Qt.TextFormat.RichText)
        self.steps.setWordWrap(True)
        left.addWidget(self.steps)
        left.addWidget(titled("<b>Report</b>", HELP["overview_report"], "Report"))
        self.report = QTextBrowser()
        self.report.setOpenLinks(False)
        self.report.anchorClicked.connect(self._report_link)
        left.addWidget(self.report, 1)
        self.report_details_btn = QPushButton("Details")
        self.report_details_btn.setCheckable(True)
        self.report_details_btn.setToolTip("Show the details: what ran and why something did not finish, frame counts, "
                                           "tracks, time ranges, the log files and every setting used")
        self.report_details_btn.toggled.connect(lambda _on: self.refresh())
        details_row = help_row(self.report_details_btn, HELP["overview_details"], "Details", stretch=False)
        details_row.layout().addStretch(1)
        left.addWidget(details_row)
        lay.addLayout(left, 1)

        form_w = QWidget()
        form = QFormLayout(form_w)
        self._form = form
        form.addRow(QLabel("<b>Game &amp; settings</b>"))
        self.name = QLineEdit()
        # the two detected teams, bound to their jersey colour (replay/team_binding.py): swatch, name, home/away
        self.team_rows: dict[str, tuple[QLabel, ColorButton, QLineEdit, QComboBox]] = {}
        for k in ("A", "B"):
            jersey, sw, nm, side = QLabel(), ColorButton(), QLineEdit(), QComboBox()
            side.addItem("Home", "home")
            side.addItem("Away", "away")
            sw.setToolTip("Jersey colour found in the video (click to change how the team is drawn)")
            nm.setPlaceholderText("Team name")
            sw.changed.connect(self._teams_edited)
            nm.textEdited.connect(self._teams_edited)
            side.currentIndexChanged.connect(lambda _i, k=k: self._side_changed(k))
            self.team_rows[k] = (jersey, sw, nm, side)
        self.teams_ok = QCheckBox("Confirmed")
        self.teams_ok.setToolTip("Tick when the names and home/away of both jersey colours are right")
        self.teams_ok.toggled.connect(lambda _on: setattr(self, "_teams_dirty", True))
        self.teams_status = QLabel()
        self.teams_status.setWordWrap(True)
        self._teams_dirty = False
        self._teams_shown = ""
        self.ref_c, self.ref_c2 = ColorButton(), ColorButton()
        self.date, self.location = QLineEdit(), QLineEdit()
        self.date.setPlaceholderText("YYYY-MM-DD")
        self.stride = QComboBox()
        self.stride.setToolTip("Playback always shows every frame; positions are interpolated between analysed "
                               "frames.\nEvery 3rd frame (~10 per second) is plenty for player movement.")
        self.device = QComboBox()
        for v, label in [("auto", "Auto (GPU if available)"), ("gpu", "GPU"), ("cpu", "CPU")]:
            self.device.addItem(label, v)
        self.device.setToolTip("GPU uses DirectX 12 (any AMD/NVIDIA/Intel card) and is several times faster.\n"
                               "Can be changed at any time without re-analysing.")
        self.variant = QComboBox()
        for v, label in [("nano", "Nano (fastest)"), ("small", "Small"), ("medium", "Medium (default)"),
                         ("large", "Large (slowest)")]:
            self.variant.addItem(label, v)
        self.overlays = QLineEdit()
        self.overlays.setToolTip("Screen rectangles covered by broadcast graphics, normalised "
                                 "[x0, y0, x1, y1] (0..1). Example: [[0.26, 0.9, 0.74, 0.975]]")
        row = lambda *w: self._row(*w)  # noqa: E731
        form.addRow("Replay name", self.name)
        for i, k in enumerate(("A", "B")):
            jersey, sw, nm, side = self.team_rows[k]
            trow = row(sw, jersey, nm, side)
            tl = trow.layout()
            sw.setFixedWidth(44)  # swatch and Home/Away stay compact; the team name gets the free width
            side.setMinimumWidth(70)
            nm.setMinimumWidth(95)
            tl.setStretchFactor(nm, 1)
            form.addRow(f"Team {i + 1}", trow)
        from replay.gui.roster_dialog import roster_button  # roster hook
        form.addRow("", row(self.teams_ok, roster_button(self),
                            HelpIcon(HELP["confirmed"], "Confirmed")))  # roster hook: "Roster..." next to the teams
        from replay.gui.scoresheet_dialog import scoresheet_button  # scoresheet hook
        form.addRow("Scoresheet", row(scoresheet_button(self)))  # scoresheet hook: "Add scoresheet..." (pöytäkirja)
        teams_status_row = help_row(self.teams_status, "", "Teams")
        self.teams_help: HelpIcon = teams_status_row.help_icon  # the long wording of the line next to it
        form.addRow("", teams_status_row)
        # scoresheet vs kits: one-click fix when the jersey numbers contradict the home/away choice
        self.sheet_conflict = QLabel()
        self.sheet_conflict.setWordWrap(True)
        self.sheet_conflict.setStyleSheet("background:#fef3c7; color:#92400e; padding:6px; border-radius:4px;")
        self.sheet_swap = QPushButton("Swap home/away")
        self.sheet_swap.setToolTip("Make the team the scoresheet points to the home team, take the team names from "
                                   "the scoresheet and rebuild the replay")
        self.sheet_swap.clicked.connect(self.apply_sheet_teams)
        self._conflict: dict | None = None
        self._conflict_key: tuple | None = None
        form.addRow("", self._row(self.sheet_conflict, self.sheet_swap))
        self.sheet_conflict.parentWidget().setVisible(False)
        form.addRow("Referee shirt / trousers", row(self.ref_c, self.ref_c2,
                                                    HelpIcon(HELP["referees"], "Referee colours")))
        self.pick_btn = QPushButton("Pick players in the video…")
        self.pick_btn.setToolTip("Click a few players of each team on a frame. Teams are then recognised from how "
                                 "their kits really look in this video - much more reliable than automatic "
                                 "detection for dark or similar colours.")
        self.pick_btn.clicked.connect(self.pick_teams)
        self.pick_status = QLabel()
        self.pick_status.setWordWrap(True)
        self.pick_help = HelpIcon(HELP["pick_teams"] + "<br><br>" + HELP["pick_auto"], "Identify teams")
        form.addRow("Identify teams", row(self.pick_btn, self.pick_help))
        form.addRow("", self.pick_status)
        form.addRow("Game date", self.date)
        form.addRow("Location", self.location)
        form.addRow("Processing device", help_row(self.device, HELP["device"], "Processing device"))
        form.addRow(QLabel("<i>Changing the settings below requires re-analysis:</i>"))
        form.addRow("Analyse", help_row(self.stride, HELP["stride"], "Analyse"))
        form.addRow("Detector size", help_row(self.variant, HELP["variant"], "Detector size"))
        self.skip_breaks = QCheckBox("Skip breaks (preliminary analysis)")
        self.skip_breaks.setToolTip("A fast preliminary analysis (about 2-3 minutes for a whole game) finds when the "
                                    "ball is in play from the scoreboard clock and camera/player movement. "
                                    "Players are then detected and tracked only during play, not in warm-ups, "
                                    "timeouts or breaks. Shown as the 'In play' timeline track.")
        form.addRow("", help_row(self.skip_breaks, HELP["skip_breaks"], "Skip breaks"))
        form.addRow("Broadcast overlay regions", help_row(self.overlays, HELP["overlays"], "Broadcast overlay regions"))
        self.save_btn = QPushButton("Save")
        self.save_btn.clicked.connect(self.save)
        self.save_msg = QLabel()
        form.addRow(row(self.save_btn, self.save_msg))
        form_w.setMaximumWidth(600)
        # in a scroll area: the form (plus the phase panel above the tabs while a job runs) must never force the
        # window taller than a laptop screen - then Save and the analysis settings were off-screen (9.08 regression)
        from PySide6.QtWidgets import QScrollArea
        form_sc = QScrollArea()
        form_sc.setWidget(form_w)
        form_sc.setWidgetResizable(True)
        form_sc.setFrameShape(QScrollArea.Shape.NoFrame)
        form_sc.setMaximumWidth(620)
        form_sc.setMinimumWidth(540)  # (a scroll area asks for little; the form with its ? icons needs this much)
        form_sc.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        lay.addWidget(form_sc)

    @staticmethod
    def _row(*widgets: QWidget) -> QWidget:
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        for x in widgets:
            h.addWidget(x)
        return w

    def load(self, p: Project) -> None:
        self.p = p
        m, st = p.meta(), p.settings()
        c = m.get("jersey_colors", {})
        self.name.setText(p.info().get("name", p.id))
        self._load_teams(m)
        self.ref_c.set_color(c.get("referee", {}).get("primary", "#9a9a9a"))
        self.ref_c2.set_color(c.get("referee", {}).get("secondary", "#141414"))
        self.date.setText(m.get("game_date") or "")
        self.location.setText(m.get("location") or "")
        vfps = float(p.info().get("video_info", {}).get("fps") or 30.0)
        self.stride.clear()
        for n in (1, 2, 3, 4, 5, 6):
            label = "every frame" if n == 1 else f"every {n}{'nd' if n == 2 else 'rd' if n == 3 else 'th'} frame"
            note = ", default" if n == 3 else ""
            self.stride.addItem(f"{label} ({vfps / n:.1f}/s{note})", n)
        self.stride.setCurrentIndex(max(0, self.stride.findData(int(st["frame_stride"]))))
        self.device.setCurrentIndex(max(0, self.device.findData(st["device"])))
        self.variant.setCurrentIndex(max(0, self.variant.findData(st["variant"])))
        self.skip_breaks.setChecked(self._skip_breaks_now(p))
        self.overlays.setText(json.dumps(m.get("overlay_regions", [])))
        self._update_pick_status()
        self.refresh()

    def _load_teams(self, m: dict) -> None:
        """Fill the team rows from the colour binding (meta.json "team_binding")."""
        from replay import team_binding as tbind
        ov = tbind.team_overview(m)
        detected = ov["A"]["detected"]
        self._teams_shown = json.dumps(m.get(tbind.FIELD), sort_keys=True)
        for k in ("A", "B"):
            jersey, sw, nm, side = self.team_rows[k]
            for w in (sw, nm, side):
                w.blockSignals(True)
            sw.set_color(ov[k]["color"])
            jersey.setText(f"{ov[k]['colour_name']} jerseys" if detected else ("Home" if k == "A" else "Away"))
            nm.setText(ov[k]["name"])
            side.setCurrentIndex(0 if ov[k]["side"] == "home" else 1)
            sw.setEnabled(detected)
            side.setEnabled(detected)
            for w in (sw, nm, side):
                w.blockSignals(False)
        self.teams_ok.blockSignals(True)
        self.teams_ok.setChecked(ov["A"]["confirmed"])
        self.teams_ok.setEnabled(detected)
        self.teams_ok.blockSignals(False)
        b = m.get(tbind.FIELD) or {}
        if not detected:
            self.teams_status.setText("")
            self.teams_help.set_help(HELP["teams_not_detected"])
        elif ov["A"]["confirmed"]:
            self.teams_status.setText("")
            self.teams_help.set_help(HELP["teams_confirmed"])
        else:
            why = HELP["teams_why_scorebar"] if b.get("source") == "scorebar" else HELP["teams_why_unknown"]
            self.teams_status.setText("<b style='color:#b45309'>Please check:</b> the team names and Home/Away.")
            self.teams_help.set_help(HELP["teams_please"].format(why=why))
        self._teams_dirty = False

    def _teams_edited(self, *_args) -> None:
        self._teams_dirty = True
        if self.teams_ok.isEnabled() and not self.teams_ok.isChecked():
            self.teams_ok.blockSignals(True)
            self.teams_ok.setChecked(True)  # editing a team row = deciding which colour is which team
            self.teams_ok.blockSignals(False)

    def _side_changed(self, k: str) -> None:
        """Exactly one team is home: switching one row switches the other."""
        other = self.team_rows["B" if k == "A" else "A"][3]
        other.blockSignals(True)
        other.setCurrentIndex(1 - self.team_rows[k][3].currentIndex())
        other.blockSignals(False)
        self._teams_dirty = True
        if self.teams_ok.isEnabled():
            self.teams_ok.setChecked(True)

    def team_rows_state(self) -> list[dict]:
        return [{"label": k, "name": self.team_rows[k][2].text(), "side": self.team_rows[k][3].currentData(),
                 "color": self.team_rows[k][1].color} for k in ("A", "B")]

    @staticmethod
    def _skip_breaks_now(p: Project) -> bool:
        """What the current analysis does: an existing run tells; otherwise the setting (default on)."""
        run = p.results / "work" / "run.json"
        if run.exists():
            try:
                return "play_intervals" in json.loads(run.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                pass
        return bool(p.settings().get("skip_breaks", True))

    def _update_pick_status(self) -> None:
        smp = (self.p.meta().get("team_samples") or {}) if self.p else {}
        n = {k: len(v) for k, v in smp.items()}
        if n.get("A") and n.get("B"):
            self.pick_status.setText(f"Using picked players: home {n['A']}, away {n['B']}, referee "
                                     f"{n.get('referee', 0)}.")
        else:
            self.pick_status.setText("")  # the "Automatic (by jersey hue)..." hint is behind the ? by the button
        self._form.setRowVisible(self.pick_status, bool(self.pick_status.text()))

    def pick_teams(self) -> None:
        p = self.p
        if p is None:
            return
        if not (p.results / "work" / "run.json").exists() or not list((p.results / "work").glob("chunk_*.npz")):
            QMessageBox.information(self, APP_NAME, "The analysis needs to have started (detected some players) "
                                                    "before you can pick them.")
            return
        dlg = TeamPickerDialog(self, p, self.win.calib_tab.frame_t)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        m = p.meta()
        m["team_samples"] = {k: v for k, v in dlg.samples.items() if v}
        p.save_meta(m)
        self._update_pick_status()
        if p.status()["analyzed"] and p.status()["calibrated"]:
            self.win.jobs.submit(p.id, BUILD)

    def save(self) -> None:
        p = self.p
        if p is None:
            return
        try:
            overlays = json.loads(self.overlays.text() or "[]")
        except ValueError:
            QMessageBox.warning(self, APP_NAME, "Overlay regions: invalid JSON.")
            return
        from replay import team_binding as tbind
        old_meta = p.meta()
        meta = {**old_meta, "game_date": self.date.text() or None, "location": self.location.text() or None,
                "overlay_regions": overlays}
        jc = dict(meta.get("jersey_colors") or {})
        jc["referee"] = {"primary": self.ref_c.color, "secondary": self.ref_c2.color}
        meta["jersey_colors"] = jc
        try:
            meta, clear_view_colors = tbind.overview_save(meta, self.team_rows_state(), self._teams_dirty,
                                                          self.teams_ok.isChecked(),
                                                          bool(p.settings().get("swap_teams", False)))
        except ValueError as e:
            QMessageBox.warning(self, APP_NAME, f"Teams: {e}.")
            return
        try:
            p.save_meta(meta)
        except Exception as e:
            QMessageBox.warning(self, APP_NAME, f"Invalid game data: {e}")
            return
        if clear_view_colors:  # the 2D view's own A/B colour overrides belonged to the old sides/colours
            st0 = p.settings()
            st0["display_colors"] = {k: v for k, v in (st0.get("display_colors") or {}).items() if k not in ("A", "B")}
            p.update_info(settings=st0)
        self._load_teams(meta)
        st = p.settings()
        new_stride, new_var = int(self.stride.currentData()), self.variant.currentData()
        old_skip, new_skip = self._skip_breaks_now(p), self.skip_breaks.isChecked()
        analysis_changed = (new_stride != int(st["frame_stride"]) or new_var != st["variant"]
                            or new_skip != old_skip) and p.status()["analysis_started"]
        if analysis_changed and QMessageBox.question(
                self, APP_NAME, "Changing which frames are analysed, the detector size or break skipping discards "
                                "the existing "
                                "analysis and runs it again. Continue?") != QMessageBox.StandardButton.Yes:
            new_stride, new_var, new_skip = int(st["frame_stride"]), st["variant"], old_skip
            self.skip_breaks.setChecked(old_skip)
            analysis_changed = False
        st.update(frame_stride=new_stride, variant=new_var, device=self.device.currentData(), skip_breaks=new_skip)
        p.update_info(name=self.name.text().strip() or p.id, settings=st)
        if analysis_changed:
            self.win.jobs.cancel(p.id)
            self.win.jobs.submit(p.id, ANALYZE, restart=True)
        elif p.status()["analyzed"] and p.status()["calibrated"]:
            self.win.jobs.submit(p.id, BUILD)
        self.save_msg.setText("Saved.")
        QTimer.singleShot(2500, lambda: self.save_msg.setText(""))
        self.win.refresh_list()

    def refresh(self) -> None:
        p = self.p
        if p is None:
            return
        if not self._teams_dirty:  # a build may have just found / re-bound the teams
            m = p.meta()
            if json.dumps(m.get("team_binding"), sort_keys=True) != self._teams_shown:
                self._load_teams(m)
        self._check_sheet_teams()
        s = p.status()

        def step(done: bool, txt: str) -> str:
            return f"<p>{'✅' if done else '⬜'} {txt}</p>"

        self.steps.setText(
            step(s["imported"], "Video imported (converted to a playable copy)")
            + step(s["analyzed"], "Players detected and tracked - automatic, in the background")
            + step(s["calibrated"], "Court calibrated - "
                   + ("automatically" + (f" + {s['keyframes']} keyframe(s)" if s["keyframes"] else "")
                      if s["auto_calibrated"] else f"{s['keyframes']} keyframe(s)")
                   + (". Automatic after the analysis (only if that fails: fix by hand in the Court fix tab)"
                      if not s["auto_calibrated"] else ". Automatic - nothing to do"))
            + step(s["replay_ready"] and not s["replay_stale"], "2D replay built (Replay tab)"))
        r = s["report"] if isinstance(s["report"], dict) else None
        try:
            self.report.setHtml(self._report_html(p, s, r))
        except Exception as e:  # noqa: BLE001 - Details must always show something, never nothing
            print("overview: report failed:", type(e).__name__, e)
            self.report.setHtml(f"<p style='color:#92400e'>Could not show the report ({html_escape(str(e))}). "
                                f"The log files are in {html_escape(str(p.results))}.</p>")

    def _report_html(self, p: Project, s: dict, r: dict | None) -> str:
        """The Report box: calibration summary (when built), the short 'Settings used', and - on Details - the
        diagnostics and all settings. Works with no report, an old report and missing files."""
        from replay.gui import overview_info as OI
        details = self.report_details_btn.isChecked()
        vfps = float(p.info().get("video_info", {}).get("fps") or 0) or None
        rows = OI.summary_rows(p.results, p.settings(), r, vfps)
        summary = OI.summary_html(rows)
        cfg = OI.read_settings(p.results)
        full = OI.settings_html(cfg) if cfg else (f"<p><b>All settings</b>: {OI.NOT_RECORDED} (no "
                                                  "config.used.json in this replay's results folder).</p>")
        if not r or not isinstance(r.get("calibration"), dict):
            if not details:
                return ("<p style='color:gray'>Available after the replay is built. Press <b>Details</b> to see "
                        "what has run so far and where the logs are.</p>" + summary)
            jobs = getattr(self.win, "jobs", None)
            job = jobs.state(p.id).public() if hasattr(jobs, "state") else {}
            return OI.details_without_report(p, s, job) + "<hr>" + summary + full
        cal = r["calibration"]
        fr = r.get("failed_ranges") or {}
        no_cal, no_pl = fr.get("no_calibration") or [], fr.get("no_players_tracked") or []

        def ranges(a):
            return ", ".join(f"<a href='seek:{x}'>{fmt_time(x)}–{fmt_time(y)}</a>" for x, y in a) or "none"

        warns = list(cal.get("warnings") or [])
        b = r.get("ball") or {}
        ball = (f"position known on {b['position_rate_pct']}% of frames (detected on {b.get('detection_rate_pct', '?')}%)"
                if b.get("position_rate_pct") is not None else "<i>not available - rebuild the replay</i>")
        valid = cal.get("valid_pct")
        gaps = len(no_cal)
        notes = []
        if valid is not None and valid < 80:
            notes.append(f"The court was found on only {valid}% of the video, so positions and stats from the rest "
                         "are less reliable.")
        if warns:
            notes.append(f"Calibration looks unreliable in {len(warns)} spot{'s' if len(warns) != 1 else ''} - "
                         "open <i>Court fix</i> to check them.")
        if gaps:
            notes.append(f"The court is lost in {gaps} stretch{'es' if gaps != 1 else ''} of the video.")
        note_html = "".join(f"<p style='color:#92400e'>&#9888; {n}</p>" for n in notes)             or "<p style='color:#166534'>Calibration looks good.</p>"
        head = f"""<table cellpadding=3>
            <tr><td>Court found</td><td><b>{'?' if valid is None else valid}%</b> of the video</td></tr>
            <tr><td>Ball</td><td>{ball}</td></tr></table>{note_html}"""
        if not details:
            return head + summary  # "Press Details for the technical diagnostics..." is behind the ? by the button
        ppl = r.get("people") or {}
        pfps = r.get("process_fps")
        warn = "<br>".join(html_escape(w) for w in warns)
        return head + f"""<hr><table cellpadding=3>
            <tr><td>Frames analysed</td><td>{r.get('frames_processed', '?')}{f' ({pfps:.2f} per second of video)' if isinstance(pfps, (int, float)) else ''}</td></tr>
            <tr><td>People tracked per frame</td><td>{ppl.get('avg_tracked_per_calibrated_frame', '?')}
                (measured {ppl.get('avg_measured_per_calibrated_frame', '?')})</td></tr>
            <tr><td>Player tracks</td><td>{ppl.get('tracks_output', '?')}</td></tr>
            <tr><td>No calibration</td><td>{ranges(no_cal)}</td></tr>
            <tr><td>No players tracked</td><td>{ranges(no_pl)}</td></tr>
            </table><p style='color:gray'>Click a time range to add a calibration keyframe there.<br>{warn}</p><hr>""" + summary + full

    def _check_sheet_teams(self) -> None:
        """Show the swap prompt when an attached scoresheet's rosters (by jersey numbers read in the video) say the
        home team wears the other kit. Recomputed only when the sheet, meta.json or the replay data change."""
        p = self.p
        jobs = getattr(self.win, "jobs", None)
        st = jobs.state(p.id) if hasattr(jobs, "state") else None
        if st is not None and (st.kind or st.queued):
            self.sheet_conflict.parentWidget().setVisible(False)  # being rebuilt: kits / labels may be changing
            self._conflict_key = None
            return
        files = [p.root / "scoresheet.json", p.meta_path, p.results / "replay_data.json"]
        key = (p.id,) + tuple(f.stat().st_mtime_ns if f.exists() else 0 for f in files)
        if key != self._conflict_key:
            self._conflict_key = key
            self._conflict = None
            if files[0].exists() and files[2].exists():
                try:
                    from replay import scoresheet as SS
                    self._conflict = SS.home_away_conflict(p.root)
                except Exception as e:  # noqa: BLE001 - a prompt must never break the Overview
                    print("overview: scoresheet team check failed:", type(e).__name__, e)
            c = self._conflict
            self.sheet_conflict.parentWidget().setVisible(c is not None)
            if c is not None:
                self.sheet_conflict.setText(f"<b>The scoresheet suggests {c['home_name']} = {c['colour']}.</b> "
                                            f"Swap? <span style='font-size:small'>{c['why']}. Now the "
                                            f"{c['colour']} team is set as away.</span>")

    def apply_sheet_teams(self) -> None:
        """One click: the kit the scoresheet points to becomes home, names from the sheet, Confirmed, Save."""
        c = self._conflict
        if c is None or self.p is None:
            return
        home_k = c["home_video_team"]
        away_k = "B" if home_k == "A" else "A"
        side = self.team_rows[home_k][3]
        side.setCurrentIndex(0)  # Home; _side_changed sets the other row to Away
        self._side_changed(home_k)
        for k, nm in ((home_k, c["home_name"]), (away_k, c["away_name"])):
            field = self.team_rows[k][2]
            cur = field.text().strip()
            if not cur or cur.lower().startswith("team ") or cur.lower() in ("home", "away"):
                field.setText(nm)
        self._teams_dirty = True
        if self.teams_ok.isEnabled():
            self.teams_ok.setChecked(True)
        self.save()
        self._conflict_key = None  # re-check after saving

    def _report_link(self, url) -> None:
        if url.toString() == "open:results" and self.p is not None:
            from PySide6.QtGui import QDesktopServices
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.p.results)))
        elif url.toString().startswith("seek:"):
            self.win.tabs.setCurrentWidget(self.win.calib_tab)
            self.win.calib_tab.seek(float(url.toString()[5:]))


# ====================================================================== calibrate tab
def _resolve_calibration(job: dict, emit) -> None:
    """Background thread: the calibration of every analysed frame, exactly as the replay build computes it.

    Emits {"sig", "t", "H", "warnings", "data"}. A job with "cams_at" (requested by "Snap to lines" with a manual
    calibration) instead emits {"sig", "cams"}: the physical camera position of the motion segment at that time,
    from the joint keyframe fit - several seconds, so only computed when needed.
    """
    data = load_pass1(job["work"])
    size, cfg = job["size"], job["cfg"]
    cal = CalibrationFile(video="video.mp4", frame_size=size, fixed_camera=job["fixed"],  # type: ignore[arg-type]
                          court=job["court"], auto=job["auto"])
    for k in job["kfs"]:
        kf = Keyframe(float(k["time_s"]), {n: tuple(v) for n, v in k["points"].items()},
                      source=k.get("source", "manual"))
        if len(kf.points) >= 4:
            try:
                kf.fit(size)
            except ValueError:
                pass
        cal.keyframes.append(kf)
    cal.keyframes.sort(key=lambda k: k.time_s)
    if job.get("cams_at") is None:
        res = resolve(data.t, data.motion, cal, cfg)
        emit({"sig": job["sig"], "t": np.asarray(data.t, float), "H": res.H, "warnings": list(res.warnings),
              "data": data})
        return
    usable = [k for k in cal.keyframes if k.points]
    cams: dict = {"seg": None, "C": {}}
    if job["fixed"]:
        cam: dict = {}
        _joint_fit([(np.eye(3), k) for k in usable], homography.floor_ref_pixel(size), size, cfg, [], {}, {}, cam)
        cams["fixed"] = cam.get("C")
    else:
        solver = FrameSolver(data.t, data.motion, size, cfg)
        cams["seg"] = solver.seg
        i = solver.index(float(job["cams_at"]))
        if i is not None:
            _, info = solver.solve(usable, float(job["cams_at"]))
            if "camera" in info:
                cams["C"][int(solver.seg[i])] = np.asarray(info["camera"]["C"], float)
    emit({"sig": job["sig"], "cams": cams})


class _CalibrationResolver(QObject):
    """Runs _resolve_calibration in one background thread; requests made meanwhile are coalesced (latest wins)."""

    finished = Signal(object)

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self._lock = threading.Lock()
        self._pending: dict | None = None
        self._busy = False

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._busy

    def request(self, job: dict) -> None:
        with self._lock:
            self._pending = job
            if self._busy:
                return
            self._busy = True
        threading.Thread(target=self._loop, daemon=True, name="calibration-resolve").start()

    def _loop(self) -> None:
        while True:
            with self._lock:
                job, self._pending = self._pending, None
                if job is None:
                    self._busy = False
                    return
            try:
                _resolve_calibration(job, self._emit)
            except Exception as e:  # e.g. a chunk being written by the analysis; retried on the next change
                traceback.print_exc()
                self._emit({"sig": job["sig"], "error": f"{type(e).__name__}: {e}"})

    def _emit(self, out: dict) -> None:
        try:
            self.finished.emit(out)
        except RuntimeError:  # the window was closed while this background resolve ran: nobody to tell
            pass


class CalibrateTab(QWidget):
    MIN_POINTS_ALONE = 4  # a keyframe on its own needs 4 points; combined with others, 2 are enough

    def __init__(self, win: "MainWindow"):
        super().__init__()
        self.win = win
        self.p: Project | None = None
        self.grabber: FrameGrabber | None = None
        # The calibration of every analysed frame is resolved in the background with the same code as the replay
        # build (calibration.resolve) - the joint keyframe fit takes seconds and must never block the GUI.
        self.resolver = _CalibrationResolver(self)
        self.resolver.finished.connect(self._resolved)
        self.res: dict | None = None  # latest background result (see _resolve_calibration)
        self._sig: tuple | None = None  # signature of the last resolve request
        self._chunks = -1
        self._cal_mtime: float | None = None
        self._auto_ver = 0
        self.auto: dict | None = None  # automatic calibration (calibration.json "auto")
        self.cur_H: np.ndarray | None = None  # image->court homography shown on the current frame
        self._cams: dict | None = None  # camera positions (manual calibration), for snapping
        self._snap_pending: float | None = None  # frame time of a snap waiting for the camera position
        self.source_note = ""
        self._resolve_timer = QTimer(self)
        self._resolve_timer.setSingleShot(True)
        self._resolve_timer.setInterval(250)
        self._resolve_timer.timeout.connect(self._request_resolve)
        self.frame_t = 0.0
        self.kfs: list[dict] = []
        self.fixed = False
        self.points: dict[str, tuple[float, float]] = {}
        self.order: list[str] = []
        self.selected: str | None = None
        self.lines: list[np.ndarray] | None = None
        self.fit_err: float | None = None
        self.per_point: dict[str, float] = {}
        self.dirty = False
        self.frame_img: np.ndarray | None = None
        self.p1 = None  # pass-1 data (detections) for masking players when matching lines
        self.camera: dict | None = None  # physical camera of the current frame's calibration (computed for snapping)
        self.snap_result = None

        lay = QHBoxLayout(self)
        left = QVBoxLayout()
        self.view = FrameView()
        self.view.clicked.connect(self.click_frame)
        left.addWidget(self.view, 1)
        tr = QHBoxLayout()
        self.play_btn = QPushButton("Play")
        self.play_btn.setFixedWidth(60)
        self.play_btn.clicked.connect(self.toggle_play)
        tr.addWidget(self.play_btn)
        for label, step in [("−10s", -10), ("−1s", -1), ("−0.1s", -0.1), ("+0.1s", 0.1), ("+1s", 1), ("+10s", 10)]:
            b = QPushButton(label)
            b.setFixedWidth(46)
            b.clicked.connect(lambda _=False, s=step: self.seek(self.frame_t + s))
            tr.addWidget(b)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setPageStep(100)  # 10 s per click on the groove
        # Every slider change (click on the groove, drag, keyboard) loads the frame, debounced so a drag
        # previews live without decoding every intermediate position.
        self._slider_timer = QTimer(self)
        self._slider_timer.setSingleShot(True)
        self._slider_timer.setInterval(120)
        self._slider_timer.timeout.connect(lambda: self.seek(self.slider.value() / 10.0))
        self.slider.valueChanged.connect(self._slider_moved)
        tr.addWidget(self.slider, 1)
        self.time_lbl = QLabel()
        self.time_lbl.setMinimumWidth(95)
        tr.addWidget(self.time_lbl)
        fit = QPushButton("Fit view")
        fit.clicked.connect(self.view.fit)
        tr.addWidget(fit)
        left.addLayout(tr)
        lay.addLayout(left, 1)
        self.play_timer = QTimer(self)
        self.play_timer.setInterval(100)
        self.play_timer.timeout.connect(self._play_step)

        side = QVBoxLayout()
        steps = QLabel("<b>1.</b> Pick a landmark on the court · <b>2.</b> click the same spot on the frame.")
        steps.setWordWrap(True)
        # the mouse/keyboard key line that was under the video is in the same ?
        side.addWidget(help_row(steps, HELP["calib_steps"] + "<br><br>" + HELP["calib_keys"],
                                "How to place points, mouse and keys"))
        self.diagram = CourtDiagram()
        self.diagram.picked.connect(self.pick)
        side.addWidget(self.diagram)
        hint = QLabel("<b>x = 0 is the LEFT basket and the bottom edge is the NEAR sideline, as seen from the "
                      "camera.</b>")
        hint.setWordWrap(True)
        side.addWidget(hint)
        self.fit_info = QLabel()
        self.fit_info.setWordWrap(True)
        self.fit_info.setMinimumHeight(70)
        self.fit_info.setStyleSheet("padding:6px; border:1px solid palette(mid); border-radius:4px;")
        side.addWidget(help_row(self.fit_info, HELP["calib_fit"], "The yellow lines"))
        btns = QHBoxLayout()
        self.save_btn = QPushButton("Save keyframe")
        self.save_btn.setDefault(True)
        self.save_btn.clicked.connect(self.save_keyframe)
        undo = QPushButton("Undo point")
        undo.clicked.connect(self.undo)
        clear = QPushButton("Clear")
        clear.clicked.connect(self.clear_points)
        for b in (self.save_btn, undo, clear):
            btns.addWidget(b)
        side.addLayout(btns)
        self.snap_btn = QPushButton("Snap to lines")
        self.snap_btn.setToolTip("Fine-tune this frame's calibration so the yellow lines sit exactly on the court "
                                 "lines visible in the video. You see the result (green) before accepting it.")
        self.snap_btn.clicked.connect(self.snap)
        self.swap_btn = QPushButton("Swap baskets")
        self.swap_btn.setToolTip("If the left and right baskets were mixed up when clicking this frame's points")
        self.swap_btn.clicked.connect(self.swap_baskets)
        self.worst_btn = QPushButton("Remove worst point")
        self.worst_btn.setToolTip("Delete the point that disagrees most with the rest (shown in red)")
        self.worst_btn.clicked.connect(self.remove_worst)
        tools = QGridLayout()  # two columns: three buttons in one row are wider than the side panel
        for i, b in enumerate((self.snap_btn, self.swap_btn, self.worst_btn)):
            tools.addWidget(b, i // 2, i % 2)
        side.addLayout(tools)
        snaprow = QHBoxLayout()
        self.accept_snap_btn = QPushButton("Accept snap (green)")
        self.accept_snap_btn.clicked.connect(self.accept_snap)
        self.reject_snap_btn = QPushButton("Reject")
        self.reject_snap_btn.clicked.connect(self.reject_snap)
        for b in (self.accept_snap_btn, self.reject_snap_btn):
            b.setVisible(False)
            snaprow.addWidget(b)
        side.addLayout(snaprow)

        self.fixed_cb = QCheckBox("Camera never moves")
        self.fixed_cb.setToolTip("All keyframes show the same view")
        self.fixed_cb.toggled.connect(self.set_fixed)
        side.addWidget(help_row(self.fixed_cb, HELP["calib_fixed"], "Camera never moves"))
        self.gemini_cb = QCheckBox("Gemini court finder (experimental)")  # opt-in, OFF (gemini_court.py)
        self.gemini_cb.setToolTip("Experimental: not reliable yet - in a first test it found too few court views.")
        self.gemini_cb.toggled.connect(self.set_gemini)
        # the long text (cost, key, caps, last run) is behind this ? (it was a grey paragraph); filled by _gemini_text
        gem_row = help_row(self.gemini_cb, HELP["calib_gemini"], "Gemini court finder")
        self.gemini_info: HelpIcon = gem_row.help_icon
        side.addWidget(gem_row)
        caps = QGridLayout()  # hard Gemini spending caps (local app setting, gemini_court.Ledger)
        from replay import gemini_court as _gc
        _led = _gc.Ledger()
        self.gemini_game_cap, self.gemini_month_cap = QDoubleSpinBox(), QDoubleSpinBox()
        for sb, val, lab in ((self.gemini_game_cap, _led.game_cap, "Max EUR per game"),
                             (self.gemini_month_cap, _led.month_cap, "per month")):
            sb.setRange(0.0, 50.0)
            sb.setDecimals(2)
            sb.setSingleStep(0.05)
            sb.setValue(val)
            sb.valueChanged.connect(self._gemini_caps)
            caps.addWidget(QLabel(lab), 0 if sb is self.gemini_game_cap else 1, 0)
            caps.addWidget(sb, 0 if sb is self.gemini_game_cap else 1, 1)
        side.addLayout(caps)
        side.addWidget(titled("<b>Keyframes</b>", HELP["calib_keyframes"] + "<br><br>" + HELP["calib_fit"],
                              "Keyframes"))
        self.kf_list = QListWidget()
        self.kf_list.itemDoubleClicked.connect(lambda it: self._go(it))
        side.addWidget(self.kf_list, 1)
        kb = QHBoxLayout()
        go = QPushButton("Go to")
        go.clicked.connect(lambda: self._go(self.kf_list.currentItem()))
        rm = QPushButton("Delete keyframe")
        rm.clicked.connect(self.delete_keyframe)
        kb.addWidget(go)
        kb.addWidget(rm)
        side.addLayout(kb)
        side_w = QWidget()
        side_w.setLayout(side)
        side_w.setMinimumHeight(560)  # scrolls in a low window instead of forcing its height
        from PySide6.QtWidgets import QScrollArea
        side_sc = QScrollArea()
        side_sc.setWidget(side_w)
        side_sc.setWidgetResizable(True)
        side_sc.setFrameShape(QScrollArea.Shape.NoFrame)
        side_sc.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)  # last resort, never clip silently
        side_sc.setFixedWidth(350)
        lay.addWidget(side_sc)

        for key, fn in [(Qt.Key.Key_Left, lambda: self.seek(self.frame_t - 1)),
                        (Qt.Key.Key_Right, lambda: self.seek(self.frame_t + 1)),
                        (Qt.Key.Key_Up, lambda: self.seek(self.frame_t + 10)),
                        (Qt.Key.Key_Down, lambda: self.seek(self.frame_t - 10)),
                        (Qt.Key.Key_Comma, lambda: self.seek(self.frame_t - 0.1)),
                        (Qt.Key.Key_Period, lambda: self.seek(self.frame_t + 0.1)),
                        (Qt.Key.Key_Space, self.toggle_play),
                        (Qt.Key.Key_Return, self.save_keyframe), (Qt.Key.Key_Enter, self.save_keyframe)]:
            sc = QShortcut(QKeySequence(key), self)
            sc.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            sc.activated.connect(fn)
        sc = QShortcut(QKeySequence.StandardKey.Undo, self)
        sc.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        sc.activated.connect(self.undo)

    def _go(self, item: QListWidgetItem | None) -> None:
        if item is not None and item.data(Qt.ItemDataRole.UserRole) is not None:
            self.seek(item.data(Qt.ItemDataRole.UserRole))

    def _duration(self) -> float:
        return self.grabber.info.duration_s if self.grabber else 0.0

    def _size(self) -> tuple[int, int]:
        if self.grabber:
            return self.grabber.info.size
        return tuple(self.p.info().get("video_info", {}).get("size", (1280, 720))) if self.p else (1280, 720)

    def _cfg(self) -> CalibrationConfig:
        return self.p.config().calibration if self.p else CalibrationConfig()

    def load(self, p: Project) -> None:
        self.stop_play()
        if self.grabber:
            self.grabber.close()
        self.p, self.grabber, self.res, self._sig, self._chunks, self.p1 = p, None, None, None, -1, None
        self._resolve_timer.stop()
        self._load_calibration(keyframes=True)
        self._clear()
        self.render_list()

    def _load_calibration(self, keyframes: bool) -> None:
        """Read calibration.json: the automatic calibration (and, unless points are being edited, the keyframes)."""
        p = self.p
        if p is None:
            return
        try:
            self._cal_mtime = p.calibration_path.stat().st_mtime
        except OSError:
            self._cal_mtime = None
        try:
            cal = p.calibration()
        except Exception as e:  # damaged file: keep what we have, say so
            self.update_info(f"<b>calibration.json could not be read:</b> {e}")
            return
        self.auto = cal.auto if isinstance(cal.auto, dict) else None
        self._auto_ver += 1
        if keyframes:
            self.kfs = [{"time_s": k.time_s, "points": dict(k.points), "source": k.source} for k in cal.keyframes]
            self.fixed = cal.fixed_camera
            self.fixed_cb.blockSignals(True)
            self.fixed_cb.setChecked(self.fixed)
            self.fixed_cb.blockSignals(False)
            self.gemini_cb.blockSignals(True)
            self.gemini_cb.setChecked(bool(self.p.settings().get("gemini_court")) if self.p else False)
            self.gemini_cb.blockSignals(False)
            self._gemini_text()
        self.diagram.update()  # the court size may have been measured

    def _auto_mode(self) -> bool:
        """An automatic calibration is in use (keyframes then refine it; see calibration._resolve_auto)."""
        return bool(self.auto and not self.fixed and any(a.get("good") for a in self.auto.get("anchors", [])))

    def _auto_C(self, t: float) -> np.ndarray:
        anc = [a for a in (self.auto or {}).get("anchors", []) if a.get("good") and "C" in a]
        if anc:
            return np.asarray(min(anc, key=lambda a: abs(float(a["t"]) - t))["C"], float)
        return np.asarray(self.auto["C"], float)

    def _n_chunks(self) -> int:
        work = self.p.results / "work" if self.p else None
        return len(list(work.glob("chunk_*.npz"))) if work is not None and work.exists() else 0

    def refresh_solver(self) -> None:
        """Called on every poll while the tab is shown: picks up new analysis chunks and calibration.json changes
        (e.g. written by the automatic calibration). Cheap when nothing changed; the work runs in the background."""
        if self.p is None:
            return
        try:
            mt = self.p.calibration_path.stat().st_mtime
        except OSError:
            mt = None
        if mt != self._cal_mtime:
            self._load_calibration(keyframes=not self.dirty)
            self.render_list()
        self._chunks = self._n_chunks()
        self._schedule_resolve()

    # -------------------------------------------------------------- background resolve
    def _kfs_for_resolve(self) -> list[dict]:
        """Saved keyframes, with this frame's points as they are now (so the lines show the result of saving)."""
        out = [k for k in self.kfs if not (self.dirty and abs(k["time_s"] - self.frame_t) < 0.06)]
        if self.dirty and len(self.points) >= (3 if self._auto_mode() else 2):
            out.append({"time_s": self.frame_t, "points": dict(self.points), "source": "manual"})
        return sorted(out, key=lambda k: k["time_s"])

    def _make_sig(self, kfs: list[dict]) -> tuple:
        return (self.p.id if self.p else None, self._chunks, self.fixed, self._auto_ver,
                json.dumps(kfs, sort_keys=True, default=float))

    def _schedule_resolve(self) -> None:
        if self.p is None or self._chunks <= 0:
            return
        if self._make_sig(self._kfs_for_resolve()) != self._sig:
            self._resolve_timer.start()  # debounced: clicking several points starts one resolve

    def _request_resolve(self) -> None:
        p = self.p
        if p is None or self._chunks <= 0:
            return
        kfs = self._kfs_for_resolve()
        sig = self._make_sig(kfs)
        if sig == self._sig:
            return
        self._sig = sig
        if not kfs and not self._auto_mode():
            self.res = {"sig": sig, "t": np.zeros(0), "H": [], "warnings": []}
            self.fit()
            return
        self.resolver.request(self._job(sig, kfs))
        self.update_info()

    def _job(self, sig: tuple, kfs: list[dict], **extra) -> dict:
        cal_court = None
        try:
            cal_court = json.loads(self.p.calibration_path.read_text(encoding="utf-8")).get("court")
        except (OSError, ValueError, AttributeError):
            pass
        return {"sig": sig, "work": self.p.results / "work", "size": self._size(), "cfg": self._cfg(),
                "fixed": self.fixed, "auto": self.auto, "court": cal_court, "kfs": kfs, **extra}

    def _resolved(self, out: dict) -> None:
        sig = out.get("sig")
        if self.p is None or not sig or sig[0] != self.p.id:
            return  # another Replay's result
        if "error" in out:
            if sig == self._sig:
                self._sig = None  # try again on the next poll
            return
        if "cams" in out:  # camera position for snapping (requested by snap())
            if sig == self._sig:
                c = out["cams"]
                if self._cams is None or self._cams.get("sig") != sig:
                    self._cams = {"sig": sig, "seg": c.get("seg"), "C": {}, "fixed": None}
                self._cams["C"].update(c.get("C") or {})
                self._cams["fixed"] = c.get("fixed", self._cams["fixed"])
                if self._snap_pending is not None and abs(self._snap_pending - self.frame_t) < 1e-6:
                    self._snap_pending = None
                    self.snap()
            return
        if sig != self._sig and self.res is not None and self.res.get("sig") == self._sig:
            return  # stale; the current result is already shown
        self.res = out
        if out.get("data") is not None:
            self.p1 = out["data"]
        if self.grabber is not None and not self.play_timer.isActive():
            self.fit()
            self.render_list()

    def _res_H(self, t: float) -> np.ndarray | None:
        r = self.res
        if not r or not len(r["t"]):
            return None
        i = int(np.argmin(np.abs(r["t"] - t)))
        if abs(r["t"][i] - t) > self._cfg().keyframe_snap_s:
            return None
        return r["H"][i]

    def _res_current(self) -> bool:
        return self.res is not None and self.res.get("sig") == self._sig and not self._resolve_timer.isActive()

    def ensure_frame(self) -> None:
        """Lazy: the video is opened only when the tab is shown."""
        if self.p is None or not self.p.video.exists():
            self.fit_info.setText("The video is still being imported…")
            return
        if self.grabber is None:
            self.grabber = FrameGrabber(self.p.video)
            self.slider.blockSignals(True)
            self.slider.setRange(0, int(self.grabber.info.duration_s * 10))
            self.slider.blockSignals(False)
            self.view._fitted = False
            self.refresh_solver()
            self.seek(self.kfs[0]["time_s"] if self.kfs else 0.0, force=True)

    def _slider_moved(self, v: int) -> None:
        self.time_lbl.setText(f"{fmt_time(v / 10)} / {fmt_time(self._duration())}")
        self._slider_timer.start()

    # -------------------------------------------------------------- playback
    def toggle_play(self) -> None:
        if self.play_timer.isActive():
            self.stop_play()
        elif self.grabber is not None:
            if self.dirty and self.points:
                self.update_info("Save or clear the points on this frame before playing.")
                return
            self.play_timer.start()
            self.play_btn.setText("Pause")

    def stop_play(self) -> None:
        if self.play_timer.isActive():
            self.play_timer.stop()
            self.render_list()
        self.play_btn.setText("Play")

    def _play_step(self) -> None:
        if self.frame_t >= self._duration() - 0.15:
            self.stop_play()
            return
        self.seek(self.frame_t + 0.1)

    def seek(self, t: float, force: bool = False) -> None:
        if self.grabber is None:
            self.ensure_frame()
            if self.grabber is None:
                return
        if self.dirty and self.points and not force and abs(t - self.frame_t) > 0.02:
            if QMessageBox.question(self, APP_NAME, "Discard the unsaved points on this frame?") != \
                    QMessageBox.StandardButton.Yes:
                self._set_slider(self.frame_t)
                return
        res = self.grabber.get(t)
        if res is None:
            return
        self.frame_t, img = res
        self.frame_img = img
        self.snap_result = None
        self._snap_pending = None
        self._show_snap_buttons(False)
        self.view.set_frame(img)
        self._set_slider(self.frame_t)
        self.time_lbl.setText(f"{fmt_time(self.frame_t)} / {fmt_time(self._duration())}")
        kf = self._kf_here()
        was_dirty = self.dirty
        self._clear()
        if kf:
            self.points = {n: tuple(v) for n, v in kf["points"].items()}
            self.order = list(self.points)
        if was_dirty:
            self._schedule_resolve()  # the discarded points no longer count
        self.fit()
        if not self.play_timer.isActive():
            self.render_list()

    def _set_slider(self, t: float) -> None:
        self.slider.blockSignals(True)
        self.slider.setValue(int(round(t * 10)))
        self.slider.blockSignals(False)

    def _kf_here(self) -> dict | None:
        return next((k for k in self.kfs if abs(k["time_s"] - self.frame_t) < 0.06), None)

    # -------------------------------------------------------------- points
    def pick(self, name: str) -> None:
        self.selected = name
        self.update_info()

    def _points_changed(self) -> None:
        self.dirty = True
        self._schedule_resolve()
        self.fit()

    def click_frame(self, x: float, y: float) -> None:
        if not self.selected:
            self.update_info("Pick a landmark on the court diagram first.")
            return
        self.stop_play()
        self.points[self.selected] = (round(x, 1), round(y, 1))
        self.order = [n for n in self.order if n != self.selected] + [self.selected]
        self.selected = None
        self.diagram.selected = None
        self._points_changed()

    def undo(self) -> None:
        if self.order:
            self.points.pop(self.order.pop(), None)
            self._points_changed()

    def _clear(self) -> None:
        self.points, self.order, self.lines, self.fit_err, self.per_point, self.dirty = {}, [], None, None, {}, False

    def clear_points(self) -> None:
        self._clear()
        self._points_changed()

    def _live_H(self) -> np.ndarray | None:
        """Calibration from this frame's points alone (immediate, while the combined one is computed)."""
        names = [n for n in self.points if n in court.KEYPOINTS]
        size = self._size()
        if self._auto_mode() and len(names) >= 3:  # same as the build: the points re-aim the automatic camera
            C = self._auto_C(self.frame_t)
            fit = camera_model.fit_camera([(np.array([court.KEYPOINTS[n] for n in names]),
                                            np.array([self.points[n] for n in names]))], size, C_fixed=C)
            if fit is not None:
                H = homography.normalize(np.linalg.inv(cam_court_to_image(C, fit.views[0], size)),
                                         homography.floor_ref_pixel(size))
                if homography.is_sane(H, size):
                    return H
        if len(names) >= 4:
            kf = Keyframe(self.frame_t, {n: self.points[n] for n in names})
            try:
                kf.fit(size)
                return kf.H
            except ValueError as e:
                self.update_info(f"Fit failed: {e}")
        return None

    def fit(self) -> None:
        """Yellow lines for the current frame + the error of each point on it. Never slow: the combined
        calibration comes from the background resolve; unsaved points get an immediate single-frame fit."""
        self.lines, self.fit_err, self.per_point, self.camera, self.cur_H = None, None, {}, None, None
        H, current = None, self._res_current()
        n_kf = len(self._kfs_for_resolve())
        if self.dirty and self.points and not current:
            H = self._live_H()
            self.source_note = "this frame's points only - combining with the rest…" if H is not None else ""
        if H is None:
            H = self._res_H(self.frame_t)
            if H is not None:
                if self._auto_mode():
                    self.source_note = "<b>automatic calibration</b>" + (
                        f" refined by {n_kf} keyframe(s)" if n_kf else "")
                else:
                    self.source_note = f"<b>{n_kf} keyframe(s) combined</b>" + (
                        "" if self._kf_here() or self.points else " - no keyframe on this frame")
                if not current:
                    self.source_note += " (updating…)"
        if H is not None:
            self.cur_H = H
            self.lines = [homography.court_to_image(H, l) for l in court.court_polylines()]
            names = [n for n in self.points if n in court.KEYPOINTS]
            if names:
                proj = homography.court_to_image(H, np.array([court.KEYPOINTS[n] for n in names], float))
                errs = np.linalg.norm(proj - np.array([self.points[n] for n in names], float), axis=1)
                errs = np.where(np.isfinite(errs), errs, 999.0)
                self.per_point = dict(zip(names, errs.tolist()))
                self.fit_err = float(np.sqrt(np.mean(errs ** 2)))
        self._redraw()
        self.diagram.placed = set(self.points)
        self.diagram.update()
        self.update_info()

    BAD_PX = 8.0

    def _redraw(self) -> None:
        colors = {n: ("#ef4444" if e > self.BAD_PX else "#22c55e") for n, e in self.per_point.items()}
        notes = {n: f"  ({e:.0f} px off)" for n, e in self.per_point.items() if e > self.BAD_PX}
        preview = None
        if self.snap_result is not None and self.camera is not None:
            Hs = cam_court_to_image(self.camera["C"], self.snap_result.ptz, self._size())
            Hi = homography.normalize(np.linalg.inv(Hs), homography.floor_ref_pixel(self._size()))
            preview = [homography.court_to_image(Hi, l) for l in court.court_polylines()]
        self.view.set_overlay(self.lines, self.points, colors, preview, notes)

    def _show_snap_buttons(self, on: bool) -> None:
        self.accept_snap_btn.setVisible(on)
        self.reject_snap_btn.setVisible(on)

    def _persons_here(self) -> np.ndarray | None:
        if self.p1 is None or not len(self.p1.t):
            return None
        i = int(np.argmin(np.abs(self.p1.t - self.frame_t)))
        if abs(self.p1.t[i] - self.frame_t) > 0.2:
            return None
        r = self.p1.rows
        return r[(r[:, 0] == self.p1.k[i]) & (r[:, 6] == 0)][:, 1:5]

    def _camera_here(self) -> dict | None:
        """Physical camera (position C + view) of the calibration shown on this frame, for snapping."""
        H, size = self.cur_H, self._size()
        if H is None:
            return None
        C = None
        cams = self._cams if self._cams is not None and self._cams.get("sig") == self._sig else {}
        if self._auto_mode():
            C = self._auto_C(self.frame_t)
        elif self.fixed:
            C = cams.get("fixed")
        elif cams.get("seg") is not None and self.res is not None and len(self.res["t"]) == len(cams["seg"]):
            i = int(np.argmin(np.abs(self.res["t"] - self.frame_t)))
            C = cams["C"].get(int(cams["seg"][i]))
        if C is None:
            return None
        H_c2i = np.linalg.inv(H)
        init = camera_model.init_from_homography(C, H_c2i, size, float(size[0]))
        if init is None:
            return None
        view = camera_model.fit_ptz_to_homography(C, H_c2i, size, init) or init
        return {"C": np.asarray(C, float), "view": view}

    def snap(self) -> None:
        if self.cur_H is None or self.frame_img is None:
            self.update_info("Snapping needs a rough calibration first: place at least 3-4 points (or have a "
                             "keyframe nearby), so the yellow lines appear on this frame.")
            return
        self.stop_play()
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self.camera = self._camera_here()
            if self.camera is None:
                if self._auto_mode() or self._sig is None or self._snap_pending is not None:
                    self.update_info("No camera position for this frame yet - try again in a moment.")
                    return
                # manual calibration: the camera position comes from the joint keyframe fit (seconds) - in the
                # background; snapping continues by itself when it is ready
                self._snap_pending = self.frame_t
                self.resolver.request(self._job(self._sig, self._kfs_for_resolve(), cams_at=self.frame_t))
                self.update_info("Measuring the camera position from the keyframes (a few seconds)… The snap "
                                 "continues automatically.")
                return
            ol = oriented_lines(self.frame_img, self._persons_here(), self.p.meta().get("overlay_regions", []))
            r = snap_view(self.camera["C"], self.camera["view"], ol)
        finally:
            QApplication.restoreOverrideCursor()
        self.snap_result = r
        self._redraw()
        self._show_snap_buttons(True)
        if r.accepted:
            self.update_info(f"<b>Snap found a better fit</b> (green). Match error {r.before:.1f} -> {r.after:.1f} px, "
                             f"{r.inliers * 100:.0f}% of the visible court lines matched. Accept if the green lines "
                             "sit on the real court lines.")
        else:
            self.update_info(f"<b>Not confident:</b> {r.reason}. The green lines show the best attempt - accept "
                             "only if they are clearly right. Otherwise add or fix points by hand.")

    def accept_snap(self) -> None:
        r = self.snap_result
        if r is None or self.camera is None:
            return
        pts = keypoints_from_view(self.camera["C"], r.ptz, self._size())
        if len(pts) < 3 or not self._fittable(pts):
            self.update_info("Too few court landmarks in view (or all on one line) to store this snap as a "
                             "keyframe.")
            return
        replaced = self._kf_here() is not None
        self.kfs = [k for k in self.kfs if abs(k["time_s"] - self.frame_t) >= 0.06]
        self.kfs.append({"time_s": self.frame_t, "points": pts, "source": "snap"})
        self.kfs.sort(key=lambda k: k["time_s"])
        self.snap_result = None
        self._show_snap_buttons(False)
        self._clear()
        self.points, self.order = dict(pts), list(pts)
        self.persist()
        self.fit()
        self.update_info(("Updated" if replaced else "Saved") + f" a snapped keyframe at {fmt_time(self.frame_t)}.")

    def reject_snap(self) -> None:
        self.snap_result = None
        self._show_snap_buttons(False)
        self._redraw()
        self.update_info()

    def swap_baskets(self) -> None:
        if not self.points:
            self.update_info("No points on this frame to swap.")
            return
        self.points = swap_left_right(self.points)
        self.order = list(self.points)
        self._points_changed()

    def remove_worst(self) -> None:
        if not self.per_point:
            self.update_info("No point errors to judge yet.")
            return
        name, e = max(self.per_point.items(), key=lambda kv: kv[1])
        self.points.pop(name, None)
        self.order = [n for n in self.order if n != name]
        self._points_changed()
        self.update_info(f"Removed {name} ({e:.0f} px off). Save the keyframe to keep the change.")

    def _fittable(self, points: dict) -> bool:
        """Keyframes with 4+ points are fitted on their own when loaded (CalibrationFile.load): points on one line
        would make every later build fail."""
        names = [n for n in points if n in court.KEYPOINTS]
        if len(names) < 4:
            return True
        try:
            Keyframe(self.frame_t, {n: tuple(points[n]) for n in names}).fit(self._size())
            return True
        except ValueError:
            return False

    def _min_points(self) -> int:
        if self._auto_mode():
            return 3  # re-aims the automatically measured camera
        others = [k for k in self.kfs if abs(k["time_s"] - self.frame_t) >= 0.06]
        return 2 if others else self.MIN_POINTS_ALONE

    def _can_save(self) -> bool:
        return len(self.points) >= self._min_points()

    def update_info(self, msg: str | None = None) -> None:
        n = len(self.points)
        if msg:
            self.fit_info.setText(msg)
            return
        parts = []
        if self.lines is not None and self.source_note:
            parts.append(f"Yellow lines: {self.source_note}")
        elif self.lines is None:
            if self._chunks <= 0 and not self.fixed:
                parts.append("Lines appear once the analysis has measured the camera motion.")
            elif self.resolver.busy or self._resolve_timer.isActive():
                parts.append("Calibrating…")
            elif self.kfs or self._auto_mode():
                parts.append("No calibration at this moment (not analysed here, or too far from the calibrated "
                             "frames). Add a keyframe here.")
            else:
                parts.append("Not calibrated yet.")
        here = f"{self.frame_t:.1f}s"
        parts += [f"<span style='color:#b45309'>{w}</span>" for w in (self.res or {}).get("warnings", [])
                  if here in w and "keyframe" in w]
        if n == 0:
            parts.append(f"Now click <b>{self.selected}</b> on the frame." if self.selected
                         else "No points on this frame. Pick a landmark on the diagram.")
        else:
            need = self._min_points() - n
            line = f"{n} point(s) on this frame" + (f", {need} more needed" if need > 0 else "")
            if self.fit_err is not None:
                e = self.fit_err
                q = "good" if e < 4 else "okay - check the lines" if e < 10 else "poor - a point is probably wrong"
                line += f" · error <b>{e:.1f} px</b> ({q})"
            parts.append(line)
            if n == 4 and not self._auto_mode() and len(self._kfs_for_resolve()) <= 1:
                parts.append("<i>With exactly 4 points on one keyframe the fit is always perfect - add a 5th "
                             "point or another keyframe to verify it.</i>")
            worst = max(self.per_point.items(), key=lambda kv: kv[1]) if self.per_point else None
            if worst and worst[1] > 6:
                parts.append(f"Largest error: {worst[0]} ({worst[1]:.1f} px)")
            if self.selected:
                parts.append(f"Now click <b>{self.selected}</b>.")
        self.fit_info.setText("<br>".join(parts))

    # -------------------------------------------------------------- keyframes
    def save_keyframe(self) -> None:
        if not self._can_save():
            self.update_info(f"This keyframe needs at least {self._min_points()} points"
                             + (" (on top of the automatic calibration)." if self._auto_mode() else
                                "; the first keyframe needs 4, later ones 2."))
            return
        if not self._fittable(self.points):
            self.update_info("<b>Not saved:</b> these points lie (almost) on one line, so they cannot fix the court "
                             "in place. Add a point away from that line (e.g. on the other sideline or the "
                             "free-throw line), or remove points so that 3 remain.")
            return
        if self.fit_err and self.fit_err > 10 and QMessageBox.question(
                self, APP_NAME, f"The error is {self.fit_err:.1f} px, which suggests a misplaced point. "
                                "Save anyway?") != QMessageBox.StandardButton.Yes:
            return
        replaced = self._kf_here() is not None
        self.kfs = [k for k in self.kfs if abs(k["time_s"] - self.frame_t) >= 0.06]
        self.kfs.append({"time_s": self.frame_t, "points": dict(self.points), "source": "manual"})
        self.kfs.sort(key=lambda k: k["time_s"])
        self.dirty = False
        self.persist()
        self.update_info(f"{'Updated the' if replaced else 'Saved a new'} keyframe at {fmt_time(self.frame_t)} "
                         f"({len(self.kfs)} keyframe{'s' if len(self.kfs) != 1 else ''} in total). "
                         "Move to another moment to add the next one.")

    def delete_keyframe(self) -> None:
        it = self.kf_list.currentItem()
        if it is None or it.data(Qt.ItemDataRole.UserRole) is None:
            return
        t = it.data(Qt.ItemDataRole.UserRole)
        if QMessageBox.question(self, APP_NAME, f"Delete the keyframe at {fmt_time(t)}?") != \
                QMessageBox.StandardButton.Yes:
            return
        self.kfs = [k for k in self.kfs if abs(k["time_s"] - t) > 1e-6]
        if abs(t - self.frame_t) < 0.06:
            self._clear()
        self.persist()
        self.fit()

    def _gemini_text(self) -> None:
        from replay import gemini_court as gc

        led = gc.Ledger()
        dur = float((self.p.info().get("video_info", {}) or {}).get("duration_s") or 0) if self.p else 0.0
        n = min(gc.MAX_FRAMES, max(1, int(dur / gc.SHOT_EVERY_S) + 1))
        last = led.last_entries(1)
        last_s = (f" Last run: {last[0]['date']}, {last[0]['frames']} frames, {last[0]['eur']:.3f} EUR."
                  if last else "")
        self.gemini_info.set_help(
            "Experimental - not reliable yet. "
            f"Sends up to {n} video frames (downscaled) to Google with your own AI Studio key - the frames leave this "
            f"PC. Cost at most about {gc.estimate_cost(n):.3f} EUR for this game (hard cap {led.game_cap:.2f} EUR per "
            f"game). {led.summary()}; when a cap is reached nothing more is sent ('Gemini budget reached'). Answers are "
            f"cached in the project, so rebuilding never sends again.{last_s} Takes effect at the next automatic "
            "calibration.")

    def _gemini_caps(self, *_):
        from replay import gemini_court as gc

        gc.Ledger().set_caps(self.gemini_game_cap.value(), self.gemini_month_cap.value())
        self._gemini_text()

    def set_gemini(self, on: bool) -> None:
        p = self.p
        if p is None:
            return
        st = dict(p.info().get("settings", {}))
        st["gemini_court"] = bool(on)
        p.update_info(settings=st)
        self._gemini_text()

    def set_fixed(self, on: bool) -> None:
        self.fixed = on
        self.persist()
        self.fit()

    def persist(self) -> None:
        p = self.p
        if p is None:
            return
        try:
            cal = p.calibration()  # keeps the automatic calibration and the measured court size
        except Exception:  # e.g. an old keyframe that no longer fits: rebuild it, keeping the automatic part
            raw = {}
            try:
                raw = json.loads(p.calibration_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                pass
            raw = raw if isinstance(raw, dict) else {}
            cal = CalibrationFile(video="video.mp4", frame_size=self._size(),  # type: ignore[arg-type]
                                  court=raw.get("court"), auto=raw.get("auto"))
        cal.video, cal.frame_size, cal.fixed_camera = "video.mp4", self._size(), self.fixed
        cal.keyframes = []
        for k in self.kfs:
            kf = Keyframe(k["time_s"], {n: tuple(v) for n, v in k["points"].items()}, source=k.get("source", "manual"))
            if len(kf.points) >= 4:
                try:
                    kf.fit(cal.frame_size)
                except ValueError:
                    pass
            cal.keyframes.append(kf)
        cal.save(p.calibration_path)
        self.auto = cal.auto if isinstance(cal.auto, dict) else None
        try:
            self._cal_mtime = p.calibration_path.stat().st_mtime
        except OSError:
            pass
        st = p.settings()
        st["fixed_camera"] = self.fixed
        p.update_info(settings=st)
        if p.status()["analyzed"] and cal.calibrated:
            self.win.jobs.submit(p.id, BUILD)
        self._schedule_resolve()
        self.render_list()
        self.win.refresh_project()

    def render_list(self) -> None:
        self.kf_list.clear()
        if not self.kfs:
            self.kf_list.addItem("None yet." + (" The automatic calibration is used." if self._auto_mode() else ""))
            return
        current = self._res_current()
        for k in self.kfs:
            H = self._res_H(k["time_s"]) if self.res else None
            e = None
            names = [n for n in k["points"] if n in court.KEYPOINTS]
            if H is not None and names:
                proj = homography.court_to_image(H, np.array([court.KEYPOINTS[n] for n in names], float))
                d = np.linalg.norm(proj - np.array([k["points"][n] for n in names], float), axis=1)
                e = float(np.sqrt(np.mean(np.where(np.isfinite(d), d, 999.0) ** 2)))
            if e is not None:
                err = f"{e:.1f} px" + ("" if current else " …")
            else:
                err = "calibrating…" if self.resolver.busy or self._resolve_timer.isActive() else \
                    "not analysed yet" if self._chunks <= 0 or not self.res else "no calibration here"
            src = {"snap": "  · snapped", "auto": "  · auto"}.get(k.get("source", "manual"), "")
            it = QListWidgetItem(f"{fmt_time(k['time_s'])}   ·   {len(k['points'])} pts   ·   {err}{src}")
            it.setData(Qt.ItemDataRole.UserRole, k["time_s"])
            if e is not None and e > 10:
                it.setForeground(QColor("#dc2626"))
                it.setToolTip("This keyframe disagrees with the combined calibration - a point is probably misplaced"
                              + (", or it does not fit the automatically measured camera." if self._auto_mode()
                                 else "."))
            if abs(k["time_s"] - self.frame_t) < 0.06:
                f = it.font()
                f.setBold(True)
                it.setFont(f)
            self.kf_list.addItem(it)


# ====================================================================== replay tab
class VideoOverlayView(QGraphicsView):
    """Video (QGraphicsVideoItem) with tracking boxes drawn on top, in video pixel coordinates."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.item = QGraphicsVideoItem()
        self.scene().addItem(self.item)
        self.setBackgroundBrush(QColor("#000"))
        self.setFrameShape(QGraphicsView.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setRenderHints(QPainter.RenderHint.Antialiasing)
        self._items: list = []
        self.set_video_size(1280, 720)

    def set_video_size(self, w: int, h: int) -> None:
        self.item.setSize(QSizeF(w, h))
        self.scene().setSceneRect(QRectF(0, 0, w, h))
        self.fitInView(self.item, Qt.AspectRatioMode.KeepAspectRatio)

    def resizeEvent(self, e) -> None:
        super().resizeEvent(e)
        self.fitInView(self.item, Qt.AspectRatioMode.KeepAspectRatio)

    def set_boxes(self, boxes: list[tuple[list[float], str, str, bool]]) -> None:
        """boxes: [(xyxy, colour, label, dashed)]"""
        for it in self._items:
            self.scene().removeItem(it)
        self._items = []
        font = QFont()
        font.setPixelSize(16)
        font.setBold(True)
        for (x1, y1, x2, y2), col, label, dashed in boxes:
            pen = QPen(QColor(col), 2.5, Qt.PenStyle.DashLine if dashed else Qt.PenStyle.SolidLine)
            pen.setCosmetic(True)
            self._items.append(self.scene().addRect(QRectF(x1, y1, x2 - x1, y2 - y1), pen))
            if label:
                t = self.scene().addSimpleText(label, font)
                t.setBrush(QColor(col))
                t.setPos(x1, y1 - 20)
                self._items.append(t)


class ReplayTab(QWidget):
    SPEEDS = [0.25, 0.5, 1.0, 2.0, 4.0]
    # replay_data.json larger than this is parsed in a worker thread (a full game: ~50 MB, 1-4 s) while the tab shows
    # "Loading the 2D replay ..."; smaller files load at once, as before
    ASYNC_LOAD_BYTES = 4_000_000
    _bg_loaded = Signal(object)  # (project, mtime, data or None, error or None) from the parsing thread

    def __init__(self, win: "MainWindow"):
        super().__init__()
        self.win = win
        self.p: Project | None = None
        self.data: dict | None = None
        self.data_mtime = 0.0
        lay = QVBoxLayout(self)
        self.msg = QLabel()
        self.msg.setWordWrap(True)
        lay.addWidget(self.msg)

        self.split = QSplitter(Qt.Orientation.Horizontal)
        self.video = VideoOverlayView()
        self.video.setMinimumWidth(320)
        self.court = ReplayCourt()
        self.split.addWidget(self.video)
        self.split.addWidget(self.court)
        self.split.setSizes([600, 600])
        # video/court on top, transport + timeline below; drag the handle to resize the timeline
        self.vsplit = QSplitter(Qt.Orientation.Vertical)
        self.vsplit.setChildrenCollapsible(False)
        self.vsplit.addWidget(self.split)
        bottom = QWidget()
        bottom_lay = QVBoxLayout(bottom)
        bottom_lay.setContentsMargins(0, 0, 0, 0)
        self.timeline = TimelineWidget(self.display_colors)
        self.timeline.seekRequested.connect(self.seek_to)
        self.court.qualityReady.connect(self.timeline.canvas.update)  # quality lane computed in the background

        self.player = QMediaPlayer(self)
        self.audio = QAudioOutput(self)
        self.player.setAudioOutput(self.audio)
        self.player.setVideoOutput(self.video.item)
        self.player.durationChanged.connect(lambda d: self.slider.setRange(0, d))
        self.player.playbackStateChanged.connect(self._state_changed)

        # ---- transport
        bar = QHBoxLayout()
        self.play_btn = QPushButton("Play")
        self.play_btn.setFixedWidth(70)
        self.play_btn.clicked.connect(self.toggle)
        bar.addWidget(self.play_btn)
        for label, step in [("−5s", -5000), ("+5s", 5000)]:
            b = QPushButton(label)
            b.setFixedWidth(48)
            b.clicked.connect(lambda _=False, s=step: self.player.setPosition(max(0, self.player.position() + s)))
            bar.addWidget(b)
        # volume: speaker / mute button + a compact slider (0-100), remembered between sessions
        self.mute_btn = QPushButton()
        self.mute_btn.setCheckable(True)
        self.mute_btn.setFixedWidth(34)
        self.volume = QSlider(Qt.Orientation.Horizontal)
        self.volume.setRange(0, 100)
        self.volume.setFixedWidth(80)
        self._volume_settings = QSettings(APP_NAME, APP_NAME)
        try:
            vol = int(self._volume_settings.value("volume", 100))
            muted = str(self._volume_settings.value("muted", "false")).lower() in ("true", "1")
        except (TypeError, ValueError):
            vol, muted = 100, False
        self.volume.setValue(max(0, min(100, vol)))
        self.mute_btn.setChecked(muted)
        self.volume.valueChanged.connect(self._volume_changed)
        self.mute_btn.toggled.connect(self._volume_changed)
        self.player.tracksChanged.connect(self._update_audio_available)
        self.player.mediaStatusChanged.connect(lambda _s: self._update_audio_available())
        bar.addWidget(self.mute_btn)
        bar.addWidget(self.volume)
        self._volume_changed()
        self._update_audio_available()
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.sliderMoved.connect(self.player.setPosition)
        self.slider.actionTriggered.connect(lambda _a: self.player.setPosition(self.slider.sliderPosition()))
        bar.addWidget(self.slider, 1)
        self.time_lbl = QLabel()
        self.time_lbl.setMinimumWidth(120)
        bar.addWidget(self.time_lbl)
        self.speed = QComboBox()
        self.speed.addItems([f"{s}×" for s in self.SPEEDS])
        self.speed.setCurrentIndex(2)
        self.speed.currentIndexChanged.connect(lambda i: self.player.setPlaybackRate(self.SPEEDS[i]))
        bar.addWidget(QLabel("Speed"))
        bar.addWidget(self.speed)
        bottom_lay.addLayout(bar)
        bottom_lay.addWidget(self.timeline, 1)
        self.vsplit.addWidget(bottom)
        self.vsplit.setStretchFactor(0, 3)
        self.vsplit.setStretchFactor(1, 1)
        self.vsplit.setSizes([520, 290])
        lay.addWidget(self.vsplit, 1)

        # ---- display options
        from replay.gui.flowlayout import FlowLayout
        opt_w = QWidget()  # the options wrap onto a second line in a narrow window (laptops)
        opt = FlowLayout(opt_w)
        sp = opt_w.sizePolicy()
        sp.setHeightForWidth(True)
        opt_w.setSizePolicy(sp)
        self.trails = QCheckBox("Tracers")
        self.trails.setToolTip("Draw each player's recent path")
        self.trail_len = QDoubleSpinBox()
        self.trail_len.setRange(0.5, 15)
        self.trail_len.setValue(3)
        self.trail_len.setSuffix(" s")
        self.smooth = QSlider(Qt.Orientation.Horizontal)
        self.smooth.setRange(0, 20)  # tenths of a second
        self.smooth.setFixedWidth(110)
        self.smooth.setToolTip("Display smoothing: averages each position over this time window (display only; "
                               "exported data is unaffected)")
        self.smooth_lbl = QLabel("Smoothing: off")
        self.smooth_lbl.setMinimumWidth(95)
        self.smooth_help = HelpIcon(HELP["smoothing"], "Smoothing")
        self.ids = QCheckBox("IDs")
        self.ids.setChecked(True)
        self.interp = QCheckBox("Interpolated")
        self.interp.setChecked(True)
        self.interp.setToolTip("Show positions filled in across short gaps (≤ 3 s, flagged in the data)")
        self.hollow = QCheckBox("Hollow when interpolated")
        self.hollow.setToolTip("Draw filled-in positions as hollow circles")
        self.hold = QDoubleSpinBox()
        self.hold.setRange(0, 15)
        self.hold.setValue(4)
        self.hold.setSuffix(" s")
        self.hold.setToolTip("Keep showing players who are briefly out of view (e.g. the camera panned away) at "
                             "their last known position, faded, for this long. Display only.")
        self.hold_lbl = QLabel("Hold unseen")
        self.hold_help = HelpIcon(HELP["hold"], "Hold unseen")
        self.interp_help = HelpIcon(HELP["interp"], "Interpolated")
        self.confidence_help = HelpIcon(HELP["confidence"], "Confidence and Legend")
        self.unknown_help = HelpIcon(HELP["unknown"], "Unclassified people and Boxes on video")
        self.unknown = QCheckBox("Unclassified people")
        self.unknown.setToolTip("People that could not be assigned to a team or as referee (often coaches or "
                                "spectators near the court)")
        self.refs = QCheckBox("Show referees")
        self.refs.setChecked(True)
        self.refs.setToolTip("Draw the referees (R) in the 2D view")
        self.headings = QCheckBox("Headings")
        self.headings.setChecked(True)
        self.headings.setToolTip("Small arrow on each moving player's dot showing where they are running")
        self.ball = QCheckBox("Ball")
        self.ball.setChecked(True)
        self.boxes = QCheckBox("Boxes on video")
        self.boxes.setToolTip("Draw the tracking boxes, IDs and ball on the video")
        self.show_video = QCheckBox("Video")
        self.show_video.setChecked(True)
        self.show_video.toggled.connect(self.video.setVisible)
        self.confidence = QCheckBox("Confidence")
        self.confidence.setChecked(False)  # default: clean view (see widgets.paint_court)
        self.confidence.setToolTip("Off: clean view - seen players solid, short gaps as small faint dots, nothing "
                                   "predicted far from a sighting or off the court. "
                                   "On: show how sure the tracking is: solid dot = seen; pale dashed dot with a halo = "
                                   "position predicted while the player is hidden (bigger halo = less sure where); "
                                   "'?' = team or identity uncertain. Jersey numbers replace track ids where known.")
        self.conf_legend = QCheckBox("Legend")
        self.conf_legend.setChecked(True)
        self.conf_legend.setToolTip("Show the key to the dot styles in the 2D view")
        self.names = QCheckBox("Names")
        self.names.setChecked(True)
        self.names.setToolTip("Player names (from the Roster) under the dots in the 2D view")
        self.names.setVisible(False)  # only offered when the Replay has a roster
        def unit(*ws: QWidget) -> QWidget:  # widgets that belong together stay on one line when the options wrap
            u = QWidget()
            h = QHBoxLayout(u)
            h.setContentsMargins(0, 0, 0, 0)
            h.setSpacing(4)
            for x in ws:
                h.addWidget(x)
            return u

        for w in (self.trails, self.trail_len, unit(self.smooth_lbl, self.smooth, self.smooth_help), self.ids,
                  unit(self.interp, self.interp_help), self.hollow,
                  unit(self.hold_lbl, self.hold, self.hold_help), self.refs, self.headings,
                  unit(self.unknown, self.unknown_help), self.ball, self.boxes,
                  unit(self.confidence, self.confidence_help), self.conf_legend, self.names, self.show_video):
            opt.addWidget(w)
        lay.addWidget(opt_w)
        for cb in (self.trails, self.ids, self.interp, self.hollow, self.refs, self.headings, self.unknown, self.ball,
                   self.boxes, self.confidence, self.conf_legend, self.names):
            cb.toggled.connect(self._apply_options)
        self.trail_len.valueChanged.connect(self._apply_options)
        self.hold.valueChanged.connect(self._apply_options)
        self.smooth.valueChanged.connect(self._apply_options)

        # ---- teams
        teams = QHBoxLayout()
        self.lbl_a, self.lbl_b = QLabel("Home"), QLabel("Away")
        self.col_a, self.col_b, self.col_ref = ColorButton(), ColorButton(), ColorButton("#334155")
        for b in (self.col_a, self.col_b, self.col_ref):
            b.changed.connect(self._colors_changed)
        self.swap = QCheckBox("Swap teams")
        self.swap.setToolTip("Exchange which players are home and away (rebuilds the replay in a few seconds)")
        self.swap.toggled.connect(self._swap_toggled)
        for w in (self.lbl_a, self.col_a, self.lbl_b, self.col_b, QLabel("Referees"), self.col_ref, self.swap):
            teams.addWidget(w)
        self.legend = HelpIcon("", "Key to the replay")  # the key (was a grey paragraph) is filled in by reload_data
        teams.addWidget(self.legend)
        teams.addStretch(1)
        lay.addLayout(teams)

        # the 2D view is redrawn at the display rate with a clock extrapolated between the media player's
        # (coarse, ~15 Hz) position updates; the rest of the UI follows at a lower rate
        self.clock = PlaybackClock()
        self._ui_state: tuple | None = None
        self._ui_frame = 0
        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.timer.setInterval(16)
        self.timer.timeout.connect(self.tick)
        self.timer.start()
        sc = QShortcut(QKeySequence(Qt.Key.Key_Space), self)
        sc.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        sc.activated.connect(self.toggle)
        for key, fn in ((Qt.Key.Key_M, self.timeline.add_mark_at_playhead),
                        (Qt.Key.Key_BracketLeft, lambda: self.timeline.jump(-1)),
                        (Qt.Key.Key_BracketRight, lambda: self.timeline.jump(1))):
            sc = QShortcut(QKeySequence(key), self)
            sc.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            sc.activated.connect(fn)
        self._coach_keys()  # coach shortcuts + ball trail toggle (replay view QoL block)

    def _coach_keys(self) -> None:
        """Arrows -+5 s, Shift+arrows -+1 s, , / . one video frame (pauses), 1-5 speed, N / Shift+N next / previous
        shot or turnover; adds the "Ball trail" option next to "Ball"."""
        from replay.gui import motion as _mo

        def jump(ms: int) -> None:
            self.player.setPosition(max(0, self.player.position() + ms))

        def frame(d: int) -> None:
            self.player.pause()
            fps = float(((self.p.info().get("video_info") or {}).get("fps") if self.p else 0) or 30.0)
            jump(int(round(d * 1000.0 / fps)))

        def event(d: int) -> None:
            marks = [m.t for _, m in self.timeline.tl.all_marks() if _mo.is_key_event(m.event_id, m.title)]
            t = _mo.next_event_time(marks, self.player.position() / 1000.0, d)
            if t is None:
                self.timeline.flash("No more shots / turnovers in that direction.")
            else:
                self.seek_to(max(0.0, t - 1.0))  # a second of run-up

        K, S = Qt.Key, Qt.KeyboardModifier.ShiftModifier
        keys = [(K.Key_Left, lambda: jump(-5000)), (K.Key_Right, lambda: jump(5000)),
                (QKeySequence(S | K.Key_Left), lambda: jump(-1000)), (QKeySequence(S | K.Key_Right), lambda: jump(1000)),
                (K.Key_Comma, lambda: frame(-1)), (K.Key_Period, lambda: frame(1)),
                (K.Key_N, lambda: event(1)), (QKeySequence(S | K.Key_N), lambda: event(-1))]
        keys += [(getattr(K, f"Key_{i + 1}"), (lambda i=i: self.speed.setCurrentIndex(i))) for i in range(len(self.SPEEDS))]
        for key, fn in keys:
            sc = QShortcut(key if isinstance(key, QKeySequence) else QKeySequence(key), self)
            sc.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            sc.activated.connect(fn)
        self.ball_trail = QCheckBox("Ball trail")
        self.ball_trail.setToolTip("Fading trail behind the ball (last 2 s of playback)")
        self.ball_trail.toggled.connect(lambda on: (setattr(self.court, "ball_trail", on), self.court.update()))
        par = self.ball.parentWidget()
        lay = par.layout() if par is not None else None
        if lay is not None and lay.indexOf(self.ball) >= 0 and hasattr(lay, "insertWidget"):
            lay.insertWidget(lay.indexOf(self.ball) + 1, self.ball_trail)
        self.play_btn.setToolTip("Space play/pause · arrows -+5 s · Shift+arrows -+1 s · , . one frame · 1-5 speed · "
                                 "N / Shift+N next / previous shot or turnover · click a dot to follow a player")

    # -------------------------------------------------------------- data
    def load(self, p: Project) -> None:
        self.player.stop()
        self.p = p
        self.data, self.data_mtime, self._loading_mt = None, 0.0, None
        size = p.info().get("video_info", {}).get("size") or [1280, 720]
        self.video.set_video_size(int(size[0]), int(size[1]))
        self.player.setSource(QUrl.fromLocalFile(str(p.video.resolve())) if p.video.exists() else QUrl())
        st = p.settings()
        for cb, key in ((self.swap, "swap_teams"),):
            cb.blockSignals(True)
            cb.setChecked(bool(st.get(key, False)))
            cb.blockSignals(False)
        self.timeline.set_project(p.root)
        self.reload_data()

    def display_colors(self) -> dict[str, str]:
        """Colours to draw: user choice > team colours set in Overview > colours detected from the jerseys."""
        p = self.p
        out = {"referee": "#334155", "unknown": "#9ca3af"}
        if self.data:
            out.update({k: v["primary"] for k, v in self.data.get("game", {}).get("teams", {}).items()})
        if p is not None:
            m = p.meta()
            if m.get("team_colors_auto") is False:
                jc = m.get("jersey_colors", {})
                out.update({"A": jc.get("home", {}).get("primary", out.get("A")),
                            "B": jc.get("away", {}).get("primary", out.get("B"))})
            user = p.settings().get("display_colors") or {}
            if not (user.get("A") and user.get("B")):  # near-identical kits: draw them clearly apart
                from replay.team_binding import distinct_colors
                try:
                    pair = distinct_colors({k: out.get(k) for k in ("A", "B")})
                    out.update({k: pair[k] for k in ("A", "B") if pair.get(k) and not user.get(k)})
                except Exception:  # noqa: BLE001 - odd colour strings: keep the detected colours
                    pass
            out.update(user)
        return out

    def reload_data(self) -> None:
        p = self.p
        if p is None:
            return
        self.timeline.set_inplay_file(p.results / "inplay.json")
        f = p.results / "replay_data.json"
        if not f.exists():
            self.data = None
            self.court.set_data(None)
            self.timeline.set_data(None)
            s = p.status()
            self.msg.setText("<b>No 2D replay yet.</b> " + (
                "Analysis is still running." if not s["analyzed"] else
                "The court is being calibrated automatically; the 2D replay is built right after."
                if not s["calibrated"] else "Building the 2D replay..."))
            self.msg.show()
            return
        try:
            st = f.stat()
        except OSError as e:
            self._apply_loaded((p, None, None, e))
            return
        mt = st.st_mtime
        if mt == self.data_mtime or mt == getattr(self, "_loading_mt", None):
            return
        if st.st_size > self.ASYNC_LOAD_BYTES:  # big (full game): parse off the GUI thread, no freeze
            if not getattr(self, "_bg_connected", False):
                self._bg_loaded.connect(self._apply_loaded)
                self._bg_connected = True
            self._loading_mt = mt
            if self.data is None:
                self.msg.setText("<b>Loading the 2D replay ...</b>")
                self.msg.show()
            threading.Thread(target=lambda: self._bg_load_worker(p, mt, f),
                             daemon=True, name="replay-load").start()
            return
        self._apply_loaded((p, mt, *self._parse_replay(p, f)))

    def _bg_load_worker(self, p: Project, mt, f: Path) -> None:
        """Thread body: parse, and ALWAYS emit a result (an error on any failure) so the tab never hangs on 'Loading'."""
        try:
            res = self._parse_replay(p, f)
        except Exception as e:  # MemoryError/RecursionError/odd files: report instead of dying silently
            res = (None, e)
        try:
            self._bg_loaded.emit((p, mt, *res))
        except RuntimeError:  # window/tab destroyed while parsing
            pass

    @staticmethod
    def _parse_replay(p: Project, f: Path):
        """(data, None) or (None, error): replay_data.json + lineup slots + roster names (no Qt: runs in a thread)."""
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or not isinstance(data.get("t"), list) or \
                    not isinstance(data.get("tracks"), list):
                raise ValueError("unexpected content")
        except (OSError, ValueError) as e:  # being rewritten by a build right now (retried when it changes) or damaged
            return None, e
        if "lineup" not in data:  # lineup slots (replay/lineup.py) for stable numbers in the 2D view
            try:
                lf = p.results / "lineup.json"
                if lf.exists():
                    from replay.gui.motion import lineup_from_file
                    lu = lineup_from_file(json.loads(lf.read_text(encoding="utf-8")), data)
                    if lu:
                        data["lineup"] = lu
            except (OSError, ValueError, TypeError, KeyError) as e:
                print("replay: lineup.json ignored:", e)
        from replay.roster_names import annotate_replay_data  # roster hook: names for the 2D dot tooltip
        try:
            annotate_replay_data(p.root, data)  # roster hook (no-op without roster.json)
        except Exception as e:  # names are optional: a bad roster.json must not block the replay
            print("replay: roster names ignored:", e)
        return data, None

    def _apply_loaded(self, item) -> None:
        p, mt, data, err = item
        if p is not self.p:  # another project was opened meanwhile
            return
        if mt is not None:
            if getattr(self, "_loading_mt", None) == mt:
                self._loading_mt = None
            self.data_mtime = mt
        if err is not None:
            if self.data is None:
                self.court.set_data(None)
                self.timeline.set_data(None)
                self.msg.setText(f"<b>The 2D replay data could not be read</b> ({err}). Rebuild the replay "
                                 "(Overview → Build replay) if this persists.")
                self.msg.show()
            return
        self.data = data
        self.msg.hide()
        self._refresh_team_widgets()
        self._apply_options()
        self.timeline.set_data(self.data)

    def _refresh_team_widgets(self) -> None:
        cols = self.display_colors()
        teams = (self.data or {}).get("game", {}).get("teams", {})
        self.lbl_a.setText(teams.get("A", {}).get("name", "Home"))
        self.lbl_b.setText(teams.get("B", {}).get("name", "Away"))
        for b, k in ((self.col_a, "A"), (self.col_b, "B"), (self.col_ref, "referee")):
            b.blockSignals(True)
            b.set_color(cols.get(k, "#ffffff"))
            b.blockSignals(False)
        self.court.set_data(self.data, cols)
        self.legend.set_help(HELP["replay_key"])

    def _colors_changed(self) -> None:
        if self.p is None:
            return
        st = self.p.settings()
        st["display_colors"] = {"A": self.col_a.color, "B": self.col_b.color, "referee": self.col_ref.color}
        self.p.update_info(settings=st)
        self.court.set_colors(st["display_colors"])
        self.timeline.refresh_auto()

    def _swap_toggled(self, on: bool) -> None:
        if self.p is None:
            return
        st = self.p.settings()
        st["swap_teams"] = on
        # colours chosen for the teams stay with the team names
        self.p.update_info(settings=st)
        if self.p.status()["analyzed"] and self.p.status()["calibrated"]:
            self.win.jobs.submit(self.p.id, BUILD)
            self.win.statusBar().showMessage("Rebuilding the replay with the teams swapped…", 4000)

    def _apply_options(self) -> None:
        c = self.court
        c.trails, c.trail_s = self.trails.isChecked(), self.trail_len.value()
        c.smooth_s = self.smooth.value() / 10.0
        self.smooth_lbl.setText("Smoothing: off" if not self.smooth.value() else f"Smoothing: {c.smooth_s:.1f} s")
        c.labels, c.show_interp = self.ids.isChecked(), self.interp.isChecked()
        c.hollow_interp, c.hold_s = self.hollow.isChecked(), self.hold.value()
        c.show_unknown, c.show_ball = self.unknown.isChecked(), self.ball.isChecked()
        c.show_referees, c.headings = self.refs.isChecked(), self.headings.isChecked()
        c.confidence = self.confidence.isChecked()
        c.show_legend = c.confidence and self.conf_legend.isChecked()
        has_names = bool((self.data or {}).get("roster_names"))
        self.names.setVisible(has_names)
        c.show_names = has_names and self.names.isChecked()
        self.conf_legend.setEnabled(c.confidence)
        c.update()
        if not self.boxes.isChecked():
            self.video.set_boxes([])

    # -------------------------------------------------------------- playback
    def toggle(self) -> None:
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def _volume_changed(self, *_a) -> None:
        v, muted = self.volume.value(), self.mute_btn.isChecked()
        self.audio.setVolume(v / 100.0)
        self.audio.setMuted(muted)
        self.mute_btn.setText("🔇" if muted or v == 0 else "🔈" if v < 50 else "🔊")
        self._volume_settings.setValue("volume", v)
        self._volume_settings.setValue("muted", muted)

    def _update_audio_available(self) -> None:
        """No sound in this video (or no video): the controls are greyed out instead of silently doing nothing."""
        has = bool(self.player.hasAudio()) if not self.player.source().isEmpty() else True
        for w in (self.volume, self.mute_btn):
            w.setEnabled(has)
        tip = ("Volume. Click the speaker to mute / unmute." if has
               else "This video has no sound, so there is no volume to set.")
        self.volume.setToolTip(tip)
        self.mute_btn.setToolTip(tip if has else tip)

    def _state_changed(self, st) -> None:
        self.play_btn.setText("Pause" if st == QMediaPlayer.PlaybackState.PlayingState else "Play")

    def seek_to(self, t: float) -> None:
        self.player.setPosition(int(max(0.0, t) * 1000))

    def tick(self) -> None:
        if not self.isVisible():
            return
        pos = self.player.position()
        playing = self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState
        t = self.clock.update(pos / 1000.0, playing, self.player.playbackRate() or 1.0, time.perf_counter())
        self.court.set_time(t)  # repaints only when t changed
        self._ui_frame += 1
        if playing and self._ui_frame % 2 == 0:
            self.timeline.set_time(t, True)  # smooth playhead at ~30 Hz
        state = (pos, playing, self.boxes.isChecked(), self.player.duration())
        if state == self._ui_state and self._ui_frame % 15:
            return  # nothing else changed: keep the GUI thread free for drawing
        self._ui_state = state
        if not self.slider.isSliderDown():
            self.slider.setValue(pos)
        if self.boxes.isChecked() and self.data:
            self.video.set_boxes(self._boxes_at(t))
        state = self.court.calib_state()
        badge = {2: "calibration: keyframe", 1: "calibration: propagated", 0: "no calibration",
                 -1: "outside analysed range"}.get(state, "")
        self.time_lbl.setText(f"{fmt_time(t)} / {fmt_time(self.player.duration() / 1000)}")
        self.win.statusBar().showMessage(f"Replay {fmt_time(t)} · {badge}")
        self.timeline.set_duration(self.player.duration() / 1000)
        self.timeline.set_time(t, playing)

    def _boxes_at(self, t: float) -> list:
        d = self.data
        i, _ = frame_index(d, t)
        if not d or not (d["t"][0] - 0.5 <= t <= d["t"][-1] + 0.5):
            return []
        cols = self.court.colors
        out = []
        for tr in d["tracks"]:
            if tr["team"] == "unknown" and not self.unknown.isChecked():
                continue
            j = i - tr["i0"]
            if 0 <= j < len(tr.get("b", [])) and tr["b"][j]:
                out.append((tr["b"][j], cols.get(tr["team"], "#ffffff"),
                            str(tr["id"]) if self.ids.isChecked() else "", False))
        ball = d.get("ball")
        if ball and self.ball.isChecked() and i < len(ball["b"]) and ball["b"][i]:
            out.append((ball["b"][i], "#ffe600", "ball", ball["f"][i] != 1))
        return out

    def stop(self) -> None:
        self.player.pause()


# ====================================================================== dialogs
class TeamPickerDialog(QDialog):
    """Click a few players of each team (and a referee) so teams are recognised from the real video colours."""

    LABELS = [("A", "Home"), ("B", "Away"), ("referee", "Referee")]
    COLORS = {"A": "#a855f7", "B": "#f97316", "referee": "#facc15"}

    def __init__(self, parent: QWidget, p: Project, start_t: float = 0.0):
        super().__init__(parent)
        self.p = p
        meta = p.meta()
        self.setWindowTitle("Identify the teams")
        self.resize(1200, 820)
        self.data = load_pass1(p.results / "work")
        self.persons = self.data.rows[self.data.rows[:, 6] == 0]
        self.grabber = FrameGrabber(p.video)
        self.samples: dict[str, list] = {k: list(v) for k, v in (meta.get("team_samples") or {}).items()}
        self.picked: list[tuple[str, int, list[float]]] = []  # (label, frame k, box) for drawing
        self.i = 0
        from replay import team_binding as tbind
        ov = tbind.team_overview(meta)  # picked players go to these (colour-bound) teams
        names = {"A": f"{ov['A']['name']} (home)", "B": f"{ov['B']['name']} (away)", "referee": "Referee"}

        v = QVBoxLayout(self)
        v.addWidget(help_row(QLabel("Choose a team below, then click a player of that team on the frame."),
                             HELP["picker"], "How to pick players"))
        row = QHBoxLayout()
        self.radio: dict[str, QPushButton] = {}
        for key, _ in self.LABELS:
            b = QPushButton(names[key])
            b.setCheckable(True)
            b.setAutoExclusive(True)
            b.setStyleSheet(f"QPushButton:checked {{ border: 2px solid {self.COLORS[key]}; font-weight: bold; }}")
            self.radio[key] = b
            row.addWidget(b)
        self.radio["A"].setChecked(True)
        row.addStretch(1)
        self.count_lbl = QLabel()
        row.addWidget(self.count_lbl)
        v.addLayout(row)
        self.view = FrameView()
        self.view.clicked.connect(self.click)
        v.addWidget(self.view, 1)
        nav = QHBoxLayout()
        for label, step in [("−10s", -10), ("−1s", -1), ("+1s", 1), ("+10s", 10)]:
            b = QPushButton(label)
            b.clicked.connect(lambda _=False, s=step: self.goto(self.data.t[self.i] + s))
            nav.addWidget(b)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, max(0, len(self.data.t) - 1))
        self.slider.valueChanged.connect(lambda i: self.show_frame(i))
        nav.addWidget(self.slider, 1)
        undo = QPushButton("Remove last")
        undo.clicked.connect(self.undo)
        clear = QPushButton("Clear all")
        clear.clicked.connect(self.clear)
        nav.addWidget(undo)
        nav.addWidget(clear)
        v.addLayout(nav)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        v.addWidget(bb)
        self.goto(start_t)

    def goto(self, t: float) -> None:
        self.slider.setValue(int(np.argmin(np.abs(self.data.t - t))))
        self.show_frame(self.slider.value())

    def frame_boxes(self) -> np.ndarray:
        return self.persons[self.persons[:, 0] == self.data.k[self.i]]

    def show_frame(self, i: int) -> None:
        self.i = int(i)
        res = self.grabber.get(float(self.data.t[self.i]))
        if res is None:
            return
        self.view.set_frame(res[1])
        for it in getattr(self, "_items", []):
            self.view.scene().removeItem(it)
        self._items = []
        k = self.data.k[self.i]
        picked_here = {tuple(np.round(b, 1)): lab for lab, kk, b in self.picked if kk == k}
        for r in self.frame_boxes():
            box = r[1:5]
            lab = picked_here.get(tuple(np.round(box, 1)))
            pen = QPen(QColor(self.COLORS[lab]) if lab else QColor(255, 255, 255, 160), 3 if lab else 1.5)
            pen.setCosmetic(True)
            self._items.append(self.view.scene().addRect(QRectF(box[0], box[1], box[2] - box[0], box[3] - box[1]),
                                                         pen))
        self._update_count()

    def click(self, x: float, y: float) -> None:
        rows = self.frame_boxes()
        inside = rows[(rows[:, 1] <= x) & (x <= rows[:, 3]) & (rows[:, 2] <= y) & (y <= rows[:, 4])]
        if not len(inside):
            return
        # smallest box containing the click = the person in front
        r = inside[np.argmin((inside[:, 3] - inside[:, 1]) * (inside[:, 4] - inside[:, 2]))]
        if not np.isfinite(r[8:11]).all():
            return
        lab = next(k for k, b in self.radio.items() if b.isChecked())
        self.samples.setdefault(lab, []).append([float(v) for v in r[8:14]])
        self.picked.append((lab, int(self.data.k[self.i]), [float(v) for v in r[1:5]]))
        self.show_frame(self.i)

    def undo(self) -> None:
        if self.picked:
            lab, _, _ = self.picked.pop()
            if self.samples.get(lab):
                self.samples[lab].pop()
            self.show_frame(self.i)

    def clear(self) -> None:
        self.samples, self.picked = {}, []
        self.show_frame(self.i)

    def _update_count(self) -> None:
        names = {k: b.text() for k, b in self.radio.items()}
        self.count_lbl.setText("Picked: " + " · ".join(f"{names[k]} {len(self.samples.get(k, []))}"
                                                        for k, _ in self.LABELS))

    def done(self, r: int) -> None:
        self.grabber.close()
        super().done(r)


class PhasePanel(QWidget):
    """Every phase of a Replay build, always visible: a bar each, done / running / waiting, % + elapsed + ETA,
    notes (fallbacks, waits, errors) and the overall time left (data: replay/gui/phases.py)."""

    STATE_COLORS = {"done": "#15803d", "running": "#1d4ed8", "waiting": "#6b7280", "skipped": "#6b7280",
                    "stopped": "#b45309", "failed": "#b91c1c"}

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        from PySide6.QtWidgets import QGridLayout
        from replay.gui.phases import PHASES
        g = QGridLayout(self)
        g.setContentsMargins(0, 2, 0, 2)
        g.setHorizontalSpacing(8)
        g.setVerticalSpacing(1)
        self.overall = QLabel()
        g.addWidget(self.overall, 0, 0, 1, 5)
        from PySide6.QtWidgets import QToolButton
        self.close_btn = QToolButton()
        self.close_btn.setText("×")
        self.close_btn.setToolTip("Hide the build phases")
        self.close_btn.setAutoRaise(True)
        self.close_btn.clicked.connect(self.dismiss)
        g.addWidget(self.close_btn, 0, 5, Qt.AlignmentFlag.AlignRight)
        self._last_snap = None  # (snap, busy, eta text) of the Replay shown, for the 1 s clock
        self._tick = QTimer(self)
        self._tick.setInterval(1000)
        self._tick.timeout.connect(self.tick)
        self._tick.start()
        self._st: dict[str, dict] = {}  # per Replay: busy / finished / sticky / opened by hand
        self._pid: str | None = None
        self.rows: dict[str, tuple] = {}
        half = (len(PHASES) + 1) // 2
        for i, (k, title) in enumerate(PHASES):
            r, c = 1 + i % half, 3 * (i // half)
            name = QLabel(f"{i + 1}  {title}")
            name.setMinimumWidth(150)
            bar = QProgressBar()
            bar.setRange(0, 1000)
            bar.setTextVisible(False)
            bar.setFixedHeight(10)
            bar.setMinimumWidth(50)
            info = QLabel()
            info.setMinimumWidth(120)
            for w in (name, info):
                w.setStyleSheet("font-size: 11px;")
            g.addWidget(name, r, c)
            g.addWidget(bar, r, c + 1)
            g.addWidget(info, r, c + 2)
            self.rows[k] = (name, bar, info)
        g.setColumnStretch(1, 1)
        g.setColumnStretch(4, 1)

    def update_from(self, snap: dict) -> None:
        from replay.gui.phases import fmt_s
        for row in snap.get("rows") or []:
            name, bar, info = self.rows[row["key"]]
            st = row["state"]
            bar.setValue(int(row["progress"] * 1000))
            if st == "running":
                txt = f"{row['progress'] * 100:.0f}%"
                if row.get("elapsed") is not None:
                    txt += f" · {fmt_s(row['elapsed'])}"
                if row.get("eta") is not None:
                    txt += f" · ~{fmt_s(row['eta'])} left"
                elif row["key"] == "calib1":
                    txt += " · alongside the detection"
                if row.get("stalled"):
                    txt += f" · no progress for {fmt_s(row['stalled'])} (still working)"
            elif st == "done":
                txt = "done" + (f" in {fmt_s(row['elapsed'])}" if row.get("elapsed") is not None else "")
            else:
                txt = st
            notes = row.get("notes") or []
            col = self.STATE_COLORS.get(st, "#6b7280")
            last = notes[-1] if notes else None
            if last is not None:
                ncol = {"warn": "#b45309", "error": "#b91c1c"}.get(last["level"], "#6b7280")
                txt = f"<span style='color:{col}'>{txt}</span> <span style='color:{ncol}'>· {last['text'][:60]}</span>"
            else:
                txt = f"<span style='color:{col}'>{txt}</span>"
            info.setText(txt)
            info.setToolTip("\n".join(n["text"] for n in notes))
            name.setStyleSheet("font-size: 11px;" + (" font-weight: bold;" if st == "running" else
                                                     " color: #6b7280;" if st in ("waiting", "skipped") else ""))
        eta = snap.get("overall_eta")
        left = f"About {fmt_s(eta)} left for the whole build" if eta is not None and snap.get("running") else ""
        self.overall.setText(self.elapsed_text(snap, left))

    @staticmethod
    def elapsed_text(snap: dict, left: str = "", now: float | None = None) -> str:
        """'Elapsed 4:12 · About 6:30 left ...' from the recorded start of the run (never from when the panel was
        opened). Only while a job runs; no ETA text unless the app has one."""
        from replay.gui.phases import fmt_clock
        run = snap.get("run")
        if not run or not run.get("running") or not run.get("started"):
            return f"<b>{left}</b>" if left else ""
        el = (time.time() if now is None else now) - float(run["started"])
        return f"<b>Elapsed {fmt_clock(el)}</b>" + (f" <span style='color:gray'>· {left}</span>" if left else "")

    def tick(self) -> None:
        """Once a second: move the elapsed clock (the rest of the panel follows the window's poll)."""
        last = getattr(self, "_last_snap", None)
        if last is not None and last[1] and self.isVisible():
            self.overall.setText(self.elapsed_text(last[0], last[2]))

    HOLD_S = 5.0  # shown this long after a build finished ("Done in X")

    @staticmethod
    def problems(snap: dict) -> bool:
        """A phase ended in an error / stop, or carries a warning (fallback, long wait): keep the panel up."""
        return any(r["state"] in ("failed", "stopped") or any(n.get("level") in ("warn", "error")
                                                               for n in r.get("notes") or [])
                   for r in snap.get("rows") or [])

    def sync(self, pid: str, snap: dict, busy: bool, now: float | None = None) -> None:
        """Visible while a job of Replay pid is queued / running, for HOLD_S after it ended ("Done in X"), until
        dismissed when a phase had a problem, or when opened by hand (show_details)."""
        from replay.gui.phases import fmt_s
        now = time.time() if now is None else now
        self._pid = pid
        st = self._st.setdefault(pid, {"busy": False, "t0": None, "done": None, "sticky": False, "manual": False})
        if busy:
            if not st["busy"]:
                st.update(t0=now, done=None, sticky=False, manual=False)
            st["busy"] = True
        elif st["busy"]:
            st.update(busy=False, done=now, sticky=self.problems(snap))
        self.update_from(snap)
        eta = snap.get("overall_eta")
        self._last_snap = (snap, busy, f"About {fmt_s(eta)} left for the whole build"
                           if eta is not None and snap.get("running") else "")
        show = busy or st["manual"] or (st["done"] is not None and (now - st["done"] < self.HOLD_S or st["sticky"]))
        if not busy and show:
            if st["manual"] and st["done"] is None:
                total = sum(r["elapsed"] or 0.0 for r in snap.get("rows") or [] if r["key"] != "calib1")
                self.overall.setText(f"<b>Last build: phase times</b> (about {fmt_s(total)} in all)")
            else:
                run = snap.get("run") or {}
                took = (run.get("elapsed") if run.get("ended") else None) or                        ((st["done"] - st["t0"]) if st["t0"] and st["done"] else None)
                bad = self.problems(snap)
                self.overall.setText(f"<b>Done in {fmt_s(took)}</b>" + (
                    " <span style='color:#b45309'>- see the notes below</span>" if bad else ""))
        self.overall.setVisible(bool(self.overall.text()))
        self.close_btn.setVisible(not busy)
        self.setVisible(show)

    def dismiss(self) -> None:
        st = self._st.get(self._pid or "")
        if st is not None:
            st.update(sticky=False, manual=False, done=None)
        self.hide()

    def show_details(self) -> None:
        st = self._st.setdefault(self._pid or "", {"busy": False, "t0": None, "done": None, "sticky": False,
                                                   "manual": False})
        st.update(manual=True, done=None)
        self.show()


class ImportDialog(QDialog):
    def __init__(self, parent: QWidget, path: str):
        super().__init__(parent)
        self.setWindowTitle("Import video")
        f = QFormLayout(self)
        f.addRow("File", QLabel(Path(path).name))
        self.name = QLineEdit(Path(path).stem)
        f.addRow("Replay name", self.name)
        self.copy = QCheckBox("Copy the original file into the Replay")
        self.copy.setToolTip("Off: the Replay keeps a playable copy and remembers where the original is.\n"
                             "On: the original is copied too (uses more disk space).\n"
                             "Either way, an exported Replay (.aareplay) always contains the original.")
        f.addRow(help_row(self.copy, HELP["import_copy"], "Copy the original file"))
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        f.addRow(bb)


def fmt_bytes(n: int) -> str:
    for unit in ("bytes", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "bytes" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


class ExportDialog(QDialog):
    """Shows what goes into the .aareplay package (always everything) and its size."""

    def __init__(self, parent: QWidget, plan: ExportPlan):
        super().__init__(parent)
        self.setWindowTitle("Export Replay")
        self.setMinimumWidth(460)
        v = QVBoxLayout(self)
        intro = QLabel(f"Saves <b>{plan.name}</b> as one <b>{PACKAGE_EXT}</b> file.")
        intro.setWordWrap(True)
        v.addWidget(help_row(intro, HELP["export_how"], "Opening the exported file"))
        rows = "".join(f"<tr><td>{g}</td><td align='right'>&nbsp;&nbsp;{fmt_bytes(b)}</td></tr>"
                       for g, b in sorted(plan.groups().items(), key=lambda kv: -kv[1]))
        self.contents = QLabel(f"<p><b>Included</b></p><table cellpadding=2>{rows}"
                               f"<tr><td><b>Total (estimated file size)</b></td>"
                               f"<td align='right'><b>&nbsp;&nbsp;{fmt_bytes(plan.total_bytes)}</b></td></tr></table>")
        v.addWidget(self.contents)
        for w in plan.warnings:
            lbl = QLabel(f"<span style='color:#b45309'>⚠ {w}</span>")
            lbl.setWordWrap(True)
            v.addWidget(lbl)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        v.addWidget(bb)


# ====================================================================== main window
class _CurrentPageStack(QStackedWidget):
    """QStackedWidget whose minimum size is that of the shown page only (the stock one uses the biggest page)."""

    def minimumSizeHint(self) -> QSize:
        w = self.currentWidget()
        return w.minimumSizeHint() if w is not None else super().minimumSizeHint()


class MainWindow(QMainWindow):
    def __init__(self, projects_dir: Path):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} {__version__}")  # version on top of the window (user, 18.10)
        self.store = ProjectStore(projects_dir)
        self.jobs = JobManager(self.store)
        self.settings = QSettings(APP_NAME, APP_NAME)
        self.current: Project | None = None

        # ---- left: projects
        left = QWidget()
        lv = QVBoxLayout(left)
        b_import = QPushButton("Import video…")
        b_import.setStyleSheet("font-weight:bold; padding:8px;")
        b_import.clicked.connect(self.import_video)
        b_open = QPushButton("Open Replay file…")
        b_open.setToolTip(f"Load a Replay exported as {PACKAGE_EXT} (or an older {LEGACY_PACKAGE_EXT} file)")
        b_open.clicked.connect(self.open_package)
        # start-up self-check banner (replay/selfcheck.py), non-blocking, dismissable
        self.banner = QWidget()
        bl = QHBoxLayout(self.banner)
        bl.setContentsMargins(6, 4, 2, 4)
        self.banner_lbl = QLabel()
        self.banner_lbl.setWordWrap(True)
        self.banner_lbl.setStyleSheet("color:#92400e;")
        bx = QPushButton("×")
        bx.setFixedWidth(22)
        bx.setFlat(True)
        bx.setToolTip("Hide this message")
        bx.clicked.connect(self.banner.hide)
        bl.addWidget(self.banner_lbl, 1)
        bl.addWidget(bx, 0, Qt.AlignmentFlag.AlignTop)
        self.banner.setStyleSheet("background:#fef3c7; border-radius:4px;")
        self.banner.hide()
        lv.addWidget(self.banner)
        lv.addWidget(b_import)
        lv.addWidget(b_open)
        lv.addWidget(QLabel("<b>Replays</b>"))
        self.list = QListWidget()
        self.list.currentItemChanged.connect(self.select_project)
        self.list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.list.customContextMenuRequested.connect(self.project_menu)
        for key, fn in [(Qt.Key.Key_F2, lambda: self.rename_project(self._item_project(self.list.currentItem()))),
                        (Qt.Key.Key_Delete, lambda: self.delete_project(self._item_project(self.list.currentItem())))]:
            sc = QShortcut(QKeySequence(key), self.list)
            sc.setContext(Qt.ShortcutContext.WidgetShortcut)
            sc.activated.connect(fn)
        lv.addWidget(self.list, 1)
        b_del = QPushButton("Delete Replay")
        b_del.clicked.connect(lambda: self.delete_project())
        lv.addWidget(b_del)
        left.setFixedWidth(280)

        # ---- right: project
        self.stack = _CurrentPageStack()
        empty = QLabel("<h2>AdvancedAnalytist</h2><p style='color:gray'>© Onni Kauppinen 2026</p>")
        empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty.setWordWrap(True)
        self.stack.addWidget(empty)

        proj = QWidget()
        pv = QVBoxLayout(proj)
        self.title = QLabel()
        self.title.setStyleSheet("font-size:18px; font-weight:bold;")
        pv.addWidget(self.title)
        job = QHBoxLayout()
        self.job_lbl = QLabel()
        self.job_lbl.setWordWrap(True)
        job.addWidget(self.job_lbl, 1)
        self.job_bar = QProgressBar()
        self.job_bar.setRange(0, 1000)
        self.job_bar.setFixedWidth(260)
        job.addWidget(self.job_bar)
        self.btn_action = QPushButton()
        self.btn_action.clicked.connect(self.job_action)
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.clicked.connect(lambda: self.current and self.jobs.cancel(self.current.id))
        self.last_build = QLabel()  # "Last build: X · details" (opens the saved phase times)
        self.last_build.setTextFormat(Qt.TextFormat.RichText)
        self.last_build.linkActivated.connect(lambda _l: (self.phase_panel.show_details(), self.refresh_project()))
        job.addWidget(self.last_build)
        job.addWidget(self.btn_action)
        job.addWidget(self.btn_stop)
        pv.addLayout(job)
        self.phase_panel = PhasePanel()  # every build phase with its own bar (always visible)
        pv.addWidget(self.phase_panel)
        self.loading = QWidget()
        lv2 = QVBoxLayout(self.loading)
        lv2.addStretch(1)
        self.loading_lbl = QLabel("<h3>Preparing video…</h3>")
        self.loading_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.loading_bar = QProgressBar()
        self.loading_bar.setRange(0, 1000)
        self.loading_bar.setMaximumWidth(420)
        lv2.addWidget(self.loading_lbl)
        lv2.addWidget(self.loading_bar, 0, Qt.AlignmentFlag.AlignHCenter)
        lv2.addStretch(2)
        pv.addWidget(self.loading, 0)
        self.tabs = QTabWidget()
        self.overview_tab = OverviewTab(self)
        self.calib_tab = CalibrateTab(self)
        self.replay_tab = ReplayTab(self)
        self.tabs.addTab(self.overview_tab, "Overview")
        self.tabs.addTab(self.replay_tab, "Replay")
        # hand calibration is a niche fix for camera views the automatic calibration gets wrong
        self.tabs.addTab(self.calib_tab, "Court fix (advanced)")
        from replay.gui.stats_tab import StatsTab  # shot chart, team / lineup stats (stats_tab.py)
        self.stats_tab = StatsTab(self)
        self.tabs.addTab(self.stats_tab, "Stats")
        self.tabs.currentChanged.connect(self.tab_changed)
        pv.addWidget(self.tabs, 1)
        # a stacked widget asks for the size of its biggest page even when another is shown; the project page also
        # scrolls in a low window, so the window fits a 1280x720 screen including the status bar
        proj_scroll = QScrollArea()
        proj_scroll.setWidgetResizable(True)
        proj_scroll.setFrameShape(QFrame.Shape.NoFrame)
        proj_scroll.setWidget(proj)
        proj_scroll.setMinimumSize(300, 200)
        self.stack.addWidget(proj_scroll)

        split = QSplitter()
        split.addWidget(left)
        split.addWidget(self.stack)
        split.setStretchFactor(1, 1)
        self.setCentralWidget(split)
        self._menus()
        from replay.gui import playlists as playlists_gui  # clip playlists dock + "Playlists" menu
        playlists_gui.install(self)
        from replay.gui import live_gui; live_gui.install(self)  # live hook: "Live" menu + live 2D reload  # noqa: E702
        from replay.gui import help_panel, replay_list, tooltips  # QoL hook: list search/sort/icons, help, tips
        replay_list.install(self); help_panel.install(self); tooltips.apply(self)  # noqa: E702

        self.timer = QTimer(self)
        self.timer.setInterval(700)
        self.timer.timeout.connect(self.poll)
        self.timer.start()
        self.refresh_list()
        # jobs cut off by closing the app, a crash or a power cut continue by themselves (not after a user Stop)
        resumed = self.jobs.resume_interrupted()
        if resumed:
            self.statusBar().showMessage(f"Continuing {len(resumed)} unfinished job(s) from the last session", 10000)
        geo = self.settings.value("geometry")
        if geo is not None:
            self.restoreGeometry(geo)
        else:
            self.resize(1500, 900)
        self._fit_to_screen()

    def _fit_to_screen(self) -> None:
        """Never open larger than the screen's usable area (a saved or default size from a bigger monitor)."""
        scr = self.screen() or QApplication.primaryScreen()
        if scr is None:
            return
        av = scr.availableGeometry()
        w, h = min(self.width(), av.width()), min(self.height(), av.height() - 40)
        if (w, h) != (self.width(), self.height()):
            self.resize(w, h)

    def _menus(self) -> None:
        m = self.menuBar().addMenu("&File")
        for label, fn, key in [("Import video…", self.import_video, "Ctrl+I"),
                               ("Open Replay file…", self.open_package, "Ctrl+O")]:
            a = QAction(label, self)
            a.setShortcut(key)
            a.triggered.connect(fn)
            m.addAction(a)
        m.addSeparator()
        # Export lives in its own top-level menu; it acts on the selected Replay
        ex = self.menuBar().addMenu("&Export")
        ex.aboutToShow.connect(lambda: self._fill_export_menu(ex, self.current, shortcut=True))
        self._fill_export_menu(ex, None, shortcut=True)
        aim = self.menuBar().addMenu("&AI")  # AI tactical analysis (replay/gui/ai_dialog.py)
        aim.aboutToShow.connect(lambda: (aim.clear(), ai_dialog.add_menu_actions(aim, self, self.current)))
        ai_dialog.add_menu_actions(aim, self, None)
        save = QAction(self)
        save.setShortcut("Ctrl+S")
        save.triggered.connect(lambda: self.export_project())
        self.addAction(save)
        a = QAction("Show Replays folder", self)
        a.triggered.connect(lambda: os.startfile(str(self.store.root.resolve())) if sys.platform == "win32"
                            else None)
        m.addAction(a)
        a = QAction("Quit", self)
        a.setShortcut("Ctrl+Q")
        a.triggered.connect(self.close)
        m.addAction(a)
        h = self.menuBar().addMenu("&Help")
        a = QAction("About", self)
        a.triggered.connect(lambda: QMessageBox.about(
            self, APP_NAME, "<b>AdvancedAnalytist</b> - basketball 2D replay from game video.<br>"
            f"Each analysed video is a Replay; share one as a single {PACKAGE_EXT} file.<br><br>"
            "Detector: RF-DETR (Apache-2.0). Tracking: ByteTrack via supervision (MIT). "
            "GUI: Qt for Python / PySide6 (LGPL-3.0). Video: FFmpeg (LGPL/GPL)."))
        h.addAction(a)

    # ------------------------------------------------------------------ projects
    def refresh_list(self) -> None:
        cur = self.current.id if self.current else None
        # pinned first; within each group newest first (store order; sort is stable)
        projects = sorted(self.store.list(), key=lambda p: not p.info().get("pinned", False))
        ids = [self.list.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.list.count())]
        if ids != [p.id for p in projects]:
            self.list.blockSignals(True)
            self.list.clear()
            for p in projects:
                it = QListWidgetItem()
                it.setData(Qt.ItemDataRole.UserRole, p.id)
                self.list.addItem(it)
                if p.id == cur:
                    self.list.setCurrentItem(it)
            self.list.blockSignals(False)
        for i, p in enumerate(projects):
            info = p.info()
            vi = info.get("video_info", {})
            item = self.list.item(i)
            item.setIcon(_pin_icon() if info.get("pinned") else _blank_icon())
            item.setToolTip("Pinned" if info.get("pinned") else "")
            item.setText(f"{info.get('name', p.id)}\n   "
                                      f"{status_text(p, self.jobs.state(p.id).public())}"
                                      f"{' · ' + fmt_dur(vi.get('duration_s')) if vi else ''}")

    EXPORT_FILES = [("Tracking data (CSV)…", "tracking.csv"), ("Tracking data (JSON)…", "tracking.json"),
                    ("Processing report (Markdown)…", "report.md"), ("Stand-alone replay (HTML)…", "replay.html")]

    def _fill_export_menu(self, menu: QMenu, p: Project | None, shortcut: bool = False) -> None:
        """Export actions for Replay ``p``; shared by the menu bar and the right-click menu."""
        for sub in menu.findChildren(QMenu):  # submenus from the previous fill
            sub.deleteLater()
        menu.clear()
        # "\t" shows a shortcut hint; the real Ctrl+S is registered once on the window (see _menus)
        a = menu.addAction(f"Export Replay ({PACKAGE_EXT})…" + ("\tCtrl+S" if shortcut else ""))
        a.setToolTip("One file with everything: original video, playback copy, analysis, 2D replay, timeline")
        a.setEnabled(p is not None and p.video.exists())
        a.triggered.connect(lambda: self.export_project(p))
        ready = p is not None and p.status()["replay_ready"]
        a = menu.addAction("Coach report (view / PDF)…")  # replay/gui/coach_report.py
        a.setToolTip("One-page post-game report: box score, shot charts, lineups, runs; click numbers for clips")
        a.setEnabled(ready)
        a.triggered.connect(lambda: __import__("replay.gui.coach_report", fromlist=["x"]).open_report(self, p))
        a = menu.addAction("Simple data export (for AI)…")
        a.setToolTip("A new folder of CSV/JSON files with instructions, ready to give to an AI for analysis")
        a.setEnabled(ready)
        a.triggered.connect(lambda: self.export_data(p))
        files = menu.addMenu("Individual files")
        files.setEnabled(ready)
        for label, name in self.EXPORT_FILES:
            a = files.addAction(label)
            a.triggered.connect(lambda _=False, n=name: self.export_file(n, p))
        if p is None:
            menu.addSeparator()
            menu.addAction("Select a Replay first").setEnabled(False)
        elif not ready:
            menu.addSeparator()
            menu.addAction("Data exports are available once the 2D replay is built").setEnabled(False)

    def _item_project(self, item: QListWidgetItem | None) -> Project | None:
        if item is None:
            return None
        try:
            return self.store.get(item.data(Qt.ItemDataRole.UserRole))
        except KeyError:
            return None

    def project_menu(self, pos) -> None:
        p = self._item_project(self.list.itemAt(pos))
        if p is None:
            return
        pinned = bool(p.info().get("pinned"))
        menu = QMenu(self)
        a_rename = menu.addAction("Rename…")
        a_pin = menu.addAction("Unpin" if pinned else "Pin to top")
        self._fill_export_menu(menu.addMenu("Export"), p)
        ai_dialog.add_menu_actions(menu, self, p)
        menu.addSeparator()
        a_delete = menu.addAction("Delete…")
        chosen = menu.exec(self.list.viewport().mapToGlobal(pos))
        if chosen is a_rename:
            self.rename_project(p)
        elif chosen is a_pin:
            p.update_info(pinned=not pinned)
            self.refresh_list()
        elif chosen is a_delete:
            self.delete_project(p)

    def rename_project(self, p: Project | None) -> None:
        if p is None:
            return
        old = p.info().get("name", p.id)
        name, ok = QInputDialog.getText(self, "Rename Replay", "New name:", QLineEdit.EchoMode.Normal, old)
        name = name.strip()
        if not ok or not name or name == old:
            return
        p.update_info(name=name[:100])
        self.refresh_list()
        if self.current and self.current.id == p.id:
            self.overview_tab.name.setText(name[:100])
            self.refresh_project()

    def select_project(self, item: QListWidgetItem | None, _prev=None) -> None:
        self.replay_tab.stop()
        if item is None:
            self.current = None
            self.stack.setCurrentIndex(0)
            return
        try:
            self.current = self.store.get(item.data(Qt.ItemDataRole.UserRole))
        except KeyError:
            self.refresh_list()
            return
        self.stack.setCurrentIndex(1)
        st = self.current.status()
        rep = st.get("report") or {}
        if st["analyzed"] and st["calibrated"] and st["replay_ready"] and rep.get("build_version", 0) < BUILD_VERSION:
            self.jobs.submit(self.current.id, BUILD)  # built by an older version: add ball, boxes, new tracking
        self.overview_tab.load(self.current)
        self.calib_tab.load(self.current)
        self.replay_tab.load(self.current)
        self.stats_tab.load(self.current)
        self.refresh_project()
        self.tab_changed(self.tabs.currentIndex())

    def show_project(self, p: Project, tab: int = 0) -> None:
        self.refresh_list()
        for i in range(self.list.count()):
            if self.list.item(i).data(Qt.ItemDataRole.UserRole) == p.id:
                self.list.setCurrentRow(i)
        self.tabs.setCurrentIndex(tab)

    def tab_changed(self, i: int) -> None:
        if self.tabs.widget(i) is self.calib_tab:
            self.calib_tab.ensure_frame()
        if self.tabs.widget(i) is not self.replay_tab:
            self.replay_tab.stop()
        else:
            self.replay_tab.reload_data()

    def import_video(self) -> None:
        start = self.settings.value("last_dir", str(Path.home()))
        path, _ = QFileDialog.getOpenFileName(self, "Import game video", start, VIDEO_FILTER)
        if not path:
            return
        self.settings.setValue("last_dir", str(Path(path).parent))
        try:
            from replay.gui.busy import run_busy
            run_busy(self, media.probe, path, text="Reading the video…")  # big files: off the GUI thread
        except Exception as e:
            friendly.warn(self, "This file can't be read as a video.", e)
            return
        dlg = ImportDialog(self, path)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        p = self.store.create(dlg.name.text().strip() or Path(path).stem,
                              {"filename": Path(path).name, "kind": "reference", "path": str(Path(path).resolve())})
        self.jobs.submit(p.id, IMPORT, source=str(Path(path).resolve()), copy_original=dlg.copy.isChecked())
        self.show_project(p)

    def open_package(self) -> None:
        start = self.settings.value("last_dir", str(Path.home()))
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Replay file", start,
            f"AdvancedAnalytist Replay (*{PACKAGE_EXT} *{LEGACY_PACKAGE_EXT});;All files (*)")
        if not path:
            return
        self.settings.setValue("last_dir", str(Path(path).parent))
        try:
            p = run_with_progress(self, "Opening Replay…",
                                  lambda progress: self.store.import_package(Path(path), progress=progress))
        except Exception as e:
            friendly.warn(self, "Could not open the Replay file.", e)
            return
        if self.store.needs_playback_copy(p):
            # the original is inside but the playback copy isn't: a format conversion, not an analysis
            try:
                run_with_progress(self, "Creating the playable copy of the video…",
                                  lambda progress: self.store.rebuild_playback_copy(p, progress))
            except Exception as e:
                friendly.warn(self, "Could not create the playable copy of the video.", e)
        s = p.status()
        if s["replay_ready"]:
            # analysis results came with the file: never re-analyse; only rebuild the (seconds-fast) replay
            # if it is older than the calibration / game info and the analysis cache is there to rebuild from
            if s["replay_stale"] and s["analyzed"] and s["calibrated"]:
                self.jobs.submit(p.id, BUILD)
        elif s["analyzed"]:
            self.jobs.submit(p.id, BUILD if s["calibrated"] else AUTOCAL)
        elif s["imported"]:
            self.jobs.submit(p.id, ANALYZE)  # nothing analysed in the file (or analysis was unfinished)
        if not s["imported"]:
            QMessageBox.warning(self, APP_NAME, "This Replay has no playable video (neither video.mp4 nor the "
                                                "original video is inside the file).")
        # straight to the 2D replay when it came with the file (the tab order changed once: look it up)
        self.show_project(p, self.tabs.indexOf(self.replay_tab) if s["replay_ready"] else 0)

    def export_project(self, p: Project | None = None) -> None:
        p = p or self.current
        if p is None:
            QMessageBox.information(self, APP_NAME, "Select a Replay first.")
            return
        try:
            plan = self.store.export_plan(p.id)
        except (KeyError, OSError) as e:
            friendly.warn(self, "Cannot export this Replay.", e)
            return
        dlg = ExportDialog(self, plan)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        start = str(Path(self.settings.value("last_dir", str(Path.home()))) /
                    f"{p.info().get('name', p.id)}{PACKAGE_EXT}")
        dst, _ = QFileDialog.getSaveFileName(self, "Export Replay", start,
                                             f"AdvancedAnalytist Replay (*{PACKAGE_EXT})")
        if not dst:
            return
        if not dst.lower().endswith(PACKAGE_EXT):
            dst += PACKAGE_EXT
        self.settings.setValue("last_dir", str(Path(dst).parent))
        try:
            run_with_progress(self, "Exporting Replay…",
                              lambda progress: self.store.export(p.id, Path(dst), progress=progress, plan=plan))
        except Exception as e:
            friendly.warn(self, "Export failed.", e)
            return
        if plan.warnings:
            QMessageBox.warning(self, APP_NAME, f"Exported to {dst}, but:\n\n" + "\n".join(plan.warnings))
        self.statusBar().showMessage(f"Exported to {dst}", 8000)

    def export_data(self, p: Project | None = None) -> None:
        """Simple data export (for AI): a new timestamped folder of CSV/JSON files + instructions."""
        from replay.data_export import export_simple

        p = p or self.current
        if p is None or not p.status()["replay_ready"]:
            QMessageBox.information(self, APP_NAME, "Nothing to export yet: build the 2D replay first.")
            return
        start = self.settings.value("data_export_dir", self.settings.value("last_dir", str(Path.home())))
        parent = QFileDialog.getExistingDirectory(
            self, "Simple data export: choose where to create the data folder", start)
        if not parent:
            return
        self.settings.setValue("data_export_dir", parent)
        try:
            out = run_with_progress(self, "Exporting data…", lambda progress: export_simple(p.root, Path(parent)))
        except Exception as e:
            friendly.warn(self, "Data export failed.", e)
            return
        self.statusBar().showMessage(f"Data exported to {out}", 8000)
        if QMessageBox.question(
                self, APP_NAME, f"Data exported to the new folder\n{out}\n\nStart with README_FOR_AI.md: give the "
                                "folder (or its files) to an AI assistant for analysis.\n\nOpen the folder now?") \
                == QMessageBox.StandardButton.Yes:
            if sys.platform == "win32":
                os.startfile(str(out))
            else:
                from PySide6.QtGui import QDesktopServices
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(out)))

    def export_file(self, name: str, p: Project | None = None) -> None:
        p = p or self.current
        if p is None or not (p.results / name).exists():
            QMessageBox.information(self, APP_NAME, "Nothing to export yet: build the replay first.")
            return
        start = str(Path(self.settings.value("last_dir", str(Path.home()))) / f"{p.info().get('name', p.id)}_{name}")
        dst, _ = QFileDialog.getSaveFileName(self, "Export", start)
        if not dst:
            return
        shutil.copy2(p.results / name, dst)
        if name == "replay.html":
            QMessageBox.information(self, APP_NAME, "The stand-alone replay plays the video side by side when "
                                                    "you load the video file in it (\"Load video\").")
        self.statusBar().showMessage(f"Saved {dst}", 8000)

    def delete_project(self, p: Project | None = None) -> None:
        p = p or self.current
        if p is None:
            return
        name = p.info().get("name", p.id)
        if QMessageBox.question(self, APP_NAME, f"Delete Replay \"{name}\" and all its data from the Replays "
                                                "folder?\nThe original video file is not touched. This cannot be "
                                                "undone.") != QMessageBox.StandardButton.Yes:
            return
        self.jobs.cancel(p.id)
        if self.current and self.current.id == p.id:  # release open files before deleting
            self.replay_tab.player.setSource(QUrl())
            if self.calib_tab.grabber:
                self.calib_tab.grabber.close()
                self.calib_tab.grabber = None
            self.current = None
            self.list.setCurrentItem(None)
        if self.jobs.state(p.id).kind:  # let a running job notice the cancel and close its files
            for _ in range(50):
                if not self.jobs.state(p.id).kind:
                    break
                QApplication.processEvents()
                threading.Event().wait(0.1)
        try:
            self.store.delete(p.id)
        except OSError as e:
            friendly.warn(self, "Could not delete everything (a file may still be in use).", e)
        self.refresh_list()

    # ------------------------------------------------------------------ job status
    def poll(self) -> None:
        self.refresh_list()
        self.refresh_project()

    def refresh_project(self) -> None:
        p = self.current
        if p is None:
            return
        j = self.jobs.state(p.id).public()
        s = p.status()
        self.title.setText(p.info().get("name", p.id))
        imported = s["imported"]
        self.loading.setVisible(not imported)
        # the tabs stay usable while the video is still being imported: game info, teams and analysis settings
        # can be set right away (only the video views wait for the playable copy)
        self.tabs.setVisible(True)
        if not imported:
            if j["error"]:
                self.loading_lbl.setText("<h3>The video could not be prepared</h3><p>" + friendly.job_error_html(j["error"]) + "</p>")
            else:
                self.loading_lbl.setText("<h3>Preparing video…</h3><p style='color:gray'>"
                                         f"{j['message'] or 'Waiting to start'}</p>")
            self.loading_bar.setValue(int(j["progress"] * 1000))
        busy = bool(j["kind"] or j["queued"])
        if j["error"]:
            self.job_lbl.setText(friendly.job_error_html(j["error"]))
        elif j["kind"]:
            eta = f" · about {fmt_dur(j['eta_s'])} left" if j["eta_s"] else ""
            self.job_lbl.setText(f"<b>{j['label']}</b> - {j['message']}{eta}")
        elif j["queued"]:
            self.job_lbl.setText("<b>Queued</b> - waiting for another Replay's job to finish")
        else:
            self.job_lbl.setText(j["message"] or ("Ready." if s["replay_ready"] else ""))
        self.job_bar.setVisible(bool(j["kind"]))
        try:
            snap = self.jobs.phases(p)
            self.phase_panel.sync(p.id, snap, busy)
            from replay.gui.phases import fmt_s
            done = [r for r in snap["rows"] if r["state"] == "done" and r.get("elapsed") is not None
                    and r["key"] != "calib1"]
            total = sum(r["elapsed"] for r in done)
            self.last_build.setText(f"<span style='color:gray'>Last build: {fmt_s(total)} · </span>"
                                    "<a href='details'>details</a>" if done and not busy else "")
            self.last_build.setVisible(bool(done) and not busy and not self.phase_panel.isVisible())
        except Exception as e:  # noqa: BLE001 - the panel must never break the window
            print("phase panel:", type(e).__name__, e)
        self.job_bar.setValue(int(j["progress"] * 1000))
        self.btn_stop.setVisible(j["kind"] in (IMPORT, ANALYZE, AUTOCAL, BUILD))
        action = None
        if j["error"] and "different settings" in j["error"]:
            action = "Re-analyze with new settings"
        elif not busy and not imported and self._import_source(p) is not None:
            action = "Retry import"  # stopped or failed import (or the app was closed during it)
        elif not busy and s["imported"] and not s["analyzed"]:
            action = "Continue analysis" if s["analysis_started"] else "Analyze"
        elif not busy and s["analyzed"] and s["calibrated"] and (not s["replay_ready"] or s["replay_stale"]):
            action = "Build replay"
        elif not busy and s["analyzed"] and not s["calibrated"]:
            action = ("Fix the court by hand →" if (j["message"] or "").startswith("Could not find the court")
                      else "Calibrate automatically")
        self.btn_action.setVisible(action is not None)
        self.btn_action.setText(action or "")
        if self.tabs.currentWidget() is self.overview_tab:
            self.overview_tab.refresh()
        if self.tabs.currentWidget() is self.replay_tab:
            self.replay_tab.reload_data()
        if self.tabs.currentWidget() is self.calib_tab and s["imported"]:
            if self.calib_tab.grabber is None:
                self.calib_tab.ensure_frame()
            else:
                self.calib_tab.refresh_solver()
        if self.replay_tab.player.source().isEmpty() and s["imported"]:
            self.replay_tab.load(p)

    @staticmethod
    def _import_source(p: Project) -> Path | None:
        """The original video a Replay was (being) imported from, if it is still on disk."""
        from replay.gui.projects import original_video
        try:
            return original_video(p)
        except OSError:
            return None

    def job_action(self) -> None:
        p = self.current
        if p is None:
            return
        text = self.btn_action.text()
        if text.startswith("Re-analyze"):
            self.jobs.submit(p.id, ANALYZE, restart=True)
        elif text == "Retry import":
            src = self._import_source(p)
            if src is not None:
                self.jobs.submit(p.id, IMPORT, source=str(src))
        elif text in ("Analyze", "Continue analysis"):
            self.jobs.submit(p.id, ANALYZE)
        elif text == "Build replay":
            self.jobs.submit(p.id, BUILD)
        elif text == "Calibrate automatically":
            self.jobs.submit(p.id, AUTOCAL)
        else:
            self.tabs.setCurrentWidget(self.calib_tab)
        self.refresh_project()

    def closeEvent(self, e) -> None:
        busy = [p for p in self.store.list() if self.jobs.state(p.id).kind]
        if busy and QMessageBox.question(
                self, APP_NAME, "Analysis is running. Quit anyway? Progress is saved and the work continues "
                                "automatically the next time you start the app.") != QMessageBox.StandardButton.Yes:
            e.ignore()
            return
        self.jobs.quitting = True  # not a user "Stop": continue on the next start
        for p in busy:
            self.jobs.cancel(p.id)
        # give the worker a moment to write its resume checkpoint
        for _ in range(100):
            if not any(self.jobs.state(p.id).kind for p in busy):
                break
            QApplication.processEvents()
            threading.Event().wait(0.1)
        self.jobs.shutdown()  # job processes (calibration / build) must not outlive the window
        self.settings.setValue("geometry", self.saveGeometry())
        self.replay_tab.player.stop()
        e.accept()


def _ensure_output_streams(projects_dir: Path) -> None:
    """Started via pythonw (no console) sys.stdout/stderr are None; progress bars and prints would crash.
    Send all output to a log file instead."""
    if sys.stdout is not None and sys.stderr is not None:
        return
    projects_dir.mkdir(parents=True, exist_ok=True)
    log = open(projects_dir / "app.log", "a", encoding="utf-8", buffering=1)
    sys.stdout = sys.stdout or log
    sys.stderr = sys.stderr or log


def _startup_banner(win: "MainWindow", projects_dir: Path) -> None:
    """Self-check after the window shows (in the background: the window keeps painting); a banner if needed."""
    try:
        from replay import selfcheck
        from replay.gui.busy import run_busy
        text = run_busy(win, lambda: selfcheck.banner_text(selfcheck.startup_check(projects_dir)),
                        text="Checking the installation…", spinner_after=3.0)
    except Exception as e:  # noqa: BLE001 - a check must never stop the app
        print("self-check failed:", type(e).__name__, e)
        return
    if text:
        win.banner_lbl.setText(text)
        win.banner.show()


def run(projects_dir: Path) -> int:
    _ensure_output_streams(projects_dir)
    import faulthandler

    try:  # native crashes (Qt / ffmpeg / OpenCV) leave a stack trace in projects/crash.log
        faulthandler.enable(open(projects_dir / "crash.log", "a", encoding="utf-8"), all_threads=True)
    except OSError:
        pass
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setStyle("Fusion")
    win = MainWindow(projects_dir)
    win.show()
    QTimer.singleShot(0, lambda: _startup_banner(win, projects_dir))
    return app.exec()
