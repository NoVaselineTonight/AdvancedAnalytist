"""Scoresheet (poytakirja) in the GUI: the Overview tab's "Add scoresheet..." button and the Scoresheet window.

Data, parsing and the cross-check live in replay/scoresheet.py (no Qt). The window shows what was read from the
sheet, fills the Replay's roster (asking before replacing names the user typed), writes the score truth for
whole-game Replays and lists the "check this" items; double-clicking an item jumps to that moment in the Replay.
Nothing is sent anywhere: everything stays in the Replay folder on this computer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog,
                               QHBoxLayout, QHeaderView, QLabel, QMessageBox, QPushButton, QSpinBox, QTableWidget,
                               QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget)

from replay import scoresheet as S

FILE_FILTER = ("Scoresheets (*.pdf *.csv *.json);;PDF (*.pdf);;CSV (*.csv);;"
               "Photos / scans (*.jpg *.jpeg *.png *.heic *.tif *.tiff);;All files (*)")
T_ROLE = Qt.ItemDataRole.UserRole + 1


# ====================================================================================== Overview button
class _SheetButton(QPushButton):
    """'Add scoresheet...' (no sheet yet) / 'Scoresheet...' (opens the window)."""

    def __init__(self, tab: Any):
        super().__init__("Add scoresheet…")
        self.tab = tab
        self.setToolTip("Add the game's official scoresheet (pöytäkirja: the federation's digital PDF, or a CSV). "
                        "Player names and numbers fill the roster, the final score becomes the score truth, and the "
                        "sheet is cross-checked against the video. Everything stays on this computer.")
        self.clicked.connect(self._clicked)

    def refresh(self) -> None:
        p = getattr(self.tab, "p", None)
        has = p is not None and S.sheet_path(p.root).exists()
        self.setText("Scoresheet…" if has else "Add scoresheet…")

    def showEvent(self, e: Any) -> None:  # noqa: N802 - Qt override
        self.refresh()
        super().showEvent(e)

    def enterEvent(self, e: Any) -> None:  # noqa: N802 - Qt override
        self.refresh()
        super().enterEvent(e)

    def _clicked(self) -> None:
        p = getattr(self.tab, "p", None)
        if p is None:
            QMessageBox.information(self.tab, "Scoresheet", "Select a Replay first.")
            return
        win = getattr(self.tab, "win", None)
        if S.sheet_path(p.root).exists():
            open_dialog(p, win, self.tab)
        else:
            sheet = add_scoresheet(p, self.tab)
            if sheet is not None:
                open_dialog(p, win, self.tab, just_imported=True)
        self.refresh()


def scoresheet_button(tab: Any) -> QPushButton:
    """Button for the Overview tab (``tab.p`` = current Project, ``tab.win`` = main window)."""
    return _SheetButton(tab)


def load_inputs(root: Path, project: Any = None) -> dict:
    """Everything the window shows, read and cross-checked (slow on a whole game: run off the GUI thread)."""
    root = Path(root)
    sheet = S.load(root)
    replay = _read(root / "results" / "replay_data.json")
    gs = _read(root / "results" / "game_state.json")
    try:
        meta = project.meta() if project is not None else _read(root / "meta.json") or {}
    except Exception:  # noqa: BLE001
        meta = _read(root / "meta.json") or {}
    rep = S.crosscheck(root, sheet, replay, gs, meta) if sheet is not None else None
    return {"sheet": sheet, "replay": replay, "gs": gs, "meta": meta, "rep": rep}


def open_dialog(p: Any, win: Any, parent: QWidget | None, just_imported: bool = False) -> None:
    """Read + cross-check in the background (busy cursor), then show the window."""
    from replay.gui.busy import run_busy
    try:
        pre = run_busy(parent, load_inputs, Path(p.root), p, text="Checking the scoresheet against the video…")
    except Exception as e:  # noqa: BLE001
        QMessageBox.warning(parent, "Scoresheet", f"The scoresheet could not be checked:\n\n{type(e).__name__}: {e}")
        return
    ScoresheetDialog(p, win, parent, just_imported=just_imported, preload=pre).exec()


def add_scoresheet(p: Any, parent: QWidget | None) -> S.Scoresheet | None:
    """Pick a file, parse it and store it in the Replay. Shows the error and returns None when it cannot be read."""
    f, _ = QFileDialog.getOpenFileName(parent, "Add the game's scoresheet", "", FILE_FILTER)
    if not f:
        return None
    try:
        from replay.gui.busy import run_busy
        return run_busy(parent, S.import_into, p.root, Path(f), text="Reading the scoresheet…")
    except S.ScannedSheetError as e:
        QMessageBox.information(parent, "Scanned scoresheet", str(e))
    except S.ScoresheetError as e:
        QMessageBox.warning(parent, "Scoresheet", f"The file could not be read as a scoresheet:\n\n{e}")
    except OSError as e:
        QMessageBox.warning(parent, "Scoresheet", f"The file could not be copied into the Replay:\n\n{e}")
    return None


def _read(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _item(text: Any, t: float | None = None, align_right: bool = False) -> QTableWidgetItem:
    it = QTableWidgetItem("" if text is None else str(text))
    it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable)
    if t is not None:
        it.setData(T_ROLE, float(t))
    if align_right:
        it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    return it


def _table(headers: list[str], stretch: int | None = None) -> QTableWidget:
    t = QTableWidget(0, len(headers))
    t.setHorizontalHeaderLabels(headers)
    t.verticalHeader().setVisible(False)
    t.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    t.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
    if stretch is not None:
        t.horizontalHeader().setSectionResizeMode(stretch, QHeaderView.ResizeMode.Stretch)
    return t


# ====================================================================================== window
class ScoresheetDialog(QDialog):
    def __init__(self, project: Any, win: Any = None, parent: QWidget | None = None, just_imported: bool = False,
                 preload: dict | None = None):
        super().__init__(parent)
        self.p, self.win = project, win
        self.root = Path(project.root)
        self.setWindowTitle("Scoresheet")
        self.resize(1000, 720)
        pre = preload or load_inputs(self.root, project)
        self.sheet, self.replay, self.gs, self.meta = pre["sheet"], pre["replay"], pre["gs"], pre["meta"]
        self._pre_rep = pre.get("rep")  # cross-check already computed (background) - used by the first refresh
        lay = QVBoxLayout(self)
        self.head = QLabel()
        self.head.setWordWrap(True)
        self.head.setTextFormat(Qt.TextFormat.RichText)
        lay.addWidget(self.head)

        row = QHBoxLayout()
        self.team_combo = QComboBox()
        self.team_combo.setToolTip("Which jersey colour in the video is which team on the sheet")
        self.team_combo.activated.connect(self._team_map_changed)
        row.addWidget(QLabel("Teams:"))
        row.addWidget(self.team_combo, 1)
        row.addWidget(QLabel("Score when this Replay starts:"))
        self.start_a, self.start_b = QSpinBox(), QSpinBox()
        for sb in (self.start_a, self.start_b):
            sb.setRange(0, 300)
        self.start_known = QCheckBox("known")
        self.start_known.setToolTip("For a clip from the middle of a game: the score on the scoreboard at its first "
                                    "frame. The sheet's running score is then matched to the baskets in the clip.")
        for w in (self.start_a, QLabel("-"), self.start_b, self.start_known):
            row.addWidget(w)
        apply_start = QPushButton("Apply")
        apply_start.clicked.connect(self._start_changed)
        row.addWidget(apply_start)
        lay.addLayout(row)
        # a clip from the middle of a game: where is it? one click per quarter (best fit of the detected baskets)
        self.where = QWidget()
        wl = QHBoxLayout(self.where)
        wl.setContentsMargins(0, 0, 0, 0)
        self.where_lbl = QLabel()
        self.where_lbl.setWordWrap(True)
        self.where_lbl.setMinimumWidth(420)
        wl.addWidget(self.where_lbl)
        self.where_btns = QHBoxLayout()
        wl.addLayout(self.where_btns)
        wl.addStretch(1)
        self.where.setStyleSheet("QWidget { background: #fef3c7; } QLabel { color: #92400e; }")
        self.where.setVisible(False)
        lay.addWidget(self.where)

        self.tabs = QTabWidget()
        lay.addWidget(self.tabs, 1)
        # check list
        chk = QWidget()
        cl = QVBoxLayout(chk)
        fl = QHBoxLayout()
        self.filter = QComboBox()
        for label, key in [("Items to check", "check"), ("Everything", "all"),
                           ("Scoresheet vs video", "sheet"), ("Fouls / free throws", "fouls"),
                           ("Video detections", "video")]:
            self.filter.addItem(label, key)
        self.filter.currentIndexChanged.connect(self._fill_checks)
        fl.addWidget(QLabel("Show:"))
        fl.addWidget(self.filter)
        fl.addStretch(1)
        fl.addWidget(QLabel("<span style='color:gray'>Double-click a row to jump to it in the Replay.</span>"))
        cl.addLayout(fl)
        self.checks = _table(["Time", "", "What", "Details", "Confidence"], stretch=3)
        self.checks.cellDoubleClicked.connect(lambda r, _c: self._jump(self.checks, r))
        cl.addWidget(self.checks, 1)
        self.notes = QLabel()
        self.notes.setWordWrap(True)
        self.notes.setTextFormat(Qt.TextFormat.RichText)
        cl.addWidget(self.notes)
        self.tabs.addTab(chk, "Scoresheet check")
        # players
        self.players = _table(["Team", "#", "Name", "Points (sheet)", "Points (video)", "FT / 2P / 3P (sheet)",
                               "Fouls (sheet)", "Number read in video", ""], stretch=2)
        self.tabs.addTab(self.players, "Players")
        # running score
        self.running = _table(["Q", "Score", "Team", "Points", "Scorer #", "Name", "Video time"], stretch=5)
        self.running.cellDoubleClicked.connect(lambda r, _c: self._jump(self.running, r))
        self.tabs.addTab(self.running, "Running score")

        btns = QHBoxLayout()
        b_roster = QPushButton("Fill roster from sheet…")
        b_roster.setToolTip("Add the sheet's names and numbers to this Replay's roster (asks before replacing "
                            "names you typed)")
        b_roster.clicked.connect(lambda: self.fill_roster(ask_always=True))
        b_truth = QPushButton("Use as score truth")
        b_truth.setToolTip("Write the sheet's final and quarter scores to score_truth.json (used by Stats)")
        b_truth.clicked.connect(lambda: self.score_truth(interactive=True))
        b_replace = QPushButton("Replace file…")
        b_replace.clicked.connect(self._replace)
        b_remove = QPushButton("Remove scoresheet")
        b_remove.clicked.connect(self._remove)
        for b in (b_roster, b_truth, b_replace, b_remove):
            btns.addWidget(b)
        btns.addStretch(1)
        box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        box.rejected.connect(self.reject)
        box.accepted.connect(self.accept)
        btns.addWidget(box)
        lay.addLayout(btns)
        self.status = QLabel()
        self.status.setWordWrap(True)
        lay.addWidget(self.status)
        self.refresh()
        if just_imported and self.sheet is not None:
            msgs = [self.fill_roster(ask_always=False), self.score_truth(interactive=False)]
            self.status.setText(" ".join(m for m in msgs if m))

    # -------------------------------------------------------------- data
    def refresh(self) -> None:
        if self.sheet is None:
            self.head.setText("<b>No scoresheet in this Replay.</b>")
            return
        s = self.sheet
        warn = (f"<br><b style='color:#b45309'>The sheet disagrees with itself in {len(s.warnings)} place(s)</b> "
                "(see the check list)." if s.warnings else "")
        self.head.setText(f"<b>{s.summary()}</b><br><span style='color:gray'>{s.source} · "
                          f"{s.game.get('competition', '')} · {s.game.get('venue', '')}. The sheet is written by "
                          "hand at the table: the final score is reliable, scorer numbers and fouls may contain "
                          f"mistakes.</span>{warn}")
        st = s.game.get("replay_start_score")
        self.start_known.setChecked(isinstance(st, list) and len(st) == 2)
        if isinstance(st, list) and len(st) == 2:
            self.start_a.setValue(int(st[0]))
            self.start_b.setValue(int(st[1]))
        self._fill_team_combo()
        pre, self._pre_rep = getattr(self, "_pre_rep", None), None
        self.rep = pre if pre is not None else S.crosscheck(self.root, s, self.replay, self.gs, self.meta)
        self._fill_checks()
        self._fill_players()
        self._fill_running()
        self._fill_where()

    def _fill_team_combo(self) -> None:
        s = self.sheet
        tm, src = S.team_map_info(s, self.replay, self.meta)
        vt = S.video_teams(self.replay, self.meta)

        def lab(m: dict) -> str:
            return ", ".join(f"{s.name(k)} = {vt.get(m[k], {}).get('colour') or 'video team ' + m[k]}" for k in S.TEAMS)
        self.team_combo.blockSignals(True)
        self.team_combo.clear()
        self.team_combo.addItem(f"{lab(tm)}   ({src})", tm)
        other = {k: ("B" if v == "A" else "A") for k, v in tm.items()}
        self.team_combo.addItem(lab(other), other)
        self.team_combo.setCurrentIndex(0)
        self.team_combo.blockSignals(False)

    def _fill_checks(self) -> None:
        rep = getattr(self, "rep", None) or {}
        key = self.filter.currentData()
        rows = [i for i in rep.get("items") or [] if key == "all" or (key == "check" and i["severity"] == "check")
                or i.get("category") == key]
        self.checks.setRowCount(0)
        for it in rows:
            r = self.checks.rowCount()
            self.checks.insertRow(r)
            mark = "⚠" if it["severity"] == "check" else "ⓘ"
            self.checks.setItem(r, 0, _item(S.fmt_t(it["t"]) if it["t"] is not None else "", it["t"]))
            self.checks.setItem(r, 1, _item(mark, it["t"]))
            self.checks.setItem(r, 2, _item(it["title"], it["t"]))
            d = _item(it.get("detail") or "", it["t"])
            d.setToolTip(it.get("detail") or "")
            self.checks.setItem(r, 3, d)
            self.checks.setItem(r, 4, _item(it.get("confidence") or "", it["t"]))
        n_all = len(rep.get("items") or [])
        notes = "<br>".join(f"· {n}" for n in (rep.get("notes") or [])[:12])
        self.notes.setText(f"<b>{rep.get('summary', '')}</b> Showing {len(rows)} of {n_all}.<br>"
                           f"<span style='color:gray'>{notes}</span>")

    def _fill_players(self) -> None:
        rep = getattr(self, "rep", None) or {}
        self.players.setRowCount(0)
        for p in rep.get("players") or []:
            r = self.players.rowCount()
            self.players.insertRow(r)
            sp = self.sheet.teams[p["team"]].player(p["number"]) if not p.get("missing") else None
            shots = "" if sp is None or sp.ft_made is None else f"{sp.ft_made} / {sp.fg2_made} / {sp.fg3_made}"
            flags = " ".join(x for x in ("(C)" if p.get("captain") else "", "starter" if p.get("starter") else "")
                             if x)
            vals = [self.sheet.name(p["team"]), p["number"], p["name"], p["sheet_points"],
                    f"{p['video_points']} ({p['video_baskets']} baskets)" if p["video_baskets"] else "",
                    shots, p["sheet_fouls"], f"{p['seen_reads']} reads, {p['seen_s']:.0f} s" if p["seen_reads"] else
                    "not read", flags or ("missing from the sheet?" if p.get("missing") else "")]
            for c, v in enumerate(vals):
                self.players.setItem(r, c, _item(v, align_right=c in (1, 3, 6)))

    def _fill_running(self) -> None:
        rep = getattr(self, "rep", None) or {}
        times: dict[tuple, float] = {}
        for pr in rep.get("pairs") or []:
            s = pr.get("sheet")
            if s and s.get("t") is not None:
                times[tuple(s["score"])] = s["t"]
        self.running.setRowCount(0)
        for e in self.sheet.running:
            r = self.running.rowCount()
            self.running.insertRow(r)
            t = times.get((e.score_a, e.score_b))
            pl = self.sheet.teams[e.team].player(e.scorer) if e.scorer else None
            vals = [e.period or "", f"{e.score_a}-{e.score_b}", self.sheet.name(e.team),
                    {1: "free throw", 2: "2", 3: "3"}.get(e.points, e.points), e.scorer or "?",
                    pl.name if pl else "", S.fmt_t(t) if t is not None else ""]
            for c, v in enumerate(vals):
                self.running.setItem(r, c, _item(v, t))

    # -------------------------------------------------------------- actions
    def _jump(self, table: QTableWidget, row: int) -> None:
        it = table.item(row, 0)
        t = it.data(T_ROLE) if it is not None else None
        if t is None:
            return
        win = self.win
        cur = getattr(win, "current", None)
        rt = getattr(win, "replay_tab", None)
        if rt is None or cur is None or Path(cur.root) != self.root:
            return
        try:
            if hasattr(win, "tabs"):
                win.tabs.setCurrentWidget(rt)
            rt.seek_to(max(0.0, float(t) - 2.0))  # a little lead-in
        except Exception as e:  # noqa: BLE001
            self.status.setText(f"Could not jump: {e}")

    def _team_map_changed(self, idx: int) -> None:
        if idx <= 0 or self.sheet is None:
            return
        self.sheet.game["team_map"] = self.team_combo.itemData(idx)
        S.save(self.root, self.sheet)
        self.status.setText("Team colours swapped for the scoresheet. Use 'Fill roster from sheet…' again if the "
                            "roster was filled with the old choice.")
        self.refresh()

    def _fill_where(self) -> None:
        """Ambiguous position of a clip in the game: offer the quarters (best fit each) as one-click choices."""
        while self.where_btns.count():
            w = self.where_btns.takeAt(0).widget()
            if w is not None:
                w.deleteLater()
        rep = getattr(self, "rep", None) or {}
        if rep.get("start_source") == "scorebar" and rep.get("pin"):  # placed by the score bar: nothing to pick
            self.where_lbl.setText(f"<b>Pinned from the score bar:</b> {rep['pin']['label']}. Set the score at the "
                                   "start by hand (above) to override.")
            self.where.setStyleSheet("QWidget { background: #dcfce7; } QLabel { color: #166534; }")
            self.where.setVisible(True)
            return
        self.where.setStyleSheet("QWidget { background: #fef3c7; } QLabel { color: #92400e; }")
        known = isinstance(self.sheet.game.get("replay_start_score"), list)
        if rep.get("full_game") or known or rep.get("alignment") not in ("low", None):
            self.where.setVisible(False)
            return
        try:
            cands = S.clip_candidates(self.sheet, self.replay, self.gs, self.meta)
        except Exception as e:  # noqa: BLE001
            print("scoresheet: clip candidates failed:", type(e).__name__, e)
            cands = []
        self.where.setVisible(bool(cands))
        if not cands:
            return
        self.where_lbl.setText("<b>Where is this clip in the game?</b> The baskets fit several places on the sheet. "
                               "Pick the quarter (the score shown is when the clip starts):")
        for c in cands:
            b = QPushButton(c["label"].split(":")[0] + (" ★" if c.get("best") else ""))
            b.setToolTip(c["label"])
            b.clicked.connect(lambda _=False, c=c: self.pick_start(c))
            self.where_btns.addWidget(b)

    def pick_start(self, c: dict) -> None:
        self.start_a.setValue(int(c["start_score"][0]))
        self.start_b.setValue(int(c["start_score"][1]))
        self.start_known.setChecked(True)
        self._start_changed()
        self.status.setText(f"Clip pinned to {c['label'].split(':')[0]}. The coach report and checks now use it.")

    def _start_changed(self) -> None:
        if self.sheet is None:
            return
        if self.start_known.isChecked():
            self.sheet.game["replay_start_score"] = [self.start_a.value(), self.start_b.value()]
        else:
            self.sheet.game.pop("replay_start_score", None)
        S.save(self.root, self.sheet)
        self.refresh()

    def fill_roster(self, ask_always: bool) -> str:
        if self.sheet is None:
            return ""
        plan = S.roster_plan(self.root, self.sheet, self.replay, self.meta)
        replace = False
        if plan["conflicts"]:
            lines = []
            for vt in S.TEAMS:
                for num, mine, theirs in plan[vt]["conflicts"]:
                    lines.append(f"#{num}: your name '{mine}', sheet '{theirs}'")
            mb = QMessageBox(self)
            mb.setWindowTitle("Roster")
            mb.setText(f"The scoresheet has different names for {plan['conflicts']} number(s) that already have a "
                       "name in this Replay's roster.")
            mb.setDetailedText("\n".join(lines))
            keep = mb.addButton("Keep my names", QMessageBox.ButtonRole.AcceptRole)
            use = mb.addButton("Use the scoresheet's names", QMessageBox.ButtonRole.DestructiveRole)
            mb.addButton(QMessageBox.StandardButton.Cancel)
            mb.setDefaultButton(keep)
            mb.exec()
            if mb.clickedButton() not in (keep, use):
                return "Roster not changed."
            replace = mb.clickedButton() is use
        elif ask_always:
            n_new = sum(len(plan[vt]["new"]) for vt in S.TEAMS)
            if n_new == 0:
                QMessageBox.information(self, "Roster", "The roster already has every name on the scoresheet.")
                return ""
            if QMessageBox.question(self, "Roster", f"Add {n_new} player(s) from the scoresheet to this Replay's "
                                    "roster?") != QMessageBox.StandardButton.Yes:
                return ""
        res = S.apply_roster(self.root, self.sheet, replace, self.replay, self.meta)
        n_new = sum(len(res[vt]["new"]) for vt in S.TEAMS)
        msg = f"Roster: {n_new} player(s) added" + (", conflicting names replaced" if replace else "") + "."
        self._reload_views()
        self.status.setText(msg)
        return msg

    def score_truth(self, interactive: bool) -> str:
        if self.sheet is None:
            return ""
        ok, why = S.write_score_truth(self.root, self.sheet, self.replay, self.meta)
        if not ok and interactive and "already exists" in why:
            if QMessageBox.question(self, "Score truth", "This Replay already has a score_truth.json that was not "
                                    "made from a scoresheet. Replace it?") == QMessageBox.StandardButton.Yes:
                ok, why = S.write_score_truth(self.root, self.sheet, self.replay, self.meta, force=True)
        elif not ok and interactive and "not the whole game" in why:
            if QMessageBox.question(self, "Score truth", why[0].upper() + why[1:] + ".\n\nUse it anyway?") == \
                    QMessageBox.StandardButton.Yes:
                ok, why = S.write_score_truth(self.root, self.sheet, self.replay, self.meta, force=True)
        msg = ("Score truth: " + why[0].lower() + why[1:]) if why else ""
        self.status.setText(msg + ".")
        return msg + "."

    def _replace(self) -> None:
        sheet = add_scoresheet(self.p, self)
        if sheet is None:
            return
        self.sheet = sheet
        msgs = [self.fill_roster(ask_always=False), self.score_truth(interactive=False)]
        self.refresh()
        self.status.setText(" ".join(m for m in msgs if m))

    def _remove(self) -> None:
        if QMessageBox.question(self, "Scoresheet", "Remove the scoresheet from this Replay? (The roster is kept.)") \
                != QMessageBox.StandardButton.Yes:
            return
        S.remove(self.root)
        self.sheet = None
        self.accept()

    def _reload_views(self) -> None:
        rt = getattr(self.win, "replay_tab", None)
        cur = getattr(self.win, "current", None)
        if rt is not None and cur is not None and Path(cur.root) == self.root and hasattr(rt, "reload_data"):
            try:
                rt.reload_data()
            except Exception:  # noqa: BLE001 - names are decoration
                pass
