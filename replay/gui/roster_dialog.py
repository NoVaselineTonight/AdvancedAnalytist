"""Team roster dialog ("Roster..." in the Overview tab) and the 2D-view name tooltip.

Data and mapping live in replay/roster_names.py. The dialog edits the per-user roster library and this Replay's
roster.json (which roster is home / away + manual player assignments of tracks and lineup slots).
"""

from __future__ import annotations

import copy
import uuid
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QGroupBox,
                               QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMessageBox, QPushButton, QTableWidget,
                               QTableWidgetItem, QTabWidget, QToolTip, QVBoxLayout, QWidget)

from replay import roster_names as rn

TEAM_TITLE = {"A": "Home", "B": "Away"}
AUTO, NOBODY = "__auto__", "__none__"


# ====================================================================================== Overview button
def roster_button(tab: Any) -> QPushButton:
    """'Roster...' button for the Overview tab (``tab.p`` = current Project, ``tab.win`` = main window)."""
    b = QPushButton("Roster…")
    b.setToolTip("Player names and numbers of both teams: names then show in the 2D replay (hover a dot), the "
                 "timeline, playlists, data exports and the AI analysis.")

    def _open() -> None:
        p = getattr(tab, "p", None)
        if p is None:
            QMessageBox.information(tab, "Roster", "Select a Replay first.")
            return
        RosterDialog(p, getattr(tab, "win", None), tab).exec()
    b.clicked.connect(_open)
    return b


def _team_names(p: Any) -> dict[str, str]:
    names = {"A": "Home", "B": "Away"}
    try:
        from replay import team_binding as tbind
        ov = tbind.team_overview(p.meta())
        for k in ("A", "B"):
            if ov[k].get("name"):
                names[k] = str(ov[k]["name"])
    except Exception:  # noqa: BLE001
        m = p.meta() if hasattr(p, "meta") else {}
        names = {"A": m.get("home_team") or "Home", "B": m.get("away_team") or "Away"}
    return names


# ====================================================================================== one team's editor
class _TeamEditor(QWidget):
    def __init__(self, dlg: "RosterDialog", team: str):
        super().__init__()
        self.dlg, self.team = dlg, team
        self.roster: rn.Roster | None = None
        lay = QVBoxLayout(self)
        top = QHBoxLayout()
        self.pick = QComboBox()
        self.pick.setToolTip("Rosters saved on this computer (reusable in every Replay)")
        self.pick.currentIndexChanged.connect(self._picked)
        new = QPushButton("New roster")
        new.clicked.connect(self._new)
        top.addWidget(QLabel("Roster:"))
        top.addWidget(self.pick, 1)
        top.addWidget(new)
        lay.addLayout(top)
        form = QFormLayout()
        self.name, self.kit = QLineEdit(), QLineEdit()
        self.name.setPlaceholderText("Team name")
        self.kit.setPlaceholderText("Kit colour note, e.g. purple shirts, white numbers")
        form.addRow("Team", self.name)
        form.addRow("Kit", self.kit)
        lay.addLayout(form)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Number", "Name", "Position"])
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.itemChanged.connect(lambda _it: self.dlg.players_changed())
        lay.addWidget(self.table, 1)
        row = QHBoxLayout()
        for label, fn in (("Add player", self._add), ("Remove selected", self._remove),
                          ("Import CSV…", self._import), ("Export CSV…", self._export)):
            b = QPushButton(label)
            b.clicked.connect(fn)
            row.addWidget(b)
        row.addStretch(1)
        lay.addLayout(row)
        self.setEnabled_editor(False)

    def setEnabled_editor(self, on: bool) -> None:
        for w in (self.name, self.kit, self.table):
            w.setEnabled(on)

    # ---- library combo
    def fill_pick(self, library: list[rn.Roster], current: rn.Roster | None) -> None:
        self.pick.blockSignals(True)
        self.pick.clear()
        self.pick.addItem("(no roster)", None)
        ids = set()
        if current is not None and all(r.id != current.id for r in library):
            self.pick.addItem(f"{current.label()} (from this Replay)", current.id)
            ids.add(current.id)
        for r in library:
            self.pick.addItem(f"{r.label()}" + (f" – {r.kit}" if r.kit else ""), r.id)
        i = self.pick.findData(current.id) if current is not None else 0
        self.pick.setCurrentIndex(max(0, i))
        self.pick.blockSignals(False)
        self.show_roster(current)

    def _picked(self, _i: int) -> None:
        self.commit()
        rid = self.pick.currentData()
        self.show_roster(self.dlg.roster_by_id(rid) if rid else None)
        self.dlg.players_changed()

    def _new(self) -> None:
        self.commit()
        r = rn.Roster(team=self.dlg.team_names.get(self.team, ""))
        self.dlg.add_to_library(r)
        self.pick.blockSignals(True)
        self.pick.addItem(r.label(), r.id)
        self.pick.setCurrentIndex(self.pick.count() - 1)
        self.pick.blockSignals(False)
        self.show_roster(r)
        self.dlg.players_changed()

    # ---- table <-> roster
    def show_roster(self, r: rn.Roster | None) -> None:
        self.roster = r
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        self.name.setText(r.team if r else "")
        self.kit.setText(r.kit if r else "")
        for p in (r.players if r else []):
            self._append(p)
        self.table.blockSignals(False)
        self.setEnabled_editor(r is not None)

    def _append(self, p: rn.Player) -> None:
        i = self.table.rowCount()
        self.table.insertRow(i)
        for c, v in enumerate((p.number, p.name, p.position)):
            self.table.setItem(i, c, QTableWidgetItem(v))

    def commit(self) -> None:
        """Editor -> self.roster (in memory)."""
        r = self.roster
        if r is None:
            return
        r.team, r.kit = self.name.text().strip(), self.kit.text().strip()
        players = []
        for i in range(self.table.rowCount()):
            vals = [(self.table.item(i, c).text() if self.table.item(i, c) else "") for c in range(3)]
            p = rn.Player(rn.norm_number(vals[0]), vals[1].strip(), vals[2].strip())
            if p.number or p.name:
                players.append(p)
        r.players = players

    def _add(self) -> None:
        if self.roster is None:
            self._new()
        self.table.blockSignals(True)
        self._append(rn.Player("", ""))
        self.table.blockSignals(False)
        self.table.editItem(self.table.item(self.table.rowCount() - 1, 0))

    def _remove(self) -> None:
        for i in sorted({ix.row() for ix in self.table.selectedIndexes()}, reverse=True):
            self.table.removeRow(i)
        self.dlg.players_changed()

    def _import(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Import roster CSV", self.dlg.start_dir(),
                                              "CSV (*.csv *.txt);;All files (*)")
        if not path:
            return
        try:
            players = rn.players_from_csv(Path(path).read_text(encoding="utf-8-sig", errors="replace"))
        except (OSError, ValueError) as e:
            QMessageBox.warning(self, "Roster", f"Could not import {Path(path).name}:\n{e}")
            return
        if self.roster is None:
            self._new()
            if not self.name.text():
                self.name.setText(Path(path).stem)
        self.commit()
        self.roster.players = players
        self.show_roster(self.roster)
        self.dlg.players_changed()

    def _export(self) -> None:
        self.commit()
        if self.roster is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export roster CSV",
                                              str(Path(self.dlg.start_dir()) / f"{self.roster.team or 'roster'}.csv"),
                                              "CSV (*.csv)")
        if path:
            try:
                Path(path).write_text(rn.roster_to_csv(self.roster), encoding="utf-8")
            except OSError as e:
                QMessageBox.warning(self, "Roster", f"Could not write the file:\n{e}")


# ====================================================================================== dialog
class RosterDialog(QDialog):
    def __init__(self, project: Any, win: Any = None, parent: QWidget | None = None):
        super().__init__(parent)
        self.p, self.win = project, win
        self.root = Path(project.root)
        self.setWindowTitle("Team rosters")
        self.resize(900, 720)
        self.team_names = _team_names(project)
        self.library = rn.load_library()
        self.assign = rn.load_assignment(self.root) or rn.Assignment()
        self.overrides = dict(self.assign.overrides)
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("Give each team a roster. Detected jersey numbers are matched to names automatically; "
                             "below you can also assign players by hand (a manual choice always wins)."))
        self.tabs = QTabWidget()
        self.editors: dict[str, _TeamEditor] = {}
        for k in ("A", "B"):
            ed = _TeamEditor(self, k)
            self.editors[k] = ed
            self.tabs.addTab(ed, f"{TEAM_TITLE[k]}: {self.team_names[k]}")
        lay.addWidget(self.tabs, 3)
        box = QGroupBox("Players seen in this Replay")
        bl = QVBoxLayout(box)
        self.seen_rows = rn.players_seen(self.root) if (self.root / "results").exists() else []
        if not self.seen_rows:
            bl.addWidget(QLabel("<span style='color:gray'>No players yet (build the replay first). Numbers read "
                                "from shirts and lineup slots appear here.</span>"))
        self.seen = QTableWidget(0, 5)
        self.seen.setHorizontalHeaderLabels(["Player in replay", "Team", "Detected #", "Seen (s)", "Player"])
        self.seen.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        self.seen.verticalHeader().setVisible(False)
        self.seen.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        bl.addWidget(self.seen)
        lay.addWidget(box, 2)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.save)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        for k, ed in self.editors.items():
            ed.fill_pick(self.library, self.assign.roster(k))
        self._fill_seen()

    # ---- helpers for editors
    def roster_by_id(self, rid: str) -> rn.Roster | None:
        """A private copy of a library roster for one team's editor (never shared between the two teams)."""
        for r in self.library:
            if r.id == rid:
                return copy.deepcopy(r)
        for k in ("A", "B"):
            r = self.assign.roster(k)
            if r is not None and r.id == rid:
                return copy.deepcopy(r)
        return None

    def add_to_library(self, r: rn.Roster) -> None:
        self.library.append(r)

    def start_dir(self) -> str:
        try:
            return str((self.win.settings.value("last_dir") if self.win is not None else None) or Path.home())
        except Exception:  # noqa: BLE001
            return str(Path.home())

    def players_changed(self) -> None:
        self._fill_seen()

    # ---- players seen
    def _fill_seen(self) -> None:
        for ed in self.editors.values():
            ed.commit()
        a = rn.Assignment(self.editors["A"].roster, self.editors["B"].roster, self.overrides)
        rm = rn.RosterMap(a)
        self.seen.setRowCount(0)
        for row in self.seen_rows:
            i = self.seen.rowCount()
            self.seen.insertRow(i)
            what = f"Lineup slot {row['id']}" if row["kind"] == "slot" else f"Track {row['id']}"
            vals = [what, self.team_names.get(row["team"], row["team"]),
                    f"#{row['number']}" if row["number"] else "", "" if row["seconds"] is None else f"{row['seconds']:.0f}"]
            for c, v in enumerate(vals):
                self.seen.setItem(i, c, QTableWidgetItem(v))
            auto = rm.player_by_number(row["team"], row["number"]) if row["number"] else None
            cb = QComboBox()
            cb.addItem(f"Automatic ({'#' + auto.number + ' ' + auto.name if auto else 'no name'})", AUTO)
            cb.addItem("Nobody / not a roster player", NOBODY)
            r = a.roster(row["team"])
            for p in (r.players if r else []):
                cb.addItem(f"#{p.number} {p.name}".strip(), f"{p.number}\t{p.name}")
            ov = self.overrides.get(row["key"])
            if ov is not None:
                if ov.get("none"):
                    cb.setCurrentIndex(1)
                else:
                    j = cb.findData(f"{rn.norm_number(ov.get('number'))}\t{ov.get('name') or ''}")
                    if j < 0:
                        j = next((x for x in range(2, cb.count())
                                  if str(cb.itemData(x)).split("\t")[0] == rn.norm_number(ov.get("number"))), -1)
                    if j < 0 and ov.get("name"):
                        cb.addItem(f"{ov.get('name')} (not in roster)", f"{rn.norm_number(ov.get('number'))}\t{ov['name']}")
                        j = cb.count() - 1
                    cb.setCurrentIndex(max(0, j))
            cb.currentIndexChanged.connect(lambda _i, key=row["key"], team=row["team"], cb=cb: self._assigned(key, team, cb))
            self.seen.setCellWidget(i, 4, cb)

    def _assigned(self, key: str, team: str, cb: QComboBox) -> None:
        d = cb.currentData()
        if d == AUTO:
            self.overrides.pop(key, None)
        elif d == NOBODY:
            self.overrides[key] = {"none": True}
        else:
            num, _, name = str(d).partition("\t")
            self.overrides[key] = {"team": team, "number": num, "name": name}

    # ---- save
    def save(self) -> None:
        for ed in self.editors.values():
            ed.commit()
        ra, rb = self.editors["A"].roster, self.editors["B"].roster
        if ra is not None and rb is not None and ra.id == rb.id and ra.to_dict() != rb.to_dict():
            # same library roster picked for both teams and edited differently: keep both as separate rosters
            rb.id = uuid.uuid4().hex[:12]
            if rb.team == ra.team:
                rb.team = f"{rb.team or 'Roster'} ({TEAM_TITLE['B'].lower()})"
        a = rn.Assignment(ra, rb, self.overrides)
        try:
            for k in ("A", "B"):
                r = a.roster(k)
                if r is not None:
                    rn.upsert_library(r)
            rn.save_assignment(self.root, a)
        except OSError as e:
            QMessageBox.warning(self, "Roster", f"Could not save the roster:\n{e}")
            return
        self._refresh_views()
        self.accept()

    def _refresh_views(self) -> None:
        """Show the names right away in the open Replay tab (2D tooltip + timeline titles)."""
        tab = getattr(self.win, "replay_tab", None)
        if tab is None or getattr(tab, "p", None) is None or Path(tab.p.root) != self.root:
            return
        try:
            if isinstance(getattr(tab, "data", None), dict):
                rn.annotate_replay_data(self.root, tab.data)
                tab.court.update()
            tab.timeline.refresh_auto()
        except Exception as e:  # noqa: BLE001 - names show after the next reload anyway
            print("roster: views not refreshed:", type(e).__name__, e)


# ====================================================================================== 2D tooltip
def dot_at(court: Any, pos) -> tuple[dict, str, bool] | None:
    """(track, label text, is_number) of the team player's dot under widget position pos in a ReplayCourt, or None.

    Uses the court's own hit test on the dots of its last paint (ReplayCourt.dot_at, builds nothing). Only when
    nothing was painted yet, falls back to the paths of the tracks alive at t (model.alive), never all tracks."""
    hit_test = getattr(court, "dot_at", None)
    if callable(hit_test) and getattr(court, "dots", None):
        return hit_test(pos)
    from replay.gui import motion
    from replay.gui.widgets import _Geom, court_size

    data, t = court.data, court.t
    if not data or not data.get("t"):
        return None
    L, W = court_size(data)
    g = _Geom(court.rect(), L, W)
    r = max(4.5, g.s * 0.40)
    model = motion.model_for(data)
    smooth_n = int(round(court.smooth_s / model.step / 2)) if court.smooth_s > 0 else 0
    paths = model.paths(smooth_n, min(max(court.hold_s, 0.0), motion.BRIDGE_MAX))
    meta = model.meta()
    best, bd = None, (1.6 * r) ** 2
    for k in model.alive(t, 0.25, court.hold_s + 0.1):
        tr = model.tracks[k]
        if tr.get("team") not in ("A", "B") or not court.visible(tr):
            continue
        for x, y, *_rest in paths[k].at(t, court.hold_s, 0.25):
            c = g.xy(x, y)
            d = (c.x() - pos.x()) ** 2 + (c.y() - pos.y()) ** 2
            if d < bd:
                best, bd = tr, d
    if best is None:
        return None
    mt = meta.get(str(best["id"]))
    text, is_num = motion.label_at(mt, t) if mt is not None else (str(best["id"]), False)
    return best, text, is_num


def court_tooltip(court: Any, e: Any) -> bool:
    """ReplayCourt ToolTip event (roster hook in widgets.py): '#23 Virtanen' for the dot under the mouse."""
    try:
        hit = dot_at(court, e.pos())
    except Exception:  # noqa: BLE001 - a tooltip must never break the view
        hit = None
    if hit is None:
        QToolTip.hideText()
        e.ignore()
        return True
    tr, text, is_num = hit
    num = text if is_num else str(((court.data.get("track_info") or {}).get(str(tr["id"])) or {}).get("number") or "")
    name = rn.dot_name(court.data, tr["id"], tr.get("team"), num)
    team = (((court.data.get("game") or {}).get("teams") or {}).get(tr.get("team")) or {}).get("name") or tr.get("team")
    tip = " ".join(x for x in (f"#{num}" if num else "", name) if x) or f"Track {tr['id']}"
    QToolTip.showText(e.globalPos(), f"{tip} – {team}", court)
    return True
