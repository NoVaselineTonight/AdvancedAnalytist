"""Clip playlists panel (Qt): pick a filter ("all our turnovers", "all their fast breaks", "every possession of
#12", ...) and watch the clips back to back in the Replay tab's video player (the 2D view follows the video).

Hooked into the main window by :func:`install` (one line in app.py): a "Playlists" dock panel (right side, can be
floated) and a "Playlists" menu. The data side (catalog, filters, clip merging, storage, export) is
``replay/playlists.py``. Playlists are saved in ``<Replay>/playlists.json`` and travel inside the ``.aareplay``
package.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QColor, QKeySequence
from PySide6.QtMultimedia import QMediaPlayer
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDockWidget, QDoubleSpinBox,
                               QFileDialog, QFormLayout, QHBoxLayout, QInputDialog, QLabel, QLineEdit, QListWidget,
                               QListWidgetItem, QMenu, QMessageBox, QProgressDialog, QPushButton, QSpinBox,
                               QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from replay import playlists as pl

if TYPE_CHECKING:  # pragma: no cover
    from replay.gui.app import MainWindow
    from replay.gui.projects import Project

COL_DONE, COL_NUM, COL_START, COL_LEN, COL_TEAM, COL_TITLE, COL_NOTE = range(7)
PLAYING_BG = QColor("#fde68a")
END_GRACE_S = 0.4  # after a seek the player may still report the old position for a moment


def _settings_coached(p: "Project | None") -> str | None:
    return pl.coached_team(p.settings()) if p is not None else None


def save_coached(p: "Project", team: str | None) -> None:
    """Store the coached team where the AI dialog keeps it (settings["ai"]["coached_team"])."""
    st = p.settings()
    ai = dict(st.get("ai") or {}) if isinstance(st.get("ai"), dict) else {}
    if team in pl.TEAMS:
        ai["coached_team"] = team
    else:
        ai.pop("coached_team", None)
    st["ai"] = ai
    p.update_info(settings=st)


# ====================================================================================== filter dialog
class FilterDialog(QDialog):
    """Build (or edit) a playlist: preset, event types, team, player, period / time range, confidence, shot
    distance, pre-/post-roll. Shows a live count of the clips it would give."""

    def __init__(self, parent: QWidget, cat: pl.Catalog, coached: str | None, playlist: pl.Playlist | None = None,
                 preset: int | None = None):
        super().__init__(parent)
        self.cat = cat
        self.coached = coached
        self.setWindowTitle("Edit playlist" if playlist else "New playlist")
        self.setMinimumWidth(560)
        v = QVBoxLayout(self)
        form = QFormLayout()
        v.addLayout(form)

        self.preset = QComboBox()
        form.addRow("Start from", self.preset)
        self.name = QLineEdit()
        self.name.setPlaceholderText("e.g. Our turnovers")
        form.addRow("Name", self.name)
        self.we = QComboBox()
        self.we.addItem("Not set", None)
        for k in pl.TEAMS:
            self.we.addItem(cat.names.get(k, k), k)
        self.we.setToolTip("The team you coach: the filters then speak of “Our team” and “Opponent”. Same setting "
                           "as in the AI analysis dialog.")
        form.addRow("We coach", self.we)

        self.cats = QListWidget()
        self.cats.setMinimumHeight(170)
        counts = cat.category_counts()
        keys = [k for k, _ in pl.CATEGORIES] + [f"track:{t['id']}" for t in cat.tracks]
        for k in keys:
            n = counts.get(k, 0)
            if n == 0 and k.startswith("track:"):
                continue
            it = QListWidgetItem(f"{cat.category_label(k)}  ({n})")
            it.setData(Qt.ItemDataRole.UserRole, k)
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(Qt.CheckState.Unchecked)
            if n == 0:
                it.setForeground(QColor("#9ca3af"))
            self.cats.addItem(it)
        form.addRow("Show", self.cats)

        self.team = QComboBox()
        form.addRow("Team", self.team)
        prow = QHBoxLayout()
        self.player = QComboBox()
        self.player.setEditable(True)
        self.player.addItem("Anyone", None)
        for key, label in cat.player_choices():
            self.player.addItem(label, key)
        self.player.lineEdit().setPlaceholderText("Anyone  ·  type #12, A3 (slot) or T42 (track id)")
        self.player.setToolTip("Jersey numbers are read automatically (when readable). Slots A1-A5 / B1-B5 follow an "
                               "on-court position across tracker ID changes. Several: separate with commas.")
        self.player_mode = QComboBox()
        self.player_mode.addItem("had the ball / named in the event", "involved")
        self.player_mode.addItem("was on the court", "on_court")
        prow.addWidget(self.player, 1)
        prow.addWidget(self.player_mode)
        form.addRow("Player", prow)

        self.period = QComboBox()
        self.period.addItem("Whole video", None)
        for i, h in enumerate(cat.halves):
            self.period.addItem(f"{h.get('label') or f'Period {i + 1}'} ({pl.fmt_clock(h.get('t0'), 0)}–"
                                f"{pl.fmt_clock(h.get('t1'), 0)})", i)
        form.addRow("Period", self.period)
        trow = QHBoxLayout()
        self.t_from, self.t_to = QLineEdit(), QLineEdit()
        for w, ph in ((self.t_from, "start"), (self.t_to, "end")):
            w.setPlaceholderText(ph + "  (m:ss)")
            trow.addWidget(w)
        form.addRow("Video time", trow)
        self.conf = QSpinBox()
        self.conf.setRange(0, 100)
        self.conf.setSuffix(" %")
        self.conf.setToolTip("Leave out automatically detected events below this confidence (your own marks always "
                             "stay)")
        form.addRow("Min. confidence", self.conf)
        drow = QHBoxLayout()
        self.dmin, self.dmax = QDoubleSpinBox(), QDoubleSpinBox()
        for w, lab in ((self.dmin, "from"), (self.dmax, "to")):
            w.setRange(0, 30)
            w.setDecimals(1)
            w.setSuffix(" m")
            w.setSpecialValueText("any")
            drow.addWidget(QLabel(lab))
            drow.addWidget(w)
        drow.addStretch(1)
        form.addRow("Shot distance", drow)
        self.text = QLineEdit()
        self.text.setPlaceholderText("words in the title or note (optional)")
        form.addRow("Text", self.text)
        rrow = QHBoxLayout()
        self.pre, self.post, self.gap = QDoubleSpinBox(), QDoubleSpinBox(), QDoubleSpinBox()
        for w, lab, tip in ((self.pre, "Before", "Seconds of video before each event (pre-roll)"),
                            (self.post, "After", "Seconds after the event ends (post-roll)"),
                            (self.gap, "Join gaps under", "Clips closer than this are joined into one")):
            w.setRange(0, 60)
            w.setDecimals(1)
            w.setSuffix(" s")
            w.setToolTip(tip)
            rrow.addWidget(QLabel(lab))
            rrow.addWidget(w)
        rrow.addStretch(1)
        form.addRow("Clip length", rrow)

        self.preview = QLabel()
        self.preview.setWordWrap(True)
        v.addWidget(self.preview)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.button(QDialogButtonBox.StandardButton.Ok).setText("Save playlist")
        bb.accepted.connect(self._accept)
        bb.rejected.connect(self.reject)
        v.addWidget(bb)

        self._loading = True
        self.we.setCurrentIndex(max(0, self.we.findData(coached)))
        self._fill_teams()
        self._fill_presets()
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(120)
        self._timer.timeout.connect(self._update_preview)
        if playlist is not None:
            self._set(playlist.name, playlist.filter, playlist.pre_s, playlist.post_s, playlist.merge_gap_s)
        elif preset is not None and 0 <= preset < len(self._presets):
            self.preset.setCurrentIndex(preset + 1)
            self._apply_preset(preset + 1)
        else:
            self._set("", pl.Filter(), pl.DEFAULT_PRE_S, pl.DEFAULT_POST_S, 0.0)
        self._loading = False
        self.preset.currentIndexChanged.connect(self._apply_preset)
        self.we.currentIndexChanged.connect(self._coached_changed)
        self.cats.itemChanged.connect(self._changed)
        for w in (self.team, self.player_mode, self.period):
            w.currentIndexChanged.connect(self._changed)
        self.player.editTextChanged.connect(self._changed)
        for w in (self.t_from, self.t_to, self.text):
            w.textChanged.connect(self._changed)
        for w in (self.conf, self.dmin, self.dmax, self.pre, self.post, self.gap):
            w.valueChanged.connect(self._changed)
        self._update_preview()

    # ---------------------------------------------------------------- fields
    def _fill_teams(self) -> None:
        cur = self.team.currentData() if self.team.count() else None
        self.team.blockSignals(True)
        self.team.clear()
        self.team.addItem("Both teams", None)
        order = [self.coached, pl.other_team(self.coached)] if self.coached in pl.TEAMS else list(pl.TEAMS)
        for k in order:
            self.team.addItem(pl.team_label(k, self.coached, self.cat.names), k)
        self.team.setCurrentIndex(max(0, self.team.findData(cur)))
        self.team.blockSignals(False)

    def _fill_presets(self) -> None:
        self._presets = pl.presets(self.coached, self.cat.names)
        cur = self.preset.currentIndex()
        self.preset.blockSignals(True)
        self.preset.clear()
        self.preset.addItem("Custom filter")
        counts = self.cat.category_counts()
        for name, f, _a, _b in self._presets:
            n = len(pl.apply_filter(self.cat, f)) if any(counts.get(c) for c in f.categories) else 0
            self.preset.addItem(f"{name}  ({n})")
        self.preset.setCurrentIndex(max(0, cur))
        self.preset.blockSignals(False)

    def _coached_changed(self) -> None:
        self.coached = self.we.currentData()
        self._fill_teams()
        self._fill_presets()
        self._changed()

    def _apply_preset(self, i: int) -> None:
        if i <= 0 or i > len(self._presets):
            return
        name, f, a, b = self._presets[i - 1]
        loading, self._loading = self._loading, True
        self._set(name, f, a, b, 0.0)
        self._loading = loading
        self._changed()

    def _set(self, name: str, f: pl.Filter, pre: float, post: float, gap: float) -> None:
        self.name.setText(name)
        for i in range(self.cats.count()):
            it = self.cats.item(i)
            on = it.data(Qt.ItemDataRole.UserRole) in f.categories
            it.setCheckState(Qt.CheckState.Checked if on else Qt.CheckState.Unchecked)
        self.team.setCurrentIndex(max(0, self.team.findData(f.team)))
        if f.players:
            labels = [self.cat.player_label(k) for k in f.players]
            j = self.player.findData(f.players[0]) if len(f.players) == 1 else -1
            if j >= 0:
                self.player.setCurrentIndex(j)
            else:
                self.player.setEditText(", ".join(_key_text(k) for k in f.players) if labels else "")
        else:
            self.player.setCurrentIndex(0)
        self.player_mode.setCurrentIndex(max(0, self.player_mode.findData(f.player_mode)))
        self.period.setCurrentIndex(max(0, self.period.findData(f.period)))
        self.t_from.setText(pl.fmt_clock(f.t_from) if f.t_from is not None else "")
        self.t_to.setText(pl.fmt_clock(f.t_to) if f.t_to is not None else "")
        self.conf.setValue(int(round(f.min_confidence * 100)))
        self.dmin.setValue(f.min_distance_m or 0.0)
        self.dmax.setValue(f.max_distance_m or 0.0)
        self.text.setText(f.text)
        self.pre.setValue(pre)
        self.post.setValue(post)
        self.gap.setValue(gap)

    def _players(self) -> list[str]:
        i = self.player.currentIndex()
        txt = self.player.currentText().strip()
        if i > 0 and txt == self.player.itemText(i):
            return [self.player.itemData(i)]
        if not txt or txt == self.player.itemText(0):
            return []
        out = []
        for part in txt.split(","):
            j = self.player.findText(part.strip())
            k = self.player.itemData(j) if j > 0 else pl.parse_player(part)
            if k is None:
                raise ValueError(f"Unknown player “{part.strip()}”: use #12, A3 (slot) or T42 (track id).")
            out.append(k)
        return out

    def filter(self) -> pl.Filter:
        """The filter in the dialog. Raises ValueError for unreadable input."""
        cats = [self.cats.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.cats.count())
                if self.cats.item(i).checkState() == Qt.CheckState.Checked]
        try:
            t0, t1 = pl.parse_clock(self.t_from.text()), pl.parse_clock(self.t_to.text())
        except ValueError:
            raise ValueError("Write video times as m:ss (for example 12:30) or seconds.") from None
        return pl.Filter(categories=cats, team=self.team.currentData(), players=self._players(),
                         player_mode=self.player_mode.currentData(), period=self.period.currentData(), t_from=t0,
                         t_to=t1, min_confidence=self.conf.value() / 100.0, min_distance_m=self.dmin.value() or None,
                         max_distance_m=self.dmax.value() or None, text=self.text.text().strip())

    def values(self) -> tuple[str, pl.Filter, float, float, float]:
        f = self.filter()
        name = self.name.text().strip() or f.describe(self.cat, self.coached)[:80]
        return name, f, self.pre.value(), self.post.value(), self.gap.value()

    # ---------------------------------------------------------------- preview / accept
    def _changed(self, *_a: Any) -> None:
        if self._loading:
            return
        if self.sender() is not self.preset and self.preset.currentIndex() > 0 and \
                self.sender() not in (self.name, self.we):
            self.preset.blockSignals(True)
            self.preset.setCurrentIndex(0)  # edited: no longer the preset
            self.preset.blockSignals(False)
        self._timer.start()

    def _update_preview(self) -> None:
        try:
            _name, f, pre, post, gap = self.values()
        except ValueError as e:
            self.preview.setText(f"<span style='color:#b91c1c'>{e}</span>")
            return
        if not f.categories:
            self.preview.setText("<span style='color:#b45309'>Tick at least one thing to show.</span>")
            return
        items = pl.apply_filter(self.cat, f)
        clips = pl.build_clips(items, pre, post, gap, self.cat.duration, f)
        tot = sum(c.duration for c in clips)
        self.preview.setText(f"<b>{len(items)}</b> events → <b>{len(clips)}</b> clips, "
                             f"{pl.fmt_clock(tot, 0)} of video" + ("" if items else
                                                                    " — nothing matches this filter yet"))

    def _accept(self) -> None:
        try:
            _n, f, *_ = self.values()
        except ValueError as e:
            QMessageBox.warning(self, "Playlist", str(e))
            return
        if not f.categories:
            QMessageBox.warning(self, "Playlist", "Tick at least one thing to show (e.g. Turnovers).")
            return
        self.accept()


def _key_text(key: str) -> str:
    kind, _, rest = key.partition(":")
    if kind == "num":
        return "#" + rest.rpartition(":")[2]
    if kind == "slot":
        return rest
    if kind == "trk":
        return "T" + rest
    return key


# ====================================================================================== export dialog
class ExportDialog(QDialog):
    def __init__(self, parent: QWidget, playlist: pl.Playlist, has_audio: bool):
        super().__init__(parent)
        self.setWindowTitle("Export playlist as video")
        form = QFormLayout(self)
        form.addRow(QLabel(f"<b>{playlist.name}</b>: {len(playlist.clips)} clips, "
                           f"{pl.fmt_clock(playlist.total_s, 0)} of video"))
        self.cards = QCheckBox("Title card before each clip (and one for the playlist)")
        self.cards.setChecked(True)
        form.addRow(self.cards)
        self.card_s = QDoubleSpinBox()
        self.card_s.setRange(0.5, 10)
        self.card_s.setValue(2.0)
        self.card_s.setSuffix(" s")
        form.addRow("Title card length", self.card_s)
        self.cards.toggled.connect(self.card_s.setEnabled)
        self.unreviewed = QCheckBox("Only clips not marked as reviewed")
        self.unreviewed.setEnabled(playlist.reviewed_count > 0)
        form.addRow(self.unreviewed)
        self.quick = QCheckBox("Quick export (faster, about 3x bigger file)")
        form.addRow(self.quick)
        self.audio = QCheckBox("Keep the sound")
        self.audio.setChecked(has_audio)
        self.audio.setEnabled(has_audio)
        form.addRow(self.audio)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.button(QDialogButtonBox.StandardButton.Ok).setText("Choose file…")
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        form.addRow(bb)

    def options(self) -> pl.ExportOptions:
        return pl.ExportOptions(title_cards=self.cards.isChecked(), card_s=self.card_s.value(),
                                intro_card=self.cards.isChecked(), only_unreviewed=self.unreviewed.isChecked(),
                                audio=self.audio.isChecked(), quick=self.quick.isChecked())


# ====================================================================================== panel
class PlaylistPanel(QWidget):
    def __init__(self, win: "MainWindow"):
        super().__init__()
        self.win = win
        self.p: "Project | None" = None
        self.book = pl.PlaylistBook()
        self._cat: pl.Catalog | None = None
        self._cat_key: tuple | None = None
        self.playing: int | None = None  # index of the clip being played
        self.play_all = False
        self._seek_at = 0.0
        self._export: dict | None = None

        v = QVBoxLayout(self)
        v.setContentsMargins(6, 6, 6, 6)
        top = QHBoxLayout()
        self.combo = QComboBox()
        self.combo.setMinimumWidth(160)
        self.combo.currentIndexChanged.connect(self._playlist_changed)
        top.addWidget(self.combo, 1)
        self.b_new = QPushButton("New…")
        self.b_new.setToolTip("Build a playlist from a filter: event type, team, player, period, ...")
        self.b_new.clicked.connect(lambda: self.new_playlist())
        top.addWidget(self.b_new)
        self.b_menu = QPushButton("⋯")
        self.b_menu.setFixedWidth(30)
        self.b_menu.setToolTip("Edit filter, rebuild, rename, delete")
        self.b_menu.clicked.connect(self._playlist_menu)
        top.addWidget(self.b_menu)
        v.addLayout(top)
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        self.summary.setStyleSheet("color:#475569")
        v.addWidget(self.summary)

        self.tree = QTreeWidget()
        self.tree.setColumnCount(7)
        self.tree.setHeaderLabels(["✓", "#", "Start", "Len", "Team", "Clip", "Note"])
        self.tree.setRootIsDecorated(False)
        self.tree.setAlternatingRowColors(True)
        self.tree.setUniformRowHeights(True)
        for c, w in ((COL_DONE, 34), (COL_NUM, 30), (COL_START, 58), (COL_LEN, 44), (COL_TEAM, 70), (COL_TITLE, 230)):
            self.tree.setColumnWidth(c, w)
        self.tree.itemDoubleClicked.connect(self._double_clicked)
        self.tree.itemChanged.connect(self._item_changed)
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._clip_menu)
        v.addWidget(self.tree, 1)

        tr = QHBoxLayout()
        self.b_prev = QPushButton("⏮ Prev")
        self.b_play = QPushButton("▶ Play all")
        self.b_next = QPushButton("Next ⏭")
        self.b_stop = QPushButton("■")
        self.b_stop.setFixedWidth(32)
        self.b_prev.clicked.connect(lambda: self.step(-1))
        self.b_next.clicked.connect(lambda: self.step(1))
        self.b_play.clicked.connect(self.play_all_clicked)
        self.b_stop.clicked.connect(self.stop)
        for b, tip in ((self.b_prev, "Previous clip"), (self.b_play, "Play the clips back to back from the selected "
                                                                      "one"), (self.b_next, "Next clip"),
                       (self.b_stop, "Stop following the playlist")):
            b.setToolTip(tip)
            tr.addWidget(b)
        v.addLayout(tr)
        ed = QHBoxLayout()
        self.b_done = QPushButton("✓ Reviewed")
        self.b_done.setToolTip("Mark the selected clip as reviewed (or not)")
        self.b_done.clicked.connect(self.toggle_reviewed)
        self.b_note = QPushButton("Note…")
        self.b_note.clicked.connect(lambda: self.edit_note())
        self.b_remove = QPushButton("Remove")
        self.b_remove.setToolTip("Remove the selected clip from this playlist")
        self.b_remove.clicked.connect(self.remove_clip)
        for b in (self.b_done, self.b_note, self.b_remove):
            ed.addWidget(b)
        v.addLayout(ed)
        self.auto_review = QCheckBox("Mark clips reviewed when played to the end")
        v.addWidget(self.auto_review)
        ex = QHBoxLayout()
        self.b_mp4 = QPushButton("Export MP4…")
        self.b_mp4.setToolTip("One video file with all the clips (optional title cards)")
        self.b_mp4.clicked.connect(lambda: self.export_mp4())
        self.b_csv = QPushButton("Export CSV…")
        self.b_csv.setToolTip("Table of the clips' times, titles, reviewed flags and notes")
        self.b_csv.clicked.connect(lambda: self.export_csv())
        ex.addWidget(self.b_mp4)
        ex.addWidget(self.b_csv)
        v.addLayout(ex)
        self.status = QLabel()
        self.status.setWordWrap(True)
        v.addWidget(self.status)

        self.watch = QTimer(self)
        self.watch.setInterval(500)
        self.watch.timeout.connect(self._watch_project)
        self.watch.start()
        self.play_timer = QTimer(self)
        self.play_timer.setInterval(40)
        self.play_timer.timeout.connect(self._follow)
        self._watch_project()

    # ---------------------------------------------------------------- project / data
    def _watch_project(self) -> None:
        cur = getattr(self.win, "current", None)
        if (cur.root if cur else None) != (self.p.root if self.p else None):
            self.set_project(cur)

    def set_project(self, p: "Project | None") -> None:
        self.stop()
        self.p = p
        self._cat, self._cat_key = None, None
        self.book = pl.PlaylistBook.load(p.root) if p is not None else pl.PlaylistBook()
        self._fill_combo()
        self._update_buttons()

    def _cat_stamp(self) -> tuple:
        root = self.p.root
        out = []
        for f in (root / "results" / "replay_data.json", root / "results" / "game_state.json",
                  root / "timeline.json", root / "results" / "ai_analysis.json"):
            try:
                st = f.stat()
                out.append((st.st_mtime_ns, st.st_size))
            except OSError:
                out.append(None)
        return (str(root), *out)

    def catalog(self) -> pl.Catalog | None:
        """Every taggable moment of the current Replay (cached until one of its source files changes)."""
        if self.p is None:
            return None
        key = self._cat_stamp()
        if self._cat is not None and key == self._cat_key:
            return self._cat
        rt = getattr(self.win, "replay_tab", None)
        replay = rt.data if rt is not None and getattr(rt, "p", None) is not None and rt.p.root == self.p.root \
            else None
        dur = (self.p.info().get("video_info") or {}).get("duration_s")
        self._cat = pl.collect(self.p.root, replay=replay, duration=float(dur) if dur else None)
        self._cat_key = self._cat_stamp()  # building the game state may have written game_state.json
        return self._cat

    def coached(self) -> str | None:
        return _settings_coached(self.p)

    def names(self) -> dict[str, str]:
        cat = self._cat
        if cat is not None:
            return cat.names
        rt = getattr(self.win, "replay_tab", None)
        return pl.team_names(getattr(rt, "data", None) if rt is not None else None)

    def save(self) -> None:
        if self.p is not None:
            self.book.save(self.p.root)

    # ---------------------------------------------------------------- playlists
    def current(self) -> pl.Playlist | None:
        pid = self.combo.currentData()
        return self.book.get(pid) if pid else None

    def _fill_combo(self, select: str | None = None) -> None:
        cur = select or self.combo.currentData()
        self.combo.blockSignals(True)
        self.combo.clear()
        for p in self.book.playlists:
            self.combo.addItem(f"{p.name}  ({len(p.clips)})", p.id)
        if not self.book.playlists:
            self.combo.addItem("No playlists yet – click New…", None)
        i = self.combo.findData(cur) if cur else -1
        self.combo.setCurrentIndex(i if i >= 0 else 0)
        self.combo.blockSignals(False)
        self._playlist_changed()

    def _playlist_changed(self, *_a: Any) -> None:
        self.stop()
        self.render()

    def render(self) -> None:
        p = self.current()
        names, coached = self.names(), self.coached()
        self.tree.blockSignals(True)
        self.tree.clear()
        if p is None:
            self.summary.setText("" if self.p is None else
                                 "Pick what you want to see – e.g. <i>all our turnovers</i>, <i>all their fast "
                                 "breaks</i>, <i>every possession of #12</i> – and watch the clips back to back.")
        else:
            self.summary.setText(self._summary_text(p))
            for i, c in enumerate(p.clips):
                it = QTreeWidgetItem(["", str(i + 1), pl.fmt_clock(c.t0, 0), f"{c.duration:.0f}s",
                                      pl.team_label(c.team, coached, names, short=True) if c.team else "", c.title,
                                      c.note.replace("\n", " ")])
                it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                it.setCheckState(COL_DONE, Qt.CheckState.Checked if c.reviewed else Qt.CheckState.Unchecked)
                it.setData(COL_DONE, Qt.ItemDataRole.UserRole, c.id)
                tip = "\n".join(dict.fromkeys(c.titles)) + (f"\n\nNote: {c.note}" if c.note else "")
                for col in range(7):
                    it.setToolTip(col, tip)
                self.tree.addTopLevelItem(it)
        self.tree.blockSignals(False)
        self._highlight()
        self._update_buttons()

    def _update_buttons(self) -> None:
        has_p = self.p is not None
        p = self.current()
        clips = bool(p and p.clips)
        self.b_new.setEnabled(has_p)
        self.b_menu.setEnabled(p is not None)
        for b in (self.b_prev, self.b_play, self.b_next, self.b_done, self.b_note, self.b_remove, self.b_csv):
            b.setEnabled(clips)
        self.b_mp4.setEnabled(clips and self._export is None and has_p and self.p.video.exists())
        self.b_stop.setEnabled(self.playing is not None)
        self.b_play.setText("⏸ Pause" if self.playing is not None and self._player_playing() else "▶ Play all")

    def new_playlist(self, preset: int | None = None) -> pl.Playlist | None:
        if self.p is None:
            return None
        cat = self.catalog()
        dlg = FilterDialog(self, cat, self.coached(), preset=preset)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return None
        return self._create_from_dialog(dlg)

    def _create_from_dialog(self, dlg: FilterDialog) -> pl.Playlist:
        if dlg.coached != self.coached():
            save_coached(self.p, dlg.coached)
        name, f, pre, post, gap = dlg.values()
        new = self.book.add(pl.make_playlist(self.catalog(), name, f, pre, post, gap))
        self.save()
        self._fill_combo(new.id)
        self.status.setText(f"Playlist “{new.name}”: {len(new.clips)} clips.")
        return new

    def add_playlist(self, playlist: pl.Playlist, show: bool = True) -> pl.Playlist:
        """Add a playlist made elsewhere (e.g. the Stats tab's "Send to playlist"), save it and select it."""
        self.book.add(playlist)
        self.save()
        self._fill_combo(playlist.id)
        if show:
            dock = self.parentWidget()
            if isinstance(dock, QDockWidget):
                dock.show()
                dock.raise_()
        self.status.setText(f"Playlist “{playlist.name}”: {len(playlist.clips)} clips.")
        return playlist

    def edit_playlist(self) -> None:
        p = self.current()
        if p is None or p.fixed:
            return
        dlg = FilterDialog(self, self.catalog(), self.coached(), playlist=p)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        if dlg.coached != self.coached():
            save_coached(self.p, dlg.coached)
        p.name, p.filter, p.pre_s, p.post_s, p.merge_gap_s = dlg.values()
        p.rebuild(self.catalog())
        self.save()
        self._fill_combo(p.id)

    def rebuild(self) -> None:
        p = self.current()
        if p is None or p.fixed:
            return
        self._cat_key = None  # re-read everything
        a, b = p.rebuild(self.catalog())
        self.save()
        self._fill_combo(p.id)
        self.status.setText(f"Rebuilt from the latest analysis: {a} → {b} clips (reviewed flags and notes kept).")

    def rename(self) -> None:
        p = self.current()
        if p is None:
            return
        name, ok = QInputDialog.getText(self, "Rename playlist", "Name:", QLineEdit.EchoMode.Normal, p.name)
        if ok and name.strip():
            p.name = name.strip()[:100]
            self.save()
            self._fill_combo(p.id)

    def delete(self) -> None:
        p = self.current()
        if p is None:
            return
        if QMessageBox.question(self, "Delete playlist", f"Delete the playlist “{p.name}”? (The video and the "
                                "events are not touched.)") != QMessageBox.StandardButton.Yes:
            return
        self.book.remove(p.id)
        self.save()
        self._fill_combo()

    def _playlist_menu(self) -> None:
        m = QMenu(self)
        fixed = bool(self.current() and self.current().fixed)
        m.addAction("Edit filter…", self.edit_playlist).setEnabled(not fixed)
        m.addAction("Rebuild from latest analysis", self.rebuild).setEnabled(not fixed)
        m.addAction("Rename…", self.rename)
        m.addSeparator()
        m.addAction("Delete playlist", self.delete)
        m.exec(self.b_menu.mapToGlobal(self.b_menu.rect().bottomLeft()))

    # ---------------------------------------------------------------- clips
    def selected_index(self) -> int:
        it = self.tree.currentItem()
        return self.tree.indexOfTopLevelItem(it) if it is not None else -1

    def _item_changed(self, it: QTreeWidgetItem, col: int) -> None:
        p = self.current()
        if p is None or col != COL_DONE:
            return
        i = self.tree.indexOfTopLevelItem(it)
        if 0 <= i < len(p.clips):
            p.clips[i].reviewed = it.checkState(COL_DONE) == Qt.CheckState.Checked
            self.save()
            self._refresh_summary()

    def _refresh_summary(self) -> None:
        p = self.current()
        if p is not None:
            self.summary.setText(self._summary_text(p))

    def _summary_text(self, p: pl.Playlist) -> str:
        what = (f"Fixed clip list{' from the ' + p.origin.capitalize() + ' tab' if p.origin else ''}" if p.fixed
                else p.filter.describe(self._cat, self.coached()))
        return (f"{what}<br>{len(p.clips)} clips · {pl.fmt_clock(p.total_s, 0)} · {p.reviewed_count} reviewed · "
                f"{p.pre_s:g} s before / {p.post_s:g} s after")

    def set_reviewed(self, i: int, on: bool) -> None:
        p = self.current()
        if p is None or not 0 <= i < len(p.clips):
            return
        p.clips[i].reviewed = on
        it = self.tree.topLevelItem(i)
        if it is not None:
            self.tree.blockSignals(True)
            it.setCheckState(COL_DONE, Qt.CheckState.Checked if on else Qt.CheckState.Unchecked)
            self.tree.blockSignals(False)
        self.save()
        self._refresh_summary()

    def toggle_reviewed(self) -> None:
        p = self.current()
        i = self.selected_index()
        if p is not None and 0 <= i < len(p.clips):
            self.set_reviewed(i, not p.clips[i].reviewed)

    def edit_note(self, i: int | None = None) -> None:
        p = self.current()
        i = self.selected_index() if i is None else i
        if p is None or not 0 <= i < len(p.clips):
            return
        c = p.clips[i]
        txt, ok = QInputDialog.getMultiLineText(self, "Clip note", f"{c.title}  ({pl.fmt_clock(c.t0, 0)})", c.note)
        if ok:
            self.set_note(i, txt)

    def set_note(self, i: int, txt: str) -> None:
        p = self.current()
        if p is None or not 0 <= i < len(p.clips):
            return
        p.clips[i].note = txt.strip()
        self.save()
        it = self.tree.topLevelItem(i)
        if it is not None:
            it.setText(COL_NOTE, p.clips[i].note.replace("\n", " "))

    def remove_clip(self) -> None:
        p = self.current()
        i = self.selected_index()
        if p is None or not 0 <= i < len(p.clips):
            return
        if self.playing is not None:
            self.stop()
        del p.clips[i]
        self.save()
        self._fill_combo(p.id)
        if p.clips:
            self.tree.setCurrentItem(self.tree.topLevelItem(min(i, len(p.clips) - 1)))

    def _double_clicked(self, it: QTreeWidgetItem, col: int) -> None:
        i = self.tree.indexOfTopLevelItem(it)
        if col == COL_NOTE:
            self.edit_note(i)
        else:
            self.play(i, play_all=False)

    def _clip_menu(self, pos) -> None:
        it = self.tree.itemAt(pos)
        if it is None:
            return
        i = self.tree.indexOfTopLevelItem(it)
        p = self.current()
        m = QMenu(self)
        m.addAction("Play this clip", lambda: self.play(i, play_all=False))
        m.addAction("Play all from here", lambda: self.play(i, play_all=True))
        m.addAction("Mark as not reviewed" if p and p.clips[i].reviewed else "Mark as reviewed",
                    lambda: self.set_reviewed(i, not p.clips[i].reviewed))
        m.addAction("Note…", lambda: self.edit_note(i))
        m.addSeparator()
        m.addAction("Remove from playlist", lambda: (self.tree.setCurrentItem(it), self.remove_clip()))
        m.exec(self.tree.viewport().mapToGlobal(pos))

    # ---------------------------------------------------------------- playback (Replay tab's player)
    def _rt(self):
        return getattr(self.win, "replay_tab", None)

    def _player_playing(self) -> bool:
        rt = self._rt()
        return rt is not None and rt.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState

    def play(self, i: int, play_all: bool = True) -> bool:
        p, rt = self.current(), self._rt()
        if p is None or rt is None or not 0 <= i < len(p.clips) or self.p is None:
            return False
        if getattr(rt, "p", None) is None or rt.p.root != self.p.root:
            rt.load(self.p)
        tabs = getattr(self.win, "tabs", None)
        if tabs is not None and tabs.currentWidget() is not rt:
            tabs.setCurrentWidget(rt)
        self.playing, self.play_all = i, play_all
        c = p.clips[i]
        rt.seek_to(c.t0)
        self._seek_at = time.perf_counter()
        rt.player.play()
        self.play_timer.start()
        self.tree.blockSignals(True)
        self.tree.setCurrentItem(self.tree.topLevelItem(i))
        self.tree.blockSignals(False)
        self._highlight()
        self.status.setText(f"Clip {i + 1} / {len(p.clips)}: {c.title}")
        self._update_buttons()
        return True

    def play_all_clicked(self) -> None:
        if self.playing is not None:
            if self._player_playing():
                self._rt().player.pause()
            else:
                self._rt().player.play()
            self._update_buttons()
            return
        i = self.selected_index()
        self.play(i if i >= 0 else 0, play_all=True)

    def step(self, d: int) -> None:
        p = self.current()
        if p is None or not p.clips:
            return
        base = self.playing if self.playing is not None else self.selected_index()
        i = (0 if d > 0 else len(p.clips) - 1) if base is None or base < 0 else base + d
        if 0 <= i < len(p.clips):
            self.play(i, play_all=self.play_all if self.playing is not None else False)

    def stop(self) -> None:
        was = self.playing is not None
        self.playing = None
        self.play_timer.stop()
        if was:
            rt = self._rt()
            if rt is not None:
                rt.player.pause()
            self.status.setText("Stopped.")
        self._highlight()
        self._update_buttons()

    def _follow(self) -> None:
        """Advance to the next clip at the end of the current one; let go when the user seeks elsewhere."""
        p, rt = self.current(), self._rt()
        if self.playing is None or p is None or rt is None or self.playing >= len(p.clips):
            self.stop()
            return
        if time.perf_counter() - self._seek_at < END_GRACE_S:
            return
        c = p.clips[self.playing]
        t = rt.player.position() / 1000.0
        dur = rt.player.duration() / 1000.0
        if t < c.t0 - 1.0 or t > c.t1 + 1.5:  # the user moved the playhead: stop following the playlist
            self.playing = None
            self.play_timer.stop()
            self.status.setText("Left the playlist (you moved the playhead). Double-click a clip to go back.")
            self._highlight()
            self._update_buttons()
            return
        ended = t >= c.t1 - 0.02 or (dur > 0 and t >= dur - 0.1 and not self._player_playing())
        if not ended:
            if self._ui_tick():
                self._update_buttons()
            return
        if self.auto_review.isChecked():
            self.set_reviewed(self.playing, True)
        if self.play_all and self.playing + 1 < len(p.clips):
            self.play(self.playing + 1, play_all=True)
            return
        rt.player.pause()
        n = len(p.clips)
        self.playing = None
        self.play_timer.stop()
        self.status.setText("End of the playlist." if self.play_all else
                            f"End of the clip ({self.selected_index() + 1} / {n}). Next ⏭ for the next one.")
        self._highlight()
        self._update_buttons()

    _tick = 0

    def _ui_tick(self) -> bool:
        self._tick += 1
        return self._tick % 6 == 0

    def _highlight(self) -> None:
        for i in range(self.tree.topLevelItemCount()):
            it = self.tree.topLevelItem(i)
            on = i == self.playing
            for col in range(7):
                it.setBackground(col, PLAYING_BG if on else QColor(0, 0, 0, 0))
            f = it.font(COL_TITLE)
            f.setBold(on)
            it.setFont(COL_TITLE, f)

    # ---------------------------------------------------------------- export
    def _default_name(self, ext: str) -> str:
        p = self.current()
        rname = self.p.info().get("name", self.p.id) if self.p else "replay"
        base = f"{rname} - {p.name if p else 'playlist'}"
        safe = "".join(ch if ch.isalnum() or ch in " -_.,()äöåÄÖÅ" else "_" for ch in base).strip()
        return str(Path.home() / f"{safe[:120]}{ext}")

    def export_csv(self, path: str | None = None) -> Path | None:
        p = self.current()
        if p is None or self.p is None:
            return None
        if path is None:
            path, _ = QFileDialog.getSaveFileName(self, "Export playlist as CSV", self._default_name(".csv"),
                                                  "CSV (*.csv)")
            if not path:
                return None
        out = pl.export_csv(p, Path(path), self.names(), self.coached())
        self.status.setText(f"Saved {out.name} ({len(p.clips)} clips).")
        return out

    def export_mp4(self, path: str | None = None, opts: pl.ExportOptions | None = None,
                   wait: bool = False) -> Path | None:
        """Export in the background (progress + cancel in a dialog). ``wait`` blocks until done (tests)."""
        p = self.current()
        if p is None or self.p is None or self._export is not None:
            return None
        vi = self.p.info().get("video_info") or {}
        has_audio = bool(vi.get("has_audio"))
        if opts is None:
            dlg = ExportDialog(self, p, has_audio)
            if dlg.exec() != QDialog.DialogCode.Accepted:
                return None
            opts = dlg.options()
        if path is None:
            path, _ = QFileDialog.getSaveFileName(self, "Export playlist as video", self._default_name(".mp4"),
                                                  "MP4 video (*.mp4)")
            if not path:
                return None
        size = tuple(int(v) for v in (vi.get("size") or (1280, 720)))[:2]
        fps = float(vi.get("fps") or 30.0)
        snapshot = pl.Playlist.from_dict(p.to_dict())  # the user may keep editing while it runs
        cancel = threading.Event()
        st: dict = {"p": 0.0, "done": False, "error": None, "out": None, "cancel": cancel}
        names, coached = self.names(), self.coached()
        rname = self.p.info().get("name", self.p.id)
        video = self.p.video

        def work() -> None:
            try:
                st["out"] = pl.export_mp4(snapshot, video, Path(path), fps=fps, size=size,  # type: ignore[arg-type]
                                          has_audio=has_audio, opts=opts, names=names, coached=coached,
                                          replay_name=rname, progress=lambda x: st.__setitem__("p", x),
                                          cancel=cancel)
            except Exception as e:  # noqa: BLE001 - shown to the user
                st["error"] = e
            st["done"] = True

        self._export = st
        th = threading.Thread(target=work, daemon=True)
        th.start()
        self._update_buttons()
        if wait:
            th.join()
            self._export_finished()
            return st["out"]
        dlg = QProgressDialog(f"Exporting “{p.name}” ({len(snapshot.clips)} clips)…", "Cancel", 0, 1000, self)
        dlg.setWindowTitle("Export playlist")
        dlg.setMinimumDuration(0)
        dlg.setAutoClose(False)
        dlg.canceled.connect(cancel.set)
        st["dialog"] = dlg
        timer = QTimer(self)
        timer.setInterval(100)

        def poll() -> None:
            dlg.setValue(int(st["p"] * 1000))
            if st["done"]:
                timer.stop()
                dlg.close()
                self._export_finished()

        timer.timeout.connect(poll)
        timer.start()
        dlg.show()
        return None

    def _export_finished(self) -> None:
        st, self._export = self._export, None
        self._update_buttons()
        if st is None:
            return
        err = st.get("error")
        if isinstance(err, pl.ExportCancelled):
            self.status.setText("Export cancelled.")
        elif err is not None:
            self.status.setText(f"<span style='color:#b91c1c'>Export failed: {err}</span>")
            if st.get("dialog") is not None:
                QMessageBox.warning(self, "Export playlist", f"The export failed:\n{err}")
        elif st.get("out") is not None:
            out = Path(st["out"])
            mb = out.stat().st_size / 1e6 if out.exists() else 0
            self.status.setText(f"Saved {out.name} ({mb:.1f} MB).")


# ====================================================================================== main-window hook
def install(win: "MainWindow") -> QDockWidget:
    """Add the Playlists dock panel and menu to the main window (call once, after the menus exist)."""
    panel = PlaylistPanel(win)
    dock = QDockWidget("Playlists", win)
    dock.setObjectName("PlaylistsDock")
    dock.setWidget(panel)
    dock.setAllowedAreas(Qt.DockWidgetArea.RightDockWidgetArea | Qt.DockWidgetArea.LeftDockWidgetArea |
                         Qt.DockWidgetArea.BottomDockWidgetArea)
    dock.setMinimumWidth(340)
    win.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
    dock.hide()

    mb = win.menuBar()
    menu = QMenu("&Playlists", win)
    help_action = next((a for a in mb.actions() if a.text().replace("&", "") == "Help"), None)
    if help_action is not None:
        mb.insertMenu(help_action, menu)
    else:
        mb.addMenu(menu)
    toggle = dock.toggleViewAction()
    toggle.setText("Show playlists panel")
    toggle.setShortcut(QKeySequence("Ctrl+Shift+L"))
    menu.addAction(toggle)

    def new(preset: int | None = None) -> None:
        if win.current is None:
            QMessageBox.information(win, "Playlists", "Select a Replay first.")
            return
        dock.show()
        dock.raise_()
        panel._watch_project()
        panel.new_playlist(preset)

    a_new = QAction("New playlist…", win)
    a_new.triggered.connect(lambda: new())
    menu.addAction(a_new)
    quick = menu.addMenu("Quick playlist")

    def fill_quick() -> None:
        quick.clear()
        cur = win.current
        names = pl.team_names(getattr(win.replay_tab, "data", None)) if cur is not None else {k: k for k in pl.TEAMS}
        for i, (name, *_rest) in enumerate(pl.presets(_settings_coached(cur), names)):
            act = quick.addAction(name + "…")
            act.triggered.connect(lambda _=False, k=i: new(k))
            act.setEnabled(cur is not None)

    quick.aboutToShow.connect(fill_quick)
    fill_quick()
    win.playlists_panel = panel  # type: ignore[attr-defined]
    win.playlists_dock = dock  # type: ignore[attr-defined]
    return dock
