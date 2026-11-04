"""The Replay list on the left of the main window: search, sort, status icons, disk size, more actions.

install(win) upgrades the existing ``win.list`` (a QListWidget whose items carry the Replay id in UserRole - the
rest of the app keeps using it unchanged):
  * a search box (name, team names, file name) and a sort choice (newest / oldest / name / size), remembered;
  * each row painted as: status icon · name (+ pin) · status line · disk size on the right;
  * status icons: analysing (blue), ready (green tick), needs attention (amber), error (red), idle (grey);
  * right-click menu: Rename, Duplicate, Open folder, Pin, Export, AI, Delete.
Disk sizes are measured in a background thread and cached (a long game's analysis folder has many files).
"""

from __future__ import annotations

import os
import sys
import threading
import time
from typing import TYPE_CHECKING

from PySide6.QtCore import QRect, QSize, Qt
from PySide6.QtGui import QColor, QFont, QIcon, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (QComboBox, QHBoxLayout, QInputDialog, QLineEdit, QListWidget, QListWidgetItem, QMenu,
                               QStyle, QStyledItemDelegate, QStyleOptionViewItem, QWidget)

from replay.gui.projects import Project, folder_size

if TYPE_CHECKING:  # pragma: no cover
    from replay.gui.app import MainWindow

SIZE_ROLE = Qt.ItemDataRole.UserRole + 1
STATUS_ROLE = Qt.ItemDataRole.UserRole + 2
KIND_ROLE = Qt.ItemDataRole.UserRole + 3
NAME_ROLE = Qt.ItemDataRole.UserRole + 4
PIN_ROLE = Qt.ItemDataRole.UserRole + 5

SORTS = [("newest", "Newest first"), ("oldest", "Oldest first"), ("name", "Name A–Z"), ("size", "Largest first")]

KIND_COLOURS = {"busy": "#2563eb", "ready": "#16a34a", "attention": "#d97706", "error": "#dc2626", "idle": "#9ca3af"}
KIND_TIPS = {"busy": "Working on it (importing, analysing or building) - you can keep using the app",
             "ready": "Ready - open the Replay tab or the coach report",
             "attention": "Needs you - see the status line (for example: fix the court, or press Build)",
             "error": "Something went wrong - select the Replay to see what to do",
             "idle": "Waiting - select the Replay and press the button at the top to continue"}


# ---------------------------------------------------------------------------------------------- pure helpers
def status_kind(status_line: str, job: dict) -> str:
    """Icon group for a Replay: busy / ready / attention / error / idle (from the status text and job state)."""
    if job.get("error"):
        return "error"
    if job.get("kind") or job.get("queued"):
        return "busy"
    s = status_line.lower()
    if s.startswith("replay ready"):
        return "attention" if "outdated" in s else "ready"
    if "court not found" in s or "not built" in s:
        return "attention"
    return "idle"


def fmt_size(n: int | None) -> str:
    if n is None:
        return "…"
    for unit, k in (("GB", 1 << 30), ("MB", 1 << 20), ("KB", 1 << 10)):
        if n >= k:
            v = n / k
            return f"{v:.1f} {unit}" if v < 100 else f"{v:.0f} {unit}"
    return f"{n} B"


def search_text(p: Project) -> str:
    info = p.info()
    src = info.get("source", {}) or {}
    try:
        meta = p.meta()
    except Exception:  # noqa: BLE001 - a broken meta.json must not break the list
        meta = {}
    parts = [info.get("name", ""), p.id, src.get("filename", ""), meta.get("home_team", ""),
             meta.get("away_team", ""), meta.get("date", "") or "", meta.get("location", "") or ""]
    return " ".join(str(x) for x in parts).lower()


def matches(p: Project, query: str) -> bool:
    words = query.lower().split()
    if not words:
        return True
    hay = search_text(p)
    return all(w in hay for w in words)


def order(projects: list[Project], sort: str, sizes: dict[str, int | None]) -> list[Project]:
    """Pinned first, then by the chosen sort (stable)."""
    if sort == "name":
        key = lambda p: str(p.info().get("name", p.id)).lower()  # noqa: E731
        out = sorted(projects, key=key)
    elif sort == "oldest":
        out = sorted(projects, key=lambda p: p.info().get("created", 0) or 0)
    elif sort == "size":
        out = sorted(projects, key=lambda p: -(sizes.get(p.id) or 0))
    else:
        out = sorted(projects, key=lambda p: p.info().get("created", 0) or 0, reverse=True)
    return sorted(out, key=lambda p: not p.info().get("pinned", False))


class SizeCache:
    """Folder sizes measured off the GUI thread; refreshed after ``ttl`` seconds (sooner for busy Replays)."""

    def __init__(self, ttl: float = 120.0, busy_ttl: float = 20.0):
        self.ttl, self.busy_ttl = ttl, busy_ttl
        self._val: dict[str, tuple[float, int]] = {}
        self._pending: set[str] = set()
        self._lock = threading.Lock()

    def get(self, p: Project, busy: bool = False) -> int | None:
        now = time.time()
        with self._lock:
            hit = self._val.get(p.id)
            stale = hit is None or now - hit[0] > (self.busy_ttl if busy else self.ttl)
            if stale and p.id not in self._pending:
                self._pending.add(p.id)
                threading.Thread(target=self._measure, args=(p.id, p.root), daemon=True).start()
        return hit[1] if hit else None

    def invalidate(self, pid: str) -> None:
        with self._lock:
            self._val.pop(pid, None)

    def _measure(self, pid: str, root) -> None:
        try:
            n = folder_size(root)
        except OSError:
            n = 0
        with self._lock:
            self._val[pid] = (time.time(), n)
            self._pending.discard(pid)

    def snapshot(self) -> dict[str, int | None]:
        with self._lock:
            return {k: v[1] for k, v in self._val.items()}


# ---------------------------------------------------------------------------------------------- icons + delegate
_ICON_CACHE: dict[str, QIcon] = {}


def status_icon(kind: str) -> QIcon:
    """Small round status badge drawn at runtime (no image files to ship)."""
    if kind in _ICON_CACHE:
        return _ICON_CACHE[kind]
    pm = QPixmap(18, 18)
    pm.fill(Qt.GlobalColor.transparent)
    qp = QPainter(pm)
    qp.setRenderHint(QPainter.RenderHint.Antialiasing)
    c = QColor(KIND_COLOURS.get(kind, "#9ca3af"))
    qp.setPen(Qt.PenStyle.NoPen)
    qp.setBrush(c)
    qp.drawEllipse(1, 1, 16, 16)
    qp.setPen(QPen(QColor("white"), 2.2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
    qp.setBrush(Qt.BrushStyle.NoBrush)
    if kind == "ready":
        path = QPainterPath()
        path.moveTo(5, 9.5)
        path.lineTo(8, 12.5)
        path.lineTo(13, 6)
        qp.drawPath(path)
    elif kind in ("error", "attention"):
        qp.drawLine(9, 5, 9, 10)
        qp.drawPoint(9, 13)
    elif kind == "busy":  # three dots = working
        for x in (5, 9, 13):
            qp.drawPoint(x, 9)
    else:  # idle: pause bars
        qp.drawLine(7, 6, 7, 12)
        qp.drawLine(11, 6, 11, 12)
    qp.end()
    _ICON_CACHE[kind] = QIcon(pm)
    return _ICON_CACHE[kind]


class ReplayRowDelegate(QStyledItemDelegate):
    """Row = status icon | bold name (+ pin) / grey status line | disk size (right-aligned 'column')."""

    def sizeHint(self, option: QStyleOptionViewItem, index) -> QSize:  # noqa: N802 - Qt API
        return QSize(200, 44)

    def paint(self, qp: QPainter, option: QStyleOptionViewItem, index) -> None:
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        opt.text = ""
        opt.icon = QIcon()
        style = opt.widget.style() if opt.widget else None
        if style is not None:
            style.drawControl(QStyle.ControlElement.CE_ItemViewItem, opt, qp, opt.widget)
        r = option.rect
        selected = bool(option.state & QStyle.StateFlag.State_Selected)
        fg = option.palette.highlightedText().color() if selected else option.palette.text().color()
        grey = fg if selected else QColor("#6b7280")
        kind = index.data(KIND_ROLE) or "idle"
        status_icon(kind).paint(qp, QRect(r.left() + 6, r.top() + 13, 18, 18))
        size_txt = index.data(SIZE_ROLE) or ""
        qp.save()
        f = QFont(option.font)
        f.setPointSizeF(max(7.0, f.pointSizeF() - 1))
        qp.setFont(f)
        qp.setPen(grey)
        size_w = qp.fontMetrics().horizontalAdvance("9999 MB") + 4
        size_rect = QRect(r.right() - size_w - 6, r.top(), size_w, r.height())
        qp.drawText(size_rect, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, size_txt)
        text_left = r.left() + 30
        text_w = size_rect.left() - text_left - 6
        qp.drawText(QRect(text_left, r.top() + 22, text_w, 18), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                    qp.fontMetrics().elidedText(index.data(STATUS_ROLE) or "", Qt.TextElideMode.ElideRight, text_w))
        fb = QFont(option.font)
        fb.setBold(True)
        qp.setFont(fb)
        qp.setPen(fg)
        name = ("📌 " if index.data(PIN_ROLE) else "") + (index.data(NAME_ROLE) or "")
        qp.drawText(QRect(text_left, r.top() + 4, text_w, 18), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                    qp.fontMetrics().elidedText(name, Qt.TextElideMode.ElideRight, text_w))
        qp.restore()


# ---------------------------------------------------------------------------------------------- the panel
class ReplayListController:
    def __init__(self, win: "MainWindow"):
        from replay.gui.app import fmt_dur, status_text  # app imports this module lazily (no import cycle)

        self.win = win
        self._status_text, self._fmt_dur = status_text, fmt_dur
        self.sizes = SizeCache()
        lst: QListWidget = win.list
        self.list = lst
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search Replays…")
        self.search.setClearButtonEnabled(True)
        self.search.setToolTip("Type part of a Replay name, a team name, the video file name or the game date")
        self.sort = QComboBox()
        for key, label in SORTS:
            self.sort.addItem(label, key)
        self.sort.setToolTip("Order of the Replays (pinned Replays always stay on top)")
        saved = win.settings.value("replay_sort", "newest")
        i = self.sort.findData(saved)
        self.sort.setCurrentIndex(i if i >= 0 else 0)
        bar = QWidget()
        h = QHBoxLayout(bar)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(self.search, 1)
        h.addWidget(self.sort)
        lay = lst.parentWidget().layout()
        lay.insertWidget(lay.indexOf(lst), bar)
        lst.setItemDelegate(ReplayRowDelegate(lst))
        lst.setUniformItemSizes(True)
        lst.setToolTip("")
        self.search.textChanged.connect(lambda _t: self.refresh(force=True))
        self.sort.currentIndexChanged.connect(self._sort_changed)
        lst.customContextMenuRequested.disconnect()
        lst.customContextMenuRequested.connect(self.menu)
        win.refresh_list = self.refresh  # every caller (poll, rename, delete, …) now uses this

    def _sort_changed(self, _i: int) -> None:
        self.win.settings.setValue("replay_sort", self.sort.currentData())
        self.refresh(force=True)

    def refresh(self, force: bool = False) -> None:
        win, lst = self.win, self.list
        cur = win.current.id if win.current else None
        projects = win.store.list()
        jobs = {p.id: win.jobs.state(p.id).public() for p in projects}
        sizes = {p.id: self.sizes.get(p, busy=bool(jobs[p.id]["kind"])) for p in projects}
        projects = order(projects, self.sort.currentData() or "newest", sizes)
        ids = [lst.item(i).data(Qt.ItemDataRole.UserRole) for i in range(lst.count())]
        if ids != [p.id for p in projects]:
            lst.blockSignals(True)
            lst.clear()
            for p in projects:
                it = QListWidgetItem()
                it.setData(Qt.ItemDataRole.UserRole, p.id)
                lst.addItem(it)
                if p.id == cur:
                    lst.setCurrentItem(it)
            lst.blockSignals(False)
        query = self.search.text().strip()
        for i, p in enumerate(projects):
            info = p.info()
            vi = info.get("video_info", {})
            it = lst.item(i)
            job = jobs[p.id]
            line = self._status_text(p, job)
            kind = status_kind(line, job)
            dur = self._fmt_dur(vi.get("duration_s")) if vi else ""
            name = info.get("name", p.id)
            it.setData(NAME_ROLE, name)
            it.setData(PIN_ROLE, bool(info.get("pinned")))
            it.setData(STATUS_ROLE, line + (f" · {dur}" if dur else ""))
            it.setData(KIND_ROLE, kind)
            it.setData(SIZE_ROLE, fmt_size(sizes[p.id]))
            it.setText(f"{name}\n   {line}")  # keeps keyboard search / accessibility working
            created = info.get("created")
            when = time.strftime("%d.%m.%Y %H:%M", time.localtime(created)) if created else "unknown"
            tip = (f"<b>{name}</b><br>{KIND_TIPS[kind]}<br><span style='color:gray'>Added {when} · "
                   f"{fmt_size(sizes[p.id])} on disk{' · pinned' if info.get('pinned') else ''}</span>")
            if job.get("error"):
                from replay.gui.friendly import explain
                a = explain(job["error"])
                tip += f"<br><b>{a.headline}</b> {a.what_to_do}"
            it.setToolTip(tip)
            it.setHidden(bool(query) and not matches(p, query))

    # ------------------------------------------------------------------ actions
    def menu(self, pos) -> None:
        from replay.gui import ai_dialog

        win = self.win
        p = win._item_project(self.list.itemAt(pos))
        if p is None:
            return
        pinned = bool(p.info().get("pinned"))
        m = QMenu(win)
        a_rename = m.addAction("Rename…\tF2")
        a_dup = m.addAction("Duplicate")
        a_dup.setToolTip("Make a full copy of this Replay (for example to try other settings without losing this one)")
        a_folder = m.addAction("Open folder")
        a_pin = m.addAction("Unpin" if pinned else "Pin to top")
        m.addSeparator()
        win._fill_export_menu(m.addMenu("Export"), p)
        ai_dialog.add_menu_actions(m, win, p)
        m.addSeparator()
        a_delete = m.addAction("Delete…\tDel")
        m.setToolTipsVisible(True)
        chosen = m.exec(self.list.viewport().mapToGlobal(pos))
        if chosen is a_rename:
            win.rename_project(p)
        elif chosen is a_dup:
            self.duplicate(p)
        elif chosen is a_folder:
            open_folder(p.root)
        elif chosen is a_pin:
            p.update_info(pinned=not pinned)
            self.refresh(force=True)
        elif chosen is a_delete:
            win.delete_project(p)

    def duplicate(self, p: Project) -> Project | None:
        from replay.gui.app import run_with_progress
        from replay.gui.friendly import warn

        win = self.win
        if win.jobs.state(p.id).kind:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.information(win, "Duplicate Replay", "This Replay is busy (analysing or building).\n\n"
                                    "What to do: wait until the job has finished, or press Stop, then duplicate it.")
            return None
        old = p.info().get("name", p.id)
        name, ok = QInputDialog.getText(win, "Duplicate Replay", "Name of the copy:", QLineEdit.EchoMode.Normal,
                                        f"{old} (copy)")
        if not ok or not name.strip():
            return None
        try:
            new = run_with_progress(win, "Copying the Replay…",
                                    lambda progress: win.store.duplicate(p.id, name.strip(), progress=progress))
        except Exception as e:  # noqa: BLE001
            warn(win, "Could not duplicate the Replay.", e)
            return None
        win.show_project(new)
        return new


def open_folder(path) -> None:
    """Show a folder in the file manager."""
    path = str(path)
    if sys.platform == "win32":
        os.startfile(path)  # noqa: S606 - opening a folder the user asked for
    else:
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices
        QDesktopServices.openUrl(QUrl.fromLocalFile(path))


def install(win: "MainWindow") -> ReplayListController:
    ctl = ReplayListController(win)
    win.replay_list = ctl
    return ctl
