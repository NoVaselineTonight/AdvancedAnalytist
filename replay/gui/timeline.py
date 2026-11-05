"""Multi-track timeline for the Replay tab: user tracks of marks/ranges, the auto-detection track, a playhead that
follows playback, scrubbing, zoom/scroll and a hover preview with the 2D court at the mark's time.

The data model and persistence live in :mod:`replay.gui.timeline_model`; this module only draws and edits it.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable

from PySide6.QtCore import QPoint, QPointF, QRect, QRectF, QSize, Qt, Signal
from PySide6.QtGui import (QColor, QFont, QFontMetrics, QKeySequence, QPainter, QPainterPath, QPalette, QPen,
                           QPixmap, QPolygonF)
from PySide6.QtWidgets import (QCheckBox, QColorDialog, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
                               QFormLayout, QFrame, QHBoxLayout, QInputDialog, QLabel, QLineEdit, QMenu,
                               QMessageBox, QPlainTextEdit, QPushButton, QScrollBar, QSizePolicy, QToolButton,
                               QVBoxLayout, QWidget)

from replay.gui import timeline_model as tm
from replay.gui.helpicon import HelpIcon

HELP_KEYS = ("<b>Timeline.</b> M: add a mark at the playhead · double-click a track to add a mark · mouse wheel: zoom "
             "· Shift+wheel: scroll · right-click for options. Select a mark: Enter edits it, Del deletes (or "
             "dismisses an automatic one), drag to move it.")
from replay.gui.widgets import QUALITY_COLORS, ColorButton, paint_court, quality_stretches

HEADER_W = 184
RULER_H = 26
ROW_H = 34
MARK_HIT = 6          # half width of a point mark's hit area (12 px)
EDGE_HIT = 5          # range end handle
PLAYHEAD = QColor("#ef4444")
MAX_PPS = 600.0       # pixels per second at full zoom
PREVIEW_SIZE = QSize(320, 190)
TICK_STEPS = [0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600]


def fmt_ts(s: float, decimals: int = 0) -> str:
    s = max(0.0, s)
    m = int(s // 60)
    sec = s - 60 * m
    if decimals:
        txt = f"{sec:0{3 + decimals}.{decimals}f}"
        if txt.startswith("60"):  # rounding up
            m, txt = m + 1, f"{0:0{3 + decimals}.{decimals}f}"
        return f"{m}:{txt}"
    return f"{m}:{int(sec):02d}"


def _mix(a: QColor, b: QColor, f: float) -> QColor:
    return QColor(int(a.red() + (b.red() - a.red()) * f), int(a.green() + (b.green() - a.green()) * f),
                  int(a.blue() + (b.blue() - a.blue()) * f))


def _alpha(c: QColor, a: int) -> QColor:
    c = QColor(c)
    c.setAlpha(a)
    return c


# ====================================================================== hover preview
class MarkPreview(QFrame):
    """One reusable popup: title, time, a small 2D court render at the mark's time, and the note."""

    def __init__(self, parent: QWidget):
        super().__init__(parent, Qt.WindowType.ToolTip | Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setFrameShape(QFrame.Shape.Box)
        self.setBackgroundRole(QPalette.ColorRole.Window)
        self.setAutoFillBackground(True)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 6, 8, 8)
        lay.setSpacing(4)
        self.title = QLabel()
        f = self.title.font()
        f.setBold(True)
        f.setPointSizeF(f.pointSizeF() * 1.1)
        self.title.setFont(f)
        self.sub = QLabel()
        self.sub.setForegroundRole(QPalette.ColorRole.PlaceholderText)
        self.court = QLabel()
        self.court.setFixedSize(PREVIEW_SIZE)
        self.note = QLabel()
        self.note.setWordWrap(True)
        self.note.setTextFormat(Qt.TextFormat.PlainText)
        self.note.setMaximumWidth(PREVIEW_SIZE.width())
        for w in (self.title, self.sub, self.court, self.note):
            lay.addWidget(w)
        self.key = None

    def show_mark(self, key, title: str, sub: str, note: str, pixmap: QPixmap, global_rect: QRect) -> None:
        """Show for ``global_rect`` (the mark's row segment, in screen coordinates); placed above it, or below
        when there is no room, so it never sits under the mouse (no enter/leave flicker)."""
        if key != self.key:
            self.key = key
            self.title.setText(title or "(untitled)")
            self.sub.setText(sub)
            self.court.setPixmap(pixmap)
            self.note.setText(note)
            self.note.setVisible(bool(note.strip()))
            self.adjustSize()
        scr = (self.screen() or self.parentWidget().screen()).availableGeometry()
        w, h = self.width(), self.height()
        x = min(max(scr.left() + 4, global_rect.left() + 14), scr.right() - w - 4)
        y = global_rect.top() - h - 6
        if y < scr.top():
            y = global_rect.bottom() + 6
        self.move(x, y)
        if not self.isVisible():
            self.show()

    def hide_preview(self) -> None:
        self.key = None
        self.hide()


# ====================================================================== mark editor
class MarkDialog(QDialog):
    def __init__(self, parent: QWidget, tl: tm.Timeline, track: tm.Track, mark: tm.Mark, max_t: float):
        super().__init__(parent)
        self.setWindowTitle("Edit detection" if mark.is_auto else "Edit mark")
        self.setMinimumWidth(420)
        lay = QVBoxLayout(self)
        form = QFormLayout()
        self.title = QLineEdit(mark.title)
        self.title.setPlaceholderText("Title")
        self.note = QPlainTextEdit(mark.note)
        self.note.setPlaceholderText("Notes…")
        self.note.setMinimumHeight(90)
        self.time = QDoubleSpinBox()
        self.time.setRange(0, max(max_t, mark.t) + 3600)
        self.time.setDecimals(2)
        self.time.setSingleStep(0.1)
        self.time.setSuffix(" s")
        self.time.setValue(mark.t)
        self.time_lbl = QLabel()
        self.time.valueChanged.connect(lambda v: self.time_lbl.setText(fmt_ts(v, 1)))
        self.time_lbl.setText(fmt_ts(mark.t, 1))
        trow = QHBoxLayout()
        trow.addWidget(self.time, 1)
        trow.addWidget(self.time_lbl)
        self.duration = QDoubleSpinBox()
        self.duration.setRange(0, 3600)
        self.duration.setDecimals(2)
        self.duration.setSingleStep(0.5)
        self.duration.setSuffix(" s")
        self.duration.setSpecialValueText("point (no duration)")
        self.duration.setValue(mark.duration)
        self.track = QComboBox()
        for t in tl.tracks:
            if t.is_auto != mark.is_auto or (t.locked and t is not track):
                continue
            self.track.addItem(t.name, t.id)
            if t is track:
                self.track.setCurrentIndex(self.track.count() - 1)
        self.custom = QCheckBox("Custom colour")
        self.custom.setChecked(mark.color is not None and not mark.is_auto)
        self.color = ColorButton(mark.color or track.color)
        self.color.setEnabled(self.custom.isChecked())
        self.custom.toggled.connect(self.color.setEnabled)
        crow = QHBoxLayout()
        crow.addWidget(self.custom)
        crow.addWidget(self.color)
        crow.addStretch(1)
        form.addRow("Title", self.title)
        form.addRow("Note", self.note)
        form.addRow("Time", trow)
        form.addRow("Duration", self.duration)
        form.addRow("Track", self.track)
        form.addRow("Colour", crow)
        lay.addLayout(form)
        self.reset = False
        if mark.is_auto:
            for w in (self.time, self.duration, self.track, self.custom, self.color):
                w.setEnabled(False)
            conf = "" if mark.confidence is None else f" · confidence {mark.confidence * 100:.0f}%"
            info = QLabel(f"<span style='color:gray'>Detected automatically{conf}. Your title and note are kept "
                          "when the replay is rebuilt.</span>")
            info.setWordWrap(True)
            lay.addWidget(info)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        if mark.is_auto:
            b = bb.addButton("Reset to detected text", QDialogButtonBox.ButtonRole.ResetRole)
            b.clicked.connect(self._reset)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self.title.setFocus()
        self.title.selectAll()

    def _reset(self) -> None:
        self.reset = True
        self.accept()

    def values(self) -> dict:
        return {"title": self.title.text().strip(), "note": self.note.toPlainText().rstrip(),
                "t": self.time.value(), "duration": self.duration.value(), "track_id": self.track.currentData(),
                "color": self.color.color if self.custom.isChecked() else None}


# ====================================================================== canvas
class TimelineCanvas(QWidget):
    """Header column + ruler + track rows, all painted here."""

    def __init__(self, owner: "TimelineWidget"):
        super().__init__(owner)
        self.o = owner
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumHeight(RULER_H + ROW_H + 4)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.hover: str | None = None        # mark id under the mouse
        self.scrubbing = False
        self.drag: dict | None = None
        # Everything except the playhead is painted into a cached pixmap: during playback only the playhead moves
        # (30 updates/s), and repainting ~550 marks of a whole game each time took >100 ms of the GUI thread.
        self._cache: QPixmap | None = None
        self._cache_key: tuple | None = None
        self._dirty = True

    def update(self, *args) -> None:  # any normal update repaints the cached layer too
        self._dirty = True
        super().update(*args)

    def update_playhead(self) -> None:
        """Repaint for a moved playhead only (the cached layer is reused)."""
        QWidget.update(self)

    # ------------------------------------------------------------------ geometry
    def x_of(self, t: float) -> float:
        return HEADER_W + (t - self.o.t0) * self.o.pps

    def t_of(self, x: float) -> float:
        return self.o.t0 + (x - HEADER_W) / self.o.pps

    def row_top(self, i: int) -> int:
        return RULER_H + i * ROW_H - self.o.vofs

    def row_at(self, y: float) -> int | None:
        if y < RULER_H:
            return None
        i = int((y - RULER_H + self.o.vofs) // ROW_H)
        return i if 0 <= i < len(self.o.tl.tracks) else None

    def track_at(self, y: float) -> tm.Track | None:
        i = self.row_at(y)
        return None if i is None else self.o.tl.tracks[i]

    def icon_rects(self, i: int) -> tuple[QRect, QRect]:
        y = self.row_top(i) + (ROW_H - 18) // 2
        return QRect(HEADER_W - 50, y, 20, 18), QRect(HEADER_W - 26, y, 20, 18)  # eye, lock

    def mark_hit(self, pos: QPointF) -> tuple[tm.Track, tm.Mark, str] | None:
        """(track, mark, 'move'|'resize') under pos; nearest mark wins."""
        if pos.x() < HEADER_W or pos.y() < RULER_H:
            return None
        tr = self.track_at(pos.y())
        if tr is None or not tr.visible:
            return None
        best, bd = None, 1e9
        x = pos.x()
        for m in tr.marks:
            x0 = self.x_of(m.t)
            if m.duration > 0:
                x1 = self.x_of(m.end)
                if not tr.is_auto and abs(x - x1) <= EDGE_HIT and x1 - x0 > 2 * EDGE_HIT:
                    return tr, m, "resize"
                if x0 - MARK_HIT <= x <= x1 + 1:
                    d = 0 if x >= x0 else x0 - x
                    if d < bd:
                        best, bd = (tr, m, "move"), d
            elif abs(x - x0) <= MARK_HIT and abs(x - x0) < bd:
                best, bd = (tr, m, "move"), abs(x - x0)
        return best

    def mark_rect(self, tr: tm.Track, m: tm.Mark) -> QRect:
        i = self.o.tl.tracks.index(tr)
        x0 = int(self.x_of(m.t)) - MARK_HIT
        x1 = int(self.x_of(m.end)) + MARK_HIT
        return QRect(x0, self.row_top(i), max(x1 - x0, 2 * MARK_HIT), ROW_H)

    # ------------------------------------------------------------------ painting
    def paintEvent(self, e) -> None:
        W, H = self.width(), self.height()
        dpr = self.devicePixelRatioF()
        o = self.o
        key = (W, H, dpr, o.t0, o.pps, o.vofs, self.hasFocus())
        if self._dirty or self._cache is None or self._cache_key != key:
            pm = QPixmap(max(1, int(W * dpr)), max(1, int(H * dpr)))
            pm.setDevicePixelRatio(dpr)
            cp = QPainter(pm)
            self._paint_static(cp, W, H)
            cp.end()
            self._cache, self._cache_key, self._dirty = pm, key, False
        qp = QPainter(self)
        qp.drawPixmap(0, 0, self._cache)
        self._paint_playhead(qp, W, H)
        qp.end()

    def _paint_playhead(self, qp: QPainter, W: int, H: int) -> None:
        o, pal = self.o, self.palette()
        qp.setRenderHint(QPainter.RenderHint.Antialiasing)
        xp = self.x_of(o.playhead)
        if HEADER_W - 1 <= xp <= W + 1:
            qp.setPen(QPen(PLAYHEAD, 2))
            qp.drawLine(QPointF(xp, RULER_H - 6), QPointF(xp, H))
            qp.setPen(Qt.PenStyle.NoPen)
            qp.setBrush(PLAYHEAD)
            qp.drawPolygon(QPolygonF([QPointF(xp - 6, RULER_H - 12), QPointF(xp + 6, RULER_H - 12),
                                      QPointF(xp + 6, RULER_H - 6), QPointF(xp, RULER_H),
                                      QPointF(xp - 6, RULER_H - 6)]))
        qp.fillRect(QRect(0, 0, HEADER_W - 1, RULER_H), pal.color(QPalette.ColorRole.Window))
        small = QFont(self.font())
        small.setPointSizeF(max(7.0, small.pointSizeF() * 0.9))
        qp.setFont(small)
        qp.setPen(pal.color(QPalette.ColorRole.PlaceholderText))
        qp.drawText(QRect(10, 0, HEADER_W - 12, RULER_H), Qt.AlignmentFlag.AlignVCenter,
                    f"{len(o.tl.tracks)} tracks · {fmt_ts(o.playhead, 1)}")

    def _paint_static(self, qp: QPainter, W: int, H: int) -> None:
        o, pal = self.o, self.palette()
        qp.setRenderHint(QPainter.RenderHint.Antialiasing)
        base, alt, win = pal.color(QPalette.ColorRole.Base), pal.color(QPalette.ColorRole.AlternateBase), \
            pal.color(QPalette.ColorRole.Window)
        text, mid = pal.color(QPalette.ColorRole.Text), pal.color(QPalette.ColorRole.Mid)
        dim = pal.color(QPalette.ColorRole.PlaceholderText)
        hl = pal.color(QPalette.ColorRole.Highlight)
        qp.fillRect(self.rect(), base)

        # ---- rows background
        tracks = o.tl.tracks
        for i, tr in enumerate(tracks):
            y = self.row_top(i)
            if y > H or y + ROW_H < RULER_H:
                continue
            bg = alt if i % 2 else base
            if tr.id == o.sel_track:
                bg = _mix(bg, hl, 0.10)
            qp.fillRect(QRect(HEADER_W, y, W - HEADER_W, ROW_H), bg)
            if not tr.visible:
                qp.fillRect(QRect(HEADER_W, y, W - HEADER_W, ROW_H),
                            QColor(mid.red(), mid.green(), mid.blue(), 40))

        # ---- grid + ruler
        step = self._tick_step()
        dec = 1 if step < 1 else 0
        first = math.floor(o.t0 / step) * step
        qp.fillRect(QRect(0, 0, W, RULER_H), win)
        grid = _alpha(mid, 70)
        small = QFont(self.font())
        small.setPointSizeF(max(7.0, small.pointSizeF() * 0.9))
        qp.setFont(small)
        t = first
        while self.x_of(t) < W:
            x = self.x_of(t)
            if x >= HEADER_W:
                qp.setPen(QPen(grid, 1))
                qp.drawLine(QPointF(x, RULER_H), QPointF(x, H))
                qp.setPen(QPen(dim, 1))
                qp.drawLine(QPointF(x, RULER_H - 7), QPointF(x, RULER_H))
                qp.setPen(text)
                qp.drawText(QPointF(x + 3, RULER_H - 9), fmt_ts(t + 1e-9, dec))
                # minor ticks
                for k in range(1, 5):
                    xm = self.x_of(t + step * k / 5)
                    if xm < W:
                        qp.setPen(QPen(dim, 1))
                        qp.drawLine(QPointF(xm, RULER_H - 3), QPointF(xm, RULER_H))
            t += step
        # end of media
        if o.duration > 0:
            xe = self.x_of(o.duration)
            if HEADER_W <= xe < W:
                qp.fillRect(QRectF(xe, RULER_H, W - xe, H - RULER_H), _alpha(mid, 50))
        qp.setPen(QPen(mid, 1))
        qp.drawLine(0, RULER_H, W, RULER_H)
        # tracking-quality lane: thin strip at the top of the ruler (colours: widgets.QUALITY_COLORS)
        if o.data:
            qcol = {k: QColor(v) for k, v in QUALITY_COLORS.items()}
            t_lo, t_hi = self.t_of(HEADER_W), self.t_of(W)
            for qa, qb, lev in quality_stretches(o.data):
                if qb < t_lo or qa > t_hi:
                    continue
                xa, xb = max(HEADER_W, self.x_of(qa)), min(W, self.x_of(qb))
                if xb > xa:
                    qp.fillRect(QRectF(xa, 0, max(1.0, xb - xa), 5), qcol[lev])

        # ---- marks
        qp.save()
        qp.setClipRect(QRect(HEADER_W, RULER_H, W - HEADER_W, H - RULER_H))
        fm = QFontMetrics(self.font())
        qp.setFont(self.font())
        for i, tr in enumerate(tracks):
            y = self.row_top(i)
            if y > H or y + ROW_H < RULER_H or not tr.visible:
                continue
            marks = sorted(tr.marks, key=lambda m: m.t)
            for k, m in enumerate(marks):
                x0 = self.x_of(m.t)
                if x0 > W + MARK_HIT or (self.x_of(m.end) < HEADER_W - 240):  # off screen (labels reach ~230 px)
                    continue
                nxt = self.x_of(marks[k + 1].t) if k + 1 < len(marks) else W + 400
                self._paint_mark(qp, fm, tr, m, y, nxt, text, hl, base)
        qp.restore()

        # ---- header column (drawn last: covers anything scrolled under it)
        qp.fillRect(QRect(0, 0, HEADER_W, H), win)
        qp.fillRect(QRect(0, 0, HEADER_W, RULER_H), win)
        qp.save()
        qp.setClipRect(QRect(0, RULER_H, HEADER_W, H - RULER_H))
        bold = QFont(self.font())
        bold.setBold(True)
        for i, tr in enumerate(tracks):
            y = self.row_top(i)
            if y > H or y + ROW_H < RULER_H:
                continue
            if tr.id == o.sel_track:
                qp.fillRect(QRect(0, y, HEADER_W, ROW_H), _alpha(hl, 60))
            col = QColor(tr.color)
            qp.fillRect(QRect(0, y + 1, 4, ROW_H - 2), col)
            qp.setPen(QPen(_alpha(text, 120), 1))
            qp.setBrush(col if tr.visible else Qt.BrushStyle.NoBrush)
            qp.drawRoundedRect(QRectF(12, y + ROW_H / 2 - 6, 12, 12), 3, 3)
            qp.setFont(bold if tr.id == o.sel_track else self.font())
            qp.setPen(text if tr.visible else dim)
            name = tr.name
            label_w = HEADER_W - 32 - 56
            qp.drawText(QRect(32, y, label_w, ROW_H), Qt.AlignmentFlag.AlignVCenter,
                        QFontMetrics(qp.font()).elidedText(name, Qt.TextElideMode.ElideRight, label_w))
            if tr.is_auto:
                qp.setFont(small)
                qp.setPen(dim)
                qp.drawText(QRect(32, y + ROW_H - 13, label_w, 12), Qt.AlignmentFlag.AlignLeft,
                            f"auto · {len(tr.marks)}")
            eye, lock = self.icon_rects(i)
            self._paint_eye(qp, eye, tr.visible, text)
            self._paint_lock(qp, lock, tr.locked, text)
            qp.setPen(QPen(_alpha(mid, 90), 1))
            qp.drawLine(0, y + ROW_H - 1, W, y + ROW_H - 1)
        qp.restore()
        qp.setPen(QPen(mid, 1))
        qp.drawLine(HEADER_W - 1, 0, HEADER_W - 1, H)
        if not tracks:
            qp.setPen(dim)
            qp.drawText(QRect(HEADER_W, RULER_H, W - HEADER_W, H - RULER_H), Qt.AlignmentFlag.AlignCenter,
                        "No tracks")
        if self.hasFocus():
            qp.setPen(QPen(_alpha(hl, 160), 1))
            qp.setBrush(Qt.BrushStyle.NoBrush)
            qp.drawRect(QRectF(0.5, 0.5, W - 1, H - 1))

    def _paint_mark(self, qp: QPainter, fm: QFontMetrics, tr: tm.Track, m: tm.Mark, y: int, next_x: float,
                    text: QColor, hl: QColor, base: QColor) -> None:
        col = QColor(m.color or tr.color)
        x0 = self.x_of(m.t)
        sel, hov = m.id == self.o.sel_mark, m.id == self.hover
        outline = QPen(hl if sel else _alpha(text, 150), 2 if sel else 1)
        top, bot = y + 5, y + ROW_H - 5
        if m.duration > 0:
            x1 = max(self.x_of(m.end), x0 + 3)
            r = QRectF(x0, top, x1 - x0, bot - top)
            qp.setPen(outline)
            qp.setBrush(_alpha(col, 150 if hov or sel else 110))
            qp.drawRoundedRect(r, 4, 4)
            qp.setPen(QPen(col, 3))
            qp.drawLine(QPointF(x0 + 1.5, top + 2), QPointF(x0 + 1.5, bot - 2))
            if not tr.is_auto and (hov or sel):  # resize grip
                qp.setPen(QPen(_alpha(text, 170), 1))
                for dx in (-4, -2):
                    qp.drawLine(QPointF(x1 + dx, top + 8), QPointF(x1 + dx, bot - 8))
            label_x, avail = x0 + 7, min(x1 - x0 - 10, next_x - x0 - 12)
        else:
            qp.setPen(QPen(col, 3 if hov or sel else 2))
            qp.drawLine(QPointF(x0, top), QPointF(x0, bot))
            s = 6.5 if hov or sel else 5.5
            cy = y + ROW_H / 2
            path = QPainterPath()
            path.moveTo(x0, cy - s)
            path.lineTo(x0 + s, cy)
            path.lineTo(x0, cy + s)
            path.lineTo(x0 - s, cy)
            path.closeSubpath()
            qp.setPen(outline)
            qp.setBrush(col)
            qp.drawPath(path)
            label_x, avail = x0 + s + 4, min(220.0, next_x - x0 - s - 10)
        if avail > 24 and m.title:
            qp.setPen(text)
            qp.drawText(QRectF(label_x, y, avail, ROW_H), Qt.AlignmentFlag.AlignVCenter,
                        fm.elidedText(m.title, Qt.TextElideMode.ElideRight, int(avail)))

    @staticmethod
    def _paint_eye(qp: QPainter, r: QRect, on: bool, c: QColor) -> None:
        qp.save()
        col = _alpha(c, 220 if on else 90)
        qp.setPen(QPen(col, 1.4))
        qp.setBrush(Qt.BrushStyle.NoBrush)
        cx, cy = r.center().x() + 0.5, r.center().y() + 0.5
        path = QPainterPath()
        path.moveTo(cx - 7, cy)
        path.quadTo(cx, cy - 7, cx + 7, cy)
        path.quadTo(cx, cy + 7, cx - 7, cy)
        qp.drawPath(path)
        if on:
            qp.setBrush(col)
            qp.drawEllipse(QPointF(cx, cy), 2.2, 2.2)
        else:
            qp.drawLine(QPointF(cx - 6, cy + 5), QPointF(cx + 6, cy - 5))
        qp.restore()

    @staticmethod
    def _paint_lock(qp: QPainter, r: QRect, on: bool, c: QColor) -> None:
        qp.save()
        col = _alpha(c, 220 if on else 90)
        qp.setPen(QPen(col, 1.4))
        cx, cy = r.center().x() + 0.5, r.center().y() + 0.5
        qp.setBrush(col if on else Qt.BrushStyle.NoBrush)
        qp.drawRoundedRect(QRectF(cx - 5, cy - 1, 10, 7), 1.5, 1.5)
        qp.setBrush(Qt.BrushStyle.NoBrush)
        sh = QPainterPath()
        if on:
            sh.moveTo(cx - 3, cy - 1)
            sh.lineTo(cx - 3, cy - 4)
            sh.arcTo(QRectF(cx - 3, cy - 7, 6, 6), 180, -180)
            sh.lineTo(cx + 3, cy - 1)
        else:
            sh.moveTo(cx + 3, cy - 1)
            sh.lineTo(cx + 3, cy - 4)
            sh.arcTo(QRectF(cx + 3, cy - 7, 6, 6), 180, -180)
            sh.lineTo(cx + 9, cy - 3)
        qp.drawPath(sh)
        qp.restore()

    def _tick_step(self) -> float:
        for s in TICK_STEPS:
            if s * self.o.pps >= 64:
                return s
        return TICK_STEPS[-1]

    # ------------------------------------------------------------------ mouse
    def mousePressEvent(self, e) -> None:
        self.setFocus()
        pos = e.position()
        o = self.o
        o.preview.hide_preview()
        if e.button() != Qt.MouseButton.LeftButton:
            return
        if pos.x() < HEADER_W:
            i = self.row_at(pos.y())
            if i is None:
                return
            tr = o.tl.tracks[i]
            eye, lock = self.icon_rects(i)
            if eye.adjusted(-2, -4, 2, 4).contains(pos.toPoint()):
                o.set_track_flag(tr, "visible", not tr.visible)
            elif lock.adjusted(-2, -4, 2, 4).contains(pos.toPoint()):
                o.set_track_flag(tr, "locked", not tr.locked)
            o.select(track=tr.id, mark=None)
            return
        hit = self.mark_hit(pos)
        if hit:
            tr, m, mode = hit
            o.select(track=tr.id, mark=m.id)
            self.drag = {"id": m.id, "x": pos.x(), "t": m.t, "dur": m.duration, "mode": mode, "moved": False,
                         "track": tr.id}
            return
        tr = self.track_at(pos.y())
        if tr is not None:
            o.select(track=tr.id, mark=None)
        self.scrubbing = True
        o.scrub(self.t_of(pos.x()))

    def mouseMoveEvent(self, e) -> None:
        pos = e.position()
        o = self.o
        if self.scrubbing:
            o.scrub(self.t_of(max(HEADER_W, min(self.width(), pos.x()))))
            return
        if self.drag is not None:
            d = self.drag
            if not d["moved"] and abs(pos.x() - d["x"]) < 4 and self.track_at(pos.y()) is not None \
                    and self.track_at(pos.y()).id == d["track"]:
                return
            tr, m = o.tl.find_mark(d["id"])
            if m is None or tr.locked or m.is_auto:
                if not d["moved"] and m is not None:
                    o.flash("Detections can't be moved; edit or dismiss them instead." if m.is_auto else
                            f"Track “{tr.name}” is locked.")
                d["moved"] = True
                d["blocked"] = True
                return
            if d.get("blocked"):
                return
            d["moved"] = True
            dt = (pos.x() - d["x"]) / o.pps
            if d["mode"] == "resize":
                m.duration = max(0.0, d["dur"] + dt)
            else:
                nt = max(0.0, d["t"] + dt)
                # snap to the playhead (hold Shift to disable)
                if not (e.modifiers() & Qt.KeyboardModifier.ShiftModifier) and \
                        abs((nt - o.playhead) * o.pps) < 6:
                    nt = o.playhead
                dst = self.track_at(pos.y())
                tid = dst.id if dst is not None and not dst.is_auto and not dst.locked and dst.visible else None
                o.tl.move_mark(m.id, nt, tid)
            o.flash(f"{m.title or 'Mark'} → {fmt_ts(m.t, 1)}" + (f" ({m.duration:.1f} s)" if m.duration else ""))
            self.update()
            return
        hit = self.mark_hit(pos)
        if hit and hit[2] == "resize":
            self.setCursor(Qt.CursorShape.SizeHorCursor)
        elif hit:
            self.setCursor(Qt.CursorShape.PointingHandCursor)
        elif pos.y() < RULER_H and pos.x() >= HEADER_W:
            self.setCursor(Qt.CursorShape.IBeamCursor)
        else:
            self.unsetCursor()
        self.set_hover(hit[0] if hit else None, hit[1] if hit else None)

    def mouseReleaseEvent(self, e) -> None:
        o = self.o
        if self.scrubbing:
            self.scrubbing = False
            o.end_scrub()
        if self.drag is not None:
            d, self.drag = self.drag, None
            _, m = o.tl.find_mark(d["id"])
            if m is not None:
                if not d["moved"]:
                    o.seek(m.t)
                elif not d.get("blocked"):
                    o.changed()

    def mouseDoubleClickEvent(self, e) -> None:
        pos = e.position()
        o = self.o
        self.scrubbing = False
        o.end_scrub()
        self.drag = None
        if e.button() != Qt.MouseButton.LeftButton:
            return
        if pos.x() < HEADER_W:
            tr = self.track_at(pos.y())
            if tr is not None and not any(r.adjusted(-2, -4, 2, 4).contains(pos.toPoint())
                                          for r in self.icon_rects(o.tl.tracks.index(tr))):
                o.rename_track(tr)
            elif tr is None and pos.y() >= RULER_H:
                o.add_track()
            return
        hit = self.mark_hit(pos)
        if hit:
            o.edit_mark(hit[1].id)
            return
        tr = self.track_at(pos.y())
        if tr is not None:
            o.add_mark(tr, self.t_of(pos.x()), edit=True)

    def wheelEvent(self, e) -> None:
        o = self.o
        dy, dx = e.angleDelta().y(), e.angleDelta().x()
        if e.position().x() < HEADER_W and not dx:
            o.vbar.setValue(o.vbar.value() - int(dy / 120 * ROW_H))
        elif dx or e.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            d = dx or dy
            o.hbar.setValue(o.hbar.value() - int(d / 120 * 0.15 * self.visible_secs() * 1000))
        elif dy:
            o.zoom(1.25 ** (dy / 120), self.t_of(e.position().x()))
        e.accept()

    def visible_secs(self) -> float:
        return max(0.1, (self.width() - HEADER_W) / self.o.pps)

    def leaveEvent(self, e) -> None:
        self.set_hover(None, None)
        self.unsetCursor()

    def set_hover(self, tr: tm.Track | None, m: tm.Mark | None) -> None:
        mid = m.id if m is not None else None
        if mid != self.hover:
            self.hover = mid
            self.update()
        if m is None or self.drag is not None or self.scrubbing:
            self.o.preview.hide_preview()
        else:
            self.o.show_preview(tr, m, self.mark_rect(tr, m))

    def keyPressEvent(self, e) -> None:
        o = self.o
        k = e.key()
        if k in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace) and o.sel_mark:
            o.delete_mark(o.sel_mark)
        elif k in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and o.sel_mark:
            o.edit_mark(o.sel_mark)
        elif k == Qt.Key.Key_Escape:
            o.select(mark=None)
        elif k in (Qt.Key.Key_Up, Qt.Key.Key_Down):
            o.select_track_step(-1 if k == Qt.Key.Key_Up else 1)
        else:
            super().keyPressEvent(e)

    def changeEvent(self, e) -> None:  # palette / style / font changes: repaint the cached layer
        self._dirty = True
        super().changeEvent(e)

    def focusInEvent(self, e) -> None:
        self.update()

    def focusOutEvent(self, e) -> None:
        self.update()

    def contextMenuEvent(self, e) -> None:
        self.o.preview.hide_preview()
        self.o.context_menu(e.pos(), e.globalPos())


# ====================================================================== timeline widget
class TimelineWidget(QWidget):
    """Toolbar + canvas + scrollbars. The Replay tab drives it with :meth:`set_time` and listens to
    ``seekRequested``."""

    seekRequested = Signal(float)

    def __init__(self, colors_fn: Callable[[], dict] | None = None, parent: QWidget | None = None):
        super().__init__(parent)
        self.colors_fn = colors_fn or (lambda: {})
        self.root: Path | None = None
        self.tl = tm.Timeline.default()
        self.data: dict | None = None
        self.duration = 0.0
        self.playhead = 0.0
        self.playing = False
        self.t0 = 0.0
        self.pps = 40.0
        self.vofs = 0
        self.sel_track: str | None = None
        self.sel_mark: str | None = None
        self._scrub_t: float | None = None
        self._fitted = False
        self._pix_cache: dict = {}

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        bar = QHBoxLayout()
        bar.setSpacing(4)
        self.btn_mark = QPushButton("Add mark")
        self.btn_mark.setToolTip("Add a mark at the playhead on the selected track (M)")
        self.btn_mark.clicked.connect(self.add_mark_at_playhead)
        self.btn_track = QPushButton("+ Add track")
        self.btn_track.clicked.connect(lambda: self.add_track())
        self.btn_prev = QToolButton()
        self.btn_prev.setText("◀")
        self.btn_prev.setToolTip("Previous mark ([)")
        self.btn_prev.clicked.connect(lambda: self.jump(-1))
        self.btn_next = QToolButton()
        self.btn_next.setText("▶")
        self.btn_next.setToolTip("Next mark (])")
        self.btn_next.clicked.connect(lambda: self.jump(1))
        self.info = QLabel()
        self.info.setForegroundRole(QPalette.ColorRole.PlaceholderText)
        self.info.setTextFormat(Qt.TextFormat.PlainText)
        self.info.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.follow = QCheckBox("Follow")
        self.follow.setChecked(True)
        self.follow.setToolTip("Keep the playhead in view while playing")
        zo, zi, fit = QToolButton(), QToolButton(), QToolButton()
        zo.setText("−")
        zi.setText("+")
        fit.setText("Fit")
        zo.setToolTip("Zoom out (mouse wheel)")
        zi.setToolTip("Zoom in (mouse wheel)")
        fit.setToolTip("Show the whole game")
        zo.clicked.connect(lambda: self.zoom(0.8))
        zi.clicked.connect(lambda: self.zoom(1.25))
        fit.clicked.connect(self.fit)
        for w in (self.btn_mark, self.btn_track, self.btn_prev, self.btn_next):
            bar.addWidget(w)
        bar.addWidget(self.info, 1)
        self.info_help = HelpIcon(HELP_KEYS, "Timeline keys")
        bar.addWidget(self.info_help)
        for w in (self.follow, zo, zi, fit):
            bar.addWidget(w)
        lay.addLayout(bar)

        grid = QHBoxLayout()
        grid.setSpacing(0)
        self.canvas = TimelineCanvas(self)
        self.vbar = QScrollBar(Qt.Orientation.Vertical)
        self.vbar.valueChanged.connect(self._vscrolled)
        grid.addWidget(self.canvas, 1)
        grid.addWidget(self.vbar)
        lay.addLayout(grid, 1)
        self.hbar = QScrollBar(Qt.Orientation.Horizontal)
        self.hbar.valueChanged.connect(self._hscrolled)
        lay.addWidget(self.hbar)
        self.preview = MarkPreview(self)
        self.setMinimumHeight(110)
        self._hint()

    # ------------------------------------------------------------------ external api
    def set_project(self, root: Path | None) -> None:
        self.preview.hide_preview()
        self.root = Path(root) if root is not None else None
        self.tl = tm.load(self.root) if self.root is not None else tm.Timeline.default()
        self._inplay_mtime = None
        self.sel_mark = None
        self.sel_track = next((t.id for t in self.tl.tracks if not t.is_auto), None)
        self.data = None
        self.duration = 0.0
        self.playhead = 0.0
        self._fitted = False
        self._pix_cache.clear()
        self.setEnabled(self.root is not None)
        if self.root is not None and not (self.root / tm.FILENAME).exists():
            self.changed()  # create the defaults on disk
        self._layout_changed()

    def set_data(self, data: dict | None) -> None:
        """New replay data (replay_data.json changed): rebuild the auto-detection track."""
        self.data = data
        self._pix_cache.clear()
        if data and data.get("t"):
            self.set_duration(max(self.duration, float(data["t"][-1])))
        self.refresh_auto()

    def _auto_events(self) -> list[dict]:
        """Detected baskets + game-state events (possessions, shots, turnovers, free throws, ...) + the light AI
        model's events; see replay/game_state.py."""
        base = (self.data or {}).get("events") or []
        if self.root is None or not self.data:
            return base
        try:
            from replay import game_state
            evs = game_state.auto_events(self.root, self.data)
            from replay.roster_names import decorate_events  # roster hook: "Pass #7 Virtanen -> #12 Korhonen"
            return decorate_events(self.root, self.data, evs)  # roster hook (unchanged without roster.json)
        except Exception as e:  # noqa: BLE001 - never lose the basket marks over a game-state problem
            print("timeline: game state unavailable:", type(e).__name__, e)
            return base

    def refresh_auto(self) -> None:
        """Re-derive auto marks (also call after team colours/names changed)."""
        self._pix_cache.clear()
        cols = self.colors_fn() or {}
        teams = {k: {"name": v.get("name") or k, "color": cols.get(k, v.get("primary"))}
                 for k, v in ((self.data or {}).get("game", {}).get("teams", {}) or {}).items()}
        if self.tl.set_events(self._auto_events(), teams):
            if self.sel_mark and self.tl.find_mark(self.sel_mark)[1] is None:
                self.sel_mark = None
            self.changed()
        else:
            self.canvas.update()

    def set_inplay_file(self, path: Path | None) -> None:
        """Mirror ``results/inplay.json`` (preliminary analysis) on the "Ball not in play" track; cheap to call often."""
        try:
            mt = path.stat().st_mtime if path is not None else None
        except OSError:
            mt = None
        if mt == getattr(self, "_inplay_mtime", None) or self.root is None:
            return
        self._inplay_mtime = mt
        if mt is None:
            return  # keep whatever the timeline has
        from replay.inplay import load_intervals

        iv = load_intervals(path)
        span = None
        try:
            import json

            d = json.loads(path.read_text(encoding="utf-8"))
            if d.get("end_s") is not None:
                span = (float(d.get("start_s") or 0.0), float(d["end_s"]))
        except (OSError, ValueError, TypeError):
            pass
        if iv is not None and self.tl.set_inplay(iv, span):
            self.changed()

    def set_duration(self, d: float) -> None:
        d = float(d or 0.0)
        if d > 0 and abs(d - self.duration) > 1e-3:
            self.duration = d
            if not self._fitted and self.canvas.width() > HEADER_W + 50:
                self.fit()
            self._update_scrollbars()

    def set_time(self, t: float, playing: bool = False) -> None:
        if self._scrub_t is not None:  # the user is scrubbing: our playhead leads
            return
        self.playing = playing
        if abs(t - self.playhead) < 1e-4:
            return
        self.playhead = t
        if playing and self.follow.isChecked() and self.canvas.drag is None:
            vis = self.canvas.visible_secs()
            if t > self.t0 + vis * 0.95 or t < self.t0:
                self._set_t0(t - vis * 0.05)
        self.canvas.update_playhead()

    # ------------------------------------------------------------------ view
    def _total(self) -> float:
        last = max((m.end for _, m in self.tl.all_marks(False)), default=0.0)
        return max(self.duration, last + 1.0, 10.0)

    def fit(self) -> None:
        self._fitted = True
        w = max(50, self.canvas.width() - HEADER_W - 12)
        self.pps = min(MAX_PPS, w / self._total())
        self._set_t0(0.0)
        self._update_scrollbars()
        self.canvas.update()

    def zoom(self, f: float, around: float | None = None) -> None:
        if around is None:
            around = self.playhead if self.t0 <= self.playhead <= self.t0 + self.canvas.visible_secs() \
                else self.t0 + self.canvas.visible_secs() / 2
        w = max(50, self.canvas.width() - HEADER_W - 12)
        new = max(min(MAX_PPS, w / self._total()), min(MAX_PPS, self.pps * f))
        x = self.canvas.x_of(around)
        self.pps = new
        self._update_scrollbars()
        self._set_t0(around - (x - HEADER_W) / new)
        self.canvas.update()

    def _set_t0(self, t0: float) -> None:
        mx = max(0.0, self._total() - self.canvas.visible_secs())
        self.t0 = max(0.0, min(mx, t0))
        self.hbar.blockSignals(True)
        self.hbar.setValue(int(self.t0 * 1000))
        self.hbar.blockSignals(False)
        self.canvas.update()

    def _update_scrollbars(self) -> None:
        vis = self.canvas.visible_secs()
        mx = max(0.0, self._total() - vis)
        self.hbar.blockSignals(True)
        self.hbar.setRange(0, int(mx * 1000))
        self.hbar.setPageStep(int(vis * 1000))
        self.hbar.setSingleStep(max(1, int(vis * 100)))
        self.hbar.setValue(int(self.t0 * 1000))
        self.hbar.blockSignals(False)
        self.hbar.setVisible(mx > 0)
        content = len(self.tl.tracks) * ROW_H
        avail = self.canvas.height() - RULER_H
        vmax = max(0, content - avail)
        self.vbar.setRange(0, vmax)
        self.vbar.setPageStep(max(1, avail))
        self.vbar.setSingleStep(ROW_H // 2)
        self.vbar.setVisible(vmax > 0)
        self.vofs = min(self.vofs, vmax)

    def _hscrolled(self, v: int) -> None:
        self.t0 = v / 1000.0
        self.canvas.update()

    def _vscrolled(self, v: int) -> None:
        self.vofs = v
        self.canvas.update()

    def _layout_changed(self) -> None:
        self._update_scrollbars()
        self.canvas.update()

    def resizeEvent(self, e) -> None:
        super().resizeEvent(e)
        if not self._fitted and self.duration > 0 and self.canvas.width() > HEADER_W + 50:
            self.fit()
        else:
            self.pps = max(self.pps, min(MAX_PPS, max(50, self.canvas.width() - HEADER_W - 12) / self._total()))
            self._update_scrollbars()
            self._set_t0(self.t0)

    def hideEvent(self, e) -> None:
        self.preview.hide_preview()
        super().hideEvent(e)

    # ------------------------------------------------------------------ seeking
    def seek(self, t: float) -> None:
        t = max(0.0, min(t, self.duration)) if self.duration > 0 else max(0.0, t)
        self.playhead = t
        self.canvas.update_playhead()
        self.seekRequested.emit(t)

    def scrub(self, t: float) -> None:
        self._scrub_t = t
        self.seek(t)

    def end_scrub(self) -> None:
        self._scrub_t = None

    def jump(self, direction: int) -> None:
        marks = [m.t for _, m in self.tl.all_marks()]
        if direction > 0:
            nxt = [t for t in marks if t > self.playhead + 0.05]
            target = min(nxt) if nxt else None
        else:
            prv = [t for t in marks if t < self.playhead - 0.05]
            target = max(prv) if prv else None
        if target is None:
            self.flash("No more marks in that direction.")
            return
        hit = next((m for _, m in self.tl.all_marks() if m.t == target), None)
        if hit is not None:
            self.select(mark=hit.id)
        self.seek(target)
        vis = self.canvas.visible_secs()
        if not self.t0 <= target <= self.t0 + vis:
            self._set_t0(target - vis * 0.3)

    # ------------------------------------------------------------------ selection / messages
    def select(self, track: str | None | bool = False, mark: str | None | bool = False) -> None:
        if track is not False:
            self.sel_track = track
        if mark is not False:
            self.sel_mark = mark
            tr = self.tl.find_mark(mark)[0] if mark else None
            if tr is not None:
                self.sel_track = tr.id
        self._hint()
        self.canvas.update()

    def select_track_step(self, d: int) -> None:
        ids = [t.id for t in self.tl.tracks]
        if not ids:
            return
        i = ids.index(self.sel_track) if self.sel_track in ids else 0
        self.select(track=ids[max(0, min(len(ids) - 1, i + d))], mark=None)

    def _hint(self) -> None:
        tr, m = self.tl.find_mark(self.sel_mark) if self.sel_mark else (None, None)
        if m is not None:
            self.info.setText(f"{m.title or 'Mark'} · {fmt_ts(m.t, 1)} · {tr.name}   —   Enter edit · Del "
                              f"{'dismiss' if m.is_auto else 'delete'} · drag to move")
        else:
            self.info.setText("")  # the default key line is behind the ? next to it (HELP_KEYS)
        self.info_help.setVisible(m is None)

    def flash(self, msg: str) -> None:
        self.info.setText(msg)

    # ------------------------------------------------------------------ persistence
    def changed(self) -> None:
        self._update_scrollbars()
        self.canvas.update()
        if self.root is None:
            return
        try:
            self.tl.save(self.root / tm.FILENAME)
        except OSError as ex:
            self.flash(f"Could not save the timeline: {ex}")

    # ------------------------------------------------------------------ track ops
    def add_track(self) -> None:
        name, ok = QInputDialog.getText(self.window(), "Add track", "Track name:",
                                        text=f"Track {len(self.tl.tracks) + 1}")
        if not ok:
            return
        tr = self.tl.add_track(name.strip() or "Track")
        self.select(track=tr.id, mark=None)
        self.changed()
        self.vbar.setValue(self.vbar.maximum())

    def rename_track(self, tr: tm.Track) -> None:
        name, ok = QInputDialog.getText(self.window(), "Rename track", "Track name:", text=tr.name)
        if ok and name.strip():
            self.tl.update_track(tr.id, name=name.strip())
            self.changed()

    def recolor_track(self, tr: tm.Track) -> None:
        c = QColorDialog.getColor(QColor(tr.color), self.window(), f"Colour of “{tr.name}”")
        if c.isValid():
            self.tl.update_track(tr.id, color=c.name())
            self.changed()

    def set_track_flag(self, tr: tm.Track, key: str, value: bool) -> None:
        self.tl.update_track(tr.id, **{key: value})
        if key == "visible" and not value:
            self.preview.hide_preview()
            if self.sel_mark and self.tl.find_mark(self.sel_mark)[0] is tr:
                self.sel_mark = None
        self.changed()

    def move_track(self, tr: tm.Track, d: int) -> None:
        if self.tl.move_track(tr.id, d):
            self.changed()

    def delete_track(self, tr: tm.Track) -> None:
        if tr.is_auto:
            return
        n = len(tr.marks)
        if QMessageBox.question(self.window(), "Delete track",
                                f"Delete the track “{tr.name}”" + (f" and its {n} mark{'s' * (n != 1)}" if n else "")
                                + "? This cannot be undone.") != QMessageBox.StandardButton.Yes:
            return
        self.tl.delete_track(tr.id)
        if self.sel_track == tr.id:
            self.sel_track = None
        if self.sel_mark and self.tl.find_mark(self.sel_mark)[1] is None:
            self.sel_mark = None
        self._hint()
        self.changed()

    # ------------------------------------------------------------------ mark ops
    def _target_track(self) -> tm.Track | None:
        tr = self.tl.track(self.sel_track) if self.sel_track else None
        if tr is not None and not tr.is_auto and not tr.locked and tr.visible:
            return tr
        tr = next((t for t in self.tl.tracks if not t.is_auto and not t.locked and t.visible), None)
        if tr is None:
            tr = self.tl.add_track("Notes")
        return tr

    def add_mark_at_playhead(self) -> None:
        tr = self._target_track()
        self.add_mark(tr, self.playhead)

    def add_mark(self, tr: tm.Track, t: float, edit: bool = False) -> tm.Mark | None:
        if tr.is_auto:
            self.flash("The auto-detection track is filled automatically; add marks to another track.")
            return None
        if tr.locked:
            self.flash(f"Track “{tr.name}” is locked.")
            return None
        if not tr.visible:
            self.tl.update_track(tr.id, visible=True)
        n = len(tr.marks) + 1
        m = self.tl.add_mark(tr.id, max(0.0, t), title=f"{tr.name.rstrip('s') or 'Mark'} {n}")
        self.select(track=tr.id, mark=m.id)
        self.changed()
        self.flash(f"Added “{m.title}” at {fmt_ts(m.t, 1)} on {tr.name} — double-click it to edit")
        if edit:
            self.edit_mark(m.id)
        return m

    def edit_mark(self, mark_id: str) -> None:
        tr, m = self.tl.find_mark(mark_id)
        if m is None:
            return
        self.preview.hide_preview()
        dlg = MarkDialog(self.window(), self.tl, tr, m, self._total())
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        if dlg.reset:
            self.tl.reset_auto(m.id)
        else:
            v = dlg.values()
            if m.is_auto:
                v = {"title": v["title"], "note": v["note"]}
            self.tl.update_mark(m.id, **v)
        self._pix_cache.clear()
        self.select(mark=m.id)
        self.changed()

    def delete_mark(self, mark_id: str) -> None:
        tr, m = self.tl.find_mark(mark_id)
        if m is None:
            return
        if tr.locked and not m.is_auto:
            self.flash(f"Track “{tr.name}” is locked.")
            return
        self.preview.hide_preview()
        self.tl.delete_mark(mark_id)
        if self.sel_mark == mark_id:
            self.sel_mark = None
        self.changed()
        self.flash(("Dismissed" if m.is_auto else "Deleted") + f" “{m.title or 'mark'}”" +
                   (" (restore it from the track's right-click menu)" if m.is_auto else ""))

    def recolor_mark(self, mark_id: str) -> None:
        tr, m = self.tl.find_mark(mark_id)
        if m is None or m.is_auto:
            return
        c = QColorDialog.getColor(QColor(m.color or tr.color), self.window(), "Mark colour")
        if c.isValid():
            self.tl.update_mark(mark_id, color=c.name())
            self.changed()

    # ------------------------------------------------------------------ preview
    def preview_pixmap(self, t: float) -> QPixmap:
        cols = self.colors_fn() or {}
        key = (round(t, 3), tuple(sorted(cols.items())))
        pm = self._pix_cache.get(key)
        if pm is None:
            dpr = self.devicePixelRatioF()
            pm = QPixmap(int(PREVIEW_SIZE.width() * dpr), int(PREVIEW_SIZE.height() * dpr))
            pm.setDevicePixelRatio(dpr)
            pm.fill(self.palette().color(QPalette.ColorRole.Base))
            qp = QPainter(pm)
            qp.setRenderHint(QPainter.RenderHint.Antialiasing)
            paint_court(qp, QRectF(0, 0, PREVIEW_SIZE.width(), PREVIEW_SIZE.height()), self.data, t, colors=cols,
                        labels=True, show_ball=True)
            if not self.data:
                qp.setPen(self.palette().color(QPalette.ColorRole.Text))
                qp.drawText(QRectF(0, 0, PREVIEW_SIZE.width(), PREVIEW_SIZE.height()),
                            Qt.AlignmentFlag.AlignCenter, "No 2D replay yet")
            qp.end()
            if len(self._pix_cache) > 64:
                self._pix_cache.clear()
            self._pix_cache[key] = pm
        return pm

    def show_preview(self, tr: tm.Track, m: tm.Mark, local_rect: QRect) -> None:
        sub = fmt_ts(m.t, 1) + (f" – {fmt_ts(m.end, 1)} ({m.duration:.1f} s)" if m.duration > 0 else "") \
            + f" · {tr.name}"
        if m.confidence is not None:
            sub += f" · confidence {m.confidence * 100:.0f}%"
        g = QRect(self.canvas.mapToGlobal(local_rect.topLeft()), local_rect.size())
        key = (m.id, m.t, m.duration, m.title, m.note, tr.name)
        self.preview.show_mark(key, m.title, sub, m.note, self.preview_pixmap(m.t), g)

    # ------------------------------------------------------------------ context menu
    def context_menu(self, pos: QPoint, gpos: QPoint) -> None:
        c = self.canvas
        menu = QMenu(self)
        p = QPointF(pos)
        hit = c.mark_hit(p) if pos.x() >= HEADER_W else None
        tr = c.track_at(pos.y())
        if hit:
            tr, m, _ = hit
            self.select(track=tr.id, mark=m.id)
            menu.addAction("Go to", lambda: self.seek(m.t))
            menu.addAction("Edit…", lambda: self.edit_mark(m.id)).setShortcut(QKeySequence(Qt.Key.Key_Return))
            if not m.is_auto:
                menu.addAction("Colour…", lambda: self.recolor_mark(m.id)).setEnabled(not tr.locked)
                if m.color:
                    menu.addAction("Use track colour", lambda: (self.tl.update_mark(m.id, color=None),
                                                                self.changed()))
                mv = menu.addMenu("Move to track")
                for t2 in self.tl.tracks:
                    if t2.is_auto or t2 is tr:
                        continue
                    a = mv.addAction(t2.name, lambda tid=t2.id: (self.tl.move_mark(m.id, m.t, tid),
                                                                  self.changed()))
                    a.setEnabled(not t2.locked and not tr.locked)
                mv.setEnabled(not tr.locked and not mv.isEmpty())
            else:
                menu.addAction(f"Copy to “{self._target_track().name}”", lambda: self._copy_auto(m))
                if m.event_id in self.tl.auto_overrides:
                    menu.addAction("Reset to detected text", lambda: (self.tl.reset_auto(m.id), self.changed()))
            menu.addSeparator()
            a = menu.addAction("Dismiss detection" if m.is_auto else "Delete mark", lambda: self.delete_mark(m.id))
            a.setShortcut(QKeySequence(Qt.Key.Key_Delete))
            a.setEnabled(m.is_auto or not tr.locked)
        elif tr is not None:
            self.select(track=tr.id)
            if pos.x() >= HEADER_W and not tr.is_auto:
                t = c.t_of(pos.x())
                menu.addAction(f"Add mark here ({fmt_ts(t, 1)})", lambda: self.add_mark(tr, t, edit=True)) \
                    .setEnabled(not tr.locked)
            if not tr.is_auto:
                menu.addAction(f"Add mark at playhead ({fmt_ts(self.playhead, 1)})",
                               lambda: self.add_mark(tr, self.playhead)).setEnabled(not tr.locked)
                menu.addSeparator()
            menu.addAction("Rename…", lambda: self.rename_track(tr))
            menu.addAction("Colour…", lambda: self.recolor_track(tr))
            a = menu.addAction("Visible", lambda: self.set_track_flag(tr, "visible", not tr.visible))
            a.setCheckable(True)
            a.setChecked(tr.visible)
            a = menu.addAction("Locked", lambda: self.set_track_flag(tr, "locked", not tr.locked))
            a.setCheckable(True)
            a.setChecked(tr.locked)
            i = self.tl.tracks.index(tr)
            menu.addAction("Move up", lambda: self.move_track(tr, -1)).setEnabled(i > 0)
            menu.addAction("Move down", lambda: self.move_track(tr, 1)).setEnabled(i < len(self.tl.tracks) - 1)
            if tr.is_auto:
                n = self.tl.dismissed_count()
                menu.addSeparator()
                menu.addAction(f"Restore dismissed detections ({n})", self._restore).setEnabled(n > 0)
            else:
                menu.addSeparator()
                menu.addAction("Delete track…", lambda: self.delete_track(tr))
            menu.addSeparator()
            menu.addAction("Add track…", self.add_track)
        else:
            menu.addAction(f"Add mark at playhead ({fmt_ts(self.playhead, 1)})", self.add_mark_at_playhead)
            menu.addAction("Add track…", self.add_track)
            menu.addSeparator()
            menu.addAction("Fit whole game", self.fit)
        menu.exec(gpos)

    def _restore(self) -> None:
        n = self.tl.restore_dismissed()
        self.changed()
        self.flash(f"Restored {n} detection{'s' * (n != 1)}.")

    def _copy_auto(self, m: tm.Mark) -> None:
        tr = self._target_track()
        nm = self.tl.add_mark(tr.id, m.t, title=m.title, note=m.note, duration=m.duration, color=m.color)
        self.select(track=tr.id, mark=nm.id)
        self.changed()
        self.flash(f"Copied “{m.title}” to {tr.name}.")
