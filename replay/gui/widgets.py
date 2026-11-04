"""Custom Qt widgets: zoomable video-frame view, clickable court diagram, 2D replay court."""

from __future__ import annotations

import bisect
import math
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import (QColor, QFont, QFontMetrics, QImage, QPainter, QPainterPath, QPen, QPixmap, QPolygonF,
                           QRadialGradient)
from PySide6.QtWidgets import (QColorDialog, QGraphicsPixmapItem, QGraphicsScene, QGraphicsView, QPushButton,
                               QSizePolicy, QWidget)

from replay import court
from replay.gui import motion
from replay.video import PRE_ROLL_S, probe

TEAM_FALLBACK = {"A": "#e5e7eb", "B": "#1f2a5a", "referee": "#334155", "unknown": "#9ca3af"}
FLOOR, FLOOR_LINE = QColor("#e9d3ae"), QColor("#ffffff")


# ------------------------------------------------------------------ frame access
class FrameGrabber:
    """Exact-frame access with a persistent capture: short forward steps don't re-seek."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        self.info = probe(path)
        self.cap: cv2.VideoCapture | None = None
        self.last_t: float | None = None

    def get(self, t: float) -> tuple[float, np.ndarray] | None:
        t = max(0.0, min(t, max(0.0, self.info.duration_s - 0.05)))
        half = 0.5 / self.info.fps
        if self.cap is None or self.last_t is None or t <= self.last_t - half or t > self.last_t + 3.0:
            if self.cap is not None:
                self.cap.release()
            self.cap = cv2.VideoCapture(self.path)
            if t - PRE_ROLL_S > 0:
                self.cap.set(cv2.CAP_PROP_POS_MSEC, (t - PRE_ROLL_S) * 1000.0)
            self.last_t = None
        while self.cap.grab():
            ft = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            self.last_t = ft
            if ft + half >= t:
                ok, img = self.cap.retrieve()
                return (ft, img) if ok else None
        self.last_t = None
        return None

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None


def bgr_to_pixmap(img: np.ndarray) -> QPixmap:
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    return QPixmap.fromImage(QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy())


# ------------------------------------------------------------------ frame view
class FrameView(QGraphicsView):
    """Shows one video frame. Left click = place point; wheel = zoom; right-drag = pan."""

    clicked = Signal(float, float)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.pix = QGraphicsPixmapItem()
        self.pix.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self.scene().addItem(self.pix)
        self.overlay_items: list = []
        self.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setBackgroundBrush(QColor("#000"))
        self.setCursor(Qt.CursorShape.CrossCursor)
        self._pan: QPointF | None = None
        self._fitted = False
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_frame(self, img: np.ndarray) -> None:
        self.pix.setPixmap(bgr_to_pixmap(img))
        self.scene().setSceneRect(QRectF(self.pix.pixmap().rect()))
        if not self._fitted:
            self.fit()

    def fit(self) -> None:
        self._fitted = True
        self.fitInView(self.pix, Qt.AspectRatioMode.KeepAspectRatio)

    def resizeEvent(self, e) -> None:
        super().resizeEvent(e)
        if self.transform().m11() <= self._fit_scale() * 1.001:
            self.fitInView(self.pix, Qt.AspectRatioMode.KeepAspectRatio)

    def _fit_scale(self) -> float:
        r = self.pix.pixmap().rect()
        if r.width() == 0:
            return 1.0
        return min(self.viewport().width() / r.width(), self.viewport().height() / r.height())

    def wheelEvent(self, e) -> None:
        f = 1.25 if e.angleDelta().y() > 0 else 0.8
        if self.transform().m11() * f < self._fit_scale():
            self.fitInView(self.pix, Qt.AspectRatioMode.KeepAspectRatio)
            return
        self.scale(f, f)

    def mousePressEvent(self, e) -> None:
        if e.button() == Qt.MouseButton.RightButton or e.button() == Qt.MouseButton.MiddleButton:
            self._pan = e.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return
        if e.button() == Qt.MouseButton.LeftButton:
            p = self.mapToScene(e.position().toPoint())
            r = self.pix.pixmap().rect()
            if 0 <= p.x() < r.width() and 0 <= p.y() < r.height():
                self.clicked.emit(p.x(), p.y())

    def mouseMoveEvent(self, e) -> None:
        if self._pan is not None:
            d = e.position() - self._pan
            self._pan = e.position()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - int(d.x()))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - int(d.y()))

    def mouseReleaseEvent(self, e) -> None:
        if self._pan is not None:
            self._pan = None
            self.setCursor(Qt.CursorShape.CrossCursor)

    def set_overlay(self, lines: list[np.ndarray] | None, points: dict[str, tuple[float, float]],
                    point_colors: dict[str, str] | None = None, preview: list[np.ndarray] | None = None,
                    point_notes: dict[str, str] | None = None) -> None:
        """lines: current calibration (yellow); preview: proposed snap (green); points coloured individually."""
        for it in self.overlay_items:
            self.scene().removeItem(it)
        self.overlay_items = []
        for group, colour in ((lines, "#ffe600"), (preview, "#22c55e")):
            if not group:
                continue
            pen = QPen(QColor(colour), 2.5 if colour == "#22c55e" else 2)
            pen.setCosmetic(True)
            for line in group:
                path, pen_down = QPainterPath(), False
                for x, y in line:
                    if not (np.isfinite(x) and np.isfinite(y)) or abs(x) > 2e4 or abs(y) > 2e4:
                        pen_down = False
                        continue
                    path.lineTo(x, y) if pen_down else path.moveTo(x, y)
                    pen_down = True
                self.overlay_items.append(self.scene().addPath(path, pen))
        font = QFont()
        font.setPointSize(9)
        for name, (x, y) in points.items():
            col = QColor((point_colors or {}).get(name, "#ff2d2d"))
            pen = QPen(col, 2)
            pen.setCosmetic(True)
            r = 6 / max(0.2, self.transform().m11())
            self.overlay_items.append(self.scene().addEllipse(x - r, y - r, 2 * r, 2 * r, pen))
            t = self.scene().addSimpleText(name + (point_notes or {}).get(name, ""), font)
            t.setBrush(col)
            t.setFlag(t.GraphicsItemFlag.ItemIgnoresTransformations)
            t.setPos(x + r, y - 3 * r)
            self.overlay_items.append(t)


# ------------------------------------------------------------------ court diagram (calibration)
class CourtDiagram(QWidget):
    """FIBA court with clickable calibration landmarks."""

    picked = Signal(str)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.selected: str | None = None
        self.placed: set[str] = set()
        self.setMinimumHeight(170)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("Click a landmark, then click the same spot on the video frame")

    def _map(self):
        pad = 12
        s = (self.width() - 2 * pad) / court.LENGTH
        return pad, s

    def sizeHint(self):
        from PySide6.QtCore import QSize
        return QSize(360, int(360 * (court.WIDTH + 2) / (court.LENGTH + 2)))

    def resizeEvent(self, e) -> None:
        pad, s = self._map()
        self.setFixedHeight(int(court.WIDTH * s + 2 * pad + 14))

    def _xy(self, p) -> QPointF:
        pad, s = self._map()
        return QPointF(pad + p[0] * s, pad + (court.WIDTH - p[1]) * s)

    def paintEvent(self, e) -> None:
        qp = QPainter(self)
        qp.setRenderHint(QPainter.RenderHint.Antialiasing)
        pad, s = self._map()
        qp.fillRect(QRectF(pad, pad, court.LENGTH * s, court.WIDTH * s), FLOOR)
        qp.setPen(QPen(QColor("#555"), 1.2))
        for line in court.court_polylines():
            path = QPainterPath(self._xy(line[0]))
            for p in line[1:]:
                path.lineTo(self._xy(p))
            qp.drawPath(path)
        for name, p in court.KEYPOINTS.items():
            sel = name == self.selected
            c = QColor("#2563eb") if sel else QColor("#16a34a") if name in self.placed else QColor("#f97316")
            qp.setBrush(c)
            qp.setPen(QPen(QColor("#111"), 1))
            r = 7 if sel else 5
            qp.drawEllipse(self._xy(p), r, r)
        qp.setPen(self.palette().text().color())
        qp.drawText(QRectF(0, self.height() - 14, self.width(), 14), Qt.AlignmentFlag.AlignCenter,
                    "near sideline (camera side)")

    def mousePressEvent(self, e) -> None:
        best, bd = None, 14.0
        for name, p in court.KEYPOINTS.items():
            q = self._xy(p)
            d = ((q.x() - e.position().x()) ** 2 + (q.y() - e.position().y()) ** 2) ** 0.5
            if d < bd:
                best, bd = name, d
        if best:
            self.selected = best
            self.update()
            self.picked.emit(best)

    def mouseMoveEvent(self, e) -> None:
        for name, p in court.KEYPOINTS.items():
            q = self._xy(p)
            if abs(q.x() - e.position().x()) < 8 and abs(q.y() - e.position().y()) < 8:
                self.setToolTip(name)
                return

    def enterEvent(self, e) -> None:
        self.setMouseTracking(True)


# ------------------------------------------------------------------ replay court
class ReplayCourt(QWidget):
    """Draws every track's court position (and the ball) at the current time.

    Display options: tracers (recent path), smoothing (moving average over a time window, display only),
    IDs, heading arrows, interpolated positions, referees, unclassified people, per-label colours.
    """

    qualityReady = Signal()  # the tracking-quality stretches (timeline lane) were computed
    trackSelected = Signal(object)  # track id of the clicked dot (None = selection cleared)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.data: dict | None = None
        self.t = 0.0
        self.trails = False
        self.trail_s = 2.0
        self.smooth_s = 0.0
        self.labels = True
        self.show_interp = True
        self.show_unknown = False
        self.show_referees = True
        self.show_ball = True
        self.hollow_interp = False
        self.headings = True
        self.hold_s = 4.0
        # off (default): clean broadcast-style view - solid dots for seen players, short gaps bridged by small faint
        # dots, nothing predicted far from the last sighting, nothing off court. On: every seen / predicted /
        # uncertain state (halo, soft dots, "?" marker) and the legend.
        self.confidence = False
        self.show_legend = True
        self.show_names = False  # roster names under the dots (data["roster_names"], roster hook)
        self.selected = None  # track id highlighted by a click: ring, trail and name
        self.select_trail_s = 6.0
        self.ball_trail = False  # fading trail behind the ball (last ball_trail_s seconds of playback)
        self.ball_trail_s = 2.0
        self._ball_hist = motion.TrailBuffer(self.ball_trail_s)
        self.paint_ms = 0.0  # last paint duration (ms), for measuring
        self.dots: list = []  # dots of the last paint: (track, x px, y px, radius px, text, is_number)
        self.colors = dict(TEAM_FALLBACK)
        self._drawn: dict = {}  # last drawn position per track (display-only speed limit, see paint_court prev)
        # background warm-up: ball, quality lane, then every track's path in small slices between frames
        self._warm = QTimer(self)
        self._warm.setInterval(5)
        self._warm.timeout.connect(self._warm_step)
        self._warm_stage = 0
        self._prep: threading.Thread | None = None
        self.setMinimumSize(360, 220)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_data(self, data: dict | None, colors: dict[str, str] | None = None) -> None:
        self.data = data
        self._drawn = {}
        if colors:
            self.colors = {**TEAM_FALLBACK, **colors}
        elif data:
            teams = data.get("game", {}).get("teams", {})
            self.colors = {**TEAM_FALLBACK, **{k: v["primary"] for k, v in teams.items()}}
        self._warm.stop()
        if data and data.get("t"):
            # nothing heavy here (a full game has thousands of tracks): paths (and the ball path, per minute)
            # are built lazily for the painted time; the quality lane (0.1-1 s on a full game) is computed in a
            # worker thread, the remaining paths in small slices between frames (_warm_step)
            m = motion.model_for(data)
            m.spans()
            m.meta()
            self._warm_stage = 0
            self._prep = threading.Thread(target=self._prepare, args=(m,), daemon=True)
            self._prep_at = time.perf_counter() + 0.4  # after the first paints (the GIL is shared)
            self._warm.start()
        self.update()

    @staticmethod
    def _prepare(m) -> None:
        try:
            m.quality()
            m.joint()  # joint extrapolation index (motion.JointDisplay), off the GUI thread
        except Exception as e:  # noqa: BLE001 - display extras must never take the view down
            print("replay view: background preparation failed:", type(e).__name__, e)

    def _paths_now(self):
        m = motion.model_for(self.data)
        smooth_n = int(round(self.smooth_s / m.step / 2)) if self.smooth_s > 0 else 0
        return m.paths(smooth_n, min(max(self.hold_s, 0.0), motion.BRIDGE_MAX))

    def _warm_step(self) -> None:
        """One slice (<= ~10 ms) of background preparation; stops itself when everything is built."""
        if not self.data or not self.data.get("t"):
            self._warm.stop()
            return
        m = motion.model_for(self.data)
        if self._warm_stage == 0:
            if self._prep.ident is None:
                if time.perf_counter() >= self._prep_at:
                    self._prep.start()
            elif not self._prep.is_alive():
                self._warm_stage = 1
                if m.quality_ready():
                    self.qualityReady.emit()
        done = self._paths_now().build_some(0.006, near=self.t, dense=self.trails)
        if done and self._warm_stage == 1:
            self._warm.stop()

    def dot_at(self, pos, team_only: bool = True):
        """(track, text, is_number) of the dot drawn nearest to widget position pos (within 1.6 radii), or None.
        Uses the last paint: no path is built."""
        best, bd = None, None
        for tr, x, y, rr, text, is_num in self.dots:
            if team_only and tr.get("team") not in ("A", "B"):
                continue
            d = (x - pos.x()) ** 2 + (y - pos.y()) ** 2
            if d <= (1.6 * rr) ** 2 and (bd is None or d < bd):
                best, bd = (tr, text, is_num), d
        return best

    def set_colors(self, colors: dict[str, str]) -> None:
        self.colors.update(colors)
        self.update()

    def set_time(self, t: float) -> None:
        if t == self.t:
            return  # paused: no need to redraw 60 times per second
        self.t = t
        self.update()

    def frame_index(self) -> tuple[int, float]:
        return frame_index(self.data, self.t)

    def calib_state(self) -> int:
        if not self.data or not self.data["t"]:
            return 0
        i, _ = self.frame_index()
        if self.t < self.data["t"][0] - 0.5 or self.t > self.data["t"][-1] + 0.5:
            return -1
        return self.data["calib"][i]

    def visible(self, tr: dict) -> bool:
        if tr["team"] == "referee":
            return self.show_referees
        return self.show_unknown or tr["team"] != "unknown"

    def paintEvent(self, e) -> None:
        t0 = time.perf_counter()
        qp = QPainter(self)
        qp.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing)
        dots: list = []
        info: dict = {}
        paint_court(qp, self.rect(), self.data, self.t, colors=self.colors, trails=self.trails,
                    trail_s=self.trail_s, smooth_s=self.smooth_s, labels=self.labels, show_interp=self.show_interp,
                    show_unknown=self.show_unknown, show_ball=self.show_ball, hollow_interp=self.hollow_interp,
                    hold_s=self.hold_s, show_referees=self.show_referees, headings=self.headings,
                    confidence=self.confidence, legend=self.confidence and self.show_legend and bool(self.data),
                    names=self.show_names, dots=dots, prev=self._drawn, out=info)
        self.dots = dots
        try:
            self._paint_extras(qp, info)
        except Exception as ex:  # noqa: BLE001 - extras must never take the view down
            print("replay view: overlay failed:", type(ex).__name__, ex)
        qp.end()
        self.paint_ms = (time.perf_counter() - t0) * 1000.0

    def event(self, e) -> bool:  # tooltip over a dot: number, name (roster.json), team, speed
        if e.type() == e.Type.ToolTip and self.data:
            from PySide6.QtWidgets import QToolTip
            tip = self.tooltip_text(e.pos())
            if tip:
                QToolTip.showText(e.globalPos(), tip, self)
            else:
                QToolTip.hideText()
                e.ignore()
            return True
        return super().event(e)

    # -------------------------------------------------------------- coach extras: selection, trails, tooltip
    def _geom(self) -> "_Geom":
        L, W = court_size(self.data)
        return _Geom(self.rect(), L, W)

    def _path_of(self, track_id):
        if not self.data or not self.data.get("t"):
            return None
        return self._paths_now().get(track_id)

    def player_label(self, tr: dict, text: str = "", is_num: bool = False) -> tuple[str, str, str]:
        """(number, name, team name) of a track for captions / tooltips."""
        info = ((self.data or {}).get("track_info") or {}).get(str(tr["id"])) or {}
        num = text if is_num else str(info.get("number") or "")
        name = ""
        if (self.data or {}).get("roster_names"):
            try:
                from replay import roster_names as rn
                name = rn.dot_name(self.data, tr["id"], tr.get("team"), num)
            except Exception:  # noqa: BLE001
                name = ""
        teams = ((self.data or {}).get("game") or {}).get("teams") or {}
        team = (teams.get(tr.get("team")) or {}).get("name") or {"referee": "Referee"}.get(tr.get("team"),
                                                                                           tr.get("team") or "")
        return num, name, team

    def tooltip_text(self, pos) -> str:
        hit = self.dot_at(pos, team_only=False)
        if hit is None:
            return ""
        tr, text, is_num = hit
        num, name, team = self.player_label(tr, text, is_num)
        who = " ".join(x for x in (f"#{num}" if num else "", name) if x) or f"Track {tr['id']}"
        speed = ""
        path = self._path_of(tr["id"])
        st = path.at(self.t, self.hold_s) if path is not None else []
        if st:
            speed = f" · {motion.speed_kmh(st[0][4], st[0][5]):.0f} km/h"
        return f"{who} – {team}{speed}"

    def select_track(self, track_id) -> None:
        self.selected = None if track_id is None else str(track_id)
        self.trackSelected.emit(self.selected)
        self.update()

    def mousePressEvent(self, e) -> None:
        if e.button() == Qt.MouseButton.LeftButton:
            hit = self.dot_at(e.position(), team_only=False)
            new = None if hit is None else str(hit[0]["id"])
            self.select_track(None if new == self.selected else new)
            e.accept()
            return
        super().mousePressEvent(e)

    def _paint_extras(self, qp: QPainter, info: dict) -> None:
        if not self.data or not self.data.get("t"):
            return
        g = self._geom()
        s = g.s
        # ball trail: positions of the drawn ball during the last ball_trail_s seconds of playback
        self._ball_hist.keep_s = self.ball_trail_s
        ball = info.get("ball")
        self._ball_hist.add(self.t, *(ball if ball else (None, None)))
        if self.ball_trail and self.show_ball and len(self._ball_hist.pts) >= 2:
            T, X, Y = zip(*self._ball_hist.pts)
            _fading_polyline(qp, g, T, X, Y, self.t, self.ball_trail_s, BALL_COLOR, max(1.5, s * 0.08))
        if self.selected is None:
            return
        hit = next((d for d in self.dots if str(d[0]["id"]) == self.selected), None)
        path = self._path_of(self.selected)
        tr = hit[0] if hit else next((x for x in self.data.get("tracks", []) if str(x["id"]) == self.selected), None)
        if tr is None:
            return
        col = QColor(self.colors.get(tr.get("team"), "#ffffff"))
        if path is not None:  # trail: where the selected player came from (not drawn over his own dot)
            qp.save()
            if hit is not None:
                clip = QPainterPath()
                clip.addRect(QRectF(self.rect()))
                hole = QPainterPath()
                hole.addEllipse(QPointF(hit[1], hit[2]), hit[3] + 1.0, hit[3] + 1.0)
                qp.setClipPath(clip.subtracted(hole))
            for T, X, Y in path.points(self.t - self.select_trail_s, self.t):
                _fading_polyline(qp, g, T, X, Y, self.t, self.select_trail_s, col, max(2.0, s * 0.12))
            qp.restore()
        if hit is None:
            return
        _tr, x, y, rr, text, is_num = hit
        ring = rr + max(3.0, s * 0.18)
        qp.setBrush(Qt.BrushStyle.NoBrush)
        qp.setPen(QPen(QColor(0, 0, 0, 160), max(3.0, s * 0.16)))
        qp.drawEllipse(QPointF(x, y), ring, ring)
        qp.setPen(QPen(QColor("#ffffff"), max(1.5, s * 0.08)))
        qp.drawEllipse(QPointF(x, y), ring, ring)
        num, name, team = self.player_label(tr, text, is_num)
        cap = " ".join(v for v in (f"#{num}" if num else "", name) if v) or f"Track {tr['id']}"
        font = _label_font(max(9, min(14, int(s * 0.38))))
        qp.setFont(font)
        fm = qp.fontMetrics()
        w, h = fm.horizontalAdvance(cap) + 10, fm.height() + 4
        bx = min(max(2.0, x - w / 2), self.width() - w - 2)
        by = y - ring - h - 3
        if by < 2:
            by = y + ring + 3
        box = QRectF(bx, by, w, h)
        qp.setPen(Qt.PenStyle.NoPen)
        qp.setBrush(QColor(17, 24, 39, 215))
        qp.drawRoundedRect(box, 4, 4)
        qp.setPen(QColor("#ffffff"))
        qp.drawText(box, Qt.AlignmentFlag.AlignCenter, cap)


def _fading_polyline(qp: QPainter, g: "_Geom", T, X, Y, t: float, span: float, col: QColor, width: float,
                     bands: int = 6) -> None:
    """Polyline through court points (times T) fading out with age (t - T) / span: a few alpha bands, one
    drawPolyline each (cheap at any length)."""
    if len(T) < 2:
        return
    pen = QPen(col, width)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    qp.setBrush(Qt.BrushStyle.NoBrush)
    pts = [g.xy(x, y) for x, y in zip(X, Y)]
    band = [min(bands - 1, max(0, int((1.0 - (t - tt) / max(span, 1e-6)) * bands))) for tt in T]
    j = 0
    while j < len(pts) - 1:
        b = band[j + 1]
        k = j + 1
        while k + 1 < len(pts) and band[k + 1] == b:
            k += 1
        pen.setColor(_alpha(col, 0.15 + 0.75 * (b + 0.5) / bands))
        qp.setPen(pen)
        qp.drawPolyline(QPolygonF(pts[j:k + 1]))
        j = k


# ------------------------------------------------------------------ shared replay maths + drawing
def frame_index(data: dict | None, t: float) -> tuple[int, float]:
    """Index of the last analysed frame at or before t, and the fraction towards the next one."""
    if not data or not data["t"]:
        return 0, 0.0
    ts = data["t"]
    i = max(0, bisect.bisect_right(ts, t) - 1)
    a = 0.0
    if i + 1 < len(ts) and t > ts[i]:
        a = min(1.0, (t - ts[i]) / max(1e-6, ts[i + 1] - ts[i]))
    return i, a


def _pos(tr: dict, i: int):
    j = i - int(tr.get("i0", 0))
    if j < 0 or j >= len(tr["x"]) or tr["x"][j] is None:
        return None
    return tr["x"][j] / 100.0, tr["y"][j] / 100.0, tr["f"][j]


_SMOOTH_CACHE: dict = {}  # (id(track), n) -> (track, xs, ys); cleared when the replay data changes
_SMOOTH_OWNER: list = [None]


def _smoothed(tr: dict, n: int) -> tuple[list, list]:
    """Moving average of a track's positions over the available samples within +/- n (computed once per track
    and window with cumulative sums; painting then only looks values up - the per-frame loops were the main
    cost of drawing with smoothing + tracers)."""
    key = (id(tr), n)
    hit = _SMOOTH_CACHE.get(key)
    if hit is not None and hit[0] is tr:
        return hit[1], hit[2]
    xs = np.array([np.nan if v is None else v for v in tr["x"]], float) / 100.0
    ys = np.array([np.nan if v is None else v for v in tr["y"]], float) / 100.0
    ok = np.isfinite(xs) & np.isfinite(ys)
    m = len(xs)
    cx = np.concatenate([[0.0], np.cumsum(np.where(ok, xs, 0.0))])
    cy = np.concatenate([[0.0], np.cumsum(np.where(ok, ys, 0.0))])
    cn = np.concatenate([[0], np.cumsum(ok)])
    idx = np.arange(m)
    lo, hi = np.clip(idx - n, 0, m), np.clip(idx + n + 1, 0, m)
    cnt = np.maximum(cn[hi] - cn[lo], 1)
    sx, sy = (cx[hi] - cx[lo]) / cnt, (cy[hi] - cy[lo]) / cnt
    if len(_SMOOTH_CACHE) > 20000:
        _SMOOTH_CACHE.clear()
    out = (tr, sx.tolist(), sy.tolist())
    _SMOOTH_CACHE[key] = out
    return out[1], out[2]


def _sample(tr: dict, i: int, smooth_n: int = 0):
    """Position of sample i, optionally averaged over the available samples within +/- smooth_n."""
    p = _pos(tr, i)
    if p is None or smooth_n <= 0:
        return p
    sx, sy = _smoothed(tr, smooth_n)
    j = i - int(tr.get("i0", 0))
    return sx[j], sy[j], p[2]


def track_position(tr: dict, i: int, a: float, smooth_n: int = 0):
    """(x, y, flag) of a track at frame i (+a towards i+1), optionally averaged over +/- smooth_n frames.

    Between samples the motion is a cubic (Catmull-Rom) curve through the neighbouring samples, so dots
    move fluidly at video frame rate even when only every 3rd frame was analysed: velocity is continuous
    instead of changing direction abruptly at every sample.
    """
    p = _sample(tr, i, smooth_n)
    if p is None:
        return None
    if a <= 0:
        return p
    q = _sample(tr, i + 1, smooth_n)
    if q is None:
        return p
    p0 = _sample(tr, i - 1, smooth_n)
    p3 = _sample(tr, i + 2, smooth_n)
    # tangents (per sample); one-sided at run ends
    m1 = ((q[0] - p0[0]) / 2, (q[1] - p0[1]) / 2) if p0 else (q[0] - p[0], q[1] - p[1])
    m2 = ((p3[0] - p[0]) / 2, (p3[1] - p[1]) / 2) if p3 else (q[0] - p[0], q[1] - p[1])
    a2, a3 = a * a, a * a * a
    h00, h10, h01, h11 = 2 * a3 - 3 * a2 + 1, a3 - 2 * a2 + a, -2 * a3 + 3 * a2, a3 - a2
    x = h00 * p[0] + h10 * m1[0] + h01 * q[0] + h11 * m2[0]
    y = h00 * p[1] + h10 * m1[1] + h01 * q[1] + h11 * m2[1]
    return x, y, p[2]


def track_velocity(tr: dict, i: int, fps: float, span_s: float = 0.2):
    """Velocity (m/s) at sample i from a central difference over about +/- span_s; None if unknown."""
    h = max(1, int(round(span_s * fps)))
    for hh in (h, max(1, h // 2), 1):
        a, b = _pos(tr, i - hh), _pos(tr, i + hh)
        if a is not None and b is not None:
            return (b[0] - a[0]) * fps / (2 * hh), (b[1] - a[1]) * fps / (2 * hh)
    a, b = _pos(tr, i), _pos(tr, i + 1)
    if a is not None and b is not None:
        return (b[0] - a[0]) * fps, (b[1] - a[1]) * fps
    a, b = _pos(tr, i - 1), _pos(tr, i)
    if a is not None and b is not None:
        return (b[0] - a[0]) * fps, (b[1] - a[1]) * fps
    return None


def _run_fade(tr: dict, i: int, a: float, fade_n: int, edges: bool = True) -> float:
    """1.0 inside a run of samples; ramps to 0 over fade_n samples at a run's first and last sample.

    edges=False only fades in at the start of the track's data (used together with "hold unseen", which
    bridges gaps and fades out by itself).
    """
    if fade_n <= 0:
        return 1.0
    back = fade_n
    for k in range(1, fade_n + 1):
        if _pos(tr, i - k) is None:
            back = k - 1
            break
    fwd = fade_n
    if edges:
        for k in range(1, fade_n + 1):
            if _pos(tr, i + k) is None:
                fwd = k - 1
                break
    if not edges and back < fade_n and i - back > int(tr.get("i0", 0)):
        back = fade_n  # a gap, not the start of the data: the hold/bridge handles it
    f_in = min(1.0, (back + a + 1) / (fade_n + 1))
    f_out = min(1.0, (fwd - a + 1) / (fade_n + 1))
    return max(0.0, min(f_in, f_out))


def _held_position(tr: dict, i: int, a: float, hold_n: int, smooth_n: int):
    """Position for display when the track has no sample at i: bridge to the next sighting or hold the last.

    Returns (x, y, flag, fade) with fade in (0, 1]; None if nothing within hold_n frames.
    """
    prev = nxt = None
    for k in range(1, hold_n + 1):
        if prev is None and _pos(tr, i - k) is not None:
            prev = (k, _sample(tr, i - k, smooth_n))
        if nxt is None and _pos(tr, i + k) is not None:
            nxt = (k, _sample(tr, i + k, smooth_n))
        if prev and nxt:
            break
    if prev and nxt:
        u = (prev[0] + a) / (prev[0] + nxt[0])
        u = u * u * (3 - 2 * u)  # ease in/out: no velocity kink where the bridge meets the real track
        p, q = prev[1], nxt[1]
        fade = max(0.35, 1.0 - min(prev[0] + a, nxt[0] - a) / hold_n)
        return p[0] + (q[0] - p[0]) * u, p[1] + (q[1] - p[1]) * u, 2, fade
    if prev:
        fade = 1.0 - (prev[0] + a) / (hold_n + 1)  # fades smoothly to nothing instead of popping off
        return (prev[1][0], prev[1][1], 2, fade) if fade > 0.02 else None
    return None


BALL_COLOR = QColor("#ffd400")  # bright yellow: distinct from any team / referee colour
COURT_SURROUND = QColor("#262a31")
COURT_FLOOR = QColor("#dcc095")
COURT_PAINT = QColor("#c8a676")
COURT_LINE = QColor("#fbf8f1")
RIM_COLOR = QColor("#e4572e")


def _lum(c: QColor) -> float:
    """Relative luminance (WCAG) of a colour, 0..1."""
    def ch(v: float) -> float:
        return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4
    return 0.2126 * ch(c.redF()) + 0.7152 * ch(c.greenF()) + 0.0722 * ch(c.blueF())


def _ink(c: QColor) -> QColor:
    """Black or white, whichever reads better on c."""
    return QColor("#111111") if _lum(c) > 0.30 else QColor("#ffffff")


def _alpha(c: QColor, a: float) -> QColor:
    c = QColor(c)
    c.setAlphaF(max(0.0, min(1.0, a)) * c.alphaF())
    return c


def _court_markings(L: float, W: float) -> tuple[list[list[tuple[float, float]]], list[list[tuple[float, float]]]]:
    """(solid polylines, dashed polylines) of a court L x W metres; basket-area markings are fixed (FIBA)."""
    C = court
    mid = W / 2
    hw = C.PAINT_HALF_WIDTH

    def arc(cx, cy, r, a0, a1, n=64):
        return [(cx + r * math.cos(a0 + (a1 - a0) * j / (n - 1)), cy + r * math.sin(a0 + (a1 - a0) * j / (n - 1)))
                for j in range(n)]

    solid = [[(0, 0), (L, 0), (L, W), (0, W), (0, 0)], [(L / 2, 0), (L / 2, W)],
             arc(L / 2, mid, C.CENTER_CIRCLE_R, 0, 2 * math.pi, 96)]
    dashed = []
    half_s, half_d = [], []
    half_s.append([(0, mid - hw), (C.FT_LINE_X, mid - hw), (C.FT_LINE_X, mid + hw), (0, mid + hw)])
    half_s.append(arc(C.FT_LINE_X, mid, C.FT_CIRCLE_R, -math.pi / 2, math.pi / 2, 48))  # outside the key
    half_d.append(arc(C.FT_LINE_X, mid, C.FT_CIRCLE_R, math.pi / 2, 3 * math.pi / 2, 48))  # inside: dashed
    side = min(C.THREE_SIDE_Y, mid - 0.5)
    a3 = math.asin(max(-1.0, min(1.0, (mid - side) / C.THREE_R)))
    three = [(0, side)] + arc(C.BASKET_X, mid, C.THREE_R, -a3, a3, 96) + [(0, W - side)]
    half_s.append(three)
    half_s.append(arc(C.BASKET_X, mid, C.NO_CHARGE_R, -math.pi / 2, math.pi / 2, 32))
    half_s.append([(1.2, mid - 0.9), (1.2, mid + 0.9)])  # backboard
    for ln in half_s:
        solid.append(ln)
        solid.append([(L - x, y) for x, y in ln])
    for ln in half_d:
        dashed.append(ln)
        dashed.append([(L - x, y) for x, y in ln])
    return solid, dashed


def court_size(data: dict | None) -> tuple[float, float]:
    """Court length/width to draw: from the replay data ("court" entry) or the current court module."""
    c = (data or {}).get("court") or {}
    try:
        return float(c.get("length") or court.LENGTH), float(c.get("width") or court.WIDTH)
    except (TypeError, ValueError):
        return court.LENGTH, court.WIDTH


class _Geom:
    """Court -> widget mapping for one paint."""

    __slots__ = ("L", "W", "pad", "s", "ox", "oy", "edge")

    def __init__(self, rect, L: float, W: float, r_px: float | None = None):
        self.L, self.W, self.pad = L, W, 0.9
        pad = self.pad
        self.s = min(rect.width() / (L + 2 * pad), rect.height() / (W + 2 * pad))
        self.ox = rect.x() + (rect.width() - (L + 2 * pad) * self.s) / 2
        self.oy = rect.y() + (rect.height() - (W + 2 * pad) * self.s) / 2
        r = max(4.5, self.s * 0.40) if r_px is None else r_px
        self.edge = max(0.0, pad - (r + 2) / max(self.s, 1e-6))

    def xy(self, x: float, y: float) -> QPointF:
        e = self.edge
        x = min(max(x, -e), self.L + e)  # keep dots inside the drawing even if a position is slightly off court
        y = min(max(y, -e), self.W + e)
        return QPointF(self.ox + (x + self.pad) * self.s, self.oy + (self.W - y + self.pad) * self.s)

    def raw(self, x: float, y: float) -> QPointF:  # no clamping (court lines)
        return QPointF(self.ox + (x + self.pad) * self.s, self.oy + (self.W - y + self.pad) * self.s)


def _draw_court(qp: QPainter, g: _Geom) -> None:
    L, W, s = g.L, g.W, g.s
    pad = g.pad

    def path_of(pts) -> QPainterPath:
        path = QPainterPath(g.raw(*pts[0]))
        for p in pts[1:]:
            path.lineTo(g.raw(*p))
        return path

    qp.setPen(Qt.PenStyle.NoPen)
    qp.setBrush(COURT_SURROUND)
    qp.drawRoundedRect(QRectF(g.ox, g.oy, (L + 2 * pad) * s, (W + 2 * pad) * s), 0.25 * s, 0.25 * s)
    qp.fillRect(QRectF(g.raw(0, W), g.raw(L, 0)), COURT_FLOOR)
    mid, hw = W / 2, court.PAINT_HALF_WIDTH
    for x0, x1 in ((0.0, court.FT_LINE_X), (L - court.FT_LINE_X, L)):
        qp.fillRect(QRectF(g.raw(x0, mid + hw), g.raw(x1, mid - hw)), COURT_PAINT)
    solid, dashed = _court_markings(L, W)
    lw = max(1.0, s * 0.06)
    pen = QPen(COURT_LINE, lw, Qt.PenStyle.SolidLine, Qt.PenCapStyle.FlatCap, Qt.PenJoinStyle.MiterJoin)
    qp.setBrush(Qt.BrushStyle.NoBrush)
    qp.setPen(pen)
    for ln in solid:
        qp.drawPath(path_of(ln))
    pen.setDashPattern([3.0, 3.0])
    qp.setPen(pen)
    for ln in dashed:
        qp.drawPath(path_of(ln))
    qp.setPen(QPen(RIM_COLOR, max(1.2, s * 0.06)))
    for bx in (court.BASKET_X, L - court.BASKET_X):
        qp.drawEllipse(g.raw(bx, mid), 0.23 * s, 0.23 * s)


_BG_CACHE: dict = {}


def _court_background(qp: QPainter, rect, L: float, W: float) -> None:
    """Draw the empty court from a cached pixmap (re-rendered only when the size / court changes): the court
    lines were ~40% of every paint."""
    dpr = _dpr(qp)
    w, h = int(math.ceil(rect.width())), int(math.ceil(rect.height()))
    if w <= 0 or h <= 0:
        return
    key = (w, h, round(L, 3), round(W, 3), dpr, court.PAINT_HALF_WIDTH, court.FT_LINE_X, court.THREE_R)
    pm = _BG_CACHE.get(key)
    if pm is None:
        if len(_BG_CACHE) > 6:
            _BG_CACHE.clear()
        pm = QPixmap(max(1, int(w * dpr)), max(1, int(h * dpr)))
        pm.setDevicePixelRatio(dpr)
        pm.fill(Qt.GlobalColor.transparent)
        p = QPainter(pm)
        p.setRenderHints(QPainter.RenderHint.Antialiasing)
        _draw_court(p, _Geom(QRectF(0, 0, rect.width(), rect.height()), L, W))
        p.end()
        _BG_CACHE[key] = pm
    qp.drawPixmap(QPointF(rect.x(), rect.y()), pm)


_FONT_CACHE: dict = {}


def _label_font(px: int) -> QFont:
    f = _FONT_CACHE.get(px)
    if f is None:
        f = QFont()
        f.setFamilies(["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"])
        f.setStyleHint(QFont.StyleHint.SansSerif)
        f.setPixelSize(max(6, px))
        f.setBold(True)
        _FONT_CACHE[px] = f
    return f


_HEAD_COLOR = QColor("#1b1d22")
_SHADOW = QColor(0, 0, 0, 70)


_SPRITES: dict = {}
_SPRITES_MAX = 2500  # dot bodies / halos are few; labels are many on a full game (one per track id) but cheap


def _dpr(qp: QPainter) -> float:
    try:
        return float(qp.device().devicePixelRatioF())
    except Exception:  # noqa: BLE001 - some paint devices have no ratio
        return 1.0


def _sprite_canvas(half: float, dpr: float) -> tuple[QPixmap, QPainter]:
    n = max(1, int(math.ceil(2 * half * dpr)))
    pm = QPixmap(n, n)
    pm.setDevicePixelRatio(dpr)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing)
    return pm, p


def _soft_fill(col: QColor) -> QColor:
    f = QColor(COURT_FLOOR)
    return QColor(int(f.red() * 0.6 + col.red() * 0.4), int(f.green() * 0.6 + col.green() * 0.4),
                  int(f.blue() * 0.6 + col.blue() * 0.4), 215)


def _dot_body(color: str, rr: float, style: str, s: float, dpr: float):
    """A dot without its label (style "solid", "hollow" or "soft"), rendered once: antialiased ellipses were
    most of the paint time, so they are never drawn per frame (labels are separate sprites: a full game has
    thousands of different ids, which made the old dot+label sprites miss the cache on every seek)."""
    key = ("body", color, round(rr, 1), style, round(s, 1), dpr)
    hit = _SPRITES.get(key)
    if hit is not None:
        return hit
    if len(_SPRITES) > _SPRITES_MAX:
        _SPRITES.clear()
    col = QColor(color)
    half = math.ceil(rr + max(2.0, s * 0.12) + 0.12 * s + 2)
    pm, p = _sprite_canvas(half, dpr)
    c = QPointF(half, half)
    if style == "soft":  # predicted / hidden: pale fill, dashed ring in the team colour, no shadow
        pen = QPen(col, max(1.8, s * 0.1), Qt.PenStyle.CustomDashLine)
        pen.setDashPattern([2.2, 1.4])
        p.setPen(pen)
        p.setBrush(_soft_fill(col))
    else:
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(_SHADOW)
        p.drawEllipse(QPointF(half + 0.07 * s, half + 0.09 * s), rr, rr)
        if style == "hollow":
            p.setPen(QPen(col, max(2.0, s * 0.12)))
            p.setBrush(_alpha(COURT_FLOOR, 0.6))
        else:
            p.setPen(QPen(_alpha(_ink(col), 0.9), max(1.2, s * 0.06)))
            p.setBrush(col)
    p.drawEllipse(c, rr, rr)
    p.end()
    _SPRITES[key] = (pm, half)
    return pm, half


def _label_ink(color: str, style: str, dim_text: bool) -> QColor:
    col = QColor(color)
    if style == "soft":
        return _alpha(_ink(_soft_fill(col)), 0.6 if dim_text else 0.85)
    tc = QColor("#111111") if style == "hollow" else _ink(col)
    return _alpha(tc, 0.62) if dim_text else tc


def _text_sprite(text: str, font: QFont, ink: QColor, rr: float, dpr: float):
    """A dot's label (centred in a 3rr x 2rr box), cached. Returns (pixmap, half width, half height)."""
    key = ("text", text, font.pixelSize(), ink.rgba(), round(rr, 1), dpr)
    hit = _SPRITES.get(key)
    if hit is not None:
        return hit
    if len(_SPRITES) > _SPRITES_MAX:
        _SPRITES.clear()
    w, h = 3 * rr, 2 * rr
    pm = QPixmap(max(1, int(math.ceil(w * dpr))), max(1, int(math.ceil(h * dpr))))
    pm.setDevicePixelRatio(dpr)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing)
    p.setPen(ink)
    p.setFont(font)
    p.drawText(QRectF(0, 0, w, h), Qt.AlignmentFlag.AlignCenter, text)
    p.end()
    hit = (pm, w / 2, h / 2)
    _SPRITES[key] = hit
    return hit


def _draw_dot(qp: QPainter, cx: float, cy: float, color: str, rr: float, style: str, text: str, font: QFont,
              s: float, dpr: float, dim_text: bool = False) -> None:
    """Dot body + label from cached sprites (painter opacity applies)."""
    pm, half = _dot_body(color, rr, style, s, dpr)
    qp.drawPixmap(QPointF(cx - half, cy - half), pm)
    if text:
        tpm, hw, hh = _text_sprite(text, font, _label_ink(color, style, dim_text), rr, dpr)
        qp.drawPixmap(QPointF(cx - hw, cy - hh), tpm)


def _dot_sprite(color: str, rr: float, hollow: bool, text: str, font: QFont, s: float, dpr: float,
                dim_text: bool = False):
    """A player's dot (shadow, fill, outline, label) as one pixmap (legend). Returns (pixmap, half size)."""
    return _composite_dot(color, rr, "hollow" if hollow else "solid", text, font, s, dpr, dim_text)


def _soft_dot_sprite(color: str, rr: float, text: str, font: QFont, s: float, dpr: float, dim_text: bool = False):
    """A predicted / hidden player: pale fill and a dashed ring in the team colour (no shadow: not "solid")."""
    return _composite_dot(color, rr, "soft", text, font, s, dpr, dim_text)


def _composite_dot(color, rr, style, text, font, s, dpr, dim_text):
    body, half = _dot_body(color, rr, style, s, dpr)
    if not text:
        return body, half
    pm, p = _sprite_canvas(half, dpr)
    _draw_dot(p, half, half, color, rr, style, text, font, s, dpr, dim_text)
    p.end()
    return pm, half


def _halo_sprite(color: str, radius: float, dpr: float):
    """Soft uncertainty halo (radial gradient) of the given radius in pixels (quantised)."""
    radius = max(4.0, round(radius / 2.0) * 2.0)
    key = ("halo", color, radius, dpr)
    hit = _SPRITES.get(key)
    if hit is not None:
        return hit
    if len(_SPRITES) > _SPRITES_MAX:
        _SPRITES.clear()
    half = math.ceil(radius + 1)
    pm, p = _sprite_canvas(half, dpr)
    col = QColor(color)
    grad = QRadialGradient(QPointF(half, half), radius)
    grad.setColorAt(0.0, _alpha(col, 0.22))
    grad.setColorAt(0.65, _alpha(col, 0.09))
    grad.setColorAt(1.0, _alpha(col, 0.0))
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(grad)
    p.drawEllipse(QPointF(half, half), radius, radius)
    p.setPen(QPen(_alpha(col, 0.25), 1.0, Qt.PenStyle.DotLine))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawEllipse(QPointF(half, half), radius * 0.66, radius * 0.66)  # ~1 sigma
    p.end()
    _SPRITES[key] = (pm, half)
    return pm, half


def _uncertain_sprite(rr: float, s: float, dpr: float):
    """Small "?" badge: team / identity uncertain."""
    rb = max(4.0, rr * 0.42)
    key = ("unc", round(rb, 1), dpr)
    hit = _SPRITES.get(key)
    if hit is not None:
        return hit
    half = math.ceil(rb + 1.5)
    pm, p = _sprite_canvas(half, dpr)
    p.setPen(QPen(QColor("#111111"), 1.0))
    p.setBrush(QColor("#fff7d6"))
    p.drawEllipse(QPointF(half, half), rb, rb)
    p.setPen(QColor("#111111"))
    p.setFont(_label_font(max(6, int(rb * 1.5))))
    p.drawText(QRectF(half - rb, half - rb, 2 * rb, 2 * rb), Qt.AlignmentFlag.AlignCenter, "?")
    p.end()
    _SPRITES[key] = (pm, half)
    return pm, half


QUALITY_COLORS = {motion.Q_GOOD: "#4caf7a", motion.Q_LINEUP: "#e8d36a", motion.Q_FEW: "#f2a33a",
                  motion.Q_CALIB_POOR: "#e8743b", motion.Q_CALIB_NONE: "#d64545"}


def quality_stretches(data: dict | None, compute: bool = False) -> list[tuple[float, float, int]]:
    """[(t0, t1, level)] tracking-quality stretches of replay data (timeline lane; cached per data). Without
    compute: [] until they were computed (ReplayCourt's background warm-up does it and emits qualityReady)."""
    if not data or not data.get("t"):
        return []
    m = motion.model_for(data)
    if not compute and not m.quality_ready():
        return []  # computed in the background by ReplayCourt (0.4-1 s on a full game)
    return m.quality()["stretches"]


_LEGEND_CACHE: dict = {}


def _legend_pixmap(colors: dict, dpr: float, px: int):
    """Small legend (cached): seen / predicted + halo / uncertain team or identity / jersey number vs track id."""
    key = (tuple(sorted(colors.items())), dpr, px)
    pm = _LEGEND_CACHE.get(key)
    if pm is not None:
        return pm
    if len(_LEGEND_CACHE) > 8:
        _LEGEND_CACHE.clear()
    font = _label_font(px)
    font_n = QFont(font)
    font_n.setBold(False)
    rows = ["seen", "predicted (halo = how unsure where)", "team / identity uncertain", "12 = jersey no., 47 = track id"]
    fm = QFontMetrics(font_n)
    rr = px * 0.62
    lh = int(px * 1.55)
    w = int(max(fm.horizontalAdvance(r) for r in rows) + rr * 2 + px * 2.2)
    h = int(lh * len(rows) + px * 0.8)
    pm = QPixmap(max(1, int(w * dpr)), max(1, int(h * dpr)))
    pm.setDevicePixelRatio(dpr)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor(20, 22, 27, 200))
    p.drawRoundedRect(QRectF(0, 0, w, h), 5, 5)
    col = QColor(colors.get("A", "#e5e7eb"))
    s = rr / 0.4
    for i, row in enumerate(rows):
        cy = px * 0.4 + lh * (i + 0.5)
        cx = px * 0.5 + rr
        if i == 0:
            spm, half = _dot_sprite(col.name(), rr, False, "", font, s, dpr)
            p.drawPixmap(QPointF(cx - half, cy - half), spm)
        elif i == 1:
            hpm, hh = _halo_sprite(col.name(), rr * 1.9, dpr)
            p.drawPixmap(QPointF(cx - hh, cy - hh), hpm)
            spm, half = _soft_dot_sprite(col.name(), rr, "", font, s, dpr)
            p.drawPixmap(QPointF(cx - half, cy - half), spm)
        elif i == 2:
            spm, half = _dot_sprite(col.name(), rr, False, "", font, s, dpr)
            p.drawPixmap(QPointF(cx - half, cy - half), spm)
            upm, uh = _uncertain_sprite(rr, s, dpr)
            p.drawPixmap(QPointF(cx - 0.72 * rr - uh, cy - 0.72 * rr - uh), upm)
        p.setPen(QColor("#e8e8e8"))
        p.setFont(font_n)
        x0 = px * 1.4 + 2 * rr if i < 3 else px * 0.5
        p.drawText(QRectF(x0, cy - lh / 2, w - x0, lh), Qt.AlignmentFlag.AlignVCenter, row)
    p.end()
    _LEGEND_CACHE[key] = pm
    return pm


def _short_name(name: str, max_len: int = 12) -> str:
    """Caption under a dot: the name, or its last word (surname) when long, elided."""
    name = " ".join(str(name).split())
    if len(name) > max_len and " " in name:
        name = name.split(" ")[-1]
    return name if len(name) <= max_len else name[:max_len - 1] + "…"


def _name_sprite(text: str, px: int, dpr: float):
    """Small name caption (dark rounded box, light text), cached. Returns (pixmap, width, height)."""
    key = ("name", text, px, dpr)
    hit = _SPRITES.get(key)
    if hit is not None:
        return hit
    if len(_SPRITES) > _SPRITES_MAX:
        _SPRITES.clear()
    f = QFont(_label_font(px))
    f.setBold(False)
    fm = QFontMetrics(f)
    w, h = fm.horizontalAdvance(text) + 6, fm.height() + 1
    pm = QPixmap(max(1, int(math.ceil(w * dpr))), max(1, int(math.ceil(h * dpr))))
    pm.setDevicePixelRatio(dpr)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor(20, 22, 27, 165))
    p.drawRoundedRect(QRectF(0, 0, w, h), 3, 3)
    p.setPen(QColor("#f4f4f4"))
    p.setFont(f)
    p.drawText(QRectF(0, 0, w, h), Qt.AlignmentFlag.AlignCenter, text)
    p.end()
    hit = (pm, w, h)
    _SPRITES[key] = hit
    return hit


def _ball_sprite(rb: float, est: bool, s: float, dpr: float):
    rb = round(rb * 4) / 4  # quantised: the size changes during a pass arc
    key = ("ball", rb, est, round(s, 1), dpr)
    hit = _SPRITES.get(key)
    if hit is not None:
        return hit
    half = math.ceil(rb * 2.0 + 2)
    pm, p = _sprite_canvas(half, dpr)
    c = QPointF(half, half)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(_alpha(BALL_COLOR, 0.22 if est else 0.35))
    p.drawEllipse(c, rb * 2.0, rb * 2.0)
    p.setPen(QPen(QColor("#111111"), max(1.2, s * 0.06), Qt.PenStyle.DashLine if est else Qt.PenStyle.SolidLine))
    p.setBrush(_alpha(BALL_COLOR, 0.8 if est else 1.0))
    p.drawEllipse(c, rb, rb)
    p.end()
    _SPRITES[key] = (pm, half)
    return pm, half


CLEAN_HIDDEN_MAX = 1.0  # s: default view draws an unseen player only this long from the nearest sighting
CLEAN_SOLID_S = 0.5  # s: ... and draws it as a normal solid dot this close to a sighting (a few missed frames)
OFF_COURT_M = 1.0  # m: default view hides positions further than this outside the court lines
DRAW_VMAX = 9.0  # m/s: fastest a dot moves on screen during continuous playback
JUMP_FADE_M = 3.0  # m: a bigger jump between two frames is shown as a fade-in at the new place
JUMP_FADE_S = 0.35  # s


def _limit_motion(ents: list, prev: dict, t: float) -> None:
    """Display-only smoothing between paints of continuous playback (prev: per track id (x, y, fade start))."""
    last = prev.get("_t")
    dt = None if last is None else t - last
    if dt is not None and abs(dt) < 1e-9:
        live = None  # same frame repainted: re-apply the stored positions only
    else:
        live = dt is not None and 0 < dt <= 0.3
    best: dict = {}
    for e in ents:
        k = str(e[0]["id"])
        if k not in best or e[4] > best[k][4]:
            best[k] = e
    new = {}
    for k, e in best.items():
        p = prev.get(k)
        f0 = None
        if p is not None and live is None:
            e[1], e[2], f0 = p
        elif p is not None and live:
            dx, dy = e[1] - p[0], e[2] - p[1]
            d = math.hypot(dx, dy)
            step = DRAW_VMAX * dt + 0.05
            f0 = p[2]
            if d > JUMP_FADE_M:
                f0 = t  # teleport: fade in at the new place instead of jumping / sliding across the court
            elif d > step:
                e[1], e[2] = p[0] + dx * step / d, p[1] + dy * step / d
        if f0 is not None and t - f0 < JUMP_FADE_S:
            e[4] *= max(0.15, (t - f0) / JUMP_FADE_S)
        elif f0 is not None:
            f0 = None
        new[k] = (e[1], e[2], f0)
    prev.clear()
    prev.update(new)
    prev["_t"] = t


def paint_court(qp: QPainter, rect, data: dict | None, t: float, colors: dict[str, str] | None = None,
                trails: bool = False, trail_s: float = 2.0, smooth_s: float = 0.0, labels: bool = True,
                show_interp: bool = True, show_unknown: bool = False, show_ball: bool = True,
                hollow_interp: bool = False, hold_s: float = 4.0, show_referees: bool = True,
                headings: bool = True, fade_s: float = 0.25, confidence: bool = False,
                legend: bool = False, names: bool = False, dots: list | None = None,
                prev: dict | None = None, out: dict | None = None) -> None:
    """Draw the court and everyone on it at time t into rect (used by the replay and timeline previews).

    Positions come from continuous trajectories (replay.gui.motion) evaluated at t, so the view can be drawn
    at any rate and stays smooth between the analysed samples.
    hold_s: players briefly out of view coast to a stop and fade out over this time; short gaps (up to
    min(hold_s, 1.5 s)) are bridged along a smooth path.
    fade_s: players appear / disappear with a short fade instead of popping.
    headings: a small wedge on each moving player's dot points where they are running.
    confidence: seen players solid; predicted / hidden ones soft (pale, dashed ring) with a halo sized by their
    position uncertainty; a "?" badge when the team / identity is uncertain; jersey numbers (or lineup slots)
    instead of track ids where known (ids then drawn lighter). legend: small key in the top-left corner.
    names: roster names (data["roster_names"], replay/roster_names.py) as a small caption under team players.
    dots: if a list is given, (track, x px, y px, radius px, text, is_number) of every drawn dot is appended.
    Without confidence (the default view): positions not seen for more than CLEAN_HIDDEN_MAX s and positions more
    than OFF_COURT_M off the court are not drawn; short hidden stretches are small faint dots; dots are spread apart
    more; labels are jersey numbers where known, else the track's label.
    out: if a dict is given, "ball" = (x m, y m) of the drawn ball (or None) is stored in it.
    prev: a dict kept by the caller between frames of continuous playback - dots never move faster than DRAW_VMAX on
    screen, and a dot whose position jumps more than JUMP_FADE_M fades in at the new place (display only).
    """
    clean = not confidence
    colors = {**TEAM_FALLBACK, **(colors or {})}
    L, W = court_size(data)
    g = _Geom(rect, L, W)
    s = g.s
    r = max(4.5, s * 0.40)
    qp.save()
    qp.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing)
    _court_background(qp, rect, L, W)

    if not data or not data.get("t") or not (data["t"][0] - 0.5 <= t <= data["t"][-1] + 0.5):
        qp.restore()
        return
    model = motion.model_for(data)
    smooth_n = int(round(smooth_s / model.step / 2)) if smooth_s > 0 else 0
    bridge = min(max(hold_s, 0.0), motion.BRIDGE_MAX)
    paths = model.paths(smooth_n, bridge)
    font = _label_font(max(7, int(r)))
    small = _label_font(max(6, int(max(7, int(r)) * 0.8)))

    def shown(tr: dict) -> bool:
        if tr["team"] == "referee":
            return show_referees
        return show_unknown or tr["team"] != "unknown"

    # ---- everyone's state at t (positions in metres)
    reach = max(hold_s, fade_s) + 0.1
    ents = []  # [tr, x, y, flag, alpha, vx, vy, path]
    tracks = model.tracks
    for k in model.alive(t, fade_s, reach):  # time index: no work for the thousands of other tracks
        tr = tracks[k]
        if not shown(tr):
            continue
        path = paths[k]  # built on first use
        if path.t1 + reach < t or path.t0 - fade_s > t:
            continue
        for x, y, flag, alpha, vx, vy in path.at(t, hold_s, fade_s):
            if alpha <= 0.01 or (flag == 2 and not show_interp):
                continue
            ents.append([tr, x, y, flag, alpha, vx, vy, path])
    faint: dict = {}  # id(entry) -> opacity factor of a briefly hidden player (default view)
    hidden_ids: set = set()  # team players not drawn because they are not seen (default view)
    if clean:
        keep = []
        for e in ents:
            x, y = e[1], e[2]
            if not (-OFF_COURT_M <= x <= L + OFF_COURT_M and -OFF_COURT_M <= y <= W + OFF_COURT_M):
                continue  # far off court: a projection error, not a player
            if e[3] == 2:
                u = e[7].unseen(t)
                if u > CLEAN_HIDDEN_MAX:
                    if e[0]["team"] in ("A", "B"):
                        hidden_ids.add(str(e[0]["id"]))
                    continue  # predicted / off-screen: only with "Confidence"
                if u > CLEAN_SOLID_S:  # a few missed frames stay solid; longer: small, faint, fading
                    faint[id(e)] = 0.55 * (1.0 - 0.45 * (u - CLEAN_SOLID_S) / (CLEAN_HIDDEN_MAX - CLEAN_SOLID_S))
            keep.append(e)
        ents = keep
    # dots never overlap / slide through each other (display only)
    pos = [[e[1], e[2]] for e in ents]
    motion.separate(pos, [e[4] * faint.get(id(e), 1.0) for e in ents], (2.05 if clean else 1.7) * r / max(s, 1e-6), iters=4 if clean else 2)
    for e, p in zip(ents, pos):
        e[1], e[2] = p[0], p[1]
    if prev is not None:
        _limit_motion(ents, prev, t)

    # ---- tracers: recent path, fading out towards the tail (a few alpha / width steps: one polyline per step;
    # wide antialiased lines are expensive, so the steps are few)
    if trails and trail_s > 0:
        qp.setBrush(Qt.BrushStyle.NoBrush)
        nb = 3
        ta = t - trail_s
        bounds = [ta + trail_s * k / nb for k in range(1, nb)]
        ox, oy, pad, Wc = g.ox, g.oy, g.pad, g.W
        pens = [QPen(QColor(0, 0, 0), max(1.0, s * (0.05 + 0.11 * (k + 0.5) / nb)), Qt.PenStyle.SolidLine,
                     Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin) for k in range(nb)]
        for e in ents:
            tr = e[0]
            pieces = e[7].points(ta, t)
            if not pieces:
                continue
            col = QColor(colors.get(tr["team"], "#ffffff"))
            for pi, (T, X, Y) in enumerate(pieces):
                pts = [QPointF(ox + (x + pad) * s, oy + (Wc - y + pad) * s) for x, y in zip(X, Y)]
                if pi == len(pieces) - 1 and T[-1] >= t - 0.5:  # the latest piece continues to the drawn dot
                    pts.append(g.xy(e[1], e[2]))
                    T = list(T) + [t]
                if len(pts) < 2:
                    continue
                cuts = [0] + [bisect.bisect_left(T, bt) for bt in bounds] + [len(pts)]
                for k in range(nb):
                    i0, i1 = max(0, cuts[k] - 1), cuts[k + 1]  # neighbouring steps share an end point
                    if i1 - i0 < 2:
                        continue
                    pen = pens[k]
                    pen.setColor(_alpha(col, (0.1 + 0.6 * (k + 0.5) / nb) * e[4]))
                    qp.setPen(pen)
                    qp.drawPolyline(pts[i0:i1])

    # ---- ball state (it follows its holder's drawn dot)
    ball_state = None
    bp = model.ball() if show_ball else None
    if bp is not None:
        drawn = {}
        for e in ents:
            k = str(e[0]["id"])
            if k not in drawn or e[4] > drawn[k][2]:
                drawn[k] = (e[1], e[2], e[4])
        def pos_of(hid, tt):
            if abs(tt - t) < 1e-9 and hid in drawn:
                return drawn[hid][0], drawn[hid][1]
            p = paths.get(hid)
            if p is None:
                return None
            st = p.at(tt, hold_s, fade_s)
            return (st[0][0], st[0][1]) if st else None

        ball_state = bp.at(t, pos_of)

    # ---- people: referees first (underneath), then players
    ents.sort(key=lambda e: 0 if e[0]["team"] == "referee" else 1)
    holder = ball_state[4] if ball_state and ball_state[2] > 0.01 else None
    hand = ball_state[5] if ball_state else 0.0
    holder_at = None
    dpr = _dpr(qp)
    meta = model.meta() if (confidence or names or labels) else None
    roster = bool(names and data.get("roster_names"))
    if roster:
        from replay import roster_names
    info = data.get("track_info") or {}

    def meta_number(tr):
        return (info.get(str(tr["id"])) or {}).get("number")
    any_numbers = bool(meta) and any(m["numbered"] or m["spans"] for m in meta.values())
    if confidence:  # uncertainty halos under every dot
        for tr, x, y, flag, fade, vx, vy, path in ents:
            if flag != 2:
                continue
            sig = path.uncertainty(t)
            rad = min(2.2 * s, max(r * 1.25, 1.5 * sig * s))
            hpm, hh = _halo_sprite(colors.get(tr["team"], "#ffffff"), rad, dpr)
            c = g.xy(x, y)
            qp.setOpacity(min(1.0, fade * 1.5))
            qp.drawPixmap(QPointF(c.x() - hh, c.y() - hh), hpm)
    for ent in ents:
        tr, x, y, flag, fade, vx, vy, _path = ent
        team = tr["team"]
        is_ref = team == "referee"
        small_dot = id(ent) in faint and holder != str(tr["id"])  # the ball holder stays solid
        if small_dot:
            fade *= faint[id(ent)]
        rr = r * (0.8 if is_ref else 1.0) * (0.78 if small_dot else 1.0)
        c = g.xy(x, y)
        cx, cy = c.x(), c.y()
        if headings:
            sp = math.hypot(vx, vy)
            if sp > 0.9:
                ux, uy = vx / sp, -vy / sp  # screen y points down
                ln = rr * (0.45 + 0.55 * min(1.0, (sp - 0.9) / 5.0))
                bw = rr * 0.62
                wedge = QPolygonF([QPointF(cx + ux * (rr + ln), cy + uy * (rr + ln)),
                                   QPointF(cx + ux * rr * 0.55 - uy * bw, cy + uy * rr * 0.55 + ux * bw),
                                   QPointF(cx + ux * rr * 0.55 + uy * bw, cy + uy * rr * 0.55 - ux * bw)])
                qp.setPen(Qt.PenStyle.NoPen)
                qp.setBrush(_HEAD_COLOR)
                qp.setOpacity(0.8 * fade * min(1.0, (sp - 0.9) / 0.6))
                qp.drawPolygon(wedge)
        dim = False
        is_num = False
        if meta is not None and not is_ref:
            mt = meta.get(str(tr["id"]))
            if mt is not None:
                text, is_num = motion.label_at(mt, t)
            else:
                text, is_num = str(tr["id"]), False
            dim = confidence and any_numbers and not is_num  # a track id / slot, not a jersey number
            label_text = text
            if not labels:
                text = ""
        else:
            mt = None
            text = "R" if is_ref else (str(tr["id"]) if labels else "")
            label_text = text
        tfont = small if (len(text) > 2 or dim) else font
        style = "soft" if confidence and flag == 2 else ("hollow" if flag == 2 and hollow_interp else "solid")
        qp.setOpacity(fade)
        _draw_dot(qp, cx, cy, colors.get(team, "#ffffff"), rr, style, text, tfont, s, dpr, dim)
        if confidence and mt is not None and mt["uncertain"]:
            upm, uh = _uncertain_sprite(rr, s, dpr)
            qp.drawPixmap(QPointF(cx - 0.72 * rr - uh, cy - 0.72 * rr - uh), upm)
        if names and roster and team in ("A", "B"):
            number = label_text if is_num else ((mt or {}).get("number") or meta_number(tr))
            nm = roster_names.dot_name(data, tr["id"], team, number or None)
            if nm:
                npm, nw, nh = _name_sprite(_short_name(nm), max(7, int(r * 0.85)), dpr)
                qp.drawPixmap(QPointF(cx - nw / 2, cy + rr + 1), npm)
        if dots is not None:
            dots.append((tr, cx, cy, rr, label_text, is_num))
        if holder is not None and holder == str(tr["id"]) and (holder_at is None or fade > holder_at[2]):
            holder_at = (c, rr, fade)
    qp.setOpacity(1.0)
    if holder_at is not None and hand > 0.01:
        c, rr, fade = holder_at
        qp.setPen(QPen(_alpha(BALL_COLOR, fade * min(1.0, hand * 1.5)), max(2.0, s * 0.1)))
        qp.setBrush(Qt.BrushStyle.NoBrush)
        ring = rr + max(2.5, s * 0.14)
        qp.drawEllipse(c, ring, ring)

    # ---- ball on top
    if ball_state is not None:
        bx, by, balpha, est, _h, hand, lift = ball_state
        if clean and not (-OFF_COURT_M <= bx <= L + OFF_COURT_M and -OFF_COURT_M <= by <= W + OFF_COURT_M):
            balpha = 0.0  # far off court: not drawn in the default view
        if out is not None:
            out["ball"] = (bx, by) if balpha > 0.01 else None
        if balpha > 0.01:
            c = g.xy(bx, by)
            if hand > 0:
                # in the holder's hands: sit on the rim of their dot instead of hiding their number
                c = QPointF(c.x() + 0.78 * r * hand, c.y() - 0.78 * r * hand)
            rb = max(3.5, s * 0.24) * (1.0 + lift)
            if lift > 0.02:  # in the air: ground shadow under the ball, the ball drawn a little higher
                qp.setPen(Qt.PenStyle.NoPen)
                qp.setBrush(_alpha(QColor(0, 0, 0, 90), balpha))
                qp.drawEllipse(c, rb * 0.7 / (1.0 + lift), rb * 0.45 / (1.0 + lift))
                c = QPointF(c.x(), c.y() - lift * s * 0.9)
            pm, half = _ball_sprite(rb, est, s, dpr)
            qp.setOpacity(balpha)
            qp.drawPixmap(QPointF(c.x() - half, c.y() - half), pm)
            qp.setOpacity(1.0)
    hidden_ids -= {str(e[0]["id"]) for e in ents}
    on_court = sum(1 for e in ents if e[0]["team"] in ("A", "B"))
    if hidden_ids and on_court < 10:  # never a silently empty court: say that players are not seen right now
        qp.setOpacity(0.85)
        qp.setFont(_label_font(max(9, min(12, int(s * 0.3)))))
        qp.setPen(QColor("#d1d5db"))
        h = max(12.0, g.pad * s * 0.8)
        qp.drawText(QRectF(rect.x() + 16, rect.y() + rect.height() - h - 2, rect.width() - 32, h),
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                    "Players out of view are not drawn (tick Confidence to show estimates)")
        qp.setOpacity(1.0)
    if legend:
        lpm = _legend_pixmap(colors, dpr, max(9, min(12, int(s * 0.32))))
        qp.setOpacity(0.92)
        qp.drawPixmap(QPointF(rect.x() + 6, rect.y() + 6), lpm)
        qp.setOpacity(1.0)
    qp.restore()


class ColorButton(QPushButton):
    changed = Signal(str)

    def __init__(self, color: str = "#ffffff", parent: QWidget | None = None):
        super().__init__(parent)
        self.setFixedWidth(56)
        self.set_color(color)
        self.clicked.connect(self._pick)

    def set_color(self, color: str) -> None:
        self.color = QColor(color).name()
        self.setToolTip(self.color)
        # Scoped selector: an unscoped style sheet would cascade into child widgets and dialogs.
        self.setObjectName("colorButton")
        self.setStyleSheet(f"QPushButton#colorButton {{ background:{self.color}; border:1px solid #888; "
                           "border-radius:4px; min-height:22px; }")

    def _pick(self) -> None:
        # Parent to the top-level window, not this button, so the button's style can't leak into the dialog.
        c = QColorDialog.getColor(QColor(self.color), self.window(), "Choose colour")
        if c.isValid():
            self.set_color(c.name())
            self.changed.emit(self.color)
