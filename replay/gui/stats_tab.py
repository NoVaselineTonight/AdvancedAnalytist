"""Stats tab: shot chart on the 2D court, team stats per period / game, lineup stats (replay/stats.py).

Every number is estimated from automatically detected events; clicking a shot, a stat row, a lineup or an event in
the "behind this number" list seeks the video (Replay tab). The detected events of the selected row can be
handed to the Playlists feature (``replay/playlists.py``): a live filter when the row maps to playlist
categories, else a fixed clip list. Export: CSV/JSON files
and a printable summary (PNG or PDF via Qt - no extra dependencies).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QImage, QPageSize, QPainter, QPainterPath, QPdfWriter, QPen
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QHeaderView,
                               QLabel, QListWidget, QListWidgetItem, QMessageBox, QPushButton, QSizePolicy,
                               QSplitter, QTableWidget, QTableWidgetItem, QTabWidget, QToolTip, QVBoxLayout, QWidget)

from replay import stats as S
from replay.gui.helpicon import HelpIcon, help_row
from replay.gui.tooltips import HELP

FALLBACK = {"A": "#e4572e", "B": "#6c3fc5"}
SEEK_ROLE = Qt.ItemDataRole.UserRole
ITEMS_ROLE = Qt.ItemDataRole.UserRole + 1


NA = "n/a"  # unknown / not detected - never shown as a confident 0


def is_sparse(st: dict | None) -> bool:
    """Same test as the coach report's "very little was detected" banner: no points and fewer than 5 shots."""
    if not st:
        return False
    pts = sum((st.get("game", {}).get(k, {}).get("points") or 0) for k in S.TEAMS)
    return pts == 0 and len(st.get("shots") or []) < 5


def _pct(v: float | None) -> str:
    return NA if v is None else f"{100 * v:.0f}%"


def fmt_value(v: Any, kind: str, sparse: bool = False) -> str:
    if v is None or (sparse and isinstance(v, (int, float)) and not isinstance(v, bool) and v == 0):
        return NA  # unknown, or a zero that only means "nothing was detected"
    if kind == "pct":
        return _pct(v)
    if kind == "time":
        return S.fmt_clock(v)
    if kind == "float":
        return f"{v:.2f}" if isinstance(v, float) and v < 10 else f"{v:.1f}" if isinstance(v, float) else str(v)
    return str(v)


# ====================================================================================== drawing
class CourtMap:
    """Court metres <-> pixels inside a rect (left baseline on the left, near sideline at the bottom)."""

    def __init__(self, rect: QRectF, L: float, W: float, pad: float = 0.6):
        self.L, self.W, self.pad = L, W, pad
        self.s = min(rect.width() / (L + 2 * pad), rect.height() / (W + 2 * pad))
        self.ox = rect.x() + (rect.width() - (L + 2 * pad) * self.s) / 2
        self.oy = rect.y() + (rect.height() - (W + 2 * pad) * self.s) / 2

    def xy(self, x: float, y: float) -> QPointF:
        return QPointF(self.ox + (x + self.pad) * self.s, self.oy + (self.W - y + self.pad) * self.s)


def draw_court(qp: QPainter, m: CourtMap, geo: dict) -> None:
    qp.setPen(Qt.PenStyle.NoPen)
    qp.setBrush(QColor("#e9d3ae"))
    qp.drawRect(QRectF(m.xy(0, m.W), m.xy(m.L, 0)))
    qp.setPen(QPen(QColor("#5b4a32"), max(1.0, m.s * 0.06)))
    qp.setBrush(Qt.BrushStyle.NoBrush)
    for line in S.court_lines(geo):
        path = QPainterPath(m.xy(*line[0]))
        for p in line[1:]:
            path.lineTo(m.xy(*p))
        qp.drawPath(path)
    qp.setBrush(QColor("#5b4a32"))
    for bx in (geo["basket_x"], m.L - geo["basket_x"]):
        qp.drawEllipse(m.xy(bx, geo.get("basket_y") or m.W / 2), m.s * 0.23, m.s * 0.23)


def draw_shots(qp: QPainter, m: CourtMap, shots: list[dict], colors: dict, selected: str | None = None,
               r_px: float | None = None) -> None:
    """Made = filled, missed = hollow ring, result unknown = dashed ring; zone '?' shots drawn lighter."""
    r = r_px or max(3.5, m.s * 0.28)
    for s in shots:
        if s.get("x") is None or s.get("y") is None:
            continue
        c = QColor(colors.get(s.get("team")) or "#64748b")
        p = m.xy(float(s["x"]), float(s["y"]))
        w = max(1.5, r * 0.45)
        if s["outcome"] == "made":
            qp.setPen(QPen(QColor("#111827"), 1.0))
            qp.setBrush(c)
        elif s["outcome"] == "missed":
            qp.setPen(QPen(c, w))
            qp.setBrush(Qt.BrushStyle.NoBrush)
        else:
            qp.setPen(QPen(c, max(1.0, w * 0.6), Qt.PenStyle.DotLine))
            qp.setBrush(Qt.BrushStyle.NoBrush)
        qp.drawEllipse(p, r, r)
        if selected and s.get("id") == selected:
            qp.setPen(QPen(QColor("#0ea5e9"), 2.0))
            qp.setBrush(Qt.BrushStyle.NoBrush)
            qp.drawEllipse(p, r + 4, r + 4)


class ShotChart(QWidget):
    """2D court with the (filtered) field-goal attempts. Click a shot -> shotClicked(shot dict)."""

    shotClicked = Signal(dict)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.geo: dict | None = None
        self.shots: list[dict] = []
        self.colors: dict = dict(FALLBACK)
        self.selected: str | None = None
        self.setMouseTracking(True)
        self.setMinimumSize(320, 190)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def sizeHint(self) -> QSize:
        return QSize(560, 330)

    def set_data(self, geo: dict | None, shots: list[dict], colors: dict) -> None:
        self.geo, self.shots, self.colors = geo, shots, {**FALLBACK, **{k: v for k, v in colors.items() if v}}
        self.update()

    def _map(self) -> CourtMap | None:
        if not self.geo:
            return None
        return CourtMap(QRectF(self.rect()).adjusted(4, 4, -4, -4), self.geo["length"], self.geo["width"])

    def paintEvent(self, _e) -> None:
        qp = QPainter(self)
        qp.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing)
        m = self._map()
        if m is None:
            qp.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No statistics yet")
            return
        draw_court(qp, m, self.geo)
        draw_shots(qp, m, self.shots, self.colors, self.selected)

    def _hit(self, pos) -> dict | None:
        m = self._map()
        if m is None:
            return None
        best, bd = None, max(8.0, m.s * 0.45)
        for s in self.shots:
            if s.get("x") is None:
                continue
            p = m.xy(float(s["x"]), float(s["y"]))
            d = math.hypot(p.x() - pos.x(), p.y() - pos.y())
            if d < bd:
                best, bd = s, d
        return best

    def mouseMoveEvent(self, e) -> None:
        s = self._hit(e.position())
        if s is None:
            QToolTip.hideText()
            self.setCursor(Qt.CursorShape.ArrowCursor)
            return
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        QToolTip.showText(e.globalPosition().toPoint(), f"{S.fmt_clock(s['t'])}  {s['title']}"
                          + (f"\nconfidence {100 * s['confidence']:.0f}%" if s.get("confidence") is not None else ""),
                          self)

    def mousePressEvent(self, e) -> None:
        s = self._hit(e.position())
        if s is not None:
            self.selected = s["id"]
            self.update()
            self.shotClicked.emit(s)


# ====================================================================================== printable summary
def render_summary(qp: QPainter, page: QRectF, st: dict, colors: dict | None = None, title: str = "") -> None:
    """One printable page: score (+ truth), team stat table, shot charts per team, zones, top lineups, notes."""
    colors = {**FALLBACK, **{k: v for k, v in (colors or {}).items() if v}}
    nm = {k: st["teams"][k]["name"] for k in S.TEAMS}
    qp.save()
    qp.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing)
    qp.fillRect(page, QColor("white"))
    W = page.width()
    u = W / 100.0  # layout unit
    x0, y = page.x() + 4 * u, page.y() + 4 * u
    inner = W - 8 * u

    def font(sz: float, bold: bool = False) -> QFont:
        f = QFont("Arial")
        f.setPixelSize(max(6, int(sz * u)))
        f.setBold(bold)
        return f

    def text(x, yy, w, h, s, f, align=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, col="#111827"):
        qp.setFont(f)
        qp.setPen(QColor(col))
        qp.drawText(QRectF(x, yy, w, h), int(align) | int(Qt.TextFlag.TextWordWrap), s)

    text(x0, y, inner, 4 * u, title or "Game statistics", font(3.0, True))
    y += 4.5 * u
    g = st["game"]
    sc = f"{nm['A']}  {g['A']['points']} – {g['B']['points']}  {nm['B']}   (detected)"
    if st["score"].get("unattributed_points"):
        sc += f"   + {st['score']['unattributed_points']} pts without team"
    text(x0, y, inner, 3.5 * u, sc, font(2.4, True))
    y += 3.6 * u
    tr = st["score"].get("truth")
    if tr and all(tr.get(k) is not None for k in S.TEAMS):
        text(x0, y, inner, 3 * u, f"Score-bar truth: {nm['A']} {tr['A']:.0f} – {tr['B']:.0f} {nm['B']}  ·  detected "
                                  f"{st['score'].get('total_pct_of_truth')} % of the points", font(1.8), col="#374151")
        y += 3.2 * u
    text(x0, y, inner, 3 * u, "Estimated from automatically detected events – not an official box score.",
         font(1.6), col="#b45309")
    y += 3.6 * u

    # team table (left) + shot charts (right)
    rows = [r for r in S.STAT_ROWS if r[0] not in ("possessions_est",)]
    rh = 2.35 * u
    cw = [22 * u, 10 * u, 10 * u]
    ty = y
    qp.setPen(QColor("#9ca3af"))
    text(x0, ty, cw[0], rh, "", font(1.6, True))
    for j, k in enumerate(S.TEAMS):
        text(x0 + cw[0] + j * cw[1], ty, cw[1], rh, nm[k][:14], font(1.5, True),
             Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, colors.get(k, "#111827"))
    ty += rh
    for i, (key, label, kind) in enumerate(rows):
        if i % 2 == 0:
            qp.fillRect(QRectF(x0, ty, sum(cw), rh), QColor("#f3f4f6"))
        text(x0 + 0.5 * u, ty, cw[0], rh, label.strip(), font(1.45))
        for j, k in enumerate(S.TEAMS):
            text(x0 + cw[0] + j * cw[1], ty, cw[1] - 0.5 * u, rh, fmt_value(g[k].get(key), kind, is_sparse(st)), font(1.45),
                 Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        ty += rh
    cx = x0 + sum(cw) + 3 * u
    cwid = x0 + inner - cx
    chh = cwid * (st["court"]["width"] + 1.2) / (st["court"]["length"] + 1.2)
    cy = y
    for k in S.TEAMS:
        text(cx, cy, cwid, 2.6 * u, f"{nm[k]} shots (filled = made, ring = missed)", font(1.5, True),
             col=colors.get(k, "#111827"))
        cy += 2.6 * u
        m = CourtMap(QRectF(cx, cy, cwid, chh), st["court"]["length"], st["court"]["width"])
        draw_court(qp, m, st["court"])
        draw_shots(qp, m, [s for s in st["shots"] if s.get("team") == k], colors, r_px=max(2.5, 0.55 * u))
        cy += chh + 0.6 * u
        z = st["zones"][k]
        ztxt = "   ".join(f"{S.ZONE_TITLES[zz]} {z[zz]['fgm']}/{z[zz]['fga']} ({_pct(z[zz]['pct'])})"
                           for zz in S.ZONES)
        text(cx, cy, cwid, 2.4 * u, ztxt, font(1.3), col="#374151")
        cy += 3.2 * u
    y = max(ty, cy) + 1.5 * u

    # periods
    if len(st["periods"]) > 1:
        line = "Points per period:  " + "   ".join(
            f"{p['label']}: {p['A']['points']}–{p['B']['points']}" for p in st["periods"])
        text(x0, y, inner, 2.8 * u, line, font(1.6))
        y += 3.2 * u
    # lineups
    lu = st.get("lineups") or {}
    text(x0, y, inner, 2.8 * u, "Lineups (top by minutes)", font(1.8, True))
    y += 3 * u
    if not lu.get("available"):
        text(x0, y, inner, 2.6 * u, lu.get("reason", "Lineups unavailable."), font(1.5), col="#6b7280")
        y += 3 * u
    else:
        for k in S.TEAMS:
            for r in [r for r in lu["rows"] if r["team"] == k][:4]:
                text(x0, y, inner, 2.4 * u,
                     f"{nm[k]}: {' '.join(r['players'])}{'  (?)' if r.get('duplicate_numbers') else ''} — "
                     f"{r['minutes']} min, +/- {r['plus_minus']:+d}, ORtg {r['off_rating'] if r['off_rating'] is not None else '–'}"
                     f", DRtg {r['def_rating'] if r['def_rating'] is not None else '–'}", font(1.4))
                y += 2.5 * u
        text(x0, y, inner, 2.4 * u, lu.get("note", ""), font(1.3), col="#6b7280")
        y += 3 * u
    # notes
    y += 1 * u
    for n in st.get("notes") or []:
        h = 2.3 * u * (1 + len(n) // 140)
        if y + h > page.bottom() - 2 * u:
            break
        text(x0, y, inner, h, "• " + n, font(1.3), col="#4b5563")
        y += h + 0.3 * u
    qp.restore()


def save_summary(path: str | Path, st: dict, colors: dict | None = None, title: str = "") -> Path:
    """Write the printable summary as PDF (A4) or PNG (by extension)."""
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        w = QPdfWriter(str(path))
        w.setPageSize(QPageSize(QPageSize.PageSizeId.A4))
        w.setResolution(150)
        w.setTitle(title or "Game statistics")
        qp = QPainter(w)
        try:
            render_summary(qp, QRectF(0, 0, w.width(), w.height()), st, colors, title)
        finally:
            qp.end()
        return path
    img = QImage(1240, 1754, QImage.Format.Format_RGB32)
    img.fill(QColor("white"))
    qp = QPainter(img)
    try:
        render_summary(qp, QRectF(0, 0, img.width(), img.height()), st, colors, title)
    finally:
        qp.end()
    if not img.save(str(path)):
        raise OSError(f"could not write {path}")
    return path


# ====================================================================================== playlists hand-off
# stats item list -> playlist categories (replay/playlists.py CATEGORIES): such playlists are live filters that
# the Playlists panel can rebuild; other lists (zones, lineup stints) become fixed clip lists
PLAYLIST_CATEGORIES = {
    "baskets": ["made"], "baskets_2": ["made"], "baskets_3": ["made"], "baskets_ft": ["made"],
    "fg_made": ["made"], "fg3_made": ["made"], "fg_attempts": ["shot"], "fg3_attempts": ["shot"],
    "ft_made": ["free_throw"], "ft_attempts": ["free_throw"], "possessions": ["possession"],
    "turnovers": ["turnover"], "steals": ["steal"], "orb": ["off_rebound"], "drb": ["def_rebound"],
    "fast_breaks": ["fast_break"], "fast_break_baskets": ["fast_break"],
}


def playlist_module() -> Any | None:
    """replay.playlists when the Playlists feature is installed (with the API used here), else None."""
    try:
        from replay import playlists
    except Exception:  # noqa: BLE001 - feature not installed
        return None
    need = ("Playlist", "PlaylistBook", "Filter", "Clip", "make_playlist", "collect")
    return playlists if all(hasattr(playlists, n) for n in need) else None


def make_stats_playlist(pl: Any, name: str, items: list[dict], flt: dict | None, catalog_fn: Callable[[], Any]
                        ) -> Any:
    """A playlist for the selected stats row: a live filter (categories / team / time range) when the row maps
    to playlist categories, else fixed clips around the listed moments."""
    if flt:
        return pl.make_playlist(catalog_fn(), name, pl.Filter(**flt))
    clips = []
    for i in sorted((i for i in items if i.get("t") is not None), key=lambda i: i["t"]):
        t0 = max(0.0, float(i["t"]) - getattr(pl, "DEFAULT_PRE_S", 4.0))
        t1 = float(i.get("t1") or i["t"]) + getattr(pl, "DEFAULT_POST_S", 2.0)
        if clips and t0 <= clips[-1].t1:
            clips[-1].t1 = max(clips[-1].t1, t1)
            clips[-1].titles.append(i["title"])
            continue
        clips.append(pl.Clip(t0=t0, t1=t1, title=i["title"], titles=[i["title"]], team=i.get("team"),
                             note="from the Stats tab (fixed list)"))
    # a filter that matches nothing: "Rebuild" in the Playlists panel would empty a fixed list, so say so
    return pl.Playlist(name=name, filter=pl.Filter(text="(fixed list from the Stats tab - do not rebuild)"),
                       clips=clips)


# ====================================================================================== tab
class StatsTab(QWidget):
    def __init__(self, win: Any):
        super().__init__()
        self.win = win
        self.p = None
        self.st: dict | None = None
        self._sig: tuple | None = None
        self._items: list[dict] = []
        self._items_name = ""
        self._items_filter: dict | None = None
        self._built = False
        self._lay = QVBoxLayout(self)

    def _ensure_ui(self) -> None:
        """Widgets are built on first use (showing the tab), not with the main window."""
        if self._built:
            return
        self._built = True
        lay = self._lay
        self.msg = QLabel()
        self.msg.setWordWrap(True)
        self.msg.setTextFormat(Qt.TextFormat.RichText)
        lay.addWidget(self.msg)

        from replay.gui.flowlayout import FlowLayout
        bar_w = QWidget()  # filters + buttons wrap onto a second line in a narrow window
        bar = FlowLayout(bar_w)
        sp = bar_w.sizePolicy()
        sp.setHeightForWidth(True)
        bar_w.setSizePolicy(sp)
        bar.addStretch = lambda *_a: None  # (QHBoxLayout API used below; a flow layout needs no stretch)
        self.period = QComboBox()
        self.team = QComboBox()
        self.player = QComboBox()
        self.outcome = QComboBox()
        self.outcome.addItem("All shots", None)
        self.outcome.addItem("Made", "made")
        self.outcome.addItem("Missed", "missed")
        for lbl, w in (("Period", self.period), ("Team", self.team), ("Shooter", self.player), ("", self.outcome)):
            if lbl:
                bar.addWidget(QLabel(lbl))
            bar.addWidget(w)
            w.currentIndexChanged.connect(self._filters_changed)
        bar.addStretch(1)
        self.goto = QCheckBox("Show in Replay on click")
        self.goto.setChecked(True)
        self.goto.setToolTip("Clicking a shot, row or event seeks the video and switches to the Replay tab")
        bar.addWidget(help_row(self.goto, HELP["stats_goto"], "Show in Replay on click", stretch=False))
        b_ref = QPushButton("Refresh")
        b_ref.clicked.connect(lambda: self.refresh(force=True))
        b_coach = QPushButton("Coach report…")  # one-page post-game report + PDF (replay/gui/coach_report.py)
        b_coach.setToolTip("Box score, team comparison, shot charts, lineups and runs on one page; click a number "
                           "to watch those moments; save as PDF; player pages")
        b_coach.clicked.connect(lambda: __import__("replay.gui.coach_report", fromlist=["x"]).open_report(
            self.win, self.p))
        bar.addWidget(b_coach)
        b_exp = QPushButton("Export CSV/JSON…")
        b_exp.clicked.connect(self.export_tables)
        b_pdf = QPushButton("Printable summary…")
        b_pdf.setToolTip("Save a one-page summary as PDF or PNG")
        b_pdf.clicked.connect(self.export_summary)
        for b in (b_ref, b_exp, b_pdf):
            bar.addWidget(b)
        # the grey disclaimer paragraph that used to sit under the tables: behind this ? (filled by _refresh)
        self.notes = HelpIcon("", "About these numbers")
        bar.addWidget(self.notes)
        lay.addWidget(bar_w)

        split = QSplitter(Qt.Orientation.Horizontal)
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        self.chart = ShotChart()
        self.chart.shotClicked.connect(lambda s: self.seek(s["t"]))
        lv.addWidget(self.chart, 3)
        self.chart_note = QLabel()
        self.chart_note.setWordWrap(True)
        self.chart_note.setStyleSheet("color: gray")
        lv.addWidget(help_row(self.chart_note, HELP["stats_shots"], "Shot chart key"))
        self.zones = QTableWidget(0, 4)
        self.zones.setHorizontalHeaderLabels(["Zone", "Made", "Att.", "FG%"])
        self._table_style(self.zones)
        self.zones.itemClicked.connect(self._zone_clicked)
        lv.addWidget(self.zones, 2)
        split.addWidget(left)

        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        self.inner = QTabWidget()
        self.team_table = QTableWidget(0, 2)
        self._table_style(self.team_table)
        self.team_table.cellClicked.connect(self._team_cell)
        self.inner.addTab(self.team_table, "Team stats")
        lw = QWidget()
        lvv = QVBoxLayout(lw)
        lvv.setContentsMargins(0, 0, 0, 0)
        self.lineup_note = QLabel()  # why lineups are unavailable (short)
        self.lineup_note.setWordWrap(True)
        lineup_row = help_row(self.lineup_note, "", "Lineups")
        self.lineup_help: HelpIcon = lineup_row.help_icon  # the longer lineup note
        lvv.addWidget(lineup_row)
        self.lineup_table = QTableWidget(0, 9)
        self.lineup_table.setHorizontalHeaderLabels(["Team", "Players", "Min", "+/-", "Pts for", "Pts ag.",
                                                     "Poss", "ORtg", "DRtg"])
        self._table_style(self.lineup_table)
        self.lineup_table.cellClicked.connect(self._lineup_cell)
        lvv.addWidget(self.lineup_table)
        self.inner.addTab(lw, "Lineups")
        rv.addWidget(self.inner, 3)
        hb = QHBoxLayout()
        self.items_lbl = QLabel("<b>Detected events behind the selected number</b>")
        hb.addWidget(self.items_lbl)
        hb.addWidget(HelpIcon(HELP["stats_events"], "Detected events"))
        hb.addStretch(1)
        self.btn_playlist = QPushButton("Send to playlist")
        self.btn_playlist.setToolTip("Hand these clips to the Playlists feature")
        self.btn_playlist.clicked.connect(self.send_to_playlist)
        self.btn_playlist.setVisible(playlist_module() is not None)
        hb.addWidget(self.btn_playlist)
        rv.addLayout(hb)
        self.items = QListWidget()
        self.items.itemClicked.connect(lambda it: self.seek(it.data(SEEK_ROLE)))
        rv.addWidget(self.items, 2)
        split.addWidget(right)
        split.setSizes([560, 520])
        lay.addWidget(split, 1)

    @staticmethod
    def _table_style(t: QTableWidget) -> None:
        t.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        t.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        t.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        t.verticalHeader().setVisible(False)
        t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        t.horizontalHeader().setStretchLastSection(True)

    # -------------------------------------------------------------- data
    def load(self, p) -> None:
        self.p = p
        self.st, self._sig = None, None
        if self.isVisible():
            self.refresh()
        elif self._built:
            self.msg.setText("")

    def showEvent(self, e) -> None:
        super().showEvent(e)
        self.refresh()

    def _signature(self) -> tuple:
        if self.p is None:
            return ()
        root = Path(self.p.root)
        out = []
        for f in (root / "results" / "game_state.json", root / "results" / "replay_data.json",
                  root / "results" / "lineup.json", root / "timeline.json", root / S.TRUTH_FILE,
                  root / "results" / S.TRUTH_FILE):
            try:
                out.append(f.stat().st_mtime)
            except OSError:
                out.append(None)
        return tuple(out)

    def colors(self) -> dict:
        rt = getattr(self.win, "replay_tab", None)
        try:
            c = rt.display_colors() if rt is not None and getattr(rt, "p", None) is self.p else {}
        except Exception:  # noqa: BLE001
            c = {}
        base = {k: (self.st or {}).get("teams", {}).get(k, {}).get("color") for k in S.TEAMS}
        out = {**FALLBACK, **{k: v for k, v in base.items() if v}}
        try:
            from replay.team_binding import distinct_colors
            out.update({k: v for k, v in distinct_colors(out).items() if k in S.TEAMS})  # near-identical kits
        except Exception:  # noqa: BLE001
            pass
        return {**out, **{k: c[k] for k in S.TEAMS if c.get(k)}}

    def refresh(self, force: bool = False) -> None:
        if self.p is None:
            return
        self._ensure_ui()
        sig = self._signature()
        if not force and self.st is not None and sig == self._sig:
            return
        if not (Path(self.p.root) / "results" / "replay_data.json").exists():
            self.st = None
            self.msg.setText("<b>No statistics yet.</b> They are computed from the 2D replay once it is built.")
            self.chart.set_data(None, [], {})
            return
        rt = getattr(self.win, "replay_tab", None)
        replay = rt.data if rt is not None and getattr(rt, "p", None) is self.p and rt.data else None
        try:
            self.st = S.for_project(self.p.root, replay=replay, max_build_frames=6000)
        except Exception as e:  # noqa: BLE001 - show, never crash the window
            self.st = None
            self.msg.setText(f"<b>Statistics could not be computed</b> ({type(e).__name__}: {e}).")
            return
        self._sig = sig
        self._fill()

    def _fill(self) -> None:
        st = self.st
        if st is None:
            return
        nm = {k: st["teams"][k]["name"] for k in S.TEAMS}
        cols = self.colors()
        g = st["game"]
        sc = st["score"]
        sparse = is_sparse(st)
        pa, pb = (NA, NA) if sparse else (g["A"]["points"], g["B"]["points"])
        txt = (f"<span style='font-size:15px'><b><span style='color:{cols['A']}'>{nm['A']}</span> {pa}"
               f" – {pb} <span style='color:{cols['B']}'>{nm['B']}</span></b></span>"
               "&nbsp;&nbsp;<span style='color:#b45309'>estimated from detected events, not the official score</span>")
        if sparse:
            txt = (f"<div style='background:#fef3c7; color:#92400e'><b>&#9888; Very little was detected in this Replay "
                   f"(no made baskets, {len(st.get('shots') or [])} shot(s)): the numbers below are not "
                   "representative of the game. n/a = not detected.</b></div>" + txt)
        if sc.get("unattributed_points"):
            txt += f" · {sc['unattributed_points']} pts without team"
        tr = sc.get("truth")
        if tr and all(tr.get(k) is not None for k in S.TEAMS):
            txt += (f"<br>Score-bar truth {tr['A']:.0f} – {tr['B']:.0f}: detected {sc.get('total_pct_of_truth')} % "
                    f"of the points (difference {sc['diff']['A']:+.0f} / {sc['diff']['B']:+.0f}).")
        self.msg.setText(txt)
        for combo, items in ((self.period, [("Whole game", None)] + [(p["label"], i) for i, p in enumerate(st["periods"])]
                              if len(st["periods"]) > 1 else [("Whole game", None)]),
                             (self.team, [("Both teams", None), (nm["A"], "A"), (nm["B"], "B")])):
            cur = combo.currentData()
            combo.blockSignals(True)
            combo.clear()
            for lab, data in items:
                combo.addItem(lab, data)
            i = combo.findData(cur)
            combo.setCurrentIndex(max(0, i))
            combo.blockSignals(False)
        self._fill_players()
        self.notes.set_help(HELP["stats_notes"].format(notes="<br>".join(st.get("notes") or [])) if st.get("notes")
                            else "")
        self._fill_lineups()
        self._filters_changed()

    def _fill_players(self) -> None:
        st = self.st or {}
        team = self.team.currentData()
        names = sorted({s["shooter_name"] for s in st.get("shots") or [] if s.get("shooter_name")
                        and (team is None or s.get("team") == team)}, key=lambda n: (len(n), n))
        cur = self.player.currentData()
        self.player.blockSignals(True)
        self.player.clear()
        self.player.addItem("Everyone", None)
        for n in names:
            self.player.addItem(n, n)
        self.player.setCurrentIndex(max(0, self.player.findData(cur)))
        self.player.blockSignals(False)

    def _period(self) -> dict | None:
        i = self.period.currentData()
        return None if i is None or self.st is None else self.st["periods"][i]

    def _filtered_shots(self) -> list[dict]:
        if self.st is None:
            return []
        return S.filter_shots(self.st["shots"], self.team.currentData(), self._period(), self.player.currentData(),
                              self.outcome.currentData())

    def _filters_changed(self, *_a) -> None:
        if self.st is None:
            return
        if self.sender() is self.team:
            self._fill_players()
        shots = self._filtered_shots()
        self.chart.set_data(self.st["court"], shots, self.colors())
        n_loc = sum(1 for s in shots if s.get("x") is not None)
        self.chart_note.setText(f"{len(shots)} field-goal attempts ({n_loc} with a court location)")  # key: behind the ?
        self._fill_zones(shots)
        self._fill_team_table()

    def _fill_zones(self, shots: list[dict]) -> None:
        zt = S.zone_table(shots)
        self.zones.setRowCount(0)
        for z in S.ZONES + ("unknown", "total"):
            r = zt[z]
            row = self.zones.rowCount()
            self.zones.insertRow(row)
            sp = is_sparse(self.st) and not r["fga"]
            vals = [S.ZONE_TITLES.get(z, "Total"), NA if sp else str(r["fgm"]), NA if sp else str(r["fga"]), _pct(r["pct"])]
            for c, v in enumerate(vals):
                it = QTableWidgetItem(v)
                it.setData(ITEMS_ROLE, z)
                if z == "total":
                    f = it.font()
                    f.setBold(True)
                    it.setFont(f)
                self.zones.setItem(row, c, it)

    def _zone_clicked(self, it: QTableWidgetItem) -> None:
        z = it.data(ITEMS_ROLE)
        shots = [s for s in self._filtered_shots() if z == "total" or s.get("zone") == z]
        self._show_items(f"{S.ZONE_TITLES.get(z, 'All zones')} shots", [S._item(s["t"], s["title"]) for s in shots])

    def _block(self, team: str) -> dict:
        p = self._period()
        return (p if p is not None else self.st["game"])[team]

    def _fill_team_table(self) -> None:
        st = self.st
        nm = {k: st["teams"][k]["name"] for k in S.TEAMS}
        cols = self.colors()
        t = self.team_table
        t.setColumnCount(3)
        t.setHorizontalHeaderLabels(["", nm["A"], nm["B"]])
        for j, k in enumerate(S.TEAMS, 1):
            h = t.horizontalHeaderItem(j)
            if h is not None:
                h.setForeground(QBrush(QColor(cols[k])))
        t.setRowCount(0)
        for key, label, kind in S.STAT_ROWS:
            row = t.rowCount()
            t.insertRow(row)
            it = QTableWidgetItem(label)
            it.setData(ITEMS_ROLE, key)
            t.setItem(row, 0, it)
            for j, k in enumerate(S.TEAMS, 1):
                v = self._block(k).get(key)
                cell = QTableWidgetItem(fmt_value(v, kind, is_sparse(st)))
                cell.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                cell.setData(ITEMS_ROLE, key)
                t.setItem(row, j, cell)
        p = self._period()
        self.inner.setTabText(0, "Team stats" + (f" – {p['label']}" if p else " – whole game"))

    def _team_cell(self, row: int, col: int) -> None:
        key = S.STAT_ROWS[row][0]
        teams = S.TEAMS if col == 0 else (S.TEAMS[col - 1],)
        lst = S.ROW_ITEMS.get(key)
        items: list[dict] = []
        for k in teams:
            items += [{**i, "team": k} for i in self._block(k).get("items", {}).get(lst, [])]
        items.sort(key=lambda i: i.get("t") or 0.0)
        who = "both teams" if col == 0 else self.st["teams"][teams[0]]["name"]
        p = self._period()
        cats = PLAYLIST_CATEGORIES.get(lst or "")
        flt = None if not cats else {"categories": list(cats), "team": None if col == 0 else teams[0],
                                     "t_from": p.get("w0", p["t0"]) if p else None,
                                     "t_to": p.get("w1", p["t1"]) if p else None,
                                     "min_confidence": S.MIN_EVENT_CONF}
        self._show_items(f"{S.STAT_ROWS[row][1].strip()} – {who}" + (f" ({p['label']})" if p else ""), items, flt)

    def _fill_lineups(self) -> None:
        lu = (self.st or {}).get("lineups") or {}
        t = self.lineup_table
        t.setRowCount(0)
        if not lu.get("available"):
            self.lineup_note.setText(lu.get("reason") or "Lineups unavailable.")
            self.lineup_help.set_help("")
            return
        self.lineup_note.setText("")
        self.lineup_help.set_help(lu.get("note", ""))  # the lineup caveat is behind the ?
        nm = {k: self.st["teams"][k]["name"] for k in S.TEAMS}
        for r in lu["rows"]:
            row = t.rowCount()
            t.insertRow(row)
            vals = [nm[r["team"]], " ".join(r["players"]) + ("  (?)" if r.get("duplicate_numbers") else ""),
                    f"{r['minutes']:.1f}", f"{r['plus_minus']:+d}", str(r["pts_for"]), str(r["pts_against"]),
                    f"{r['poss_for']}/{r['poss_against']}", fmt_value(r["off_rating"], "float"),
                    fmt_value(r["def_rating"], "float")]
            for c, v in enumerate(vals):
                it = QTableWidgetItem(v)
                if c >= 2:
                    it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                t.setItem(row, c, it)

    def _lineup_cell(self, row: int, _col: int) -> None:
        r = self.st["lineups"]["rows"][row]
        self._show_items(f"Stints of {' '.join(r['players'])}", r["stints"])
        if r["stints"]:
            self.seek(r["stints"][0]["t"])

    def _show_items(self, name: str, items: list[dict], flt: dict | None = None) -> None:
        self._items, self._items_name, self._items_filter = items, name, flt
        self.items_lbl.setText(f"<b>{name}</b> – {len(items)} detected" if items else f"<b>{name}</b> – none detected")
        self.items.clear()
        for i in items:
            if i.get("t") is None:
                continue
            it = QListWidgetItem(f"{S.fmt_clock(i['t'])}   {i['title']}")
            it.setData(SEEK_ROLE, float(i["t"]))
            self.items.addItem(it)
        self.btn_playlist.setEnabled(bool(items))

    # -------------------------------------------------------------- actions
    def seek(self, t: float | None) -> None:
        if t is None:
            return
        rt = getattr(self.win, "replay_tab", None)
        if rt is None:
            return
        t = max(0.0, float(t) - 1.0)  # a second of lead-in
        if self.goto.isChecked() and hasattr(self.win, "tabs"):
            self.win.tabs.setCurrentWidget(rt)
        rt.seek_to(t)

    def _playlist_panel(self) -> Any | None:
        try:
            from replay.gui.playlists import PlaylistPanel
        except Exception:  # noqa: BLE001
            return None
        panel = self.win.findChild(PlaylistPanel) if hasattr(self.win, "findChild") else None
        if panel is not None and getattr(panel, "p", None) is not None and Path(panel.p.root) == Path(self.p.root):
            return panel
        return None

    def send_to_playlist(self) -> None:
        """Hand the selected row's moments to the Playlists feature (new playlist in <Replay>/playlists.json)."""
        pl = playlist_module()
        if pl is None or self.p is None or not self._items:
            return
        panel = self._playlist_panel()

        def catalog():
            if panel is not None:
                return panel.catalog()
            rt = getattr(self.win, "replay_tab", None)
            replay = rt.data if rt is not None and getattr(rt, "p", None) is self.p else None
            return pl.collect(self.p.root, replay=replay)

        try:
            playlist = make_stats_playlist(pl, f"Stats: {self._items_name}", self._items, self._items_filter, catalog)
            if panel is not None:
                panel.book.add(playlist)
                panel.save()
                panel._fill_combo(select=playlist.id)
                panel.render()
                dock = panel.parentWidget()
                if dock is not None:
                    dock.show()
            else:
                book = pl.PlaylistBook.load(self.p.root)
                book.add(playlist)
                book.save(self.p.root)
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "Stats", "Could not create the playlist:\n" + str(e))
            return
        kind = "filter" if self._items_filter else "fixed list"
        self.win.statusBar().showMessage(f"Playlist '{playlist.name}' created ({len(playlist.clips)} clips, {kind})",
                                         6000)

    def _ask_dir(self, title: str) -> str:
        st = getattr(self.win, "settings", None)
        start = st.value("data_export_dir", str(Path.home())) if st is not None else str(Path.home())
        return QFileDialog.getExistingDirectory(self, title, start)

    def export_tables(self) -> None:
        if self.st is None:
            return
        d = self._ask_dir("Export statistics (CSV + JSON) into folder")
        if not d:
            return
        try:
            S.write_exports(self.st, Path(d))
        except OSError as e:
            QMessageBox.warning(self, "Stats", f"Export failed:\n{e}")
            return
        self.win.statusBar().showMessage(f"Statistics exported to {d}", 6000)

    def export_summary(self) -> None:
        if self.st is None:
            return
        name = (self.p.info().get("name") if self.p is not None and hasattr(self.p, "info") else None) or "game"
        path, _f = QFileDialog.getSaveFileName(self, "Save printable summary", str(Path.home() / f"{name}_stats.pdf"),
                                               "PDF (*.pdf);;PNG image (*.png)")
        if not path:
            return
        try:
            save_summary(path, self.st, self.colors(), f"{name} – statistics")
        except OSError as e:
            QMessageBox.warning(self, "Stats", f"Saving failed:\n{e}")
            return
        self.win.statusBar().showMessage(f"Summary saved to {path}", 6000)
