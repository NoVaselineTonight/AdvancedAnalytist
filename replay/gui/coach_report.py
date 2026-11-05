"""One-page post-game coach report (in-app view + PDF) and player pages. Data: :mod:`replay.coach`.

The page is painted in fixed logical coordinates (PAGE_W x PAGE_H, A4 landscape proportions) by
:func:`paint_report`; the dialog scales it to the window and the PDF writer to the paper. Every number that has
moments behind it is a hot spot: clicking it in the dialog opens those moments as a video playlist (Playlists panel).

Entry points: :func:`open_report` (Stats tab button, Export menu), :func:`save_pdf` (optionally with one page per
player). The dialog's "Export" menu also writes the box score as CSV (:mod:`replay.coach_export`), and
"Copy summary" puts a short plain-text game summary on the clipboard.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PySide6.QtCore import QLocale, QMarginsF, QPointF, QRectF, QSize, Qt, QTimer
from PySide6.QtGui import (QColor, QFont, QGuiApplication, QImage, QPageLayout, QPageSize, QPainter, QPainterPath,
                           QPdfWriter, QPen)
from PySide6.QtWidgets import (QApplication, QComboBox, QDialog, QDockWidget, QFileDialog, QHBoxLayout, QLabel,
                               QMenu, QMessageBox, QPushButton, QScrollArea, QSizePolicy, QToolTip, QVBoxLayout,
                               QWidget)

from replay import coach as C
from replay.gui.helpicon import HelpIcon
from replay import coach_export as CE
from replay.gui.stats_tab import FALLBACK, CourtMap, draw_court, draw_shots

PAGE_W, PAGE_H = 1600.0, 1131.0
INK, DIM, LINE, HEAD_BG = QColor("#111827"), QColor("#6b7280"), QColor("#d1d5db"), QColor("#f3f4f6")
PROV_MARK = {"measured": "", "inferred": "*", "team": "†", "not measured": "‡"}


def _font(px: float, bold: bool = False) -> QFont:
    f = QFont()
    f.setFamilies(["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"])
    f.setPixelSize(max(6, int(px)))
    f.setBold(bold)
    return f


def _colors(rep: dict) -> dict:
    cols = {k: (rep["teams"][k].get("color") or FALLBACK[k]) for k in C.TEAMS}
    try:
        from replay.team_binding import distinct_colors
        cols = {**cols, **{k: v for k, v in distinct_colors(cols).items() if k in cols}}  # near-identical kits
    except Exception:  # noqa: BLE001
        pass
    return cols


class Hits(list):
    """(rect, title, moments) of every clickable number on the page."""

    def add(self, rect: QRectF, title: str, moments: list | None) -> None:
        if moments:
            self.append((QRectF(rect), title, [m for m in moments if m]))

    def at(self, p: QPointF):
        return next((h for h in self if h[0].contains(p)), None)


def _text(qp: QPainter, rect: QRectF, txt: str, px: float = 13, bold: bool = False, color: QColor = INK,
          align=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter) -> None:
    qp.setFont(_font(px, bold))
    qp.setPen(color)
    qp.drawText(rect, int(align), txt)


def _box_table(qp: QPainter, x: float, y: float, w: float, rep: dict, team: str, hits: Hits) -> float:
    """Box score of one team; returns the y below it."""
    name = rep["teams"][team]["name"]
    col = QColor(_colors(rep)[team])
    qp.fillRect(QRectF(x, y, 8, 22), col)
    _text(qp, QRectF(x + 14, y, w, 22), f"{name}  –  {rep['score'][team]} pts", 16, True)
    y += 26
    cols = C.COLUMNS
    first = 120.0
    cw = (w - first) / len(cols)
    qp.fillRect(QRectF(x, y, w, 22), HEAD_BG)
    _text(qp, QRectF(x + 6, y, first, 22), "Player", 12, True, DIM)
    for i, (key, title, prov) in enumerate(cols):
        _text(qp, QRectF(x + first + i * cw, y, cw, 22), title + PROV_MARK.get(prov, ""), 12, True, DIM,
              Qt.AlignmentFlag.AlignCenter)
    y += 22
    rows = rep["box"][team]["rows"] + [rep["box"][team]["total"]]
    for j, r in enumerate(rows):
        total = r["player"] == "Team"
        if total:
            qp.setPen(QPen(INK, 1.2))
            qp.drawLine(QPointF(x, y), QPointF(x + w, y))
        elif j % 2:
            qp.fillRect(QRectF(x, y, w, 21), QColor("#fafafa"))
        dim_row = r["player"] == "Unidentified"
        label = r["player"] if not dim_row else "Unidentified players"
        if r.get("name"):  # from the scoresheet / roster
            qp.setFont(_font(13))
            label = qp.fontMetrics().elidedText(f"{r['player']} {r['name']}", Qt.TextElideMode.ElideRight,
                                                int(first - 8))
        _text(qp, QRectF(x + 6, y, first, 21), label, 13, total, DIM if dim_row else INK)
        official = set(r.get("official") or [])
        for i, (key, title, prov) in enumerate(cols):
            cell = QRectF(x + first + i * cw, y, cw, 21)
            txt = C.cell_text(r, key)
            if key in official:
                txt += "°"
            has = bool(r.get("moments", {}).get(key))
            colr = QColor("#1d4ed8") if has else (DIM if txt == "–" else INK)
            if key in official and not has:
                colr = QColor("#047857")
            _text(qp, cell, txt, 13, total or key in official, colr, Qt.AlignmentFlag.AlignCenter)
            hits.add(cell, f"{name} {r['player']} – {title}", r.get("moments", {}).get(key))
        y += 21
    return y + 10


def _compare_table(qp: QPainter, x: float, y: float, w: float, rep: dict, hits: Hits) -> float:
    cols = _colors(rep)
    _text(qp, QRectF(x, y, w, 24), "Team comparison", 16, True)
    y += 28
    cw = 115.0
    for k, cx in (("A", x + w - 2 * cw), ("B", x + w - cw)):
        qp.fillRect(QRectF(cx + 6, y + 6, 10, 10), QColor(cols[k]))
        qp.setFont(_font(12, True))
        name = qp.fontMetrics().elidedText(rep["teams"][k]["name"], Qt.TextElideMode.ElideRight, int(cw - 24))
        _text(qp, QRectF(cx + 20, y, cw - 22, 22), name, 12, True, DIM)
    y += 24
    for j, r in enumerate(rep["compare"]):
        if j % 2:
            qp.fillRect(QRectF(x, y, w, 21), QColor("#fafafa"))
        _text(qp, QRectF(x + 4, y, w - 2 * cw, 21), r["title"], 13)
        for k, cx in (("A", x + w - 2 * cw), ("B", x + w - cw)):
            v = r.get(k)
            if v is None:
                txt = "–"
            elif r["key"].endswith("pct"):
                txt = f"{v * 100:.0f}%"
            elif isinstance(v, float):
                txt = f"{v:.2f}"
            else:
                txt = str(v)
            cell = QRectF(cx, y, cw, 21)
            ms = r.get(f"moments_{k}")
            if r.get("official"):  # from the scoresheet
                _text(qp, cell, txt + "°", 13, True, QColor("#047857"), Qt.AlignmentFlag.AlignCenter)
                hits.add(cell, f"{rep['teams'][k]['name']} – {r['title']}", ms)
                continue
            _text(qp, cell, txt, 13, False, QColor("#1d4ed8") if ms else INK, Qt.AlignmentFlag.AlignCenter)
            hits.add(cell, f"{rep['teams'][k]['name']} – {r['title']}", ms)
        y += 21
    return y + 12


def _momentum(qp: QPainter, rect: QRectF, rep: dict, hits: Hits) -> None:
    _text(qp, QRectF(rect.x(), rect.y(), rect.width(), 24), "Score margin and runs", 16, True)
    r = rect.adjusted(36, 30, -8, -22)
    mo = rep["momentum"]
    cols = _colors(rep)
    qp.setPen(QPen(LINE, 1))
    qp.setBrush(Qt.BrushStyle.NoBrush)
    qp.drawRect(r)
    if len(mo) < 2 or mo[-1][0] <= mo[0][0]:
        _text(qp, r, "No made baskets detected", 13, False, DIM, Qt.AlignmentFlag.AlignCenter)
        return
    t0, t1 = mo[0][0], mo[-1][0]
    span = max(4, max(abs(a - b) for _t, a, b in mo))
    X = lambda t: r.x() + (t - t0) / (t1 - t0) * r.width()  # noqa: E731
    Y = lambda d: r.center().y() - d / span * r.height() / 2  # noqa: E731
    qp.setPen(QPen(LINE, 1, Qt.PenStyle.DashLine))
    qp.drawLine(QPointF(r.x(), Y(0)), QPointF(r.right(), Y(0)))
    for p in rep.get("periods") or []:
        if p.get("t0") and t0 < p["t0"] < t1:
            qp.drawLine(QPointF(X(p["t0"]), r.y()), QPointF(X(p["t0"]), r.bottom()))
    for run in rep["runs"]:
        band = QRectF(X(run["t0"]), r.y(), max(3.0, X(run["t1"]) - X(run["t0"])), r.height())
        c = QColor(cols[run["team"]])
        c.setAlpha(45)
        qp.fillRect(band, c)
        hits.add(band, f"{rep['teams'][run['team']]['name']} {run['title']}", run["moments"])
    path = QPainterPath(QPointF(X(mo[0][0]), Y(mo[0][1] - mo[0][2])))
    for t, a, b in mo[1:]:
        path.lineTo(QPointF(X(t), path.currentPosition().y()))
        path.lineTo(QPointF(X(t), Y(a - b)))
    qp.setPen(QPen(INK, 2))
    qp.drawPath(path)
    _text(qp, QRectF(rect.x(), r.y() - 4, 34, 20), f"+{span}", 11, False, QColor(cols["A"]))
    _text(qp, QRectF(rect.x(), r.bottom() - 16, 34, 20), f"-{span}", 11, False, QColor(cols["B"]))
    _text(qp, QRectF(r.x(), r.bottom() + 2, r.width(), 18),
          f"{rep['teams']['A']['name']} ahead above the line · runs of {C.RUN_MIN_POINTS}+ unanswered points shaded",
          11, False, DIM)


def _shot_chart(qp: QPainter, rect: QRectF, rep: dict, team: str, hits: Hits) -> None:
    shots = [s for s in rep["shots"] if s["team"] == team]
    made = sum(1 for s in shots if s.get("outcome") == "made")
    unplaced = sum(1 for s in shots if s.get("x") is None or s.get("y") is None)
    _text(qp, QRectF(rect.x(), rect.y(), rect.width(), 22),
          f"Shots – {rep['teams'][team]['name']} ({made}/{len(shots)} made"
          + (f", {unplaced} without a court location" if unplaced else "") + ")", 14, True)
    geo = rep.get("court")
    if not geo:
        return
    m = CourtMap(rect.adjusted(0, 24, 0, 0), geo["length"], geo["width"])
    draw_court(qp, m, geo)
    draw_shots(qp, m, shots, _colors(rep), r_px=max(4.0, m.s * 0.32))
    for s in shots:
        if s.get("x") is not None:
            c = m.xy(float(s["x"]), float(s["y"]))
            hits.add(QRectF(c.x() - 8, c.y() - 8, 16, 16), s.get("title") or "Shot",
                     [C._moment(s.get("t"), s.get("title") or "Shot", team, s.get("t_rim"))])


def _lineups(qp: QPainter, x: float, y: float, w: float, rep: dict, hits: Hits) -> float:
    _text(qp, QRectF(x, y, w, 24), "Most used 5-man units", 16, True)
    y += 28
    lu = rep["lineups"]
    if not lu.get("available"):
        _text(qp, QRectF(x, y, w, 40), "Lineups need jersey numbers / the lineup solver: " + (lu.get("note") or
              "not available for this Replay."), 12, False, DIM, Qt.AlignmentFlag.AlignLeft | Qt.TextFlag.TextWordWrap)
        return y + 44
    cols = _colors(rep)
    for k in C.TEAMS:
        qp.fillRect(QRectF(x, y + 5, 8, 12), QColor(cols[k]))
        _text(qp, QRectF(x + 14, y, w, 22), rep["teams"][k]["name"], 13, True)
        _text(qp, QRectF(x + w - 160, y, 160, 22), "MIN     +/-", 12, True, DIM, Qt.AlignmentFlag.AlignRight)
        y += 22
        for row in lu.get(k) or []:
            seen: set = set()
            names = []
            for pn in row["players"]:  # unnumbered slots and repeated numbers are not identified players
                names.append(pn if pn.startswith("#") and pn not in seen else "?")
                seen.add(pn)
            players = " ".join(names)
            _text(qp, QRectF(x + 14, y, w - 180, 20), players, 13)
            cell = QRectF(x + w - 160, y, 160, 20)
            pm = row.get("pm")
            _text(qp, cell, f"{C.fmt_min(row['seconds'])}     {'–' if pm is None else f'{pm:+d}'}", 13, False,
                  QColor("#1d4ed8"), Qt.AlignmentFlag.AlignRight)
            hits.add(QRectF(x, y, w, 20), f"{rep['teams'][k]['name']} unit {players}", row["moments"])
            y += 20
        y += 6
    return y


def _period_table(qp: QPainter, x: float, y: float, w: float, rep: dict, pt: dict, hits: Hits) -> float:
    cols = _colors(rep)
    _text(qp, QRectF(x, y, w, 24), "By period", 15, True)
    _text(qp, QRectF(x + 120, y, w - 120, 24), f"periods from: {pt['source']}", 12, False, DIM)
    y += 26
    heads = ["", rep["teams"]["A"]["name"], "FG", "3PT", rep["teams"]["B"]["name"], "FG", "3PT", "Scoresheet"]
    cw = [230, 130, 80, 80, 130, 80, 80, 170]
    xs = [x]
    for c in cw[:-1]:
        xs.append(xs[-1] + c)
    qp.fillRect(QRectF(x, y, sum(cw), 20), HEAD_BG)
    for i, hd in enumerate(heads):
        _text(qp, QRectF(xs[i] + 4, y, cw[i] - 8, 20), hd[:16], 12, True,
              QColor(cols["A"]) if i == 1 else QColor(cols["B"]) if i == 4 else DIM,
              Qt.AlignmentFlag.AlignLeft if i == 0 else Qt.AlignmentFlag.AlignCenter)
    y += 20
    for r in pt["rows"]:
        _text(qp, QRectF(xs[0] + 4, y, cw[0] - 8, 20), r["label"], 13)
        for j, k in enumerate(("A", "B")):
            b = {"pts": 0, "fgm": 0, "fga": 0, "fg3m": 0, "fg3a": 0, "moments": [], **(r.get(k) or {})}
            base = 1 + 3 * j
            for i, txt in enumerate((str(b["pts"]), f"{b['fgm']}-{b['fga']}", f"{b['fg3m']}-{b['fg3a']}")):
                cell = QRectF(xs[base + i], y, cw[base + i], 20)
                _text(qp, cell, txt, 13, i == 0, QColor("#1d4ed8") if b["moments"] else INK,
                      Qt.AlignmentFlag.AlignCenter)
                hits.add(cell, f"{rep['teams'][k]['name']} – {r['label']}", b["moments"])
        o = r.get("official")
        _text(qp, QRectF(xs[7], y, cw[7], 20), "–" if not o else f"{o['A']} – {o['B']}°", 13, bool(o),
              QColor("#047857") if o else DIM, Qt.AlignmentFlag.AlignCenter)
        y += 20
    return y


def _sheet_hints(qp: QPainter, x: float, y: float, w: float, h: float, rep: dict, hits: Hits) -> None:
    sh = rep.get("sheet")
    if not sh or h < 40:
        return
    _text(qp, QRectF(x, y, w, 22), "Check against the scoresheet", 15, True)
    y += 24
    hints = sh.get("hints") or []
    if not hints:
        _text(qp, QRectF(x, y, w, 20), "Detected points agree with the scoresheet.", 12, False, QColor("#047857"))
        return
    for hnt in hints[: max(1, int((h - 30) // 19))]:
        cell = QRectF(x, y, w, 19)
        _text(qp, cell, "• " + hnt["text"], 12, False, QColor("#b45309"))
        hits.add(cell, hnt["text"], hnt.get("moments"))
        y += 19


def paint_report(qp: QPainter, rep: dict, title: str = "", hits: Hits | None = None) -> Hits:
    """Paint the whole page in PAGE_W x PAGE_H logical units."""
    hits = hits if hits is not None else Hits()
    qp.save()
    qp.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing)
    qp.fillRect(QRectF(0, 0, PAGE_W, PAGE_H), QColor("#ffffff"))
    cols = _colors(rep)
    a, b = rep["teams"]["A"]["name"], rep["teams"]["B"]["name"]
    _text(qp, QRectF(30, 18, 900, 34), f"{a}  {rep['score']['A']} – {rep['score']['B']}  {b}", 28, True)
    sub = (title + " · " if title else "") + "Post-game report · " + rep["score_note"]
    sh_ = rep.get("sheet")
    if sh_ and rep.get("score_detected"):
        d = rep["score_detected"]
        sub += (" · score " + ("from the scoresheet (official)" if sh_["full_game"] else
                               "of this stretch from the scoresheet") + f", detected {d['A']}–{d['B']}")
        if sh_.get("start_source") == "scorebar" and sh_.get("pin_label"):
            sub += f" · clip placed by the score bar: {sh_['pin_label'].split(' (')[0]}"
        elif not sh_["full_game"] and sh_.get("alignment") == "low":
            sub += " · sheet matched by the order of scoring only (low confidence): pick the quarter in Overview › Scoresheet…"
    _text(qp, QRectF(30, 52, 1100, 22), sub, 13, False, DIM)
    n_shots = len(rep.get("shots") or [])
    if sum(rep["score"].values()) == 0 and n_shots < 5:
        _text(qp, QRectF(700, 18, PAGE_W - 730, 56), "Very little was detected in this Replay (no made baskets, "
              f"{n_shots} shot(s)): the numbers below are not representative of the game.", 14, True,
              QColor("#b45309"), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter | Qt.TextFlag.TextWordWrap)
    qp.fillRect(QRectF(30, 80, PAGE_W - 60, 3), QColor(cols["A"]))
    qp.fillRect(QRectF(PAGE_W / 2, 80, PAGE_W / 2 - 30, 3), QColor(cols["B"]))
    # left: box scores + shot charts; right: comparison, momentum, lineups
    lx, lw, rx, rw = 30.0, 1000.0, 1060.0, 510.0
    y = 96.0
    for k in C.TEAMS:
        y = _box_table(qp, lx, y, lw, rep, k, hits)
    pt = rep.get("period_table") or {"rows": []}
    period_h = 34 + 20 * (len(pt["rows"]) + 1) if pt["rows"] else 0
    sy = y + 6
    sh = max(150.0, min(340.0, PAGE_H - 130 - sy - period_h))
    _shot_chart(qp, QRectF(lx, sy, lw / 2 - 10, sh), rep, "A", hits)
    _shot_chart(qp, QRectF(lx + lw / 2 + 10, sy, lw / 2 - 10, sh), rep, "B", hits)
    if pt["rows"]:
        _period_table(qp, lx, sy + sh + 8, lw, rep, pt, hits)
    ry = _compare_table(qp, rx, 96.0, rw, rep, hits)
    _momentum(qp, QRectF(rx, ry, rw, 200), rep, hits)
    ly = _lineups(qp, rx, ry + 208, rw, rep, hits)
    _sheet_hints(qp, rx, ly + 4, rw, PAGE_H - 130 - ly, rep, hits)
    # footer: what was measured vs inferred
    fy = PAGE_H - 118
    qp.setPen(QPen(LINE, 1))
    qp.drawLine(QPointF(30, fy - 6), QPointF(PAGE_W - 30, fy - 6))
    notes = ("How sure: green ° = official (from the attached scoresheet). " if rep.get("sheet") else
             "How sure: numbers come from automatic detection, not an official scoresheet. ")
    notes += ("* inferred (detected events + a rule) · † team only · ‡ not measured (shown as –) · plain = measured. "
             "Blue numbers: click to watch those moments. ")
    prov = rep["provenance"]
    notes += " ".join(prov[k] for k in ("pts", "ast", "min", "pm", "pf"))
    qp.setFont(_font(12))
    qp.setPen(DIM)
    qp.drawText(QRectF(30, fy, PAGE_W - 60, 112), int(Qt.AlignmentFlag.AlignLeft | Qt.TextFlag.TextWordWrap), notes)
    qp.restore()
    return hits


def render_image(rep: dict, width: int = 1600, title: str = "") -> QImage:
    img = QImage(width, int(width * PAGE_H / PAGE_W), QImage.Format.Format_ARGB32)
    img.fill(QColor("#ffffff"))
    qp = QPainter(img)
    qp.scale(width / PAGE_W, width / PAGE_W)
    paint_report(qp, rep, title)
    qp.end()
    return img


# ====================================================================================== page 2: possessions + shot quality
OUTCOME_COLORS = {"made": "#15803d", "ft": "#0e7490", "miss": "#b45309", "turnover": "#b91c1c",
                  "period_end": "#6b7280", "unknown": "#9ca3af"}


def filtered_possessions(rep: dict, team: str | None = None, outcome: str | None = None) -> list[dict]:
    return [p for p in rep.get("possessions") or [] if (team is None or p["team"] == team)
            and (outcome is None or p["outcome"] == outcome)]


def paint_page2(qp: QPainter, rep: dict, title: str = "", hits: Hits | None = None, team: str | None = None,
                outcome: str | None = None, page_no: str = "") -> Hits:
    """Possession timeline (left) and shot quality map (right), PAGE_W x PAGE_H."""
    hits = hits if hits is not None else Hits()
    qp.save()
    qp.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing)
    qp.fillRect(QRectF(0, 0, PAGE_W, PAGE_H), QColor("#ffffff"))
    cols = _colors(rep)
    names = {k: rep["teams"][k]["name"] for k in C.TEAMS}
    _text(qp, QRectF(30, 14, 1100, 30), "Possessions and shot quality" + (f" – {title}" if title else ""), 22, True)
    if page_no:
        _text(qp, QRectF(PAGE_W - 130, 14, 100, 30), page_no, 12, False, DIM, Qt.AlignmentFlag.AlignRight)
    # ---- possession timeline
    poss = filtered_possessions(rep, team, outcome)
    allp = rep.get("possessions") or []
    filt = ", ".join(x for x in ((names[team] if team else ""), (C.OUTCOME_TITLES.get(outcome, "") if outcome else ""))
                     if x)
    _text(qp, QRectF(30, 52, 960, 24), f"Possessions ({len(poss)}" + (f" of {len(allp)}, {filt}" if filt else "")
          + ") – click a row to watch it", 15, True)
    x0, y = 30.0, 82.0
    cw = [70, 150, 60, 330, 110, 50, 190]
    heads = ["Start", "Team", "Length", "Outcome", "Shooter", "Pts", "How sure"]
    xs = [x0]
    for c in cw[:-1]:
        xs.append(xs[-1] + c)
    qp.fillRect(QRectF(x0, y, sum(cw), 20), HEAD_BG)
    for i, hd in enumerate(heads):
        _text(qp, QRectF(xs[i] + 4, y, cw[i] - 8, 20), hd, 12, True, DIM)
    y += 20
    max_rows = int((PAGE_H - 70 - y) // 19)
    for j, p in enumerate(poss[:max_rows]):
        row = QRectF(x0, y, sum(cw), 19)
        if j % 2:
            qp.fillRect(row, QColor("#fafafa"))
        tname = names.get(p["team"], "Team unclear")  # (possession model v6: spells the evidence does not decide)
        qp.fillRect(QRectF(xs[1] + 2, y + 5, 8, 9), QColor(cols.get(p["team"], "#9ca3af")))
        vals = [C.fmt_min(p["t0"]), tname, "–" if p["duration_s"] is None else f"{p['duration_s']:.0f} s",
                p["detail"],
                p["shooter"] if p.get("shooter") and p["shooter"] != "Unidentified" else "–",
                str(p["points"]) if p["points"] else "", p["provenance"]]
        for i, v in enumerate(vals):
            colr = QColor(OUTCOME_COLORS.get(p["outcome"], INK)) if i == 3 else (DIM if i == 6 else INK)
            _text(qp, QRectF(xs[i] + (16 if i == 1 else 4), y, cw[i] - 8, 19), v, 12 if i == 6 else 13, i == 3, colr)
        hits.add(row, f"Possession {C.fmt_min(p['t0'])} {tname} – {p['detail']}", p["moments"])
        y += 19
    if len(poss) > max_rows:
        _text(qp, QRectF(x0, y + 2, 960, 18), f"… {len(poss) - max_rows} more (filter by team / outcome in the app)",
              12, False, DIM)
    if not allp:
        _text(qp, QRectF(x0, y + 4, 960, 20), "No possessions detected.", 13, False, DIM)
    # outcome summary per team
    sy = PAGE_H - 62
    for k_i, k in enumerate(C.TEAMS):
        mine = [p for p in allp if p["team"] == k]
        if not mine:
            continue
        parts = []
        for o, t in C.OUTCOMES:
            n = sum(1 for p in mine if p["outcome"] == o)
            if n:
                parts.append(f"{t} {n}")
        pts = sum(p["points"] for p in mine)
        n_poss = sum(1 for p in mine if p["duration_s"] is not None)
        _text(qp, QRectF(x0, sy + 18 * k_i, 960, 18), f"{names[k]}: {n_poss} possessions, {pts} pts "
              + (f"({pts / n_poss:.2f} per possession) · " if n_poss else "· ") + " · ".join(parts), 12, False,
              QColor(cols[k]))
    # ---- shot quality
    sq = rep.get("shot_quality")
    rx, rw = 1020.0, PAGE_W - 1050.0
    _text(qp, QRectF(rx, 52, rw, 24), "Shot quality by zone", 15, True)
    if not sq:
        _text(qp, QRectF(rx, 80, rw, 20), "No located shots.", 13, False, DIM)
        qp.restore()
        return hits
    y = 82.0
    zc = [150, 50, 50, 50, 70, 70, 90]
    zx = [rx]
    for c in zc[:-1]:
        zx.append(zx[-1] + c)
    for k in C.TEAMS:
        qp.fillRect(QRectF(rx, y + 5, 8, 12), QColor(cols[k]))
        _text(qp, QRectF(rx + 14, y, rw, 22), names[k], 14, True)
        y += 24
        qp.fillRect(QRectF(rx, y, sum(zc), 20), HEAD_BG)
        for i, hd in enumerate(["Zone", "FGA", "FGM", "Pts", "Pts/shot", "Expected", ""]):
            _text(qp, QRectF(zx[i] + 4, y, zc[i] - 8, 20), hd, 12, True, DIM,
                  Qt.AlignmentFlag.AlignLeft if i in (0, 6) else Qt.AlignmentFlag.AlignCenter)
        y += 20
        for zr in sq[k]:
            row = QRectF(rx, y, sum(zc), 20)
            pps, xpps = zr["pps"], zr["xpps"]
            good = pps is not None and pps >= xpps
            vals = [zr["title"], str(zr["fga"]), str(zr["fgm"]), str(zr["pts"]),
                    "–" if pps is None else f"{pps:.2f}", f"{xpps:.2f}"]
            for i, v in enumerate(vals):
                colr = INK
                if i == 4 and pps is not None:
                    colr = QColor("#15803d") if good else QColor("#b45309")
                _text(qp, QRectF(zx[i] + 4, y, zc[i] - 8, 20), v, 13, i == 4,
                      QColor("#1d4ed8") if i == 1 and zr["moments"] else colr,
                      Qt.AlignmentFlag.AlignLeft if i == 0 else Qt.AlignmentFlag.AlignCenter)
            # bar: points per shot vs expected (0..1.5)
            bx, bw = zx[6] + 4, zc[6] - 8
            qp.fillRect(QRectF(bx, y + 7, bw, 6), QColor("#f3f4f6"))
            if pps is not None:
                qp.fillRect(QRectF(bx, y + 7, bw * min(1.0, pps / 1.5), 6),
                            QColor("#15803d") if good else QColor("#f59e0b"))
            qp.setPen(QPen(INK, 1.5))
            xe = bx + bw * min(1.0, xpps / 1.5)
            qp.drawLine(QPointF(xe, y + 4), QPointF(xe, y + 16))
            hits.add(row, f"{names[k]} – shots from {zr['title']}", zr["moments"])
            y += 20
        un = (sq.get("unlocated") or {}).get(k) or 0
        if un:
            _text(qp, QRectF(rx, y, rw, 18), f"{un} shot(s) without a court location are not in the table.", 11,
                  False, DIM)
            y += 18
        y += 12
    note = ("Pts/shot = points per field-goal attempt from the zone (made baskets measured at the rim, shot place "
            "from the shooter's foot point: inferred). Expected = the zone's value x this game's FG % there "
            f"(both teams), shrunk to {C.PRIOR_FG:.0%} when there are few shots - it says whether a zone paid off "
            "better than shots there usually did in this game. Restricted area = within about 1.75 m of the basket. "
            "Green = better than expected, orange = worse. The line in the bar = expected.")
    qp.setFont(_font(11))
    qp.setPen(DIM)
    qp.drawText(QRectF(rx, y + 4, rw, PAGE_H - y - 20), int(Qt.AlignmentFlag.AlignLeft | Qt.TextFlag.TextWordWrap),
                note)
    qp.restore()
    return hits


def save_pdf(rep: dict, path: Path | str, title: str = "", players: list[dict] | None = None) -> Path:
    """One A4 landscape page; with ``players`` (from :func:`player_pages`) one more page per player."""
    w = QPdfWriter(str(path))
    w.setPageLayout(QPageLayout(QPageSize(QPageSize.PageSizeId.A4), QPageLayout.Orientation.Landscape,
                                QMarginsF(8, 8, 8, 8), QPageLayout.Unit.Millimeter))
    w.setResolution(150)
    w.setTitle(f"Post-game report {title}".strip())
    qp = QPainter(w)
    vp = qp.viewport()
    s = min(vp.width() / PAGE_W, vp.height() / PAGE_H)
    qp.scale(s, s)
    try:  # always end the painter: a painter left active on a destroyed writer aborts the process
        paint_report(qp, rep, title)
        page2 = bool(rep.get("possessions") or rep.get("shot_quality"))
        n = 1 + page2 + len(players or [])
        if page2:
            w.newPage()
            paint_page2(qp, rep, title, page_no=f"2 / {n}")  # possessions + shot quality
        for i, pg in enumerate(players or [], start=2 + page2):
            w.newPage()
            paint_player_sheet(qp, pg, rep, title, f"{i} / {n}")
    finally:
        qp.end()
    return Path(path)


def player_pages(rep: dict, catalog: Any, replay: dict) -> list[dict]:
    """Player-page data of every player whose jersey number was read in the video (team A first, by number).
    Scorers known only from the scoresheet have no positions in the video: no page."""
    return [C.player_page(rep, catalog, replay, k, r["player"]) for k in C.TEAMS for r in rep["box"][k]["rows"]
            if r["player"].startswith("#") and not r.get("sheet_only")]


def load_player_pages(p: Any, rep: dict) -> list[dict]:
    """Read the Replay's tracks and jersey-number catalog (a few seconds on a full game)."""
    from replay import playlists as pl
    catalog = pl.collect(p.root)
    replay = json.loads((p.results / "replay_data.json").read_text(encoding="utf-8"))
    return player_pages(rep, catalog, replay)


def _shot_list(qp: QPainter, rect: QRectF, pg: dict, rep: dict) -> None:
    """The player's shots in video order, in columns: video time, period, what happened."""
    shots = sorted(pg.get("shots") or [], key=lambda s: s.get("t") or 0.0)
    _text(qp, QRectF(rect.x(), rect.y(), rect.width(), 22), f"Shots in video order ({len(shots)})", 14, True)
    if not shots:
        _text(qp, QRectF(rect.x(), rect.y() + 24, rect.width(), 20), "No shots attributed to this player.", 12, False,
              DIM)
        return
    ncol, lh = 3, 19.0
    cw = rect.width() / ncol
    per_col = max(1, int((rect.height() - 26) // lh))
    shown = shots[: ncol * per_col]
    for i, s in enumerate(shown):
        x = rect.x() + (i // per_col) * cw
        y = rect.y() + 26 + (i % per_col) * lh
        t = s.get("t")
        per = CE.period_of(rep, t)
        when = (CE.clock(t) if t is not None else "–") + (f"  {per}" if per else "")
        _text(qp, QRectF(x, y, 110, lh), when, 12, False, DIM)
        qp.setFont(_font(12))
        txt = qp.fontMetrics().elidedText(s.get("title") or "Shot", Qt.TextElideMode.ElideRight, int(cw - 124))
        col = QColor("#047857") if s.get("outcome") == "made" else INK
        _text(qp, QRectF(x + 112, y, cw - 120, lh), txt, 12, False, col)
    if len(shown) < len(shots):
        _text(qp, QRectF(rect.x(), rect.bottom() - lh, rect.width(), lh),
              f"… and {len(shots) - len(shown)} more", 12, False, DIM, Qt.AlignmentFlag.AlignRight)


def paint_player_sheet(qp: QPainter, pg: dict, rep: dict, title: str = "", page: str = "") -> None:
    """A printable player page (PAGE_W x PAGE_H): the in-app player page plus the full box-score line."""
    qp.save()
    qp.fillRect(QRectF(0, 0, PAGE_W, PAGE_H), QColor("#ffffff"))
    foot = 64.0
    geo = pg.get("court") or rep.get("court") or {}
    L, W = float(geo.get("length") or 28.0), float(geo.get("width") or 15.0)
    half = (PAGE_W - 20 - 40 - 20) / 2
    court_h = half / (L + 1.2) * (W + 1.2)  # CourtMap pads 0.6 m: the courts fill the half width
    top_h = min(PAGE_H - 20 - foot - 120, 130 + court_h + 40)  # paint_player: 130 header, 40 bottom margin
    paint_player(qp, QRectF(10, 10, PAGE_W - 20, top_h), pg, rep)
    _shot_list(qp, QRectF(30, 10 + top_h, PAGE_W - 60, PAGE_H - foot - 20 - top_h), pg, rep)
    fy = PAGE_H - foot - 4
    qp.setPen(QPen(LINE, 1))
    qp.drawLine(QPointF(30, fy), QPointF(PAGE_W - 30, fy))
    r = pg.get("row") or {}
    off = set(r.get("official") or [])
    cells = [f"{t}{'°' if key in off else PROV_MARK.get(prov, '')} {C.cell_text(r, key)}" for key, t, prov in C.COLUMNS]
    _text(qp, QRectF(30, fy + 6, PAGE_W - 60, 22), "   ".join(cells), 13)
    notes = ("° official (scoresheet) · * inferred · † team only · ‡ not measured · plain = measured. "
             "Positions: time on court per square metre from the player's tracks (jersey number read in the video).")
    _text(qp, QRectF(30, fy + 30, PAGE_W - 360, 22), notes, 11, False, DIM)
    _text(qp, QRectF(PAGE_W - 330, fy + 30, 300, 22), " · ".join(x for x in (title, page) if x), 11, False, DIM,
          Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    qp.restore()


# ====================================================================================== player page
def paint_player(qp: QPainter, rect: QRectF, pg: dict, rep: dict) -> None:
    qp.save()
    qp.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing)
    qp.fillRect(rect, QColor("#ffffff"))
    team = pg["team"]
    col = QColor(_colors(rep)[team])
    x, y, w = rect.x() + 20, rect.y() + 14, rect.width() - 40
    qp.fillRect(QRectF(x, y + 4, 10, 26), col)
    r = pg.get("row") or {}
    who = f"{pg['player']} {r['name']}" if r.get("name") else pg["player"]
    _text(qp, QRectF(x + 18, y, w, 34), f"{who} – {rep['teams'][team]['name']}", 24, True)
    y += 40
    line = "   ".join(f"{t} {C.cell_text(r, k)}" for k, t, _p in C.COLUMNS if k not in ("pf", "ft"))
    _text(qp, QRectF(x, y, w, 22), line, 14)
    y += 24
    _text(qp, QRectF(x, y, w, 20), f"Tracked on court for {C.fmt_min(pg['tracked_s'])} (tracker IDs "
          f"{', '.join(pg['tracks'][:12])}{'…' if len(pg['tracks']) > 12 else ''})", 12, False, DIM)
    y += 28
    geo = pg.get("court") or rep.get("court")
    half = (w - 20) / 2
    for i, label in enumerate(("Where he/she played (time per square metre)", "Shots")):
        _text(qp, QRectF(x + i * (half + 20), y, half, 22), label, 14, True)
    if not geo:
        qp.restore()
        return
    h = rect.bottom() - y - 40
    m1 = CourtMap(QRectF(x, y + 24, half, h), geo["length"], geo["width"])
    draw_court(qp, m1, geo)
    heat = pg["heat"]
    mx = max([v for row in heat for v in row] + [1e-9])
    b = pg["bin_m"]
    qp.save()
    qp.setClipRect(QRectF(m1.xy(0, m1.W), m1.xy(m1.L, 0)), Qt.ClipOperation.IntersectClip)  # last bins overhang
    qp.setPen(Qt.PenStyle.NoPen)
    for j, row in enumerate(heat):
        for i, v in enumerate(row):
            if v <= 0:
                continue
            c = QColor(col)
            c.setAlpha(int(40 + 200 * (v / mx) ** 0.6))
            p0, p1 = m1.xy(i * b, (j + 1) * b), m1.xy((i + 1) * b, j * b)
            qp.fillRect(QRectF(p0, p1), c)
    qp.restore()
    m2 =CourtMap(QRectF(x + half + 20, y + 24, half, h), geo["length"], geo["width"])
    draw_court(qp, m2, geo)
    draw_shots(qp, m2, pg["shots"], _colors(rep), r_px=max(5.0, m2.s * 0.34))
    if not pg["shots"]:
        _text(qp, QRectF(x + half + 20, y + 24, half, h), "No shots attributed", 13, False, DIM,
              Qt.AlignmentFlag.AlignCenter)
    if not pg["tracks"]:
        _text(qp, QRectF(x, y + 24, half, h), "No jersey number read for this player: no positions", 13, False, DIM,
              Qt.AlignmentFlag.AlignCenter)
    qp.restore()


class _PlayerView(QWidget):
    def __init__(self, pg: dict, rep: dict):
        super().__init__()
        self.pg, self.rep = pg, rep
        self.setMinimumSize(900, 460)

    def paintEvent(self, _e) -> None:
        qp = QPainter(self)
        paint_player(qp, QRectF(self.rect()), self.pg, self.rep)
        qp.end()


# ====================================================================================== dialog
class _PageView(QWidget):
    """The report page scaled to the widget width; click a blue number -> playlist."""

    def __init__(self, dlg: "CoachReportDialog"):
        super().__init__()
        self.dlg = dlg
        self.hits = Hits()
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def _scale(self) -> float:
        return max(0.3, self.width() / PAGE_W)

    def resizeEvent(self, e) -> None:
        self.setFixedHeight(int(2 * PAGE_H * self._scale()))  # page 1 + page 2 (possessions, shot quality)
        super().resizeEvent(e)

    def sizeHint(self) -> QSize:
        return QSize(1200, int(2 * 1200 * PAGE_H / PAGE_W))

    def paintEvent(self, _e) -> None:
        qp = QPainter(self)
        s = self._scale()
        qp.scale(s, s)
        self.hits = paint_report(qp, self.dlg.rep, self.dlg.title)
        qp.translate(0, PAGE_H)
        h2 = paint_page2(qp, self.dlg.rep, self.dlg.title, team=self.dlg.f_team.currentData(),
                         outcome=self.dlg.f_outcome.currentData())
        for rect, title, moments in h2:
            self.hits.append((rect.translated(0, PAGE_H), title, moments))
        qp.end()

    def _hit(self, pos):
        s = self._scale()
        return self.hits.at(QPointF(pos.x() / s, pos.y() / s))

    def mouseMoveEvent(self, e) -> None:
        h = self._hit(e.position())
        self.setCursor(Qt.CursorShape.PointingHandCursor if h else Qt.CursorShape.ArrowCursor)
        if h:
            QToolTip.showText(e.globalPosition().toPoint(), f"{h[1]}: {len(h[2])} moment(s) – click to watch")

    def mousePressEvent(self, e) -> None:
        h = self._hit(e.position())
        if h is not None:
            self.dlg.open_playlist(h[1], h[2])


class CoachReportDialog(QDialog):
    def __init__(self, win: Any, p: Any, rep: dict):
        super().__init__(win)
        self.win, self.p, self.rep = win, p, rep
        self.title = p.info().get("name", p.id) if p is not None else ""
        self.setWindowTitle(f"Coach report – {self.title}")
        self.resize(1280, 900)
        lay = QVBoxLayout(self)
        bar = QHBoxLayout()
        self.hint_text = "Click any blue number to watch those moments (a playlist opens in the Playlists panel)."
        # the hint line is behind a ? (its text stays in self.hint_text)
        self.hint_help = HelpIcon(self.hint_text, "Blue numbers")
        bar.addWidget(self.hint_help)
        self.hint = QLabel("")  # feedback line ("Saved ...") - empty until something happens
        self.hint.setWordWrap(True)
        bar.addWidget(self.hint, 1)
        self.player = QComboBox()
        self.player.addItem("Player page…", None)
        for k in C.TEAMS:
            for r in rep["box"][k]["rows"]:
                if r["player"].startswith("#"):
                    self.player.addItem(f"{r['player']} – {rep['teams'][k]['name']}", (k, r["player"]))
        self.player.activated.connect(self._player_page)
        bar.addWidget(self.player)
        # filters of the possession list (page 2)
        self.f_team, self.f_outcome = QComboBox(), QComboBox()
        self.f_team.addItem("All teams", None)
        for k in C.TEAMS:
            self.f_team.addItem(rep["teams"][k]["name"], k)
        self.f_outcome.addItem("All outcomes", None)
        for key, t in C.OUTCOMES:
            self.f_outcome.addItem(t, key)
        for w in (self.f_team, self.f_outcome):
            w.setToolTip("Filter the possession list (page 2)")
            w.currentIndexChanged.connect(lambda _i: self.view.update())
            bar.addWidget(w)
        pdf = QPushButton("Save PDF…")
        pdf.clicked.connect(self.save)
        bar.addWidget(pdf)
        self.export_btn = QPushButton("Export")
        self.export_btn.setToolTip("Box score for a spreadsheet, or a printable PDF with a page per player")
        menu = QMenu(self.export_btn)
        menu.addAction("Box score as CSV (one file per team + team comparison)…", lambda: self.export_box_csv())
        if CE.has_xlsx():
            menu.addAction("Box score as Excel workbook…", lambda: self.export_box_xlsx())
        menu.addSeparator()
        menu.addAction("Printable PDF: report + one page per player…", lambda: self.save_with_players())
        self.export_btn.setMenu(menu)
        bar.addWidget(self.export_btn)
        self.copy_btn = QPushButton("Copy summary")
        self.copy_btn.setToolTip("Copy a short text summary (score, top scorers, shooting, key runs) for a message")
        self.copy_btn.clicked.connect(lambda: self.copy_summary())
        bar.addWidget(self.copy_btn)
        close = QPushButton("Close")
        close.clicked.connect(self.close)
        bar.addWidget(close)
        lay.addLayout(bar)
        self.view = _PageView(self)
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setWidget(self.view)
        lay.addWidget(sc, 1)

    def open_playlist(self, name: str, moments: list) -> None:
        from replay import playlists as pl
        dur = float((self.p.info().get("video_info") or {}).get("duration_s") or 0.0) or None if self.p else None
        playlist = pl.fixed_playlist(name, moments, duration=dur, origin="coach report")
        panel = None
        for d in self.win.findChildren(QDockWidget) if self.win is not None else []:
            if d.objectName() == "PlaylistsDock":
                panel = d.widget()
        if panel is not None and hasattr(panel, "add_playlist"):
            panel.add_playlist(playlist, show=True)
        else:  # no panel (e.g. tests): store it with the Replay
            book = pl.PlaylistBook.load(self.p.root)
            book.add(playlist)
            book.save(self.p.root)
        if self.win is not None and hasattr(self.win, "statusBar"):
            self.win.statusBar().showMessage(f"Playlist “{name}”: {len(playlist.clips)} clip(s)", 6000)
        self.last_playlist = playlist

    def _player_page(self, i: int) -> None:
        data = self.player.itemData(i)
        self.player.setCurrentIndex(0)
        if data:
            open_player_page(self, self.p, self.rep, *data)

    def save(self) -> None:
        start = str(Path.home() / f"{self.title} report.pdf")
        dst, _ = QFileDialog.getSaveFileName(self, "Save coach report", start, "PDF (*.pdf)")
        if not dst:
            return
        if not dst.lower().endswith(".pdf"):
            dst += ".pdf"
        try:
            save_pdf(self.rep, dst, self.title)
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "Coach report", f"Could not save the PDF:\n{e}")
            return
        if self.win is not None and hasattr(self.win, "statusBar"):
            self.win.statusBar().showMessage(f"Saved {dst}", 6000)

    # ------------------------------------------------------------------ exports
    def _note(self, msg: str, ms: int = 8000) -> None:
        """Feedback in the dialog's hint line (the main window's status bar may be hidden behind the dialog)."""
        self.last_note = msg
        self.hint.setText(msg)
        self.hint.setStyleSheet("color: #047857")
        QTimer.singleShot(ms, self._reset_note)
        if self.win is not None and hasattr(self.win, "statusBar"):
            self.win.statusBar().showMessage(msg, ms)

    def _reset_note(self) -> None:
        self.hint.setText("")
        self.hint.setStyleSheet("")

    def _start_name(self, suffix: str) -> str:
        return str(Path.home() / f"{CE._clean_name(self.title or 'Game')} {suffix}")

    @staticmethod
    def csv_format() -> tuple[str, str]:
        """(delimiter, decimal): ';' and ',' where the system uses a decimal comma (e.g. Finnish Excel)."""
        return (";", ",") if QLocale.system().decimalPoint() == "," else (",", ".")

    def export_box_csv(self, base: str | None = None) -> list[Path]:
        if base is None:
            base, _ = QFileDialog.getSaveFileName(
                self, "Export box score (CSV) – the team names are added to the file name",
                self._start_name("box score.csv"), "CSV (*.csv)", options=QFileDialog.Option.DontConfirmOverwrite)
            if not base:
                return []
            exist = [q for q in CE.csv_paths(base, self.rep).values() if q.exists()]
            if exist and QMessageBox.question(self, "Export box score", "Replace these files?\n"
                                              + "\n".join(q.name for q in exist)) != QMessageBox.StandardButton.Yes:
                return []
        delim, dec = self.csv_format()
        try:
            paths = CE.write_csv(self.rep, base, delim, dec)
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "Export box score", f"Could not write the CSV files:\n{e}")
            return []
        self._note(f"Saved {len(paths)} CSV files in {paths[0].parent}: " + ", ".join(q.name for q in paths))
        return paths

    def export_box_xlsx(self, dst: str | None = None) -> Path | None:
        """Only offered when openpyxl is installed (CE.has_xlsx)."""
        if dst is None:
            dst, _ = QFileDialog.getSaveFileName(self, "Export box score (Excel)", self._start_name("box score.xlsx"),
                                                 "Excel (*.xlsx)")
            if not dst:
                return None
        if not dst.lower().endswith(".xlsx"):
            dst += ".xlsx"
        try:
            out = CE.write_xlsx(self.rep, dst)
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "Export box score", f"Could not write the workbook:\n{e}")
            return None
        self._note(f"Saved {out}")
        return out

    def save_with_players(self, dst: str | None = None) -> Path | None:
        """Printable PDF: page 1 the report, then one page per player whose jersey number was read."""
        if dst is None:
            dst, _ = QFileDialog.getSaveFileName(self, "Save printable report with player pages",
                                                 self._start_name("report + players.pdf"), "PDF (*.pdf)")
            if not dst:
                return None
        if not dst.lower().endswith(".pdf"):
            dst += ".pdf"
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            pages = load_player_pages(self.p, self.rep)
            out = save_pdf(self.rep, dst, self.title, players=pages)
        except Exception as e:  # noqa: BLE001
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, "Coach report", f"Could not save the PDF:\n{type(e).__name__}: {e}")
            return None
        QApplication.restoreOverrideCursor()
        self._note(f"Saved {out} (report + {len(pages)} player page(s))")
        return out

    def copy_summary(self) -> str:
        txt = CE.summary_text(self.rep, self.title)
        QGuiApplication.clipboard().setText(txt)
        self._note("Summary copied – paste it into a message (WhatsApp, e-mail, …)")
        return txt


def open_player_page(parent: QWidget, p: Any, rep: dict, team: str, label: str) -> QDialog:
    from replay import playlists as pl
    try:
        catalog = pl.collect(p.root)
        replay = json.loads((p.results / "replay_data.json").read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        QMessageBox.warning(parent, "Player page", f"Could not read the Replay: {e}")
        return None
    pg = C.player_page(rep, catalog, replay, team, label)
    dlg = QDialog(parent)
    dlg.setWindowTitle(f"{label} – {rep['teams'][team]['name']}")
    dlg.resize(1100, 560)
    v = QVBoxLayout(dlg)
    v.addWidget(_PlayerView(pg, rep), 1)
    dlg.show()
    return dlg


def open_report(win: Any, p: Any) -> CoachReportDialog | None:
    """Build the report for Replay p and show it (non-modal)."""
    if p is None or not p.status().get("replay_ready"):
        QMessageBox.information(win, "Coach report", "The coach report needs the 2D replay: build it first.")
        return None
    try:
        rep = C.build(p.root)
    except Exception as e:  # noqa: BLE001
        QMessageBox.warning(win, "Coach report", f"Could not build the report:\n{type(e).__name__}: {e}")
        return None
    dlg = CoachReportDialog(win, p, rep)
    dlg.show()
    return dlg
