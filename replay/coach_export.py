"""Coach report exports without Qt: box score CSV files (one per team + team comparison) and a short plain-text
game summary for messaging apps. Data: :func:`replay.coach.build`. The dialog buttons are in
``replay/gui/coach_report.py``.

Every exported number keeps its "how sure" marker, as a separate ``Source`` column (never glued to the value):
``official`` (from the attached scoresheet), ``measured``, ``inferred``, ``team only`` and ``not measured``
(left empty, never 0) - the same vocabulary as the report's footer.

Excel: .xlsx is written only when openpyxl is installed (it is not in this venv), so the dialog offers CSV.
CSV files are UTF-8 with a BOM (Excel shows ä/ö right); with a decimal comma (Finnish Excel) use
``delimiter=";", decimal=","`` so the file opens in columns.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

from replay import coach as C

# (header, key in the box row, kind, provenance key in C.COLUMNS)
BOX_FIELDS = [
    ("MIN", "min_s", "min", "min"), ("PTS", "pts", "int", "pts"), ("FGM", "fgm", "int", "fg"),
    ("FGA", "fga", "int", "fg"), ("FG%", ("fgm", "fga"), "pct", "fg"), ("3PM", "fg3m", "int", "fg3"),
    ("3PA", "fg3a", "int", "fg3"), ("3P%", ("fg3m", "fg3a"), "pct", "fg3"), ("FTM", "ftm", "int", "ft"),
    ("FTA", "fta", "int", "ft"), ("OREB", "oreb", "int", "oreb"), ("DREB", "dreb", "int", "dreb"),
    ("REB", ("oreb", "dreb"), "sum", "oreb"), ("AST", "ast", "int", "ast"), ("STL", "stl", "int", "stl"),
    ("TOV", "tov", "int", "tov"), ("PF", "pf", "int", "pf"), ("+/-", "pm", "int", "pm"),
]
BOX_HEADER = ["Player", "Name"] + [h for h, *_ in BOX_FIELDS] + ["Source"]
_PROV_WORD = {"measured": "measured", "inferred": "inferred", "team": "team only", "not measured": "not measured"}
_COL_PROV = {k: p for k, _t, p in C.COLUMNS}
_COL_TITLE = {k: t for k, t, _p in C.COLUMNS}
COMPARE_SOURCE = {"points": "measured", "steals": "measured"}  # every other comparison row is inferred


def has_xlsx() -> bool:
    try:
        import openpyxl  # noqa: F401
    except ImportError:
        return False
    return True


def _num(v: float | None, decimal: str, nd: int = 1) -> str:
    if v is None:
        return ""
    s = f"{v:.{nd}f}"
    return s.replace(".", decimal) if decimal != "." else s


def _safe_text(s: Any) -> str:
    """Text cell that a spreadsheet will not run as a formula."""
    s = "" if s is None else str(s)
    return "'" + s if s[:1] in ("=", "+", "-", "@") else s


def _row_label(r: dict) -> str:
    return "Unidentified players" if r["player"] == "Unidentified" else r["player"]


def row_source(r: dict) -> str:
    """Which numbers of this row are official / measured / inferred / not measured, e.g.
    'official: PTS FT PF (video PTS 12) · measured: STL · inferred: MIN FG 3PT OREB DREB AST TOV +/-'."""
    team_row = r["player"] == "Team"
    official = set(r.get("official") or [])
    groups: dict[str, list[str]] = {}
    for key, title, prov in C.COLUMNS:
        if key in official:
            g = "official"
        elif prov == "team":
            g = "measured" if team_row else "team only"
        else:
            g = _PROV_WORD.get(prov, prov)
        if team_row and key in ("min", "pm"):
            continue  # no team minutes / +/-
        groups.setdefault(g, []).append(title)
    order = ("official", "measured", "inferred", "team only", "not measured")
    parts = []
    for g in order:
        if groups.get(g):
            txt = f"{g}: {' '.join(groups[g])}"
            if g == "official":
                txt += " (scoresheet"
                if r.get("pts_detected") is not None and r.get("pts_detected") != r.get("pts")                         and not r.get("sheet_only"):
                    txt += f", video PTS {r['pts_detected']}"
                txt += ")"
            parts.append(txt)
    if r.get("sheet_only"):
        parts.insert(0, "on the scoresheet only (not identified in the video)")
    return " · ".join(parts)


def box_rows(rep: dict, team: str, decimal: str = ".") -> list[list[str]]:
    """Header + one row per player + the Team row. Not-measured cells are empty."""
    out = [list(BOX_HEADER)]
    rows = list(rep["box"][team]["rows"]) + [rep["box"][team]["total"]]
    for r in rows:
        cells = [_safe_text(_row_label(r)), _safe_text(r.get("name") or "")]
        for _h, key, kind, _prov in BOX_FIELDS:
            if kind == "min":
                v = r.get("min_s")
                cells.append(_num(v / 60.0, decimal) if v else "")
            elif kind == "pct":
                m, a = r.get(key[0]), r.get(key[1])
                cells.append(_num(100.0 * m / a, decimal) if m is not None and a else "")
            elif kind == "sum":
                vs = [r.get(k) for k in key]
                cells.append("" if any(v is None for v in vs) else str(sum(vs)))
            else:
                v = r.get(key)  # players' FT only exist when the scoresheet gives them
                cells.append("" if v is None else str(int(v)))
        cells.append(row_source(r))
        out.append(cells)
    return out


def compare_rows(rep: dict, decimal: str = ".") -> list[list[str]]:
    a, b = rep["teams"]["A"]["name"], rep["teams"]["B"]["name"]
    out = [["Stat", _safe_text(a), _safe_text(b), "Source"]]
    for r in rep.get("compare") or []:
        pct = r["key"].endswith("pct")
        title = r["title"] if not pct or r["title"].endswith("%") else r["title"] + " %"
        cells = [title]
        for k in C.TEAMS:
            v = r.get(k)
            if v is None:
                cells.append("")
            elif pct:
                cells.append(_num(100.0 * v, decimal))
            elif isinstance(v, float):
                cells.append(_num(v, decimal, 2))
            else:
                cells.append(str(v))
        cells.append("official (scoresheet)" if r.get("official") else COMPARE_SOURCE.get(r["key"], "inferred"))
        out.append(cells)
    return out


def _clean_name(s: str) -> str:
    s = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", str(s)).strip(" .")
    return s or "team"


def csv_paths(base: Path | str, rep: dict) -> dict[str, Path]:
    """'<stem> - <team A>.csv', '<stem> - <team B>.csv', '<stem> - team comparison.csv' next to base."""
    base = Path(base)
    stem = base.stem if base.suffix.lower() == ".csv" else base.name
    folder = base.parent
    names = {k: _clean_name(rep["teams"][k]["name"]) for k in C.TEAMS}
    if names["A"] == names["B"]:
        names = {k: f"{names[k]} {k}" for k in C.TEAMS}
    out = {k: folder / f"{stem} - {names[k]}.csv" for k in C.TEAMS}
    out["compare"] = folder / f"{stem} - team comparison.csv"
    return out


def write_csv(rep: dict, base: Path | str, delimiter: str = ",", decimal: str = ".") -> list[Path]:
    """Write the three CSV files; returns their paths."""
    paths = csv_paths(base, rep)
    tables = {k: box_rows(rep, k, decimal) for k in C.TEAMS}
    tables["compare"] = compare_rows(rep, decimal)
    for key, path in paths.items():
        with open(path, "w", encoding="utf-8-sig", newline="") as fh:
            csv.writer(fh, delimiter=delimiter).writerows(tables[key])
    return list(paths.values())


def write_xlsx(rep: dict, path: Path | str) -> Path:
    """One workbook: a sheet per team + 'Team comparison'. Needs openpyxl (see :func:`has_xlsx`)."""
    import openpyxl
    from openpyxl.styles import Font

    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    used: set[str] = set()

    def sheet(title: str, rows: list[list[str]]) -> None:
        t = _clean_name(title).replace("[", "(").replace("]", ")")[:31]
        while t in used:
            t = t[:29] + f"_{len(used)}"
        used.add(t)
        ws = wb.create_sheet(t)
        for r in rows:
            vals = []
            for c in r:
                c = c[1:] if c.startswith("'") else c
                try:
                    vals.append(int(c) if re.fullmatch(r"-?\d+", c) else float(c) if re.fullmatch(r"-?\d+\.\d+", c)
                                else c)
                except ValueError:
                    vals.append(c)
            ws.append(vals)
        for cell in ws[1]:
            cell.font = Font(bold=True)
        ws.freeze_panes = "B2"

    for k in C.TEAMS:
        sheet(rep["teams"][k]["name"], box_rows(rep, k))
    sheet("Team comparison", compare_rows(rep))
    wb.save(str(path))
    return Path(path)


# ====================================================================================== summary
def period_of(rep: dict, t: float | None) -> str:
    if t is None:
        return ""
    rows = (rep.get("period_table") or {}).get("rows") or []
    if len(rows) < 2:  # one span = the whole Replay: its label says nothing about when
        return ""
    for r in rows:
        if r.get("t0") is not None and r.get("t1") is not None and r["t0"] <= t < r["t1"]:
            return str(r["label"]).replace(" (from the scoresheet)", "")
    return ""


def clock(t: float) -> str:
    t = max(0, int(round(t)))
    h, rest = divmod(t, 3600)
    m, s = divmod(rest, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def summary_text(rep: dict, title: str = "", top: int = 3, max_runs: int = 3) -> str:
    """Short plain-text summary: score, top scorers, team shooting, key runs."""
    names = {k: rep["teams"][k]["name"] for k in C.TEAMS}
    sc = rep["score"]
    sh = rep.get("sheet")
    if sh and sh.get("full_game") and rep.get("score_official"):
        how = "final, from the scoresheet"
    elif sh and rep.get("score_official_window"):
        how = "this part of the game, from the scoresheet"
    else:
        how = "detected from the video"
    lines = []
    if title:
        lines.append(title)
    lines.append(f"{names['A']} {sc['A']} – {sc['B']} {names['B']} ({how})")
    lines.append("")
    lines.append("Top scorers")
    for k in C.TEAMS:
        rows = [r for r in rep["box"][k]["rows"] if r["player"] != "Unidentified" and (r.get("pts") or 0) > 0]
        rows.sort(key=lambda r: -(r.get("pts") or 0))
        txt = ", ".join(f"{r['player']}{(' ' + r['name']) if r.get('name') else ''} {r['pts']}" for r in rows[:top])
        lines.append(f"{names[k]}: {txt or '–'}")
    lines.append("")
    lines.append("Shooting (detected)")
    for k in C.TEAMS:
        t = rep["box"][k]["total"]
        fgm, fga, m3, a3 = (int(t.get(x) or 0) for x in ("fgm", "fga", "fg3m", "fg3a"))
        parts = [f"FG {fgm}/{fga}" + (f" ({100.0 * fgm / fga:.0f}%)" if fga else ""),
                 f"3PT {m3}/{a3}" + (f" ({100.0 * m3 / a3:.0f}%)" if a3 else "")]
        if t.get("fta"):
            parts.append(f"FT {int(t.get('ftm') or 0)}/{int(t['fta'])}")
        lines.append(f"{names[k]}: " + ", ".join(parts))
    runs = sorted(rep.get("runs") or [], key=lambda r: (-r["points"], r["t0"]))[:max_runs]
    if runs:
        lines.append("")
        lines.append("Key runs")
        for r in sorted(runs, key=lambda r: r["t0"]):
            per = period_of(rep, r["t0"])
            where = f"{per}, " if per else ""
            lines.append(f"{names[r['team']]} {r['points']}-0 ({where}video {clock(r['t0'])}–{clock(r['t1'])})")
    lines.append("")
    lines.append("From AdvancedAnalytist video analysis; detected numbers can be off.")
    return "\n".join(lines)
