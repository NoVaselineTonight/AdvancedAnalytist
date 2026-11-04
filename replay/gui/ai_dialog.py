"""GUI for the AI tactical analysis (Gemini): settings dialog with a background job, results viewer, timeline track.

Entry points used by app.py: :func:`add_menu_actions` (menu bar and the Replays list right-click menu),
:func:`open_ai_analysis`, :func:`show_ai_results`.

The API key is remembered per Windows user in QSettings (the registry under HKCU), encrypted with Windows DPAPI
when available - never in the Replay folder, never in results or exports, never logged.
"""

from __future__ import annotations

import base64
import html
import json
import sys
import threading
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import QObject, QSettings, Qt, QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices, QIcon, QPixmap
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
                               QFormLayout, QHBoxLayout, QLabel, QLineEdit, QMenu, QMessageBox, QPlainTextEdit,
                               QProgressBar, QPushButton, QRadioButton, QTextBrowser, QVBoxLayout, QWidget)

from replay import ai_analysis as ai
from replay import ai_light
from replay.gui import timeline_model as tm

if TYPE_CHECKING:
    from replay.gui.app import MainWindow
    from replay.gui.projects import Project

KEY_SETTING = "ai/gemini_api_key"
MODEL_SETTING = "ai/model"
LIGHT_KEY_SETTING = "ai/typesafe_api_key"
LIGHT_URL_SETTING = "ai/typesafe_base_url"
LIGHT_MODEL_SETTING = "ai/typesafe_model"
TIER_LABELS = {"python": "Python (on this computer)", "light": "Light-work model (TypeSafe AI Jev)",
               "gemini": "Gemini"}
PHASE_LABELS = {"offense": "Offense", "defense": "Defense", "transition": "Transition", "other": "Other"}


# ====================================================================== key storage (per OS user)
def _dpapi(data: bytes, protect: bool) -> bytes | None:
    """Windows DPAPI (CryptProtectData / CryptUnprotectData) for the current user; None elsewhere or on error."""
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        class Blob(ctypes.Structure):
            _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_char))]

        buf = ctypes.create_string_buffer(data, len(data))
        src, dst = Blob(len(data), ctypes.cast(buf, ctypes.POINTER(ctypes.c_char))), Blob()
        crypt = ctypes.windll.crypt32
        fn = crypt.CryptProtectData if protect else crypt.CryptUnprotectData
        if not fn(ctypes.byref(src), None, None, None, None, 0x1, ctypes.byref(dst)):  # 0x1 = UI_FORBIDDEN
            return None
        try:
            return ctypes.string_at(dst.pbData, dst.cbData)
        finally:
            ctypes.windll.kernel32.LocalFree(dst.pbData)
    except Exception:
        return None


def load_api_key(settings: QSettings, name: str = KEY_SETTING) -> str:
    v = settings.value(name, "")
    if not isinstance(v, str) or not v:
        return ""
    kind, _, payload = v.partition(":")
    try:
        raw = base64.b64decode(payload)
    except ValueError:
        return ""
    if kind == "dpapi":
        out = _dpapi(raw, protect=False)
        return out.decode("utf-8") if out else ""
    return raw.decode("utf-8", "replace") if kind == "plain" else ""


def save_api_key(settings: QSettings, key: str | None, name: str = KEY_SETTING) -> None:
    key = (key or "").strip()
    if not key:
        settings.remove(name)
        return
    enc = _dpapi(key.encode("utf-8"), protect=True)
    settings.setValue(name, ("dpapi:" + base64.b64encode(enc).decode()) if enc
                      else ("plain:" + base64.b64encode(key.encode("utf-8")).decode()))


# ====================================================================== teams / colours
def project_teams(win: "MainWindow", p: "Project") -> dict[str, dict]:
    """{"A": {"name", "color"}, "B": ...} with the colours the 2D view uses for this Replay."""
    rt = getattr(win, "replay_tab", None)
    colors: dict[str, str] = {}
    data: dict = {}
    if rt is not None and getattr(rt, "p", None) is not None and rt.p.id == p.id and getattr(rt, "data", None):
        colors, data = rt.display_colors(), rt.data
    else:
        try:
            data = json.loads((p.results / "replay_data.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
        colors = {k: v.get("primary") for k, v in (data.get("game", {}).get("teams", {}) or {}).items()}
        m = p.meta()
        if m.get("team_colors_auto") is False:
            jc = m.get("jersey_colors", {})
            colors.update({"A": jc.get("home", {}).get("primary", colors.get("A")),
                           "B": jc.get("away", {}).get("primary", colors.get("B"))})
        colors.update(p.settings().get("display_colors") or {})
    from replay import team_binding as tbind

    teams = (data.get("game", {}) or {}).get("teams", {}) or {}
    ov = tbind.team_overview(p.meta())  # the same colour-bound names/colours as the Overview tab
    return {k: {"name": str((teams.get(k) or {}).get("name") or ov[k]["name"]),
                "color": colors.get(k) or (teams.get(k) or {}).get("primary") or ov[k]["color"],
                "jersey": str((teams.get(k) or {}).get("colour_name") or ov[k]["colour_name"]),
                "side": tbind.side_text(teams.get(k) or ov[k], k)} for k in ("A", "B")}


def _swatch(color: str) -> QIcon:
    pm = QPixmap(18, 18)
    pm.fill(QColor(color))
    return QIcon(pm)


# ====================================================================== background job
class _Job(QObject):
    progress = Signal(float, str)
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, root: Path, opts: ai.Options, key: str, light: tuple[str, str, str] | None = None):
        super().__init__()
        self.root, self.opts, self._key = root, opts, key
        self._light = light  # (key, base url, model) or None
        self.cancel = threading.Event()

    def start(self) -> None:
        threading.Thread(target=self._run, daemon=True, name="ai-analysis").start()

    def _run(self) -> None:
        key, self._key = self._key, ""
        light, self._light = self._light, None
        try:
            client = ai.GeminiClient(key)
            jev = ai_light.JevClient(*light) if light else None
            res = ai.run_analysis(self.root, self.opts, client, lambda f, m: self.progress.emit(f, m), self.cancel,
                                  light_client=jev)
            if not self.cancel.is_set():
                self.done.emit(res)
        except ai.Cancelled:
            pass
        except (ai.AIError, ai_light.LightError) as e:
            if not self.cancel.is_set():
                self.failed.emit(str(e))
        except ValueError as e:  # e.g. no tracking data
            self.failed.emit(str(e))
        except Exception as e:  # never include the key: messages from our code paths are already scrubbed
            self.failed.emit(ai._scrub(ai._scrub(f"{type(e).__name__}: {e}", key), light[0] if light else None))


# ====================================================================== settings dialog
class AIAnalysisDialog(QDialog):
    def __init__(self, win: "MainWindow", p: "Project"):
        super().__init__(win)
        self.win, self.p = win, p
        self.job: _Job | None = None
        self.setWindowTitle(f"AI analysis – {p.info().get('name', p.id)}")
        self.setMinimumWidth(560)
        qs: QSettings = win.settings
        prev = (p.settings().get("ai") or {}) if isinstance(p.settings().get("ai"), dict) else {}
        self.teams = project_teams(win, p)

        lay = QVBoxLayout(self)
        intro = QLabel("Sends the 2D tracking data of this Replay (positions only, no video) to Google's Gemini "
                       "API with your own API key and asks for a tactical review. The issues appear on the "
                       f"<b>{ai.AI_TRACK_NAME}</b> timeline track.")
        intro.setWordWrap(True)
        lay.addWidget(intro)
        form = QFormLayout()
        lay.addLayout(form)

        # --- key
        self.key = QLineEdit(load_api_key(qs))
        self.key.setEchoMode(QLineEdit.EchoMode.Password)
        self.key.setPlaceholderText("Google AI Studio API key")
        show = QPushButton("Show")
        show.setCheckable(True)
        show.toggled.connect(lambda on: self.key.setEchoMode(QLineEdit.EchoMode.Normal if on
                                                             else QLineEdit.EchoMode.Password))
        get = QLabel(f"<a href='{ai.API_KEY_URL}'>Get a key</a>")
        get.setOpenExternalLinks(True)
        row = QHBoxLayout()
        row.addWidget(self.key, 1)
        row.addWidget(show)
        row.addWidget(get)
        form.addRow("API key:", self._wrap(row))
        self.remember = QCheckBox("Remember the keys on this computer (for this Windows user only, encrypted)")
        self.remember.setChecked(bool(self.key.text()) or qs.value("ai/remember", "true") in (True, "true"))
        form.addRow("", self.remember)

        # --- team
        self.team_group = QButtonGroup(self)
        trow = QHBoxLayout()
        for k in ("A", "B"):
            tk = self.teams[k]
            jersey = tk.get("jersey") or ""
            info = ", ".join(x for x in (f"{jersey} jerseys" if jersey and jersey not in tk["name"] else "",
                                         tk.get("side") or ("home" if k == "A" else "away")) if x)
            rb = QRadioButton(f"{tk['name']}  ({info})")
            rb.setIcon(_swatch(self.teams[k]["color"]))
            rb.setProperty("team", k)
            self.team_group.addButton(rb)
            trow.addWidget(rb)
            if prev.get("coached_team") == k:
                rb.setChecked(True)
        trow.addStretch(1)
        form.addRow("Which team are you coaching?", self._wrap(trow))

        # --- level context
        self.context = QPlainTextEdit(str(prev.get("level_context") or ""))
        self.context.setPlaceholderText(
            "Level and context, e.g. \"NAISTEN 1D, we play mostly man-to-man, working on transition defence\" "
            "or \"WU13 1D – first season of 5v5\". Write in the language you want the answer in.")
        self.context.setFixedHeight(80)
        form.addRow("Context for skill level:", self.context)

        # --- model / sampling
        self.model = QComboBox()
        self.model.setEditable(True)
        self.model.addItems(ai.MODELS)
        self.model.setCurrentText(str(prev.get("model") or qs.value(MODEL_SETTING, ai.DEFAULT_MODEL)))
        self.model.setToolTip("Gemini model name (see ai.google.dev/gemini-api/docs/models). You can type another.")
        self.hz = QDoubleSpinBox()
        self.hz.setRange(0.5, 5.0)
        self.hz.setSingleStep(0.5)
        self.hz.setSuffix(" samples/s")
        self.hz.setValue(float(prev.get("sample_hz") or ai.DEFAULT_SAMPLE_HZ))
        self.hz.setToolTip("How often positions are sent. Higher = more detail but more tokens (cost/quota).")
        mrow = QHBoxLayout()
        mrow.addWidget(self.model, 1)
        mrow.addWidget(QLabel("Detail:"))
        mrow.addWidget(self.hz)
        form.addRow("Model:", self._wrap(mrow))

        # --- light-work model (TypeSafe AI Jev): optional cheap tier
        light = QLabel("<b>Light-work model (TypeSafe AI Jev)</b> – optional. Labels the game segments "
                       "(phase, defensive scheme, worth reviewing, data reliability) and uncertain tracks, so Gemini "
                       "gets only the detail it needs. Without a key this is done by rules on this computer and "
                       "Gemini gets the full detail.")
        light.setWordWrap(True)
        form.addRow(light)
        self.light_key = QLineEdit(load_api_key(qs, LIGHT_KEY_SETTING))
        self.light_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.light_key.setPlaceholderText("TypeSafe AI API key (leave empty to skip)")
        lshow = QPushButton("Show")
        lshow.setCheckable(True)
        lshow.toggled.connect(lambda on: self.light_key.setEchoMode(QLineEdit.EchoMode.Normal if on
                                                                   else QLineEdit.EchoMode.Password))
        lrow = QHBoxLayout()
        lrow.addWidget(self.light_key, 1)
        lrow.addWidget(lshow)
        form.addRow("Light-work key:", self._wrap(lrow))
        self.light_url = QLineEdit(str(qs.value(LIGHT_URL_SETTING, ai_light.DEFAULT_BASE_URL)))
        self.light_model = QComboBox()
        self.light_model.setEditable(True)
        self.light_model.addItems(ai_light.MODELS)
        self.light_model.setCurrentText(str(qs.value(LIGHT_MODEL_SETTING, ai_light.DEFAULT_MODEL)))
        urow = QHBoxLayout()
        urow.addWidget(self.light_url, 2)
        urow.addWidget(QLabel("Model:"))
        urow.addWidget(self.light_model, 1)
        form.addRow("Light-work API:", self._wrap(urow))

        # --- progress
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.hide()
        lay.addWidget(self.bar)
        lay.addWidget(self.status)

        self.buttons = QDialogButtonBox()
        self.run_btn = self.buttons.addButton("Run analysis", QDialogButtonBox.ButtonRole.AcceptRole)
        self.prev_btn = self.buttons.addButton("Show last result", QDialogButtonBox.ButtonRole.ActionRole)
        self.prev_btn.setEnabled(ai.load_results(p.root) is not None)
        self.cancel_btn = self.buttons.addButton(QDialogButtonBox.StandardButton.Close)
        self.run_btn.clicked.connect(self.start)
        self.prev_btn.clicked.connect(lambda: show_ai_results(self.win, self.p))
        self.cancel_btn.clicked.connect(self._cancel_or_close)
        lay.addWidget(self.buttons)

    @staticmethod
    def _wrap(layout) -> QWidget:
        w = QWidget()
        layout.setContentsMargins(0, 0, 0, 0)
        w.setLayout(layout)
        return w

    def _team(self) -> str | None:
        b = self.team_group.checkedButton()
        return b.property("team") if b is not None else None

    def _set_running(self, on: bool) -> None:
        for w in (self.key, self.context, self.model, self.hz, self.remember, self.run_btn, self.prev_btn,
                  self.light_key, self.light_url, self.light_model):
            w.setEnabled(not on)
        for b in self.team_group.buttons():
            b.setEnabled(not on)
        self.bar.setVisible(on)
        self.cancel_btn.setText("Cancel" if on else "Close")

    def start(self) -> None:
        key, team = self.key.text().strip(), self._team()
        if not key:
            self.status.setText("<span style='color:#b91c1c'>Enter your Google AI Studio API key.</span>")
            return
        if team is None:
            self.status.setText("<span style='color:#b91c1c'>Choose the team you are coaching.</span>")
            return
        model = self.model.currentText().strip() or ai.DEFAULT_MODEL
        qs: QSettings = self.win.settings
        save_api_key(qs, key if self.remember.isChecked() else None)
        lkey = self.light_key.text().strip()
        save_api_key(qs, lkey if self.remember.isChecked() else None, LIGHT_KEY_SETTING)
        lurl = self.light_url.text().strip() or ai_light.DEFAULT_BASE_URL
        lmodel = self.light_model.currentText().strip() or ai_light.DEFAULT_MODEL
        qs.setValue(LIGHT_URL_SETTING, lurl)
        qs.setValue(LIGHT_MODEL_SETTING, lmodel)
        if lkey and not lurl.startswith("https://"):
            self.status.setText("<span style='color:#b91c1c'>The light-work API URL must start with https://</span>")
            return
        qs.setValue("ai/remember", "true" if self.remember.isChecked() else "false")
        qs.setValue(MODEL_SETTING, model)
        st = self.p.settings()
        st["ai"] = {"coached_team": team, "level_context": self.context.toPlainText().strip(), "model": model,
                    "sample_hz": self.hz.value()}
        self.p.update_info(settings=st)
        opts = ai.Options(coached_team=team, level_context=st["ai"]["level_context"], model=model,
                          sample_hz=self.hz.value(),
                          team_colors={k: v["color"] for k, v in self.teams.items()})
        self.job = _Job(self.p.root, opts, key, (lkey, lurl, lmodel) if lkey else None)
        self.job.progress.connect(self._progress)
        self.job.done.connect(self._done)
        self.job.failed.connect(self._failed)
        self._set_running(True)
        self._progress(0.0, "Starting")
        self.job.start()

    def _stale(self) -> bool:
        """A signal from a job that was cancelled (or replaced): it may still finish or fail late - ignore it."""
        return self.sender() is not None and self.sender() is not self.job

    def _progress(self, f: float, msg: str) -> None:
        if self._stale():
            return
        self.bar.setValue(int(max(0.0, min(1.0, f)) * 1000))
        self.status.setText(html.escape(msg))

    def _failed(self, msg: str) -> None:
        if self._stale():
            return
        self.job = None
        self._set_running(False)
        self.status.setText(f"<span style='color:#b91c1c'><b>AI analysis failed.</b> {html.escape(msg)}</span>")

    def _done(self, result: dict) -> None:
        if self._stale():
            return
        self.job = None
        self._set_running(False)
        try:
            ai.save_results(self.p.root, result)
            n = apply_results_to_timeline(self.win, self.p, result)
        except OSError as e:
            self.status.setText(f"<span style='color:#b91c1c'>Could not save the result: {html.escape(str(e))}</span>")
            return
        self.win.statusBar().showMessage(f"AI analysis ready: {n} issue(s) on the '{ai.AI_TRACK_NAME}' track", 6000)
        self.accept()
        show_ai_results(self.win, self.p)

    def _cancel_or_close(self) -> None:
        if self.job is not None:
            self.job.cancel.set()
            self.job = None  # its late signals are ignored (it stops after the current request returns)
            self._set_running(False)
            self.status.setText("Cancelled.")
        else:
            self.reject()

    def closeEvent(self, e) -> None:
        if self.job is not None:
            self.job.cancel.set()
            self.job = None
        super().closeEvent(e)

    def reject(self) -> None:
        if self.job is not None:
            self.job.cancel.set()
            self.job = None
        super().reject()


# ====================================================================== results viewer
def _fmt_t(s: float) -> str:
    m, sec = divmod(max(0.0, float(s)), 60)
    return f"{int(m)}:{int(sec):02d}"


def results_html(res: dict) -> str:
    e = html.escape
    ov = res.get("overview") or {}
    ct = res.get("coached_team") or {}
    out = [f"<h3>AI analysis – <span style='color:{e(str(ct.get('color', '')))}'>&#9632;</span> "
           f"{e(str(ct.get('name', '')))}</h3>",
           f"<p style='color:gray'>{e(str(res.get('model', '')))} · {e(str(res.get('created', ''))[:16])}"
           + (f" · level: {e(str(res.get('level_context')))}" if res.get("level_context") else "") + "</p>"]
    if ov.get("summary"):
        out.append("<p>" + e(ov["summary"]).replace("\n", "<br>") + "</p>")
    for key, label, col in (("went_well", "Went well", "#15803d"), ("to_improve", "To improve", "#b45309")):
        if ov.get(key):
            out.append(f"<h4 style='color:{col}'>{label}</h4><ul>"
                       + "".join(f"<li>{e(s)}</li>" for s in ov[key]) + "</ul>")
    issues = res.get("issues") or []
    out.append(f"<h4>Issues ({len(issues)}) – click a time to jump there</h4>")
    for it in issues:
        a, b = float(it.get("t_start", 0)), float(it.get("t_end", 0))
        phase = str(it.get("team_phase") or "other")
        col = ai.PHASE_COLORS.get(phase, "#6b7280")
        out.append(f"<p><a href='t:{a}'>{_fmt_t(a)}–{_fmt_t(b)}</a> "
                   f"<span style='color:{col}'><b>[{e(PHASE_LABELS.get(phase, phase))}]</b></span> "
                   f"<b>{e(str(it.get('title', '')))}</b> <i style='color:gray'>{e(str(it.get('category', '')))}</i>"
                   f"<br>{e(str(it.get('description', '')))}</p>")
    gs = res.get("game_state") or {}
    if gs.get("attack_direction"):
        names = {"A": "", "B": ""}
        names[(res.get("coached_team") or {}).get("label", "A")] = str(ct.get("name", ""))
        names[(res.get("opponent") or {}).get("label", "B")] = str((res.get("opponent") or {}).get("name", ""))
        dirs = "".join(
            f"<li>{e(str(d.get('label')))}: {e(names.get('A') or 'A')} → basket {e(str(d.get('A')))}, "
            f"{e(names.get('B') or 'B')} → basket {e(str(d.get('B')))} (confidence {float(d.get('confidence') or 0):.2f})"
            "</li>" if d.get("A") else f"<li>{e(str(d.get('label')))}: attack direction unknown</li>"
            for d in gs["attack_direction"])
        kinds = ", ".join(f"{e(str(k).replace('_', ' '))} {v}" for k, v in sorted((gs.get("events_by_type") or {}).items()))
        nl = len(res.get("light_events") or [])
        out.append(f"<h4>Game state given to the AI</h4><ul>{dirs}<li>{gs.get('possessions', 0)} possessions, "
                   f"{gs.get('free_throw_situations', 0)} free-throw situation(s); events: {kinds or 'none'}"
                   + (f"; {nl} lost ball(s) labelled by the light model" if nl else "") + "</li></ul>")
    if res.get("tiers"):
        out.append("<h4>Who did what</h4><ul>" + "".join(
            f"<li><b>{e(TIER_LABELS.get(str(t.get('tier')), str(t.get('tier'))))}</b> – {e(str(t.get('task', '')))}: "
            f"{e(str(t.get('detail', '')))}</li>" for t in res["tiers"]) + "</ul>")
    if res.get("warnings"):
        out.append("<h4 style='color:gray'>Notes about the data</h4><ul style='color:gray'>"
                   + "".join(f"<li>{e(str(w))}</li>" for w in res["warnings"]) + "</ul>")
    return "".join(out)


class AIResultsDialog(QDialog):
    def __init__(self, win: "MainWindow", p: "Project", res: dict):
        super().__init__(win)
        self.win, self.p = win, p
        self.setWindowTitle(f"AI analysis – {p.info().get('name', p.id)}")
        self.resize(720, 760)
        lay = QVBoxLayout(self)
        view = QTextBrowser()
        view.setOpenLinks(False)
        view.anchorClicked.connect(self._link)
        view.setHtml(results_html(res))
        lay.addWidget(view, 1)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        rerun = bb.addButton("Run again…", QDialogButtonBox.ButtonRole.ActionRole)
        rerun.clicked.connect(lambda: (self.close(), open_ai_analysis(self.win, self.p)))
        bb.rejected.connect(self.close)
        lay.addWidget(bb)

    def _link(self, url: QUrl) -> None:
        s = url.toString()
        if not s.startswith("t:"):
            QDesktopServices.openUrl(url)
            return
        try:
            t = float(s[2:])
        except ValueError:
            return
        win = self.win
        if win.current is None or win.current.id != self.p.id:
            return
        win.tabs.setCurrentWidget(win.replay_tab)
        win.replay_tab.seek_to(t)


# ====================================================================== glue
def apply_results_to_timeline(win: "MainWindow", p: "Project", result: dict) -> int:
    """Replace the marks of the AI analysis track; through the timeline widget if it shows this Replay."""
    rt = getattr(win, "replay_tab", None)
    widget = getattr(rt, "timeline", None) if rt is not None else None
    if widget is not None and getattr(widget, "root", None) is not None and \
            Path(widget.root).resolve() == Path(p.root).resolve():
        tr = ai.apply_to_timeline(widget.tl, result)
        widget.changed()
        widget.refresh_auto()  # the light model's events (e.g. lost balls) join the "Auto detection" track
    else:
        tl = tm.load(p.root)
        tr = ai.apply_to_timeline(tl, result)
        tm.save(tl, p.root)
    return len(tr.marks)


def _ready(p: "Project | None") -> bool:
    try:
        return p is not None and bool(p.status()["replay_ready"])
    except Exception:
        return False


def open_ai_analysis(win: "MainWindow", p: "Project | None" = None) -> None:
    p = p or win.current
    if p is None:
        QMessageBox.information(win, "AI analysis", "Select a Replay first.")
        return
    if not _ready(p):
        QMessageBox.information(win, "AI analysis", "The AI analysis needs the 2D replay: analyse the video and "
                                                    "calibrate the court first.")
        return
    AIAnalysisDialog(win, p).exec()


def show_ai_results(win: "MainWindow", p: "Project | None" = None) -> None:
    p = p or win.current
    res = ai.load_results(p.root) if p is not None else None
    if res is None:
        QMessageBox.information(win, "AI analysis", "No AI analysis has been run for this Replay yet.")
        return
    dlg = AIResultsDialog(win, p, res)
    dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
    dlg.show()


def add_menu_actions(menu: QMenu, win: "MainWindow", p: "Project | None") -> None:
    """'AI analysis…' and 'Show AI analysis' for Replay ``p`` (None = the selected one when triggered)."""
    target = (lambda: p) if p is not None else (lambda: win.current)
    a = menu.addAction("AI analysis…")
    a.setToolTip("Tactical review of the 2D tracking by Google Gemini (your own API key)")
    a.triggered.connect(lambda: open_ai_analysis(win, target()))
    b = menu.addAction("Show AI analysis")
    b.triggered.connect(lambda: show_ai_results(win, target()))
    if p is not None:
        a.setEnabled(_ready(p))
        b.setEnabled(ai.load_results(p.root) is not None)
