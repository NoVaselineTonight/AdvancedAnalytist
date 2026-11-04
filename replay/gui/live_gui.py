"""GUI side of live mode (replay/live.py): a "Live" menu, a status line and live reloading of the Replay tab.

* Live > New live Replay from a recording... : pick the file OBS / the capture card is writing (.ts, .mkv,
  fragmented .mp4) or a local HLS playlist (.m3u8); optionally reuse the court of an earlier Replay of the same
  gym. A separate ``python -m replay live`` process analyses it while it grows.
* Live > Live analysis of this Replay (toggle): stop it (the analysis so far is finished like a normal one),
  or resume it.
* Live > Follow the live edge: after each 2D update the playhead jumps to the newest analysed moment;
  otherwise the playhead stays where it is (the data under it is reloaded).

Hooked into the main window with one line in app.py (``live_gui.install(self)``).
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

from PySide6.QtCore import QTimer, QUrl
from PySide6.QtGui import QAction
from PySide6.QtMultimedia import QMediaPlayer
from PySide6.QtWidgets import QCheckBox, QFileDialog, QInputDialog, QLabel, QMessageBox

from replay import live
from replay.gui.projects import Project

RECORDING_FILTER = ("Recordings and HLS playlists (*.ts *.mts *.m2ts *.mkv *.mp4 *.flv *.mov *.m3u8);;"
                    "All files (*)")


def live_source(p: Project) -> str | None:
    """The recording a live Replay follows (None for ordinary Replays)."""
    lv = p.info().get("live")
    return lv.get("source") if isinstance(lv, dict) else None


def live_command(p: Project) -> list[str]:
    cfg_path = live.live_dir(p.results) / "config.json"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(p.config().to_dict(), indent=2), encoding="utf-8")
    return [sys.executable, "-m", "replay", "live", "--video", str(live_source(p)), "--out", str(p.results),
            "--calibration", str(p.calibration_path), "--meta", str(p.meta_path), "--config", str(cfg_path),
            "--on-restart", "stop", "--no-final-build", "--player-link", str(p.video)]


def start_live(p: Project) -> subprocess.Popen:
    """Start (or resume) the live analysis process of Replay ``p``; output goes to results/live/process.log."""
    p.results.mkdir(parents=True, exist_ok=True)
    ld = live.live_dir(p.results)
    ld.mkdir(parents=True, exist_ok=True)
    (ld / live.STOP_FILE).unlink(missing_ok=True)
    (p.results / "PAUSED").unlink(missing_ok=True)
    log = open(ld / "process.log", "a", encoding="utf-8")
    root = Path(live.__file__).resolve().parent.parent
    return subprocess.Popen(live_command(p), cwd=str(root), stdout=log, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


def describe(st: dict | None, out: Path) -> str:
    """One status line for the status bar."""
    if not st:
        return ""
    state = st.get("state")
    if state == "finished":
        return "Live analysis finished"
    if state == "error":
        return f"<span style='color:#b91c1c'>Live analysis failed: {st.get('error')}</span>"
    if not live.is_running(out):
        return ("<span style='color:#b45309'>Live analysis is not running</span> "
                "(Live menu → Live analysis of this Replay to resume)")
    parts = ["<b style='color:#dc2626'>● LIVE</b>"]
    if st.get("analysed_t"):
        parts.append(f"analysed to {_mmss(st['analysed_t'])}")
    if st.get("replay_t") is not None:
        parts.append(f"2D to {_mmss(st['replay_t'])}")
        if st.get("behind_s") is not None:
            parts.append(f"{st['behind_s']:.0f} s behind the analysis")
    elif state in ("analysing", "waiting"):
        parts.append(st.get("message") or "")
    if st.get("calibration"):
        parts.append(st["calibration"])
    return " · ".join(p for p in parts if p)


def _mmss(t: float) -> str:
    t = float(t)
    return f"{int(t // 60)}:{t % 60:04.1f}"


class LiveController:
    def __init__(self, win):
        self.win = win
        self.procs: dict[str, subprocess.Popen] = {}
        self.data_mtime: dict[str, float] = {}
        self.built: set[str] = set()
        self.label = QLabel()
        self.follow = QCheckBox("Follow live edge")
        self.follow.setToolTip("After each 2D update, jump to the newest analysed moment")
        self.follow.setChecked(win.settings.value("live_follow", True, type=bool))
        self.follow.toggled.connect(lambda on: win.settings.setValue("live_follow", on))
        sb = win.statusBar()
        sb.addPermanentWidget(self.label)
        sb.addPermanentWidget(self.follow)
        self.label.hide()
        self.follow.hide()
        self.menu = win.menuBar().addMenu("&Live")
        self.menu.aboutToShow.connect(self.fill_menu)
        self.fill_menu()
        self.timer = QTimer(win)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.tick)
        self.timer.start()
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self.shutdown)

    # ------------------------------------------------------------------ menu
    def fill_menu(self) -> None:
        m = self.menu
        m.clear()
        a = m.addAction("New live Replay from a recording…")
        a.setToolTip("Analyse a recording while it is still being written (OBS / capture card .ts, .mkv, "
                     "fragmented .mp4, or a local HLS .m3u8 playlist)")
        a.triggered.connect(self.new_live_replay)
        p = self.win.current
        a = QAction("Live analysis of this Replay", m)
        a.setCheckable(True)
        src = live_source(p) if p is not None else None
        a.setEnabled(bool(src))
        a.setChecked(bool(p is not None and src and live.is_running(p.results)))
        if not src:
            a.setToolTip("Only for Replays made with 'New live Replay'")
        a.toggled.connect(lambda on, p=p: self.toggle(p, on))
        m.addAction(a)
        a = QAction("Follow the live edge", m)
        a.setCheckable(True)
        a.setChecked(self.follow.isChecked())
        a.toggled.connect(self.follow.setChecked)
        m.addAction(a)

    def new_live_replay(self) -> None:
        win = self.win
        start = win.settings.value("last_live_dir", str(Path.home()))
        path, _ = QFileDialog.getOpenFileName(win, "Recording that is being written", start, RECORDING_FILTER)
        if not path:
            return
        win.settings.setValue("last_live_dir", str(Path(path).parent))
        name, ok = QInputDialog.getText(win, "New live Replay", "Name:", text=f"Live {Path(path).stem}")
        if not ok:
            return
        courts = {f"{q.info().get('name', q.id)}": q for q in win.store.list()
                  if q.calibration_summary().get("auto_calibrated")}
        reuse = None
        if courts:
            first = "Find the court automatically (after about a minute of play)"
            item, ok = QInputDialog.getItem(win, "New live Replay", "Same gym and camera as an earlier Replay?",
                                            [first, *courts], 0, False)
            if not ok:
                return
            reuse = courts.get(item)
        p = win.store.create(name.strip() or Path(path).stem,
                             {"filename": Path(path).name, "kind": "reference", "path": str(Path(path).resolve())})
        p.update_info(live={"source": str(Path(path).resolve()), "started": time.time()})
        try:
            from replay.video import probe

            i = probe(path)
            p.update_info(video_info={"duration_s": i.duration_s, "size": [i.width, i.height], "fps": i.fps})
        except Exception:  # noqa: BLE001 - nothing readable yet (OBS just started): set when known
            pass
        if reuse is not None:
            try:
                live.reuse_court(reuse.calibration_path, p.calibration_path)
            except SystemExit as e:
                QMessageBox.warning(win, "Live", str(e))
        self.procs[p.id] = start_live(p)
        win.show_project(p, win.tabs.indexOf(win.replay_tab))

    def toggle(self, p: Project | None, on: bool) -> None:
        if p is None or not live_source(p):
            return
        if on and not live.is_running(p.results):
            if (p.results / "work" / "DONE").exists():
                QMessageBox.information(self.win, "Live", "The live analysis of this Replay has finished. Start a "
                                                          "new live Replay for a new recording.")
                return
            self.procs[p.id] = start_live(p)
        elif not on and live.is_running(p.results):
            live.request_stop(p.results)

    # ------------------------------------------------------------------ periodic
    def tick(self) -> None:
        p = self.win.current
        for pid, proc in list(self.procs.items()):
            if proc.poll() is not None:
                self.procs.pop(pid, None)
        if p is None or not (live.live_dir(p.results)).exists():
            self.label.hide()
            self.follow.hide()
            return
        st = live.read_status(p.results)
        self.label.setText(describe(st, p.results))
        self.label.show()
        running = live.is_running(p.results)
        self.follow.setVisible(running)
        if p.video.exists() and not p.info().get("video_info"):
            try:
                from replay.video import probe

                i = probe(p.video)
                p.update_info(video_info={"duration_s": i.duration_s, "size": [i.width, i.height], "fps": i.fps})
            except Exception:  # noqa: BLE001
                pass
        if st and st.get("state") == "finished" and p.id not in self.built:
            self.built.add(p.id)
            s = p.status()
            if s["analyzed"] and s["calibrated"] and not s["replay_ready"]:
                from replay.gui.jobs import BUILD

                self.win.jobs.submit(p.id, BUILD)  # the full build: jersey numbers, colours, ... for the whole game
        self.refresh_replay(p)

    def refresh_replay(self, p: Project) -> None:
        """New 2D data: reload it without losing the playhead (or jump to the live edge when following)."""
        tab = self.win.replay_tab
        f = p.results / "replay_data.json"
        try:
            mt = f.stat().st_mtime
        except OSError:
            return
        if self.data_mtime.get(p.id) == mt or tab.p is None or tab.p.id != p.id:
            return
        if self.win.tabs.currentWidget() is not tab:
            return  # reloaded when the tab is shown (tab_changed -> reload_data)
        self.data_mtime[p.id] = mt
        player = tab.player
        pos = player.position()
        playing = player.playbackState() == QMediaPlayer.PlaybackState.PlayingState
        tab.reload_data()
        t_end = float((tab.data or {}).get("t", [0.0])[-1] or 0.0) if (tab.data or {}).get("t") else 0.0
        follow = self.follow.isChecked() and live.is_running(p.results)
        target = max(0, int((t_end - 3.0) * 1000)) if follow else pos
        if p.video.exists() and (player.source().isEmpty() or player.duration() < t_end * 1000 - 1500):
            # the player only knows the video length from when it opened it: reopen, then restore the playhead
            def restore(status, target=target, playing=playing):
                if status in (QMediaPlayer.MediaStatus.LoadedMedia, QMediaPlayer.MediaStatus.BufferedMedia):
                    player.mediaStatusChanged.disconnect(restore)
                    player.setPosition(target)
                    if playing or follow:
                        player.play()

            player.mediaStatusChanged.connect(restore)
            player.setSource(QUrl.fromLocalFile(str(p.video.resolve())))
        elif follow and abs(target - pos) > 1500:
            player.setPosition(target)

    def shutdown(self) -> None:
        """App quits: live processes started here finish what they have (STOP) and exit by themselves."""
        for pid, proc in self.procs.items():
            if proc.poll() is None:
                try:
                    live.request_stop(self.win.store.get(pid).results)
                except (KeyError, OSError):
                    pass


def install(win) -> LiveController:
    win.live = LiveController(win)
    return win.live
