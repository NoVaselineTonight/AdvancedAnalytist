"""Background jobs for the GUI: one worker thread, one CPU-heavy job at a time.

Job chain for a new video:  import (proxy) -> analyze (preliminary analysis + pass 1) -> autocal (automatic
court calibration, skipped when the user calibrated by hand) -> build (pass 2)

The automatic calibration and the build run in a separate process (``_child_main``): they are pure-Python heavy and
in a thread they starved the GUI of the GIL (the 2D view stuttered for minutes on a whole game). In a process they can
also be stopped at any time, and a build interrupted by quitting the app is redone on the next start
(``results/JOB_<kind>`` marker files, see :func:`interrupted_jobs`).
"""

from __future__ import annotations

import atexit
import json
import multiprocessing as mp
import queue
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from replay.gui import media
from replay.gui.projects import Project, ProjectStore
from replay import inplay as _inplay
from replay.inplay import intervals_for_pass1
from replay.metadata import metadata_from_dict
from replay.pipeline import Cancelled, run_pass1, run_pass2

IMPORT, ANALYZE, BUILD, AUTOCAL = "import", "analyze", "build", "autocal"
LABELS = {IMPORT: "Preparing video", ANALYZE: "Detecting and tracking players", BUILD: "Building 2D replay",
          AUTOCAL: "Calibrating the court automatically"}
PRELIMINARY_LABEL = "Preliminary analysis"
CHILD_KINDS = (AUTOCAL, BUILD)  # run in their own process
PAUSED = "PAUSED"  # results/PAUSED: the user pressed Stop - the job chain is not continued automatically


def _import_source(p: Project) -> Path | None:
    from replay.gui.projects import original_video
    try:
        return original_video(p)
    except OSError:
        return None


class JobError(Exception):
    """A job failed with a message meant for the user."""


def marker(p: Project, kind: str) -> Path:
    """results/JOB_<kind>: exists while a separate-process job runs (left behind when the app was closed)."""
    return p.results / f"JOB_{kind.rpartition(':')[2]}"


def interrupted_jobs(p: Project) -> list[str]:
    """Separate-process jobs that were running when the app was closed or crashed (to be redone on start)."""
    return [k for k in CHILD_KINDS if marker(p, k).exists()]


# ---------------------------------------------------------------------- child process side
def _child_autocal(p: Project, progress, cancel) -> dict | None:
    from replay import autocourt
    from replay.pipeline import load_pass1

    progress(0.0, "Loading analysis")
    data = load_pass1(p.results / "work")
    res = None
    if p.settings().get("gemini_court"):  # opt-in: court keypoints from Gemini (gemini_court.py, hard cost caps)
        from replay import gemini_court
        res = gemini_court.for_project(p, data, tuple(p.calibration().frame_size), progress, cancel)
        if res is not None and len(res.frames) < gemini_court.GOOD_MIN_FRAMES:
            res = None
    if res is None:
        res = autocourt.calibrate_best(p.video, data, tuple(p.calibration().frame_size), progress=progress,
                                       cancel=cancel, work=p.results / "work",  # work: features from pass 1
                                       detector=p.config().calibration.court_detector)
    if cancel.is_set():
        return {"stopped": True}
    if res is None:
        return None
    return {"length": float(res.length), "width": float(res.width), "spec": res.spec,
            "good_frac": float(res.good_frac), "message": res.message, "auto": res.to_json()}


def _child_build(p: Project, progress, cancel) -> dict:
    progress(0.0, "Projecting positions onto the court")
    rep = run_pass2(p.results, p.config(), metadata_from_dict(p.meta()), p.calibration_path, p.video,
                    swap_teams=bool(p.settings().get("swap_teams", False)))
    return {"valid_pct": rep["calibration"]["valid_pct"],
            "people": rep["people"]["avg_tracked_per_calibrated_frame"]}


def _analyze_body(p: Project, st, restart: bool = False) -> None:
    """Preliminary analysis + pass 1 (resumable). ``st`` has progress/message/label/started/cancel."""
    st.label, st.message = PRELIMINARY_LABEL, "Starting"
    cfg = p.config()
    meta = metadata_from_dict(p.meta())

    info = {"det": ""}

    def prog(done: int, total: int) -> None:
        st.progress = done / max(1, total)
        st.message = f"Frame {done} / {total}{info['det']}"

    def status(msg: str) -> None:
        info["det"] = f" · {msg.replace('Detector: ', '')}"
        st.message = msg

    p.results.mkdir(exist_ok=True)
    # Settings > "Skip breaks": the preliminary analysis finds when the ball is in play (results/inplay.json)
    cfg.inplay.enabled = bool(p.settings().get("skip_breaks", cfg.inplay.enabled))
    (p.results / "config.used.json").write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")

    def pre_prog(x: float) -> None:
        st.progress = x
        st.message = f"Finding when the ball is in play ({x * 100:.0f}%)"

    def pre_log(msg: str) -> None:
        st.message = msg

    try:
        intervals = intervals_for_pass1(p.video, p.results, cfg.inplay, restart=restart, progress=pre_prog,
                                        cancel=st.cancel, log=pre_log)
    except _inplay.Cancelled:
        raise media.Cancelled()
    st.label = None
    st.progress, st.started = 0.0, time.time()  # the ETA restarts for the detection pass
    if intervals is not None:
        dur = float(p.info().get("video_info", {}).get("duration_s") or intervals[-1][1])
        st.message = "In play: " + _inplay.describe(intervals, dur)
    else:
        st.message = "Loading detector"
    run_pass1(p.video, p.results, cfg, meta, restart=restart, progress=prog, cancel=st.cancel, status=status,
              play_intervals=intervals)
    st.progress = 1.0


class _ChildState:
    """In the job process: attribute writes (progress, message, label, started) are sent to the GUI's JobState."""

    def __init__(self, q, cancel):
        object.__setattr__(self, "_q", q)
        object.__setattr__(self, "cancel", cancel)
        object.__setattr__(self, "_last", 0.0)

    def __setattr__(self, k: str, v) -> None:
        object.__setattr__(self, k, v)
        if k == "progress":  # pass 1 reports every frame: at most ~20 updates/s are enough
            now = time.time()
            if now - self._last < 0.05 and 0 < v < 1:
                return
            object.__setattr__(self, "_last", now)
        self._q.put(("set", k, v))


def _child_analyze(p: Project, progress, cancel, restart: bool = False) -> dict:
    _analyze_body(p, progress.state, restart)
    return {}


_CHILD_JOBS = {ANALYZE: _child_analyze, AUTOCAL: _child_autocal, BUILD: _child_build}


def _log_path(p: Project, kind: str) -> Path:
    return p.results / f"{kind.rpartition(':')[2]}.log"


class _Tee:
    """Job-process stdout: to the log file, and every complete line also to the GUI (phase panel notes / stages).
    Progress bars that redraw with carriage returns only reach the file."""

    def __init__(self, fh, q):
        self.fh, self.q, self.buf = fh, q, ""

    def write(self, text: str) -> int:
        self.fh.write(text)
        self.buf += text
        while "\n" in self.buf:
            line, self.buf = self.buf.split("\n", 1)
            line = line.rsplit("\r", 1)[-1].strip()
            if line:
                try:
                    self.q.put(("line", line[:300]))
                except Exception:  # noqa: BLE001 - the GUI may be gone
                    pass
        if len(self.buf) > 4000:
            self.buf = self.buf[-1000:]
        return len(text)

    def flush(self) -> None:
        self.fh.flush()

    def isatty(self) -> bool:
        return False

    def fileno(self) -> int:
        return self.fh.fileno()


def _child_main(kind: str, root: str, q, cancel, kwargs: dict | None = None) -> None:
    """Entry point of the job process. Output goes to results/<kind>.log (the GUI may run without a console)."""
    import sys

    p = Project(Path(root))
    try:
        sys.stdout = sys.stderr = _Tee(open(_log_path(p, kind), "w", encoding="utf-8", buffering=1), q)
    except OSError:
        pass
    try:
        fn = _CHILD_JOBS.get(kind)
        if fn is None:  # "package.module:function" (tests)
            import importlib
            mod, _, name = kind.partition(":")
            fn = getattr(importlib.import_module(mod), name)
        def progress(x: float, m: str = "") -> None:
            q.put(("progress", float(x), str(m)))

        progress.state = _ChildState(q, cancel)
        q.put(("done", fn(p, progress, cancel, **(kwargs or {}))))
    except SystemExit as e:
        q.put(("error", str(e)))
    except BaseException as e:  # noqa: BLE001 - reported to the GUI
        traceback.print_exc()
        try:  # stability hook: the crash log keeps the full report (replay/gui/crashguard.py)
            from replay.gui import crashguard
            crashguard.record(f"job {kind}", e)
        except Exception:  # noqa: BLE001 - never hide the job's own error
            pass
        q.put(("error", f"{type(e).__name__}: {e}"))
    finally:
        try:
            sys.stdout.flush()
        except Exception:  # noqa: BLE001
            pass


@dataclass
class JobState:
    kind: str | None = None
    progress: float = 0.0
    message: str = ""
    error: str | None = None
    queued: list[str] = field(default_factory=list)
    started: float | None = None
    cancel: threading.Event = field(default_factory=threading.Event)
    label: str | None = None  # overrides LABELS[kind] for a sub-step (e.g. the preliminary analysis)
    lines: list = field(default_factory=list)  # log lines of the job processes (phase panel)
    tracker: Any = None  # replay.gui.phases.PhaseTracker of this Replay (created by JobManager.phases)

    def public(self) -> dict[str, Any]:
        eta = None
        if self.kind and self.started and 0.02 < self.progress < 1:
            el = time.time() - self.started
            eta = el / self.progress - el
        return {"kind": self.kind, "label": self.label or LABELS.get(self.kind or "", ""),
                "progress": round(self.progress, 4),
                "message": self.message, "error": self.error, "queued": list(self.queued),
                "eta_s": round(eta) if eta else None}


class JobManager:
    def __init__(self, store: ProjectStore):
        self.store = store
        self.q: queue.Queue[tuple[str, str, dict]] = queue.Queue()
        self.states: dict[str, JobState] = {}
        self.lock = threading.Lock()
        self.procs: set = set()  # running job processes
        self.quitting = False  # set by the app before it cancels jobs on quit (that is not a user "Stop")
        threading.Thread(target=self._worker, daemon=True, name="replay-worker").start()
        atexit.register(self.shutdown)  # runs before multiprocessing's exit handler, which would wait for them

    def shutdown(self) -> None:
        """Stop job processes (the app quits). A build stopped like this is redone on the next start."""
        with self.lock:
            procs = list(self.procs)
        for proc in procs:
            if proc.is_alive():
                proc.terminate()
        for proc in procs:
            proc.join(3)

    def phases(self, p: Project) -> dict:
        """The phase panel of Replay p (replay/gui/phases.py): observes this job state; cheap, polled by the GUI."""
        from replay.gui.phases import PhaseTracker
        st = self.state(p.id)
        if st.tracker is None:
            vi = p.info().get("video_info") or {}
            st.tracker = PhaseTracker(p.root, float(vi.get("duration_s") or 0.0) or None)
        elif not st.tracker.video_s:
            st.tracker.video_s = float((p.info().get("video_info") or {}).get("duration_s") or 0.0) or None
        tr = st.tracker
        status = p.status() if not tr._loaded else None
        tr.observe(st.kind, st.label, st.progress, st.message, st.error, st.lines, status=status)
        return tr.snapshot()

    def state(self, pid: str) -> JobState:
        with self.lock:
            return self.states.setdefault(pid, JobState())

    def submit(self, pid: str, kind: str, **kwargs: Any) -> None:
        st = self.state(pid)
        with self.lock:
            # the same job already running is enough - unless it is being cancelled (e.g. Overview > Save with
            # new analysis settings cancels the running analysis and immediately asks for a restart)
            if kind in st.queued or st.kind == kind and kind != BUILD and not st.cancel.is_set():
                return
            st.queued.append(kind)
            st.error = None
        self.q.put((pid, kind, kwargs))

    def cancel(self, pid: str) -> None:
        st = self.state(pid)
        with self.lock:
            st.queued.clear()
        st.cancel.set()

    # ------------------------------------------------------------------ worker
    def _worker(self) -> None:
        while True:
            pid, kind, kwargs = self.q.get()
            st = self.state(pid)
            with self.lock:
                if kind not in st.queued:  # cancelled while queued
                    continue
                st.queued.remove(kind)
                st.kind, st.progress, st.message, st.started, st.label = kind, 0.0, "", time.time(), None
                st.cancel.clear()
            project = None
            try:
                project = self.store.get(pid)
                (project.results / PAUSED).unlink(missing_ok=True)
                getattr(self, f"_run_{kind}")(project, st, **kwargs)
            except (Cancelled, media.Cancelled):
                st.message = ("Import stopped. Press Retry import to start it again." if kind == IMPORT
                              else "Stopped. Progress is kept; press Continue analysis to go on.")
            except KeyError:
                pass  # project deleted meanwhile
            except (SystemExit, JobError) as e:  # e.g. the pipeline refuses to mix settings: report, don't exit
                st.error = str(e)
            except Exception as e:  # surfaced in the GUI
                traceback.print_exc()
                st.error = f"{type(e).__name__}: {e}"
            finally:
                if st.cancel.is_set() and not self.quitting and project is not None:
                    try:  # the user pressed Stop: remember it, so the next start doesn't continue by itself
                        project.results.mkdir(exist_ok=True)
                        (project.results / PAUSED).write_text(kind, encoding="utf-8")
                    except OSError:
                        pass
                with self.lock:
                    st.kind, st.started, st.label = None, None, None

    # ------------------------------------------------------------------ resume after a restart / crash
    def resume_interrupted(self) -> list[tuple[str, str]]:
        """Continue every job chain that was cut off by closing the app (or a crash / power cut), unless the user
        had pressed Stop. Returns [(replay id, job kind)] submitted."""
        out = []
        for p in self.store.list():
            try:
                s = p.status()
                if s["paused"]:
                    continue
                from replay.live import owns_replay  # live hook: live analysis continues these itself
                if owns_replay(p):
                    continue
                kind, kw = None, {}
                if not s["imported"]:
                    src = _import_source(p)
                    if src is not None and not p.info().get("imported_from"):
                        kind, kw = IMPORT, {"source": str(src)}
                elif s["analysis_started"] and not s["analyzed"]:
                    kind = ANALYZE
                elif s["analyzed"] and marker(p, AUTOCAL).exists():
                    kind = AUTOCAL
                elif s["analyzed"] and s["calibrated"] and (s["build_interrupted"] or not s["replay_ready"]):
                    kind = BUILD
            except Exception:  # noqa: BLE001 - a damaged Replay must not stop the app from starting
                traceback.print_exc()
                continue
            if kind:
                self.submit(p.id, kind, **kw)
                self.state(p.id).message = "Continuing where it stopped when the app was closed"
                out.append((p.id, kind))
        return out

    def _run_import(self, p: Project, st: JobState, source: str, copy_original: bool = False) -> None:
        if copy_original:
            st.message = "Copying the original video into the Replay"
            dst = p.root / "source" / Path(source).name
            dst.parent.mkdir(exist_ok=True)
            total, done = max(1, Path(source).stat().st_size), 0
            with open(source, "rb") as fi, open(dst, "wb") as fo:
                while chunk := fi.read(16 * 1024 * 1024):
                    if st.cancel.is_set():
                        fo.close()
                        dst.unlink(missing_ok=True)
                        raise media.Cancelled()
                    fo.write(chunk)
                    done += len(chunk)
                    st.progress = 0.3 * done / total
            p.update_info(source={**p.info().get("source", {}), "kind": "copy", "path": f"source/{dst.name}"})
            source = str(dst)
        st.message = "Reading video"

        base = st.progress

        def prog(x: float) -> None:
            st.progress = base + (1 - base) * x
            st.message = f"Converting to a playable copy ({x * 100:.0f}%)"

        info = media.make_proxy(source, p.video, prog, st.cancel)
        p.update_info(video_info={"duration_s": info.duration_s, "size": [info.width, info.height],
                                  "fps": info.fps, "codec": info.codec, "has_audio": info.has_audio})
        st.progress = 1.0
        self.submit(p.id, ANALYZE)

    def _run_analyze(self, p: Project, st: JobState, restart: bool = False) -> None:
        from replay.live import owns_replay  # live hook
        if owns_replay(p):  # live hook
            raise JobError("This Replay is analysed live (Live menu); stop the live analysis first.")  # live hook
        st.label, st.message = PRELIMINARY_LABEL, "Starting"
        self._in_process(ANALYZE, p, st, restart=restart)
        st.progress = 1.0
        cal = p.calibration()
        if cal.keyframes and not cal.auto:
            self.submit(p.id, BUILD)  # the user calibrated by hand: keep it
        else:
            self.submit(p.id, AUTOCAL)

    def _in_process(self, kind: str, p: Project, st: JobState, **kwargs: Any) -> Any:
        """Run _CHILD_JOBS[kind] in a separate process; progress goes to ``st``. Stop = ask nicely, then kill."""
        ctx = mp.get_context("spawn")
        q, cancel = ctx.Queue(), ctx.Event()
        proc = ctx.Process(target=_child_main, args=(kind, str(p.root), q, cancel, kwargs),
                           name=f"replay-{kind.rpartition(':')[2]}")
        p.results.mkdir(exist_ok=True)
        marker(p, kind).write_text(str(time.time()), encoding="utf-8")
        proc.start()
        with self.lock:
            self.procs.add(proc)
        kill_at = None
        try:
            while True:
                if st.cancel.is_set() and kill_at is None:
                    cancel.set()
                    # pass 1 stops at the next frame and keeps its checkpoint; the build has no cancel points
                    kill_at = time.time() + {ANALYZE: 60.0, AUTOCAL: 5.0}.get(kind, 0.5)
                if kill_at is not None and time.time() > kill_at and proc.is_alive():
                    proc.terminate()
                try:
                    msg = q.get(timeout=0.2)
                except queue.Empty:
                    if proc.is_alive():
                        continue
                    try:
                        msg = q.get(timeout=1.0)  # anything sent just before it ended
                    except queue.Empty:
                        if st.cancel.is_set():
                            raise Cancelled()
                        raise JobError(f"{LABELS.get(kind, 'The job')} stopped unexpectedly (exit code {proc.exitcode}). "
                                       f"Details: {_log_path(p, kind)}")
                if msg[0] == "progress":
                    st.progress, st.message = msg[1], msg[2] or st.message
                elif msg[0] == "line":
                    st.lines.append(msg[1])
                    if len(st.lines) > 2000:
                        del st.lines[:1000]
                        if st.tracker is not None:
                            st.tracker._lines_seen = max(0, st.tracker._lines_seen - 1000)
                elif msg[0] == "set":
                    if msg[1] in ("progress", "message", "label", "started"):
                        setattr(st, msg[1], msg[2])
                elif msg[0] == "done":
                    return msg[1]
                elif msg[0] == "error":
                    if st.cancel.is_set():
                        raise Cancelled()
                    raise JobError(f"{LABELS.get(kind, 'The job')} failed: {msg[1]}")
        finally:
            if proc.is_alive():
                proc.join(5)
            if proc.is_alive():
                proc.terminate()
                proc.join(3)
            with self.lock:
                self.procs.discard(proc)
            # a stopped build may have left half-written outputs: keep its marker (rebuilt on the next start)
            if not (kind == BUILD and st.cancel.is_set()):
                marker(p, kind).unlink(missing_ok=True)

    def _run_autocal(self, p: Project, st: JobState) -> None:
        """Automatic court calibration (replay/autocourt.py) -> calibration.json, then build the replay."""
        st.message = "Loading analysis"
        try:
            res = self._in_process(AUTOCAL, p, st)
        except Cancelled:
            st.message = "Automatic calibration stopped."
            return
        if res and res.get("stopped"):
            st.message = "Automatic calibration stopped."
            return
        if res is None or res["good_frac"] < 0.05:
            st.message = ("Could not find the court automatically. Fix it by hand in the 'Court fix (advanced)' tab "
                          "(click a few court points).")
            return
        cal = p.calibration()  # re-read: the user may have edited keyframes meanwhile
        cal.court = {"length": round(res["length"], 3), "width": round(res["width"], 3), "spec": res["spec"]}
        cal.auto = res["auto"]
        cal.save(p.calibration_path)
        st.progress = 1.0
        st.message = "Court found: " + res["message"]
        self.submit(p.id, BUILD)

    def _run_build(self, p: Project, st: JobState) -> None:
        st.message = "Projecting positions onto the court"
        try:
            rep = self._in_process(BUILD, p, st)
        except Cancelled:
            st.message = "Build stopped. The replay is marked outdated; press Build replay to build it again."
            return
        st.progress = 1.0
        if not rep.get("valid_pct") or not rep.get("people"):  # never a silently empty 2D view
            st.message = ("<b style='color:#b45309'>Replay built, but no players could be placed on the court</b> "
                          f"(court calibration valid on {rep.get('valid_pct', 0)}% of frames, "
                          f"{rep.get('people', 0)} people per frame). Check the court in 'Court fix (advanced)'.")
        elif rep["valid_pct"] < 70 or rep["people"] < 5:  # usable, but say plainly that it is partial
            st.message = (f"<b style='color:#b45309'>Replay ready, but partial:</b> the court was found on only "
                          f"{rep['valid_pct']}% of frames and {rep['people']} people per frame are shown (10 are "
                          "on court). Gaps show as an empty court; fixing the court by hand in 'Court fix "
                          "(advanced)' at those moments helps.")
        else:
            st.message = (f"Replay ready: calibration valid on {rep['valid_pct']}% of frames, "
                          f"{rep['people']} people per frame")
