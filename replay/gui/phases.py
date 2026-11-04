"""The phases of a Replay build, for the always-visible progress panel (no Qt here, unit-testable).

A :class:`PhaseTracker` per Replay follows the job manager's state (kind / label / progress / message, polled by
the GUI) and the job processes' log lines (``JobState.lines``), and keeps, per phase: state (waiting / running /
done / skipped / stopped / failed), progress, start / end time, a smoothed ETA and notes (fallbacks, waits,
errors). Durations of finished phases are kept in ``results/phases.json`` so a Replay opened later still shows
them. Nothing here changes the workflow: it only observes it.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

PHASES = [  # (key, title)
    ("import", "Import (convert / copy)"),
    ("prelim", "Finding live play"),
    ("pass1", "Detecting & tracking players"),
    ("calib1", "Calibration during the analysis"),
    ("autocal", "Court calibration"),
    ("jersey", "Jersey numbers"),
    ("teams", "Team colours / referees"),
    ("build", "Building the replay"),
    ("report", "Report / exports"),
]
TITLES = dict(PHASES)
PHASE_JOB = {"import": "import", "prelim": "analyze", "pass1": "analyze", "calib1": "analyze", "autocal": "autocal",
             "jersey": "build", "teams": "build", "build": "build", "report": "build"}
ORDER = [k for k, _t in PHASES]
# seconds of work per second of video when nothing is measured yet (GPU, 720p; measured on 3-min clips)
DEFAULT_RATE = {"import": 0.005, "prelim": 0.03, "pass1": 1.2, "calib1": 0.0, "autocal": 0.2, "jersey": 0.1,
                "teams": 0.05, "build": 0.25, "report": 0.03}
DEFAULT_FIXED = {"prelim": 3.0, "pass1": 10.0, "autocal": 30.0, "jersey": 10.0, "build": 25.0, "report": 5.0}
FILENAME = "phases.json"
STALL_S = 45.0  # a running phase whose progress has not moved for this long says so
WAITING, RUNNING, DONE, SKIPPED, STOPPED, FAILED = "waiting", "running", "done", "skipped", "stopped", "failed"

# build (pass 2) log lines -> (phase that starts / progresses, progress of "build", phases finished by it)
BUILD_LINES = [
    (re.compile(r"^camera type"), 0.05, []),
    (re.compile(r"^filtered tracks"), 0.30, []),
    (re.compile(r"^lineup: \d+ pieces"), 0.45, ["teams"]),
    (re.compile(r"^final limits"), 0.55, []),
    (re.compile(r"^baskets"), 0.65, []),
    (re.compile(r"^jersey numbers"), 0.80, ["jersey"]),
    (re.compile(r"^game state"), 0.95, ["build"]),
]
NOTE_PATTERNS = [  # log lines worth showing on a row: (pattern, phase, level)
    (re.compile(r"fell back|fallback|full search", re.I), None, "warn"),
    (re.compile(r"^Frame taps \(finish wait ([0-9.]+)s\)"), "pass1", "info"),
    (re.compile(r"^Calibration during the analysis: (\d+) round"), "calib1", "info"),
    (re.compile(r"failed|error", re.I), None, "warn"),
]


def fmt_s(s: float | None) -> str:
    if s is None:
        return "–"
    s = max(0.0, float(s))
    if s < 60:
        return f"{s:.0f} s"
    if s < 3600:
        return f"{int(s // 60)}:{int(s % 60):02d}"
    return f"{int(s // 3600)}:{int(s // 60) % 60:02d}:{int(s % 60):02d}"


class PhaseTracker:
    def __init__(self, root: Path | str | None = None, video_s: float | None = None):
        self.root = Path(root) if root else None
        self.video_s = video_s
        self.ph: dict[str, dict] = {k: {"state": WAITING, "progress": 0.0, "t0": None, "t1": None, "eta": None,
                                        "notes": [], "rate": None} for k in ORDER}
        self._last_kind: str | None = None
        self._lines_seen = 0
        self._loaded = False

    # ------------------------------------------------------------------ persistence
    def load(self, status: dict | None = None) -> None:
        """Phases of a Replay built earlier: done (with durations from phases.json) as far as its status says."""
        self._loaded = True
        saved = {}
        if self.root is not None:
            try:
                saved = json.loads((self.root / "results" / FILENAME).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                saved = {}
        s = status or {}
        done_upto = []
        if s.get("imported"):
            done_upto.append("import")
        if s.get("analyzed"):
            done_upto += ["prelim", "pass1", "calib1"]
        if s.get("calibrated"):
            done_upto.append("autocal")
        if s.get("replay_ready") and not s.get("replay_stale"):
            done_upto += ["jersey", "teams", "build", "report"]
        for k in ORDER:
            sv = saved.get(k) if isinstance(saved, dict) else None
            if k in done_upto or (isinstance(sv, dict) and sv.get("state") in (DONE, SKIPPED)):
                p = self.ph[k]
                if p["state"] == WAITING:
                    p.update(state=(sv or {}).get("state") or DONE, progress=1.0)
                    p["dur"] = (sv or {}).get("s")
                    p["notes"] = list((sv or {}).get("notes") or [])

    def _save(self) -> None:
        if self.root is None:
            return
        out = {}
        for k, p in self.ph.items():
            if p["state"] in (DONE, SKIPPED):
                dur = p.get("dur")
                if dur is None and p["t0"] and p["t1"]:
                    dur = round(p["t1"] - p["t0"], 1)
                out[k] = {"state": p["state"], "s": dur, "notes": p["notes"][-3:]}
        try:
            (self.root / "results").mkdir(parents=True, exist_ok=True)
            (self.root / "results" / FILENAME).write_text(json.dumps(out, indent=1), encoding="utf-8")
        except OSError:
            pass

    # ------------------------------------------------------------------ transitions
    def start(self, k: str, now: float | None = None) -> None:
        p = self.ph[k]
        if p["state"] != RUNNING:
            p.update(state=RUNNING, t0=now or time.time(), t1=None, progress=0.0, eta=None, rate=None)
            p.pop("dur", None)

    def finish(self, k: str, state: str = DONE, now: float | None = None) -> None:
        p = self.ph[k]
        if p["state"] in (DONE, SKIPPED) and state in (DONE, SKIPPED):
            return
        if p["state"] == WAITING and state == DONE:
            p["t0"] = p["t0"] or (now or time.time())
        p.update(state=state, t1=now or time.time(), eta=0.0 if state == DONE else None)
        if state in (DONE, SKIPPED):
            p["progress"] = 1.0
        self._save()

    def note(self, k: str, text: str, level: str = "info") -> None:
        notes = self.ph[k]["notes"]
        item = {"text": text[:160], "level": level}
        if item not in notes:
            notes.append(item)
            del notes[:-4]

    def set_progress(self, k: str, x: float, now: float | None = None) -> None:
        p = self.ph[k]
        now = now or time.time()
        if p["state"] != RUNNING:
            self.start(k, now)
        x = max(0.0, min(1.0, float(x)))
        new = max(p["progress"], x) if x < 1 else 1.0
        if new > p["progress"] + 1e-4 or p.get("moved") is None:
            p["moved"] = now
        p["progress"] = new
        el = now - (p["t0"] or now)
        if p["progress"] > 0.02 and el > 1.0:
            rate = p["progress"] / el  # fraction per second, smoothed
            p["rate"] = rate if p["rate"] is None else 0.7 * p["rate"] + 0.3 * rate
            p["eta"] = (1.0 - p["progress"]) / max(p["rate"], 1e-6)

    # ------------------------------------------------------------------ observing the job manager
    def observe(self, kind: str | None, label: str | None, progress: float, message: str, error: str | None,
                lines: list[str] | None = None, now: float | None = None, status: dict | None = None) -> None:
        now = now or time.time()
        if not self._loaded:
            self.load(status)
        for ln in (lines or [])[self._lines_seen:]:
            self.line(ln, now)
        self._lines_seen = len(lines or [])
        prev = self._last_kind
        if kind != prev:
            self._kind_changed(prev, kind, error, message, now)
            self._last_kind = kind
        if kind == "import":
            self.set_progress("import", progress, now)
        elif kind == "analyze":
            for k in ("import",):
                self.finish(k, now=now)
            if label:  # preliminary analysis
                self.set_progress("prelim", progress, now)
            else:
                if self.ph["prelim"]["state"] == RUNNING:
                    self.finish("prelim", now=now)
                self.set_progress("pass1", progress, now)
                if self.ph["calib1"]["state"] == WAITING:
                    self.start("calib1", now)
                self.ph["calib1"]["progress"] = self.ph["pass1"]["progress"]
        elif kind == "autocal":
            self.set_progress("autocal", progress, now)
            if re.search(r"fell back|full search", message or "", re.I):
                self.note("autocal", message, "warn")
        elif kind == "build":
            for k in ("jersey", "teams", "build"):
                if self.ph[k]["state"] == WAITING:
                    self.start(k, now)
            # a build reports little progress itself: the log lines move the bar; time moves it a little too
            p = self.ph["build"]
            if p["state"] == RUNNING and p["t0"]:
                est = self._estimate("build") or 60.0
                p["progress"] = max(p["progress"], min(0.9, (now - p["t0"]) / max(est, 1.0) * 0.9))
                self.set_progress("build", p["progress"], now)
            for k in ("jersey", "teams"):
                if self.ph[k]["state"] == RUNNING:
                    self.ph[k]["progress"] = max(self.ph[k]["progress"], self.ph["build"]["progress"])

    def _kind_changed(self, prev: str | None, kind: str | None, error: str | None, message: str,
                      now: float) -> None:
        running = [k for k in ORDER if self.ph[k]["state"] == RUNNING]
        if kind is None:  # a job ended
            stopped = "stopped" in (message or "").lower()
            for k in running:
                if error:
                    self.finish(k, FAILED, now)
                    self.note(k, error, "error")
                elif stopped:
                    self.finish(k, STOPPED, now)
                elif prev == "analyze" and k in ("prelim", "pass1", "calib1"):
                    self.finish(k, now=now)
                elif prev == "autocal" and k == "autocal":
                    self.finish(k, now=now)
                    if (message or "").startswith("Could not find the court"):
                        self.ph[k]["state"] = FAILED
                        self.note(k, message, "error")
                elif prev == "build":
                    self.finish(k, now=now)
            if prev == "build" and not error and not stopped:
                for k in ("jersey", "teams", "build", "report"):
                    self.finish(k, now=now)
                if "partial" in (message or "") or "no players" in (message or ""):
                    self.note("build", re.sub(r"<[^>]+>", "", message)[:160], "warn")
            return
        # a later job started: every phase of the earlier jobs is over (polling can miss a short job entirely)
        order = ["import", "analyze", "autocal", "build"]
        if kind in order:
            for k in ORDER:
                if order.index(PHASE_JOB[k]) < order.index(kind) and self.ph[k]["state"] in (RUNNING, WAITING):
                    if k == "autocal" and self.ph[k]["state"] == WAITING:
                        continue  # handled below (skipped = calibrated by hand)
                    self.finish(k, now=now)
        if kind == "analyze":
            self.finish("import", now=now)
            for k in ("prelim", "pass1", "calib1", "autocal", "jersey", "teams", "build", "report"):
                if self.ph[k]["state"] in (FAILED, STOPPED, DONE) and k not in ("prelim",):
                    self.ph[k].update(state=WAITING, progress=0.0, t0=None, t1=None, eta=None)
        if kind == "autocal":
            for k in ("prelim", "pass1", "calib1"):
                if self.ph[k]["state"] in (RUNNING, WAITING):
                    self.finish(k, now=now)
        if kind == "build":
            if self.ph["autocal"]["state"] == WAITING:
                self.finish("autocal", SKIPPED, now)  # calibrated by hand
                self.note("autocal", "court calibrated by hand", "info")
            for k in ("jersey", "teams", "build", "report"):
                self.ph[k].update(state=WAITING, progress=0.0, t0=None, t1=None, eta=None)

    def line(self, text: str, now: float | None = None) -> None:
        """A log line of a job process."""
        now = now or time.time()
        t = text.strip()
        for rx, x, finished in BUILD_LINES:
            if rx.search(t):
                self.set_progress("build", x, now)
                for k in finished:
                    if k == "build":
                        self.start("report", now)
                    else:
                        self.finish(k, now=now)
                if t.startswith("jersey numbers"):
                    m = re.match(r"jersey numbers: (\d+) of (\d+) tracks", t)
                    if m:
                        self.note("jersey", f"numbers read on {m.group(1)} of {m.group(2)} tracks")
        for rx, ph, level in NOTE_PATTERNS:
            m = rx.search(t)
            if not m:
                continue
            k = ph or next((k for k in reversed(ORDER) if self.ph[k]["state"] == RUNNING), None)
            if k is None:
                continue
            if ph == "pass1":
                wait = float(m.group(1))
                if wait > 10:
                    self.note("pass1", f"waited {fmt_s(wait)} for frame readers (score bar, colours)", "warn")
            elif ph == "calib1":
                self.note("calib1", f"{m.group(1)} round(s)")
                self.finish("calib1", now=now)
            else:
                self.note(k, t, level)
            break

    # ------------------------------------------------------------------ estimates / snapshot
    def _estimate(self, k: str) -> float | None:
        """Expected total seconds of phase k from defaults x video length (or the measured pass-1 speed)."""
        if not self.video_s:
            return None
        base = DEFAULT_FIXED.get(k, 0.0) + DEFAULT_RATE.get(k, 0.0) * self.video_s
        p1 = self.ph["pass1"]
        if k != "pass1" and p1.get("rate") and p1["state"] == RUNNING:
            # a slow machine is slow everywhere: scale the defaults by the measured pass-1 speed
            measured = 1.0 / p1["rate"] if p1["rate"] else None
            if measured:
                base *= max(0.3, min(5.0, measured / max(DEFAULT_RATE["pass1"] * self.video_s, 1.0)))
        return base

    def snapshot(self, now: float | None = None) -> dict:
        now = now or time.time()
        rows, remaining, any_running = [], 0.0, False
        for k in ORDER:
            p = self.ph[k]
            el = None
            if p["t0"]:
                el = (p["t1"] or now) - p["t0"]
            elif p.get("dur") is not None:
                el = p["dur"]
            eta = p["eta"]
            stalled = None
            if p["state"] == RUNNING and k != "calib1" and p.get("moved") and now - p["moved"] > STALL_S:
                stalled = round(now - p["moved"])
            if k == "calib1" and p["state"] == RUNNING:
                eta = None  # runs inside the detection pass: no time of its own
            elif p["state"] == RUNNING:
                any_running = True
                if eta is None:
                    est = self._estimate(k)
                    eta = None if est is None or el is None else max(0.0, est - el)
                remaining += eta or 0.0
            elif p["state"] == WAITING:
                est = self._estimate(k)
                remaining += est or 0.0
            rows.append({"key": k, "title": TITLES[k], "state": p["state"], "progress": round(p["progress"], 4),
                         "elapsed": None if el is None else round(el, 1), "eta": None if eta is None else round(eta),
                         "notes": list(p["notes"]), "stalled": stalled})
        return {"rows": rows, "overall_eta": round(remaining) if any_running and self.video_s else None,
                "running": any_running}
