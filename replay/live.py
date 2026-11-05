"""Live mode: a near-live 2D replay while the game is still being recorded.

Input: a local video file that keeps growing (OBS / a capture card writing .ts, .mkv or fragmented .mp4) or a
local HLS folder (``.m3u8`` + segments, assembled here into one growing ``live/stream.ts``). Nothing is fetched
from the internet: HLS entries that are URLs are refused.

How it works (one process, ``python -m replay live --video <growing file> --out <dir>``)

  poll the file  ->  preliminary analysis of the new part (ball in play?)  ->  pass 1 on the new part only
  (detect + track, the same chunk files as ``process``)  ->  court calibration
  ->  pass 2 over a trailing window in a worker process  ->  merged ``replay_data.json`` (atomic)

* Pass 1 keeps its tracker/motion state between steps and appends chunk_NNNNN.npz files to ``<out>/work``;
  the last ``holdback_s`` of decoded video is never committed (the last frame of a growing file may be half
  written), it is decoded again on the next step.
* Calibration: the camera position and the court are found once (the full automatic calibration on the first
  ``calib_min_s`` of play, in a separate process, or reused with ``--reuse-court`` / an existing calibration
  file); after that only the per-frame camera pose is added: anchor frames every ``anchor_every_s`` (vanishing
  points + line pixels in a small process pool) are aligned with autocourt.track_chunk and appended to the
  calibration's anchors. If new frames stop aligning (camera moved) the full calibration is re-run on the last
  ``recalib_window_s``.
* Pass 2 runs on the last ``window_s`` of samples (hard links of the chunk files in ``live/window/work``).
  The first ``freeze_margin_s`` of each window is context only: everything before it is frozen as shown
  before, the rest is replaced; tracks crossing the seam keep their id when they line up. In the default fast
  mode the optional steps that decode the video again (jersey numbers, colour resampling, second-look
  detector, re-ID) are skipped; the final build when the recording ends runs them.
* When the recording ends (no growth for ``idle_end_s``, ``#EXT-X-ENDLIST``, or ``live/STOP``) everything
  left is committed, ``work/run.json`` gets exactly the signature ``process`` would write and ``work/DONE``
  is created: the folder is then an ordinary analysed Replay (``postprocess`` / the GUI build work on it).

Files in ``<out>/live/``: LIVE (exists while live analysis owns the folder), STOP (ask it to finish),
FINISHED, status.json (heartbeat + progress, read by the GUI), updates.jsonl (one line per 2D update: the
delay log), live.log / pass2.log, window/ (pass-2 working copy), stream.ts (assembled HLS).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import pickle
import queue
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np

LIVE_DIR = "live"
MARKER = "LIVE"
STOP_FILE = "STOP"
FINISHED = "FINISHED"
STATUS = "status.json"
UPDATES = "updates.jsonl"
FAST_SUFFIX = "#live-fast"  # appended to the window's run.json "video": pass 2 then finds no video to re-decode


@dataclass
class LiveConfig:
    poll_s: float = 2.0  # how often the file is checked
    min_new_s: float = 10.0  # analyse the new part at most this often (wall seconds) while the file grows
    max_step_s: float = 60.0  # at most this much video per pass-1 step (catching up after a pause)
    holdback_s: float = 1.0  # the newest video is decoded again next time (a half-written last frame)
    idle_end_s: float = 60.0  # no growth for this long = the recording has ended
    auto_calibrate: bool = True  # search the court automatically when there is no calibration
    calib_min_s: float = 60.0  # analysed video needed before the court is searched automatically
    calib_retry_s: float = 60.0  # ... and how much more before trying again when it was not found
    anchor_every_s: float = 2.0  # one calibration anchor frame per this many seconds
    anchor_workers: int = 2
    max_anchors_per_update: int = 40  # keeps the loop responsive after a long calibration
    window_s: float = 120.0  # pass 2 runs on this much of the newest analysed video
    freeze_margin_s: float = 30.0  # first part of each window = context only (the 2D shown there is kept)
    fast: bool = True  # live pass 2 skips the steps that decode the video again (final build runs them)
    final_build: bool = True  # full pass 2 when the recording ends
    skip_breaks: bool | None = None  # None = follow the config's 'inplay' (preliminary analysis) setting
    recalib_window_s: float = 180.0
    recalib_after_s: float = 60.0  # recalibrate when anchors of this long mostly fail to align ...
    recalib_min_good: float = 0.3  # ... i.e. fewer than this share is good
    on_restart: str = "new"  # the file was replaced (new recording): "new" = continue in <out>-restartN, "stop"
    player_link: str | None = None  # hard link to the video for a player (GUI: <Replay>/video.mp4)

    @classmethod
    def from_args(cls, a: argparse.Namespace) -> "LiveConfig":
        lc = cls()
        for name in ("poll_s", "window_s", "idle_end_s", "calib_min_s", "anchor_every_s", "min_new_s"):
            v = getattr(a, name, None)
            if v is not None:
                setattr(lc, name, float(v))
        if getattr(a, "skip_breaks", None) is not None:
            lc.skip_breaks = a.skip_breaks
        if getattr(a, "no_final_build", False):
            lc.final_build = False
        if getattr(a, "full_quality", False):
            lc.fast = False
        if getattr(a, "on_restart", None):
            lc.on_restart = a.on_restart
        lc.player_link = getattr(a, "player_link", None)
        return lc


# ============================================================================== small helpers
def write_atomic(path: Path, text: str) -> None:
    """Temp file + os.replace. On Windows the replace fails while a reader has the file open: retried."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    for attempt in range(50):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.05 * (1 + attempt % 5))
    tmp.unlink(missing_ok=True)
    raise PermissionError(f"could not replace {path} (kept open by another program)")


def live_dir(out: Path) -> Path:
    return Path(out) / LIVE_DIR


def is_live(out: Path) -> bool:
    """True while a live session owns ``out`` (results folder): the normal analysis must not run on it."""
    return (live_dir(out) / MARKER).exists()


def owns_replay(project) -> bool:
    """GUI Replay (gui.projects.Project) that live analysis owns or will continue: the normal job chain
    (import / analyse) must leave it alone. Once the live analysis has finished it is an ordinary Replay."""
    try:
        if is_live(project.results):
            return True
        lv = project.info().get("live")
        return isinstance(lv, dict) and bool(lv.get("source")) and not (project.results / "work" / "DONE").exists()
    except OSError:
        return False


def read_status(out: Path) -> dict | None:
    try:
        return json.loads((live_dir(out) / STATUS).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def is_running(out: Path, stale_s: float = 30.0) -> bool:
    """A live process is alive for this folder (its heartbeat is recent)."""
    st = read_status(out)
    return bool(st and is_live(out) and st.get("state") not in ("finished", "error")
                and time.time() - float(st.get("heartbeat", 0)) < stale_s)


def request_stop(out: Path) -> None:
    d = live_dir(out)
    d.mkdir(parents=True, exist_ok=True)
    (d / STOP_FILE).write_text(str(time.time()), encoding="utf-8")


# ============================================================================== sources
class GrowingFile:
    """A file somebody is still writing. poll() -> 'missing' | 'grew' | 'same' | 'restarted'."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.video = self.path
        self.size = -1
        self.ident: tuple | None = None
        self.last_growth = time.time()
        self.ended = False  # a plain file never says so: idle_end_s decides

    def poll(self) -> str:
        try:
            st = self.path.stat()
        except OSError:
            return "missing"
        ident = (st.st_ino, getattr(st, "st_birthtime", None) or st.st_ctime)
        if self.ident is not None and (st.st_size < self.size or (st.st_ino and ident[0] != self.ident[0])):
            self.ident, self.size = ident, st.st_size
            self.last_growth = time.time()
            return "restarted"
        self.ident = ident
        if st.st_size != self.size:
            self.size = st.st_size
            self.last_growth = time.time()
            return "grew"
        return "same"

    @property
    def idle_s(self) -> float:
        return time.time() - self.last_growth


class HlsFolder:
    """A local HLS recording (playlist + segment files). Complete segments listed in the playlist are appended
    to one growing file (``assembled``); #EXT-X-ENDLIST ends the recording. URLs are refused (no downloads)."""

    def __init__(self, playlist: Path, assembled: Path):
        playlist = Path(playlist)
        if playlist.is_dir():
            found = sorted(playlist.glob("*.m3u8"))
            if not found:
                raise FileNotFoundError(f"No .m3u8 playlist in {playlist}")
            playlist = found[0]
        self.playlist = playlist
        self.video = Path(assembled)
        self.state_path = self.video.with_suffix(".segments.json")
        self.done: list[str] = []
        if self.state_path.exists() and self.video.exists():
            try:
                self.done = json.loads(self.state_path.read_text(encoding="utf-8"))["segments"]
            except (OSError, ValueError, KeyError):
                self.done = []
        if not self.done and self.video.exists():
            self.video.unlink()
        self.ended = False
        self.last_growth = time.time()
        self.size = self.video.stat().st_size if self.video.exists() else -1

    @staticmethod
    def parse(text: str) -> tuple[list[str], str | None, bool]:
        """(segment URIs in order, init-section URI or None, ended)."""
        segs, init, ended = [], None, False
        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#EXT-X-ENDLIST"):
                ended = True
            elif line.startswith("#EXT-X-MAP"):
                for part in line.split(":", 1)[1].split(","):
                    if part.strip().startswith("URI="):
                        init = part.split("=", 1)[1].strip().strip('"')
            elif not line.startswith("#"):
                segs.append(line)
        return segs, init, ended

    def _local(self, uri: str) -> Path:
        if "://" in uri or uri.startswith("//"):
            raise ValueError(f"HLS segment '{uri}' is a URL; live mode only reads local files (no downloads)")
        return (self.playlist.parent / uri).resolve()

    def poll(self) -> str:
        try:
            text = self.playlist.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return "missing"
        segs, init, ended = self.parse(text)
        new = [s for s in segs if s not in set(self.done)]
        grew = False
        self.video.parent.mkdir(parents=True, exist_ok=True)
        with open(self.video, "ab") as out:
            if init and not self.done and out.tell() == 0:
                out.write(self._local(init).read_bytes())
            for s in new:
                p = self._local(s)
                try:
                    a = p.stat().st_size
                    time.sleep(0.05)
                    if a == 0 or p.stat().st_size != a:  # still being written: next poll
                        break
                    out.write(p.read_bytes())
                except OSError:
                    break
                self.done.append(s)
                grew = True
        if grew:
            write_atomic(self.state_path, json.dumps({"segments": self.done}))
            self.last_growth = time.time()
            self.size = self.video.stat().st_size
        self.ended = ended and not [s for s in segs if s not in set(self.done)]
        if not self.done:
            return "missing"
        return "grew" if grew else "same"

    @property
    def idle_s(self) -> float:
        return time.time() - self.last_growth


def make_source(video: Path, out: Path):
    video = Path(video)
    if video.suffix.lower() == ".m3u8" or video.is_dir():
        return HlsFolder(video, live_dir(out) / "stream.ts")
    return GrowingFile(video)


# ============================================================================== preliminary analysis
class LiveInplay:
    """Incremental preliminary analysis (replay/inplay.py): ball in play or break, from a fast low-resolution
    decode of only the new part. Undecided samples near the live edge count as play."""

    def __init__(self, cfg, out: Path, log: Callable[[str], None]):
        self.cfg = cfg
        self.out = Path(out)
        self.log = log
        self.ex = None
        self.size: tuple[int, int] | None = None
        self.intervals: list[tuple[float, float]] = []
        self.end_s = 0.0

    def update(self, video: Path, final: bool = False, holdback_s: float = 1.0, max_s: float = 600.0) -> None:
        import imageio_ffmpeg

        from replay import inplay

        if self.ex is None:
            from replay.video import probe

            info = probe(video)
            self.size = inplay.analysis_size(info.width, info.height, self.cfg.analysis_width)
            self.ex = inplay.SignalExtractor(self.size, self.cfg.sample_fps, 0.0)
        W, H = self.size
        fps = self.cfg.sample_fps
        start = len(self.ex.pan) / fps
        cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-nostdin"]
        if start > 0:
            cmd += ["-ss", f"{start:.3f}"]
        cmd += ["-i", str(video), "-t", f"{max_s:.1f}", "-an", "-sn", "-dn", "-vf",
                f"fps={fps},scale={W}:{H}:flags=area,format=gray", "-f", "rawvideo", "-pix_fmt", "gray", "-"]
        frames = []
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        try:
            while True:
                buf = proc.stdout.read(W * H)
                if len(buf) < W * H:
                    break
                frames.append(np.frombuffer(buf, np.uint8).reshape(H, W))
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.wait()
        keep = len(frames) if final else max(0, len(frames) - int(np.ceil(holdback_s * fps)) - 1)
        for f in frames[:keep]:
            self.ex.add(f)
        if keep:
            self._classify()

    def _classify(self) -> None:
        from replay import inplay

        sig = self.ex.signals()
        if not len(sig.t):
            return
        self.end_s = float(sig.t[-1]) + 1.0 / sig.fps
        res = inplay.analyse_signals(sig, self.cfg, 0.0, self.end_s)
        self.intervals = list(res.intervals)
        try:
            inplay.save_result(res, self.out / inplay.FILENAME, None, self.cfg)
        except OSError:
            pass

    def is_play(self, t: float) -> bool:
        """Play, or not decided yet (near the live edge / beyond the analysed part)."""
        edge = self.end_s - (self.cfg.pad_s + 5.0)
        if t >= edge:
            return True
        return any(a <= t < b for a, b in self.intervals)


# ============================================================================== pass 1
class LivePass1:
    """Incremental pass 1: the same chunk files, state and sample grid as pipeline.run_pass1."""

    def __init__(self, out: Path, video: Path, cfg, meta, log: Callable[[str], None]):
        from replay.video import probe

        self.out, self.work = Path(out), Path(out) / "work"
        self.cfg, self.meta, self.log = cfg, meta, log
        self.work.mkdir(parents=True, exist_ok=True)
        info = probe(video)
        if not (info.fps > 0 and info.width > 0 and info.height > 0):
            raise ValueError("not enough video yet to read its format")
        self.size = info.size
        self.fps = cfg.effective_fps(info.fps)
        self.detector = None
        self.get_detector: Callable[[], Any] | None = None
        self.stop = threading.Event()
        self.capped = False  # the last step stopped at its size limit: more is waiting
        self.gap = False  # the next analysed sample follows skipped (break) samples
        self.n_pos = 0  # analysed samples so far (= positions in load_pass1 arrays)
        self.t_last = -1.0  # time of the last committed sample
        self.motions: list[np.ndarray | None] = []
        self.chunk_t: dict[int, tuple[int, float, float]] = {}
        self._resume_or_start()

    # ---------------------------------------------------------------- state
    def _resume_or_start(self) -> None:
        from replay.camera import MotionEstimator
        from replay.pipeline import _complete_chunks, _latest_state
        from replay.tracking import CMCByteTrack

        last = _latest_state(self.work)
        if last is not None:
            for c in _complete_chunks(self.work):
                if c > last:  # chunk without its state: redone
                    (self.work / f"chunk_{c:05d}.npz").unlink()
            with open(self.work / f"state_{last:05d}.pkl", "rb") as fh:
                st = pickle.load(fh)
            self.tracker, self.motion, self.next_k = st["tracker"], st["motion"], int(st["next_k"])
            self.chunk_id = last + 1
            self.gap = bool(st.get("live", {}).get("gap", False))
            for c in _complete_chunks(self.work):
                with np.load(self.work / f"chunk_{c:05d}.npz") as z:
                    t, mo = z["t"], z["motion"]
                self.chunk_t[c] = (len(t), float(t[0]) if len(t) else 0.0, float(t[-1]) if len(t) else 0.0)
                self.n_pos += len(t)
                self.motions += [None if not np.isfinite(m).all() else m for m in mo]
                if len(t):
                    self.t_last = float(t[-1])
            self.log(f"Live: resuming after chunk {last} ({self.n_pos} samples, t={self.t_last:.1f}s)")
        else:
            from replay.pipeline import _complete_chunks as cc

            for c in cc(self.work):
                (self.work / f"chunk_{c:05d}.npz").unlink()
            self.tracker = CMCByteTrack(self.cfg.tracking, self.fps)
            self.motion = (MotionEstimator(self.cfg.motion, self.size, self.meta.overlay_regions)
                           if self.cfg.motion.enabled else None)
            self.next_k, self.chunk_id = 0, 0

    def write_signature(self, video: Path, live: bool, play_intervals=None) -> dict:
        from replay.pipeline import _run_signature

        sig = _run_signature(Path(video), self.cfg, 0.0, None, self.fps)
        if play_intervals is not None:
            sig["play_intervals"] = [[round(float(a), 2), round(float(b), 2)] for a, b in play_intervals]
        if live:
            sig["live"] = True
        sig = json.loads(json.dumps(sig))
        write_atomic(self.work / "run.json", json.dumps(sig, indent=2))
        return sig

    # ---------------------------------------------------------------- one step
    def _samples(self, video: Path, max_samples: int, final: bool, holdback_s: float,
                 is_play: Callable[[float], bool] | None) -> Iterator[tuple[Any, bool]]:
        """Committable samples (all but the newest ``holdback_s``), break samples skipped (gap flag)."""
        from replay.video import VideoSampler

        sampler = VideoSampler(video, self.fps, 0.0, 1e12)
        hold: deque = deque()
        n = 0
        gen = sampler.iter_samples(self.next_k)

        def emit(smp):
            if is_play is not None and not is_play(smp.t):
                self.skip_next_k = smp.k + 1
                self.gap = True
                return None
            g, self.gap = self.gap, False
            return smp, g

        try:
            for smp in gen:
                hold.append(smp)
                while hold and hold[-1].t - hold[0].t >= holdback_s:
                    item = emit(hold.popleft())
                    n += 1
                    if item is not None:
                        yield item
                    if n >= max_samples or self.stop.is_set():
                        self.capped = n >= max_samples
                        return
            if final:
                while hold:
                    item = emit(hold.popleft())
                    if item is not None:
                        yield item
        finally:
            gen.close()

    def step(self, video: Path, max_s: float, holdback_s: float, final: bool = False,
             is_play: Callable[[float], bool] | None = None,
             on_sample: Callable[[int, Any, np.ndarray, bool], None] | None = None) -> int:
        """Analyse what is new in ``video``; returns the number of committed (analysed) samples."""
        from replay.detection import BALL, PERSON, build_detector
        from replay.features import jersey_features
        from replay.pipeline import _ChunkBuffer, _Stage, _tracker_break

        if self.detector is None and self.get_detector is not None:
            self.detector = self.get_detector()  # loaded in the background while waiting for the recording
        if self.detector is None:
            self.log("Loading detector ...")
            self.detector = build_detector(self.cfg.detector)
        if not getattr(self, "_det_logged", False):
            self._det_logged = True
            self.log(f"Detector: {self.detector.name}; analysing {self.fps:.2f} frames/s of video")
        det = self.detector
        if hasattr(det, "prepare") and hasattr(det, "detect_prepared"):
            prepare, infer = det.prepare, det.detect_prepared
        else:
            prepare, infer = (lambda img: None), (lambda _x, img: det.detect(img))
        self.skip_next_k = -1
        self.capped = False
        max_samples = max(1, int(min(max_s, 1e7) * self.fps))
        reader = _Stage(lambda: self._samples(video, max_samples, final, holdback_s, is_play), 8)
        prep = _Stage(lambda: ((s, g, prepare(s.image)) for s, g in reader), 4)
        detect = _Stage(lambda: ((s, g, infer(x, s.image)) for s, g, x in prep), 16)
        buf = _ChunkBuffer()
        done = 0
        try:
            for s, gap, dets in detect:
                persons = dets.subset(dets.cls == PERSON)
                balls = dets.subset(dets.cls == BALL)
                if gap:  # jumped over a skipped break: like run_pass1
                    if self.motion is not None:
                        self.motion.prev_gray = self.motion.prev_mask = None
                    _tracker_break(self.tracker)
                if self.motion is not None:
                    M, inliers = self.motion.update(s.image, persons.xyxy)
                    M_approx = self.motion.fallback(M)
                else:
                    M, inliers, M_approx = np.eye(3), -1, None
                if M is not None:
                    self.tracker.apply_camera_motion(M)
                elif M_approx is not None:
                    self.tracker.apply_camera_motion(M_approx, approx=True)
                ids = self.tracker.update(persons)
                feats = jersey_features(s.image, persons.xyxy)
                buf.add(s.k, s.frame, s.t, M, inliers, persons, ids, feats, balls)
                pos = self.n_pos
                self.n_pos += 1
                self.motions.append(M)
                self.next_k = s.k + 1
                self.t_last = float(s.t)
                done += 1
                if on_sample is not None:
                    on_sample(pos, s, persons.xyxy, gap)
                if len(buf) >= self.cfg.chunk_frames:
                    self._flush(buf)
        finally:
            reader.close()
            prep.close()
            detect.close()
        if self.skip_next_k > self.next_k:
            self.next_k = self.skip_next_k
        if len(buf):
            self._flush(buf)
        return done

    def _flush(self, buf) -> None:
        t = list(buf.t)
        state = {"tracker": self.tracker, "motion": self.motion, "next_k": self.next_k,
                 "live": {"gap": self.gap}}
        buf.flush(self.work, self.chunk_id, state)
        self.chunk_t[self.chunk_id] = (len(t), float(t[0]), float(t[-1]))
        self.chunk_id += 1


# ============================================================================== calibration
def _slice_pass1(data, i0: int, i1: int):
    from replay.pipeline import Pass1Data

    k = data.k[i0:i1]
    rows = data.rows[(data.rows[:, 0] >= k[0]) & (data.rows[:, 0] <= k[-1])] if len(k) else data.rows[:0]
    return Pass1Data(k=k, frame=data.frame[i0:i1], t=data.t[i0:i1], motion=list(data.motion[i0:i1]),
                     inliers=data.inliers[i0:i1], rows=rows, signature=data.signature)


def _calib_child(work: str, video: str, size: tuple[int, int], i0: int, i1: int, result: str, log: str) -> None:
    """Separate process: the full automatic calibration on samples [i0, i1) -> JSON file ``result``."""
    try:
        sys.stdout = sys.stderr = open(log, "a", encoding="utf-8", buffering=1)
    except OSError:
        pass
    from replay import autocourt
    from replay.pipeline import load_pass1

    out: dict[str, Any] = {"ok": False, "i0": i0}
    try:
        data = _slice_pass1(load_pass1(Path(work)), i0, i1)
        last = [-1.0]

        def prog(x: float, msg: str) -> None:
            if x - last[0] >= 0.1 or x >= 1.0:
                last[0] = x
                print(f"  [{x * 100:3.0f}%] {msg}")

        res = autocourt.calibrate(Path(video), data, tuple(size), progress=prog)
        if res is not None:
            out = {"ok": True, "i0": i0, "length": float(res.length), "width": float(res.width), "spec": res.spec,
                   "good_frac": float(res.good_frac), "message": res.message, "auto": res.to_json()}
    except BaseException as e:  # noqa: BLE001 - reported through the result file
        import traceback

        traceback.print_exc()
        out["error"] = f"{type(e).__name__}: {e}"
    write_atomic(Path(result), json.dumps(out))


class LiveCalib:
    """Court calibration for a growing analysis: found once, then only the camera pose of new frames is added."""

    MASK_FRAMES = 20

    def __init__(self, out: Path, calib_path: Path, size: tuple[int, int], fps: float, lcfg: LiveConfig,
                 log: Callable[[str], None]):
        self.out, self.calib_path, self.size, self.fps = Path(out), Path(calib_path), tuple(size), fps
        self.lcfg, self.log = lcfg, log
        self.pool = None
        self.pending: list = []  # futures of anchor features
        self.ready: list = []  # Anchor objects, features done, not aligned yet
        self.tail: list = []  # last aligned anchors (continuity for the next stretch)
        self.mask_buf: list = []
        self.static = None
        self.mask_done = False
        self.last_grid_t = -1e9
        self.child: mp.Process | None = None
        self.child_kind = ""
        self.child_result: Path | None = None
        self.last_attempt_n = -10 ** 9
        self.last_recal_t = -1e9
        self.recent: deque = deque()  # (t, good) of newly aligned anchors
        self.message = ""
        auto = self.auto()
        if auto and auto.get("anchors"):
            self.last_grid_t = max(float(a["t"]) for a in auto["anchors"])

    # ---------------------------------------------------------------- calibration file
    def _load_raw(self) -> dict | None:
        try:
            return json.loads(self.calib_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def auto(self) -> dict | None:
        d = self._load_raw()
        a = d.get("auto") if isinstance(d, dict) else None
        return a if isinstance(a, dict) and a.get("C") is not None else None

    def manual(self) -> bool:
        d = self._load_raw()
        return bool(isinstance(d, dict) and d.get("keyframes") and not self.auto())

    def calibrated(self) -> bool:
        d = self._load_raw()
        if not isinstance(d, dict):
            return False
        if d.get("keyframes"):
            return True
        a = d.get("auto") or {}
        return any(x.get("good") for x in a.get("anchors") or [])

    def _save(self, mutate: Callable[[dict], None]) -> None:
        """Re-read, change, write atomically (the user or the GUI may have edited it meanwhile)."""
        d = self._load_raw() or {"video": "video.mp4", "frame_size": list(self.size), "keyframes": []}
        d.setdefault("frame_size", list(self.size))
        mutate(d)
        write_atomic(self.calib_path, json.dumps(d, indent=2))

    # ---------------------------------------------------------------- anchor features
    def _ensure_pool(self):
        if self.pool is None:
            from concurrent.futures import ProcessPoolExecutor

            self.pool = ProcessPoolExecutor(max(1, self.lcfg.anchor_workers), mp_context=mp.get_context("spawn"))
        return self.pool

    def add_frame(self, pos: int, smp, boxes: np.ndarray, gap: bool) -> None:
        if not (gap or smp.t - self.last_grid_t >= self.lcfg.anchor_every_s - 1e-6):
            return
        if not self.lcfg.auto_calibrate and self.auto() is None:
            return  # nothing could use the anchor (no automatic calibration will be made)
        self.last_grid_t = smp.t
        item = (pos, float(smp.t), int(smp.frame), smp.image.copy(), np.asarray(boxes, float).copy())
        if not self.mask_done:
            self.mask_buf.append(item)
            if len(self.mask_buf) >= self.MASK_FRAMES:
                self._make_mask()
            return
        self._submit(item)

    def _make_mask(self) -> None:
        from replay import autocourt as ac

        try:
            self.static = ac.static_overlay_mask([it[3] for it in self.mask_buf])
        except Exception as e:  # noqa: BLE001 - the mask only removes overlay clutter
            self.log(f"Live: overlay mask failed ({e}); continuing without")
            self.static = None
        self.mask_done = True
        buf, self.mask_buf = self.mask_buf, []
        for it in buf:
            self._submit(it)

    def _submit(self, item) -> None:
        from replay import autocourt as ac

        pos, t, frame, img, boxes = item
        self.pending.append(self._ensure_pool().submit(ac._anchor_features,
                                                       (pos, t, frame, img, boxes, self.static, None)))

    def collect(self, wait: bool = False) -> None:
        if wait and not self.mask_done and self.mask_buf:
            self._make_mask()
        keep = []
        for f in self.pending:
            if wait or f.done():
                try:
                    self.ready.append(f.result())
                except Exception as e:  # noqa: BLE001
                    self.log(f"Live: anchor frame skipped ({e})")
            else:
                keep.append(f)
        self.pending = keep
        self.ready.sort(key=lambda a: a.t)

    # ---------------------------------------------------------------- full calibration (separate process)
    def _start_child(self, work: Path, video: Path, i0: int, i1: int, kind: str) -> None:
        ld = live_dir(self.out)
        self.child_result = ld / f"calib_{kind}.json"
        self.child_result.unlink(missing_ok=True)
        ctx = mp.get_context("spawn")
        self.child = ctx.Process(target=_calib_child, name=f"replay-live-{kind}",
                                 args=(str(work), str(video), self.size, i0, i1, str(self.child_result),
                                       str(ld / "calibration.log")))
        self.child.start()
        self.child_kind = kind
        self.message = ("Finding the court automatically" if kind == "initial" else
                        "Camera moved? Re-finding the court")
        self.log(f"Live: {self.message} (samples {i0}..{i1}, separate process)")

    def _poll_child(self) -> None:
        if self.child is None or self.child.is_alive():
            return
        self.child.join(1)
        res = None
        try:
            res = json.loads(self.child_result.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            res = {"ok": False, "error": f"exit code {self.child.exitcode}"}
        kind, self.child = self.child_kind, None
        if not res.get("ok") or res.get("good_frac", 0) < 0.05:
            self.message = "Court not found yet" + (f" ({res.get('error')})" if res.get("error") else "")
            self.log(f"Live: {kind} calibration: court not found {res.get('error') or ''}")
            return
        i0 = int(res.get("i0", 0))
        auto = res["auto"]
        for a in auto.get("anchors", []):
            a["i"] = int(a["i"]) + i0
        t0 = min((float(a["t"]) for a in auto.get("anchors", [])), default=0.0)

        def mutate(d: dict) -> None:
            old = d.get("auto") if isinstance(d.get("auto"), dict) else None
            if kind == "initial" or not old:
                d["auto"] = auto
            else:  # recalibration: newer anchors replace those of the re-checked stretch
                keep = [a for a in old.get("anchors", []) if float(a["t"]) < t0]
                d["auto"] = {**auto, "anchors": keep + auto.get("anchors", [])}
            d["court"] = {"length": round(res["length"], 3), "width": round(res["width"], 3), "spec": res["spec"]}

        self._save(mutate)
        last = max((float(a["t"]) for a in auto.get("anchors", [])), default=t0)
        self.ready = [a for a in self.ready if a.t > last]
        self.tail = []
        self.recent.clear()
        self.message = "Court found: " + res.get("message", "")
        self.log(f"Live: {kind} calibration done: {res.get('message', '')}")

    # ---------------------------------------------------------------- per-frame pose of new frames
    def update(self, work: Path, video: Path, motions: list, times_of: Callable[[], np.ndarray],
               final: bool = False) -> int:
        """Collect anchor features, start/finish the full calibration, align new anchors. Returns the number of
        anchors added to the calibration file."""
        self.collect(wait=final)
        self._poll_child()
        n = len(motions)
        if self.child is not None:
            return 0
        auto = self.auto()
        if auto is None:
            if self.manual() or n == 0 or not self.lcfg.auto_calibrate:
                return 0  # hand calibration: resolved through the camera motion, nothing to add
            dur = n / max(self.fps, 1e-6)
            due = dur >= self.lcfg.calib_min_s and (n - self.last_attempt_n) / self.fps >= self.lcfg.calib_retry_s
            if due or (final and n - self.last_attempt_n > 0):
                self.last_attempt_n = n
                self._start_child(work, video, 0, n, "initial")
            return 0
        added = self._extend(auto, motions, n, final)
        if not final:
            self._check_recalibration(work, video, times_of, n)
        return added

    def _extend(self, auto: dict, motions: list, n: int, final: bool) -> int:
        from replay import autocourt as ac
        from replay import court

        known = max((float(a["t"]) for a in auto.get("anchors", [])), default=-1e9)
        new = [a for a in self.ready if a.i < n and a.t > known + 1e-6]
        self.ready = [a for a in self.ready if a.i >= n]  # not committed yet (cannot happen) / later
        if not new:
            return 0
        cap = len(new) if final else self.lcfg.max_anchors_per_update
        new, rest = new[:cap], new[cap:]
        self.ready = rest + self.ready
        d = self._load_raw() or {}
        c = d.get("court") or {}
        L, W, spec = float(c.get("length", 28.0)), float(c.get("width", 15.0)), c.get("spec", "FIBA")
        classes = tuple(int(x) for x in auto.get("line_classes", (0,)))
        good_c = [a["C"] for a in auto.get("anchors", []) if a.get("good") and "C" in a]
        C = np.asarray(good_c[-1] if good_c else auto["C"], float)
        G, seg = ac.chain(motions, n)
        court.configure(L, W, spec)
        pts, _ = ac.model_points_dirs(0.25)
        pts_c, _ = ac.model_points_dirs(0.35)
        entries = []
        for s_ in sorted({int(seg[a.i]) for a in new}):
            mine = [a for a in new if int(seg[a.i]) == s_]
            prev = [a for a in self.tail if a.i < n and int(seg[a.i]) == s_]
            mem = prev + mine
            for a in mem:
                a.seg = int(seg[a.i])
            dists = {a.i: ac._dist_maps(a, ac.with_weak(classes)) for a in mem}
            ac.track_chunk(mem, {a.i: G[a.i] for a in mem}, {a.i: C for a in mem}, self.size, pts, pts_c, dists)
            for a in mine:
                if a.R is None:
                    continue
                v = ac.ptz_from_R(a.R, a.f)
                entries.append({"t": round(a.t, 4), "i": int(a.i), "seg": int(a.seg), "pan": round(v.pan, 6),
                                "tilt": round(v.tilt, 6), "roll": round(v.roll, 6), "f": round(v.f, 2),
                                "cost": round(float(a.cost), 3), "inliers": round(float(a.inliers), 3),
                                "good": bool(a.good), "C": [round(float(x), 4) for x in C], "live": True})
                self.recent.append((a.t, bool(a.good)))
            good = [a for a in mem if a.good]
            self.tail = (good or mem)[-2:]
        if entries:
            def mutate(dd: dict) -> None:
                au = dd.get("auto") or auto
                au.setdefault("anchors", []).extend(entries)
                dd["auto"] = au
            self._save(mutate)
        ng = sum(e["good"] for e in entries)
        self.message = f"Court: {ng} / {len(entries)} new frames aligned"
        return len(entries)

    def _check_recalibration(self, work: Path, video: Path, times_of: Callable[[], np.ndarray], n: int) -> None:
        if not self.recent:
            return
        t_new = self.recent[-1][0]
        while self.recent and self.recent[0][0] < t_new - self.lcfg.recalib_after_s:
            self.recent.popleft()
        if len(self.recent) < 8 or t_new - self.last_recal_t < self.lcfg.recalib_window_s:
            return
        frac = sum(g for _t, g in self.recent) / len(self.recent)
        if frac >= self.lcfg.recalib_min_good:
            return
        times = times_of()
        i0 = int(np.searchsorted(times, t_new - self.lcfg.recalib_window_s))
        if n - i0 < 30:
            return
        self.last_recal_t = t_new
        self.log(f"Live: only {frac * 100:.0f}% of the last frames align with the court - recalibrating")
        self._start_child(work, video, i0, n, "recal")

    def backfill(self, work: Path, video: Path, fps: float) -> int:
        """After a restart of the live process: anchor frames for samples analysed before the restart but not
        yet in the calibration (decoded again by timestamp, like pass 1)."""
        from replay.pipeline import load_pass1
        from replay.video import VideoSampler

        auto = self.auto()
        if auto is None:
            return 0
        try:
            data = load_pass1(Path(work))
        except SystemExit:
            return 0
        last = max((float(a["t"]) for a in auto.get("anchors", [])), default=-1e9)
        want: dict[int, int] = {}  # k -> position
        g = last
        for i in range(len(data.t)):
            if data.t[i] > last and data.t[i] - g >= self.lcfg.anchor_every_s - 1e-6:
                want[int(data.k[i])] = i
                g = float(data.t[i])
        if not want:
            return 0
        persons = data.rows[data.rows[:, 6] == 0]
        gen = VideoSampler(video, fps, 0.0, 1e12).iter_samples(min(want))
        n = 0
        try:
            for smp in gen:
                if smp.k > max(want):
                    break
                if smp.k in want:
                    self.last_grid_t = -1e9  # force: this is a grid frame
                    self.add_frame(want[smp.k], smp, persons[persons[:, 0] == smp.k][:, 1:5], False)
                    n += 1
        finally:
            gen.close()
        self.last_grid_t = max(self.last_grid_t, g)
        self.log(f"Live: {n} anchor frames from before the restart queued")
        return n

    def wait_child(self, timeout: float | None = None) -> None:
        if self.child is not None:
            self.child.join(timeout)
            self._poll_child()

    def close(self) -> None:
        if self.child is not None and self.child.is_alive():
            self.child.terminate()
            self.child.join(5)
        if self.pool is not None:
            self.pool.shutdown(wait=False, cancel_futures=True)
            self.pool = None


# ============================================================================== pass 2 window + merge
def _chunk_table(work: Path, cache: dict) -> list[tuple[int, int, float, float]]:
    """[(chunk id, n samples, first t, last t)] of the complete chunks (cached per id)."""
    from replay.pipeline import _complete_chunks

    out = []
    for c in _complete_chunks(work):
        if c not in cache:
            try:
                with np.load(work / f"chunk_{c:05d}.npz") as z:
                    t, fr = z["t"], z["frame"]
            except (OSError, ValueError, KeyError):
                break  # being replaced right now: next time
            cache[c] = (len(t), float(t[0]) if len(t) else 0.0, float(t[-1]) if len(t) else 0.0, t, fr)
        n, a, b = cache[c][:3]
        out.append((c, n, a, b))
    return out


def _link(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _cut_track(tr: dict, a: int, b: int) -> dict | None:
    """Part of a track (lists indexed from tr['i0']) within global sample positions [a, b); leading and trailing
    samples without a position are dropped. None if nothing is left."""
    m = len(tr.get("x") or [])
    i0 = int(tr["i0"])
    s, e = max(a, i0), min(b, i0 + m)
    if e <= s:
        return None
    out = {k: (v[s - i0:e - i0] if isinstance(v, list) and len(v) == m else v) for k, v in tr.items()}
    out["i0"] = s
    xs = out["x"]
    lo = next((j for j, v in enumerate(xs) if v is not None), None)
    if lo is None:
        return None
    hi = max(j for j, v in enumerate(xs) if v is not None) + 1
    if lo or hi < len(xs):
        mm = len(xs)
        out = {k: (v[lo:hi] if isinstance(v, list) and len(v) == mm else v) for k, v in out.items()}
        out["i0"] = s + lo
    return out


def _concat_track(head: dict, tail: dict) -> dict:
    """``tail`` (starting after ``head`` ends) appended to ``head``; the gap is filled with no-position samples."""
    mh, mt = len(head["x"]), len(tail["x"])
    gap = int(tail["i0"]) - (int(head["i0"]) + mh)
    out = dict(head)
    for k in set(head) | set(tail):
        hv, tv = head.get(k), tail.get(k)
        h_list = isinstance(hv, list) and len(hv) == mh
        t_list = isinstance(tv, list) and len(tv) == mt
        if not (h_list or t_list):
            continue
        fill = 0 if k in ("f", "st") else None
        hv = hv if h_list else [fill] * mh
        tv = tv if t_list else [fill] * mt
        out[k] = hv + [fill] * max(0, gap) + tv
    return out


class LiveMerger:
    """Frozen past + the newest pass-2 window -> one replay_data.json document."""

    def __init__(self, doc: dict | None = None):
        self.doc = doc if isinstance(doc, dict) and isinstance(doc.get("t"), list) else None

    @staticmethod
    def _empty_ball(m: int) -> dict:
        return {"x": [None] * m, "y": [None] * m, "f": [0] * m, "b": [None] * m, "o": [None] * m}

    def merge(self, w: dict, w0: int, t_all: list[float], frame_all: list[int], margin: int,
              link_cm: float = 150.0) -> dict:
        """``w``: replay_data of the window whose first sample is global position ``w0``; ``t_all``/``frame_all``:
        times / frames of every analysed sample (for gaps); ``margin``: samples at the window start that are
        context only (the frozen document is kept there)."""
        prev = self.doc
        n_w = len(w["t"])
        N = w0 + n_w
        if prev is not None:
            n_p = len(prev["t"])
            ok = n_p <= len(t_all) and (n_p == 0 or abs(prev["t"][-1] - t_all[n_p - 1]) < 1e-3
                                        and abs(prev["t"][0] - t_all[0]) < 1e-3)
            if not ok:  # analysis restarted/resumed differently: start over
                prev = None
        n_p = len(prev["t"]) if prev is not None else 0
        # f: frozen document kept before f; s: the window is used from s on; [f, s) = no document covers it
        f = min(w0 + max(0, margin), n_p) if prev is not None else 0
        f = max(0, min(f, N))
        s = max(f, w0)

        # ------------------------------------------------ per-sample arrays
        def head(key, fill):
            v = list(prev[key][:f]) if prev is not None else []
            if key == "t":
                v += [round(float(x), 3) for x in t_all[len(v):s]]
            elif key == "frame":
                v += [int(x) for x in frame_all[len(v):s]]
            v += [fill] * (s - len(v))
            return v

        out = {k: v for k, v in w.items() if k not in ("tracks", "track_info", "events", "lineup", "t", "frame",
                                                      "calib", "ball", "n")}
        out["t"] = head("t", None) + list(w["t"][s - w0:])
        out["frame"] = head("frame", None) + list(w["frame"][s - w0:])
        out["calib"] = head("calib", 0) + list(w["calib"][s - w0:])
        pb = (prev or {}).get("ball") or self._empty_ball(n_p)
        wb = w.get("ball") or self._empty_ball(n_w)
        ball = {}
        for k in ("x", "y", "f", "b", "o"):
            fill = 0 if k == "f" else None
            hv = list((pb.get(k) or [])[:f])
            hv += [fill] * (s - len(hv))
            tv = list((wb.get(k) or [fill] * n_w)[s - w0:])
            ball[k] = hv + tv
        out["ball"] = ball
        # ------------------------------------------------ tracks
        frozen: dict[Any, dict] = {}
        info_prev = (prev or {}).get("track_info") or {}
        for tr in (prev or {}).get("tracks") or []:
            c = _cut_track(tr, 0, f)
            if c is not None:
                frozen[c["id"]] = c
        int_ids = [int(i) for i in frozen if isinstance(i, int) or str(i).isdigit()]
        next_id = max(int_ids, default=0) + 1
        alive = {tid: tr for tid, tr in frozen.items() if tr["i0"] + len(tr["x"]) == f}  # seen at f-1
        newt = []
        for tr in w.get("tracks") or []:
            g = dict(tr)
            g["i0"] = int(tr["i0"]) + w0
            c = _cut_track(g, s, N)
            if c is not None:
                newt.append((tr["id"], c))
        # seam: a window track starting at f continues a frozen track alive at f-1 when they line up
        pairs = []
        for wid, c in newt:
            if c["i0"] != f or s != f:
                continue
            for tid, fr in alive.items():
                if fr.get("team") != c.get("team") and "unknown" not in (fr.get("team"), c.get("team")):
                    continue
                d = float(np.hypot(c["x"][0] - fr["x"][-1], c["y"][0] - fr["y"][-1]))
                if d <= link_cm:
                    pairs.append((d, wid, tid))
        pairs.sort(key=lambda p: p[0])
        used_w, used_f, id_map = set(), set(), {}
        for _d, wid, tid in pairs:
            if wid in used_w or tid in used_f:
                continue
            used_w.add(wid)
            used_f.add(tid)
            id_map[wid] = tid
        info_w = w.get("track_info") or {}
        kept = {str(x) for x in frozen}
        track_info = {str(k): v for k, v in info_prev.items() if str(k) in kept}
        for wid, c in newt:
            if wid in id_map:
                tid = id_map[wid]
                joined = _concat_track(frozen[tid], {**c, "id": tid, "team": frozen[tid].get("team")})
                frozen[tid] = joined
                continue
            id_map[wid] = next_id
            c = dict(c)
            c["id"] = next_id
            frozen[next_id] = c
            if str(wid) in info_w:
                track_info[str(next_id)] = info_w[str(wid)]
            next_id += 1
        out["tracks"] = sorted(frozen.values(), key=lambda tr: (tr["i0"], str(tr["id"])))
        out["track_info"] = track_info
        # ------------------------------------------------ events
        t_f = out["t"][f] if f < len(out["t"]) else float("inf")  # frozen events before, the window's after
        ev = [e for e in (prev or {}).get("events") or [] if float(e.get("t", 0)) < t_f]
        for e in w.get("events") or []:
            if float(e.get("t", 0)) < t_f:
                continue
            e = json.loads(json.dumps(e))
            det = e.get("details")
            if isinstance(det, dict):
                for k, v in list(det.items()):
                    if k.endswith("track") and v in id_map:
                        det[k] = id_map[v]
            ev.append(e)
        out["events"] = ev
        out["n"] = len(out["t"])
        self.doc = out
        return out


def window_pass2(out: Path, cfg, meta_path: str | None, calib_path: Path, video: Path, lcfg: LiveConfig,
                 merger: LiveMerger, cache: dict, log: Callable[[str], None] = print) -> dict | None:
    """Pass 2 on the newest ``window_s`` of analysed samples, merged into ``out/replay_data.json``."""
    from replay.calibration import CalibrationFile
    from replay.metadata import load_metadata
    from replay.pipeline import run_pass2

    out = Path(out)
    work = out / "work"
    table = _chunk_table(work, cache)
    if not table or not calib_path.exists():
        return None
    t_end = table[-1][3]
    first = len(table) - 1  # first chunk that reaches into the window
    while first > 0 and table[first - 1][3] >= t_end - lcfg.window_s:
        first -= 1
    sel = table[first:]
    w0 = sum(n for _c, n, _a, _b in table[:first])
    win = live_dir(out) / "window"
    ww = win / "work"
    ww.mkdir(parents=True, exist_ok=True)
    want = {f"chunk_{c:05d}.npz" for c, *_ in sel}
    for p in ww.glob("chunk_*.npz"):
        if p.name not in want:
            p.unlink(missing_ok=True)
    for name in sorted(want):
        _link(work / name, ww / name)
    sig = json.loads((work / "run.json").read_text(encoding="utf-8"))
    if lcfg.fast:
        sig["video"] = str(sig["video"]) + FAST_SUFFIX
    (ww / "run.json").write_text(json.dumps(sig, indent=2), encoding="utf-8")
    if not CalibrationFile.load(calib_path).calibrated:
        return None
    t0 = time.time()
    rep = run_pass2(win, cfg, load_metadata(meta_path), Path(calib_path), None if lcfg.fast else Path(video))
    w = json.loads((win / "replay_data.json").read_text(encoding="utf-8"))
    t_all, fr_all = [], []
    for c, *_ in table:
        t_all += cache[c][3].tolist()
        fr_all += cache[c][4].tolist()
    margin = int(round(lcfg.freeze_margin_s * float(sig["process_fps"])))
    doc = merger.merge(w, w0, t_all, fr_all, margin)
    try:
        src = os.path.relpath(Path(video).resolve(), out.resolve()).replace(os.sep, "/")
    except ValueError:
        src = Path(video).resolve().as_uri()
    doc["video_src"] = src
    doc["live"] = {"updated": time.time(), "t_end": doc["t"][-1] if doc["t"] else 0.0, "window_t0": table[first][2],
                   "frozen_until_t": doc["t"][min(len(doc["t"]) - 1, max(0, w0 + margin))] if doc["t"] else 0.0,
                   "pass2_s": round(time.time() - t0, 2)}
    write_atomic(out / "replay_data.json", json.dumps(doc, separators=(",", ":")))
    info = {"wall": time.time(), "t_end": doc["live"]["t_end"], "n": len(doc["t"]), "pass2_s": doc["live"]["pass2_s"],
            "window_s": round(t_end - table[first][2], 1), "valid_pct": rep["calibration"]["valid_pct"],
            "people": rep["people"]["avg_tracked_per_calibrated_frame"]}
    with open(live_dir(out) / UPDATES, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(info) + "\n")
    return info


def _pass2_worker(out: str, cfg_dict: dict, meta_path: str | None, calib_path: str, video: str, lcfg_dict: dict,
                  q_in, q_out) -> None:
    """Separate process: window pass 2 whenever asked (its pure-Python work would starve pass 1 of the GIL)."""
    try:
        sys.stdout = sys.stderr = open(live_dir(Path(out)) / "pass2.log", "a", encoding="utf-8", buffering=1)
    except OSError:
        pass
    from replay.config import Config

    cfg = Config.from_dict(cfg_dict)
    lcfg = LiveConfig(**lcfg_dict)
    prev = None
    try:
        prev = json.loads((Path(out) / "replay_data.json").read_text(encoding="utf-8"))
        if "live" not in prev:
            prev = None
    except (OSError, ValueError):
        prev = None
    merger, cache = LiveMerger(prev), {}
    while True:
        msg = q_in.get()
        if msg == "stop":
            break
        try:
            info = window_pass2(Path(out), cfg, meta_path, Path(calib_path), Path(video), lcfg, merger, cache)
            q_out.put(("done", info))
        except BaseException as e:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            q_out.put(("error", f"{type(e).__name__}: {e}"))


class Pass2Worker:
    def __init__(self, out: Path, cfg, meta_path, calib_path: Path, video: Path, lcfg: LiveConfig):
        ctx = mp.get_context("spawn")
        self.q_in, self.q_out = ctx.Queue(), ctx.Queue()
        lc = asdict(lcfg)
        self.proc = ctx.Process(target=_pass2_worker, name="replay-live-pass2", daemon=True,
                                args=(str(out), cfg.to_dict(), None if meta_path is None else str(meta_path),
                                      str(calib_path), str(video), lc, self.q_in, self.q_out))
        self.proc.start()
        self.busy = False
        self.asked_at = 0.0

    def ask(self) -> None:
        self.q_in.put("run")
        self.busy, self.asked_at = True, time.time()

    def poll(self) -> list[tuple[str, Any]]:
        msgs = []
        while True:
            try:
                msgs.append(self.q_out.get_nowait())
            except queue.Empty:
                break
        if msgs:
            self.busy = False
        if self.busy and not self.proc.is_alive():
            self.busy = False
            msgs.append(("error", f"pass-2 worker stopped (exit code {self.proc.exitcode})"))
        return msgs

    def wait(self, timeout: float = 600.0) -> list[tuple[str, Any]]:
        end = time.time() + timeout
        msgs = []
        while self.busy and time.time() < end:
            msgs += self.poll()
            time.sleep(0.2)
        return msgs

    def stop(self) -> None:
        try:
            self.q_in.put("stop")
            self.proc.join(10)
        finally:
            if self.proc.is_alive():
                self.proc.terminate()
                self.proc.join(3)


# ============================================================================== the loop
class LiveSession:
    def __init__(self, video: Path, out: Path, cfg, meta_path: str | None, calib_path: Path, lcfg: LiveConfig,
                 log: Callable[[str], None] | None = None, stop_event: threading.Event | None = None):
        self.video_arg, self.out = Path(video), Path(out)
        self.cfg, self.meta_path, self.calib_path, self.lcfg = cfg, meta_path, Path(calib_path), lcfg
        self.stop_event = stop_event or threading.Event()
        self.ld = live_dir(self.out)
        self.ld.mkdir(parents=True, exist_ok=True)
        self._logfh = open(self.ld / "live.log", "a", encoding="utf-8", buffering=1)
        self._ext_log = log
        self.state = "starting"
        self.message = ""
        self.p1: LivePass1 | None = None
        self.inplay: LiveInplay | None = None
        self.calib: LiveCalib | None = None
        self.worker: Pass2Worker | None = None
        self.last_update: dict | None = None
        self.updates = 0
        self.started = time.time()
        self.first_seen_wall: float | None = None
        self.decoded_t = 0.0
        self.restarted = False
        self.finalizing = False
        self.error: str | None = None

    def log(self, msg: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        try:
            self._logfh.write(line + "\n")
        except (OSError, ValueError):
            pass
        (self._ext_log or print)(msg)

    # ---------------------------------------------------------------- status
    def write_status(self) -> None:
        p1 = self.p1
        st = {"pid": os.getpid(), "heartbeat": time.time(), "state": self.state, "message": self.message,
              "video": str(self.video_arg), "started": self.started,
              "analysed_t": round(p1.t_last, 2) if p1 else 0.0, "samples": p1.n_pos if p1 else 0,
              "calibrated": bool(self.calib and self.calib.calibrated()),
              "calibration": self.calib.message if self.calib else "",
              "replay_t": (self.last_update or {}).get("t_end"), "updates": self.updates,
              "last_pass2_s": (self.last_update or {}).get("pass2_s"),
              "behind_s": None, "error": self.error}
        if p1 and self.last_update:
            st["behind_s"] = round(max(0.0, p1.t_last - float(self.last_update["t_end"])), 1)
        try:
            write_atomic(self.ld / STATUS, json.dumps(st, indent=1))
        except OSError:
            pass

    def _stop_requested(self) -> bool:
        return self.stop_event.is_set() or (self.ld / STOP_FILE).exists()

    # ---------------------------------------------------------------- main
    def run(self) -> dict:
        from replay.config import used_config_dict
        from replay.metadata import load_metadata

        (self.ld / MARKER).write_text(str(time.time()), encoding="utf-8")
        (self.ld / FINISHED).unlink(missing_ok=True)
        (self.ld / STOP_FILE).unlink(missing_ok=True)
        (self.out / "config.used.json").write_text(json.dumps(used_config_dict(self.cfg), indent=2), encoding="utf-8")
        src = make_source(self.video_arg, self.out)
        self.source = src
        self.meta = load_metadata(self.meta_path)
        skip = self.cfg.inplay.enabled if self.lcfg.skip_breaks is None else bool(self.lcfg.skip_breaks)
        last_step = 0.0
        waiting_logged = False
        # heavy imports + detector load in the background while the recording starts; a heartbeat thread keeps
        # status.json fresh while a long step runs (the GUI treats a stale heartbeat as 'not running')
        self._preload = threading.Thread(target=self._preload_fn, daemon=True, name="live-preload")
        self._preload.start()
        self._beat_stop = threading.Event()
        threading.Thread(target=self._heartbeat, daemon=True, name="live-heartbeat").start()
        try:
            while True:
                if self._stop_requested():
                    self.log("Live: stop requested")
                    return self.finalize(ended=False)
                st = src.poll()
                if st == "restarted":
                    self.log("Live: the recording file was replaced (new recording)")
                    self.restarted = True
                    return self.finalize(ended=False, lost_tail=True)
                if st == "missing" or src.size <= 0:
                    if not waiting_logged:
                        self.log(f"Live: waiting for {self.video_arg} ...")
                        waiting_logged = True
                    self.state, self.message = "waiting", "Waiting for the recording to start"
                    self.write_status()
                    time.sleep(self.lcfg.poll_s)
                    continue
                if self.p1 is None:
                    try:
                        self._open(src.video, skip)
                    except (FileNotFoundError, ValueError, OSError) as e:
                        self.state = "waiting"
                        self.message = (f"Cannot read the recording yet ({e}). An MP4 that is still being written "
                                        "can only be read once it is closed: record to MKV, TS or fragmented MP4.")
                        self.write_status()
                        if src.idle_s > self.lcfg.idle_end_s and not waiting_logged:
                            self.log("Live: " + self.message)
                        waiting_logged = True
                        time.sleep(self.lcfg.poll_s)
                        continue
                ended = src.ended or src.idle_s >= self.lcfg.idle_end_s
                worked = False
                if time.time() - last_step >= self.lcfg.min_new_s or ended or self.p1.capped:
                    last_step = time.time()
                    worked = self._step(final=False) > 0
                self._pass2_tick()
                if ended and not worked:
                    self.log("Live: the recording has ended" + (" (end of playlist)" if src.ended else
                                                                f" (no new data for {src.idle_s:.0f} s)"))
                    return self.finalize(ended=True)
                self.write_status()
                if not worked:
                    time.sleep(self.lcfg.poll_s)
        except BaseException as e:
            self.error = f"{type(e).__name__}: {e}"
            self.state = "error"
            self.log(f"Live: error: {self.error}")
            self.write_status()
            raise
        finally:
            self._close()

    def _preload_fn(self) -> None:
        try:
            import replay.detection as det
            import replay.pipeline  # noqa: F401
            import replay.tracking  # noqa: F401

            self._detector = det.build_detector(self.cfg.detector)
        except BaseException as e:  # noqa: BLE001 - loaded again (with the error shown) when needed
            self._detector = None
            self.log(f"Live: detector preload failed ({e})")

    def _get_detector(self):
        self._preload.join()
        return getattr(self, "_detector", None)

    def _heartbeat(self) -> None:
        while not self._beat_stop.wait(self.lcfg.poll_s):
            if self.state in ("finished", "error"):
                return
            self.write_status()

    def _open(self, video: Path, skip: bool) -> None:
        self.p1 = LivePass1(self.out, video, self.cfg, self.meta, self.log)
        self.p1.get_detector = self._get_detector
        self.p1.write_signature(video, live=True)
        if skip:
            self.inplay = LiveInplay(self.cfg.inplay, self.out, self.log)
        self.calib = LiveCalib(self.out, self.calib_path, self.p1.size, self.p1.fps, self.lcfg, self.log)
        if self.p1.n_pos:  # resumed after the live process was stopped / crashed
            self.calib.backfill(self.out / "work", video, self.p1.fps)
            self.calib.last_grid_t = max(self.calib.last_grid_t, self.p1.t_last)
            self._new_samples = 1  # rebuild the 2D once
        if self.lcfg.player_link:
            self._link_player(video)
        if self.calib.auto() is not None or self.calib.calibrated():  # start it now: its imports take a while
            self.worker = Pass2Worker(self.out, self.cfg, self.meta_path, self.calib_path, video, self.lcfg)
        self.log(f"Live: analysing {video} at {self.p1.fps:.2f} samples/s"
                 + (" (skipping breaks)" if skip else ""))

    def _link_player(self, video: Path) -> None:
        dst = Path(self.lcfg.player_link)
        if dst.exists():
            return
        try:
            os.link(video, dst)
            self.log(f"Live: video for the player: {dst} (hard link)")
        except OSError as e:
            self.log(f"Live: no hard link for the player ({e}); the video is copied when the recording ends")

    def _times(self) -> np.ndarray:
        cache: dict = getattr(self, "_tcache", {})
        self._tcache = cache
        table = _chunk_table(self.out / "work", cache)
        return np.concatenate([cache[c][3] for c, *_ in table]) if table else np.zeros(0)

    def _step(self, final: bool) -> int:
        p1, video = self.p1, self.source.video
        self.state = "analysing"
        if self.inplay is not None:
            try:
                self.inplay.update(video, final=final, holdback_s=self.lcfg.holdback_s)
            except Exception as e:  # noqa: BLE001 - optional: analyse everything instead
                self.log(f"Live: preliminary analysis failed ({e}); analysing everything")
                self.inplay = None
        t0, n0 = time.time(), p1.n_pos
        p1.stop.clear()

        def on_sample(pos, s, boxes, gap):
            self.calib.add_frame(pos, s, boxes, gap)
            if pos % 50 == 0 and not self.finalizing and self._stop_requested():
                p1.stop.set()  # finish this step early; finalize() commits the rest

        n = p1.step(video, self.lcfg.max_step_s if not final else 1e9, self.lcfg.holdback_s, final=final,
                    is_play=self.inplay.is_play if self.inplay is not None else None, on_sample=on_sample)
        if n:
            dt = time.time() - t0
            self.message = f"Analysed up to {p1.t_last / 60:.1f} min ({n} samples in {dt:.1f} s)"
            self.log(f"Live: pass 1 +{n} samples -> t={p1.t_last:.1f}s in {dt:.1f}s "
                     f"({n / max(p1.fps, 1e-6) / max(dt, 1e-6):.2f}x real time)")
        added = self.calib.update(self.out / "work", video, p1.motions, self._times, final=final)
        if added:
            self.log(f"Live: {self.calib.message}")
        self._new_samples = getattr(self, "_new_samples", 0) + (p1.n_pos - n0) + added
        return n

    def _handle(self, msgs: list) -> None:
        for kind, info in msgs:
            if kind == "done" and info:
                self.last_update = info
                self.updates += 1
                self.log(f"Live: 2D updated to t={info['t_end']:.1f}s (pass 2 {info['pass2_s']:.1f}s, "
                         f"calibrated {info['valid_pct']}%)")
            elif kind == "error":
                self.log(f"Live: pass 2 failed: {info}")

    def _pass2_tick(self, force: bool = False) -> None:
        if self.worker is not None:
            self._handle(self.worker.poll())
        if not (self.calib and self.calib.calibrated()):
            return
        if getattr(self, "_new_samples", 0) <= 0 and not force:
            return
        if self.worker is None:
            self.worker = Pass2Worker(self.out, self.cfg, self.meta_path, self.calib_path, self.source.video, self.lcfg)
        if not self.worker.busy:
            self._new_samples = 0
            self.worker.ask()

    # ---------------------------------------------------------------- end
    def finalize(self, ended: bool, lost_tail: bool = False) -> dict:
        """Commit what is left, finish the calibration, write the signature 'process' would write + DONE."""
        self.state = "finishing"
        self.finalizing = True
        self.message = "Recording ended: finishing the analysis"
        self.write_status()
        summary: dict[str, Any] = {"out": str(self.out), "restarted": self.restarted}
        p1 = self.p1
        if p1 is None:
            self.state = "finished"
            self.write_status()
            return summary
        if not lost_tail:
            while self._step(final=ended) > 0:  # catch up completely
                pass
        self.calib.collect(wait=True)
        if self.calib.child is not None:
            self.log("Live: waiting for the court calibration ...")
            self.calib.wait_child()
        self.calib.update(self.out / "work", self.source.video, p1.motions, self._times, final=True)
        self.calib.wait_child()
        while self.calib.update(self.out / "work", self.source.video, p1.motions, self._times, final=True):
            pass
        if self.worker is not None or (not self.lcfg.final_build and self.calib.calibrated()):
            # the 2D catches up to the end (unless the final build replaces it anyway); the worker must be idle
            # before anything else writes replay_data.json
            if self.worker is not None:
                self._handle(self.worker.wait())
            if not self.lcfg.final_build:
                self._pass2_tick(force=True)
                if self.worker is not None:
                    self._handle(self.worker.wait())
            if self.worker is not None:
                self.worker.stop()
                self.worker = None
        video = self.source.video
        if not lost_tail:
            iv = self.inplay.intervals if self.inplay is not None else None
            p1.write_signature(video, live=False, play_intervals=iv)
            if p1.n_pos:
                (self.out / "work" / "DONE").write_text(f"{time.time() - self.started:.1f}s live\n", encoding="utf-8")
        if self.lcfg.player_link and not Path(self.lcfg.player_link).exists() and not lost_tail:
            try:
                shutil.copy2(video, self.lcfg.player_link)
            except OSError as e:
                self.log(f"Live: could not copy the video for the player: {e}")
        built = False
        if self.calib.calibrated() and p1.n_pos and not lost_tail:
            if self.lcfg.final_build:
                self.log("Live: final build (full pass 2 with all steps) ...")
                self.message = "Final build of the whole recording"
                self.write_status()
                from replay.metadata import load_metadata
                from replay.pipeline import run_pass2

                try:
                    rep = run_pass2(self.out, self.cfg, load_metadata(self.meta_path), self.calib_path, video)
                    summary["final"] = {"valid_pct": rep["calibration"]["valid_pct"],
                                        "people": rep["people"]["avg_tracked_per_calibrated_frame"]}
                    built = True
                except Exception as e:  # noqa: BLE001 - the live 2D stays
                    self.log(f"Live: final build failed ({e}); the live replay is kept")
        summary.update(samples=p1.n_pos, analysed_t=p1.t_last, updates=self.updates, built=built,
                       calibrated=self.calib.calibrated(), seconds=round(time.time() - self.started, 1))
        self.state = "finished"
        self.message = ("Recording replaced by a new one; this part is finished" if lost_tail else
                        "Finished" + (" - final build done" if built else ""))
        (self.ld / FINISHED).write_text(json.dumps(summary), encoding="utf-8")
        (self.ld / MARKER).unlink(missing_ok=True)
        (self.ld / STOP_FILE).unlink(missing_ok=True)
        self.write_status()
        self.log(f"Live: finished: {summary}")
        return summary

    def _close(self) -> None:
        if getattr(self, "_beat_stop", None) is not None:
            self._beat_stop.set()
        if self.worker is not None:
            self.worker.stop()
            self.worker = None
        if self.calib is not None:
            self.calib.close()
        try:
            self._logfh.close()
        except OSError:
            pass


def run_live(video: Path, out: Path, cfg=None, meta_path: str | None = None, calib_path: Path | None = None,
             lcfg: LiveConfig | None = None, log: Callable[[str], None] | None = None,
             stop_event: threading.Event | None = None) -> dict:
    """Live analysis of ``video`` (growing file / HLS playlist or folder) into ``out``; returns a summary when
    the recording ends or a stop is requested. A replaced recording continues in ``<out>-restartN``
    (lcfg.on_restart == 'new')."""
    from replay.config import Config

    cfg = cfg or Config()
    lcfg = lcfg or LiveConfig()
    out = Path(out)
    calib_path = Path(calib_path) if calib_path else out / "calibration.json"
    n = 0
    while True:
        out.mkdir(parents=True, exist_ok=True)
        s = LiveSession(video, out, cfg, meta_path, calib_path, lcfg, log, stop_event).run()
        if not s.get("restarted") or lcfg.on_restart != "new":
            return s
        n += 1
        base = Path(str(out).split("-restart")[0])
        out = base.parent / f"{base.name}-restart{n}"
        calib_new = out / "calibration.json"
        out.mkdir(parents=True, exist_ok=True)
        if calib_path.exists():  # same gym and camera: keep the court, not the per-frame anchors
            reuse_court(calib_path, calib_new)
        calib_path = calib_new


def reuse_court(src: Path, dst: Path) -> None:
    """Calibration for a new recording of the same court and camera: court size + camera position from ``src``
    (an automatic calibration), no per-frame anchors (live mode adds them)."""
    d = json.loads(Path(src).read_text(encoding="utf-8"))
    auto = d.get("auto")
    if not isinstance(auto, dict) or auto.get("C") is None:
        raise SystemExit(f"{src} has no automatic calibration (camera position) to reuse")
    good = [a for a in auto.get("anchors", []) if a.get("good") and "C" in a]
    C = good[-1]["C"] if good else auto["C"]
    new = {"video": d.get("video", "video.mp4"), "frame_size": d.get("frame_size"), "keyframes": [],
           "fixed_camera": False, "court": d.get("court"),
           "auto": {**{k: v for k, v in auto.items() if k != "anchors"}, "C": C, "anchors": [],
                    "message": f"court and camera position reused from {Path(src).name}"}}
    write_atomic(Path(dst), json.dumps(new, indent=2))


# ============================================================================== simulation (tests, measurements)
def make_segments(clip: Path, folder: Path, seg_s: float = 2.0, duration_s: float | None = None
                  ) -> list[tuple[Path, float, float]]:
    """Cut a local clip into MPEG-TS segments (stream copy, continuous timestamps): [(file, start_s, end_s)]."""
    import imageio_ffmpeg

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    lst = folder / "segments.csv"
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-i", str(clip)]
    if duration_s:
        cmd += ["-t", f"{duration_s:.3f}"]
    cmd += ["-map", "0:v:0", "-map", "0:a?", "-c", "copy", "-f", "segment", "-segment_time", f"{seg_s}",
            "-segment_format", "mpegts", "-reset_timestamps", "0", "-segment_list", str(lst),
            "-segment_list_type", "csv", str(folder / "seg_%05d.ts")]
    subprocess.run(cmd, check=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    out = []
    for line in lst.read_text(encoding="utf-8").splitlines():
        name, a, b = line.rsplit(",", 2)
        out.append((folder / name, float(a), float(b)))
    return out


def simulate_recording(segments: list[tuple[Path, float, float]], dst: Path, speed: float = 1.0, hls: bool = False,
                       parts: int = 4, log_path: Path | None = None, stop: threading.Event | None = None,
                       on_done: Callable[[], None] | None = None) -> threading.Thread:
    """Background thread that 'records': appends the segments to ``dst`` in real time (each in ``parts`` pieces,
    so the file often ends mid-frame), or with ``hls`` writes them into folder ``dst`` with a growing playlist.
    Each completed segment is logged as {"t0", "t1", "wall"} (JSON lines) for delay measurements."""
    dst = Path(dst)
    stop = stop or threading.Event()

    def run() -> None:
        wall0 = time.time()
        t_base = segments[0][1] if segments else 0.0
        logf = open(log_path, "a", encoding="utf-8", buffering=1) if log_path else None
        listed: list[tuple[str, float]] = []
        if hls:
            dst.mkdir(parents=True, exist_ok=True)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(b"")
        try:
            for path, t0, t1 in segments:
                data = path.read_bytes()
                if hls:
                    wait = wall0 + (t1 - t_base) / speed - time.time()
                    if wait > 0 and stop.wait(wait):
                        return
                    (dst / path.name).write_bytes(data)
                    listed.append((path.name, t1 - t0))
                    _write_playlist(dst / "live.m3u8", listed, ended=False)
                else:
                    with open(dst, "ab") as fh:
                        for j in range(parts):
                            wait = wall0 + (t0 - t_base + (t1 - t0) * (j + 1) / parts) / speed - time.time()
                            if wait > 0 and stop.wait(wait):
                                return
                            fh.write(data[len(data) * j // parts:len(data) * (j + 1) // parts])
                            fh.flush()
                if logf:
                    logf.write(json.dumps({"t0": t0 - t_base, "t1": t1 - t_base, "wall": time.time()}) + "\n")
            if hls:
                _write_playlist(dst / "live.m3u8", listed, ended=True)
        finally:
            if logf:
                logf.close()
            if on_done:
                on_done()

    th = threading.Thread(target=run, daemon=True, name="simulated-recording")
    th.start()
    return th


def _write_playlist(path: Path, listed: list[tuple[str, float]], ended: bool) -> None:
    lines = ["#EXTM3U", "#EXT-X-VERSION:3",
             f"#EXT-X-TARGETDURATION:{int(np.ceil(max((d for _n, d in listed), default=2.0)))}",
             "#EXT-X-MEDIA-SEQUENCE:0"]
    for name, d in listed:
        lines += [f"#EXTINF:{d:.3f},", name]
    if ended:
        lines.append("#EXT-X-ENDLIST")
    write_atomic(path, "\n".join(lines) + "\n")


def delay_report(rec_log: Path, updates: Path) -> dict:
    """End-to-end delay: for each recorded segment, wall time from the segment being written to the first 2D
    update that contains its last frame."""
    recs = [json.loads(x) for x in Path(rec_log).read_text(encoding="utf-8").splitlines() if x.strip()]
    ups = [json.loads(x) for x in Path(updates).read_text(encoding="utf-8").splitlines() if x.strip()] \
        if Path(updates).exists() else []
    delays = []
    for r in recs:
        hit = next((u for u in ups if u["t_end"] >= r["t1"] - 0.5 and u["wall"] >= r["wall"]), None)
        if hit is not None:
            delays.append(hit["wall"] - r["wall"])
    if not delays:
        return {"segments": len(recs), "updates": len(ups), "measured": 0}
    d = np.array(delays)
    rec_s = recs[-1]["t1"] - recs[0]["t0"] if recs else 0.0
    return {"segments": len(recs), "updates": len(ups), "measured": len(d),
            "delay_median_s": round(float(np.median(d)), 1), "delay_p90_s": round(float(np.percentile(d, 90)), 1),
            "delay_min_s": round(float(d.min()), 1), "delay_max_s": round(float(d.max()), 1),
            "recorded_s": round(rec_s, 1),
            "pass2_median_s": round(float(np.median([u["pass2_s"] for u in ups])), 1) if ups else None}


# ============================================================================== CLI
def add_cli(sub) -> None:
    """``replay live`` (hooked into replay/cli.py)."""
    c = sub.add_parser("live", help="near-live analysis of a recording that is still being written "
                                    "(growing .ts/.mkv/fragmented .mp4, or a local HLS .m3u8 folder)")
    c.add_argument("--video", required=True, help="growing video file, or HLS playlist (.m3u8) / folder")
    c.add_argument("--out", required=True, help="output folder (like 'process --out')")
    c.add_argument("--calibration", help="calibration JSON (default: <out>/calibration.json)")
    c.add_argument("--reuse-court", help="calibration of an earlier recording with the same court and camera: "
                                         "reuse its court size and camera position (no automatic search)")
    c.add_argument("--meta", help="game metadata JSON")
    c.add_argument("--config", help="JSON with config overrides")
    c.add_argument("--stride", type=int, help="analyse every Nth video frame (default 3)")
    c.add_argument("--device", choices=["auto", "gpu", "cpu"])
    c.add_argument("--variant", choices=["nano", "small", "medium", "large"])
    c.add_argument("--window", dest="window_s", type=float, help="pass-2 window, seconds (default 120)")
    c.add_argument("--poll", dest="poll_s", type=float, help="file check interval, seconds (default 2)")
    c.add_argument("--every", dest="min_new_s", type=float, help="analyse new video at most every N s (default 10)")
    c.add_argument("--idle-end", dest="idle_end_s", type=float,
                   help="recording counts as ended after this many seconds without growth (default 60)")
    c.add_argument("--calib-after", dest="calib_min_s", type=float,
                   help="seconds of video before the court is searched automatically (default 60)")
    c.add_argument("--skip-breaks", dest="skip_breaks", action="store_true", default=None)
    c.add_argument("--no-skip-breaks", dest="skip_breaks", action="store_false")
    c.add_argument("--no-final-build", action="store_true", help="no full pass 2 when the recording ends")
    c.add_argument("--full-quality", action="store_true",
                   help="live pass 2 also runs the video-decoding steps (much slower)")
    c.add_argument("--on-restart", choices=["new", "stop"], help="the file is replaced by a new recording")
    c.add_argument("--player-link", help="(GUI) hard link the video here for the player")
    c.add_argument("--simulate-from", help="TEST: 'record' this local clip into --video in real time (MPEG-TS) "
                                           "and measure the end-to-end delay")
    c.add_argument("--simulate-hls", action="store_true", help="TEST: simulate an HLS folder instead")
    c.add_argument("--speed", type=float, default=1.0, help="TEST: simulated recording speed (1 = real time)")
    c.add_argument("--duration", type=float, help="TEST: only simulate the first N seconds of the clip")
    c.set_defaults(func=cmd_live)


def cmd_live(a: argparse.Namespace) -> None:
    from replay.config import Config

    cfg = Config.load(a.config)
    if a.stride:
        cfg.frame_stride = a.stride
    if a.device:
        cfg.detector.device = a.device
    if a.variant:
        cfg.detector.variant = a.variant
    lcfg = LiveConfig.from_args(a)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    calib = Path(a.calibration) if a.calibration else out / "calibration.json"
    if a.reuse_court:
        reuse_court(Path(a.reuse_court), calib)
        print(f"Court and camera position reused from {a.reuse_court} -> {calib}")
    sim = None
    stop = threading.Event()
    rec_log = live_dir(out) / "recording.jsonl"
    if a.simulate_from:
        live_dir(out).mkdir(parents=True, exist_ok=True)
        rec_log.unlink(missing_ok=True)
        (live_dir(out) / UPDATES).unlink(missing_ok=True)
        segdir = live_dir(out) / "sim_segments"
        shutil.rmtree(segdir, ignore_errors=True)
        segs = make_segments(Path(a.simulate_from), segdir, 2.0, a.duration)
        print(f"Simulating a recording of {a.simulate_from}: {len(segs)} segments, {segs[-1][2]:.0f} s, "
              f"speed {a.speed}x -> {a.video}")
        sim = simulate_recording(segs, Path(a.video), a.speed, hls=a.simulate_hls, log_path=rec_log)
        if not a.simulate_hls and lcfg.idle_end_s > 20:
            lcfg.idle_end_s = 20.0
    t0 = time.time()
    summary = run_live(Path(a.video), out, cfg, a.meta, calib, lcfg, stop_event=stop)
    print(json.dumps(summary, indent=1))
    if sim is not None:
        rep = delay_report(rec_log, live_dir(out) / UPDATES)
        rep["wall_s"] = round(time.time() - t0, 1)
        try:
            rep["cpu_load_pct"] = _cpu_load()
        except Exception:  # noqa: BLE001
            pass
        write_atomic(live_dir(out) / "delay.json", json.dumps(rep, indent=1))
        print("Delay:", json.dumps(rep))


def _cpu_load() -> float | None:
    if sys.platform != "win32":
        return round(os.getloadavg()[0], 2)
    r = subprocess.run(["powershell", "-NoProfile", "-Command",
                        "(Get-CimInstance Win32_Processor | Measure-Object -Property LoadPercentage -Average).Average"],
                       capture_output=True, text=True, timeout=30,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    return float(r.stdout.strip()) if r.stdout.strip() else None


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="replay.live")
    sub = p.add_subparsers(dest="cmd", required=True)
    add_cli(sub)
    a = p.parse_args(["live", *(argv if argv is not None else sys.argv[1:])])
    a.func(a)


if __name__ == "__main__":
    main()
