"""Processing pipeline.

Pass 1 (slow, resumable): sample frames -> detect -> camera motion -> track -> jersey features.
    Output: ``<out>/work/chunk_NNNNN.npz`` + ``state_NNNNN.pkl`` per chunk. A crash loses at
    most one chunk; re-running the same command resumes after the last complete chunk.
Pass 2 (fast, re-runnable): calibration -> court projection -> filtering -> teams -> smoothing
    -> CSV/JSON export, HTML viewer, report. Re-run it after adding calibration keyframes
    or changing filter/team/smoothing settings; no video decoding needed.
"""

from __future__ import annotations

import json
import os
import pickle
import queue
import re
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
from tqdm import tqdm

from replay.camera import MotionEstimator
from replay.config import Config
from replay.detection import BALL, PERSON, Detections, build_detector
from replay.features import FEATURE_DIM, jersey_features
from replay.metadata import GameMeta
from replay.video import VideoSampler, probe

if TYPE_CHECKING:  # pass 1 only; importing it (supervision) costs seconds, so pass 2 / the GUI skip it
    from replay.tracking import CMCByteTrack

PIPELINE_VERSION = 1
_CHUNK_RE = re.compile(r"chunk_(\d{5})\.npz")
_STATE_RE = re.compile(r"state_(\d{5})\.pkl")


# ---------------------------------------------------------------------------- pass 1
def _run_signature(video: Path, cfg: Config, start_s: float, end_s: float | None, fps: float) -> dict:
    det = _without_off({k: v for k, v in cfg.to_dict()["detector"].items() if k not in ("backend", "device")})
    return {
        # version 2 = low-score person rows stored (Task 9); only when that switch is on
        "pipeline_version": 2 if det.get("store_threshold") is not None else PIPELINE_VERSION,
        "video": str(video.resolve()),
        "video_size_bytes": video.stat().st_size,
        "start_s": start_s,
        "end_s": end_s,
        "process_fps": round(fps, 6),
        "chunk_frames": cfg.chunk_frames,
        "detector": det,
        "tracking": _without_off(cfg.to_dict()["tracking"]),
        "motion": _without_off(cfg.to_dict()["motion"]),
    }


# opt-in pass-1 switches (TRACKING_FIX_PLAN Tasks 6/7): left out of the signature while off, so existing analyses
# stay valid until a switch is turned on
_OFF_VALUES = {"fallback_samples": 0, "kalman_fps_scale": False, "box_expand": 0.0, "store_threshold": None}


def _without_off(d: dict) -> dict:
    return {k: v for k, v in d.items() if not (k in _OFF_VALUES and v == _OFF_VALUES[k])}


def _complete_chunks(work: Path) -> list[int]:
    """Chunk files are written via atomic rename, so every chunk_NNNNN.npz present is complete."""
    return sorted(int(m.group(1)) for p in work.glob("chunk_*.npz") if (m := _CHUNK_RE.fullmatch(p.name)))


def _latest_state(work: Path) -> int | None:
    ids = [int(m.group(1)) for p in work.glob("state_*.pkl") if (m := _STATE_RE.fullmatch(p.name))]
    return max(ids) if ids else None


class Cancelled(Exception):
    """Raised when a GUI user stops analysis; progress so far is kept and resumes next time."""


# ---------------------------------------------------------------------------- in-play skipping
def _play_k_ranges(sampler: VideoSampler, intervals: list[tuple[float, float]] | None,
                   total: int) -> list[tuple[int, int]]:
    """Sample-index ranges [k0, k1) to analyse. ``intervals`` (absolute video seconds, from the preliminary
    analysis in replay/inplay.py) limit pass 1 to the in-play parts; None = everything."""
    if intervals is None:
        return [(0, total)] if total > 0 else []
    fps, t0 = sampler.process_fps, sampler.start_s
    out: list[list[int]] = []
    for a, b in sorted(intervals):
        k0 = max(0, int(np.ceil((a - t0) * fps - 1e-6)))
        k1 = min(total, int(np.floor((b - t0) * fps + 1e-6)) + 1)
        if k1 <= k0:
            continue
        if out and k0 <= out[-1][1]:
            out[-1][1] = max(out[-1][1], k1)
        else:
            out.append([k0, k1])
    return [(a, b) for a, b in out]


def _segment_of(ranges: list[tuple[int, int]], k: int) -> int | None:
    for i, (a, b) in enumerate(ranges):
        if a <= k < b:
            return i
    return None


def _done_before(ranges: list[tuple[int, int]], k: int) -> int:
    """Number of analysed samples with index < k."""
    return sum(max(0, min(b, k) - a) for a, b in ranges)


def _iter_ranges(sampler: VideoSampler, ranges: list[tuple[int, int]], first_k: int):
    """Samples of the given ranges from ``first_k`` on; seeks over the skipped spans."""
    for a, b in ranges:
        if b <= first_k:
            continue
        gen = sampler.iter_samples(max(a, first_k))
        try:
            for smp in gen:
                if smp.k >= b:
                    break
                yield smp
        finally:
            gen.close()


def _tracker_break(tracker: CMCByteTrack) -> None:
    """Forget all live tracks (time jumps over a skipped break); ID counters continue, so IDs stay unique."""
    bt = tracker.tracker
    bt.tracked_tracks, bt.lost_tracks, bt.removed_tracks = [], [], []


def run_pass1(video: Path, out: Path, cfg: Config, meta: GameMeta, start_s: float = 0.0,
              end_s: float | None = None, restart: bool = False,
              progress: Callable[[int, int], None] | None = None,
              cancel: threading.Event | None = None,
              status: Callable[[str], None] | None = None,
              play_intervals: list[tuple[float, float]] | None = None, taps: list | None = None) -> Path:
    """``play_intervals``: only analyse these spans (absolute seconds; see replay/inplay.py). Skipped spans
    leave gaps in the sample index k: no rows, and motion=None on the first sample after each gap, so pass 2
    treats them like a break in the camera chain and tracks do not bridge them.

    ``taps``: replay.frame_tap.FrameTap consumers that get the decoded frames (+ person boxes) on their own
    worker threads ("decode once, fan out"); they never change the pass-1 output."""
    from replay.tracking import CMCByteTrack

    work = out / "work"
    fps = cfg.effective_fps(probe(video).fps)
    sig = _run_signature(video, cfg, start_s, end_s, fps)
    if play_intervals is not None:
        sig["play_intervals"] = [[round(float(a), 2), round(float(b), 2)] for a, b in play_intervals]
    sig = json.loads(json.dumps(sig))  # tuples -> lists, as stored
    sig_path = work / "run.json"
    if work.exists() and sig_path.exists():
        old = json.loads(sig_path.read_text(encoding="utf-8"))
        if old != sig:
            if not restart:
                raise SystemExit(
                    f"The saved analysis in {work} used different settings/video. Use --restart (GUI: "
                    "'Re-analyze with new settings') to discard it, or choose another --out directory.")
            shutil.rmtree(work)
    elif restart and work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    sig_path.write_text(json.dumps(sig, indent=2), encoding="utf-8")

    sampler = VideoSampler(video, fps, start_s, end_s)
    ranges = _play_k_ranges(sampler, play_intervals, sampler.expected_samples())
    if play_intervals is not None and not ranges:
        print("Preliminary analysis found no play in range; analysing everything.")
        ranges = _play_k_ranges(sampler, None, sampler.expected_samples())
    total = _done_before(ranges, 1 << 62)
    if play_intervals is not None:
        print(f"Analysing {len(ranges)} in-play spans: {total} of {sampler.expected_samples()} samples")
    if (work / "DONE").exists():
        print(f"Pass 1 already complete ({len(_complete_chunks(work))} chunks).")
        return work

    last = _latest_state(work)
    if last is not None:
        # a chunk without its state (crash between the two writes) is redone
        for c in _complete_chunks(work):
            if c > last:
                (work / f"chunk_{c:05d}.npz").unlink()
        with open(work / f"state_{last:05d}.pkl", "rb") as fh:
            state = pickle.load(fh)
        tracker, motion, next_k, chunk_id = state["tracker"], state["motion"], state["next_k"], last + 1
        print(f"Resuming after chunk {last} at sample {next_k} (t={sampler.sample_time(next_k):.1f}s)")
    else:
        for c in _complete_chunks(work):
            (work / f"chunk_{c:05d}.npz").unlink()
        tracker = CMCByteTrack(cfg.tracking, fps)
        motion = MotionEstimator(cfg.motion, sampler.info.size, meta.overlay_regions) if cfg.motion.enabled else None
        next_k, chunk_id = 0, 0

    print("Loading detector ...")
    detector = build_detector(cfg.detector)
    print(f"Detector: {detector.name} ({detector.license_note}); analysing {fps:.2f} frames/s of video")
    if status is not None:
        status(f"Detector: {detector.name}")

    from replay.frame_tap import FrameTapRunner, TapFrame, TapInfo, _LazyFrame

    resumed = last is not None
    if taps is None:  # default: jersey / team-colour / re-ID caches from these frames (replay/pass1_taps.py)
        from replay.pass1_taps import default_taps

        taps = default_taps(cfg)
    tap_run = FrameTapRunner(taps or [], TapInfo(video=Path(video), work=work, process_fps=fps,
                                                 frame_size=tuple(sampler.info.size), video_fps=sampler.info.fps,
                                                 resumed=resumed, cfg=cfg))
    n_seen = 0 if not resumed else -1  # analysed samples so far (= Pass1Data index) / rows so far
    n_rows = 0 if not resumed else -1
    completed = False
    buf = _ChunkBuffer()
    t0 = time.time()
    bar = tqdm(total=total, initial=_done_before(ranges, next_k), unit="frame", desc="detect+track",
               smoothing=0.05)
    last_k = next_k - 1
    prev_seg = _segment_of(ranges, next_k - 1) if next_k > 0 else None
    # Four overlapping stages: decode (thread) -> detector pre-processing (thread) -> inference (thread)
    # -> motion/track/features/chunk writing (here). The inference thread only feeds the GPU; the queue in
    # front of the main stage absorbs the pauses while a chunk is written.
    reader = _Stage(lambda: _iter_ranges(sampler, ranges, next_k), 8)
    if hasattr(detector, "prepare") and hasattr(detector, "detect_prepared"):
        prepare, infer = detector.prepare, detector.detect_prepared
    else:  # plain detector with only detect()
        prepare, infer = (lambda img: None), (lambda _x, img: detector.detect(img))
    prep = _Stage(lambda: ((smp, prepare(smp.image)) for smp in reader), 4)
    from replay.frame_tap import GPU_LOCK

    def infer_locked(x, img):  # one GPU model call at a time (DirectML is not safe with concurrent sessions)
        with GPU_LOCK:
            return infer(x, img)

    detect = _Stage(lambda: ((smp, infer_locked(x, smp.image)) for smp, x in prep), 16)
    try:
        for s, dets in detect:
            persons = dets.subset(dets.cls == PERSON)
            balls = dets.subset(dets.cls == BALL)
            seg = _segment_of(ranges, s.k)
            if prev_seg is not None and seg != prev_seg:  # jumped over a skipped break
                if motion is not None:
                    motion.prev_gray = motion.prev_mask = None  # -> M is None for this sample
                _tracker_break(tracker)
            prev_seg = seg
            if motion is not None:
                M, inliers = motion.update(s.image, persons.xyxy)
                M_approx = motion.fallback(M)
            else:
                M, inliers, M_approx = np.eye(3), -1, None
            if M is not None:
                tracker.apply_camera_motion(M)
            elif M_approx is not None:  # estimate failed: last pan rate, wider gate (Task 6; off by default)
                tracker.apply_camera_motion(M_approx, approx=True)
            ids = tracker.update(persons)
            feats = jersey_features(s.image, persons.xyxy)
            buf.add(s.k, s.frame, s.t, M, inliers, persons, ids, feats, balls)
            if tap_run:
                tap_run.offer(_LazyFrame(s.k, s.t, lambda s=s, persons=persons, ids=ids, i=n_seen, r0=n_rows: TapFrame(
                    k=s.k, i=i, frame=s.frame, t=s.t, image=s.image, persons=persons.xyxy, person_conf=persons.conf,
                    bytetrack_ids=np.asarray(ids), row0=r0)))
            if not resumed:
                n_seen += 1
                n_rows += len(persons) + len(balls)
            last_k = s.k
            bar.update(1)
            if progress is not None:
                progress(_done_before(ranges, s.k + 1), total)
            if cancel is not None and cancel.is_set():
                buf.flush(work, chunk_id, {"tracker": tracker, "motion": motion, "next_k": s.k + 1})
                bar.close()
                raise Cancelled()
            if len(buf) >= cfg.chunk_frames:
                buf.flush(work, chunk_id, {"tracker": tracker, "motion": motion, "next_k": s.k + 1})
                chunk_id += 1
        bar.close()
        if len(buf):
            buf.flush(work, chunk_id, {"tracker": tracker, "motion": motion, "next_k": last_k + 1})
        completed = True
    finally:
        # upstream first: each closed stage also ends the iteration of the stage consuming it
        reader.close()
        prep.close()
        detect.close()
        if tap_run:  # taps finish (write their caches) before DONE: pass 2 never sees a half-written cache
            tw = time.time()
            tap_run.close(completed)
            st = tap_run.stats()
            print(f"Frame taps (finish wait {time.time() - tw:.1f}s): {st}")
            try:
                (work / "frame_taps.json").write_text(json.dumps(st, indent=1), encoding="utf-8")
            except OSError:
                pass
    (work / "DONE").write_text(f"{time.time() - t0:.1f}s\n", encoding="utf-8")
    print(f"Pass 1 done in {time.time() - t0:.0f}s")
    return work


_END = object()


class _Stage:
    """Runs a generator in a background thread and buffers up to ``size`` results (pipeline parallelism).

    Iterating yields the results in order; an exception in the thread is re-raised in the consumer.
    """

    def __init__(self, make_gen: Callable[[], Any], size: int):
        self.q: queue.Queue = queue.Queue(size)
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, args=(make_gen,), daemon=True)
        self.thread.start()

    def _put(self, item: Any) -> bool:
        while not self.stop.is_set():
            try:
                self.q.put(item, timeout=0.2)
                return True
            except queue.Full:
                continue
        return False

    def _run(self, make_gen: Callable[[], Any]) -> None:
        try:
            for item in make_gen():
                if not self._put(item):
                    return
        except BaseException as e:  # forwarded to the consumer
            self._put(("__error__", e))
            return
        self._put(_END)

    def __iter__(self):
        while True:
            try:
                item = self.q.get(timeout=0.2)
            except queue.Empty:
                if self.stop.is_set():  # closed: a downstream stage thread consuming us ends too
                    return
                continue
            if item is _END:
                return
            if isinstance(item, tuple) and len(item) == 2 and item[0] == "__error__":
                raise item[1]
            yield item

    def close(self) -> None:
        self.stop.set()
        while True:  # unblock a producer waiting on a full queue
            try:
                self.q.get_nowait()
            except queue.Empty:
                break
        self.thread.join(timeout=5)


class _ChunkBuffer:
    def __init__(self) -> None:
        self.clear()

    def clear(self) -> None:
        self.k: list[int] = []
        self.frame: list[int] = []
        self.t: list[float] = []
        self.motion: list[np.ndarray] = []
        self.inliers: list[int] = []
        self.rows: list[np.ndarray] = []  # [k, x1, y1, x2, y2, conf, cls, track_id, f0..f5]

    def __len__(self) -> int:
        return len(self.k)

    def add(self, k: int, frame: int, t: float, M: np.ndarray | None, inliers: int, persons: Detections,
            ids: np.ndarray, feats: np.ndarray, balls: Detections) -> None:
        self.k.append(k)
        self.frame.append(frame)
        self.t.append(t)
        self.motion.append(M if M is not None else np.full((3, 3), np.nan))
        self.inliers.append(inliers)
        n, m = len(persons), len(balls)
        if n + m == 0:
            return
        rows = np.full((n + m, 8 + FEATURE_DIM), np.nan, np.float64)
        rows[:, 0] = k
        rows[:n, 1:5], rows[:n, 5], rows[:n, 6], rows[:n, 7] = persons.xyxy, persons.conf, PERSON, ids
        rows[:n, 8:] = feats
        rows[n:, 1:5], rows[n:, 5], rows[n:, 6], rows[n:, 7] = balls.xyxy, balls.conf, BALL, -1
        self.rows.append(rows)

    def flush(self, work: Path, chunk_id: int, state: dict) -> None:
        rows = np.concatenate(self.rows) if self.rows else np.zeros((0, 8 + FEATURE_DIM))
        tmp = work / f"tmpchunk_{chunk_id:05d}.npz"
        np.savez_compressed(tmp, k=np.array(self.k), frame=np.array(self.frame), t=np.array(self.t),
                            motion=np.array(self.motion), inliers=np.array(self.inliers), rows=rows)
        tmp.replace(work / f"chunk_{chunk_id:05d}.npz")
        # state is written last: a chunk only counts as complete once its state exists
        stmp = work / f"state_{chunk_id:05d}.tmp"
        with open(stmp, "wb") as fh:
            pickle.dump(state, fh)
        stmp.replace(work / f"state_{chunk_id:05d}.pkl")
        old = work / f"state_{chunk_id - 2:05d}.pkl"  # keep the last two states only
        if old.exists():
            old.unlink()
        self.clear()


# ---------------------------------------------------------------------------- pass 2
@dataclass
class Pass1Data:
    k: np.ndarray
    frame: np.ndarray
    t: np.ndarray
    motion: list[np.ndarray | None]
    inliers: np.ndarray
    rows: np.ndarray
    signature: dict


def load_pass1(work: Path) -> Pass1Data:
    chunks = _complete_chunks(work)
    if not chunks:
        raise SystemExit(f"No processed chunks in {work}; run `process` first.")
    def _read(c: int) -> dict[str, np.ndarray]:  # decompress every array now (zlib releases the GIL)
        with np.load(work / f"chunk_{c:05d}.npz") as z:
            return {name: z[name] for name in ("k", "frame", "t", "motion", "inliers", "rows")}

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as ex:
        parts = list(ex.map(_read, chunks))
    motion = [None if not np.isfinite(m).all() else m for p in parts for m in p["motion"]]
    return Pass1Data(
        k=np.concatenate([p["k"] for p in parts]),
        frame=np.concatenate([p["frame"] for p in parts]),
        t=np.concatenate([p["t"] for p in parts]),
        motion=motion,
        inliers=np.concatenate([p["inliers"] for p in parts]),
        rows=np.concatenate([p["rows"] for p in parts]),
        signature=json.loads((work / "run.json").read_text(encoding="utf-8")),
    )


def run_pass2(out: Path, cfg: Config, meta: GameMeta, calib_path: Path, video_for_viewer: Path | None = None,
              swap_teams: bool = False) -> dict:
    """See replay/postprocess.py."""
    from replay.postprocess import run_pass2 as _run

    return _run(out, cfg, meta, calib_path, video_for_viewer, swap_teams)
