"""Frame tap: give the frames pass 1 already decodes to other consumers ("decode once, fan out").

pass 1 (pipeline.run_pass1(..., taps=[...])) calls, per analysed sample, ``tap.wants(k, t)`` in its main loop
(must be cheap) and, when True, hands a ``TapFrame`` to the tap's own worker thread through a bounded queue.
The image is the one pass 1 decoded (BGR uint8, H x W x 3) and is SHARED: consumers must not modify it (copy
first). A consumer that needs processes (e.g. a calibration feature pool) submits from ``process`` to its pool.

Lifecycle (all calls from pass 1, in this order):
    start(info: TapInfo)              main thread, before the first frame (info.resumed: pass 1 continues a
                                      previous run -> earlier frames were not seen; most taps then do nothing)
    wants(k, t) -> bool               main thread, every sample (cheap)
    process(frame: TapFrame)          the tap's worker thread, frames strictly in sample order
    finish(work, completed: bool)     the tap's worker thread, after the last frame; completed = pass 1 wrote
                                      DONE (False on cancel / error). Write caches here (pass 2 then finds them).

Back-pressure: ``queue_size`` frames per tap. When a tap's queue is full pass 1 waits (``blocking=True``, the
default: nothing is lost, outputs stay identical) or skips the frame for that tap (``blocking=False``). The time
pass 1 spent waiting per tap is reported in work/DONE and printed (``FrameTapRunner.stats``).
"""

from __future__ import annotations

import queue
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# DirectML (ONNX Runtime) crashes (access violation) when two sessions run at the same time from different threads:
# every GPU model call in pass 1 (detector, and the taps' OCR / re-ID models) holds this lock. Re-entrant, cheap.
GPU_LOCK = threading.RLock()


@dataclass
class TapInfo:
    video: Path
    work: Path
    process_fps: float
    frame_size: tuple[int, int]  # (w, h)
    video_fps: float
    resumed: bool  # pass 1 resumed: frames before the first one seen now were analysed in an earlier run
    cfg: Any = None  # replay.config.Config of the run


@dataclass
class TapFrame:
    k: int  # sample index (Pass1Data.k)
    i: int  # position of this sample in the analysed sequence (= Pass1Data index once loaded; -1 if resumed)
    frame: int  # video frame number
    t: float  # timestamp, s (= Pass1Data.t)
    image: np.ndarray  # BGR, shared, read-only
    persons: np.ndarray  # (n, 4) person boxes xyxy (pass-1 detections of this sample)
    person_conf: np.ndarray  # (n,)
    bytetrack_ids: np.ndarray  # (n,) ByteTrack ids
    row0: int  # index of this sample's first person row in Pass1Data.rows (-1 if resumed); persons are rows
    #           row0 .. row0 + n - 1 (balls follow)


class FrameTap:
    """Base class; override what you need."""

    name = "tap"
    queue_size = 32
    blocking = True

    def start(self, info: TapInfo) -> None:
        self.info = info

    def wants(self, k: int, t: float) -> bool:
        return True

    def process(self, frame: TapFrame) -> None:
        pass

    def finish(self, work: Path, completed: bool) -> None:
        pass


@dataclass
class _Slot:
    tap: FrameTap
    q: queue.Queue
    thread: threading.Thread | None = None
    wait_s: float = 0.0
    frames: int = 0
    skipped: int = 0
    error: str | None = None
    work_s: float = 0.0
    finish_s: float = 0.0
    backlog_at_close: int = 0
    done: threading.Event = field(default_factory=threading.Event)


_STOP = object()


class FrameTapRunner:
    """Owns one worker thread + bounded queue per tap (used by pipeline.run_pass1)."""

    def __init__(self, taps: list[FrameTap], info: TapInfo):
        self.slots: list[_Slot] = []
        for tap in taps or []:
            try:
                tap.start(info)
            except Exception:  # noqa: BLE001 - a tap must never break pass 1
                print(f"frame tap {getattr(tap, 'name', tap)}: start failed\n{traceback.format_exc()}")
                continue
            s = _Slot(tap, queue.Queue(max(1, int(getattr(tap, "queue_size", 32)))))
            s.thread = threading.Thread(target=self._run, args=(s,), daemon=True, name=f"tap-{tap.name}")
            s.thread.start()
            self.slots.append(s)
        self._completed = False
        self._work = info.work

    def __bool__(self) -> bool:
        return bool(self.slots)

    def _run(self, s: _Slot) -> None:
        while True:
            item = s.q.get()
            if item is _STOP:
                break
            if s.error is not None:
                continue  # failed tap: drain
            t0 = time.perf_counter()
            try:
                s.tap.process(item)
            except Exception:  # noqa: BLE001
                s.error = traceback.format_exc()
                print(f"frame tap {s.tap.name}: disabled after an error\n{s.error}")
            s.work_s += time.perf_counter() - t0
        try:
            if s.error is None:
                t0 = time.perf_counter()
                s.tap.finish(self._work, self._completed)
                s.finish_s = time.perf_counter() - t0
                s.work_s += s.finish_s
        except Exception:  # noqa: BLE001
            s.error = traceback.format_exc()
            print(f"frame tap {s.tap.name}: finish failed\n{s.error}")
        s.done.set()

    def offer(self, make_frame) -> None:
        """make_frame() -> TapFrame, built at most once and only if some tap wants this sample."""
        frame = None
        for s in self.slots:
            if s.error is not None:
                continue
            try:
                if not s.tap.wants(make_frame.k, make_frame.t):
                    continue
            except Exception:  # noqa: BLE001
                s.error = traceback.format_exc()
                continue
            if frame is None:
                frame = make_frame()
            if s.tap.blocking:
                t0 = time.perf_counter()
                s.q.put(frame)
                s.wait_s += time.perf_counter() - t0
                s.frames += 1
            else:
                try:
                    s.q.put_nowait(frame)
                    s.frames += 1
                except queue.Full:
                    s.skipped += 1

    def close(self, completed: bool) -> None:
        """Stop the workers (after the last frame) and let every tap finish; waits for them."""
        self._completed = completed
        for s in self.slots:
            s.backlog_at_close = s.q.qsize()
            s.q.put(_STOP)
        for s in self.slots:
            s.done.wait()
            if s.thread is not None:
                s.thread.join(timeout=5)

    def stats(self) -> dict:
        return {s.tap.name: {"frames": s.frames, "skipped": s.skipped, "pass1_wait_s": round(s.wait_s, 2),
                             "work_s": round(s.work_s, 2), "finish_s": round(s.finish_s, 2),
                             "backlog_at_close": s.backlog_at_close, "ok": s.error is None} for s in self.slots}


class _LazyFrame:
    """Callable that builds the TapFrame on demand (offer() only builds it when a tap wants the sample)."""

    def __init__(self, k: int, t: float, build):
        self.k, self.t, self._build = k, t, build

    def __call__(self) -> TapFrame:
        return self._build()
