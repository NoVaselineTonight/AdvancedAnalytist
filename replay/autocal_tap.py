"""Automatic court calibration, image part, measured DURING pass 1 (frame tap, SPEED_PLAN S3).

pass 1 already decodes every analysed frame; this tap takes one frame per second (the calibration's anchor
grid), and a small CPU process pool measures its line segments, vanishing-point rotation and line pixels
(autocourt.anchor_raw) while the GPU detects. At the end of pass 1 the results are written to
work/autocal_features.pkl; autocourt.calibrate(..., work=...) then starts from them instead of reading the video
again. Pass 1's own output never changes (frame taps cannot alter it).

Use: run_pass1(..., taps=[AutocalTap()]).
"""

from __future__ import annotations

import os
import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from replay import autocourt
from replay.frame_tap import FrameTap, TapFrame, TapInfo

# streaming (early court lock): first calibration round after this many anchors (~ seconds of video), then a
# round every STREAM_EVERY anchors; each round aligns only the new frames if the court stays the same
STREAM_FIRST = int(os.environ.get("REPLAY_AUTOCAL_STREAM_FIRST", 150))
STREAM_EVERY = int(os.environ.get("REPLAY_AUTOCAL_STREAM_EVERY", 150))
STREAM_WORKERS = 4  # pass 1 is mostly GPU-bound; the rounds must keep up with it
STREAM_FINISH_WAIT_S = 0.0  # pass 1 never waits for a running round (the autocal job does the fit)
MAX_SMALL = 80  # frames kept (small, evenly spread over the video) for finding broadcast overlays


def _workers() -> int:
    return max(1, min(4, (os.cpu_count() or 4) // 3))  # pass 1 needs the rest (decode, tracking)


class AutocalTap(FrameTap):
    name = "autocal"
    queue_size = 8
    blocking = True  # never lose an anchor; the pool keeps up easily (1 frame per video second)

    def __init__(self, every_s: float = 1.0):
        self.every_s = every_s

    def start(self, info: TapInfo) -> None:
        super().start(info)
        self.enabled = not info.resumed  # a resumed pass 1 did not show us its earlier frames
        self.last_t = -1e9
        self.pool = None
        self.futs: list = []
        self.raws: list = []
        self.small: list[np.ndarray] = []
        self.small_stride, self.n_seen = 1, 0
        self.stream_pool, self.stream_fut, self.next_round = None, None, STREAM_FIRST
        self.last_avail = 0
        self.rounds: list = []
        self.written, self.n_shards = 0, 0
        self.chunk_n: dict[str, int] = {}  # samples per complete pass-1 chunk file
        try:  # an old state / old features of another run must never be reused
            import shutil

            (Path(info.work) / autocourt.STATE_FILE).unlink(missing_ok=True)
            (Path(info.work) / autocourt.FEATURES_FILE).unlink(missing_ok=True)
            shutil.rmtree(Path(info.work) / autocourt.FEATURES_DIR, ignore_errors=True)
        except OSError:
            pass

    def wants(self, k: int, t: float) -> bool:
        if not self.enabled or t - self.last_t < self.every_s - 1e-6:
            return False
        self.last_t = t
        return True

    def process(self, frame: TapFrame) -> None:
        if frame.i < 0:
            return
        if self.pool is None:
            self.pool = ProcessPoolExecutor(_workers())
        img = frame.image.copy()  # the image is shared with pass 1
        boxes = np.asarray(frame.persons, float).reshape(-1, 4)
        self.futs.append(self.pool.submit(autocourt.anchor_raw, (frame.i, frame.t, frame.frame, img, boxes)))
        if self.n_seen % self.small_stride == 0:  # evenly spread sample for the overlay mask
            hh = int(round(img.shape[0] * autocourt.WORK_W / img.shape[1]))
            self.small.append(cv2.resize(img, (autocourt.WORK_W, hh), interpolation=cv2.INTER_AREA))
            if len(self.small) > MAX_SMALL:
                self.small = self.small[::2]
                self.small_stride *= 2
        self.n_seen += 1
        while len(self.futs) > 3 * _workers():  # bounded memory; blocks pass 1 only if the pool falls behind
            self.raws.append(self.futs.pop(0).result())
        if STREAM_FIRST > 0 and self.n_seen >= self.next_round:
            self._stream_round()

    def _available_anchors(self) -> int:
        """Anchors whose sample is already in a complete pass-1 chunk (what a round can use)."""
        for f in Path(self.info.work).glob("chunk_*.npz"):
            if f.name not in self.chunk_n:
                try:
                    with np.load(f) as z:
                        self.chunk_n[f.name] = int(len(z["t"]))
                except Exception:  # noqa: BLE001 - being written / unreadable: later
                    continue
        n = sum(self.chunk_n.values())
        while self.futs and self.futs[0].done():  # measured anchors, in order
            self.raws.append(self.futs.pop(0).result())
        return sum(1 for r in self.raws if r[0] < n)

    def _write_features(self, work: Path) -> None:
        """Only the anchors measured since the last write go to a new shard (a long video has ~1 GB of them)."""
        d = Path(work) / autocourt.FEATURES_DIR
        d.mkdir(exist_ok=True)
        for name, obj in ((f"shard_{self.n_shards:04d}.pkl", self.raws[self.written:]), ("small.pkl", self.small)):
            if name.startswith("shard") and not obj:
                continue
            tmp = d / (name + ".tmp")
            tmp.write_bytes(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))
            tmp.replace(d / name)
        if len(self.raws) > self.written:
            self.n_shards += 1
            self.written = len(self.raws)

    def _stream_round(self) -> None:
        """Calibrate on what pass 1 has written so far, in a background process (never waits for it)."""
        if self.stream_fut is not None:
            if not self.stream_fut.done():
                return
            try:
                self.rounds.append(self.stream_fut.result())
            except Exception as e:  # noqa: BLE001 - streaming is only a head start
                self.rounds.append({"ok": False, "error": repr(e)})
            self.stream_fut = None
        avail = self._available_anchors()
        if avail < max(autocourt.STREAM_MIN_LOCK, self.last_avail + STREAM_EVERY // 2):
            return  # pass 1 has not written enough new data yet: try again with the next frame
        self.last_avail = avail
        self.next_round = self.n_seen + STREAM_EVERY
        self._write_features(self.info.work)
        if self.stream_pool is None:
            self.stream_pool = ProcessPoolExecutor(1)
        self.stream_fut = self.stream_pool.submit(autocourt.stream_job, str(self.info.video), str(self.info.work),
                                                  tuple(self.info.frame_size), STREAM_WORKERS)

    def finish(self, work: Path, completed: bool) -> None:
        try:
            if not completed or not self.enabled:
                for f in self.futs:
                    f.cancel()
                return
            self.raws += [f.result() for f in self.futs]
            self.futs = []
            if len(self.raws) < 3:
                return
            if self.stream_fut is not None:  # a running round is usually worth finishing (bounded wait)
                try:
                    self.stream_fut.result(timeout=STREAM_FINISH_WAIT_S)
                except Exception:  # noqa: BLE001 - timeout / error: terminated below
                    pass
            if self.stream_fut is not None and self.stream_fut.done():
                try:
                    self.rounds.append(self.stream_fut.result())
                except Exception as e:  # noqa: BLE001
                    self.rounds.append({"ok": False, "error": repr(e)})
                self.stream_fut = None
            self._write_features(work)
            if self.rounds:
                print(f"Calibration during the analysis: {len(self.rounds)} round(s), last: {self.rounds[-1]}")
        finally:
            if self.pool is not None:
                self.pool.shutdown(wait=False, cancel_futures=True)  # never block pass 1 here
            if self.stream_pool is not None:
                if self.stream_fut is not None and not self.stream_fut.done():
                    # pass 1 must not wait for a running round: stop it (the previous round's state stays valid;
                    # a round writes its state atomically at its very end)
                    for proc in list(getattr(self.stream_pool, "_processes", {}).values()):
                        try:
                            proc.terminate()
                        except Exception:  # noqa: BLE001
                            pass
                self.stream_pool.shutdown(wait=False, cancel_futures=True)
