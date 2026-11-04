"""Video reading with timestamp-based downsampling.

Phone and stream recordings are often variable-frame-rate, so samples are
chosen by *timestamp* on a regular grid ``start + k / process_fps`` rather than
by taking every n-th frame. Sample ``k`` is therefore well defined across runs,
which is what makes resume-after-crash possible.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")  # silence h264 warnings after seeks
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

PRE_ROLL_S = 2.0  # decode this much before a seek target so reference frames are valid


@dataclass
class VideoInfo:
    path: str
    fps: float
    frame_count: int
    width: int
    height: int

    @property
    def duration_s(self) -> float:
        return self.frame_count / self.fps if self.fps else 0.0

    @property
    def size(self) -> tuple[int, int]:
        return (self.width, self.height)


@dataclass
class Sample:
    k: int  # sample index on the processing grid
    frame: int  # original video frame index (derived from timestamp)
    t: float  # original timestamp, seconds
    image: np.ndarray  # BGR


def probe(path: str | Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    info = VideoInfo(str(path), float(cap.get(cv2.CAP_PROP_FPS) or 30.0), int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                     int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    return info


class VideoSampler:
    def __init__(self, path: str | Path, process_fps: float, start_s: float = 0.0, end_s: float | None = None):
        self.info = probe(path)
        self.path = str(path)
        self.process_fps = float(process_fps)
        self.start_s = float(start_s)
        self.end_s = float(end_s) if end_s is not None else self.info.duration_s
        if self.process_fps > self.info.fps + 1e-6:
            raise ValueError(f"process_fps {process_fps} exceeds video fps {self.info.fps:.2f}")

    def sample_time(self, k: int) -> float:
        return self.start_s + k / self.process_fps

    def expected_samples(self) -> int:
        return max(0, int(np.floor((self.end_s - self.start_s) * self.process_fps + 1e-6)))

    def iter_samples(self, first_k: int = 0) -> Iterator[Sample]:
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            raise FileNotFoundError(self.path)
        try:
            target = self.sample_time(first_k)
            seek_to = max(0.0, target - PRE_ROLL_S)
            if seek_to > 0:
                cap.set(cv2.CAP_PROP_POS_MSEC, seek_to * 1000.0)
            half_frame = 0.5 / self.info.fps
            k = first_k
            while True:
                target = self.sample_time(k)
                if target >= self.end_s:
                    return
                if not cap.grab():
                    return
                t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                if t + half_frame < target:
                    continue
                ok, image = cap.retrieve()
                if not ok:
                    return
                yield Sample(k=k, frame=int(round(t * self.info.fps)), t=t, image=image)
                k += 1
                # If the source has a gap longer than one sample period, skip the missed grid points.
                while self.sample_time(k) + half_frame <= t:
                    k += 1
        finally:
            cap.release()


def read_frame_at(path: str | Path, t: float) -> tuple[float, np.ndarray] | None:
    """Accurately read the frame at time ``t`` (seconds). Returns (actual_t, image)."""
    info = probe(path)
    cap = cv2.VideoCapture(str(path))
    try:
        seek_to = max(0.0, t - PRE_ROLL_S)
        if seek_to > 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, seek_to * 1000.0)
        half_frame = 0.5 / info.fps
        while cap.grab():
            ft = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if ft + half_frame >= t:
                ok, img = cap.retrieve()
                return (ft, img) if ok else None
        return None
    finally:
        cap.release()
