"""Calibration features measured during pass 1 (replay/autocal_tap.py) = the ones autocourt measures itself."""

from __future__ import annotations

from types import SimpleNamespace

import cv2
import numpy as np

from replay import autocourt
from replay.autocal_tap import AutocalTap
from replay.frame_tap import TapFrame, TapInfo


def _image():
    img = np.full((360, 640, 3), (190, 190, 188), np.uint8)
    for x in range(40, 600, 80):
        cv2.line(img, (x, 120), (x + 30, 350), (30, 30, 30), 2)
    for y in range(140, 350, 50):
        cv2.line(img, (20, y), (620, y + 10), (30, 30, 30), 2)
    cv2.ellipse(img, (320, 250), (120, 40), 0, 0, 360, (30, 30, 30), 2)
    return img


def test_raw_then_finish_equals_direct_features():
    img = _image()
    a = autocourt._anchor_features((5, 0.5, 15, img, None, None, None))
    b = autocourt._anchor_finish((autocourt.anchor_raw((5, 0.5, 15, img, None)), None))
    key = lambda x: set(map(tuple, x.tolist()))  # noqa: E731 - same pixels (order is irrelevant downstream)
    ka, kb = key(a.lines), key(b.lines)
    assert len(ka & kb) >= 0.98 * max(len(ka), len(kb))
    assert (a.vp is None) == (b.vp is None)
    if a.vp is not None:
        assert np.allclose(a.vp, b.vp, atol=1e-3)


def test_tap_writes_features_that_calibrate_accepts(tmp_path):
    tap = AutocalTap()
    tap.start(TapInfo(video=tmp_path / "v.mp4", work=tmp_path, process_fps=10.0, frame_size=(640, 360),
                      video_fps=30.0, resumed=False))
    ts, frames = [], []
    for i in range(40):
        t = i * 0.1
        if tap.wants(i, t):
            tap.process(TapFrame(k=i, i=i, frame=3 * i, t=t, image=_image(), persons=np.zeros((0, 4)),
                                 person_conf=np.zeros(0), bytetrack_ids=np.zeros(0), row0=-1))
        ts.append(t)
        frames.append(3 * i)
    tap.finish(tmp_path, True)
    data = SimpleNamespace(t=np.array(ts), frame=np.array(frames))
    d = autocourt.load_features(tmp_path, data)
    assert d is not None and [r[0] for r in d["raws"]] == [0, 10, 20, 30]
    data.frame = data.frame + 1  # other pass-1 data -> cache ignored
    assert autocourt.load_features(tmp_path, data) is None
