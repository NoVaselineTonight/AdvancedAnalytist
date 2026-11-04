"""Frame tap runner: frames in order on the worker thread, finish() after the last one, errors isolated."""

from pathlib import Path

import numpy as np

from replay.frame_tap import FrameTap, FrameTapRunner, TapFrame, TapInfo, _LazyFrame


class _Rec(FrameTap):
    name = "rec"
    queue_size = 2

    def start(self, info):
        self.seen, self.done = [], None

    def wants(self, k, t):
        return k % 2 == 0

    def process(self, f):
        self.seen.append(f.k)

    def finish(self, work, completed):
        self.done = completed


class _Bad(FrameTap):
    name = "bad"

    def process(self, f):
        raise RuntimeError("boom")


def _frame(k):
    return _LazyFrame(k, k / 10, lambda: TapFrame(k=k, i=k, frame=3 * k, t=k / 10, image=np.zeros((4, 4, 3), np.uint8),
                                                 persons=np.zeros((0, 4)), person_conf=np.zeros(0),
                                                 bytetrack_ids=np.zeros(0), row0=0))


def test_runner_order_finish_and_isolation(tmp_path: Path):
    rec, bad = _Rec(), _Bad()
    run = FrameTapRunner([rec, bad], TapInfo(video=tmp_path / "v.mp4", work=tmp_path, process_fps=10.0,
                                            frame_size=(4, 4), video_fps=30.0, resumed=False))
    for k in range(20):
        run.offer(_frame(k))
    run.close(True)
    assert rec.seen == list(range(0, 20, 2)) and rec.done is True
    st = run.stats()
    assert st["rec"]["ok"] and st["rec"]["frames"] == 10 and not st["bad"]["ok"]


def test_clip_crops_round_trip(tmp_path: Path):
    from types import SimpleNamespace

    from replay import team_appearance as ta
    img = np.random.default_rng(0).integers(0, 255, (360, 640, 3), np.uint8)
    sq = ta.clip_square(img, [100, 50, 160, 250])
    assert sq.shape == (224, 224, 3) and ta.clip_square(img, [10, 10, 30, 40]) is None  # too small
    import cv2
    data = SimpleNamespace(signature={"video": "v"}, rows=np.zeros((5, 14)))
    ok, buf = cv2.imencode(".webp", ta.clip_box(img, [100, 50, 160, 250]), [cv2.IMWRITE_WEBP_QUALITY, 101])
    ta.save_clip_crops(tmp_path, data, 2.0, [3], [buf.tobytes()])
    got = ta.load_clip_crops(tmp_path, data)
    assert list(got) == [3] and got[3].shape == (224, 224, 3)
    assert np.array_equal(got[3], sq)  # lossless: the same pixels clip_roles.crop would use
    data2 = SimpleNamespace(signature={"video": "other"}, rows=np.zeros((5, 14)))
    assert ta.load_clip_crops(tmp_path, data2) == {}  # key mismatch
