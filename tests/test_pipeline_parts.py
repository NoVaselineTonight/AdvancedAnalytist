import cv2
import numpy as np
import pytest

from replay.config import Config
from replay.pipeline import _Stage
from replay.video import VideoSampler


def _write_video(path, n=24, fps=30.0):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (64, 48))
    for i in range(n):
        img = np.zeros((48, 64, 3), np.uint8)
        img[:, :, 0] = i * 10  # frame number encoded in the blue channel
        w.write(img)
    w.release()


@pytest.mark.parametrize("stride", [1, 2, 3])
def test_stride_samples_every_nth_frame(tmp_path, stride):
    path = tmp_path / "v.avi"
    _write_video(path)
    cfg = Config(frame_stride=stride)
    fps = cfg.effective_fps(30.0)
    frames = [s.frame for s in VideoSampler(path, fps).iter_samples()]
    assert frames == list(range(0, 24, stride))


def test_stage_preserves_order_and_forwards_errors():
    st = _Stage(lambda: iter(range(100)), 4)
    assert list(st) == list(range(100))

    def boom():
        yield 1
        raise ValueError("bad frame")

    st = _Stage(boom, 2)
    it = iter(st)
    assert next(it) == 1
    with pytest.raises(ValueError, match="bad frame"):
        next(it)


def test_stage_close_unblocks_producer():
    st = _Stage(lambda: iter(range(10_000)), 2)
    next(iter(st))
    st.close()
    assert not st.thread.is_alive()


def test_store_low_score_persons_switch():
    """Task 9: with store_threshold, person boxes >= store are kept, balls only >= threshold; signature v2 only when on."""
    import numpy as np

    from replay.config import Config
    from replay.detection import PERSON, BALL, Detections, StoreLowDetector, activation_threshold
    from replay.pipeline import PIPELINE_VERSION, _without_off

    class Fake:
        name, license_note, threshold = "fake", "", 0.3

        def detect(self, img):
            conf = np.array([0.9, 0.2, 0.05, 0.2], np.float32)
            keep = conf > self.threshold
            d = Detections(np.zeros((4, 4), np.float32), conf, np.array([PERSON, PERSON, PERSON, BALL], np.int32))
            return d.subset(keep)

    det = StoreLowDetector(Fake(), 0.3, 0.12)
    d = det.detect(None)
    assert np.allclose(sorted(d.conf.tolist()), [0.2, 0.9]) and (d.cls == PERSON).all()
    cfg = Config()
    assert "store_threshold" not in _without_off(cfg.to_dict()["detector"])
    assert activation_threshold({"detector": {"threshold": 0.3}}) == 0.0  # old runs: nothing filtered
    assert activation_threshold({"detector": {"threshold": 0.3, "store_threshold": 0.12}}) == 0.3
    assert PIPELINE_VERSION == 1


def test_low_score_rows_only_extend_identities():
    import numpy as np

    from replay.config import CourtTrackConfig
    from replay.tracklets import link_tracks

    fps = 10.0
    k = np.arange(23)
    xy = np.c_[np.linspace(5, 9.4, 23), np.full(23, 7.0)]
    R = np.tile(np.eye(2) * 0.04, (23, 1, 1))
    feats = np.tile([40, 30, -20, 50, 0, 0], (23, 1)).astype(float)
    bt = np.r_[np.full(10, 1), np.full(3, -1), np.full(10, 2)]  # ByteTrack broke across 3 low-score samples
    conf = np.r_[np.full(10, 0.8), np.full(3, 0.15), np.full(10, 0.8)]
    boxes = np.c_[100 + 10 * k, np.full(23, 300.0), 150 + 10 * k, np.full(23, 440.0)].astype(float)
    res = link_tracks(k, xy, R, feats, bt, conf, fps, CourtTrackConfig(), boxes=boxes, n_samples=23)
    assert len(set(res.ids[res.ids > 0].tolist())) == 1  # one identity through the crumbs
    lone = link_tracks(np.arange(5), xy[:5], R[:5], feats[:5], np.full(5, -1), np.full(5, 0.15), fps,
                       CourtTrackConfig(), boxes=boxes[:5], n_samples=5)
    assert (lone.ids == 0).all()  # crumbs alone never form an identity
