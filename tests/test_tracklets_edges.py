"""Edge cases of the offline linker (replay/tracklets.py) - cloud task 6."""

import numpy as np
import pytest

from replay.config import CourtTrackConfig
from replay.tracklets import (CameraFrames, _median, add_identity_confidence, detection_confidence, link_tracks)

FPS = 10.0


def _lt(k, xy, *, bt=None, conf=None, feats=None, **kw):
    k = np.asarray(k)
    n = len(k)
    xy = np.asarray(xy, float).reshape(n, 2)
    R = np.tile(np.eye(2) * 0.04, (n, 1, 1))
    return link_tracks(k, xy, R, np.zeros((n, 6)) if feats is None else feats,
                       np.full(n, -1) if bt is None else np.asarray(bt), np.full(n, 0.9) if conf is None else conf,
                       FPS, CourtTrackConfig(), **kw)


def test_empty_and_single_detection():
    r = _lt(np.zeros(0, int), np.zeros((0, 2)))
    assert r.ids.shape == (0,) and r.identities == {}
    r = _lt([0], [[5.0, 5.0]])
    assert r.ids.tolist() == [1] and r.identities[1]["detections"] == 1 and r.identities[1]["links"] == 0


def test_low_score_only_identities_are_left_out():
    r = _lt(range(5), np.c_[np.linspace(5, 6, 5), np.full(5, 5.0)], conf=np.full(5, 0.1))
    assert r.ids.tolist() == [0] * 5 and r.stats["identities"] == 0


def test_without_boxes_court_positions_alone_link():
    """Bug 4.11: link_tracks without boxes crashed in the summary cache (full_h is None)."""
    n = 20
    k = np.repeat(np.arange(n), 2)
    xy = np.c_[np.tile([5.0, 20.0], n), np.full(2 * n, 5.0)]  # two people 15 m apart, no ByteTrack ids
    r = _lt(k, xy)
    assert len(set(r.ids.tolist())) == 2
    assert len(set(r.ids[0::2].tolist())) == 1 and len(set(r.ids[1::2].tolist())) == 1


def test_without_court_or_boxes_nothing_is_linked_but_nothing_crashes():
    n = 10
    r = link_tracks(np.arange(n), np.full((n, 2), np.nan), np.full((n, 2, 2), np.nan), np.zeros((n, 6)),
                    np.full(n, -1), np.full(n, 0.9), FPS, CourtTrackConfig())
    assert (r.ids > 0).all()


def test_nan_kit_features_follow_bytetrack():
    n = 10
    r = _lt(range(n), np.c_[np.linspace(5, 6, n), np.full(n, 5.0)], bt=np.full(n, 3), feats=np.full((n, 6), np.nan))
    assert set(r.ids.tolist()) == {1}


def test_two_boxes_in_one_sample_are_two_identities():
    k = np.repeat(np.arange(5), 2)
    r = _lt(k, np.full((10, 2), 5.0))
    assert set(r.ids[0::2].tolist()) != set(r.ids[1::2].tolist())
    assert len(set(r.ids[0::2].tolist())) == 1


def test_camera_frames_without_motion():
    assert CameraFrames(None, 5).transform(0, 3) is None
    cf = CameraFrames([None] + [np.eye(3)] * 4, 5)
    assert np.allclose(cf.transform(0, 4), np.eye(3)) and np.allclose(cf.transform(4, 0), np.eye(3))


@pytest.mark.parametrize("a", [np.array([[1.0, np.nan], [3.0, 2.0]]), np.array([[1.0, 5], [3, 2], [0, 7], [9, 1]]),
                               np.array([[2.0, 1.0]]), np.array([[1.0, 2.0], [np.nan, 4.0], [5.0, 6.0]])])
def test_median_matches_numpy(a):
    assert np.array_equal(_median(a), np.median(a, axis=0), equal_nan=True)


def test_detection_and_track_confidence():
    dc = detection_confidence(np.array([0, 1, 2, 1]), {1: {"confidence": 0.5}, 2: {}})
    assert np.isnan(dc[0]) and dc[1] == 0.5 and np.isnan(dc[2]) and dc[3] == 0.5
    info = {7: {}, 8: {}}
    tid = np.array([1, 1, 1, 2])
    add_identity_confidence(info, {"det_confidence": np.array([0.2, 0.4, np.nan, np.nan])}, tid,
                            np.ones(4, bool), {1: 7, 2: 8})
    assert info[7]["identity_confidence"] == 0.3 and "identity_confidence" not in info[8]
    add_identity_confidence(info, {"det_confidence": np.zeros(2)}, tid, np.ones(4, bool), {1: 7})  # wrong length
    add_identity_confidence(info, None, tid, np.ones(4, bool), {1: 7})
    assert info[7]["identity_confidence"] == 0.3
