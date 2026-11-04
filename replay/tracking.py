"""Multi-object tracking: supervision's ByteTrack plus camera-motion compensation.

ByteTrack predicts each track with a Kalman filter in *image* space. When the
camera pans, every player shifts in the image even if they stand still, the
prediction misses, and IDs switch. Before each update we warp every track's
Kalman state by the estimated frame-to-frame camera homography (the "GMC"
idea from BoT-SORT), so prediction happens in a camera-stabilised frame.
"""

from __future__ import annotations

import warnings

import numpy as np
import supervision as sv

from replay.config import TrackingConfig
from replay.detection import Detections


class CMCByteTrack:
    def __init__(self, cfg: TrackingConfig, process_fps: float):
        self.cfg = cfg
        # supervision computes max_time_lost = frame_rate / 30 * lost_track_buffer
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            self.tracker = self._new_tracker(cfg, process_fps)

    @staticmethod
    def _new_tracker(cfg: TrackingConfig, process_fps: float) -> sv.ByteTrack:
        tr = sv.ByteTrack(
            track_activation_threshold=cfg.activation_threshold,
            lost_track_buffer=int(round(cfg.lost_track_buffer_s * 30)),
            minimum_matching_threshold=cfg.matching_threshold,
            frame_rate=process_fps,
            minimum_consecutive_frames=cfg.min_consecutive_frames,
        )
        CMCByteTrack._scale_kalman(tr, cfg, process_fps)
        return tr

    @staticmethod
    def _scale_kalman(tr, cfg: TrackingConfig, process_fps: float) -> None:
        """Task 7: supervision's noise weights are per 30-fps frame; one sample here spans 30/process_fps frames.
        Both Kalman instances (activate uses kalman_filter, multi_predict uses shared_kalman) are scaled."""
        if not getattr(cfg, "kalman_fps_scale", False) or not process_fps:
            return
        r = 30.0 / float(process_fps)
        for name in ("kalman_filter", "shared_kalman"):
            kf = getattr(tr, name, None)
            if kf is not None and hasattr(kf, "_std_weight_position"):
                kf._std_weight_position = (1.0 / 20) * r
                kf._std_weight_velocity = (1.0 / 160) * r ** 1.5

    # supervision's ByteTrack class is wrapped by a decorator, so pickle can't find it by name.
    # Persist its attribute dict and rebuild a fresh instance on load (used for resume).
    def __getstate__(self) -> dict:
        return {"cfg": self.cfg, "tracker_state": self.tracker.__dict__}

    def __setstate__(self, state: dict) -> None:
        self.cfg = state["cfg"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            self.tracker = sv.ByteTrack()
        self.tracker.__dict__.update(state["tracker_state"])  # Kalman instances (scaled or not) come with it

    def apply_camera_motion(self, M: np.ndarray, approx: bool = False) -> None:
        """Warp all live track states by M (previous frame pixels -> current frame pixels).

        approx: M is a guess (last pan rate during a motion dropout, Task 6): position covariances are widened by
        (0.1 h)^2 so a wrong guess widens the gate instead of mis-centring it."""
        if not self.cfg.camera_motion_compensation:
            return
        A = M[:2, :2] / M[2, 2]
        for track in list(self.tracker.tracked_tracks) + list(self.tracker.lost_tracks):
            mean = getattr(track, "mean", None)
            if mean is None:
                continue
            # state = [cx, cy, aspect, h, vcx, vcy, va, vh]
            c = np.array([mean[0], mean[1], 1.0])
            q = M @ c
            if abs(q[2]) < 1e-9:
                continue
            new_c = q[:2] / q[2]
            # local scale of the warp at this point (zoom)
            J = (A - np.outer(new_c, M[2, :2] / M[2, 2])) / (q[2] / M[2, 2])
            s = float(np.sqrt(abs(np.linalg.det(J))))
            mean[0], mean[1] = new_c
            mean[3] *= s
            mean[4:6] = J @ mean[4:6]
            mean[7] *= s
            cov = getattr(track, "covariance", None)
            if approx and cov is not None:
                v = (0.1 * mean[3]) ** 2
                cov[0, 0] += v
                cov[1, 1] += v

    def update(self, persons: Detections) -> np.ndarray:
        """Returns a track id per input detection (-1 = not (yet) assigned to a confirmed track)."""
        ids = np.full(len(persons), -1, np.int64)
        xyxy = persons.xyxy.astype(np.float32).reshape(-1, 4)
        e = float(getattr(self.cfg, "box_expand", 0.0) or 0.0)
        if e > 0 and len(xyxy):  # Task 7: expanded boxes keep IoU with a lagging prediction (ids are all we read)
            pad = 0.5 * e * (xyxy[:, 3] - xyxy[:, 1])
            xyxy = xyxy + np.stack([-pad, -pad, pad, pad], 1)
        dets = sv.Detections(
            xyxy=xyxy,
            confidence=persons.conf.astype(np.float32),
            class_id=np.zeros(len(persons), int),
            data={"det_idx": np.arange(len(persons))},
        )
        out = self.tracker.update_with_detections(dets)
        if len(out) and out.tracker_id is not None:
            ids[np.asarray(out.data["det_idx"], int)] = np.asarray(out.tracker_id, np.int64)
        return ids
