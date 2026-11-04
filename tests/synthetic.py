"""A synthetic pinhole camera looking at the court, used to generate exact ground truth."""

from __future__ import annotations

import numpy as np

FRAME = (1280, 720)


def look_at_camera(eye, target, f=900.0, size=FRAME):
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    fwd = target - eye
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.stack([right, down, fwd])  # world -> camera
    t = -R @ eye
    K = np.array([[f, 0, size[0] / 2], [0, f, size[1] / 2], [0, 0, 1]])
    return K, R, t


def court_to_image_matrix(K, R, t):
    """Exact court(z=0) -> image homography."""
    return K @ np.column_stack([R[:, 0], R[:, 1], t])


def project_court(K, R, t, pts):
    G = court_to_image_matrix(K, R, t)
    q = np.hstack([np.asarray(pts, float), np.ones((len(pts), 1))]) @ G.T
    return q[:, :2] / q[:, 2:3]


def sideline_camera():
    # elevated 6 m, 9 m behind the near sideline, looking at mid-court
    return look_at_camera(eye=(14.0, -9.0, 6.0), target=(14.0, 7.5, 0.0), f=700.0)
