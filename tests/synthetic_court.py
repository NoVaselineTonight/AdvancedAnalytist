"""Synthetic basketball videos with a known camera, for testing the automatic calibration on many
kinds of gyms and cameras (court colour, size, marking standard, camera height/position, panning,
zooming, a camera that moves along the sideline).

render_video() writes an MP4 and returns pass-1-like data (times, frames, floor motion between samples)
plus the true camera for every sample.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from replay import court
from replay.autocourt import court_to_image_R
from replay.camera_model import rotation


@dataclass
class Scene:
    length: float = 28.0
    width: float = 15.0
    spec: str = "FIBA"
    floor_bgr: tuple = (70, 120, 185)  # wood
    line_bgr: tuple = (240, 240, 240)  # white lines
    other_lines_bgr: tuple | None = (60, 150, 60)  # a volleyball court in another colour (clutter)
    size: tuple[int, int] = (1280, 720)


@dataclass
class CameraPath:
    duration_s: float = 40.0
    fps: float = 10.0
    C0: tuple = (14.0, -6.0, 5.0)
    C1: tuple | None = None  # end position (moving camera); None = static
    pan_range_deg: tuple = (50.0, 130.0)  # panned back and forth between these
    tilt_deg: float = 14.0
    f0: float = 1100.0
    f1: float | None = None  # zoom to this focal length (None = constant)
    roll_deg: float = 0.3


@dataclass
class SynthData:
    k: np.ndarray
    frame: np.ndarray
    t: np.ndarray
    motion: list
    inliers: np.ndarray
    rows: np.ndarray
    signature: dict
    C: np.ndarray  # (n, 3) true camera positions
    R: list  # true rotations
    f: np.ndarray


def _floor_texture(scene: Scene, mpp: float, margin: float, rng) -> tuple[np.ndarray, float]:
    W = int((scene.length + 2 * margin) / mpp)
    H = int((scene.width + 2 * margin) / mpp)
    tex = np.zeros((H, W, 3), np.uint8)
    tex[:] = scene.floor_bgr
    # planks: subtle stripes along x + noise
    for y in range(0, H, max(1, int(0.12 / mpp))):
        tex[y:y + 1] = np.clip(np.array(scene.floor_bgr) * 0.92, 0, 255)
    noise = rng.normal(0, 6, (H, W, 1))
    tex = np.clip(tex.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    th = max(1, int(round(0.05 / mpp)))

    def to_px(p):
        return np.int32(np.round((np.asarray(p) + margin) / mpp)).reshape(-1, 1, 2)

    if scene.other_lines_bgr is not None:  # volleyball court 18 x 9, centred, other colour
        cx, cy = scene.length / 2, scene.width / 2
        vb = [np.array([[cx - 9, cy - 4.5], [cx + 9, cy - 4.5], [cx + 9, cy + 4.5], [cx - 9, cy + 4.5],
                        [cx - 9, cy - 4.5]]),
              np.array([[cx - 3, cy - 4.5], [cx - 3, cy + 4.5]]), np.array([[cx + 3, cy - 4.5], [cx + 3, cy + 4.5]])]
        for l in vb:
            cv2.polylines(tex, [to_px(l)], False, scene.other_lines_bgr, th, cv2.LINE_AA)
    court.configure(scene.length, scene.width, scene.spec)
    for l in court.court_polylines(n_arc=180):
        cv2.polylines(tex, [to_px(l)], False, scene.line_bgr, th, cv2.LINE_AA)
    return tex, margin


def _wall_texture(rng) -> np.ndarray:
    tex = np.full((400, 2000, 3), (205, 205, 200), np.uint8)
    for x in range(0, 2000, 40):  # wall bars / panels: vertical structure
        tex[60:300, x:x + 4] = (60, 90, 120)
    tex[300:310] = (120, 120, 120)
    tex[50:60] = (90, 90, 90)
    return np.clip(tex.astype(np.float32) + rng.normal(0, 4, tex.shape), 0, 255).astype(np.uint8)


def render_video(path: Path, scene: Scene, cam: CameraPath, seed: int = 0) -> SynthData:
    rng = np.random.default_rng(seed)
    w, h = scene.size
    mpp, margin = 0.02, 4.0
    floor, _ = _floor_texture(scene, mpp, margin, rng)
    wall = _wall_texture(rng)
    wall_mpp = (scene.length + 2 * margin + 20) / wall.shape[1]
    wall_y = scene.width + margin  # far wall plane
    # texture px -> court metres
    T_floor = np.array([[mpp, 0, -margin], [0, mpp, -margin], [0, 0, 1]])
    n = int(cam.duration_s * cam.fps)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), cam.fps, (w, h))
    ts = np.arange(n) / cam.fps
    Cs, Rs, fs, Hs = [], [], [], []
    for j, t in enumerate(ts):
        u = t / cam.duration_s
        ph = 0.5 - 0.5 * np.cos(2 * np.pi * t / 20.0)  # pan back and forth every 20 s
        pan = np.radians(cam.pan_range_deg[0] + (cam.pan_range_deg[1] - cam.pan_range_deg[0]) * ph)
        C = np.array(cam.C0, float) if cam.C1 is None else (1 - u) * np.array(cam.C0) + u * np.array(cam.C1)
        f = cam.f0 if cam.f1 is None else cam.f0 * (cam.f1 / cam.f0) ** u
        R = rotation(pan, np.radians(cam.tilt_deg), np.radians(cam.roll_deg))
        Hc = court_to_image_R(C, R, f, (w, h))
        img = np.full((h, w, 3), 30, np.uint8)
        # wall: plane y = wall_y, coords (x, z) -> world (x, wall_y, z)
        K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])
        Pw = K @ np.column_stack([R[:, 0], R[:, 2], R[:, 1] * wall_y - R @ C])  # (x, z, 1) -> image
        Tw = np.array([[wall_mpp, 0, -margin - 10], [0, -8.0 / wall.shape[0], 8.0], [0, 0, 1]])
        cv2.warpPerspective(wall, Pw @ Tw, (w, h), img, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
        # floor (only where it is in front of the camera)
        Hf = Hc @ T_floor
        cv2.warpPerspective(floor, Hf, (w, h), img, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
        img = np.clip(img.astype(np.float32) + rng.normal(0, 3, img.shape), 0, 255).astype(np.uint8)
        vw.write(img)
        Cs.append(C)
        Rs.append(R)
        fs.append(f)
        Hs.append(Hc)
    vw.release()
    motion: list = [None]
    for j in range(1, n):  # floor-plane motion, as pass 1 would measure it
        M = Hs[j] @ np.linalg.inv(Hs[j - 1])
        motion.append(M / M[2, 2])
    court.configure(28.0, 15.0, "FIBA")
    return SynthData(k=np.arange(n), frame=np.arange(n), t=ts, motion=motion, inliers=np.full(n, 500),
                     rows=np.zeros((0, 14)), signature={"process_fps": cam.fps}, C=np.array(Cs), R=Rs,
                     f=np.array(fs))


def reprojection_error(Hs_est: list, data: SynthData, scene: Scene) -> np.ndarray:
    """Median pixel distance between true and estimated court lines per sample (NaN = no estimate)."""
    court.configure(scene.length, scene.width, scene.spec)
    pts = np.vstack(court.court_polylines(n_arc=48))
    P = np.column_stack([pts, np.ones(len(pts))])
    w, h = scene.size
    out = np.full(len(Hs_est), np.nan)
    for j, H in enumerate(Hs_est):
        if H is None:
            continue
        Ht = court_to_image_R(data.C[j], data.R[j], data.f[j], (w, h))
        a, b = P @ Ht.T, P @ H.T
        ok = (a[:, 2] > 0) & (b[:, 2] > 0)
        pa, pb = a[ok, :2] / a[ok, 2:3], b[ok, :2] / b[ok, 2:3]
        vis = (pa[:, 0] > 0) & (pa[:, 0] < w) & (pa[:, 1] > 0) & (pa[:, 1] < h)
        if vis.sum() > 10:
            out[j] = float(np.median(np.linalg.norm(pa[vis] - pb[vis], axis=1)))
    return out
