"""Planar homography maths between image pixels and court metres.

Convention: ``H`` maps IMAGE pixels -> COURT metres (``court ~ H @ [u, v, 1]``).
Every H is normalised so that the homogeneous scale ``w`` is positive for
points on the visible floor (see :func:`normalize`). Image points above the
horizon then get ``w <= 0`` and are reported as invalid instead of being
projected to a mirrored, meaningless court position.
"""

from __future__ import annotations

import cv2
import numpy as np

_EPS = 1e-9


def _homog(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, float).reshape(-1, 2)
    return np.hstack([pts, np.ones((len(pts), 1))])


def apply(H: np.ndarray, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns (projected Nx2, w N). Projected points with w ~ 0 are NaN."""
    q = _homog(pts) @ np.asarray(H, float).T
    w = q[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        out = q[:, :2] / w[:, None]
    out[np.abs(w) < _EPS] = np.nan
    return out, w


def project(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Project points; points on the wrong side of the horizon (w <= 0) become NaN."""
    out, w = apply(H, pts)
    out[w <= _EPS] = np.nan
    return out


def normalize(H: np.ndarray, ref_pixel: tuple[float, float]) -> np.ndarray:
    """Scale H to unit Frobenius norm with w(ref_pixel) > 0.

    ``ref_pixel`` must be a pixel that sees the floor (bottom-centre of the frame
    works for any normal sideline/baseline camera).
    """
    H = np.asarray(H, float)
    H = H / np.linalg.norm(H)
    _, w = apply(H, np.array([ref_pixel]))
    return -H if w[0] < 0 else H


def floor_ref_pixel(frame_size: tuple[int, int]) -> tuple[float, float]:
    w, h = frame_size
    return (w / 2.0, h * 0.95)


def fit(image_pts: np.ndarray, court_pts: np.ndarray, frame_size: tuple[int, int],
        ransac_px: float | None = None, ref_pixel: tuple[float, float] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Fit image->court H from >= 4 correspondences. Returns (H, inlier mask)."""
    image_pts = np.asarray(image_pts, np.float64).reshape(-1, 2)
    court_pts = np.asarray(court_pts, np.float64).reshape(-1, 2)
    if len(image_pts) < 4:
        raise ValueError("Need at least 4 point correspondences.")
    if ransac_px is not None and len(image_pts) > 4:
        # RANSAC threshold is in destination units (metres); fit the inverse to threshold in pixels.
        Hc2i, mask = cv2.findHomography(court_pts, image_pts, cv2.RANSAC, ransac_px)
        if Hc2i is None:
            raise ValueError("Homography fit failed (degenerate points?).")
        H = np.linalg.inv(Hc2i)
    else:
        H, mask = cv2.findHomography(image_pts, court_pts, 0)
        if H is None:
            raise ValueError("Homography fit failed (degenerate points?).")
    mask = np.ones(len(image_pts), bool) if mask is None else mask.ravel().astype(bool)
    return normalize(H, ref_pixel or floor_ref_pixel(frame_size)), mask


def court_to_image(H: np.ndarray, court_pts: np.ndarray) -> np.ndarray:
    Hinv = np.linalg.inv(H)
    out, w = apply(Hinv, court_pts)
    # A court point is visible only if the image point maps back with positive w.
    _, w_back = apply(H, np.nan_to_num(out))
    out[(w_back <= _EPS) | ~np.isfinite(out).all(axis=1)] = np.nan
    return out


def reprojection_error_px(H: np.ndarray, image_pts: np.ndarray, court_pts: np.ndarray) -> np.ndarray:
    """Per-point pixel distance between clicked points and court points projected into the image."""
    proj = court_to_image(H, court_pts)
    return np.linalg.norm(proj - np.asarray(image_pts, float).reshape(-1, 2), axis=1)


def jacobian(H: np.ndarray, pixel: tuple[float, float], step: float = 0.5) -> np.ndarray:
    """2x2 derivative d(court metres)/d(image pixels) at ``pixel`` (central differences)."""
    u, v = pixel
    pts = np.array([[u + step, v], [u - step, v], [u, v + step], [u, v - step]])
    p = project(H, pts)
    return np.stack([(p[0] - p[1]) / (2 * step), (p[2] - p[3]) / (2 * step)], axis=1)


def project_each(Hs: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """``project`` with one homography per point set: Hs (N, 3, 3), pts (N, m, 2) -> (N, m, 2)."""
    pts = np.asarray(pts, float)
    q = np.concatenate([pts, np.ones(pts.shape[:-1] + (1,))], axis=-1) @ np.swapaxes(np.asarray(Hs, float), -1, -2)
    w = q[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        out = q[..., :2] / w[..., None]
    out[w <= _EPS] = np.nan  # covers |w| < eps of ``apply`` too
    return out


def jacobian_each(Hs: np.ndarray, pixels: np.ndarray, step: float = 0.5) -> np.ndarray:
    """``jacobian`` for many pixels, one homography each: Hs (N, 3, 3), pixels (N, 2) -> (N, 2, 2)."""
    pixels = np.asarray(pixels, float).reshape(-1, 2)
    u, v = pixels[:, 0], pixels[:, 1]
    pts = np.stack([np.c_[u + step, v], np.c_[u - step, v], np.c_[u, v + step], np.c_[u, v - step]], axis=1)
    p = project_each(Hs, pts)
    return np.stack([(p[:, 0] - p[:, 1]) / (2 * step), (p[:, 2] - p[:, 3]) / (2 * step)], axis=2)


def blend(Hs: list[np.ndarray], weights: list[float], frame_size: tuple[int, int]) -> np.ndarray | None:
    """Weighted average of homographies, done in court space on a grid of floor pixels."""
    w, h = frame_size
    gu, gv = np.meshgrid(np.linspace(0.05 * w, 0.95 * w, 5), np.linspace(0.45 * h, 0.95 * h, 4))
    grid = np.stack([gu.ravel(), gv.ravel()], axis=1)
    acc = np.zeros_like(grid)
    total = 0.0
    projected = [project(H, grid) for H in Hs]
    valid = np.all([np.isfinite(p).all(axis=1) for p in projected], axis=0)
    if valid.sum() < 4:
        return None
    for p, wt in zip(projected, weights):
        acc += wt * p
        total += wt
    acc /= total
    try:
        H, _ = fit(grid[valid], acc[valid], frame_size)
    except ValueError:
        return None
    return H


def is_sane(H: np.ndarray | None, frame_size: tuple[int, int]) -> bool:
    """Cheap degeneracy check: finite, well-conditioned, floor reference pixel projects."""
    if H is None or not np.isfinite(H).all():
        return False
    if np.linalg.cond(H) > 1e10:
        return False
    return bool(np.isfinite(project(H, np.array([floor_ref_pixel(frame_size)]))).all())


def camera_from_homography(H: np.ndarray, frame_size: tuple[int, int]) -> dict | None:
    """Pinhole camera behind an image->court floor homography (square pixels, principal point at the image centre).

    Returns {"P": 3x4 court(x, y, z up) -> image projection, "C": camera centre in court metres (z = height),
    "f": focal length px} or None when the homography does not determine a camera (e.g. a frontal view, or a
    fitted H that no real camera can produce). Used to reason about points ABOVE the floor (heads, airborne feet).
    """
    if H is None or not np.isfinite(H).all():
        return None
    try:
        G = np.linalg.inv(np.asarray(H, float))  # court -> image
    except np.linalg.LinAlgError:
        return None
    w, h = frame_size
    T = np.array([[1.0, 0, -w / 2], [0, 1.0, -h / 2], [0, 0, 1.0]])
    g = T @ G
    g = g / np.linalg.norm(g[:, :2])
    g1, g2 = g[:, 0], g[:, 1]
    # r1.r2 = 0 and |r1| = |r2| with K = diag(f, f, 1): linear in x = 1/f^2
    a = np.array([g1[0] * g2[0] + g1[1] * g2[1], g1[0] ** 2 + g1[1] ** 2 - g2[0] ** 2 - g2[1] ** 2])
    b = np.array([g1[2] * g2[2], g1[2] ** 2 - g2[2] ** 2])
    if a @ a < 1e-18:
        return None
    x = -(a @ b) / (a @ a)
    if not np.isfinite(x) or x <= 0:
        return None
    f = 1.0 / np.sqrt(x)
    M = np.diag([1 / f, 1 / f, 1.0]) @ g
    s = np.sqrt(np.linalg.norm(M[:, 0]) * np.linalg.norm(M[:, 1]))
    r1, r2, t = M[:, 0] / s, M[:, 1] / s, M[:, 2] / s
    # nearest true rotation to [r1 r2 r1xr2]
    U, _, Vt = np.linalg.svd(np.column_stack([r1, r2, np.cross(r1, r2)]))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        return None
    if t[2] < 0:  # the floor must be in front of the camera
        R[:, :2] *= -1
        t = -t
    C = -R.T @ t
    if C[2] < 0:  # z must point up (towards the camera side of the floor)
        R[:, 2] *= -1
        C = -R.T @ t
    if C[2] <= 0:
        return None
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])
    P = K @ np.column_stack([R, t])
    return {"P": P, "C": C, "f": float(f)}
