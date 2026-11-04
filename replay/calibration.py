"""Calibration keyframes and per-frame homography resolution.

A *keyframe* is a video time where the user clicked >= 4 known court points.
For a fixed camera one keyframe is enough. For a panning camera, the per-frame
camera motion measured in pass 1 carries each keyframe's homography forwards
and backwards in time; between two keyframes both estimates are blended so the
accumulated drift is pulled back to each keyframe. Frames that cannot be
reached (motion chain broken, too far from any keyframe) get no homography and
are reported as calibration failures - we never guess.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from replay import camera_model, court, homography
from replay.config import CalibrationConfig

KEYFRAME, PROPAGATED, NONE = "keyframe", "propagated", "none"


@dataclass
class Keyframe:
    time_s: float
    points: dict[str, tuple[float, float]]  # keypoint name -> image pixel
    H: np.ndarray | None = None
    reproj_px: float | None = None
    source: str = "manual"  # manual | snap (user-accepted line snap) | auto (automatic refinement)

    def fit(self, frame_size: tuple[int, int]) -> None:
        names = [n for n in self.points if n in court.KEYPOINTS]
        if len(names) < 4:
            raise ValueError("A keyframe needs at least 4 court points.")
        img = np.array([self.points[n] for n in names], float)
        crt = np.array([court.KEYPOINTS[n] for n in names], float)
        self.H, _ = homography.fit(img, crt, frame_size)
        err = homography.reprojection_error_px(self.H, img, crt)
        self.reproj_px = float(np.sqrt(np.nanmean(err**2)))


@dataclass
class CalibrationFile:
    video: str
    frame_size: tuple[int, int]
    fixed_camera: bool = False
    keyframes: list[Keyframe] = field(default_factory=list)
    court: dict | None = None  # {"length", "width", "spec"} - measured by the automatic calibration
    auto: dict | None = None  # automatic calibration (autocourt.py): camera position + anchor views

    def apply_court(self) -> None:
        """Make the whole program use this video's court size / marking standard."""
        c = self.court or {}
        court.configure(c.get("length", court.DEFAULT_LENGTH), c.get("width", court.DEFAULT_WIDTH),
                        c.get("spec", "FIBA"))

    @property
    def calibrated(self) -> bool:
        return bool(self.keyframes) or bool(self.auto and (any(a.get("good") for a in self.auto.get("anchors", []))
                                                         or self.auto.get("frames")))

    def save(self, path: str | Path) -> None:
        data = {
            "video": self.video,
            "frame_size": list(self.frame_size),
            "fixed_camera": self.fixed_camera,
            "coordinate_system": "metres; origin = left baseline x near sideline (as seen from camera); "
                                 "x along the sideline 0..length, y along the baseline 0..width",
            "court": self.court,
            "auto": self.auto,
            "keyframes": [
                {"time_s": round(k.time_s, 4),
                 "points": {n: [round(p[0], 2), round(p[1], 2)] for n, p in k.points.items()},
                 "source": k.source,
                 "H_image_to_court": None if k.H is None else k.H.tolist(),
                 "reprojection_px": k.reproj_px}
                for k in sorted(self.keyframes, key=lambda k: k.time_s)
            ],
        }
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CalibrationFile":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        cf = cls(video=data.get("video", ""), frame_size=tuple(data["frame_size"]),  # type: ignore[arg-type]
                 fixed_camera=bool(data.get("fixed_camera", False)), court=data.get("court"),
                 auto=data.get("auto"))
        cf.apply_court()  # keypoints below depend on the court size
        for k in data.get("keyframes", []):
            kf = Keyframe(time_s=float(k["time_s"]), points={n: tuple(p) for n, p in k["points"].items()},
                          source=str(k.get("source", "manual")))
            if len(kf.points) >= 4:
                kf.fit(cf.frame_size)  # always refit from the points so the file stays the source of truth
            cf.keyframes.append(kf)
        cf.keyframes.sort(key=lambda k: k.time_s)
        return cf

    def keyframe_at(self, t: float, tol: float = 1e-3) -> Keyframe | None:
        for k in self.keyframes:
            if abs(k.time_s - t) <= tol:
                return k
        return None


@dataclass
class ResolvedCalibration:
    H: list[np.ndarray | None]
    quality: list[str]  # KEYFRAME | PROPAGATED | NONE
    distance_s: np.ndarray  # time to nearest usable keyframe (inf if none)
    warnings: list[str]
    point_errors: dict = field(default_factory=dict)  # (keyframe time, point name) -> px error in the joint fit
    keyframe_errors: dict = field(default_factory=dict)  # keyframe time -> rms px error in the joint fit


def resolve(times: np.ndarray, motions: list[np.ndarray | None], calib: CalibrationFile,
            cfg: CalibrationConfig) -> ResolvedCalibration:
    """Per-sample homographies from keyframes + frame-to-frame motion.

    ``motions[i]`` maps sample i-1 pixels -> sample i pixels (None = unknown / broken).
    """
    n = len(times)
    size = calib.frame_size
    ref = homography.floor_ref_pixel(size)
    warnings: list[str] = []
    point_errors: dict[tuple[float, str], float] = {}
    kf_errors: dict[float, float] = {}
    usable = [k for k in calib.keyframes if len(k.points) >= 1]

    if calib.auto and not calib.fixed_camera:
        return _resolve_auto(times, motions, calib, cfg, usable)

    if calib.fixed_camera:
        # every keyframe shares the same pixel space: fit all clicked points together
        if not usable:
            return ResolvedCalibration([None] * n, [NONE] * n, np.full(n, np.inf), ["no usable keyframe"])
        H = _joint_fit([(np.eye(3), k) for k in usable], ref, size, cfg, warnings, point_errors, kf_errors)
        if H is None:
            return ResolvedCalibration([None] * n, [NONE] * n, np.full(n, np.inf), warnings, point_errors, kf_errors)
        return ResolvedCalibration([H] * n, [KEYFRAME] * n, np.zeros(n), warnings, point_errors, kf_errors)

    # Camera motion links every frame to the first frame of its unbroken segment: G[i] maps frame-i pixels
    # into that reference frame (an image mosaic). All keyframes of a segment are then fitted *jointly*,
    # so clicks on different parts of the court reinforce each other instead of each keyframe being
    # extrapolated on its own.
    G, seg = _segment_transforms(motions, n)
    kf_index: dict[int, Keyframe] = {}
    for kf in usable:
        if n == 0:
            break
        i = int(np.argmin(np.abs(times - kf.time_s)))
        if abs(times[i] - kf.time_s) > cfg.keyframe_snap_s:
            warnings.append(f"keyframe {kf.time_s:.2f}s is outside the processed range; ignored")
            continue
        kf_index[i] = kf

    Hs: list[np.ndarray | None] = [None] * n
    quality = [NONE] * n
    dist = np.full(n, np.inf)
    for s in np.unique(seg):
        members = np.where(seg == s)[0]
        kfs = [(G[i], kf_index[i], i) for i in members if i in kf_index]
        if not kfs:
            continue
        cam: dict = {}
        Hg = _joint_fit([(g, k) for g, k, _ in kfs], _apply(G[kfs[0][2]], ref), size, cfg, warnings,
                        point_errors, kf_errors, cam)
        if Hg is None:
            continue
        kf_times = np.array([times[i] for _, _, i in kfs])
        H_ref_c2i = np.linalg.inv(Hg)
        C = cam["C"]
        view = cam["view"]
        chain: dict[int, tuple] = {}
        live = [j for j in members if float(np.min(np.abs(kf_times - times[j]))) <= cfg.max_propagation_s]
        # The camera model is fitted on every STRIDE-th frame (plus keyframes); camera angles in between are
        # interpolated - camera motion is smooth, and this keeps pass 2 fast on full games.
        STRIDE, MAX_SHIFT_PX = 5, 25.0
        anchors_set = {i for _, _, i in kfs if i in set(live)}
        if live:
            anchors_set |= {live[0], live[-1]}
            centre = np.array([size[0] / 2, size[1] / 2])
            moved, since = 0.0, 0
            for j in live[1:]:
                M = motions[j] if motions[j] is not None else np.eye(3)
                moved += float(np.linalg.norm(np.array(_apply(M, tuple(centre))) - centre))
                since += 1
                if moved > MAX_SHIFT_PX or since >= STRIDE:  # adaptive: dense anchors during fast pans
                    anchors_set.add(j)
                    moved, since = 0.0, 0
        anchors = sorted(anchors_set)
        fitted: dict[int, tuple] = {}
        for j in anchors:
            H_c2i, view, ok = _on_camera_ex(C, np.linalg.inv(G[j]) @ H_ref_c2i, size, view)
            fitted[j] = (H_c2i, view, ok)
        a_idx = np.array(anchors)
        for j in live:
            d = float(np.min(np.abs(kf_times - times[j])))
            if j in fitted:
                chain[j] = (*fitted[j], d)
                continue
            k = int(np.searchsorted(a_idx, j))
            lo, hi = fitted[a_idx[k - 1]], fitted[a_idx[k]]
            H_chain = np.linalg.inv(G[j]) @ H_ref_c2i
            if lo[2] and hi[2]:
                u = (j - a_idx[k - 1]) / (a_idx[k] - a_idx[k - 1])
                v0, v1 = lo[1], hi[1]
                v = camera_model.PTZ(v0.pan + (v1.pan - v0.pan) * u, v0.tilt + (v1.tilt - v0.tilt) * u,
                                     v0.roll + (v1.roll - v0.roll) * u, v0.f * (v1.f / v0.f) ** u)
                chain[j] = (camera_model.court_to_image(C, v, size), v, True, d)
            else:
                chain[j] = (H_chain, lo[1], False, d)
        # Local corrections: each keyframe's own view (same camera position) vs the joint/chain view at that
        # moment; the difference is blended linearly in time, so every keyframe fixes drift around it.
        corr_t, corr_v = [], []
        for _g, kf, i in kfs:
            names = [n for n in kf.points if n in court.KEYPOINTS]
            if len(names) < 3 or i not in chain or not chain[i][2]:
                continue
            own = camera_model.fit_camera([(np.array([court.KEYPOINTS[n] for n in names]),
                                            np.array([kf.points[n] for n in names]))], size, C_fixed=C,
                                          init_views=[chain[i][1]])
            if own is None or own.rms_px > cfg.max_reprojection_px:
                continue
            cv = chain[i][1]
            o = own.views[0]
            w = _correction_weight(kf, len(names))
            corr_t.append(times[i])
            corr_v.append([w * (o.pan - cv.pan), w * (o.tilt - cv.tilt), w * (o.roll - cv.roll), w * np.log(o.f / cv.f)])
        order = np.argsort(corr_t)
        corr_t, corr_v = np.array(corr_t)[order], np.array(corr_v).reshape(-1, 4)[order]
        for j, (H_c2i, view, ok, d) in chain.items():
            if ok and len(corr_t):
                dv = [np.interp(times[j], corr_t, corr_v[:, c]) for c in range(4)]
                v2 = camera_model.PTZ(view.pan + dv[0], view.tilt + dv[1], view.roll + dv[2], view.f * np.exp(dv[3]))
                H_c2i = camera_model.court_to_image(C, v2, size)
            H = homography.normalize(np.linalg.inv(H_c2i), ref)
            if homography.is_sane(H, size):
                Hs[j], dist[j] = H, d
                quality[j] = KEYFRAME if j in kf_index else PROPAGATED
    return ResolvedCalibration(Hs, quality, dist, warnings, point_errors, kf_errors)


def _resolve_auto(times, motions, calib: CalibrationFile, cfg: CalibrationConfig, usable) -> ResolvedCalibration:
    """Automatic calibration (autocourt.py); manual keyframes, if any, act as extra trusted anchors."""
    from replay import autocourt

    size = calib.frame_size
    ref = homography.floor_ref_pixel(size)
    warnings: list[str] = []
    C = np.asarray(calib.auto["C"], float)
    extra = []
    for kf in usable:
        names = [nm for nm in kf.points if nm in court.KEYPOINTS]
        if len(names) < 3:
            continue
        fit = camera_model.fit_camera([(np.array([court.KEYPOINTS[nm] for nm in names]),
                                        np.array([kf.points[nm] for nm in names]))], size, C_fixed=C)
        if fit is None or fit.rms_px > cfg.max_reprojection_px:
            warnings.append(f"keyframe {kf.time_s:.1f}s does not fit the automatic camera; ignored")
            continue
        v = fit.views[0]
        extra.append((kf.time_s, camera_model.rotation(v.pan, v.tilt, v.roll), v.f))
    Hc, d = autocourt.resolve_auto(np.asarray(times, float), motions, calib.auto, size, extra,
                                   max_gap_s=cfg.max_propagation_s)
    n = len(times)
    Hs: list[np.ndarray | None] = [None] * n
    quality = [NONE] * n
    for i, H_c2i in enumerate(Hc):
        if H_c2i is None:
            continue
        H = homography.normalize(np.linalg.inv(H_c2i), ref)
        if homography.is_sane(H, size):
            Hs[i] = H
            quality[i] = KEYFRAME if d[i] < 0.05 else PROPAGATED
    if not any(h is not None for h in Hs):
        warnings.append("automatic calibration produced no usable frames")
    return ResolvedCalibration(Hs, quality, d, warnings)


def _apply(H: np.ndarray, p: tuple[float, float]) -> tuple[float, float]:
    q = H @ np.array([p[0], p[1], 1.0])
    return (float(q[0] / q[2]), float(q[1] / q[2]))


def _correction_weight(kf: Keyframe, n_points: int) -> float:
    """How much a single keyframe may bend the calibration around it.

    A few hand-clicked points are noisy, so they only partly override the joint fit; line-snapped keyframes
    are measured against many line pixels and are trusted almost fully.
    """
    if kf.source in ("snap", "auto"):
        return 0.9
    return n_points / (n_points + 4.0)


def _segment_transforms(motions: list[np.ndarray | None], n: int) -> tuple[list[np.ndarray], np.ndarray]:
    """G[i]: frame-i pixels -> pixels of the first frame of i's unbroken motion segment."""
    G: list[np.ndarray] = []
    seg = np.zeros(n, int)
    s = -1
    for i in range(n):
        if i == 0 or motions[i] is None:
            s += 1
            G.append(np.eye(3))
        else:
            g = G[i - 1] @ np.linalg.inv(motions[i])
            G.append(g / g[2, 2])
        seg[i] = s
    return G, seg


def _joint_fit(items: list[tuple[np.ndarray, Keyframe]], ref_pixel: tuple[float, float], size: tuple[int, int],
               cfg: CalibrationConfig, warnings: list[str], point_errors: dict, kf_errors: dict,
               camera_out: dict | None = None) -> np.ndarray | None:
    """Fit the clicked points of several keyframes with ONE physical camera (see camera_model.py).

    items: (G, keyframe) with G mapping that keyframe's pixels into the common reference view.
    Returns the reference-pixels -> court homography; camera_out receives {"C", "view"} for per-frame use.
    """
    img, crt, owner = [], [], []
    for g, kf in items:
        for name, p in kf.points.items():
            if name in court.KEYPOINTS:
                img.append(_apply(g, p))
                crt.append(court.KEYPOINTS[name])
                owner.append((kf, name, g))
    if len(img) < 3:
        warnings.append("fewer than 3 calibration points in a camera segment")
        return None
    img_a, crt_a = np.array(img), np.array(crt)
    fit = camera_model.fit_camera([(crt_a, img_a)], size)
    if fit is None:
        warnings.append("calibration fit failed")
        return None
    view = fit.views[0]
    H_c2i = camera_model.court_to_image(fit.C, view, size)
    Hg = homography.normalize(np.linalg.inv(H_c2i), ref_pixel)
    if camera_out is not None:
        camera_out.update(C=fit.C, view=view)
    per_kf: dict[float, list[float]] = {}
    for (kf, name, g), c, e in zip(owner, crt_a, fit.per_point_px[0]):
        # error in the keyframe's own pixels (G is ~a rotation, so reference-pixel error is close enough,
        # but convert properly for honesty)
        H_frame = homography.normalize(Hg @ g, homography.floor_ref_pixel(size))
        e = float(homography.reprojection_error_px(H_frame, np.array([kf.points[name]]), c[None])[0])
        point_errors[(round(kf.time_s, 3), name)] = e
        per_kf.setdefault(round(kf.time_s, 3), []).append(e)
        if not np.isfinite(e) or e > cfg.max_reprojection_px:
            warnings.append(f"point {name} in keyframe {kf.time_s:.1f}s is {e:.0f} px off in the combined "
                            "calibration - probably misplaced")
    for t, errs in per_kf.items():
        kf_errors[t] = float(np.sqrt(np.nanmean(np.square(errs))))
    return Hg


def _on_camera(C: np.ndarray, H_c2i: np.ndarray, size: tuple[int, int], init: "camera_model.PTZ"):
    """Snap an arbitrary court->image homography onto the physical camera (C fixed). Returns (H_c2i, view)."""
    H, view, _ok = _on_camera_ex(C, H_c2i, size, init)
    return H, view


def _on_camera_ex(C: np.ndarray, H_c2i: np.ndarray, size: tuple[int, int], init: "camera_model.PTZ"):
    """Like _on_camera, plus whether the physical-camera view was accepted (False = propagated H kept)."""
    view = camera_model.fit_ptz_to_homography(C, H_c2i, size, init)
    if view is None:
        return H_c2i, camera_model.init_from_homography(C, H_c2i, size, init.f) or init, False
    H_cam = camera_model.court_to_image(C, view, size)
    # Safety: the model view must reproduce the motion-propagated calibration where the court is visible;
    # otherwise the optimiser found a wrong solution and the propagated calibration is kept as is.
    w, h = size
    gx, gy = np.meshgrid(np.linspace(0, 28, 15), np.linspace(0, 15, 9))
    grid = np.column_stack([gx.ravel(), gy.ravel(), np.ones(gx.size)])
    a, b = grid @ H_c2i.T, grid @ H_cam.T
    ok = (a[:, 2] > 0) & (b[:, 2] > 0)
    pa, pb = a[ok, :2] / a[ok, 2:3], b[ok, :2] / b[ok, 2:3]
    inside = (pa[:, 0] > 0) & (pa[:, 0] < w) & (pa[:, 1] > 0) & (pa[:, 1] < h)
    if inside.sum() < 4 or np.median(np.linalg.norm(pa[inside] - pb[inside], axis=1)) > 15.0:
        return H_c2i, camera_model.init_from_homography(C, H_c2i, size, init.f) or init, False
    return H_cam, view, True


class FrameSolver:
    """Calibration for single frames, for interactive use (calibration UI).

    Caches the camera-motion chain once; each query refits only the segment containing the frame.
    """

    def __init__(self, times: np.ndarray, motions: list[np.ndarray | None], frame_size: tuple[int, int],
                 cfg: CalibrationConfig):
        self.times = np.asarray(times, float)
        self.size = frame_size
        self.cfg = cfg
        self.G, self.seg = _segment_transforms(motions, len(self.times))

    def index(self, t: float) -> int | None:
        if len(self.times) == 0:
            return None
        i = int(np.argmin(np.abs(self.times - t)))
        return i if abs(self.times[i] - t) <= self.cfg.keyframe_snap_s else None

    def solve(self, keyframes: list[Keyframe], t: float) -> tuple[np.ndarray | None, dict]:
        """Homography for the frame at time t from all keyframes in its camera segment."""
        info: dict = {"keyframes_used": 0, "warnings": [], "point_errors": {}, "keyframe_errors": {}}
        i = self.index(t)
        if i is None:
            return None, info
        members = []
        for kf in keyframes:
            j = self.index(kf.time_s)
            if j is not None and self.seg[j] == self.seg[i] and kf.points:
                members.append((self.G[j], kf))
        info["keyframes_used"] = len(members)
        if not members:
            return None, info
        ref = homography.floor_ref_pixel(self.size)
        cam: dict = {}
        Hg = _joint_fit(members, _apply(members[0][0], ref), self.size, self.cfg, info["warnings"],
                        info["point_errors"], info["keyframe_errors"], cam)
        if Hg is None:
            return None, info
        C = cam["C"]
        H_ref = np.linalg.inv(Hg)
        H_c2i, view, ok = _on_camera_ex(C, np.linalg.inv(self.G[i]) @ H_ref, self.size, cam["view"])
        if ok:  # same local keyframe corrections as resolve(), so the preview matches the built replay
            corr = []
            for kf in keyframes:
                j = self.index(kf.time_s)
                names = [n for n in kf.points if n in court.KEYPOINTS]
                if j is None or self.seg[j] != self.seg[i] or len(names) < 3:
                    continue
                _, cv, okj = _on_camera_ex(C, np.linalg.inv(self.G[j]) @ H_ref, self.size, cam["view"])
                if not okj:
                    continue
                own = camera_model.fit_camera([(np.array([court.KEYPOINTS[n] for n in names]),
                                                np.array([kf.points[n] for n in names]))], self.size,
                                              C_fixed=C, init_views=[cv])
                if own is None or own.rms_px > self.cfg.max_reprojection_px:
                    continue
                o = own.views[0]
                w = _correction_weight(kf, len(names))
                corr.append((self.times[j], [w * (o.pan - cv.pan), w * (o.tilt - cv.tilt), w * (o.roll - cv.roll),
                                             w * np.log(o.f / cv.f)]))
            if corr:
                corr.sort(key=lambda c: c[0])
                ct = np.array([c[0] for c in corr])
                cvv = np.array([c[1] for c in corr])
                dv = [np.interp(self.times[i], ct, cvv[:, c]) for c in range(4)]
                view = camera_model.PTZ(view.pan + dv[0], view.tilt + dv[1], view.roll + dv[2], view.f * np.exp(dv[3]))
                H_c2i = camera_model.court_to_image(C, view, self.size)
        info["camera"] = {"C": C, "view": view}
        H = homography.normalize(np.linalg.inv(H_c2i), ref)
        return (H if homography.is_sane(H, self.size) else None), info
