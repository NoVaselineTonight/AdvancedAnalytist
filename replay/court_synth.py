"""Synthetic basketball-court images with exact labels, for training the court detector (replay/courtnet.py).

Every sample randomises the court (FIBA / NBA / NCAA / old FIBA markings, non-standard sizes), the floor (wood
grain, plain sports floor, painted keys / outer areas, logos, other sports' lines), the line colour, the camera
(high broadcast camera with pan and 15-70 deg zoom, low sideline gym camera, corner camera), people (occlusion),
a scorebug overlay, motion blur, lighting / colour cast and compression noise.

Labels (at the network's output resolution): court-line mask and one Gaussian heatmap per court keypoint
(court.KEYPOINTS order, camera always on the near side y < 0, so names are unambiguous).
Pure numpy + OpenCV (own code).
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from replay import court
from replay.camera_model import rotation

IN_W, IN_H = 320, 192  # network input
OUT_W, OUT_H = 160, 96  # network output (stride 2)
SPECS = ("FIBA", "FIBA", "NBA", "NCAA", "FIBA_OLD")


def kp_names() -> list[str]:
    court.configure(28.0, 15.0, "FIBA")
    return list(court.KEYPOINTS)


def K_of(f: float, w: int, h: int) -> np.ndarray:
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])


def look_at(C: np.ndarray, P: np.ndarray, roll: float) -> np.ndarray:
    d = P - C
    pan = math.atan2(d[1], d[0])
    tilt = math.atan2(-d[2], math.hypot(d[0], d[1]))
    return rotation(pan, tilt, roll)


def court_to_image(C, R, f, w, h) -> np.ndarray:
    return K_of(f, w, h) @ np.column_stack([R[:, 0], R[:, 1], -R @ C])


def _sample_camera(rng, L, W, p_tele: float = 0.0):
    if rng.random() < p_tele:
        # TV main camera: high, at centre court, looking 10-25 deg down, telephoto zoom following the play
        C = np.array([L / 2 + rng.uniform(-2, 2), -rng.uniform(10, 30), 0.0])
        P = np.array([rng.uniform(0.05 * L, 0.95 * L), rng.uniform(0.25 * W, 0.75 * W), 0.0])
        dist = math.hypot(P[0] - C[0], P[1] - C[1])
        C[2] = dist * math.tan(math.radians(rng.uniform(10, 25)))
        R = look_at(C, P, math.radians(rng.uniform(-1.5, 1.5)))
        return C, R, rng.uniform(10, 40), "tele"
    kind = rng.choice(["broadcast", "gym", "corner"], p=[0.4, 0.35, 0.25])
    if kind == "broadcast":
        C = np.array([L / 2 + rng.uniform(-4, 4), -rng.uniform(8, 25), rng.uniform(5, 15)])
        hfov = rng.uniform(15, 70)
    elif kind == "gym":
        C = np.array([rng.uniform(0.25 * L, 0.75 * L), -rng.uniform(1, 7), rng.uniform(1.2, 3.5)])
        hfov = rng.uniform(40, 80)
    else:
        x = rng.uniform(-6, 2) if rng.random() < 0.5 else rng.uniform(L - 2, L + 6)
        C = np.array([x, -rng.uniform(1, 8), rng.uniform(1.5, 6)])
        hfov = rng.uniform(35, 75)
    P = np.array([rng.uniform(0, L), rng.uniform(0.15 * W, 0.85 * W), 0.0])
    R = look_at(C, P, math.radians(rng.uniform(-3, 3)))
    return C, R, hfov, kind


def _rand_color(rng, lo=0, hi=255):
    return tuple(int(v) for v in rng.integers(lo, hi, 3))


def _floor_texture(rng, L, W, mpp, margin, tv: bool = False):
    """BGR floor texture covering [-margin, L+margin] x [-margin, W+margin] metres + the court line mask."""
    Wt, Ht = int((L + 2 * margin) / mpp), int((W + 2 * margin) / mpp)
    r = rng.random() * 0.5 if tv and rng.random() < 0.8 else rng.random()  # TV courts: mostly wood
    if r < 0.5:  # wood
        base = np.array([rng.uniform(60, 110), rng.uniform(110, 170), rng.uniform(160, 230)])
    elif r < 0.8:  # plain sports floor (grey / blue / green / red-brown)
        base = np.array(_rand_color(rng, 40, 220), float)
    else:
        base = np.array(_rand_color(rng, 20, 120), float)
    tex = np.empty((Ht, Wt, 3), np.float32)
    tex[:] = base
    if r < 0.5 or rng.random() < 0.3:  # planks / grain along x
        rows = rng.normal(0, rng.uniform(2, 12), (Ht, 1, 1)).astype(np.float32)
        tex += rows
        tex += rng.normal(0, rng.uniform(1, 8), (Ht, Wt, 1)).astype(np.float32)

    def px(p):
        return np.int32(np.round((np.asarray(p, float) + margin) / mpp)).reshape(-1, 1, 2)

    # painted areas: outside the court, keys, centre circle, logos
    if rng.random() < (0.75 if tv else 0.35):  # painted area outside the lines (dark blue / black on TV courts)
        outer = np.array(_rand_color(rng, 10, 90) if tv and rng.random() < 0.7 else _rand_color(rng, 10, 200), float)
        m = np.ones((Ht, Wt), np.uint8)
        cv2.rectangle(m, tuple(px((0, 0))[0, 0]), tuple(px((L, W))[0, 0]), 0, -1)
        tex[m > 0] = outer
    if rng.random() < 0.4:
        kc = _rand_color(rng)
        for side in (0, 1):
            x0, s = (0, 1) if side == 0 else (L, -1)
            poly = [(x0, court.MID_Y - court.PAINT_HALF_WIDTH), (x0 + s * court.FT_LINE_X, court.MID_Y - court.PAINT_HALF_WIDTH),
                    (x0 + s * court.FT_LINE_X, court.MID_Y + court.PAINT_HALF_WIDTH), (x0, court.MID_Y + court.PAINT_HALF_WIDTH)]
            cv2.fillPoly(tex, [px(poly)], kc)
    if tv and rng.random() < 0.8:  # big centre-court logo + baseline/sideline lettering
        cc = px((L / 2, W / 2))[0, 0]
        r_ = int(rng.uniform(2.0, 4.5) / mpp)
        cv2.ellipse(tex, tuple(cc), (r_, int(r_ * rng.uniform(0.5, 1.0))), 0, 0, 360, _rand_color(rng), -1)
        for _ in range(rng.integers(3, 10)):
            o = (rng.integers(-r_, r_), rng.integers(-r_ // 2, r_ // 2))
            cv2.rectangle(tex, (cc[0] + o[0], cc[1] + o[1]), (cc[0] + o[0] + int(rng.integers(10, 60)),
                                                               cc[1] + o[1] + int(rng.integers(10, 40))), _rand_color(rng), -1)
        for _ in range(rng.integers(0, 3)):  # big lettering in the painted margins
            x = rng.uniform(0, L)
            y = rng.choice([-1.2, W + 1.2])
            p0 = px((x, y))[0, 0]
            cv2.rectangle(tex, tuple(p0 - np.array([int(3 / mpp), int(0.6 / mpp)])),
                          tuple(p0 + np.array([int(3 / mpp), int(0.6 / mpp)])), _rand_color(rng, 150, 255), -1)
    for _ in range(rng.integers(0, 4)):  # logos / floor graphics
        c = (rng.uniform(0, L), rng.uniform(0, W))
        ax = (int(rng.uniform(1, 4) / mpp), int(rng.uniform(0.5, 2.5) / mpp))
        col = _rand_color(rng)
        if rng.random() < 0.5:
            cv2.ellipse(tex, tuple(px(c)[0, 0]), ax, rng.uniform(0, 180), 0, 360, col, -1)
        else:
            p0 = px(c)[0, 0]
            cv2.rectangle(tex, tuple(p0 - np.array(ax)), tuple(p0 + np.array(ax)), col, -1)
    for _ in range(rng.integers(0, 3)):  # other sports' lines
        col = _rand_color(rng)
        cx, cy, hw, hh = rng.uniform(0.3 * L, 0.7 * L), W / 2, rng.uniform(4, 13), rng.uniform(3, 7.5)
        th = max(1, int(0.05 / mpp))
        cv2.rectangle(tex, tuple(px((cx - hw, cy - hh))[0, 0]), tuple(px((cx + hw, cy + hh))[0, 0]), col, th)
        cv2.line(tex, tuple(px((cx, cy - hh))[0, 0]), tuple(px((cx, cy + hh))[0, 0]), col, th)
    # basketball lines
    lc = rng.random()
    line_col = (240, 240, 240) if lc < 0.55 else (25, 25, 25) if lc < 0.8 else _rand_color(rng)
    th = max(1, int(round(rng.uniform(0.04, 0.08) / mpp)))
    for l in court.court_polylines(n_arc=96):
        if rng.random() < 0.03:
            continue  # a worn / hidden line
        cv2.polylines(tex, [px(l)], False, line_col, th, cv2.LINE_AA)
    return np.clip(tex, 0, 255).astype(np.uint8)


def _depth(H, P, C, R):
    """Camera depth (m) of court points P (N,2): from the camera (C, R) or, when C is None, from H itself (H must
    keep the K @ [r1 r2 -RC] scale, as court_to_image / autocourt.resolve_auto give it: third row = depth)."""
    if C is None:
        return (np.column_stack([P, np.ones(len(P))]) @ np.asarray(H, float).T)[:, 2]
    return (R @ (np.column_stack([P, np.zeros(len(P))]) - C).T).T[:, 2]


def _project_lines(H, w, h, C=None, R=None):
    """Court polylines in image pixels, only the parts in front of the camera (C, R; or H's own depth)."""
    out = []
    for l in court.court_polylines(n_arc=64):
        dense = []
        for a, b in zip(l[:-1], l[1:]):
            n = max(2, int(np.linalg.norm(b - a) / 0.2))
            dense.append(a + (b - a) * np.linspace(0, 1, n, endpoint=False)[:, None])
        dense.append(l[-1:])
        P = np.vstack(dense)
        ok = _depth(H, P, C, R) > 0.3
        q = np.column_stack([P, np.ones(len(P))]) @ H.T
        q = q[:, :2] / np.where(np.abs(q[:, 2:3]) < 1e-9, 1e-9, q[:, 2:3])
        run = []
        for j in range(len(P)):
            if ok[j] and np.all(np.abs(q[j]) < 4 * max(w, h)):
                run.append(q[j])
            elif len(run) > 1:
                out.append(np.array(run)); run = []
            else:
                run = []
        if len(run) > 1:
            out.append(np.array(run))
    return out


def render(rng: np.random.Generator, p_tele: float = 0.0):
    """One training sample: (image uint8 IN_HxIN_Wx3 BGR, line mask float32 OUT_HxOUT_W, heatmaps K x OUT_H x OUT_W,
    meta dict)."""
    spec = SPECS[rng.integers(len(SPECS))]
    base = court.STANDARDS[spec]
    L = float(np.clip(base["length"] * rng.uniform(0.85, 1.03), 20, 30))
    W = float(np.clip(base["width"] * rng.uniform(0.88, 1.03), 11, 16))
    court.configure(L, W, spec)
    names = list(court.KEYPOINTS)
    for _try in range(20):
        C, R, hfov, kind = _sample_camera(rng, L, W, p_tele)
        f = (IN_W / 2) / math.tan(math.radians(hfov) / 2)
        H = court_to_image(C, R, f, IN_W, IN_H)
        kp = np.array([court.KEYPOINTS[n] for n in names])
        cam = (R @ (np.column_stack([kp, np.zeros(len(kp))]) - C).T).T
        q = np.column_stack([kp, np.ones(len(kp))]) @ H.T
        uv = q[:, :2] / q[:, 2:3]
        vis = (cam[:, 2] > 0.3) & (uv[:, 0] >= 0) & (uv[:, 0] < IN_W) & (uv[:, 1] >= 0) & (uv[:, 1] < IN_H)
        if vis.sum() >= 4:
            break
    tv = kind in ("tele", "broadcast")
    # background: stands / walls; TV arenas: a crowd (dense multi-coloured speckle)
    img = np.empty((IN_H, IN_W, 3), np.uint8)
    img[:] = _rand_color(rng, 10, 160)
    if tv and rng.random() < 0.8:
        crowd = rng.integers(0, 255, (IN_H // 3 + 1, IN_W // 3 + 1, 3)).astype(np.float32)
        crowd = crowd * rng.uniform(0.3, 0.9) + np.array(_rand_color(rng, 0, 120)) * rng.uniform(0.2, 0.6)
        img = cv2.resize(np.clip(crowd, 0, 255).astype(np.uint8), (IN_W, IN_H), interpolation=cv2.INTER_NEAREST)
    else:
        for _ in range(rng.integers(20, 80)):
            x, y = int(rng.integers(0, IN_W)), int(rng.integers(0, IN_H))
            cv2.rectangle(img, (x, y), (x + int(rng.integers(2, 40)), y + int(rng.integers(2, 25))), _rand_color(rng), -1)
    if tv and rng.random() < 0.7:  # LED ad boards behind the far sideline (vertical plane at y = W + gap)
        gap, hb = rng.uniform(1.5, 4.0), rng.uniform(0.8, 1.2)
        xs = np.linspace(-6, L + 6, 13)
        for x0, x1 in zip(xs[:-1], xs[1:]):
            quad = np.array([[x0, W + gap, 0], [x1, W + gap, 0], [x1, W + gap, hb], [x0, W + gap, hb]])
            cam = (R @ (quad - C).T).T
            if (cam[:, 2] <= 0.3).any():
                continue
            q = (K_of(f, IN_W, IN_H) @ cam.T).T
            uv = q[:, :2] / q[:, 2:3]
            if np.abs(uv).max() > 5 * IN_W:
                continue
            cv2.fillPoly(img, [np.int32(uv).reshape(-1, 1, 2)], _rand_color(rng, 60, 255))
    # floor (texture limited to what is in front of the camera)
    mpp, margin = 0.03, 7.0
    tex = _floor_texture(rng, L, W, mpp, margin, tv=tv)
    T = np.array([[mpp, 0, -margin], [0, mpp, -margin], [0, 0, 1.0]])
    Ht, Wt = tex.shape[:2]
    yy, xx = np.mgrid[0:Ht:4, 0:Wt:4]
    P = np.column_stack([xx.ravel() * mpp - margin, yy.ravel() * mpp - margin, np.zeros(xx.size)])
    depth = (R @ (P - C).T)[2].reshape(xx.shape)
    alpha = cv2.resize((depth > 0.5).astype(np.uint8) * 255, (Wt, Ht), interpolation=cv2.INTER_NEAREST)
    texa = np.dstack([tex, alpha])
    warped = cv2.warpPerspective(texa, H @ T, (IN_W, IN_H), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(0, 0, 0, 0))
    a = warped[..., 3:4].astype(np.float32) / 255
    img = (img * (1 - a) + warped[..., :3] * a).astype(np.uint8)
    # people
    for _ in range(rng.integers(0, 14)):
        p = np.array([rng.uniform(-1, L + 1), rng.uniform(-1, W + 1), 0.0])
        if (R @ (p - C))[2] < 1.0:
            continue
        top = p + np.array([0, 0, rng.uniform(1.6, 2.1)])
        q = [court_to_image(C, R, f, IN_W, IN_H) @ np.array([p[0], p[1], 1.0])]
        b = q[0][:2] / q[0][2]
        tc = K_of(f, IN_W, IN_H) @ (R @ (top - C))
        t2 = tc[:2] / tc[2]
        hgt = max(2.0, b[1] - t2[1])
        wid = max(1, int(hgt * rng.uniform(0.22, 0.35)))
        cv2.rectangle(img, (int(b[0] - wid / 2), int(t2[1])), (int(b[0] + wid / 2), int(b[1])), _rand_color(rng), -1)
    # scorebug / overlays (nearly always on TV)
    if rng.random() < (0.9 if tv else 0.5):
        y0 = int(IN_H * rng.uniform(0.82, 0.92)) if rng.random() < 0.7 else int(IN_H * rng.uniform(0.02, 0.1))
        x0, x1 = int(IN_W * rng.uniform(0.05, 0.3)), int(IN_W * rng.uniform(0.6, 0.95))
        cv2.rectangle(img, (x0, y0), (x1, y0 + int(IN_H * 0.07)), _rand_color(rng), -1)
        for _ in range(rng.integers(2, 8)):
            xa = int(rng.integers(x0, x1 - 5))
            cv2.rectangle(img, (xa, y0 + 2), (xa + int(rng.integers(4, 30)), y0 + int(IN_H * 0.05)), _rand_color(rng), -1)
    # photometric
    img = img.astype(np.float32)
    img = img * rng.uniform(0.6, 1.3) + rng.uniform(-30, 30)
    img *= rng.uniform(0.85, 1.15, 3)
    if rng.random() < 0.4:
        k = int(rng.integers(3, 9))
        ker = np.zeros((k, k), np.float32)
        ker[k // 2] = 1.0 / k
        M = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), rng.uniform(0, 180), 1)
        ker = cv2.warpAffine(ker, M, (k, k))
        img = cv2.filter2D(img, -1, ker / max(ker.sum(), 1e-6))
    img += rng.normal(0, rng.uniform(0, 6), img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    if rng.random() < 0.5:
        ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(30, 90))])
        img = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    line, heat = targets(H, C, R, names)
    return img, line, heat, {"spec": spec, "L": L, "W": W, "H": H, "kind": kind}


def targets(H: np.ndarray, C, R, names: list[str], sigma: float = 1.5):
    """Line mask + keypoint heatmaps at OUT resolution for court->IN-image homography H. C, R = the camera
    (synthetic samples); C=None: visibility from H's own depth row (real frames, make_courtnet_real.py)."""
    s = OUT_W / IN_W
    S = np.diag([s, s, 1.0])
    line = np.zeros((OUT_H, OUT_W), np.uint8)
    for run in _project_lines(H, IN_W, IN_H, C, R):
        cv2.polylines(line, [np.int32(np.round(run * s)).reshape(-1, 1, 2)], False, 1, 1, cv2.LINE_AA)
    heat = np.zeros((len(names), OUT_H, OUT_W), np.float32)
    yy, xx = np.mgrid[0:OUT_H, 0:OUT_W]
    kp = np.array([court.KEYPOINTS[n] for n in names])
    depth = _depth(H, kp, C, R)
    q = np.column_stack([kp, np.ones(len(kp))]) @ (S @ H).T
    for k in range(len(names)):
        if depth[k] <= 0.3:
            continue
        u, v = q[k, :2] / q[k, 2]
        if -3 <= u < OUT_W + 3 and -3 <= v < OUT_H + 3:
            heat[k] = np.exp(-((xx - u) ** 2 + (yy - v) ** 2) / (2 * sigma ** 2))
    return line.astype(np.float32), heat
