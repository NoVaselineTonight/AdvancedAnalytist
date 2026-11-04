"""Court detector: a small U-Net that predicts the court-line mask and court keypoint heatmaps.

Trained locally on synthetic courts (replay/court_synth.py, tools/train_courtnet.py), own code; PyTorch (BSD) for
training only, inference through ONNX Runtime (MIT) - CPU or DirectML.
Input: BGR frame resized to 320 x 192, RGB float /255. Output (stride 2, 160 x 96): channel 0 = line logit,
channels 1.. = keypoint logits in court.KEYPOINTS order (camera on the near side).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

IN_W, IN_H, STRIDE = 320, 192, 2
MODEL_DIR = Path(__file__).resolve().parents[1] / "data" / "courtnet"


def build_model(n_kp: int):
    import torch
    from torch import nn

    def block(ci, co):
        return nn.Sequential(nn.Conv2d(ci, co, 3, padding=1, bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
                             nn.Conv2d(co, co, 3, padding=1, bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True))

    class UNet(nn.Module):
        def __init__(self):
            super().__init__()
            ch = [24, 40, 64, 96, 160]
            self.e0 = block(3, ch[0])  # 1
            self.e1 = block(ch[0], ch[1])  # 1/2
            self.e2 = block(ch[1], ch[2])  # 1/4
            self.e3 = block(ch[2], ch[3])  # 1/8
            self.e4 = block(ch[3], ch[4])  # 1/16
            self.pool = nn.MaxPool2d(2)
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            self.d3 = block(ch[4] + ch[3], ch[3])
            self.d2 = block(ch[3] + ch[2], ch[2])
            self.d1 = block(ch[2] + ch[1], ch[1])
            self.head = nn.Conv2d(ch[1], 1 + n_kp, 1)

        def forward(self, x):
            e0 = self.e0(x)
            e1 = self.e1(self.pool(e0))
            e2 = self.e2(self.pool(e1))
            e3 = self.e3(self.pool(e2))
            e4 = self.e4(self.pool(e3))
            d3 = self.d3(torch.cat([self.up(e4), e3], 1))
            d2 = self.d2(torch.cat([self.up(d3), e2], 1))
            d1 = self.d1(torch.cat([self.up(d2), e1], 1))
            return self.head(d1)

    return UNet()


def preprocess(img_bgr: np.ndarray) -> np.ndarray:
    x = cv2.resize(img_bgr, (IN_W, IN_H), interpolation=cv2.INTER_AREA)[:, :, ::-1].astype(np.float32) / 255.0
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])


class CourtNet:
    """ONNX Runtime inference. predict(img) -> (line_prob (96,160), heat_prob (K,96,160))."""

    def __init__(self, path: Path | None = None, providers: list[str] | None = None, threads: int | None = None):
        import onnxruntime as ort

        path = Path(path or MODEL_DIR / "courtnet.onnx")
        so = ort.SessionOptions()
        if threads:  # evaluation next to other CPU jobs
            so.intra_op_num_threads = int(threads)
            so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(str(path), so, providers=providers or ["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name

    def predict(self, img_bgr: np.ndarray):
        y = self.sess.run(None, {self.inp: preprocess(img_bgr)})[0][0]
        p = 1.0 / (1.0 + np.exp(-y))
        return p[0], p[1:]


def peaks(heat: np.ndarray, size: tuple[int, int], thr: float = 0.3) -> dict[int, tuple[float, float, float]]:
    """Keypoint index -> (x, y, score) in full-frame pixels (sub-pixel by local centroid)."""
    w, h = size
    sx, sy = w / heat.shape[2], h / heat.shape[1]
    out = {}
    for k in range(heat.shape[0]):
        hm = heat[k]
        j = int(np.argmax(hm))
        v, u = divmod(j, hm.shape[1])
        if hm[v, u] < thr:
            continue
        v0, v1, u0, u1 = max(0, v - 2), min(hm.shape[0], v + 3), max(0, u - 2), min(hm.shape[1], u + 3)
        win = hm[v0:v1, u0:u1]
        yy, xx = np.mgrid[v0:v1, u0:u1]
        cu, cv_ = float((win * xx).sum() / win.sum()), float((win * yy).sum() / win.sum())
        out[k] = ((cu + 0.5) * sx, (cv_ + 0.5) * sy, float(hm[v, u]))
    return out


# ------------------------------------------------------------------------- homography from the predictions
def court_points(names: list[str]) -> np.ndarray:
    from replay import court

    return np.array([court.KEYPOINTS[n] for n in names], float)


def frame_H(pk: dict, cxy: np.ndarray, size: tuple[int, int], ransac_frac: float = 0.012, min_peaks: int = 4,
            min_inliers: int = 4):
    """court->image homography from one frame's keypoint peaks (RANSAC). Returns (H, inlier keypoint ids) or None.
    The defaults are the original loose settings (gemini_court); the stage-0 solver path (autocourt.calibrate_net)
    passes SOLVER_RANSAC_FRAC / SOLVER_MIN_PEAKS / SOLVER_MIN_INLIERS."""
    if len(pk) < max(4, min_peaks):
        return None
    ids = np.array(sorted(pk))
    src = cxy[ids].astype(np.float32)
    dst = np.array([pk[k][:2] for k in ids], np.float32)
    H, inl = cv2.findHomography(src, dst, cv2.RANSAC, ransac_frac * size[0])
    if H is None or inl is None or int(inl.sum()) < max(4, min_inliers):
        return None
    return H, ids[inl.ravel() > 0]


def _affine_kp(names: list[str], spec: str):
    """keypoint court positions as P0 + L*dL + W*dW for this marking standard (exact: markings are placed
    relative to baselines / sidelines / centre lines)."""
    from replay import court

    L0, W0, S0 = court.LENGTH, court.WIDTH, court.SPEC
    try:
        court.configure(28.0, 15.0, spec)
        a = court_points(names)
        court.configure(29.0, 15.0, spec)
        b = court_points(names)
        court.configure(28.0, 16.0, spec)
        c = court_points(names)
    finally:
        court.configure(L0, W0, S0)
    dL, dW = b - a, c - a
    return a - 28.0 * dL - 15.0 * dW, dL, dW


def _size_pen(spec: str, L: float, W: float) -> float:
    """Tiny tie-breaker towards the standard's size (frames that see one end only do not constrain it)."""
    from replay import court

    st = court.STANDARDS[spec]
    return 0.05 * (abs(L - st["length"]) + abs(W - st["width"]))


def choose_court(frames_pk: list[dict], names: list[str], size: tuple[int, int], specs=None):
    """Court (spec, L, W) whose keypoints let the most frames' peaks agree with ONE homography each:
    score = sum over frames of RANSAC inliers. Coarse grid, then a finer one around the best."""
    specs = specs or ["FIBA", "NBA", "NCAA", "FIBA_OLD"]
    frames_pk = [p for p in frames_pk if len(p) >= 5]
    if not frames_pk:
        return None

    def score(spec, L, W, aff):
        P0, dL, dW = aff
        cxy = P0 + L * dL + W * dW
        tot = 0
        for pk in frames_pk:
            r = frame_H(pk, cxy, size)
            if r is not None:
                tot += len(r[1])
        return tot

    best = (-1, None)
    for spec in specs:
        aff = _affine_kp(names, spec)
        for L in np.arange(20.0, 29.6, 0.5):
            for W in np.arange(12.0, 16.1, 0.5):
                s = score(spec, L, W, aff) * (1.0 if spec == "FIBA" else 0.99) - _size_pen(spec, L, W)
                if s > best[0]:
                    best = (s, (spec, float(L), float(W)))
    spec, L0, W0 = best[1]
    aff = _affine_kp(names, spec)
    for L in np.arange(L0 - 0.5, L0 + 0.51, 0.125):
        for W in np.arange(W0 - 0.5, W0 + 0.51, 0.125):
            s = score(spec, L, W, aff) * (1.0 if spec == "FIBA" else 0.99) - _size_pen(spec, L, W)
            if s > best[0]:
                best = (s, (spec, float(L), float(W)))
    return best[1] + (best[0],)


def choose_court_scored(frames: list[tuple[dict, np.ndarray]], names: list[str], size: tuple[int, int], specs=None,
                        ransac_frac: float = 0.006, min_peaks: int = 6, min_inliers: int = 5, refine_top: int = 3,
                        max_frames: int = 40):
    """Stage-0 court choice: like choose_court, but each frame (peaks, distance map) contributes the MASK SCORE of
    its RANSAC homography (court_solver.score_H; 0 when implausible / too few visible samples) instead of its
    inlier count, so a few junk peaks that happen to agree cannot outweigh a frame whose court lies on the lines.
    A finer grid is searched around each of the best ``refine_top`` coarse candidates; the returned score is the
    sum of the REFINED homographies' mask scores for the winner. Returns (spec, L, W, score) or None."""
    from replay import court, court_solver as cs

    specs = specs or ["FIBA", "NBA", "NCAA", "FIBA_OLD"]
    frames = [f for f in frames if len(f[0]) >= min_peaks]
    if not frames:
        return None
    frames = frames[:: max(1, len(frames) // max_frames)]
    L0_, W0_, S0_ = court.LENGTH, court.WIDTH, court.SPEC

    def score(spec, L, W, aff, refine=False):
        P0, dL, dW = aff
        cxy = P0 + L * dL + W * dW
        court.configure(float(L), float(W), spec)
        pts, cls = cs.class_samples(0.4)
        tot = 0.0
        for pk, dist in frames:
            r = frame_H(pk, cxy, size, ransac_frac, min_peaks, min_inliers)
            if r is None:
                continue
            H = r[0] / r[0][2, 2]
            if refine:
                H = cs.refine_H(dist, H, size, pts, blurs=(4.0, 1.5), max_nfev=30)
            if not cs.plausible(H, size, pts)[0]:
                continue
            sc = cs.score_H(dist, H, size, pts, cls)
            if sc["n"] >= cs.MIN_VISIBLE:
                tot += sc["score"]
        return tot

    try:
        cand = []
        for spec in specs:
            aff = _affine_kp(names, spec)
            for L in np.arange(20.0, 29.6, 0.5):
                for W in np.arange(12.0, 16.1, 0.5):
                    s = score(spec, L, W, aff) * (1.0 if spec == "FIBA" else 0.99) - _size_pen(spec, L, W)
                    cand.append((s, spec, float(L), float(W)))
        cand.sort(key=lambda c: -c[0])
        if cand[0][0] <= 0:
            return None
        best = (-1e9, None)
        for s0, spec, L0, W0 in cand[:max(1, refine_top)]:
            aff = _affine_kp(names, spec)
            for L in np.arange(L0 - 0.5, L0 + 0.51, 0.25):
                for W in np.arange(W0 - 0.5, W0 + 0.51, 0.25):
                    s = score(spec, L, W, aff) * (1.0 if spec == "FIBA" else 0.99) - _size_pen(spec, L, W)
                    if s > best[0]:
                        best = (s, (spec, float(L), float(W)))
        spec, L, W = best[1]
        s = score(spec, L, W, _affine_kp(names, spec), refine=True)
        return spec, L, W, s
    finally:
        court.configure(L0_, W0_, S0_)
