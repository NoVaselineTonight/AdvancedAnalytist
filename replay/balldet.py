"""Dedicated ball detector (NOT wired by default): a small fully-convolutional heatmap network.

Trained locally (tools/train_balldet.py) on the user's own footage and the approved external venues: auto-labels from
confident ball-track frames (ball_track.link_ball over the pass-1 candidates), hard negatives (candidates the
tracker rejects: heads, logos, rims, spare balls) and random background. PyTorch (BSD) for training only;
inference through ONNX Runtime (MIT) on CPU or DirectML.

Input: the frame scaled to ``SCALE_H`` px height (any aspect), 4 channels = RGB + the previous analysed frame in gray
(motion cue: a ball moves, a logo does not), float /255. Output (stride 4): channel 0 = ball-centre logit, channel 1 =
ball diameter / 32 (in scaled pixels). :func:`detect` returns boxes in original-frame pixels.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

SCALE_H = 720  # frames are scaled to this height (720p broadcast/phone = 1.0)
STRIDE = 4
CROP = 192  # training crop (scaled pixels)
MODEL_DIR = Path(__file__).resolve().parents[1] / "data" / "balldet"


def build_model():
    import torch
    from torch import nn

    def cbr(ci, co, s=1):
        return nn.Sequential(nn.Conv2d(ci, co, 3, stride=s, padding=1, bias=False), nn.BatchNorm2d(co),
                             nn.ReLU(inplace=True))

    class BallNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.s1 = nn.Sequential(cbr(4, 24, 2), cbr(24, 24))  # 1/2
            self.s2 = nn.Sequential(cbr(24, 48, 2), cbr(48, 48))  # 1/4
            self.s3 = nn.Sequential(cbr(48, 64, 2), cbr(64, 64), cbr(64, 64))  # 1/8 (context)
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            self.fuse = nn.Sequential(cbr(48 + 64, 48), cbr(48, 32))
            self.head = nn.Conv2d(32, 2, 1)
            with torch.no_grad():
                self.head.bias[0] = -4.0  # rare positives

        def forward(self, x):
            a = self.s2(self.s1(x))
            b = self.s3(a)
            return self.head(self.fuse(torch.cat([a, self.up(b)], 1)))

    return BallNet()


def prepare(img_bgr: np.ndarray, prev_bgr: np.ndarray | None) -> tuple[np.ndarray, float]:
    """(1, 4, H, W) float32 input and the scale factor (scaled / original)."""
    s = SCALE_H / img_bgr.shape[0]
    w = int(round(img_bgr.shape[1] * s / STRIDE / 2)) * STRIDE * 2
    im = cv2.resize(img_bgr, (w, SCALE_H), interpolation=cv2.INTER_AREA)
    pv = cv2.resize(prev_bgr if prev_bgr is not None else img_bgr, (w, SCALE_H), interpolation=cv2.INTER_AREA)
    x = np.concatenate([im[:, :, ::-1], cv2.cvtColor(pv, cv2.COLOR_BGR2GRAY)[:, :, None]], 2)
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None], np.float32) / 255.0, s


def decode(out: np.ndarray, s: float, thr: float = 0.3, max_n: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Network output (1, 2, h, w) -> (boxes (n, 4) in original pixels, scores)."""
    heat = 1.0 / (1.0 + np.exp(-out[0, 0]))
    size = np.maximum(out[0, 1] * 32.0, 4.0)
    m = heat == cv2.dilate(heat, np.ones((3, 3), np.uint8))
    ys, xs = np.nonzero(m & (heat >= thr))
    order = np.argsort(-heat[ys, xs])[:max_n]
    boxes, scores = [], []
    for k in order:
        y, x = ys[k], xs[k]
        cx, cy, d = (x + 0.5) * STRIDE / s, (y + 0.5) * STRIDE / s, size[y, x] / s
        boxes.append([cx - d / 2, cy - d / 2, cx + d / 2, cy + d / 2])
        scores.append(float(heat[y, x]))
    return np.array(boxes, float).reshape(-1, 4), np.array(scores, float)


class BallDet:
    """ONNX inference. ``providers`` default: DirectML when available, else CPU."""

    def __init__(self, path: Path | str | None = None, providers: list[str] | None = None):
        import onnxruntime as ort
        path = Path(path or MODEL_DIR / "balldet.onnx")
        avail = ort.get_available_providers()
        providers = providers or [p for p in ("DmlExecutionProvider", "CPUExecutionProvider") if p in avail]
        self.sess = ort.InferenceSession(str(path), providers=providers)
        self.name = self.sess.get_inputs()[0].name

    def detect(self, img_bgr: np.ndarray, prev_bgr: np.ndarray | None = None, thr: float = 0.3
               ) -> tuple[np.ndarray, np.ndarray]:
        x, s = prepare(img_bgr, prev_bgr)
        out = self.sess.run(None, {self.name: x})[0]
        return decode(out, s, thr)


# ====================================================================================== pass-2 hook (OFF by default)
import os  # noqa: E402

ENABLED = os.environ.get("REPLAY_BALLDET", "0") == "1"
CACHE = "balldet_rows.npz"


def augment(work: Path, data, video, thr: float = 0.4, model_path: Path | None = None):
    """Pass-2 hook (env REPLAY_BALLDET=1, needs data/balldet/balldet.onnx): the dedicated ball detector runs on every
    analysed sample (previous sample = motion channel); its detections that are not near a pass-1 ball candidate are
    appended as extra ball rows (class BALL, conf = heatmap score), so ball tracking / scoring see them. Cached in
    work/balldet_rows.npz keyed by the model file and the samples. Unchanged when disabled / no model / no video."""
    import hashlib
    import time
    from replay.detection import BALL
    if not ENABLED:
        return data
    model_path = Path(model_path or MODEL_DIR / "balldet.onnx")
    if not model_path.exists() or not len(data.k):
        return data
    st = model_path.stat()
    key = hashlib.sha1(f"{st.st_size}-{st.st_mtime}-{len(data.k)}-{int(data.k[0])}-{int(data.k[-1])}-{thr}".encode()).hexdigest()
    path = Path(work) / CACHE
    new = None
    if path.exists():
        try:
            with np.load(path) as z:
                new = z["rows"] if str(z["key"]) == key else None
        except Exception:  # noqa: BLE001
            new = None
    if new is None:
        from replay.redetect import iter_frames
        from replay.team_appearance import resolve_video
        vid = resolve_video(data, video)
        if vid is None:
            return data
        t0 = time.perf_counter()
        det = BallDet(model_path)
        width = data.rows.shape[1]
        p1 = data.rows[data.rows[:, 6] == BALL]
        out, prev = [], None
        for i, img in iter_frames(Path(vid), data, list(range(len(data.k)))):
            boxes, sc = det.detect(img, prev, thr=thr)
            prev = img
            if not len(boxes):
                continue
            mine = p1[p1[:, 0] == data.k[i]]
            for b, c in zip(boxes, sc):
                cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                if len(mine) and np.any(np.hypot((mine[:, 1] + mine[:, 3]) / 2 - cx,
                                                 (mine[:, 2] + mine[:, 4]) / 2 - cy) < max(8.0, b[2] - b[0])):
                    continue
                row = np.zeros(width)
                row[0], row[1:5], row[5], row[6], row[7] = data.k[i], b, c, BALL, -1
                out.append(row)
        new = np.array(out, float).reshape(-1, width)
        print(f"balldet: {len(new)} extra ball candidates on {len(data.k)} samples, {time.perf_counter() - t0:.0f} s")
        try:
            tmp = Path(work) / "balldet_rows.tmp.npz"
            np.savez_compressed(tmp, key=np.array(key), rows=new)
            os.replace(tmp, path)
        except OSError:
            pass
    if not len(new):
        return data
    from replay.redetect import _with_rows
    return _with_rows(data, new)
