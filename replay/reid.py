"""Person re-identification embeddings (appearance model trained on the user's own game video).

A small CNN (``ReidNet``, trained by tools/reid/train.py, exported to ONNX) maps a person crop to a 128-d
L2-normalised vector; the cosine similarity of two vectors says how likely the two crops show the same person.
It is meant to tell apart players of the SAME team (same jersey colours): number, hair, skin, shoes, socks, body
shape and height - things the colour descriptors in team_appearance.py cannot see.

API
  crop_boxes(image_bgr, boxes)      -> (n, H, W, 3) uint8 crops, the exact preprocessing used in training
  ReidModel(path).embed(crops)      -> (n, D) float32 unit vectors (onnxruntime, DirectML when available)
  embed(crops)                      -> same, with a lazily loaded default model (models/reid/reid.onnx)
  compute_embeddings(data, video)   -> ReidEmbeddings for the pass-1 person rows (sampled every ``every_s``)
  load_or_compute(work, data, video)-> cached ``results/work/reid_embeddings.npz`` (see data/benchmark/notes_reid.md)
  load_embeddings(work)             -> cached embeddings or None

Nothing in pass 2 uses this by default (``ReidConfig.enabled`` is False); consumers opt in.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "reid" / "reid.onnx"
CROP_H, CROP_W = 128, 64
PAD_X, PAD_Y = 0.06, 0.03  # crop margin as a fraction of the box size
CACHE_NAME = "reid_embeddings.npz"
CACHE_VERSION = 1


from replay.config import ReidConfig  # noqa: E402,F401 - moved to config.py (Config.reid); re-exported



# ----------------------------------------------------------------------------------------------- preprocessing
def crop_box(image_bgr: np.ndarray, box, out_h: int = CROP_H, out_w: int = CROP_W) -> np.ndarray | None:
    """One person crop (box x1 y1 x2 y2 in pixels, with a small margin), resized to out_h x out_w (aspect not kept).

    Parts of the margin outside the image are filled by edge replication. None for a degenerate box."""
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    bw, bh = x2 - x1, y2 - y1
    if not (bw >= 2 and bh >= 4 and np.isfinite([x1, y1, x2, y2]).all()):
        return None
    x1, x2 = x1 - PAD_X * bw, x2 + PAD_X * bw
    y1, y2 = y1 - PAD_Y * bh, y2 + PAD_Y * bh
    ix1, iy1, ix2, iy2 = int(np.floor(x1)), int(np.floor(y1)), int(np.ceil(x2)), int(np.ceil(y2))
    cx1, cy1, cx2, cy2 = max(ix1, 0), max(iy1, 0), min(ix2, w), min(iy2, h)
    if cx2 - cx1 < 2 or cy2 - cy1 < 2:
        return None
    crop = image_bgr[cy1:cy2, cx1:cx2]
    if (cx1, cy1, cx2, cy2) != (ix1, iy1, ix2, iy2):
        crop = cv2.copyMakeBorder(crop, cy1 - iy1, iy2 - cy2, cx1 - ix1, ix2 - cx2, cv2.BORDER_REPLICATE)
    interp = cv2.INTER_AREA if crop.shape[0] > out_h else cv2.INTER_LINEAR
    return cv2.resize(crop, (out_w, out_h), interpolation=interp)


def crop_boxes(image_bgr: np.ndarray, boxes) -> tuple[np.ndarray, np.ndarray]:
    """Crops of all boxes -> ((m, H, W, 3) uint8, (m,) indices of the boxes that gave a crop)."""
    out, idx = [], []
    for i, b in enumerate(np.asarray(boxes, float).reshape(-1, 4)):
        c = crop_box(image_bgr, b)
        if c is not None:
            out.append(c)
            idx.append(i)
    if not out:
        return np.zeros((0, CROP_H, CROP_W, 3), np.uint8), np.zeros(0, int)
    return np.stack(out), np.array(idx, int)


# ----------------------------------------------------------------------------------------------- inference
class ReidModel:
    """ONNX re-ID model. Input: (n, 3, H, W) float32 BGR 0..255 (normalisation is inside the graph)."""

    def __init__(self, path: str | Path | None = None, device: str = "auto"):
        import onnxruntime as ort

        self.path = Path(path) if path else MODEL_PATH
        if not self.path.exists():
            raise FileNotFoundError(f"re-ID model not found: {self.path} (train it with tools/reid/train.py)")
        providers = ["CPUExecutionProvider"]
        if device != "cpu" and "DmlExecutionProvider" in ort.get_available_providers():
            providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self.session = ort.InferenceSession(str(self.path), opts, providers=providers)
        self.device = "GPU (DirectML)" if self.session.get_providers()[0] == "DmlExecutionProvider" else "CPU"
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        shp = inp.shape
        self.h = shp[2] if isinstance(shp[2], int) else CROP_H
        self.w = shp[3] if isinstance(shp[3], int) else CROP_W
        self.dim = int(self.session.get_outputs()[0].shape[-1])
        meta = self.session.get_modelmeta().custom_metadata_map or {}
        self.info = dict(meta)

    def embed(self, crops, batch: int = 64) -> np.ndarray:
        """(n, H, W, 3) uint8 BGR crops (or a list of them, any size - resized) -> (n, D) float32 unit vectors."""
        if isinstance(crops, np.ndarray) and crops.ndim == 4:
            arr = crops
        else:
            arr = [np.asarray(c) for c in crops]
            if not arr:
                return np.zeros((0, self.dim), np.float32)
            arr = np.stack([c if c.shape[:2] == (self.h, self.w) else cv2.resize(c, (self.w, self.h)) for c in arr])
        if arr.shape[1:3] != (self.h, self.w):
            arr = np.stack([cv2.resize(c, (self.w, self.h)) for c in arr])
        out = np.zeros((len(arr), self.dim), np.float32)
        for s in range(0, len(arr), batch):
            x = np.ascontiguousarray(arr[s:s + batch].transpose(0, 3, 1, 2), dtype=np.float32)
            out[s:s + batch] = self.session.run(None, {self.input_name: x})[0]
        n = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(n, 1e-8)


_DEFAULT: dict[tuple, ReidModel] = {}


def available(path: str | Path | None = None) -> bool:
    return (Path(path) if path else MODEL_PATH).exists()


def get_model(path: str | Path | None = None, device: str = "auto") -> ReidModel:
    key = (str(path or MODEL_PATH), device)
    if key not in _DEFAULT:
        _DEFAULT[key] = ReidModel(path, device)
    return _DEFAULT[key]


def embed(crops, path: str | Path | None = None, device: str = "auto") -> np.ndarray:
    """embed(crops) -> (n, D) unit vectors with the default model (see ReidModel.embed)."""
    return get_model(path, device).embed(crops)


def similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity matrix of two sets of unit vectors (NaN rows give NaN)."""
    return np.asarray(a, np.float32) @ np.asarray(b, np.float32).T


# ----------------------------------------------------------------------------------------------- per-row cache
@dataclass
class ReidEmbeddings:
    emb: np.ndarray  # (n_rows, D) float16 unit vectors; NaN = not sampled / not a person / too small
    stats: dict = field(default_factory=dict)

    @property
    def sampled(self) -> np.ndarray:
        return np.isfinite(self.emb[:, 0])

    def get(self, rows) -> np.ndarray:
        return self.emb[rows].astype(np.float32)


def _model_digest(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()[:12]


def _cache_key(data, cfg: ReidConfig, model_digest: str) -> str:
    sig = data.signature
    h = hashlib.sha1()
    h.update(json.dumps({"v": CACHE_VERSION, "video": sig.get("video"), "size": sig.get("video_size_bytes"),
                         "every": round(float(cfg.every_s), 4), "min_h": cfg.min_box_h, "model": model_digest,
                         "n": int(len(data.rows))}, sort_keys=True).encode())
    h.update(np.ascontiguousarray(data.rows[:, :8], np.float32).tobytes())
    return h.hexdigest()[:20]


def compute_embeddings(data, video: str | Path, cfg: ReidConfig | None = None, model: ReidModel | None = None,
                       progress=None) -> ReidEmbeddings | None:
    """Decode the sampled frames sequentially and embed every person box (height >= min_box_h) in them."""
    from replay.detection import PERSON
    from replay.team_appearance import sample_frames

    cfg = cfg or ReidConfig()
    model = model or get_model(cfg.model_path, cfg.device)
    rows = data.rows
    emb = np.full((len(rows), model.dim), np.nan, np.float16)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return None
    t0 = time.perf_counter()
    stats = {"device": model.device, "frames": 0, "boxes": 0, "embed_s": 0.0}
    idx_of_k = {int(k): i for i, k in enumerate(data.k)}
    row_i = np.array([idx_of_k.get(int(k), -1) for k in rows[:, 0]], int)
    rows_at: dict[int, list[int]] = {}
    from replay.detection import activation_threshold

    person = np.flatnonzero((rows[:, 6] == PERSON) & (rows[:, 4] - rows[:, 2] >= cfg.min_box_h)
                            & (rows[:, 5] >= activation_threshold(data.signature)))  # no Task-9 crumbs
    for r in person:
        rows_at.setdefault(int(row_i[r]), []).append(int(r))
    want = [int(i) for i in sample_frames(data.t, cfg.every_s) if int(i) in rows_at]
    pend_rows: list[int] = []
    pend_crops: list[np.ndarray] = []

    def flush():
        if pend_rows:
            te = time.perf_counter()
            emb[pend_rows] = model.embed(np.stack(pend_crops), cfg.batch).astype(np.float16)
            stats["embed_s"] += time.perf_counter() - te
            stats["boxes"] += len(pend_rows)
            pend_rows.clear()
            pend_crops.clear()

    try:
        if not want:
            return ReidEmbeddings(emb, stats)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        half = 0.5 / fps
        start = float(data.signature.get("start_s") or 0.0)
        seek_to = max(0.0, min(float(data.t[want[0]]), start) - 2.0)
        if seek_to > 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, seek_to * 1000.0)
        j = 0
        while j < len(want):
            if not cap.grab():
                break
            ft = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if ft + half < float(data.t[want[j]]):
                continue
            while j < len(want) and float(data.t[want[j]]) + half < ft:
                j += 1
            if j >= len(want) or abs(ft - float(data.t[want[j]])) > half:
                continue
            ok, img = cap.retrieve()
            if ok:
                rr = np.array(rows_at[want[j]], int)
                crops, ok_i = crop_boxes(img, rows[rr, 1:5])
                pend_rows.extend(int(r) for r in rr[ok_i])
                pend_crops.extend(crops)
                stats["frames"] += 1
            j += 1
            if len(pend_rows) >= 4 * cfg.batch:
                flush()
            if progress is not None and j % 50 == 0:
                progress(j / len(want))
        flush()
    finally:
        cap.release()
    stats["total_s"] = round(time.perf_counter() - t0, 2)
    stats["embed_s"] = round(stats["embed_s"], 2)
    return ReidEmbeddings(emb, stats)


def load_embeddings(work: Path) -> ReidEmbeddings | None:
    """The cached embeddings of a results/work folder (whatever settings produced them); None when absent."""
    path = Path(work) / CACHE_NAME
    if not path.exists():
        return None
    try:
        with np.load(path) as z:  # close the file: Windows cannot replace an open file later
            return ReidEmbeddings(z["emb"], json.loads(str(z["stats"])))
    except Exception:  # noqa: BLE001 - corrupt cache
        return None


def save_embeddings(work: Path, res: ReidEmbeddings, key: str = "") -> Path:
    work = Path(work)
    tmp = work / (CACHE_NAME[:-4] + ".tmp.npz")
    np.savez_compressed(tmp, key=np.array(key), emb=res.emb.astype(np.float16),
                        stats=np.array(json.dumps(res.stats)))
    path = work / CACHE_NAME
    tmp.replace(path)
    return path


def load_or_compute(work: Path, data, video: str | Path | None, cfg: ReidConfig | None = None,
                    progress=None) -> ReidEmbeddings | None:
    """Cached per-row embeddings, recomputed when the pass-1 run, settings or model changed.

    None when the model or the video is unavailable and there is no (matching) cache."""
    cfg = cfg or ReidConfig()
    work = Path(work)
    mpath = Path(cfg.model_path) if cfg.model_path else MODEL_PATH
    if not mpath.exists():
        return None
    key = _cache_key(data, cfg, _model_digest(mpath))
    path = work / CACHE_NAME
    if path.exists():
        try:
            with np.load(path) as z:
                if str(z["key"]) == key:
                    return ReidEmbeddings(z["emb"], json.loads(str(z["stats"])))
        except Exception:  # noqa: BLE001
            pass
    if video is None or not Path(video).exists():
        return None
    res = compute_embeddings(data, video, cfg, get_model(mpath, cfg.device), progress)
    if res is not None:
        res.stats["model"] = str(mpath.name)
        save_embeddings(work, res, key)
    return res


class ReidTap:
    """Pass-1 frame tap (replay.frame_tap): embeds the person boxes of the frames pass 1 decodes exactly like
    compute_embeddings (same grid, boxes, batches) and writes work/reid_embeddings.npz with the key pass 2
    checks. Only when re-ID is enabled and the model exists; off for resumed pass-1 runs."""

    name = "reid"
    queue_size = 16
    blocking = True

    def __init__(self, cfg: ReidConfig | None = None, enabled: bool = True):
        self.cfg, self.enabled = cfg or ReidConfig(), enabled

    def start(self, info) -> None:
        cfg = self.cfg
        self.mpath = Path(cfg.model_path) if cfg.model_path else MODEL_PATH
        self.active = bool(self.enabled and not info.resumed and self.mpath.exists())
        self.last_t = -np.inf
        self.model = None
        self.emb: dict[int, np.ndarray] = {}
        self.pend_rows: list[int] = []
        self.pend_crops: list[np.ndarray] = []
        self.stats = {"device": "", "frames": 0, "boxes": 0, "embed_s": 0.0}
        self.t0 = time.perf_counter()

    def wants(self, k: int, t: float) -> bool:
        if not self.active or t - self.last_t < self.cfg.every_s - 1e-6:
            return False
        self.last_t = t  # the greedy grid of sample_frames(data.t, every_s)
        return True

    def _flush(self) -> None:
        if self.pend_rows:
            te = time.perf_counter()
            from replay.frame_tap import GPU_LOCK

            with GPU_LOCK:
                out = self.model.embed(np.stack(self.pend_crops), self.cfg.batch).astype(np.float16)
            for r, e in zip(self.pend_rows, out):
                self.emb[r] = e
            self.stats["embed_s"] += time.perf_counter() - te
            self.stats["boxes"] += len(self.pend_rows)
            self.pend_rows.clear()
            self.pend_crops.clear()

    def process(self, f) -> None:
        b = np.asarray(f.persons, float).reshape(-1, 4)
        sel = np.flatnonzero(b[:, 3] - b[:, 1] >= self.cfg.min_box_h)
        if not len(sel):
            return
        if self.model is None:
            from replay.frame_tap import GPU_LOCK

            with GPU_LOCK:
                self.model = get_model(self.mpath, self.cfg.device)
            self.stats["device"] = self.model.device
        crops, ok_i = crop_boxes(f.image, b[sel])
        rr = f.row0 + sel
        self.pend_rows.extend(int(r) for r in rr[ok_i])
        self.pend_crops.extend(crops)
        self.stats["frames"] += 1
        if len(self.pend_rows) >= 4 * self.cfg.batch:
            self._flush()

    def finish(self, work: Path, completed: bool) -> None:
        if not self.active or not completed:
            return
        from replay.pipeline import load_pass1

        if self.model is None:
            self.model = get_model(self.mpath, self.cfg.device)
            self.stats["device"] = self.model.device
        self._flush()
        data = load_pass1(Path(work))
        emb = np.full((len(data.rows), self.model.dim), np.nan, np.float16)
        for r, e in self.emb.items():
            emb[r] = e
        self.stats["total_s"] = round(time.perf_counter() - self.t0, 2)
        self.stats["embed_s"] = round(self.stats["embed_s"], 2)
        self.stats["model"] = str(self.mpath.name)
        self.stats["source"] = "pass1-tap"
        save_embeddings(work, ReidEmbeddings(emb, self.stats), _cache_key(data, self.cfg, _model_digest(self.mpath)))
