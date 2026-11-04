"""CLIP (ViT-B/32, LAION-2B, MIT) image embeddings of person crops for roles and team grouping (pass 2).

Model files: models/clip_b32_laion/vision.onnx + role_text.npy + roles.json, made by tools/clip/export_clip.py from
the safetensors weights (no torch / tokenizer needed here). Missing files -> ``available()`` is False and callers
keep the colour method.

API
  available()                               -> bool
  embed_rows(work, data, video, rows, every_s) -> {row: (512,) unit vector}, cached in work/clip_embeddings.npz
  track_roles(emb_by_track)                 -> {track: {role: prob}} (zero-shot, mean embedding per track)
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MD = ROOT / "models" / "clip_b32_laion"
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)
SIZE = 224
MIN_BOX_H = 40
_SESSION = None


def available() -> bool:
    return (MD / "vision.onnx").exists() and (MD / "role_text.npy").exists() and (MD / "roles.json").exists()


def _session():
    global _SESSION
    if _SESSION is None:
        import onnxruntime as ort

        prov = [p for p in ("DmlExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
        _SESSION = ort.InferenceSession(str(MD / "vision.onnx"), providers=prov)
    return _SESSION


def crop(img: np.ndarray, box) -> np.ndarray | None:
    """Person crop padded to a square (whole body kept), resized to 224, CLIP-normalised (3, 224, 224)."""
    x1, y1, x2, y2 = (int(round(v)) for v in box[:4])
    H, W = img.shape[:2]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
    if y2 - y1 < MIN_BOX_H or x2 - x1 < 8:
        return None
    c = img[y1:y2, x1:x2]
    h, w = c.shape[:2]
    s = max(h, w)
    sq = np.full((s, s, 3), 127, np.uint8)
    sq[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = c
    sq = cv2.resize(sq, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    x = (sq[..., ::-1].astype(np.float32) / 255.0 - MEAN) / STD
    return x.transpose(2, 0, 1)


def embed(crops: list[np.ndarray], batch: int = 32) -> np.ndarray:
    s = _session()
    out = []
    for i in range(0, len(crops), batch):
        out.append(s.run(None, {"pixel_values": np.stack(crops[i:i + batch]).astype(np.float32)})[0])
    return np.concatenate(out) if out else np.zeros((0, 512), np.float32)


def _key(data, rows: np.ndarray, every_s: float) -> str:
    h = hashlib.sha1()
    st = (MD / "vision.onnx").stat()
    h.update(json.dumps({"v": 1, "video": data.signature.get("video"), "every": every_s, "model": st.st_size},
                        sort_keys=True).encode())
    h.update(np.ascontiguousarray(rows, np.int64).tobytes())
    h.update(np.ascontiguousarray(data.rows[:, :8], np.float32).tobytes())
    return h.hexdigest()[:20]


def embed_rows(work: Path, data, video, rows: np.ndarray, every_s: float = 0.5) -> dict[int, np.ndarray]:
    """Embeddings of the given pass-1 person rows on a sample grid (one decoded frame every ``every_s``)."""
    from replay.team_appearance import resolve_video, sample_frames

    work = Path(work)
    rows = np.unique(np.asarray(rows, int))
    path = work / "clip_embeddings.npz"
    key = _key(data, rows, every_s)
    if path.exists():
        try:
            with np.load(path) as z:
                if str(z["key"]) == key:
                    return dict(zip(z["rows"].tolist(), z["emb"].astype(np.float32)))
        except Exception:  # noqa: BLE001
            pass
    vid = resolve_video(data, video)
    if vid is None:
        return {}
    t0 = time.perf_counter()
    idx_of_k = {int(k): i for i, k in enumerate(data.k)}
    grid = set(sample_frames(data.t, every_s).tolist())
    at: dict[int, list[int]] = {}
    for r in rows:
        i = idx_of_k.get(int(data.rows[r, 0]), -1)
        if i in grid:
            at.setdefault(i, []).append(int(r))
    want = sorted(at)
    out_rows, crops = [], []
    cap = cv2.VideoCapture(str(vid))
    if len(want) <= SEEK_MAX_FRAMES:  # few frames: seek to each (decoding everything up to the last costs more)
        try:
            for i in want:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(data.frame[i]))
                ok, img = cap.read()
                if ok:
                    for r in at[i]:
                        c = crop(img, data.rows[r, 1:5])
                        if c is not None:
                            out_rows.append(r)
                            crops.append(c)
        finally:
            cap.release()
        return dict(zip(out_rows, embed(crops)))
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        half = 0.5 / fps
        start = float(data.signature.get("start_s") or 0.0)
        if want:
            seek = max(0.0, min(float(data.t[want[0]]), start) - 2.0)
            if seek > 0:
                cap.set(cv2.CAP_PROP_POS_MSEC, seek * 1000.0)
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
                for r in at[want[j]]:
                    c = crop(img, data.rows[r, 1:5])
                    if c is not None:
                        out_rows.append(r)
                        crops.append(c)
            j += 1
    finally:
        cap.release()
    emb = embed(crops)
    print(f"CLIP: {len(crops)} crops from {len(want)} frames embedded in {time.perf_counter() - t0:.1f} s")
    try:
        np.savez_compressed(work / "clip_embeddings.tmp.npz", key=np.array(key), rows=np.array(out_rows, np.int64),
                            emb=emb.astype(np.float16))
        (work / "clip_embeddings.tmp.npz").replace(path)
    except OSError:
        pass
    return dict(zip(out_rows, emb))


def role_text() -> tuple[list[str], np.ndarray, float]:
    meta = json.loads((MD / "roles.json").read_text(encoding="utf-8"))
    return meta["roles"], np.load(MD / "role_text.npy"), float(meta.get("logit_scale", 100.0))


def track_roles(emb_by_track: dict[int, np.ndarray]) -> dict[int, dict[str, float]]:
    """Zero-shot role probabilities per track from its mean (unit) embedding."""
    names, T, scale = role_text()
    out = {}
    for t, E in emb_by_track.items():
        E = np.asarray(E, np.float32).reshape(-1, T.shape[1])
        if not len(E):
            continue
        m = E.mean(0)
        m = m / max(float(np.linalg.norm(m)), 1e-9)
        z = scale * (T @ m)
        p = np.exp(z - z.max())
        p /= p.sum()
        out[int(t)] = dict(zip(names, p.tolist()))
    return out


# ----------------------------------------------------------------------------------------------- gym bench cue
BENCH_DROP_P = 0.85  # mean probe probability "bench" above which a kept track is dropped
PLAYER_MIN_S, PLAYER_MIN_SPEED, PLAYER_IN_M = 5.0, 0.4, 1.0
BENCH_MIN_S, BENCH_STILL_M, BENCH_NEAR_LINE_M = 10.0, 0.8, 1.5
MIN_CROPS = 20
BENCH_REASONS = ("static (bench/wall)", "seated", "next to the bench")


def _probe2(X: np.ndarray, y: np.ndarray, iters: int = 300) -> tuple[np.ndarray, float]:
    """Class-balanced logistic regression (L2, gradient descent): returns (w, b)."""
    w, b = np.zeros(X.shape[1]), 0.0
    cw = np.where(y == 1, len(y) / (2 * max(1, y.sum())), len(y) / (2 * max(1, (1 - y).sum())))
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ w + b)))
        g = (p - y) * cw
        w -= 0.5 * (X.T @ g / len(y) + 1e-3 * w)
        b -= 0.5 * float(g.mean())
    return w, b


def _pick_rows(rows: np.ndarray, det_box: np.ndarray, det_i: np.ndarray, fps: float, n: int, min_dt_s: float) -> list:
    """At most n rows of one track: the largest boxes, at least min_dt_s apart (spread over the track)."""
    h = det_box[:, 3] - det_box[:, 1]
    out, used = [], []
    for j in np.argsort(-h):
        if h[j] < MIN_BOX_H:
            break
        if all(abs(det_i[j] - u) >= min_dt_s * fps for u in used):
            out.append(int(rows[j]))
            used.append(det_i[j])
            if len(out) >= n:
                break
    return out


def embed_rows_exact(data, video, rows) -> dict[int, np.ndarray]:
    """Embeddings of exactly these pass-1 rows (frames decoded in one sequential pass up to the last one needed)."""
    from replay.team_appearance import resolve_video

    vid = resolve_video(data, video)
    if vid is None or not len(rows):
        return {}
    idx_of_k = {int(k): i for i, k in enumerate(data.k)}
    at: dict[int, list[int]] = {}
    for r in rows:
        at.setdefault(idx_of_k.get(int(data.rows[r, 0]), -1), []).append(int(r))
    at.pop(-1, None)
    want = sorted(at)
    out_rows, crops = [], []
    cap = cv2.VideoCapture(str(vid))
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        half = 0.5 / fps
        start = float(data.signature.get("start_s") or 0.0)
        seek = max(0.0, min(float(data.t[want[0]]), start) - 2.0)
        if seek > 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, seek * 1000.0)
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
                for r in at[want[j]]:
                    c = crop(img, data.rows[r, 1:5])
                    if c is not None:
                        out_rows.append(r)
                        crops.append(c)
            j += 1
    finally:
        cap.release()
    return dict(zip(out_rows, embed(crops)))


SEEK_MAX_FRAMES = 150
CROPS_PER_TRACK = 4
CAND_MIN_S = 3.0  # shorter kept tracks are left to roster.py
CROP_MIN_DT_S = 1.0
MOVER_SAMPLE = 20  # confident on-court movers embedded as "player" examples
BENCH_LABEL_MAX = 20  # roster-dropped bench tracks embedded as "bench" examples
CAND_EDGE_M = 1.0  # candidate: median position within this of a line (or outside)
CAND_SCORE = 0.0  # ... or median roster score below this (low player-likeness)


def bench_refine(work, data, video, det_row, det_i, det_xy, tid, cand, roster, fps: float, every_s: float = 0.5,
                 det_box=None) -> int:
    """Gym footage: per-game CLIP probe "on-court player" vs "bench / coach / spectator". Only CANDIDATE tracks
    (near/outside the lines, standing still, or low roster player-likeness) are embedded - at most CROPS_PER_TRACK
    large boxes each, spread in time - plus examples: confident on-court movers (player) and tracks roster.py already
    called bench / seated (bench). Candidates the probe calls bench (mean p >= BENCH_DROP_P) are dropped
    ("bench (appearance)"). Returns the number of dropped tracks."""
    from replay import court

    if not available():
        return 0
    t0 = time.perf_counter()
    det_row = np.asarray(det_row, int)
    det_box = data.rows[det_row, 1:5] if det_box is None else det_box
    tracks = np.unique(tid[cand & (tid > 0)])
    movers, benches, cands, sel = [], [], [], {}
    for t in tracks:
        m = cand & (tid == t)
        p = det_xy[m]
        k = det_i[m]
        okp = np.isfinite(p).all(axis=1)
        if okp.sum() < 3:
            continue
        pp = p[okp]
        dur = (k.max() - k.min()) / fps
        med = np.median(pp, axis=0)
        inside = min(med[0], court.LENGTH - med[0], med[1], court.WIDTH - med[1])
        o = np.argsort(k[okp])
        speed = float(np.linalg.norm(np.diff(pp[o], axis=0), axis=1).sum()) / max(dur, 1e-6)
        still = float(np.percentile(np.linalg.norm(pp - med, axis=1), 90)) < BENCH_STILL_M
        kept = bool(roster.keep[m].any())
        bench_rule = np.isin(roster.reason[m], BENCH_REASONS).mean() > 0.5
        mover = dur >= PLAYER_MIN_S and inside >= PLAYER_IN_M and speed >= PLAYER_MIN_SPEED
        score = float(np.nanmedian(roster.score[m])) if np.isfinite(roster.score[m]).any() else 0.0
        if kept and not mover and dur >= CAND_MIN_S and (inside <= CAND_EDGE_M or still or score < CAND_SCORE):
            cands.append(int(t))
        elif mover and kept:
            movers.append(int(t))
        elif bench_rule or (still and dur >= BENCH_MIN_S and inside <= BENCH_NEAR_LINE_M):
            benches.append(int(t))
        sel[int(t)] = m
    rng = np.random.default_rng(0)
    movers = list(rng.permutation(movers)[:MOVER_SAMPLE]) if movers else []
    benches = list(rng.permutation(benches)[:BENCH_LABEL_MAX]) if benches else []
    rows_of = {}
    for t in set(cands) | set(movers) | set(benches):
        m = sel[t]
        rows_of[t] = _pick_rows(det_row[m], det_box[m], det_i[m], fps, CROPS_PER_TRACK, CROP_MIN_DT_S)
    want = sorted({r for v in rows_of.values() for r in v})
    t1 = time.perf_counter()
    emb = embed_rows_exact(data, video, want)
    t_emb = time.perf_counter() - t1
    X, y = [], []
    for ts, c in ((movers, 0), (benches, 1)):
        for t in ts:
            e = [emb[r] for r in rows_of[t] if r in emb]
            X += e
            y += [c] * len(e)
    y = np.array(y)
    if len(y) < 2 * MIN_CROPS or y.sum() < MIN_CROPS or (1 - y).sum() < MIN_CROPS:
        print(f"CLIP bench cue: too few examples (player {int((1 - y).sum())}, bench {int(y.sum())} crops) - skipped"
              f" ({time.perf_counter() - t0:.1f} s)")
        return 0
    w, b = _probe2(np.array(X), y)
    dropped = 0
    for t in cands:
        e = [emb[r] for r in rows_of[t] if r in emb]
        if len(e) < 2:
            continue
        pb = float(np.mean(1 / (1 + np.exp(-(np.array(e) @ w + b)))))
        if pb >= BENCH_DROP_P:
            m = roster.keep & (tid == t)
            roster.keep[m] = False
            roster.reason[m] = "bench (appearance)"
            dropped += 1
    if dropped:
        roster.dropped["bench (appearance)"] = roster.dropped.get("bench (appearance)", 0) + dropped
    print(f"CLIP bench cue: {dropped} of {len(cands)} candidate tracks dropped ({len(want)} crops embedded, probe on "
          f"{int((1 - y).sum())} player / {int(y.sum())} bench crops, {time.perf_counter() - t0:.1f} s, of which decode+embed "
          f"{t_emb:.1f} s)")
    return dropped
