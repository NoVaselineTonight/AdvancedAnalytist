"""Out-of-domain check of the re-ID model (trained on the koristv game only) on the external footage.

1. TrackID3x3 Indoor (CC BY 4.0, ground-truth MOT identities, 42 short 3x3 clips, 6 players each): query = GT crop
   of player i at frame f; gallery = GT crops of all 6 players >= --min-dt s away in the same clip.
   rank-1 / mAP. Real identity labels (but players wear differently coloured bibs, so colour is strong there).
2. Broadcast segments (hs_hazen, tbf_turkey, ven_superliga, westwood_girls; no identity labels): the pseudo-label
   gap test of evaluate.py with clean ByteTrack pieces; distractors = the 3 people in the positive's frame whose
   colour histogram is closest to the positive (team-mates / same kit), since team labels are unknown.
Baselines: colour histogram (HSV 8x4x4) and random. Writes data/reid/external_eval.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

from evaluate import GAP_BINS, colour_hist  # noqa: E402
from extract_crops import build_pieces, coverage  # noqa: E402

from replay.pipeline import load_pass1  # noqa: E402
from replay.reid import ReidModel, crop_box  # noqa: E402

EXT = ROOT / "data" / "external"


def retrieval(E: np.ndarray, ident: np.ndarray, group: np.ndarray, t: np.ndarray, min_dt: float) -> dict:
    S = E @ E.T
    r1, aps = [], []
    for i in range(len(E)):
        g = (group == group[i]) & (np.abs(t - t[i]) >= min_dt)
        if not (g & (ident == ident[i])).any() or g.sum() < 2:
            continue
        o = np.flatnonzero(g)[np.argsort(-S[i, g])]
        rel = (ident[o] == ident[i]).astype(float)
        r1.append(rel[0])
        prec = np.cumsum(rel) / np.arange(1, len(rel) + 1)
        aps.append(float((prec * rel).sum() / rel.sum()))
    return {"r1": round(float(np.mean(r1)), 4), "map": round(float(np.mean(aps)), 4), "queries": len(r1)}


def trackid3x3(model: ReidModel, every: int, min_dt: float, max_videos: int) -> dict:
    gt_dir = EXT / "trackid3x3" / "github" / "ground_truth" / "Indoor" / "MOT"
    vid_dir = EXT / "trackid3x3" / "videos" / "Indoor"
    crops, ident, group, tt = [], [], [], []
    for vi, gt in enumerate(sorted(gt_dir.glob("*.txt"))[:max_videos]):
        video = vid_dir / (gt.stem + ".mp4")
        if not video.exists():
            continue
        a = np.loadtxt(gt, delimiter=",", ndmin=2)
        cap = cv2.VideoCapture(str(video))
        fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
        f = 0
        while True:
            ok, img = cap.read()
            if not ok:
                break
            f += 1  # MOT frames are 1-based
            if (f - 1) % every:
                continue
            for r in a[a[:, 0] == f]:
                x, y, w, h = r[2:6]
                c = crop_box(img, (x, y, x + w, y + h)) if h >= 40 else None
                if c is not None:
                    crops.append(c)
                    ident.append(int(r[1]))
                    group.append(vi)
                    tt.append(f / fps)
        cap.release()
    crops = np.stack(crops)
    ident, group, tt = np.array(ident), np.array(group), np.array(tt)
    out = {"crops": len(crops), "videos": int(len(np.unique(group))), "min_dt_s": min_dt}
    rnd = np.random.default_rng(0).normal(size=(len(crops), 32))
    for name, E in (("reid", model.embed(crops)), ("colour_hist", colour_hist(crops)),
                    ("random", rnd / np.linalg.norm(rnd, axis=1, keepdims=True))):
        out[name] = retrieval(E, ident, group, tt, min_dt)
    return out


def broadcast(model: ReidModel, seg: Path, n_hard: int = 3) -> dict | None:
    work = seg / "out" / "work"
    if not (work / "DONE").exists() or not (work / "team_appearance.npz").exists():
        return None
    data = load_pass1(work)
    rows = data.rows
    idx_of_k = {int(k): i for i, k in enumerate(data.k)}
    si = np.array([idx_of_k.get(int(k), -1) for k in rows[:, 0]], int)
    t = data.t[si].astype(float)
    cov = coverage(rows, si)
    with np.load(work / "team_appearance.npz") as z:
        sd = np.sqrt(np.clip(z["desc"].astype(np.float32), 0, None))
    pieces = build_pieces(rows, t, cov, sd, 50, 0.5, 0.35, 1.0, 1.5)
    rng = np.random.default_rng(0)
    trials = []  # (q_row, p_row, bin)
    for p in pieces:
        tp = t[p]
        qs = rng.choice(p, min(6, len(p)), replace=False)
        for q in qs:
            for b, (lo, hi) in enumerate(GAP_BINS):
                cand = p[(tp - t[q] >= lo) & (tp - t[q] < hi)]
                if len(cand):
                    trials.append((int(q), int(rng.choice(cand)), b))
    if not trials:
        return None
    by_si: dict[int, list[int]] = {}
    person = np.flatnonzero((rows[:, 6] == 0) & (rows[:, 4] - rows[:, 2] >= 40))
    for r in person:
        by_si.setdefault(int(si[r]), []).append(int(r))
    piece_of = np.full(len(rows), -1)
    for i, p in enumerate(pieces):
        piece_of[p] = i
    need = {r for q, p, _ in trials for r in (q, *by_si.get(int(si[p]), []))}
    need = sorted(need)
    video = Path(data.signature.get("video") or "")
    if not video.is_absolute():
        video = ROOT / video
    want_si = sorted({int(si[r]) for r in need})
    rows_at: dict[int, list[int]] = {}
    for r in need:
        rows_at.setdefault(int(si[r]), []).append(r)
    crop_of: dict[int, np.ndarray] = {}
    cap = cv2.VideoCapture(str(video))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    half = 0.5 / fps
    start = float(data.signature.get("start_s") or 0.0)
    if start > 2:
        cap.set(cv2.CAP_PROP_POS_MSEC, (start - 2) * 1000.0)
    j = 0
    while j < len(want_si):
        if not cap.grab():
            break
        ft = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        tj = float(data.t[want_si[j]])
        if ft + half < tj:
            continue
        while j < len(want_si) and float(data.t[want_si[j]]) + half < ft:
            j += 1
        if j >= len(want_si) or abs(ft - float(data.t[want_si[j]])) > half:
            continue
        ok, img = cap.retrieve()
        if ok:
            for r in rows_at[want_si[j]]:
                c = crop_box(img, rows[r, 1:5])
                if c is not None:
                    crop_of[r] = c
        j += 1
    cap.release()
    keys = sorted(crop_of)
    pos = {r: i for i, r in enumerate(keys)}
    C = np.stack([crop_of[r] for r in keys])
    H = colour_hist(C)
    rnd = np.random.default_rng(1).normal(size=(len(C), 32))
    embs = {"reid": model.embed(C), "colour_hist": H, "random": rnd / np.linalg.norm(rnd, axis=1, keepdims=True)}
    hits = {k: {b: [] for b in range(len(GAP_BINS))} for k in embs}
    n_trials = 0
    for q, p, b in trials:
        if q not in pos or p not in pos:
            continue
        others = [r for r in by_si.get(int(si[p]), []) if r in pos and r != p and piece_of[r] != piece_of[p]]
        if not others:
            continue
        # hardest-by-colour distractors: closest kit to the positive (team-mates)
        oc = np.array([H[pos[r]] @ H[pos[p]] for r in others])
        negs = [others[i] for i in np.argsort(-oc)[:n_hard]]
        n_trials += 1
        for k, E in embs.items():
            sq = E[pos[q]]
            hits[k][b].append(float(sq @ E[pos[p]] > max(sq @ E[pos[n]] for n in negs)))
    out = {"pieces": len(pieces), "trials": n_trials}
    for k in embs:
        allh = [h for b in hits[k].values() for h in b]
        out[k] = {f"{lo:g}-{hi:g}s": (round(float(np.mean(hits[k][b])), 4) if hits[k][b] else None)
                  for b, (lo, hi) in enumerate(GAP_BINS)}
        out[k]["all"] = round(float(np.mean(allh)), 4) if allh else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default=str(ROOT / "models/reid/reid.onnx"))
    ap.add_argument("--out", default=str(ROOT / "data/reid/external_eval.json"))
    ap.add_argument("--every", type=int, default=10, help="TrackID3x3: every n-th frame")
    ap.add_argument("--min-dt", type=float, default=3.0)
    ap.add_argument("--max-videos", type=int, default=42)
    ap.add_argument("--skip-trackid", action="store_true")
    args = ap.parse_args()
    model = ReidModel(args.onnx)
    res = {"model": model.info}
    if not args.skip_trackid:
        res["trackid3x3_indoor"] = trackid3x3(model, args.every, args.min_dt, args.max_videos)
        print("trackid3x3", json.dumps(res["trackid3x3_indoor"]), flush=True)
    for name in ("hs_hazen", "tbf_turkey", "ven_superliga", "westwood_girls"):
        for seg in sorted((EXT / name).glob("seg_*")):
            if seg.is_dir():
                r = broadcast(model, seg)
                if r:
                    res[f"{name}/{seg.name}"] = r
                    print(name, seg.name, json.dumps(r), flush=True)
    Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
