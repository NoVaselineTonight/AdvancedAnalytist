"""Held-out evaluation of re-ID embeddings on the crop set of extract_crops.py (split 1 = Q4 of the game).

Metrics (higher is better; all on crops never used in training):
  gap_r1[bin]   tracklet retrieval: query = crop of clean piece A at t; candidates = crop of A at t+gap (gap bin
                2-5 / 5-10 / 10-30 s) + crops of every OTHER person of the SAME team in that same frame.
                Rank-1 = the true one is the most similar. chance = mean 1/len(candidates).
  num_r1/num_map  jersey-number identity: query = crop with a confident number read; gallery = number-read crops of
                the same team more than 30 s away; correct = nearest neighbour has the same number. (OCR noise caps
                this below 100 %.)
Baselines: colour histogram (HSV 8x4x4 of the crop, Hellinger) and random.

Usage: python tools/reid/evaluate.py --data data/reid/fullgame [--onnx models/reid/reid.onnx]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

GAP_BINS = [(2.0, 5.0), (5.0, 10.0), (10.0, 30.0)]


def team_of_crops(crops: np.ndarray) -> np.ndarray:
    """0 orange/red, 1 purple, 2 other (grey/black referees, spectators) from the torso hue of each crop."""
    out = np.full(len(crops), 2, np.int8)
    for i, c in enumerate(crops):
        hsv = cv2.cvtColor(np.ascontiguousarray(c[22:62, 14:50]), cv2.COLOR_BGR2HSV).reshape(-1, 3)
        s, v, h = hsv[:, 1], hsv[:, 2], hsv[:, 0]
        orange = ((s > 150) & (v > 70) & ((h < 12) | (h > 168))).mean()  # stricter than skin / wood
        purple = ((s > 70) & (h > 120) & (h < 165)).mean()
        if max(orange, purple) > 0.2:
            out[i] = 0 if orange > purple else 1
    return out


def colour_hist(crops: np.ndarray) -> np.ndarray:
    feats = np.zeros((len(crops), 128), np.float32)
    for i, c in enumerate(crops):
        hsv = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
        hst = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256]).ravel()
        feats[i] = np.sqrt(hst / max(hst.sum(), 1))
    return feats / np.maximum(np.linalg.norm(feats, axis=1, keepdims=True), 1e-8)


def gap_trials(meta: dict, idx: np.ndarray, team: np.ndarray, rng: np.random.Generator, max_q_per_piece: int = 6):
    """List of (query, positive, [negatives], bin) index triples into the eval set (positions in idx)."""
    t = meta["t"][idx]
    piece = meta["piece"][idx]
    # crops by frame (same t) for same-frame negatives
    order = np.argsort(t, kind="stable")
    ts = t[order]
    trials = []
    by_piece: dict[int, np.ndarray] = {}
    for i, p in enumerate(piece):
        if p >= 0:
            by_piece.setdefault(int(p), []).append(i)
    for p, members in by_piece.items():
        members = np.array(members)
        mt = t[members]
        qs = members if len(members) <= max_q_per_piece else rng.choice(members, max_q_per_piece, replace=False)
        for q in qs:
            for b, (lo, hi) in enumerate(GAP_BINS):
                cand = members[(mt - t[q] >= lo) & (mt - t[q] < hi)]
                if not len(cand):
                    continue
                pos = int(rng.choice(cand))
                if team[pos] > 1:
                    continue
                a, z = np.searchsorted(ts, t[pos] - 0.02), np.searchsorted(ts, t[pos] + 0.02)
                same_frame = order[a:z]
                negs = [int(j) for j in same_frame if piece[j] != p and team[j] == team[pos]]
                if negs:
                    trials.append((int(q), pos, negs, b))
    return trials


def eval_gap(E: np.ndarray, trials) -> dict:
    res = {}
    for b, (lo, hi) in enumerate(GAP_BINS):
        tb = [tr for tr in trials if tr[3] == b]
        if not tb:
            continue
        hits = [float(E[q] @ E[p] > max(E[q] @ E[n] for n in negs)) for q, p, negs, _ in tb]
        res[f"{lo:g}-{hi:g}s"] = round(float(np.mean(hits)), 4)
    allh = [float(E[q] @ E[p] > max(E[q] @ E[n] for n in negs)) for q, p, negs, _ in trials]
    res["all"] = round(float(np.mean(allh)), 4) if allh else None
    return res


def eval_numbers(E: np.ndarray, meta: dict, idx: np.ndarray, team: np.ndarray, min_dt: float = 30.0) -> dict:
    code = meta["code"][idx]
    cc = meta["code_conf"][idx]
    t = meta["t"][idx]
    sel = np.flatnonzero((code >= 0) & (cc >= 0.9) & (team <= 1))
    if len(sel) < 10:
        return {}
    r1, aps, nq = [], [], 0
    S = E[sel] @ E[sel].T
    for a, i in enumerate(sel):
        g = (team[sel] == team[i]) & (np.abs(t[sel] - t[i]) > min_dt)
        if not (g & (code[sel] == code[i])).any() or g.sum() < 2:
            continue
        s = np.where(g, S[a], -np.inf)
        o = np.argsort(-s)[: int(g.sum())]
        rel = (code[sel][o] == code[i]).astype(float)
        r1.append(rel[0])
        prec = np.cumsum(rel) / np.arange(1, len(rel) + 1)
        aps.append(float((prec * rel).sum() / rel.sum()))
        nq += 1
    return {"num_r1": round(float(np.mean(r1)), 4), "num_map": round(float(np.mean(aps)), 4), "num_queries": nq}


def load_eval(data: Path):
    meta = dict(np.load(data / "meta.npz"))
    crops = np.load(data / "crops.npy", mmap_mode="r")
    idx = np.flatnonzero((meta["split"] == 1) & meta["got"])
    ev = np.asarray(crops[idx])
    team = team_of_crops(ev)
    rng = np.random.default_rng(0)
    trials = gap_trials(meta, idx, team, rng)
    return meta, idx, ev, team, trials


def evaluate_embeddings(E, meta, idx, team, trials) -> dict:
    out = {"gap_r1": eval_gap(E, trials)}
    out.update(eval_numbers(E, meta, idx, team))
    return out


def chance(trials) -> float:
    return round(float(np.mean([1.0 / (1 + len(n)) for _, _, n, _ in trials])), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data/reid/fullgame"))
    ap.add_argument("--onnx", default=None)
    ap.add_argument("--out", default=None, help="write the metrics json here")
    args = ap.parse_args()
    meta, idx, ev, team, trials = load_eval(Path(args.data))
    print(f"eval crops {len(idx)}, teams {np.bincount(team, minlength=3).tolist()}, gap trials {len(trials)}, "
          f"mean candidates {np.mean([1 + len(n) for _, _, n, _ in trials]):.2f}, chance r1 {chance(trials)}")
    res = {"chance_gap_r1": chance(trials), "n_eval_crops": int(len(idx)), "n_gap_trials": len(trials)}
    res["colour_hist"] = evaluate_embeddings(colour_hist(ev), meta, idx, team, trials)
    rnd = np.random.default_rng(1).normal(size=(len(ev), 32)).astype(np.float32)
    res["random"] = evaluate_embeddings(rnd / np.linalg.norm(rnd, axis=1, keepdims=True), meta, idx, team, trials)
    if args.onnx:
        import time

        from replay.reid import ReidModel

        m = ReidModel(args.onnx)
        t0 = time.perf_counter()
        E = m.embed(ev)
        dt = time.perf_counter() - t0
        res["reid"] = evaluate_embeddings(E, meta, idx, team, trials)
        res["reid"]["embed_crops_per_s"] = round(len(ev) / dt, 1)
        res["reid"]["device"] = m.device
        res["reid"]["model_info"] = m.info
    print(json.dumps(res, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
