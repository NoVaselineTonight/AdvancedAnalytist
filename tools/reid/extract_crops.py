"""Build the re-ID training/eval crop set from a pass-1 work folder of the user's own game.

Pseudo-labels come from *clean* ByteTrack pieces: a ByteTrack ID is cut wherever the box is gapped (> max_gap_s),
overlapped by another person box (identity swaps happen there) or its jersey colour histogram jumps. Pieces of the
same moment (present in the same frame) are guaranteed different people; crops of one piece are the same person.

Output (data/reid/<name>/):
  crops.npy   (n, 128, 64, 3) uint8 BGR (replay.reid.crop_box preprocessing)
  meta.npz    row (pass-1 row index), piece, k, t, frame, box, conf, cover, code/code_conf (jersey read),
              desc (40-d colour histogram, NaN if unsampled), split (0 train, 1 eval)
  pieces.npz  per piece: bt id, first/last k, t0, t1, n_rows

Usage: python tools/reid/extract_crops.py --work data/fullgame/out/work --video <ts> --out data/reid/fullgame
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from replay.pipeline import load_pass1  # noqa: E402
from replay.reid import CROP_H, CROP_W, crop_box  # noqa: E402


def coverage(rows: np.ndarray, row_frame: np.ndarray) -> np.ndarray:
    """Max fraction of each person box covered by another person box of the same frame."""
    cov = np.zeros(len(rows), np.float32)
    order = np.argsort(row_frame, kind="stable")
    rf = row_frame[order]
    starts = np.flatnonzero(np.r_[True, rf[1:] != rf[:-1]])
    ends = np.r_[starts[1:], len(rf)]
    for s, e in zip(starts, ends):
        if e - s < 2:
            continue
        idx = order[s:e]
        b = rows[idx, 1:5]
        ix1 = np.maximum(b[:, None, 0], b[None, :, 0])
        iy1 = np.maximum(b[:, None, 1], b[None, :, 1])
        ix2 = np.minimum(b[:, None, 2], b[None, :, 2])
        iy2 = np.minimum(b[:, None, 3], b[None, :, 3])
        inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
        np.fill_diagonal(inter, 0)
        area = np.maximum((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]), 1e-6)
        cov[idx] = (inter / area[:, None]).max(axis=1)
    return cov


def colour_cuts(sd: np.ndarray, w: int = 4, thr: float = 0.36, min_len: int = 3) -> list[int]:
    """Change points (indices into sd) of a sequence of sqrt colour histograms (NaN rows skipped)."""
    fin = np.flatnonzero(np.isfinite(sd[:, 0]))
    if len(fin) < 2 * min_len:
        return []
    v = sd[fin]
    best, bj = 0.0, -1
    for j in range(min_len, len(v) - min_len + 1):
        a = np.median(v[max(0, j - w):j], axis=0)
        b = np.median(v[j:j + w], axis=0)
        d = float(np.linalg.norm(a - b) / np.sqrt(2))
        if d > best:
            best, bj = d, j
    if best < thr:
        return []
    cut = int(fin[bj])
    left = colour_cuts(sd[:cut], w, thr, min_len)
    right = colour_cuts(sd[cut:], w, thr, min_len)
    return left + [cut] + [cut + r for r in right]


def build_pieces(rows, t, cov, sd, min_h, min_conf, max_cover, max_gap_s, min_len_s):
    """-> list of arrays of row indices (time-ordered), one per clean piece."""
    person = (rows[:, 6] == 0) & (rows[:, 7] >= 0)
    good = person & (rows[:, 4] - rows[:, 2] >= min_h) & (rows[:, 5] >= min_conf)
    pieces = []
    ids = rows[:, 7]
    cand = np.flatnonzero(person)
    order = cand[np.lexsort((t[cand], ids[cand]))]
    bt = ids[order]
    starts = np.flatnonzero(np.r_[True, bt[1:] != bt[:-1]])
    ends = np.r_[starts[1:], len(bt)]
    for s, e in zip(starts, ends):
        idx = order[s:e]
        # break at gaps and at overlap events (a covered row also ends the piece: swaps happen there)
        brk = np.r_[True, np.diff(t[idx]) > max_gap_s] | (cov[idx] > max_cover)
        seg_id = np.cumsum(brk)
        keep = good[idx] & (cov[idx] <= max_cover)
        for g in np.unique(seg_id):
            seg = idx[(seg_id == g) & keep]
            if len(seg) < 3:
                continue
            bounds = [0] + colour_cuts(sd[seg]) + [len(seg)]
            for a, b in zip(bounds[:-1], bounds[1:]):
                p = seg[a:b]
                if len(p) >= 3 and t[p[-1]] - t[p[0]] >= min_len_s:
                    pieces.append(p)
    return pieces


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=str(ROOT / "data/fullgame/out/work"))
    ap.add_argument("--video", required=True, help="game video (local file)")
    ap.add_argument("--out", default=str(ROOT / "data/reid/fullgame"))
    ap.add_argument("--eval-from", type=float, default=3767.0, help="t >= this (s) is the held-out eval split (Q4)")
    ap.add_argument("--train-step", type=int, default=4, help="take every n-th row of a train piece")
    ap.add_argument("--eval-step", type=int, default=5)
    ap.add_argument("--max-per-piece", type=int, default=24)
    ap.add_argument("--min-h", type=float, default=50)
    ap.add_argument("--max-cover", type=float, default=0.35)
    ap.add_argument("--limit-s", type=float, default=None, help="only t < this (quick tests)")
    args = ap.parse_args()

    t0 = time.perf_counter()
    work = Path(args.work)
    data = load_pass1(work)
    rows = data.rows
    idx_of_k = {int(k): i for i, k in enumerate(data.k)}
    si = np.array([idx_of_k.get(int(k), -1) for k in rows[:, 0]], int)
    t = data.t[si].astype(np.float64)
    cov = coverage(rows, si)
    z = np.load(work / "team_appearance.npz")
    desc = z["desc"].astype(np.float32)
    sd = np.sqrt(np.clip(desc, 0, None))
    code = np.full(len(rows), -1, np.int16)
    code_conf = np.full(len(rows), np.nan, np.float32)
    jn = work / "jersey_numbers.npz"
    if jn.exists():
        zj = np.load(jn)
        code, code_conf = zj["code"].astype(np.int16), zj["conf"].astype(np.float32)
    print(f"loaded {len(rows)} rows in {time.perf_counter() - t0:.1f}s", flush=True)

    pieces = build_pieces(rows, t, cov, sd, args.min_h, 0.5, args.max_cover, 1.0, 1.5)
    if args.limit_s:
        pieces = [p for p in pieces if t[p[-1]] < args.limit_s]
    print(f"{len(pieces)} clean pieces; median len {np.median([t[p[-1]] - t[p[0]] for p in pieces]):.1f}s",
          flush=True)

    sel_rows, sel_piece, split = [], [], []
    piece_info = []
    for pi, p in enumerate(pieces):
        ev = t[p[0]] >= args.eval_from
        if (not ev) and t[p[-1]] >= args.eval_from:
            continue  # straddles the split
        take = p[:: (args.eval_step if ev else args.train_step)]
        if not ev and len(take) > args.max_per_piece:
            take = take[np.linspace(0, len(take) - 1, args.max_per_piece).round().astype(int)]
        sel_rows.extend(take.tolist())
        sel_piece.extend([pi] * len(take))
        split.extend([int(ev)] * len(take))
        piece_info.append((pi, rows[p[0], 7], t[p[0]], t[p[-1]], len(p)))
    # confident jersey reads not already in the set (for number-identity eval / training)
    have = set(sel_rows)
    extra = np.flatnonzero((code >= 0) & (code_conf >= 0.9) & (rows[:, 4] - rows[:, 2] >= args.min_h))
    if args.limit_s:
        extra = extra[t[extra] < args.limit_s]
    extra = [int(r) for r in extra if int(r) not in have]
    sel_rows.extend(extra)
    sel_piece.extend([-1] * len(extra))
    split.extend([int(t[r] >= args.eval_from) for r in extra])

    sel_rows = np.array(sel_rows, np.int64)
    order = np.argsort(t[sel_rows], kind="stable")
    sel_rows = sel_rows[order]
    sel_piece = np.array(sel_piece, np.int32)[order]
    split = np.array(split, np.int8)[order]
    n = len(sel_rows)
    print(f"{n} crops ({(split == 0).sum()} train, {(split == 1).sum()} eval, {len(extra)} extra number reads)",
          flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    crops = np.lib.format.open_memmap(out / "crops.tmp.npy", mode="w+", dtype=np.uint8, shape=(n, CROP_H, CROP_W, 3))
    got = np.zeros(n, bool)

    cap = cv2.VideoCapture(args.video)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    half = 0.5 / fps
    st = t[sel_rows]
    uniq_t, first = np.unique(st, return_index=True)
    bounds = np.r_[first[1:], n]
    j = 0
    td = time.perf_counter()
    start = float(data.signature.get("start_s") or 0.0)
    seek_to = max(0.0, min(float(uniq_t[0]), start) - 2.0)
    if seek_to > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, seek_to * 1000.0)
    while j < len(uniq_t):
        if not cap.grab():
            break
        ft = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if ft + half < uniq_t[j]:
            continue
        while j < len(uniq_t) and uniq_t[j] + half < ft:
            j += 1
        if j >= len(uniq_t) or abs(ft - uniq_t[j]) > half:
            continue
        ok, img = cap.retrieve()
        if ok:
            for q in range(first[j], bounds[j]):
                c = crop_box(img, rows[sel_rows[q], 1:5])
                if c is not None:
                    crops[q] = c
                    got[q] = True
        j += 1
        if j % 2000 == 0:
            el = time.perf_counter() - td
            print(f"  {j}/{len(uniq_t)} frames  t={ft:.0f}s  {el:.0f}s", flush=True)
    cap.release()
    crops.flush()
    del crops
    (out / "crops.tmp.npy").replace(out / "crops.npy")
    pinfo = np.array(piece_info, np.float64).reshape(-1, 5)
    np.savez(out / "meta.npz", row=sel_rows, piece=sel_piece, split=split, got=got,
             k=rows[sel_rows, 0].astype(np.int64), t=st, frame=data.frame[si[sel_rows]],
             box=rows[sel_rows, 1:5].astype(np.float32), conf=rows[sel_rows, 5].astype(np.float32),
             bt=rows[sel_rows, 7].astype(np.int64), cover=cov[sel_rows], code=code[sel_rows],
             code_conf=code_conf[sel_rows], desc=desc[sel_rows].astype(np.float16))
    np.savez(out / "pieces.npz", piece=pinfo[:, 0].astype(np.int32), bt=pinfo[:, 1].astype(np.int64),
             t0=pinfo[:, 2], t1=pinfo[:, 3], n_rows=pinfo[:, 4].astype(np.int32))
    info = {"work": str(work), "video": args.video, "n": int(n), "got": int(got.sum()), "eval_from": args.eval_from,
            "pieces": len(pieces), "seconds": round(time.perf_counter() - t0, 1), "args": vars(args)}
    (out / "info.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    print(json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
