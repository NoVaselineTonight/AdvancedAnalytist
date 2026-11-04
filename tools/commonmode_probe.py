"""Common-mode calibration jitter probe (TRACKING_FIX_PLAN Task 5, step 1).

Is there a per-sample shift shared by ALL people (calibration jitter) in dead-ball phases, beyond what independent
player motion gives? For each benchmark output folder: per-sample common-mode step = |median over tracks of the
displacement vector between consecutive samples| (seen rows only, >= 5 tracks), on DEAD samples
(court_tracker.MotionPrior); null = the same with each track's displacements shuffled in time.
Two position sources: 'raw' = box foot point projected with the run's per-sample homography (H.npz, the
measurement), 'out' = the exported (smoothed) position.

usage: python tools/commonmode_probe.py <run>/out [<run>/out ...]
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from replay.court_tracker import DEAD, MotionPrior  # noqa: E402


def load(out: Path):
    rows = list(csv.DictReader(open(out / "tracking.csv", encoding="utf-8")))
    frames = sorted({int(r["frame"]) for r in rows})
    idx = {f: i for i, f in enumerate(frames)}
    t = np.zeros(len(frames))
    for r in rows:
        t[idx[int(r["frame"])]] = float(r["timestamp"])
    def is_seen(r):  # older outputs have no 'source' column: measured = not interpolated
        return r["source"] == "seen" if "source" in r else r.get("interpolated") == "False"

    seen = [r for r in rows if r["object_type"] in ("player", "referee") and is_seen(r)
            and r["img_x1"] not in ("", None)]
    i = np.array([idx[int(r["frame"])] for r in seen])
    tid = np.array([int(r["track_id"]) for r in seen])
    out_xy = np.array([[float(r["x_m"]), float(r["y_m"])] for r in seen])
    foot = np.array([[(float(r["img_x1"]) + float(r["img_x2"])) / 2, float(r["img_y2"])] for r in seen])
    raw = np.full_like(out_xy, np.nan)
    hf = out.parent / "H.npz"
    if hf.exists():
        H = np.load(hf)["H"]
        if len(H) == len(frames):
            for k in range(len(seen)):
                h = H[i[k]]
                if np.isfinite(h).all():
                    q = h @ np.array([foot[k, 0], foot[k, 1], 1.0])
                    raw[k] = q[:2] / q[2]
    return t, i, tid, out_xy, raw


def steps(n, i, tid, xy, rng=None, min_tracks=5, pool=None):
    """per-sample |median displacement| (nan when < min_tracks); rng -> shuffled-in-time null (each track's
    displacements permuted among its samples inside ``pool`` (bool per sample, e.g. the DEAD phase) only, so the
    null keeps the phase's speed distribution)."""
    per = [[] for _ in range(n)]
    for k in np.unique(tid):
        m = np.flatnonzero((tid == k) & np.isfinite(xy[:, 0]))
        if len(m) < 3:
            continue
        o = m[np.argsort(i[m])]
        ii, p = i[o], xy[o]
        ok = np.diff(ii) == 1
        d, at = np.diff(p, axis=0)[ok], ii[1:][ok]
        if rng is not None and len(d) > 1:
            q = np.flatnonzero(pool[at]) if pool is not None else np.arange(len(d))
            if len(q) > 1:
                d = d.copy()
                d[q] = d[q[rng.permutation(len(q))]]
        for a, v in zip(at, d):
            per[a].append(v)
    return np.array([np.linalg.norm(np.median(np.array(v), axis=0)) if len(v) >= min_tracks else np.nan
                     for v in per])


def main(dirs):
    print(f"{'clip':10s} {'src':4s} {'dead_samples':>12s} {'cm_median':>9s} {'cm_p90':>7s} {'null_median':>11s} "
          f"{'null_p90':>8s} {'ratio':>6s}")
    res = {}
    for d in dirs:
        out = Path(d)
        clip = out.parent.name
        t, i, tid, out_xy, raw = load(out)
        n = len(t)
        mp = MotionPrior.from_detections(t, i, out_xy, tid)
        dead = mp.phase == DEAD
        for src, xy in (("raw", raw), ("out", out_xy)):
            cm = steps(n, i, tid, xy)
            nulls = [steps(n, i, tid, xy, np.random.default_rng(s), pool=dead) for s in range(5)]
            sel = dead & np.isfinite(cm)
            if sel.sum() < 5:
                print(f"{clip:10s} {src:4s} {int(sel.sum()):12d}  (too few dead samples)")
                continue
            nv = np.concatenate([nl[dead & np.isfinite(nl)] for nl in nulls])
            cmm, nm = float(np.median(cm[sel])), float(np.median(nv))
            print(f"{clip:10s} {src:4s} {int(sel.sum()):12d} {cmm:9.4f} {np.percentile(cm[sel], 90):7.4f} "
                  f"{nm:11.4f} {np.percentile(nv, 90):8.4f} {cmm / max(nm, 1e-9):6.2f}")
            res[(clip, src)] = (cmm, nm)
    return res


if __name__ == "__main__":
    main(sys.argv[1:])
