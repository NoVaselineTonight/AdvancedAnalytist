"""Labeller B: build a DENSE identity ground truth (every analysed sample, ~10 fps) from pass-1 ByteTrack tracklets that
a human (me) assigns to identities by looking at contact sheets.

  sheets <clip_dir> <t0> <t1>             -> scratch/dense/<clip>/tl_<page>.jpg (8 crops per tracklet) + tracklets.json
  ball   <clip_dir> <t0> <t1>             -> scratch/dense/<clip>/ball_<page>.jpg (best ball detection per sample)
  build  <clip_dir> <t0> <t1> <assign.json> <clip name> [--gap 2.0]
         -> data/benchmark/labels/labellerB/players_<clip name>.json (dense: true)

assign.json: {"tracklets": {"<bytetrack id>": "<spec>" | [[k_from, k_to, "<spec>"], ...]},
              "ball_reject_k": [k, ...], "notes": "..."}
<spec> = "<team>-<number>" for players (team word = kit colour, e.g. "white-13", "blue-1"), "<team>-?<letter>" for a player
whose number is unknown, "ref-<n>", "bench", "spect", "dup", "x" (ignore / junk). Tracklets not listed are dropped
(= not a person on court). Gaps of the same player up to --gap seconds are filled by linear box interpolation
(occluded: true, interp: true). Ball: the best cls-1 detection per sample (conf >= 0.25) unless rejected; gaps <= 1 s are
interpolated. Frames are the pass-1 sample frames (video frame index), complete = true.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
SCRATCH = Path(r"C:/Users/onnik/AppData/Local/Temp/claude/C--Users-onnik-Desktop/032042cc-6aa6-4bc1-897f-cbf1d1dcb2bf/scratchpad/dense")


def _load(clip_dir: Path):
    from replay.pipeline import load_pass1
    d = load_pass1(clip_dir / "out" / "work")
    video = json.loads((clip_dir / "out" / "work" / "run.json").read_text(encoding="utf-8"))["video"]
    vp = Path(video)
    return d, (vp if vp.is_absolute() else ROOT / vp)


def _window(d, t0, t1):
    ks = np.where((d.t >= t0) & (d.t < t1))[0]
    return set(int(k) for k in d.k[ks]), {int(d.k[i]): (int(d.frame[i]), float(d.t[i])) for i in ks}


def _frames(video: Path, wanted: set[int]):
    import cv2
    cap = cv2.VideoCapture(str(video))
    out, f, last = {}, 0, max(wanted)
    while f <= last:
        if f in wanted:
            ok, img = cap.read()
            if not ok:
                break
            out[f] = img
        elif not cap.grab():
            break
        f += 1
    return out


def sheets(clip_dir: Path, t0: float, t1: float):
    import cv2
    d, video = _load(clip_dir)
    ks, kinfo = _window(d, t0, t1)
    r = d.rows[(d.rows[:, 6] == 0) & np.isin(d.rows[:, 0], list(ks))]
    by = defaultdict(list)
    for row in r:
        by[int(row[7])].append(row)
    tls = {tid: sorted(v, key=lambda x: x[0]) for tid, v in by.items() if tid >= 0 and len(v) >= 2}
    pick = {}
    for tid, v in tls.items():
        idx = np.linspace(0, len(v) - 1, min(8, len(v))).round().astype(int)
        pick[tid] = [v[i] for i in sorted(set(idx))]
    need = {kinfo[int(row[0])][0] for v in pick.values() for row in v}
    imgs = _frames(video, need)
    out = SCRATCH / clip_dir.name
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    strips = []
    for tid in sorted(tls, key=lambda t: tls[t][0][0]):
        v = tls[tid]
        summary[tid] = {"k0": int(v[0][0]), "k1": int(v[-1][0]), "n": len(v),
                        "t0": round(kinfo[int(v[0][0])][1], 2), "t1": round(kinfo[int(v[-1][0])][1], 2)}
        crops = []
        for row in pick[tid]:
            img = imgs.get(kinfo[int(row[0])][0])
            if img is None:
                continue
            x1, y1, x2, y2 = (int(round(c)) for c in row[1:5])
            pad = int(0.15 * (y2 - y1))
            c = img[max(0, y1 - pad):y2 + pad, max(0, x1 - pad):x2 + pad]
            if c.size == 0:
                continue
            c = cv2.resize(c, (max(1, int(c.shape[1] * 150 / c.shape[0])), 150))
            cv2.putText(c, str(int(row[0])), (2, 145), 0, 0.4, (0, 255, 255), 1)
            crops.append(c)
        if not crops:
            continue
        strip = np.hstack(crops)
        head = np.zeros((150, 130, 3), np.uint8)
        cv2.putText(head, f"T{tid}", (4, 40), 0, 0.9, (255, 255, 255), 2)
        cv2.putText(head, f"k{summary[tid]['k0']}-{summary[tid]['k1']}", (4, 80), 0, 0.45, (200, 200, 200), 1)
        cv2.putText(head, f"n={len(v)}", (4, 110), 0, 0.45, (200, 200, 200), 1)
        strip = np.hstack([head, strip])
        strip = strip[:, :1500] if strip.shape[1] > 1500 else np.hstack([strip, np.zeros((150, 1500 - strip.shape[1], 3), np.uint8)])
        strips.append(strip)
    for p in range(0, len(strips), 6):
        cv2.imwrite(str(out / f"tl_{p // 6:02d}.jpg"), np.vstack(strips[p:p + 6]))
    (out / "tracklets.json").write_text(json.dumps(summary, indent=0), encoding="utf-8")
    print(len(strips), "tracklets ->", out, (len(strips) + 5) // 6, "pages")


def _best_ball(d, ks):
    b = d.rows[(d.rows[:, 6] == 1) & np.isin(d.rows[:, 0], list(ks)) & (d.rows[:, 5] >= 0.25)]
    best = {}
    for row in b:
        k = int(row[0])
        if k not in best or row[5] > best[k][5]:
            best[k] = row
    return best


def ball(clip_dir: Path, t0: float, t1: float):
    import cv2
    d, video = _load(clip_dir)
    ks, kinfo = _window(d, t0, t1)
    best = _best_ball(d, ks)
    imgs = _frames(video, {kinfo[k][0] for k in best})
    out = SCRATCH / clip_dir.name
    out.mkdir(parents=True, exist_ok=True)
    tiles = []
    for k in sorted(best):
        row = best[k]
        img = imgs.get(kinfo[k][0])
        if img is None:
            continue
        cx, cy = (row[1] + row[3]) / 2, (row[2] + row[4]) / 2
        x1, y1 = int(max(0, cx - 60)), int(max(0, cy - 60))
        c = img[y1:y1 + 120, x1:x1 + 120].copy()
        c = cv2.resize(c, (120, 120))
        cv2.rectangle(c, (int(row[1] - x1), int(row[2] - y1)), (int(row[3] - x1), int(row[4] - y1)), (0, 255, 0), 1)
        cv2.putText(c, f"{k} {row[5]:.2f}", (2, 12), 0, 0.38, (0, 255, 255), 1)
        tiles.append(c)
    while len(tiles) % 12:
        tiles.append(np.zeros((120, 120, 3), np.uint8))
    rows = [np.hstack(tiles[i:i + 12]) for i in range(0, len(tiles), 12)]
    for p in range(0, len(rows), 8):
        cv2.imwrite(str(out / f"ball_{p // 8:02d}.jpg"), np.vstack(rows[p:p + 8]))
    print(len(best), "ball samples of", len(ks), "->", out)


def _spec_at(spec, k):
    if isinstance(spec, str):
        return spec
    for a, b, s in spec:
        if a <= k <= b:
            return s
    return None


def build(clip_dir: Path, t0: float, t1: float, assign_p: Path, clip: str, gap_s: float = 2.0):
    d, _ = _load(clip_dir)
    ks, kinfo = _window(d, t0, t1)
    A = json.loads(assign_p.read_text(encoding="utf-8"))
    tl = {int(k): v for k, v in A["tracklets"].items()}
    per_k = defaultdict(list)
    sightings = defaultdict(dict)  # pid -> k -> box
    for row in d.rows[(d.rows[:, 6] == 0) & np.isin(d.rows[:, 0], list(ks))]:
        tid, k = int(row[7]), int(row[0])
        if tid not in tl:
            continue
        s = _spec_at(tl[tid], k)
        if not s or s == "x":
            continue
        box = [round(float(c), 1) for c in row[1:5]]
        if s in ("bench", "spect", "dup"):
            per_k[k].append({"box": box, "kind": {"bench": "bench", "spect": "spectator", "dup": "duplicate"}[s],
                             "team": None, "pid": None, "src": f"T{tid}"})
            continue
        if s in sightings and k in sightings[s]:  # two tracklets claim the same person at k: keep the larger box
            old = sightings[s][k]
            if (box[2] - box[0]) * (box[3] - box[1]) <= (old[2] - old[0]) * (old[3] - old[1]):
                per_k[k].append({"box": box, "kind": "duplicate", "team": None, "pid": None, "src": f"T{tid}"})
                continue
            per_k[k].append({"box": old, "kind": "duplicate", "team": None, "pid": None, "src": "dup"})
        sightings[s][k] = box
    kt = sorted(ks)
    kpos = {k: i for i, k in enumerate(kt)}
    max_gap = int(round(gap_s * 10))
    for pid, seen in sightings.items():
        kind = "referee" if pid.startswith("ref") else "player"
        team = None if kind == "referee" else pid.split("-")[0]
        sk = sorted(seen)
        for k in sk:
            per_k[k].append({"box": seen[k], "kind": kind, "team": team, "pid": pid, "occluded": False, "on_court": True})
        for a, b in zip(sk, sk[1:]):
            ia, ib = kpos[a], kpos[b]
            if 1 < ib - ia <= max_gap:
                for i in range(ia + 1, ib):
                    w = (i - ia) / (ib - ia)
                    box = [round((1 - w) * p + w * q, 1) for p, q in zip(seen[a], seen[b])]
                    per_k[kt[i]].append({"box": box, "kind": kind, "team": team, "pid": pid, "occluded": True,
                                         "interp": True, "on_court": True})
    best = _best_ball(d, ks)
    rej = set(A.get("ball_reject_k", []))
    bk = sorted(k for k in best if k not in rej)
    balls = {k: [round(float(c), 1) for c in best[k][1:5]] for k in bk}
    for a, b in zip(bk, bk[1:]):
        ia, ib = kpos[a], kpos[b]
        if 1 < ib - ia <= 10:
            for i in range(ia + 1, ib):
                w = (i - ia) / (ib - ia)
                balls.setdefault(kt[i], [round((1 - w) * p + w * q, 1) for p, q in zip(balls[a], balls[b])])
    frames = []
    for k in kt:
        f, t = kinfo[k]
        fr = {"frame": f, "t": round(t, 3), "k": k, "status": "done", "complete": True, "people": per_k.get(k, [])}
        if k in balls:
            fr["ball"] = {"box": balls[k], "interp": k not in best or k in rej}
        frames.append(fr)
    doc = {"format": "aa-bench-players@1", "dense": True, "clip": clip, "labeller": "labeller B (tracklet assignment + interpolation)",
           "window_s": [t0, t1], "notes": A.get("notes", ""), "frames": frames}
    out = ROOT / "data/benchmark/labels/labellerB" / f"players_{clip}.json"
    out.write_text(json.dumps(doc, indent=0), encoding="utf-8")
    npl = sum(1 for f in frames for p in f["people"] if p["kind"] == "player")
    print(out, len(frames), "frames,", npl, "player boxes,", len(sightings), "identities,", len(balls), "ball boxes")


if __name__ == "__main__":
    cmd, clip_dir, t0, t1 = sys.argv[1], Path(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
    if cmd == "sheets":
        sheets(clip_dir, t0, t1)
    elif cmd == "ball":
        ball(clip_dir, t0, t1)
    elif cmd == "build":
        build(clip_dir, t0, t1, Path(sys.argv[5]), sys.argv[6], float(sys.argv[7]) if len(sys.argv) > 7 else 2.0)
