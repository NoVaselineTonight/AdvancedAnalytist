"""Labeller B dense GT, keyframe method: a human labels every pass-1 person box on KEYFRAMES (every 1 s) with a global
identity; between keyframes the identity follows the pass-1 ByteTrack tracklet only when it is consistent with both
ends (same tracklet), otherwise the box is taken from the detection that best matches the linear interpolation, or
interpolated (occluded) when no detection matches.

  render <clip_dir> <k0> <k1> <step>   -> scratch/dense/<clip>/kf_<k>.jpg  (people P<n> sorted by x, ball B)
  build  <clip_dir> <k0> <k1> <codes.json> <clip name>  -> data/benchmark/labels/labellerB/players_<clip name>.json

codes.json: {"notes": "...", "kf": {"<k>": {"P1": "<pid>", "P3": "ref-1", "+1": "<pid>@x1,y1,x2,y2", "ball": "B" | "x" |
             "x1,y1,x2,y2"}}}
pid: "<kit colour>-<number>" or "<kit colour>-?<letter>" (number unknown), "ref-<n>"; boxes not listed are not on-court people.
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
MIN_CONF = 0.3


def _load(clip_dir: Path):
    from replay.pipeline import load_pass1
    d = load_pass1(clip_dir / "out" / "work")
    video = Path(json.loads((clip_dir / "out" / "work" / "run.json").read_text(encoding="utf-8"))["video"])
    return d, (video if video.is_absolute() else ROOT / video)


def people_at(d, k):
    r = d.rows[(d.rows[:, 0] == k) & (d.rows[:, 6] == 0) & (d.rows[:, 5] >= MIN_CONF)]
    r = r[np.argsort((r[:, 1] + r[:, 3]) / 2)]
    return {f"P{i + 1}": row for i, row in enumerate(r)}


def ball_at(d, k):
    b = d.rows[(d.rows[:, 0] == k) & (d.rows[:, 6] == 1) & (d.rows[:, 5] >= 0.2)]
    return b[np.argmax(b[:, 5])] if len(b) else None


def render(clip_dir: Path, k0: int, k1: int, step: int):
    import cv2
    d, video = _load(clip_dir)
    kidx = {int(k): i for i, k in enumerate(d.k)}
    want = {int(d.frame[kidx[k]]): k for k in range(k0, k1 + 1, step) if k in kidx}
    out = SCRATCH / clip_dir.name
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    f, last = 0, max(want)
    while f <= last:
        if f in want:
            ok, img = cap.read()
            if not ok:
                break
            k = want[f]
            for name, row in people_at(d, k).items():
                x1, y1, x2, y2 = (int(c) for c in row[1:5])
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
                cv2.putText(img, name[1:], (x1, max(12, y1 - 3)), 0, 0.55, (0, 255, 255), 2)
            b = ball_at(d, k)
            if b is not None:
                x1, y1, x2, y2 = (int(c) for c in b[1:5])
                cv2.rectangle(img, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), (255, 0, 255), 2)
            cv2.putText(img, f"k{k}", (10, 40), 0, 1.2, (255, 255, 255), 3)
            cv2.imwrite(str(out / f"kf_{k:04d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        elif not cap.grab():
            break
        f += 1
    print("rendered", len(want), "keyframes ->", out)


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def build(clip_dir: Path, k0: int, k1: int, codes_p: Path, clip: str):
    d, _ = _load(clip_dir)
    C = json.loads(codes_p.read_text(encoding="utf-8"))
    kfs = sorted(int(k) for k in C["kf"])
    kidx = {int(k): i for i, k in enumerate(d.k)}
    # identity -> keyframe -> (box, tracklet or None)
    at: dict[str, dict[int, tuple[list, int | None]]] = defaultdict(dict)
    balls: dict[int, tuple[list, bool]] = {}
    for k in kfs:
        ppl = people_at(d, k)
        for ref, code in C["kf"][str(k)].items():
            if ref == "ball":
                if code == "B":
                    b = ball_at(d, k)
                    if b is not None:
                        balls[k] = ([round(float(c), 1) for c in b[1:5]], False)
                elif code != "x":
                    balls[k] = ([float(c) for c in code.split(",")], False)
                continue
            if ref.startswith("+"):
                pid, box = code.split("@")
                at[pid][k] = ([float(c) for c in box.split(",")], None)
            else:
                row = ppl[ref]
                at[code][k] = ([round(float(c), 1) for c in row[1:5]], int(row[7]))
    frames_people: dict[int, list] = defaultdict(list)
    ks = [k for k in range(k0, k1 + 1) if k in kidx]
    rows_k = defaultdict(list)
    for row in d.rows[(d.rows[:, 6] == 0) & (d.rows[:, 0] >= k0) & (d.rows[:, 0] <= k1)]:
        rows_k[int(row[0])].append(row)

    def put(k, pid, box, interp, src):
        kind = "referee" if pid.startswith("ref") else "player"
        frames_people[k].append({"box": [round(c, 1) for c in box], "kind": kind,
                                 "team": None if kind == "referee" else pid.split("-")[0], "pid": pid,
                                 "occluded": interp, "interp": interp, "on_court": True, "src": src})

    for pid, seen in at.items():
        sk = sorted(seen)
        for k in sk:
            put(k, pid, seen[k][0], False, "kf")
        for a, b in zip(kfs, kfs[1:]):
            ina, inb = a in seen, b in seen
            if not (ina or inb):
                continue
            for k in [x for x in ks if a < x < b]:
                w = (k - a) / (b - a)
                ta = seen[a][1] if ina else None
                tb = seen[b][1] if inb else None
                cand = rows_k.get(k, [])
                pick = None
                if ina and inb:
                    exp = [(1 - w) * p + w * q for p, q in zip(seen[a][0], seen[b][0])]
                    if ta is not None and ta == tb:
                        same = [r for r in cand if int(r[7]) == ta]
                        pick = same[0] if same else None
                    if pick is None:
                        pref = ta if w < 0.5 else tb
                        best = max(cand, key=lambda r: (_iou(r[1:5], exp) + (0.2 if int(r[7]) == pref else 0)),
                                   default=None)
                        if best is not None and _iou(best[1:5], exp) >= 0.35:
                            pick = best
                    if pick is not None:
                        put(k, pid, [float(c) for c in pick[1:5]], False, f"T{int(pick[7])}")
                    else:
                        put(k, pid, exp, True, "interp")
                else:  # seen at one end only: follow its tracklet while it lasts (entering / leaving the view)
                    t = ta if ina else tb
                    same = [r for r in cand if t is not None and int(r[7]) == t]
                    if same:
                        put(k, pid, [float(c) for c in same[0][1:5]], False, f"T{t}")
    # ball: keyframe decisions + the detector in between when it stays near the interpolated path
    ball_out = {}
    for k, (box, interp) in balls.items():
        ball_out[k] = (box, interp)
    bk = sorted(balls)
    for a, b in zip(bk, bk[1:]):
        for k in [x for x in ks if a < x < b]:
            w = (k - a) / (b - a)
            exp = [(1 - w) * p + w * q for p, q in zip(balls[a][0], balls[b][0])]
            det = ball_at(d, k)
            ec = ((exp[0] + exp[2]) / 2, (exp[1] + exp[3]) / 2)
            if det is not None:
                dc = ((det[1] + det[3]) / 2, (det[2] + det[4]) / 2)
                if np.hypot(dc[0] - ec[0], dc[1] - ec[1]) < 120:
                    ball_out[k] = ([round(float(c), 1) for c in det[1:5]], False)
                    continue
            if b - a <= 10:
                ball_out[k] = ([round(c, 1) for c in exp], True)
    frames = []
    for k in ks:
        i = kidx[k]
        fr = {"frame": int(d.frame[i]), "t": round(float(d.t[i]), 3), "k": k, "status": "done", "complete": True,
              "keyframe": k in C["kf"], "people": frames_people.get(k, [])}
        if k in ball_out:
            fr["ball"] = {"box": ball_out[k][0], "interp": ball_out[k][1]}
        frames.append(fr)
    doc = {"format": "aa-bench-players@1", "dense": True, "clip": clip,
           "labeller": "labeller B (keyframes every 1 s by eye + tracklet propagation)",
           "k_range": [k0, k1], "notes": C.get("notes", ""), "frames": frames}
    out = ROOT / "data/benchmark/labels/labellerB" / f"players_{clip}.json"
    out.write_text(json.dumps(doc, indent=0), encoding="utf-8")
    n = sum(len(f["people"]) for f in frames)
    ni = sum(1 for f in frames for p in f["people"] if p["interp"])
    print(out, len(frames), "frames,", n, "boxes (", ni, "interpolated ),", len(at), "identities,", len(ball_out), "ball")


if __name__ == "__main__":
    cmd, clip_dir = sys.argv[1], Path(sys.argv[2])
    if cmd == "render":
        render(clip_dir, int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
    else:
        build(clip_dir, int(sys.argv[3]), int(sys.argv[4]), Path(sys.argv[5]), sys.argv[6])
