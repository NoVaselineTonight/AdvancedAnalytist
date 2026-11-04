"""Measure detection gains from more input resolution vs cost (no pipeline change).

  python tools/res_probe.py [--clips a,b] [--out data/resume/res_probe]

Variants on the benchmark's labelled frames (data/benchmark/labels players_*):
  base     RF-DETR medium ONNX, full frame -> 576x576 (today's pass 1)
  t1x2     same model on 1x2 overlapping tiles (~1.8x linear resolution), merged
  t2x2     2x2 tiles (~2.2x)
  large    RF-DETR large ONNX, full frame -> 704x704 (bigger model AND higher input; no other export is possible
           without downloading weights)
  ballcrop base + one extra 640x360 window around the ball tracker's expected position (link_ball path), i.e. ~2.2x
           linear resolution where the ball should be
Metrics per clip/venue: player recall (labelled player/referee boxes matched IoU>=0.5 by a person box >= 0.3), far-court
recall (labelled boxes in the smallest third of that venue), ball detections per frame (>= 0.3) and NEW ball detections
(no base ball within 20 px; sample crops saved for a visual check), GPU ms per frame.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u > 0 else 0.0


def main() -> None:
    import replay.benchmark as bm
    from replay import calibration as cm
    from replay.ball_track import link_ball
    from replay.config import BallConfig
    from replay.detection import BALL, PERSON, Detections, TiledDetector
    from replay.detection.onnx_backend import ONNXDetector
    from replay.pipeline import load_pass1
    from replay.redetect import iter_frames

    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default="")
    ap.add_argument("--out", default=str(ROOT / "data" / "resume" / "res_probe"))
    ap.add_argument("--max-frames", type=int, default=24)
    a = ap.parse_args()
    out = Path(a.out)
    (out / "newballs").mkdir(parents=True, exist_ok=True)
    med = ONNXDetector("medium", 0.3, device="auto")
    lrg = ONNXDetector("large", 0.3, device="auto")
    variants = {"base": med, "t1x2": TiledDetector(med, 1, 2, 0.15), "t2x2": TiledDetector(med, 2, 2, 0.15),
                "large": lrg}
    warm = np.zeros((720, 1280, 3), np.uint8)
    for det in variants.values():  # first inference compiles the GPU graph: keep it out of the timing
        det.detect(warm)
    reg = bm.clip_registry()
    names = [c for c in reg if "players" in bm.label_files(c) and bm.ready(reg[c])]
    if a.clips:
        names = [c for c in names if c in a.clips.split(",")]
    res = json.loads((out / "results.json").read_text(encoding="utf-8")) if (out / "results.json").exists() else {}
    for c in names:
        if c in res:
            continue
        video = Path(json.loads((reg[c]["src"] / "work/run.json").read_text())["video"])
        if not video.exists():
            print(c, "video missing:", video)
            continue
        lab = bm.load_labels(bm.label_files(c)["players"])
        frs = [f for f in lab["frames"] if f.get("status", "done") == "done"
               and any(p.get("box") for p in f.get("people", []))]
        frs = frs[:: max(1, len(frs) // a.max_frames)][: a.max_frames]
        data = load_pass1(reg[c]["src"] / "work")
        n = len(data.k)
        G, seg = cm._segment_transforms(data.motion, n)
        idx = {int(k): i for i, k in enumerate(data.k)}
        br = data.rows[data.rows[:, 6] == BALL]
        ball_rows = [np.zeros((0, 5)) for _ in range(n)]
        bi = np.array([idx[int(k)] for k in br[:, 0]], int)
        for i in np.unique(bi):
            ball_rows[int(i)] = br[bi == i][:, [1, 2, 3, 4, 5]]
        tr = link_ball(n, ball_rows, G, seg, float(data.signature["process_fps"]), BallConfig(min_conf=0.3))
        fi = np.asarray(data.frame)
        want = {}
        for f in frs:
            i = int(np.argmin(np.abs(fi - int(f["frame"]))))
            if abs(int(fi[i]) - int(f["frame"])) <= 2:
                want[i] = f
        heights = [p["box"][3] - p["box"][1] for f in want.values() for p in f["people"]
                   if p.get("box") and p.get("kind", "player") in ("player", "referee")]
        far_h = float(np.percentile(heights, 33)) if heights else 0
        st = {v: {"hit": 0, "far_hit": 0, "ms": 0.0, "balls": 0, "newballs": 0} for v in list(variants) + ["ballcrop"]}
        st["_n"] = {"frames": 0, "people": 0, "far": 0, "far_h": round(far_h, 1)}
        for i, img in iter_frames(video, data, sorted(want)):
            f = want[i]
            gt = [p["box"] for p in f["people"] if p.get("box") and p.get("kind", "player") in ("player", "referee")]
            st["_n"]["frames"] += 1
            st["_n"]["people"] += len(gt)
            st["_n"]["far"] += sum(b[3] - b[1] <= far_h for b in gt)
            dets = {}
            for v, det in variants.items():
                t0 = time.perf_counter()
                d = det.detect(img)
                st[v]["ms"] += 1000 * (time.perf_counter() - t0)
                dets[v] = d
            # ballcrop: base + a 640x360 window around the expected ball position
            H, W = img.shape[:2]
            t0 = time.perf_counter()
            extra = Detections.empty()
            if np.isfinite(tr.img[i]).all():
                cw, chh = min(W, 640), min(H, 360)
                x0 = int(np.clip(tr.img[i, 0] - cw / 2, 0, W - cw))
                y0 = int(np.clip(tr.img[i, 1] - chh / 2, 0, H - chh))
                extra = med.detect(np.ascontiguousarray(img[y0:y0 + chh, x0:x0 + cw]))
                extra = extra.subset(extra.cls == BALL)
                extra.xyxy = extra.xyxy + np.array([x0, y0, x0, y0], np.float32)
            st["ballcrop"]["ms"] += st["base"]["ms"] / st["_n"]["frames"] + 1000 * (time.perf_counter() - t0)
            dets["ballcrop"] = Detections.concat([dets["base"], extra])
            base_b = dets["base"].xyxy[dets["base"].cls == BALL]
            for v, d in dets.items():
                pb = d.xyxy[d.cls == PERSON]
                for g in gt:
                    ok = any(iou(g, q) >= 0.5 for q in pb)
                    st[v]["hit"] += ok
                    st[v]["far_hit"] += ok and (g[3] - g[1] <= far_h)
                bb = d.xyxy[(d.cls == BALL) & (d.conf >= 0.3)]
                st[v]["balls"] += len(bb)
                for q in bb:
                    cq = ((q[0] + q[2]) / 2, (q[1] + q[3]) / 2)
                    if not any(np.hypot(cq[0] - (r[0] + r[2]) / 2, cq[1] - (r[1] + r[3]) / 2) < 20 for r in base_b):
                        st[v]["newballs"] += 1
                        if v != "base":
                            s = int(max(48, 4 * max(q[2] - q[0], q[3] - q[1])))
                            pad = cv2.copyMakeBorder(img, s, s, s, s, cv2.BORDER_CONSTANT)
                            x, y = int(cq[0] - s / 2) + s, int(cq[1] - s / 2) + s
                            crop = pad[y:y + s, x:x + s].copy()
                            cv2.imwrite(str(out / "newballs" / f"{v}__{c}__{int(data.frame[i])}__{int(cq[0])}_{int(cq[1])}.jpg"),
                                        cv2.resize(crop, (96, 96)))
        nn = max(1, st["_n"]["frames"])
        for v in list(variants) + ["ballcrop"]:
            st[v]["ms"] = round(st[v]["ms"] / nn, 1)
        res[c] = st
        print(c, json.dumps(st), flush=True)
        (out / "results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
