"""Label-quality filter for data/balldet/mined/<venue>/labels.json (run after tools/balldet_mine.py).

  python tools/balldet_filter.py [--static-s 5] [--venues a,b]

Static filter: a mined ball box whose camera-stabilised position keeps getting ball detections for >= STATIC_S
seconds (ball candidates within max(6 px, 0.7 x box size) in >= 50 % of the samples of a static_s window
containing the box; same camera segment) and NOT held (centre inside a person box above the hips (70 %)) is a static object the detector
mistakes for a ball (red wall pads, cones, ball rack): the annotation is moved to ignore_regions (neither ball nor
negative), kind "static_ball". The first run saves the original as labels.raw.json and always filters
from that file, so re-running is idempotent. counts.json is rewritten; the README table is refreshed by
tools/balldet_filter.py --readme.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MINED = ROOT / "data" / "balldet" / "mined"


def _sources() -> dict[str, dict]:
    spec = importlib.util.spec_from_file_location("train_balldet", ROOT / "tools" / "train_balldet.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return {s["name"]: s for s in mod.sources()}


def static_flags(src: dict, frames_boxes: list[tuple[int, list[float]]], static_s: float) -> list[bool]:
    """Per (video frame, bbox xywh): True when the stabilised spot keeps ball detections for >= static_s."""
    from replay import calibration as cm
    from replay.detection import BALL
    from replay.pipeline import load_pass1

    data = load_pass1(Path(src["work"]))
    n = len(data.k)
    fps = float(data.signature["process_fps"])
    G, seg = cm._segment_transforms(data.motion, n)
    idx = {int(k): i for i, k in enumerate(data.k)}
    i_of_frame = {int(f): i for i, f in enumerate(data.frame)}
    from replay.detection import PERSON

    prs = data.rows[(data.rows[:, 6] == PERSON) & (data.rows[:, 5] >= 0.3)]
    b = data.rows[(data.rows[:, 6] == BALL) & (data.rows[:, 5] >= 0.3)]
    bi = np.array([idx[int(k)] for k in b[:, 0]], int)
    c = np.c_[(b[:, 1] + b[:, 3]) / 2, (b[:, 2] + b[:, 4]) / 2, np.ones(len(b))]
    P = np.zeros((len(b), 2))
    for j in range(len(b)):
        v = G[bi[j]] @ c[j]
        P[j] = v[:2] / v[2]
    out = []
    for frame, (x, y, w, h) in frames_boxes:
        i = i_of_frame.get(int(frame))
        if i is None:
            out.append(False)
            continue
        v = G[i] @ np.array([x + w / 2, y + h / 2, 1.0])
        p = v[:2] / v[2]
        near = (seg[bi] == seg[i]) & (np.hypot(*(P - p).T) < max(6.0, 0.7 * max(w, h)))
        # the spot is occupied for most of a static_s window around this sample (a revisit later is not static)
        w = int(round(static_s * fps))
        hit = np.unique(bi[near])
        best = 0.0
        for s0 in range(i - w, i + 1):
            m = ((hit >= s0) & (hit < s0 + w)).sum()
            best = max(best, m / w)
        # a ball HELD still (referee before an inbound / free throw) sits inside a person box above the hips (70 %):
        # not a static object
        cx, cy = x + w / 2, y + h / 2
        pb = prs[prs[:, 0] == data.k[i]][:, 1:5]
        pw, ph = pb[:, 2] - pb[:, 0], pb[:, 3] - pb[:, 1]  # held at the side: up to half a box width outside
        held = bool(len(pb) and ((pb[:, 0] - 0.5 * pw <= cx) & (cx <= pb[:, 2] + 0.5 * pw) & (pb[:, 1] <= cy)
                                 & (cy <= pb[:, 1] + 0.7 * ph) & (ph >= 2.5 * max(w, h))).any())
        out.append(bool(best >= 0.5 and not held))
    return out


def count(doc: dict) -> dict:
    cnt = {"images": len(doc["images"]), "holdout_images": sum(bool(im.get("holdout")) for im in doc["images"])}
    for an in doc["annotations"]:
        cnt["pos_" + an["kind"]] = cnt.get("pos_" + an["kind"], 0) + 1
    for nr in doc["neg_regions"]:
        cnt["neg_" + nr["kind"]] = cnt.get("neg_" + nr["kind"], 0) + 1
    if doc.get("ignore_regions"):
        cnt["ignored_static"] = len(doc["ignore_regions"])
    return cnt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--static-s", type=float, default=5.0)
    ap.add_argument("--venues", default="")
    a = ap.parse_args()
    srcs = _sources()
    table = json.loads((MINED / "counts.json").read_text(encoding="utf-8"))
    hc_path = MINED / "handcheck.json"
    hc = json.loads(hc_path.read_text(encoding="utf-8"))["boxes"] if hc_path.exists() else []
    report = {}
    for vd in sorted(d for d in MINED.iterdir() if (d / "labels.json").exists()):
        if a.venues and vd.name not in a.venues.split(","):
            continue
        raw = vd / "labels.raw.json"
        if not raw.exists():
            shutil.copy2(vd / "labels.json", raw)
        doc = json.loads(raw.read_text(encoding="utf-8"))
        ims = {im["id"]: im for im in doc["images"]}
        by_clip: dict[str, list[int]] = {}
        for j, an in enumerate(doc["annotations"]):
            by_clip.setdefault(ims[an["image_id"]]["clip"], []).append(j)
        drop = set()
        for clip, js in by_clip.items():
            fl = static_flags(srcs[clip], [(ims[doc["annotations"][j]["image_id"]]["frame"],
                                            doc["annotations"][j]["bbox"]) for j in js], a.static_s)
            drop.update(j for j, f in zip(js, fl) if f)
        keep = [an for j, an in enumerate(doc["annotations"]) if j not in drop]
        moved = [an for j, an in enumerate(doc["annotations"]) if j in drop]
        doc["annotations"] = keep
        # ignore (neither ball nor negative): visual check 20.10 - 12 of 16 flagged were real static objects (red
        # wall pads, cone, ball rack) but 4 were balls held by a referee whose person box was missed
        doc["ignore_regions"] = [{"image_id": an["image_id"], "bbox": an["bbox"], "kind": "static_ball",
                                  "filtered": f"stabilised spot has ball detections for >= {a.static_s:g} s"}
                                 for an in moved]
        doc["filters"] = {"static_s": a.static_s, "moved_to_neg": len(moved)}
        (vd / "labels.json").write_text(json.dumps(doc), encoding="utf-8")
        table[vd.name] = count(doc)
        # agreement with the hand check
        hit = {"verified_filtered": 0, "wrong_filtered": 0, "verified": 0, "wrong": 0}
        mv = {(ims[an["image_id"]]["file_name"], tuple(round(v, 1) for v in an["bbox"])) for an in moved}
        for b in hc:
            if b["venue"] != vd.name:
                continue
            key = (b["file_name"].split("/", 1)[1], tuple(round(v, 1) for v in b["bbox"]))
            k = "verified" if b["verified"] else "wrong"
            hit[k] += 1
            hit[k + "_filtered"] += key in mv
        report[vd.name] = {"moved": len(moved), **hit}
        print(vd.name, report[vd.name], flush=True)
    (MINED / "counts.json").write_text(json.dumps(table, indent=1), encoding="utf-8")
    (MINED / "filter_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
