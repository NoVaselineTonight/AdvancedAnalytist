"""Hard-example mining for the dedicated ball detector (complements tools/train_balldet.py "build").

  python tools/balldet_mine.py [--per-venue 300] [--venues hs_hazen,ncaa] [--force]

Writes data/balldet/mined/<venue>/ (never data/balldet/ds/, which the builder owns):
  frames/<clip>__<frame>.jpg + frames/<clip>__<frame>_prev.jpg  (native resolution; prev = previous analysed frame)
  labels.json  COCO-style: images [{id, file_name, prev_file_name, width, height, venue, clip, frame, t}],
               annotations [{image_id, bbox [x,y,w,h], category_id 1, kind, conf, tracklet_len}]  (verified balls),
               neg_regions [{image_id, bbox, kind}]  (hard negatives: head | rim | static | rejected)
plus data/balldet/mined/splits.json (whole held-out venues + held-out koristv clips) and README.md (count table).

Positives are VERIFIED only: link_ball path nodes (the ball tracker's chosen detection) in tracklets with >= MIN_TL
sightings. Hard kinds: "rim" (inside a made-basket window from the score-bar labels, or within 3 rim radii of the
projected rim), "hand" (centre inside a person box above the hips), "flight" (fast in camera-stabilised pixels and
not in a person box). Plain positives are not mined (the builder has them) except those sharing a saved frame.
Hard negatives: heads (top of person boxes far from the ball), rims (projected rim ellipse without the ball),
static (rejected candidates that stay at one stabilised spot for >= 3 s: logos, ball rack, lights) and other
rejected candidates (conf >= 0.5, >= 60 px from the ball).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "data" / "balldet" / "mined"
MIN_TL = 4
HOLDOUT_VENUES = ("hs_hazen", "ven_superliga")
HOLDOUT_CLIPS = ("koristv/p4400", "koristv/b3min", "koristv/c300")
SCORE_FILES = {"koristv/c300": "c300", "koristv/c900": "c900", "koristv/c1550": "c1550", "koristv/c2000": "c2000",
               "koristv/c2800": "c2800", "koristv/c3350": "c3350", "koristv/c4000": "c4000", "koristv/c4850": "c4850",
               "koristv/b3min": "bench3min", "koristv/p4400": "clip-4400-40428b", "koristv/fullgame": "fullgame"}


def _sources() -> list[dict]:
    spec = importlib.util.spec_from_file_location("train_balldet", ROOT / "tools" / "train_balldet.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.sources()


def _calibration(work: Path) -> Path | None:
    for d in (work.parent.parent, work.parent):
        for name in ("clip.calibration.json", "fullgame.calibration.json", "calibration.json"):
            if (d / name).exists():
                return d / name
    return None


def _made_windows(name: str) -> list[tuple[float, float]]:
    f = ROOT / "data" / "benchmark" / "labels" / f"scores_{SCORE_FILES.get(name, '-')}.json"
    if not f.exists():
        return []
    ev = json.loads(f.read_text(encoding="utf-8")).get("events") or []
    return [(float(e["t_bar"]) - 9.0, float(e["t_bar"]) - 1.0) for e in ev if "t_bar" in e]


def _apply(M, p):
    v = M @ np.array([p[0], p[1], 1.0])
    return v[:2] / v[2]


def mine_source(src: dict, rng) -> tuple[list[dict], object]:
    """Candidate frames of one clip: [{i, t, balls: [(box, kind, conf, tl)], negs: [(box, kind)], hard: int}]."""
    from replay import calibration as cm
    from replay import court
    from replay.ball_track import link_ball
    from replay.config import BallConfig, Config
    from replay.detection import BALL, PERSON
    from replay.pipeline import load_pass1
    from replay.scoring import rim_tracks

    data = load_pass1(Path(src["work"]))
    n = len(data.k)
    fps = float(data.signature["process_fps"])
    G, seg = cm._segment_transforms(data.motion, n)
    idx = {int(k): i for i, k in enumerate(data.k)}
    rows = data.rows
    ri = np.array([idx[int(k)] for k in rows[:, 0]], int)
    balls = rows[:, 6] == BALL
    ball_rows = [np.zeros((0, 5)) for _ in range(n)]
    for i in np.unique(ri[balls]):
        ball_rows[int(i)] = rows[balls & (ri == i)][:, [1, 2, 3, 4, 5]]
    persons = {int(i): rows[(rows[:, 6] == PERSON) & (ri == i) & (rows[:, 5] >= 0.3)][:, 1:5]
               for i in np.unique(ri[rows[:, 6] == PERSON])}
    tr = link_ball(n, ball_rows, G, seg, fps, BallConfig(min_conf=0.3))
    tl_n: dict[int, int] = {}
    for i in range(n):
        if tr.observed[i]:
            tl_n[int(tr.tracklet[i])] = tl_n.get(int(tr.tracklet[i]), 0) + 1

    rims = None
    cal = _calibration(Path(src["work"]))
    if cal is not None:
        try:
            cf = cm.CalibrationFile.load(cal)
            if cf.court:
                court.set_dimensions(cf.court["length"], cf.court["width"])
            res = cm.resolve(data.t, data.motion, cf, Config().calibration)
            rims, _ = rim_tracks(res.H, data.t, cf.frame_size, Config().scoring, G, seg)
        except Exception as e:  # noqa: BLE001 - rims are optional
            print("  no rims:", e)
    windows = _made_windows(src["name"])

    # static rejected candidates: stabilised position repeated over >= 3 s (logos, ball rack, lights)
    rej = []  # (i, box, conf, stabilised centre)
    for i in range(n):
        ch = tr.boxes[i] if tr.observed[i] else None
        for b in ball_rows[i]:
            if b[4] < 0.3:
                continue
            c = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
            if ch is not None and np.hypot(c[0] - (ch[0] + ch[2]) / 2, c[1] - (ch[1] + ch[3]) / 2) < 40:
                continue
            rej.append((i, b, _apply(G[i], c), seg[i]))
    static_ids = set()
    if rej:
        P = np.array([r[2] for r in rej])
        S = np.array([r[3] for r in rej])
        I = np.array([r[0] for r in rej])
        for a in range(len(rej)):
            d = max(6.0, 0.7 * (rej[a][1][2] - rej[a][1][0]))
            near = (S == S[a]) & (np.hypot(*(P - P[a]).T) < d)
            if near.sum() >= 5 and (I[near].max() - I[near].min()) / fps >= 3.0:
                static_ids.add(a)

    frames: list[dict] = []
    for i in range(1, n):
        item = {"i": i, "t": float(data.t[i]), "balls": [], "negs": [], "hard": 0}
        ball_c = None
        if tr.observed[i] and tl_n.get(int(tr.tracklet[i]), 0) >= MIN_TL:
            b = tr.boxes[i]
            c = np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
            d = max(1.0, 0.5 * ((b[2] - b[0]) + (b[3] - b[1])))
            ball_c = c
            kind = "plain"
            in_person = any(p[0] <= c[0] <= p[2] and p[1] <= c[1] <= p[1] + 0.65 * (p[3] - p[1])
                            for p in persons.get(i, []))
            rim_near = False
            if rims is not None:
                for side in "LR":
                    r = rims[side][i]
                    if r is not None and np.hypot(c[0] - r[0], c[1] - r[1]) < 3 * max(r[2], 4):
                        rim_near = True
            in_window = any(a <= data.t[i] <= b_ for a, b_ in windows)
            speed = 0.0
            for j in (i - 1, i + 1):
                if 0 <= j < n and tr.observed[j] and tr.tracklet[j] == tr.tracklet[i] and seg[j] == seg[i]:
                    bj = tr.boxes[j]
                    cj = _apply(G[j], ((bj[0] + bj[2]) / 2, (bj[1] + bj[3]) / 2))
                    speed = max(speed, float(np.hypot(*(cj - _apply(G[i], c)))))
            if rim_near or in_window:
                kind = "rim"
            elif in_person:
                kind = "hand"
            elif speed > 2.0 * d:
                kind = "flight"
            item["balls"].append(([float(v) for v in b[:4]], kind, float(b[4]), tl_n[int(tr.tracklet[i])]))
            item["hard"] = int(kind != "plain")
        # hard negatives
        for p in persons.get(i, []):
            w, h = p[2] - p[0], p[3] - p[1]
            if h < 40:
                continue
            hb = [p[0] + 0.25 * w, p[1], p[2] - 0.25 * w, p[1] + 0.16 * h]
            hc = ((hb[0] + hb[2]) / 2, (hb[1] + hb[3]) / 2)
            if ball_c is None or np.hypot(hc[0] - ball_c[0], hc[1] - ball_c[1]) > 2.5 * h * 0.16 + 20:
                item["negs"].append(([float(v) for v in hb], "head"))
        if rims is not None:
            for side in "LR":
                r = rims[side][i]
                if r is None or not (0 <= r[0] < 4000):
                    continue
                if ball_c is not None and np.hypot(r[0] - ball_c[0], r[1] - ball_c[1]) < 3 * r[2]:
                    continue
                item["negs"].append(([r[0] - r[2], r[1] - 0.5 * r[2], r[0] + r[2], r[1] + 0.5 * r[2]], "rim"))
        frames.append(item)
    by_i = {f["i"]: f for f in frames}
    for a, (i, b, _, _) in enumerate(rej):
        if i in by_i:
            kind = "static" if a in static_ids else ("rejected" if b[4] >= 0.5 else None)
            if kind:
                by_i[i]["negs"].append(([float(v) for v in b[:4]], kind))
    return frames, data


def choose(frames: list[dict], k: int, rng) -> list[dict]:
    """Up to k frames: hard positives first (rim / flight / hand balanced, 70 %), then frames with static/rim
    negatives."""
    pos = {kd: [f for f in frames if f["balls"] and f["balls"][0][1] == kd] for kd in ("rim", "flight", "hand")}
    quota, out = int(0.7 * k), []
    left = dict(pos)
    while quota > 0 and any(left.values()):
        kinds = [kd for kd in left if left[kd]]
        share = max(1, quota // len(kinds))
        for kd in kinds:
            xs = left[kd]
            sel = sorted(rng.choice(len(xs), min(share, len(xs), quota), replace=False)) if quota > 0 else []
            out += [xs[j] for j in sel]
            quota -= len(sel)
            left[kd] = [x for j, x in enumerate(xs) if j not in set(sel)]
    have = {f["i"] for f in out}
    negf = [f for f in frames if f["i"] not in have and any(kd in ("static", "rim") for _, kd in f["negs"])]
    rest = k - len(out)
    if negf and rest > 0:
        sel = rng.choice(len(negf), min(rest, len(negf)), replace=False)
        out += [negf[j] for j in sorted(sel)]
    return sorted(out, key=lambda f: f["i"])


def main() -> None:
    from replay.redetect import iter_frames

    ap = argparse.ArgumentParser()
    ap.add_argument("--per-venue", type=int, default=300)
    ap.add_argument("--venues", default="")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    srcs = _sources()
    venues = sorted({s["venue"] for s in srcs})
    if a.venues:
        venues = [v for v in venues if v in a.venues.split(",")]
    table = json.loads((OUT / "counts.json").read_text(encoding="utf-8")) if (OUT / "counts.json").exists() else {}
    for v in venues:
        vd = OUT / v
        if (vd / "labels.json").exists() and not a.force:
            print(v, "exists (use --force)")
            continue
        (vd / "frames").mkdir(parents=True, exist_ok=True)
        vs = [s for s in srcs if s["venue"] == v]
        k_each = max(20, a.per_venue // max(1, len(vs)))
        doc = {"venue": v, "images": [], "annotations": [], "neg_regions": [],
               "categories": [{"id": 1, "name": "ball"}]}
        for s in vs:
            print(v, s["name"], flush=True)
            frames, data = mine_source(s, rng)
            sel = choose(frames, k_each, rng)
            want = sorted({f["i"] for f in sel} | {f["i"] - 1 for f in sel})
            need = {f["i"]: f for f in sel}
            prev = {}
            clip = s["name"].split("/", 1)[1]
            for i, img in iter_frames(Path(s["video"]), data, want):
                prev[i] = img
                prev.pop(i - 2, None)
                f = need.get(i)
                if f is None or (i - 1) not in prev:
                    continue
                fn = f"{clip}__{int(data.frame[i])}"
                cv2.imwrite(str(vd / "frames" / f"{fn}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
                cv2.imwrite(str(vd / "frames" / f"{fn}_prev.jpg"), prev[i - 1], [cv2.IMWRITE_JPEG_QUALITY, 88])
                iid = len(doc["images"]) + 1
                H, W = img.shape[:2]
                doc["images"].append({"id": iid, "file_name": f"frames/{fn}.jpg", "prev_file_name": f"frames/{fn}_prev.jpg",
                                      "width": W, "height": H, "venue": v, "clip": s["name"],
                                      "frame": int(data.frame[i]), "prev_frame": int(data.frame[i - 1]), "t": f["t"],
                                      "holdout": v in HOLDOUT_VENUES or s["name"] in HOLDOUT_CLIPS})
                for b, kd, cf, tl in f["balls"]:
                    doc["annotations"].append({"id": len(doc["annotations"]) + 1, "image_id": iid,
                                               "bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]], "category_id": 1,
                                               "kind": kd, "conf": round(cf, 3), "tracklet_len": int(tl),
                                               "source": "link_ball"})
                for b, kd in f["negs"]:
                    x1, y1 = max(0.0, b[0]), max(0.0, b[1])
                    x2, y2 = min(float(W), b[2]), min(float(H), b[3])
                    if x2 - x1 >= 3 and y2 - y1 >= 3:
                        doc["neg_regions"].append({"image_id": iid, "bbox": [x1, y1, x2 - x1, y2 - y1], "kind": kd})
        (vd / "labels.json").write_text(json.dumps(doc), encoding="utf-8")
        cnt = {"images": len(doc["images"]), "holdout_images": sum(im["holdout"] for im in doc["images"])}
        for an in doc["annotations"]:
            cnt["pos_" + an["kind"]] = cnt.get("pos_" + an["kind"], 0) + 1
        for nr in doc["neg_regions"]:
            cnt["neg_" + nr["kind"]] = cnt.get("neg_" + nr["kind"], 0) + 1
        table[v] = cnt
        (OUT / "counts.json").write_text(json.dumps(table, indent=1), encoding="utf-8")
        print(v, cnt, flush=True)
    (OUT / "splits.json").write_text(json.dumps({"holdout_venues": list(HOLDOUT_VENUES),
                                                 "holdout_clips": list(HOLDOUT_CLIPS),
                                                 "note": "whole venues held out; koristv clips with hand ball labels"},
                                                indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
