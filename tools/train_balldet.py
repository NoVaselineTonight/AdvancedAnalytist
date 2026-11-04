"""Dedicated ball detector: dataset, training (CPU, PyTorch), ONNX export, per-venue evaluation (replay/balldet.py).

  python tools/train_balldet.py build   [--out data/balldet/ds] [--max-pos 400]   # crops from all venues
  python tools/train_balldet.py train   [--epochs 30]                              # CPU
  python tools/train_balldet.py export                                             # data/balldet/balldet.onnx
  python tools/train_balldet.py eval    [--model ...]                              # held-out venues + hand labels

Labels (no manual labelling needed for training):
* positives: ball candidates on the ball tracker's chosen path (ball_track.link_ball over pass-1 candidates) in
  tracklets with >= MIN_TRACK observed nodes and conf >= POS_CONF; also lower-confidence nodes (0.3..POS_CONF) inside
  long tracklets - the tracker's temporal evidence teaches the detector balls it was unsure about;
* hard negatives: pass-1 ball candidates the tracker rejected (static: logos, rims, spare balls; or never linked: heads,
  shoes, lights), conf >= 0.3, at least NEG_DIST px from the chosen ball;
* background: random crops of frames without any candidate.
Crops (CROP x CROP at SCALE_H height) are taken around the target with a random offset; the previous analysed frame
gives the motion channel.
Venues: HOLDOUT venues are never used for training; the three hand-labelled koristv clips (p4400, b3min, c300 - 61
ball labels in the ballev scratch) are also held out.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from replay import balldet as bd  # noqa: E402

DS = ROOT / "data" / "balldet" / "ds"
HOLDOUT_VENUES = {"hs_hazen", "ven_superliga"}
HOLDOUT_SOURCES = {"koristv/p4400", "koristv/b3min", "koristv/c300"}
POS_CONF, LOW_CONF, MIN_TRACK, NEG_DIST = 0.5, 0.3, 5, 40.0


def sources() -> list[dict]:
    """[{venue, name, work, video}] for every analysed clip with pass-1 output + video."""
    out = []
    fg = ROOT / "data/fullgame/out/work"
    if (fg / "DONE").exists():
        out.append({"venue": "koristv", "name": "koristv/fullgame", "work": fg,
                    "video": json.loads((fg / "run.json").read_text(encoding="utf-8"))["video"]})
    for d in sorted((ROOT / "data/clips").glob("c*")):
        if (d / "out/work/DONE").exists() and (d / "clip.mp4").exists():
            out.append({"venue": "koristv", "name": f"koristv/{d.name}", "work": d / "out/work", "video": d / "clip.mp4"})
    b3 = ROOT / "data/bench3min"
    if (b3 / "out/work/DONE").exists():
        out.append({"venue": "koristv", "name": "koristv/b3min", "work": b3 / "out/work", "video": b3 / "clip.mp4"})
    p44s = [p for p in sorted((ROOT.parent / "AdvancedAnalytist" / "projects").glob("clip-4400-*"))
            if (p / "results/work/DONE").exists() and (p / "video.mp4").exists()]
    if p44s:
        p44 = p44s[-1]
        out.append({"venue": "koristv", "name": "koristv/p4400", "work": p44 / "results/work", "video": p44 / "video.mp4"})
    for w in sorted((ROOT / "data/external").glob("*/seg_*/out/work")):
        if not (w / "DONE").exists():
            continue
        seg = w.parents[1]
        vids = [p for p in seg.parent.glob(seg.name + ".*") if p.suffix in (".mp4", ".webm", ".mkv")]
        if vids:
            out.append({"venue": seg.parent.name, "name": f"{seg.parent.name}/{seg.name}", "work": w, "video": vids[0]})
    return out


def is_holdout(src: dict) -> bool:
    return src["venue"] in HOLDOUT_VENUES or src["name"] in HOLDOUT_SOURCES


def labels_for(src: dict, max_pos: int, max_neg: int, max_bg: int, rng) -> list[dict]:
    """Auto-labels of one clip: [{i, kind: pos|neg|bg, cx, cy, d}] (original pixels)."""
    from replay import calibration as cm
    from replay.ball_track import link_ball
    from replay.config import BallConfig
    from replay.detection import BALL
    from replay.pipeline import load_pass1
    data = load_pass1(Path(src["work"]))
    n = len(data.k)
    fps = float(data.signature["process_fps"])
    G, seg = cm._segment_transforms(data.motion, n)
    idx = {int(k): i for i, k in enumerate(data.k)}
    rows = data.rows[data.rows[:, 6] == BALL]
    ball_rows = [np.zeros((0, 5)) for _ in range(n)]
    bi = np.array([idx[int(k)] for k in rows[:, 0]], int)
    for i in np.unique(bi):
        ball_rows[int(i)] = rows[bi == i][:, [1, 2, 3, 4, 5]]
    tr = link_ball(n, ball_rows, G, seg, fps, BallConfig(min_conf=LOW_CONF))
    tl_n = {}
    for i in range(n):
        if tr.observed[i]:
            tl_n[tr.tracklet[i]] = tl_n.get(tr.tracklet[i], 0) + 1
    pos, neg, bg = [], [], []

    def ball_at(i):  # the confirmed game ball of this frame (a negative crop may contain it -> it stays a positive)
        if tr.observed[i] and tl_n.get(tr.tracklet[i], 0) >= MIN_TRACK:
            b = tr.boxes[i]
            return {"bx": (b[0] + b[2]) / 2, "by": (b[1] + b[3]) / 2, "bd": 0.5 * ((b[2] - b[0]) + (b[3] - b[1]))}
        if np.isfinite(tr.img[i, 0]):
            return {"bx": float(tr.img[i, 0]), "by": float(tr.img[i, 1]), "bd": -1.0}  # position only: skip crop
        return {}
    for i in range(1, n):
        cand = ball_rows[i]
        if tr.observed[i] and tl_n.get(tr.tracklet[i], 0) >= MIN_TRACK:
            b = tr.boxes[i]
            if b[4] >= LOW_CONF:
                pos.append({"i": i, "kind": "pos", "cx": (b[0] + b[2]) / 2, "cy": (b[1] + b[3]) / 2,
                            "d": 0.5 * ((b[2] - b[0]) + (b[3] - b[1])), "conf": float(b[4])})
        for x1, y1, x2, y2, c in cand:
            if c < LOW_CONF:
                continue
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            if tr.observed[i] and np.hypot(cx - tr.img[i, 0], cy - tr.img[i, 1]) < NEG_DIST:
                continue
            if not np.isfinite(tr.img[i, 0]) or np.hypot(cx - tr.img[i, 0], cy - tr.img[i, 1]) >= NEG_DIST:
                neg.append({"i": i, "kind": "neg", "cx": cx, "cy": cy, "d": 0.0, "conf": float(c), **ball_at(i)})
        if not len(cand) and not np.isfinite(tr.img[i, 0]):
            bg.append({"i": i, "kind": "bg", "cx": -1.0, "cy": -1.0, "d": 0.0, "conf": 0.0})

    def pick(xs, k):
        if len(xs) <= k:
            return xs
        sel = rng.choice(len(xs), k, replace=False)
        return [xs[j] for j in sorted(sel)]
    return pick(pos, max_pos) + pick(neg, max_neg) + pick(bg, max_bg), data


def build(args) -> None:
    from replay.redetect import iter_frames
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    summary = {}
    for src in sources():
        name = src["name"].replace("/", "__")
        f = out / f"{name}.npz"
        if f.exists() and not args.force:
            z = np.load(f)
            summary[src["name"]] = {"venue": src["venue"], "n": int(len(z["kind"])), "cached": True}
            continue
        t0 = time.time()
        big = src["name"] == "koristv/fullgame"
        labs, data = labels_for(src, args.max_pos * (6 if big else 1), args.max_neg * (6 if big else 1),
                                args.max_bg * (6 if big else 1), rng)
        want = sorted({l["i"] for l in labs} | {l["i"] - 1 for l in labs})
        need = {l["i"] for l in labs}
        frames: dict[int, np.ndarray] = {}
        imgs, prevs, tg, kinds, confs = [], [], [], [], []
        by_i: dict[int, list[dict]] = {}
        for l in labs:
            by_i.setdefault(l["i"], []).append(l)
        for i, img in iter_frames(Path(src["video"]), data, want):
            s = bd.SCALE_H / img.shape[0]
            im = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s != 1 else img
            frames[i] = im
            frames.pop(i - 2, None)
            if i not in need or (i - 1) not in frames:
                continue
            pv = cv2.cvtColor(frames[i - 1], cv2.COLOR_BGR2GRAY)
            H, W = im.shape[:2]
            for l in by_i[i]:
                if l["kind"] == "bg":
                    cx, cy = rng.uniform(bd.CROP / 2, W - bd.CROP / 2), rng.uniform(bd.CROP / 2, H - bd.CROP / 2)
                else:
                    cx, cy = l["cx"] * s, l["cy"] * s
                ox, oy = (rng.uniform(-60, 60, 2) if l["kind"] != "bg" else (0, 0))
                x0 = int(np.clip(cx + ox - bd.CROP / 2, 0, max(0, W - bd.CROP)))
                y0 = int(np.clip(cy + oy - bd.CROP / 2, 0, max(0, H - bd.CROP)))
                c = im[y0:y0 + bd.CROP, x0:x0 + bd.CROP]
                p = pv[y0:y0 + bd.CROP, x0:x0 + bd.CROP]
                if c.shape[:2] != (bd.CROP, bd.CROP):
                    continue
                if l["kind"] == "pos":
                    t_ = [cx - x0, cy - y0, l["d"] * s]
                else:
                    t_ = [-1, -1, 0]
                    if "bx" in l:
                        bx_, by_ = l["bx"] * s - x0, l["by"] * s - y0
                        if -8 <= bx_ <= bd.CROP + 8 and -8 <= by_ <= bd.CROP + 8:
                            if l["bd"] < 0:  # the ball is (probably) in this crop but its box is unknown
                                continue
                            t_ = [bx_, by_, l["bd"] * s]
                imgs.append(c)
                prevs.append(p)
                tg.append(t_)
                kinds.append({"pos": 0, "neg": 1, "bg": 2}[l["kind"]])
                confs.append(l["conf"])
        np.savez_compressed(f, img=np.array(imgs, np.uint8).reshape(-1, bd.CROP, bd.CROP, 3),
                            prev=np.array(prevs, np.uint8).reshape(-1, bd.CROP, bd.CROP),
                            target=np.array(tg, np.float32).reshape(-1, 3), kind=np.array(kinds, np.int8),
                            conf=np.array(confs, np.float32), venue=np.array(src["venue"]),
                            holdout=np.array(is_holdout(src)))
        k = np.array(kinds)
        summary[src["name"]] = {"venue": src["venue"], "n": len(k), "pos": int((k == 0).sum()),
                                "neg": int((k == 1).sum()), "bg": int((k == 2).sum()), "holdout": is_holdout(src),
                                "s": round(time.time() - t0)}
        print(src["name"], summary[src["name"]], flush=True)
        (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")


def load_ds(ds: Path, holdout: bool) -> dict:
    parts = []
    for f in sorted(ds.glob("*.npz")):
        z = np.load(f)
        if bool(z["holdout"]) == holdout:
            parts.append({k: z[k] for k in ("img", "prev", "target", "kind")} | {"venue": str(z["venue"])})
    if not parts:
        return {}
    cat = {k: np.concatenate([p[k] for p in parts]) for k in ("img", "prev", "target", "kind")}
    cat["venue"] = np.concatenate([[p["venue"]] * len(p["kind"]) for p in parts])
    return cat


def targets(tg: np.ndarray, hw: int) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian heatmap + size map at stride STRIDE for a batch of (cx, cy, d)."""
    n = len(tg)
    yy, xx = np.mgrid[0:hw, 0:hw]
    heat = np.zeros((n, hw, hw), np.float32)
    size = np.zeros((n, hw, hw), np.float32)
    for k, (cx, cy, d) in enumerate(tg):
        if cx < 0:
            continue
        gx, gy = cx / bd.STRIDE - 0.5, cy / bd.STRIDE - 0.5
        sig = max(0.8, d / bd.STRIDE / 3)
        heat[k] = np.exp(-((xx - gx) ** 2 + (yy - gy) ** 2) / (2 * sig ** 2))
        size[k] = d / 32.0
    return heat, size


def train(args) -> None:
    import torch
    torch.set_num_threads(args.threads)
    tr = load_ds(Path(args.ds), holdout=False)
    print("train crops", len(tr["kind"]), "pos", int((tr["kind"] == 0).sum()), flush=True)
    model = build_model_cached()
    ck = Path(args.ds).parent / "balldet.pt"
    if args.resume and ck.exists():
        model.load_state_dict(torch.load(ck))
        print("resumed from", ck, flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    hw = bd.CROP // bd.STRIDE
    n = len(tr["kind"])
    rng = np.random.default_rng(1)
    steps = args.epochs * max(1, n // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps)
    step, t0 = 0, time.time()
    last_ck = t0
    for ep in range(args.epochs):
        perm = rng.permutation(n)
        tot = 0.0
        for b0 in range(0, n - args.batch + 1, args.batch):
            sel = perm[b0:b0 + args.batch]
            img = tr["img"][sel].astype(np.float32) / 255.0
            pv = tr["prev"][sel].astype(np.float32)[..., None] / 255.0
            tg = tr["target"][sel].copy()
            if rng.random() < 0.5:  # horizontal flip
                img, pv = np.ascontiguousarray(img[:, :, ::-1]), np.ascontiguousarray(pv[:, :, ::-1])
                tg[:, 0] = np.where(tg[:, 0] >= 0, bd.CROP - tg[:, 0], tg[:, 0])
            img = img * rng.uniform(0.75, 1.25, (len(sel), 1, 1, 1)).astype(np.float32)  # brightness
            for q in range(len(sel)):  # scale: TV close-ups have balls 4-5x bigger than the gym camera's
                f = float(np.exp(rng.uniform(np.log(0.6), np.log(2.5)))) if rng.random() < 0.5 else 1.0
                if f == 1.0:
                    continue
                cs = bd.CROP
                sz = int(round(cs * f))
                a_ = cv2.resize(img[q], (sz, sz))
                b_ = cv2.resize(pv[q, :, :, 0], (sz, sz))
                cx_, cy_, d_ = tg[q]
                if f > 1:  # enlarge, crop around the ball (or centre)
                    ox = int(np.clip((cx_ * f if cx_ >= 0 else sz / 2) - cs / 2 + rng.uniform(-40, 40), 0, sz - cs))
                    oy = int(np.clip((cy_ * f if cy_ >= 0 else sz / 2) - cs / 2 + rng.uniform(-40, 40), 0, sz - cs))
                    img[q] = a_[oy:oy + cs, ox:ox + cs]
                    pv[q, :, :, 0] = b_[oy:oy + cs, ox:ox + cs]
                    tg[q] = [cx_ * f - ox, cy_ * f - oy, d_ * f] if cx_ >= 0 else [-1, -1, 0]
                else:  # shrink, pad (reflect)
                    pad = cs - sz
                    img[q] = cv2.copyMakeBorder(a_, 0, pad, 0, pad, cv2.BORDER_REFLECT)
                    pv[q, :, :, 0] = cv2.copyMakeBorder(b_, 0, pad, 0, pad, cv2.BORDER_REFLECT)
                    tg[q] = [cx_ * f, cy_ * f, d_ * f] if cx_ >= 0 else [-1, -1, 0]
                if tg[q][0] >= 0 and not (0 <= tg[q][0] < cs and 0 <= tg[q][1] < cs):
                    tg[q] = [-1, -1, 0]
            x = torch.from_numpy(np.ascontiguousarray(np.concatenate([img[..., ::-1], pv], 3).transpose(0, 3, 1, 2)))
            heat, size = targets(tg, hw)
            out = model(x)
            loss = focal(out[:, 0], torch.from_numpy(heat)) + \
                0.1 * (torch.abs(out[:, 1] - torch.from_numpy(size)) * torch.from_numpy((heat > 0.6).astype(np.float32))).sum() / \
                max(1.0, float((heat > 0.6).sum()))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss.detach())
            step += 1
            if time.time() - last_ck > 1200:
                torch.save(model.state_dict(), Path(args.ds).parent / "balldet.pt")
                last_ck = time.time()
        print(f"epoch {ep + 1}/{args.epochs} loss {tot / max(1, n // args.batch):.4f} ({time.time() - t0:.0f} s)", flush=True)
        torch.save(model.state_dict(), Path(args.ds).parent / "balldet.pt")
        torch.save(model.state_dict(), Path(args.ds).parent / f"balldet_r2_ep{ep + 1}.pt")


def focal(logit, heat, alpha: float = 2.0, beta: float = 4.0):
    import torch
    p = torch.sigmoid(logit).clamp(1e-4, 1 - 1e-4)
    pos = heat >= 0.99
    lp = -((1 - p) ** alpha) * torch.log(p) * pos
    ln = -((1 - heat) ** beta) * (p ** alpha) * torch.log(1 - p) * (~pos)
    return (lp.sum() + ln.sum()) / max(1.0, float(pos.sum()))


def build_model_cached():
    return bd.build_model()


def export(args) -> None:
    import torch
    m = bd.build_model()
    m.load_state_dict(torch.load(Path(args.ds).parent / "balldet.pt"))
    m.eval()
    x = torch.zeros(1, 4, bd.SCALE_H, 1280)
    path = Path(args.ds).parent / "balldet.onnx"
    torch.onnx.export(m, x, str(path), input_names=["x"], output_names=["y"], opset_version=17,
                      dynamic_axes={"x": {2: "h", 3: "w"}, "y": {2: "h4", 3: "w4"}}, dynamo=False)
    print("exported", path)


def evaluate(args) -> None:
    """Per held-out venue, on held-out crops: recall of tracker-confirmed balls (within max(6 px, 0.75 d)) and false
    detections per crop on negatives/background, new model vs the pass-1 detector's labels (proxy: the pass-1
    detector found all positives by construction, so its recall here is 100 % at conf >= 0.3; the comparison that
    matters is the hand-label set and make recall in the pipeline)."""
    import onnxruntime as ort
    sess = ort.InferenceSession(str(Path(args.ds).parent / "balldet.onnx"), providers=["CPUExecutionProvider"])
    ho = load_ds(Path(args.ds), holdout=True)
    res = {}
    for v in sorted(set(ho["venue"])):
        m = ho["venue"] == v
        tp = fn = fp = 0
        for k in np.flatnonzero(m):
            x = np.concatenate([ho["img"][k][:, :, ::-1], ho["prev"][k][..., None]], 2).transpose(2, 0, 1)[None]
            out = sess.run(None, {"x": np.ascontiguousarray(x, np.float32) / 255.0})[0]
            boxes, _ = bd.decode(out, 1.0, thr=args.thr)
            cx, cy, d = ho["target"][k]
            hit = False
            for b in boxes:
                bx, by = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                if cx >= 0 and np.hypot(bx - cx, by - cy) <= max(6.0, 0.75 * d):
                    hit = True
                else:
                    fp += 1
            if cx >= 0:
                tp += hit
                fn += not hit
        res[v] = {"recall": round(tp / max(1, tp + fn), 3), "fp_per_crop": round(fp / max(1, int(m.sum())), 3),
                  "crops": int(m.sum())}
        print(v, res[v], flush=True)
    (Path(args.ds).parent / "eval.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


def build_mined(args) -> None:
    """data/balldet/mined/<venue>/labels.json (redetect agent's mining: rim / hand / flight positives, head / rim /
    static / rejected negative regions, frame + previous frame jpgs) -> ds/mined__<venue>.npz crops (same format)."""
    rng = np.random.default_rng(2)
    root = ROOT / "data/balldet/mined"
    hold = json.loads((root / "splits.json").read_text(encoding="utf-8")) if (root / "splits.json").exists() else {}
    for lab in sorted(root.glob("*/labels.json")):
        venue = lab.parent.name
        d = json.loads(lab.read_text(encoding="utf-8"))
        ann: dict[int, list] = {}
        for a in d["annotations"]:
            ann.setdefault(a["image_id"], []).append(a)
        neg: dict[int, list] = {}
        for r in d.get("neg_regions") or []:
            neg.setdefault(r["image_id"], []).append(r)
        imgs, prevs, tg, kinds, confs, ho = [], [], [], [], [], []
        for im in d["images"]:
            f, fp = lab.parent / im["file_name"], lab.parent / im["prev_file_name"]
            img, pv = cv2.imread(str(f)), cv2.imread(str(fp))
            if img is None or pv is None:
                continue
            s = bd.SCALE_H / img.shape[0]
            if s != 1:
                img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
                pv = cv2.resize(pv, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            pg = cv2.cvtColor(pv, cv2.COLOR_BGR2GRAY)
            H, W = img.shape[:2]
            ign = [(r["bbox"][0] * s, r["bbox"][1] * s, (r["bbox"][0] + r["bbox"][2]) * s, (r["bbox"][1] + r["bbox"][3]) * s)
                   for r in d.get("ignore_regions") or [] if r["image_id"] == im["id"]]
            balls = [((a["bbox"][0] + a["bbox"][2] / 2) * s, (a["bbox"][1] + a["bbox"][3] / 2) * s,
                      0.5 * (a["bbox"][2] + a["bbox"][3]) * s, a) for a in ann.get(im["id"], [])]
            centres = [(bx, by, 0) for bx, by, _, _ in balls]
            nr = neg.get(im["id"], [])
            heads = [r for r in nr if r["kind"] == "head"]
            other = [r for r in nr if r["kind"] != "head"]
            for r in other + [heads[k] for k in rng.choice(len(heads), min(2, len(heads)), replace=False)] if heads                     else other:
                x, y, w, h = r["bbox"]
                centres.append(((x + w / 2) * s, (y + h / 2) * s, 1))
            for cx, cy, kind in centres:
                ox, oy = rng.uniform(-60, 60, 2)
                x0 = int(np.clip(cx + ox - bd.CROP / 2, 0, max(0, W - bd.CROP)))
                y0 = int(np.clip(cy + oy - bd.CROP / 2, 0, max(0, H - bd.CROP)))
                c, p_ = img[y0:y0 + bd.CROP, x0:x0 + bd.CROP], pg[y0:y0 + bd.CROP, x0:x0 + bd.CROP]
                if c.shape[:2] != (bd.CROP, bd.CROP):
                    continue
                if any(ix1 > x0 and ix0 < x0 + bd.CROP and iy1 > y0 and iy0 < y0 + bd.CROP for ix0, iy0, ix1, iy1 in ign):
                    continue  # an ignore region (static ball-like object / unsure) inside: neither ball nor negative
                t_ = [-1, -1, 0]
                for bx, by, bdia, _ in balls:
                    if -8 <= bx - x0 <= bd.CROP + 8 and -8 <= by - y0 <= bd.CROP + 8:
                        t_ = [bx - x0, by - y0, bdia]
                imgs.append(c), prevs.append(p_), tg.append(t_), kinds.append(0 if t_[0] >= 0 else kind)
                confs.append(1.0)
                ho.append(bool(im.get("holdout")))
        k = np.array(kinds)
        hset = np.array(ho, bool)
        for flag, tag in ((False, "train"), (True, "holdout")):
            sel = hset == flag
            if not sel.any():
                continue
            np.savez_compressed(Path(args.ds) / f"mined__{venue}__{tag}.npz",
                                img=np.array(imgs, np.uint8)[sel], prev=np.array(prevs, np.uint8)[sel],
                                target=np.array(tg, np.float32)[sel], kind=np.array(kinds, np.int8)[sel],
                                conf=np.array(confs, np.float32)[sel], venue=np.array(venue), holdout=np.array(flag))
        print(venue, "crops", len(k), "pos", int((k == 0).sum()), "holdout", int(hset.sum()), flush=True)


def eval_check(args) -> None:
    """Hand-verified ball boxes of the held-out venues (data/balldet/mined/handcheck.json, redetect agent): recall of
    the pass-1 detector (conf >= 0.3, its pass-1 rows) vs the new model (thr), a hit = a detection centre within
    max(12 px, half the box diagonal) of the verified box centre; plus false detections per image far from every
    annotated ball of that image (labels.json)."""
    from replay.detection import BALL
    from replay.pipeline import load_pass1
    root = ROOT / "data/balldet/mined"
    boxes = [b for b in json.loads((root / "handcheck.json").read_text(encoding="utf-8"))["boxes"] if b.get("verified")]
    labs = {}
    for f in root.glob("*/labels.json"):
        d = json.loads(f.read_text(encoding="utf-8"))
        ids = {im["file_name"]: im["id"] for im in d["images"]}
        by = {}
        for a in d["annotations"]:
            by.setdefault(a["image_id"], []).append(a["bbox"])
        labs[f.parent.name] = (ids, by)
    srcs = {s_["name"]: s_ for s_ in sources()}
    model = bd.BallDet(Path(args.ds).parent / "balldet.onnx", ["CPUExecutionProvider"])
    p1cache = {}
    res: dict[str, dict] = {}
    for b in boxes:
        v = b["venue"]
        r = res.setdefault(v, {"n": 0, "p1": 0, "new": 0, "p1_fp": 0, "new_fp": 0})
        x, y, w, h = b["bbox"]
        cx, cy, tol = x + w / 2, y + h / 2, max(12.0, 0.5 * float(np.hypot(w, h)))
        img = cv2.imread(str(root / b["file_name"]))
        pv = cv2.imread(str(root / b["prev_file_name"]))
        others = []
        ids, by = labs.get(v, ({}, {}))
        fn = b["file_name"].split("/", 1)[1]
        for ob in by.get(ids.get(fn), []):
            others.append((ob[0] + ob[2] / 2, ob[1] + ob[3] / 2, max(12.0, 0.5 * float(np.hypot(ob[2], ob[3])))))
        others.append((cx, cy, tol))
        bx, sc = model.detect(img, pv, thr=args.thr)
        cen = np.c_[(bx[:, 0] + bx[:, 2]) / 2, (bx[:, 1] + bx[:, 3]) / 2] if len(bx) else np.zeros((0, 2))
        r["n"] += 1
        r["new"] += int(any(np.hypot(c[0] - cx, c[1] - cy) <= tol for c in cen))
        r["new_fp"] += sum(1 for c in cen if all(np.hypot(c[0] - ox, c[1] - oy) > ot for ox, oy, ot in others))
        if b["clip"] in srcs:
            if b["clip"] not in p1cache:
                dd = load_pass1(Path(srcs[b["clip"]]["work"]))
                p1cache[b["clip"]] = (dd, dd.rows[(dd.rows[:, 6] == BALL) & (dd.rows[:, 5] >= 0.3)])
            dd, rows = p1cache[b["clip"]]
            ii = np.flatnonzero(dd.frame == b["frame"])
            if len(ii):
                rr = rows[rows[:, 0] == dd.k[ii[0]]]
                c1 = np.c_[(rr[:, 1] + rr[:, 3]) / 2, (rr[:, 2] + rr[:, 4]) / 2]
                r["p1"] += int(any(np.hypot(c[0] - cx, c[1] - cy) <= tol for c in c1))
                r["p1_fp"] += sum(1 for c in c1 if all(np.hypot(c[0] - ox, c[1] - oy) > ot for ox, oy, ot in others))
    for v, r in sorted(res.items()):
        n = max(1, r["n"])
        print(f"{v}: n={r['n']}  pass-1 recall {r['p1'] / n:.0%} fp/img {r['p1_fp'] / n:.2f}  |  new recall "
              f"{r['new'] / n:.0%} fp/img {r['new_fp'] / n:.2f}")
    (Path(args.ds).parent / "eval_check.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


HAND = {"p4400": "koristv/p4400", "b3min": "koristv/b3min", "c300": "koristv/c300"}


def eval_hand(args) -> None:
    """The hand-labelled koristv frames (data/balldet/hand_labels/labels_<clip>.json: analysed-frame index i + ball
    image point; frames where the ball is visible): recall within 12 px of the labelled point and false detections
    (> 12 px from it) per frame, pass-1 detector (conf >= 0.3) vs the new model (thr)."""
    from replay.detection import BALL
    from replay.pipeline import load_pass1
    from replay.redetect import iter_frames
    model = bd.BallDet(Path(args.ds).parent / "balldet.onnx", ["CPUExecutionProvider"]) if args.model != "none" else None
    srcs = {s["name"]: s for s in sources()}
    tot = {"p1": [0, 0, 0], "new": [0, 0, 0]}  # tp, n, fp
    for clip, name in HAND.items():
        f = ROOT / "data/balldet/hand_labels" / f"labels_{clip}.json"
        if not f.exists() or name not in srcs:
            continue
        labs = {l["i"]: l for l in json.loads(f.read_text(encoding="utf-8")) if l.get("ball") and not l.get("hidden")}
        data = load_pass1(Path(srcs[name]["work"]))
        rows = data.rows[data.rows[:, 6] == BALL]
        want = sorted(set(labs) | {i - 1 for i in labs})
        prev = {}
        for i, img in iter_frames(Path(srcs[name]["video"]), data, want):
            prev[i] = img
            if i not in labs:
                continue
            bx, by = labs[i]["ball"]
            r = rows[(rows[:, 0] == data.k[i]) & (rows[:, 5] >= 0.3)]
            d1 = np.hypot((r[:, 1] + r[:, 3]) / 2 - bx, (r[:, 2] + r[:, 4]) / 2 - by)
            tot["p1"][0] += int((d1 <= 12).any())
            tot["p1"][1] += 1
            tot["p1"][2] += int((d1 > 12).sum())
            if model is not None:
                boxes, _ = model.detect(img, prev.get(i - 1), thr=args.thr)
                d2 = np.hypot((boxes[:, 0] + boxes[:, 2]) / 2 - bx, (boxes[:, 1] + boxes[:, 3]) / 2 - by)
                tot["new"][0] += int((d2 <= 12).any())
                tot["new"][1] += 1
                tot["new"][2] += int((d2 > 12).sum())
    for k, (tp, n, fp) in tot.items():
        if n:
            print(f"{k}: recall {tp}/{n} = {tp / n:.0%}, false detections per frame {fp / n:.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "buildmined", "train", "export", "eval", "evalhand", "evalcheck", "sources"])
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--model", default="")
    ap.add_argument("--out", default=str(DS))
    ap.add_argument("--ds", default=str(DS))
    ap.add_argument("--max-pos", type=int, default=400)
    ap.add_argument("--max-neg", type=int, default=250)
    ap.add_argument("--max-bg", type=int, default=100)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--thr", type=float, default=0.3)
    a = ap.parse_args()
    if a.cmd == "sources":
        for s in sources():
            print(s["venue"], s["name"], "HOLDOUT" if is_holdout(s) else "", s["video"])
        return
    {"build": build, "train": train, "export": export, "eval": evaluate, "evalhand": eval_hand, "buildmined": build_mined, "evalcheck": eval_check}[a.cmd](a)


if __name__ == "__main__":
    main()
