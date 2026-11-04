"""Per-venue evaluation of a court-detector checkpoint (replay/courtnet.py) - the harness of court_design.md section 6.

usage: python tools/court_eval.py --model data/courtnet/courtnet.onnx --out DIR [--venues a,b,...] [--every 1.0]
                                  [--threads 2] [--tag name] [--no-solver]
Per venue (data/external/* segments, home-gym clips, trackid3x3 GT):
- perception on reference frames (venues with a trusted H: classic good anchors or the trackid3x3 GT keyframe):
  line_recall = share of projected court-line samples (inside the frame) with mask prob > 0.5 within 1 px at the
  net's 160x96 output; line_prec = share of mask pixels (> 0.5) within 2 px of the true lines; kp_hit = visible
  keypoints whose channel's peak (> 0.3) lies within 4 px (160x96) of the truth; kp_false = channels of keypoints
  NOT in view that still peak > 0.3 (should be 0).
- solver: autocourt.calibrate_net (keypoint RANSAC + painted-line check) -> court, frames located / net samples,
  samples covered / all samples (resolve), seconds.
- sheets: <out>/<tag>_<venue>_raw.jpg (8 evenly spaced samples: line mask red, peaks > 0.3 green, truth yellow when
  known) and <out>/<tag>_<venue>_located.jpg (8 located frames with the court drawn).
Appends one JSON line per venue to <out>/results.jsonl and prints it. Frames from data/external are never written
anywhere but DIR (licences: data/external/*/ATTRIBUTION.md).
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

from replay import autocourt, court, courtnet  # noqa: E402
from replay.court_synth import IN_H, IN_W, OUT_H, OUT_W, _project_lines, targets  # noqa: E402
from replay.pipeline import load_pass1  # noqa: E402

EXT = ROOT / "data" / "external"
CLIPS = ROOT / "data" / "clips"


def _seg(venue: str, seg: str, ref: bool, **kw) -> dict:
    d = EXT / venue / seg
    vid = next((v for v in (d.with_suffix(".mp4"), d.with_suffix(".webm")) if v.exists()), None)
    return dict(video=vid, work=d / "out" / "work", cal=d / "calibration.json", ref=ref, **kw)


VENUES = {
    # held out of the real training set
    "ncaa_2965": _seg("ncaa", "seg_2965", False),
    "ncaa_3635": _seg("ncaa", "seg_3635", False),
    "ncaa_4465": _seg("ncaa", "seg_4465", False),
    "westwood_1500": _seg("westwood_girls", "seg_1500", False),
    # calibration_gt.json (imported TrackID3x3 homography) does NOT lie on the floor lines by eye (04.11 raw sheet):
    # solver / coverage only, no perception reference (gt= would enable it)
    "trackid_S1T1": _seg("trackid3x3", "seg_basket_S1T1_pre", False),
    # training venues (sanity: the net must at least fit what it was trained on)
    "tbf_900": _seg("tbf_turkey", "seg_900", True),
    "ven_1500": _seg("ven_superliga", "seg_1500", True),
    "hazen_600": _seg("hs_hazen", "seg_600", True),
    "c300": dict(video=CLIPS / "c300" / "clip.mp4", work=CLIPS / "c300" / "out" / "work", cal=CLIPS / "c300" / "clip.calibration.json", ref=True),
}
HELD_OUT = ("ncaa_2965", "ncaa_3635", "ncaa_4465", "westwood_1500", "trackid_S1T1")


def reference_H(v: dict, cal: dict | None, data, size) -> dict[int, np.ndarray]:
    """sample index -> trusted court->image H (court configured by the caller)."""
    if v.get("gt"):
        g = json.loads(Path(v["gt"]).read_text(encoding="utf-8"))
        kf = g["keyframes"][0]
        Hi2c = np.asarray(kf["H_image_to_court"], float).reshape(3, 3)
        H = np.linalg.inv(Hi2c)
        # sign so that the court centre is in front of the camera (depth row)
        q = H @ np.array([court.LENGTH / 2, court.WIDTH / 2, 1.0])
        if q[2] < 0:
            H = -H
        return {i: H for i in range(len(data.t))}  # fixed camera
    if not cal or not v.get("ref"):
        return {}
    Hs, dist = autocourt.resolve_auto(np.asarray(data.t, float), data.motion, cal["auto"], size)
    return {i: Hs[i] for i in range(len(data.t)) if Hs[i] is not None and dist[i] <= 0.05}


def perception(net, img, H_ref, size, names):
    """Metrics of one frame against its reference H (full-frame pixels)."""
    S = np.diag([IN_W / size[0], IN_H / size[1], 1.0])
    line_t, heat_t = targets(S @ H_ref, None, None, names)
    lp, hm = net.predict(img)
    m = (lp > 0.5).astype(np.uint8)
    lt = (line_t > 0.5).astype(np.uint8)
    out = {}
    if lt.sum() >= 20:
        near = cv2.dilate(m, np.ones((3, 3), np.uint8))
        out["line_recall"] = float(near[lt > 0].mean())
        near_t = cv2.dilate(lt, np.ones((5, 5), np.uint8))
        out["line_prec"] = float(near_t[m > 0].mean()) if m.sum() else 0.0
        out["mask_frac"] = float(m.mean())
    vis = heat_t.max(axis=(1, 2)) > 0.5
    hits, false = 0, 0
    for k in range(len(names)):
        j = int(np.argmax(hm[k]))
        v, u = divmod(j, hm.shape[2])
        pk = hm[k][v, u]
        if vis[k]:
            jt = int(np.argmax(heat_t[k]))
            vt, ut = divmod(jt, OUT_W)
            if pk > 0.3 and np.hypot(u - ut, v - vt) <= 4:
                hits += 1
        elif pk > 0.3:
            false += 1
    out["kp_vis"] = int(vis.sum())
    out["kp_hit"] = hits
    out["kp_false"] = false
    out["kp_notvis"] = int((~vis).sum())
    return out, lp, hm


def draw_lines(img, H, color, scale=1.0):
    h, w = img.shape[:2]
    for run in _project_lines(np.diag([scale, scale, 1.0]) @ H, w, h, None, None):
        cv2.polylines(img, [np.int32(np.round(run)).reshape(-1, 1, 2)], False, color, 1, cv2.LINE_AA)


def tile(imgs, cols=4):
    while len(imgs) % cols:
        imgs.append(np.zeros_like(imgs[0]))
    return np.vstack([np.hstack(imgs[r:r + cols]) for r in range(0, len(imgs), cols)])


def raw_tile(img, lp, hm, size, H_ref, label):
    t = cv2.resize(img, (IN_W * 2, IN_H * 2), interpolation=cv2.INTER_AREA)
    m = cv2.resize((lp > 0.5).astype(np.uint8), (IN_W * 2, IN_H * 2), interpolation=cv2.INTER_NEAREST) > 0
    t[m] = (0.4 * t[m] + 0.6 * np.array([0, 0, 255])).astype(np.uint8)
    if H_ref is not None:
        draw_lines(t, np.diag([IN_W * 2 / size[0], IN_H * 2 / size[1], 1.0]) @ H_ref, (0, 255, 255))
    for k, (x, y, s) in courtnet.peaks(hm, (IN_W * 2, IN_H * 2), 0.3).items():
        cv2.circle(t, (int(x), int(y)), 4, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(t, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(t, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return t


def evaluate(name: str, v: dict, model: Path, out: Path, every: float, threads: int, tag: str, solver: bool) -> dict:
    t0 = time.time()
    data = load_pass1(v["work"])
    video = Path(data.signature.get("video", "")) if data.signature.get("video") else None
    if video is None or not video.exists():
        video = v["video"]
    cal = json.loads(Path(v["cal"]).read_text(encoding="utf-8")) if v["cal"] and Path(v["cal"]).exists() else None
    if cal:
        size = tuple(cal["frame_size"])
    else:
        cap = cv2.VideoCapture(str(video))
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        cap.release()
    ref_court = (json.loads(Path(v["gt"]).read_text(encoding="utf-8"))["court"] if v.get("gt") else
                 (cal or {}).get("court"))
    res = {"tag": tag, "venue": name, "model": str(model), "n": int(len(data.t)), "size": list(size)}
    net = courtnet.CourtNet(model, threads=threads)
    names = list(court.KEYPOINTS)
    # --- perception against the reference H (when trusted)
    refs = {}
    if ref_court:
        court.configure(float(ref_court["length"]), float(ref_court["width"]), ref_court.get("spec", "FIBA"))
        refs = reference_H(v, cal, data, size)
    raw_idx = list(np.linspace(0, len(data.t) - 1, 8).astype(int))
    ref_idx = sorted(refs)
    ref_idx = ref_idx[:: max(1, len(ref_idx) // 40)][:40] if ref_idx else []
    want = sorted(set(raw_idx) | set(ref_idx))
    f2i = {int(data.frame[i]): i for i in want}
    per, raw_tiles = [], []
    for fno, img in autocourt.read_frames(video, list(f2i)):
        i = f2i[fno]
        if i in refs:
            court.configure(float(ref_court["length"]), float(ref_court["width"]), ref_court.get("spec", "FIBA"))
            p, lp, hm = perception(net, img, refs[i], size, names)
            per.append(p)
        else:
            lp, hm = net.predict(img)
        if i in raw_idx:
            raw_tiles.append(raw_tile(img, lp, hm, size, refs.get(i), f"{name} t={data.t[i]:.1f}"))
    if per:
        res["ref_frames"] = len(per)
        for k in ("line_recall", "line_prec", "mask_frac"):
            vals = [p[k] for p in per if k in p]
            if vals:
                res[k] = round(float(np.mean(vals)), 3)
        vis = sum(p["kp_vis"] for p in per)
        nv = sum(p["kp_notvis"] for p in per)
        res["kp_hit"] = round(sum(p["kp_hit"] for p in per) / max(1, vis), 3)
        res["kp_false"] = round(sum(p["kp_false"] for p in per) / max(1, nv), 3)
    if raw_tiles:
        cv2.imwrite(str(out / f"{tag}_{name}_raw.jpg"), tile(raw_tiles), [cv2.IMWRITE_JPEG_QUALITY, 80])
    # --- solver (the current calibrate_net path)
    if solver:
        t1 = time.time()
        r = autocourt.calibrate_net(video, data, size, model=model, every_s=every)
        res["solver_s"] = round(time.time() - t1, 1)
        if r is None:
            res.update(court=None, located=0, covered=0)
        else:
            court.configure(r.length, r.width, r.spec)
            Hs, _d = autocourt.resolve_auto(np.asarray(data.t, float), data.motion, r.to_json(), size)
            res.update(court=f"{r.spec} {r.length:.2f}x{r.width:.2f}", located=len(r.frames),
                       net_samples=int(round(r.good_frac and len(r.frames) / r.good_frac or 0)),
                       covered=int(sum(h is not None for h in Hs)))
            if r.frames:
                sel = [r.frames[j] for j in np.linspace(0, len(r.frames) - 1, min(8, len(r.frames))).astype(int)]
                f2 = {int(data.frame[fr["i"]]): fr for fr in sel}
                tiles = []
                for fno, img in autocourt.read_frames(video, list(f2)):
                    fr = f2[fno]
                    t = cv2.resize(img, (IN_W * 2, IN_H * 2), interpolation=cv2.INTER_AREA)
                    H = np.asarray(fr["H"], float).reshape(3, 3)
                    q = H @ np.array([court.LENGTH / 2, court.WIDTH / 2, 1.0])
                    if q[2] < 0:
                        H = -H
                    draw_lines(t, np.diag([IN_W * 2 / size[0], IN_H * 2 / size[1], 1.0]) @ H, (0, 255, 255))
                    cv2.putText(t, f"{name} t={fr['t']:.1f}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(t, f"{name} t={fr['t']:.1f}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                    tiles.append(t)
                if tiles:
                    cv2.imwrite(str(out / f"{tag}_{name}_located.jpg"), tile(tiles), [cv2.IMWRITE_JPEG_QUALITY, 80])
    res["s"] = round(time.time() - t0, 1)
    court.configure(28.0, 15.0, "FIBA")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(courtnet.MODEL_DIR / "courtnet.onnx"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--venues", default=",".join(HELD_OUT))
    ap.add_argument("--every", type=float, default=1.0, help="net sample spacing for the solver (s)")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--no-solver", action="store_true", help="perception metrics + raw sheets only")
    a = ap.parse_args()
    cv2.setNumThreads(a.threads)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = a.tag or Path(a.model).stem
    for name in a.venues.split(","):
        v = VENUES[name]
        try:
            res = evaluate(name, v, Path(a.model), out, a.every, a.threads, tag, not a.no_solver)
        except Exception as e:  # noqa: BLE001 - one venue must not stop the table
            res = {"tag": tag, "venue": name, "error": repr(e)}
        line = json.dumps(res)
        print(line, flush=True)
        with open(out / "results.jsonl", "a", encoding="utf-8") as f:
            f.write(line + "\n")


if __name__ == "__main__":
    main()
