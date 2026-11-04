"""Real training frames for the court detector from venues whose classic calibration is verified good by overlay.

v2 (04.11): several venues, labels from the per-sample homography of the classic calibration (autocourt.resolve_auto
on the verified good anchors, samples within --max-anchor-s of an anchor, spaced --every s apart). Each item: image
320x192 (BGR), H court -> 320x192 pixels (scale kept: third row = camera depth, so court_synth.targets(H, None, None,
names) knows what is in front of the camera), venue / source tags (leave-one-venue-out), t, frame, court (L, W, spec).
Held out on purpose (never listed in SOURCES): NCAA, westwood_girls, trackid3x3.

usage: python tools/make_courtnet_real.py [--out data/courtnet/real_v2.npz] [--sheets DIR] [--drop drop.json]
                                          [--venues koristv,tbf_turkey,ven_superliga,hs_hazen] [--every 0.5]
                                          [--max-anchor-s 1.0]
--sheets DIR writes one contact sheet per source (court drawn from the label) for the by-eye check; --drop is a JSON
{"<venue>/<name>": [[t0, t1], ...]} of time ranges (s) to leave out after that check ([0, 1e9] drops the source).
The old format (C, R, f per frame; home gym only) is still read by tools/train_courtnet.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from replay import autocourt, court  # noqa: E402
from replay.court_synth import IN_H, IN_W, targets  # noqa: E402
from replay.pipeline import load_pass1  # noqa: E402

EXT = ROOT / "data" / "external"
# the benchmark's p4400 copy has no video; the user's own copy of the clip (read only) when present
P4400_VIDEO = ROOT.parent / "AdvancedAnalytist" / "data" / "clip_4400.mp4"
MIN_LINE_PX = 60  # label line pixels at 160x96: fewer = the court is (almost) out of view


def sources(venues: set[str]) -> list[tuple[str, str, Path, Path, Path]]:
    """(venue, name, video, pass-1 work, calibration.json) for the training venues."""
    out = []
    if "koristv" in venues:
        for d in sorted((ROOT / "data" / "clips").glob("c*")):
            if (d / "clip.calibration.json").exists() and (d / "out" / "work" / "DONE").exists():
                out.append(("koristv", d.name, d / "clip.mp4", d / "out" / "work", d / "clip.calibration.json"))
        b = ROOT / "data" / "bench3min"
        if (b / "out" / "work" / "DONE").exists():
            out.append(("koristv", "bench3min", b / "clip.mp4", b / "out" / "work", b / "clip.calibration.json"))
        p = ROOT / "data" / "benchmark" / "sources" / "p4400"
        if P4400_VIDEO.exists() and (p / "results" / "work" / "DONE").exists():
            out.append(("koristv", "p4400", P4400_VIDEO, p / "results" / "work", p / "calibration.json"))
    for venue in ("tbf_turkey", "ven_superliga", "hs_hazen"):
        if venue not in venues:
            continue
        for d in sorted((EXT / venue).glob("seg_*")):
            if not d.is_dir() or not (d / "calibration.json").exists() or not (d / "out" / "work" / "DONE").exists():
                continue
            vid = next((v for v in (d.with_suffix(".webm"), d.with_suffix(".mp4")) if v.exists()), None)
            if vid is not None:
                out.append((venue, d.name, vid, d / "out" / "work", d / "calibration.json"))
    return out


def select(cal: dict, data, every: float, max_anchor_s: float, drops: list) -> list[tuple[int, np.ndarray]]:
    """(sample index, H court->full-frame pixels) of the labelled samples."""
    size = tuple(cal["frame_size"])
    auto = cal.get("auto") or {}
    if not any(a.get("good") for a in auto.get("anchors", [])):
        return []
    Hs, dist = autocourt.resolve_auto(np.asarray(data.t, float), data.motion, auto, size)
    out, last = [], -1e9
    for i in range(len(data.t)):
        t = float(data.t[i])
        if Hs[i] is None or dist[i] > max_anchor_s or t - last < every - 1e-6:
            continue
        if any(t0 <= t <= t1 for t0, t1 in drops):
            continue
        out.append((i, Hs[i]))
        last = t
    return out


def draw_court(img: np.ndarray, H: np.ndarray, color=(0, 255, 255)) -> None:
    """Court lines of the current court drawn with H (court -> this image's pixels)."""
    h, w = img.shape[:2]
    from replay.court_synth import _project_lines

    for run in _project_lines(H, w, h, None, None):
        cv2.polylines(img, [np.int32(np.round(run)).reshape(-1, 1, 2)], False, color, 1, cv2.LINE_AA)


def sheet(imgs: list[np.ndarray], Hs: list[np.ndarray], labels: list[str], cols: int = 4, scale: int = 2) -> np.ndarray:
    tiles = []
    for img, H, lab in zip(imgs, Hs, labels):
        t = cv2.resize(img, (IN_W * scale, IN_H * scale), interpolation=cv2.INTER_CUBIC)
        draw_court(t, np.diag([scale, scale, 1.0]) @ H)
        cv2.putText(t, lab, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(t, lab, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(t)
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[r:r + cols]) for r in range(0, len(tiles), cols)]
    return np.vstack(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data" / "courtnet" / "real_v2.npz"))
    ap.add_argument("--sheets", default=None, help="folder for the per-source contact sheets (by-eye check)")
    ap.add_argument("--drop", default=None, help="JSON {source: [[t0, t1], ...]} of time ranges to leave out")
    ap.add_argument("--venues", default="koristv,tbf_turkey,ven_superliga,hs_hazen")
    ap.add_argument("--every", type=float, default=0.5, help="minimum spacing of labelled frames (s)")
    ap.add_argument("--max-anchor-s", type=float, default=1.0, help="frames further from a good anchor are skipped")
    ap.add_argument("--sheet-n", type=int, default=12)
    ap.add_argument("--filter-only", action="store_true",
                    help="apply --drop to the existing --out file (no video decoding) and rewrite it")
    a = ap.parse_args()
    drops = json.loads(Path(a.drop).read_text(encoding="utf-8")) if a.drop else {}
    if a.filter_only:
        z = np.load(a.out, allow_pickle=True)
        R_ = {k: z[k] for k in z.files}
        keep = np.array([not any(t0 <= t <= t1 for t0, t1 in drops.get(str(s), []))
                         for s, t in zip(R_["src"], R_["t"])])
        np.savez_compressed(a.out, **{k: v[keep] for k, v in R_.items()})
        ven, cnt = np.unique(R_["venue"][keep], return_counts=True)
        src, scnt = np.unique(R_["src"][keep], return_counts=True)
        print(f"filtered {a.out}: {int(keep.sum())} / {len(keep)} kept; per venue {dict(zip(ven, cnt.tolist()))}; "
              f"per source {dict(zip(src, scnt.tolist()))}")
        return
    imgs, Hs, venues, names, ts, frames, courts, sizes = [], [], [], [], [], [], [], []
    per_source = {}
    for venue, name, video, work, calp in sources(set(a.venues.split(","))):
        key = f"{venue}/{name}"
        cal = json.loads(calp.read_text(encoding="utf-8"))
        c = cal["court"]
        court.configure(float(c["length"]), float(c["width"]), c.get("spec", "FIBA"))
        kp = list(court.KEYPOINTS)
        data = load_pass1(work)
        sel = select(cal, data, a.every, a.max_anchor_s, drops.get(key, []))
        if not sel:
            print(f"{key}: no usable frames")
            continue
        w, h = cal["frame_size"]
        S = np.diag([IN_W / w, IN_H / h, 1.0])
        by_frame = {int(data.frame[i]): (i, H) for i, H in sel}
        n0 = len(imgs)
        for fno, img in autocourt.read_frames(video, list(by_frame)):
            i, H = by_frame[fno]
            if img.shape[1] != w or img.shape[0] != h:
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            Hin = S @ H
            line, _heat = targets(Hin, None, None, kp)
            if float(line.sum()) < MIN_LINE_PX:  # court (almost) out of view: no label worth learning
                continue
            imgs.append(cv2.resize(img, (IN_W, IN_H), interpolation=cv2.INTER_AREA))
            Hs.append(Hin)
            venues.append(venue)
            names.append(key)
            ts.append(float(data.t[i]))
            frames.append(int(fno))
            courts.append((float(c["length"]), float(c["width"]), c.get("spec", "FIBA")))
            sizes.append((w, h))
        per_source[key] = len(imgs) - n0
        print(f"{key}: {len(sel)} selected, {per_source[key]} kept ({c.get('spec')} {c['length']} x {c['width']})")
        if a.sheets and per_source[key]:
            Path(a.sheets).mkdir(parents=True, exist_ok=True)
            ids = np.linspace(n0, len(imgs) - 1, min(a.sheet_n, per_source[key])).astype(int)
            sh = sheet([imgs[j] for j in ids], [Hs[j] for j in ids], [f"{name} t={ts[j]:.1f}" for j in ids])
            cv2.imwrite(str(Path(a.sheets) / f"{venue}_{name}.jpg"), sh, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not imgs:
        raise SystemExit("no frames")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, img=np.stack(imgs), H=np.stack(Hs), venue=np.array(venues), src=np.array(names),
                        t=np.array(ts), frame=np.array(frames), court=np.array(courts, dtype=object),
                        size=np.array(sizes))
    by_venue = {}
    for v in venues:
        by_venue[v] = by_venue.get(v, 0) + 1
    print(f"wrote {a.out}: {len(imgs)} frames; per venue {by_venue}; per source {per_source}")


if __name__ == "__main__":
    main()
