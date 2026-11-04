"""Second-look detector (pass-2 step 1b): re-run the person detector on upscaled crops where a player is missing.

Pass 1 runs RF-DETR once on the full 720p frame. Players who are small (far court), partly hidden behind a
team-mate (the paint) or blurred are sometimes missed for a few samples; ByteTrack then shows a gap or starts a
new ID. Here we look again, but only where something is expected:

  * gap probes: a ByteTrack track disappears for < ``max_gap_s`` and reappears (same ID), or ends and a
    compatible track starts nearby soon after (ID switch). The box at the missing samples is predicted by
    interpolating the two anchor boxes in camera-stabilised pixels (pass-1 camera motion).
  * cluster probes: two or more overlapping people with a foot in a key area (paint + 1 m) - the place where a
    hidden third player is most often lost.

Each probe crops a square around the expected location (the detector resizes its input, so a small crop is an
upscale) and runs the same detector (replay/detection, ONNX/DirectML, same weights) with a lower threshold. A new
box is kept only when it is consistent with what was expected:
  * gap: position/height close to the prediction, jersey colour histogram (team_appearance descriptors, Hellinger
    distance) close to the track's own anchor boxes, not a duplicate of an existing pass-1 box;
  * cluster: plausible person shape and height for the cluster, foot on court, colour like some on-court person in
    the same frame, not a duplicate, AND found again at the next sample (temporal confirmation).
Accepted boxes become ordinary pass-1 rows (jersey features from the frame; ByteTrack ID of the gap's track for
same-ID gaps, -1 otherwise), so every later step (projection, tracklets, roster, teams) treats them normally.

Budgeted (``crops_per_s`` crops per second of video, gap probes first) and cached in ``work/redetect.npz`` keyed by
the pass-1 rows, video and settings. Frames are decoded sequentially (grab without conversion is cheap).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np

os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")
import cv2  # noqa: E402

VERSION = 5
CACHE = "redetect.npz"


from replay.config import RedetectConfig  # noqa: E402,F401 - moved to config.py (Config.redetect); re-exported



def config_from(cfg) -> RedetectConfig:
    """RedetectConfig from Config.redetect (if the shared config gets one) + env REPLAY_REDETECT=0/1."""
    rc = getattr(cfg, "redetect", None)
    rc = replace(rc) if isinstance(rc, RedetectConfig) else RedetectConfig()
    env = os.environ.get("REPLAY_REDETECT")
    if env is not None:
        rc.enabled = env.strip() not in ("", "0", "false", "no")
    for f in ("crops_per_s", "threshold", "max_gap_s", "clusters"):  # experiments: REPLAY_REDETECT_CROPS_PER_S=0.5
        v = os.environ.get("REPLAY_REDETECT_" + f.upper())
        if v is not None:
            setattr(rc, f, v.strip() not in ("0", "false", "no") if f == "clusters" else float(v))
    return rc


@dataclass
class Probe:
    i: int  # sample index
    region: tuple[float, float, float, float]  # crop in frame pixels
    kind: str  # "gap" | "cluster"
    priority: float  # lower = earlier
    pred: np.ndarray | None = None  # expected box (gap)
    bt: int = -1  # ByteTrack ID given to an accepted box
    anchors: list[int] = field(default_factory=list)  # pass-1 row indices whose colour the new box must match
    group: int = -1  # cluster: probes of one cluster at consecutive samples
    ref_h: float = 0.0  # cluster: median box height


# ----------------------------------------------------------------------------- geometry helpers
def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    ix = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = ix * iy
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + ab[None, :] - inter, 1e-9)


def _apply(G: np.ndarray, p) -> np.ndarray:
    v = G @ np.array([p[0], p[1], 1.0])
    return v[:2] / v[2]


def _segments(motion: list, n: int) -> tuple[list[np.ndarray], np.ndarray]:
    from replay.calibration import _segment_transforms

    return _segment_transforms(motion, n)


def square_region(cx: float, cy: float, side: float, size: tuple[int, int]) -> tuple[float, float, float, float]:
    W, H = size
    side = float(min(side, W, H))
    x0 = min(max(0.0, cx - side / 2), W - side)
    y0 = min(max(0.0, cy - side / 2), H - side)
    return (x0, y0, x0 + side, y0 + side)


def _foot_court(res, i: int, box) -> np.ndarray | None:
    if res is None or res.H[i] is None:
        return None
    from replay import homography

    p = homography.project(res.H[i], np.array([[(box[0] + box[2]) / 2, box[3]]], float))[0]
    return p if np.isfinite(p).all() else None


# ----------------------------------------------------------------------------- probes
def gap_probes(data, res, rc: RedetectConfig, size: tuple[int, int]) -> list[Probe]:
    from replay import court
    from replay.detection import PERSON

    rows, n = data.rows, len(data.k)
    fps = float(data.signature.get("process_fps") or 10.0)
    G, seg = _segments(data.motion, n)
    idx_of_k = {int(k): i for i, k in enumerate(data.k)}
    person = np.flatnonzero(rows[:, 6] == PERSON)
    ri = np.array([idx_of_k.get(int(k), -1) for k in rows[person, 0]], int)
    boxes_at: dict[int, np.ndarray] = {}
    for i in np.unique(ri):
        boxes_at[int(i)] = rows[person[ri == i], 1:5]
    max_gap = int(round(rc.max_gap_s * fps))
    step = max(1, int(round(rc.probe_every_s * fps)))

    tracks: dict[int, np.ndarray] = {}  # bt -> person row indices sorted by sample
    bt = rows[person, 7].astype(int)
    for b in np.unique(bt[bt >= 0]):
        m = person[bt == b]
        m = m[np.argsort(ri[bt == b], kind="stable")]
        if len(m) >= rc.min_track_samples:
            tracks[int(b)] = m

    def on_court(r: int) -> bool:  # bench people sit right outside the lines: no margin
        p = _foot_court(res, idx_of_k[int(rows[r, 0])], rows[r, 1:5])
        return p is None or bool(court.in_court(p, rc.court_margin_m)[0])  # unknown calibration: image cues only

    gaps: list[tuple[int, int, int]] = []  # (row a, row b, bt id or -1)
    for b, m in tracks.items():
        ii = np.array([idx_of_k[int(rows[r, 0])] for r in m])
        for j in np.flatnonzero((np.diff(ii) > 1) & (np.diff(ii) <= max_gap)):
            gaps.append((int(m[j]), int(m[j + 1]), b))
    # ID switches: a track ends, a compatible one starts soon after near the same stabilised position
    starts = sorted(((idx_of_k[int(rows[m[0], 0])], b) for b, m in tracks.items()))
    used_b: set[int] = set()
    for a, ma in tracks.items():
        ra = int(ma[-1])
        ia = idx_of_k[int(rows[ra, 0])]
        ba = rows[ra, 1:5]
        ha = ba[3] - ba[1]
        best = None
        for ib, b in starts:
            if ib <= ia + 1 or ib > ia + max_gap or b == a or b in used_b or seg[ib] != seg[ia]:
                continue
            rb = int(tracks[b][0])
            bb = rows[rb, 1:5]
            hb = bb[3] - bb[1]
            if not 0.7 < hb / max(ha, 1) < 1.4:
                continue
            fa = _apply(G[ia], ((ba[0] + ba[2]) / 2, ba[3]))
            fb = _apply(G[ib], ((bb[0] + bb[2]) / 2, bb[3]))
            dt = (ib - ia) / fps
            d = float(np.linalg.norm(fa - fb))
            if d < ha * (0.6 + 1.2 * dt) and (best is None or d < best[0]):
                best = (d, rb, b)
        if best is not None:
            used_b.add(best[2])
            gaps.append((ra, best[1], -1))

    probes: list[Probe] = []
    for ra, rb, b in gaps:
        if not (on_court(ra) and on_court(rb)):
            continue
        ia, ib = idx_of_k[int(rows[ra, 0])], idx_of_k[int(rows[rb, 0])]
        if seg[ia] != seg[ib]:
            continue
        A, B = rows[ra, 1:5], rows[rb, 1:5]
        ha, hb = A[3] - A[1], B[3] - B[1]
        wa, wb = A[2] - A[0], B[2] - B[0]
        fa, fb = _apply(G[ia], ((A[0] + A[2]) / 2, A[3])), _apply(G[ib], ((B[0] + B[2]) / 2, B[3]))
        miss = list(range(ia + 1, ib))
        mid = (len(miss) - 1) / 2
        pick = sorted(miss, key=lambda i: abs(i - (ia + ib) / 2))  # middle first
        chosen: list[int] = []
        for i in pick:
            if all(abs(i - c) >= step for c in chosen):
                chosen.append(i)
        for rank, i in enumerate(chosen):
            if seg[i] != seg[ia]:
                continue
            u = (i - ia) / (ib - ia)
            f = _apply(np.linalg.inv(G[i]), (1 - u) * fa + u * fb)
            h, w = (1 - u) * ha + u * hb, (1 - u) * wa + u * wb
            pred = np.array([f[0] - w / 2, f[1] - h, f[0] + w / 2, f[1]], float)
            fc = _foot_court(res, i, pred)
            if fc is not None and not court.in_court(fc, rc.court_margin_m)[0]:
                continue
            ex = boxes_at.get(i)
            if ex is not None and len(ex) and _iou(pred, ex).max() > 0.6:
                continue  # someone is detected right there already (often the same player under another ID)
            side = max(rc.min_crop_px, rc.crop_scale * h)
            reg = square_region(f[0], f[1] - h / 2, side, size)
            probes.append(Probe(i, reg, "gap", rank + 0.001 * abs(i - mid), pred, b, [ra, rb]))
    return probes


def cluster_probes(data, res, rc: RedetectConfig, size: tuple[int, int]) -> list[Probe]:
    """Overlapping people with a foot in a key area; one probe pair (i, i+1) per side every cluster_every_s."""
    from replay import court, homography
    from replay.detection import PERSON

    if res is None:
        return []
    rows, n = data.rows, len(data.k)
    fps = float(data.signature.get("process_fps") or 10.0)
    idx_of_k = {int(k): i for i, k in enumerate(data.k)}
    person = rows[:, 6] == PERSON
    every = max(1, int(round(rc.cluster_every_s * fps)))
    last = {0: -10**9, 1: -10**9}
    ft, ph = court.FT_LINE_X + rc.key_margin_m, court.PAINT_HALF_WIDTH + rc.key_margin_m
    probes: list[Probe] = []
    G, seg = _segments(data.motion, n)
    by_i: dict[int, np.ndarray] = {}
    pr = np.flatnonzero(person)
    for r, k in zip(pr, rows[pr, 0]):
        by_i.setdefault(idx_of_k[int(k)], []).append(r)
    group = 0
    for i in range(n - 1):
        if res.H[i] is None or i not in by_i or seg[i + 1] != seg[i]:
            continue
        b = rows[np.array(by_i[i]), 1:5]
        p = homography.project(res.H[i], np.c_[(b[:, 0] + b[:, 2]) / 2, b[:, 3]])
        ok = np.isfinite(p).all(axis=1) & court.in_court(p, 0.5)
        if ok.sum() < 2:
            continue
        b, p = b[ok], p[ok]
        in_key = (np.abs(p[:, 1] - court.MID_Y) <= ph) & ((p[:, 0] <= ft) | (p[:, 0] >= court.LENGTH - ft))
        if not in_key.any():
            continue
        w, h = b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]
        ox = np.minimum(b[:, None, 2], b[None, :, 2]) - np.maximum(b[:, None, 0], b[None, :, 0])
        oy = np.minimum(b[:, None, 3], b[None, :, 3]) - np.maximum(b[:, None, 1], b[None, :, 1])
        adj = (ox > 0.3 * np.minimum(w[:, None], w[None])) & (oy > 0.3 * np.minimum(h[:, None], h[None]))
        np.fill_diagonal(adj, False)
        for s in np.flatnonzero(in_key & adj.any(axis=1)):
            side = 0 if p[s, 0] < court.LENGTH / 2 else 1
            if i - last[side] < every:
                continue
            comp = {int(s)}  # connected component through overlaps
            todo = [int(s)]
            while todo:
                q = todo.pop()
                for t in np.flatnonzero(adj[q]):
                    if int(t) not in comp:
                        comp.add(int(t))
                        todo.append(int(t))
            cb = b[sorted(comp)]
            x0, y0, x1, y1 = cb[:, 0].min(), cb[:, 1].min(), cb[:, 2].max(), cb[:, 3].max()
            mh = float(np.median(cb[:, 3] - cb[:, 1]))
            sidepx = max(rc.min_crop_px, (x1 - x0) + mh, 1.5 * mh)
            c = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
            Mi = np.linalg.inv(G[i + 1]) @ G[i]  # sample i -> i+1 pixels
            c2 = _apply(Mi, c)
            probes.append(Probe(i, square_region(c[0], c[1], sidepx, size), "cluster", 0.0, group=group, ref_h=mh))
            probes.append(Probe(i + 1, square_region(c2[0], c2[1], sidepx, size), "cluster", 0.0, group=group,
                                ref_h=mh))
            group += 1
            last[side] = i
    return probes


def build_hints(data, tracking_csv: Path, rc: RedetectConfig) -> list[dict]:
    """Hidden-player hints from a previous pass-2 output: interpolated (predicted) rows of team A/B tracks.

    One hint per predicted row: sample index, court position, expected box height and the pass-1 rows of the
    track's measured sightings right before/after the hidden stretch (colour anchors). With lineup.py on, a slot
    that is temporarily empty shows up exactly like this."""
    import csv

    from replay.detection import PERSON

    idx_of_frame = {int(f): i for i, f in enumerate(data.frame)}
    fps = float(data.signature.get("process_fps") or 10.0)
    tracks: dict[str, list] = {}
    with open(tracking_csv, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r.get("object_type") != "player" or r.get("team") not in ("A", "B"):
                continue
            i = idx_of_frame.get(int(r["frame"]))
            if i is None:
                continue
            box = [float(r[k]) for k in ("img_x1", "img_y1", "img_x2", "img_y2")] if r.get("img_x1") else None
            tracks.setdefault(r["track_id"], []).append((i, float(r["x_m"]), float(r["y_m"]),
                                                         r["interpolated"] == "True", box))
    person = data.rows[:, 6] == PERSON

    def row_of(i: int, box) -> int:
        cand = np.flatnonzero(person & (data.rows[:, 0] == data.k[i]))
        if not len(cand):
            return -1
        d = np.abs(data.rows[cand, 1:5] - np.asarray(box)).max(axis=1)
        return int(cand[np.argmin(d)]) if d.min() < 1.0 else -1

    hints: list[dict] = []
    max_len = int(round(rc.max_hint_gap_s * fps))
    for tr in tracks.values():
        tr.sort(key=lambda a: a[0])
        j = 0
        while j < len(tr):
            if not tr[j][3]:
                j += 1
                continue
            e = j
            while e + 1 < len(tr) and tr[e + 1][3] and tr[e + 1][0] == tr[e][0] + 1:
                e += 1
            before = tr[j - 1] if j > 0 and not tr[j - 1][3] and tr[j - 1][4] else None
            after = tr[e + 1] if e + 1 < len(tr) and not tr[e + 1][3] and tr[e + 1][4] else None
            ends = [a for a in (before, after) if a is not None]
            if ends and e - j + 1 <= max_len:
                anchors = [a for a in (row_of(x[0], x[4]) for x in ends) if a >= 0]
                hs = [x[4][3] - x[4][1] for x in ends]
                if anchors:
                    for q in range(j, e + 1):
                        u = (q - j + 1) / (e - j + 2)
                        h = hs[0] if len(hs) == 1 else (1 - u) * hs[0] + u * hs[1]
                        hints.append({"i": int(tr[q][0]), "xy": [tr[q][1], tr[q][2]], "h": round(float(h), 1),
                                      "anchors": anchors, "gap": [int(tr[j][0]), int(tr[e][0])]})
            j = e + 1
    return hints


def hint_probes(data, res, rc: RedetectConfig, size: tuple[int, int], hints: list[dict]) -> list[Probe]:
    """Probes at the court positions the previous output predicted for hidden players (see build_hints)."""
    from replay import homography
    from replay.detection import PERSON

    if res is None or not hints:
        return []
    fps = float(data.signature.get("process_fps") or 10.0)
    step = max(1, int(round(rc.probe_every_s * fps)))
    rows = data.rows
    person = rows[:, 6] == PERSON
    by_gap: dict[tuple, list[dict]] = {}
    for h in hints:
        by_gap.setdefault((tuple(h["gap"]), tuple(h["anchors"])), []).append(h)
    probes: list[Probe] = []
    for (gap, _), hs in by_gap.items():
        mid = (gap[0] + gap[1]) / 2
        chosen: list[dict] = []
        for h in sorted(hs, key=lambda h: abs(h["i"] - mid)):  # middle of the hidden stretch first
            if all(abs(h["i"] - c["i"]) >= step for c in chosen):
                chosen.append(h)
        for rank, h in enumerate(chosen):
            i = int(h["i"])
            if i >= len(res.H) or res.H[i] is None:
                continue
            f = homography.project(np.linalg.inv(res.H[i]), np.array([h["xy"]], float))[0]
            if not np.isfinite(f).all():
                continue
            hh = float(h["h"])
            pred = np.array([f[0] - 0.2 * hh, f[1] - hh, f[0] + 0.2 * hh, f[1]], float)
            if pred[2] < 0 or pred[0] > size[0] or pred[3] < 0.3 * hh or pred[1] > size[1]:
                continue  # off screen
            ex = rows[person & (rows[:, 0] == data.k[i])][:, 1:5]
            if len(ex) and _iou(pred, ex).max() > 0.6:
                continue  # detected there already
            reg = square_region(f[0], f[1] - hh / 2, max(rc.min_crop_px, rc.crop_scale * 1.2 * hh), size)
            probes.append(Probe(i, reg, "slot", rank + 0.001 * abs(i - mid), pred, -1, list(h["anchors"])))
    return probes


def load_hints(work: Path, data, rc: RedetectConfig) -> list[dict]:
    """Frozen hints: built once from the previous output (out/tracking.csv) and stored in work/redetect_hints.json
    keyed by the pass-1 rows, so later builds do not chase their own output."""
    if not rc.slots:
        return []
    path = work / "redetect_hints.json"
    rk = hashlib.sha1(np.ascontiguousarray(data.rows[:, :8], np.float32).tobytes()).hexdigest()[:20]
    if path.exists():
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
            if doc.get("rows_key") == rk:
                return doc["hints"]
        except (OSError, ValueError, KeyError):
            pass
    csv_path = work.parent / "tracking.csv"
    if not csv_path.exists():
        return []
    try:
        hints = build_hints(data, csv_path, rc)
    except (OSError, KeyError, ValueError) as e:
        print("redetect: could not read hints:", e)
        return []
    try:
        path.write_text(json.dumps({"rows_key": rk, "source": str(csv_path), "hints": hints}), encoding="utf-8")
    except OSError:
        pass
    return hints


def select(probes: list[Probe], budget: int, gap_share: float) -> list[Probe]:
    # gap and slot probes alternate (each in priority order): slots are many and must not starve the gaps
    lists = [sorted((p for p in probes if p.kind == k), key=lambda p: p.priority) for k in ("gap", "slot")]
    gaps = [p for pair in zip(*lists) for p in pair] + [p for lst in lists for p in lst[min(map(len, lists)):]]
    groups: dict[int, list[Probe]] = {}
    for p in probes:
        if p.kind == "cluster":
            groups.setdefault(p.group, []).append(p)
    ng = min(len(gaps), budget, max(int(round(budget * gap_share)), budget - 2 * len(groups)))
    out = gaps[:ng]
    left = budget - len(out)
    gl = list(groups.values())
    if gl and left >= 2:  # spread the cluster pairs evenly over the video
        take = min(len(gl), left // 2)
        for j in np.linspace(0, len(gl) - 1, take).round().astype(int):
            out.extend(gl[int(j)])
    return out


# ----------------------------------------------------------------------------- detection + acceptance
def _hellinger(a: np.ndarray, b: np.ndarray) -> float:
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return np.nan
    return float(np.sqrt(max(0.0, 1.0 - np.sum(np.sqrt(np.clip(a, 0, None) * np.clip(b, 0, None))))))


def detect_in_region(detector, img: np.ndarray, region, threshold: float):
    from replay.detection import PERSON

    x0, y0, x1, y1 = (int(round(v)) for v in region)
    crop = np.ascontiguousarray(img[y0:y1, x0:x1])
    if crop.size == 0:
        return np.zeros((0, 4)), np.zeros(0)
    old = getattr(detector, "threshold", None)
    if old is not None:
        detector.threshold = threshold
    try:
        d = detector.detect(crop)
    finally:
        if old is not None:
            detector.threshold = old
    m = (d.cls == PERSON) & (d.conf >= threshold)
    bx = d.xyxy[m].astype(float) + np.array([x0, y0, x0, y0], float)
    # boxes cut by the crop border (not the image border) are partial people
    H, W = img.shape[:2]
    e = 2.0
    cut = ((bx[:, 0] <= x0 + e) & (x0 > 0)) | ((bx[:, 2] >= x1 - e) & (x1 < W)) | \
          ((bx[:, 1] <= y0 + e) & (y0 > 0)) | ((bx[:, 3] >= y1 - e) & (y1 < H))
    return bx[~cut], d.conf[m][~cut].astype(float)


def _accept_gap(p: Probe, bx, cf, img, existing, rows, anchor_desc, rc) -> tuple[np.ndarray, float] | None:
    from replay.team_appearance import box_descriptor

    pred = p.pred
    ph, pw = pred[3] - pred[1], pred[2] - pred[0]
    tol = 1.5 if p.kind == "slot" else 1.0  # a court-predicted slot position is less certain than a box gap
    best = None
    for b, c in zip(bx, cf):
        h, w = b[3] - b[1], b[2] - b[0]
        if not (0.8 / tol ** 0.5 < h / ph < 1.25 * tol ** 0.5) or h < 1.2 * w:
            continue
        if abs((b[0] + b[2]) / 2 - (pred[0] + pred[2]) / 2) > 0.5 * pw + 0.2 * tol * ph \
                or abs(b[3] - pred[3]) > 0.2 * tol * ph or abs(b[1] - pred[1]) > 0.25 * tol * ph:
            # whole person where expected, not a torso or a merged pair
            continue
        if len(existing) and _iou(b, existing).max() >= rc.dup_iou:
            continue
        d = _hellinger(box_descriptor(img, b), anchor_desc) if anchor_desc is not None else np.nan
        if not np.isfinite(d) or d > rc.max_hellinger:
            continue
        s = c - d
        if best is None or s > best[2]:
            best = (b, float(c), s)
    return None if best is None else (best[0], best[1])


def _accept_cluster(p: Probe, bx, cf, img, existing, res, court_ok, rc) -> list[tuple[np.ndarray, float]]:
    from replay.team_appearance import box_descriptor

    out = []
    ex_desc = [box_descriptor(img, e) for e, okc in zip(existing, court_ok) if okc]
    for b, c in zip(bx, cf):
        h, w = b[3] - b[1], b[2] - b[0]
        if c < rc.cluster_min_conf or h < 1.2 * w or not (0.75 < h / max(p.ref_h, 1) < 1.35):
            continue
        if len(existing) and _iou(b, existing).max() >= rc.dup_iou:
            continue
        from replay import court

        f = _foot_court(res, p.i, b)
        if f is None or not court.in_court(f, 0.5)[0]:
            continue
        d = box_descriptor(img, b)
        dists = [_hellinger(d, e) for e in ex_desc]
        dists = [x for x in dists if np.isfinite(x)]
        if not dists or min(dists) > rc.max_hellinger:
            continue
        out.append((b, float(c)))
    return out


def run_probes(data, res, probes: list[Probe], frames, detector, rc: RedetectConfig) -> tuple[np.ndarray, dict]:
    """frames: iterable of (sample index, BGR image) in increasing order covering every probe sample and the
    gap anchors' samples. Returns (new rows (m, 14), stats)."""
    from replay import court, homography
    from replay.detection import PERSON
    from replay.features import jersey_features
    from replay.team_appearance import box_descriptor

    rows = data.rows
    idx_of_k = {int(k): i for i, k in enumerate(data.k)}
    by_i: dict[int, list[Probe]] = {}
    for p in probes:
        by_i.setdefault(p.i, []).append(p)
    anchor_rows_at: dict[int, set[int]] = {}
    for p in probes:
        for r in p.anchors:
            anchor_rows_at.setdefault(idx_of_k[int(rows[r, 0])], set()).add(r)
    person = rows[:, 6] == PERSON
    desc_of: dict[int, np.ndarray] = {}
    pending: list[tuple[Probe, np.ndarray, list]] = []  # gap probes waiting for their later anchor
    cluster_hits: dict[int, dict[int, list]] = {}
    accepted: list[tuple[int, np.ndarray, float, int, np.ndarray]] = []  # (i, box, conf, bt, feat)
    st = {"probes": 0, "gap_probes": 0, "slot_probes": 0, "cluster_probes": 0, "gap_added": 0, "slot_added": 0,
          "cluster_added": 0, "raw_boxes": 0}

    def flush_pending(final: bool):
        keep = []
        for p, img_small, cand in pending:
            ds = [desc_of.get(r) for r in p.anchors]
            if not final and any(d is None for d in ds):
                keep.append((p, img_small, cand))
                continue
            ds = [d for d in ds if d is not None and np.isfinite(d).all()]
            anchor = np.mean(ds, axis=0) if ds else None
            got = cand(anchor)
            if got is not None:
                accepted.append((p.i, got[0], got[1], p.bt, got[2]))
                st[p.kind + "_added"] += 1
        pending[:] = keep

    for i, img in frames:
        for r in anchor_rows_at.get(i, ()):
            desc_of[r] = box_descriptor(img, rows[r, 1:5])
        here = by_i.get(i, [])
        if here:
            ex = rows[person & (rows[:, 0] == data.k[i])][:, 1:5]
            court_ok = np.ones(len(ex), bool)
            if res is not None and res.H[i] is not None and len(ex):
                pc = homography.project(res.H[i], np.c_[(ex[:, 0] + ex[:, 2]) / 2, ex[:, 3]])
                court_ok = np.isfinite(pc).all(axis=1) & court.in_court(pc, 0.5)
            for p in here:
                bx, cf = detect_in_region(detector, img, p.region, rc.threshold)
                st["probes"] += 1
                st[p.kind + "_probes"] += 1
                st["raw_boxes"] += len(bx)
                if p.kind in ("gap", "slot"):
                    def cand(anchor, p=p, bx=bx, cf=cf, img=img, ex=ex):
                        g = _accept_gap(p, bx, cf, img, ex, rows, anchor, rc)
                        return None if g is None else (g[0], g[1], jersey_features(img, g[0][None])[0])
                    # the image is only needed until the later anchor is decoded: keep candidates' data small
                    pending.append((p, None, cand))
                else:
                    hits = _accept_cluster(p, bx, cf, img, ex, res, court_ok, rc)
                    cluster_hits.setdefault(p.group, {})[i] = [(b, c, jersey_features(img, b[None])[0])
                                                               for b, c in hits]
        flush_pending(False)
    flush_pending(True)

    # cluster: a new person must be found at both samples of the pair at a consistent stabilised position
    G, _ = _segments(data.motion, len(data.k))
    for g, hits in cluster_hits.items():
        if len(hits) != 2:
            continue
        (i0, h0), (i1, h1) = sorted(hits.items())
        M = np.linalg.inv(G[i1]) @ G[i0]
        for b0, c0, f0 in h0:
            fp = _apply(M, ((b0[0] + b0[2]) / 2, b0[3]))
            hgt = b0[3] - b0[1]
            match = [(b1, c1, f1) for b1, c1, f1 in h1
                     if np.hypot((b1[0] + b1[2]) / 2 - fp[0], b1[3] - fp[1]) < 0.35 * hgt]
            if match:
                b1, c1, f1 = match[0]
                accepted.append((i0, b0, c0, -1, f0))
                accepted.append((i1, b1, c1, -1, f1))
                st["cluster_added"] += 2

    # one new box per person and sample
    out: list[np.ndarray] = []
    by_s: dict[int, list] = {}
    for a in accepted:
        by_s.setdefault(a[0], []).append(a)
    for i, items in by_s.items():
        items.sort(key=lambda a: -a[2])
        kept: list = []
        for a in items:
            if all(_iou(a[1], k[1])[0, 0] < rc.dup_iou for k in kept):
                kept.append(a)
        for i_, b, c, btid, f in kept:
            out.append(np.r_[data.k[i_], b, c, PERSON, btid, f].astype(np.float64))
    new = np.array(out, float).reshape(-1, rows.shape[1] if rows.ndim == 2 else 14)
    st["added"] = len(new)
    return new, st


# ----------------------------------------------------------------------------- frames + cache + entry point
def iter_frames(video: Path, data, want: list[int]):
    """Sequentially decode the samples ``want`` (sorted sample indices) -> (i, image)."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        half = 0.5 / fps
        start = float(data.signature.get("start_s") or 0.0)
        seek_to = max(0.0, min(float(data.t[want[0]]), start) - 2.0) if want else 0.0
        if seek_to > 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, seek_to * 1000.0)
        j = 0
        while j < len(want):
            if not cap.grab():
                break
            ft = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if ft + half < float(data.t[want[j]]):
                continue
            while j < len(want) and float(data.t[want[j]]) + half < ft:
                j += 1
            if j >= len(want) or abs(ft - float(data.t[want[j]])) > half:
                continue
            ok, img = cap.retrieve()
            if ok:
                yield int(want[j]), img
            j += 1
    finally:
        cap.release()


def _cache_key(data, rc: RedetectConfig, det_sig: dict, hints: list | None = None) -> str:
    h = hashlib.sha1()
    sig = data.signature
    cfgd = {k: v for k, v in asdict(rc).items() if k != "enabled"}
    h.update(json.dumps({"v": VERSION, "video": sig.get("video"), "size": sig.get("video_size_bytes"),
                         "cfg": cfgd, "det": det_sig, "n": int(len(data.rows))}, sort_keys=True).encode())
    h.update(np.ascontiguousarray(data.rows[:, :8], np.float32).tobytes())
    if hints:
        h.update(json.dumps(hints, sort_keys=True).encode())
    return h.hexdigest()[:20]


def _with_rows(data, new: np.ndarray):
    if not len(new):
        return data
    rows = np.concatenate([data.rows, new.astype(data.rows.dtype)])
    rows = rows[np.argsort(rows[:, 0], kind="stable")]
    return replace(data, rows=rows)


def augment(work: Path, data, res, video, cfg, size: tuple[int, int] | None = None, detector=None):
    """Pass-2 hook: data with second-look person rows appended (unchanged when disabled / no video / no cache)."""
    rc = config_from(cfg)
    if not rc.enabled or not len(data.rows):
        return data
    from replay.team_appearance import resolve_video

    work = Path(work)
    det_cfg = getattr(cfg, "detector", None)
    det_sig = {"variant": getattr(det_cfg, "variant", "medium")}
    hints = load_hints(work, data, rc)
    key = _cache_key(data, rc, det_sig, hints)
    path = work / CACHE
    if path.exists():
        try:
            with np.load(path) as z:  # closed at once: Windows cannot replace an open file later
                hit = str(z["key"]) == key
                new = z["rows"] if hit else None
            if hit:
                print(f"redetect: {len(new)} second-look detections (cached)")
                return _with_rows(data, new)
        except Exception:  # noqa: BLE001 - corrupt/old cache: recompute
            pass
    vid = resolve_video(data, video)
    if vid is None:
        return data
    t0 = time.perf_counter()
    if size is None:
        cap = cv2.VideoCapture(str(vid))
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        cap.release()
    probes = gap_probes(data, res, rc, size) + hint_probes(data, res, rc, size, hints)
    if rc.clusters:
        probes += cluster_probes(data, res, rc, size)
    duration = float(data.t[-1] - data.t[0]) if len(data.t) > 1 else 0.0
    budget = int(round(rc.crops_per_s * max(duration, 1.0)))
    chosen = select(probes, budget, rc.gap_share)
    if not chosen:
        new, st = np.zeros((0, data.rows.shape[1])), {"probes": 0, "added": 0}
    else:
        if detector is None:
            from replay.config import DetectorConfig
            from replay.detection import build_detector

            dc = DetectorConfig(**{**(asdict(det_cfg) if det_cfg is not None else {}), "tiles": (1, 1)})
            detector = build_detector(dc)
        idx_of_k = {int(k): i for i, k in enumerate(data.k)}
        want = {p.i for p in chosen} | {idx_of_k[int(data.rows[r, 0])] for p in chosen for r in p.anchors}
        new, st = run_probes(data, res, chosen, iter_frames(vid, data, sorted(want)), detector, rc)
    st.update(candidates=len(probes), budget=budget, seconds=round(time.perf_counter() - t0, 1))
    print("redetect:", json.dumps(st))
    try:
        tmp = work / "redetect.tmp.npz"
        np.savez_compressed(tmp, key=np.array(key), rows=new, stats=np.array(json.dumps(st)))
        os.replace(tmp, path)
    except OSError:
        pass
    return _with_rows(data, new)


# ----------------------------------------------------------------------------- Task 9 crumbs: gap fillers only
def fill_crumbs(data, res, cfg):
    """Pass-2 step 1c: decide which stored low-score person rows ("crumbs", conf < the run's detector threshold;
    only present when pass 1 ran with DetectorConfig.store_threshold) are kept.

    mode "fill" (default): a crumb survives only where it fills a gap (<= crumb_max_gap_s) inside an existing
    identity: a ByteTrack track of normal-score rows that disappears and resumes (same id), or ends and is
    continued by a compatible track nearby (ID switch). The crumb must sit at the motion-compensated interpolation
    of the gap ends (position/height), match the anchors' torso colour, and not duplicate a normal-score box.
    Accepted crumbs take the gap's ByteTrack id (same-id gaps) so the linker joins them; all other crumbs are
    removed, so they can never start an identity, enter the caps or be drawn on their own.
    mode "all": every crumb is passed on (Task-9 v1, measured worse); "none": all crumbs removed.
    Runs without crumbs are returned unchanged."""
    from replay.detection import PERSON, activation_threshold

    thr = activation_threshold(data.signature)
    rows = data.rows
    if thr <= 0 or not len(rows):
        return data
    rc = config_from(cfg)
    mode = os.environ.get("REPLAY_CRUMBS", rc.crumbs)
    lo_mask = (rows[:, 6] == PERSON) & (rows[:, 5] < thr)
    if not lo_mask.any() or mode == "all":
        return data
    keep = ~lo_mask
    if mode == "fill":
        acc, bt_new = crumb_fills(data, res, rc, thr)
        keep[acc] = True
        if len(acc):
            rows = rows.copy()
            rows[acc, 7] = bt_new
    print(f"crumbs: {int(lo_mask.sum())} low-score rows, kept {int((keep & lo_mask).sum())} as gap fillers ({mode})")
    return replace(data, rows=rows[keep])


def crumb_fills(data, res, rc: RedetectConfig, thr: float) -> tuple[np.ndarray, np.ndarray]:
    """(accepted crumb row indices, ByteTrack id to give each) - see fill_crumbs."""
    from replay import court
    from replay.detection import PERSON

    rows, n = data.rows, len(data.k)
    fps = float(data.signature.get("process_fps") or 10.0)
    G, seg = _segments(data.motion, n)
    idx_of_k = {int(k): i for i, k in enumerate(data.k)}
    ri = np.array([idx_of_k.get(int(k), -1) for k in rows[:, 0]], int)
    person = rows[:, 6] == PERSON
    hi = np.flatnonzero(person & (rows[:, 5] >= thr))
    lo = np.flatnonzero(person & (rows[:, 5] < thr))
    lo_at: dict[int, list[int]] = {}
    for r in lo:
        lo_at.setdefault(int(ri[r]), []).append(int(r))
    hi_at: dict[int, np.ndarray] = {}
    for i in np.unique(ri[hi]):
        hi_at[int(i)] = rows[hi[ri[hi] == i], 1:5]
    max_gap = int(round(rc.crumb_max_gap_s * fps))

    tracks: dict[int, np.ndarray] = {}
    bt = rows[hi, 7].astype(int)
    for b in np.unique(bt[bt >= 0]):
        m = hi[bt == b]
        m = m[np.argsort(ri[m], kind="stable")]
        if len(m) >= rc.min_track_samples:
            tracks[int(b)] = m

    def on_court(r: int) -> bool:
        p = _foot_court(res, int(ri[r]), rows[r, 1:5])
        return p is None or bool(court.in_court(p, rc.court_margin_m)[0])

    gaps: list[tuple[int, int, int]] = []
    for b, m in tracks.items():
        ii = ri[m]
        for j in np.flatnonzero((np.diff(ii) > 1) & (np.diff(ii) <= max_gap)):
            gaps.append((int(m[j]), int(m[j + 1]), b))
    starts = sorted((int(ri[m[0]]), b) for b, m in tracks.items())
    used_b: set[int] = set()
    for a, ma in tracks.items():
        ra = int(ma[-1])
        ia = int(ri[ra])
        ba = rows[ra, 1:5]
        ha = ba[3] - ba[1]
        best = None
        for ib, b in starts:
            if ib <= ia + 1 or ib > ia + max_gap or b == a or b in used_b or seg[ib] != seg[ia]:
                continue
            rb = int(tracks[b][0])
            bb = rows[rb, 1:5]
            if not 0.7 < (bb[3] - bb[1]) / max(ha, 1) < 1.4:
                continue
            fa = _apply(G[ia], ((ba[0] + ba[2]) / 2, ba[3]))
            fb = _apply(G[ib], ((bb[0] + bb[2]) / 2, bb[3]))
            d = float(np.linalg.norm(fa - fb))
            if d < ha * (0.6 + 1.2 * (ib - ia) / fps) and (best is None or d < best[0]):
                best = (d, rb, b)
        if best is not None:
            used_b.add(best[2])
            gaps.append((ra, best[1], -1))

    best_at: dict[tuple[int, int], tuple[float, int, int]] = {}  # (gap, sample) -> (score, crumb row, bt)
    for ra, rb, b in gaps:
        if not (on_court(ra) and on_court(rb)):
            continue
        ia, ib = int(ri[ra]), int(ri[rb])
        if seg[ia] != seg[ib]:
            continue
        A, B = rows[ra, 1:5], rows[rb, 1:5]
        ha, hb, wa, wb = A[3] - A[1], B[3] - B[1], A[2] - A[0], B[2] - B[0]
        fa, fb = _apply(G[ia], ((A[0] + A[2]) / 2, A[3])), _apply(G[ib], ((B[0] + B[2]) / 2, B[3]))
        lab = np.nanmean(rows[[ra, rb], 8:11], axis=0) if np.isfinite(rows[[ra, rb], 8:11]).any() else None
        for i in range(ia + 1, ib):
            gkey = (ra, i)
            if seg[i] != seg[ia] or i not in lo_at:
                continue
            u = (i - ia) / (ib - ia)
            f = _apply(np.linalg.inv(G[i]), (1 - u) * fa + u * fb)
            h, w = (1 - u) * ha + u * hb, (1 - u) * wa + u * wb
            pred = np.array([f[0] - w / 2, f[1] - h, f[0] + w / 2, f[1]], float)
            ex = hi_at.get(i)
            if ex is not None and len(ex) and _iou(pred, ex).max() > 0.6:
                continue  # a normal-score box is already there
            for r in lo_at[i]:
                c = rows[r, 1:5]
                same_bt = b >= 0 and int(rows[r, 7]) == b
                ch, cw = c[3] - c[1], c[2] - c[0]
                if not (0.75 < ch / h < 1.33) or ch < 1.1 * cw:
                    continue
                if abs((c[0] + c[2]) / 2 - f[0]) > 0.5 * w + (0.35 if same_bt else 0.25) * h \
                        or abs(c[3] - f[1]) > (0.3 if same_bt else 0.2) * h:
                    continue
                if ex is not None and len(ex) and _iou(c, ex).max() >= rc.dup_iou:
                    continue
                if lab is not None and np.isfinite(rows[r, 8:11]).all() and not same_bt \
                        and float(np.linalg.norm(rows[r, 8:11] - lab)) > rc.crumb_max_lab_de:
                    continue
                score = float(_iou(c, pred)[0, 0]) + (1.0 if same_bt else 0.0)
                if gkey not in best_at or score > best_at[gkey][0]:
                    best_at[gkey] = (score, r, b)  # one crumb per gap and sample
    acc: dict[int, tuple[float, int]] = {}
    for score, r, b in best_at.values():
        if r not in acc or score > acc[r][0]:
            acc[r] = (score, b)
    if not acc:
        return np.zeros(0, int), np.zeros(0)
    idx = np.array(sorted(acc), int)
    return idx, np.array([acc[r][1] for r in idx], float)
