"""Rich jersey-colour descriptors for team classification, re-sampled from the video in pass 2.

Why: pass 1 stores one median Lab colour per torso. On this kind of footage (side-on players, thin arms,
floor showing through, dark purple kits) that median is often skin or floor, so purple and orange players get
almost the same value. A colour *histogram* of the upper body keeps the minority-but-decisive jersey pixels:
a purple player has a clear purple lobe even when half the box is arm and floor.

Descriptor per person box (DESC_DIM values, sums to 1):
  * chromatic pixels (chroma >= ACHROMA_C): N_HUE hue sectors x N_CHROMA chroma levels
  * achromatic pixels: lightness bins (black, dark grey, light grey, white)
Pixels are weighted by a horizontal Gaussian centred on the box (the person is in the middle, background at
the sides). Boxes largely hidden behind a person standing in front are flagged via ``occlusion``.

Pass 2 decodes a sparse set of frames (one every ``TeamConfig.resample_every_s`` seconds) sequentially - cheap,
because decoding without colour conversion runs at >1000 fps for 720p - and caches the descriptors in the work
folder (``team_appearance.npz``), keyed by the pass-1 run and the sampling settings.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")
import cv2  # noqa: E402

DESC_VERSION = 3
BEHIND_WEIGHT = 0.5  # a person behind (overlapping the box) counts this much as occlusion
N_HUE = 12  # 30 degree sectors, sector 0 centred on hue 0 (+a axis)
CHROMA_EDGES = (7.0, 14.0, 24.0)  # chromatic levels: [7,14), [14,24), [24,inf)
N_CHROMA = len(CHROMA_EDGES)
ACHROMA_C = CHROMA_EDGES[0]
L_EDGES = (20.0, 45.0, 70.0)  # achromatic lightness bins: black | dark grey | light grey | white
N_ACH = len(L_EDGES) + 1
DESC_DIM = N_HUE * N_CHROMA + N_ACH
REGION = (0.12, 0.62, 0.12, 0.88)  # (y0, y1, x0, x1) of the box: shirt + shorts
X_SIGMA = 0.22  # horizontal Gaussian weight, fraction of the box width
MAX_SIDE = 40  # region is resized to at most this many pixels per side (speed; colours survive)
MIN_BOX_H = 24  # px; smaller boxes give no descriptor
# dense sampling where people crowd (identity agent): pile-ups are where the tracker splits people into tiny pieces
# that the regular grid (every resample_every_s) misses entirely; there every sample is described, but only the
# boxes that overlap someone (CROWD_OVERLAP of the smaller box)
# kit histogram only from the torso pixels nobody in front covers (identity agent, 18.10): a sample whose visible
# share is below MASK_MIN_VISIBLE gives no descriptor
# broadcast footage: small players, the wooden floor fills much of the shirt+shorts region -> torso only, and pixels
# close to the floor colour (estimated per frame below the people's feet) are masked (env AA_KIT_FLOOR=1)
FLOOR_MASK = os.environ.get("AA_KIT_FLOOR") == "1"  # set per build by set_broadcast_mode (env AA_KIT_FLOOR=0/1 forces)
BROADCAST_MAX_PERSPECTIVE = 3.5  # ground-scale ratio mid/bottom of the image: low gym camera 4.8-5.6, elevated
# broadcast / high-school cameras 1.8-2.7 (measured 18.10 on c300, c900, b3min, NCAA, hs_hazen, tbf_turkey, ven)


def perspective_ratio(H_list, size):
    """Median over calibrated samples of (court metres per pixel at 45 % image height) / (at 90 %): large for a low
    camera (strong perspective), small for an elevated broadcast camera."""
    from replay import homography

    W, Hh = size
    rs = []
    for H in list(H_list)[::10]:
        if H is None:
            continue
        sc = []
        for y in (0.45 * Hh, 0.9 * Hh):
            p = homography.project(H, np.array([[W / 2 - 5, y], [W / 2 + 5, y]], float))
            sc.append(float(np.linalg.norm(p[1] - p[0])) / 10.0)
        if np.isfinite(sc).all() and sc[1] > 0:
            rs.append(sc[0] / sc[1])
    return float(np.median(rs)) if rs else None


def set_broadcast_mode(H_list, size) -> bool:
    """Broadcast footage (elevated camera, small players over a big floor) -> floor-masked torso kit histograms and
    zebra referees (FLOOR_MASK). Home-gym footage keeps the default descriptors exactly. Logged."""
    global FLOOR_MASK
    env = os.environ.get("AA_KIT_FLOOR")
    if env in ("0", "1"):
        FLOOR_MASK = env == "1"
        why = f"forced by AA_KIT_FLOOR={env}"
    else:
        r = perspective_ratio(H_list, size)
        FLOOR_MASK = r is not None and r < BROADCAST_MAX_PERSPECTIVE
        why = f"perspective ratio {None if r is None else round(r, 2)} (broadcast below {BROADCAST_MAX_PERSPECTIVE})"
    print(f"camera type: {'broadcast' if FLOOR_MASK else 'gym side camera'} - {why}; kit colours "
          f"{'from the torso with the floor colour masked, striped referees' if FLOOR_MASK else 'default'}")
    return FLOOR_MASK
TORSO_REGION = (0.14, 0.45, 0.2, 0.8)
FLOOR_DELTA = 14.0  # Lab distance (lightness half-weighted) counted as floor
ZEBRA_MIN_H = 80.0  # px; referee stripes are measured on large boxes only (with FLOOR_MASK, broadcast mode)
ZEBRA_LIGHT, ZEBRA_DARK = 62.0, 35.0
ZEBRA_MIN_TRANS = 4
MASK_OCCLUDERS = os.environ.get("AA_KIT_MASK", "0") == "1"
MASK_MIN_VISIBLE = 0.4
DENSE_CROWDS = os.environ.get("AA_DENSE_KIT", "0") == "1"  # gated 18.10: phantoms/IDs worse -> off
CROWD_OVERLAP = 0.2


# ----------------------------------------------------------------------------- bins
def bin_index(L: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Descriptor bin per Lab pixel (real Lab units)."""
    C = np.hypot(a, b)
    hue = np.degrees(np.arctan2(b, a)) % 360.0
    hs = np.floor(((hue + 180.0 / N_HUE) % 360.0) / (360.0 / N_HUE)).astype(int) % N_HUE
    cl = np.searchsorted(np.asarray(CHROMA_EDGES), C, side="right") - 1  # -1 = achromatic
    lb = np.searchsorted(np.asarray(L_EDGES), L, side="right")
    return np.where(cl >= 0, hs * N_CHROMA + np.maximum(cl, 0), N_HUE * N_CHROMA + lb)


def bin_centers() -> np.ndarray:
    """Representative Lab colour of each bin (DESC_DIM, 3) - for display colours and debugging."""
    out = np.zeros((DESC_DIM, 3))
    cmid = [10.0, 19.0, 34.0]
    for h in range(N_HUE):
        ang = np.radians(h * 360.0 / N_HUE)
        for c in range(N_CHROMA):
            out[h * N_CHROMA + c] = (45.0, cmid[c] * np.cos(ang), cmid[c] * np.sin(ang))
    for j, L in enumerate((10.0, 32.0, 57.0, 85.0)):
        out[N_HUE * N_CHROMA + j] = (L, 0.0, 0.0)
    return out


def chromatic_slice() -> slice:
    return slice(0, N_HUE * N_CHROMA)


def hue_mass(desc: np.ndarray) -> np.ndarray:
    """(..., DESC_DIM) -> (..., N_HUE, N_CHROMA) chromatic part."""
    return desc[..., : N_HUE * N_CHROMA].reshape(*desc.shape[:-1], N_HUE, N_CHROMA)


# ----------------------------------------------------------------------------- per box
def lab_float(image_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[..., 0] *= 100.0 / 255.0
    lab[..., 1:] -= 128.0
    return lab


def box_descriptor(image_bgr: np.ndarray, box) -> np.ndarray:
    """Weighted colour histogram of the shirt+shorts region of one person box; NaN when too small."""
    return box_analysis(image_bgr, box)[0]


def floor_lab(image_bgr: np.ndarray, boxes: np.ndarray) -> np.ndarray | None:
    """Median Lab of the court floor just below the people's feet in this frame (None when too little is visible)."""
    H, W = image_bgr.shape[:2]
    px = []
    for x1, y1, x2, y2 in np.asarray(boxes, float).reshape(-1, 4):
        h = y2 - y1
        ya, yb = int(y2 + 0.02 * h), int(min(H, y2 + 0.12 * h))
        xa, xb = int(max(0, x1 - 0.3 * (x2 - x1))), int(min(W, x2 + 0.3 * (x2 - x1)))
        if yb - ya >= 2 and xb - xa >= 4:
            px.append(lab_float(image_bgr[ya:yb, xa:xb]).reshape(-1, 3))
    if not px:
        return None
    px = np.concatenate(px)
    return np.median(px, axis=0) if len(px) >= 50 else None


def zebra_score(image_bgr: np.ndarray, box) -> float:
    """Referee shirt ("zebra"): vertical black/white stripes, measured on LIGHTNESS only (broadcast colour casts make
    white read as coloured). Torso columns are classed light (L > ZEBRA_LIGHT) / dark (L < ZEBRA_DARK) by majority;
    score = column consistency x light/dark balance when the column pattern alternates >= ZEBRA_MIN_TRANS times.
    NaN for boxes lower than ZEBRA_MIN_H px (stripes are 1-2 px there)."""
    x1, y1, x2, y2 = (float(v) for v in box)
    w, h = x2 - x1, y2 - y1
    if h < ZEBRA_MIN_H or w < 10:
        return float("nan")
    H, W = image_bgr.shape[:2]
    ya, yb = max(0, int(y1 + 0.14 * h)), min(H, int(y1 + 0.42 * h))
    xa, xb = max(0, int(x1 + 0.2 * w)), min(W, int(x1 + 0.8 * w))
    if yb - ya < 4 or xb - xa < 8:
        return float("nan")
    L = cv2.cvtColor(image_bgr[ya:yb, xa:xb], cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32) * (100.0 / 255.0)
    light, dark = L > ZEBRA_LIGHT, L < ZEBRA_DARK
    n = np.maximum((light | dark).sum(axis=0), 1)
    fl = light.sum(axis=0) / n
    valid = (light | dark).mean(axis=0) > 0.5
    if valid.sum() < 6:
        return 0.0
    pat = np.where(fl > 0.7, 1, np.where(fl < 0.3, -1, 0))[valid]
    nz = pat[pat != 0]
    trans = int((np.diff(nz) != 0).sum()) if len(nz) > 1 else 0
    if trans < ZEBRA_MIN_TRANS:
        return 0.0
    cons = float(np.abs(2 * fl[valid] - 1).mean())
    pl = float(light[:, valid].mean())
    balance = min(pl, 1.0 - pl) / 0.5
    return cons * balance


def box_analysis(image_bgr: np.ndarray, box, others: np.ndarray | None = None, floor: np.ndarray | None = None
                 ) -> tuple[np.ndarray, np.ndarray]:
    """(histogram, mean Lab of the clearly coloured pixels per hue sector (N_HUE, 3), NaN where none).

    The per-sector colours give the real jersey colour for display (the histogram only knows the sector)."""
    x1, y1, x2, y2 = (float(v) for v in box)
    w, h = x2 - x1, y2 - y1
    out = np.full(DESC_DIM, np.nan, np.float32)
    cols = np.full((N_HUE, 3), np.nan, np.float32)
    if h < MIN_BOX_H or w < 6:
        return out, cols
    H, W = image_bgr.shape[:2]
    reg = TORSO_REGION if FLOOR_MASK else REGION
    ya, yb = int(y1 + reg[0] * h), int(y1 + reg[1] * h)
    xa, xb = int(x1 + reg[2] * w), int(x1 + reg[3] * w)
    ya, yb, xa, xb = max(0, ya), min(H, yb), max(0, xa), min(W, xb)
    if yb - ya < 4 or xb - xa < 3:
        return out, cols
    patch = image_bgr[ya:yb, xa:xb]
    s = min(1.0, MAX_SIDE / max(patch.shape[:2]))
    if s < 1.0:
        patch = cv2.resize(patch, (max(2, round(patch.shape[1] * s)), max(2, round(patch.shape[0] * s))),
                           interpolation=cv2.INTER_AREA)
    lab = lab_float(patch).reshape(-1, 3)
    ph, pw = patch.shape[:2]
    # pixel x in box-width units, relative to the box centre
    xs = (xa + (np.arange(pw) + 0.5) * (xb - xa) / pw - (x1 + x2) / 2) / max(w, 1.0)
    wx = np.exp(-0.5 * (xs / X_SIGMA) ** 2)
    wt = np.tile(wx, ph)
    if MASK_OCCLUDERS and others is not None and len(others):
        # pixels covered by a person standing IN FRONT (box bottom lower in the image) show their kit, not ours
        o = np.asarray(others, float).reshape(-1, 4)
        front = (o[:, 3] > y2 + 0.02 * h) & ~np.all(np.isclose(o, [x1, y1, x2, y2]), axis=1)
        o = o[front & (o[:, 0] < xb) & (o[:, 2] > xa) & (o[:, 1] < yb) & (o[:, 3] > ya)]
        if len(o):
            px = xa + (np.arange(pw) + 0.5) * (xb - xa) / pw
            py = ya + (np.arange(ph) + 0.5) * (yb - ya) / ph
            cov = np.zeros((ph, pw), bool)
            for ox1, oy1, ox2, oy2 in o:
                cov |= ((py >= oy1) & (py <= oy2))[:, None] & ((px >= ox1) & (px <= ox2))[None, :]
            vis = ~cov.reshape(-1)
            if (wt * vis).sum() < MASK_MIN_VISIBLE * wt.sum():
                return out, cols
            wt = wt * vis
    if FLOOR_MASK and floor is not None:  # floor showing between arms / beside a thin player is not the kit
        near = np.linalg.norm((lab - floor) * (0.5, 1.0, 1.0), axis=1) < FLOOR_DELTA
        if (wt * ~near).sum() < 0.3 * wt.sum():
            return out, cols
        wt = wt * ~near
    idx = bin_index(lab[:, 0], lab[:, 1], lab[:, 2])
    hist = np.bincount(idx, weights=wt, minlength=DESC_DIM).astype(np.float32)
    tot = hist.sum()
    strong = idx < N_HUE * N_CHROMA
    strong &= (idx % N_CHROMA) >= 1
    if strong.any():
        sec = idx[strong] // N_CHROMA
        ws = wt[strong]
        sw = np.bincount(sec, weights=ws, minlength=N_HUE)
        for c in range(3):
            v = np.bincount(sec, weights=ws * lab[strong, c], minlength=N_HUE)
            with np.errstate(invalid="ignore", divide="ignore"):
                cols[:, c] = np.where(sw > 0, v / np.maximum(sw, 1e-9), np.nan)
    return (hist / tot if tot > 0 else out), cols


def occlusion(boxes: np.ndarray) -> np.ndarray:
    """Share of each box's shirt region covered by other people's boxes (people behind count half)."""
    boxes = np.asarray(boxes, float).reshape(-1, 4)
    n = len(boxes)
    occ = np.zeros(n)
    for i in range(n):
        x1, y1, x2, y2 = boxes[i]
        w, h = x2 - x1, y2 - y1
        rx1, rx2 = x1 + REGION[2] * w, x1 + REGION[3] * w
        ry1, ry2 = y1 + REGION[0] * h, y1 + REGION[1] * h
        area = max(1e-6, (rx2 - rx1) * (ry2 - ry1))
        cover = 0.0
        for j in range(n):
            if j == i:
                continue
            ix = max(0.0, min(rx2, boxes[j, 2]) - max(rx1, boxes[j, 0]))
            iy = max(0.0, min(ry2, boxes[j, 3]) - max(ry1, boxes[j, 1]))
            # j's feet higher in the image = j is behind: its kit still shows around our person's outline
            behind = boxes[j, 3] < y2 - 0.02 * h
            cover = max(cover, ix * iy / area * (BEHIND_WEIGHT if behind else 1.0))
        occ[i] = cover
    return occ


def image_descriptors(image_bgr: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    return image_analysis(image_bgr, boxes)[0]


def image_analysis(image_bgr: np.ndarray, boxes: np.ndarray, others: np.ndarray | None = None
                   ) -> tuple[np.ndarray, np.ndarray]:
    """others: all person boxes of the frame (for MASK_OCCLUDERS); default = boxes."""
    boxes = np.asarray(boxes, float).reshape(-1, 4)
    if not len(boxes):
        return np.zeros((0, DESC_DIM)), np.zeros((0, N_HUE, 3))
    others = boxes if others is None else np.asarray(others, float).reshape(-1, 4)
    floor = floor_lab(image_bgr, others) if FLOOR_MASK else None
    res = [box_analysis(image_bgr, b, others, floor) for b in boxes]
    return np.stack([r[0] for r in res]), np.stack([r[1] for r in res])


def descriptor_from_lab(labs: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Pseudo-descriptor from a few Lab colours (e.g. the pass-1 torso/shorts medians) - fallback without video."""
    labs = np.asarray(labs, float).reshape(-1, 3)
    w = np.ones(len(labs)) if weights is None else np.asarray(weights, float)
    ok = np.isfinite(labs).all(axis=1)
    out = np.full(DESC_DIM, np.nan)
    if not ok.any():
        return out
    idx = bin_index(labs[ok, 0], labs[ok, 1], labs[ok, 2])
    h = np.bincount(idx, weights=w[ok], minlength=DESC_DIM)
    return h / h.sum()


# ----------------------------------------------------------------------------- pass-2 resampling + cache
def pseudo_descriptors(feats: np.ndarray, shorts_weight: float = 0.35) -> np.ndarray:
    """Per-detection pseudo-descriptors from the 6 pass-1 features (torso Lab, shorts Lab).

    Each detection becomes a two-spike histogram; averaged over a track it is the distribution of its
    torso/shorts colours. Much weaker than real descriptors, but lets pass 2 run without the video."""
    f = np.asarray(feats, float).reshape(-1, 6)
    n = len(f)
    out = np.zeros((n, DESC_DIM))
    rows = np.arange(n)
    for part, w in ((f[:, :3], 1.0 - shorts_weight), (f[:, 3:], shorts_weight)):
        ok = np.isfinite(part).all(axis=1)
        idx = bin_index(*np.nan_to_num(part).T)
        np.add.at(out, (rows[ok], idx[ok]), w)
    tot = out.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = out / tot
    out[tot[:, 0] <= 0] = np.nan
    return out


@dataclass
class Appearance:
    desc: np.ndarray  # (n_rows, DESC_DIM) float32; NaN rows = not sampled / not a person / too small
    occ: np.ndarray  # (n_rows,) share of the shirt hidden by someone in front; NaN = not sampled
    rows: np.ndarray | None = None  # pass-1 row index of each entry, when this is a subset
    height: np.ndarray | None = None  # box height (px) per entry
    samples: dict | None = None  # user-picked reference players: {"A"|"B"|"referee": (n, DESC_DIM) descriptors}
    sector_lab: np.ndarray | None = None  # (n_rows, N_HUE, 3) mean Lab of the clearly coloured pixels per hue sector
    zebra: np.ndarray | None = None  # (n_rows,) referee stripe score (zebra_score), FLOOR_MASK only; NaN = not measured
    clip: np.ndarray | None = None  # (n_rows, 512) CLIP image embeddings (clip_roles.py), broadcast mode only; NaN rows

    def subset(self, rows: np.ndarray, height: np.ndarray | None = None) -> "Appearance":
        rows = np.asarray(rows, int)
        return Appearance(self.desc[rows], self.occ[rows], rows, height, None,
                          None if self.sector_lab is None else self.sector_lab[rows],
                          None if self.zebra is None else self.zebra[rows])


def _cache_key(data, every_s: float) -> str:
    sig = data.signature
    h = hashlib.sha1()
    key = {"v": DESC_VERSION, "video": sig.get("video"), "size": sig.get("video_size_bytes"),
           "every": round(float(every_s), 4), "region": REGION, "n": int(len(data.rows))}
    # experiment options enter the key ONLY when on: the default key must stay the one every existing cache has
    # (benchmark clips without a video cannot recompute a cache, and a changed default key silently drops the kits)
    if DENSE_CROWDS:
        key["dense"] = [DENSE_CROWDS, CROWD_OVERLAP]
    if MASK_OCCLUDERS:
        key["mask"] = [MASK_OCCLUDERS, MASK_MIN_VISIBLE]
    if FLOOR_MASK:
        key["floor"] = [FLOOR_MASK, TORSO_REGION, FLOOR_DELTA, "zebra1", ZEBRA_MIN_H, ZEBRA_LIGHT, ZEBRA_DARK]
    h.update(json.dumps(key, sort_keys=True).encode())
    h.update(np.ascontiguousarray(data.rows[:, :8], np.float32).tobytes())
    return h.hexdigest()[:20]


def sample_frames(t: np.ndarray, every_s: float) -> np.ndarray:
    """Indices of samples on a regular time grid (at least every_s apart)."""
    keep, last = [], -np.inf
    for i, ti in enumerate(np.asarray(t, float)):
        if ti - last >= every_s - 1e-6:
            keep.append(i)
            last = ti
    return np.array(keep, int)


def crowded_rows(rows: np.ndarray, row_i: np.ndarray, person: np.ndarray, frac: float = CROWD_OVERLAP) -> dict:
    """{sample index: person rows whose box overlaps another person's box by more than frac of the smaller one}."""
    out: dict[int, list[int]] = {}
    idx = np.flatnonzero(person & (row_i >= 0))
    order = idx[np.argsort(row_i[idx], kind="stable")]
    ii = row_i[order]
    for grp in np.split(order, np.flatnonzero(np.diff(ii)) + 1):
        if len(grp) < 2:
            continue
        b = rows[grp, 1:5].astype(float)
        ix = np.clip(np.minimum(b[:, None, 2], b[None, :, 2]) - np.maximum(b[:, None, 0], b[None, :, 0]), 0, None)
        iy = np.clip(np.minimum(b[:, None, 3], b[None, :, 3]) - np.maximum(b[:, None, 1], b[None, :, 1]), 0, None)
        area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        f = ix * iy / np.maximum(1.0, np.minimum(area[:, None], area[None, :]))
        np.fill_diagonal(f, 0.0)
        hit = f.max(axis=1) > frac
        if hit.any():
            out[int(row_i[grp[0]])] = [int(r) for r in grp[hit]]
    return out


def compute_appearance(data, video: str | Path, every_s: float, progress=None) -> Appearance | None:
    """Decode the sampled frames sequentially and describe every person box in them."""
    from replay.detection import PERSON

    rows = data.rows
    n = len(rows)
    desc = np.full((n, DESC_DIM), np.nan, np.float32)
    occ = np.full(n, np.nan, np.float32)
    sector_lab = np.full((n, N_HUE, 3), np.nan, np.float32)
    zebra = np.full(n, np.nan, np.float32)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return None
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        half = 0.5 / fps
        want = sample_frames(data.t, every_s)
        idx_of_k = {int(k): i for i, k in enumerate(data.k)}
        row_i = np.array([idx_of_k.get(int(k), -1) for k in rows[:, 0]], int)
        from replay.detection import activation_threshold

        person = (rows[:, 6] == PERSON) & (rows[:, 5] >= activation_threshold(data.signature))  # no Task-9 crumbs
        rows_at: dict[int, np.ndarray] = {}
        for r in np.where(person)[0]:
            rows_at.setdefault(int(row_i[r]), []).append(r)
        want = [int(i) for i in want if i in rows_at]
        grid = set(want)
        crowd = crowded_rows(rows, row_i, person) if DENSE_CROWDS else {}
        extra = {i: r for i, r in crowd.items() if i not in grid}  # off-grid samples: only the crowded boxes
        want = np.array(sorted(grid | set(extra)), int)
        if not len(want):
            return Appearance(desc, occ, sector_lab=sector_lab)
        start = float(data.signature.get("start_s") or 0.0)
        first_t = float(data.t[want[0]])
        seek_to = max(0.0, min(first_t, start) - 2.0)
        if seek_to > 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, seek_to * 1000.0)
        j = 0
        while j < len(want):
            if not cap.grab():
                break
            ft = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            target = float(data.t[want[j]])
            if ft + half < target:
                continue
            while j < len(want) and float(data.t[want[j]]) + half < ft:  # target missed (gap in the source)
                j += 1
            if j >= len(want) or abs(ft - float(data.t[want[j]])) > half:
                continue
            ok, img = cap.retrieve()
            if ok:
                allr = np.array(rows_at[int(want[j])], int)
                rr = np.array(extra.get(int(want[j])) or allr, int)
                desc[rr], sector_lab[rr] = image_analysis(img, rows[rr, 1:5], rows[allr, 1:5])
                if FLOOR_MASK:
                    zebra[rr] = [zebra_score(img, b) for b in rows[rr, 1:5]]
                o = occlusion(rows[allr, 1:5])  # occlusion against everybody in the frame
                occ[rr] = o[np.searchsorted(allr, rr)] if len(rr) != len(allr) else o
            j += 1
            if progress is not None and j % 50 == 0:
                progress(j / len(want))
    finally:
        cap.release()
    return Appearance(desc, occ, sector_lab=sector_lab, zebra=zebra if FLOOR_MASK else None)


def resolve_video(data, video: str | Path | None) -> Path | None:
    candidates = [Path(v) for v in (video, data.signature.get("video")) if v]
    return next((v for v in candidates if v.exists()), None)


def load_or_compute(work: Path, data, video: str | Path | None, every_s: float) -> Appearance | None:
    """Cached descriptors for every person row; None when the video is unavailable and nothing is cached."""
    work = Path(work)
    path = work / "team_appearance.npz"
    key = _cache_key(data, every_s)
    if path.exists():
        try:
            with np.load(path) as z:  # closed again: on Windows an open file cannot be replaced
                if str(z["key"]) == key:
                    return Appearance(z["desc"].astype(np.float32), z["occ"].astype(np.float32),
                                      sector_lab=z["sector_lab"].astype(np.float32),
                                      zebra=z["zebra"].astype(np.float32) if "zebra" in z.files else None)
        except Exception:  # noqa: BLE001 - corrupt/old cache: recompute
            pass
    vid = resolve_video(data, video)
    if vid is None:
        if path.exists():
            print("WARNING: team appearance cache does not match this build and the video is unavailable - "
                  "team colours fall back to the pass-1 colour medians (team/referee accuracy drops)", flush=True)
        return None
    app = compute_appearance(data, vid, every_s)
    if app is None:
        return None
    try:
        save_appearance(work, key, app)
    except OSError:
        pass
    return app


def find_sample_rows(rows: np.ndarray, sample_feats: list) -> list[int]:
    """Pass-1 row indices whose stored jersey features equal the user-picked samples (the GUI stores features)."""
    out = []
    f = rows[:, 8:14].astype(np.float64)
    for s in sample_feats:
        s = np.asarray(s, np.float64).reshape(-1)[:6]
        both_nan = np.isnan(f) & np.isnan(s)
        eq = (np.abs(f - s) < 1e-3) | both_nan
        hit = np.flatnonzero(eq.all(axis=1) & (rows[:, 6] == 0))
        if len(hit):
            out.append(int(hit[0]))
    return out


def describe_rows(data, video: str | Path, row_idx: list[int]) -> dict[int, np.ndarray]:
    """Descriptors of a few specific rows (seeks per frame; meant for a handful of user-picked samples)."""
    from replay.video import read_frame_at

    out: dict[int, np.ndarray] = {}
    idx_of_k = {int(k): i for i, k in enumerate(data.k)}
    by_frame: dict[int, list[int]] = {}
    for r in row_idx:
        i = idx_of_k.get(int(data.rows[r, 0]))
        if i is not None:
            by_frame.setdefault(i, []).append(r)
    for i, rr in by_frame.items():
        got = read_frame_at(video, float(data.t[i]))
        if got is None:
            continue
        for r in rr:
            d = box_descriptor(got[1], data.rows[r, 1:5])
            if np.isfinite(d).all():
                out[r] = d
    return out


def save_appearance(work: Path, key: str, app: Appearance) -> None:
    """Write work/team_appearance.npz (the cache load_or_compute reads), atomically."""
    tmp = Path(work) / "team_appearance.tmp.npz"
    extra = {} if app.zebra is None else {"zebra": app.zebra.astype(np.float16)}
    np.savez_compressed(tmp, key=np.array(key), desc=app.desc.astype(np.float16), occ=app.occ.astype(np.float16),
                        sector_lab=app.sector_lab.astype(np.float16), **extra)
    os.replace(tmp, Path(work) / "team_appearance.npz")


class AppearanceTap:
    """Pass-1 frame tap (replay.frame_tap): describes the person boxes of the frames pass 1 decodes, exactly like
    compute_appearance (same grid, same boxes, same crowded off-grid boxes) and writes work/team_appearance.npz
    with the key pass 2 checks, so pass 2 does not decode the video again. Off for resumed pass-1 runs."""

    name = "team_colours"
    queue_size = 16
    blocking = True

    def __init__(self, every_s: float, enabled: bool = True, clip_every_s: float | None = None):
        self.every_s, self.enabled = float(every_s), enabled
        # CLIP person crops (clip_roles bench cue) on their own sparse grid; 0/None = off (only when a consumer
        # such as FilterConfig.clip_bench is on, see pass1_taps.py); env AA_CLIP_TAP_S overrides
        env = os.environ.get("AA_CLIP_TAP_S")
        self.clip_every_s = float(env) if env else (0.0 if clip_every_s is None else float(clip_every_s))  # default off

    def start(self, info) -> None:
        self.active = bool(self.enabled and not info.resumed)
        self.last_t = -np.inf
        self.res: dict[int, tuple] = {}  # row -> (desc, sector_lab, occ)
        self.clip_last = -np.inf
        self.clip_rows: list[int] = []
        self.clip_jpg: list[bytes] = []
        self.clip_s = 0.0

    def wants(self, k: int, t: float) -> bool:
        if not self.active:
            return False
        on = t - self.last_t >= self.every_s - 1e-6  # the greedy grid of sample_frames(data.t, every_s)
        if on:
            self.last_t = t
        clip = self.clip_every_s > 0 and t - self.clip_last >= self.clip_every_s - 1e-6
        if clip:
            self.clip_last = t
        self._on = getattr(self, "_on", {})
        self._on[k] = (on, clip)
        return on or clip or DENSE_CROWDS  # crowded boxes of off-grid samples are described too

    def process(self, f) -> None:
        on, clip = self._on.pop(f.k, (False, False))
        n = len(f.persons)
        if not n:
            return
        if clip:
            tc = time.perf_counter()
            for j, box in enumerate(np.asarray(f.persons, float).reshape(-1, 4)):
                c = clip_box(f.image, box)
                if c is not None:  # box pixels, lossless PNG (JPEG cost CLIP up to 0.11 cosine; WebP 15 ms/crop)
                    ok, buf = cv2.imencode(".png", c, [cv2.IMWRITE_PNG_COMPRESSION, 1])
                    if ok:
                        self.clip_rows.append(int(f.row0 + j))
                        self.clip_jpg.append(buf.tobytes())
            self.clip_s += time.perf_counter() - tc
        if not on and not DENSE_CROWDS:
            return
        b = np.asarray(f.persons, float).reshape(-1, 4)
        allr = f.row0 + np.arange(n)
        if on:
            sel = np.arange(n)
        else:
            if n < 2:
                return
            ix = np.clip(np.minimum(b[:, None, 2], b[None, :, 2]) - np.maximum(b[:, None, 0], b[None, :, 0]), 0, None)
            iy = np.clip(np.minimum(b[:, None, 3], b[None, :, 3]) - np.maximum(b[:, None, 1], b[None, :, 1]), 0, None)
            area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
            fr = ix * iy / np.maximum(1.0, np.minimum(area[:, None], area[None, :]))
            np.fill_diagonal(fr, 0.0)
            sel = np.flatnonzero(fr.max(axis=1) > CROWD_OVERLAP)
            if not len(sel):
                return
        d, s = image_analysis(f.image, b[sel], b)
        o = occlusion(b)  # occlusion against everybody in the frame
        zb = [zebra_score(f.image, bb) for bb in b[sel]] if FLOOR_MASK else [np.nan] * len(sel)
        for j, r in enumerate(allr[sel]):
            self.res[int(r)] = (d[j], s[j], o[sel[j]], zb[j])

    def finish(self, work: Path, completed: bool) -> None:
        if not self.active or not completed:
            return
        from replay.pipeline import load_pass1

        data = load_pass1(Path(work))
        n = len(data.rows)
        desc = np.full((n, DESC_DIM), np.nan, np.float32)
        occ = np.full(n, np.nan, np.float32)
        sector_lab = np.full((n, N_HUE, 3), np.nan, np.float32)
        zebra = np.full(n, np.nan, np.float32)
        for r, (d, s, o, zb) in self.res.items():
            desc[r], sector_lab[r], occ[r], zebra[r] = d, s, o, zb
        save_appearance(work, _cache_key(data, self.every_s),
                        Appearance(desc, occ, sector_lab=sector_lab, zebra=zebra if FLOOR_MASK else None))
        if self.clip_rows:
            save_clip_crops(work, data, self.clip_every_s, self.clip_rows, self.clip_jpg, self.clip_s)


# ---------------------------------------------------------------------------- CLIP person crops from pass 1
CLIP_TAP_EVERY_S = 2.0  # one frame every 2 s: all person boxes >= CLIP_MIN_BOX_H (clip_roles bench cue)
CLIP_SIZE = 224
CLIP_MIN_BOX_H = 40


def clip_box(img: np.ndarray, box) -> np.ndarray | None:
    """The person box pixels exactly as clip_roles.crop cuts them; None for boxes < CLIP_MIN_BOX_H high or < 8 px
    wide."""
    x1, y1, x2, y2 = (int(round(v)) for v in box[:4])
    H, W = img.shape[:2]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
    if y2 - y1 < CLIP_MIN_BOX_H or x2 - x1 < 8:
        return None
    return img[y1:y2, x1:x2]


def clip_pad(c: np.ndarray) -> np.ndarray:
    """clip_roles.crop without the normalisation: box pixels padded to a square (grey 127), resized to 224."""
    h, w = c.shape[:2]
    s = max(h, w)
    sq = np.full((s, s, 3), 127, np.uint8)
    sq[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = c
    return cv2.resize(sq, (CLIP_SIZE, CLIP_SIZE), interpolation=cv2.INTER_AREA)


def clip_square(img: np.ndarray, box) -> np.ndarray | None:
    c = clip_box(img, box)
    return None if c is None else clip_pad(c)


def _clip_key(data, every_s: float) -> str:
    h = hashlib.sha1()
    h.update(json.dumps({"v": 2, "video": data.signature.get("video"), "every": round(float(every_s), 4),
                         "n": int(len(data.rows)), "size": CLIP_SIZE, "min_h": CLIP_MIN_BOX_H}, sort_keys=True).encode())
    h.update(np.ascontiguousarray(data.rows[:, :8], np.float32).tobytes())
    return h.hexdigest()[:20]


def save_clip_crops(work: Path, data, every_s: float, rows: list[int], jpgs: list[bytes], crop_s: float = 0.0) -> None:
    """work/clip_crops.npz: rows (pass-1 row index), offsets + one uint8 blob of lossless PNG box crops, key,
    stats."""
    lens = np.array([len(b) for b in jpgs], np.int64)
    off = np.r_[0, np.cumsum(lens)]
    blob = np.frombuffer(b"".join(jpgs), np.uint8)
    tmp = Path(work) / "clip_crops.tmp.npz"
    np.savez(tmp, key=np.array(_clip_key(data, every_s)), rows=np.array(rows, np.int64), off=off, blob=blob,
             stats=np.array(json.dumps({"crops": len(rows), "every_s": every_s, "crop_s": round(crop_s, 2),
                                        "mb": round(blob.nbytes / 1e6, 1)})))
    os.replace(tmp, Path(work) / "clip_crops.npz")


def load_clip_crops(work: Path, data=None, rows=None) -> dict[int, np.ndarray]:
    """{pass-1 row: 224 x 224 uint8 BGR square crop (= clip_roles.crop before normalising)} from pass 1 (empty when there is no matching cache). With
    ``data`` the cache key is checked; ``rows`` limits the decoding to these rows. Feed clip_roles: normalise like
    clip_roles.crop does ((x[..., ::-1] / 255 - MEAN) / STD, CHW)."""
    path = Path(work) / "clip_crops.npz"
    if not path.exists():
        return {}
    try:
        with np.load(path) as z:
            if data is not None:
                st = json.loads(str(z["stats"]))
                if str(z["key"]) != _clip_key(data, float(st.get("every_s", CLIP_TAP_EVERY_S))):
                    return {}
            rr, off, blob = z["rows"], z["off"], z["blob"]
    except Exception:  # noqa: BLE001 - corrupt cache
        return {}
    want = None if rows is None else set(int(r) for r in rows)
    out = {}
    for j, r in enumerate(rr.tolist()):
        if want is not None and r not in want:
            continue
        img = cv2.imdecode(blob[off[j]:off[j + 1]], cv2.IMREAD_COLOR)
        if img is not None:
            out[r] = clip_pad(img)
    return out


def clip_embed_cached(work: Path, data, rows) -> dict[int, np.ndarray]:
    """CLIP embeddings (clip_roles model) of those ``rows`` that pass 1 cropped (work/clip_crops.npz): no video
    decoding. Rows without a stored crop are simply missing from the result (pick rows on the 2-s grid)."""
    from replay import clip_roles

    crops = load_clip_crops(work, data, rows)
    if not crops or not clip_roles.available():
        return {}
    rr = sorted(crops)
    x = [((crops[r][..., ::-1].astype(np.float32) / 255.0 - clip_roles.MEAN) / clip_roles.STD).transpose(2, 0, 1)
         for r in rr]
    return dict(zip(rr, clip_roles.embed(x)))
