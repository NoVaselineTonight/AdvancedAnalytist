"""General broadcast score-bar reader: finds the score overlay by itself and reads a score timeline.

No layout is assumed (KoriTV bottom bar, NCAA scorebug, ...):

1. **Find the overlay** - frames sampled over the whole video (``locate_every_s``) at a small working width.
   Overlay pixels are *static* (the per-pixel median absolute deviation over time is small while the camera pans
   under them) and *textured* (strong edges in the median frame). Their union, closed into blocks, gives text
   blocks: words / numbers of the overlay (team names, scores, clock, period, logos).
2. **Read the blocks** every ``every_s`` seconds with the PP-OCRv4 text recogniser (replay/jersey_numbers.Recognizer,
   ONNX; CPU by default so it never competes with the GPU detector).
3. **Classify the blocks** from their reads over time: clock (``mm:ss`` / ``ss.t``, mostly decreasing), numbers
   that only go up in steps of 1-3 (scores), constant numbers (seeds, period), text (names).
   The two best score blocks are the two sides of the bar (``side 0`` = left / top). The neighbouring text
   block of each side is its label (team name or abbreviation) when one exists.
4. **Smooth** each score: a new value is accepted only when read in ``confirm`` consecutive samples and it is an
   increase of 1-3 points (a larger jump is accepted after a gap with no readable bar); scores never go down
   unless a lower value is read stably for ``correction_s`` (a scorer's correction).

Output (``read_scorebar``): dict with ``sides`` (label + block box per side), ``timeline`` [(t, s0, s1)] at every
change, ``samples`` [(t, s0, s1, clock, period)] and ``stats``. CLI: ``python -m replay.scorebar <video> [out.json]``.
Team mapping (which side is which kit / scoresheet team) is left to the caller (labels / colours).
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")
import cv2  # noqa: E402

CLOCK_RE = re.compile(r"^(\d{1,2})[:.](\d{2})|^(\d{1,2})\.(\d)$")  # prefix: a shot clock may follow
PERIOD_RE = re.compile(r"^(1ST|2ND|3RD|4TH|[1-4]Q|Q[1-4]|OT\d?|[1-4]\.?|1E|2E|HALF|H[12]|P[1-4])$", re.I)


@dataclass
class ScorebarConfig:
    every_s: float = 1.0  # reading interval
    locate_every_s: float = 4.0  # sampling for finding the overlay
    locate_max: int = 80  # at most this many frames for finding it
    work_w: int = 640  # working width for finding it
    mad_static: float = 10.0  # gray levels: 80th percentile deviation from the median below this = static
    edge_thr: float = 60.0  # Sobel magnitude of the median frame: text edges
    min_block_h: int = 5  # px at work_w
    max_block_h: int = 60
    pad: float = 0.4  # crop padding around a block (x its height): a lone digit needs room to be read clearly
    confirm: int = 2  # samples a new score must be read in a row
    max_step: int = 3  # points per basket
    correction_s: float = 20.0
    device: str = "cpu"


@dataclass
class Block:
    x0: int
    y0: int
    x1: int
    y1: int  # full-resolution box
    reads: list = field(default_factory=list)  # (t, text, conf)

    @property
    def h(self) -> int:
        return self.y1 - self.y0


# ----------------------------------------------------------------------------------------------- frames
def _grab_frames(video: str | Path, every_s: float, max_n: int | None = None, t0: float = 0.0):
    """Yield (t, frame) every ``every_s`` seconds by sequential grabbing (decode only what is used)."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        dur = n_frames / fps if n_frames else None
        step = every_s
        if max_n and dur:
            step = max(every_s, dur / max_n)
        nxt, i, got = t0, 0, 0
        while True:
            if not cap.grab():
                break
            t = i / fps
            i += 1
            if t + 0.5 / fps < nxt:
                continue
            ok, f = cap.retrieve()
            if ok:
                yield t, f
                got += 1
                if max_n and got >= max_n:
                    break
            nxt += step
    finally:
        cap.release()


# ----------------------------------------------------------------------------------------------- locate
def locate_panels(frames: list[np.ndarray], cfg: ScorebarConfig) -> tuple[list[tuple[int, int, int, int]], np.ndarray]:
    """Overlay panels (full-resolution boxes): pixels with strong edges that stay the same in most frames
    (80th percentile of the deviation from the median is small - static walls of a fixed gym camera move a
    little with every pan, the overlay never)."""
    H, W = frames[0].shape[:2]
    s = cfg.work_w / W
    small = np.stack([cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (cfg.work_w, round(H * s)),
                                 interpolation=cv2.INTER_AREA) for f in frames]).astype(np.float32)
    med = np.median(small, axis=0)
    dev = np.percentile(np.abs(small - med), 80, axis=0)
    gx = cv2.Sobel(med, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(med, cv2.CV_32F, 0, 1, ksize=3)
    mask = ((np.hypot(gx, gy) > cfg.edge_thr) & (dev < cfg.mad_static)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5)))
    n, _, st, _ = cv2.connectedComponentsWithStats(mask, 8)
    out = []
    for j in range(1, n):
        x, y, w, h, a = st[j]
        if w < 12 or h < cfg.min_block_h or h > 0.3 * mask.shape[0] or a < 40:
            continue
        p = 2
        out.append((int(max(0, (x - p) / s)), int(max(0, (y - p) / s)), int(min(W, (x + w + p) / s)),
                    int(min(H, (y + h + p) / s))))
    return out, mask


def _words(gray: np.ndarray, min_h: int) -> list[tuple[int, int, int, int]]:
    """Text words in a panel crop: characters = thin bright-on-dark (top-hat) or dark-on-bright (black-hat)
    strokes, as connected components, grouped along a line."""
    ph = gray.shape[0]
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, ph // 3) | 1, max(3, ph // 3) | 1))
    chars = []
    for pol, op in enumerate((cv2.MORPH_TOPHAT, cv2.MORPH_BLACKHAT)):
        hat = cv2.morphologyEx(gray, op, k)
        bw = (hat > max(30, 0.4 * float(hat.max()))).astype(np.uint8)
        n, _, st, _ = cv2.connectedComponentsWithStats(bw, 8)
        for x, y, w, h, a in st[1:]:
            if min_h <= h <= 0.95 * ph and w <= 1.3 * h and 0.15 * w * h <= a <= 0.85 * w * h:  # not a solid box
                chars.append((int(x), int(y), int(x + w), int(y + h), pol))
    chars = sorted(set(chars))
    # drop boxes contained in a bigger one (MSER nests)
    keep = []
    for c in sorted(chars, key=lambda c: -(c[2] - c[0]) * (c[3] - c[1])):
        if not any(k[4] == c[4] and k[0] <= c[0] and k[1] <= c[1] and k[2] >= c[2] and k[3] >= c[3] for k in keep):
            keep.append(c)
    keep.sort()
    words: list[list[int]] = []
    for c in keep:
        h = c[3] - c[1]
        for w_ in words:
            wh = w_[3] - w_[1]
            gap = 0.7 if abs(h - wh) < 0.12 * max(h, wh) else 0.35  # same height: also across a colon
            if (w_[4] == c[4] and c[0] - w_[2] < gap * max(h, wh) and abs(h - wh) < 0.35 * max(h, wh)
                    and min(c[3], w_[3]) - max(c[1], w_[1]) > 0.5 * min(h, wh) and c[0] >= w_[0]):
                w_[:] = [min(w_[0], c[0]), min(w_[1], c[1]), max(w_[2], c[2]), max(w_[3], c[3]), c[4]]
                break
        else:
            words.append(list(c))
    return [tuple(w[:4]) for w in words]


def locate_blocks(frames: list[np.ndarray], cfg: ScorebarConfig, panels: list | None = None,
                  min_h: int | None = None) -> tuple[list[Block], np.ndarray | None]:
    """Text blocks (words / numbers) of the overlay that recur in many sampled frames. ``panels`` given (e.g. the
    whole bar strip): skip finding them."""
    mask = None
    if panels is None:
        panels, mask = locate_panels(frames, cfg)
    H, W = frames[0].shape[:2]
    if min_h is None:
        min_h = max(6, int(round(cfg.min_block_h * W / cfg.work_w)))
    pick = frames[:: max(1, len(frames) // 16)]
    out: list[Block] = []
    for (x0, y0, x1, y1) in panels:
        found: list[list] = []  # [first box, member boxes]
        for f in pick:
            g = cv2.cvtColor(f[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
            for wx0, wy0, wx1, wy1 in _words(g, min_h):
                b = [wx0 + x0, wy0 + y0, wx1 + x0, wy1 + y0]
                for fb in found:  # same row and overlapping: the same field (a number may grow 9 -> 10 -> 100)
                    o = fb[0]
                    h = max(o[3] - o[1], b[3] - b[1])
                    ix = min(o[2], b[2]) - max(o[0], b[0])
                    if abs((o[1] + o[3]) - (b[1] + b[3])) / 2 < 0.4 * h and ix > 0.3 * min(o[2] - o[0], b[2] - b[0]):
                        fb[1].append(b)
                        break
                else:
                    found.append([b, [b]])
        for _, mem in found:
            if len(mem) >= 0.4 * len(pick):
                m = np.array(mem, float)
                # a number merged with a glued logo / neighbour in some frames (NCAA "50" + the "A" logo) is much
                # wider than the field usually is: leave those out, else the field grows over the logo and is then
                # dropped as a box around text
                # ("usually" = the most common width; a word's letter fragments - several per frame, other polarity -
                # never make the word itself the outlier: "2ND" must not shrink to its "N")
                w = m[:, 2] - m[:, 0]
                vals, cnt = np.unique(np.round(w / 4.0), return_counts=True)
                wide = w > 4.0 * vals[cnt.argmax()] + 2.0 + np.percentile(m[:, 3] - m[:, 1], 25)
                if (wide.any() and (~wide).sum() >= 0.4 * len(pick)
                        and np.median(w[~wide]) >= 0.25 * np.median(w[wide])):
                    m = m[~wide]
                b = [int(np.percentile(m[:, 0], 10)), int(np.median(m[:, 1])), int(np.percentile(m[:, 2], 90)),
                     int(np.median(m[:, 3]))]  # widest usual extent, not one merged outlier
                h = b[3] - b[1]
                p = int(cfg.pad * h)
                out.append(Block(max(0, b[0] - p), max(0, b[1] - p), min(W, b[2] + p), min(H, b[3] + p)))
    # a block that contains another block's centre is a box around text (KoriTV: the two score boxes): drop it
    keep = []
    for b in out:
        inside = [o for o in out if o is not b and b.x0 <= (o.x0 + o.x1) / 2 <= b.x1
                  and b.y0 <= (o.y0 + o.y1) / 2 <= b.y1 and (o.x1 - o.x0) * o.h < (b.x1 - b.x0) * b.h]
        if not inside:
            keep.append(b)
            continue
        # ... unless the other block only sticks into one end of it (a field whose extent grew over its neighbour in
        # some frames, NCAA "32" | "2ND"): cut it back to where the neighbour begins
        cx = (b.x0 + b.x1) / 2
        if all(o.x1 > b.x1 or o.x0 < b.x0 for o in inside):
            x0 = max([b.x0] + [int(o.x1 - 0.2 * o.h) for o in inside if (o.x0 + o.x1) / 2 < cx])
            x1 = min([b.x1] + [int(o.x0 + 0.2 * o.h) for o in inside if (o.x0 + o.x1) / 2 >= cx])
            if x1 - x0 >= 0.5 * (b.x1 - b.x0):
                b.x0, b.x1 = x0, x1
                keep.append(b)
    return keep, mask


# ----------------------------------------------------------------------------------------------- reading
_FOLD: dict = {}


def _fold(rec, probs: np.ndarray) -> np.ndarray:
    """Add the probability of full-width / compatibility forms ("２", "："; the model's charset has both) to
    their ASCII class: otherwise a clear "2" can come out at 0.5 confidence, split between the two."""
    key = id(rec)
    if key not in _FOLD:
        import unicodedata

        idx = {c: i for i, c in enumerate(rec.charset)}
        pairs = []
        for i, c in enumerate(rec.charset):
            n = unicodedata.normalize("NFKC", c) if c else c
            if n != c and len(n) == 1 and n in idx and n.isascii():
                pairs.append((i, idx[n]))
        _FOLD[key] = np.array(pairs, int).reshape(-1, 2)
    pr = _FOLD[key]
    if not len(pr):
        return probs
    p = probs.copy()
    np.add.at(p, (slice(None), pr[:, 1]), probs[:, pr[:, 0]])
    p[:, pr[:, 0]] = 0.0
    return p


def _greedy(rec, probs: np.ndarray) -> tuple[str, float]:
    probs = _fold(rec, probs)
    a = probs.argmax(axis=1)
    pm = probs.max(axis=1)
    chars, confs = [], []
    for j, c in enumerate(a):
        if c and (j == 0 or c != a[j - 1]):
            chars.append(rec.charset[c])
            confs.append(float(pm[j]))
    return "".join(chars).strip(), (float(np.mean(confs)) if confs else 0.0)


def read_blocks(rec, frame: np.ndarray, blocks: list[Block], t: float) -> None:
    crops = [frame[b.y0:b.y1, b.x0:b.x1] for b in blocks]
    ok = [i for i, c in enumerate(crops) if c.size and c.shape[0] >= 6 and c.shape[1] >= 4]
    if not ok:
        return
    probs = rec.probs([crops[i] for i in ok], max_w=320)
    for i, p in zip(ok, probs):
        txt, conf = _greedy(rec, p)
        blocks[i].reads.append((t, txt.replace(" ", ""), conf))


# ----------------------------------------------------------------------------------------------- classify
def _digits(s: str) -> bool:
    """ASCII digits only: the OCR charset has '²', '①' ... for which str.isdigit() is True but int() raises."""
    return s.isascii() and s.isdigit()


def _num(s: str) -> int | None:
    """A score: 1-3 ASCII digits. A team logo glued to the number is split off: one trailing capital ("86A") or
    non-ASCII sign ("74风", not a digit-like "1²"), or
    1-2 leading non-digit characters of a logo read as letters / symbols before the trailing digit run ("A48",
    "@48") - never a period / clock / label text ("Q4", "OT1", "FOULS:1", "1:05")."""
    m = re.match(r"^([0-9]{1,3})([A-Z]|[^\x00-\x7f])?$", s)  # the logo may also come out as a CJK sign ("74风")
    if m and not (m.group(2) and not m.group(2).isascii() and unicodedata.category(m.group(2)) in ("No", "Nd")):
        return int(m.group(1))
    m = re.match(r"^([^0-9:.\s]{1,2})([0-9]{1,3})$", s)
    if m and not PERIOD_RE.match(s) and not re.match(r"^(OT|TO|[QPH#]|NO)", s, re.I):
        return int(m.group(2))
    return None


def _clock_s(s: str) -> float | None:
    m = CLOCK_RE.match(s)
    if not m:
        return None
    if m.group(1) is not None:
        return int(m.group(1)) * 60 + int(m.group(2))
    return int(m.group(3)) + int(m.group(4)) / 10


def classify(blocks: list[Block]) -> dict:  # noqa: C901
    """Role per block: 'score' candidates ranked, 'clock', 'period', 'text'."""
    info = []
    for i, b in enumerate(blocks):
        reads = [(t, s, c) for t, s, c in b.reads if s and c >= 0.6]
        n_all = max(1, len(b.reads))
        nums = [(t, _num(s)) for t, s, c in reads if _num(s) is not None]
        clocks = [(t, _clock_s(s)) for t, s, c in reads if _clock_s(s) is not None]
        periods = [s for t, s, c in reads if PERIOD_RE.match(s) and _num(s) is None]
        d = {"i": i, "h": b.h, "n": len(reads), "num_frac": len(nums) / n_all, "clock_frac": len(clocks) / n_all,
             "period_frac": len(periods) / n_all, "text": max(set(s for _, s, _ in reads), default="",
                                                             key=lambda x: sum(1 for _, s, _ in reads if s == x))}
        if nums:
            v = np.array([x for _, x in nums], float)
            # stable values (same read twice in a row) and their changes
            stab = [v[k] for k in range(1, len(v)) if v[k] == v[k - 1]]
            ch = np.diff([x for k, x in enumerate(stab) if k == 0 or x != stab[k - 1]]) if stab else np.zeros(0)
            d.update(ups=int(((ch >= 1) & (ch <= 3)).sum()), downs=int((ch < 0).sum()), big=int((ch > 3).sum()),
                     n_values=len(set(v.tolist())), vmax=float(v.max()), steps23=int(((ch >= 2) & (ch <= 3)).sum()))
            # change times of the stable value: a clock's minutes change every ~60 s, regularly
            tt = [t for (t, x), (_, y) in zip(nums[1:], nums[:-1]) if x == y]
            st_v = [x for (t, x), (_, y) in zip(nums[1:], nums[:-1]) if x == y]
            ct = [tt[k] for k in range(1, len(st_v)) if st_v[k] != st_v[k - 1]]
            gaps = np.diff(ct) if len(ct) >= 4 else np.zeros(0)
            d["ticking"] = bool(len(gaps) >= 3 and np.mean(np.abs(gaps - 60.0) < 8.0) >= 0.6)
            d["chg_rate"] = float(np.mean(v[1:] != v[:-1])) if len(v) > 1 else 0.0  # per consecutive read
            d["span"] = float(nums[-1][0] - nums[0][0])
        info.append(d)
    # fast numbers (clock seconds, shot clock: a new value in many samples) and the number right before a fast one
    # on the same row (clock minutes "05" of "05 07") are clock parts, never scores
    fast = {d["i"] for d in info if d.get("n_values", 0) > 6 and d.get("chg_rate", 0) > 0.3}
    clock_parts = set(fast)
    pairs = []  # (minutes block, seconds block) of a clock drawn as two numbers ("05" ":" "07")
    for d in info:
        if "n_values" not in d or d["i"] in fast:
            continue
        b = blocks[d["i"]]
        for j in fast:
            o = blocks[j]
            hh = max(b.h, o.h)
            if abs((o.y0 + o.y1) - (b.y0 + b.y1)) / 2 < 0.5 * hh and -0.8 * hh <= o.x0 - b.x1 < 1.5 * hh:
                clock_parts |= {d["i"], j}
                pairs.append((d["i"], j))
    score = [d for d in info if d["num_frac"] >= 0.5 and d.get("vmax", 999) <= 250 and d["i"] not in clock_parts
             and not d.get("ticking") and d.get("downs", 0) <= max(1, d.get("ups", 0) // 10) and d["clock_frac"] < 0.3]
    # scores: numbers that went up in basket steps (2 / 3 points: no clock, period or seed does that) first, then
    # any rise, then the larger ones (a seed "1" / period "3" is small)
    score.sort(key=lambda d: (min(d.get("steps23", 0), 1), min(d.get("ups", 0), 1), d.get("vmax", 0), d["h"]),
               reverse=True)
    clock = max((d for d in info if d["clock_frac"] >= 0.4), key=lambda d: d["clock_frac"], default=None)
    # ordinal periods ("2ND", "Q3", "OT") beat bare numbers that also fit the period pattern
    period = max((d for d in info if d["period_frac"] >= 0.4),
                 key=lambda d: (bool(re.search(r"\d(ST|ND|RD|TH)|Q\d|\dQ|OT", d["text"], re.I)), d["period_frac"]),
                 default=None)
    pair = None
    if clock is None and pairs:  # the pair whose seconds stay within 0-59
        pair = max(pairs, key=lambda pr: sum(1 for _, x, c in blocks[pr[1]].reads if _digits(x) and int(x) < 60))
    cb = blocks[clock["i"]] if clock else blocks[pair[1]] if pair else None

    def near_clock(d):  # a bare-number period sits next to the clock, not next to a team name (seed "1")
        o = blocks[d["i"]]
        return (cb is not None and abs((o.y0 + o.y1) - (cb.y0 + cb.y1)) / 2 < 0.6 * cb.h
                and max(o.x0 - cb.x1, cb.x0 - o.x1) < 4 * cb.h)

    if (period is not None and cb is not None and not near_clock(period)
            and not re.search(r"\d(ST|ND|RD|TH)|Q\d|\dQ|OT", period["text"], re.I)):
        period = None
    if period is None and cb is not None:  # KoriTV: a constant small number next to the clock ("05:07 | 3")
        top = {d["i"] for d in (score_pair(blocks, score) or ())}
        near = [d for d in info if d["i"] not in clock_parts and d["i"] not in top and d.get("vmax", 99) <= 9
                and d["num_frac"] >= 0.6 and d.get("chg_rate", 1) < 0.05
                and near_clock(d)]
        cx = (cb.x0 + cb.x1) / 2
        period = min(near, key=lambda d: abs((blocks[d["i"]].x0 + blocks[d["i"]].x1) / 2 - cx), default=None)
    return {"info": info, "score": score, "clock": clock, "period": period, "clock_pair": pair}


GENERIC = re.compile(r"FOUL|POSS|BONUS|TIME|TOL|BONUS|PERIOD|QTR|HALF|FINAL|FOUR", re.I)


def _side_label(blocks: list[Block], info: list[dict], k: int, used: set) -> str:
    """Team name / abbreviation of a score: the nearest unused text block on the same row, to the left first
    (NCAA "MICHIGAN 53 | ARIZONA 34"), else to the right (KoriTV "KLA 15 | 65 LEP")."""
    b = blocks[k]
    cands = []
    for d in info:
        j = d["i"]
        if (j in used or j == k or d["num_frac"] > 0.3 or d["clock_frac"] > 0.3 or d["period_frac"] > 0.3
                or not re.search(r"[A-Za-z]{2,}", d["text"]) or GENERIC.search(d["text"])):
            continue
        o = blocks[j]
        if abs((o.y0 + o.y1) / 2 - (b.y0 + b.y1) / 2) > 0.6 * max(b.h, o.h):
            continue
        left = o.x1 <= b.x0 + 0.3 * b.h
        cands.append((0 if left else 1, abs((o.x0 + o.x1) / 2 - (b.x0 + b.x1) / 2), j, d["text"]))
    if not cands:
        return ""
    best = min(cands)
    used.add(best[2])
    return best[3]


# ----------------------------------------------------------------------------------------------- smoothing
def smooth_scores(series: list[tuple[float, int | None]], cfg: ScorebarConfig) -> list[tuple[float, int | None]]:
    """(t, raw value or None) -> (t, smoothed value). See module docstring."""
    out, cur = [], None
    cand, cand_n, cand_t0 = None, 0, None
    last_seen = None
    for t, v in series:
        if v is not None:
            if cur is None:
                if v == cand:
                    cand_n += 1
                else:
                    cand, cand_n, cand_t0 = v, 1, t
                if cand_n >= cfg.confirm:
                    cur = cand
            elif v != cur:
                if v == cand:
                    cand_n += 1
                else:
                    cand, cand_n, cand_t0 = v, 1, t
                gap = t - (last_seen if last_seen is not None else t)
                step = cand - cur
                ok_up = 1 <= step <= cfg.max_step or (step > cfg.max_step and gap > 10.0)
                ok_down = step < 0 and t - cand_t0 >= cfg.correction_s
                if cand_n >= cfg.confirm and (ok_up or ok_down):
                    cur = cand
                    for j in range(len(out) - 1, -1, -1):  # back-date to the first read of the new value
                        if out[j][0] < cand_t0:
                            break
                        out[j] = (out[j][0], cur)
            else:
                cand, cand_n = None, 0
            last_seen = t
        out.append((t, cur))
    return out


# ----------------------------------------------------------------------------------------------- main
def read_scorebar(video: str | Path, cfg: ScorebarConfig | None = None, recognizer=None, t_max: float | None = None
                  ) -> dict:
    """Decode the video once (a frame every ``every_s``), then locate + read (see read_frames)."""
    cfg = cfg or ScorebarConfig()
    frames = ((t, f) for t, f in _grab_frames(video, cfg.every_s) if t_max is None or t <= t_max)
    return read_frames(frames, cfg, recognizer)


class ScorebarReader:
    """Incremental reader: ``add(t, frame)`` for frames at >= ``every_s`` spacing, ``result()`` at the end. The first
    ``locate_n`` frames are kept JPEG-compressed until the overlay is located; after that only its blocks are read
    (a few small crops per frame, CPU)."""

    locate_n = 30  # frames (1 min at 2 s) per attempt to find the bar
    max_tries = 3  # attempts (minutes) before giving up when text is seen but no two number fields; no text at all
    # in the first minute = no overlay: give up at once (hs_hazen: decode stops after ~1 min of video)
    retry_gaps = (300.0, 600.0)  # no text at all: sleep this many video seconds, then try to locate again (twice)
    static_every = 10  # text that never changes (team names, logo) is read only every this many samples
    wide_n = 60  # frames (2 min at 2 s): the bar found in minute 1 is located again over this many frames (a camera
    # that stood still for a minute makes the whole gym look like overlay: c4000 found the wall clock, not the bar)

    def __init__(self, cfg: ScorebarConfig | None = None, recognizer=None):
        self.cfg = cfg or ScorebarConfig()
        self.rec = recognizer
        self.buf: list[tuple[float, bytes]] = []
        self.blocks: list[Block] | None = None
        self.ts: list[float] = []
        self.strips: list[bytes | None] = []  # per sample: JPEG of the overlay rows (strip_box)
        self.strip_box: tuple[int, int, int, int] | None = None
        self.t_loc = self.t_ocr = 0.0
        self.no_bar = False
        self.tries = 0
        self.blank = 0  # failed attempts without any text
        self.wake = -1.0  # video time before which frames are skipped (waiting for the next locate retry)
        self.static: set[int] = set()  # indices of blocks with constant text
        self.held: list[tuple[float, bytes]] | None = None  # minute-1 frames + the next ones, for the wider locate
        self.held_n0 = 0

    def _recognizer(self):
        if self.rec is None:
            from replay.jersey_numbers import Recognizer

            self.rec = Recognizer(device=self.cfg.device)
        return self.rec

    def add(self, t: float, frame: np.ndarray) -> None:
        if self.no_bar or t < self.wake:
            return
        if self.blocks is None:
            self.buf.append((t, cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes()))
            if len(self.buf) >= self.locate_n:
                self._locate()
            return
        if self.held is not None:
            self.held.append((t, cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes()))
        self._read(t, frame)
        if self.held is not None and len(self.held) >= self.wide_n:
            self._locate_wide()

    def _read(self, t: float, frame: np.ndarray) -> None:
        t1 = time.perf_counter()
        if self.blocks:
            if self.static and len(self.ts) % self.static_every:
                live = [b for k, b in enumerate(self.blocks) if k not in self.static]
            else:
                live = self.blocks
            read_blocks(self._recognizer(), frame, live, t)
            if self.strip_box is None:  # the band of rows holding the most blocks (the bar, not a corner logo)
                cy = lambda b: (b.y0 + b.y1) / 2  # noqa: E731
                bs = max(([o for o in self.blocks if abs(cy(o) - cy(b)) < 2.5 * b.h] for b in self.blocks), key=len)
                self.strip_box = (min(b.x0 for b in bs), min(b.y0 for b in bs), max(b.x1 for b in bs),
                                  max(b.y1 for b in bs))
            x0, y0, x1, y1 = self.strip_box
            self.strips.append(cv2.imencode(".jpg", frame[y0:y1, x0:x1], [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes())
        else:
            self.strips.append(None)
        self.t_ocr += time.perf_counter() - t1
        self.ts.append(t)

    def _locate(self) -> None:
        dec = lambda b: cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)  # noqa: E731
        t1 = time.perf_counter()
        n_loc = min(self.cfg.locate_max, max(5, int(len(self.buf) * self.cfg.every_s / self.cfg.locate_every_s)))
        step = max(1, int(round(len(self.buf) / n_loc)))
        self.blocks = locate_blocks([dec(b) for _, b in self.buf[::step]], self.cfg)[0] if len(self.buf) >= 5 else []
        self.t_loc += time.perf_counter() - t1
        buf, self.buf = self.buf, []
        last_t = buf[-1][0] if buf else 0.0
        for t, b in buf:
            self._read(t, dec(b))
        # a score bar has at least two number fields read in most samples; else try the next minute, then give up
        if self._n_number_fields(self.blocks or []) < 2:
            if self.blocks:
                self.tries += 1
                if self.tries >= self.max_tries:
                    self.no_bar = True
            else:
                # no text at all in a whole minute (intro / black / title card, or no overlay): retry later in the
                # video (twice) instead of giving up for the whole game
                self.blank += 1
                self.tries += 1
                if self.blank > len(self.retry_gaps):
                    self.no_bar = True
                else:
                    self.wake = last_t + self.retry_gaps[self.blank - 1]
            self.blocks, self.ts, self.strips, self.strip_box = None, [], [], None
            return
        self._mark_static()
        if len(buf) >= self.locate_n:  # keep the frames for the 2-minute locate
            self.held, self.held_n0 = buf, len(buf)

    @staticmethod
    def _n_number_fields(blocks: list[Block]) -> int:
        return sum(1 for b in blocks if sum(1 for _, x, c in b.reads if c >= 0.6 and _num(x) is not None)
                   >= 0.5 * max(1, len(b.reads)))

    def _mark_static(self) -> None:
        self.static = set()
        for k, b in enumerate(self.blocks or []):
            txt = [x for _, x, c in b.reads]
            # digits => may be a frozen clock / period label / score: must keep being read every sample
            if txt and not any(ch.isdigit() for x in txt for ch in x) and txt.count(txt[0]) >= 0.9 * len(txt):
                self.static.add(k)

    def _locate_wide(self) -> None:
        """Locate the bar once more over the first ``wide_n`` frames (as before the 1-minute early give-up) and
        re-read them with those blocks; the minute-1 blocks are kept when the wider locate finds < 2 number
        fields."""
        held, self.held = self.held, None
        if not held or len(held) <= self.held_n0:
            return
        dec = lambda b: cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)  # noqa: E731
        t1 = time.perf_counter()
        n_loc = min(self.cfg.locate_max, max(5, int(len(held) * self.cfg.every_s / self.cfg.locate_every_s)))
        step = max(1, int(round(len(held) / n_loc)))
        blocks = locate_blocks([dec(b) for _, b in held[::step]], self.cfg)[0]
        self.t_loc += time.perf_counter() - t1
        if not blocks:
            return
        for b in blocks:
            b.reads = []
        old = (self.blocks, self.ts, self.strips, self.strip_box, self.static)
        self.blocks, self.ts, self.strips, self.strip_box, self.static = blocks, [], [], None, set()
        for t, b in held:
            self._read(t, dec(b))
        if self._n_number_fields(blocks) < 2:
            self.blocks, self.ts, self.strips, self.strip_box, self.static = old
            return
        self._mark_static()

    def _relocate_on_strips(self) -> None:
        """Long videos: the fields were found in the first minutes (e.g. a one-digit 0-0 score at the start of a
        game). Find them again on bar strips spread over the whole video and read all strips with them."""
        idx = [k for k, jp in enumerate(self.strips) if jp is not None]
        if len(idx) < 2 * self.wide_n or self.strip_box is None:
            return
        dec = lambda b: cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)  # noqa: E731
        sel = idx[:: max(1, len(idx) // 48)]
        imgs = [dec(self.strips[k]) for k in sel]
        h0 = min(b.h for b in self.blocks) if self.blocks else 8
        sx0, sy0, _, _ = self.strip_box
        sh, sw = imgs[0].shape[:2]
        bl, _ = locate_blocks(imgs, self.cfg, panels=[(0, 0, sw, sh)], min_h=max(6, int(0.4 * h0)))
        if len(bl) < 2:
            return
        t1 = time.perf_counter()
        rec = self._recognizer()
        for b in bl:  # back to frame coordinates
            b.x0, b.x1, b.y0, b.y1 = b.x0 + sx0, b.x1 + sx0, b.y0 + sy0, b.y1 + sy0
            b.reads = []

        def iou(p, q):
            ix = min(p.x1, q.x1) - max(p.x0, q.x0)
            iy = min(p.y1, q.y1) - max(p.y0, q.y0)
            if ix <= 0 or iy <= 0:
                return 0.0
            return ix * iy / ((p.x1 - p.x0) * p.h + (q.x1 - q.x0) * q.h - ix * iy)

        todo = []
        for b in bl:  # a field found again at (almost) the same place keeps its reads: re-read only the changed ones
            old = max(self.blocks or [], key=lambda o: iou(o, b), default=None)
            if old is not None and iou(old, b) >= 0.8:
                b.reads = list(old.reads)
            else:
                todo.append(b)
        for j in range(0, len(idx), 64):  # batched: many small crops per model call
            crops, who = [], []
            for k in idx[j:j + 64]:
                img = dec(self.strips[k])
                for b in todo:
                    c = img[b.y0 - sy0:b.y1 - sy0, b.x0 - sx0:b.x1 - sx0]
                    if c.size and c.shape[0] >= 6 and c.shape[1] >= 4:
                        crops.append(c)
                        who.append((b, self.ts[k]))
            for (b, t), pp in zip(who, rec.probs(crops, max_w=320) if crops else []):
                txt, conf = _greedy(rec, pp)
                b.reads.append((t, txt.replace(" ", ""), conf))
        for b in todo:
            b.reads.sort(key=lambda r: r[0])
        self.blocks = bl
        self.t_ocr += time.perf_counter() - t1

    def result(self) -> dict:
        if self.no_bar:
            return {"error": "no score bar", "no_bar": True, "sides": [], "timeline": [], "samples": [], "clock": [],
                    "summary": {"sure": False}, "stats": {"tries": self.tries, "locate_s": round(self.t_loc, 1),
                                                          "ocr_s": round(self.t_ocr, 1)}}
        if self.blocks is None:
            if self.buf:
                self._locate()
            if self.blocks is None:
                return {"error": "no score bar", "sides": [], "timeline": [], "samples": [], "clock": [],
                        "summary": {"sure": False}, "stats": {"tries": self.tries}}
        if self.held is not None:  # a video shorter than wide_n frames: the wider locate over all of it
            self._locate_wide()
        self.static = set()  # indices refer to the blocks replaced below
        self._relocate_on_strips()
        return _result(self.blocks, self.ts, self.cfg, self.t_loc, self.t_ocr, self.rec, self)


def read_frames(frames, cfg: ScorebarConfig | None = None, recognizer=None) -> dict:
    """Score timeline from (t, BGR frame) pairs at >= ``every_s`` spacing (e.g. tapped from a decode)."""
    rd = ScorebarReader(cfg, recognizer)
    t0 = time.perf_counter()
    for t, f in frames:
        rd.add(t, f)
        if rd.no_bar:  # stop decoding: nothing to read in this video
            break
    res = rd.result()
    res.setdefault("stats", {})["total_s"] = round(time.perf_counter() - t0, 1)
    return res


def _similar_h(d: dict, e: dict) -> bool:
    return min(d["h"], e["h"]) >= 0.7 * max(d["h"], e["h"])


def _plausible(d: dict) -> bool:
    """Score-like field: not clock-like (a new value in many samples / every ~60 s) and rising at a rate a team can
    score (a few baskets a minute at most)."""
    if d.get("ticking") or d.get("chg_rate", 0.0) > 0.3:
        return False
    return d.get("ups", 0) <= 4.0 * max(1.0, d.get("span", 0.0) / 60.0)


def score_pair(blocks: list[Block], sc: list[dict]) -> tuple[dict, dict] | None:
    """The two score fields among the ranked candidates ``sc`` (classify()["score"]). The two scores are drawn
    alike: the best PAIR of similar height (>= 0.7; a seed "1" next to a big "86" is not a partner), ranked by: side
    by side or stacked, both plausible, how many of the two rose in 2/3-point steps, rose at all, the smaller
    maximum (a 0-9 seed / period loses), reads. No such pair: the old pairing (the two best candidates)."""
    def aligned(d, e):
        a, b = blocks[d["i"]], blocks[e["i"]]
        h = max(a.h, b.h)
        row = abs((a.y0 + a.y1) - (b.y0 + b.y1)) / 2 < 0.6 * h and min(abs(b.x0 - a.x1), abs(a.x0 - b.x1)) < 12 * h
        col = min(a.x1, b.x1) - max(a.x0, b.x0) > 0 and min(abs(b.y0 - a.y1), abs(a.y0 - b.y1)) < 3 * h
        return row or col

    def seed_like(d):  # a constant one-digit field (seed "1", period, fouls): never a score with another such field
        return d.get("n_values", 0) <= 1 and d.get("vmax", 0) <= 9

    best, key = None, None
    for i, d in enumerate(sc):
        for e in sc[i + 1:]:
            if not _similar_h(d, e) or (seed_like(d) and seed_like(e)):
                continue
            k = (aligned(d, e), _plausible(d) and _plausible(e),
                 min(d.get("steps23", 0), 1) + min(e.get("steps23", 0), 1),
                 min(d.get("ups", 0), 1) + min(e.get("ups", 0), 1),
                 min(d.get("vmax", 0), e.get("vmax", 0)), min(d["n"], e["n"]))
            if key is None or k > key:
                best, key = (d, e), k
    if best is None and len(sc) >= 2 and not (seed_like(sc[0]) and seed_like(sc[1])):
        best = (sc[0], sc[1])
    return best


def _result(blocks: list[Block], ts: list[float], cfg: ScorebarConfig, t_loc: float, t_ocr: float, rec,
            reader=None) -> dict:
    if not blocks or len(ts) < 5:
        return {"error": "no score bar", "sides": [], "timeline": [], "samples": [], "summary": {"sure": False},
                "stats": {"samples": len(ts), "blocks": len(blocks or [])}}
    cls = classify(blocks)
    res = {"sides": [], "timeline": [], "samples": [], "blocks": len(blocks)}
    pair = score_pair(blocks, cls["score"])
    if pair is not None:
        a, b = pair[0]["i"], pair[1]["i"]
        A, B = blocks[a], blocks[b]
        if (A.x0, A.y0) > (B.x0, B.y0) if abs(A.y0 - B.y0) < A.h else A.y0 > B.y0:
            a, b = b, a
        used = {a, b}
        res["sides"] = [{"label": _side_label(blocks, cls["info"], k, used),
                         "box": [blocks[k].x0, blocks[k].y0, blocks[k].x1, blocks[k].y1]} for k in (a, b)]

        def series(k):
            m = {t: _num(s) if c >= 0.6 else None for t, s, c in blocks[k].reads}
            return [(t, m.get(t)) for t in ts]

        s0, s1 = smooth_scores(series(a), cfg), smooth_scores(series(b), cfg)
        res["clock"] = game_clock_series(blocks, cls, ts, reader)
        clk = {t: (clock_text(c) if c is not None else "") for t, _p, c, _r in res["clock"]}
        per = {t: (p or "") for t, p, _c, _r in res["clock"]}
        prev = None
        for (t, x), (_, y) in zip(s0, s1):
            res["samples"].append((round(t, 2), x, y, clk.get(t, ""), per.get(t, "")))
            if (x, y) != prev and x is not None and y is not None:
                res["timeline"].append((round(t, 2), x, y))
                prev = (x, y)
    res["stats"] = {"locate_s": round(t_loc, 1), "ocr_s": round(t_ocr, 1), "total_s": round(t_loc + t_ocr, 1),
                    "samples": len(ts), "blocks": len(blocks), "device": getattr(rec, "device", "")}
    res["roles"] = {"score": [d["i"] for d in pair] if pair else [d["i"] for d in cls["score"][:2]], "clock_pair": cls.get("clock_pair"),
                    "clock": cls["clock"]["i"] if cls["clock"] else None,
                    "period": cls["period"]["i"] if cls["period"] else None}
    res["summary"] = summary(res, cfg)
    return res


# ----------------------------------------------------------------------------------------------- game clock
def clock_text(sec: float) -> str:
    if sec < 60 and sec != int(sec):
        return f"{sec:.1f}"
    return f"{int(sec) // 60:02d}:{int(sec) % 60:02d}"


def _raw_clock(blocks: list[Block], cls: dict, ts: list[float]) -> dict:
    """t -> raw clock seconds (or None) from the clock block (prefix "mm:ss" / "ss.t"; a shot clock glued behind
    it is dropped) or from a minutes + seconds block pair."""
    out: dict = {}
    if cls.get("clock"):
        for t, x, c in blocks[cls["clock"]["i"]].reads:
            out[t] = _clock_s(x) if c >= 0.6 else None
    elif cls.get("clock_pair"):
        mm = {t: x for t, x, c in blocks[cls["clock_pair"][0]].reads if c >= 0.6}
        for t, x, c in blocks[cls["clock_pair"][1]].reads:
            m = mm.get(t, "")
            ok = c >= 0.6 and _digits(x) and len(x) == 2 and int(x) < 60 and _digits(m) and len(m) <= 2
            out[t] = int(m) * 60 + int(x) if ok else None
    return out


def _period_value(x: str) -> int | None:
    m = re.match(r"^(?:Q|P)?([1-9])(?:ST|ND|RD|TH|Q|E|\.)?$", x.strip(), re.I)
    if m:
        return int(m.group(1))
    m = re.match(r"^OT(\d?)$", x.strip(), re.I)
    return 4 + int(m.group(1) or 1) if m else None


def smooth_periods(raw: list[tuple[float, int | None]], confirm: int = 3) -> list[int | None]:
    """Period per sample: a new value needs ``confirm`` reads in a row; it only goes up."""
    out, cur, cand, n = [], None, None, 0
    for _t, v in raw:
        if v is not None and v != cur:
            cand, n = (v, n + 1) if v == cand else (v, 1)
            if n >= confirm and (cur is None or v > cur):
                cur, cand, n = v, None, 0
        elif v == cur:
            cand, n = None, 0
        out.append(cur)
    first = next((x for x in out if x is not None), None)  # before the first confirmation: that period
    return [first if x is None and k < out.index(first) else x for k, x in enumerate(out)] if first else out


def smooth_clock(raw: list[tuple[float, float | None]], periods: list[int | None], confirm: int = 2,
                 tol: float = 1.6) -> list[tuple[float | None, bool | None]]:
    """(clock seconds, running) per sample. The clock either stands still or moves by the elapsed time in its
    direction (counting down, or up on some local bars - found from the reads); anything else is a misread unless
    ``confirm`` consecutive reads agree with each other (a jump: new period, clock set by the officials)."""
    ok = [(t, v) for t, v in raw if v is not None]
    steps = [(b[1] - a[1]) / max(1e-6, b[0] - a[0]) for a, b in zip(ok, ok[1:]) if 0 < b[0] - a[0] <= 6]
    down = sum(1 for x in steps if -1.6 < x < -0.4)
    up = sum(1 for x in steps if 0.4 < x < 1.6)
    sign = -1.0 if down >= up else 1.0
    out: list[tuple[float | None, bool | None]] = []
    cur = cur_t = None
    cand: list[tuple[float, float]] = []
    prev_p = None
    for (t, v), p in zip(raw, periods):
        if p is not None and prev_p is not None and p != prev_p:
            cur = None  # new period: the clock restarts
        prev_p = p if p is not None else prev_p
        if v is None:
            out.append((None if cur is None else cur, None))  # keep the last value, running unknown
            continue
        if cur is None:
            cand.append((t, v))
            if len(cand) >= confirm and all(_fits(cand[k - 1], cand[k], sign, tol) for k in range(1, len(cand))):
                cur, cur_t, cand = v, t, []
                out.append((cur, None))
            else:
                cand = cand[-confirm:]
                out.append((None, None))
            continue
        dt = t - cur_t
        stopped = abs(v - cur) <= 0.5
        running = abs(v - (cur + sign * dt)) <= tol and dt > 0
        if stopped or running:
            out.append((v, bool(running and not stopped)))
            cur, cur_t, cand = v, t, []
            continue
        cand.append((t, v))  # a jump: needs agreeing reads
        if len(cand) >= confirm and all(_fits(cand[k - 1], cand[k], sign, tol) for k in range(1, len(cand))):
            cur, cur_t, cand = v, t, []
            out.append((cur, None))
        else:
            cand = cand[-confirm:]
            out.append((cur, None))
    return out


def _fits(a: tuple[float, float], b: tuple[float, float], sign: float, tol: float) -> bool:
    dt = b[0] - a[0]
    return abs(b[1] - a[1]) <= 0.5 or abs(b[1] - (a[1] + sign * dt)) <= tol


def _reread_pair(blocks: list[Block], pair: tuple[int, int], reader) -> dict:
    """A clock drawn as minutes + seconds blocks: read the region spanning both (plus room for a lost leading
    digit) as one line from the stored strips -> {t: seconds or None}."""
    a, b = blocks[pair[0]], blocks[pair[1]]
    h = max(a.h, b.h)
    sx0, sy0, _, _ = reader.strip_box
    box = (max(0, int(a.x0 - 0.6 * h) - sx0), min(a.y0, b.y0) - sy0, b.x1 - sx0, max(a.y1, b.y1) - sy0)
    crops, tt = [], []
    for t, jp in zip(reader.ts, reader.strips):
        if jp is None:
            continue
        img = cv2.imdecode(np.frombuffer(jp, np.uint8), cv2.IMREAD_COLOR)
        c = img[max(0, box[1]):box[3], max(0, box[0]):box[2]]
        if c.shape[0] >= 6 and c.shape[1] >= 4:  # the pair may lie (partly) outside the stored bar strip
            crops.append(c)
            tt.append(t)
    out = {t: None for t, jp in zip(reader.ts, reader.strips) if jp is not None}
    rec = reader._recognizer()
    for i in range(0, len(crops), 256):
        for t, pp in zip(tt[i:i + 256], rec.probs(crops[i:i + 256], max_w=320)):
            txt, c = _greedy(rec, pp)
            txt = txt.replace(" ", "").replace(";", ":").replace(".", ":") if ":" not in txt else txt.replace(" ", "")
            if txt.isdigit() and len(txt) in (3, 4):  # colon not read: "0441" -> "04:41"
                txt = txt[:-2] + ":" + txt[-2:]
            out[t] = _clock_s(txt) if c >= 0.6 else None
    return out


def game_clock_series(blocks: list[Block], cls: dict, ts: list[float], reader=None) -> list:
    """[(t, period or None, clock seconds or None, running True/False/None)] for every sample."""
    if cls.get("clock_pair") and reader is not None and reader.strip_box is not None:
        raw = _reread_pair(blocks, cls["clock_pair"], reader)
    else:
        raw = _raw_clock(blocks, cls, ts)
    if cls.get("period"):
        pr = {t: (_period_value(x) if c >= 0.6 else None) for t, x, c in blocks[cls["period"]["i"]].reads}
    else:
        pr = {}
    periods = smooth_periods([(t, pr.get(t)) for t in ts])
    clk = smooth_clock([(t, raw.get(t)) for t in ts], periods)
    return [(round(t, 2), p, (None if c is None else round(float(c), 1)), r) for t, p, (c, r) in zip(ts, periods, clk)]


def clock_at(res: dict, t: float) -> tuple[int | None, float | None, bool | None]:
    """(period, clock seconds, running) at video time t from a reading (the sample at or before t)."""
    best = None
    for row in res.get("clock") or []:
        if row[0] <= t + 1e-6:
            best = row
        else:
            break
    return (None, None, None) if best is None else (best[1], best[2], best[3])


def played_seconds(res: dict, t0: float, t1: float) -> float | None:
    """Game-clock seconds that ran between video times t0 and t1 (sum over sample pairs with the clock running,
    same period); None when the clock was not read there."""
    rows = [r for r in res.get("clock") or [] if t0 - 1e-6 <= r[0] <= t1 + 1e-6]
    if len(rows) < 2 or not any(r[2] is not None for r in rows):
        return None
    tot = 0.0
    for a, b in zip(rows, rows[1:]):
        if a[2] is not None and b[2] is not None and a[1] == b[1]:
            tot += abs(b[2] - a[2]) if b[3] else 0.0
    return round(tot, 1)


def summary(res: dict, cfg: ScorebarConfig | None = None) -> dict:
    """What callers may trust: start / end score (side 0, side 1) and whether the reading is ``sure``: both sides
    read in most samples, no score correction, the first value locked within the first fifth of the clip."""
    allx = res.get("samples") or []
    smp = [x for x in allx if x[1] is not None and x[2] is not None]
    if not smp or len(allx) < 5:
        return {"sure": False}
    tl = res.get("timeline") or []
    down = any(b[1] < a[1] or b[2] < a[2] for a, b in zip(tl, tl[1:]))
    t0, t1 = allx[0][0], allx[-1][0]
    early = smp[0][0] <= t0 + 0.2 * max(1.0, t1 - t0)
    sure = len(smp) >= 0.6 * len(allx) and not down and early
    gaps = np.diff([x[0] for x in tl])  # safety: score changes every ~60 s like clock minutes -> not a score
    if len(gaps) >= 4 and np.mean(np.abs(gaps - 60.0) < 8.0) >= 0.5:
        sure = False
    per = [x[4] for x in allx if x[4]]
    return {"sure": bool(sure), "start": [smp[0][1], smp[0][2]], "end": [smp[-1][1], smp[-1][2]],
            "t_start": smp[0][0], "t_end": smp[-1][0], "labels": [x["label"] for x in res.get("sides") or []],
            "period_start": per[0] if per else "", "period_end": per[-1] if per else "",
            "changes": max(0, len(tl) - 1), "read_share": round(len(smp) / len(allx), 2)}


# ----------------------------------------------------------------------------------------------- pipeline glue
CACHE = "scorebar.json"
VERSION = 1


def load_result(work: Path) -> dict | None:
    try:
        doc = json.loads((Path(work) / CACHE).read_text(encoding="utf-8"))
        return doc if doc.get("version") == VERSION else None
    except (OSError, ValueError):
        return None


def save_result(work: Path, res: dict) -> None:
    try:
        tmp = Path(work) / (CACHE + ".tmp")
        tmp.write_text(json.dumps(dict(res, version=VERSION), default=str), encoding="utf-8")
        os.replace(tmp, Path(work) / CACHE)
    except OSError:
        pass


def load_or_compute(work: Path, video: str | Path | None, every_s: float = 2.0) -> dict | None:
    """Pass 2: the cached reading (pass-1 tap or an earlier pass 2), else read the video now at ``every_s`` on the
    CPU. Off with env REPLAY_SCOREBAR=0. Never raises."""
    if os.environ.get("REPLAY_SCOREBAR", "1") == "0":
        return None
    doc = load_result(work)
    if doc is not None or video is None or not Path(video).exists():
        return doc
    try:
        res = read_scorebar(video, ScorebarConfig(every_s=every_s))
    except Exception as e:  # noqa: BLE001 - optional
        print("score bar: skipped:", type(e).__name__, e)
        return None
    save_result(work, res)
    sm = res.get("summary") or {}
    print("score bar:", "sure" if sm.get("sure") else "unsure", sm.get("start"), "->", sm.get("end"), res.get("stats"))
    return res


def pin_candidate(running: list, gs: dict | None) -> dict | None:
    """Scoresheet hook (scoresheet.clip_candidates): where a clip sits in the game from the score bar read at its
    start (and end). ``running`` = the sheet's running score (items with score_a, score_b, period, seq). Returns one
    candidate {"start_score", "period", "seq", "flip", "label", "best", "source": "scorebar"} only when the reading
    is sure and exactly one place in the sheet fits (start score; end score later in the same game, either side
    order); else None (the caller keeps its own matching)."""
    sm = ((gs or {}).get("scorebar") or {}).get("summary") or {}
    if not sm.get("sure") or not running or not sm.get("start") or not sm.get("end"):
        return None
    scores = [(0, 0)] + [(int(ev.score_a), int(ev.score_b)) for ev in running]
    hits = []
    for flip in (False, True):
        st = tuple(sm["start"][::-1] if flip else sm["start"])
        en = tuple(sm["end"][::-1] if flip else sm["end"])
        for i, sc in enumerate(scores):
            if sc == st and en in scores[i:]:
                hits.append((i, flip))
    if len(hits) != 1:
        return None
    i, flip = hits[0]
    ev = running[min(i, len(running) - 1)]  # the next scoring event after the clip start: its quarter
    start = list(scores[i])
    period = ev.period
    return {"start_score": start, "period": period, "seq": ev.seq, "flip": flip, "source": "scorebar", "best": True,
            "teams_ok": 0, "points_ok": 0, "scorers_ok": 0, "n": 0,
            "label": (f"Q{period}" if period and period <= 4 else f"OT{period - 4}" if period else "?")
            + f" from {start[0]}–{start[1]} (read from the score bar)"}


class ScorebarTap:
    """Pass-1 frame tap (replay.frame_tap): reads the bar from the frames pass 1 decodes anyway, every ``every_s``,
    with the CPU recogniser in the tap's own thread (never on the GPU the detector uses). Non-blocking: when
    behind, frames are skipped. Writes work/scorebar.json, which pass 2 then reuses."""

    name = "scorebar"
    queue_size = 4
    blocking = False

    def __init__(self, every_s: float = 2.0):
        self.every_s = every_s

    def start(self, info) -> None:
        self.active = not info.resumed
        self.last_t = -np.inf
        self.reader = ScorebarReader(ScorebarConfig(every_s=self.every_s)) if self.active else None

    def wants(self, k: int, t: float) -> bool:
        if not self.active or t - self.last_t < self.every_s - 1e-6:
            return False
        self.last_t = t
        return True

    def process(self, f) -> None:
        self.reader.add(float(f.t), f.image)

    def finish(self, work: Path, completed: bool) -> None:
        if self.active and completed:
            save_result(work, self.reader.result())


if __name__ == "__main__":
    vid = sys.argv[1]
    r = read_scorebar(vid)
    js = json.dumps(r, indent=1, default=str)
    if len(sys.argv) > 2:
        Path(sys.argv[2]).write_text(js, encoding="utf-8")
    print(json.dumps({k: r[k] for k in ("sides", "stats", "roles") if k in r}), "\ntimeline:", r.get("timeline")[:40])
