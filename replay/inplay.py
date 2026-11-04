"""Preliminary analysis: find when the ball is in play, so pass 1 can skip breaks.

Runs before the expensive detection pass and needs only a fast low-resolution decode (ffmpeg, ~2 frames/s,
640 px wide, grey). Measured on the 85-min koristv test game: ~100x real time on a Ryzen 5600X.

Signals (all computed from the video itself, no models, no OCR):

* **Scoreboard clock.** A broadcast score bar shows a game clock whose digits change once per second while it
  runs. It is found automatically: an 8x8-pixel block that is "digital" (between samples 1 s apart it is either
  exactly unchanged or clearly changed, never in between), changes often, and sits next to perfectly static
  overlay blocks (the bar). The bar's static blocks also tell whether the bar is on screen. A clock frozen for a
  long time (pre-game, quarter breaks, halftime) is strong evidence of a break. A *running* clock is not trusted
  as evidence of play: amateur operators often leave it running through breaks.
* **Camera activity.** During play the operator pans to follow the ball; during breaks the camera stays still.
  Measured as the global shift between consecutive samples (phase correlation on a 160-px thumbnail).
* **Activity on the floor.** Share of floor pixels that change after compensating the camera shift (moving
  players). Players standing at the benches (timeouts, breaks) give very low values; normalised by the video's
  own typical in-play level, so it adapts to zoom and resolution.

Combination: a "quiet" stretch (low floor activity and almost no panning, with hysteresis) longer than
``min_break_s`` is a break, and so is a clock frozen for ``clock_stop_break_s`` while the bar is visible.
Everything else is play. Short play blips between breaks are dropped, the play intervals are padded generously
(missing play is worse than analysing a little too much) and merged.

Output: :class:`InPlayResult` with ``intervals`` [(start_s, end_s), ...] in absolute video seconds, saved as
``results/inplay.json`` by :func:`save_result`.
"""

from __future__ import annotations

import json
import subprocess
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from replay.config import InPlayConfig

FORMAT = "advanced-analytist/inplay@1"
FILENAME = "inplay.json"
CLASSIFIER = 2  # bump when the classification changes: cached inplay.json files are recomputed
BLOCK = 8  # px, clock search grid at analysis resolution
THUMB_W = 160
# A frozen clock is NOT a break while the floor and the camera clearly show play: amateur operators also stop
# (or forget) the clock during live play (full koristv game: 2280-2415 s, end of Q2, clock frozen at 00:00 for
# 135 s while 7 baskets were scored). Real breaks: smoothed activity <= ~0.36 x the in-play level and almost no
# panning; that live stretch: 0.83 x and pan share 0.32.
FROZEN_PLAY_ACT_REL = 0.6  # smoothed floor activity at least this x the in-play level ...
FROZEN_PLAY_PAN = 0.2  # ... and the camera pans in at least this share of samples = play despite the clock


class Cancelled(Exception):
    pass


@dataclass
class Signals:
    t: np.ndarray  # absolute sample times, seconds
    pan: np.ndarray  # camera shift since the previous sample, fraction of frame width
    activity: np.ndarray  # share of floor pixels changed after camera compensation
    block_diff: np.ndarray  # (n, by, bx) uint8: mean |I_t - I_(t-1s)| per 8x8 block
    block_mean: np.ndarray  # (n, by, bx) uint8: mean intensity per block
    size: tuple[int, int]  # analysis frame size (w, h)
    fps: float
    lag: int  # samples per second (block_diff lag)


@dataclass
class ClockInfo:
    found: bool
    blocks: list[tuple[int, int]] = field(default_factory=list)  # (row, col) of the ticking blocks
    bar_blocks: int = 0
    box_rel: list[float] | None = None  # [x0, y0, x1, y1] of the ticking blocks, fraction of frame
    tick_rate: float = 0.0  # share of bar-visible samples in which the clock changed
    present_frac: float = 0.0  # share of samples with the bar on screen


@dataclass
class InPlayResult:
    intervals: list[tuple[float, float]]
    start_s: float
    end_s: float
    clock: ClockInfo
    quiet_threshold: float
    play_level: float
    elapsed_s: float = 0.0
    # per-second traces for display / debugging
    trace: dict = field(default_factory=dict)

    @property
    def in_play_s(self) -> float:
        return float(sum(e - s for s, e in self.intervals))

    def to_dict(self) -> dict:
        return {"format": FORMAT, "start_s": round(self.start_s, 3), "end_s": round(self.end_s, 3),
                "intervals": [[round(s, 2), round(e, 2)] for s, e in self.intervals],
                "in_play_s": round(self.in_play_s, 1), "elapsed_s": round(self.elapsed_s, 1),
                "clock": asdict(self.clock), "quiet_threshold": round(self.quiet_threshold, 5),
                "play_level": round(self.play_level, 5), "trace": self.trace}


# ---------------------------------------------------------------------------- decoding / signals
def _video_size(video: Path) -> tuple[int, int, float]:
    from replay.video import probe

    i = probe(video)
    return i.width, i.height, i.duration_s


def analysis_size(w: int, h: int, width: int) -> tuple[int, int]:
    W = min(width, w)
    W -= W % 2
    H = int(round(h * W / w))
    return W, H - H % 2


class SignalExtractor:
    """Feed grey frames (analysis resolution) one at a time; collects all per-sample signals."""

    def __init__(self, size: tuple[int, int], fps: float, t0: float = 0.0):
        self.W, self.H = size
        self.fps, self.t0 = fps, t0
        self.lag = max(1, int(round(fps)))
        self.bx, self.by = self.W // BLOCK, self.H // BLOCK
        self.th_size = (THUMB_W, max(16, int(round(THUMB_W * self.H / self.W))))
        th_h = self.th_size[1]
        self.pc_rows = (int(0.15 * th_h), int(0.80 * th_h))  # walls + floor: no score bar / logo rows
        self.act_rows = (int(0.35 * th_h), int(0.80 * th_h))  # floor, where the players are
        self.win = cv2.createHanningWindow((THUMB_W, self.pc_rows[1] - self.pc_rows[0]), cv2.CV_32F)
        self.hist: deque[np.ndarray] = deque(maxlen=self.lag)
        self.prev_th: np.ndarray | None = None
        self.pan: list[float] = []
        self.act: list[float] = []
        self.bdiff: list[np.ndarray] = []
        self.bmean: list[np.ndarray] = []

    def add(self, gray: np.ndarray) -> None:
        crop = gray[:self.by * BLOCK, :self.bx * BLOCK]
        self.bmean.append(cv2.resize(crop, (self.bx, self.by), interpolation=cv2.INTER_AREA))
        if len(self.hist) == self.lag:
            d = cv2.absdiff(crop, self.hist[0])
            self.bdiff.append(cv2.resize(d, (self.bx, self.by), interpolation=cv2.INTER_AREA))
        else:
            self.bdiff.append(np.zeros((self.by, self.bx), np.uint8))
        self.hist.append(crop)

        th = cv2.resize(gray, self.th_size, interpolation=cv2.INTER_AREA).astype(np.float32)
        pan, act = 0.0, 0.0
        if self.prev_th is not None:
            r0, r1 = self.pc_rows
            (dx, dy), _ = cv2.phaseCorrelate(self.prev_th[r0:r1], th[r0:r1], self.win)
            pan = float(np.hypot(dx, dy)) / THUMB_W
            warped = cv2.warpAffine(self.prev_th, np.float32([[1, 0, dx], [0, 1, dy]]), self.th_size,
                                    borderMode=cv2.BORDER_REPLICATE)
            a0, a1 = self.act_rows
            diff = np.abs(warped[a0:a1, 4:-4] - th[a0:a1, 4:-4])
            act = float((diff > 15).mean())
        self.prev_th = th
        self.pan.append(pan)
        self.act.append(act)

    def signals(self) -> Signals:
        n = len(self.pan)
        empty = np.zeros((0, self.by, self.bx), np.uint8)
        return Signals(t=self.t0 + np.arange(n) / self.fps, pan=np.array(self.pan), activity=np.array(self.act),
                       block_diff=np.array(self.bdiff) if n else empty, block_mean=np.array(self.bmean) if n else empty,
                       size=(self.W, self.H), fps=self.fps, lag=self.lag)


def extract_signals(video: str | Path, cfg: InPlayConfig, start_s: float = 0.0, end_s: float | None = None,
                    progress: Callable[[float], None] | None = None,
                    cancel: threading.Event | None = None) -> Signals:
    import imageio_ffmpeg

    video = Path(video)
    w, h, duration = _video_size(video)
    end = duration if end_s is None else min(end_s, duration or end_s)
    W, H = analysis_size(w, h, cfg.analysis_width)
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-nostdin"]
    if start_s > 0:
        cmd += ["-ss", f"{start_s:.3f}"]
    cmd += ["-i", str(video)]
    if end_s is not None:
        cmd += ["-t", f"{max(0.0, end - start_s):.3f}"]
    cmd += ["-an", "-sn", "-dn", "-vf", f"fps={cfg.sample_fps},scale={W}:{H}:flags=area,format=gray",
            "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    ex = SignalExtractor((W, H), cfg.sample_fps, start_s)
    expected = max(1.0, (end - start_s) * cfg.sample_fps)
    nbytes = W * H
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=nbytes * 4,
                            creationflags=flags)
    try:
        while True:
            buf = proc.stdout.read(nbytes)
            if len(buf) < nbytes:
                break
            ex.add(np.frombuffer(buf, np.uint8).reshape(H, W))
            n = len(ex.pan)
            if n % 50 == 0:
                if progress is not None:
                    progress(min(1.0, n / expected))
                if cancel is not None and cancel.is_set():
                    raise Cancelled()
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait()
    if not ex.pan:
        err = proc.stderr.read().decode("utf-8", "replace") if proc.stderr else ""
        raise RuntimeError(f"Preliminary analysis could not decode {video}: {err.strip()[-300:]}")
    return ex.signals()


# ---------------------------------------------------------------------------- clock
def find_clock(sig: Signals, cfg: InPlayConfig) -> tuple[ClockInfo, np.ndarray, np.ndarray]:
    """Locate a ticking scoreboard clock. Returns (info, tick[n] bool, present[n] bool)."""
    n = len(sig.t)
    none = (ClockInfo(False), np.zeros(n, bool), np.zeros(n, bool))
    if n < 6 * sig.lag:
        return none
    D = sig.block_diff[sig.lag:].astype(np.float32)  # first `lag` samples have no 1-s partner
    zero = D < cfg.clock_zero_level
    hi = D >= cfg.clock_change_level
    digital = (zero | hi).mean(0)
    rate = hi.mean(0)
    static = zero.mean(0)  # overlay background blocks: ~always unchanged
    by, bx = static.shape
    # The digits sit *inside* the bar: static overlay blocks on both the left and the right in the same block
    # row (within 3 blocks). A scene block at the edge of, above or below the bar fails this.
    pad = np.pad(static, ((0, 0), (3, 3)))
    left = np.zeros_like(static)
    right = np.zeros_like(static)
    for dx in (1, 2, 3):
        left = np.maximum(left, pad[:, 3 - dx:3 - dx + bx])
        right = np.maximum(right, pad[:, 3 + dx:3 + dx + bx])
    neigh = np.minimum(left, right)
    score = digital * rate * neigh
    score[:, [0, -1]] = 0
    score[[0, -1], :] = 0
    r, c = np.unravel_index(int(np.argmax(score)), score.shape)
    if digital[r, c] < cfg.clock_min_digital or rate[r, c] < cfg.clock_min_tick_rate or neigh[r, c] < 0.5:
        return none
    # the digit blocks that change together with the best one (seconds digit; minutes change rarely)
    near = [(rr, cc) for rr in range(max(0, r - 1), min(by, r + 2)) for cc in range(max(0, c - 3), min(bx, c + 4))]
    blocks = [(rr, cc) for rr, cc in near if digital[rr, cc] >= cfg.clock_min_digital
              and rate[rr, cc] >= 0.5 * rate[r, c] and np.mean(hi[:, rr, cc] == hi[:, r, c]) > 0.9]
    # bar background: static overlay blocks in a window around the clock
    bar = [(rr, cc) for rr in range(max(0, r - 2), min(by, r + 3)) for cc in range(max(0, c - 12), min(bx, c + 13))
           if static[rr, cc] >= 0.6 and (rr, cc) not in blocks]
    if len(bar) < 4:
        return none
    M = sig.block_mean.astype(np.int16)
    ref = np.median(np.stack([M[:, rr, cc] for rr, cc in bar], 1), 0)
    dev = np.abs(np.stack([M[:, rr, cc] for rr, cc in bar], 1) - ref[None])
    present = (dev <= cfg.clock_bar_tolerance).mean(1) >= 0.75
    Dfull = sig.block_diff.astype(np.float32)
    tick = np.zeros(n, bool)
    for rr, cc in blocks:
        tick |= Dfull[:, rr, cc] >= cfg.clock_change_level
    tick[:sig.lag] = False
    present[:sig.lag] = present[sig.lag] if n > sig.lag else False
    tick &= present
    pf = float(present.mean())
    tr = float(tick[present].mean()) if present.any() else 0.0
    rows = [b[0] for b in blocks]
    cols = [b[1] for b in blocks]
    W, H = sig.size
    box = [min(cols) * BLOCK / W, min(rows) * BLOCK / H, (max(cols) + 1) * BLOCK / W, (max(rows) + 1) * BLOCK / H]
    info = ClockInfo(True, [(int(a), int(b)) for a, b in blocks], len(bar), [round(v, 4) for v in box],
                     round(tr, 3), round(pf, 3))
    if pf < 0.2:
        return ClockInfo(False, info.blocks, info.bar_blocks, info.box_rel, info.tick_rate, info.present_frac), \
            np.zeros(n, bool), np.zeros(n, bool)
    return info, tick, present


# ---------------------------------------------------------------------------- classification
def _smooth(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1 or len(x) == 0:
        return x.astype(float)
    k = min(k, len(x))
    c = np.cumsum(np.r_[0.0, np.pad(x.astype(float), (k // 2, k - 1 - k // 2), mode="edge")])
    return (c[k:] - c[:-k]) / k


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index runs where mask is True."""
    m = np.r_[False, mask.astype(bool), False]
    d = np.flatnonzero(m[1:] != m[:-1])
    return list(zip(d[::2].tolist(), d[1::2].tolist()))


def _hysteresis(low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """True on runs of ``high`` that contain at least one ``low`` sample (low implies high)."""
    out = np.zeros(len(high), bool)
    for a, b in _runs(high):
        if low[a:b].any():
            out[a:b] = True
    return out


def classify(sig: Signals, cfg: InPlayConfig, tick: np.ndarray | None = None,
             present: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """Per-sample in-play flag (before padding) and the intermediate values."""
    n = len(sig.t)
    fps = sig.fps
    k = max(1, int(round(cfg.window_s * fps)))
    act_s = _smooth(sig.activity, k)
    pan_s = _smooth(sig.pan > cfg.pan_move, k)
    play_level = float(np.percentile(act_s, 75)) if n else 0.0
    thr = float(np.clip(cfg.quiet_rel * play_level, cfg.quiet_min, cfg.quiet_max))
    low = (act_s < thr) & (pan_s < cfg.quiet_pan_frac)
    high = (act_s < thr * cfg.hysteresis) & (pan_s < 2 * cfg.quiet_pan_frac)
    quiet = _hysteresis(low, high)
    brk = np.zeros(n, bool)
    for a, b in _runs(quiet):
        if (b - a) / fps >= cfg.min_break_s:
            brk[a:b] = True
    # A camera that normally follows play but stays still for a long time = break, even if people walk around
    # (teams walking back on court before a restart). Disabled for footage that hardly ever pans.
    pan_level = float(np.percentile(pan_s, 75)) if n else 0.0
    still = np.zeros(n, bool)
    if pan_level >= cfg.min_pan_level:
        for a, b in _runs(pan_s < cfg.quiet_pan_frac):
            # ... but half-court play and free throws also keep the camera still: require clearly less movement
            # on the floor than in play
            if (b - a) / fps >= cfg.still_break_s and \
                    np.median(sig.activity[a:b]) < cfg.still_activity_rel * play_level:
                still[a:b] = True
        brk |= still
    frozen = np.zeros(n, bool)
    if tick is not None and present is not None and cfg.use_clock and present.any():
        for a, b in _runs(~tick):
            if (b - a) / fps >= cfg.clock_stop_break_s and present[a:b].mean() >= 0.8:
                # the clock may restart a moment after play resumes: keep the last seconds as play
                frozen[a:max(a, b - int(round(cfg.clock_restart_slack_s * fps)))] = True
        # (never below 2x the quiet threshold: footage with almost no floor activity gives a tiny play level)
        lively = (act_s >= max(FROZEN_PLAY_ACT_REL * play_level, 2.0 * thr)) & (pan_s >= FROZEN_PLAY_PAN)
        frozen &= ~lively
        for a, b in _runs(frozen):  # what is left between lively stretches must still be a long stop
            if (b - a) / fps < cfg.clock_stop_break_s:
                frozen[a:b] = False
        brk |= frozen
    play = ~brk
    for a, b in _runs(play):  # short play blips between two breaks (warm-up shots, camera jolts)
        if (b - a) / fps < cfg.min_play_s and a > 0 and b < n:
            play[a:b] = False
    return play, {"activity": act_s, "pan": pan_s, "threshold": thr, "play_level": play_level, "quiet": quiet,
                  "frozen": frozen, "still": still, "pan_level": pan_level}


def to_intervals(play: np.ndarray, t: np.ndarray, fps: float, pad_s: float, merge_gap_s: float,
                 start_s: float, end_s: float) -> list[tuple[float, float]]:
    out: list[list[float]] = []
    for a, b in _runs(play):
        s = max(start_s, float(t[a]) - pad_s)
        e = min(end_s, float(t[b - 1]) + 1.0 / fps + pad_s)
        if out and s - out[-1][1] <= merge_gap_s:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out if e > s]


def analyse_signals(sig: Signals, cfg: InPlayConfig, start_s: float, end_s: float) -> InPlayResult:
    info, tick, present = find_clock(sig, cfg) if cfg.use_clock else (ClockInfo(False), None, None)
    if not info.found:
        tick = present = None
    play, dbg = classify(sig, cfg, tick, present)
    iv = to_intervals(play, sig.t, sig.fps, cfg.pad_s, cfg.merge_gap_s, start_s, end_s)
    # 1-s traces for the GUI / debugging
    step = max(1, int(round(sig.fps)))
    trace = {"t0": round(float(sig.t[0]) if len(sig.t) else start_s, 3), "dt": step / sig.fps,
             "activity": [round(float(v), 4) for v in dbg["activity"][::step]],
             "pan": [round(float(v), 3) for v in dbg["pan"][::step]],
             "play": [int(v) for v in play[::step]]}
    if tick is not None:
        trace["clock"] = [(-1 if not p else int(c)) for c, p in zip(tick[::step], present[::step])]
    return InPlayResult(iv, start_s, end_s, info, dbg["threshold"], dbg["play_level"], trace=trace)


def run_preliminary(video: str | Path, cfg: InPlayConfig, start_s: float = 0.0, end_s: float | None = None,
                    progress: Callable[[float], None] | None = None,
                    cancel: threading.Event | None = None) -> InPlayResult:
    """Preliminary analysis of a whole video (or [start_s, end_s]): returns the in-play intervals."""
    t0 = time.time()
    _, _, duration = _video_size(Path(video))
    end = duration if end_s is None else min(end_s, duration or end_s)
    sig = extract_signals(video, cfg, start_s, end_s, progress, cancel)
    end = max(end, float(sig.t[-1]) + 1.0 / sig.fps) if end_s is None else end
    res = analyse_signals(sig, cfg, start_s, end)
    res.elapsed_s = time.time() - t0
    if progress is not None:
        progress(1.0)
    return res


# ---------------------------------------------------------------------------- io
def save_result(res: InPlayResult, path: Path, video: Path | None = None, cfg: InPlayConfig | None = None,
                requested_end_s: float | None = None) -> None:
    d = res.to_dict()
    d["requested_end_s"] = requested_end_s
    d["classifier"] = CLASSIFIER
    if video is not None:
        d["video"] = str(Path(video).resolve())
        d["video_size_bytes"] = Path(video).stat().st_size
    if cfg is not None:
        d["config"] = asdict(cfg)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(d), encoding="utf-8")
    tmp.replace(path)


def load_intervals(path: Path) -> list[tuple[float, float]] | None:
    try:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return [(float(s), float(e)) for s, e in d["intervals"]]
    except (OSError, ValueError, KeyError, TypeError):
        return None


def cached_result(path: Path, video: Path, cfg: InPlayConfig, start_s: float = 0.0,
                  end_s: float | None = None) -> list[tuple[float, float]] | None:
    """Intervals from an existing inplay.json made for this video, range and settings, else None."""
    try:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        if d.get("video") != str(Path(video).resolve()) or d.get("video_size_bytes") != Path(video).stat().st_size:
            return None
        if abs(float(d.get("start_s", 0)) - start_s) > 1e-3 or d.get("requested_end_s") != end_s:
            return None
        if d.get("config") != json.loads(json.dumps(asdict(cfg))) or d.get("classifier", 1) != CLASSIFIER:
            return None
        return [(float(s), float(e)) for s, e in d["intervals"]]
    except (OSError, ValueError, KeyError, TypeError):
        return None


def describe(intervals: list[tuple[float, float]], total_s: float) -> str:
    tot = sum(e - s for s, e in intervals)
    pct = 100.0 * tot / total_s if total_s > 0 else 0.0
    return f"{len(intervals)} in-play intervals, {tot / 60:.1f} of {total_s / 60:.1f} min ({pct:.0f}%)"


def intervals_for_pass1(video: Path, out: Path, cfg: InPlayConfig, start_s: float = 0.0, end_s: float | None = None,
                        restart: bool = False, progress: Callable[[float], None] | None = None,
                        cancel: threading.Event | None = None,
                        log: Callable[[str], None] = print) -> list[tuple[float, float]] | None:
    """The play intervals to hand to ``run_pass1`` (None = analyse everything).

    * disabled in the config -> None
    * an existing pass-1 run that was started *without* intervals is resumed as it was (None), unless restarting
    * otherwise ``out/inplay.json`` is reused when it matches this video + settings, else the preliminary
      analysis runs and writes it.
    """
    if not cfg.enabled:
        return None
    run_json = Path(out) / "work" / "run.json"
    if not restart and run_json.exists():
        try:
            old = json.loads(run_json.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            old = {}
        if "play_intervals" not in old:
            log("Existing analysis covers the whole video; continuing it without the preliminary analysis.")
            return None
        return [(float(a), float(b)) for a, b in old["play_intervals"]]
    path = Path(out) / FILENAME
    cached = cached_result(path, video, cfg, start_s, end_s)
    if cached is not None:
        log(f"Preliminary analysis: reusing {path} ({len(cached)} in-play intervals)")
        return cached
    log("Preliminary analysis: finding when the ball is in play ...")
    res = run_preliminary(video, cfg, start_s, end_s, progress, cancel)
    save_result(res, path, video, cfg, end_s)
    total = res.end_s - res.start_s
    log(f"Preliminary analysis done in {res.elapsed_s:.0f}s ({total / max(res.elapsed_s, 1e-6):.0f}x real time): "
        + describe(res.intervals, total))
    return res.intervals
