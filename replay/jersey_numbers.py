"""Jersey number recognition (pass 2, cached) and per-track voting.

Pipeline
  1. **Sampling** - frames on a regular time grid (``JerseyConfig.every_s``) are decoded sequentially from the
     video (like team_appearance.py). In each, person boxes that are large enough, fully inside the image and
     not overlapping anybody else's box give a few tight "number windows" on the torso (front and back numbers
     sit at ~10-40 % of the box height); one window is read first, the others only when it shows a digit.
  2. **Recognition** - PP-OCRv4 mobile text recognition (PaddleOCR, Apache-2.0; converted to ONNX by
     replay/ppocr_onnx.py), run with ONNX Runtime (DirectML GPU, CPU fallback). Small crops are upscaled with a
     cubic filter. The CTC output is decoded **digits only**: per time step the choice is blank, one of the ten
     digits, or "something else" (summed probability of all other characters); a read is one contiguous run of
     1-2 digits, a valid basketball number (0-99 and 00; no leading zero otherwise), whose every digit is
     confident. Everything else is no read.
  3. **Voting** (``vote``) - per track / identity: a number is accepted only when read consistently: at least
     ``min_reads`` agreeing reads and a clear majority over contradicting reads. A single-digit read that is
     part of the leading two-digit number ("4" or "1" of "41": one digit hidden by an arm or a fold) neither
     supports nor contradicts it; a one-digit number is doubted twice as much by two-digit reads containing it,
     and is rejected outright when such a two-digit number was read twice ("6" of a half-hidden "36").
  4. **Same-team uniqueness** (``resolve_conflicts``) - two simultaneously present tracks of one team cannot
     wear the same number: the one with more votes keeps it. Reads on the other team's jersey colour and
     one-digit numbers that are part of a team-mate's number are dropped (``identity_numbers``).
  5. **Identity cue** (``piece_numbers`` / ``link_cost``) - tracklets.py does not link pieces that show
     different numbers (penalty, or impossible when both are well read).

Reads are cached per pass-1 row in ``work/jersey_numbers.npz`` (code, confidence, jersey colour behind the
read), keyed by the pass-1 run and the sampling settings, so pass 2 stays re-runnable without the video.

Number codes (int): -1 no read, 0..99 the number, 100 = "00".

API for other modules (identity linking, lineup, exports)
  * ``load_reads(work) -> JerseyReads | None`` - the cached reads only (never decodes video). Per pass-1 row
    (index into Pass1Data.rows): ``code`` (int16, -1 none), ``conf`` (mean digit probability; NaN = row not
    sampled), ``lab`` (jersey colour behind the read). File: ``results/work/jersey_numbers.npz`` (keys key, code,
    conf, lab, stats). ``load_or_compute(work, data, video)`` computes it when missing (pass 2 does this).
  * ``vote(codes)`` / ``vote_groups(group, codes)`` - accepted number per group with support / against counts.
  * ``identity_numbers(tid, codes, k, team_of, ...)`` - accepted number per identity incl. same-team
    uniqueness and the colour / one-digit checks (what pass 2 writes).
  * ``piece_numbers(codes)`` + ``link_cost(na, nb)`` - linking cue (bonus / penalty / None = impossible).
  * Output: tracking.json "tracks" and replay_data.json "track_info": ``number`` (text, e.g. "7", "23", "00";
    only when accepted) and ``number_reads`` (agreeing reads, >= 3).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")
import cv2  # noqa: E402

CACHE_VERSION = 2
NO_READ = -1
DOUBLE_ZERO = 100
MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "pp-ocrv4-mobile-rec"
MODEL_PATH = MODEL_DIR / "pp-ocrv4-mobile-rec.onnx"
REC_H = 48  # model input height


@dataclass
class JerseyConfig:
    enabled: bool = True
    every_s: float = 0.1  # sampling grid of decoded frames (0.1 = every pass-1 sample at 10 fps); numbers are
    # visible only in some poses, so dense sampling pays (clip-4400: 12 / 18 / 33 tracks got a number at
    # 0.3 / 0.2 / 0.1 s)
    cpu_every_s: float = 0.3  # without a GPU: OCR is ~10x slower (clip-4400: 185 s per video minute at 0.1 s
    # on a loaded 8-core CPU), so sample sparser (fewer numbers, ~12 instead of ~24 tracks on clip-4400)
    min_box_h: float = 95.0  # px; smaller boxes gave <1% reads at 720p
    max_overlap: float = 0.02  # torso region covered by another person's box (share) -> skipped
    behind_margin: float = 0.0  # >0: overlap by a box whose bottom is this much (x our height) higher is ignored
    edge_px: float = 2.0  # boxes touching the image border are cut -> skipped
    # number windows (y0, y1, x0, x1) as fractions of the box: the recognizer reads a text *line*, so the window
    # must be tight around the number; boxes vary (arms up, bent over), so a few vertical positions are tried.
    # Chosen on clip-4400 (most two-digit reads among 3-6 window sets; numbers sit at ~10-40 % of the box height)
    # The first ``track_windows`` (centred) windows give ``code`` = the reads the tracker / linker / lineup use
    # (unchanged since 13.10). The extra windows (bands shifted left / right: a turned player's number is
    # off-centre) are read only when the first window shows a clear digit (``extra_signal``) and only feed
    # ``code_label`` = the reads behind the number LABELS (+25 % two-digit reads on clip-4400, same precision).
    # Feeding them to the linker cost 0.9-1.7 points of player recall (18.10 gate), hence the split.
    windows: tuple = ((0.12, 0.28, 0.27, 0.73), (0.16, 0.32, 0.27, 0.73), (0.20, 0.36, 0.27, 0.73),
                      (0.25, 0.41, 0.27, 0.73), (0.14, 0.36, 0.22, 0.78),
                      (0.20, 0.36, 0.15, 0.61), (0.20, 0.36, 0.39, 0.85))
    track_windows: int = 5
    # 2 extra windows at signal 0.5: +9 % crops for 32 of the 33 extra-window reads the 6-window set found
    # (clip-4400 sample; 6 windows at 0.2 cost +43 % reading time in the 18.10 gate)
    extra_signal: float = 0.5
    # region (y0, y1, x0, x1) that must not be covered by another person (the centred windows' union)
    overlap_region: tuple = (0.12, 0.41, 0.22, 0.78)
    # cascade: these windows first; the others only when one of them shows a digit-like shape (any digit
    # probability >= first_signal). clip-4400: 45 % of the crops for 81 % of the reads
    first_windows: tuple = (2,)
    first_signal: float = 0.02
    crop_h: int = 64  # crops are resized to this height before the model's 48 (cubic upscaling of small ones)
    max_w: int = 160  # model input width cap (px at height 48)
    batch: int = 128  # crops per model call (similar widths are batched together)
    min_digit_conf: float = 0.80  # every digit of a read must be at least this probable
    min_read_conf: float = 0.85  # mean digit probability of a read
    min_single_conf: float = 0.92  # one-digit reads (stripes, arms and logos give more false single digits)
    # voting
    min_reads: int = 3  # agreeing reads needed to accept a number
    majority: float = 0.66  # agreeing / (agreeing + contradicting) reads
    # one-digit numbers 0-9 are off by default: on clip-4400 / bench3min most accepted one-digit numbers were a
    # half-hidden two-digit number or a stripe ("0", "6" of 36, "3" of 36); two-digit ones were right
    allow_single_digit: bool = False
    # tracks with fewer reads take a number another track of the team already carries (see _anchor); 0 = off
    anchor_reads: int = 1  # p4400 / c900 player-time numbered +6 / +5 points, no new wrong number (18.10)
    # speed (reading is ~70 % of pass 2 on a full game): per ByteTrack id, once its reads already give a confident
    # number (``decided_reads`` agreeing reads at the ``majority``), keep checking it only every
    # ``decided_every_s``; undecided ids are read at most every ``track_every_s``. 0 = read every sample.
    decided_reads: int = 0  # OFF: on clip-4400 (6 / 2 s) it saved ~20 % of the reading time, but 2 of 13
    # numbered tracks lost their number and identity linking changed slightly
    decided_every_s: float = 2.0
    track_every_s: float = 0.0
    # (identity linking in tracklets.py: see piece_numbers / link_cost at the end of this module)
    device: str = "auto"  # auto | gpu | cpu


WIDE_WINDOWS = ((0.12, 0.28, 0.27, 0.73), (0.16, 0.32, 0.27, 0.73), (0.20, 0.36, 0.27, 0.73),
                (0.25, 0.41, 0.27, 0.73), (0.14, 0.36, 0.22, 0.78),
                (0.16, 0.32, 0.15, 0.61), (0.16, 0.32, 0.39, 0.85), (0.20, 0.36, 0.15, 0.61),
                (0.20, 0.36, 0.39, 0.85), (0.25, 0.41, 0.15, 0.61), (0.25, 0.41, 0.39, 0.85))


def default_config() -> JerseyConfig:
    """JerseyConfig defaults, overridable for A/B runs with env REPLAY_JERSEY_CFG='{"anchor_reads": 0, ...}'."""
    raw = os.environ.get("REPLAY_JERSEY_CFG")
    if not raw:
        return JerseyConfig()
    over = json.loads(raw)
    tup = lambda v: tuple(tup(x) for x in v) if isinstance(v, list) else v  # noqa: E731
    return JerseyConfig(**{k: tup(v) for k, v in over.items()})


# ----------------------------------------------------------------------------------------------- codes
def code_of(s: str | None) -> int:
    """'23' -> 23, '00' -> 100, '0' -> 0; anything else (None, '07', '123', 'ab') -> -1."""
    if not s or not s.isdigit() or len(s) > 2:
        return NO_READ
    if s == "00":
        return DOUBLE_ZERO
    if len(s) == 2 and s[0] == "0":
        return NO_READ
    return int(s)


def text_of(code: int) -> str | None:
    code = int(code)
    if code < 0:
        return None
    return "00" if code == DOUBLE_ZERO else str(code)


def is_partial(read: int, full: int) -> bool:
    """A one-digit read that is one of the two digits of the two-digit number ``full`` (other digit missed)."""
    r, f = text_of(read), text_of(full)
    return r is not None and f is not None and len(r) == 1 and len(f) == 2 and r in f


# ----------------------------------------------------------------------------------------------- decoding
def load_charset(path: Path | None = None) -> list[str]:
    """CTC classes: [blank] + character_dict + [space] (PaddleOCR CTCLabelDecode with use_space_char)."""
    p = Path(path) if path else MODEL_DIR / "charset.txt"
    chars = p.read_text(encoding="utf-8").split("\n")
    if chars and chars[-1] == "":
        chars = chars[:-1]
    return [""] + chars + [" "]


def digit_classes(charset: list[str]) -> np.ndarray:
    """Class index of '0'..'9' (index = digit)."""
    return np.array([charset.index(str(d)) for d in range(10)], int)


def decode_digits(probs: np.ndarray, digit_idx: np.ndarray, min_digit_conf: float = 0.8,
                  min_read_conf: float = 0.85) -> tuple[int, float]:
    """Digits-only greedy CTC decode of one (T, C) softmax output -> (number code, confidence) or (-1, conf).

    Per step: blank, digit d, or "other" (summed mass of every other character). Consecutive equal choices
    collapse (CTC). A read is exactly one run of 1-2 digits with no "other" character between them; letters
    elsewhere on the crop (team names) are allowed around it.
    """
    p = np.asarray(probs, np.float32)
    if p.ndim != 2 or not len(p):
        return NO_READ, 0.0
    blank = p[:, 0]
    dig = p[:, digit_idx]
    other = np.clip(1.0 - blank - dig.sum(axis=1), 0.0, None)
    choice = np.argmax(np.c_[blank, dig, other], axis=1)  # 0 blank, 1..10 digits, 11 other
    step_p = np.c_[blank, dig, other][np.arange(len(p)), choice]
    tokens: list[tuple[int, float]] = []  # (choice, max prob over its run)
    prev = -1
    for c, sp in zip(choice.tolist(), step_p.tolist()):
        if c != prev and c != 0:
            tokens.append((c, sp))
        elif c == prev and c != 0:
            tokens[-1] = (c, max(tokens[-1][1], sp))
        prev = c
    # digit runs separated by "other" tokens
    runs: list[list[tuple[int, float]]] = []
    cur: list[tuple[int, float]] = []
    for c, sp in tokens:
        if 1 <= c <= 10:
            cur.append((c - 1, sp))
        else:
            if cur:
                runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    if len(runs) != 1:
        return NO_READ, 0.0
    run = runs[0]
    confs = [sp for _, sp in run]
    conf = float(np.mean(confs))
    code = code_of("".join(str(d) for d, _ in run))
    if code < 0 or min(confs) < min_digit_conf or conf < min_read_conf:
        return NO_READ, conf
    return code, conf


# ----------------------------------------------------------------------------------------------- recognizer
class Recognizer:
    """PP-OCRv4 mobile text-line recognizer on ONNX Runtime (DirectML GPU when available, else CPU)."""

    def __init__(self, model: Path | None = None, device: str = "auto"):
        import onnxruntime as ort

        path = Path(model) if model else MODEL_PATH
        if not path.exists():
            raise FileNotFoundError(f"jersey number model missing: {path} (see models/pp-ocrv4-mobile-rec/NOTICE.txt)")
        so = ort.SessionOptions()
        so.log_severity_level = 3
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        avail = ort.get_available_providers()
        if device in ("auto", "gpu") and "DmlExecutionProvider" in avail:
            so.enable_mem_pattern = False
            so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            providers = [("DmlExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
        elif device in ("auto", "gpu") and "CUDAExecutionProvider" in avail:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            so.intra_op_num_threads = max(1, (os.cpu_count() or 2) // 2)
            providers = ["CPUExecutionProvider"]
        try:
            self.session = ort.InferenceSession(str(path), sess_options=so, providers=providers)
        except Exception:  # noqa: BLE001 - GPU runtime present but unusable: CPU
            so = ort.SessionOptions()
            so.log_severity_level = 3
            self.session = ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])
        ep = self.session.get_providers()[0]
        self.device = {"DmlExecutionProvider": "GPU (DirectML)", "CUDAExecutionProvider": "GPU (CUDA)"}.get(ep, "CPU")
        self.charset = load_charset(path.parent / "charset.txt")
        self.digit_idx = digit_classes(self.charset)

    @staticmethod
    def prepare(crop_bgr: np.ndarray, max_w: int = 160) -> np.ndarray:
        """(3, 48, w) float32: height 48, aspect kept (width capped), PaddleOCR normalisation."""
        h, w = crop_bgr.shape[:2]
        nw = int(min(max_w, max(8, np.ceil(REC_H * w / max(h, 1)))))
        interp = cv2.INTER_CUBIC if h < REC_H else cv2.INTER_AREA
        r = cv2.resize(crop_bgr, (nw, REC_H), interpolation=interp).astype(np.float32)
        return ((r / 255.0 - 0.5) / 0.5).transpose(2, 0, 1)

    def probs(self, crops: list[np.ndarray], max_w: int = 160, batch: int = 128) -> list[np.ndarray]:
        """Softmax output (T, C) per crop. Crops of similar width are batched (zero padding on the right)."""
        if not crops:
            return []
        xs = [self.prepare(c, max_w) for c in crops]
        order = np.argsort([x.shape[2] for x in xs], kind="stable")
        out: list[np.ndarray | None] = [None] * len(xs)
        for s in range(0, len(order), batch):
            idx = order[s:s + batch]
            W = max(xs[i].shape[2] for i in idx)
            W = int(np.ceil(W / 8) * 8)
            X = np.zeros((len(idx), 3, REC_H, W), np.float32)
            for j, i in enumerate(idx):
                X[j, :, :, : xs[i].shape[2]] = xs[i]
            P = self.session.run(None, {self.session.get_inputs()[0].name: X})[0]
            for j, i in enumerate(idx):
                t = max(1, int(np.ceil(xs[i].shape[2] / 8)))  # steps covering the crop (padding excluded)
                out[i] = P[j, :t]
        return out  # type: ignore[return-value]

    def text(self, probs: np.ndarray) -> str:
        """Unrestricted greedy CTC decode (debugging / contact sheets)."""
        a = probs.argmax(axis=1)
        return "".join(self.charset[c] for j, c in enumerate(a) if c and (j == 0 or c != a[j - 1]))

    def read(self, crops: list[np.ndarray], cfg: JerseyConfig | None = None) -> list[tuple[int, float]]:
        """(number code or -1, confidence) per crop."""
        return [(c, p) for c, p, _ in self.read_with_signal(crops, cfg)]

    def read_with_signal(self, crops: list[np.ndarray], cfg: JerseyConfig | None = None
                         ) -> list[tuple[int, float, float]]:
        """(code, confidence, digit signal) per crop; signal = highest digit probability at any step (a crop
        without any digit-like shape scores ~0: its other windows are not worth reading)."""
        cfg = cfg or default_config()
        out = []
        for p in self.probs(crops, cfg.max_w, cfg.batch):
            c, conf = decode_digits(p, self.digit_idx, cfg.min_digit_conf, cfg.min_read_conf)
            if 0 <= c <= 9 and conf < cfg.min_single_conf:
                c = NO_READ
            out.append((c, conf, float(p[:, self.digit_idx].max()) if len(p) else 0.0))
        return out


# ----------------------------------------------------------------------------------------------- crops
def eligible(boxes: np.ndarray, size: tuple[int, int], cfg: JerseyConfig) -> np.ndarray:
    """Boxes worth reading: tall enough, inside the image, torso not overlapped by any other person box."""
    b = np.asarray(boxes, float).reshape(-1, 4)
    n = len(b)
    if not n:
        return np.zeros(0, bool)
    W, H = size
    h = b[:, 3] - b[:, 1]
    w = b[:, 2] - b[:, 0]
    ok = (h >= cfg.min_box_h) & (w >= 8)
    ok &= (b[:, 0] > cfg.edge_px) & (b[:, 1] > cfg.edge_px) & (b[:, 2] < W - cfg.edge_px) & (b[:, 3] < H - cfg.edge_px)
    y0, y1, x0, x1 = cfg.overlap_region
    tx1, tx2 = b[:, 0] + x0 * w, b[:, 0] + x1 * w
    ty1, ty2 = b[:, 1] + y0 * h, b[:, 1] + y1 * h
    area = np.maximum(1e-6, (tx2 - tx1) * (ty2 - ty1))
    ix = np.clip(np.minimum(tx2[:, None], b[None, :, 2]) - np.maximum(tx1[:, None], b[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(ty2[:, None], b[None, :, 3]) - np.maximum(ty1[:, None], b[None, :, 1]), 0, None)
    cover = ix * iy / area[:, None]
    np.fill_diagonal(cover, 0.0)
    if cfg.behind_margin > 0:  # someone standing BEHIND (feet higher in the image) cannot hide our number
        behind = b[None, :, 3] < b[:, None, 3] - cfg.behind_margin * h[:, None]
        cover = np.where(behind, 0.0, cover)
    ok &= cover.max(axis=1) <= cfg.max_overlap
    return ok


def torso_crops(image_bgr: np.ndarray, box, cfg: JerseyConfig) -> list[np.ndarray | None]:
    """One crop per number window (cfg.windows); None where the window falls outside the image."""
    return [window_crop(image_bgr, box, w, cfg.crop_h) for w in cfg.windows]


def window_crop(image_bgr: np.ndarray, box, window, crop_h: int = 64) -> np.ndarray | None:
    x1, y1, x2, y2 = (float(v) for v in box)
    w, h = x2 - x1, y2 - y1
    y0r, y1r, x0r, x1r = window
    H, W = image_bgr.shape[:2]
    ya, yb = max(0, int(round(y1 + y0r * h))), min(H, int(round(y1 + y1r * h)))
    xa, xb = max(0, int(round(x1 + x0r * w))), min(W, int(round(x1 + x1r * w)))
    if yb - ya < 8 or xb - xa < 6:
        return None
    c = image_bgr[ya:yb, xa:xb]
    s = crop_h / c.shape[0]
    if s > 1.0:  # upscale small crops (cubic, then the model input resize goes down to 48)
        c = cv2.resize(c, (max(4, round(c.shape[1] * s)), crop_h), interpolation=cv2.INTER_CUBIC)
    return c


def lab_pixels(crop_bgr: np.ndarray) -> np.ndarray:
    """(n, 3) Lab pixels (real units: L 0..100, a/b around 0) of a crop."""
    lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    lab[:, 0] *= 100.0 / 255.0
    lab[:, 1:] -= 128.0
    return lab


def best_read(reads: list[tuple[int, float]]) -> tuple[int, float]:
    """Combine the reads of one box's windows: the most frequent valid number (ties: higher confidence)."""
    valid = [(c, p) for c, p in reads if c >= 0]
    if not valid:
        return NO_READ, max((p for _, p in reads), default=0.0)
    score: dict[int, tuple[int, float]] = {}
    for c, p in valid:
        n, m = score.get(c, (0, 0.0))
        score[c] = (n + 1, max(m, p))
    c = max(score, key=lambda k: score[k])
    return c, score[c][1]


# ----------------------------------------------------------------------------------------------- pass-2 cache
@dataclass
class JerseyReads:
    code: np.ndarray  # (n_rows,) int16 number code per pass-1 row, -1 = no read / not sampled
    conf: np.ndarray  # (n_rows,) float16 read confidence (also for rejected reads), NaN = not sampled
    stats: dict = field(default_factory=dict)
    lab: np.ndarray | None = None  # (n_rows, 3) median Lab of the read's number window (jersey colour), NaN = none
    code_label: np.ndarray | None = None  # (n_rows,) reads incl. the extra windows: for number labels only

    def label_codes(self) -> np.ndarray:
        return self.code if self.code_label is None else self.code_label

    @property
    def sampled(self) -> np.ndarray:
        return np.isfinite(self.conf)


def _cache_key(data, cfg: JerseyConfig) -> str:
    sig = data.signature
    h = hashlib.sha1()
    h.update(json.dumps({"v": CACHE_VERSION, "video": sig.get("video"), "size": sig.get("video_size_bytes"),
                         "every": round(float(cfg.every_s), 4), "min_h": cfg.min_box_h, "ov": cfg.max_overlap,
                         "bm": cfg.behind_margin,
                         "win": [list(w) for w in cfg.windows], "orr": list(cfg.overlap_region),
                         "crop_h": cfg.crop_h, "max_w": cfg.max_w,
                         "dc": cfg.min_digit_conf, "rc": cfg.min_read_conf, "sc": cfg.min_single_conf,
                         "n": int(len(data.rows)), "first": list(cfg.first_windows), "fs": cfg.first_signal,
                         "tw": cfg.track_windows, "xs": cfg.extra_signal,
                         # read gating only when on: with it off the reads are the same as before (caches stay valid)
                         **({"gate": [cfg.decided_reads, cfg.decided_every_s, cfg.track_every_s, cfg.majority]}
                            if cfg.decided_reads > 0 or cfg.track_every_s > 0 else {})},
                        sort_keys=True).encode())
    h.update(np.ascontiguousarray(data.rows[:, :8], np.float32).tobytes())
    return h.hexdigest()[:20]


class _ReadSession:
    """Incremental jersey reading: frames in, reads per pass-1 row out. Used by compute_reads (frames decoded from
    the video) and by the pass-1 frame tap (frames pass 1 already decoded, see JerseyTap) - same crops, same
    batches, same order, so both give identical reads.

    add_frame(img, rr, boxes, bt, t): rr = pass-1 row indices of this frame's eligible person boxes, boxes their
    xyxy, bt their ByteTrack ids, t the sample time. finish(n_rows) -> JerseyReads."""

    def __init__(self, cfg: JerseyConfig, recognizer: Recognizer | None = None, keep_crops: dict | None = None):
        from concurrent.futures import ThreadPoolExecutor

        self.cfg = cfg
        self.t_start = time.perf_counter()
        # the ONNX session (DirectML: ~10 s to create) is built in the reading thread while decoding starts
        self.ex = ThreadPoolExecutor(max_workers=1)
        self.rec_f = self.ex.submit(self._make_recognizer, recognizer)
        self.stats = {"device": "", "frames": 0, "boxes": 0, "crops": 0, "reads": 0, "decode_s": 0.0, "ocr_s": 0.0,
                      "skipped": 0}
        # (row, crop per window - only the first windows filled while collecting -, frame, box, bt id, t): the
        # other windows are cut in process() only for boxes whose first window shows a digit (same pixels)
        self.pending: list = []
        self.first = [w for w in cfg.first_windows if 0 <= w < len(cfg.windows)] or list(range(len(cfg.windows)))
        self.rest = [w for w in range(len(cfg.windows)) if w not in self.first]
        # read gating per ByteTrack id (see JerseyConfig.decided_reads): codes read so far, decided flag, last try
        self.bt_codes: dict[int, list[int]] = {}
        self.bt_decided: set[int] = set()
        self.bt_last: dict[int, float] = {}
        self.code: dict[int, int] = {}
        self.code_label: dict[int, int] = {}
        self.conf: dict[int, float] = {}
        self.lab: dict[int, np.ndarray] = {}
        self.keep_crops = keep_crops
        self.inflight: list = []

    def gate(self, b: int, t: float) -> bool:
        """Applied in batch order inside the reading thread, so the result never depends on thread timing."""
        cfg = self.cfg
        if cfg.decided_reads <= 0 and cfg.track_every_s <= 0:
            return True
        if b < 0:
            return True
        gap = cfg.decided_every_s if b in self.bt_decided else cfg.track_every_s
        last = self.bt_last.get(b)
        if gap > 0 and last is not None and t - last < gap - 1e-6:
            self.stats["skipped"] += 1
            return False
        self.bt_last[b] = t
        return True

    def _make_recognizer(self, recognizer):
        from replay.frame_tap import GPU_LOCK

        if recognizer is not None:
            return recognizer
        with GPU_LOCK:  # session creation next to a running detector session (pass-1 tap)
            return Recognizer(device=self.cfg.device)

    def _read(self, cs, c):
        from replay.frame_tap import GPU_LOCK

        rec = self.rec_f.result()
        with GPU_LOCK:  # DirectML: no concurrent sessions (the pass-1 detector runs in another thread)
            if hasattr(rec, "read_with_signal"):
                return rec.read_with_signal(cs, c)
            return [(a, b, 1.0) for a, b in rec.read(cs, c)]  # recognizer without signal

    def _run(self, items: list[tuple[int, np.ndarray]]) -> list[tuple[int, float, float]]:
        if not items:
            return []
        t0 = time.perf_counter()
        res = self._read([c for _, c in items], self.cfg)
        self.stats["ocr_s"] += time.perf_counter() - t0
        self.stats["crops"] += len(items)
        return res

    def _process(self, batch: list) -> None:
        """OCR one batch of boxes (runs in the reading thread, batches strictly in order)."""
        cfg, first, rest = self.cfg, self.first, self.rest
        pending = [(r, cs) for r, cs, _, _, b, t in batch if self.gate(b, t)]
        frame_of = {r: (img, box) for r, _, img, box, _, _ in batch}
        bt_of = {r: b for r, _, _, _, b, _ in batch}
        if not pending:
            return
        per_row: dict[int, list[tuple[int, float]]] = {r: [] for r, _ in pending}  # centred windows
        extra_row: dict[int, list[tuple[int, float]]] = {r: [] for r, _ in pending}  # extra (label) windows
        stage1 = [(r, cs[w]) for r, cs in pending for w in first if cs[w] is not None]
        signal: dict[int, float] = {}
        for (r, _), (c, p, sg) in zip(stage1, self._run(stage1)):
            per_row[r].append((c, p))
            signal[r] = max(signal.get(r, 0.0), sg)

        def todo(r: int, w: int) -> bool:
            thr = cfg.first_signal if w < cfg.track_windows else cfg.extra_signal
            return signal.get(r, 1.0) >= thr

        for r, cs in pending:
            img, box = frame_of[r]
            for w in rest:
                if todo(r, w):
                    cs[w] = window_crop(img, box, cfg.windows[w], cfg.crop_h)
        # centred and extra crops in separate model calls: the model's attention sees the batch's padded width,
        # so mixing them would change the centred (tracking) reads
        for target, sel in ((per_row, lambda w: w < cfg.track_windows), (extra_row, lambda w: w >= cfg.track_windows)):
            stage2 = [(r, cs[w]) for r, cs in pending for w in rest if sel(w) and todo(r, w) and cs[w] is not None]
            for (r, _), (c, p, _) in zip(stage2, self._run(stage2)):
                target[r].append((c, p))
        for r, rr in per_row.items():
            if rr or extra_row[r]:
                self.code_label[r] = best_read(rr + extra_row[r])[0]
            if rr:
                c, p = best_read(rr)
                self.code[r], self.conf[r] = c, p
                self.stats["reads"] += int(c >= 0)
                b = bt_of[r]
                if c >= 0 and b >= 0 and cfg.decided_reads > 0 and b not in self.bt_decided:
                    self.bt_codes.setdefault(b, []).append(int(c))
                    if vote(self.bt_codes[b], cfg.decided_reads, cfg.majority).number >= 0:
                        self.bt_decided.add(b)
        for r, cs in pending:  # jersey colour behind a read (team check of the read, see identity_numbers)
            c = next((cs[w] for w in first if cs[w] is not None), None)
            if self.code_label.get(r, NO_READ) >= 0 and c is not None:
                self.lab[r] = np.median(lab_pixels(c), axis=0)
        self.stats["boxes"] += len(per_row)

    # decoding and OCR (one worker thread; ONNX Runtime and OpenCV release the GIL) overlap. One batch is read
    # while the next is collected; batches are processed in order, so the reads are identical to a sequential run.
    def _flush(self, wait: bool = False) -> None:
        if self.pending:
            while self.inflight:  # at most one batch in flight: bounded memory
                self.inflight.pop(0).result()
            self.inflight.append(self.ex.submit(self._process, list(self.pending)))
            self.pending.clear()
        if wait:
            while self.inflight:
                self.inflight.pop(0).result()

    def add_frame(self, img: np.ndarray, rr, boxes: np.ndarray, bt, t: float) -> None:
        cfg, first = self.cfg, self.first
        for r, box, b in zip(rr, boxes, bt):
            box = np.asarray(box, float)
            cs: list[np.ndarray | None] = [None] * len(cfg.windows)
            for w in first:
                cs[w] = window_crop(img, box, cfg.windows[w], cfg.crop_h)
            if any(cs[w] is not None for w in first):
                self.pending.append((int(r), cs, img, box, int(b), float(t)))
                if self.keep_crops is not None:
                    self.keep_crops[int(r)] = window_crop(img, box, (0.05, 0.6, 0.0, 1.0), 120)
        self.stats["frames"] += 1
        if len(self.pending) >= 2 * cfg.batch:
            self._flush()

    def finish(self, n: int) -> JerseyReads:
        try:
            self._flush(wait=True)
        finally:
            self.ex.shutdown(wait=True)
        code = np.full(n, NO_READ, np.int16)
        code_label = np.full(n, NO_READ, np.int16)
        conf = np.full(n, np.nan, np.float32)
        lab = np.full((n, 3), np.nan, np.float32)
        for d, src in ((code, self.code), (code_label, self.code_label), (conf, self.conf), (lab, self.lab)):
            for r, v in src.items():
                d[r] = v
        st = self.stats
        st["device"] = getattr(self.rec_f.result(), "device", "") if self.rec_f.done() else ""
        st["total_s"] = round(time.perf_counter() - self.t_start, 2)
        st["decode_s"] = round(st["decode_s"], 2)
        st["ocr_s"] = round(st["ocr_s"], 2)
        return JerseyReads(code, conf, st, lab, code_label)

    def abort(self) -> None:
        self.pending.clear()
        self.ex.shutdown(wait=True)


def compute_reads(data, video: str | Path, cfg: JerseyConfig, recognizer: Recognizer | None = None,
                  progress=None, keep_crops: dict | None = None) -> JerseyReads | None:
    """Decode the sampled frames sequentially and read the number of every eligible person box.

    keep_crops: optional dict filled with {row index: torso crop} (contact sheets / debugging)."""
    from replay.detection import PERSON
    from replay.team_appearance import sample_frames

    rows = data.rows
    n = len(rows)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return None
    sess = _ReadSession(cfg, recognizer, keep_crops)
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        half = 0.5 / fps
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        idx_of_k = {int(k): i for i, k in enumerate(data.k)}
        row_i = np.array([idx_of_k.get(int(k), -1) for k in rows[:, 0]], int)
        rows_at: dict[int, list[int]] = {}
        from replay.detection import activation_threshold

        thr = activation_threshold(data.signature)  # Task-9 low-score crumbs are never read
        for r in np.flatnonzero((rows[:, 6] == PERSON) & (rows[:, 5] >= thr)):
            rows_at.setdefault(int(row_i[r]), []).append(int(r))
        want = []
        for i in sample_frames(data.t, cfg.every_s):
            rr = rows_at.get(int(i))
            if rr and eligible(rows[rr, 1:5], size, cfg).any():
                want.append(int(i))
        if want:
            start = float(data.signature.get("start_s") or 0.0)
            seek_to = max(0.0, min(float(data.t[want[0]]), start) - 2.0)
            if seek_to > 0:
                cap.set(cv2.CAP_PROP_POS_MSEC, seek_to * 1000.0)
        j = 0
        td = time.perf_counter()
        while j < len(want):
            if not cap.grab():
                break
            ft = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if ft + half < float(data.t[want[j]]):
                continue
            while j < len(want) and float(data.t[want[j]]) + half < ft:  # target missed (gap in the source)
                j += 1
            if j >= len(want) or abs(ft - float(data.t[want[j]])) > half:
                continue
            ok, img = cap.retrieve()
            if ok:
                rr = np.array(rows_at[want[j]], int)
                el = rr[eligible(rows[rr, 1:5], size, cfg)]
                bt = [int(rows[r, 7]) if np.isfinite(rows[r, 7]) else -1 for r in el]
                sess.stats["decode_s"] += time.perf_counter() - td
                sess.add_frame(img, el, rows[el, 1:5], bt, float(data.t[want[j]]))
                td = time.perf_counter()
            j += 1
            if progress is not None and j % 50 == 0:
                progress(j / len(want))
        sess.stats["decode_s"] += time.perf_counter() - td
    except BaseException:
        cap.release()
        sess.abort()
        raise
    cap.release()
    return sess.finish(n)


def effective_config(cfg: JerseyConfig | None = None) -> JerseyConfig:
    """The settings pass 2 actually reads with (CPU fallback: sparser sampling)."""
    cfg = cfg or default_config()
    if not _gpu_available(cfg.device) and cfg.cpu_every_s > cfg.every_s:
        from dataclasses import replace

        cfg = replace(cfg, every_s=cfg.cpu_every_s)
    return cfg


def save_reads(work: Path, key: str, reads: JerseyReads) -> None:
    """Write the cache file pass 2 reads (work/jersey_numbers.npz), atomically."""
    tmp = Path(work) / "jersey_numbers.tmp.npz"
    np.savez_compressed(tmp, key=np.array(key), code=reads.code, conf=reads.conf.astype(np.float16),
                        stats=np.array(json.dumps(reads.stats)), lab=reads.lab.astype(np.float16),
                        code_label=reads.label_codes())
    os.replace(tmp, Path(work) / "jersey_numbers.npz")


class JerseyTap:
    """Pass-1 frame tap (replay.frame_tap): reads jersey numbers from the frames pass 1 decodes and writes
    work/jersey_numbers.npz exactly as pass 2 would (same frames, crops, batches -> same reads); pass 2 then finds
    the cache and does not decode the video again. Off for resumed pass-1 runs (earlier frames not seen)."""

    name = "jersey"
    queue_size = 16
    blocking = True

    def __init__(self, cfg: JerseyConfig | None = None):
        self.cfg0 = cfg

    def start(self, info) -> None:
        from replay.frame_tap import FrameTap  # noqa: F401 - interface documentation

        self.cfg = effective_config(self.cfg0)
        self.active = bool(self.cfg.enabled and not info.resumed and MODEL_PATH.exists())
        self.size = tuple(info.frame_size)
        self.sess = _ReadSession(self.cfg) if self.active else None
        self.last_t = -np.inf  # team_appearance.sample_frames grid, incremental

    def wants(self, k: int, t: float) -> bool:
        if not self.active or t - self.last_t < self.cfg.every_s - 1e-6:
            return False
        self.last_t = t  # same greedy grid as sample_frames(data.t, every_s)
        return True

    def process(self, f) -> None:
        n = len(f.persons)
        if not n:
            return
        el = np.flatnonzero(eligible(np.asarray(f.persons, float).reshape(-1, 4), self.size, self.cfg))
        if not len(el):
            return
        self.sess.add_frame(f.image, f.row0 + el, np.asarray(f.persons, float)[el],
                            [int(b) for b in np.asarray(f.bytetrack_ids)[el]], float(f.t))

    def finish(self, work: Path, completed: bool) -> None:
        if not self.active:
            return
        if not completed:
            self.sess.abort()
            return
        from replay.pipeline import load_pass1

        data = load_pass1(Path(work))
        reads = self.sess.finish(len(data.rows))
        reads.stats["source"] = "pass1-tap"
        save_reads(work, _cache_key(data, self.cfg), reads)


def _gpu_available(device: str) -> bool:
    if device == "cpu":
        return False
    try:
        import onnxruntime as ort

        return any(p in ort.get_available_providers() for p in ("DmlExecutionProvider", "CUDAExecutionProvider"))
    except Exception:  # noqa: BLE001
        return False


def load_reads(work: Path) -> JerseyReads | None:
    """The cached reads of a results/work folder (whatever settings produced them); None when there are none."""
    path = Path(work) / "jersey_numbers.npz"
    if not path.exists():
        return None
    try:
        with np.load(path) as z:  # closed at once: Windows cannot replace an open cache file
            lab = z["lab"].astype(np.float32) if "lab" in z.files else None
            cl = z["code_label"].astype(np.int16) if "code_label" in z.files else None
            return JerseyReads(z["code"].astype(np.int16), z["conf"].astype(np.float32), json.loads(str(z["stats"])),
                               lab, cl)
    except Exception:  # noqa: BLE001 - corrupt cache
        return None


def load_or_compute(work: Path, data, video: str | Path | None, cfg: JerseyConfig | None = None
                    ) -> JerseyReads | None:
    """Cached reads for every pass-1 row; None when disabled, or the video/model is unavailable and nothing is
    cached."""
    cfg = cfg or default_config()
    if not cfg.enabled:
        return None
    if not _gpu_available(cfg.device) and cfg.cpu_every_s > cfg.every_s:
        from dataclasses import replace

        cfg = replace(cfg, every_s=cfg.cpu_every_s)
    work = Path(work)
    path = work / "jersey_numbers.npz"
    key = _cache_key(data, cfg)
    if path.exists():
        try:
            with np.load(path) as z:
                if str(z["key"]) == key:
                    return JerseyReads(z["code"].astype(np.int16), z["conf"].astype(np.float32),
                                       json.loads(str(z["stats"])), z["lab"].astype(np.float32),
                                       z["code_label"].astype(np.int16) if "code_label" in z.files else None)
        except Exception:  # noqa: BLE001 - corrupt/old cache: recompute
            pass
    from replay.team_appearance import resolve_video

    vid = resolve_video(data, video)
    if vid is None or not MODEL_PATH.exists():
        return None
    try:
        reads = compute_reads(data, vid, cfg)
    except Exception as e:  # noqa: BLE001 - numbers are optional: never fail pass 2 because of them
        print("jersey numbers: skipped:", e)
        return None
    if reads is None:
        return None
    try:
        tmp = work / "jersey_numbers.tmp.npz"
        np.savez_compressed(tmp, key=np.array(key), code=reads.code, conf=reads.conf.astype(np.float16),
                            stats=np.array(json.dumps(reads.stats)), lab=reads.lab.astype(np.float16),
                            code_label=reads.label_codes())
        os.replace(tmp, path)
    except OSError:
        pass
    return reads


# ----------------------------------------------------------------------------------------------- voting
@dataclass
class Vote:
    number: int  # code, -1 = none accepted
    support: int  # reads of the accepted (or leading) number
    against: int  # contradicting reads
    reads: int  # all reads
    counts: dict  # code -> count

    @property
    def text(self) -> str | None:
        return text_of(self.number)


def _against(counts: dict[int, int], cand: int) -> int:
    """Reads contradicting ``cand``. One-digit reads that are part of a two-digit ``cand`` do not count (a digit
    was hidden); for a one-digit ``cand``, two-digit reads containing it count twice (the usual error is a
    missed digit, not an invented one, so "24" reads make a "4" doubtful). A one-digit read that also fits
    another two-digit number read on the track ("1" with "19" and "21") is ambiguous and counts against."""
    out = 0
    others = [v for v in counts if v >= 10 and v != cand]
    for v, m in counts.items():
        if v == cand or (is_partial(v, cand) and not any(is_partial(v, o) for o in others)):
            continue
        out += 2 * m if is_partial(cand, v) else m
    return out


def vote(codes, min_reads: int = 3, majority: float = 0.66) -> Vote:
    """Accept a number only when read consistently (see module docstring)."""
    c = np.asarray(codes, int).reshape(-1)
    c = c[c >= 0]
    counts: dict[int, int] = {}
    for v in c.tolist():
        counts[v] = counts.get(v, 0) + 1
    if not counts:
        return Vote(NO_READ, 0, 0, 0, {})
    best, best_key = NO_READ, None
    for cand, sup in counts.items():
        key = (sup - _against(counts, cand), sup, len(text_of(cand) or ""))  # ties: the two-digit number
        if best_key is None or key > best_key:
            best, best_key = cand, key
    sup = counts[best]
    against = _against(counts, best)
    ok = sup >= min_reads and sup >= majority * (sup + against)
    if best < 10:  # one digit: two or more reads of a two-digit number containing it -> it is that number, cut
        ok &= sum(m for v, m in counts.items() if is_partial(best, v)) < 2
    return Vote(best if ok else NO_READ, sup, against, int(len(c)), counts)


def vote_groups(group: np.ndarray, codes: np.ndarray, min_reads: int = 3, majority: float = 0.66
                ) -> dict[int, Vote]:
    """vote() per group id (e.g. track id per detection); groups <= 0 are ignored."""
    group = np.asarray(group, int)
    codes = np.asarray(codes, int)
    out = {}
    for g in np.unique(group[(group > 0) & (codes >= 0)]):
        out[int(g)] = vote(codes[group == g], min_reads, majority)
    return out


def resolve_conflicts(votes: dict[int, Vote], team: dict[int, str], present: dict[int, np.ndarray],
                      teams: tuple = ("A", "B"), min_shared: int = 3) -> dict[int, Vote]:
    """One number per player per team at a time: tracks of one team that carry the same number and are present
    together (at least ``min_shared`` common samples) -> the one with more votes (support - against, then
    support) keeps it, the others lose it. Tracks at different times may share a number (the same player under
    two track ids). present: track -> sample indices where it is detected."""
    out = dict(votes)
    by_key: dict[tuple[str, int], list[int]] = {}
    for tid, v in votes.items():
        if v.number >= 0 and team.get(tid) in teams:
            by_key.setdefault((team[tid], v.number), []).append(tid)
    for _, tids in by_key.items():
        if len(tids) < 2:
            continue
        tids.sort(key=lambda t: (-(votes[t].support - votes[t].against), -votes[t].support, t))
        kept: list[int] = []
        for t in tids:
            mine = np.asarray(present.get(t, ()), int)
            if any(len(np.intersect1d(mine, np.asarray(present.get(k, ()), int))) >= min_shared for k in kept):
                v = votes[t]
                out[t] = Vote(NO_READ, v.support, v.against, v.reads, v.counts)
            else:
                kept.append(t)
    return out


def team_colour_ok(read_lab: np.ndarray, team: list, team_lab: dict, margin: float = 8.0) -> np.ndarray:
    """Per read: False when the jersey colour behind the number is clearly the OTHER team's (the read belongs to
    an opponent the track's box was on for a moment). Compared in the a/b plane; unknown colours pass."""
    read_lab = np.asarray(read_lab, float).reshape(-1, 3)
    ok = np.ones(len(read_lab), bool)
    labs = {k: np.asarray(v, float)[:3] for k, v in (team_lab or {}).items() if k in ("A", "B") and v is not None}
    if len(labs) < 2:
        return ok
    for i, (lab, t) in enumerate(zip(read_lab, team)):
        if t not in labs or not np.isfinite(lab).all():
            continue
        other = "B" if t == "A" else "A"
        d_own = float(np.linalg.norm(lab[1:] - labs[t][1:]))
        d_other = float(np.linalg.norm(lab[1:] - labs[other][1:]))
        ok[i] = not d_other + margin < d_own
    return ok


SINGLE_DIGIT_STRONG = 8  # reads a one-digit number needs when a team-mate wears a two-digit number containing it


def identity_numbers(tid: np.ndarray, codes: np.ndarray, k: np.ndarray, team_of, cfg: JerseyConfig | None = None,
                     teams: tuple = ("A", "B"), read_lab: np.ndarray | None = None,
                     team_lab: dict | None = None) -> dict[int, Vote]:
    """Accepted number per player identity (tid > 0, team in ``teams``) after voting and same-team uniqueness.
    Referees / unknown tracks get none (a referee's box often holds a player's number nearby).

    tid, codes, k: per detection (identity, read code, sample index); team_of(tid) -> team label.
    read_lab (per detection, JerseyReads.lab) + team_lab (team -> jersey Lab, TeamResult.centroids): reads on
    the other team's jersey colour are ignored. A one-digit number that is part of a team-mate's two-digit number
    ("3" of "36", seen half hidden) needs SINGLE_DIGIT_STRONG reads."""
    cfg = cfg or default_config()
    tid = np.asarray(tid, int)
    codes = np.asarray(codes, int).copy()
    k = np.asarray(k, int)
    if read_lab is not None and team_lab:
        has = codes >= 0
        ok = team_colour_ok(np.asarray(read_lab)[has], [team_of(g) for g in tid[has]], team_lab)
        codes[np.flatnonzero(has)[~ok]] = NO_READ
    votes = {g: v for g, v in vote_groups(tid, codes, cfg.min_reads, cfg.majority).items() if team_of(g) in teams}
    present = {g: np.unique(k[tid == g]) for g in votes}
    out = resolve_conflicts(votes, {g: team_of(g) for g in votes}, present, teams)
    two = {(team_of(g), v.number) for g, v in out.items() if v.number >= 10}
    for g, v in out.items():
        mate_has_it = any(tm == team_of(g) and is_partial(v.number, n) for tm, n in two)
        if 0 <= v.number < 10 and (not cfg.allow_single_digit or (v.support < SINGLE_DIGIT_STRONG and mate_has_it)):
            out[g] = Vote(NO_READ, v.support, v.against, v.reads, v.counts)
    if cfg.anchor_reads > 0:
        _anchor(out, votes, team_of, present, cfg)
    return out


def _anchor(out: dict[int, Vote], votes: dict[int, Vote], team_of, present: dict, cfg: JerseyConfig) -> None:
    """Second pass: a track with only a few reads (fewer than min_reads) takes a two-digit number that another
    track of its team ALREADY carries (accepted by the full vote) when all its reads agree with it and it is
    never on court together with a carrier of that number (min 3 shared samples): it is that player under
    another track id. A lone wrong read rarely satisfies all three (number on the team, no contradiction,
    time-exclusive)."""
    anchors: dict[tuple[str, int], list[int]] = {}
    for g, v in out.items():
        if v.number >= 10:
            anchors.setdefault((team_of(g), v.number), []).append(g)
    for g, v in votes.items():
        if out[g].number >= 0 or not v.counts:
            continue
        cand = [c for c in v.counts if c >= 10]
        if len(cand) != 1 or v.counts[cand[0]] < cfg.anchor_reads:
            continue
        c = cand[0]
        if _against(v.counts, c) > 0 or (team_of(g), c) not in anchors:
            continue
        mine = present.get(g, np.zeros(0, int))
        holders = [h for h, w in out.items() if w.number == c and team_of(h) == team_of(g)]
        if any(len(np.intersect1d(mine, present.get(h, np.zeros(0, int)))) >= 3 for h in holders):
            continue
        out[g] = Vote(c, v.counts[c], 0, v.reads, v.counts)


def track_info_fields(v: Vote | None) -> dict:
    """Fields added to a track's track_info (tracking.json "tracks", replay_data.json "track_info")."""
    if v is None or v.number < 0:
        return {}
    return {"number": v.text, "number_reads": int(v.support)}


# ----------------------------------------------------------------------------------------------- identity linking
LINK_MIN_READS = 2  # a piece "has" a one-digit number read at least this often (two-digit numbers: once;
# wrong two-digit reads are rare, one-digit ones come from stripes, arms and partly hidden numbers)
LINK_MATCH_BONUS = 3.0  # cost bonus for linking two pieces with the same single number
LINK_CONFLICT_COST = 8.0  # one read on the weaker side: +8 ...
LINK_VETO_READS = 2  # ... and no link at all when both sides read their number at least this often


def piece_numbers(codes, min_reads: int = LINK_MIN_READS) -> dict[int, int]:
    """{number code: reads} of the numbers on one tracklet (one-digit ones read at least ``min_reads`` times)."""
    c = np.asarray(codes, int)
    c = c[c >= 0]
    vals, cnt = np.unique(c, return_counts=True)
    return {int(v): int(m) for v, m in zip(vals, cnt) if m >= min_reads or v >= 10}


def _compatible(a: int, b: int) -> bool:
    return a == b or is_partial(a, b) or is_partial(b, a)


def link_cost(na: dict[int, int], nb: dict[int, int]) -> float | None:
    """Cost term for linking two tracklets with number reads na, nb (piece_numbers): negative = same player,
    positive = different numbers, None = impossible (both numbers read consistently and they differ).

    Reads are very precise (wrong reads rarely repeat), but a number is only visible from some angles, so the
    absence of a number says nothing. A piece that already holds two numbers (two players) never earns a bonus.
    """
    if not na or not nb:
        return 0.0
    ta = {v: m for v, m in na.items() if v >= 10}
    tb = {v: m for v, m in nb.items() if v >= 10}
    if ta and tb:  # two-digit numbers on both sides decide (a one-digit "4" fits both "24" and "43")
        if set(ta) & set(tb):
            return -LINK_MATCH_BONUS if len(ta) == 1 and set(ta) == set(tb) else 0.0
        weaker = min(max(ta.values()), max(tb.values()))
        return None if weaker >= LINK_VETO_READS else LINK_CONFLICT_COST * weaker
    if any(_compatible(a, b) for a in na for b in nb):
        if len(na) == 1 and len(nb) == 1 and next(iter(na)) == next(iter(nb)):
            return -LINK_MATCH_BONUS
        return 0.0
    weaker = min(max(na.values()), max(nb.values()))
    if weaker >= LINK_VETO_READS:
        return None
    return LINK_CONFLICT_COST * weaker
