"""OPT-IN court keypoints from Google Gemini for hard footage (broadcast cuts, zoom) - same interface as the court
detector (courtnet): per frame {keypoint index: (x_px, y_px, score)} -> court choice -> per-frame homography (RANSAC)
-> located frames for autocourt (AutoResult.frames = [{"t", "i", "H", "source"}], the calibrate_net format).

Privacy: only when the user ticks "Use Gemini to find the court" in the calibration step. At most MAX_FRAMES video
frames, downscaled to JPEG_MAX_W px, are sent to Google with the user's own AI Studio key (read at run time from the
app settings; never stored in project files, exports or logs).

Wallet safety (hard limits, not just an estimate):
  * cheapest model by default (gemini-2.5-flash-lite), thinking off, output capped;
  * ONE request per frame, at most MAX_RETRIES retry on a network error, never on an API error;
  * per game: <= MAX_FRAMES frames and <= the per-game cap (default 0.10 EUR); per calendar month: <= the monthly
    cap (default 2 EUR) - checked BEFORE every request with a conservative estimate, the real usage reported by the
    API is booked after it; a request that could cross a cap is not sent ("Gemini budget reached");
  * local ledger (date / project / frames / tokens / EUR) in the user's app-data folder (Ledger);
  * per-project cache work/gemini_court.json: frames already answered are never sent again (rebuilds are free).

Steps: select_frames (one per camera shot + one per SHOT_EVERY_S within a shot when the view changed) -> ask (JSON
[{name, x, y, confidence}], 0-1000 coordinates, visible points only) -> courtnet.choose_court + frame_H (RANSAC)
-> line_support check on the frame's own painted lines (rejects H that do not explain them).
"""
from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np

DEFAULT_MODEL = "gemini-2.5-flash-lite"  # cheapest current model (see PRICES); "gemini-2.5-flash" selectable
MAX_FRAMES = 60  # hard cap per game
SHOT_EVERY_S = 10.0
MIN_NEW_VIEW_PX = 0.25  # a pick within a shot needs this much camera motion (fraction of the width) since the last
MIN_CONF = 0.4
MIN_INLIERS = 5
MIN_LINE_SUPPORT = 0.6  # (0.45 let a visibly wrong NCAA fit through at 0.53) share of projected court-line samples (inside the image) that land on line pixels
MIN_LINE_SAMPLES = 30  # ... and at least this many projected samples inside the image (calibration agent's rule)
LINE_DILATE_PX = 3  # thin tolerance (a 7-px one let random lines "hit" dense crowd / graphics edges)
JPEG_MAX_W = 768  # frames are downscaled to this width before they leave the PC
MAX_RETRIES = 1  # one retry on a network error / timeout, never on an API error (4xx: quota, key, blocked)
# Paid-tier list prices, USD per 1M tokens, from https://ai.google.dev/gemini-api/docs/pricing - CHECKED ON 2026-10-18:
#   gemini-2.5-flash-lite: input (text/image/video) 0.10, output (incl. thinking) 0.40
#   gemini-2.5-flash:      input (text/image/video) 0.30, output (incl. thinking) 2.50
# (the free tier is free of charge; the caps assume the paid tier, i.e. the worst case)
PRICES = {"gemini-2.5-flash-lite": (0.10, 0.40), "gemini-2.5-flash": (0.30, 2.50)}
USD_TO_EUR = 1.0  # deliberately conservative: 1 USD is counted as 1 EUR (a dollar is worth less than a euro)
# conservative pre-request token ESTIMATE (the ledger books the real usage the API reports)
TOKENS_IN_IMAGE = 600  # a 768-px frame is about one image tile; counted generously
TOKENS_IN_TEXT = 900  # instructions + keypoint definitions
TOKENS_OUT = 1024  # = maxOutputTokens: the answer can never be longer
GAME_CAP_EUR = 0.10  # default hard cap per game (app setting)
MONTH_CAP_EUR = 2.00  # default hard cap per calendar month (app setting)
CACHE_NAME = "gemini_court.json"  # per project (work folder): answers per video frame - rebuilds never re-send

SIDE = {"L": "the court end on the LEFT side of the image (as seen from the camera)",
        "R": "the court end on the RIGHT side of the image (as seen from the camera)"}
DEFS = {
    "corner_near": "corner where the baseline of {end} meets the NEAR sideline (closest to the camera)",
    "corner_far": "corner where the baseline of {end} meets the FAR sideline",
    "three_base_near": "where the three-point line meets the baseline of {end}, near-sideline side",
    "three_base_far": "where the three-point line meets the baseline of {end}, far-sideline side",
    "paint_base_near": "where the painted lane (key) side line meets the baseline of {end}, near side",
    "paint_base_far": "where the painted lane (key) side line meets the baseline of {end}, far side",
    "ft_near": "near corner of the free-throw line of {end} (free-throw line meets the lane line, near side)",
    "ft_far": "far corner of the free-throw line of {end} (free-throw line meets the lane line, far side)",
    "ft_circle_top": "point of the free-throw circle farthest from the baseline of {end} (towards half court)",
    "three_break_near": "where the straight part of the three-point line of {end} turns into the arc, near side",
    "three_break_far": "where the straight part of the three-point line of {end} turns into the arc, far side",
    "three_top": "top of the three-point arc of {end} (farthest from its baseline, on the court's long axis)",
}
CENTRE = {"mid_near": "intersection of the NEAR sideline and the half-court line",
          "mid_far": "intersection of the FAR sideline and the half-court line",
          "center_circle_near": "where the centre circle crosses the half-court line on the near side",
          "center_circle_far": "where the centre circle crosses the half-court line on the far side"}

SYSTEM = ("You locate basketball court markings in a video frame. Return ONLY points you can actually see on the "
          "floor (painted line intersections), never guess hidden ones. If you are unsure about a point, OMIT it - "
          "an empty list is a good answer. Coordinates are normalised to 0-1000 (x from the left edge, y from the "
          "top edge of the image). Confidence 0-1.")
# the easy, unambiguous points only (corners, half-court line ends, lane corners on the baseline and the free-throw line)
ASK_POINTS = [f"{e}_{k}" for e in ("L", "R") for k in ("corner_near", "corner_far", "paint_base_near", "paint_base_far",
                                                        "ft_near", "ft_far")] + ["mid_near", "mid_far"]
SCHEMA = {"type": "object", "properties": {"points": {"type": "array", "items": {
    "type": "object", "properties": {"name": {"type": "string"}, "x": {"type": "number"}, "y": {"type": "number"},
                                     "confidence": {"type": "number"}},
    "required": ["name", "x", "y", "confidence"]}}}, "required": ["points"]}


def keypoint_definitions(names: list[str]) -> dict[str, str]:
    out = {}
    for n in names:
        if n in CENTRE:
            out[n] = CENTRE[n]
        elif n[:2] in ("L_", "R_") and n[2:] in DEFS:
            out[n] = DEFS[n[2:]].format(end=SIDE[n[0]])
    return out


def prompt(names: list[str]) -> str:
    lines = "\n".join(f"- {n}: {t}" for n, t in keypoint_definitions([n for n in names if n in ASK_POINTS]).items())
    return ("Court keypoints (the camera stands beside the NEAR sideline; LEFT/RIGHT as seen in the image):\n"
            f"{lines}\nReturn the visible ones as JSON {{\"points\": [{{name, x, y, confidence}}]}}.")


# ----------------------------------------------------------------------------------------------- cost
def token_cost_eur(model: str, tokens_in: int, tokens_out: int) -> float:
    pin, pout = PRICES.get(model, max(PRICES.values()))  # unknown model: the most expensive known price
    return (tokens_in * pin + tokens_out * pout) / 1e6 * USD_TO_EUR


def request_estimate_eur(model: str = DEFAULT_MODEL) -> float:
    """Conservative cost of ONE frame request (before it is sent)."""
    return token_cost_eur(model, TOKENS_IN_IMAGE + TOKENS_IN_TEXT, TOKENS_OUT)


def estimate_cost(n_frames: int, model: str = DEFAULT_MODEL) -> float:
    """Conservative EUR for n frames (shown in the GUI as 'at most about X EUR')."""
    return n_frames * request_estimate_eur(model)


class BudgetReached(Exception):
    """Gemini budget reached (frames per game, EUR per game or per month): nothing more is sent."""


def default_ledger_path() -> Path:
    base = os.environ.get("APPDATA") or str(Path.home() / ".config")
    return Path(base) / "AdvancedAnalytist" / "gemini_ledger.json"


class Ledger:
    """Local spending ledger + caps (JSON in the user's app-data folder, never in project files):
    {"game_cap_eur", "month_cap_eur", "entries": [{"date", "month", "project", "frames", "tokens_in", "tokens_out",
    "eur", "model"}]}."""

    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else default_ledger_path()
        self.doc = {"game_cap_eur": GAME_CAP_EUR, "month_cap_eur": MONTH_CAP_EUR, "entries": []}
        try:
            self.doc.update(json.loads(self.path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            pass

    @property
    def game_cap(self) -> float:
        return float(self.doc.get("game_cap_eur", GAME_CAP_EUR))

    @property
    def month_cap(self) -> float:
        return float(self.doc.get("month_cap_eur", MONTH_CAP_EUR))

    def set_caps(self, game_eur: float | None = None, month_eur: float | None = None) -> None:
        if game_eur is not None:
            self.doc["game_cap_eur"] = max(0.0, float(game_eur))
        if month_eur is not None:
            self.doc["month_cap_eur"] = max(0.0, float(month_eur))
        self.save()

    def month_spent(self, month: str | None = None) -> float:
        month = month or time.strftime("%Y-%m")
        return float(sum(e.get("eur", 0.0) for e in self.doc["entries"] if e.get("month") == month))

    def add(self, project: str, frames: int, tokens_in: int, tokens_out: int, eur: float, model: str) -> None:
        try:  # the file may have changed since this ledger was read (caps set in the app, another game's entry)
            disk = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(disk, dict) and isinstance(disk.get("entries"), list):
                self.doc.update(disk)
        except (OSError, ValueError):
            pass
        self.doc["entries"].append({"date": time.strftime("%Y-%m-%d %H:%M"), "month": time.strftime("%Y-%m"),
                                    "project": project, "frames": int(frames), "tokens_in": int(tokens_in),
                                    "tokens_out": int(tokens_out), "eur": round(float(eur), 6), "model": model})
        self.save()

    def save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.doc, indent=1), encoding="utf-8")
            tmp.replace(self.path)
        except OSError:
            pass

    def summary(self) -> str:
        return (f"This month {self.month_spent():.3f} of {self.month_cap:.2f} EUR used; per game at most "
                f"{self.game_cap:.2f} EUR")

    def last_entries(self, n: int = 5) -> list[dict]:
        return list(self.doc["entries"][-n:])


class Spender:
    """Hard caps for one game: refuses a request whose conservative estimate could push this game's or this month's
    spending over its cap; books the real usage after each request; one ledger entry at the end."""

    def __init__(self, ledger: Ledger, model: str, project: str, max_frames: int = MAX_FRAMES):
        self.ledger, self.model, self.project, self.max_frames = ledger, model, project, min(max_frames, MAX_FRAMES)
        self.eur = 0.0
        self.frames = self.tin = self.tout = 0
        self.month0 = ledger.month_spent()

    def allow(self) -> None:
        nxt = request_estimate_eur(self.model)
        if self.frames >= self.max_frames:
            raise BudgetReached(f"Gemini frame limit reached ({self.max_frames} per game)")
        if self.eur + nxt > self.ledger.game_cap:
            raise BudgetReached(f"Gemini budget reached for this game ({self.ledger.game_cap:.2f} EUR)")
        if self.month0 + self.eur + nxt > self.ledger.month_cap:
            raise BudgetReached(f"Gemini budget reached for this month ({self.ledger.month_cap:.2f} EUR)")

    def book(self, usage: dict | None) -> None:
        u = usage or {}
        tin = int(u.get("promptTokenCount") or TOKENS_IN_IMAGE + TOKENS_IN_TEXT)
        tout = int((u.get("candidatesTokenCount") or 0) + (u.get("thoughtsTokenCount") or 0)) or TOKENS_OUT
        self.frames += 1
        self.tin += tin
        self.tout += tout
        self.eur += token_cost_eur(self.model, tin, tout)

    def close(self) -> None:
        if self.frames:
            self.ledger.add(self.project, self.frames, self.tin, self.tout, self.eur, self.model)


# ----------------------------------------------------------------------------------------------- frames + client
def select_frames(data, size: tuple[int, int], max_frames: int = MAX_FRAMES, every_s: float = SHOT_EVERY_S) -> list[int]:
    """Sample indices: the first sample of every camera shot (motion chain break) + one per every_s inside a shot
    when the camera has moved at least MIN_NEW_VIEW_PX of the width since the last pick; only samples whose pass-1
    person boxes look like a wide view (wide_by_boxes, when rows are known). Evenly thinned to max."""
    n = len(data.t)
    if not n:
        return []
    if getattr(data, "rows", None) is not None:
        ok = [i for i in range(n) if wide_by_boxes(data, i, size)]
        if ok:
            sub = type("D", (), {})()
            sub.t, sub.frame = np.asarray(data.t)[ok], np.asarray(data.frame)[ok]
            sub.motion = [None if j == 0 or ok[j] != ok[j - 1] + 1 else data.motion[ok[j]] for j in range(len(ok))]                 if data.motion is not None else None
            return [ok[j] for j in select_frames(sub, size, max_frames, every_s)]
    picks, last_t, moved = [0], float(data.t[0]), 0.0
    c = np.array([size[0] / 2, size[1] / 2, 1.0])
    for i in range(1, n):
        m = data.motion[i] if data.motion is not None and i < len(data.motion) else None
        if m is None or not np.isfinite(np.asarray(m, float)).all():  # cut / lost tracking: new shot
            picks.append(i)
            last_t, moved = float(data.t[i]), 0.0
            continue
        q = np.asarray(m, float) @ c
        moved += float(np.hypot(*(q[:2] / q[2] - c[:2]))) / size[0]
        if data.t[i] - last_t >= every_s and moved >= MIN_NEW_VIEW_PX:
            picks.append(i)
            last_t, moved = float(data.t[i]), 0.0
    max_frames = min(max_frames, MAX_FRAMES)
    if len(picks) > max_frames:
        picks = [picks[int(j)] for j in np.linspace(0, len(picks) - 1, max_frames).round()]
    return picks


WIDE_MAX_BOX_H = 0.25  # median person box height / frame height (close-ups have big people)
WIDE_MIN_PEOPLE = 5
WIDE_MIN_SURE_PEOPLE = 6  # boxes with detector score > 0.5 (NCAA: wide views 7-9, close-ups / crowd 2-4)
WIDE_MIN_LONG_LINES = 4  # LSD segments longer than WIDE_LINE_LEN of the width in the lower 2/3 of the image
WIDE_LINE_LEN = 0.12
WIDE_MIN_FLOOR = 0.25  # share of the lower half of the image close to the floor colour
WIDE_MIN_SMOOTH = 0.35  # share of 16-px tiles in the lower half that are smooth (floor), not crowd / faces / clothes


def _person_rows_by_k(data) -> dict:
    """{sample k: its pass-1 person rows}, built once per data object (one pass over all rows)."""
    cache = getattr(data, "_gemini_rows_by_k", None)
    if cache is None:
        p = data.rows[data.rows[:, 6] == 0]
        p = p[np.argsort(p[:, 0], kind="stable")]
        ks, start = np.unique(p[:, 0], return_index=True)
        cache = dict(zip(ks.astype(int).tolist(), np.split(p, start[1:])))
        try:
            data._gemini_rows_by_k = cache
        except AttributeError:  # read-only object: no caching
            pass
    return cache


def wide_by_boxes(data, i: int, size: tuple[int, int]) -> bool:
    """Cheap pre-filter from the pass-1 person boxes of sample i: several people, none huge."""
    rows = _person_rows_by_k(data).get(int(data.k[i]), np.zeros((0, data.rows.shape[1])))
    if len(rows) < WIDE_MIN_PEOPLE or int((rows[:, 5] > 0.5).sum()) < WIDE_MIN_SURE_PEOPLE:
        return False  # wide game views show many clearly detected players; close-ups / crowd shots do not
    return float(np.median(rows[:, 4] - rows[:, 2])) / size[1] <= WIDE_MAX_BOX_H


def wide_court_view(img: np.ndarray, boxes: np.ndarray | None = None) -> bool:
    """A wide view of the court (worth sending): long straight floor lines and a large visible floor area."""
    from replay import autocourt
    from replay.team_appearance import floor_lab, lab_float

    H, W = img.shape[:2]
    Ls, ln = autocourt.line_segments(img, min_len=WIDE_LINE_LEN * W)
    low = (Ls[:, 1] + Ls[:, 3]) / 2 > H / 3 if len(Ls) else np.zeros(0, bool)
    if int(low.sum()) < WIDE_MIN_LONG_LINES:
        return False
    g = cv2.cvtColor(img[H // 2:], cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (W // 2, (H - H // 2) // 2), interpolation=cv2.INTER_AREA)
    lap = np.abs(cv2.Laplacian(g.astype(np.float32), cv2.CV_32F))
    th, tw = g.shape[0] // 8, g.shape[1] // 8
    tiles = lap[:th * 8, :tw * 8].reshape(th, 8, tw, 8).mean(axis=(1, 3))
    if float((tiles < 6.0).mean()) < WIDE_MIN_SMOOTH:
        return False
    fl = floor_lab(img, boxes) if boxes is not None and len(boxes) else None
    if fl is None:
        return True
    small = cv2.resize(img[H // 2:], (160, 45), interpolation=cv2.INTER_AREA)
    d = np.linalg.norm((lab_float(small).reshape(-1, 3) - fl) * (0.5, 1.0, 1.0), axis=1)
    return float((d < 18.0).mean()) >= WIDE_MIN_FLOOR


def _jpeg(img: np.ndarray) -> bytes:
    h, w = img.shape[:2]
    if w > JPEG_MAX_W:
        img = cv2.resize(img, (JPEG_MAX_W, round(h * JPEG_MAX_W / w)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes() if ok else b""


class GeminiVision:
    """Image + text -> JSON with the user's key (REST, key only in the request header). Mockable: tests pass any
    object with ``model`` and ``locate(jpeg, system, user, schema) -> (text, usageMetadata dict)``."""

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL, timeout_s: float = 60.0):
        self._key, self.model, self.timeout_s = api_key, model, timeout_s

    def __repr__(self) -> str:  # never show the key
        return f"GeminiVision(model={self.model!r})"

    def locate(self, jpeg: bytes, system: str, user: str, schema: dict) -> tuple[str, dict]:
        import urllib.request

        from replay.ai_analysis import API_BASE, response_text

        body = {"systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [
                    {"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(jpeg).decode("ascii")}},
                    {"text": user}]}],
                "generationConfig": {"responseMimeType": "application/json", "responseSchema": schema,
                                     "temperature": 0.0, "maxOutputTokens": TOKENS_OUT,
                                     "thinkingConfig": {"thinkingBudget": 0}}}
        req = urllib.request.Request(f"{API_BASE}/models/{self.model}:generateContent",
                                     data=json.dumps(body).encode("utf-8"), method="POST",
                                     headers={"Content-Type": "application/json", "x-goog-api-key": self._key})
        with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
            data = json.loads(r.read().decode("utf-8"))
        return response_text(data), data.get("usageMetadata") or {}


def ask(vision, jpeg: bytes, names: list[str]) -> tuple[str, dict]:
    """ONE request; at most MAX_RETRIES retry on a network error / timeout (never on an API error)."""
    import socket
    import urllib.error

    for attempt in range(MAX_RETRIES + 1):
        try:
            return vision.locate(jpeg, SYSTEM, prompt(names), SCHEMA)
        except urllib.error.HTTPError:
            raise
        except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError):
            if attempt >= MAX_RETRIES:
                raise
    raise RuntimeError("unreachable")


def parse_points(text: str, names: list[str], size: tuple[int, int], min_conf: float = MIN_CONF) -> dict:
    """JSON answer -> {keypoint index: (x_px, y_px, confidence)} (unknown names, low-confidence points and
    out-of-range coordinates dropped)."""
    try:
        pts = json.loads(text).get("points") or []
    except (ValueError, AttributeError, TypeError):
        return {}
    idx = {n: i for i, n in enumerate(names)}
    out = {}
    for p in pts:
        try:
            n, x, y, c = str(p["name"]), float(p["x"]), float(p["y"]), float(p.get("confidence", 0))
        except (KeyError, TypeError, ValueError):
            continue
        if n in idx and n in ASK_POINTS and c >= min_conf and 0 <= x <= 1000 and 0 <= y <= 1000:
            out[idx[n]] = (x / 1000 * size[0], y / 1000 * size[1], c)
    return out


def line_support(img: np.ndarray, H: np.ndarray, size: tuple[int, int]) -> float:
    """Share of the projected court lines (inside the image) that fall on the frame's own line pixels."""
    from replay import autocourt, court

    pts = np.asarray(autocourt.line_pixels(img))  # (N, 3) [x, y, colour class] at autocourt.WORK_W scale
    W, Hh = size
    if pts.ndim != 2 or not len(pts):
        return 0.0
    sc = W / float(autocourt.WORK_W)
    m = np.zeros((Hh, W), np.uint8)
    xy = np.round(pts[:, :2].astype(float) * sc).astype(int)
    ok = (xy[:, 0] >= 0) & (xy[:, 0] < W) & (xy[:, 1] >= 0) & (xy[:, 1] < Hh)
    m[xy[ok, 1], xy[ok, 0]] = 1
    m = cv2.dilate(m, np.ones((LINE_DILATE_PX + int(sc), LINE_DILATE_PX + int(sc)), np.uint8))  # + scale gaps
    hit = tot = 0
    for poly in court.court_polylines():
        P = np.asarray(poly, float)
        q = np.c_[P, np.ones(len(P))] @ H.T
        ok = q[:, 2] > 1e-9
        uv = q[ok, :2] / q[ok, 2:3]
        inside = (uv[:, 0] >= 0) & (uv[:, 0] < size[0]) & (uv[:, 1] >= 0) & (uv[:, 1] < size[1])
        uv = uv[inside].astype(int)
        tot += len(uv)
        hit += int(m[uv[:, 1], uv[:, 0]].sum())
    return hit / tot if tot >= MIN_LINE_SAMPLES else 0.0


# ----------------------------------------------------------------------------------------------- cache + main
def load_cache(work: Path | None) -> dict:
    try:
        return json.loads((Path(work) / CACHE_NAME).read_text(encoding="utf-8")) if work else {}
    except (OSError, ValueError):
        return {}


def save_cache(work: Path | None, doc: dict) -> None:
    if not work:
        return
    try:
        tmp = Path(work) / (CACHE_NAME + ".tmp")
        tmp.write_text(json.dumps(doc), encoding="utf-8")
        tmp.replace(Path(work) / CACHE_NAME)
    except OSError:
        pass


def keypoints_per_frame(video: Path, data, size: tuple[int, int], vision, names: list[str], progress=None,
                        cancel=None, max_frames: int = MAX_FRAMES, work: Path | None = None,
                        ledger: Ledger | None = None, project: str = "", read_frames=None):
    """{sample index: keypoint peaks} for the selected frames (cached answers first, then capped requests),
    the decoded images, the spender and the stop reason ("" = not stopped)."""
    from replay import autocourt

    prog = progress or (lambda x, m: None)
    read_frames = read_frames or autocourt.read_frames
    picks = select_frames(data, size, max_frames)
    f2i = {int(data.frame[i]): i for i in picks}
    cache = load_cache(work)
    answers = cache.setdefault("frames", {})
    spender = Spender(ledger or Ledger(), getattr(vision, "model", DEFAULT_MODEL), project, max_frames)
    pks, imgs, stop = {}, {}, ""
    try:
        for j, (fno, img) in enumerate(read_frames(video, list(f2i), cancel)):
            i = f2i[fno]
            text = answers.get(str(fno))
            if text is None and not stop and vision is not None:
                boxes = (data.rows[(data.rows[:, 0] == data.k[i]) & (data.rows[:, 6] == 0), 1:5]
                         if getattr(data, "rows", None) is not None else None)
                if not wide_court_view(img, boxes):
                    continue  # close-up / crowd shot: not worth a request
                try:
                    spender.allow()
                    text, usage = ask(vision, _jpeg(img), names)
                    spender.book(usage)
                    answers[str(fno)] = text
                    save_cache(work, cache)
                except BudgetReached as e:
                    stop = str(e)
                except Exception as e:  # noqa: BLE001 - network / API error: skip the frame, never leak the key
                    print(f"Gemini court: frame {fno} skipped ({type(e).__name__})")
            if text is None:
                continue
            pks[i] = parse_points(text, names, size)
            imgs[i] = img
            prog(0.9 * (j + 1) / max(1, len(picks)), f"Asking Gemini for the court: frame {j + 1} / {len(picks)}")
    finally:
        spender.close()
    if stop:
        print(f"{stop} - no more frames sent")
    return pks, imgs, spender, stop


def api_key_from_app_settings() -> str:
    """The user's AI Studio key from the app settings (same storage as the AI dialog); "" when none. Never logged."""
    try:
        from PySide6.QtCore import QSettings

        from replay.gui.ai_dialog import load_api_key

        return load_api_key(QSettings("AdvancedAnalytist", "AdvancedAnalytist"))
    except Exception:  # noqa: BLE001 - no GUI libraries / settings: no key
        return ""


def for_project(p, data, size, progress=None, cancel=None):
    """Opt-in entry for the calibration job: None unless the project ticked "gemini_court" and a key exists.
    Cached answers are used even without a key (no request then)."""
    st = p.settings()
    if not st.get("gemini_court"):
        return None
    key = api_key_from_app_settings()
    vision = GeminiVision(key, st.get("gemini_model") or DEFAULT_MODEL) if key else None
    return calibrate_gemini(p.video, data, size, vision, progress, cancel, work=p.results / "work",
                            project=str(getattr(p, "id", "")))


GOOD_MIN_FRAMES = 5  # a Gemini result is used when it located at least this many frames


def calibrate_gemini(video: Path, data, size: tuple[int, int], vision, progress=None, cancel=None,
                     max_frames: int = MAX_FRAMES, work: Path | None = None, ledger: Ledger | None = None,
                     project: str = "", read_frames=None):
    """Located frames from Gemini keypoints (AutoResult like autocourt.calibrate_net) or None. ``vision`` may be
    None: then only cached answers are used (no request at all)."""
    from replay import autocourt, court, courtnet

    prog = progress or (lambda x, m: None)
    t0 = time.time()
    names = list(court.KEYPOINTS)
    pks, imgs, spender, stop = keypoints_per_frame(video, data, size, vision, names, progress, cancel, max_frames,
                                                   work, ledger, project, read_frames)
    if cancel is not None and cancel.is_set():
        return None
    ch = courtnet.choose_court([p for p in pks.values() if p], names, size)
    if ch is None:
        return None
    spec, L, W, _ = ch
    court.configure(L, W, spec)
    cxy = courtnet.court_points(names)
    frames, rejected = [], 0
    for i in sorted(pks):
        r = courtnet.frame_H(pks[i], cxy, size)
        if r is None or len(r[1]) < MIN_INLIERS:
            continue
        H = r[0] / r[0][2, 2]
        if line_support(imgs[i], H, size) < MIN_LINE_SUPPORT:
            rejected += 1
            continue
        frames.append({"t": round(float(data.t[i]), 4), "i": int(i), "H": [round(float(v), 8) for v in H.ravel()],
                       "source": "gemini"})
    res = autocourt.AutoResult(C=np.zeros(3), length=L, width=W, spec=spec, line_classes=(), frames=frames)
    res.good_frac = len(frames) / max(1, len(pks))
    res.message = (f"Gemini court keypoints: {spec} markings, court {L:.2f} x {W:.2f} m; {len(frames)} / {len(pks)} "
                   f"frames located, {rejected} rejected by the line check; this run cost at most "
                   f"{spender.eur:.3f} EUR ({spender.frames} frames sent){'; ' + stop if stop else ''} "
                   f"({time.time() - t0:.0f} s)")
    prog(1.0, res.message)
    return res
