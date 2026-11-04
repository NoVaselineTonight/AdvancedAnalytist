"""Manual court calibration UI (OpenCV window).

Workflow per keyframe:
  1. Navigate to a moment where >= 4 court landmarks are clearly visible.
  2. Click a landmark on the court diagram (right panel) to select it,
     then click the same spot in the video frame. Repeat for >= 4 points.
     More points (spread over the visible floor) = better calibration.
  3. The projected court lines are drawn live; check they sit on the real lines.
  4. Press ENTER to store the keyframe. The file is saved after every change.

For a panning camera, add a keyframe every time the camera has moved to a
clearly different view, and at least every ~30-60 s. Tracking between
keyframes is handled by camera-motion propagation in the pipeline.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from replay import court, homography
from replay.calibration import CalibrationFile, Keyframe
from replay.video import probe, read_frame_at

WIN = "AdvancedAnalytist - court calibration"
PANEL_W = 470
S = 15  # diagram pixels per metre
DX, DY = 25, 40
KEY_LEFT, KEY_UP, KEY_RIGHT, KEY_DOWN = 2424832, 2490368, 2555904, 2621440

HELP = [
    "click diagram point, then click video",
    "ENTER save keyframe   x delete keyframe",
    "u undo point   c clear points",
    "<- -> 1s   up/down 10s   , . 0.1s",
    "n / p  next / prev keyframe",
    "g  go to time (type in terminal)",
    "o  overlay   f  fixed-camera flag",
    "q / ESC  save and quit",
]


def _diagram_xy(p: tuple[float, float]) -> tuple[int, int]:
    return int(DX + p[0] * S), int(DY + (court.WIDTH - p[1]) * S)


class CalibrationUI:
    def __init__(self, video: str | Path, out_path: str | Path, start_s: float = 0.0):
        self.video = str(video)
        self.info = probe(video)
        self.out_path = Path(out_path)
        if self.out_path.exists():
            self.calib = CalibrationFile.load(self.out_path)
        else:
            self.calib = CalibrationFile(video=str(Path(video).resolve()), frame_size=self.info.size)
        self.scale = min(1.0, 1280.0 / self.info.width)
        self.t = start_s
        self.frame_t = 0.0
        self.frame: np.ndarray | None = None
        self.points: dict[str, tuple[float, float]] = {}
        self.order: list[str] = []
        self.selected: str | None = None
        self.overlay = True
        self.mouse = (-1, -1)
        self.message = ""
        self.preview_H: np.ndarray | None = None
        self.preview_err: float | None = None

    # ------------------------------------------------------------------ io
    def load_frame(self) -> None:
        self.t = float(np.clip(self.t, 0.0, max(0.0, self.info.duration_s - 0.1)))
        res = read_frame_at(self.video, self.t)
        if res is None:
            self.message = "could not read frame"
            return
        self.frame_t, self.frame = res
        kf = self.calib.keyframe_at(self.frame_t, tol=0.02)
        self.points = dict(kf.points) if kf else {}
        self.order = list(self.points)
        self._refit()

    def save(self) -> None:
        self.calib.save(self.out_path)

    # ------------------------------------------------------------------ logic
    def _refit(self) -> None:
        self.preview_H, self.preview_err = None, None
        if len(self.points) >= 4:
            kf = Keyframe(self.frame_t, dict(self.points))
            try:
                kf.fit(self.info.size)
                self.preview_H, self.preview_err = kf.H, kf.reproj_px
            except ValueError as e:
                self.message = str(e)

    def store_keyframe(self) -> None:
        if len(self.points) < 4:
            self.message = "need at least 4 points"
            return
        kf = Keyframe(self.frame_t, dict(self.points))
        try:
            kf.fit(self.info.size)
        except ValueError as e:
            self.message = f"fit failed: {e}"
            return
        self.calib.keyframes = [k for k in self.calib.keyframes if abs(k.time_s - self.frame_t) > 0.02]
        self.calib.keyframes.append(kf)
        self.calib.keyframes.sort(key=lambda k: k.time_s)
        self.save()
        self.message = f"saved keyframe @ {self.frame_t:.2f}s  ({len(self.points)} pts, {kf.reproj_px:.1f}px)"

    def delete_keyframe(self) -> None:
        before = len(self.calib.keyframes)
        self.calib.keyframes = [k for k in self.calib.keyframes if abs(k.time_s - self.frame_t) > 0.02]
        if len(self.calib.keyframes) < before:
            self.save()
            self.message = "keyframe deleted"

    def on_mouse(self, event: int, x: int, y: int, flags: int, param: object) -> None:
        self.mouse = (x, y)
        if event != cv2.EVENT_LBUTTONDOWN or self.frame is None:
            return
        vw = int(self.info.width * self.scale)
        if x >= vw:  # panel: pick nearest keypoint
            best, bd = None, 14.0
            for name, p in court.KEYPOINTS.items():
                dx, dy = _diagram_xy(p)
                d = float(np.hypot(dx - (x - vw), dy - y))
                if d < bd:
                    best, bd = name, d
            if best:
                self.selected = best
                self.message = f"selected {best} - now click it in the video"
        elif self.selected:
            self.points[self.selected] = (x / self.scale, y / self.scale)
            if self.selected in self.order:
                self.order.remove(self.selected)
            self.order.append(self.selected)
            self.message = f"placed {self.selected}"
            self.selected = None
            self._refit()
        else:
            self.message = "select a point on the diagram first"

    # ------------------------------------------------------------------ drawing
    def render(self) -> np.ndarray:
        assert self.frame is not None
        vid = cv2.resize(self.frame, None, fx=self.scale, fy=self.scale) if self.scale != 1 else self.frame.copy()
        H = self.preview_H
        if self.overlay and H is not None:
            for line in court.court_polylines():
                pts = homography.court_to_image(H, line) * self.scale
                for a, b in zip(pts[:-1], pts[1:]):
                    if np.isfinite(a).all() and np.isfinite(b).all():
                        cv2.line(vid, tuple(np.int32(a)), tuple(np.int32(b)), (0, 255, 255), 1, cv2.LINE_AA)
        for name, (u, v) in self.points.items():
            p = (int(u * self.scale), int(v * self.scale))
            cv2.circle(vid, p, 5, (0, 0, 255), 2)
            cv2.putText(vid, name, (p[0] + 6, p[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

        panel = np.full((max(vid.shape[0], 620), PANEL_W, 3), 32, np.uint8)
        for line in court.court_polylines():
            pts = np.array([_diagram_xy(tuple(p)) for p in line], np.int32)
            cv2.polylines(panel, [pts], False, (200, 200, 200), 1, cv2.LINE_AA)
        for name, p in court.KEYPOINTS.items():
            c = (0, 200, 0) if name in self.points else (0, 140, 255)
            if name == self.selected:
                c = (255, 255, 0)
            cv2.circle(panel, _diagram_xy(p), 5 if name == self.selected else 4, c, -1)
        txt = [
            f"t = {self.frame_t:8.2f} s   keyframes: {len(self.calib.keyframes)}"
            + ("  [FIXED CAMERA]" if self.calib.fixed_camera else ""),
            f"points: {len(self.points)}" + (f"   fit error: {self.preview_err:.1f}px" if self.preview_err is not None else ""),
            "keyframe here: " + ("YES" if self.calib.keyframe_at(self.frame_t, 0.02) else "no"),
            f"selected: {self.selected or '-'}",
            "x=0 is the LEFT basket, y=0 the NEAR sideline",
            "(as seen from the camera)",
            "",
            *HELP,
        ]
        y0 = DY + int(court.WIDTH * S) + 30
        for i, s in enumerate(txt):
            cv2.putText(panel, s, (12, y0 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)
        cv2.putText(panel, self.message[:60], (12, panel.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 220, 255), 1, cv2.LINE_AA)
        self._loupe(vid, panel)
        if vid.shape[0] < panel.shape[0]:
            vid = cv2.copyMakeBorder(vid, 0, panel.shape[0] - vid.shape[0], 0, 0, cv2.BORDER_CONSTANT)
        return np.hstack([vid, panel])

    def _loupe(self, vid: np.ndarray, panel: np.ndarray) -> None:
        mx, my = self.mouse
        if not (0 <= mx < vid.shape[1] and 0 <= my < vid.shape[0]):
            return
        r, zoom = 15, 4
        src = cv2.copyMakeBorder(vid, r, r, r, r, cv2.BORDER_CONSTANT)[my:my + 2 * r, mx:mx + 2 * r]
        big = cv2.resize(src, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)
        c = big.shape[0] // 2
        cv2.line(big, (c, 0), (c, big.shape[0]), (0, 0, 255), 1)
        cv2.line(big, (0, c), (big.shape[1], c), (0, 0, 255), 1)
        y = DY + int(court.WIDTH * S) + 18 * 7 + 20
        x = PANEL_W - big.shape[1] - 8
        panel[y:y + big.shape[0], x:x + big.shape[1]] = big

    # ------------------------------------------------------------------ main loop
    def run(self) -> None:
        cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WIN, self.on_mouse)
        self.load_frame()
        while True:
            if self.frame is not None:
                cv2.imshow(WIN, self.render())
            key = cv2.waitKeyEx(30)
            if key == -1:
                if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                    break
                continue
            ch = chr(key) if 0 <= key < 256 else ""
            step = {KEY_RIGHT: 1.0, KEY_LEFT: -1.0, KEY_UP: 10.0, KEY_DOWN: -10.0}.get(key)
            if ch in ".,":
                step = 0.1 if ch == "." else -0.1
            if step is not None:
                self.t = self.frame_t + step
                self.load_frame()
            elif ch in ("q", "\x1b"):
                break
            elif ch in ("\r", "\n"):
                self.store_keyframe()
            elif ch == "x":
                self.delete_keyframe()
            elif ch == "u" and self.order:
                self.points.pop(self.order.pop(), None)
                self._refit()
            elif ch == "c":
                self.points, self.order = {}, []
                self._refit()
            elif ch == "o":
                self.overlay = not self.overlay
            elif ch == "f":
                self.calib.fixed_camera = not self.calib.fixed_camera
                self.save()
                self.message = f"fixed camera: {self.calib.fixed_camera}"
            elif ch in ("n", "p"):
                ts = [k.time_s for k in self.calib.keyframes]
                cand = [t for t in ts if t > self.frame_t + 0.02] if ch == "n" else [t for t in ts if t < self.frame_t - 0.02]
                if cand:
                    self.t = min(cand) if ch == "n" else max(cand)
                    self.load_frame()
            elif ch == "g":
                try:
                    self.t = _parse_time(input("go to time (s or mm:ss): "))
                    self.load_frame()
                except ValueError:
                    self.message = "bad time"
        self.save()
        cv2.destroyAllWindows()
        print(f"Calibration saved to {self.out_path} ({len(self.calib.keyframes)} keyframes)")


def _parse_time(s: str) -> float:
    s = s.strip()
    if ":" in s:
        parts = [float(p) for p in s.split(":")]
        out = 0.0
        for p in parts:
            out = out * 60 + p
        return out
    return float(s)
