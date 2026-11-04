# AdvancedAnalytist: basketball 2D replay from game video

Turns a recorded basketball game into a top-down 2D replay: every visible person's
court position (in metres) over time, as CSV/JSON and as an interactive HTML viewer.

**Status: Milestone 1 + desktop app** — manual calibration (fixed *or* panning camera), player
detection, tracking, team classification, 2D replay of players. Ball tracking is
M2, automatic court-line detection is M3, full-game tuning and GPU speed-ups are M4.

---

## Setup (Windows, macOS, Linux)

Python 3.11+.

```bash
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate

# CPU-only PyTorch (use this on AMD GPUs / machines without NVIDIA):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# (NVIDIA GPU instead: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124)

pip install -e ".[rfdetr,dev]"
pytest                             # homography / court / smoothing tests
```

On first run RF-DETR downloads its COCO weights (~130 MB for *medium*) into `~/.roboflow/models`.

### Hardware notes / speed
* **GPU (default when available):** the detector runs through ONNX Runtime + DirectML, so any DirectX 12
  card works, AMD included (no CUDA/ROCm needed). The ONNX model is exported once from the RF-DETR weights
  into `models/` on first use (~10 s). Detections match the PyTorch model (same boxes, scores within 0.002).
* Measured on Ryzen 5 5600X + RX 6700 XT **while a game was using the CPU and GPU**: 30-s clip analysed in
  43 s (~8 analysed frames/s), versus ~0.9 frames/s for PyTorch on the CPU under the same load. Expect more
  on an idle machine.
* Decoding, detection and tracking run as three overlapping threads, so the GPU isn't left waiting.
* **Analyse every Nth frame** (GUI: Overview → Analyse; CLI: `--stride N`). Default every 3rd frame
  (~10 per second of video). Playback always shows every frame; court positions are interpolated between
  analysed frames. Every 2nd frame is 1.5× the work, every frame 3×.
* Processing device: Auto / GPU / CPU (GUI: Overview → Processing device; CLI: `--device`). It can be
  switched without re-analysing; results are the same.
* CPU fallback: PyTorch on the CPU (~3 frames/s on an idle 5600X). On NVIDIA, install `onnxruntime-gpu` and
  the CUDA provider is used automatically.

---

## Desktop app (recommended)

```bash
python -m replay gui
```

A normal desktop window (Qt). No browser, no server. Each analysed video is a **Replay**
(listed on the left; right-click a Replay to rename, pin, export or delete it).

1. **Import video…** Pick any video file: mp4, mov, mkv, avi, ts/mts/m2ts, webm, wmv, flv, 3gp,
   mpg and anything else FFmpeg reads. The app makes an H.264 playback copy and immediately
   starts detecting and tracking players in the background. You can keep working, close the app
   (progress is kept) or stop and continue later.
2. **Calibrate** tab: pick a court landmark on the diagram, then click the same spot on the frame
   (mouse wheel zooms, right-drag pans). Save a keyframe with Enter. You can do this while the
   analysis is still running.
3. When both are done the 2D replay is built automatically. The **Replay** tab plays the video
   and the top-down court side by side, in sync, with 0.25×–4× speed, trails and track IDs.
4. **Export** menu (also in the Replay list's right-click menu):
   * **Export Replay (.aareplay)…** (Ctrl+S) writes ONE file with everything: the original video,
     the playback copy, the analysis cache, the 2D replay data, calibration, game info and the
     timeline with all marks and notes, plus a `manifest.json` that is checked when loading.
     **File → Open Replay file…** loads it on any computer without re-analysing (older `.aaproj`
     files still open). Videos are stored uncompressed in the zip, so multi-GB exports are fast.
   * **Simple data export (for AI)…** creates a new folder `<name>_data_<YYYYMMDD-HHMMSS>/` with
     CSV/JSON files and a `README_FOR_AI.md` explaining the data, see below.
   * **Individual files ▸** the CSV/JSON tracking data, the processing report or the stand-alone
     HTML replay.

Replays are saved automatically in `AdvancedAnalytist/projects/` (change it with
`--projects <folder>`). Re-calibrating or editing team colours rebuilds the replay in seconds.
Changing the analysis frame rate or detector size re-runs the analysis.

### Simple data export (for AI)

Every export creates a new folder, so earlier exports are never overwritten:

| file | contents |
|---|---|
| `README_FOR_AI.md` | instructions for an AI analyst, filled in with this game's numbers: data description, coordinate system, quality flags, suggested analyses with formulas, pitfalls, a step-by-step plan and a report format |
| `game.json` | teams and colours, court geometry (lines, baskets), sampling, coverage statistics, export time |
| `players.csv` | `t, frame, track_id, team, team_name, role, x_m, y_m, confidence, interpolated, calibration` |
| `ball.csv` | `t, frame, x_m, y_m, state, owner_track_id, owner_team, owner_team_name` |
| `events.csv` / `events.json` | detected baskets with a running **detected (not official)** score, 2 points per basket |
| `timeline_marks.csv` / `timeline.json` | the timeline: user marks and notes, auto-detected marks (with the user's edits; dismissed ones left out) |
| `track_summary.csv` | per track: team, first/last seen, time tracked, % interpolated, distance, mean/max speed, mean position |
| `possession_summary.csv` | ball possession time and number of possessions per team |
| `replay_2d.html` | stand-alone 2D replay with the baskets on the time line and the running score |
| `report.md` | the processing report |

The same function is available without the GUI:
`python -c "from replay.data_export import export_simple; print(export_simple('projects/<id>', 'exports'))"`.

## Command line

```bash
# 0. (recommended for testing) cut a short clip, re-encoded to browser-playable H.264
python -m replay clip --video game.ts --start 44:00 --duration 180 --out data/clip.mp4

# 1. detection + tracking (slow, resumable; Ctrl-C and re-run the same command to continue)
python -m replay process --video data/clip.mp4 --meta examples/game.example.json --out results/clip

# 2. calibration: opens a window, click court landmarks (see below)
python -m replay calibrate --video data/clip.mp4

# 3. build outputs (fast; re-run whenever you add keyframes or change settings)
python -m replay postprocess --out results/clip --meta examples/game.example.json

# 4. QA video: court lines + track IDs + mini-map burned into the footage
python -m replay check --out results/clip
```

Steps 1 and 2 are independent: calibration can be done before, during or after
detection. If a calibration file exists when `process` finishes, it runs step 3 automatically.

Outputs in `results/clip/`:

| file | contents |
|---|---|
| `tracking.csv` / `tracking.json` | one row per person per processed frame: `frame, timestamp, track_id, object_type, team, x_m, y_m, confidence, interpolated, calibration` |
| `replay.html` | self-contained viewer: play/pause, scrub, 0.25×–4× speed, trails, optional synced video |
| `report.md` / `report.json` | calibration coverage, players per frame, failed time ranges |
| `check.mp4` | QA video (from `check`) |
| `work/` | resumable intermediate results (safe to delete after processing) |

### Court coordinate system
FIBA court, 28 m × 15 m, metres. **Origin = the corner where the LEFT baseline meets the
NEAR sideline, as seen from the camera.** `x` runs along the sideline (0 → 28, left → right
basket). `y` runs along the baseline (0 → 15, near → far sideline). In the viewer the near
sideline is at the bottom, which matches the camera's point of view.

### Calibration UI
* Right panel: court diagram with clickable landmarks (orange = unused, green = placed).
  Click a landmark there, then click the same spot on the video. A magnifier helps you aim.
* With ≥ 4 points the court lines are projected onto the video **live**. They must sit on
  the real lines everywhere, not just at the points you clicked. The fit error should be < 5 px.
* `ENTER` stores a keyframe; the file `<video>.calibration.json` is saved on every change.
* Navigation: `←/→` 1 s, `↑/↓` 10 s, `,`/`.` 0.1 s, `n`/`p` next/previous keyframe, `g` go to time.
* `f` toggles **fixed camera** mode (one keyframe is used for the whole video).
* **Panning camera:** add a keyframe whenever the view has changed substantially, and at
  least every 30–60 s. Between keyframes, positions use calibration propagated through
  measured camera motion. Use `check` to verify the lines stay glued to the floor.
* Tips: use line intersections on the floor (paint corners, free-throw line ends, centre
  circle on the centre line), spread across the visible area. Gym floors with several sports'
  lines are confusing; basketball lines are usually the darkest/widest set.

### Metadata JSON
See [`examples/game.example.json`](examples/game.example.json). Team `A` = home, `B` = away.
`jersey_colors` seed the team clustering. They only need to be approximately right; the
clusters adapt to the actual lighting. `overlay_regions` are normalised `[x0, y0, x1, y1]`
screen rectangles covered by broadcast graphics (scorebug, logo). They are excluded from
camera-motion estimation, and players whose feet are hidden behind them are skipped.

### Configuration
Every tunable is in [`replay/config.py`](replay/config.py). Override with
`--config my.json`, e.g. `{"process_fps": 5, "detector": {"variant": "small", "tiles": [1, 2]}}`.

---

## How to film for the best results
1. **Tripod, elevated.** As high as possible (balcony, top of the bleachers, or a 3–5 m light
   stand). Elevation is the biggest single factor: players occlude each other less, and foot
   positions map to the court far more accurately.
2. **Whole court in frame, camera at the centre line**, on the sideline opposite the benches.
3. **Don't zoom and don't pan.** A fixed wide shot means one calibration for the whole game
   (`f` in the calibration UI) and the most reliable results. If you must pan, pan smoothly
   on a tripod head and never zoom.
4. Landscape orientation, 1080p or higher, 30+ fps, lock exposure/focus, avoid back-light from
   windows.
5. Keep the camera running through the whole game. Every restart needs a new calibration.

---

## Known limitations (read these)
* **Positions are foot positions from a single camera.** The point under the bottom of each
  person's bounding box is projected onto the floor. Jumping players, occluded feet and
  sloppy boxes give errors of ~0.3–1 m near the camera, more on the far side. The smoother
  reduces jitter but cannot remove bias.
* **Panning/zooming footage (like broadcast streams)** is handled by propagating manual
  keyframes through measured camera motion. Drift accumulates between keyframes. If the
  motion estimate fails (fast whip pans, motion blur, cuts), those frames are marked as
  having **no calibration** and no positions are output for them. We never guess. Lens
  distortion is not modelled yet, so wide phone lenses bend straight lines near the image edges.
* **Only what the camera sees is tracked.** A panning camera that shows half the court means
  half the players are simply absent from the replay at those times.
* **Track IDs are not identities.** IDs break and switch on occlusion, when players leave the
  frame, and during fast pans. One real player usually ends up with several track IDs over a
  game. Jersey numbers are read automatically (`replay/jersey_numbers.py`) and shown where they
  were read consistently (at least 3 agreeing reads); they also keep the tracker from joining two
  players with different numbers. Numbers are small and often hidden in this kind of footage, so
  most tracks get none: an empty number means "not read", never "no number".
* **Team classification is colour-based.** It fails if teams wear similar colours, and coaches
  or bench players in team colours near the court edge can be classified as players. Referees
  are a separate class when their outfit differs from both teams.
* **Ball: not yet (M2).** When it arrives, positions of a ball in the air will be estimates
  (a floor projection is wrong for an airborne ball) and will be marked as such.
* The pretrained COCO detector is not basketball-specific. Fine-tuning would help but needs a
  labelled dataset. Nothing is trained without your go-ahead.

---

## Models and licences
| component | licence |
|---|---|
| RF-DETR code (`rfdetr` package, Roboflow) | Apache-2.0 |
| RF-DETR COCO weights, Nano/Small/Medium/Large (downloaded by the package) | Apache-2.0 (per Roboflow; XL/2XL "plus" models use a different licence and are not used) |
| supervision (ByteTrack implementation) | MIT |
| OpenCV, NumPy, SciPy, scikit-learn | Apache-2.0 / BSD |
| ONNX Runtime (DirectML build) - GPU inference | MIT |
| imageio-ffmpeg (bundled ffmpeg binary: import/conversion, `clip`, `check`) | BSD-2 wrapper; ffmpeg binary is GPL/LGPL |
| Qt for Python / PySide6 (desktop app, video playback) | LGPL-3.0 |
| PP-OCRv4_mobile_rec text recognition (PaddleOCR) - jersey numbers; `models/pp-ocrv4-mobile-rec/` (source https://huggingface.co/PaddlePaddle/PP-OCRv4_mobile_rec, rev. a2deb05; converted to ONNX by `replay/ppocr_onnx.py`; LICENSE + NOTICE.txt with checksums in that folder) | Apache-2.0 |

No Ultralytics/YOLO (AGPL-3.0) code or weights are used. The detector sits behind an interface
(`replay/detection/`), so other backends can be added.

### Appearance (re-ID) model - trained in this project
`models/reid/reid.onnx` (used by `replay/reid.py`, opt-in, off by default) is a small CNN (`tools/reid/net.py`,
project code, 0.85 M parameters, 128x64 crops -> 128-d vector) trained **from scratch** (no pretrained weights, no
external data) on CPU, with the user's approval, on crops from the user's own game video only.
- Data: `tools/reid/extract_crops.py` on the full koristv game (KlaNMKY-LePy): 64 265 person crops from 7 010
  "clean" ByteTrack pieces (cut at gaps, box overlaps and jersey-colour jumps). Train = Q1-Q3 (t < 3767 s, 44 747
  crops / 4 943 pieces); held-out eval = Q4 (17 427 crops) plus confident jersey-number reads.
- Labels: self-supervised - crops of one piece = same person; pieces present at the same moment = different people
  (same-team players are the hard negatives). Loss: multi-positive InfoNCE + batch-hard triplet within a moment.
- Training: `tools/reid/train.py` (AdamW, cosine LR, checkpoint/resume); log `models/reid/train_log.json`,
  summary `models/reid/train_info.json` (epochs/iterations, minutes, best eval).
- Trained: 150 min CPU, 7 950 iterations of 96 crops (~17 passes over the train crops), best iteration 7 500
  (`models/reid/train_info.json`, eval history `train_log.json`).
- Evaluation (`tools/reid/evaluate.py`, Q4 held out): same-team jersey-number identity >30 s apart - re-ID
  r1 0.682 / mAP 0.590 vs colour histogram 0.537 / 0.324, random 0.147 / 0.167; same-team pair AUC 0.885 single
  crop, 0.952 with 5-crop means. Other games (`tools/reid/eval_external.py`): no better than colour (TrackID3x3 mAP
  0.48 vs 0.51; broadcast segments 0.77 vs 0.80) - the model is specific to this game; retrain per game if needed.
  Details: `data/benchmark/notes_reid.md`.

## Project layout
```
replay/
  cli.py            commands: gui, clip, process, calibrate, postprocess, check, info
  data_export.py    "Simple data export (for AI)": documented CSV/JSON folder
  gui/              desktop app: app.py (windows), widgets.py, projects.py (Replay save/load/.aareplay),
                    jobs.py (background analysis), media.py (format conversion)
  config.py         every tunable
  court.py          FIBA geometry, keypoints, coordinate system
  homography.py     image<->court maths (fit, project, Jacobian, blend)
  calibration.py    keyframes + per-frame homography via camera-motion propagation
  calibrate_ui.py   OpenCV click UI
  camera.py         frame-to-frame camera motion (optical flow + RANSAC)
  detection/        Detector interface, RF-DETR backend, tiling
  tracking.py       ByteTrack + camera-motion compensation
  features.py       jersey colour features
  teams.py          seeded KMeans team classification
  jersey_numbers.py jersey number reading (PP-OCRv4 on torso crops, cached per pass-1 row), per-track voting
  ppocr_onnx.py     one-off converter of the PaddleOCR recognition model to ONNX
  smoothing.py      Kalman + RTS smoother, gap handling
  pipeline.py       pass 1 (resumable) / pass 2
  export.py, report.py, debug_video.py, viewer/template.html
tests/              court + homography + smoothing tests (synthetic camera)
```
