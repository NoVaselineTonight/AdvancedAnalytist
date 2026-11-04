"""A minimal, analysed-looking project folder for export tests (no video decoding, no analysis)."""

from __future__ import annotations

import json
import time
from pathlib import Path

from replay.export import COLUMNS

TEAMS = {"A": {"label": "A", "name": "Kouvot", "primary": "#ff8800", "secondary": None},
         "B": {"label": "B", "name": "Tapiolan Honka", "primary": "#1144aa", "secondary": None}}
FPS = 10.0
N = 40  # analysed frames, t = 0.0 .. 3.9 s


def _row(k, tid, otype, team, x, y, interp=False, state="", owner=None, calib="propagated"):
    return [k * 3, round(k / FPS, 3), tid, otype, team, x, y, 0.9, interp, calib, state,
            "" if owner is None else owner, "", "", "", ""]


def make_project(root: Path, *, events: bool = True, timeline: bool = True, original: Path | None = None,
                 copy_original: bool = False) -> Path:
    root.mkdir(parents=True)
    res = root / "results"
    (res / "work").mkdir(parents=True)
    src = {"filename": "game.mts", "kind": "reference", "path": str(original) if original else "C:/nowhere/game.mts"}
    if copy_original:
        (root / "source").mkdir()
        (root / "source" / "game.mts").write_bytes(b"ORIGINAL" * 100)
        src = {"filename": "game.mts", "kind": "copy", "path": "source/game.mts"}
    info = {"format": "advanced-analytist/project@1", "name": "Kouvot vs Honka", "created": time.time(),
            "source": src, "settings": {"frame_stride": 3},
            "video_info": {"duration_s": 4.0, "size": [1280, 720], "fps": 30.0}}
    (root / "project.json").write_text(json.dumps(info), encoding="utf-8")
    (root / "meta.json").write_text(json.dumps({"home_team": "Kouvot", "away_team": "Tapiolan Honka"}),
                                    encoding="utf-8")
    (root / "calibration.json").write_text(json.dumps({"video": "video.mp4", "frame_size": [1280, 720], "keyframes": [{"time_s": 1.0, "points": {"center": [640, 400]}}]}), encoding="utf-8")
    (root / "video.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"V" * 5000)
    (res / "work" / "run.json").write_text(json.dumps({"video": "C:/old/video.mp4", "video_size_bytes": 1}),
                                           encoding="utf-8")
    (res / "work" / "DONE").write_text("1s\n", encoding="utf-8")
    (res / "work" / "chunk_00000.npz").write_bytes(b"N" * 3000)

    rows = []
    for k in range(N):
        rows.append(_row(k, 1, "player", "A", 5.0 + 0.3 * k, 7.0, interp=(k % 10 == 5)))  # 3 m/s
        rows.append(_row(k, 2, "player", "B", 20.0, 3.0 + 0.1 * k))
        rows.append(_row(k, 3, "referee", "referee", 14.0, 1.0, calib="keyframe"))
        if k < 20:
            rows.append(_row(k, 4, "player", "unknown", 27.5, 14.8))
        state = "possessed" if k < 25 else "estimated"
        rows.append(_row(k, "ball", "ball", "", 5.0 + 0.3 * k, 7.0, interp=state != "possessed", state=state,
                         owner=1 if state == "possessed" else None))
    ev = [{"id": "basket-30-L", "type": "basket", "t": 1.0, "frame": 30, "basket": "L", "team": "A",
           "confidence": 0.9, "source": "auto", "details": {"rim_dist_m": 0.2}},
          {"id": "basket-60-R", "type": "basket", "t": 2.0, "frame": 60, "basket": "R", "team": "B",
           "confidence": 0.6, "source": "auto", "details": {}},
          {"id": "basket-90-L", "type": "basket", "t": 3.0, "frame": 90, "basket": "L", "team": "A",
           "confidence": 0.8, "source": "auto", "details": {}},
          {"id": "basket-99-R", "type": "basket", "t": 3.3, "frame": 99, "basket": "R", "team": "B",
           "confidence": 0.3, "source": "auto", "details": {}}]
    tracking = {"format": "advanced-analytist/tracking@2", "game": {"teams": TEAMS, "game_date": "2026-10-01",
                                                                   "location": "Kouvola"},
                "process_fps": FPS, "video": "x", "columns": COLUMNS, "rows": rows,
                "tracks": {"1": {"team": "A", "team_confidence": 0.9}, "2": {"team": "B", "team_confidence": 0.8},
                           "3": {"team": "referee", "team_confidence": 0.7},
                           "4": {"team": "unknown", "team_confidence": 0.2}}}
    replay = {"game": {"teams": TEAMS}, "fps": FPS, "t": [round(k / FPS, 3) for k in range(N)],
              "frame": [k * 3 for k in range(N)], "calib": [1] * N,
              "court": {"length": 28, "width": 15, "lines": [], "baskets": [[1.575, 7.5], [26.425, 7.5]]},
              "tracks": [], "ball": None, "track_info": {}, "n": N, "video_src": "../video.mp4"}
    if events:
        tracking["events"] = ev
        replay["events"] = ev
    (res / "tracking.json").write_text(json.dumps(tracking), encoding="utf-8")
    (res / "replay_data.json").write_text(json.dumps(replay), encoding="utf-8")
    (res / "tracking.csv").write_text("frame\n", encoding="utf-8")
    (res / "replay.html").write_text("<html>old</html>", encoding="utf-8")
    (res / "report.json").write_text(json.dumps({
        "frames_processed": N, "process_fps": FPS, "time_range_s": [0.0, 3.9],
        "calibration": {"valid_pct": 97.5, "keyframe_frames": 1, "propagated_frames": 38, "missing_frames": 1,
                        "warnings": []},
        "people": {"avg_tracked_per_calibrated_frame": 3.5, "avg_measured_per_calibrated_frame": 3.2,
                   "tracks_output": 4, "team_centroids_lab": [[1, 2, 3]]},
        "ball": {"detection_rate_pct": 80.0, "position_rate_pct": 100.0, "possessed_pct": 62.5},
        "failed_ranges": {"no_calibration": [], "no_players_tracked": []}}), encoding="utf-8")
    (res / "report.md").write_text("# Processing report\n", encoding="utf-8")
    if timeline:
        (root / "timeline.json").write_text(json.dumps({
            "format": "advanced-analytist/timeline@1",
            "tracks": [
                {"id": "auto1", "name": "Auto detection", "color": "#8b5cf6", "visible": True, "locked": False,
                 "kind": "auto", "marks": []},
                {"id": "notes", "name": "Notes", "color": "#0ea5e9", "visible": True, "locked": False, "kind": "user",
                 "marks": [{"id": "m1", "t": 1.5, "duration": 2.0, "title": "Fast break",
                            "note": "Kouvot run the floor after a steal", "color": None}]},
            ],
            "auto_overrides": {"basket-99-R": {"dismissed": True},
                               "basket-60-R": {"title": "Honka layup", "note": "contested"}}}), encoding="utf-8")
    # make the replay up to date (tracking.json newer than calibration/meta)
    old = time.time() - 100
    import os
    for f in ("calibration.json", "meta.json"):
        os.utime(root / f, (old, old))
    return root
