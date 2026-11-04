"""Processing report: calibration coverage, tracking density, failure ranges."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from replay.calibration import KEYFRAME, NONE, PROPAGATED
from replay.export import _write_text_atomic

if TYPE_CHECKING:
    from replay.calibration import ResolvedCalibration
    from replay.config import Config
    from replay.export import TrackRow
    from replay.metadata import GameMeta
    from replay.pipeline import Pass1Data
    from replay.teams import TeamResult


BUILD_VERSION = 8  # bump when pass 2 changes in a way that old replays should be rebuilt


def _ranges(t: np.ndarray, bad: np.ndarray, min_len_s: float) -> list[tuple[float, float]]:
    out = []
    i, n = 0, len(bad)
    while i < n:
        if bad[i]:
            j = i
            while j + 1 < n and bad[j + 1]:
                j += 1
            if t[j] - t[i] >= min_len_s:
                out.append((round(float(t[i]), 2), round(float(t[j]), 2)))
            i = j + 1
        else:
            i += 1
    return out


def _fmt(s: float) -> str:
    s = round(float(s), 2)  # round first: 59.996 s is "01:00.00", not "00:60.00"
    return f"{int(s // 60):02d}:{s % 60:05.2f}"


def _scoring_section(events: list[dict] | None, meta: "GameMeta", game_state: dict | None = None) -> dict:
    """Made baskets per team. With the game state (replay/game_state.py) the team of each basket is the one
    attacking that basket (attack direction per half) or the free-throw shooter's team, which is far more
    reliable than the last ball holder that scoring.py credits; points are the game state's guess (1/2/3)."""
    ev = [e for e in (events or []) if e.get("type") == "basket"]
    if game_state:
        from replay.game_state import merge_events
        ev = [e for e in merge_events(ev, game_state) if e.get("type") == "basket"]
    by_team = {"A": 0, "B": 0, "unknown": 0}
    points = {"A": 0, "B": 0, "unknown": 0}
    for e in ev:
        k = e.get("team") if e.get("team") in ("A", "B") else "unknown"
        by_team[k] += 1
        points[k] += int(e.get("points_guess") or 2)
    return {
        "total": len(ev),
        "baskets_by_team": by_team,
        "points_guess_by_team": points,
        "team_names": {"A": meta.home.name, "B": meta.away.name, "unknown": "unknown"},
        "events": [{"t": e["t"], "frame": e["frame"], "basket": e["basket"], "team": e.get("team"),
                    "confidence": e["confidence"], "label": e.get("label") or "Made basket",
                    "points_guess": e.get("points_guess"),
                    "team_source": ((e.get("details") or {}).get("game_state") or {}).get("team_source")
                    or ("free-throw shooter" if e.get("points_guess") == 1 else "last holder")}
                   for e in ev],
        "note": "made baskets detected from the ball going down through the rim (tuned for precision, so some "
                "are missed); team = attacking team of that basket (attack direction from the game state) when "
                "known, else the last ball holder; points are a guess (free throw / 2 / 3 from the shot position)",
    }


def build_report(out: Path, data: "Pass1Data", res: "ResolvedCalibration", rows: list["TrackRow"],
                 teams: "TeamResult", meta: "GameMeta", cfg: "Config", ball_frames: list | None = None,
                 filtered: dict | None = None, events: list[dict] | None = None,
                 game_state: dict | None = None) -> dict:
    n = len(data.k)
    q = Counter(res.quality)
    per_frame = np.zeros(n, int)
    per_frame_measured = np.zeros(n, int)
    team_counts: Counter[str] = Counter()
    # sample index -> frame position; k has gaps when pass 1 skipped breaks (preliminary analysis)
    pos = {int(k): i for i, k in enumerate(data.k)}
    for r in rows:
        if r.object_type in ("player", "referee") and r.team != "unknown":
            i = pos.get(int(r.k))
            if i is None:
                continue
            per_frame[i] += 1
            if not r.interpolated:
                per_frame_measured[i] += 1
    calibrated = np.array([x != NONE for x in res.quality])
    for r in rows:
        team_counts["ball" if r.object_type == "ball" else r.team] += 1
    motion_known = np.array([m is not None for m in data.motion])
    raw_persons = data.rows[data.rows[:, 6] == 0]
    bf = ball_frames or []
    ball_seen = np.array([f.box is not None for f in bf]) if bf else np.zeros(n, bool)
    ball_pos = np.array([f.x is not None and f.state != "guessed" for f in bf]) if bf else np.zeros(n, bool)
    ball_guess = np.array([f.state == "guessed" for f in bf]) if bf else np.zeros(n, bool)
    ball_poss = np.array([f.state == "possessed" for f in bf]) if bf else np.zeros(n, bool)
    cal_frames = max(1, int(calibrated.sum()))
    rep = {
        "build_version": BUILD_VERSION,
        "frames_processed": n,
        "time_range_s": [round(float(data.t[0]), 2), round(float(data.t[-1]), 2)] if n else None,
        "process_fps": data.signature["process_fps"],
        "calibration": {
            "valid_pct": round(100.0 * calibrated.mean(), 1) if n else 0.0,
            "keyframe_frames": q[KEYFRAME], "propagated_frames": q[PROPAGATED], "missing_frames": q[NONE],
            "camera_motion_known_pct": round(100.0 * motion_known.mean(), 1) if n else 0.0,
            "warnings": res.warnings,
        },
        "people": {
            "raw_person_detections_per_frame": round(len(raw_persons) / max(1, n), 2),
            "avg_tracked_per_calibrated_frame": round(float(per_frame[calibrated].mean()), 2) if calibrated.any() else 0,
            "avg_measured_per_calibrated_frame": round(float(per_frame_measured[calibrated].mean()), 2)
            if calibrated.any() else 0,
            "tracks_output": len({r.track_id for r in rows if r.object_type != "ball"}),  # people (players/referees)
            "rows_by_team": dict(team_counts),
            "team_clustering_seeded_by_metadata": teams.seeded,
            "team_colors": teams.colors,
            "tracks_filtered_out": filtered or {},
            "team_centroids_lab": teams.centroids,
        },
        "ball": {
            "detection_rate_pct": round(100.0 * float((ball_seen & calibrated).sum()) / cal_frames, 1),
            "position_rate_pct": round(100.0 * float((ball_pos & calibrated).sum()) / cal_frames, 1),
            "guessed_pct": round(100.0 * float((ball_guess & calibrated).sum()) / cal_frames, 1),
            "possessed_pct": round(100.0 * float((ball_poss & calibrated).sum()) / cal_frames, 1),
            "note": "possessed = at the holder's position; estimated = interpolated between possessions "
                    "(passes/shots); a ball in the air is never projected directly",
        },
        "scoring": _scoring_section(events, meta, game_state),
        "failed_ranges": {
            "no_calibration": _ranges(data.t, ~calibrated, 1.0),
            "no_players_tracked": _ranges(data.t, calibrated & (per_frame == 0), 1.0),
        },
    }
    _write_text_atomic(out / "report.json", json.dumps(rep, indent=2))  # tmp + os.replace (export.py)

    lines = [
        "# Processing report", "",
        f"- Frames processed: **{n}** at {rep['process_fps']} fps "
        f"({_fmt(rep['time_range_s'][0])} - {_fmt(rep['time_range_s'][1])})" if n else "- No frames",
        f"- Valid court calibration: **{rep['calibration']['valid_pct']}%** "
        f"(keyframe {q[KEYFRAME]}, propagated {q[PROPAGATED]}, missing {q[NONE]})",
        f"- Camera motion estimated for {rep['calibration']['camera_motion_known_pct']}% of frames",
        f"- Avg people tracked per calibrated frame: **{rep['people']['avg_tracked_per_calibrated_frame']}** "
        f"(measured, not interpolated: {rep['people']['avg_measured_per_calibrated_frame']})",
        f"- Raw person detections per frame (incl. bench/crowd): {rep['people']['raw_person_detections_per_frame']}",
        f"- Tracks output: {rep['people']['tracks_output']}; rows by team: {dict(team_counts)}",
        f"- Ball: tracked in **{rep['ball']['detection_rate_pct']}%** of calibrated frames; court position in "
        f"{rep['ball']['position_rate_pct']}% (held by a player: {rep['ball']['possessed_pct']}%, rest estimated); "
        f"guessed for display only (never seen there): {rep['ball']['guessed_pct']}%",
        f"- Tracks removed as bench/coaches/spectators: {filtered or {}}", "",
        "## Time ranges with failed calibration (>= 1 s)",
        *([f"- {_fmt(a)} - {_fmt(b)}" for a, b in rep["failed_ranges"]["no_calibration"]] or ["- none"]),
        "", "## Calibrated time ranges with no tracked players (>= 1 s)",
        *([f"- {_fmt(a)} - {_fmt(b)}" for a, b in rep["failed_ranges"]["no_players_tracked"]] or ["- none"]),
    ]
    sc = rep["scoring"]
    lines += ["", "## Scoring (automatic, from the ball through the rim)",
              f"- Made baskets detected: **{sc['total']}** - " + ", ".join(
                  f"{sc['team_names'].get(k, k)}: {v} (~{sc['points_guess_by_team'][k]} pts)"
                  for k, v in sc["baskets_by_team"].items()),
              *[f"- {_fmt(e['t'])} {e['label']} at basket {e['basket']}, team "
                f"{sc['team_names'].get(e['team'] or 'unknown', 'unknown')} ({e['team_source']}; "
                f"confidence {e['confidence']:.2f})" for e in sc["events"]],
              f"- {sc['note']}"]
    if res.warnings:
        lines += ["", "## Calibration warnings", *[f"- {w}" for w in res.warnings]]
    _write_text_atomic(out / "report.md", "\n".join(lines) + "\n")
    return rep
