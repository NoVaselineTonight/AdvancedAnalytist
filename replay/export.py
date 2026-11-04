"""Output writers: tracking CSV/JSON and the self-contained HTML replay."""

from __future__ import annotations

import csv
import math
import io
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from replay import court
from replay.calibration import KEYFRAME, NONE

if TYPE_CHECKING:
    from replay.ball import BallFrame
    from replay.calibration import ResolvedCalibration
    from replay.metadata import GameMeta
    from replay.pipeline import Pass1Data

COLUMNS = ["frame", "timestamp", "track_id", "object_type", "team", "x_m", "y_m", "confidence", "interpolated",
           "calibration", "ball_state", "ball_owner", "img_x1", "img_y1", "img_x2", "img_y2", "pos_state",
           "pos_sigma_m", "source", "player_id", "id_confidence"]
# source: how a person row's position was obtained (from pos_state; see README_FOR_AI "Which rows to trust")
SOURCE_OF_STATE = {1: "seen", 2: "bridged", 3: "rejected", 4: "hidden", 5: "offscreen"}
SOURCES = ("seen", "bridged", "rejected", "hidden", "offscreen", "predicted", "unsure", "clamped")
# id_confidence = the output id's identity_confidence x this factor per source
SOURCE_ID_FACTOR = {"seen": 1.0, "bridged": 0.7, "rejected": 0.7, "hidden": 0.7, "offscreen": 0.7, "predicted": 0.4, "unsure": 0.3, "clamped": 0.5}


@dataclass
class TrackRow:
    frame: int
    t: float
    k: int
    track_id: int | str
    object_type: str  # player | referee | ball
    team: str  # A | B | referee | unknown ("" for the ball)
    x: float
    y: float
    confidence: float
    interpolated: bool  # players: filled across a short gap; ball: position estimated (not in a player's hands)
    calib: str  # keyframe | propagated
    box: tuple[float, float, float, float] | None = None  # image pixels of the detection (measured rows only)
    ball_state: str = ""  # possessed | estimated (ball rows only)
    ball_owner: int | None = None  # track id of the player holding the ball
    # per-sample confidence (smoothing.annotate_row; people only): 1 seen, 2 predicted on a visible spot,
    # 3 rejected outlier, 4 hidden behind a nearer person, 5 out of view; sigma = 1-sigma position error (m)
    pos_state: int | None = None
    pos_sigma: float | None = None
    predicted: bool = False  # lineup.predicted_rows: an out-of-view slot drifted by the motion prior
    unsure: bool = False  # lineup exact-5 mode 3: a 6th person of the team in live play (kept, not drawn)
    player_id: int | None = None  # continuous player id (lineup.player_id_of; survives output-id breaks)
    id_confidence: float | None = None  # identity confidence of this row (0..1), see SOURCE_ID_FACTOR
    clamped_from: str = ""  # source of the row before it was clamped
    clamped: bool = False  # position was > OFF_COURT_EXPORT_M outside the court and was moved to the edge

    def as_list(self) -> list[Any]:
        b = [round(v, 1) for v in self.box] if self.box is not None else ["", "", "", ""]
        return [self.frame, round(self.t, 3), self.track_id, self.object_type, self.team, round(self.x, 3),
                round(self.y, 3), round(self.confidence, 3), self.interpolated, self.calib, self.ball_state,
                "" if self.ball_owner is None else self.ball_owner, *b,
                "" if self.pos_state is None else int(self.pos_state),
                "" if self.pos_sigma is None else round(float(self.pos_sigma), 3),
                source_of(self), "" if self.player_id is None else int(self.player_id),
                "" if self.id_confidence is None else round(float(self.id_confidence), 3)]


def source_of(row: TrackRow) -> str:
    """seen | bridged | rejected | hidden | offscreen | predicted for person rows; "" for the ball."""
    if row.object_type == "ball":
        return ""
    if getattr(row, "clamped", False):
        return "clamped"
    if getattr(row, "unsure", False):
        return "unsure"
    if getattr(row, "predicted", False):
        return "predicted"
    st = getattr(row, "pos_state", None)
    if st in SOURCE_OF_STATE:
        return SOURCE_OF_STATE[int(st)]
    return "bridged" if row.interpolated else "seen"


OFF_COURT_EXPORT_M = 1.0  # no calibration-failure person position is exported further outside the lines than this
OFF_COURT_HARD_M = 3.0  # a measured person row this far outside the lines is a calibration failure whatever its calib


def clamp_off_court(rows: list[TrackRow], length: float | None = None, width: float | None = None,
                    margin: float = OFF_COURT_EXPORT_M, hard: float = OFF_COURT_HARD_M) -> int:
    """Move person rows that are far outside the court because of a calibration failure to the edge of the allowed
    area (``margin`` outside the lines) and flag them ``clamped`` (source "clamped"; the source the row had before
    is kept in ``row.clamped_from``). Only rows that can be a calibration failure are touched: players of team A / B
    whose frame has no keyframe calibration (propagated) or that lie more than ``hard`` metres outside. Referees,
    unsure/6th persons, rows with other teams (bench, coaches) and non-finite positions are never moved, so a
    throw-in passer 1.5-3 m behind the line keeps his real position. ``length``/``width`` default to the project court
    (``court.dimensions()``). Returns the number of rows moved."""
    if length is None or width is None:
        from replay import court
        dl, dw = court.dimensions()
        length = dl if length is None else length
        width = dw if width is None else width
    n = 0
    for r in rows:
        if r.object_type != "player" or r.team not in ("A", "B") or r.unsure or r.clamped:
            continue
        x0, y0 = float(r.x), float(r.y)
        if not (math.isfinite(x0) and math.isfinite(y0)):
            continue
        out = max(-x0, x0 - float(length), -y0, y0 - float(width))  # distance outside the lines (negative: inside)
        if out <= margin or (r.calib == "keyframe" and out <= hard):
            continue
        x = min(max(x0, -margin), float(length) + margin)
        y = min(max(y0, -margin), float(width) + margin)
        r.clamped_from = source_of(r)
        r.x, r.y, r.clamped = x, y, True
        r.box = None  # the detection box no longer belongs to this position
        r.pos_sigma = max(float(r.pos_sigma or 0.0), float(out))  # moved by up to `out` metres: the error is that big
        n += 1
    return n


def mark_predicted(rows: list[TrackRow]) -> list[TrackRow]:
    """Flag rows from lineup.predicted_rows (source "predicted"); returns the same list."""
    for r in rows:
        r.predicted = True
    return rows


def fill_identity_fields(rows: list[TrackRow], track_info: dict) -> None:
    """player_id (track_info "player_id", from lineup) and id_confidence (track_info "identity_confidence" x the
    source factor) for person rows; values already set on a row are kept."""
    for r in rows:
        if r.object_type == "ball":
            continue
        ti = track_info.get(r.track_id) or track_info.get(str(r.track_id)) or {}
        if r.player_id is None and ti.get("player_id") is not None:
            r.player_id = int(ti["player_id"])
        idc = ti.get("identity_confidence")
        if r.id_confidence is None and isinstance(idc, (int, float)):
            r.id_confidence = float(idc) * SOURCE_ID_FACTOR.get(source_of(r), 0.7)


def _write_text_atomic(path: Path, text: str, newline: str | None = None) -> None:
    """Write via a temporary file in the same folder + os.replace: readers (GUI, benchmark) never see a
    half-written file, and a crash leaves the previous version in place."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8", newline=newline) as fh:  # same newline handling as before
            fh.write(text)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def write_outputs(out: Path, rows: list[TrackRow], data: "Pass1Data", res: "ResolvedCalibration",
                  meta: "GameMeta", track_info: dict[int, dict], video: Path | None,
                  team_colors: dict[str, str] | None = None, ball_frames: "list[BallFrame] | None" = None,
                  events: list[dict] | None = None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fill_identity_fields(rows, track_info)
    rows = sorted(rows, key=lambda r: (r.k, str(r.track_id)))
    buf = io.StringIO(newline="")
    w = csv.writer(buf)
    w.writerow(COLUMNS)
    for r in rows:
        w.writerow(r.as_list())
    _write_text_atomic(out / "tracking.csv", buf.getvalue(), newline="")

    game = meta.to_public_dict()
    if team_colors:
        for label, c in team_colors.items():
            if label in game["teams"]:
                game["teams"][label]["primary"] = c
    doc = {
        "format": "advanced-analytist/tracking@2",
        "coordinate_system": {
            "units": "metres", "court": f"{court.SPEC} {court.LENGTH:g} x {court.WIDTH:g}",
            "origin": "left baseline x near sideline (as seen from the camera)",
            "x": f"along the sideline, 0..{court.LENGTH:g}", "y": f"along the baseline, 0..{court.WIDTH:g}",
            "length": court.LENGTH, "width": court.WIDTH,
        },
        "game": game,
        "process_fps": data.signature["process_fps"],
        "video": data.signature["video"],
        "tracks": {str(k): v for k, v in track_info.items()},
        "columns": COLUMNS,
        "rows": [r.as_list() for r in rows],
        "events": list(events or []),  # e.g. made baskets (replay/scoring.py)
    }
    _write_text_atomic(out / "tracking.json", json.dumps(doc, separators=(",", ":")))
    build_viewer(out, rows, data, res, game, track_info, video, ball_frames, events)


def _cm(v: float) -> int:
    return int(round(v * 100))


def viewer_payload(rows: list[TrackRow], data: "Pass1Data", res: "ResolvedCalibration", game: dict,
                   track_info: dict[int, dict], ball_frames: "list[BallFrame] | None" = None,
                   events: list[dict] | None = None) -> dict[str, Any]:
    n = len(data.k)
    # sample position of each analysed sample number k (k has gaps where breaks were skipped)
    pos_of_k = {int(k): i for i, k in enumerate(data.k)}
    # Each track's lists are indexed by sample: value j belongs to sample i0 + j. Built by position (not by
    # appending in row order), so a duplicate row or unsorted input can never shift a track or run it past the
    # last sample (that made replay_data.json disagree with tracking.csv). Duplicate (track, sample): the
    # measured row wins, else the first one.
    by_track: dict[Any, dict[int, TrackRow]] = {}
    for r in rows:
        if r.object_type == "ball" or getattr(r, "unsure", False):
            continue  # unsure rows (lineup exact-5 mode 3) stay in the CSV but are not drawn
        i = pos_of_k.get(int(r.k))
        if i is None:
            continue
        slot = by_track.setdefault(r.track_id, {})
        old = slot.get(i)
        if old is None or (old.interpolated and not r.interpolated):
            slot[i] = r
    tracks: dict[Any, dict[str, Any]] = {}
    for tid, slot in by_track.items():
        ii = sorted(slot)
        i0, m = ii[0], ii[-1] - ii[0] + 1
        first = slot[i0]
        tr = {"id": tid, "team": first.team, "type": first.object_type, "i0": i0,
              "x": [None] * m, "y": [None] * m, "f": [0] * m, "b": [None] * m}
        pid = next((r.player_id for r in slot.values() if r.player_id is not None), None)
        if pid is None:
            pid = (track_info.get(tid) or track_info.get(str(tid)) or {}).get("player_id")
        if pid is not None:
            tr["pid"] = int(pid)  # continuous player id (gui/motion.py labels by it)
        has_conf = any(r.pos_state is not None for r in slot.values())
        if has_conf:  # per-sample confidence for the UI (gui/motion.py): "st" state, "u" sigma in centimetres
            tr["st"] = [0] * m
            tr["u"] = [None] * m
        for i in ii:
            r, j = slot[i], i - i0
            tr["x"][j] = _cm(r.x)  # centimetres
            tr["y"][j] = _cm(r.y)
            tr["f"][j] = 2 if r.interpolated else 1
            tr["b"][j] = None if r.box is None else [int(round(v)) for v in r.box]
            if has_conf:
                tr["st"][j] = 0 if r.pos_state is None else int(r.pos_state)
                tr["u"][j] = None if r.pos_sigma is None else _cm(r.pos_sigma)
        tracks[tid] = tr
    ball = None
    if ball_frames:
        ball = {"x": [], "y": [], "f": [], "b": [], "o": []}
        for bf in ball_frames:
            ok = bf.x is not None and res.H[bf.i] is not None
            ball["x"].append(_cm(bf.x) if ok else None)
            ball["y"].append(_cm(bf.y) if ok else None)
            ball["f"].append({"possessed": 1, "estimated": 2, "guessed": 3}.get(bf.state, 0) if ok else 0)  # 3: ball.py GUESSED (display only)
            ball["b"].append(None if bf.box is None else [int(round(v)) for v in bf.box])
            ball["o"].append(bf.owner if ok else None)
    calib_code = {NONE: 0, KEYFRAME: 2}
    return {
        "game": game,
        "fps": data.signature["process_fps"],
        "t": [round(float(t), 3) for t in data.t],
        "frame": [int(f) for f in data.frame],
        "calib": [calib_code.get(q, 1) for q in res.quality],
        "court": {"length": court.LENGTH, "width": court.WIDTH,
                  "lines": [np.round(l, 3).tolist() for l in court.court_polylines()],
                  "baskets": court.baskets()},
        "tracks": list(tracks.values()),
        "ball": ball,
        "track_info": {str(k): v for k, v in track_info.items()},
        "events": list(events or []),
        "n": n,
    }


def build_viewer(out: Path, rows: list[TrackRow], data: "Pass1Data", res: "ResolvedCalibration",
                 game: dict, track_info: dict[int, dict], video: Path | None,
                 ball_frames: "list[BallFrame] | None" = None, events: list[dict] | None = None) -> Path:
    payload = viewer_payload(rows, data, res, game, track_info, ball_frames, events)
    if video is None:
        video = Path(data.signature["video"])
    try:
        src = os.path.relpath(Path(video).resolve(), out.resolve()).replace(os.sep, "/")
    except ValueError:  # different drive on Windows
        src = Path(video).resolve().as_uri()
    payload["video_src"] = src
    text = json.dumps(payload, separators=(",", ":"))  # serialised once: same text for the file and the HTML
    _write_text_atomic(out / "replay_data.json", text)
    template = (Path(__file__).parent / "viewer" / "template.html").read_text(encoding="utf-8")
    blob = text.replace("</", "<\\/")
    html = template.replace("/*__DATA__*/null", blob)
    path = out / "replay.html"
    _write_text_atomic(path, html)
    return path
