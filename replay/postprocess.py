"""Pass 2: from pass-1 detections to court positions, teams, ball and outputs. Fast; no video decoding.

Steps
  1. calibration per frame (keyframes combined through camera motion)
  2. project every person's foot point onto the court; drop feet that are hidden or far off court
  3. court-space tracking (occlusion-robust IDs, see court_tracker.py)
  4. track-level filtering: bench, coaches and spectators out
  5. team classification (+ optional A/B swap), at most 5 per team and 3 referees per frame
  6. smoothing (short gaps interpolated and flagged)
  7. ball tracking and ball court positions; made baskets from the ball through the rim (scoring.py)
  8. CSV/JSON/HTML/report
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from replay import calibration as calib_mod
from replay import court, footpoint, homography
from replay.ball import boxes_by_frame, estimate_ball, holders_from_rows
from replay.config import Config
from replay.consolidate import consolidate_drop  # noqa: F401 (kept for other callers)
from replay.dedup import best_per_sample
from replay import limits
from replay.court_tracker import reattach_enabled, track_people
from replay import tracklets
from replay.detection import BALL, PERSON
from replay.export import TrackRow, write_outputs
from replay.metadata import GameMeta
from replay.report import build_report
from replay.roster import filter_players, limit_shown_rows
from replay.scoring import detect_baskets, rim_tracks
from replay import smoothing
from replay.smoothing import smooth_track
from replay import team_binding
from replay.teams import A, B, REF, UNKNOWN, appearance_for, classify, split_mixed_tracks


def _overlay_boxes(meta: GameMeta, size: tuple[int, int]) -> np.ndarray:
    w, h = size
    return np.array([[r[0] * w, r[1] * h, r[2] * w, r[3] * h] for r in meta.overlay_regions]).reshape(-1, 4)


def run_pass2(out: Path, cfg: Config, meta: GameMeta, calib_path: Path, video_for_viewer: Path | None = None,
              swap_teams: bool = False) -> dict:
    from replay.pipeline import load_pass1  # local import: pipeline imports this module

    data = load_pass1(out / "work")
    calib = calib_mod.CalibrationFile.load(calib_path)
    size = calib.frame_size
    fps = float(data.signature["process_fps"])
    n = len(data.k)
    idx_of_k = {int(k): i for i, k in enumerate(data.k)}

    res = calib_mod.resolve(data.t, data.motion, calib, cfg.calibration)
    from replay import team_appearance as _ta  # broadcast vs gym camera -> kit descriptor mode (identity agent)
    _ta.set_broadcast_mode(res.H, calib.frame_size)
    for w in res.warnings:
        print("calibration:", w)
    from replay import redetect  # 1b. second-look detector on crops where a player went missing (redetect.py)
    data = redetect.augment(out / "work", data, res, video_for_viewer, cfg, size)
    data = __import__("replay.balldet", fromlist=["x"]).augment(out / "work", data, video_for_viewer)  # REPLAY_BALLDET=1
    data = redetect.fill_crumbs(data, res, cfg)  # 1c. Task-9 low-score rows: gap fillers only (none stored = no-op)

    # ------------------------------------------------------------------ 2. project people
    rows = data.rows
    overlay = _overlay_boxes(meta, size)
    det_row = np.flatnonzero(rows[:, 6] == PERSON)
    det_i = np.array([idx_of_k[int(k)] for k in rows[det_row, 0]], int)
    # robust foot points + per-detection covariance (footpoint.py: airborne players, cut boxes, glitches)
    det_row, det_i, det_xy, det_R, det_foot_flag = footpoint.project_people(rows, det_row, det_i, res.H, size, overlay,
                                                                          cfg, fps, data.motion)
    r = rows[det_row]
    det_feat = r[:, 8:]
    det_bt = r[:, 7].astype(int)
    det_conf = r[:, 5].astype(float)
    det_box = r[:, 1:5].reshape(-1, 4)

    # jersey numbers (jersey_numbers.py): cached reads per pass-1 row; identity cue for linking + track labels
    from replay import jersey_numbers
    numbers = jersey_numbers.load_or_compute(out / "work", data, video_for_viewer)
    det_num = None if numbers is None else numbers.code[det_row].astype(int)

    # ------------------------------------------------------------------ 3. court tracking
    from replay import corrections  # the coach's corrections ledger (Config.corrections, off by default)
    ledger_doc = corrections.load_if_enabled(Path(out).parent, cfg)  # project folder (game-state rebuilds: same)
    ledger = corrections.resolve_identity(ledger_doc, data.t, det_i, det_xy)
    if ledger_doc:
        print(f"corrections: {len(ledger_doc['ops'])} ops"
              + (f", {len(ledger.unresolved)} identity ops without a detection" if ledger else ""))
    from replay import parallel_pass2  # S4: steps 3 + 4 per game segment in a process pool (off by default)
    par = None
    if parallel_pass2.enabled(cfg) and not ledger:  # identity corrections need the whole game in one linker
        par = parallel_pass2.track_and_filter(
            out, data, cfg, fps, size, res.H, det_i, det_xy, det_R, det_feat, det_bt, det_conf, det_box, det_num,
            tracklets.cues_for(out / "work", data, video_for_viewer, cfg, det_row, meta), reattach_enabled(cfg.smoothing))
    if par is not None:
        tid, det_xy, det_R, id_info, roster = par
        keep = tid > 0
        fc = cfg.filter
    else:
        tid, det_xy, det_R = track_people(det_i, det_xy, det_R, det_feat, det_bt, det_conf, fps, cfg.court_tracking,
                                          boxes=det_box, motion=data.motion, n_samples=n, H=res.H, numbers=det_num,
                                          size=size, info=(id_info := {}),  # identity cues + confidence (tracklets.py)
                                          reattach=reattach_enabled(cfg.smoothing),  # field + env REPLAY_REATTACH
                                          ledger=ledger,  # corrections ledger (None = no effect)
                                          cues=tracklets.cues_for(out / "work", data, video_for_viewer, cfg, det_row,
                                                                  meta))
        keep = tid > 0

        # -------------------------------------------------------------- 4. who is a player? (roster.py)
        fc = cfg.filter
        roster = filter_players(data, det_i, det_box, det_feat, tid, keep, det_xy, fps, size, fc, H=res.H)
        if getattr(fc, "clip_bench", False) or os.environ.get("AA_CLIP_BENCH") == "1":  # identity agent: CLIP bench cue
            from replay import clip_roles
            clip_roles.bench_refine(out / "work", data, video_for_viewer, det_row, det_i, det_xy, tid, keep, roster, fps)
    keep &= roster.keep
    dropped_reason = roster.dropped
    if dropped_reason:
        print("filtered tracks:", dropped_reason)

    # ------------------------------------------------------------------ 5. teams + per-frame caps
    app = appearance_for(out / "work", data, video_for_viewer, cfg.teams, np.array(det_row, int)[keep], meta)
    teams = classify(tid[keep], det_i[keep], det_feat[keep], meta, cfg.teams, appearance=app, xy=det_xy[keep])
    tid = split_mixed_tracks(tid, keep, det_i, teams, cfg.teams)  # identity switches between kits (teams.py)
    # stable A/B: bound to the jersey colours stored in meta.json; swap_teams toggles the binding (team_binding.py)
    teams, meta, _ = team_binding.bind(out, meta, teams, swap_teams, video_for_viewer)
    team_of = lambda t: teams.track_team.get(int(t), UNKNOWN)  # noqa: E731
    keep &= ~roster.hide  # bench crowd: in the team colours, never shown (roster.py)
    from replay import lineup  # 5 + 5 continuous on-court slots: track ids -> slot ids, extras out (lineup.py)
    tid_before_lineup = tid.copy()
    tid, keep = lineup.apply(out, tid, keep, det_i, det_xy, det_feat, det_num, data.t, teams, rank=roster.rank,
                             det_row=det_row)
    if ledger and (id_info or {}).get("ledger_labels"):  # a slot must not join two sides of a coach's split
        from replay.court_tracker import _undo_ledger_conflicts
        tid, n_undone = _undo_ledger_conflicts(tid_before_lineup, tid, id_info["ledger_labels"])
        if n_undone:
            print(f"corrections: {n_undone} lineup joins undone (they broke the corrections ledger)")
    _ut, _uc = np.unique(tid[keep], return_counts=True)  # speed: one pass instead of a mask per track
    score = {int(t): c * teams.track_share.get(int(t), 0.5) for t, c in zip(_ut, _uc)}
    # per-detection priority for the caps: player score + deep-inside-court bonus (roster.cap_rank); ranking by
    # track length instead let long bench/sideline tracks push out players seen right now
    rank = roster.rank if roster.rank is not None else np.zeros(len(tid))
    caps = [((A, B), fc.max_players), ((REF,), fc.max_referees)]
    order = np.argsort(det_i)
    sorted_i = det_i[order]  # speed: each frame's detections are one contiguous slice of `order` (same order)
    for i in np.unique(det_i[keep]):
        sel = order[np.searchsorted(sorted_i, i):np.searchsorted(sorted_i, i, side="right")]
        sel = sel[keep[sel]]
        for labels, cap in caps:
            mine = [d for d in sel if team_of(tid[d]) in labels]
            if len(mine) > cap:
                mine.sort(key=lambda d: (-rank[d], -score[int(tid[d])]))
                keep[mine[cap:]] = False

    # ------------------------------------------------------------------ 6. smoothing -> rows
    hidden = smoothing.hidden_classifier(res.H, rows, idx_of_k, size, overlay, cfg.smoothing, t=data.t)  # occlusion cues
    first_seen: dict[int, int] = {}
    for t, i in zip(tid[keep], det_i[keep]):
        first_seen[int(t)] = min(first_seen.get(int(t), i), i)
    new_id = {t: j + 1 for j, t in enumerate(sorted(first_seen, key=lambda t: first_seen[t]))}
    out_rows: list[TrackRow] = []
    row_rank: dict[int, float] = {}  # id(row) -> rank of its detection (measured rows)
    players_at: list[list[tuple[int, np.ndarray, np.ndarray]]] = [[] for _ in range(n)]
    _kidx = np.flatnonzero(keep)  # speed: detections of each track as ascending index arrays (== the old masks)
    _ko = np.argsort(tid[_kidx], kind="stable")
    _kt, _ks = np.unique(tid[_kidx][_ko], return_index=True)
    _rows_of = dict(zip(_kt.tolist(), np.split(_kidx[_ko], _ks[1:])))
    _order = sorted(first_seen, key=lambda t: first_seen[t])
    _m_of = {}
    for t in _order:
        m = _rows_of[t]
        _m_of[t] = m[best_per_sample(det_i[m], det_conf[m], det_R[m])]  # one detection per sample: the best (dedup.py)
    # speed: every track smoothed up front (a process pool on long clips; same code and inputs -> identical segments)
    _segs = lineup.smooth_slots([(t, det_i[_m_of[t]], det_xy[_m_of[t]], det_R[_m_of[t]]) for t in _order], fps,
                                cfg.smoothing, hidden=hidden, n_frames=n)
    for t, _segs_t in zip(_order, _segs):
        m = _m_of[t]
        ii, xy_t, R_t, conf_t, box_t = det_i[m], det_xy[m], det_R[m], det_conf[m], det_box[m]
        conf_at = dict(zip(ii.tolist(), conf_t.tolist()))
        box_at = dict(zip(ii.tolist(), box_t))
        rank_at = dict(zip(ii.tolist(), rank[m].tolist()))
        team = team_of(t)
        mean_conf = float(np.mean(conf_t))
        for seg in _segs_t:  # lineup.smooth_slot(t, ii, xy_t, R_t, fps, cfg.smoothing, hidden=hidden) = smooth_track
            for i, p, measured in zip(seg.k, seg.xy, seg.measured):
                i = int(i)
                if res.H[i] is None:
                    continue  # never output a position in a frame without calibration
                box = box_at.get(i) if measured else None
                out_rows.append(TrackRow(
                    frame=int(data.frame[i]), t=float(data.t[i]), k=int(data.k[i]), track_id=new_id[t],
                    object_type="referee" if team == REF else "player", team=team, x=float(p[0]), y=float(p[1]),
                    confidence=float(conf_at.get(i, 0.5 * mean_conf)) if measured else 0.5 * mean_conf,
                    interpolated=not measured, calib=res.quality[i],
                    box=None if box is None else tuple(float(v) for v in box)))
                smoothing.annotate_row(out_rows[-1], seg, i)  # pos_state / pos_sigma for the UI (smoothing.py)
                if measured:
                    row_rank[id(out_rows[-1])] = rank_at[i]
                if box is not None and team in (A, B):
                    players_at[i].append((new_id[t], box, p))

    # final per-frame limit incl. filled-in positions: basketball has 10 players and 2-3 referees on court.
    # Detections go first, most player-like first: a filled-in gap of a long track must not push out a player
    # who is seen right now (that was the main way visible players got lost). Filled-in rows: fixed track order.
    out_rows += __import__("replay.export", fromlist=["x"]).mark_predicted(  # Task 8: source "predicted"
        lineup.predicted_rows(out_rows, new_id, data, res.H, det_i, det_xy, tid, keep))  # out-of-view slots
    strength = {new_id[t]: score[t] for t in first_seen}
    by_frame: dict[int, list[TrackRow]] = {}
    for r in out_rows:
        by_frame.setdefault(r.k, []).append(r)
    # reality check per track SEGMENT (limits.py): ghosts of split ids, duplicates, team-less / off-court runs, caps
    # of 5 per team and 3 referees - whole runs, never single rows after smoothing (that made dots blink)
    why: dict = {}
    if limits.enabled(fc):
        drop = limits.final_limits(out_rows, row_rank, strength, fps, fc, A, B, REF,
                                   present={int(data.k[i]) for i in range(n) if res.H[i] is not None}, reasons=why)
    else:  # old per-frame limits (A/B reference)
        drop = limit_shown_rows(out_rows, (((A, B), fc.max_players_shown), ((REF,), fc.max_referees)), row_rank,
                                strength)
        drop |= consolidate_drop([r for r in out_rows if id(r) not in drop], strength, A, B, REF, rank=row_rank)
    if why:
        from collections import Counter
        print("final limits (rows dropped):", dict(Counter(why.values())), "of", len(out_rows))
    out_rows = [r for r in out_rows if id(r) not in drop]
    n_cl = __import__("replay.export", fromlist=["x"]).clamp_off_court(out_rows, *court.dimensions())  # project court; calibration failures only
    if n_cl:
        print(f"off-court guard: {n_cl} person rows moved to the court edge (source 'clamped')")
    players_at = [[pa for pa in pl if any(r.track_id == pa[0] and id(r) not in drop for r in by_frame.get(int(data.k[i]), []))]
                  for i, pl in enumerate(players_at)]

    # ------------------------------------------------------------------ 7. ball
    G, seg_id = calib_mod._segment_transforms(data.motion, n)
    balls = rows[rows[:, 6] == BALL]
    ball_rows = [np.zeros((0, 5)) for _ in range(n)]
    ball_i = np.array([idx_of_k[int(k)] for k in balls[:, 0]], int)
    for i in np.unique(ball_i):
        ball_rows[int(i)] = balls[ball_i == i][:, [1, 2, 3, 4, 5]]
    # ball.py / ball_track.py: image trajectory, holders (players + referees), rims, untracked people's feet
    rims, cams = rim_tracks(res.H, data.t, size, cfg.scoring, G, seg_id)
    holders, track_xy = holders_from_rows(out_rows, idx_of_k, n)
    persons = boxes_by_frame(rows[rows[:, 6] == PERSON], data.k)
    ball_frames = estimate_ball(n, ball_rows, G, seg_id, fps, cfg.ball, holders, track_xy, rims, persons, res.H)
    for bf in ball_frames:
        if bf.x is None or res.H[bf.i] is None:
            continue
        out_rows.append(TrackRow(
            frame=int(data.frame[bf.i]), t=float(data.t[bf.i]), k=int(data.k[bf.i]), track_id="ball",
            object_type="ball", team="", x=bf.x, y=bf.y, confidence=bf.conf if bf.conf else 0.3,
            interpolated=bf.state != "possessed", calib=res.quality[bf.i], box=bf.box, ball_state=bf.state,
            ball_owner=bf.owner))

    # ------------------------------------------------------------------ 7b. made baskets (scoring.py)
    from replay import rimlook  # 7a. second look at the rims after shot attempts (rimlook.py, off unless REPLAY_RIMLOOK=1)
    ball_rows = rimlook.augment(out / "work", data, ball_rows, rims, video_for_viewer, cfg, size, ball_frames=ball_frames)
    events = detect_baskets(res.H, data.t, data.frame, ball_rows, ball_frames,
                            {new_id[t]: team_of(t) for t in first_seen}, size, fps, cfg.scoring, G=G, seg=seg_id,
                            quality=res.quality, rims=rims, cams=cams)
    if events:
        print("baskets:", ", ".join(f"{e['t']:.1f}s {e['basket']} team={e['team']} ({e['confidence']:.2f})"
                                    for e in events))

    shown = {r.track_id for r in out_rows if r.object_type != "ball"}  # tracks the per-frame caps removed entirely
    track_info = {new_id[t]: {"team": team_of(t), "team_confidence": round(teams.track_share.get(t, 0.0), 3)}
                  for t in first_seen if new_id[t] in shown}
    tracklets.add_identity_confidence(track_info, id_info, tid, keep, new_id)  # "identity_confidence" 0..1
    if det_num is not None:  # accepted jersey number per identity ("number", "number_reads")
        nums = jersey_numbers.identity_numbers(tid, numbers.label_codes()[det_row].astype(int), det_i, team_of,
                                               team_lab=teams.centroids,
                                               read_lab=None if numbers.lab is None else numbers.lab[det_row])
        for t in first_seen:
            if new_id[t] in track_info:
                track_info[new_id[t]].update(jersey_numbers.track_info_fields(nums.get(int(t))))
        print(f"jersey numbers: {sum('number' in v for v in track_info.values())} of {len(track_info)} tracks",
              numbers.stats)
    lineup.add_track_info(track_info, new_id)  # slots A1..B5 + player profiles (team + number) (lineup.py)
    track_info, _ = __import__("replay.handover", fromlist=["x"]).stitch(out_rows, track_info, events)  # handover.py
    write_outputs(out, out_rows, data, res, meta, track_info, video_for_viewer, team_colors=teams.colors,
                  ball_frames=ball_frames, events=events)
    __import__("replay.ball_events", fromlist=["x"]).save_motion(out, G, seg_id)  # pan-free ball speeds (ball_events.py)
    from replay import scorebar  # broadcast score bar -> work/scorebar.json (cached; game_state carries it)
    scorebar.load_or_compute(out / "work", video_for_viewer or data.signature.get("video"))
    from replay.game_state import after_pass2  # 7c. possessions, attack direction, game events
    gs_doc = after_pass2(out, ball_rows, rims, data.t, corrections=ledger_doc)
    return build_report(out, data, res, out_rows, teams, meta, cfg, ball_frames=ball_frames,
                        filtered=dropped_reason, events=events, game_state=gs_doc)
