"""Parallel pass 2 by game segment (SPEED_PLAN S4).

Court tracking (court_tracker.track_people) and the roster filter (roster.filter_players) are the two biggest CPU
stages of pass 2 and only look at time-local evidence. The game is split at long dead balls / breaks (>= 20 s
without live play: preliminary-analysis activity below its quiet threshold, or holes in the analysed samples), each
segment is tracked + filtered in its own process, and the per-detection results are stitched back (identity ids
made unique per segment). Everything after that (teams, lineup slots + profiles, jersey numbers, smoothing, ball,
scoring, game state) stays global in postprocess.py. Identities never cross a cut: at a >= 20 s dead ball the
lineup step joins them again (substitutions happen there anyway).

Off by default (Config.parallel_pass2; env REPLAY_PARALLEL_P2=1/0 overrides). With a single segment the serial
code path runs unchanged (byte-identical output).
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import fields, is_dataclass, replace
from pathlib import Path

import numpy as np

MIN_DEAD_S = 20.0  # a cut needs at least this long without live play
MIN_SEGMENT_S = 60.0  # no segment shorter than this (cuts too close together are merged)


def enabled(cfg) -> bool:
    env = os.environ.get("REPLAY_PARALLEL_P2")
    if env:
        return env.strip() not in ("0", "false", "no")
    return bool(getattr(cfg, "parallel_pass2", False))


def _quiet_cuts(work: Path) -> list[float]:
    """Middle times of runs of the preliminary analysis' activity trace below its quiet threshold, >= MIN_DEAD_S."""
    p = Path(work).parent / "inplay.json"
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        tr = d["trace"]
        act = np.asarray(tr["activity"], float)
        t0, dt, thr = float(tr["t0"]), float(tr["dt"]), float(d["quiet_threshold"])
    except Exception:  # noqa: BLE001 - no preliminary analysis: only sample holes
        return []
    quiet = act < thr
    cuts, i = [], 0
    while i < len(quiet):
        if quiet[i]:
            j = i
            while j < len(quiet) and quiet[j]:
                j += 1
            if (j - i) * dt >= MIN_DEAD_S:
                cuts.append(t0 + (i + j) / 2 * dt)
            i = j
        else:
            i += 1
    return cuts


def _activity(work: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        d = json.loads((Path(work).parent / "inplay.json").read_text(encoding="utf-8"))
        a = np.asarray(d["trace"]["activity"], float)
        return float(d["trace"]["t0"]) + np.arange(len(a)) * float(d["trace"]["dt"]), a
    except Exception:  # noqa: BLE001
        return np.zeros(0), np.zeros(0)


def segments(work: Path, t: np.ndarray) -> list[tuple[int, int]]:
    """Sample index ranges [i0, i1) of the game segments."""
    t = np.asarray(t, float)
    n = len(t)
    if n == 0:
        return []
    cut_i = set(int(i) for i in np.flatnonzero(np.diff(t) >= MIN_DEAD_S) + 1)  # holes (skipped breaks)
    for tc in _quiet_cuts(work):
        i = int(np.searchsorted(t, tc))
        if 0 < i < n:
            cut_i.add(i)
    force = int(os.environ.get("REPLAY_P2_FORCE_SEGMENTS", "0") or 0)  # test / A-B only: N parts at quiet seconds
    if force > 1 and not cut_i:
        act_t, act = _activity(work)
        for j in range(1, force):
            tc = t[0] + (t[-1] - t[0]) * j / force
            if len(act):
                near = np.flatnonzero(np.abs(act_t - tc) <= 30.0)
                if len(near):
                    tc = float(act_t[near[np.argmin(act[near])]])
            cut_i.add(int(np.searchsorted(t, tc)))
    bounds = [0]
    for i in sorted(cut_i):
        if t[i] - t[bounds[-1]] >= MIN_SEGMENT_S and t[n - 1] - t[i] >= MIN_SEGMENT_S:
            bounds.append(i)
    bounds.append(n)
    return [(a, b) for a, b in zip(bounds[:-1], bounds[1:]) if b > a]


def _slice_cues(cues, m: np.ndarray):
    if cues is None:
        return None
    if is_dataclass(cues):
        kw = {}
        for f in fields(cues):
            v = getattr(cues, f.name)
            kw[f.name] = v[m] if isinstance(v, np.ndarray) and len(v) == len(m) else v
        return replace(cues, **kw)
    return cues


def _seg_job(args):
    (seg_data, det_i, det_xy, det_R, det_feat, det_bt, det_conf, det_box, det_num, H, cues, cfg, fps, size,
     reattach) = args
    from replay.court_tracker import track_people
    from replay.roster import filter_players

    t0 = time.perf_counter()
    info: dict = {}
    tid, xy, R = track_people(det_i, det_xy, det_R, det_feat, det_bt, det_conf, fps, cfg.court_tracking,
                              boxes=det_box, motion=seg_data.motion, n_samples=len(seg_data.k), H=H, numbers=det_num,
                              size=size, info=info, reattach=reattach, cues=cues)
    t1 = time.perf_counter()
    keep = tid > 0
    roster = filter_players(seg_data, det_i, det_box, det_feat, tid, keep, xy, fps, size, cfg.filter, H=H)
    return tid, xy, R, info.get("det_confidence"), roster, (t1 - t0, time.perf_counter() - t1)


def _merge_roster(parts, masks, n_det: int):
    from replay.roster import RosterResult

    def full(name, fill, dtype=None):
        arrs = [getattr(r, name) for r in parts]
        if any(a is None for a in arrs):
            return None
        a0 = np.asarray(arrs[0])
        out = np.full((n_det,) + a0.shape[1:], fill, dtype=dtype or a0.dtype)
        for a, m in zip(arrs, masks):
            out[m] = a
        return out

    def full_dict(name):
        keys = set().union(*(getattr(r, name).keys() for r in parts))
        out = {}
        for k in keys:
            a0 = next(np.asarray(getattr(r, name)[k]) for r in parts if k in getattr(r, name))
            arr = np.full((n_det,) + a0.shape[1:], np.nan if a0.dtype.kind == "f" else 0, a0.dtype)
            for r, m in zip(parts, masks):
                if k in getattr(r, name):
                    arr[m] = getattr(r, name)[k]
            out[k] = arr
        return out

    dropped: dict[str, int] = {}
    for r in parts:
        for k, v in (r.dropped or {}).items():
            dropped[k] = dropped.get(k, 0) + int(v)
    kw = dict(keep=full("keep", False, bool), score=full("score", np.nan, float),
              evidence=full_dict("evidence"), raw=full_dict("raw"), reason=full("reason", "", object),
              dropped=dropped, rank=full("rank", 0.0, float), hide=full("hide", False, bool))
    known = {f.name for f in fields(RosterResult)}
    return RosterResult(**{k: v for k, v in kw.items() if k in known})


def track_and_filter(out: Path, data, cfg, fps: float, size, H, det_i, det_xy, det_R, det_feat, det_bt, det_conf,
                     det_box, det_num, cues, reattach):
    """Steps 3 + 4 of pass 2 per segment in a process pool. Returns (tid, det_xy, det_R, id_info, roster) like the
    serial path, or None when there is only one segment (the caller then runs the serial code)."""
    from replay.pipeline import Pass1Data

    segs = segments(Path(out) / "work", data.t)
    if len(segs) < 2:
        return None
    rows = data.rows
    rk = rows[:, 0]
    jobs, masks = [], []
    for i0, i1 in segs:
        m = (det_i >= i0) & (det_i < i1)
        k0, k1 = int(data.k[i0]), int(data.k[i1 - 1])
        rsel = (rk >= k0) & (rk <= k1)
        motion = list(data.motion[i0:i1])
        if motion:
            motion[0] = None  # a segment starts a new camera chain
        seg_data = Pass1Data(k=data.k[i0:i1], frame=data.frame[i0:i1], t=data.t[i0:i1], motion=motion,
                             inliers=data.inliers[i0:i1], rows=rows[rsel], signature=data.signature)
        jobs.append((seg_data, det_i[m] - i0, det_xy[m], det_R[m], det_feat[m], det_bt[m], det_conf[m], det_box[m],
                     None if det_num is None else det_num[m], list(H[i0:i1]), _slice_cues(cues, m), cfg, fps, size,
                     reattach))
        masks.append(m)
    n_det = len(det_i)
    t0 = time.perf_counter()
    workers = max(1, min(len(jobs), (os.cpu_count() or 2) - 2))
    with ProcessPoolExecutor(workers) as ex:
        results = list(ex.map(_seg_job, jobs))
    tid = np.zeros(n_det, int)
    xy = np.array(det_xy, float, copy=True)
    R = np.array(det_R, float, copy=True)
    dc = np.full(n_det, np.nan)
    have_dc = True
    offset = 0
    for (t_s, xy_s, R_s, dc_s, _r, _tm), m in zip(results, masks):
        t_s = np.asarray(t_s, int)
        tid[m] = np.where(t_s > 0, t_s + offset, 0)
        offset += int(t_s.max(initial=0))
        xy[m], R[m] = xy_s, R_s
        if dc_s is None:
            have_dc = False
        else:
            dc[m] = dc_s
    roster = _merge_roster([r for _a, _b, _c, _d, r, _e in results], masks, n_det)
    times = [tm for *_x, tm in results]
    print(f"parallel pass 2: {len(segs)} segments on {workers} processes, steps 3+4 in "
          f"{time.perf_counter() - t0:.1f} s (serial sum {sum(a + b for a, b in times):.1f} s)")
    return tid, xy, R, ({"det_confidence": dc} if have_dc else {}), roster
