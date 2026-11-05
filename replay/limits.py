"""Final reality check of the people drawn on the court, decided per track SEGMENT, never per single row
(tracking-fix plan, Task 2; replaces roster.limit_shown_rows + consolidate.consolidate_drop in pass 2).

Deleting single rows after smoothing (frame by frame: cap of 10, "ghost" next to a teammate, duplicate, off court)
made dots blink: a row survived in sample k, was deleted in k+1 and came back in k+2. Here every rule looks at runs
of consecutive samples (samples without calibration are not breaks):

  1. ghost: a filled-in tail of a track (no measurement of it follows) is dropped from the moment another track of
     the same team STARTS next to it - the tracker split one person into two ids. Filled rows that bridge to a
     later measurement of the same track (a player hidden behind a teammate) are always kept;
  2. duplicate: two seen tracks of one team whose boxes overlap (IoU > 0.5) in most samples of their overlap are
     one person: the weaker track goes for the whole overlap;
  3. no team / off court: people of no team are shown only seen and inside the lines; off-court runs shorter than
     ``FilterConfig.offcourt_run_s`` inside a track are foot-point noise and are kept (a frame where most seen
     people are far off court has a shifted calibration: player-like rows stay, see consolidate.py);
  4. caps (5 per team, 3 referees, team-less people only in free player places): for every run of samples where
     a group is over its cap, whole tracks are evicted for the whole run, lowest run-rank first; a candidate whose
     eviction would cut a hole shorter than ``FilterConfig.min_evict_s`` into its track goes last (hysteresis).
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from replay import court
from replay.consolidate import (GHOST_M, SAME_PERSON_M, OFF_COURT_M, PER_TEAM, REFEREES, SHIFT_MIN_RANK, SHIFT_MIN_SEEN,
                                SHIFT_SHARE, _iou, partial_rows)

DUP_IOU = 0.5
DUP_SHARE = 0.6
FILLED_SCORE = -5.0
FILL_MAX_SIGMA_M = 1.0  # filled-in rows more uncertain than this are not drawn
HOLE_FILL = 3  # cap drops of at most this many samples inside a kept track are undone (no blinking)
ISLAND = 2  # kept pieces of at most this many samples between cap drops are dropped too
JOIN_GAP = 2  # over-cap rows of one track at most this many samples apart are evicted together (no flicker)
SHORT_HOLE_BONUS = 1.0  # rank bonus of a candidate whose eviction would cut a hole < min_evict_s into its track  # cap rank of a filled-in row (roster ranks of seen people are ~0..7)


def final_limits(out_rows, row_rank, strength, fps, cfg, team_a="A", team_b="B", ref="referee", present=None,
                 reasons=None) -> set[int]:
    """pass-2 entry: segment rules (FilterConfig.segment_rules) or the smoothed per-frame limits (default)."""
    pre: set[int] = set()
    if _env_on("REPLAY_MERGE_DOUBLES", MERGE_DOUBLES):
        why0 = reasons if reasons is not None else {}
        pre = merge_double_ids(out_rows, row_rank, reasons=why0)
        out_rows = [r for r in out_rows if id(r) not in pre]
    if _env_on("REPLAY_BENCH_STRIP", BENCH_STRIP):
        bs = bench_strip_rows(out_rows, reasons=reasons if reasons is not None else {}, ref=ref)
        pre |= bs
        out_rows = [r for r in out_rows if id(r) not in bs]
    import os

    env = os.environ.get("REPLAY_SEGMENT_RULES", "").strip()
    rules = env not in ("0", "false", "no") if env else bool(getattr(cfg, "segment_rules", False))
    if rules:
        return pre | apply_segment_limits(out_rows, row_rank, strength, fps, cfg, team_a, team_b, ref, present, reasons)
    return pre | smoothed_frame_limits(out_rows, row_rank, strength, cfg, team_a, team_b, ref, present, reasons,
                                       fps=fps)


MERGE_DOUBLES = False  # two output ids on one person at the same time -> one id (env REPLAY_MERGE_DOUBLES)
DOUBLE_M = 0.8  # positions this close ...
DOUBLE_SHARE = 0.7  # ... in this share of the overlap (at least DOUBLE_MIN samples) = one person
DOUBLE_MIN = 3

# bench strip (data/plans/bench_guard.md item (c)): an output id whose MEASURED feet all lie within BENCH_STRIP_IN_M
# inside or anywhere beyond a sideline - and that therefore never entered the court - is a bench person in team kit
# (c300 x 19-21 m, p4400 x 21-23 m: 2-6 row ids the whole-life bench guard cannot score) and is not drawn at all.
# Rows near a baseline corner (inbounder) count as court rows. Referees are never touched. env REPLAY_BENCH_STRIP=1.
BENCH_STRIP = False
BENCH_STRIP_IN_M = 0.5
BENCH_STRIP_CORNER_M = 4.0
BENCH_STRIP_SHARE = 1.0  # share of measured rows that must be strip rows (env REPLAY_BENCH_STRIP_SHARE; 1.0 = all)
BENCH_STRIP_MAX_DEEP = 0  # measured rows deeper than BENCH_STRIP_DEEP_M allowed (env REPLAY_BENCH_STRIP_MAX_DEEP)
BENCH_STRIP_DEEP_M = 2.5
STICKY_STRIP_FILLS = False  # env REPLAY_STICKY_STRIP_FILLS=1: a fill whose nearest sighting was in the strip never holds a place


def _strip_env() -> tuple[float, float, int]:
    import os

    g = os.environ.get
    return (float(g("REPLAY_BENCH_STRIP_IN_M", "") or BENCH_STRIP_IN_M), float(g("REPLAY_BENCH_STRIP_SHARE", "") or BENCH_STRIP_SHARE),
            int(g("REPLAY_BENCH_STRIP_MAX_DEEP", "") or BENCH_STRIP_MAX_DEEP))


def bench_strip_rows(rows, reasons: dict | None = None, ref: str = "referee") -> set[int]:
    """ids (``id(row)``) of every row of the ids whose measured rows (box present) are all in the bench strip."""
    L, W = court.LENGTH, court.WIDTH
    in_m, share, max_deep = _strip_env()  # A/B knobs for the gate
    tracks: dict = defaultdict(list)
    for r in rows:
        if getattr(r, "object_type", "player") == "ball":
            continue
        tracks[r.track_id].append(r)
    out: set[int] = set()
    for t, rs in tracks.items():
        if any(getattr(r, "object_type", "") == ref or r.team == ref for r in rs):
            continue
        meas = [r for r in rs if getattr(r, "box", None) is not None]
        if not meas:
            continue
        n_strip = n_deep = 0
        for r in meas:
            off = max(-r.y, r.y - W)  # signed metres beyond the nearest sideline
            if off < -in_m or (off < 0 and (r.x < BENCH_STRIP_CORNER_M or r.x > L - BENCH_STRIP_CORNER_M)):
                n_deep += off < -BENCH_STRIP_DEEP_M
            else:
                n_strip += 1
        if n_strip >= share * len(meas) and n_deep <= max_deep:
            for r in rs:
                out.add(id(r))
                if reasons is not None:
                    reasons[id(r)] = "bench strip"
    return out


def _env_on(name: str, default: bool) -> bool:
    import os

    env = os.environ.get(name, "").strip()
    return env not in ("0", "false", "no") if env else default


def merge_double_ids(rows, row_rank: dict, reasons: dict | None = None, numbers: dict | None = None) -> set[int]:
    """Two output ids drawn on ONE person at the same time (benchmark 19.10: 10 of 17 missed handovers on clip-4400)
    become one id: same team, both present in >= DOUBLE_MIN common samples and within DOUBLE_M in >= DOUBLE_SHARE of
    them, no conflicting jersey numbers (``numbers``: id -> number, optional). The later-starting id is renamed to
    the earlier one (rows are modified in place: track_id / player_id); in the common samples one row is kept (seen
    before filled-in, then the higher rank). Returns id() of the rows to drop."""
    why = reasons if reasons is not None else {}
    by_id: dict = defaultdict(dict)
    for r in rows:
        if getattr(r, "object_type", "player") != "ball":
            by_id[r.track_id][r.k] = r
    start = {t: min(v) for t, v in by_id.items()}
    team = {t: next(iter(v.values())).team for t, v in by_id.items()}
    by_k: dict = defaultdict(list)
    for t, v in by_id.items():
        for k, r in v.items():
            by_k[k].append(r)
    close: dict = defaultdict(int)
    apart: dict = defaultdict(int)
    common: dict = defaultdict(int)
    for k, rs in by_k.items():
        for a in range(len(rs)):
            for b in range(a + 1, len(rs)):
                ra, rb = rs[a], rs[b]
                if ra.team != rb.team or ra.team not in ("A", "B", "referee"):
                    continue
                key = tuple(sorted((ra.track_id, rb.track_id), key=str))
                if (not ra.interpolated and not rb.interpolated and ra.box is not None and rb.box is not None
                        and _iou(ra.box, rb.box) < DUP_IOU):
                    apart[key] += 1  # two different detections at the same time: two people (a screen), not one
                elif math.hypot(ra.x - rb.x, ra.y - rb.y) < DOUBLE_M:
                    close[key] += 1
    for key in close:
        ta, tb = key
        common[key] = len(by_id[ta].keys() & by_id[tb].keys())
    parent: dict = {}

    def root(t):
        while parent.get(t, t) != t:
            t = parent[t]
        return t

    pairs = sorted((k for k in close if close[k] >= DOUBLE_MIN and close[k] >= DOUBLE_SHARE * common[k]
                    and apart[k] < 2),
                   key=lambda k: -close[k])
    for ta, tb in pairs:
        ra_, rb_ = root(ta), root(tb)
        if ra_ == rb_ or team[ra_] != team[rb_]:
            continue
        if numbers and numbers.get(ra_) is not None and numbers.get(rb_) is not None                 and str(numbers[ra_]) != str(numbers[rb_]):
            continue
        keep, gone = (ra_, rb_) if start[ra_] <= start[rb_] else (rb_, ra_)
        parent[gone] = keep
    if not parent:
        return set()
    drop: set[int] = set()
    merged: dict = defaultdict(dict)
    for t, v in by_id.items():
        for k, r in v.items():
            merged[root(t)].setdefault(k, []).append(r)
    for t, ks in merged.items():
        pid = None
        for k, rs in ks.items():
            for r in rs:
                if r.track_id == t and getattr(r, "player_id", None) is not None:
                    pid = r.player_id
        for k, rs in ks.items():
            if len(rs) > 1:
                rs.sort(key=lambda r: (bool(r.interpolated), -float(row_rank.get(id(r), 0.0))))
                for r in rs[1:]:
                    drop.add(id(r))
                    why[id(r)] = "double id"
            for r in rs:
                if r.track_id != t:
                    r.track_id = t
                    if pid is not None and hasattr(r, "player_id"):
                        r.player_id = pid
    return drop


def enabled(cfg) -> bool:
    """FilterConfig.segment_limits (default on); env REPLAY_SEGMENT_LIMITS=0/1 overrides it for A/B runs."""
    import os

    env = os.environ.get("REPLAY_SEGMENT_LIMITS", "").strip()
    if env:
        return env not in ("0", "false", "no")
    return bool(getattr(cfg, "segment_limits", True))


def _filled(r) -> bool:
    return bool(r.interpolated)


def apply_segment_limits(out_rows, row_rank: dict, strength: dict, fps: float, cfg, team_a: str = "A",
                         team_b: str = "B", ref: str = "referee", present=None, reasons: dict | None = None
                         ) -> set[int]:
    """ids (``id(row)``) of rows to drop. ``out_rows``: TrackRow-like (k, track_id, team, x, y, interpolated, box,
    object_type); ``row_rank``: id(row) -> player-likeness of its detection; ``strength``: track_id -> number (tie
    break); ``present``: sample indices k with calibration (default: those of the rows) - runs are built over them,
    so a calibration hole does not split a run."""
    why = reasons if reasons is not None else {}
    rows = [r for r in out_rows if getattr(r, "object_type", "player") != "ball"]
    if not rows:
        return set()
    ks = sorted(set(present) | {r.k for r in rows}) if present is not None else sorted({r.k for r in rows})
    pos = {k: j for j, k in enumerate(ks)}
    min_evict = max(1, int(round(float(getattr(cfg, "min_evict_s", 1.0)) * fps)))
    off_run = max(1, int(round(float(getattr(cfg, "offcourt_run_s", 0.5)) * fps)))
    L, W = court.LENGTH, court.WIDTH
    drop: set[int] = set()

    def kill(r, reason):
        if id(r) not in drop:
            drop.add(id(r))
            why[id(r)] = reason

    tracks: dict = defaultdict(list)
    for r in rows:
        tracks[r.track_id].append(r)
    for t in tracks:
        tracks[t].sort(key=lambda r: pos[r.k])
    by_k: dict = defaultdict(list)
    for r in rows:
        by_k[r.k].append(r)
    mean_rank = {}
    for t, rs in tracks.items():
        v = [row_rank[id(r)] for r in rs if id(r) in row_rank]
        mean_rank[t] = float(np.mean(v)) if v else 0.0

    # ---- 1. ghosts: a filled tail of a track next to a track of the same team that starts during it
    starts = {t: (pos[rs[0].k], rs[0]) for t, rs in tracks.items()}
    for t, rs in tracks.items():
        j = len(rs)
        while j > 0 and _filled(rs[j - 1]):
            j -= 1
        tail = rs[j:]
        if not tail or j == 0:
            continue
        p0, p1 = pos[tail[0].k], pos[tail[-1].k]
        cut = None
        for u, (ps, r0) in starts.items():
            if u == t or r0.team != rs[0].team or _filled(r0) or not (p0 <= ps <= p1):
                continue
            here = next((r for r in tail if pos[r.k] >= ps), None)
            if here is not None and np.hypot(here.x - r0.x, here.y - r0.y) < GHOST_M:
                cut = ps if cut is None else min(cut, ps)
        if cut is not None:
            for r in tail:
                if pos[r.k] >= cut:
                    kill(r, "ghost")

    # ---- 1b. a filled-in stretch lying on top of a seen same-team person (< SAME_PERSON_M in most of its samples)
    # is that person under a second id (the 0.8 m rule of consolidate.py, decided for the whole stretch)
    seen_xy = defaultdict(list)
    for r in rows:
        if not _filled(r) and id(r) not in drop:
            seen_xy[(r.k, r.team)].append((r.x, r.y, r.track_id))
    first_seen, last_seen = {}, {}
    for t, rs in tracks.items():
        ps = [pos[r.k] for r in rs if not _filled(r) and id(r) not in drop]
        if ps:
            first_seen[t], last_seen[t] = min(ps), max(ps)
    for t, rs in tracks.items():
        j = 0
        while j < len(rs):
            if not _filled(rs[j]):
                j += 1
                continue
            e = j
            while e + 1 < len(rs) and _filled(rs[e + 1]):
                e += 1
            if 0 < j and e < len(rs) - 1:
                # bridges to a later sighting of this id: a hidden player, kept - unless it lies ON a seen teammate
                # (< SAME_PERSON_M) most of the time: then it is that person drawn twice
                on = sum(1 for r in rs[j:e + 1] if any(u != t and np.hypot(x - r.x, y - r.y) < SAME_PERSON_M
                                                         for x, y, u in seen_xy.get((r.k, r.team), ())))
                # ... or it lies next to (< GHOST_M) a seen teammate whose track starts or ends inside the stretch:
                # the tracker handed this person over to that id (a player hidden behind someone is next to a
                # teammate that is seen before, during and after the gap)
                p0, p1 = pos[rs[j].k], pos[rs[e].k]
                near = defaultdict(int)
                for r in rs[j:e + 1]:
                    for x, y, u in seen_xy.get((r.k, r.team), ()):
                        if u != t and np.hypot(x - r.x, y - r.y) < GHOST_M:
                            near[u] += 1
                handover = any(c >= DUP_SHARE * (e - j + 1) and (first_seen[u] > p0 or last_seen[u] < p1)
                               for u, c in near.items())
                if e - j + 1 > HOLE_FILL and (on >= DUP_SHARE * (e - j + 1) or handover):  # (shorter: would blink)
                    for r in rs[j:e + 1]:
                        kill(r, "ghost")
                j = e + 1
                continue
            flags = [any(u != t and np.hypot(x - r.x, y - r.y) < GHOST_M
                         for x, y, u in seen_xy.get((r.k, r.team), ())) for r in rs[j:e + 1]]
            if any(flags):  # an unbridged tail/head meeting a seen teammate: the person goes on under that id
                on = [q for q, f in enumerate(flags) if f]
                # a tail goes from the moment it lies on the other person, a head until it stops doing so
                part = rs[j + on[0]:e + 1] if j > 0 else rs[j:j + on[-1] + 1]
                for r in part:
                    kill(r, "ghost")
            j = e + 1

    # ---- 1c. filled-in positions that are too uncertain to draw (the middle of long gaps, where the tracker may
    # also have joined two different people): whole runs with pos_sigma > FILL_MAX_SIGMA_M, never single samples
    for t, rs in tracks.items():
        bad = [_filled(r) and float(getattr(r, "pos_sigma", 0.0) or 0.0) > FILL_MAX_SIGMA_M for r in rs]
        flags = _fill_runs(bad, rs, pos, True, HOLE_FILL) if any(bad) else bad
        for r, f in zip(rs, flags):
            if f:
                kill(r, "uncertain fill")

    # ---- 2. duplicates: seen boxes of two tracks of one team overlapping in most of their common samples
    seen_at = {t: {r.k: r for r in rs if not _filled(r) and r.box is not None} for t, rs in tracks.items()}
    cand = defaultdict(set)
    for k, rs in by_k.items():
        s = [r for r in rs if not _filled(r) and r.box is not None and id(r) not in drop]
        for a in range(len(s)):
            for b in range(a + 1, len(s)):
                if s[a].team == s[b].team and _iou(s[a].box, s[b].box) > DUP_IOU:
                    cand[tuple(sorted((s[a].track_id, s[b].track_id), key=str))].add(k)
    for (ta, tb), dup_ks in cand.items():
        common = set(seen_at[ta]) & set(seen_at[tb])
        if not common or len(dup_ks) < DUP_SHARE * len(common):
            continue
        weak = min((ta, tb), key=lambda t: (mean_rank[t], strength.get(t, 0.0), len(tracks[t])))
        lo, hi = min(pos[k] for k in common), max(pos[k] for k in common)
        for r in tracks[weak]:
            if lo <= pos[r.k] <= hi:
                kill(r, "duplicate")

    # ---- 3. no team / off court (per run)
    shifted = {}
    for k, rs in by_k.items():
        seen = [r for r in rs if not _filled(r) and r.team in (team_a, team_b, ref)]
        out = sum(1 for r in seen if _out(r, L, W))
        shifted[k] = len(seen) >= SHIFT_MIN_SEEN and out >= SHIFT_SHARE * len(seen)
    for t, rs in tracks.items():
        if rs[0].team not in (team_a, team_b, ref):
            for r in rs:
                if _filled(r) or not (0 <= r.x <= L and 0 <= r.y <= W):
                    kill(r, "no team")
        bad = [_out(r, L, W) and not (shifted[r.k] and not _filled(r)
                                      and float(row_rank.get(id(r), 0.0)) >= SHIFT_MIN_RANK) for r in rs]
        j = 0
        while j < len(rs):
            if not bad[j]:
                j += 1
                continue
            e = j
            while e + 1 < len(rs) and bad[e + 1] and pos[rs[e + 1].k] - pos[rs[e].k] <= 1:
                e += 1
            if e - j + 1 >= off_run or (j == 0 and e == len(rs) - 1) or any(_filled(r) for r in rs[j:e + 1]):
                for r in rs[j:e + 1]:
                    kill(r, "off court")
            j = e + 1

    # ---- 4. caps per group, evicting whole tracks for whole conflict runs
    part = set()
    for k, rs in by_k.items():
        n_team = {team_a: 0, team_b: 0, ref: 0}
        for r in rs:
            if id(r) not in drop and r.team in n_team:
                n_team[r.team] += 1
        if any(n_team[g] > (REFEREES if g == ref else PER_TEAM) for g in n_team):
            part |= partial_rows([r for r in rs if id(r) not in drop])

    def row_score(r):
        if id(r) in part:
            return -1e3
        if _filled(r) or id(r) not in row_rank:
            return FILLED_SCORE  # seen people first: a filled-in / predicted stretch gives way before anyone seen
        return float(row_rank[id(r)])

    # per sample: over a cap, the lowest (filled first, then roster rank) give way - exactly the old per-frame
    # order; then per track the decisions are smoothed: a drop of <= HOLE_FILL samples between kept rows is undone
    # (the row stays, the count is over the cap for a moment), a kept island of <= ISLAND samples between drops goes
    cap_drop: set[int] = set()
    for k, rs in by_k.items():
        alive = [r for r in rs if id(r) not in drop]
        groups = {team_a: [], team_b: [], ref: [], None: []}
        for r in alive:
            groups[r.team if r.team in (team_a, team_b, ref) else None].append(r)
        for g in (team_a, team_b, ref):
            cap = REFEREES if g == ref else PER_TEAM
            mem = sorted(groups[g], key=lambda r: (-row_score(r), -strength.get(r.track_id, 0.0), str(r.track_id)))
            cap_drop.update(id(r) for r in mem[cap:])
        free = max(0, 2 * PER_TEAM - sum(1 for r in groups[team_a] + groups[team_b] if id(r) not in cap_drop))
        mem = sorted(groups[None], key=lambda r: (-row_score(r), -strength.get(r.track_id, 0.0), str(r.track_id)))
        cap_drop.update(id(r) for r in mem[free:])
    for t, rs in tracks.items():
        alive = [r for r in rs if id(r) not in drop]
        flag = [id(r) in cap_drop for r in alive]
        if any(flag):
            flag = _fill_runs(flag, alive, pos, True, HOLE_FILL)  # short drops inside a kept track: undone
            flag = _fill_runs(flag, alive, pos, False, ISLAND)  # short kept islands between drops: dropped
        for r, f in zip(alive, flag):
            if f:
                kill(r, "cap")
    return drop


def smoothed_frame_limits(out_rows, row_rank: dict, strength: dict, cfg, team_a: str = "A", team_b: str = "B",
                          ref: str = "referee", present=None, reasons: dict | None = None, fps: float = 10.0) -> set[int]:
    """The per-frame limits of before (roster.limit_shown_rows + consolidate.consolidate_drop: cap of 10, 5 per team,
    3 referees, duplicates, ghosts, no team, off court) followed by per-track smoothing of the decisions: a drop of
    <= HOLE_FILL samples inside a kept stretch is undone and a kept island of <= ISLAND samples between drops goes.
    Same people as before in every frame but a few, without the 1-3 sample blinking."""
    from replay.consolidate import consolidate_drop
    from replay.roster import limit_shown_rows

    why = reasons if reasons is not None else {}
    rows = [r for r in out_rows if getattr(r, "object_type", "player") != "ball"]
    if not rows:
        return set()
    import os

    ks = sorted(set(present) | {r.k for r in rows}) if present is not None else sorted({r.k for r in rows})
    pos = {k: j for j, k in enumerate(ks)}
    cw: dict = {}
    hole_fill = int(os.environ.get("REPLAY_HOLE_FILL", HOLE_FILL))  # experiment overrides (A/B runs)
    island = int(os.environ.get("REPLAY_ISLAND", ISLAND))
    sticky = sticky_enabled(cfg)
    weak: set[int] = set()
    reset: set = set()
    if sticky:
        # the per-frame rules without their caps, then caps that keep evicting the SAME people for a whole
        # over-the-cap episode (per-frame caps swap who is the weakest from sample to sample: count jumps)
        nocap = consolidate_drop(rows, strength, team_a, team_b, ref, rank=row_rank, reasons=cw)
        old = {i for i in nocap if cw.get(i) != "cap"}
        old |= _stale_predictions(rows, pos, old, cw, PRED_MAX_AGE_S * fps)
        if not _env_on("REPLAY_STICKY_FILLS", STICKY_FILLS):  # step E: uncertain / old fills never hold a place
            weak = _weak_fills(rows, pos, PRED_MAX_AGE_S * fps,
                               strip_in_m=_strip_env()[0] if _env_on("REPLAY_STICKY_STRIP_FILLS", STICKY_STRIP_FILLS) else None)
            reset = _dead_samples(rows)
        old |= _sticky_caps(rows, row_rank, strength, pos, old, cw, team_a, team_b, ref, weak=weak, reset=reset)
    else:
        caps = (((team_a, team_b), cfg.max_players_shown), ((ref,), cfg.max_referees))
        if _env_on("REPLAY_TEAM_CAP_FIRST", bool(getattr(cfg, "team_cap_first", False))):
            from replay.consolidate import PER_TEAM
            caps = (((team_a,), PER_TEAM), ((team_b,), PER_TEAM)) + caps
        old = limit_shown_rows(rows, caps, row_rank, strength)
        old |= consolidate_drop([r for r in rows if id(r) not in old], strength, team_a, team_b, ref, rank=row_rank,
                                reasons=cw)
    tracks: dict = defaultdict(list)
    for r in rows:
        tracks[r.track_id].append(r)
    for rs in tracks.values():
        rs.sort(key=lambda r: pos[r.k])
    import os

    if os.environ.get("REPLAY_KEEP_HIDDEN", "").strip() not in ("", "0", "false", "no"):  # Review Focus 1 (opt-in:
        old -= _hidden_next_to_teammate(tracks, pos, old, cw)  # gated r10-hidden-s3: phantoms 0.68 -> 0.70 WORSE)
    drop: set[int] = set()
    for rs in tracks.values():
        f = [id(r) in old for r in rs]
        if any(f):
            f = _fill_runs(f, rs, pos, True, hole_fill)
            f = _fill_runs(f, rs, pos, False, island)
        for r, x in zip(rs, f):
            if x:
                drop.add(id(r))
                why[id(r)] = cw.get(id(r), "cap")
    if sticky and os.environ.get("REPLAY_DRAW_CAP", "").strip() not in ("", "0", "false", "no"):  # draw level: the hole filling above must not put more than 5 + 5 + 3 on the court; the same
        hard: dict = {}  # extras stay out for the whole over-the-cap episode (no new blinking)
        capped = _sticky_caps(rows, row_rank, strength, pos, drop, hard, team_a, team_b, ref, avoid_holes=hole_fill,
                              weak=weak, reset=reset)
        # short holes this cut into a track are refilled where the team still has a free place at that moment
        n_at: dict = defaultdict(int)
        for r in rows:
            if id(r) not in drop and id(r) not in capped:
                n_at[(r.k, _grp(r, team_a, team_b, ref))] += 1
        for rs in tracks.values():
            f = [id(r) in capped for r in rs]
            if not any(f):
                continue
            u = _fill_runs(f, rs, pos, True, hole_fill)  # False where a short hole would be undone
            for r, x, keep_out in zip(rs, f, u):
                grp = _grp(r, team_a, team_b, ref)
                if x and not keep_out and grp is not None:
                    key = (r.k, grp)
                    if n_at[key] < (REFEREES if grp == ref else PER_TEAM):
                        n_at[key] += 1
                        capped.discard(id(r))
        for i in capped:
            drop.add(i)
            why[i] = "cap (drawn)"
    return drop


SWAP_MARGIN = 1.0  # an evicted seen person comes back in place of a kept one only when clearly more player-like
STICKY_CAP = False  # FilterConfig.sticky_cap / env REPLAY_STICKY_CAP: caps with memory (identity_design.md step E)
STICKY_FILLS = False  # env REPLAY_STICKY_FILLS=1: the 19.10 variant where every filled-in row holds its cap place


def sticky_enabled(cfg) -> bool:
    """FilterConfig.sticky_cap (default off); env REPLAY_STICKY_CAP=0/1 overrides it for A/B runs."""
    return _env_on("REPLAY_STICKY_CAP", bool(getattr(cfg, "sticky_cap", STICKY_CAP)))


def _weak_fills(rows, pos, max_age: float, strip_in_m: float | None = None) -> set[int]:
    """Filled-in rows that must never hold a cap place against anyone (step E, fill-aware ranking): position spread
    ``pos_sigma`` > FILL_MAX_SIGMA_M, or more than ``max_age`` samples from the nearest sighting of their own track
    (a hidden player is drawn on where he was last seen; after a second that is a guess). Drawn only while the
    group fits its cap anyway. ``strip_in_m``: also fills whose nearest sighting stood in the bench strip (within
    this many metres inside / anywhere beyond a sideline) - bench people filled in along the line."""
    out: set[int] = set()
    tracks: dict = defaultdict(list)
    seen: dict = defaultdict(list)
    W = court.WIDTH
    for r in rows:
        if r.interpolated:
            tracks[r.track_id].append(r)
        else:
            seen[r.track_id].append((pos[r.k], max(-r.y, r.y - W)))
    for t, rs in tracks.items():
        sv = sorted(seen.get(t, ()))
        sa = np.array([p for p, _ in sv])
        for r in rs:
            if float(getattr(r, "pos_sigma", 0.0) or 0.0) > FILL_MAX_SIGMA_M or not len(sa):
                out.add(id(r))
                continue
            p = pos[r.k]
            j = int(np.searchsorted(sa, p))
            dl = abs(p - sa[j - 1]) if j > 0 else 1e9
            dr = abs(sa[j] - p) if j < len(sa) else 1e9
            if min(dl, dr) > max_age:
                out.add(id(r))
            elif strip_in_m is not None and sv[j - 1 if dl <= dr else j][1] >= -strip_in_m:
                out.add(id(r))
    return out


def _dead_samples(rows) -> set:
    """Sample indices k where a dead ball starts (lineup.DEAD windows of the current build): sticky cap episodes
    are released there, so the caps re-evaluate when substitutions happen."""
    try:
        from replay import lineup
        windows = list(lineup.DEAD)
    except Exception:  # noqa: BLE001
        return set()
    if not windows:
        return set()
    kt: dict = {}
    for r in rows:
        t = getattr(r, "t", None)
        if t is None:
            return set()
        kt.setdefault(r.k, float(t))
    ks = sorted(kt, key=lambda k: kt[k])
    ts = np.array([kt[k] for k in ks])
    out = set()
    for a, b in windows:
        j = int(np.searchsorted(ts, float(a)))
        if j < len(ks) and ts[j] <= float(b):
            out.add(ks[j])
    return out


def _sticky_caps(rows, row_rank, strength, pos, gone, cw, team_a, team_b, ref, avoid_holes: int = 0,
                 weak: set | None = None, reset: set | None = None) -> set[int]:
    """Per-team caps (5 / 5 / 3 referees / team-less only in free places) with memory: within one over-the-cap
    episode the evicted ids stay evicted (re-admitted only when a place frees up, or swapped for a kept one that is
    SWAP_MARGIN less player-like); the episode ends when the group fits its cap again. ``weak`` (id(row)): filled-in
    rows that never hold a place - over the cap they go first and their id is NOT remembered as evicted (it competes
    afresh when seen again); under the cap they take the free places, lowest pos_sigma first. ``reset``: samples k
    where every episode is released (dead balls)."""
    weak = weak or set()
    reset = reset or set()
    part_cache: dict = {}
    by_k: dict = defaultdict(list)
    for r in rows:
        if id(r) not in gone:
            by_k[r.k].append(r)

    def score(r, rs):
        if r.k not in part_cache:
            part_cache[r.k] = partial_rows(rs)
        if id(r) in part_cache[r.k]:
            return -1e3
        if r.interpolated or id(r) not in row_rank:
            # hidden behind someone (state 4) before predicted / off-screen / rejected fills
            return FILLED_SCORE if getattr(r, "pos_state", 4) == 4 else FILLED_SCORE - 2.0
        return float(row_rank[id(r)])

    out: set[int] = set()
    alive_at: dict = defaultdict(set)  # track -> sample positions where it is drawn (before these caps)
    for r in rows:
        if id(r) not in gone:
            alive_at[r.track_id].add(pos[r.k])

    def hole_risk(r):
        """evicting r now would cut a hole of <= avoid_holes samples into its track (it is drawn just before and
        again within avoid_holes samples)"""
        if not avoid_holes:
            return 0
        p, a = pos[r.k], alive_at[r.track_id]
        return int((p - 1) in a and any((p + d) in a for d in range(1, avoid_holes + 1)))

    evicted = {team_a: set(), team_b: set(), ref: set(), None: set()}
    for k in sorted(by_k, key=lambda k: pos[k]):
        rs = by_k[k]
        shown_ab = 0
        if k in reset:
            for E in evicted.values():
                E.clear()  # dead ball: substitutions possible, the caps re-evaluate
        for g in (team_a, team_b, ref, None):
            mem = [r for r in rs if (r.team == g if g is not None else r.team not in (team_a, team_b, ref))]
            cap = REFEREES if g == ref else PER_TEAM if g is not None else max(0, 2 * PER_TEAM - shown_ab)
            E = evicted[g]
            soft = [r for r in mem if id(r) in weak]
            if soft:
                mem = [r for r in mem if id(r) not in weak]
            if len(mem) <= cap:
                E.clear()  # episode over: everybody (who can hold a place) fits
                free = cap - len(mem)
                soft.sort(key=lambda r: (float(getattr(r, "pos_sigma", 0.0) or 0.0), str(r.track_id)))
                for r in soft[free:]:
                    out.add(id(r))
                    cw[id(r)] = "cap"
                if g in (team_a, team_b):
                    shown_ab += len(mem) + min(free, len(soft))
                continue
            for r in soft:  # over the cap: fills that cannot hold a place go, whoever they are
                out.add(id(r))
                cw[id(r)] = "cap"
            sc = {id(r): (-hole_risk(r), score(r, rs), strength.get(r.track_id, 0.0)) for r in mem}
            kept = [r for r in mem if r.track_id not in E]
            out_now = [r for r in mem if r.track_id in E]
            kept.sort(key=lambda r: sc[id(r)])
            while len(kept) > cap:  # new extra: the weakest goes (and stays out)
                r = kept.pop(0)
                E.add(r.track_id)
                out_now.append(r)
            out_now.sort(key=lambda r: sc[id(r)], reverse=True)
            while out_now and len(kept) < cap:  # a place is free: the best evicted one comes back
                r = out_now.pop(0)
                E.discard(r.track_id)
                kept.append(r)
            kept.sort(key=lambda r: sc[id(r)])
            if out_now and kept and sc[id(out_now[0])][1] > sc[id(kept[0])][1] + SWAP_MARGIN:
                a, b = out_now.pop(0), kept.pop(0)  # clearly better one waiting: swap
                E.discard(a.track_id)
                E.add(b.track_id)
                kept.append(a)
                out_now.append(b)
            for r in out_now:
                out.add(id(r))
                cw[id(r)] = "cap"
            if g in (team_a, team_b):
                shown_ab += len(kept)
    return out


def _hidden_next_to_teammate(tracks: dict, pos: dict, old: set, cw: dict) -> set[int]:
    """Rows the per-frame "ghost" rule removed that belong to a filled stretch BETWEEN two sightings of their own
    track, where every seen same-team person near it (< GHOST_M) is seen before, during and after the stretch: a
    player hidden behind a teammate (screen, post-up, free-throw lane), not a second id of that teammate."""
    seen_span = {}
    seen_at: dict = defaultdict(list)
    for t, rs in tracks.items():
        ps = [pos[r.k] for r in rs if not r.interpolated]
        if ps:
            seen_span[t] = (min(ps), max(ps))
        for r in rs:
            if not r.interpolated and id(r) not in old:
                seen_at[(r.k, r.team)].append(r)
    keep: set[int] = set()
    for t, rs in tracks.items():
        j = 0
        while j < len(rs):
            if not rs[j].interpolated:
                j += 1
                continue
            e = j
            while e + 1 < len(rs) and rs[e + 1].interpolated:
                e += 1
            run = rs[j:e + 1]
            if 0 < j and e < len(rs) - 1 and any(cw.get(id(r)) == "ghost" for r in run):
                p0, p1 = pos[rs[j - 1].k], pos[rs[e + 1].k]
                near = {q.track_id for r in run for q in seen_at.get((r.k, r.team), ())
                        if q.track_id != t and np.hypot(q.x - r.x, q.y - r.y) < GHOST_M}
                if near and all(seen_span.get(u, (p0 + 1, p0))[0] <= p0 and seen_span[u][1] >= p1 for u in near):
                    keep.update(id(r) for r in run if cw.get(id(r)) == "ghost")
            j = e + 1
    return keep


PRED_MAX_AGE_S = 1.0  # predicted / off-screen rows (no box, not hidden behind someone) this long after the last sighting go


def _stale_predictions(rows, pos, gone, cw, max_age: float) -> set[int]:
    """Box-less predicted (pos_state 2), rejected (3) and off-screen (5) rows more than ``max_age`` samples away
    from the nearest sighting of their own track (both directions); hidden-behind fills (state 4) are kept."""
    out: set[int] = set()
    tracks: dict = defaultdict(list)
    for r in rows:
        tracks[r.track_id].append(r)
    for rs in tracks.values():
        seen = sorted(pos[r.k] for r in rs if not r.interpolated)
        if not seen:
            continue
        sa = np.array(seen)
        for r in rs:
            if not r.interpolated or getattr(r, "pos_state", 4) == 4 or id(r) in gone:
                continue
            p = pos[r.k]
            j = np.searchsorted(sa, p)
            d = min(abs(p - sa[j - 1]) if j > 0 else 1e9, abs(sa[j] - p) if j < len(sa) else 1e9)
            if d > max_age:
                out.add(id(r))
                cw[id(r)] = "stale prediction"
    return out


def _grp(r, team_a, team_b, ref):
    return r.team if r.team in (team_a, team_b, ref) else None


def _fill_runs(flag: list, alive: list, pos: dict, value: bool, max_len: int) -> list:
    """Runs of ``value`` of at most ``max_len`` samples with consecutive neighbours of the other value on both
    sides are flipped (computed on the input flags, so the result does not depend on the scan order)."""
    out = list(flag)
    n = len(flag)
    q = 0
    while q < n:
        e = q
        while e + 1 < n and flag[e + 1] == flag[q] and pos[alive[e + 1].k] - pos[alive[e].k] <= 1:
            e += 1
        if (flag[q] == value and e - q + 1 <= max_len and q > 0 and e < n - 1
                and pos[alive[q].k] - pos[alive[q - 1].k] <= 1 and pos[alive[e + 1].k] - pos[alive[e].k] <= 1):
            for x in range(q, e + 1):
                out[x] = not value
        q = e + 1
    return out


def _out(r, L: float, W: float) -> bool:
    return not (-OFF_COURT_M <= r.x <= L + OFF_COURT_M and -OFF_COURT_M <= r.y <= W + OFF_COURT_M)
