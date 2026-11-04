"""Ball tracking: image trajectory (ball_track.py) -> court position per analysed frame.

A floor homography only gives correct court positions for things *on the floor*, and the ball is mostly in
someone's hands or in the air, so an airborne ball is never projected through the floor. Instead:

1. **Image trajectory** (ball_track.link_ball): all ball candidates of the clip are linked offline into one
   physically plausible path (static false positives removed, isolated false positives skipped); short gaps
   are filled along the arc in the image.
2. **Anchors** - frames where the court position is known:
   * POSSESSED - the ball is in/next to a player's (or referee's) box: the ball is at that person's court
     position (owner = track id). Ambiguous overlaps are resolved by horizontal offset, by depth (ball image
     size vs. box height) and by hysteresis (the current holder keeps the ball unless clearly beaten).
   * at a rim (ball within a few rim radii of the projected ring): the ball is at that basket.
   * held by a person who is not a tracked player (box missing from the tracks): that person's foot point.
3. **Gaps between anchors** (up to ``max_fill_s``):
   * same holder on both sides and short: still POSSESSED by that holder (dribbling, ball hidden by the body),
     the position follows the holder;
   * otherwise ESTIMATED: the ball stays with the last holder while unseen, then flies (time-linear on the
     floor, which is exact for the ground projection of a parabolic flight) to the next anchor in a flight
     time of distance / ``pass_speed_mps``.
4. At the ends of long gaps the ball stays with its last / next holder for ``hold_extend_s`` (ESTIMATED).
5. Everything still unknown is GUESSED so that the ball never vanishes: it stays with the last known holder /
   place and moves to the next known position just before it is seen again (a pass at ``pass_speed_mps``);
   before the first / after the last sighting it stays where it was. Guessed positions are display-only:
   analyses (game_state) treat them as unknown.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from replay import court
from replay.ball_track import BallTrack, link_ball
from replay.config import BallConfig

POSSESSED, ESTIMATED, GUESSED = "possessed", "estimated", "guessed"

Holder = tuple  # (track_id, image box (4,), court xy (2,)[, team])


@dataclass
class BallFrame:
    i: int  # index into the processed-frame arrays
    box: tuple[float, float, float, float] | None  # image box of the detected ball (None = not seen)
    conf: float
    x: float | None = None  # court metres
    y: float | None = None
    state: str = ""
    owner: int | None = None
    img: tuple[float, float] | None = None  # image position of the ball (detected or interpolated)
    source: str = ""  # holder | rim | person | fill | follow


def track_ball(n: int, ball_rows: list[np.ndarray], G: list[np.ndarray], seg: np.ndarray, fps: float,
               cfg: BallConfig) -> list[tuple[float, float, float, float, float] | None]:
    """ball_rows[i]: (m, 5) [x1, y1, x2, y2, conf] candidates in frame i. Returns the chosen box per frame."""
    return link_ball(n, ball_rows, G, seg, fps, cfg).boxes


def _holder_score(bx: float, by: float, bs: float, pb: np.ndarray, cfg: BallConfig) -> float | None:
    """Lower = better; None = the ball is not at this person."""
    w, h = max(1.0, pb[2] - pb[0]), max(1.0, pb[3] - pb[1])
    if not (pb[0] - cfg.holder_margin_w * w <= bx <= pb[2] + cfg.holder_margin_w * w
            and pb[1] - 0.05 * h <= by <= pb[3] + 0.05 * h):
        return None
    s = abs(bx - (pb[0] + pb[2]) / 2) / w
    if np.isfinite(bs) and bs > 0:
        depth = abs(np.log((bs / h) / cfg.holder_ratio))  # ball much too big/small for this person = other depth
        if depth > np.log(cfg.holder_ratio_tol):
            return None
        s += 0.6 * depth
    return float(s)


def ball_court_positions(boxes: list[tuple | None], players_at: list[list[Holder]], fps: float, cfg: BallConfig,
                         track: BallTrack | None = None, track_xy: dict[Any, dict[int, np.ndarray]] | None = None,
                         rims: dict[str, Sequence[tuple[float, float, float] | None]] | None = None,
                         persons: Sequence[np.ndarray] | None = None, H: Sequence[np.ndarray | None] | None = None
                         ) -> list[BallFrame]:
    """Court position of the ball per analysed frame.

    boxes[i]: chosen ball detection (x1, y1, x2, y2, conf) or None; players_at[i]: [(track_id, image box,
    court xy[, team])] of the people who can hold the ball; track: image trajectory (else from ``boxes``);
    track_xy[track_id][i]: court position of every output track per frame (incl. gap-filled) so that a hidden
    ball can follow its holder; rims: scoring.rim_tracks output; persons[i]: (m, 4) all person boxes of frame i
    (pass 1) with H[i] the image->court homography, for holders that are not tracked players.
    """
    n = len(boxes)
    frames = [BallFrame(i, None if b is None else tuple(b[:4]), 0.0 if b is None else float(b[4]))
              for i, b in enumerate(boxes)]
    if track is not None:
        img, size, observed = track.img, track.size, track.observed
    else:
        img, size, observed = np.full((n, 2), np.nan), np.full(n, np.nan), np.array([b is not None for b in boxes])
        for i, b in enumerate(boxes):
            if b is not None:
                img[i] = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
                size[i] = np.sqrt(max(1.0, (b[2] - b[0]) * (b[3] - b[1])))
    track_xy = track_xy or {}
    baskets = dict(zip("LR", court.baskets()))

    # ---- free throws (dead ball, shooter at the line): only the shooter / referees can have the ball there
    ft = free_throw_windows(players_at, track_xy, fps, cfg) if cfg.ft_prior else []
    banned: dict[int, set] = {}
    for w in ft:
        for i in range(w["i0"], w["i1"] + 1):
            banned[i] = w["others"] - {w["shooter_at"].get(i)}  # the player at the line now (track ids change)
    # ---- who holds the ball: best holder sequence over all sightings (holder_decode)
    # sightings shortly before the ball reaches a rim: the shooter may have caught it just before (no hysteresis)
    near_rim = np.zeros(n, bool)
    for side, rr in (rims or {}).items():
        for i in range(n):
            r = rr[i] if i < len(rr) else None
            if r is None or not observed[i] or not np.isfinite(img[i, 0]):
                continue
            u, v, rad = r
            near_rim[i] = near_rim[i] or (abs(img[i, 0] - u) <= cfg.rim_radius_r * rad
                                          and -2.0 * cfg.rim_radius_r * rad <= img[i, 1] - v <= cfg.rim_radius_r * rad)
    win = max(1, int(round(cfg.shot_soon_s * fps)))
    shot_soon = np.array([near_rim[i + 1:i + 1 + win].any() for i in range(n)], bool)
    if cfg.holder_decode:
        held = decode_holders(img, size, observed, players_at, fps, cfg, banned, shot_soon)
    else:
        held = greedy_holders(img, size, observed, players_at, fps, cfg, banned)

    # ---- anchors
    for i, f in enumerate(frames):
        if not np.isfinite(img[i, 0]):
            continue
        f.img = (float(img[i, 0]), float(img[i, 1]))
        if not observed[i]:
            continue  # interpolated image positions never create anchors (only the gap filling below uses them)
        bx, by, bs = img[i, 0], img[i, 1], size[i]
        best = held.get(i)
        if best is not None:
            f.owner, (f.x, f.y), f.state, f.source = best[0], (float(best[2][0]), float(best[2][1])), POSSESSED, "holder"
            continue
        if rims:
            for side, rr in rims.items():
                r = rr[i] if i < len(rr) else None
                if r is None:
                    continue
                u, v, rad = r
                if abs(bx - u) <= cfg.rim_radius_r * rad and -2.0 * cfg.rim_radius_r * rad <= by - v <= cfg.rim_radius_r * rad:
                    f.x, f.y, f.state, f.source = float(baskets[side][0]), float(baskets[side][1]), ESTIMATED, "rim"
                    break
            if f.source:
                continue
        if i in banned:
            continue  # free throw: a ball near a lane player is the shooter's ball seen in front of / behind them
        if cfg.person_anchors and persons is not None and H is not None and H[i] is not None and len(persons[i]):
            best_box, bscore = None, np.inf
            for pb in persons[i]:
                s = _holder_score(bx, by, bs, np.asarray(pb, float), cfg)
                if s is not None and s < bscore:
                    best_box, bscore = pb, s
            if best_box is not None:
                from replay import homography

                p = homography.project(H[i], np.array([[(best_box[0] + best_box[2]) / 2, best_box[3]]]))[0]
                if np.isfinite(p).all() and court.in_court(p, 1.0)[0]:
                    f.x, f.y, f.state, f.source = float(p[0]), float(p[1]), ESTIMATED, "person"

    _drop_blips(frames, fps)
    _drop_inconsistent(frames, fps, cfg)

    # ---- gaps between anchors
    def xy_of(owner: Any, i: int) -> np.ndarray | None:
        if owner is None:
            return None
        p = track_xy.get(owner, {}).get(i)
        return None if p is None else np.asarray(p, float)

    anchors = [f.i for f in frames if f.source in ("holder", "rim", "person")]
    max_fill = int(round(cfg.max_fill_s * fps))
    ext = int(round(cfg.hold_extend_s * fps))
    for a, b in zip(anchors[:-1], anchors[1:]):
        if b - a <= 1:
            continue
        fa, fb = frames[a], frames[b]
        if b - a > max_fill:
            _extend(frames, a, min(b, a + ext + 1), fa, xy_of, cfg.holder_decode)
            _extend(frames, b, max(a, b - ext - 1), fb, xy_of, cfg.holder_decode)
            continue
        pa, pb = np.array([fa.x, fa.y]), np.array([fb.x, fb.y])
        if fa.owner is not None and fa.owner == fb.owner and (b - a) <= cfg.same_holder_possessed_s * fps:
            for i in range(a + 1, b):
                p = xy_of(fa.owner, i)
                if p is None:
                    u = (i - a) / (b - a)
                    p = pa + (pb - pa) * u
                g = frames[i]
                g.x, g.y, g.state, g.owner, g.source = float(p[0]), float(p[1]), POSSESSED, fa.owner, "follow"
            continue
        # held by the last holder while unseen, then a flight to the next anchor
        tf = float(np.clip(np.linalg.norm(pb - pa) / cfg.pass_speed_mps, 0.2, cfg.max_flight_s))
        release = a
        while release + 1 < b and frames[release + 1].img is None and (b - release - 1) > tf * fps:
            release += 1
        start = pa
        for i in range(a + 1, release + 1):
            p = xy_of(fa.owner, i)
            p = (start if cfg.holder_decode else pa) if p is None else p  # holder not tracked in this frame
            g = frames[i]
            g.x, g.y, g.state, g.source, g.owner = float(p[0]), float(p[1]), ESTIMATED, "follow", fa.owner
            start = p
        for i in range(release + 1, b):
            u = (i - release) / (b - release)
            p = start + (pb - start) * u
            g = frames[i]
            g.x, g.y, g.state, g.source = float(p[0]), float(p[1]), ESTIMATED, "fill"
    if anchors:
        _extend(frames, anchors[-1], min(n, anchors[-1] + ext + 1), frames[anchors[-1]], xy_of, cfg.holder_decode)
        _extend(frames, anchors[0], max(-1, anchors[0] - ext - 1), frames[anchors[0]], xy_of, cfg.holder_decode)
    if cfg.slow_fill_holder:
        _slow_fill_holders(frames, img, size, players_at, fps, cfg, banned)
    for w in ft:  # free throws: while not seen in flight / at the rim the ball is with the shooter (or a referee)
        for i in range(w["i0"], w["i1"] + 1):
            g = frames[i]
            if g.source in ("holder", "rim") or (g.owner is not None and g.owner not in w["others"]):
                continue  # seen with someone allowed (shooter, referee) / at the rim
            if g.source == "fill" and g.img is not None:
                continue  # a seen flight (the attempt)
            sh = w["shooter_at"].get(i, w["shooter"])
            p = xy_of(sh, i)
            if p is None:  # shooter not tracked in this frame: nearest frame where they are (standing still)
                p = next((q for k in range(1, int(2 * fps)) for q in (xy_of(sh, i - k), xy_of(sh, i + k))
                          if q is not None), None)
            if p is None:
                continue
            g.x, g.y, g.owner, g.source = float(p[0]), float(p[1]), sh, "follow"
            g.state = POSSESSED  # dead ball at the line: the shooter has it (scoring credits the team from this)
    _guess_unknown(frames, fps, cfg, xy_of)
    return frames


def _slow_fill_holders(frames: list[BallFrame], img: np.ndarray, size: np.ndarray, players_at: list[list[Holder]],
                       fps: float, cfg: BallConfig, banned: dict[int, set]) -> None:
    """A "fill" (no holder seen) lasting much longer than any pass is not a flight: the ball is in somebody's hands
    without a clean sighting. Where the image trajectory (detected or interpolated along its arc) puts the ball
    inside a player's box, the ball is at that player (ESTIMATED, source "imgholder") instead of drifting along a
    straight line between far-apart anchors."""
    runs: list[list[int]] = []
    for f in frames:
        if f.source == "fill":
            if runs and runs[-1][-1] == f.i - 1:
                runs[-1].append(f.i)
            else:
                runs.append([f.i])
    for r in runs:
        if len(r) < cfg.slow_fill_s * fps:
            continue
        who: list[Holder | None] = []
        for i in r:
            best, bs = None, np.inf
            if frames[i].img is not None and np.isfinite(img[i, 0]):
                for p in players_at[i]:
                    if p[0] in banned.get(i, ()):
                        continue
                    sc = _holder_score(img[i, 0], img[i, 1], size[i], np.asarray(p[1], float), cfg)
                    if sc is not None and sc < bs:
                        best, bs = p, sc
            who.append(best)
        # only steady stretches with the same player (a ball passing in front of bodies flickers between them)
        k = 0
        while k < len(r):
            tid = None if who[k] is None else who[k][0]
            m = k
            while m + 1 < len(r) and who[m + 1] is not None and tid is not None and who[m + 1][0] == tid:
                m += 1
            if tid is not None and m - k + 1 >= cfg.slow_fill_steady * fps:
                for j in range(k, m + 1):
                    g = frames[r[j]]
                    g.x, g.y, g.source = float(who[j][2][0]), float(who[j][2][1]), "imgholder"
            k = m + 1


def free_throw_windows(players_at: list[list[Holder]], track_xy: dict[Any, dict[int, np.ndarray]], fps: float,
                       cfg: BallConfig) -> list[dict]:
    """Free-throw formations from the players' court positions: one player at a free-throw line (on the court
    axis), at least ``ft_min_lane`` others along that lane, everybody nearly still, for ``ft_min_s``. Returns
    [{i0, i1, side, shooter, others}] (others = every other player track seen in the window)."""
    n = len(players_at)
    if n == 0:
        return []
    L, mid = court.LENGTH, court.MID_Y
    team = {}
    for pl in players_at:
        for p in pl:
            if len(p) > 3:
                team[p[0]] = p[3]
    half = max(1, int(round(0.5 * fps)))

    def speed(tid: Any, i: int) -> float | None:
        tr = track_xy.get(tid, {})
        a, b = tr.get(i - half), tr.get(i + half)
        if a is None or b is None:
            return None
        return float(np.linalg.norm(np.asarray(b) - np.asarray(a)) / (2 * half / fps))

    hit = np.zeros(n, bool)
    shooter_at: list[Any] = [None] * n
    side_at: list[str] = [""] * n
    for i in range(n):
        pos = [(tid, np.asarray(tr[i], float)) for tid, tr in track_xy.items()
               if i in tr and team.get(tid) in ("A", "B")]
        if len(pos) < 3:
            continue
        for side, x_ft, sgn in (("L", court.FT_LINE_X, 1), ("R", L - court.FT_LINE_X, -1)):
            base = 0.0 if side == "L" else L
            cand = [(tid, p) for tid, p in pos if -0.5 <= sgn * (p[0] - x_ft) <= 1.2 and abs(p[1] - mid) <= 1.3]
            if not cand:
                continue
            tid, sp = min(cand, key=lambda c: abs(c[1][1] - mid) + abs(c[1][0] - x_ft))
            lane = [q for t2, q in pos if t2 != tid and 1.2 <= abs(q[1] - mid) <= 4.0
                    and 0.5 <= sgn * (q[0] - base) <= x_ft - 0.3]
            if len(lane) < cfg.ft_min_lane:
                continue
            v = speed(tid, i)
            if v is not None and v > cfg.ft_still_mps:
                continue
            hit[i], shooter_at[i], side_at[i] = True, tid, side
            break
    # close short holes, keep long runs
    out: list[dict] = []
    gap = int(round(cfg.ft_gap_s * fps))
    runs: list[list[int]] = []
    for i in np.flatnonzero(hit):
        if runs and i - runs[-1][1] <= gap and side_at[runs[-1][1]] == side_at[i]:
            runs[-1][1] = int(i)
        else:
            runs.append([int(i), int(i)])
    for a, b in runs:
        if (b - a) / fps < cfg.ft_min_s:
            continue
        ids = [shooter_at[i] for i in range(a, b + 1) if shooter_at[i] is not None]
        vals, cnt = np.unique(np.asarray(ids, dtype=object).astype(str), return_counts=True)
        best = vals[int(np.argmax(cnt))]
        shooter = next(x for x in ids if str(x) == best)
        others = {tid for tid, tr in track_xy.items() if tid != shooter and team.get(tid) in ("A", "B")
                  and any(i in tr for i in range(a, b + 1))}
        # the player at the line per frame (the shooter's track id may change between attempts); gaps: nearest
        at = {i: shooter_at[i] for i in range(a, b + 1) if shooter_at[i] is not None}
        keys = sorted(at)
        for i in range(a, b + 1):
            if i not in at:
                k = int(np.argmin([abs(i - j) for j in keys]))
                at[i] = at[keys[k]]
        out.append({"i0": a, "i1": b, "side": side_at[a], "shooter": shooter, "others": others, "shooter_at": at})
    return out


def greedy_holders(img: np.ndarray, size: np.ndarray, observed: np.ndarray, players_at: list[list[Holder]],
                   fps: float, cfg: BallConfig, banned: dict[int, set] | None = None) -> dict[int, Holder]:
    """Per sighting the best-scoring holder, with hysteresis for the current holder (``holder_keep_bonus``)."""
    banned = banned or {}
    keep_frames = max(1, int(round(0.5 * fps)))
    out: dict[int, Holder] = {}
    last_owner, last_i = None, -10 ** 9
    for i in range(len(observed)):
        if not observed[i] or not np.isfinite(img[i, 0]):
            continue
        best, bscore = None, np.inf
        for p in players_at[i]:
            if p[0] in banned.get(i, ()):
                continue
            sc = _holder_score(img[i, 0], img[i, 1], size[i], np.asarray(p[1], float), cfg)
            if sc is None:
                continue
            if p[0] == last_owner and i - last_i <= keep_frames:
                sc -= cfg.holder_keep_bonus
            if sc < bscore:
                best, bscore = p, sc
        if best is not None:
            out[i] = best
            last_owner, last_i = best[0], i
    return out


def decode_holders(img: np.ndarray, size: np.ndarray, observed: np.ndarray, players_at: list[list[Holder]],
                   fps: float, cfg: BallConfig, banned: dict[int, set] | None = None,
                   shot_soon: np.ndarray | None = None) -> dict[int, Holder]:
    """Best holder sequence over the frames where the ball is seen (Viterbi).

    States per sighting: each person whose box can hold the ball (``_holder_score``), or FREE (not held: flight,
    loose, at the rim, someone untracked). The ball stays with its holder unless the evidence keeps pointing
    elsewhere: changing holder costs ``holder_switch_cost``; moving it to a player further away than a hand-off
    (``handoff_m``) without a FREE sighting in between is only allowed as fast as a pass flies and costs extra
    (``holder_jump_cost``) - a single wrong overlap (the ball in front of another player's body) cannot snap the
    ball across the court and back."""
    banned = banned or {}
    idx = [i for i in range(len(observed)) if observed[i] and np.isfinite(img[i, 0])]
    if not idx:
        return {}
    FREE = None
    states: list[list[tuple[Any, float, Holder | None]]] = []
    for i in idx:
        bx, by, bs = img[i, 0], img[i, 1], size[i]
        st: list[tuple[Any, float, Holder | None]] = []
        for p in players_at[i]:
            if p[0] in banned.get(i, ()):
                continue
            sc = _holder_score(bx, by, bs, np.asarray(p[1], float), cfg)
            if sc is not None:
                st.append((p[0], sc, p))
        st.append((FREE, cfg.free_cost if st else 0.0, None))
        states.append(st)
    cost = [np.array([c for _, c, _ in states[0]])]
    back: list[np.ndarray] = [np.full(len(states[0]), -1)]
    for k in range(1, len(idx)):
        dt = (idx[k] - idx[k - 1]) / fps
        prev, cur = states[k - 1], states[k]
        T = np.zeros((len(prev), len(cur)))
        for a, (ta, _, pa) in enumerate(prev):
            for b, (tb, _, pb) in enumerate(cur):
                if ta == tb:
                    continue
                if ta is FREE or tb is FREE:
                    T[a, b] = cfg.free_switch_cost
                    continue
                d = float(np.linalg.norm(np.asarray(pa[2], float) - np.asarray(pb[2], float)))
                T[a, b] = cfg.holder_switch_cost
                if d > cfg.handoff_m and not (shot_soon is not None and shot_soon[idx[k]]):
                    # a pass nobody saw: possible, but must be physically possible and is rare (except a catch
                    # right before a shot: the shooter often has the ball for a single sighting)
                    T[a, b] += cfg.holder_jump_cost * max(0.0, 1.0 - dt / cfg.jump_free_s) \
                        if d <= cfg.handoff_m + cfg.pass_speed_mps * max(dt, 0.1) else 1e6
        tot = cost[-1][:, None] + T
        arg = np.argmin(tot, axis=0)
        cost.append(tot[arg, np.arange(len(cur))] + np.array([c for _, c, _ in cur]))
        back.append(arg)
    out: dict[int, Holder] = {}
    j = int(np.argmin(cost[-1]))
    for k in range(len(idx) - 1, -1, -1):
        tid, _, p = states[k][j]
        if tid is not FREE:
            out[idx[k]] = p
        j = int(back[k][j]) if k else j
    return out


def _guess_unknown(frames: list[BallFrame], fps: float, cfg: BallConfig, xy_of: Any) -> None:
    """Fill every remaining unknown frame (GUESSED, see the module docstring)."""
    n = len(frames)
    known = [f.i for f in frames if f.x is not None]
    if not known:
        return
    for k in range(len(known) + 1):
        a = known[k - 1] if k > 0 else None
        b = known[k] if k < len(known) else None
        lo, hi = (a + 1 if a is not None else 0), (b if b is not None else n)
        if hi <= lo:
            continue
        fa = frames[a] if a is not None else None
        fb = frames[b] if b is not None else None
        pa = None if fa is None else np.array([fa.x, fa.y], float)
        pb = None if fb is None else np.array([fb.x, fb.y], float)
        if pa is not None and pb is not None:
            tf = float(np.clip(np.linalg.norm(pb - pa) / cfg.pass_speed_mps, 0.3, cfg.max_flight_s))
            release = max(lo, hi - int(round(tf * fps)))
        else:
            release = hi
        start = pa if pa is not None else pb
        for i in range(lo, hi):
            g = frames[i]
            if i < release or pb is None:
                q = xy_of(fa.owner, i) if fa is not None else None  # still with the last holder if tracked
                p = q if q is not None else start
                if cfg.holder_decode or i == release - 1 or pb is None:
                    start = p  # (new tracker: the flight to the next sighting starts where the holder was last)
            else:
                u = (i - release + 1) / (hi - release + 1)
                p = start + (pb - start) * u
            g.x, g.y, g.state, g.source = float(p[0]), float(p[1]), GUESSED, "guess"


def _extend(frames: list[BallFrame], a: int, stop: int, fa: BallFrame, xy_of: Any, hold: bool = False) -> None:
    """Keep the ball with anchor fa's holder (or at its place) from a towards stop (exclusive), either way."""
    step = 1 if stop > a else -1
    last = np.array([fa.x, fa.y], float)
    for i in range(a + step, stop, step):
        g = frames[i]
        if g.x is not None:
            break
        p = xy_of(fa.owner, i)
        if p is None:
            if fa.owner is None and fa.source != "rim":
                break
            if fa.owner is not None and (not hold or xy_of(fa.owner, a) is None):
                break  # holder without a track: nothing to follow (old tracker: holder not tracked here any more)
            p = last  # holder briefly not tracked: stays where they were last seen (no jump back and forth)
        last = p
        g.x, g.y, g.state, g.source, g.owner = float(p[0]), float(p[1]), ESTIMATED, "follow", fa.owner


def _drop_blips(frames: list[BallFrame], fps: float) -> None:
    """A very short possession by B in the middle of A's possession is an overlap mistake: give it back to A."""
    blip = max(1, int(round(0.3 * fps)))
    poss = [f.i for f in frames if f.source == "holder"]
    runs: list[list[int]] = []
    for i in poss:
        if runs and frames[runs[-1][-1]].owner == frames[i].owner and i - runs[-1][-1] <= blip:
            runs[-1].append(i)
        else:
            runs.append([i])
    for k in range(1, len(runs) - 1):
        prev, cur, nxt = runs[k - 1], runs[k], runs[k + 1]
        if (cur[-1] - cur[0] < blip and frames[prev[-1]].owner == frames[nxt[0]].owner
                and nxt[0] - prev[-1] <= 3 * blip):
            for i in cur:
                f = frames[i]
                f.x = f.y = None
                f.state, f.owner, f.source = "", None, ""


def _drop_inconsistent(frames: list[BallFrame], fps: float, cfg: BallConfig) -> None:
    """Remove anchors the ball cannot have reached: implied speed from the previous kept anchor above
    ``max_court_speed_mps`` (+1 m slack) while the next anchor agrees with the previous one."""
    def ok(fa: BallFrame, fb: BallFrame) -> bool:
        d = float(np.hypot(fb.x - fa.x, fb.y - fa.y))  # type: ignore[operator]
        return d <= cfg.max_court_speed_mps * abs(fb.i - fa.i) / fps + 1.0

    for _ in range(3):
        anchors = [f for f in frames if f.source in ("holder", "rim", "person")]
        bad = set()
        for k in range(1, len(anchors)):
            prev, cur = anchors[k - 1], anchors[k]
            if prev.i in bad and k >= 2:
                prev = anchors[k - 2]
            if ok(prev, cur):
                continue
            nxt = anchors[k + 1] if k + 1 < len(anchors) else None
            if nxt is not None and ok(prev, nxt) and not ok(cur, nxt):
                bad.add(cur.i)  # cur is the outlier
            elif nxt is not None and ok(cur, nxt) and k >= 2 and ok(anchors[k - 2], prev):
                continue  # a real (fast) change: both sides consistent
            elif nxt is None or not ok(cur, nxt):
                bad.add(cur.i)
        if not bad:
            break
        for f in frames:
            if f.i in bad:
                f.x = f.y = None
                f.state, f.owner, f.source = "", None, ""


def estimate_ball(n: int, ball_rows: list[np.ndarray], G: list[np.ndarray], seg: np.ndarray, fps: float,
                  cfg: BallConfig, players_at: list[list[Holder]], track_xy: dict[Any, dict[int, np.ndarray]],
                  rims: dict | None = None, persons: Sequence[np.ndarray] | None = None,
                  H: Sequence[np.ndarray | None] | None = None) -> list[BallFrame]:
    """Pass-2 entry point: image trajectory + court positions (see the module docstring)."""
    track = link_ball(n, ball_rows, G, seg, fps, cfg)
    frames = ball_court_positions(track.boxes, players_at, fps, cfg, track=track, track_xy=track_xy, rims=rims,
                                  persons=persons, H=H)
    import os
    dbg = os.environ.get("AA_BALL_DEBUG")
    if dbg:  # developer diagnostics: per-frame source / state / owner
        import json
        from pathlib import Path
        Path(dbg).write_text(json.dumps([[f.source, f.state, None if f.owner is None else str(f.owner),
                                          None if f.x is None else round(f.x, 2), None if f.y is None else round(f.y, 2),
                                          None if f.img is None else [round(v) for v in f.img]] for f in frames]),
                             encoding="utf-8")
    return frames


def holders_from_rows(rows: Sequence[Any], idx_of_k: dict[int, int], n: int
                      ) -> tuple[list[list[Holder]], dict[Any, dict[int, np.ndarray]]]:
    """From pass-2 output rows (export.TrackRow): people who can hold the ball per frame (players and referees
    with a measured box) and every track's court position per frame."""
    holders: list[list[Holder]] = [[] for _ in range(n)]
    track_xy: dict[Any, dict[int, np.ndarray]] = {}
    for r in rows:
        if r.object_type not in ("player", "referee"):
            continue
        i = idx_of_k.get(int(r.k))
        if i is None:
            continue
        xy = np.array([r.x, r.y], float)
        track_xy.setdefault(r.track_id, {})[i] = xy
        if r.box is not None:
            holders[i].append((r.track_id, np.asarray(r.box, float), xy, r.team))
    return holders, track_xy


def boxes_by_frame(rows: np.ndarray, ks: Sequence[int]) -> list[np.ndarray]:
    """Pass-1 rows [k, x1, y1, x2, y2, ...] -> (m, 4) boxes per analysed frame k in ``ks``."""
    rows = rows[np.argsort(rows[:, 0], kind="stable")]
    k = np.asarray(ks, float)
    lo, hi = np.searchsorted(rows[:, 0], k, "left"), np.searchsorted(rows[:, 0], k, "right")
    return [rows[a:b, 1:5] for a, b in zip(lo, hi)]
