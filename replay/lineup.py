"""On-court lineup: 5 continuous player slots per team (pass 2, after teams are known, before smoothing).

Basketball facts used here: during live play each team has exactly 5 players on the court, and substitutions
happen only in dead balls (breaks, time-outs, free throws, stoppages). The tracker (tracklets.py /
court_tracker.py) produces many shorter identities per real player (a player leaves the picture, is hidden, the
tracker loses them); this module chains them into 10 slots (A1-A5, B1-B5). A slot is one real player between
substitutions: it persists while the player is off screen and is re-attached when a piece of track continues it
(position reachable at running speed from where the slot was last seen, similar jersey/shorts colour and body,
the same jersey number).

Algorithm (offline, whole clip):
  * pieces = the kept team-A/B tracks (one identity each); quality = detections x team share,
  * pieces are inserted best-first into the slot timelines of their team: a piece may go into a slot only where
    it does not overlap the slot's pieces in time (a short overlap at the same spot is the same person under two
    IDs), and only if the jump from the previous piece and to the next one is physically reachable; among the
    feasible slots the one with the smallest added continuity cost wins (position residual, colour, gap length,
    jersey numbers: same number = strong bonus, different numbers = a different player, allowed only across a
    dead ball),
  * pieces that fit no slot as a whole are cut where a slot is free; leftovers may join the other team's free
    slots (mis-teamed); the rest are extra people of that team at that moment (bench, duplicate). By default they
    keep their own track id and the later per-frame caps decide (drop_extras=True removes them).
Output: track ids of the kept detections are replaced by slot ids, so smoothing fills short gaps between the
pieces of a slot (flagged "interpolated" = position predicted) and longer gaps stay empty (slot unknown there).
``results/lineup.json`` lists which pieces form each slot, the slot numbers and the dead-ball windows used.

On by default (since 13.10); environment variable REPLAY_LINEUP=0 turns it off, 1 on (off = every track is its own
identity, the old behaviour).
"""

from __future__ import annotations

import bisect
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

TEAMS = ("A", "B")
PRED_STRIDE = 3  # samples between exact motion-prior predictions of an out-of-view slot
PRED_MAX_SPEED_MPS = 8.0  # a predicted dot never moves faster than a sprinting player (debug run 19.10: the
# motion prior pulled off-court dots back onto the court at 11+ m/s)
REID_ROWS = 12  # embeddings kept per piece
REID_MIN_ROWS = 3  # fewer embedded crops: no re-ID opinion (single crops are noisy)


def reid_sim(p: "_Piece", q: "_Piece") -> float | None:
    """Median pairwise cosine similarity of two pieces' re-ID embeddings (replay/reid.py); None = unknown."""
    if p.emb is None or q.emb is None:
        return None
    return float(np.median(p.emb @ q.emb.T))
_APP_SIGMA = np.array([12.0, 7.0, 7.0, 14.0, 8.0, 8.0])  # Lab torso + shorts noise (as tracklets.py)


@dataclass
class LineupConfig:
    enabled: bool = True  # ON since 13.10 (gate c300 seeds 3: IDs/min 88.6 -> 81.5 better, transfers 5 -> 6 within
    # noise, recall/team acc equal); REPLAY_LINEUP=0 turns it off
    slots_per_team: int = 5
    max_speed_mps: float = 7.0  # reach between two pieces of one slot (running speed)
    reach_slack_m: float = 1.5  # foot-point noise
    walk_mps: float = 2.0  # typical displacement rate used to score a residual (not a gate)
    overlap_tol_s: float = 0.6  # two pieces of one slot may overlap this long (same person, two IDs)
    new_slot_cost: float = 3.0  # cost of opening an empty slot instead of continuing one
    gap_cost_per_s: float = 0.03
    app_weight: float = 0.4
    number_bonus: float = 6.0
    sub_cost: float = 2.0  # a different jersey number across a dead ball (substitution)
    # soft limits: a piece that fits no slot cleanly is still a visible player (the slot then has an identity
    # break, cost below). Only time overlap is a hard limit (one slot = one person at a time).
    jump_penalty: float = 20.0  # unreachable jump between two pieces of a slot (when hard_jump is False)
    hard_jump: bool = True  # unreachable jump = never the same slot (soft penalties caused ID transfers)
    live_sub_penalty: float = 20.0  # different jersey numbers without a dead ball in between
    dead_gap_s: float = 2.0  # a hole in the sample times this long = skipped break (pass 1 skips breaks)
    still_mps: float = 0.7  # median player speed below this ...
    still_min_s: float = 4.0  # ... this long = stoppage (dead ball)
    split_rounds: int = 0  # >0: pieces that fit no slot as a whole are cut where slots are free (rounds); 0 since
    # 13.10: cutting a tracker track into slot + extra parts gave it two output ids (IDs/min up)
    min_split_len: int = 3  # samples
    gap_split_s: float = float("inf")  # cut tracks into pieces at internal gaps longer than this (1.0 tried:
    # re-deciding the tracker's own links added ID transfers; inf = trust the tracker's identities)
    same_track_bonus: float = 4.0  # two pieces of one tracker identity continuing each other in a slot
    # a 6th person of a team while the other team has a free slot is most likely mis-teamed (team accuracy is
    # ~95 %): such leftover pieces may join the other team's slots at this extra cost (None = never)
    flip_team_cost: float | None = None  # 3.0 tried: team accuracy fell (p4400 95.8 -> 91.8 %)
    # smoothing: a slot's samples are smoothed in parts split at piece boundaries ("tid": wherever the tracker
    # identity changes; "jump": only where the change is not physically reachable; "none"), so the smoother does
    # not reject the first foot points of a new piece as outliers of the previous one
    smooth_break: str = "jump"  # chains only join confident pieces, so bridge them unless the jump is impossible
    # output ids: a slot (the <= 5-per-team constraint) carries ONE output id only across confident re-attachments
    # (same jersey number, same tracker id, or a gap <= confident_gap_s that is reachable); otherwise the slot
    # continues under a new output id - a wrong re-attachment would move an id to another person (ID transfer)
    chain_ids: bool = True
    confident_gap_s: float = 4.0  # 2/3/4/8 s measured (13.10): 4 s = best ID cut for +0..2 transfers
    confident_speed_mps: float = 7.0  # ... and the jump fits this speed
    # re-ID (replay/reid.py embeddings, when cached): median pairwise cosine of two pieces. Same player ~0.9,
    # same-team other player ~0.66 (notes_reid.md). Below reid_min_sim a re-attachment is never confident
    # (teammates crossing out of view); at/above reid_strong_sim it is confident up to reid_long_gap_s
    use_reid: bool = True
    reid_min_sim: float = 0.75
    reid_strong_sim: float = 0.85
    reid_long_gap_s: float = 20.0
    reid_weight: float = 5.0  # slot cost += reid_weight * (0.8 - sim)
    # without re-ID embeddings (normal app builds): one output id only across same number / same tracker id or
    # a gap <= noreid_gap_s at <= noreid_speed_mps (13.10: 1 s kept transfers at the OFF level, 4 s did not)
    noreid_gap_s: float = 1.0
    crumb_max_samples: int = 0  # 0 = off. 5 tried 18.10: recall 89 -> 57 % (crumb chains smoothed as one id) - REVERTED
    crumb_gap_s: float = 3.0  # ... up to this long (they carry too little evidence for a new identity)
    number_min_reads: int = 5  # same number makes a re-attachment confident only with this many reads on both sides
    noreid_speed_mps: float = 3.0
    # off-screen slots: after a slot's last output row, keep showing a predicted position from the game-aware
    # motion prior (court_tracker.MotionPrior) while its uncertainty stays below this (metres); then unknown
    predict_max_sigma_m: float = 2.5
    rank_weight: float = 0.7  # insertion order: detections x team share x exp(w * player-likeness rank (roster))
    # exactly 5 per team in live play: extra pieces lose their detections at live samples where all 5 slots of the
    # team are busy (a slot piece spans the sample), and out-of-view predictions never make a 6th row; dead balls
    # (substitutions, benches walking on) are left to the final caps. Env REPLAY_LINEUP_EXACT5=1/0 overrides.
    exact5: bool = False
    exact5_hold: int = 2  # samples a seen slot player keeps the slot busy (hysteresis against flicker)
    exact5_dup_m: float = 0.8  # mode 2: two ids this close at the same samples = one person twice
    exact5_min_rank: float = 0.0  # mode 2: ids with a lower mean roster rank in a stretch are not counted (caps)
    drop_extras: bool = False  # True: drop leftover pieces; False: they keep their own track id (the per-frame
    # caps decide) - measured: dropping lost 1.5-3 recall points, keeping costs more output ids


@dataclass
class _Piece:
    tid: int
    team: str
    i: np.ndarray  # sample indices (sorted, unique)
    xy: np.ndarray
    feat: np.ndarray
    quality: float
    number: int = -1
    share: float = 0.5
    emb: np.ndarray | None = None  # up to REID_ROWS re-ID embeddings (L2-normalised) of this piece
    reads: int = 0  # jersey-number reads supporting ``number``

    @property
    def i0(self) -> int:
        return int(self.i[0])

    @property
    def i1(self) -> int:
        return int(self.i[-1])

    def start(self) -> np.ndarray:
        return np.median(self.xy[:3], axis=0)

    def end(self) -> np.ndarray:
        return np.median(self.xy[-3:], axis=0)


# slot id -> sample indices where the slot's smoothing starts a new part (set by apply, used by smooth_slot)
BREAKS: dict[int, np.ndarray] = {}
SLOT_LABEL: dict[int, str] = {}  # slot id -> "A1".."B5" (set by apply, used by add_track_info)


UNSURE_DET: list[np.ndarray] = []  # exact-5 mode 3: per-detection "unsure" mask of the last apply
UNSURE_RANGES: dict[int, list[tuple[int, int]]] = {}  # tid after lineup -> sample ranges where it is unsure
CHAIN_SPAN: dict[int, tuple[float, float]] = {}  # slot id -> (t first, t last) of its pieces
DEAD: list[tuple[float, float]] = []  # dead-ball windows (s) of the last apply
OUT: list[Path] = []  # results folder of the last apply (lineup.json gets the profiles)
PLAYER_ID: dict[int, int] = {}  # output id -> continuous player id (set by add_track_info, see player_id_of)


def player_id_of(output_id) -> int | None:
    """Continuous player id of an output track id (survives output-id breaks), or None when unknown: all output
    ids with the same player profile ("A#23"), and all output ids of one lineup slot ("A1".."B5") with no dead
    ball between them, share one id (numbered 1.. by first appearance). Set by add_track_info."""
    try:
        return PLAYER_ID.get(int(output_id))
    except (TypeError, ValueError):
        return None


def _assign_player_ids(track_info: dict, oid_of: dict, span: dict, dead_between) -> None:
    """Union of (same profile) and (same slot, consecutive pieces, no dead ball in between) -> PLAYER_ID."""
    parent: dict[int, int] = {}

    def find(a: int) -> int:
        while parent.setdefault(a, a) != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    by_profile: dict[str, list[int]] = {}
    for oid, v in track_info.items():
        if v.get("profile"):
            by_profile.setdefault(v["profile"], []).append(int(oid))
    for oids in by_profile.values():
        for o in oids:
            union(oids[0], o)
    by_slot: dict[str, list[int]] = {}
    for sid, oid in oid_of.items():
        if oid in span:
            by_slot.setdefault(SLOT_LABEL[sid], []).append(int(oid))
    for oids in by_slot.values():
        oids.sort(key=lambda o: span[o][0])
        find(oids[0])
        for a, b in zip(oids, oids[1:]):
            if not dead_between(span[a][1], span[b][0]):
                union(a, b)
            else:
                find(b)
    first: dict[int, float] = {}
    for o in parent:
        r = find(o)
        t0 = span[o][0] if o in span else float("inf")
        first[r] = min(first.get(r, float("inf")), t0)
    number = {r: j + 1 for j, r in enumerate(sorted(first, key=lambda r: (first[r], r)))}
    for o in parent:
        PLAYER_ID[o] = number[find(o)]
        if o in track_info:
            track_info[o]["player_id"] = PLAYER_ID[o]


def add_track_info(track_info: dict, new_id: dict) -> None:
    """Pass-2 hook (after the jersey numbers are in track_info): ``slot`` = "A1".."B5" and persistent player
    profiles. A profile = one real player through the whole clip: all output ids of a team with the same accepted
    jersey number, plus number-less ids of the SAME slot that touch such an id with no dead ball in between
    (substitutions only happen in dead balls; cardinality 5 per team) and no time overlap with another id of that
    number. Fields: ``profile`` ("A#23"), ``profile_source`` ("number" | "lineup"); inferred ids also get
    ``number`` (+ ``number_source`` "lineup") so roster names (roster_names.py) reach them; ``player_id``
    (continuous player id, see player_id_of). Nothing when off."""
    PLAYER_ID.clear()
    if not SLOT_LABEL:
        return
    oid_of = {sid: new_id[sid] for sid in SLOT_LABEL if sid in new_id and new_id[sid] in track_info}
    for sid, oid in oid_of.items():
        track_info[oid]["slot"] = SLOT_LABEL[sid]
    span = {oid_of[s]: CHAIN_SPAN.get(s) for s in oid_of if CHAIN_SPAN.get(s)}
    num_of = {}
    for oid, v in track_info.items():
        if v.get("number") is not None and v.get("team") in TEAMS:
            num_of[oid] = str(v["number"])
            v["profile"], v["profile_source"] = f"{v['team']}#{v['number']}", "number"

    def dead_between(a: float, b: float) -> bool:
        lo, hi = min(a, b), max(a, b)
        return any(d0 < hi and d1 > lo for d0, d1 in DEAD)

    def overlaps(oid: int, number: str, team: str) -> bool:
        a = span.get(oid)
        if a is None:
            return True
        for o2, n2 in num_of.items():
            b = span.get(o2)
            if o2 != oid and n2 == number and track_info[o2].get("team") == team and (
                    b is None or (a[0] < b[1] and b[0] < a[1])):
                return True
        return False

    by_slot: dict[str, list[int]] = {}
    for sid, oid in oid_of.items():
        if oid in span:
            by_slot.setdefault(SLOT_LABEL[sid], []).append(oid)
    for label, oids in by_slot.items():
        oids.sort(key=lambda o: span[o][0])
        for _ in range(2):  # forwards then backwards: a number spreads along the slot until a dead ball
            for a, b in zip(oids, oids[1:]):
                for src, dst in ((a, b), (b, a)):
                    n = num_of.get(src)
                    if n is None or dst in num_of:
                        continue
                    team = track_info[src]["team"]
                    gap = (span[a][1], span[b][0])
                    if track_info[dst].get("team") != team or dead_between(*gap) or overlaps(dst, n, team):
                        continue
                    num_of[dst] = n
                    track_info[dst].update({"number": n, "number_source": "lineup", "profile": f"{team}#{n}",
                                            "profile_source": "lineup"})
            oids.reverse()
    if OUT:
        try:
            path = OUT[0] / "lineup.json"
            doc = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
            prof: dict[str, list] = {}
            for oid, v in track_info.items():
                if v.get("profile"):
                    prof.setdefault(v["profile"], []).append(oid)
            doc["profiles"] = {k: sorted(v) for k, v in sorted(prof.items())}
            path.write_text(json.dumps(doc, indent=1), encoding="utf-8")
        except (OSError, ValueError):
            pass
    _assign_player_ids(track_info, oid_of, span, dead_between)
    n_inf = sum(1 for v in track_info.values() if v.get("profile_source") == "lineup")
    print(f"lineup: {len({v['profile'] for v in track_info.values() if v.get('profile')})} player profiles, "
          f"{n_inf} ids got their number from the lineup")


@dataclass
class LineupResult:
    tid: np.ndarray
    keep: np.ndarray
    slots: dict[str, list[_Piece]] = field(default_factory=dict)
    slot_id: dict[str, int] = field(default_factory=dict)
    extras: list[_Piece] = field(default_factory=list)
    dead: list[tuple[float, float, str]] = field(default_factory=list)
    chains: list[tuple[str, int, list[_Piece]]] = field(default_factory=list)  # (slot label, output id, pieces)


def exact5_mode(cfg: LineupConfig) -> int:
    """0 off, 1 = drop extras where all 5 slots are busy, 2 = evict the least player-like piece per >5 stretch."""
    env = os.environ.get("REPLAY_LINEUP_EXACT5")
    if env is not None:
        e = env.strip().lower()
        return int(e) if e in ("2", "3") else (0 if e in ("0", "false", "no", "off", "") else 1)
    return 1 if cfg.exact5 else 0


def exact5_on(cfg: LineupConfig) -> bool:
    return exact5_mode(cfg) > 0


def evict_stretches(tid: np.ndarray, keep: np.ndarray, det_i: np.ndarray, det_xy: np.ndarray, t: np.ndarray,
                    team_of, share: dict, rank: np.ndarray | None, dead: np.ndarray, cfg: LineupConfig) -> np.ndarray:
    """Exact-5 mode 2: for each live stretch where a team has more than 5 identities present (seen within
    +-exact5_hold samples), score every identity over the WHOLE stretch - roster rank (kit/bench zone/on-court
    position/posture evidence, roster.cap_rank), team share, duplicate overlap with a team-mate (< dup_m at the
    same samples: the weaker of the pair is penalised) - and remove the least player-like one for the whole
    stretch (also if it would hold a slot); repeat until <= 5. Returns the new keep mask."""
    keep = keep.copy()
    unsure = np.zeros(len(tid), bool)
    n = len(t)
    rk = np.zeros(len(tid)) if rank is None or len(rank) != len(tid) else np.nan_to_num(np.asarray(rank, float))
    h = cfg.exact5_hold
    n_evicted = 0
    for team in TEAMS:
        sel = np.flatnonzero(keep & (tid > 0) & np.array([team_of(g) == team for g in tid]))
        if not len(sel):
            continue
        for _ in range(3):  # stretches can merge after evictions; a few passes are enough
            present: dict[int, np.ndarray] = {}
            for g in np.unique(tid[sel]):
                ii = det_i[sel[tid[sel] == g]]
                b = np.zeros(n, bool)
                for d in range(-h, h + 1):
                    j = ii + d
                    b[j[(j >= 0) & (j < n)]] = True
                present[int(g)] = b
            cnt = np.sum(list(present.values()), axis=0) if present else np.zeros(n)
            over = (cnt > cfg.slots_per_team) & ~dead
            if not over.any():
                break
            idx = np.flatnonzero(over)
            runs = np.split(idx, np.flatnonzero(np.diff(idx) > 2 * h + 1) + 1)
            changed = False
            for run in runs:
                a, b = int(run[0]), int(run[-1])
                while True:
                    inside = [g for g, pr in present.items() if pr[a:b + 1].any()]
                    if len(inside) <= cfg.slots_per_team:
                        break
                    rank_of, pos = {}, {}
                    for g in inside:
                        m = sel[(tid[sel] == g) & (det_i[sel] >= a) & (det_i[sel] <= b) & keep[sel]]
                        rank_of[g] = float(np.mean(rk[m])) if len(m) else -9.0
                        pos[g] = {int(i): det_xy[d] for i, d in zip(det_i[m], m)}
                    score = {g: rank_of[g] + np.log(max(0.05, share.get(g, 0.5))) for g in inside}
                    weaker_dup = set()
                    for x in inside:  # duplicates: two ids of one person at the same samples
                        for y in inside:
                            if x < y:
                                both = set(pos[x]) & set(pos[y])
                                if len(both) >= 3 and np.median([np.linalg.norm(pos[x][i] - pos[y][i]) for i in both])                                         < cfg.exact5_dup_m:
                                    weaker_dup.add(min((x, y), key=lambda g: score[g]))
                    # fix 3: count only ids that would be SHOWN - not bench-like (rank <= exact5_min_rank) and not the
                    # weaker box of a duplicate pair; those are left to the final caps, never counted or evicted here
                    shown = [g for g in inside if rank_of[g] > cfg.exact5_min_rank and g not in weaker_dup]
                    if len(shown) <= cfg.slots_per_team:
                        break
                    c = np.sum([present[g][a:b + 1] for g in shown], axis=0)
                    if c.max() <= cfg.slots_per_team:
                        break
                    worst = min(shown, key=lambda g: score[g])
                    m = (tid == worst) & keep & (det_i >= a) & (det_i <= b)
                    if exact5_mode(cfg) == 3:  # keep the detections, mark them unsure (not drawn)
                        unsure[m] = True
                        sel = sel[~unsure[sel]]
                    else:
                        keep[m] = False
                    n_evicted += int(m.sum())
                    present[worst][a:b + 1] = False
                    changed = True
            sel = sel[keep[sel]]
            if not changed:
                break
    print(f"lineup: exact 5 (stretches) - {n_evicted} detections of the least player-like ids "
          f"{'marked unsure' if exact5_mode(cfg) == 3 else 'evicted'}")
    UNSURE_DET.clear()
    UNSURE_DET.append(unsure)
    return keep


def enabled(cfg: LineupConfig | None = None) -> bool:
    env = os.environ.get("REPLAY_LINEUP")
    if env is not None:
        return env.strip() not in ("0", "false", "no", "off", "")
    return (cfg or LineupConfig()).enabled


# ------------------------------------------------------------------------------------------- dead balls
def dead_mask(t: np.ndarray, det_i: np.ndarray, det_xy: np.ndarray, tid: np.ndarray, sel: np.ndarray,
              cfg: LineupConfig) -> tuple[np.ndarray, list[tuple[float, float, str]]]:
    """Per sample: True = dead ball (a substitution may happen). Sources: holes in the sample times (pass 1
    skipped a break) and stretches where the players barely move (free throws, time-outs, stoppages)."""
    n = len(t)
    dead = np.zeros(n, bool)
    windows: list[tuple[float, float, str]] = []
    if n == 0:
        return dead, windows
    gaps = np.flatnonzero(np.diff(t) > cfg.dead_gap_s)
    for g in gaps:
        dead[g:g + 2] = True
        windows.append((float(t[g]), float(t[g + 1]), "break"))
    # median player speed per sample
    speeds: list[list[float]] = [[] for _ in range(n)]
    ii, xy, tt = det_i[sel], det_xy[sel], tid[sel]
    o = np.lexsort((ii, tt))
    ii, xy, tt = ii[o], xy[o], tt[o]
    same = (tt[1:] == tt[:-1]) & (ii[1:] > ii[:-1])
    dt = t[ii[1:]] - t[ii[:-1]]
    ok = same & (dt > 0) & (dt <= 0.6)
    v = np.linalg.norm(xy[1:] - xy[:-1], axis=1) / np.where(dt > 0, dt, 1.0)
    for j in np.flatnonzero(ok):
        speeds[int(ii[j + 1])].append(float(v[j]))
    med = np.array([np.median(s) if len(s) >= 3 else np.nan for s in speeds])
    fps = 1.0 / max(1e-6, float(np.median(np.diff(t)))) if n > 1 else 10.0
    w = max(1, int(round(1.0 * fps)))
    sm = np.full(n, np.nan)
    for i in range(n):
        seg = med[max(0, i - w):i + w + 1]
        seg = seg[np.isfinite(seg)]
        if len(seg) >= max(2, w // 2):
            sm[i] = np.median(seg)
    still = np.isfinite(sm) & (sm < cfg.still_mps)
    i = 0
    while i < n:
        if not still[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and still[j + 1] and t[j + 1] - t[j] <= cfg.dead_gap_s:
            j += 1
        if t[j] - t[i] >= cfg.still_min_s:
            dead[i:j + 1] = True
            windows.append((float(t[i]), float(t[j]), "still"))
        i = j + 1
    return dead, sorted(windows)


# ------------------------------------------------------------------------------------------- costs
def _app_dist(a: np.ndarray, b: np.ndarray) -> float:
    if not (np.isfinite(a).all() and np.isfinite(b).all()) or len(a) < 6 or len(b) < 6:
        return 1.0
    z = (a[:6] - b[:6]) / _APP_SIGMA
    return float(np.mean(np.minimum(z ** 2, 9.0)))


class _Solver:
    def __init__(self, t: np.ndarray, dead: np.ndarray, cfg: LineupConfig):
        self.t, self.cfg = t, cfg
        self.dead_cum = np.r_[0, np.cumsum(dead)]

    def dead_between(self, a: int, b: int) -> bool:
        lo, hi = min(a, b), max(a, b)
        return bool(self.dead_cum[hi + 1] - self.dead_cum[lo] > 0)

    def cont(self, p: _Piece, q: _Piece) -> float:
        """Cost of q continuing p in one slot (p starts first); inf = impossible."""
        c = self.cfg
        dt = float(self.t[q.i0] - self.t[p.i1])
        if dt < -c.overlap_tol_s:
            return np.inf
        if dt < 0:  # short overlap: must be the same spot at the same time
            j = min(int(np.searchsorted(p.i, q.i0)), len(p.i) - 1)
            d = float(np.linalg.norm(p.xy[j] - q.xy[0]))
            return 0.5 * d if d <= c.reach_slack_m else np.inf
        d = float(np.linalg.norm(p.end() - q.start()))
        cost = min(0.5 * (d / (1.0 + c.walk_mps * dt)) ** 2, c.jump_penalty) + c.gap_cost_per_s * dt \
            + c.app_weight * _app_dist(p.feat, q.feat)
        if d > c.max_speed_mps * dt + c.reach_slack_m:
            if c.hard_jump:
                return np.inf  # unreachable from where the slot was last seen: another person
            cost += c.jump_penalty
        if p.tid == q.tid:
            cost -= c.same_track_bonus
        elif c.use_reid:
            sim = reid_sim(p, q)
            if sim is not None:
                cost += c.reid_weight * (0.8 - sim)
        if p.number >= 0 and q.number >= 0:
            if p.number == q.number:
                cost -= c.number_bonus
            elif self.dead_between(p.i1, q.i0):
                cost += c.sub_cost
            else:
                cost += c.live_sub_penalty
        return cost

    def insert_cost(self, slot: list[_Piece], q: _Piece) -> tuple[float, int]:
        """Added cost of putting q into the slot timeline (sorted by start) and the insert position."""
        if not slot:
            return self.cfg.new_slot_cost, 0
        pos = bisect.bisect_left([p.i0 for p in slot], q.i0)
        prev = slot[pos - 1] if pos > 0 else None
        nxt = slot[pos] if pos < len(slot) else None
        # q must not overlap any other piece of the slot beyond the tolerance
        if prev is not None and pos > 1 and self.t[slot[pos - 2].i1] - self.t[q.i0] > self.cfg.overlap_tol_s:
            return np.inf, pos
        cost = 0.0
        if prev is not None:
            a = self.cont(prev, q)
            if not np.isfinite(a):
                return np.inf, pos
            cost += a
        if nxt is not None:
            if self.t[q.i1] - self.t[nxt.i0] > self.cfg.overlap_tol_s:
                return np.inf, pos
            b = self.cont(q, nxt)
            if not np.isfinite(b):
                return np.inf, pos
            cost += b
        if prev is not None and nxt is not None:
            old = self.cont(prev, nxt)
            cost -= old if np.isfinite(old) else 0.0
        return cost, pos


# ------------------------------------------------------------------------------------------- main
def _insert_all(solver: _Solver, slots: dict[str, list[_Piece]], pieces: list[_Piece], penalty: float = 0.0
                ) -> list[_Piece]:
    extras = []
    for q in sorted(pieces, key=lambda p: (-p.quality, p.i0, p.tid)):
        best, best_cost, best_pos = None, np.inf, 0
        for label, slot in slots.items():
            if label[0] != q.team:
                continue
            cst, pos = solver.insert_cost(slot, q)
            cst += penalty
            if cst < best_cost:
                best, best_cost, best_pos = label, cst, pos
        if best is None:
            extras.append(q)
        else:
            slots[best].insert(best_pos, q)
    return extras


def _sub(q: _Piece, idx: np.ndarray) -> _Piece:
    return _Piece(q.tid, q.team, q.i[idx], q.xy[idx], q.feat, len(idx) * max(0.05, q.share), q.number, q.share,
                  q.emb)


def _split_extras(extras: list[_Piece], slots: dict[str, list[_Piece]], n: int, min_len: int) -> tuple[
        list[_Piece], list[_Piece]]:
    """Cut pieces that fit no slot as a whole into runs of samples where the same set of their team's slots is
    free (a piece overlapping a slot only partly is still a visible player the rest of the time). Returns
    (sub-pieces to insert, leftover parts that stay extra)."""
    busy = {}
    for label, ps in slots.items():
        b = np.zeros(n + 1, bool)
        for p in ps:
            b[p.i0:p.i1 + 1] = True
        busy[label] = b
    subs, rest = [], []
    for q in extras:
        labels = [lab for lab in slots if lab[0] == q.team]
        free = np.array([[not busy[lab][i] for lab in labels] for i in q.i], bool).reshape(len(q.i), len(labels))
        key = [tuple(r) for r in free]
        start = 0
        for j in range(1, len(q.i) + 1):
            if j == len(q.i) or key[j] != key[start] or q.i[j] - q.i[j - 1] > 5:
                idx = np.arange(start, j)
                if any(key[start]) and len(idx) >= min_len:
                    subs.append(_sub(q, idx))
                else:
                    rest.append(_sub(q, idx))
                start = j
    return subs, rest


def solve(pieces: list[_Piece], t: np.ndarray, dead: np.ndarray, cfg: LineupConfig
          ) -> tuple[dict[str, list[_Piece]], list[_Piece]]:
    solver = _Solver(t, dead, cfg)
    slots = {f"{team}{j + 1}": [] for team in TEAMS for j in range(cfg.slots_per_team)}
    extras = _insert_all(solver, slots, pieces)
    for _ in range(cfg.split_rounds):
        if not extras:
            break
        subs, rest = _split_extras(extras, slots, len(t), cfg.min_split_len)
        if not subs:
            break
        extras = rest + _insert_all(solver, slots, subs)
    if cfg.flip_team_cost is not None and extras:
        other = {TEAMS[0]: TEAMS[1], TEAMS[1]: TEAMS[0]}
        flipped = [_Piece(p.tid, other[p.team], p.i, p.xy, p.feat, p.quality, -1, p.share, p.emb)
                   for p in extras if len(p.i) >= cfg.min_split_len]
        short = [p for p in extras if len(p.i) < cfg.min_split_len]
        subs, rest = _split_extras(flipped, slots, len(t), cfg.min_split_len)
        left = rest + _insert_all(solver, slots, subs, cfg.flip_team_cost)
        extras = short + [_Piece(p.tid, other[p.team], p.i, p.xy, p.feat, p.quality, p.number, p.share, p.emb)
                          for p in left]
    return slots, extras


def build_pieces(tid: np.ndarray, keep: np.ndarray, det_i: np.ndarray, det_xy: np.ndarray, det_feat: np.ndarray,
                 team_of, share: dict, numbers: dict | None = None, t: np.ndarray | None = None,
                 gap_split_s: float = 1.0, rank: np.ndarray | None = None, rank_weight: float = 0.0,
                 det_emb: np.ndarray | None = None) -> list[_Piece]:
    out = []
    for g in np.unique(tid[keep]):
        g = int(g)
        team = team_of(g)
        if g <= 0 or team not in TEAMS:
            continue
        m = np.flatnonzero(keep & (tid == g))
        o = np.argsort(det_i[m], kind="stable")
        m = m[o]
        ii, u = np.unique(det_i[m], return_index=True)
        m = m[u]
        f = det_feat[m]
        f = f[np.isfinite(f).all(axis=1)] if f.ndim == 2 else f
        feat = np.median(f, axis=0) if len(f) else np.full(det_feat.shape[1] if det_feat.ndim == 2 else 6, np.nan)
        sh = float(share.get(g, 0.5))
        num = -1
        reads = 0
        if numbers and g in numbers and numbers[g] is not None:
            num = int(getattr(numbers[g], "number", -1))
            reads = int(getattr(numbers[g], "support", 0))
        xy = det_xy[m].astype(float)
        ii = ii.astype(int)
        cuts = np.flatnonzero(np.diff(t[ii]) > gap_split_s) + 1 if t is not None else []
        rk = None if rank is None or len(rank) != len(det_i) else np.nan_to_num(np.asarray(rank, float)[m])
        for idx in np.split(np.arange(len(ii)), cuts):
            w = 1.0 if rk is None else float(np.exp(rank_weight * np.clip(np.mean(rk[idx]), -3.0, 3.0)))
            emb = None
            if det_emb is not None:
                e = det_emb[m[idx]]
                e = e[np.isfinite(e).all(axis=1)]
                if len(e) >= REID_MIN_ROWS:
                    e = e[np.linspace(0, len(e) - 1, min(len(e), REID_ROWS)).astype(int)]
                    emb = e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-6)
            out.append(_Piece(g, team, ii[idx], xy[idx], feat, len(idx) * max(0.05, sh) * w, num, sh, emb, reads))
    return out


def apply(out: Path | None, tid: np.ndarray, keep: np.ndarray, det_i: np.ndarray, det_xy: np.ndarray,
          det_feat: np.ndarray, det_num: np.ndarray | None, t: np.ndarray, teams, cfg: LineupConfig | None = None,
          rank: np.ndarray | None = None, det_row: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Pass-2 hook. Returns (tid, keep) with team-A/B identities replaced by slot ids (new ids above every
    existing one; teams.track_team / track_share get entries for them) and extra people dropped. Referees and
    team-less tracks are left unchanged. Never raises: on any problem the inputs are returned unchanged."""
    import time
    t_start = time.perf_counter()
    cfg = cfg or LineupConfig()
    BREAKS.clear()
    SLOT_LABEL.clear()
    UNSURE_DET.clear()
    UNSURE_RANGES.clear()
    CHAIN_SPAN.clear()
    DEAD.clear()
    OUT.clear()
    if not enabled(cfg):
        return tid, keep
    over = os.environ.get("REPLAY_LINEUP_CFG")  # experiments: JSON overrides of LineupConfig fields
    if over:
        from dataclasses import replace
        cfg = replace(cfg, **json.loads(over))
    try:
        det_emb = _det_embeddings(out, det_row, len(det_i)) if cfg.use_reid else None
        if det_emb is None:  # no re-ID: positions alone made wrong re-attachments (ID transfers) -> conservative
            from dataclasses import replace
            cfg = replace(cfg, confident_gap_s=cfg.noreid_gap_s, confident_speed_mps=cfg.noreid_speed_mps)
        res = assign(tid, keep, det_i, det_xy, det_feat, det_num, t, teams, cfg, rank, det_emb)
    except Exception as e:  # noqa: BLE001 - the lineup is an improvement, never a reason to fail the build
        print("lineup failed:", type(e).__name__, e)
        return tid, keep
    solver = _Solver(np.asarray(t, float), np.zeros(len(t), bool), cfg)
    for label, sid, ps in res.chains:
        BREAKS[sid] = np.array(_breaks(ps, solver, cfg), int)
        SLOT_LABEL[sid] = label
        CHAIN_SPAN[sid] = (float(t[ps[0].i0]), float(max(t[p.i1] for p in ps)))
        teams.track_team[sid] = label[0]
        w = sum(len(p.i) for p in ps)
        teams.track_share[sid] = sum(p.share * len(p.i) for p in ps) / w if w else 0.5
    DEAD.extend((a, b) for a, b, _ in res.dead)
    if out is not None:
        OUT.append(Path(out))
    n_ex = int(sum(len(p.i) for p in res.extras))
    print(f"lineup: {sum(len(v) for v in res.slots.values())} pieces into {len(res.slot_id)} slots "
          f"({len(res.chains)} output ids), "
          f"{len(res.extras)} extra pieces ({n_ex} detections) {'dropped' if cfg.drop_extras else 'kept with own ids'}, "
          f"{len(res.dead)} dead-ball windows, {time.perf_counter() - t_start:.2f} s")
    dump = os.environ.get("REPLAY_LINEUP_DUMP")  # debugging: inputs and outputs of the lineup step
    if dump and out is not None:
        np.savez_compressed(Path(out) / "lineup_dump.npz", det_i=det_i, det_xy=det_xy, tid_in=tid, keep_in=keep,
                            tid_out=res.tid, keep_out=res.keep)
    if out is not None:
        try:
            save(Path(out), res, t, slot_reads(res, det_i, det_num, t))
        except OSError as e:
            print("lineup: could not write lineup.json:", e)
    return res.tid, res.keep


def _breaks(ps: list[_Piece], solver: _Solver, cfg: LineupConfig) -> list[int]:
    out = []
    for p, q in zip(ps, ps[1:]):
        if cfg.smooth_break == "tid" and p.tid != q.tid:
            out.append(q.i0)
        elif cfg.smooth_break == "jump":
            dt = float(solver.t[q.i0] - solver.t[p.i1])
            d = float(np.linalg.norm(p.end() - q.start()))
            if dt <= 0 or d > cfg.max_speed_mps * dt + cfg.reach_slack_m:
                out.append(q.i0)
    return out


def smooth_slot(track: int, k: np.ndarray, z: np.ndarray, R: np.ndarray, fps: float, scfg, hidden=None) -> list:
    """smoothing.smooth_track for one output identity; slots are smoothed in parts (see LineupConfig.smooth_break).
    Identical to smooth_track for every other track."""
    from replay.smoothing import smooth_track
    br = BREAKS.get(int(track))
    if br is None or not len(br):
        return smooth_track(k, z, R, fps, scfg, hidden=hidden)
    order = np.argsort(k, kind="stable")
    k, z, R = k[order], z[order], R[order]
    segs = []
    for idx in np.split(np.arange(len(k)), np.searchsorted(k, br)):
        if len(idx):
            segs += smooth_track(k[idx], z[idx], R[idx], fps, scfg, hidden=hidden)
    return segs


# ---- process pool for smooth_slot over all output tracks (identical: each track is smoothed by the same code from the
# same inputs in a worker; results come back in job order). Same switch as the roster pool (roster._use_pool: serial
# for short clips, REPLAY_PARALLEL=0/1, never nested; REPLAY_POOL_WORKERS caps the workers).
_SM: dict = {}
SMOOTH_PAR_MIN_FRAMES = 5000  # worker start-up (court_tracker/scipy import + context) ~ the serial time of a 3-min clip

def _sm_init(breaks, fps, scfg, hidden):
    BREAKS.clear()
    BREAKS.update(breaks)
    _SM.update(fps=fps, scfg=scfg, hidden=hidden)


def _sm_job(batch):
    return [smooth_slot(t, k, z, R, _SM["fps"], _SM["scfg"], hidden=_SM["hidden"]) for t, k, z, R in batch]


def smooth_slots(jobs: list, fps: float, scfg, hidden=None, n_frames: int = 0) -> list[list]:
    """[smooth_slot(t, k, z, R, fps, scfg, hidden) for (t, k, z, R) in jobs], pooled for long clips."""
    from replay.roster import _use_pool
    forced = os.environ.get("REPLAY_PARALLEL", "").strip().lower() in ("1", "true", "yes")
    workers = _use_pool(n_frames) if len(jobs) > 1 and (forced or n_frames >= SMOOTH_PAR_MIN_FRAMES) else 0
    if not workers:
        return [smooth_slot(t, k, z, R, fps, scfg, hidden=hidden) for t, k, z, R in jobs]
    from concurrent.futures import ProcessPoolExecutor
    # batches of about equal sample spans, in job order (the parent concatenates them back in the same order)
    w = np.array([int(k.max() - k.min()) + 1 if len(k) else 1 for _, k, _, _ in jobs], float)
    edges = np.searchsorted(np.cumsum(w), np.linspace(0, w.sum(), workers * 8 + 1)[1:-1], side="right")
    batches = [b for b in np.split(np.arange(len(jobs)), np.unique(edges)) if len(b)]
    out: list[list] = []
    with ProcessPoolExecutor(workers, initializer=_sm_init, initargs=(dict(BREAKS), fps, scfg, hidden)) as ex:
        for res in ex.map(_sm_job, [[jobs[j] for j in b] for b in batches]):
            out += res
    return out


def confident(p: _Piece, q: _Piece, solver: _Solver, cfg: LineupConfig) -> bool:
    """May q (next piece of a slot) keep p's output id?"""
    if p.number >= 0 and q.number >= 0:
        if p.number != q.number:
            return False
        if min(p.reads, q.reads) >= cfg.number_min_reads:
            return True  # well-read same number: same player (one wrong number link = one ID transfer)
    dt = float(solver.t[q.i0] - solver.t[p.i1])
    d = float(np.linalg.norm(p.end() - q.start()))
    if dt <= 0:
        return d <= cfg.reach_slack_m
    if d > cfg.max_speed_mps * dt + cfg.reach_slack_m:
        return False
    if p.tid == q.tid:
        return True
    if min(len(p.i), len(q.i)) < cfg.crumb_max_samples and dt <= cfg.crumb_gap_s:
        return True  # a short re-appearance (roster.short_but_continuing kept it) continues the slot's player
    sim = reid_sim(p, q) if cfg.use_reid else None
    if sim is not None and sim < cfg.reid_min_sim:
        return False  # looks like another person (e.g. a team-mate who crossed out of view)
    if sim is not None and sim >= cfg.reid_strong_sim and dt <= cfg.reid_long_gap_s:
        return True
    return dt <= cfg.confident_gap_s and d <= cfg.confident_speed_mps * dt + cfg.reach_slack_m


def _det_embeddings(out, det_row, n: int) -> np.ndarray | None:
    """Per-detection re-ID embeddings from the cached results/work/reid_embeddings.npz (NaN = none)."""
    if out is None or det_row is None or len(det_row) != n:
        return None
    try:
        from replay import reid
        e = reid.load_embeddings(Path(out) / "work")
    except Exception:  # noqa: BLE001
        return None
    if e is None or e.emb is None or len(e.emb) <= int(np.max(det_row, initial=0)):
        return None
    det_emb = np.asarray(e.emb, np.float32)[np.asarray(det_row, int)]
    print(f"lineup: re-ID embeddings for {int(np.isfinite(det_emb).all(axis=1).sum())} of {n} detections")
    return det_emb


def assign(tid, keep, det_i, det_xy, det_feat, det_num, t, teams, cfg: LineupConfig, rank=None,
           det_emb=None) -> LineupResult:
    tid = np.asarray(tid, int).copy()
    keep = np.asarray(keep, bool).copy()
    team_of = lambda g: teams.track_team.get(int(g), "?")  # noqa: E731
    numbers = None
    if det_num is not None:
        from replay import jersey_numbers
        sel = keep & (tid > 0)
        numbers = jersey_numbers.identity_numbers(tid[sel], np.asarray(det_num)[sel], det_i[sel], team_of,
                                                  team_lab=getattr(teams, "centroids", None) or None)
    if exact5_mode(cfg) in (2, 3):
        sel0 = keep & np.array([team_of(g) in TEAMS for g in tid])
        dead0, _ = dead_mask(np.asarray(t, float), det_i, det_xy, tid, sel0, cfg)
        keep = evict_stretches(tid, keep, det_i, det_xy, np.asarray(t, float), team_of, teams.track_share, rank,
                               dead0, cfg)
    pieces = build_pieces(tid, keep, det_i, det_xy, det_feat, team_of, teams.track_share, numbers,
                          np.asarray(t, float), cfg.gap_split_s, rank, cfg.rank_weight, det_emb)
    sel = keep & np.isin(tid, [p.tid for p in pieces])
    dead, windows = dead_mask(np.asarray(t, float), det_i, det_xy, tid, sel, cfg)
    slots, extras = solve(pieces, np.asarray(t, float), dead, cfg)
    base = int(tid.max(initial=0)) + 1
    res = LineupResult(tid, keep, slots, {}, extras, windows)
    new_tid = tid.copy()
    solver = _Solver(np.asarray(t, float), dead, cfg)
    next_id = base
    for label, ps in slots.items():
        for k, p in enumerate(ps):
            if k == 0 or not (not cfg.chain_ids or confident(ps[k - 1], p, solver, cfg)):
                res.chains.append((label, next_id, []))
                next_id += 1
            res.chains[-1][2].append(p)
            new_tid[keep & (tid == p.tid) & np.isin(det_i, p.i)] = res.chains[-1][1]
        if ps:
            res.slot_id[label] = next(c[1] for c in res.chains if c[0] == label)
    if cfg.drop_extras:
        for p in extras:
            keep[keep & (tid == p.tid) & np.isin(det_i, p.i)] = False
    elif exact5_mode(cfg) == 1 and extras:
        n = len(t)
        busy = {team: np.zeros(n, int) for team in TEAMS}
        for label, ps in slots.items():
            b = np.zeros(n, bool)
            for p in ps:  # busy = the slot's player is SEEN within +-exact5_hold samples (not just inside its span)
                for d in range(-cfg.exact5_hold, cfg.exact5_hold + 1):
                    j = p.i + d
                    b[j[(j >= 0) & (j < n)]] = True
            busy[label[0]] += b
        full = {team: (busy[team] >= cfg.slots_per_team) & ~dead for team in TEAMS}
        n_drop = 0
        for p in extras:
            cut = p.i[full[p.team][p.i]]
            if len(cut):
                m = keep & (tid == p.tid) & np.isin(det_i, cut)
                n_drop += int(m.sum())
                keep[m] = False
        print(f"lineup: exact 5 per team - {n_drop} extra detections at live samples with all 5 slots busy dropped")
    if exact5_mode(cfg) == 3 and UNSURE_DET and len(UNSURE_DET[0]) == len(tid):
        u = UNSURE_DET[0] & keep
        for g in np.unique(new_tid[u]):  # output id -> sample ranges where it is the unsure 6th person
            ii = np.sort(det_i[u & (new_tid == g)])
            runs = np.split(ii, np.flatnonzero(np.diff(ii) > 2 * cfg.exact5_hold + 1) + 1)
            UNSURE_RANGES[int(g)] = [(int(r[0]), int(r[-1])) for r in runs if len(r)]
    res.tid, res.keep = new_tid, keep
    return res


def predicted_rows(out_rows: list, new_id: dict, data, H: list, det_i: np.ndarray, det_xy: np.ndarray,
                   tid: np.ndarray, keep: np.ndarray, cfg: LineupConfig | None = None) -> list:
    """Pass-2 hook (step 6, after smoothing): rows for on-court slots that are out of view - a slot whose output
    rows stop (player left the picture / hidden for longer than the smoother bridges) keeps its last output id
    with a predicted position (court_tracker.MotionPrior: dead ball / half court / transition) while the
    prediction's sigma <= predict_max_sigma_m and until the slot is seen again. Rows are flagged interpolated,
    pos_state 5 (out of view), pos_sigma = prediction sigma. Nothing when the lineup is off.
    Exact-5 mode 3: first marks the rows of the unsure 6th person (UNSURE_RANGES) ``row.unsure = True``."""
    if UNSURE_RANGES:
        k_index = {int(k): i for i, k in enumerate(data.k)}
        rng = {new_id[g]: r for g, r in UNSURE_RANGES.items() if g in new_id}
        n_u = 0
        for r in out_rows:
            rr = rng.get(r.track_id)
            if rr:
                i = k_index.get(int(r.k))
                if i is not None and any(a <= i <= b for a, b in rr):
                    r.unsure = True
                    n_u += 1
        print(f"lineup: {n_u} rows marked unsure (6th person of a team, kept but not drawn)")
    if not SLOT_LABEL:
        return []
    cfg = cfg or LineupConfig()
    import time
    t_start = time.perf_counter()
    try:
        from replay.court_tracker import MotionPrior
        from replay.export import TrackRow
        from replay.smoothing import STATE_OFFSCREEN

        t = np.asarray(data.t, float)
        n = len(t)
        idx_of_k = {int(k): i for i, k in enumerate(data.k)}
        sel = keep & np.isin(tid, list(SLOT_LABEL))  # on-court slots only (enough for the phases; 2-4x faster)
        prior = MotionPrior.from_detections(t, det_i[sel], det_xy[sel], tid[sel])
        out_of = {new_id[sid]: lab for sid, lab in SLOT_LABEL.items() if sid in new_id}
        by_slot: dict[str, dict[int, object]] = {}
        for r in out_rows:
            lab = out_of.get(r.track_id)
            if lab is not None:
                by_slot.setdefault(lab, {})[idx_of_k[int(r.k)]] = r
        rows = []
        cap5 = exact5_on(cfg)
        count: dict[tuple[int, str], int] = {}
        if cap5:
            for r in out_rows:
                if r.team in TEAMS and not getattr(r, "unsure", False):
                    key = (idx_of_k[int(r.k)], r.team)
                    count[key] = count.get(key, 0) + 1
        for lab, at in by_slot.items():
            ii = np.array(sorted(at), int)
            nxt = np.r_[ii[1:], n]
            for a, b in zip(ii, nxt):
                if b - a <= 1:
                    continue
                r0 = at[int(a)]
                prev = at.get(int(a) - 1)
                v0 = np.zeros(2)
                if prev is not None and t[a] > t[a - 1]:
                    v0 = (np.array([r0.x, r0.y]) - np.array([prev.x, prev.y])) / (t[a] - t[a - 1])
                xy0 = np.array([r0.x, r0.y], float)
                # knots every PRED_STRIDE samples (MotionPrior.predict costs O(steps) per call), linear in between
                knots = {int(a): (xy0, 0.0)}
                last = int(a)
                prev_p, prev_t = xy0.copy(), float(t[a])
                for i in range(int(a) + 1, int(b)):
                    if i not in knots:
                        k_next = min(last + PRED_STRIDE, int(b) - 1)
                        knots[k_next] = prior.predict(int(a), xy0, v0, k_next)
                        (pa, sa), (pb, sb) = knots[last], knots[k_next]
                        for j in range(last + 1, k_next):
                            w = (j - last) / (k_next - last)
                            knots[j] = ((1 - w) * np.asarray(pa) + w * np.asarray(pb), (1 - w) * sa + w * sb)
                        last = k_next
                    p, sig = knots[i]
                    p = np.asarray(p, float)
                    step, reach = p - prev_p, PRED_MAX_SPEED_MPS * max(0.0, float(t[i]) - prev_t)
                    if float(np.hypot(*step)) > reach:  # speed limit along the predicted path
                        p = prev_p + step * (reach / float(np.hypot(*step)))
                    prev_p, prev_t = p, float(t[i])
                    if sig > cfg.predict_max_sigma_m:
                        break
                    if H[i] is None:
                        continue
                    if cap5:
                        if count.get((i, r0.team), 0) >= cfg.slots_per_team:
                            continue  # never a 6th row of the team
                        count[(i, r0.team)] = count.get((i, r0.team), 0) + 1
                    row = TrackRow(frame=int(data.frame[i]), t=float(t[i]), k=int(data.k[i]), track_id=r0.track_id,
                                   object_type=r0.object_type, team=r0.team, x=float(p[0]), y=float(p[1]),
                                   confidence=round(0.3 * max(0.0, 1.0 - sig / cfg.predict_max_sigma_m), 3),
                                   interpolated=True, calib=r0.calib, box=None)
                    row.pos_state, row.pos_sigma = STATE_OFFSCREEN, float(sig)
                    rows.append(row)
        print(f"lineup: {len(rows)} predicted rows for out-of-view slots ({time.perf_counter() - t_start:.2f} s)")
        return rows
    except Exception as e:  # noqa: BLE001
        print("lineup predictions failed:", type(e).__name__, e)
        return []


def slot_reads(res: LineupResult, det_i: np.ndarray, det_num: np.ndarray | None, t: np.ndarray) -> dict:
    """Jersey-number reads per slot over time {label: [[t, number], ...]} (detections of the slot's output ids with a
    read). Minutes (stats.lineup_stats) decide a slot's player per stretch between dead balls from these reads, so a
    tracker identity running through a substitution does not carry one number for the whole game."""
    if det_num is None or not len(det_num):
        return {}
    det_num = np.asarray(det_num, int)
    label_of = {sid: label for label, sid, _ps in res.chains}
    out: dict[str, list] = {}
    m = det_num >= 0
    for d in np.flatnonzero(m & res.keep & np.isin(res.tid, list(label_of))):
        out.setdefault(label_of[int(res.tid[d])], []).append([round(float(t[det_i[d]]), 1), int(det_num[d])])
    for v in out.values():
        v.sort()
    return out


def save(out: Path, res: LineupResult, t: np.ndarray, reads: dict | None = None) -> Path:
    doc = {"format": "advanced-analytist/lineup@1",
           "slots": {label: {"id": res.slot_id.get(label), "team": label[0],
                             "numbers": sorted({p.number for p in ps if p.number >= 0}),
                             "pieces": [[p.tid, round(float(t[p.i0]), 2), round(float(t[p.i1]), 2), p.number]
                                        for p in ps]}
                     for label, ps in res.slots.items()},
"chains": [[label, sid, [p.tid for p in ps]] for label, sid, ps in res.chains],
           "extra_tracks": [[p.tid, p.team, round(float(t[p.i0]), 2), round(float(t[p.i1]), 2)] for p in res.extras],
           "dead_windows": [[round(a, 2), round(b, 2), k] for a, b, k in res.dead],
           "reads": reads or {}}
    path = out / "lineup.json"
    path.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    return path
