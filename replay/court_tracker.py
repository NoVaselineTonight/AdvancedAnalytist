"""Second-stage tracking in court coordinates (metres).

Pass 2 calls ``track_people`` (bottom of this file). By default it runs the offline tracklet linker in
replay/tracklets.py (whole-video, court + camera-stabilised image cues, fewer identity switches); setting
``court_tracking.method = "online"`` selects the frame-by-frame tracker below.

ByteTrack (first stage, pass 1) works in image space. When a player walks behind
another, their box disappears; when they reappear the box is somewhere else in
the image, so ByteTrack starts a new ID. On the court, however, an occluded
player keeps a predictable position: they were standing *behind* someone, a
metre or two away, moving at running speed at most.

This tracker re-associates all detections frame by frame on the court:
  * each track has a constant-velocity Kalman filter in metres; while unseen,
    its uncertainty grows and its velocity decays (players stop and turn),
  * a detection may join a track if it lies within the track's uncertainty
    (Mahalanobis gate) AND within running reach of the last sighting,
  * costs combine position, jersey colour (same team = cheap) and the
    ByteTrack ID (same first-stage ID = strong hint, but never forced),
  * assignment per frame is optimal (Hungarian algorithm),
  * a track survives up to ``max_lost_s`` unseen, so a player hidden for a few
    seconds resumes with the same ID.

Positions are never invented here; gaps are handled later by the smoother
(short gaps interpolated and flagged, long gaps left empty).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from replay.config import CourtTrackConfig

_BIG = 1e6


@dataclass
class _Track:
    id: int
    x: np.ndarray  # [x, y, vx, vy]
    P: np.ndarray
    last_k: int
    feat: np.ndarray
    bt_id: int
    hits: int = 1
    k_pred: int = 0
    seen: list[int] = field(default_factory=list)


def track_on_court(k: np.ndarray, xy: np.ndarray, R: np.ndarray, feats: np.ndarray, bt_ids: np.ndarray,
                   conf: np.ndarray, fps: float, cfg: CourtTrackConfig) -> np.ndarray:
    """Returns a court-track id (>= 1) per detection, or 0 for detections that start no track."""
    order = np.argsort(k, kind="stable")
    out = np.zeros(len(k), int)
    tracks: list[_Track] = []
    next_id = 1
    max_lost = int(round(cfg.max_lost_s * fps))
    dt1 = 1.0 / fps
    q = cfg.accel_noise
    frames = np.unique(k)
    pos = {int(f): order[k[order] == f] for f in frames}

    for f in frames:
        f = int(f)
        idx = pos[f]
        # ---- predict
        for t in tracks:
            for step_k in range(t.k_pred + 1, f + 1):
                F = np.eye(4)
                F[0, 2] = F[1, 3] = dt1
                if step_k - t.last_k > 1:
                    t.x[2:] *= cfg.lost_velocity_damping  # unseen for a while: assume the player slows down
                t.x = F @ t.x
                g = np.array([[dt1 ** 2 / 2], [dt1]])
                Q1 = q ** 2 * (g @ g.T)
                Q = np.zeros((4, 4))
                Q[np.ix_([0, 2], [0, 2])] = Q1
                Q[np.ix_([1, 3], [1, 3])] = Q1
                t.P = F @ t.P @ F.T + Q
            t.k_pred = f
        # ---- costs
        n_t, n_d = len(tracks), len(idx)
        C = np.full((n_t, n_d), _BIG)
        for a, t in enumerate(tracks):
            lost_s = (f - t.last_k) / fps
            reach = cfg.max_speed_mps * lost_s + cfg.reach_slack_m
            for b, d in enumerate(idx):
                y = xy[d] - t.x[:2]
                dist = float(np.hypot(*y))
                if dist > reach:
                    continue
                S = t.P[:2, :2] + R[d]
                m2 = float(y @ np.linalg.solve(S, y))
                if m2 > cfg.gate_chi2:
                    continue
                app = 0.5
                if np.isfinite(feats[d]).all() and np.isfinite(t.feat).all():
                    app = float(np.linalg.norm(feats[d, :3] - t.feat[:3])) / 25.0
                c = 0.5 * m2 + cfg.appearance_weight * app
                if bt_ids[d] >= 0 and bt_ids[d] == t.bt_id:
                    c -= cfg.same_id_bonus
                C[a, b] = c + cfg.same_id_bonus  # keep costs positive
        matched_d: set[int] = set()
        if n_t and n_d:
            rows, cols = linear_sum_assignment(C)
            for a, b in zip(rows, cols):
                if C[a, b] >= _BIG:
                    continue
                t, d = tracks[a], idx[b]
                Hm = np.zeros((2, 4))
                Hm[0, 0] = Hm[1, 1] = 1
                S = Hm @ t.P @ Hm.T + R[d]
                K = t.P @ Hm.T @ np.linalg.inv(S)
                t.x = t.x + K @ (xy[d] - Hm @ t.x)
                t.P = (np.eye(4) - K @ Hm) @ t.P
                if np.isfinite(feats[d]).all():
                    t.feat = feats[d].copy() if not np.isfinite(t.feat).all() else 0.9 * t.feat + 0.1 * feats[d]
                if bt_ids[d] >= 0:
                    t.bt_id = int(bt_ids[d])
                t.last_k, t.hits = f, t.hits + 1
                out[d] = t.id
                matched_d.add(b)
        # ---- new tracks
        for b, d in enumerate(idx):
            if b in matched_d or conf[d] < cfg.new_track_min_conf:
                continue
            P = np.diag([R[d][0, 0], R[d][1, 1], 4.0, 4.0])
            tracks.append(_Track(next_id, np.array([xy[d, 0], xy[d, 1], 0.0, 0.0]), P, f,
                                 feats[d].copy(), int(bt_ids[d]), k_pred=f))
            out[d] = next_id
            next_id += 1
        # ---- retire
        tracks = [t for t in tracks if f - t.last_k <= max_lost]
    return out


def track_people(k: np.ndarray, xy: np.ndarray, R: np.ndarray, feats: np.ndarray, bt_ids: np.ndarray,
                 conf: np.ndarray, fps: float, cfg: CourtTrackConfig, boxes: np.ndarray | None = None,
                 motion: list | None = None, n_samples: int | None = None,
                 H: list | None = None, numbers: np.ndarray | None = None,
                 size: tuple[int, int] | None = None, cues=None, info: dict | None = None,
                 reattach: bool | None = None, ledger=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pass-2 step 3 entry point (see CourtTrackConfig.method).

    Returns (identity per detection, >= 1 or 0 = left out; court positions; covariances). The offline method
    re-estimates the foot of boxes cut short by an occluder, so positions may differ from the input.
    numbers: optional jersey number read per detection (jersey_numbers.py; offline method only).
    size: frame size (w, h); with it (and REATTACH_HIDDEN) identities that end and restart across a gap where the
    player was hidden behind someone / out of view are joined (reattach_hidden).
    ledger: optional corrections.IdentityConstraints (offline method only): the coach's hard identity constraints;
    a re-attach that would join two conflicting identities is undone.
    """
    if cfg.method == "online":
        if ledger:
            print("corrections: identity ops need the offline linker (court_tracking.method 'offline'): ignored")
        return track_on_court(k, xy, R, feats, bt_ids, conf, fps, cfg), xy, R
    from replay import tracklets
    from replay.tracklets import detection_confidence, link_tracks

    view = None  # camera footprint + occluders for the joint tracker (identity agent, joint_tracker.py)
    if tracklets.joint_enabled(cfg) and H is not None and boxes is not None and size is not None:
        view = OcclusionContext.from_boxes(H, k, boxes, size)
    res = link_tracks(k, xy, R, feats, bt_ids, conf, fps, cfg, boxes=boxes, motion=motion, n_samples=n_samples, H=H,
                      numbers=numbers, cues=cues, view=view, ledger=ledger)  # cues/info/view: identity agent
    ids = res.ids
    if info is not None:
        info["identities"], info["det_confidence"] = res.identities, detection_confidence(res.ids, res.identities)
        if res.labels:
            info["ledger_labels"] = res.labels  # later id joins (lineup) must respect them too
    if (reattach_enabled(None) if reattach is None else reattach) and H is not None and boxes is not None \
            and size is not None:
        ctx = OcclusionContext.from_boxes(H, k, boxes, size)
        before = ids
        ids, n_merged = reattach_hidden(ids, k, res.xy, feats, fps, ctx, numbers=numbers)
        if res.labels:
            ids, n_undone = _undo_ledger_conflicts(before, ids, res.labels)
            n_merged -= n_undone
        print(f"occlusion re-attach: {n_merged} identities joined across hidden gaps")
    return ids, res.xy, res.R


def _undo_ledger_conflicts(before: np.ndarray, after: np.ndarray, labels: dict) -> tuple[np.ndarray, int]:
    """A post-linking join (re-attach) must not break the corrections ledger: every joined identity whose
    detections carry conflicting ledger labels gets its original ids back. -> (ids, joins undone)."""
    from replay import corrections

    out = np.asarray(after).copy()
    undone = 0
    for u in np.unique(out[out > 0]):
        m = out == u
        orig = np.unique(before[m])
        if len(orig) < 2:
            continue
        if corrections.conflict_any([corrections.piece_labels(labels, np.flatnonzero(m & (before == o)))
                                     for o in orig]):
            out[m] = before[m]
            undone += len(orig) - 1
    return out, undone


# ====================================================================================== occlusion (image space)
VISIBLE, OCCLUDED, OFFSCREEN, UNKNOWN_VIS = 0, 1, 2, 3


class OcclusionContext:
    """Why is a player not detected at sample i? Answers from image-space cues only (works with imperfect
    calibration: the court position is only used to find the image spot).

    For a court position p at sample i the foot pixel u = H_i^-1 p is checked against
      * the frame (outside / under the bottom edge / under an overlay)       -> OFFSCREEN
      * every pass-1 person box whose box covers u (with some slack) and whose bottom is at or below u (the
        other person stands nearer to the camera, so they hide the player standing behind them) -> OCCLUDED
      * otherwise                                                           -> VISIBLE (a missed detection)
    Without calibration at i -> UNKNOWN_VIS.
    """

    def __init__(self, H: list, boxes_per_sample: list[np.ndarray], size: tuple[int, int],
                 overlay: np.ndarray | None = None, slack_rel: float = 0.15):
        self.n = len(H)
        self.Hinv = [None if h is None else np.linalg.inv(np.asarray(h, float)) for h in H]
        self.boxes = boxes_per_sample
        self.size = size
        self.overlay = np.zeros((0, 4)) if overlay is None else np.asarray(overlay, float).reshape(-1, 4)
        self.slack_rel = slack_rel

    @classmethod
    def from_boxes(cls, H: list, det_i: np.ndarray, boxes: np.ndarray, size: tuple[int, int]) -> "OcclusionContext":
        n = len(H)
        per: list[np.ndarray] = [np.zeros((0, 4)) for _ in range(n)]
        order = np.argsort(det_i, kind="stable")
        ii = np.asarray(det_i)[order]
        b = np.asarray(boxes, float).reshape(-1, 4)[order]
        for grp in np.split(np.arange(len(ii)), np.flatnonzero(np.diff(ii)) + 1):
            if len(grp) and 0 <= ii[grp[0]] < n:
                per[int(ii[grp[0]])] = b[grp]
        return cls(H, per, size)

    @classmethod
    def from_rows(cls, H: list, rows: np.ndarray, idx_of_k: dict[int, int], size: tuple[int, int],
                  overlay: np.ndarray | None = None, person_cls: int = 0) -> "OcclusionContext":
        per = rows[rows[:, 6] == person_cls]
        boxes: list[np.ndarray] = [np.zeros((0, 4)) for _ in range(len(H))]
        if len(per):
            ii = np.array([idx_of_k.get(int(k), -1) for k in per[:, 0]])
            order = np.argsort(ii, kind="stable")
            ii, b = ii[order], per[order, 1:5].astype(float)
            cuts = np.flatnonzero(np.diff(ii)) + 1
            for grp in np.split(np.arange(len(ii)), cuts):
                if len(grp) and ii[grp[0]] >= 0:
                    boxes[int(ii[grp[0]])] = b[grp]
        return cls(H, boxes, size, overlay)

    def foot_pixel(self, i: int, p: np.ndarray) -> np.ndarray | None:
        h = self.Hinv[i] if 0 <= i < self.n else None
        if h is None:
            return None
        q = h @ np.array([p[0], p[1], 1.0])
        if abs(q[2]) < 1e-12:
            return None
        return q[:2] / q[2]

    def classify(self, i: int, p: np.ndarray) -> tuple[int, int]:
        """(visibility code, index of the occluding box in boxes[i] or -1)."""
        u = self.foot_pixel(int(i), p)
        if u is None or not np.isfinite(u).all():
            return UNKNOWN_VIS, -1
        w, hgt = self.size
        if u[0] < 0 or u[0] > w or u[1] < 0 or u[1] > hgt - 3:
            return OFFSCREEN, -1
        for ox1, oy1, ox2, oy2 in self.overlay:
            if ox1 <= u[0] <= ox2 and oy1 - 4 <= u[1] <= oy2:
                return OFFSCREEN, -1
        b = self.boxes[int(i)]
        if len(b):
            bh = b[:, 3] - b[:, 1]
            sx = self.slack_rel * 0.5 * bh  # widths ~ half the height
            cover = ((b[:, 0] - sx <= u[0]) & (u[0] <= b[:, 2] + sx) & (b[:, 1] <= u[1])
                     & (b[:, 3] >= u[1] - self.slack_rel * bh))
            if cover.any():
                j = int(np.flatnonzero(cover)[np.argmax(b[cover, 3])])
                return OCCLUDED, j
        return VISIBLE, -1

    def behind_point(self, i: int, p: np.ndarray, max_m: float = 2.0) -> np.ndarray | None:
        """Nearest court point within max_m of p whose foot pixel is hidden behind a person box at sample i (foot
        between 40 % of the box height and its bottom, inside the box horizontally), or None."""
        u = self.foot_pixel(int(i), p)
        h = self.Hinv[int(i)] if 0 <= int(i) < self.n else None
        b = self.boxes[int(i)] if 0 <= int(i) < self.n else np.zeros((0, 4))
        if u is None or h is None or not len(b) or not np.isfinite(u).all():
            return None
        Hf = np.linalg.inv(h)
        best, best_d = None, max_m
        for x1, y1, x2, y2 in b:
            bh = y2 - y1
            tgt = np.array([np.clip(u[0], x1 + 0.2 * (x2 - x1), x2 - 0.2 * (x2 - x1)),
                            np.clip(u[1], y1 + 0.4 * bh, y2 - 0.02 * bh)])
            q = Hf @ np.array([tgt[0], tgt[1], 1.0])
            if abs(q[2]) < 1e-12:
                continue
            c = q[:2] / q[2]
            d = float(np.hypot(*(c - p)))
            if np.isfinite(d) and d < best_d:
                best, best_d = c, d
        return best

    def classify_many(self, ks: np.ndarray, xy: np.ndarray) -> np.ndarray:
        return np.array([self.classify(int(i), p)[0] for i, p in zip(ks, xy)], np.int8)

    def behind_limit(self, i: int, j: int) -> tuple[float, float, float] | None:
        """Image-space constraint while hidden behind box j at sample i: (x1, x2, y2) - the hidden player's foot
        pixel must lie inside [x1, x2] horizontally and at or above y2 (further from the camera)."""
        b = self.boxes[int(i)][j]
        return float(b[0]), float(b[2]), float(b[3])


# ====================================================================================== game-aware motion prior
DEAD, HALF_COURT, TRANSITION = 0, 1, 2
PHASE_NAMES = {DEAD: "dead", HALF_COURT: "half_court", TRANSITION: "transition"}


@dataclass
class MotionPrior:
    """Per-sample game phase from the players' collective motion (a cheap rule; optionally overridden by the
    dead-ball windows of game_state.json):
      DEAD        everybody (median player) slower than ``dead_speed`` for at least ``dead_min_s`` - free throws,
                  fouls, inbounds, time-outs: hidden players are assumed to stand still;
      TRANSITION  the players' mean velocity along the court is above ``transition_vx`` - fast break / getting
                  back: hidden players keep running towards the same end (``flow`` = +1 to the right / -1 left);
      HALF_COURT  otherwise: hidden players slow down and stay in the half where the play is (``play_x``).
    """

    t: np.ndarray  # sample times (s)
    phase: np.ndarray  # (n,) int8
    flow: np.ndarray  # (n,) mean player velocity along x, m/s (smoothed)
    speed: np.ndarray  # (n,) median player speed, m/s (smoothed)
    play_x: np.ndarray  # (n,) median x of the seen players (nan when nobody seen)

    @classmethod
    def from_detections(cls, t: np.ndarray, det_i: np.ndarray, det_xy: np.ndarray, tid: np.ndarray,
                        dead_speed: float = 0.7, dead_min_s: float = 2.0, transition_vx: float = 2.2,
                        window_s: float = 1.0, dead_windows: list[tuple[float, float]] | None = None
                        ) -> "MotionPrior":
        n = len(t)
        fps = 1.0 / max(1e-6, float(np.median(np.diff(t)))) if n > 1 else 10.0
        lag = max(1, int(round(0.5 * fps)))
        vx_s = [[] for _ in range(n)]
        sp_s = [[] for _ in range(n)]
        xs = [[] for _ in range(n)]
        for t_id in np.unique(tid[tid > 0]):
            m = np.flatnonzero(tid == t_id)
            ii = det_i[m]
            o = np.argsort(ii, kind="stable")
            ii, xy = ii[o], det_xy[m][o]
            ii, u = np.unique(ii, return_index=True)
            xy = xy[u]
            pos = {int(a): b for a, b in zip(ii, xy)}
            for a, p in pos.items():
                xs[a].append(p[0])
                q = pos.get(a - lag)
                if q is not None:
                    dt = t[a] - t[a - lag]
                    if dt > 0:
                        v = (p - q) / dt
                        vx_s[a].append(v[0])
                        sp_s[a].append(float(np.hypot(*v)))
        nanmed = lambda lst: float(np.median(lst)) if len(lst) >= 3 else np.nan  # noqa: E731
        speed = np.array([nanmed(s) for s in sp_s])
        flow = np.array([float(np.mean(v)) if len(v) >= 3 else np.nan for v in vx_s])
        play_x = np.array([nanmed(x) for x in xs])
        w = max(1, int(round(window_s * fps)))
        speed, flow, play_x = (_nan_smooth(a, w) for a in (speed, flow, play_x))
        phase = np.full(n, HALF_COURT, np.int8)
        slow = np.nan_to_num(speed, nan=np.inf) < dead_speed
        dmin = int(round(dead_min_s * fps))
        i = 0
        while i < n:
            if slow[i]:
                j = i
                while j < n and slow[j]:
                    j += 1
                if j - i >= dmin:
                    phase[i:j] = DEAD
                i = j
            else:
                i += 1
        if dead_windows:
            for a, b in dead_windows:
                phase[(t >= a) & (t <= b)] = DEAD
        fast = (np.abs(np.nan_to_num(flow)) > transition_vx) & (phase != DEAD)
        phase[fast] = TRANSITION
        return cls(np.asarray(t, float), phase, flow, speed, play_x)

    @classmethod
    def from_rows(cls, H: list, rows: np.ndarray, idx_of_k: dict[int, int], t: np.ndarray, person_cls: int = 0,
                  **kw) -> "MotionPrior":
        """Prior straight from pass-1 rows: person foot points projected with H, inside the court lines only
        (bench and spectators out), ByteTrack ids as tracks. Usable before court tracking has run."""
        import replay.court as court

        per = rows[(rows[:, 6] == person_cls) & (rows[:, 7] >= 0)]
        ii = np.array([idx_of_k.get(int(k), -1) for k in per[:, 0]], int)
        ok = np.array([0 <= i < len(H) and H[i] is not None for i in ii], bool)
        per, ii = per[ok], ii[ok]
        if not len(per):
            return cls.from_detections(t, np.zeros(0, int), np.zeros((0, 2)), np.zeros(0, int), **kw)
        Hs = np.array([H[i] for i in ii], float)
        foot = np.c_[(per[:, 1] + per[:, 3]) / 2, per[:, 4], np.ones(len(per))]
        q = np.einsum("nij,nj->ni", Hs, foot)
        with np.errstate(divide="ignore", invalid="ignore"):
            xy = q[:, :2] / q[:, 2:3]
        inside = np.isfinite(xy).all(axis=1) & (q[:, 2] > 0)
        inside &= (xy[:, 0] >= 0) & (xy[:, 0] <= court.LENGTH) & (xy[:, 1] >= 0) & (xy[:, 1] <= court.WIDTH)
        return cls.from_detections(t, ii[inside], xy[inside], per[inside, 7].astype(int) + 1, **kw)

    def sigma_scale(self, i: np.ndarray) -> np.ndarray:
        """Multiplier of the gap spread (pos_sigma) per sample: hidden players barely move in dead balls."""
        ph = self.phase[np.clip(np.asarray(i, int), 0, len(self.phase) - 1)]
        return np.where(ph == DEAD, 0.4, np.where(ph == TRANSITION, 1.3, 1.0))

    def predict(self, i0: int, xy0: np.ndarray, v0: np.ndarray, i1: int, base_sigma: float = 0.3,
                tau_s: float = 1.0, run_speed: float = 5.0) -> tuple[np.ndarray, float]:
        """Open-ended prediction for a player last seen at sample i0 (position xy0, velocity v0 m/s) at a later
        sample i1, following the phases in between. Returns (court xy, 1-sigma metres). For hidden lineup
        slots (lineup.py) and anything that must place a player who has not been seen for a while."""
        import replay.court as court

        p = np.asarray(xy0, float).copy()
        v = np.asarray(v0, float).copy()
        var = base_sigma ** 2
        for i in range(int(i0) + 1, int(i1) + 1):
            dt = float(self.t[i] - self.t[i - 1]) if i < len(self.t) else 0.1
            ph = self.phase[min(i, len(self.phase) - 1)]
            if ph == DEAD:
                v *= np.exp(-dt / 1.0)  # walking to positions, then standing
                var += 0.7 ** 2 * dt
            elif ph == TRANSITION:
                fl = float(self.flow[min(i, len(self.flow) - 1)])
                target = np.array([np.sign(fl) * max(run_speed, abs(fl)), 0.0])
                a = 1 - np.exp(-dt / 0.7)  # players align with the flow within ~1 s
                v = (1 - a) * v + a * target
                var += 2.5 ** 2 * dt
            else:
                v *= np.exp(-dt / tau_s)
                px = self.play_x[min(i, len(self.play_x) - 1)]
                if np.isfinite(px):  # stay in the half where the play is
                    mid = court.LENGTH / 2
                    if (px - mid) * (p[0] - mid) < 0 and abs(p[0] - mid) > 1.0:
                        v[0] += np.sign(px - mid) * 3.0 * dt  # drift back over the half line
                var += 1.6 ** 2 * dt
            p = p + v * dt
            p[0] = float(np.clip(p[0], 0.0, court.LENGTH))
            p[1] = float(np.clip(p[1], 0.0, court.WIDTH))
        return p, float(np.sqrt(var))


def _nan_smooth(a: np.ndarray, w: int) -> np.ndarray:
    """Centred moving average over a window of w samples, ignoring nan (all-nan windows stay nan)."""
    ok = np.isfinite(a)
    v = np.where(ok, a, 0.0)
    ker = np.ones(2 * (w // 2) + 1)
    num = np.convolve(v, ker, "same")
    den = np.convolve(ok.astype(float), ker, "same")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / np.maximum(den, 1e-9), np.nan)



# ====================================================================================== hidden-gap re-attach
REATTACH_HIDDEN = os.environ.get("REPLAY_REATTACH", "0") == "1"  # env override (see reattach_enabled)


def reattach_enabled(scfg) -> bool:
    """reattach_hidden on? env REPLAY_REATTACH (0/1) first, then SmoothingConfig.reattach_hidden (``scfg`` may be
    None = env only)."""
    env = os.environ.get("REPLAY_REATTACH")
    if env in ("0", "1"):
        return env == "1"
    return bool(getattr(scfg, "reattach_hidden", False))
REATTACH_MAX_GAP_S = 6.0
REATTACH_MIN_GAP_S = 0.2
REATTACH_SPEED_MPS = 6.0
REATTACH_SLACK_M = 1.5
REATTACH_HIDDEN_SHARE = 0.7
REATTACH_MAX_LAB = 14.0  # torso + shorts a*/b* distance of the identities' medians
REATTACH_MAX_COST = 3.0


def reattach_hidden(ids: np.ndarray, det_i: np.ndarray, xy: np.ndarray, feats: np.ndarray, fps: float,
                    ctx: OcclusionContext, numbers: np.ndarray | None = None) -> tuple[np.ndarray, int]:
    """Join identity A (ends) to identity B (starts later) when the player was plausibly hidden in between:
    reachable at running speed, same kit colours (median over each identity), no conflicting jersey number, and
    along the straight line between the two sightings the spot is behind a nearer person or out of view for at
    least REATTACH_HIDDEN_SHARE of the samples. One-to-one optimal assignment of ends to starts (chains form).
    Returns (new ids, number of joins)."""
    ids = np.asarray(ids).copy()
    uniq = [int(u) for u in np.unique(ids[ids > 0])]
    if len(uniq) < 2:
        return ids, 0
    info = {}
    for u in uniq:
        m = np.flatnonzero(ids == u)
        o = m[np.argsort(det_i[m], kind="stable")]
        ii = det_i[o]
        f = feats[o]
        fk = np.nanmedian(f[:, [1, 2, 4, 5]], axis=0) if np.isfinite(f[:, [1, 2, 4, 5]]).any() else np.full(4, np.nan)
        num = set()
        if numbers is not None:
            num = {int(c) for c in numbers[o] if c >= 0}
        info[u] = dict(s=int(ii[0]), e=int(ii[-1]), ps=np.median(xy[o[:3]], axis=0), pe=np.median(xy[o[-3:]], axis=0),
                       lab=fk, num=num, n=len(o))
    ends = sorted(uniq, key=lambda u: info[u]["e"])
    starts = sorted(uniq, key=lambda u: info[u]["s"])
    max_gap = int(round(REATTACH_MAX_GAP_S * fps))
    min_gap = max(1, int(round(REATTACH_MIN_GAP_S * fps)))
    cand = []
    for a in ends:
        A = info[a]
        for b in starts:
            if a == b:
                continue
            B = info[b]
            gap = B["s"] - A["e"]
            if gap < min_gap or gap > max_gap:
                continue
            dt = gap / fps
            d = float(np.hypot(*(B["ps"] - A["pe"])))
            if d > REATTACH_SPEED_MPS * dt + REATTACH_SLACK_M:
                continue
            if A["num"] and B["num"] and not (A["num"] & B["num"]):
                continue
            dl = float(np.linalg.norm(A["lab"] - B["lab"])) if np.isfinite(A["lab"]).all() and np.isfinite(B["lab"]).all() else 7.0
            if dl > REATTACH_MAX_LAB:
                continue
            ks = np.arange(A["e"] + 1, B["s"])
            fr = ((ks - A["e"]) / gap)[:, None]
            codes = ctx.classify_many(ks, A["pe"] + fr * (B["ps"] - A["pe"]))
            known = codes != UNKNOWN_VIS
            if not known.any() or float(np.mean(codes[known] != VISIBLE)) < REATTACH_HIDDEN_SHARE:
                continue
            cost = d / (REATTACH_SPEED_MPS * dt + REATTACH_SLACK_M) + dl / REATTACH_MAX_LAB + dt / REATTACH_MAX_GAP_S
            if A["num"] and B["num"]:
                cost -= 0.5
            if cost <= REATTACH_MAX_COST:
                cand.append((cost, a, b))
    if not cand:
        return ids, 0
    ai = {a: j for j, a in enumerate(sorted({c[1] for c in cand}))}
    bi = {b: j for j, b in enumerate(sorted({c[2] for c in cand}))}
    C = np.full((len(ai), len(bi)), _BIG)
    for cost, a, b in cand:
        C[ai[a], bi[b]] = cost
    rows, cols = linear_sum_assignment(C)
    inv_b = {j: b for b, j in bi.items()}
    inv_a = {j: a for a, j in ai.items()}
    nxt = {inv_a[r]: inv_b[c] for r, c in zip(rows, cols) if C[r, c] < _BIG}
    # follow chains a -> b -> c ...; every chain takes the id of its first member
    has_prev = set(nxt.values())
    joined = 0
    for a in list(nxt):
        if a in has_prev:
            continue
        cur, seen = a, {a}
        while cur in nxt and nxt[cur] not in seen:
            cur = nxt[cur]
            seen.add(cur)
            ids[ids == cur] = a
            joined += 1
    return ids, joined
