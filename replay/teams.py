"""Team classification from jersey colours, per court track.

What goes wrong with simple colour rules on real footage (measured on the KoriTV games): a box is mostly arms,
skin, floor and background; the torso *median* of a side-on player in a dark purple kit is skin/floor-coloured,
so purple and orange players get nearly the same value. Nominal kit colours (meta) are often stale placeholders.

So, fully automatic:

1. Per detection: a colour histogram of shirt + shorts (``team_appearance``: hue sector x chroma level for
   coloured pixels, lightness bins for grey ones), re-sampled from the video in pass 2 and cached. Without the
   video the pass-1 torso/shorts medians are turned into (weaker) pseudo-histograms.
2. Per track: occlusion-weighted mean histogram over all its sampled detections (boxes whose shirt is covered by
   someone standing in front are left out).
3. Weighted k-means (Hellinger distance, weight = how long the track is seen) into a few clusters: two teams,
   referees, and "others" (bench, coaches, spectators that slipped through the court filter).
4. Cluster roles: team 1 = heavy and colourful cluster; team 2 = the heaviest colourful cluster of a clearly
   different hue (or the most different one when a team wears white/black); referee = mostly grey/black cluster.
   Tracks in "other" clusters join a team/referees only if they lie within that cluster's own spread.
5. Which team is home (A) / away (B): user-picked reference players (meta "team_samples") seed the clusters and
   decide it; otherwise user-set kit colours decide it *if they match* the detected colours; stale/placeholder
   colours are ignored. Display colours come from the actual jersey pixels.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import cv2
import numpy as np

from replay import team_appearance as ta
from replay.config import TeamConfig
from replay.features import rgb_to_lab
from replay.metadata import GameMeta, rgb_to_hex

A, B, REF, UNKNOWN = "A", "B", "referee", "unknown"
# placeholders written by the GUI / metadata defaults - never real kit colours
OLD_DEFAULT_COLORS = {"#f5f5f5", "#1f2a5a", "#141e50"}


@dataclass
class TeamResult:
    track_team: dict[int, str]
    track_share: dict[int, float]
    centroids: dict[str, list[float]]  # representative jersey Lab per label
    seeded: bool  # True = A/B order and display colours come from the user (picked players or kit colours)
    colors: dict[str, str]  # display colour per label (#rrggbb)
    notes: list[str] = field(default_factory=list)
    # colour signature per team label for the stable team binding (replay/team_binding.py): the team cluster's
    # mean descriptor histogram, in descriptor space ``space`` ("video<v>" or "pass1<v>")
    hists: dict[str, list[float]] = field(default_factory=dict)
    space: str = ""
    # per input detection: the label its own colour histogram points to (nearest cluster's role; "unknown" for
    # bench/other clusters, "" = no histogram). Used by :func:`split_mixed_tracks`.
    det_label: np.ndarray | None = None
    # k-means centres (sqrt-histogram space ``space``) and the role of each ("unknown" = bench/other cluster)
    cluster_centres: np.ndarray | None = None
    cluster_roles: list[str] = field(default_factory=list)
    track_cluster: dict[int, int] = field(default_factory=dict)  # k-means cluster of each track (diagnostics)


# ----------------------------------------------------------------------------- colour helpers
def lab_to_hex(lab: np.ndarray, vivid: bool = True) -> str:
    L, a, b = float(lab[0]), float(lab[1]), float(lab[2])
    px = np.array([[[L * 255 / 100, a + 128, b + 128]]], np.float32).clip(0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(px, cv2.COLOR_LAB2BGR)[0, 0]
    if vivid:  # video colours are dull; make the replay dots recognisable
        hsv = cv2.cvtColor(bgr[None, None, :], cv2.COLOR_BGR2HSV)[0, 0].astype(int)
        if hsv[1] > 25:  # leave greys grey
            hsv[1] = min(255, int(hsv[1] * 1.6) + 40)
            hsv[2] = max(hsv[2], 210)
        bgr = cv2.cvtColor(np.uint8(hsv)[None, None, :], cv2.COLOR_HSV2BGR)[0, 0]
    return rgb_to_hex((int(bgr[2]), int(bgr[1]), int(bgr[0])))


def colors_are_user_set(meta: GameMeta) -> bool:
    """True when the meta holds kit colours the user chose (not GUI placeholders)."""
    raw = meta.raw
    jc = raw.get("jersey_colors") or {}
    if not jc or raw.get("team_colors_auto") is True:
        return False
    prim = {str(jc.get(t, {}).get("primary", "")).lower() for t in ("home", "away")}
    return not (prim & OLD_DEFAULT_COLORS) and "" not in prim


def _hue(a, b):
    return np.degrees(np.arctan2(b, a))


def _angdiff(h1, h2):
    return np.abs((np.asarray(h1) - np.asarray(h2) + 180.0) % 360.0 - 180.0)


def _color_cost(lab1, lab2) -> float:
    """0..180: hue difference for coloured pairs; lightness difference for grey pairs; 180 for a mix."""
    c1, c2 = np.hypot(lab1[1], lab1[2]), np.hypot(lab2[1], lab2[2])
    col1, col2 = c1 >= 10, c2 >= 8
    if col1 and col2:
        return float(_angdiff(_hue(lab1[1], lab1[2]), _hue(lab2[1], lab2[2])))
    if not col1 and not col2:
        return float(abs(lab1[0] - lab2[0]) * 1.8)
    return 180.0


# ----------------------------------------------------------------------------- clustering
def _kmeans(X: np.ndarray, w: np.ndarray, k: int, rng: np.random.Generator, seeds: np.ndarray | None = None,
            n_init: int = 8, iters: int = 60) -> tuple[np.ndarray, np.ndarray]:
    """Weighted k-means; the first len(seeds) centres start at (and are anchored to) the given seeds."""
    n = len(X)
    k = max(1, min(k, n))
    ns = 0 if seeds is None else len(seeds)
    anchor = 0.25 * w.sum() / max(k, 1)  # pseudo-weight that keeps seeded centres near the user's samples
    best = (np.inf, None, None)
    for _ in range(n_init):
        C = np.zeros((k, X.shape[1]))
        if ns:
            C[:ns] = seeds
        for j in range(ns, k):  # k-means++ for the free centres
            if j == 0:
                C[0] = X[rng.choice(n, p=w / w.sum())]
                continue
            d = ((X[:, None, :] - C[None, :j]) ** 2).sum(-1).min(axis=1) * w
            C[j] = X[rng.choice(n, p=d / d.sum())] if d.sum() > 0 else X[rng.integers(n)]
        lab = np.zeros(n, int)
        for it in range(iters):
            d = ((X[:, None, :] - C[None]) ** 2).sum(-1)
            new = d.argmin(axis=1)
            if it and np.array_equal(new, lab):
                break
            lab = new
            for j in range(k):
                m = lab == j
                wj = w[m].sum()
                if j < ns:
                    C[j] = ((X[m] * w[m, None]).sum(0) + seeds[j] * anchor) / (wj + anchor)
                elif wj > 0:
                    C[j] = (X[m] * w[m, None]).sum(0) / wj
        inertia = float((d[np.arange(n), lab] * w).sum())
        if inertia < best[0]:
            best = (inertia, lab.copy(), C.copy())
        if ns >= k:
            break
    return best[1], best[2]


@dataclass
class _Cluster:
    mu: np.ndarray  # weighted mean histogram
    weight: float  # observations
    colourful: float  # share of clearly coloured pixels (chroma >= 14)
    hue_vec: np.ndarray  # (N_HUE,) coloured mass per hue sector (circularly smoothed)
    ach: float  # achromatic share
    black: float
    grey: float
    distinct: float  # coloured mass (chroma >= 14) above what everybody has (skin, floor tint)
    distinct_hue: np.ndarray  # (N_HUE,) hue profile of that excess


def _smooth_hue(hv: np.ndarray) -> np.ndarray:
    return 0.5 * hv + 0.25 * np.roll(hv, 1) + 0.25 * np.roll(hv, -1)


def _cluster_stats(D: np.ndarray, w: np.ndarray, common: np.ndarray | None = None) -> _Cluster:
    mu = (D * w[:, None]).sum(0) / max(w.sum(), 1e-9)
    hm = ta.hue_mass(mu)
    hv = _smooth_hue(hm[:, 1:].sum(axis=1))
    ach = mu[ta.N_HUE * ta.N_CHROMA:]
    ex = ta.hue_mass(np.clip(mu - (common if common is not None else 0.0), 0, None))[:, 1:].sum(axis=1)
    return _Cluster(mu, float(w.sum()), float(hm[:, 1:].sum()), hv, float(ach.sum()), float(ach[0]),
                    float(ach[1] + ach[2]), float(ex.sum()), _smooth_hue(ex))


def _weighted_median(D: np.ndarray, w: np.ndarray) -> np.ndarray:
    out = np.zeros(D.shape[1])
    cw = None
    for j in range(D.shape[1]):
        o = np.argsort(D[:, j])
        cw = np.cumsum(w[o])
        out[j] = D[o[np.searchsorted(cw, 0.5 * cw[-1])], j]
    return out


ZEBRA_REF = 0.25  # median zebra_score of a track's large boxes above this = referee (team_appearance.zebra_score)
ZEBRA_MIN_SAMPLES = 3
REF_OUT_M = 1.0  # broadcast referee: median position at most this far outside a sideline / baseline
REF_STILL_SAMPLES = 200  # ... and not standing still (p90 distance from its median < REF_STILL_M) for this long
REF_STILL_M = 1.0
def _clip_probe_referees(track_ids, T, zb, track_team, clip) -> dict | None:
    """Broadcast: train a small linear probe on this game's CLIP crop embeddings (referee = tracks whose best large
    boxes show stripes, team A / B = colour-labelled tracks without stripes), then call every track by its mean
    probe probability. {track: is_referee} or None without embeddings / too few labels."""
    if clip is None:
        return None
    E = np.asarray(clip, np.float32)
    ok_row = np.isfinite(E).all(axis=1)
    X, y, rows_of = [], [], {}
    for t in T:
        m = (track_ids == t) & ok_row
        rows_of[int(t)] = np.flatnonzero(m)
        v = zb[track_ids == t]
        v = v[np.isfinite(v)]
        top = float(np.sort(v)[-3:].mean()) if len(v) >= 3 else np.nan
        lab = track_team.get(int(t))
        if top >= PROBE_REF_ZEBRA:
            c = 0
        elif lab in (A, B) and not (top >= PROBE_PLAYER_MAX_ZEBRA):
            c = 1 if lab == A else 2
        else:
            continue
        X.append(E[rows_of[int(t)]])
        y += [c] * len(rows_of[int(t)])
    if not X:
        return None
    X, y = np.concatenate(X), np.array(y)
    cnt = np.bincount(y, minlength=3)
    if cnt[0] < PROBE_MIN_CROPS or min(cnt[1], cnt[2]) < PROBE_MIN_CROPS:
        return None
    W, b = np.zeros((X.shape[1], 3)), np.zeros(3)
    cw = len(y) / (3 * cnt)
    for _ in range(300):  # class-balanced multinomial logistic regression (plain gradient descent, L2)
        Z = X @ W + b
        P = np.exp(Z - Z.max(axis=1, keepdims=True))
        P /= P.sum(axis=1, keepdims=True)
        G = P
        G[np.arange(len(y)), y] -= 1
        G *= cw[y][:, None]
        W -= 0.5 * (X.T @ G / len(y) + 1e-3 * W)
        b -= 0.5 * G.mean(axis=0)
    out = {}
    for t, rr in rows_of.items():
        if len(rr) >= 2:
            Z = E[rr] @ W + b
            P = np.exp(Z - Z.max(axis=1, keepdims=True))
            P /= P.sum(axis=1, keepdims=True)
            out[t] = bool(P.mean(axis=0).argmax() == 0)
    return out


PROBE_REF_ZEBRA = 0.3  # auto-label: mean of a track's 3 best zebra scores
PROBE_PLAYER_MAX_ZEBRA = 0.05
PROBE_MIN_CROPS = 20
SEED_CHECK = True  # see classify: seeded team clusters must be on court
SEED_MIN_TEAM_PRESENCE = 1.5  # people per frame (a team has ~5 on court; < 1.5 = the sample matched no team kit)


def _cos(u, v) -> float:
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    return float(u @ v / (nu * nv)) if nu > 0 and nv > 0 else 0.0


def _assign_roles(cl: list[_Cluster], cfg: TeamConfig) -> tuple[int | None, int | None, int | None]:
    """Indices of the two team clusters and the referee cluster."""
    if not cl:
        return None, None, None
    # colour that sets a cluster apart (skin and floor tint, shared by everyone, do not count)
    team_score = [c.weight * (c.distinct + 0.02) for c in cl]
    t1 = int(np.argmax(team_score))
    cand = [j for j in range(len(cl)) if j != t1 and _cos(cl[j].distinct_hue, cl[t1].distinct_hue) < cfg.same_hue_cos]
    t2 = None
    colourful = [j for j in cand if cl[j].distinct >= cfg.min_team_colourful]
    if colourful:
        t2 = max(colourful, key=lambda j: team_score[j])
    elif cand:  # a team in white/black/grey: the heaviest cluster that differs most from team 1
        def sep(j):
            return float(np.linalg.norm(np.sqrt(cl[j].mu) - np.sqrt(cl[t1].mu)))
        t2 = max(cand, key=lambda j: cl[j].weight * sep(j))
    return t1, t2, _referee_cluster(cl, {t1, t2}, cfg)


def _presence(track_ids: np.ndarray, frames: np.ndarray, xy: np.ndarray | None, T: np.ndarray, lab_of: np.ndarray,
              k: int, margin: float = 1.0) -> np.ndarray:
    """Mean number of each cluster's people per frame (inside the court + margin when positions are known): two
    teams have ~5 people on court at a time, referees 2-3, bench / spectators few that get through the filters."""
    track_ids = np.asarray(track_ids)
    frames = np.asarray(frames)
    cl_of = dict(zip(T.tolist(), np.asarray(lab_of).tolist()))
    cl = np.array([cl_of.get(int(t), -1) for t in track_ids])
    ok = cl >= 0
    if xy is not None and len(xy) == len(track_ids):
        from replay import court
        xy = np.asarray(xy, float)
        ok &= ((xy[:, 0] >= -margin) & (xy[:, 0] <= court.LENGTH + margin) & (xy[:, 1] >= -margin)
               & (xy[:, 1] <= court.WIDTH + margin))
    nfr = max(1, len(np.unique(frames)))
    return np.bincount(cl[ok], minlength=k)[:k] / nfr


def _hel(a: _Cluster, b: _Cluster) -> float:
    p, q = np.clip(a.mu, 0, None), np.clip(b.mu, 0, None)
    return float(np.sqrt(max(0.0, 1.0 - np.sqrt(p * q).sum() / np.sqrt(max(p.sum() * q.sum(), 1e-12)))))


def _different_kit(a: _Cluster, b: _Cluster, cfg: TeamConfig, referee: bool = False) -> bool:
    """Another kit? Two coloured kits: different hue. Otherwise: histograms clearly apart. ``referee``: a kit of the
    same hue as a team still differs when it has clearly more/less black (referee shirt with black trousers)."""
    if a.distinct >= cfg.min_team_colourful and b.distinct >= cfg.min_team_colourful:
        if _cos(a.distinct_hue, b.distinct_hue) < cfg.same_hue_cos:
            return True
        return referee and _hel(a, b) >= cfg.team_min_sep and abs(a.black - b.black) >= 0.1
    return _hel(a, b) >= cfg.team_min_sep


def _assign_roles_presence(cl: list[_Cluster], pres: np.ndarray, cfg: TeamConfig, spread: np.ndarray | None = None
                           ) -> tuple[int | None, int | None, int | None]:
    """Roles by who is on court: the two kits with the most people on court at a time are the teams (whatever their
    colour - black, white or coloured), the referee kit is the next most present distinct kit with at most about
    three people at a time (any colour; grey/black preferred)."""
    spread = np.zeros(len(cl)) if spread is None else spread
    all_live = [j for j in range(len(cl)) if cl[j].weight > 0]
    # a kit is uniform: a loose cluster is a crowd of spectators / staff in all kinds of clothes
    live = [j for j in all_live if spread[j] <= cfg.team_max_spread]
    if len(live) < 2:
        return None, None, None
    t1 = max(live, key=lambda j: pres[j])
    cand = [j for j in live if j != t1 and _different_kit(cl[j], cl[t1], cfg)]
    if not cand:
        return t1, None, None
    t2 = max(cand, key=lambda j: pres[j])
    if pres[t2] < cfg.team_min_presence:
        return None, None, None  # no second kit really on court: fall back to the colour rules
    rf = None
    if cfg.use_referee_class:
        rc = [j for j in all_live if j not in (t1, t2) and pres[j] <= cfg.ref_max_presence
              and spread[j] <= cfg.ref_max_spread
              and _different_kit(cl[j], cl[t1], cfg, True) and _different_kit(cl[j], cl[t2], cfg, True)]
        if rc:
            # grey shirts weigh most (all-black clusters are usually coaches / staff); presence only as sqrt
            score = {j: np.sqrt(pres[j]) * (cl[j].grey + 0.1 * cl[j].black + 0.05) for j in rc}
            best = max(score, key=score.get)
            if pres[best] >= cfg.ref_min_presence:
                rf = best
    return t1, t2, rf


def _referee_cluster(cl: list[_Cluster], exclude: set, cfg: TeamConfig) -> int | None:
    if not cfg.use_referee_class:
        return None
    rc = [j for j in range(len(cl)) if j not in exclude and cl[j].weight > 0 and cl[j].ach >= cfg.ref_min_achromatic
          and cl[j].colourful <= cfg.ref_max_colourful]
    # referee shirts are grey (or striped); all-black clusters are usually coaches and bench staff
    return max(rc, key=lambda j: np.sqrt(cl[j].weight) * (cl[j].grey + 0.1 * cl[j].black)) if rc else None


def _merge_team_twins(cl: list[_Cluster], role: dict[int, str], cfg: TeamConfig, seeded: bool = False) -> None:
    """k-means may cut one kit into two clusters (side-on / front-on, light / shade): give the twin the role."""
    rel = cfg.twin_min_rel_seeded if seeded else cfg.twin_min_rel
    for j, c in enumerate(cl):
        if j in role or c.weight <= 0:
            continue
        for r, lab in list(role.items()):
            if lab not in (A, B):
                continue
            t = cl[r]
            if (_cos(c.distinct_hue, t.distinct_hue) >= cfg.twin_hue_cos and c.distinct >= rel * t.distinct
                    and c.black <= t.black + 0.15):
                role[j] = lab
                break


def _display_lab(mu_team: np.ndarray, mu_other: np.ndarray | None) -> np.ndarray:
    """Representative jersey colour: the bins this team has more of than the other team."""
    centers = ta.bin_centers()
    ex = mu_team - (mu_other if mu_other is not None else 0.0)
    ex = np.clip(ex, 0, None)
    chrom = ex[: ta.N_HUE * ta.N_CHROMA]
    # weight higher chroma levels up: they carry the kit colour, the low level is mixed with skin/floor
    chrom = chrom * np.tile([0.5, 1.0, 1.5], ta.N_HUE)
    if chrom.sum() > 0.02:
        ab = (chrom[:, None] * centers[: ta.N_HUE * ta.N_CHROMA, 1:]).sum(0) / chrom.sum()
        ach = mu_team[ta.N_HUE * ta.N_CHROMA:]
        L = float((ach * centers[ta.N_HUE * ta.N_CHROMA:, 0]).sum() / max(ach.sum(), 1e-9)) if ach.sum() > 0 else 45
        return np.array([np.clip(L, 30, 60), ab[0], ab[1]])
    ach = ex[ta.N_HUE * ta.N_CHROMA:]
    if ach.sum() <= 0:
        ach = mu_team[ta.N_HUE * ta.N_CHROMA:]
    L = float((ach * centers[ta.N_HUE * ta.N_CHROMA:, 0]).sum() / max(ach.sum(), 1e-9))
    return np.array([L, 0.0, 0.0])


def _wmedian(v: np.ndarray, w: np.ndarray) -> float:
    o = np.argsort(v)
    cw = np.cumsum(w[o])
    return float(v[o][np.searchsorted(cw, 0.5 * cw[-1])])


def _measured_lab(mu_team: np.ndarray, mu_other: np.ndarray | None, desc: np.ndarray,
                  sector_lab: np.ndarray) -> np.ndarray | None:
    """Real jersey colour: the measured pixel colours in the hue sectors that set this team apart."""
    ex = ta.hue_mass(np.clip(mu_team - (mu_other if mu_other is not None else 0.0), 0, None))[:, 1:].sum(axis=1)
    if ex.max() <= 0.02:
        return None
    secs = np.flatnonzero(ex >= 0.5 * ex.max())
    S = sector_lab[:, secs, :]
    M = np.nan_to_num(ta.hue_mass(desc)[:, secs, 1:].sum(-1)) * ex[secs]
    ok = np.isfinite(S).all(axis=-1) & (M > 0)
    if ok.sum() < 3:
        return None
    vals, wts = S[ok], M[ok]
    return np.array([_wmedian(vals[:, c], wts) for c in range(3)])


def _track_hist(D: np.ndarray, wt: np.ndarray) -> np.ndarray | None:
    ok = np.isfinite(D).all(axis=1) & (wt > 0)
    if not ok.any():
        return None
    return (D[ok] * wt[ok, None]).sum(0) / wt[ok].sum()


def _detection_weights(appearance, n: int, cfg: TeamConfig) -> np.ndarray:
    occ = np.nan_to_num(np.asarray(appearance.occ, float), nan=1.0) if appearance is not None else np.zeros(n)
    w = np.clip(1.0 - occ / max(cfg.max_occlusion, 1e-6), 0.0, 1.0) ** 2
    if appearance is not None and appearance.height is not None:
        w = w * np.clip(np.asarray(appearance.height, float) / 80.0, 0.3, 1.0)
    return w


def _ref_shirt(h: np.ndarray, cfg: TeamConfig) -> bool:
    """Referee-like track histogram: few coloured pixels, a grey shirt share, and grey not swamped by black (staff
    in all-black clothes have almost no grey)."""
    if cfg.ref_shirt_min_grey <= 0 or not np.isfinite(h).all():
        return False
    ach = h[ta.N_HUE * ta.N_CHROMA:]
    colourful = float(ta.hue_mass(h)[:, 1:].sum())
    grey, black = float(ach[1] + ach[2]), float(ach[0])
    return (colourful <= cfg.ref_shirt_max_colourful and grey >= cfg.ref_shirt_min_grey
            and grey >= cfg.ref_shirt_grey_black * black and black >= cfg.ref_min_black)


def _in_court_share(track_ids: np.ndarray, T: np.ndarray, xy: np.ndarray, margin: float = 0.5) -> np.ndarray:
    """Share of each track's foot points inside the court (+ margin); 1 where unknown."""
    from replay import court
    xy = np.asarray(xy, float).reshape(-1, 2)
    if len(xy) != len(track_ids):
        return np.ones(len(T))
    inside = ((xy[:, 0] >= -margin) & (xy[:, 0] <= court.LENGTH + margin) & (xy[:, 1] >= -margin)
              & (xy[:, 1] <= court.WIDTH + margin))
    out = np.ones(len(T))
    for i, t in enumerate(T):
        m = track_ids == t
        if m.any():
            out[i] = float(inside[m].mean())
    return out


def _sideline_outside(track_ids: np.ndarray, T: np.ndarray, xy: np.ndarray | None) -> dict[int, float]:
    """Median distance (m) each track's foot points lie beyond a sideline (0 inside the court's width)."""
    if xy is None:
        return {}
    from replay import court
    xy = np.asarray(xy, float).reshape(-1, 2)
    if len(xy) != len(track_ids):
        return {}
    out = np.maximum(np.maximum(-xy[:, 1], xy[:, 1] - court.WIDTH), 0.0)
    res = {}
    for t in T:
        m = (track_ids == t) & np.isfinite(out)
        if m.any():
            res[int(t)] = float(np.median(out[m]))
    return res


# ----------------------------------------------------------------------------- main entry points
def appearance_for(work, data, video, cfg: TeamConfig, rows: np.ndarray, meta: GameMeta | None = None,
                   clip: bool = True):
    """Video-sampled descriptors for the given pass-1 rows (cached in the work folder); None without video."""
    if not cfg.resample_from_video:
        return None
    try:
        app = ta.load_or_compute(work, data, video, cfg.resample_every_s)
    except Exception as e:  # noqa: BLE001 - never let the optional resampling break pass 2
        print("team colours: video resampling failed:", e)
        app = None
    if app is None:
        return None
    rows = np.asarray(rows, int)
    out = app.subset(rows, data.rows[rows, 4] - data.rows[rows, 2])
    if clip and ta.FLOOR_MASK and cfg.clip_roles:  # broadcast: CLIP embeddings for a per-game referee/team probe
        try:
            from replay import clip_roles

            if clip_roles.available():
                emb = clip_roles.embed_rows(work, data, video, rows, cfg.clip_every_s)
                E = np.full((len(rows), 512), np.nan, np.float32)
                for i, r in enumerate(rows):
                    if int(r) in emb:
                        E[i] = emb[int(r)]
                out.clip = E
        except Exception as e:  # noqa: BLE001 - optional model: never break pass 2
            print("CLIP roles unavailable:", e)
    smp = (meta.raw.get("team_samples") or {}) if meta is not None else {}
    if smp:
        vid = ta.resolve_video(data, video)
        out.samples = {}
        for lab, feats in smp.items():
            if lab not in (A, B, REF) or not feats:
                continue
            rr = ta.find_sample_rows(data.rows, feats)
            have = {r: app.desc[r] for r in rr if np.isfinite(app.desc[r]).all()}
            missing = [r for r in rr if r not in have]
            if missing and vid is not None:
                have.update(ta.describe_rows(data, vid, missing))
            if have:
                out.samples[lab] = np.stack(list(have.values()))
    return out


def classify(track_ids: np.ndarray, frames: np.ndarray, feats: np.ndarray, meta: GameMeta, cfg: TeamConfig,
             seed: int = 0, appearance: ta.Appearance | None = None, xy: np.ndarray | None = None,
             _ignore_samples: bool = False) -> TeamResult:
    """track_ids/frames/feats: one row per detection (feats = the 6 pass-1 jersey features).

    ``appearance`` (from :func:`appearance_for`) holds the video-sampled descriptors of the same detections.
    ``xy`` (optional): court position (m) per detection - referees do not sit/stand far outside a sideline.
    """
    track_ids = np.asarray(track_ids)
    feats = np.asarray(feats, float).reshape(len(track_ids), -1)
    notes: list[str] = []
    tids, counts = np.unique(track_ids, return_counts=True)
    n_obs = dict(zip(tids.tolist(), counts.tolist()))
    pseudo = ta.pseudo_descriptors(feats[:, :6]) if feats.shape[1] >= 6 else np.full((len(feats), ta.DESC_DIM), np.nan)

    samples_feat = {} if _ignore_samples else {
        k: v for k, v in (meta.raw.get("team_samples") or {}).items() if k in (A, B, REF) and v}
    real = None
    if appearance is not None and np.isfinite(appearance.desc).all(axis=1).sum() >= 2:
        real = np.asarray(appearance.desc, float)
    if real is None and A in samples_feat and B in samples_feat:
        return classify_from_samples(track_ids, feats, samples_feat, meta, cfg)

    dw = _detection_weights(appearance if real is not None else None, len(track_ids), cfg)
    # ---- per-track histograms in the primary space (video descriptors) with pseudo fallback
    H_real, H_pseudo = {}, {}
    for t in tids:
        m = track_ids == t
        h = None
        if real is not None:
            h = _track_hist(real[m], dw[m])
            if h is None:  # every sighting occluded: take them anyway
                h = _track_hist(real[m], np.ones(int(m.sum())))
        if h is not None:
            H_real[int(t)] = h
        hp = _track_hist(pseudo[m], np.ones(int(m.sum())))
        if hp is not None:
            H_pseudo[int(t)] = hp
    space = H_real if len(H_real) >= 2 else H_pseudo
    if real is not None and space is H_pseudo:
        notes.append("too few video-sampled tracks; using pass-1 colours")
    used_real = space is H_real and real is not None

    T = np.array(sorted(space), int)
    if len(T) == 0:
        return TeamResult({int(t): UNKNOWN for t in tids}, {int(t): 0.0 for t in tids}, {}, False, {}, notes)
    D = np.stack([space[t] for t in T])
    X = np.sqrt(D)
    W = np.array([n_obs[t] for t in T], float)
    if cfg.cluster_court_weight > 0 and xy is not None:
        # spectators, bench and table staff must not shape the kit clusters: weight tracks by time on court
        W = W * (1.0 - cfg.cluster_court_weight + cfg.cluster_court_weight * _in_court_share(track_ids, T, xy))

    # ---- seeds from user-picked players
    seed_desc = {}
    if used_real and appearance.samples and not _ignore_samples:
        seed_desc = {k: v.mean(axis=0) for k, v in appearance.samples.items() if len(v)}
    elif not used_real and samples_feat:
        seed_desc = {k: ta.pseudo_descriptors(np.array(v, float)).mean(axis=0) for k, v in samples_feat.items()}
    seeded_by_samples = A in seed_desc and B in seed_desc
    seed_labels = [lab for lab in (A, B, REF) if lab in seed_desc] if seeded_by_samples else []
    seed_labels = seed_labels[:len(T)]  # fewer tracks than picked kits (short clip): k-means cannot take more seeds

    rng = np.random.default_rng(seed)
    k = min(max(cfg.n_clusters, len(seed_labels) + 1), len(T))
    seeds = np.sqrt(np.stack([seed_desc[lab] for lab in seed_labels])) if seed_labels else None
    lab_of, C = _kmeans(X, W, k, rng, seeds=seeds)
    common = _weighted_median(D, W)
    clusters = [_cluster_stats(D[lab_of == j], W[lab_of == j], common) if (lab_of == j).any() else
                _cluster_stats(C[j][None] ** 2, np.ones(1) * 1e-9, common) for j in range(len(C))]
    if os.environ.get("TEAMS_DEBUG"):
        for j, c in enumerate(clusters):
            print(f"cluster {j}: n={int((lab_of == j).sum())} W={c.weight:.0f} colourful={c.colourful:.2f} "
                  f"distinct={c.distinct:.2f} ach={c.ach:.2f} black={c.black:.2f} grey={c.grey:.2f} "
                  f"hue={np.round(100 * c.distinct_hue).astype(int).tolist()}")

    if seeded_by_samples and cfg.role_by_presence and SEED_CHECK and not _ignore_samples:
        # picked / automatic samples that match no kit on court (a bad crop, a bench player) would steer the
        # clustering: then a real team ends up as "referee"/"unknown" (NCAA broadcast, 18.10). Check the seeded
        # team clusters are really on court; otherwise classify from the data alone.
        pres = _presence(track_ids, frames, xy, T, lab_of, len(C))
        weak = [lab for j, lab in enumerate(seed_labels) if lab in (A, B) and pres[j] < SEED_MIN_TEAM_PRESENCE]
        if weak:
            r = classify(track_ids, frames, feats, meta, cfg, seed, appearance, xy, _ignore_samples=True)
            r.notes.append(f"team samples ignored: {', '.join(weak)} kit has {SEED_MIN_TEAM_PRESENCE:.1f}"
                           " or fewer people on court per frame")
            return r
    if seeded_by_samples:
        role = {j: lab for j, lab in enumerate(seed_labels)}
        if REF not in seed_labels:
            rf = _referee_cluster(clusters, set(role), cfg)
            if rf is not None:
                role[rf] = REF
    else:
        t1 = t2 = rf = None
        if cfg.role_by_presence:
            pres = _presence(track_ids, frames, xy, T, lab_of, len(C))
            spread = np.array([float(np.average(np.linalg.norm(X[lab_of == j] - C[j], axis=1), weights=W[lab_of == j]))
                               if (lab_of == j).any() else np.inf for j in range(len(C))])
            t1, t2, rf = _assign_roles_presence(clusters, pres, cfg, spread)
            if os.environ.get("TEAMS_DEBUG"):
                print("presence per frame:", np.round(pres, 2).tolist(), "spread:", np.round(spread, 3).tolist(),
                      "-> teams", t1, t2, "referee", rf)
        if t1 is None or t2 is None:
            t1, t2, rf = _assign_roles(clusters, cfg)
        role = {}
        if t1 is not None and t2 is not None:
            # deterministic default: the team seen more is A, unless the user's kit colours say otherwise
            first, second = (t1, t2) if clusters[t1].weight >= clusters[t2].weight else (t2, t1)
            role = {first: A, second: B}
        elif t1 is not None:
            role = {t1: A}
            notes.append("only one team colour found")
        if rf is not None:
            role[rf] = REF
    _merge_team_twins(clusters, role, cfg, seeded_by_samples)

    # ---- per-track labels: own cluster role, or the nearest role cluster if within its spread
    def dist(x, c):
        return float(np.linalg.norm(x - c))

    radius = {}
    for j, lab in role.items():
        m = lab_of == j
        if m.any():
            dd = np.array([dist(x, C[j]) for x in X[m]])
            radius[j] = max(cfg.min_fit_radius, float(np.quantile(dd, 0.9)) * cfg.fit_radius_scale)
        else:
            radius[j] = cfg.min_fit_radius
    # team-colour mass per track: coloured pixels in the hue sectors that set each team apart. k-means may cut
    # a team in two (e.g. side-on vs front-on players); a track with as much team colour as the team's own
    # members, and clearly more than of the other team's colour, belongs to it.
    team_js = [j for j, lab in role.items() if lab in (A, B)]
    strong = ta.hue_mass(D)[:, :, 2]  # (tracks, N_HUE) saturated pixels only: skin and hair rarely get there
    tmass, tmin = {}, {}
    for j in team_js:
        others = [o for o in team_js if role[o] != role[j]]
        ref_mu = (sum(clusters[o].mu * clusters[o].weight for o in others) / sum(clusters[o].weight for o in others)
                  if others else common)
        ex = ta.hue_mass(np.clip(clusters[j].mu - ref_mu, 0, None))[:, 1:].sum(1)
        secs = ex >= 0.5 * ex.max() if ex.max() > 0 else np.zeros(ta.N_HUE, bool)
        tmass[j] = strong[:, secs].sum(1)
        core = lab_of == j
        tmin[j] = cfg.team_mass_frac * float(np.median(tmass[j][core])) if core.any() else np.inf
    has_ref = REF in role.values()
    ref_achromatic = all(clusters[j].colourful <= cfg.ref_max_colourful for j, v in role.items() if v == REF)
    off_side = _sideline_outside(track_ids, T, xy)
    track_team: dict[int, str] = {}
    for ti, t in enumerate(T):
        j = int(lab_of[ti])
        if j in role:
            lab = role[j]
            if lab == REF and ref_achromatic and float(D[ti, -3] + D[ti, -2]) < cfg.ref_min_grey and not (
                    used_real and _ref_shirt(D[ti], cfg)):
                lab = UNKNOWN  # all black (coach, staff), no grey referee shirt
            if lab == REF and ref_achromatic and used_real and not ta.FLOOR_MASK and float(D[ti, -4]) < cfg.ref_min_black:
                lab = UNKNOWN  # no black trousers: a spectator in a grey top, not a referee
            if lab == REF and off_side.get(int(t), 0.0) > cfg.ref_max_sideline_out_m:
                lab = UNKNOWN  # sits/stands beyond a sideline: bench, table, coach
            track_team[int(t)] = lab
            continue
        # only teams adopt look-alikes: an all-black coach is near the referee cluster, but is no referee
        near = sorted(team_js, key=lambda r: dist(X[ti], C[r]))
        lab = UNKNOWN
        if near and dist(X[ti], C[near[0]]) <= radius[near[0]]:
            lab = role[near[0]]
        else:
            for r in team_js:
                other = max((tmass[o][ti] for o in team_js if role[o] != role[r]), default=0.0)
                if tmass[r][ti] >= max(tmin[r], cfg.min_team_colourful) and tmass[r][ti] >= 1.25 * other:
                    lab = role[r]
                    break
        if (lab == UNKNOWN and has_ref and ref_achromatic and used_real and _ref_shirt(D[ti], cfg)
                and off_side.get(int(t), 0.0) <= cfg.ref_max_sideline_out_m):
            lab = REF  # grey shirt over black (k-means put the referee into a second achromatic cluster)
        track_team[int(t)] = lab
    if appearance is not None and appearance.zebra is not None:  # broadcast: referee = striped shirt, on court, moving
        zb = np.asarray(appearance.zebra, float)
        from replay import court

        clip_ref = _clip_probe_referees(track_ids, T, zb, track_team, appearance.clip)

        for t in T:
            m = track_ids == t
            v = zb[m]
            v = v[np.isfinite(v)]
            zebra = len(v) >= ZEBRA_MIN_SAMPLES and float(np.median(v)) >= ZEBRA_REF
            on_court = moving = True
            if xy is not None and len(xy) == len(track_ids):
                p = np.asarray(xy, float)[m]
                p = p[np.isfinite(p).all(axis=1)]
                if len(p):
                    med = np.median(p, axis=0)
                    on_court = bool(-REF_OUT_M <= med[0] <= court.LENGTH + REF_OUT_M
                                    and -REF_OUT_M <= med[1] <= court.WIDTH + REF_OUT_M)
                    fr_t = np.asarray(frames)[m]
                    long = (fr_t.max() - fr_t.min()) >= REF_STILL_SAMPLES
                    moving = not (long and float(np.percentile(np.linalg.norm(p - med, axis=1), 90)) < REF_STILL_M)
            if clip_ref is not None and int(t) in clip_ref:
                zebra = clip_ref[int(t)]  # per-game CLIP probe (trained on this game's zebra + colour labels)
            if zebra and on_court and moving:
                track_team[int(t)] = REF
            elif track_team.get(int(t)) == REF:
                track_team[int(t)] = UNKNOWN  # broadcast: no striped shirt (white warm-ups on the bench, staff)

    # ---- display colours / representative Lab
    team_cluster = {lab: j for j, lab in role.items()}
    mu = {lab: clusters[j].mu for lab, j in team_cluster.items()}
    disp_lab = {}
    for lab in (A, B):
        if lab in mu:
            other = mu.get(B if lab == A else A)
            disp_lab[lab] = _display_lab(mu[lab], other)
            if used_real and appearance.sector_lab is not None:
                members = np.isin(track_ids, [t for t, v in track_team.items() if v == lab])
                measured = _measured_lab(mu[lab], other, real[members], appearance.sector_lab[members])
                if measured is not None:
                    disp_lab[lab] = measured
    if cfg.achromatic_kit_names:
        # white / black kits: name and show them white or black (their 'measured' colour is skin or floor tint)
        def ach_L(m):
            ach = m[ta.N_HUE * ta.N_CHROMA:]
            return float((ach * np.array([10.0, 32.0, 57.0, 85.0])).sum() / max(ach.sum(), 1e-9))
        def light(m):
            return float(m[ta.N_HUE * ta.N_CHROMA + 2] + m[ta.N_HUE * ta.N_CHROMA + 3])
        achro = {lab: clusters[team_cluster[lab]].distinct < cfg.min_team_colourful
                 or float(np.hypot(disp_lab[lab][1], disp_lab[lab][2])) < 9.0
                 or (light(mu[lab]) >= 0.1 and light(mu[lab]) >= 0.6 * clusters[team_cluster[lab]].distinct)
                 for lab in disp_lab if lab in (A, B)}
        for lab, is_ach in achro.items():
            if not is_ach:
                continue
            o = B if lab == A else A
            ref_L = ach_L(mu[o]) if achro.get(o) else 50.0
            disp_lab[lab] = np.array([85.0, 0.0, 0.0]) if ach_L(mu[lab]) >= ref_L else np.array([12.0, 0.0, 0.0])
    if REF in mu:  # referees: their grey/black, not the skin tones that dominate the coloured pixels
        ach = mu[REF][ta.N_HUE * ta.N_CHROMA:]
        Ls = ta.bin_centers()[ta.N_HUE * ta.N_CHROMA:, 0]
        disp_lab[REF] = np.array([float((ach * Ls).sum() / max(ach.sum(), 1e-9)), 0.0, 0.0])

    # ---- A/B order and colours from user-set kit colours (only when they match what is on screen)
    seeded = seeded_by_samples
    colors = {lab: lab_to_hex(disp_lab[lab]) for lab in (A, B) if lab in disp_lab}
    if not seeded_by_samples and A in disp_lab and B in disp_lab and colors_are_user_set(meta):
        u = [rgb_to_lab(meta.home.primary), rgb_to_lab(meta.away.primary)]
        straight = [_color_cost(u[0], disp_lab[A]), _color_cost(u[1], disp_lab[B])]
        crossed = [_color_cost(u[0], disp_lab[B]), _color_cost(u[1], disp_lab[A])]
        best = straight if sum(straight) <= sum(crossed) else crossed
        if max(best) <= cfg.user_color_max_cost:
            seeded = True
            if best is crossed:
                role = {j: {A: B, B: A}.get(v, v) for j, v in role.items()}
                track_team = {t: {A: B, B: A}.get(v, v) for t, v in track_team.items()}
                disp_lab = {({A: B, B: A}.get(k, k)): v for k, v in disp_lab.items()}
                team_cluster = {({A: B, B: A}.get(k, k)): v for k, v in team_cluster.items()}
            colors = {A: rgb_to_hex(meta.home.primary), B: rgb_to_hex(meta.away.primary)}
        else:
            notes.append("kit colours in the game data do not match the video; ignored")
    elif seeded_by_samples and colors_are_user_set(meta):
        colors = {A: rgb_to_hex(meta.home.primary), B: rgb_to_hex(meta.away.primary)}

    # ---- confidence: share of a track's sightings whose own histogram agrees with the track label
    det_space = real if used_real else pseudo
    Xd = np.sqrt(np.clip(np.nan_to_num(det_space, nan=0.0), 0, None))
    has = np.isfinite(det_space).all(axis=1)
    role_lab = {j: lab for lab, j in team_cluster.items()}
    cent_ids = sorted(role_lab)  # nearest labelled cluster per sighting
    if cent_ids:
        dd = ((Xd[:, None, :] - C[cent_ids][None]) ** 2).sum(-1)
        det_lab = np.array([role_lab[cent_ids[j]] for j in dd.argmin(axis=1)], dtype=object)
    else:
        det_lab = np.full(len(Xd), UNKNOWN, dtype=object)
    track_share: dict[int, float] = {}
    for t in tids:
        t = int(t)
        lab = track_team.get(t)
        if lab is None:  # no histogram at all: try the pseudo space against the labelled tracks
            lab = UNKNOWN
            track_team[t] = lab
        m = (track_ids == t) & has
        share = float(np.mean(det_lab[m] == lab)) if m.any() else 0.0
        track_share[t] = max(share, 0.3) if lab != UNKNOWN else 0.0

    # tracks that only exist in the pseudo space (no video sample): nearest labelled prototype there
    if used_real:
        missing = [int(t) for t in tids if int(t) not in H_real and int(t) in H_pseudo]
        protos = {}
        for lab in (A, B, REF):
            mem = [t for t, v in track_team.items() if v == lab and t in H_pseudo and t in H_real]
            if mem:
                P = np.sqrt(np.stack([H_pseudo[t] for t in mem]))
                c = P.mean(axis=0)
                protos[lab] = (c, max(cfg.min_fit_radius, float(np.quantile(np.linalg.norm(P - c, axis=1), 0.9))))
        for t in missing:
            x = np.sqrt(H_pseudo[t])
            best = min(protos, key=lambda lab: np.linalg.norm(x - protos[lab][0]), default=None)
            if best is not None and np.linalg.norm(x - protos[best][0]) <= protos[best][1]:
                track_team[t], track_share[t] = best, 0.3

    # per-detection evidence for splitting tracks that switch people (nearest cluster of any role)
    det_all = np.full(len(Xd), "", dtype=object)
    if has.any():
        near = ((Xd[has][:, None, :] - C[None]) ** 2).sum(-1).argmin(axis=1)
        det_all[has] = [role.get(int(j), UNKNOWN) for j in near]
        if used_real:  # referee evidence needs the grey shirt (an all-black coach is near the referee cluster)
            hi = np.flatnonzero(has)
            grey = det_space[hi, -3] + det_space[hi, -2]
            det_all[hi[(det_all[hi] == REF) & (grey < cfg.ref_min_grey)]] = UNKNOWN
    if not used_real:
        det_all[:] = ""  # pass-1 colours are too weak to split identities
    elif cfg.adopt_vote_share > 0:
        # a track left "unknown" (its mean histogram fell into a bench/other cluster: dark kits seen side-on) whose
        # own sightings clearly point to one team (and hardly to the other team or the referees) joins that team
        for t in tids:
            t = int(t)
            if track_team.get(t) != UNKNOWN:
                continue
            ev = det_all[(track_ids == t) & has]
            if len(ev) < 6:
                continue
            cnt = {lab: int(np.sum(ev == lab)) for lab in (A, B, REF)}
            for lab, other in ((A, B), (B, A)):
                if (cnt[lab] >= cfg.adopt_vote_share * len(ev) and cnt[lab] >= 3 * cnt[other]
                        and cnt[REF] <= 0.2 * len(ev)):
                    track_team[t], track_share[t] = lab, max(0.3, cnt[lab] / len(ev))
                    break

    centroids = {lab: np.round(v, 1).tolist() for lab, v in disp_lab.items()}
    for lab in (A, B):
        if lab not in colors and lab in disp_lab:
            colors[lab] = lab_to_hex(disp_lab[lab])
    hists = {lab: np.round(clusters[j].mu, 5).tolist() for lab, j in team_cluster.items() if lab in (A, B)}
    space = f"{'video' if used_real else 'pass1'}{ta.DESC_VERSION}"
    return TeamResult(track_team, track_share, centroids, seeded, colors, notes, hists, space, det_all, C.copy(),
                      [role.get(j, UNKNOWN) for j in range(len(C))], {int(t): int(lab_of[i]) for i, t in enumerate(T)})


# ----------------------------------------------------------------------------- identity switches between kits
def _viterbi(ev: list[str], states: list[str], switch_cost: float) -> list[str]:
    """Most likely label sequence: cost 1 per sighting that disagrees with the state, ``switch_cost`` per change."""
    n, m = len(ev), len(states)
    cost = np.array([0.0 if states[s] == ev[0] else 1.0 for s in range(m)])
    back = np.zeros((n, m), int)
    for i in range(1, n):
        trans = cost[:, None] + switch_cost * (1 - np.eye(m))
        back[i] = trans.argmin(axis=0)
        cost = trans.min(axis=0) + np.array([0.0 if states[s] == ev[i] else 1.0 for s in range(m)])
    path = [int(cost.argmin())]
    for i in range(n - 1, 0, -1):
        path.append(int(back[i, path[-1]]))
    return [states[s] for s in path[::-1]]


def split_mixed_tracks(tid: np.ndarray, keep: np.ndarray, det_i: np.ndarray, teams: TeamResult,
                       cfg: TeamConfig) -> np.ndarray:
    """Split court tracks whose sightings show different kits for a sustained time (an identity switch between,
    say, an orange player and the referee) into one track per kit. Short contrary runs never flip a label: a
    change costs ``cfg.split_switch_cost`` disagreeing sightings, so one identity keeps one label unless the colour
    evidence clearly changes. Returns the new track id per detection (all of ``tid``) and updates ``teams``
    (track_team / track_share for the new ids). Tracks that are not split keep their id and label."""
    tid = np.asarray(tid).copy()
    if not cfg.split_mixed_tracks or teams.det_label is None:
        return tid
    sub = np.flatnonzero(keep)
    lab = np.asarray(teams.det_label, dtype=object)
    if len(lab) != len(sub):
        return tid
    next_id = int(tid.max()) + 1 if len(tid) else 1
    # grey/black evidence (referee or bench/other) is one "neutral" state: single referee sightings often look
    # like the other achromatic clusters, so only a change between kits (or kit <-> neutral) splits a track
    states = [A, B, "neutral"]
    n_split = 0
    ts, ds = tid[sub], np.asarray(det_i)[sub]  # speed: group detections by track once (not a mask per track)
    o = np.argsort(ts, kind="stable")
    ut, st = np.unique(ts[o], return_index=True)
    for t, m in zip(ut, np.split(o, st[1:])):
        m = m[np.argsort(ds[m], kind="stable")]
        evm = m[(lab[m] != "") & ((lab[m] != UNKNOWN) | cfg.split_on_unknown)]
        if len(evm) < 2 * cfg.split_min_evidence:
            continue
        evl = lab[evm]
        path = _viterbi([v if v in (A, B) else "neutral" for v in evl], states, cfg.split_switch_cost)
        segs = []  # [label, first ev pos, last ev pos]
        for j, s in enumerate(path):
            if segs and segs[-1][0] == s:
                segs[-1][2] = j
            else:
                segs.append([s, j, j])
        # segments with too little evidence join their neighbour (the longer one)
        while len(segs) > 1:
            short = [k for k, sg in enumerate(segs) if sg[2] - sg[1] + 1 < cfg.split_min_evidence]
            if not short:
                break
            k = short[0]
            nb = k - 1 if k == len(segs) - 1 or (k > 0 and segs[k - 1][2] - segs[k - 1][1] >= segs[k + 1][2] - segs[k + 1][1]) else k + 1
            lo, hi = min(segs[k][1], segs[nb][1]), max(segs[k][2], segs[nb][2])
            segs[nb] = [segs[nb][0], lo, hi]
            del segs[k]
            merged = []
            for sg in segs:
                if merged and merged[-1][0] == sg[0]:
                    merged[-1][2] = sg[2]
                else:
                    merged.append(sg)
            segs = merged
        if len(segs) < 2:
            continue
        # every sighting (with or without a histogram) belongs to the segment of the nearest evidence in time
        fr_ev = ds[evm]
        fr_all = ds[m]
        bounds = [(fr_ev[segs[k][2]] + fr_ev[segs[k + 1][1]]) / 2.0 for k in range(len(segs) - 1)]
        seg_of = np.searchsorted(np.asarray(bounds), fr_all, side="right")
        sizes = np.bincount(seg_of, minlength=len(segs))
        main = int(sizes.argmax())
        orig = teams.track_team.get(int(t), UNKNOWN)
        for k, sg in enumerate(segs):
            sel = sub[m[seg_of == k]]
            evk = evl[sg[1]:sg[2] + 1]
            lab_k = sg[0]
            if lab_k == "neutral":
                n_ref = float(np.sum(evk == REF))
                if k == main and orig in (REF, UNKNOWN):
                    lab_k = orig
                else:
                    lab_k = REF if n_ref >= cfg.split_ref_frac * len(evk) else UNKNOWN
            share = float(np.mean(evk == lab_k)) if len(evk) else 0.0
            nid = int(t) if k == main else next_id
            if k != main:
                next_id += 1
                tid[sel] = nid
            teams.track_team[nid] = lab_k
            teams.track_share[nid] = 0.0 if lab_k == UNKNOWN else max(share, 0.3)
        n_split += 1
    if n_split:
        teams.notes.append(f"{n_split} tracks split where the jersey colour changed (identity switch)")
    return tid


# ----------------------------------------------------------------------------- legacy: picked samples, no video
SAMPLE_W = np.array([0.5, 1, 1, 0.25, 0.5, 0.5])  # lightness counts less than colour; shorts less than shirt
SAMPLE_MAX_DIST = 14.0


def _prep_samples(f: np.ndarray) -> np.ndarray:
    f = np.asarray(f, float).reshape(-1, 6).copy()
    miss = ~np.isfinite(f[:, 3:]).all(axis=1)
    f[miss, 3:] = f[miss, :3]
    return f * SAMPLE_W


def classify_from_samples(track_ids: np.ndarray, feats: np.ndarray, samples: dict[str, list], meta: GameMeta,
                          cfg: TeamConfig) -> TeamResult:
    """Nearest reference colour in the 6 pass-1 features (used when the video is not available)."""
    cents = {lab: _prep_samples(np.array(v)).mean(axis=0) for lab, v in samples.items() if len(v)}
    track_team, track_share = {}, {}
    valid = np.isfinite(feats[:, :3]).all(axis=1)
    for t in np.unique(track_ids):
        m = (track_ids == t) & valid
        if not m.any():
            track_team[int(t)], track_share[int(t)] = UNKNOWN, 0.0
            continue
        tf = np.median(_prep_samples(feats[m]), axis=0)
        d = {lab: float(np.linalg.norm(tf - c)) for lab, c in cents.items()}
        lab = min(d, key=d.get)
        far = d[lab] > SAMPLE_MAX_DIST
        track_team[int(t)] = UNKNOWN if far else lab
        track_share[int(t)] = 0.0 if far else float(1 - d[lab] / SAMPLE_MAX_DIST)
    centroids = {lab: (c[:3] / SAMPLE_W[:3]).round(1).tolist() for lab, c in cents.items()}
    if colors_are_user_set(meta):
        colors = {A: rgb_to_hex(meta.home.primary), B: rgb_to_hex(meta.away.primary)}
    else:
        colors = {lab: lab_to_hex(np.array(centroids[lab])) for lab in (A, B) if lab in centroids}
    return TeamResult(track_team, track_share, centroids, True, colors)


def swap_ab(result: TeamResult) -> TeamResult:
    """Exchange which players are home (A) and away (B).

    User-set colours belong to the team names, so they stay; detected colours belong to the players, so they move.
    """
    sw = {A: B, B: A}
    result.track_team = {t: sw.get(v, v) for t, v in result.track_team.items()}
    result.centroids = {sw.get(k, k): v for k, v in result.centroids.items()}
    result.hists = {sw.get(k, k): v for k, v in result.hists.items()}
    if getattr(result, "det_label", None) is not None:
        result.det_label = np.array([sw.get(v, v) for v in result.det_label], dtype=object)
    if getattr(result, "cluster_roles", None):
        result.cluster_roles = [sw.get(v, v) for v in result.cluster_roles]
    if not result.seeded:
        result.colors = {sw.get(k, k): v for k, v in result.colors.items()}
    return result
