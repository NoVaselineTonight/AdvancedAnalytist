"""Tracking metrics for the benchmark (owner: benchmark agent): HOTA / IDF1 / MOTA on dense identity labels,
hide-and-predict extrapolation error for players (offline) and the ball (in the pass-2 worker), ball position error.

Conventions: a "sequence" is a list of timesteps; each timestep has gt ids, tracker ids and a similarity matrix
(|gt| x |tr|, 0..1). Image similarity = box IoU; court similarity = max(0, 1 - d / COURT_SIM_M).
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

ALPHAS = np.arange(0.05, 0.99, 0.05)
COURT_SIM_M = 2.0  # court similarity reaches 0 at 2 m (0.5 at 1 m)
EXTRAP_S = (1, 3, 5, 10)


# ---------------------------------------------------------------------------------------------- HOTA / IDF1 / MOTA
def hota(steps: list[tuple[list, list, np.ndarray]]) -> dict:
    """HOTA (Luiten et al. 2020, TrackEval algorithm), averaged over alpha 0.05..0.95. Returns HOTA, DetA, AssA,
    LocA."""
    gid = {g: n for n, g in enumerate(sorted({g for s in steps for g in s[0]}, key=str))}
    tid = {t: n for n, t in enumerate(sorted({t for s in steps for t in s[1]}, key=str))}
    G, T = len(gid), len(tid)
    if G == 0:
        return {"HOTA": None, "DetA": None, "AssA": None, "LocA": None}
    pot = np.zeros((G, T))
    gcount, tcount = np.zeros((G, 1)), np.zeros((1, T))
    for g, t, sim in steps:
        gi, ti = [gid[x] for x in g], [tid[x] for x in t]
        if gi and ti:
            denom = sim.sum(0)[None, :] + sim.sum(1)[:, None] - sim
            s_iou = np.where(denom > 1e-9, sim / np.maximum(denom, 1e-9), 0.0)
            pot[np.ix_(gi, ti)] += s_iou
        gcount[gi] += 1
        tcount[0, ti] += 1
    align = pot / np.maximum(gcount + tcount - pot, 1e-9)
    A = len(ALPHAS)
    TP, FN, FP, loc = np.zeros(A), np.zeros(A), np.zeros(A), np.zeros(A)
    matches = np.zeros((A, G, T))
    for g, t, sim in steps:
        gi, ti = [gid[x] for x in g], [tid[x] for x in t]
        if not gi:
            FP += len(ti)
            continue
        if not ti:
            FN += len(gi)
            continue
        score = align[np.ix_(gi, ti)] * sim
        r, c = linear_sum_assignment(-score)
        for a, alpha in enumerate(ALPHAS):
            ok = sim[r, c] >= alpha - 1e-9
            n = int(ok.sum())
            TP[a] += n
            FN[a] += len(gi) - n
            FP[a] += len(ti) - n
            loc[a] += sim[r[ok], c[ok]].sum()
            matches[a, np.array(gi)[r[ok]], np.array(ti)[c[ok]]] += 1
    det = TP / np.maximum(TP + FN + FP, 1)
    ass = np.zeros(A)
    for a in range(A):
        m = matches[a]
        ass_a = m / np.maximum(gcount + tcount - m, 1)
        ass[a] = (m * ass_a).sum() / max(TP[a], 1)
    h = np.sqrt(det * ass)
    return {"HOTA": float(h.mean()), "DetA": float(det.mean()), "AssA": float(ass.mean()),
            "LocA": float((loc / np.maximum(TP, 1)).mean())}


def idf1(steps: list[tuple[list, list, np.ndarray]], thr: float = 0.5) -> dict:
    """IDF1 = 2 IDTP / (gt dets + tracker dets); IDTP from the best one-to-one gt-id <-> tracker-id assignment
    of frames where the pair is matchable (similarity >= thr)."""
    pair = defaultdict(int)
    ng = nt = 0
    for g, t, sim in steps:
        ng += len(g)
        nt += len(t)
        for i, j in zip(*np.nonzero(sim >= thr)):
            pair[(g[i], t[j])] += 1
    if ng == 0:
        return {"IDF1": None, "IDP": None, "IDR": None}
    gs = sorted({p[0] for p in pair}, key=str)
    ts = sorted({p[1] for p in pair}, key=str)
    idtp = 0
    if gs and ts:
        M = np.zeros((len(gs), len(ts)))
        gi, ti = {g: n for n, g in enumerate(gs)}, {t: n for n, t in enumerate(ts)}
        for (g, t), v in pair.items():
            M[gi[g], ti[t]] = v
        r, c = linear_sum_assignment(-M)
        idtp = int(M[r, c].sum())
    return {"IDF1": 2 * idtp / max(ng + nt, 1), "IDP": idtp / max(nt, 1), "IDR": idtp / max(ng, 1)}


def mota(steps: list[tuple[list, list, np.ndarray]], thr: float = 0.5) -> dict:
    """CLEAR MOT: per-step assignment (sim >= thr) preferring the previous partner; IDSW = a gt matched to a
    different tracker id than the last time it was matched. MOTA = 1 - (FN + FP + IDSW) / gt."""
    last: dict = {}
    prev_pairs: set = set()
    tp = fn = fp = idsw = ng = 0
    simsum = 0.0
    for g, t, sim in steps:
        ng += len(g)
        if not g or not t:
            fn += len(g)
            fp += len(t)
            prev_pairs = set()
            continue
        score = np.where(sim >= thr, sim, 0.0)
        bonus = np.array([[1000.0 if (gg, tt) in prev_pairs else 0.0 for tt in t] for gg in g])
        r, c = linear_sum_assignment(-(score + bonus * (score > 0)))
        ok = [(i, j) for i, j in zip(r, c) if sim[i, j] >= thr]
        cur = set()
        for i, j in ok:
            if g[i] in last and last[g[i]] != t[j]:
                idsw += 1
            last[g[i]] = t[j]
            cur.add((g[i], t[j]))
            simsum += sim[i, j]
        tp += len(ok)
        fn += len(g) - len(ok)
        fp += len(t) - len(ok)
        prev_pairs = cur
    if ng == 0:
        return {"MOTA": None, "MOTP": None, "IDSW": None}
    return {"MOTA": 1 - (fn + fp + idsw) / ng, "MOTP": simsum / max(tp, 1), "IDSW": idsw}


def iou_matrix(a: list, b: list) -> np.ndarray:
    if not a or not b:
        return np.zeros((len(a), len(b)))
    A, B = np.asarray(a, float), np.asarray(b, float)
    ix = np.clip(np.minimum(A[:, None, 2], B[None, :, 2]) - np.maximum(A[:, None, 0], B[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(A[:, None, 3], B[None, :, 3]) - np.maximum(A[:, None, 1], B[None, :, 1]), 0, None)
    inter = ix * iy
    area = lambda X: (X[:, 2] - X[:, 0]) * (X[:, 3] - X[:, 1])  # noqa: E731
    u = area(A)[:, None] + area(B)[None, :] - inter
    return np.where(u > 0, inter / np.maximum(u, 1e-9), 0.0)


def court_sim(a: list, b: list) -> np.ndarray:
    if not a or not b:
        return np.zeros((len(a), len(b)))
    d = np.linalg.norm(np.asarray(a, float)[:, None, :] - np.asarray(b, float)[None, :, :], axis=2)
    return np.clip(1.0 - d / COURT_SIM_M, 0.0, 1.0)


IGNORE_KINDS = ("ignore", "bench", "spectator", "duplicate")


def build_steps(frames: list[dict], outs_at, label_xy_at=None, space: str = "image") -> list:
    """Timesteps from dense label frames. ``outs_at(frame) -> shown output rows`` ({id, box, xy}).
    image: gt = labelled people with a box (not hidden), tracker = rows with a detection box, IoU.
    court: gt = labelled people with a court point (``court`` or the projected box foot), tracker = all shown rows
    (predicted ones too), court similarity. Tracker rows on ignore/bench/spectator labels are dropped; on frames
    with ``complete: false`` tracker rows that overlap no gt are dropped (not counted as FP)."""
    steps = []
    for fr in frames:
        outs = outs_at(int(fr["frame"]))
        people = fr.get("people", [])
        gts, ign = [], []
        for n, p in enumerate(people):
            kind = p.get("kind", "player")
            if kind in IGNORE_KINDS:
                if p.get("box"):
                    ign.append(p["box"])
                continue
            if kind not in ("player", "referee") or not p.get("pid") or p.get("hidden"):
                continue
            if space == "image":
                if p.get("box"):
                    gts.append((p["pid"], p["box"]))
            else:
                xy = p.get("court") or (label_xy_at(fr, n) if label_xy_at else None)
                if xy is not None:
                    gts.append((p["pid"], xy))
        ign += [b for b in fr.get("ignore", [])]
        if space == "image":
            trs = [(o["id"], o["box"]) for o in outs if o.get("box")]
            if ign and trs:
                keep = iou_matrix([b for _, b in trs], ign).max(1) < 0.5
                trs = [x for x, k in zip(trs, keep) if k]
            sim = iou_matrix([b for _, b in gts], [b for _, b in trs])
        else:
            trs = [(o["id"], o["xy"]) for o in outs]
            if ign:
                ign_ids = {o["id"] for o in outs if o.get("box")
                           and iou_matrix([o["box"]], ign).max() >= 0.5}
                trs = [x for x in trs if x[0] not in ign_ids]
            sim = court_sim([b for _, b in gts], [b for _, b in trs])
        if not fr.get("complete", True) and trs:
            keep = sim.max(0) > 0.05 if len(gts) else np.zeros(len(trs), bool)
            trs = [x for x, k in zip(trs, keep) if k]
            sim = sim[:, keep] if len(gts) else np.zeros((0, len(trs)))
        steps.append(([g for g, _ in gts], [t for t, _ in trs], sim))
    return steps


def identity_scores(frames: list[dict], outs_at, label_xy_at=None) -> dict:
    """HOTA/DetA/AssA/IDF1/MOTA/IDSW in image space (+ HOTA and IDF1 in court space when possible)."""
    res: dict = {}
    st = build_steps(frames, outs_at, label_xy_at, "image")
    if any(s[0] for s in st):
        res.update({k: v for k, v in hota(st).items()})
        res.update(idf1(st))
        res.update(mota(st))
    stc = build_steps(frames, outs_at, label_xy_at, "court")
    if any(s[0] for s in stc):
        h = hota(stc)
        res["HOTA_court"] = h["HOTA"]
        res["AssA_court"] = h["AssA"]
        res["IDF1_court"] = idf1(stc)["IDF1"]
    res["dense_frames"] = len(frames)
    res["dense_gt"] = sum(len(s[0]) for s in st)
    return res


def ball_error(frames: list[dict], ball_at, project=None) -> dict:
    """Ball labels (dense ``ball``: center [u,v], optional ground [u,v] / court [x,y], state) vs the output ball.
    image error = px between the labelled centre and the output ball box centre (frames where both exist);
    court error = m between the labelled ground point (projected) / court and the output position;
    ball recall = labelled visible balls with an output position."""
    img, crt = [], []
    vis = shown = 0
    for fr in frames:
        b = fr.get("ball")
        if not b or b.get("state") == "hidden" or not b.get("center"):
            continue
        vis += 1
        o = ball_at(int(fr["frame"]))
        if o is None:
            continue
        shown += 1
        if o.get("box"):
            bx = o["box"]
            img.append(math.dist(b["center"], ((bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2)))
        g = b.get("court") or (project(fr, b["ground"]) if project and b.get("ground") else None)
        if g is not None:
            crt.append(math.dist(g, o["xy"]))
    q = lambda v, p: float(np.percentile(v, p)) if v else None  # noqa: E731
    return {"ball_labelled": vis, "ball_recall": shown / vis if vis else None, "ball_img_err_px": q(img, 50),
            "ball_img_err_p90_px": q(img, 90), "ball_court_err_m": q(crt, 50), "ball_court_err_p90_m": q(crt, 90)}


# ---------------------------------------------------------------------------------------------- extrapolation
def _windows(k: np.ndarray, L: int, margin: int, stride: int, max_gap: int = 3) -> list[tuple[int, int]]:
    """Hide windows [a, a+L) inside runs of (nearly) continuous samples k, with ``margin`` seen samples on both
    sides."""
    out = []
    if len(k) < 2:
        return out
    breaks = np.where(np.diff(k) > max_gap)[0]
    starts = np.r_[0, breaks + 1]
    ends = np.r_[breaks, len(k) - 1]
    for s, e in zip(starts, ends):
        a = k[s] + margin
        while a + L + margin <= k[e]:
            out.append((int(a), int(a + L)))
            a += L + stride
    return out


def player_extrapolation(tracks: dict, fps: float, smoothing_cfg=None, prior=None, max_windows: int = 60) -> dict:
    """Hide-and-predict on the output's own seen stretches. ``tracks``: {id: (k int array, xy (n,2))} of measured
    samples. For each length L (1/3/5/10 s) up to ``max_windows`` windows spread over all tracks:
    fill = both ends known (smoothing.smooth_track bridging the gap - how gaps are filled in the output), error over
    every hidden sample; extrap = only the past known (MotionPrior.predict if a prior is given, else constant
    velocity decaying over 1 s), error at the window's last sample (L s after the last seen one)."""
    from dataclasses import replace

    from replay import smoothing

    res: dict = {}
    for L_s in EXTRAP_S:
        L = max(1, int(round(L_s * fps)))
        margin = max(2, int(round(1.0 * fps)))
        cands = [(tid, a, b) for tid, (k, _) in sorted(tracks.items(), key=lambda x: str(x[0]))
                 for a, b in _windows(k, L, margin, int(round(2 * fps)))]
        if len(cands) > max_windows:  # deterministic, evenly spread over tracks and time
            cands = [cands[int(i)] for i in np.linspace(0, len(cands) - 1, max_windows)]
        fill_err, ext_err = [], []
        for tid, a, b in cands:
            k, xy = tracks[tid]
            hide = (k >= a) & (k < b)
            truth_k, truth = k[hide], xy[hide]
            if len(truth_k) < max(1, L // 2):
                continue
            keep = ~hide
            if smoothing_cfg is not None:
                cfg = replace(smoothing_cfg, max_gap_s=L_s + 1.0)
                if hasattr(cfg, "occluded_max_gap_s"):
                    cfg = replace(cfg, occluded_max_gap_s=0.0)
                lo, hi = a - 3 * margin, b + 3 * margin  # local stretch only (speed)
                loc = keep & (k >= lo) & (k < hi)
                R = np.repeat((np.eye(2) * 0.15 ** 2)[None], loc.sum(), 0)
                pred = {}
                for seg in smoothing.smooth_track(k[loc], xy[loc], R, fps, cfg):
                    for kk, p in zip(seg.k, seg.xy):
                        pred[int(kk)] = p
                e = [math.dist(pred[int(kk)], tt) for kk, tt in zip(truth_k, truth) if int(kk) in pred]
                if e:
                    fill_err.append(float(np.mean(e)))
            past = np.flatnonzero(k < a)
            if len(past) >= 3:
                i0 = past[-1]
                j = past[max(0, len(past) - 1 - int(round(0.5 * fps)))]
                dt0 = max((k[i0] - k[j]) / fps, 1e-3)
                v0 = (xy[i0] - xy[j]) / dt0
                i1 = int(truth_k[-1])
                if prior is not None:
                    p1, _ = prior.predict(int(k[i0]), xy[i0], v0, i1)
                else:
                    tau = 1.0
                    h = (i1 - k[i0]) / fps
                    p1 = xy[i0] + v0 * tau * (1 - math.exp(-h / tau))
                ext_err.append(math.dist(p1, truth[-1]))
        for name, v in (("fill", fill_err), ("extrap", ext_err)):
            res[f"{name}_{L_s}s_med"] = float(np.median(v)) if v else None
            res[f"{name}_{L_s}s_p90"] = float(np.percentile(v, 90)) if v else None
            res[f"{name}_{L_s}s_n"] = len(v)
    return res


def ball_hide_windows(has: np.ndarray, L: int, margin: int, stride: int) -> list[tuple[int, int]]:
    """Windows [a, a+L) where the ball was detected in >= 60 % of the samples (and on both margins)."""
    out, a, n = [], margin, len(has)
    while a + L + margin <= n:
        seg = has[a - margin:a + L + margin]
        if seg.mean() >= 0.6 and has[a - margin:a].any() and has[a + L:a + L + margin].any():
            out.append((a, a + L))
            a += L + stride
        else:
            a += max(1, L // 4)
    return out
