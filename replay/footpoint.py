"""Pass-2 step 2: robust foot points on the court + per-detection position uncertainty.

The court position of a person is the image foot point (box bottom centre) projected through the floor homography.
That assumption breaks in a few recurring ways; this module repairs what can be repaired and, for the rest, says
how unsure the position is (covariance in metres, used by the tracker, the smoother and the UI):

* airborne players (shots, rebounds, layups): the box bottom is above the floor, so the projected point slides
  AWAY from the camera along the viewing ray - with a low camera (~2 m) a 0.4 m jump moves the point by 25 % of its
  distance, several metres. Signature: the body's metric size (pixel height x metres per pixel at the foot) jumps
  up while the position makes a radial out-and-back excursion. Airborne samples are replaced by the
  takeoff -> landing line (takeoff position when no landing is seen) and get a larger uncertainty.
* single-sample box-bottom glitches (detector cuts the legs, merges with someone): the metric size drops/jumps
  for one or two samples with no matching motion; the foot is re-derived from the box top and the person's typical
  pixel height at that distance - but only where that is well conditioned (see ``_head_foot``).
* box cut by the image bottom / a score-bar overlay: same head-based estimate, kept only when its uncertainty
  is acceptable (with a camera at head height the head carries no depth information, so these stay dropped).
* uncertainty: pixel noise grows with box height and box quality (cut, glitch, airborne), mapped to metres with the
  projection Jacobian (1 px far away = many cm).

Per-identity statistics use the pass-1 ByteTrack id (step 3's court identities do not exist yet).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from replay import homography
from replay.config import FootConfig


@dataclass
class FootResult:
    xy: np.ndarray  # (n, 2) court metres (NaN = no usable foot)
    R: np.ndarray  # (n, 2, 2) covariance, m^2
    foot_px: np.ndarray  # (n, 2) image foot point actually used
    flag: np.ndarray  # (n,) int: 0 box bottom, 1 airborne (replaced), 2 glitch fixed, 3 recovered from head
    stats: dict


BOTTOM, AIRBORNE, GLITCH, FROM_HEAD = 0, 1, 2, 3


def _scales(Hs: np.ndarray, px: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Projection Jacobians (n, 2, 2) and lateral metres per pixel (n,) at image points."""
    J = homography.jacobian_each(Hs, px)
    s = np.linalg.norm(J[:, :, 0], axis=1)
    return J, np.where(np.isfinite(s) & (s > 0), s, np.nan)


def _rolling_median(v: np.ndarray, half: int) -> np.ndarray:
    """NaN-aware centred rolling median over the sample sequence."""
    n = len(v)
    if n == 0:
        return v.copy()
    pad = np.concatenate([np.full(half, np.nan), v, np.full(half, np.nan)])
    win = np.lib.stride_tricks.sliding_window_view(pad, 2 * half + 1)
    with np.errstate(all="ignore"), _quiet():
        return np.nanmedian(win, axis=1)


class _quiet:
    def __enter__(self):
        import warnings
        self._w = warnings.catch_warnings()
        self._w.__enter__()
        warnings.simplefilter("ignore", RuntimeWarning)

    def __exit__(self, *a):
        return self._w.__exit__(*a)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index runs where mask is True."""
    if not mask.any():
        return []
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def _head_foot(H: np.ndarray, top: np.ndarray, size_m: float, v_lo: float) -> tuple[np.ndarray, float] | None:
    """Foot pixel below the head ``top`` (x, y) for a person of metric size ``size_m`` (pixel height x lateral
    metres-per-pixel at the foot). Solved on the vertical line under the head; returns (foot pixel, d foot_v /
    d size_m) or None. The derivative tells how well conditioned it is (huge when the camera is at head height)."""

    span = max(v_lo - top[1], 4.0)
    v = top[1] + np.geomspace(2.0, 6.0 * span, 240)  # candidate foot rows, all at once
    _, s = _scales(np.repeat(H[None], len(v), axis=0), np.c_[np.full(len(v), top[0]), v])
    f = (v - top[1]) * s  # metric size implied by a foot at row v
    hit = np.flatnonzero(np.isfinite(f[:-1]) & np.isfinite(f[1:]) & (f[:-1] < size_m) & (f[1:] >= size_m))
    if not len(hit):
        return None
    i = hit[0]
    slope = (f[i + 1] - f[i]) / (v[i + 1] - v[i])  # metres of size per pixel of foot row
    if not np.isfinite(slope) or slope <= 1e-6:
        return None
    vf = v[i] + (size_m - f[i]) / slope
    return np.array([top[0], vf]), 1.0 / slope


def estimate_feet(boxes: np.ndarray, Hs: np.ndarray, sample: np.ndarray, track: np.ndarray, fps: float,
                  frame_size: tuple[int, int], cut: np.ndarray | None = None, occluded: np.ndarray | None = None,
                  cfg: FootConfig | None = None, motion: list | None = None) -> FootResult:
    """Court foot positions + covariances for person detections.

    boxes (n, 4) image boxes; Hs (n, 3, 3) image->court homography of each detection's sample; sample (n,) sample
    index; track (n,) pass-1 ByteTrack id (<= 0 = none); cut (n,) bool: bottom of the box is the image edge or an
    overlay (feet not visible); occluded (n,) bool: someone in front covers the bottom of the box; motion: pass-1
    camera motion per sample (motion[i] maps sample i-1 pixels -> sample i), used to compare box ends over time.
    """
    cfg = cfg or FootConfig()
    n = len(boxes)
    boxes = np.asarray(boxes, float).reshape(n, 4)
    Hs = np.asarray(Hs, float).reshape(n, 3, 3)
    cut = np.zeros(n, bool) if cut is None else np.asarray(cut, bool)
    occluded = np.zeros(n, bool) if occluded is None else np.asarray(occluded, bool)
    foot = np.c_[(boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]]
    top = np.c_[(boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 1]]
    h_px = np.maximum(1.0, boxes[:, 3] - boxes[:, 1])
    flag = np.zeros(n, int)
    stats = {"detections": n, "bottom_glitch": 0, "airborne": 0, "airborne_segments": 0, "glitch_fixed": 0, "cut": int(cut.sum()),
             "cut_recovered": 0}
    if n == 0:
        return FootResult(np.zeros((0, 2)), np.zeros((0, 2, 2)), np.zeros((0, 2)), flag, stats)

    xy = homography.project_each(Hs, foot[:, None, :])[:, 0]
    J, s = _scales(Hs, foot)
    size = h_px * s  # metric body size (~ constant per person on the floor)
    xy[cut] = np.nan  # no visible feet: only a head-based estimate may fill these in
    extra_var = np.zeros(n)  # isotropic extra variance (m^2)
    px_factor = np.where(occluded, cfg.occluded_px_factor, 1.0)

    if cfg.enabled:
        order = np.lexsort((sample, track))
        tr_sorted = track[order]
        bounds = np.flatnonzero(np.diff(tr_sorted)) + 1
        half = max(2, int(round(cfg.size_window_s * fps)))
        max_jump = max(1, int(round(cfg.jump_max_s * fps)))
        max_glitch = max(1, int(round(cfg.glitch_max_s * fps)))
        typical = np.full(n, np.nan)
        cam_xy = _camera_ground(Hs, sample, frame_size)
        groups = [g for g in np.split(order, bounds) if track[g[0]] > 0 and len(g) >= 5]
        if cfg.bottom_hampel and motion is not None:
            fixed = _bottom_glitches(groups, boxes, sample, motion, cut, cfg)
            if cfg.hampel_skip_occluded:  # guard: leave boxes covered by someone in front to tracklets
                fixed = {j: v for j, v in fixed.items() if not occluded[j]}
            for j, v in fixed.items():
                foot[j, 1] = v
                px_factor[j] = max(px_factor[j], cfg.glitch_px_factor)
                flag[j] = GLITCH
            if fixed:
                ids = np.fromiter(fixed, int)
                xy[ids] = homography.project_each(Hs[ids], foot[ids][:, None, :])[:, 0]
                J[ids], s[ids] = _scales(Hs[ids], foot[ids])
                size[ids] = (foot[ids, 1] - boxes[ids, 1]) * s[ids]
            stats["bottom_glitch"] = len(fixed)
        ratio = np.full(n, np.nan)
        for grp in groups:
            sz = np.where(cut[grp], np.nan, size[grp])
            typical[grp] = _rolling_median(sz, half)
            ratio[grp] = sz / typical[grp]
        # common mode: when EVERYONE in a frame looks bigger/smaller at once, the frame's calibration moved, not
        # the people (nobody jumps in sync) - divide it out
        common = _frame_median(ratio, sample, cfg.common_mode_min)
        ratio = ratio / common
        stats["frames_common_mode"] = int(np.sum(np.abs(np.unique(common[np.isfinite(ratio)]) - 1.0) > 0.05))
        for grp in groups:
            if cfg.airborne:
                _airborne(grp, ratio[grp], xy, J, cam_xy, sample, cfg, max_jump, flag, extra_var, stats,
                          occluded if cfg.airborne_skip_occluded else None)
            if cfg.fix_glitches:
                low = (ratio[grp] < cfg.glitch_ratio) & (flag[grp] == BOTTOM) & ~cut[grp] & ~occluded[grp]
                for a, b in _runs(low):
                    if b - a > max_glitch:
                        continue  # a long run is a real change (sitting down, track switched person)
                    for j in grp[a:b]:
                        if _fix_from_head(j, Hs, top, typical[j], cfg, xy, foot, extra_var):
                            flag[j] = GLITCH
                            stats["glitch_fixed"] += 1
        if cfg.recover_cut:
            pop = np.nanmedian(size[~cut]) if (~cut).any() else np.nan
            for j in np.flatnonzero(cut):
                ref = typical[j] if np.isfinite(typical[j]) else pop
                if top[j, 1] > 3 and _fix_from_head(j, Hs, top, ref, cfg, xy, foot, extra_var):
                    flag[j] = FROM_HEAD
                    stats["cut_recovered"] += 1

    # covariance: pixel noise through the Jacobian at the foot actually used
    Jf, _ = _scales(Hs, foot)
    s_px = np.maximum(cfg.min_px_sigma, cfg.foot_px_sigma_rel * h_px) * px_factor
    R = Jf @ np.swapaxes(Jf, 1, 2) * (s_px ** 2)[:, None, None]
    R += np.eye(2)[None] * (cfg.min_sigma_m ** 2 + extra_var)[:, None, None]
    bad = ~np.isfinite(R).all(axis=(1, 2)) | ~np.isfinite(xy).all(axis=1)
    xy[bad] = np.nan
    return FootResult(xy, R, foot, flag, stats)


def _airborne(grp, ratio, xy, J, cam_xy, sample, cfg, max_jump, flag, extra_var, stats, occluded=None) -> None:
    """Find out-and-back radial excursions with an enlarged metric size inside one ByteTrack track."""
    fin = np.isfinite(xy[grp]).all(axis=1) & np.isfinite(ratio)
    core = (ratio > cfg.jump_size_ratio) & fin
    # hysteresis: take-off and landing samples (feet just off the floor) belong to the jump as well
    loose = (ratio > 1.0 + cfg.jump_hysteresis * (cfg.jump_size_ratio - 1.0)) & fin
    for a, b in _runs(loose):
        if not core[a:b].any():
            continue
        if b - a > max_jump:
            continue
        # grounded anchors: the nearest samples before/after with a normal size
        ia = a - 1 if a > 0 else None
        ib = b if b < len(grp) else None
        if ia is not None and not np.isfinite(xy[grp[ia]]).all():
            ia = None
        if ib is not None and not np.isfinite(xy[grp[ib]]).all():
            ib = None
        if ia is None and ib is None:
            continue
        seg = grp[a:b]
        if occluded is not None and occluded[seg].any():
            continue  # bottom covered by someone in front: a short box, not a jump (tracklets handles those)
        ks = sample[seg].astype(float)
        if ia is not None and ib is not None:
            pa, pb = xy[grp[ia]], xy[grp[ib]]
            ka, kb = float(sample[grp[ia]]), float(sample[grp[ib]])
            if (kb - ka) > max_jump + 2:
                continue  # the anchors are too far apart in time (gap in the track)
            u = ((ks - ka) / max(kb - ka, 1e-9))[:, None]
            base = pa + u * (pb - pa)
        else:
            anchor = grp[ia] if ia is not None else grp[ib]
            if abs(float(sample[anchor]) - ks.mean()) > max_jump:
                continue
            base = np.repeat(xy[anchor][None], len(seg), axis=0)
        if np.isfinite(cam_xy[seg]).all():
            # physics check: feet at height z seen from a camera at height Hc land (q - 1) x distance further away,
            # q = Hc / (Hc - z) = the metric-size ratio. Arms up: size up, no displacement; running away: the
            # displacement is in the anchor line already.
            away = base - cam_xy[seg]
            dist = np.linalg.norm(away, axis=1)
            u_r = away / np.maximum(dist, 1e-9)[:, None]
            e_pred = (ratio[a:b] - 1.0) * dist
        else:  # no camera: the court direction of moving UP in the image
            up = -J[seg][:, :, 1]
            nrm = np.linalg.norm(up, axis=1)
            if not np.isfinite(nrm).all() or (nrm <= 0).any():
                continue
            u_r = up / nrm[:, None]
            e_pred = None
        d = xy[seg] - base
        e = np.einsum("ij,ij->i", d, u_r)
        perp = np.abs(d[:, 0] * u_r[:, 1] - d[:, 1] * u_r[:, 0])
        if np.nanmax(e) < cfg.jump_min_radial_m or np.nanmedian(e) <= 0:
            continue
        if np.median(perp) > 0.5 * np.median(e) + 0.3:
            continue  # sideways movement, not an excursion along the viewing ray
        if e_pred is not None:
            cons = np.median(e) / max(float(np.median(e_pred)), 1e-6)
            if not (cfg.jump_consistency[0] <= cons <= cfg.jump_consistency[1]):
                continue
        xy[seg] = base
        flag[seg] = AIRBORNE
        extra_var[seg] += cfg.airborne_sigma_m ** 2
        stats["airborne"] += len(seg)
        stats["airborne_segments"] += 1


def _neighbour_transforms(motion: list, n_samples: int) -> dict[int, np.ndarray]:
    """{d: (n_samples, 3, 3)} pixel transforms from sample i+d into sample i (NaN where the motion chain breaks)."""
    M = np.full((n_samples + 3, 3, 3), np.nan)  # M[m]: sample m-1 -> m
    Minv = np.full_like(M, np.nan)
    for m in range(min(len(motion), n_samples)):
        mm = motion[m]
        if mm is None:
            continue
        mm = np.asarray(mm, float)
        if not np.isfinite(mm).all():
            continue
        try:
            Minv[m] = np.linalg.inv(mm)
        except np.linalg.LinAlgError:
            continue
        M[m] = mm
    i = np.arange(n_samples)
    out = {
        -1: M[i],  # i-1 -> i
        -2: M[i] @ np.where(i[:, None, None] >= 1, M[np.maximum(i - 1, 0)], np.nan),  # i-2 -> i-1 -> i
        1: Minv[i + 1],  # i+1 -> i
        2: Minv[i + 1] @ Minv[i + 2],  # i+2 -> i+1 -> i
    }
    return out


def _bottom_glitches(groups, boxes, sample, motion, cut, cfg) -> dict[int, float]:
    """Hampel test on the box bottom against the track's neighbouring samples (camera motion removed).

    The bottom is replaced when it leaves the line through the neighbours' bottoms while the box TOP stays on its
    own line: legs cut off / box merged with something below for one or two samples. A jump moves both ends, arms
    up only the top, real running both ends smoothly - none of those is touched. Returns {detection: new foot row}.
    Vectorised over all detections (neighbour pairs tried in the order (1,1), (1,2), (2,1); the first pair whose
    two neighbours exist decides).
    """
    if not groups:
        return {}
    G = np.concatenate(groups)
    gid = np.concatenate([np.full(len(g), q) for q, g in enumerate(groups)])
    k = sample[G].astype(int)
    T = _neighbour_transforms(motion, int(sample.max()) + 1)
    m = len(G)
    nb = {}
    for o in (1, 2):  # neighbour detection o positions before/after in the same track, exactly o samples apart
        for sgn in (-1, 1):
            idx = np.arange(m) + sgn * o
            ok = (idx >= 0) & (idx < m)
            idx_c = np.clip(idx, 0, m - 1)
            ok &= (gid[idx_c] == gid) & (k[idx_c] - k == sgn * o)
            ok &= ~cut[G[idx_c]]
            nb[sgn * o] = np.where(ok, G[idx_c], -1)

    def mapped(d: int) -> np.ndarray:
        """(m, 2, 2): [top, bottom] centre points of neighbour d mapped into each detection's frame."""
        j = nb[d]
        jj = np.maximum(j, 0)
        cx = (boxes[jj, 0] + boxes[jj, 2]) / 2
        pts = np.stack([np.c_[cx, boxes[jj, 1], np.ones(m)], np.c_[cx, boxes[jj, 3], np.ones(m)]], axis=1)
        Tm = T[d][k]
        q = np.einsum("nij,npj->npi", Tm, pts)
        with np.errstate(all="ignore"):
            r = q[..., :2] / q[..., 2:3]
        r[j < 0] = np.nan
        return r

    P = {d: mapped(d) for d in (-2, -1, 1, 2)}
    h = boxes[G, 3] - boxes[G, 1]
    thr = np.maximum(cfg.hampel_min_px, cfg.hampel_rel * h)
    decided = cut[G].copy()
    out: dict[int, float] = {}
    for da, db in ((1, 1), (1, 2), (2, 1)):
        ea, eb = P[-da], P[db]
        avail = ~decided & np.isfinite(ea).all(axis=(1, 2)) & np.isfinite(eb).all(axis=(1, 2))
        if not avail.any():
            continue
        u = da / (da + db)
        pred = ea + u * (eb - ea)
        dev_b = boxes[G, 3] - pred[:, 1, 1]
        dev_t = boxes[G, 1] - pred[:, 0, 1]
        ha = np.maximum(ea[:, 1, 1] - ea[:, 0, 1], 1.0)
        hb = np.maximum(eb[:, 1, 1] - eb[:, 0, 1], 1.0)
        with np.errstate(all="ignore"):
            agree = np.abs(np.log(ha / hb)) <= cfg.hampel_rel * 2
        hit = avail & agree & (np.abs(dev_b) > thr) & (np.abs(dev_t) < 0.5 * np.abs(dev_b)) & (np.abs(dev_t) < thr)
        for q in np.flatnonzero(hit):
            out[int(G[q])] = float(pred[q, 1, 1])
        decided |= avail
    return out


def _frame_median(v: np.ndarray, sample: np.ndarray, min_n: int) -> np.ndarray:
    """Per-detection median of v over its sample (1.0 where fewer than min_n finite values)."""
    out = np.ones(len(v))
    fin = np.isfinite(v)
    if not fin.any():
        return out
    idx = np.flatnonzero(fin)
    o = idx[np.argsort(sample[idx], kind="stable")]
    b = np.flatnonzero(np.diff(sample[o])) + 1
    keys, meds = [], []
    for g in np.split(o, b):
        if len(g) >= min_n:
            keys.append(sample[g[0]])
            meds.append(float(np.median(v[g])))
    if keys:
        keys, meds = np.array(keys), np.array(meds)
        pos = np.clip(np.searchsorted(keys, sample), 0, len(keys) - 1)
        hit = keys[pos] == sample
        out[hit] = meds[pos[hit]]
    return out


def _camera_ground(Hs: np.ndarray, sample: np.ndarray, frame_size: tuple[int, int]) -> np.ndarray:
    """Court xy under the camera for each detection (NaN when the homography gives no plausible camera)."""
    uniq, first, inv = np.unique(sample, return_index=True, return_inverse=True)
    cams = np.full((len(uniq), 2), np.nan)
    for u, f in enumerate(first):
        cam = homography.camera_from_homography(Hs[f], frame_size)
        if cam is not None and 0.5 < cam["C"][2] < 60:
            cams[u] = cam["C"][:2]
    return cams[inv.reshape(-1)]


def _fix_from_head(j, Hs, top, size_ref, cfg, xy, foot, extra_var) -> bool:
    if not np.isfinite(size_ref) or size_ref <= 0:
        return False
    sol = _head_foot(Hs[j], top[j], float(size_ref), foot[j, 1])
    if sol is None:
        return False
    fpx, dv_dsize = sol
    sig_v = abs(dv_dsize) * cfg.head_size_sigma_rel * size_ref  # foot-row error from the size spread (pixels)
    p = homography.project(Hs[j], fpx[None])[0]
    if not np.isfinite(p).all():
        return False
    Jv = homography.jacobian(Hs[j], tuple(fpx))[:, 1]
    sig_m = float(np.linalg.norm(Jv) * sig_v)
    if not np.isfinite(sig_m) or sig_m > cfg.max_head_sigma_m:
        return False
    xy[j] = p
    foot[j] = fpx
    extra_var[j] += sig_m ** 2
    return True


def _feet_hidden(b: np.ndarray, sample: np.ndarray) -> np.ndarray:
    """Per box: another box in the same sample overlaps it horizontally and reaches lower in the image while
    starting above its bottom (someone in front covers the feet). Same rule as tracklets._Data.feet_hidden."""
    out = np.zeros(len(b), bool)
    order = np.argsort(sample, kind="stable")
    for grp in np.split(order, np.flatnonzero(np.diff(sample[order])) + 1):
        if len(grp) < 2:
            continue
        g = b[grp]
        ix = np.clip(np.minimum(g[:, None, 2], g[None, :, 2]) - np.maximum(g[:, None, 0], g[None, :, 0]), 0, None)
        covers = (ix > 0.3 * (g[:, 2] - g[:, 0])[:, None]) & (g[None, :, 3] > g[:, None, 3] + 2)             & (g[None, :, 1] < g[:, None, 3])
        np.fill_diagonal(covers, False)
        out[grp] = covers.any(axis=1)
    return out


def _plain_feet(b: np.ndarray, Hs: np.ndarray, fcfg: FootConfig) -> tuple[np.ndarray, np.ndarray]:
    """Box bottom centre through H + Jacobian covariance (the original step-2 projection)."""
    foot = np.c_[(b[:, 0] + b[:, 2]) / 2, b[:, 3]]
    p = homography.project_each(Hs, foot[:, None, :])[:, 0]
    J = homography.jacobian_each(Hs, foot)
    sigma_px = np.maximum(1.0, fcfg.foot_px_sigma_rel * (b[:, 3] - b[:, 1]))
    R = J @ np.swapaxes(J, 1, 2) * (sigma_px ** 2)[:, None, None] + np.eye(2) * fcfg.min_sigma_m ** 2
    return p, R


def project_people(rows: np.ndarray, det_row: np.ndarray, det_i: np.ndarray, H: list, size: tuple[int, int],
                   overlay: np.ndarray, cfg, fps: float, motion: list | None = None):
    """postprocess step 2: person rows -> (det_row, det_i, court xy, covariance, foot flag), detections whose foot
    is unusable or far off court removed. ``cfg`` is the pass-2 Config; ``cfg.foot`` (FootConfig) if present."""
    from replay import court

    fcfg = getattr(cfg, "foot", None) or FootConfig()
    if os.environ.get("AA_FOOT") in ("0", "1"):  # experiment switch (A/B runs): 0 = plain box bottom, 1 = robust
        fcfg = FootConfig(**{**fcfg.__dict__, "enabled": os.environ["AA_FOOT"] == "1"})
    parts = os.environ.get("AA_FOOT_PARTS")  # ablation (A/B runs): comma list of airborne,hampel,glitch
    if parts is not None:
        on = set(parts.split(","))
        fcfg = FootConfig(**{**fcfg.__dict__, "airborne": "airborne" in on, "bottom_hampel": "hampel" in on,
                             "fix_glitches": "glitch" in on, "hampel_skip_occluded": "hso" in on})
    fcfg = FootConfig(**{**fcfg.__dict__, "foot_px_sigma_rel": cfg.smoothing.foot_px_sigma_rel,
                         "min_sigma_m": cfg.smoothing.min_meas_sigma_m})
    ok = np.array([H[i] is not None for i in det_i], bool)
    b = rows[det_row, 1:5]
    cut = np.zeros(len(det_row), bool)
    if cfg.filter.drop_cut_feet:  # boxes touching the bottom edge / an overlay have no visible feet
        cut |= b[:, 3] >= size[1] - 3
        for ox1, oy1, ox2, oy2 in overlay:
            cut |= (b[:, 2] > ox1) & (b[:, 0] < ox2) & (oy1 - 4 <= b[:, 3]) & (b[:, 3] <= oy2)
    if not fcfg.enabled or fcfg.keep_old_membership:
        ok &= ~cut
    det_row, det_i, cut = det_row[ok], det_i[ok], cut[ok]
    b = rows[det_row, 1:5]
    Hs = np.array([H[i] for i in det_i], float).reshape(-1, 3, 3)
    p0, R0 = _plain_feet(b, Hs, fcfg)  # original behaviour: plain box bottom centre
    ok = np.isfinite(p0).all(axis=1) & court.in_court(np.nan_to_num(p0, nan=1e6), cfg.filter.court_margin_m)
    ok &= np.isfinite(R0).all(axis=(1, 2))
    p, R, flag = p0, R0, np.zeros(len(b), int)
    if fcfg.enabled:
        occ = _feet_hidden(b, det_i)
        res = estimate_feet(b, Hs, det_i, rows[det_row, 7].astype(int), fps, size, cut=cut, cfg=fcfg, motion=motion,
                            occluded=occ)
        print("foot points:", {k: v for k, v in res.stats.items() if v})
        good = np.isfinite(res.xy).all(axis=1) & np.isfinite(res.R).all(axis=(1, 2))
        if fcfg.max_shift_m > 0:  # guard: a repair that moves the point further than this is more likely wrong
            good &= ~(np.linalg.norm(np.nan_to_num(res.xy - p0, nan=0.0), axis=1) > fcfg.max_shift_m)
        if fcfg.keep_old_membership:
            # positions only: exactly the detections the plain projection keeps (a "repaired" foot of a box that
            # projected far off court is mostly a bench person / coach half hidden behind a player - rescuing them
            # added phantoms and pushed real players out of the per-frame caps); where the robust estimate has no
            # value the plain one is kept
            ok &= ~cut
            p, R, flag = p0.copy(), R0.copy(), np.zeros(len(b), int)
            p[good], R[good], flag[good] = res.xy[good], res.R[good], res.flag[good]
        else:
            p, R, flag = res.xy, res.R, res.flag
            ok = good & court.in_court(np.nan_to_num(p, nan=1e6), cfg.filter.court_margin_m)
    jit = os.environ.get("AA_FOOT_JITTER")  # experiment only: "sigma_m,seed" - benchmark noise floor
    if jit:
        sig, seed = jit.split(",")
        p = p + np.random.default_rng(int(seed)).normal(0.0, float(sig), p.shape)
    return det_row[ok], det_i[ok], p[ok].reshape(-1, 2), R[ok].reshape(-1, 2, 2), flag[ok]
