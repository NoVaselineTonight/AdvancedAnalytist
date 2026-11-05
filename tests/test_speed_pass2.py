"""Pass-2 speed-ups must not change a single bit: roster tensors on survivors only, roster/smoothing process pools."""

from __future__ import annotations

import numpy as np

from replay import lineup, roster
from replay.config import FilterConfig, SmoothingConfig
from tests.test_roster import FPS, SIZE, _scene


def _old_transforms(self, f, J):  # roster._Scene.transforms before 05.11 (full-tensor version), the reference
    T = self.Ginv[J] @ self.Gs[f]
    cfg = self.cfg
    src = self._sources(f)
    if src is None or not len(J):
        return T
    src, head, cs, b = src
    p, w = roster._map(T, src)
    ph, _ = roster._map(T, head)
    hm = np.linalg.norm(p - ph, axis=2)
    cap = 20.0 + cfg.realign_drift_rel * np.abs(self.path[J] - self.path[f])
    fJ = self.foot[J]
    ox = fJ[:, None, :, 0] - p[:, :, None, 0]
    oy = fJ[:, None, :, 1] - p[:, :, None, 1]
    with np.errstate(invalid="ignore", divide="ignore"):
        jj, si, ti = np.nonzero((np.maximum(np.abs(ox), np.abs(oy)) <= cap[:, None, None]) & (w[:, :, None] > 0))
        ratio = self.h[J[jj], ti] / hm[jj, si]
        dc = self.col[J[jj], ti] - cs[si]
        dcol = np.sqrt(dc[:, 0] * dc[:, 0] + dc[:, 1] * dc[:, 1] + dc[:, 2] * dc[:, 2])
        valid = (ratio > 0.7) & (ratio < 1.45) & ~(dcol > cfg.realign_max_color)
    jj, si, ti = jj[valid], si[valid], ti[valid]
    if not len(jj):
        return T
    o = np.stack([ox[jj, si, ti], oy[jj, si, ti]], axis=1)
    K = int(np.ceil(cap.max() / b)) + 1
    gx = np.clip(np.round(o[:, 0] / b).astype(int) + K, 0, 2 * K)
    gy = np.clip(np.round(o[:, 1] / b).astype(int) + K, 0, 2 * K)
    hist = np.zeros((len(J), 2 * K + 3, 2 * K + 3))
    np.add.at(hist, (jj, gx + 1, gy + 1), 1.0)
    sm = sum(hist[:, 1 + dx:2 * K + 2 + dx, 1 + dy:2 * K + 2 + dy] for dx in (-1, 0, 1) for dy in (-1, 0, 1))
    sm[:, K, K] += 0.5
    flat = sm.reshape(len(J), -1).argmax(axis=1)
    peak = sm.reshape(len(J), -1)[np.arange(len(J)), flat]
    cx, cy = (flat // (2 * K + 1) - K) * b, (flat % (2 * K + 1) - K) * b
    fix = (peak >= cfg.realign_min_support) & ~((cx == 0) & (cy == 0) & (peak < cfg.realign_min_support + 0.5))
    if not fix.any():
        return T
    near = fix[jj] & (np.maximum(np.abs(o[:, 0] - cx[jj]), np.abs(o[:, 1] - cy[jj])) <= 1.5 * b)
    cnt = np.bincount(jj[near], minlength=len(J))
    sx = np.bincount(jj[near], weights=o[near, 0], minlength=len(J))
    sy = np.bincount(jj[near], weights=o[near, 1], minlength=len(J))
    with np.errstate(invalid="ignore", divide="ignore"):
        dx = np.where(cnt > 0, sx / cnt, cx)
        dy = np.where(cnt > 0, sy / cnt, cy)
    A = np.flatnonzero(fix)
    S = np.repeat(np.eye(3)[None], len(A), axis=0)
    S[:, 0, 2], S[:, 1, 2] = dx[A], dy[A]
    T[A] = S @ T[A]
    return T


def _old_hit(scene, fj, hgt, J, qs, q_col):  # the old full-tensor `hit` of roster.occupancy
    cfg = scene.cfg
    with np.errstate(invalid="ignore"):
        fJ = scene.foot[J]
        dx = fj[:, :, None, 0] - fJ[:, None, :, 0]
        dy = fj[:, :, None, 1] - fJ[:, None, :, 1]
        d = np.sqrt(dx * dx + dy * dy)
        ratio = scene.h[J][:, None, :] / hgt[..., None]
        same = (d < cfg.static_radius_rel * hgt[..., None]) & (ratio > 1 / cfg.static_height_ratio) \
            & (ratio < cfg.static_height_ratio)
        if q_col is not None and cfg.static_color_veto:
            a, qi, ti = np.nonzero(same)
            dc = scene.col[J[a], ti] - q_col[qs[qi]]
            dcol = np.sqrt(dc[:, 0] * dc[:, 0] + dc[:, 1] * dc[:, 1] + dc[:, 2] * dc[:, 2])
            same[a, qi, ti] = ~(dcol > cfg.realign_max_color)
        return same.any(axis=2)


def _scene_obj(pan, seed, cfg=None):
    i, box, track, kind, G, seg, n, feat = _scene(seconds=30.0, pan=pan, seed=seed)
    sc = roster._Scene(i, box, G, seg, n, SIZE, cfg or FilterConfig(), feat)
    return sc, i, box, track, feat


def test_transforms_on_survivors_equal_full_tensor_version():
    for pan, seed in ((0.0, 0), (150.0, 1), (300.0, 3)):
        sc, *_ = _scene_obj(pan, seed)
        for f in range(0, sc.n, 7):
            J = np.arange(max(0, f - 120), min(sc.n, f + 121), 3)
            new = sc.transforms(f, J)
            old = _old_transforms(sc, f, J)
            assert new.tobytes() == old.tobytes()


def test_occupancy_hit_equals_full_tensor_version():
    for pan, seed in ((0.0, 0), (200.0, 2)):
        sc, i, box, track, feat = _scene_obj(pan, seed)
        occ, vis = roster.occupancy(i, box, sc, FPS, feat[:, :3])
        # reference: occupancy rebuilt from the old hit
        cfg, n, seg = sc.cfg, sc.n, sc.seg
        step = max(1, int(round(cfg.static_sample_s * FPS)))
        half = int(round(cfg.static_window_s * FPS))
        gap = int(round(cfg.static_min_dt_s * FPS))
        W, H = sc.size
        ref = np.full(len(i), np.nan)
        by = roster._by_frame(i, n)
        for f in range(n):
            qs = by[f]
            if not len(qs):
                continue
            J = np.arange(max(0, f - half), min(n, f + half + 1), step)
            J = J[(np.abs(J - f) >= gap) & (seg[J] == seg[f]) & (np.abs(sc.path[J] - sc.path[f]) <= cfg.static_max_travel_px)]
            if not len(J):
                continue
            b = box[qs]
            foot = roster._feet(b)
            head = foot - np.stack([np.zeros(len(b)), b[:, 3] - b[:, 1]], axis=1)
            T = sc.transforms(f, J)
            fj, w1 = roster._map(T, foot)
            hj, w2 = roster._map(T, head)
            hgt = np.linalg.norm(fj - hj, axis=2)
            with np.errstate(invalid="ignore"):
                visible = ((w1 > 0) & (w2 > 0) & (fj[..., 0] > 8.0) & (fj[..., 0] < W - 8.0)
                           & (fj[..., 1] > 8.0) & (fj[..., 1] < H - 8.0) & np.isfinite(hgt))
            hit = _old_hit(sc, fj, hgt, J, qs, feat[:, :3])
            nv = visible.sum(axis=0)
            ok = nv * step / FPS >= cfg.static_min_visible_s
            ref[qs[ok]] = (hit & visible).sum(axis=0)[ok] / nv[ok]
        assert occ.tobytes() == ref.tobytes()


def test_roster_pool_equals_serial(monkeypatch):
    sc, i, box, track, feat = _scene_obj(120.0, 4)
    monkeypatch.setenv("REPLAY_PARALLEL", "0")
    serial = roster._heavy(i, box, track, sc, FPS, feat[:, :3])
    monkeypatch.setenv("REPLAY_PARALLEL", "1")
    monkeypatch.setenv("REPLAY_POOL_WORKERS", "2")
    pooled = roster._heavy(i, box, track, sc, FPS, feat[:, :3])
    for a, b in zip(serial, pooled):
        assert a.tobytes() == b.tobytes()


def test_pool_workers_env_cap(monkeypatch):
    monkeypatch.setenv("REPLAY_POOL_WORKERS", "1")
    assert roster.pool_workers() == 1
    monkeypatch.delenv("REPLAY_POOL_WORKERS")
    assert 1 <= roster.pool_workers() <= 4


def _tracks(seed=0, n=12):
    rng = np.random.default_rng(seed)
    jobs = []
    for t in range(1, n + 1):
        k = np.sort(rng.choice(np.arange(600), size=rng.integers(5, 300), replace=False))
        z = np.cumsum(rng.normal(0, 0.1, (len(k), 2)), axis=0) + rng.uniform(0, 20, 2)
        R = np.tile(np.eye(2) * 0.05, (len(k), 1, 1))
        jobs.append((t, k, z, R))
    return jobs


def test_smoothing_pool_equals_serial(monkeypatch):
    jobs = _tracks()
    monkeypatch.setattr(lineup, "BREAKS", {3: np.array([200]), 5: np.array([100, 400])})
    cfg = SmoothingConfig()
    monkeypatch.setenv("REPLAY_PARALLEL", "0")
    serial = lineup.smooth_slots(jobs, FPS, cfg, n_frames=600)
    assert len(serial) == len(jobs)
    monkeypatch.setenv("REPLAY_PARALLEL", "1")
    monkeypatch.setenv("REPLAY_POOL_WORKERS", "2")
    pooled = lineup.smooth_slots(jobs, FPS, cfg, n_frames=600)
    assert len(pooled) == len(serial)
    for a, b in zip(serial, pooled):
        assert len(a) == len(b)
        for s, u in zip(a, b):
            for f in s.__dataclass_fields__:
                x, y = getattr(s, f), getattr(u, f)
                assert (x is None and y is None) or np.ascontiguousarray(x).tobytes() == np.ascontiguousarray(y).tobytes()
    # serial smooth_slots == the plain per-track smooth_slot calls postprocess made before
    for (t, k, z, R), segs in zip(jobs, serial):
        ref = lineup.smooth_slot(t, k, z, R, FPS, cfg)
        assert [s.xy.tobytes() for s in ref] == [s.xy.tobytes() for s in segs]
