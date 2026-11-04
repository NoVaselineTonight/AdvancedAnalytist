"""Player vs bench/coach/spectator filtering (replay/roster.py) on synthetic scenes."""

from __future__ import annotations

import numpy as np

from replay import court
from replay.config import FilterConfig
from replay.export import TrackRow
from replay.roster import (cap_rank, classify_roster, distance_inside_court, distance_off_court,
                           limit_shown_rows)

FPS = 10.0
SIZE = (1280, 720)
HEAD_Y = 250.0  # camera at head height: standing people's heads share this image row


def _standing(x: float, y_foot: float) -> list[float]:
    h = y_foot - HEAD_Y
    return [x - h / 5.6, y_foot - h, x + h / 5.6, y_foot]


def _seated(x: float, y_foot: float) -> list[float]:
    h = 0.6 * (y_foot - HEAD_Y)
    return [x - h / 2.8, y_foot - h, x + h / 2.8, y_foot]


def _scene(seconds: float = 60.0, pan: float = 0.0, n_players: int = 8, n_bench: int = 6, sub_at: float | None = None,
           seed: int = 0):
    """World x (pixels at zoom 1); camera x offset c_k; image x = world x - c_k. Returns detections + motions.

    kinds: 0 player, 1 bench (seated in a row along the far wall), 2 the substitute (bench, then player),
    3 a coach standing still next to the bench."""
    rng = np.random.default_rng(seed)
    n = int(seconds * FPS)
    cam = pan * np.sin(2 * np.pi * np.arange(n) / (30 * FPS))
    pos = np.column_stack([rng.uniform(200, 1100, n_players), rng.uniform(420, 620, n_players)])
    vel = rng.normal(0, 150, (n_players, 2))  # px/s; ~0.6-2 body heights per second like real players
    bench_x = 120 + 55 * np.arange(n_bench)
    sub_pos = np.array([120 + 55.0 * n_bench, 370.0])
    sub_vel = np.array([25.0, 20.0])
    i, box, track, kind, col = [], [], [], [], []
    team_a, team_b, black = [50.0, 55.0, 50.0], [35.0, 40.0, -45.0], [15.0, 0.0, 0.0]  # torso Lab
    for k in range(n):
        vel += rng.normal(0, 300, vel.shape) / FPS
        vel = np.clip(vel, -300, 300)
        pos += vel / FPS
        for d in (0, 1):  # bounce inside the court area
            lo, hi = ((100, 1180), (410, 680))[d]
            out = (pos[:, d] < lo) | (pos[:, d] > hi)
            vel[out, d] *= -1
            pos[:, d] = np.clip(pos[:, d], lo, hi)
        for p in range(n_players):
            i.append(k); box.append(_standing(pos[p, 0] - cam[k], pos[p, 1])); track.append(1 + p); kind.append(0)
            col.append(team_a if p % 2 else team_b)
        for b, x in enumerate(bench_x):
            jitter = rng.normal(0, 1.0, 2)
            i.append(k); box.append(_seated(x - cam[k] + jitter[0], 370 + jitter[1])); track.append(100 + b)
            kind.append(1); col.append(team_a)  # bench players wear the team's jerseys
        if n_bench:
            i.append(k); box.append(_standing(60 - cam[k] + rng.normal(0, 1.5), 372)); track.append(300); kind.append(3)
            col.append(black)
        if sub_at is not None:
            if k / FPS >= sub_at:  # gets up and plays
                sub_pos += sub_vel / FPS
                if not 150 < sub_pos[0] < 1150:
                    sub_vel[0] *= -1
                if not 380 < sub_pos[1] < 640:
                    sub_vel[1] *= -1
                b = _standing(sub_pos[0] - cam[k], sub_pos[1])
            else:
                b = _seated(sub_pos[0] - cam[k], sub_pos[1])
            i.append(k); box.append(b); track.append(200); kind.append(2); col.append(team_a)
    motions = [None] + [np.array([[1, 0, -(cam[k] - cam[k - 1])], [0, 1, 0], [0, 0, 1.0]]) for k in range(1, n)]
    box = np.array(box)
    vis = (box[:, 0] > 0) & (box[:, 2] < SIZE[0])  # what the camera sees
    G = [np.eye(3)]
    for k in range(1, n):
        G.append(G[-1] @ np.linalg.inv(motions[k]))
    col = np.array(col) + rng.normal(0, 3, (len(col), 3))
    feat = np.concatenate([col, col], axis=1)
    return (np.array(i)[vis], box[vis], np.array(track)[vis], np.array(kind)[vis], G, np.zeros(n, int), n, feat[vis])


def _run(scene, cfg: FilterConfig | None = None, **kw):
    i, box, track, kind, G, seg, n, feat = scene
    return classify_roster(i, box, track, FPS, SIZE, cfg or FilterConfig(), n_frames=n, G=G, seg=seg, feat=feat, **kw)


def test_bench_dropped_players_kept_fixed_camera():
    sc = _scene(pan=0.0)
    r = _run(sc)
    kind = sc[3]
    assert r.keep[kind == 0].mean() > 0.95
    assert r.keep[kind == 1].mean() < 0.05
    assert r.keep[kind == 3].mean() < 0.05  # standing still by the bench for a minute
    assert r.dropped  # bench tracks reported


def test_bench_dropped_with_panning_camera():
    sc = _scene(pan=250.0, seed=1)
    r = _run(sc)
    kind = sc[3]
    assert r.keep[kind == 0].mean() > 0.9
    assert r.keep[kind == 1].mean() < 0.1
    assert np.nanmean(r.raw["static"][kind == 1]) > 0.8  # spot occupancy survives the pan


def test_substitute_decided_per_time_window():
    sc = _scene(seconds=80.0, sub_at=40.0, seed=2)
    r = _run(sc)
    i, kind = sc[0], sc[3]
    sub = kind == 2
    early = sub & (i < 15 * FPS)  # decisions blur over ~+-15 s around the change (sliding windows)
    late = sub & (i > 55 * FPS)
    assert r.keep[early].mean() < 0.1
    assert r.keep[late].mean() > 0.9


def test_per_frame_cap():
    sc = _scene(n_players=16, n_bench=0, seed=3)
    cfg = FilterConfig(max_on_court=13)
    r = _run(sc, cfg)
    counts = np.bincount(sc[0][r.keep], minlength=sc[6])
    assert counts.max() <= 13
    assert counts.mean() > 12


def test_court_position_evidence_and_candidates():
    sc = _scene(n_bench=3, seed=4)
    i, box, track, kind = sc[:4]
    xy = np.where(kind[:, None] == 0, [[10.0, 7.0]], [[12.0, 15.0 + 4.0]])  # bench 4 m behind the far sideline
    cand = np.ones(len(i), bool)
    cand[::7] = False
    r = _run(sc, court_xy=xy, candidate=cand)
    assert not r.keep[~cand].any()
    assert r.keep[(kind == 0) & cand].mean() > 0.95
    assert r.raw["off_court_m"][kind == 1].min() > 3.9


def test_distance_off_court_uses_current_court_size():
    pts = np.array([[1.0, 1.0], [-2.0, 5.0], [10.0, 17.0], [np.nan, 1.0]])
    d = distance_off_court(pts)
    assert d[0] == 0 and abs(d[1] - 2.0) < 1e-9 and abs(d[2] - (17.0 - court.WIDTH)) < 1e-9
    assert np.isnan(d[3])
    if hasattr(court, "set_dimensions"):
        old = (court.LENGTH, court.WIDTH)
        try:
            court.set_dimensions(26.0, 14.0)
            assert abs(distance_off_court(np.array([[27.0, 5.0]]))[0] - 1.0) < 1e-9
        finally:
            court.set_dimensions(*old)


def test_empty_input():
    r = classify_roster(np.zeros(0, int), np.zeros((0, 4)), np.zeros(0, int), FPS, SIZE, FilterConfig(),
                        n_frames=5)
    assert r.keep.shape == (0,) and r.dropped == {}


def test_distance_inside_court():
    pts = np.array([[1.0, 5.0], [court.LENGTH / 2, court.WIDTH / 2], [-1.0, 5.0], [np.nan, 1.0]])
    d = distance_inside_court(pts)
    assert abs(d[0] - 1.0) < 1e-9 and abs(d[1] - court.WIDTH / 2) < 1e-9 and d[2] == 0 and np.isnan(d[3])


def test_cap_rank_prefers_people_deep_inside():
    cfg = FilterConfig()
    r = cap_rank(np.array([2.0, 3.0, 2.0]), np.array([3.0, 0.1, np.nan]), cfg)
    assert r[0] > r[1] > r[2]  # player at the free-throw line beats a sideline track with a better score
    assert abs(r[2] - 2.0) < 1e-9  # unknown position: no bonus


def test_still_person_kept_deep_inside_dropped_at_sideline():
    """Someone standing still for a minute is bench/staff at the sideline, but a player (free throw, dead ball)
    when deep inside the court."""
    sc = _scene(n_bench=3, seed=5)
    kind = sc[3]
    still = kind == 3  # stands still next to the bench the whole time
    for spot, expect in (([13.0, 7.0], True), ([13.0, court.WIDTH - 0.3], False)):
        xy = np.where(kind[:, None] == 0, [[10.0, 7.0]], [[12.0, court.WIDTH + 1.5]])
        xy[still] = spot
        r = _run(sc, court_xy=xy)
        assert (r.keep[still].mean() > 0.9) == expect, (spot, r.keep[still].mean())
        assert r.keep[kind == 0].mean() > 0.95
        assert r.keep[kind == 1].mean() < 0.05


def _row(k, tid, team="A", interp=False):
    return TrackRow(frame=k, t=k / 10, k=k, track_id=tid, object_type="player", team=team, x=1.0, y=1.0,
                    confidence=0.9, interpolated=interp, calib="keyframe")


def test_limit_shown_rows_prefers_detections_then_rank():
    caps = ((("A", "B"), 10), (("referee",), 3))
    rows = [_row(0, t, "A" if t % 2 else "B") for t in range(1, 10)]  # 9 detections, short tracks
    rows += [_row(0, t, interp=True) for t in (20, 21, 22)]  # 3 filled-in gaps of long tracks
    rows += [_row(0, 30, "referee")]
    rank = {id(r): float(r.track_id) for r in rows if not r.interpolated}
    strength = {r.track_id: (1000.0 if r.interpolated else 1.0) for r in rows}
    drop = limit_shown_rows(rows, caps, rank, strength)
    kept = [r for r in rows if id(r) not in drop]
    assert sum(r.team in ("A", "B") for r in kept) == 10
    assert len(drop) == 2 and all(r.interpolated for r in rows if id(r) in drop)  # gap fills go first
    # too many detections: the lowest rank goes
    rows2 = [_row(1, t) for t in range(1, 13)]
    rank2 = {id(r): float(r.track_id) for r in rows2}
    drop2 = limit_shown_rows(rows2, caps, rank2, {r.track_id: 1.0 for r in rows2})
    assert sorted(r.track_id for r in rows2 if id(r) in drop2) == [1, 2]


def test_bench_crowd_counts_seated_neighbours_at_the_same_depth():
    from replay.roster import bench_crowd

    # frame 0: a standing person (0) beside two seated people (1, 2) at the same foot row, a player far away (3)
    box = np.array([[100, 100, 140, 220], [160, 150, 200, 220], [210, 150, 250, 222], [600, 300, 640, 420]], float)
    i = np.zeros(4, int)
    occ = np.array([0.1, 0.9, 0.95, 0.1])
    posture = np.array([0.3, -0.8, -0.8, 0.3])
    c = bench_crowd(i, box, occ, posture, 1)
    assert c[0] == 2 and c[3] == 0


def test_calibration_plausible_flags_wrong_scale():
    from replay.roster import calibration_plausible

    # 100 px tall people; ground scale 0.017 m/px -> 1.7 m (plausible), 0.005 m/px -> 0.5 m (calibration wrong)
    good, bad = np.diag([0.017, 0.017, 1.0]), np.diag([0.005, 0.005, 1.0])
    i = np.arange(20)
    box = np.tile([100.0, 100.0, 140.0, 200.0], (20, 1))
    ok = calibration_plausible(i, box, [good] * 10 + [bad] * 10, 20, 1.0)
    assert ok[:8].all() and not ok[12:].any()
    assert calibration_plausible(i, box, None, 20, 1.0).all()


def test_short_track_continuing_a_player_is_kept():
    from replay.config import FilterConfig
    from replay.roster import short_but_continuing
    fps = 10.0
    i = np.r_[np.arange(0, 30), np.arange(40, 43), np.arange(50, 80)]  # long id 1, 3-sample id 2, long id 3
    track = np.r_[np.full(30, 1), np.full(3, 2), np.full(30, 3)]
    xy = np.c_[np.r_[np.linspace(5, 8, 30), [8.5, 8.6, 8.7], np.linspace(9, 12, 30)], np.full(63, 7.0)]
    feat = np.tile([40, 30, -20, 50, 0, 0], (63, 1)).astype(float)
    keep = short_but_continuing(track, i, xy, feat, fps, FilterConfig(short_keep_window_s=3.0))
    assert keep[30:33].all() and not keep[:30].any()
    far = xy.copy()
    far[30:33, 0] = 25.0  # unreachable: 17 m in 1 s
    assert not short_but_continuing(track, i, far, feat, fps, FilterConfig(short_keep_window_s=3.0))[30:33].any()
    other_kit = feat.copy()
    other_kit[30:33, :3] = [40, -30, 40]
    assert not short_but_continuing(track, i, xy, other_kit, fps, FilterConfig(short_keep_window_s=3.0))[30:33].any()
