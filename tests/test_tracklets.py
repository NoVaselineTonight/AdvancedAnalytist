"""Offline identity tracking (replay/tracklets.py): synthetic crossings, occlusions, partial boxes, pans."""

import numpy as np

from replay.config import CourtTrackConfig
from replay.court_tracker import track_people
from replay.tracklets import CameraFrames, link_tracks

from synthetic import court_to_image_matrix, sideline_camera

FPS = 10.0
ORANGE = [40.0, 30.0, 22.0, 35.0, 25.0, 16.0]
PURPLE = [28.0, 12.0, -12.0, 28.0, 10.0, -10.0]


def _project3d(cam, pts):
    K, R, t = cam
    q = (K @ (R @ np.asarray(pts, float).T + t[:, None])).T
    return q[:, :2] / q[:, 2:3]


def _scene(paths, kits, bt_of, hidden=None, partial=None, cam=None, height=1.8):
    """paths: {pid: (n,2) court positions per sample}; kits: {pid: 6 features}; bt_of(pid, k) -> ByteTrack ID;
    hidden: {pid: samples not detected}; partial: {pid: samples whose box ends at the knees}."""
    cam = cam or sideline_camera()
    G = court_to_image_matrix(*cam)
    H = np.linalg.inv(G)
    rows = []
    for pid, path in paths.items():
        for k, (x, y) in enumerate(path):
            if hidden and k in hidden.get(pid, ()):
                continue
            foot, head = _project3d(cam, [[x, y, 0.0], [x, y, height]])
            h = foot[1] - head[1]
            y2 = foot[1] - (0.45 * h if partial and k in partial.get(pid, ()) else 0.0)
            box = [foot[0] - 0.18 * h, head[1], foot[0] + 0.18 * h, y2]
            rows.append((k, pid, box, bt_of(pid, k)))
    rows.sort(key=lambda r: r[0])
    k = np.array([r[0] for r in rows])
    pid = np.array([r[1] for r in rows])
    boxes = np.array([r[2] for r in rows], float)
    bt = np.array([r[3] for r in rows])
    feet = np.c_[(boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3], np.ones(len(rows))] @ H.T
    xy = feet[:, :2] / feet[:, 2:3]  # measured court position (wrong for partial boxes, as in pass 2)
    R = np.tile(np.eye(2) * 0.04, (len(rows), 1, 1))
    rng = np.random.default_rng(0)
    feats = np.array([kits[p] for p in pid]) + rng.normal(0, 2.0, (len(rows), 6))
    n = int(k.max()) + 1
    motion = [None] + [np.eye(3)] * (n - 1)
    return dict(k=k, pid=pid, boxes=boxes, bt=bt, xy=xy, R=R, feats=feats, conf=np.full(len(rows), 0.9),
                motion=motion, n=n, H=[H] * n)


def _run(s, cfg=None, court=True, **kw):
    xy = s["xy"] if court else np.full_like(s["xy"], np.nan)
    R = s["R"] if court else np.full_like(s["R"], np.nan)
    return link_tracks(s["k"], xy, R, s["feats"], s["bt"], s["conf"], FPS, cfg or CourtTrackConfig(),
                       boxes=s["boxes"], motion=s["motion"], n_samples=s["n"], H=s["H"] if court else None, **kw)


def _purity(ids, pid):
    """Each person maps to exactly one identity and each identity to one person."""
    per_person = {p: set(ids[pid == p]) for p in np.unique(pid)}
    per_id = {i: set(pid[ids == i]) for i in np.unique(ids) if i > 0}
    return per_person, per_id


def test_crossing_with_bytetrack_swap_is_repaired():
    """Two players of different teams cross; ByteTrack swaps their IDs at the crossing."""
    n = 60
    a = np.c_[np.linspace(6, 14, n), np.full(n, 6.0)]
    b = np.c_[np.linspace(14, 6, n), np.full(n, 6.3)]
    s = _scene({1: a, 2: b}, {1: ORANGE, 2: PURPLE},
               bt_of=lambda p, k: (p if k < 30 else 3 - p))  # IDs swap at the crossing
    res = _run(s)
    per_person, per_id = _purity(res.ids, s["pid"])
    assert all(len(v) == 1 for v in per_person.values()), per_person
    assert all(len(v) == 1 for v in per_id.values()), per_id


def test_occlusion_gap_keeps_identity():
    """A player hidden for 2 s behind another comes back with a new ByteTrack ID."""
    n = 80
    a = np.c_[np.linspace(5, 13, n), np.full(n, 7.0)]
    b = np.c_[np.full(n, 10.0), np.linspace(3, 6, n)]
    s = _scene({1: a, 2: b}, {1: ORANGE, 2: PURPLE}, bt_of=lambda p, k: p * 10 + (k >= 40),
               hidden={1: range(25, 45)})
    for court in (True, False):  # also from image cues alone (no calibration)
        res = _run(s, court=court)
        per_person, per_id = _purity(res.ids, s["pid"])
        assert len(per_person[1]) == 1 and len(per_person[2]) == 1, (court, per_person)
        assert all(len(v) == 1 for v in per_id.values())


def test_same_team_crossing_keeps_motion():
    """Same kit: only motion can tell them apart. Two runners (4 m/s) cross at right angles; both are lost
    for 0.5 s in the overlap and come back with new ByteTrack IDs: the identities must continue straight on."""
    n = 40
    t = np.arange(n) / FPS
    a = np.c_[4 + 4.0 * t, np.full(n, 8.0)]
    b = np.c_[np.full(n, 12.0), 0.0 + 4.0 * t]
    s = _scene({1: a, 2: b}, {1: ORANGE, 2: ORANGE}, bt_of=lambda p, k: p * 10 + (k >= 23),
               hidden={1: range(18, 23), 2: range(18, 23)})
    res = _run(s)
    per_person, per_id = _purity(res.ids, s["pid"])
    assert all(len(v) == 1 for v in per_id.values()), per_id
    assert all(len(v) == 1 for v in per_person.values()), per_person


def test_partial_box_foot_is_reestimated():
    """Samples 15..24: a team-mate standing in front hides player 1's legs; the box ends at the knees."""
    n = 40
    a = np.c_[np.linspace(8, 12, n), np.full(n, 9.0)]
    front = a + 0.15 * (np.array([14.0, -9.0]) - a)  # on the line to the camera (synthetic.sideline_camera)
    s = _scene({1: a, 2: front}, {1: ORANGE, 2: PURPLE}, bt_of=lambda p, k: p, partial={1: range(15, 25)},
               hidden={2: [k for k in range(n) if not 15 <= k < 25]})
    m = s["pid"] == 1
    raw_err = np.linalg.norm(s["xy"][m][15:25] - a[15:25], axis=1)
    res = _run(s)
    err = np.linalg.norm(res.xy[m][15:25] - a[15:25], axis=1)
    assert res.partial[m][15:25].all() and not res.partial[m][:10].any()
    assert raw_err.min() > 1.0  # knees projected as feet: metres off
    assert err.max() < 0.3


def test_bench_person_not_linked_to_player():
    """A seated person near the sideline and a player who appears later nearby stay apart (height differs)."""
    n = 60
    bench = np.c_[np.full(n, 10.0), np.full(n, 15.6)]
    player = np.c_[np.linspace(9, 11, n), np.full(n, 14.0)]
    s_b = _scene({1: bench}, {1: ORANGE}, bt_of=lambda p, k: 1, hidden={1: range(30, 60)}, height=1.1)
    s_p = _scene({2: player}, {2: ORANGE}, bt_of=lambda p, k: 2, hidden={2: range(0, 33)})
    s = {key: np.concatenate([s_b[key], s_p[key]]) for key in ("k", "pid", "boxes", "bt", "xy", "R", "feats", "conf")}
    s.update(motion=[None] + [np.eye(3)] * (n - 1), n=n, H=[s_b["H"][0]] * n)
    res = _run(s)
    assert set(res.ids[s["pid"] == 1]).isdisjoint(res.ids[s["pid"] == 2])


def test_camera_frames_match_direct_chain():
    rng = np.random.default_rng(3)
    n = 150
    motion = [None]
    for _ in range(n - 1):
        M = np.eye(3)
        M[0, 2] = rng.normal(-6, 1)  # steady pan
        M[1, 2] = rng.normal(0, 0.5)
        M[0, 0] = M[1, 1] = 1 + rng.normal(0, 0.002)
        motion.append(M)
    motion[100] = None  # broken chain
    cam = CameraFrames(motion, n, block=16)
    T = cam.transform(5, 90)
    direct = np.eye(3)
    for i in range(6, 91):
        direct = motion[i] @ direct
    p = np.array([640.0, 360.0, 1.0])
    q1, q2 = T @ p, direct @ p
    assert np.allclose(q1[:2] / q1[2], q2[:2] / q2[2], atol=1e-6)
    assert cam.transform(90, 5) is not None and cam.transform(95, 105) is None


def test_track_people_dispatch():
    n = 30
    a = np.c_[np.linspace(5, 8, n), np.full(n, 7.0)]
    s = _scene({1: a}, {1: ORANGE}, bt_of=lambda p, k: 1)
    for method in ("online", "offline"):
        cfg = CourtTrackConfig(method=method)
        ids, xy, R = track_people(s["k"], s["xy"], s["R"], s["feats"], s["bt"], s["conf"], FPS, cfg,
                                  boxes=s["boxes"], motion=s["motion"], n_samples=s["n"], H=s["H"])
        assert len(set(ids)) == 1 and ids[0] > 0 and xy.shape == s["xy"].shape and R.shape == s["R"].shape


def test_absorb_takes_up_to_eight_crumbs():
    from replay.config import CourtTrackConfig
    assert CourtTrackConfig().absorb_max_dets == 8
