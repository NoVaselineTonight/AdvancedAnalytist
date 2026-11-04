"""Team classification: colour histograms per box, clustering into teams/referees, meta colour handling."""

from types import SimpleNamespace

import cv2
import numpy as np

from replay import team_appearance as ta
from replay.config import TeamConfig
from replay.features import jersey_features
from replay.metadata import metadata_from_dict
from replay.teams import A, B, REF, UNKNOWN, classify, swap_ab

FLOOR = (232, 238, 242)  # BGR, very light floor: achromatic like a gym floor, but far from every kit (FLOOR_MASK)
SKIN = (120, 150, 205)
KITS = {"orange": (20, 60, 210), "purple": (90, 25, 70), "ref": (150, 150, 150), "bench": (25, 25, 35)}
SHORTS = {"orange": (20, 60, 210), "purple": (90, 25, 70), "ref": (20, 20, 20), "bench": (25, 25, 35)}


def _person(img, box, kit, rng, side_on=False):
    """Paint a crude person: head, shirt, shorts, skin arms (side-on players show mostly arm), legs."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    cx = (x1 + x2) // 2
    sw = int(w * (0.18 if side_on else 0.32))
    img[y1:y1 + int(0.15 * h), cx - sw // 2:cx + sw // 2] = SKIN
    img[y1 + int(0.15 * h):y1 + int(0.50 * h), cx - sw:cx + sw] = KITS[kit]
    img[y1 + int(0.50 * h):y1 + int(0.65 * h), cx - sw:cx + sw] = SHORTS[kit]
    arm = int(w * (0.22 if side_on else 0.08))  # arm in front of the shirt
    img[y1 + int(0.18 * h):y1 + int(0.50 * h), cx - arm // 2:cx + arm // 2] = SKIN
    img[y1 + int(0.65 * h):y2, cx - sw // 2:cx + sw // 2] = SKIN
    noise = rng.normal(0, 4, img[y1:y2, x1:x2].shape)
    img[y1:y2, x1:x2] = np.clip(img[y1:y2, x1:x2] + noise, 0, 255)


def _scene(kits, rng):
    img = np.zeros((360, 120 * len(kits), 3), np.uint8)
    img[:] = FLOOR
    boxes = []
    for i, kit in enumerate(kits):
        box = (i * 120 + 20, 60, i * 120 + 100, 300)
        _person(img, box, kit, rng, side_on=bool(rng.random() < 0.5))
        boxes.append(box)
    return img, np.array(boxes, float)


def _dataset(rng, n_per=12, tracks=None):
    """Detections of tracks with known kits: (track_ids, frames, feats, Appearance, truth)."""
    tracks = tracks or (["orange"] * 6 + ["purple"] * 6 + ["ref"] * 2 + ["bench"] * 3)
    tid, fr, feats, desc, sl, truth = [], [], [], [], [], {}
    for f in range(n_per):
        img, boxes = _scene(tracks, rng)
        d, s = ta.image_analysis(img, boxes)
        jf = jersey_features(img, boxes)
        for i, kit in enumerate(tracks):
            tid.append(i + 1)
            fr.append(f)
            feats.append(jf[i])
            desc.append(d[i])
            sl.append(s[i])
            truth[i + 1] = kit
    n = len(tid)
    app = ta.Appearance(np.array(desc), np.zeros(n), np.arange(n), np.full(n, 240.0), None, np.array(sl))
    return np.array(tid), np.array(fr), np.array(feats), app, truth


def _check_teams(res, truth, allow_unknown_bench=True):
    lab_of = {}
    for t, kit in truth.items():
        lab_of.setdefault(kit, set()).add(res.track_team[t])
    assert len(lab_of["orange"]) == 1 and len(lab_of["purple"]) == 1
    assert lab_of["orange"] != lab_of["purple"]
    assert lab_of["orange"] | lab_of["purple"] == {A, B}
    assert lab_of["ref"] == {REF}
    if allow_unknown_bench:
        assert lab_of["bench"] <= {UNKNOWN, REF}
    return next(iter(lab_of["orange"])), next(iter(lab_of["purple"]))


def _hue_deg(hexcol):
    rgb = np.array([[[int(hexcol[i:i + 2], 16) for i in (5, 3, 1)]]], np.uint8)  # to BGR
    return float(cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV_FULL)[0, 0, 0]) * 360 / 255


def test_descriptor_sees_purple_despite_arm_and_floor():
    rng = np.random.default_rng(0)
    img = np.zeros((300, 120, 3), np.uint8)
    img[:] = FLOOR
    _person(img, (20, 30, 100, 290), "purple", rng, side_on=True)
    d = ta.box_descriptor(img, (20, 30, 100, 290))
    assert np.isclose(d.sum(), 1.0)
    hm = ta.hue_mass(d)[:, 1:].sum(axis=1)  # clearly coloured mass per hue sector
    purple = hm[9:].sum()  # sectors 270..330 deg
    assert purple > 0.15
    # the pass-1 median of the same box is skin/floor-coloured: that is why pass 2 re-samples
    med = jersey_features(img, np.array([[20, 30, 100, 290]]))[0, :3]
    assert np.degrees(np.arctan2(med[2], med[1])) > -10


def test_occlusion_counts_people_in_front_fully_and_behind_half():
    boxes = np.array([[0, 0, 100, 200], [0, 0, 100, 220], [0, 0, 100, 180]], float)
    occ = ta.occlusion(boxes)
    assert occ[1] == 0.5 * 1.0 or np.isclose(occ[1], 0.5)  # only people behind it
    assert np.isclose(occ[0], 1.0)  # box 1 stands in front of box 0
    assert np.isclose(occ[2], 1.0)


def test_auto_teams_and_referees_with_stale_placeholder_colours():
    rng = np.random.default_rng(1)
    tid, fr, feats, app, truth = _dataset(rng)
    meta = metadata_from_dict({"team_colors_auto": False,  # what the GUI writes on save, with placeholders
                               "jersey_colors": {"home": {"primary": "#f5f5f5"}, "away": {"primary": "#1f2a5a"}}})
    res = classify(tid, fr, feats, meta, TeamConfig(), appearance=app)
    orange, purple = _check_teams(res, truth)
    assert not res.seeded
    # display colours come from the jerseys: reddish-orange and purple, not white/navy placeholders
    assert _hue_deg(res.colors[orange]) < 40 or _hue_deg(res.colors[orange]) > 340
    assert 260 < _hue_deg(res.colors[purple]) < 330
    assert all(res.track_share[t] > 0.5 for t, k in truth.items() if k in ("orange", "purple"))
    sw = swap_ab(res)
    assert sw.track_team[1] == (B if orange == A else A)


def test_user_colours_order_teams_only_when_they_match():
    rng = np.random.default_rng(2)
    tid, fr, feats, app, truth = _dataset(rng)
    for home, away, want_seeded in (("#5a2d82", "#e8501e", True), ("#e8501e", "#5a2d82", True),
                                    ("#2e8b57", "#ffd700", False)):  # green/yellow: not this game
        meta = metadata_from_dict({"team_colors_auto": False,
                                   "jersey_colors": {"home": {"primary": home}, "away": {"primary": away}}})
        res = classify(tid, fr, feats, meta, TeamConfig(), appearance=app)
        orange, purple = _check_teams(res, truth)
        assert res.seeded == want_seeded
        if want_seeded:
            assert (purple == A) == (home == "#5a2d82")
            assert res.colors[A] == home
        else:
            assert any("do not match" in n for n in res.notes)


def test_picked_samples_decide_home_and_away():
    rng = np.random.default_rng(3)
    tid, fr, feats, app, truth = _dataset(rng)
    orange_rows = [i for i, t in enumerate(tid) if truth[t] == "orange"][:3]
    purple_rows = [i for i, t in enumerate(tid) if truth[t] == "purple"][:3]
    app.samples = {A: app.desc[purple_rows], B: app.desc[orange_rows]}
    meta = metadata_from_dict({"team_samples": {A: [[0] * 6], B: [[0] * 6]}})
    res = classify(tid, fr, feats, meta, TeamConfig(), appearance=app)
    orange, purple = _check_teams(res, truth)
    assert purple == A and orange == B and res.seeded


def test_without_video_pass1_features_still_separate_teams():
    rng = np.random.default_rng(4)
    tid, fr, feats, _, truth = _dataset(rng, tracks=["orange"] * 6 + ["purple"] * 6 + ["ref"] * 2)
    # front-on players: the pass-1 torso median is the shirt colour
    res = classify(tid, fr, feats, metadata_from_dict({}), TeamConfig(), appearance=None)
    lab = {k: {res.track_team[t] for t, kk in truth.items() if kk == k} for k in ("orange", "purple")}
    assert len(lab["orange"]) == 1 and len(lab["purple"]) == 1 and lab["orange"] != lab["purple"]
    assert lab["orange"] | lab["purple"] == {A, B}


def test_compute_appearance_from_video_and_cache(tmp_path):
    rng = np.random.default_rng(5)
    fps, n = 10.0, 12
    path = tmp_path / "v.avi"
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (240, 360))
    img, boxes = _scene(["orange", "purple"], rng)
    for _ in range(n):
        vw.write(img)
    vw.release()
    rows = []
    for k in range(n):
        for j, b in enumerate(boxes):
            rows.append([k, *b, 0.9, 0, j + 1, *([np.nan] * 6)])
    data = SimpleNamespace(k=np.arange(n), t=np.arange(n) / fps, frame=np.arange(n), rows=np.array(rows, float),
                           signature={"video": str(path), "video_size_bytes": 1, "start_s": 0.0})
    app = ta.load_or_compute(tmp_path, data, path, every_s=0.3)
    sampled = np.isfinite(app.desc).all(axis=1)
    assert 6 <= sampled.sum() <= 10  # frames 0, 3, 6, 9 x 2 people
    hm = ta.hue_mass(app.desc[sampled])[:, :, 1:].sum(-1)
    assert (hm[0::2, 9:].sum(1) < 0.05).all() and (hm[1::2, 9:].sum(1) > 0.1).all()  # orange vs purple person
    assert (tmp_path / "team_appearance.npz").exists()
    path.unlink()  # cached: no video needed any more
    again = ta.load_or_compute(tmp_path, data, path, every_s=0.3)
    assert again is not None and np.allclose(np.nan_to_num(again.desc), np.nan_to_num(app.desc), atol=1e-3)


def test_more_picked_kits_than_tracks_does_not_crash():
    """Bug hunt 19.10: picked samples for A, B and referee but only two described tracks (short clip) -> the seeded
    k-means got 3 seeds for k = 2 and raised."""
    rng = np.random.default_rng(3)
    tid, fr, feats, app, truth = _dataset(rng, n_per=4, tracks=["orange", "purple"])
    app.samples = {A: app.desc[tid == 2][:2], B: app.desc[tid == 1][:2], REF: app.desc[tid == 1][:1]}
    meta = metadata_from_dict({"team_samples": {A: [[0] * 6], B: [[0] * 6], REF: [[0] * 6]}})
    res = classify(tid, fr, feats, meta, TeamConfig(), appearance=app)
    assert set(res.track_team) == {1, 2}
