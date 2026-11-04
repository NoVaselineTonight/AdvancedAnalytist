"""replay/bench_metrics.py: HOTA / IDF1 / MOTA / extrapolation helpers."""

import numpy as np
import pytest

from replay import bench_metrics as bmx


def _steps(swap_at=None, n=10):
    steps = []
    for t in range(n):
        tr = ["a", "b"] if swap_at is None or t < swap_at else ["b", "a"]
        steps.append((["g1", "g2"], tr, np.eye(2)))
    return steps


def test_perfect_tracking_scores_one():
    st = _steps()
    assert abs(bmx.hota(st)["HOTA"] - 1.0) < 1e-9
    assert bmx.idf1(st)["IDF1"] == 1.0
    m = bmx.mota(st)
    assert m["MOTA"] == 1.0 and m["IDSW"] == 0


def test_identity_swap_lowers_association_not_detection():
    st = _steps(swap_at=5)
    h = bmx.hota(st)
    assert abs(h["DetA"] - 1.0) < 1e-9 and h["AssA"] < 0.6
    assert bmx.idf1(st)["IDF1"] == 0.5
    assert bmx.mota(st)["IDSW"] == 2


def test_missed_and_false_detections():
    st = [(["g1"], ["x"], np.array([[0.0]]))] * 4
    m = bmx.mota(st)
    assert m["MOTA"] == -1.0  # every gt missed and every output false
    assert bmx.hota(st)["HOTA"] == 0.0


def test_build_steps_ignore_and_incomplete():
    frames = [{"frame": 0, "complete": False, "people": [
        {"pid": "p1", "kind": "player", "box": [0, 0, 10, 10]},
        {"kind": "bench", "box": [100, 0, 110, 10]}]}]
    outs = [{"id": "1", "box": [0, 0, 10, 10], "xy": (0, 0)}, {"id": "2", "box": [100, 0, 110, 10], "xy": (5, 5)},
            {"id": "3", "box": [300, 0, 310, 10], "xy": (9, 9)}]
    st = bmx.build_steps(frames, lambda f: outs)
    assert st[0][0] == ["p1"] and st[0][1] == ["1"]  # bench output ignored, unrelated output dropped (incomplete)


def test_player_extrapolation_on_straight_line():
    k = np.arange(0, 200)
    xy = np.c_[k * 0.1, np.full(len(k), 5.0)]  # 1 m/s at 10 fps
    r = bmx.player_extrapolation({"1": (k, xy)}, 10.0, None, None)
    assert r["extrap_1s_med"] < 0.5 and r["extrap_1s_n"] > 0


def test_continuity_metrics_count_blinks_and_volatility():
    from replay.benchmark import continuity_metrics
    fps = 10.0
    rows = []
    for k in range(0, 100):          # id 1 present except samples 40 and 41 (one 2-sample hole)
        if k not in (40, 41):
            rows.append({"frame": 3 * k, "timestamp": k / fps, "track_id": 1, "object_type": "player", "team": "A"})
    for k in range(0, 100):          # id 2: present 0..49 only
        if k < 50:
            rows.append({"frame": 3 * k, "timestamp": k / fps, "track_id": 2, "object_type": "player", "team": "B"})
    for k in range(0, 100):          # 4 B players popping in for sample 70 only -> a count jump of 4
        if k == 70:
            for t in (3, 4, 5, 6):
                rows.append({"frame": 3 * k, "timestamp": k / fps, "track_id": t, "object_type": "player", "team": "B"})
    m = continuity_metrics(rows)
    assert m["blink_holes_per_min"] == pytest.approx(1 / (10.0 / 60.0), rel=1e-6)   # 1 hole in a 10 s clip
    assert m["count_volatility"] == pytest.approx(2 / 99, rel=1e-6)                 # jumps at 69->70 and 70->71
    assert m["median_id_s"] == pytest.approx(0.1, abs=0.06)                         # four 1-sample ids dominate the median


def test_continuity_metrics_fps_drift_does_not_merge_samples():
    from replay.benchmark import continuity_metrics
    fps = 9.99001  # real process_fps: round(t * 10) would merge two samples about every 100 s
    rows = [{"frame": 3 * k, "timestamp": round(k / fps, 3), "track_id": tid, "object_type": "player", "team": "A"}
            for k in range(1500) for tid in range(10)]
    m = continuity_metrics(rows)
    assert m["blink_holes_per_min"] == 0 and m["count_volatility"] == 0
