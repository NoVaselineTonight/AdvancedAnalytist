"""Metric code of replay/benchmark.py (no pass 2 runs)."""

import numpy as np

from replay import benchmark as bm


def _o(id_, box, team="A", typ="player", xy=(0.0, 0.0), t=0.0):
    return {"id": str(id_), "box": box, "team": team, "type": typ, "xy": xy, "t": t, "frame": 0, "interp": False}


def _output(rows_by_frame, colours=None, events=()):
    frames = sorted(rows_by_frame)
    ids_time = {}
    for rows in rows_by_frame.values():
        for r in rows:
            ids_time.setdefault(r["id"], []).append(r["t"])
    return {"rows": rows_by_frame, "colours": colours or {"A": "orange", "B": "purple"}, "events": list(events),
            "ids_time": ids_time, "frames": frames, "t": [], "court": {}}


def test_iou_and_matching_greedy_one_to_one():
    assert bm.iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert bm.iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    labels = [{"box": [0, 0, 10, 20]}, {"box": [2, 0, 12, 20]}]
    outs = [_o(1, [2, 0, 12, 20])]
    m = bm.match_frame(labels, outs)
    assert m == [(1, 0)]  # the better overlap wins, the output is used once


def test_court_matching_without_box():
    m = bm.match_frame([{"court": [5.0, 5.0]}], [_o(1, None, xy=(5.5, 5.0)), _o(2, None, xy=(9, 9))])
    assert m == [(0, 0)]


def test_player_recall_phantoms_team_and_referees():
    box = lambda x: [x, 100, x + 40, 200]  # noqa: E731
    out = _output({0: [_o(1, box(0), "A"), _o(2, box(100), "A"), _o(3, box(300), "referee", "referee"),
                       _o(4, box(500), "B"), _o(5, box(700), "B")]})
    labels = {"frames": [{"frame": 0, "complete": True, "people": [
        {"box": box(0), "kind": "player", "team": "orange"},
        {"box": box(100), "kind": "player", "team": "purple"},  # team wrong
        {"box": box(200), "kind": "player", "team": "purple"},  # missed
        {"box": box(300), "kind": "referee"},
        {"box": box(500), "kind": "bench"},  # bench person shown = phantom
        {"box": box(700), "kind": "ignore"},  # unsure: neither hit nor phantom
    ]}]}
    r = bm.player_metrics(labels, out)
    assert r["player_recall"] == 2 / 3
    assert r["phantoms_per_frame"] == 1.0 and r["bench_shown_per_frame"] == 1.0
    assert r["team_acc"] == 0.5 and r["team_n"] == 2
    assert r["referee_recall"] == 1.0 and r["referee_precision"] == 1.0


def test_team_accuracy_permutation():
    r = bm.team_accuracy([("orange", "B"), ("purple", "A"), ("orange", "B")], {"A": "orange", "B": "purple"})
    assert r["team_acc"] == 0.0 and r["team_acc_perm"] == 1.0


def test_identity_switches_fragmentation_transfers():
    pid = {"p1": [(0.0, "1"), (1.0, "1"), (2.0, "2"), (3.0, "2")],  # one switch, 2 IDs
           "p2": [(0.0, "3"), (1.0, None), (2.0, "3")]}  # missed once, no switch
    oid = {"1": [(0.0, "p1"), (1.0, "p1")], "2": [(2.0, "p1"), (3.0, "p1")], "3": [(0.0, "p2"), (2.0, "p2"), (3.0, "p1")]}
    r = bm.identity_metrics(pid, oid)
    assert r["id_switches"] == 1
    assert r["id_fragmentation"] == 1.5
    assert r["id_transfers"] == 1
    assert r["id_duration_s"] == round((1.0 + 1.0 + 2.0) / 3, 2)


def test_identity_through_player_metrics_uses_nearest_sample():
    box = [0, 0, 40, 100]
    out = _output({0: [_o(1, box, t=0.0)], 3: [_o(2, box, t=0.1)]})
    labels = {"frames": [{"frame": 0, "people": [{"box": box, "pid": "x"}]},
                         {"frame": 4, "people": [{"box": box, "pid": "x"}]}]}  # 4 -> nearest sample 3
    r = bm.player_metrics(labels, out)
    assert r["player_recall"] == 1.0 and r["id_switches"] == 1 and r["persons"] == 1


def test_scores_matching_and_team():
    out = _output({}, events=[{"type": "basket", "t": 10.0, "team": "A"}, {"type": "basket", "t": 50.0, "team": "B"},
                              {"type": "basket", "t": 90.0, "team": "A"}])
    gt = {"events": [{"t": 14.0, "team": "orange", "points": 2}, {"t": 52.0, "team": "orange", "points": 2},
                     {"t": 70.0, "team": "purple", "points": 2}]}
    r = bm.score_metrics(gt, out)
    assert r["basket_recall"] == 2 / 3 and r["basket_precision"] == 2 / 3
    assert r["basket_team_acc"] == 0.5


def test_calib_keypoints_and_stale_judgements(tmp_path):
    f = tmp_path / "cal.json"
    f.write_text("{}", encoding="utf-8")
    H = np.stack([np.eye(3) * 0.1, np.eye(3) * 0.1])
    H[:, 2, 2] = 1.0  # pixel/10 = metres
    doc = {"judgements": {"calibration_sha1": "nope", "frames": {"0": "right"}},
           "keypoints": [{"frame": 0, "points": [{"img": [100, 50], "court": [10.0, 5.0]}]},
                         {"frame": 3, "points": [{"img": [100, 50], "court": [12.0, 5.0]}]}]}
    r = bm.calib_metrics(doc, f, H, [0, 3])
    assert r["judged_right_pct"] is None
    assert r["kp_right_pct"] == 50.0 and r["kp_wrong_pct"] == 50.0


def test_overall_and_diff():
    a = {"player_recall": 0.5, "labelled_players": 10, "id_switches": 2}
    b = {"player_recall": 1.0, "labelled_players": 30, "id_switches": 1}
    o = bm.overall([a, b])
    assert abs(o["player_recall"] - 35 / 40) < 1e-9 and o["id_switches"] == 3
    d = bm.diff({"player_recall": 0.9, "id_switches": 4}, {"player_recall": 0.8, "id_switches": 3})
    assert any("better" in x for x in d) and any("WORSE" in x for x in d)


def test_scorebar_reader_format_and_edges():
    rows = {0: [_o(1, None, t=0.0)], 3: [_o(1, None, t=120.0)]}
    out = _output(rows, events=[{"type": "basket", "t": 20.0, "team": "A"}, {"type": "basket", "t": 115.0, "team": "B"}])
    gt = {"events": [{"t_bar": 1.0, "jersey": "orange"}, {"t_bar": 25.0, "jersey": "orange"}]}
    r = bm.score_metrics(gt, out)
    assert r["baskets_gt"] == 1 and r["basket_recall"] == 1.0
    assert r["basket_precision"] == 1.0  # the unmatched basket at 115 s may be confirmed after the clip
    assert r["basket_team_acc"] == 1.0


def test_load_labels_later_file_wins(tmp_path, monkeypatch):
    monkeypatch.setattr(bm, "LABELS", tmp_path)
    (tmp_path / "x").mkdir()
    (tmp_path / "players_c1.json").write_text('{"frames": [{"frame": 1, "people": []}, {"frame": 2, "people": []}]}',
                                              encoding="utf-8")
    (tmp_path / "x" / "players_c1.json").write_text('{"frames": [{"frame": 2, "people": [{"box": [0,0,1,1]}]}]}',
                                                    encoding="utf-8")
    files = bm.label_files("c1")["players"]
    d = bm.load_labels(files)
    assert [f["frame"] for f in d["frames"]] == [1, 2] and len(d["frames"][1]["people"]) == 1


def test_predicted_rows_match_by_court_distance_and_unverified_are_not_phantoms():
    H = np.stack([np.eye(3) * 0.01])  # pixel / 100 = metres
    H[:, 2, 2] = 1.0
    box = [100, 100, 140, 300]  # foot (120, 300) -> (1.2, 3.0) m
    out = _output({0: [_o(1, None, xy=(1.5, 3.0)),  # predicted row near the labelled (hidden) player
                       _o(2, None, xy=(9.0, 9.0)),  # predicted row where nobody is labelled
                       _o(3, [500, 100, 540, 300], xy=(5.0, 3.0))]})  # detected row on nobody = phantom
    labels = {"frames": [{"frame": 0, "complete": True, "people": [{"box": box, "kind": "player", "team": "orange"}]}]}
    r = bm.player_metrics(labels, out, H, [0])
    assert r["player_recall"] == 1.0 and r["player_recall_measured"] == 0.0
    assert r["unverified_per_frame"] == 1.0 and r["phantoms_per_frame"] == 1.0


def test_team_accuracy_generic_labels():
    r = bm.team_accuracy([("offence", "B"), ("defence", "A"), ("offence", "A")], {"A": "orange", "B": "purple"})
    assert r["team_acc"] is None and abs(r["team_acc_perm"] - 2 / 3) < 1e-9
    r = bm.team_accuracy([("purple", "A"), ("purple", "A"), ("orange", "B")], {"A": "orange", "B": "purple"})
    assert r["team_acc"] == 0.0 and r["team_acc_perm"] == 1.0


def test_segment_scoped_pids_are_not_transfers_across_segments():
    pid = {"Oa_A": [(0.0, "1")], "Oa_B": [(10.0, "1")], "x": [(11.0, "1")]}
    oid = {"1": [(0.0, "Oa_A", True, "A"), (10.0, "Oa_B", True, "B"), (11.0, "x", False, "B")]}
    r = bm.identity_metrics(pid, oid)
    assert r["id_transfers"] == 1  # only Oa_B -> x inside segment B


def test_aggregate_and_significance():
    r1 = [{"clip": "a", "player_recall": 0.90, "labelled_players": 10}]
    r2 = [{"clip": "a", "player_recall": 0.94, "labelled_players": 10}]
    r3 = [{"clip": "a", "player_recall": 0.92, "labelled_players": 10}]
    means, sds = bm.aggregate([r1, r2, r3])
    assert abs(means[0]["player_recall"] - 0.92) < 1e-9 and abs(sds["a"]["player_recall"] - 0.02) < 1e-9
    doc = {"replicas": [r1, r2, r3], "clips": means}
    m, sd, n = bm.overall_stats(doc)
    assert n == 3 and abs(sd["player_recall"] - 0.02) < 1e-9
    small = bm.diff({"player_recall": 0.93}, m, {"player_recall": 0.02}, sd, 3, 3)
    big = bm.diff({"player_recall": 0.99}, m, {"player_recall": 0.02}, sd, 3, 3)
    assert "within noise" in small[0] and "better" in big[0]
    single = bm.diff({"player_recall": 0.93}, {"player_recall": 0.92})
    assert "noise unknown" in single[0]


def _write_output(out, rows):
    import csv
    import json

    from replay.export import COLUMNS
    out.mkdir(parents=True)
    with open(out / "tracking.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in COLUMNS})
    (out / "tracking.json").write_text(json.dumps({"game": {"teams": {}}}), encoding="utf-8")
    (out / "replay_data.json").write_text(json.dumps({"frame": [0, 3], "t": [0.0, 0.1]}), encoding="utf-8")


def test_load_output_scores_track_or_player_id(tmp_path):
    row = lambda f, tid, pid: {"frame": f, "timestamp": f / 30, "track_id": tid, "object_type": "player",  # noqa: E731
                               "team": "A", "x_m": 1, "y_m": 2, "interpolated": "False", "player_id": pid}
    _write_output(tmp_path / "out", [row(0, 7, 3), row(3, 9, 3), row(3, 12, "")])
    tr = bm.load_output(tmp_path / "out")
    assert sorted(o["id"] for rs in tr["rows"].values() for o in rs) == ["12", "7", "9"]
    pl = bm.load_output(tmp_path / "out", "player_id")
    assert sorted(o["id"] for rs in pl["rows"].values() for o in rs) == ["p3", "p3", "t12"]
    assert set(pl["ids_time"]) == {"p3", "t12"}
    import pytest
    with pytest.raises(ValueError):
        bm.load_output(tmp_path / "out", "slot")


def test_bench_ids_drawn_and_corrections_estimate():
    from replay import bench_metrics as bmx

    box = lambda x: [x, 100, x + 40, 200]  # noqa: E731
    frames = []
    for k in range(3):
        frames.append({"frame": k, "complete": True, "people": [{"box": box(0), "kind": "bench"},
                                                                {"box": box(200), "kind": "player", "pid": "Zyxa"}]})
    out = _output({0: [_o(1, box(0), t=0.0), _o(5, box(200), t=0.0)],
                   1: [_o(2, box(0), t=0.1), _o(5, box(200), t=0.1)],
                   2: [_o(2, box(0), t=0.2), _o(6, box(200), t=0.2)]})
    r = bm.player_metrics({"frames": frames}, out)
    assert r["bench_ids_drawn"] == 2 and r["id_switches"] == 1
    rows = [{"id": "1", "team": "A"}, {"id": "1", "team": "B"}, {"id": "2", "team": "A"}, {"id": "3", "team": "referee"}]
    assert bmx.cross_team_ids(rows) == 1
    est = bmx.corrections_per_game(2, 1, 3, clip_s=240.0)
    assert est == {"corrections_clip": 6, "corrections_per_game": 60.0}
    assert bmx.corrections_per_game(None, 0, None, None) == {"corrections_clip": 0, "corrections_per_game": None}
    out["t"] = [0.0, 0.1, 0.2]
    e = bm.corrections_estimate({"id_transfers": 1, "bench_ids_drawn": 2}, out)
    assert e["cross_team_ids"] == 0 and e["corrections_clip"] == 3
    assert e["corrections_per_game"] == round(3 * bmx.GAME_S / 0.2, 1)
