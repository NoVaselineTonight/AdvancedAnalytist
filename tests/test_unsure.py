"""Unsure-segment detector (replay/unsure.py) on synthetic pass-2 rows."""

from replay import unsure as U

FPS = 10.0


def rows(tid, t0, t1, *, x=10.0, y=7.0, team="A", conf=0.9, pid="", source="seen", dx=0.0):
    out = []
    n = int(round((t1 - t0) * FPS))
    for k in range(n + 1):
        t = round(t0 + k / FPS, 2)
        out.append({"timestamp": t, "track_id": tid, "object_type": "player", "team": team, "x_m": x + dx * k,
                    "y_m": y, "id_confidence": conf, "player_id": pid, "source": source})
    return out


def kinds(segs, tid):
    return {k for s in segs if s["id"] == str(tid) for k in s["kinds"]}


def test_clean_output_has_nothing_unsure():
    r = rows(1, 0, 10, dx=0.1) + rows(2, 0, 10, team="B", y=5.0, dx=0.1)
    assert U.unsure_segments(r, {"1": {"team": "A"}, "2": {"team": "B"}}) == []
    assert U.unsure_segments([]) == []


def test_low_confidence_run_is_its_own_segment():
    r = rows(1, 0, 3, dx=0.1) + rows(1, 3.1, 5, conf=0.3, dx=0.1) + rows(1, 5.1, 8, dx=0.1)
    segs = U.unsure_segments(r)
    assert len(segs) == 1 and segs[0]["kinds"] == ["low_confidence"]
    assert segs[0]["t0"] == 3.1 and segs[0]["t1"] == 5.0
    assert abs(segs[0]["score"] - U.W_LOW * (0.6 - 0.3) / 0.6) < 1e-3


def test_long_gap_counts_only_measured_rows():
    r = rows(1, 0, 2, dx=0.1) + rows(1, 2.1, 8, source="predicted") + rows(1, 8.1, 10, dx=0.1)
    segs = U.unsure_segments(r)
    assert [s["kinds"] for s in segs] == [["long_gap"]]
    assert segs[0]["t0"] == 2.0 and segs[0]["t1"] == 8.1
    short = rows(1, 0, 2) + rows(1, 5.0, 7)  # 3 s gap: below GAP_S
    assert "long_gap" not in kinds(U.unsure_segments(short), 1)


def test_cross_team_id_and_player_id():
    r = rows(1, 0, 6, dx=0.1) + rows(1, 6.1, 7, team="B", dx=0.1)
    segs = U.unsure_segments(r)
    s = next(s for s in segs if "cross_team" in s["kinds"])
    assert s["t0"] == 6.1 and s["t1"] == 7.0
    r = rows(1, 0, 6, pid=4, dx=0.1) + rows(2, 6.5, 8, team="B", pid=4, y=5.0, dx=0.1)
    assert kinds(U.unsure_segments(r), 2) == {"cross_team"} and kinds(U.unsure_segments(r), 1) == set()


def test_bench_zone_identity():
    r = rows(7, 0, 20, x=20.0, y=15.3) + rows(1, 0, 20, dx=0.05)
    segs = U.unsure_segments(r, {"7": {"team": "A"}, "1": {"team": "A"}}, length=28.0, width=15.0)
    assert kinds(segs, 7) == {"bench_zone"} and kinds(segs, 1) == set()
    ref = U.unsure_segments(rows(7, 0, 20, x=20.0, y=15.3), {"7": {"team": "referee"}})
    assert ref == []  # referees stand on the sideline


def test_number_conflicts():
    r = rows(1, 0, 10, dx=0.1) + rows(2, 5, 12, y=3.0, dx=0.1)
    tracks = {"1": {"team": "A", "number": "12"}, "2": {"team": "A", "number": "12"}}
    segs = U.unsure_segments(r, tracks)
    assert kinds(segs, 1) == {"number_conflict"} and kinds(segs, 2) == {"number_conflict"}
    s = next(s for s in segs if s["id"] == "1")
    assert s["t0"] == 5.0 and s["t1"] == 10.0
    tracks["2"]["team"] = "B"  # the other team's #12: fine
    assert U.unsure_segments(r, tracks) == []
    r = rows(1, 0, 10, pid=3, dx=0.1) + rows(2, 11, 13, pid=3, dx=0.1)
    segs = U.unsure_segments(r, {"1": {"team": "A", "number": "7"}, "2": {"team": "A", "number": "9"}})
    assert kinds(segs, 2) == {"number_conflict"} and kinds(segs, 1) == set()


def test_ranking_and_merging():
    r = (rows(1, 0, 6, dx=0.1) + rows(1, 6.1, 7, team="B", conf=0.2, dx=0.1)  # cross team + low conf: merged
         + rows(2, 0, 3, y=3.0, conf=0.5, dx=0.1))
    segs = U.unsure_segments(r)
    assert [s["id"] for s in segs] == ["1", "2"]
    assert segs[0]["kinds"] == ["cross_team", "low_confidence"] and segs[0]["score"] > segs[1]["score"]


def test_rows_from_tracking():
    doc = {"columns": ["timestamp", "track_id"], "rows": [[0.0, 1], [0.1, 1]]}
    assert U.rows_from_tracking(doc) == [{"timestamp": 0.0, "track_id": 1}, {"timestamp": 0.1, "track_id": 1}]
    assert U.rows_from_tracking({}) == []


def test_runs_on_a_project_tracking_json(tmp_path):
    import json

    from fake_project import make_project

    root = make_project(tmp_path / "p")
    doc = json.loads((root / "results" / "tracking.json").read_text(encoding="utf-8"))
    segs = U.unsure_segments(U.rows_from_tracking(doc), doc.get("tracks"))
    assert isinstance(segs, list) and all(s["id"] != "ball" for s in segs)
