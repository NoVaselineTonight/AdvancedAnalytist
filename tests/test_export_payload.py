"""export.py: replay_data tracks are aligned to samples (never past the end), carry per-sample confidence,
and outputs are written atomically."""

import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from replay.calibration import KEYFRAME
from replay.export import COLUMNS, TrackRow, viewer_payload, write_outputs
from replay.metadata import load_metadata


def _data(n: int):
    k = np.arange(n) * 2 + 5  # gaps in k like real pass-1 data
    return SimpleNamespace(k=k, frame=k * 3, t=k / 10.0, signature={"process_fps": 10.0, "video": "v.mp4"})


def _row(data, i, tid=7, interp=False, st=None, sg=None, x=1.0):
    r = TrackRow(frame=int(data.frame[i]), t=float(data.t[i]), k=int(data.k[i]), track_id=tid, object_type="player",
                 team="A", x=x, y=2.0, confidence=0.9, interpolated=interp, calib=KEYFRAME)
    if st is not None:
        r.pos_state, r.pos_sigma = st, sg
    return r


def test_tracks_aligned_with_duplicates_and_unsorted_rows() -> None:
    n = 10
    d = _data(n)
    res = SimpleNamespace(quality=[KEYFRAME] * n, H=[np.eye(3)] * n)
    rows = [_row(d, 8, x=8.0), _row(d, 3, x=3.0), _row(d, 3, interp=True, x=99.0), _row(d, 5, interp=True, x=5.0),
            _row(d, 9, x=9.0), _row(d, 9, x=77.0)]
    p = viewer_payload(rows, d, res, {}, {})
    (tr,) = p["tracks"]
    assert tr["i0"] == 3 and len(tr["x"]) == 7 and tr["i0"] + len(tr["x"]) <= n
    assert tr["x"] == [300, None, 500, None, None, 800, 900]  # measured beats filled; first measured kept
    assert tr["f"] == [1, 0, 2, 0, 0, 1, 1]
    assert "st" not in tr  # no confidence on these rows


def test_confidence_fields_and_csv(tmp_path: Path) -> None:
    n = 6
    d = _data(n)
    res = SimpleNamespace(quality=[KEYFRAME] * n, H=[np.eye(3)] * n)
    rows = [_row(d, 1, st=1, sg=0.12), _row(d, 2, interp=True, st=4, sg=0.756), _row(d, 4, st=1, sg=0.1)]
    write_outputs(tmp_path, rows, d, res, load_metadata(None), {7: {"team": "A"}}, Path("v.mp4"))
    import json
    p = json.loads((tmp_path / "replay_data.json").read_text(encoding="utf-8"))
    (tr,) = p["tracks"]
    assert tr["st"] == [1, 4, 0, 1] and tr["u"] == [12, 76, None, 10]
    with open(tmp_path / "tracking.csv", encoding="utf-8") as fh:
        got = list(csv.DictReader(fh))
    assert list(got[0]) == COLUMNS and got[1]["pos_state"] == "4" and got[1]["pos_sigma_m"] == "0.756"
    assert not list(tmp_path.glob(".*.tmp"))  # atomic writes leave no temporary files


def test_source_column_and_player_id(tmp_path):
    from replay.export import TrackRow, COLUMNS
    r = TrackRow(frame=0, t=0.0, k=0, track_id=7, object_type="player", team="A", x=1.0, y=2.0,
                 confidence=0.8, interpolated=True, calib="propagated")
    r.pos_state, r.pos_sigma = 4, 0.6
    r.player_id, r.id_confidence = 3, 0.63
    assert COLUMNS[-3:] == ["source", "player_id", "id_confidence"]
    assert r.as_list()[-3:] == ["hidden", 3, 0.63]
    r2 = TrackRow(frame=0, t=0.0, k=0, track_id=7, object_type="player", team="A", x=1.0, y=2.0,
                  confidence=0.8, interpolated=False, calib="propagated")
    assert r2.as_list()[-3:] == ["seen", "", ""]


def test_player_id_and_id_confidence_reach_all_outputs(tmp_path):
    import json as _json
    from replay.export import mark_predicted
    n = 6
    d = _data(n)
    res = SimpleNamespace(quality=[KEYFRAME] * n, H=[np.eye(3)] * n)
    rows = [_row(d, 1, tid=7, st=1, sg=0.1), _row(d, 2, tid=7, interp=True, st=4, sg=0.5),
            *mark_predicted([_row(d, 3, tid=7, interp=True, st=5, sg=1.0)]), _row(d, 4, tid=9, st=1, sg=0.1)]
    info = {7: {"team": "A", "player_id": 2, "identity_confidence": 0.8}, 9: {"team": "A", "player_id": 2}}
    write_outputs(tmp_path, rows, d, res, load_metadata(None), info, Path("v.mp4"))
    with open(tmp_path / "tracking.csv", encoding="utf-8") as fh:
        got = list(csv.DictReader(fh))
    assert [g["source"] for g in got] == ["seen", "hidden", "predicted", "seen"]
    assert [g["player_id"] for g in got] == ["2", "2", "2", "2"]
    assert [g["id_confidence"] for g in got] == ["0.8", "0.56", "0.32", ""]
    p = _json.loads((tmp_path / "replay_data.json").read_text(encoding="utf-8"))
    assert {t["id"]: t.get("pid") for t in p["tracks"]} == {7: 2, 9: 2}


def test_lineup_player_ids_follow_slots_and_profiles():
    from replay import lineup
    lineup.SLOT_LABEL.clear(); lineup.CHAIN_SPAN.clear(); lineup.DEAD.clear(); lineup.OUT.clear()
    try:
        lineup.SLOT_LABEL.update({101: "A1", 102: "A1", 103: "A1", 104: "A2"})
        lineup.CHAIN_SPAN.update({101: (0.0, 10.0), 102: (11.0, 20.0), 103: (40.0, 50.0), 104: (0.0, 50.0)})
        lineup.DEAD.append((25.0, 35.0))  # a dead ball between 102 and 103: possible substitution
        new_id = {101: 1, 102: 2, 103: 3, 104: 4}
        info = {o: {"team": "A"} for o in new_id.values()}
        lineup.add_track_info(info, new_id)
        pid = {o: lineup.player_id_of(o) for o in new_id.values()}
        assert pid[1] == pid[2] and pid[3] != pid[1] and pid[4] not in (pid[1], pid[3])
        assert info[1]["player_id"] == pid[1] and lineup.player_id_of(999) is None
    finally:
        lineup.SLOT_LABEL.clear(); lineup.CHAIN_SPAN.clear(); lineup.DEAD.clear(); lineup.PLAYER_ID.clear()
