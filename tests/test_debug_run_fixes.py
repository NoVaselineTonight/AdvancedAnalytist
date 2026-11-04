"""Fixes from the pass-2 debug run on the external clips (labeller B / data agent, 20.10)."""
import argparse
import json

import pytest


def test_postprocess_without_calibration_exits_with_a_message(tmp_path):
    """ncaa seg_2965: the automatic calibration found no court -> postprocess crashed with FileNotFoundError."""
    from replay import cli
    from replay.config import Config

    out = tmp_path / "out"
    (out / "work").mkdir(parents=True)
    (out / "config.used.json").write_text(json.dumps(Config().to_dict()), encoding="utf-8")
    (out / "work" / "run.json").write_text(json.dumps({"video": str(tmp_path / "clip.mp4")}), encoding="utf-8")
    a = argparse.Namespace(out=str(out), meta=None, calibration=str(tmp_path / "missing.json"), config=None)
    with pytest.raises(SystemExit) as e:
        cli.cmd_postprocess(a)
    assert "No court calibration" in str(e.value)


def test_no_person_row_is_exported_far_off_court_unflagged():
    """hs_hazen seg_2700: 6.8 % of player rows 2+ m outside the court (shifted-calibration frames kept on purpose)."""
    from replay.export import SOURCES, TrackRow, clamp_off_court, source_of

    def row(x, y, kind="player"):
        return TrackRow(frame=0, t=0.0, k=0, track_id=1, object_type=kind, team="A" if kind == "player" else "",
                        x=x, y=y, confidence=0.9, interpolated=False, calib="propagated")

    rows = [row(-3.3, 8.7), row(27.5, -2.0), row(10.0, 5.0), row(-0.5, 15.4), row(-5.0, 5.0, "ball")]
    assert clamp_off_court(rows, 26.0, 14.5) == 2
    assert (rows[0].x, rows[0].y) == (-1.0, 8.7) and (rows[1].x, rows[1].y) == (27.0, -1.0)
    assert source_of(rows[0]) == "clamped" and "clamped" in SOURCES
    assert source_of(rows[2]) == "seen" and source_of(rows[3]) == "seen"  # within 1 m: untouched
    assert rows[4].x == -5.0  # the ball is not a person
    for r in rows[:4]:
        assert -1.0 <= r.x <= 27.0 and -1.0 <= r.y <= 15.5


def test_off_court_person_rows_are_moved_to_the_edge_and_flagged():
    """hs_hazen seg_2700: 6.8 % of player rows > 2 m off the court (shifted-calibration frames kept by the limits)."""
    from replay.export import TrackRow, clamp_off_court, source_of

    def row(x, y, kind="player"):
        return TrackRow(frame=0, t=0.0, k=0, track_id=1, object_type=kind, team="A" if kind == "player" else "",
                        x=x, y=y, confidence=0.9, interpolated=False, calib="propagated")

    rows = [row(-3.3, 8.7), row(27.5, -0.5), row(10.0, 5.0), row(-0.8, 15.6), row(-5.0, 5.0, "ball")]
    assert clamp_off_court(rows, 26.0, 15.0) == 2
    assert (rows[0].x, rows[0].y) == (-1.0, 8.7) and source_of(rows[0]) == "clamped"
    assert (rows[1].x, rows[1].y) == (27.0, -0.5)
    assert not rows[2].clamped and not rows[3].clamped and source_of(rows[2]) == "seen"
    assert rows[4].x == -5.0  # the ball is not a person position
    assert rows[0].as_list()[-3] == "clamped"
