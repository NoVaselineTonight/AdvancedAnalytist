"""Fixes for review_9_10 findings 1, 3, 4, 5 (scorebar static rule + blank-start retry, export clamp, autocal_tap)."""
from __future__ import annotations

import numpy as np
import pytest

from replay import scorebar as sb
from replay.export import TrackRow, clamp_off_court, source_of


def _reader_with(monkeypatch, texts):
    """Reader whose locate finds one block per entry of ``texts`` (list of per-sample text lists)."""
    cv2 = pytest.importorskip("cv2")
    rd = sb.ScorebarReader(sb.ScorebarConfig(), object())
    blocks = [sb.Block(10 + 40 * i, 10, 40 + 40 * i, 30) for i in range(len(texts))]

    def fake_locate(imgs, cfg, **kw):
        return blocks, None

    def fake_read(rec, frame, live, t):
        for b in live:
            b.reads.append((t, texts[blocks.index(b)][len(b.reads) % len(texts[0])], 0.9))

    monkeypatch.setattr(sb, "locate_blocks", fake_locate)
    monkeypatch.setattr(sb, "read_blocks", fake_read)
    return rd, blocks, np.zeros((40, 400, 3), np.uint8)


def test_frozen_clock_and_period_are_not_static(monkeypatch):
    n = 30
    # block 0/1: scores (change), 2: frozen clock "10:00", 3: period "Q1", 4: team name (no digits -> static)
    texts = [["1"] * 15 + ["2"] * 15, ["3"] * 15 + ["4"] * 15, ["10:00"] * n, ["Q1"] * n, ["LAKERS"] * n]
    rd, blocks, img = _reader_with(monkeypatch, texts)
    for k in range(rd.locate_n):
        rd.add(2.0 * k, img)
    assert rd.blocks is not None
    assert rd.static == {4}, rd.static  # only the digit-free team name may be skipped
    # every following sample reads the clock and the period
    rd.add(99.0, img)  # 30 samples so far: a full-read sample
    before = [len(blocks[i].reads) for i in range(5)]
    rd.add(100.0, img)
    assert [len(blocks[i].reads) - before[i] for i in range(5)] == [1, 1, 1, 1, 0]


def test_blank_first_minute_retries_later_in_the_video():
    cv2 = pytest.importorskip("cv2")

    class NoRec:
        device = "cpu"

        def __call__(self, *a, **k):
            return []

    rd = sb.ScorebarReader(sb.ScorebarConfig(), NoRec())
    flat = np.full((120, 200, 3), 90, np.uint8)
    for k in range(rd.locate_n):
        rd.add(2.0 * k, flat)
    assert not rd.no_bar and rd.blocks is None and rd.wake > 60
    n_buf = len(rd.buf)
    rd.add(rd.wake - 1.0, flat)  # asleep: ignored
    assert len(rd.buf) == n_buf == 0
    rd.add(rd.wake + 1.0, flat)  # awake again: collecting for the next locate
    assert len(rd.buf) == 1


def _row(x, y, team="A", kind="player", calib="keyframe", **kw):
    return TrackRow(frame=0, t=0.0, k=0, track_id=1, object_type=kind, team=team, x=x, y=y, confidence=0.9,
                    interpolated=False, calib=calib, box=(1.0, 2.0, 3.0, 4.0), **kw)


def test_clamp_only_calibration_failures_keeps_source_and_uses_project_court():
    thrower = _row(-2.0, 5.0)  # inbound passer 2 m behind the baseline, good calibration: untouched
    ref = _row(-6.0, 5.0, team="referee", kind="referee", calib="propagated")
    other = _row(-6.0, 5.0, team="unknown", calib="propagated")
    nan = _row(float("nan"), 5.0, calib="propagated")
    bad_calib = _row(-2.0, 5.0, calib="propagated", pos_sigma=0.2)
    far = _row(40.0, 5.0)  # >3 m outside even with a keyframe
    rows = [thrower, ref, other, nan, bad_calib, far]
    assert clamp_off_court(rows, 28.0, 15.0) == 2
    assert (thrower.x, ref.x, other.x) == (-2.0, -6.0, -6.0) and not thrower.clamped
    assert not nan.clamped and np.isnan(nan.x)
    assert bad_calib.clamped and bad_calib.x == -1.0 and bad_calib.clamped_from == "seen"
    assert source_of(bad_calib) == "clamped" and bad_calib.box is None and bad_calib.pos_sigma >= 1.0
    assert far.x == 29.0  # 28 m court + 1 m margin (project dimensions, not the module constant)
    # default dimensions come from the project court
    from replay import court
    L, W = court.dimensions()
    r = _row(L + 10.0, 5.0)
    assert clamp_off_court([r]) == 1 and r.x == L + 1.0


def test_autocal_finish_does_not_block_on_pool(monkeypatch):
    import inspect
    from replay import autocal_tap
    src = inspect.getsource(autocal_tap.AutocalTap.finish)
    assert "shutdown(wait=True" not in src
    assert src.count("shutdown(wait=False") == 2
