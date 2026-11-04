"""Preliminary analysis (replay/inplay.py) and break skipping in pass 1."""

import json

import cv2
import numpy as np
import pytest

from replay import inplay
from replay import pipeline
from replay.config import Config, InPlayConfig
from replay.detection import PERSON, Detections
from replay.gui import timeline_model as tm
from replay.metadata import load_metadata


def _signals(activity, pan, fps=2.0):
    n = len(activity)
    z = np.zeros((n, 2, 2), np.uint8)
    return inplay.Signals(t=np.arange(n) / fps, pan=np.asarray(pan, float), activity=np.asarray(activity, float),
                          block_diff=z, block_mean=z, size=(16, 16), fps=fps, lag=2)


def _play_break_play(break_s, fps=2.0, play_s=120.0):
    rng = np.random.default_rng(0)
    n_play, n_brk = int(play_s * fps), int(break_s * fps)
    act = np.r_[rng.uniform(0.15, 0.3, n_play), rng.uniform(0.0, 0.03, n_brk), rng.uniform(0.15, 0.3, n_play)]
    pan = np.r_[rng.uniform(0, 0.02, n_play), np.zeros(n_brk), rng.uniform(0, 0.02, n_play)]
    return _signals(act, pan, fps)


def test_long_quiet_stretch_is_a_break_and_intervals_are_padded():
    cfg = InPlayConfig()
    sig = _play_break_play(150)
    res = inplay.analyse_signals(sig, cfg, 0.0, float(sig.t[-1]) + 0.5)
    assert len(res.intervals) == 2
    (a0, b0), (a1, b1) = res.intervals
    assert a0 == 0.0 and b1 == pytest.approx(390.0, abs=0.6)
    # the break is 120..270 s; detected edges within the smoothing window, plus padding
    assert 120.0 <= b0 <= 120.0 + cfg.window_s / 2 + cfg.pad_s + 1
    assert 270.0 - cfg.window_s / 2 - cfg.pad_s - 1 <= a1 <= 270.0
    assert res.in_play_s < 300


def test_short_quiet_stretch_stays_in_play():
    sig = _play_break_play(25)  # a free throw / slow possession: shorter than min_break_s
    res = inplay.analyse_signals(sig, InPlayConfig(), 0.0, float(sig.t[-1]) + 0.5)
    assert len(res.intervals) == 1


def test_to_intervals_merges_short_gaps():
    play = np.array([1] * 10 + [0] * 10 + [1] * 10 + [0] * 100 + [1] * 10, bool)
    t = np.arange(len(play)) / 2.0
    iv = inplay.to_intervals(play, t, 2.0, pad_s=2.0, merge_gap_s=5.0, start_s=0.0, end_s=70.0)
    # first gap (5 s, minus 4 s of padding) merges; the 50-s gap stays
    assert iv == [(0.0, 17.0), (63.0, 70.0)]


def _frame(i, running, rng_tex, fps=2.0):
    """640x360 grey frame: panning textured scene + a static score bar whose clock ticks once a second."""
    tex = rng_tex
    x0 = (i * 7) % 200
    img = tex[:, x0:x0 + 640].copy()
    cv2.rectangle(img, (220, 320), (460, 345), 30, -1)  # bar
    sec = (i // int(fps)) if running(i) else 1000
    digit = np.full((16, 16), 30, np.uint8)
    cv2.putText(digit, str(sec % 10), (2, 14), cv2.FONT_HERSHEY_PLAIN, 1.0, 240, 1)
    img[324:340, 400:416] = digit
    return img


def test_clock_is_found_and_long_freeze_is_a_break():
    rng = np.random.default_rng(1)
    tex = cv2.GaussianBlur(rng.integers(0, 255, (360, 840), dtype=np.uint8), (5, 5), 0)
    n = 600  # 300 s at 2 fps
    running = lambda i: not (200 <= i < 400)  # noqa: E731  clock frozen 100..200 s
    ex = inplay.SignalExtractor((640, 360), 2.0)
    for i in range(n):
        ex.add(_frame(i, running, tex))
    sig = ex.signals()
    cfg = InPlayConfig()
    info, tick, present = inplay.find_clock(sig, cfg)
    assert info.found
    x0, y0, x1, y1 = info.box_rel
    assert 0.6 <= x0 and x1 <= 0.67 and 0.88 <= y0 and y1 <= 0.96  # the digit at x 400..416, y 324..340
    assert present.mean() > 0.95
    assert tick[20:190].mean() > 0.9 and tick[210:390].mean() < 0.05
    # the scene keeps panning throughout, so only the clock can reveal the break
    res = inplay.analyse_signals(sig, cfg, 0.0, 300.0)
    assert len(res.intervals) == 2
    assert 100 <= res.intervals[0][1] <= 112 and 180 <= res.intervals[1][0] <= 200


def test_no_clock_when_there_is_no_overlay():
    rng = np.random.default_rng(2)
    tex = cv2.GaussianBlur(rng.integers(0, 255, (360, 840), dtype=np.uint8), (5, 5), 0)
    ex = inplay.SignalExtractor((640, 360), 2.0)
    for i in range(80):
        ex.add(tex[:, (i * 7) % 200:(i * 7) % 200 + 640].copy())
    info, tick, present = inplay.find_clock(ex.signals(), InPlayConfig())
    assert not info.found and not tick.any()


# ---------------------------------------------------------------------------- pass-1 integration
def test_play_k_ranges(tmp_path):
    class S:
        process_fps, start_s = 10.0, 0.0

    r = pipeline._play_k_ranges(S, [(0.0, 0.95), (2.0, 2.95), (2.9, 3.5)], 30)
    assert r == [(0, 10), (20, 30)]
    assert pipeline._play_k_ranges(S, None, 30) == [(0, 30)]
    assert pipeline._done_before(r, 25) == 15
    assert pipeline._segment_of(r, 15) is None and pipeline._segment_of(r, 21) == 1


def _textured_video(path, n=90, fps=30.0):
    rng = np.random.default_rng(3)
    tex = cv2.GaussianBlur(rng.integers(0, 255, (240, 480, 3), dtype=np.uint8), (3, 3), 0)
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (320, 240))
    for i in range(n):
        w.write(np.ascontiguousarray(tex[:, i:i + 320]))
    w.release()


class _FakeDetector:
    name, license_note = "fake", "test"

    def detect(self, image):
        return Detections(np.array([[100, 60, 130, 160]], np.float32), np.array([0.9], np.float32),
                          np.array([PERSON], np.int32))


def test_pass1_skips_breaks_and_marks_boundaries(tmp_path, monkeypatch):
    video = tmp_path / "v.avi"
    _textured_video(video)
    monkeypatch.setattr(pipeline, "build_detector", lambda cfg: _FakeDetector())
    cfg = Config(frame_stride=3, chunk_frames=4)
    cfg.motion.min_inliers = 10
    seen = []
    work = pipeline.run_pass1(video, tmp_path / "out", cfg, load_metadata(None),
                              play_intervals=[(0.0, 0.95), (2.0, 2.95)], progress=lambda d, t: seen.append((d, t)))
    data = pipeline.load_pass1(work)
    assert data.k.tolist() == list(range(10)) + list(range(20, 30))
    assert seen[-1] == (20, 20)
    i = data.k.tolist().index(20)
    assert data.motion[0] is None and data.motion[i] is None  # boundary after the skipped span
    assert data.motion[1] is not None and data.motion[i + 1] is not None
    assert set(np.unique(data.rows[:, 0]).astype(int)) <= set(data.k.tolist())
    sig = json.loads((work / "run.json").read_text(encoding="utf-8"))
    assert sig["play_intervals"] == [[0.0, 0.95], [2.0, 2.95]]
    # different intervals = different analysis
    with pytest.raises(SystemExit):
        pipeline.run_pass1(video, tmp_path / "out", cfg, load_metadata(None), play_intervals=[(0.0, 2.0)])


def test_pass1_resume_inside_play_intervals(tmp_path, monkeypatch):
    video = tmp_path / "v.avi"
    _textured_video(video)
    monkeypatch.setattr(pipeline, "build_detector", lambda cfg: _FakeDetector())
    cfg = Config(frame_stride=3, chunk_frames=4)
    cfg.motion.min_inliers = 10
    iv = [(0.0, 0.95), (2.0, 2.95)]
    ev = __import__("threading").Event()

    def prog(done, total):
        if done == 12:
            ev.set()

    with pytest.raises(pipeline.Cancelled):
        pipeline.run_pass1(video, tmp_path / "out", cfg, load_metadata(None), play_intervals=iv, progress=prog,
                           cancel=ev)
    work = pipeline.run_pass1(video, tmp_path / "out", cfg, load_metadata(None), play_intervals=iv)
    data = pipeline.load_pass1(work)
    assert data.k.tolist() == list(range(10)) + list(range(20, 30))
    assert data.motion[data.k.tolist().index(20)] is None


def test_intervals_for_pass1_keeps_old_whole_video_runs(tmp_path):
    out = tmp_path / "out"
    (out / "work").mkdir(parents=True)
    (out / "work" / "run.json").write_text(json.dumps({"pipeline_version": 1}), encoding="utf-8")
    assert inplay.intervals_for_pass1(tmp_path / "missing.mp4", out, InPlayConfig(), log=lambda m: None) is None
    (out / "work" / "run.json").write_text(json.dumps({"play_intervals": [[1, 2]]}), encoding="utf-8")
    assert inplay.intervals_for_pass1(tmp_path / "missing.mp4", out, InPlayConfig(), log=lambda m: None) == [(1, 2)]
    assert inplay.intervals_for_pass1(tmp_path / "x.mp4", out, InPlayConfig(enabled=False)) is None


def test_save_and_cache_roundtrip(tmp_path):
    video = tmp_path / "v.avi"
    _textured_video(video, n=10)
    res = inplay.InPlayResult([(1.0, 5.0)], 0.0, 10.0, inplay.ClockInfo(False), 0.04, 0.2)
    cfg = InPlayConfig()
    inplay.save_result(res, tmp_path / "inplay.json", video, cfg)
    assert inplay.load_intervals(tmp_path / "inplay.json") == [(1.0, 5.0)]
    assert inplay.cached_result(tmp_path / "inplay.json", video, cfg) == [(1.0, 5.0)]
    assert inplay.cached_result(tmp_path / "inplay.json", video, InPlayConfig(pad_s=1.0)) is None


def test_preliminary_on_real_decode(tmp_path):
    video = tmp_path / "v.avi"
    _textured_video(video, n=90)
    res = inplay.run_preliminary(video, InPlayConfig())
    assert res.intervals and res.intervals[0][0] == 0.0
    assert len(res.trace["activity"]) >= 2


def test_timeline_not_in_play_track():
    tl = tm.Timeline.default()
    assert tl.set_inplay([(10.0, 70.0), (100.0, 160.5)], span=(0.0, 200.0))
    tr = tl.track(tm.DEAD_TRACK_ID)
    assert tr.locked and tr.name == "Ball not in play"
    assert [(m.t, m.duration) for m in tr.marks] == [(0.0, 10.0), (70.0, 30.0), (160.5, 39.5)]
    assert tl.tracks.index(tr) == 1
    assert tl.track(tm.INPLAY_TRACK_ID) is None
    assert not tl.set_inplay([(10.0, 70.0), (100.0, 160.5)], span=(0.0, 200.0))
    tl2 = tm.Timeline.from_dict(json.loads(json.dumps(tl.to_dict())))
    assert len(tl2.track(tm.DEAD_TRACK_ID).marks) == 3
    assert tl2.set_inplay(None) and tl2.track(tm.DEAD_TRACK_ID) is None


def test_frozen_clock_during_clear_play_is_not_a_break():
    """Operator stops the clock but play goes on (busy floor + panning camera): still play. A frozen clock with a
    quiet floor and a still camera stays a break."""
    fps, n = 2.0, 1200
    rng = np.random.default_rng(3)
    act = rng.uniform(0.15, 0.3, n)
    pan = rng.uniform(0.0, 0.02, n)
    pan[rng.random(n) < 0.4] = 0.01  # camera panning in ~40 % of the samples
    act[800:1000] = rng.uniform(0.0, 0.03, 200)  # 100 s real break: quiet floor, still camera
    pan[800:1000] = 0.0
    sig = _signals(act, pan, fps)
    tick = np.ones(n, bool)
    tick[300:600] = False  # clock frozen 150..300 s during play
    tick[780:1000] = False  # ... and in the real break
    present = np.ones(n, bool)
    play, dbg = inplay.classify(sig, InPlayConfig(), tick, present)
    assert play[320:580].all()
    assert not play[820:980].any()
