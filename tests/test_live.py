"""Live mode (replay/live.py): growing files, local HLS, incremental pass 1, merging of pass-2 windows.

Everything is local: a synthetic clip is made with the bundled ffmpeg and 'recorded' into a temp file / HLS
folder in pieces (simulate_recording). The detector is replaced by a cheap fake one."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from replay import live
from replay.detection import BALL, PERSON
from replay.detection.base import Detections

ROOT = Path(__file__).resolve().parent.parent
C300 = ROOT / "data" / "clips" / "c300"


def _ffmpeg() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


@pytest.fixture(scope="module")
def clip(tmp_path_factory) -> Path:
    """12 s synthetic H.264 clip, 320x240 @ 30 fps, a keyframe every second."""
    p = tmp_path_factory.mktemp("liveclip") / "clip.mp4"
    subprocess.run([_ffmpeg(), "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-f", "lavfi", "-i",
                    "testsrc2=size=320x240:rate=30", "-t", "12", "-c:v", "libx264", "-g", "30", "-pix_fmt",
                    "yuv420p", str(p)], check=True)
    return p


@pytest.fixture(scope="module")
def segments(clip, tmp_path_factory):
    return live.make_segments(clip, tmp_path_factory.mktemp("segs"), 1.0)


class FakeDetector:
    """One 'person' walking left to right and a ball; deterministic from the image content."""

    name = "fake"
    license_note = "test"

    def __init__(self):
        self.calls = 0

    def detect(self, img):
        self.calls += 1
        h, w = img.shape[:2]
        x = (self.calls * 2) % (w - 40)
        return Detections(np.array([[x, 60, x + 30, 200], [100, 50, 108, 58]], np.float32),
                          np.array([0.9, 0.6], np.float32), np.array([PERSON, BALL], np.int32))


@pytest.fixture
def fake_detector(monkeypatch):
    import replay.detection as det

    made = []

    def build(_cfg):
        d = FakeDetector()
        made.append(d)
        return d

    monkeypatch.setattr(det, "build_detector", build)
    return made


def _cfg(chunk: int = 25):
    from replay.config import Config

    cfg = Config()
    cfg.chunk_frames = chunk
    cfg.inplay.enabled = False
    return cfg


def _grow(segments, dst: Path, upto: int, parts_done: int = 0) -> None:
    """Append segments[:upto] to dst (synchronously)."""
    with open(dst, "ab") as fh:
        for p, _a, _b in segments[parts_done:upto]:
            fh.write(p.read_bytes())


# ------------------------------------------------------------------------------------------------- helpers
def test_write_atomic_replaces_and_leaves_no_temp(tmp_path):
    f = tmp_path / "a" / "x.json"
    live.write_atomic(f, "1")
    live.write_atomic(f, "2")
    assert f.read_text(encoding="utf-8") == "2"
    assert [p.name for p in f.parent.iterdir()] == ["x.json"]


def test_growing_file_detects_growth_idle_and_restart(tmp_path):
    f = tmp_path / "rec.ts"
    g = live.GrowingFile(f)
    assert g.poll() == "missing"
    f.write_bytes(b"x" * 100)
    assert g.poll() == "grew"
    assert g.poll() == "same"
    with open(f, "ab") as fh:
        fh.write(b"y" * 10)
    assert g.poll() == "grew"
    f.write_bytes(b"z" * 5)  # OBS started a new recording into the same file name
    assert g.poll() == "restarted"
    assert g.idle_s < 5


def test_hls_folder_assembles_complete_local_segments(tmp_path, segments):
    folder = tmp_path / "hls"
    folder.mkdir()
    out = tmp_path / "out"
    src = live.make_source(folder / "live.m3u8", out)
    assert isinstance(src, live.HlsFolder)
    listed = []
    for p, a, b in segments[:3]:
        shutil.copy(p, folder / p.name)
        listed.append((p.name, b - a))
    live._write_playlist(folder / "live.m3u8", listed, ended=False)
    assert src.poll() == "grew"
    assert src.video.stat().st_size == sum(p.stat().st_size for p, _a, _b in segments[:3])
    assert src.poll() == "same" and not src.ended
    for p, a, b in segments[3:]:
        shutil.copy(p, folder / p.name)
        listed.append((p.name, b - a))
    live._write_playlist(folder / "live.m3u8", listed, ended=True)
    assert src.poll() == "grew" and src.ended
    # the assembled stream decodes to the full length
    from replay.video import probe

    assert probe(src.video).duration_s == pytest.approx(12.0, abs=0.6)
    # resumable: a new object continues after the segments already appended
    again = live.HlsFolder(folder, src.video)
    assert again.poll() == "same" and again.ended


def test_hls_refuses_urls(tmp_path):
    (tmp_path / "live.m3u8").write_text("#EXTM3U\n#EXTINF:2,\nhttps://example.invalid/seg1.ts\n", encoding="utf-8")
    src = live.HlsFolder(tmp_path / "live.m3u8", tmp_path / "out" / "stream.ts")
    with pytest.raises(ValueError, match="URL"):
        src.poll()


# ------------------------------------------------------------------------------------------------- merging
def _doc(n, t0, tracks, events=(), info=None):
    return {"game": {}, "fps": 10.0, "t": [round(t0 + i / 10, 3) for i in range(n)],
            "frame": [int(round((t0 + i / 10) * 30)) for i in range(n)], "calib": [1] * n,
            "court": {"length": 26.0}, "tracks": tracks,
            "ball": {"x": [1] * n, "y": [2] * n, "f": [1] * n, "b": [None] * n, "o": [None] * n},
            "track_info": info or {str(tr["id"]): {"team": tr["team"]} for tr in tracks},
            "events": list(events), "n": n}


def _tr(tid, i0, xs, team="A"):
    return {"id": tid, "team": team, "type": "player", "i0": i0, "x": list(xs), "y": [500] * len(xs),
            "f": [1] * len(xs), "b": [None] * len(xs)}


def test_merger_freezes_past_links_seam_and_keeps_ids():
    t_all = [round(i / 10, 3) for i in range(100)]
    fr_all = [int(round(t * 30)) for t in t_all]
    m = live.LiveMerger()
    # first window: samples 0..59, one player walking, one referee
    d1 = m.merge(_doc(60, 0.0, [_tr(1, 0, range(1000, 1060)), _tr(2, 10, [700] * 50, "referee")],
                      events=[{"id": "e1", "t": 2.0, "type": "basket"}]), 0, t_all, fr_all, margin=10)
    assert d1["n"] == 60 and [tr["id"] for tr in d1["tracks"]] == [1, 2]
    # second window starts at sample 30 (+ margin 10 = frozen up to 40): window ids are new numbers (7, 9)
    w = _doc(70, 3.0, [_tr(7, 0, range(1030, 1100)),  # same walker -> continues id 1 at the seam
                       _tr(9, 15, [300] * 20, "B")],  # someone new -> next free id
             events=[{"id": "old", "t": 3.5, "type": "x"}, {"id": "e2", "t": 8.0, "type": "basket",
                                                            "details": {"shooter_track": 7}}])
    d2 = m.merge(w, 30, t_all, fr_all, margin=10)
    assert d2["n"] == 100 and d2["t"] == t_all and d2["frame"] == fr_all
    by = {tr["id"]: tr for tr in d2["tracks"]}
    assert set(by) == {1, 2, 3}
    walker = by[1]
    assert walker["i0"] == 0 and len(walker["x"]) == 100
    assert walker["x"][:40] == list(range(1000, 1040))  # frozen part as shown before
    assert walker["x"][40:] == list(range(1040, 1100))  # from the new window
    assert by[2]["i0"] == 10 and len(by[2]["x"]) == 30  # referee: only its frozen part (ended at 40)
    assert by[3]["i0"] == 45 and by[3]["team"] == "B"
    assert d2["track_info"]["3"]["team"] == "B" and "9" not in d2["track_info"]
    assert [e["id"] for e in d2["events"]] == ["e1", "e2"]  # the window's event before the seam is dropped
    assert d2["events"][1]["details"]["shooter_track"] == 1
    assert d2["ball"]["x"] == [1] * 100
    # every per-sample list of every track has the same length
    for tr in d2["tracks"]:
        assert len({len(v) for v in tr.values() if isinstance(v, list)}) == 1


def test_merger_fills_gap_when_window_starts_after_document():
    t_all = [round(i / 10, 3) for i in range(80)]
    fr_all = list(range(80))
    m = live.LiveMerger()
    m.merge(_doc(20, 0.0, [_tr(1, 0, range(20))]), 0, t_all, fr_all, margin=5)
    d = m.merge(_doc(30, 5.0, [_tr(4, 0, range(30))]), 50, t_all, fr_all, margin=5)
    assert d["n"] == 80 and d["t"] == t_all
    assert d["calib"][20:50] == [0] * 30 and d["ball"]["x"][20:50] == [None] * 30
    ids = sorted(tr["id"] for tr in d["tracks"])
    assert ids == [1, 2] and next(tr for tr in d["tracks"] if tr["id"] == 2)["i0"] == 50


def test_merger_starts_over_when_the_analysis_changed():
    m = live.LiveMerger(_doc(30, 100.0, [_tr(5, 0, range(30))]))
    t_all = [round(i / 10, 3) for i in range(30)]
    d = m.merge(_doc(30, 0.0, [_tr(1, 0, range(30))]), 0, t_all, list(range(30)), margin=5)
    assert d["t"] == t_all and [tr["id"] for tr in d["tracks"]] == [1]


# ------------------------------------------------------------------------------------------------- pass 1
def test_incremental_pass1_on_growing_file_matches_one_pass(tmp_path, segments, fake_detector, clip):
    """Pass 1 run in steps on a growing, truncated file commits each sample exactly once, never the newest
    second (half-written frames), and ends with the same sample grid as the finished file."""
    from replay.metadata import metadata_from_dict
    from replay.pipeline import load_pass1
    from replay.video import VideoSampler

    rec = tmp_path / "rec.ts"
    _grow(segments, rec, 4)
    data = rec.read_bytes()
    rec.write_bytes(data[: len(data) - 3000])  # the writer is in the middle of a frame
    out = tmp_path / "out"
    cfg = _cfg()
    p1 = live.LivePass1(out, rec, cfg, metadata_from_dict({}), print)
    p1.write_signature(rec, live=True)
    n1 = p1.step(rec, 60, 1.0)
    assert n1 > 0 and p1.t_last < segments[3][2] - 1.0 + 0.15  # newest second held back
    with open(rec, "ab") as fh:
        fh.write(data[len(data) - 3000:])
    _grow(segments, rec, 8, 4)
    n2 = p1.step(rec, 60, 1.0)
    _grow(segments, rec, len(segments), 8)
    n3 = p1.step(rec, 60, 1.0, final=True)
    assert n2 > 0 and n3 > 0
    d = load_pass1(out / "work")
    ref = [s.k for s in VideoSampler(rec, p1.fps, 0.0, 1e12).iter_samples(0)]
    assert d.k.tolist() == ref == sorted(set(ref))
    assert len(p1.motions) == len(d.k) == p1.n_pos
    assert len(fake_detector) == 1  # detector loaded once
    # a new object resumes from the saved state
    p1b = live.LivePass1(out, rec, cfg, metadata_from_dict({}), print)
    assert p1b.n_pos == p1.n_pos and p1b.next_k == p1.next_k
    assert p1b.step(rec, 60, 1.0, final=True) == 0


def test_pass1_skips_breaks_and_marks_the_gap(tmp_path, segments, fake_detector):
    from replay.metadata import metadata_from_dict
    from replay.pipeline import load_pass1

    rec = tmp_path / "rec.ts"
    _grow(segments, rec, len(segments))
    p1 = live.LivePass1(tmp_path / "out", rec, _cfg(), metadata_from_dict({}), print)
    p1.write_signature(rec, live=True)
    p1.step(rec, 60, 1.0, final=True, is_play=lambda t: not (4.0 <= t < 7.0))
    d = load_pass1(tmp_path / "out" / "work")
    assert not np.any((d.t >= 4.0) & (d.t < 7.0))
    first_after = int(np.argmax(d.t >= 7.0))
    assert d.motion[first_after] is None  # camera chain broken over the skipped break (as in run_pass1)


# ------------------------------------------------------------------------------------------------- session
def _lcfg(**kw):
    lc = live.LiveConfig(poll_s=0.2, min_new_s=0.5, idle_end_s=2.5, holdback_s=1.0, skip_breaks=False,
                         final_build=False, auto_calibrate=False)
    for k, v in kw.items():
        setattr(lc, k, v)
    return lc


@pytest.mark.parametrize("hls", [False, True])
def test_live_session_follows_recording_and_ends_as_normal_analysis(tmp_path, segments, fake_detector, hls):
    """A recording written in real time (x4) is analysed while it grows; when it stops the folder is an ordinary
    finished pass 1 that run_pass1 accepts as complete (no re-analysis, no 'different settings' error)."""
    from replay.metadata import metadata_from_dict
    from replay.pipeline import load_pass1, run_pass1

    dst = tmp_path / ("hls" if hls else "rec.ts")
    out = tmp_path / "out"
    log = tmp_path / "rec.jsonl"
    seen = []
    th = live.simulate_recording(segments, dst, speed=4.0, hls=hls, log_path=log)
    src = dst / "live.m3u8" if hls else dst
    cfg = _cfg()

    def watch():
        while th.is_alive():
            st = live.read_status(out)
            if st:
                seen.append(st)
            time.sleep(0.1)

    w = threading.Thread(target=watch, daemon=True)
    w.start()
    summary = live.run_live(src, out, cfg, None, None, _lcfg(idle_end_s=1.5 if hls else 2.5))
    th.join(5)
    assert summary["samples"] > 0 and not summary["restarted"]
    assert any(s["state"] == "analysing" and 0 < s["analysed_t"] < 11 for s in seen)  # worked while growing
    assert (out / "work" / "DONE").exists() and (out / "live" / "FINISHED").exists()
    assert not live.is_live(out)
    video = out / "live" / "stream.ts" if hls else dst
    d = load_pass1(out / "work")
    assert d.t[-1] > 11.0 and len(d.k) == len(set(d.k.tolist()))
    sig = json.loads((out / "work" / "run.json").read_text(encoding="utf-8"))
    assert "live" not in sig and sig["video"] == str(video.resolve())
    run_pass1(video, out, cfg, metadata_from_dict({}))  # accepted as complete: returns without analysing anything
    assert len(fake_detector) == 1


def test_live_session_stop_file_finishes(tmp_path, segments, fake_detector):
    dst = tmp_path / "rec.ts"
    _grow(segments, dst, 6)
    out = tmp_path / "out"
    res = {}

    def run():
        res["s"] = live.run_live(dst, out, _cfg(), None, None, _lcfg(idle_end_s=600))

    th = threading.Thread(target=run, daemon=True)
    th.start()
    end = time.time() + 60
    while time.time() < end and not (live.read_status(out) or {}).get("samples"):
        time.sleep(0.1)
    assert live.is_running(out)
    live.request_stop(out)
    th.join(60)
    assert not th.is_alive() and res["s"]["samples"] > 0
    assert (out / "work" / "DONE").exists() and not (out / "live" / "STOP").exists()
    assert (live.read_status(out) or {}).get("state") == "finished"


def test_reuse_court_keeps_camera_and_court_but_no_anchors(tmp_path):
    src = tmp_path / "a.json"
    src.write_text(json.dumps({"video": "v.mp4", "frame_size": [1280, 720], "keyframes": [],
                               "court": {"length": 26.0, "width": 14.5, "spec": "FIBA"},
                               "auto": {"C": [13, -1.7, 1.9], "line_classes": [0, 1], "anchors": [
                                   {"t": 1.0, "good": True, "C": [12.9, -1.7, 1.94]}]}}), encoding="utf-8")
    live.reuse_court(src, tmp_path / "b.json")
    d = json.loads((tmp_path / "b.json").read_text(encoding="utf-8"))
    assert d["auto"]["anchors"] == [] and d["auto"]["C"] == [12.9, -1.7, 1.94]
    assert d["court"]["length"] == 26.0 and d["auto"]["line_classes"] == [0, 1]


@pytest.mark.skipif(not (C300 / "out" / "work" / "DONE").exists() or os.environ.get("RUN_SLOW") != "1",
                    reason="needs data/clips/c300; slow (~50 s): RUN_SLOW=1")
def test_window_pass2_on_real_clip_merges_into_replay_data(tmp_path):
    """Pass 2 on a trailing window of a real analysed clip, twice, merged: one consistent replay_data.json."""
    from replay.config import Config

    out = tmp_path / "out"
    (out / "work").mkdir(parents=True)
    src = C300 / "out" / "work"
    for name in ("chunk_00000.npz", "chunk_00001.npz", "run.json"):
        shutil.copy(src / name, out / "work" / name)
    cfg = Config.from_dict(json.loads((C300 / "out" / "config.used.json").read_text(encoding="utf-8")))
    calib = C300 / "clip.calibration.json"
    lc = live.LiveConfig(window_s=25.0, freeze_margin_s=5.0)
    merger, cache = live.LiveMerger(), {}
    (out / "live").mkdir()
    a = live.window_pass2(out, cfg, None, calib, C300 / "clip.mp4", lc, merger, cache)
    shutil.copy(src / "chunk_00002.npz", out / "work" / "chunk_00002.npz")
    b = live.window_pass2(out, cfg, None, calib, C300 / "clip.mp4", lc, merger, cache)
    assert a and b and b["t_end"] > a["t_end"] + 25
    doc = json.loads((out / "replay_data.json").read_text(encoding="utf-8"))
    assert doc["n"] == len(doc["t"]) == len(doc["calib"]) == len(doc["ball"]["x"]) == 900
    assert doc["t"] == sorted(doc["t"]) and doc["live"]["t_end"] == doc["t"][-1]
    assert sum(1 for tr in doc["tracks"] if tr["i0"] + len(tr["x"]) > 600) >= 5  # people in the newest part
    for tr in doc["tracks"]:
        assert 0 <= tr["i0"] and tr["i0"] + len(tr["x"]) <= doc["n"]
    lines = (out / "live" / "updates.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert not list((out / "live" / "window" / "work").glob("jersey_numbers.npz"))  # fast mode: no video steps


# ------------------------------------------------------------------------------------------------- GUI
def _live_replay_doc(n: int) -> dict:
    return {"game": {"teams": {}}, "fps": 10.0, "t": [round(i / 10, 3) for i in range(n)],
            "frame": [i * 3 for i in range(n)], "calib": [1] * n,
            "court": {"length": 26, "width": 14, "lines": [], "baskets": [[1.575, 7.0], [24.425, 7.0]]},
            "tracks": [_tr(1, 0, range(500, 500 + n))], "ball": None, "track_info": {}, "events": [], "n": n,
            "live": {"t_end": (n - 1) / 10}}


def test_gui_live_menu_status_and_reload_keeps_playhead(tmp_path, monkeypatch):
    """Live Replays: menu entry, status line, the normal job chain leaves them alone, and new 2D data is
    reloaded while the Replay tab stays where it was."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication

    from replay.gui import app as A
    from replay.gui import jobs as J

    QApplication.instance() or QApplication([])
    monkeypatch.setattr(A, "QSettings", lambda *a: QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat))
    started = []
    from replay.gui import live_gui

    monkeypatch.setattr(live_gui, "start_live", lambda p: started.append(p.id))
    win = A.MainWindow(tmp_path / "projects")
    try:
        p = win.store.create("Live game", {"filename": "rec.ts", "kind": "reference", "path": str(tmp_path / "rec.ts")})
        p.update_info(live={"source": str(tmp_path / "rec.ts")})
        ld = live.live_dir(p.results)
        ld.mkdir(parents=True)
        (ld / live.MARKER).write_text("1", encoding="utf-8")
        (ld / live.STATUS).write_text(json.dumps({"state": "analysing", "heartbeat": time.time(), "analysed_t": 75.0,
                                                  "replay_t": 40.0, "behind_s": 35.0, "calibration": ""}),
                                      encoding="utf-8")
        (p.results / "work").mkdir()
        (p.results / "work" / "run.json").write_text("{}", encoding="utf-8")
        live.write_atomic(p.results / "replay_data.json", json.dumps(_live_replay_doc(400)))
        # the job chain must not pick the live Replay up (it would re-analyse / re-import the growing file)
        assert win.jobs.resume_interrupted() == []
        with pytest.raises(J.JobError, match="live"):
            win.jobs._run_analyze(p, J.JobState())
        assert any(a.text() == "&Live" for a in win.menuBar().actions())
        win.show_project(p, win.tabs.indexOf(win.replay_tab))
        ctl = win.live
        ctl.follow.setChecked(False)
        ctl.tick()
        assert "LIVE" in ctl.label.text() and "1:15.0" in ctl.label.text()
        tab = win.replay_tab
        assert tab.data is not None and tab.data["n"] == 400
        # new data arrives: reloaded on the next tick, the playhead is not moved (not following)
        time.sleep(0.05)
        live.write_atomic(p.results / "replay_data.json", json.dumps(_live_replay_doc(700)))
        os.utime(p.results / "replay_data.json", None)
        before = tab.player.position()
        ctl.tick()
        assert tab.data["n"] == 700 and tab.player.position() == before
        # the toggle on a running live Replay asks it to stop
        ctl.fill_menu()
        toggle = next(a for a in ctl.menu.actions() if a.text() == "Live analysis of this Replay")
        assert toggle.isEnabled() and toggle.isChecked()
        toggle.setChecked(False)
        assert (ld / live.STOP_FILE).exists()
        # finished: an ordinary Replay again; the full build is queued once
        submitted = []
        monkeypatch.setattr(win.jobs, "submit", lambda pid, kind, **kw: submitted.append(kind))
        (ld / live.MARKER).unlink()
        (p.results / "work" / "DONE").write_text("x", encoding="utf-8")
        (ld / live.STATUS).write_text(json.dumps({"state": "finished", "heartbeat": time.time()}), encoding="utf-8")
        live.write_atomic(p.calibration_path, json.dumps({"video": "v", "frame_size": [1280, 720], "keyframes": [],
                                                          "auto": {"C": [1, 2, 3], "anchors": [{"t": 1, "good": True}]}}))
        ctl.tick()
        ctl.tick()
        assert submitted == [J.BUILD] and "finished" in ctl.label.text()
        assert not live.owns_replay(p)
    finally:
        win.live.timer.stop()
        win.timer.stop()
        win.close()
