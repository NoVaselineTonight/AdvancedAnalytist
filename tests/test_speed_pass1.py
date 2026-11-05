"""speed-pass1 (swarm 05.11): pass-1 waits removed without changing outputs."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from replay import scorebar as sb


def _frames(n=40):
    cv2 = pytest.importorskip("cv2")
    rng = np.random.default_rng(1)
    out = []
    for k in range(n):
        f = cv2.GaussianBlur(rng.integers(0, 255, (360, 640, 3), dtype=np.uint8), (9, 9), 0)
        cv2.rectangle(f, (200, 300), (440, 340), (20, 20, 20), -1)
        cv2.putText(f, f"KLA  {15 + k // 8}  {59 + k // 12}", (210, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2)
        out.append((k * 2.0, f))
    return out


def _run_tap(work: Path, frames, monkeypatch, defer: str) -> None:
    monkeypatch.setenv("REPLAY_SCOREBAR_DEFER", defer)
    tap = sb.ScorebarTap(every_s=2.0)
    tap.start(SimpleNamespace(resumed=False, work=work))
    for k, (t, f) in enumerate(frames):
        if tap.wants(k, t):
            tap.process(SimpleNamespace(t=t, image=f))
    tap.finish(work, True)
    return tap


def _norm(doc: dict) -> str:
    doc = dict(doc)
    doc.pop("stats", None)  # timings
    return json.dumps(doc, sort_keys=True, default=str)


def test_deferred_scorebar_finish_gives_the_same_reading(tmp_path, monkeypatch):
    pytest.importorskip("onnxruntime")
    if not sb.Path(__import__("replay.jersey_numbers", fromlist=["x"]).MODEL_PATH).exists():
        pytest.skip("OCR model missing")
    frames = _frames()
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    _run_tap(a, frames, monkeypatch, "0")
    inline = sb.load_result(a)
    assert inline is not None and not (a / sb.PENDING).exists()
    tap = _run_tap(b, frames, monkeypatch, "1")
    assert tap.deferred  # pass 1 did not wait for the end-of-video work
    deferred = sb.load_or_compute(b, None)  # pass 2: waits for the background finisher
    assert deferred is not None and _norm(deferred) == _norm(inline)
    assert not (b / sb.PENDING).exists()
    for _ in range(100):  # the finisher writes its log just after the reading
        if (b / "scorebar_finish.log").exists():
            break
        __import__("time").sleep(0.1)
    assert "finished" in (b / "scorebar_finish.log").read_text(encoding="utf-8")  # done by the background process


def test_deferred_finish_with_a_relative_work_dir(tmp_path, monkeypatch):
    pytest.importorskip("onnxruntime")
    if not sb.Path(__import__("replay.jersey_numbers", fromlist=["x"]).MODEL_PATH).exists():
        pytest.skip("OCR model missing")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "w").mkdir()
    tap = _run_tap(Path("w"), _frames(), monkeypatch, "1")
    assert tap.deferred and sb.load_or_compute(Path("w"), None) is not None
    for _ in range(100):
        if (tmp_path / "w" / "scorebar_finish.log").exists():
            break
        __import__("time").sleep(0.1)
    assert "finished" in (tmp_path / "w" / "scorebar_finish.log").read_text(encoding="utf-8")


def test_pending_state_finished_inline_when_no_finisher(tmp_path):
    rd = sb.ScorebarReader(sb.ScorebarConfig(every_s=2.0), recognizer=_FakeRec())
    for t, f in _frames():
        rd.add(t, f)
    sb._save_pending(tmp_path, rd)
    assert rd.rec is not None  # restored after pickling
    st = sb._load_pending(tmp_path)
    st["reader"].rec = _FakeRec()
    want = sb._load_pending(tmp_path)["reader"]
    want.rec = _FakeRec()
    res = sb._finish_state(tmp_path, st)
    assert _norm(res) == _norm(want.result())
    assert sb.load_result(tmp_path) is not None and not (tmp_path / sb.PENDING).exists()


def test_stale_pending_of_another_run_is_not_saved(tmp_path):
    rd = sb.ScorebarReader(sb.ScorebarConfig(every_s=2.0), recognizer=_FakeRec())
    for t, f in _frames(12):
        rd.add(t, f)
    sb._save_pending(tmp_path, rd)
    st = sb._load_pending(tmp_path)
    st["reader"].rec = _FakeRec()
    sb._save_pending(tmp_path, rd)  # a newer pass 1 wrote its own state meanwhile
    sb._finish_state(tmp_path, st)
    assert sb.load_result(tmp_path) is None and (tmp_path / sb.PENDING).exists()


class _FakeRec:
    device = "fake"
    charset = ["", "1", "5"]

    def probs(self, crops, max_w=320):
        out = []
        for _ in crops:
            p = np.zeros((3, 3), np.float32)
            p[0, 1] = p[1, 0] = p[2, 2] = 1.0
            out.append(p)
        return out
