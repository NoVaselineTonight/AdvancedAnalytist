"""Gemini court keypoints (replay/gemini_court.py): hard cost caps, ledger, cache (no re-send), retries, parsing and
the keypoints -> homography path, all with a fake client (nothing is sent anywhere)."""

import json
import socket
import urllib.error
from types import SimpleNamespace

import numpy as np
import pytest

from replay import court
from replay import gemini_court as gc


class FakeVision:
    """Answers with the true keypoints of a known court->image homography; counts calls."""

    def __init__(self, H=None, size=(1280, 720), model=gc.DEFAULT_MODEL, fail=None):
        self.model, self.H, self.size, self.calls, self.fail = model, H, size, 0, list(fail or [])

    def locate(self, jpeg, system, user, schema):
        self.calls += 1
        if self.fail:
            raise self.fail.pop(0)
        pts = []
        if self.H is not None:
            for name, (x, y) in court.KEYPOINTS.items():
                q = self.H @ np.array([x, y, 1.0])
                u, v = q[:2] / q[2]
                if 0 <= u < self.size[0] and 0 <= v < self.size[1]:
                    pts.append({"name": name, "x": u / self.size[0] * 1000, "y": v / self.size[1] * 1000,
                                "confidence": 0.9})
        return json.dumps({"points": pts}), {"promptTokenCount": 1200, "candidatesTokenCount": 300}


def _data(n):
    return SimpleNamespace(t=np.arange(n) * 10.0, frame=np.arange(n) * 300, motion=[None] * n, k=np.arange(n))


def _frames(video, frames, cancel=None):
    for f in frames:
        yield f, np.zeros((720, 1280, 3), np.uint8)


def _H():
    """A plausible court->image homography (broadcast-like view of the whole court)."""
    court.configure(28.0, 15.0, "FIBA")
    src = np.array([[0, 0], [28, 0], [28, 15], [0, 15]], np.float32)
    dst = np.array([[100, 650], [1180, 650], [1000, 250], [280, 250]], np.float32)
    import cv2
    return cv2.getPerspectiveTransform(src, dst)


def test_default_model_is_the_cheapest_and_60_frames_fit_the_game_cap():
    assert gc.DEFAULT_MODEL == min(gc.PRICES, key=lambda m: sum(gc.PRICES[m]))
    assert gc.estimate_cost(gc.MAX_FRAMES) <= gc.GAME_CAP_EUR
    assert gc.estimate_cost(gc.MAX_FRAMES, "gemini-2.5-flash") > gc.estimate_cost(gc.MAX_FRAMES)


def test_game_cap_stops_before_it_is_crossed(tmp_path):
    led = gc.Ledger(tmp_path / "l.json")
    led.set_caps(game_eur=3.5 * gc.request_estimate_eur())  # room for 3 requests, not 4
    sp = gc.Spender(led, gc.DEFAULT_MODEL, "p")
    sent = 0
    with pytest.raises(gc.BudgetReached, match="this game"):
        for _ in range(10):
            sp.allow()
            sp.book({"promptTokenCount": gc.TOKENS_IN_IMAGE + gc.TOKENS_IN_TEXT, "candidatesTokenCount": gc.TOKENS_OUT})
            sent += 1
    assert sent == 3 and sp.eur <= led.game_cap


def test_monthly_cap_counts_earlier_games_and_is_never_exceeded(tmp_path):
    led = gc.Ledger(tmp_path / "l.json")
    led.set_caps(game_eur=1.0, month_eur=0.01)
    led.add("old", 10, 1, 1, 0.0095, gc.DEFAULT_MODEL)  # earlier game this month
    sp = gc.Spender(gc.Ledger(tmp_path / "l.json"), gc.DEFAULT_MODEL, "p")  # re-read from disk
    with pytest.raises(gc.BudgetReached, match="this month"):
        sp.allow()


def test_frame_cap_is_hard_even_if_asked_for_more(tmp_path):
    led = gc.Ledger(tmp_path / "l.json")
    led.set_caps(game_eur=100.0, month_eur=100.0)
    sp = gc.Spender(led, gc.DEFAULT_MODEL, "p", max_frames=500)
    for _ in range(gc.MAX_FRAMES):
        sp.allow()
        sp.book({})
    with pytest.raises(gc.BudgetReached, match="frame limit"):
        sp.allow()
    assert len(gc.select_frames(_data(300), (1280, 720), max_frames=500)) == gc.MAX_FRAMES


def test_ledger_books_real_usage_once_per_game(tmp_path):
    led = gc.Ledger(tmp_path / "l.json")
    sp = gc.Spender(led, gc.DEFAULT_MODEL, "proj")
    sp.allow()
    sp.book({"promptTokenCount": 1000, "candidatesTokenCount": 200})
    sp.close()
    e = gc.Ledger(tmp_path / "l.json").last_entries(1)[0]
    assert e["project"] == "proj" and e["frames"] == 1 and e["tokens_in"] == 1000 and e["tokens_out"] == 200
    assert e["eur"] == pytest.approx(gc.token_cost_eur(gc.DEFAULT_MODEL, 1000, 200))


def test_one_retry_on_network_errors_only():
    v = FakeVision(fail=[urllib.error.URLError("down")])
    gc.ask(v, b"", ["mid_near"])
    assert v.calls == 2
    v = FakeVision(fail=[socket.timeout(), socket.timeout()])
    with pytest.raises(socket.timeout):
        gc.ask(v, b"", ["mid_near"])
    assert v.calls == 2
    v = FakeVision(fail=[urllib.error.HTTPError("u", 429, "quota", None, None)])
    with pytest.raises(urllib.error.HTTPError):
        gc.ask(v, b"", ["mid_near"])
    assert v.calls == 1


def test_parse_points_keeps_only_known_confident_in_range_points():
    names = ["mid_near", "mid_far"]
    txt = json.dumps({"points": [{"name": "mid_near", "x": 500, "y": 1000, "confidence": 0.9},
                                 {"name": "mid_far", "x": 500, "y": 100, "confidence": 0.1},
                                 {"name": "bogus", "x": 1, "y": 1, "confidence": 1},
                                 {"name": "mid_far", "x": 1500, "y": 1, "confidence": 1}]})
    assert gc.parse_points(txt, names, (1000, 500)) == {0: (500.0, 500.0, 0.9)}
    assert gc.parse_points("not json", names, (1000, 500)) == {}


def test_located_frames_and_cache_means_a_rebuild_sends_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(gc, "line_support", lambda img, H, size: 1.0)
    monkeypatch.setattr(gc, "wide_court_view", lambda img, boxes=None: True)
    H = _H()
    led = gc.Ledger(tmp_path / "l.json")
    v = FakeVision(H)
    res = gc.calibrate_gemini("video.mp4", _data(6), (1280, 720), v, work=tmp_path, ledger=led, project="p",
                              read_frames=_frames)
    assert v.calls == 6 and len(res.frames) == 6 and res.spec == "FIBA"
    H1 = np.array(res.frames[0]["H"]).reshape(3, 3)
    q = H1 @ np.array([14.0, 7.5, 1.0])
    p = H @ np.array([14.0, 7.5, 1.0])
    assert np.allclose(q[:2] / q[2], p[:2] / p[2], atol=2.0)
    assert len(gc.Ledger(tmp_path / "l.json").last_entries()) == 1
    v2 = FakeVision(H)
    res2 = gc.calibrate_gemini("video.mp4", _data(6), (1280, 720), v2, work=tmp_path, ledger=led, project="p",
                               read_frames=_frames)
    assert v2.calls == 0 and len(res2.frames) == 6  # every frame answered from the project cache
    assert len(gc.Ledger(tmp_path / "l.json").last_entries()) == 1  # nothing new booked


def test_budget_reached_stops_sending_but_keeps_what_was_found(tmp_path, monkeypatch):
    monkeypatch.setattr(gc, "line_support", lambda img, H, size: 1.0)
    monkeypatch.setattr(gc, "wide_court_view", lambda img, boxes=None: True)
    led = gc.Ledger(tmp_path / "l.json")
    led.set_caps(game_eur=2.5 * gc.request_estimate_eur())
    v = FakeVision(_H())
    res = gc.calibrate_gemini("video.mp4", _data(6), (1280, 720), v, work=tmp_path, ledger=led, project="p",
                              read_frames=_frames)
    spent = gc.Ledger(tmp_path / "l.json").last_entries(1)[0]["eur"]
    assert 2 <= v.calls < 6 and "budget reached" in res.message
    assert spent <= led.game_cap  # real usage booked; the next conservative estimate would have crossed the cap


def test_close_ups_are_never_sent(tmp_path):
    """A frame that is not a wide court view (here: black) costs nothing."""
    v = FakeVision(_H())
    gc.calibrate_gemini("video.mp4", _data(4), (1280, 720), v, work=tmp_path, ledger=gc.Ledger(tmp_path / "l.json"),
                        read_frames=_frames)
    assert v.calls == 0


def test_line_check_needs_enough_visible_samples():
    img = np.zeros((720, 1280, 3), np.uint8)
    H_far = np.array([[1, 0, 5000], [0, 1, 5000], [0, 0, 1.0]])  # court projected outside the image
    assert gc.line_support(img, H_far, (1280, 720)) == 0.0


def test_only_easy_points_are_asked_and_kept():
    txt = gc.prompt(list(court.KEYPOINTS))
    assert "L_corner_near" in txt and "three_top" not in txt and "omit" in gc.SYSTEM.lower()
    names = list(court.KEYPOINTS)
    ans = json.dumps({"points": [{"name": "L_three_top", "x": 1, "y": 1, "confidence": 1}]})
    assert gc.parse_points(ans, names, (1280, 720)) == {}


def test_frame_selection_prefilter_is_linear_in_the_rows():
    """Bug hunt 19.10: wide_by_boxes scanned ALL pass-1 rows for every sample (full game: ~11 min before the first
    request). Same picks as the per-sample definition, in well under a second for 6000 samples."""
    import time

    n, per = 6000, 12
    rng = np.random.default_rng(0)
    k = np.repeat(np.arange(n), per)
    rows = np.zeros((n * per, 14))
    rows[:, 0] = k
    rows[:, 2] = 300 + rng.normal(0, 5, len(k))
    rows[:, 4] = rows[:, 2] + np.where(k % 7 == 0, 400, 120)  # every 7th sample a close-up (huge people)
    rows[:, 5] = 0.9
    rows[:, 6] = 0
    data = SimpleNamespace(t=np.arange(n) / 10.0, frame=np.arange(n) * 3, k=np.arange(n), rows=rows,
                           motion=[None] + [np.eye(3)] * (n - 1))
    t0 = time.perf_counter()
    picks = gc.select_frames(data, (1280, 720), max_frames=60)
    assert time.perf_counter() - t0 < 3.0
    assert picks and all(i % 7 != 0 for i in picks)
    assert gc.wide_by_boxes(data, 7, (1280, 720)) is False and gc.wide_by_boxes(data, 8, (1280, 720)) is True


def test_ledger_entry_does_not_revert_caps_or_entries_written_meanwhile(tmp_path):
    """Bug hunt 19.10: a calibration's ledger is read when the run starts and saved when it ends; a cap the user
    lowered in the app meanwhile (or another game's entry) was overwritten by the stale copy."""
    run = gc.Ledger(tmp_path / "l.json")  # calibration starts
    gui = gc.Ledger(tmp_path / "l.json")
    gui.set_caps(game_eur=0.05, month_eur=0.5)  # the user lowers the caps during the run
    gc.Ledger(tmp_path / "l.json").add("other", 3, 1, 1, 0.001, gc.DEFAULT_MODEL)  # another game finished
    run.add("p", 5, 10, 10, 0.002, gc.DEFAULT_MODEL)  # calibration ends
    disk = gc.Ledger(tmp_path / "l.json")
    assert disk.month_cap == 0.5 and disk.game_cap == 0.05
    assert [e["project"] for e in disk.last_entries(5)] == ["other", "p"]
