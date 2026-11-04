"""General score-bar reader (replay/scorebar.py): smoothing, block roles, overlay location on synthetic frames."""

import numpy as np
import pytest

from replay import scorebar as sb


def test_smoothing_monotonic_basket_steps():
    cfg = sb.ScorebarConfig()
    raw = [(0, 10), (1, 10), (2, 12), (3, 12), (4, 18), (5, 12), (6, 13), (7, 13), (8, None), (9, 13), (10, 9)]
    out = dict(sb.smooth_scores(raw, cfg))
    assert out[1] == 10 and out[2] == 12  # change back-dated to its first read
    assert out[4] == 12 and out[5] == 12  # one misread (18) ignored
    assert out[6] == 13 and out[9] == 13 and out[10] == 13  # never down on a single read


def test_smoothing_correction_after_long_stable_lower_value():
    cfg = sb.ScorebarConfig(correction_s=5)
    raw = [(t, 20) for t in range(5)] + [(t, 18) for t in range(5, 14)]
    out = dict(sb.smooth_scores(raw, cfg))
    assert out[4] == 20 and out[13] == 18 and out[5] == 18  # accepted after 5 s, back-dated


def _blk(x, y, reads, h=20):
    b = sb.Block(x, y, x + 2 * h, y + h)
    b.reads = [(float(t), s, 0.95) for t, s in enumerate(reads)]
    return b


def test_classify_picks_scores_not_clock_or_seed():
    n = 40
    blocks = [
        _blk(100, 600, ["KLA"] * n),
        _blk(200, 600, ["15"] * 20 + ["17"] * 20),  # score
        _blk(260, 600, ["59"] * 10 + ["61"] * 30),  # score
        _blk(400, 600, ["05"] * 20 + ["06"] * 20),  # clock minutes ...
        _blk(445, 600, [f"{i:02d}" for i in range(n)]),  # ... and seconds
        _blk(500, 600, ["3"] * n),  # period
        _blk(50, 600, ["1"] * n),  # seed
    ]
    cls = sb.classify(blocks)
    assert [d["i"] for d in cls["score"][:2]] == [2, 1]


def test_locate_finds_static_text_on_moving_background():
    cv2 = pytest.importorskip("cv2")
    rng = np.random.default_rng(0)
    frames = []
    for k in range(20):
        f = rng.integers(0, 255, (360, 640, 3), dtype=np.uint8)  # "camera" content changes every frame
        f = cv2.GaussianBlur(f, (9, 9), 0)
        cv2.rectangle(f, (200, 300), (440, 340), (20, 20, 20), -1)
        cv2.putText(f, "KLA  15  59", (210, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        frames.append(f)
    blocks, _ = sb.locate_blocks(frames, sb.ScorebarConfig())
    assert blocks and all(290 <= b.y0 and b.y1 <= 350 and 190 <= b.x0 and b.x1 <= 450 for b in blocks)


class _Ev:
    def __init__(self, a, b, period, seq):
        self.score_a, self.score_b, self.period, self.seq = a, b, period, seq


def _running():
    out, a, b, seq = [], 0, 0, 0
    for period, pts in ((1, [(2, 0), (0, 3), (2, 0)]), (2, [(0, 2), (1, 0), (0, 2), (3, 0)])):
        for da, db in pts:
            a, b, seq = a + da, b + db, seq + 1
            out.append(_Ev(a, b, period, seq))
    return out  # 2-0 2-3 4-3 | 4-5 5-5 5-7 8-7


def test_pin_candidate_uses_only_a_sure_unique_reading():
    run = _running()
    gs = {"scorebar": {"summary": {"sure": True, "start": [4, 3], "end": [5, 7]}}}
    c = sb.pin_candidate(run, gs)
    assert c["start_score"] == [4, 3] and c["period"] == 2 and not c["flip"] and c["best"]
    gs["scorebar"]["summary"].update(start=[3, 4], end=[7, 5])  # bar shows the sheet's team B first
    c = sb.pin_candidate(run, gs)
    assert c["start_score"] == [4, 3] and c["flip"]
    gs["scorebar"]["summary"]["sure"] = False
    assert sb.pin_candidate(run, gs) is None
    gs["scorebar"]["summary"].update(sure=True, start=[9, 9], end=[9, 9])  # not in the sheet
    assert sb.pin_candidate(run, gs) is None
    assert sb.pin_candidate(run, None) is None


def test_summary_sure_and_unsure():
    s = [(float(t), 10, 12, "", "") for t in range(10)]
    res = {"samples": s, "timeline": [(0.0, 10, 12)], "sides": []}
    assert sb.summary(res)["sure"] and sb.summary(res)["start"] == [10, 12]
    res["samples"] = [(float(t), None, None, "", "") for t in range(6)] + s[6:]
    assert not sb.summary(res)["sure"]  # mostly unreadable, locked late


def test_tap_writes_cache(tmp_path):
    cv2 = pytest.importorskip("cv2")
    from types import SimpleNamespace

    tap = sb.ScorebarTap(every_s=2.0)
    tap.start(SimpleNamespace(resumed=False))
    rng = np.random.default_rng(1)
    for k in range(30):
        t = k * 1.0
        if tap.wants(k, t):
            f = cv2.GaussianBlur(rng.integers(0, 255, (360, 640, 3), dtype=np.uint8), (9, 9), 0)
            cv2.rectangle(f, (200, 300), (440, 340), (20, 20, 20), -1)
            cv2.putText(f, "KLA  15  59", (210, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            tap.reader.rec = _FakeRec()
            tap.process(SimpleNamespace(t=t, image=f))
    tap.finish(tmp_path, True)
    doc = sb.load_result(tmp_path)
    assert doc is not None and doc["stats"]["samples"] == 15


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


def test_ocr_unicode_digit_reads_do_not_crash():
    """The PP-OCR charset has '²', '①' ... : str.isdigit() is True for them but int() raises - a single such read
    must not crash classify / the clock reader (bug hunt 19.10)."""
    n = 40
    blocks = [
        _blk(200, 600, ["15"] * 19 + ["1²"] + ["17"] * 20),
        _blk(260, 600, ["59"] * 10 + ["①"] + ["61"] * 29),
        _blk(400, 600, ["05"] * 20 + ["0²"] + ["06"] * 19),  # clock minutes ...
        _blk(445, 600, [f"{i:02d}" for i in range(n - 1)] + ["5²"]),  # ... and seconds
    ]
    cls = sb.classify(blocks)
    assert [d["i"] for d in cls["score"][:2]] == [1, 0]
    assert sb._num("1²") is None and sb._num("①") is None and sb._num("12") == 12
    sb.game_clock_series(blocks, cls, [float(t) for t in range(n)])  # clock pair path: no ValueError


def test_clock_smoothing_countdown_stop_misread_and_period():
    ts = [float(2 * k) for k in range(12)]
    # counts down 600, 598, ... stops at 592 for 3 samples, one misread (999), then runs again
    raw = [600, 598, 596, 594, 592, 592, 592, 999, 588, 586, None, 582]
    per = sb.smooth_periods([(t, 2) for t in ts])
    out = sb.smooth_clock(list(zip(ts, raw)), per)
    vals = [c for c, _ in out]
    assert vals[1:7] == [598, 596, 594, 592, 592, 592]
    assert vals[7] == 592  # misread ignored
    assert out[2][1] is True and out[5][1] is False
    assert vals[10] == 586 and vals[11] == 582
    res = {"clock": [(t, p, c, r) for t, p, (c, r) in zip(ts, per, out)]}
    assert sb.clock_at(res, 9.0)[1] == 592
    assert sb.played_seconds(res, 0.0, 12.0) == 6.0  # 598->592 running; 600->598 unknown at start


def test_clock_counts_up_and_period_only_rises():
    ts = [float(2 * k) for k in range(8)]
    out = sb.smooth_clock(list(zip(ts, [60, 62, 64, 66, 68, 70, 72, 74])), [3] * 8)
    assert [r for _, r in out][2:] == [True] * 6
    assert sb.smooth_periods([(0, 3), (1, 3), (2, 3), (3, 2), (4, 3), (5, 4), (6, 4), (7, 4)]) == [3, 3, 3, 3, 3, 3, 3, 4]
    assert sb._period_value("2ND") == 2 and sb._period_value("Q3") == 3 and sb._period_value("OT") == 5


def test_clock_pair_outside_the_bar_strip_does_not_crash():
    """Bug hunt 19.10: _reread_pair cut the clock region out of the stored bar strips without checking that it lies
    inside them; a clock in another row than the score band gave empty crops -> cv2.resize error in the recogniser
    -> the whole score-bar reading was lost."""
    import cv2
    from types import SimpleNamespace

    from replay.jersey_numbers import Recognizer

    class FakeRec:
        charset = ["", "0", "1"]

        def probs(self, crops, max_w=160):
            out = []
            for c in crops:
                Recognizer.prepare(c, max_w)  # the real preprocessing (raises on an empty crop)
                out.append(np.eye(3)[[1, 0, 2]])
            return out

    strip = np.full((30, 300, 3), 200, np.uint8)
    jp = cv2.imencode(".jpg", strip)[1].tobytes()
    reader = SimpleNamespace(ts=[0.0, 1.0], strips=[jp, jp], strip_box=(100, 600, 400, 630),
                             _recognizer=lambda: FakeRec())
    blocks = [sb.Block(150, 100, 180, 120), sb.Block(190, 100, 220, 120)]  # clock pair far above the strip
    out = sb._reread_pair(blocks, (0, 1), reader)
    assert out == {0.0: None, 1.0: None}


def test_gives_up_early_without_a_score_bar():
    cv2 = pytest.importorskip("cv2")
    rng = np.random.default_rng(2)
    used = []

    def frames():
        for k in range(1000):
            used.append(k)
            yield float(2 * k), cv2.GaussianBlur(rng.integers(0, 255, (180, 320, 3), dtype=np.uint8), (9, 9), 0)

    res = sb.read_frames(frames(), sb.ScorebarConfig(every_s=2.0), recognizer=_FakeRec())
    assert res.get("no_bar") and res["summary"]["sure"] is False
    assert len(used) <= 1000  # stops decoding once the retries are used up


def test_pair_is_best_similar_height_pair_not_top_candidate_plus_first_partner():
    """23.10 fix: _result paired the top-ranked candidate with the first similar-height one. A tall team-seed / logo
    number ranked first must not hide the real pair; two constant one-digit seeds are never a score pair."""
    n = 40
    blocks = [
        _blk(300, 600, ["1"] * n, h=20),  # seed (left)
        _blk(400, 600, ["50"] * 20 + ["52"] * 20, h=40),  # score
        _blk(520, 600, ["1"] * n, h=20),  # seed (right)
        _blk(700, 600, ["32"] * 10 + ["35"] * 30, h=40),  # score
        _blk(100, 560, ["7"] * 20 + ["9"] * 20, h=60),  # a tall number elsewhere (e.g. a lower-third stat)
    ]
    cls = sb.classify(blocks)
    pair = sb.score_pair(blocks, cls["score"])
    assert sorted(d["i"] for d in pair) == [1, 3]
    seeds = [_blk(300, 600, ["1"] * n), _blk(520, 600, ["1"] * n)]
    assert sb.score_pair(seeds, sb.classify(seeds)["score"]) is None


def test_pair_falls_back_to_old_pairing_without_a_similar_height_pair():
    n = 40
    blocks = [_blk(400, 600, ["50"] * 20 + ["52"] * 20, h=40), _blk(700, 600, ["32"] * 10 + ["35"] * 30, h=20)]
    pair = sb.score_pair(blocks, sb.classify(blocks)["score"])
    assert pair is not None and sorted(d["i"] for d in pair) == [0, 1]


def test_logo_glued_numbers_are_split():
    assert sb._num("86A") == 86 and sb._num("74风") == 74 and sb._num("A48") == 48 and sb._num("@48") == 48
    for s in ("Q4", "OT1", "P2", "H1", "TO3", "FOULS:1", "1:05", "2ND", "ABC48"):
        assert sb._num(s) is None, s


def test_locate_number_glued_to_logo_in_some_frames_stays_its_own_field(monkeypatch):
    """seg_2965 (word boxes as found on the real bar): the score "50" is one word with the "A" logo next to it in 40 %
    of the frames. The field must not grow over the logo (it was then dropped as a 'box around text' -> no left
    score, no sides); a word's tiny letter fragments must still not split a team name."""
    per_frame = []
    for k in range(20):
        words = [(724, 957, 783, 1013), (842, 952, 974, 981)]  # logo, "ARIZONA"
        words.append((626, 947, 785, 1016) if k % 5 >= 3 else (637, 960, 695, 1009))  # "50" glued / alone
        if k % 2:
            words.append((914, 957, 925, 977))  # a letter fragment inside "ARIZONA"
        per_frame.append(sorted(words))
    calls = iter(per_frame)
    monkeypatch.setattr(sb, "_words", lambda g, min_h: next(calls))
    frames = [np.zeros((1080, 1920, 3), np.uint8) for _ in range(20)]
    blocks, _ = sb.locate_blocks(frames, sb.ScorebarConfig(), panels=[(0, 0, 1920, 1080)])
    boxes = [(b.x0, b.x1) for b in blocks]
    assert any(b.x0 < 637 and 695 < b.x1 < 724 for b in blocks), boxes  # the score alone
    assert any(b.x0 < 842 and b.x1 > 974 for b in blocks), boxes  # the whole name


def test_locate_field_that_sometimes_grows_into_its_neighbour_is_cut_back(monkeypatch):
    """seg_2965 right score "32" merged with the "2" of "2ND" in some frames: kept, cut back before "2ND"."""
    per_frame = []
    for k in range(20):
        sc = (1134, 947, 1241, 1008) if k % 3 == 0 else (1134, 960, 1191, 1008)
        per_frame.append([sc, (1226, 972, 1278, 1004)])
    calls = iter(per_frame)
    monkeypatch.setattr(sb, "_words", lambda g, min_h: next(calls))
    frames = [np.zeros((1080, 1920, 3), np.uint8) for _ in range(20)]
    blocks, _ = sb.locate_blocks(frames, sb.ScorebarConfig(), panels=[(0, 0, 1920, 1080)])
    assert any(b.x0 <= 1134 and 1191 <= b.x1 < 1252 for b in blocks), [(b.x0, b.x1) for b in blocks]


def test_seed_next_to_team_name_is_not_the_period():
    """seg_4465: the ordinal "2ND" block was not found; the constant seed "1" next to the team name must not become
    the period (it gave P1 in the 2nd half). A bare number next to the clock still is (KoriTV "05:07 | 4")."""
    n = 40
    blocks = [
        _blk(300, 600, ["1"] * n, h=20),  # seed
        _blk(400, 600, ["50"] * 20 + ["52"] * 20, h=20),
        _blk(700, 600, ["32"] * 10 + ["35"] * 30, h=20),
        _blk(1300, 600, [f"4:{59 - i:02d}" for i in range(n)], h=20),  # clock
    ]
    assert sb.classify(blocks)["period"] is None
    blocks.append(_blk(1400, 600, ["2"] * n, h=20))  # period next to the clock
    assert sb.classify(blocks)["period"]["i"] == 4
