"""Jersey numbers (replay/jersey_numbers.py): digit decoding, voting, uniqueness, identity-linking cue.

The recognizer is mocked everywhere except one optional smoke test that reads rendered digits with the real
model (skipped when models/pp-ocrv4-mobile-rec is missing)."""

import numpy as np
import pytest

from replay import jersey_numbers as jn
from replay.config import CourtTrackConfig
from replay.tracklets import link_tracks

from test_tracklets import FPS, ORANGE, _scene

DIG = np.arange(1, 11)  # mock charset: class 0 blank, 1..10 digits '0'..'9', 11.. letters
N_CLASSES = 20


def _probs(seq, p=0.97):
    """(T, C) softmax rows from a token list: None = blank, int = digit, 'x' = some letter."""
    out = np.full((len(seq), N_CLASSES), (1 - p) / (N_CLASSES - 1), np.float32)
    for t, s in enumerate(seq):
        c = 0 if s is None else (15 if s == "x" else int(DIG[s]))
        out[t, c] = p
    return out


# ----------------------------------------------------------------------------------------------- codes
def test_codes():
    assert jn.code_of("23") == 23 and jn.code_of("0") == 0 and jn.code_of("00") == jn.DOUBLE_ZERO
    assert jn.code_of("07") == -1 and jn.code_of("123") == -1 and jn.code_of("") == -1 and jn.code_of("a1") == -1
    assert jn.text_of(jn.DOUBLE_ZERO) == "00" and jn.text_of(7) == "7" and jn.text_of(-1) is None
    assert jn.is_partial(4, 41) and jn.is_partial(1, 41) and not jn.is_partial(2, 41) and not jn.is_partial(41, 4)
    assert jn.is_partial(0, jn.DOUBLE_ZERO)


# ----------------------------------------------------------------------------------------------- decoding
def test_decode_digits():
    d = lambda seq, **kw: jn.decode_digits(_probs(seq, **kw), DIG)  # noqa: E731
    assert d([None, 2, 2, None, 4, None])[0] == 24
    assert d([None, 4, None, 4, None])[0] == 44  # blank between repeats: two digits
    assert d([None, 4, 4, 4, None])[0] == 4  # CTC collapse
    assert d([None, 0, None, 0, None])[0] == jn.DOUBLE_ZERO
    assert d(["x", "x", None, 2, 3, None])[0] == 23  # team name letters around the number are fine
    assert d([2, "x", 3])[0] == -1  # digits split by a letter: not one number
    assert d([1, None, 2, None, 3])[0] == -1  # three digits
    assert d([None, 0, 7, None])[0] == -1  # leading zero
    assert d([None, None, None])[0] == -1
    assert d([None, 2, 4, None], p=0.6)[0] == -1  # not confident
    code, conf = d([None, 2, 4, None])
    assert code == 24 and conf > 0.9


# ----------------------------------------------------------------------------------------------- voting
def test_vote_needs_consistent_reads():
    assert jn.vote([24, 24, 24]).number == 24
    assert jn.vote([24, 24]).number == -1  # too few
    assert jn.vote([24, 24, 24, 43, 43]).number == -1  # no clear majority
    assert jn.vote([24, 24, 24, 24, 43, -1, -1]).number == 24
    assert jn.vote([]).number == -1 and jn.vote([-1, -1]).number == -1


def test_vote_partial_digits_do_not_contradict():
    v = jn.vote([41, 41, 41, 4, 4, 1, 1, 1])  # one digit hidden: neutral for the two-digit number
    assert v.number == 41 and v.against == 0
    # a one-digit number is doubted by two-digit reads containing it (missed digits are the usual error)
    assert jn.vote([4, 4, 4, 4, 24, 24]).number == -1
    assert jn.vote([4, 4, 4, 4]).number == 4
    assert jn.vote([6] * 9 + [36, 36]).number == -1  # "6" of a half-hidden "36"
    assert jn.vote([6] * 9 + [36]).number == 6


def test_vote_groups_and_tie_prefers_two_digits():
    g = np.array([1, 1, 1, 2, 2, 2, 0, 0, 0])
    c = np.array([7, 7, 7, 13, 13, 13, 13, 13, 13])
    v = jn.vote_groups(g, c)
    assert set(v) == {1, 2} and v[1].number == 7 and v[2].number == 13


def test_resolve_conflicts_same_team_same_time():
    votes = {1: jn.vote([24] * 6), 2: jn.vote([24] * 3), 3: jn.vote([24] * 4), 4: jn.vote([24] * 3)}
    team = {1: "A", 2: "A", 3: "A", 4: "B"}
    present = {1: np.arange(0, 100), 2: np.arange(50, 80), 3: np.arange(200, 300), 4: np.arange(0, 100)}
    out = jn.resolve_conflicts(votes, team, present)
    assert out[1].number == 24  # more votes keeps it
    assert out[2].number == -1  # same team, same time: loses it
    assert out[3].number == 24  # same team, later (the same player under another track id)
    assert out[4].number == 24  # other team


def test_identity_numbers_players_only():
    tid = np.array([1] * 4 + [2] * 4)
    codes = np.array([10] * 4 + [21] * 4)
    k = np.r_[np.arange(4), np.arange(4)]
    out = jn.identity_numbers(tid, codes, k, lambda t: {1: "A", 2: "referee"}[t])
    assert out[1].number == 10 and 2 not in out
    assert jn.track_info_fields(out[1]) == {"number": "10", "number_reads": 4}
    assert jn.track_info_fields(None) == {}


def test_reads_on_the_other_teams_colour_are_ignored():
    team_lab = {"A": [33, 34, 26], "B": [15, 18, -14]}  # orange, purple
    tid = np.array([1] * 6)
    codes = np.array([43] * 4 + [-1, -1])
    purple = [[20, 17, -12]] * 6
    out = jn.identity_numbers(tid, codes, np.arange(6), lambda t: "A", read_lab=purple, team_lab=team_lab)
    assert 1 not in out or out[1].number == -1  # an orange track's "43" reads were all on a purple shirt
    out = jn.identity_numbers(tid, codes, np.arange(6), lambda t: "B", read_lab=purple, team_lab=team_lab)
    assert out[1].number == 43
    assert jn.team_colour_ok([[np.nan] * 3], ["A"], team_lab).all()  # unknown colour passes


def test_single_digit_part_of_teammates_number_needs_more_reads():
    tid = np.array([1] * 6 + [2] * 5 + [3] * 9)
    codes = np.array([36] * 6 + [6] * 5 + [6] * 9)
    k = np.r_[np.arange(6), np.arange(100, 105), np.arange(200, 209)]
    out = jn.identity_numbers(tid, codes, k, lambda t: "B")
    assert out[1].number == 36 and out[2].number == -1 and out[3].number == -1  # one digit: off by default
    out = jn.identity_numbers(tid, codes, k, lambda t: "B", jn.JerseyConfig(allow_single_digit=True))
    assert out[1].number == 36 and out[2].number == -1  # "6" of a half-hidden 36
    assert out[3].number == 6  # but read often enough: a real #6


# ----------------------------------------------------------------------------------------------- linking cue
def test_link_cost():
    assert jn.piece_numbers([24, -1, 3, 4, 4]) == {24: 1, 4: 2}  # one-digit numbers need two reads
    assert jn.link_cost({24: 2}, {24: 1}) < 0
    assert jn.link_cost({}, {24: 3}) == 0.0
    assert jn.link_cost({24: 2}, {4: 2}) == 0.0  # partial: neutral
    assert jn.link_cost({24: 1}, {43: 1}) > 0  # conflicting, weak evidence: penalty
    assert jn.link_cost({24: 2}, {43: 3}) is None  # conflicting, both read consistently: impossible
    assert jn.link_cost({24: 3, 43: 3}, {24: 2}) == 0.0  # mixed piece: no bonus
    assert jn.link_cost({43: 5, 4: 2}, {24: 3}) is None  # the "4" must not bridge 43 and 24


def _gap_scene():
    """One player walks off to the right and is lost at sample 30; a team-mate (same kit) appears 0.6 s later
    where the first would be. Motion and colour cannot tell them apart."""
    n = 70
    t = np.arange(n) / FPS
    a = np.c_[5 + 1.5 * t, np.full(n, 7.0)]
    b = a.copy()
    return _scene({1: a, 2: b}, {1: ORANGE, 2: ORANGE}, bt_of=lambda p, k: p,
                  hidden={1: range(30, n), 2: range(0, 36)})


def _numbers(s, reads):
    num = np.full(len(s["k"]), -1)
    for pid, code in reads.items():
        idx = np.flatnonzero(s["pid"] == pid)
        num[idx[::6]] = code  # a read every 0.6 s
    return num


def test_numbers_prevent_wrong_link():
    s = _gap_scene()
    cfg = CourtTrackConfig()
    args = (s["k"], s["xy"], s["R"], s["feats"], s["bt"], s["conf"], FPS, cfg)
    kw = dict(boxes=s["boxes"], motion=s["motion"], n_samples=s["n"], H=s["H"])
    a, b = s["pid"] == 1, s["pid"] == 2
    base = link_tracks(*args, **kw)
    assert set(base.ids[a]) == set(base.ids[b])  # without numbers: linked into one identity
    res = link_tracks(*args, numbers=_numbers(s, {1: 24, 2: 43}), **kw)
    assert set(res.ids[a]).isdisjoint(res.ids[b])
    same = link_tracks(*args, numbers=_numbers(s, {1: 24, 2: 24}), **kw)
    assert set(same.ids[a]) == set(same.ids[b])


def test_chain_without_reads_in_the_middle_is_cut():
    """24 ... (piece without reads) ... 43 linked in one round: the chain must be cut somewhere."""
    n = 90
    t = np.arange(n) / FPS
    a = np.c_[5 + 1.5 * t, np.full(n, 7.0)]
    s = _scene({1: a, 2: a.copy(), 3: a.copy()}, {1: ORANGE, 2: ORANGE, 3: ORANGE}, bt_of=lambda p, k: p,
               hidden={1: range(30, n), 2: [k for k in range(n) if not 33 <= k < 55], 3: range(0, 58)})
    num = _numbers(s, {1: 24, 3: 43})
    res = link_tracks(s["k"], s["xy"], s["R"], s["feats"], s["bt"], s["conf"], FPS, CourtTrackConfig(),
                      boxes=s["boxes"], motion=s["motion"], n_samples=s["n"], H=s["H"], numbers=num)
    assert set(res.ids[s["pid"] == 1]).isdisjoint(res.ids[s["pid"] == 3])


# ----------------------------------------------------------------------------------------------- game state + exports
def test_lineup_slots_follow_jersey_numbers():
    from replay import game_state as gs
    from test_game_state import FPS as GFPS, Replay, still

    r = Replay(12)
    r.player("1", "A", 0, 3, still(20.0, 5.0))  # #23
    r.player("2", "A", 0, 3, still(10.0, 5.0))  # #8
    r.player("7", "A", 4, 12, still(11.0, 5.0))  # #23 again, but appears next to where #8 was
    r.player("9", "A", 4, 12, still(20.5, 5.0))  # no number
    doc = r.doc()
    doc["track_info"] = {"1": {"team": "A", "number": "23"}, "2": {"team": "A", "number": "8"},
                         "7": {"team": "A", "number": "23"}, "9": {"team": "A"}}
    lu = gs.analyse(doc)["lineup"]
    slot = lu["slot_of_track"]
    assert slot["7"] == slot["1"] and slot["9"] != slot["1"]
    assert lu["numbers"][slot["1"]] == "23" and lu["numbers"][slot["2"]] == "8"
    gi = gs.load_input(doc)
    at = {s["slot"]: s for s in gs.lineup_at(gi, lu, int(6 * GFPS))["A"]}
    assert at[slot["7"]]["number"] == "23" and at[slot["7"]]["visible"]


def test_data_export_has_jersey_numbers(tmp_path):
    import csv
    import json

    from fake_project import make_project
    from replay.data_export import export_simple

    proj = make_project(tmp_path / "proj")
    tj = proj / "results" / "tracking.json"
    tracking = json.loads(tj.read_text(encoding="utf-8"))
    tracking["tracks"]["1"].update(number="23", number_reads=5)
    tj.write_text(json.dumps(tracking), encoding="utf-8")
    out = export_simple(proj, tmp_path / "out")
    with open(out / "players.csv", newline="", encoding="utf-8") as fh:
        players = list(csv.DictReader(fh))
    assert {p["jersey_number"] for p in players if p["track_id"] == "1"} == {"23"}
    assert {p["jersey_number"] for p in players if p["track_id"] != "1"} == {""}
    with open(out / "track_summary.csv", newline="", encoding="utf-8") as fh:
        summ = {s["track_id"]: s for s in csv.DictReader(fh)}
    assert summ["1"]["jersey_number"] == "23" and summ["1"]["jersey_number_reads"] == "5"
    assert "jersey_number" in (out / "README_FOR_AI.md").read_text(encoding="utf-8")


def test_ai_digest_shows_jersey_numbers(tmp_path):
    import json

    from fake_project import make_project
    from replay import ai_analysis as ai

    root = make_project(tmp_path / "p")
    tj = root / "results" / "tracking.json"
    tracking = json.loads(tj.read_text(encoding="utf-8"))
    tracking["tracks"]["1"].update(number="23", number_reads=4)
    tj.write_text(json.dumps(tracking), encoding="utf-8")
    d = ai.build_game_digest(root, sample_hz=2.0)
    assert "## JERSEY NUMBERS" in d.header and "#23" in d.header
    first = d.frames[0][1]
    assert "A 1#23:5.0,7.0" in first and "held A1#23" in first


# ----------------------------------------------------------------------------------------------- crops + plumbing
def test_eligible_boxes():
    cfg = jn.JerseyConfig()
    boxes = np.array([[100, 100, 150, 250],  # ok
                      [300, 100, 350, 150],  # too small
                      [0, 100, 50, 250],  # touches the border
                      [500, 100, 550, 250], [520, 120, 570, 270]])  # overlapping pair
    ok = jn.eligible(boxes, (1280, 720), cfg)
    assert ok.tolist() == [True, False, False, False, False]
    img = np.zeros((720, 1280, 3), np.uint8)
    crops = jn.torso_crops(img, boxes[0], cfg)
    assert len(crops) == len(cfg.windows) and all(c.shape[0] >= cfg.crop_h for c in crops)


def test_best_read_combines_windows():
    assert jn.best_read([(-1, 0.3), (24, 0.95), (24, 0.9), (4, 0.99)]) == (24, 0.95)
    assert jn.best_read([(-1, 0.3), (-1, 0.5)])[0] == -1


class _FakeRecognizer:
    device = "fake"

    def __init__(self):
        self.calls = 0

    def read(self, crops, cfg=None):
        self.calls += len(crops)
        return [(24, 0.97)] * len(crops)


class _Data:
    def __init__(self, video, n):
        self.t = np.arange(n) / 10.0
        self.k = np.arange(n) * 3
        self.signature = {"video": str(video), "start_s": 0.0, "video_size_bytes": 1}
        rows = []
        for i, k in enumerate(self.k):
            rows.append([k, 100, 100, 150, 250, 0.9, 0, 1] + [0.0] * 6)  # eligible player
            rows.append([k, 400, 100, 420, 140, 0.9, 0, 2] + [0.0] * 6)  # too small
        self.rows = np.array(rows, float)


def test_compute_reads_with_mock_recognizer(tmp_path):
    cv2 = pytest.importorskip("cv2")
    video = tmp_path / "v.avi"
    w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (640, 360))
    if not w.isOpened():
        pytest.skip("no video writer")
    for _ in range(90):
        w.write(np.full((360, 640, 3), 128, np.uint8))
    w.release()
    data = _Data(video, 30)
    rec = _FakeRecognizer()
    cfg = jn.JerseyConfig(every_s=0.5)
    reads = jn.compute_reads(data, video, cfg, recognizer=rec)
    assert reads is not None
    small = data.rows[:, 1] == 400
    assert (reads.code[small] == -1).all() and not np.isfinite(reads.conf[small]).any()
    got = np.flatnonzero(reads.code == 24)
    assert 4 <= len(got) <= 7  # 3 s of video, a frame every 0.5 s
    assert rec.calls == len(got) * len(cfg.windows)
    assert np.allclose(reads.lab[got], [53.7, 0, 0], atol=3)  # grey (128) frame: colour behind the reads


class _FakeSignalRecognizer(_FakeRecognizer):
    def read_with_signal(self, crops, cfg=None):
        self.calls += len(crops)
        return [(-1, 0.0, 0.0)] * len(crops)  # nothing digit-like anywhere


def test_cascade_skips_other_windows_without_digit_signal(tmp_path):
    cv2 = pytest.importorskip("cv2")
    video = tmp_path / "v.avi"
    w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (640, 360))
    if not w.isOpened():
        pytest.skip("no video writer")
    for _ in range(60):
        w.write(np.full((360, 640, 3), 128, np.uint8))
    w.release()
    data = _Data(video, 20)
    rec = _FakeSignalRecognizer()
    cfg = jn.JerseyConfig(every_s=0.5)
    reads = jn.compute_reads(data, video, cfg, recognizer=rec)
    boxes = int(np.isfinite(reads.conf).sum())
    assert boxes >= 3 and rec.calls == boxes * len(cfg.first_windows)
    assert (reads.code == -1).all()


# ----------------------------------------------------------------------------------------------- real model
@pytest.mark.skipif(not jn.MODEL_PATH.exists(), reason="jersey number model not installed")
def test_real_model_reads_rendered_numbers():
    cv2 = pytest.importorskip("cv2")
    rec = jn.Recognizer(device="cpu")
    crops = []
    for text in ("23", "5", "41"):
        img = np.full((64, 30 + 34 * len(text), 3), 255, np.uint8)
        cv2.putText(img, text, (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (0, 0, 0), 4)
        crops.append(img)
    reads = rec.read(crops, jn.JerseyConfig(min_single_conf=0.8))
    assert [jn.text_of(c) for c, _ in reads] == ["23", "5", "41"]


def test_load_reads_cache(tmp_path):
    assert jn.load_reads(tmp_path) is None
    np.savez_compressed(tmp_path / "jersey_numbers.npz", key=np.array("x"), code=np.array([-1, 23], np.int16),
                        conf=np.array([np.nan, 0.9], np.float16), lab=np.zeros((2, 3), np.float16),
                        stats=np.array("{}"))
    r = jn.load_reads(tmp_path)
    assert r.code.tolist() == [-1, 23] and r.sampled.tolist() == [False, True]


def test_decided_tracks_are_read_sparsely(tmp_path):
    """Speed gate: once a ByteTrack id has a confident number it is only re-checked every decided_every_s."""
    cv2 = pytest.importorskip("cv2")
    video = tmp_path / "v.avi"
    w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (640, 360))
    if not w.isOpened():
        pytest.skip("no video writer")
    for _ in range(300):
        w.write(np.full((360, 640, 3), 128, np.uint8))
    w.release()
    data = _Data(video, 100)  # 10 s, a sample every 0.1 s
    full = jn.compute_reads(data, video, jn.JerseyConfig(every_s=0.1, batch=4, decided_reads=0),
                            recognizer=_FakeRecognizer())
    gated = jn.compute_reads(data, video, jn.JerseyConfig(every_s=0.1, batch=4, decided_reads=6,
                                                          decided_every_s=2.0), recognizer=_FakeRecognizer())
    n_full, n_gated = int((full.code == 24).sum()), int((gated.code == 24).sum())
    assert n_full >= 90
    assert 6 <= n_gated <= 20 and gated.stats["skipped"] == n_full - n_gated
    assert vote_ok(gated.code)


def vote_ok(codes) -> bool:
    return jn.vote(codes).number == 24
