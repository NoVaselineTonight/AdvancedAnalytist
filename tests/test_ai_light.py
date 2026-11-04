"""Two-tier AI analysis: Python cleaning/segmentation, TypeSafe Jev labelling (mocked) and Gemini (mocked)."""

import io
import json

import pytest
from fake_project import FPS, TEAMS, _row, make_project

from replay import ai_analysis as ai
from replay import ai_light as L

JEV_KEY = "tsk_TESTKEY_abcdefghijklmnopqrstuvwxyz"
N = 400  # 40 s at 10 fps


def _replay_tracks_and_ball(rows):
    """replay_data.json tracks + ball built from the same rows as tracking.json (as export.viewer_payload does),
    so the game state (replay/game_state.py reads replay_data) sees the same holders as ai_light."""
    tracks, ball = {}, {"x": [None] * N, "y": [None] * N, "f": [0] * N, "b": [None] * N, "o": [None] * N}
    for r in rows:
        k, tid, typ, team, x, y, state, owner = r[0] // 3, r[2], r[3], r[4], r[5], r[6], r[10], r[11]
        if typ == "ball":
            ball["x"][k], ball["y"][k] = round(x * 100), round(y * 100)
            ball["f"][k] = 1 if state == "possessed" else 2
            ball["o"][k] = str(owner) if owner != "" else None
            continue
        tr = tracks.setdefault(tid, {"id": tid, "team": team, "type": typ, "i0": k, "x": [], "y": [], "f": []})
        j = k - tr["i0"]
        for key, fill in (("x", None), ("y", None), ("f", 0)):
            tr[key] += [fill] * (j + 1 - len(tr[key]))
        tr["x"][j], tr["y"][j], tr["f"][j] = round(x * 100), round(y * 100), 1
    return {"tracks": list(tracks.values()), "ball": ball}


def make_long_project(root):
    """A: holds the ball 0-10 s attacking L (x 20 -> 5); nobody 10-16 s; B: 16-30 s attacking R (x 5 -> 26);
    track 5 = ghost of track 1; track 6 = bench person; one impossible jump of track 2; no calibration 34-36 s."""
    make_project(root)
    rows, calib = [], []
    for k in range(N):
        t = k / FPS
        cal = not (34.0 <= t < 36.0)
        calib.append(1 if cal else 0)
        if not cal:
            continue
        ax = 20 - 1.5 * min(t, 10) if t < 16 else 6 + (t - 16) * 1.5
        rows.append(_row(k, 1, "player", "A", round(ax, 2), 7.0))
        rows.append(_row(k, 5, "player", "A", round(ax + 0.1, 2), 7.1))  # ghost
        x2 = 12.0 if k != 50 else 20.0  # one 8 m jump at t = 5.0
        rows.append(_row(k, 2, "player", "A", x2, 4.0))
        rows.append(_row(k, 3, "player", "B", round(ax - 1.0, 2), 8.0))
        rows.append(_row(k, 4, "player", "B", 14.0, 10.0))
        rows.append(_row(k, 6, "player", "B", 10.0, -1.5))  # bench
        rows.append(_row(k, 7, "referee", "referee", 14.0, 0.5))
        if t < 10:
            rows.append(_row(k, "ball", "ball", "", round(ax, 2), 7.0, state="possessed", owner=1))
        elif 16 <= t < 30:
            rows.append(_row(k, "ball", "ball", "", round(ax - 1.0, 2), 8.0, state="possessed", owner=3))
    res = root / "results"
    tr = json.loads((res / "tracking.json").read_text(encoding="utf-8"))
    tr["rows"] = rows
    tr["events"] = [{"id": "basket-270-R", "type": "basket", "t": 27.0, "frame": 810, "basket": "R", "team": "B",
                     "confidence": 0.9}]
    tr["tracks"] = {str(i): {"team": t} for i, t in [(1, "A"), (2, "A"), (3, "B"), (4, "B"), (5, "A"), (6, "B"),
                                                      (7, "referee")]}
    (res / "tracking.json").write_text(json.dumps(tr), encoding="utf-8")
    rd = json.loads((res / "replay_data.json").read_text(encoding="utf-8"))
    rd.update(t=[round(k / FPS, 3) for k in range(N)], frame=[3 * k for k in range(N)], calib=calib, n=N,
              events=tr["events"], game={"teams": TEAMS}, **_replay_tracks_and_ball(rows))
    (res / "replay_data.json").write_text(json.dumps(rd), encoding="utf-8")
    return root


@pytest.fixture
def proj(tmp_path):
    return make_long_project(tmp_path / "long")


def test_cleaning_finds_ghost_bench_glitch_and_gaps(proj):
    g = L.load_game(proj)
    c = L.clean_data(g)
    assert c.ghost_tracks == {"5": "1"}
    assert c.bench_tracks == {"6"}
    assert ("2", 5.0) in c.glitch_rows and len([r for r in c.glitch_rows if r[0] == "2"]) == 1
    assert c.calib_gaps == [(34.0, 36.0)]
    assert c.excluded_tracks == {"5", "6"}
    d = c.to_dict()
    assert d["decided_by"] == {"5": "python", "6": "python"}


def test_segmentation_and_rule_labels(proj):
    g = L.load_game(proj)
    c = L.clean_data(g)
    # game-state path (possessions from replay/game_state.py; model v6 tiles live play, so ends differ from the rules)
    gsegs = L.segment_game(g, c, coached="A")
    ga = gsegs[0]
    gb = next(s for s in gsegs if s["ball_team"] == "opponent")
    assert ga["ball_team"] == "coached" and ga["t0"] == 0.0 and 9.0 <= ga["t1"] <= 16.0 and ga["attacks"] == "L"
    assert gb["attacks"] == "R" and gb["attack_basket_source"] == "game state" and gb["possession_change"]
    assert gb["baskets"] == [{"t": 27.0, "team": "opponent"}] and gb["phase"] == "transition_defense"
    # a live-play possession whose team is unclear (v6) is "unclear", never a dead ball
    assert all(s["phase"] == "unclear" for s in gsegs if s.get("kind") == "possession" and s["ball_team"] == "none")
    # rule segmentation from the ball holders (used when no game state is available)
    segs = L._segment_game_raw(g, c, coached="A")
    kinds = [(s["ball_team"], s["phase"]) for s in segs]
    assert kinds[0] == ("coached", "offense_half_court") or kinds[0][0] == "coached"
    a = next(s for s in segs if s["ball_team"] == "coached")
    b = next(s for s in segs if s["ball_team"] == "opponent")
    assert a["t0"] == 0.0 and 9.0 <= a["t1"] <= 10.0 and a["attacks"] == "L"
    assert b["attacks"] == "R" and b["attack_basket_source"] == "detected baskets"
    assert b["started_in_back_court"] and b["crossed_half_after_s"] is not None and b["crossed_half_after_s"] <= 8
    assert b["phase"] == "transition_defense" and b["possession_change"]
    assert any(s["ball_team"] == "none" for s in segs if 10 <= s["t0"] < 16)
    assert b["baskets"] == [{"t": 27.0, "team": "opponent"}]
    assert all(s["review"] is None and s["phase_source"] == "rules" for s in segs)
    table = L.segment_table(segs)
    assert table.splitlines()[0].startswith("id|t0|t1|phase") and len(table.splitlines()) == len(segs) + 1


# ---------------------------------------------------------------------- Jev client (mocked HTTP)
class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_jev_client_request_format(monkeypatch):
    sent = {}

    def fake(req, timeout):
        sent.update(url=req.full_url, headers=dict(req.header_items()), body=json.loads(req.data))
        return _Resp(json.dumps({"model": "jev-1.13.0", "answers": {"q": {"type": "noul", "noul": 0.9}},
                                 "usage": {"input_tokens": 300, "output_tokens": 20}}).encode())
    monkeypatch.setattr("urllib.request.urlopen", fake)
    c = L.JevClient(JEV_KEY)
    assert JEV_KEY not in repr(c)
    out = c.evaluate({"s": 1}, {"q": {"type": "noul", "instructions": "x"}})
    assert out == {"q": {"type": "noul", "noul": 0.9}}
    assert sent["url"] == "https://api.typesafe.ai/v1/systemone"
    assert sent["headers"]["Authorization"] == f"Bearer {JEV_KEY}"
    assert sent["body"] == {"state": {"s": 1}, "model": "jev-latest", "questions": {"q": {"type": "noul",
                                                                                          "instructions": "x"}}}
    assert c.usage == {"input_tokens": 300, "output_tokens": 20, "requests": 1} and c.model_used == "jev-1.13.0"
    with pytest.raises(L.LightError):
        L.JevClient("")
    with pytest.raises(L.LightError):
        L.JevClient(JEV_KEY, base_url="http://evil.example.com")


@pytest.mark.parametrize("code,msg", [(401, "rejected the API key"), (422, "422"), (429, "rate limit"),
                                     (529, "overloaded")])
def test_jev_errors(monkeypatch, code, msg):
    import urllib.error

    def fake(req, timeout):
        raise urllib.error.HTTPError(req.full_url, code, "x", {}, io.BytesIO(f"bad {JEV_KEY}".encode()))
    monkeypatch.setattr("urllib.request.urlopen", fake)
    with pytest.raises(L.LightError, match=msg) as e:
        L.JevClient(JEV_KEY).evaluate("s", {})
    assert JEV_KEY not in str(e.value)


class FakeJev:
    """Answers every question: phase = the rule label flipped to the coached view, review high only for s-ids
    listed, reliable 0.8; tracks -> bench."""

    def __init__(self, review_ids=(), fail=None):
        self.review_ids, self.fail = set(review_ids), fail
        self.calls = []
        self.model, self.model_used = "jev-latest", "jev-1.13.0"
        self.usage = {"input_tokens": 0, "output_tokens": 0, "requests": 0}

    def evaluate(self, state, questions):
        if self.fail:
            raise L.LightError(self.fail)
        self.calls.append((state, questions))
        self.usage["requests"] += 1
        self.usage["input_tokens"] += 100 * len(questions)
        out = {}
        for qid, q in questions.items():
            sid, _, kind = qid.partition("__")
            if qid.startswith("trk_"):
                out[qid] = {"type": "choice", "choice": "bench_or_sideline", "confidence": 0.9}
            elif kind == "phase":
                out[qid] = {"type": "choice", "choice": "dead_ball" if q["instructions"]["segment"]["ball_team"]
                            == "none" else "transition_defense", "confidence": 0.77}
            elif kind == "scheme":
                out[qid] = {"type": "choice", "choice": "zone", "confidence": 0.6}
            elif kind == "review":
                out[qid] = {"type": "noul", "noul": 0.9 if sid in self.review_ids else 0.1}
            elif kind == "reliable":
                out[qid] = {"type": "noul", "noul": 0.8}
        return out


def test_label_with_jev_updates_segments(proj):
    g = L.load_game(proj)
    c = L.clean_data(g)
    c.ambiguous_tracks["4"] = L._track_facts(g, "4", [r for r in g.players if r["track"] == "4"])
    segs = L.segment_game(g, c, "A")
    jev = FakeJev(review_ids={"s1"})
    r = L.label_with_jev(jev, g, "A", segs, c)
    state, qs = jev.calls[0]
    assert state["coached_team"] == "Kouvot" and state["court_m"]["length"] == 28
    assert {q["type"] for q in qs.values()} == {"choice", "noul"}
    assert qs["s0__phase"]["criteria"] == L.PHASES and "segment" in qs["s0__phase"]["instructions"]
    assert r["segments_labelled"] == len(segs) and r["tracks_decided"] == 1
    assert all(s["phase_source"] == "jev" and s["reliable"] == 0.8 and s["defense_scheme"] == "zone" for s in segs)
    assert segs[1]["review"] == 0.9 and segs[0]["review"] == 0.1
    assert "4" in c.bench_tracks and c.decided_by["4"] == "jev" and not c.ambiguous_tracks


def test_review_windows():
    segs = [{"t0": 0, "t1": 10, "review": 0.1}, {"t0": 10, "t1": 20, "review": 0.9},
            {"t0": 20, "t1": 30, "review": 0.2, "baskets": [1]}, {"t0": 30, "t1": 40, "review": 0.1},
            {"t0": 40, "t1": 50, "review": 0.1}, {"t0": 50, "t1": 60, "review": 0.1}]
    assert ai.review_windows(segs) == [(8, 32)]
    assert ai.review_windows([{"t0": 0, "t1": 1, "review": None}]) is None
    assert ai.review_windows([{"t0": 0, "t1": 1, "review": 0.9}]) is None  # everything selected


# ---------------------------------------------------------------------- whole two-tier run
GOOD = json.dumps({"overview": {"went_well": ["w"], "to_improve": ["i"], "summary": "s"},
                   "issues": [{"t_start": 17, "t_end": 25, "title": "Slow getting back", "description": "d",
                               "category": "transition defense", "team_phase": "transition"}]})


class FakeGemini:
    def __init__(self):
        self.calls = []

    def generate(self, model, system, user, schema):
        self.calls.append(user)
        return GOOD


def test_two_tier_run_with_jev(proj):
    g = L.load_game(proj)
    sel = next(s["id"] for s in L.segment_game(g, L.clean_data(g), "A")  # the opponent possession with the basket
               if s["ball_team"] == "opponent" and s.get("baskets"))
    gem, jev = FakeGemini(), FakeJev(review_ids={sel})
    res = ai.run_analysis(proj, ai.Options(coached_team="A"), gem, light_client=jev)
    user = gem.calls[0]
    assert "## SEGMENTS" in user and "|jev0.77|" in user
    assert "ONLY around the segments selected" in user
    sample_ts = [float(ln.split("|")[0]) for ln in user.split("## TRACKING SAMPLES\n")[1].splitlines()[1:] if ln]
    s1 = next(s for s in res["segments"] if s["id"] == sel)
    assert sample_ts and all(s1["t0"] - 2.01 <= t <= s1["t1"] + 2.01 or t >= 25 for t in sample_ts)
    assert min(sample_ts) > 10  # the unselected coached possession (0-10 s) gets no raw samples
    assert " 5:" not in user and " 6:" not in user  # ghost and bench tracks are not sent
    assert "impossible jumps" in user
    tiers = [t["tier"] for t in res["tiers"]]
    assert tiers == ["python", "python", "light", "gemini"]
    assert res["light_model"]["used"] and res["light_model"]["model"] == "jev-1.13.0"
    assert res["raw_sample_windows_s"] and res["cleaning"]["duplicate_tracks"] == {"5": "1"}
    assert res["issues"][0]["title"] == "Slow getting back"
    assert JEV_KEY not in json.dumps(res)


def test_two_tier_falls_back_without_or_with_failing_jev(proj):
    gem = FakeGemini()
    res = ai.run_analysis(proj, ai.Options(coached_team="A"), gem)
    assert [t["tier"] for t in res["tiers"]] == ["python", "python", "gemini"]
    assert not res["light_model"]["used"] and res["raw_sample_windows_s"] is None
    assert any("not configured" in w for w in res["warnings"])
    assert "|rules|" in gem.calls[0] and "ONLY around" not in gem.calls[0]

    gem = FakeGemini()
    res = ai.run_analysis(proj, ai.Options(coached_team="A"), gem,
                          light_client=FakeJev(fail="TypeSafe AI rejected the API key (401)."))
    assert any("Light-work model failed" in w and "401" in w for w in res["warnings"])
    assert not res["light_model"]["used"] and "|rules|" in gem.calls[0]


def test_key_storage_for_both_providers(tmp_path):
    pytest.importorskip("PySide6")
    from PySide6.QtCore import QSettings
    from replay.gui import ai_dialog
    qs = QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat)
    ai_dialog.save_api_key(qs, "gem-key-1234567890", ai_dialog.KEY_SETTING)
    ai_dialog.save_api_key(qs, JEV_KEY, ai_dialog.LIGHT_KEY_SETTING)
    qs.sync()
    text = (tmp_path / "s.ini").read_text(encoding="utf-8", errors="ignore")
    assert JEV_KEY not in text and "gem-key-1234567890" not in text
    assert ai_dialog.load_api_key(qs, ai_dialog.LIGHT_KEY_SETTING) == JEV_KEY
    assert ai_dialog.load_api_key(qs) == "gem-key-1234567890"
