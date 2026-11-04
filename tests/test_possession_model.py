"""Possession model v6 (replay/game_state.py, data/plans/possession_design.md): possessions are segments between
hard boundaries (makes, shots reaching the rim, dead balls, confirmed holder changes), their team an evidence vote.
Every confirmed make lies inside a possession of the scoring team by construction. Opt-in: analyse(v6=True) or
REPLAY_POSSESSION_V6=1."""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import pytest

from replay import coach as C
from replay import game_state as gs
from replay import stats as S
from tests.test_game_state import BL, BR, FPS, L, MID, Replay, W, still


def make_event(t: float, side: str, team: str | None, k: int) -> dict:
    return {"id": f"basket-{k}-{side}", "type": "basket", "t": t, "frame": int(t * 30), "basket": side,
            "team": team, "confidence": 0.9, "source": "auto"}


def base(seconds: float) -> Replay:
    """A attacks L (established by A's make at L at 5.6 s): A "1" holds 0-4 s, shoots, scores."""
    r = Replay(seconds)
    r.player("1", "A", 0, seconds, lambda t: (10 - 1.5 * min(t, 4), 6.0))
    r.player("2", "A", 0, seconds, lambda t: (12.0, 10.0))
    r.player("3", "B", 0, seconds, lambda t: (9.0 + 1.2 * min(max(t - 7, 0), 7), 7.0))
    r.player("4", "B", 0, seconds, lambda t: (L - 3.0, 9.0))
    r.player("5", "referee", 0, seconds, still(14.0, 0.5), typ="referee")
    r.hold("1", 0.0, 4.0)
    r.fly(4.0, 5.5, BL)
    r.events.append(make_event(5.6, "L", "A", 1))
    return r


def poss_at(d: dict, t: float, team: str | None = None) -> dict | None:
    return next((p for p in d["possessions"] if p["t0"] - 0.05 <= t <= p["t1"] + 0.05
                 and (team is None or p["team"] == team)), None)


def test_off_is_unchanged_and_on_is_versioned():
    r = base(12)
    off = gs.analyse(r.doc(), v6=False)
    on = gs.analyse(r.doc(), v6=True)
    assert off["engine_version"] == gs.ENGINE_VERSION and "possession_model" not in off
    assert on["engine_version"] == gs.ENGINE_VERSION_V6 and on["possession_model"]["version"] == 6
    assert all("outcome" in p for p in on["possessions"]) and all("outcome" not in p for p in off["possessions"])


def test_make_splits_possession():
    """B holds, A holds briefly (4 frames) + rim make at A's basket, B inbounds -> 3 possessions around the
    make, A's contains it (C2 / C4)."""
    r = base(32)
    r.hold("3", 7.0, 15.0)  # B brings the ball up
    r.fly(15.0, 16.5, BR, frm=(20.0, 7.0))  # B misses at R
    r.player("8", "A", 0, 32, still(5.0, 9.0))
    r.hold("8", 17.0, 17.4)  # A seen with the ball for 0.4 s only ...
    r.fly(17.4, 19.5, BL, frm=(5.0, 9.0))  # ... and scores at L
    r.events.append(make_event(19.6, "L", "A", 2))
    r.hold("3", 21.0, 30.0)  # B inbounds
    d = gs.analyse(r.doc(), v6=True)
    teams = [p["team"] for p in d["possessions"] if p["team"]]
    assert teams == ["A", "B", "A", "B"], d["possessions"]
    pa = poss_at(d, 19.6, "A")
    assert pa is not None and pa["outcome"] == "made" and pa["points"] == 2
    assert poss_at(d, 25.0)["team"] == "B" and poss_at(d, 25.0)["start"] == "after made basket"
    assert d["basket_annotations"]["basket-2-L"]["team"] == "A"


def test_shot_after_last_holder():
    """Holder seen until t=5, the ball unseen, release ~8, make at 10 -> the possession reaches the make (C3)."""
    r = Replay(16)
    r.player("1", "A", 0, 16, lambda t: (10 - 1.0 * min(t, 5), 6.0))
    r.player("3", "B", 0, 16, still(9.0, 9.0))
    r.player("4", "B", 0, 16, still(16.0, 9.0))
    r.hold("1", 0.0, 5.0)
    r.fly(8.0, 10.0, BL, frm=(5.0, 6.0))
    r.events.append(make_event(10.1, "L", "A", 1))
    d = gs.analyse(r.doc(), v6=True)
    p = poss_at(d, 10.1, "A")
    assert p is not None and p["t1"] >= 10.0 and p["outcome"] == "made"


def test_mislabelled_holders_in_the_attacked_paint():
    """Tracks labelled B hold the ball 6 s inside the paint A attacks, then A scores there: the possession is A's
    (make + location beat the labels) and the shot is credited to A (C1 / C6)."""
    r = base(30)
    r.player("9", "B", 0, 30, still(3.5, 7.0))  # "B" standing next to A's basket (mislabelled A player)
    r.hold("9", 10.0, 16.0)
    r.fly(16.0, 17.5, BL, frm=(3.5, 7.0))
    r.events.append(make_event(17.6, "L", None, 2))
    r.hold("3", 19.0, 26.0)
    d = gs.analyse(r.doc(), v6=True)
    p = poss_at(d, 17.6)
    assert p["team"] == "A" and p["outcome"] == "made", p
    assert p["evidence"]["make"] > 0 and p["evidence"]["location"] > 0 and p["evidence"]["holder"] < 0
    assert d["basket_annotations"]["basket-2-L"]["team"] == "A"
    assert poss_at(d, 22.0)["team"] == "B"


def test_scorebar_boundary_without_rim_detection():
    """Two detected A makes map bar side 1 to A; a third +2 on side 1 at t=50 with no detection becomes a make
    boundary ~4 s before the change: A's possession ends "made" there, B's starts after it (C7)."""
    r = Replay(64)
    r.player("1", "A", 0, 64, lambda t: (10 - 1.5 * min(t, 4), 6.0))
    r.player("2", "A", 0, 64, still(12.0, 10.0))
    r.player("3", "B", 0, 64, still(15.0, 7.0))
    r.player("4", "B", 0, 64, still(L - 3.0, 9.0))
    r.player("5", "referee", 0, 64, still(14.0, 0.5), typ="referee")
    r.hold("1", 0.0, 4.0)
    r.fly(4.0, 5.5, BL)
    r.events.append(make_event(5.6, "L", "A", 1))
    r.hold("3", 8.0, 20.0)
    r.hold("2", 22.0, 28.0)
    r.fly(28.0, 29.5, BL, frm=(12.0, 10.0))
    r.events.append(make_event(29.6, "L", "A", 2))
    r.hold("3", 32.0, 38.0)
    r.hold("2", 39.0, 45.0)  # A has the ball, the ball is lost from view, the bar adds 2 for A at 50
    r.hold("3", 52.0, 62.0)
    bar = {"timeline": [[0.0, 0, 0], [10.0, 0, 2], [34.0, 0, 4], [50.0, 0, 6]]}
    d = gs.analyse(r.doc(), v6=True, scorebar=bar)
    pm = d["possession_model"]
    assert pm["bar_used"] and pm["side_team"] == {1: "A", 0: "B"} and pm["synthetic"] == 1
    p = poss_at(d, 45.9, "A")
    assert p is not None and p["outcome"] == "made" and p["points"] == 2 and abs(p["t1"] - 46.0) < 0.2
    assert any(s.get("synthetic") for s in p["shots"])
    q = poss_at(d, 55.0)
    assert q["team"] == "B" and q["start"] == "after made basket"
    assert d["basket_annotations"]["basket-2-L"]["team_source"] == "score bar"


def ft_after_live_make() -> Replay:
    """A scores at R at 9.6 s, then shoots two free throws at R from ~16 s; everybody else stands still the whole
    time (so the free-throw set-up would reach back over the make without the guard)."""
    r = Replay(40)
    ftx = L - 5.8
    # shooter: walks in along y to the free-throw spot (arrives at 15 s), holds the ball 2-8 s on the way
    r.player("1", "A", 0, 40, lambda t: (ftx - 0.3, MID + 7.4 * max(0.0, 1 - t / 15)))
    for k, (dx, side) in enumerate([(1.8, -1), (2.8, -1), (3.8, -1), (1.8, 1), (2.8, 1), (3.8, 1)]):
        r.player(str(10 + k), "B" if k % 2 == 0 else "A", 0, 40, still(L - dx, MID + side * 2.9))
    r.player("20", "A", 0, 40, still(16.0, 4.0))
    r.player("21", "B", 0, 40, still(16.0, 11.0))
    r.player("30", "referee", 0, 40, still(25.0, 1.0), typ="referee")
    r.hold("1", 2.0, 8.0)
    r.fly(8.0, 9.5, BR)
    r.events.append(make_event(9.6, "R", "A", 1))
    r.hold("1", 16.0, 20.0)
    r.fly(20.0, 21.0, BR)
    r.hold("1", 23.0, 26.0)
    r.fly(26.0, 27.0, BR)
    r.events.append(make_event(27.1, "R", "A", 2))
    r.hold("21", 29.0, 36.0)
    return r


def test_ft_window_does_not_swallow_live_make():
    r = ft_after_live_make()
    on = gs.analyse(r.doc(), v6=True)
    w = on["free_throw_windows"][0]
    assert w["side"] == "R" and w["t_setup"] >= 10.5, w
    assert on["basket_annotations"]["basket-1-R"]["kind"] == "field_goal"
    p = poss_at(on, 9.6, "A")
    assert p is not None and p["outcome"] in ("made", "ft") and p["points"] >= 2
    assert poss_at(on, 27.1, "A") is not None  # the made free throw lies in A's possession too
    off = gs.analyse(r.doc(), v6=False)
    assert off["free_throw_windows"][0]["t_setup"] < 9.0  # (C5: the old set-up reaches back over the make)


def rebound_scene(offensive: bool) -> Replay:
    r = base(32)
    r.hold("3", 7.0, 14.0)
    r.fly(14.0, 15.5, BR, frm=(17.4, 7.0))  # B misses at R
    if offensive:
        r.player("6", "B", 0, 32, still(L - 3.0, 6.5))
        r.hold("6", 16.0, 22.0)  # B rebounds under its own basket
        r.fly(22.0, 23.5, BR, frm=(L - 3.0, 6.5))
        r.hold("2", 25.0, 30.0)
    else:
        r.player("7", "A", 0, 32, still(L - 3.0, 6.5))
        r.hold("7", 16.0, 19.0)  # A rebounds and passes out
        r.hold("2", 19.5, 24.0)
    return r


def test_offensive_rebound_continues_possession():
    d = gs.analyse(rebound_scene(True).doc(), v6=True)
    p = poss_at(d, 10.0, "B")
    assert p is not None and p["t1"] >= 23.4 and p["off_rebounds"] == 1 and p["outcome"] == "missed", p
    assert poss_at(d, 27.0)["team"] == "A" and poss_at(d, 27.0)["start"] == "after shot"
    d = gs.analyse(rebound_scene(False).doc(), v6=True)
    p = poss_at(d, 10.0, "B")
    assert p is not None and abs(p["t1"] - 15.5) < 0.3 and p["outcome"] == "missed"
    q = poss_at(d, 18.0)
    assert q["team"] == "A" and q["start"] == "after shot" and q["t0"] == p["t1"]


def test_unclear_team_never_a_confident_wrong_team():
    r = Replay(10)
    for k in range(4):
        r.player(str(k + 1), "A" if k < 2 else "B", 0, 10, still(8.0 + 3 * k, 6.0 + k))
    d = gs.analyse(r.doc(), v6=True)
    assert [p["team"] for p in d["possessions"]] == [None]
    p = d["possessions"][0]
    assert p["outcome"] == "unknown" and p["confidence"] == 0.0 and p["points"] == 0
    assert not [e for e in d["events"] if e["type"] in ("steal", "turnover")]
    assert d["possession_model"]["n_unclear"] == 1


def _check_invariants(doc: dict, replay: dict, label: str) -> None:
    poss = doc["possessions"]
    ann = doc.get("basket_annotations") or {}
    for e in replay.get("events") or []:
        if e.get("type") != "basket" or e.get("id") is None:
            continue
        a = ann.get(str(e["id"])) or {}
        if a.get("team") not in ("A", "B") or a.get("kind") == "not_on_sheet":
            continue
        t = float(e["t"])
        assert any(p["team"] == a["team"] and p["t0"] - 0.5 <= t <= p["t1"] + 2.0 for p in poss), \
            f"{label}: make at {t} ({a['team']}, {a.get('team_source')}) outside its team's possessions"
    for p, q in zip(poss, poss[1:]):
        assert q["t0"] >= p["t1"] - 1e-6, f"{label}: overlap {p['id']} {q['id']}"
        if p["outcome"] == "made" and p.get("make_source") in ("bar", "sheet") and q["kind"] == "live" \
                and q["start"] == "after made basket":
            # after a confirmed make the other team inbounds (a rim-only "make" may really be a miss + put-back)
            assert q["team"] != p["team"], f"{label}: {p['id']} made, {q['id']} same team after it"


def _eval_dirs() -> list[Path]:
    out = [Path(p).parent for p in sorted(glob.glob("data/benchmark/runs/*/out/replay_data.json"))
           if "_t" not in Path(p).parent.parent.name][:8]
    extra = os.environ.get("REPLAY_POSSESSION_EVAL_DIR")  # KoriTV project copies (results folders)
    if extra:
        out += [Path(p).parent for p in sorted(glob.glob(os.path.join(extra, "*", "results", "replay_data.json")))]
    return out


@pytest.mark.parametrize("res", _eval_dirs() or [None], ids=lambda p: "none" if p is None else p.parent.name)
def test_invariants_on_real_data(res):
    if res is None:
        pytest.skip("no benchmark runs present")
    replay = json.loads((res / "replay_data.json").read_text(encoding="utf-8"))
    if len(replay.get("t") or []) > 40000 and not os.environ.get("REPLAY_POSSESSION_EVAL_DIR"):
        pytest.skip("full game: set REPLAY_POSSESSION_EVAL_DIR")
    inplay = gs._read(res / "inplay.json")
    old = gs._read(res / gs.FILENAME)
    rim_ball = ((old or {}).get("inputs") or {}).get("rim_ball") if isinstance(old, dict) else None
    from replay import scorebar
    sb = scorebar.load_result(res / "work")
    doc = gs.analyse(replay, inplay if isinstance(inplay, dict) else None, rim_ball, v6=True, scorebar=sb)
    _check_invariants(doc, replay, res.parent.name)
    live = [p for p in doc["possessions"] if p["kind"] == "live"]
    t0, t1 = doc["time_range_s"]
    cover = sum(p["t1"] - p["t0"] for p in doc["possessions"]) / max(1.0, t1 - t0)
    assert cover >= 0.5, f"{res.parent.name}: possessions cover only {cover:.2f} of the time"
    assert all(p["team"] is None or p["confidence"] > 0 for p in live)


def test_coach_rows_and_stats_merge_follow_the_model():
    poss = [{"id": "p0", "team": "A", "t0": 0.0, "t1": 10.0, "duration_s": 10.0, "kind": "live", "outcome": "made",
             "points": 2, "shots": [{"t": 8.5, "basket": "L", "made": True, "basket_event": "b1"}], "off_rebounds": 0,
             "evidence": {"make": 5.0, "holder": -3.0, "location": 1.0}, "confidence": 0.9, "start": "start"},
            {"id": "p1", "team": "B", "t0": 10.0, "t1": 18.0, "duration_s": 8.0, "kind": "live", "outcome": "turnover",
             "points": 0, "shots": [], "off_rebounds": 0, "evidence": {"holder": -4.0}, "confidence": 0.9,
             "start": "after made basket"},
            {"id": "p2", "team": "A", "t0": 18.0, "t1": 30.0, "duration_s": 12.0, "kind": "live", "outcome": "made",
             "points": 3, "shots": [{"t": 28.0, "basket": "L", "made": True, "basket_event": "b2"}], "off_rebounds": 1,
             "evidence": {"make": 6.0}, "confidence": 1.0, "start": "steal"},
            {"id": "p3", "team": None, "t0": 30.0, "t1": 36.0, "duration_s": 6.0, "kind": "live", "outcome": "unknown",
             "points": 0, "shots": [], "off_rebounds": 0, "evidence": {}, "confidence": 0.0, "start": "after made basket"},
            {"id": "p4", "team": "A", "t0": 36.0, "t1": 44.0, "duration_s": 8.0, "kind": "live", "outcome": "missed",
             "points": 0, "shots": [{"t": 43.0, "basket": "L", "made": False, "basket_event": None}], "off_rebounds": 0,
             "evidence": {"holder": 3.0}, "confidence": 0.8, "start": "after dead ball"},
            {"id": "p5", "team": "B", "t0": 44.0, "t1": 50.0, "duration_s": 6.0, "kind": "live", "outcome": "period_end",
             "points": 0, "shots": [], "off_rebounds": 0, "evidence": {"holder": -3.0}, "confidence": 0.8,
             "start": "after shot"}]
    gs_doc = {"possessions": poss, "events": [{"type": "steal", "team": "A", "t": 17.5, "confidence": 0.9}],
              "time_range_s": [0.0, 50.0], "breaks": [], "halves": [{"t0": 0.0, "t1": 50.0}]}
    st = {"shots": [{"t": 8.5, "t_rim": 9.5, "team": "A", "outcome": "made", "value": 2, "points": 2, "shooter": "7",
                     "basket_event": "b1"},
                    {"t": 28.0, "t_rim": 29.0, "team": "A", "outcome": "made", "value": 3, "points": 3, "shooter": None,
                     "basket_event": "b2"},
                    {"t": 43.0, "t_rim": 44.0, "team": "A", "outcome": "missed", "value": 2, "points": 0, "shooter": None}],
          "periods": [{"t0": 0.0, "t1": 50.0}]}
    who = C._Who(type("Cat", (), {"track_number": {"7": "12"}, "track_team": {"7": "A"}})())
    rows = C.possession_timeline(st, gs_doc, who)
    assert [r["outcome"] for r in rows] == ["made", "turnover", "made", "unknown", "miss", "period_end"]
    assert rows[0]["detail"].startswith("Made 2 by #12") and rows[0]["points"] == 2
    assert rows[1]["detail"] == "Turnover (steal)" and rows[2]["points"] == 3 and "1 off. reb." in rows[2]["detail"]
    assert rows[3]["team"] is None and "unclear" in rows[3]["detail"]
    assert "defensive rebound" in rows[4]["detail"]
    assert not [r for r in rows if "possession not detected" in r["detail"]]
    # a row never carries another team than the one scoring in it
    for r in rows:
        for s in st["shots"]:
            if s["outcome"] == "made" and r["t0"] - 0.5 <= s["t"] <= r["t1"] + 0.5:
                assert r["team"] == s["team"]
    # stats: never merged across a made basket / turnover; the unclear spell stays a row of its own
    m = S.merge_possessions(poss, 3.0, [], [])
    assert [(p["team"], p["outcome"]) for p in m] == [("A", "made"), ("B", "turnover"), ("A", "made"), (None, "unknown"),
                                                      ("A", "missed"), ("B", "period_end")]
    st2 = S.compute(gs_doc, [], {})
    assert (st2["game"]["A"]["possessions"], st2["game"]["B"]["possessions"]) == (3, 2)


def test_build_for_results_with_the_flag(tmp_path, monkeypatch):
    res = tmp_path / "results"
    res.mkdir()
    (res / "replay_data.json").write_text(json.dumps(base(12).doc()), encoding="utf-8")
    monkeypatch.setenv("REPLAY_POSSESSION_V6", "1")
    doc = gs.build_for_results(res)
    assert doc["engine_version"] == 6 and gs.is_current(res, gs._read(res / gs.FILENAME))
    monkeypatch.setenv("REPLAY_POSSESSION_V6", "0")
    assert not gs.is_current(res, gs._read(res / gs.FILENAME))  # the other model's file is rebuilt
    doc = gs.load_or_build(res)
    assert doc["engine_version"] == gs.ENGINE_VERSION


def test_possession_never_seen_held_is_capped():
    """A make with no holder ever seen (seen_s == 0) used to report 0.99: the team rests on the make alone."""
    r = Replay(12)
    r.player("1", "A", 0, 12, still(6.0, 6.0))
    r.player("3", "B", 0, 12, still(9.0, 7.0))
    r.fly(3.0, 5.5, BL, frm=(8.0, 7.0))
    r.events.append(make_event(5.6, "L", "A", 1))
    d = gs.analyse(r.doc(), v6=True)
    p = poss_at(d, 3.0, "A")
    assert p is not None and p["outcome"] == "made" and p["seen_s"] == 0
    assert p["confidence"] <= gs.UNSEEN_CONF_CAP
    assert all(q["confidence"] <= gs.UNSEEN_CONF_CAP for q in d["possessions"] if q["seen_s"] == 0)
    # a possession whose holder is seen keeps its full confidence
    d = gs.analyse(base(12).doc(), v6=True)
    p = poss_at(d, 2.0, "A")
    assert p["seen_s"] > 0 and p["confidence"] > gs.UNSEEN_CONF_CAP
