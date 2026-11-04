"""Ball events (replay/ball_events.py): passes, bad passes / interceptions, steals, shot enrichment, states."""

from __future__ import annotations

from replay import ball_events as be
from replay import game_state as gs
from tests.test_game_state import BR, FPS, Replay, live_play


def _boxes(doc: dict) -> dict:
    """Distinct image boxes per player (the shared fixture gives every player the same box)."""
    for tr in doc["tracks"]:
        tr["b"] = [[x * 0.4, 300 + y * 2, x * 0.4 + 30, 400 + y * 2] for x, y in zip(tr["x"], tr["y"])]
    return doc


def _run(r: Replay) -> dict:
    doc = _boxes(r.doc())
    g = gs.analyse(doc)
    be.attach(g, doc)
    return g


def _arc(r: Replay, t0: float, t1: float, p0: tuple, p1: tuple, bounce: bool = False, lob: bool = False):
    """Image boxes of a ball flying from image point p0 to p1 (bounce: down to the floor in the middle)."""
    i0, i1 = int(round(t0 * FPS)), int(round(t1 * FPS))
    for i in range(i0, i1):
        u = (i - i0 + 0.5) / (i1 - i0)
        x = p0[0] + (p1[0] - p0[0]) * u
        y = p0[1] + (p1[1] - p0[1]) * u
        if bounce:
            y += 120 * (1 - abs(2 * u - 1))
        if lob:
            y -= 160 * 4 * u * (1 - u)
        r.ball["b"][i] = [x - 5, y - 5, x + 5, y + 5]
        r.ball["f"][i], r.ball["o"][i] = 2, None


def passing_play(bounce=False, intercept=False):
    r = Replay(20)
    def move(x, y):  # live play: everybody moves ~1.5 m/s
        return lambda t: (x + 1.5 * (t % 2.0), y)

    r.player("1", "A", 0, 20, move(10.0, 5.0))
    r.player("2", "A", 0, 20, move(16.0, 9.0))
    r.player("3", "B", 0, 20, move(14.0, 7.5))
    r.player("4", "B", 0, 20, move(22.0, 7.0))
    r.hold("1", 0.0, 3.0)
    _arc(r, 3.0, 3.8, (300, 400), (600, 380), bounce=bounce)
    if intercept:
        r.hold("3", 3.8, 12.0)
    else:
        r.hold("2", 3.8, 12.0)
    return r


def test_completed_pass_and_type():
    g = _run(passing_play())
    p = g["ball_events"]["passes"]
    assert len(p) == 1 and p[0]["passer"] == "1" and p[0]["receiver"] == "2" and p[0]["team"] == "A"
    assert 5.0 < p[0]["length_m"] < 10.0 and p[0]["type"] == "direct"
    ev = [e for e in g["events"] if e["type"] == "pass"]
    assert len(ev) == 1 and ev[0]["title"].startswith("Pass T1 -> T2") and ev[0]["source"] == be.SOURCE
    assert not [e for e in g["events"] if e["type"] in ("steal", "turnover", "bad_pass")]
    st = {s[2] for s in g["ball_events"]["states"]}
    assert {"held", "air"} <= st


def test_bounce_pass():
    g = _run(passing_play(bounce=True))
    assert g["ball_events"]["passes"][0]["type"] == "bounce"


def test_jersey_number_in_title():
    r = passing_play()
    doc = _boxes(r.doc())
    doc["track_info"] = {"1": {"number": "7"}, "2": {"number": "12"}}
    g = gs.analyse(doc)
    be.attach(g, doc)
    assert any(e["title"].startswith("Pass #7 -> #12") for e in g["events"])


def test_bad_pass_intercepted():
    g = _run(passing_play(intercept=True))
    types = [e["type"] for e in g["events"]]
    assert "bad_pass" in types and types.count("steal") == 1 and "pass" not in types
    bp = next(e for e in g["events"] if e["type"] == "bad_pass")
    assert bp["team"] == "A" and bp["title"].startswith("Bad pass (stolen) by T1")
    st = next(e for e in g["events"] if e["type"] == "steal")
    assert st["team"] == "B" and st["details"]["kind"] == "interception" and st["source"] == be.SOURCE
    assert g["ball_events"]["turnovers"][0]["kind"] == "bad_pass_intercepted"


def test_steal_from_handler_replaces_game_state_steal():
    doc = _boxes(live_play(steal=True).doc())
    g = gs.analyse(doc)
    assert any(e["type"] == "steal" and e["source"] == gs.SOURCE for e in g["events"])
    be.attach(g, doc)
    steals = [e for e in g["events"] if e["type"] == "steal"]
    assert len(steals) == 1 and steals[0]["source"] == be.SOURCE and steals[0]["team"] == "B"
    assert steals[0]["details"]["kind"] == "steal" and steals[0]["title"].startswith("Steal by T3")
    assert any(e["type"] == "lost_ball" and e["team"] == "A" for e in g["events"])
    assert g["ball_events"]["replaced_game_state_events"]


def test_shot_enriched_no_pass_or_turnover():
    doc = _boxes(live_play(steal=False).doc())
    g = gs.analyse(doc)
    be.attach(g, doc)
    types = [e["type"] for e in g["events"]]
    assert "pass" not in types and "bad_pass" not in types and "steal" not in types
    shot = next(e for e in g["events"] if e["type"] == "shot")
    assert shot["title"].startswith("Shot (missed) from") and shot["details"]["outcome"] == "missed"
    s = g["ball_events"]["shots"][0]
    assert s["team"] == "A" and s["basket"] == "R" and s["position"] is not None and s["rebound_team"] == "B"
    tabs = be.tables(g["ball_events"])
    assert len(tabs["shots.csv"]) == 2 and tabs["shots.csv"][0][0] == "t_release"


def test_id_switch_is_not_a_pass():
    r = Replay(10)
    r.player("1", "A", 0, 4.0, lambda t: (10.0, 5.0))
    r.player("9", "A", 4.0, 10, lambda t: (10.3, 5.1))  # same player, new track id
    r.player("3", "B", 0, 10, lambda t: (20.0, 5.0))
    r.hold("1", 0.0, 3.9)
    r.hold("9", 4.2, 9.0)
    g = _run(r)
    assert not g["ball_events"]["passes"]


def test_game_state_file_rebuilds_with_ball_events(tmp_path):
    import json
    res = tmp_path / "results"
    res.mkdir()
    (res / "replay_data.json").write_text(json.dumps(_boxes(passing_play().doc())), encoding="utf-8")
    doc = gs.load_or_build(res)
    assert be.is_current(doc) and doc["ball_events"]["passes"]
    old = dict(doc)
    old.pop("ball_events")
    assert not gs.is_current(res, old)
    assert gs.is_current(res, json.loads((res / "game_state.json").read_text(encoding="utf-8")))
    assert BR  # (imported fixture constants stay in use)


def test_ball_speed_removes_camera_pan(tmp_path):
    import numpy as np
    n = 6
    t = np.arange(n) / FPS
    # the ball is still in the gym; the camera pans 30 px per frame, so the ball moves -30 px/frame in the image
    img = np.array([[600 - 30 * i, 300, 10.0] for i in range(n)])
    raw = be.ball_speed_mps(img, t)
    assert np.nanmin(raw) > 5.0  # 3 diameters per frame at 10 fps = 7.2 m/s without compensation
    G = np.stack([np.array([[1, 0, 30.0 * i], [0, 1, 0], [0, 0, 1]]) for i in range(n)])  # frame i -> frame 0
    be.save_motion(tmp_path, G, np.zeros(n, int))
    chain = be.load_motion(tmp_path, n)
    assert chain is not None and be.load_motion(tmp_path, n + 1) is None
    comp = be.ball_speed_mps(img, t, chain)
    assert np.nanmax(comp) < 0.1


def test_dribble_and_air_states():
    r = passing_play()
    doc = _boxes(r.doc())
    # player 1's box at frame 10 and a ball low in it (dribble)
    tr = next(tr for tr in doc["tracks"] if tr["id"] == "1")
    bx = tr["b"][10 - tr["i0"]]
    doc["ball"]["b"][10] = [bx[0] + 10, bx[3] - 15, bx[0] + 20, bx[3] - 5]
    g = gs.analyse(doc)
    be.attach(g, doc)
    st = g["ball_events"]["states"]
    assert any(s[2] == "dribble" and s[0] <= 1.0 <= s[1] for s in st)
    assert any(s[2] == "air" for s in st)


def test_unconfirmed_game_state_steal_is_demoted():
    doc = live_play(steal=True).doc()  # every player has the same image box: no ball-level evidence
    g = gs.analyse(doc)
    st0 = next(e for e in g["events"] if e["type"] == "steal")
    be.attach(g, doc)
    st = next(e for e in g["events"] if e["type"] == "steal")
    assert st["source"] == gs.SOURCE and st["confidence"] < st0["confidence"]
    assert "not confirmed" in st["title"] and g["ball_events"]["demoted_game_state_events"] == [st0["id"]]


def _flight_play(catch: bool):
    """A1 holds the ball, throws it (seen in the air above everybody's head); with ``catch`` A2 holds it for one
    frame only (too short for the holder-run logic), otherwise nobody catches it."""
    r = passing_play()
    for i in range(r.n):  # clear the ball
        r.ball["x"][i] = r.ball["y"][i] = r.ball["b"][i] = r.ball["o"][i] = None
        r.ball["f"][i] = 0
    r.hold("1", 0.0, 3.0)
    for i in range(30, 38):
        x = 60 + 40 * (i - 29)
        r.ball["b"][i] = [x - 5, 200, x + 5, 210]
        r.ball["f"][i] = 2
    if catch:
        r.hold("2", 3.8, 3.9)
    doc = _boxes(r.doc())
    for i in range(30):  # the held ball sits in A1's hands (chest height)
        tr = doc["tracks"][0]
        bx = tr["b"][i]
        doc["ball"]["b"][i] = [bx[0] + 10, bx[1] + 20, bx[0] + 20, bx[1] + 30]
    if catch:  # ... and in A2's hands when caught
        tr2 = doc["tracks"][1]
        bx = tr2["b"][38 - tr2["i0"]]
        doc["ball"]["b"][38] = [bx[0] + 10, bx[1] + 20, bx[0] + 20, bx[1] + 30]
    return doc


def test_flight_pass_found_without_long_holder_runs():
    doc = _flight_play(catch=True)
    g = gs.analyse(doc)
    d = be.analyse(doc, g)
    fl = [p for p in d["passes"] if p.get("source") == "flight"]
    assert len(fl) == 1 and fl[0]["passer"] == "1" and fl[0]["receiver"] == "2"
    d0 = be.analyse(doc, g, be.BEConfig(use_flights=False))
    assert not d0["passes"]


def test_bad_pass_out_of_bounds():
    doc = _flight_play(catch=False)
    gsd = {"possessions": [{"t0": 0.0, "t1": 3.0, "team": "A"}, {"t0": 6.0, "t1": 12.0, "team": "B"}],
           "events": [{"type": "inbound", "t": 6.0}], "shots": [], "dead_balls": []}
    d = be.analyse(doc, gsd)
    bp = [e for e in d["events"] if e["type"] == "bad_pass"]
    assert len(bp) == 1 and bp[0]["team"] == "A" and bp[0]["title"].startswith("Bad pass (out of bounds) by T1")
    assert d["turnovers"][0]["kind"] == "bad_pass_out_of_bounds"
    gsd["events"] = []  # no out-of-bounds call: just a loose ball
    assert not [e for e in be.analyse(doc, gsd)["events"] if e["type"] == "bad_pass"]
