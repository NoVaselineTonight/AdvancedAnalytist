"""Basketball game state engine (replay/game_state.py): free throws, possessions, turnovers, rebounds, attack
direction per half, lineup slots, event merging for the timeline / exports / AI."""

from __future__ import annotations

import json

import pytest

from replay import game_state as gs
from replay.gui import timeline_model as tm

FPS = 10.0
L, W = 28.0, 15.0
MID = W / 2
BR = (L - 1.575, MID)  # right basket
BL = (1.575, MID)
TEAMS = {"A": {"label": "A", "name": "Kouvot", "primary": "#ff8800"},
         "B": {"label": "B", "name": "Honka", "primary": "#1144aa"}}


class Replay:
    """Builds a replay_data.json-like dict frame by frame."""

    def __init__(self, seconds: float):
        self.n = int(round(seconds * FPS))
        self.t = [round(k / FPS, 3) for k in range(self.n)]
        self.tracks: dict = {}
        self.ball = {"x": [None] * self.n, "y": [None] * self.n, "f": [0] * self.n, "b": [None] * self.n,
                     "o": [None] * self.n}
        self.events: list = []

    def player(self, tid, team, t0, t1, pos, typ="player"):
        """pos(t) -> (x, y) metres, visible in [t0, t1)."""
        i0, i1 = int(round(t0 * FPS)), int(round(t1 * FPS))
        tr = {"id": tid, "team": team, "type": typ, "i0": i0, "x": [], "y": [], "f": [], "b": []}
        for i in range(i0, min(i1, self.n)):
            x, y = pos(i / FPS)
            tr["x"].append(int(round(x * 100)))
            tr["y"].append(int(round(y * 100)))
            tr["f"].append(1)
            tr["b"].append([0, 0, 10, 40])
        self.tracks[tid] = tr

    def hold(self, tid, t0, t1):
        tr = self.tracks[tid]
        for i in range(int(round(t0 * FPS)), int(round(t1 * FPS))):
            j = i - tr["i0"]
            if 0 <= j < len(tr["x"]):
                self.ball["x"][i], self.ball["y"][i] = tr["x"][j], tr["y"][j]
                self.ball["f"][i], self.ball["o"][i], self.ball["b"][i] = 1, tid, [1, 1, 5, 5]

    def fly(self, t0, t1, to, frm=None):
        """Ball in flight (estimated) from ``frm`` (default: where it was) to ``to``; arriving at the basket floor
        point = a rim anchor of the ball track."""
        i0, i1 = int(round(t0 * FPS)), int(round(t1 * FPS))
        if frm is None:
            frm = (self.ball["x"][i0 - 1] / 100, self.ball["y"][i0 - 1] / 100)
        for i in range(i0, i1):
            u = min(1.0, (i - i0 + 1) / max(1, (i1 - i0 - 2)))
            x, y = frm[0] + (to[0] - frm[0]) * u, frm[1] + (to[1] - frm[1]) * u
            self.ball["x"][i], self.ball["y"][i] = int(round(x * 100)), int(round(y * 100))
            self.ball["f"][i], self.ball["o"][i], self.ball["b"][i] = 2, None, [1, 1, 5, 5]

    def doc(self):
        return {"game": {"teams": TEAMS, "court_standard": "FIBA"}, "fps": FPS, "t": self.t,
                "frame": [3 * k for k in range(self.n)], "calib": [1] * self.n,
                "court": {"length": L, "width": W, "baskets": [list(BL), list(BR)]},
                "tracks": list(self.tracks.values()), "ball": self.ball, "events": self.events, "n": self.n,
                "track_info": {}}


def still(x, y):
    return lambda t: (x, y)


def free_throw_replay():
    """A shoots two free throws at basket R (FT line x = 22.2): a still line-up for 14 s, the ball at the
    spot, two flights to the rim; then B gets the ball after the second (missed) attempt."""
    r = Replay(30)
    ftx = L - 5.8
    def run_after(x, y, t_go=12.5, vx=-3.0):  # still during the free throws, then running back up the court
        return lambda t: (x + (0 if t < t_go else vx * (t - t_go)), y)

    r.player("1", "A", 0, 30, run_after(ftx - 0.3, MID))  # shooter
    for k, (dx, side) in enumerate([(1.8, -1), (2.8, -1), (3.8, -1), (1.8, 1), (2.8, 1), (3.8, 1)]):
        team = "B" if k % 2 == 0 else "A"
        r.player(str(10 + k), team, 0, 30, run_after(L - dx, MID + side * 2.9))
    r.player("20", "A", 0, 30, run_after(16.0, 4.0))
    r.player("21", "B", 0, 30, run_after(16.0, 11.0))
    r.player("30", "referee", 0, 30, still(25.0, 1.0), typ="referee")
    r.hold("1", 1.0, 5.0)
    r.fly(5.0, 6.0, BR)
    r.hold("1", 8.0, 11.0)
    r.fly(11.0, 12.0, BR)
    r.hold("10", 12.5, 18.0)  # B rebounds and runs the other way
    r.events.append({"id": "basket-15-R", "type": "basket", "t": 5.8, "frame": 174, "basket": "R", "team": "A",
                     "confidence": 0.9, "source": "auto"})
    return r


def test_free_throws_are_detected_and_not_passes():
    d = gs.analyse(free_throw_replay().doc())
    assert len(d["free_throw_windows"]) == 1
    w = d["free_throw_windows"][0]
    assert w["side"] == "R" and w["team"] == "A"
    ft = [e for e in d["events"] if e["type"] == "free_throw"]
    assert [e["label"] for e in ft] == ["Free throw 1/2", "Free throw 2/2"]
    assert all(e["team"] == "A" for e in ft)
    # the made basket is annotated as a free throw worth 1 point
    a = d["basket_annotations"]["basket-15-R"]
    assert a["points_guess"] == 1 and a["kind"] == "free_throw"
    # nothing inside the free throws is a possession, a pass, a steal or a turnover
    for e in d["events"]:
        if w["t_setup"] <= e["t"] <= w["t1"]:
            assert e["type"] in ("free_throws", "free_throw"), e
    assert not [p for p in d["possessions"] if w["t0"] <= p["t0"] <= w["t1"]]
    seg = next(s for s in d["segments"] if s["kind"] == "free_throws")
    assert seg["t0"] <= 2.0 and seg["t1"] >= 11.0
    # A shoots free throws at R -> A attacks R
    assert d["attack_direction"][0]["A"] == "R" and d["attack_direction"][0]["B"] == "L"


def live_play(steal: bool):
    """A brings the ball up towards R (0-6 s), then either B steals it in the middle of the court, or A shoots
    and B rebounds; B goes the other way."""
    r = Replay(20)
    r.player("1", "A", 0, 20, lambda t: (8 + 1.5 * min(t, 6), 6.0))
    r.player("2", "A", 0, 20, lambda t: (10 + 1.5 * min(t, 6), 10.0))
    r.player("3", "B", 0, 20, lambda t: (9 + 1.5 * min(t, 6) - (0 if t < 6 else 1.8 * (t - 6)), 7.0))
    r.player("4", "B", 0, 20, lambda t: (20.0, 9.0))
    r.player("5", "referee", 0, 20, still(14.0, 0.5), typ="referee")
    r.hold("1", 0.0, 5.8)
    if steal:
        r.hold("3", 6.0, 12.0)
    else:
        r.fly(6.0, 7.4, BR, frm=(16.5, 6.0))
        r.hold("4", 7.8, 12.0)
    r.events = []
    return r


def test_steal_vs_rebound():
    cfg = gs.GSConfig()
    d = gs.analyse(live_play(steal=True).doc(), cfg=cfg)
    teams = [p["team"] for p in d["possessions"]]
    assert teams == ["A", "B"]
    st = [e for e in d["events"] if e["type"] == "steal"]
    assert len(st) == 1 and st[0]["team"] == "B" and abs(st[0]["t"] - 6.0) < 0.3
    assert d["possessions"][1]["start"] == "steal"

    d = gs.analyse(live_play(steal=False).doc(), cfg=cfg)
    types = [e["type"] for e in d["events"]]
    assert "shot" in types and "def_rebound" in types and "steal" not in types and "turnover" not in types
    shot = next(e for e in d["events"] if e["type"] == "shot")
    assert shot["team"] == "A" and shot["basket"] == "R"
    reb = next(e for e in d["events"] if e["type"] == "def_rebound")
    assert reb["team"] == "B"


def test_short_holder_mixups_are_not_possession_changes():
    r = live_play(steal=True)
    r.hold("3", 2.0, 2.3)  # 0.3 s "B holds" inside A's possession: overlap mix-up
    d = gs.analyse(r.doc())
    assert [p["team"] for p in d["possessions"]] == ["A", "B"]
    assert sum(e["type"] in ("steal", "turnover") for e in d["events"]) == 1


def test_attack_direction_switches_at_half_time():
    r = free_throw_replay()
    inplay = {"intervals": [[0.0, 14.0], [600.0, 700.0]]}  # 586 s break = half time
    d = gs.analyse(r.doc(), inplay=inplay)
    assert [b["kind"] for b in d["breaks"]] == ["half_time"]
    first, second = d["attack_direction"]
    assert first["A"] == "R" and second["A"] == "L" and second["B"] == "R"
    assert any(e["type"] == "half_time" for e in d["events"])
    assert gs.attacked_side(d["attack_direction"], "B", 600.5) == "R"


def test_lineup_keeps_off_screen_players_on_court():
    r = Replay(12)
    r.player("1", "A", 0, 3, lambda t: (20.0, 5.0))
    r.player("7", "A", 4, 12, lambda t: (21.0, 5.2))  # same player, new tracker ID after 1 s off screen
    r.player("2", "A", 0, 12, still(15.0, 10.0))
    r.player("3", "B", 0, 12, still(18.0, 7.0))
    d = gs.analyse(r.doc())
    lu = d["lineup"]
    assert lu["slot_of_track"]["1"] == lu["slot_of_track"]["7"]
    assert lu["slot_of_track"]["2"] != lu["slot_of_track"]["1"]
    gi = gs.load_input(r.doc())
    at = gs.lineup_at(gi, lu, int(3.5 * FPS))  # between the two tracks
    a = {s["slot"]: s for s in at["A"]}
    slot = lu["slot_of_track"]["1"]
    assert len(at["A"]) == 5 and len(at["B"]) == 5  # always 5 per team
    assert a[slot]["visible"] is False and a[slot]["last_x"] == pytest.approx(20.0, abs=0.2)


def test_merge_titles_and_timeline():
    d = gs.analyse(free_throw_replay().doc())
    replay_events = [{"id": "basket-15-R", "type": "basket", "t": 5.8, "basket": "R", "team": "B",
                      "confidence": 0.9, "source": "auto"}]
    merged = gs.merge_events(replay_events, d, [{"id": "jev-x", "type": "turnover", "t": 20.0, "team": "B",
                                                "label": "Turnover? (Jev)", "suffix": "", "source": "jev",
                                                "confidence": 0.7}])
    b = next(e for e in merged if e["id"] == "basket-15-R")
    assert b["team"] == "A" and b["team_detected"] == "B" and b["points_guess"] == 1  # credited to the shooter
    assert [e["t"] for e in merged] == sorted(e["t"] for e in merged)
    tl = tm.Timeline.default()
    tl.set_events(merged, {"A": {"name": "Kouvot Renamed", "color": "#ff8800"}, "B": {"name": "Honka"}})
    titles = [m.title for m in tl.auto_track.marks]
    assert "Made free throw 1/2 – Kouvot Renamed" in titles
    assert "Free throw 2/2 – Kouvot Renamed (result not seen)" in titles or \
        "Free throw 2/2 – Kouvot Renamed (missed)" in titles
    assert "Turnover? (Jev) – Honka" in titles
    # user marks survive re-runs, auto marks are replaced
    tl.add_mark(tl.tracks[1].id, 3.0, "my note")
    tl.set_events([], {})
    assert not tl.auto_track.marks and tl.tracks[1].marks[0].title == "my note"


def test_files_build_load_and_auto_events(tmp_path):
    res = tmp_path / "results"
    res.mkdir()
    (res / "replay_data.json").write_text(json.dumps(free_throw_replay().doc()), encoding="utf-8")
    doc = gs.build_for_results(res, rim_ball=[[50, "R", 0.1, -1.0, 0.8]])
    assert doc and gs.is_current(res, doc) and doc["inputs"]["rim_ball"]
    assert gs.load_or_build(res)["created"] == doc["created"]  # current file is reused
    (tmp_path / "results" / "ai_analysis.json").write_text(json.dumps({"light_events": [
        {"id": "jev-1", "type": "turnover", "t": 25.0, "team": "A", "label": "Turnover? (Jev)",
         "confidence": 0.8, "source": "jev"},
        {"id": "jev-2", "type": "turnover", "t": 26.0, "team": "A", "label": "Turnover? (Jev)",
         "confidence": 0.1, "source": "jev"}]}), encoding="utf-8")
    ev = gs.auto_events(tmp_path, json.loads((res / "replay_data.json").read_text(encoding="utf-8")))
    ids = {e["id"] for e in ev}
    assert "jev-1" in ids and "jev-2" not in ids  # very uncertain events stay off the timeline
    assert any(e["type"] == "free_throws" for e in ev)


def test_text_helpers():
    d = gs.analyse(free_throw_replay().doc())
    lines = gs.direction_lines(d, {"A": "Kouvot", "B": "Honka"})
    assert "Kouvot (A) attacks basket R" in lines[0]
    ev = gs.event_lines(d["events"], {"A": "Kouvot", "B": "Honka"})
    assert any("Free throw 1/2 – Kouvot" in ln for ln in ev)
    assert any("ALWAYS has 5 players" in s for s in gs.SEMANTICS)
    assert gs.segment_table(d["segments"]).splitlines()[0].startswith("id|t0|t1|kind")


def test_empty_and_robust():
    d = gs.analyse({"t": [], "tracks": [], "ball": None})
    assert d["events"] == [] and d["possessions"] == []
    r = Replay(5)
    r.player("1", "A", 0, 5, still(10, 5))
    d = gs.analyse(r.doc())  # no ball at all: no team possession (v6: one "unclear" spell, team None)
    assert all(p["team"] is None for p in d["possessions"]) and d["attack_direction"][0]["A"] is None


# ---------------------------------------------------------------------- integration: AI tiers and the export
def _project(tmp_path):
    """A Replay folder whose tracking.json and replay_data.json both hold the free-throw replay + live play."""
    from fake_project import _row, make_project

    root = make_project(tmp_path / "p")
    r = free_throw_replay()
    doc = r.doc()
    rows = []
    for tr in doc["tracks"]:
        for j, (x, y) in enumerate(zip(tr["x"], tr["y"])):
            k = tr["i0"] + j
            team = "referee" if tr["type"] == "referee" else tr["team"]
            rows.append(_row(k, int(tr["id"]), tr["type"], team, x / 100, y / 100))
    b = doc["ball"]
    for k in range(doc["n"]):
        if b["x"][k] is not None:
            st = "possessed" if b["f"][k] == 1 else "estimated"
            rows.append(_row(k, "ball", "ball", "", b["x"][k] / 100, b["y"][k] / 100, interp=st != "possessed",
                             state=st, owner=int(b["o"][k]) if b["o"][k] else None))
    res = root / "results"
    tr = json.loads((res / "tracking.json").read_text(encoding="utf-8"))
    tr["rows"], tr["events"] = rows, doc["events"]
    tr["tracks"] = {t["id"]: {"team": "referee" if t["type"] == "referee" else t["team"]} for t in doc["tracks"]}
    (res / "tracking.json").write_text(json.dumps(tr), encoding="utf-8")
    (res / "replay_data.json").write_text(json.dumps(doc), encoding="utf-8")
    return root


def test_ai_tiers_use_the_game_state(tmp_path):
    from replay import ai_analysis as ai
    from replay import ai_light as L

    root = _project(tmp_path)
    light = L.run_light(root, "A", None)
    kinds = [s.get("kind") for s in light.segments]
    assert "free_throws" in kinds and "possession" in kinds
    ft = next(s for s in light.segments if s["kind"] == "free_throws")
    assert ft["phase"] == "dead_ball" and ft["free_throws_by"] == "coached" and L.segment_questions(ft) == {}
    poss = next(s for s in light.segments if s["kind"] == "possession")
    qs = L.segment_questions(poss)
    assert f"{poss['id']}__lost_ball" in qs and qs[f"{poss['id']}__lost_ball"]["type"] == "noul"
    # Jev says the possession ended with a lost ball -> a light event for the timeline
    poss["lost_ball"] = 0.8
    ev = L.light_events(light.segments, "A", [])
    assert ev and ev[0]["source"] == "jev" and ev[0]["type"] == "turnover" and ev[0]["team"] == "B"
    assert not L.light_events(light.segments, "A", [{"type": "turnover", "t": poss["t1"]}])  # rules had it

    dg = ai.build_game_digest(root, None, 2.0, game=light.game, clean=light.clean)
    assert "## BASKETBALL RULES" in dg.header and "ALWAYS has 5 players" in dg.header
    assert "Kouvot (A) attacks basket R" in dg.header
    lines = [ln for _, ln in dg.frames]
    assert any("|st FT A@R dead|" in ln and "ball" in ln and " dead" in ln for ln in lines)
    assert any("|off " in ln for ln in lines) or all("A5" in ln or "|A " not in ln for ln in lines)
    user = ai.build_user_prompt(dg, dg.t_min, dg.t_max, lines, 1, 1, L.segment_table(light.segments))
    assert "## GAME EVENTS IN THIS PART" in user and "Free throw 1/2" in user
    system = ai.build_system_prompt(dg.teams, "A", "")
    assert "NOT a pass" in system and "5 players on court" in system


def test_export_has_game_state(tmp_path):
    from replay import data_export as dx

    root = _project(tmp_path)
    out = dx.export_simple(root, tmp_path / "exp")
    assert (out / "possessions.csv").exists() and (out / "game_state.json").exists()
    readme = (out / "README_FOR_AI.md").read_text(encoding="utf-8")
    assert "## 6b. Game state" in readme and "attacks basket R" in readme and "NOT a pass" in readme
    ev = (out / "events.csv").read_text(encoding="utf-8")
    assert "Made free throw 1/2" in ev and "free_throw" in ev
    # a free throw counts 1 point in the detected score
    doc = json.loads((out / "events.json").read_text(encoding="utf-8"))
    assert doc["detected_final_score_by_label"]["A"] == 1


def test_short_half_time_is_found_relative_to_other_breaks():
    cfg = gs.GSConfig()
    # youth game: 2 min quarter breaks, a 6.5 min half time (below halftime_min_s)
    inplay = {"intervals": [[0, 1100], [1220, 2170], [2570, 3680], [3760, 5100]]}
    brks = gs.breaks_from_inplay(inplay, 0, 5100, cfg)
    assert [b["kind"] for b in brks] == ["quarter_break", "half_time", "timeout"]
    # a single long-ish break alone is not called half time (could be a quarter break)
    assert gs.breaks_from_inplay({"intervals": [[0, 100], [400, 500]]}, 0, 500, cfg)[0]["kind"] != "half_time"


def test_attacker_of_and_basket_annotation_by_direction():
    d = [{"t0": 0.0, "t1": 100.0, "A": "R", "B": "L", "confidence": 0.6},
         {"t0": 200.0, "t1": 300.0, "A": "L", "B": "R", "confidence": 0.2}]
    assert gs.attacker_of(d, "R", 50.0, 0.3) == "A" and gs.attacker_of(d, "L", 50.0, 0.3) == "B"
    assert gs.attacker_of(d, "L", 250.0, 0.3) is None  # too unsure
    assert gs.attacker_of(d, "L", 150.0, 0.3) is None and gs.attacker_of(d, None, 50.0, 0.3) is None
    ann = {"b1": {"team": "A", "points_guess": 1, "label": "Made free throw 1/2", "title": "x"}}
    out = gs.merge_events([{"id": "b1", "type": "basket", "t": 10.0, "team": "B", "basket": "R"}], {"basket_annotations": ann})
    assert out[0]["team"] == "A" and out[0]["team_detected"] == "B" and out[0]["points_guess"] == 1


def test_half_time_in_the_middle_when_no_break_is_clearly_longest():
    # full game where the preliminary analysis gives similar breaks: the one in the middle third = half time
    inplay = {"intervals": [[0, 1100], [1220, 2174], [2240, 2423], [2572, 3686], [3768, 5114]]}
    brks = gs.breaks_from_inplay(inplay, 0, 5114, gs.GSConfig())
    assert [(b["t0"], b["kind"]) for b in brks if b["kind"] == "half_time"] == [(2423.0, "half_time")]
    # a short video never gets a half time from this rule
    short = {"intervals": [[0, 300], [420, 600], [720, 900]]}
    assert all(b["kind"] != "half_time" for b in gs.breaks_from_inplay(short, 0, 900, gs.GSConfig()))
