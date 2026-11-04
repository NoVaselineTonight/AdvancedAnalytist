"""Lineup solver (replay/lineup.py): 5 continuous slots per team."""

import numpy as np

from replay import lineup
from replay.lineup import LineupConfig, _Piece, dead_mask, solve
from replay.teams import TeamResult


def _piece(tid, team, i0, i1, x0, x1, y=5.0, number=-1, fps=10.0):
    i = np.arange(i0, i1 + 1)
    x = np.linspace(x0, x1, len(i))
    return _Piece(tid, team, i, np.c_[x, np.full(len(i), y)], np.zeros(6), float(len(i)), number, 1.0)


def test_pieces_of_one_player_chain_into_one_slot():
    t = np.arange(300) / 10.0
    a = _piece(1, "A", 0, 49, 2, 7)  # 5 m in 5 s
    b = _piece(2, "A", 70, 120, 8, 12)  # reappears 2 s later 1 m further: same player
    c = _piece(3, "A", 0, 120, 20, 20, y=10)  # another player all the time
    slots, extras = solve([a, b, c], t, np.zeros(300, bool), LineupConfig())
    holding = {lab: [p.tid for p in ps] for lab, ps in slots.items() if ps}
    assert sorted(holding.values()) == [[1, 2], [3]]
    assert not extras


def test_never_more_than_five_per_team_and_extras_left_out():
    t = np.arange(100) / 10.0
    ps = [_piece(g, "A", 0, 99, 2 + 3 * g, 2 + 3 * g, number=-1) for g in range(1, 7)]  # 6 people all the time
    ps[5].quality = 1.0  # the weakest one
    slots, extras = solve(ps, t, np.zeros(100, bool), LineupConfig(flip_team_cost=None))
    assert sum(1 for lab, s in slots.items() if lab[0] == "A" and s) == 5
    assert [p.tid for p in extras] == [6]
    # with the other team short of people, the 6th 'A' is taken as a mis-teamed B player
    slots, extras = solve(ps, t, np.zeros(100, bool), LineupConfig(flip_team_cost=3.0))
    assert not extras and [p.tid for lab, s in slots.items() if lab[0] == "B" for p in s] == [6]


def test_extra_piece_is_cut_where_a_slot_is_free():
    t = np.arange(100) / 10.0
    ps = [_piece(g, "A", 0, 99, 2 + 3 * g, 2 + 3 * g) for g in range(1, 5)]
    ps.append(_piece(5, "A", 0, 49, 20, 20))  # 5th player leaves the picture at 5 s
    ps.append(_piece(6, "A", 30, 99, 24, 24))  # 6th person from 3 s: overlaps player 5 for 2 s only
    ps[-1].quality = 1.0
    slots, extras = solve(ps, t, np.zeros(100, bool), LineupConfig(flip_team_cost=None, hard_jump=False, split_rounds=2))
    shown6 = sum(len(p.i) for s in slots.values() for p in s if p.tid == 6)
    assert shown6 >= 45 and sum(len(p.i) for p in extras) <= 25


def test_unreachable_jump_is_not_the_same_slot():
    t = np.arange(100) / 10.0
    a = _piece(1, "B", 0, 30, 2, 2)
    b = _piece(2, "B", 32, 60, 20, 20)  # 18 m in 0.2 s: impossible
    slots, _ = solve([a, b], t, np.zeros(100, bool), LineupConfig())
    assert all(len(s) <= 1 for s in slots.values())


def test_different_numbers_only_across_dead_ball():
    t = np.arange(400) / 10.0
    cfg = LineupConfig(slots_per_team=2)
    a = _piece(1, "A", 0, 100, 5, 5, number=7)
    b = _piece(2, "A", 150, 250, 6, 6, number=9)
    slots, extras = solve([a, b], t, np.zeros(400, bool), cfg)
    assert not extras and all(len(s) <= 1 for s in slots.values())  # live play: no substitution
    dead = np.zeros(400, bool)
    dead[120:130] = True
    slots, extras = solve([a, b], t, dead, cfg)
    assert not extras and [p.tid for p in slots["A1"]] == [1, 2]


def test_same_number_wins_over_closer_position():
    t = np.arange(400) / 10.0
    a = _piece(1, "A", 0, 50, 5, 5, number=12)
    o = _piece(2, "A", 0, 50, 10, 10)
    b = _piece(3, "A", 100, 150, 10.5, 10.5, number=12)  # closer to piece 2's end, but wears 12
    slots, _ = solve([a, o, b], t, np.zeros(400, bool), LineupConfig(new_slot_cost=50.0))
    holding = sorted([p.tid for p in s] for s in slots.values() if s)
    assert [1, 3] in holding


def test_dead_mask_from_sample_gap_and_stillness():
    t = np.r_[np.arange(100) / 10.0, 30 + np.arange(100) / 10.0]
    # 5 players standing still in the second part, running in the first
    det_i, det_xy, tid = [], [], []
    for g in range(5):
        for i in range(200):
            det_i.append(i)
            x = 2 + 3 * g + (0.5 * t[i] if i < 100 else 0.0)
            det_xy.append((x, 5.0))
            tid.append(g + 1)
    det_i, det_xy, tid = np.array(det_i), np.array(det_xy), np.array(tid)
    dead, windows = dead_mask(t, det_i, det_xy, tid, np.ones(len(tid), bool), LineupConfig(still_mps=0.3))
    assert dead[99] and dead[100]  # the 20-s hole
    assert dead[150]  # standing still
    assert not dead[50]
    assert {k for _, _, k in windows} == {"break", "still"}


def test_apply_relabels_and_registers_slot_teams(tmp_path, monkeypatch):
    monkeypatch.setenv("REPLAY_LINEUP", "1")
    t = np.arange(200) / 10.0
    tid, det_i, xy = [], [], []
    for g, (i0, i1, x) in {1: (0, 60, 3.0), 2: (66, 199, 3.5), 3: (0, 199, 15.0), 9: (0, 199, 8.0)}.items():
        for i in range(i0, i1 + 1):
            tid.append(g)
            det_i.append(i)
            xy.append((x, 5.0))
    tid, det_i, xy = np.array(tid), np.array(det_i), np.array(xy)
    feat = np.zeros((len(tid), 6))
    teams = TeamResult({1: "A", 2: "A", 3: "B", 9: "referee"}, {1: 0.9, 2: 0.9, 3: 0.8, 9: 1.0}, {}, False, {})
    new_tid, keep = lineup.apply(tmp_path, tid, np.ones(len(tid), bool), det_i, xy, feat, None, t, teams)
    assert (new_tid[tid == 9] == 9).all()  # referees untouched
    assert len(set(new_tid[tid == 1]) | set(new_tid[tid == 2])) == 1  # one slot
    sid = int(new_tid[tid == 1][0])
    assert teams.track_team[sid] == "A" and keep.all()
    assert (tmp_path / "lineup.json").exists()
    info = {7: {}, 8: {}}
    lineup.add_track_info(info, {sid: 7, 9: 8})
    assert info[7]["slot"].startswith("A") and "slot" not in info[8]
    monkeypatch.setenv("REPLAY_LINEUP", "0")
    same, _ = lineup.apply(tmp_path, tid, np.ones(len(tid), bool), det_i, xy, feat, None, t, teams)
    assert (same == tid).all()


def test_hard_jump_veto():
    t = np.arange(100) / 10.0
    a = _piece(1, "A", 0, 40, 2, 2)
    b = _piece(2, "A", 45, 90, 15, 15)  # 13 m in 0.5 s
    slots, _ = solve([a, b], t, np.zeros(100, bool), LineupConfig(slots_per_team=1, new_slot_cost=100.0))
    assert [p.tid for p in slots["A1"]] in ([1], [2])


def test_unconfident_reattachment_gets_a_new_output_id(tmp_path, monkeypatch):
    monkeypatch.setenv("REPLAY_LINEUP", "1")
    monkeypatch.delenv("REPLAY_LINEUP_CFG", raising=False)
    t = np.arange(400) / 10.0
    tid, det_i, xy = [], [], []
    # A: piece 1 (0-5 s), piece 2 after a 1-s gap nearby (confident), piece 3 after a 10-s gap (not confident)
    for g, (i0, i1, x) in {1: (0, 50, 3.0), 2: (60, 150, 4.0), 3: (250, 399, 6.0)}.items():
        for i in range(i0, i1 + 1):
            tid.append(g)
            det_i.append(i)
            xy.append((x, 5.0))
    tid, det_i, xy = np.array(tid), np.array(det_i), np.array(xy)
    teams = TeamResult({1: "A", 2: "A", 3: "A"}, {1: 0.9, 2: 0.9, 3: 0.9}, {}, False, {})
    cfg = LineupConfig(enabled=True, slots_per_team=1)
    new_tid, _ = lineup.apply(tmp_path, tid, np.ones(len(tid), bool), det_i, xy, np.zeros((len(tid), 6)), None, t,
                              teams, cfg)
    assert new_tid[tid == 1][0] == new_tid[tid == 2][0] != new_tid[tid == 3][0]
    info = {}
    for g in set(new_tid.tolist()):
        info[g] = {}
    lineup.add_track_info(info, {g: g for g in info})
    assert {v["slot"] for v in info.values()} == {"A1"}


def test_out_of_view_slot_gets_short_prediction():
    from types import SimpleNamespace

    from replay.export import TrackRow

    n = 100
    t = np.arange(n) / 10.0
    data = SimpleNamespace(t=t, k=np.arange(n) * 3, frame=np.arange(n) * 3)
    lineup.SLOT_LABEL.clear()
    lineup.SLOT_LABEL[50] = "A1"
    rows = [TrackRow(frame=3 * i, t=t[i], k=3 * i, track_id=1, object_type="player", team="A", x=5 + 0.1 * i,
                     y=5.0, confidence=0.9, interpolated=False, calib="keyframe") for i in range(30)]
    det_i = np.repeat(np.arange(30), 5)
    det_xy = np.c_[np.tile(np.arange(5) * 3.0, 30) + 2, np.full(150, 5.0)]
    tid = np.tile(np.arange(1, 6), 30)
    pred = lineup.predicted_rows(rows, {50: 1}, data, [np.eye(3)] * n, det_i, det_xy, tid, np.ones(150, bool))
    lineup.SLOT_LABEL.clear()
    assert pred and all(r.track_id == 1 and r.interpolated for r in pred)
    assert min(r.k for r in pred) == 90 and len(pred) < 70  # starts right after the last row, stops when unsure
    assert all(getattr(r, "pos_sigma") <= LineupConfig().predict_max_sigma_m for r in pred)


def test_reid_vetoes_a_teammate_and_allows_a_long_gap_match():
    from replay.lineup import _Solver, confident

    t = np.arange(400) / 10.0
    solver = _Solver(t, np.zeros(400, bool), LineupConfig())
    rng = np.random.default_rng(0)
    me, mate = rng.normal(size=128), rng.normal(size=128)

    def emb(v):
        e = v + 0.3 * rng.normal(size=(8, 128))
        return e / np.linalg.norm(e, axis=1, keepdims=True)

    a = _piece(1, "A", 0, 50, 5, 5)
    b = _piece(2, "A", 60, 100, 5.5, 5.5)  # 1 s later, 0.5 m away: confident by position
    a.emb, b.emb = emb(me), emb(mate)
    assert not confident(a, b, solver, LineupConfig())  # but re-ID says another person
    b.emb = emb(me)
    assert confident(a, b, solver, LineupConfig())
    c = _piece(3, "A", 200, 260, 9, 9)  # 15 s later: only re-ID makes it confident
    c.emb = emb(me)
    assert confident(a, c, solver, LineupConfig())
    c.emb = None
    assert not confident(a, c, solver, LineupConfig())


def test_profiles_spread_numbers_along_a_slot_until_a_dead_ball():
    lineup.SLOT_LABEL.clear(); lineup.CHAIN_SPAN.clear(); lineup.DEAD.clear(); lineup.OUT.clear()
    lineup.SLOT_LABEL.update({101: "A1", 102: "A1", 103: "A1", 104: "A2"})
    lineup.CHAIN_SPAN.update({101: (0, 10), 102: (12, 30), 103: (50, 60), 104: (0, 60)})
    lineup.DEAD.append((35.0, 45.0))  # substitution window between 102 and 103
    info = {1: {"team": "A", "number": "23"}, 2: {"team": "A"}, 3: {"team": "A"}, 4: {"team": "A", "number": "7"}}
    lineup.add_track_info(info, {101: 1, 102: 2, 103: 3, 104: 4})
    lineup.SLOT_LABEL.clear(); lineup.CHAIN_SPAN.clear(); lineup.DEAD.clear()
    assert info[1]["profile"] == info[2]["profile"] == "A#23" and info[2]["number_source"] == "lineup"
    assert "profile" not in info[3] and "number" not in info[3]  # after the dead ball: maybe a substitute
    assert info[4]["profile"] == "A#7" and info[3].get("slot") == "A1"


def test_exact5_drops_extras_only_where_all_slots_are_busy(monkeypatch):
    monkeypatch.setenv("REPLAY_LINEUP", "1")
    monkeypatch.setenv("REPLAY_LINEUP_EXACT5", "1")
    monkeypatch.delenv("REPLAY_LINEUP_CFG", raising=False)
    t = np.arange(100) / 10.0
    tid, det_i, xy = [], [], []
    for g in range(1, 6):  # 5 A players all the time
        for i in range(100):
            tid.append(g); det_i.append(i); xy.append((2 + 3 * g, 2.0 + 0.3 * (i % 30)))  # running
    for i in range(20, 40):  # a 6th A person (bench kid) for 2 s
        tid.append(6); det_i.append(i); xy.append((14.0, 1.0))
    tid, det_i, xy = np.array(tid), np.array(det_i), np.array(xy)
    teams = TeamResult({g: "A" for g in range(1, 7)}, {g: 0.9 for g in range(1, 6)} | {6: 0.3}, {}, False, {})
    _, keep = lineup.apply(None, tid, np.ones(len(tid), bool), det_i, xy, np.zeros((len(tid), 6)), None, t, teams,
                           LineupConfig(enabled=True))
    assert keep[tid <= 5].all() and not keep[tid == 6].any()


def test_exact5_mode2_evicts_least_player_like_even_from_a_slot(monkeypatch):
    monkeypatch.setenv("REPLAY_LINEUP", "1")
    monkeypatch.setenv("REPLAY_LINEUP_EXACT5", "2")
    monkeypatch.delenv("REPLAY_LINEUP_CFG", raising=False)
    t = np.arange(100) / 10.0
    tid, det_i, xy, rank = [], [], [], []
    for g in range(1, 7):  # 6 'A' identities all the time; id 6 is the bench kid (low rank) but long
        for i in range(100):
            tid.append(g); det_i.append(i); xy.append((2 + 3 * g, 2.0 + 0.3 * (i % 30))); rank.append(0.5 if g == 6 else 1.0)
    tid, det_i, xy, rank = np.array(tid), np.array(det_i), np.array(xy), np.array(rank)
    teams = TeamResult({g: "A" for g in range(1, 7)}, {g: 0.9 for g in range(1, 7)}, {}, False, {})
    _, keep = lineup.apply(None, tid, np.ones(len(tid), bool), det_i, xy, np.zeros((len(tid), 6)), None, t, teams,
                           LineupConfig(enabled=True), rank=rank)
    assert keep[tid <= 5].all() and not keep[tid == 6].any()


def test_exact5_mode3_keeps_the_6th_person_but_marks_it_unsure(monkeypatch):
    from types import SimpleNamespace

    from replay.export import TrackRow, source_of
    monkeypatch.setenv("REPLAY_LINEUP", "1")
    monkeypatch.setenv("REPLAY_LINEUP_EXACT5", "3")
    monkeypatch.delenv("REPLAY_LINEUP_CFG", raising=False)
    t = np.arange(100) / 10.0
    tid, det_i, xy, rank = [], [], [], []
    for g in range(1, 7):
        for i in range(100):
            tid.append(g); det_i.append(i); xy.append((2 + 3 * g, 2.0 + 0.3 * (i % 30))); rank.append(0.5 if g == 6 else 1.0)
    tid, det_i, xy, rank = np.array(tid), np.array(det_i), np.array(xy), np.array(rank)
    teams = TeamResult({g: "A" for g in range(1, 7)}, {g: 0.9 for g in range(1, 7)}, {}, False, {})
    new_tid, keep = lineup.apply(None, tid, np.ones(len(tid), bool), det_i, xy, np.zeros((len(tid), 6)), None, t,
                                 teams, LineupConfig(enabled=True), rank=rank)
    assert keep.all()
    g6 = int(new_tid[tid == 6][0])
    rows = [TrackRow(frame=3 * i, t=t[i], k=3 * i, track_id=g6 + 100, object_type="player", team="A", x=20.0, y=5.0,
                     confidence=0.9, interpolated=False, calib="keyframe") for i in range(100)]
    data = SimpleNamespace(t=t, k=np.arange(100) * 3, frame=np.arange(100) * 3)
    lineup.predicted_rows(rows, {g6: g6 + 100}, data, [np.eye(3)] * 100, det_i, xy, new_tid, keep)
    assert all(r.unsure for r in rows) and source_of(rows[0]) == "unsure"
    lineup.SLOT_LABEL.clear(); lineup.UNSURE_RANGES.clear()


def test_count_volatility_drawn_ignores_unsure_rows():
    from replay.benchmark import continuity_metrics
    rows = [{"frame": k, "timestamp": k / 10, "track_id": i, "object_type": "player", "team": "A",
             "source": "unsure" if i >= 5 else "seen"} for k in range(50) for i in range(7) if i < 5 or k % 2]
    m = continuity_metrics(rows, fps=10.0)
    assert m["count_volatility"] > 0.5 and m["count_volatility_drawn"] == 0.0


def test_predicted_rows_never_move_faster_than_a_sprint(monkeypatch):
    """Debug run 19.10 (hs_hazen seg_2700): the motion prior pulled an off-court dot back onto the court at 11+ m/s."""
    from types import SimpleNamespace

    from replay import court_tracker
    from replay.export import TrackRow

    class FarPrior:  # predicts a point 10 m away immediately
        def predict(self, a, xy0, v0, k):
            return np.asarray(xy0, float) + np.array([10.0, 0.0]), 0.5

    monkeypatch.setattr(court_tracker.MotionPrior, "from_detections", staticmethod(lambda *a, **k: FarPrior()))
    n = 60
    t = np.arange(n) / 10.0
    data = SimpleNamespace(t=t, k=np.arange(n) * 3, frame=np.arange(n) * 3)
    lineup.SLOT_LABEL.clear()
    lineup.SLOT_LABEL[50] = "A1"
    rows = [TrackRow(frame=3 * i, t=t[i], k=3 * i, track_id=1, object_type="player", team="A", x=-3.0,
                     y=5.0, confidence=0.9, interpolated=False, calib="keyframe") for i in range(10)]
    det_i = np.arange(10)
    pred = lineup.predicted_rows(rows, {50: 1}, data, [np.eye(3)] * n, det_i, np.c_[np.full(10, -3.0), np.full(10, 5.0)],
                                 np.full(10, 1), np.ones(10, bool))
    lineup.SLOT_LABEL.clear()
    assert pred
    seq = [(rows[-1].t, rows[-1].x, rows[-1].y)] + [(r.t, r.x, r.y) for r in sorted(pred, key=lambda r: r.t)]
    for (ta, xa, ya), (tb, xb, yb) in zip(seq, seq[1:]):
        assert np.hypot(xb - xa, yb - ya) <= lineup.PRED_MAX_SPEED_MPS * (tb - ta) + 1e-6
