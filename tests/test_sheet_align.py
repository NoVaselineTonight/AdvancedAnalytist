"""Scoresheet reconciliation of detected baskets (replay/sheet_align.py)."""

from __future__ import annotations

from replay import sheet_align as sa


def _sheet():
    ev = []
    seq = [("B", 2, "28", "fg2", 1), ("A", 1, "6", "ft", 1), ("A", 1, "6", "ft", 1), ("B", 2, "10", "fg2", 1),
           ("B", 2, "21", "fg2", 2), ("A", 3, "6", "fg3", 2), ("B", 2, "28", "fg2", 2)]
    for k, (team, pts, num, kind, per) in enumerate(seq, 1):
        ev.append({"seq": k, "team": team, "points": pts, "scorer": num, "kind": kind, "period": per})
    return ev


def test_segments_from_breaks():
    segs = sa.segments([{"kind": "quarter_break", "t0": 100, "t1": 150}, {"kind": "half_time", "t0": 300, "t1": 400}],
                       700)
    assert [s[2] for s in segs] == [{1}, {2}, {3, 4}]
    assert sa.segments([], 100) == [(0.0, 100, {1, 2, 3, 4})]


def test_alignment_maps_detections_and_drops_false_ones():
    sheet = _sheet()
    segs = sa.segments([{"kind": "quarter_break", "t0": 100, "t1": 150}], 300)
    dets = [{"t": 20, "team": "B", "kind": "fg"},   # sheet 0
            {"t": 40, "team": "A", "kind": "ft"},   # sheet 1 (2nd FT missed)
            {"t": 90, "team": "B", "kind": "fg"},   # sheet 3
            {"t": 95, "team": "A", "kind": "fg"},   # false basket (nothing left on the sheet in period 1)
            {"t": 200, "team": "A", "kind": "fg"},  # sheet 5 (period 2)
            {"t": 260, "team": "B", "kind": "fg"}]  # sheet 6
    al = sa.align(sheet, dets, segs)
    d2s = al["det_to_sheet"]
    assert d2s[0] == 0 and d2s[1] in (1, 2) and d2s[2] == 3 and d2s[3] is None and d2s[4] == 5 and d2s[5] == 6
    assert al["sheet_to_det"][4] is None  # period-2 basket nobody detected
    est = sa.estimate_times(len(sheet), al["sheet_to_det"], [d["t"] for d in dets], [s["period"] for s in sheet],
                            segs)
    assert 150 - 30 <= est[4] <= 200


def test_period_constraint():
    sheet = _sheet()
    segs = sa.segments([{"kind": "quarter_break", "t0": 100, "t1": 150}], 300)
    al = sa.align(sheet, [{"t": 250, "team": "B", "kind": "fg"}], segs)
    assert al["det_to_sheet"][0] in (4, 6)  # a period-2 detection can only be a period-2 sheet event


def test_check_makes_flags_make_without_play_moving_away():
    from replay import ball_events as be
    n = 300
    t = [i / 10 for i in range(n)]

    def tracks(move):  # 5 players near the right basket; after t=10 s they run to the left end (or stay)
        out = []
        for k in range(5):
            xs = [int((25 - (min(20.0, max(0.0, (i / 10 - 10) * 3)) if move else 0) - k * 0.3) * 100) for i in range(n)]
            out.append({"id": k, "type": "player", "team": "A", "i0": 0, "x": xs, "y": [700] * n, "f": [1] * n})
        return out
    for move, flagged in ((True, False), (False, True)):
        rd = {"t": t, "court": {"length": 28.0}, "tracks": tracks(move),
              "events": [{"id": "b1", "type": "basket", "t": 10.0, "basket": "R"}]}
        gs = {"basket_annotations": {}, "breaks": []}
        assert (be.check_makes(gs, rd) == ["b1"]) is flagged


def test_scorebar_reconciliation():
    from replay import ball_events as be
    sb = {"timeline": [[0, 0, 0], [20, 0, 2], [40, 1, 2], [60, 1, 4], [90, 1, 7]]}
    ev = be.bar_events(sb)
    assert [(e["side"], e["points"]) for e in ev] == [(1, 2), (0, 1), (1, 2), (1, 3)]
    rd = {"events": [{"id": "b1", "type": "basket", "t": 16.0, "team": "A", "basket": "R"},
                     {"id": "b2", "type": "basket", "t": 36.0, "team": "B", "basket": "L"},
                     {"id": "b3", "type": "basket", "t": 65.0, "team": "B", "basket": "L"},   # no bar change after it
                     {"id": "b4", "type": "basket", "t": 85.0, "team": "A", "basket": "R"}]}
    gs = {"basket_annotations": {}, "shots": []}
    doc = be.reconcile_with_scorebar(gs, rd, sb)
    ann = gs["basket_annotations"]
    assert doc["side_team"] == {0: "B", 1: "A"}
    assert ann["b1"]["points_guess"] == 2 and ann["b4"]["points_guess"] == 3 and ann["b2"]["kind"] == "free_throw"
    assert ann["b3"]["kind"] == "not_on_sheet" and len(doc["missing"]) == 1 and doc["missing"][0]["team"] == "A"
    # a bar that explains almost nothing is not used
    gs2 = {"basket_annotations": {}, "shots": []}
    assert be.reconcile_with_scorebar(gs2, rd, {"timeline": [[0, 0, 0], [500, 0, 1]]}) is None
    assert gs2["scorebar_alignment"]["used"] is False
