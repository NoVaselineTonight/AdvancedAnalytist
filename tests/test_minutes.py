"""Game-clock minutes (replay/minutes.py) and their use in stats.lineup_stats (9.08)."""

import pytest

from replay import minutes as M
from replay import stats as S


def _samples(running):
    """Score-bar samples every 2 s; the clock (mm:ss, counting down from 10:00) runs only inside ``running``."""
    out, clock = [], 600.0
    t = 0.0
    while t <= 200.0:
        out.append([t, 0, 0, f"{int(clock) // 60:02d}:{int(clock) % 60:02d}", "1"])
        if any(a <= t < b for a, b in running):
            clock -= 2.0
        t += 2.0
    return out


def test_clock_source_counts_only_running_clock():
    sb = {"samples": _samples([(0, 60), (100, 160)])}
    gc = M.GameClock({"time_range_s": [0, 200]}, sb, {"period_scores": {"A": [1, 2, 3, 4], "B": [1, 2, 3, 4]}})
    assert gc.source == "clock" and gc.game_s == 2400.0
    assert gc.seconds(0, 200) == pytest.approx(120.0, abs=4.0)
    assert gc.dead_between(70, 90) and not gc.dead_between(10, 50)


def test_live_fallback_is_scaled_to_the_game_length():
    gs = {"time_range_s": [0, 6000], "breaks": [{"t0": 1000, "t1": 1200}], "dead_balls": [{"t0": 50, "t1": 80}]}
    gc = M.GameClock(gs, None, {"period_scores": {"A": [1, 2, 3, 4]}})
    assert gc.source == "live" and gc.total_s() == pytest.approx(2400.0)
    assert gc.warnings and "scaled" in gc.warnings[0]


def test_sanity_flags_impossible_minutes():
    gc = M.GameClock({"time_range_s": [0, 2400]}, None, {"period_scores": {"A": [1, 2, 3, 4]}})
    w = M.sanity({("B", "#24"): 4700.0, ("B", "#20"): 4600.0, ("B", "#5"): 4000.0}, gc)
    assert any("#24" in x and "longer than the game" in x for x in w)
    assert any("Team B" in x for x in w)


def test_slot_label_from_reads_between_dead_balls():
    # one tracker piece runs through a dead ball (t 290-310); reads say #8 before and #9 after it
    lineup = {"slots": {"A1": {"pieces": [[1, 0, 600, 8]]}},
              "reads": {"A1": [[50.0, 8], [100.0, 8], [400.0, 9], [500.0, 9]]}}
    gs = {"time_range_s": [0, 600], "breaks": [{"t0": 290.0, "t1": 310.0}]}
    lu = S.lineup_stats(lineup, gs, [], [])
    secs = {tuple(r["players"]): r["seconds"] for r in lu["rows"]}
    assert secs[("#8",)] == pytest.approx(290.0) and secs[("#9",)] == pytest.approx(290.0)


def test_numbers_not_on_the_scoresheet_are_ignored():
    lineup = {"slots": {"A1": {"pieces": [[1, 0, 600, 8]]}},
              "reads": {"A1": [[50.0, 24], [60.0, 24], [100.0, 8]]}}  # #24 is not on team A's sheet
    gs = {"time_range_s": [0, 600]}
    sheet = {"teams": {"A": {"players": [{"number": "8", "played": True}, {"number": "11"}]}}}
    lu = S.lineup_stats(lineup, gs, [], [], clock=M.GameClock(gs, None, sheet))
    assert [r["players"] for r in lu["rows"]] == [["#8"]]
