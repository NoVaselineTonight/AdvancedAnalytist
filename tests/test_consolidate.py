from types import SimpleNamespace as R

from replay import court
from replay.consolidate import consolidate_drop


def row(tid, team, x, y, interp=False, k=0, typ="player", box=None):
    return R(k=k, track_id=tid, team=team, x=x, y=y, interpolated=interp, object_type=typ, box=box)


def test_duplicates_ghosts_caps_and_offcourt():
    court.configure(28.0, 15.0, "FIBA")
    rows = [row(1, "A", 5, 5, box=(100, 100, 140, 220)), row(2, "A", 5.3, 5.2, box=(102, 98, 141, 221)),  # same box
            row(7, "A", 20, 5, box=(400, 100, 440, 220)), row(8, "A", 20.4, 5.1, box=(445, 100, 485, 220)),  # teammates
            row(3, "A", 10, 5, interp=True), row(4, "A", 11, 5),  # filled-in ghost 1 m from a seen dot
            row(5, "B", 40, 5),  # far off court
            row(6, "unknown", 14, 7, interp=True),  # filled-in person of no team
            row(9, "", 14, 7, typ="ball")]
    rows += [row(10 + j, "B", 2 + 3 * j, 10) for j in range(7)]  # 7 purple players: only 5 can be on court
    rows = [r for r in rows if not (r.team == "A" and r.track_id in (3, 4))] + [row(3, "A", 10, 5, interp=True),
                                                                               row(4, "A", 11, 5)]
    strength = {r.track_id: 100 - (r.track_id if isinstance(r.track_id, int) else 0) for r in rows}
    drop = consolidate_drop(rows, strength)
    gone = {r.track_id for r in rows if id(r) in drop}
    assert 2 in gone and 1 not in gone  # weaker duplicate removed
    assert 7 not in gone and 8 not in gone  # two teammates standing close are both real
    assert 3 in gone and 4 not in gone  # ghost removed, seen player kept
    assert 5 in gone and 6 in gone
    assert 9 not in gone  # the ball is never touched
    assert sum(1 for r in rows if r.team == "B" and id(r) not in drop) == 5


def test_seen_beats_filled_in_for_the_cap():
    court.configure(28.0, 15.0, "FIBA")
    rows = [row(j, "A", 2 + 3 * j, 5, interp=(j == 0)) for j in range(6)]
    drop = consolidate_drop(rows, {j: (1000 if j == 0 else 1) for j in range(6)})
    assert {r.track_id for r in rows if id(r) in drop} == {0}



def test_partial_box_gives_way_at_the_cap():
    court.configure(28.0, 15.0, "FIBA")
    # 6 seen team-A rows; one is the upper part of a (team-B) player's box: it goes, not the weakest real player
    rows = [row(j, "A", 2 + 3 * j, 5, box=(100 * j, 100, 100 * j + 40, 220)) for j in range(5)]
    rows.append(row(9, "A", 20, 9, box=(902, 101, 938, 180)))  # inside 900-940 x 100-230, same top
    rows.append(row(10, "B", 20, 9.5, box=(900, 100, 940, 230)))
    rank = {id(r): 1.0 for r in rows}
    rank[id(rows[5])] = 9.0  # even when it ranks highest
    drop = consolidate_drop(rows, {r.track_id: 1 for r in rows}, rank=rank)
    assert {r.track_id for r in rows if id(r) in drop} == {9}


def test_shifted_calibration_keeps_player_like_people_off_court():
    court.configure(28.0, 15.0, "FIBA")
    # most seen people 2-3 m "behind the baseline": the calibration is shifted, the moving players stay;
    # a low-score (bench-like) person out there still goes
    rows = [row(j, "A" if j % 2 else "B", 3 + 4 * j, -2.5, box=(50 * j, 100, 50 * j + 30, 200)) for j in range(5)]
    rows.append(row(9, "B", 20, -2.5, box=(600, 100, 630, 200)))
    rank = {id(r): 2.5 for r in rows}
    rank[id(rows[-1])] = 0.3
    drop = consolidate_drop(rows, {r.track_id: 1 for r in rows}, rank=rank)
    assert {r.track_id for r in rows if id(r) in drop} == {9}
    # normal frame: the same far-off person alone is removed
    rows2 = [row(j, "A", 3 + 4 * j, 5, box=(50 * j, 100, 50 * j + 30, 200)) for j in range(4)] + [row(9, "B", 20, -2.5)]
    drop2 = consolidate_drop(rows2, {r.track_id: 1 for r in rows2}, rank={id(r): 2.5 for r in rows2})
    assert {r.track_id for r in rows2 if id(r) in drop2} == {9}
