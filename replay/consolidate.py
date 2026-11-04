"""Final per-frame reality check of the people shown on the court (no phantom players).

After tracking, team labels and gap filling, a frame can still contain things that cannot be real:
the same player twice (an old track filled across a gap next to the new track that re-found them),
two same-team dots on top of each other, more than five players of one team, people of no team, or people
far outside the court. Basketball gives hard limits, so every frame is checked against them:

  * at most 5 players per team and 3 referees on the court;
  * two SEEN dots are one person only if their boxes in the video overlap heavily (teammates do stand close);
  * a filled-in (interpolated) dot near a really seen dot of the same team is a ghost of that player;
  * people without a team are shown only if they are clearly on the court and really seen;
  * nobody more than 1.5 m outside the court lines.

Seen rows always win over filled-in ones; among equals the stronger (longer, more player-like) track wins.
"""

from __future__ import annotations

import numpy as np

from replay import court

SAME_PERSON_M = 0.8  # two same-team dots closer than this are one person
GHOST_M = 1.5  # a filled-in dot this close to a seen same-team dot is that player's ghost
OFF_COURT_M = 1.5  # nobody is drawn further outside the lines
PARTIAL_INSIDE = 0.85  # a box this much inside a taller box ...
PARTIAL_TOP = 0.08  # ... with its top this close (share of the taller box) is a partial box of that person
SHIFT_MIN_SEEN = 4  # frames with at least this many seen team/referee rows ...
SHIFT_SHARE = 0.5  # ... of which this share is far off the court: calibration shifted, rule 1 skipped
SHIFT_MIN_RANK = 1.5  # roster rank (player score) an off-court seen row needs to stay in a shifted frame
PER_TEAM = 5
REFEREES = 3


def _iou(a, b) -> float:
    if a is None or b is None:
        return 0.0
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def partial_rows(rows) -> set[int]:
    """id() of the seen rows whose box lies (>= PARTIAL_INSIDE of its area) inside another seen row's taller box
    with the same top: most likely a second, partial box of that person (legs hidden). Rows of one frame."""
    seen = [r for r in rows if not r.interpolated and getattr(r, "box", None) is not None
            and getattr(r, "object_type", "") != "ball"]
    out: set[int] = set()
    for r in seen:
        a = r.box
        area = max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))
        for q in seen:
            b = q.box
            if q is r or (b[3] - b[1]) <= (a[3] - a[1]):
                continue
            inter = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
            if inter / area >= PARTIAL_INSIDE and abs(a[1] - b[1]) <= PARTIAL_TOP * (b[3] - b[1]):
                out.add(id(r))
                break
    return out


def consolidate_drop(rows, strength: dict, team_a: str = "A", team_b: str = "B", ref: str = "referee",
                     rank: dict | None = None, reasons: dict | None = None) -> set[int]:
    """ids (``id(row)``) of rows to remove. ``rows``: TrackRow-like objects (k, track_id, team, x, y,
    interpolated, object_type, box); ``strength``: track_id -> how player-like the track is; ``rank``: id(row) ->
    how player-like this detection is right now (preferred over strength for seen rows). ``reasons`` (optional
    dict): filled with id(row) -> why it was dropped ("off court", "no team", "duplicate", "ghost", "cap")."""
    rank = rank or {}
    why = reasons if reasons is not None else {}
    by_k: dict[int, list] = {}
    for r in rows:
        if r.object_type == "ball":
            continue
        by_k.setdefault(r.k, []).append(r)
    drop: set[int] = set()
    L, W = court.LENGTH, court.WIDTH
    for rs in by_k.values():
        # 1. far off the court - unless most of the seen people are "off court": then the calibration of this frame
        # is shifted (seen on other gyms: everyone 1-3 m behind the baseline) and the rule would delete the players
        def _out(r):
            return not (-OFF_COURT_M <= r.x <= L + OFF_COURT_M and -OFF_COURT_M <= r.y <= W + OFF_COURT_M)
        seen = [r for r in rs if not r.interpolated and r.team in (team_a, team_b, ref)]
        shifted = len(seen) >= SHIFT_MIN_SEEN and sum(map(_out, seen)) >= SHIFT_SHARE * len(seen)
        keep = []
        for r in rs:
            # in a shifted frame only clearly player-like seen people stay (the bench along the lines does not)
            if not _out(r) or (shifted and not r.interpolated and float(rank.get(id(r), 0.0)) >= SHIFT_MIN_RANK):
                keep.append(r)
            else:
                drop.add(id(r))
                why[id(r)] = "off court"
        # 2. people of no team: only if really seen and clearly on the court
        rest = []
        for r in keep:
            if r.team not in (team_a, team_b, ref) and (r.interpolated or not (0 <= r.x <= L and 0 <= r.y <= W)):
                drop.add(id(r))
                why[id(r)] = "no team"
            else:
                rest.append(r)
        # 3. per group: seen first, then stronger; dedupe and cap
        groups: dict[str, list] = {}
        for r in rest:
            groups.setdefault(r.team, []).append(r)
        for team, members in groups.items():
            members.sort(key=lambda r: (bool(r.interpolated), -float(rank.get(id(r), 0.0)),
                                        -float(strength.get(r.track_id, 0.0))))
            kept: list = []
            for r in members:
                p = np.array([r.x, r.y])
                dup = False
                for q in kept:
                    d = float(np.hypot(q.x - p[0], q.y - p[1]))
                    both_seen = not r.interpolated and not q.interpolated
                    same = (_iou(getattr(r, "box", None), getattr(q, "box", None)) > 0.5 if both_seen
                            else d < SAME_PERSON_M)
                    if same or (r.interpolated and not q.interpolated and d < GHOST_M):
                        dup = "duplicate" if same else "ghost"
                        break
                if dup:
                    drop.add(id(r))
                    why[id(r)] = dup
                else:
                    kept.append(r)
            cap = REFEREES if team == ref else PER_TEAM if team in (team_a, team_b) else 0
            if team not in (team_a, team_b, ref):
                # unknown-team people only fill free player places
                free = 2 * PER_TEAM - sum(len([x for x in groups.get(t, []) if id(x) not in drop])
                                          for t in (team_a, team_b))
                cap = max(0, min(len(kept), free))
            if len(kept) > cap:  # over the limit: a partial second box of someone gives way first
                part = partial_rows(rs)
                kept.sort(key=lambda r: (bool(r.interpolated), id(r) in part, -float(rank.get(id(r), 0.0)),
                                         -float(strength.get(r.track_id, 0.0))))
            for r in kept[cap:]:
                drop.add(id(r))
                why[id(r)] = "cap"
    return drop
