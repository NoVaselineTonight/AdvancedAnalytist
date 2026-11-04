"""Playing time on the GAME CLOCK (minutes per player / lineup) and the dead balls where substitutions happen.

Video time is not game time: the clock stops for fouls, free throws, time-outs, out of bounds and between periods,
so minutes counted in video time come out about twice too large (user's full game: team totals ~383 min in a
4 x 10 min game). :class:`GameClock` turns a video interval into game seconds:

* **clock** (preferred): the broadcast score-bar clock (replay/scorebar.py, ``work/scorebar.json`` samples
  ``[t, s0, s1, clock, period]``). Between two readings the clock RAN when it went down by about the elapsed video
  time (0.5x .. 1.5x, misreads tolerated); those video intervals are game time, everything else is stopped clock.
* **live** (fallback, no readable clock): video time minus breaks, dead balls (game_state ``dead_balls``) and
  free-throw windows; then scaled down so the total is not longer than the game (clock stops the live estimate
  cannot see: out of bounds, violations).

The game length comes from the scoresheet (number of periods) x ``period_s`` (FIBA: 10 min), or from the clock
readings' periods. :func:`sanity` checks player minutes <= game length and team total = 5 x game length.
Dead balls (``dead_windows``) = clock-stopped spans >= ``DEAD_MIN_S`` (clock source) or game_state's dead balls,
breaks and free throws - the moments a slot may change its player (lineup labels are re-checked there).
"""

from __future__ import annotations

from typing import Any

PERIOD_S = 600.0  # FIBA period length; youth / other leagues: pass period_s
DEAD_MIN_S = 3.0  # a stopped clock this long = a dead ball (substitutions possible)
MIN_RUN_READS = 3  # clock readings needed before the clock source is trusted


def _clock_s(v: Any) -> float | None:
    if v is None or v == "":
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    try:
        if ":" in s:
            m, sec = s.split(":", 1)
            return int(m) * 60 + float(sec)
        return float(s)
    except ValueError:
        return None


def _merge(iv: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[list[float]] = []
    for a, b in sorted(iv):
        if b <= a:
            continue
        if out and a <= out[-1][1] + 1e-6:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def _overlap(a: float, b: float, iv: list[tuple[float, float]]) -> float:
    return sum(max(0.0, min(b, y) - max(a, x)) for x, y in iv)


def clock_running(samples: list) -> list[tuple[float, float]]:
    """Video intervals where the score-bar clock ran (see module docstring)."""
    pts = [(float(x[0]), c) for x in samples or [] if len(x) > 3 and (c := _clock_s(x[3])) is not None]
    run = []
    for (t0, c0), (t1, c1) in zip(pts, pts[1:]):
        dt, dc = t1 - t0, c0 - c1
        if 0 < dt <= 30.0 and 0.5 * dt <= dc <= 1.5 * dt + 0.6:
            run.append((t0, t1))
    return _merge(run)


def game_length_s(sheet: dict | None, samples: list | None = None, period_s: float = PERIOD_S) -> float | None:
    """Regulation game length: scoresheet periods (incl. overtimes of 5 min) x period_s; else clock periods."""
    if isinstance(sheet, dict):
        ps = sheet.get("period_scores") or (sheet.get("game") or {}).get("period_scores_page1") or {}
        n = max((len(v) for v in ps.values() if isinstance(v, list)), default=0) if isinstance(ps, dict) else 0
        if n:
            return min(n, 4) * period_s + max(0, n - 4) * 300.0
    pers = {str(x[4]) for x in samples or [] if len(x) > 4 and x[4]}
    if pers:
        return max(len(pers), 1) * period_s
    return None


class GameClock:
    """Video seconds -> game seconds, dead-ball windows and the source used ("clock" | "live" | "video")."""

    def __init__(self, gs: dict | None = None, scorebar: dict | None = None, sheet: dict | None = None,
                 period_s: float = PERIOD_S):
        gs = gs or {}
        samples = (scorebar or {}).get("samples") or []
        self.game_s = game_length_s(sheet, samples, period_s)
        self.warnings: list[str] = []
        self.scale = 1.0
        # scoresheet rosters (team -> jersey numbers that played): reads of other numbers are misreads
        self.rosters: dict[str, set[int]] = {}
        for k, tm in ((sheet or {}).get("teams") or {}).items() if isinstance((sheet or {}).get("teams"), dict) else []:
            nums = set()
            for pl in (tm or {}).get("players") or []:
                try:
                    if pl.get("played", True) is not False:
                        nums.add(int(str(pl.get("number")).strip()))
                except (TypeError, ValueError):
                    pass
            if nums:
                self.rosters[str(k)] = nums
        run = clock_running(samples)
        n_reads = sum(1 for x in samples if len(x) > 3 and _clock_s(x[3]) is not None)
        tr = gs.get("time_range_s") or [0.0, 0.0]
        t0, t1 = float(tr[0] or 0.0), float(tr[1] or 0.0)
        if n_reads >= MIN_RUN_READS and run:
            self.source = "clock"
            self.running = run
            gaps = [(a[1], b[0]) for a, b in zip(run, run[1:])]
            self.dead = [(a, b) for a, b in gaps if b - a >= DEAD_MIN_S]
        else:
            stops = [(float(b["t0"]), float(b["t1"])) for key in ("breaks", "dead_balls")
                     for b in gs.get(key) or [] if b.get("t0") is not None and b.get("t1") is not None]
            stops += [(float(w["t0"]), float(w["t1"])) for w in gs.get("free_throw_windows") or []
                      if w.get("t0") is not None and w.get("t1") is not None]
            stops = _merge(stops)
            self.dead = stops
            live, cur = [], t0
            for a, b in stops:
                if a > cur:
                    live.append((cur, min(a, t1)))
                cur = max(cur, b)
            if cur < t1:
                live.append((cur, t1))
            self.running = _merge(live)
            self.source = "live" if stops else "video"
            total = sum(b - a for a, b in self.running)
            if self.game_s and total > self.game_s:
                self.scale = self.game_s / total  # clock stops the estimate cannot see (out of bounds, ...)
                self.warnings.append(f"No readable game clock: minutes estimated from live play and scaled by "
                                     f"{self.scale:.2f} to the {self.game_s / 60:.0f}-min game length.")
            elif not self.game_s:
                self.warnings.append("No readable game clock and no game length: minutes are live-play video time.")

    def seconds(self, a: float, b: float) -> float:
        """Game seconds between video times a and b."""
        return _overlap(a, b, self.running) * self.scale

    def total_s(self) -> float:
        return sum(b - a for a, b in self.running) * self.scale

    def dead_between(self, a: float, b: float) -> bool:
        lo, hi = min(a, b), max(a, b)
        return any(x < hi and y > lo for x, y in self.dead)

    def info(self) -> dict:
        return {"source": self.source, "game_length_s": self.game_s, "played_s": round(self.total_s(), 1),
                "scale": round(self.scale, 3), "dead_balls": len(self.dead)}


def sanity(player_s: dict[tuple[str, str], float], clock: GameClock, per_team: int = 5) -> list[str]:
    """Warnings for impossible minutes: a player longer than the game, a team total far from 5 x game length."""
    out = []
    g = clock.game_s or clock.total_s()
    if not g:
        return out
    for (team, label), s in sorted(player_s.items()):
        if s > g * 1.02:
            out.append(f"{team} {label}: {s / 60:.1f} min is longer than the game ({g / 60:.0f} min) - "
                       "probably two players read as one number.")
    for team in sorted({t for t, _ in player_s}):
        tot = sum(s for (t, _), s in player_s.items() if t == team)
        if tot > per_team * g * 1.05:
            out.append(f"Team {team}: {tot / 60:.0f} player-minutes > {per_team} x {g / 60:.0f} min - minutes "
                       "are overcounted (duplicate numbers in two slots).")
    return out
