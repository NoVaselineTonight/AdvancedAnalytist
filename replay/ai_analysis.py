"""AI tactical analysis of an analysed Replay with Google's Gemini API (the user's own AI Studio key).

Flow: :func:`build_game_digest` turns the 2D tracking data into a compact, documented text (downsampled to a few
frames per second, rounded coordinates, one line per sample) -> :func:`build_prompts` adds the instructions (how to
read the data, which team the user coaches, the competition level to calibrate against, the strict JSON answer
format) -> :class:`GeminiClient` calls ``models.generateContent`` in JSON mode (google-genai SDK when installed,
otherwise the documented REST endpoint) -> :func:`parse_response` validates the answer -> long games are analysed
in time chunks and merged (:func:`run_analysis`) -> :func:`save_results` writes ``results/ai_analysis.json`` and
:func:`apply_to_timeline` puts every issue on the "AI analysis" timeline track.

The API key is only ever passed in memory to the client. It is never written to the Replay folder, to results, to
packages or exports, and it is scrubbed from every error message. No Qt here (unit-testable without a GUI).
"""

from __future__ import annotations

import json
import math
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

from replay import __version__, ai_light, court
from replay import data_export as dx
from replay import game_state as gsm

FORMAT = "advanced-analytist/ai-analysis@1"
RESULTS_FILE = "ai_analysis.json"  # in <project>/results/

# Current Gemini models per https://ai.google.dev/gemini-api/docs/models (checked 2026-10). The field in the dialog is
# editable, so a newer model can be typed in without a code change.
DEFAULT_MODEL = "gemini-3.8-flash"
MODELS = ["gemini-3.8-flash", "gemini-3.5-flash-lite", "gemini-3.1-pro-preview"]
API_BASE = "https://generativelanguage.googleapis.com/v1beta"
API_KEY_URL = "https://aistudio.google.com/apikey"

DEFAULT_SAMPLE_HZ = 2.0
OFFSCREEN_EVERY_S = 5.0  # sample lines list the off-screen on-court players this often
DEFAULT_CHUNK_S = 600.0  # 10 min of video per request (~50k tokens of data at 2 Hz)
REQUEST_TIMEOUT_S = 300.0
TIME_TOLERANCE_S = 5.0  # issues this far outside the analysed range are clamped, further ones dropped

TEAM_PHASES = ("offense", "defense", "transition")
PHASE_ALIASES = {"offence": "offense", "attack": "offense", "attacking": "offense", "hyökkäys": "offense",
                 "defence": "defense", "defending": "defense", "puolustus": "defense",
                 "transition offense": "transition", "transition defense": "transition", "fast break": "transition",
                 "siirtymä": "transition", "vaihto": "transition"}
PHASE_COLORS = {"offense": "#f97316", "defense": "#0ea5e9", "transition": "#10b981"}

AI_TRACK_ID = "ai-analysis"
AI_TRACK_NAME = "AI analysis"
AI_TRACK_COLOR = "#ec4899"

# Response schema (OpenAPI subset accepted by generateContent's responseSchema).
_STR_LIST = {"type": "ARRAY", "items": {"type": "STRING"}}
OVERVIEW_SCHEMA = {
    "type": "OBJECT",
    "properties": {"went_well": _STR_LIST, "to_improve": _STR_LIST, "summary": {"type": "STRING"}},
    "required": ["went_well", "to_improve", "summary"],
}
RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "overview": OVERVIEW_SCHEMA,
        "issues": {"type": "ARRAY", "items": {
            "type": "OBJECT",
            "properties": {
                "t_start": {"type": "NUMBER", "description": "video seconds"},
                "t_end": {"type": "NUMBER", "description": "video seconds"},
                "title": {"type": "STRING"},
                "description": {"type": "STRING"},
                "category": {"type": "STRING"},
                "team_phase": {"type": "STRING", "enum": list(TEAM_PHASES)},
            },
            "required": ["t_start", "t_end", "title", "description", "category", "team_phase"],
        }},
    },
    "required": ["overview", "issues"],
}
MERGE_SCHEMA = {"type": "OBJECT", "properties": {"overview": OVERVIEW_SCHEMA}, "required": ["overview"]}


class AIError(Exception):
    """A problem to show to the user as is (already in plain language, never contains the API key)."""


class Cancelled(Exception):
    pass


def _scrub(text: str, secret: str | None) -> str:
    text = str(text)
    if secret:
        text = text.replace(secret, "***")
    return re.sub(r"(key=)[A-Za-z0-9_\-]{12,}", r"\1***", text)


# ====================================================================== options / digest
@dataclass
class Options:
    coached_team: str = "A"  # "A" or "B"
    level_context: str = ""  # free text from the coach, e.g. "NAISTEN 1D" or "WU13 1D"
    model: str = DEFAULT_MODEL
    sample_hz: float = DEFAULT_SAMPLE_HZ
    chunk_s: float = DEFAULT_CHUNK_S
    team_colors: dict[str, str] = field(default_factory=dict)  # playback colours from the 2D view (override)


@dataclass
class Digest:
    """Compact game data for the model. ``frames`` = (t, line) per sample; ``header`` = text shared by all chunks."""
    header: str
    frames: list[tuple[float, str]]
    t_min: float
    t_max: float
    teams: dict[str, dict]
    court_size: tuple[float, float]
    info: dict
    events: list = field(default_factory=list)  # game events (baskets, shots, turnovers, free throws, ...)

    def chunks(self, chunk_s: float) -> list[tuple[float, float, list[str]]]:
        """Split the samples into time windows of about ``chunk_s`` seconds: [(t0, t1, lines)]."""
        if not self.frames:
            return []
        out: list[tuple[float, float, list[str]]] = []
        start = self.frames[0][0]
        cur: list[tuple[float, str]] = []
        for t, line in self.frames:
            if cur and t - start >= chunk_s:
                out.append((cur[0][0], cur[-1][0], [ln for _, ln in cur]))
                cur, start = [], t
            cur.append((t, line))
        if cur:
            # a short tail (< 1/4 chunk) joins the previous chunk instead of being analysed alone
            if out and cur[-1][0] - cur[0][0] < chunk_s / 4:
                t0, _, lines = out.pop()
                out.append((t0, cur[-1][0], lines + [ln for _, ln in cur]))
            else:
                out.append((cur[0][0], cur[-1][0], [ln for _, ln in cur]))
        return out


def _c(v: float) -> str:
    s = f"{v:.1f}"
    return "0.0" if s == "-0.0" else s


def _court_size(d: dx._Data) -> tuple[float, float, list[list[float]]]:
    c = d.replay.get("court") if isinstance(d.replay.get("court"), dict) else {}
    length = dx._f(c.get("length")) or court.LENGTH
    width = dx._f(c.get("width")) or court.WIDTH
    baskets = c.get("baskets") or [list(b) for b in court.baskets()]
    return float(length), float(width), [[round(float(b[0]), 3), round(float(b[1]), 3)] for b in baskets]


def _team_info(d: dx._Data, colors: dict[str, str] | None) -> dict[str, dict]:
    colors = colors or {}
    return {k: {"label": k, "name": d.team_names[k], "color": colors.get(k) or d.team_colors.get(k) or "unknown",
                "jersey": getattr(d, "team_jersey", {}).get(k, ""), "side": getattr(d, "team_sides", {}).get(k, "")}
            for k in ("A", "B")}


def time_windows(t0: float, t1: float, chunk_s: float) -> list[tuple[float, float]]:
    """Windows of ``chunk_s`` covering [t0, t1]; a short tail (< 1/4 window) joins the previous window."""
    out, a = [], t0
    while a < t1 - 1e-9:
        out.append((a, min(t1, a + chunk_s)))
        a += chunk_s
    if len(out) > 1 and out[-1][1] - out[-1][0] < chunk_s / 4:
        last = out.pop()
        out[-1] = (out[-1][0], last[1])
    return out or [(t0, t1)]


def build_game_digest(project_root: Path, team_colors: dict[str, str] | None = None,
                      sample_hz: float = DEFAULT_SAMPLE_HZ, game: Any = None, clean: Any = None,
                      keep: list[tuple[float, float]] | None = None) -> Digest:
    """Read the Replay's tracking data and build the compact representation sent to the model.

    ``game``/``clean`` (from :mod:`replay.ai_light`) reuse loaded data and leave out the rows the cleaning
    removed (impossible jumps, duplicate and bench tracks). ``keep`` = time windows to include raw samples for
    (None = all)."""
    if game is None:
        from replay.ai_light import load_game
        game = load_game(Path(project_root))
    d, ball = game.d, game.ball
    excluded = clean.excluded_tracks if clean is not None else set()
    glitches = clean.glitch_rows if clean is not None else set()
    players = [r for r in game.players if r["track"] not in excluded and (r["track"], r["t"]) not in glitches]
    length, width, baskets = game.length, game.width, game.baskets
    teams = _team_info(d, team_colors)
    team_of_track = game.team_of_track

    by_t: dict[float, list[dict]] = {}
    for r in players:
        by_t.setdefault(r["t"], []).append(r)
    ball_t = {r["t"]: r for r in ball}
    gsd = getattr(game, "gs", None) if isinstance(getattr(game, "gs", None), dict) else None
    lu = (gsd or {}).get("lineup") or {}
    slot_of: dict[str, str] = dict(lu.get("slot_of_track") or {})
    slot_spans = lu.get("slots") or {}
    # jersey numbers read from the video (replay/jersey_numbers.py; tracking.json "tracks" -> "number")
    number_of = {str(k): str(v["number"]) for k, v in d.track_info.items()
                 if isinstance(v, dict) and v.get("number") not in (None, "")}
    # PRIVACY: player names (often minors) and licence numbers are never put into AI prompts: team + jersey only.
    name_of: dict = {}
    last_pos: dict[str, tuple[float, float, float]] = {}  # track -> (t, x, y) of its last row
    for r in players:
        last_pos[r["track"]] = (r["t"], r["x"], r["y"])
    segs = sorted((gsd or {}).get("segments") or [], key=lambda sg: sg["t0"])
    names = {k: teams[k]["name"] for k in ("A", "B")}

    def state_at(t: float) -> str:
        for sg in segs:
            if sg["t0"] - 1e-6 <= t <= sg["t1"] + 1e-6:
                k = sg.get("kind")
                if k == "possession":
                    return f"{sg.get('team')}>{sg.get('basket') or '?'}"
                if k == "free_throws":
                    return f"FT {sg.get('team') or '?'}@{sg.get('basket') or '?'} dead"
                if k in ("timeout", "quarter_break", "half_time", "stoppage"):
                    return "dead"
                return "loose"
        return ""

    def offscreen(t: float, visible: set[str]) -> str:
        parts = []
        for label, spans in sorted(slot_spans.items()):
            if label in visible:
                continue
            last = None
            for tid, a, b in spans:
                if b <= t + 1e-6 and tid in last_pos and last_pos[tid][0] <= t + 1e-6:
                    last = last_pos[tid]
            parts.append(f"{label}@?" if last is None else f"{label}@{last[1]:.0f},{last[2]:.0f}/{t - last[0]:.0f}s")
        return " ".join(parts)
    calib_t: dict[float, bool] = {}
    rt, rc = d.replay.get("t") or [], d.replay.get("calib") or []
    for i, t in enumerate(rt):
        if i < len(rc):
            calib_t[round(float(t), 3)] = bool(rc[i])
    all_t = sorted(set(by_t) | set(ball_t) | {round(float(t), 3) for t in rt})

    step = 1.0 / max(0.1, float(sample_hz))
    frames: list[tuple[float, str]] = []
    nxt = -math.inf
    next_off = -math.inf
    full_range = (all_t[0], all_t[-1]) if all_t else (0.0, 0.0)
    for t in all_t:
        if t + 1e-6 < nxt:
            continue
        if keep is not None and not any(a - 1e-6 <= t <= b + 1e-6 for a, b in keep):
            continue
        nxt = t + step
        rows = by_t.get(t, [])
        if not rows and t not in ball_t:
            calibrated = calib_t.get(round(t, 3), True)
            frames.append((t, f"{t:.1f}|{'no calibration' if not calibrated else 'nobody tracked'}"))
            continue
        groups: dict[str, list[str]] = {"A": [], "B": [], "R": []}
        n_unknown = 0
        visible_slots: set[str] = set()
        for r in sorted(rows, key=lambda r: (slot_of.get(r["track"]) or "~", dx._id_key(r["track"]))):
            key = "R" if r["role"] == "referee" else r["team"] if r["team"] in ("A", "B") else None
            if key is None:
                n_unknown += 1
                continue
            label = r["track"]
            if key in ("A", "B") and slot_of:
                label = slot_of.get(r["track"]) or f"x{r['track']}"  # x = not placed in the 5 on-court slots
                visible_slots.add(label)
            if key in ("A", "B") and r["track"] in number_of:
                label += f"#{number_of[r['track']]}"
            groups[key].append(f"{label}:{_c(r['x'])},{_c(r['y'])}{'*' if r['interp'] else ''}")
        st = state_at(t) if segs else ""
        parts = [f"{t:.1f}"] + ([f"st {st}"] if st else []) + [f"{k} {' '.join(v)}" for k, v in groups.items() if v]
        b = ball_t.get(t)
        if b is not None:
            owner = clean.canonical(b["owner"]) if clean is not None and b["owner"] else b["owner"]
            if st.startswith(("FT", "dead")):
                parts.append(f"ball {_c(b['x'])},{_c(b['y'])} dead")
            elif b["state"] == "possessed" and owner and owner not in excluded:
                owner_team = team_of_track.get(owner, "")
                lab = slot_of.get(owner) or owner
                holder = f"{lab}" if slot_of.get(owner) else f"{owner_team}{owner}" if owner_team in ("A", "B") \
                    else f"?{owner}"
                if owner_team in ("A", "B") and owner in number_of:
                    holder += f"#{number_of[owner]}"
                parts.append(f"ball {_c(b['x'])},{_c(b['y'])} held {holder}")
            else:
                parts.append(f"ball {_c(b['x'])},{_c(b['y'])} loose")
        if n_unknown:
            parts.append(f"U{n_unknown}")
        if slot_spans and t + 1e-6 >= next_off:
            next_off = t + OFFSCREEN_EVERY_S
            off = offscreen(t, visible_slots)
            if off:
                parts.append(f"off {off}")
        frames.append((t, "|".join(parts)))

    all_events, _dismissed = dx._scored_events(d)
    events = [e for e in all_events if e.get("type") == "basket"]
    user_tracks, _auto, _doc = dx._timeline(d, all_events)
    user_tracks = [tr for tr in user_tracks if tr.get("id") != AI_TRACK_ID and tr.get("name") != AI_TRACK_NAME]
    rep = d.report
    cal = rep.get("calibration", {}) or {}
    ppl = rep.get("people", {}) or {}
    n_frames = len(rt) or len(all_t)
    warnings = dx._sanity_warnings(d, players, ball, n_frames, cal.get("valid_pct"), ppl)
    t_min, t_max = full_range
    n_interp = sum(1 for r in players if r["interp"])

    L: list[str] = []
    add = L.append
    add("## GAME")
    add("Replay: (name withheld for privacy)")  # free text may contain names
    if d.game.get("game_date") or d.meta.get("game_date"):
        add(f"Date: {d.game.get('game_date') or d.meta.get('game_date')}")
    for k in ("A", "B"):
        extra = ", ".join(x for x in (f"{teams[k].get('jersey')} jerseys" if teams[k].get("jersey") else "",
                                      teams[k].get("side") or "") if x)
        add(f"Team {k} = {teams[k]['name']} ({extra + '; ' if extra else ''}playback colour {teams[k]['color']})")
    add("Referees are listed as R. People of no team (bench, coaches, spectators) are left out; Un at the end of a "
        "line counts them.")
    add(f"Analysed video time: {t_min:.1f}-{t_max:.1f} s ({dx.fmt_clock(t_min)}-{dx.fmt_clock(t_max)}). "
        "Times are VIDEO seconds and include stoppages; there is no game clock.")
    add("")
    add("## BASKETBALL RULES FOR READING THIS DATA (always true, whatever the samples seem to show)")
    for ln in gsm.SEMANTICS:
        add(f"- {ln}")
    add("")
    add("## COURT")
    add(f"Size {length:g} m x {width:g} m (courts vary; use these numbers). Origin (0,0) = corner of the LEFT "
        "baseline and the NEAR sideline (camera side).")
    add(f"x = along the sideline, 0 = left baseline .. {length:g} = right baseline; half-court line x = "
        f"{length / 2:g}. y = across, 0 = near sideline .. {width:g} = far sideline; middle y = {width / 2:g}.")
    add(f"Baskets (floor point under the ring): L = ({baskets[0][0]:g}, {baskets[0][1]:g}), R = "
        f"({baskets[1][0]:g}, {baskets[1][1]:g}). Three-point line 6.75 m from the basket (0.9 m from the "
        "sideline in the corners); free-throw line 5.8 m from the baseline; paint 4.9 m wide.")
    add("")
    add("## ATTACK DIRECTION (from the game-state engine; teams switch baskets at half time)")
    if gsd and gsd.get("attack_direction"):
        for ln in gsm.direction_lines(gsd, names):
            add(ln)
        add("Use this table for every offence/defence statement. Basket L is at small x, basket R at large x.")
    else:
        add("- Not determined for this Replay: infer it carefully (where a team's baskets and free throws are, "
            "where its players stand while it holds the ball for a long time).")
    add("")
    add("## DATA FORMAT (one line per sample)")
    if slot_of:
        add(f"Sampled at about {sample_hz:g} per second. Line = `t|st <state>|A A1:x,y ...|B B1:x,y ...|R id:x,y|"
            "ball x,y held <slot> | loose | dead|Un|off <slot>@x,y/<s> ...`.")
        add("- t = video seconds. st = game state at that moment: `A>R` = team A has the ball and attacks basket "
            "R; `FT B@L dead` = free throws by team B at basket L (dead ball); `dead` = break / stoppage; `loose` "
            "= nobody controls the ball (shot in the air, rebound, loose ball, unclear).")
        add("- A1..A5 / B1..B5 = the 5 ON-COURT SLOTS of each team (the same slot follows a player across tracker "
            "ID changes, heuristically; not a jersey number). `x<id>` = a team track that could not be placed in "
            "a slot (duplicate or bench person: treat with caution).")
        add(f"- `off ...` (every {OFFSCREEN_EVERY_S:g} s) lists on-court players NOT visible right now with their "
            "last known position and how many seconds ago it was (`@?` = not seen yet). They ARE on the court.")
    else:
        add(f"Sampled at about {sample_hz:g} per second from the tracking. Line = `t|A id:x,y ...|B id:x,y ...|R "
            "id:x,y|ball x,y held <team><id> or loose|Un`.")
        add("- t = video seconds. id = tracker ID: NOT a jersey number and not a stable identity (one player can "
            "get several IDs; an ID can jump to another person after an occlusion).")
    if number_of:
        add("- `#23` after a player label = JERSEY NUMBER read from the video (reliable: accepted only after several "
            "agreeing reads). Same team + same number = same player, whatever the slot or track. Most labels have "
            "no number (it is visible only from some angles): never guess one.")
    add("- x,y = metres on the court (floor point under the feet), rounded to 0.1 m.")
    add("- A trailing * = position interpolated across a short gap (not measured).")
    add("- `ball ... held A3` = that player holds the ball (position = holder's position). `loose` = in "
        "flight/loose, position interpolated between holders (rough). `dead` = dead ball (free throw, break): "
        "ball moves then are NOT passes.")
    add("- A line `t|no calibration` = the court could not be located (no positions); `t|nobody tracked` = "
        "calibrated but no detections. Missing players mean NOT VISIBLE (single panning camera, it mostly "
        "follows the ball), not absent from the court.")
    if number_of:
        add("")
        add("## JERSEY NUMBERS (read from the video)")
        for k in ("A", "B"):
            per: dict[str, set[str]] = {}
            for tid, num in number_of.items():
                if team_of_track.get(tid) == k:
                    per.setdefault(num, set()).add(slot_of.get(tid) or f"track {tid}")
            if per:
                add(f"- {teams[k]['name']} ({k}): " + "; ".join(
                    f"#{num} ({', '.join(sorted(v))})" for num, v in sorted(per.items(), key=lambda x: (len(x[0]), x[0]))))
        add("Name players by number where known (e.g. \"#23 of " + teams["A"]["name"] + "\"); others by slot.")
    add("")
    add("## DATA QUALITY")
    add(f"- Court calibration valid on {cal.get('valid_pct', 'n/a')} % of analysed frames; positions can be off "
        "by 0.5-2 m, more at the far sideline.")
    add(f"- About {ppl.get('avg_tracked_per_calibrated_frame', 'n/a')} people tracked per calibrated frame "
        "(10 players + 2-3 referees are on court).")
    if players:
        add(f"- {dx._pct(n_interp, len(players))} % of person positions are interpolated.")
    for a, b in ((rep.get("failed_ranges") or {}).get("no_calibration") or [])[:20]:
        add(f"- No calibration {a:.1f}-{b:.1f} s.")
    for w in warnings:
        add(f"- WARNING: {w}")
    if clean is not None:
        c = clean.to_dict()
        add(f"- Cleaned before sending: {c['glitch_rows_removed']} impossible jumps (> 11 m/s) removed; "
            f"{len(c['duplicate_tracks'])} duplicate track(s) and {len(c['bench_or_sideline_tracks'])} "
            "bench/sideline track(s) left out.")
        if c["tracking_gaps_s"]:
            add("- Calibrated but nobody tracked: " + ", ".join(f"{a}-{b} s" for a, b in c["tracking_gaps_s"][:20]))
    add("")
    add("## DETECTED BASKETS (automatic, not the official score; 2 points unless recognised as a free throw (1) "
        "or a probable 3)")
    if events:
        for e in events[:200]:
            add(f"- t={(dx._f(e.get('t')) or 0):.1f} Basket - basket {e.get('basket') or '?'} by "
                f"{e['team_name'] or 'unknown team'} (confidence {(dx._f(e.get('confidence')) or 0):.2f})"
                f" -> {e['score_A']}-{e['score_B']}" + ("; note: [withheld]" if e.get("note") else ""))
    else:
        add("- none detected")
    add("")
    add("## COACH'S TIMELINE NOTES (written by a human: trustworthy context)")
    n = 0
    for tr in user_tracks:
        for m in tr["marks"]:
            if n >= 150:
                break
            end = f"-{m['t'] + m['duration']:.1f}" if m["duration"] > 0 else ""
            # PRIVACY: coach free text (track name, title, note) may contain player names -> neutral placeholders
            note = ": [note withheld]" if m["note"] else ""
            add(f"- [timeline] t={m['t']:.1f}{end} {'[title withheld]' if m['title'] else '(no title)'}{note}")
            n += 1
    if n == 0:
        add("- none")
    info = {"samples": len(frames), "sample_hz": sample_hz, "player_rows": len(players), "ball_rows": len(ball),
            "warnings": warnings, "events": len(events), "game_events": len(all_events) - len(events),
            "user_marks": n, "game_state": bool(gsd)}
    return Digest(header="\n".join(L), frames=frames, t_min=t_min, t_max=t_max, teams=teams,
                  court_size=(length, width), info=info,
                  events=[e for e in all_events if e.get("type") != "basket"
                          and (dx._f(e.get("confidence")) or 1.0) >= gsm.TIMELINE_MIN_CONF])


# ====================================================================== prompts
LEVEL_GUIDE = (
    "Calibrate everything to the competition level and age group described by the coach. Examples: an adult "
    "national league (e.g. 'NAISTEN 1D' = Finnish women's 1st division, 'Miesten 1-divari', 'Korisliiga') runs "
    "real sets, help rotations, zone and press variations: be specific and demanding (rotation timing, tagging "
    "the roller, closeout angles, spacing against the zone, shot-clock management, matchup choices). A youth "
    "level (e.g. 'WU13 1D' = girls under 13, 'P12', 'T14') is about fundamentals of team play: spread the floor, "
    "run back on defence, see the ball and your player, box out, give the ball handler outlets, simple "
    "man-to-man principles: keep it simple, encouraging, and do not expect adult-level schemes. If no level is "
    "given, assume an amateur adult team and say so in the summary.")


def _jersey(team: dict) -> str:
    return f"{team['jersey']} jerseys, " if team.get("jersey") else ""


def build_system_prompt(teams: dict[str, dict], coached: str, level_context: str) -> str:
    other = "B" if coached == "A" else "A"
    ct, ot = teams[coached], teams[other]
    level = level_context.strip() or "(not given)"
    return "\n".join([
        "You are an experienced basketball coach and video analyst. You get 2D player tracking data of a game "
        "(positions on the court over time), measured automatically from one video, and you write a TACTICAL "
        "analysis for the coaching staff of one team.",
        "",
        f"The user coaches team {coached} = {ct['name']} ({_jersey(ct)}colour {ct['color']}). The opponent is team "
        f"{other} = {ot['name']} ({_jersey(ot)}colour {ot['color']}). Analyse from {ct['name']}'s point of view: what they did well and "
        "what they should improve, in offense, defense and transition. Mention the opponent only where it "
        "explains what the coached team should do.",
        "",
        f"Competition level / context from the coach: \"\"\"{level}\"\"\"",
        LEVEL_GUIDE,
        "",
        "Scope: ONLY team-level tactics that positions can show: spacing and floor balance, ball and player "
        "movement, set/action execution, pick-and-roll coverage, help defence and rotations, closeouts, zone vs "
        "man coverage, matchups in transition, transition defence (getting back, protecting the basket, "
        "matching up), fast breaks and lane filling, press and press break, rebounding positioning (boxing out, "
        "crashing, guarding against the break), late-clock and out-of-bounds situations. Do NOT comment on "
        "individual technique (shooting form, footwork, dribbling) or anything the data cannot show (talk, "
        "effort, referee calls). Never invent numbers, jersey numbers or player names; refer to players by role "
        "or position (\"the weak-side wing\", \"the big at the low post\").",
        "Be honest about the data: positions are imperfect, players outside the camera view are not in the "
        "samples (but they ARE on the court), IDs are not identities. Base every issue on a pattern you can see "
        "in the data at the given times; prefer patterns that repeat. If the data cannot support a point, leave "
        "it out.",
        "",
        "Basketball facts you must respect (the data was prepared with them):",
        "- Both teams always have 5 players on court in live play. A player missing from a sample is outside the "
        "camera view, not absent: never claim a team played with fewer players or left an area empty only "
        "because nobody is tracked there.",
        "- Free throws (segments/state `FT ... dead`, events 'Free throw k/n') are dead balls: the ball going from "
        "the referee to the shooter is NOT a pass, the line-up along the lane is NOT spacing or a set, and they "
        "are not turnovers. Do not report tactical issues inside free throws, time-outs or stoppages.",
        "- Which basket each team attacks is given in ATTACK DIRECTION (it switches at half time). Who has the "
        "ball is given by the possession segments (`st` in the samples) and the GAME EVENTS log (shots, "
        "rebounds, turnovers, steals, fast breaks, inbounds). Use them as the facts of the game; only contradict "
        "them if the samples clearly show otherwise, and then say so.",
        "- Offence = the team with the ball attacking its basket; defence = the other team between the ball and "
        "its own basket. Never swap them.",
        "",
        "Answer with JSON only, exactly this structure:",
        '{"overview": {"went_well": [string, ...], "to_improve": [string, ...], "summary": string}, '
        '"issues": [{"t_start": number, "t_end": number, "title": string, "description": string, '
        '"category": string, "team_phase": "offense" | "defense" | "transition"}]}',
        "- went_well / to_improve: 3-6 short, concrete bullet points each (team tactics).",
        "- summary: 3-6 sentences for the coach.",
        "- issues: concrete moments worth showing at a video session, ordered by time. t_start/t_end in VIDEO "
        "seconds within the analysed range (a situation usually lasts 3-15 s; start a little before it "
        "develops). title <= 8 words. description: what happened, why it matters, what to do instead (2-4 "
        "sentences). category: a short tactical label such as spacing, help defense, transition defense, "
        "rebounding, press break, set execution, zone offense, man defense, pick and roll defense, fast break. "
        "Aim for about 4-12 issues per 10 minutes of play: the most important ones, not every possession.",
        "- team_phase is from the coached team's point of view and must be exactly one of offense, defense, "
        "transition (keep these three words in English).",
        "",
        "Language: write all text values (bullets, summary, titles, descriptions, categories) in the language "
        "the coach wrote the context in (for example Finnish if the context is Finnish). If the context is "
        "empty or only a code like 'WU13 1D', use the language of any words in it; if still unclear, English. "
        "Keep the JSON keys in English.",
    ])


SEGMENT_GUIDE = (
    "## SEGMENTS (prepared by the light tier: Python statistics, labels by rules or by the TypeSafe Jev model)\n"
    "One row per possession / dead-ball spell, from the COACHED team's point of view. Columns: id; t0,t1 = video "
    "seconds; phase = offense_half_court | defense_half_court | transition_offense | transition_defense | "
    "dead_ball | unclear; src = rules or jev<confidence>; ball = who holds the ball (coached/opponent/none); "
    "attacks = basket the ball team attacks (L/R); cross_s = seconds until the ball crossed half court; passes; "
    "chg = y if possession changed; c_vis,o_vis = coached/opponent players visible on average; "
    "c_spread,o_spread = mean distance (m) of a team's visible players to their centroid (spacing); "
    "def_behind = defenders between the ball and their basket on average; scheme = defensive scheme guess; "
    "review = probability a coach wants to see it (jev); reliable = data completeness 0..1; baskets = "
    "t + c(oached)/o(pponent)/u(nknown); kind = possession | free_throws | timeout | quarter_break | half_time | "
    "stoppage | loose_or_unclear (game-state engine); start/end = how the possession started/ended (after shot "
    "= rebound, steal, after free throws, inbound, made basket, missed shot, turnover); lost = probability the "
    "team lost the ball (jev); events = detected events inside. These are measured facts/estimates; check them "
    "against the samples.")


def build_user_prompt(digest: Digest, t0: float, t1: float, lines: list[str], part: int, parts: int,
                      segments: str | None = None, selective: bool = False) -> str:
    head = digest.header
    if segments:
        head += f"\n\n{SEGMENT_GUIDE}\n{segments}"
    if selective:
        lines = ["(Raw samples are included ONLY around the segments selected for review; use the segment table "
                 "for the rest of the game.)"] + list(lines)
    if digest.events:
        names = {k: v["name"] for k, v in digest.teams.items()}
        ev = gsm.event_lines(digest.events, names, limit=300, t_range=(t0, t1))
        if ev:
            head += ("\n\n## GAME EVENTS IN THIS PART (automatic, with confidence; FT = free throw, dead ball)\n"
                     + "\n".join(ev))
    scope = (f"## THIS REQUEST\nPart {part} of {parts} of the game: video time {t0:.1f}-{t1:.1f} s. Report issues "
             "only inside this window. The overview should describe this part."
             if parts > 1 else f"## THIS REQUEST\nThe whole analysed game: video time {t0:.1f}-{t1:.1f} s.")
    return f"{head}\n\n{scope}\n\n## TRACKING SAMPLES\n" + "\n".join(lines) + "\n"


def build_merge_prompt(overviews: list[tuple[float, float, dict]]) -> str:
    parts = []
    for i, (t0, t1, ov) in enumerate(overviews, 1):
        parts.append(f"### Part {i} ({t0:.0f}-{t1:.0f} s)\n" + json.dumps(ov, ensure_ascii=False))
    return ("The game was analysed in parts. Merge the partial overviews below into ONE overview of the whole "
            "game for the coach: combine repeated points, keep the most important 3-6 bullets per list, mention "
            "trends over the game (e.g. something that improved after half time), and keep the same language. "
            "Answer with JSON {\"overview\": {\"went_well\": [...], \"to_improve\": [...], \"summary\": \"...\"}}.\n\n"
            + "\n\n".join(parts))


def estimate_tokens(text: str) -> int:
    return int(len(text) / 3.5) + 1


# ====================================================================== parsing
def _extract_json(text: str) -> Any:
    s = (text or "").strip()
    m = re.match(r"^```(?:json)?\s*(.*?)\s*```$", s, re.S)
    if m:
        s = m.group(1)
    try:
        return json.loads(s)
    except ValueError:
        pass
    a, b = s.find("{"), s.rfind("}")
    if 0 <= a < b:
        try:
            return json.loads(s[a:b + 1])
        except ValueError:
            pass
    raise AIError("The AI answer was not valid JSON (it may have been cut off). Try again, or try another model.")


def _str_list(v: Any) -> list[str]:
    if isinstance(v, str):
        v = [v]
    if not isinstance(v, list):
        return []
    return [str(x).strip() for x in v if isinstance(x, (str, int, float)) and str(x).strip()]


def _overview(v: Any) -> dict:
    v = v if isinstance(v, dict) else {}
    return {"went_well": _str_list(v.get("went_well")), "to_improve": _str_list(v.get("to_improve")),
            "summary": str(v.get("summary") or "").strip()}


def _phase(v: Any) -> str:
    s = str(v or "").strip().lower()
    s = PHASE_ALIASES.get(s, s)
    return s if s in TEAM_PHASES else "transition" if "transition" in s else "defense" if "def" in s else \
        "offense" if "off" in s else "other"


def parse_response(text: str, t_min: float, t_max: float, warnings: list[str] | None = None) -> dict:
    """Validate the model's answer. Issues outside [t_min, t_max] (more than a few seconds) are dropped, the rest
    clamped; malformed issues are skipped. Problems are appended to ``warnings``."""
    warnings = warnings if warnings is not None else []
    data = _extract_json(text)
    if not isinstance(data, dict):
        raise AIError("The AI answer had an unexpected structure (not a JSON object).")
    if "overview" not in data and "issues" not in data:
        raise AIError("The AI answer did not contain an overview or issues.")
    issues, dropped, bad = [], 0, 0
    for it in data.get("issues") or [] if isinstance(data.get("issues"), list) else []:
        if not isinstance(it, dict):
            bad += 1
            continue
        a, b = dx._f(it.get("t_start")), dx._f(it.get("t_end"))
        title = str(it.get("title") or "").strip()
        if a is None or not (title or it.get("description")):
            bad += 1
            continue
        if b is None:
            b = a
        if b < a:
            a, b = b, a
        if b < t_min - TIME_TOLERANCE_S or a > t_max + TIME_TOLERANCE_S:
            dropped += 1
            continue
        a, b = max(t_min, min(t_max, a)), max(t_min, min(t_max, b))
        issues.append({"t_start": round(a, 2), "t_end": round(b, 2),
                       "title": title or str(it.get("description"))[:60],
                       "description": str(it.get("description") or "").strip(),
                       "category": str(it.get("category") or "").strip() or "other",
                       "team_phase": _phase(it.get("team_phase"))})
    if dropped:
        warnings.append(f"{dropped} issue(s) with times outside the analysed range were left out.")
    if bad:
        warnings.append(f"{bad} malformed issue(s) were left out.")
    issues.sort(key=lambda i: (i["t_start"], i["t_end"]))
    return {"overview": _overview(data.get("overview")), "issues": issues}


def merge_overviews_locally(overviews: list[dict], cap: int = 8) -> dict:
    """Fallback merge (no extra API call): concatenate unique bullets and summaries."""
    out = {"went_well": [], "to_improve": [], "summary": ""}
    for k in ("went_well", "to_improve"):
        seen = set()
        for ov in overviews:
            for s in ov.get(k, []):
                if s.lower() not in seen:
                    seen.add(s.lower())
                    out[k].append(s)
        out[k] = out[k][:cap]
    out["summary"] = "\n\n".join(ov["summary"] for ov in overviews if ov.get("summary"))
    return out


# ====================================================================== client
class Client(Protocol):
    def generate(self, model: str, system: str, user: str, schema: dict) -> str: ...


class GeminiClient:
    """``models.generateContent`` in JSON mode. Uses the official google-genai SDK when it is installed, otherwise
    the documented REST endpoint. The key lives only in this object."""

    def __init__(self, api_key: str, timeout_s: float = REQUEST_TIMEOUT_S, use_sdk: bool | None = None):
        key = (api_key or "").strip()
        if not key:
            raise AIError("Enter your Google AI Studio API key.")
        self._key = key
        self.timeout_s = timeout_s
        if use_sdk is None:
            try:
                import google.genai  # noqa: F401
                use_sdk = True
            except ImportError:
                use_sdk = False
        self.use_sdk = use_sdk

    def __repr__(self) -> str:  # never show the key
        return f"GeminiClient(sdk={self.use_sdk})"

    def generate(self, model: str, system: str, user: str, schema: dict) -> str:
        try:
            return self._sdk(model, system, user, schema) if self.use_sdk else self._rest(model, system, user, schema)
        except AIError:
            raise
        except Exception as e:  # map everything to a readable, key-free message
            raise AIError(_friendly_error(e, self._key, model)) from None

    def _sdk(self, model: str, system: str, user: str, schema: dict) -> str:
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=self._key,
                              http_options=types.HttpOptions(timeout=int(self.timeout_s * 1000)))
        resp = client.models.generate_content(
            model=model, contents=user,
            config=types.GenerateContentConfig(system_instruction=system, response_mime_type="application/json",
                                               response_schema=schema, temperature=0.4))
        text = resp.text
        if not text:
            raise AIError(_empty_reason(_dump(resp)))
        return text

    def _rest(self, model: str, system: str, user: str, schema: dict) -> str:
        import urllib.request
        body = {"systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": user}]}],
                "generationConfig": {"responseMimeType": "application/json", "responseSchema": schema,
                                     "temperature": 0.4}}
        req = urllib.request.Request(f"{API_BASE}/models/{model}:generateContent",
                                     data=json.dumps(body).encode("utf-8"), method="POST",
                                     headers={"Content-Type": "application/json", "x-goog-api-key": self._key})
        with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
            data = json.loads(r.read().decode("utf-8"))
        return response_text(data)


def _dump(resp: Any) -> dict:
    try:
        return resp.model_dump(mode="json", by_alias=True, exclude_none=True)
    except Exception:
        return {}


def response_text(data: dict) -> str:
    """Text of a REST generateContent response; readable error when it was blocked or empty."""
    cands = data.get("candidates") or []
    texts = []
    for c in cands[:1]:
        for p in ((c.get("content") or {}).get("parts") or []):
            if isinstance(p, dict) and isinstance(p.get("text"), str) and not p.get("thought"):
                texts.append(p["text"])
    if texts:
        return "".join(texts)
    raise AIError(_empty_reason(data))


def _empty_reason(data: dict) -> str:
    fb = data.get("promptFeedback") or data.get("prompt_feedback") or {}
    block = fb.get("blockReason") or fb.get("block_reason")
    if block:
        return f"The request was blocked by the API ({block})."
    cands = data.get("candidates") or []
    reason = (cands[0].get("finishReason") or cands[0].get("finish_reason")) if cands else None
    if reason == "MAX_TOKENS":
        return "The AI answer was cut off (output limit reached). Try a lower sample rate or a stronger model."
    return f"The AI returned an empty answer{f' (finish reason {reason})' if reason else ''}. Try again."


def _friendly_error(e: Exception, key: str, model: str) -> str:
    code, status, msg = None, "", str(e)
    # google-genai SDK errors
    if hasattr(e, "code") and isinstance(getattr(e, "code"), int):
        code, status, msg = e.code, str(getattr(e, "status", "") or ""), str(getattr(e, "message", "") or e)
    # urllib HTTPError
    try:
        import urllib.error
        if isinstance(e, urllib.error.HTTPError):
            code = e.code
            try:
                err = json.loads(e.read().decode("utf-8")).get("error", {})
                status, msg = str(err.get("status") or ""), str(err.get("message") or msg)
            except Exception:
                pass
        elif isinstance(e, urllib.error.URLError):
            return _scrub(f"Network error: could not reach the Gemini API ({e.reason}). Check the internet "
                          "connection.", key)
    except ImportError:
        pass
    low = (msg + " " + status).lower()
    if code is None:
        name = type(e).__name__.lower()
        if "timeout" in name or "timed out" in low:
            return "The Gemini API did not answer in time. Try again, or analyse at a lower sample rate."
        if any(s in name for s in ("connect", "network", "gaierror", "oserror")) or "getaddrinfo" in low:
            return _scrub(f"Network error: could not reach the Gemini API ({e}). Check the internet connection.",
                          key)
        return _scrub(f"Unexpected error from the Gemini API: {type(e).__name__}: {msg}", key)
    if "api key" in low and ("invalid" in low or "not valid" in low) or "api_key_invalid" in low:
        return "The API key was rejected (invalid key). Copy it again from Google AI Studio."
    if code in (401, 403):
        return _scrub(f"Permission denied ({code}): the key is not allowed to use this model or the Gemini API "
                      f"is not enabled for its project. Details: {msg}", key)
    if code == 404:
        return _scrub(f"Model '{model}' was not found ({msg}). Check the model name.", key)
    if code == 429 or "resource_exhausted" in low:
        return _scrub(f"Quota or rate limit reached (429). Wait a minute and try again, or check your AI Studio "
                      f"plan. Details: {msg}", key)
    if code == 400:
        return _scrub(f"The API rejected the request (400): {msg}", key)
    if code >= 500:
        return _scrub(f"The Gemini service had a problem ({code}). Try again in a moment. Details: {msg}", key)
    return _scrub(f"Gemini API error {code}: {msg}", key)


# ====================================================================== run
def _check(cancel: threading.Event | None) -> None:
    if cancel is not None and cancel.is_set():
        raise Cancelled()


def _call_with_retry(client: Client, model: str, system: str, user: str, schema: dict,
                     cancel: threading.Event | None, retries: int = 2) -> str:
    for attempt in range(retries + 1):
        _check(cancel)
        try:
            return client.generate(model, system, user, schema)
        except AIError as e:
            transient = any(s in str(e) for s in ("(500)", "(502)", "(503)", "(504)", "service had a problem",
                                                  "did not answer in time"))
            if not transient or attempt == retries:
                raise
            for _ in range(int(5 * (attempt + 1) / 0.25)):  # back off, cancellable
                _check(cancel)
                time.sleep(0.25)
    raise AssertionError("unreachable")


def review_windows(segments: list[dict], pad_s: float = 2.0) -> list[tuple[float, float]] | None:
    """Time windows that get raw samples when the light model has rated the segments: segments it marks as worth
    reviewing (or with a basket); at least the top quarter. None = everything (nothing rated)."""
    rated = [s for s in segments if isinstance(s.get("review"), (int, float))]
    if not rated:
        return None
    chosen = [s for s in segments if (s.get("review") or 0) >= 0.5 or s.get("baskets")]
    if len(chosen) < max(1, len(segments) // 4):
        top = sorted(rated, key=lambda s: -s["review"])[:max(1, len(segments) // 4)]
        chosen = sorted({id(s): s for s in chosen + top}.values(), key=lambda s: s["t0"])
    if len(chosen) == len(segments):
        return None
    out: list[list[float]] = []
    for s in sorted(chosen, key=lambda s: s["t0"]):
        a, b = s["t0"] - pad_s, s["t1"] + pad_s
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def run_analysis(project_root: Path, opts: Options, client: Client,
                 progress: Callable[[float, str], None] | None = None,
                 cancel: threading.Event | None = None, light_client: Any = None) -> dict:
    """Analyse the Replay and return the result document (not saved; see :func:`save_results`).

    Two tiers: :func:`replay.ai_light.run_light` cleans and segments the data in Python and, when
    ``light_client`` (a :class:`replay.ai_light.JevClient`) is given, labels the segments with TypeSafe Jev; then
    ``client`` (Gemini) gets the cleaned header, the segment table and the raw samples it needs."""
    prog = progress or (lambda f, m: None)
    if opts.coached_team not in ("A", "B"):
        raise AIError("Choose the team you are coaching.")
    prog(0.01, "Light tier: cleaning and segmenting the tracking data")
    light = ai_light.run_light(Path(project_root), opts.coached_team, light_client,
                               lambda f, m: prog(0.01 + 0.14 * f, m), cancel)
    _check(cancel)
    keep = review_windows(light.segments) if light.used_jev else None
    digest = build_game_digest(project_root, opts.team_colors, opts.sample_hz, game=light.game,
                               clean=light.clean, keep=keep)
    if not digest.frames and not light.segments:
        raise AIError("This Replay has no tracking samples to analyse.")
    system = build_system_prompt(digest.teams, opts.coached_team, opts.level_context)
    windows = time_windows(digest.t_min, digest.t_max, max(60.0, opts.chunk_s))
    n = len(windows)
    warnings: list[str] = list(digest.info.get("warnings") or []) + list(light.notes)
    parsed: list[tuple[float, float, dict]] = []
    tokens = 0
    for i, (t0, t1) in enumerate(windows, 1):
        last = i == n
        inside = (lambda t, a=t0, b=t1, last=last: a - 1e-6 <= t <= b + 1e-6 if last else a - 1e-6 <= t < b - 1e-6)
        lines = [ln for t, ln in digest.frames if inside(t)]
        segs = [sg for sg in light.segments if inside(sg["t0"])]
        user = build_user_prompt(digest, t0, t1, lines, i, n, ai_light.segment_table(segs) if segs else None,
                                 selective=keep is not None)
        tokens += estimate_tokens(system) + estimate_tokens(user)
        prog(0.15 + 0.77 * (i - 1) / n, f"Asking {opts.model}" + (f" (part {i}/{n}, {dx.fmt_clock(t0)}-"
                                                                 f"{dx.fmt_clock(t1)})" if n > 1 else "")
             + " – this can take a minute")
        text = _call_with_retry(client, opts.model, system, user, RESPONSE_SCHEMA, cancel)
        _check(cancel)
        try:
            res = parse_response(text, t0, t1, warnings)
        except AIError:
            # one more try for a malformed answer
            text = _call_with_retry(client, opts.model, system, user, RESPONSE_SCHEMA, cancel)
            res = parse_response(text, t0, t1, warnings)
        parsed.append((t0, t1, res))
    issues = sorted((it for _, _, r in parsed for it in r["issues"]), key=lambda i: (i["t_start"], i["t_end"]))
    if n == 1:
        overview = parsed[0][2]["overview"]
    else:
        prog(0.93, "Merging the parts into one overview")
        try:
            text = _call_with_retry(client, opts.model, system, build_merge_prompt(
                [(a, b, r["overview"]) for a, b, r in parsed]), MERGE_SCHEMA, cancel)
            overview = _overview(_extract_json(text).get("overview"))
            if not (overview["summary"] or overview["went_well"] or overview["to_improve"]):
                raise AIError("empty merge")
        except AIError:
            overview = merge_overviews_locally([r["overview"] for _, _, r in parsed])
            warnings.append("The overview of the parts was merged without the AI (the merge request failed).")
    prog(1.0, "Done")
    teams = digest.teams
    tiers = list(light.log) + [{
        "tier": "gemini", "task": "tactical analysis",
        "detail": f"{opts.model}: overview and {len(issues)} issue timestamp(s) from {n} request(s)"
                  + (" + 1 merge request" if n > 1 else "") + f", about {tokens} input tokens; raw samples "
                  + ("only for the segments selected for review" if keep is not None else "for the whole game")}]
    return {
        "format": FORMAT,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "app_version": __version__,
        "model": opts.model,
        "light_model": {"provider": ai_light.PROVIDER_NAME, "used": light.used_jev, **light.jev},
        "tiers": tiers,
        "coached_team": {**teams[opts.coached_team]},
        "opponent": {**teams["B" if opts.coached_team == "A" else "A"]},
        "level_context": opts.level_context,
        "sample_hz": opts.sample_hz,
        "parts": [[round(a, 2), round(b, 2)] for a, b in windows],
        "time_range_s": [digest.t_min, digest.t_max],
        "court_m": list(digest.court_size),
        "estimated_input_tokens": tokens,
        "raw_sample_windows_s": None if keep is None else [[round(a, 1), round(b, 1)] for a, b in keep],
        "cleaning": light.clean.to_dict(),
        "segments": light.segments,
        "light_events": list(getattr(light, "events", []) or []),
        "game_state": _gs_summary(getattr(light.game, "gs", None) if light.game is not None else None),
        "overview": overview,
        "issues": issues,
        "warnings": warnings,
    }


def _gs_summary(gsd: dict | None) -> dict | None:
    if not isinstance(gsd, dict):
        return None
    kinds: dict[str, int] = {}
    for e in gsd.get("events") or []:
        kinds[e.get("type")] = kinds.get(e.get("type"), 0) + 1
    return {"attack_direction": gsd.get("attack_direction"), "possessions": len(gsd.get("possessions") or []),
            "free_throw_situations": len(gsd.get("free_throw_windows") or []), "events_by_type": kinds}


# ====================================================================== storage / timeline
def results_path(project_root: Path) -> Path:
    return Path(project_root) / "results" / RESULTS_FILE


def save_results(project_root: Path, result: dict) -> Path:
    path = results_path(project_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {k: v for k, v in result.items() if "key" not in k.lower()}  # defensive: never persist a key
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)
    return path


def load_results(project_root: Path) -> dict | None:
    try:
        d = json.loads(results_path(project_root).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(d, dict) or d.get("format") != FORMAT:
        return None
    d["overview"] = _overview(d.get("overview"))
    d["issues"] = [i for i in (d.get("issues") or []) if isinstance(i, dict) and dx._f(i.get("t_start")) is not None]
    return d


def apply_to_timeline(tl: Any, result: dict) -> Any:
    """Put the issues on the "AI analysis" track of a :class:`replay.gui.timeline_model.Timeline` (created if
    missing), replacing the marks of a previous run. Returns the track."""
    from replay.gui import timeline_model as tm
    tr = tl.track(AI_TRACK_ID) or next((t for t in tl.tracks if t.name == AI_TRACK_NAME and not t.is_auto), None)
    if tr is None:
        tr = tm.Track(AI_TRACK_NAME, AI_TRACK_COLOR, tm.USER, id=AI_TRACK_ID)
        tl.tracks.append(tr)
    tr.marks = []
    for it in result.get("issues") or []:
        a = max(0.0, dx._f(it.get("t_start")) or 0.0)
        b = max(a, dx._f(it.get("t_end")) or a)
        phase = it.get("team_phase") or ""
        tag = " · ".join(x for x in (it.get("category"), phase) if x)
        note = (it.get("description") or "") + (f"\n[{tag}]" if tag else "")
        tr.marks.append(tm.Mark(t=a, duration=b - a, title=str(it.get("title") or ""), note=note.strip(),
                                color=PHASE_COLORS.get(phase)))
    tr.marks.sort(key=lambda m: m.t)
    return tr
