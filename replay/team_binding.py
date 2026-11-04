"""Stable team identity: bind the two detected teams to their jersey colours, not to cluster order.

Problem: team classification (teams.py) finds two jersey clusters and calls them A/B in an order that depends on
the data (which team is seen more), so a rebuild after any change could swap them. meta.json home/away names are
often placeholders ("Home"/"Away"), so "Home" ended up on whichever colour happened to be A.

Binding (stored in meta.json under ``team_binding``; without a meta.json in ``<out>/team_binding.json``):

    {"version": 1,
     "home": {"color": "#rrggbb", "lab": [L, a, b], "hist": {"video3": [...], "pass1_3": [...]}, "colour_name": "purple"},
     "away": {...},
     "confirmed": false,          # True once the user decided (Overview, picked players, kit colours, swap)
     "source": "auto" | "scorebar" | "user" | "samples" | "kit_colours" | "swap",
     "swap_flag": false,          # project setting "swap_teams" when the binding was last written
     "samples_key": ..., "kit_key": ...}   # fingerprints of the user's picked players / kit colours

``home`` is always label ``A``, ``away`` label ``B`` (as everywhere downstream).

On every build (:func:`bind`):
  * no binding yet: the classification's own order (user-picked players / matching user kit colours decide it; else
    a best guess - the score-bar colour cue when the bar shows team colours) is stored;
  * binding exists: the new clusters are matched to the stored colour signatures (2x2 assignment on histogram
    Hellinger + Lab hue distance), so A/B never swap between rebuilds;
  * the user picked new players / set new kit colours: that wins and re-writes the binding;
  * the "swap teams" setting is a toggle relative to the binding: when it differs from ``swap_flag`` the sides are
    exchanged once and the binding is re-written (colours follow the players, names stay with home/away).

Names (:func:`resolved_names`): while the sides are not confirmed by the user, teams are called after their jersey
colour ("Team orange" / "Team purple") instead of a possibly wrong "Home"/"Away" or real team name.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

FIELD = "team_binding"
FILENAME = "team_binding.json"
VERSION = 1
SIDES = ("home", "away")
LABEL_OF = {"home": "A", "away": "B"}
SIDE_OF = {"A": "home", "B": "away"}
PLACEHOLDER_NAMES = {"", "home", "away", "home team", "away team", "a", "b", "team a", "team b", "kotijoukkue",
                     "vierasjoukkue", "koti", "vieras"}
AMBIGUOUS_REL = 0.08  # straight/crossed costs this close (relative) = colours cannot tell the teams apart


# ============================================================================ colours and names
def _hex_rgb(h: str) -> tuple[int, int, int] | None:
    try:
        h = str(h).strip().lstrip("#")
        if len(h) != 6:
            return None
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    except (TypeError, ValueError):
        return None


def hex_to_lab(h: str) -> np.ndarray | None:
    rgb = _hex_rgb(h)
    if rgb is None:
        return None
    from replay.features import rgb_to_lab
    return np.asarray(rgb_to_lab(rgb), float)


def colour_name(color: str | None) -> str:
    """Plain colour word for a #rrggbb colour (red, orange, yellow, green, teal, blue, purple, pink, white, grey,
    black)."""
    rgb = _hex_rgb(color or "")
    if rgb is None:
        return "unknown"
    r, g, b = (c / 255.0 for c in rgb)
    mx, mn = max(r, g, b), min(r, g, b)
    v, s = mx, (0.0 if mx <= 0 else (mx - mn) / mx)
    if s < 0.22 or mx - mn < 0.08:
        return "white" if v > 0.82 else "black" if v < 0.25 else "grey"
    if mx == r:
        hue = (60 * ((g - b) / (mx - mn))) % 360
    elif mx == g:
        hue = 60 * ((b - r) / (mx - mn)) + 120
    else:
        hue = 60 * ((r - g) / (mx - mn)) + 240
    if v < 0.2:
        return "black"
    for limit, name in ((12, "red"), (42, "orange"), (68, "yellow"), (160, "green"), (195, "teal"), (255, "blue"),
                        (295, "purple"), (340, "pink"), (361, "red")):
        if hue < limit:
            return name
    return "red"


def lab_colour_name(lab) -> str:
    """Colour word for a measured (dull) jersey Lab colour - better than the boosted display colour, which turns
    orange-red kits pure red."""
    L, a, b = (float(v) for v in lab)
    C = float(np.hypot(a, b))
    if C < 9:
        return "white" if L > 75 else "black" if L < 22 else "grey"
    h = float(np.degrees(np.arctan2(b, a))) % 360
    if h < 45 and C < 32 and L < 35:
        return "red"  # dark, dull red-orange = maroon / burgundy (a real orange kit is far more saturated)
    for limit, name in ((22, "red"), (68, "orange"), (100, "yellow"), (165, "green"), (215, "teal"), (285, "blue"),
                        (325, "purple"), (352, "pink"), (361, "red")):
        if h < limit:
            return name
    return "red"


def distinct_colors(cols: dict, min_de: float = 45.0) -> dict:
    """Display colours for teams "A" / "B": when the two are nearly the same (light vs dark orange, orange vs red) push
    the lighter one lighter and the darker one darker (keeping the hue) until they read as different teams on screen.
    The detected jersey colours themselves are not changed, only how the teams are drawn."""
    import colorsys
    a, b = cols.get("A"), cols.get("B")
    la, lb = hex_to_lab(a or ""), hex_to_lab(b or "")
    if la is None or lb is None or float(np.linalg.norm(la - lb)) >= min_de:
        return dict(cols)

    def hls(h):
        r, g, bl = _hex_rgb(h)  # type: ignore[misc]
        return colorsys.rgb_to_hls(r / 255, g / 255, bl / 255)

    def hexof(hh, ll, ss):
        r, g, bl = colorsys.hls_to_rgb(hh, min(max(ll, 0.0), 1.0), min(max(ss, 0.0), 1.0))
        return "#%02x%02x%02x" % (round(r * 255), round(g * 255), round(bl * 255))

    (ha, l_a, sa), (hb, l_b, sb) = hls(a), hls(b)
    light_is_a = la[0] >= lb[0]
    hl, ll_, sl = (ha, l_a, sa) if light_is_a else (hb, l_b, sb)
    hd, ld, sd = (hb, l_b, sb) if light_is_a else (ha, l_a, sa)
    light = hexof(hl, max(ll_, 0.68), max(sl, 0.75))
    dark = hexof(hd, min(ld, 0.26), max(sd, 0.75))
    out = dict(cols)
    out["A"], out["B"] = (light, dark) if light_is_a else (dark, light)
    return out


def colour_team_names(color_a: str | None, color_b: str | None, words: tuple[str, str] | None = None) -> dict[str, str]:
    """{"A": "Team orange", "B": "Team purple"}; "Team A (light orange)" / "Team B (dark orange)" when both get the
    same colour word.
    ``words``: colour words to use (e.g. from the measured jersey Lab) instead of naming the display colours."""
    na, nb = words if words else (colour_name(color_a), colour_name(color_b))
    if na == nb:
        la, lb = hex_to_lab(color_a or ""), hex_to_lab(color_b or "")
        if la is not None and lb is not None and abs(la[0] - lb[0]) > 1:
            ta, tb = (f"light {na}", f"dark {nb}") if la[0] > lb[0] else (f"dark {na}", f"light {nb}")
        else:
            ta, tb = na, nb
        # short and unmistakable in table headers: "Team A (light orange)" / "Team B (dark orange)"
        return {"A": f"Team A ({ta})", "B": f"Team B ({tb})"}
    return {"A": f"Team {na}", "B": f"Team {nb}"}


def is_placeholder_name(name: Any) -> bool:
    return not isinstance(name, str) or name.strip().lower() in PLACEHOLDER_NAMES


# ============================================================================ binding I/O
def valid(b: Any) -> bool:
    return (isinstance(b, dict) and all(isinstance(b.get(s), dict) and _hex_rgb(b[s].get("color", "")) for s in SIDES))


def _key(obj: Any) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def samples_key(raw: dict) -> str | None:
    """Fingerprint of the user-picked players (only when both teams have samples)."""
    smp = raw.get("team_samples") or {}
    if not (smp.get("A") and smp.get("B")):
        return None
    return _key({k: smp.get(k) for k in ("A", "B")})


def kit_key(raw: dict) -> str | None:
    """Fingerprint of user-set kit colours (None when they are automatic / placeholders)."""
    from replay.metadata import metadata_from_dict
    from replay.teams import colors_are_user_set
    try:
        if not colors_are_user_set(metadata_from_dict(raw)):
            return None
    except Exception:  # noqa: BLE001
        return None
    jc = raw.get("jersey_colors") or {}
    return _key({s: str((jc.get(s) or {}).get("primary", "")).lower() for s in SIDES})


def location(out: Path, meta_path: Path | None) -> tuple[Path, bool]:
    """(file, is_meta_json): the meta.json in use, else the project's meta.json next to results/, else
    <out>/team_binding.json."""
    if meta_path is not None:
        return Path(meta_path), True
    out = Path(out)
    if out.name == "results" and (out.parent / "meta.json").exists():
        return out.parent / "meta.json", True
    return out / FILENAME, False


def _read(path: Path) -> dict:
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def _write_json(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def load(out: Path, meta_path: Path | None) -> dict | None:
    path, is_meta = location(out, meta_path)
    d = _read(path)
    b = d.get(FIELD) if is_meta else d
    return copy.deepcopy(b) if valid(b) else None


def save(out: Path, meta_path: Path | None, binding: dict, swap_kit_colours: bool = False) -> bool:
    """Write the binding (only the ``team_binding`` field of meta.json - user fields stay). Returns True when the
    file changed. ``swap_kit_colours``: the sides were exchanged by the swap setting - user-set kit colours
    (team_colors_auto false) belong to the players, so they are exchanged too."""
    path, is_meta = location(out, meta_path)
    if not is_meta:
        if _read(path) == binding:
            return False
        _write_json(path, binding)
        return True
    doc = _read(path)
    if not doc and path.exists():
        return False  # unreadable meta.json: never overwrite the user's file
    new = dict(doc)
    if swap_kit_colours and new.get("team_colors_auto") is False and isinstance(new.get("jersey_colors"), dict):
        jc = dict(new["jersey_colors"])
        jc["home"], jc["away"] = jc.get("away", {}), jc.get("home", {})
        new["jersey_colors"] = jc
        binding = {**binding, "kit_key": kit_key(new)}
    new[FIELD] = binding
    if new == doc:
        return False
    _write_json(path, new)
    return True


# ============================================================================ signatures + matching
def _signature(teams, label: str) -> dict | None:
    if label not in teams.colors and label not in teams.centroids:
        return None
    color = teams.colors.get(label)
    lab = teams.centroids.get(label)
    if color is None and lab is not None:
        from replay.teams import lab_to_hex
        color = lab_to_hex(np.asarray(lab, float))
    if lab is None and color is not None:
        lab_arr = hex_to_lab(color)
        lab = None if lab_arr is None else np.round(lab_arr, 1).tolist()
    sig = {"color": color, "lab": [round(float(v), 1) for v in lab] if lab is not None else None,
           "hist": {}, "colour_name": lab_colour_name(lab) if lab is not None else colour_name(color)}
    if teams.space and label in teams.hists:
        sig["hist"][teams.space] = [round(float(v), 5) for v in teams.hists[label]]
    return sig


def _hellinger(p, q) -> float:
    p, q = np.clip(np.asarray(p, float), 0, None), np.clip(np.asarray(q, float), 0, None)
    if p.shape != q.shape or p.sum() <= 0 or q.sum() <= 0:
        return np.nan
    p, q = p / p.sum(), q / q.sum()
    return float(np.sqrt(max(0.0, 1.0 - np.sqrt(p * q).sum())))


def sig_cost(stored: dict, new: dict) -> float:
    """Dissimilarity of two colour signatures (about 0..2): histogram Hellinger distance in a shared descriptor
    space plus the Lab hue/lightness difference of the representative colours."""
    from replay.teams import _color_cost
    cost, n = 0.0, 0
    for space, h in (new.get("hist") or {}).items():
        hs = (stored.get("hist") or {}).get(space)
        if hs is not None:
            d = _hellinger(hs, h)
            if np.isfinite(d):
                cost += 2.0 * d
                n += 1
                break
    la, lb = stored.get("lab"), new.get("lab")
    if la is None and stored.get("color"):
        la = hex_to_lab(stored["color"])
    if lb is None and new.get("color"):
        lb = hex_to_lab(new["color"])
    if la is not None and lb is not None:
        cost += _color_cost(np.asarray(la, float), np.asarray(lb, float)) / 180.0
        n += 1
    return cost if n else 1.0


def match(binding: dict, new: dict[str, dict]) -> tuple[bool, float, float]:
    """(crossed, straight cost, crossed cost): crossed = new label A looks like the stored away team."""
    h, a = binding["home"], binding["away"]
    if "A" in new and "B" in new:
        straight = sig_cost(h, new["A"]) + sig_cost(a, new["B"])
        crossed = sig_cost(h, new["B"]) + sig_cost(a, new["A"])
    elif "A" in new:  # one team found: which stored team is it?
        straight, crossed = sig_cost(h, new["A"]), sig_cost(a, new["A"])
    elif "B" in new:
        straight, crossed = sig_cost(a, new["B"]), sig_cost(h, new["B"])
    else:
        return False, 0.0, 0.0
    return crossed < straight, straight, crossed


def swap_labels(teams):
    """Exchange A and B in a TeamResult completely (players, colours, signatures)."""
    sw = {"A": "B", "B": "A"}
    teams.track_team = {t: sw.get(v, v) for t, v in teams.track_team.items()}
    teams.centroids = {sw.get(k, k): v for k, v in teams.centroids.items()}
    teams.colors = {sw.get(k, k): v for k, v in teams.colors.items()}
    teams.hists = {sw.get(k, k): v for k, v in teams.hists.items()}
    if getattr(teams, "det_label", None) is not None:
        teams.det_label = np.array([sw.get(v, v) for v in teams.det_label], dtype=object)
    if getattr(teams, "cluster_roles", None):
        teams.cluster_roles = [sw.get(v, v) for v in teams.cluster_roles]
    return teams


# ============================================================================ score-bar cue (no OCR)
def scorebar_runs(video: str | Path, box_rel: list[float], n_frames: int = 14) -> list[dict]:
    """Coloured static segments of the broadcast score bar row: [{"x": centre (0..1), "w": width, "lab": [...]}].

    The bar row is the clock's row (found by the preliminary analysis, replay/inplay.py); pixels that stay the same
    over frames sampled across the video are overlay, everything else is scene. Score bars usually list the home
    team first (left); when the team boxes carry the kit colours this tells home from away."""
    import cv2
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return []
    try:
        nf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if nf < 10 or W < 50 or H < 50:
            return []
        y0, y1 = float(box_rel[1]), float(box_rel[3])
        hh = max(y1 - y0, 0.01)
        ya, yb = max(0, int((y0 - 0.6 * hh) * H)), min(H, int(np.ceil((y1 + 0.6 * hh) * H)))
        if yb - ya < 3:
            return []
        bands = []
        for f in np.linspace(0.05 * nf, 0.95 * nf, n_frames).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
            ok, img = cap.read()
            if ok and img is not None and img.shape[0] == H:
                bands.append(img[ya:yb])
    finally:
        cap.release()
    if len(bands) < 5:
        return []
    from replay.team_appearance import lab_float
    L = np.stack([lab_float(b) for b in bands])  # (n, h, W, 3)
    med = np.median(L, axis=0)
    static = (np.abs(L - med[None]).max(axis=-1) <= 6.0).mean(axis=0) >= 0.8
    chroma = np.hypot(med[..., 1], med[..., 2])
    colourful = static & (chroma >= 18) & (med[..., 0] >= 15)
    col = colourful.mean(axis=0) >= 0.35
    from replay.teams import _color_cost
    segs, x = [], 0  # [x0, x1) runs of coloured columns; pieces of one box (split by text/compression) merge
    while x < W:
        if not col[x]:
            x += 1
            continue
        x2 = x
        while x2 < W and col[x2]:
            x2 += 1
        lab = med[:, x:x2][colourful[:, x:x2]].mean(axis=0)
        if segs and x - segs[-1][1] <= max(3, 0.03 * W) and _color_cost(segs[-1][2], lab) <= 20:
            a = segs[-1][0]
            segs[-1] = [a, x2, med[:, a:x2][colourful[:, a:x2]].mean(axis=0)]
        else:
            segs.append([x, x2, lab])
        x = x2
    return [{"x": (a + b) / 2 / W, "w": (b - a) / W, "lab": [round(float(v), 1) for v in lab]}
            for a, b, lab in segs if b - a >= max(4, 0.012 * W)]


def scorebar_home(runs: list[dict], team_labs: dict[str, Any], max_cost: float = 30.0) -> str | None:
    """Label ("A"/"B") of the team whose colour box is further left on the score bar, or None when the bar does not
    show both team colours unambiguously."""
    from replay.teams import _color_cost
    if "A" not in team_labs or "B" not in team_labs or len(runs) < 2:
        return None
    la, lb = (np.asarray(team_labs[k], float) for k in ("A", "B"))
    if min(np.hypot(la[1], la[2]), np.hypot(lb[1], lb[2])) < 10 or _color_cost(la, lb) < 60:
        return None  # white/black kits or similar hues: a coloured box would not tell them apart
    best = {}
    for k, lab in (("A", la), ("B", lb)):
        c = [(_color_cost(np.asarray(r["lab"], float), lab), i) for i, r in enumerate(runs)]
        ok = [ci for ci in c if ci[0] <= max_cost]
        if not ok:
            return None
        best[k] = min(ok)[1]
    if best["A"] == best["B"]:
        return None
    # every run must clearly belong to one team (a third coloured box of a team colour = unclear layout)
    return "A" if runs[best["A"]]["x"] < runs[best["B"]]["x"] else "B"


def _scorebar_guess(out: Path, video: Path | None, teams) -> str | None:
    if video is None or not Path(video).exists():
        return None
    ip = _read(Path(out) / "inplay.json")
    clock = ip.get("clock") or {}
    if not clock.get("found") or not clock.get("box_rel"):
        return None
    try:
        runs = scorebar_runs(video, clock["box_rel"])
    except Exception as e:  # noqa: BLE001 - only a hint
        print("team binding: score bar check failed:", e)
        return None
    labs = {k: (teams.centroids.get(k) if teams.centroids.get(k) is not None else hex_to_lab(teams.colors.get(k, "")))
            for k in ("A", "B")}
    return scorebar_home(runs, {k: v for k, v in labs.items() if v is not None})


# ============================================================================ main entry: bind a classification
def _new_binding(teams, swap_flag: bool, source: str, confirmed: bool, raw: dict, note: str = "") -> dict:
    b = {"version": VERSION, "home": _signature(teams, "A"), "away": _signature(teams, "B"), "confirmed": confirmed,
         "source": source, "swap_flag": bool(swap_flag), "samples_key": samples_key(raw), "kit_key": kit_key(raw)}
    if note:
        b["note"] = note
    return b


def bind(out: Path, meta, teams, swap_teams: bool = False, video: Path | None = None):
    """Make the A/B labels of ``teams`` (a teams.TeamResult) stable and correct; returns (teams, meta, binding).

    The returned meta is a copy whose home/away names are the bound team names and whose ``team_extra`` carries
    side/confirmation/colour-name info for the outputs. The binding is persisted (see module docstring)."""
    raw = meta.raw or {}
    meta_path = getattr(meta, "path", None)
    swap_teams = bool(swap_teams)
    have = {k for k in ("A", "B") if k in teams.colors or k in teams.centroids}
    b = load(out, meta_path)
    if b is not None and any(str(n).startswith("team samples ignored") for n in getattr(teams, "notes", [])):
        b = None  # the stored binding came from samples that match no kit on court (teams.SEED_CHECK): rebind
    s_key, k_key = samples_key(raw), kit_key(raw)
    swapped_by_toggle = False

    if b is None:
        if have != {"A", "B"}:  # nothing to bind yet: keep the old behaviour
            if swap_teams:
                swap_labels(teams)
            return teams, _bound_meta(meta, None, teams), None
        if teams.seeded:
            source, confirmed, note = ("samples" if s_key else "kit_colours"), True, ""
        elif swap_teams:
            source, confirmed, note = "swap", True, ""
        else:
            home = _scorebar_guess(out, video, teams)
            if home == "B":
                swap_labels(teams)
            source = "scorebar" if home else "auto"
            note = ("home = team listed first on the score bar (its box has that kit colour); please confirm"
                    if home else "home/away not known yet: teams are named after their jersey colour until confirmed")
            confirmed = False
        if swap_teams:
            swap_labels(teams)
        b = _new_binding(teams, swap_teams, source, confirmed, raw, note)
    else:
        user_samples = bool(s_key) and s_key != b.get("samples_key") and teams.seeded
        user_kit = bool(k_key) and k_key != b.get("kit_key") and teams.seeded and not s_key
        if user_samples or user_kit:  # the user just told us: their picked players / kit colours win
            b = _new_binding(teams, swap_teams, "samples" if user_samples else "kit_colours", True, raw)
        else:
            new = {k: sig for k in ("A", "B") if (sig := _signature(teams, k)) is not None}
            crossed, c_straight, c_crossed = match(b, new)
            if crossed:
                swap_labels(teams)
            if new and abs(c_straight - c_crossed) <= AMBIGUOUS_REL * max(c_straight, c_crossed, 1e-6):
                teams.notes.append("team colours are too similar to keep the teams apart reliably")
            # complete the stored signatures with the current descriptor space (sides never change here)
            for side, lab in (("home", "A"), ("away", "B")):
                sig = _signature(teams, lab)
                if sig and teams.space and teams.space in sig["hist"] and teams.space not in b[side].get("hist", {}):
                    b[side].setdefault("hist", {})[teams.space] = sig["hist"][teams.space]
            if swap_teams != bool(b.get("swap_flag", False)):  # the user toggled "swap teams" since the last build
                swap_labels(teams)
                b["home"], b["away"] = b["away"], b["home"]
                b.update(confirmed=True, source="swap", swap_flag=swap_teams)
                b.pop("note", None)
                swapped_by_toggle = True
        b["samples_key"], b["kit_key"] = s_key, (b.get("kit_key") if swapped_by_toggle else k_key)

    try:
        save(out, meta_path, b, swap_kit_colours=swapped_by_toggle)
    except OSError as e:
        print("team binding: could not save:", e)
    # display colours: the bound (stable) colours, follow the players
    teams.colors = {**teams.colors, "A": b["home"]["color"], "B": b["away"]["color"]}
    return teams, _bound_meta(meta, b, teams), b


# ============================================================================ names for every consumer
def _words(binding: dict) -> tuple[str, str]:
    return tuple(binding[s].get("colour_name") or colour_name(binding[s]["color"]) for s in SIDES)  # type: ignore


def resolved_names(raw: dict, binding: dict | None) -> dict[str, str]:
    """Team names per label. Confirmed binding: the meta names (or colour names when those are placeholders).
    Unconfirmed: colour names ("Team orange"), so a guess never shows as a wrong real name."""
    home, away = raw.get("home_team"), raw.get("away_team")
    if not valid(binding):
        return {"A": home if not is_placeholder_name(home) else "Home",
                "B": away if not is_placeholder_name(away) else "Away"}
    cn = colour_team_names(binding["home"]["color"], binding["away"]["color"], _words(binding))
    if not binding.get("confirmed"):
        return cn
    return {"A": home.strip() if not is_placeholder_name(home) else cn["A"],
            "B": away.strip() if not is_placeholder_name(away) else cn["B"]}


def team_extra(binding: dict | None) -> dict[str, dict]:
    if not valid(binding):
        return {}
    conf = bool(binding.get("confirmed"))
    return {LABEL_OF[s]: {"side": s, "side_confirmed": conf, "colour_name": binding[s].get("colour_name") or
                          colour_name(binding[s]["color"]), "binding_source": binding.get("source", "auto")}
            for s in SIDES}


def _bound_meta(meta, binding: dict | None, teams):
    from replay.metadata import parse_color
    names = resolved_names(meta.raw or {}, binding)
    home, away = dataclasses.replace(meta.home, name=names["A"]), dataclasses.replace(meta.away, name=names["B"])
    if valid(binding):
        try:
            home = dataclasses.replace(home, primary=parse_color(binding["home"]["color"]))
            away = dataclasses.replace(away, primary=parse_color(binding["away"]["color"]))
        except ValueError:
            pass
    return dataclasses.replace(meta, home=home, away=away, team_extra=team_extra(binding))


def team_overview(raw: dict, binding: dict | None = None) -> dict[str, dict]:
    """For the GUI: {"A": {"name", "color", "colour_name", "side", "confirmed"}, "B": ...} from meta.json content."""
    b = binding if binding is not None else raw.get(FIELD)
    names = resolved_names(raw, b if valid(b) else None)
    out = {}
    for lab, side in (("A", "home"), ("B", "away")):
        sig = b[side] if valid(b) else {}
        jc = ((raw.get("jersey_colors") or {}).get(side) or {}).get("primary")
        color = sig.get("color") or jc or ("#f5f5f5" if lab == "A" else "#1f2a5a")
        out[lab] = {"name": names[lab], "color": color, "colour_name": sig.get("colour_name") or colour_name(color),
                    "side": side,
                    "confirmed": bool(valid(b) and b.get("confirmed")), "detected": valid(b)}
    return out


def side_text(team: dict, label: str | None = None) -> str:
    """'home' / 'away' when confirmed (or for outputs from before the binding), else 'home/away not confirmed'."""
    if team.get("side_confirmed", team.get("confirmed")) is False:
        return "home/away not confirmed"
    return str(team.get("side") or SIDE_OF.get(label or team.get("label") or "", ""))


def apply_user_choice(raw: dict, rows: list[dict], swap_flag: bool) -> dict:
    """Overview edit. ``rows`` = [{"label": current label "A"/"B", "name": str, "side": "home"|"away",
    "color": "#rrggbb"}] for both detected teams. Returns the new meta dict (binding confirmed by the user)."""
    m = copy.deepcopy(raw)
    b = m.get(FIELD)
    if not valid(b):
        raise ValueError("no detected teams yet")
    if len(rows) != 2 or {r["side"] for r in rows} != set(SIDES):
        raise ValueError("one team must be home and the other away")
    old = {"A": b["home"], "B": b["away"]}
    new_b = dict(b)
    for r in rows:
        sig = dict(old[r["label"]])
        if r.get("color") and _hex_rgb(r["color"]) and r["color"].lower() != str(sig.get("color", "")).lower():
            sig["color"] = r["color"]  # display colour only; matching keeps using the measured histogram/Lab
        new_b[r["side"]] = sig
        m["home_team" if r["side"] == "home" else "away_team"] = (r.get("name") or "").strip() or \
            f"Team {sig.get('colour_name') or colour_name(sig['color'])}"
    new_b.update(confirmed=True, source="user", swap_flag=bool(swap_flag))
    new_b.pop("note", None)
    m[FIELD] = new_b
    # informational kit colours follow the binding; colours stay automatic (the binding carries them)
    jc = dict(m.get("jersey_colors") or {})
    for side in SIDES:
        jc[side] = {**(jc.get(side) or {}), "primary": new_b[side]["color"]}
    m["jersey_colors"] = jc
    m["team_colors_auto"] = True
    new_b["kit_key"] = kit_key(m)
    return m


def overview_save(meta: dict, rows: list[dict], dirty: bool, confirmed: bool, swap_flag: bool) -> tuple[dict, bool]:
    """The Overview tab's Save for the team rows. Returns (new meta dict, clear the 2D view's A/B colour overrides).

    Any edit of the team rows (name, colour, Home/Away) or ticking "Confirmed" confirms the binding: the user has
    looked at which jersey colour is which team. Without detected teams only the names are stored."""
    m = dict(meta)
    if not valid(m.get(FIELD)):
        by_side = {r["side"]: (r.get("name") or "").strip() for r in rows if r.get("side") in SIDES}
        if by_side:
            m["home_team"], m["away_team"] = by_side.get("home", m.get("home_team")), by_side.get("away", m.get("away_team"))
        return m, False
    old = team_overview(meta)
    if not dirty and not (confirmed and not old["A"]["confirmed"]):
        return m, False
    m = apply_user_choice(m, rows, swap_flag)
    clear = any(r["side"] != old[r["label"]]["side"] or str(r.get("color", "")).lower() != old[r["label"]]["color"].lower()
                for r in rows)
    return m, clear
