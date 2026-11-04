"""Corrections ledger: the coach's manual fixes, kept as hard constraints that survive every rebuild.

File ``<project>/corrections.json`` (next to calibration.json / meta.json), written only by the GUI / the coach,
never by the analysis: a rebuild reads it and leaves it byte-for-byte unchanged.

Schema (``FORMAT``, ``version`` 1)::

    {"format": "advanced-analytist/corrections@1", "version": 1,
     "ops": [{"kind": ..., "t0": s, "t1": s, ...}, ...]}

Every op has ``kind``, ``t0`` and ``t1`` (seconds of video; ``t1`` defaults to ``t0``) and may carry ``id`` and
``note`` (free text, stays local). Ops that point at a person carry ``at`` = [x, y] court metres at ``t0`` (the
detection nearest to it, within ``PICK_RADIUS_M``, is the person).

identity (``apply_identity``, hard constraints of the tracklet linker, tracklets.link_tracks):
  * ``split``  at=[x,y]: the person at ``at`` at ``t0`` is NOT the person their track showed just before ``t0``
    (an ID switch): the tracklet is cut there and the two sides are never linked again.
  * ``merge``  at=[x,y] (at t0), to=[x,y] (at t1): the two are the same person (forced link).
  * ``assign`` at=[x,y], player=<label>: the person at ``at`` during [t0, t1] is ``player`` (a roster slot or
    jersey number, NEVER a name: this file may be read by tools that talk to AI services). Pieces with the same
    label are one identity; pieces with different labels are never linked.
labels (stored and validated, not applied by the analysis yet):
  * ``team`` at, team="A"|"B"; ``role`` at, role="player"|"referee"|"coach"|"bench"|"other"; ``hide`` at.
events (``apply_events``, possession model v6 in game_state.py):
  * ``make``   team, points (1..3), optional side "L"|"R": a made basket at ``t0`` (a possession boundary; it
    overrides the team / points of a detected make within ``MAKE_MATCH_S``).
  * ``anchor`` team: team T has the ball during [t0, t1] (both ends are possession boundaries).

Nothing here touches the analysis unless ``Config.corrections`` is on (env REPLAY_CORRECTIONS=1/0 overrides);
with no ops every hook is a no-op.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

FILENAME = "corrections.json"
FORMAT = "advanced-analytist/corrections@1"
VERSION = 1
KINDS = ("split", "merge", "assign", "team", "role", "hide", "anchor", "make")
IDENTITY_KINDS = ("split", "merge", "assign")
EVENT_KINDS = ("make", "anchor")
TEAMS = ("A", "B")
ROLES = ("player", "referee", "coach", "bench", "other")
PICK_RADIUS_M = 1.5  # an op's ``at`` picks the nearest detection within this distance
PICK_WINDOW_S = 0.5  # ... in the analysed sample nearest t (searching up to this far for a sample with people)
SPLIT_BACK_S = 1.0  # split: the "before" side is searched up to this long before t0 when the tracklet starts at t0
MAKE_MATCH_S = 2.0  # a user make overrides a detected make this close in time


class CorrectionsError(ValueError):
    pass


# ----------------------------------------------------------------------------------------------- schema / file
def enabled(cfg: Any = None) -> bool:
    """Config.corrections (default False); env REPLAY_CORRECTIONS=1/0 overrides."""
    env = os.environ.get("REPLAY_CORRECTIONS")
    if env in ("0", "1"):
        return env == "1"
    return bool(getattr(cfg, "corrections", False))


def path_for(project_dir: Path) -> Path:
    return Path(project_dir) / FILENAME


def _num(v: Any, name: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v):
        raise CorrectionsError(f"{name} must be a finite number, got {v!r}")
    return float(v)


def _xy(v: Any, name: str) -> list[float]:
    if not isinstance(v, (list, tuple)) or len(v) != 2:
        raise CorrectionsError(f"{name} must be [x, y] court metres, got {v!r}")
    return [_num(v[0], name), _num(v[1], name)]


def normalize_op(op: dict) -> dict:
    """A validated copy of one op (unknown keys are kept as they are, so newer files survive a round trip)."""
    if not isinstance(op, dict):
        raise CorrectionsError(f"an op must be an object, got {type(op).__name__}")
    kind = op.get("kind")
    if kind not in KINDS:
        raise CorrectionsError(f"unknown op kind {kind!r} (one of {', '.join(KINDS)})")
    out = dict(op)
    out["t0"] = _num(op.get("t0"), "t0")
    out["t1"] = _num(op.get("t1", out["t0"]), "t1")
    if out["t1"] < out["t0"]:
        raise CorrectionsError(f"{kind}: t1 {out['t1']} is before t0 {out['t0']}")
    if kind in ("split", "merge", "assign", "team", "role", "hide"):
        out["at"] = _xy(op.get("at"), f"{kind}.at")
    if kind == "merge":
        out["to"] = _xy(op.get("to"), "merge.to")
    if kind == "assign":
        p = op.get("player")
        if isinstance(p, bool) or not isinstance(p, (str, int)) or str(p).strip() == "":
            raise CorrectionsError(f"assign.player must be a slot / jersey label, got {p!r}")
    if kind in ("team", "make", "anchor") and op.get("team") not in TEAMS:
        raise CorrectionsError(f"{kind}.team must be 'A' or 'B', got {op.get('team')!r}")
    if kind == "role" and op.get("role") not in ROLES:
        raise CorrectionsError(f"role.role must be one of {', '.join(ROLES)}, got {op.get('role')!r}")
    if kind == "make":
        pts = op.get("points", 2)
        if isinstance(pts, bool) or pts not in (1, 2, 3):
            raise CorrectionsError(f"make.points must be 1, 2 or 3, got {pts!r}")
        out["points"] = int(pts)
        if op.get("side") is not None and op.get("side") not in ("L", "R"):
            raise CorrectionsError(f"make.side must be 'L' or 'R', got {op.get('side')!r}")
    return out


def normalize(doc: dict | None) -> dict:
    """A validated ledger document (``None`` = empty ledger). Op order is kept (it is the coach's edit order)."""
    if doc is None:
        return {"format": FORMAT, "version": VERSION, "ops": []}
    if not isinstance(doc, dict):
        raise CorrectionsError("the ledger must be a JSON object")
    v = doc.get("version", VERSION)
    if not isinstance(v, int) or v > VERSION:
        raise CorrectionsError(f"ledger version {v!r} is newer than this app ({VERSION})")
    ops = doc.get("ops", [])
    if not isinstance(ops, list):
        raise CorrectionsError("ops must be a list")
    out = {k: val for k, val in doc.items() if k not in ("format", "version", "ops")}
    out.update(format=FORMAT, version=VERSION, ops=[normalize_op(o) for o in ops])
    return out


def dumps(doc: dict) -> str:
    """Canonical text: the same ledger always gives the same bytes."""
    return json.dumps(normalize(doc), ensure_ascii=False, sort_keys=True, indent=1) + "\n"


def load(project_dir: Path) -> dict:
    """The project's ledger (empty when there is none). Raises CorrectionsError on a broken file: corrections
    are never dropped silently."""
    p = path_for(project_dir)
    if not p.exists():
        return normalize(None)
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except ValueError as e:
        raise CorrectionsError(f"{p.name} is not valid JSON: {e}") from e
    return normalize(doc)


def save(project_dir: Path, doc: dict) -> Path:
    """Write the ledger atomically; an unchanged ledger is not rewritten (the file stays byte-for-byte equal)."""
    p = path_for(project_dir)
    text = dumps(doc)
    try:
        if p.read_text(encoding="utf-8") == text:
            return p
    except OSError:
        pass
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, p)
    return p


def load_if_enabled(project_dir: Path, cfg: Any = None) -> dict | None:
    """The ledger when corrections are enabled and it has ops, else None (= every hook is a no-op). A broken
    ledger is reported and ignored here: the analysis must not fail on it (the GUI shows the error)."""
    if not enabled(cfg):
        return None
    try:
        doc = load(project_dir)
    except CorrectionsError as e:
        print("corrections:", e)
        return None
    return doc if doc["ops"] else None


# ----------------------------------------------------------------------------------------------- identity
@dataclass
class IdentityConstraints:
    """Ledger identity ops resolved to the linker's detections (indices into link_tracks' arrays).

    picks: per op index, the picked detection(s) {"at": det, "to": det, "before": det}; unresolved ops are left out
    (no detection near ``at``) and counted in ``unresolved``."""
    ops: list[dict]
    t: np.ndarray  # time (s) per analysed sample
    picks: dict[int, dict[str, int]] = field(default_factory=dict)
    unresolved: list[int] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.picks)


def _sample_dets(k: np.ndarray, t: np.ndarray, t0: float, window_s: float) -> list[int]:
    """Samples present in ``k`` ordered by distance in time to t0 (within window_s)."""
    ks = np.unique(k)
    if not len(ks):
        return []
    tt = t[ks]
    near = np.abs(tt - t0) <= window_s + 1e-9
    cand = ks[near]
    return [int(c) for c in cand[np.argsort(np.abs(t[cand] - t0), kind="stable")]]


def _pick(k: np.ndarray, xy: np.ndarray, t: np.ndarray, t0: float, at: list[float], window_s: float = PICK_WINDOW_S,
          radius: float = PICK_RADIUS_M, exclude: set | None = None) -> int | None:
    p = np.asarray(at, float)
    for s in _sample_dets(k, t, t0, window_s):
        idx = np.flatnonzero(k == s)
        if exclude:
            idx = np.array([i for i in idx if int(i) not in exclude], int)
        if not len(idx):
            continue
        d = np.linalg.norm(xy[idx] - p, axis=1)
        d = np.where(np.isfinite(d), d, np.inf)
        j = int(np.argmin(d))
        if d[j] <= radius:
            return int(idx[j])
    return None


def resolve_identity(doc: dict | None, t: np.ndarray, k: np.ndarray, xy: np.ndarray) -> IdentityConstraints | None:
    """Pick the detections the ledger's identity ops point at. t: time per sample; k: sample per detection;
    xy: court position per detection (NaN = none). None when there is nothing to apply."""
    if not doc:
        return None
    ops = [o for o in doc.get("ops", []) if o.get("kind") in IDENTITY_KINDS]
    if not ops:
        return None
    t = np.asarray(t, float)
    k = np.asarray(k, int)
    xy = np.asarray(xy, float).reshape(len(k), 2)
    ic = IdentityConstraints(ops=ops, t=t)
    for i, op in enumerate(ops):
        a = _pick(k, xy, t, op["t0"], op["at"])
        if a is None:
            ic.unresolved.append(i)
            continue
        pick = {"at": a}
        if op["kind"] == "merge":
            b = _pick(k, xy, t, op["t1"], op["to"])
            if b is None or b == a:
                ic.unresolved.append(i)
                continue
            pick["to"] = b
        ic.picks[i] = pick
    return ic


def sample_times(n_samples: int, fps: float) -> np.ndarray:
    return np.arange(int(n_samples)) / float(fps)


def apply_identity(ic: IdentityConstraints | None, pieces: list[np.ndarray], k: np.ndarray, xy: np.ndarray
                   ) -> tuple[list[np.ndarray], dict[int, frozenset]]:
    """Cut the raw tracklets (detection index arrays, sorted by sample) where ``split`` ops say so and label the
    detections every identity op points at. -> (pieces, labels {detection: frozenset of labels}).

    Labels: ``p:<player>`` (assign), ``m:<i>`` (merge), ``sL:<i>`` / ``sR:<i>`` (the two sides of split i).
    :func:`conflict` / :func:`forced` compare the label sets of two pieces. No ops: pieces unchanged, no labels."""
    if not ic:
        return pieces, {}
    k = np.asarray(k, int)
    xy = np.asarray(xy, float)
    pieces = [np.asarray(p, int) for p in pieces]
    piece_of = {int(d): j for j, p in enumerate(pieces) for d in p}
    labels: dict[int, set] = {}

    def tag(dets, lab: str) -> None:
        for d in dets:
            labels.setdefault(int(d), set()).add(lab)

    def in_piece(d: int) -> int | None:
        """d itself, or (d was left out of every tracklet, e.g. a second box with the same ByteTrack id in one
        sample) the nearest detection of the same sample that is in one, within PICK_RADIUS_M."""
        if d in piece_of:
            return d
        same = [int(e) for e in np.flatnonzero(k == k[d]) if int(e) in piece_of]
        if not same:
            return None
        dist = np.linalg.norm(xy[same] - xy[d], axis=1)
        dist = np.where(np.isfinite(dist), dist, np.inf)
        j = int(np.argmin(dist))
        return same[j] if dist[j] <= PICK_RADIUS_M else None

    for i in list(ic.picks):  # every pick must lie in a tracklet
        new = {key: in_piece(d) for key, d in ic.picks[i].items()}
        if any(v is None for v in new.values()) or ("to" in new and new["to"] == new["at"]):
            del ic.picks[i]
            ic.unresolved.append(i)
        else:
            ic.picks[i] = new
    ic.unresolved.sort()

    # splits first (they change the pieces the other ops tag)
    for i, op in enumerate(ic.ops):
        if op["kind"] != "split" or i not in ic.picks:
            continue
        d = ic.picks[i]["at"]
        j = piece_of[d]
        p = pieces[j]
        kd = k[d]
        left, right = p[k[p] < kd], p[k[p] >= kd]
        if not len(left):  # the tracklet starts here: the "before" side is the nearest person just before t0
            prev = np.flatnonzero((k < kd) & (ic.t[np.clip(k, 0, len(ic.t) - 1)] >= ic.t[kd] - SPLIT_BACK_S - 1e-9))
            prev = np.array([e for e in prev if int(e) in piece_of], int)  # (only detections in a tracklet)
            b = None
            if len(prev):
                dist = np.linalg.norm(xy[prev] - xy[d], axis=1)
                dist = np.where(np.isfinite(dist), dist, np.inf)
                last = k[prev].max()
                on_last = dist + 1e3 * (k[prev] != last)  # the latest sample first
                j2 = int(np.argmin(on_last))
                if dist[j2] <= 2 * PICK_RADIUS_M:
                    b = int(prev[j2])
            if b is None:
                continue
            ic.picks[i]["before"] = b
            pb = pieces[piece_of[b]]
            left = pb[k[pb] < kd]
        else:
            pieces[j] = left
            pieces.append(right)
            for dd in right:
                piece_of[int(dd)] = len(pieces) - 1
            ic.picks[i]["before"] = int(left[-1])
        tag(left, f"sL:{i}")
        tag(right, f"sR:{i}")
    for i, op in enumerate(ic.ops):
        if i not in ic.picks or op["kind"] == "split":
            continue
        if op["kind"] == "merge":
            for key in ("at", "to"):
                tag(pieces[piece_of[ic.picks[i][key]]], f"m:{i}")
        elif op["kind"] == "assign":
            p = pieces[piece_of[ic.picks[i]["at"]]]
            tt = ic.t[k[p]]
            inside = p[(tt >= op["t0"] - 1e-6) & (tt <= op["t1"] + 1e-6)]
            tag(inside if len(inside) else [ic.picks[i]["at"]], f"p:{op['player']}")
    return pieces, {d: frozenset(v) for d, v in labels.items()}


def piece_labels(labels: dict[int, frozenset], det: np.ndarray) -> frozenset:
    if not labels:
        return frozenset()
    out: set = set()
    for d in det:
        s = labels.get(int(d))
        if s:
            out |= s
    return frozenset(out)


def conflict(a: frozenset, b: frozenset) -> bool:
    """Two label sets that must never be one identity: different players, or the two sides of one split."""
    if not a or not b:
        return False
    pa = {x for x in a if x.startswith("p:")}
    pb = {x for x in b if x.startswith("p:")}
    if pa and pb and (len(pa | pb) > 1):
        return True
    for x in a:
        if x.startswith("sL:") and "sR:" + x[3:] in b:
            return True
        if x.startswith("sR:") and "sL:" + x[3:] in b:
            return True
    return False


def conflict_any(sets: list[frozenset]) -> bool:
    """Any two of the label sets conflict (they must not all be one identity)."""
    acc: frozenset = frozenset()
    for s in sets:
        if conflict(acc, s):
            return True
        acc = acc | s
    return False


def forced(a: frozenset, b: frozenset) -> bool:
    """Two label sets that must be one identity (same player label or the two ends of one merge)."""
    if not a or not b or conflict(a, b):
        return False
    return any(x[:2] in ("p:", "m:") and x in b for x in a)


# ----------------------------------------------------------------------------------------------- events
def apply_events(doc: dict | None) -> dict | None:
    """Possession constraints for game_state.possessions_v6 from the ledger's event ops, or None when there are
    none: {"makes": [{t, team, points, side}], "anchors": [{t0, t1, team}], "boundaries": [t, ...]}.
    Makes are boundaries by themselves (v6 splits at every make); each anchor's ends are boundaries."""
    if not doc:
        return None
    makes, anchors, bounds = [], [], []
    for op in doc.get("ops", []):
        if op.get("kind") == "make":
            makes.append({"t": float(op["t0"]), "team": op["team"], "points": int(op.get("points", 2)),
                          "side": op.get("side")})
        elif op.get("kind") == "anchor":
            anchors.append({"t0": float(op["t0"]), "t1": float(op["t1"]), "team": op["team"]})
            bounds += [float(op["t0"]), float(op["t1"])]
    if not makes and not anchors:
        return None
    makes.sort(key=lambda m: m["t"])
    anchors.sort(key=lambda a: (a["t0"], a["t1"]))
    return {"makes": makes, "anchors": anchors, "boundaries": sorted(set(bounds))}
