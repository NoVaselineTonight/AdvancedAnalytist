"""Benchmark: rebuild pass 2 for every labelled test clip into a scratch copy and score it against labels.

    python -m replay benchmark [--clips c300,p4400] [--quick] [--jobs N] [--no-run] [--tag NAME]

Never writes into the original clip folders: each clip's pass-1 results (work/ + small JSON inputs), its calibration
and meta are copied to data/benchmark/runs/<clip>/ and pass 2 is rebuilt there with the CURRENT code and CURRENT
config defaults (plus data/benchmark/config.json overrides if present). Results: data/benchmark/results.md +
results.json, previous runs in data/benchmark/history/. Label format: data/benchmark/README.md.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import os
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent.parent
ROOT = Path(os.environ.get("AA_BENCH_ROOT") or CODE_ROOT)  # data root (a code snapshot runs with the real data)
BENCH = ROOT / "data" / "benchmark"
LABELS = BENCH / "labels"
RUNS = BENCH / "runs"
HISTORY = BENCH / "history"
CACHE = BENCH / "cache"
CACHED = ("jersey_numbers.npz", "team_appearance.npz")  # slow video-reading caches pass 2 keeps in work/

IOU_MIN = 0.5
COURT_MATCH_M = 1.5  # label without box: matched to an output position within this distance
QUICK_CLIPS = ("p4400", "c300")
TEAM_COLOURS = ("orange", "purple")


# ---------------------------------------------------------------------------------------------- clip registry
def clip_registry() -> dict[str, dict]:
    """name -> {src (pass-1 results folder with work/), calibration, meta (optional)}. data/benchmark/clips.json
    entries (same keys, paths relative to the project root) override/extend the built-in list."""
    reg: dict[str, dict] = {}
    p = ROOT / "data" / "benchmark" / "sources" / "p4400"  # own copy: the user's Replay folder can be deleted
    reg["p4400"] = {"src": p / "results", "calibration": p / "calibration.json", "meta": p / "meta.json"}
    reg["b3min"] = {"src": ROOT / "data/bench3min/out", "calibration": ROOT / "data/bench3min/clip.calibration.json"}
    for d in sorted((ROOT / "data" / "clips").glob("c*")):
        if d.is_dir():
            reg[d.name] = {"src": d / "out", "calibration": d / "clip.calibration.json"}
    for d in sorted((ROOT / "data" / "external").glob("*/seg_*")):
        if d.is_dir():
            reg[f"{d.parent.name}_{d.name}"] = {"src": d / "out", "calibration": d / "calibration.json"}
    extra = BENCH / "clips.json"
    if extra.exists():
        for k, v in json.loads(extra.read_text(encoding="utf-8")).items():
            reg[k] = {kk: (ROOT / vv if kk in ("src", "calibration", "meta") else vv) for kk, vv in v.items()}
    for v in reg.values():
        for k in ("src", "calibration", "meta"):
            if k in v and v[k] is not None:
                v[k] = Path(v[k])
    return reg


ALIASES = {"b3min": ["bench3min"], "p4400": ["clip-4400-40428b"]}  # other names labellers used for a clip
_NOT_LABELS = {"drafts", "check", "people_frames", "dense"}


def label_files(clip: str) -> dict[str, list[Path]]:
    """kind -> label files for the clip: labels/<kind>_<clip>.json first, then labeller subfolders
    (labels/<labeller>/<kind>_<clip>.json; later files win for the same frame)."""
    out: dict[str, list[Path]] = {}
    subs = sorted(d for d in LABELS.iterdir() if d.is_dir() and d.name not in _NOT_LABELS) if LABELS.exists() else []
    names = [clip] + ALIASES.get(clip, [])
    dense = [f for n in names for f in [LABELS / "dense" / f"{n}.json"] if f.exists()]
    if dense:
        out["dense"] = dense
    for kind in ("players", "calib", "scores"):
        fs = [f for n in names for f in [LABELS / f"{kind}_{n}.json"] + [d / f"{kind}_{n}.json" for d in subs]
              if f.exists()]
        if fs:
            out[kind] = fs
    return out


def load_labels(files: list[Path]) -> dict:
    """Merge label documents: frames by frame number (later file wins), calib keypoints/judgements and score
    events from the last file that has them."""
    doc: dict = {}
    frames: dict[int, dict] = {}
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        for fr in d.get("frames", []):
            frames[int(fr["frame"])] = fr
        doc.update({k: v for k, v in d.items() if k != "frames"})
    if frames:
        doc["frames"] = [frames[k] for k in sorted(frames)]
    doc["sources"] = [str(f.relative_to(LABELS)) for f in files]
    return doc


def ready(info: dict) -> bool:
    return (Path(info["src"]) / "work" / "DONE").exists() and Path(info["calibration"]).exists()


# ---------------------------------------------------------------------------------------------- scratch copy + run
_SKIP = {"tracking.csv", "tracking.json", "replay_data.json", "replay.html", "replay_2d.html", "report.md",
         "report.json", "team_binding.json", "check.mp4", "events.json", "game.json", "README_FOR_AI.md"}


def prepare_run(clip: str, info: dict, run: Path | None = None) -> Path:
    """Fresh run folder (default data/benchmark/runs/<clip>/) = copy of the clip's pass-1 work + inputs."""
    run = run or RUNS / clip
    if run.exists():
        shutil.rmtree(run)
    out = run / "out"
    out.mkdir(parents=True)
    src = Path(info["src"])
    shutil.copytree(src / "work", out / "work", ignore=shutil.ignore_patterns("*.tmp*"))
    for f in src.iterdir():
        if f.is_file() and f.name not in _SKIP and f.suffix in (".json",):
            shutil.copy2(f, out / f.name)
    for f in (CACHE / clip).glob("*.npz"):  # video-derived caches from earlier benchmark runs (validated by key)
        shutil.copy2(f, out / "work" / f.name)
    shutil.copy2(info["calibration"], run / "calibration.json")
    if info.get("meta") and Path(info["meta"]).exists():
        shutil.copy2(info["meta"], run / "meta.json")
    return run


def _worker(run: Path) -> None:
    """Runs in a subprocess: pass 2 on run/out with per-stage timing; saves run/timing.json and run/H.npz."""
    import numpy as np

    from replay import calibration as calib_mod
    from replay import postprocess as pp
    from replay.config import Config
    from replay.metadata import load_metadata

    timing: dict[str, float] = defaultdict(float)
    depth = [0]
    captured: dict = {}

    def timed(fn, name):
        def wrap(*a, **k):
            if depth[0]:
                return fn(*a, **k)
            depth[0] += 1
            t0 = time.perf_counter()
            try:
                return fn(*a, **k)
            finally:
                timing[name] += time.perf_counter() - t0
                depth[0] -= 1
        wrap.__wrapped__ = fn
        return wrap

    stage_of = {"track_people": "court_tracking", "filter_players": "roster", "appearance_for": "teams",
                "classify": "teams", "smooth_track": "smoothing", "limit_shown_rows": "caps",
                "consolidate_drop": "caps", "rim_tracks": "ball", "estimate_ball": "ball", "holders_from_rows": "ball",
                "boxes_by_frame": "ball", "detect_baskets": "scoring", "write_outputs": "export",
                "build_report": "report"}
    for name, obj in list(vars(pp).items()):  # every imported replay.* function (new hooks show under their name)
        if callable(obj) and not isinstance(obj, type) and getattr(obj, "__module__", "").startswith("replay.") \
                and obj.__module__ != pp.__name__ and not name.startswith("_"):
            setattr(pp, name, timed(obj, stage_of.get(name, name)))
    orig_resolve = calib_mod.resolve

    def resolve(*a, **k):
        res = orig_resolve(*a, **k)
        captured["H"] = res.H
        return res
    calib_mod.resolve = timed(resolve, "calibration")
    for modname, fname, stage in (("replay.jersey_numbers", "load_or_compute", "jersey_numbers"),
                                  ("replay.team_binding", "bind", "teams"),
                                  ("replay.game_state", "after_pass2", "game_state")):
        mod = __import__(modname, fromlist=[fname])
        if hasattr(mod, fname):
            setattr(mod, fname, timed(getattr(mod, fname), stage))

    if hasattr(pp, "estimate_ball"):
        timed_ball = pp.estimate_ball

        def capture_ball(*a, **k):
            res_ = timed_ball(*a, **k)
            captured["ball_args"], captured["ball_kwargs"], captured["ball_res"] = a, k, res_
            return res_
        pp.estimate_ball = capture_ball

    out = run / "out"
    cfg = Config()
    over = BENCH / "config.json"
    if over.exists():
        cfg = Config.from_dict(json.loads(over.read_text(encoding="utf-8")))
    sig = json.loads((out / "work" / "run.json").read_text(encoding="utf-8"))
    meta = run / "meta.json"
    t0 = time.perf_counter()
    pp.run_pass2(out, cfg, load_metadata(str(meta) if meta.exists() else None), run / "calibration.json",
                 Path(sig["video"]))
    total = time.perf_counter() - t0
    timing = {k: round(v, 2) for k, v in timing.items()}
    timing["other"] = round(max(0.0, total - sum(timing.values())), 2)
    timing["total"] = round(total, 2)
    (run / "timing.json").write_text(json.dumps(timing, indent=1), encoding="utf-8")
    H = captured.get("H") or []
    arr = np.full((len(H), 3, 3), np.nan)
    for i, h in enumerate(H):
        if h is not None:
            arr[i] = h
    np.savez_compressed(run / "H.npz", H=arr)
    if "ball_args" in captured and os.environ.get("AA_BENCH_NO_EXTRAP") != "1":
        try:
            t1 = time.perf_counter()
            res_b = ball_extrapolation(captured["ball_args"], captured["ball_kwargs"], captured["ball_res"],
                                       getattr(timed_ball, "__wrapped__", timed_ball))
            res_b["ball_extrap_s"] = round(time.perf_counter() - t1, 1)
            (run / "ball_extrap.json").write_text(json.dumps(res_b, indent=1), encoding="utf-8")
        except Exception as e:  # noqa: BLE001 - a metric must never break the run
            (run / "ball_extrap.json").write_text(json.dumps({"error": repr(e)}), encoding="utf-8")


def ball_extrapolation(args: tuple, kwargs: dict, full: list, estimate) -> dict:
    """Hide the ball detections in windows of 1/3/5/10 s (where the ball was detected >= 60 %), re-run the
    pipeline's own ball estimator and compare its court position with the full run's at hidden samples that had
    a detection."""
    import numpy as np

    from replay.bench_metrics import EXTRAP_S, ball_hide_windows

    n, ball_rows, _, _, fps = args[0], args[1], args[2], args[3], float(args[4])
    has = np.array([len(r) > 0 for r in ball_rows], bool)
    truth = {bf.i: (bf.x, bf.y) for bf in full if bf.x is not None and bf.box is not None}
    out: dict = {}
    for L_s in EXTRAP_S:
        L = max(1, int(round(L_s * fps)))
        wins = ball_hide_windows(has, L, max(2, int(round(fps))), int(round(3 * fps)))[:40]
        if not wins:
            out[f"ball_fill_{L_s}s_med"] = None
            continue
        rows = list(ball_rows)
        hidden = set()
        for a, b in wins:
            for i in range(a, b):
                rows[i] = np.zeros((0, ball_rows[i].shape[1] if ball_rows[i].ndim == 2 else 5))
                hidden.add(i)
        pred = estimate(n, rows, *args[2:], **kwargs)
        err, miss = [], 0
        for bf in pred:
            if bf.i in hidden and bf.i in truth:
                if bf.x is None:
                    miss += 1
                else:
                    err.append(math.dist((bf.x, bf.y), truth[bf.i]))
        out[f"ball_fill_{L_s}s_med"] = float(np.median(err)) if err else None
        out[f"ball_fill_{L_s}s_p90"] = float(np.percentile(err, 90)) if err else None
        out[f"ball_fill_{L_s}s_missing"] = miss / max(1, miss + len(err))
        out[f"ball_fill_{L_s}s_n"] = len(err)
    return out


def snapshot_code(dst: Path, src: Path | None = None) -> Path | None:
    """Frozen copy of replay/ (+ junctions to models/, tools/, data/) so other agents' edits during a long run cannot
    mix code versions between replicas. Returns the code root, or None (then the live tree is used)."""
    try:
        src = Path(src or CODE_ROOT)
        shutil.copytree(src / "replay", dst / "replay", ignore=shutil.ignore_patterns("__pycache__", "*.tmp*"))
        for name in ("models", "tools", "data"):  # code-relative paths (models, courtnet data)
            if (src / name).exists():
                r = subprocess.run(["cmd", "/c", "mklink", "/J", str(dst / name), str(src / name)],
                                   capture_output=True) if os.name == "nt" else None
                if r is None:
                    os.symlink(src / name, dst / name)
                elif r.returncode != 0:
                    raise OSError(r.stderr.decode(errors="replace"))
        return dst
    except OSError as e:
        print(f"  note: code snapshot failed ({e}); using the live tree")
        return None


def run_clip(clip: str, info: dict, log=print, run: Path | None = None, env_extra: dict | None = None) -> dict:
    run = prepare_run(clip, info, run)
    env = dict(os.environ, PYTHONUTF8="1", AA_BENCH_ROOT=str(ROOT), **(env_extra or {}))
    t0 = time.time()
    with open(run / "pass2.log", "w", encoding="utf-8") as fh:
        p = subprocess.run([sys.executable, "-m", "replay.benchmark", "_worker", str(run)],
                           cwd=str(env.get("AA_BENCH_CODE") or CODE_ROOT), env=env,
                           stdout=fh, stderr=subprocess.STDOUT)
    ok = p.returncode == 0 and (run / "out" / "tracking.csv").exists()
    if ok:
        (CACHE / clip).mkdir(parents=True, exist_ok=True)
        for name in CACHED:  # newest of X.npz / X.tmp.npz (a cache that could not be replaced on Windows)
            cands = [f for f in (run / "out" / "work" / name, run / "out" / "work" / name.replace(".npz", ".tmp.npz"))
                     if f.exists()]
            if cands:
                shutil.copy2(max(cands, key=lambda f: f.stat().st_mtime), CACHE / clip / name)
    rep_ = f" [{env_extra['AA_FOOT_JITTER']}]" if env_extra and env_extra.get('AA_FOOT_JITTER') else ""
    log(f"  {clip}{rep_}: pass 2 {'ok' if ok else 'FAILED (see ' + str(run / 'pass2.log') + ')'} "
        f"in {time.time() - t0:.0f}s")
    return {"ok": ok, "wall_s": round(time.time() - t0, 1)}


# ---------------------------------------------------------------------------------------------- output loading
def load_output(out: Path) -> dict:
    """tracking.csv rows grouped by video frame + team colour map + events + sample frames."""
    rows_by_frame: dict[int, list[dict]] = defaultdict(list)
    ball_by_frame: dict[int, dict] = {}
    ids_time: dict[str, list[float]] = defaultdict(list)
    with open(out / "tracking.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["object_type"] == "ball":
                if r["x_m"]:
                    ball_by_frame[int(r["frame"])] = {
                        "xy": (float(r["x_m"]), float(r["y_m"])), "state": r.get("ball_state", ""),
                        "box": [float(r[k]) for k in ("img_x1", "img_y1", "img_x2", "img_y2")] if r["img_x1"] else None}
                continue
            rec = {"frame": int(r["frame"]), "t": float(r["timestamp"]), "id": r["track_id"],
                   "type": r["object_type"], "team": r["team"], "xy": (float(r["x_m"]), float(r["y_m"])),
                   "interp": r["interpolated"] == "True", "source": r.get("source") or "",
                   "box": [float(r[k]) for k in ("img_x1", "img_y1", "img_x2", "img_y2")] if r["img_x1"] else None}
            rows_by_frame[rec["frame"]].append(rec)
            ids_time[rec["id"]].append(rec["t"])
    doc = json.loads((out / "tracking.json").read_text(encoding="utf-8"))
    colours = {k: (v.get("colour_name") or "").lower() for k, v in (doc.get("game", {}).get("teams") or {}).items()}
    rd = json.loads((out / "replay_data.json").read_text(encoding="utf-8"))
    return {"rows": rows_by_frame, "ball": ball_by_frame, "colours": colours, "events": doc.get("events") or [], "ids_time": ids_time,
            "frames": rd.get("frame") or [], "t": rd.get("t") or [], "court": doc.get("coordinate_system", {})}


def nearest_sample_frame(frames: list[int], f: int, max_gap: int = 2) -> int | None:
    """Output rows exist only on analysed samples (every Nth video frame): the closest sample to label frame f."""
    if not frames:
        return None
    import bisect
    j = bisect.bisect_left(frames, f)
    best = min((c for c in (j - 1, j) if 0 <= c < len(frames)), key=lambda c: abs(frames[c] - f))
    return frames[best] if abs(frames[best] - f) <= max_gap else None


# ---------------------------------------------------------------------------------------------- metrics
def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u > 0 else 0.0


def _greedy(pairs, used_l, used_o, res):
    for _, i, j in sorted(pairs, reverse=True):
        if i not in used_l and j not in used_o:
            used_l.add(i)
            used_o.add(j)
            res.append((i, j))


def match_frame(labels: list[dict], outs: list[dict], label_xy: list | None = None) -> list[tuple[int, int]]:
    """Greedy one-to-one matching (best pair first). Stage 1: image boxes, IoU >= IOU_MIN. Stage 2: the rest by
    court distance <= COURT_MATCH_M - box-less outputs (predicted/interpolated rows) against the label's court
    point (``court``, else ``label_xy[i]`` = its box foot projected with the run's calibration), and labels with
    only a court position against any output. Returns (label index, output index) pairs."""
    pairs = []
    for i, l in enumerate(labels):
        for j, o in enumerate(outs):
            if l.get("box") and o.get("box"):
                s = iou(l["box"], o["box"])
                if s >= IOU_MIN:
                    pairs.append((s, i, j))
    used_l, used_o, res = set(), set(), []
    _greedy(pairs, used_l, used_o, res)
    pairs = []
    for i, l in enumerate(labels):
        if i in used_l:
            continue
        xy = l.get("court") or (label_xy[i] if label_xy is not None else None)
        if xy is None:
            continue
        for j, o in enumerate(outs):
            if j in used_o or (o.get("box") and l.get("box")):
                continue  # two boxes that do not overlap are not the same person
            d = math.dist(xy, o["xy"])
            if d <= COURT_MATCH_M:
                pairs.append((-d, i, j))
    _greedy(pairs, used_l, used_o, res)
    return res


def _ratio(a, b):
    return None if not b else a / b


def player_metrics(label_doc: dict, output: dict, H=None, sample_frames=None) -> dict:
    """Detection / team / identity / position metrics from a players_<clip>.json label document."""
    frames = [f for f in label_doc.get("frames", []) if f.get("status", "done") == "done"]
    frames.sort(key=lambda f: f["frame"])
    c = Counter()
    team_pairs = []  # (label colour, output team letter)
    pid_seq: dict[str, list[tuple[float, str]]] = defaultdict(list)  # pid -> [(t, output id)] (None = missed)
    oid_seq: dict[str, list[tuple[float, str]]] = defaultdict(list)  # output id -> [(t, pid)]
    pos_err, foot_err = [], []
    for fr in frames:
        of = nearest_sample_frame(output["frames"], int(fr["frame"])) if output["frames"] else int(fr["frame"])
        outs = [o for o in output["rows"].get(of, [])] if of is not None else []
        shown = [o for o in outs if o["type"] in ("player", "referee") and o["team"] not in ("unknown", "")]
        people = fr.get("people", [])
        label_xy = None
        if H is not None and sample_frames is not None and of is not None:
            label_xy = [_project(H, sample_frames, of, ((p["box"][0] + p["box"][2]) / 2, p["box"][3]))
                        if p.get("box") else None for p in people]
        m = match_frame(people, shown, label_xy)
        ml = {i: j for i, j in m}
        t = shown[0]["t"] if shown else fr.get("t", fr["frame"] / 30.0)
        c["frames"] += 1
        if of is None or not outs:
            c["frames_without_output"] += 1
        for i, l in enumerate(people):
            kind = l.get("kind", "player")
            j = ml.get(i)
            o = shown[j] if j is not None else None
            if kind == "player":
                c["players"] += 1
                c["players_hit"] += o is not None
                c["players_hit_measured"] += o is not None and o.get("box") is not None
                if not l.get("occluded"):
                    c["players_vis"] += 1
                    c["players_vis_hit"] += o is not None
                if o is not None and o["type"] == "referee":
                    c["player_as_referee"] += 1
            elif kind == "referee":
                c["referees"] += 1
                c["referees_hit"] += o is not None and o["type"] == "referee"
            if o is not None and kind == "player" and l.get("team") and o["team"] in ("A", "B"):
                team_pairs.append((l["team"], o["team"]))
            if kind not in ("player", "referee"):
                continue
            if l.get("pid"):
                pid_seq[l["pid"]].append((t, o["id"] if o is not None else None))
                if o is not None:
                    scoped = l.get("pid_scope") == "segment" or (
                        "pid_scope" not in l and re.search(r"_[A-Z]$", str(l["pid"])) is not None)
                    oid_seq[o["id"]].append((t, l["pid"], scoped, fr.get("segment")))
            if o is not None and l.get("court"):
                pos_err.append(math.dist(l["court"], o["xy"]))
            elif o is not None and l.get("foot") and H is not None and sample_frames is not None:
                p = _project(H, sample_frames, of, l["foot"])
                if p is not None:
                    foot_err.append(math.dist(p, o["xy"]))
        if fr.get("complete", True):
            c["complete_frames"] += 1
            real = {j for i, j in m if people[i].get("kind", "player") in ("player", "referee", "ignore")}
            matched = {j for _, j in m}
            # box-less output rows (predicted / interpolated) that match no label at all: often a player hidden
            # behind someone (cannot be labelled) - reported separately, not as phantoms
            unverified = {j for j, o in enumerate(shown) if o.get("box") is None and j not in matched}
            c["unverified"] += len(unverified)
            c["phantoms"] += sum(1 for j, o in enumerate(shown) if j not in real and j not in unverified)
            c["bench_shown"] += sum(1 for i, j in m if people[i].get("kind") == "bench")
            c["dups_shown"] += sum(1 for i, j in m if people[i].get("kind") == "duplicate")
            c["ref_out"] += sum(1 for o in shown if o["type"] == "referee")
            c["ref_out_hit"] += sum(1 for i, j in m if shown[j]["type"] == "referee"
                                    and people[i].get("kind") == "referee")
    res = {
        "label_frames": c["frames"], "labelled_players": c["players"],
        "player_recall": _ratio(c["players_hit"], c["players"]),
        "player_recall_measured": _ratio(c["players_hit_measured"], c["players"]),
        "unverified_per_frame": _ratio(c["unverified"], c["complete_frames"]),
        "player_recall_visible": _ratio(c["players_vis_hit"], c["players_vis"]), "visible_players": c["players_vis"],
        "phantoms_per_frame": _ratio(c["phantoms"], c["complete_frames"]),
        "bench_shown_per_frame": _ratio(c["bench_shown"], c["complete_frames"]),
        "dups_shown_per_frame": _ratio(c["dups_shown"], c["complete_frames"]),
        "referee_recall": _ratio(c["referees_hit"], c["referees"]),
        "referee_precision": _ratio(c["ref_out_hit"], c["ref_out"]),
        "frames_without_output": c["frames_without_output"],
    }
    res.update(team_accuracy(team_pairs, output["colours"]))
    res.update(identity_metrics(pid_seq, oid_seq))
    res["position_error_m"] = round(sum(pos_err) / len(pos_err), 3) if pos_err else None
    res["position_n"] = len(pos_err)
    res["foot_vs_output_m"] = round(sum(foot_err) / len(foot_err), 3) if foot_err else None
    return res


def team_accuracy(pairs: list[tuple[str, str]], colours: dict[str, str]) -> dict:
    """team_acc: output letter -> its colour name (tracking.json) must equal the labelled colour.
    team_acc_perm: best A/B <-> colour assignment (pure clustering quality, ignores naming)."""
    if not pairs:
        return {"team_acc": None, "team_acc_perm": None, "team_n": 0}
    names = {v for v in colours.values() if v}
    labelled = {l for l, _ in pairs}
    # direct accuracy only when the labels use the output's colour names (TrackID3x3: offence/defence -> None)
    # colour-word labels whose words the output never uses = a naming failure (0 %), not "unknown"
    colour_words = {"orange", "purple", "white", "black", "red", "blue", "yellow", "green", "grey", "gray", "maroon",
                    "navy", "pink", "brown", "gold", "silver", "teal", "violet", "light", "dark"}
    direct = (sum(colours.get(o) == l for l, o in pairs) / len(pairs)
              if labelled & names or labelled & colour_words else None)
    ref = pairs[0][0]  # generic: the first labelled team value <-> A, everything else <-> B (and the swap)
    same = sum((l == ref) == (o == "A") for l, o in pairs)
    return {"team_acc": direct, "team_acc_perm": max(same, len(pairs) - same) / len(pairs), "team_n": len(pairs)}


def identity_metrics(pid_seq: dict, oid_seq: dict) -> dict:
    """ID switches: consecutive matched labelled sightings of one person with different output IDs.
    Fragmentation: mean number of distinct output IDs per labelled person. Transfers: an output ID that moves
    from one labelled person to another. Identity duration: mean time a person keeps one output ID (s)."""
    if not pid_seq:
        return {"id_switches": None, "id_fragmentation": None, "id_transfers": None, "id_duration_s": None,
                "persons": 0}
    switches, frags, durations = 0, [], []
    for pid, seq in pid_seq.items():
        seq = sorted(seq)
        hits = [(t, o) for t, o in seq if o is not None]
        if hits:
            frags.append(len({o for _, o in hits}))
        run_start, prev = None, None
        for t, o in hits:
            if prev is None or o != prev[1]:
                if prev is not None:
                    switches += 1
                    durations.append(prev[0] - run_start)
                run_start = t
            prev = (t, o)
        if prev is not None:
            durations.append(prev[0] - run_start)
    transfers = 0
    for oid, seq in oid_seq.items():  # entries (t, pid[, segment_scoped, segment])
        seq = sorted(seq, key=lambda e: e[0])
        for a, b in zip(seq, seq[1:]):
            if a[1] == b[1]:
                continue
            scoped = (len(a) > 2 and a[2]) or (len(b) > 2 and b[2])
            if scoped and (a[3] if len(a) > 3 else None) != (b[3] if len(b) > 3 else None):
                continue  # segment-scoped pids in different segments may be the same person: not provable
            transfers += 1
    return {"id_switches": switches, "id_fragmentation": round(sum(frags) / len(frags), 3) if frags else None,
            "id_transfers": transfers, "id_duration_s": round(sum(durations) / len(durations), 2) if durations else None,
            "persons": len(pid_seq)}


def output_identity_stats(output: dict) -> dict:
    """Label-free: distinct shown player/referee IDs per minute and mean ID lifetime."""
    ts = [t for rows in output["rows"].values() for t in (r["t"] for r in rows)]
    if not ts:
        return {"ids_per_min": None, "id_life_s": None}
    span = max(max(ts) - min(ts), 1.0)
    lives = [max(v) - min(v) for v in output["ids_time"].values()]
    return {"ids_per_min": round(len(lives) / (span / 60.0), 2), "id_life_s": round(sum(lives) / len(lives), 2)}


def continuity_metrics(rows: list[dict], fps: float | None = None, sample_t: list | None = None) -> dict:
    """Label-free continuity of the drawn people (tracking.json/csv rows as dicts: frame, timestamp, track_id,
    object_type, team). blink_holes_per_min: holes of 1-3 missing samples inside one output id, per minute;
    count_volatility: share of consecutive sample pairs whose drawn A/B/referee count differs by >= 2;
    median_id_s: median output-id length (first..last sample + one sample) in seconds."""
    import numpy as np

    people = [r for r in rows if r.get("object_type") != "ball"]
    if not people:
        return {"blink_holes_per_min": None, "count_volatility": None, "count_volatility_drawn": None,
                "median_id_s": None}
    ts = sorted({float(r["timestamp"]) for r in people})
    if fps is None:
        fps = 1.0 / float(np.median(np.diff(ts))) if len(ts) > 1 else 10.0
    # sample index: position in the run's sample times (replay_data.json "t") when given, else the rank among the
    # rows' distinct timestamps. NOT round(t * fps): process_fps is 9.99, so rounding drifts and merges two samples
    # every ~100 s (fake count jumps of +10 and fake 1-sample holes).
    grid = np.asarray(sorted({round(float(x), 3) for x in (sample_t if sample_t else ts)}), float)

    def k_of(r):
        t = round(float(r["timestamp"]), 3)
        j = int(np.searchsorted(grid, t))
        if j >= len(grid) or (j > 0 and abs(grid[j - 1] - t) < abs(grid[j] - t)):
            j -= 1
        return j
    by_id: dict = defaultdict(set)
    count: dict = defaultdict(int)
    count_drawn: dict = defaultdict(int)  # drawn = not source "unsure" (lineup exact-5 mode 3 keeps but hides them)
    for r in people:
        k = k_of(r)
        by_id[str(r["track_id"])].add(k)
        if r.get("team") in ("A", "B", "referee"):
            count[k] += 1
            if r.get("source") != "unsure":
                count_drawn[k] += 1
    holes, lengths = 0, []
    for ks in by_id.values():
        a = np.array(sorted(ks))
        gaps = np.diff(a) - 1
        holes += int(((gaps >= 1) & (gaps <= 3)).sum())
        lengths.append((a[-1] - a[0] + 1) / fps)
    duration = ts[-1] - ts[0] + 1.0 / fps
    k0, k1 = min(k_of(r) for r in people), max(k_of(r) for r in people)
    c = np.array([count.get(k, 0) for k in range(k0, k1 + 1)])
    vol = float((np.abs(np.diff(c)) >= 2).mean()) if len(c) > 1 else 0.0
    cd = np.array([count_drawn.get(k, 0) for k in range(k0, k1 + 1)])
    vol_d = float((np.abs(np.diff(cd)) >= 2).mean()) if len(cd) > 1 else 0.0
    return {"blink_holes_per_min": holes / (duration / 60.0), "count_volatility": vol,
            "count_volatility_drawn": vol_d,
            "median_id_s": float(np.median(lengths))}


def _project(H, sample_frames, frame, pt):
    import numpy as np
    try:
        i = sample_frames.index(frame)
    except ValueError:
        return None
    h = H[i]
    if not np.isfinite(h).all():
        return None
    v = h @ np.array([pt[0], pt[1], 1.0])
    return (float(v[0] / v[2]), float(v[1] / v[2])) if abs(v[2]) > 1e-12 else None


def calib_metrics(doc: dict, calib_file: Path, H=None, sample_frames=None) -> dict:
    """Judgements count only if made on the same calibration file (sha1); keypoints are scored live against the
    rebuilt per-frame homographies: a frame is 'right' when its mean keypoint error <= 0.5 m, 'off' <= 1.5 m."""
    res: dict = {}
    j = doc.get("judgements") or {}
    if j.get("frames"):
        sha = hashlib.sha1(calib_file.read_bytes()).hexdigest() if calib_file.exists() else ""
        vals = list(j["frames"].values())
        if not j.get("calibration_sha1") or j["calibration_sha1"] == sha:
            res["judged_right_pct"] = 100.0 * vals.count("right") / len(vals)
            res["judged_n"] = len(vals)
        else:
            res["judged_right_pct"] = None
            res["judged_note"] = "stale (calibration file changed since judged)"
    kps = doc.get("keypoints") or []
    if kps and H is not None:
        verdicts, errs = [], []
        for k in kps:
            f = nearest_sample_frame(sample_frames, int(k["frame"]), 0)
            e = [math.dist(p, pt["court"]) for pt in k.get("points", [])
                 if (p := _project(H, sample_frames, f, pt["img"])) is not None] if f is not None else []
            if not e:
                verdicts.append("wrong")  # no calibration at a frame where the court is visible
                continue
            m = sum(e) / len(e)
            errs.append(m)
            verdicts.append("right" if m <= 0.5 else "off" if m <= 1.5 else "wrong")
        res["kp_right_pct"] = 100.0 * verdicts.count("right") / len(verdicts)
        res["kp_wrong_pct"] = 100.0 * verdicts.count("wrong") / len(verdicts)
        res["kp_mean_err_m"] = round(sum(errs) / len(errs), 3) if errs else None
        res["kp_frames"] = len(verdicts)
    return res


def score_metrics(doc: dict, output: dict) -> dict:
    """Made baskets vs the score-bar ground truth. GT event times are when the score bar changed (t_kind
    'scorebar', default: an output basket 11 s to 1.5 s earlier matches) or when the ball went
    through ('ball': +-2 s)."""
    ts = [r["t"] for rows in output["rows"].values() for r in rows]
    end = max(ts) if ts else 1e9
    gt = []
    for g in doc.get("events", []):  # also the score-bar reader's format (t_bar, jersey)
        g = dict(g)
        if "t_bar" in g:
            g["t"], g["t_kind"] = g["t_bar"], "scorebar"
            g["team"] = g.get("jersey", g.get("team"))
        if g.get("t_kind", "scorebar") == "scorebar" and g["t"] < 2.0:
            continue  # the basket happened before the clip
        gt.append(g)
    lo, hi = doc.get("range_s") or [-1e9, 1e9]
    outs = [e for e in output["events"] if e.get("type") == "basket" and lo - 2 <= e["t"] <= hi + 2]
    pairs = []
    for i, g in enumerate(gt):
        before, after = (11.0, -1.5) if g.get("t_kind", "scorebar") == "scorebar" else (2.0, 2.0)
        for j, o in enumerate(outs):
            dt = g["t"] - o["t"]
            if -after <= dt <= before:
                pairs.append((abs(dt - 5.0) if g.get("t_kind", "scorebar") == "scorebar" else abs(dt), i, j))
    pairs.sort()
    ug, uo, m = set(), set(), []
    for _, i, j in pairs:
        if i not in ug and j not in uo:
            ug.add(i)
            uo.add(j)
            m.append((i, j))
    team_ok = [output["colours"].get(outs[j].get("team")) == gt[i].get("team") for i, j in m if gt[i].get("team")]
    # an unmatched detection in the last 9.5 s may be confirmed by the score bar only after the clip ends
    n_out = sum(1 for j, o in enumerate(outs) if j in uo or o["t"] <= end - 9.5)
    return {"basket_recall": _ratio(len(m), len(gt)), "basket_precision": _ratio(len(m), n_out),
            "basket_team_acc": _ratio(sum(team_ok), len(team_ok)), "baskets_gt": len(gt), "baskets_out": len(outs)}


# ---------------------------------------------------------------------------------------------- evaluate + report
def evaluate_clip(clip: str, info: dict, run: Path) -> dict:
    import numpy as np

    out = run / "out"
    res: dict = {"clip": clip}
    if not (out / "tracking.csv").exists():
        res["error"] = "no output"
        return res
    output = load_output(out)
    H = None
    if (run / "H.npz").exists():
        H = np.load(run / "H.npz")["H"]
        if len(H) != len(output["frames"]):
            H = None
    res.update(output_identity_stats(output))
    res.update(continuity_metrics([{"timestamp": r["t"], "track_id": r["id"], "object_type": r["type"],
                                    "team": r["team"], "frame": r["frame"], "source": r.get("source", "")}
                                   for rows in output["rows"].values() for r in rows],
                                  1.0 / float(np.median(np.diff(output["t"]))) if len(output["t"]) > 1 else None,
                                  sample_t=output["t"] or None))
    lf = label_files(clip)
    if "players" in lf:
        res.update(player_metrics(load_labels(lf["players"]), output, H, output["frames"]))
    if "calib" in lf:
        res.update(calib_metrics(load_labels(lf["calib"]), Path(info["calibration"]), H, output["frames"]))
    if "scores" in lf:
        res.update(score_metrics(load_labels(lf["scores"]), output))
    dense_doc = load_labels(lf["dense"]) if "dense" in lf else (
        load_labels(lf["players"]) if "players" in lf and clip.startswith("trackid3x3") else None)
    if dense_doc is not None:
        res.update(tracking_metrics(dense_doc, output, H))
    try:
        res.update(extrapolation_metrics(output))
    except Exception as e:  # noqa: BLE001
        res["extrap_error"] = repr(e)
    if (run / "ball_extrap.json").exists():
        res.update(json.loads((run / "ball_extrap.json").read_text(encoding="utf-8")))
    res["label_sources"] = {k: [str(f.relative_to(LABELS)) for f in v] for k, v in lf.items()}
    if (run / "timing.json").exists():
        res["timing"] = json.loads((run / "timing.json").read_text(encoding="utf-8"))
    return res


def _shown_at(output: dict):
    def at(f: int) -> list[dict]:
        of = nearest_sample_frame(output["frames"], f) if output["frames"] else f
        rows = output["rows"].get(of, []) if of is not None else []
        return [o for o in rows if o["type"] in ("player", "referee") and o["team"] not in ("unknown", "")]
    return at


def tracking_metrics(doc: dict, output: dict, H) -> dict:
    """HOTA / IDF1 / MOTA (bench_metrics) + ball error on dense identity labels."""
    from replay import bench_metrics as bmx

    frames = [f for f in doc.get("frames", []) if f.get("status", "done") == "done"]
    sf = output["frames"]

    def label_xy(fr, n):
        p = fr["people"][n]
        if H is None or not p.get("box"):
            return None
        of = nearest_sample_frame(sf, int(fr["frame"]))
        return _project(H, sf, of, ((p["box"][0] + p["box"][2]) / 2, p["box"][3])) if of is not None else None

    def ball_at(f):
        of = nearest_sample_frame(sf, f)
        return output["ball"].get(of) if of is not None else None

    def project(fr, pt):
        of = nearest_sample_frame(sf, int(fr["frame"]))
        return _project(H, sf, of, pt) if H is not None and of is not None else None

    res = bmx.identity_scores(frames, _shown_at(output), label_xy)
    if any(f.get("ball") for f in frames):
        res.update(bmx.ball_error(frames, ball_at, project))
    return res


def extrapolation_metrics(output: dict) -> dict:
    """Players: hide 1/3/5/10 s of the output's own measured stretches, fill (smoothing) / extrapolate
    (court_tracker.MotionPrior) and compare (bench_metrics.player_extrapolation)."""
    import numpy as np

    from replay import bench_metrics as bmx
    from replay.config import Config

    sf = output["frames"]
    if len(sf) < 2:
        return {}
    idx = {f: i for i, f in enumerate(sf)}
    ts = output.get("t") or []
    fps = 1.0 / float(np.median(np.diff(ts))) if len(ts) > 1 else 10.0
    per: dict = defaultdict(list)
    for f, rows in output["rows"].items():
        for r in rows:
            if r.get("box") is not None and not r.get("interp") and f in idx and r["type"] in ("player", "referee"):
                per[r["id"]].append((idx[f], r["xy"]))
    tracks = {}
    for tid, v in per.items():
        v.sort()
        tracks[tid] = (np.array([a for a, _ in v], int), np.array([b for _, b in v], float))
    cfg = Config()
    over = BENCH / "config.json"
    if over.exists():
        cfg = Config.from_dict(json.loads(over.read_text(encoding="utf-8")))
    prior = None
    try:
        from replay.court_tracker import MotionPrior

        det_i = np.concatenate([k for k, _ in tracks.values()]) if tracks else np.zeros(0, int)
        det_xy = np.concatenate([x for _, x in tracks.values()]) if tracks else np.zeros((0, 2))
        tid = np.concatenate([np.full(len(k), n + 1) for n, (k, _) in enumerate(tracks.values())]) if tracks \
            else np.zeros(0, int)
        prior = MotionPrior.from_detections(np.asarray(ts, float), det_i, det_xy, tid)
    except Exception:  # noqa: BLE001 - fall back to constant velocity
        prior = None
    return bmx.player_extrapolation(tracks, fps, cfg.smoothing, prior)


# (key, header, format, better: +1 higher / -1 lower / 0 neutral)
COLUMNS = [
    ("player_recall", "player recall", "pct", 1), ("player_recall_visible", "recall (unoccluded)", "pct", 1),
    ("player_recall_measured", "recall (detected rows)", "pct", 1), ("phantoms_per_frame", "phantoms/frame", "f2", -1),
    ("unverified_per_frame", "unverified predictions/frame", "f2", 0),
    ("bench_shown_per_frame", "bench shown/frame", "f2", -1), ("dups_shown_per_frame", "dups shown/frame", "f2", -1),
    ("team_acc", "team acc", "pct", 1), ("team_acc_perm", "team acc (perm)", "pct", 1),
    ("referee_recall", "ref recall", "pct", 1), ("referee_precision", "ref prec", "pct", 1),
    ("id_switches", "ID switches", "int", -1), ("id_fragmentation", "IDs/person", "f2", -1),
    ("id_transfers", "ID transfers", "int", -1), ("id_duration_s", "ID duration s", "f1", 1),
    ("ids_per_min", "out IDs/min", "f1", -1),
    ("blink_holes_per_min", "blink holes/min", "f1", -1), ("count_volatility", "count volatility", "pct", -1),
    ("count_volatility_drawn", "count volatility (drawn)", "pct", -1),
    ("median_id_s", "median ID s", "f1", 1), ("id_life_s", "out ID life s", "f1", 1),
    ("position_error_m", "pos err m", "f2", -1), ("foot_vs_output_m", "foot->out m", "f2", -1),
    ("judged_right_pct", "calib judged right %", "f1", 1), ("kp_right_pct", "calib kp right %", "f1", 1),
    ("kp_mean_err_m", "calib kp err m", "f2", -1),
    ("HOTA", "HOTA", "pct", 1), ("DetA", "DetA", "pct", 1), ("AssA", "AssA", "pct", 1), ("IDF1", "IDF1", "pct", 1),
    ("MOTA", "MOTA", "pct", 1), ("IDSW", "IDSW (dense)", "int", -1), ("HOTA_court", "HOTA court", "pct", 1),
    ("IDF1_court", "IDF1 court", "pct", 1),
    ("ball_recall", "ball recall", "pct", 1), ("ball_img_err_px", "ball err px", "f1", -1),
    ("ball_court_err_m", "ball err m", "f2", -1),
    ("fill_1s_med", "fill 1s m", "f2", -1), ("fill_3s_med", "fill 3s m", "f2", -1),
    ("fill_5s_med", "fill 5s m", "f2", -1), ("fill_10s_med", "fill 10s m", "f2", -1),
    ("fill_3s_p90", "fill 3s p90", "f2", -1),
    ("extrap_1s_med", "extrap 1s m", "f2", -1), ("extrap_3s_med", "extrap 3s m", "f2", -1),
    ("extrap_5s_med", "extrap 5s m", "f2", -1), ("extrap_10s_med", "extrap 10s m", "f2", -1),
    ("extrap_3s_p90", "extrap 3s p90", "f2", -1),
    ("ball_fill_1s_med", "ball fill 1s m", "f2", -1), ("ball_fill_3s_med", "ball fill 3s m", "f2", -1),
    ("ball_fill_5s_med", "ball fill 5s m", "f2", -1), ("ball_fill_10s_med", "ball fill 10s m", "f2", -1),
    ("ball_fill_3s_p90", "ball fill 3s p90", "f2", -1),
    ("basket_recall", "basket recall", "pct", 1), ("basket_precision", "basket prec", "pct", 1),
    ("basket_team_acc", "basket team", "pct", 1),
]
COUNT_KEYS = ("dense_gt", "labelled_players", "visible_players", "label_frames", "team_n", "persons", "position_n", "baskets_gt", "kp_frames")


GROUPS = [
    ("People (sparse labels)", {"player_recall", "player_recall_visible", "player_recall_measured", "phantoms_per_frame",
                                "unverified_per_frame", "bench_shown_per_frame", "dups_shown_per_frame", "team_acc",
                                "team_acc_perm", "referee_recall", "referee_precision", "position_error_m",
                                "foot_vs_output_m"}),
    ("Identity", {"blink_holes_per_min", "count_volatility", "count_volatility_drawn", "median_id_s", "id_switches", "id_fragmentation", "id_transfers", "id_duration_s", "ids_per_min", "id_life_s",
                  "HOTA", "DetA", "AssA", "IDF1", "MOTA", "IDSW", "HOTA_court", "IDF1_court"}),
    ("Extrapolation (hide and predict; median m)", {k for k, *_ in COLUMNS if "fill_" in k or "extrap_" in k}),
    ("Ball, baskets, calibration", {"ball_recall", "ball_img_err_px", "ball_court_err_m", "basket_recall",
                                    "basket_precision", "basket_team_acc", "judged_right_pct", "kp_right_pct",
                                    "kp_mean_err_m"}),
]


def _fmt(v, f):
    if v is None:
        return "-"
    return {"pct": f"{100 * v:.1f}%", "f2": f"{v:.2f}", "f1": f"{v:.1f}", "int": f"{int(v)}"}[f]


def overall(clips: list[dict]) -> dict:
    """Pooled metrics: ratios weighted by their denominators, counts summed, the rest averaged."""
    weights = {"player_recall": "labelled_players", "player_recall_measured": "labelled_players",
               "unverified_per_frame": "label_frames", "HOTA": "dense_gt", "DetA": "dense_gt", "AssA": "dense_gt",
               "IDF1": "dense_gt", "MOTA": "dense_gt", "HOTA_court": "dense_gt", "IDF1_court": "dense_gt", "player_recall_visible": "visible_players", "phantoms_per_frame": "label_frames",
               "bench_shown_per_frame": "label_frames", "dups_shown_per_frame": "label_frames", "team_acc": "team_n",
               "team_acc_perm": "team_n", "id_fragmentation": "persons", "id_duration_s": "persons",
               "position_error_m": "position_n", "kp_right_pct": "kp_frames", "kp_mean_err_m": "kp_frames",
               "basket_recall": "baskets_gt"}
    o: dict = {}
    for key, _, _, _ in COLUMNS:
        vals = [(c[key], c.get(weights.get(key, ""), 1) or 1) for c in clips if c.get(key) is not None]
        if not vals:
            o[key] = None
        elif key in ("id_switches", "id_transfers", "IDSW"):
            o[key] = sum(v for v, _ in vals)
        else:
            o[key] = sum(v * w for v, w in vals) / sum(w for _, w in vals)
    for k in COUNT_KEYS:
        o[k] = sum(c.get(k) or 0 for c in clips)
    t: dict = defaultdict(float)
    for c in clips:
        for k, v in (c.get("timing") or {}).items():
            t[k] += v
    o["timing"] = {k: round(v, 1) for k, v in t.items()}
    return o


def diff(cur: dict, prev: dict, cur_sd: dict | None = None, prev_sd: dict | None = None,
         n_cur: int = 1, n_prev: int = 1) -> list[str]:
    """Markdown rows for changed metrics. With replica spreads (--seeds) a change counts only when it exceeds
    2 standard errors: SE = sqrt(sd_now^2/n_now + sd_before^2/n_before); a run without replicas borrows the other
    run's sd (one sample). Without any sd the verdict is unqualified ("noise unknown")."""
    lines = []
    for key, head, f, better in COLUMNS:
        a, b = prev.get(key), cur.get(key)
        if a is None or b is None or abs(a - b) < 1e-9:
            continue
        sa = (prev_sd or {}).get(key)
        sb = (cur_sd or {}).get(key)
        if sa is None and sb is not None:
            sa, n_a = sb, 1
        else:
            n_a = n_prev
        if sb is None and sa is not None:
            sb, n_b = sa, 1
        else:
            n_b = n_cur
        good = (b - a) * better > 0
        verdict = "changed" if not better else "better" if good else "WORSE"
        if sa is None:
            lines.append(f"| {head} | {_fmt(a, f)} | {_fmt(b, f)} | - | {verdict} (noise unknown) |")
            continue
        se2 = 2 * math.sqrt(sa ** 2 / max(n_a, 1) + sb ** 2 / max(n_b, 1))
        sig = abs(b - a) > se2
        lines.append(f"| {head} | {_fmt(a, f)} | {_fmt(b, f)} | {_fmt(se2, f)} | "
                     f"{verdict if sig else '~ within noise'} |")
    return lines


NUMERIC = [k for k, _, _, _ in COLUMNS]


def aggregate(replicas: list[list[dict]]) -> tuple[list[dict], dict]:
    """Per-clip mean over jitter replicas (same clip order in each) + {clip: {metric: sd}}."""
    means, sds = [], {}
    for idx, first in enumerate(replicas[0]):
        group = [r[idx] for r in replicas if not r[idx].get("error")]
        m = dict(first)
        sds[first["clip"]] = {}
        if not group:
            means.append(m)
            continue
        m.pop("error", None)
        for k in NUMERIC:
            v = [g[k] for g in group if g.get(k) is not None]
            if v:
                m[k] = sum(v) / len(v)
                sds[first["clip"]][k] = _sd(v)
        tim: dict = defaultdict(list)
        for g in group:
            for kk, vv in (g.get("timing") or {}).items():
                tim[kk].append(vv)
        m["timing"] = {kk: round(sum(v) / len(v), 2) for kk, v in tim.items()}
        means.append(m)
    return means, sds


def _sd(v: list[float]) -> float:
    if len(v) < 2:
        return 0.0
    mu = sum(v) / len(v)
    return math.sqrt(sum((x - mu) ** 2 for x in v) / (len(v) - 1))


def overall_stats(doc: dict, clips: set | None = None) -> tuple[dict, dict | None, int]:
    """(overall mean, overall sd or None, number of replicas) of a result doc, optionally on a clip subset."""
    pick = lambda cs: [c for c in cs if (clips is None or c["clip"] in clips) and not c.get("error")]  # noqa: E731
    reps = doc.get("replicas")
    if not reps:
        return (doc.get("overall", {}) if clips is None else overall(pick(doc.get("clips", [])))), None, 1
    ovs = [overall(pick(r)) for r in reps]
    mean, sd = {}, {}
    for k in NUMERIC:
        v = [o[k] for o in ovs if o.get(k) is not None]
        if v:
            mean[k] = sum(v) / len(v)
            sd[k] = _sd(v)
    return mean, sd, len(reps)


def write_report(doc: dict, prev: dict | None) -> str:
    L = [f"# Benchmark results ({doc['created']}, tag: {doc.get('tag') or '-'})", "",
         f"Clips: {', '.join(c['clip'] for c in doc['clips'])}. Mode: {'quick' if doc.get('quick') else 'full'}. "
         "Label format + metric definitions: README.md. '-' = no labels of that kind for the clip."
         + (f" Values = mean ±sd over {doc['seeds']} foot-point jitter replicas ({doc.get('jitter_m')} m)."
            if doc.get("seeds") else " Single run (no noise estimate; use --seeds 3 to gate decisions)."), ""]
    for title, group in GROUPS:
        cols = [c for c in COLUMNS if c[0] in group
                and any(x.get(c[0]) is not None for x in doc["clips"] + [doc["overall"]])]
        if not cols:
            continue
        L += [f"### {title}", ""]
        L.append("| clip | " + " | ".join(h for _, h, _, _ in cols) + " |")
        L.append("|" + "---|" * (len(cols) + 1))
        for c in doc["clips"] + [dict(doc["overall"], clip="**overall**")]:
            if c.get("error"):
                L.append(f"| {c['clip']} | ERROR: {c['error']} |")
                continue
            if all(c.get(k) is None for k, _, _, _ in cols):
                continue
            sd = doc.get("overall_sd") if c["clip"] == "**overall**" else (doc.get("clips_sd") or {}).get(c["clip"])
            cell = lambda k, f: _fmt(c.get(k), f) + (f" ±{_fmt(sd[k], f)}" if sd and sd.get(k) else "")  # noqa: E731
            L.append(f"| {c['clip']} | " + " | ".join(cell(k, f) for k, _, f, _ in cols) + " |")
        L.append("")
    L += ["", "## Runtime per stage (s, pass 2)", ""]
    stages = sorted({k for c in doc["clips"] for k in (c.get("timing") or {})} - {"total"}) + ["total"]
    L.append("| clip | " + " | ".join(stages) + " |")
    L.append("|" + "---|" * (len(stages) + 1))
    for c in doc["clips"] + [dict(doc["overall"], clip="**sum**")]:
        t = c.get("timing") or {}
        L.append(f"| {c['clip']} | " + " | ".join(f"{t[s]:.1f}" if s in t else "-" for s in stages) + " |")
    L += ["", "## Change vs previous run", ""]
    if prev is None:
        L.append("(no previous run)")
    else:
        same = [c["clip"] for c in doc["clips"]] == [c["clip"] for c in prev.get("clips", [])]
        common = {c["clip"] for c in doc["clips"]} & {c["clip"] for c in prev.get("clips", [])}
        L.append(f"Previous: {prev.get('created')} (tag {prev.get('tag') or '-'}; env {prev.get('env') or '-'})"
                 + ("" if same else f" - different clip sets: overall compared on the common clips "
                                    f"{', '.join(sorted(common))}"))
        sub = None if same else common
        cm, cs, cn = overall_stats(doc, sub)
        pm, ps, pn = overall_stats(prev, sub)
        rows = diff(cm, pm, cs, ps, cn, pn)
        L += ["", f"Significance: a change must exceed 2 SE (replicas: now {cn if cs else 0}, before {pn if ps else 0}).",
              "", "| overall metric | before | now | 2 SE | |", "|---|---|---|---|---|"] + (rows or ["| (no change) | | | | |"])
        pc = {c["clip"]: c for c in prev.get("clips", [])}
        for c in doc["clips"]:
            if c["clip"] in pc:
                r = diff(c, pc[c["clip"]], (doc.get("clips_sd") or {}).get(c["clip"]),
                         (prev.get("clips_sd") or {}).get(c["clip"]), doc.get("seeds") or 1, prev.get("seeds") or 1)
                if r:
                    L += ["", f"{c['clip']}:", "", "| metric | before | now | 2 SE | |", "|---|---|---|---|---|"] + r
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="replay benchmark")
    ap.add_argument("--clips", help="comma-separated clip names (default: every ready clip with labels)")
    ap.add_argument("--quick", action="store_true", help=f"only {', '.join(QUICK_CLIPS)} (< 5 min)")
    ap.add_argument("--jobs", type=int, default=3, help="clips rebuilt in parallel (1 = clean per-stage timing)")
    ap.add_argument("--no-run", action="store_true", help="re-score the existing runs/ without rebuilding")
    ap.add_argument("--seeds", type=int, default=0,
                    help="N foot-point jitter replicas (AA_FOOT_JITTER) -> mean ±sd, significance in the diff")
    ap.add_argument("--code", help="run the workers on a snapshot of ANOTHER code tree (e.g. the live app folder); "
                                   "data, labels and metrics stay from this tree")
    ap.add_argument("--live-code", action="store_true",
                    help="run workers on the live replay/ (default: a frozen snapshot taken at start)")
    ap.add_argument("--jitter", type=float, default=0.01, help="jitter sigma in metres for --seeds (default 0.01)")
    ap.add_argument("--all", action="store_true", help="also clips without labels (label-free metrics + runtime)")
    ap.add_argument("--tag", default="", help="name for this run in the history")
    ap.add_argument("--vs", default="", help="compare with the latest history run with this tag (e.g. morning-base)")
    a = ap.parse_args(argv)
    reg = clip_registry()
    if a.clips:
        names = [c.strip() for c in a.clips.split(",") if c.strip()]
        unknown = [c for c in names if c not in reg]
        if unknown:
            raise SystemExit(f"unknown clips {unknown}; known: {sorted(reg)}")
    elif a.quick:
        names = [c for c in QUICK_CLIPS if c in reg]
    else:
        names = [c for c in reg if a.all or label_files(c)]
    names = [c for c in names if ready(reg[c])]
    if not names:
        raise SystemExit("no ready clips (pass 1 DONE + calibration) with labels")
    print(f"benchmark: {', '.join(names)}")
    t0 = time.time()
    if a.seeds and a.no_run:
        raise SystemExit("--seeds needs fresh runs (drop --no-run)")
    run_of = {c: RUNS / c for c in names}
    replicas, clips_sd = None, None
    if not a.no_run:  # private folders: several agents may run the benchmark at the same time
        tmp = RUNS / "_work" / f"{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}"
        code = None if a.live_code and not a.code else snapshot_code(tmp / "code", a.code)
        if a.code and code is None:
            raise SystemExit("--code: snapshot failed")
        code_env = {"AA_BENCH_CODE": str(code)} if code else {}
        seeds = list(range(1, a.seeds + 1)) or [None]
        tasks = [(c, sd) for sd in seeds for c in names]
        rdir = lambda c, sd: tmp / (c if sd is None else f"{c}_s{sd}")  # noqa: E731
        with ThreadPoolExecutor(max(1, a.jobs)) as ex:
            list(ex.map(lambda t: run_clip(t[0], reg[t[0]], run=rdir(*t), env_extra=dict(code_env, **(
                {} if t[1] is None else {"AA_FOOT_JITTER": f"{a.jitter},{t[1]}"}))), tasks))
        run_of = {c: rdir(c, seeds[0]) for c in names}
        if a.seeds:
            replicas = [[evaluate_clip(c, reg[c], rdir(c, sd)) for c in names] for sd in seeds]
            clips, clips_sd = aggregate(replicas)
    if replicas is None:
        clips = [evaluate_clip(c, reg[c], run_of[c]) for c in names]
    if not a.no_run:  # publish as runs/<clip> (latest run wins; used by --no-run and the label tool)
        for c in names:
            dst = RUNS / c
            try:
                if dst.exists():
                    shutil.rmtree(dst)
                os.replace(run_of[c], dst)
            except OSError as e:
                print(f"  note: could not publish {run_of[c]} as {dst}: {e}")
        if code:  # remove the junctions themselves first - never let rmtree walk into the real models/tools/data
            for name in ("models", "tools", "data"):
                j = code / name
                try:
                    if os.path.lexists(j):
                        (os.unlink if os.path.islink(j) else os.rmdir)(j)
                except OSError as e:
                    print(f"  note: could not remove {j}: {e}; leaving {tmp} in place")
                    code = "keep"
        if code != "keep":
            shutil.rmtree(tmp, ignore_errors=True)
    doc = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "tag": a.tag, "quick": a.quick, "clips": clips,
           "overall": overall([c for c in clips if not c.get("error")]), "wall_s": round(time.time() - t0, 1),
           "seeds": a.seeds, "jitter_m": a.jitter if a.seeds else None, "clips_sd": clips_sd,
           "replicas": None if replicas is None else [[{k: v for k, v in c.items() if k != "timing"} for c in r]
                                                      for r in replicas],
           "env": {k: v for k, v in os.environ.items() if k.startswith("REPLAY_")},
           "code": a.code or ("live" if a.live_code else "snapshot"),
           "config_overrides": json.loads((BENCH / "config.json").read_text(encoding="utf-8"))
           if (BENCH / "config.json").exists() else None}
    if replicas is not None:
        m, sd, _ = overall_stats(doc)
        doc["overall"].update(m)
        doc["overall_sd"] = sd
    HISTORY.mkdir(parents=True, exist_ok=True)
    key = ",".join(names)
    prev = None
    for f in sorted(HISTORY.glob("*.json"), reverse=True):  # latest run on the same clip set, else the latest
        d = json.loads(f.read_text(encoding="utf-8"))
        if a.vs:
            if d.get("tag") == a.vs:
                prev = d
                break
            continue
        if prev is None:
            prev = d
        if ",".join(c["clip"] for c in d.get("clips", [])) == key:
            prev = d
            break
    md = write_report(doc, prev)
    (BENCH / "results.json").write_text(json.dumps(doc, indent=1, default=float), encoding="utf-8")
    (BENCH / "results.md").write_text(md, encoding="utf-8")
    stamp = time.strftime("%Y%m%d-%H%M%S") + (f"-{a.tag}" if a.tag else "")
    (HISTORY / f"{stamp}.json").write_text(json.dumps(doc, indent=1, default=float), encoding="utf-8")
    print(md)
    print(f"wrote {BENCH / 'results.md'} ({doc['wall_s']:.0f}s)")
    return 0


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "_worker":
        _worker(Path(sys.argv[2]))
    else:
        sys.exit(main(sys.argv[1:]))
