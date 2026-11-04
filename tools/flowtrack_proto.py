"""flowtrack_proto.py - standalone prototype: global assignment (Hungarian = min-cost path cover) over tracklet
pieces with team cardinality (<= 5 on court per team) and a bench-zone penalty. NEW FILE, touches nothing in replay/.

    PYTHONUTF8=1 .venv/Scripts/python tools/flowtrack_proto.py [--clip c300] [--scratch DIR] [--max-gap 4]

What it does
 1. copies data/benchmark/runs/<clip> to the scratch folder and runs replay.postprocess.run_pass2 on the copy with
    court_tracker.track_people monkey-patched to CAPTURE its arguments (k, xy, R, feats, bt_ids, conf, fps, cfg,
    boxes, motion, H, numbers, cues) and abort - so pass-2 steps 1-2 build the detections exactly as the pipeline does.
 2. builds the tracklet pieces with the existing tracklets._Data / build_tracklets / summarise.
 3. baseline: tracklets.link_tracks on the same input.
 4. proto: rounds of global linear_sum_assignment over all pieces (cost = tracklets._link_cost, no-link cost =
    threshold + cardinality bonus when the team already has >= 5 live identities, bench pieces penalised),
    gaps 0.5 s .. --max-gap s. Bench pieces (standing outside the sideline) are dropped from the output.
 5. scores both with replay.benchmark.output_identity_stats / continuity_metrics / player_metrics on a synthetic
    "output" built from detections (pre-roster, pre-smoothing: apples to apples), writes <scratch>/flowtrack_<clip>.json.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from replay import court  # noqa: E402
from replay import tracklets as tl  # noqa: E402


class _Captured(Exception):
    pass


# ----------------------------------------------------------------------------------------------- 1. capture
def capture_tracking_input(clip: str, scratch: Path) -> dict:
    """Run pass 2 on a scratch copy of the benchmark run until court tracking; return track_people's arguments."""
    from replay import benchmark as bm
    from replay import court_tracker, postprocess as pp
    from replay.config import Config
    from replay.metadata import load_metadata

    src = bm.BENCH / "runs" / clip
    run = scratch / f"run_{clip}"
    if not (run / "out" / "work" / "run.json").exists():
        shutil.copytree(src, run, dirs_exist_ok=True)
    cfg = Config()
    over = bm.BENCH / "config.json"
    if over.exists():
        cfg = Config.from_dict(json.loads(over.read_text(encoding="utf-8")))
    sig = json.loads((run / "out" / "work" / "run.json").read_text(encoding="utf-8"))
    meta = run / "meta.json"
    got: dict = {}
    names = ["k", "xy", "R", "feats", "bt_ids", "conf", "fps", "cfg"]

    def fake_track_people(*a, **kw):
        got.update(dict(zip(names, a)))
        got.update(kw)
        raise _Captured()

    orig = pp.track_people
    pp.track_people = fake_track_people
    try:
        pp.run_pass2(run / "out", cfg, load_metadata(str(meta) if meta.exists() else None), run / "calibration.json",
                     Path(sig["video"]))
    except _Captured:
        pass
    finally:
        pp.track_people = orig
        court_tracker.track_people = orig
    if "k" not in got:
        raise RuntimeError("track_people was not reached")
    rd = json.loads((src / "out" / "replay_data.json").read_text(encoding="utf-8"))
    got["frames"], got["t"] = rd.get("frame") or [], rd.get("t") or []
    return got


# ----------------------------------------------------------------------------------------------- 2. pieces
def build_pieces(inp: dict):
    k = np.asarray(inp["k"], int)
    n = len(k)
    D = tl._Data(k, np.asarray(inp["xy"], float).reshape(n, 2), np.asarray(inp["R"], float).reshape(n, 2, 2),
                 np.asarray(inp["feats"], float).reshape(n, -1), np.asarray(inp["conf"], float), inp["fps"], inp["cfg"],
                 None if inp.get("boxes") is None else np.asarray(inp["boxes"], float).reshape(n, 4),
                 inp.get("motion"), inp.get("n_samples"), inp.get("H"), numbers=inp.get("numbers"), cues=inp.get("cues"))
    pieces = [D.summarise(p) for p in tl.build_tracklets(D, np.asarray(inp["bt_ids"], int))]
    return D, pieces


def piece_team(D, pieces) -> np.ndarray:
    """0/1 team per piece from a 2-means on the mean jersey Lab features (-1 = no kit / unknown)."""
    F = np.array([np.nanmean(D.feats[p.det], axis=0) for p in pieces])
    ok = np.isfinite(F).all(axis=1)
    team = np.full(len(pieces), -1, int)
    if ok.sum() < 4:
        return team
    X = F[ok]
    w = np.array([len(p.det) for p in pieces])[ok].astype(float)
    rng = np.random.default_rng(0)
    c = X[rng.choice(len(X), 2, replace=False)]
    for _ in range(30):
        d = ((X[:, None, :] - c[None]) ** 2).sum(-1)
        lab = d.argmin(1)
        for j in range(2):
            if (lab == j).any():
                c[j] = np.average(X[lab == j], axis=0, weights=w[lab == j])
    d = ((X[:, None, :] - c[None]) ** 2).sum(-1)
    lab = d.argmin(1)
    # far from both centres (referee / bench in other clothes) -> unknown
    spread = np.sqrt(np.median(d.min(1)))
    far = np.sqrt(d.min(1)) > 3.0 * max(spread, 1e-6)
    lab[far] = -1
    team[ok] = lab
    return team


def is_bench(D, p, margin: float = 0.75) -> bool:
    """Piece whose median position stands outside the court sideline (bench / coaches / table)."""
    xy = D.xy[p.det]
    xy = xy[np.isfinite(xy).all(axis=1)]
    if len(xy) == 0:
        return False
    m = np.median(xy, axis=0)
    outside = m[1] < -margin or m[1] > court.WIDTH + margin or m[0] < -margin or m[0] > court.LENGTH + margin
    speed = float(np.linalg.norm(p.cv1)) if p.cv1 is not None else 0.0
    return bool(outside and speed < 1.5)


# ----------------------------------------------------------------------------------------------- 4. global solve
def live_count(pieces, team, bench, k, n_samples):
    """live identities per team per sample (on-court pieces only)."""
    cnt = np.zeros((2, n_samples + 1), int)
    for p, t, b in zip(pieces, team, bench):
        if t >= 0 and not b:
            cnt[t, p.k0:p.k1 + 1] += 1
    return cnt


def solve_round(D, pieces, max_gap_s: float, thr: float, team, bench, n_samples, card_bonus=3.0, bench_pen=4.0,
                team_pen=6.0):
    """One global assignment (sparse min-weight bipartite matching = min-cost path cover): ends x [starts | private
    no-link column]. Candidates are pre-gated with vectorised time + court-reach tests before tracklets._link_cost
    is called. 'no link' = thr (+ card_bonus for a successor whose team already has > 5 live ids at its start, i.e.
    continuing someone is preferred). Returns links [(a, b)]."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import min_weight_full_bipartite_matching

    cfg = D.cfg
    n = len(pieces)
    if n < 2:
        return [], 0
    max_gap = max(1, int(round(max_gap_s * D.fps)))
    k0 = np.array([p.k0 for p in pieces])
    k1 = np.array([p.k1 for p in pieces])
    c0 = np.array([p.c0 if p.c0 is not None else (np.nan, np.nan) for p in pieces], float)
    c1 = np.array([p.c1 if p.c1 is not None else (np.nan, np.nan) for p in pieces], float)
    cv0 = np.array([p.cvar0 for p in pieces], float)
    cv1 = np.array([p.cvar1 for p in pieces], float)
    order = np.argsort(k0, kind="stable")
    ks = k0[order]
    cnt = live_count(pieces, team, bench, D.k, n_samples)
    start_cost = np.full(n, thr)
    for j, b in enumerate(pieces):
        if team[j] >= 0 and not bench[j] and cnt[team[j], b.k0] > 5:
            start_cost[j] += card_bonus
    rows, cols, vals = [], [], []
    n_cand = 0
    for i in range(n):
        a = pieces[i]
        lo = np.searchsorted(ks, a.k1 - cfg.max_overlap_samples + 1, "left")
        hi = np.searchsorted(ks, a.k1 + max_gap, "right")
        cand = order[lo:hi]
        cand = cand[(k0[cand] > a.k0) & (k1[cand] > a.k1) & (cand != i)]
        if len(cand) == 0:
            continue
        g = np.maximum(k0[cand] - a.k1, 0) / D.fps
        if np.isfinite(c1[i]).all():  # vectorised court reach gate (same formula as _link_cost)
            d = np.linalg.norm(c0[cand] - c1[i], axis=1)
            reach = cfg.max_speed_mps * g + cfg.reach_slack_m + 2.0 * np.sqrt(cv1[i] + cv0[cand])
            ok = np.isnan(d) | (d <= reach)
            cand, g = cand[ok], g[ok]
        for j, gj in zip(cand, g):
            n_cand += 1
            c = tl._link_cost(D, a, pieces[j], max_gap)
            if c is None:
                continue
            if team[i] >= 0 and team[j] >= 0 and team[i] != team[j]:
                c += team_pen
            if bench[i] != bench[j]:
                c += bench_pen
            if c < start_cost[j]:
                rows.append(i)
                cols.append(int(j))
                vals.append(c)
    if not rows:
        return [], n_cand
    off = 1.0 - min(0.0, min(vals))
    rows_all = np.r_[rows, np.arange(n)]
    cols_all = np.r_[cols, n + np.arange(n)]
    vals_all = np.r_[np.array(vals) + off, np.full(n, thr + off)]
    M = csr_matrix((vals_all, (rows_all, cols_all)), shape=(n, 2 * n))
    r, c = min_weight_full_bipartite_matching(M)
    cost_of = {(a, b): v for a, b, v in zip(rows, cols, vals)}
    links = [(int(a), int(b)) for a, b in zip(r, c) if b < n]
    D.link_info = {(a, b): (float(cost_of[(a, b)]), max(0.0, (pieces[b].k0 - pieces[a].k1) / D.fps), float(thr), False)
                   for a, b in links}
    return links, n_cand


def flow_link(D, pieces, rounds, n_samples, cuts=True, **kw):
    team = piece_team(D, pieces)
    bench = np.array([is_bench(D, p) for p in pieces])
    stats = {"rounds": []}
    for gap_s, thr in rounds:
        links, n_cand = solve_round(D, pieces, gap_s, thr, team, bench, n_samples, **kw)
        n_raw = len(links)
        if cuts:  # the existing guards from link_tracks: pile-ups, appearance chains, jersey-number chains
            info = D.link_info
            if gap_s <= D.cfg.link_rounds_s[0]:
                links = tl._pileup_cuts(D, pieces, links)
            links = tl._chain_appearance_cuts(D, pieces, links, gap_s)
            if D.numbers is not None:
                links = tl._number_chain_cuts(D, pieces, links, gap_s)
            D.link_info = info
        stats["rounds"].append({"gap_s": gap_s, "thr": thr, "candidates": n_cand, "links": len(links),
                                "cut": n_raw - len(links), "pieces_before": len(pieces)})
        if links:
            # carry team/bench through the merge: _merge keeps the predecessor's order -> recompute cheaply
            pieces = tl._merge(D, pieces, links)
            team = piece_team(D, pieces)
            bench = np.array([is_bench(D, p) for p in pieces])
    stats["pieces"] = len(pieces)
    stats["bench_pieces"] = int(bench.sum())
    return pieces, team, bench, stats


def ids_from_pieces(D, pieces, keep_mask) -> np.ndarray:
    ids = np.zeros(len(D.k), int)
    nid = 1
    for p, keep in sorted(zip(pieces, keep_mask), key=lambda z: (z[0].k0, z[0].k1)):
        if not keep:
            continue
        ids[p.det] = nid
        nid += 1
    return ids


# ----------------------------------------------------------------------------------------------- 5. scoring
def synth_output(inp: dict, ids: np.ndarray, xy: np.ndarray, team_of_id: dict) -> dict:
    """benchmark.load_output-shaped dict from detection ids (one row per detection; team A/B by id)."""
    frames, ts = inp["frames"], inp["t"]
    k = np.asarray(inp["k"], int)
    boxes = inp.get("boxes")
    rows = defaultdict(list)
    ids_time = defaultdict(list)
    for d in range(len(k)):
        if ids[d] <= 0 or k[d] >= len(frames):
            continue
        f, t = int(frames[k[d]]), float(ts[k[d]])
        rec = {"frame": f, "t": t, "id": str(ids[d]), "type": "player", "team": team_of_id.get(int(ids[d]), "A"),
               "xy": (float(xy[d, 0]), float(xy[d, 1])), "interp": False, "source": "",
               "box": None if boxes is None else [float(v) for v in boxes[d]]}
        rows[f].append(rec)
        ids_time[rec["id"]].append(t)
    return {"rows": rows, "ball": {}, "colours": {}, "events": [], "ids_time": ids_time, "frames": frames, "t": ts,
            "court": {}}


def score(clip: str, inp: dict, ids: np.ndarray, xy: np.ndarray, team_of_id: dict, H) -> dict:
    from replay import benchmark as bm

    out = synth_output(inp, ids, xy, team_of_id)
    res = {}
    res.update(bm.output_identity_stats(out))
    res.update(bm.continuity_metrics([{"timestamp": r["t"], "track_id": r["id"], "object_type": r["type"],
                                       "team": r["team"], "frame": r["frame"], "source": ""}
                                      for rs in out["rows"].values() for r in rs],
                                     1.0 / float(np.median(np.diff(out["t"]))) if len(out["t"]) > 1 else None,
                                     sample_t=out["t"] or None))
    lf = bm.label_files(clip)
    if "players" in lf:
        try:
            Harr = None
            if H is not None:
                Harr = np.full((len(H), 3, 3), np.nan)
                for i, h in enumerate(H):
                    if h is not None:
                        Harr[i] = h
            pm = bm.player_metrics(bm.load_labels(lf["players"]), out, Harr, out["frames"])
            res.update({kk: pm.get(kk) for kk in ("player_recall", "id_switches", "id_fragmentation", "id_transfers",
                                                   "id_duration_s", "persons", "phantoms_per_frame")})
        except Exception as e:  # noqa: BLE001
            res["label_error"] = repr(e)
    res["n_ids"] = int(len(set(ids[ids > 0])))
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="c300")
    ap.add_argument("--scratch", default=None)
    ap.add_argument("--max-gap", type=float, default=4.0)
    ap.add_argument("--thr", type=float, default=None, help="no-link cost (default: cfg.link_thresholds[0])")
    ap.add_argument("--card-bonus", type=float, default=3.0)
    ap.add_argument("--bench-pen", type=float, default=4.0)
    ap.add_argument("--team-pen", type=float, default=6.0)
    ap.add_argument("--no-cuts", action="store_true", help="skip the pileup/appearance/number chain cut passes")
    ap.add_argument("--min-len-s", type=float, default=0.0, help="drop output ids shorter than this (both arms)")
    a = ap.parse_args(argv)
    scratch = Path(a.scratch) if a.scratch else ROOT / "data" / "benchmark" / "runs" / "_work" / "flowtrack_proto"
    scratch.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    inp = capture_tracking_input(a.clip, scratch)
    print(f"captured {len(inp['k'])} detections in {time.perf_counter() - t0:.1f} s")
    cfg = inp["cfg"]
    n_samples = int(inp.get("n_samples") or (max(inp["k"]) + 1))

    # baseline: the existing linker on the same input
    t1 = time.perf_counter()
    base = tl.link_tracks(inp["k"], inp["xy"], inp["R"], inp["feats"], inp["bt_ids"], inp["conf"], inp["fps"], cfg,
                          boxes=inp.get("boxes"), motion=inp.get("motion"), n_samples=inp.get("n_samples"),
                          H=inp.get("H"), numbers=inp.get("numbers"), cues=inp.get("cues"))
    base_s = time.perf_counter() - t1
    base_ids = base.ids.copy()
    if a.min_len_s > 0:  # same short-id filter on the baseline (roster/caps drop crumbs later in the pipeline)
        kk = np.asarray(inp["k"], int)
        for i in np.unique(base_ids[base_ids > 0]):
            m = base_ids == i
            if (kk[m].max() - kk[m].min() + 1) / inp["fps"] < a.min_len_s:
                base_ids[m] = 0

    # proto
    t2 = time.perf_counter()
    D, pieces = build_pieces(inp)
    n_pieces0 = len(pieces)
    thr0 = a.thr if a.thr is not None else float(cfg.link_thresholds[0])
    gaps = [g for g in (0.5, 1.0, 2.0, 4.0, 8.0) if g <= a.max_gap] or [a.max_gap]
    rounds = [(g, thr0 + 0.5 * i) for i, g in enumerate(gaps)]
    pieces, team, bench, st = flow_link(D, pieces, rounds, n_samples, cuts=not a.no_cuts, card_bonus=a.card_bonus,
                                        bench_pen=a.bench_pen, team_pen=a.team_pen)
    keep = ~bench
    if a.min_len_s > 0:
        keep &= np.array([(p.k1 - p.k0 + 1) / D.fps >= a.min_len_s for p in pieces])
    ids = ids_from_pieces(D, pieces, keep)
    team_of_id = {}
    nid = 1
    for p, kp, t in sorted(zip(pieces, keep, team), key=lambda z: (z[0].k0, z[0].k1)):
        if kp:
            team_of_id[nid] = "AB?"[t] if t >= 0 else "A"
            nid += 1
    proto_s = time.perf_counter() - t2

    H = inp.get("H")
    res = {"clip": a.clip, "detections": int(len(inp["k"])), "pieces": n_pieces0,
           "baseline": dict(score(a.clip, inp, base_ids, base.xy, {}, H), seconds=round(base_s, 1),
                            identities=base.stats.get("identities")),
           "proto": dict(score(a.clip, inp, ids, D.xy, team_of_id, H), seconds=round(proto_s, 1), **st),
           "params": vars(a), "rounds": rounds}
    (scratch / f"flowtrack_{a.clip}.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    keys = ["n_ids", "ids_per_min", "median_id_s", "id_life_s", "id_switches", "id_transfers", "id_fragmentation",
            "player_recall", "phantoms_per_frame", "seconds"]
    print(f"\n{a.clip}: {res['detections']} detections, {n_pieces0} pieces; rounds {rounds}")
    print(f"{'metric':20s} {'baseline':>12s} {'proto':>12s}")
    for kk in keys:
        print(f"{kk:20s} {str(res['baseline'].get(kk)):>12s} {str(res['proto'].get(kk)):>12s}")
    print("proto rounds:", st["rounds"], "bench pieces dropped:", st["bench_pieces"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
