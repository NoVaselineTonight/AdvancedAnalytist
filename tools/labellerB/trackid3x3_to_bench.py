"""Convert TrackID3x3 Indoor ground truth (data/external/trackid3x3/github/ground_truth/Indoor) into the benchmark label
format aa-bench-players@1 (data/benchmark/README.md) -> data/benchmark/labels/labellerB/players_trackid3x3_seg_<video>.json.

GT facts (checked against the repo's script/Indoor/create_minimap.ipynb and frame 1 of basket_S1T1_pre):
- MOT/<video>.txt: frame (1-based = video frame index + 1), id 1..6, x, y, w, h (image px of the 1280x720 video).
- transformed_MOT/<video>_color.txt: frame, id, X, Y, bib colour; X = cm from the baseline (0..950), Y = cm across the
  court (0..1505). Ids 1-3 = offence (bibs red/blue/pink), 4-6 = defence (black/orange/yellow): bibs are per PLAYER,
  so the team label here is "offence"/"defence", not a kit colour.
- Only the 6 on-court players are annotated (spectators/coaches/camera operator are not) -> complete=False.
Positions: `court_tid` = [X, Y] in metres in the TrackID frame (baseline at X=0, half court 9.50 x 15.05 m); this is an
extra key, the benchmark's `court` field is NOT filled because the output's court frame depends on the clip calibration.
Usage: python tools/labellerB/trackid3x3_to_bench.py [--every 10]
"""
import argparse, json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GT = ROOT / "data/external/trackid3x3/github/ground_truth/Indoor"
OUT = ROOT / "data/benchmark/labels/labellerB"


def convert(video: str, every: int) -> dict:
    boxes = defaultdict(dict)
    for line in (GT / "MOT" / f"{video}.txt").read_text(encoding="utf-8").split():
        p = line.split(",")
        f, i = int(float(p[0])), int(float(p[1]))
        x, y, w, h = (float(v) for v in p[2:6])
        boxes[f][i] = [round(x, 1), round(y, 1), round(x + w, 1), round(y + h, 1)]
    pos, bib = defaultdict(dict), {}
    tp = GT / "transformed_MOT" / f"{video}_color.txt"
    if tp.exists():
        for line in tp.read_text(encoding="utf-8").split():
            p = line.split(",")
            f, i = int(float(p[0])), int(float(p[1]))
            pos[f][i] = [round(float(p[2]) / 100, 3), round(float(p[3]) / 100, 3)]
            bib.setdefault(i, p[4] if len(p) > 4 else None)
    frames = []
    for f in sorted(boxes)[::every]:
        people = []
        for i, b in sorted(boxes[f].items()):
            person = {"box": b, "kind": "player", "team": "offence" if i <= 3 else "defence",
                      "pid": f"tid-{i}-{bib.get(i) or i}", "bib": bib.get(i)}
            if i in pos.get(f, {}):
                person["court_tid"] = pos[f][i]
                person["court"] = pos[f][i]  # same frame as calibration_gt.json (x = X, y = Y, court 28 x 15.05)
            people.append(person)
        frames.append({"frame": f - 1, "status": "done", "complete": False, "people": people})
    return {"format": "aa-bench-players@1", "clip": f"trackid3x3_seg_{video}", "labeller": "labellerB (TrackID3x3 GT)",
            "notes": "converted from TrackID3x3 Indoor GT (CC BY 4.0); only the 6 on-court players are annotated; "
                     "team = offence/defence (per-player bibs); court_tid = metres, baseline frame of the half court",
            "frames": frames}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=int, default=10)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in sorted((GT / "MOT").glob("*.txt")):
        doc = convert(p.stem, a.every)
        (OUT / f"players_trackid3x3_seg_{p.stem}.json").write_text(json.dumps(doc, indent=1), encoding="utf-8")
        n += len(doc["frames"])
    print("videos", len(list((GT / "MOT").glob("*.txt"))), "frames", n)


if __name__ == "__main__":
    main()
