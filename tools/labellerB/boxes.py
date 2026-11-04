"""Print the O/D boxes of draft frames. Usage: python tools/labellerB/boxes.py <clip> [frames,comma]"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
d = json.loads((ROOT / f"data/benchmark/labels/drafts/players_{sys.argv[1]}.json").read_text(encoding="utf-8"))
want = set(int(v) for v in sys.argv[2].split(",")) if len(sys.argv) > 2 else None
for f in d["frames"]:
    if want and f["frame"] not in want:
        continue
    print("frame", f["frame"], "t", f.get("t"))
    for c in f["people"] + f.get("candidates", []):
        print("  ", c["ref"], [int(v) for v in c["box"]], c.get("kind", ""), c.get("team", ""), c.get("out_track", ""), c.get("conf", ""))
