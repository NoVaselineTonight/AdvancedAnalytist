"""Pick burst frames (pass-1 sample frames) for hand labelling: bursts at the given seconds, 3 frames ~1.5 s apart.
Usage: python tools/labellerB/pick_frames.py <out dir> [t1,t2,...]"""
import bisect, json, sys
from pathlib import Path
out = Path(sys.argv[1])
rd = json.loads((out / "replay_data.json").read_text(encoding="utf-8"))
fr, t = rd["frame"], rd["t"]
starts = [float(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2 else "20,65,110,155".split(","))]
pick = []
for s in starts:
    for k in range(3):
        j = min(bisect.bisect_left(t, s + 1.5 * k), len(t) - 1)
        pick.append(fr[j])
print(",".join(str(f) for f in sorted(set(pick))))
