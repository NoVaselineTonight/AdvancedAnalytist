"""Set / replace one keyframe's codes: python tools/labellerB/kfset.py <codes.json> <k> '<json dict>'"""
import json, sys
from pathlib import Path
p = Path(sys.argv[1]); C = json.loads(p.read_text(encoding="utf-8"))
C["kf"][sys.argv[2]] = json.loads(sys.argv[3])
C["kf"] = {k: C["kf"][k] for k in sorted(C["kf"], key=int)}
p.write_text(json.dumps(C, indent=1), encoding="utf-8"); print("kf", sys.argv[2], len(C["kf"][sys.argv[2]]), "entries;", len(C["kf"]), "keyframes")
