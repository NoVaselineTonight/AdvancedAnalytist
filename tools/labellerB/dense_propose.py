"""Propose identities for the next keyframe from ByteTrack tracklet continuity and draw them for checking.
python tools/labellerB/dense_propose.py <clip_dir> <codes.json> <k>  -> scratch/dense/<clip>/pr_<k>.jpg + prints proposal JSON
Proposal: a box whose tracklet carried pid X at an earlier labelled keyframe (latest wins) gets X; others '?'.
Labels drawn: 'P<n>:<pid>' (green = proposed, red = unknown)."""
import json, sys
from pathlib import Path
import cv2, numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "labellerB"))
from dense_keyframes import SCRATCH, _load, people_at, ball_at
clip_dir, codes_p, k = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
d, video = _load(clip_dir)
C = json.loads(codes_p.read_text(encoding="utf-8")) if codes_p.exists() else {"kf": {}}
t2pid = {}
for kk in sorted(int(x) for x in C["kf"] if int(x) < k):
    ppl = people_at(d, kk)
    for ref, code in C["kf"][str(kk)].items():
        if ref in ppl and int(ppl[ref][7]) >= 0:
            t2pid[int(ppl[ref][7])] = code
img = cv2.imread(str(SCRATCH / clip_dir.name / f"kf_{k:04d}.jpg"))
prop = {}
for name, row in people_at(d, k).items():
    pid = t2pid.get(int(row[7]), "?") if int(row[7]) >= 0 else "?"
    prop[name] = pid
    x1, y1, x2, y2 = (int(c) for c in row[1:5])
    col = (0, 255, 0) if pid != "?" else (0, 0, 255)
    cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
    cv2.putText(img, f"{name[1:]}:{pid}", (x1, min(1075, y2 + 16)), 0, 0.55, col, 2)
cv2.imwrite(str(SCRATCH / clip_dir.name / f"pr_{k:04d}.jpg"), img)
print(json.dumps(prop))
