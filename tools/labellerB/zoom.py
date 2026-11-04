"""Crop + enlarge a region of a clip's video frame with selected candidate boxes drawn (thin, coloured, labelled).
Usage: python tools/labellerB/zoom.py <clip> <frame> x1,y1,x2,y2 [refs,comma] -> scratch image path printed"""
import json, sys
from pathlib import Path
import cv2
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
clip, frame = sys.argv[1], int(sys.argv[2])
x1, y1, x2, y2 = (int(v) for v in sys.argv[3].split(","))
want = set(sys.argv[4].split(",")) if len(sys.argv) > 4 else None
d = json.loads((ROOT / f"data/benchmark/labels/drafts/players_{clip}.json").read_text(encoding="utf-8"))
f = next(fr for fr in d["frames"] if fr["frame"] == frame)
from replay.benchmark import clip_registry
out = Path(clip_registry()[clip]["src"])
video = json.loads((out / "work/run.json").read_text(encoding="utf-8"))["video"]
cap = cv2.VideoCapture(str(ROOT / video) if not Path(video).is_absolute() else video)
cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
ok, img = cap.read()
cols = [(0, 255, 255), (255, 0, 255), (0, 255, 0), (255, 128, 0), (0, 128, 255), (255, 255, 255), (128, 0, 255)]
k = 0
for c in f["people"] + f.get("candidates", []):
    if want and c["ref"] not in want:
        continue
    b = [int(v) for v in c["box"]]
    col = cols[k % len(cols)]; k += 1
    cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), col, 1)
    cv2.putText(img, c["ref"], (b[0] + 2, b[1] + 12), 0, 0.45, col, 1)
crop = img[max(0, y1):y2, max(0, x1):x2]
s = min(1280 / crop.shape[1], 900 / crop.shape[0])
crop = cv2.resize(crop, None, fx=s, fy=s)
p = Path(r"C:/Users/onnik/AppData/Local/Temp/claude/C--Users-onnik-Desktop/032042cc-6aa6-4bc1-897f-cbf1d1dcb2bf/scratchpad/zoom.jpg")
cv2.imwrite(str(p), crop); print(p)
