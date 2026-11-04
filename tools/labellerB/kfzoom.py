"""Crop + 2x enlarge a rendered keyframe: python tools/labellerB/kfzoom.py <kf jpg> x1,y1,x2,y2 [more boxes...] -> scratch/kz.jpg
(several regions are stacked side by side)."""
import sys
import cv2
import numpy as np
img = cv2.imread(sys.argv[1]); parts = []
for spec in sys.argv[2:]:
    x1, y1, x2, y2 = (int(v) for v in spec.split(","))
    c = img[y1:y2, x1:x2]
    s = 520 / c.shape[0]
    parts.append(cv2.resize(c, None, fx=s, fy=s))
out = np.hstack(parts)
if out.shape[1] > 1900:
    out = cv2.resize(out, None, fx=1900 / out.shape[1], fy=1900 / out.shape[1])
p = r"C:/Users/onnik/AppData/Local/Temp/claude/C--Users-onnik-Desktop/032042cc-6aa6-4bc1-897f-cbf1d1dcb2bf/scratchpad/kz.jpg"
cv2.imwrite(p, out); print(p)
