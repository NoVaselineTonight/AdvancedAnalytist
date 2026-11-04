"""Fit the image -> court homography of TrackID3x3 Indoor (fixed camera) from GT foot points (MOT box bottom-centre) and
GT court positions (transformed_MOT, cm). Prints the fit error per video and writes data/external/trackid3x3/gt_homography.json
(H maps image px -> metres in the TrackID frame: X from the baseline, Y across)."""
import json
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
GT = ROOT / "data/external/trackid3x3/github/ground_truth/Indoor"


def pairs(video):
    box = {}
    for l in (GT / "MOT" / f"{video}.txt").read_text(encoding="utf-8").split():
        p = [float(v) for v in l.split(",")[:6]]
        box[(int(p[0]), int(p[1]))] = (p[2] + p[4] / 2, p[3] + p[5])
    img, crt = [], []
    for l in (GT / "transformed_MOT" / f"{video}.txt").read_text(encoding="utf-8").split():
        p = l.split(",")
        k = (int(float(p[0])), int(float(p[1])))
        if k in box:
            img.append(box[k]); crt.append((float(p[2]) / 100, float(p[3]) / 100))
    return np.float32(img), np.float32(crt)


allI, allC, per = [], [], {}
for f in sorted((GT / "MOT").glob("*.txt")):
    I, C = pairs(f.stem)
    if len(I) < 8:
        continue
    H, _ = cv2.findHomography(I, C, 0)
    e = np.linalg.norm(cv2.perspectiveTransform(I[None], H)[0] - C, axis=1)
    per[f.stem] = {"n": len(I), "fit_err_m_median": round(float(np.median(e)), 3)}
    allI.append(I); allC.append(C)
I, C = np.concatenate(allI), np.concatenate(allC)
H, _ = cv2.findHomography(I, C, 0)
e = np.linalg.norm(cv2.perspectiveTransform(I[None], H)[0] - C, axis=1)
for v, r in per.items():
    Iv, Cv = pairs(v)
    r["global_err_m_median"] = round(float(np.median(np.linalg.norm(cv2.perspectiveTransform(Iv[None], H)[0] - Cv, axis=1))), 3)
print("global: n", len(I), "median err m", round(float(np.median(e)), 3), "p90", round(float(np.percentile(e, 90)), 3))
for v, r in per.items():
    print(v, r)
(ROOT / "data/external/trackid3x3/gt_homography.json").write_text(json.dumps(
    {"H_img_to_m": H.tolist(), "frame": "TrackID half court: X m from baseline (0..9.5), Y m across (0..15.05)",
     "global_median_err_m": round(float(np.median(e)), 3), "per_video": per}, indent=1), encoding="utf-8")
