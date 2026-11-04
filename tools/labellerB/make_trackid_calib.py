"""Write a GT calibration file (fixed camera, one keyframe) for processed TrackID3x3 Indoor clips from the exact GT
homography (data/external/trackid3x3/gt_homography.json). Court frame: TrackID X (m from baseline) -> our x, Y -> our y,
court 28 x 15.05 m (TrackID width), FIBA markings; only the near half is visible. Our court keypoints in the visible
half are projected into the image with the inverse GT homography (points outside the frame are dropped).
Output: data/external/trackid3x3/seg_<video>/calibration_gt.json (the failed auto calibration stays absent).
Usage: python tools/labellerB/make_trackid_calib.py"""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT))
from replay import court
from replay.calibration import CalibrationFile, Keyframe

H = np.array(json.loads((ROOT / "data/external/trackid3x3/gt_homography.json").read_text(encoding="utf-8"))["H_img_to_m"])
Hi = np.linalg.inv(H)
for d in sorted((ROOT / "data/external/trackid3x3").glob("seg_*")):
    if not d.is_dir():
        continue
    court.configure(28.0, 15.05, "FIBA")
    pts = {}
    for n, (x, y) in court.KEYPOINTS.items():
        if x > 9.6:
            continue
        q = Hi @ np.array([x, y, 1.0])
        u, v = q[0] / q[2], q[1] / q[2]
        if -5 <= u <= 1285 and -5 <= v <= 725:
            pts[n] = (float(u), float(v))
    cf = CalibrationFile(video=str(d.with_suffix(".mp4").relative_to(ROOT)), frame_size=(1280, 720), fixed_camera=True,
                         court={"length": 28.0, "width": 15.05, "spec": "FIBA", "source": "TrackID3x3 GT homography"})
    kf = Keyframe(time_s=0.0, points=pts, source="manual")
    kf.fit(cf.frame_size)
    cf.keyframes.append(kf)
    cf.save(d / "calibration_gt.json")
    print(d.name, len(pts), "points, reprojection px", round(kf.reproj_px, 3), sorted(pts))
