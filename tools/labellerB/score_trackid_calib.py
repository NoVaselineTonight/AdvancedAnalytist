"""Score the automatic court calibration of processed TrackID3x3 clips against the exact GT homography
(data/external/trackid3x3/gt_homography.json, image px -> metres in TrackID's half-court frame).

For every analysed sample of a clip, the GT foot points of the labelled players (MOT box bottom-centre) and a grid of
floor pixels inside the GT half court are projected with the clip's resolved homography (image -> output court metres)
and with the GT homography. Output and GT court frames differ by a convention (which corner is the origin, axis order,
mirror), so the two point sets are aligned with the best rigid transform (rotation + translation, mirror allowed,
NO scale), fitted over the whole clip. Reported:
  rigid_err_m   median / p90 distance after that one alignment (= real calibration error, convention-free)
  scale         best similarity scale (1.0 = metric size right; < 1 = output court too small)
  valid_pct     samples with a homography
Usage: python tools/labellerB/score_trackid_calib.py [clip dirs...]   (default: data/external/trackid3x3/seg_*)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
GT = ROOT / "data/external/trackid3x3/github/ground_truth/Indoor"


def _proj(H, pts):
    p = np.c_[pts, np.ones(len(pts))] @ np.asarray(H, float).T
    return p[:, :2] / p[:, 2:3]


def _rigid(A, B, allow_scale=False):
    """Best B ~ s R A + t (R may include a mirror). Returns (residuals, s)."""
    ma, mb = A.mean(0), B.mean(0)
    a, b = A - ma, B - mb
    U, S, Vt = np.linalg.svd(a.T @ b)
    R = (U @ Vt).T
    s = S.sum() / (a ** 2).sum() if allow_scale else 1.0
    return np.linalg.norm((s * a @ R.T) - b, axis=1), s


def score(clip_dir: Path) -> dict:
    from replay import calibration as calib_mod
    from replay.config import Config
    from replay import court
    from replay.pipeline import load_pass1

    video = clip_dir.name[len("seg_"):]
    out = clip_dir / "out"
    cal_path = clip_dir / "calibration.json"
    if not (out / "work").exists() or not cal_path.exists():
        return {"clip": clip_dir.name, "error": "not processed / no calibration"}
    Hgt = np.array(json.loads((ROOT / "data/external/trackid3x3/gt_homography.json").read_text(encoding="utf-8"))["H_img_to_m"])
    data = load_pass1(out / "work")
    calib = calib_mod.CalibrationFile.load(cal_path)
    calib.apply_court()
    res = calib_mod.resolve(data.t, data.motion, calib, Config().calibration)
    # GT foot points per video frame (MOT frame - 1)
    feet: dict[int, list] = {}
    for l in (GT / "MOT" / f"{video}.txt").read_text(encoding="utf-8").split():
        p = [float(v) for v in l.split(",")[:6]]
        feet.setdefault(int(p[0]) - 1, []).append((p[2] + p[4] / 2, p[3] + p[5]))
    # fixed floor grid inside the GT half court (metres -> image via inverse GT H)
    gx, gy = np.meshgrid(np.linspace(0.5, 9.0, 8), np.linspace(0.5, 14.5, 12))
    grid_img = _proj(np.linalg.inv(Hgt), np.c_[gx.ravel(), gy.ravel()])
    A, B, valid = [], [], 0
    for i, f in enumerate(np.asarray(data.frame)):
        H = res.H[i] if i < len(res.H) else None
        if H is None:
            continue
        valid += 1
        pts = np.vstack([grid_img] + ([np.array(feet[int(f)])] if int(f) in feet else []))
        A.append(_proj(H, pts)); B.append(_proj(Hgt, pts))
    r = {"clip": clip_dir.name, "samples": len(data.t), "valid_pct": round(100 * valid / max(1, len(data.t)), 1),
         "court_lw": [court.LENGTH, court.WIDTH], "auto": bool(calib.auto)}
    if not A:
        return r
    A, B = np.vstack(A), np.vstack(B)
    ok = np.isfinite(A).all(1) & (np.abs(A) < 1e4).all(1)
    e, _ = _rigid(A[ok], B[ok])
    _, s = _rigid(A[ok], B[ok], allow_scale=True)
    r.update({"rigid_err_m_median": round(float(np.median(e)), 3), "rigid_err_m_p90": round(float(np.percentile(e, 90)), 3),
              "scale": round(float(s), 3), "n_points": int(ok.sum())})
    return r


def main():
    dirs = [Path(a) for a in sys.argv[1:]] or sorted(d for d in (ROOT / "data/external/trackid3x3").glob("seg_*") if d.is_dir())
    for d in dirs:
        try:
            print(json.dumps(score(d)))
        except Exception as ex:  # keep scoring the other clips
            print(json.dumps({"clip": d.name, "error": repr(ex)}))


if __name__ == "__main__":
    main()
