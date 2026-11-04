"""Speed-agent changes must not change results: parallel pass-1 loading, single JSON serialisation."""

import json
from pathlib import Path

import numpy as np

from replay.pipeline import load_pass1


def _write_chunk(work: Path, c: int, k0: int, rng: np.random.Generator, nan_motion: bool) -> None:
    n = 5
    motion = rng.normal(size=(n, 3, 3))
    if nan_motion:
        motion[2] = np.nan
    rows = rng.normal(size=(7, 14))
    rows[:, 0] = k0 + rng.integers(0, n, 7)
    np.savez_compressed(work / f"chunk_{c:05d}.npz", k=np.arange(k0, k0 + n), frame=np.arange(k0, k0 + n) * 3,
                        t=np.arange(k0, k0 + n) / 10.0, motion=motion, inliers=np.arange(n), rows=rows)
    (work / f"state_{c:05d}.pkl").write_bytes(b"")


def test_load_pass1_matches_serial(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    for c in range(12):
        _write_chunk(tmp_path, c, c * 5, rng, nan_motion=c % 3 == 0)
    (tmp_path / "run.json").write_text(json.dumps({"process_fps": 10.0}), encoding="utf-8")
    d = load_pass1(tmp_path)
    parts = [np.load(tmp_path / f"chunk_{c:05d}.npz") for c in range(12)]
    for f in ("k", "frame", "t", "inliers", "rows"):
        ref = np.concatenate([p[f] for p in parts])
        assert np.array_equal(getattr(d, f), ref) and getattr(d, f).dtype == ref.dtype
    ref_m = [None if not np.isfinite(m).all() else m for p in parts for m in p["motion"]]
    assert len(d.motion) == len(ref_m)
    assert all((a is None and b is None) or np.array_equal(a, b) for a, b in zip(d.motion, ref_m))
    assert d.signature == {"process_fps": 10.0}


def test_frame_slices_match_mask() -> None:
    """postprocess step 5: searchsorted slices of argsort(det_i) == the old per-frame boolean mask, same order."""
    rng = np.random.default_rng(1)
    det_i = rng.integers(0, 300, 5000)
    order = np.argsort(det_i)
    sorted_i = det_i[order]
    for i in np.unique(det_i):
        a = order[(det_i[order] == i)]
        b = order[np.searchsorted(sorted_i, i):np.searchsorted(sorted_i, i, side="right")]
        assert np.array_equal(a, b)
