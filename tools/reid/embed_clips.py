"""Compute re-ID embedding caches for the benchmark clips (never writes into the original clip folders).

The cache is computed on the clip's pass-1 work (read only) and written to data/benchmark/cache/<clip>/
reid_embeddings.npz - replay.benchmark copies that folder into each fresh run's work/ (the cache key ties it to
the pass-1 rows + model, so a stale file is ignored and recomputed by load_or_compute).
Also writes a copy to --also-into <dir>/<clip>/ if given.

Usage: python tools/reid/embed_clips.py [--clips c300,p4400] [--model models/reid/reid.onnx]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from replay import reid  # noqa: E402
from replay.benchmark import CACHE, clip_registry  # noqa: E402
from replay.pipeline import load_pass1  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default=None, help="comma-separated (default: koristv clips p4400, b3min, c*)")
    ap.add_argument("--model", default=str(reid.MODEL_PATH))
    args = ap.parse_args()
    reg = clip_registry()
    names = args.clips.split(",") if args.clips else [n for n in reg if n in ("p4400", "b3min") or n.startswith("c")]
    cfg = reid.ReidConfig(enabled=True, model_path=args.model)
    summary = {}
    for name in names:
        info = reg[name]
        src = Path(info["src"]) / "work"
        if not (src / "DONE").exists():
            print(f"{name}: no finished pass 1, skipped", flush=True)
            continue
        data = load_pass1(src)
        video = Path(data.signature.get("video") or "")
        if not video.is_absolute():
            video = ROOT / video
        dst = CACHE / name
        dst.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        with tempfile.TemporaryDirectory() as tmp:  # compute in a scratch work dir (load_or_compute writes there)
            tw = Path(tmp)
            if (dst / reid.CACHE_NAME).exists():
                shutil.copy2(dst / reid.CACHE_NAME, tw / reid.CACHE_NAME)
            res = reid.load_or_compute(tw, data, video if video.exists() else None, cfg)
            if res is None:
                print(f"{name}: video {video} missing, skipped", flush=True)
                continue
            shutil.copy2(tw / reid.CACHE_NAME, dst / reid.CACHE_NAME)
        summary[name] = {"boxes": int(res.sampled.sum()), "rows": len(data.rows), "s": round(time.perf_counter() - t0, 1),
                         **{k: res.stats.get(k) for k in ("device", "frames")}}
        print(name, json.dumps(summary[name]), flush=True)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
