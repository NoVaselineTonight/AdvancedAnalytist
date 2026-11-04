"""Start-up self-check (well under 1 s; nothing heavy is imported).

    from replay.selfcheck import startup_check, banner_text
    checks = startup_check(projects_dir)     # runs the checks, writes warnings to the log (stdout -> app.log)
    text = banner_text(checks)               # "" when all is well, else one short line for a non-blocking banner

Checks: GPU runtime (ONNX Runtime with DirectML), model files, ffmpeg, free disk space (>= 10 GB).
The GPU check looks at the installed package instead of importing onnxruntime (that alone takes 3+ s); pass
``deep=True`` to ask ONNX Runtime itself (slow).
"""

from __future__ import annotations

import datetime as _dt
import importlib.util
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MIN_FREE_GB = 10.0
LOW_FREE_GB = 2.0
OK, WARN, ERROR = "ok", "warn", "error"


@dataclass
class Check:
    name: str  # "gpu" | "models" | "ffmpeg" | "disk"
    level: str  # ok | warn | error
    message: str


# (relative path, what it is for, level when missing)
def model_files(variant: str = "medium") -> list[tuple[str, str, str]]:
    return [
        (f"models/rfdetr-{variant}/rfdetr-{variant}.onnx", f"player/ball detector ({variant})", ERROR),
        ("models/reid/reid.onnx", "player re-identification", WARN),
        ("models/pp-ocrv4-mobile-rec/pp-ocrv4-mobile-rec.onnx", "jersey numbers", WARN),
        ("models/clip_b32_laion/vision.onnx", "referee / role detection", WARN),
        ("data/courtnet/courtnet.onnx", "automatic court calibration", WARN),
    ]


def _ort_dists(site: Path) -> dict[str, str]:
    """{distribution name: version} of the ONNX Runtime builds in ``site`` (a glob: importlib.metadata needs ~1 s)."""
    out = {}
    for d in site.glob("onnxruntime*.dist-info"):
        name, _, ver = d.name[:-len(".dist-info")].partition("-")
        out[name.replace("_", "-").lower()] = ver
    return out


def check_gpu(deep: bool = False) -> Check:
    if deep or "onnxruntime" in sys.modules:
        try:
            import onnxruntime as ort

            prov = ort.get_available_providers()
        except Exception as e:  # noqa: BLE001
            return Check("gpu", ERROR, f"ONNX Runtime does not load ({type(e).__name__}: {e}). Analysis cannot run.")
        if "DmlExecutionProvider" in prov or "CUDAExecutionProvider" in prov:
            return Check("gpu", OK, f"GPU runtime available ({', '.join(prov)}).")
        return Check("gpu", WARN, "No GPU runtime (DirectML) found: analysis runs on the CPU and is much slower.")
    spec = importlib.util.find_spec("onnxruntime")
    if spec is None or not spec.submodule_search_locations:
        return Check("gpu", ERROR, "ONNX Runtime is not installed: analysis cannot run.")
    pkg = Path(list(spec.submodule_search_locations)[0])
    dml = (pkg / "capi" / "DirectML.dll").exists()
    has = _ort_dists(pkg.parent)
    if "onnxruntime-directml" in has and len(has) > 1:
        others = ", ".join(n for n in has if n != "onnxruntime-directml")
        return Check("gpu", WARN, f"Both onnxruntime-directml and {others} are installed; they overwrite each "
                                  "other and the GPU may not be used. Keep only onnxruntime-directml.")
    if dml or "onnxruntime-gpu" in has:
        return Check("gpu", OK, f"GPU runtime installed (onnxruntime-directml {has.get('onnxruntime-directml', '?')}).")
    return Check("gpu", WARN, "No GPU runtime (DirectML) installed: analysis runs on the CPU and is much slower.")


def check_models(root: Path = ROOT, variant: str = "medium") -> list[Check]:
    out = []
    for rel, what, level in model_files(variant):
        p = root / rel
        try:
            ok = p.is_file() and p.stat().st_size > 0
        except OSError:
            ok = False
        if not ok:
            out.append(Check("models", level, f"Model file missing: {rel} ({what})."))
    if not out:
        out.append(Check("models", OK, "All model files present."))
    return out


def check_ffmpeg() -> Check:
    exe = os.environ.get("IMAGEIO_FFMPEG_EXE")
    if exe and Path(exe).is_file():
        return Check("ffmpeg", OK, f"ffmpeg: {exe}")
    spec = importlib.util.find_spec("imageio_ffmpeg")
    if spec is not None and spec.submodule_search_locations:
        b = Path(list(spec.submodule_search_locations)[0]) / "binaries"
        hits = sorted(b.glob("ffmpeg*.exe" if os.name == "nt" else "ffmpeg*")) if b.is_dir() else []
        if hits:
            return Check("ffmpeg", OK, f"ffmpeg: {hits[-1]}")
    exe = shutil.which("ffmpeg")
    if exe:
        return Check("ffmpeg", OK, f"ffmpeg: {exe}")
    return Check("ffmpeg", ERROR, "ffmpeg was not found: videos cannot be imported or exported.")


def check_disk(path: Path, min_gb: float = MIN_FREE_GB) -> Check:
    p = Path(path)
    while not p.exists() and p.parent != p:
        p = p.parent
    try:
        free = shutil.disk_usage(p).free / 1024 ** 3
    except OSError as e:
        return Check("disk", WARN, f"Free disk space could not be checked ({e}).")
    if free < LOW_FREE_GB:
        return Check("disk", ERROR, f"Only {free:.1f} GB free on the projects drive: analysis will fail. "
                                    f"Free at least {min_gb:.0f} GB.")
    if free < min_gb:
        return Check("disk", WARN, f"Only {free:.1f} GB free on the projects drive (at least {min_gb:.0f} GB "
                                   "recommended for a full game).")
    return Check("disk", OK, f"{free:.0f} GB free.")


def run_checks(projects_dir: Path | None = None, root: Path = ROOT, variant: str = "medium",
               deep: bool = False) -> list[Check]:
    projects_dir = Path(projects_dir) if projects_dir else root / "projects"
    out: list[Check] = []
    for fn in (lambda: [check_gpu(deep)], lambda: check_models(root, variant), lambda: [check_ffmpeg()],
               lambda: [check_disk(projects_dir)]):
        try:
            out.extend(fn())
        except Exception as e:  # noqa: BLE001 - a broken check must not stop the app
            out.append(Check("selfcheck", WARN, f"A start-up check failed ({type(e).__name__}: {e})."))
    return out


def problems(checks: list[Check]) -> list[Check]:
    return [c for c in checks if c.level != OK]


def messages(checks: list[Check]) -> list[str]:
    """The warning / error texts (for a banner or a tooltip); [] when all is well."""
    return [("Error: " if c.level == ERROR else "Warning: ") + c.message for c in problems(checks)]


def banner_text(checks: list[Check]) -> str:
    """One line for a non-blocking banner; "" when nothing is wrong. Errors first."""
    bad = sorted(problems(checks), key=lambda c: c.level != ERROR)
    if not bad:
        return ""
    head = bad[0].message
    return head if len(bad) == 1 else f"{head} (+{len(bad) - 1} more start-up warning{'s' if len(bad) > 2 else ''}" \
                                      f" - see app.log)"


def log_checks(checks: list[Check], stream=None) -> None:
    stream = stream if stream is not None else sys.stdout
    if stream is None:
        return
    stamp = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        for c in checks:
            if c.level != OK:
                stream.write(f"{stamp} [self-check] {c.level.upper()}: {c.message}\n")
        if not problems(checks):
            stream.write(f"{stamp} [self-check] OK (GPU runtime, models, ffmpeg, disk space)\n")
        stream.flush()
    except Exception:  # noqa: BLE001
        pass


def startup_check(projects_dir: Path | None = None, variant: str = "medium", log: bool = True) -> list[Check]:
    t0 = time.perf_counter()
    checks = run_checks(projects_dir, variant=variant)
    if log:
        log_checks(checks)
        if time.perf_counter() - t0 > 1.0 and sys.stdout is not None:
            print(f"[self-check] took {time.perf_counter() - t0:.1f} s")
    return checks


if __name__ == "__main__":  # python -m replay.selfcheck [projects_dir]
    cs = run_checks(Path(sys.argv[1]) if len(sys.argv) > 1 else None, deep="--deep" in sys.argv)
    for c in cs:
        print(f"{c.level.upper():5}  {c.name:7}  {c.message}")
    print("banner:", banner_text(cs) or "(none)")
