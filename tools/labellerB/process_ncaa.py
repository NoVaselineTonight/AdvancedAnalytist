"""Run `replay process` on the NCAA clips one at a time (data/external/ncaa/seg_*.mp4). Progress: data/external/ncaa/progress.txt."""
import os, subprocess, sys
from pathlib import Path
root = Path("data/external/ncaa"); prog = root / "progress.txt"
done = set(prog.read_text(encoding="utf-8").split("\n")) if prog.exists() else set()
for v in sorted(root.glob("seg_*.mp4")):
    d = v.with_suffix(""); key = f"done {d.as_posix()}"
    if key in done:
        continue
    d.mkdir(exist_ok=True)
    with open(d / "process.log", "w", encoding="utf-8") as log:
        r = subprocess.run([sys.executable, "-m", "replay", "process", "--video", str(v), "--out", str(d / "out"),
                            "--calibration", str(d / "calibration.json")], stdout=log, stderr=subprocess.STDOUT,
                           env=dict(os.environ, PYTHONUTF8="1"))
    with open(prog, "a", encoding="utf-8") as f:
        f.write(f"{key}\n" if r.returncode == 0 else f"failed {d.as_posix()} rc={r.returncode}\n")
with open(prog, "a", encoding="utf-8") as f:
    f.write("ALL\n")
