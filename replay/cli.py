"""Command-line entry point: ``python -m replay <command> ...``"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from replay.config import Config, used_config_dict
from replay.metadata import load_metadata


def _time(s: str) -> float:
    """Seconds, mm:ss or hh:mm:ss."""
    out = 0.0
    for part in s.split(":"):
        out = out * 60 + float(part)
    return out


def cmd_calibrate(a: argparse.Namespace) -> None:
    from replay.calibrate_ui import CalibrationUI

    out = Path(a.out) if a.out else Path(a.video).with_suffix(".calibration.json")
    CalibrationUI(a.video, out, start_s=_time(a.start)).run()


def cmd_process(a: argparse.Namespace) -> None:
    from replay.pipeline import run_pass1

    cfg = Config.load(a.config)
    if a.stride:
        cfg.frame_stride = a.stride
    elif a.fps:
        cfg.frame_stride, cfg.process_fps = None, a.fps
    if a.device:
        cfg.detector.device = a.device
    if a.variant:
        cfg.detector.variant = a.variant
    if a.fixed_camera:
        cfg.motion.enabled = False
    if a.preliminary is not None:
        cfg.inplay.enabled = a.preliminary
    meta = load_metadata(a.meta)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.used.json").write_text(json.dumps(used_config_dict(cfg), indent=2), encoding="utf-8")
    end = _time(a.end) if a.end else (_time(a.start) + _time(a.duration) if a.duration else None)
    from replay.inplay import intervals_for_pass1

    intervals = intervals_for_pass1(Path(a.video), out, cfg.inplay, _time(a.start), end, restart=a.restart)
    run_pass1(Path(a.video), out, cfg, meta, _time(a.start), end, restart=a.restart, play_intervals=intervals)
    calib = Path(a.calibration) if a.calibration else Path(a.video).with_suffix(".calibration.json")
    if not calib.exists() and not _autocalibrate(Path(a.video), out, calib):
        print(f"\nThe court could not be found automatically.\nRun:  python -m replay calibrate --video {a.video}\n"
              f"then: python -m replay postprocess --out {out} --meta {a.meta} --calibration {calib}")
        return
    _postprocess(out, cfg, a.meta, calib, Path(a.video))


def _autocalibrate(video: Path, out: Path, calib: Path) -> bool:
    """Automatic court calibration from pass-1 results; writes the calibration file. False = not found."""
    import cv2

    from replay import autocourt
    from replay.calibration import CalibrationFile
    from replay.pipeline import load_pass1

    data = load_pass1(out / "work")
    cap = cv2.VideoCapture(str(video))
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    last = [-1.0]

    def prog(x: float, msg: str) -> None:
        if x - last[0] >= 0.05 or x >= 1.0:
            last[0] = x
            print(f"  [{x * 100:3.0f}%] {msg}")

    print("Calibrating the court automatically ...")
    try:
        det = Config.from_dict(json.loads((out / "config.used.json").read_text(encoding="utf-8"))).calibration.court_detector
    except Exception:  # noqa: BLE001
        det = "off"
    res = autocourt.calibrate_best(video, data, size, progress=prog, work=out / "work", detector=det)
    if res is None or res.good_frac < 0.05:
        return False
    cal = CalibrationFile.load(calib) if calib.exists() else CalibrationFile(video=video.name, frame_size=size)
    cal.court = {"length": round(res.length, 3), "width": round(res.width, 3), "spec": res.spec}
    cal.auto = res.to_json()
    cal.save(calib)
    print("Court found:", res.message)
    return True


def cmd_autocalibrate(a: argparse.Namespace) -> None:
    out = Path(a.out)
    calib = Path(a.calibration) if a.calibration else Path(a.video).with_suffix(".calibration.json")
    if not _autocalibrate(Path(a.video), out, calib):
        raise SystemExit("The court could not be found automatically; calibrate by hand (python -m replay calibrate).")
    print(f"Wrote {calib}")


def cmd_postprocess(a: argparse.Namespace) -> None:
    cfg = Config.load(a.config) if a.config else Config.from_dict(
        json.loads((Path(a.out) / "config.used.json").read_text(encoding="utf-8")))
    sig = json.loads((Path(a.out) / "work" / "run.json").read_text(encoding="utf-8"))
    video = Path(sig["video"])
    calib = Path(a.calibration) if a.calibration else video.with_suffix(".calibration.json")
    if not calib.exists():  # e.g. the automatic calibration did not find the court: say so instead of a traceback
        raise SystemExit(f"No court calibration at {calib} (the automatic calibration may not have found the court). "
                         f"Calibrate first: python -m replay calibrate --video {video} (or autocalibrate).")
    _postprocess(Path(a.out), cfg, a.meta, calib, video)


def _postprocess(out: Path, cfg: Config, meta_path: str | None, calib: Path, video: Path) -> None:
    from replay.pipeline import run_pass2

    rep = run_pass2(out, cfg, load_metadata(meta_path), calib, video)
    print(f"\nCalibration valid: {rep['calibration']['valid_pct']}% of frames; "
          f"avg tracked people/frame: {rep['people']['avg_tracked_per_calibrated_frame']}")
    print(f"Wrote {out / 'tracking.csv'}, tracking.json, replay.html, report.md")


def cmd_check(a: argparse.Namespace) -> None:
    from replay.debug_video import render_check_video

    out = Path(a.out)
    cfg = Config.from_dict(json.loads((out / "config.used.json").read_text(encoding="utf-8")))
    sig = json.loads((out / "work" / "run.json").read_text(encoding="utf-8"))
    calib = Path(a.calibration) if a.calibration else Path(sig["video"]).with_suffix(".calibration.json")
    dst = render_check_video(out, calib, cfg, out / "check.mp4", _time(a.max) if a.max else None)
    print(f"Wrote {dst}")


def cmd_clip(a: argparse.Namespace) -> None:
    """Cut a test clip and re-encode it as browser-playable H.264 MP4."""
    import imageio_ffmpeg

    ff = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [ff, "-hide_banner", "-loglevel", "error", "-y", "-ss", str(_time(a.start)), "-i", a.video,
           "-t", str(_time(a.duration)), "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", a.out]
    print("Encoding clip ...")
    subprocess.run(cmd, check=True)
    print(f"Wrote {a.out}")


def cmd_gui(a: argparse.Namespace) -> None:
    from replay.gui.app import run

    sys.exit(run(Path(a.projects)))


def cmd_preliminary(a: argparse.Namespace) -> None:
    """Only the fast preliminary analysis: print/save when the ball is in play."""
    from replay import inplay

    cfg = Config.load(a.config).inplay
    end = _time(a.end) if a.end else None
    res = inplay.run_preliminary(Path(a.video), cfg, _time(a.start), end)
    out = Path(a.out) if a.out else Path(a.video).with_suffix(".inplay.json")
    inplay.save_result(res, out, Path(a.video), cfg, end)
    total = res.end_s - res.start_s
    c = res.clock
    print(f"Done in {res.elapsed_s:.0f}s ({total / max(res.elapsed_s, 1e-6):.0f}x real time). "
          f"Scoreboard clock: {'found at ' + str(c.box_rel) + f', ticking {c.tick_rate:.0%}' if c.found else 'none'}")
    print(inplay.describe(res.intervals, total))
    for s, e in res.intervals:
        print(f"  {int(s // 60):3d}:{s % 60:04.1f} - {int(e // 60):3d}:{e % 60:04.1f}  ({e - s:5.0f} s)")
    print(f"Wrote {out}")


def cmd_info(a: argparse.Namespace) -> None:
    from replay.video import probe

    i = probe(a.video)
    print(f"{i.width}x{i.height} @ {i.fps:.3f} fps, {i.frame_count} frames, {i.duration_s / 60:.1f} min")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="replay", description="Basketball 2D replay from game video")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("calibrate", help="click court points to calibrate the camera (opens a window)")
    c.add_argument("--video", required=True)
    c.add_argument("--out", help="calibration JSON (default: <video>.calibration.json)")
    c.add_argument("--start", default="0", help="time to open at (s or mm:ss)")
    c.set_defaults(func=cmd_calibrate)

    c = sub.add_parser("autocalibrate", help="find the court automatically (after process / pass 1)")
    c.add_argument("--video", required=True)
    c.add_argument("--out", required=True, help="the --out folder of 'process'")
    c.add_argument("--calibration", help="calibration JSON to write (default: <video>.calibration.json)")
    c.set_defaults(func=cmd_autocalibrate)

    c = sub.add_parser("process", help="detect + track (resumable), then build outputs if calibrated")
    c.add_argument("--video", required=True)
    c.add_argument("--meta", help="game metadata JSON")
    c.add_argument("--out", required=True)
    c.add_argument("--calibration", help="calibration JSON (default: <video>.calibration.json)")
    c.add_argument("--config", help="JSON with config overrides")
    c.add_argument("--start", default="0")
    c.add_argument("--end")
    c.add_argument("--duration")
    c.add_argument("--stride", type=int, help="analyse every Nth video frame (default 3; 1 = every frame)")
    c.add_argument("--fps", type=float, help="analyse at a fixed rate instead of --stride")
    c.add_argument("--device", choices=["auto", "gpu", "cpu"], help="detector device (default auto = GPU if any)")
    c.add_argument("--variant", choices=["nano", "small", "medium", "large"], help="RF-DETR size")
    c.add_argument("--fixed-camera", action="store_true", help="camera never moves; skip motion estimation")
    c.add_argument("--restart", action="store_true", help="discard previous partial results in --out")
    c.add_argument("--preliminary", dest="preliminary", action="store_true", default=None,
                   help="run the preliminary analysis first and only detect/track while the ball is in play "
                        "(default: on, see config 'inplay')")
    c.add_argument("--no-preliminary", dest="preliminary", action="store_false",
                   help="analyse the whole video (no break skipping)")
    c.set_defaults(func=cmd_process)

    c = sub.add_parser("postprocess", help="re-run calibration/teams/smoothing/export without the video")
    c.add_argument("--out", required=True)
    c.add_argument("--meta")
    c.add_argument("--calibration")
    c.add_argument("--config")
    c.set_defaults(func=cmd_postprocess)

    c = sub.add_parser("check", help="render a QA video: court-line overlay, track IDs, mini-map")
    c.add_argument("--out", required=True)
    c.add_argument("--calibration")
    c.add_argument("--max", help="only the first N seconds")
    c.set_defaults(func=cmd_check)

    c = sub.add_parser("clip", help="cut a test clip to H.264 MP4 (browser-playable)")
    c.add_argument("--video", required=True)
    c.add_argument("--start", required=True)
    c.add_argument("--duration", required=True)
    c.add_argument("--out", required=True)
    c.set_defaults(func=cmd_clip)

    c = sub.add_parser("gui", help="start the desktop application")
    c.add_argument("--projects", default=str(Path(__file__).resolve().parent.parent / "projects"),
                   help="folder where projects are stored")
    c.set_defaults(func=cmd_gui)

    c = sub.add_parser("preliminary", help="fast preliminary analysis only: when is the ball in play?")
    c.add_argument("--video", required=True)
    c.add_argument("--out", help="JSON output (default: <video>.inplay.json)")
    c.add_argument("--config", help="JSON with config overrides (section 'inplay')")
    c.add_argument("--start", default="0")
    c.add_argument("--end")
    c.set_defaults(func=cmd_preliminary)

    c = sub.add_parser("info", help="print video properties")
    c.add_argument("--video", required=True)
    c.set_defaults(func=cmd_info)

    from replay.live import add_cli as _live_cli; _live_cli(sub)  # live hook: `replay live` (replay/live.py)  # noqa: E702

    if argv is None and sys.argv[1:2] == ["benchmark"] or argv and argv[:1] == ["benchmark"]:  # replay/benchmark.py
        from replay.benchmark import main as bench_main; return bench_main((sys.argv[2:] if argv is None else argv[1:]))  # noqa: E702
    a = p.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    sys.exit(main())
