"""Video import: probe any ffmpeg-readable file and create a browser-playable H.264 MP4.

The playback copy ("proxy") is also what gets analysed, so replay timestamps and
video playback times are guaranteed to match.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import imageio_ffmpeg

# Everything ffmpeg can demux works; this list only drives the file picker filter.
VIDEO_EXTENSIONS = [
    ".mp4", ".m4v", ".mov", ".qt", ".mkv", ".webm", ".avi", ".wmv", ".asf", ".flv", ".f4v", ".ts", ".mts",
    ".m2ts", ".m2t", ".mpg", ".mpeg", ".mpe", ".vob", ".3gp", ".3g2", ".ogv", ".mxf", ".dv", ".h264", ".hevc",
    ".265", ".y4m", ".divx", ".rm", ".rmvb",
]
MAX_PROXY_HEIGHT = 1080  # 4K is downscaled: faster analysis, and the detector doesn't benefit much


class Cancelled(Exception):
    pass


@dataclass
class ProbeResult:
    duration_s: float
    codec: str
    width: int
    height: int
    fps: float
    has_audio: bool


# ffmpeg must not read the keyboard / stdin (a stray key or a closed stdin can stop it silently), must not open a
# console window, and must not receive the Ctrl+C / close events of a console the app was started from.
_DETACH = {"stdin": subprocess.DEVNULL,
           "creationflags": (subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP)
           if sys.platform == "win32" else 0}


def ffmpeg_exe() -> str:
    return imageio_ffmpeg.get_ffmpeg_exe()


def probe(path: str | Path) -> ProbeResult:
    proc = subprocess.run([ffmpeg_exe(), "-hide_banner", "-nostdin", "-i", str(path)], capture_output=True, text=True,
                          encoding="utf-8", errors="replace", **_DETACH)
    err = proc.stderr
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", err)
    duration = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3)) if m else 0.0
    v = re.search(r"Stream #\S+.*?Video:\s*([a-zA-Z0-9_]+)[^\n]*?(\d{2,5})x(\d{2,5})", err)
    if not v:
        raise ValueError("No video stream found - is this a video file?")
    fps_m = re.search(r"Video:[^\n]*?(\d+(?:\.\d+)?)\s*fps", err)
    return ProbeResult(duration, v.group(1).lower(), int(v.group(2)), int(v.group(3)),
                       float(fps_m.group(1)) if fps_m else 0.0, bool(re.search(r"Stream #\S+.*?Audio:", err)))


def _clock(s: float) -> str:
    s = max(0.0, float(s))
    return f"{int(s // 3600)}:{int(s // 60) % 60:02d}:{int(s % 60):02d}"


def failure_message(code: int, stderr: str, reached_s: float, info: ProbeResult) -> str:
    """Never empty: what ffmpeg said (last lines), how far it got and its exit code (Windows crash codes in hex)."""
    code32 = code & 0xFFFFFFFF
    known = {0xC0000005: "ffmpeg crashed (access violation)", 0xC0000409: "ffmpeg crashed (stack buffer overrun)",
             0xC0000017: "out of memory", 0xC000013A: "ffmpeg was interrupted (Ctrl+C or the console was closed)",
             0xC00000FD: "ffmpeg crashed (stack overflow)"}
    why = known.get(code32, "")
    code_txt = f"0x{code32:08X}" if code32 >= 0x80000000 else str(code)
    tail = " | ".join(ln.strip() for ln in stderr.strip().splitlines()[-6:] if ln.strip())[-500:]
    where = (f"after converting {_clock(reached_s)} of {_clock(info.duration_s)}" if reached_s > 0
             else "before converting anything")
    msg = (f"The video could not be converted ({info.codec} {info.width}x{info.height}): ffmpeg stopped {where} "
           f"with exit code {code_txt}" + (f" - {why}" if why else "") + ".")
    msg += f" ffmpeg said: {tail}." if tail else " ffmpeg gave no reason."
    return msg + " Press Retry import to try again."


_CPU_ENCODER = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-g", "250", "-pix_fmt", "yuv420p"]
# AMD GPU encoder (RX 6000 etc.): takes the encoding off the CPU (the CPU then only decodes; measured 18.10 on an AV1
# 1080p source: about the same wall time as libx264 veryfast, but the CPU stays free for analysis). Peak-limited VBR
# keeps files near libx264 crf 20 size; same keyframe interval so seeking in the player is as fast as before.
_GPU_ENCODERS = {"h264_amf": ["-c:v", "h264_amf", "-quality", "balanced", "-rc", "vbr_peak", "-b:v", "8M",
                              "-maxrate", "12M", "-g", "250", "-pix_fmt", "yuv420p"]}
_gpu_ok: dict[str, bool] = {}
# hardware decoding (D3D11, any vendor) straight into GPU frames for the GPU encoder; ffmpeg's own error -> next attempt
_GPU_DECODE = ["-hwaccel", "d3d11va", "-hwaccel_output_format", "d3d11", "-extra_hw_frames", "24"]


def gpu_encoder() -> list[str] | None:
    """Encoder args of a working GPU H.264 encoder, or None (no AMD GPU / driver, or AA_GPU_ENCODE=0).

    Each encoder is tried once per process on a 1-second test picture, so a missing driver costs well under a second.
    """
    if os.environ.get("AA_GPU_ENCODE", "1") == "0":
        return None
    for name, args in _GPU_ENCODERS.items():
        if name not in _gpu_ok:
            try:
                r = subprocess.run([ffmpeg_exe(), "-hide_banner", "-nostdin", "-loglevel", "error", "-f", "lavfi",
                                    "-i", "testsrc2=size=1280x720:rate=30", "-t", "1", *args, "-f", "null", "-"],
                                   capture_output=True, timeout=30, **_DETACH)
                _gpu_ok[name] = r.returncode == 0
            except Exception:  # any trouble starting the test encode -> CPU encoder
                _gpu_ok[name] = False
        if _gpu_ok[name]:
            return args
    return None


def make_proxy(src: str | Path, dst: str | Path, progress: Callable[[float], None] | None = None,
               cancel: threading.Event | None = None) -> ProbeResult:
    """Write an H.264/AAC MP4 with timestamps starting at 0.

    H.264 sources at <= 1080p are remuxed without re-encoding (fast, lossless). Everything else is re-encoded on
    the GPU when a GPU encoder works, otherwise (or if the GPU conversion fails) with libx264 on the CPU.
    """
    final = Path(dst)
    # Write under a temporary name: nothing (player, frame reader) may open a half-written file.
    dst = final.with_name(final.stem + ".importing" + final.suffix)
    info = probe(src)
    if info.codec == "h264" and info.height <= MAX_PROXY_HEIGHT:
        _convert(src, dst, info, ["-c:v", "copy"], progress, cancel)
    else:
        vf = f"scale=-2:{MAX_PROXY_HEIGHT}" if info.height > MAX_PROXY_HEIGHT else "scale=trunc(iw/2)*2:trunc(ih/2)*2"
        attempts: list[tuple[list[str], list[str]]] = []  # (input args, video args), fastest first
        gpu = gpu_encoder()
        if gpu is not None:
            if (sys.platform == "win32" and info.height <= MAX_PROXY_HEIGHT and info.width % 2 == 0
                    and info.height % 2 == 0 and os.environ.get("AA_GPU_DECODE", "1") != "0"):
                # decode on the GPU too and keep the frames there (the bundled ffmpeg has only the slow libaom AV1
                # software decoder): measured 18.10 on the AV1 1080p NCAA game 4.3x real time vs 1.6x
                attempts.append((_GPU_DECODE, [a for a in gpu if a not in ("-pix_fmt", "yuv420p")]))
            attempts.append(([], ["-vf", vf, *gpu]))
        attempts.append(([], ["-vf", vf, *_CPU_ENCODER]))
        for n, (in_args, video_args) in enumerate(attempts):
            try:
                _convert(src, dst, info, video_args, progress, cancel, in_args)
                break
            except RuntimeError:
                if n == len(attempts) - 1:
                    raise
                if progress:
                    progress(0.0)  # next (slower but safer) way from the start
    Path(dst).replace(final)
    return probe(final)


def _convert(src: str | Path, dst: Path, info: ProbeResult, video_args: list[str],
             progress: Callable[[float], None] | None, cancel: threading.Event | None,
             in_args: list[str] | None = None) -> None:
    # "warning": when ffmpeg stops, the last lines say why (only the tail is kept, see _drain)
    cmd = [ffmpeg_exe(), "-hide_banner", "-nostdin", "-loglevel", "warning", "-y", "-fflags", "+genpts",
           *(in_args or []), "-i", str(src),
           "-map", "0:v:0", "-map", "0:a:0?", "-sn", "-dn", *video_args]
    cmd += ["-c:a", "aac", "-b:a", "128k", "-avoid_negative_ts", "make_zero", "-movflags", "+faststart",
            "-progress", "pipe:1", "-nostats", str(dst)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8",
                            errors="replace", **_DETACH)
    assert proc.stdout is not None
    # stderr is drained in the background: broadcast captures (.ts) can print thousands of decoder warnings, and
    # a full stderr pipe would block ffmpeg (and the import) forever
    err_tail: list[str] = []

    def _drain() -> None:
        for ln in proc.stderr:  # type: ignore[union-attr]
            err_tail.append(ln)
            if len(err_tail) > 200:
                del err_tail[:100]

    drain = threading.Thread(target=_drain, daemon=True)
    drain.start()
    reached = 0.0  # seconds of video converted so far (for the error message)
    for line in proc.stdout:
        if cancel is not None and cancel.is_set():
            proc.kill()
            proc.wait()
            Path(dst).unlink(missing_ok=True)
            raise Cancelled()
        if line.startswith("out_time_us="):
            try:
                reached = max(reached, int(line.split("=")[1]) / 1e6)
            except ValueError:
                continue
            if progress and info.duration_s > 0:
                progress(min(1.0, reached / info.duration_s))
    proc.wait()
    drain.join(timeout=5)
    if proc.returncode != 0:
        Path(dst).unlink(missing_ok=True)
        raise RuntimeError(failure_message(proc.returncode, "".join(err_tail), reached, info))
