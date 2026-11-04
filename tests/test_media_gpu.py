"""Video import: GPU encoder for re-encoded sources, CPU fallback when the GPU path fails."""
import subprocess

import pytest

from replay.gui import media


def _make_mpeg4(path):
    subprocess.run([media.ffmpeg_exe(), "-hide_banner", "-nostdin", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    "testsrc2=size=640x360:rate=30", "-t", "1", "-c:v", "mpeg4", str(path)], check=True)


def test_reencode_uses_gpu_and_falls_back_to_cpu(tmp_path, monkeypatch):
    src = tmp_path / "in.avi"
    _make_mpeg4(src)
    used = []
    real = media._convert

    def fake(s, d, info, args, progress, cancel, in_args=None):
        enc = args[args.index("-c:v") + 1]
        used.append(enc)
        if enc == "h264_amf":
            raise RuntimeError("GPU failed")
        return real(s, d, info, args, progress, cancel, in_args)

    monkeypatch.setattr(media, "gpu_encoder", lambda: media._GPU_ENCODERS["h264_amf"])
    monkeypatch.setattr(media, "_convert", fake)
    out = media.make_proxy(src, tmp_path / "out.mp4")
    assert used[-1] == "libx264" and used[:-1] and set(used[:-1]) == {"h264_amf"}  # GPU decode+encode, GPU encode, CPU
    assert out.codec == "h264" and (tmp_path / "out.mp4").exists()


def test_no_gpu_means_cpu_only(tmp_path, monkeypatch):
    src = tmp_path / "in.avi"
    _make_mpeg4(src)
    monkeypatch.setenv("AA_GPU_ENCODE", "0")
    assert media.gpu_encoder() is None
    assert media.make_proxy(src, tmp_path / "out.mp4").codec == "h264"


def test_real_gpu_encode_if_available(tmp_path):
    if media.gpu_encoder() is None:
        pytest.skip("no working GPU H.264 encoder on this machine")
    src = tmp_path / "in.avi"
    _make_mpeg4(src)
    out = media.make_proxy(src, tmp_path / "out.mp4")
    assert out.codec == "h264" and abs(out.duration_s - 1.0) < 0.2
