"""The default pass-1 frame taps ("decode once, fan out", SPEED_PLAN S2): jersey numbers, team colours and (when
on) re-ID embeddings are computed from the frames pass 1 already decodes and written as the same cache files pass 2
reads (work/jersey_numbers.npz, team_appearance.npz, reid_embeddings.npz). Pass 2 then skips its own video decode.
Env REPLAY_FRAME_TAPS=0 turns them off (pass 2 then decodes the video as before). The jersey tap is opt-in
(REPLAY_JERSEY_TAP=1): its OCR shares the GPU with the detector and slows pass 1 by the OCR time."""

from __future__ import annotations

import os


def default_taps(cfg) -> list:
    if os.environ.get("REPLAY_FRAME_TAPS", "1") == "0":
        return []
    taps: list = []
    try:
        from replay import jersey_numbers

        jc = jersey_numbers.default_config()
        # OFF by default (9.02): pass 1 is GPU-bound and DirectML runs one model at a time, so OCR in pass 1 delays
        # the detector by the whole OCR time (2.5-min 1080p cut: pass 1 79 s -> 128 s). Pass 2 reads the numbers
        # instead (same reads). REPLAY_JERSEY_TAP=1 turns the tap on (worth it only when the GPU has idle time).
        if jc.enabled and os.environ.get("REPLAY_JERSEY_TAP", "0") == "1":
            taps.append(jersey_numbers.JerseyTap(jc))
    except Exception as e:  # noqa: BLE001 - optional
        print("frame tap jersey: unavailable:", e)
    try:
        from replay.team_appearance import AppearanceTap

        tc = cfg.teams
        if getattr(tc, "resample_from_video", False):
            # CLIP crops (~430 MB per full game) only when something uses them: FilterConfig.clip_bench / AA_CLIP_BENCH
            from replay.team_appearance import CLIP_TAP_EVERY_S

            clip_on = bool(getattr(cfg.filter, "clip_bench", False)) or os.environ.get("AA_CLIP_BENCH") == "1"
            taps.append(AppearanceTap(tc.resample_every_s, clip_every_s=CLIP_TAP_EVERY_S if clip_on else 0.0))
    except Exception as e:  # noqa: BLE001
        print("frame tap team colours: unavailable:", e)
    try:  # broadcast score bar, CPU OCR every 2 s in its own thread, never waits (scorebar.py)
        if os.environ.get("REPLAY_SCOREBAR", "1") != "0":
            from replay.scorebar import ScorebarTap

            taps.append(ScorebarTap())
    except Exception as e:  # noqa: BLE001
        print("frame tap score bar: unavailable:", e)
    try:
        from replay import reid
        from replay.tracklets import reid_enabled

        rc = getattr(cfg, "reid", None) or reid.ReidConfig()
        if rc.enabled and reid_enabled(cfg):
            taps.append(reid.ReidTap(rc))
    except Exception as e:  # noqa: BLE001
        print("frame tap re-ID: unavailable:", e)
    try:
        from replay.autocal_tap import AutocalTap  # court-calibration features (calibration agent)

        taps.append(AutocalTap())
    except Exception as e:  # noqa: BLE001
        print("frame tap autocal: unavailable:", e)
    return taps
