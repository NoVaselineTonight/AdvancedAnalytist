"""What the Overview says about HOW a Replay was made (no Qt here, unit-testable).

* the settings it was analysed / built with (results/config.used.json + the detector line of results/analyze.log),
  as a short plain-language summary and as a full grouped list that marks values differing from today's defaults;
* the "Details" page for a Replay that has no built report yet: what ran, what did not finish and why, how many
  frames were analysed, where the logs are.

Everything is read defensively: a missing, half-written or old file never raises - it just shows less.
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Any

from replay import __version__

NOT_RECORDED = "not recorded"
UNKNOWN_VERSION = "unknown (built before 9.13)"
LOGS = (("analyze.log", "Player analysis"), ("autocal.log", "Court calibration"), ("build.log", "Replay build"))

GROUPS = [  # (config key, friendly title)
    ("detector", "Player detector"), ("tracking", "Tracking players"), ("motion", "Camera movement"),
    ("calibration", "Court calibration"), ("filter", "Who counts as a player"), ("teams", "Teams and referees"),
    ("court_tracking", "Following players on the court"), ("ball", "Ball"), ("smoothing", "Smoothing"),
    ("foot", "Feet and jumps"), ("scoring", "Baskets"), ("inplay", "Skipping breaks"),
    ("redetect", "Finding lost players again"), ("reid", "Recognising players by look (re-ID)"),
]
TOP_LEVEL = [("frame_stride", "Analyse every Nth frame"), ("process_fps", "Frames per second (when no stride)"),
             ("chunk_frames", "Frames per saved chunk"), ("parallel_pass2", "Parallel replay build"),
             ("corrections", "Use your corrections")]
COURT_DETECTORS = {"off": "off (lines only)", "fallback": "on when the lines are not enough",
                   "first": "on, tried first"}


def _e(x: Any) -> str:
    return html.escape(str(x))


def read_json(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def read_settings(results: Path) -> dict | None:
    """results/config.used.json as a dict, or None when it is missing / unreadable."""
    d = read_json(Path(results) / "config.used.json")
    return d if isinstance(d, dict) else None


def version_text(cfg: dict | None) -> str:
    if cfg is None:
        return NOT_RECORDED
    v = cfg.get("app_version")
    return f"version {v}" if v else UNKNOWN_VERSION


def clean_log(path: Path, tail: int = 6, max_bytes: int = 1_500_000) -> list[str]:
    """Readable lines of a job log: progress-bar redraws removed, the last ``tail`` lines kept (all if tail=0)."""
    try:
        raw = Path(path).read_bytes()
    except OSError:
        return []
    if len(raw) > max_bytes:  # keep the head (detector line) and the end
        raw = raw[:30_000] + b"\n" + raw[-(max_bytes - 30_000):]
    text = raw.decode("utf-8", errors="replace")
    out = []
    for seg in re.split(r"[\r\n]+", text):
        seg = seg.strip()
        if not seg or "%|" in seg and "/s]" in seg:
            continue
        seg = re.sub(r"^(?:[\w+ ]+:\s+\d+%\|[^\n]*?\]\s*)+", "", seg)  # progress bar glued to a real line
        if seg:
            out.append(seg)
    return out[-tail:] if tail else out


def detector_from_log(results: Path) -> dict:
    """The 'Detector: NAME on DEVICE (...); analysing X frames/s of video' line of analyze.log (all optional)."""
    info: dict[str, Any] = {}
    lines = clean_log(Path(results) / "analyze.log", tail=0)
    for ln in lines:
        m = re.search(r"Detector: (\S+) on (.*?)(?:;\s*analysing ([0-9.]+) frames/s|$)", ln)
        if m:
            rest = m.group(2)
            cut = re.search(r" \((?=[^()]*(?:weights|:))", rest)
            info["model"] = m.group(1)
            info["device"] = (rest[:cut.start()] if cut else rest).strip()
            if m.group(3):
                info["fps"] = float(m.group(3))
            break
    for ln in lines:
        m = re.search(r"(\d+) of (\d+) samples", ln)
        if m:
            info["frames"] = int(m.group(1))
            break
    return info


def frames_analysed(results: Path, report: dict | None, cfg: dict | None, log_info: dict) -> str | None:
    if report and report.get("frames_processed"):
        return f"{int(report['frames_processed']):,}"
    if log_info.get("frames"):
        return f"{log_info['frames']:,}"
    chunks = len(list((Path(results) / "work").glob("chunk_*.npz"))) if (Path(results) / "work").is_dir() else 0
    if chunks:
        per = int((cfg or {}).get("chunk_frames") or 300)
        return f"about {chunks * per:,} (saved in {chunks} chunk{'s' if chunks != 1 else ''})"
    return None


def _fps_text(cfg: dict | None, log_info: dict, report: dict | None, video_fps: float | None) -> str | None:
    fps = log_info.get("fps") or (report or {}).get("process_fps")
    stride = (cfg or {}).get("frame_stride")
    if not fps and cfg:
        fps = (video_fps / stride) if (stride and video_fps) else cfg.get("process_fps")
    if not fps:
        return None
    txt = f"{float(fps):.1f} per second of video"
    return txt + (f" (every {int(stride)}{'nd' if stride == 2 else 'rd' if stride == 3 else 'th'} frame)"
                  if stride and int(stride) > 1 else " (every frame)" if stride == 1 else "")


def summary_rows(results: Path, project_settings: dict | None, report: dict | None,
                 video_fps: float | None = None) -> list[tuple[str, str]]:
    """The always-visible short version: (label, text) pairs, in plain words."""
    cfg = read_settings(results)
    log_info = detector_from_log(results)
    rows: list[tuple[str, str]] = []
    if cfg is None and not log_info:
        return [("Settings used", NOT_RECORDED), ("Built with", NOT_RECORDED)]
    det = (cfg or {}).get("detector") or {}
    model = log_info.get("model") or (f"rfdetr-{det['variant']}" if det.get("variant") else None)
    if model:
        where = f" on {log_info['device']}" if log_info.get("device") else ""
        res = det.get("resolution")
        tiles = det.get("tiles") or [1, 1]
        extra = f", {res} px" if res else ", standard picture size"
        if cfg is not None and list(tiles) != [1, 1]:
            extra += f", picture split in {tiles[0]}x{tiles[1]} tiles"
        rows.append(("Player detector", f"{model}{where}{extra}"))
    else:
        rows.append(("Player detector", NOT_RECORDED))
    fps = _fps_text(cfg, log_info, report, video_fps)
    rows.append(("Frames analysed", fps or NOT_RECORDED))
    standard = (project_settings or {}).get("court_standard") or "FIBA"
    cd = ((cfg or {}).get("calibration") or {}).get("court_detector")
    rows.append(("Court", f"{standard} lines; court finder {COURT_DETECTORS.get(cd, cd) if cd else NOT_RECORDED}"))
    if cfg is not None:
        ball = cfg.get("ball") or {}
        rows.append(("Ball", f"tracked (min. confidence {ball['min_conf']})" if "min_conf" in ball else "tracked"))
        rows.append(("Player re-ID", "on" if (cfg.get("reid") or {}).get("enabled") else "off"))
        rows.append(("Your corrections", "on" if cfg.get("corrections") else "off"))
    else:
        rows += [("Ball", NOT_RECORDED), ("Player re-ID", NOT_RECORDED), ("Your corrections", NOT_RECORDED)]
    rows.append(("Built with", version_text(cfg)))
    return rows


def summary_html(rows: list[tuple[str, str]]) -> str:
    body = "".join(f"<tr><td style='color:gray'>{_e(a)}</td><td>{_e(b)}</td></tr>" for a, b in rows)
    return f"<p><b>Settings used</b></p><table cellpadding=2>{body}</table>"


# ---------------------------------------------------------------- the full list
def _fmt(v: Any) -> str:
    if v is None:
        return "default"
    if isinstance(v, bool):
        return "on" if v else "off"
    if isinstance(v, (list, tuple)):
        return ", ".join(_fmt(x) for x in v)
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def settings_diff(cfg: dict, defaults: dict) -> list[str]:
    """Names (section.key) of the values that differ from the defaults."""
    out = []
    for k, v in cfg.items():
        if k == "app_version":
            continue
        if isinstance(v, dict):
            dv = defaults.get(k) if isinstance(defaults.get(k), dict) else {}
            out += [f"{k}.{kk}" for kk, vv in v.items() if kk in dv and _norm(vv) != _norm(dv[kk])]
        elif k in defaults and _norm(v) != _norm(defaults[k]):
            out.append(k)
    return out


def _norm(v: Any) -> Any:
    return [_norm(x) for x in v] if isinstance(v, (list, tuple)) else v


def settings_html(cfg: dict, defaults: dict | None = None) -> str:
    """Every setting, grouped. Values that differ from today's defaults are highlighted and show the default."""
    if defaults is None:
        from replay.config import Config
        defaults = Config().to_dict()
    changed = set(settings_diff(cfg, defaults))

    def row(label: str, value: Any, default: Any, key: str) -> str:
        if key in changed:
            return (f"<tr><td>{_e(label)}</td><td style='background:#fef3c7;color:#92400e'><b>{_e(_fmt(value))}</b>"
                    f"</td><td style='color:gray'>default: {_e(_fmt(default))}</td></tr>")
        return f"<tr><td>{_e(label)}</td><td>{_e(_fmt(value))}</td><td></td></tr>"

    parts = [f"<p><b>All settings</b> <span style='color:gray'>(built with {_e(version_text(cfg))}; "
             f"{len(changed)} differ from today's defaults, shown in orange)</span></p>"]
    top = "".join(row(t, cfg[k], defaults.get(k), k) for k, t in TOP_LEVEL if k in cfg)
    if top:
        parts.append(f"<p><i>General</i></p><table cellpadding=2>{top}</table>")
    for key, title in GROUPS:
        sec = cfg.get(key)
        if not isinstance(sec, dict) or not sec:
            continue
        dsec = defaults.get(key) if isinstance(defaults.get(key), dict) else {}
        rows = "".join(row(n, v, dsec.get(n), f"{key}.{n}") for n, v in sec.items())
        n_diff = sum(1 for n in sec if f"{key}.{n}" in changed)
        flag = f" <span style='color:#92400e'>({n_diff} changed)</span>" if n_diff else ""
        parts.append(f"<p><i>{_e(title)}</i>{flag}</p><table cellpadding=2>{rows}</table>")
    return "".join(parts)


# ---------------------------------------------------------------- Details with no built report
def _phase_lines(results: Path, status: dict) -> tuple[list[str], list[str]]:
    from replay.gui.phases import PHASES, fmt_s
    saved = read_json(Path(results) / "phases.json")
    saved = saved if isinstance(saved, dict) else {}
    done, todo = [], []
    ok = {"import": status.get("imported"), "prelim": status.get("analyzed"), "pass1": status.get("analyzed"),
          "calib1": status.get("analyzed"), "autocal": status.get("calibrated"),
          "jersey": status.get("replay_ready"), "teams": status.get("replay_ready"),
          "build": status.get("replay_ready"), "report": status.get("replay_ready")}
    for key, title in PHASES:
        sv = saved.get(key) if isinstance(saved.get(key), dict) else {}
        if ok.get(key) or sv.get("state") in ("done", "skipped"):
            took = f" - took {fmt_s(sv['s'])}" if sv.get("s") else ""
            done.append(f"{title}{' (skipped)' if sv.get('state') == 'skipped' else ''}{took}")
        else:
            todo.append(title)
    return done, todo


def details_without_report(p: Any, status: dict, job: dict | None) -> str:
    """Details for a Replay that has no built report: always something useful."""
    results = Path(p.results)
    job = job or {}
    done, todo = _phase_lines(results, status)
    out = ["<p><b>No replay has been built yet, so there is no report. Here is what is known.</b></p>"]
    out.append("<p><b>What ran</b></p>" + ("<ul>" + "".join(f"<li>&#10003; {_e(d)}</li>" for d in done) + "</ul>"
                                           if done else "<p style='color:gray'>Nothing has finished yet.</p>"))
    why = []
    if job.get("error"):
        why.append(f"Last error: {job['error']}")
    msg = job.get("message") or ""
    if msg.startswith("Could not find the court") or (status.get("analyzed") and not status.get("calibrated")):
        why.append(msg or "The court could not be found automatically. Open <i>Court fix</i> and mark a few "
                          "court points by hand, then the replay can be built.")
    if status.get("build_interrupted") or (results / "JOB_build").exists():
        why.append("The replay build was interrupted (the app was closed or stopped). Press Build replay to run it again.")
    if (results / "JOB_autocal").exists():
        why.append("The automatic court calibration started but did not finish.")
    if status.get("analysis_started") and not status.get("analyzed") and not job.get("kind"):
        why.append("The player analysis was stopped before it finished. It continues where it left off.")
    for fname, title in LOGS:
        tail = [t for t in clean_log(results / fname, tail=6)
                if re.search(r"error|fail|could not|traceback|exception|no court|not found", t, re.I)]
        if tail:
            why.append(f"{title} log says: " + " | ".join(tail[-2:]))
    if todo:
        out.append("<p><b>Not done yet</b></p><ul>" + "".join(f"<li>{_e(t)}</li>" for t in todo) + "</ul>")
    if why:
        out.append("<p><b>Why</b></p><ul>" + "".join(f"<li>{w if 'Court fix' in w else _e(w)}</li>" for w in why)
                   + "</ul>")
    elif todo:
        out.append("<p style='color:gray'>No error was recorded.</p>")
    cfg = read_settings(results)
    fr = frames_analysed(results, None, cfg, detector_from_log(results))
    out.append(f"<p><b>Frames analysed:</b> {_e(fr) if fr else 'not known yet'}</p>")
    logs = []
    for fname, title in LOGS:
        f = results / fname
        if f.exists():
            tail = clean_log(f, tail=4)
            logs.append(f"<li><b>{_e(title)}</b> - {_e(f)} ({f.stat().st_size // 1024} KB)"
                        + (("<br><span style='color:gray'>" + "<br>".join(_e(t[:200]) for t in tail) + "</span>")
                           if tail else "<br><span style='color:gray'>(empty - nothing was written)</span>") + "</li>")
    out.append("<p><b>Logs</b> - <a href='open:results'>open the results folder</a></p>"
               + ("<ul>" + "".join(logs) + "</ul>" if logs else
                  f"<p style='color:gray'>No log files yet. They are written to {_e(results)}.</p>"))
    return "".join(out)
