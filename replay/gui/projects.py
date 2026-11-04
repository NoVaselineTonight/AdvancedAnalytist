"""Project storage: one folder per analysed video.

projects/<id>/
    project.json        name, source info, analysis settings
    meta.json           game metadata (teams, colours, overlays)
    calibration.json    court calibration keyframes
    video.mp4           H.264 playback copy - this is what is analysed and replayed
    source/<file>       original upload (only when it was uploaded or copied in)
    results/            tracking.csv/json, replay.html, report, work/ (resumable pass-1 data)

    timeline.json       user timeline: tracks, marks and notes

In the GUI a project is called a "Replay". It exports to one ``.aareplay`` file: a zip of the folder plus the
original video and a manifest.json - everything needed to open it on another computer without re-analysing.
Legacy ``.aaproj`` files (no manifest) still load.
State is derived from which files exist, so a project folder copied by hand also just works.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
import uuid
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from replay.calibration import CalibrationFile
from replay.config import Config
from replay.metadata import load_metadata

PROJECT_FORMAT = "advanced-analytist/project@1"
PACKAGE_FORMAT = "advanced-analytist/replay-package@1"
PACKAGE_VERSION = 1
PACKAGE_EXT = ".aareplay"
LEGACY_PACKAGE_EXT = ".aaproj"
MANIFEST = "manifest.json"
STORED_SUFFIXES = {".mp4", ".m4v", ".mov", ".mkv", ".ts", ".avi", ".npz", ".webm", ".m2ts", ".mts", ".wmv", ".flv",
                   ".mpg", ".mpeg", ".mxf", ".3gp", ".jpg", ".png", ".zip"}
COPY_CHUNK = 8 * 1024 * 1024
DEFAULT_SETTINGS = {"frame_stride": 3, "variant": "medium", "device": "auto", "fixed_camera": False,
                    "config_overrides": {}}
DEFAULT_META = {
    "home_team": "Home", "away_team": "Away", "team_colors_auto": True,
    "jersey_colors": {"home": {"primary": "#f5f5f5"}, "away": {"primary": "#1f2a5a"},
                      "referee": {"primary": "#9a9a9a", "secondary": "#141414"}},
    "court_standard": "FIBA", "overlay_regions": [],
}


def _slug(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", name).strip("-").lower()[:40]
    return s or "project"


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


_STAT_CACHE: dict[tuple[str, str], tuple[tuple[int, int], Any]] = {}


def _cached(path: Path, kind: str, compute: Callable[[], Any]) -> Any:
    """``compute()`` memoised on the file's (mtime, size): the GUI polls every Replay's status several times a
    second, and calibration.json of a long game (automatic calibration) is large."""
    try:
        st = path.stat()
        stamp = (st.st_mtime_ns, st.st_size)
    except OSError:
        return compute()
    key = (str(path), kind)
    hit = _STAT_CACHE.get(key)
    if hit is not None and hit[0] == stamp:
        return hit[1]
    val = compute()
    if len(_STAT_CACHE) > 512:
        _STAT_CACHE.clear()
    _STAT_CACHE[key] = (stamp, val)
    return val


def _write_json(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)


@dataclass
class Project:
    root: Path

    # ------------------------------------------------------------------ paths
    @property
    def id(self) -> str:
        return self.root.name

    @property
    def video(self) -> Path:
        return self.root / "video.mp4"

    @property
    def results(self) -> Path:
        return self.root / "results"

    @property
    def calibration_path(self) -> Path:
        return self.root / "calibration.json"

    @property
    def meta_path(self) -> Path:
        return self.root / "meta.json"

    # ------------------------------------------------------------------ data
    def info(self) -> dict[str, Any]:
        return _read_json(self.root / "project.json", {})

    def save_info(self, info: dict[str, Any]) -> None:
        _write_json(self.root / "project.json", info)

    def update_info(self, **changes: Any) -> dict[str, Any]:
        info = self.info()
        info.update(changes)
        self.save_info(info)
        return info

    def meta(self) -> dict[str, Any]:
        return _read_json(self.meta_path, DEFAULT_META)

    def save_meta(self, meta: dict[str, Any]) -> None:
        load_metadata_check(meta)
        _write_json(self.meta_path, meta)

    def settings(self) -> dict[str, Any]:
        st = {**DEFAULT_SETTINGS, **self.info().get("settings", {})}
        if "process_fps" in st:  # projects from before frame_stride existed
            fps = self.info().get("video_info", {}).get("fps") or 30.0
            st["frame_stride"] = max(1, round(fps / float(st.pop("process_fps"))))
        return st

    def config(self) -> Config:
        st = self.settings()
        cfg = Config.from_dict(st.get("config_overrides") or {})
        cfg.frame_stride = int(st["frame_stride"])
        cfg.detector.variant = st["variant"]
        cfg.detector.device = st["device"]
        # Camera motion is always estimated: it's cheap, and "fixed camera" is a calibration-only choice,
        # so toggling it never invalidates the analysis.
        return cfg

    def calibration(self) -> CalibrationFile:
        if self.calibration_path.exists():
            return CalibrationFile.load(self.calibration_path)
        size = tuple(self.info().get("video_info", {}).get("size", (1280, 720)))
        return CalibrationFile(video="video.mp4", frame_size=size,  # type: ignore[arg-type]
                               fixed_camera=bool(self.settings()["fixed_camera"]))

    def calibration_summary(self) -> dict[str, Any]:
        """Keyframe count / automatic calibration, read straight from calibration.json.

        Cheap and side-effect free (polled several times a second for every Replay): CalibrationFile.load would
        refit every keyframe and reconfigure the global court size (court.configure) - which must not flip
        between Replays while a build of another Replay runs in the background.
        """
        if not self.calibration_path.exists():
            return {"keyframes": 0, "auto_calibrated": False, "calibrated": False,
                    "fixed_camera": bool(self.settings()["fixed_camera"])}
        return dict(_cached(self.calibration_path, "summary", self._calibration_summary))

    def _calibration_summary(self) -> dict[str, Any]:
        d = _read_json(self.calibration_path, None)
        if not isinstance(d, dict):
            return {"keyframes": 0, "auto_calibrated": False, "calibrated": False,
                    "fixed_camera": bool(self.settings()["fixed_camera"])}
        auto = d.get("auto") if isinstance(d.get("auto"), dict) else None
        auto_ok = bool(auto and any(isinstance(a, dict) and a.get("good") for a in auto.get("anchors") or []))
        n = len(d.get("keyframes") or [])
        return {"keyframes": n, "auto_calibrated": auto_ok, "calibrated": bool(n) or auto_ok,
                "fixed_camera": bool(d.get("fixed_camera", False))}

    def status(self) -> dict[str, Any]:
        work = self.results / "work"
        cal = self.calibration_summary()
        tracking = self.results / "tracking.json"
        replay_ready = tracking.exists()
        # results/JOB_build (replay/gui/jobs.py): a build was interrupted - its outputs may be half-written
        build_interrupted = (self.results / "JOB_build").exists()
        stale = replay_ready and (build_interrupted or any(
            p.exists() and p.stat().st_mtime > tracking.stat().st_mtime
            for p in (self.calibration_path, self.meta_path)))
        rp = self.results / "report.json"
        report = _cached(rp, "json", lambda: _read_json(rp, None)) if rp.exists() else None
        return {
            "imported": self.video.exists(),
            "analyzed": (work / "DONE").exists(),
            "analysis_started": (work / "run.json").exists(),
            "keyframes": cal["keyframes"],
            "auto_calibrated": cal["auto_calibrated"],
            "calibrated": cal["calibrated"],  # manual keyframes and/or the automatic calibration
            "fixed_camera": cal["fixed_camera"],
            "replay_ready": replay_ready,
            "replay_stale": stale,
            "build_interrupted": build_interrupted,
            "paused": (self.results / "PAUSED").exists(),  # the user pressed Stop: don't continue by itself
            "report": report,
        }

    def summary(self) -> dict[str, Any]:
        info = self.info()
        return {"id": self.id, "name": info.get("name", self.id), "created": info.get("created"),
                "source": info.get("source", {}), "video_info": info.get("video_info", {}),
                "settings": self.settings(), "status": self.status()}


@dataclass
class ExportPlan:
    name: str
    files: list[tuple[Path, str]] = field(default_factory=list)  # (file on disk, path inside the package)
    original: str | None = None  # package path of the original video
    warnings: list[str] = field(default_factory=list)

    @property
    def total_bytes(self) -> int:
        return sum(f.stat().st_size for f, _ in self.files if f.exists())

    def groups(self) -> dict[str, int]:
        """Bytes per kind of content, for showing what will be included."""
        out: dict[str, int] = {}
        for f, rel in self.files:
            g = ("Original video" if rel == self.original or rel.startswith("source/")
                 else "Playback video (video.mp4)" if rel == "video.mp4"
                 else "Analysis cache (results/work)" if rel.startswith("results/work/")
                 else "2D replay, tracking data and report" if rel.startswith("results/")
                 else "Timeline (tracks, marks and notes)" if rel == "timeline.json"
                 else "Clip playlists" if rel == "playlists.json"  # replay/playlists.py
                 else "Calibration, game info and settings")
            out[g] = out.get(g, 0) + (f.stat().st_size if f.exists() else 0)
        return out


def load_metadata_check(meta: dict[str, Any]) -> None:
    """Raise ValueError if the metadata would not parse in the pipeline."""
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "m.json"
        p.write_text(json.dumps(meta), encoding="utf-8")
        load_metadata(p)


class ProjectStore:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "_incoming").mkdir(exist_ok=True)

    def list(self) -> list[Project]:
        out = [Project(p) for p in self.root.iterdir() if p.is_dir() and (p / "project.json").exists()]
        return sorted(out, key=lambda p: p.info().get("created", 0), reverse=True)

    def get(self, pid: str) -> Project:
        if not re.fullmatch(r"[a-z0-9-]+", pid):
            raise KeyError(pid)
        p = Project(self.root / pid)
        if not (p.root / "project.json").exists():
            raise KeyError(pid)
        return p

    def create(self, name: str, source: dict[str, Any]) -> Project:
        pid = f"{_slug(name)}-{uuid.uuid4().hex[:6]}"
        root = self.root / pid
        root.mkdir(parents=True)
        p = Project(root)
        p.save_info({"format": PROJECT_FORMAT, "name": name, "created": time.time(), "source": source,
                     "settings": dict(DEFAULT_SETTINGS)})
        _write_json(p.meta_path, DEFAULT_META)
        return p

    def delete(self, pid: str) -> None:
        shutil.rmtree(self.get(pid).root)

    def duplicate(self, pid: str, name: str | None = None,
                  progress: Callable[[float], None] | None = None) -> Project:
        """Copy a Replay into a new folder (everything: video, analysis, calibration, timeline, playlists).

        Job markers (results/JOB_*, PAUSED) and half-written files are left out, so the copy never starts a job by
        itself. The copy gets a new id, the given name (default "<name> (copy)") and a fresh creation time.
        """
        src = self.get(pid)
        info = src.info()
        new_name = (name or f"{info.get('name', src.id)} (copy)")[:100]
        dst_root = self.root / f"{_slug(new_name)}-{uuid.uuid4().hex[:6]}"
        files = [f for f in sorted(src.root.rglob("*")) if f.is_file() and not _skip_on_copy(f, src.root)]
        total = sum(f.stat().st_size for f in files) or 1
        done = 0
        dst_root.mkdir(parents=True)
        try:
            for f in files:
                target = dst_root / f.relative_to(src.root)
                target.parent.mkdir(parents=True, exist_ok=True)
                with open(f, "rb") as fi, open(target, "wb") as fo:
                    while chunk := fi.read(COPY_CHUNK):
                        fo.write(chunk)
                        done += len(chunk)
                        if progress:
                            progress(min(1.0, done / total))
                shutil.copystat(f, target)  # keep mtimes: "replay outdated" compares them
        except BaseException:
            shutil.rmtree(dst_root, ignore_errors=True)
            raise
        p = Project(dst_root)
        new_info = dict(info)
        new_info.update(name=new_name, created=time.time(), duplicated_from=src.id)
        new_info.pop("pinned", None)
        p.save_info(new_info)
        self._repoint_work(p)
        if progress:
            progress(1.0)
        return p

    def incoming_path(self, filename: str) -> Path:
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(filename).name) or "upload"
        return self.root / "_incoming" / f"{uuid.uuid4().hex[:8]}_{safe}"

    # ------------------------------------------------------------------ export / import
    def export_plan(self, pid: str) -> ExportPlan:
        """Everything that goes into a Replay package: the whole project folder plus the original video."""
        p = self.get(pid)
        plan = ExportPlan(name=p.info().get("name", p.id))
        for f in sorted(p.root.rglob("*")):
            if not f.is_file() or f.name.endswith((".tmp", ".part")):
                continue
            rel = f.relative_to(p.root).as_posix()
            if rel == MANIFEST:  # regenerated on export
                continue
            plan.files.append((f, rel))
        have = {rel for _, rel in plan.files}
        orig = original_video(p)
        if orig is not None:
            try:
                rel = orig.resolve().relative_to(p.root.resolve()).as_posix()
            except ValueError:  # referenced from elsewhere on disk
                rel = f"source/{orig.name}"
                plan.files.append((orig, rel))
            plan.original = rel
        else:
            src = p.info().get("source", {}) or {}
            shown = src.get("path") or src.get("filename") or "unknown"
            plan.warnings.append(f"The original video file was not found ({shown}); the package contains only "
                                 "the playback copy.")
        if "video.mp4" not in have:
            plan.warnings.append("The playback copy video.mp4 is missing.")
        return plan

    def export(self, pid: str, dst: Path, progress: Callable[[float], None] | None = None,
               plan: ExportPlan | None = None) -> ExportPlan:
        """Write the Replay as one ``.aareplay`` file (zip; videos stored uncompressed, everything included).

        Returns the plan that was written; ``plan.warnings`` lists anything that was missing.
        """
        from replay import __version__

        p = self.get(pid)
        plan = plan or self.export_plan(pid)
        info = p.info()
        pkg_info = dict(info)  # inside the package the original lives in source/; the loaded Replay uses that
        if plan.original:
            src = dict(info.get("source", {}) or {})
            if src.get("kind") != "copy" and src.get("path"):
                src["original_path"] = src["path"]
            src.update(kind="copy", path=plan.original, filename=Path(plan.original).name)
            pkg_info["source"] = src
        info_bytes = json.dumps(pkg_info, indent=2).encode("utf-8")
        contents = [{"path": rel, "size": len(info_bytes) if rel == "project.json" else f.stat().st_size}
                    for f, rel in plan.files]
        if not any(c["path"] == "project.json" for c in contents):
            raise ValueError("project.json is missing.")
        manifest = {
            "format": PACKAGE_FORMAT, "version": PACKAGE_VERSION,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "created_ts": round(time.time(), 3),
            "app": "AdvancedAnalytist", "app_version": __version__, "project_format": PROJECT_FORMAT,
            "replay_name": plan.name, "original_video": plan.original, "playback_video": "video.mp4",
            "warnings": plan.warnings, "total_bytes": sum(c["size"] for c in contents), "contents": contents,
        }
        total = manifest["total_bytes"] or 1
        done = 0
        dst = Path(dst)
        tmp = Path(str(dst) + ".part")
        try:
            with zipfile.ZipFile(tmp, "w", allowZip64=True) as z:
                z.writestr(MANIFEST, json.dumps(manifest, indent=2), compress_type=zipfile.ZIP_DEFLATED)
                for (f, rel), c in zip(plan.files, contents):
                    if rel == "project.json":
                        z.writestr(rel, info_bytes, compress_type=zipfile.ZIP_DEFLATED)
                        done += c["size"]
                        continue
                    zi = zipfile.ZipInfo.from_file(f, rel)
                    stored = f.suffix.lower() in STORED_SUFFIXES or c["size"] > 256 * 1024 * 1024
                    zi.compress_type = zipfile.ZIP_STORED if stored else zipfile.ZIP_DEFLATED
                    with open(f, "rb") as fi, z.open(zi, "w", force_zip64=True) as fo:
                        while chunk := fi.read(COPY_CHUNK):
                            fo.write(chunk)
                            done += len(chunk)
                            if progress:
                                progress(min(1.0, done / total))
            tmp.replace(dst)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        if progress:
            progress(1.0)
        return plan

    @staticmethod
    def read_manifest(package: Path) -> dict[str, Any] | None:
        """The package manifest, or None for a legacy ``.aaproj`` file. Raises ValueError if it is invalid."""
        with zipfile.ZipFile(package) as z:
            return _validate_manifest(z)

    def import_package(self, package: Path, progress: Callable[[float], None] | None = None) -> Project:
        """Load a ``.aareplay`` (or legacy ``.aaproj``) file as a new Replay. Never analyses anything."""
        package = Path(package)
        try:
            z = zipfile.ZipFile(package)
        except zipfile.BadZipFile as e:
            raise ValueError("Not an AdvancedAnalytist Replay file (not a zip archive).") from e
        with z:
            manifest = _validate_manifest(z)
            names = z.namelist()
            if "project.json" not in names:
                raise ValueError("Not an AdvancedAnalytist Replay file (project.json missing).")
            info = json.loads(z.read("project.json"))
            if info.get("format") != PROJECT_FORMAT:
                raise ValueError(f"Unsupported Replay format: {info.get('format')!r}")
            pid = f"{_slug(info.get('name', 'project'))}-{uuid.uuid4().hex[:6]}"
            root = (self.root / pid).resolve()
            for n in names:
                target = (root / n).resolve()
                if root not in target.parents and target != root:  # zip-slip protection
                    raise ValueError(f"Unsafe path in Replay file: {n}")
            members = [zi for zi in z.infolist() if not zi.is_dir() and zi.filename != MANIFEST]
            total = sum(zi.file_size for zi in members) or 1
            done = 0
            root.mkdir(parents=True)
            try:
                for zi in members:
                    target = root / zi.filename
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with z.open(zi) as fi, open(target, "wb") as fo:
                        while chunk := fi.read(COPY_CHUNK):
                            fo.write(chunk)
                            done += len(chunk)
                            if progress:
                                progress(min(1.0, done / total))
                    # keep the modification times: "replay outdated" compares calibration/meta vs tracking.json
                    ts = time.mktime(zi.date_time + (0, 0, -1))
                    os.utime(target, (ts, ts))
            except BaseException:
                shutil.rmtree(root, ignore_errors=True)
                raise
        p = Project(root)
        extra = {"package": {k: manifest.get(k) for k in ("format", "version", "created", "app_version")}} \
            if manifest else {}
        p.update_info(imported_from=package.name, imported_at=time.time(), **extra)
        self._repoint_work(p)
        return p

    @staticmethod
    def needs_playback_copy(p: Project) -> bool:
        """True when video.mp4 is missing but the original video is available, so it can be rebuilt."""
        return not p.video.exists() and original_video(p) is not None

    @classmethod
    def rebuild_playback_copy(cls, p: Project, progress: Callable[[float], None] | None = None) -> None:
        """Re-create video.mp4 from the original video (a format conversion, not an analysis)."""
        from replay.gui import media

        src = original_video(p)
        if src is None:
            raise FileNotFoundError("The original video is not available.")
        info = media.make_proxy(src, p.video, progress)
        if not p.info().get("video_info"):
            p.update_info(video_info={"duration_s": info.duration_s, "size": [info.width, info.height],
                                      "fps": info.fps, "codec": info.codec, "has_audio": info.has_audio})
        cls._repoint_work(p)

    @staticmethod
    def _repoint_work(p: Project) -> None:
        """Pass-1 data remembers the absolute video path; point it at this copy."""
        run = p.results / "work" / "run.json"
        if run.exists():
            sig = _read_json(run, {})
            sig["video"] = str(p.video.resolve())
            if p.video.exists() and "video_size_bytes" in sig:
                sig["video_size_bytes"] = p.video.stat().st_size
            _write_json(run, sig)


def _skip_on_copy(f: Path, root: Path) -> bool:
    rel = f.relative_to(root)
    return (f.name.endswith((".tmp", ".part")) or f.name == "PAUSED" or f.name.startswith("JOB_")
            or (rel.parts and rel.parts[0] == "_incoming"))


def folder_size(root: Path) -> int:
    """Bytes used by a Replay folder (all files, recursively). Missing/locked files count as 0."""
    total = 0
    for dirpath, _dirs, names in os.walk(root):
        for n in names:
            try:
                total += os.stat(os.path.join(dirpath, n)).st_size
            except OSError:
                pass
    return total


def original_video(p: Project) -> Path | None:
    """The original video file of a Replay, if it is available on this computer."""
    src = p.info().get("source", {}) or {}
    if src.get("path"):
        f = Path(src["path"])
        f = f if f.is_absolute() else p.root / f
        if f.is_file():
            return f
    d = p.root / "source"
    files = sorted(f for f in d.iterdir() if f.is_file()) if d.is_dir() else []
    return files[0] if files else None


def _validate_manifest(z: zipfile.ZipFile) -> dict[str, Any] | None:
    """Check manifest.json against the zip. None = legacy package without a manifest."""
    if MANIFEST not in z.namelist():
        return None
    try:
        m = json.loads(z.read(MANIFEST))
    except ValueError as e:
        raise ValueError("The Replay file is damaged (manifest.json is not valid JSON).") from e
    if not isinstance(m, dict) or not str(m.get("format", "")).startswith("advanced-analytist/replay-package@"):
        raise ValueError("Not an AdvancedAnalytist Replay file (unknown manifest format).")
    try:
        version = int(m.get("version", 0))
    except (TypeError, ValueError):
        version = 0
    if version < 1:
        raise ValueError("The Replay file is damaged (manifest has no valid version).")
    if version > PACKAGE_VERSION:
        raise ValueError(f"This Replay was made with a newer version of AdvancedAnalytist "
                         f"({m.get('app_version')}); please update the app.")
    sizes = {zi.filename: zi.file_size for zi in z.infolist()}
    contents = m.get("contents")
    if not isinstance(contents, list):
        raise ValueError("The Replay file is damaged (manifest has no contents list).")
    for c in contents:
        path = c.get("path") if isinstance(c, dict) else None
        if path not in sizes:
            raise ValueError(f"The Replay file is incomplete: {path} is missing.")
        if c.get("size") is not None and int(c["size"]) != sizes[path]:
            raise ValueError(f"The Replay file is damaged: {path} has the wrong size.")
    return m
