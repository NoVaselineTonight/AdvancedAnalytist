import json
import zipfile

import pytest
from fake_project import make_project

from replay.gui.projects import MANIFEST, PACKAGE_FORMAT, Project, ProjectStore


def _store_with_project(tmp_path, **kw):
    store = ProjectStore(tmp_path / "projects")
    make_project(store.root / "kouvot-honka-abc123", **kw)
    return store, store.get("kouvot-honka-abc123")


def test_roundtrip_includes_original_and_needs_no_analysis(tmp_path):
    orig = tmp_path / "elsewhere" / "game.mts"
    orig.parent.mkdir()
    orig.write_bytes(b"ORIGINAL-VIDEO" * 1000)
    store, p = _store_with_project(tmp_path, original=orig)
    seen = []
    plan = store.export(p.id, tmp_path / "out.aareplay", progress=seen.append)
    assert plan.warnings == [] and plan.original == "source/game.mts"
    assert seen and seen[-1] == 1.0

    with zipfile.ZipFile(tmp_path / "out.aareplay") as z:
        names = z.namelist()
        assert names[0] == MANIFEST
        for n in ["source/game.mts", "video.mp4", "project.json", "meta.json", "calibration.json", "timeline.json",
                  "results/tracking.json", "results/replay_data.json", "results/replay.html",
                  "results/work/run.json", "results/work/DONE", "results/work/chunk_00000.npz"]:
            assert n in names, n
        assert z.getinfo("video.mp4").compress_type == zipfile.ZIP_STORED
        assert z.getinfo("source/game.mts").compress_type == zipfile.ZIP_STORED
        m = json.loads(z.read(MANIFEST))
        assert m["format"] == PACKAGE_FORMAT and m["version"] == 1 and m["replay_name"] == "Kouvot vs Honka"
        assert m["app_version"] and m["created"]
        sizes = {c["path"]: c["size"] for c in m["contents"]}
        assert sizes["source/game.mts"] == orig.stat().st_size
        assert sizes["project.json"] == z.getinfo("project.json").file_size
        info = json.loads(z.read("project.json"))
        assert info["source"]["kind"] == "copy" and info["source"]["path"] == "source/game.mts"
        assert info["source"]["original_path"] == str(orig)

    # load on "another computer": the original is no longer at its old place
    orig.unlink()
    store2 = ProjectStore(tmp_path / "other")
    assert store2.read_manifest(tmp_path / "out.aareplay")["replay_name"] == "Kouvot vs Honka"
    q = store2.import_package(tmp_path / "out.aareplay")
    assert (q.root / "source" / "game.mts").read_bytes() == b"ORIGINAL-VIDEO" * 1000
    assert not (q.root / MANIFEST).exists()
    assert (q.root / "timeline.json").exists()
    s = q.status()
    assert s["imported"] and s["analyzed"] and s["replay_ready"]
    assert not s["replay_stale"], "modification times must survive so the replay isn't rebuilt"
    assert not store2.needs_playback_copy(q)
    run = json.loads((q.results / "work" / "run.json").read_text(encoding="utf-8"))
    assert run["video"] == str(q.video.resolve())
    assert run["video_size_bytes"] == q.video.stat().st_size

    # re-export of the loaded Replay keeps the original (it is now inside the project)
    plan2 = store2.export(q.id, tmp_path / "again.aareplay")
    assert plan2.original == "source/game.mts" and not plan2.warnings
    with zipfile.ZipFile(tmp_path / "again.aareplay") as z:
        assert z.namelist().count("source/game.mts") == 1
        assert z.namelist().count(MANIFEST) == 1


def test_missing_original_warns_but_exports(tmp_path):
    store, p = _store_with_project(tmp_path)  # source path C:/nowhere/game.mts does not exist
    plan = store.export_plan(p.id)
    assert plan.original is None and any("original video" in w for w in plan.warnings)
    assert "Original video" not in plan.groups() and plan.total_bytes > 0
    plan = store.export(p.id, tmp_path / "x.aareplay")
    assert plan.warnings
    q = ProjectStore(tmp_path / "other").import_package(tmp_path / "x.aareplay")
    assert q.status()["replay_ready"]


def test_copied_original_is_included_once(tmp_path):
    store, p = _store_with_project(tmp_path, copy_original=True)
    plan = store.export_plan(p.id)
    assert plan.original == "source/game.mts" and not plan.warnings
    assert [rel for _, rel in plan.files].count("source/game.mts") == 1
    assert set(plan.groups()) >= {"Original video", "Playback video (video.mp4)", "Analysis cache (results/work)",
                                  "Timeline (tracks, marks and notes)"}


def test_playback_copy_rebuild_needed_when_missing(tmp_path):
    store, p = _store_with_project(tmp_path, copy_original=True)
    p.video.unlink()
    store.export(p.id, tmp_path / "x.aareplay")
    q = ProjectStore(tmp_path / "other").import_package(tmp_path / "x.aareplay")
    assert ProjectStore.needs_playback_copy(q)


def test_manifest_validation(tmp_path):
    store, p = _store_with_project(tmp_path)
    pkg = tmp_path / "x.aareplay"
    store.export(p.id, pkg)

    def rewrite(dst, fn):
        with zipfile.ZipFile(pkg) as zi, zipfile.ZipFile(dst, "w") as zo:
            for item in zi.infolist():
                data = zi.read(item)
                res = fn(item.filename, data)
                if res is not None:
                    zo.writestr(item.filename, res)

    def bump(name, data):
        if name == MANIFEST:
            m = json.loads(data)
            m["version"] = 99
            return json.dumps(m)
        return data

    rewrite(tmp_path / "newer.aareplay", bump)
    with pytest.raises(ValueError, match="newer version"):
        store.import_package(tmp_path / "newer.aareplay")

    rewrite(tmp_path / "missing.aareplay", lambda n, d: None if n == "results/tracking.json" else d)
    with pytest.raises(ValueError, match="incomplete"):
        store.import_package(tmp_path / "missing.aareplay")

    rewrite(tmp_path / "trunc.aareplay", lambda n, d: d[:10] if n == "video.mp4" else d)
    with pytest.raises(ValueError, match="damaged"):
        store.import_package(tmp_path / "trunc.aareplay")

    (tmp_path / "junk.aareplay").write_bytes(b"not a zip")
    with pytest.raises(ValueError, match="not a zip"):
        store.import_package(tmp_path / "junk.aareplay")
    # nothing half-imported was left behind
    assert [x.id for x in store.list()] == [p.id]


def test_zip_slip_rejected(tmp_path):
    store = ProjectStore(tmp_path / "projects")
    pkg = tmp_path / "evil.aaproj"
    with zipfile.ZipFile(pkg, "w") as z:
        z.writestr("project.json", json.dumps({"format": "advanced-analytist/project@1", "name": "evil"}))
        z.writestr("../../escaped.txt", "x")
    with pytest.raises(ValueError, match="Unsafe path"):
        store.import_package(pkg)
    assert not (tmp_path / "escaped.txt").exists()
    assert store.list() == []


def test_legacy_aaproj_still_loads(tmp_path):
    store, p = _store_with_project(tmp_path)
    pkg = tmp_path / "old.aaproj"
    with zipfile.ZipFile(pkg, "w") as z:  # the old format: the folder zipped, no manifest
        for f in sorted(p.root.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(p.root).as_posix())
    assert store.read_manifest(pkg) is None
    q = store.import_package(pkg)
    assert isinstance(q, Project) and q.status()["replay_ready"] and q.info()["imported_from"] == "old.aaproj"


def test_experiment_switches_are_config_fields(monkeypatch):
    from replay.config import Config
    cfg = Config()
    assert cfg.court_tracking.use_reid is False and cfg.court_tracking.joint_tracker in (False, True)
    assert cfg.smoothing.reattach_hidden is False
    assert cfg.redetect.enabled is False and cfg.reid.enabled is False
    cfg2 = Config.from_dict({"court_tracking": {"use_reid": True}, "reid": {"enabled": True, "every_s": 0.5}})
    assert cfg2.court_tracking.use_reid is True and cfg2.reid.every_s == 0.5
    from replay import tracklets
    monkeypatch.delenv("AA_IDENTITY_REID", raising=False)
    assert tracklets.reid_enabled(cfg2) is True and tracklets.reid_enabled(cfg) is False
    monkeypatch.setenv("AA_IDENTITY_REID", "1")
    assert tracklets.reid_enabled(cfg) is True     # env still overrides for A/B runs
    from replay import court_tracker, redetect, reid
    assert redetect.RedetectConfig is type(cfg.redetect) and reid.ReidConfig is type(cfg.reid)  # re-exported
    monkeypatch.delenv("REPLAY_REATTACH", raising=False)
    assert court_tracker.reattach_enabled(cfg.smoothing) is False
    cfg.smoothing.reattach_hidden = True
    assert court_tracker.reattach_enabled(cfg.smoothing) is True
    monkeypatch.setenv("REPLAY_REATTACH", "0")
    assert court_tracker.reattach_enabled(cfg.smoothing) is False
    from replay.pipeline import _run_signature  # pass-2-only switches never invalidate pass 1
    import inspect
    assert "redetect" not in inspect.getsource(_run_signature) and "reid" not in inspect.getsource(_run_signature)
