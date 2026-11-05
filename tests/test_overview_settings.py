"""Overview: the Details button always shows something; "Settings used" summary and full list."""

import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

from fake_project import make_project  # noqa: E402

LOG = ("Analysing 1 in-play spans: 2435 of 2435 samples\nLoading detector ...\n"
       "Detector: rfdetr-medium on GPU (DirectML) (RF-DETR COCO weights: Apache-2.0 (Roboflow); ONNX Runtime: MIT); "
       "analysing 9.99 frames/s of video\n"
       "detect+track:   0%|          | 0/2435 [00:00<?, ?frame/s]\r"
       "detect+track:   1%|1 | 27/2435 [00:00<01:22, 29.35frame/s]\rPass 1 done in 79s\n")


@pytest.fixture(scope="module")
def app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture
def win(app, tmp_path, monkeypatch):
    from PySide6.QtCore import QSettings

    from replay.gui import app as A
    monkeypatch.setattr(A, "QSettings", lambda *a: QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat))
    projects = tmp_path / "projects"
    projects.mkdir()
    make_project(projects / "game-aaa111")
    w = A.MainWindow(projects)
    monkeypatch.setattr(w.jobs, "submit", lambda *a, **kw: None)
    yield w
    w.close()


def _overview(win, details):
    p = win.store.get("game-aaa111")
    t = win.overview_tab
    t.load(p)
    t.report_details_btn.setChecked(details)
    t.refresh()
    return p, t.report.toPlainText()


def _config(p, **extra):
    from replay.config import Config, used_config_dict
    d = used_config_dict(Config())
    d.update(extra)
    (p.results / "config.used.json").write_text(json.dumps(d), encoding="utf-8")
    return d


def test_details_with_no_report_shows_what_ran_and_logs(win):
    p = win.store.get("game-aaa111")
    (p.results / "report.json").unlink(missing_ok=True)
    (p.results / "analyze.log").write_text(LOG, encoding="utf-8")
    (p.results / "autocal.log").write_text("", encoding="utf-8")
    (p.results / "phases.json").write_text(json.dumps({"pass1": {"state": "done", "s": 97.3, "notes": []}}),
                                           encoding="utf-8")
    p.calibration_path.unlink()  # the court was never calibrated
    _, short = _overview(win, False)
    assert "Press Details" in short and "Settings used" in short
    _, txt = _overview(win, True)
    assert "No replay has been built yet" in txt
    assert "Detecting & tracking players" in txt and "took 1:37" in txt
    assert "Not done yet" in txt and "Court calibration" in txt
    assert "2,435" in txt  # frames analysed, from the log
    assert "nothing was written" in txt  # empty autocal.log
    assert str(p.results) in txt
    assert "detect+track" not in txt  # progress-bar redraws are filtered out of the log tails


def test_details_with_no_report_and_no_files_still_shows_something(win):
    p = win.store.get("game-aaa111")
    (p.results / "report.json").unlink(missing_ok=True)
    _, txt = _overview(win, True)
    assert "No replay has been built yet" in txt and "not recorded" in txt


def test_details_reacts_to_the_button_and_survives_an_old_report(win):
    p = win.store.get("game-aaa111")
    (p.results / "report.json").write_text(json.dumps({"calibration": {}, "people": {}}), encoding="utf-8")
    _config(p)
    _, short = _overview(win, False)
    _, full = _overview(win, True)
    assert full != short and "All settings" in full and "Frames analysed" in full
    (p.results / "report.json").write_text(json.dumps({
        "frames_processed": 5, "process_fps": 10.0, "calibration": {"valid_pct": 90.0},
        "people": {"tracks_output": 3}, "failed_ranges": {"no_calibration": [], "no_players_tracked": []}}),
        encoding="utf-8")
    _, full = _overview(win, True)
    assert "Player tracks" in full and "All settings" in full


def test_summary_in_plain_words(win):
    p = win.store.get("game-aaa111")
    _config(p)
    (p.results / "analyze.log").write_text(LOG, encoding="utf-8")
    _, txt = _overview(win, False)
    assert "rfdetr-medium on GPU (DirectML)" in txt and "RF-DETR COCO" not in txt
    assert "10.0 per second" in txt or "9.99" in txt or "10.0" in txt
    assert "FIBA" in txt and "Player re-ID" in txt and "Your corrections" in txt
    import replay as _r
    assert f"version {_r.__version__}" in txt


def test_full_settings_highlight_changes_against_defaults():
    from replay.config import Config
    from replay.gui import overview_info as OI
    cfg = Config().to_dict()
    cfg["app_version"] = "9.13"
    cfg["reid"]["enabled"] = True
    cfg["frame_stride"] = 2
    assert sorted(OI.settings_diff(cfg, Config().to_dict())) == ["frame_stride", "reid.enabled"]
    h = OI.settings_html(cfg)
    assert "2 differ" in h and "default: 3" in h and "default: off" in h and "Player detector" in h


def test_missing_and_old_files(tmp_path):
    from replay.gui import overview_info as OI
    assert OI.version_text(None) == "not recorded"
    assert OI.version_text({"frame_stride": 3}) == "unknown (built before 9.13)"
    assert OI.version_text({"app_version": "9.13"}) == "version 9.13"
    assert OI.read_settings(tmp_path) is None
    (tmp_path / "config.used.json").write_text("{half a file", encoding="utf-8")
    assert OI.read_settings(tmp_path) is None
    assert dict(OI.summary_rows(tmp_path, {}, None))["Built with"] == "not recorded"
    (tmp_path / "config.used.json").write_text(json.dumps({"frame_stride": 3, "detector": {"variant": "small"}}),
                                               encoding="utf-8")
    rows = dict(OI.summary_rows(tmp_path, {"court_standard": "NBA"}, None))
    assert rows["Built with"] == "unknown (built before 9.13)" and "rfdetr-small" in rows["Player detector"]
    assert "NBA" in rows["Court"]


def test_app_version_is_recorded_but_never_a_setting(capsys):
    from replay import __version__
    from replay.config import Config, used_config_dict
    d = used_config_dict(Config())
    assert d["app_version"] == __version__
    Config.from_dict(json.loads(json.dumps(d)))
    assert capsys.readouterr().out == ""  # loads silently
    assert "app_version" not in Config().to_dict()


def test_volume_slider_wired_remembered_and_compact(win, tmp_path):
    from PySide6.QtCore import QSettings
    rt = win.replay_tab
    assert rt.volume.minimum() == 0 and rt.volume.maximum() == 100 and rt.volume.toolTip()
    assert rt.volume.value() == 100 and abs(rt.audio.volume() - 1.0) < 1e-6  # default 100
    rt.volume.setValue(35)
    assert abs(rt.audio.volume() - 0.35) < 1e-3 and not rt.audio.isMuted()
    rt.mute_btn.setChecked(True)
    assert rt.audio.isMuted()
    st = QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat)
    assert int(st.value("volume")) == 35 and str(st.value("muted")).lower() == "true"
    assert rt.volume.width() <= 80 and rt.mute_btn.width() <= 40  # compact: leaves the other controls alone
    rt.player.setSource(__import__("PySide6.QtCore", fromlist=["QUrl"]).QUrl())
    rt._update_audio_available()
    assert rt.volume.isEnabled()  # nothing loaded: not locked
