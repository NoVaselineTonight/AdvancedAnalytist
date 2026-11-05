"""Total elapsed clock of the phase panel (replay/gui/phases.py PhaseTracker.run, app.PhasePanel)."""

import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from replay.gui.phases import ORDER, PhaseTracker, fmt_clock  # noqa: E402


def _snap(state="done"):
    rows = [{"key": k, "title": k, "state": state, "progress": 1.0, "elapsed": 1.0, "eta": None, "notes": [],
             "stalled": None} for k in ORDER]
    return {"rows": rows, "overall_eta": None, "running": False}


def test_fmt_clock():
    assert fmt_clock(7) == "0:07" and fmt_clock(252) == "4:12" and fmt_clock(3729) == "1:02:09"


def test_total_elapsed_clock_runs_across_phases_and_survives_restart(tmp_path):
    tr = PhaseTracker(tmp_path, video_s=180.0)
    # the job began at 1000, but the panel first looks at 1040: elapsed counts from the recorded start
    tr.observe("import", None, 0.5, "", None, [], now=1040.0, status={}, started=1000.0)
    assert tr.snapshot(now=1040.0)["run"]["elapsed"] == 40.0
    tr.observe("analyze", None, 0.2, "", None, [], now=1100.0, status={}, started=1090.0)
    assert tr.snapshot(now=1252.0)["run"] == {"started": 1000.0, "ended": None, "running": True, "elapsed": 252.0}
    tr.observe(None, None, 0.0, "", None, [], now=1300.0, status={})  # the chain ends
    snap = tr.snapshot(now=5000.0)
    assert snap["run"]["running"] is False and snap["run"]["elapsed"] == 300.0
    tr2 = PhaseTracker(tmp_path, video_s=180.0)  # another session / project switch: the total is remembered
    tr2.load({})
    assert tr2.snapshot(now=9000.0)["run"]["elapsed"] == 300.0
    tr.observe("build", None, 0.1, "", None, [], now=2000.0, status={}, started=2000.0)  # much later: a new run
    assert tr.snapshot(now=2005.0)["run"]["started"] == 2000.0


def test_unfinished_run_of_an_earlier_session_does_not_tick(tmp_path):
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "phases.json").write_text(json.dumps({"_run": {"t0": 5.0, "t1": None}}),
                                                     encoding="utf-8")
    tr = PhaseTracker(tmp_path)
    tr.load({})
    assert tr.snapshot(now=100.0)["run"] is None


def test_panel_shows_live_elapsed_and_eta_only_when_known():
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    from replay.gui.app import PhasePanel
    snap = _snap("running")
    snap["run"] = {"started": 1000.0, "ended": None, "running": True, "elapsed": 0.0}
    snap["running"] = True
    assert "Elapsed 4:12" in PhasePanel.elapsed_text(snap, "", now=1252.0)
    assert "left" not in PhasePanel.elapsed_text(snap, "", now=1252.0)  # no invented ETA
    assert "About 1:00 left" in PhasePanel.elapsed_text(snap, "About 1:00 left", now=1252.0)
    pnl = PhasePanel()
    pnl.sync("p", snap, busy=True, now=1252.0)
    assert "Elapsed" in pnl.overall.text()
    done = _snap()
    done["run"] = {"started": 1000.0, "ended": 1681.0, "running": False, "elapsed": 681.0}
    pnl.sync("p", done, busy=False, now=1682.0)
    assert "Done in 11:21" in pnl.overall.text()
