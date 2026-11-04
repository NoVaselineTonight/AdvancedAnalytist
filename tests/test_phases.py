"""Phase panel model (replay/gui/phases.py): follows the job chain, ETAs, notes, persistence."""

import json

from replay.gui.phases import PhaseTracker


def _states(tr):
    return {r["key"]: r["state"] for r in tr.snapshot(now=tr._now)["rows"]}


def test_follows_the_job_chain_and_persists(tmp_path):
    tr = PhaseTracker(tmp_path, video_s=180.0)
    t = 1000.0
    lines = []

    def obs(kind, label=None, prog=0.0, msg="", err=None):
        tr._now = t
        tr.observe(kind, label, prog, msg, err, lines, now=t, status={})

    obs("import", prog=0.5)
    t += 2
    obs("analyze", "Preliminary analysis", 0.4)
    assert _states(tr)["import"] == "done" and _states(tr)["prelim"] == "running"
    t += 5
    obs("analyze", None, 0.1)
    t += 30
    obs("analyze", None, 0.5)
    snap = tr.snapshot(now=t)
    row = {r["key"]: r for r in snap["rows"]}
    assert row["pass1"]["state"] == "running" and 25 <= row["pass1"]["eta"] <= 40  # half done in 30 s
    assert row["calib1"]["state"] == "running" and row["calib1"]["eta"] is None
    assert snap["overall_eta"] and snap["overall_eta"] > row["pass1"]["eta"]  # later phases are estimated too
    lines.append("Frame taps (finish wait 130.6s): {...}")
    t += 30
    obs("autocal", None, 0.2, "Refining the court")  # analyze -> autocal (polling may miss the end of a job)
    s = _states(tr)
    assert s["pass1"] == "done" and s["calib1"] == "done" and s["autocal"] == "running"
    assert any("waited" in n["text"] for n in tr.ph["pass1"]["notes"])
    t += 100
    obs("autocal", None, 0.2)  # no progress for 100 s
    assert {r["key"]: r for r in tr.snapshot(now=t)["rows"]}["autocal"]["stalled"] >= 99
    t += 10
    obs("build", None, 0.0, "Projecting positions onto the court")
    lines += ["camera type: gym", "lineup: 51 pieces into 10 slots", "jersey numbers: 17 of 166 tracks {...}",
              "game state: 7 possessions"]
    t += 20
    obs("build", None, 0.0)
    s = _states(tr)
    assert s["autocal"] == "done" and s["teams"] == "done" and s["jersey"] == "done" and s["report"] == "running"
    t += 5
    obs(None, None, 1.0, "Replay ready: calibration valid on 98% of frames, 10 people per frame")
    assert set(_states(tr).values()) == {"done"}
    saved = json.loads((tmp_path / "results" / "phases.json").read_text(encoding="utf-8"))
    assert saved["autocal"]["state"] == "done" and saved["jersey"]["notes"][0]["text"].startswith("numbers read on 17")
    # a Replay opened later shows the same phases as done, with their durations
    tr2 = PhaseTracker(tmp_path, 180.0)
    tr2._now = t
    tr2.observe(None, None, 0.0, "", None, [], now=t, status={"imported": True, "analyzed": True, "calibrated": True,
                                                              "replay_ready": True})
    rows = {r["key"]: r for r in tr2.snapshot(now=t)["rows"]}
    assert rows["build"]["state"] == "done" and rows["autocal"]["elapsed"] is not None


def test_errors_stops_and_hand_calibration(tmp_path):
    tr = PhaseTracker(None, 60.0)
    tr._now = 1.0
    tr.observe("analyze", None, 0.3, "", None, [], now=1.0, status={})
    tr._now = 5.0
    tr.observe(None, None, 0.3, "Stopped. Progress is kept; press Continue analysis to go on.", None, [], now=5.0)
    assert _states(tr)["pass1"] == "stopped"
    tr.observe("build", None, 0.0, "", None, [], now=9.0)  # calibrated by hand: no autocal job
    tr._now = 9.0
    assert _states(tr)["autocal"] == "skipped"
    tr.observe(None, None, 0.0, "", "Building 2D replay failed: ValueError: x", [], now=12.0)
    tr._now = 12.0
    s = _states(tr)
    assert s["build"] == "failed" and any(n["level"] == "error" for n in tr.ph["build"]["notes"])


def _snap(note_level=None, state="done"):
    from replay.gui.phases import ORDER
    rows = [{"key": k, "title": k, "state": state, "progress": 1.0, "elapsed": 10.0, "eta": None, "stalled": None,
             "notes": [{"text": "fell back to the full search", "level": note_level}] if note_level and k == "autocal"
             else []} for k in ORDER]
    return {"rows": rows, "overall_eta": None, "running": False}


def test_panel_shows_only_while_building_and_briefly_after():
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    from replay.gui.app import PhasePanel
    pnl = PhasePanel()
    run = _snap(state="running")
    run["running"] = True
    pnl.sync("p", run, busy=True, now=100.0)
    assert not pnl.isHidden() and pnl.close_btn.isHidden()
    pnl.sync("p", _snap(), busy=False, now=160.0)  # just finished: "Done in 1:00"
    assert not pnl.isHidden() and "Done in 1:00" in pnl.overall.text()
    pnl.sync("p", _snap(), busy=False, now=166.0)  # 6 s later: gone
    assert pnl.isHidden()
    pnl.show_details()  # "Last build ... details"
    pnl.sync("p", _snap(), busy=False, now=200.0)
    assert not pnl.isHidden() and "Last build" in pnl.overall.text()
    pnl.dismiss()
    pnl.sync("p", _snap(), busy=False, now=201.0)
    assert pnl.isHidden()
    # a fallback / error note keeps it up until the x
    pnl.sync("q", run, busy=True, now=300.0)
    pnl.sync("q", _snap("warn"), busy=False, now=310.0)
    pnl.sync("q", _snap("warn"), busy=False, now=400.0)
    assert not pnl.isHidden() and "notes" in pnl.overall.text()
    pnl.close_btn.click()
    pnl.sync("q", _snap("warn"), busy=False, now=401.0)
    assert pnl.isHidden()
