"""Job functions run in a separate process by tests/test_realuse_qa.py (must be importable by the child)."""

import os
import time


def ok(p, progress, cancel):
    progress(0.5, "half way")
    return {"valid_pct": 90.0, "people": 9.5}


def empty(p, progress, cancel):
    return {"valid_pct": 0.0, "people": 0.0}


def boom(p, progress, cancel):
    raise ValueError("court too small")


def crash(p, progress, cancel):
    print("about to crash", flush=True)
    os._exit(3)


def slow(p, progress, cancel):
    progress(0.1, "working")
    end = time.time() + 60
    while time.time() < end:  # ignores the cancel request, like the build (no cancel points)
        time.sleep(0.05)
    return {}


def stateful(p, progress, cancel, restart=False):
    st = progress.state  # what the analysis uses: attribute writes go to the GUI's JobState
    st.label = "Preliminary analysis"
    st.message = f"restart={restart}"
    st.progress = 0.25
    st.started = 123.0
    return {}
