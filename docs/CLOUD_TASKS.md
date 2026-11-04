# Cloud tasks: brief for unattended Claude Code cloud sessions

AdvancedAnalytist is a native PySide6 desktop app for basketball coaches. It turns game video into 2D tracking and coach
analytics (`replay/`). The live app and all game data are on the owner's Windows PC (AMD GPU, DirectML). This repo holds
**code only**.

## Hard rules
1. **Never add game data, video, images, rosters, scoresheets, player names, model weights or measurement logs to this
   repo** (GDPR: the players are minors). `.gitignore` is an allow-list; never weaken it, and never use `git add -f`.
   Tests use fake names only (e.g. "Testaaja", "Zyxa", "Dora").
2. Work on a branch named `cloud/<date>-<topic>`. Push it and open a **draft PR**. Never push to `main` and never merge.
   The owner's local session tests on real games and merges.
3. Every behaviour change goes behind a `Config` field that is **OFF by default**, unless it is a pure bug fix with a test.
   Default output must stay identical.
4. Licence-permissive dependencies only (no AGPL/Ultralytics). No new downloads of models or datasets.
5. Never store or print API keys. Nothing in the code may send player names to an AI service.
6. Do not bump `replay/__init__.py` `__version__`; the owner's release does that.

## Setup (Linux cloud)
```
pip install -e . pytest
QT_QPA_PLATFORM=offscreen python -m pytest -q -x --deselect "tests/test_live.py::test_live_session_follows_recording_and_ends_as_normal_analysis"
```
Tests that need local videos or models skip themselves. Tests that fail only because the platform is Linux (DirectML,
Windows paths, fonts): note them in the PR and leave them alone.

## Task queue (top first; finish one with tests, commit, then take the next)
1. **Corrections ledger (pro-tracking plan, phase 2 core).** New `replay/corrections.py`:
   - Schema `{version, ops: [{t0, t1, kind: split|merge|assign|team|role|hide|anchor|make, ...}]}`, load/save next to
     the project's other JSON files.
   - `apply_identity(ops, pieces)`, wired as hard constraints into `replay/tracklets.py` `link_tracks`: a forced link has
     zero cost and vetoes its competitors; a cut has infinite cost. Use a hook that does nothing when there are no ops.
   - `apply_events(ops)`: make/boundary constraints for `replay/game_state.py` possession v6 (`v6_vote`).
   - `tests/test_corrections.py`, using `tests/fake_project.py` / `tests/synthetic.py` fixtures. Corrections must survive
     a rebuild byte-for-byte.
2. **Possession v6 confidence cap.** In `replay/game_state.py`, when a possession's team is inferred only from a make (no
   holder ever seen, `seen_s == 0`), cap its confidence (e.g. at 0.6) instead of reporting 0.99. Add a test and check that
   `tests/test_possession_model.py` and `tests/test_ai_light.py` still pass.
3. **Court solver `refine_H` bug** (`replay/court_solver.py`): the refinement step never moves `H` away from `H0`. Find out
   why using synthetic courts (`tests/synthetic_court.py`), fix it, and add a test where a perturbed `H0` gets closer to
   the true `H`.
4. **Benchmark `--column player_id|track_id`** in `replay/benchmark.py` / `replay/bench_metrics.py`, plus a "corrections
   per game" estimate (ID transfers + cross-team links + bench IDs drawn). Unit tests on synthetic data.
5. **Unsure-segment detector**: rank segments with `id_confidence < 0.6`, re-attachments over 4 s, cross-team evidence,
   bench-zone IDs and number conflicts, from existing pass-2 outputs. Pure function plus tests; no GUI yet.

## Reporting
The PR description lists, per task: what changed, the flag name (if any), tests added, and the suite result. It also
lists anything that must be checked on real games on the owner's PC. Keep it short.
