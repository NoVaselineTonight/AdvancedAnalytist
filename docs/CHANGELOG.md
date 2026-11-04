# Advanced Analytist: Changelog

Newest version first. Each entry lists what changed, written from a coach's point of view.

## 9.10 (2026-10-20)

- The 2D replay estimates where players and the ball are when they are out of camera view. The estimates are rebuilt and look more believable.
- The coach report has a new page 2 with a possession timeline and a shot-quality map.
- The 2D viewer is easier to use: click and hover on players, a ball trail, and keyboard shortcuts.
- Players drawn off the court are flagged, so they are not mistaken for players on the floor.
- Predicted player speeds are capped, so dots no longer jump across the floor.
- The app starts faster, and replays load in the background.
- A replay can be viewed without a court calibration file, with a rougher picture.
- The analysis window fits 1280x720 screens.
- Automatic court calibration no longer delays the end of the first analysis pass.
- When a game has no score bar, the score bar reader gives up early instead of wasting time.

## 9.09 (2026-10-19)

- Team and settings tabs stay usable while a replay is being built, so you can set teams during import.
- The Overview screen scrolls on small windows, and the build progress rows fit better.

## 9.08 (2026-10-19)

- Minutes are measured on the game clock. A player's minutes count only while the clock runs, and each team totals about five players times the game length.
- On full games, the app reads the score, clock and quarter from the video's score bar. Baskets are checked against the score bar.
- A build progress panel shows every build phase with its own progress bar and time estimate. It hides when nothing is running.
- The coach report can be exported to CSV, PDF, or a text summary that you can copy.
- Clearer error messages and tooltips, a better replay list, and a help screen.
- Short clips calibrate faster (about 311 s down to 131 s in testing).
- Crash protection, a startup self-check, and log rotation. Problems are logged instead of silently losing work.
- Five bug fixes.

## 9.07 (2026-10-19)

- The app reads the score from the video's score bar during full games.
- Short clips are matched to the scoresheet automatically when the score bar confirms the quarter and score (4 of 4 test clips).

## 9.06 (2026-10-18)

- Baskets are aligned with the scoresheet. Per-player points match the scoresheet totals (average per-player error went from 7.3 to 1.3 points in testing).
- The coach report shows a full-game summary of mismatches, and flags low-confidence notes.
- Short clips get a quarter picker.
- Experimental court detection with Gemini is available. It is off by default and has spending caps (0.10 EUR per game, 2 EUR per month).
- Fixes for the app freezing during builds.

## 9.05 (2026-10-18)

- The coach report is split by period. Scoresheets can be merged into it, and the app can match teams automatically and asks before swapping them.
- Better handling of broadcast-style camera angles, and better detection of team colours and referees.
- Fix for the benchmark cache.

## 9.04 (2026-10-18)

- New coach report: box score, team comparison, shot charts, scoring runs, five-player units, and individual player pages.
- Stats are marked as measured or inferred. Clicking a stat opens its clips as a playlist. The report can be exported to PDF.

## 9.03 (2026-10-18)

- Cleaner 2D view by default. Confidence halos are off, short gaps are handled better, players off the court are hidden, and overlapping dots are spread apart.
- Implausible jumps (over 3 m) are faded, and drawn speed is capped at 9 m/s.

## 9.02 (2026-10-18)

- Jersey-number reading during the first pass is off by default. It was slowing the first pass by about half on large files. Team colours and calibration still run.
- Projects made with 9.0 or 9.01 recalculate team colours once when opened, using the video.

## 9.01 (2026-10-18)

- Video import uses the graphics card to decode and encode video. A 30-second AV1 clip imports in about 7.5 s, roughly four times real time.
- If the graphics card path fails, the app falls back to the CPU. The graphics card path can be turned off.

## 9.0 (2026-10-18)

- Faster builds: court calibration starts during the first analysis pass, and the court can be locked earlier.
- Graphics-card video encoding, with CPU fallback.
- The version number is shown in the window title.

## Known limitations

- **Court calibration on broadcast and TV-style footage is unreliable.** Automatic calibration can fail on these clips (one NCAA test aligned only 1 of 120 frames). The manual calibration overlay can also look wrong. Positions on these clips are rough.
- **Ball tracking is not reliable yet.** The ball can jump or be placed wrongly, especially around passes and free throws. A dedicated ball detector is not yet in use.
- **Basket attribution needs a scoresheet or score bar.** Without either, points may be credited to the wrong team or player.
- **Players can get a new ID after leaving the view.** One player's stats can be split across two IDs. Some players are also missed or counted as extra.
- **Five-on-court tracking can miss substitutions.** The exact five-player mode is opt-in.
- **Jersey numbers may be missing.** Jersey reading is off by default, so numbers may come only from the scoresheet or roster.
- **Dark kits and similar colours can confuse teams.** Dark uniforms can be read as referees or unknown, and referee detection is weak on some venues.
- **Known issue in 9.10:** a change to referee detection for broadcast footage creates false referee tracks, mostly spectators and bench players. It will be reverted in 9.11.
- **Estimated positions are estimates.** Positions for players out of view are predicted, and the coach report marks them as inferred.
- **Several improvements are off by default** while they are being tested: re-identification by appearance, foot-point refinement, pile-up handling, the learned court detector, and the Gemini court detection.
- **Live mode** (analysing a game while it is still being filmed) is not part of these versions.
- **Full-game builds are still slow.** Expect tens of minutes on the current PC.
