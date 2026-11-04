# AdvancedAnalytist: user guide for coaches

AdvancedAnalytist turns a recorded basketball game into a top-down 2D replay and a coach report
(box score, shot charts, lineups, possession timeline, clips you can click). Everything runs on your
own computer. Nothing is uploaded unless you switch on an optional Gemini feature (see
[the Gemini section](#10-optional-gemini-court-detection-and-its-cost-caps)).

---

## Contents

1. [Getting started](#1-getting-started)
2. [Importing a game video](#2-importing-a-game-video)
3. [What the progress panel shows](#3-what-the-progress-panel-shows)
4. [Setting the teams and home/away](#4-setting-the-teams-and-homeaway)
5. [Attaching the scoresheet](#5-attaching-the-scoresheet)
6. [Reading the 2D replay](#6-reading-the-2d-replay)
7. [The coach report](#7-the-coach-report)
8. [Playlists (clips you can watch back to back)](#8-playlists)
9. [Exports](#9-exports)
10. [Optional: Gemini court detection and its cost caps](#10-optional-gemini-court-detection-and-its-cost-caps)
11. [Troubleshooting](#11-troubleshooting)
12. [Known limits](#12-known-limits)

---

## 1. Getting started

Start the app with `python -m replay gui` (a normal desktop window, no browser needed).

On the first start the window shows **Getting started** with three steps. You can open it again
any time with **Help > Getting started (F1)**.

1. **Import the game video** (Ctrl+I).
2. **Wait.** The analysis runs by itself in the background. Meanwhile type the team names and
   attach the scoresheet.
3. **Watch and share.** Open the Replay tab or the coach report, and export when you are happy.

Each analysed game is called a **Replay**. Your Replays are listed on the left. Next to each one is
an icon:

| Icon | Meaning |
|---|---|
| Blue | Still working on it |
| Green tick | Ready |
| Amber | Needs you (the status line under the name says what to do) |
| Red | Something went wrong (see [Troubleshooting](#11-troubleshooting)) |
| Grey | Idle |

Right-click a Replay to rename, duplicate, pin, open its folder, export or delete it. The box above
the list searches names, team names and file names, and you can sort by newest, oldest, name or size.

Tip: hover the mouse over almost any button and a short explanation appears.

### Filming tips that make a big difference

- Film from high up (balcony, top of the bleachers, or a tall tripod). Height matters most.
- Put the camera at the centre line, on the side opposite the benches, with the whole court in view.
- Do not zoom or pan if you can avoid it. A fixed wide shot gives the most reliable results.
- Landscape, 1080p or better, 30 fps or more. Let the camera run for the whole game.

---

## 2. Importing a game video

1. Click **Import video...** (or press Ctrl+I) and pick the file. MP4 is best, but MOV, MKV, AVI,
   TS/MTS, WebM and most other formats work.
2. In the small dialog, give the Replay a name. The tick box **Copy the original file into the
   Replay** is off by default: the app then keeps a playable copy and remembers where the original
   is. Turn it on if you want the original stored inside the Replay too (uses more disk space).
3. Press OK. The app makes a playable copy (large 4K video is scaled down to 1080p) and starts
   analysing straight away.

Your original file is never changed.

You can close the app while it works. The progress is kept and it continues next time. A whole
game takes roughly the length of the game on a normal laptop, and is faster with a graphics card
(any DirectX 12 card works, AMD included).

Already have a Replay from someone else? Use **File > Open Replay file...** and pick the
`.aareplay` file. Nothing is analysed again.

Where are Replays saved? In an `AdvancedAnalytist/projects` folder. **File > Show Replays folder**
opens it.

---

## 3. What the progress panel shows

While a Replay is being built, a panel above the tabs shows every phase with its own bar:
done, running or waiting, plus percent, time used, and an estimate of the time left.

| Phase | What is happening |
|---|---|
| **Import (convert / copy)** | Making the playable copy of your video. |
| **Finding live play** | A quick first pass that finds when the ball is actually in play (using the scoreboard clock and movement), so warm-ups, timeouts and breaks can be skipped. Only if "Skip breaks" is ticked in the Overview tab. |
| **Detecting & tracking players** | The big one. Finds every person in the video and follows them. |
| **Calibration during the analysis** | Working out where the court is, while the analysis runs. |
| **Court calibration** | Fitting the court lines to the picture so video positions become metres on the court. |
| **Jersey numbers** | Reading shirt numbers where they are visible. |
| **Team colours / referees** | Deciding who is on which team, and who is a referee. |
| **Building the replay** | Putting it all together into the 2D replay, finding baskets, shots and possessions. |
| **Report / exports** | Preparing the coach report data. |

Good to know:

- A note next to a phase (for example "fell back" or a long wait) is a warning, not a failure.
  The panel stays open when a phase had a problem.
- If a bar has not moved for about 45 seconds it says so. Large games can be slow in some phases.
- Afterwards the Overview tab shows "Last build: ..." with the phase times. Click it to see them.
- Changing the **Analyse** rate or **Detector size** (Overview tab) re-runs the analysis.
  Changing team colours, names or the court calibration only rebuilds the replay (seconds).
  Changing the **Processing device** (Auto / GPU / CPU) never needs a re-analysis.

---

## 4. Setting the teams and home/away

Open the **Overview** tab. Under **Game & settings** you see one row per detected jersey colour:

- a colour swatch (the colour found in the video; click it to change how the team is drawn),
- the team name box,
- **Home** or **Away**.

Type both team names and choose Home or Away, then tick **Confirmed** and press **Save**. The replay
rebuilds in a few seconds.

Other useful buttons there:

- **Roster...** type or edit player names and numbers.
- **Pick players in the video...** if the automatic team detection is wrong (dark or similar
  colours), click a few players of each team on a frame, plus a referee. Teams are then recognised
  from how the kits really look in your video. This is much more reliable.
- **Referee shirt / trousers** colours, if referees are mixed up with a team.
- **Game date** and **Location**: optional, shown in exports.
- **Broadcast overlay regions**: only for broadcast footage; screen rectangles covered by
  the scorebug or logos, so they are ignored.

If an attached scoresheet disagrees with your home/away choice (jersey numbers point the other
way), an amber message appears with a **Swap home/away** button. One click fixes the choice,
takes the names from the scoresheet and rebuilds the replay.

The Replay tab also has a swap button that exchanges home and away.

---

## 5. Attaching the scoresheet

The scoresheet (poytakirja) makes the report much more trustworthy: real player names and numbers,
and the official score and per-player points.

1. In the Overview tab, click **Add scoresheet...** (or use the link in Getting started).
2. Pick the file. Best is the federation's digital PDF, or a CSV. Photos and scans are listed in the
   file picker, but a scanned sheet may not be readable (the app tells you if so).
3. A **Scoresheet** window opens showing what was read, with the roster. It fills the Replay's
   roster and asks before replacing names you typed yourself.
4. It also lists **check this** items: places where the video and the sheet disagree. Double-click
   an item to jump to that moment in the Replay.

After this the button reads **Scoresheet...** and reopens the window.

Everything stays in the Replay folder on your computer. Nothing is sent anywhere.

Notes:

- If the Replay is a clip rather than a whole game, only the part of the sheet that matches the clip
  is used for the score.
- Personal fouls are only "official" for whole games where the sheet provides them.

---

## 6. Reading the 2D replay

Open the **Replay** tab. The video plays on one side and the top-down court on the other, always in sync.
The court is a standard FIBA court, 28 m by 15 m. Team dots show where each player is, `R` marks
referees, and the ball is drawn when it is tracked.

### Keyboard shortcuts

| Key | What it does |
|---|---|
| Space | Play / pause |
| Left / Right | Back / forward 5 seconds |
| Shift + Left / Right | Back / forward 1 second |
| `,` and `.` | Previous / next single frame |
| 1 to 5 | Playback speed (the speed list, slow to fast, up to 4x) |
| N / Shift+N | Jump to the next / previous shot or turnover (starts one second before it) |
| M | Add a mark on the timeline at the current time |
| `[` and `]` | Jump to the previous / next timeline mark |
| Ctrl+I | Import video |
| Ctrl+S | Export Replay (.aareplay) |
| F1 | Getting started |
| Ctrl+Shift+L | Show / hide the Playlists panel |
| F2 / Delete (in the list) | Rename / delete the selected Replay |

### Clicking players

**Click a dot** in the 2D court to follow that player.

### Display switches

Below the replay you can turn on or off:

- **Trails** (each player's recent path) and **Ball trail** (last two seconds).
- **Display smoothing** (display only).
- **Interpolated positions**: positions filled in across short gaps (up to 3 seconds), optionally drawn
  as hollow circles so you can tell them from real detections.
- **Hold**: keep showing players who are briefly out of view at their last place.
- **Unknown** people (could not be assigned to a team: often coaches or bench players) and **Referees**.
- **Headings** (small arrow showing which way a moving player runs).
- **Boxes** on the video (tracking boxes and IDs).
- **Names** from the Roster under the dots.
- A **confidence** style and legend: solid dots are players actually seen, faint or hollow ones are
  filled-in across a short gap.

### Timeline

Under the video the timeline shows detected baskets, shots and turnovers, and your own marks and notes.
Press M to add a mark at the current time. Automatically found marks can be edited or dismissed.

Hint: a track ID (the number near a dot) is not a person. See [Known limits](#12-known-limits).

---

## 7. The coach report

Open it from the **Stats** tab, the Export menu or the Getting started page. It becomes available
once the 2D replay is built (green tick).

Page 1 has the score, the box score for both teams, a team comparison, a momentum/runs view, shot charts,
lineups and a period (quarter) table. Page 2 has the possession list (filter by team and outcome) and the
shot quality map. **Player page...** (drop-down at the top) opens one page per player.

### Click any blue number

Every number with moments behind it is clickable. Click it and those moments open as a video playlist in
the Playlists panel (for example all of one player's turnovers, or every made three). Hover for a note about
how that number is produced.

### Measured, inferred, official: how sure is this number?

This is the most important thing to know when reading the report. Every number has a trust level,
shown by a small mark in the report:

| Mark | Level | Meaning |
|---|---|---|
| (none) | **Measured** | Counted directly from something the software sees, for example a ball going through the rim, or a ball changing team (steals). |
| `*` | **Inferred** | Derived from what was seen plus a simple rule. For example: the scorer is taken to be the player holding the ball at the release; an assist is a pass caught by the scorer within 3 seconds before a made field goal; a rebound goes to the first player to control the ball after a miss; 2 vs 3 points is decided from the shot distance. These can be wrong. |
| `†` | **Team only** | Only counted for the whole team (free throws: the shooter is not identified). |
| `‡` | **Not measured** | Not detected at all (personal fouls). Shown as a dash, never as 0. |
| green `°` | **Official** | Taken from the scoresheet you attached. Where the sheet knows a number, it replaces the detected one. |

Rule of thumb for coaches:

- Trust **official** numbers fully (they come from the scorer).
- Treat **measured** numbers as reliable but check unusual ones in the video.
- Treat **inferred** numbers as good estimates for trends and patterns. Click the number and watch the
  clips before you use one for a single player in an important decision.
- A dash means "we do not know", not "zero".

Without a scoresheet the footer says the numbers come from automatic detection and are not official.
Attaching the scoresheet is the single best way to make the report more trustworthy.

### Report tools

- **Save PDF...** writes the report as a printable PDF.
- **Export** menu: box score as CSV (one file per team plus a team comparison), Excel workbook (when
  available), or a printable PDF with one page per player.
- **Copy summary** puts a short text (score, top scorers, shooting, key runs) on the clipboard, ready to
  paste into a message.

### The shot quality map

It shows points per shot by court zone, compared with what the zone would normally be worth given this
game's shooting percentage there. Made baskets are measured at the rim; the shot place comes from the
shooter's foot position, so it is inferred and accurate to about half a metre.

---

## 8. Playlists

A playlist is a list of clips you can watch back to back in the Replay tab's video player. The 2D view
follows the video.

- Open the **Playlists** panel (right side, Ctrl+Shift+L) or the **Playlists** menu.
- **New playlist...** lets you pick a filter such as all our turnovers, all their fast breaks, or every
  possession of #12.
- **Quick playlist** gives ready-made filters.
- Clicking a number in the coach report creates a playlist from it.
- Right-click a clip for: Play this clip, Play all from here, Mark as reviewed / not reviewed, Note..., and
  Remove from playlist.
- The playlist menu also has Edit filter..., Rebuild from latest analysis, Rename... and Delete playlist.
- A playlist can be exported as a **CSV** list, or as one **MP4** video with title cards in between
  (options: title cards, an intro card, only clips you have not reviewed yet, audio on or off, and a
  quick export that is faster but gives a bigger file).

Playlists are saved with the Replay and travel inside the `.aareplay` file.

---

## 9. Exports

Use the **Export** menu, or right-click a Replay in the list. Exports are available once the 2D replay is built.

| Export | What you get |
|---|---|
| **Export Replay (.aareplay)** (Ctrl+S) | One file with everything: the original video, the playable copy, the analysis, the replay, calibration, game info, timeline marks and playlists. Open it on any computer with **File > Open Replay file...** without analysing again. Large for long games, but quick to write. |
| **Coach report (view / PDF)** | The report, as a PDF, with an optional page per player. |
| **Box score as CSV / Excel** | From the coach report window's Export menu. |
| **Simple data export (for AI)** | A new folder named like `<name>_data_<date-time>` with CSV and JSON files and a `README_FOR_AI.md` that explains the data, so you can give the folder to an AI assistant. Earlier exports are never overwritten. |
| **Individual files** | The tracking CSV/JSON, the processing report, or a stand-alone HTML replay you can open in any browser. |
| **Playlist CSV / MP4** | From the Playlists panel (see above). |

The data export folder includes: `players.csv` (positions), `ball.csv`, `events.csv` (detected baskets with a
running score that is detected, not official), timeline marks, `track_summary.csv`, `possession_summary.csv`,
`game.json`, a stand-alone `replay_2d.html` and the report.

If an export says "Nothing to export yet", build the 2D replay first (wait for the green tick).

---

## 10. Optional: Gemini court detection and its cost caps

**This is off by default and experimental.** In a first test it found too few court views to be reliable.
Most games do not need it: the normal way is to place a few court landmarks yourself in the
**Court fix (advanced)** tab. Gemini is only meant for hard footage such as broadcast streams with cuts and zoom.

What it does: it asks Google's Gemini AI to point out court markings in a small number of video frames,
which helps the app work out where the court is.

What leaves your computer (only when you tick the box): at most 60 still frames per game, shrunk to
768 pixels wide, sent to Google using **your own** Google AI Studio API key. The key is stored on your
computer (encrypted for your Windows user) and is never put in Replay files, exports or logs.

To use it:

1. Add your Google AI Studio API key in the **AI** menu.
2. In the **Court fix (advanced)** tab, tick **Use Gemini to find the court (experimental)**.

### The cost caps (hard limits)

| Limit | Default |
|---|---|
| Frames per game | 60 |
| Spend per game | 0.10 EUR |
| Spend per calendar month | 2.00 EUR |

- You can change the two euro caps in the Court fix tab (**Max EUR per game** / **per month**).
- Before every request the app estimates its worst-case cost. If a request could push you over a cap, it
  is not sent and you see "Gemini budget reached".
- It uses the cheapest model by default, one request per frame, and one retry at most on a network error
  (never after an API error).
- Frames already answered are remembered per Replay, so rebuilding never sends them again.
- A local spending log is kept in your app-data folder.
- The caps count 1 US dollar as 1 euro, so the real cost is a little lower. Prices change: check Google's
  current price list if the exact amount matters to you.

If you do not tick the box, nothing is ever sent to Google for court detection.

(The separate **AI** menu, a tactical review, also uses your own Google key and sends data to Google when you
run it. Only run it if you are comfortable with that.)

---

## 11. Troubleshooting

| What you see | What it likely means | What to do |
|---|---|---|
| "No video stream found - is this a video file?" | The file is not a video, or is damaged/incomplete. | Try playing it in another player. Re-download or re-copy it. |
| Import fails with a message from ffmpeg and an exit code | The video could not be converted (unusual codec, damaged file, disk full). | Check free disk space. Try converting it to MP4 with another tool and import that. |
| "This Replay has no playable video" | The playable copy is missing. | Re-import the original, or open the `.aareplay` file again. |
| Amber icon | The Replay needs you. | Read the status line under the name. Often it needs the court calibration or a team confirmation. |
| Red icon / error in the progress panel | A phase failed. | Open the phase details (click "Last build"). Press the main action button to try again. Re-running is safe: finished work is kept. |
| Analysis stopped or app closed | Nothing is lost. | Select the Replay and press the action button to continue. |
| Coach report: "available once the 2D replay is built" | The replay is not finished yet. | Wait for the green tick. |
| "Nothing to export yet: build the replay first" | Same as above. | Wait for the green tick. |
| Team dots swapped, wrong home/away | Colours were matched the wrong way round. | Change Home/Away in the Overview tab and Save, or use **Swap home/away**. Use **Pick players in the video...** if colours are similar. |
| Coaches/bench people shown as players | They wear team colours near the court edge. | Untick **Unknown** in the Replay tab, or fix teams with **Pick players in the video...**. |
| Lines of the court look off in the 2D view | Court calibration is not right. | Open **Court fix (advanced)**: click the court landmark in the diagram, then the same spot in the video. Use at least 4 points, spread out. Press Enter to save. The error should be small (under about 5 px). |
| Many frames with no positions | Camera panned fast, cut, or the court was not visible. | Add calibration keyframes in the Court fix tab at those times. The app never guesses positions. |
| Scoresheet "could not be read" | The file is not a recognised scoresheet. | Use the digital PDF or a CSV. |
| Scanned scoresheet message | A photo or scan cannot be read reliably. | Use the federation's digital PDF if you can. |
| Scoresheet and video disagree ("check this") | Either side may be wrong. | Double-click the item to see that moment in the video. |
| "Gemini budget reached" | A spending cap would have been crossed. | Raise the cap in the Court fix tab (only if you want to), or place landmarks by hand. |
| Gemini finds few or no court views | Known weakness of the experiment. | Use manual landmarks instead. |
| Slow analysis | CPU only, or other programs using the graphics card. | Set **Processing device** to Auto or GPU. Use a smaller detector size or analyse every 3rd frame (default). Close games and other heavy programs. |
| App will not start | Python or packages missing. | Reinstall following the README setup section. |

If something still looks wrong, note the Replay name and the message and tell the person who supports
the app. The processing report (Export > Individual files) records what happened during the build.

---

## 12. Known limits

Please read these. They explain the cases where the report can be wrong.

- **Positions come from one camera.** The app works out where each player's feet are. Jumping,
  hidden feet and rough boxes give errors of roughly 0.3 to 1 metre near the camera and more on the far
  side. It cannot see through other players.
- **Only what the camera sees is tracked.** If the camera shows half the court, the other half is simply
  empty at that time.
- **Track IDs are not people.** One player usually gets several IDs over a game (after being hidden, leaving
  the picture, or fast pans). Shirt numbers are read where they are clear, and only shown when read the same
  way at least three times. An empty number means "not read", not "no number". Per-player stats lean on the
  roster and scoresheet to be useful: attach the scoresheet.
- **Teams are told apart by colour.** If both teams wear similar colours, results suffer. Use **Pick
  players in the video...**.
- **Broadcast footage is harder.** Panning, zooming, cuts, replays, close-ups and on-screen graphics
  make calibration much less reliable. The app carries your calibration through the camera motion, but
  drift builds up between your keyframes. Fast whip pans, blur and cuts leave gaps (no calibration, so
  no positions). Add keyframes often (at least every 30 to 60 seconds), mark graphics areas as overlay
  regions, and expect to review the numbers. A fixed, high, wide shot of the whole court is far better.
- **Lens distortion is not corrected.** Wide phone lenses bend lines near the edges.
- **Some statistics are inferred, not seen.** Scorer, assists, rebounds, turnovers and 2 vs 3 points
  rely on rules (see [the report section](#7-the-coach-report)). Free throws are per team only, and
  personal fouls are not detected. Without a scoresheet, the score is detected, not official.
- **The ball is harder than players.** When it is in the air or hidden, its position is an estimate.
- **The detector is general, not basketball-specific.** Unusual views or very crowded courts reduce quality.
- **No identity across games.** Each Replay stands alone.

Rule of thumb: use the report for patterns, trends and finding clips worth watching. For anything that
matters, click through to the video and check it with your own eyes.

---

*Copyright Onni Kauppinen 2026.*
