# AdvancedAnalytist – Privacy Policy

Version 1.0 · effective 5 November 2026

This policy explains what data the AdvancedAnalytist software ("the Software") handles, where it goes, and who is
responsible for it. It is part of the Terms of Use.

## 1. Short version
* **The Software runs on your computer. Your videos, rosters, names, statistics and Replays stay on your computer.**
* **The Author (Onni Kauppinen) receives none of your data.** The Software has no user accounts, no telemetry, no
  usage analytics, no crash upload, no advertising and no tracking.
* Data leaves your computer **only** if you turn on an optional AI feature with your own API key (section 4), when a
  model is downloaded the first time it is used (section 5), or when you yourself export and share a file.
* **You (or your club) are the controller** of the personal data in your videos and Replays and are responsible for
  handling it lawfully (section 3).

## 2. Data the Software handles on your computer
Stored in the Replays (projects) folder and in your user settings on your computer:

* game videos and frames taken from them, which show players, coaches, referees and spectators, often children;
* tracking results: positions, tracker IDs, team assignments, jersey numbers read from the video;
* rosters and scoresheets you import: player names, jersey numbers and, if present, licence numbers;
* statistics, reports, playlists, notes, corrections and logs;
* settings, including your AI API keys, which are **encrypted with Windows DPAPI** and readable only by your Windows
  user account, and a local ledger of AI usage and costs.

The Author has no access to any of this. You can delete it at any time by deleting the Replay (or the Replays folder),
and the keys by clearing them in the AI settings.

## 3. Who is responsible (GDPR)
The Author provides software only. Because the Author does not receive, store or access your data, **the Author is
neither a controller nor a processor** of it under the GDPR.

**You, or the club, team or school you work for, are the controller.** That means you decide why and how the data is
processed and you must, among other things:

* have a lawful basis for recording and analysing the people in the video (for example legitimate interest of the
  club or consent), and get the consent of guardians where it is needed for children;
* tell the players, and the guardians of minors, what is recorded and why, how long it is kept and who sees it;
* follow your club's, school's and federation's policies on filming and on children's data;
* keep the data secure (for example a password-protected Windows account and disk encryption) and only as long as
  needed;
* answer requests to see, correct or delete data;
* decide, before you use an AI feature, whether sending the data described in section 4 is lawful and appropriate,
  and if needed sign the provider's data processing terms.

## 4. Optional AI features (off by default)
These run only after you turn them on and enter your own API key. Each request goes **directly from your computer to
the provider**, never through the Author.

| Feature | Provider | What is sent | What is never sent |
|---|---|---|---|
| AI analysis | Google (Gemini API) | Text game data: sampled positions, tracker IDs, team names and colours, jersey numbers, statistics, segment tables | Video, player names, licence numbers, Replay title and your notes (replaced by placeholders) |
| AI court finder | Google (Gemini API) | **Still images from your video**, reduced in size, which can show players, including children | Names, rosters, statistics |
| Light-work labels | TypeSafe AI (Jev) | Segment statistics and team names | Video, images, player names, licence numbers |

The provider processes this data under **its own terms and privacy policy and your contract with it**, not this
policy. At the time of writing, Google's terms for **unpaid** (free-tier) Gemini API use allow Google to use what you
send to improve its products, which can include review by people. If you send images of children, use a paid
account with suitable data terms or do not use the AI court finder. Check the provider's current terms yourself; they
can change.

Answers are saved in your Replay folder. Images and data already answered are cached locally so that they are not
sent again.

## 5. Other network connections
* **Model downloads:** some detection models (for example RF-DETR weights) are downloaded from their publisher the
  first time they are used. This request contains no game data; like any download it reveals your IP address to the
  publisher's server.
* **Links:** links in the Software (for example "Get a key") open in your web browser.
* The Software makes no other connections. Live mode only reads local files.

## 6. Sharing files
Exported Replays (.aareplay), reports, playlists and clips can contain names, numbers, statistics and video. They go
only where you send them, and you are responsible for that disclosure.

## 7. Children
The Software is a tool for adult coaches and analysts and is not intended to be used by children. The data it handles
is often about children; see section 3 for your responsibilities.

## 8. Changes
If this policy changes, the Software shows the new version the next time it starts.

## 9. Contact
Questions about the Software: Onni Kauppinen · onni.kauppinen@gmail.com. Questions about the data in your Replays go to
the controller (section 3), not the Author. You can complain to your data protection authority; in Finland that is the
Data Protection Ombudsman (tietosuoja.fi).
