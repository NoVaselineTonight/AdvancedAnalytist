#!/bin/bash
# Labeller B: analyse the external .mp4 segments (westwood_girls + TrackID3x3) AFTER the lead's process.sh is ALL (one GPU job at a time).
# Idempotent: skips segments listed in data/external/progress_mp4.txt.
cd /c/Users/onnik/Desktop/AdvancedAnalytist
until grep -q ALL data/external/progress.txt 2>/dev/null; do sleep 30; done
for f in data/external/trackid3x3/seg_*.mp4 data/external/westwood_girls/seg_*.mp4; do
  d="${f%.mp4}"; grep -qx "done $d" data/external/progress_mp4.txt 2>/dev/null && continue
  mkdir -p "$d"
  PYTHONUTF8=1 .venv/Scripts/python -m replay process --video "$f" --out "$d/out" --calibration "$d/calibration.json" > "$d/process.log" 2>&1
  echo "done $d rc=$?" | sed 's/ rc=0$//' >> data/external/progress_mp4.txt
done
echo ALL >> data/external/progress_mp4.txt
