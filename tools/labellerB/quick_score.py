"""Quick-score clips against their labels on the EXISTING output (no pass-2 rebuild): python tools/labellerB/quick_score.py clip [clip...]"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from replay.benchmark import clip_registry, evaluate_clip
KEYS = ["label_frames", "labelled_players", "player_recall", "player_recall_visible", "phantoms_per_frame", "bench_shown_per_frame",
        "dups_shown_per_frame", "referee_recall", "referee_precision", "team_acc", "team_acc_perm", "team_n", "id_switches",
        "id_fragmentation", "id_transfers", "id_duration_s", "ids_per_min", "position_error_m", "frames_without_output"]
reg = clip_registry()
for c in sys.argv[1:]:
    info = reg[c]
    r = evaluate_clip(c, info, Path(info["src"]).parent)
    print(c, {k: (round(r[k], 3) if isinstance(r.get(k), float) else r.get(k)) for k in KEYS if k in r})
