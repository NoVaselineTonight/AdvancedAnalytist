"""Labeller B: turn a label_tool draft + my per-frame codes into data/benchmark/labels/labellerB/players_<clip>.json.

Codes file (JSON): {"clip": "...", "notes": "...", "frames": {"<frame>": {"<ref>": "<code>", "+1": "<code>@x1,y1,x2,y2"}, ...},
                    "skip": {"<frame>": "reason"}}
<code> = kind[:team[:pid]] with kind p=player r=referee b=bench s=spectator d=duplicate i=ignore, optional "/occ" suffix
(occluded). Every output box (O<n>) MUST be coded; detections (D<n>) not coded are dropped; "+n" adds a hand-drawn box.
Frames listed in "skip" are not written (e.g. no court view). Frames merge by number with an existing labellerB file.
Usage: python tools/labellerB/apply_codes.py <codes.json>
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LAB = ROOT / "data/benchmark/labels"
KIND = {"p": "player", "r": "referee", "b": "bench", "s": "spectator", "d": "duplicate", "i": "ignore"}


def parse(code: str) -> dict:
    occ = code.endswith("/occ")
    code = code[:-4] if occ else code
    parts = code.split(":")
    kind = KIND[parts[0]]
    team = parts[1] if len(parts) > 1 and parts[1] else None
    pid = parts[2] if len(parts) > 2 and parts[2] else None
    p = {"kind": kind, "team": team if kind == "player" else None, "pid": pid, "occluded": occ,
         "on_court": kind in ("player", "referee")}
    if kind in ("bench", "spectator") and team:
        p["kit"] = team
    return p


def main():
    codes = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    clip = codes["clip"]
    draft = json.loads((LAB / "drafts" / f"players_{clip}.json").read_text(encoding="utf-8"))
    byf = {f["frame"]: f for f in draft["frames"]}
    out_p = LAB / "labellerB" / f"players_{clip}.json"
    old = json.loads(out_p.read_text(encoding="utf-8")) if out_p.exists() else {"frames": []}
    frames = {f["frame"]: f for f in old["frames"]}
    for fs, fc in codes["frames"].items():
        f = byf[int(fs)]
        refs = {p["ref"]: p for p in f["people"]}
        refs.update({c["ref"]: c for c in f.get("candidates", [])})
        missing = [r for r in refs if r.startswith("O") and r not in fc]
        if missing:
            raise SystemExit(f"{clip} frame {fs}: output boxes not coded: {missing}")
        people = []
        for ref, code in fc.items():
            if ref.startswith("_"):
                continue
            if ref.startswith("+"):
                code, box = code.split("@")
                p = parse(code)
                p["box"] = [float(v) for v in box.split(",")]
                p["src"] = "hand"
            else:
                if ref not in refs:
                    raise SystemExit(f"{clip} frame {fs}: unknown ref {ref}")
                p = parse(code)
                p["box"] = refs[ref]["box"]
                p["src"] = ref
            people.append(p)
        frames[int(fs)] = {"frame": int(fs), "t": f.get("t"), "status": "done", "complete": True,
                           "segment": fc.get("_seg") if isinstance(fc.get("_seg"), str) else None, "people": people}
    for fs in codes.get("skip", {}):
        frames.pop(int(fs), None)
    doc = {"format": "aa-bench-players@1", "clip": clip, "labeller": "labeller B (Claude, manual frame-by-frame)",
           "notes": codes.get("notes", ""), "skipped": codes.get("skip", {}),
           "frames": [frames[k] for k in sorted(frames)]}
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    print(out_p, len(doc["frames"]), "frames", sum(len(f["people"]) for f in doc["frames"]), "people")


if __name__ == "__main__":
    main()
