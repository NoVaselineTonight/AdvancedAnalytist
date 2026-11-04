"""Hand-check helper for mined ball boxes (held-out venues).

  python tools/balldet_handcheck.py sheet <venue> [n=40]     -> data/balldet/mined/<venue>/handcheck_sheet.png
  python tools/balldet_handcheck.py save <venue> <ok idx,...> [--wrong idx,...]  -> merged into mined/handcheck.json

The sheet shows n random annotations (numbered tiles, box in green, 4x context). A human marks which tiles show
the ball exactly in the box; "save" stores them with verified=true (wrong ones with verified=false).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MINED = ROOT / "data" / "balldet" / "mined"


def _pick(venue: str, n: int) -> list[dict]:
    doc = json.loads((MINED / venue / "labels.json").read_text(encoding="utf-8"))
    ims = {im["id"]: im for im in doc["images"]}
    anns = doc["annotations"]
    if venue == "koristv":  # only the held-out clips (p4400, b3min, c300) get hand checks
        anns = [a for a in anns if ims[a["image_id"]].get("holdout")]
    rng = np.random.default_rng(1)
    sel = rng.choice(len(anns), min(n, len(anns)), replace=False) if anns else []
    return [{"ann": anns[int(j)], "image": ims[anns[int(j)]["image_id"]]} for j in sorted(sel)]


def sheet(venue: str, n: int) -> None:
    items = _pick(venue, n)
    tiles = []
    for t, it in enumerate(items):
        img = cv2.imread(str(MINED / venue / it["image"]["file_name"]))
        x, y, w, h = it["ann"]["bbox"]
        side = int(max(64.0, 3 * max(w, h)))
        im = img.copy()
        lw = max(1, int(round(side / 180)))  # ~1 px after the resize
        cv2.rectangle(im, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), lw)
        im = cv2.copyMakeBorder(im, side, side, side, side, cv2.BORDER_CONSTANT)  # square crop even at the edges
        x0, y0 = int(x + w / 2 - side / 2) + side, int(y + h / 2 - side / 2) + side
        crop = cv2.resize(im[y0:y0 + side, x0:x0 + side], (180, 180), interpolation=cv2.INTER_AREA)
        cv2.putText(crop, f"{t} {it['ann']['kind'][:4]}", (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        tiles.append(crop)
    while len(tiles) % 8:
        tiles.append(np.zeros((180, 180, 3), np.uint8))
    rows = [np.hstack(tiles[r:r + 8]) for r in range(0, len(tiles), 8)]
    cv2.imwrite(str(MINED / venue / "handcheck_sheet.png"), np.vstack(rows))
    print(len(items), "tiles ->", MINED / venue / "handcheck_sheet.png")


def save(venue: str, ok: list[int], wrong: list[int], n: int) -> None:
    items = _pick(venue, n)
    path = MINED / "handcheck.json"
    doc = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"format": "balldet-handcheck@1",
                                                                            "boxes": []}
    doc["boxes"] = [b for b in doc["boxes"] if b["venue"] != venue]
    for idx, flag in [(j, True) for j in ok] + [(j, False) for j in wrong]:
        it = items[idx]
        doc["boxes"].append({"venue": venue, "file_name": f"{venue}/{it['image']['file_name']}",
                             "prev_file_name": f"{venue}/{it['image']['prev_file_name']}", "clip": it["image"]["clip"],
                             "frame": it["image"]["frame"], "bbox": it["ann"]["bbox"], "kind": it["ann"]["kind"],
                             "verified": flag})
    path.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    print(venue, len(ok), "verified,", len(wrong), "wrong")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "sheet":
        sheet(a[1], int(a[2]) if len(a) > 2 else 40)
    else:
        wrong = []
        if "--wrong" in a:
            k = a.index("--wrong")
            wrong = [int(v) for v in a[k + 1].split(",") if v]
            del a[k:k + 2]
        save(a[1], [int(v) for v in a[2].split(",") if v], wrong, int(a[3]) if len(a) > 3 else 40)
