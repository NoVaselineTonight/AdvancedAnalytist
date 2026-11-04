"""Train ReidNet on the crop set of extract_crops.py (split 0 only) with same-moment contrastive batches.

Self-supervised pseudo-labels: a batch is made of W "moments"; each moment = up to P clean pieces that are all
present at the same time (so certainly different people - same-team players included, which are the hard
negatives), K crops per piece (same person, spread over the piece). The loss (supervised-contrastive / multi-
positive InfoNCE + batch-hard triplet) only uses pairs inside one moment: across moments the identity is unknown.

Usage: python tools/reid/train.py --data data/reid/fullgame --minutes 150 --out models/reid
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

from evaluate import chance, evaluate_embeddings, load_eval  # noqa: E402
from net import ReidNet, export_onnx  # noqa: E402


class Sampler:
    def __init__(self, meta: dict, train_idx: np.ndarray, rng: np.random.Generator):
        self.rng = rng
        piece = meta["piece"][train_idx]
        t = meta["t"][train_idx]
        self.members: dict[int, np.ndarray] = {}
        for p in np.unique(piece[piece >= 0]):
            self.members[int(p)] = train_idx[piece == p]
        self.pids = np.array(sorted(self.members))
        self.t0 = np.array([meta["t"][self.members[p]].min() for p in self.pids])
        self.t1 = np.array([meta["t"][self.members[p]].max() for p in self.pids])
        self.n = np.array([len(self.members[p]) for p in self.pids])
        self.times = t[piece >= 0]

    def moment(self, P: int, K: int) -> list[np.ndarray]:
        for _ in range(50):
            T = float(self.rng.choice(self.times))
            act = np.flatnonzero((self.t0 <= T + 0.5) & (self.t1 >= T - 0.5) & (self.n >= 2))
            if len(act) >= 3:
                break
        if len(act) > P:
            act = self.rng.choice(act, P, replace=False)
        out = []
        for a in act:
            m = self.members[int(self.pids[a])]
            out.append(self.rng.choice(m, K, replace=len(m) < K))
        return out


def augment(x: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
    """x (n, 3, H, W) float 0..255 -> jittered copy: scale/shift crop, brightness/contrast, blur, random erasing.
    No hue change (jersey colour is real information) and no horizontal flip (jersey numbers)."""
    n, _, H, W = x.shape
    out = torch.empty_like(x)
    for i in range(n):
        s = rng.uniform(0.85, 1.0)
        h, w = int(H * s), int(W * rng.uniform(0.85, 1.0))
        y0, x0 = rng.integers(0, H - h + 1), rng.integers(0, W - w + 1)
        c = x[i:i + 1, :, y0:y0 + h, x0:x0 + w]
        c = F.interpolate(c, size=(H, W), mode="bilinear", align_corners=False)[0]
        if rng.random() < 0.3:  # simulate low-res / motion blur
            f = rng.uniform(0.4, 0.7)
            c = F.interpolate(F.interpolate(c[None], scale_factor=f, mode="bilinear", align_corners=False),
                              size=(H, W), mode="bilinear", align_corners=False)[0]
        c = (c - 110) * rng.uniform(0.75, 1.25) + 110 + rng.uniform(-25, 25)
        if rng.random() < 0.4:  # occluder patch
            eh, ew = int(H * rng.uniform(0.15, 0.4)), int(W * rng.uniform(0.2, 0.6))
            ey, ex = rng.integers(0, H - eh + 1), rng.integers(0, W - ew + 1)
            c[:, ey:ey + eh, ex:ex + ew] = torch.from_numpy(rng.uniform(0, 255, (3, 1, 1)).astype(np.float32))
        out[i] = c.clamp(0, 255)
    return out


def loss_fn(z: torch.Tensor, label: torch.Tensor, group: torch.Tensor, temp: float = 0.07, margin: float = 0.3):
    S = z @ z.T
    n = len(z)
    eye = torch.eye(n, dtype=torch.bool)
    valid = (group[:, None] == group[None, :]) & ~eye
    pos = valid & (label[:, None] == label[None, :])
    neg = valid & (label[:, None] != label[None, :])
    # multi-positive InfoNCE within the moment
    logits = (S / temp).masked_fill(~valid, -1e9)
    logp = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    has = pos.any(1) & neg.any(1)
    supcon = -((logp * pos).sum(1)[has] / pos.sum(1)[has]).mean()
    # batch-hard triplet (cosine distance)
    d = 1 - S
    hp = d.masked_fill(~pos, -1).max(1).values
    hn = d.masked_fill(~neg, 9).min(1).values
    trip = F.relu(hp - hn + margin)[has].mean()
    acc = (hp < hn)[has].float().mean()
    return supcon + trip, float(supcon), float(trip), float(acc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data/reid/fullgame"))
    ap.add_argument("--out", default=str(ROOT / "models/reid"))
    ap.add_argument("--minutes", type=float, default=150)
    ap.add_argument("--moments", type=int, default=3)
    ap.add_argument("--widths", default="24,32,64,96,128")
    ap.add_argument("--P", type=int, default=8)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--max-iters", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt-every-min", type=float, default=5.0, help="save out/reid_ckpt.pt this often")
    ap.add_argument("--resume", action="store_true", help="continue from out/reid_ckpt.pt (same args)")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    data = Path(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = dict(np.load(data / "meta.npz"))
    crops = np.load(data / "crops.npy", mmap_mode="r")
    train_idx = np.flatnonzero((meta["split"] == 0) & meta["got"] & (meta["piece"] >= 0))
    print(f"loading {len(train_idx)} train crops", flush=True)
    store = {int(i): j for j, i in enumerate(train_idx)}
    train_crops = np.asarray(crops[train_idx])
    sampler = Sampler(meta, train_idx, rng)
    print(f"{len(sampler.pids)} train pieces", flush=True)
    emeta, eidx, ev, eteam, trials = load_eval(data)
    print(f"eval: {len(eidx)} crops, {len(trials)} gap trials (chance r1 {chance(trials)})", flush=True)

    widths = tuple(int(v) for v in args.widths.split(","))
    model = ReidNet(dim=args.dim, widths=widths)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    budget = args.minutes * 60
    t_start = time.perf_counter()
    log = []
    best = (-1.0, None)

    def run_eval(it: int):
        nonlocal best
        # embed through the exported ONNX graph on the GPU (DirectML): the CPU is busy training
        from replay.reid import ReidModel

        model.eval()
        export_onnx(model, out / "eval_tmp.onnx")
        model.train()
        E = ReidModel(out / "eval_tmp.onnx").embed(ev, batch=256)
        r = evaluate_embeddings(E, emeta, eidx, eteam, trials)
        score = (r["gap_r1"].get("all") or 0) + r.get("num_map", 0)
        rec = {"iter": it, "min": round((time.perf_counter() - t_start) / 60, 1), **r, "score": round(score, 4)}
        log.append(rec)
        print("EVAL", json.dumps(rec), flush=True)
        if score > best[0]:
            best = (score, it)
            torch.save(model.state_dict(), out / "reid_best.pt")
        (out / "train_log.json").write_text(json.dumps(log, indent=1), encoding="utf-8")

    it = 0
    ema = None
    done_s = 0.0  # training seconds of earlier (resumed) runs
    ckpt = out / "reid_ckpt.pt"
    if args.resume and ckpt.exists():
        st = torch.load(ckpt, weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        it, done_s, log, best = st["it"], st["elapsed_s"], st["log"], tuple(st["best"])
        rng.bit_generator.state = st["rng"]
        print(f"resumed from {ckpt} at iter {it} ({done_s / 60:.1f} min trained)", flush=True)
    else:
        run_eval(0)

    def save_ckpt():
        tmp = out / "reid_ckpt.tmp.pt"
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "it": it, "log": log, "best": list(best),
                    "elapsed_s": done_s + time.perf_counter() - t_run, "rng": rng.bit_generator.state,
                    "args": vars(args)}, tmp)
        tmp.replace(ckpt)

    t_run = time.perf_counter()
    last_ckpt = t_run
    while True:
        el = done_s + time.perf_counter() - t_run
        if el > budget or (args.max_iters and it >= args.max_iters):
            break
        frac = min(1.0, el / budget)
        lr = args.lr * (min(1.0, it / 300)) * (0.5 * (1 + np.cos(np.pi * frac)) * 0.97 + 0.03)
        for g in opt.param_groups:
            g["lr"] = lr
        idx, lab, grp = [], [], []
        for w in range(args.moments):
            for p, members in enumerate(sampler.moment(args.P, args.K)):
                idx.extend(store[int(i)] for i in members)
                lab.extend([w * 100 + p] * len(members))
                grp.extend([w] * len(members))
        x = torch.from_numpy(train_crops[idx].transpose(0, 3, 1, 2).astype(np.float32))
        x = augment(x, rng)
        z = model(x)
        loss, sc, tr, acc = loss_fn(z, torch.tensor(lab), torch.tensor(grp))
        opt.zero_grad()
        loss.backward()
        opt.step()
        it += 1
        ema = (sc, tr, acc) if ema is None else tuple(0.98 * a + 0.02 * b for a, b in zip(ema, (sc, tr, acc)))
        if time.perf_counter() - last_ckpt > args.ckpt_every_min * 60:
            save_ckpt()
            last_ckpt = time.perf_counter()
        if it % 20 == 0:
            print(f"it {it} {el / 60:.1f}min lr {lr:.2e} supcon {ema[0]:.3f} trip {ema[1]:.3f} acc {ema[2]:.3f} "
                  f"({el / it:.2f}s/it, batch {len(idx)})", flush=True)
        if it % args.eval_every == 0:
            run_eval(it)
            save_ckpt()
    run_eval(it)
    save_ckpt()
    model.load_state_dict(torch.load(out / "reid_best.pt"))
    model.eval()
    info = {"trained_on": str(data), "train_crops": int(len(train_idx)), "train_pieces": int(len(sampler.pids)),
            "iters": it, "best_iter": best[1], "minutes": round((done_s + time.perf_counter() - t_run) / 60, 1),
            "args": json.dumps(vars(args))}
    export_onnx(model, out / "reid.onnx", info)
    (out / "train_info.json").write_text(json.dumps({**info, "final": log[-1], "best": best}, indent=1),
                                         encoding="utf-8")
    print("DONE", json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
