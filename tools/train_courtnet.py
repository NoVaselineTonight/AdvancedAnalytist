"""Train the court detector (replay/courtnet.py) on synthetic courts + real labelled frames, CPU, for a time budget.

usage: python tools/train_courtnet.py [--hours 3] [--workers 5] [--threads 6] [--batch 16] [--out data/courtnet]
                                      [--real data/courtnet/real_v2.npz] [--p-real 0.25] [--p-tele 0.0]
                                      [--exclude-venues ncaa,westwood_girls] [--resume | --init CKPT.pt]
                                      [--save-min 20] [--tag realft]
Writes <out>/train.log (loss), <out>/courtnet.pt (checkpoint, every --save-min min) and <out>/courtnet.onnx (export);
with --tag also a snapshot <out>/courtnet_<tag>_<step>.onnx per save (per-checkpoint evaluation, tools/court_eval.py).
Real frames: data/courtnet/real_v2.npz (tools/make_courtnet_real.py: H per frame + venue tags; venue-balanced
sampling, flip / crop-zoom / photometric augmentation) or the old real.npz (C, R, f per frame).
Own code; PyTorch (BSD-3). No external data.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import torch  # noqa: E402


class Synth(torch.utils.data.IterableDataset):
    """Endless synthetic + real samples (module level: DataLoader workers are spawned processes on Windows)."""

    def __init__(self, real: str | None = None, p_real: float = 0.25, p_tele: float = 0.0,
                 exclude_venues: tuple[str, ...] = (), aug: bool = True):
        self.real, self.p_real, self.p_tele = real, p_real, p_tele
        self.exclude, self.aug = tuple(exclude_venues), aug

    def _real_sample(self, R_, rng):
        """Old real.npz format: camera (C, R, f) per frame."""
        from replay import court, court_synth

        k = int(rng.integers(len(R_["img"])))
        L, W, spec = R_["court"][k]
        court.configure(float(L), float(W), str(spec))
        fx, fy = R_["f"][k]
        Kp = np.array([[fx, 0, court_synth.IN_W / 2], [0, fy, court_synth.IN_H / 2], [0, 0, 1.0]])
        C, R = R_["C"][k], R_["R"][k]
        H = Kp @ np.column_stack([R[:, 0], R[:, 1], -R @ C])
        line, heat = court_synth.targets(H, C, R, list(court.KEYPOINTS))
        img = np.clip(R_["img"][k].astype(np.float32) * rng.uniform(0.7, 1.3) + rng.uniform(-25, 25), 0, 255)
        return img.astype(np.uint8), line, heat

    def _real_sample_v2(self, R_, rng):
        """real_v2.npz: H per frame (court -> 320x192, depth row kept); venue-balanced; flip + crop-zoom + photometric
        augmentation with the label re-projected (H' = S @ H @ M)."""
        import cv2

        from replay import court, court_synth

        W_, H_ = court_synth.IN_W, court_synth.IN_H
        k = int(rng.choice(len(R_["img"]), p=R_["_w"]))
        L, W, spec = R_["court"][k]
        court.configure(float(L), float(W), str(spec))
        H = np.asarray(R_["H"][k], float).copy()
        img = R_["img"][k]
        if self.aug and rng.random() < 0.5:  # horizontal flip = the mirrored court (markings are symmetric)
            img = img[:, ::-1]
            F = np.array([[-1.0, 0, W_ - 1], [0, 1, 0], [0, 0, 1]])
            Mx = np.array([[-1.0, 0, float(L)], [0, 1, 0], [0, 0, 1]])
            H = F @ H @ Mx
        if self.aug and rng.random() < 0.5:  # crop-zoom 1-1.7x (telephoto-like views, keypoints leaving the frame)
            z = rng.uniform(1.0, 1.7)
            cw, ch = W_ / z, H_ / z
            x0, y0 = rng.uniform(0, W_ - cw), rng.uniform(0, H_ - ch)
            S = np.array([[z, 0, -z * x0], [0, z, -z * y0], [0, 0, 1.0]])
            img = cv2.warpAffine(np.ascontiguousarray(img), S[:2], (W_, H_), flags=cv2.INTER_LINEAR)
            H = S @ H
        line, heat = court_synth.targets(H, None, None, list(court.KEYPOINTS))
        img = img.astype(np.float32)
        if self.aug:
            img = img * rng.uniform(0.65, 1.3) + rng.uniform(-25, 25)
            img *= rng.uniform(0.9, 1.1, 3)
            if rng.random() < 0.25:
                kk = int(rng.integers(1, 4)) * 2 + 1
                img = cv2.GaussianBlur(img, (kk, kk), 0)
            img += rng.normal(0, rng.uniform(0, 5), img.shape)
        img = np.clip(img, 0, 255).astype(np.uint8)
        if self.aug and rng.random() < 0.3:
            ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(30, 80))])
            img = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        return img, line, heat

    def _load_real(self):
        if not self.real or not Path(self.real).exists():
            return None
        z = np.load(self.real, allow_pickle=True)  # own file (make_courtnet_real.py)
        R_ = {k: z[k] for k in z.files}
        if "H" not in R_:
            return R_
        keep = np.array([v not in self.exclude for v in R_["venue"]])
        if not keep.any():
            return None
        R_ = {k: v[keep] for k, v in R_.items()}
        # venue-balanced sampling (a venue with many clips must not dominate): weight 1 / sqrt(n_venue)
        ven, cnt = np.unique(R_["venue"], return_counts=True)
        n_of = dict(zip(ven, cnt))
        w = np.array([1.0 / np.sqrt(n_of[v]) for v in R_["venue"]])
        R_["_w"] = w / w.sum()
        return R_

    def __iter__(self):
        import cv2

        from replay import court_synth

        cv2.setNumThreads(1)
        wi = torch.utils.data.get_worker_info()
        rng = np.random.default_rng((wi.id if wi else 0) * 1000003 + int(time.time() * 1000) % 1000003)
        R_ = self._load_real()
        while True:
            if R_ is not None and rng.random() < self.p_real:
                img, line, heat = self._real_sample_v2(R_, rng) if "H" in R_ else self._real_sample(R_, rng)
            else:
                img, line, heat, _m = court_synth.render(rng, p_tele=self.p_tele)
            x = img[:, :, ::-1].astype(np.float32).transpose(2, 0, 1) / 255.0
            y = np.concatenate([line[None], heat], 0)
            yield torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(y)


def loss_fn(logits, y):
    import torch
    import torch.nn.functional as F

    line_l = F.binary_cross_entropy_with_logits(logits[:, :1], y[:, :1], pos_weight=torch.tensor(8.0))
    p = torch.sigmoid(logits[:, 1:])
    t = y[:, 1:]
    kp_l = ((p - t) ** 2 * (1 + 30 * t)).mean() * 10
    return line_l + kp_l, float(line_l), float(kp_l)


def export(model, path: Path, n_kp: int) -> None:
    import torch

    from replay.courtnet import IN_H, IN_W

    model.eval()
    torch.onnx.export(model, torch.zeros(1, 3, IN_H, IN_W), str(path), input_names=["image"], output_names=["out"],
                      opset_version=17, dynamic_axes={"image": {0: "n"}, "out": {0: "n"}}, dynamo=False)
    model.train()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=3.0)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "data" / "courtnet"))
    ap.add_argument("--resume", action="store_true", help="continue from <out>/courtnet.pt (model + optimizer + step)")
    ap.add_argument("--init", default=None, help="start from this checkpoint's weights (fresh optimizer; step kept)")
    ap.add_argument("--p-tele", type=float, default=0.0, help="share of TV main-camera (high, telephoto) samples")
    ap.add_argument("--p-real", type=float, default=0.25, help="share of real frames")
    ap.add_argument("--real", default=None, help="real frames npz (default <out>/real_v2.npz, else <out>/real.npz)")
    ap.add_argument("--exclude-venues", default="", help="comma-separated venue tags held out (real_v2 only)")
    ap.add_argument("--no-aug", action="store_true", help="no flip / crop-zoom / photometric augmentation of real frames")
    ap.add_argument("--save-min", type=float, default=20.0, help="checkpoint + ONNX export every this many minutes")
    ap.add_argument("--tag", default=None, help="also keep a snapshot <out>/courtnet_<tag>_<step>.onnx per save")
    a = ap.parse_args()
    import torch

    from replay import court_synth
    from replay.courtnet import build_model

    torch.set_num_threads(a.threads)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    names = court_synth.kp_names()
    model = build_model(len(names))
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    step0 = 0
    if a.resume and (out / "courtnet.pt").exists():
        ck = torch.load(out / "courtnet.pt", map_location="cpu", weights_only=False)  # own checkpoint
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step0 = ck["step"]
    elif a.init:
        ck = torch.load(a.init, map_location="cpu", weights_only=False)  # own checkpoint
        model.load_state_dict(ck["model"])
        step0 = ck.get("step", 0)
    real = a.real or (str(out / "real_v2.npz") if (out / "real_v2.npz").exists() else str(out / "real.npz"))
    excl = tuple(v for v in a.exclude_venues.split(",") if v)
    ds = Synth(real, a.p_real, a.p_tele, excl, not a.no_aug)
    dl = torch.utils.data.DataLoader(ds, batch_size=a.batch, num_workers=a.workers, persistent_workers=a.workers > 0,
                                     prefetch_factor=4 if a.workers > 0 else None)
    log = open(out / "train.log", "a", encoding="utf-8")
    R_ = ds._load_real()
    n_real = 0 if R_ is None else len(R_["img"])
    venues = {} if R_ is None or "venue" not in R_ else dict(zip(*np.unique(R_["venue"], return_counts=True)))
    print(f"{time.strftime('%d.%m %H:%M:%S')} params {n_par / 1e6:.2f} M, keypoints {len(names)}, budget {a.hours} h, "
          f"start step {step0}, lr {a.lr}, threads {a.threads}, workers {a.workers}, p_real {a.p_real} p_tele {a.p_tele}, "
          f"real {real}: {n_real} frames {({k: int(v) for k, v in venues.items()})}, excluded {excl}, aug {not a.no_aug}",
          file=log, flush=True)
    t0 = time.time()
    t_save = t0
    budget = a.hours * 3600
    step = step0
    run = []
    for x, y in dl:
        frac = min(1.0, (time.time() - t0) / budget)
        for g in opt.param_groups:
            g["lr"] = a.lr * (0.5 * (1 + np.cos(np.pi * frac)) * 0.95 + 0.05)
        logits = model(x)
        loss, ll, kl = loss_fn(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        step += 1
        run.append((float(loss), ll, kl))
        if step % 50 == 0:
            m = np.mean(run, 0)
            run = []
            print(f"{time.strftime('%H:%M:%S')} step {step} loss {m[0]:.4f} line {m[1]:.4f} kp {m[2]:.4f} "
                  f"lr {opt.param_groups[0]['lr']:.2e} {(time.time() - t0) / 60:.1f} min", file=log, flush=True)
        done = time.time() - t0 >= budget
        if time.time() - t_save > a.save_min * 60 or done:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "names": names},
                       out / "courtnet.pt")
            export(model, out / "courtnet.onnx", len(names))
            if a.tag:
                shutil.copyfile(out / "courtnet.onnx", out / f"courtnet_{a.tag}_{step}.onnx")
            t_save = time.time()
            print(f"saved at step {step}", file=log, flush=True)
        if done:
            break
    log.close()


if __name__ == "__main__":
    main()
