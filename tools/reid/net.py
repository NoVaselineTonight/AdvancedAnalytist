"""ReidNet: a small residual CNN for 128x64 person crops (trained from scratch on CPU). Licence: project code."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

MEAN, STD = 110.0, 64.0  # BGR 0..255 -> roughly zero-mean unit-variance (inside the graph, so ONNX takes raw pixels)


class Block(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.skip = None
        if stride != 1 or cin != cout:
            self.skip = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)), inplace=True)
        y = self.b2(self.c2(y))
        return F.relu(y + (x if self.skip is None else self.skip(x)), inplace=True)


class ReidNet(nn.Module):
    """(n, 3, 128, 64) BGR 0..255 -> (n, dim) L2-normalised embedding.

    Global average pooling plus three horizontal stripes (head/torso, shorts, legs/shoes) so body-part cues
    (number on the torso, socks, shoes) keep their own channels."""

    def __init__(self, dim: int = 128, widths=(32, 48, 96, 128, 192)):
        super().__init__()
        w0, w1, w2, w3, w4 = widths
        self.stem = nn.Sequential(nn.Conv2d(3, w0, 3, 2, 1, bias=False), nn.BatchNorm2d(w0), nn.ReLU(inplace=True))
        self.layers = nn.Sequential(Block(w0, w1, 1), Block(w1, w2, 2), Block(w2, w3, 2), Block(w3, w4, 2),
                                    Block(w4, w4, 1))
        self.n_parts = 3
        feat = w4 * (1 + self.n_parts)
        self.neck = nn.BatchNorm1d(feat)
        self.fc = nn.Linear(feat, dim)
        self.dim = dim

    def forward(self, x):
        x = (x - MEAN) / STD
        f = self.layers(self.stem(x))  # (n, C, 8, 4) for 128x64 input
        g = f.mean(dim=(2, 3))
        h = int(f.shape[2])  # 8 for 128-px crops: rows 0-2 head/torso, 3-4 shorts, 5-7 legs/shoes
        cut = [0, max(1, round(h * 0.375)), max(2, round(h * 0.625)), h]
        parts = torch.cat([f[:, :, cut[i]:cut[i + 1]].mean(dim=(2, 3)) for i in range(self.n_parts)], 1)
        z = self.fc(self.neck(torch.cat([g, parts], 1)))
        return F.normalize(z, dim=1)


def export_onnx(model: ReidNet, path, meta: dict | None = None, h: int = 128, w: int = 64) -> None:
    import onnx

    model = model.eval()
    dummy = torch.zeros(2, 3, h, w)
    torch.onnx.export(model, dummy, str(path), input_names=["crops"], output_names=["embedding"],
                      dynamic_axes={"crops": {0: "n"}, "embedding": {0: "n"}}, opset_version=17, dynamo=False)
    if meta:
        m = onnx.load(str(path))
        for k, v in meta.items():
            p = m.metadata_props.add()
            p.key, p.value = str(k), str(v)
        onnx.save(m, str(path))
