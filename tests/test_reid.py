import sys
from pathlib import Path

import numpy as np
import pytest

from replay import reid

ROOT = Path(__file__).resolve().parent.parent


def test_crop_box_shape_and_border():
    img = np.zeros((100, 200, 3), np.uint8)
    img[:, :100] = 200
    c = reid.crop_box(img, (90, 10, 110, 90))
    assert c.shape == (reid.CROP_H, reid.CROP_W, 3) and c.dtype == np.uint8
    assert c[:, :10].mean() > 150 and c[:, -10:].mean() < 50  # left half bright, right half dark
    edge = reid.crop_box(img, (-5, -5, 20, 50))  # partly outside: replicated border, no exception
    assert edge.shape == (reid.CROP_H, reid.CROP_W, 3)
    assert reid.crop_box(img, (10, 10, 10.5, 50)) is None
    crops, idx = reid.crop_boxes(img, np.array([[0, 0, 20, 40], [5, 5, 5, 5], [50, 20, 80, 90]]))
    assert crops.shape[0] == 2 and list(idx) == [0, 2]


def test_default_is_opt_in():
    assert reid.ReidConfig().enabled is False


def test_cache_roundtrip(tmp_path):
    emb = np.full((5, 8), np.nan, np.float16)
    emb[1] = 0.5
    reid.save_embeddings(tmp_path, reid.ReidEmbeddings(emb, {"boxes": 1}), key="abc")
    got = reid.load_embeddings(tmp_path)
    assert got is not None and got.stats["boxes"] == 1
    assert list(got.sampled) == [False, True, False, False, False]
    assert reid.load_embeddings(tmp_path / "missing") is None
    reid.save_embeddings(tmp_path, reid.ReidEmbeddings(emb, {"boxes": 2}), key="abd")  # file not left open
    assert reid.load_embeddings(tmp_path).stats["boxes"] == 2


def test_onnx_model_embeds(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("onnx")
    sys.path.insert(0, str(ROOT / "tools" / "reid"))
    from net import ReidNet, export_onnx

    torch.manual_seed(0)
    net = ReidNet(dim=16, widths=(8, 8, 8, 8, 8))
    path = tmp_path / "tiny.onnx"
    export_onnx(net, path, {"note": "test"})
    m = reid.ReidModel(path, device="cpu")
    assert m.dim == 16 and m.info.get("note") == "test"
    rng = np.random.default_rng(0)
    crops = rng.integers(0, 255, (3, reid.CROP_H, reid.CROP_W, 3), dtype=np.uint8)
    e = m.embed(crops)
    assert e.shape == (3, 16)
    assert np.allclose(np.linalg.norm(e, axis=1), 1.0, atol=1e-4)
    e2 = m.embed([crops[0], np.ascontiguousarray(crops[1][::2, ::2])])  # list input, other size -> resized
    assert e2.shape == (2, 16)
    assert np.allclose(e2[0], e[0], atol=1e-4)
    with torch.no_grad():
        ref = net.eval()(torch.from_numpy(crops.transpose(0, 3, 1, 2).astype(np.float32))).numpy()
    assert np.allclose(ref, e, atol=1e-3)
