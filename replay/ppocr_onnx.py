"""One-off converter: PaddleOCR PP-OCRv4 mobile text recognition (Paddle 3 "PIR" inference model) -> ONNX.

Why not paddle2onnx: the official converter has no wheel for Python 3.13 (this project's interpreter), pins
onnx<=1.17 (the project uses a newer onnx) and needs the full PaddlePaddle runtime (~105 MB) to read the PIR
``inference.json``. The model, however, is a small static graph of ~30 standard operators, stored as plain JSON
plus a ``save_combine`` parameter file, so it is translated here directly with the ``onnx`` package.

Source (Apache-2.0): https://huggingface.co/PaddlePaddle/PP-OCRv4_mobile_rec  (files inference.json,
inference.pdiparams, config.json). Usage::

    python -m replay.ppocr_onnx <folder with the downloaded files> models/pp-ocrv4-mobile-rec/pp-ocrv4-mobile-rec.onnx

Output: input ``x`` (N, 3, 48, W) float32, BGR, (px/255 - 0.5)/0.5; output ``probs`` (N, W/8, 6625) softmax over
[CTC blank] + character_dict + [space]. The character list is written next to the model (``charset.txt``).
Correctness is checked functionally (tests/test_jersey_numbers.py reads rendered text; see the README).
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np


# ----------------------------------------------------------------------------------------- parameter file
def _varint(b: bytes, i: int) -> tuple[int, int]:
    r = s = 0
    while True:
        c = b[i]
        i += 1
        r |= (c & 0x7F) << s
        s += 7
        if not c & 0x80:
            return r, i


def _tensor_desc(b: bytes) -> tuple[int, list[int]]:
    """Paddle VarType.TensorDesc protobuf: field 1 data_type, field 2 dims (packed or not)."""
    i, dt, dims = 0, -1, []
    while i < len(b):
        key, i = _varint(b, i)
        f, wt = key >> 3, key & 7
        if wt == 0:
            v, i = _varint(b, i)
            if f == 1:
                dt = v
            elif f == 2:
                dims.append(v - (1 << 64) if v >= 1 << 63 else v)
        elif wt == 2:
            n, i = _varint(b, i)
            sub, i = b[i:i + n], i + n
            if f == 2:
                j = 0
                while j < len(sub):
                    v, j = _varint(sub, j)
                    dims.append(v)
        else:
            raise ValueError(f"unexpected protobuf wire type {wt}")
    return dt, dims


def read_pdiparams(path: Path) -> list[np.ndarray]:
    """All tensors of a Paddle ``save_combine`` file, in file order (= parameters sorted by name)."""
    raw = Path(path).read_bytes()
    i, out = 0, []
    while i < len(raw):
        i += 4  # tensor version
        (lod,) = struct.unpack_from("<Q", raw, i)
        i += 8
        for _ in range(lod):
            (sz,) = struct.unpack_from("<Q", raw, i)
            i += 8 + sz
        i += 4  # dense tensor version
        (ds,) = struct.unpack_from("<i", raw, i)
        i += 4
        dt, dims = _tensor_desc(raw[i:i + ds])
        i += ds
        if dt != 5:  # FP32
            raise ValueError(f"unsupported parameter dtype {dt}")
        n = int(np.prod(dims)) if dims else 1
        out.append(np.frombuffer(raw, "<f4", n, i).reshape(dims).copy())
        i += 4 * n
    return out


# ----------------------------------------------------------------------------------------- graph
def _attrs(op: dict) -> dict:
    a = op.get("A", [])
    if not isinstance(a, list) or (a and not isinstance(a[0], dict)):
        return {}

    def val(v):
        d = v["D"]
        return [val(x) for x in d] if isinstance(d, list) and d and isinstance(d[0], dict) else d

    return {x["N"]: val(x["AT"]) for x in a}


def _outs(op: dict) -> list[int]:
    o = op.get("O", [])
    return [x["%"] for x in (o if isinstance(o, list) else [o])]


def convert(src: Path, dst: Path, opset: int = 17) -> Path:
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    src, dst = Path(src), Path(dst)
    prog = json.loads((src / "inference.json").read_text(encoding="utf-8"))
    ops = prog["program"]["regions"][0]["blocks"][0]["ops"]
    params = [(op["A"][3], op["O"]["TT"]["D"][1], _outs(op)[0]) for op in ops if op["#"] == "p"]
    tensors = read_pdiparams(src / "inference.pdiparams")
    by_name = sorted(params)
    if len(tensors) != len(by_name) or any(list(t.shape) != s for t, (_, s, _) in zip(tensors, by_name)):
        raise ValueError("parameter file does not match the program (shapes differ)")

    nodes, inits = [], []
    name = {}  # PIR value id -> ONNX tensor name
    const = {}  # PIR value id -> numpy constant (known at conversion time)
    lists = {}  # PIR value id -> list of value ids (builtin.combine)
    cnt = [0]

    def fresh(base: str) -> str:
        cnt[0] += 1
        return f"{base}_{cnt[0]}"

    def add_const(arr: np.ndarray, base: str = "c") -> str:
        n = fresh(base)
        inits.append(numpy_helper.from_array(np.asarray(arr), n))
        return n

    def node(op_type: str, inputs: list[str], n_out: int = 1, **kw) -> list[str]:
        outs = [fresh(op_type.lower()) for _ in range(n_out)]
        nodes.append(helper.make_node(op_type, inputs, outs, **kw))
        return outs

    for (pname, _, vid), arr in zip(by_name, tensors):
        name[vid] = pname
        inits.append(numpy_helper.from_array(arr, pname))

    graph_in = graph_out = None
    for op in ops:
        if op["#"] == "p":
            continue
        kind = op["#"].split(".", 1)[1]
        a = _attrs(op)
        ins = [x["%"] for x in op.get("I", [])]
        outs = _outs(op)
        X = lambda j: name[ins[j]]  # noqa: E731

        if kind == "data":
            graph_in = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["N", 3, 48, "W"])
            name[outs[0]] = "x"
        elif kind == "fetch":
            graph_out = X(0)
        elif kind == "full_int_array":
            v = np.array(a["value"], np.int64)
            const[outs[0]] = v
            name[outs[0]] = add_const(v, "ints")
        elif kind == "full":
            dt = {"float32": np.float32, "int64": np.int64, "int32": np.int32}[a["dtype"]]
            v = np.full(a["shape"], a["value"], dt)
            const[outs[0]] = v
            name[outs[0]] = add_const(v, "full")
        elif kind in ("conv2d", "depthwise_conv2d"):
            if a.get("padding_algorithm", "EXPLICIT") != "EXPLICIT" or a.get("data_format", "NCHW") != "NCHW":
                raise ValueError(f"unsupported conv attributes {a}")
            p = list(a["paddings"])
            pads = [p[0], p[1], p[0], p[1]] if len(p) == 2 else [p[0], p[2], p[1], p[3]]
            name[outs[0]] = node("Conv", [X(0), X(1)], strides=a["strides"], pads=pads, dilations=a["dilations"],
                                 group=int(a["groups"]))[0]
        elif kind == "batch_norm_":  # inputs: x, mean, variance, scale, bias
            name[outs[0]] = node("BatchNormalization", [X(0), X(3), X(4), X(1), X(2)],
                                 epsilon=float(a["epsilon"]))[0]
        elif kind in ("add", "multiply"):
            name[outs[0]] = node({"add": "Add", "multiply": "Mul"}[kind], [X(0), X(1)])[0]
        elif kind == "reshape":
            name[outs[0]] = node("Reshape", [X(0), X(1)])[0]
        elif kind == "hardswish":
            name[outs[0]] = node("HardSwish", [X(0)])[0]
        elif kind == "hardsigmoid":
            name[outs[0]] = node("HardSigmoid", [X(0)], alpha=float(a["slope"]), beta=float(a["offset"]))[0]
        elif kind == "relu":
            name[outs[0]] = node("Relu", [X(0)])[0]
        elif kind == "swish":
            s = node("Sigmoid", [X(0)])[0]
            name[outs[0]] = node("Mul", [X(0), s])[0]
        elif kind == "pool2d":
            ksize = [int(v) for v in const[ins[1]]]
            if a["pooling_type"] != "avg" or a.get("global_pooling"):
                raise ValueError(f"unsupported pool {a}")
            if a["adaptive"]:
                if ksize != [1, 1]:
                    raise ValueError(f"unsupported adaptive pool size {ksize}")
                name[outs[0]] = node("GlobalAveragePool", [X(0)])[0]
            else:
                p = list(a["paddings"])
                pads = [p[0], p[1], p[0], p[1]] if len(p) == 2 else [p[0], p[2], p[1], p[3]]
                name[outs[0]] = node("AveragePool", [X(0)], kernel_shape=ksize, strides=a["strides"], pads=pads,
                                     ceil_mode=int(a["ceil_mode"]), count_include_pad=0 if a["exclusive"] else 1)[0]
        elif kind in ("assign", "dropout"):  # inference: identity (dropout mode upscale_in_train)
            if kind == "dropout" and not a.get("is_test", True):
                raise ValueError("dropout not in test mode")
            name[outs[0]] = node("Identity", [X(0)])[0]
        elif kind == "shape64":
            name[outs[0]] = node("Shape", [X(0)])[0]
        elif kind == "slice":
            axes = add_const(np.array(a["axes"], np.int64))
            y = node("Slice", [X(0), X(1), X(2), axes])[0]
            dec = list(a.get("decrease_axis", []))
            name[outs[0]] = node("Squeeze", [y, add_const(np.array(dec, np.int64))])[0] if dec else y
        elif kind == "flatten":
            s0, s1 = int(a["start_axis"]), int(a["stop_axis"])
            if (s0, s1) != (2, 3):
                raise ValueError(f"unsupported flatten {s0}..{s1}")
            name[outs[0]] = node("Reshape", [X(0), add_const(np.array([0, 0, -1], np.int64))])[0]
        elif kind == "transpose":
            name[outs[0]] = node("Transpose", [X(0)], perm=[int(v) for v in a["perm"]])[0]
        elif kind == "layer_norm":
            name[outs[0]] = node("LayerNormalization", [X(0), X(1), X(2)], axis=int(a["begin_norm_axis"]),
                                 epsilon=float(a["epsilon"]))[0]
        elif kind == "matmul":
            xa, xb = X(0), X(1)
            if a.get("transpose_x") or a.get("transpose_y"):
                raise ValueError("transposed matmul not supported")
            name[outs[0]] = node("MatMul", [xa, xb])[0]
        elif kind == "scale":  # out = x * scale + bias (bias_after_scale)
            if not a.get("bias_after_scale", True):
                raise ValueError("scale with bias before scale not supported")
            y = node("Mul", [X(0), X(1)])[0]
            b = float(a.get("bias", 0.0))
            name[outs[0]] = node("Add", [y, add_const(np.array(b, np.float32))])[0] if b else y
        elif kind == "softmax":
            name[outs[0]] = node("Softmax", [X(0)], axis=int(a["axis"]))[0]
        elif kind == "combine":
            lists[outs[0]] = ins
        elif kind == "stack":
            if int(a["axis"]) != 0:
                raise ValueError("stack axis != 0")
            ax = add_const(np.array([0], np.int64))
            parts = [node("Unsqueeze", [name[v], ax])[0] for v in lists[ins[0]]]
            name[outs[0]] = node("Concat", parts, axis=0)[0]
        elif kind == "concat":
            axis = int(np.asarray(const[ins[1]]).reshape(-1)[0])
            name[outs[0]] = node("Concat", [name[v] for v in lists[ins[0]]], axis=axis)[0]
        elif kind == "squeeze":
            name[outs[0]] = node("Squeeze", [X(0), add_const(np.asarray(const[ins[1]], np.int64).reshape(-1))])[0]
        else:
            raise ValueError(f"unsupported operator {op['#']}")

    if graph_in is None or graph_out is None:
        raise ValueError("program has no data/fetch op")
    nodes.append(helper.make_node("Identity", [graph_out], ["probs"]))
    used = {i for n in nodes for i in n.input}
    inits = [t for t in inits if t.name in used]  # constants only needed at conversion time (pool sizes, axes)
    out_info = helper.make_tensor_value_info("probs", TensorProto.FLOAT, ["N", "T", 6625])
    g = helper.make_graph(nodes, "pp_ocrv4_mobile_rec", [graph_in], [out_info], inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", opset)], producer_name="replay.ppocr_onnx")
    m.ir_version = 8
    m.doc_string = ("PP-OCRv4_mobile_rec (PaddleOCR, Apache-2.0) converted from "
                    "https://huggingface.co/PaddlePaddle/PP-OCRv4_mobile_rec by replay/ppocr_onnx.py")
    onnx.checker.check_model(m)
    dst.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(m, str(dst))
    cfg = json.loads((src / "config.json").read_text(encoding="utf-8"))
    chars = cfg["PostProcess"]["character_dict"]
    (dst.parent / "charset.txt").write_text("\n".join(chars) + "\n", encoding="utf-8")
    return dst


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    print("written:", convert(Path(sys.argv[1]), Path(sys.argv[2])))
