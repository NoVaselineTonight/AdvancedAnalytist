"""Export the CLIP ViT-B/32 (LAION-2B, MIT) vision tower to ONNX and precompute role-prompt text embeddings.

    python tools/clip/export_clip.py      (writes models/clip_b32_laion/vision.onnx, role_text.npy, roles.json)

Only the safetensors weights are loaded (use_safetensors=True, no pickle). The app then needs neither torch nor a
tokenizer: replay/clip_roles.py runs vision.onnx (DirectML when available) and compares with role_text.npy.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import CLIPModel, CLIPTokenizer

ROOT = Path(__file__).resolve().parents[2]
MD = ROOT / "models" / "clip_b32_laion"
ROLES = {
    "referee": ["a basketball referee in a black and white striped shirt", "a referee with a whistle"],
    "player": ["a basketball player in a team jersey", "a basketball player in a sports uniform"],
    "coach": ["a coach in street clothes", "a man in a suit on the sideline"],
    "spectator": ["a spectator", "a fan in the crowd"],
    "bench": ["a person sitting on a bench", "a substitute sitting on the bench in a warm-up shirt"],
}


class Vision(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.v, self.p = m.vision_model, m.visual_projection

    def forward(self, x):
        e = self.p(self.v(pixel_values=x).pooler_output)
        return e / e.norm(dim=-1, keepdim=True)


def main():
    m = CLIPModel.from_pretrained(MD, use_safetensors=True).eval()
    tok = CLIPTokenizer.from_pretrained(MD)
    with torch.no_grad():
        names, embs = [], []
        for role, prompts in ROLES.items():
            t = tok(prompts, padding=True, return_tensors="pt")
            e = m.text_projection(m.text_model(**t).pooler_output)
            e = e / e.norm(dim=-1, keepdim=True)
            e = e.mean(0)
            embs.append((e / e.norm()).numpy())
            names.append(role)
        np.save(MD / "role_text.npy", np.stack(embs).astype(np.float32))
        (MD / "roles.json").write_text(json.dumps({"roles": names, "prompts": ROLES,
                                                   "logit_scale": float(m.logit_scale.exp())}, indent=1))
        vis = Vision(m).eval()
        x = torch.randn(2, 3, 224, 224)
        ref = vis(x).numpy()
        torch.onnx.export(vis, x, MD / "vision.onnx", input_names=["pixel_values"], output_names=["embedding"],
                          dynamic_axes={"pixel_values": {0: "n"}, "embedding": {0: "n"}}, opset_version=17,
                          dynamo=False)
    import onnxruntime as ort
    s = ort.InferenceSession(str(MD / "vision.onnx"), providers=["CPUExecutionProvider"])
    out = s.run(None, {"pixel_values": x.numpy()})[0]
    print("onnx vs torch max abs diff", float(np.abs(out - ref).max()), "cos", float((out * ref).sum(1).min()))


if __name__ == "__main__":
    sys.exit(main())
