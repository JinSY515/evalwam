#!/usr/bin/env python3
"""Uniform (or weighted) average of full-model checkpoints of one run ("model soup" / post-hoc SWA).
  ckpt_soup.py --out <dir> --ckpts <dirA> <dirB> ... [--weights 0.5 0.5]
Writes <out>/model.safetensors (fp32), global_step.txt = 'soup', soup.json (sources, weights), .DL_OK."""
import argparse, json, os, torch
from safetensors import safe_open
from safetensors.torch import save_file

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True); ap.add_argument("--ckpts", nargs="+", required=True)
ap.add_argument("--weights", type=float, nargs="*")
a = ap.parse_args()
w = a.weights or [1.0 / len(a.ckpts)] * len(a.ckpts)
assert len(w) == len(a.ckpts) and abs(sum(w) - 1) < 1e-6
fs = [safe_open(os.path.join(c, "model.safetensors"), "pt") for c in a.ckpts]
keys = list(fs[0].keys())
for f in fs[1:]:
    assert set(f.keys()) == set(keys), "key mismatch"
out = {}
for k in keys:
    acc = None
    for f, wi in zip(fs, w):
        t = f.get_tensor(k).double() * wi
        acc = t if acc is None else acc + t
    out[k] = acc.to(fs[0].get_slice(k).get_dtype() == "F32" and torch.float32 or torch.bfloat16).contiguous()
os.makedirs(a.out, exist_ok=True)
save_file(out, os.path.join(a.out, "model.safetensors"), metadata={"format": "pt"})
open(os.path.join(a.out, "global_step.txt"), "w").write("soup")
json.dump({"ckpts": a.ckpts, "weights": w}, open(os.path.join(a.out, "soup.json"), "w"), indent=1)
open(os.path.join(a.out, ".DL_OK"), "w").close()
print(f"{len(keys)} tensors -> {a.out}")
