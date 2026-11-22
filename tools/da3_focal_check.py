#!/usr/bin/env python3
"""DA3 rotation-magnitude bias check on GT valbench videos: DA3's estimated focal vs the GT camera (fx=256 @512),
and the net yaw DA3 recovers with (a) its own intrinsics (as in the suite) vs (b) the GT intrinsics passed in.
Same call as eval_camera_control.estimate_da3 (DA3-LARGE-1.1, ref_view_strategy=first). Args: out_json n_samples"""
import json, os, sys, glob
import numpy as np, torch
from PIL import Image
from depth_anything_3.api import DepthAnything3
R = "/root/project/egowam/results_syn/goalprog_mode_real6_nomr_hg0_grad0.5/ckpta7500_8750/valbench5s_all120"
rows = list(__import__("csv").DictReader(open("/root/project/egowam/results_syn/coherence_validation/syn_valbench120/perturb_rows.csv")))
B = [r for r in rows if r["family"] == "rot_drift" and float(r["mag"]) == 0]
B.sort(key=lambda r: -abs(float(r["gt_net_yaw"])))
N = int(sys.argv[2]); pick = B[1:N - 4] + B[-4:]          # skip the 192 deg U-turn outlier; + 4 near-straight
meta = json.load(open(f"{R}/samples_meta.json"))
m = DepthAnything3.from_pretrained(os.environ.get("DA3_MODEL", "depth-anything/DA3-LARGE-1.1")).to("cuda").eval()
def yaw_net(c2w):
    f = c2w[:, :3, 2]; y = np.unwrap(np.arctan2(f[:, 0], f[:, 2]))   # rotation about camera-frame vertical, frame-0 coords
    return float(np.degrees(y[-1] - y[0]))
def rel(c2w): return np.linalg.inv(c2w[0]) @ c2w
out = []
for r in pick:
    k = str(int(r["sample"])); info = meta[k]; s0 = int(info["start_frame"])
    fr = [np.asarray(Image.open(f"{info['seq_dir']}/images/{s0 + t:06d}.png").convert("RGB")) for t in range(77)]
    K = np.load(f"{info['seq_dir']}/intrinsics.npy").ravel()[:4]; K3 = np.array([[K[0], 0, K[2]], [0, K[1], K[3]], [0, 0, 1.0]])
    res = {}
    for tag, kw in (("own", {}), ("gtK", {"intrinsics": np.repeat(K3[None], 77, 0)})):
        with torch.no_grad():
            p = m.inference(fr, ref_view_strategy="first", **kw)
        ext = np.asarray(p.extrinsics, np.float64); c2w = np.stack([np.linalg.inv(np.vstack([e, [0, 0, 0, 1]])) for e in ext])
        Ki = np.asarray(p.intrinsics, np.float64)
        sc = 512.0 / max(np.asarray(p.processed_images).shape[1:3]) if hasattr(p, "processed_images") else float("nan")
        res[tag] = {"net_yaw": yaw_net(rel(c2w)), "fx_proc": float(Ki[:, 0, 0].mean()), "w_proc": int(np.asarray(p.processed_images).shape[2]) if hasattr(p, "processed_images") else None}
    # GT net yaw in the camera frame (CARLA yaw is about the vertical, same sign convention handled by abs ratio)
    out.append({"sample": int(k), "gt_net_yaw": float(r["gt_net_yaw"]), "suite_da3_net_yaw": float(r["da3_net_yaw"]), **{f"{t}_{a}": v for t, d in res.items() for a, v in d.items()}})
    print(out[-1], flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1)
