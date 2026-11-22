#!/usr/bin/env python3
"""SSIM / LPIPS(alex) / PSNR of the drift end points (why_residual.py --decode), from the saved jpg frames f20/40/60/76
(f00 = latent 0, held at GT during the drift, left out). Same ssim / lpips_dist as tools/eval_video_quality.py.
Writes <dir>/<tag>NNN_q.json: {variant: {psnr, ssim, lpips}} for direct / residual / gtvae (vs gt)."""
import glob, json, os, sys
import numpy as np
from PIL import Image
sys.path.insert(0, "/root/project/evalwam/tools")
import eval_video_quality as Q  # noqa: E402

D = "/root/project/egowam/results_syn/goalprog_mode_real6_nomr_hg0_grad0.5_direct/vcons_cmp"
F = (20, 40, 60, 76)
rd = lambda p: np.asarray(Image.open(p).convert("RGB"))
for d, tag in [(f"{D}/why", "w")] + [(f"{D}/why_lookout", x) for x in "ABC"]:
    for j in sorted(glob.glob(f"{d}/{tag}[0-9][0-9][0-9].json")):
        b = j[:-5]; out = {}
        for v in ("direct", "residual", "gtvae"):
            ps, ss, lp = [], [], []
            for f in F:
                g, x = rd(f"{b}_gt_f{f:02d}.jpg"), rd(f"{b}_{v}_f{f:02d}.jpg")
                ps.append(Q.psnr(g, x)); ss.append(Q.ssim(g, x)); lp.append(Q.lpips_dist(g, x, "alex"))
            out[v] = {"psnr": float(np.mean(ps)), "ssim": float(np.mean(ss)), "lpips": float(np.mean(lp))}
        json.dump(out, open(f"{b}_q.json", "w"))
    print(tag, "done", flush=True)
