#!/usr/bin/env python3
"""What does each L_vcons push on a REAL generated video?  Per latent element of every training pair (t <- s, masked as in
training, only e_t / x_t gets gradient), with W linear:
    a = x_t - W(x_s)   (direct residual)  = b + c
    b = e_t - W(e_s)   (residual residual, e = x - z)
    c = z_t - W(z_s)   (GT floor term: what GT itself leaves after warping)
L1 gradient on x_t:  direct -> sign(a),  residual -> sign(b)  (= the error-correcting direction).
Stats (masked elements): agree = P(sign a == sign b), dom = P(|c| > |b|), agree when c dominates / when it does not,
follow_c = P(sign a == sign c); and cosines of the summed per-element gradients (direct vs residual, direct vs sign c).
Generated videos: synthetic = valbench (residual ck 2250, direct ck 2000); real = LookOut A/B/C (residual = ours a7500_8750,
direct = ck 2000), geometry as why_residual.py.   python why_grad.py --bench valbench|A|B|C --idx ... --out DIR"""
import argparse, json, os, sys
import cv2, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vcons_cmp import COS, DEV, frames, LE, rescale_intrinsics, _carla_pose_to_mat  # noqa: E402
from utils.warp_torch import warp_feat, gt_visibility, relative_pose  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--idx", type=int, nargs="+", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--bench", default="valbench", choices=["valbench", "A", "B", "C"]); a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
H_ = "/root/project/egowam/results_syn/goalprog_mode_real6_nomr_hg0_grad0.5"
LOSUB = {"A": "lookout_1c_all51_lh", "B": "lookout_1c_s100_all51_lh", "C": "lookout_turn77val35_lh"}
if a.bench == "valbench":
    RUNS = {"res": f"{H_}/ckpt2250/valbench5s_all120", "dir": f"{H_}/ckptdirect2000/valbench5s_all120"}; OFF = 0
else:
    RUNS = {"res": f"{H_}/ckpta7500_8750/{LOSUB[a.bench]}", "dir": f"{H_}/ckptdirect2000/{LOSUB[a.bench]}"}; OFF = 4
    EXL = json.load(open(f"{H_}/multi/v2/lookout_AB_exclude17.json"))
    if a.bench in "AB":
        a.idx = [i for i in a.idx if i not in set(EXL[a.bench])]
meta = json.load(open(f"{RUNS['res']}/samples_meta.json"))
depth_dir = lambda sd: {"valbench": f"{sd}/depth", "A": f"{sd}/depth_da3", "B": sd.replace("_lh/", "/") + "/depth_da3_b", "C": f"{sd}/depth"}[a.bench]
TAG = "w" if a.bench == "valbench" else a.bench
codec = LE.Codec()
toT = lambda L: torch.tensor(np.stack(L), device=DEV).permute(3, 0, 1, 2)[None].float() / 127.5 - 1
for i in a.idx:
    out = f"{a.out}/{TAG}{i:03d}.json"
    if os.path.exists(out) or str(i) not in meta:
        continue
    m = meta[str(i)]; sd, st = m["seq_dir"].rstrip("/"), int(m["start_frame"])
    gt = [cv2.cvtColor(cv2.imread(f"{sd}/images/{st + k:06d}.png"), cv2.COLOR_BGR2RGB) for k in range(77)]
    vids = {k: frames(f"{d}/sample_{i:02d}.mp4") for k, d in RUNS.items()}
    f0 = {k: float(10 * np.log10(255 ** 2 / max(np.mean((v[0].astype(np.float32) - cv2.resize(gt[0], v[0].shape[1::-1]).astype(np.float32)) ** 2), 1e-9)))
          for k, v in vids.items()}
    vids = {k: [cv2.resize(f, gt[0].shape[1::-1]) if f.shape != gt[0].shape else f for f in v] for k, v in vids.items()}
    with torch.no_grad():
        z = codec.enc(toT(gt)); X = {k: codec.enc(toT(v)) for k, v in vids.items()}
    for x in X.values():
        x[:, :, 0] = z[:, :, 0]
    _, C, Tl, h, w = z.shape
    key = [st + 4 * k for k in range(Tl)]
    D = torch.stack([torch.tensor(np.load(f"{depth_dir(sd)}/{f:06d}.npy").astype(np.float32)) for f in key])[None].to(DEV)
    dg = torch.nn.functional.interpolate(D, size=(h, w), mode="nearest")
    P = np.load(f"{sd}/poses.npy")
    c2w = torch.tensor(np.stack([_carla_pose_to_mat(*P[f + OFF]) for f in key]), device=DEV).float()[None]
    K64 = rescale_intrinsics(torch.tensor(np.load(f"{sd}/intrinsics.npy"), device=DEV).float()[None], (h, w), D.shape[-2:])
    pairs = [(j + g, j) for g in (1, 2, 4, 8) for j in range(Tl - g)]
    pairs = pairs + [(s, t) for t, s in pairs] + [(j, 0) for j in range(1, Tl)]
    R = {"idx": i, "bench": a.bench, "seq": sd, "start": st, "f0_psnr": f0}
    with torch.no_grad():
        for k, x in X.items():
            e = x - z; S = {"n": 0, "agree": 0, "dom": 0, "agree_dom": 0, "agree_nodom": 0, "follow_c": 0}
            GD = torch.zeros_like(x); GR = torch.zeros_like(x); GC = torch.zeros_like(x)
            for t, s in pairs:
                T_ = relative_pose(c2w, torch.tensor([t], device=DEV), torch.tensor([s], device=DEV))
                M = gt_visibility(dg[:, t:t + 1], dg[:, s:s + 1], T_, K64)[0]          # (h,w)
                if not M.any():
                    continue
                wx, _ = warp_feat(x[:, :, s], dg[:, t:t + 1], T_, K64); wz, _ = warp_feat(z[:, :, s], dg[:, t:t + 1], T_, K64)
                aa = (x[:, :, t] - wx)[0][:, M]; cc = (z[:, :, t] - wz)[0][:, M]; bb = aa - cc     # (C, n)
                sa, sb, sc = torch.sign(aa), torch.sign(bb), torch.sign(cc)
                dom = cc.abs() > bb.abs(); ag = sa == sb
                S["n"] += aa.numel(); S["agree"] += int(ag.sum()); S["dom"] += int(dom.sum())
                S["agree_dom"] += int((ag & dom).sum()); S["agree_nodom"] += int((ag & ~dom).sum()); S["follow_c"] += int((sa == sc).sum())
                GD[0, :, t][:, M] += sa; GR[0, :, t][:, M] += sb; GC[0, :, t][:, M] += sc
            n = max(S["n"], 1); nd = max(S["dom"], 1); nn = max(S["n"] - S["dom"], 1)
            cos = lambda u, v: float((u * v).sum() / ((u.norm() * v.norm()).clamp(min=1e-9)))
            R[k] = {"agree": S["agree"] / n, "dom": S["dom"] / n, "agree_dom": S["agree_dom"] / nd, "agree_nodom": S["agree_nodom"] / nn,
                    "follow_c": S["follow_c"] / n, "cos_DR": cos(GD, GR), "cos_DC": cos(GD, GC), "cos_RC": cos(GR, GC),
                    "lat_err": float(e[:, :, 1:].abs().mean())}
    json.dump(R, open(out, "w"), indent=1)
    print(TAG, i, "f0", {k: round(v, 1) for k, v in f0.items()}, {k: {q: round(v, 3) for q, v in R[k].items()} for k in X}, flush=True)
