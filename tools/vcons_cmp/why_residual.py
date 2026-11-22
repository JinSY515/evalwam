#!/usr/bin/env python3
"""Why residual L_vcons beats direct, with NO trained model in the loop (loss-only, confound-free).

Training geometry / pairs / mask exactly as vcons_cmp.py (= train_hist_cond.py). For each valbench window:
 A. ranking test: GT and degraded copies of GT (pixel-space, then VAE-encoded; latent 0 := GT as in training)
      blur2 / blur4 (Gaussian sigma px), tavg5 / tavg9 (temporal box mean over frames), still (every frame = frame 0),
      + the two generated videos (residual ck 2250, direct ck 2000)
    -> direct loss, residual loss, latent error |x - z|, PSNR / LPIPS-free sharpness vs GT.
    A sound consistency loss must score GT lowest; anything scoring below GT is rewarded for being wrong.
 B. drift test: start AT GT latents, minimise each loss alone (Adam, only frames 1.., warped side stop-grad as in training),
    decode -> where does the loss pull a perfect video?  residual: gradient is 0 at GT, nothing moves.
  python why_residual.py --idx ... --out DIR [--decode]
"""
import argparse, json, os, sys
import cv2, numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vcons_cmp import COS, RUNS, DEV, frames, LE, vcons_loss, rescale_intrinsics, _carla_pose_to_mat  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--idx", type=int, nargs="+", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--decode", action="store_true"); ap.add_argument("--steps", type=int, default=300)
ap.add_argument("--bench", default="valbench", choices=["valbench", "A", "B", "C"])
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
# LookOut (real Aria, unseen): exclusion-filtered A/B/C as /early; corrected (_lh) poses, poses.npy[f + 4] is frame f
# (largest GT-visibility of the offset/flip options on A, B and C), DA3 pseudo-GT depth (A: depth_da3, B: depth_da3_b made for
# frames 100..176 with the same add_lookout_depth_da3.py, C: depth). No generated videos on LookOut: loss-only tests.
AV = "/root/project/egowam/results_syn/goalprog_mode_real6_nomr_hg0_grad0.5/ckpta7500_8750"
LOSUB = {"A": "lookout_1c_all51_lh", "B": "lookout_1c_s100_all51_lh", "C": "lookout_turn77val35_lh"}
if a.bench == "valbench":
    meta = json.load(open(f"{RUNS['res']}/samples_meta.json")); OFF = 0
else:
    meta = json.load(open(f"{AV}/{LOSUB[a.bench]}/samples_meta.json")); OFF = 4
    EXL = json.load(open("/root/project/egowam/results_syn/goalprog_mode_real6_nomr_hg0_grad0.5/multi/v2/lookout_AB_exclude17.json"))
    if a.bench in "AB":
        a.idx = [i for i in a.idx if i not in set(EXL[a.bench])]


def depth_dir(sd):
    return {"valbench": f"{sd}/depth", "A": f"{sd}/depth_da3", "B": sd.replace("_lh/", "/") + "/depth_da3_b", "C": f"{sd}/depth"}[a.bench]
TAG = "w" if a.bench == "valbench" else a.bench
codec = LE.Codec()
toT = lambda L: torch.tensor(np.stack(L), device=DEV).permute(3, 0, 1, 2)[None].float() / 127.5 - 1
toU8 = lambda v: ((v[0].permute(1, 2, 3, 0).clamp(-1, 1).cpu().numpy() + 1) * 127.5).round().astype(np.uint8)   # (T,H,W,3)


def psnr(A, B):
    m = np.mean((np.stack(A).astype(np.float32) - np.stack(B).astype(np.float32)) ** 2)
    return float(10 * np.log10(255 ** 2 / max(m, 1e-9)))


def sharp(F):
    return float(np.mean([cv2.Laplacian(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY).astype(np.float32), cv2.CV_32F).var() for f in F[1:]]))


for i in a.idx:
    out = f"{a.out}/{TAG}{i:03d}.json"
    if os.path.exists(out):
        continue
    m = meta[str(i)]; sd, st = m["seq_dir"].rstrip("/"), int(m["start_frame"])
    gt = [cv2.cvtColor(cv2.imread(f"{sd}/images/{st + k:06d}.png"), cv2.COLOR_BGR2RGB) for k in range(77)]
    G = np.stack(gt).astype(np.float32)
    var = {"GT": gt,
           "blur2": [cv2.GaussianBlur(f, (0, 0), 2) for f in gt], "blur4": [cv2.GaussianBlur(f, (0, 0), 4) for f in gt],
           "tavg5": [np.clip(G[max(0, k - 2):k + 3].mean(0), 0, 255).astype(np.uint8) for k in range(77)],
           "tavg9": [np.clip(G[max(0, k - 4):k + 5].mean(0), 0, 255).astype(np.uint8) for k in range(77)],
           "still": [gt[0]] * 77}
    if a.bench == "valbench":
        var["gen_res"] = frames(f"{RUNS['res']}/sample_{i:02d}.mp4"); var["gen_dir"] = frames(f"{RUNS['dir']}/sample_{i:02d}.mp4")
    with torch.no_grad():
        Z = {k: codec.enc(toT(v)) for k, v in var.items()}
    z = Z["GT"]
    for k in Z:
        Z[k][:, :, 0] = z[:, :, 0]
    _, C, Tl, h, w = z.shape
    key = [st + 4 * k for k in range(Tl)]
    D = torch.stack([torch.tensor(np.load(f"{depth_dir(sd)}/{f:06d}.npy").astype(np.float32)) for f in key])[None].to(DEV)
    dg = torch.nn.functional.interpolate(D, size=(h, w), mode="nearest")
    P = np.load(f"{sd}/poses.npy")
    c2w = torch.tensor(np.stack([_carla_pose_to_mat(*P[f + OFF]) for f in key]), device=DEV).float()[None]
    K64 = rescale_intrinsics(torch.tensor(np.load(f"{sd}/intrinsics.npy"), device=DEV).float()[None], (h, w), D.shape[-2:])
    pairs = [(j + g, j) for g in (1, 2, 4, 8) for j in range(Tl - g)]
    pairs = pairs + [(s, t) for t, s in pairs]
    pw = [1.0] * len(pairs) + [1.0] * (Tl - 1)
    pairs = pairs + [(j, 0) for j in range(1, Tl)]
    learn = torch.ones(1, Tl, device=DEV); learn[:, 0] = 0
    L = lambda x, mode: vcons_loss(x, z, dg, c2w, K64, pairs, pair_w=pw, learn=learn, mode=mode)[0]
    R = {"idx": i, "bench": a.bench, "seq": sd, "start": st, "rank": {}}
    with torch.no_grad():
        for k, x in Z.items():
            R["rank"][k] = {"direct": float(L(x, "direct")), "residual": float(L(x, "residual")),
                            "lat_err": float((x - z)[:, :, 1:].abs().mean()), "psnr": psnr(var[k], gt) if k != "GT" else None,
                            "sharp_ratio": sharp(var[k]) / max(sharp(gt), 1e-6)}
    # B. drift from GT under each loss alone
    R["drift"] = {}
    for mode in ("direct", "residual"):
        x = z.clone().requires_grad_(True)
        opt = torch.optim.Adam([x], lr=0.02)
        tr = []
        for it in range(a.steps):
            opt.zero_grad(); l = L(x, mode)
            if l.requires_grad:
                l.backward(); x.grad[:, :, 0] = 0; opt.step()
            if it % 25 == 0 or it == a.steps - 1:
                tr.append((it, float(l), float((x.detach() - z)[:, :, 1:].abs().mean())))
        xf = x.detach()
        d = {"trace": tr, "loss_start": tr[0][1], "loss_end": tr[-1][1], "lat_dist_end": tr[-1][2],
             "direct_end": float(L(xf, "direct")), "residual_end": float(L(xf, "residual"))}
        if a.decode:
            with torch.no_grad():
                v = toU8(codec.dec(xf))[:77]
            d["psnr_vs_gt"] = psnr(list(v), gt); d["sharp_ratio"] = sharp(list(v)) / max(sharp(gt), 1e-6)
            for kf in (0, 20, 40, 60, 76):
                cv2.imwrite(f"{a.out}/{TAG}{i:03d}_{mode}_f{kf:02d}.jpg", cv2.cvtColor(v[kf], cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
            if mode == "direct":
                with torch.no_grad():
                    g0 = toU8(codec.dec(z))[:77]          # VAE round-trip of GT = the fair reference for decoded drift
                d["psnr_gt_roundtrip"] = psnr(list(g0), gt)
                for kf in (0, 20, 40, 60, 76):
                    cv2.imwrite(f"{a.out}/{TAG}{i:03d}_gtvae_f{kf:02d}.jpg", cv2.cvtColor(g0[kf], cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
                    cv2.imwrite(f"{a.out}/{TAG}{i:03d}_gt_f{kf:02d}.jpg", cv2.cvtColor(gt[kf], cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
        R["drift"][mode] = d
    json.dump(R, open(out, "w"), indent=1)
    rk = R["rank"]
    print(i, "direct:", {k: round(v["direct"], 3) for k, v in rk.items()}, "| drift direct", round(R["drift"]["direct"]["loss_start"], 3), "->",
          round(R["drift"]["direct"]["loss_end"], 3), "dist", round(R["drift"]["direct"]["lat_dist_end"], 3),
          "| residual dist", round(R["drift"]["residual"]["lat_dist_end"], 3), flush=True)
