#!/usr/bin/env python3
"""direct vs residual L_vcons: what each loss sees, measured on the SAME valbench5s windows for
  residual = main run ckpt 2250 (goalprog_mode_real6_nomr_hg0_grad0.5, vcons_mode=residual, closest step to the ablation)
  direct   = ablation ckpt 2000 (goalprog_mode_real6_nomr_hg0_grad0.5_direct, vcons_mode=direct)
Geometry exactly as training (train_hist_cond.py): GT metric depth at latent key frames start+4k, c2w = _carla_pose_to_mat
(no offset / flip for CARLA), intrinsics.npy rescaled to the 64x64 latent grid, gt_visibility mask, latent 0 := GT.
Losses = utils/warp_torch.vcons_loss with the training pair set (gaps 1,2,4,8 both directions + anchors (j,0)).

Per sample -> <out>/s<idx>.npz:
  stats  floor (= direct loss at GT), raw_{res,dir} (direct loss on the output), resid_{res,dir} (residual loss on the output),
         err_{res,dir} (|x - z|)
  per latent t (gap 1, s = t-1): floor_t, raw_t, resid_t, err_t ; maps (t, 64, 64) for the figure samples
  per pixel frame: sharp (variance of Laplacian, gray), motion (mean |frame_f - frame_{f-1}|) for gt / res / dir
  python vcons_cmp.py --idx 0 1 2 ... --gpu 0 --out DIR [--maps]
"""
import argparse, json, os, sys
import cv2, numpy as np, torch

COS = "/root/project/egowam/diffusers/examples/cosmos"
sys.path.insert(0, COS); sys.path.insert(0, f"{COS}/tools")
import latent_equivariance as LE                                                     # noqa: E402
from utils.warp_torch import warp_feat, gt_visibility, relative_pose, rescale_intrinsics, vcons_loss  # noqa: E402
from utils.data_utils import _carla_pose_to_mat                                      # noqa: E402

H_ = "/root/project/egowam/results_syn/goalprog_mode_real6_nomr_hg0_grad0.5"
RUNS = {"res": f"{H_}/ckpt2250/valbench5s_all120", "dir": f"{H_}/ckptdirect2000/valbench5s_all120"}
DEV = "cuda"


def frames(p, n=77):
    cap = cv2.VideoCapture(p); out = []
    while len(out) < n:
        ok, f = cap.read()
        if not ok:
            break
        out.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    assert len(out) == n, (p, len(out))
    return out


def pix_stats(F):
    g = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY).astype(np.float32) for f in F]
    sharp = np.array([cv2.Laplacian(x, cv2.CV_32F).var() for x in g])
    motion = np.array([0.0] + [np.abs(g[i] - g[i - 1]).mean() for i in range(1, len(g))])
    return sharp, motion


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--idx", type=int, nargs="+", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--maps", action="store_true")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    meta = json.load(open(f"{RUNS['res']}/samples_meta.json"))
    codec = LE.Codec()
    toT = lambda L: torch.tensor(np.stack(L), device=DEV).permute(3, 0, 1, 2)[None].float() / 127.5 - 1
    for i in a.idx:
        out = f"{a.out}/s{i:03d}.npz"
        if os.path.exists(out):
            continue
        m = meta[str(i)]; sd, st = m["seq_dir"].rstrip("/"), int(m["start_frame"])
        gt = [cv2.cvtColor(cv2.imread(f"{sd}/images/{st + k:06d}.png"), cv2.COLOR_BGR2RGB) for k in range(77)]
        vids = {k: frames(f"{d}/sample_{i:02d}.mp4") for k, d in RUNS.items()}
        with torch.no_grad():
            z = codec.enc(toT(gt)); X = {k: codec.enc(toT(v)) for k, v in vids.items()}
        for x in X.values():
            x[:, :, 0] = z[:, :, 0]
        _, C, Tl, h, w = z.shape
        key = [st + 4 * k for k in range(Tl)]
        D = torch.stack([torch.tensor(np.load(f"{sd}/depth/{f:06d}.npy").astype(np.float32)) for f in key])[None].to(DEV)
        dg = torch.nn.functional.interpolate(D, size=(h, w), mode="nearest")
        P = np.load(f"{sd}/poses.npy")
        c2w = torch.tensor(np.stack([_carla_pose_to_mat(*P[f]) for f in key]), device=DEV).float()[None]
        K = torch.tensor(np.load(f"{sd}/intrinsics.npy"), device=DEV).float()[None]
        K64 = rescale_intrinsics(K, (h, w), D.shape[-2:])
        pairs = [(j + g, j) for g in (1, 2, 4, 8) for j in range(Tl - g)]
        pairs = pairs + [(s, t) for t, s in pairs]
        pw = [1.0] * len(pairs) + [1.0] * (Tl - 1)
        pairs = pairs + [(j, 0) for j in range(1, Tl)]
        learn = torch.ones(1, Tl, device=DEV); learn[:, 0] = 0
        R = {}
        with torch.no_grad():
            for k, x in X.items():
                Lr, dr = vcons_loss(x, z, dg, c2w, K64, pairs, pair_w=pw, learn=learn, mode="residual")
                Ld, dd = vcons_loss(x, z, dg, c2w, K64, pairs, pair_w=pw, learn=learn, mode="direct")
                R[f"resid_{k}"], R[f"raw_{k}"], R[f"err_{k}"] = float(Lr), float(Ld), float(dr["vcons_err"])
                R["floor"], R["mask_frac"] = float(dr["vcons_floor"]), float(dr["vcons_mask_frac"])
            # per latent t, gap 1 (t <- t-1): maps + masked means
            per = {n: [] for n in ("floor_t", "raw_t_res", "raw_t_dir", "resid_t_res", "resid_t_dir", "err_t_res", "err_t_dir", "mask_t")}
            maps = {n: [] for n in ("floor", "raw_res", "raw_dir", "resid_res", "resid_dir", "mask")}
            for t in range(1, Tl):
                ti, si = torch.tensor([t], device=DEV), torch.tensor([t - 1], device=DEV)
                T_ = relative_pose(c2w, ti, si)
                M = gt_visibility(dg[:, t:t + 1], dg[:, t - 1:t], T_, K64)[0]
                wz, _ = warp_feat(z[:, :, t - 1], dg[:, t:t + 1], T_, K64)
                fl = (z[:, :, t] - wz).abs().mean(1)[0]
                mm = lambda r: float(r[M].mean()) if M.any() else float("nan")
                per["floor_t"].append(mm(fl)); per["mask_t"].append(float(M.float().mean())); maps["floor"].append(fl); maps["mask"].append(M)
                for k, x in X.items():
                    wx, _ = warp_feat(x[:, :, t - 1], dg[:, t:t + 1], T_, K64)
                    rw = (x[:, :, t] - wx).abs().mean(1)[0]
                    e = x - z; we, _ = warp_feat(e[:, :, t - 1], dg[:, t:t + 1], T_, K64)
                    rr = (e[:, :, t] - we).abs().mean(1)[0]
                    per[f"raw_t_{k}"].append(mm(rw)); per[f"resid_t_{k}"].append(mm(rr))
                    per[f"err_t_{k}"].append(float(e[0, :, t].abs().mean()))
                    maps[f"raw_{k}"].append(rw); maps[f"resid_{k}"].append(rr)
        sh = {}; mo = {}
        for k, F in (("gt", gt), ("res", vids["res"]), ("dir", vids["dir"])):
            sh[k], mo[k] = pix_stats(F)
        save = dict(stats=json.dumps(R), **{k: np.array(v, np.float32) for k, v in per.items()},
                    **{f"sharp_{k}": v for k, v in sh.items()}, **{f"motion_{k}": v for k, v in mo.items()})
        if a.maps:
            save.update({f"map_{k}": torch.stack(v).float().cpu().numpy().astype(np.float16) for k, v in maps.items()})
        np.savez_compressed(out, **save)
        print(i, json.dumps({k: round(v, 3) for k, v in R.items()}), flush=True)


if __name__ == "__main__":
    main()
