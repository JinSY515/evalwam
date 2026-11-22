#!/usr/bin/env python3
"""Why the visibility mask in residual L_vcons, with NO trained model (loss-only, like why_residual.py).

Geometry / depth / poses / pairs exactly as why_grad.py (= training): latent 64x64 grid, GT (valbench) or DA3 pseudo-GT
(LookOut) depth at key frames start+4k, gaps 1,2,4,8 both directions (anchor pairs (j,0) left out: they compare against the
given frame 0, not against another view).
  M   = gt_visibility (in frame & depth-consistent = not occluded & NEAR < d < FAR)   <- training mask
  F   = in_frame only (what a loss "without the visibility mask" would average over)
  O   = F & ~M, split into Oc = occluded/disoccluded (valid depth) and Or = out of depth range (sky, d >= FAR)
Synthetic error fields added to the GT latent z (x = z + e), so the true error is known:
  consistent    e_t(p) = g(X_world(p)): every 3D point carries ONE error, seen identically from every frame
                -> a geometrically perfect video-with-error; a correct consistency loss must give ~0
  inconsistent  same g, but each frame gets its own random phase -> errors disagree across views; loss must be large
g = 16-channel sum of 6 random 3D sinusoids (wavelength 4-16 m), scaled to mean |e| = 0.47 (= the latent error of the
generated videos, vcons_cmp: 0.469). Reported per window:
  frac_occ = |O| / |F|; residual |e_t - W(e_s)| averaged over M / F / O, for both fields;
  floor_vis / floor_occ = |z_t - W(z_s)| over M / O (how wrong the warp correspondence itself is in each region).
  python why_mask.py --bench valbench|A|B|C --idx ... --out DIR [--gpu N]
"""
import argparse, json, os, sys
import cv2, numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vcons_cmp import COS, DEV, LE, rescale_intrinsics, _carla_pose_to_mat  # noqa: E402
from utils.warp_torch import warp_feat, gt_visibility, relative_pose, _project, _pixel_grid, _C3_T, NEAR, FAR  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--idx", type=int, nargs="+", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--bench", default="valbench", choices=["valbench", "A", "B", "C"]); a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
H_ = "/root/project/egowam/results_syn/goalprog_mode_real6_nomr_hg0_grad0.5"
LOSUB = {"A": "lookout_1c_all51_lh", "B": "lookout_1c_s100_all51_lh", "C": "lookout_turn77val35_lh"}
if a.bench == "valbench":
    MD = f"{H_}/ckpt2250/valbench5s_all120"; OFF = 0
else:
    MD = f"{H_}/ckpta7500_8750/{LOSUB[a.bench]}"; OFF = 4
    EXL = json.load(open(f"{H_}/multi/v2/lookout_AB_exclude17.json"))
    if a.bench in "AB":
        a.idx = [i for i in a.idx if i not in set(EXL[a.bench])]
meta = json.load(open(f"{MD}/samples_meta.json"))
depth_dir = lambda sd: {"valbench": f"{sd}/depth", "A": f"{sd}/depth_da3", "B": sd.replace("_lh/", "/") + "/depth_da3_b", "C": f"{sd}/depth"}[a.bench]
TAG = {"valbench": "m", "A": "mA", "B": "mB", "C": "mC"}[a.bench]
codec = LE.Codec()
toT = lambda L: torch.tensor(np.stack(L), device=DEV).permute(3, 0, 1, 2)[None].float() / 127.5 - 1
TARGET = 0.469


def world_pts(dg, c2w, K):
    """(1,T,h,w) depth, (1,T,4,4) c2w, (1,4) K -> (T,h,w,3) world points (CARLA axes), same arithmetic as _project."""
    T, h, w = dg.shape[1:]
    uu, vv = _pixel_grid(h, w, dg.device, dg.dtype)
    fx, fy, cx, cy = K[0].unbind(-1)
    d = dg[0]
    p_pin = torch.stack([(uu - cx) / fx * d, (vv - cy) / fy * d, d], -1)          # (T,h,w,3)
    p_cam = p_pin @ dg.new_tensor(_C3_T).T
    return torch.einsum("tij,thwj->thwi", c2w[0, :, :3, :3], p_cam) + c2w[0, :, None, None, :3, 3]


def field(X, rng, C, per_frame):
    T = X.shape[0]
    k = rng.normal(size=(6, 3)); k /= np.linalg.norm(k, axis=1, keepdims=True)
    k *= (2 * np.pi / rng.uniform(4, 16, size=(6, 1)))
    A = rng.normal(size=(C, 6)) / np.sqrt(6)
    ph = rng.uniform(0, 2 * np.pi, size=(T if per_frame else 1, 6))
    ph = np.repeat(ph, T, 0) if not per_frame else ph
    kt, At, pt = (X.new_tensor(v) for v in (k, A, ph))
    s = torch.sin(torch.einsum("thwc,kc->thwk", X, kt) + pt[:, None, None, :])      # (T,h,w,6)
    return torch.einsum("thwk,ck->cthw", s, At)[None]                                # (1,C,T,h,w)


for i in a.idx:
    out = f"{a.out}/{TAG}{i:03d}.json"
    if os.path.exists(out) or str(i) not in meta:
        continue
    m = meta[str(i)]; sd, st = m["seq_dir"].rstrip("/"), int(m["start_frame"])
    gt = [cv2.cvtColor(cv2.imread(f"{sd}/images/{st + k:06d}.png"), cv2.COLOR_BGR2RGB) for k in range(77)]
    with torch.no_grad():
        z = codec.enc(toT(gt))
    _, C, Tl, h, w = z.shape
    key = [st + 4 * k for k in range(Tl)]
    D = torch.stack([torch.tensor(np.load(f"{depth_dir(sd)}/{f:06d}.npy").astype(np.float32)) for f in key])[None].to(DEV)
    dg = torch.nn.functional.interpolate(D, size=(h, w), mode="nearest")
    P = np.load(f"{sd}/poses.npy")
    c2w = torch.tensor(np.stack([_carla_pose_to_mat(*P[f + OFF]) for f in key]), device=DEV).float()[None]
    K64 = rescale_intrinsics(torch.tensor(np.load(f"{sd}/intrinsics.npy"), device=DEV).float()[None], (h, w), D.shape[-2:])
    pairs = [(j + g, j) for g in (1, 2, 4, 8) for j in range(Tl - g)]
    pairs = pairs + [(s, t) for t, s in pairs]
    rng = np.random.default_rng(i)
    with torch.no_grad():
        X = world_pts(dg.clamp(max=1000.0), c2w, K64)
        E = {"consistent": field(X, rng, C, False)}
        rng = np.random.default_rng(i)                                  # same k / A, only the phases differ per frame
        E["inconsistent"] = field(X, rng, C, True)
        for kk in E:
            E[kk] = E[kk] * (TARGET / E[kk].abs().mean().clamp(min=1e-9))
        acc = {f"{kk}_{r}": [0.0, 0] for kk in E for r in ("M", "F", "O", "Oc", "Or")}
        acc.update({"floor_M": [0.0, 0], "floor_O": [0.0, 0], "floor_Oc": [0.0, 0]}); nM = nF = nO = nOc = 0
        for t, s in pairs:
            T_ = relative_pose(c2w, torch.tensor([t], device=DEV), torch.tensor([s], device=DEV))
            M = gt_visibility(dg[:, t:t + 1], dg[:, s:s + 1], T_, K64)[0]
            _, _, Fm = _project(dg[:, t:t + 1], T_, K64); Fm = Fm[0]
            O = Fm & ~M; d_t = dg[0, t]; Oc = O & (d_t > NEAR) & (d_t < FAR); Or = O & ~Oc
            nM += int(M.sum()); nF += int(Fm.sum()); nO += int(O.sum()); nOc += int(Oc.sum())
            for kk, e in E.items():
                we, _ = warp_feat(e[:, :, s], dg[:, t:t + 1], T_, K64)
                r = (e[:, :, t] - we).abs().mean(1)[0]                    # (h,w)
                for nm, R in (("M", M), ("F", Fm), ("O", O), ("Oc", Oc), ("Or", Or)):
                    if R.any():
                        acc[f"{kk}_{nm}"][0] += float(r[R].sum()); acc[f"{kk}_{nm}"][1] += int(R.sum())
            wz, _ = warp_feat(z[:, :, s], dg[:, t:t + 1], T_, K64)
            fz = (z[:, :, t] - wz).abs().mean(1)[0]
            for nm, R in (("M", M), ("O", O), ("Oc", Oc)):
                if R.any():
                    acc[f"floor_{nm}"][0] += float(fz[R].sum()); acc[f"floor_{nm}"][1] += int(R.sum())
    R = {"idx": i, "bench": a.bench, "seq": sd, "start": st, "frac_occ": nO / max(nF, 1), "frac_occ_depth": nOc / max(nF, 1), "frac_mask": nM / max(nF, 1),
         "err_mean": TARGET, **{k: (v[0] / v[1] if v[1] else None) for k, v in acc.items()}}
    json.dump(R, open(out, "w"), indent=1)
    print(TAG, i, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in R.items() if k not in ("seq",)}, flush=True)
