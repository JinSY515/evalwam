#!/usr/bin/env python3
"""Design choices of residual L_vcons, with NO retraining (loss-only + the trained models' outputs).

Geometry / depth / poses exactly as training and as why_mask.py: latent 64x64 grid, key frames start+4k (20 latents),
GT depth (valbench) or DA3 pseudo-GT depth (LookOut A/B/C, poses.npy[f + 4], A/B exclusion list), gt_visibility mask.
Per window, for EVERY ordered latent pair (t <- s), t >= 1 (t = learnable target, s = warped source; s = 0 is an anchor),
grouped later by gap |t - s| = 1..19:
  nF, nM, nOc          in-frame / visibility-mask / occluded-with-valid-depth pixel counts
  floor_M              |z_t - W(z_s)| on M  -- how wrong the GT correspondence is even for the GT video (view-dependent
                        appearance, VAE non-equivariance, interpolation, depth error on DA3)
  cons_*, incons_*     residual of the synthetic 3D-consistent / per-frame-inconsistent error fields of why_mask.py (mean
                        |e| 0.469) over M and F -> separation ratio = how well the loss tells a consistent error from an
                        inconsistent one at that gap, with and without the mask
  {model}_err_M, {model}_res_M   generated video: |e_t| and |e_t - W(e_s)| on M, e = x - z, for
                        ours (avg 7500+8750, trained with gaps 1,2,4,8 + anchors) and abl4 (IV_cont5k ck 6000, no L_vcons)
Drift (start at ours' generated latent, frame 0 := GT as in training, Adam lr 0.02, --steps; only L_vcons, nothing else):
  train      stop-grad on the warped source, visibility mask, gaps 1,2,4,8 both directions + anchors (j <- 0)
  nosg       same, gradient also through W(e_s)
  nomask     same as train, mask = in-frame only
  g1 g2 g4 g8 g16   only that gap (both directions, incl. pairs with frame 0), mask, stop-grad
  anchor     only (j <- 0)
 -> per variant: grad cosine with the error-correcting direction -(x - z) at step 0, |x - z| at the end (all / visible from
    frame 0 / not visible from frame 0), and the masked residual at gaps 1 / 4 / 8 / 16 at the end (consistency).
  python vcons_design.py --bench valbench|A|B|C --idx ... --out DIR [--steps 200]
"""
import argparse, json, os, sys
import cv2, numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vcons_cmp import DEV, LE, frames, rescale_intrinsics, _carla_pose_to_mat  # noqa: E402
from utils.warp_torch import warp_feat, gt_visibility, relative_pose, _project, _pixel_grid, _C3_T, vcons_loss, NEAR, FAR  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--idx", type=int, nargs="+", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--bench", default="valbench", choices=["valbench", "A", "B", "C"]); ap.add_argument("--steps", type=int, default=200)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
H_ = "/root/project/egowam/results_syn/goalprog_mode_real6_nomr_hg0_grad0.5"
AB4 = "/root/project/egowam/results_syn/abl/IV_cont5k/ckpt6000"
LOSUB = {"valbench": "valbench5s_all120", "A": "lookout_1c_all51_lh", "B": "lookout_1c_s100_all51_lh", "C": "lookout_turn77val35_lh"}
SUB = LOSUB[a.bench]; OFF = 0 if a.bench == "valbench" else 4
RUNS = {"ours": f"{H_}/ckpta7500_8750/{SUB}", "abl4": f"{AB4}/{SUB}"}
if a.bench in "AB":
    EXL = json.load(open(f"{H_}/multi/v2/lookout_AB_exclude17.json")); a.idx = [i for i in a.idx if i not in set(EXL[a.bench])]
meta = json.load(open(f"{RUNS['ours']}/samples_meta.json"))
depth_dir = lambda sd: {"valbench": f"{sd}/depth", "A": f"{sd}/depth_da3", "B": sd.replace("_lh/", "/") + "/depth_da3_b", "C": f"{sd}/depth"}[a.bench]
TAG = {"valbench": "v", "A": "A", "B": "B", "C": "C"}[a.bench]
codec = LE.Codec()
toT = lambda L: torch.tensor(np.stack(L), device=DEV).permute(3, 0, 1, 2)[None].float() / 127.5 - 1
TARGET = 0.469
TRAIN_GAPS = (1, 2, 4, 8)
EVAL_GAPS = (1, 2, 4, 8, 16)


# world_pts / field: copied from why_mask.py (same synthetic error fields)
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





class Geo:
    """All ordered pairs (t <- s), t >= 1, geometry precomputed once (GT only, no grad)."""

    def __init__(self, dg, c2w, K):
        Tl = dg.shape[1]
        self.pairs = [(t, s) for t in range(1, Tl) for s in range(Tl) if s != t]
        P = len(self.pairs); self.idx = {p: k for k, p in enumerate(self.pairs)}
        ti = torch.tensor([p[0] for p in self.pairs], device=DEV); si = torch.tensor([p[1] for p in self.pairs], device=DEV)
        self.ti, self.si = ti, si
        self.T = relative_pose(c2w, ti, si)                                                 # (P,4,4)
        self.K = K.expand(P, 4).contiguous()
        self.dt, self.ds = dg[0, ti][:, None], dg[0, si][:, None]                           # (P,1,h,w)
        self.M = gt_visibility(self.dt, self.ds, self.T, self.K)                            # (P,h,w)
        _, _, Fm = _project(self.dt, self.T, self.K); self.F = Fm
        d_t = dg[0, ti]
        self.Oc = Fm & ~self.M & (d_t > NEAR) & (d_t < FAR)

    def warp(self, feat_s, sel):                                                            # feat_s (n,C,h,w)
        return warp_feat(feat_s, self.dt[sel], self.T[sel], self.K[sel])[0]


def residual_maps(G, e, sel, chunk=96):
    """|e_t - W(e_s)| channel mean for pairs sel -> (n,h,w)."""
    out = []
    for c in range(0, len(sel), chunk):
        ss = sel[c:c + chunk]
        out.append((e[0][:, G.ti[ss]].permute(1, 0, 2, 3) - G.warp(e[0][:, G.si[ss]].permute(1, 0, 2, 3), ss)).abs().mean(1))
    return torch.cat(out)


def loss(G, x, z, sel, sg=True, mask="M"):
    e = x - z
    es = e[0][:, G.si[sel]].permute(1, 0, 2, 3)
    if sg:
        es = es.detach()
    r = (e[0][:, G.ti[sel]].permute(1, 0, 2, 3) - G.warp(es, sel)).abs().mean(1)
    W = (G.M if mask == "M" else G.F)[sel].float()
    return (r * W).sum() / W.sum().clamp(min=1.0)


for i in a.idx:
    out = f"{a.out}/{TAG}{i:03d}.json"
    if os.path.exists(out) or str(i) not in meta:
        continue
    m = meta[str(i)]; sd, st = m["seq_dir"].rstrip("/"), int(m["start_frame"])
    gt = [cv2.cvtColor(cv2.imread(f"{sd}/images/{st + k:06d}.png"), cv2.COLOR_BGR2RGB) for k in range(77)]
    vids = {k: frames(f"{d}/sample_{i:02d}.mp4") for k, d in RUNS.items() if os.path.exists(f"{d}/sample_{i:02d}.mp4")}
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
    R = {"idx": i, "bench": a.bench, "seq": sd, "start": st, "models": list(X)}
    with torch.no_grad():
        G = Geo(dg, c2w, K64)
        allp = torch.arange(len(G.pairs), device=DEV)
        rng = np.random.default_rng(i)
        Xw = world_pts(dg.clamp(max=1000.0), c2w, K64)
        E = {"cons": field(Xw, rng, C, False)}
        rng = np.random.default_rng(i)
        E["incons"] = field(Xw, rng, C, True)
        for k in E:
            E[k] = E[k] * (TARGET / E[k].abs().mean().clamp(min=1e-9)); E[k][:, :, 0] = 0          # frame 0 = given image
        S = {"t": [p[0] for p in G.pairs], "s": [p[1] for p in G.pairs],
             "nF": G.F.sum((1, 2)).tolist(), "nM": G.M.sum((1, 2)).tolist(), "nOc": G.Oc.sum((1, 2)).tolist()}
        msum = lambda r, Mk: (r * Mk.float()).sum((1, 2)).tolist()
        S["floor_M"] = msum(residual_maps(G, z, allp), G.M)
        for k, e in E.items():
            r = residual_maps(G, e, allp); S[f"{k}_M"] = msum(r, G.M); S[f"{k}_F"] = msum(r, G.F)
        for k, x in X.items():
            e = x - z; r = residual_maps(G, e, allp)
            S[f"{k}_res_M"] = msum(r, G.M)
            S[f"{k}_err_M"] = msum(e[0][:, G.ti].abs().mean(0), G.M)
        R["pairs"] = S
        # sanity: our stop-grad loss on the training pair set == utils.warp_torch.vcons_loss
        tp = [(j + g, j) for g in TRAIN_GAPS for j in range(Tl - g)]; tp = tp + [(s, t) for t, s in tp]
        tp = [p for p in tp if p[0] >= 1] + [(j, 0) for j in range(1, Tl)]
        sel_train = torch.tensor([G.idx[p] for p in tp], device=DEV)
        if "ours" in X:
            learn = torch.ones(1, Tl, device=DEV); learn[:, 0] = 0
            ref = float(vcons_loss(X["ours"], z, dg, c2w, K64, tp, learn=learn, mode="residual")[0])
            mine = float(loss(G, X["ours"], z, sel_train))
            assert abs(ref - mine) < 1e-4 * max(1.0, ref), (ref, mine)
            R["sanity_vcons"] = [ref, mine]
    if "ours" not in X:
        json.dump(R, open(out, "w")); continue

    def sel(cond):
        return torch.tensor([k for k, (t, s) in enumerate(G.pairs) if cond(t, s)], device=DEV)
    sel_gap = lambda g: sel(lambda t, s: abs(t - s) == g and s >= 1)                     # frame-0 pairs (anchors) apart
    sel_pairs = sel(lambda t, s: abs(t - s) in TRAIN_GAPS and s >= 1)                    # random-pair part of training
    sel_anchor = torch.tensor([G.idx[(j, 0)] for j in range(1, Tl)], device=DEV)
    # (1) step-0 gradient of L_vcons on each model's generated video: cosine with the error-correcting direction -(x - z).
    #     anchors (j <- 0) have e_0 = 0 -> |e_j| = plain L1 to GT whatever the mask / stop-grad, so they are kept apart.
    GV = {"pairs_sg_M": (sel_pairs, True, "M"), "pairs_nosg_M": (sel_pairs, False, "M"), "pairs_sg_F": (sel_pairs, True, "F"),
          "pairs_nosg_F": (sel_pairs, False, "F"), "anchor_M": (sel_anchor, True, "M"), "anchor_F": (sel_anchor, True, "F"),
          "train": (sel_train, True, "M"), **{f"g{g}": (sel_gap(g), True, "M") for g in (1, 2, 4, 8, 12, 16)}}
    R["cos0"] = {}
    for mk_, x0 in X.items():
        e0 = x0 - z; R["cos0"][mk_] = {}
        for name, (ss, sg, mk) in GV.items():
            x = x0.clone().requires_grad_(True); loss(G, x, z, ss, sg, mk).backward(); gr = x.grad[:, :, 1:]
            R["cos0"][mk_][name] = float((gr * e0[:, :, 1:]).sum() / ((gr.norm() * e0[:, :, 1:].norm()).clamp(min=1e-9)))
    # (2) drift: minimise only L_vcons from ours' generated latent with one pair set -> how much error is removed and which
    #     gaps become consistent (pair sets without frame 0 can only make errors agree, not remove them)
    DV = {"train": sel_train, "pairs": sel_pairs, "anchor": sel_anchor, **{f"g{g}": sel_gap(g) for g in (1, 2, 4, 8, 16)}}
    vis0 = torch.zeros(1, 1, Tl, h, w, dtype=torch.bool, device=DEV)
    for j in range(1, Tl):
        vis0[0, 0, j] = G.M[G.idx[(j, 0)]]
    nov = ~vis0; nov[:, :, 0] = False
    x0 = X["ours"]

    def ev(x):
        with torch.no_grad():
            e = (x - z).abs().mean(1, keepdim=True)
            o = {"err_all": float(e[:, :, 1:].mean()), "err_vis0": float(e[vis0].mean()), "err_novis0": float(e[nov].mean()) if nov.any() else None}
            for g in EVAL_GAPS:
                sg_ = sel_gap(g); r = residual_maps(G, x - z, sg_); Mk = G.M[sg_].float()
                o[f"res_g{g}"] = float((r * Mk).sum() / Mk.sum().clamp(min=1.0))
            return o
    R["start"] = ev(x0); R["drift"] = {}
    for name, ss in DV.items():
        x = x0.clone().requires_grad_(True)
        opt = torch.optim.Adam([x], lr=0.02); tr = []
        for it in range(a.steps):
            opt.zero_grad(); l = loss(G, x, z, ss); l.backward(); x.grad[:, :, 0] = 0; opt.step()
            if it % 50 == 0 or it == a.steps - 1:
                tr.append([it, float(l.detach())])
        R["drift"][name] = {"trace": tr, **ev(x.detach())}
    json.dump(R, open(out, "w"))
    print(TAG, i, {k: round(v, 3) for k, v in R["cos0"]["ours"].items()}, {k: round(v["err_all"], 3) for k, v in R["drift"].items()}, flush=True)
