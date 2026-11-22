#!/usr/bin/env python3
"""Traversal-difficulty analysis: does ours' margin over the baselines grow on harder traversals? (no training)

Windows: synthetic = valbench5s 120, real = LookOut A+B+C 120 (A/B exclusion list, C = turn77 val 35, _lh = left/right
fixed GT), i.e. the sets of /ablation and /rebuttal. Models: ours avg 7500+8750, abl1 (compact action token, joint FM,
ck 3000) and the 10 baselines of /cmp (per-window metrics from their suite_*.json, matched to ours by (seq tail, first
frame) exactly as cmp.py; metric mapping = cmp.CMET, i.e. the paper-table convention).
Difficulty factors, from the GT window only (identical for every model):
  dist      start-goal straight-line distance [m]       |xy(goal) - xy(start)|
  rot       cumulative rotation [deg]                    sum |dyaw| over the 77 window frames
  sharp     sharpest turn [deg / s]                      max |yaw(t+16) - yaw(t)| over the window (16 frames ~ 1 s)
  unseen    unobserved extent                           1 - fraction of the LAST window frame's pixels (64x64, valid depth)
                                                         that are visible in the FIRST frame (warp_torch.gt_visibility:
                                                         in frame & depth-consistent); depth = CARLA GT / DA3 pseudo-GT
Per factor: tertiles over the bench's windows -> per-tertile mean per model; paired gap ours - baseline per tertile with
bootstrap 95% CI; trend = Spearman(factor, per-window gap); joint OLS gap ~ z(dist) + z(rot) (+ bootstrap CI) to
separate the two factors (they are correlated).
  python traversal_difficulty.py   -> results_syn/traversal_difficulty/difficulty.json
"""
import csv, json, os, sys
import numpy as np
import torch
from scipy import stats

sys.path.insert(0, "/root/project/evalwam/tools/ckpt_viewer")
sys.path.insert(0, "/root/project/evalwam/tools/vcons_cmp")
import cmp  # noqa: E402
from vcons_cmp import rescale_intrinsics, _carla_pose_to_mat  # noqa: E402
from utils.warp_torch import gt_visibility, relative_pose, NEAR, FAR  # noqa: E402

H = "/root/project/egowam/results_syn/goalprog_mode_real6_nomr_hg0_grad0.5/ckpta7500_8750"
I1 = "/root/project/egowam/results_syn/abl/I_jointfm/ckpt3000"
V2 = "/root/project/egowam/results_syn/goalprog_mode_real6_nomr_hg0_grad0.5/multi/v2"
OUT = "/root/project/egowam/results_syn/traversal_difficulty"; os.makedirs(OUT, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
EXL = json.load(open(f"{V2}/lookout_AB_exclude17.json"))
BENCH = {"syn": [("valbench", "valbench5s", "valbench5s_all120", None, 0)],
         "real": [("A", "lookout", "lookout_1c_all51_lh", set(EXL["A"]), 4), ("B", "lookout", "lookout_1c_s100_all51_lh", set(EXL["B"]), 4),
                  ("C", "lookout", "lookout_turn77val35_lh", None, 4)]}
DEPTH = {"valbench": lambda sd: f"{sd}/depth", "A": lambda sd: f"{sd}/depth_da3",
         "B": lambda sd: sd.replace("_lh/", "/") + "/depth_da3_b", "C": lambda sd: f"{sd}/depth"}
MET = ["goal_succ", "coh_rot", "cam_rot", "act_rot", "coh_ate", "met3r", "lpips", "psnr"]
CM = {k: (o, b, lab, d) for k, o, b, lab, d, *_ in cmp.CMET}
MODELS = [("ours", "ours"), ("abl1", "abl1 (compact action)")] + cmp.MODELS
rng = np.random.default_rng(0)


def num(x):
    try:
        v = float(x); return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def rows(d):
    o = {}
    for k in MET:
        fn = CM[k][0][0]
        if fn not in o:
            p = f"{d}/{fn}"
            o[fn] = {r["sample"]: r for r in csv.DictReader(open(p))} if os.path.exists(p) else {}
    return o


def unseen(sd, st, tag, off):
    """1 - visible fraction of the last window frame's pixels in the first frame (64x64 latent grid, like L_vcons)."""
    try:
        f0, f1 = st, st + 76
        D = torch.stack([torch.tensor(np.load(f"{DEPTH[tag](sd)}/{f:06d}.npy").astype(np.float32)) for f in (f0, f1)])[None].to(DEV)
    except FileNotFoundError:
        return None
    dg = torch.nn.functional.interpolate(D, size=(64, 64), mode="nearest")
    P = np.load(f"{sd}/poses.npy")
    c2w = torch.tensor(np.stack([_carla_pose_to_mat(*P[f + off]) for f in (f0, f1)]), device=DEV).float()[None]
    K = rescale_intrinsics(torch.tensor(np.load(f"{sd}/intrinsics.npy"), device=DEV).float()[None], (64, 64), D.shape[-2:])
    T_ = relative_pose(c2w, torch.tensor([1], device=DEV), torch.tensor([0], device=DEV))
    M = gt_visibility(dg[:, 1:2], dg[:, 0:1], T_, K)[0]
    valid = (dg[0, 1] > NEAR) & (dg[0, 1] < FAR)
    return float(1 - M[valid].float().mean()) if valid.any() else None


def factors(sd, st, goal, tag, off):
    P = np.load(f"{sd}/poses.npy")
    W = P[st + off: st + off + 77]
    yaw = np.unwrap(np.deg2rad(W[:, 5]))
    g = P[min(goal + off, len(P) - 1)]
    return {"dist": float(np.linalg.norm(g[:2] - W[0, :2])), "rot": float(np.degrees(np.abs(np.diff(yaw)).sum())),
            "sharp": float(np.degrees(np.abs(yaw[16:] - yaw[:-16]).max())), "unseen": unseen(sd, st, tag, off),
            "path": float(np.linalg.norm(np.diff(W[:, :2], axis=0), axis=1).sum())}


def collect(group):
    S = []
    for tag, folder, sub, excl, off in BENCH[group]:
        meta = json.load(open(f"{H}/{sub}/samples_meta.json"))
        ro, r1 = rows(f"{H}/{sub}"), rows(f"{I1}/{sub}")
        idx = cmp.Cmp(None).index(folder)
        for i in range(len(meta)):
            if excl and i in excl:
                continue
            m = meta[str(i)]; sd, st = m["seq_dir"].rstrip("/"), int(m["start_frame"])
            hit = idx.get((cmp._tail(sd), st))
            f = factors(sd, st, int(m["goal_frame"]), tag, off)
            v = {}
            for k in MET:
                (fn, col), (bf, bk) = CM[k][0], CM[k][1]
                v[k] = [num(ro.get(fn, {}).get(f"sample_{i:02d}", {}).get(col)), num(r1.get(fn, {}).get(f"sample_{i:02d}", {}).get(col))]
                for mid, _ in cmp.MODELS:
                    p = f"{cmp.CMP_ROOT}/{mid}/{folder}/{hit[0]}/{bf}" if hit else None
                    v[k].append(cmp._val(json.load(open(p)), bk) if p and os.path.exists(p) else None)
            S.append({"set": tag, "i": i, "seq": sd, "matched": bool(hit), **f, "m": v})
    return S


def boot_ci(x, n=4000):
    x = np.asarray(x, float)
    if len(x) < 3:
        return [None, None]
    b = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def analyse(S):
    out = {"n": len(S), "n_matched": sum(s["matched"] for s in S), "models": [l for _, l in MODELS], "factors": {}}
    for fac in ("dist", "rot", "sharp", "unseen"):
        x = np.array([s[fac] if s[fac] is not None else np.nan for s in S])
        ok = np.isfinite(x); t = np.quantile(x[ok], [1 / 3, 2 / 3])
        bins = [ok & (x <= t[0]), ok & (x > t[0]) & (x <= t[1]), ok & (x > t[1])]
        F = {"cut": [float(t[0]), float(t[1])], "range": [[float(x[b].min()), float(x[b].max())] for b in bins],
             "n": [int(b.sum()) for b in bins], "mean": {}, "gap": {}, "trend": {}}
        for k in MET:
            M = np.array([[np.nan if v is None else v for v in s["m"][k]] for s in S])        # (N, n_models)
            F["mean"][k] = [[None if not np.isfinite(M[b, j]).any() else float(np.nanmean(M[b, j])) for b in bins] for j in range(M.shape[1])]
            F["gap"][k], F["trend"][k] = {}, {}
            for j in range(1, M.shape[1]):
                g = M[:, 0] - M[:, j]                                                         # ours - baseline, paired
                if np.isfinite(g).sum() < 10:
                    continue
                F["gap"][k][j] = [{"mean": float(np.nanmean(g[b])), "ci": boot_ci(g[b][np.isfinite(g[b])]),
                                   "n": int(np.isfinite(g[b]).sum())} for b in bins]
                okg = ok & np.isfinite(g)
                r, p = stats.spearmanr(x[okg], g[okg])
                F["trend"][k][j] = {"rho": float(r), "p": float(p)}
        out["factors"][fac] = F
    # joint: gap ~ z(dist) + z(rot)   (do the two factors carry independent effects?)
    d = np.array([s["dist"] for s in S]); r = np.array([s["rot"] for s in S])
    Z = np.c_[np.ones(len(S)), (d - d.mean()) / d.std(), (r - r.mean()) / r.std()]
    out["corr_dist_rot"] = float(stats.spearmanr(d, r)[0])
    out["joint"] = {}
    for k in ("goal_succ", "coh_rot", "cam_rot", "act_rot"):
        M = np.array([[np.nan if v is None else v for v in s["m"][k]] for s in S]); out["joint"][k] = {}
        for j in range(1, M.shape[1]):
            g = M[:, 0] - M[:, j]; okg = np.isfinite(g)
            if okg.sum() < 20:
                continue
            beta = np.linalg.lstsq(Z[okg], g[okg], rcond=None)[0]
            bs = []
            for _ in range(2000):
                ii = rng.integers(0, okg.sum(), okg.sum())
                bs.append(np.linalg.lstsq(Z[okg][ii], g[okg][ii], rcond=None)[0])
            bs = np.array(bs)
            out["joint"][k][j] = {"b_dist": float(beta[1]), "b_rot": float(beta[2]), "ci_dist": [float(np.percentile(bs[:, 1], q)) for q in (2.5, 97.5)],
                                  "ci_rot": [float(np.percentile(bs[:, 2], q)) for q in (2.5, 97.5)], "n": int(okg.sum())}
    return out


R = {}
for grp in ("syn", "real"):
    S = collect(grp)
    json.dump(S, open(f"{OUT}/windows_{grp}.json", "w"))
    R[grp] = analyse(S)
    print(grp, R[grp]["n"], "matched", R[grp]["n_matched"], {f: R[grp]["factors"][f]["range"] for f in R[grp]["factors"]}, flush=True)
R["meta"] = {"metric_labels": {k: CM[k][2] for k in MET}, "metric_dir": {k: CM[k][3] for k in MET}, "models": [l for _, l in MODELS]}
json.dump(R, open(f"{OUT}/difficulty.json", "w"), indent=1)
