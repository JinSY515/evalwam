#!/usr/bin/env python3
"""Goal-aware supervision & goal appearance augmentation -- everything that needs NO retraining.

Inputs (already produced): ours avg 7500+8750 on valbench5s 120, original goal (mode 2) vs the same windows with the goal
frame repainted by telestyle (mode 3, multi/goalstyle_vb.sh). Per window:
  A  goal metrics (suite_metrics.csv) paired: exact McNemar for success, Wilcoxon + bootstrap CI for continuous
  B  per-variant breakdown (autumn / spring / photoreal / golden_hour / dawn)
  C  goal-state head readout (_pred_goalstate.npy, (1,20,3) = [remaining dist / 5 m, unit ego dir cos, sin]) vs the
     training target (train_hist_cond.py: dist = |goal_xy - pos_xy|, dir = goal vector in the ego frame), computed on
     (a) the GT trajectory and (b) the model's own predicted camera trajectory (pred_poses.npy; action_poses.npy = GT); key frames 1..19 (0 = conditioning)
       dir error = angle(pred dir, target dir) [deg], dist error = |5 * pred - dist| [m]
  D  which of the two read-outs goes with goal success (correlational, not causal): Spearman with final position /
     heading error, success rate by tertile of each read-out error, and under appearance change Δdir / Δdist vs Δsuccess
  E  existing trained ladder runs with / without goal aux (confounded: different steps, data, other losses)
  python goal_aug_analysis.py  -> results_syn/goal_validation/goal_aug.json
"""
import csv, json, os
import numpy as np
from scipy import stats

H = "/root/project/egowam/results_syn/goalprog_mode_real6_nomr_hg0_grad0.5"
AB = "/root/project/egowam/results_syn/abl"
RUNS = {"orig": f"{H}/ckpta7500_8750/valbench5s_all120", "style": f"{H}/ckpta7500_8750_goalstyle/valbench5s_all120"}
OUT = "/root/project/egowam/results_syn/goal_validation"; os.makedirs(OUT, exist_ok=True)
SC = 5.0
rng = np.random.default_rng(0)


def rows(d):
    return {int(r["sample"].split("_")[-1]): r for r in csv.DictReader(open(f"{d}/suite_metrics.csv"))}


def f(x):
    try:
        v = float(x); return v if np.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


def target(P, goal_xy):
    """P (20,6) CARLA [x,y,z,roll,pitch,yaw deg] at key frames -> (dist m, unit ego dir (20,2)), as train_hist_cond.py."""
    vec = goal_xy[None] - P[:, :2]; yaw = np.deg2rad(P[:, 5]); c, s = np.cos(yaw), np.sin(yaw)
    fwd, lat = vec[:, 0] * c + vec[:, 1] * s, -vec[:, 0] * s + vec[:, 1] * c
    dist = np.linalg.norm(vec, axis=1)
    return dist, np.stack([fwd, lat], 1) / (dist[:, None] + 1e-6)


def readout(d, i, m):
    gs = np.load(f"{d}/sample_{i:02d}_pred_goalstate.npy")[0]                   # (20,3)
    P = np.load(m["seq_dir"].rstrip("/") + "/poses.npy"); st = int(m["start_frame"])
    g = P[int(m["goal_frame"]), :2]; key = np.arange(0, 77, 4)[: gs.shape[0]]
    pv = gs[:, 1:3] / (np.linalg.norm(gs[:, 1:3], axis=1, keepdims=True) + 1e-9)
    o = {}
    for ref, PP in (("gt", P[st + key]), ("self", np.load(f"{d}/sample_{i:02d}_pred_poses.npy")[key])):
        dist, u = target(PP, g)
        k = np.arange(1, len(key))
        ang = np.degrees(np.arccos(np.clip((pv[k] * u[k]).sum(1), -1, 1)))
        de = np.abs(SC * gs[k, 0] - dist[k])
        o[ref] = {"dir": float(ang.mean()), "dist": float(de.mean()), "dir_early": float(ang[:4].mean()),
                  "dist_early": float(de[:4].mean()), "dist_rel": float((de / np.maximum(dist[k], 1.0)).mean())}
    o["d0"] = float(np.linalg.norm(g - P[st, :2]))
    return o


def boot(a, b, n=5000):
    dlt = b - a; idx = rng.integers(0, len(dlt), (n, len(dlt)))
    return [float(np.percentile(dlt[idx].mean(1), q)) for q in (2.5, 97.5)]


meta = {k: json.load(open(f"{d}/samples_meta.json")) for k, d in RUNS.items()}
R = {k: rows(d) for k, d in RUNS.items()}
N = sorted(set(R["orig"]) & set(R["style"]))
var = {i: meta["style"][str(i)].get("goal_variant", "?") for i in N}
out = {"n": len(N), "runs": RUNS}

# ---- A paired goal metrics
MET = [("goal_da3_success", "SR (DA3 궤적)", 1), ("goal_act_success", "SR (action 궤적)", 1),
       ("goal_da3_final_pos_path", "endpoint 오차/경로 (DA3)", -1), ("goal_act_final_pos_path", "endpoint 오차/경로 (action)", -1),
       ("goal_da3_final_head_deg", "도착 heading 오차° (DA3)", -1), ("goal_act_final_head_deg", "도착 heading 오차° (action)", -1)]
A = []
for key, lab, sgn in MET:
    a = np.array([f(R["orig"][i].get(key)) for i in N]); b = np.array([f(R["style"][i].get(key)) for i in N])
    ok = np.isfinite(a) & np.isfinite(b); a, b = a[ok], b[ok]
    r = {"key": key, "label": lab, "dir": sgn, "n": int(ok.sum()), "orig": float(a.mean()), "style": float(b.mean()),
         "delta": float((b - a).mean()), "ci": boot(a, b)}
    if key.endswith("success"):
        lost, won = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
        r.update(lost=lost, won=won, p=float(stats.binomtest(min(lost, won), lost + won, 0.5).pvalue) if lost + won else 1.0)
    else:
        r["p"] = float(stats.wilcoxon(a, b).pvalue) if np.any(a != b) else 1.0
        r["median_orig"], r["median_style"] = float(np.median(a)), float(np.median(b))
    A.append(r)
for k_, lab, src in (("coh_roterr", "VA: coherence RotErr°", "cam77"), ("act_roterr", "AF: action RotErr°", "cam77"),
                     ("act_ate_path", "AF: action ATE/경로", "cam77")):
    c = [json.load(open(f"{d}/cam77.json"))[k_]["mean"] for d in RUNS.values()]
    A.append({"key": k_, "label": lab, "dir": -1, "n": 120, "orig": c[0], "style": c[1], "delta": c[1] - c[0]})
out["A"] = A

# ---- B per variant
B = {}
for v in sorted(set(var.values())):
    ii = [i for i in N if var[i] == v]
    B[v] = {"n": len(ii), **{f"{k}_{r}": float(np.nanmean([f(R[r][i].get(k)) for i in ii]))
                             for k in ("goal_da3_success", "goal_da3_final_pos_path", "goal_da3_final_head_deg") for r in RUNS}}
out["B"] = B

# ---- C readout
RO = {r: {i: readout(d, i, meta[r][str(i)]) for i in N} for r, d in RUNS.items()}
C = {}
for ref in ("gt", "self"):
    for q in ("dir", "dist", "dir_early", "dist_early", "dist_rel"):
        a = np.array([RO["orig"][i][ref][q] for i in N]); b = np.array([RO["style"][i][ref][q] for i in N])
        C[f"{ref}_{q}"] = {"orig": float(a.mean()), "style": float(b.mean()), "median_orig": float(np.median(a)),
                           "median_style": float(np.median(b)), "rel_change": float(b.mean() / a.mean() - 1),
                           "ci": boot(a, b), "p": float(stats.wilcoxon(a, b).pvalue)}
out["C"] = C
out["d0_mean"] = float(np.mean([RO["orig"][i]["d0"] for i in N]))

# ---- D which read-out goes with goal success (orig run; and paired change under style)
D = {}
for r in RUNS:
    s = np.array([f(R[r][i]["goal_da3_success"]) for i in N]); pe = np.array([f(R[r][i]["goal_da3_final_pos_path"]) for i in N])
    he = np.array([f(R[r][i]["goal_da3_final_head_deg"]) for i in N])
    for q in ("dir", "dist", "dir_early", "dist_early"):         # early = key frames 1..4 (before paths diverge)
        e = np.array([RO[r][i]["gt"][q] for i in N]); t = np.quantile(e, [1 / 3, 2 / 3])
        bins = [s[e <= t[0]], s[(e > t[0]) & (e <= t[1])], s[e > t[1]]]
        D[f"{r}_{q}"] = {"rho_pos": float(stats.spearmanr(e, pe, nan_policy="omit")[0]),
                         "rho_head": float(stats.spearmanr(e, he, nan_policy="omit")[0]),
                         "sr_tertile": [float(np.nanmean(x)) for x in bins],
                         "mwu_p": float(stats.mannwhitneyu(e[s == 1], e[s == 0]).pvalue),
                         "err_succ": float(e[s == 1].mean()), "err_fail": float(e[s == 0].mean())}
s0 = np.array([f(R["orig"][i]["goal_da3_success"]) for i in N]); s1 = np.array([f(R["style"][i]["goal_da3_success"]) for i in N])
for q in ("dir", "dist"):
    dq = np.array([RO["style"][i]["gt"][q] - RO["orig"][i]["gt"][q] for i in N])
    D[f"delta_{q}"] = {"lost": float(dq[(s0 == 1) & (s1 == 0)].mean()), "kept": float(dq[(s0 == 1) & (s1 == 1)].mean()),
                       "rho_dsucc": float(stats.spearmanr(dq, s1 - s0)[0])}
out["D"] = D

# ---- E existing ladder runs (confounded)
E = []
for lab, d, note in (("abl2 · goal aux 없음 · ck 3000", f"{AB}/II_nogoalaux/ckpt3000", "raymap 토큰, depth·vcons·mode embed 없음, 옛 raymap convention"),
                     ("abl4 · goal aux 있음 · ck 6000", f"{AB}/IV_cont5k/ckpt6000", "raymap 토큰 + DPT depth, vcons·mode embed 없음"),
                     ("ours 2250 · goal aux 있음", f"{H}/ckpt2250", "vcons residual + mode embed, 같은 run의 2250 step"),
                     ("ours avg 7500+8750", f"{H}/ckpta7500_8750", "")):
    p = f"{d}/valbench5s_all120/suite_summary.json"
    if os.path.exists(p):
        s = json.load(open(p)); g = lambda k: s.get(k) if not isinstance(s.get(k), dict) else s[k].get("mean")
        E.append({"label": lab, "note": note, **{k: g(k) for k in ("goal_da3_success", "goal_act_success",
                                                                    "goal_da3_final_pos_path", "goal_da3_final_head_deg")}})
out["E"] = E
json.dump(out, open(f"{OUT}/goal_aug.json", "w"), indent=1)
print(json.dumps({k: out[k] for k in ("A", "B", "C", "D", "E")}, indent=1, default=float)[:6000])
