#!/usr/bin/env python3
"""Controlled perturbation test for the Video-Action Coherence metric (coh_* in eval_suite.py).

Video  = the GT video; its camera is recovered by DA3 (viz_da3_fix/sample_NN_da3_gt_poses.npy, written by
         viz_da3_pointcloud.py --da3_on_gt during the normal suite).
Action = the GT trajectory, reduced exactly as eval_suite.py reduces a model's action: 20 key frames
         (0,4,..,76) -> per-step ego actions (dx_fwd, dy_lat, dyaw) -> integrated from the GT start pose.
Then perturbations of growing magnitude are injected into the ACTION only and the coherence metric
(eval_suite.cam_block(est, DA3 planar)) is recomputed. Everything that the Sim3 alignment removes by design
(a constant rotation / offset / uniform speed scale) is tested separately as an invariance check.

  0   floor     GT action vs DA3(GT video)                  = measurement floor of the metric
  rot_drift     dyaw_k += theta / n       (heading drifts linearly to theta at the last frame; _out / _in = signed
                                           with / against the window's net turn: over- / under-rotation)
  rot_noise     dyaw_k += N(0, sigma)     (5 seeds)
  lat_drift     dy_k   += rho * dx_k      (constant sideways slip = a Sim3 of the positions: expected blind spot)
  lat_bend      dy_k   += 2 rho u dx_k    (slip ramping 0 -> 2 rho: the path bends)
  trans_noise   (dx, dy)_k += N(0, c * mean step)  (5 seeds)
  speed_prof    dx_k   *= 1 + beta * (2u - 1), u = 0..1 (accelerate: same path, wrong timing)
  speed_scale   dx_k, dy_k *= s           (uniform; Sim3 scale removes it -> should stay at the floor)
  swap          GT action of another window (fixed derangement) = fully mismatched reference
  model         (--model_dir) the model's own action vs DA3 of its own video = the reported number, for reference
Measurement bias (step 4): DA3(GT video) vs GT, 6-DoF over all 77 frames (eval_suite's da3_* block) and the
slope of DA3's cumulative yaw vs GT's (rotation magnitude bias).

Convention = cam77.py (the tables' coherence rows): all 77 frames, planar (x, y, yaw), eval_cam_robust.cam_metrics
(Sim3 on positions + separate rotation alignment). Positions are the GT-fitted DA3 poses (metres), so RotErr / ATE/Len /
RPE rot equal cam77's (scale-free); CamMC is in metres here, raw DA3 units in cam77.
Usage: coherence_perturb.py --results_dir <flat dir with samples_meta.json + viz_da3_fix> --out <dir> [--model_dir <dir>]
"""
import argparse, csv, importlib.util, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_s = importlib.util.spec_from_file_location("r", os.path.join(HERE, "eval_cam_robust.py"))
rb = importlib.util.module_from_spec(_s); _s.loader.exec_module(rb); es = rb.es

FAMILIES = {
    "rot_drift": [0, 1, 2, 3, 5, 10, 15, 20, 30, 45],            # deg at the last key frame (+ = left)
    "rot_drift_out": [0, 1, 2, 3, 5, 10, 15, 20, 30, 45],        # same, signed WITH the window's net turn (over-rotate)
    "rot_drift_in": [0, 1, 2, 3, 5, 10, 15, 20, 30, 45],         # same, signed AGAINST it (under-rotate, DA3's bias side)
    "rot_noise": [0, 0.25, 0.5, 1, 2, 4],                        # deg per step (std)
    "lat_drift": [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3],           # constant slip = a Sim3 of the positions (blind spot)
    "lat_bend": [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3],            # slip ramping 0 -> rho: the path bends sideways
    "trans_noise": [0, 0.05, 0.1, 0.2, 0.4, 0.8],                # per-step (dx, dy) noise, std = c * mean step length
    "speed_prof": [0, 0.1, 0.25, 0.5, 0.75],                     # speed ramp amplitude
    "speed_scale": [1, 0.5, 0.8, 1.25, 2],                       # invariance check
}
METRICS = ["coh_roterr", "coh_ate_path", "coh_cammc", "coh_rpe_rot", "coh_transerr_path"]


def perturb(act, fam, m, rng):
    a = act.copy(); n = len(a)
    sg = 1.0 if abs(act[:, 2].sum()) < 1.0 else float(np.sign(act[:, 2].sum()))
    if fam == "rot_drift":
        a[:, 2] += m / n
    elif fam == "rot_drift_out":
        a[:, 2] += sg * m / n
    elif fam == "rot_drift_in":
        a[:, 2] -= sg * m / n
    elif fam == "lat_bend":
        a[:, 1] += m * a[:, 0] * np.linspace(0, 1, n) * 2
    elif fam == "trans_noise":
        st = np.linalg.norm(act[:, :2], axis=1).mean(); a[:, :2] += rng.normal(0, m * st, (n, 2))
    elif fam == "rot_noise":
        a[:, 2] += rng.normal(0, m, n)
    elif fam == "lat_drift":
        a[:, 1] += m * a[:, 0]
    elif fam == "speed_prof":
        u = np.linspace(0, 1, n); a[:, 0] *= 1 + m * (2 * u - 1); a[:, 1] *= 1 + m * (2 * u - 1)
    elif fam == "speed_scale":
        a[:, :2] *= m
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--model_dir", default=None, help="flat model results dir on the same windows (pred_poses + DA3 of its video)")
    a = ap.parse_args()
    D = a.results_dir; os.makedirs(a.out, exist_ok=True)
    meta = json.load(open(os.path.join(D, "samples_meta.json")))
    S = []
    for k in sorted(meta, key=int):
        info = meta[k]; i = int(k)
        f = os.path.join(D, "viz_da3_fix", f"sample_{i:02d}_da3_gt_poses.npy")
        if not os.path.exists(f):
            continue
        P = np.load(os.path.join(info["seq_dir"], "poses.npy")); s0 = int(info["start_frame"])
        nf = int(info.get("num_frames", es.NF)); keys = np.arange(0, nf, es.VS)
        win = P[s0:s0 + nf]; gt = win[:, [0, 1, 5]]                      # all frames (cam77 convention)
        c2w = np.load(f)[:nf]
        d = dict(i=i, win=win, gt=gt, act=es.ego_actions(win), c2w=c2w, dk=es.da3_planar(c2w))
        if a.model_dir:                                                     # the model on the same window
            pp, dz = f"{a.model_dir}/sample_{i:02d}_pred_poses.npy", f"{a.model_dir}/viz_da3_fix/sample_{i:02d}_da3_poses.npy"
            if os.path.exists(pp) and os.path.exists(dz):
                d["m_act"], d["m_dk"] = es.ego_actions(np.load(pp)[:nf]), es.da3_planar(np.load(dz)[:nf])
        S.append(d)
    print(f"{len(S)} samples with DA3 on the GT video")

    def coh(act, s, dk=None):
        dk = s["dk"] if dk is None else dk
        est = es.integrate(act, *s["gt"][0])
        return rb.block(es.planar_mats(est), es.planar_mats(dk), dk, "coh")

    rows = []
    for s in S:
        base = {"sample": s["i"], "path_len": es.path_len(s["gt"]),
                "abs_yaw_total": float(np.abs(s["act"][:, 2]).sum()), "net_yaw": float(s["act"][:, 2].sum())}
        # measurement bias: DA3(GT video) vs GT, 6-DoF all frames; cumulative yaw of DA3 vs GT
        gt6 = np.stack([es.ecc.carla_pose_to_mat(*p) @ es.C_CV2CARLA for p in s["win"]])
        base.update(rb.block(s["c2w"], gt6, s["win"][:, :2], "da3"))
        dyaw = np.unwrap(np.radians(s["dk"][:, 2])); gyaw = np.unwrap(np.radians(s["gt"][:, 2]))
        base["da3_net_yaw"] = float(np.degrees(dyaw[-1] - dyaw[0])); base["gt_net_yaw"] = float(np.degrees(gyaw[-1] - gyaw[0]))
        for fam, mags in FAMILIES.items():
            for m in mags:
                seeds = range(a.seeds) if fam in ("rot_noise", "trans_noise") and m > 0 else [0]
                acc = {k: [] for k in METRICS}
                for sd in seeds:
                    r = coh(perturb(s["act"], fam, m, np.random.default_rng(1000 * s["i"] + sd)), s)
                    for k in METRICS:
                        acc[k].append(r[k])
                rows.append({**base, "family": fam, "mag": m, **{k: float(np.nanmean(v)) for k, v in acc.items()}})
    for s in S:
        if "m_act" in s:
            r = coh(s["m_act"], s, s["m_dk"])
            y = np.unwrap(np.radians(s["m_dk"][:, 2]))
            rows.append({"sample": s["i"], "path_len": es.path_len(s["gt"]), "family": "model", "mag": 1,
                         "m_act_net_yaw": float(s["m_act"][:, 2].sum()), "m_da3_net_yaw": float(np.degrees(y[-1] - y[0])),
                         "gt_net_yaw": float(s["act"][:, 2].sum()), **{k: v for k, v in r.items() if k in METRICS}})
    # fully mismatched: GT action of window j=(i+N/2) mod N paired with window i's DA3
    N = len(S)
    for n, s in enumerate(S):
        o = S[(n + N // 2) % N]
        rows.append({"sample": s["i"], "path_len": es.path_len(s["gt"]), "family": "swap", "mag": 1, **{k: v for k, v in coh(o["act"], s).items() if k in METRICS}})

    with open(os.path.join(a.out, "perturb_rows.csv"), "w", newline="") as f:
        keys = sorted({k for r in rows for k in r}); w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); [w.writerow(r) for r in rows]

    # summary: mean / median per (family, mag); monotonicity = Spearman(mag, err) per sample, averaged
    from scipy.stats import spearmanr, wilcoxon
    summ = {"n": N, "families": {}}
    floor = {k: np.array([r[k] for r in rows if r["family"] == "rot_drift" and r["mag"] == 0]) for k in METRICS}
    for fam in list(FAMILIES) + ["swap"] + (["model"] if any(r["family"] == "model" for r in rows) else []):
        F = {}
        mags = FAMILIES.get(fam, [1])
        for m in mags:
            R = [r for r in rows if r["family"] == fam and r["mag"] == m]
            F[str(m)] = {}
            for k in METRICS:
                v = np.array([r[k] for r in R], float)
                d = {"mean": float(np.nanmean(v)), "median": float(np.nanmedian(v))}
                if fam not in ("swap", "model") and m == mags[0]:
                    pass
                else:
                    ok = np.isfinite(v) & np.isfinite(floor[k])
                    try:
                        d["p_vs_floor"] = float(wilcoxon(v[ok], floor[k][ok], alternative="greater").pvalue)
                    except ValueError:
                        d["p_vs_floor"] = 1.0
                    d["frac_above_floor"] = float(np.mean(v[ok] > floor[k][ok] + 1e-9))
                F[str(m)][k] = d
        if fam in FAMILIES and fam != "speed_scale":
            for k in METRICS:
                rho = []
                for s in S:
                    R = sorted([r for r in rows if r["family"] == fam and r["sample"] == s["i"]], key=lambda r: r["mag"])
                    x, y = [r["mag"] for r in R], [r[k] for r in R]
                    if np.all(np.isfinite(y)) and np.ptp(y) > 0:
                        rho.append(spearmanr(x, y).correlation)
                F.setdefault("_spearman", {})[k] = {"mean": float(np.mean(rho)), "frac_eq_1": float(np.mean(np.isclose(rho, 1.0)))}
        summ["families"][fam] = F
    B = [r for r in rows if r["family"] == "rot_drift" and r["mag"] == 0]
    gy, dy = np.array([r["gt_net_yaw"] for r in B]), np.array([r["da3_net_yaw"] for r in B])
    sl = float(np.polyfit(gy, dy, 1)[0]) if np.ptp(gy) > 0 else float("nan")
    Mr = [r for r in rows if r["family"] == "model"]
    if Mr:
        g2 = np.array([r["gt_net_yaw"] for r in Mr]); big = np.abs(g2) > 20
        summ["model_net_yaw_slope_vs_gt"] = {"pred_action": float(np.polyfit(g2[big], [r["m_act_net_yaw"] for r, b in zip(Mr, big) if b], 1)[0]),
                                             "da3_generated": float(np.polyfit(g2[big], [r["m_da3_net_yaw"] for r, b in zip(Mr, big) if b], 1)[0]), "n": int(big.sum())}
    big = np.abs(gy) > 20
    summ["measurement"] = {"net_yaw_slope_turns_gt20": float(np.polyfit(gy[big], dy[big], 1)[0]), "n_turns_gt20": int(big.sum()),
                           "net_yaw_ratio_median_turns_gt20": float(np.median(dy[big] / gy[big])),
                "da3_vs_gt_6dof": {k: float(np.nanmean([r[k] for r in B])) for k in ["da3_roterr", "da3_transerr", "da3_ate_path", "da3_cammc", "da3_rpe_rot"]},
        "net_yaw_slope_da3_vs_gt": sl, "net_yaw_corr": float(np.corrcoef(gy, dy)[0, 1]),
        "net_yaw_abs_err_mean": float(np.mean(np.abs(dy - gy)))}
    # stratified by motion (path length / total |yaw| terciles): floor and response to a 10 deg drift / 5% slip
    st = {}
    for var in ("path_len", "abs_yaw_total"):
        x = np.array([r[var] for r in B]); q = np.quantile(x, [1 / 3, 2 / 3])
        for b, (lo, hi) in enumerate([(-np.inf, q[0]), (q[0], q[1]), (q[1], np.inf)]):
            ids = {r["sample"] for r in B if lo < r[var] <= hi}
            g = lambda fam, m, k: float(np.nanmean([r[k] for r in rows if r["family"] == fam and r["mag"] == m and r["sample"] in ids]))
            st[f"{var}_t{b + 1}"] = {"n": len(ids), "range": [float(min(x[[r['sample'] in ids for r in B]])), float(max(x[[r['sample'] in ids for r in B]]))],
                                     "floor_roterr": g("rot_drift", 0, "coh_roterr"), "drift10_roterr": g("rot_drift", 10, "coh_roterr"),
                                     "floor_ate_path": g("lat_drift", 0, "coh_ate_path"), "slip5_ate_path": g("lat_drift", 0.05, "coh_ate_path"),
                                     "floor_cammc": g("rot_drift", 0, "coh_cammc"), "drift10_cammc": g("rot_drift", 10, "coh_cammc")}
    summ["stratified"] = st
    json.dump(summ, open(os.path.join(a.out, "perturb_summary.json"), "w"), indent=1)
    print(json.dumps(summ["measurement"], indent=1))


if __name__ == "__main__":
    main()
