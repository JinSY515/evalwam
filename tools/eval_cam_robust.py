#!/usr/bin/env python3
"""Robust re-scoring of the camera metrics of eval_suite.py (CPU only, reads the saved poses; does not
touch suite_summary.json). Same samples / key frames / references as eval_suite.py, plus:

  * Sim3 scale of every fit is kept. A fit is DEGENERATE when s < 0.5 or s > 2: the free scale has
    collapsed (one trajectory almost static / uncorrelated) and TransErr-type numbers become
    meaningless (s -> 0 makes the estimate a point at the reference centroid).
  * act_se3: generated action vs GT with the scale FIXED to 1 (training data is metric, so a speed
    error is an error; Sim3 absorbs it). Rotation metrics are unchanged by the scale.
  * summary per metric: mean (all), median (all), mean over non-degenerate samples, #degenerate.

  python3 tools/eval_cam_robust.py --results_dir <dir>   -> <dir>/cam_robust.csv, <dir>/cam_robust.json
"""
import argparse, csv, importlib.util, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
def _load(name, f):
    s = importlib.util.spec_from_file_location(name, os.path.join(HERE, f)); m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m); return m
es = _load("es", "eval_suite.py"); ecc = es.ecc
LO, HI = 0.5, 2.0


def cam_metrics(est, ref, with_scale):        # == ecc.camera_metrics, scale optional
    R_e, C_e = ecc.poses_to_RC(est); R_g, C_g = ecc.poses_to_RC(ref); N = len(R_g)
    s, R_al, t_al = ecc.umeyama_sim3(C_e, C_g, with_scale)
    C_ea = (s * (R_al @ C_e.T)).T + t_al
    trans = np.linalg.norm(C_ea - C_g, axis=1)
    Wl, Wr = ecc.align_rotations(R_e, R_g); R_ga = np.stack([Wl @ R_g[i] @ Wr for i in range(N)])
    rot = np.array([ecc.geodesic_deg(R_ga[i], R_e[i]) for i in range(N)])
    cammc = np.array([np.linalg.norm(np.hstack([R_e[i], C_ea[i:i + 1].T]) - np.hstack([R_ga[i], C_g[i:i + 1].T]))
                      for i in range(N)])
    dRe = [R_e[i].T @ R_e[i + 1] for i in range(N - 1)]; dRg = [R_ga[i].T @ R_ga[i + 1] for i in range(N - 1)]
    rpe_rot = np.array([ecc.geodesic_deg(dRe[i], dRg[i]) for i in range(N - 1)])
    dCe, dCg = np.diff(C_ea, axis=0), np.diff(C_g, axis=0)
    rpe_tr = np.linalg.norm(dCe - dCg, axis=1)
    rpe_cam = np.array([np.linalg.norm(np.hstack([dRe[i] - dRg[i], (dCe[i] - dCg[i])[:, None]])) for i in range(N - 1)])
    return dict(roterr=rot.mean(), transerr=trans.mean(), ate=np.sqrt((trans ** 2).mean()), cammc=cammc.mean(),
                rpe_rot=rpe_rot.mean(), rpe_trans=rpe_tr.mean(), rpe_cammc=rpe_cam.mean(), scale=s)


def block(est, ref, ref_xy, prefix, with_scale=True):
    m = cam_metrics(est, ref, with_scale); L = es.path_len(ref_xy); n = len(est)
    if L > 1e-6:
        m.update(transerr_path=m["transerr"] / L, ate_path=m["ate"] / L, rpe_trans_path=m["rpe_trans"] * (n - 1) / L)
    m["path"] = L
    return {f"{prefix}_{k}": float(v) for k, v in m.items()}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--results_dir", required=True)
    ap.add_argument("--da3_subdir", default="viz_da3_fix"); a = ap.parse_args(); D = a.results_dir
    meta = json.load(open(os.path.join(D, "samples_meta.json"))); rows = []
    for k in sorted(meta, key=int):
        i, info = int(k), meta[k]
        P = np.load(os.path.join(info["seq_dir"], "poses.npy")); s0 = int(info["start_frame"])
        nf = int(info.get("num_frames", es.NF)); KEYS = np.arange(0, nf, es.VS)
        win = P[s0:s0 + nf]; gt = win[KEYS][:, [0, 1, 5]]
        row = {"sample": f"sample_{i:02d}"}
        pa, pp = (os.path.join(D, f"sample_{i:02d}_{x}.npy") for x in ("pred_actions", "pred_poses"))
        act = np.load(pa)[1:] if os.path.exists(pa) else es.ego_actions(np.load(pp)[KEYS]) if os.path.exists(pp) else None
        if act is not None:
            est = es.integrate(act, *gt[0])
            row.update(block(es.planar_mats(est), es.planar_mats(gt), gt, "act"))
            row.update(block(es.planar_mats(est), es.planar_mats(gt), gt, "act_se3", with_scale=False))
        dz = os.path.join(D, a.da3_subdir, f"sample_{i:02d}_da3_poses.npy")
        if os.path.exists(dz):
            c2w = np.load(dz); dk = es.da3_planar(c2w[KEYS])
            if act is not None:
                row.update(block(es.planar_mats(est), es.planar_mats(dk), dk, "coh"))
            gt6 = np.stack([ecc.carla_pose_to_mat(*p) @ es.C_CV2CARLA for p in win])
            row.update(block(c2w, gt6, win[:, :2], "da3"))
        rows.append(row)
    keys = sorted({k for r in rows for k in r})
    with open(os.path.join(D, "cam_robust.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); [w.writerow(r) for r in rows]
    summ = {"n": len(rows)}
    for p in ("act", "act_se3", "coh", "da3"):
        rs = [r for r in rows if f"{p}_roterr" in r]
        if not rs:
            continue
        if p == "act_se3":    # degenerate = the Sim3 fit of the same pair collapsed
            ok = [LO <= r["act_scale"] <= HI for r in rs]
        else:
            ok = [LO <= r[f"{p}_scale"] <= HI for r in rs]
        summ[f"{p}__n"], summ[f"{p}__deg"] = len(rs), int(len(rs) - sum(ok))
        for k in sorted({k[len(p) + 1:] for r in rs for k in r if k.startswith(p + "_")}):
            if p == "act" and k.startswith("se3_"):
                continue
            v = np.array([r.get(f"{p}_{k}", np.nan) for r in rs], float); g = v[np.array(ok)]
            summ[f"{p}_{k}"] = {"mean": float(np.nanmean(v)), "median": float(np.nanmedian(v)),
                                "mean_ok": float(np.nanmean(g)) if np.isfinite(g).any() else None}
    json.dump(summ, open(os.path.join(D, "cam_robust.json"), "w"), indent=1)
    print(D, {p: (summ.get(f"{p}__n"), summ.get(f"{p}__deg")) for p in ("act", "coh", "da3")})


if __name__ == "__main__":
    main()
