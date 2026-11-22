#!/usr/bin/env python3
"""Extended metric suite (pose / depth part) for one valbench5s results dir (flat, 120 samples).

Writes <results_dir>/suite_metrics.csv (per sample) and suite_summary.json (means). Set-level
video metrics (FVD, MUSIQ) come from eval_fvd_musiq.py; PSNR/SSIM/LPIPS from
video_quality_metrics.csv; depth coherence from depth_vs_da3_metrics.csv (C_* = predicted depth
vs DA3 on the generated RGB).

COMMON REPRESENTATION. Every run is reduced to the same thing before scoring, so a joint-FM
action head (run I) and a raymap model are comparable:
  key frames k = 0,4,..,76 (one per latent frame, 20 poses) ; planar ego state (x, y, yaw).
  generated action a_1..a_19 = per-latent ego-motion (dx_fwd, dy_lat, dyaw), from
      _pred_actions.npy rows 1.. (action head)          or
      key-frame deltas of _pred_poses.npy (invert_raymap s3 on the predicted raymap).
  a_0 is GIVEN to the model, so the generated trajectory is integrated from the GT start pose.
  DA3 camera of the generated RGB = viz_da3_fix/*_da3_poses.npy (already sim3-aligned to GT by
  viz_da3_pointcloud.py, so its "DA3 units" are metres), yaw from the camera forward (+z) axis.

METRICS
  faithfulness  generated action vs GT     camera_metrics (Sim3 + rotation alignment), plus
                ATE = RMSE of the Sim3-aligned position error; X / path = X / GT path length;
                RPE / path = sum of per-step RPE / GT path length (= mean RPE / mean step).
  coherence     generated action vs DA3(generated RGB)  -- same metrics, DA3 as reference.
  goal          start-anchored, NO alignment: final pos err / path, final heading err,
                arrival = first key frame within 10% of path of the GT final position / 19
                (NaN if never; arrival_rate reported), success = pos/path < 0.1 and heading < 10 deg.
                Done twice: generated action vs GT, DA3 camera vs GT.
  extra         DA3 camera (6-DoF, all 77 frames) vs GT, Sim3.
"""
import argparse, csv, importlib.util, json, math, os, re
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_s = importlib.util.spec_from_file_location("ecc", os.path.join(HERE, "eval_camera_control.py"))
ecc = importlib.util.module_from_spec(_s); _s.loader.exec_module(ecc)
VS, NF = 4, 77
KEYS = np.arange(0, NF, VS)              # 20 key frames
C_CV2CARLA = np.eye(4); C_CV2CARLA[:3, :3] = [[0, 0, 1], [1, 0, 0], [0, -1, 0]]


def ego_actions(p):                       # (N,>=6 carla) -> (N-1,3)  == train_hist_cond.compute_ego_actions
    p = np.asarray(p, float)
    dxw, dyw = p[1:, 0] - p[:-1, 0], p[1:, 1] - p[:-1, 1]
    yaw = np.deg2rad(p[:-1, 5])
    dyaw = (p[1:, 5] - p[:-1, 5] + 180) % 360 - 180
    return np.stack([dxw * np.cos(yaw) + dyw * np.sin(yaw), -dxw * np.sin(yaw) + dyw * np.cos(yaw), dyaw], -1)


def integrate(a, x0, y0, yaw0):           # -> (len(a)+1, 3) planar (x, y, yaw_deg)
    out, x, y, yaw = [(x0, y0, yaw0)], x0, y0, float(yaw0)
    for dx, dy, dyaw in a:
        r = math.radians(yaw)
        x += dx * math.cos(r) - dy * math.sin(r); y += dx * math.sin(r) + dy * math.cos(r)
        yaw += dyaw; out.append((x, y, yaw))
    return np.array(out)


def planar_mats(xyyaw):
    return np.stack([ecc.carla_pose_to_mat(x, y, 0.0, 0.0, 0.0, yaw) for x, y, yaw in xyyaw])


def da3_planar(c2w):
    f = c2w[:, :3, 2]                     # opencv camera forward = +z
    return np.stack([c2w[:, 0, 3], c2w[:, 1, 3], np.degrees(np.arctan2(f[:, 1], f[:, 0]))], -1)


def path_len(xy):
    return float(np.linalg.norm(np.diff(np.asarray(xy)[:, :2], axis=0), axis=1).sum())


def cam_block(est_mats, ref_mats, ref_xy, prefix):
    m = ecc.camera_metrics(est_mats, ref_mats)
    Ce, Cg = est_mats[:, :3, 3], ref_mats[:, :3, 3]
    s, R, t = ecc.umeyama_sim3(Ce, Cg, True)
    err = np.linalg.norm((s * (R @ Ce.T)).T + t - Cg, axis=1)
    L = path_len(ref_xy)
    n = len(est_mats)
    return {f"{prefix}_roterr": m["roterr"], f"{prefix}_transerr": m["transerr"],
            f"{prefix}_ate": float(np.sqrt((err ** 2).mean())), f"{prefix}_cammc": m["cammc"],
            f"{prefix}_transerr_path": m["transerr"] / L if L > 1e-6 else np.nan,
            f"{prefix}_ate_path": float(np.sqrt((err ** 2).mean())) / L if L > 1e-6 else np.nan,
            f"{prefix}_rpe_rot": m["rpe_rot_deg"], f"{prefix}_rpe_trans": m["rpe_trans"],
            f"{prefix}_rpe_cammc": m["rpe_camera"],
            f"{prefix}_rpe_trans_path": m["rpe_trans"] * (n - 1) / L if L > 1e-6 else np.nan,
            f"{prefix}_scale": m["scale"]}


def goal_block(est, gt, prefix):          # est/gt (20,3) planar, same world frame, start-anchored
    L = path_len(gt)
    goal = gt[-1, :2]
    pos = float(np.linalg.norm(est[-1, :2] - goal)) / L if L > 1e-6 else np.nan
    head = abs((est[-1, 2] - gt[-1, 2] + 180) % 360 - 180)
    d = np.linalg.norm(est[:, :2] - goal, axis=1)
    hit = np.where(d < 0.1 * L)[0]
    arr = float(hit[0] / (len(gt) - 1)) if len(hit) else np.nan
    return {f"{prefix}_final_pos_path": pos, f"{prefix}_final_head_deg": float(head),
            f"{prefix}_arrival_ratio": arr, f"{prefix}_arrived": float(len(hit) > 0),
            f"{prefix}_success": float(pos < 0.1 and head < 10.0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--da3_subdir", default="viz_da3_fix")
    a = ap.parse_args()
    D = a.results_dir
    meta = json.load(open(os.path.join(D, "samples_meta.json")))
    depth = {}
    dp = os.path.join(D, "depth_vs_da3_metrics.csv")
    if os.path.exists(dp):
        depth = {r["sample"]: r for r in csv.DictReader(open(dp))}
    rows = []
    for k in sorted(meta, key=int):
        i, info = int(k), meta[k]
        P = np.load(os.path.join(info["seq_dir"], "poses.npy")); s0 = int(info["start_frame"])
        nf = int(info.get("num_frames", NF))           # 77 valbench, 153 LookOut rollouts
        KEYS = np.arange(0, nf, VS)
        win = P[s0:s0 + nf]
        gt = win[KEYS][:, [0, 1, 5]]
        row = {"sample": f"sample_{i:02d}"}
        pa = os.path.join(D, f"sample_{i:02d}_pred_actions.npy")
        pp = os.path.join(D, f"sample_{i:02d}_pred_poses.npy")
        if os.path.exists(pa):
            act, row["action_src"] = np.load(pa)[1:], "action_head"
        elif os.path.exists(pp):
            act, row["action_src"] = ego_actions(np.load(pp)[KEYS]), "raymap_inverted"
        else:
            act = None
        if act is not None:
            est = integrate(act, *gt[0])
            row.update(cam_block(planar_mats(est), planar_mats(gt), gt, "act"))
            row.update(goal_block(est, gt, "goal_act"))
        dz = os.path.join(D, a.da3_subdir, f"sample_{i:02d}_da3_poses.npy")
        if os.path.exists(dz):
            c2w = np.load(dz)
            dk = da3_planar(c2w[KEYS])
            row.update(goal_block(dk, gt, "goal_da3"))
            if act is not None:
                row.update(cam_block(planar_mats(est), planar_mats(dk), dk, "coh"))
            # saved DA3 c2w are OpenCV-camera axes (+z fwd, det -1 after the GT fit); express GT
            # the same way: carla pose @ C, C = opencv->carla(fwd,right,up) (det -1, CARLA is
            # left-handed). Checked: roterr 9.25 vs camera_control_metrics.csv 9.20 (20 samples).
            gt6 = np.stack([ecc.carla_pose_to_mat(*p) @ C_CV2CARLA for p in win])
            row.update(cam_block(c2w, gt6, win[:, :2], "da3"))
        r = depth.get(f"sample_{i:02d}")
        if r:
            for c in ["C_absrel", "C_delta1", "C_delta15"]:
                if r.get(c) not in (None, ""):
                    row["depth_" + c] = float(r[c])
        rows.append(row)
    keys = sorted({k for r in rows for k in r})
    with open(os.path.join(D, "suite_metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); [w.writerow(r) for r in rows]
    summ = {"n": len(rows)}
    for k in keys:
        if k in ("sample", "action_src"):
            continue
        v = np.array([r[k] for r in rows if k in r], float)
        summ[k] = float(np.nanmean(v)) if len(v) and np.isfinite(v).any() else None
        summ[k + "__n"] = int(np.isfinite(v).sum())
    for p in ("goal_act", "goal_da3"):        # arrival ratio only over samples that arrived
        if p + "_arrived" in summ:
            summ[p + "_arrival_rate"] = summ.pop(p + "_arrived")
    json.dump(summ, open(os.path.join(D, "suite_summary.json"), "w"), indent=1)
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in summ.items()
                      if not k.endswith("__n")}, indent=0))


if __name__ == "__main__":
    main()
