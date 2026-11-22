#!/usr/bin/env python3
"""Re-score a LookOut flat results dir against the CORRECTED (CARLA left-handed) GT poses.

lookout_val_cv/poses.npy is mirrored L/R (Aria world is right-handed, make_lookout_cv.py fixed the
camera axes with a proper rotation only): GT image shift vs dyaw corr +0.63 (CARLA -0.87). The
corrected tree is lookout_val_cv_lh (y, roll, yaw negated). Generation never read GT poses
(pred_raymap starts the camera latent from noise), so only GT-pose metrics change.

Builds <dir>_lh next to <dir>:
  * every file of <dir> symlinked (videos, pred poses/depth, FVD/MEt3R/goal-image/VQ/depth jsons+csvs),
  * samples_meta.json with seq_dir -> lookout_val_cv_lh,
  * viz_da3_fix/: DA3 poses re-fitted to the corrected GT with the SAME fit as viz_da3_pointcloud.py
    --da3_fit rot (orientation fit with free determinant, then scale/translation under it). The saved
    poses are a similarity transform of the raw DA3 output, so re-fitting them == fitting the raw ones.
then runs eval_suite.py and eval_cam_robust.py on it (suite_summary.json / cam_robust.json there).

  python3 lookout_lh_reeval.py <flat dir> [<flat dir> ...]
"""
import json, os, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OLD, NEW = "/root/project/carla/results/lookout_val_cv", "/root/project/carla/results/lookout_val_cv_lh"
import importlib.util
_s = importlib.util.spec_from_file_location("es", os.path.join(HERE, "eval_suite.py")); es = importlib.util.module_from_spec(_s)
_s.loader.exec_module(es); ecc = es.ecc
POSE_OUT = {"suite_metrics.csv", "suite_summary.json", "cam_robust.csv", "cam_robust.json", "samples_meta.json",
            "viz_da3_fix", "suite.log", "cam77.csv", "cam77.json"}


def refit(c2w, gt6):
    """viz_da3_pointcloud.py --da3_fit rot: R free det from orientations, then s,t for positions."""
    c2w = np.array(c2w, float).copy(); n = min(len(c2w), len(gt6))
    U, _, Vt = np.linalg.svd(np.einsum("nij,nkj->ik", gt6[:n, :3, :3], c2w[:n, :3, :3])); R = U @ Vt
    src, dst = c2w[:n, :3, 3], gt6[:n, :3, 3]
    rs = src @ R.T; mu_s, mu_d = rs.mean(0), dst.mean(0)
    s = float(((rs - mu_s) * (dst - mu_d)).sum() / max(((rs - mu_s) ** 2).sum(), 1e-12)); t = mu_d - s * mu_s
    c2w[:, :3, 3] = s * (c2w[:, :3, 3] @ R.T) + t
    c2w[:, :3, :3] = np.einsum("ij,njk->nik", R, c2w[:, :3, :3])
    return c2w, s


def one(D):
    D = D.rstrip("/"); L = D + "_lh"; os.makedirs(L + "/viz_da3_fix", exist_ok=True)
    for f in os.listdir(D):
        if f in POSE_OUT or os.path.lexists(f"{L}/{f}"):
            continue
        os.symlink(f"{D}/{f}", f"{L}/{f}")
    meta = json.load(open(f"{D}/samples_meta.json"))
    for k, m in meta.items():
        assert m["seq_dir"].startswith(OLD + "/"), m["seq_dir"]
        m["seq_dir"] = NEW + m["seq_dir"][len(OLD):]
    json.dump(meta, open(f"{L}/samples_meta.json", "w"), indent=1)
    V = f"{D}/viz_da3_fix"; nfit = 0
    for f in os.listdir(V):
        if not f.endswith("_da3_poses.npy"):
            if not os.path.lexists(f"{L}/viz_da3_fix/{f}"): os.symlink(f"{V}/{f}", f"{L}/viz_da3_fix/{f}")
            continue
        i = str(int(f.split("_")[1])); m = meta[i]
        P = np.load(os.path.join(m["seq_dir"], "poses.npy")); s0 = int(m["start_frame"]); nf = int(m.get("num_frames", 77))
        gt6 = np.stack([ecc.carla_pose_to_mat(*p) @ es.C_CV2CARLA for p in P[s0:s0 + nf]])
        c2w, s = refit(np.load(f"{V}/{f}"), gt6)
        np.save(f"{L}/viz_da3_fix/{f}", c2w); nfit += 1
    py = "/venv/da3/bin/python" if os.path.exists("/venv/da3/bin/python") else sys.executable
    subprocess.run([py, f"{HERE}/eval_suite.py", "--results_dir", L], stdout=open(f"{L}/suite.log", "w"),
                   stderr=subprocess.STDOUT, check=True)
    subprocess.run([sys.executable, f"{HERE}/eval_cam_robust.py", "--results_dir", L], stdout=subprocess.DEVNULL, check=True)
    print(f"{L}: {len(meta)} samples, {nfit} DA3 refits")


if __name__ == "__main__":
    for d in sys.argv[1:]:
        one(d)
