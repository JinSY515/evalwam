#!/usr/bin/env python3
"""Video-action coherence in the convention of the external comparison table:
  * all 77 frames (not the 20 latent key frames),
  * reference = DA3 on the generated RGB in RAW DA3 units: the saved viz_da3_fix poses were Sim3-fitted to
    GT with scale s (logged per sample by viz_da3_pointcloud.py: "(7-dof -> GT, s=...)"); positions / |s|
    undo it (the rotation/translation of that fit is absorbed by the Sim3 below),
  * RPE trans / path = mean per-step RPE / path (not the sum),
  * metrics: ecc.camera_metrics-equivalent (eval_cam_robust.cam_metrics), Sim3 est -> ref.
Writes <results_dir>/coh77.json (mean, median, n, n_bad_fit = samples whose DA3->GT fit had s <= 0).

  python3 coh77.py <ckpt dir with valbench5s_all120 [+ valbench5s_split/<d>/suite_4axis.log]>
"""
import glob, importlib.util, json, os, re, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_s = importlib.util.spec_from_file_location("r", os.path.join(HERE, "eval_cam_robust.py")); r = importlib.util.module_from_spec(_s)
_s.loader.exec_module(r); es = r.es
PAT = re.compile(r"sample (\d+)/da3: .*\(7-dof -> GT, s=(-?[\d.]+)\)")


def scales(log):
    out = {}
    for line in open(log, errors="ignore"):
        m = PAT.search(line)
        if m:
            out[int(m.group(1))] = float(m.group(2))     # last occurrence wins (re-runs)
    return out


def flat_scales(D, meta):
    """Flat dir (LookOut / DL3DV): s from its own suite_da3.log. A <dir>_lh copy (lookout_lh_reeval.py) re-fitted
    the saved poses once more (scale s2), so the raw-unit factor there is s1 * s2."""
    lh = D.endswith("_lh"); O = D[:-3] if lh else D
    S1 = scales(f"{O}/suite_da3.log")
    if not lh:
        return S1
    _l = importlib.util.spec_from_file_location("l", os.path.join(HERE, "lookout_lh_reeval.py"))
    l = importlib.util.module_from_spec(_l); _l.loader.exec_module(l)
    S = {}
    for k, info in meta.items():
        i = int(k); f = f"{O}/viz_da3_fix/sample_{i:02d}_da3_poses.npy"
        if i not in S1 or not os.path.exists(f):
            continue
        s0 = int(info["start_frame"]); nf = int(info.get("num_frames", 77))
        P = np.load(info["seq_dir"] + "/poses.npy")[s0:s0 + nf]
        gt6 = np.stack([es.ecc.carla_pose_to_mat(*p) @ es.C_CV2CARLA for p in P])
        S[i] = S1[i] * l.refit(np.load(f), gt6)[1]
    return S


def main(C, sub="valbench5s_all120"):
    D = f"{C}/{sub}" if sub else C; meta = json.load(open(f"{D}/samples_meta.json"))
    S = {}                                               # all120 index -> s
    own = f"{D}/suite_4axis.log"
    if not sub:
        S = flat_scales(D, meta)
    elif os.path.exists(own) and scales(own):
        S = scales(own)
    else:
        cache = {}
        for k in meta:
            tgt = os.path.realpath(f"{D}/sample_{int(k):02d}.mp4")     # .../valbench5s/<d>/sample_JJ.mp4
            d, j = tgt.split("/")[-2], int(re.search(r"sample_(\d+)", tgt).group(1))
            if d not in cache:
                cache[d] = scales(f"{C}/valbench5s_split/{d}/suite_4axis.log")
            if j in cache[d]:
                S[int(k)] = cache[d][j]
    rows, bad = [], 0
    for k, info in meta.items():
        i = int(k); nf = int(info.get("num_frames", 77)); K = np.arange(nf); s0 = int(info["start_frame"])
        pp, dz = f"{D}/sample_{i:02d}_pred_poses.npy", f"{D}/viz_da3_fix/sample_{i:02d}_da3_poses.npy"
        if not (os.path.exists(pp) and os.path.exists(dz) and i in S):
            continue
        if S[i] <= 0:
            bad += 1
        gt = np.load(info["seq_dir"] + "/poses.npy")[s0:s0 + nf][:, [0, 1, 5]]
        est = es.integrate(es.ego_actions(np.load(pp)[:nf]), *gt[0])
        c2w = np.load(dz)[:nf].copy(); c2w[:, :3, 3] /= abs(S[i])
        dk = es.da3_planar(c2w)
        m = r.cam_metrics(es.planar_mats(est), es.planar_mats(dk), True); L = es.path_len(dk)
        rows.append(dict(roterr=m["roterr"], transerr=m["transerr"], ate=m["ate"], cammc=m["cammc"],
                         transerr_path=m["transerr"] / L, ate_path=m["ate"] / L, rpe_rot=m["rpe_rot"],
                         rpe_trans=m["rpe_trans"], rpe_cammc=m["rpe_cammc"], rpe_trans_path=m["rpe_trans"] / L,
                         path=L, s=S[i]))
    out = {"n": len(rows), "n_bad_fit": bad, "n_meta": len(meta)}
    for key in rows[0]:
        v = np.array([x[key] for x in rows], float)
        out[key] = {"mean": float(v.mean()), "median": float(np.median(v))}
    json.dump(out, open(f"{D}/coh77.json", "w"), indent=1)
    import csv
    with open(f"{D}/coh77.csv", "w", newline="") as f:      # per sample (pooling across dirs)
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); [w.writerow(x) for x in rows]
    print(D.replace("/root/project/egowam/results_syn/",""), "n", len(rows), "bad fits", bad, "path(raw) med", round(out["path"]["median"], 3))


if __name__ == "__main__":
    flat = "--flat" in sys.argv                          # args are flat results dirs (LookOut/DL3DV), not ckpt dirs
    for c in [a for a in sys.argv[1:] if a != "--flat"]:
        main(c.rstrip("/"), None if flat else "valbench5s_all120")
