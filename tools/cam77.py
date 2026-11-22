#!/usr/bin/env python3
"""Camera metrics in the convention of the external comparison table, for one flat results dir.

  act : generated action (raymap-inverted poses, or run-I action head) vs GT          [metres]
  coh : generated action vs DA3 on the generated RGB, DA3 in RAW DA3 units            [DA3 units]
  * all 77 frames (not the 20 latent key frames); Sim3 est -> ref (ecc.camera_metrics equivalent)
  * RPE trans / path = MEAN per-step RPE / path (eval_suite.py uses the sum)
  * raw DA3 units: the saved viz_da3_fix poses were Sim3-fitted to GT with scale s; positions / |s| undo it.
    s comes from the viz_da3_pointcloud.py log line "sample N/da3: ... (7-dof -> GT, s=...)". The FIRST such
    line per sample is the generated-video pass; suite_4axis.log later runs DA3 on the GT video (depth floor)
    and logs it in the same format (its s is identical across runs) -- never take the last one.
    A <dir>_lh LookOut copy re-fitted the poses once more (lookout_lh_reeval.py): factor s1 * s2.
  * run I (action head, one action per latent frame) has no per-frame poses: each latent action is split
    into 4 equal sub-steps (flagged "act_src": "head_upsampled").
Writes <dir>/cam77.json (+ cam77.csv per sample).

  python3 cam77.py <results dir> [...]
"""
import csv, glob, importlib.util, json, os, re, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_s = importlib.util.spec_from_file_location("r", os.path.join(HERE, "eval_cam_robust.py")); r = importlib.util.module_from_spec(_s)
_s.loader.exec_module(r); es = r.es
PAT = re.compile(r"sample (\d+)/da3: .*\(7-dof -> GT, s=(-?[\d.]+)\)")
KEYS = ["roterr", "transerr", "ate", "cammc", "transerr_path", "ate_path", "rpe_rot", "rpe_trans", "rpe_cammc",
        "rpe_trans_path"]


def first_scales(log):
    out = {}
    for line in open(log, errors="ignore"):
        m = PAT.search(line)
        if m and int(m.group(1)) not in out:
            out[int(m.group(1))] = float(m.group(2))
    return out


def find_scales(D, meta):
    """index -> s of the generated-video DA3 fit. Tries the dir's own logs, the eval4axis logs next to it, then
    per-difficulty split dirs (valbench5s_all120 is a symlink merge of valbench5s/<d>/)."""
    base = D[:-3] if D.endswith("_lh") else D
    # cam77_scales.log = hand-recovered generated-pass lines (first choice when present)
    cands = [f"{base}/cam77_scales.log", f"{base}/suite_da3.log", f"{base}/suite_4axis.log", f"{os.path.dirname(base)}/eval4axis.log",
             f"{base}_eval4axis.log"]
    for c in cands:
        if os.path.exists(c):
            S = first_scales(c)
            if len(S) >= len(meta):
                return S, os.path.relpath(c, os.path.dirname(base))
    S, cache = {}, {}
    for k in meta:
        tgt = os.path.realpath(f"{base}/sample_{int(k):02d}.mp4")
        d, j = tgt.split("/")[-2], int(re.search(r"sample_(\d+)", os.path.basename(tgt)).group(1))
        if d not in cache:
            cache[d] = {}
            for c in (f"{os.path.dirname(base)}/valbench5s_split/{d}/cam77_scales.log",
                      f"{os.path.dirname(base)}/valbench5s_split/{d}/suite_da3.log",
                      f"{os.path.dirname(base)}/valbench5s_split/{d}/suite_4axis.log"):
                if os.path.exists(c) and first_scales(c):
                    cache[d] = first_scales(c); break
        if j in cache[d]:
            S[int(k)] = cache[d][j]
    return S, "valbench5s_split/*"


def lh_factor(D, meta, S):
    if not D.endswith("_lh"):
        return S
    _l = importlib.util.spec_from_file_location("l", os.path.join(HERE, "lookout_lh_reeval.py"))
    l = importlib.util.module_from_spec(_l); _l.loader.exec_module(l)
    out = {}
    for k, info in meta.items():
        i = int(k); f = f"{D[:-3]}/viz_da3_fix/sample_{i:02d}_da3_poses.npy"
        if i in S and os.path.exists(f):
            s0 = int(info["start_frame"]); nf = int(info.get("num_frames", 77))
            P = np.load(info["seq_dir"] + "/poses.npy")[s0:s0 + nf]
            gt6 = np.stack([es.ecc.carla_pose_to_mat(*p) @ es.C_CV2CARLA for p in P])
            out[i] = S[i] * l.refit(np.load(f), gt6)[1]
    return out


def block(est, ref, pre):
    m = r.cam_metrics(es.planar_mats(est), es.planar_mats(ref), True); L = es.path_len(ref)
    d = dict(roterr=m["roterr"], transerr=m["transerr"], ate=m["ate"], cammc=m["cammc"], transerr_path=m["transerr"] / L,
             ate_path=m["ate"] / L, rpe_rot=m["rpe_rot"], rpe_trans=m["rpe_trans"], rpe_cammc=m["rpe_cammc"],
             rpe_trans_path=m["rpe_trans"] / L, path=L, scale=m["scale"])
    return {f"{pre}_{k}": float(v) for k, v in d.items()}


def main(D):
    D = D.rstrip("/"); meta = json.load(open(f"{D}/samples_meta.json"))
    S, src = find_scales(D, meta); S = lh_factor(D, meta, S)
    rows, src_act = [], set()
    for k, info in meta.items():
        i = int(k); s0 = int(info["start_frame"]); nf = int(info.get("num_frames", 77))
        gt = np.load(info["seq_dir"] + "/poses.npy")[s0:s0 + nf][:, [0, 1, 5]]
        pp, pa = f"{D}/sample_{i:02d}_pred_poses.npy", f"{D}/sample_{i:02d}_pred_actions.npy"
        if os.path.exists(pp):
            act = es.ego_actions(np.load(pp)[:nf]); src_act.add("raymap_inverted")
        elif os.path.exists(pa):
            act = np.repeat(np.load(pa)[1:] / 4.0, 4, axis=0)[:nf - 1]; src_act.add("head_upsampled")
        else:
            continue
        est = es.integrate(act, *gt[0])
        row = {"sample": f"sample_{i:02d}"}; row.update(block(est, gt, "act"))
        dz = f"{D}/viz_da3_fix/sample_{i:02d}_da3_poses.npy"
        if os.path.exists(dz) and i in S and abs(S[i]) < 1e-6:
            row["da3_s"] = S[i]                       # collapsed DA3 fit (logged s=0): no usable reference
        elif os.path.exists(dz) and i in S:
            c2w = np.load(dz)[:nf].copy(); c2w[:, :3, 3] /= abs(S[i])
            row.update(block(est, es.da3_planar(c2w), "coh")); row["da3_s"] = S[i]
        rows.append(row)
    out = {"n": len(rows), "act_src": sorted(src_act), "scale_src": src, "n_scale": len(S),
           "n_bad_fit": int(sum(1 for x in rows if x.get("da3_s", 1) <= 0))}
    for p in ("act", "coh"):
        for k in KEYS + ["path"]:
            v = np.array([x[f"{p}_{k}"] for x in rows if f"{p}_{k}" in x], float)
            if len(v):
                out[f"{p}_{k}"] = {"mean": float(v.mean()), "median": float(np.median(v)), "n": int(len(v))}
    json.dump(out, open(f"{D}/cam77.json", "w"), indent=1)
    keys = sorted({k for x in rows for k in x})
    with open(f"{D}/cam77.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); [w.writerow(x) for x in rows]
    print(f"{D.replace('/root/project/egowam/results_syn/', '')}: n {len(rows)} act {sorted(src_act)} "
          f"scales {len(S)} ({src}) bad {out['n_bad_fit']} | coh path med {out.get('coh_path', {}).get('median', float('nan')):.3f}")


if __name__ == "__main__":
    for d in sys.argv[1:]:
        main(d)
