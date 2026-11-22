#!/usr/bin/env python3
"""Recon / BEV similarity of a rollout to GT, per clip of a merged rollout viz dir -> <roll_dir>/recon_sim.csv.

Compares the clouds the BEV/orbit panels draw (recon_orbit_rollout.build, same max_depth/stride):
  GT  = "GT d + GT pose"       (panel 1)
  OUR = "PRED d + PRED pose"   (panel 4, what the rollout actually gives)
Each cloud is centred on its own median, exactly like the BEV panels (--frame each), so the numbers follow what
the eye compares: no fitting of OUR onto GT. OUR's track is re-anchored on the GT start pose (recon_fig.build_fig):
pred_poses[0] is stored mirrored, which rotated the whole OUR cloud on the page BEV by the start-heading error.

  bev_f1     BEV occupancy F-score, 0.5 m cells (>=3 pts), 1 m tolerance              (higher = more similar)
  bev_iou    BEV occupancy IoU, 0.5 m cells                                            (higher)
  rec_f1     3D F-score at 0.5 m on 0.25 m voxel clouds (orbit view)                    (higher)
  area_err   |log(area OUR / area GT)|, BEV explored area as labelled on the panel      (lower)
  traj_err   mean distance of the two camera tracks in the panel frame / GT path length (lower)
  sim_rank   mean rank of bev_f1, rec_f1, area_err, traj_err                           (lower)

  python recon_sim.py <roll_dir> [--procs 16]
"""
import argparse, csv, json, math, os, sys
from multiprocessing import Pool
import numpy as np
from scipy.ndimage import binary_dilation
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from recon_fig import build_fig  # noqa: E402  (page clouds, pred track re-anchored on the GT start pose)

CELL, TOL, MINPTS, LIM = 0.5, 1.0, 3, 60.0
VOX, TOL3 = 0.25, 0.5


def occ(P, cell):
    q = np.floor(P[:, :2] / cell).astype(np.int64)
    k, n = np.unique(q, axis=0, return_counts=True)
    return {tuple(x) for x in k[n >= MINPTS]}


def grid(sets):
    allk = np.array([k for s in sets for k in s]); lo = allk.min(0); sh = allk.max(0) - lo + 1
    out = []
    for s in sets:
        g = np.zeros(sh, bool)
        if s:
            a = np.array(list(s)) - lo; g[a[:, 0], a[:, 1]] = True
        out.append(g)
    return out


def vox(P, v):
    return np.unique(np.floor(P / v).astype(np.int64), axis=0) * v + v / 2


def one(args):
    roll_dir, j = args
    if not os.path.exists(f"{roll_dir}/rollout_{j:02d}_pred_poses.npy"):
        return j, None, "no pred poses"
    (G, O), _, tracks = build_fig(roll_dir, j, 35.0, 6, 0.0, rgb=False)
    cg, co = np.median(G, 0), np.median(O, 0)
    G = G - cg; O = O - co
    G = G[np.abs(G[:, :2]).max(1) < LIM]; O = O[np.abs(O[:, :2]).max(1) < LIM]
    # BEV occupancy
    gg, go = grid([occ(G, CELL), occ(O, CELL)])
    r = int(round(TOL / CELL)); st = np.ones((2 * r + 1, 2 * r + 1), bool)
    prec = (go & binary_dilation(gg, st)).sum() / max(go.sum(), 1)
    rec = (gg & binary_dilation(go, st)).sum() / max(gg.sum(), 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    iou = (gg & go).sum() / max((gg | go).sum(), 1)
    # explored area as on the BEV label (cells of the panel size)
    EX = max(max(float(np.percentile(np.abs(np.concatenate([P[:, 0], P[:, 1]])), 98)) for P in (G, O)), 5.0)
    cA = 2 * EX / 400
    ag, ao = len(occ(G, cA)) * cA * cA, len(occ(O, cA)) * cA * cA
    aerr = abs(math.log(max(ao, 1e-6) / max(ag, 1e-6)))
    # 3D F-score
    VG, VO = vox(G, VOX), vox(O, VOX)
    dgo = cKDTree(VG).query(VO, distance_upper_bound=TOL3)[0]; dog = cKDTree(VO).query(VG, distance_upper_bound=TOL3)[0]
    p3, r3 = np.isfinite(dgo).mean(), np.isfinite(dog).mean()
    f3 = 2 * p3 * r3 / max(p3 + r3, 1e-9)
    # camera tracks in the panel frame
    tg, to = tracks[0] - cg, tracks[1] - co
    plen = float(np.linalg.norm(np.diff(tg, axis=0), axis=1).sum())
    terr = float(np.linalg.norm(tg - to, axis=1).mean()) / max(plen, 1e-6)
    return j, dict(bev_f1=f1, bev_iou=iou, rec_f1=f3, area_err=aerr, area_ratio=ao / max(ag, 1e-6),
                   traj_err=terr, path_m=plen), None


def ranks(vals, d):
    it = sorted(vals.items(), key=lambda t: -t[1] * d); out, i = {}, 0
    while i < len(it):
        k = i
        while k + 1 < len(it) and it[k + 1][1] == it[i][1]:
            k += 1
        for t in it[i:k + 1]:
            out[t[0]] = (i + k) / 2 + 1
        i = k + 1
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("roll_dir"); ap.add_argument("--procs", type=int, default=16)
    a = ap.parse_args()
    js = sorted(int(k) for k in json.load(open(f"{a.roll_dir}/rollout_meta.json")))
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    with Pool(a.procs) as pool:
        res = pool.map(one, [(a.roll_dir, j) for j in js])
    R = {j: m for j, m, e in res if m is not None}
    for j, m, e in res:
        if m is None:
            print(f"[skip] {j}: {e}")
    rk = {k: ranks({j: R[j][k] for j in R}, d) for k, d in (("bev_f1", 1), ("rec_f1", 1), ("area_err", -1), ("traj_err", -1))}
    for j in R:
        R[j]["sim_rank"] = sum(rk[k][j] for k in rk) / len(rk)
    cols = ["bev_f1", "bev_iou", "rec_f1", "area_err", "area_ratio", "traj_err", "path_m", "sim_rank"]
    with open(f"{a.roll_dir}/recon_sim.csv", "w") as fo:
        w = csv.writer(fo); w.writerow(["sample"] + cols)
        for j in sorted(R):
            w.writerow([f"rollout_{j:02d}"] + [f"{R[j][c]:.6f}" for c in cols])
    print(f"{a.roll_dir}/recon_sim.csv  ({len(R)} clips)")
    for i, j in enumerate(sorted(R, key=lambda j: R[j]["sim_rank"])):
        m = R[j]
        print(f"{i + 1:2d} j{j:02d}  rank {m['sim_rank']:.3f}  bevF1 {m['bev_f1']:.3f}  IoU {m['bev_iou']:.3f}  "
              f"3dF1 {m['rec_f1']:.3f}  area {m['area_ratio']:.3f}x  traj {m['traj_err']:.3f}")


if __name__ == "__main__":
    main()
