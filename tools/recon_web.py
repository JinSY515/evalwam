#!/usr/bin/env python3
"""Dense colored point clouds of one valbench sample for an interactive three.js viewer (recon_web.html).

Three clouds over the 20 latent key frames (frame 4i of the 77-frame window):
  PRED : the model's predicted depth (SI-disparity, one affine per sample fitted to GT disparity -- the best
         case, as in tools/recon_orbit.py) at the raymap-inverted predicted poses; colors = generated video.
  DA3  : DA3 run on the generated video (viz_da3_fix: depth + c2w already Sim3-fitted to GT); colors = generated.
  GT   : CARLA GT depth at GT poses; colors = GT frames.
World = CARLA (x fwd, y right, z up, left-handed); written as three.js (X=x, Y=z, Z=y) which flips handedness
back to right-handed. Points are voxel-deduplicated, centred on the GT cloud, stored as int16 centimetres.

  python3 recon_web.py --results_dir <valbench5s_all120> --sample 41 --template recon_web.html --out page.html
"""
import argparse, base64, json, math, os
import cv2, numpy as np

_C3 = np.array([[0, 0, 1], [1, 0, 0], [0, -1, 0]], float)      # pinhole (x right, y down, z fwd) -> CARLA cam
NEAR = 0.5


def c2w(v):
    x, y, z, rr, pp, yy = [float(t) for t in v[:6]]
    r, p, w = map(math.radians, (rr, pp, yy))
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(w), math.sin(w)
    M = np.eye(4)
    M[0, :3] = [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr]
    M[1, :3] = [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr]
    M[2, :3] = [-sp, cp * sr, cp * cr]
    M[:3, 3] = (x, y, z); return M


def dec(a, o):
    H, W = a.shape[:2]
    yi = ((np.arange(o) + .5) * H / o).astype(int).clip(0, H - 1); xi = ((np.arange(o) + .5) * W / o).astype(int).clip(0, W - 1)
    return a[yi][:, xi]


def pix(h, w, fx, fy, cx, cy, d):
    vv, uu = np.mgrid[0:h, 0:w]
    return np.stack([(uu + .5 - cx) / fx * d, (vv + .5 - cy) / fy * d, d], -1).reshape(-1, 3)


def frames(mp4):
    cap, out = cv2.VideoCapture(mp4), []
    while True:
        ok, f = cap.read()
        if not ok:
            return out
        out.append(f[..., ::-1])            # RGB


def voxel(P, C, Z, vs):
    k = np.floor(P / vs).astype(np.int64)
    _, idx = np.unique(k, axis=0, return_index=True)
    return P[idx], C[idx], Z[idx]


def build(D, i, res=256, max_depth=40.0, voxel_m=0.07):
    """-> (meta dict, {name: (P carla xyz, C rgb uint8, Z cam depth)} sorted by depth, centre)."""
    class _A: pass
    a = _A(); a.res, a.max_depth, a.voxel = res, max_depth, voxel_m
    return _build(D, i, a)


def _build(D, i, a):
    m = json.load(open(f"{D}/samples_meta.json"))[str(i)]; sd = m["seq_dir"]; st = int(m["start_frame"])
    R = a.res; key = [4 * t for t in range(20)]
    K = np.load(f"{sd}/intrinsics.npy").astype(float); sc = R / 512.0
    fx, fy, cx, cy = K[0] * sc, K[1] * sc, K[2] * sc, K[3] * sc
    gp = np.load(f"{sd}/poses.npy"); pp = np.load(f"{D}/sample_{i:02d}_pred_poses.npy")
    dg = np.stack([dec(np.load(f"{sd}/depth/{st + t:06d}.npy").astype(np.float64), R) for t in key])
    gen = frames(f"{D}/sample_{i:02d}.mp4"); gimg = [cv2.imread(f"{sd}/images/{st + t:06d}.png")[..., ::-1] for t in key]
    rgb_gen = np.stack([dec(gen[t], R) for t in key]); rgb_gt = np.stack([dec(x, R) for x in gimg])
    # PRED depth: SI-disparity -> one affine to GT disparity (best case)
    p = np.load(f"{D}/sample_{i:02d}_pred_depth.npy").astype(np.float64)
    p = np.stack([dec(x, R) for x in p])
    v = (dg > NEAR) & (dg < 200); tg = np.where(v, 1 / np.clip(dg, 1e-3, None), 0.)
    pf, tf = p[v], tg[v]; n = len(pf)
    s, t = np.linalg.solve(np.array([[(pf * pf).sum(), pf.sum()], [pf.sum(), n]]) + 1e-9 * np.eye(2), [(pf * tf).sum(), tf.sum()])
    dp = 1.0 / np.clip(s * p + t, 1 / 200.0, 1 / NEAR)
    da3d = np.load(f"{D}/viz_da3_fix/sample_{i:02d}_da3_depth.npy").astype(np.float64)
    da3c = np.load(f"{D}/viz_da3_fix/sample_{i:02d}_da3_poses.npy")
    da3d = np.stack([dec(x, R) for x in da3d])
    clouds = {}
    for name, dep, poses, col, kind in (("GT", dg, gp, rgb_gt, "carla_gt"), ("PRED", dp, pp, rgb_gen, "carla_pred"),
                                        ("DA3", da3d, da3c, rgb_gen, "opencv")):
        P, Cc, Z = [], [], []
        for j, f in enumerate(key):
            d = dep[j]; ok = ((d > NEAR) & (d < a.max_depth) & (dg[j] < a.max_depth)).reshape(-1)
            Z.append(d.reshape(-1)[ok])
            q = pix(R, R, fx, fy, cx, cy, d)
            if kind == "opencv":
                M = poses[f]; w = q @ M[:3, :3].T + M[:3, 3]
            else:
                M = c2w(poses[st + f] if kind == "carla_gt" else poses[f]); w = (q @ _C3.T) @ M[:3, :3].T + M[:3, 3]
            P.append(w[ok]); Cc.append(col[j].reshape(-1, 3)[ok])
        P, Cc, Z = voxel(np.concatenate(P), np.concatenate(Cc), np.concatenate(Z), a.voxel)
        o = np.argsort(Z, kind="stable")                 # nearest first: the viewer clips depth with a draw range
        clouds[name] = (P[o], Cc[o], Z[o])
    ctr = np.median(clouds["GT"][0], 0)
    gt_tr = np.array([gp[st + f, :3] for f in range(77)]); pr_tr = pp[:77, :3]; da_tr = da3c[:77, :3, 3]
    goal = gp[min(int(m["goal_frame"]), len(gp) - 1), :3]
    meta = {"sample": i, "seq": sd.split("val_bench5s/")[-1], "start": st, "goal": int(m["goal_frame"]),
            "voxel": a.voxel, "max_depth": a.max_depth}
    return meta, clouds, ctr, {"GT": gt_tr, "PRED": pr_tr, "DA3": da_tr}, goal


to3 = lambda P: np.stack([P[:, 0], P[:, 2], P[:, 1]], 1)                  # CARLA -> three.js (Y up)


def pack(meta, clouds, ctr, trajs, goal):
    """Binary blob for the multi-sample viewer: uint32 header length | JSON header | per cloud int16 pos, uint8 rgb,
    uint8 depth (0.2 m steps), each array 4-byte aligned. Positions = centimetres around ctr, three.js axes."""
    head = dict(meta); head["clouds"] = {}; head["traj"] = {k: np.round(to3(T - ctr), 3).tolist() for k, T in trajs.items()}
    head["goal_pt"] = np.round(to3((goal - ctr)[None])[0], 3).tolist()
    bufs, off = [], 0
    for k, (P, Cc, Z) in clouds.items():
        arrs = [np.round(to3(P - ctr) * 100).clip(-32767, 32767).astype(np.int16), Cc.astype(np.uint8),
                np.round(Z / 0.2).clip(0, 255).astype(np.uint8)]
        ent = {"n": int(len(P))}
        for name, x in zip(("pos", "col", "dep"), arrs):
            b = np.ascontiguousarray(x).tobytes(); pad = (-len(b)) % 4
            ent[name] = off; bufs.append(b + b"\0" * pad); off += len(b) + pad
        head["clouds"][k] = ent
    hj = json.dumps(head).encode(); hj += b" " * ((-len(hj)) % 4)
    return np.uint32(len(hj)).tobytes() + hj + b"".join(bufs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True); ap.add_argument("--sample", type=int, required=True)
    ap.add_argument("--template", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--res", type=int, default=256); ap.add_argument("--max_depth", type=float, default=40.0)
    ap.add_argument("--voxel", type=float, default=0.05); ap.add_argument("--label", default="")
    ap.add_argument("--clip", type=float, default=15.0, help="initial depth clip in the viewer (m); slider goes to --max_depth")
    a = ap.parse_args()
    meta, clouds, ctr, trajs, goal = _build(a.results_dir, a.sample, a)
    b64 = lambda x: base64.b64encode(np.ascontiguousarray(x).tobytes()).decode()
    data = dict(meta, label=a.label, clip=a.clip, clouds={}, traj={})
    for k, (P, Cc, Z) in clouds.items():
        q = np.round(to3(P - ctr) * 100).clip(-32767, 32767).astype(np.int16)
        zq = np.round(Z / 0.2).clip(0, 255).astype(np.uint8)      # camera depth, 0.2 m steps, ascending
        data["clouds"][k] = {"n": int(len(P)), "pos": b64(q), "col": b64(Cc.astype(np.uint8)), "dep": b64(zq)}
    data["traj"] = {k: np.round(to3(T - ctr), 3).tolist() for k, T in trajs.items()}
    data["goal_pt"] = np.round(to3((goal - ctr)[None])[0], 3).tolist()
    html = open(a.template).read().replace("/*__DATA__*/null", json.dumps(data))
    open(a.out, "w").write(html)
    print({k: v["n"] for k, v in data["clouds"].items()}, f"{os.path.getsize(a.out) / 1e6:.1f} MB -> {a.out}")


if __name__ == "__main__":
    main()
