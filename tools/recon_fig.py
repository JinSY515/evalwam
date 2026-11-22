#!/usr/bin/env python3
"""Figure stills of the /roll recon panels, no text: GT (GT d + GT pose) and Ours (PRED d + PRED pose).

Same clouds / colouring / framing as recon_orbit_rollout.py (orbit mp4 + BEV png on the /roll page), but
  * only the two panels, each saved as its own PNG, no labels / colour bar
  * orbit -> fixed stills at 4 azimuths (the mp4's frames 0/18/36/54 of 72), elev 22
  * BEV keeps the camera path (yellow) + start dot (green), as on the page
Both panels share one extent (metres per pixel), p98 over the two clouds.
Cleaner than the page: one pixel mask for both clouds -- GT and PRED depth < --max_depth (20 m, page 35 m) and no
flying pixel (relative depth jump to a neighbour > --edge) in either depth map.

  python recon_fig.py <roll_dir> [--samples 12 30 ...] [--out <roll_dir>/recon_fig]
  -> <out>/sNN/frames/{gt,ours}/fKKK.png   (ours = rollout_NN.mp4 frame t, gt = GT images/<start + t>, t = --ks)
     <out>/sNN/recon/{gt,ours}/azAAA.png   (orbit stills)
     <out>/sNN/bev/{gt,ours}.png
  The /roll page's 저장 button writes the same layout into cherry_export (roll.py export).
"""
import argparse, json, math, os, sys
from multiprocessing import Pool
import cv2, numpy as np
import imageio.v2 as iio

sys.path.insert(0, "/root/project/egowam/diffusers/examples/cosmos/tools")
from recon_orbit_rollout import build, c2w, dec, imgdec, unproj, NEAR, FAR  # noqa: E402

A = None


def splat(yv, xv, RGB, BY, BX):
    cnt, _, _ = np.histogram2d(yv, xv, bins=[BY, BX])
    dens = np.log1p(cnt); dens = dens / max(dens.max(), 1e-9)
    cs = np.stack([np.histogram2d(yv, xv, bins=[BY, BX], weights=RGB[:, c])[0] for c in range(3)], -1)
    im = cs / np.maximum(cnt, 1.0)[..., None]
    return (im * (0.25 + 0.75 * dens ** 0.5)[..., None]).clip(0, 255).astype(np.uint8)


def edge(d, thr):
    """flying pixels: relative depth jump to any 4-neighbour > thr"""
    r = np.zeros(d.shape, bool)
    for ax in (0, 1):
        g = np.abs(np.diff(d, axis=ax)) / np.minimum(np.delete(d, -1, ax), np.delete(d, 0, ax))
        m = g > thr
        if ax == 0:
            r[:-1] |= m; r[1:] |= m
        else:
            r[:, :-1] |= m; r[:, 1:] |= m
    return r


def build_fig(roll_dir, j, max_depth, stride, edge_thr, reanchor=True, rgb=True):
    """recon_orbit_rollout.build (same global depth affine, frames, poses), only GT d + GT pose and PRED d + PRED pose,
    with ONE pixel mask for both: GT and PRED depth < max_depth and no flying pixel in either (edge_thr, 0 = off)."""
    v = json.load(open(f"{roll_dir}/rollout_meta.json"))[str(j)]
    sd = v["seq_dir"]; st, F, OV, K = int(v["start_frame"]), int(v["F"]), int(v["overlap"]), int(v["num_chunks"])
    p = np.load(f"{roll_dir}/rollout_{j:02d}_pred_depth.npy").astype(np.float64)
    if p.ndim == 5:
        p = p[:, :, 0]
    T_lat, W_ = p.shape[1], p.shape[-1]
    seen, FR = set(), []
    for c in range(K):
        for k in range(T_lat):
            f = st + c * (F - OV) + 4 * k
            if f not in seen:
                seen.add(f); FR.append((c, k, f))
    gp = np.load(f"{sd}/poses.npy"); i0 = np.load(f"{sd}/intrinsics.npy").astype(float)
    nd = np.load(f"{sd}/depth/{FR[0][2]:06d}.npy").shape[0]; intr = tuple(x * W_ / nd for x in i0)
    dg = {f: dec(np.load(f"{sd}/depth/{f:06d}.npy").astype(np.float64), W_) for _, _, f in FR}
    P = np.concatenate([p[c, k][(dg[f] > NEAR) & (dg[f] < FAR)] for c, k, f in FR])
    T = np.concatenate([(1.0 / np.clip(dg[f], 1e-3, None))[(dg[f] > NEAR) & (dg[f] < FAR)] for c, k, f in FR])
    A_ = np.array([[(P * P).sum(), P.sum()], [P.sum(), len(P)]]); b = np.array([(P * T).sum(), T.sum()])
    s, t = np.linalg.solve(A_ + 1e-9 * np.eye(2), b)
    raw = np.load(f"{roll_dir}/rollout_{j:02d}_pred_poses.npy")
    cg = {f: c2w(gp[f]) for _, _, f in FR}
    # pred_poses[0] is the GT start pose MIRRORED (y, roll, yaw negated: invert_raymap anchor convention) while the motion
    # after it is in GT handedness -> the whole track is rotated by the start-heading error. Frame 0 IS the GT frame, so
    # re-anchor: ours_t = G_0 P_0^-1 P_t (keeps every relative motion, start = GT start pose).
    A0 = cg[st] @ np.linalg.inv(c2w(raw[0])) if reanchor else np.eye(4)
    cp = {f: A0 @ c2w(raw[f - st]) for _, _, f in FR}
    dp = {f: 1.0 / np.clip(s * p[c, k] + t, 1 / FAR, 1 / NEAR) for c, k, f in FR}
    M = {}
    for _, _, f in FR:
        m = (dg[f] > NEAR) & (dg[f] < max_depth) & (dp[f] < max_depth)
        if edge_thr > 0:
            m &= ~edge(dg[f], edge_thr) & ~edge(dp[f], edge_thr)
        M[f] = m.ravel()
    clouds = [np.concatenate([unproj(D[f], C[f], intr)[M[f]][::stride] for _, _, f in FR]) for D, C in ((dg, cg), (dp, cp))]
    RGB = None if not rgb else np.concatenate([imgdec(f"{sd}/images/{f:06d}.png", W_).reshape(-1, 3)[M[f]][::stride] for _, _, f in FR]).astype(np.float64)
    tracks = [np.stack([C[f][:3, 3] for _, _, f in FR]) for C in (cg, cp)]
    return clouds, RGB, tracks


def render(roll_dir, j, path, ks=(0, 76, 152, 228, 304), size=400, stride=6, elev=22.0, az=(0, 90, 180, 270),
           frames_only=False, max_depth=20.0, edge_thr=0.05):
    """Write everything for clip j; path(name) -> output file, name in gt_fKKK / ours_fKKK / {gt,ours}_bev /
    {gt,ours}_azAAA. Returns None or an error string."""
    meta = json.load(open(f"{roll_dir}/rollout_meta.json"))[str(j)]
    st, sd = int(meta["start_frame"]), meta["seq_dir"].rstrip("/")
    rd = iio.get_reader(f"{roll_dir}/rollout_{j:02d}.mp4"); fr = [f[..., :3] for f in rd]; rd.close()
    for t in ks:
        iio.imwrite(path(f"ours_f{t:03d}"), fr[t])
        iio.imwrite(path(f"gt_f{t:03d}"), iio.imread(f"{sd}/images/{st + t:06d}.png")[..., :3])
    if frames_only:
        return None
    if not os.path.exists(f"{roll_dir}/rollout_{j:02d}_pred_poses.npy"):
        return "no pred poses"
    clouds, RGB, tracks = build_fig(roll_dir, j, max_depth, stride, edge_thr)
    CT = [np.median(P, 0) for P in clouds]
    EX = max(max(float(np.percentile(np.abs(np.concatenate([(P - c)[:, 0], (P - c)[:, 1]])), 98))
                 for P, c in zip(clouds, CT)), 5.0)
    S = size; SX = int(round(S * 1.1)); SX += SX & 1
    BY = np.linspace(-EX, EX, S + 1); BX = np.linspace(-EX * 1.1, EX * 1.1, SX + 1)
    lw = max(2, round(2 * S / 400))
    for tag, k in (("gt", 0), ("ours", 1)):
        Q = clouds[k] - CT[k]
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            im = splat(Q[:, 1], Q[:, 0], RGB, BY, BX)
        TR = tracks[k] - CT[k]
        px = ((TR[:, 0] - BX[0]) / (BX[-1] - BX[0]) * SX).astype(np.int32)
        py = ((TR[:, 1] - BY[0]) / (BY[-1] - BY[0]) * S).astype(np.int32)
        cv2.polylines(im, [np.stack([px, py], -1).reshape(-1, 1, 2)], False, (0, 255, 255), lw, cv2.LINE_AA)
        cv2.circle(im, (int(px[0]), int(py[0])), 2 * lw, (0, 255, 0), -1)
        cv2.imwrite(path(f"{tag}_bev"), im)
        el = math.radians(elev)
        for a_ in az:
            th = math.radians(a_)
            fwd = np.array([math.cos(th) * math.cos(el), math.sin(th) * math.cos(el), math.sin(el)])
            rgt = np.cross(fwd, [0, 0, 1.0]); rgt /= np.linalg.norm(rgt); up2 = np.cross(rgt, fwd)
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                im = splat(-(Q @ up2), Q @ rgt, RGB, BY, BX)
            cv2.imwrite(path(f"{tag}_az{a_:03d}"), im)
    return None


def layout(root):
    """name -> file under root:  frames/{gt,ours}/fKKK.png   recon/{gt,ours}/azAAA.png   bev/{gt,ours}.png"""
    def path(n):
        tag, key = n.split("_", 1)
        p = (f"{root}/frames/{tag}/{key}.png" if key.startswith("f") else
             f"{root}/bev/{tag}.png" if key == "bev" else f"{root}/recon/{tag}/{key}.png")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        return p
    return path


def one(j):
    a = A
    return j, render(a.roll_dir, j, layout(f"{a.out}/s{j:02d}"), a.ks, a.size, a.stride, a.elev, a.az, a.frames_only,
                     a.max_depth, a.edge)


def main():
    global A
    ap = argparse.ArgumentParser()
    ap.add_argument("roll_dir"); ap.add_argument("--samples", type=int, nargs="*")
    ap.add_argument("--out"); ap.add_argument("--size", type=int, default=400); ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--elev", type=float, default=22.0); ap.add_argument("--az", type=int, nargs="+", default=[0, 90, 180, 270])
    ap.add_argument("--ks", type=int, nargs="+", default=[0, 76, 152, 228, 304])
    ap.add_argument("--frames_only", action="store_true")
    ap.add_argument("--max_depth", type=float, default=20.0, help="m; drop pixels where GT OR PRED depth is farther")
    ap.add_argument("--edge", type=float, default=0.05, help="flying-pixel threshold (relative depth jump), 0 = off")
    ap.add_argument("--procs", type=int, default=16)
    A = ap.parse_args()
    A.out = A.out or f"{A.roll_dir}/recon_fig"; os.makedirs(A.out, exist_ok=True)
    js = A.samples or sorted(int(k) for k in json.load(open(f"{A.roll_dir}/rollout_meta.json")))
    with Pool(min(A.procs, len(js))) as p:
        for j, e in p.imap_unordered(one, js):
            if e:
                print(f"[skip] {j}: {e}")
    print(f"{A.out}  ({len(os.listdir(A.out))} files)")


if __name__ == "__main__":
    main()
