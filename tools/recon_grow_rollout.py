#!/usr/bin/env python3
"""Growing point-cloud video of a long rollout: frame by frame, the cloud contains only the frames seen so far.

Same inputs and depth/pose handling as recon_orbit_rollout.py (its build(): global affine depth fit, seam frames
de-duplicated, GT RGB colours). For each unique frame i of the rollout the panels show
  [ GT image | generated frame ]  +  one panel per (depth, pose) combo with the points of frames 0..i,
the camera path so far (yellow) and the current camera (red dot + heading).
The view is fixed (each panel framed on its own final cloud, common scale, so the reconstruction grows into the frame) with a slow azimuth sweep;
brightness uses the final cloud's density scale so it does not flicker as points accumulate.

  python3 tools/recon_grow_rollout.py --roll_dir <merged rollout dir> --sample 12 --out grow_s12.mp4
"""
import argparse, math, os, sys
import cv2, numpy as np
import imageio.v2 as imageio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recon_orbit_rollout import build  # noqa: E402


def txt(im, s, xy, sc=0.42):
    cv2.putText(im, s, xy, cv2.FONT_HERSHEY_SIMPLEX, sc, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(im, s, xy, cv2.FONT_HERSHEY_SIMPLEX, sc, (255, 255, 255), 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roll_dir", required=True)
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--panels", default="0,3", help="combo indices: 0 GT d+GT pose, 1 PRED d+GT pose, 2 GT d+PRED pose, 3 PRED d+PRED pose")
    ap.add_argument("--size", type=int, default=420)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--max_depth", type=float, default=35.0)
    ap.add_argument("--elev", type=float, default=35.0)
    ap.add_argument("--sweep", type=float, default=60.0, help="total azimuth sweep (deg) over the video")
    ap.add_argument("--hold", type=int, default=12, help="extra frames holding the final cloud")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--extent_pct", type=float, default=98.0)
    a = ap.parse_args()

    names, clouds, ctr, RGB, info = build(a.roll_dir, a.sample, a.max_depth, a.stride, True)
    FR, CNT = info["frames"], np.array(info["counts"])
    END = np.cumsum(CNT)
    assert END[-1] == len(clouds[0]), (END[-1], len(clouds[0]))
    sel = [int(x) for x in a.panels.split(",") if int(x) < len(names)]
    st = info["start"]
    print(f"[info] {len(FR)} frames, {len(clouds[0]):,} pts/combo, panels {[names[k] for k in sel]}")

    S = a.size + (a.size & 1); SX = int(round(S * 1.1)); SX += SX & 1
    # each panel centred on its own final cloud (a drifting PRED-pose cloud is not cut off), one common extent
    CT = {k: np.median(clouds[k], 0) for k in sel}
    EX = max([float(np.percentile(np.abs(np.concatenate([(clouds[k] - CT[k])[:, 0], (clouds[k] - CT[k])[:, 1]])), a.extent_pct))
              for k in sel] + [5.0])
    BY = np.linspace(-EX, EX, S + 1); BX = np.linspace(-EX * 1.1, EX * 1.1, SX + 1)

    # initial heading of the GT camera -> look along it from behind/above
    R0 = info["cam_rots"][0][0]; h0 = R0[:3, 0]          # CARLA forward = +x of the camera-to-world rotation
    az0 = math.atan2(h0[1], h0[0])
    el = math.radians(a.elev)

    def basis(fi, n):
        th = az0 + math.radians(a.sweep) * (fi / max(n - 1, 1) - 0.5)
        fwd = np.array([math.cos(th) * math.cos(el), math.sin(th) * math.cos(el), math.sin(el)])
        rgt = np.cross(fwd, [0, 0, 1.0]); rgt /= np.linalg.norm(rgt); up2 = np.cross(rgt, fwd)
        return rgt, up2

    def to_px(P, rgt, up2, c0):
        Q = P - c0
        return np.stack([((Q @ rgt) - BX[0]) / (BX[-1] - BX[0]) * SX, ((-(Q @ up2)) - BY[0]) / (BY[-1] - BY[0]) * S], -1)

    # fixed brightness scale from the final clouds (mid-sweep view)
    rgt, up2 = basis(0, 2)
    DMAX = []
    for k in sel:
        Q = clouds[k] - CT[k]
        c, _, _ = np.histogram2d(-(Q @ up2), Q @ rgt, bins=[BY, BX]); DMAX.append(np.log1p(c.max()))

    cap = cv2.VideoCapture(os.path.join(a.roll_dir, f"rollout_{a.sample:02d}.mp4")); gen = []
    while True:
        ok, im = cap.read()
        if not ok:
            break
        gen.append(im)
    sd = info["seq_dir"]; V = S // 2

    n = len(FR) + a.hold
    wr = imageio.get_writer(a.out, fps=a.fps, macro_block_size=1)
    for fi in range(n):
        i = min(fi, len(FR) - 1); f = FR[i]
        rgt, up2 = basis(fi, n)
        gt = cv2.imread(f"{sd}/images/{f:06d}.png"); gi = gen[min(f - st, len(gen) - 1)] if gen else np.zeros_like(gt)
        col = np.concatenate([cv2.resize(gt, (V, V)), cv2.resize(gi, (V, V))], 0)
        txt(col, "GT", (6, 18)); txt(col, "generated", (6, V + 18))
        col = np.concatenate([col, np.zeros((S - col.shape[0], V, 3), np.uint8)], 0) if col.shape[0] < S else col[:S]
        row = [col]
        for k, dm in zip(sel, DMAX):
            P = clouds[k][:END[i]]; Q = P - CT[k]
            yv, xv = -(Q @ up2), Q @ rgt
            cnt, _, _ = np.histogram2d(yv, xv, bins=[BY, BX])
            cs = np.stack([np.histogram2d(yv, xv, bins=[BY, BX], weights=RGB[:END[i], c])[0] for c in range(3)], -1)
            dens = np.log1p(cnt) / max(dm, 1e-9)
            im = (cs / np.maximum(cnt, 1.0)[..., None] * (0.25 + 0.75 * np.clip(dens, 0, 1) ** 0.5)[..., None]).clip(0, 255).astype(np.uint8)
            tr = to_px(info["cam_tracks"][k][:i + 1], rgt, up2, CT[k]).astype(np.int32)
            cv2.polylines(im, [tr.reshape(-1, 1, 2)], False, (0, 255, 255), 2, cv2.LINE_AA)
            hd = to_px(info["cam_tracks"][k][i] + 2.0 * info["cam_rots"][k][i][:3, 0], rgt, up2, CT[k]).astype(np.int32)
            cv2.line(im, tuple(tr[-1]), tuple(hd), (0, 0, 255), 2, cv2.LINE_AA)
            cv2.circle(im, tuple(tr[-1]), 5, (0, 0, 255), -1)
            txt(im, names[k], (6, 20)); row.append(im)
        mos = np.concatenate(row, 1)
        bar = np.zeros((24, mos.shape[1], 3), np.uint8)
        c_ = next(c for c in range(info["K"]) if f < st + (c + 1) * (info["F"] - info["overlap"]) + info["overlap"])
        txt(bar, f"frame {f - st:3d}/{FR[-1] - st}  |  chunk {c_ + 1}/{info['K']}  |  points so far {END[i]:,}  |  "
                 f"yellow = camera path, red = current camera  |  colour = GT RGB", (10, 17))
        mos = np.concatenate([mos, bar], 0)
        if mos.shape[0] & 1:
            mos = np.concatenate([mos, np.zeros((1,) + mos.shape[1:], np.uint8)], 0)
        wr.append_data(cv2.cvtColor(mos, cv2.COLOR_BGR2RGB))
    wr.close()
    print(f"[ok] {a.out}  {n} frames, {mos.shape[1]}x{mos.shape[0]}, {os.path.getsize(a.out) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
