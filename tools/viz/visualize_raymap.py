#!/usr/bin/env python3
"""Visualize the raymap (Plücker ray field) as a colorized VIDEO — "the camera as an image".

Reproduces transformer_cosmos.convert_to_plucker (frame-0-relative) in numpy, from CARLA
poses. Each output frame = colorized [ray-direction | ray-moment] for that timestep; the mp4
therefore shows how the ray field evolves = the camera motion. Counterfactual modes
(real/static/mirror) can be stacked to see that the "action as an image" differs per mode.

Usage:
  python visualize_raymap.py --poses <seq>/poses.npy --intrinsics <seq>/intrinsics.npy
  python visualize_raymap.py --poses <seq>/poses.npy --intrinsics <seq>/intrinsics.npy \
         --start 4 --num_frames 77 --modes real static mirror --out raymap.mp4
"""
import os, argparse, math
import numpy as np


def carla_pose_to_mat(x, y, z, roll, pitch, yaw):
    """CARLA [x,y,z,roll,pitch,yaw] (deg) -> world_T_cam (4,4). Matches data_utils._carla_pose_to_mat."""
    r, p, yw = math.radians(roll), math.radians(pitch), math.radians(yaw)
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(yw), math.sin(yw)
    M = np.eye(4)
    M[0, 0] = cy*cp; M[0, 1] = cy*sp*sr - sy*cr; M[0, 2] = cy*sp*cr + sy*sr
    M[1, 0] = sy*cp; M[1, 1] = sy*sp*sr + cy*cr; M[1, 2] = sy*sp*cr - cy*sr
    M[2, 0] = -sp;   M[2, 1] = cp*sr;            M[2, 2] = cp*cr
    M[0, 3] = x; M[1, 3] = y; M[2, 3] = z
    return M


def mirror_poses_ego(poses):
    poses = np.asarray(poses, float).copy()
    x0, y0, yaw0 = poses[0, 0], poses[0, 1], math.radians(poses[0, 5])
    c, s = math.cos(-yaw0), math.sin(-yaw0)
    ci, si = math.cos(yaw0), math.sin(yaw0)
    out = poses.copy()
    for i in range(len(poses)):
        dx, dy = poses[i, 0] - x0, poses[i, 1] - y0
        ex = c*dx - s*dy; ey = -(s*dx + c*dy)
        out[i, 0] = x0 + (ci*ex - si*ey)
        out[i, 1] = y0 + (si*ex + ci*ey)
        out[i, 5] = poses[0, 5] - (poses[i, 5] - poses[0, 5])
        out[i, 3] = -poses[i, 3]
    return out


def compute_plucker(poses, intr, H, W):
    """poses:(F,6), intr:[fx,fy,cx,cy] native res. Returns dir_world,(F,H,W,3), moment,(F,H,W,3).
    Frame-0-relative, matching convert_to_plucker."""
    ext = np.stack([carla_pose_to_mat(*p) for p in poses])         # (F,4,4) world_T_cam
    R = ext[:, :3, :3]; t = ext[:, :3, 3:]                          # (F,3,3),(F,3,1)
    R0inv = R[0].T
    Rrel = np.einsum('ij,fjk->fik', R0inv, R)                       # (F,3,3)
    campos = np.einsum('ij,fjk->fik', R0inv, (t - t[0])).squeeze(-1)  # (F,3)
    fx, fy, cx, cy = [float(v) for v in np.asarray(intr).ravel()[:4]]
    ow, oh = 2 * cx, 2 * cy                                         # native res implied by principal point
    sx, sy = W / ow, H / oh
    fx, fy, cx, cy = fx*sx, fy*sy, cx*sx, cy*sy
    ys, xs = np.meshgrid(np.arange(H) + 0.5, np.arange(W) + 0.5, indexing='ij')
    nx = (xs - cx) / fx; ny = (ys - cy) / fy
    dirs = np.stack([nx, ny, np.ones_like(nx)], -1)                # (H,W,3) camera frame
    dw = np.einsum('fij,hwj->fhwi', Rrel, dirs)                    # (F,H,W,3)
    dw = dw / np.clip(np.linalg.norm(dw, axis=-1, keepdims=True), 1e-8, None)
    cp = campos[:, None, None, :]                                  # (F,1,1,3)
    mom = np.cross(np.broadcast_to(cp, dw.shape), dw)              # (F,H,W,3)
    return dw, mom


def colorize(dw, mom):
    dir_rgb = ((dw + 1) / 2 * 255).clip(0, 255).astype(np.uint8)   # unit dir -> RGB (normal-map style)
    lo, hi = np.percentile(mom, 2), np.percentile(mom, 98)
    mom_rgb = (np.clip((mom - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)
    return dir_rgb, mom_rgb


def transform(poses, mode):
    if mode == "static":
        return np.repeat(poses[:1], len(poses), 0)
    if mode == "mirror":
        return mirror_poses_ego(poses)
    if mode == "reverse":
        return poses[::-1].copy()
    return poses  # real


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poses", required=True)
    ap.add_argument("--intrinsics", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--images", default=None,
                    help="원본 프레임 디렉토리(png/jpg). 미지정 시 <poses 폴더>/images 자동. 왼쪽 패널로 표시")
    ap.add_argument("--height", type=int, default=128)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--num_frames", type=int, default=0)    # 0 = all
    ap.add_argument("--modes", nargs="+", default=["real"], help="real static mirror reverse")
    ap.add_argument("--fps", type=int, default=16)
    args = ap.parse_args()
    import imageio

    poses = np.load(args.poses).astype(np.float64)
    intr = np.load(args.intrinsics).astype(np.float64)
    if args.num_frames > 0:
        poses = poses[args.start:args.start + args.num_frames]
    out = args.out or (os.path.splitext(args.poses)[0] + "_raymap.mp4")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)

    # 원본 영상(왼쪽 패널): poses 옆 images/ 를 기본으로, [start:start+num_frames] 슬라이스해 (H,W) 리사이즈.
    # 없으면 rgb=None 으로 두어 기존 레이아웃([dir|moment]) 그대로 동작.
    import glob as _glob
    rgb = None
    _img_dir = args.images or os.path.join(os.path.dirname(os.path.abspath(args.poses)), "images")
    _files = sorted(_glob.glob(os.path.join(_img_dir, "*.png")) +
                    _glob.glob(os.path.join(_img_dir, "*.jpg")))
    if args.num_frames > 0:
        _files = _files[args.start:args.start + args.num_frames]
    if _files:
        import cv2
        rgb = np.stack([cv2.resize(imageio.imread(f)[:, :, :3], (args.width, args.height))
                        for f in _files])                      # (F, H, W, 3) uint8
        print(f"[raymap] 원본 영상 왼쪽 패널: {len(_files)} 프레임 ({_img_dir})")
    else:
        print(f"[raymap] 원본 영상 없음({_img_dir}) → dir|moment 만 표시")

    per_mode = []
    for mode in args.modes:
        dw, mom = compute_plucker(transform(poses, mode), intr, args.height, args.width)
        per_mode.append((mode,) + colorize(dw, mom))
    F = min(len(x[1]) for x in per_mode)

    if rgb is not None:
        F = min(F, len(rgb))
    with imageio.get_writer(out, fps=args.fps, macro_block_size=1) as w:
        for i in range(F):
            # 각 모드 행 = [원본 | dir | moment] (원본 없으면 [dir | moment])
            rows = []
            for _, dr, mr in per_mode:
                cols = [rgb[i]] if rgb is not None else []
                cols += [dr[i], mr[i]]
                rows.append(np.concatenate(cols, axis=1))
            w.append_data(np.concatenate(rows, axis=0))                                # modes stacked vertically
    _cols = "[orig | dir | moment]" if rgb is not None else "[dir | moment]"
    print(f"[raymap] {out}  ({F} frames)  layout: rows=modes{args.modes}, cols={_cols}")


if __name__ == "__main__":
    main()
