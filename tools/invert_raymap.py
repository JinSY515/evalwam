#!/usr/bin/env python3
"""Invert a raymap (Plucker ray field) back to CARLA camera poses.

`--pred_raymap` GENERATES the camera, but the only score it currently gets is an MSE in VAE
latent space (sample_NN_raymap.json), which says "better than predicting zeros" and nothing
about metres or degrees. The raymap is a Plucker field, so the poses that produced it are
recoverable in closed form -- this tool does that and scores the result as a trajectory.

The inverse, per frame, from direction d(u,v) (unit) and moment m(u,v) (metres):

  d_cam(u,v) = normalize([(x-cx)/fx, (y-cy)/fy, 1]),  x = col+0.5, y = row+0.5
  R_rel      = argmin_R ||R d_cam - d_world||  = _proj_so3(d_world^T d_cam)     (Procrustes)
  d_hat      = d_cam R_rel^T                                                    (rigid-consistent)
  m = o x d  =>  sum(d x m) = (N I - D^T D) o   =>  o = solve(N I - D^T D, sum(d_hat x m))

The moment equation is linear in o and its normal equations collapse to one 3x3 solve (no
lstsq on a 3N x 3 stack). cond(G) = 2.274 for this 90-degree FOV grid, so it is well posed.

Two things that are easy to get wrong and are asserted/handled here:
  * convert_to_plucker re-expresses the window RELATIVE TO FRAME 0 (transformer_cosmos.py
    :1057-1061). Absolute world placement is NOT in the raymap; frame 0's pose is taken from
    the GT action_poses. That is not cheating, it is what the representation contains.
  * the translation solve uses d_hat (the field re-synthesized from the recovered R), not the
    decoded d. The decoded d carries per-pixel VAE noise that would bias o. The decoded d is
    kept only as a residual diagnostic.

Stages, cheapest first. Each one bounds the next:
  s1  GT poses -> plucker -> invert.                No VAE, no GPU, no weights. Gate: < 1e-6.
  s2  GT poses -> build_raymap_latent -> invert.    The VAE+clamp noise floor. GPU.
  s3  pred_raymap.npy -> invert.                    The real measurement. GPU.
An s3 number is reportable only if s2_floor < s3_value / 3. s2 is carried into the same CSV
row as s3 so the two can never be read apart.

Usage:
  # s1 needs nothing but numpy -- run it first, anywhere
  python invert_raymap.py --results_dir <triptych dir> --stage s1

  # s2/s3 load the Cosmos VAE (no DiT, no T5, no checkpoint)
  python invert_raymap.py --results_dir <triptych dir> --stage all \
      --model_name nvidia/Cosmos-Predict2.5-2B --revision diffusers/base/post-trained \
      --ray_downsample 2 --raymap_moment_scale 8.0 --frustum --num_samples 8
"""
import os
import re
import sys
import json
import math
import glob
import types
import argparse
import functools
import importlib.util

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
COSMOS = os.environ.get("EGOWAM_COSMOS", "/root/project/egowam/diffusers/examples/cosmos")
SRC = os.path.abspath(os.path.join(COSMOS, "..", "..", "src"))

# RAYMAP_CAM_FIX=opencv (transformer_cosmos.py:43-51) builds the plucker field from R_abs @ C, so
# the field holds C^T R_rel C and C^T t_rel. Undone in invert_rays with the SAME env var the
# model was trained/inferred with; unset -> unchanged. s2 picks the fix up automatically because
# it calls the production convert_to_plucker, which reads the same variable at import.
_CAM_FIX = os.environ.get("RAYMAP_CAM_FIX", "").strip().lower() in ("1", "on", "true", "opencv")
_C = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


def _load(name, path):
    """Load a sibling script as a module. Same shim as eval_rollout.py:49-59 -- these tools are
    not a package, so a plain import does not reach them."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_ECC = _VR = None


def _ecc():
    """tools/eval_camera_control.py -- camera_metrics, _proj_so3, geodesic_deg, _write_csv,
    carla_pose_to_mat. Top-level imports there are stdlib+numpy (the SfM estimators import
    lazily), so this stays cheap."""
    global _ECC
    if _ECC is None:
        _ECC = _load("ecc", os.path.join(HERE, "eval_camera_control.py"))
    return _ECC


def _vr():
    """tools/viz/visualize_raymap.py -- compute_plucker, an INDEPENDENT numpy replica of
    convert_to_plucker. Used as s1's forward map so the round trip is torch-vs-numpy, i.e. a
    real second opinion rather than the same code twice."""
    global _VR
    if _VR is None:
        _VR = _load("vr", os.path.join(HERE, "viz", "visualize_raymap.py"))
    return _VR


# ── the inverse of _carla_pose_to_mat (data_utils.py:118-132) ───────────────────────────────
# M = Rz(yaw) Ry(pitch) Rx(roll); nothing in the repo extracts the angles back out.
def mat_to_carla_pose(M):
    """world_T_cam (4,4) -> [x, y, z, roll, pitch, yaw] with angles in degrees, float64."""
    M = np.asarray(M, dtype=np.float64)
    pitch = math.asin(float(np.clip(-M[2, 0], -1.0, 1.0)))
    if abs(M[2, 0]) > 1.0 - 1e-9:                    # gimbal lock: roll and yaw degenerate
        yaw, roll = math.atan2(-M[0, 1], M[1, 1]), 0.0
    else:
        yaw = math.atan2(M[1, 0], M[0, 0])
        roll = math.atan2(M[2, 1], M[2, 2])
    return np.array([M[0, 3], M[1, 3], M[2, 3],
                     math.degrees(roll), math.degrees(pitch), math.degrees(yaw)],
                    dtype=np.float64)


def poses_to_mats(poses):
    """(F,6) CARLA rows -> (F,4,4) world_T_cam, via the eval script's own converter."""
    m = _ecc()
    return np.stack([m.carla_pose_to_mat(*p) for p in np.asarray(poses, dtype=np.float64)])


# ── geometry ────────────────────────────────────────────────────────────────────────────────
def pixel_ray_dirs(intr4, Hr, Wr, orig_wh):
    """Camera-frame unit ray directions on the raymap's own grid, (Hr*Wr, 3) float64.

    Mirrors convert_to_plucker:1063-1086 exactly: anisotropic rescale of BOTH f and c by
    (Wr/original_w, Hr/original_h), pixel CENTRES (+0.5), and indexing="ij" so the row index
    feeds y. Getting either the +0.5 or the y/x order wrong still produces a plausible-looking
    wrong answer, which is why s1 gates at 1e-6."""
    fx, fy, cx, cy = [float(v) for v in np.asarray(intr4, dtype=np.float64).ravel()[:4]]
    ow, oh = float(orig_wh[0]), float(orig_wh[1])
    sx, sy = Wr / ow, Hr / oh
    fx, fy, cx, cy = fx * sx, fy * sy, cx * sx, cy * sy
    ys, xs = np.meshgrid(np.arange(Hr, dtype=np.float64) + 0.5,
                         np.arange(Wr, dtype=np.float64) + 0.5, indexing="ij")
    d = np.stack([(xs - cx) / fx, (ys - cy) / fy, np.ones_like(xs)], axis=-1).reshape(-1, 3)
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    return np.ascontiguousarray(d)


def invert_frame(d_world, moment, d_cam):
    """One frame: (R_rel, o, diag). d_world/moment (N,3) decoded, d_cam (N,3) analytic unit."""
    A = np.ascontiguousarray(np.asarray(d_cam, dtype=np.float64))
    B = np.asarray(d_world, dtype=np.float64)
    n = np.linalg.norm(B, axis=-1, keepdims=True)
    # renormalize: this is what makes ROTATION robust -- all radial VAE error is discarded here
    # and only angular error survives into the Procrustes fit, averaged over every pixel
    B = np.ascontiguousarray(B / np.maximum(n, 1e-12))
    m = np.asarray(moment, dtype=np.float64)

    # einsum, not `@`: on macOS Accelerate np.matmul emits spurious divide-by-zero/overflow
    # warnings on tall (N,3) operands. Values are identical; einsum and np.dot stay quiet.
    R = _ecc()._proj_so3(np.einsum("ni,nj->ij", B, A))   # argmax tr(R A^T B) = closest rotation
    d_hat = np.ascontiguousarray(np.einsum("ij,nj->ni", R, A))   # unit, rigid-consistent

    N = A.shape[0]
    G = N * np.eye(3) - np.einsum("ni,nj->ij", d_hat, d_hat)  # sum(d x (o x d)) = (N I - D^TD) o
    rhs = np.cross(d_hat, m).sum(axis=0)
    o = np.linalg.solve(G, rhs)

    cosang = np.clip((B * d_hat).sum(-1), -1.0, 1.0)
    diag = {
        "proc_resid_deg": float(np.degrees(np.arccos(cosang)).mean()),
        "mom_resid_m": float(np.linalg.norm(m - np.cross(np.broadcast_to(o, d_hat.shape), d_hat),
                                            axis=-1).mean()),
        "cond_G": float(np.linalg.cond(G)),
    }
    return R, o, diag


def invert_rays(dirs, moments, intr4, orig_wh, anchor_pose, moment_scale=None):
    """(F,H,W,3) direction + moment fields -> ((F,6) absolute CARLA poses, (F,6) relative, diag).

    Relative poses are the raw content of the raymap (frame 0 = identity at the origin);
    absolute ones are those re-anchored on the GT frame-0 pose:
        R_abs = R0 R_rel,  t_abs = R0 o + t0."""
    dirs = np.asarray(dirs)
    moments = np.asarray(moments)
    F, Hr, Wr = dirs.shape[0], dirs.shape[1], dirs.shape[2]
    d_cam = pixel_ray_dirs(intr4, Hr, Wr, orig_wh)

    M0 = poses_to_mats(np.asarray(anchor_pose, dtype=np.float64)[None])[0]
    R0, t0 = M0[:3, :3], M0[:3, 3]

    abs_rows, rel_rows, diags = [], [], []
    for f in range(F):
        d = dirs[f].reshape(-1, 3).astype(np.float64)
        m = moments[f].reshape(-1, 3).astype(np.float64)
        R_rel, o, dg = invert_frame(d, m, d_cam)
        if _CAM_FIX:                                  # field = (C^T R C, C^T t) -> (R, t)
            R_rel, o = _C @ R_rel @ _C.T, _C @ o
        if moment_scale is not None:
            # fraction of moment components sitting in the outer 5% of the clamp budget: the
            # real windows here reach ~0.92 of it, so this is the number that says whether a
            # given sequence is about to saturate and lose translation irrecoverably
            dg["clamp_frac"] = float((np.abs(m) > 0.95 * moment_scale).mean())
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R_rel, o
        rel_rows.append(mat_to_carla_pose(T))
        T[:3, :3], T[:3, 3] = np.einsum("ij,jk->ik", R0, R_rel), R0 @ o + t0
        abs_rows.append(mat_to_carla_pose(T))
        diags.append(dg)

    agg = {k: float(np.mean([d[k] for d in diags])) for k in diags[0]}
    agg["cond_G"] = float(np.max([d["cond_G"] for d in diags]))      # worst case, not mean
    return np.stack(abs_rows), np.stack(rel_rows), agg


def _forward_plucker_camfix(poses, intr4, Hr, Wr, orig_wh):
    """numpy replica of convert_to_plucker WITH the cam fix (rot_abs @ C, relative to frame 0)."""
    Ms = poses_to_mats(poses)
    R = np.einsum("fij,jk->fik", Ms[:, :3, :3], _C)
    R0t = R[0].T
    Rrel = np.einsum("ij,fjk->fik", R0t, R)
    pos = np.einsum("ij,fj->fi", R0t, Ms[:, :3, 3] - Ms[0, :3, 3])
    dc = pixel_ray_dirs(intr4, Hr, Wr, orig_wh)
    dirs = np.einsum("fij,nj->fni", Rrel, dc)
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
    moms = np.cross(np.broadcast_to(pos[:, None, :], dirs.shape), dirs)
    F = len(Ms)
    return dirs.reshape(F, Hr, Wr, 3), moms.reshape(F, Hr, Wr, 3)


def forward_plucker_numpy(poses, intr4, Hr, Wr, orig_wh=None):
    """GT poses -> (F,H,W,3) dir + moment, via visualize_raymap.compute_plucker, frame by frame.

    Per-frame so a 77x256x256x6 float64 field never materializes at once. compute_plucker
    gauges to ITS OWN first row, so each call gets [frame0, frame_f] and keeps index 1."""
    poses = np.asarray(poses, dtype=np.float64)
    if _CAM_FIX:
        return _forward_plucker_camfix(poses, intr4, Hr, Wr, orig_wh)
    cp = _vr().compute_plucker
    dirs, moms = [], []
    for f in range(len(poses)):
        dw, mm = cp(poses[[0, f]], intr4, Hr, Wr)
        dirs.append(dw[1 if f else 0])
        moms.append(mm[1 if f else 0])
    return np.stack(dirs), np.stack(moms)


# ── metrics ─────────────────────────────────────────────────────────────────────────────────
def pose_errors(pred, gt, prefix=""):
    """Alignment-free errors. camera_metrics sim3-aligns WITH scale, which makes its transerr
    blind to a systematic moment-scale error -- the most likely failure mode here. Frame 0 is
    anchored and the scale is metric, so these are the numbers to trust.

    Units are metres throughout: CARLA is UE4-based and metric. Confirmed on this data by
    z = 1.14..1.30 m (walking eye height) and a 0.095 m per-frame step over a 7.2 m window."""
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    dt = np.linalg.norm(pred[:, :3] - gt[:, :3], axis=1)
    dyaw = (pred[:, 5] - gt[:, 5] + 180.0) % 360.0 - 180.0
    dp = np.linalg.norm(pred[:, :3] - pred[0, :3], axis=1)
    dg = np.linalg.norm(gt[:, :3] - gt[0, :3], axis=1)
    lp = float(np.linalg.norm(np.diff(pred[:, :3], axis=0), axis=1).sum())
    lg = float(np.linalg.norm(np.diff(gt[:, :3], axis=0), axis=1).sum())
    return {
        prefix + "abs_trans_err": float(dt.mean()),
        prefix + "abs_trans_err_last": float(dt[-1]),
        prefix + "abs_trans_err_max": float(dt.max()),
        prefix + "rel_disp_err": float(np.abs(dp - dg).mean()),
        prefix + "yaw_err_deg": float(np.abs(dyaw).mean()),
        prefix + "yaw_err_deg_last": float(abs(dyaw[-1])),
        prefix + "path_len_pred_m": lp,
        prefix + "path_len_gt_m": lg,
        prefix + "path_len_ratio": float(lp / lg) if lg > 1e-6 else float("nan"),
        prefix + "max_disp_m": float(dg.max()),
        # 8 of these 120 windows are PURE ROTATION: XY path exactly 0.0000 m with up to 356 deg
        # of yaw (sample_104/18/25/27/81/87/90/93). Their moment field is ~zero, so any
        # translation metric is a free win there and would dilute the mean. Flagged, not dropped
        # -- they stay valid rotation tests and are the cleanest probe of whether the moment
        # channel decodes to zero.
        #
        # The test is NET DISPLACEMENT, not arc length: every window carries ~1.0 m of walking
        # z-bob (sum|dz| = 1.008 m over a 0.060 m z range), which inflates the arc length of
        # even a perfectly stationary window to ~1 m. Net displacement separates the two cleanly
        # -- 0.039 m for these versus 6.7 m for a moving window.
        prefix + "stationary": bool(float(dg.max()) < 0.5),
    }


def ate_rpe(pred, gt, horizons=(1, 4, 8, 20, 76)):
    """ATE and RPE on the recovered SE(3) poses, with NO alignment of any kind.

    camera_metrics also reports an RPE, but it sim3-aligns first (it was written for monocular
    SfM, where scale and world frame are unknown) and only at horizon 1. Here the raymap is
    metric and frame 0 is anchored on GT, so aligning would only hide error.

    RPE is additionally gauge-free: E = (T_gt,i^-1 T_gt,i+d)^-1 (T_pred,i^-1 T_pred,i+d) cancels
    both the anchor and the world frame, so it scores the predicted MOTION alone. That makes it
    the number to quote if borrowing GT's frame 0 is ever questioned -- and it is what the raymap
    actually contains, since the representation is frame-0-relative by construction.

    Horizons: 1 = per-frame step, 4 = the VAE temporal stride (one latent frame), 8/20 = drift
    over longer spans, 76 = the whole window (start -> end relative pose, a single pair)."""
    g = _ecc().geodesic_deg
    P, G = poses_to_mats(pred), poses_to_mats(gt)
    dt = np.linalg.norm(P[:, :3, 3] - G[:, :3, 3], axis=1)
    rot = np.array([g(P[i, :3, :3], G[i, :3, :3]) for i in range(len(P))])
    out = {
        "ate_trans_rmse_m": float(np.sqrt((dt ** 2).mean())),
        "ate_trans_mean_m": float(dt.mean()),
        "ate_rot_rmse_deg": float(np.sqrt((rot ** 2).mean())),
        "ate_rot_mean_deg": float(rot.mean()),
    }
    inv = np.linalg.inv
    for d in horizons:
        if d >= len(P):
            continue
        et, er = [], []
        for i in range(len(P) - d):
            E = inv(inv(G[i]) @ G[i + d]) @ (inv(P[i]) @ P[i + d])
            et.append(float(np.linalg.norm(E[:3, 3])))
            er.append(g(E[:3, :3], np.eye(3)))
        out["rpe%d_trans_m" % d] = float(np.mean(et))
        out["rpe%d_rot_deg" % d] = float(np.mean(er))
    return out


def action_metrics_from_poses(pred, gt):
    """ADE/FDE in ego-action space plus the hold-a0 baseline, reusing the inference script's own
    action_metrics. Most CARLA walking is straight, so ADE alone cannot tell you whether the
    recovered trajectory beat "keep going". Needs torch (that module imports it) -> pod only."""
    icl = _load("icl", os.path.join(COSMOS, "scripts", "inference", "infer_cosmos_lora.py"))
    import torch

    def _act(p):
        t = torch.from_numpy(np.asarray(p, dtype=np.float64))[None]
        return icl.compute_ego_actions(t)[0].numpy()

    return {("act_" + k): v for k, v in
            icl.action_metrics(_act(pred), _act(gt), float(gt[0, 5])).items()}


# ── viz ─────────────────────────────────────────────────────────────────────────────────────
def frustum_png(gt, pred, intr4, path, stride=8, near=1.5, extent=9.0, title=None):
    """View frustums of GT (blue) vs recovered (red) cameras, 3D + a BEV panel.

    Both panels are FRAME-0 RELATIVE and, by default, use FIXED axes (+-`extent` metres) so
    every sample is drawn at the same scale and the pictures can be compared at a glance --
    per-sample autoscaling makes a 0.2 m error and a 5 m error look identical. Measured over all
    120 windows, max |x-x0| and |y-y0| is 7.37 m, so +-9 m holds every GT trajectory plus the
    1.5 m frustums. Pass extent=0 to fall back to per-sample autoscale.

    BEV is not decoration: in the 3D view a lateral drift of a metre is invisible against a
    seven-metre forward run, and lateral drift is exactly what a wrong moment channel gives.

    Units are metres. (x, y, yaw) is self-consistent as a RIGHT-handed 2D frame -- measured
    corr(d_yaw, cross(heading, step)) = +0.94 on this data -- so rising yaw reads
    counter-clockwise here, matching the plucker forward map the model was trained on. CARLA's
    world is left-handed though, so left/right in this panel may be mirrored with respect to
    the RGB video."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)

    fx, fy, cx, cy = [float(v) for v in np.asarray(intr4, dtype=np.float64).ravel()[:4]]
    W, H = 2.0 * cx, 2.0 * cy                       # native res implied by the principal point
    corners = np.array([[(0 - cx) / fx, (0 - cy) / fy, 1.0],
                        [(W - cx) / fx, (0 - cy) / fy, 1.0],
                        [(W - cx) / fx, (H - cy) / fy, 1.0],
                        [(0 - cx) / fx, (H - cy) / fy, 1.0]]) * near

    fig = plt.figure(figsize=(13, 6))
    ax3 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2)
    idx = list(range(0, len(gt), max(1, stride)))
    if idx[-1] != len(gt) - 1:
        idx.append(len(gt) - 1)

    org = poses_to_mats(gt)[0][:3, 3]
    outside = False
    for poses, col, lbl in ((gt, "tab:blue", "GT"), (pred, "tab:red", "recovered")):
        mats = poses_to_mats(poses)
        C = mats[:, :3, 3] - org
        outside = outside or (extent and float(np.abs(C[:, :2]).max()) > extent)
        ax3.plot(C[:, 0], C[:, 1], C[:, 2], color=col, lw=1.2, label=lbl)
        ax2.plot(C[:, 0], C[:, 1], color=col, lw=1.2, label=lbl)
        for f in idx:
            R, t = mats[f, :3, :3], mats[f, :3, 3] - org
            q = np.einsum("ij,kj->ki", R, corners) + t
            for k in range(4):                       # apex -> corner, and the near-plane loop
                ax3.plot(*zip(t, q[k]), color=col, lw=0.5, alpha=0.7)
                ax3.plot(*zip(q[k], q[(k + 1) % 4]), color=col, lw=0.5, alpha=0.7)
            ax2.plot(*zip(t[:2], q[:, :2].mean(0)), color=col, lw=0.6, alpha=0.7)
        ax3.scatter(*C[0], color=col, s=25)
        ax2.scatter(*C[0, :2], color=col, s=25)
    ax3.text(0, 0, 0, "  t=0", fontsize=8)

    # The 3D panel ALWAYS autoscales: its job is to show which way the frustums point, and on a
    # shared +-9 m cube a 7 m walk with 1.5 m frustums collapses into an unreadable blob (the
    # window spans 0.06 m in z against 7 m in xy, so a cube wastes almost all of its volume).
    # Comparability lives in the BEV panel, which is the one with fixed axes.
    allC = np.concatenate([poses_to_mats(gt)[:, :3, 3], poses_to_mats(pred)[:, :3, 3]]) - org
    ctr, rad = allC.mean(0), max(float(np.abs(allC - allC.mean(0)).max()), near) * 1.6
    for a, i in (("x", 0), ("y", 1), ("z", 2)):
        getattr(ax3, "set_%slim" % a)(ctr[i] - rad, ctr[i] + rad)
    if extent:
        lim = (-float(extent), float(extent))
        ticks = np.arange(-float(extent) + (float(extent) % 3), float(extent) + 1e-9, 3.0)
        ax2.set_xlim(*lim); ax2.set_ylim(*lim)
        ax2.set_xticks(ticks); ax2.set_yticks(ticks)
    ax3.set_box_aspect((1, 1, 1))
    ax3.set_xlabel("x-x0 (m)"); ax3.set_ylabel("y-y0 (m)"); ax3.set_zlabel("z-z0 (m)")
    ax3.set_title("frustums (autoscaled)", fontsize=9)
    ax3.legend(fontsize=8)
    ax2.set_aspect("equal"); ax2.grid(alpha=0.3)
    ax2.set_xlabel("x-x0 (m)"); ax2.set_ylabel("y-y0 (m)")
    ax2.set_title("BEV, frame-0 relative"
                  + (" (fixed +-%g m)" % extent if extent else " (autoscale)")
                  + (" -- CLIPPED" if outside else ""), fontsize=9)
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ── VAE (s2/s3) ─────────────────────────────────────────────────────────────────────────────
def load_vae(model_name, revision, device, dtype="bf16"):
    """The Cosmos VAE alone. latents_mean/std come from vae.config only (see
    pipeline_cosmos2_5_predict.py:240-243), so no DiT, no T5 and no fine-tuned checkpoint are
    needed -- the raymap codec is the base VAE either way."""
    if SRC not in sys.path:
        sys.path.insert(0, SRC)
    import torch
    from diffusers import AutoencoderKLWan
    td = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]
    vae = AutoencoderKLWan.from_pretrained(model_name, subfolder="vae", revision=revision,
                                           torch_dtype=td).to(device).eval()
    mean = torch.tensor(vae.config.latents_mean, dtype=torch.float32, device=device)
    std = 1.0 / torch.tensor(vae.config.latents_std, dtype=torch.float32, device=device)
    shape = (1, -1, 1, 1, 1)
    return vae, mean.view(shape), std.view(shape)


def decode_raymap_rays(vae, raymap_lat, latents_mean, latents_std, moment_scale):
    """raymap latent (1, 2z, T_lat, h, w) -> (dirs, moments), each (T, H, W, 3) float64 METRIC.

    The float sibling of data_utils.decode_raymap_pixels, which is viz-only: it returns uint8
    after px/2+0.5, and 1/255 of [-1,1] at moment_scale=8 is 0.125 m of moment resolution --
    that quantization alone would floor the translation error above 0.1 m. Here the raw
    vae.decode(...).sample is kept, the direction field is renormalized (which discards all
    radial VAE error) and the moment field is multiplied back by moment_scale."""
    import torch
    with torch.no_grad():
        def _dec(lat_z):
            mu = (lat_z / latents_std) + latents_mean
            px = vae.decode(mu.to(vae.dtype)).sample                 # (1,3,T,H,W), ~[-1,1]
            return px[0].permute(1, 2, 3, 0).float().cpu().numpy().astype(np.float64)
        z = raymap_lat.shape[1] // 2
        d = _dec(raymap_lat[:, :z])
        m = _dec(raymap_lat[:, z:]) * float(moment_scale)
    d /= np.maximum(np.linalg.norm(d, axis=-1, keepdims=True), 1e-12)
    return d, m


def build_gt_latent(vae, latents_mean, latents_std, poses, intr4, args, device):
    """GT poses -> raymap latent, through the PRODUCTION build_raymap_latent (unmodified).

    convert_to_plucker touches no module state, so a SimpleNamespace carrying it unbound is a
    sufficient `dit_m` -- the transformer never gets instantiated."""
    if SRC not in sys.path:
        sys.path.insert(0, SRC)
    if COSMOS not in sys.path:
        sys.path.insert(0, COSMOS)
    import torch
    from diffusers.models.transformers.transformer_cosmos import CosmosTransformer3DModel
    from utils.data_utils import build_raymap_latent

    dit_stub = types.SimpleNamespace(
        convert_to_plucker=functools.partial(CosmosTransformer3DModel.convert_to_plucker, None))
    ext = torch.from_numpy(poses_to_mats(poses)).to(device=device, dtype=torch.float32)[None]
    intr = torch.tensor([float(v) for v in np.asarray(intr4).ravel()[:4]],
                        dtype=torch.float32, device=device)[None, None].expand(1, len(poses), 4)
    return build_raymap_latent(dit_stub, vae, ext, intr, args.height, args.width,
                               args.raymap_moment_scale, latents_mean, latents_std,
                               ray_downsample=args.ray_downsample)


# ── driver ──────────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--stage", default="s1", choices=["s1", "s2", "s3", "all"])
    ap.add_argument("--num_samples", type=int, default=0, help="0 = all")
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--num_frames", type=int, default=77)
    # NOT in the checkpoint config: a wrong value here silently yields a plausible wrong answer,
    # so the resolved grid is echoed and asserted against the .npy shape before anything runs.
    ap.add_argument("--ray_downsample", type=int, default=2)
    ap.add_argument("--raymap_moment_scale", type=float, default=8.0)
    ap.add_argument("--model_name", default="nvidia/Cosmos-Predict2.5-2B")
    ap.add_argument("--revision", default="diffusers/base/post-trained")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--intrinsics_override", default=None,
                    help="fx,fy,cx,cy. samples_meta.json holds POD-absolute seq_dir paths, so "
                         "this is the fallback when intrinsics.npy is not reachable.")
    ap.add_argument("--meta", default="samples_meta.json")
    ap.add_argument("--out_csv", default="raymap_inversion_metrics.csv")
    ap.add_argument("--save_poses", action="store_true", default=True)
    ap.add_argument("--no_save_poses", dest="save_poses", action="store_false")
    ap.add_argument("--report_actions", action="store_true",
                    help="ADE/FDE + hold-a0 baseline in ego-action space (needs torch)")
    ap.add_argument("--frustum", action="store_true")
    ap.add_argument("--frustum_stride", type=int, default=8)
    ap.add_argument("--near", type=float, default=1.5)
    ap.add_argument("--bev_extent", type=float, default=9.0,
                    help="half-width in metres of the FIXED frustum/BEV axes, shared by every "
                         "sample so the pictures are comparable. 0 = per-sample autoscale.")
    ap.add_argument("--gate_s1", type=float, default=1e-6)
    args = ap.parse_args()

    stages = ["s1", "s2", "s3"] if args.stage == "all" else [args.stage]
    Hr = args.height // args.ray_downsample
    Wr = args.width // args.ray_downsample
    t_lat = (args.num_frames - 1) // 4 + 1
    print("[cfg] ray grid %dx%d (=%d/%d), latent %dx%d, T_lat %d, moment_scale %g"
          % (Hr, Wr, args.height, args.ray_downsample, Hr // 8, Wr // 8, t_lat,
             args.raymap_moment_scale))

    meta_path = os.path.join(args.results_dir, args.meta)
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    override = ([float(v) for v in args.intrinsics_override.split(",")]
                if args.intrinsics_override else None)

    pose_files = sorted(glob.glob(os.path.join(args.results_dir, "sample_*_action_poses.npy")))
    if args.num_samples:
        pose_files = pose_files[:args.num_samples]
    if not pose_files:
        sys.exit("[fatal] no sample_*_action_poses.npy under %s" % args.results_dir)
    print("[cfg] %d samples, stages %s" % (len(pose_files), stages))
    print("[cfg] RAYMAP_CAM_FIX %s" % ("ON (opencv): undoing rot_abs @ C" if _CAM_FIX else
                                        "off (old convention)"))

    need_vae = any(s in stages for s in ("s2", "s3"))
    vae = lmean = lstd = None
    if need_vae:
        print("[vae] loading %s (%s) on %s" % (args.model_name, args.revision, args.device))
        vae, lmean, lstd = load_vae(args.model_name, args.revision, args.device, args.dtype)

    rows, fails = [], []
    for pf in pose_files:
        tag = re.search(r"(sample_\d+)_action_poses\.npy$", os.path.basename(pf)).group(1)
        idx = str(int(tag.split("_")[1]))
        gt = np.load(pf).astype(np.float64)
        if len(gt) != args.num_frames:
            print("[%s] skip: %d poses, expected %d" % (tag, len(gt), args.num_frames))
            continue

        intr = override
        if intr is None:
            sd = (meta.get(idx) or {}).get("seq_dir")
            ip = os.path.join(sd, "intrinsics.npy") if sd else None
            if ip and os.path.exists(ip):
                intr = np.load(ip).astype(np.float64).ravel()[:4].tolist()
        if intr is None:
            print("[%s] skip: no intrinsics (pass --intrinsics_override fx,fy,cx,cy)" % tag)
            continue

        row = {"sample": tag, "dir": args.results_dir, "cam_fix": int(_CAM_FIX)}
        orig_wh = (args.width, args.height)

        # s1 -- pure geometry. The gate that says the inverse itself is right.
        if "s1" in stages:
            d, m = forward_plucker_numpy(gt, intr, Hr, Wr, orig_wh)
            pa, pr, dg = invert_rays(d, m, intr, orig_wh, gt[0], args.raymap_moment_scale)
            e = pose_errors(pa, gt, "s1_")
            row.update(e)
            row.update({("s1_" + k): v for k, v in dg.items()})
            row.update({("s1_" + k): v for k, v in ate_rpe(pa, gt).items()})
            worst = max(e["s1_abs_trans_err_max"], e["s1_yaw_err_deg"])
            ok = worst < args.gate_s1 and dg["cond_G"] < 10.0
            row["s1_pass"] = bool(ok)
            print("[%s] s1 trans %.2e m  yaw %.2e deg  cond %.2f  %s"
                  % (tag, e["s1_abs_trans_err_max"], e["s1_yaw_err_deg"], dg["cond_G"],
                     "PASS" if ok else "FAIL"))
            if not ok:
                fails.append(tag)

        # s2 -- the VAE+clamp floor. s3 measures nothing this does not bound.
        if "s2" in stages:
            lat = build_gt_latent(vae, lmean, lstd, gt, intr, args, args.device)
            assert lat.shape[3] == Hr // 8 and lat.shape[2] == t_lat, \
                "[%s] latent %s vs expected (1,32,%d,%d,%d)" % (tag, tuple(lat.shape), t_lat,
                                                                Hr // 8, Wr // 8)
            d, m = decode_raymap_rays(vae, lat, lmean, lstd, args.raymap_moment_scale)
            pa, pr, dg = invert_rays(d, m, intr, orig_wh, gt[0], args.raymap_moment_scale)
            row.update(pose_errors(pa, gt, "s2_"))
            row.update({("s2_" + k): v for k, v in dg.items()})
            row.update({("s2_" + k): v for k, v in ate_rpe(pa, gt).items()})
            row.update({("s2_" + k): v for k, v in _ecc().camera_metrics(
                poses_to_mats(pa), poses_to_mats(gt)).items()})
            print("[%s] s2 floor: trans %.4f m  yaw %.3f deg  ate_rot %.3f deg  "
                  "roterr %.3f deg  clamp_frac %.4f"
                  % (tag, row["s2_abs_trans_err"], row["s2_yaw_err_deg"],
                     row["s2_ate_rot_mean_deg"], row["s2_roterr"],
                     dg.get("clamp_frac", float("nan"))))

        # s3 -- the actual prediction.
        if "s3" in stages:
            rp = pf.replace("_action_poses.npy", "_pred_raymap.npy")
            if not os.path.exists(rp):
                print("[%s] skip s3: no %s" % (tag, os.path.basename(rp)))
            else:
                import torch
                lat = torch.from_numpy(np.load(rp)).to(args.device)
                assert lat.shape[3] == Hr // 8 and lat.shape[2] == t_lat, \
                    "[%s] pred_raymap %s disagrees with --ray_downsample %d/--num_frames %d" \
                    % (tag, tuple(lat.shape), args.ray_downsample, args.num_frames)
                d, m = decode_raymap_rays(vae, lat, lmean, lstd, args.raymap_moment_scale)
                pa, pr, dg = invert_rays(d, m, intr, orig_wh, gt[0], args.raymap_moment_scale)
                row.update(pose_errors(pa, gt))
                row.update(dg)
                row.update(ate_rpe(pa, gt))
                row.update(_ecc().camera_metrics(poses_to_mats(pa), poses_to_mats(gt)))
                if args.report_actions:
                    row.update(action_metrics_from_poses(pa, gt))
                if args.save_poses:
                    np.save(os.path.join(args.results_dir, "%s_pred_poses.npy" % tag), pa)
                    np.save(os.path.join(args.results_dir, "%s_pred_poses_rel.npy" % tag), pr)
                if args.frustum:
                    frustum_png(gt, pa, intr,
                                os.path.join(args.results_dir, "%s_frustum.png" % tag),
                                stride=args.frustum_stride, near=args.near,
                                extent=args.bev_extent,
                                title="%s  ATE %.3f m / rot %.2f deg  |  RPE1 %.3f m / %.2f deg"
                                      % (tag, row["ate_trans_rmse_m"], row["ate_rot_mean_deg"],
                                         row["rpe1_trans_m"], row["rpe1_rot_deg"]))
                # the interpretation rule, applied in-row so the two can never be read apart
                if "s2_abs_trans_err" in row:
                    row["trans_above_floor"] = bool(
                        row["s2_abs_trans_err"] < row["abs_trans_err"] / 3.0)
                print("[%s] s3: ATE %.4f m (rot %.3f deg)  RPE1 %.4f m / %.3f deg  "
                      "rel_disp %.4f m"
                      % (tag, row["ate_trans_rmse_m"], row["ate_rot_mean_deg"],
                         row["rpe1_trans_m"], row["rpe1_rot_deg"], row["rel_disp_err"]))

        rows.append(row)
        _ecc()._write_csv(rows, os.path.join(args.results_dir, args.out_csv))

    # summary
    print("\n[done] %d samples -> %s"
          % (len(rows), os.path.join(args.results_dir, args.out_csv)))
    for pre, lbl in (("s1_", "s1 (geometry)"), ("s2_", "s2 (VAE floor)"), ("", "s3 (predicted)")):
        keys = [pre + k for k in ("abs_trans_err", "yaw_err_deg", "rel_disp_err")]
        if not all(any(k in r for r in rows) for k in keys):
            continue
        have = [r for r in rows if all(k in r for k in keys)]
        # translation means over MOVING windows only: the pure-rotation ones have zero GT
        # translation and would flatter the average
        sel = [r for r in have if not r.get(pre + "stationary")]
        n_st = len(have) - len(sel)
        if sel:
            a = np.array([tuple(r[k] for k in keys) for r in sel])
            print("  %-16s trans %.5f m  yaw %.4f deg  rel_disp %.5f m   (n=%d moving%s)"
                  % (lbl, a[:, 0].mean(), a[:, 1].mean(), a[:, 2].mean(), len(a),
                     ", %d stationary excluded" % n_st if n_st else ""))
        rk = [pre + k for k in ("ate_rot_mean_deg", "rpe1_trans_m", "rpe1_rot_deg")]
        if all(any(k in r for r in have) for k in rk):
            b = np.array([tuple(r[k] for k in rk) for r in have])
            print("  %-16s ate_rot %.4f deg | rpe1 %.5f m / %.4f deg   (n=%d incl stationary)"
                  % ("", b[:, 0].mean(), b[:, 1].mean(), b[:, 2].mean(), len(b)))
    if "s1" in stages:
        print("  s1 gate %g: %d/%d pass%s"
              % (args.gate_s1, len(rows) - len(fails), len(rows),
                 ("  FAILED: %s" % fails) if fails else ""))
    if "s2" in stages and "s3" in stages:
        good = [r for r in rows if r.get("trans_above_floor")]
        print("  s3 translation is above the s2 floor (>3x) in %d/%d samples%s"
              % (len(good), len(rows), "" if good else " -- report rotation/yaw only"))
    if fails:
        sys.exit(1)


if __name__ == "__main__":
    main()
