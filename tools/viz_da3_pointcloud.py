#!/usr/bin/env python3
"""Fly the camera frustum through the reconstructed point cloud, as a video.

eval_camera_control.py scores DA3's estimated trajectory against the CARLA GT poses and keeps
only numbers; it throws the predicted DEPTH away ("We use camera pose only"). This tool keeps it
and unprojects it, so you can watch the scene the generated video implies and see the frustum
travel through it next to the GT path.

Why that is worth more than the numbers: a trajectory metric cannot tell you the SCENE collapsed.
A point cloud shows both at once.

Two sources, same renderer:
  --source gt    CARLA depth/ + poses.npy      -> the instrument's noise floor, and the only
                 path that needs no GPU. Run this first: if the GT cloud already looks smeared,
                 that is the reconstruction pipeline, not the model.
  --source da3   DepthAnything3 on the generated video. Monocular, so its units are arbitrary:
                 the Sim3 scale fitted against the GT trajectory is applied to the poses AND to
                 the depth, or cloud and GT path would not share a unit.
  --source both  both clouds in ONE 3D axis. Legitimate precisely because of the line above: the
                 Sim3 already put the DA3 cloud in CARLA metres in the GT world frame, so the
                 overlay measures the generated scene against the real one instead of two
                 pictures side by side. Panels become [generated | GT | 3D].

Trajectories are a SEPARATE axis from the cloud source, and they are not alignment-equivalent:
  GT (blue)            CARLA poses -- the truth.
  DA3 (red)            from the generated pixels. 7 dof (origin, orientation, SCALE) are solved
                       from the GT, so it is graded on a curve.
  raymap (purple)      --show_raymap. invert_raymap.py's recovered poses, re-anchored on the GT
                       frame 0: ZERO dof fitted, same footing as green. Needs
                       `invert_raymap.py --stage s3` to have been run first.
  action head (green)  --show_action. sample_NN_pred_actions.npy integrated in metres and pinned
                       to the GT start pose: ZERO dof fitted. Landing on the blue path therefore
                       means strictly more here than it does for red. Green vs red is the model's
                       own self-consistency -- do its pixels and its action head agree?

Axis convention, the easy thing to get wrong: CARLA poses need the pinhole<->CARLA swap _C
(data_utils.py:302), DA3 returns OpenCV w2c and must NOT get it.

  python3 tools/viz_da3_pointcloud.py --results_dir <dir> --source gt --samples 0,1,2
  python3 tools/viz_da3_pointcloud.py --results_dir <dir> --source da3 --probe
"""
import argparse
import glob
import importlib.util
import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEPTH_CLIP_DEFAULT = 50.0
# pinhole <-> CARLA camera axes; identical to utils/data_utils.py:302
_C = np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=np.float64)


def _ecc():
    """tools/eval_camera_control.py as a module (eval_rollout.py:49-59 shim). Stdlib+numpy at
    import time, so this is cheap and pulls in umeyama_sim3 / carla_pose_to_mat / estimate_da3."""
    spec = importlib.util.spec_from_file_location("ecc", os.path.join(HERE, "eval_camera_control.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def unproject(depth, rgb, c2w, intr4, depth_clip, carla_axes, rng, n_points):
    """One frame -> (P,3) world points + (P,3) uint8 colours. Mirrors data_utils.py:296-310."""
    H, W = depth.shape
    fx, fy, cx, cy = [float(v) for v in np.asarray(intr4, dtype=np.float64).ravel()[:4]]
    if (H, W) != rgb.shape[:2]:                      # DA3 may return depth at its own resolution
        sx, sy = W / rgb.shape[1], H / rgb.shape[0]
        fx, cx, fy, cy = fx * sx, cx * sx, fy * sy, cy * sy
        from PIL import Image
        rgb = np.array(Image.fromarray(rgb).resize((W, H), Image.BILINEAR))
    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    d = depth.astype(np.float64).ravel()
    valid = (d > 0.1) & (d < depth_clip)
    if not valid.any():
        return np.zeros((0, 3)), np.zeros((0, 3), np.uint8)
    idx = np.flatnonzero(valid)
    if n_points and len(idx) > n_points:             # subsample per frame, not globally
        idx = rng.choice(idx, n_points, replace=False)
    dd = d[idx]
    pts = np.stack([(xs.ravel()[idx] - cx) * dd / fx,
                    (ys.ravel()[idx] - cy) * dd / fy,
                    dd, np.ones(len(idx))], axis=0)
    if carla_axes:
        pts = np.einsum("ij,jn->in", _C, pts)
    # einsum, not `@`: macOS Accelerate emits spurious divide-by-zero/overflow warnings on a
    # (4,4) @ (4,N) with large N. Values are identical; einsum stays quiet.
    world = np.einsum("ij,jn->in", c2w, pts)[:3].T
    return world, rgb.reshape(-1, 3)[idx]


def frustum_corners(intr4, near, wh):
    """4 image-corner rays at `near`, in camera coords (pinhole)."""
    fx, fy, cx, cy = [float(v) for v in np.asarray(intr4, dtype=np.float64).ravel()[:4]]
    W, H = wh
    return np.array([[(0 - cx) / fx, (0 - cy) / fy, 1.0],
                     [(W - cx) / fx, (0 - cy) / fy, 1.0],
                     [(W - cx) / fx, (H - cy) / fy, 1.0],
                     [(0 - cx) / fx, (H - cy) / fy, 1.0]]) * near


def draw_frustum(ax, c2w, corners, colour, carla_axes, lw=1.2, zorder=4):
    R, t = c2w[:3, :3], c2w[:3, 3]
    cc = corners if not carla_axes else (_C[:3, :3] @ corners.T).T
    q = (R @ cc.T).T + t
    for k in range(4):
        ax.plot(*zip(t, q[k]), color=colour, lw=lw, alpha=0.95, zorder=zorder)
        ax.plot(*zip(q[k], q[(k + 1) % 4]), color=colour, lw=lw, alpha=0.95, zorder=zorder)


def load_gt(seq_dir, start, n, ecc):
    poses = np.load(os.path.join(seq_dir, "poses.npy"))
    idx = list(range(start, min(start + n, len(poses))))
    c2w = np.stack([ecc.carla_pose_to_mat(*poses[i]) for i in idx])
    intr = np.load(os.path.join(seq_dir, "intrinsics.npy")).astype(np.float64).ravel()[:4]
    return c2w, intr, idx


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--source", default="gt", choices=["gt", "da3", "both"],
                    help="both = draw the CARLA and the DA3 cloud in one 3D axis "
                         "(they already share the metric GT world frame)")
    ap.add_argument("--samples", default="all",
                    help="comma-separated sample indices, or 'all' for every entry in "
                         "samples_meta.json (120 on the valbench5s manifest)")
    ap.add_argument("--max_samples", type=int, default=0,
                    help="cap after expanding --samples (0 = no cap)")
    ap.add_argument("--meta", default="samples_meta.json")
    ap.add_argument("--images_subdir", default="images")
    ap.add_argument("--depth_subdir", default="depth")
    ap.add_argument("--cloud_stride", type=int, default=8, help="unproject every Nth frame")
    ap.add_argument("--points_per_frame", type=int, default=3000)
    ap.add_argument("--depth_clip", type=float, default=DEPTH_CLIP_DEFAULT)
    ap.add_argument("--near", type=float, default=1.5, help="frustum size in metres")
    ap.add_argument("--box_margin", type=float, default=12.0,
                    help="metres of scene kept around the trajectory extent")
    ap.add_argument("--point_size", type=float, default=2.5)
    ap.add_argument("--point_alpha", type=float, default=0.75)
    ap.add_argument("--show_action", action="store_true",
                    help="overlay the action-head trajectory from sample_NN_pred_actions.npy. "
                         "Metric and anchored at the GT start -- zero dof fitted, "
                         "unlike the DA3 path whose 7 dof come from the GT.")
    ap.add_argument("--action_color", default="tab:green")
    ap.add_argument("--da3_anchor", default="gt", choices=["gt", "action", "raymap"],
                    help="what the DA3 sim3 is fitted to. 'action'/'raymap' make the "
                         "figure a GT-free self-consistency check and imply the matching "
                         "--show_* flag. raymap gives 77 correspondences to action's 20.")
    ap.add_argument("--align_mode", default="native",
                    choices=["native", "equalized", "da3", "both", "all"],
                    help="native: draw each trajectory as measured (action/raymap "
                         "0-dof metric, DA3 7-dof). equalized: sim3 action/raymap to "
                         "GT too, so every path gets the same 7 dof DA3 gets. both: render "
                         "one figure of each from a SINGLE DA3 pass. Unrelated to "
                         "--source both.")
    ap.add_argument("--show_raymap", action="store_true",
                    help="overlay the raymap-inverted trajectory from sample_NN_pred_poses.npy "
                         "(produced by invert_raymap.py --stage s3). Also metric and anchored "
                         "on the GT frame 0 -- zero dof fitted, like --show_action.")
    ap.add_argument("--raymap_color", default="#d400c8",
                    help="vivid magenta: complementary to both cloud tints "
                         "(teal/orange) and distinct from the blue GT and red DA3 "
                         "lines. tab:purple was too desaturated to read on cloud")
    ap.add_argument("--raymap_lw", type=float, default=3.4,
                    help="matches the GT line weight; the dash pattern is what "
                         "separates them, not thickness")
    ap.add_argument("--no_traj_halo", dest="traj_halo", action="store_false",
                    help="drop the white outline behind the raymap line; the halo is "
                         "what keeps it readable where it crosses dense cloud")
    ap.add_argument("--floor_csv", default="da3_floor_metrics.csv",
                    help="written by da3_floor.py; lets the DA3 line subtract the "
                         "instrument error and show what the generation added")
    ap.add_argument("--raymap_csv", default="raymap_inversion_metrics.csv",
                    help="written by invert_raymap.py --stage s3; supplies the "
                         "raymap-vs-GT numbers shown in the title")
    ap.add_argument("--raymap_poses_suffix", default="_pred_poses.npy",
                    help="swap for e.g. _pred_poses_rel.npy or an s2-floor variant")
    ap.add_argument("--vae_stride", type=int, default=4,
                    help="video frames per action step (VAE temporal stride)")
    ap.add_argument("--cloud_color", default="auto", choices=["auto", "rgb", "tint"],
                    help="auto: true colour for a single source, both flat (teal/orange) "
                         "for --source both")
    ap.add_argument("--gt_tint", default="#2f9e9e")
    ap.add_argument("--da3_tint", default="#ff6f3c")
    ap.add_argument("--no_gt_frame", dest="show_gt_frame", action="store_false",
                    help="drop the GT-frame panel (--source da3 only; gt mode never shows it)")
    ap.add_argument("--no_mirror_y", dest="mirror_y", action="store_false",
                    help="plot CARLA coords on raw right-handed axes (left/right will appear "
                         "mirrored versus the video). Default inverts y to match CARLA.")
    ap.add_argument("--no_frustum_on_top", dest="frustum_on_top", action="store_false",
                    help="let the cloud occlude the frustum (matplotlib's depth sort). Default is "
                         "to draw frustum and paths over the points.")
    ap.add_argument("--z_frac", type=float, default=0.35,
                    help="z half-range as a fraction of the xy half-range (the scene is flat)")
    ap.add_argument("--elev", type=float, default=25.0)
    ap.add_argument("--azim", type=float, default=-60.0)
    ap.add_argument("--fps", type=int, default=16)
    ap.add_argument("--frame_stride", type=int, default=1, help="render every Nth video frame")
    ap.add_argument("--out_subdir", default="viz_da3")
    ap.add_argument("--save_depth", action="store_true",
                    help="also write sample_NN_da3_depth.npy: the DA3 depth AFTER the Sim3 scale, "
                         "so it is in the same metres as the GT depth. Only the latent frames "
                         "(0,4,..) at 256^2 float16 -- that is the resolution the predicted depth "
                         "lives at, so nothing is lost, and all 77 frames at 504^2 would be 39MB "
                         "per sample. Consumed by tools/eval_depth_vs_da3.py.")
    ap.add_argument("--da3_on_gt", action="store_true",
                    help="run DA3 on the GT frames instead of the generated mp4, and save as "
                         "_da3_gt_depth.npy. This is the FLOOR for the video<->depth comparison: "
                         "how well DA3 recovers depth on this content at all. It is not optional "
                         "-- 'DA3(generated) vs GT depth' also carries the generated video "
                         "diverging from GT, so it cannot serve as that baseline.")
    ap.add_argument("--da3_fit", default="rot", choices=["rot", "pos"],
                    help="what the DA3 alignment is solved from. 'rot': orientations, "
                         "determinant free -- fixes the mirrored cloud/heading. "
                         "'pos': positions with det=+1, the old behaviour.")
    ap.add_argument("--da3_ref_view", default="first",
                    choices=["first", "middle", "saddle_balanced", "saddle_sim_range"],
                    help="which frame anchors DA3's pose estimate. Default 'first' "
                         "matches raymap/action (both anchored at frame 0); DA3's own "
                         "default is 'saddle_balanced', which every earlier run used.")
    ap.add_argument("--axis_check", action="store_true",
                    help="per layer, report whether the image bottom unprojects BELOW "
                         "the camera and the top above it. Ground-truth CARLA gives "
                         "0.60 / 3.81 at a camera height of 1.85 m; a flipped image y "
                         "swaps them.")
    ap.add_argument("--probe", action="store_true", help="print the DA3 result fields and exit")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.da3_anchor == "action":
        a.show_action = True          # it is the alignment target; loading it is not optional
    if a.da3_anchor == "raymap":
        a.show_raymap = True
    if a.align_mode != "native" and a.da3_anchor != "gt":
        # `equalized` means "everyone carries the same GT-fitted handicap". With the DA3 layer
        # anchored to a path that equalizing then MOVES, the figure would show DA3 fitted to a
        # line that is no longer drawn -- a quietly wrong picture rather than a different one.
        raise SystemExit(f"--align_mode {a.align_mode} needs --da3_anchor gt "
                         f"(got {a.da3_anchor}): equalizing moves the very path DA3 was fitted to")

    ecc = _ecc()
    meta_p = os.path.join(a.results_dir, a.meta)
    meta = json.load(open(meta_p)) if os.path.exists(meta_p) else {}
    if not meta:
        raise SystemExit(f"[fatal] {meta_p} missing")
    out_dir = os.path.join(a.results_dir, a.out_subdir)
    os.makedirs(out_dir, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    import imageio.v2 as imageio
    from PIL import Image

    if a.samples.strip().lower() in ("all", "*", ""):
        sids = sorted(meta.keys(), key=lambda k: int(k))
    else:
        sids = [x.strip() for x in a.samples.split(",") if x.strip()]
    if a.max_samples:
        sids = sids[:a.max_samples]
    print(f"[viz] {len(sids)} samples, source={a.source} -> {out_dir}")
    for n_done, sid in enumerate(sids, 1):
        if sid not in meta:
            print(f"  [skip] sample {sid} not in {a.meta}")
            continue
        info = meta[sid]
        seq_dir, start = info["seq_dir"], int(info["start_frame"])
        n = int(info.get("num_frames", 77))
        gt_c2w, intr, fidx = load_gt(seq_dir, start, n, ecc)
        img_dir = os.path.join(seq_dir, a.images_subdir)
        gt_rgb = [np.array(Image.open(os.path.join(img_dir, f"{i:06d}.png")).convert("RGB"))
                  for i in fidx]

        ray = load_raymap_traj(a, a.results_dir, sid, ecc) if a.show_raymap else None
        if ray is not None:
            print(f"    raymap: {len(ray['c2w'])} poses (1/frame), {ray['label']}")
        act = load_action_traj(a, a.results_dir, sid, ecc) if a.show_action else None
        if act is not None:
            # The comparison trap this tool kept setting: the action ADE is a ZERO-dof number and
            # the DA3 one is fitted with SEVEN, so printing them side by side ranks them backwards.
            # Giving the action path the same sim3 makes the two directly comparable -- and the
            # gap between the two action numbers is exactly how much of its error is a global
            # scale/rotation drift that the alignment forgives.
            _m = act["fidx"] < len(gt_c2w)
            _A, _G = act["c2w"][_m, :3, 3], gt_c2w[act["fidx"][_m], :3, 3]
            if len(_A) >= 3:
                _s, _R, _t = ecc.umeyama_sim3(_A, _G, True)
                _d = np.linalg.norm(((_s * (_R @ _A.T)).T + _t)[:, :2] - _G[:, :2], axis=1)
                _raw = np.linalg.norm(_A[:, :2] - _G[:, :2], axis=1).mean()
                _pct = (f"; {100 * (1 - _d.mean() / _raw):.0f}% absorbed by the fit"
                        if _raw > 1e-9 else "")
                act["label"] += (f"\naction vs GT:  {_d.mean():.2f} m   "
                                 f"(7-dof -> GT{_pct})")

        if act is not None:
            print(f"    action: {len(act['c2w'])} keys @ frames "
                  f"{act['fidx'][0]}..{act['fidx'][-1]} (stride {a.vae_stride}), {act['label']}")

        # The DA3 layer is fitted to this. Built BEFORE the layers because build_layer needs it.
        if a.da3_anchor in ("action", "raymap"):
            tr = act if a.da3_anchor == "action" else ray
            if tr is None:
                print(f"  [skip] sample {sid}: --da3_anchor {a.da3_anchor} but no such trajectory")
                continue
            # raymap has one pose per video frame, so its correspondence indices are just 0..F-1;
            # action only has a key every vae_stride frames and carries them in "fidx".
            idx = tr.get("fidx")
            if idx is None:
                idx = np.arange(len(tr["c2w"]))
            align = (tr["c2w"][:, :3, 3], idx, a.da3_anchor)
            align_R = tr["c2w"][:, :3, :3] @ _C[:3, :3]
        else:
            align = (gt_c2w[:, :3, 3], np.arange(len(gt_c2w)), "GT")
            align_R = gt_c2w[:, :3, :3] @ _C[:3, :3]   # GT 의 pinhole->world (det -1)

        kinds = ["gt", "da3"] if a.source == "both" else [a.source]
        layers = []
        for kind in kinds:
            L = build_layer(kind, a, ecc, sid, seq_dir, fidx, gt_c2w, intr, gt_rgb, out_dir,
                            align, align_R)
            if L == "probe":
                return
            if L is None:
                layers = None
                break
            accumulate(L, a, np.random.default_rng(a.seed))
            if L["P"] is None:
                print(f"  [skip] sample {sid}: empty {kind} cloud")
                layers = None
                break
            layers.append(L)
        if not layers:
            continue
        da3 = next((L for L in layers if L["kind"] == "da3"), None)
        if a.axis_check:
            for L in layers:
                r = axis_report(L)
                if r is None:
                    print(f"    [axis] {L['kind']}: no depth"); continue
                zb, zt, zc = r
                # The bottom half is the robust test -- the ground is always close enough to have
                # depth. The top can legitimately be empty (open sky), and reporting that as a
                # flip would cry wolf on every frame facing upward.
                if not np.isfinite(zb):
                    ok = "no ground data"
                elif zb >= zc:
                    ok = "FLIPPED (ground above camera)"
                elif not np.isfinite(zt):
                    ok = "OK (ground below; sky had no depth)"
                elif zt <= zc:
                    ok = "SUSPECT (top not above camera)"
                else:
                    ok = "OK"
                _t = f"{zt:6.2f}" if np.isfinite(zt) else "   n/a"
                print(f"    [axis] {L['kind']:4s} cam z {zc:6.2f} | image bottom -> {zb:6.2f} "
                      f"| image top -> {_t}   {ok}")
        for L in layers:
            print(f"  [{n_done}/{len(sids)}] sample {sid}/{L['kind']}: "
                  f"{len(L['P'])} points, {len(L['c2w'])} poses, {L['label']}")

        # One figure per alignment policy. Inside a single figure every path then carries the
        # same handicap, so "who sits closer to blue" is a fair question there; across the two
        # figures the difference is exactly what the 7 dof buy. Built from one DA3 pass -- DA3 is
        # the expensive step and it does not depend on the policy.
        views = []
        if a.align_mode in ("native", "both", "all"):
            views.append(("", act, ray,
                          "alignment: action/raymap 0-dof | DA3 7-dof -> "
                          + align[2], "0-dof"))
        if a.align_mode in ("equalized", "both", "all"):
            views.append(("_align-equalized",
                          sim3_onto(act, gt_c2w, ecc, "GT"), sim3_onto(ray, gt_c2w, ecc, "GT"),
                          "alignment: ALL trajectories 7-dof -> GT", "7-dof -> GT"))
        if a.align_mode in ("da3", "all") and da3 is not None:
            # The DA3 reconstruction is the reference here, so it does NOT move -- the cloud stays
            # where it is and the 0-dof paths are laid into it. GT-free: this figure never uses the
            # ground truth to place anything except the (unmoved) blue reference line.
            _D = da3["c2w"]
            views.append(("_align-da3",
                          sim3_onto(act, _D, ecc, "DA3"), sim3_onto(ray, _D, ecc, "DA3"),
                          "alignment: action/raymap 7-dof -> DA3 (DA3 field is the reference; "
                          "cloud unmoved)", "7-dof -> DA3"))
        for _vtag, act, ray, _apolicy, _leg in views:
            # View box: the trajectory alone is ~12 m while the cloud runs out to depth_clip, so a
            # trajectory-sized box hides almost every point. Take the path extent, widen it by
            # --box_margin, and clip every cloud to that -- distant facades stay out of the way. The
            # box comes from the GT path in all modes, so the two sources are framed identically.
            ctr = gt_c2w[:, :3, 3].mean(0)
            _ext = [np.abs(gt_c2w[:, :3, 3] - ctr).max()]
            # A badly wrong action path can leave the GT extent; the box has to follow it or the very
            # failure the overlay exists to show would be cropped out of frame.
            if act is not None:
                _ext.append(np.abs(act["c2w"][:, :3, 3] - ctr).max())
            if ray is not None:
                _ext.append(np.abs(ray["c2w"][:, :3, 3] - ctr).max())
            if da3 is not None:
                _ext.append(np.abs(da3["c2w"][:, :3, 3] - ctr).max())
            rad = max(float(max(_ext)), 1.0) + a.box_margin
            for L in layers:
                L["vis"] = (np.abs(L["P"] - ctr) < rad).all(1)
            if sum(int(L["vis"].sum()) for L in layers) < 100:   # degenerate box -> fit the cloud
                allP = np.concatenate([L["P"] for L in layers])
                if len(allP) == 0:
                    # The sim3 collapsed (Umeyama returned a NEGATIVE scale, e.g. s=-1.627 on a
                    # near-stationary window), so every point was culled and percentile would
                    # raise IndexError on the empty array. da3_poses.npy is written earlier, in
                    # build_layer, so the metrics are already safe -- only this figure is lost.
                    # Keep the existing centre/radius and carry on rather than killing the run.
                    print(f"    [skip figure] sample {sid}: empty cloud, degenerate sim3; "
                          f"da3_poses.npy already saved")
                else:
                    lo, hi = np.percentile(allP, [2, 98], axis=0)
                    ctr = (lo + hi) / 2; rad = float((hi - lo).max()) / 2
                    for L in layers:
                        L["vis"] = (np.abs(L["P"] - ctr) < rad).all(1)
            print("    box +-{:.1f} m around {}; {}".format(
                rad, np.round(ctr, 1),
                ", ".join(f"{L['kind']} {int(L['vis'].sum())}/{len(L['P'])} pts" for L in layers)))

            # Left panel = what was actually fed to the reconstruction. --source gt has only the GT
            # video, the other two modes have the generated one; the GT panel is then a second column
            # so the two videos can be read against each other frame by frame.
            # ONE scatter, not one per layer. Two collections would make draw order decide who is
            # visible -- the second painted a flat sheet over the first, which is exactly what the
            # overlay must not do. Inside a single Path3DCollection matplotlib depth-sorts the markers
            # (art3d: z_markers_idx = argsort(vzs)[::-1]), so the clouds interleave properly. Flat
            # colours are expanded to per-point RGB here so both kinds can share one `c` array.
            import matplotlib.colors as mcolors
            from matplotlib.lines import Line2D
            from matplotlib import patheffects
            SP = np.concatenate([L["P"][L["vis"]] for L in layers])
            SC = np.concatenate([
                L["C"][L["vis"]] if L["flat"] is None else
                np.tile(np.asarray(mcolors.to_rgb(L["flat"]), np.float32), (int(L["vis"].sum()), 1))
                for L in layers])
            # A 2.5 pt scatter is invisible in a legend, so the clouds get proper marker proxies.
            proxies = [Line2D([], [], marker="o", ms=6, ls="", color=(L["flat"] or "0.45"),
                              label=f"cloud: {L['kind']}") for L in layers]

            left_rgb = da3["rgbs"] if da3 is not None else gt_rgb
            left_name = "generated" if da3 is not None else "GT"
            show_gt = a.show_gt_frame and da3 is not None
            n_render = max(len(L["c2w"]) for L in layers)
            # Two anchors are two different measurements; sharing a filename would let the second
            # run silently overwrite the first and leave no way to tell them apart afterwards.
            _atag = "" if a.da3_anchor == "gt" else f"_anchor-{a.da3_anchor}"
            out_mp4 = os.path.join(out_dir,
                                   f"sample_{int(sid):02d}_{a.source}{_atag}{_vtag}_pointcloud.mp4")
            with imageio.get_writer(out_mp4, fps=a.fps, macro_block_size=1) as w:
                for t_ in range(0, n_render, max(1, a.frame_stride)):
                    ncol = 3 if show_gt else 2
                    fig = plt.figure(figsize=(6 * ncol, 6))
                    gs = fig.add_gridspec(1, ncol, width_ratios=([1, 1, 2.0] if show_gt else [1, 1.5]))
                    ax0 = fig.add_subplot(gs[0, 0])
                    ax0.imshow(left_rgb[min(t_, len(left_rgb) - 1)])
                    ax0.set_axis_off(); ax0.set_title(f"{left_name} frame {t_}", fontsize=9)
                    if show_gt:
                        axg = fig.add_subplot(gs[0, 1])
                        axg.imshow(gt_rgb[min(t_, len(gt_rgb) - 1)])
                        axg.set_axis_off(); axg.set_title(f"GT frame {t_}", fontsize=9)
                    # computed_zorder=False (mpl>=3.5): 3D axes otherwise sort every artist by depth,
                    # which buries the frustum inside the cloud. With it off, the manual zorder below
                    # wins and the frustum/paths always draw on top of the points.
                    ax = fig.add_subplot(gs[0, ncol - 1], projection="3d",
                                         computed_zorder=not a.frustum_on_top)
                    ax.scatter(SP[:, 0], SP[:, 1], SP[:, 2], c=SC, s=a.point_size,
                               alpha=a.point_alpha, linewidths=0, depthshade=False, zorder=1)
                    # --source gt reconstructs FROM the GT poses, so there is no second trajectory to
                    # show: drawing it again in red just hid the blue one under an identical line and
                    # gave the legend two entries for the same thing.
                    g = gt_c2w[:, :3, 3]
                    ax.plot(g[:, 0], g[:, 1], g[:, 2], color="tab:blue", lw=3.4,
                            label="GT (CARLA poses)", zorder=3)
                    draw_frustum(ax, gt_c2w[min(t_, len(gt_c2w) - 1)],
                                 frustum_corners(intr, a.near, (gt_rgb[0].shape[1],
                                                                gt_rgb[0].shape[0])),
                                 "tab:blue", True)
                    if da3 is not None:
                        e = da3["c2w"][:, :3, 3]
                        ax.plot(e[:, 0], e[:, 1], e[:, 2], color="tab:red", lw=1.8,
                                label=f"DA3 (7-dof -> {align[2]})", zorder=4)
                        dr = da3["rgbs"][0]
                        draw_frustum(ax, da3["c2w"][min(t_, len(da3["c2w"]) - 1)],
                                     frustum_corners(da3["intr"], a.near,
                                                     (dr.shape[1], dr.shape[0])),
                                     "tab:red", False)
                    if ray is not None:
                        # One pose per video frame and no alignment, so it goes straight into the
                        # same CARLA world as the blue GT path -- same footing as the green action
                        # path, and a harder test than the sim3-fitted red one.
                        r_ = ray["c2w"][:, :3, 3]
                        # A thin line loses to a few thousand scatter points wherever it crosses
                        # them, so this one gets a white outline: the halo, not the colour, is
                        # what keeps it readable inside the cloud. Same weight as the
                        # GT line -- the dash pattern is what tells them apart.
                        _pe = ([patheffects.withStroke(linewidth=a.raymap_lw + 2.2,
                                               foreground="white")]
                               if a.traj_halo else None)
                        ax.plot(r_[:, 0], r_[:, 1], r_[:, 2], color=a.raymap_color,
                                lw=a.raymap_lw, ls="-.",
                                label=f"raymap ({_leg})", zorder=5, path_effects=_pe)
                        draw_frustum(ax, ray["c2w"][min(t_, len(ray["c2w"]) - 1)],
                                     frustum_corners(intr, a.near, (gt_rgb[0].shape[1],
                                                                    gt_rgb[0].shape[0])),
                                     a.raymap_color, True)
                    if act is not None:
                        # Metric and unaligned, so it is drawn straight into the same CARLA world as
                        # the blue GT path. Its 20 keys are 4 video frames apart (VAE stride), which
                        # is why it looks coarser than the 77-point DA3 path -- that is sampling, not
                        # error. The frustum steps to the key nearest the frame being shown.
                        a_ = act["c2w"][:, :3, 3]
                        ax.plot(a_[:, 0], a_[:, 1], a_[:, 2], color=a.action_color, lw=1.8,
                                ls="--", marker="o", ms=3, label=f"action head ({_leg})",
                                zorder=4)
                        ki = int(np.clip(round(t_ / max(1, a.vae_stride)), 0, len(act["c2w"]) - 1))
                        draw_frustum(ax, act["c2w"][ki],
                                     frustum_corners(intr, a.near, (gt_rgb[0].shape[1],
                                                                    gt_rgb[0].shape[0])),
                                     a.action_color, True)
                    # The scene is flat -- a cube would spend half the panel on empty sky, so z gets
                    # its own shorter half-range and the box aspect matches it (not isotropic).
                    zr = rad * a.z_frac
                    ax.set_xlim(ctr[0] - rad, ctr[0] + rad); ax.set_ylim(ctr[1] - rad, ctr[1] + rad)
                    ax.set_zlim(ctr[2] - zr, ctr[2] + zr)
                    # CARLA/UE4 is LEFT-handed (x fwd, y RIGHT, z up); matplotlib axes are
                    # right-handed, so plotting CARLA coords straight mirrors left and right. Verified
                    # against pixels: over this window yaw goes +43.9 deg and image content slides
                    # LEFT, i.e. the ego turns RIGHT -- but on un-inverted axes the path reads
                    # counter-clockwise (a left turn). Flipping y makes the display left-handed too,
                    # so cloud, path and frustum all match what the video shows.
                    if a.mirror_y:
                        ax.invert_yaxis()
                    ax.set_box_aspect((1, 1, a.z_frac)); ax.view_init(elev=a.elev, azim=a.azim)
                    ax.set_xlabel("x (m)")
                    ax.set_ylabel("y (m)" + (" [inv]" if a.mirror_y else "")); ax.set_zlabel("z (m)")
                    ax.legend(handles=proxies + ax.get_legend_handles_labels()[0],
                              fontsize=7, loc="upper right")
                    ax.set_title("\n".join(x["label"] for x in layers)
                                 + (f"\n{act['label']}" if act is not None else "")
                                 + (f"\n{ray['label']}" if ray is not None else "")
                                 + "".join(f"\n{x}" for x in
                                            (da3_vs(da3, ray, a.da3_anchor),
                                             da3_vs(da3, act, a.da3_anchor)) if x)
                                 + f"\n{_apolicy}",
                                 fontsize=9)
                    fig.tight_layout()
                    fig.canvas.draw()
                    w.append_data(np.asarray(fig.canvas.buffer_rgba())[:, :, :3])
                    if t_ == 0:
                        fig.savefig(out_mp4.replace(".mp4", "_t0.png"), dpi=110)
                    plt.close(fig)
            print(f"  -> {out_mp4}")


def integrate_action_poses(pred_actions, gt_poses, pose_to_mat, vae_stride=4):
    """Predicted (dx_fwd, dy_lat, d_yaw) -> CARLA-world 4x4 poses. NO alignment of any kind.

    This is the same integration inference itself does
    (infer_cosmos_lora.py:52 `integrate_ego_actions`), extended to also return the running yaw so
    a frustum can be drawn. Two things must match the repo or the comparison is dishonest:

      * `a_0` is GIVEN to the model, not predicted, so it is dropped -- exactly what
        action_metrics (:89) and _save_trajectory_plot (:269) do when they score/plot.
      * the integration is seeded with the GT start yaw, so the result lands in CARLA world by
        adding the GT start position. Nothing is fitted to the GT path.

    That 0-dof anchoring is the whole point: the DA3 trajectory gets 7 dof solved from the GT
    (origin, orientation, scale), this one gets none, so sitting on the GT path means far more
    here than it does there.

    One deliberate difference from the repo function: yaw is accumulated in float64. The repo
    seeds `yaw` with a Python float and adds float32 deltas, which under NumPy 2's NEP 50 weak
    promotion collapses the accumulator to float32 -- and would NOT under NumPy 1, so matching it
    exactly would make this tool's output depend on the numpy version. Measured gap over all 40
    saved samples: 2.8e-06 m, against an ADE of ~0.85 m. Irrelevant to the picture, worth knowing
    if a number here is ever compared digit-for-digit with sample_NN_traj.png.

    Returns (M,4,4) poses and the video frame index of each, or None if the shapes do not line up.
    """
    acts = np.asarray(pred_actions, np.float64)
    if acts.ndim != 2 or acts.shape[1] != 3 or len(acts) < 2:
        return None, None
    x0, y0, z0, yaw0 = gt_poses[0, 0], gt_poses[0, 1], gt_poses[0, 2], gt_poses[0, 5]
    pos, yaw = np.zeros(2), float(yaw0)
    xy, yaws = [pos.copy()], [yaw]
    for dx_fwd, dy_lat, d_yaw in acts[1:]:                      # a_0 dropped: it is conditioning
        r = np.deg2rad(yaw)
        pos = pos + np.array([dx_fwd * np.cos(r) - dy_lat * np.sin(r),
                              dx_fwd * np.sin(r) + dy_lat * np.cos(r)])
        yaw += d_yaw
        xy.append(pos.copy()); yaws.append(yaw)
    xy = np.asarray(xy) + np.array([x0, y0])                 # start-relative -> CARLA world
    # One action step spans one latent frame = `vae_stride` video frames.
    fidx = np.arange(0, len(xy)) * vae_stride
    keep = fidx < len(gt_poses)
    xy, yaws, fidx = xy[keep], np.asarray(yaws)[keep], fidx[keep]
    # z / roll / pitch are NOT predicted -- the head outputs 3 numbers. GT values are borrowed so
    # the path sits on the ground instead of floating; only x, y and yaw are the model's own.
    out = np.stack([pose_to_mat(xy[i, 0], xy[i, 1], gt_poses[fidx[i], 2],
                                gt_poses[fidx[i], 3], gt_poses[fidx[i], 4], yaws[i])
                    for i in range(len(xy))])
    return out, fidx


def _csv_row(path, sample):
    """One row of a metrics CSV as {col: float}, or {} if the file or row is not there. Columns
    that are not numeric (sample, dir, error) are dropped rather than raising."""
    if not os.path.exists(path):
        return {}
    import csv
    try:
        for r in csv.DictReader(open(path)):
            if r.get("sample") == sample:
                out = {}
                for k, v in r.items():
                    try:
                        out[k] = float(v)
                    except (TypeError, ValueError):
                        pass
                return out
    except Exception as e:
        print(f"  [warn] {os.path.basename(path)} unreadable: {e}")
    return {}


def load_raymap_traj(a, results_dir, sid, ecc):
    """sample_NN_pred_poses.npy -> world poses. Simpler than the action path: nothing is
    integrated, because invert_raymap.py already hands back CARLA poses.

    invert_raymap.py:164-169 re-anchors the recovered poses on the GT frame-0 pose
    (`R_abs = R0 R_rel`, `t_abs = R0 o + t0`), so like the action head this trajectory has ZERO
    dof fitted to the GT -- unlike the DA3 one, whose origin, orientation and scale are all
    solved from it. And because decode_raymap_rays (:412) decodes to pixel resolution, there is
    one pose per VIDEO frame, so no stride mapping is needed either.

    The file is NOT produced by inference: it comes from `invert_raymap.py --stage s3`, which
    must be run on the results dir first.
    """
    base = os.path.join(results_dir, f"sample_{int(sid):02d}")
    pp = base + a.raymap_poses_suffix
    if not os.path.exists(pp):
        print(f"  [raymap] sample {sid}: {os.path.basename(pp)} 없음 -> 건너뜀 "
              f"(먼저 `invert_raymap.py --results_dir ... --stage s3`)")
        return None
    P = np.load(pp)
    if P.ndim != 2 or P.shape[1] != 6:
        print(f"  [raymap] sample {sid}: {os.path.basename(pp)} 모양 {P.shape}, (F,6) 아님 -> 건너뜀")
        return None
    poses = np.stack([ecc.carla_pose_to_mat(*row) for row in P])
    # invert_raymap.py already scores this against the GT in BOTH handicaps and writes them to
    # raymap_inversion_metrics.csv (:537 keys rows as "sample_NN"). Recomputing here would risk
    # disagreeing with the CSV, so the numbers are read from it:
    #   ate_trans_rmse_m / ate_rot_mean_deg -> ate_rpe, "NO alignment of any kind" (:257)
    #   ade / fde                           -> camera_metrics, which sim3-aligns WITH scale (:591)
    # The second is the only one comparable to DA3's; the first is the one that can still see a
    # systematic moment-scale error, which sim3 would silently absorb (:217).
    # One vocabulary everywhere: "0-dof" = nothing fitted, "7-dof -> <target>" = a sim3 solved
    # against <target>. Never "sim3-aligned" on its own -- the target IS the meaning, and the same
    # path fitted to GT vs to DA3 answers different questions with different numbers.
    lines = ["raymap"]
    m = _csv_row(os.path.join(results_dir, a.raymap_csv), f"sample_{int(sid):02d}")
    if m:
        # ate_trans_mean_m (ate_rpe, unaligned) and transerr (camera_metrics, sim3) are BOTH mean
        # 3D translation error, so the ratio between them is meaningful. ate_trans_rmse_m / ade
        # are not a valid pair -- rmse vs mean, 3D vs 2D.
        u, al = m.get("ate_trans_mean_m"), m.get("transerr")
        if u is not None:
            lines.append(f"raymap vs GT:  {u:.2f} m   (0-dof, nothing forgiven)")
        if al is not None:
            pct = f"; {100 * (1 - al / u):.0f}% absorbed by the fit" if u and u > 1e-9 else ""
            lines.append(f"raymap vs GT:  {al:.2f} m   (7-dof -> GT{pct})")
        if "rpe1_trans_m" in m:
            lines.append(f"raymap vs GT:  RPE1 {m['rpe1_trans_m']:.3f} m"
                         + (f" / {m['rpe1_rot_deg']:.2f} deg" if "rpe1_rot_deg" in m else "")
                         + "   (per-step, anchor-free)")
    lbl = "\n".join(lines)
    return {"c2w": poses, "label": lbl, "name": "raymap"}


def load_action_traj(a, results_dir, sid, ecc):
    """sample_NN_{pred_actions,gt_poses}.npy -> world poses. Written by inference at
    infer_cosmos_lora.py:1548/:1102, so this costs nothing to read -- no re-inference."""
    import json
    base = os.path.join(results_dir, f"sample_{int(sid):02d}")
    pa, gp = base + "_pred_actions.npy", base + "_gt_poses.npy"
    if not (os.path.exists(pa) and os.path.exists(gp)):
        print(f"  [action] sample {sid}: {os.path.basename(pa)} / _gt_poses.npy 없음 -> 건너뜀")
        return None
    poses, fidx = integrate_action_poses(np.load(pa), np.load(gp),
                                         ecc.carla_pose_to_mat, a.vae_stride)
    if poses is None:
        print(f"  [action] sample {sid}: pred_actions 모양이 (T,3) 이 아님 -> 건너뜀")
        return None
    m = {}
    mp = base + "_metrics.json"
    if os.path.exists(mp):
        try:
            m = json.load(open(mp))
        except Exception:
            m = {}
    lbl = "action head"
    if "ade" in m:
        lbl += f"\naction vs GT:  {m['ade']:.2f} m   (0-dof, nothing forgiven)"
    return {"c2w": poses, "fidx": fidx, "label": lbl, "name": "action head"}


def da3_vs(da3, tr, anchor):
    """`DA3 vs <traj>` distance, measured where the DA3 layer already sits -- NO refit onto the
    other path. Refitting would hand 7 dof to absorbing exactly the global scale/rotation drift
    that is these heads' likeliest failure (invert_raymap.py:217 makes the same point about sim3
    hiding a moment-scale error), which is why --da3_anchor always reports a smaller number.

    Recomputed per alignment view: in the `equalized` figure the other path has moved, so a value
    carried over from the native figure would be describing lines that are no longer there.
    """
    if da3 is None or tr is None or tr.get("name") == anchor:
        return None
    idx = tr.get("fidx")
    idx = np.arange(len(tr["c2w"])) if idx is None else idx
    m = idx < len(da3["c2w"])
    if not m.any():
        return None
    d = np.linalg.norm(da3["c2w"][idx[m], :3, 3] - tr["c2w"][m, :3, 3], axis=1)
    return (f"DA3 vs {tr.get('name', 'traj')}:  {d.mean():.2f} m   "
            f"(0-dof, no refit)")


def sim3_onto(L, tgt_c2w, ecc, tgt_name="GT"):
    """Sim3-fit a 0-dof trajectory onto another path, for the non-native figures.

    Two targets, two questions:
      tgt=GT   the `equalized` figure -- give this path the SAME 7 dof DA3 already gets, so
               "who sits closer to the truth" becomes a fair question inside one picture.
      tgt=DA3  the `da3` figure -- lay this path into the field the generated pixels imply.
               Note the direction: --da3_anchor moves DA3 (and its point cloud) onto this path,
               this moves the path onto DA3 and leaves the cloud where it is. Same disagreement,
               but only here does the reconstructed scene stay put as the reference.

    Not a correction -- a handicap match. The metric original is what you would actually deploy;
    this copy exists only so that "who sits closer to the blue path" is a fair question inside one
    picture. The caller keeps the original, so both figures can be produced from one load.
    """
    if L is None:
        return None
    idx = L.get("fidx")
    idx = np.arange(len(L["c2w"])) if idx is None else idx
    m = idx < len(tgt_c2w)
    if int(m.sum()) < 3:
        return L
    s, R, t = ecc.umeyama_sim3(L["c2w"][m, :3, 3], tgt_c2w[idx[m], :3, 3], True)
    C = L["c2w"].copy()
    for i in range(len(C)):
        C[i][:3, 3] = s * (R @ C[i][:3, 3]) + t
        C[i][:3, :3] = R @ C[i][:3, :3]
    d = np.linalg.norm(C[m, :2, 3] - tgt_c2w[idx[m], :2, 3], axis=1)
    out = dict(L)
    out["c2w"] = C
    out["label"] = (f"{L.get('name', 'traj')} vs {tgt_name}:  {d.mean():.2f} m   "
                    f"(7-dof -> {tgt_name}, s={s:.3f})")
    return out


def axis_report(L, frac=0.25, fidx=None):
    """Does the bottom of the image land BELOW the camera in world z, and the top above?

    The simplest possible check of a layer's image-axis convention, and the one that settled the
    CARLA side: with the correct convention the ground (image bottom) sits under the camera and
    the sky (image top) above it; with a flipped image y they swap. It needs no second frame, no
    alignment and no reprojection -- which is exactly why it is trustworthy where a reprojection
    test is not.

    Returns (z_bottom, z_top, z_cam) in world metres, or None if the layer has no usable depth.
    """
    if not L.get("depths"):
        return None
    i = (fidx if fidx is not None else
         next((k for k, d in enumerate(L["depths"]) if d is not None), None))
    if i is None or L["depths"][i] is None:
        return None
    d = np.asarray(L["depths"][i], np.float64)
    H, W = d.shape
    fx, fy, cx, cy = [float(v) for v in np.asarray(L["intr"], np.float64).ravel()[:4]]
    if (H, W) != tuple(np.asarray(L["rgbs"][i]).shape[:2]):
        sx, sy = W / L["rgbs"][i].shape[1], H / L["rgbs"][i].shape[0]
        fx, cx, fy, cy = fx * sx, cx * sx, fy * sy, cy * sy
    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    out = []
    # The top strip is usually sky, so a near clip that suits the ground leaves it empty. Give it
    # a far one; the bottom keeps the near clip that makes the ground unambiguous.
    for rows, far in ((slice(int(H * (1 - frac)), H), 30.0), (slice(0, int(H * frac)), 120.0)):
        m = np.zeros_like(d, bool); m[rows, :] = True
        m &= (d > 0.5) & (d < far)
        idx = np.flatnonzero(m.ravel())
        if len(idx) < 100:
            out.append(float("nan")); continue
        dd = d.ravel()[idx]
        p = np.stack([(xs.ravel()[idx] - cx) * dd / fx,
                      (ys.ravel()[idx] - cy) * dd / fy, dd, np.ones(len(idx))], 0)
        if L["carla_axes"]:
            p = np.einsum("ij,jn->in", _C, p)
        out.append(float(np.median(np.einsum("ij,jn->in", L["c2w"][i], p)[2])))
    return out[0], out[1], float(L["c2w"][i][2, 3])


def build_layer(kind, a, ecc, sid, seq_dir, fidx, gt_c2w, intr_gt, gt_rgb, out_dir,
                align, tgt_R=None):
    """One (poses + depth) source, resolved into the SAME metric world frame as the GT path.

    That shared frame is the whole reason --source both is a legitimate overlay and not two
    pictures stacked in one axis: the GT layer is in CARLA metres by construction, and the DA3
    layer is put there by one Sim3 whose scale is applied to the poses AND to the depth.

    Returns the layer dict, None to skip the sample, or "probe" when --probe already printed.
    """
    if kind == "gt":
        depths, dd = [], os.path.join(seq_dir, a.depth_subdir)
        for i in fidx:
            p = os.path.join(dd, f"{i:06d}.npy")
            depths.append(np.load(p).astype(np.float32) if os.path.exists(p) else None)
        if all(d is None for d in depths):
            raise SystemExit(f"[fatal] no depth under {dd}")
        return dict(kind="gt", c2w=gt_c2w, depths=depths, rgbs=gt_rgb, intr=intr_gt,
                    carla_axes=True, label="CARLA depth @ GT poses")

    if a.da3_on_gt:
        # The floor pass: DA3 on the real frames. gt_rgb is already loaded for the "gt" layer,
        # so this costs nothing extra and guarantees both passes see the identical frame set.
        frames = list(gt_rgb)
    else:
        vp = os.path.join(a.results_dir, f"sample_{int(sid):02d}.mp4")
        if not os.path.exists(vp):
            print(f"  [skip] {vp} missing")
            return None
        frames = ecc.extract_frames(vp)
    pred = _da3_infer(frames, a.probe, a.da3_ref_view)
    if a.probe:
        return "probe"
    c2w, depths, intr_da3, confs = pred
    # Monocular -> arbitrary units. One Sim3 puts poses AND depth in metres in a shared world, so
    # the layers can occupy one figure. WHAT it is fitted to decides what the picture measures:
    #   --da3_anchor gt      the usual reading -- how close the generated video's camera motion
    #                        is to the truth, with origin/orientation/scale forgiven.
    #   --da3_anchor action  fits the model's own action head instead. The GT never enters, so
    #                        the residual is pure self-consistency: do the pixels this model
    #                        produced agree with the motion its action head predicted?
    tgt_C, tgt_idx, tgt_name = align
    m = tgt_idx < len(c2w)
    src, dst = c2w[tgt_idx[m], :3, 3], tgt_C[m]
    if len(src) < 3:
        print(f"  [skip] sample {sid}: only {len(src)} correspondences to fit a sim3 against "
              f"{tgt_name}")
        return None
    if a.da3_fit == "rot" and tgt_R is not None:
        # Fit the rotation to the ORIENTATIONS, with the determinant left free; then solve scale
        # and translation for the positions under that rotation.
        #
        # umeyama_sim3 forces det=+1 and sees only positions. CARLA's pinhole->world map is
        # improper (det(M_gt @ _C) = -1), DA3's is proper, and these walking paths are nearly
        # straight -- so a reflection costs the position fit almost nothing and the solver picked
        # whichever, leaving the orientations mirrored. Measured over 97 samples, the rotation-fit
        # determinant is -1 in 97/97.
        #
        # What it buys (97 samples): down axis 144.0 -> 1.8 deg, right 43.4 -> 4.8, forward
        # 7.4 -> 5.5, and turn-direction agreement with pixel-measured image motion 41% -> 82%
        # (GT itself scores 91%). Position error goes 0.233 -> 0.350 m, which is the honest
        # number: the old 0.233 came from forcing positions to fit under a mirrored orientation.
        G = tgt_R[m]
        U, _, Vt = np.linalg.svd(np.einsum("nij,nkj->ik", G, c2w[tgt_idx[m], :3, :3]))
        R = U @ Vt
        rs = np.einsum("ij,nj->ni", R, src)
        mu_s, mu_d = rs.mean(0), dst.mean(0)
        s = float(((rs - mu_s) * (dst - mu_d)).sum() / max(((rs - mu_s) ** 2).sum(), 1e-12))
        t = mu_d - s * mu_s
    else:
        s, R, t = ecc.umeyama_sim3(src, dst, True)
    for i in range(len(c2w)):
        c2w[i][:3, 3] = s * (R @ c2w[i][:3, 3]) + t
        c2w[i][:3, :3] = R @ c2w[i][:3, :3]
    depths = [None if d is None else np.asarray(d, np.float64) * s for d in depths]
    _atag = "" if a.da3_anchor == "gt" else f"_anchor-{a.da3_anchor}"
    # --da3_on_gt poses are from the GT video, NOT the generated one. They must not land
    # on the same filename: eval_channel_vs_da3.py reads _da3_poses.npy and would then
    # compare raymap against a DA3 run on a different video, silently.
    _ptag = "_gt" if a.da3_on_gt else ""
    np.save(os.path.join(out_dir, f"sample_{int(sid):02d}_da3{_ptag}_poses{_atag}.npy"), c2w)
    if a.save_depth:
        _dl = [d for d in depths if d is not None]
        if _dl:
            _h, _w = _dl[0].shape[-2:]
            _fh, _fw = np.asarray(frames[0]).shape[:2]
            # Pixel-wise depth comparison assumes DA3 output (u,v) corresponds to input (u,v).
            # A pure resize preserves that; a crop or pad does not. Aspect ratio plus a centred
            # principal point is what distinguishes them, so check both rather than assume.
            assert abs(_h / _w - _fh / _fw) < 1e-2, \
                f"DA3 depth {(_h,_w)} vs frame {(_fh,_fw)}: aspect differs -- not a pure resize"
            if intr_da3 is not None:
                assert abs(intr_da3[2] - _w / 2) <= max(2.0, 0.01 * _w) and \
                       abs(intr_da3[3] - _h / 2) <= max(2.0, 0.01 * _h), \
                    f"DA3 principal point {intr_da3[2:]} off-centre for {(_h,_w)} -- cropped?"
            _W = 256
            _yi = ((np.arange(_W) + 0.5) * _h / _W).astype(int)
            _xi = ((np.arange(_W) + 0.5) * _w / _W).astype(int)
            _k = [i for i in range(0, len(depths), 4) if depths[i] is not None]
            _st = np.stack([np.asarray(depths[i])[np.ix_(_yi, _xi)] for i in _k]).astype(np.float16)
            _nm = "_da3_gt_depth" if a.da3_on_gt else "_da3_depth"
            np.save(os.path.join(out_dir, f"sample_{int(sid):02d}{_nm}{_atag}.npy"), _st)
            if confs is not None and len(confs) == len(depths):
                # same frames and grid as the depth above, so masks index both identically
                _cs = np.stack([np.asarray(confs[i])[np.ix_(_yi, _xi)] for i in _k]).astype(np.float16)
                _cn = "_da3_gt_conf" if a.da3_on_gt else "_da3_conf"
                np.save(os.path.join(out_dir, f"sample_{int(sid):02d}{_cn}{_atag}.npy"), _cs)
            else:
                print(f"  [note] sample {sid}: DA3 returned no confidence -> no _da3_conf.npy")
    # Residual after the fit IS the error -- the alignment only removed the unobservable 7 dof.
    d = np.linalg.norm(c2w[tgt_idx[m], :3, 3] - dst, axis=1)
    # The floor is DA3-on-the-GT-video measured AGAINST GT, so it can only be subtracted from a
    # DA3-vs-GT number. Against `action` or `raymap` the two quantities share no reference and the
    # difference would be arithmetic with no meaning.
    _fl = (_csv_row(os.path.join(a.results_dir, a.floor_csv),
                    f"sample_{int(sid):02d}").get("floor_transerr")
           if tgt_name == "GT" else None)
    # Errors do not add linearly, so this is an attribution aid, not an identity.
    if _fl is None:
        _at = ""
    elif d.mean() <= _fl:
        _at = f"; at or below the instrument floor {_fl:.2f} -- nothing attributable"
    else:
        _at = f"; instrument floor {_fl:.2f} -> generation +{d.mean() - _fl:.2f}"
    lbl = (f"DA3 depth @ DA3 poses (7-dof -> {tgt_name}, s={s:.3f})\n"
           f"DA3 vs {tgt_name}:  {d.mean():.2f} m   (7-dof -> {tgt_name}{_at})")
    return dict(kind="da3", c2w=c2w, depths=depths, rgbs=frames[:len(c2w)],
                intr=intr_da3 if intr_da3 is not None else intr_gt, carla_axes=False,
                label=lbl)


def accumulate(L, a, rng):
    """Unproject every cloud_stride-th frame of one layer and pick its scatter colour."""
    P, C = [], []
    for i in range(0, min(len(L["depths"]), len(L["c2w"])), max(1, a.cloud_stride)):
        if L["depths"][i] is None:
            continue
        p, c = unproject(L["depths"][i], L["rgbs"][i], L["c2w"][i], L["intr"], a.depth_clip,
                         L["carla_axes"], rng, a.points_per_frame)
        P.append(p); C.append(c)
    L["P"] = np.concatenate(P) if P else None
    L["C"] = (np.concatenate(C).astype(np.float32) / 255.0) if P else None
    # Overlaid, two true-colour clouds of the same street are unreadable: the real scene here is
    # pale pavement, which vanishes on a white ground, and whichever cloud draws second wins. So
    # `auto` paints BOTH flat in `both` mode -- teal vs orange, complementary -- and the eye reads
    # disagreement as colour separation. Single-source runs keep true colour: there is nothing to
    # separate from, and the real texture is what makes a collapsed scene obvious.
    mode = a.cloud_color
    if mode == "auto":
        mode = "tint" if a.source == "both" else "rgb"
    L["flat"] = None if mode == "rgb" else (a.gt_tint if L["kind"] == "gt" else a.da3_tint)


def _da3_infer(frames, probe, ref_view="first"):
    """DA3 gives pose AND depth; eval_camera_control keeps only the pose. Field names are not
    documented in this repo, so --probe prints them before anything depends on them."""
    from depth_anything_3.api import DepthAnything3
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mid = os.environ.get("DA3_MODEL", "depth-anything/DA3-LARGE-1.1")
    model = DepthAnything3.from_pretrained(mid).to(dev).eval()
    with torch.no_grad():
        # ref_view_strategy is DA3's own choice of which frame anchors its pose parameterisation.
        # The default "saddle_balanced" picks one mid-sequence, and measured error grows with
        # distance from it (corr +0.48, 88/97 samples) -- which puts frame 0 at the far end, at
        # the 91st percentile of error in the median sample. "first" anchors on frame 0 instead,
        # matching how raymap/action are anchored, so the three become comparable.
        # This gives DA3 no ground truth: it only reorders its own reference, unlike passing
        # `extrinsics`, which conditions the model (da3.py:126) and would make pose eval circular.
        pred = model.inference([np.asarray(f) for f in frames],
                               ref_view_strategy=ref_view)
    if probe:
        print("[probe] fields:", [x for x in dir(pred) if not x.startswith("_")])
        for name in ("depth", "depths", "points", "intrinsics", "confidence", "conf"):
            if hasattr(pred, name):
                v = getattr(pred, name)
                print(f"  {name}: {getattr(v, 'shape', type(v))}")
        return None

    def np_(x):
        return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)

    ext = np_(pred.extrinsics).astype(np.float64)              # (N,3,4) w2c
    c2w = np.zeros((len(ext), 4, 4))
    for i in range(len(ext)):
        m = np.eye(4); m[:3, :4] = ext[i]; c2w[i] = np.linalg.inv(m)
    dep = None
    for name in ("depth", "depths"):
        if hasattr(pred, name):
            dep = [np_(d).squeeze() for d in np_(getattr(pred, name))]
            break
    if dep is None:
        raise SystemExit("[fatal] DA3 result has no depth field -- run --probe and update _da3_infer")
    intr = None
    if hasattr(pred, "intrinsics"):
        K = np_(pred.intrinsics)
        K0 = K[0] if K.ndim == 3 else K
        intr = np.array([K0[0, 0], K0[1, 1], K0[0, 2], K0[1, 2]], dtype=np.float64)
    conf = None
    for name in ("conf", "confidence"):
        if getattr(pred, name, None) is not None:
            conf = [np_(c).squeeze() for c in np_(getattr(pred, name))]
            break
    return c2w, dep, intr, conf


if __name__ == "__main__":
    main()
