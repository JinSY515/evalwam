#!/usr/bin/env python3
"""Does the generated VIDEO agree with the model's own camera channel?

Every other metric here is scored against the CARLA ground truth, so it answers "is the
channel right". This one never uses GT as the target: it fits DA3's monocular pose estimate
-- read off the GENERATED pixels -- onto the channel's own trajectory and reports the
residual. Three numbers in the same units then settle whether the channel steers the video:

    ch_ade     video's camera  vs  this model's channel   (DA3 anchored on the channel)
    gt_ade     video's camera  vs  GT                     (DA3 anchored on GT)
    floor_ade  GT video        vs  GT                     (DA3's own error, da3_floor.py)

    ch_ade ~ floor_ade   video and channel agree inside instrument noise -> channel steers
    ch_ade > gt_ade      the video matches the TRUTH better than it matches its own channel,
                         so that channel did not produce it -> parallel head
                         (this reading needs no DoF matching: all three come from one path)

CROSS-CHANNEL READS ARE NARROWER. `ade` is xy-only (camera_metrics computes d_xy), so it is
comparable between channels. `roterr` is NOT: integrate_action_poses borrows z/roll/pitch from
GT ("the head outputs 3 numbers"), so the action channel's rotation is partly ground truth.
Use --frames latent to equalise the 20-vs-77 correspondence count before comparing, and read
only ade / fde / rpe_trans / scale_ratio across channels.

SCALE IS REPORTED SEPARATELY because camera_metrics fits a sim3 and therefore forgives it --
and scale is where these heads fail (action path_len_ratio 0.589 vs raymap 0.956).
viz_da3_pointcloud.py:da3_vs: "Refitting would hand 7 dof to absorbing exactly the global
scale/rotation drift that is these heads' likeliest failure." DA3's units are arbitrary, so
only the RATIO of the two fitted scales means anything: 1.0 = the channel agrees with GT.

The raymap side's inversion error is deliberately NOT subtracted. Using raymap as an action
representation requires inverting it, so invert_raymap's residual is a real cost of that
representation, not a handicap to remove.

    python3 tools/eval_channel_vs_da3.py --results_dir <dir> --channel raymap
    python3 tools/eval_channel_vs_da3.py --results_dir <dir> --channel action --frames latent
"""
import argparse
import csv
import importlib.util
import json
import os
import statistics as st

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def _shim(name):
    """tools/<name> as a module (da3_floor.py:_ecc idiom; both targets guard on __main__)."""
    spec = importlib.util.spec_from_file_location(name[:-3], os.path.join(HERE, name))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--channel", default="action", choices=["action", "raymap"])
    ap.add_argument("--da3_poses_dir", default="viz_da3_fix")
    ap.add_argument("--meta", default="samples_meta.json")
    ap.add_argument("--poses_name", default="poses.npy",
                    help="GT poses inside the sequence dir, as eval_camera_control.py reads them. "
                         "Per-sample _gt_poses.npy is not used: the raymap runs do not write it.")
    ap.add_argument("--raymap_poses_suffix", default="_pred_poses.npy")
    ap.add_argument("--frames", default="native", choices=["native", "latent"],
                    help="native: every frame the channel provides (77 raymap, 20 action). "
                         "latent: both on the VAE-stride frames, so a cross-channel comparison "
                         "fits its sim3 from the same number of correspondences.")
    ap.add_argument("--vae_stride", type=int, default=4)
    ap.add_argument("--floor_csv", default="da3_floor_fix.csv",
                    help="da3_floor.py output; supplies floor_ade for the inequality above")
    ap.add_argument("--out_csv", default="channel_vs_da3_metrics.csv")
    a = ap.parse_args()

    ecc = _shim("eval_camera_control.py")
    vz = _shim("viz_da3_pointcloud.py")

    D = a.results_dir
    dd = a.da3_poses_dir if os.path.isabs(a.da3_poses_dir) else os.path.join(D, a.da3_poses_dir)
    meta = json.load(open(os.path.join(D, a.meta)))

    floor = {}
    fp = a.floor_csv if os.path.isabs(a.floor_csv) else os.path.join(D, a.floor_csv)
    if os.path.exists(fp):
        for r in csv.DictReader(open(fp)):
            try:
                floor[r["sample"]] = float(r["floor_ade"])
            except (KeyError, ValueError):
                pass
    else:
        print(f"  [note] {fp} absent -- floor_ade will be blank. Run tools/da3_floor.py "
              f"or the `ch_ade ~ floor` half of the reading is unavailable.")

    rows, skipped = [], []
    for k in sorted(meta, key=lambda x: int(x)):
        sid = int(k)
        tag = f"sample_{sid:02d}"
        mm = meta[k]
        p_da3 = os.path.join(dd, f"{tag}_da3_poses.npy")
        if not os.path.exists(p_da3):
            skipped.append(f"{tag}: no {os.path.relpath(p_da3, D)}")
            continue
        da3 = np.load(p_da3)
        seq_dir = mm["seq_dir"]
        start = int(mm.get("start_frame", mm.get("start", 0)))
        nf = int(mm.get("num_frames", len(da3)))
        p_gt = os.path.join(seq_dir, a.poses_name)
        if not os.path.exists(p_gt):
            skipped.append(f"{tag}: no {p_gt}")
            continue

        if a.channel == "action":
            p_ch = os.path.join(D, f"{tag}_pred_actions.npy")
            if not os.path.exists(p_ch):
                skipped.append(f"{tag}: no pred_actions.npy")
                continue
            gt_win = np.load(p_gt)[start:start + nf]          # (nf, 6), what the head integrates on
            ch, fidx = vz.integrate_action_poses(
                np.load(p_ch), gt_win, ecc.carla_pose_to_mat, a.vae_stride)
        else:
            p_ch = os.path.join(D, f"{tag}{a.raymap_poses_suffix}")
            if not os.path.exists(p_ch):
                skipped.append(f"{tag}: no {a.raymap_poses_suffix} "
                               f"(run invert_raymap.py --stage s3)")
                continue
            raw = np.load(p_ch)
            # invert_raymap writes the CARLA 6-vector, not 4x4. Convert with the same function
            # eval_camera_control uses for GT, or the two live in different conventions.
            ch = (np.stack([ecc.carla_pose_to_mat(*p) for p in raw])
                  if raw.ndim == 2 and raw.shape[1] == 6 else raw)
            fidx = np.arange(len(ch))

        if a.frames == "latent":
            m = (fidx % a.vae_stride) == 0
            ch, fidx = ch[m], fidx[m]
        keep = fidx < len(da3)
        ch, fidx = ch[keep], fidx[keep]
        if len(fidx) < 3:
            skipped.append(f"{tag}: only {len(fidx)} correspondences")
            continue

        gt = ecc.load_target_poses(p_gt, start + fidx)
        m_ch = ecc.camera_metrics(da3[fidx], ch)   # DA3 -> channel. GT never enters.
        m_gt = ecc.camera_metrics(da3[fidx], gt)   # same DA3 -> GT. scale + context reference.
        r = {"sample": tag, "n": m_ch["n"]}
        for kk in ("ade", "fde", "transerr", "roterr", "rotstep_err",
                   "rpe_trans", "rpe_rot_deg", "rpe_camera", "cammc"):
            r[f"ch_{kk}"] = m_ch[kk]
        r["gt_ade"] = m_gt["ade"]
        r["floor_ade"] = floor.get(tag, "")
        r["scale_to_channel"] = m_ch["scale"]
        r["scale_to_gt"] = m_gt["scale"]
        r["scale_ratio"] = (m_ch["scale"] / m_gt["scale"]) if m_gt["scale"] else float("nan")
        rows.append(r)

    for s in skipped[:12]:
        print(f"  [skip] {s}")
    if len(skipped) > 12:
        print(f"  [skip] ... and {len(skipped) - 12} more")
    if not rows:
        raise SystemExit(f"[ch-da3] nothing scored. {dd}/sample_NN_da3_poses.npy present? "
                         f"viz_da3_pointcloud.py and da3_floor.py both write them.")

    out = a.out_csv if os.path.isabs(a.out_csv) else os.path.join(D, a.out_csv)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def med(kk):
        v = [r[kk] for r in rows if r[kk] != ""]
        return st.median(v) if v else None

    def f(x):
        return "     -" if x is None else f"{x:6.4f}"

    print(f"\n[ch-da3] channel={a.channel}  frames={a.frames}  n={len(rows)}  -> {out}")
    print(f"  THE INEQUALITY   (xy ADE, m)")
    print(f"    ch_ade    {f(med('ch_ade'))}   video vs its own channel")
    print(f"    gt_ade    {f(med('gt_ade'))}   video vs GT")
    print(f"    floor_ade {f(med('floor_ade'))}   DA3's own error on the GT video")
    _c, _g, _fl = med("ch_ade"), med("gt_ade"), med("floor_ade")
    if _c is not None and _g is not None:
        print("    -> " + ("ch > gt: video matches TRUTH better than its own channel "
                           "(parallel head)" if _c > _g else
                           "ch <= gt: channel is at least as close as GT"))
    if _c is not None and _fl is not None:
        print(f"    -> ch/floor = {_c / _fl:.2f}x  (1.0x = agreement within instrument noise)")
    print(f"  SHAPE   fde {f(med('ch_fde'))}  rpe_trans {f(med('ch_rpe_trans'))}  "
          f"roterr {f(med('ch_roterr'))}deg")
    print(f"          (roterr is NOT cross-channel comparable: action borrows roll/pitch from GT)")
    print(f"  SCALE   to_channel {f(med('scale_to_channel'))}  to_gt {f(med('scale_to_gt'))}  "
          f"ratio {f(med('scale_ratio'))}   <- 1.0 = agrees with GT")


if __name__ == "__main__":
    main()
