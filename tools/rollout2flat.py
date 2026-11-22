#!/usr/bin/env python3
"""Turn LookOut rollout dirs (rollout_NN_*, 2 chunks x 77 frames, overlap 1 = 153 frames) into one
flat valbench-style dir so the extended suite (eval_suite / fvd / goal image / met3r / depth C)
runs on it unchanged.

  sample_MM.mp4                 -> symlink to rollout_NN.mp4                       (153 frames)
  sample_MM_pred_poses.npy      -> symlink to rollout_NN_pred_poses.npy  (invert_raymap_rollout,
                                   chunks CHAINED from the GT start: the honest long-horizon track)
  sample_MM_pred_actions.npy    -> symlink to rollout_NN_pred_actions.npy (infer_rollout --pred_action, 1 chunk)
  sample_MM_pred_depth.npy      =  concat(chunk0 latents, chunk1 latents[1:])  (39 latents = frames
                                   0,4,..,152; chunk1 latent 0 is the overlap frame 76)
  samples_meta.json             {seq_dir, start_frame, num_frames=153, seq_idx, goal_frame=last goal}

Several rollout dirs are concatenated in the order given (first half, then _h2).
  python3 tools/rollout2flat.py <out_dir> <roll_dir> [<roll_dir> ...]
"""
import json, os, sys
import numpy as np


def main():
    out, rolls = sys.argv[1], sys.argv[2:]
    os.makedirs(out, exist_ok=True)
    meta, j = {}, 0
    for R in rolls:
        rm = json.load(open(os.path.join(R, "rollout_meta.json")))
        for k in sorted(rm, key=int):
            v, n = rm[k], int(k)
            src = os.path.join(R, f"rollout_{n:02d}")
            if not os.path.exists(src + ".mp4"):
                print(f"[skip] {src}.mp4 missing"); continue
            dst = os.path.join(out, f"sample_{j:02d}")
            for suf in (".mp4", "_pred_poses.npy", "_pred_actions.npy"):   # _pred_actions: joint-FM action-token runs
                if os.path.exists(src + suf) and not os.path.lexists(dst + suf):
                    os.symlink(os.path.abspath(src + suf), dst + suf)
            if os.path.exists(src + "_pred_depth.npy") and not os.path.exists(dst + "_pred_depth.npy"):
                d = np.load(src + "_pred_depth.npy")               # (chunks, 20, h, w)
                np.save(dst + "_pred_depth.npy", np.concatenate([d[0]] + [c[1:] for c in d[1:]], 0))
            meta[str(j)] = {"seq_dir": v["seq_dir"], "start_frame": int(v["start_frame"]),
                            "num_frames": int(v["num_frames"]), "seq_idx": j,
                            "goal_frame": int(v["goal_frames"][-1]), "src": src}
            j += 1
    json.dump(meta, open(os.path.join(out, "samples_meta.json"), "w"), indent=1)
    print(f"{j} samples -> {out}")


if __name__ == "__main__":
    main()
