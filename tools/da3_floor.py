#!/usr/bin/env python3
"""DA3's own error, measured on the GT video. The baseline every da3_* number is missing.

eval_camera_control.py reports `da3_transerr` for a GENERATED video and stops there. That number
is a sum of two things it cannot separate:

    da3_error(generated)  =  what the generation got wrong  +  what DA3 gets wrong anyway

Run the same estimator on the REAL video of the same window and the second term is measured
directly, so the first can finally be attributed. Without this line, "DA3 ADE 0.22 m" says
nothing about generation quality -- it could be almost entirely the instrument.

Nothing here is new machinery: estimate_da3 / camera_metrics / load_target_poses / extract_frames
all come from eval_camera_control.py via an importlib shim, so the floor is produced by exactly
the code path that produced the number it is a floor for. Only the frames differ.

Why the GT frames go through mp4 by default (--encode mp4): the generated side is read back from
a compressed file, so a floor measured on lossless PNGs would be a floor for a different, easier
input and would overstate how much of the error belongs to the generation. --encode raw gives the
ideal-input floor instead, and the gap between the two is the codec's own contribution.

  python3 tools/da3_floor.py --results_dir <dir>
  python3 tools/da3_floor.py --results_dir <dir> --encode raw --samples 0,1,2
"""
import argparse
import csv
import importlib.util
import json
import os
import statistics as st
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def _ecc():
    """tools/eval_camera_control.py as a module (eval_rollout.py:49-59 idiom)."""
    spec = importlib.util.spec_from_file_location(
        "ecc", os.path.join(HERE, "eval_camera_control.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _encode_roundtrip(frames, fps, ecc):
    """PNG frames -> mp4 -> decoded frames, so the floor sees the same codec loss as the
    generated side. export_to_video is what inference itself writes with, so this is the same
    encoder rather than merely a similar one."""
    try:
        from diffusers.utils import export_to_video
    except Exception:
        export_to_video = None
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "gt.mp4")
        if export_to_video is not None:
            from PIL import Image
            export_to_video([Image.fromarray(f) for f in frames], p, fps=fps)
        else:
            import imageio.v2 as imageio
            with imageio.get_writer(p, fps=fps, macro_block_size=1) as w:
                for f in frames:
                    w.append_data(f)
        return ecc.extract_frames(p)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_dir", required=True,
                    help="holds samples_meta.json; the floor CSV is written here too")
    ap.add_argument("--meta", default="samples_meta.json")
    ap.add_argument("--images_subdir", default="images")
    ap.add_argument("--samples", default="all")
    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--encode", default="mp4", choices=["mp4", "raw"],
                    help="mp4 (default): match the generated side's codec loss. "
                         "raw: the ideal-input floor.")
    ap.add_argument("--fps", type=int, default=16)
    ap.add_argument("--no_save_poses", dest="save_poses", action="store_false",
                    help="skip sample_NN_floor_poses.npy (per-frame poses; needed "
                         "to compare WHERE the error sits, not just how much)")
    ap.add_argument("--out_csv", default="da3_floor_metrics.csv")
    ap.add_argument("--compare_csv", default="camera_control_metrics.csv",
                    help="generated-video metrics to subtract the floor from, if present")
    a = ap.parse_args()

    ecc = _ecc()
    from PIL import Image
    meta_p = os.path.join(a.results_dir, a.meta)
    if not os.path.exists(meta_p):
        raise SystemExit(f"[fatal] {meta_p} missing")
    meta = json.load(open(meta_p))
    sids = (sorted(meta, key=int) if a.samples.strip().lower() in ("all", "*", "")
            else [x.strip() for x in a.samples.split(",") if x.strip()])
    if a.max_samples:
        sids = sids[:a.max_samples]
    print(f"[floor] {len(sids)} samples, encode={a.encode} -> {a.results_dir}/{a.out_csv}")

    rows = []
    for n_done, sid in enumerate(sids, 1):
        if sid not in meta:
            print(f"  [skip] sample {sid} not in {a.meta}")
            continue
        info = meta[sid]
        seq_dir = info["seq_dir"]
        start = int(info.get("start_frame", 0))
        num = int(info.get("num_frames", 0)) or 77
        stride = int(info.get("frame_stride", 1))
        # Same indexing as eval_camera_control.py:268 so the floor lines up frame-for-frame with
        # the generated-video row it will be subtracted from.
        fidx = [start + i * stride for i in range(num)]
        img_dir = os.path.join(seq_dir, a.images_subdir)
        try:
            frames = [np.array(Image.open(os.path.join(img_dir, f"{i:06d}.png")).convert("RGB"))
                      for i in fidx]
        except FileNotFoundError as e:
            print(f"  [skip] sample {sid}: {e}")
            continue
        if a.encode == "mp4":
            frames = _encode_roundtrip(frames, a.fps, ecc)[:num]
        ip = os.path.join(seq_dir, "intrinsics.npy")
        intr = np.load(ip) if os.path.exists(ip) else None
        poses, n, err = ecc.estimate_da3(frames, intr)
        row = {"sample": f"sample_{int(sid):02d}", "n_frames": len(frames), "encode": a.encode}
        if poses is None:
            row["error"] = err
            print(f"  [{n_done}/{len(sids)}] sample {sid}: FAILED {err}")
            rows.append(row)
            continue
        gt = ecc.load_target_poses(os.path.join(seq_dir, "poses.npy"), fidx)
        k = min(len(poses), len(gt))
        m = ecc.camera_metrics(poses[:k], gt[:k])
        if a.save_poses:
            # Aggregates cannot answer WHERE in the sequence the estimator struggles. The
            # generated-video side spikes ~3x at frame 0 (63 samples); without the same per-frame
            # profile from the GT video there is no way to tell whether that is the estimator
            # (endpoint / reference-view selection) or the generation dragging the joint solve.
            np.save(os.path.join(a.results_dir, f"sample_{int(sid):02d}_floor_poses.npy"),
                    poses[:k])
        row.update({f"floor_{kk}": vv for kk, vv in m.items()})
        rows.append(row)
        print(f"  [{n_done}/{len(sids)}] sample {sid}: floor ADE {m['ade']:.3f} m  "
              f"transerr {m['transerr']:.3f} m  roterr {m['roterr']:.2f} deg  scale {m['scale']:.3f}")
        _write(rows, os.path.join(a.results_dir, a.out_csv))   # incremental: survive an abort

    _write(rows, os.path.join(a.results_dir, a.out_csv))
    _summarize(rows, a)


def _write(rows, path):
    if not rows:
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _summarize(rows, a):
    ok = [r for r in rows if "floor_ade" in r]
    if not ok:
        print("[floor] no successful samples")
        return
    fl = {k: [r[k] for r in ok] for k in ("floor_ade", "floor_transerr", "floor_roterr")}
    print(f"\n[floor] {len(ok)} samples, encode={a.encode}")
    for k, v in fl.items():
        print(f"  {k:16s} mean {st.mean(v):7.3f}  median {st.median(v):7.3f}  max {max(v):7.3f}")

    cp = os.path.join(a.results_dir, a.compare_csv)
    if not os.path.exists(cp):
        print(f"\n[floor] {a.compare_csv} not here -- run eval_camera_control.py on the generated "
              f"videos to get the other half of the comparison.")
        return
    gen = {r["sample"]: r for r in csv.DictReader(open(cp))}
    pairs = [(float(r["floor_ade"]), float(gen[r["sample"]]["da3_ade"]))
             for r in ok
             if r["sample"] in gen and gen[r["sample"]].get("da3_ade") not in (None, "", "nan")]
    if not pairs:
        print(f"\n[floor] no sample names shared with {a.compare_csv}")
        return
    f_m, g_m = st.mean([p[0] for p in pairs]), st.mean([p[1] for p in pairs])
    print(f"\n[floor] paired with {a.compare_csv} on {len(pairs)} samples (ADE, metres):")
    print(f"  DA3 on GT video        {f_m:.3f}   <- the instrument")
    print(f"  DA3 on generated video {g_m:.3f}")
    d = g_m - f_m
    pct = f"  ({100 * d / g_m:+.0f}% of the reported number)" if g_m else ""
    print(f"  attributable to generation: {d:+.3f}{pct}")
    # Errors do not add linearly, so this is an attribution aid, not an identity. If the floor is
    # close to the generated number, the metric is mostly measuring DA3, not the model.
    if g_m and (g_m - f_m) / g_m < 0.25:
        print("  [warn] the floor explains most of the reported error -- da3_ade is measuring "
              "the estimator here, not the generation.")


if __name__ == "__main__":
    main()
