#!/usr/bin/env python3
"""Set-level video metrics for one results dir: FVD (I3D, cd-fvd) on the first 16 frames and on
all frames, plus MUSIQ (pyiqa, KonIQ) of the generated frames. Run in /venv/metrics.

Reference set = the GT clips of the SAME windows (samples_meta.json), resized like the generated
video. GT features/scores are cached per window list under results_syn/_fvd_cache/, so every run
on the same bench shares one reference. NOTE: 120 clips is a small set for FVD -- compare runs on
the same bench only, never against published FVD numbers.

  /venv/metrics/bin/python tools/eval_fvd_musiq.py --results_dir <dir>   -> fvd_musiq.json
"""
import argparse, hashlib, json, os, re
import numpy as np, cv2, imageio.v3 as iio, torch

CACHE = "/root/project/egowam/results_syn/_fvd_cache"
RES = 256                      # cd-fvd load_videos default; I3D preprocessing resizes from here
MUSIQ_STRIDE = 4               # score every 4th frame (one per latent frame)


def gt_clip(info, n, hw):
    s = int(info["start_frame"]); st = int(info.get("frame_stride", 1))
    fr = [cv2.cvtColor(cv2.imread(f"{info['seq_dir']}/images/{s + k * st:06d}.png"), cv2.COLOR_BGR2RGB) for k in range(n)]
    return np.stack([cv2.resize(f, hw, interpolation=cv2.INTER_AREA) for f in fr])


def small(v):
    return np.stack([cv2.resize(f, (RES, RES), interpolation=cv2.INTER_AREA) for f in v])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--video_re", default=r"sample_(\d+)\.mp4$")
    ap.add_argument("--batch", type=int, default=8)
    a = ap.parse_args()
    from cdfvd import fvd
    import pyiqa
    D = a.results_dir
    meta = json.load(open(os.path.join(D, "samples_meta.json")))
    vids = sorted((int(m.group(1)), f) for f in os.listdir(D) if (m := re.match(a.video_re, f)))
    vids = [(i, f) for i, f in vids if str(i) in meta]
    first = iio.imread(os.path.join(D, vids[0][1]))
    T, H, W = first.shape[:3]
    key = hashlib.md5(json.dumps([(meta[str(i)]["seq_dir"], meta[str(i)]["start_frame"]) for i, _ in vids]).encode()
                      ).hexdigest()[:12] + f"_T{T}_{RES}"
    os.makedirs(CACHE, exist_ok=True)
    musiq = pyiqa.create_metric("musiq", device="cuda")
    evs = {L: fvd.cdfvd("i3d", n_real="full", n_fake="full", device="cuda") for L in (16, T)}
    real_paths = {L: os.path.join(CACHE, f"real_{key}_L{L}.npz") for L in evs}
    have_real = all(os.path.exists(p) for p in real_paths.values())
    if have_real:
        for L, ev in evs.items():
            ev.load_real_stats(real_paths[L])
    gt_musiq_p = os.path.join(CACHE, f"musiq_gt_{key}.json")
    gt_musiq = json.load(open(gt_musiq_p)) if os.path.exists(gt_musiq_p) else None

    def mq(v):
        x = torch.from_numpy(v[::MUSIQ_STRIDE]).permute(0, 3, 1, 2).float().div(255).cuda()
        with torch.no_grad():
            return float(musiq(x).mean())

    gen_mq, gt_mq = [], []
    for b in range(0, len(vids), a.batch):
        chunk = vids[b:b + a.batch]
        fake = [iio.imread(os.path.join(D, f))[:T] for _, f in chunk]
        gen_mq += [mq(v) for v in fake]
        fake = np.stack([small(v) for v in fake])
        for L, ev in evs.items():
            ev.add_fake_stats(fake[:, :L])
        if not have_real or gt_musiq is None:
            real = [gt_clip(meta[str(i)], T, (W, H)) for i, _ in chunk]
            if gt_musiq is None:
                gt_mq += [mq(v) for v in real]
            if not have_real:
                real = np.stack([small(v) for v in real])
                for L, ev in evs.items():
                    ev.add_real_stats(real[:, :L])
        print(f"  {min(b + a.batch, len(vids))}/{len(vids)}", flush=True)
    if not have_real:
        for L, ev in evs.items():
            ev.save_real_stats(real_paths[L])
    if gt_musiq is None:
        gt_musiq = {"musiq_gt": float(np.mean(gt_mq)), "n": len(gt_mq)}
        json.dump(gt_musiq, open(gt_musiq_p, "w"))
    out = {"n": len(vids), "frames": T, "fvd16": float(evs[16].compute_fvd_from_stats(evs[16].fake_stats, evs[16].real_stats)),
           f"fvd{T}": float(evs[T].compute_fvd_from_stats(evs[T].fake_stats, evs[T].real_stats)),
           "musiq": float(np.mean(gen_mq)), "musiq_gt_ref": gt_musiq["musiq_gt"], "reference": key}
    json.dump(out, open(os.path.join(D, "fvd_musiq.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
