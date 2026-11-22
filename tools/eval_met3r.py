#!/usr/bin/env python3
"""MEt3R (CVPR'25, multi-view consistency of generated frames; lower = more consistent).
Run in /venv/met3r (pytorch3d + FeatUp + met3r built for sm_120).

Pairs: frame k vs frame k+STRIDE for k = 0, STRIDE, ... (STRIDE=4 = one latent frame apart, 19 pairs
for 77 frames). Consecutive pixel frames at 16 fps are nearly identical, so stride 1 has little
signal. Settings are the README defaults: MASt3R backbone, DINO16 + FeatUp, cosine, img_size 256.
The GT clips of the same windows get the same score once (cached in results_syn/_fvd_cache/) as a
reference.

  /venv/met3r/bin/python tools/eval_met3r.py --results_dir <dir>  -> met3r_metrics.csv + met3r.json
"""
import argparse, csv, hashlib, json, os
import numpy as np, cv2, imageio.v3 as iio, torch

CACHE = "/root/project/egowam/results_syn/_fvd_cache"
IMG = 256


def prep(v):                                  # (T,H,W,3) uint8 -> (T,3,IMG,IMG) in [-1,1]
    v = np.stack([cv2.resize(f, (IMG, IMG), interpolation=cv2.INTER_AREA) for f in v])
    return torch.from_numpy(v).permute(0, 3, 1, 2).float().div(127.5).sub(1)


def gt_clip(info, n):
    s = int(info["start_frame"]); st = int(info.get("frame_stride", 1))
    return np.stack([cv2.cvtColor(cv2.imread(f"{info['seq_dir']}/images/{s + k * st:06d}.png"), cv2.COLOR_BGR2RGB)
                     for k in range(n)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--batch", type=int, default=19)
    a = ap.parse_args()
    from met3r import MEt3R
    m = MEt3R(img_size=IMG, use_norm=True, backbone="mast3r", feature_backbone="dino16",
              feature_backbone_weights="mhamilton723/FeatUp", upsampler="featup", distance="cosine",
              freeze=True).cuda().eval()

    def score(v):
        x = prep(v)
        idx = np.arange(0, len(x) - a.stride, a.stride)
        pairs = torch.stack([torch.stack([x[i], x[i + a.stride]]) for i in idx])
        out = []
        with torch.no_grad():
            for b in range(0, len(pairs), a.batch):
                s, *_ = m(images=pairs[b:b + a.batch].cuda())
                out.append(s.float().cpu())
        torch.cuda.empty_cache()
        return float(torch.cat(out).mean())

    D = a.results_dir
    meta = json.load(open(os.path.join(D, "samples_meta.json")))
    keys = [k for k in sorted(meta, key=int) if os.path.exists(os.path.join(D, f"sample_{int(k):02d}.mp4"))]
    # clip length is part of the key: LookOut 1-chunk (77 frames) and 2-chunk (153) share seq_dir+start_frame,
    # and the 1-chunk run silently reused the 153-frame GT reference before this was added.
    ck = hashlib.md5(json.dumps([(meta[k]["seq_dir"], meta[k]["start_frame"], int(meta[k].get("num_frames", 77)))
                                 for k in keys]).encode()).hexdigest()[:12]
    gp = os.path.join(CACHE, f"met3r_gt_{ck}_s{a.stride}.json")
    gt = json.load(open(gp)) if os.path.exists(gp) else None
    rows, gts = [], []
    for j, k in enumerate(keys):
        v = iio.imread(os.path.join(D, f"sample_{int(k):02d}.mp4"))
        rows.append({"sample": f"sample_{int(k):02d}", "met3r": score(v)})
        if gt is None:
            gts.append(score(gt_clip(meta[k], len(v))))
        if (j + 1) % 10 == 0:
            print(f"  {j + 1}/{len(keys)}  mean {np.mean([r['met3r'] for r in rows]):.4f}", flush=True)
    if gt is None:
        os.makedirs(CACHE, exist_ok=True)
        gt = {"met3r_gt": float(np.mean(gts)), "n": len(gts)}
        json.dump(gt, open(gp, "w"))
    with open(os.path.join(D, "met3r_metrics.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["sample", "met3r"]); w.writeheader(); [w.writerow(r) for r in rows]
    out = {"n": len(rows), "met3r": float(np.mean([r["met3r"] for r in rows])), "met3r_gt_ref": gt["met3r_gt"],
           "stride": a.stride, "img_size": IMG}
    json.dump(out, open(os.path.join(D, "met3r.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
