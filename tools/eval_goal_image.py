#!/usr/bin/env python3
"""Goal-image metrics: last generated frame vs the GT goal frame (the frame the model was
conditioned on as its goal, samples_meta.json "goal_frame"). Run in /venv/metrics.

  LPIPS (alex, same net as video_quality_metrics.csv)              lower = closer
  DINO similarity = cosine of DINOv2-base CLS embeddings (HF facebook/dinov2-base)   higher = closer

  /venv/metrics/bin/python tools/eval_goal_image.py --results_dir <dir>  -> goal_image_metrics.csv + goal_image.json
"""
import argparse, csv, json, os, re
import numpy as np, cv2, imageio.v3 as iio, torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--dino", default="facebook/dinov2-base")
    a = ap.parse_args()
    import lpips
    from transformers import AutoImageProcessor, AutoModel
    D = a.results_dir
    meta = json.load(open(os.path.join(D, "samples_meta.json")))
    lp = lpips.LPIPS(net="alex", verbose=False).cuda().eval()
    proc = AutoImageProcessor.from_pretrained(a.dino)
    dino = AutoModel.from_pretrained(a.dino).cuda().eval()
    rows = []
    for k in sorted(meta, key=int):
        info = meta[k]
        vp = os.path.join(D, f"sample_{int(k):02d}.mp4")
        if not os.path.exists(vp) or "goal_frame" not in info:
            continue
        last = iio.imread(vp)[-1]
        h, w = last.shape[:2]
        g = cv2.cvtColor(cv2.imread(f"{info['seq_dir']}/images/{int(info['goal_frame']):06d}.png"), cv2.COLOR_BGR2RGB)
        g = cv2.resize(g, (w, h), interpolation=cv2.INTER_AREA)
        with torch.no_grad():
            t = lambda x: torch.from_numpy(x).permute(2, 0, 1)[None].float().div(127.5).sub(1).cuda()
            l = float(lp(t(last), t(g)))
            f = dino(**proc(images=[last, g], return_tensors="pt").to("cuda")).last_hidden_state[:, 0]
            s = float(torch.nn.functional.cosine_similarity(f[0:1], f[1:2]))
        rows.append({"sample": f"sample_{int(k):02d}", "goal_frame": int(info["goal_frame"]),
                     "goal_lpips": l, "goal_dino_sim": s})
    with open(os.path.join(D, "goal_image_metrics.csv"), "w", newline="") as fh:
        w_ = csv.DictWriter(fh, fieldnames=list(rows[0])); w_.writeheader(); [w_.writerow(r) for r in rows]
    out = {"n": len(rows), "goal_lpips": float(np.mean([r["goal_lpips"] for r in rows])),
           "goal_dino_sim": float(np.mean([r["goal_dino_sim"] for r in rows])), "dino_model": a.dino}
    json.dump(out, open(os.path.join(D, "goal_image.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
