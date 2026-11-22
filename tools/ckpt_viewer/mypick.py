"""Manual cherry-pick page /mypick: pick any sample, scrub its frames, hand-pick the frames to combine.

Reads the same per-frame data as /cherry (cache/cherry/ck<ck>/<bench>/<name>.npz via Cherry.load: LPIPS / PSNR vs GT,
rows = ours + cmp.MODELS) and reuses its image / save routes (/cherryimg, /cherrystrip, /cherryroute, /api/cherry/export),
so this file only adds the sample list and the per-frame curve. Nothing in cherry.py is changed.
Routes: /mypick  /api/mypick/samples?ck=&bench=  /api/mypick/curve?ck=&bench=&name=  /mypickv/<ck>/<bench>/<name>/<gt|ours|model>.mp4
Videos (autoplay grid): GT = the 77 GT frames start..start+76 (16 fps, same index k as ours), ours = sample_NN.mp4, baselines =
cmp video.mp4 (frame k = GT first+k, own fps -> the page plays them at fps/16 x so frame k lines up across tiles).
"""
import json, math, os
import numpy as np

from cmp import MODELS, CMP_ROOT
from cherry import BENCHES, LABEL
from cherry_compute import sub_for


def _f(x):
    x = float(x)
    return None if math.isnan(x) or math.isinf(x) else round(x, 4)


def samples(cherry, ck, bench):
    """[{bench, name, i, ours (mean LPIPS), margin (mean best-baseline LPIPS - ours), win (#frames ours < every baseline)}]"""
    out = []
    for b in (BENCHES if bench == "all" else [bench]):
        for s in cherry.load(ck, b):
            lp = s["lp"]; o = lp[0]; best = np.nanmin(lp[1:], axis=0); ok = np.isfinite(o) & np.isfinite(best)
            out.append({"bench": b, "label": LABEL[b], "name": s["name"], "i": s["i"],
                        "ours": _f(np.nanmean(o)), "margin": _f(np.mean(best[ok] - o[ok])) if ok.any() else None,
                        "win": int(np.sum(ok & (o < best)))})
    return json.dumps(out, separators=(",", ":")).encode()


def _info(cherry, ck, bench, name):
    s = next((x for x in cherry.load(ck, bench) if x["name"] == name), None) if bench in BENCHES else None
    if s is None:
        return None, None, None
    _, folder, sub, *_ = sub_for(ck, bench)
    info = json.load(open(f"{cherry.D.R}/ckpt{ck}/{sub}/samples_meta.json"))[str(s["i"])]
    return s, folder, (sub, info)


def videos(cherry, cmp, ck, bench, name):
    s, folder, (sub, info) = _info(cherry, ck, bench, name)
    base = f"mypickv/{ck}/{bench}/{name}"
    out = [{"id": "gt", "label": "GT", "fps": 16.0, "src": f"{base}/gt.mp4"},
           {"id": "ours", "label": f"Ours ck{ck}", "fps": 16.0, "src": f"{base}/ours.mp4"}]
    for mid, ml in MODELS:
        if folder and os.path.isfile(f"{CMP_ROOT}/{mid}/{folder}/{name}/video.mp4"):
            out.append({"id": mid, "label": ml, "fps": cmp.model_fps(mid, folder), "src": f"{base}/{mid}.mp4"})
    return out


def video_path(cherry, cmp, ck, bench, name, vid):
    s, folder, x = _info(cherry, ck, bench, name)
    if s is None:
        return None
    sub, info = x
    if vid == "gt":
        return cherry.D.clip(info["seq_dir"].rstrip("/"), int(s["start"]), int(s["lp"].shape[1]))
    if vid == "ours":
        p = f"{cherry.D.R}/ckpt{ck}/{sub}/sample_{s['i']:02d}.mp4"
        return p if os.path.isfile(p) else None
    return cmp.video(vid, bench, name)


def curve(cherry, ck, bench, name, cmp=None):
    """per-frame LPIPS / PSNR of ours + every baseline for one sample (None where a model has no frame)."""
    s = next((x for x in cherry.load(ck, bench) if x["name"] == name), None) if bench in BENCHES else None
    if s is None:
        return None
    lp, ps = s["lp"], s["ps"]
    best = np.nanmin(lp[1:], axis=0)
    return json.dumps({"bench": bench, "name": name, "i": s["i"], "start": s["start"], "nf": int(lp.shape[1]),
                       "models": ["ours"] + [m[1] for m in MODELS],
                       "lpips": [[_f(v) for v in row] for row in lp], "psnr": [[_f(v) for v in row] for row in ps],
                       "best": [_f(v) for v in best],
                       "videos": videos(cherry, cmp, ck, bench, name) if cmp else []}, separators=(",", ":")).encode()
