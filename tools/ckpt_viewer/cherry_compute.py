#!/usr/bin/env python3
"""Per-frame similarity to GT for ours vs the 7 baselines (cherry-pick finder data).

For every sample of the cmp benches (valbench5s 120, LookOut A f004 / B f100 minus the exclusion list), frame k = GT frame
start+k (k = 0..76, ours has 77 frames, baselines 79 with frame k = GT first+k). All clips are cut to the GT rows that every
baseline covers (GT y 108..404 of 512; baselines were generated from different vertical crops of the 512x512 GT) and resized
to 384x222, then LPIPS (alex) and PSNR vs GT are computed. CPU only.

  cherry_compute.py <ckpt> [bench ...]      -> cache/cherry/ck<ckpt>/<bench>/<sample name>.npz  {lpips, psnr: [8, 77]}, rows = ours + MODELS
"""
import glob, json, os, sys, time
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cmp import CMP_ROOT, MODELS, CB, _tail, sample_filter

R = "/root/project/egowam/results_syn/goalprog_mode_real6_nomr_hg0_grad0.5"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "cherry")
Y0, Y1, W, H = 108, 404, 384, 222
NF = 77
torch = LP = None   # loaded in main() only, so the viewer server (no lpips) can import cut() / constants


def sub_for(ck, bench):
    """CB row with our results subdir resolved for this ckpt. LookOut C: ck7500 = lookout_turn77v2/samples_lh (82-clip run,
    idx82 subset via sample_filter); averaged ckpts (ck "a4000_7500" -> ckpta4000_7500) = lookout_turn77val35(_lh), the 35 clips 0..34."""
    b = CB[bench]; sub = b[2]
    if bench == "lookout_c" and not os.path.exists(f"{R}/ckpt{ck}/{sub}/samples_meta.json"):
        sub = next((x for x in ("lookout_turn77val35_lh", "lookout_turn77val35") if os.path.exists(f"{R}/ckpt{ck}/{x}/samples_meta.json")), sub)
    return (b[0], b[1], sub) + tuple(b[3:])


def filter_for(ck, bench):
    if bench == "lookout_c" and sub_for(ck, bench)[2] != CB[bench][2]:
        return None, set()          # the val35 dirs hold exactly the 35 clips
    return sample_filter(bench)


def _load():
    global torch, LP
    import torch as _t, lpips
    torch = _t; torch.set_num_threads(max(4, os.cpu_count() // 2))
    LP = lpips.LPIPS(net="alex", verbose=False).eval()


def read_video(p, n):
    cap = cv2.VideoCapture(p); fr = []
    while len(fr) < n:
        ok, f = cap.read()
        if not ok: break
        fr.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    return fr


def cut(img, crop):
    """img (h, w, 3) whose rows span GT rows crop=(y0, y1) of the 512 GT (None = 0..512) -> GT rows Y0..Y1 at W x H."""
    h, w = img.shape[:2]
    if crop is None and abs(w / h - 1) > 0.02:   # no crop info but not square (VGGT-Diff 832x480): centre crop of the GT
        c = 512 * h / w; crop = (0, (512 - c) / 2, 512, (512 + c) / 2)
    y0, y1 = (0, 512) if crop is None else (crop[1], crop[3])
    a = int(round((Y0 - y0) / (y1 - y0) * h)); b = int(round((Y1 - y0) / (y1 - y0) * h))
    return cv2.resize(img[max(a, 0):min(b, h)], (W, H), interpolation=cv2.INTER_AREA)


def lp(a, b):
    ta = torch.from_numpy(np.stack(a)).permute(0, 3, 1, 2).float() / 127.5 - 1
    tb = torch.from_numpy(np.stack(b)).permute(0, 3, 1, 2).float() / 127.5 - 1
    with torch.no_grad():
        return np.concatenate([LP(ta[i:i + 16], tb[i:i + 16]).flatten().numpy() for i in range(0, len(ta), 16)])


def psnr(a, b):
    return np.array([10 * np.log10(255 ** 2 / max(np.mean((x.astype(float) - y.astype(float)) ** 2), 1e-6)) for x, y in zip(a, b)])


def main():
    _load()
    ck = sys.argv[1]; ck = int(ck) if ck.isdigit() else ck; benches = sys.argv[2:] or ["lookout_f004", "lookout_f100", "lookout_c", "valbench5s"]
    for bench in benches:
        _, folder, sub, *_ = sub_for(ck, bench)
        D = f"{R}/ckpt{ck}/{sub}"; meta = json.load(open(f"{D}/samples_meta.json"))
        idx = {}
        for p in glob.glob(f"{CMP_ROOT}/{MODELS[0][0]}/{folder}/*/meta.json"):
            j = json.load(open(p)); idx[(_tail(j["seq_dir"]), int(j["first_idx"]))] = os.path.basename(os.path.dirname(p))
        fmode, fset = filter_for(ck, bench)
        od = f"{OUT}/ck{ck}/{bench}"; os.makedirs(od, exist_ok=True)
        t0 = time.time(); n = 0
        for k in sorted(meta, key=int):
            i = int(k)
            if (fmode == "exclude" and i in fset) or (fmode == "include" and i not in fset): continue
            info = meta[k]; seq = info["seq_dir"].rstrip("/"); s0 = int(info["start_frame"])
            name = idx.get((_tail(seq), s0))
            if not name or os.path.exists(f"{od}/{name}.npz"): continue
            gt = [cut(cv2.cvtColor(cv2.imread(f"{seq}/images/{s0 + f:06d}.png"), cv2.COLOR_BGR2RGB), None) for f in range(NF)]
            rows_l, rows_p = [], []
            srcs = [(f"{D}/sample_{i:02d}.mp4", None)] + [(f"{CMP_ROOT}/{m}/{folder}/{name}/video.mp4",
                    json.load(open(f"{CMP_ROOT}/{m}/{folder}/{name}/meta.json")).get("gt_crop_box_xyxy")) for m, _ in MODELS]
            for p, crop in srcs:
                fr = read_video(p, NF) if os.path.exists(p) else []
                fr = [cut(x, crop) for x in fr]
                m = min(len(fr), NF)
                l = np.full(NF, np.nan); q = np.full(NF, np.nan)
                if m:
                    l[:m] = lp(fr[:m], gt[:m]); q[:m] = psnr(fr[:m], gt[:m])
                rows_l.append(l); rows_p.append(q)
            np.savez(f"{od}/{name}.npz", lpips=np.array(rows_l), psnr=np.array(rows_p), i=i, start=s0)
            n += 1
            if n % 10 == 0: print(f"[{bench}] {n} samples, {(time.time() - t0) / n:.1f}s/sample", flush=True)
        print(f"[{bench}] done {n} new samples in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
