#!/usr/bin/env python3
"""Interim valbench5s metrics on whatever samples exist so far, PAIRED across runs.

Only (difficulty, sample_idx) pairs present in EVERY run are scored, so rows are comparable
while runs are still generating. Per-sample results are cached (cache json next to the report),
so each call only scores new samples.

  /venv/da3/bin/python tools/interim_metrics.py            # writes results_syn/abl/METRICS.md
Per sample: PSNR/SSIM/LPIPS(alex) over all 77 frames vs GT, plus brightness / saturation
offset (gen - GT) averaged over frames 32-48 (where the mid-sequence drift sits).
raymap MSE is read from each run's inference log (pred_raymap runs only).
"""
import json, os, re, sys, statistics as st, datetime
import numpy as np, cv2, imageio.v3 as iio

RS = "/root/project/egowam/results_syn"
RUNS = [  # (label, per-difficulty dir pattern, log pattern)
    ("old vcons ck3000",    f"{RS}/plucker_fidelity/e2/full/{{d}}",                             f"{RS}/plucker_fidelity/e2/{{d}}.log"),
    ("abl I jointfm ck3000", f"{RS}/abl/I_jointfm/ckpt3000/valbench5s/{{d}}",                   f"{RS}/abl/I_{{d}}.log"),
    ("abl III goalaux ck8000", f"{RS}/abl/III_goalaux/ckpt8000/valbench5s/{{d}}",               f"{RS}/abl/III_{{d}}.log"),
    ("abl IV depthpred ck3000", f"{RS}/abl/IV_depthpred/ckpt3000/valbench5s/{{d}}",             f"{RS}/abl/IV_{{d}}.log"),
    ("vcons_mode ck500",    f"{RS}/goalprog_dl3dv_camfix_d10_dpt_vcons_mode/ckpt500/valbench5s/{{d}}", f"{RS}/goalprog_dl3dv_camfix_d10_dpt_vcons_mode/ckpt500/valbench5s/{{d}}.log"),
    ("maskfix ck500",       f"{RS}/goalprog_mode_real6_nomr_maskfix/ckpt500/valbench5s/{{d}}",  f"{RS}/goalprog_mode_real6_nomr_maskfix/ckpt500/valbench5s/{{d}}.log"),
    ("maskfix ck2000",      f"{RS}/goalprog_mode_real6_nomr_maskfix/ckpt2000/valbench5s/{{d}}", f"{RS}/goalprog_mode_real6_nomr_maskfix/ckpt2000/valbench5s/{{d}}.log"),
]
DIFFS = ["easy", "medium", "hard"]
OFF120 = {"easy": 0, "hard": 40, "medium": 80}      # per-difficulty idx -> *_all120 idx
ALL120 = {  # where invert_raymap already wrote _pred_poses.npy for the merged dirs
    "old vcons ck3000": f"{RS}/plucker_fidelity/e2/full_all120",
    "maskfix ck500": f"{RS}/goalprog_mode_real6_nomr_maskfix/ckpt500/valbench5s_all120",
    "maskfix ck2000": f"{RS}/goalprog_mode_real6_nomr_maskfix/ckpt2000/valbench5s_all120",
}
VS = 4  # VAE temporal stride: one action per latent frame (key frames 0,4,..,76)


# ---- action metrics: numpy copy of infer_cosmos_lora.{compute_ego_actions,integrate_ego_actions,
# gt_actions_from_poses,action_metrics}. Checked against run I's own _metrics.json every call.
def ego_actions(p):
    p = np.asarray(p, dtype=np.float64)
    dxw, dyw = p[1:, 0] - p[:-1, 0], p[1:, 1] - p[:-1, 1]
    yaw = np.deg2rad(p[:-1, 5])
    dyaw = (p[1:, 5] - p[:-1, 5] + 180) % 360 - 180
    return np.stack([dxw * np.cos(yaw) + dyw * np.sin(yaw), -dxw * np.sin(yaw) + dyw * np.cos(yaw), dyaw], -1)

def integrate(a, yaw0):
    pos, yaw, tr = np.zeros(2), float(yaw0), [np.zeros(2)]
    for dx, dy, dyaw in a:
        r = np.deg2rad(yaw)
        pos = pos + np.array([dx * np.cos(r) - dy * np.sin(r), dx * np.sin(r) + dy * np.cos(r)])
        yaw += dyaw; tr.append(pos.copy())
    return np.array(tr)

def action_metrics(pred, gt, yaw0):
    pf, gf = pred[1:], gt[1:]
    bf = np.repeat(gt[:1], len(gf), 0)
    gt_t = integrate(gf, yaw0)
    ade_fde = lambda t: (float(np.linalg.norm(t - gt_t, axis=1).mean()), float(np.linalg.norm(t - gt_t, axis=1)[-1]))
    ade, fde = ade_fde(integrate(pf, yaw0)); bade, _ = ade_fde(integrate(bf, yaw0))
    return {"ade": ade, "fde": fde, "base_ade": bade, "beats": float(ade < bade),
            "mae_yaw": float(np.abs(pf[:, 2] - gf[:, 2]).mean()),
            "rot_err": float(pf[:, 2].sum() - gf[:, 2].sum())}

def gt_actions(d, i):
    info = json.load(open(os.path.join(d, "samples_meta.json")))[str(i)]
    P = np.load(os.path.join(info["seq_dir"], "poses.npy")); s = int(info["start_frame"])
    win = P[s:s + 77]
    a0 = ego_actions(P[[max(0, s - VS), s]])
    return np.concatenate([a0, ego_actions(win[::VS])]), win

def action_score(lab, d, dd, i):
    gt, win = gt_actions(d, i)
    pa = os.path.join(d, f"sample_{i:02d}_pred_actions.npy")
    if os.path.exists(pa):                                   # joint-FM action head (run I)
        pred = np.load(pa)
    else:                                                    # raymap runs: inverted camera poses
        cands = [os.path.join(d, f"sample_{i:02d}_pred_poses.npy")]
        if lab in ALL120:
            cands.append(os.path.join(ALL120[lab], f"sample_{i + OFF120[dd]:02d}_pred_poses.npy"))
        pp = next((c for c in cands if os.path.exists(c)), None)
        if pp is None:
            return None
        pred = np.concatenate([gt[:1], ego_actions(np.load(pp)[::VS])])   # a_0 is given
    return action_metrics(pred, gt, win[0, 5])
OUT = f"{RS}/abl/METRICS.md"
CACHE = f"{RS}/abl/.metrics_cache.json"
MID = range(32, 49)

_lp = None
def lpips_fn():
    global _lp
    if _lp is None:
        import torch, lpips
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _lp = (lpips.LPIPS(net="alex", verbose=False).to(dev).eval(), dev)
    return _lp

def lpips_video(g, t):
    import torch
    m, dev = lpips_fn()
    out = []
    with torch.no_grad():
        for i in range(0, len(g), 16):
            a = torch.from_numpy(g[i:i+16]).permute(0, 3, 1, 2).float().div(127.5).sub(1).to(dev)
            b = torch.from_numpy(t[i:i+16]).permute(0, 3, 1, 2).float().div(127.5).sub(1).to(dev)
            out += m(a, b).flatten().tolist()
    return float(np.mean(out))

def done_samples(d):
    if not os.path.isdir(d):
        return set()
    meta = os.path.join(d, "samples_meta.json")
    have = {int(m.group(1)) for f in os.listdir(d) if (m := re.match(r"sample_(\d+)\.mp4$", f))}
    if not os.path.exists(meta):
        return set()
    try:
        mm = json.load(open(meta))
    except Exception:
        return set()
    # mp4 written + its meta row has a goal frame = sample finished
    return {i for i in have if str(i) in mm and "goal_frame" in mm[str(i)]}

def score(d, i):
    from skimage.metrics import structural_similarity as ssim
    info = json.load(open(os.path.join(d, "samples_meta.json")))[str(i)]
    gen = iio.imread(os.path.join(d, f"sample_{i:02d}.mp4"))
    st0 = int(info["start_frame"]); h, w = gen.shape[1:3]
    gt = np.stack([cv2.resize(cv2.cvtColor(cv2.imread(f"{info['seq_dir']}/images/{st0+k:06d}.png"), cv2.COLOR_BGR2RGB),
                              (w, h), interpolation=cv2.INTER_AREA) for k in range(len(gen))])
    mse = ((gen.astype(np.float32) - gt.astype(np.float32)) ** 2).reshape(len(gen), -1).mean(1)
    psnr = float(np.mean(10 * np.log10(255 ** 2 / np.maximum(mse, 1e-8))))
    ss = float(np.mean([ssim(gt[k], gen[k], channel_axis=2, data_range=255) for k in range(len(gen))]))
    hsv = lambda x: cv2.cvtColor(x, cv2.COLOR_RGB2HSV)
    lum = float(np.mean([gen[k].mean() - gt[k].mean() for k in MID]))
    sat = float(np.mean([hsv(gen[k])[..., 1].mean() - hsv(gt[k])[..., 1].mean() for k in MID]))
    return {"psnr": psnr, "ssim": ss, "lpips": lpips_video(gen, gt), "lum_mid": lum, "sat_mid": sat}

def raymap_mse(log):
    if not os.path.exists(log):
        return []
    return [float(x) for x in re.findall(r"raymap MSE ([\d.]+) vs baseline", open(log, errors="ignore").read())]

def main():
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    avail = {lab: {dd: done_samples(p.format(d=dd)) for dd in DIFFS} for lab, p, _ in RUNS}
    common = {dd: set.intersection(*[avail[lab][dd] for lab, _, _ in RUNS]) for dd in DIFFS}
    n_new = 0
    for lab, p, _ in RUNS:
        for dd in DIFFS:
            for i in sorted(common[dd]):
                k = f"{lab}|{dd}|{i}"
                if k not in cache:
                    cache[k] = score(p.format(d=dd), i); n_new += 1
                    if n_new % 20 == 0:
                        json.dump(cache, open(CACHE, "w"))
    json.dump(cache, open(CACHE, "w"))

    # action metrics: cheap, recomputed every call (pred_poses may appear later)
    act, chk = {}, []
    for lab, p, _ in RUNS:
        for dd in DIFFS:
            for i in sorted(common[dd]):
                r = action_score(lab, p.format(d=dd), dd, i)
                if r is not None:
                    act.setdefault(lab, []).append(r)
                mj = os.path.join(p.format(d=dd), f"sample_{i:02d}_metrics.json")
                if r is not None and os.path.exists(mj):          # self-check vs run I's own numbers
                    chk.append(abs(json.load(open(mj))["ade"] - r["ade"]))

    L = [f"# valbench5s interim metrics ({datetime.datetime.utcnow():%m-%d %H:%M} UTC)\n",
         "Paired: only (difficulty, sample) present in every run. "
         f"n = {sum(len(c) for c in common.values())} "
         f"(easy {len(common['easy'])}, medium {len(common['medium'])}, hard {len(common['hard'])}).",
         "lum/sat = gen - GT over frames 32-48 (0 = same as GT). raymap MSE = all samples logged so far.\n",
         "| run | PSNR ^ | SSIM ^ | LPIPS v | lum mid | sat mid | raymap MSE v (n) | progress e/m/h |",
         "|---|---|---|---|---|---|---|---|"]
    for lab, p, lg in RUNS:
        r = [cache[f"{lab}|{dd}|{i}"] for dd in DIFFS for i in sorted(common[dd])]
        rm = sum((raymap_mse(lg.format(d=dd)) for dd in DIFFS), [])
        prog = "/".join(str(len(avail[lab][dd])) for dd in DIFFS)
        if r:
            m = lambda key: st.mean(x[key] for x in r)
            L.append(f"| {lab} | {m('psnr'):.2f} | {m('ssim'):.3f} | {m('lpips'):.3f} | {m('lum_mid'):+.1f} | "
                     f"{m('sat_mid'):+.1f} | {(f'{st.mean(rm):.3f} ({len(rm)})') if rm else '-'} | {prog} |")
        else:
            L.append(f"| {lab} | - | - | - | - | - | - | {prog} |")
    L += ["\n## Action (ego-motion per latent frame, a_1..a_19 integrated; a_0 is given)",
          "Run I: action head output. Raymap runs: camera poses inverted from the predicted raymap "
          "(invert_raymap s3). Same paired samples as above; n < paired n means poses not inverted yet.\n",
          "| run | n | ADE (m) v | FDE (m) v | hold-a0 ADE | beats hold-a0 | yaw MAE (deg/step) v | total rot err (deg) |",
          "|---|---|---|---|---|---|---|---|"]
    for lab, _, _ in RUNS:
        r = act.get(lab, [])
        if r:
            m = lambda key: st.mean(x[key] for x in r)
            L.append(f"| {lab} | {len(r)} | {m('ade'):.3f} | {m('fde'):.3f} | {m('base_ade'):.3f} | "
                     f"{100 * m('beats'):.0f}% | {m('mae_yaw'):.2f} | {st.mean(abs(x['rot_err']) for x in r):.1f} |")
        else:
            L.append(f"| {lab} | 0 | pending | | | | | |")
    if chk:
        L.append(f"\nself-check vs run I _metrics.json: max |ADE diff| = {max(chk):.2e} over {len(chk)} samples")
    txt = "\n".join(L) + "\n"
    open(OUT, "w").write(txt)
    print(txt)
    print(f"[interim] scored {n_new} new samples -> {OUT}")

if __name__ == "__main__":
    main()
