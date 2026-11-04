#!/usr/bin/env python3
"""E2 self-consistency: is the GENERATED raymap a valid raymap of *some* rigid camera?

Reference = the clean Plücker field of the generated raymap's own best-fit camera (per-frame
Procrustes R + least-squares o, invert_frame). Its VAE floor ("self-floor") = that clean field of
the recovered trajectory through VAE encode-decode (plucker_fidelity.py --mode pred --self_ref).

  fig_e2_self_consistency.{pdf,png}
    (a) per metric: E1 (GT->VAE) | E2 floor (GT->VAE) | E2 self-floor (gen traj->VAE) | generated
    (b) per window: generated vs self-floor  ‖m̂ − o×d̂‖, y=x and y=3x guides
    (c) residual maps ‖m̂ − o×d̂‖ of the median window: self-floor vs generated, 3 frames
  table_e2_self_consistency.tex

Panel (c) needs the VAE once (cached to e2_self_maps.npz); --no_gpu reuses the cache.
"""
import argparse
import csv
import glob
import os
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from plucker_fidelity_figure import INK, INK2, REC_C, panel_label, style  # noqa: E402

LEVELS = ["easy", "medium", "hard"]
LV_C = {"easy": "#2a78d6", "medium": "#eb6834", "hard": "#1baf7a"}
COND = [  # key prefix, source, label, color
    ("", "e1", "E1: GT raymap → VAE", "#c9c8c3"),
    ("floor_", "e2", "E2 floor: GT raymap → VAE", "#a3a29d"),
    ("selffloor_", "e2", "E2 self-floor: gen. trajectory → VAE", "#6b6a66"),
    ("", "e2", "E2 generated raymap", REC_C),
]
METRICS = [  # key, label, scale, fmt
    ("proc_resid_deg", "∠(d̂, R·d_cam) [°]", 1, "%.2f"),
    ("mom_resid_m", "‖m̂ − o×d̂‖ [cm]", 100, "%.1f"),
    ("plucker_abs_m_mean", "|d̂·m̂| [cm]", 100, "%.1f"),
    ("dnorm_err_mean", "|‖d̂‖−1| [×10⁻³]", 1000, "%.1f"),
]
FRAMES = (20, 48, 76)


def rows_of(pat):
    return sum((list(csv.DictReader(open(p))) for p in sorted(glob.glob(pat))), [])


def col(rows, k):
    return np.array([float(r[k]) for r in rows])


def residual_maps(d, m, intr4, orig_wh, frames):
    import invert_raymap as ir
    F, H, W, _ = d.shape
    A = ir.pixel_ray_dirs(intr4, H, W, orig_wh)
    out = []
    for f in frames:
        R, o, _ = ir.invert_frame(d[f].reshape(-1, 3), m[f].reshape(-1, 3), A)
        dh = np.einsum("ij,nj->ni", R, A)
        r = np.linalg.norm(m[f].reshape(-1, 3) - np.cross(np.broadcast_to(o, dh.shape), dh), axis=-1)
        out.append(r.reshape(H, W))
    return np.stack(out)


def compute_maps(out_dir, lv, sample_id, cache):
    import torch
    import invert_raymap as ir
    import plucker_fidelity as pfm
    a = types.SimpleNamespace(height=512, width=512, num_frames=77, ray_downsample=2,
                              raymap_moment_scale=8.0, device="cuda", max_windows=0,
                              intrinsics_override=None,
                              pred_dir=os.path.join(out_dir, "e2", "full", lv))
    w = [x for x in pfm.pred_windows(a) if x["id"] == sample_id][0]
    vae, lmean, lstd = ir.load_vae("nvidia/Cosmos-Predict2.5-2B", "diffusers/base/post-trained",
                                   "cuda", "bf16")
    lat = torch.from_numpy(np.load(w["lat"])).cuda().float()
    lat = lat[None] if lat.ndim == 4 else lat
    d, m = pfm.decode_raw(vae, lat, lmean, lstd, 8.0)
    gen = residual_maps(d, m, w["intr"], (512, 512), FRAMES)
    pa, _, _ = ir.invert_rays(d / np.linalg.norm(d, axis=-1, keepdims=True), m, w["intr"],
                              (512, 512), w["poses"][0], 8.0)
    with torch.no_grad():
        s_lat = ir.build_gt_latent(vae, lmean, lstd, pa, w["intr"], a, "cuda")
    d2, m2 = pfm.decode_raw(vae, s_lat, lmean, lstd, 8.0)
    sf = residual_maps(d2, m2, w["intr"], (512, 512), FRAMES)
    np.savez(cache, gen=gen, selffloor=sf, lv=lv, id=sample_id)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="/root/project/egowam/results_syn/plucker_fidelity")
    ap.add_argument("--no_gpu", action="store_true")
    args = ap.parse_args()
    od = args.out_dir
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})

    e1 = rows_of(os.path.join(od, "e1_shard*of4.csv"))
    e2 = {lv: rows_of(os.path.join(od, "e2s_%s.csv" % lv)) for lv in LEVELS}
    e2all = sum(e2.values(), [])
    src = {"e1": e1, "e2": e2all}
    print("[self] E1 %d  E2 %d" % (len(e1), len(e2all)))

    # median window (pooled) by generated mom_resid, for panel (c)
    g = col(e2all, "mom_resid_m")
    i_med = int(np.argsort(g)[len(g) // 2])
    lv_med = [lv for lv in LEVELS for _ in e2[lv]][i_med]
    id_med = e2all[i_med]["id"]
    cache = os.path.join(od, "e2_self_maps.npz")
    if not args.no_gpu and not (os.path.exists(cache)
                                and str(np.load(cache)["id"]) == id_med
                                and str(np.load(cache)["lv"]) == lv_med):
        compute_maps(od, lv_med, id_med, cache)
    maps = np.load(cache)

    fig = plt.figure(figsize=(7.2, 5.6))
    top = fig.add_gridspec(1, 2, width_ratios=[1.45, 1.0], wspace=0.42,
                           left=0.17, right=0.97, top=0.93, bottom=0.53)

    # (a) per-metric dot + IQR, 4 conditions
    ax = fig.add_subplot(top[0, 0])
    style(ax)
    y = np.arange(len(METRICS))[::-1].astype(float)
    offs = np.linspace(0.27, -0.27, len(COND))
    for i, (k, lbl, s, fmt) in enumerate(METRICS):
        meds = []
        for (pre, sk, cl, c), dy in zip(COND, offs):
            v = col(src[sk], pre + k) * s
            q1, q2, q3 = np.percentile(v, [25, 50, 75])
            meds.append(q2)
            ax.plot([q1, q3], [y[i] + dy] * 2, color=c, lw=2.0, solid_capstyle="round")
            ax.plot(q2, y[i] + dy, "o", ms=4.2, color=c, mec="white", mew=0.6, zorder=3)
        ax.text(1.01, y[i] + offs[-1], "%.1f×" % (meds[3] / meds[2]),
                transform=ax.get_yaxis_transform(), fontsize=6.5, color=INK, va="center",
                clip_on=False)
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels([m[1] for m in METRICS], fontsize=6.5, color=INK2)
    ax.grid(axis="y", visible=False)
    for pre, sk, cl, c in COND:
        ax.plot([], [], "o-", color=c, ms=4, lw=2, label=cl)
    ax.legend(fontsize=5.6, frameon=False, loc="upper center", bbox_to_anchor=(0.4, -0.12),
              ncol=2, handlelength=1.4, columnspacing=1.0)
    ax.set_title("residual vs own best-fit camera (median, bar: IQR, log)\n"
                 "label: generated / self-floor", fontsize=6.5, color=INK2, pad=3)
    panel_label(ax, "a")

    # (b) per-window scatter
    ax = fig.add_subplot(top[0, 1])
    style(ax)
    for lv in LEVELS:
        x = col(e2[lv], "selffloor_mom_resid_m") * 100
        yy = col(e2[lv], "mom_resid_m") * 100
        ax.plot(x, yy, "o", ms=3.6, color=LV_C[lv], mec="white", mew=0.4, alpha=0.9,
                label=lv, zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    lo, hi = 1.5, 80
    t = np.array([lo, hi])
    ax.plot(t, t, color=INK2, lw=0.8, ls="--")
    ax.plot(t, 3 * t, color=INK2, lw=0.8, ls=":")
    ax.text(30, 24, "y = x", fontsize=6, color=INK2, rotation=0)
    ax.text(9, 34, "3×", fontsize=6, color=INK2)
    ax.set_xlim(lo, 15)
    ax.set_ylim(lo, hi)
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
    for axis, tk in ((ax.xaxis, [2, 5, 10]), (ax.yaxis, [2, 5, 10, 20, 50])):
        axis.set_major_locator(FixedLocator(tk))
        axis.set_major_formatter(FixedFormatter(["%g" % t for t in tk]))
        axis.set_minor_locator(NullLocator())
    ax.tick_params(labelsize=6.5)
    xs = col(e2all, "selffloor_mom_resid_m")
    gs_ = col(e2all, "mom_resid_m")
    ax.text(0.97, 0.05, "%d/%d windows > 3×\nmedian %.1f×"
            % ((gs_ > 3 * xs).sum(), len(xs), np.median(gs_ / xs)),
            transform=ax.transAxes, fontsize=6.3, color=INK, ha="right", va="bottom")
    ax.set_xlabel("self-floor ‖m̂ − o×d̂‖ [cm]", fontsize=6.5, color=INK2, labelpad=1)
    ax.set_ylabel("generated ‖m̂ − o×d̂‖ [cm]", fontsize=6.5, color=INK2, labelpad=1)
    ax.legend(fontsize=6, frameon=False, loc="upper left", handletextpad=0.2)
    panel_label(ax, "b")

    # (c) residual maps, shared colour scale
    bot = fig.add_gridspec(2, len(FRAMES) + 1, width_ratios=[1] * len(FRAMES) + [0.06],
                           wspace=0.06, hspace=0.08, left=0.2, right=0.8, top=0.36,
                           bottom=0.02)
    vmax = float(np.percentile(maps["gen"], 99)) * 100
    for r, (key, name) in enumerate((("selffloor", "self-floor"), ("gen", "generated"))):
        for c, f in enumerate(FRAMES):
            ax = fig.add_subplot(bot[r, c])
            im = ax.imshow(maps[key][c] * 100, cmap="Blues", vmin=0, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if r == 0:
                ax.set_title("frame %d" % f, fontsize=6.5, color=INK2, pad=2)
            if c == 0:
                ax.set_ylabel(name, fontsize=6.5, color=INK2)
                if r == 0:
                    panel_label(ax, "c")
            ax.text(0.03, 0.04, "mean %.1f cm" % (maps[key][c].mean() * 100), fontsize=5.6,
                    color=INK, transform=ax.transAxes,
                    bbox=dict(fc="white", ec="none", alpha=0.75, pad=1))
    cax = fig.add_subplot(bot[:, -1])
    cb = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=5.6, colors=INK2)
    cb.outline.set_visible(False)
    cb.set_label("‖m̂ − o×d̂‖ [cm]", fontsize=6, color=INK2)
    fig.text(0.5, 0.385, "median window (%s, %s): residual map w.r.t. its own best-fit camera"
             % (lv_med, id_med), fontsize=6.3, color=INK2, ha="center")

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(od, "fig_e2_self_consistency.%s" % ext), dpi=220)
    plt.close(fig)

    # table
    tex = [r"\begin{tabular}{lcccc}", r"\toprule",
           r"Metric & E1 (GT$\to$VAE) & E2 floor & E2 self-floor & E2 generated \\", r"\midrule"]
    for k, lbl, s, fmt in METRICS:
        v = [np.median(col(src[sk], pre + k)) * s for pre, sk, _, _ in COND]
        tex.append(lbl.replace("‖", r"$\|$").replace("°", r"$^\circ$").replace("⁻³", "$^{-3}$")
                   + " & " + " & ".join(fmt % x for x in v[:3])
                   + " & %s (%.1f$\\times$)" % (fmt % v[3], v[3] / v[2]) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(od, "table_e2_self_consistency.tex"), "w").write("\n".join(tex) + "\n")
    print("\n".join(tex))
    print("[done]")


if __name__ == "__main__":
    main()
