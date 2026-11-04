#!/usr/bin/env python3
"""Aggregate plucker_fidelity.py shards -> summary.md + per-frame plot.

  python plucker_fidelity_summary.py --out_dir <dir> --tag e1 [--tag e2]
"""
import argparse
import csv
import glob
import os

import numpy as np

KEYS = [
    ("dnorm_err_mean", "|‖d̂‖−1| mean", "{:.2e}"),
    ("plucker_cos_p50", "|cos(d̂,m̂)| p50", "{:.4f}"),
    ("plucker_cos_p95", "|cos(d̂,m̂)| p95", "{:.4f}"),
    ("plucker_abs_m_mean", "|d̂·m̂| mean [m]", "{:.4f}"),
    ("dir_ang_deg_mean", "∠(d̂,d) mean [°]", "{:.3f}"),
    ("mom_err_m_mean", "‖m̂−m‖ mean [m]", "{:.4f}"),
    ("mom_par_over_err", "‖m̂∥d̂‖ / ‖m̂−m‖", "{:.3f}"),
    ("proc_resid_deg", "Procrustes resid [°]", "{:.3f}"),
    ("mom_resid_m", "‖m̂−o×d̂‖ [m]", "{:.4f}"),
    ("ate_trans_rmse_m", "ATE trans [m]", "{:.4f}"),
    ("proj_ate_trans_rmse_m", "ATE trans, projected [m]", "{:.4f}"),
    ("ate_rot_mean_deg", "ATE rot [°]", "{:.3f}"),
    ("rpe1_trans_m", "RPE@1 trans [m]", "{:.4f}"),
    ("rpe4_trans_m", "RPE@4 trans [m]", "{:.4f}"),
    ("rpe1_rot_deg", "RPE@1 rot [°]", "{:.3f}"),
    ("abs_trans_err_last", "trans err @last [m]", "{:.4f}"),
]


def load(out_dir, tag):
    rows = []
    for p in sorted(glob.glob(os.path.join(out_dir, "%s*.csv" % tag))):
        if os.path.basename(p).startswith(tag + "_shard") or os.path.basename(p) == tag + ".csv":
            rows += list(csv.DictReader(open(p)))
    pfs = {}
    for p in sorted(glob.glob(os.path.join(out_dir, "%s*_per_frame.npz" % tag))):
        z = np.load(p)
        for k in z.files:
            if k != "ids":
                pfs.setdefault(k, []).append(z[k])
    pfs = {k: np.concatenate(v) for k, v in pfs.items()}
    return rows, pfs


def col(rows, k):
    out = []
    for r in rows:
        try:
            out.append(float(r[k]))
        except (KeyError, ValueError, TypeError):
            out.append(np.nan)
    return np.array(out)


def table(groups, prefix=""):
    names = list(groups)
    hdr = "| metric | " + " | ".join("%s (n=%d)" % (n, len(groups[n])) for n in names) + " |"
    lines = [hdr, "|---|" + "---|" * len(names)]
    for k, lbl, fmt in KEYS:
        cells = []
        for n in names:
            v = col(groups[n], prefix + k)
            v = v[np.isfinite(v)]
            cells.append("—" if v.size == 0 else
                         (fmt.format(np.median(v)) + " / " + fmt.format(np.percentile(v, 95))))
        lines.append("| %s | %s |" % (lbl, " | ".join(cells)))
    return "\n".join(lines)


def per_frame_plot(pfs, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ks = [k for k in ("plucker_cos", "plucker_abs_m", "dnorm_err", "dir_ang_deg", "mom_err_m")
          if k in pfs]
    fig, ax = plt.subplots(1, len(ks), figsize=(3.6 * len(ks), 3), squeeze=False)
    for a, k in zip(ax[0], ks):
        v = pfs[k]
        T = v.shape[1]
        a.plot(np.nanmedian(v, 0), lw=1.2, label="median")
        a.fill_between(np.arange(T), np.nanpercentile(v, 25, 0), np.nanpercentile(v, 75, 0),
                       alpha=0.3)
        for b in range(1, T, 4):                     # first frame of each latent chunk
            a.axvline(b, color="k", lw=0.3, alpha=0.3)
        a.set_title(k, fontsize=9); a.set_xlabel("frame")
    fig.suptitle(title + "  (grey lines: 4-frame VAE temporal chunks)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="/root/project/egowam/results_syn/plucker_fidelity")
    ap.add_argument("--tag", action="append", default=None)
    args = ap.parse_args()
    tags = args.tag or ["e1"]
    md = ["# Plücker fidelity through the Wan VAE (production raymap: dir | mom, separate encode)",
          "", "cells = median / p95 over windows. moment_scale 8, ray grid 256², 77 frames.", ""]
    for tag in tags:
        rows, pfs = load(args.out_dir, tag)
        if not rows:
            continue
        md += ["## %s  (%d windows)" % (tag, len(rows)), ""]
        disp = col(rows, "max_disp_m")
        town = [r["seq"].split("/")[0] if r.get("seq") else "?" for r in rows]
        groups = {"all": rows}
        for lo, hi, n in ((0, 0.5, "stationary"), (0.5, 4, "slow"), (4, 1e9, "fast")):
            g = [r for r, d in zip(rows, disp) if lo <= d < hi]
            if g:
                groups["%s [%g–%g m)" % (n, lo, hi if hi < 1e8 else np.inf)] = g
        md += ["### by window displacement", "", table(groups), ""]
        tg = {t: [r for r, tt in zip(rows, town) if tt == t] for t in sorted(set(town))}
        md += ["### by town", "", table(tg), ""]
        if any("floor_ate_trans_rmse_m" in r for r in rows):
            md += ["### VAE floor (GT latent, same windows)", "",
                   table({"floor": rows}, prefix="floor_"), ""]
            ab = col(rows, "trans_above_floor")
            md += ["trans error above floor (>3×) in %d / %d windows" % (np.nansum(ab), len(ab)), ""]
            lr = col(rows, "lat_rel_err")
            md += ["latent rel. err pred vs GT: median %.3f" % np.nanmedian(lr), ""]
        if pfs:
            png = os.path.join(args.out_dir, "%s_per_frame.png" % tag)
            per_frame_plot(pfs, png, tag)
            md += ["![per-frame](%s)" % os.path.basename(png), ""]
            if "plucker_cos" in pfs:
                v = np.nanmedian(pfs["plucker_cos"], 0)
                ph = [np.nanmean(v[1:][(np.arange(len(v) - 1) % 4) == j]) for j in range(4)]
                md += ["plucker_cos by position in 4-frame chunk (frames 1..76): "
                       + ", ".join("%d: %.4f" % (j, p) for j, p in enumerate(ph)), ""]
    p = os.path.join(args.out_dir, "summary.md")
    open(p, "w").write("\n".join(md))
    print("\n".join(md))
    print("[done]", p)


if __name__ == "__main__":
    main()
