#!/usr/bin/env python3
"""E2: generated raymap (DiT) vs the VAE floor on the same windows.

Reads e2_{easy,medium,hard}.csv / _per_frame.npz from plucker_fidelity.py --mode pred
(each row carries the floor_ GT-latent metrics for the same window).

  fig_e2_pred_vs_floor.{pdf,png}
    (a) per-window ATE: generated vs VAE floor (log-log), y=x and y=3x guides
    (b) geometry of the decoded field: floor vs generated, per metric (median, IQR)
    (c) BEV: GT vs trajectory recovered from the GENERATED raymap (median-ATE window / level)
  table_e2_pred_vs_floor.tex
"""
import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from plucker_fidelity_figure import (GT_C, INK, INK2, MUTED, REC_C, panel_label, style,  # noqa
                                     to_ego, window_motion)

LEVELS = ["easy", "medium", "hard"]
LV_C = {"easy": "#2a78d6", "medium": "#eb6834", "hard": "#1baf7a"}   # categorical slots 1-3
FLOOR_C = "#8a8985"


def load(out_dir):
    rows, pf = {}, {}
    for lv in LEVELS:
        p = os.path.join(out_dir, "e2_%s.csv" % lv)
        if not os.path.exists(p):
            continue
        rows[lv] = list(csv.DictReader(open(p)))
        z = np.load(os.path.join(out_dir, "e2_%s_per_frame.npz" % lv))
        pf[lv] = {k: z[k] for k in z.files}
    return rows, pf


def col(rows, k):
    return np.array([float(r[k]) for r in rows])


METRICS = [  # key, label, scale, fmt
    ("ate_trans_rmse_m", "ATE trans [cm]", 100, "%.1f"),
    ("ate_rot_mean_deg", "ATE rot [°]", 1, "%.2f"),
    ("rpe4_trans_m", "RPE@4 trans [cm]", 100, "%.1f"),
    ("dir_ang_deg_mean", "∠(d̂,d) [°]", 1, "%.2f"),
    ("mom_err_m_mean", "‖m̂−m‖ [cm]", 100, "%.1f"),
    ("plucker_abs_m_mean", "|d̂·m̂| [cm]", 100, "%.1f"),
    ("mom_resid_m", "‖m̂−o×d̂‖ [cm]", 100, "%.1f"),
    ("proc_resid_deg", "Procrustes resid [°]", 1, "%.2f"),
]


def table(rows, path):
    tex = [r"\begin{tabular}{l" + "cc" * len(LEVELS) + "}", r"\toprule",
           " & " + " & ".join(r"\multicolumn{2}{c}{%s (n=%d)}" % (lv, len(rows[lv]))
                              for lv in LEVELS if lv in rows) + r" \\",
           "Metric & " + " & ".join(["floor & gen."] * len(rows)) + r" \\", r"\midrule"]
    md = ["| metric | " + " | ".join("%s floor | %s gen. | ratio" % (lv, lv) for lv in rows) + " |",
          "|---|" + "---|---|---|" * len(rows)]
    for k, lbl, s, fmt in METRICS:
        cells_t, cells_m = [], []
        for lv in LEVELS:
            if lv not in rows:
                continue
            f = np.median(col(rows[lv], "floor_" + k)) * s
            g = np.median(col(rows[lv], k)) * s
            cells_t += [fmt % f, fmt % g]
            cells_m += [fmt % f, fmt % g, "%.1f×" % (g / f)]
        tex.append(lbl.replace("‖", r"$\|$").replace("°", r"$^\circ$") + " & "
                   + " & ".join(cells_t) + r" \\")
        md.append("| %s | %s |" % (lbl, " | ".join(cells_m)))
    above = []
    for lv in rows:
        a = np.array([r["trans_above_floor"] == "True" for r in rows[lv]])
        above.append("%s %d/%d" % (lv, a.sum(), len(a)))
    tex += [r"\bottomrule", r"\end{tabular}"]
    open(path, "w").write("\n".join(tex) + "\n")
    md += ["", "ATE above 3× floor: " + ", ".join(above)]
    return "\n".join(md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="/root/project/egowam/results_syn/plucker_fidelity")
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})

    rows, pf = load(args.out_dir)
    print("[e2] levels:", {k: len(v) for k, v in rows.items()})

    fig = plt.figure(figsize=(7.2, 5.4))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.85], hspace=0.45,
                             left=0.08, right=0.94, top=0.95, bottom=0.08)
    top = outer[0].subgridspec(1, 2, width_ratios=[1.0, 1.45], wspace=0.32)

    # (a) per-window ratio: generated ATE / floor ATE (same window), stacked by level
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
    ax = fig.add_subplot(top[0, 0])
    style(ax)
    ratios = {lv: col(rows[lv], "ate_trans_rmse_m") / col(rows[lv], "floor_ate_trans_rmse_m")
              for lv in rows}
    allr = np.concatenate(list(ratios.values()))
    bins = np.logspace(np.log10(1), np.log10(max(allr.max() * 1.1, 100)), 22)
    ax.hist([ratios[lv] for lv in rows], bins=bins, stacked=True,
            color=[LV_C[lv] for lv in rows], label=list(rows), edgecolor="white", linewidth=0.6)
    ax.set_xscale("log")
    ax.axvline(1, color=INK2, lw=0.8, ls="--")
    ax.axvline(3, color=INK2, lw=0.8, ls=":")
    yt = ax.get_ylim()[1]
    ax.text(1.05, yt * 0.97, "= floor", fontsize=6, color=INK2, va="top")
    ax.text(3.15, yt * 0.97, "3×", fontsize=6, color=INK2, va="top")
    tk = [1, 3, 10, 30, 100]
    ax.xaxis.set_major_locator(FixedLocator(tk))
    ax.xaxis.set_major_formatter(FixedFormatter(["%g×" % t for t in tk]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("generated ATE / VAE-floor ATE (same window)", fontsize=7, color=INK2,
                  labelpad=1)
    ax.set_ylabel("windows", fontsize=7, color=INK2, labelpad=1)
    ax.legend(fontsize=6, frameon=False, loc="upper right")
    ax.text(0.97, 0.62, "median %.1f×\n%d/%d windows > 3×"
            % (np.median(allr), (allr > 3).sum(), len(allr)),
            transform=ax.transAxes, fontsize=6.5, color=INK, ha="right", va="top")
    panel_label(ax, "a")

    # (b) geometry metrics: floor vs generated (all levels pooled), ratio bars on log axis
    ax = fig.add_subplot(top[0, 1])
    style(ax)
    pooled = sum((rows[lv] for lv in rows), [])
    ks = [m for m in METRICS if m[0] != "rpe4_trans_m"]
    y = np.arange(len(ks))[::-1]
    for i, (k, lbl, s, fmt) in enumerate(ks):
        f = col(pooled, "floor_" + k) * s
        g = col(pooled, k) * s
        for v, c, dy in ((f, FLOOR_C, 0.17), (g, REC_C, -0.17)):
            q1, q2, q3 = np.percentile(v, [25, 50, 75])
            ax.plot([q1, q3], [y[i] + dy] * 2, color=c, lw=2.0, solid_capstyle="round")
            ax.plot(q2, y[i] + dy, "o", ms=4.5, color=c, mec="white", mew=0.6, zorder=3)
        ax.text(1.01, y[i], "%.1f×" % (np.median(g) / np.median(f)),
                transform=ax.get_yaxis_transform(), fontsize=6.5, color=INK, va="center",
                clip_on=False)
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels([m[1] for m in ks], fontsize=6.5, color=INK2)
    ax.plot([], [], "o-", color=FLOOR_C, ms=4, lw=2, label="VAE floor (GT raymap)")
    ax.plot([], [], "o-", color=REC_C, ms=4, lw=2, label="generated raymap")
    ax.legend(fontsize=6, frameon=False, loc="lower right", handlelength=1.6)
    ax.set_xlabel("median (bar: IQR), log scale — label: generated / floor",
                  fontsize=6.5, color=INK2, labelpad=1)
    ax.grid(axis="y", visible=False)
    panel_label(ax, "b")

    # (c) BEV: median-ATE window per level
    bot = outer[1].subgridspec(1, len(rows), wspace=0.35)
    for j, lv in enumerate(rows):
        a = col(rows[lv], "ate_trans_rmse_m")
        i = int(np.argsort(a)[len(a) // 2])
        ids = [str(s) for s in pf[lv]["ids"]]
        k = ids.index(rows[lv][i]["id"])
        g, r = pf[lv]["gt_poses"][k], pf[lv]["rec_poses"][k]
        ax = fig.add_subplot(bot[0, j])
        style(ax)
        gx, gy = to_ego(g, g[0])
        rx, ry = to_ego(r, g[0])
        ax.plot(gx, gy, color=GT_C, lw=3.0, solid_capstyle="round", label="GT", zorder=2)
        ax.plot(rx, ry, color=LV_C[lv], lw=1.3, ls=(0, (3, 1.5)), label="generated raymap",
                zorder=3)
        ax.plot(gx[0], gy[0], "o", ms=3.5, color=INK, zorder=4)
        ax.set_aspect("equal", adjustable="datalim")
        _, yaw = window_motion(g)
        ax.set_title("%s · median window · Δyaw %+.0f°\nATE %.0f cm  (VAE floor %.0f cm)"
                     % (lv, yaw, 100 * a[i], 100 * float(rows[lv][i]["floor_ate_trans_rmse_m"])),
                     fontsize=6.3, color=INK2, pad=3)
        ax.set_xlabel("right [m]", fontsize=6.5, color=INK2, labelpad=1)
        if j == 0:
            ax.set_ylabel("forward [m]", fontsize=6.5, color=INK2, labelpad=1)
            panel_label(ax, "c")
            ax.legend(fontsize=6, frameon=False, loc="best", handlelength=2.2)

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(args.out_dir, "fig_e2_pred_vs_floor.%s" % ext), dpi=220)
    plt.close(fig)
    md = table(rows, os.path.join(args.out_dir, "table_e2_pred_vs_floor.tex"))
    open(os.path.join(args.out_dir, "summary_e2.md"), "w").write(md + "\n")
    print(md)
    print("[done]")


if __name__ == "__main__":
    main()
