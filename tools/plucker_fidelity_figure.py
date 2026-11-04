#!/usr/bin/env python3
"""Paper figure: the production raymap (dir | mom, separate VAE encode) survives the VAE.

  (a) GT vs VAE-reconstructed raymap (dir, mom) at 3 frames + error maps   [one window, GPU]
  (b) BEV trajectories: GT vs recovered from the reconstructed raymap      [4 windows]
  (c) information preserved: net displacement & total yaw, GT vs recovered [all windows]
  (d) recovery error vs motion magnitude                                   [all windows]
  + LaTeX table of the headline numbers.

Inputs: e1_shard*of4.csv / e1_shard*of4_per_frame.npz from plucker_fidelity.py --mode gt.
Run with RAYMAP_CAM_FIX=opencv (panel a re-encodes one window).
"""
import argparse
import csv
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# palette (reference instance, light mode; print figure)
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8985"
GRID = "#e4e3df"
GT_C = "#8a8985"        # GT: neutral
REC_C = "#2a78d6"       # reconstructed: categorical slot 1
SEQ_CMAP = "Blues"      # magnitude: one hue, light -> dark


def load(out_dir, tag):
    rows = []
    for p in sorted(glob.glob(os.path.join(out_dir, "%s_shard*of*.csv" % tag))):
        rows += list(csv.DictReader(open(p)))
    pf = {}
    for p in sorted(glob.glob(os.path.join(out_dir, "%s_shard*of*_per_frame.npz" % tag))):
        z = np.load(p)
        for k in z.files:
            pf.setdefault(k, []).append(z[k])
    pf = {k: np.concatenate(v) for k, v in pf.items()}
    return rows, pf


def fcol(rows, k):
    return np.array([float(r[k]) for r in rows])


def window_motion(P):
    """(F,6) CARLA rows -> net displacement [m], total |yaw change| [deg] (unwrapped)."""
    disp = float(np.linalg.norm(P[-1, :2] - P[0, :2]))
    yaw = np.degrees(np.unwrap(np.radians(P[:, 5])))
    return disp, float(yaw[-1] - yaw[0])


def to_ego(P, ref):
    """BEV coordinates relative to ref pose: forward up, right to the right."""
    th = np.radians(ref[5])
    d = P[:, :2] - ref[:2]
    fwd = d[:, 0] * np.cos(th) + d[:, 1] * np.sin(th)
    right = -d[:, 0] * np.sin(th) + d[:, 1] * np.cos(th)   # CARLA: y points right
    return right, fwd


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
        ax.spines[s].set_linewidth(0.6)
    ax.tick_params(colors=INK2, labelsize=7, width=0.6, length=2.5)
    ax.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)


def panel_label(ax, s):
    ax.text(-0.02, 1.04, s, transform=ax.transAxes, fontsize=9, fontweight="bold",
            color=INK, ha="right", va="bottom")


# ── (a) qualitative ────────────────────────────────────────────────────────────────────────
def recon_window(seq_dir, start, args):
    import torch
    import invert_raymap as ir
    from plucker_fidelity import decode_raw, gt_field
    P = np.load(os.path.join(seq_dir, "poses.npy")).astype(np.float64)[start:start + 77]
    I = np.load(os.path.join(seq_dir, "intrinsics.npy")).astype(np.float64).ravel()[:4]
    vae, lm, ls = ir.load_vae(args.model_name, args.revision, args.device, "bf16")
    with torch.no_grad():
        lat = ir.build_gt_latent(vae, lm, ls, P, I, args, args.device)
    d_hat, m_hat = decode_raw(vae, lat, lm, ls, args.raymap_moment_scale)
    d_gt, m_gt = gt_field(P, I, args, args.device)
    return d_gt, m_gt, d_hat, m_hat


def panel_a(fig, gs, fields, ms, frames=(20, 48, 76)):
    import matplotlib.pyplot as plt  # noqa: F401
    d_gt, m_gt, d_hat, m_hat = fields
    vis_d = lambda d: np.clip((d + 1) / 2, 0, 1)
    vis_m = lambda m: np.clip((m / ms + 1) / 2, 0, 1)
    du = d_hat / np.linalg.norm(d_hat, axis=-1, keepdims=True)
    ang = np.degrees(np.arccos(np.clip((du * d_gt).sum(-1), -1, 1)))
    merr = np.linalg.norm(m_hat - m_gt, axis=-1) * 100.0       # cm
    cols = [("direction  d", "d", f) for f in frames] + [("moment  m", "m", f) for f in frames]
    rows = ["GT", "VAE recon.", "error"]
    sub = gs.subgridspec(3, len(cols), wspace=0.04, hspace=0.06)
    ims = {}
    for r, rn in enumerate(rows):
        for c, (title, kind, f) in enumerate(cols):
            ax = fig.add_subplot(sub[r, c])
            if r == 0:
                img = vis_d(d_gt[f]) if kind == "d" else vis_m(m_gt[f])
                ax.imshow(img)
                ax.set_title("t=%d" % f, fontsize=7, color=INK2, pad=2)
            elif r == 1:
                img = vis_d(d_hat[f]) if kind == "d" else vis_m(m_hat[f])
                ax.imshow(img)
            else:
                if kind == "d":
                    ims["d"] = ax.imshow(ang[f], cmap=SEQ_CMAP, vmin=0, vmax=2.0)
                else:
                    ims["m"] = ax.imshow(merr[f], cmap=SEQ_CMAP, vmin=0, vmax=20.0)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_linewidth(0.4); s.set_color(MUTED)
            if c == 0:
                ax.set_ylabel(rn, fontsize=7, color=INK2)
            if r == 0 and c in (0, len(frames)):
                ax.text(0.0, 1.32, title, transform=ax.transAxes, fontsize=8, color=INK,
                        ha="left")
            if r == 0 and c == 0:
                panel_label(ax, "a")
            if r == 2 and c == len(frames) - 1:
                cax = ax.inset_axes([1.04, 0.0, 0.06, 1.0])
                cb = fig.colorbar(ims["d"], cax=cax)
                cb.set_label("∠ err [°]", fontsize=6, color=INK2)
                cb.ax.tick_params(labelsize=5.5, colors=INK2, width=0.4, length=2)
                cb.outline.set_linewidth(0.4)
            if r == 2 and c == len(cols) - 1:
                cax = ax.inset_axes([1.04, 0.0, 0.06, 1.0])
                cb = fig.colorbar(ims["m"], cax=cax)
                cb.set_label("‖Δm‖ [cm]", fontsize=6, color=INK2)
                cb.ax.tick_params(labelsize=5.5, colors=INK2, width=0.4, length=2)
                cb.outline.set_linewidth(0.4)


# ── (b) trajectories ───────────────────────────────────────────────────────────────────────
def pick_traj(pf, n=4):
    """Diverse, non-cherry-picked windows: the window closest to each yaw-change target."""
    G, R = pf["gt_poses"], pf["rec_poses"]
    mot = np.array([window_motion(g) for g in G])
    ok = mot[:, 0] > 3.0
    out = []
    for target in (0.0, 45.0, -90.0, 160.0)[:n]:
        cand = np.where(ok)[0]
        i = cand[np.argmin(np.abs(mot[cand, 1] - target))]
        out.append(i)
    return out


def panel_b(fig, gs, pf, rows, idx):
    sub = gs.subgridspec(1, len(idx), wspace=0.35)
    ate = {r["id"]: float(r["ate_trans_rmse_m"]) for r in rows}
    for j, i in enumerate(idx):
        ax = fig.add_subplot(sub[0, j])
        style(ax)
        g, r = pf["gt_poses"][i], pf["rec_poses"][i]
        gx, gy = to_ego(g, g[0])
        rx, ry = to_ego(r, g[0])
        ax.plot(gx, gy, color=GT_C, lw=3.0, solid_capstyle="round", label="GT", zorder=2)
        ax.plot(rx, ry, color=REC_C, lw=1.2, ls=(0, (3, 1.5)), label="from VAE recon.", zorder=3)
        ax.plot(gx[0], gy[0], "o", ms=3.5, color=INK, zorder=4)
        ax.set_aspect("equal", adjustable="datalim")
        _, yaw = window_motion(g)
        ax.set_title("Δyaw %+.0f°  ·  ATE %.1f cm" % (yaw, 100 * ate.get(str(pf["ids"][i]), np.nan)),
                     fontsize=6.5, color=INK2, pad=3)
        ax.set_xlabel("right [m]", fontsize=6.5, color=INK2, labelpad=1)
        if j == 0:
            ax.set_ylabel("forward [m]", fontsize=6.5, color=INK2, labelpad=1)
            panel_label(ax, "b")
            ax.legend(fontsize=6, frameon=False, loc="upper left", handlelength=2.2)


# ── (c) information preserved, (d) error vs motion ─────────────────────────────────────────
def scatter_xy(ax, x, y, xl, yl, unit):
    style(ax)
    lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
    pad = 0.04 * (hi - lo)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=MUTED, lw=0.8, ls="--", zorder=1)
    ax.scatter(x, y, s=10, color=REC_C, alpha=0.75, edgecolor="white", linewidth=0.4, zorder=2)
    ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(lo - pad, hi + pad)
    ax.set_aspect("equal")
    r2 = 1 - np.sum((y - x) ** 2) / np.sum((x - x.mean()) ** 2)
    mae = np.median(np.abs(y - x))
    ax.text(0.04, 0.96, "R² = %.4f\nmedian |err| = %s" % (r2, unit(mae)),
            transform=ax.transAxes, fontsize=6.5, color=INK, va="top")
    ax.set_xlabel(xl, fontsize=7, color=INK2, labelpad=1)
    ax.set_ylabel(yl, fontsize=7, color=INK2, labelpad=1)
    return r2, mae


def panel_cd(fig, gs, pf, rows):
    G, R = pf["gt_poses"], pf["rec_poses"]
    mg = np.array([window_motion(g) for g in G])
    mr = np.array([window_motion(r) for r in R])
    sub = gs.subgridspec(1, 3, wspace=0.55)
    ax = fig.add_subplot(sub[0, 0])
    stats = {}
    stats["disp"] = scatter_xy(ax, mg[:, 0], mr[:, 0], "GT net displacement [m]",
                               "recovered [m]", lambda v: "%.1f cm" % (100 * v))
    panel_label(ax, "c")
    ax = fig.add_subplot(sub[0, 1])
    stats["yaw"] = scatter_xy(ax, mg[:, 1], mr[:, 1], "GT total Δyaw [°]", "recovered [°]",
                              lambda v: "%.2f°" % v)
    # (d)
    ax = fig.add_subplot(sub[0, 2])
    style(ax)
    ids = [str(i) for i in pf["ids"]]
    ate = {r["id"]: 100 * float(r["ate_trans_rmse_m"]) for r in rows}
    path = np.array([np.linalg.norm(np.diff(g[:, :3], axis=0), axis=1).sum() for g in G])
    a = np.array([ate[i] for i in ids])
    ax.scatter(path, a, s=10, color=REC_C, alpha=0.75, edgecolor="white", linewidth=0.4)
    med = np.median(a)
    ax.axhline(med, color=INK2, lw=0.8, ls="--")
    ax.set_ylim(0, max(12, np.percentile(a, 99.5) * 1.2))
    ax.set_xlabel("GT path length [m]", fontsize=7, color=INK2, labelpad=1)
    ax.set_ylabel("ATE [cm]", fontsize=7, color=INK2, labelpad=1)
    moving = path > 1.0
    rel = np.median(a[moving] / (100 * path[moving])) * 100
    ax.text(0.04, 0.96, "median ATE = %.1f cm (dashed)\nATE / path = %.2f%%" % (med, rel),
            transform=ax.transAxes,
            fontsize=6.5, color=INK, va="top")
    panel_label(ax, "d")
    stats["rel_pct"] = rel
    return stats


def latex_table(rows, path):
    def q(k, s=1.0, fmt="%.2f"):
        v = fcol(rows, k) * s
        return (fmt % np.median(v)) + " / " + (fmt % np.percentile(v, 95))
    lines = [
        r"\begin{tabular}{llc}",
        r"\toprule",
        r"Level & Metric & median / p95 \\",
        r"\midrule",
        r"Ray & $|\lVert\hat d\rVert-1|$ ($\times10^{-3}$) & %s \\" % q("dnorm_err_mean", 1e3),
        r"    & $\angle(\hat d, d)$ [$^\circ$] & %s \\" % q("dir_ang_deg_mean"),
        r"    & $\lVert\hat m-m\rVert$ [cm] & %s \\" % q("mom_err_m_mean", 100, "%.1f"),
        r"    & $|\hat d\cdot\hat m|$ [cm] & %s \\" % q("plucker_abs_m_mean", 100, "%.1f"),
        r"Camera & $\lVert\hat m-o\times\hat d\rVert$ [cm] & %s \\" % q("mom_resid_m", 100, "%.1f"),
        r"       & Procrustes residual [$^\circ$] & %s \\" % q("proc_resid_deg"),
        r"Trajectory & ATE trans [cm] & %s \\" % q("ate_trans_rmse_m", 100, "%.1f"),
        r"           & ATE rot [$^\circ$] & %s \\" % q("ate_rot_mean_deg"),
        r"           & RPE@4 trans [cm] & %s \\" % q("rpe4_trans_m", 100, "%.1f"),
        r"           & ATE after $d\!\cdot\!m{=}0$ projection [cm] & %s \\"
        % q("proj_ate_trans_rmse_m", 100, "%.1f"),
        r"\bottomrule",
        r"\end{tabular}",
    ]
    open(path, "w").write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="/root/project/egowam/results_syn/plucker_fidelity")
    ap.add_argument("--tag", default="e1")
    ap.add_argument("--data_root", default="/root/data/carla_x")
    ap.add_argument("--model_name", default="nvidia/Cosmos-Predict2.5-2B")
    ap.add_argument("--revision", default="diffusers/base/post-trained")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--ray_downsample", type=int, default=2)
    ap.add_argument("--raymap_moment_scale", type=float, default=8.0)
    ap.add_argument("--max_step", type=float, default=1.0)
    ap.add_argument("--qual_window", default=None, help="id 'Town/Weather/seq_XXXX@start'")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})

    rows, pf = load(args.out_dir, args.tag)
    assert "rec_poses" in pf, "rerun plucker_fidelity.py (needs rec_poses in per_frame npz)"
    # drop simulator glitches: walking is ~0.08 m/frame (p99 0.11); >1 m/frame is a teleport/
    # fall-through-map, whose moment saturates the clamp -- a data defect, not a VAE property
    step = np.linalg.norm(np.diff(pf["gt_poses"][:, :, :3], axis=1), axis=2).max(1)
    keep = step <= args.max_step
    bad = [str(i) for i in pf["ids"][~keep]]
    pf = {k: v[keep] for k, v in pf.items()}
    rows = [r for r in rows if r["id"] not in set(bad)]
    byid = {r["id"]: r for r in rows}
    ids = [str(i) for i in pf["ids"]]
    print("[fig] %d windows; excluded %d with GT step > %.1f m/frame: %s"
          % (len(ids), len(bad), args.max_step, bad))

    traj_idx = pick_traj(pf)
    if args.qual_window:
        qid = args.qual_window
    else:  # the ~45° turn window from panel b (picked by motion, not by error)
        qid = ids[traj_idx[1]]
    seq, st = qid.rsplit("@", 1)
    fields = recon_window(os.path.join(args.data_root, seq), int(st), args)
    print("[fig] qualitative window", qid, "ATE %.3f m" % float(byid[qid]["ate_trans_rmse_m"]))

    fig = plt.figure(figsize=(7.2, 6.6))
    outer = fig.add_gridspec(3, 1, height_ratios=[1.25, 0.85, 0.95], hspace=0.42,
                             left=0.07, right=0.93, top=0.94, bottom=0.07)
    panel_a(fig, outer[0], fields, args.raymap_moment_scale)
    panel_b(fig, outer[1], pf, rows, traj_idx)
    stats = panel_cd(fig, outer[2], pf, rows)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(args.out_dir, "fig_raymap_vae_fidelity.%s" % ext),
                    dpi=220)
    plt.close(fig)
    latex_table(rows, os.path.join(args.out_dir, "table_raymap_vae_fidelity.tex"))
    print("[fig] disp R2 %.5f (median err %.3f m), yaw R2 %.5f (median err %.3f deg), "
          "ATE/path %.2f%%" % (stats["disp"][0], stats["disp"][1], stats["yaw"][0],
                               stats["yaw"][1], stats["rel_pct"]))
    print("[done]", args.out_dir)


if __name__ == "__main__":
    main()
