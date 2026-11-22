#!/usr/bin/env python3
"""Figures for /vcons/why.html (called by build_why.py). Matplotlib, light surface; residual = series-1 blue (solid, circle),
direct = series-2 orange (dashed, square) -- identity never by colour alone; English text (no CJK font on this server)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES, DIR, INK, MUT, GRID, SURF = "#2a78d6", "#eb6834", "#1d1d1f", "#6b6b70", "#e3e3e8", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID, "axes.labelcolor": INK,
                     "xtick.color": MUT, "ytick.color": MUT, "text.color": INK, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
                     "axes.titlesize": 11, "lines.linewidth": 2})
VORD = [("GT", "GT"), ("tavg5", "t-avg\n5"), ("tavg9", "t-avg\n9"), ("blur2", "blur\n2"), ("blur4", "blur\n4"), ("still", "frozen")]


def iqr(a):
    return np.percentile(a, 25), np.percentile(a, 75)


def landscape(SETS, out):
    """loss / (direct loss of GT) for GT and degraded GT. Below the 1-line = direct prefers it to the truth."""
    fig, axs = plt.subplots(1, len(SETS), figsize=(3.9 * len(SETS), 4.3), sharey=True)
    for ax, (title, R) in zip(axs, SETS):
        x = np.arange(len(VORD))
        for mode, col, ls, mk, lab in (("direct", DIR, "--", "s", "direct"), ("residual", RES, "-", "o", "residual")):
            med, lo, hi = [], [], []
            for k, _ in VORD:
                v = np.array([r["rank"][k][mode] / r["rank"]["GT"]["direct"] for r in R])
                med.append(np.median(v)); a, b = iqr(v); lo.append(a); hi.append(b)
            med = np.array(med)
            ax.fill_between(x, lo, hi, color=col, alpha=0.12, linewidth=0)
            ax.plot(x, med, ls=ls, color=col, marker=mk, markersize=7, markeredgecolor=SURF, markeredgewidth=1.5)
            ax.annotate(lab, (x[-1], med[-1]), xytext=(6, 0), textcoords="offset points", va="center", color=INK, fontsize=9)
        ax.axhline(1, color=MUT, lw=1, ls=":")
        bel = [i for i, (k, _) in enumerate(VORD) if k != "GT" and np.median([r["rank"][k]["direct"] / r["rank"]["GT"]["direct"] for r in R]) < 1]
        for i in bel:
            ax.axvspan(i - 0.35, i + 0.35, color=DIR, alpha=0.08, linewidth=0)
        ax.set_xticks(x); ax.set_xticklabels([l for _, l in VORD], fontsize=8)
        ax.set_title(f"{title} (n={len(R)})"); ax.set_xlim(-0.4, len(VORD) - 0.2)
    axs[0].set_ylabel("loss / direct loss of GT")
    axs[0].legend(handles=[plt.Line2D([], [], color=RES, ls="-", marker="o", label="residual"),
                           plt.Line2D([], [], color=DIR, ls="--", marker="s", label="direct")], loc="upper left", frameon=False, fontsize=9)
    fig.supxlabel("t-avg = temporal mean over 5 / 9 frames (ghosting) · blur = spatial Gaussian sigma px · frozen = every frame = frame 0 · "
                  "dotted = GT's direct score · shaded = direct scores it BELOW the truth", fontsize=8, color=MUT)
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)


def per_sample(SETS, out):
    """per sample: loss(temporal avg 5) - loss(GT). < 0 = the loss prefers the ghosted copy."""
    fig, ax = plt.subplots(figsize=(10, 3.8))
    rng = np.random.default_rng(0)
    for j, (title, R) in enumerate(SETS):
        for mode, col, mk, off in (("direct", DIR, "s", -0.17), ("residual", RES, "o", 0.17)):
            d = np.array([r["rank"]["tavg5"][mode] - r["rank"]["GT"][mode] for r in R])
            ax.scatter(j + off + rng.uniform(-0.08, 0.08, len(d)), d, s=14, color=col, marker=mk, alpha=0.75, linewidths=0)
            frac = np.mean(d < 0)
            ax.text(j + off, ax.get_ylim()[0] if False else -0.12, f"{frac:.3f}", ha="center", fontsize=8, color=INK)
    ax.axhline(0, color=MUT, lw=1)
    ax.set_xticks(range(len(SETS))); ax.set_xticklabels([t for t, _ in SETS])
    ax.set_ylabel("loss(temporal avg 5) - loss(GT)")
    ax.scatter([], [], color=DIR, marker="s", label="direct"); ax.scatter([], [], color=RES, marker="o", label="residual")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.set_title("each dot = one window; below 0 = loss prefers the ghosted copy to the truth (numbers = fraction below 0)", fontsize=10)
    ax.set_ylim(-0.14, max(0.45, ax.get_ylim()[1]))
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)


def drift(SETS, out):
    """minimise each loss alone starting AT GT: top = the loss, bottom = latent distance from GT (two charts, never dual axis)."""
    fig, axs = plt.subplots(2, len(SETS), figsize=(4.2 * len(SETS), 6.2), sharex=True)
    for j, (title, R) in enumerate(SETS):
        for mode, col, ls, lab in (("direct", DIR, "--", "direct"), ("residual", RES, "-", "residual")):
            T = np.array([[t[0] for t in r["drift"][mode]["trace"]] for r in R][0])
            L = np.array([[t[1] for t in r["drift"][mode]["trace"]] for r in R])
            Dd = np.array([[t[2] for t in r["drift"][mode]["trace"]] for r in R])
            for row, Y in ((0, L), (1, Dd)):
                ax = axs[row, j]; m = np.median(Y, 0); a, b = np.percentile(Y, 25, 0), np.percentile(Y, 75, 0)
                ax.fill_between(T, a, b, color=col, alpha=0.15, linewidth=0); ax.plot(T, m, ls=ls, color=col, label=lab)
        axs[0, j].set_title(f"{title}")
        axs[1, j].set_xlabel("optimisation step")
    axs[0, 0].set_ylabel("loss being minimised"); axs[1, 0].set_ylabel("latent distance from GT")
    axs[0, 0].legend(loc="upper right", frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)


def psnr_bars(SETS, out):
    """decoded PSNR vs GT after the drift; residual end point = GT through the VAE (ceiling)."""
    fig, ax = plt.subplots(figsize=(8, 3.6))
    x = np.arange(len(SETS)); w = 0.36
    for k, (mode, col, hatch, lab) in enumerate((("residual", RES, "", "minimise residual (stays at GT)"), ("direct", DIR, "//", "minimise direct"))):
        v = [np.mean([r["drift"][mode]["psnr_vs_gt"] for r in R]) for _, R in SETS]
        bars = ax.bar(x + (k - 0.5) * (w + 0.02), v, w, color=col, hatch=hatch, edgecolor=SURF, linewidth=2, label=lab)
        for b_, val in zip(bars, v):
            ax.text(b_.get_x() + b_.get_width() / 2, val + 0.4, f"{val:.3f}", ha="center", fontsize=8, color=INK)
    ax.set_xticks(x); ax.set_xticklabels([t for t, _ in SETS]); ax.set_ylabel("decoded PSNR vs GT (dB)")
    ax.legend(frameon=False, fontsize=9, loc="upper right"); ax.set_ylim(0, 40); ax.grid(axis="x", visible=False)
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)


def quality_bars(SETS, out):
    """decoded quality vs GT after the drift: PSNR (77 frames), SSIM / LPIPS (frames 20/40/60/76, why_quality.py).
    residual end point = GT through the VAE (ceiling)."""
    M = [("PSNR (dB) - higher better", lambda r, m: r["drift"][m]["psnr_vs_gt"], 40),
         ("SSIM - higher better", lambda r, m: r["q"][m]["ssim"], 1.0),
         ("LPIPS - lower better", lambda r, m: r["q"][m]["lpips"], 0.8)]
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
    x = np.arange(len(SETS)); w = 0.36
    for ax, (title, get, top) in zip(axs, M):
        for k, (mode, col, hatch, lab) in enumerate((("residual", RES, "", "minimise residual (stays at GT)"), ("direct", DIR, "//", "minimise direct"))):
            v = [np.mean([get(r, mode) for r in R if "q" in r or title.startswith("PSNR")]) for _, R in SETS]
            bars = ax.bar(x + (k - 0.5) * (w + 0.02), v, w, color=col, hatch=hatch, edgecolor=SURF, linewidth=2, label=lab)
            for b_, val in zip(bars, v):
                ax.text(b_.get_x() + b_.get_width() / 2, val + top * 0.01, f"{val:.3f}", ha="center", va="bottom", fontsize=8, color=INK)
        ax.set_xticks(x); ax.set_xticklabels([t_ for t_, _ in SETS]); ax.set_title(title); ax.set_ylim(0, top * 1.08)
        ax.grid(axis="x", visible=False)
    h_, l_ = axs[0].get_legend_handles_labels(); fig.legend(h_, l_, frameon=False, fontsize=8, loc="lower right", ncol=2)
    fig.supxlabel("PSNR on all 77 frames; SSIM / LPIPS(alex) on frames 20, 40, 60, 76 (frame 0 is held at GT)", fontsize=8, color=MUT)
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)


def grad_split(SETS, out):
    """where the direct gradient points on real generated videos (why_grad.py): share of masked latent elements where
    the model-error term dominates (direct = residual direction by construction), the GT floor term dominates and the
    direct gradient happens to agree, and the GT term dominates and it points AGAINST the error-correcting direction."""
    fig, ax = plt.subplots(figsize=(10, 3.4))
    LIGHT = "#9cc3ee"
    for j, (title, R) in enumerate(SETS):
        rows = [r[k] for r in R for k in ("res", "dir")]
        nod = np.mean([1 - r["dom"] for r in rows]); da = np.mean([r["dom"] * r["agree_dom"] for r in rows])
        do = np.mean([r["dom"] * (1 - r["agree_dom"]) for r in rows])
        left = 0
        for v, col, hatch, lab in ((nod, RES, "", "model error dominates -> direct = residual direction"),
                                   (da, LIGHT, "", "GT term dominates, same direction"),
                                   (do, DIR, "//", "GT term dominates, OPPOSITE to the error fix")):
            ax.barh(j, v, left=left, color=col, hatch=hatch, edgecolor=SURF, linewidth=2, label=lab if j == 0 else None)
            ax.text(left + v / 2, j, f"{v:.3f}", ha="center", va="center", fontsize=9, color=INK,
                    bbox=dict(boxstyle="round,pad=0.2", fc=SURF, ec="none", alpha=0.9))
            left += v
    ax.set_yticks(range(len(SETS))); ax.set_yticklabels([f"{t} ({len(R)} x 2 models)" for t, R in SETS]); ax.invert_yaxis()
    ax.set_xlim(0, 1); ax.set_xlabel("share of masked latent elements (all training pairs)"); ax.grid(False)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2)
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)
