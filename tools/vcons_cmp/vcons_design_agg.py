#!/usr/bin/env python3
"""Aggregate vcons_design.py -> design/summary.json  (synthetic = valbench 120, real = LookOut A+B+C 120).
Pair statistics are pixel-pooled like the loss itself (sums over windows / sum of mask pixels); 95% CI = bootstrap over windows.
Frame-0 pairs (anchors, e_0 = 0) are kept apart from the gap statistics."""
import glob, json, os
import numpy as np

D = "/root/project/egowam/results_syn/goalprog_mode_real6_nomr_hg0_grad0.5_direct/vcons_cmp/design"
GAPS = list(range(1, 20)); TRAIN_GAPS = (1, 2, 4, 8)
rng = np.random.default_rng(0)
NB = 2000


def load(sel):
    R = [json.load(open(f)) for f in sorted(glob.glob(f"{D}/[vABC][0-9][0-9][0-9].json"))]
    return [r for r in R if sel(r["bench"]) and "drift" in r]


def gap_tables(R):
    """per window, per gap: sums -> (W, G) arrays."""
    keys = ["nF", "nM", "nOc", "floor_M", "cons_M", "incons_M", "cons_F", "incons_F",
            "ours_res_M", "ours_err_M", "abl4_res_M", "abl4_err_M", "npair"]
    A = {k: np.zeros((len(R), len(GAPS))) for k in keys}
    An = {k: np.zeros((len(R), 19)) for k in ("nF", "nM", "floor_M")}
    for w, r in enumerate(R):
        S = r["pairs"]; t = np.array(S["t"]); s = np.array(S["s"]); g = np.abs(t - s)
        for gi, G in enumerate(GAPS):
            k = (g == G) & (s >= 1)
            A["npair"][w, gi] = k.sum()
            for key in keys[:-1]:
                if key in S:
                    A[key][w, gi] = np.array(S[key])[k].sum()
        for j in range(1, 20):
            k = (t == j) & (s == 0)
            for key in An:
                An[key][w, j - 1] = np.array(S[key])[k].sum()
    return A, An


def pooled(A, num, den, idx=None):
    idx = np.arange(A[num].shape[0]) if idx is None else idx
    return A[num][idx].sum(0) / np.maximum(A[den][idx].sum(0), 1e-9)


def boot(fn, W):
    b = np.array([fn(rng.integers(0, W, W)) for _ in range(NB)])
    return np.percentile(b, 2.5, 0), np.percentile(b, 97.5, 0)


def summarize(R):
    W = len(R); A, An = gap_tables(R)
    M = lambda k, i=None: pooled(A, k, "nM", i)
    stats = {
        "inframe": lambda i=None: A["nF"][i if i is not None else slice(None)].sum(0) / np.maximum(A["npair"][i if i is not None else slice(None)].sum(0) * 4096, 1),
        "frac_mask": lambda i=None: pooled(A, "nM", "nF", i), "frac_occ": lambda i=None: pooled(A, "nOc", "nF", i),
        "floor_M": lambda i=None: M("floor_M", i), "cons_M": lambda i=None: M("cons_M", i), "incons_M": lambda i=None: M("incons_M", i),
        "sep_M": lambda i=None: M("incons_M", i) / M("cons_M", i),
        "sep_F": lambda i=None: pooled(A, "incons_F", "nF", i) / pooled(A, "cons_F", "nF", i),
        "ours_ratio": lambda i=None: M("ours_res_M", i) / M("ours_err_M", i),
        "abl4_ratio": lambda i=None: M("abl4_res_M", i) / M("abl4_err_M", i),
        "ratio_diff": lambda i=None: M("ours_res_M", i) / M("ours_err_M", i) - M("abl4_res_M", i) / M("abl4_err_M", i),
        "ours_res": lambda i=None: M("ours_res_M", i), "abl4_res": lambda i=None: M("abl4_res_M", i),
        "ours_err": lambda i=None: M("ours_err_M", i), "abl4_err": lambda i=None: M("abl4_err_M", i)}
    out = {"n": W, "gaps": GAPS, "gap": {}}
    for k, fn in stats.items():
        lo, hi = boot(fn, W)
        out["gap"][k] = {"v": fn().tolist(), "lo": lo.tolist(), "hi": hi.tolist()}
    # effective share of one training step's masked pixels per gap: 8 random pairs drawn uniformly from the
    # gap-1,2,4,8 candidates (both directions) + 19 anchors (j <- 0); the loss is normalised by all masked pixels
    nMp = A["nM"].sum(0) / np.maximum(A["npair"].sum(0), 1)                     # mean mask px per pair at each gap
    cand = {g: 2 * (20 - g) for g in TRAIN_GAPS}; tot = sum(cand.values())
    px = {f"g{g}": 8 * cand[g] / tot * nMp[g - 1] for g in TRAIN_GAPS}
    px["anchor"] = float(An["nM"].sum() / W)
    s = sum(px.values()); out["train_share"] = {k: float(v / s) for k, v in px.items()}
    out["anchor"] = {"frac_mask": (An["nM"].sum(0) / np.maximum(An["nF"].sum(0), 1)).tolist(),
                     "floor_M": (An["floor_M"].sum(0) / np.maximum(An["nM"].sum(0), 1)).tolist()}
    # gradient alignment (cos with -(x - z)) per model / variant: mean over windows + CI, and paired differences
    out["cos0"] = {}
    for mdl in ("ours", "abl4"):
        C = {k: np.array([r["cos0"][mdl][k] for r in R if mdl in r["cos0"]]) for k in R[0]["cos0"]["ours"]}
        o = {}
        for k, v in C.items():
            lo, hi = boot(lambda i: v[i].mean(), len(v)); o[k] = {"v": float(v.mean()), "lo": float(lo), "hi": float(hi)}
        for a_, b_ in (("pairs_sg_M", "pairs_nosg_M"), ("pairs_sg_M", "pairs_sg_F"), ("anchor_M", "anchor_F")):
            d = C[a_] - C[b_]; lo, hi = boot(lambda i: d[i].mean(), len(d))
            o[f"{a_}-{b_}"] = {"v": float(d.mean()), "lo": float(lo), "hi": float(hi)}
        out["cos0"][mdl] = o
    ev = ["err_all", "err_vis0", "err_novis0"] + [f"res_g{g}" for g in (1, 2, 4, 8, 16)]
    out["drift"] = {"start": {k: float(np.nanmean([r["start"][k] if r["start"][k] is not None else np.nan for r in R])) for k in ev}}
    for v in R[0]["drift"]:
        out["drift"][v] = {k: float(np.nanmean([r["drift"][v][k] if r["drift"][v][k] is not None else np.nan for r in R])) for k in ev}
    return out


S = {"synthetic": summarize(load(lambda b: b == "valbench")), "real": summarize(load(lambda b: b != "valbench"))}
json.dump(S, open(f"{D}/summary.json", "w"), indent=1)
for n, s in S.items():
    g = s["gap"]; print(n, s["n"], "train_share", {k: round(v, 3) for k, v in s["train_share"].items()})
    for gi in (0, 1, 3, 7, 11, 15):
        print(f"  gap {gi + 1:2d}", " ".join(f"{k} {g[k]['v'][gi]:.3f}" for k in ("inframe", "frac_mask", "frac_occ", "floor_M", "cons_M", "sep_M", "sep_F", "ours_ratio", "abl4_ratio", "ratio_diff")))
    print("  cos0", {k: round(v["v"], 3) for k, v in s["cos0"]["ours"].items()})
    print("  cos0 abl4", {k: round(v["v"], 3) for k, v in s["cos0"]["abl4"].items()})
    for k, v in s["drift"].items():
        print("  drift", k, {a: round(b, 3) for a, b in v.items()})
