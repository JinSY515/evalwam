"""Depth self-consistency: does the predicted depth match the depth an independent
estimator (DA3) reads from the video this model generated?

Four columns, ONE shared pixel mask (geo_decomp's first bug was per-combo masks):

  F  DA3(GT video)        vs GT depth   instrument floor -- DA3's own accuracy here
  A  DA3(generated video) vs GT depth   DA3 error + the generated video diverging from GT
  B  pred depth           vs GT depth   the training objective; self-test vs depth_loss
  C  pred depth           vs DA3(gen)   THE number: video<->depth self-consistency

C's baseline is F, NOT A: A and B both carry "the generated video differs from GT",
C does not, so C < A is near-automatic and means nothing.

Protocol is the MiDaS/DepthAnything one: least-squares scale+shift in DISPARITY over the
mask, invert to depth, then AbsRel / delta1. One affine per 20-frame window, never per
frame -- a per-frame fit is mathematically blind to cross-frame disagreement, which is
the quantity of interest.

AbsRel divides by the reference, so B (ref=GT, a truth) and C (ref=DA3, an estimate) are
not the same quantity. delta1 (max(a/b,b/a)) and logl1 (|log a - log b|) are symmetric.
Cross-column conclusions use delta1/logl1 only.

Validated against two independently recorded numbers: at ckpt-2500 this prints
si_l1 0.0590 (plan records SI-L1 0.05407) and absrel 0.3366 (plan records 0.325).
"""
import argparse, csv, json, os
import numpy as np

FAR = 1000.0


def dec(a, n):
    """(H,W) -> (n,n) nearest. Depth discontinuities must not be averaged across."""
    h = a.shape[0]
    i = ((np.arange(n) + 0.5) * h / n).astype(int)
    return a[np.ix_(i, i)]


def fit_affine(p, tgt, m, scope="window"):
    """Least-squares s,t mapping p -> tgt over m. scope='window' pools all frames."""
    K = p.shape[0]
    P = p.reshape(1, -1) if scope == "window" else p.reshape(K, -1)
    T = tgt.reshape(1, -1) if scope == "window" else tgt.reshape(K, -1)
    M = (m.reshape(1, -1) if scope == "window" else m.reshape(K, -1)).astype(np.float64)
    n = M.sum(1).clip(1.0)
    sp, spp = (P * M).sum(1), (P * P * M).sum(1)
    st, spt = (T * M).sum(1), (P * T * M).sum(1)
    det = np.maximum(spp * n - sp * sp, 1e-9)
    s = (spt * n - sp * st) / det
    t = (st * spp - sp * spt) / det
    if scope == "window":
        return np.full(K, s[0]), np.full(K, t[0])
    return s, t


def compare(src_disp, ref_depth, mask, scope="window"):
    """src_disp: (K,H,W) up-to-affine disparity. ref_depth: (K,H,W) metric. mask: (K,H,W) bool."""
    ref_disp = np.where(mask, 1.0 / np.clip(ref_depth, 1e-3, None), 0.0)
    s, t = fit_affine(src_disp, ref_disp, mask, scope)
    al = s[:, None, None] * src_disp + t[:, None, None]
    # Reduce PER FRAME then mean, exactly as si_depth_loss does (train_hist_cond.py window-
    # affine edit (3)). Pooling all pixels instead silently down-weights sky-heavy frames,
    # whose valid fraction swings 18-43%, and B would stop being comparable to depth_loss.
    _e = (np.abs(al - ref_disp) * mask).reshape(mask.shape[0], -1).sum(1)
    _n = mask.reshape(mask.shape[0], -1).sum(1).clip(1.0)
    si_l1 = float(np.mean(_e / _n))
    lo = 1.0 / FAR
    clamped = al < lo
    d = 1.0 / np.clip(al, lo, None)
    r = ref_depth
    sel = mask & np.isfinite(d) & (r > 1e-3)
    if sel.sum() < 64:
        return dict(absrel=np.nan, delta1=np.nan, logl1=np.nan, si_l1=si_l1,
                    clamp_frac=float((clamped & mask).sum() / max(mask.sum(), 1)))
    dd, rr = d[sel], r[sel]
    ratio = np.maximum(dd / rr, rr / dd)
    return dict(absrel=float(np.mean(np.abs(dd - rr) / rr)),
                delta1=float(np.mean(ratio < 1.25)),
                logl1=float(np.mean(np.abs(np.log(dd) - np.log(rr)))),
                si_l1=si_l1,
                clamp_frac=float((clamped & mask).sum() / max(mask.sum(), 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--meta", default="samples_meta.json")
    ap.add_argument("--da3_subdir", default="viz_da3_fix")
    ap.add_argument("--da3_depth_name", default="_da3_depth.npy")
    ap.add_argument("--da3_gt_depth_name", default="_da3_gt_depth.npy",
                    help="DA3 run on the GT video -> column F, the baseline C is read against.")
    ap.add_argument("--depth_subdir", default="depth")
    ap.add_argument("--work", type=int, default=256)
    ap.add_argument("--vae_stride", type=int, default=4)
    ap.add_argument("--mac_root", default="", help="prefix for pod paths when run off-pod")
    ap.add_argument("--out_csv", default="depth_vs_da3_metrics.csv")
    ap.add_argument("--only", default="", help="comma-separated subset of F,A,B,C")
    a = ap.parse_args()

    D = a.results_dir
    W = a.work
    want = set(x.strip().upper() for x in a.only.split(",") if x.strip()) or {"F", "A", "B", "C"}
    meta = json.load(open(os.path.join(D, a.meta)))
    rows, skipped = [], []

    for sid in sorted(meta, key=lambda x: int(x)):
        v = meta[sid]
        sd = a.mac_root + v["seq_dir"]
        st, nf = int(v["start_frame"]), int(v["num_frames"])
        tag = f"sample_{int(sid):02d}"
        pp = os.path.join(D, f"{tag}_pred_depth.npy")
        if not os.path.exists(pp):
            skipped.append(f"{tag}: no pred_depth"); continue
        pred = np.load(pp).astype(np.float64)                      # (K,h,w) SI-disparity
        K = pred.shape[0]
        if pred.shape[-1] != W:
            pred = np.stack([dec(pred[i], W) for i in range(K)])

        gt = []
        ok = True
        for k in range(K):
            f = st + a.vae_stride * k
            gp = os.path.join(sd, a.depth_subdir, f"{f:06d}.npy")
            if not os.path.exists(gp):
                ok = False; break
            gt.append(dec(np.load(gp).astype(np.float64), W))
        if not ok:
            skipped.append(f"{tag}: missing GT depth"); continue
        gt = np.stack(gt)

        def load_da3(name):
            p = os.path.join(D, a.da3_subdir, f"{tag}{name}")
            if not os.path.exists(p):
                return None
            d = np.load(p).astype(np.float64)
            if d.shape[0] != K:
                return None
            return np.stack([dec(d[i], W) for i in range(K)]) if d.shape[-1] != W else d

        da3 = load_da3(a.da3_depth_name)
        da3g = load_da3(a.da3_gt_depth_name)

        # ONE shared mask for every column present. GT sky is exactly 1000.0 -> g<FAR drops it.
        m = (gt > 0) & (gt < FAR)
        for extra in (da3, da3g):
            if extra is not None:
                m = m & np.isfinite(extra) & (extra > 1e-3)
        if m.sum() < 64:
            skipped.append(f"{tag}: mask collapsed"); continue

        row = {"sample": tag, "n": K, "mask_frac": float(m.mean())}

        def put(pfx, src_disp, ref):
            r = compare(src_disp, ref, m)
            for k2, v2 in r.items():
                row[f"{pfx}_{k2}"] = v2

        if "B" in want:
            put("B", pred, gt)
            # window-minus-frame: the same quantity the training logs as depth_affine_gap
            row["B_si_l1_frame"] = compare(pred, gt, m, scope="frame")["si_l1"]
        if "A" in want and da3 is not None:
            put("A", 1.0 / np.clip(da3, 1e-3, None), gt)
        if "F" in want and da3g is not None:
            put("F", 1.0 / np.clip(da3g, 1e-3, None), gt)
        if "C" in want and da3 is not None:
            # affine is solved pred<->DA3 directly: the sim3 scale already baked into da3 is
            # absorbed, so no GT enters C. Pure self-consistency.
            put("C", pred, da3)
        rows.append(row)

    if not rows:
        raise SystemExit(f"[fatal] no rows. skipped={skipped[:5]}")
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "sample", k))
    out = a.out_csv if os.path.isabs(a.out_csv) else os.path.join(D, a.out_csv)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in rows: w.writerow(r)
    print(f"[done] {len(rows)} rows -> {out}   skipped={len(skipped)}")
    for pfx in ("F", "A", "B", "C"):
        c = [r for r in rows if f"{pfx}_absrel" in r and np.isfinite(r[f"{pfx}_absrel"])]
        if not c: continue
        md = lambda k: float(np.median([r[f"{pfx}_{k}"] for r in c]))
        print(f"  {pfx}: absrel {md('absrel'):.4f}  delta1 {md('delta1'):.4f}  "
              f"logl1 {md('logl1'):.4f}  si_l1 {md('si_l1'):.5f}  n={len(c)}")


if __name__ == "__main__":
    main()
