#!/usr/bin/env python3
"""Rank inference-metric CSV rows best-first, printed to terminal.

--backend picks how camera metrics (colmap_*/mast3r_*) are ranked:
  combine (default): average the two backends' ADE (min-max normalized)
  colmap | mast3r  : rank by that backend's ADE and show that backend's metric family
Advanced overrides: --sort <col> (single column), --combine a,b,c (custom composite).
Bottom rows print mean/median of every shown metric over all rows.
"""
import argparse, csv, math

LOWER = ("ade", "fde", "err", "lpips", "roterr", "transerr", "cammc",
         "dist", "loss", "rotstep", "scale")
HIGHER = ("psnr", "ssim", "success", "iou", "acc")
BACKENDS = ("colmap", "mast3r")
FAMILY = ("ade", "fde", "roterr", "transerr", "cammc", "rotstep_err")


def is_lower_better(col):
    c = col.lower()
    if any(h in c for h in HIGHER):
        return False
    if any(l in c for l in LOWER):
        return True
    return True


def to_f(x):
    try:
        return float(x)
    except Exception:
        return math.nan


def pick_default_sort(cols):
    for p in ("colmap_ade", "ade", "lpips", "psnr", "ssim"):
        for c in cols:
            if c == p or c.endswith("_" + p) or p in c:
                return c
    return cols[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--backend", choices=["combine", "colmap", "mast3r"], default="combine",
                    help="combine=두 백엔드 ADE 평균 / colmap|mast3r=해당 백엔드 계열만")
    ap.add_argument("--sort", default=None, help="single column to rank by (overrides --backend)")
    ap.add_argument("--combine", default=None,
                    help="custom composite: comma cols to min-max normalize + average")
    ap.add_argument("--top", type=int, default=0, help="0 = all")
    ap.add_argument("--cols", default=None, help="comma list of extra columns to show")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv)))
    if not rows:
        raise SystemExit("empty csv")
    cols = list(rows[0].keys())
    numeric = [c for c in cols if c not in ("sample", "dir")
               and any(not math.isnan(to_f(r[c])) for r in rows[:20])]

    ccols = None
    sort_col = None
    show = None

    if args.combine:
        ccols = [c for c in args.combine.split(",") if c]
    elif args.sort:
        sort_col = args.sort
    elif args.backend in ("colmap", "mast3r"):
        b = args.backend
        sort_col = f"{b}_ade" if f"{b}_ade" in cols else pick_default_sort(
            [c for c in numeric if c.startswith(b + "_")] or numeric)
        fam = [f"{b}_{m}" for m in FAMILY if f"{b}_{m}" in cols]
        show = ["sample", sort_col] + [c for c in fam if c != sort_col]
    else:
        primary = pick_default_sort(numeric)
        for b in BACKENDS:
            if primary.startswith(b + "_"):
                rest = primary[len(b) + 1:]
                sib = [f"{x}_{rest}" for x in BACKENDS if f"{x}_{rest}" in cols]
                if len(sib) > 1:
                    ccols = sib
                break
        if ccols is None:
            sort_col = primary

    if ccols is not None:
        bad = [c for c in ccols if c not in cols]
        if bad:
            raise SystemExit(f"no such column(s): {bad}\ncolumns: {cols}")

        def badness(col):
            vals = [to_f(r[col]) for r in rows]
            fin = [v for v in vals if not math.isnan(v)]
            lo, hi = min(fin), max(fin)
            rng = hi - lo
            low_c = is_lower_better(col)
            out = []
            for v in vals:
                if math.isnan(v) or rng == 0:
                    out.append(1.0 if math.isnan(v) else 0.0)
                else:
                    n = (v - lo) / rng
                    out.append(n if low_c else 1 - n)
            return out

        per = [badness(c) for c in ccols]
        for i, r in enumerate(rows):
            r["_combined"] = sum(p[i] for p in per) / len(per)
        rows.sort(key=lambda r: r["_combined"])
        sort_col = "_combined"
        show = ["sample", "_combined"] + ccols
        print(f"# {args.csv}  |  정렬: combine({','.join(ccols)}) 정규화 평균 (낮을수록 좋음)  |  n={len(rows)}")
    else:
        if sort_col not in cols:
            raise SystemExit(f"no such column: {sort_col}\ncolumns: {cols}")
        low = is_lower_better(sort_col)
        rows.sort(key=lambda r: (math.isnan(to_f(r[sort_col])),
                                 to_f(r[sort_col]) if low else -to_f(r[sort_col])))
        if show is None:
            show = ["sample", sort_col]
            if args.cols:
                show += [c for c in args.cols.split(",") if c in cols and c not in show]
            else:
                show += [c for c in numeric if c != sort_col][:6]
        print(f"# {args.csv}  |  정렬: {sort_col} "
              f"({'낮을수록' if low else '높을수록'} 좋음)  |  n={len(rows)}")

    hdr = f"{'#':>4} " + " ".join(f"{c:>11}" for c in show)
    print(hdr); print("-" * len(hdr))
    for i, r in enumerate(rows if args.top <= 0 else rows[:args.top], 1):
        cells = []
        for c in show:
            if c == "sample":
                cells.append(f"{r[c]:>11}")
            else:
                v = to_f(r[c])
                cells.append(f"{v:>11.3f}" if not math.isnan(v) else f"{'nan':>11}")
        print(f"{i:>4} " + " ".join(cells))

    import statistics as st
    print("-" * len(hdr))

    def _col_stat(c, fn):
        v = [to_f(r[c]) for r in rows if not math.isnan(to_f(r[c]))]
        return fn(v) if v else float("nan")

    for label, fn in (("mean", lambda v: sum(v) / len(v)), ("median", st.median)):
        cells = []
        for c in show:
            if c == "sample":
                cells.append(f"{label:>11}")
            else:
                sv = _col_stat(c, fn)
                cells.append(f"{sv:>11.3f}" if not math.isnan(sv) else f"{'nan':>11}")
        print(f"{'':>4} " + " ".join(cells))


if __name__ == "__main__":
    main()
