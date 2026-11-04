#!/usr/bin/env python3
"""Average metric CSV(s) over samples and print to terminal, plus per-sample
ranking (--sort), two-CSV delta (--diff), worst-in-both (--both), composite (--combine).
--only adds each named metric's A/B value next to every listed sample in --diff/--both.
"""
import argparse
import csv
import glob
import math
import os

SKIP = {"sample", "error", "dir", "n"}


def _is_skip(col):
    return col in SKIP or col.endswith("_n") or col.endswith("_success") or col.endswith("_err")


def col_means(path, only=None):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}, 0
    out = {}
    for k in rows[0].keys():
        if _is_skip(k):
            continue
        if only and k not in only and k.split("_", 1)[-1] not in only:
            continue
        vals = []
        for r in rows:
            try:
                v = float(r.get(k, ""))
            except (TypeError, ValueError):
                continue
            if not math.isnan(v):
                vals.append(v)
        if vals:
            out[k] = (sum(vals) / len(vals), len(vals))
    return out, len(rows)


def resolve(inputs):
    paths = []
    for p in inputs:
        if os.path.isdir(p):
            paths += sorted(glob.glob(os.path.join(p, "*_metrics.csv")))
        elif any(c in p for c in "*?["):
            paths += sorted(glob.glob(p))
        else:
            paths.append(p)
    seen, uniq = set(), []
    for p in paths:
        if p not in seen and os.path.exists(p):
            seen.add(p)
            uniq.append(p)
    return uniq


def resolve_col(cols, name):
    if name in cols:
        return name
    for c in cols:
        if c.split("_", 1)[-1] == name or c.endswith("_" + name):
            return c
    return None


def _trim_keep(pairs, frac):
    if frac and frac > 0 and len(pairs) > 2:
        k = int(frac * len(pairs))
        if k > 0:
            ordered = sorted(pairs, key=lambda x: x[1])
            return set(s for s, _ in ordered[k:len(ordered) - k])
    return set(s for s, _ in pairs)


def _col_mean(fbys, samples, col):
    xs = [fbys[s][col] for s in samples if s in fbys and col in fbys[s]]
    return (sum(xs) / len(xs)) if xs else None


def sort_view(path, metric, asc, top, extra, trim=0.0):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print(f"  (empty) {path}"); return
    col = resolve_col(rows[0].keys(), metric)
    if not col:
        print(f"  [skip] {tag(path)}: no column matching '{metric}'"); return
    ex_cols = [resolve_col(rows[0].keys(), e) for e in extra]
    ex_cols = [c for c in ex_cols if c and c != col]
    scored = []
    for r in rows:
        try:
            v = float(r.get(col, ""))
        except (TypeError, ValueError):
            continue
        if math.isnan(v):
            continue
        scored.append((r.get("sample", "?"), v, r))
    scored.sort(key=lambda x: x[1], reverse=not asc)
    shown = scored[:top] if top else scored
    order = "asc" if asc else "desc"
    print("=" * 60)
    print(f"  {tag(path)}  -  sort by {col} ({order}), {len(shown)}/{len(scored)} samples")
    print("=" * 60)
    ws = max([len("sample")] + [len(s) for s, _, _ in shown]) if shown else 6
    print(f"  {'#':>3}  {'sample':<{ws}}  {col:>12}" + "".join(f"  {c:>12}" for c in ex_cols))
    for i, (s, v, r) in enumerate(shown, 1):
        line = f"  {i:>3}  {s:<{ws}}  {v:>12.4f}"
        for c in ex_cols:
            try:
                line += f"  {float(r[c]):>12.4f}"
            except (TypeError, ValueError):
                line += f"  {'-':>12}"
        print(line)
    mean = sum(v for _, v, _ in scored) / len(scored) if scored else float("nan")
    print(f"  {'':>3}  {'[mean]':<{ws}}  {mean:>12.4f}")
    fbys, _flds = _full_by_sample(path)
    all_s = [s for s, _, _ in scored]
    keep = _trim_keep([(s, v) for s, v, _ in scored], trim)
    trimmed = trim and trim > 0 and 0 < len(keep) < len(scored)
    if trimmed:
        tv = [v for s, v, _ in scored if s in keep]
        print(f"  {'':>3}  {f'[trim {trim:g}]':<{ws}}  {sum(tv)/len(tv):>12.4f}   (n={len(tv)})")
    for c in ex_cols:
        ma = _col_mean(fbys, all_s, c)
        if ma is None: continue
        extra_s = ""
        if trimmed:
            mk = _col_mean(fbys, [s for s in all_s if s in keep], c)
            extra_s = f"   trim={mk:.4f}" if mk is not None else ""
        print(f"  {'':>3}  {('mean ' + c.split('_', 1)[-1]):<{ws}}  {ma:>12.4f}{extra_s}")
    print()


def _metric_by_sample(path, metric):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}, None
    col = resolve_col(rows[0].keys(), metric)
    if not col:
        return {}, None
    out = {}
    for r in rows:
        try:
            v = float(r.get(col, ""))
        except (TypeError, ValueError):
            continue
        if not math.isnan(v):
            out[r.get("sample", "?")] = v
    return out, col


def _full_by_sample(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}, []
    out = {}
    for r in rows:
        d = {}
        for k, v in r.items():
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if not math.isnan(fv):
                d[k] = fv
        out[r.get("sample", "?")] = d
    return out, list(rows[0].keys())


def _extra_cols(fields, extra, drop):
    cols = []
    for e in extra:
        c = resolve_col(fields, e)
        if c and c not in drop and c not in cols:
            cols.append(c)
    return cols


def _extra_header(ex_cols):
    return "".join(f"  {c.split('_', 1)[-1][:9]:>9}.A  {c.split('_', 1)[-1][:9]:>9}.B" for c in ex_cols)


def _extra_cells(fa, fb, ex_cols):
    out = ""
    for c in ex_cols:
        va, vb = fa.get(c), fb.get(c)
        out += f"  {(f'{va:.4f}' if va is not None else '-'):>11}  {(f'{vb:.4f}' if vb is not None else '-'):>11}"
    return out


def diff_view(pathA, pathB, metric, asc, top, by_abs, extra, trim=0.0):
    mA, cA = _metric_by_sample(pathA, metric)
    mB, cB = _metric_by_sample(pathB, metric)
    if not cA or not cB:
        miss = pathA if not cA else pathB
        print(f"  [skip] no column matching '{metric}' in {tag(miss)}"); return
    common = sorted(set(mA) & set(mB))
    if not common:
        print("  [skip] no shared samples between the two CSVs"); return
    fA, fldA = _full_by_sample(pathA)
    fB, _ = _full_by_sample(pathB)
    ex_cols = _extra_cols(fldA, extra, {cA})
    diffs = [(s, mA[s], mB[s], mB[s] - mA[s]) for s in common]
    key = (lambda x: abs(x[3])) if by_abs else (lambda x: x[3])
    diffs.sort(key=key, reverse=not asc)
    shown = diffs[:top] if top else diffs
    order = ("|d|" if by_abs else "d") + (" asc" if asc else " desc")
    print("=" * 72)
    print(f"  DIFF {cB} :  B - A   (sort {order}, {len(shown)}/{len(common)} shared samples)")
    print(f"    A = {tag(pathA)}")
    print(f"    B = {tag(pathB)}")
    only_a, only_b = sorted(set(mA) - set(mB)), sorted(set(mB) - set(mA))
    if only_a or only_b:
        print(f"    (unmatched: A-only={len(only_a)}, B-only={len(only_b)})")
    print("=" * 72)
    ws = max([len("sample")] + [len(s) for s, *_ in shown])
    print(f"  {'#':>3}  {'sample':<{ws}}  {'A':>12}  {'B':>12}  {'delta(B-A)':>12}" + _extra_header(ex_cols))
    for i, (s, a, b, d) in enumerate(shown, 1):
        print(f"  {i:>3}  {s:<{ws}}  {a:>12.4f}  {b:>12.4f}  {d:>+12.4f}" + _extra_cells(fA.get(s, {}), fB.get(s, {}), ex_cols))
    ma = sum(x[1] for x in diffs) / len(diffs)
    mb = sum(x[2] for x in diffs) / len(diffs)
    print(f"  {'':>3}  {'[mean]':<{ws}}  {ma:>12.4f}  {mb:>12.4f}  {mb - ma:>+12.4f}")
    keep = None
    if trim and trim > 0:
        keepA = _trim_keep([(s, mA[s]) for s in common], trim)
        keepB = _trim_keep([(s, mB[s]) for s in common], trim)
        _k = [s for s in common if s in keepA and s in keepB]
        if 0 < len(_k) < len(common):
            keep = _k
            ta = sum(mA[s] for s in keep) / len(keep); tb = sum(mB[s] for s in keep) / len(keep)
            print(f"  {'':>3}  {f'[trim {trim:g}]':<{ws}}  {ta:>12.4f}  {tb:>12.4f}  {tb - ta:>+12.4f}   (n={len(keep)})")
    for c in ex_cols:
        aM, bM = _col_mean(fA, common, c), _col_mean(fB, common, c)
        if aM is None and bM is None: continue
        line = (f"  {'':>3}  {('mean ' + c.split('_', 1)[-1]):<{ws}}  "
                f"{(f'{aM:.4f}' if aM is not None else '-'):>12}  {(f'{bM:.4f}' if bM is not None else '-'):>12}")
        if keep:
            aT, bT = _col_mean(fA, keep, c), _col_mean(fB, keep, c)
            line += f"   trim: A={(f'{aT:.4f}' if aT is not None else '-')} B={(f'{bT:.4f}' if bT is not None else '-')}"
        print(line)
    print()


def both_view(pathA, pathB, metric, top, asc, extra):
    mA, cA = _metric_by_sample(pathA, metric)
    mB, cB = _metric_by_sample(pathB, metric)
    if not cA or not cB:
        miss = pathA if not cA else pathB
        print(f"  [skip] no column matching '{metric}' in {tag(miss)}"); return
    common = sorted(set(mA) & set(mB))
    if not common:
        print("  [skip] no shared samples between the two CSVs"); return
    fA, fldA = _full_by_sample(pathA)
    fB, _ = _full_by_sample(pathB)
    ex_cols = _extra_cols(fldA, extra, {cA})
    K = top or max(1, len(common) // 10)
    ordA = sorted(common, key=lambda s: mA[s], reverse=not asc)
    ordB = sorted(common, key=lambda s: mB[s], reverse=not asc)
    rankA = {s: i for i, s in enumerate(ordA, 1)}
    rankB = {s: i for i, s in enumerate(ordB, 1)}
    setA, setB = set(ordA[:K]), set(ordB[:K])
    inter = sorted(setA & setB, key=lambda s: rankA[s] + rankB[s])
    worst = "smallest" if asc else "largest"
    print("=" * 72)
    print(f"  WORST-IN-BOTH  {cA}   (worst = {worst} value, worst-{K} of {len(common)} shared)")
    print(f"    A = {tag(pathA)}")
    print(f"    B = {tag(pathB)}")
    print(f"    overlap: {len(inter)} / {K}   (expected by chance ~ {K * K / len(common):.1f})")
    print("=" * 72)
    if not inter:
        print(f"  (no sample in both worst-{K}; try larger --top)"); print(); return
    ws = max([len("sample")] + [len(s) for s in inter])
    print(f"  {'#':>3}  {'sample':<{ws}}  {'A_val':>10}  {'A_rank':>7}  {'B_val':>10}  {'B_rank':>7}" + _extra_header(ex_cols))
    for i, s in enumerate(inter, 1):
        print(f"  {i:>3}  {s:<{ws}}  {mA[s]:>10.4f}  {rankA[s]:>7}  {mB[s]:>10.4f}  {rankB[s]:>7}" + _extra_cells(fA.get(s, {}), fB.get(s, {}), ex_cols))
    print()


def combine_view(path, metrics, higher, asc, top):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print(f"  (empty) {path}"); return
    cols = {m: resolve_col(rows[0].keys(), m) for m in metrics}
    missing = [m for m, c in cols.items() if not c]
    if missing:
        print(f"  [skip] {tag(path)}: no column for {missing}"); return
    hi = set()
    for h in higher:
        c = resolve_col(rows[0].keys(), h)
        if c:
            hi.add(c)
    recs = []
    for r in rows:
        vals = {}
        ok = True
        for m, c in cols.items():
            try:
                v = float(r.get(c, ""))
            except (TypeError, ValueError):
                ok = False; break
            if math.isnan(v):
                ok = False; break
            vals[c] = v
        if ok:
            recs.append((r.get("sample", "?"), vals))
    if not recs:
        print(f"  [skip] {tag(path)}: no sample has all of {list(cols.values())}"); return
    stats = {}
    for c in cols.values():
        xs = [v[c] for _, v in recs]
        mu = sum(xs) / len(xs)
        var = sum((x - mu) ** 2 for x in xs) / len(xs)
        stats[c] = (mu, math.sqrt(var))
    scored = []
    for s, vals in recs:
        z = 0.0
        for c, (mu, sd) in stats.items():
            zc = (vals[c] - mu) / sd if sd > 0 else 0.0
            z += -zc if c in hi else zc
        scored.append((s, z, vals))
    scored.sort(key=lambda x: x[1], reverse=not asc)
    shown = scored[:top] if top else scored
    order = "asc" if asc else "desc"
    comp_cols = list(cols.values())
    print("=" * 72)
    print(f"  {tag(path)}  -  COMBINE z-score of {comp_cols}")
    print(f"    higher-better (negated): {sorted(hi) or '-'}   sort composite {order}, {len(shown)}/{len(scored)}")
    print("=" * 72)
    ws = max([len("sample")] + [len(s) for s, _, _ in shown])
    print(f"  {'#':>3}  {'sample':<{ws}}  {'composite':>10}" + "".join(f"  {c.split('_',1)[-1][:10]:>11}" for c in comp_cols))
    for i, (s, z, vals) in enumerate(shown, 1):
        print(f"  {i:>3}  {s:<{ws}}  {z:>+10.3f}" + "".join(f"  {vals[c]:>11.4f}" for c in comp_cols))
    print()
    print("  note: composite = sum of per-metric z-scores (equal weight, scale-free).")
    print()


def tag(path):
    stem = os.path.basename(path).replace("_metrics.csv", "").replace(".csv", "")
    parent = os.path.basename(os.path.dirname(path))
    return f"{parent}/{stem}" if parent else stem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="CSV file(s), dir(s), or glob(s)")
    ap.add_argument("--only", default="", help="comma metrics: subset (mean) / extra A/B cols (diff/both)")
    ap.add_argument("--sort", default="", help="metric to rank samples by")
    ap.add_argument("--asc", action="store_true", help="ascending (default descending)")
    ap.add_argument("--top", type=int, default=0, help="show only top-N rows (0 = all)")
    ap.add_argument("--trim", type=float, default=0.0,
                    help="drop top+bottom FRAC of samples by the sort metric before averaging (e.g. 0.05). 0=off.")
    ap.add_argument("--diff", action="store_true", help="two CSVs: rank shared samples by delta (B-A)")
    ap.add_argument("--abs", action="store_true", help="--diff: rank by |delta|")
    ap.add_argument("--both", action="store_true", help="two CSVs: samples in worst-K of BOTH")
    ap.add_argument("--combine", default="", help="comma metrics -> composite z-score rank")
    ap.add_argument("--higher_better", default="", help="--combine: better-when-higher metrics (z negated)")
    args = ap.parse_args()
    only = set(x.strip() for x in args.only.split(",") if x.strip()) or None
    extra = [x.strip() for x in args.only.split(",") if x.strip()]

    paths = resolve(args.inputs)
    if not paths:
        print("[mean_metrics] no CSV found for:", args.inputs)
        return

    if args.combine:
        metrics = [x.strip() for x in args.combine.split(",") if x.strip()]
        higher = [x.strip() for x in args.higher_better.split(",") if x.strip()]
        for p in paths:
            combine_view(p, metrics, higher, args.asc, args.top)
        return

    if args.both:
        if not args.sort:
            print("[mean_metrics] --both requires --sort <metric>"); return
        if len(paths) != 2:
            print(f"[mean_metrics] --both needs exactly 2 CSVs (got {len(paths)}): {paths}"); return
        both_view(paths[0], paths[1], args.sort, args.top, args.asc, extra)
        return

    if args.diff:
        if not args.sort:
            print("[mean_metrics] --diff requires --sort <metric>"); return
        if len(paths) != 2:
            print(f"[mean_metrics] --diff needs exactly 2 CSVs (got {len(paths)}): {paths}"); return
        diff_view(paths[0], paths[1], args.sort, args.asc, args.top, args.abs, extra, args.trim)
        return

    if args.sort:
        for p in paths:
            sort_view(p, args.sort, args.asc, args.top, extra, args.trim)
        return

    per = []
    for p in paths:
        means, nrow = col_means(p, only)
        per.append((p, means, nrow))
        print("=" * 60)
        print(f"  {tag(p)}   (mean over {nrow} samples)")
        print("=" * 60)
        if not means:
            print("  (no numeric metric columns)")
        else:
            wk = max(len(k) for k in means)
            for k, (v, nv) in means.items():
                note = f"  (n={nv})" if nv != nrow else ""
                print(f"  {k:<{wk}} {v:>12.4f}{note}")
        print()

    if len(per) >= 2:
        shared = set.intersection(*[set(m) for _, m, _ in per]) if per else set()
        if shared:
            cols = sorted(shared)
            wt = max(len(tag(p)) for p, _, _ in per)
            print("#" * 60)
            print("  COMPARISON (shared metrics)")
            print("#" * 60)
            print("  " + " " * wt + "  " + "  ".join(f"{c[:10]:>10}" for c in cols))
            for p, m, _ in per:
                line = "  ".join(f"{m[c][0]:>10.4f}" for c in cols)
                print(f"  {tag(p):<{wt}}  {line}")


if __name__ == "__main__":
    main()
