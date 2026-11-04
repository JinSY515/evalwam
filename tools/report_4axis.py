"""Print the four evaluation axes for one or many checkpoints as aligned terminal tables.

Reads only CSVs that eval_4axis.sh already wrote -- it computes nothing and writes
nothing, so it is safe to run at any time, including while a run is still going.

One table per axis, one row per checkpoint, so the comparison that matters
(fine vs geo across steps) reads straight down a column.

Output is ASCII only: the pod terminal renders Hangul as underscores.

  python3 tools/report_4axis.py                     # discover every ckpt under RUNS
  python3 tools/report_4axis.py --dirs A B          # explicit pred_depth dirs
  python3 tools/report_4axis.py --csv               # machine-readable, for a spreadsheet
"""
import argparse, csv, glob, os, statistics as st, sys

RS_DEFAULT = "/root/project/ego_wam/results_syn"
RUNS_DEFAULT = "goalprog_appearance_depth_fine,goalprog_appearance_depth_fine_geo"


def _label(run):
    """Row label. MUST be unique across --runs or two rows read identically and the
    comparison is unreadable -- that is exactly what the old
    `"geo" if endswith("_geo") else "fine"` did to every ablation (…_d10 -> "fine").
    Keeps 'fine'/'geo' byte-identical for the two default runs so old tables still match."""
    base = "goalprog_appearance_depth_fine"
    if run == base:
        return "fine"
    if run.startswith(base + "_"):
        return run[len(base) + 1:]      # _geo -> geo, _d10 -> d10, _geo_joint -> geo_joint
    return run


def rd(d, name):
    p = os.path.join(d, name)
    if not os.path.exists(p):
        return []
    with open(p) as f:
        return list(csv.DictReader(f))


def md(rows, key):
    v = []
    for r in rows:
        try:
            x = float(r.get(key, ""))
            if x == x:
                v.append(x)
        except (TypeError, ValueError):
            pass
    return st.median(v) if v else float("nan")


def lt_frac(rows, a, b):
    n = c = 0
    for r in rows:
        try:
            x, y = float(r[a]), float(r[b])
        except (KeyError, TypeError, ValueError):
            continue
        n += 1
        c += x < y
    return f"{c}/{n}" if n else ""


def num(x, p=4):
    """Numbers become strings here, not at print time, so the table can size its columns."""
    return "" if x != x else f"{x:.{p}f}"


# Direction markers. ASCII on purpose: the pod terminal has no Hangul glyphs, so Unicode
# arrows are not worth the risk either. "" = context column with no good/bad direction.
HI, LO, ONE = "^", "v", "~1"
# Colour when stdout is a terminal, plain markers otherwise: ANSI escapes survive neither
# a pipe into a file nor a paste into a document, and the ranking is information, so it has
# to survive in both forms.
BEST, SECOND = "*", "+"
C_BEST, C_SECOND, C_OFF = "\033[1;32m", "\033[33m", "\033[0m"
USE_COLOR = False
LEGEND = ""


def _legend():
    b = (C_BEST + "best" + C_OFF) if USE_COLOR else "* best"
    s2 = (C_SECOND + "2nd" + C_OFF) if USE_COLOR else "+ 2nd"
    return ("^ higher is better   v lower is better   ~1 target is 1.0   "
            f"{b}   {s2}\n"
            "         rank is among the MODEL rows only. It says nothing about the floor:\n"
            "         the best row can still be far above the instrument limit.")


def _val(cell):
    """Parse a printed cell back to a number for ranking. 'a/b' (ch<gt) -> a/b,
    trailing 'x' (ratios) stripped. Anything else is unrankable."""
    c = cell.strip().rstrip("x")
    if not c:
        return None
    if "/" in c:
        a, _, b = c.partition("/")
        try:
            return float(a) / float(b) if float(b) else None
        except ValueError:
            return None
    try:
        v = float(c)
    except ValueError:
        return None
    return v if v == v else None


def _rank(rows, idx, d, is_model):
    """Indices of the best and 2nd-best MODEL row in column idx, by direction d.
    Floor and 'x floor' rows never compete -- they are the ruler, not an entrant."""
    cand = [(i, _val(r[idx])) for i, r in enumerate(rows) if is_model(r)]
    cand = [(i, v) for i, v in cand if v is not None]
    if len(cand) < 2:
        return set(), set()
    key = {HI: lambda v: -v, LO: lambda v: v, ONE: lambda v: abs(v - 1.0)}[d]
    cand.sort(key=lambda t: key(t[1]))
    b = {i for i, v in cand if key(v) == key(cand[0][1])}
    rest = [(i, v) for i, v in cand if i not in b]
    if not rest or len(cand) < 3:
        return b, set()
    return b, {i for i, v in rest if key(v) == key(rest[0][1])}


def table(title, headers, rows, notes=(), left=2, dirs=None, rank=True):
    """Width-aware ASCII table. The first `left` columns are labels (left-aligned),
    everything after is numeric (right-aligned) so decimal points line up.
    `dirs` adds a marker row under the header saying which way is good, and -- where a
    direction exists -- marks the best/2nd-best model row in that column."""
    rows = [[("" if c is None else str(c)) for c in r] for r in rows]
    if not rows:
        return
    dirs = list(dirs) + [""] * (len(headers) - len(dirs)) if dirs else None
    mark = [[""] * len(headers) for _ in rows]
    if dirs and rank:
        # A label column entry like "  (floor)" or "  x floor" is not a competitor.
        is_model = lambda r: not r[1].strip().startswith(("(", "x "))
        for idx, d in enumerate(dirs):
            if not d or idx < left:
                continue
            b, s2 = _rank(rows, idx, d, is_model)
            for i in range(len(rows)):
                if rows[i][idx].strip():
                    mark[i][idx] = BEST if i in b else (SECOND if i in s2 else "")
        if not USE_COLOR:
            # A marker changes the string, so it must be in place BEFORE widths are taken.
            for i, r in enumerate(rows):
                for idx, d in enumerate(dirs):
                    if d and idx >= left and r[idx].strip():
                        r[idx] = r[idx] + (mark[i][idx] or " ")
    cols = [headers] + ([dirs] if dirs else []) + rows
    w = [max(len(c[i]) for c in cols) for i in range(len(headers))]
    def line(cells, pad=" ", mk=None):
        out = []
        for i in range(len(w)):
            t = cells[i].ljust(w[i], pad) if i < left else cells[i].rjust(w[i], pad)
            # Pad the PLAIN text first. An escape code is zero-width on screen but counts
            # in len(), so colouring before padding misaligns every column after it.
            if mk and mk[i]:
                t = (C_BEST if mk[i] == BEST else C_SECOND) + t + C_OFF
            out.append(t)
        return "  " + "  ".join(out).rstrip()
    print("\n" + title)
    print(line(headers))
    if dirs:
        print(line(dirs))
    print(line(["-" * x for x in w], "-"))
    for i, r in enumerate(rows):
        print(line(r, mk=mark[i] if USE_COLOR else None))
    for n in notes:
        print("  " + n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rs", default=os.environ.get("RS", RS_DEFAULT))
    ap.add_argument("--runs", default=RUNS_DEFAULT)
    ap.add_argument("--tag", default="valbench5s")
    ap.add_argument("--dirs", nargs="*", default=None)
    ap.add_argument("--csv", action="store_true", help="one flat CSV to stdout instead of tables")
    ap.add_argument("--color", default="auto", choices=["auto", "always", "never"],
                    help="auto = colour only when stdout is a terminal. Without colour the "
                         "ranking falls back to * / + so a piped or pasted table keeps it.")
    a = ap.parse_args()
    global USE_COLOR, LEGEND
    USE_COLOR = a.color == "always" or (a.color == "auto" and sys.stdout.isatty())
    LEGEND = _legend()

    items = []
    if a.dirs:
        for d in a.dirs:
            d = d.rstrip("/")
            # d = <rs>/<run>/inference-f77/<tag>/<ckpt>/pred_depth
            # parents: pred_depth -> ckpt -> tag -> inference-f77 -> run
            # The old code stopped at <tag>, so every --dirs row was labelled
            # "valbench5s" and two runs were indistinguishable.
            _p = d.split(os.sep)
            _run = _p[-5] if len(_p) >= 5 else "?"
            items.append((_run, os.path.basename(os.path.dirname(d)), d))
    else:
        for run in a.runs.split(","):
            run = run.strip()
            if not run:
                continue
            pat = os.path.join(a.rs, run, "inference-f77", a.tag, "checkpoint-*", "pred_depth")
            for d in sorted(glob.glob(pat),
                            key=lambda p: int(os.path.basename(os.path.dirname(p)).split("-")[-1])):
                items.append((run, os.path.basename(os.path.dirname(d)), d))
    if not items:
        raise SystemExit("[report] nothing found -- check --rs / --runs / --tag")

    R = []
    for run, ck, d in items:
        R.append(dict(run=_label(run),
                      ck=ck.replace("checkpoint-", "ck"), d=d,
                      vq=rd(d, "video_quality_metrics.csv"),
                      ch=rd(d, "channel_vs_da3_metrics.csv"),
                      cm=rd(d, "camera_control_metrics.csv"),
                      fl=rd(d, "da3_floor_metrics.csv"),
                      dp=rd(d, "depth_vs_da3_metrics.csv")))

    if a.csv:
        w = csv.writer(sys.stdout)
        w.writerow(["run", "ckpt", "n_vq", "psnr", "ssim", "lpips",
                    "n_ch", "ch_ade", "gt_ade", "floor_ade", "scale_ratio",
                    "n_cam", "ate", "roterr", "fde", "rpe_trans", "rpe_rot", "cammc", "scale",
                    "n_dep"] + [f"{p}_{m}" for m in ("delta1", "logl1") for p in "FABC"])
        for r in R:
            w.writerow([r["run"], r["ck"], len(r["vq"]), md(r["vq"], "psnr"), md(r["vq"], "ssim"),
                        md(r["vq"], "lpips"), len(r["ch"]), md(r["ch"], "ch_ade"),
                        md(r["ch"], "gt_ade"), md(r["ch"], "floor_ade"), md(r["ch"], "scale_ratio"),
                        len(r["cm"]), md(r["cm"], "da3_transerr"), md(r["cm"], "da3_roterr"),
                        md(r["cm"], "da3_fde"), md(r["cm"], "da3_rpe_trans"),
                        md(r["cm"], "da3_rpe_rot_deg"), md(r["cm"], "da3_cammc"),
                        md(r["cm"], "da3_scale"), len(r["dp"])]
                       + [md(r["dp"], f"{p}_{m}") for m in ("delta1", "logl1") for p in "FABC"])
        return

    print("\nlegend:  " + LEGEND)
    table("#2  VIDEO QUALITY",
          ["run", "ckpt", "n", "psnr", "ssim", "lpips"],
          [[r["run"], r["ck"], len(r["vq"]), num(md(r["vq"], "psnr"), 3),
            num(md(r["vq"], "ssim")), num(md(r["vq"], "lpips"))] for r in R if r["vq"]],
          dirs=["", "", "", HI, HI, LO])

    rows = []
    for r in R:
        if not r["ch"]:
            continue
        ca, fa = md(r["ch"], "ch_ade"), md(r["ch"], "floor_ade")
        rows.append([r["run"], r["ck"], len(r["ch"]), num(ca), num(md(r["ch"], "gt_ade")), num(fa),
                     num(ca / fa, 2) if fa == fa and fa > 0 else "",
                     lt_frac(r["ch"], "ch_ade", "gt_ade"), num(md(r["ch"], "scale_ratio"), 3)])
    table("#3  ACTION (raymap) <-> VIDEO      -- does the channel steer the video?",
          ["run", "ckpt", "n", "ch_ade", "gt_ade", "floor", "ch/fl", "ch<gt", "scale"], rows,
          dirs=["", "", "", LO, "", "", LO, HI, ONE], notes=[
           "good = ch_ade near floor AND ch_ade < gt_ade.",
           "floor is a RATIO only: ch_ade is raymap<->DA3 and shares no GT reference, so",
           "ch_ade - floor would be arithmetic without meaning (viz_da3_pointcloud.py:873-875).",
           "ch_roterr is a known bug (all 90.0003 deg) and is not reported."])

    def cam_group(r, keys, extra=None):
        """model row, its floor row, and the ratio between them -- kept together so the
        floor is impossible to overlook. `extra` holds already-ratio columns (drift, scale)
        which get no 'x floor' entry: a ratio of ratios is not a thing anyone reads."""
        g = [md(r["cm"], "da3_" + k) for k in keys]
        f = [md(r["fl"], "floor_" + k) for k in keys] if r["fl"] else None
        ge = [v(r["cm"], "da3_") for v in (extra or [])]
        fe = [v(r["fl"], "floor_") for v in (extra or [])] if r["fl"] else None
        out = [[r["run"], r["ck"], len(r["cm"])]
               + [num(x, 4 if k != "roterr" and "rot" not in k else 3) for x, k in zip(g, keys)]
               + [num(x, 2) for x in ge]]
        if f:
            out.append(["", "  (floor)", len(r["fl"])]
                       + [num(x, 4 if k != "roterr" and "rot" not in k else 3)
                          for x, k in zip(f, keys)]
                       + [num(x, 2) for x in fe])
            out.append(["", "  x floor", ""]
                       + [num(a / b, 2) + "x" if b == b and b > 0 else ""
                          for a, b in zip(g, f)]
                       + [""] * len(ge))
        return out

    drift_t = lambda rows, p: md(rows, p + "transerr_last") / md(rows, p + "transerr")
    drift_r = lambda rows, p: md(rows, p + "roterr_last") / md(rows, p + "roterr")
    scale_c = lambda rows, p: md(rows, p + "scale")

    rows = []
    for r in R:
        if r["cm"]:
            rows += cam_group(r, ["transerr", "roterr", "fde"], [drift_t, drift_r, scale_c])
    table("#1a  CAMERA -- ABSOLUTE (DA3 vs GT, after sim3)",
          ["run", "ckpt", "n", "ATE(m)", "roterr(deg)", "fde(m)", "drift_t", "drift_r", "scale"],
          rows, dirs=["", "", "", LO, LO, LO, ONE, ONE, ""], notes=[
           "ATE = transerr (3D). roterr is ABSOLUTE rotation -- a different axis from rpe_rot;",
           "  a run can be bad here and fine there (large but steady) or the reverse.",
           "drift = last-frame error / mean error. Near 1 the error does not accumulate;",
           "  above the floor's own drift it does. Compare each row to the floor row, not to 1.",
           "scale = the sim3 scale DA3 needed. It is arbitrary per video, so the VALUE means",
           "  nothing -- only the model-vs-floor GAP does: a gap says DA3 reads the generated",
           "  video at a different metric size than the real one, which sim3 hides from ATE.",
           "'x floor' is blank for drift and scale: they are already ratios."])

    rows = []
    for r in R:
        if r["cm"]:
            rows += cam_group(r, ["rpe_trans", "rpe_rot_deg", "rpe_camera", "cammc"])
    table("#1b  CAMERA -- RELATIVE / COMPOSITE",
          ["run", "ckpt", "n", "rpe_t(m)", "rpe_rot(deg)", "rpe_cam", "cammc"],
          rows, dirs=["", "", "", LO, LO, LO, LO], notes=[
           "rpe_camera and cammc are composite 3x4 Frobenius distances: reported because",
           "  WorldCam/GameCraft report them, not because they localise a failure.",
           "an 'x floor' near 1.00x means that channel is AT the instrument limit -- further",
           "  improvement there cannot be measured with DA3, so do not chase it."])

    rows = []
    for r in R:
        if not r["dp"]:
            continue
        fl1 = md(r["dp"], "F_logl1")
        cf = md(r["dp"], "C_logl1") / fl1 if fl1 == fl1 and fl1 > 0 else float("nan")
        rows.append([r["run"], r["ck"], len(r["dp"])]
                    + [num(md(r["dp"], f"{p}_delta1"), 3) for p in "FABC"]
                    + [num(fl1, 3), num(md(r["dp"], "C_logl1"), 3), num(cf, 2)])
    table("#4  DEPTH <-> VIDEO      F=DA3(GT vid)  A=DA3(gen vid)  B=pred  C=pred vs DA3(gen)",
          ["run", "ckpt", "n", "F d1", "A d1", "B d1", "C d1", "F lgl", "C lgl", "C/F"], rows,
          dirs=["", "", "", HI, HI, HI, HI, LO, LO, ONE], notes=[
           "Baseline for C is F, NOT A: A and B also carry 'the generated video differs",
           "from GT', C does not, so C < A is near-automatic and means nothing.",
           "C/F near 1.0 = consistent to the measurement limit."])

    miss = [[r["run"], r["ck"], " ".join(n for n, k in
             (("#2", "vq"), ("#3", "ch"), ("#1", "cm"), ("#4", "dp")) if not r[k])] for r in R]
    miss = [m for m in miss if m[2]]
    if miss:
        table("NOT MEASURED YET", ["run", "ckpt", "axes"], miss, left=3)


if __name__ == "__main__":
    main()
