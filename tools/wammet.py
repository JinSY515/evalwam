#!/usr/bin/env python3
"""Terminal metric viewer for valbench5s / DL3DV fwd27 / LookOut runs.

  wammet                                   # default set: baselines + every hg0_grad0.5 ckpt with results
  wammet hg0_grad0.5                       # all ckpts of a run   (name may omit 'goalprog_mode_real6_nomr_')
  wammet hg0_grad0.5/ckpt1750 maskfix/ckpt3000 d10_dpt_vcons/ckpt3000
  wammet -b vb            # only one benchmark: vb | dl3dv | lo   (comma list ok)
  wammet --full           # every numeric key in suite_summary.json, not just the curated list
  wammet --csv            # machine-readable (bench,metric,run,value)
  wammet --list           # list every run/ckpt that has results

Best value per row is green, second yellow (only when >= 2 runs have the metric).
Reads: suite_summary.json, fvd_musiq.json, video_quality_metrics.csv, met3r.json, goal_image.json.
"""
import argparse, csv, glob, json, math, os, re, signal, statistics as st, sys
signal.signal(signal.SIGPIPE, signal.SIG_DFL)

RS = "/root/project/egowam/results_syn"
BENCH = {"vb": ("valbench5s (120)", "valbench5s_all120"),
         "dl3dv": ("DL3DV fwd27 (27)", "dl3dv_fwd27/samples"),
         "lo": ("LookOut (51x2 chunks)", "lookout_turn_cv_all51")}
PREFIXES = ["", "goalprog_mode_real6_nomr_", "goalprog_dl3dv_camfix_", "goalprog_"]
DEFAULT = ["maskfix/ckpt3000", "d10_dpt_vcons/ckpt3000", "hg0_grad0.5"]

# (group, label, key, direction)  direction: -1 lower better, +1 higher better, 0 neutral
CURATED = [
    ("Video quality", "FVD (full length)", "fvd:fvd_full", -1),
    ("Video quality", "FVD16", "fvd:fvd16", -1),
    ("Video quality", "MUSIQ", "fvd:musiq", +1),
    ("Video quality", "MUSIQ (GT ref)", "fvd:musiq_gt_ref", 0),
    ("Video quality", "PSNR", "vq:psnr", +1),
    ("Video quality", "SSIM", "vq:ssim", +1),
    ("Video quality", "LPIPS", "vq:lpips", -1),
    ("Video quality", "MEt3R", "met3r:met3r", -1),
    ("Video quality", "MEt3R (GT ref)", "met3r:met3r_gt_ref", 0),
    ("Action faithfulness (pred action vs GT)", "ATE (m)", "act_ate", -1),
    ("Action faithfulness (pred action vs GT)", "ATE / path", "act_ate_path", -1),
    ("Action faithfulness (pred action vs GT)", "RotErr (deg)", "act_roterr", -1),
    ("Action faithfulness (pred action vs GT)", "TransErr (m)", "act_transerr", -1),
    ("Action faithfulness (pred action vs GT)", "CamMC", "act_cammc", -1),
    ("Action faithfulness (pred action vs GT)", "RPE rot (deg)", "act_rpe_rot", -1),
    ("Action faithfulness (pred action vs GT)", "RPE trans", "act_rpe_trans", -1),
    ("Coherence (pred action vs DA3 on generated video)", "ATE (m)", "coh_ate", -1),
    ("Coherence (pred action vs DA3 on generated video)", "RotErr (deg)", "coh_roterr", -1),
    ("Coherence (pred action vs DA3 on generated video)", "CamMC", "coh_cammc", -1),
    ("Coherence (pred action vs DA3 on generated video)", "RPE rot (deg)", "coh_rpe_rot", -1),
    ("Coherence (pred action vs DA3 on generated video)", "RPE trans", "coh_rpe_trans", -1),
    ("Generated-video camera (DA3) vs GT", "ATE (m)", "da3_ate", -1),
    ("Generated-video camera (DA3) vs GT", "RotErr (deg)", "da3_roterr", -1),
    ("Generated-video camera (DA3) vs GT", "CamMC", "da3_cammc", -1),
    ("Generated-video camera (DA3) vs GT", "RPE rot (deg)", "da3_rpe_rot", -1),
    ("Generated-video camera (DA3) vs GT", "RPE trans", "da3_rpe_trans", -1),
    ("Goal", "success (DA3 cam)", "goal_da3_success", +1),
    ("Goal", "arrival rate (DA3 cam)", "goal_da3_arrival_rate", +1),
    ("Goal", "final pos/path (DA3 cam)", "goal_da3_final_pos_path", -1),
    ("Goal", "final heading (DA3 cam, deg)", "goal_da3_final_head_deg", -1),
    ("Goal", "success (pred action)", "goal_act_success", +1),
    ("Goal", "arrival rate (pred action)", "goal_act_arrival_rate", +1),
    ("Goal", "final pos/path (pred action)", "goal_act_final_pos_path", -1),
    ("Goal", "final heading (pred action, deg)", "goal_act_final_head_deg", -1),
    ("Goal", "goal image DINO sim", "goal:goal_dino_sim", +1),
    ("Goal", "goal image LPIPS", "goal:goal_lpips", -1),
    ("Depth (pred vs DA3 on generated video)", "delta1", "depth_C_delta1", +1),
    ("Depth (pred vs DA3 on generated video)", "delta1.5", "depth_C_delta15", +1),
    ("Depth (pred vs DA3 on generated video)", "AbsRel", "depth_C_absrel", -1),
]


def resolve(spec):
    """'hg0_grad0.5/ckpt1750' | 'hg0_grad0.5' (all ckpts) | full path -> [(label, ckpt_dir)]"""
    spec = spec.rstrip("/")
    if spec.startswith(RS):
        spec = spec[len(RS) + 1:]
    run, _, ck = spec.partition("/")
    ck = ck.split("/")[0]
    for p in PREFIXES:
        rd = f"{RS}/{p}{run}"
        if os.path.isdir(rd):
            cks = [f"{rd}/{ck}"] if ck else sorted(glob.glob(f"{rd}/ckpt*"),
                                                   key=lambda d: int(re.sub(r"\D", "", os.path.basename(d)) or 0))
            out = [(f"{run}/{os.path.basename(c)}", c) for c in cks if os.path.isdir(c)]
            if out:
                return out
    sys.exit(f"[wammet] no run matches '{spec}' under {RS}")


def load(d):
    if not os.path.isdir(d):
        return None
    r = {}
    p = f"{d}/suite_summary.json"
    if os.path.exists(p):
        r.update({k: v for k, v in json.load(open(p)).items()
                  if isinstance(v, (int, float)) and not k.endswith("__n")})
    for f, tag in (("fvd_musiq.json", "fvd"), ("met3r.json", "met3r"), ("goal_image.json", "goal")):
        p = f"{d}/{f}"
        if os.path.exists(p):
            j = json.load(open(p))
            for k, v in j.items():
                if isinstance(v, (int, float)):
                    r[f"{tag}:{k}"] = v
            if tag == "fvd" and "frames" in j and f"fvd{j['frames']}" in j:
                r["fvd:fvd_full"] = j[f"fvd{j['frames']}"]
    p = f"{d}/video_quality_metrics.csv"
    if os.path.exists(p):
        rows = list(csv.DictReader(open(p)))
        if rows:
            for k in ("psnr", "ssim", "lpips"):
                col = next((c for c in rows[0] if c.lower().startswith(k)), None)
                if col:
                    v = [float(x[col]) for x in rows if x[col] not in ("", "nan")]
                    if v:
                        r[f"vq:{k}"] = st.mean(v)
    return r or None


def fmt(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "-"
    a = abs(v)
    return f"{v:.0f}" if a >= 1000 else f"{v:.1f}" if a >= 100 else f"{v:.2f}" if a >= 10 else f"{v:.3f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*")
    ap.add_argument("-b", "--bench", default="vb,dl3dv,lo")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--csv", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--no-color", action="store_true")
    a = ap.parse_args()

    if a.list:
        for b, (name, sub) in BENCH.items():
            hits = sorted(glob.glob(f"{RS}/*/ckpt*/{sub}/suite_summary.json"))
            print(f"{name}:")
            for h in hits:
                print("   ", h[len(RS) + 1:].split(f"/{sub}")[0].replace("goalprog_mode_real6_nomr_", ""))
        return

    runs = [x for s in (a.runs or DEFAULT) for x in resolve(s)]
    seen, uniq = set(), []
    for lab, d in runs:
        if d not in seen:
            seen.add(d); uniq.append((lab, d))
    color = sys.stdout.isatty() and not a.no_color and not a.csv
    G, Y, B, DIM, R0 = ("\033[32m", "\033[33m", "\033[1m", "\033[2m", "\033[0m") if color else ("",) * 5

    if a.csv:
        w = csv.writer(sys.stdout); w.writerow(["bench", "metric", "run", "value"])
    for b in a.bench.split(","):
        name, sub = BENCH[b]
        cols = [(lab, load(f"{d}/{sub}")) for lab, d in uniq]
        cols = [(lab, m) for lab, m in cols if m]
        if not cols:
            continue
        if a.full:
            keys = sorted(set().union(*[m.keys() for _, m in cols]))
            rows = [("all", k, k, next((dd for _, _, kk, dd in CURATED if kk == k), 0)) for k in keys]
        else:
            rows = CURATED
        if a.csv:
            for _, lab, k, _ in rows:
                for cl, m in cols:
                    if k in m:
                        w.writerow([b, k, cl, m[k]])
            continue
        lw = max(len(r[1]) for r in rows) + 2
        cw = [max(len(cl), 7) + 2 for cl, _ in cols]
        print(f"\n{B}=== {name} ==={R0}")
        print(" " * lw + "".join(f"{cl:>{w_}}" for (cl, _), w_ in zip(cols, cw)))
        grp = None
        for g, lab, k, dirn in rows:
            vals = [m.get(k) for _, m in cols]
            if all(v is None for v in vals):
                continue
            if g != grp:
                grp = g; print(f"{DIM}-- {g}{R0}")
            rank = {}
            got = [(v, i) for i, v in enumerate(vals) if v is not None]
            if dirn and len(got) >= 2:
                order = sorted(got, key=lambda t: -dirn * t[0])
                rank[order[0][1]] = G
                if len(got) >= 3:
                    rank[order[1][1]] = Y
            line = f"{lab:<{lw}}"
            for i, (v, w_) in enumerate(zip(vals, cw)):
                s = fmt(v)
                line += " " * (w_ - len(s)) + (rank.get(i, "") + s + R0 if i in rank else s)
            print(line)
    if not a.csv:
        print(f"\n{DIM}green = best, yellow = 2nd.  dirs: {RS}/<run>/<ckpt>/{{valbench5s_all120, dl3dv_fwd27/samples, lookout_turn_cv_all51}}{R0}")


if __name__ == "__main__":
    main()
