#!/usr/bin/env python3
"""Collect the extended suite into one table (rows = metric list, columns = runs).
Reads suite_summary.json, fvd_musiq.json, video_quality_metrics.csv from each run dir.
  python3 tools/report_suite.py          -> terminal table in the metric-list order (+ writes results_syn/EVAL_SUITE.md)
  python3 tools/report_suite.py --md     -> print the markdown instead
  python3 tools/report_suite.py --no-color   -> terminal table without colours (best=green, 2nd=yellow per row)
"""
import csv, glob, json, os, sys, statistics as st, datetime

RS = "/root/project/egowam/results_syn"
B = f"{RS}/goalprog_mode_real6_nomr_maskfix"
RUNS = [
    ("old vcons ck3000", f"{RS}/plucker_fidelity/e2/full_all120"),
    ("I jointfm ck3000", f"{RS}/abl/I_jointfm/ckpt3000/valbench5s_all120"),
    ("III goalaux ck8000", f"{RS}/abl/III_goalaux/ckpt8000/valbench5s_all120"),
    ("IV depthpred ck3000", f"{RS}/abl/IV_depthpred/ckpt3000/valbench5s_all120"),
    ("IV depthpred ck5000", f"{RS}/abl/IV_depthpred/ckpt5000/valbench5s_all120"),
    ("vcons_mode ck500", f"{RS}/goalprog_dl3dv_camfix_d10_dpt_vcons_mode/ckpt500/valbench5s_all120"),
    ("maskfix ck500", f"{B}/ckpt500/valbench5s_all120"),
    ("maskfix ck1000*", f"{B}/ckpt1000/valbench5s"),
    ("maskfix ck2000", f"{B}/ckpt2000/valbench5s_all120"),
    ("maskfix ck2500", f"{B}/ckpt2500/valbench5s_all120"),
    ("maskfix ck3000", f"{B}/ckpt3000/valbench5s_all120"),
    ("maskfix ck3500", f"{B}/ckpt3500/valbench5s_all120"),
]
G_FAITH, G_COH = "Action faithfulness (vs GT, Sim3)", "Video-action coherence (vs DA3 on generated RGB)"
G_GA, G_GD = "Goal achievement (generated action vs GT)", "Goal achievement (DA3 camera of generated RGB vs GT)"
CAM = [("abs RotErr (deg)", "roterr", "v"), ("abs TransErr (m)", "transerr", "v"), ("abs TransErr RMSE = ATE (m)", "ate", "v"),
       ("abs CamMC", "cammc", "v"), ("abs TransErr / path length", "transerr_path", "v"),
       ("ATE / path length", "ate_path", "v"), ("RPE RotErr (deg)", "rpe_rot", "v"),
       ("RPE TransErr (m)", "rpe_trans", "v"), ("RPE CamMC", "rpe_cammc", "v"), ("RPE TransErr / path length", "rpe_trans_path", "v")]
GOAL = [("final position error / path length", "final_pos_path", "v"), ("final heading error (deg)", "final_head_deg", "v"),
        ("arrival time ratio (arrived only)", "arrival_ratio", "~1"), ("arrival rate", "arrival_rate", "^"),
        ("success rate (<10%, <10 deg)", "success", "^")]
G_VQ, G_3D, G_GOAL = "Video quality", "3D consistency", "Goal achievement"
G_EX = "Extra: DA3 camera of generated RGB vs GT (Sim3)"
ROWS = ([(G_VQ, "FVD (16 frames)", "fvd:fvd16", "v"), (G_VQ, "FVD (all frames)", "fvd:fvdall", "v"),
         (G_3D, "PSNR", "vq:psnr", "^"), (G_3D, "SSIM", "vq:ssim", "^"),
         (G_3D, "LPIPS", "vq:lpips", "v"), (G_3D, "MEt3R", "m3:met3r", "v")]
        + [(G_FAITH, n, "s:act_" + k, d) for n, k, d in CAM]
        + [(G_COH, "depth AbsRel", "s:depth_C_absrel", "v"), (G_COH, "depth delta < 1.25", "s:depth_C_delta1", "^"),
           (G_COH, "depth delta < 1.5", "s:depth_C_delta15", "^")]
        + [(G_COH, "cam " + n.replace(" (m)", " (DA3 units)").replace("abs TransErr RMSE = ATE (DA3 units)", "abs TransErr RMSE = ATE"),
            "s:coh_" + k, d) for n, k, d in CAM]
        + [(G_GOAL, "goal pose success rate", "s:goal_da3_success", "^"),
           (G_GOAL, "goal image LPIPS", "gi:goal_lpips", "v"),
           (G_GOAL, "goal image DINO similarity", "gi:goal_dino_sim", "^")]
        + [(G_EX, n, "s:da3_" + k, d) for n, k, d in CAM if k != "ate" and k != "rpe_trans_path"])
LOOKOUT = [("vcons ck3000", f"{RS}/goalprog_dl3dv_camfix_d10_dpt_vcons/ckpt3000/lookout_turn_cv_all51"),
           ("maskfix ck2000", f"{B}/ckpt2000/lookout_turn_cv_all51")]
SHORT_L = ["vcons3k", "mf2000"]
SHORT = ["vcons3k", "I_3k", "III_8k", "IV_3k", "IV_5k", "vmode500", "mf500", "mf1000*", "mf2000", "mf2500", "mf3000", "mf3500"]
DL3DV = [("vcons ck3000", f"{RS}/goalprog_dl3dv_camfix_d10_dpt_vcons/ckpt3000/dl3dv_fwd27/samples"),
         ("maskfix ck3000", f"{B}/ckpt3000/dl3dv_fwd27/samples"),
         ("maskfix ck3000 hg0", f"{B}/ckpt3000/dl3dv_fwd27_hg0/samples")]
SHORT_D = ["vcons3k", "mf3000", "mf3k_hg0"]

# hg0_grad0.5 (HF clwm515/wam): every ckpt whose suite finished is picked up automatically, per benchmark
H = f"{RS}/goalprog_mode_real6_nomr_hg0_grad0.5"
for _n in sorted(int(os.path.basename(q)[4:]) for q in glob.glob(f"{H}/ckpt*") if os.path.basename(q)[4:].isdigit()):
    for _lst, _sh, _sub in ((RUNS, SHORT, "valbench5s_all120"), (LOOKOUT, SHORT_L, "lookout_turn_cv_all51"),
                            (DL3DV, SHORT_D, "dl3dv_fwd27/samples")):
        if os.path.exists(f"{H}/ckpt{_n}/{_sub}/suite_summary.json"):
            _lst.append((f"hg0_grad0.5 ck{_n}", f"{H}/ckpt{_n}/{_sub}")); _sh.append(f"g05_{_n}")
# IV continued (goalprog_dl3dv_camfix_d10_dpt_cont5k ck1000 = d10_dpt 5000 + 1000): column right after IV_5k once evaluated
_p = f"{RS}/abl/IV_cont5k/ckpt6000/valbench5s_all120"
if os.path.exists(f"{_p}/suite_summary.json"):
    _i = SHORT.index("IV_5k") + 1
    RUNS.insert(_i, ("IV cont5k ck6000 (d10_dpt 5000+1000)", _p)); SHORT.insert(_i, "IV_6k")

# II (= III w/o goal aux, goalprog_appearance_nogoalaux ck3000): column right after I_3k
_p2 = f"{RS}/abl/II_nogoalaux/ckpt3000/valbench5s_all120"
if os.path.exists(f"{_p2}/suite_summary.json"):
    _i = SHORT.index("I_3k") + 1
    RUNS.insert(_i, ("II nogoalaux ck3000", _p2)); SHORT.insert(_i, "II_3k")


def load(d):
    # LookOut GT poses were mirrored L/R (lookout_val_cv); <dir>_lh holds the re-scored copy against
    # lookout_val_cv_lh (tools/lookout_lh_reeval.py) -- prefer it whenever it exists.
    if os.path.exists(os.path.join(d.rstrip("/") + "_lh", "suite_summary.json")):
        d = d.rstrip("/") + "_lh"
    out = {}
    p = os.path.join(d, "suite_summary.json")
    if os.path.exists(p):
        out.update({"s:" + k: v for k, v in json.load(open(p)).items()})
    p = os.path.join(d, "fvd_musiq.json")
    if os.path.exists(p):
        f = json.load(open(p))
        out.update({"fvd:" + k: v for k, v in f.items()})
        out["fvd:fvdall"] = f.get(f"fvd{f.get('frames')}")
    p = os.path.join(d, "goal_image.json")
    if os.path.exists(p):
        out.update({"gi:" + k: v for k, v in json.load(open(p)).items()})
    # MEt3R: adjacent-frame pairs (stride 1, the external-table convention) from <dir>/met3r_s1/ when present,
    # else the older stride-4 met3r.json (one latent frame apart; ~2x larger values, not comparable).
    _b = d[:-3] if d.endswith("_lh") else d
    p = next((q for q in (os.path.join(d, "met3r_s1", "met3r.json"), os.path.join(_b, "met3r_s1", "met3r.json"),
                          os.path.join(d, "met3r.json")) if os.path.exists(q)), None)
    if p:
        out.update({"m3:" + k: v for k, v in json.load(open(p)).items()})
        out["m3:stride"] = json.load(open(p)).get("stride", 4)
    p = os.path.join(d, "video_quality_metrics.csv")
    if os.path.exists(p):
        r = [x for x in csv.DictReader(open(p)) if x.get("psnr")]
        for k in ("psnr", "ssim", "lpips"):
            out["vq:" + k] = st.mean(float(x[k]) for x in r)
    return out


def fmt(v):
    if v is None:
        return "–"
    if isinstance(v, (int, float)):
        return f"{v:.3f}"   # user: always 3 decimals
    return str(v)


def tfmt(v):
    if not isinstance(v, (int, float)):
        return "-"
    a = abs(v)
    return f"{v:.1f}" if a >= 100 else (f"{v:.2f}" if a >= 10 else f"{v:.4f}")


GREEN, YELLOW, RESET = "\033[1;32m", "\033[1;33m", "\033[0m"


def rank_colors(vals, d):
    """Per-row colours: best -> GREEN, 2nd -> YELLOW, by direction d ('v' low, '^' high, '~1' closest to 1).
    Ties share the colour. Non-numeric cells are not ranked."""
    key = {"v": lambda v: v, "^": lambda v: -v, "~1": lambda v: abs(v - 1.0)}.get(d)
    num = sorted({key(v) for v in vals if isinstance(v, (int, float))}) if key else []
    out = []
    for v in vals:
        if not isinstance(v, (int, float)) or not num:
            out.append("")
        elif key(v) == num[0]:
            out.append(GREEN)
        elif len(num) > 1 and key(v) == num[1]:
            out.append(YELLOW)
        else:
            out.append("")
    return out


def term(data, runs=RUNS, shorts=SHORT, title="valbench5s, 120 windows (CARLA, 77 frames)   "
         "(^ higher better, otherwise lower better; * ck1000 medium/hard used other goal frames)", nw=46, cw=9):
    ref = {"m3:met3r": next((x["m3:met3r_gt_ref"] for x in data if "m3:met3r_gt_ref" in x), None)}
    data = data + [ref]
    short = shorts + ["GT"]
    bar = "-" * (nw + 2 + cw * len(short))
    out = [title,
           *[f"  {s:<9}= {n}" for s, (n, _) in zip(shorts, runs)],
           "  GT       = GT clips of the same windows (reference, where defined)", "",
           " " * (nw + 2) + "".join(f"{s:>{cw}}" for s in short)]
    g0 = None
    for g, n, k, d in ROWS:
        if g != g0:
            out += [bar, g]; g0 = g
        name = n + (" ^" if d == "^" else (" ~1" if d == "~1" else ""))
        vals = [x.get(k) for x in data]
        cols = rank_colors(vals[:len(runs)], d) + [""] * (len(vals) - len(runs)) if COLOR else [""] * len(vals)
        out.append(f"  {name:<{nw}}" + "".join(
            (c + f"{tfmt(v):>{cw}}" + RESET) if c else f"{tfmt(v):>{cw}}" for v, c in zip(vals, cols)))
    out.append(bar)
    return "\n".join(out)


COLOR = "--no-color" not in sys.argv


def main():
    data = [load(d) for _, d in RUNS]
    L = [f"# Extended metric suite -- valbench5s, 120 windows ({datetime.datetime.now(datetime.timezone.utc):%m-%d %H:%M} UTC)\n",
         "v lower better, ^ higher better, ~1 closer to 1 better. – = not computed / not applicable "
         "(I has no depth/raymap head, III has no depth head; MEt3R: MASt3R+DINO16/FeatUp cosine, frame pairs 4 apart).",
         "* ck1000: medium/hard rows used different goal frames (run as one 120-sample job).\n",
         "| group | metric | dir | " + " | ".join(n for n, _ in RUNS) + " |",
         "|---|---|---|" + "---|" * len(RUNS)]
    for g, n, k, d in ROWS:
        L.append(f"| {g} | {n} | {d} | " + " | ".join(fmt(x.get(k)) for x in data) + " |")
    L += ["", "Definitions: see evalwam/tools/eval_suite.py header. Action = planar ego-motion per latent frame "
          "(20 key frames), integrated from the GT start (a_0 is given). Raymap runs: poses inverted from the "
          "predicted raymap; run I: action head. Coherence uses DA3 on the generated RGB as reference; its saved "
          "poses are already sim3-fit to GT, so 'DA3 units' are metres. Goal metrics are start-anchored, no alignment "
          "(action) / DA3 poses as saved (DA3). FVD: I3D (cd-fvd) vs GT clips of the same 120 windows -- small set, "
          "compare runs only."]
    txt = "\n".join(L) + "\n"
    open(f"{RS}/EVAL_SUITE.md", "w").write(txt)
    if "--md" in sys.argv:
        print(txt)
    else:
        print(term(data))
        dl = [load(d) for _, d in LOOKOUT]
        print("\n" + term(dl, LOOKOUT, SHORT_L, "LookOut rollouts, 51 seqs (real Aria, 153 frames = 2 chunks, chained; "
                                             "no GT depth -> depth = pred vs DA3 only)"))
        dd = [load(d) for _, d in DL3DV]
        print("\n" + term(dd, DL3DV, SHORT_D, "DL3DV fwd27, 27 windows (real, 77 frames; depth = pred vs DA3)"))


if __name__ == "__main__":
    main()
