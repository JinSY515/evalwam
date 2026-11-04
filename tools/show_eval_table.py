#!/usr/bin/env python3
"""valbench5s 평가를 축별(VIDEO QUALITY / DEPTH / CAMERA) 표 셋으로 보여준다.
열마다 최고=초록, 2위=노랑.

순위는 **열 단위**이고 방향은 지표마다 다르다 (GROUPS 의 마지막 필드).
F_d1 / A_* / maskfrac / succ / scale 은 계측기 바닥값·커버리지·진단값이라 색을 안 칠한다 --
런끼리 같아야 정상이고, 달라지면 비교가 깨졌다는 신호다.

Usage:
  python3 tools/show_eval_table.py
  python3 tools/show_eval_table.py --runs goalprog_dl3dv_camfix_d10_dpt
  python3 tools/show_eval_table.py --only depth
  python3 tools/show_eval_table.py --no-color | tee table.txt
"""
import argparse, csv, os, sys
import numpy as np

G, Y, RS, DIM = "\033[1;32m", "\033[1;33m", "\033[0m", "\033[2m"
VQ, DP, CM = "video_quality_metrics.csv", "depth_vs_da3_metrics.csv", "camera_control_metrics.csv"

# 이름, CSV, 열, 폭, 높을수록좋음(None=순위없음)
GROUPS = [
    ("VIDEO QUALITY  (생성 영상 vs 실제 CARLA 프레임, 프레임 평균)", [
        ("psnr",  VQ, "psnr",  9, True),
        ("ssim",  VQ, "ssim",  8, True),
        ("lpips", VQ, "lpips", 9, False),
    ]),
    ("DEPTH  B=예측vs GT / C=예측vs DA3(생성) / A=DA3(생성)vs GT / F=DA3(GT)vs GT=바닥값", [
        ("B_d1",     DP, "B_delta1",   9, True),
        ("B_absrel", DP, "B_absrel",  10, False),
        ("B_si_l1",  DP, "B_si_l1",    9, False),
        ("B_logl1",  DP, "B_logl1",    9, False),
        ("C_d1",     DP, "C_delta1",   9, True),
        ("C_logl1",  DP, "C_logl1",    9, False),
        ("A_d1",     DP, "A_delta1",   9, None),
        ("F_d1",     DP, "F_delta1",   9, None),
        ("maskfrac", DP, "mask_frac", 10, None),
    ]),
    ("CAMERA  (생성 영상에서 SfM 복원한 궤적 vs GT 포즈, Sim3 정렬 후)", [
        ("succ",     CM, "da3_success",        7, None),
        ("ade",      CM, "da3_ade",            8, False),
        ("fde",      CM, "da3_fde",            8, False),
        ("transerr", CM, "da3_transerr",      10, False),
        ("tr_last",  CM, "da3_transerr_last",  9, False),
        ("roterr",   CM, "da3_roterr",         9, False),
        ("rot_last", CM, "da3_roterr_last",   10, False),
        ("rotstep",  CM, "da3_rotstep_err",    9, False),
        ("rpe_tr",   CM, "da3_rpe_trans",      9, False),
        ("rpe_rot",  CM, "da3_rpe_rot_deg",   10, False),
        ("scale",    CM, "da3_scale",          8, None),
    ]),
]


def mean_col(rows, key):
    v = []
    for x in rows:
        s = x.get(key)
        if s in (None, "", "nan", "None"):
            continue
        v.append(1.0 if s == "True" else 0.0 if s == "False" else float(s))
    return float(np.mean(v)) if v else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/root/project/ego_wam/results_syn")
    ap.add_argument("--runs", default="goalprog_dl3dv_camfix_d10_dpt,"
                                      "goalprog_appearance_depth_fine_d10_dpt,"
                                      "goalprog_appearance_depth_fine_d10_dpt_vcons")
    ap.add_argument("--tag", default="valbench5s")
    ap.add_argument("--sub", default="pred_depth")
    ap.add_argument("--only", default="", help="vq | depth | camera (쉼표로 여러 개)")
    ap.add_argument("--no-color", dest="color", action="store_false", default=True)
    a = ap.parse_args()
    color = a.color and sys.stdout.isatty()
    want = [x.strip().lower() for x in a.only.split(",") if x.strip()]

    recs = []
    for r in [x.strip() for x in a.runs.split(",") if x.strip()]:
        base = f"{a.root}/{r}/inference-f77/{a.tag}"
        if not os.path.isdir(base):
            continue
        for c in sorted(os.listdir(base), key=lambda x: int(x.split("-")[-1])):
            D = f"{a.root}/{r}/inference-f77/{a.tag}/{c}/{a.sub}"
            cache = {}
            for f in (VQ, DP, CM):
                p = f"{D}/{f}"
                cache[f] = list(csv.DictReader(open(p))) if os.path.exists(p) else []
            if not any(cache.values()):
                continue
            rec = {"run": r.replace("goalprog_", "").replace("appearance_depth_fine_", ""),
                   "ckpt": int(c.split("-")[-1]),
                   "n": max(len(v) for v in cache.values())}
            for _, cols in GROUPS:
                for _, f, k, _, _ in cols:
                    rec[k] = mean_col(cache[f], k) if cache[f] else float("nan")
            recs.append(rec)
    if not recs:
        print("평가 CSV 를 못 찾았다 - --root/--runs/--tag 확인"); return

    lab = lambda x: f"{x['run']:26s}{x['ckpt']:>6d}{x['n']:>5d}  "
    arrow = {True: "\u2191", False: "\u2193", None: " "}
    for gi, (title, cols) in enumerate(GROUPS):
        key = ("vq", "depth", "camera")[gi]
        if want and key not in want:
            continue
        rank = {}
        for _, f, k, _, hi in cols:
            if hi is None:
                continue
            fin = [(x[k], i) for i, x in enumerate(recs) if np.isfinite(x.get(k, np.nan))]
            if len(fin) < 2:
                continue
            fin.sort(key=lambda t: -t[0] if hi else t[0])
            rank[k] = (fin[0][1], fin[1][1])
        head = f"{'run':26s}{'ckpt':>6s}{'n':>5s}  " + "".join(
            f"{nm + arrow[hi]:>{w}s}" for nm, _, _, w, hi in cols)
        print(f"\n{'[' + key.upper() + ']':12s}{title}")
        print(head)
        print("-" * len(head))
        for i, x in enumerate(recs):
            line = lab(x)
            for _, f, k, w, hi in cols:
                v = x.get(k, float("nan"))
                s = f"{v:>{w}.3f}" if np.isfinite(v) else f"{'-':>{w}s}"
                if color and k in rank:
                    if rank[k][0] == i: s = G + s + RS
                    elif rank[k][1] == i: s = Y + s + RS
                elif color and hi is None:
                    s = DIM + s + RS
                line += s
            print(line)
    if color:
        print(f"\n{G}초록=최고{RS}  {Y}노랑=2위{RS}  {DIM}흐림=순위없음(바닥값·커버리지·진단){RS}")


if __name__ == "__main__":
    main()
