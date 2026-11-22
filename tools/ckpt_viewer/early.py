"""/early page: rolling valbench-hard early evals (first 10/20/30/40 windows) of the hg0 ckpts, weight-averaged ckpts
and ft4000 ckpts, + the valbench 120 table and live job progress. Rendered server-side on each request."""
import csv, glob, html, math, importlib.util, json, os, re, time

_s = importlib.util.spec_from_file_location("report_suite", "/root/project/evalwam/tools/report_suite.py")
rs = importlib.util.module_from_spec(_s); _s.loader.exec_module(rs)
H = rs.H; F = H + "_ft4000"
EARLY = [("7500", f"{H}/ckpt7500"), ("8750", f"{H}/ckpt8750"), ("avg 4000+7500", f"{H}/ckpta4000_7500"),
         ("avg 7500+8750", f"{H}/ckpta7500_8750"), ("avg 4000+7500+8750", f"{H}/ckpta4000_7500_8750"),
         ("EMA 8000", f"{H}/ckptema8000"), ("avg 7500+EMA 8000", f"{H}/ckpta7500_e8000"), ("EMA 8250", f"{H}/ckptema8250"), ("avg 7500+EMA 8250", f"{H}/ckpta7500_e8250"), ("EMA 8500", f"{H}/ckptema8500"), ("avg 7500+EMA 8500", f"{H}/ckpta7500_e8500"), ("EMA 8750", f"{H}/ckptema8750"), ("avg 7500+EMA 8750", f"{H}/ckpta7500_e8750"), ("avg 4000+EMA 8750", f"{H}/ckpta4000_e8750"), ("raw 9000", f"{H}/ckptr9000"), ("avg 7500+raw 9000", f"{H}/ckpta7500_r9000"), ("avg 7500+8750+raw 9000", f"{H}/ckpta7500_8750_r9000"), ("avg 4000+7500+raw 9000", f"{H}/ckpta4000_7500_r9000"), ("avg 4000+raw 9000", f"{H}/ckpta4000_r9000")]   # ft4000 runs deliberately left out (user: don't compare ft)
FULL = [(str(n), f"{H}/ckpt{n}/valbench5s_all120") for n in (4000, 5250, 6750, 7500, 8750)] + [("avg 4000+7500", f"{H}/ckpta4000_7500/valbench5s_all120"), ("avg 7500+8750", f"{H}/ckpta7500_8750/valbench5s_all120"), ("avg 4000+7500+8750", f"{H}/ckpta4000_7500_8750/valbench5s_all120"), ("EMA 8000", f"{H}/ckptema8000/valbench5s_all120"), ("avg 7500+EMA 8000", f"{H}/ckpta7500_e8000/valbench5s_all120"), ("EMA 8250", f"{H}/ckptema8250/valbench5s_all120"), ("avg 7500+EMA 8250", f"{H}/ckpta7500_e8250/valbench5s_all120"), ("EMA 8500", f"{H}/ckptema8500/valbench5s_all120"), ("avg 7500+EMA 8500", f"{H}/ckpta7500_e8500/valbench5s_all120"), ("EMA 8750", f"{H}/ckptema8750/valbench5s_all120"), ("avg 7500+EMA 8750", f"{H}/ckpta7500_e8750/valbench5s_all120"), ("avg 4000+EMA 8750", f"{H}/ckpta4000_e8750/valbench5s_all120"), ("raw 9000", f"{H}/ckptr9000/valbench5s_all120"), ("avg 7500+raw 9000", f"{H}/ckpta7500_r9000/valbench5s_all120"), ("avg 7500+8750+raw 9000", f"{H}/ckpta7500_8750_r9000/valbench5s_all120"), ("avg 4000+7500+raw 9000", f"{H}/ckpta4000_7500_r9000/valbench5s_all120"), ("avg 4000+raw 9000", f"{H}/ckpta4000_r9000/valbench5s_all120")]
K = [("PSNR", "vq:psnr", "^"), ("SSIM", "vq:ssim", "^"), ("LPIPS", "vq:lpips", "v"),
     ("act RotErr", "s:act_roterr", "v"), ("act ATE", "s:act_ate", "v"), ("act RPE Rot", "s:act_rpe_rot", "v"),
     ("depth AbsRel", "s:depth_C_absrel", "v"), ("depth δ&lt;1.25", "s:depth_C_delta1", "^"),
     ("cam RotErr", "s:coh_roterr", "v"), ("cam ATE", "s:coh_ate", "v"),
     ("goal success", "s:goal_da3_success", "^"), ("goal LPIPS", "gi:goal_lpips", "v"), ("goal DINO", "gi:goal_dino_sim", "^")]
KF = [("FVD 16f", "fvd:fvd16", "v"), ("FVD all", "fvd:fvdall", "v")] + K[:3] + [("MEt3R", "m3:met3r", "v")] + K[3:]

CSS = """:root{--bg:#fafafa;--fg:#1d1d1f;--mut:#6b6b70;--line:#e3e3e8;--best:#d9f2e1;--bestfg:#11612f;--2nd:#fff3cd;--card:#fff}
@media(prefers-color-scheme:dark){:root{--bg:#141416;--fg:#ececf0;--mut:#9a9aa2;--line:#2c2c31;--best:#16402a;--bestfg:#8fe0ad;--2nd:#3d3413;--card:#1c1c1f}}
body{background:var(--bg);color:var(--fg);font:14px/1.45 -apple-system,system-ui,sans-serif;margin:0;padding:20px 16px}
h1{font-size:20px;margin:0 0 4px}h2{font-size:16px;margin:28px 0 8px}.mut{color:var(--mut);font-size:12px}
table{border-collapse:collapse;background:var(--card);font-variant-numeric:tabular-nums;margin-bottom:6px}
th,td{border:1px solid var(--line);padding:4px 10px;text-align:right;white-space:nowrap}th{font-weight:600}
td:first-child,th:first-child{text-align:left}td.b{background:var(--best);color:var(--bestfg);font-weight:700}td.s{background:var(--2nd)}
.wrap{overflow-x:auto}th.grp{text-align:left;background:var(--bg);color:var(--mut);font-size:12px;padding-top:10px}.na{color:var(--mut)}"""


def _load(d):
    """report_suite.load + camera rows from cam77 (see _c77); rs.load reads <d>_lh when it exists, so does cam77."""
    try:
        x = rs.load(d) if os.path.exists(f"{d}/suite_summary.json") else None
    except Exception:
        return None
    dd = d.rstrip("/") + "_lh" if os.path.exists(d.rstrip("/") + "_lh/suite_summary.json") else d
    return _c77(x, dd)


ALL = [(g, html.escape(n), k, d) for g, n, k, d in rs.ROWS]   # report_suite order, every metric
# camera rows (action faithfulness + video-action coherence): external-table convention from cam77.py (all 77 frames,
# RPE per frame, RPE/path = mean step / path) instead of eval_suite's 20 latent key frames (RPE per 4-frame step, sum / path)
CAMK = {f"s:{p}_{k}" for p in ("act", "coh") for _, k, _ in rs.CAM}


def _c77(x, d):
    """override the camera rows of a loaded dir with its cam77.json means; drop them when cam77 is missing (never mix)."""
    if not x:
        return x
    try:
        c = json.load(open(f"{d}/cam77.json"))
    except Exception:
        c = {}
    for k in CAMK:
        v = c.get(k[2:])
        x[k] = v["mean"] if isinstance(v, dict) else None
    return x


def _table(cols, rows=None):
    data = [x for _, x in cols]
    out = ["<div class=wrap><table><tr><th>지표</th>" + "".join(f"<th>{html.escape(n)}</th>" for n, _ in cols) + "</tr>"]
    g0 = None
    for g, name, key, d in (rows or ALL):
        v = [x.get(key) if x else None for x in data]
        nums = sorted({a for a in v if isinstance(a, (int, float))}, reverse=(d == "^"))
        if not nums:
            continue
        if g != g0:
            out.append(f"<tr><th colspan={len(cols) + 1} class=grp>{html.escape(g)}</th></tr>"); g0 = g
        cells = []
        for a in v:
            if not isinstance(a, (int, float)):
                cells.append("<td class=na>–</td>"); continue
            c = " class=b" if a == nums[0] and len(nums) > 1 else (" class=s" if len(nums) > 2 and a == nums[1] else "")
            cells.append(f"<td{c}>{rs.fmt(a)}</td>")
        out.append(f"<tr><td>{name} {'↑' if d == '^' else ('→1' if d == '~1' else '↓')}</td>" + "".join(cells) + "</tr>")
    return "\n".join(out) + "</table></div>"


DIFFS = ("hard", "easy", "medium")


def _ninfer(d, df="hard"):
    return len(glob.glob(f"{d}/valbench5s/{df}/sample_*_vis.mp4"))


STAGES = [("eval_video_quality", "화질"), ("eval_camera_control", "카메라 포즈"), ("eval_channel_vs_da3", "채널"),
          ("invert_raymap", "raymap 역변환"), ("eval_depth_vs_da3", "depth 비교"), ("eval_suite", "메트릭 집계"),
          ("eval_fvd_musiq", "FVD"), ("eval_goal_image", "goal 이미지"), ("eval_met3r", "MEt3R"), ("cam77", "cam77"),
          ("invert_raymap_rollout", "포즈 복원"), ("run_suite", "120 정식 평가")]


def _stage(D, PS):
    """current step of the eval running on dir D: the tool whose --results_dir is D (DA3 with a sample counter)."""
    try:
        N = len(json.load(open(f"{D}/samples_meta.json")))
    except Exception:
        N = 0
    out = []
    for ln in PS:
        if not re.search(rf"--results_dir {re.escape(D)}(/_inc)?(/|\s|$)", ln) and not re.search(rf"cam77\.py {re.escape(D)}(\s|$)", ln):
            continue
        if "viz_da3_pointcloud" in ln:
            c = 0
            for lg in ("suite_4axis.log", "suite_da3.log"):
                try:
                    c = max(c, len(re.findall(r"sample \d+/da3", open(f"{D}/{lg}", errors="ignore").read())))
                except Exception:
                    pass
            if "--da3_on_gt" in ln:
                out.append(f"GT DA3 {max(c - N, 0)}/{N}"); continue
            c = max(c, len(glob.glob(f"{D}/viz_da3_fix/*_da3_poses.npy"))) if c <= N else N
            out.append(f"DA3 {min(c, N)}/{N}"); continue
        for key, lab in sorted(STAGES, key=lambda x: -len(x[0])):
            if key in ln:
                out.append(lab); break
    return " + ".join(dict.fromkeys(out)) or "준비"



# ---- LookOut 120 = A39 (start 4) + B46 (start 100) + C35 (turn77 val set), exclusion list applied, corrected (_lh) GT ----
V2 = f"{H}/multi/v2"
LO = [("4000", f"{H}/ckpt4000"), ("7500", f"{H}/ckpt7500"), ("avg 4000+7500", f"{H}/ckpta4000_7500"),
      ("avg 7500+8750", f"{H}/ckpta7500_8750"), ("avg 4000+7500+8750", f"{H}/ckpta4000_7500_8750"), ("EMA 8000", f"{H}/ckptema8000"), ("avg 7500+EMA 8000", f"{H}/ckpta7500_e8000"), ("EMA 8250", f"{H}/ckptema8250"), ("avg 7500+EMA 8250", f"{H}/ckpta7500_e8250"), ("EMA 8500", f"{H}/ckptema8500"), ("avg 7500+EMA 8500", f"{H}/ckpta7500_e8500"), ("EMA 8750", f"{H}/ckptema8750"), ("avg 7500+EMA 8750", f"{H}/ckpta7500_e8750"), ("avg 4000+EMA 8750", f"{H}/ckpta4000_e8750"), ("raw 9000", f"{H}/ckptr9000"), ("avg 7500+raw 9000", f"{H}/ckpta7500_r9000"), ("avg 7500+8750+raw 9000", f"{H}/ckpta7500_8750_r9000"), ("avg 4000+7500+raw 9000", f"{H}/ckpta4000_7500_r9000"), ("avg 4000+raw 9000", f"{H}/ckpta4000_r9000")]
RC4 = [("7500", f"{H}/ckpt7500"), ("avg 4000+7500", f"{H}/ckpta4000_7500"), ("avg 7500+8750", f"{H}/ckpta7500_8750"),
       ("avg 4000+7500+8750", f"{H}/ckpta4000_7500_8750"), ("EMA 8000", f"{H}/ckptema8000"), ("avg 7500+EMA 8000", f"{H}/ckpta7500_e8000"), ("EMA 8250", f"{H}/ckptema8250"), ("avg 7500+EMA 8250", f"{H}/ckpta7500_e8250"), ("EMA 8500", f"{H}/ckptema8500"), ("avg 7500+EMA 8500", f"{H}/ckpta7500_e8500")]


def _runlab(p):
    m = re.search(r"/ckpt((?:a|ema)?[\de_]+)(?:/|$)", p)
    if not m:
        return "?"
    r = m.group(1)
    if r.startswith("ema"):
        return "EMA " + r[3:]
    return "avg " + r[1:].replace("_", "+").replace("e", "EMA ") if r.startswith("a") else r


def _tasklab(p):
    rest = re.sub(r".*/ckpt(?:a|ema)?[\de_]+/?", "", p.rstrip("/")).replace("/_inc", "")
    lh = rest.endswith("_lh"); rest = rest[:-3] if lh else rest
    T = {"lookout_1c": "LookOut A 앞 26", "lookout_1c_p1": "LookOut A 뒤 25", "lookout_1c_s100": "LookOut B 앞 26",
         "lookout_1c_s100_p1": "LookOut B 뒤 25", "lookout_turn77val35": "LookOut C 35", "lookout_1c_all51": "LookOut A 평가",
         "lookout_1c_s100_all51": "LookOut B 평가", "lookout_rc4_all62": "rollout c4 평가", "valbench5s_all120": "valbench 120 합본 평가"}
    if rest in T:
        t = T[rest]
    elif rest.startswith("valbench5s/"):
        t = f"valbench {rest.split('/')[1]} 인퍼"
    elif rest.startswith("valbench5s_split/"):
        t = f"valbench {rest.split('/')[1]} 40개 평가"
    elif rest.startswith("early/"):
        a, k = rest.split("/")[1].rsplit("_", 1); t = f"{a} {k}개 평가"
    elif rest.startswith("lookout_rc4/parts/"):
        t = f"rollout c4 파트 {rest.rsplit('/', 1)[1][4:]} 인퍼 (00~15 중)"
    else:
        t = rest
    return t + (" (좌우 보정)" if lh else "")


def _rc4_state(d, PS):
    """rollout-c4 queue state of a run: parts done / running / waiting + eval state, or None when never queued."""
    C = f"{d}/lookout_rc4/parts"
    if not os.path.isdir(C):
        return None
    try:
        jobs = [l.strip() for l in open(f"{V2}/jobs.txt") if l.startswith("rc4_")]
    except Exception:
        jobs = [f"rc4_{i:02d}" for i in range(16)]
    done = clips = 0; run = []
    for j in jobs:
        n = len(glob.glob(f"{C}/{j}/rollout_??.mp4")); clips += n
        try:
            ok = "Done." in open(f"{C}/{j}.log", errors="ignore").read()
        except Exception:
            ok = False
        if ok and n == len(json.load(open(f"{V2}/jobs/{j}.json"))["val_sequences"]):
            done += 1
        elif any(re.search(re.escape(f"{C}/{j}") + r"(\s|$)", x) and "infer_rollout" in x for x in PS):
            run.append(j)
    D = f"{d}/lookout_rc4_all62"
    if os.path.exists(f"{D}_lh/suite_summary.json"):
        ev = "평가 완료"
    elif any(re.search(re.escape(D) + r"(?=[\s/]|$)", a) or "suite_long" in a and os.path.basename(d) in a for a in PS):
        ev = "평가 중 · " + _stage(D, PS)
    else:
        ev = "16개 파트 끝나면 평가"
    return done, run, len(jobs) - done - len(run), clips, ev


def _rd(p):
    try:
        return {r["sample"]: r for r in csv.DictReader(open(p))}
    except Exception:
        return {}


def _lo_set(run, which):
    """(dir, [sample names]) of LookOut set A/B/C for a run, or None when not evaluated."""
    ex = json.load(open(f"{V2}/lookout_AB_exclude17.json"))
    if which in ("A", "B"):
        d = f"{run}/lookout_1c{'' if which == 'A' else '_s100'}_all51_lh"
        if not os.path.exists(f"{d}/suite_metrics.csv"):
            return None
        return d, [f"sample_{i:02d}" for i in range(51) if i not in set(ex[which])]
    for d in (f"{run}/lookout_turn77val35_lh", f"{run}/lookout_turn77val35"):
        if os.path.exists(f"{d}/suite_metrics.csv"):
            return d, [f"sample_{i:02d}" for i in range(35)]
    d = f"{run}/lookout_turn77v2/samples_lh"
    if os.path.exists(f"{d}/suite_metrics.csv"):
        return d, [f"sample_{i:02d}" for i in json.load(open(f"{V2}/turn77_union_list.json"))["idx82"]]
    return None


def _lo_vals(d, names):
    T = {"vq": _rd(f"{d}/video_quality_metrics.csv"), "s": _rd(f"{d}/suite_metrics.csv"), "gi": _rd(f"{d}/goal_image_metrics.csv"),
         "c": _rd(f"{d}/cam77.csv")}
    _b = d[:-3] if d.endswith("_lh") else d          # MEt3R (stride 1, per sample) lives next to the generated videos, not the _lh GT copy
    T["m3"] = next((r for r in (_rd(f"{d}/met3r_s1/met3r_metrics.csv"), _rd(f"{_b}/met3r_s1/met3r_metrics.csv")) if r), {})
    out = {}
    for _, _, key, _ in ALL:
        t, col = key.split(":", 1)
        if key in CAMK:
            t = "c"
        if t not in T:
            continue
        v = []
        for n in names:
            x = T[t].get(n, {}).get(col)
            try:
                x = float(x)
            except (TypeError, ValueError):
                continue
            if math.isfinite(x):
                v.append(x)
        out[key] = v
    return out


def _lo_section():
    P = ["<h2 id=lookout>LookOut (실제 Aria) · 제외명단 제외 · 좌우 보정 GT</h2><div class=mut>A = start 4 (39), B = start 100 (46), "
         "C = turn77 val set (35). 샘플별 값의 평균 · FVD는 같은 샘플 집합 전체로 계산 · 카메라 지표 = cam77 (77프레임) · MEt3R = stride 1 (인접 프레임) 샘플별 평균 · 한 세트라도 없는 지표는 합본에서 –</div>"]
    prog = []
    for n, d in LO:
        if n.startswith("avg"):
            c = [len(glob.glob(f"{d}/lookout_1c/rollout_??.mp4")), len(glob.glob(f"{d}/lookout_1c_s100/rollout_??.mp4")),
                 len(glob.glob(f"{d}/lookout_turn77val35/sample_*_vis.mp4"))]
            if any(c) or os.path.isdir(f"{d}/lookout_1c"):
                prog.append(f"{n}: 인퍼 A {c[0]}/51 · B {c[1]}/51 · C {c[2]}/35")
    if prog:
        P.append("<div class=mut>" + " &nbsp;|&nbsp; ".join(prog) + "</div>")
    rows = ALL                                       # MEt3R included: stride-1 per-sample means over the filtered samples
    sets = {n: {w: _lo_set(d, w) for w in "ABC"} for n, d in LO}
    vals = {n: {w: (_lo_vals(*sets[n][w]) if sets[n][w] else None) for w in "ABC"} for n, _ in LO}
    for w, lab in (("ABC", "A+B+C 120"), ("A", "A · start 4 · 39"), ("B", "B · start 100 · 46"), ("C", "C · turn77 val set · 35")):
        cols = []
        for n, _ in LO:
            if all(vals[n][x] for x in w):
                cols.append((n, {k: (sum((vals[n][x][k] for x in w), []) if all(vals[n][x][k] for x in w) else [])
                                 for k in vals[n][w[0]]}))   # never pool a metric that one set lacks
        cols = [(n, {k: (sum(v) / len(v) if v else None) for k, v in x.items()}) for n, x in cols]
        for n, x in cols:   # set-level FVD over the same filtered samples (multi/lo_fvd.py -> lookout_fvd/<set>/fvd_musiq.json)
            try:
                f = json.load(open(f"{dict(LO)[n]}/lookout_fvd/{w}/fvd_musiq.json"))
                x["fvd:fvd16"], x["fvd:fvdall"] = f["fvd16"], f.get(f"fvd{f.get('frames')}")
            except Exception:
                pass
        if cols:
            P.append(f"<h2>LookOut {lab}</h2>" + _table(cols, rows))
    return "\n".join(P)

def page():
    t = time.strftime("%m-%d %H:%M")
    P = [f"<!doctype html><html><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
         f"<meta http-equiv=refresh content=60><title>Early evals</title><style>{CSS}</style></head><body>",
         f"<h1>중간 평가 · valbench</h1><div class=mut>{t} 기준 · 60초마다 자동 새로고침 · 초록 = 행별 최고, 노랑 = 2위 · "
         "같은 샘플 prefix(sample_00..k-1)끼리만 비교 · 지표는 report_suite 순서 전체, 측정 안 된 행은 생략 (중간 평가는 MEt3R 미측정)</div>"]
    # progress: which eval dirs are being evaluated right now (and at which step / sample) vs queued on a flock
    import subprocess
    PP = [l.strip().split(None, 1) for l in subprocess.run(["ps", "-eo", "pid,args"], capture_output=True, text=True).stdout.splitlines()[1:]]
    PP = [(int(x[0]), x[1]) for x in PP if len(x) == 2]
    PS = [a for _, a in PP]
    GMAP = {}
    try:
        idx = dict(l.split(", ") for l in subprocess.run(["nvidia-smi", "--query-gpu=uuid,index", "--format=csv,noheader"],
                                                         capture_output=True, text=True).stdout.splitlines())
        for l in subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"],
                                capture_output=True, text=True).stdout.splitlines():
            q, u = l.split(", "); GMAP[int(q)] = idx.get(u, "?")
    except Exception:
        pass
    def gpu_of(*pats):
        """GPU index(es) of the processes whose command line contains one of pats (a path as a whole token)."""
        gs = set()
        for pid, a in PP:
            if not any(re.search(re.escape(t) + r"(?=[\s/]|$)", a) for t in pats):
                continue
            g = GMAP.get(pid)
            if g is None:
                try:
                    env = open(f"/proc/{pid}/environ", "rb").read().split(b"\0")
                    g = next((e.split(b"=", 1)[1].decode() for e in env if e.startswith(b"CUDA_VISIBLE_DEVICES=")), None) or \
                        next((e.split(b"=", 1)[1].decode() for e in env if e.startswith(b"GPU=")), None)
                except Exception:
                    g = None
            if g is not None and g.isdigit():
                gs.add(g)
        return (" · GPU" + ",".join(sorted(gs))) if gs else ""
    # ---- what is running right now, per GPU (inference + eval python processes) ----
    ET = {}
    for l in subprocess.run(["ps", "-eo", "pid,etimes"], capture_output=True, text=True).stdout.splitlines()[1:]:
        try:
            a_, b_ = l.split(); ET[int(a_)] = int(b_)
        except ValueError:
            pass
    def _pgpu(pid):
        g = GMAP.get(pid)
        if g is None:
            try:
                env = open(f"/proc/{pid}/environ", "rb").read().split(b"\0")
                g = next((e.split(b"=", 1)[1].decode() for e in env if e.startswith(b"CUDA_VISIBLE_DEVICES=")), "?")
            except Exception:
                g = "?"
        return g
    def _arg(a, k):
        m = re.search(rf"{k}[ =](\S+)", a); return m.group(1) if m else None
    JOBS = {}
    for pid, a in PP:
        if not re.match(r"(\S*/)?python\S*\s", a) or "/ckpt" not in a:
            continue
        if "recon_grow_rollout" in a:   # CPU job: one row per viz dir with the clip count
            V = _arg(a, "--roll_dir")
            if V and V not in JOBS:
                k = len(glob.glob(f"{V}/recon/grow_rollout_s??.mp4")); n = len(glob.glob(f"{V}/rollout_??.mp4"))
                JOBS[V] = ("CPU", _runlab(V), "rollout c4 점진 확장 recon 영상", f"{k}/{n}개", ET.get(pid, 0), "시각화")
            continue
        if re.search(r"infer_(cosmos_lora|rollout)\.py", a):
            O = _arg(a, "--output_dir")
            if not O:
                continue
            n = _arg(a, "--num_samples") or (40 if "/valbench5s/" in O else "?")
            k = len(glob.glob(f"{O}/rollout_??.mp4")) if "infer_rollout" in a else len(glob.glob(f"{O}/sample_*_vis.mp4"))
            JOBS[O] = (_pgpu(pid), _runlab(O), _tasklab(O), f"{k}/{n}개", ET.get(pid, 0), "인퍼")
        else:
            m = re.search(r"(?:--results_dir|--roll_dir) (\S+)|(?:cam77|lh|lookout_lh_reeval)\.py (?:\S+lookout\S+_cv )?(\S+/ckpt\S+)", a)
            if not m:
                continue
            D = (m.group(1) or m.group(2)).rstrip("/").replace("/_inc", "")
            if D in JOBS:
                continue
            st = _stage(D, PS)
            if st == "준비":
                st = "좌우 보정 재평가" if re.search(r"lh\.py|lookout_lh_reeval", a) else (next((lab for key, lab in sorted(STAGES, key=lambda x: -len(x[0])) if key in a), "평가"))
            JOBS[D] = (_pgpu(pid), _runlab(D), _tasklab(D), st, ET.get(pid, 0), "평가")
    try:
        fr = {l.split(", ")[0]: l.split(", ")[1:] for l in subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
              "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.splitlines()}
    except Exception:
        fr = {}
    rows_ = sorted(JOBS.values(), key=lambda x: (x[0], -x[4]))
    P.append("<h2>지금 돌아가는 작업</h2><div class=mut>GPU별 실행 중인 인퍼·평가 프로세스 (ps + nvidia-smi 기준, 새로고침마다 갱신)</div>"
             "<div class=wrap><table><tr><th>GPU</th><th>메모리 · 사용률</th><th>run</th><th>작업</th><th>종류</th><th>진행</th><th>경과</th></tr>")
    for g in sorted(fr):
        js = [r for r in rows_ if r[0] == g]
        mem = f"{int(fr[g][0]) // 1024}/{int(fr[g][1]) // 1024} GB · {fr[g][2]}%"
        if not js:
            P.append(f"<tr><td>{g}</td><td>{mem}</td><td class=na colspan=5>비어 있음</td></tr>"); continue
        for i, (_, rn, tk, pr, et, kind) in enumerate(js):
            P.append(f"<tr><td>{g if i == 0 else ''}</td><td>{mem if i == 0 else ''}</td><td>{html.escape(rn)}</td><td>{html.escape(tk)}</td>"
                     f"<td>{kind}</td><td class=s>{html.escape(pr)}</td><td>{et // 3600}h {et % 3600 // 60:02d}m</td></tr>")
    for r in rows_:
        if r[0] not in fr:
            P.append(f"<tr><td>{html.escape(r[0])}</td><td></td><td>{html.escape(r[1])}</td><td>{html.escape(r[2])}</td><td>{r[5]}</td>"
                     f"<td class=s>{html.escape(r[3])}</td><td>{r[4] // 3600}h {r[4] % 3600 // 60:02d}m</td></tr>")
    P.append("</table></div>")
    # queued (waiting for a GPU slot): LookOut sets / rc4 parts not started yet
    QL = []
    qalive = " ".join(PS)
    for n, d in LO:
        if not n.startswith("avg"):
            continue
        b = os.path.basename(d)
        if f"lo120_queue.sh {b[4:]}" in qalive or f"lo_c35_goalmatch.sh {b[4:]}" in qalive:
            for w, subs in (("A", ("lookout_1c", "lookout_1c_p1")), ("B", ("lookout_1c_s100", "lookout_1c_s100_p1")), ("C", ("lookout_turn77val35",))):
                left = [x for x in subs if not os.path.isdir(f"{d}/{x}")]
                if left:
                    QL.append((n, f"LookOut {w} " + ", ".join(_tasklab(f"{d}/{x}").split(" ", 2)[-1] for x in left), "GPU 메모리 32GB 이상 비면 시작 (3분 간격)"))
    for n, d in RC4:
        st = _rc4_state(d, PS)
        if st and st[2]:
            QL.append((n, f"rollout c4 파트 {st[2]}개", "레인이 앞 파트 끝내면 다음 파트 (GPU 32GB 이상 빈 곳)"))
    for n, d in EARLY:
        if n.startswith("avg") and os.path.isdir(f"{d}/valbench5s") and not os.path.exists(f"{d}/valbench5s_all120/suite_summary.json"):
            left = [df for df in DIFFS if not os.path.exists(f"{d}/valbench5s_split/{df}/SPLIT_EVAL_DONE")]
            if left:
                QL.append((n, f"valbench {'·'.join(left)} 40개 평가 → 120 합본", "각 난이도 인퍼 끝나면 같은 GPU에서 바로"))
    if QL:
        P.append("<h2>대기열</h2><div class=wrap><table><tr><th>run</th><th>작업</th><th>시작 조건</th></tr>"
                 + "".join(f"<tr><td>{html.escape(a)}</td><td>{html.escape(b)}</td><td>{html.escape(c)}</td></tr>" for a, b, c in QL)
                 + "</table></div>")
    RUN, WAIT = set(), set()
    for ln in PS:
        m = re.search(r"(?:suite_n(?:_g)?\.sh|inc_prep\.py|eval_vb\.sh \d+ \w+) (\S+/(?:early/\w+|valbench5s_split/\w+))", ln)
        if m:
            (WAIT if ln.startswith("flock") else RUN).add(m.group(1).rstrip("/"))
    WAIT -= RUN
    P.append("<h2>진행 상황</h2><div class=wrap><table><tr><th>run</th><th>난이도</th><th>인퍼</th>"
             + "".join(f"<th>{k}개</th>" for k in (10, 20, 30, 40)) + "</tr>")
    def cell(e):
        if os.path.exists(f"{e}/cam77.json") or (e.endswith("valbench5s_split/" + e.rsplit("/", 1)[-1]) and os.path.exists(f"{e}/SPLIT_EVAL_DONE")):
            return "<td class=b>✓</td>"
        if e in RUN:
            return f"<td class=s>평가 중 · {html.escape(_stage(e, PS))}{gpu_of(e)}</td>"
        if e in WAIT:
            return "<td>대기 (잠금)</td>"
        if os.path.isdir(e):
            return "<td>예정</td>"
        return None
    for n, d in EARLY:
        for df in DIFFS:
            if df != "hard" and n in ("7500", "8750"):
                continue
            if df != "hard" and not os.path.isdir(f"{d}/valbench5s/{df}") and not os.path.isdir(f"{d}/early/{df}_10"):
                continue
            cells = []
            for k in (10, 20, 30, 40):
                c = cell(f"{d}/early/{df}_{k}")
                if k == 40 and (c is None or c.startswith("<td>대기")):
                    c2 = cell(f"{d}/valbench5s_split/{df}") if os.path.isdir(f"{d}/valbench5s_split/{df}") else None
                    if os.path.exists(f"{d}/valbench5s_split/{df}/suite_summary.json") and n in ("7500", "8750"):
                        c2 = "<td class=b>✓</td>"
                    c = c2 or c
                cells.append(c or "<td class=na>–</td>")
            gi = gpu_of(f"{d}/valbench5s/{df}")
            P.append(f"<tr><td>{n}</td><td>{df}</td><td{' class=s' if gi else ''}>{_ninfer(d, df)}/40{gi}</td>" + "".join(cells) + "</tr>")
    # LookOut sets of the avg runs: inference count + GPU, eval state (one cell)
    for n, d in LO:
        if not n.startswith("avg") or not any(os.path.isdir(f"{d}/{x}") for x in ("lookout_1c", "lookout_1c_s100", "lookout_turn77val35")):
            continue
        for w, sub, tot in (("A", "lookout_1c", 51), ("B", "lookout_1c_s100", 51), ("C", "lookout_turn77val35", 35)):
            src = f"{d}/{sub}"
            ni = len(glob.glob(f"{src}/rollout_??.mp4")) if w != "C" else len(glob.glob(f"{src}/sample_*_vis.mp4"))
            gi = gpu_of(src) if any((re.search(r"infer_|run_lookout\.sh", a) and re.search(re.escape(src) + r"(\s|$)", a)) for a in PS) else ""
            D = f"{src}_all51" if w != "C" else src
            lh = f"{D}_lh"
            run_ev = [a for a in PS if not re.search(r"infer_|run_lookout\.sh|run_mode\.sh|lo120", a) and (re.search(re.escape(D) + r"(?=[\s/]|$)", a) or re.search(re.escape(src) + r"(\s|$)", a)
                      or (w != "C" and "suite_lookout1c" in a and a.endswith(f"{os.path.basename(d)} {sub}")))]
            if os.path.exists(f"{lh}/suite_metrics.csv") and not run_ev:
                ev = "<td class=b colspan=4>✓ 평가 완료</td>"
            elif run_ev:
                st = _stage(D, PS)
                if st == "준비" and any("invert_raymap_rollout" in a for a in run_ev):
                    st = "포즈 복원"
                if any("lookout_lh_reeval" in a or "lh.py" in a for a in run_ev):
                    st = "좌우 보정 재평가"
                ev = f"<td class=s colspan=4>평가 중 · {html.escape(st)}{gpu_of(D, src)}</td>"
            elif ni and gi:
                ev = "<td colspan=4>인퍼 끝나면 평가</td>"
            else:
                ev = "<td class=na colspan=4>–</td>"
            P.append(f"<tr><td>{n}</td><td>LookOut {w}</td><td{' class=s' if gi else ''}>{ni}/{tot}{gi}</td>{ev}</tr>")
    for n, d in RC4:
        st = _rc4_state(d, PS)
        if not st or n == "7500":
            continue
        done, run, wait, clips, ev = st
        gi = gpu_of(*[f"{d}/lookout_rc4/parts/{j}" for j in run]) if run else ""
        P.append(f"<tr><td>{n}</td><td>rollout c4</td><td{' class=s' if run else ''}>{clips}/62{gi}</td>"
                 f"<td colspan=4{' class=b' if ev == '평가 완료' else (' class=s' if ev.startswith('평가 중') else '')}>"
                 f"{'✓ ' if ev == '평가 완료' else ''}{html.escape(ev)} · 파트 완료 {done}/16"
                 f"{(' · 실행 중 ' + ', '.join(j[4:] for j in run)) if run else ''}{(' · 대기 ' + str(wait)) if wait else ''}</td></tr>")
    P.append("</table></div><div class=mut>rollout c4 = 62개 클립 × 4청크 (305프레임), 7500과 같은 16개 파트·seed → goal 프레임 동일</div>")
    # ---- evaluation order (what runs now, what the avg lanes take next, queued jobs) ----
    def pref(S):
        k = 0
        while os.path.exists(f"{S}/sample_{k:02d}_vis.mp4"):
            k += 1
        return k
    L = []
    for e in sorted(RUN):
        nm = next((n for n, d in EARLY if e.startswith(d + "/")), "ft (원래 대기열)")
        L.append(("지금", f"{nm} · {e.rsplit('/', 2)[-2] if 'split' in e else ''}{e.rsplit('/', 1)[-1].replace('_', ' ')}개".replace("valbench5s_split", "40개 정식 ").replace("hard개", "hard 40개 정식").replace("easy개", "easy 40개 정식").replace("medium개", "medium 40개 정식"),
                  f"{_stage(e, PS)}{gpu_of(e)}"))
    nxt = []
    for n, d in EARLY:
        if not n.startswith("avg"):
            continue
        for df in DIFFS:
            S = f"{d}/valbench5s/{df}"
            if not os.path.isdir(S):
                continue
            p = pref(S); busy = False
            for k in (10, 20, 30, 40):
                if k == 40 and df != "hard":
                    continue
                e = f"{d}/early/{df}_{k}"
                if e in RUN or os.path.isdir(f"{e}/.claim"):
                    busy = True; continue
                if os.path.exists(f"{e}/cam77.json"):
                    continue
                if busy:
                    nxt.append((f"{n} · {df} {k}개", "앞 구간이 끝나면 그 결과를 이어받아 새 샘플만"))
                elif p >= k:
                    nxt.append((f"{n} · {df} {k}개", "avg 줄이 다음에 바로 잡음"))
                else:
                    nxt.append((f"{n} · {df} {k}개", f"인퍼 {k}개 도달 후 (지금 {p})"))
                    break
    for i, (a_, b_) in enumerate(nxt, 1):
        L.append((f"다음 {i}", a_, b_))
    for n, d in EARLY:
        if not n.startswith("avg") or not os.path.isdir(f"{d}/valbench5s/easy"):
            continue
        if os.path.exists(f"{d}/VALBENCH_EVAL_DONE"):
            st = "완료"
        else:
            left = [df for df in DIFFS if pref(f"{d}/valbench5s/{df}") < 40]
            st = (f"인퍼 남음: {', '.join(left)}" if left else "진행 중") + " → hard·easy·medium 40개 정식(이어받기) → 120 합본 + cam77 + MEt3R"
        L.append(("정식", f"{n} · valbench 120", st))
    for n, d in LO:
        if not n.startswith("avg") or not os.path.isdir(f"{d}/lookout_1c"):
            continue
        L.append(("LookOut", f"{n} · A·B·C", "각 세트 인퍼(+분할 합치기) 끝나는 대로 → suite → 좌우 보정 재평가 → cam77 (A·B: GPU2·0, C: GPU3)"))
    W = []
    for pid, a_ in PP:
        m = re.match(r"flock \S+/\.early\.lock bash \S+suite_n\.sh (\S+)", a_)
        if m:
            try:
                W.append((os.stat(f"/proc/{pid}").st_mtime, m.group(1)))
            except Exception:
                pass
    for i, (_, e) in enumerate(sorted(W), 1):
        nm = next((n for n, d in EARLY if e.startswith(d + "/")), "ft/기타")
        L.append((f"원래 대기열 {i}", f"{nm} · {e.rsplit('/', 1)[-1].replace('_', ' ')}개", "다른 세션 스크립트 · 한 번에 하나 · 이미 끝난 건 집계만 다시"))
    # completed evals, newest first (cam77.json / suite_summary.json time) = the record of what was evaluated when
    done = []
    for n, d in EARLY + [x for x in LO if x[0].startswith("avg")]:
        if n in ("7500", "8750"):
            continue
        for e in glob.glob(f"{d}/early/*") + glob.glob(f"{d}/valbench5s_split/*") + [f"{d}/valbench5s_all120"] + \
                 [f"{d}/{x}" for x in ("lookout_1c_all51_lh", "lookout_1c_s100_all51_lh", "lookout_turn77val35")]:
            f = f"{e}/cam77.json" if os.path.exists(f"{e}/cam77.json") else f"{e}/suite_summary.json"
            if os.path.exists(f) and not os.path.exists(f"{e}/.claim"):
                lab = e.rsplit("/", 1)[-1]
                lab = {"valbench5s_all120": "valbench 120 정식"}.get(lab, lab.replace("_lh", "").replace("_", " "))
                if "/valbench5s_split/" in e:
                    lab = f"{lab} 40개 정식"
                elif "/early/" in e:
                    lab += "개"
                done.append((os.path.getmtime(f), f"{n} · {lab}"))
    done = sorted(set(done), reverse=True)
    D_ = "".join(f"<tr><td>{time.strftime('%m-%d %H:%M', time.localtime(t))}</td><td>{html.escape(x)}</td></tr>" for t, x in done[:30])
    P.append("<h2>평가 순서</h2><div class=mut>avg 평가 줄 = GPU0~3에 하나씩, avg 4000+7500 → avg 7500+8750, hard → easy → medium, "
             "10 → 20 → 30 (→ hard 40) 순. 같은 run·난이도의 작은 구간이 돌고 있으면 큰 구간은 기다렸다가 그 결과를 이어받아 새 샘플만 계산.</div>"
             "<div class=wrap><table><tr><th>순서</th><th>평가</th><th>상태</th></tr>"
             + "".join(f"<tr><td>{html.escape(x)}</td><td>{html.escape(y)}</td><td>{html.escape(z)}</td></tr>" for x, y, z in L)
             + "</table></div>"
             + ("<h2>평가 완료 기록 (최근 30, 최신 순)</h2><div class=wrap><table><tr><th>완료 시각</th><th>평가</th></tr>" + D_ + "</table></div>" if D_ else ""))
    # pooled difficulty combos: same prefix per difficulty for every column, sample-weighted mean of the per-dir means
    def get(d, df, k):
        x = _load(f"{d}/early/{df}_{k}")
        return x if x is not None or k != 40 else _load(f"{d}/valbench5s_split/{df}")
    def maxk(d, df):
        return next((k for k in (40, 30, 20, 10) if get(d, df, k) is not None), 0)
    shown = set()
    for combo in (("easy", "medium", "hard"), ("easy", "hard"), ("medium", "hard"), ("easy", "medium")):
        runs = [(n, d) for n, d in EARLY if all(get(d, df, 40) is not None for df in combo)]
        names = {n for n, _ in runs}
        if not any(n not in ("7500", "8750") for n in names) or any(names <= s0 for s0 in shown):
            continue
        ks = {df: (40 if all(get(d, df, 40) is not None for _, d in runs) else 0) for df in combo}   # final 40-per-difficulty only (no 10/20/30 prefixes)
        if not all(ks.values()):
            continue
        cols = []
        for n, d in runs:
            parts = [(ks[df], get(d, df, ks[df])) for df in combo]
            keys = set.intersection(*[set(x) for _, x in parts])
            pooled = {}
            for key in keys:
                if key.startswith(("fvd:", "m3:")):
                    continue
                vals = [(w, x[key]) for w, x in parts]
                if all(isinstance(v, (int, float)) and not isinstance(v, bool) for _, v in vals):
                    pooled[key] = sum(w * v for w, v in vals) / sum(w for w, _ in vals)
            cols.append((n, pooled))
        shown.add(frozenset(names))
        lab = " + ".join(f"{df} {ks[df]}" for df in combo)
        P.append(f"<h2>조합 · {lab} = {sum(ks.values())}개</h2><div class=mut>난이도별 같은 샘플 범위를 샘플 수로 가중평균 "
                 "(FVD·MEt3R은 샘플 단위 평균이 아니라 제외)</div>" + _table(cols, [r for r in ALL if not r[2].startswith(("m3:", "fvd:"))]))
    # early tables per difficulty, newest prefix first; a table only when a non-reference run (beyond 7500/8750) has it
    for df in DIFFS:
        for k in (40,):                     # final 40 only; 10/20/30 prefixes are not shown (user)
            cols = []
            for n, d in EARLY:
                x = _load(f"{d}/early/{df}_{k}")
                if x is None and k == 40:
                    x = _load(f"{d}/valbench5s_split/{df}")
                if x is not None:
                    cols.append((n, x))
            if cols and (df == "hard" or any(n not in ("7500", "8750") for n, _ in cols)):
                P.append(f"<h2>{df} {k}개 (정식)</h2>" + _table(cols, [r for r in ALL if not r[2].startswith("m3:")]))
    P.append(_lo_section())
    cols = [(n, _load(f"{d}/lookout_rc4_all62_lh")) for n, d in RC4]
    cols = [(n, x) for n, x in cols if x]
    if len(cols) > 1:
        P.append("<h2>rollout c4 · LookOut 62개 × 4청크 (305프레임) · 좌우 보정 GT</h2><div class=mut>77프레임 벤치와 직접 비교 불가</div>" + _table(cols))
    cols = [(n, _load(d)) for n, d in FULL]
    cols = [(n, x) for n, x in cols if x]
    P.append("<h2>valbench5s 전체 120개 (easy+medium+hard)</h2><div class=mut>action faithfulness · coherence 카메라 지표 = cam77 "
             "(77프레임 전체, RPE는 프레임당, RPE/path = 프레임당 평균 ÷ 경로 길이). cam77 없는 열은 –</div>" + _table(cols))
    P.append("</body></html>")
    return "\n".join(P).encode()
