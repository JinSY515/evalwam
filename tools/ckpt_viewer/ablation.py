"""/ablation page: ablation runs grouped by AXIS (what is varied), each axis rendered with the same benches and metrics
as /early (valbench5s 120 + per difficulty 40, LookOut A+B+C 120 / A / B / C with the exclusion list, cam77 camera rows,
MEt3R s1, set-level FVD). Inside an axis every comparison GROUP has a base column; each other column gets a Δ vs that base
(green = better, red = worse, by the metric's direction).
  axis warp: L_vcons matching (direct vs ours = residual)
  axis cam : camera representation -- ladder abl1 (I, compact action token) / abl2 (II, raymap tokens, no goal aux) vs ours,
             and the ray-fusion pair A (channel concat, shared RoPE) vs B (separate tokens, independent RoPE), which
             are both Cosmos-base-from-scratch at 1000 steps and are compared with each other only (Δ = B − A).
New ablations: add to ABL and to an axis group. Rendered server-side on each request (server.py reloads early + this module)."""
import glob, html, json, os, time

import early

H = early.H
AB = "/root/project/egowam/results_syn/abl"
OURS = ("ours · avg 7500+8750", f"{H}/ckpta7500_8750")
OURS_A4 = ("ours avg 4000+7500 · 메인 run", f"{H}/ckpta4000_7500")   # same main run, other averaged ckpt (column, not a registry run)
ABL = {
    "direct": {"col": "direct matching · ck 2000", "dir": f"{H}/ckptdirect2000", "split": True,
               "what": "L_vcons direct matching |x̂0_t − W(x̂0_s)| (GT floor 미차감). ours = residual matching",
               "ckpt": "goalprog_mode_real6_nomr_hg0_grad0.5_direct/checkpoint-2000 (학습 2061 step에서 중단)",
               "pipe": "multi/direct2000_full.sh · tmux full_direct2000 · 로그 multi/full_direct2000.out"},
    "IV": {"col": "abl4 · L_vcons 없음 · ck 6000", "dir": f"{AB}/IV_cont5k/ckpt6000",
           "what": "warping loss(L_vcons) 자체가 없음: raymap 토큰 + pred_raymap + goal aux + DPT depth (ladder IV), mode embed 없음",
           "ckpt": "goalprog_dl3dv_camfix_d10_dpt_cont5k/checkpoint-1000 (= d10_dpt 5000 + 1000 = 6000 step)",
           "pipe": "valbench: IV_cont5k/run_all.sh (11-09) · LookOut: abl/lo120_iv.sh · tmux lo120_iv · 로그 abl/lo120abl_IV.out"},
    "I": {"col": "abl1 · action 토큰 · ck 3000", "dir": f"{AB}/I_jointfm/ckpt3000",
          "what": "카메라 없이 압축 action 토큰을 joint FM으로 예측 (raymap 없음). 카메라 = action 적분",
          "ckpt": "abl_interpolate_hist_goal_predaction/checkpoint-3000 (clwm515/wam)",
          "pipe": "valbench: evalwam run_abl.sh I · LookOut: infer_rollout --pred_action (joint_denoise, 1청크) + a0 좌우 보정 · "
                  "abl/I_lookout.sh (1샘플 검증 → lo120_abl.sh I) · tmux ablI_lookout"},
    "II": {"col": "abl2 · raymap 토큰 · ck 3000", "dir": f"{AB}/II_nogoalaux/ckpt3000",
           "what": "raymap 토큰 (ds 2) + pred_raymap, goal aux · depth · mode embed 없음, 옛 raymap convention",
           "ckpt": "goalprog_appearance_nogoalaux/checkpoint-3000",
           "pipe": "valbench: evalwam run_abl.sh II · LookOut: abl/lo120_abl.sh II · tmux lo120abl_II · 로그 abl/lo120abl_II.out"},
    "RA": {"col": "A · ray channel · ck 1000", "dir": f"{AB}/RA_raychannel/ckpt1000",
           "what": "raymap을 비디오 latent에 채널 concat (ds 1, 위치당 토큰 1개) + RoPE 공유",
           "ckpt": "clwm515/abl abl_ray_channel/checkpoint-1000 (Cosmos base에서 scratch, aux loss 없음)",
           "pipe": "코드 c7ab179 (worktree egowam_rayabl) · abl/vb_rayabl.sh RA + abl/lo120_abl.sh RA · tmux rayabl_RA"},
    "RB": {"col": "B · ray token · ck 1000", "dir": f"{AB}/RB_raytoken/ckpt1000",
           "what": "raymap 별도 토큰 (ds 2, 토큰 +25%) + 독립 3D RoPE (비디오/hist/goal과 겹치지 않는 좌표)",
           "ckpt": "clwm515/abl abl_ray_token/checkpoint-1000 (Cosmos base에서 scratch, aux loss 없음)",
           "pipe": "코드 c7ab179 (worktree egowam_rayabl) · abl/vb_rayabl.sh RB + abl/lo120_abl.sh RB · tmux rayabl_RB"},
    "RA1500": {"col": "A · ray channel · ck 1500", "dir": f"{AB}/RA_raychannel/ckpt1500",
               "what": "A와 같은 run (raymap 채널 concat, ds 1, RoPE 공유), 1500 step",
               "ckpt": "clwm515/abl abl_ray_channel/checkpoint-1500 (Cosmos base에서 scratch, aux loss 없음)",
               "pipe": "코드 c7ab179 · abl/run_ray1500.sh RA (= CK=1500 vb_rayabl.sh + lo120_abl.sh + lo_post_abl.sh) · tmux ray1500_RA"},
    "RB1500": {"col": "B · ray token · ck 1500", "dir": f"{AB}/RB_raytoken/ckpt1500",
               "what": "B와 같은 run (raymap 별도 토큰, ds 2, 독립 3D RoPE), 1500 step",
               "ckpt": "clwm515/abl abl_ray_token/checkpoint-1500 (Cosmos base에서 scratch, aux loss 없음)",
               "pipe": "코드 c7ab179 · abl/run_ray1500.sh RB (= CK=1500 vb_rayabl.sh + lo120_abl.sh + lo_post_abl.sh) · tmux ray1500_RB"},
    "goalstyle": {"col": "ours · goal 외형 변화 (mode 3)", "dir": f"{H}/ckpta7500_8750_goalstyle", "lookout": False,
                  "lo_note": "미실행 (synthetic만 요청)",
                  "what": "같은 avg 7500+8750, 같은 창/seed/goal 프레임. goal 이미지만 telestyle 편집본(Qwen-Image-Edit-2509, "
                          "autumn·spring·photoreal·golden_hour·dawn 중 창마다 1개)으로 바꾸고 mode 3 (카메라 예측 + goal 변형). "
                          "평가 GT·goal 지표는 원본 goal 프레임 기준",
                  "ckpt": "avg_eval/checkpoint-a7500_8750 (학습 시 goal_swap_prob 0.5)",
                  "pipe": "multi/goalstyle_vb.sh · tmux goalstyle_vb · 로그 multi/goalstyle_vb.out · 편집본 ckpta7500_8750_goalstyle/edit/styled.json"},
    "camcond": {"col": "ours · 카메라 조건 (mode 0)", "dir": f"{H}/ckpta7500_8750_camcond",
                "what": "같은 avg 7500+8750 모델에 GT 카메라(raymap)를 입력으로 줌 (mode_idx 0 = cond+faithful). "
                        "ours 열 = 카메라를 스스로 예측 (mode_idx 2). 창·seed·goal 동일",
                "ckpt": "avg_eval/checkpoint-a7500_8750 (학습 시 raymap_pred_prob 0.5 → mode 0도 학습됨)",
                "pipe": "multi/camcond_vb.sh · tmux camcond_vb · 로그 multi/camcond_vb.out · "
                        "real(LookOut) = abl/lo120_camcond.sh (--cond_mirror_lr: 조건 카메라만 좌우 보정, GT 평가는 _lh) · tmux camcond_lo"},
}
# same-run reference near the ablation's step (valbench only: early main ckpts have no LookOut 120)
STEPREF = ("참고 · 메인 ck 2250", f"{H}/ckpt2250")
AXES = [
    {"id": "warp", "title": "축 1 · warping loss (L_vcons 매칭 방식)",
     "groups": [{"base": OURS, "runs": ["direct", "IV"], "ref": STEPREF, "scale": ["direct"],
                 "note": "direct는 2000 step(학습 중단), ours는 7500·8750 평균이라 학습량이 다름. 같은 run의 2250 step 결과를 "
                         "valbench 표에 참고 열로 붙임 (메인 run 초기 ckpt는 LookOut 120 평가가 없음). abl4 = L_vcons 없이 학습한 run "
                         "(ladder IV, 6000 step, mode embed 없음): warping loss가 없을 때 대비. "
                         "'direct 추정' 열 = 지표별로 ours × (direct 2000 / 메인 2250): 같은 학습량에서 direct가 ours 대비 "
                         "얼마나 차이 날지 대충 환산한 값 (valbench만, 메인 2250은 LookOut 120 평가가 없음)."}]},
    {"id": "cam", "title": "축 2 · 카메라 표현",
     "groups": [{"base": OURS, "runs": [OURS_A4, "I", "II"], "sub": "ladder: abl1 · abl2 vs ours",
                 "note": "abl1·abl2는 학습 step·데이터·보조 loss가 ours와 다름 (ladder 단계). abl1은 카메라 출력이 action 토큰이라 "
                         "카메라 지표는 action 적분 경로 기준. ours avg 4000+7500 = 같은 메인 run의 다른 평균 ckpt (Δ = 열 − avg 7500+8750)."},
                {"base": "RA", "runs": ["RB", "II", "I"], "sub": "카메라 주입 방식: A (ray channel) vs B (ray token) vs abl2 (raymap 토큰) vs abl1 (compact action 토큰)",
                 "note": "A·B는 raymap 주입 방식만 다르고 나머지 동일 (Cosmos base에서 scratch, 1000 step, mode_idx 2). "
                         "ours와 학습량이 달라 A·B끼리 비교: 기준 = A, Δ = 열 − A. abl2(raymap 토큰 ds 2)를 함께 붙임 — "
                         "abl1(compact action 토큰, raymap 없음, 카메라 = action 적분)도 붙임. "
                         "단 abl1·abl2는 3000 step·mode embed 없음 (abl2는 옛 raymap convention)이라 A·B(1000 step)와 학습량·설정이 다름."},
                {"base": "RA1500", "runs": ["RB1500", "RA", "RB"], "sub": "1500 step: A (ray channel) vs B (ray token), + 1000 step 참고", "partial": True,
                 "note": "A·B 모두 1500 step (같은 데이터·설정, raymap 주입 방식만 다름). 기준 = A 1500, Δ = 열 − A 1500. "
                         "1000 step 열은 학습 진행 참고 (A 1000의 Δ = A가 1000 → 1500에서 변한 양의 반대 부호)."},
                {"base": "RA1500", "runs": ["RB1500", "II", "I"], "sub": "카메라 주입 방식 @1500: A (ray channel) vs B (ray token) vs abl2 (raymap 토큰) vs abl1 (compact action 토큰)",
                 "note": "위 1000 step 묶음과 같은 구성, A·B만 1500 step. 기준 = A 1500, Δ = 열 − A 1500. "
                         "abl1·abl2는 3000 step·mode embed 없음 (abl2는 옛 raymap convention), abl1은 raymap 없이 카메라 = action 적분 → "
                         "학습량·설정이 A·B(1500 step)와 다름."}]},
    {"id": "goalapp", "title": "축 4 · goal 외형 변화 (원본 goal vs 편집 goal)",
     "groups": [{"base": OURS, "runs": ["goalstyle"], "syn_only": True,
                 "note": "같은 모델·같은 창. goal 이미지만 다른 계절/시간대/실사풍으로 바꿔도 같은 장소로 가는지. "
                         "goal image LPIPS·DINO는 원본 goal 프레임과 비교 (도착해야 할 곳). synthetic만."}]},
    {"id": "mode", "title": "축 3 · 카메라 입력 모드 (예측 vs 조건)",
     "groups": [{"base": OURS, "runs": ["camcond"],
                 "note": "같은 모델·같은 창. 조건 모드는 카메라가 입력이라 '예측 카메라 vs GT' 행이 무의미하거나 비어 있음 → "
                         "'생성 영상의 DA3 카메라 vs GT'(Extra) 행으로 카메라를 얼마나 따랐는지 비교. real은 LookOut GT 좌우 반전을 조건 카메라에만 보정해서 넣음."}]},
]


def _run(k):
    """(column name, dir) of a registry key or an explicit (name, dir) pair."""
    return k if isinstance(k, tuple) else (ABL[k]["col"], ABL[k]["dir"])


CSS = early.CSS + """td.up{color:#11612f}td.dn{color:#b42318}@media(prefers-color-scheme:dark){td.up{color:#8fe0ad}td.dn{color:#ff8a80}}
th.d,td.d{font-size:12px;border-left:2px solid var(--line)}.abl td{text-align:left;white-space:normal}"""


def _dtable(cols, rows=None):
    """early._table + one Δ (vs the first column = ours) per later column that is not a reference."""
    rows = rows or early.ALL
    data = [x for _, x, _ in cols]
    hd = "".join(f"<th>{html.escape(n)}</th>" for n, _, _ in cols)
    hd += "".join(f"<th class=d>Δ {html.escape(n.split(' · ')[0])}</th>" for n, _, dl in cols[1:] if dl)
    out = [f"<div class=wrap><table><tr><th>지표</th>{hd}</tr>"]
    g0 = None
    for g, name, key, d in rows:
        v = [x.get(key) if x else None for x in data]
        nums = sorted({a for a in v if isinstance(a, (int, float))}, reverse=(d == "^"))
        if not nums:
            continue
        if g != g0:
            out.append(f"<tr><th colspan={len(cols) + sum(1 for c in cols[1:] if c[2]) + 1} class=grp>{html.escape(g)}</th></tr>"); g0 = g
        cells = []
        for a in v:
            if not isinstance(a, (int, float)):
                cells.append("<td class=na>–</td>"); continue
            c = " class=b" if a == nums[0] and len(nums) > 1 else ""
            cells.append(f"<td{c}>{early.rs.fmt(a)}</td>")
        for (n, x, dl), a in zip(cols[1:], v[1:]):
            if not dl:
                continue
            o = v[0]
            if not (isinstance(a, (int, float)) and isinstance(o, (int, float))):
                cells.append("<td class='d na'>–</td>"); continue
            dd = a - o; better = (dd > 0) if d == "^" else (dd < 0) if d == "v" else None
            cls = "d" + ("" if dd == 0 or better is None else (" up" if better else " dn"))
            cells.append(f"<td class='{cls}'>{dd:+.3f}</td>")
        out.append(f"<tr><td>{name} {'↑' if d == '^' else ('→1' if d == '~1' else '↓')}</td>" + "".join(cells) + "</tr>")
    return "\n".join(out) + "</table></div>"


def _lo(run, w):
    """LookOut set w (A, B, C or ABC pooled) values for a run dir, as /early computes them; None if any set is missing."""
    sets = {x: early._lo_set(run, x) for x in w}
    if not all(sets.values()):
        return None
    vals = {x: early._lo_vals(*sets[x]) for x in w}
    keys = vals[w[0]]
    pooled = {k: (sum((vals[x][k] for x in w), []) if all(vals[x][k] for x in w) else []) for k in keys}
    out = {k: (sum(v) / len(v) if v else None) for k, v in pooled.items()}
    try:
        f = json.load(open(f"{run}/lookout_fvd/{w}/fvd_musiq.json"))
        out["fvd:fvd16"], out["fvd:fvdall"] = f["fvd16"], f.get(f"fvd{f.get('frames')}")
    except Exception:
        pass
    return out


def _progress(a):
    d = a["dir"]; n = lambda p: len(glob.glob(p))
    sp = a.get("split")
    # synthetic = valbench5s (3 x 40 windows), real = LookOut A+B+C (51 + 51 + 35 rollouts; 120 after the exclusion list)
    ns = sum(n(f'{d}/valbench5s/{df}/sample_??.mp4') for df in ("hard", "easy", "medium"))
    done120 = os.path.exists(f"{d}/VALBENCH_EVAL_DONE") or os.path.exists(f"{d}/valbench5s_all120/suite_summary.json")
    vb = (f"인퍼 {ns}/120 · 평가 {'✓' if done120 else '–'} · MEt3R s1 "
          f"{'✓' if os.path.exists(f'{d}/valbench5s_all120/met3r_s1/met3r.json') else '–'}")
    if a.get("lookout") is False:
        return f"synthetic: {vb}<br>real: {a.get('lo_note', '결정 대기')}"
    nr = sum(n(f"{d}/{s}/rollout_??.mp4") for s in ("lookout_1c", "lookout_1c_p1", "lookout_1c_s100", "lookout_1c_s100_p1")) \
        + n(f"{d}/lookout_turn77val35/sample_*_vis.mp4")
    lo = (f"인퍼 {nr}/137 · 평가 {'✓' if all(early._lo_set(d, x) for x in 'ABC') else '–'}"
          f" · FVD {'✓' if os.path.exists(f'{d}/lookout_fvd/ABC/fvd_musiq.json') else '–'}")
    return f"synthetic: {vb}<br>real: {lo}"


def _who(p):
    """(ablation column or None, run label, task label) of a results path."""
    for a in ABL.values():
        if p == a["dir"] or p.startswith(a["dir"] + "/"):
            return a["col"], a["col"], early._tasklab("/ckpt0/" + p[len(a["dir"]) + 1:])
    return None, early._runlab(p), early._tasklab(p)


def _gpu_section():
    """per GPU: every inference / eval python process (as /early), ablation jobs highlighted; + queued ablation steps."""
    import re, subprocess
    PP = [l.strip().split(None, 1) for l in subprocess.run(["ps", "-eo", "pid,args"], capture_output=True, text=True).stdout.splitlines()[1:]]
    PP = [(int(x[0]), x[1]) for x in PP if len(x) == 2]
    PS = [a for _, a in PP]
    ET = {}
    for l in subprocess.run(["ps", "-eo", "pid,etimes"], capture_output=True, text=True).stdout.splitlines()[1:]:
        try:
            q, e = l.split(); ET[int(q)] = int(e)
        except ValueError:
            pass
    GMAP = {}
    try:
        idx = dict(l.split(", ") for l in subprocess.run(["nvidia-smi", "--query-gpu=uuid,index", "--format=csv,noheader"],
                                                         capture_output=True, text=True).stdout.splitlines())
        for l in subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"],
                                capture_output=True, text=True).stdout.splitlines():
            q, u = l.split(", "); GMAP[int(q)] = idx.get(u, "?")
    except Exception:
        pass

    def pgpu(pid):
        g = GMAP.get(pid)
        if g is None:
            try:
                env = open(f"/proc/{pid}/environ", "rb").read().split(b"\0")
                g = next((e.split(b"=", 1)[1].decode() for e in env if e.startswith(b"CUDA_VISIBLE_DEVICES=")), "?")
            except Exception:
                g = "?"
        return g

    def arg(a, k):
        m = re.search(rf"{k}[ =](\S+)", a); return m.group(1) if m else None
    JOBS = {}
    for pid, a in PP:
        if not re.match(r"(\S*/)?python\S*\s", a) or "/ckpt" not in a or "recon_grow_rollout" in a:
            continue
        if re.search(r"infer_(cosmos_lora|rollout)\.py", a):
            O = arg(a, "--output_dir")
            if not O:
                continue
            n = arg(a, "--num_samples") or (40 if "/valbench5s/" in O else "?")
            k = len(glob.glob(f"{O}/rollout_??.mp4")) if "infer_rollout" in a else len(glob.glob(f"{O}/sample_*_vis.mp4"))
            JOBS[O] = (pgpu(pid), *_who(O), f"{k}/{n}개", ET.get(pid, 0), "인퍼")
        else:
            m = re.search(r"(?:--results_dir|--roll_dir) (\S+)|(?:cam77|lh|lookout_lh_reeval)\.py (?:\S+lookout\S+_cv )?(\S+/ckpt\S+)", a)
            if not m:
                continue
            D = (m.group(1) or m.group(2)).rstrip("/").replace("/_inc", "")
            if D in JOBS:
                continue
            st = early._stage(D, PS)
            if st == "준비":
                st = "좌우 보정 재평가" if re.search(r"lh\.py|lookout_lh_reeval", a) else next(
                    (lab for key, lab in sorted(early.STAGES, key=lambda x: -len(x[0])) if key in a), "평가")
            JOBS[D] = (pgpu(pid), *_who(D), st, ET.get(pid, 0), "평가")
    try:
        fr = {l.split(", ")[0]: l.split(", ")[1:] for l in subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True).stdout.splitlines()}
    except Exception:
        fr = {}
    rows = sorted(JOBS.values(), key=lambda x: (x[0], x[1] is None, -x[5]))
    nab = sum(1 for r in rows if r[1])
    P = [f"<h2>지금 돌아가는 작업 · GPU별</h2><div class=mut>ablation 작업 {nab}개 = 굵게 · 같은 GPU를 쓰는 다른 run 작업은 흐리게 "
         "(ps + nvidia-smi 기준, 새로고침마다 갱신)</div><div class=wrap><table><tr><th>GPU</th><th>메모리 · 사용률</th><th>run</th>"
         "<th>작업</th><th>종류</th><th>진행</th><th>경과</th></tr>"]

    def tr(g, mem, r):
        _, ab, rn, tk, pr, et, kind = r
        st = "" if ab else " style='color:var(--mut)'"
        b0, b1 = ("<b>", "</b>") if ab else ("", "")
        return (f"<tr{st}><td>{g}</td><td>{mem}</td><td>{b0}{html.escape(rn)}{b1}</td><td>{b0}{html.escape(tk)}{b1}</td><td>{kind}</td>"
                f"<td{' class=s' if ab else ''}>{html.escape(pr)}</td><td>{et // 3600}h {et % 3600 // 60:02d}m</td></tr>")
    for g in sorted(fr):
        js = [r for r in rows if r[0] == g]
        mem = f"{int(fr[g][0]) // 1024}/{int(fr[g][1]) // 1024} GB · {fr[g][2]}%"
        if not js:
            P.append(f"<tr><td>{g}</td><td>{mem}</td><td class=na colspan=5>비어 있음</td></tr>"); continue
        for i, r in enumerate(js):
            P.append(tr(g if i == 0 else "", mem if i == 0 else "", r))
    for r in rows:
        if r[0] not in fr:
            P.append(tr(html.escape(r[0]), "", r))
    P.append("</table></div>")
    # queued ablation steps (not started yet)
    Q = []
    alive = " ".join(PS)
    for a in ABL.values():
        d = a["dir"]
        for df in ("hard", "easy", "medium"):
            if not os.path.isdir(f"{d}/valbench5s/{df}"):
                Q.append((a["col"], f"valbench {df} 인퍼", "GPU 메모리 32GB 이상 비면 시작"))
            elif a.get("split") and os.path.exists(f"{d}/valbench5s/{df}.log") and "Done." in open(f"{d}/valbench5s/{df}.log", errors="ignore").read() \
                    and not os.path.exists(f"{d}/valbench5s_split/{df}/SPLIT_EVAL_DONE") and f"{d}/valbench5s_split/{df}" not in alive:
                Q.append((a["col"], f"valbench {df} 40개 평가", "GPU 메모리 30GB 이상 비면 시작"))
        if a.get("lookout") is False:
            Q.append((a["col"], "LookOut A/B/C 인퍼", a.get("lo_note", "결정 대기")))
        else:
            for w, subs in (("A", ("lookout_1c", "lookout_1c_p1")), ("B", ("lookout_1c_s100", "lookout_1c_s100_p1")), ("C", ("lookout_turn77val35",))):
                left = [early._tasklab(f"/ckpt0/{x}") for x in subs if not os.path.isdir(f"{d}/{x}")]
                if left:
                    Q.append((a["col"], ", ".join(left) + " 인퍼", "GPU 메모리 32GB 이상 비면 시작 (3분 간격)"))
        if a.get("split") and all(os.path.exists(f"{d}/valbench5s_split/{df}/SPLIT_EVAL_DONE") for df in ("hard", "easy", "medium")) and \
                not os.path.exists(f"{d}/VALBENCH_EVAL_DONE") and f"{d}/valbench5s_all120" not in alive:
            Q.append((a["col"], "valbench 120 합본 평가", "세 난이도 평가 끝 → GPU 30GB 이상"))
    if Q:
        P.append("<h2>대기열 · ablation</h2><div class=wrap><table><tr><th>ablation</th><th>작업</th><th>시작 조건</th></tr>"
                 + "".join(f"<tr><td>{html.escape(x)}</td><td>{html.escape(y)}</td><td>{html.escape(z)}</td></tr>" for x, y, z in Q)
                 + "</table></div>")
    return "\n".join(P)


def _group_tables(g):
    """valbench 120 / per difficulty + LookOut tables for one comparison group (base column first, Δ vs base)."""
    base = _run(g["base"])
    runs = [(base[0], base[1], False)] + [(*_run(k), True) for k in g["runs"]]
    ref = [(g["ref"][0], g["ref"][1], False)] if g.get("ref") else []
    P = []
    cols = [(n, early._load(f"{d}/valbench5s_all120"), dl) for n, d, dl in runs + ref]
    if g.get("scale") and ref and cols[0][1] and cols[-1][1]:
        # rough step-matched estimate: ours x (run / same-run ckpt at the run's step), per metric (valbench only)
        o, rf = cols[0][1], cols[-1][1]; est = []
        for k in g["scale"]:
            n, d = _run(k); x = early._load(f"{d}/valbench5s_all120")
            if x:
                e = {m: o[m] * x[m] / rf[m] for m in o if all(isinstance(z.get(m), (int, float)) for z in (o, x, rf))
                     and abs(rf[m]) > 1e-9}
                est.append((f"{n.split(' · ')[0].split()[0]} 추정", e, True))
        cols = cols[:len(runs)] + est + cols[len(runs):]
    # only two sets on this page: synthetic = valbench5s 120, real = LookOut A+B+C 120 (pooled; no per-difficulty / per-set tables)
    if any(x for _, x, dl in cols if dl) and cols[0][1]:
        P.append("<h3>synthetic · valbench5s 120</h3>" + _dtable([c for c in cols if c[1]]))
    elif any(x for _, x, dl in cols if dl):   # base still running: show what exists, no Δ until the base lands
        P.append(f"<h3>synthetic · valbench5s 120</h3><div class=mut>기준 열 {html.escape(cols[0][0])} 평가 진행 중 → "
                 "끝나면 기준 열과 Δ가 붙음</div>" + _dtable([(n, x, False) for n, x, _ in cols if x]))
    else:
        P.append("<h3>synthetic · valbench5s 120</h3><div class=mut>평가 대기 중</div>")
    if g.get("syn_only"):
        return "\n".join(P)
    cols = [(n, _lo(d, "ABC"), dl) for n, d, dl in runs]
    if any(x for _, x, dl in cols if dl) and cols[0][1]:
        P.append("<h3>real · LookOut A+B+C 120</h3>" + _dtable([c for c in cols if c[1]]))
    else:
        P.append("<h3>real · LookOut A+B+C 120</h3><div class=mut>평가 대기 중 (A·B·C 세트가 모두 끝나면 표가 생김)</div>")
        if g.get("partial"):   # interim: each LookOut set as soon as it lands (per-set FVD/MEt3R come with the pooled run)
            NS = {"A": "A · lookout_1c 51 (start 4 + 100)", "B": "B · 51 (DA3 깊이)", "C": "C · turn77val35"}
            for w in "ABC":
                c = [(n, _lo(d, w), dl) for n, d, dl in runs]
                if not any(x for _, x, _ in c):
                    continue
                if c[0][1]:
                    P.append(f"<h3>중간 결과 · LookOut {NS[w]}</h3>" + _dtable([x for x in c if x[1]]))
                else:
                    P.append(f"<h3>중간 결과 · LookOut {NS[w]}</h3><div class=mut>기준 열 {html.escape(c[0][0])}의 이 세트는 진행 중 → Δ 없음</div>"
                             + _dtable([(n, x, False) for n, x, _ in c if x]))
    return "\n".join(P)


def page():
    t = time.strftime("%m-%d %H:%M")
    P = [f"<!doctype html><html><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
         f"<meta http-equiv=refresh content=60><title>Ablations</title><style>{CSS}</style></head><body>",
         f"<h1>Ablation 비교</h1><div class=mut>{t} 기준 · 60초마다 새로고침 · 축(바꾼 것)별로 나눔 · 각 비교 묶음의 첫 열 = 기준 · "
         "초록 칸 = 행별 최고 · Δ = 열 − 기준 (초록 = 기준보다 좋음, 빨강 = 나쁨) · 지표·세트 정의는 /early와 동일 "
         "(카메라 = cam77, MEt3R = stride 1, LookOut = 제외명단 적용 + 좌우 보정 GT) · "
         + " · ".join(f"<a href='#{x['id']}'>{html.escape(x['title'])}</a>" for x in AXES)
         + " · <a href=early>/early</a> · <a href=./>홈</a></div>"]
    try:
        P.append(_gpu_section())
    except Exception as e:   # never lose the tables over the live status
        P.append(f"<div class=mut>작업 상태를 읽지 못함: {html.escape(str(e))}</div>")
    for x in AXES:
        P.append(f"<h2 id={x['id']}>{html.escape(x['title'])}</h2>")
        keys = [k for g in x["groups"] for k in g["runs"] + [g["base"]] if isinstance(k, str)]
        keys = list(dict.fromkeys(keys))
        P.append("<div class=wrap><table class=abl><tr><th>run</th><th>바뀐 것</th><th>체크포인트</th><th>진행</th></tr>")
        for k in keys:
            a = ABL[k]
            P.append(f"<tr><td>{html.escape(a['col'])}</td><td>{html.escape(a['what'])}</td><td>{html.escape(a['ckpt'])}"
                     f"<div class=mut>{html.escape(a['pipe'])}</div></td><td class=mut>{_progress(a)}</td></tr>")
        P.append("</table></div>")
        for g in x["groups"]:
            if g.get("sub"):
                P.append(f"<h3 style='margin-top:18px'>▸ {html.escape(g['sub'])}</h3>")
            P.append(f"<div class=mut>{html.escape(g['note'])}</div>")
            try:
                P.append(_group_tables(g))
            except Exception as e:
                P.append(f"<div class=mut>표를 만들지 못함: {html.escape(repr(e))}</div>")
    P.append("</body></html>")
    return "\n".join(P).encode()
