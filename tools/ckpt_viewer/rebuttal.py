"""/rebuttal : Video-Action Coherence metric validation (controlled perturbation test + DA3 measurement bias).

Reads results_syn/coherence_validation/{syn_valbench120,real_lookoutC35}/perturb_summary.json (tools/coherence_perturb.py)
and da3_focal/focal.log (tools/da3_focal_check.py). Rebuilt on every request (server.py reloads this module).
"""
import ast, html, json, os
import numpy as np

CV = "/root/project/egowam/results_syn/coherence_validation"
SETS = [("syn", "synthetic · valbench5s 120", "syn_valbench120"), ("real", "real · LookOut C 35 (좌우 보정 GT)", "real_lookoutC35")]
K = {"coh_roterr": "RotErr (deg)", "coh_ate_path": "ATE / Len", "coh_cammc": "CamMC"}

CSS = """
:root{color-scheme:light;--bg:#fcfcfb;--card:#ffffff;--ink:#0b0b0b;--ink2:#52514e;--mut:#7a7975;--line:#e4e3df;--grid:#efeeea;
--s1:#2a78d6;--s2:#eb6834;--up:#b42318;--dn:#11612f;--hl:#f3f7fd}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--bg:#1a1a19;--card:#222221;--ink:#f0efec;--ink2:#c3c2b7;
--mut:#9a998f;--line:#3a3a37;--grid:#2c2c2a;--s1:#3987e5;--s2:#d95926;--up:#ff8a80;--dn:#8fe0ad;--hl:#24303f}}
:root[data-theme="dark"]{color-scheme:dark;--bg:#1a1a19;--card:#222221;--ink:#f0efec;--ink2:#c3c2b7;--mut:#9a998f;--line:#3a3a37;
--grid:#2c2c2a;--s1:#3987e5;--s2:#d95926;--up:#ff8a80;--dn:#8fe0ad;--hl:#24303f}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,-apple-system,"Segoe UI","Noto Sans KR",sans-serif}
main{max-width:1120px;margin:0 auto;padding:24px 16px 64px}h1{font-size:24px;margin:0 0 4px}h2{font-size:18px;margin:36px 0 8px;
padding-top:12px;border-top:1px solid var(--line)}h3{font-size:15px;margin:18px 0 6px}.mut{color:var(--ink2)}.small{font-size:13px}
.tl{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px;margin:16px 0}.tile{background:var(--card);
border:1px solid var(--line);border-radius:8px;padding:12px 14px}.tile b{display:block;font-size:22px;font-variant-numeric:tabular-nums}
.tile span{color:var(--ink2);font-size:13px}table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums;
background:var(--card)}th,td{border-bottom:1px solid var(--line);padding:5px 8px;text-align:right;white-space:nowrap}th{color:var(--ink2);
font-weight:600}td:first-child,th:first-child{text-align:left}.tw{overflow-x:auto;margin:8px 0}td.up{color:var(--up)}td.dn{color:var(--dn)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:14px}.panel{background:var(--card);border:1px solid var(--line);
border-radius:8px;padding:10px 12px;position:relative}.panel h4{margin:0 0 2px;font-size:14px}.panel .sub{font-size:12px;color:var(--ink2)}
svg{display:block;width:100%;height:auto;overflow:visible}.ax{stroke:var(--line)}.gl{stroke:var(--grid)}.tk{fill:var(--mut);font-size:11px}
.lg{display:flex;gap:14px;font-size:12px;color:var(--ink2);margin:8px 0}.lg i{display:inline-block;width:14px;height:2px;vertical-align:middle;
margin-right:5px}.tip{position:absolute;pointer-events:none;background:var(--card);border:1px solid var(--line);border-radius:6px;
padding:6px 8px;font-size:12px;box-shadow:0 2px 8px rgba(0,0,0,.12);display:none;white-space:nowrap;z-index:5}
.note{background:var(--hl);border-radius:8px;padding:10px 14px;margin:10px 0}code{font-size:12px}
"""

JS = """
document.querySelectorAll('.panel').forEach(p=>{const tip=p.querySelector('.tip');const svg=p.querySelector('svg');
 const D=JSON.parse(p.dataset.pts);const X=JSON.parse(p.dataset.xs);const cross=svg.querySelector('.cross');
 svg.addEventListener('mousemove',e=>{const r=svg.getBoundingClientRect();const vx=(e.clientX-r.left)/r.width*svg.viewBox.baseVal.width;
  let best=0,bd=1e9;X.forEach((x,i)=>{const d=Math.abs(x-vx);if(d<bd){bd=d;best=i}});
  cross.setAttribute('x1',X[best]);cross.setAttribute('x2',X[best]);cross.style.display='';
  tip.innerHTML=D.map(s=>`<div><i style="display:inline-block;width:10px;height:2px;background:${s.c};vertical-align:middle;margin-right:5px"></i>${s.n}: <b>${s.v[best]}</b>${s.p[best]}</div>`).join('')+`<div class=mut>크기 ${p.dataset.unit.replace('{}',JSON.parse(p.dataset.ms)[best])}</div>`;
  tip.style.display='block';const pr=p.getBoundingClientRect();let lx=e.clientX-pr.left+12;if(lx>pr.width-180)lx=e.clientX-pr.left-190;
  tip.style.left=lx+'px';tip.style.top=(e.clientY-pr.top+10)+'px'});
 svg.addEventListener('mouseleave',()=>{tip.style.display='none';cross.style.display='none'})});
"""


def _load():
    S = {}
    for k, _, d in SETS:
        p = f"{CV}/{d}/perturb_summary.json"
        if os.path.exists(p):
            S[k] = json.load(open(p))
    return S


def _v(S, k, fam, m, met):
    try:
        return S[k]["families"][fam][str(m)][met]
    except KeyError:
        return None


def f3(x):
    return "–" if x is None else f"{x:.3f}"


def panel(S, title, sub, fam, mags, met, unit):
    W, H, L, R, T, B = 360, 210, 46, 12, 10, 30
    ser = []
    for i, (k, name, _) in enumerate(SETS):
        if k not in S:
            continue
        ys = [(_v(S, k, fam, m, met) or {}).get("mean") for m in mags]
        ps = [(_v(S, k, fam, m, met) or {}).get("p_vs_floor") for m in mags]
        ser.append((name.split(" · ")[0], f"var(--s{i + 1})", ys, ps))
    allv = [y for _, _, ys, _ in ser for y in ys if y is not None]
    lo, hi = min(allv + [0]), max(allv)
    hi = hi * 1.08 if hi > 0 else 1
    n = len(mags)
    X = [L + (W - L - R) * j / (n - 1) for j in range(n)]
    Y = lambda v: T + (H - T - B) * (1 - (v - lo) / (hi - lo))
    g = []
    for t in range(5):
        v = lo + (hi - lo) * t / 4; y = Y(v)
        g.append(f'<line class=gl x1={L} x2={W - R} y1={y:.1f} y2={y:.1f} /><text class=tk x={L - 6} y={y + 4:.1f} text-anchor=end>{v:.3f}</text>')
    for j, m in enumerate(mags):
        g.append(f'<text class=tk x={X[j]:.1f} y={H - B + 16} text-anchor=middle>{m}</text>')
    g.append(f'<line class=ax x1={L} x2={W - R} y1={H - B} y2={H - B} />')
    for nm, c, ys, ps in ser:
        if ys[0] is not None:
            g.append(f'<line x1={L} x2={W - R} y1={Y(ys[0]):.1f} y2={Y(ys[0]):.1f} stroke="{c}" stroke-width=1 stroke-dasharray="3 3" opacity=.55 />')
        pts = " ".join(f"{X[j]:.1f},{Y(y):.1f}" for j, y in enumerate(ys) if y is not None)
        g.append(f'<polyline points="{pts}" fill=none stroke="{c}" stroke-width=2 stroke-linejoin=round />')
        for j, y in enumerate(ys):
            if y is None:
                continue
            sig = ps[j] is not None and ps[j] < 0.01
            g.append(f'<circle cx={X[j]:.1f} cy={Y(y):.1f} r=4 fill="{c if sig else "var(--card)"}" stroke="{c}" stroke-width=2 />')
    g.append(f'<line class=cross x1=0 x2=0 y1={T} y2={H - B} stroke="var(--mut)" stroke-width=1 style="display:none" />')
    pts = [{"n": nm, "c": c, "v": [f3(y) for y in ys], "p": [("" if p is None else (" · 유의" if p < 0.01 else " · n.s.")) for p in ps]} for nm, c, ys, ps in ser]
    a = lambda o: html.escape(json.dumps(o), quote=True)
    return (f'<div class=panel data-pts="{a(pts)}" data-xs="{a([round(x, 1) for x in X])}" data-ms="{a(mags)}" data-unit="{html.escape(unit)}">'
            f'<h4>{html.escape(title)}</h4><div class=sub>{html.escape(sub)} · y = {K[met]}</div>'
            f'<svg viewBox="0 0 {W} {H}" role=img aria-label="{html.escape(title)}">{"".join(g)}</svg><div class=tip></div></div>')


def table(S, rows, mets=("coh_roterr", "coh_ate_path", "coh_cammc")):
    """rows: (label, fam, mag). Each cell = mean; vs floor coloured when p < 0.01."""
    h = "<tr><th>조건</th>" + "".join(f"<th>{n.split(' · ')[0]} {K[m]}</th>" for k, n, _ in SETS if k in S for m in mets) + "</tr>"
    b = []
    for lab, fam, m in rows:
        c = []
        for k, _, _ in SETS:
            if k not in S:
                continue
            for met in mets:
                d = _v(S, k, fam, m, met) or {}
                fl = (_v(S, k, "rot_drift", 0, met) or {}).get("mean")
                cls = ""
                if d.get("p_vs_floor") is not None and d.get("mean") is not None and fl is not None:
                    cls = " class=up" if d["p_vs_floor"] < 0.01 and d["mean"] > fl else (" class=dn" if d["mean"] < fl - 1e-9 else "")
                c.append(f"<td{cls}>{f3(d.get('mean'))}</td>")
        b.append(f"<tr><td>{html.escape(lab)}</td>{''.join(c)}</tr>")
    return f"<div class=tw><table>{h}{''.join(b)}</table></div>"


def focal():
    p = f"{CV}/da3_focal/focal.log"
    if not os.path.exists(p):
        return None
    L = [ast.literal_eval(l) for l in open(p) if l.startswith("{")]
    if not L:
        return None
    import statistics as st
    fx = [r["own_fx_proc"] for r in L]; w = L[0]["own_w_proc"]; gfx = 256 * w / 512
    big = [r for r in L if abs(r["gt_net_yaw"]) > 20]
    return {"n": len(L), "fx": st.mean(fx), "fx_sd": st.pstdev(fx), "gfx": gfx, "w": w, "nb": len(big),
            "ratio": st.median(r["own_net_yaw"] / r["gt_net_yaw"] for r in big),
            "ratio_K": st.median(r["gtK_net_yaw"] / r["gt_net_yaw"] for r in big),
            "ratio_corr": st.median(r["own_net_yaw"] * r["own_fx_proc"] / gfx / r["gt_net_yaw"] for r in big)}


VC = "/root/project/egowam/results_syn/goalprog_mode_real6_nomr_hg0_grad0.5_direct/vcons_cmp"


def _why():
    """why_residual.py (loss-only) per window -> per set lists; why_mask.py -> per set lists."""
    import glob, statistics as st
    out = {}
    for name, files in (("synthetic", glob.glob(f"{VC}/why/w[0-9][0-9][0-9].json")),
                        ("real", glob.glob(f"{VC}/why_lookout/[ABC][0-9][0-9][0-9].json"))):
        R = [json.load(open(f)) for f in files]
        if not R:
            continue
        Q = [json.load(open(f[:-5] + "_q.json")) for f in files if os.path.exists(f[:-5] + "_q.json")]
        md = lambda v: st.median(v) if v else None
        g = lambda v, k: v["rank"][k]
        out[name] = {"n": len(R),
                     "floor_med": md([g(r, "GT")["direct"] for r in R]), "floor_min": min(g(r, "GT")["direct"] for r in R),
                     "floor_pos": sum(g(r, "GT")["direct"] > 0 for r in R), "res_gt_max": max(g(r, "GT")["residual"] for r in R),
                     "tavg5_pref_dir": sum(g(r, "tavg5")["direct"] < g(r, "GT")["direct"] for r in R) / len(R),
                     "tavg5_pref_res": sum(g(r, "tavg5")["residual"] < g(r, "GT")["residual"] for r in R) / len(R),
                     "drift_dir_end": md([r["drift"]["direct"]["loss_end"] for r in R]),
                     "drift_dir_dist": md([r["drift"]["direct"]["lat_dist_end"] for r in R]),
                     "drift_res_dist": max(r["drift"]["residual"]["lat_dist_end"] for r in R),
                     "psnr_dir": md([q["direct"]["psnr"] for q in Q]), "psnr_res": md([q["residual"]["psnr"] for q in Q]),
                     "psnr_vae": md([q["gtvae"]["psnr"] for q in Q]), "lpips_dir": md([q["direct"]["lpips"] for q in Q]),
                     "lpips_res": md([q["residual"]["lpips"] for q in Q])}
    mk = {}
    M = [json.load(open(f)) for f in glob.glob(f"{VC}/why_mask2/*.json")]
    for name, sel in (("synthetic", lambda r: r["bench"] == "valbench"), ("real", lambda r: r["bench"] != "valbench")):
        S = [r for r in M if sel(r)]
        if not S:
            continue
        md = lambda k: st.median([r[k] for r in S if r.get(k) is not None])
        mk[name] = {"n": len(S), **{k: md(k) for k in ("frac_mask", "frac_occ_depth", "consistent_M", "consistent_F", "consistent_Oc",
                                                       "inconsistent_M", "inconsistent_F", "inconsistent_Oc", "floor_M", "floor_Oc")},
                    "sep_M": st.median(r["inconsistent_M"] / r["consistent_M"] for r in S),
                    "sep_F": st.median(r["inconsistent_F"] / r["consistent_F"] for r in S)}
    return out, mk


def vcons_section():
    W, MK = _why()
    if not W and not MK:
        return ""
    P = ["<h2 id=vcons>Direct latent matching vs Residual matching · 메커니즘 (학습된 모델 없이 loss만)</h2>"
         "<div class=mut>학습 코드와 같은 loss · geometry · pair (gap 1,2,4,8 양방향 + anchor). direct = |x̂_t − W(x̂_s)|, "
         "residual = |e_t − W(e_s)|, e = x̂ − z. 모델을 쓰지 않으므로 학습 step · 데이터 차이 같은 교란이 없음. "
         "synthetic = valbench5s 120 (CARLA GT depth), real = LookOut A+B+C 120 (DA3 pseudo-GT depth, 좌우 보정 pose). "
         "자세한 그림 · 유도: <a href=/vcons/why.html>/vcons/why.html</a></div>"]
    if W:
        rows = [("GT latent 자체의 loss (완벽한 복원)", lambda w: f"{f3(w['floor_med'])} (중앙값, 최소 {f3(w['floor_min'])}; {w['floor_pos']}/{w['n']}개 창에서 0보다 큼)",
                 lambda w: f"{f3(w['res_gt_max'])} (최대, 모든 창)"),
                ("시간 평균(잔상) 영상을 GT보다 좋게 본 창의 비율", lambda w: f3(w["tavg5_pref_dir"]), lambda w: f3(w["tavg5_pref_res"])),
                ("GT에서 출발해 loss만 300 step 최소화: 끝 loss", lambda w: f3(w["drift_dir_end"]), lambda w: "0.000"),
                ("　〃　GT latent에서 멀어진 거리", lambda w: f3(w["drift_dir_dist"]), lambda w: f3(w["drift_res_dist"]) + " (최대)"),
                ("　〃　도착한 영상의 PSNR vs GT (VAE 상한)", lambda w: f"{f3(w['psnr_dir'])} dB", lambda w: f"{f3(w['psnr_res'])} dB (상한 {f3(w['psnr_vae'])})"),
                ("　〃　LPIPS ↓", lambda w: f3(w["lpips_dir"]), lambda w: f3(w["lpips_res"]))]
        h = "<tr><th>측정</th>" + "".join(f"<th>{n} · direct</th><th>{n} · residual</th>" for n in W) + "</tr>"
        b = "".join(f"<tr><td>{html.escape(lab)}</td>" + "".join(f"<td>{fd(w)}</td><td>{fr(w)}</td>" for w in W.values()) + "</tr>"
                    for lab, fd, fr in rows)
        P.append(f"<h3>① direct는 정답에도 penalty를 준다</h3><div class=tw><table>{h}{b}</table></div>")
        P.append("<div class=mut small>direct의 GT 점수 = warp floor ‖A z‖ (시점에 따른 외관 변화 · VAE non-equivariance · 보간 오차 · 가림). "
                 "residual은 정의상 GT에서 0이고 기울기도 0 → GT에서 출발하면 움직이지 않음. direct는 GT에서 출발해도 loss가 줄어드는 방향(프레임끼리 같게 = 평탄화)으로 영상을 끌고 감.</div>")
        P.append("<div class=grid><figure class=panel style='margin:0'><img src=/vcons/why_w071.jpg style='width:100%' alt='drift from GT'>"
                 "<div class=sub>위부터 GT · GT의 VAE 왕복 · GT에서 direct만 최소화 · GT에서 residual만 최소화 (synthetic 창 71)</div></figure>"
                 "<figure class=panel style='margin:0'><img src=/vcons/warp_bad_s098.png style='width:100%' alt='warp floor'>"
                 "<div class=sub>GT 두 프레임 사이 warp floor |z_t − W(z_s)|: 보이는 영역에서도 0이 아니고 (위), 가림이 큰 구간에서 mask가 거의 비어 버림 (아래)</div></figure></div>")
    if MK:
        h = "<tr><th>측정 (중앙값)</th>" + "".join(f"<th>{n}</th>" for n in MK) + "</tr>"
        rows = [("화면 안 투영 중 가시성 mask 비율", "frac_mask"), ("가려진 영역 비율 (깊이 유효)", "frac_occ_depth"),
                ("3D 일관 오차 · residual · mask 있음 ↓", "consistent_M"), ("3D 일관 오차 · residual · mask 없음", "consistent_F"),
                ("3D 일관 오차 · residual · 가려진 영역만", "consistent_Oc"), ("비일관 오차 · residual · mask 있음", "inconsistent_M"),
                ("비일관 오차 · residual · mask 없음", "inconsistent_F"), ("구분 비율 비일관 ÷ 일관 · mask 있음 ↑", "sep_M"),
                ("구분 비율 비일관 ÷ 일관 · mask 없음", "sep_F"), ("GT warp floor · 보이는 영역", "floor_M"), ("GT warp floor · 가려진 영역", "floor_Oc")]
        b = "".join(f"<tr><td>{lab}</td>" + "".join(f"<td>{f3(m[k])}</td>" for m in MK.values()) + "</tr>" for lab, k in rows)
        P.append("<h3>② Residual에서 visibility mask가 필요한 이유</h3><div class=mut>GT latent에 오차를 직접 넣어 정답을 아는 상태로 비교 (x = z + e, "
                 "평균 |e| = 0.469 = 생성 영상의 실제 latent 오차). <b>3D 일관 오차</b> = 3D 점마다 오차 하나 (어느 프레임에서 봐도 같음) → "
                 "기하적으로 완벽히 일관된 영상, 올바른 일관성 loss라면 ~0. <b>비일관 오차</b> = 같은 장, 프레임마다 위상만 다름 → 커야 함. "
                 "mask 없음 = 화면 안에 투영되는 모든 픽셀을 평균 (가림 · 하늘 포함).</div>")
        P.append(f"<div class=tw><table>{h}{b}</table></div>")
        P.append("<div class=mut small>가려진 영역에서 W(e_s)는 앞을 가린 다른 표면의 오차를 가져오므로, 완벽히 일관된 오차도 비일관 오차의 절반 이상만큼 벌을 받음 "
                 "(direct가 GT에서 받는 잘못된 penalty와 같은 구조). mask가 있으면 남는 값은 bilinear 보간 오차뿐. mask를 빼면 일관/비일관 구분력이 약 1/4–1/5로 떨어짐. "
                 "mask는 GT depth · pose로만 만들고 기울기를 흘리지 않음 (모델이 mask를 비워 loss를 피할 수 없음). 코드: tools/vcons_cmp/why_mask.py</div>")
    return "".join(P)


VD = "/root/project/egowam/results_syn/goalprog_mode_real6_nomr_hg0_grad0.5_direct/vcons_cmp/design/summary.json"


def vdesign_section():
    """Residual L_vcons design choices without retraining (tools/vcons_cmp/vcons_design.py + vcons_design_agg.py)."""
    if not os.path.exists(VD):
        return ""
    S = json.load(open(VD))
    ci = lambda d, i: f"<div class=small>[{d['lo'][i]:+.3f}, {d['hi'][i]:+.3f}]</div>"
    P = ["<h2 id=vdesign>Residual consistency 설계 선택 (재학습 없음)</h2>"
         "<div class=mut>학습과 같은 geometry (latent 64×64, key frame start+4k, GT depth / LookOut은 DA3 pseudo-GT, gt_visibility mask). "
         "창마다 latent 쌍 (t ← s) 380개 전부를 gap |t − s|별로 모음 (frame 0과의 쌍 = anchor는 따로). 픽셀 가중 (= loss와 같은 평균), "
         "괄호 = 창 bootstrap 95% CI. 생성 영상: ours (avg 7500+8750, 학습 gap 1·2·4·8 + anchor), abl4 (IV_cont5k ck 6000, L_vcons 없음; "
         "다른 학습 차이도 있음). λ_vcons 변화는 재학습이 필요해 여기서는 다루지 않음. "
         "코드: tools/vcons_cmp/vcons_design.py, vcons_design_agg.py</div>"]
    GS = [1, 2, 3, 4, 6, 8, 12, 16]
    for name, s in S.items():
        g = s["gap"]
        h = ("<tr><th>gap</th><th>화면 안에 투영되는 비율</th><th>그중 mask 비율</th><th>가려진 비율</th><th>GT warp floor ↓</th>"
             "<th>구분력 · mask 있음 ↑</th><th>구분력 · mask 없음</th><th>|e| ours / abl4</th><th>잔차 ours / abl4</th><th>잔차비 ours / abl4</th><th>잔차비 ours − abl4</th></tr>")
        b = ""
        for G in GS:
            i = G - 1; d = g["ratio_diff"]["v"][i]; sig = (g["ratio_diff"]["lo"][i] > 0) == (g["ratio_diff"]["hi"][i] > 0)
            b += (f"<tr><td>{G}{' *' if G in (1, 2, 4, 8) else ''}</td><td>{g['inframe']['v'][i]:.3f}</td><td>{g['frac_mask']['v'][i]:.3f}</td>"
                  f"<td>{g['frac_occ']['v'][i]:.3f}</td><td>{g['floor_M']['v'][i]:.3f}</td><td>{g['sep_M']['v'][i]:.3f}</td><td>{g['sep_F']['v'][i]:.3f}</td>"
                  f"<td>{g['ours_err']['v'][i]:.3f} / {g['abl4_err']['v'][i]:.3f}</td><td>{g['ours_res']['v'][i]:.3f} / {g['abl4_res']['v'][i]:.3f}</td>"
                  f"<td>{g['ours_ratio']['v'][i]:.3f} / {g['abl4_ratio']['v'][i]:.3f}</td>"
                  f"<td style='color:var({'--dn' if d < 0 else '--up'});{'font-weight:600' if sig else ''}'>{d:+.3f}{ci(g['ratio_diff'], i)}</td></tr>")
        P.append(f"<h3>① frame-pair 거리 · {name} ({s['n']}개 창)</h3><div class=tw><table>{h}{b}</table></div>")
        ts = s["train_share"]
        P.append("<div class=mut small>* = 학습에 쓰는 gap. 한 학습 step의 mask 픽셀 중 각 쌍 종류의 몫: "
                 + " · ".join(f"{k} {v:.3f}" for k, v in ts.items()) + "</div>")
    P.append("<div class=mut small><b>GT warp floor</b> = GT 영상끼리 |z_t − W(z_s)| (mask 안): 정답 영상도 남기는 대응 오차 (시점에 따른 외관 변화 · VAE · 보간 · "
             "real은 DA3 depth 오차까지). <b>구분력</b> = 비일관 오차 ÷ 3D 일관 오차의 residual (why_mask와 같은 합성 오차장, |e| 0.469): "
             "loss가 '틀린 일관성'과 '맞는 일관성'을 몇 배로 구분하는지. <b>잔차비</b> = |e_t − W(e_s)| ÷ |e_t| (mask 안): 생성 영상 오차가 "
             "시점 사이에서 얼마나 일치하는지 (3D 일관이면 보간 오차만큼, 서로 무관한 오차면 약 1.414).</div>")
    # ② drift
    EV = [("err_all", "|x − z| 전체"), ("err_vis0", "frame 0에서 보이는 곳"), ("err_novis0", "frame 0에서 안 보이는 곳")] + \
         [(f"res_g{g}", f"잔차 gap {g}") for g in (1, 2, 4, 8, 16)]
    VN = [("start", "시작 (ours 생성 영상)"), ("train", "학습 설정: gap 1·2·4·8 + anchor"), ("pairs", "gap 1·2·4·8, anchor 없음"),
          ("anchor", "anchor (j ← 0)만"), ("g1", "gap 1만"), ("g2", "gap 2만"), ("g4", "gap 4만"), ("g8", "gap 8만"), ("g16", "gap 16만")]
    for name, s in S.items():
        d = s["drift"]
        h = "<tr><th>쌍 구성</th>" + "".join(f"<th>{l}</th>" for _, l in EV) + "</tr>"
        b = "".join(f"<tr><td>{l}</td>" + "".join(f"<td>{d[v][k]:.3f}</td>" for k, _ in EV) + "</tr>" for v, l in VN if v in d)
        P.append(f"<h3>② 어떤 거리의 쌍이 무엇을 고치나 · {name} (loss만 최소화)</h3><div class=tw><table>{h}{b}</table></div>")
    P.append("<div class=mut small>ours 생성 latent에서 출발해 L_vcons 하나만 Adam (lr 0.02, 200 step)으로 최소화 (frame 0 = GT 고정, mask · stop-grad 학습과 동일). "
             "anchor가 없는 쌍은 오차를 서로 맞출 수만 있고, 0으로 보낼 기준(frame 0)은 anchor로만 들어옴. "
             "실제 학습에서는 L_gen이 함께 있으므로 크기가 아니라 방향(어느 gap이 일관해지는지)을 보는 표.</div>")
    # ③ stop-grad / mask
    GV = [("pairs_sg_M", "쌍 · stop-grad · mask (학습)"), ("pairs_nosg_M", "쌍 · gradient 양쪽 · mask"), ("pairs_sg_F", "쌍 · stop-grad · mask 없음"),
          ("pairs_nosg_F", "쌍 · gradient 양쪽 · mask 없음"), ("anchor_M", "anchor · mask"), ("anchor_F", "anchor · mask 없음"),
          ("train", "학습 전체 (쌍 + anchor)")] + [(f"g{g}", f"gap {g}만") for g in (1, 2, 4, 8, 12, 16)] + \
         [("pairs_sg_M-pairs_nosg_M", "차이: stop-grad − 양쪽"), ("pairs_sg_M-pairs_sg_F", "차이: mask − mask 없음"), ("anchor_M-anchor_F", "차이: anchor mask − 없음")]
    h = "<tr><th>L_vcons 변형</th>" + "".join(f"<th>{n} · {m}</th>" for n in S for m in ("ours", "abl4")) + "</tr>"
    b = ""
    for k, l in GV:
        b += f"<tr><td>{l}</td>"
        for s in S.values():
            for m in ("ours", "abl4"):
                c = s["cos0"][m].get(k)
                b += "<td>–</td>" if c is None else f"<td>{c['v']:+.3f}<div class=small>[{c['lo']:+.3f}, {c['hi']:+.3f}]</div></td>"
        b += "</tr>"
    P.append("<h3>③ stop-gradient · visibility mask: gradient가 정답 쪽을 가리키는가</h3>"
             "<div class=mut>생성 영상에서 L_vcons의 −gradient와 정답 방향 −(x − z)의 cosine (frame 1–19). 1 = 오차를 그대로 줄이는 방향, 0 = 무관.</div>"
             f"<div class=tw><table>{h}{b}</table></div>")
    P.append("<div class=mut small>stop-grad 유무: 방향 차이 없음 (차이 CI가 +0.001–+0.009). mask를 빼면 cosine이 오히려 높아짐: 가려진 픽셀의 짝 W(e_s)는 "
             "다른 표면의 오차라 e_t와 무관하고, 무관한 값과의 L1은 사실상 |e_t|에 대한 L1이라 GT 쪽을 가리킴. 즉 이 기준은 '정답 쪽으로 미는가'만 보고 "
             "'일관성만 벌하는가'는 보지 못함. mask의 근거는 3D 일관 오차를 벌하지 않는다는 점 (위 vcons ② 표와 ①의 구분력 열).</div>")
    s, r = S.get("synthetic"), S.get("real")
    if s and r:
        gs, gr = s["gap"], r["gap"]
        P.append("<div class=note><b>요약.</b> "
                 f"(1) <b>거리가 멀수록 대응이 나빠짐</b>: gap 1 → 16에서 화면 안 픽셀 중 mask 비율 {gs['frac_mask']['v'][0]:.3f} → {gs['frac_mask']['v'][15]:.3f} "
                 f"(real {gr['frac_mask']['v'][0]:.3f} → {gr['frac_mask']['v'][15]:.3f}), GT warp floor {gs['floor_M']['v'][0]:.3f} → {gs['floor_M']['v'][15]:.3f} "
                 f"(real {gr['floor_M']['v'][0]:.3f} → {gr['floor_M']['v'][15]:.3f}), 구분력 {gs['sep_M']['v'][0]:.3f} → {gs['sep_M']['v'][15]:.3f}배 "
                 f"(real {gr['sep_M']['v'][0]:.3f} → {gr['sep_M']['v'][15]:.3f}). mask 없이는 {gs['sep_F']['v'][0]:.3f} → {gs['sep_F']['v'][15]:.3f}배라 "
                 "먼 쌍일수록 mask가 더 필요함. "
                 f"(2) <b>가까운 쌍과 anchor가 역할을 나눔</b>: 한 gap만 쓰면 그 gap만 일관해지고 (gap 8만: gap 1 잔차 {s['drift']['g8']['res_g1']:.3f}, 시작 {s['drift']['start']['res_g1']:.3f}), "
                 f"anchor만 쓰면 frame 0에서 보이는 곳만 고쳐짐 (안 보이는 곳 {s['drift']['anchor']['err_novis0']:.3f} 그대로). 학습 구성 (gap 1·2·4·8 + anchor)이 오차를 가장 많이 줄이고 "
                 f"({s['drift']['start']['err_all']:.3f} → {s['drift']['train']['err_all']:.3f}), 학습에 없는 gap 16도 {s['drift']['start']['res_g16']:.3f} → {s['drift']['train']['res_g16']:.3f}. "
                 f"(3) <b>stop-gradient</b>: loss만으로는 차이 없음. (4) 학습된 모델 비교 (교란 있음): ours는 |e|가 abl4보다 작지만 시점 간 잔차는 거의 같음 → 이 latent 지표로는 "
                 "L_vcons 효과가 따로 보이지 않음. (5) λ_vcons는 재학습 필요.</div>")
    return "".join(P)


GC = "/root/project/egowam/results_syn/gtcam_cmp"
MFIG = "/root/project/egowam/results_syn/goalprog_mode_real6_nomr_hg0_grad0.5_direct/vcons_cmp/page/motion_met3r.png"


def _motion_scatter(d, title, rho_m):
    """inline SVG: x = motion / GT, y = MEt3R (stride 1); ours blue, Aether orange, other baselines gray; OLS line over all."""
    W_, H_, L, R_, T, B = 600, 400, 54, 18, 34, 46
    NAME = {"aether_planning": "Aether (planning)"}
    pts = [(o["motion_ratio"], o["met3r"], m, NAME.get(m, o["label"])) for m, o in d.items()]
    xmax = max(1.6, max(p[0] for p in pts) * 1.08)
    step = next(s_ for s_ in (0.01, 0.02, 0.025, 0.05) if max(p[1] for p in pts) * 1.1 / s_ <= 6)
    ymax = step * np.ceil(max(p[1] for p in pts) * 1.1 / step)
    X = lambda v: L + v / xmax * (W_ - L - R_); Y = lambda v: H_ - B - v / ymax * (H_ - T - B)
    col = lambda m: "var(--s1)" if m.startswith("ours") else ("var(--s2)" if m.startswith("aether") else "var(--mut)")
    g = [f"<svg viewBox='0 0 {W_} {H_}' style='width:100%;max-width:{W_}px;height:auto' role=img aria-label='{html.escape(title)}'>",
         f"<text x={L} y=18 font-size=13 font-weight=600 fill='var(--ink)'>{html.escape(title)}</text>",
         f"<text x={W_ - R_} y=18 font-size=11 text-anchor=end fill='var(--ink2)'>모델 간 Spearman {rho_m:.3f}</text>"]
    for k in range(int(round(ymax / step)) + 1):                          # y grid
        v = step * k; y = Y(v)
        g.append(f"<line x1={L} x2={W_ - R_} y1={y:.1f} y2={y:.1f} stroke='var(--grid)'/><text x={L - 6} y={y + 4:.1f} font-size=10 text-anchor=end fill='var(--mut)'>{v:.3f}</text>")
    for v in [i * 0.2 for i in range(int(xmax / 0.2) + 1)]:                 # x ticks
        g.append(f"<text x={X(v):.1f} y={H_ - B + 16} font-size=10 text-anchor=middle fill='var(--mut)'>{v:.1f}</text>")
    g.append(f"<line x1={L} x2={W_ - R_} y1={H_ - B} y2={H_ - B} stroke='var(--line)'/>")
    g.append(f"<line x1={X(1):.1f} x2={X(1):.1f} y1={T} y2={H_ - B} stroke='var(--ink2)' stroke-dasharray='4 4'/>"
             f"<text x={X(1) + 4:.1f} y={T + 10} font-size=10 fill='var(--ink2)'>GT와 같은 움직임</text>")
    xs = np.array([p[0] for p in pts]); ys = np.array([p[1] for p in pts]); b1, b0 = np.polyfit(xs, ys, 1)
    g.append(f"<line x1={X(0):.1f} y1={Y(b0):.1f} x2={X(xmax):.1f} y2={Y(b0 + b1 * xmax):.1f} stroke='var(--mut)' stroke-width=1 stroke-dasharray='2 3' opacity=.7/>")
    g.append(f"<text x={W_ - R_} y={H_ - 8} font-size=11 text-anchor=end fill='var(--ink2)'>움직임 / GT (프레임 간 평균 변화, 같은 영역) →</text>"
             f"<text x=12 y={T + (H_ - T - B) / 2:.0f} font-size=11 fill='var(--ink2)' transform='rotate(-90 12 {T + (H_ - T - B) / 2:.0f})' text-anchor=middle>MEt3R (stride 1) ↓ 좋음</text>")
    # labels: first free slot among right / left / above / below (boxes vs placed labels and all points), else nudge down
    P_ = [(X(x), Y(y)) for x, y, _, _ in pts]
    boxes = []

    def free(bx):
        x0, y0, x1, y1 = bx
        if x0 < L or x1 > W_ - 2 or y0 < T - 6 or y1 > H_ - B:
            return False
        if any(not (x1 < a0 or x0 > a1 or y1 < b0_ or y0 > b1_) for a0, b0_, a1, b1_ in boxes):
            return False
        return not any(x0 - 6 < cx < x1 + 6 and y0 - 6 < cy < y1 + 6 for cx, cy in P_)
    for x, y, m, l in sorted(pts, key=lambda p: -p[0]):
        px, py = X(x), Y(y); w = 6.2 * len(l) + 4; hh = 12
        cand = [("start", px + 9, py + 4, (px + 8, py - 8, px + 8 + w, py + 4)), ("end", px - 9, py + 4, (px - 8 - w, py - 8, px - 8, py + 4)),
                ("middle", px, py - 11, (px - w / 2, py - 21, px + w / 2, py - 9)), ("middle", px, py + 20, (px - w / 2, py + 9, px + w / 2, py + 21))]
        pick = next((c for c in cand if free(c[3])), None)
        lead = False
        if pick is None:
            anc, lx, ly, bx = cand[0]
            for k in range(1, 12):
                b2 = (bx[0], bx[1] + 13 * k, bx[2], bx[3] + 13 * k)
                if free(b2):
                    pick = (anc, lx, ly + 13 * k, b2); lead = True; break
            pick = pick or cand[0]
        anc, lx, ly, bx = pick; boxes.append(bx)
        g.append(f"<g><title>{html.escape(l)} · 움직임/GT {x:.3f} · MEt3R {y:.3f}</title>"
                 f"<circle cx={px:.1f} cy={py:.1f} r={7 if m.startswith('ours') else 5.5} fill='{col(m)}' stroke='var(--card)' stroke-width=2/>"
                 + (f"<line x1={px:.1f} y1={py:.1f} x2={lx - 2:.1f} y2={ly - 4:.1f} stroke='var(--line)'/>" if lead else "")
                 + f"<text x={lx:.1f} y={ly:.1f} font-size=11 text-anchor={anc} fill='var(--ink{'' if m.startswith(('ours', 'aether')) else '2'})'"
                 + (" font-weight=600" if m.startswith("ours") else "") + f">{html.escape(l)}</text></g>")
    g.append("</svg>")
    # var() is not valid inside SVG presentation attributes (fill='var(--x)' is ignored -> black / no stroke, invisible in
    # dark mode): move every fill / stroke that uses a CSS variable into the style attribute
    import re as _re

    def _fix(mt):
        tag = mt.group(0); st = []
        for a in ("fill", "stroke"):
            m2 = _re.search(rf" {a}='(var\(--[\w-]+\))'", tag)
            if m2:
                st.append(f"{a}:{m2.group(1)}"); tag = tag.replace(m2.group(0), "")
        return tag if not st else tag.replace(">", f" style='{';'.join(st)}'>", 1) if not tag.endswith("/>") else tag[:-2] + f" style='{';'.join(st)}'/>"
    return _re.sub(r"<(?:line|circle|text)\b[^>]*>", _fix, "".join(g))


def gtcam_section():
    """GT-camera-conditioned: ours camcond vs Aether gtcam (tools/gtcam_cmp.py, tools/gtcam_fair.py), metric groups and order
    as /early · /ablation (report_suite.ROWS)."""
    if not os.path.exists(f"{GC}/gtcam_cmp.json"):
        return ""
    R = json.load(open(f"{GC}/gtcam_cmp.json"))
    F = json.load(open(f"{GC}/fair.json")) if os.path.exists(f"{GC}/fair.json") else {}
    FV = json.load(open(f"{GC}/fair_fvd.json")) if os.path.exists(f"{GC}/fair_fvd.json") else {}
    ML = [("ours_cc", "ours · GT 카메라"), ("aether_gtcam", "Aether · GT 카메라"), ("ours_plan", "ours · planning"), ("aether_plan", "Aether · planning")]
    # (group, label, source, key, dir)   source: fv = same-crop set-level, fair = same-crop per window, cmp = /cmp per window
    ROWS = [("Video quality", "FVD16", "fv", "fvd16", -1), ("Video quality", "FVD77 (전체 프레임)", "fv", "fvd77", -1),
            ("Video quality", "MUSIQ", "fv", "musiq", 1),
            ("3D consistency", "PSNR (77프레임)", "fair", "psnr77", 1), ("3D consistency", "SSIM (77프레임)", "fair", "ssim77", 1),
            ("3D consistency", "LPIPS (77프레임)", "fair", "lpips77", -1),
            ("3D consistency", "움직임: 프레임 간 평균 변화 (0–255, 프레임 1–76) — 참고", "fair", "motion", 0),
            ("3D consistency", "움직임 / GT — 참고", "fair", "motion_ratio", 0),
            ("3D consistency", "MEt3R (stride 1)", "fair", "met3r_s1", -1), ("3D consistency", "MEt3R (stride 4)", "fair", "met3r", -1),
            ("3D consistency", "움직임: 프레임 16–76 — 참고", "fair", "motion_t16", 0),
            ("3D consistency", "움직임 / GT: 프레임 16–76 — 참고", "fair", "motion_ratio_t16", 0),
            ("3D consistency", "MEt3R (stride 1) · 프레임 16–76", "fair", "met3r_s1_t16", -1),
            ("3D consistency", "MEt3R (stride 4) · 프레임 16–76", "fair", "met3r_t16", -1),
            ("Action faithfulness (AF · action vs GT)", "RotErr° ‡", "cmp", "act_rot", -1),
            ("Action faithfulness (AF · action vs GT)", "ATE ‡", "cmp", "act_ate", -1),
            ("Video-action coherence (VA · action vs 영상의 DA3)", "depth AbsRel", "cmp", "d_absrel", -1),
            ("Video-action coherence (VA · action vs 영상의 DA3)", "depth δ<1.25", "cmp", "d_d1", 1),
            ("Video-action coherence (VA · action vs 영상의 DA3)", "depth δ<1.5", "cmp", "d_d15", 1),
            ("Video-action coherence (VA · action vs 영상의 DA3)", "cam RotErr° ‡", "cmp", "coh_rot", -1),
            ("Video-action coherence (VA · action vs 영상의 DA3)", "cam ATE / path ‡", "cmp", "coh_ate", -1),
            ("Video-action coherence (VA · action vs 영상의 DA3)", "cam CamMC ‡ (77프레임, DA3 단위)", "cmp", "coh_cammc", -1),
            ("Goal achievement", "goal SR (DA3 카메라)", "cmp", "goal_succ", 1), ("Goal achievement", "goal SR (action) ‡", "cmp", "goal_act", 1),
            ("Goal achievement", "goal 이미지 LPIPS †", "cmp", "goal_lpips", -1), ("Goal achievement", "goal 이미지 DINO †", "cmp", "goal_dino", 1),
            ("DA3 camera vs GT (VM · 영상이 주어진/생성 카메라를 따르는가)", "RotErr°", "cmp", "cam_rot", -1),
            ("DA3 camera vs GT (VM · 영상이 주어진/생성 카메라를 따르는가)", "TransErr", "cmp", "cam_trans", -1),
            ("DA3 camera vs GT (VM · 영상이 주어진/생성 카메라를 따르는가)", "ATE", "cmp", "cam_ate", -1)]

    def get(grp, src, k, m):
        if src == "fv":
            return FV.get(grp, {}).get(m, {}).get(k)
        if src == "fair":
            return F.get(grp, {}).get("mean", {}).get(k, {}).get(m)
        return R[grp]["groups"]["all"]["mean"].get(k, {}).get(m)

    def pair(grp, src, k):
        if src == "fv":
            return None
        if src == "fair":
            return F.get(grp, {}).get("paired", {}).get("ours_cc-aether_gtcam", {}).get(k)
        return R[grp]["groups"]["all"]["paired"].get("ours_cc-aether_gtcam", {}).get(k)

    def dcell(p, d):
        if not p:
            return "<td>–</td>"
        good = p["d"] * d > 0; sig = (p["ci"][0] > 0) == (p["ci"][1] > 0)
        return (f"<td style='color:var({'--dn' if good else '--up'});{'font-weight:600' if sig else ''}'>{p['d']:+.3f}"
                f"<div class=small>[{p['ci'][0]:+.3f}, {p['ci'][1]:+.3f}]</div></td>")
    f3_ = lambda v: "<td>–</td>" if v is None else f"<td>{v:.3f}</td>"
    SRC = {"fv": "같은 crop · 세트", "fair": "같은 crop", "cmp": "/cmp 규약"}
    P = ["<h2 id=gtcam>GT 카메라를 줬을 때: ours vs Aether</h2>"
         "<div class=mut>두 모델 모두 GT 카메라 경로를 조건으로 받아 생성. ours = avg 7500+8750, GT raymap 입력 (mode 0); "
         "Aether = AetherV1 prediction 모드 (GT camera raymap, 41프레임 2청크, 둘째 청크는 첫 청크의 마지막 생성 프레임에서 시작; "
         "HF wandering_inference_comparison_output/aether_gtcam). planning 열 = 각 모델이 카메라도 직접 만든 기존 결과 (참고). "
         "창 = /ablation과 같은 synthetic 120 · real LookOut A+B+C 120. 지표 묶음 · 순서 = /early · /ablation과 같음. "
         "<b>같은 crop</b> = 모든 영상을 Aether가 보는 영역 (512×512 GT의 중앙 띠 512×342)으로 자르고 GT 창의 같은 77프레임에 맞춰 다시 계산 "
         "(PSNR/SSIM/LPIPS = eval_video_quality.py 함수, 77프레임 전체 평균; MEt3R = eval_met3r.py 설정; FVD = I3D cd-fvd 256², 기준 = 같은 crop의 GT). "
         "<b>/cmp 규약</b> = 논문 표와 같은 창별 값. 차이 괄호 = paired bootstrap 95% CI (세트 단위 FVD/MUSIQ는 CI 없음). 같은 crop 칸 아래 작은 글씨 = 창별 중앙값 (n &lt; 120이면 계산 중).</div>"]
    import glob as _g, time as _t
    cnt = {}
    for f in _g.glob(f"{GC}/fair/*.json"):
        m_, t_ = os.path.basename(f).split("_")[0], os.path.basename(f).split("_")[1][0]
        cnt.setdefault(m_, [0, 0])[0 if t_ == "v" else 1] += 1
    lab_ = {"fid": "PSNR/SSIM/LPIPS", "f0": "프레임 0", "met3r": "MEt3R s4", "met3r1": "MEt3R s1", "motion": "움직임",
            "met3rT": "MEt3R s4 · 16–76", "met3r1T": "MEt3R s1 · 16–76"}
    last = max((os.path.getmtime(f"{GC}/{x}") for x in ("fair.json", "gtcam_cmp.json") if os.path.exists(f"{GC}/{x}")), default=0)
    P.append("<div class=note>마지막 집계 " + _t.strftime("%m-%d %H:%M:%S", _t.localtime(last)) + " (진행 중에는 2분마다 자동 재집계) · 같은 crop 진행: "
             + " · ".join(f"{lab_.get(k, k)} syn {v[0]}/120 · real {v[1]}/120" for k, v in sorted(cnt.items())) + "</div>")
    for grp, title in (("synthetic", "synthetic · valbench5s 120"), ("real", "real · LookOut A+B+C 120")):
        h = ("<tr><th>지표</th><th>계산</th>" + "".join(f"<th>{l}</th>" for _, l in ML) + "<th>ours − Aether (GT 카메라)</th></tr>")
        b = ""; g0 = None
        for g, lab, src, k, d in ROWS:
            vals = [get(grp, src, k, m) for m, _ in ML]
            if all(v is None for v in vals):
                continue
            if g != g0:
                b += f"<tr><th colspan={len(ML) + 3} class=grp style='text-align:left'>{html.escape(g)}</th></tr>"; g0 = g
            trivial = k in ("act_rot", "act_ate", "goal_act")      # ours GT-camera run: action = the given GT camera itself
            def cell(m, v):
                if trivial and m == "ours_cc":
                    return f"<td class=small>{'0' if k != 'goal_act' else '1'} (입력 = GT, 정의상)</td>"
                if src == "fair" and v is not None:
                    md = F[grp].get("median", {}).get(k, {}).get(m); n_ = F[grp].get("count", {}).get(k, {}).get(m)
                    return (f"<td>{v:.3f}<div class=small>중앙 {md:.3f}" + (f" · n {n_}" if n_ and n_ < 120 else "") + "</div></td>") if md is not None else f3_(v)
                return f3_(v)
            cells = "".join(cell(m, v) for (m, _), v in zip(ML, vals))
            arrow = "" if d == 0 else ("↑" if d > 0 else "↓")
            p_ = pair(grp, src, k)
            dc = (f"<td>{p_['d']:+.3f}<div class=small>[{p_['ci'][0]:+.3f}, {p_['ci'][1]:+.3f}]</div></td>" if p_ else "<td>–</td>") if d == 0 else dcell(p_, d)
            gtv = F.get(grp, {}).get("mean", {}).get(k, {}).get("gt") if src == "fair" else None
            lab2 = lab + (f" (GT {gtv:.3f})" if gtv is not None and k.startswith("motion") else "")
            b += (f"<tr><td>{html.escape(lab2)} {arrow}</td><td class=small>{SRC[src]}</td>" + cells
                  + ("<td class=small>비교 불가</td>" if trivial else dc) + "</tr>")
        ref = []
        if FV.get(grp, {}).get("gt", {}).get("musiq") is not None:
            ref.append(f"GT MUSIQ {FV[grp]['gt']['musiq']:.3f}")
        for k, n in (("met3r_s1", "s1"), ("met3r", "s4"), ("met3r_s1_t16", "s1 · 16–76"), ("met3r_t16", "s4 · 16–76")):
            v = F.get(grp, {}).get("mean", {}).get(k, {}).get("gt"); md = F.get(grp, {}).get("median", {}).get(k, {}).get("gt")
            c_ = F.get(grp, {}).get("count", {}).get(k, {}).get("gt")
            if v is not None:
                ref.append(f"GT MEt3R {n} 평균 {v:.3f} · 중앙 {md:.3f}" + (f" (n {c_})" if c_ and c_ < 120 else ""))
        P.append(f"<h3>{title}</h3><div class=tw><table>{h}{b}</table></div>"
                 + (f"<div class=mut small>참고: {' · '.join(ref)}</div>" if ref else ""))
    if os.path.exists(f"{GC}/motion_all.json"):
        from scipy import stats as _st
        MA = json.load(open(f"{GC}/motion_all.json"))
        P.append("<h3 id=motion>모든 비교 모델: 움직임 양 vs MEt3R</h3><div class=mut>움직임 = 프레임 간 평균 변화 (0–255, 프레임 1–76)를 각 모델이 보는 영역 "
                 "(그 모델의 gt_crop_box, 없으면 전체)에서 재고, 같은 영역의 GT와 나눈 비율. MEt3R = /cmp 규약의 창별 값 (stride 1; 모델마다 해상도 · 영역이 달라 "
                 "절대값은 참고). ρ = 창별 Spearman (움직임 비율 vs MEt3R), 기울기 = 움직임 비율 +0.1당 MEt3R 변화. 코드: tools/motion_all.py</div>")
        figs, tabs = [], []
        FVJ = f"{GC}/motion_fig_values.json"          # exactly the plotted points (tools/motion_fig.py)
        FV_ = json.load(open(FVJ)) if os.path.exists(FVJ) else {}
        for grp, title in (("synthetic", "Synthetic"), ("real", "Real")):
            d = MA.get(grp, {}); fv = FV_.get(grp)
            if not d or not fv:
                continue
            h = ("<tr><th>모델</th><th>계열</th><th>움직임 / GT</th><th>MEt3R ↓</th><th>값 출처</th>"
                 "<th>참고: 창별 ρ (p) · 전체 프레임</th></tr>")
            b_ = ""
            for q in fv["points"]:
                o = d.get(q["model"], {}); m = q["model"]
                bold = " style='font-weight:600'" if m.startswith("ours") else ""
                rho = (f"{o['rho']:+.3f} ({'&lt;0.001' if o['rho_p'] < 1e-3 else format(o['rho_p'], '.3f')})"
                       if "rho" in o and m not in ("ours_cc", "aether_gtcam", "ours_plan") else "–")
                b_ += (f"<tr{bold}><td>{html.escape(q['label'])}</td><td>{html.escape(q['family'])}</td><td>{q['motion_ratio']:.3f}</td>"
                       f"<td>{q['met3r']:.3f}</td><td class=small>{html.escape(q['source'])}</td><td>{rho}</td></tr>")
            tabs.append(f"<h4>{title} — 모델 간 Spearman {fv['spearman']:.3f}</h4><div class=tw><table>{h}{b_}</table></div>")
        P.append("<div class=mut small>색 = 방법 계열 (진보라 Joint · 마젠타 Pose-then-generate · 라벤더 Plan-then-generate · 연분홍 Video-only), "
                 "속 빈 원 = + GT camera, 기울어진 점선 = 모든 모델에 맞춘 직선, 세로 점선 = GT와 같은 움직임. 그림과 표는 같은 값 (tools/motion_fig.py → "
                 "gtcam_cmp/motion_fig_values.json): Ours는 논문 표 값, + GT camera 두 run은 프레임 16–76 · 공통 crop 값, 나머지는 /cmp 창별 평균.</div>"
                 + (f"<img src='/vcons/motion_met3r.png?v={int(os.path.getmtime(MFIG))}' style='width:100%;max-width:1300px;background:#fff;border-radius:8px' "
                    "alt='motion vs MEt3R scatter'>" if os.path.exists(MFIG) else "")
                 + f"<details><summary>표로 보기</summary>{''.join(tabs)}</details>")
        EXJ = "/root/project/egowam/results_syn/gtcam_cmp/examples/examples.json"
        if os.path.exists(EXJ):
            P.append("<h3 id=met3r_ex>예시: MEt3R이 낮은 건 덜 움직여서</h3><div class=mut>같은 창 · 같은 영역 (512×512 GT의 중앙 띠 512×342) · 같은 77프레임에서 "
                     "GT / Ours / NWM / EgoWM의 MEt3R (stride 1)과 움직임 (GT 대비)을 새로 잼. 고른 기준: Ours 움직임이 GT의 0.9–1.1배이고 같은 영역 MEt3R이 GT와 가까우며, "
                     "NWM · EgoWM은 GT의 0.3배 미만으로 움직이고 MEt3R이 Ours보다 낮은 창 (real은 구조가 뚜렷하고 GT 시점 변화가 큰 창으로 고름: 쇼핑몰 통로 · 거리). "
                     "NWM · EgoWM은 장면이 거의 안 움직이면서 무너지는데도 MEt3R이 GT보다 낮게 나옴. 코드: tools/met3r_examples.py, tools/met3r_example_fig.py</div>")
            _sel = os.path.join(os.path.dirname(EXJ), "selected.json")
            for k in (json.load(open(_sel)) if os.path.exists(_sel) else json.load(open(EXJ))):
                f_ = f"{os.path.dirname(MFIG)}/met3r_example_{k}.png"
                if os.path.exists(f_):
                    P.append(f"<img src='/vcons/met3r_example_{k}.png?v={int(os.path.getmtime(f_))}' style='width:100%;max-width:1300px;background:#fff;"
                             f"border-radius:8px;margin:8px 0' alt='MEt3R example {k}'>")
        P.append("<div class=mut small>움직임이 GT보다 적은 모델일수록 MEt3R이 낮음 (거의 같은 프레임끼리 비교하면 불일치가 생길 여지가 없음). "
                 "MEt3R 단독 순위는 3D 일관성보다 움직임 양을 더 많이 반영하므로, 움직임과 함께 보고해야 함.</div>")
    P.append("<div class=mut small>† 입력 차이: ours는 goal 이미지도 받고 Aether GT 카메라 모드는 받지 않음 → goal 이미지 지표와 화질 (특히 뒤쪽 프레임)은 ours에 유리 "
             "(Aether planning은 goal을 마지막 프레임으로 받음). ‡ action의 뜻이 두 모델에서 다름: ours GT 카메라 run의 action = 주어진 GT 카메라 그대로 "
             "(AF는 정의상 0, goal SR(action)은 정의상 1, VA cam = 주어진 카메라 vs 영상의 DA3 카메라), Aether GT 카메라 run의 action = Aether가 "
             "영상과 함께 다시 생성한 카메라. VA depth = 모델이 낸 depth vs 영상의 DA3 depth. VM은 Aether는 crop된 띠에서, ours는 512×512 전체에서 DA3를 돌린 값. "
             "코드: tools/gtcam_cmp.py, tools/gtcam_fair.py</div>")
    return "".join(P)

GV = "/root/project/egowam/results_syn/goal_validation/goal_aug.json"


def goal_section():
    """Goal-aware supervision & goal appearance augmentation, no retraining (tools/goal_aug_analysis.py)."""
    if not os.path.exists(GV):
        return ""
    o = json.load(open(GV)); A, B, C, D, E = o["A"], o["B"], o["C"], o["D"], o["E"]
    pv = lambda p: "" if p is None else ("&lt;0.001" if p < 1e-3 else f"{p:.3f}")
    P = ["<h2 id=goal>Goal-aware supervision · goal 외형 augmentation (재학습 없이)</h2>"
         "<div class=mut>재학습 없이 할 수 있는 범위: ① 같은 geometry의 goal에 외형만 바꾼 paired 테스트 (ours) "
         "② ours goal-state head의 방향 · 거리 판독이 goal 도달과 얼마나 같이 가는지 (상관) ③ 이미 학습된 ladder run (교란 있음). "
         "방향만 / 거리만 / augmentation 없는 모델의 <b>인과적</b> 비교는 재학습이 필요해 여기에 없음.</div>"]
    # 1 paired appearance test
    r = []
    for a in A:
        d = a["delta"]; good = (d * a["dir"]) > 0
        extra = f"{a['lost']}개 실패로 · {a['won']}개 성공으로 (McNemar p {pv(a['p'])})" if "lost" in a else \
                (f"중앙값 {f3(a['median_orig'])} → {f3(a['median_style'])} (Wilcoxon p {pv(a['p'])})" if "median_orig" in a else "")
        ci = f" [{f3(a['ci'][0])}, {f3(a['ci'][1])}]" if "ci" in a else ""
        r.append(f"<tr><td>{html.escape(a['label'])}</td><td>{f3(a['orig'])}</td><td>{f3(a['style'])}</td>"
                 f"<td style='color:var({'--dn' if good else '--up'})'>{d:+.3f}{ci}</td><td class=small>{extra}</td></tr>")
    P.append("<h3>① 같은 geometry, goal 외형만 변경 (ours avg 7500+8750, synthetic 120, paired)</h3>"
             "<div class=mut>창 · seed · goal 프레임 동일. goal 이미지만 telestyle 편집본 (Qwen-Image-Edit-2509, autumn · spring · photoreal · "
             "golden_hour · dawn 중 창마다 1개)으로 바꾸고 mode 3 (카메라 예측 + goal 변형 플래그). 도착 판정 · goal 지표는 원본 goal 위치 기준. "
             "Δ 괄호 = paired bootstrap 95% CI.</div>"
             "<div class=tw><table><tr><th>지표</th><th>원본 goal</th><th>편집 goal</th><th>Δ</th><th>paired</th></tr>" + "".join(r) + "</table></div>")
    P.append("<div class=note>위치는 대부분 유지: DA3 궤적 기준 SR 변화는 유의하지 않고 (9개 잃고 4개 얻음), endpoint 오차도 CI가 0을 포함. "
             "떨어지는 것은 <b>도착 heading</b>과 action 궤적 기준 지표 (유의). 즉 외형이 바뀌면 '어디로'보다 '어느 방향을 보고 도착하는지'가 흔들림. "
             "주의: 편집 스타일 5종은 학습 TeleStyle과 같은 종류라 ours에게는 학습 분포 안. augmentation 없는 모델과의 비교가 없어 "
             "'augmentation 덕분'이라고는 말할 수 없음 (재학습 필요).</div>")
    # per variant
    rr = "".join(f"<tr><td>{v}</td><td>{b['n']}</td><td>{f3(b['goal_da3_success_orig'])} → {f3(b['goal_da3_success_style'])}</td>"
                 f"<td>{f3(b['goal_da3_final_pos_path_orig'])} → {f3(b['goal_da3_final_pos_path_style'])}</td>"
                 f"<td>{f3(b['goal_da3_final_head_deg_orig'])} → {f3(b['goal_da3_final_head_deg_style'])}</td></tr>" for v, b in B.items())
    P.append("<h3>스타일별</h3><div class=tw><table><tr><th>편집 스타일</th><th>n</th><th>SR (DA3)</th><th>endpoint 오차/경로</th>"
             "<th>도착 heading 오차°</th></tr>" + rr + "</table></div>"
             "<div class=mut small>창마다 스타일이 1개라 스타일 간 비교는 창 난이도와 섞여 있음 (n 16–31). spring에서 SR 하락이 몰려 있고, photoreal은 heading 오차가 가장 크게 늘어남.</div>")
    P.append("<figure class=panel style='margin:8px 0'><img src=/vcons/goalstyle_examples.jpg style='width:100%;max-width:1024px' alt='goal appearance examples'>"
             "<div class=sub>열: 원본 goal · 편집 goal · 마지막 생성 프레임 (원본 goal) · 마지막 생성 프레임 (편집 goal). 앞 5줄 = 스타일별 첫 창 (둘 다 성공), "
             "마지막 줄 = 편집 후 실패한 예. 생성 영상은 대체로 현재 장면의 외형을 유지 (autumn은 낙엽 일부가 옮겨짐 → goal LPIPS 상승 요인)</div></figure>")
    # 2 readout
    def crow(lab, k, unit=""):
        c = C[k]
        return (f"<tr><td>{lab}</td><td>{f3(c['orig'])}{unit}</td><td>{f3(c['style'])}{unit}</td>"
                f"<td>{c['rel_change']:+.1%}</td><td>[{f3(c['ci'][0])}, {f3(c['ci'][1])}]</td><td>{pv(c['p'])}</td></tr>")
    P.append("<h3>② goal-state head의 방향 · 거리 판독</h3><div class=mut>ours는 매 key frame마다 goal까지 [남은 거리, ego 기준 방향]을 예측하도록 "
             "학습됨 (goal aux = 거리 smooth-L1 + 2 × 방향 cosine loss). 저장된 예측 (_pred_goalstate.npy)을 학습 target과 같은 식으로 계산한 정답과 비교. "
             f"정답 기준 = 모델 자신의 예측 카메라 궤적 (pred_poses) / GT 궤적. key frame 1–19 평균, '초기' = 1–4. 시작 시 goal까지 평균 {f3(o['d0_mean'])} m.</div>")
    P.append("<div class=tw><table><tr><th>판독 오차 (평균)</th><th>원본 goal</th><th>편집 goal</th><th>상대 변화</th><th>Δ 95% CI</th><th>Wilcoxon p</th></tr>"
             + crow("방향 오차° · 자기 궤적 기준", "self_dir") + crow("거리 오차 m · 자기 궤적 기준", "self_dist")
             + crow("방향 오차° · 자기 궤적 · 초기", "self_dir_early") + crow("거리 오차 m · 자기 궤적 · 초기", "self_dist_early")
             + crow("방향 오차° · GT 궤적 기준", "gt_dir") + crow("거리 오차 m · GT 궤적 기준", "gt_dist") + "</table></div>")
    # 3 which goes with success
    def drow(lab, k):
        d = D[k]
        return (f"<tr><td>{lab}</td><td>{f3(d['sr_tertile'][0])} / {f3(d['sr_tertile'][1])} / {f3(d['sr_tertile'][2])}</td>"
                f"<td>{f3(d['err_succ'])} vs {f3(d['err_fail'])}</td><td>{pv(d['mwu_p'])}</td><td>{f3(d['rho_pos'])}</td><td>{f3(d['rho_head'])}</td></tr>")
    P.append("<h3>③ 방향과 거리 중 무엇이 goal 도달과 같이 가는가 (상관, 인과 아님)</h3>"
             "<div class=tw><table><tr><th>판독 오차 (GT 궤적 기준)</th><th>오차 3분위별 SR (작음 / 중간 / 큼)</th><th>오차: 성공 vs 실패</th>"
             "<th>Mann–Whitney p</th><th>ρ (endpoint 오차)</th><th>ρ (heading 오차)</th></tr>"
             + drow("원본 goal · 방향", "orig_dir") + drow("원본 goal · 거리", "orig_dist")
             + drow("원본 goal · 방향 · 초기 (궤적 갈라지기 전)", "orig_dir_early") + drow("원본 goal · 거리 · 초기", "orig_dist_early")
             + drow("편집 goal · 방향", "style_dir") + drow("편집 goal · 거리", "style_dist") + "</table></div>")
    dd, dt = D["delta_dir"], D["delta_dist"]
    P.append(f"<div class=note>방향 판독이 정확한 창은 거의 다 도착 (하위 1/3 SR {f3(D['orig_dir']['sr_tertile'][0])} vs 상위 1/3 "
             f"{f3(D['orig_dir']['sr_tertile'][2])}). 궤적이 갈라지기 전인 초기 판독에서도 같은 경향 ({f3(D['orig_dir_early']['sr_tertile'][0])} vs "
             f"{f3(D['orig_dir_early']['sr_tertile'][2])}). 거리 판독 오차는 원본 goal에서 SR과 관계 없음 (p {pv(D['orig_dist']['mwu_p'])}). "
             f"외형 변경으로 성공을 잃은 창은 방향 판독 오차가 평균 {dd['lost']:+.3f}° 늘었고, 성공을 유지한 창은 {dd['kept']:+.3f}°. "
             "→ goal aux 중 도달과 같이 움직이는 것은 방향 신호이고, 거리 신호는 원본 goal에서는 SR과 관계 없음. "
             "단, 상관이라 '방향 supervision이 SR을 올린다'는 인과 주장은 direction-only / distance-only 재학습이 있어야 가능.</div>")
    # 4 ladder
    rows = "".join(f"<tr><td>{html.escape(e['label'])}</td><td>{f3(e['goal_da3_success'])}</td><td>{f3(e['goal_da3_final_pos_path'])}</td>"
                   f"<td>{f3(e['goal_da3_final_head_deg'])}</td><td class=small>{html.escape(e['note'])}</td></tr>" for e in E)
    P.append("<h3>④ 기존 학습 run: goal aux 유무 (교란 있음)</h3><div class=tw><table><tr><th>run</th><th>SR (DA3)</th><th>endpoint 오차/경로</th>"
             "<th>도착 heading 오차°</th><th>다른 차이</th></tr>" + rows + "</table></div>"
             "<div class=mut small>abl2 → abl4에서 SR 0.542 → 0.667이지만 step · depth head · raymap convention이 함께 바뀜. "
             "goal aux만 뺀 run이 아니라 rebuttal의 인과 근거로는 약함. 코드: evalwam/tools/goal_aug_analysis.py · 결과: results_syn/goal_validation/goal_aug.json</div>")
    return "".join(P)


TD = "/root/project/egowam/results_syn/traversal_difficulty/difficulty.json"


def traversal_section():
    """Traversal difficulty: ours vs Aether / abl1 / all baselines by GT difficulty tertile (tools/traversal_difficulty.py)."""
    if not os.path.exists(TD):
        return ""
    R = json.load(open(TD)); M = R["meta"]["models"]; AE = M.index("Aether")
    pv = lambda p: "&lt;0.001" if p < 1e-3 else f"{p:.3f}"
    FAC = [("dist", "시작–goal 직선거리 (m)"), ("rot", "누적 회전 (°)"), ("sharp", "가장 급한 회전 (° / 16 프레임)"),
           ("unseen", "처음 화면에 없던 영역 비율 (마지막 프레임)")]
    MET = [("goal_succ", "Goal SR ↑", 1), ("coh_rot", "VA · coherence RotErr° ↓", -1), ("cam_rot", "VM · 영상 카메라 RotErr° ↓", -1),
           ("act_rot", "AF · action RotErr° ↓", -1)]
    P = ["<h2 id=trav>Traversal 난이도별 분석 (재학습 없음)</h2>"
         "<div class=mut>창마다 GT만으로 난이도를 계산해 (모든 모델에 같은 값) 3등분하고, 같은 창끼리 ours − baseline 차이를 봄. "
         "baseline 지표 = /cmp와 같은 창별 값 (논문 표 규약). 차이 괄호 = paired bootstrap 95% CI, ρ = 난이도와 창별 차이의 Spearman "
         "(Goal SR은 +면 ours 우위가 커짐, RotErr는 −면 ours 우위가 커짐). 구간당 40개 창.</div>"]
    def gcell(g, d):
        if g is None or g["ci"][0] is None:
            return "<td>–</td>"
        good = g["mean"] * d > 0; sig = (g["ci"][0] > 0) == (g["ci"][1] > 0)
        return (f"<td style='color:var({'--dn' if good else '--up'});{'font-weight:600' if sig else ''}'>{g['mean']:+.3f}"
                f"<div class=small>[{g['ci'][0]:+.3f}, {g['ci'][1]:+.3f}]</div></td>")
    for grp, title in (("syn", "synthetic · valbench5s 120"), ("real", "real · LookOut A+B+C 120")):
        G = R[grp]
        P.append(f"<h3>{title}</h3>")
        for fac, flab in FAC:
            F = G["factors"][fac]
            rg = " / ".join(f"{a:.3f}–{b:.3f}" for a, b in F["range"])
            h = ("<tr><th rowspan=2>지표</th><th colspan=3>ours</th><th colspan=3>Aether</th><th colspan=3>abl1 (compact action)</th>"
                 "<th colspan=3>ours − Aether</th><th>추세</th><th colspan=3>ours − abl1</th><th>추세</th></tr><tr>"
                 + "<th>하</th><th>중</th><th>상</th>" * 4 + "<th>ρ (p)</th>" + "<th>하</th><th>중</th><th>상</th>" + "<th>ρ (p)</th></tr>")
            b = ""
            for k, lab, d in MET:
                mm = F["mean"][k]
                cells = "".join(f"<td>{f3(x)}</td>" for j in (0, AE, 1) for x in mm[j])
                for j in (AE, 1):
                    gp = F["gap"][k].get(str(j)); tr = F["trend"][k].get(str(j))
                    cells += "".join(gcell(x, d) for x in gp) if gp else "<td>–</td>" * 3
                    cells += f"<td>{tr['rho']:+.3f} ({pv(tr['p'])})</td>" if tr else "<td>–</td>"
                b += f"<tr><td>{lab}</td>{cells}</tr>"
            P.append(f"<div class=mut><b>{flab}</b> · 하 / 중 / 상 구간 = {rg}</div><div class=tw><table>{h}{b}</table></div>")
        rows = ""
        for k, lab, d in MET:
            for j in (AE, 1):
                v = G["joint"].get(k, {}).get(str(j))
                if v:
                    rows += (f"<tr><td>{lab}</td><td>{html.escape(M[j])}</td><td>{v['b_dist']:+.3f} [{v['ci_dist'][0]:+.3f}, {v['ci_dist'][1]:+.3f}]</td>"
                             f"<td>{v['b_rot']:+.3f} [{v['ci_rot'][0]:+.3f}, {v['ci_rot'][1]:+.3f}]</td></tr>")
        P.append(f"<div class=mut>거리와 회전을 함께 넣은 회귀 (ours − baseline 차이 ~ z(거리) + z(누적 회전), 계수 = 1 sd당 변화, 95% CI). "
                 f"거리–회전 Spearman {G['corr_dist_rot']:+.3f}.</div><div class=tw><table><tr><th>지표</th><th>vs</th><th>거리 계수</th><th>회전 계수</th></tr>{rows}</table></div>")
        # all baselines, the two strongest factors, SR + VM
        ah = "<tr><th>baseline</th>" + "".join(f"<th>{fl} · {ml}<div class=small>하 → 상</div></th>" for fac, fl in (("rot", "누적 회전"), ("unseen", "안 보이던 영역"))
                                               for k, ml, _ in (MET[0], MET[2])) + "</tr>"
        ar = ""
        for j in range(1, len(M)):
            c = ""
            for fac in ("rot", "unseen"):
                for k, _, d in (MET[0], MET[2]):
                    gp = G["factors"][fac]["gap"][k].get(str(j))
                    c += (f"<td>{gp[0]['mean']:+.3f} → <b style='color:var({'--dn' if gp[2]['mean'] * d > 0 else '--up'})'>{gp[2]['mean']:+.3f}</b></td>"
                          if gp else "<td>–</td>")
            ar += f"<tr><td>{html.escape(M[j])}</td>{c}</tr>"
        P.append(f"<details><summary>모든 baseline: ours − baseline, 쉬운 1/3 → 어려운 1/3</summary><div class=tw><table>{ah}{ar}</table></div></details>")
    P.append("<div class=note><b>요약</b><ul>"
             "<li><b>synthetic, Aether 대비</b>: 처음 화면에 없던 영역이 많을수록, 급회전이 클수록 ours 우위가 커짐 — 영상 카메라 RotErr 차이 "
             "+0.902 → −5.288 (ρ −0.403, p&lt;0.001), action RotErr +0.484 → −4.464 (ρ −0.316), Goal SR −0.100 → +0.175 (ρ +0.244, p 0.007). "
             "쉬운 1/3에서는 Aether와 비슷하거나 약간 뒤짐.</li>"
             "<li><b>synthetic, abl1 대비</b>: 모든 구간에서 ours가 앞서지만 격차가 난이도에 따라 유의하게 커지지는 않음.</li>"
             "<li><b>VA (coherence)</b>: 두 세트 모두 모든 구간에서 ours가 Aether보다 1.071–2.788° 낮음 — 난이도와 무관한 일정한 우위.</li>"
             "<li><b>real</b>: 가설과 반대. 회전이 큰 1/3에서는 모든 모델의 SR이 낮고 (ours 0.125) ours의 SR 우위가 사라짐 (Aether 대비 +0.250 → −0.100, ρ −0.242). "
             "영상 카메라 RotErr는 real에서 Aether가 전 구간 더 좋고 회전이 클수록 격차가 커짐.</li>"
             "<li>시작–goal 거리: 5초 창이라 범위가 좁아 (synthetic 3.0–7.3 m) 뚜렷한 추세 없음. 회귀로 거리와 회전을 분리해도 거리 계수는 모두 CI가 0을 포함.</li>"
             "</ul>정리하면 \"어려운 traversal에서 우위가 커진다\"는 synthetic의 시점 변화 (안 보이던 영역 · 급회전)에서 Aether 대비로만 지지되고, real에서는 지지되지 않음. "
             "MEt3R · LPIPS는 ours와 baseline의 계산 규약 (stride · 해상도)이 달라 이 표에서 뺌. FVD는 구간당 40개라 신뢰할 수 없어 뺌. "
             "코드: evalwam/tools/traversal_difficulty.py · 결과: results_syn/traversal_difficulty/</div>")
    return "".join(P)


def page():
    S = _load()
    P = [f"<!doctype html><html lang=ko><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
         f"<title>Rebuttal Experiments</title><style>{CSS}</style></head><body><main>",
         "<h1>Rebuttal 실험</h1><div class=mut>Video–Action Coherence 지표 검증 (controlled perturbation · DA3 측정 편향) · "
         "<a href=#vcons>Direct vs Residual 메커니즘</a> · <a href=#goal>Goal supervision · 외형 augmentation</a> · <a href=#trav>Traversal 난이도</a> · <a href=#vdesign>Residual 설계 선택</a> · <a href=#gtcam>GT 카메라: ours vs Aether</a> · <a href=/ablation>/ablation</a> · <a href=/>홈</a></div>"]
    if not S:
        P.append("<p>결과 없음: tools/coherence_perturb.py 를 먼저 실행</p></main></body></html>")
        return "".join(P).encode()
    sy, re_ = S.get("syn"), S.get("real")
    fl = lambda s, m: s["families"]["rot_drift"]["0"][m]["mean"]
    md = lambda s, m: s["families"].get("model", {}).get("1", {}).get(m, {}).get("mean")
    P.append("<div class=tl>")
    for k, name, _ in SETS:
        s = S.get(k)
        if not s:
            continue
        P.append(f"<div class=tile><span>{name} · GT 영상 vs GT 궤적 (측정 하한)</span><b>RotErr {fl(s, 'coh_roterr'):.3f}</b>"
                 f"<span>ours {f3(md(s, 'coh_roterr'))} · 뒤섞은 궤적 {s['families']['swap']['1']['coh_roterr']['mean']:.3f}</span></div>")
    F = focal()
    if F:
        P.append(f"<div class=tile><span>DA3 초점거리 추정 (GT 영상 {F['n']}개)</span><b>fx {F['fx']:.3f}</b>"
                 f"<span>GT {F['gfx']:.3f} ({F['w']}px) → 회전 {F['ratio']:.3f}배로 과소추정</span></div>")
    P.append("</div>")

    P.append("<div class=note><b>요약</b><ul>"
             "<li>지표는 기하학적 불일치가 커질수록 단조 증가: 과회전 drift·회전 노이즈·평행이동 노이즈·속도 프로파일·경로 휨 모두 "
             "크기에 따라 증가하고, 궤적을 뒤섞으면 측정 하한의 약 5–7배.</li>"
             "<li>Sim3 정렬이 흡수하는 변화(전체 속도 배율, 상수 오프셋, <b>일정한 옆미끄러짐</b>)에는 반응하지 않음 — 앞의 둘은 의도된 불변성, "
             "옆미끄러짐(heading과 이동 방향의 어긋남)은 사각지대.</li>"
             "<li><b>DA3가 회전을 약 0.8배로 작게 복원</b> (초점거리를 약 35% 크게 추정). 그래서 회전을 덜 하는 action은 일정 범위까지 "
             "점수가 오히려 좋아짐 (synthetic ~5°, real ~20°). real에서는 ours(카메라 예측)가 GT 자신보다 낮은 점수.</li></ul></div>")

    P.append("<h2>1 · 측정 하한과 ours</h2><div class=mut>GT 영상의 DA3 카메라 vs GT 궤적 = 완벽히 일치하는 action이 받을 점수. "
             "같은 계산(cam77: 77프레임, planar, Sim3 + 회전 정렬)으로 ours를 계산하면 표의 coherence 값과 동일.</div>")
    P.append(table(S, [("GT vs GT (측정 하한)", "rot_drift", 0), ("ours · avg 7500+8750 (자기 action vs 자기 영상)", "model", 1),
                       ("뒤섞은 궤적 (다른 창의 GT action)", "swap", 1)]))

    P.append("<h2>2–3 · Perturbation 크기에 따른 변화</h2><div class=mut>GT action에 perturbation을 넣고 GT 영상의 DA3 카메라와 비교. "
             "채운 점 = 측정 하한보다 유의하게 큼 (paired Wilcoxon p &lt; 0.01), 빈 점 = 유의하지 않음, 점선 = 측정 하한.</div>")
    P.append("<div class=lg>" + "".join(f"<span><i style='background:var(--s{i + 1})'></i>{n}</span>" for i, (k, n, _) in enumerate(SETS) if k in S) + "</div>")
    P.append("<div class=grid>")
    P.append(panel(S, "회전 drift · 더 도는 방향", "마지막 프레임 heading이 θ만큼 더 돔", "rot_drift_out", [0, 1, 2, 3, 5, 10, 15, 20, 30, 45], "coh_roterr", "θ = {}°"))
    P.append(panel(S, "회전 drift · 덜 도는 방향 (DA3 편향 쪽)", "마지막 프레임 heading이 θ만큼 덜 돔", "rot_drift_in", [0, 1, 2, 3, 5, 10, 15, 20, 30, 45], "coh_roterr", "θ = {}°"))
    P.append(panel(S, "회전 노이즈", "프레임마다 dyaw += N(0, σ)", "rot_noise", [0, 0.25, 0.5, 1, 2, 4], "coh_roterr", "σ = {}°/frame"))
    P.append(panel(S, "평행이동 노이즈", "(dx, dy) += N(0, c · 평균 step)", "trans_noise", [0, 0.05, 0.1, 0.2, 0.4, 0.8], "coh_ate_path", "c = {}"))
    P.append(panel(S, "경로 휨", "옆미끄러짐이 0 → 2ρ로 증가", "lat_bend", [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3], "coh_ate_path", "ρ = {}"))
    P.append(panel(S, "속도 프로파일", "같은 경로, 다른 타이밍 (가속)", "speed_prof", [0, 0.1, 0.25, 0.5, 0.75], "coh_ate_path", "β = {}"))
    P.append("</div>")
    P.append("<h3>표 · 단계별 값</h3>")
    rows = []
    for lab, fam, mags, u in [("과회전 drift", "rot_drift_out", [1, 3, 5, 10, 20, 45], "°"), ("저회전 drift", "rot_drift_in", [1, 3, 5, 10, 20, 45], "°"),
                              ("회전 노이즈", "rot_noise", [0.25, 1, 4], "°/frame"), ("평행이동 노이즈", "trans_noise", [0.05, 0.2, 0.8], ""),
                              ("경로 휨", "lat_bend", [0.05, 0.1, 0.3], ""), ("속도 프로파일", "speed_prof", [0.1, 0.5, 0.75], "")]:
        rows += [(f"{lab} {m}{u}", fam, m) for m in mags]
    P.append(table(S, [("측정 하한", "rot_drift", 0)] + rows))
    P.append("<div class=mut small>빨강 = 측정 하한보다 유의하게 큼, 초록 = 측정 하한보다 작음 (편향 때문에 '더 일치'로 보이는 경우).</div>")

    P.append("<h2>불변성 · 사각지대</h2><div class=mut>Sim3 정렬(위치)과 회전 정렬을 따로 하므로 아래 변화는 점수에 영향이 없음.</div>")
    P.append(table(S, [("측정 하한", "rot_drift", 0), ("전체 속도 ×0.5", "speed_scale", 0.5), ("전체 속도 ×2", "speed_scale", 2),
                       ("일정한 옆미끄러짐 ρ=0.1", "lat_drift", 0.1), ("일정한 옆미끄러짐 ρ=0.3", "lat_drift", 0.3)]))
    P.append("<div class=mut small>일정한 옆미끄러짐 = 카메라는 정면을 보는데 비스듬히 이동. 위치 궤적 전체가 회전·확대된 것과 같아서 Sim3가 흡수함.</div>")

    P.append("<h2>4 · DA3 측정 편향</h2>")
    mrows = []
    for k, name, _ in SETS:
        s = S.get(k)
        if not s:
            continue
        m = s["measurement"]; d6 = m["da3_vs_gt_6dof"]; ms = s.get("model_net_yaw_slope_vs_gt", {})
        mrows.append(f"<tr><td>{name}</td><td>{d6['da3_roterr']:.3f}</td><td>{d6['da3_ate_path']:.3f}</td><td>{d6['da3_rpe_rot']:.3f}</td>"
                     f"<td>{m['net_yaw_ratio_median_turns_gt20']:.3f} (n={m['n_turns_gt20']})</td><td>{m['net_yaw_corr']:.3f}</td>"
                     f"<td>{f3(ms.get('pred_action'))}</td><td>{f3(ms.get('da3_generated'))}</td></tr>")
    P.append("<div class=tw><table><tr><th>세트</th><th>DA3 vs GT RotErr (6-DoF)</th><th>ATE/Len</th><th>RPE rot</th>"
             "<th>net yaw 비율 DA3/GT (20° 넘는 회전)</th><th>net yaw 상관</th><th>ours action / GT 기울기</th><th>DA3(ours 영상) / GT 기울기</th></tr>"
             + "".join(mrows) + "</table></div>")
    if F:
        P.append(f"<div class=note>DA3 초점거리: GT 영상 {F['n']}개 평균 fx {F['fx']:.3f} (sd {F['fx_sd']:.3f}) vs GT {F['gfx']:.3f} "
                 f"({F['w']}px 처리 해상도 기준, 시야각 90°) → 약 {F['fx'] / F['gfx'] - 1:.0%} 크게 추정. 20° 넘는 회전 {F['nb']}개에서 "
                 f"net yaw 비율 {F['ratio']:.3f}, 초점거리로 1차 보정하면 {F['ratio_corr']:.3f}. GT intrinsics만 넣으면 DA3가 쓰지 않아 "
                 f"결과 동일 ({F['ratio_K']:.3f}, extrinsics 없이 intrinsics만 넣은 경우).</div>")
    P.append("<div class=mut>DA3는 GT 영상과 생성 영상에 똑같이 적용되므로 같은 창에서 모델끼리 비교(paired)하면 편향이 상쇄되는 방향이지만, "
             "절대값은 '회전을 덜 하는 action'에 유리. ours의 예측 action도 GT보다 덜 돔(위 표) → real에서 ours가 측정 하한보다 낮은 이유.</div>")

    P.append("<h2>움직임 크기별 (synthetic)</h2><div class=mut>정적·작은 움직임에서 점수가 좋아지는지: 창을 총 회전량 / 경로 길이로 3등분.</div>")
    if sy:
        st = sy["stratified"]
        r = []
        for key, lab in [("abs_yaw_total_t1", "총 회전 하위 1/3"), ("abs_yaw_total_t2", "총 회전 중간"), ("abs_yaw_total_t3", "총 회전 상위 1/3"),
                         ("path_len_t1", "경로 길이 하위 1/3"), ("path_len_t2", "경로 길이 중간"), ("path_len_t3", "경로 길이 상위 1/3")]:
            v = st[key]
            r.append(f"<tr><td>{lab}</td><td>{v['range'][0]:.3f} – {v['range'][1]:.3f}</td><td>{v['n']}</td><td>{v['floor_roterr']:.3f}</td>"
                     f"<td>{v['drift10_roterr']:.3f}</td><td>{v['floor_ate_path']:.3f}</td><td>{v['slip5_ate_path']:.3f}</td></tr>")
        P.append("<div class=tw><table><tr><th>구간</th><th>범위 (° 또는 m)</th><th>n</th><th>하한 RotErr</th><th>10° drift RotErr</th>"
                 "<th>하한 ATE/Len</th><th>5% 일정 옆미끄러짐 ATE/Len</th></tr>" + "".join(r) + "</table></div>")
        P.append("<div class=mut small>회전이 적은 창일수록 하한이 낮음 → 같은 창끼리 비교해야 공정. 회전이 적은 창에서도 10° drift를 넣으면 "
                 "크게 오름 (민감도 유지). ATE/Len은 경로 길이에 대해 평평.</div>")

    P.append(vcons_section())
    P.append(goal_section())
    P.append(traversal_section())
    P.append(vdesign_section())
    P.append(gtcam_section())
    P.append("<h2>방법 (coherence 검증)</h2><div class=mut small><ul>"
             "<li>영상 = GT 영상, 카메라 = DA3-LARGE-1.1 (ref view = first, suite와 동일), 파일 viz_da3_fix/sample_NN_da3_gt_poses.npy</li>"
             "<li>action = GT 궤적 → 프레임별 ego action (dx, dy, dyaw) → GT 시작 pose에서 적분. perturbation은 action에만 넣음</li>"
             "<li>지표 = cam77.py와 동일 (77프레임 planar x, y, yaw · eval_cam_robust.cam_metrics). CamMC는 미터 단위 (표의 cam77은 raw DA3 단위)</li>"
             "<li>노이즈 계열은 seed 5개 평균 · 유의성 = paired Wilcoxon (측정 하한 대비, 단측) · 단조성 = sample별 Spearman ρ 평균</li>"
             "<li>코드: evalwam/tools/coherence_perturb.py, da3_focal_check.py · 결과: results_syn/coherence_validation/</li></ul></div>")
    P.append(f"<script>{JS}</script></main></body></html>")
    return "".join(P).encode()
