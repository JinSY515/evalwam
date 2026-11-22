#!/usr/bin/env python3
"""/vcons/why.html: why residual beats direct, loss-only (why_residual.py outputs) on valbench 120 + LookOut A/B/C,
+ on valbench generated videos: does each loss value track the real error (vcons_cmp.py npz)?"""
import glob, html, json, os, shutil, sys
import numpy as np
from PIL import Image, ImageDraw

D = "/root/project/egowam/results_syn/goalprog_mode_real6_nomr_hg0_grad0.5_direct/vcons_cmp"
PG = f"{D}/page"
# two sets: synthetic = valbench5s 120 (CARLA), real = LookOut A+B+C pooled (39 + 46 + 35 = 120, exclusion-filtered)
SETS = [("synthetic", "synthetic · valbench5s 120 (CARLA, GT metric depth)", [(f"{D}/why", "w")]),
        ("real", "real · LookOut A+B+C 120 (실제 Aria, unseen, DA3 depth)", [(f"{D}/why_lookout", x) for x in "ABC"])]
VAR = [("GT", "GT"), ("blur2", "공간 흐림 σ2"), ("blur4", "공간 흐림 σ4"), ("tavg5", "시간 평균 5프레임"), ("tavg9", "시간 평균 9프레임"),
       ("still", "정지 (모든 프레임 = 0번)"), ("gen_res", "생성 · residual ck 2250"), ("gen_dir", "생성 · direct ck 2000")]
f3 = lambda v: "–" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.3f}"


def load(d, tag):
    out = []
    for f in sorted(glob.glob(f"{d}/{tag}[0-9][0-9][0-9].json")):
        r = dict(json.load(open(f)), _d=d, _tag=tag)
        if os.path.exists(f[:-5] + "_q.json"):          # SSIM / LPIPS of the drift end points (why_quality.py)
            r["q"] = json.load(open(f[:-5] + "_q.json"))
        out.append(r)
    return out


def load_set(src):
    return [r for d, tag in src for r in load(d, tag)]


def sheet(d, tag, i, out):
    rows = [("GT", "gt"), ("GT through VAE only", "gtvae"), ("direct-optimised from GT", "direct"), ("residual-optimised from GT", "residual")]
    W = 220; c = Image.new("RGB", (W * 5, W * 4), "white"); dr = ImageDraw.Draw(c)
    for r, (lab, k) in enumerate(rows):
        for j, f in enumerate((0, 20, 40, 60, 76)):
            p = f"{d}/{tag}{i:03d}_{k}_f{f:02d}.jpg"
            if os.path.exists(p):
                c.paste(Image.open(p).resize((W, W)), (j * W, r * W))
        dr.rectangle((0, r * W, 180, r * W + 14), fill="black"); dr.text((3, r * W + 1), lab, fill="white")
    c.save(out, quality=88)


SEC = []
for key, title, src in SETS:
    R = load_set(src)
    if not R:
        SEC.append(f"<h2>{html.escape(title)}</h2><div class=mut>계산 중</div>"); continue
    n = len(R)
    T = [f"<h2>{html.escape(title)} · {n}개</h2><h3>① 순위: GT를 가장 좋게 보는가</h3><table><tr><th>영상</th><th>direct loss</th>"
         "<th>direct가 GT보다 낮게(좋게) 본 비율</th><th>residual loss</th><th>residual이 GT보다 낮게 본 비율</th><th>latent 오차 |x−z|</th><th>PSNR vs GT</th><th>선명도 ÷ GT</th></tr>"]
    for k, lab in VAR:
        rs = [r["rank"][k] for r in R if k in r["rank"]]
        if not rs:
            continue
        g = [r["rank"]["GT"] for r in R if k in r["rank"]]
        dl = np.array([x["direct"] for x in rs]); rl = np.array([x["residual"] for x in rs])
        bd = np.mean(dl < np.array([x["direct"] for x in g])) if k != "GT" else None
        br = np.mean(rl < np.array([x["residual"] for x in g]) - 1e-9) if k != "GT" else None
        ps = [x["psnr"] for x in rs if x["psnr"] is not None]
        hi = " style='background:var(--bad)'" if bd is not None and bd > 0.5 else ""
        T.append(f"<tr><td>{html.escape(lab)}</td><td>{f3(dl.mean())}</td><td{hi}>{f3(bd)}</td><td>{f3(rl.mean())}</td><td>{f3(br)}</td>"
                 f"<td>{f3(np.mean([x['lat_err'] for x in rs]))}</td><td>{f3(np.mean(ps) if ps else None)}</td><td>{f3(np.mean([x['sharp_ratio'] for x in rs]))}</td></tr>")
    T.append("</table>")
    dd = [r["drift"]["direct"] for r in R]; rd = [r["drift"]["residual"] for r in R]
    qm = lambda v, k: np.mean([r["q"][v][k] for r in R if "q" in r]) if any("q" in r for r in R) else None
    mean = lambda L, k: np.mean([x[k] for x in L if x.get(k) is not None]) if any(x.get(k) is not None for x in L) else None
    T.append("<h3>② drift: GT에서 출발해 loss 하나만 최소화</h3><table><tr><th>loss</th><th>loss 시작 → 끝</th><th>GT와 latent 거리 (끝)</th>"
             "<th>디코드 PSNR vs GT</th><th>SSIM</th><th>LPIPS</th><th>선명도 ÷ GT</th></tr>"
             f"<tr><td>direct</td><td>{f3(mean(dd, 'loss_start'))} → {f3(mean(dd, 'loss_end'))}</td><td>{f3(mean(dd, 'lat_dist_end'))}</td>"
             f"<td>{f3(mean(dd, 'psnr_vs_gt'))}</td><td>{f3(qm('direct', 'ssim'))}</td><td>{f3(qm('direct', 'lpips'))}</td><td>{f3(mean(dd, 'sharp_ratio'))}</td></tr>"
             f"<tr><td>residual</td><td>{f3(mean(rd, 'loss_start'))} → {f3(mean(rd, 'loss_end'))}</td><td>{f3(mean(rd, 'lat_dist_end'))}</td>"
             f"<td>{f3(mean(rd, 'psnr_vs_gt'))}</td><td>{f3(qm('residual', 'ssim'))}</td><td>{f3(qm('residual', 'lpips'))}</td><td>{f3(mean(rd, 'sharp_ratio'))}</td></tr>"
             f"<tr><td class=mut>참고: GT를 VAE만 거친 것</td><td></td><td>0.000</td><td>{f3(mean(dd, 'psnr_gt_roundtrip'))}</td><td>{f3(qm('gtvae', 'ssim'))}</td><td>{f3(qm('gtvae', 'lpips'))}</td><td></td></tr></table>"
             "<div class=mut>PSNR = 77프레임 전체, SSIM · LPIPS(alex) = 20·40·60·76번 프레임 (0번은 GT로 고정)</div>")
    # two example sheets: largest and median drift
    order = sorted(R, key=lambda r: -r["drift"]["direct"]["lat_dist_end"])
    for r in (order[0], order[len(order) // 2]):
        fn = f"why_{r['_tag']}{r['idx']:03d}.jpg"
        sheet(r["_d"], r["_tag"], r["idx"], f"{PG}/{fn}")
        T.append(f"<div class=mut>{'' if key == 'synthetic' else 'LookOut ' + r['_tag'] + ' · '}{html.escape(os.path.basename(r['seq']))} · start {r.get('start', '')} · direct drift: loss "
                 f"{f3(r['drift']['direct']['loss_start'])} → {f3(r['drift']['direct']['loss_end'])}, PSNR {f3(r['drift']['direct'].get('psnr_vs_gt'))}</div>"
                 f"<img src='{fn}' loading=lazy>")
    SEC.append("".join(T))

# ---------------- figures + interpretation ----------------
import why_figs as WF  # noqa: E402
FS = [(k, load_set(src)) for k, _, src in SETS]
FS = [(t, R) for t, R in FS if R]
WF.landscape(FS, f"{PG}/why_landscape.png"); WF.per_sample(FS, f"{PG}/why_persample.png")
WF.drift(FS, f"{PG}/why_drift.png"); WF.quality_bars(FS, f"{PG}/why_psnr.png")
fr = {t: np.mean([r["rank"]["tavg5"]["direct"] < r["rank"]["GT"]["direct"] for r in R]) for t, R in FS}
fl = {t: np.median([r["rank"]["tavg5"]["direct"] / r["rank"]["GT"]["direct"] for r in R]) for t, R in FS}
ps = {t: (np.mean([r["drift"]["direct"]["psnr_vs_gt"] for r in R]), np.mean([r["drift"]["residual"]["psnr_vs_gt"] for r in R])) for t, R in FS}
le = {t: (np.mean([r["drift"]["direct"]["loss_start"] for r in R]), np.mean([r["drift"]["direct"]["loss_end"] for r in R]),
          np.mean([r["drift"]["direct"]["lat_dist_end"] for r in R])) for t, R in FS}
qs = {t_: (np.mean([r["q"]["direct"]["ssim"] for r in R if "q" in r]), np.mean([r["q"]["residual"]["ssim"] for r in R if "q" in r])) for t_, R in FS}
ql = {t_: (np.mean([r["q"]["direct"]["lpips"] for r in R if "q" in r]), np.mean([r["q"]["residual"]["lpips"] for r in R if "q" in r])) for t_, R in FS}
rng_ = lambda dct, i=None: f"{min(v if i is None else v[i] for v in dct.values()):.3f}~{max(v if i is None else v[i] for v in dct.values()):.3f}"
ALLR = [r for _, R in FS for r in R]
_fl = np.array([r["rank"]["GT"]["direct"] for r in ALLR]); _rz = np.array([r["rank"]["GT"]["residual"] for r in ALLR])
_t0 = np.array([r["drift"]["direct"]["trace"][0][1] for r in ALLR]); _t1 = np.array([r["drift"]["direct"]["trace"][1][1] for r in ALLR])
_le = np.array([r["drift"]["direct"]["loss_end"] for r in ALLR]); _rd = np.array([r["drift"]["residual"]["lat_dist_end"] for r in ALLR])
LOGIC = f"""<h2>논리: 왜 direct가 residual보다 나쁠 수밖에 없는가</h2>
<div class=box><b>표기</b> 영상 latent <code>x = (x_1..x_T)</code>, 정답 <code>z</code>, 오차 <code>e = x − z</code>.
GT depth·pose로 만든 warp <code>W_(s→t)</code>는 x에 대해 <b>선형</b>(값을 옮기기만 함). 모든 학습 pair를 모은 선형 연산 
<code>(A x)_(t,s) = M ⊙ (x_t − W_(s→t) x_s)</code> (M = GT 가시성 마스크, x와 무관).<br>
그러면 두 loss는 <code>L_D(x) = ‖A x‖₁</code> (direct), <code>L_R(x) = ‖A (x − z)‖₁ = ‖A e‖₁</code> (residual).</div>

<div class=box style='margin-top:8px'><b>① residual의 최소점은 정답이다.</b> <code>L_R ≥ 0</code>이고 <code>L_R(z) = ‖A·0‖ = 0</code> → z는 L_R의 전역 최소점.
생성 loss(denoising)도 정답 z를 향하므로 <code>L_gen + λ·L_R</code>을 최소화해도 정답이 그대로 해 — <b>λ를 얼마로 하든 정답을 옮기지 않음</b>.
<br><span class=mut>실측 synthetic 120 + real 120 = 240개 창: L_R(GT) 최대 {_rz.max():.3f}, GT에서 L_R만 최소화하면 이동 거리 최대 {_rd.max():.3f}.</span></div>

<div class=box style='margin-top:8px'><b>② direct의 최소점은 정답이 아니다.</b> L_D는 볼록(선형 연산의 노름)이고 최솟값 0은 <code>A x = 0</code>인 영상,
즉 "모든 pair에서 x_t = W(x_s)"인 <b>완벽히 warp-일관된 영상</b>에서 나옴 (예: 무늬 없는 평탄한 영상 — 옮겨도 같음).
반면 정답의 점수는 <code>L_D(z) = ‖A z‖ = floor</code>이고, 실제 영상에서 floor > 0 이다:
<code>A z = 0</code>이려면 정지·램버시안 장면 + 완벽한 depth/pose + 시점에 따른 반사·노출 변화 없음 + VAE latent가 warp와 교환됨(실제로는 아님) + 움직이는 물체 없음이 모두 필요.
<br><span class=mut>실측: floor > 0인 창 {(_fl > 0).sum()}/{len(_fl)} (최소 {_fl.min():.3f}, 중앙값 {np.median(_fl):.3f}). 
direct만 최소화하면 floor의 {np.median(_le / _fl):.3f}배(중앙값)까지 내려감 → 정답보다 훨씬 "좋은" 해가 존재.</span></div>

<div class=box style='margin-top:8px'><b>③ 그래서 direct는 정답 위에서도 출력을 민다.</b> L_D의 정답에서의 (부분)기울기 <code>g = Aᵀ sign(A z)</code>는 A z ≠ 0이면 일반적으로 0이 아님.
생성 loss를 정답 근처에서 이차식 <code>½‖x − z‖²_H</code>로 보면, 합친 목표의 해는 <code>x* = z − λ H⁻¹ g + O(λ²)</code> → <b>λ에 비례하는 편향</b>, 방향은 A x를 줄이는 쪽(프레임끼리 같아지는 쪽 = 잔상·평탄화).
λ를 줄이면 편향은 줄지만 일관성 효과도 같이 줄어듦 — direct에는 "편향 없이 일관성만 얻는" λ가 없음.
<br><span class=mut>실측: GT에서 direct를 25 step 최소화하자 {(_t1 < _t0).sum()}/{len(_t0)}개 창 모두 loss가 내려감 (중앙값 −{np.median(_t0 - _t1):.3f}) = 기울기가 0이 아님.</span></div>

<div class=box style='margin-top:8px'><b>④ direct = residual + 정답과 무관한 장면 고유의 힘.</b> <code>A x = A e + A z</code>이므로 
<code>L_D(x) = ‖A e + A z‖</code>. 삼각부등식으로 <code>|L_D(x) − L_R(x)| ≤ ‖A z‖ = floor</code>.
즉 direct는 residual에 "floor만큼의 항"을 더한 것이고, 그 항의 기울기는 오차 e가 0일 때도 <code>Aᵀ sign(A z)</code>로 남는 <b>장면마다 고정된 밀어내는 힘</b>.
그래서 direct는 데이터(정답)를 보지 않는 사전분포 "영상은 warp해도 안 변해야 한다"를 거는 셈이고, 실제 영상은 그 사전분포를 만족하지 않음.
residual은 정답을 기준으로 오차의 일관성만 요구 → 같은 정보(흔들리는 오차, 기하 어긋남은 벌줌)를 편향 없이 줌.</div>

<div class=box style='margin-top:8px'><b>⑤ 둘이 같아지는 경우.</b> floor = ‖A z‖ = 0이면 <code>L_D(x) = ‖A e‖ = L_R(x)</code>로 완전히 같음 
(완벽한 warp가 가능한 정지 합성 장면). 즉 direct가 나쁜 이유는 정확히 floor > 0 하나이고, 그 조건은 실측한 모든 창에서 성립.
<br><b>결론</b>: floor > 0인 한, residual은 정답을 최소점으로 두는 일관성 loss이고 direct는 정답을 최소점에서 밀어내는 loss다. 
같은 λ에서 direct는 항상 (λ에 비례하는) 편향을 추가하므로, 학습이 정답 쪽으로 수렴하는 한 residual보다 나을 수 없다.</div>
<div class=mut>주의: 실제 학습에서 x̂0는 노이즈 수준 σ에서의 모델 추정이라 생성 loss의 해는 정답 z가 아니라 사후 평균(흐릿함)일 수 있음. 
그 경우에도 ④는 그대로 성립 — direct의 추가 항은 z와 무관하게 A x를 줄이는 쪽으로 밀고, residual은 z 기준. vcons는 학습에서 σ ≤ 0.8에만 걸려 x̂0가 z에 가까운 구간.</div>
"""
VIS = f"""<h2>시각화 · 해석</h2>
<h3>그림 1 · loss 지형: 정답(GT)이 가장 낮은 점수를 받는가</h3><img src=why_landscape.png>
<div class=box><b>읽는 법</b> 가로 = GT와 GT를 일부러 망가뜨린 버전, 세로 = 각 loss ÷ GT의 direct 점수(점선 = 1). 선 = 중앙값, 띠 = 25~75% 구간. 
주황 칸 = direct가 중앙값 기준으로 정답보다 좋게(낮게) 평가한 버전.<br>
<b>해석</b> residual(파랑 실선)은 GT에서 0이고, 어떤 방향으로 망가뜨려도 올라감 → 최저점이 정확히 정답. 
direct(주황 점선)는 GT가 1인데, 앞뒤 프레임을 섞은 <b>시간 평균</b> 버전이 1 아래로 내려감 (중앙값 {rng_(fl)}) → 프레임끼리 비슷해지기만 하면 점수가 좋아지므로, 
잔상·번짐이 생긴 영상을 정답보다 낫다고 봄. 공간 흐림·정지 영상은 direct도 대체로 벌주지만, 그건 warp가 어긋나서이지 정답을 알아서가 아님.</div>
<h3>그림 2 · 샘플마다: 잔상 영상 − 정답</h3><img src=why_persample.png>
<div class=box><b>읽는 법</b> 점 하나 = 창 하나, 0 아래 = 그 loss가 잔상 영상(시간 평균 5)을 정답보다 좋게 봄, 아래 숫자 = 0 아래 비율.<br>
<b>해석</b> direct는 synthetic {fr['synthetic']:.3f} · real {fr['real']:.3f}의 샘플에서 잔상을 선호 — 일부 샘플의 우연이 아니라 대부분의 장면에서 생기는 구조적 성질. 
residual은 모든 점이 0 위 (비율 0.000): 정답보다 나은 '가짜 정답'이 없음. 학습에 안 쓴 실제 Aria 영상(real)에서도 같은 모양.</div>
<h3>그림 3 · 정답에서 출발해 loss만 줄이면 어디로 가는가</h3><img src=why_drift.png>
<div class=box><b>읽는 법</b> 위 = 줄이는 loss, 아래 = GT latent와의 거리 (두 값은 단위가 달라 따로 그림). 출발점은 모두 정확한 정답.<br>
<b>해석</b> direct는 정답에서 출발해도 loss가 계속 내려가고 (평균 {rng_(le, 0)} → {rng_(le, 1)}), 그만큼 정답에서 멀어짐 (끝 거리 {rng_(le, 2)}, 
실제 생성 영상의 latent 오차 0.471~0.557(synthetic, residual·direct 모델)의 절반 정도 — 정답에서 출발했는데도 이만큼 밀려남). 즉 direct의 기울기는 정답 위에서도 0이 아니고, 출력을 정답 밖으로 미는 힘이 학습 내내 걸림 — diffusion loss가 반대쪽에서 버텨야 함.
residual은 정답에서 기울기 0이라 선이 0에 붙어 움직이지 않음 → diffusion loss와 같은 곳을 가리킴.</div>
<h3>그림 4 · 그렇게 도착한 영상의 품질</h3><img src=why_psnr.png>
<div class=box><b>해석</b> residual 쪽 막대는 GT를 VAE에 통과시킨 상한({rng_(ps, 1)} dB)과 같음 (SSIM · LPIPS도 상한과 같음). direct를 끝까지 줄인 영상은 {rng_(ps, 0)} dB로 무너지고,
SSIM {rng_(qs, 0)} (상한 {rng_(qs, 1)}) · LPIPS {rng_(ql, 0)} (상한 {rng_(ql, 1)}) — 구조(SSIM)와 지각적 거리(LPIPS) 모두 같은 결론. 
아래 예시처럼 회색으로 평탄해진 영상 (무늬가 없으면 warp해도 차이가 없어 direct 점수가 0에 가까워짐). direct loss의 '완벽한 해'는 내용이 사라진 영상.</div>
@@GRAD@@
<h3>한 줄 요약</h3><div class=box>direct는 "프레임끼리 같게"를 요구해서 최적점이 정답이 아니라 잔상·평탄화 쪽에 있고, residual은 "정답과의 차이가 프레임끼리 일관되게"를 요구해서 최적점이 정확히 정답이다. 
그래서 residual은 생성 loss와 같은 방향으로 일관성을 더하고, direct는 생성 loss와 서로 다른 곳을 당긴다. 이 결론은 학습된 모델·데이터와 무관하게 loss만으로 나오며, synthetic(CARLA)과 real(실제 Aria) 영상에서 같다.
실제 생성 영상 위에서도 direct 기울기의 약 1/3은 모델 오차가 아니라 장면이 정하고, 그중 약 2/3가 오차를 고치는 방향과 반대 (그림 5) — 영상을 스스로 일관되게는 만들지만 정답(카메라·goal) 쪽으로 당기는 힘은 약해짐.</div>"""

# ③ does the loss value track the real error? valbench generated videos (2 models x 120)
Z = [np.load(f) for f in sorted(glob.glob(f"{D}/s[0-9][0-9][0-9].npz"))]
C3 = ""
if Z:
    S = [json.loads(str(z["stats"])) for z in Z]
    fl = np.array([s["floor"] for s in S])
    raw = np.concatenate([[s["raw_res"] for s in S], [s["raw_dir"] for s in S]])
    res = np.concatenate([[s["resid_res"] for s in S], [s["resid_dir"] for s in S]])
    err = np.concatenate([[s["err_res"] for s in S], [s["err_dir"] for s in S]])
    fl2 = np.concatenate([fl, fl])
    c = lambda a, b: float(np.corrcoef(a, b)[0, 1])
    C3 = ("<h2>③ loss 값이 실제 오차를 반영하는가 (synthetic 생성 영상 240개 = 2 모델 × 120)</h2><table><tr><th></th><th>실제 오차 |x̂−z|와 상관</th>"
          "<th>장면 고유값(GT floor)과 상관</th></tr>"
          f"<tr><td>direct loss</td><td>{c(raw, err):.3f}</td><td>{c(raw, fl2):.3f}</td></tr>"
          f"<tr><td>residual loss</td><td>{c(res, err):.3f}</td><td>{c(res, fl2):.3f}</td></tr></table>"
          f"<div class=mut>direct loss 평균 {raw.mean():.3f} 중 GT floor 평균 {fl.mean():.3f} = {fl.mean() / raw.mean():.3f}는 GT도 못 줄이는 부분 (모델 품질과 무관한 장면 값)</div>")


# ---------------- ⑥ on real generated videos: where does the direct gradient point? (why_grad.py) ----------------
GD = f"{D}/why_grad"
GS = [("synthetic", [json.load(open(f)) for f in sorted(glob.glob(f"{GD}/w[0-9][0-9][0-9].json"))]),
      ("real", [json.load(open(f)) for t_ in "ABC" for f in sorted(glob.glob(f"{GD}/{t_}[0-9][0-9][0-9].json"))])]
GS = [(k, R) for k, R in GS if R]
GRAD = ""
if GS:
    WF.grad_split(GS, f"{PG}/why_grad.png")
    gm = lambda R, q: np.mean([r[m][q] for r in R for m in ("res", "dir")])
    gv = {k: {q: gm(R, q) for q in ("agree", "dom", "agree_dom", "cos_DR", "cos_RC")} for k, R in GS}
    rg = lambda q, f=lambda v: v: "~".join(f"{x:.3f}" for x in sorted({round(f(gv[k][q]), 3) for k in gv}))
    nwin = sum(len(R) for _, R in GS)
    allwin = all(r[m]["agree_dom"] < 0.5 for _, R in GS for r in R for m in ("res", "dir"))
    GRAD = f"""<h3>그림 5 · 실제 생성 영상 위에서: direct의 기울기는 오차를 고치는 쪽을 가리키는가</h3><img src=why_grad.png>
<div class=box><b>무엇을 쟀나</b> 학습된 두 모델(residual · direct)이 실제로 만든 영상 (synthetic 120 + real 120 창 × 2 모델 = {2 * nwin}개)에서, 학습과 같은 pair·마스크로 모든 latent 원소의 L1 기울기 방향을 비교.
warp W가 선형이라 direct 항은 정확히 두 조각으로 나뉨: <code>x_t − W(x_s) = [e_t − W(e_s)] + [z_t − W(z_s)]</code> = (모델 오차 항 b) + (GT 자체 항 c).
L1의 기울기는 부호뿐이라 direct는 <code>sign(b + c)</code>, residual은 <code>sign(b)</code> = 오차를 고치는 방향.<br>
<b>결과</b> (synthetic · real)
<br>· 원소의 {rg('dom', lambda v: 1 - v)}: 모델 오차 항이 더 커서 direct도 residual과 같은 방향 (파랑).
<br>· 원소의 {rg('dom')}: GT 자체 항이 더 커서 direct의 방향을 <b>모델 오차가 아니라 장면이 결정</b>. 이 원소들에서 오차를 고치는 방향과 같은 비율은 {rg('agree_dom')}뿐 —
동전 던지기(0.500)보다 낮음 = <b>고치는 방향과 반대로 미는 쪽이 더 많음</b> (주황). {"모든 창, 두 모델 모두에서 0.500 미만." if allwin else ""}
<br>· 전체로 보면 direct 기울기의 {rg('agree', lambda v: 1 - v)}가 오차를 고치는 방향과 반대이고, 두 기울기 벡터의 cosine은 {rg('cos_DR')} (1이면 같은 신호).<br>
<b>왜 반대인가</b> 모델 오차 항과 GT 자체 항은 음의 상관 (cosine {rg('cos_RC')}). GT 자체 항이 큰 곳 = 가림 경계·반사·조명처럼 시점에 따라 GT가 실제로 변하는 곳인데,
생성 영상은 그곳을 GT보다 매끈하게(덜 변하게) 만드는 경향이 있음 → 오차 b ≈ −c. 이때 residual은 "GT처럼 변화를 되살려라", direct는 "변하지 마라(warp와 같게 둬라)"로 정반대.
즉 direct는 생성 모델이 틀리기 쉬운 바로 그 부분(시점 의존 외관)에서 틀린 쪽을 강화함.</div>
<div class=box style='margin-top:8px'><b>학습된 모델 결과와의 연결 (해석)</b> 정식 평가에서 direct 모델(ck 2000)은 residual(ck 2250)보다 MEt3R는 좋지만(0.091 vs 0.117),
카메라 RotErr(14.874 vs 8.709)·goal 성공률(0.342 vs 0.575)·depth δ&lt;1.25(0.194 vs 0.767)가 크게 나쁨 (<a href=./>/vcons</a> 표).
위 측정과 맞는 설명: direct의 vcons 신호 중 약 1/3은 정답과 무관하게 장면이 정하는 "프레임끼리 같게" 힘이고, 이 힘은 GT 카메라를 따르는지와 상관없이
영상을 스스로 일관되게 만드는 쪽으로 작용 → 자기 일관성(MEt3R)은 오르지만, 정답 기준으로 오차를 줄이는 신호(cosine {rg('cos_DR')})는 그만큼 약하고 일부는 반대 →
카메라·goal처럼 정답에 맞춰야 하는 지표가 나빠짐. residual은 같은 일관성 압력을 정답 기준으로만 줘서 이 충돌이 없음.
<br><span class=mut>주의: 이 연결은 해석. direct run은 step(2000 vs 2250)과 학습 데이터 구성도 달라 loss 하나만 바뀐 비교가 아님. 그림 1~4는 모델 없이 loss만 잰 것이라 이 교란이 없음. 그림 5는 두 모델의 출력 위에서 잰 것이지만, 어느 모델의 출력이든 같은 패턴이라(두 모델 출력 간 차이 0.04 이내) 모델 특성이 아니라 loss의 성질.</span></div>
"""

CSS = """:root{--bg:#fafafa;--fg:#1d1d1f;--mut:#6b6b70;--line:#e3e3e8;--card:#fff;--bad:#fde2e1}
@media(prefers-color-scheme:dark){:root{--bg:#141416;--fg:#ececf0;--mut:#9a9aa2;--line:#2c2c31;--card:#1c1c1f;--bad:#4a1f1d}}
body{background:var(--bg);color:var(--fg);font:14px/1.5 -apple-system,system-ui,sans-serif;margin:0;padding:20px 16px;max-width:1400px}
h1{font-size:20px;margin:0 0 4px}h2{font-size:16px;margin:30px 0 8px}h3{font-size:14px;margin:16px 0 6px}.mut{color:var(--mut);font-size:12px}
table{border-collapse:collapse;background:var(--card);font-variant-numeric:tabular-nums;display:block;overflow-x:auto;margin-bottom:6px}
th,td{border:1px solid var(--line);padding:4px 10px;text-align:right;white-space:nowrap}td:first-child,th:first-child{text-align:left}
img{max-width:100%;height:auto;display:block;margin:6px 0}code{font-size:12px}.box{background:var(--card);border:1px solid var(--line);padding:10px 14px;border-radius:6px}"""
P = f"""<!doctype html><html><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'><title>why residual</title><style>{CSS}</style></head><body>
<h1>왜 residual이 direct보다 나은가 · loss만 비교 (학습된 모델 없음)</h1>
<div class=mut>학습 코드와 같은 loss·geometry·pair (gaps 1,2,4,8 양방향 + anchor). 모델을 쓰지 않으므로 학습 데이터·step 차이 같은 교란이 없음 ·
synthetic = valbench5s 120 (CARLA) · real = LookOut A+B+C 120 (학습에 안 쓴 실제 Aria 영상, DA3 pseudo-GT depth, 좌우 보정 pose) · <a href=./>/vcons</a> · <a href=../ablation>/ablation</a></div>
<div class=box><b>direct</b> <code>|x̂_t − W(x̂_s)|</code> · <b>residual</b> <code>|e_t − W(e_s)|, e = x̂ − x_gt</code><br>
① 좋은 일관성 loss라면 GT가 가장 낮아야 함 — 빨간 칸 = 망가뜨린 영상을 GT보다 좋게 본 샘플이 절반 넘음.<br>
② GT에서 출발해 loss만 최소화: residual은 GT에서 기울기 0이라 그대로, direct는 GT를 벗어나 어디로 가는지.<br>
③ loss 값이 실제 오차를 따라가는지, 아니면 장면 고유값(GT floor)을 따라가는지.</div>
{LOGIC}
{VIS}
<h2>세부 표</h2>
{C3}
{''.join(SEC)}
</body></html>"""
P = P.replace("@@GRAD@@", GRAD)
# cache-bust: the viewer sends files with max-age=300, so a rebuilt figure under the same name can show stale in the browser
import re as _re, time as _time
_v = int(_time.time())
P = _re.sub(r"(<img src=')?(why_[\w.\-]+\.(?:png|jpg))", lambda m: m.group(0) + f"?v={_v}", P)
open(f"{PG}/why.html", "w").write(P)
print("wrote", f"{PG}/why.html")
