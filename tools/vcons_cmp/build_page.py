#!/usr/bin/env python3
"""Static page for /vcons from vcons_cmp.py outputs: <out>/page/{index.html, *.png, *.mp4}."""
import glob, html, json, os, subprocess, sys
import cv2, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

O = sys.argv[1]; PG = f"{O}/page"; os.makedirs(PG, exist_ok=True)
H_ = "/root/project/egowam/results_syn/goalprog_mode_real6_nomr_hg0_grad0.5"
RUNS = {"res": f"{H_}/ckpt2250/valbench5s_all120", "dir": f"{H_}/ckptdirect2000/valbench5s_all120"}
meta = json.load(open(f"{RUNS['res']}/samples_meta.json"))
Z = {int(os.path.basename(f)[1:4]): np.load(f) for f in sorted(glob.glob(f"{O}/s[0-9][0-9][0-9].npz"))}
IDX = sorted(Z); N = len(IDX)
S = {i: json.loads(str(Z[i]["stats"])) for i in IDX}
COL = {"gt": "#6b6b70", "res": "#2a6fdb", "dir": "#d9480f"}
LAB = {"gt": "GT", "res": "residual (main ck 2250)", "dir": "direct (ck 2000)"}
f3 = lambda v: f"{v:.3f}"

# ---------------- aggregate table ----------------
def arr(k):
    return np.array([S[i][k] for i in IDX])
fl = arr("floor")
rows = []
def row(name, a_res, a_dir, better, note=""):
    d = a_dir - a_res
    rows.append((name, a_res.mean(), a_dir.mean(), d.mean(), float(np.mean(d < 0)), better, note))
rel = lambda k, m: np.array([np.mean(Z[i][f"{k}_{m}"][1:] / np.maximum(Z[i][f"{k}_gt"][1:], 1e-6)) for i in IDX])
row("direct loss  |x̂_t − W(x̂_s)|", arr("raw_res"), arr("raw_dir"), "", "GT 값(floor) 평균 " + f3(fl.mean()))
row("direct loss ÷ GT floor", arr("raw_res") / fl, arr("raw_dir") / fl, "", "1 = GT만큼 일관 · 1 미만 = GT보다 '더' 일관 (덜 움직이거나 흐려서)")
row("residual loss  |e_t − W(e_s)|", arr("resid_res"), arr("resid_dir"), "v", "GT에서 0")
row("per-frame error |x̂ − z|", arr("err_res"), arr("err_dir"), "v", "latent")
row("선명도 ÷ GT (Laplacian 분산)", rel("sharp", "res"), rel("sharp", "dir"), "", "1 = GT만큼 선명")
row("움직임 ÷ GT (인접 프레임 차)", rel("motion", "res"), rel("motion", "dir"), "", "1 = GT만큼 움직임")
T = ["<table><tr><th>지표 (valbench 120, 같은 창·같은 goal)</th><th>residual · ck 2250</th><th>direct · ck 2000</th><th>Δ direct − residual</th><th>direct가 더 작은 샘플 비율</th><th>참고</th></tr>"]
for name, r, d, dd, fr, b, note in rows:
    frs = f"{fr:.3f}"
    T.append(f"<tr><td>{html.escape(name)}</td><td>{f3(r)}</td><td>{f3(d)}</td><td>{dd:+.3f}</td><td>{frs}</td><td class=mut>{html.escape(note)}</td></tr>")
T.append("</table>")

# ---------------- curves ----------------
def mean_t(k):
    return np.nanmean(np.stack([Z[i][k] for i in IDX]), 0)
fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
t = np.arange(1, len(mean_t("floor_t")) + 1)
ax[0].plot(t, mean_t("floor_t"), color=COL["gt"], lw=2, label="GT floor (direct loss if output = GT)")
ax[0].plot(t, mean_t("raw_t_res"), color=COL["res"], lw=2, label=LAB["res"])
ax[0].plot(t, mean_t("raw_t_dir"), color=COL["dir"], lw=2, label=LAB["dir"])
ax[0].set_title("direct loss per latent t (gap 1)", fontsize=10); ax[0].set_xlabel("latent t"); ax[0].legend(fontsize=8)
for j, (k, ttl) in enumerate((("sharp", "sharpness ÷ GT"), ("motion", "motion ÷ GT"))):
    for m in ("res", "dir"):
        y = np.nanmean(np.stack([Z[i][f"{k}_{m}"] / np.maximum(Z[i][f"{k}_gt"], 1e-6) for i in IDX]), 0)
        ax[j + 1].plot(np.arange(len(y))[1:], y[1:], color=COL[m], lw=1.6, label=LAB[m])
    ax[j + 1].axhline(1, color=COL["gt"], ls="--", lw=1); ax[j + 1].set_title(ttl, fontsize=10); ax[j + 1].set_xlabel("frame"); ax[j + 1].legend(fontsize=8)
for a_ in ax:
    a_.grid(alpha=.25)
fig.tight_layout(); fig.savefig(f"{PG}/curves.png", dpi=110); plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
x, y = arr("raw_res") / fl, arr("raw_dir") / fl
ax[0].scatter(x, y, s=12, color=COL["dir"]); lo, hi = min(x.min(), y.min()) * .95, max(x.max(), y.max()) * 1.05
ax[0].plot([lo, hi], [lo, hi], color=COL["gt"], lw=1); ax[0].axhline(1, ls="--", color=COL["gt"], lw=.8); ax[0].axvline(1, ls="--", color=COL["gt"], lw=.8)
ax[0].set_xlabel("residual: direct loss ÷ GT floor"); ax[0].set_ylabel("direct: direct loss ÷ GT floor"); ax[0].set_title("per sample (dashed = GT level)", fontsize=10)
xs, ys = rel("sharp", "res"), rel("sharp", "dir")
ax[1].scatter(xs, ys, s=12, color=COL["dir"]); lo, hi = min(xs.min(), ys.min()) * .95, max(xs.max(), ys.max()) * 1.05
ax[1].plot([lo, hi], [lo, hi], color=COL["gt"], lw=1)
ax[1].set_xlabel("residual: sharpness ÷ GT"); ax[1].set_ylabel("direct: sharpness ÷ GT"); ax[1].set_title("per sample", fontsize=10)
for a_ in ax:
    a_.grid(alpha=.25)
fig.tight_layout(); fig.savefig(f"{PG}/scatter.png", dpi=110); plt.close(fig)

# ---------------- examples ----------------
def frames(p, ks):
    cap = cv2.VideoCapture(p); out, f = {}, 0
    while True:
        ok, im = cap.read()
        if not ok:
            break
        if f in ks:
            out[f] = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        f += 1
    return out

d_ratio = (arr("raw_dir") - arr("raw_res")) / fl
pick = list(dict.fromkeys([IDX[j] for j in np.argsort(d_ratio)[:4]] + [IDX[j] for j in np.argsort(np.abs(d_ratio - np.median(d_ratio)))[:2]]))
cm = plt.get_cmap("magma").copy(); cm.set_bad((.55, .55, .55))
EX = []
for i in pick:
    z = Z[i]; m = meta[str(i)]; sd, st = m["seq_dir"].rstrip("/"), int(m["start_frame"])
    ts = [4, 10, 16]
    ks = [4 * tt for tt in ts]
    gt = {k: cv2.cvtColor(cv2.imread(f"{sd}/images/{st + k:06d}.png"), cv2.COLOR_BGR2RGB) for k in ks}
    V = {k_: frames(f"{RUNS[k_]}/sample_{i:02d}.mp4", set(ks)) for k_ in RUNS}
    vmax = float(np.nanpercentile(np.concatenate([z[f"map_{n}"].astype(np.float32).ravel() for n in ("floor", "raw_res", "raw_dir")]), 99))
    fig, axs = plt.subplots(len(ts), 6, figsize=(19, 3.3 * len(ts)))
    for r, (tt, k) in enumerate(zip(ts, ks)):
        M = z["map_mask"][tt - 1].astype(bool)
        def hm(ax, key, ttl):
            v = np.ma.array(z[f"map_{key}"][tt - 1].astype(np.float32), mask=~M)
            ax.imshow(cv2.resize(v.filled(np.nan), (512, 512), interpolation=cv2.INTER_NEAREST), cmap=cm, vmin=0, vmax=vmax)
            ax.set_title(f"{ttl}  {np.nanmean(v.filled(np.nan)):.3f}", fontsize=8)
        axs[r, 0].imshow(gt[k]); axs[r, 0].set_title(f"GT frame {k} (latent {tt})", fontsize=8)
        axs[r, 1].imshow(V["res"][k]); axs[r, 1].set_title(LAB["res"], fontsize=8)
        axs[r, 2].imshow(V["dir"][k]); axs[r, 2].set_title(LAB["dir"], fontsize=8)
        hm(axs[r, 3], "floor", "GT floor = direct loss if output = GT")
        hm(axs[r, 4], "raw_res", "residual output |x_t - W(x_s)|")
        hm(axs[r, 5], "raw_dir", "direct output |x_t - W(x_s)|")
    for ax in axs.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"sample {i:02d} · {os.path.relpath(sd, '/root/data/eval_bench/val_bench5s')} · start {st} · gap 1 (t <- t-1) · gray = outside GT visibility mask", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{PG}/ex_{i:03d}.png", dpi=90); plt.close(fig)
    mp4 = f"{PG}/ex_{i:03d}.mp4"
    if not os.path.exists(mp4):
        gtv = f"{PG}/_gt_{i:03d}.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", "16", "-start_number", str(st), "-i", f"{sd}/images/%06d.png",
                        "-frames:v", "77", "-pix_fmt", "yuv420p", gtv], check=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", gtv, "-i", f"{RUNS['res']}/sample_{i:02d}.mp4", "-i", f"{RUNS['dir']}/sample_{i:02d}.mp4",
                        "-filter_complex", "[0:v]scale=384:384[a];[1:v]scale=384:384[b];[2:v]scale=384:384[c];[a][b][c]hstack=3",
                        "-r", "16", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", mp4], check=True)
        os.remove(gtv)
    s = S[i]
    EX.append(f"<h3>sample {i:02d} · direct loss ÷ floor: residual {s['raw_res'] / s['floor']:.3f} · direct {s['raw_dir'] / s['floor']:.3f}</h3>"
              f"<video src='ex_{i:03d}.mp4' controls loop muted playsinline></video><div class=mut>왼쪽부터 GT · {LAB['res']} · {LAB['dir']}</div>"
              f"<img src='ex_{i:03d}.png' loading=lazy>")

_r = {n: (r, d, fr) for n, r, d, _, fr, _, _ in rows}
_k = lambda pre: next(v for n, v in _r.items() if n.startswith(pre))
a1, a2, a3 = _k("direct loss ÷"), _k("선명도"), _k("움직임")
FIND = (f"① GT 기하로 잰 프레임 간 차이: direct loss ÷ floor = residual {a1[0]:.3f} · direct {a1[1]:.3f} (direct가 더 낮은 샘플 {a1[2]:.3f}). "
        "처음 가설(direct가 floor 아래로 = GT보다 정적/흐리게)은 <b>맞지 않음</b>.<br>"
        f"② 움직임 ÷ GT = residual {a3[0]:.3f} · direct {a3[1]:.3f}, 선명도 ÷ GT = residual {a2[0]:.3f} · direct {a2[1]:.3f} → direct가 더 많이 움직이고 더 흐림.<br>"
        "③ 그런데 아래 정식 평가에서 MEt3R(영상 자체 기준 3D 일관성)은 direct가 <b>더 좋음</b>, 카메라 충실도(RotErr·ATE)와 goal 성공률은 크게 나쁨. "
        "즉 direct 영상은 스스로는 일관되지만 <b>지시한 카메라 경로를 덜 따름</b> → GT 기하로 warp하면 어긋나서 ①·②가 커짐 (떨림이라기보다 경로 이탈).<br>"
        "해석(가설): direct의 목표 floor는 GT로도 도달 못 하는 값이라 gradient가 'GT에 가까워지는 것'과 '프레임끼리 비슷해지는 것'을 섞어 밀고, "
        "후자는 카메라를 덜 따라도 줄일 수 있음. residual은 GT 대비 오차만 맞추므로 카메라를 따르는 것과 충돌하지 않음.")

# formal eval (valbench 120) from the same loaders as /early and /ablation
sys.path.insert(0, "/root/project/evalwam/tools/ckpt_viewer")
import early  # noqa: E402
EVC = [("residual · 메인 ck 2250", f"{H_}/ckpt2250/valbench5s_all120"), ("direct · ck 2000", f"{H_}/ckptdirect2000/valbench5s_all120"),
       ("참고 · ours avg 7500+8750", f"{H_}/ckpta7500_8750/valbench5s_all120")]
EX_ = {n: early._load(d) for n, d in EVC}
KEYS = ["fvd:fvd16", "vq:psnr", "vq:lpips", "m3:met3r", "s:act_roterr", "s:act_ate", "s:coh_roterr", "s:coh_ate",
        "s:depth_C_delta1", "s:goal_da3_success", "gi:goal_lpips"]
EV = ["<table><tr><th>지표 (valbench 120)</th>" + "".join(f"<th>{html.escape(n)}</th>" for n, _ in EVC) + "<th>Δ direct − residual</th></tr>"]
for g, name, key, d in early.ALL:
    if key not in KEYS:
        continue
    v = [EX_[n].get(key) if EX_[n] else None for n, _ in EVC]
    if not all(isinstance(x, (int, float)) for x in v):
        continue
    dd = v[1] - v[0]; good = (dd < 0) if d == "v" else (dd > 0)
    EV.append(f"<tr><td>{name} {'↑' if d == '^' else '↓'}</td>" + "".join(f"<td>{x:.3f}</td>" for x in v)
              + f"<td style='color:{'#11612f' if good else '#b42318'}'>{dd:+.3f}</td></tr>")
EV.append("</table>")
CSS = """:root{--bg:#fafafa;--fg:#1d1d1f;--mut:#6b6b70;--line:#e3e3e8;--card:#fff}
@media(prefers-color-scheme:dark){:root{--bg:#141416;--fg:#ececf0;--mut:#9a9aa2;--line:#2c2c31;--card:#1c1c1f}}
body{background:var(--bg);color:var(--fg);font:14px/1.5 -apple-system,system-ui,sans-serif;margin:0;padding:20px 16px;max-width:1500px}
h1{font-size:20px;margin:0 0 4px}h2{font-size:16px;margin:28px 0 8px}h3{font-size:14px;margin:22px 0 6px}.mut{color:var(--mut);font-size:12px}
table{border-collapse:collapse;background:var(--card);font-variant-numeric:tabular-nums;display:block;overflow-x:auto}
th,td{border:1px solid var(--line);padding:4px 10px;text-align:right;white-space:nowrap}td:first-child,th:first-child{text-align:left}
img,video{max-width:100%;height:auto;display:block;margin:6px 0;background:var(--card)}code{font-size:12px}
.box{background:var(--card);border:1px solid var(--line);padding:10px 14px;border-radius:6px}"""
P = f"""<!doctype html><html><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>
<title>direct vs residual</title><script>if(!location.pathname.endsWith('/'))location.replace(location.pathname+'/'+location.search)</script><style>{CSS}</style></head><body>
<h1>L_vcons · direct vs residual</h1>
<div class=mut>valbench5s {N}개 창 (두 run 모두 같은 창·같은 goal) · residual = 메인 run ck 2250 (ablation step에 가장 가까운 메인 ckpt), direct = ablation ck 2000 ·
geometry·마스크·pair 구성은 학습 코드 그대로 (GT metric depth, CARLA pose, gaps 1,2,4,8 양방향 + anchor) · 모든 값은 latent(64×64) 기준 · <a href=../ablation>/ablation</a> · <a href=../>홈</a></div>
<div class=box style='margin-top:10px'>👉 <a href=why.html><b>왜 residual이 direct보다 나은가 — loss만 비교 (valbench 120 + LookOut A/B/C)</b></a> · 학습된 모델 없이 loss 자체를 시험해 교란 없음</div>
<h2>두 loss</h2><div class=box>
<b>residual</b> (ours): <code>|e_t − W(e_s)|</code>, <code>e = x̂0 − x0_gt</code> → 출력이 GT와 같으면 <b>0</b>. GT와의 오차가 프레임 사이에서 일관되게만 강제.<br>
<b>direct</b>: <code>|x̂0_t − W(x̂0_s)|</code> → 출력이 GT와 같아도 <b>GT floor</b> (가림·드러남, 시점에 따른 반사·조명, warp 오차)만큼 남음.
그래서 direct는 floor 아래(GT보다 더 정적이거나 흐린 출력)로 내려가는 쪽에 보상을 줄 수 있음 — 이게 처음 가설.</div>
<h2>관찰 (120개)</h2><div class=box>{FIND}</div>
<div class=mut>비교 조건의 한계: (1) step 2000 vs 2250 (250 step 차이) · (2) direct run은 학습 데이터도 다름 — DL3DV 45-scene 부분집합(메인 590), telestyle goal 이미지 ~60%, GPU 2장×GA 8 (실효 batch 같음). 즉 loss 하나만 바뀐 비교가 아님.</div>
<h2>정식 평가 (/ablation과 같은 값)</h2>{''.join(EV)}<div class=mut>카메라 = cam77 · depth δ = 예측 depth vs 생성 영상의 DA3 depth</div>
<h2>latent 일관성 집계 (120개)</h2>{''.join(T)}
<h2>시간에 따른 변화</h2><img src=curves.png><div class=mut>왼쪽: 회색 선(GT floor) 아래로 내려가면 GT보다 프레임 간 차이가 작다는 뜻. 가운데·오른쪽: 1 = GT 수준.</div>
<h2>샘플별</h2><img src=scatter.png>
<h2>warp가 잘 안 맞는 샘플 · 경향</h2><div class=box>
GT floor (= GT 영상끼리 warp해도 남는 오차) 중앙값 0.310 · 상위 10% 0.359 이상 · 최대 0.476.<br>
<b>완전히 무너지는 건 s098 하나</b>: 카메라가 덤불 속으로 들어가 화면 71%가 3 m 이내 잎사귀 + 한 step에 14° 회전 → 가시 영역 1.3%, floor 0.754 (아래 그림).<br>
<b>경향</b>: floor는 회전량(상관 +0.349)과 가까운 물체 비율(+0.291)에 따라 커지고, 난이도(0.311~0.316)·이동 거리(−0.046)와는 거의 무관.
latent step당 yaw 변화 &lt;1° → floor 0.242 · 가시 0.600, 6~10° → 0.316 · 0.511, 10° 이상 → 0.309 · 0.446.
날씨는 비·노을(HardRainSunset 0.343)이 높고 맑은/흐린 노을(0.284~0.289)이 낮음 — 젖은 노면 반사·빗줄기처럼 시점에 따라 바뀌는 외관.<br>
<b>direct와의 관계</b>: direct가 residual보다 나쁜 샘플 비율은 floor 하위·중위·상위 1/3에서 0.800 · 0.850 · 0.850으로 비슷하고,
절대 차이는 오히려 floor가 낮은(warp가 잘 되는) 샘플에서 더 큼 (+0.051 vs +0.030). → direct의 열세는 warp가 나쁜 샘플 때문이 아님.</div>
<img src=warp_bad_s098.png loading=lazy>
<h2>예시</h2><div class=mut>direct − residual 차이(÷ floor)가 가장 큰 4개 + 중앙값 근처 2개. 지도 = gap 1 (t ← t−1), 세 지도 같은 색 범위.</div>
{''.join(EX)}
</body></html>"""
open(f"{PG}/index.html", "w").write(P)
print(PG, N, "samples", pick)
