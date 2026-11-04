#!/usr/bin/env python3
"""Evaluate autoregressive ROLLOUT outputs (drift / quality-over-horizon).

Input: <roll_dir>/rollout_NN.mp4 + rollout_NN_action_poses.npy + rollout_meta.json.
GT frame for rollout frame j == seq_dir/images/{start+j:06d}.png.
Viz (--viz): [GT | rollout] side-by-side mp4 + PSNR/LPIPS-vs-frame curve with chunk lines.
"""
import os, sys, argparse, csv, json, glob, re, importlib.util
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

def psnr(a, b):
    a = a.astype(np.float64); b = b.astype(np.float64)
    mse = np.mean((a - b) ** 2)
    return 99.0 if mse <= 1e-10 else float(20 * np.log10(255.0) - 10 * np.log10(mse))

_LP = {"m": None, "torch": None, "tried": False, "dev": "cpu"}
def lpips_dist(a, b, net="alex"):
    if not _LP["tried"]:
        _LP["tried"] = True
        try:
            import torch, lpips
            _LP["torch"] = torch; _LP["dev"] = "cuda" if torch.cuda.is_available() else "cpu"
            _LP["m"] = lpips.LPIPS(net=net).to(_LP["dev"]).eval()
        except Exception as e:
            _LP["m"] = None; _LP["err"] = str(e)
    if _LP["m"] is None:
        return float("nan")
    torch = _LP["torch"]
    def t(x): return torch.from_numpy(x.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(0).to(_LP["dev"])
    with torch.no_grad():
        return float(_LP["m"](t(a), t(b)).item())

def read_video(p):
    import imageio.v3 as iio
    v = iio.imread(p)
    return [np.asarray(v[i]) for i in range(len(v))]

def read_img(p):
    from PIL import Image
    return np.array(Image.open(p).convert("RGB"))

def resize(img, hw):
    from PIL import Image
    H, W = hw
    return img if img.shape[:2] == (H, W) else np.array(Image.fromarray(img).resize((W, H)))

_ECC = None
def _ecc():
    global _ECC
    if _ECC is None:
        spec = importlib.util.spec_from_file_location("ecc", os.path.join(HERE, "eval_camera_control.py"))
        _ECC = importlib.util.module_from_spec(spec); spec.loader.exec_module(_ECC)
    return _ECC

# Metrics copied from eval_camera_control.camera_metrics() into each CSV row: ATE-style
# (sim3-aligned absolute) + RPE (frame-to-frame relative). "n" is skipped -> row["n"] is frame count.
_MET_KEYS = ("roterr", "transerr", "cammc", "rpe_rot_deg", "rpe_trans", "rpe_camera")

def _depth_to_color(arr, depth_max=20.0):
    """Colorize a metric-depth map exactly like the training viz (_depth_to_color):
    TURBO colormap over a fixed [0, depth_max] metres scale; invalid(>=9999) -> black.
    Matches train_cosmos_predict25_*.py so pred depth is comparable to CARLA GT depth.
    arr: float (...,H,W) metres -> uint8 RGB (...,H,W,3)."""
    a = np.asarray(arr, dtype=np.float32)
    valid = a < 9999.0
    norm = (np.clip(a, 0.0, depth_max) / depth_max * 255.0).astype(np.uint8)
    color = None
    try:                                     # cv2 TURBO == training (byte-identical on server)
        import cv2
        color = cv2.cvtColor(cv2.applyColorMap(norm, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    except Exception:
        try:
            import matplotlib.cm as cm
            color = (cm.get_cmap("turbo")(norm / 255.0)[..., :3] * 255).astype(np.uint8)
        except Exception:                    # last resort: grayscale
            color = np.repeat(norm[..., None], 3, axis=-1)
    color = np.ascontiguousarray(color)
    color[~valid] = 0
    return color

def depth_to_frames(depth_path, F, nchunks, hw, mode="metric"):
    """(K,T_lat,Hp,Wp) predicted-depth latents -> list of L colormapped uint8 RGB frames.
    Each chunk's T_lat maps are temporally expanded (nearest) to its RGB frame count
    (F for chunk 0, F-1 after) so the panel aligns with the rollout video."""
    if not os.path.exists(depth_path):
        return None
    d = np.asarray(np.load(depth_path), dtype=np.float32)
    if d.ndim == 5 and d.shape[2] == 1:      # (K,T_lat,1,Hp,Wp)
        d = d[:, :, 0]
    if d.ndim == 3:                          # (T_lat,Hp,Wp) single chunk
        d = d[None]
    if d.ndim != 4:
        return None
    K, T_lat = d.shape[0], d.shape[1]
    per_chunk = [F] + [F - 1] * (K - 1)
    maps = []
    for k in range(K):
        n = per_chunk[k] if k < len(per_chunk) else (F - 1)
        idx = np.linspace(0, T_lat - 1, max(n, 1)).round().astype(int)
        for i in idx:
            maps.append(d[k, i])
    if not maps:
        return None
    H, W = hw
    if mode == "si":
        # SI-disparity 출력은 미터가 아니다 (실측 0.0359~0.1504, median 0.0544). metric 스케일
        # [0,20] m 로 칠하면 median 이 0.7/255 가 되어 패널 전체가 한 색이 된다.
        # 전역 p1/p99 로 [0,20] 에 맞춘다. 청크별로 정규화하면 이음새마다 밝기가 인위적으로
        # 튀므로(청크 median 0.0503~0.0569 로 이미 일관) 반드시 전역 하나를 쓴다.
        _lo, _hi = float(np.percentile(d, 1.0)), float(np.percentile(d, 99.0))
        if _hi - _lo > 1e-9:
            maps = [(m - _lo) / (_hi - _lo) * 20.0 for m in maps]
    # Fixed 0-20m TURBO, identical to training _depth_to_color (metric metres; no per-clip rescale).
    return [resize(_depth_to_color(m), (H, W)) for m in maps]


def _label_panel(img, text):
    """Draw a text label (top-left, dark box) on an RGB uint8 frame. No-op if PIL missing."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        return img
    im = Image.fromarray(np.ascontiguousarray(img).astype(np.uint8)).convert("RGB")
    dr = ImageDraw.Draw(im)
    try:
        font = ImageFont.load_default(size=max(14, im.height // 20))
    except TypeError:
        font = ImageFont.load_default()
    try:
        l, t, r, b = dr.textbbox((0, 0), text, font=font); tw, th = r - l, b - t
    except Exception:
        tw, th = len(text) * 7, 12
    p = 4
    dr.rectangle([0, 0, tw + 2 * p, th + 2 * p], fill=(0, 0, 0))
    dr.text((p, p), text, fill=(255, 255, 0), font=font)
    return np.asarray(im)

def make_sidebyside(gen, gt, path, fps=16, depth=None, goal=None,
                    depth_label="Depth (pred)"):
    import imageio
    T = min(len(gen), len(gt))
    if depth is not None:
        T = min(T, len(depth))
    if goal is not None:
        T = min(T, len(goal))
    if T == 0:
        return
    w = imageio.get_writer(path, fps=fps)
    for i in range(T):
        g = gen[i]; t = resize(gt[i], g.shape[:2])
        panels = [_label_panel(t, "GT"), _label_panel(g, "Rollout")]
        if goal is not None:
            panels.append(_label_panel(resize(goal[i], g.shape[:2]), "Goal"))
        if depth is not None:
            panels.append(_label_panel(resize(depth[i], g.shape[:2]), depth_label))
        frame = np.concatenate(panels, axis=1)
        frame[:2, :, :] = 255
        w.append_data(frame)
    w.close()

def make_quality_curve(psnrs, lpipss, F, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    x = np.arange(len(psnrs))
    fig, ax1 = plt.subplots(figsize=(9, 3.5))
    ax1.plot(x, psnrs, color="tab:blue", label="PSNR")
    ax1.set_xlabel("rollout frame"); ax1.set_ylabel("PSNR", color="tab:blue")
    ax2 = ax1.twinx()
    if not np.all(np.isnan(lpipss)):
        ax2.plot(x, lpipss, color="tab:red", label="LPIPS")
    ax2.set_ylabel("LPIPS", color="tab:red")
    for c in range(1, len(psnrs) // (F - 1) + 1):
        ax1.axvline(c * (F - 1), color="gray", ls="--", lw=0.7)
    ax1.set_title("quality vs rollout horizon (dashed = chunk boundary)")
    fig.tight_layout(); fig.savefig(path, dpi=90); plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roll_dir", required=True)
    ap.add_argument("--meta", default="rollout_meta.json")
    ap.add_argument("--out_csv", default=None)
    ap.add_argument("--lpips_net", default="alex")
    ap.add_argument("--viz", action="store_true")
    ap.add_argument("--drift", action="store_true")
    ap.add_argument("--drift_chunked", action="store_true",
                    help="estimate drift per chunk (F frames each, overlap 1) instead of the whole "
                         "rollout at once -> bounds da3/colmap memory; metrics averaged over chunks")
    ap.add_argument("--no_quality", action="store_true",
                    help="skip PSNR/LPIPS video-quality metrics (+ their GT reads and quality viz); "
                         "compute only the --drift camera metric (much faster: no per-frame LPIPS)")
    ap.add_argument("--no_viz_depth", action="store_true", help="omit predicted-depth panel")
    ap.add_argument("--viz_depth_mode", default="metric", choices=["metric", "si"],
                    help="metric: 기존 동작 (미터 [0,20] TURBO). si: SI-disparity 출력을 전역 "
                         "p1/p99 로 정규화. 우리 DepthUpHead 출력은 미터가 아니라 disparity "
                         "(실측 0.036~0.150) 라 metric 스케일에서는 전부 한 색이 된다.")
    ap.add_argument("--skip_existing", action="store_true",
                    help="viz/<base>_vs_gt.mp4 가 이미 있고 기존 CSV 에 그 샘플 행이 있으면 "
                         "비싼 재계산(영상 읽기+GT 읽기+LPIPS)을 건너뛰고 그 행을 재사용한다. "
                         "CSV 는 재사용 행 + 새 행을 합쳐 쓰므로 완전하게 유지된다 -- 샘플별 "
                         "skip 이 완성된 CSV 를 부분으로 덮는 사고를 막는 게 핵심이다. "
                         "끝의 chunk 별 요약표는 '새로 계산한 샘플' 만 집계한다.")
    ap.add_argument("--estimator", default="colmap")
    args = ap.parse_args()

    meta = json.load(open(os.path.join(args.roll_dir, args.meta)))
    out_csv = args.out_csv or os.path.join(args.roll_dir, "rollout_metrics.csv")
    viz_dir = os.path.join(args.roll_dir, "viz")
    if args.viz:
        os.makedirs(viz_dir, exist_ok=True)
    vids = sorted(glob.glob(os.path.join(args.roll_dir, "rollout_*.mp4")))
    print(f"[info] {len(vids)} rollouts")

    prev_rows = {}
    if args.skip_existing and os.path.exists(out_csv):
        with open(out_csv, newline="") as f:
            for r in csv.DictReader(f):
                if r.get("sample"):
                    prev_rows[r["sample"]] = r
        print(f"[info] 기존 CSV {len(prev_rows)}행 로드 -> --skip_existing 재사용 후보")

    rows = []
    reused = 0
    per_chunk_all = {}
    for vp in vids:
        base = os.path.splitext(os.path.basename(vp))[0]
        idx = int(re.search(r"(\d+)", base).group(1))
        info = meta.get(str(idx))
        if info is None:
            continue
        # 비싼 작업(read_video / GT 읽기 / LPIPS) 앞에서 걸러야 실제로 빨라진다
        if args.skip_existing and base in prev_rows:
            _vok = (not args.viz) or os.path.exists(os.path.join(viz_dir, f"{base}_vs_gt.mp4"))
            if _vok:
                rows.append(prev_rows[base]); reused += 1
                print(f"  {base}: [기존 결과 재사용]", flush=True)
                continue
        seq_dir = info["seq_dir"]; start = int(info["start_frame"]); F = int(info["F"])
        nchunks = int(info.get("num_chunks", 1))
        gen = read_video(vp)
        gt = []
        if not args.no_quality:
            for j in range(len(gen)):
                fp = os.path.join(seq_dir, "images", f"{start+j:06d}.png")
                if not os.path.exists(fp):
                    break
                gt.append(read_img(fp))
        k = len(gen) if args.no_quality else min(len(gen), len(gt))
        hw = gen[0].shape[:2]
        if args.no_quality:
            ps = lp = None
            row = {"sample": base, "n": k}
        else:
            ps = np.array([psnr(resize(gt[j], hw), gen[j]) for j in range(k)])
            lp = np.array([lpips_dist(resize(gt[j], hw), gen[j], args.lpips_net) for j in range(k)])
            row = {"sample": base, "n": k,
                   "psnr_mean": float(np.mean(ps)), "lpips_mean": float(np.nanmean(lp)),
                   "psnr_first_chunk": float(np.mean(ps[:F])), "psnr_last_chunk": float(np.mean(ps[-(F - 1):]))}
            for j in range(k):
                c = min(j // (F - 1), nchunks - 1)
                per_chunk_all.setdefault(c, []).append((ps[j], lp[j]))
        if args.drift:
            ip = os.path.join(seq_dir, "intrinsics.npy")
            intr = np.load(ip) if os.path.exists(ip) else None
            apos = np.load(vp.replace(".mp4", "_action_poses.npy"))
            m = _ecc()
            if args.drift_chunked:
                # Per-chunk (F frames, overlap 1) estimation -> bounds estimator memory; avg metrics.
                _acc = {_key: [] for _key in _MET_KEYS}; _errs = []
                for _c in range(max(nchunks, 1)):
                    _lo = _c * (F - 1); _hi = min(_lo + F, k)
                    if _hi - _lo < 2:
                        continue
                    _gm = np.stack([m.carla_pose_to_mat(*p) for p in apos[_lo:_hi]])
                    _po, _n, _er = m.ESTIMATORS[args.estimator](gen[_lo:_hi], intr)
                    if _po is None:
                        _errs.append(f"c{_c}:{_er}"); continue
                    _kk = min(len(_po), len(_gm)); _cm = m.camera_metrics(_po[:_kk], _gm[:_kk])
                    for _key in _acc:
                        if _key in _cm:
                            _acc[_key].append(_cm[_key])
                if _acc["roterr"]:
                    for _key in _acc:
                        if _acc[_key]:
                            row[_key] = float(np.mean(_acc[_key]))
                if _errs:
                    row["drift_err"] = ";".join(_errs)[:200]
            else:
                gt_mats = np.stack([m.carla_pose_to_mat(*p) for p in apos[:k]])
                poses, n, err = m.ESTIMATORS[args.estimator](gen[:k], intr)
                if poses is not None:
                    kk = min(len(poses), len(gt_mats)); cm = m.camera_metrics(poses[:kk], gt_mats[:kk])
                    row.update({_k: cm[_k] for _k in _MET_KEYS if _k in cm})
                else:
                    row["drift_err"] = err
        rows.append(row)
        if args.viz and not args.no_quality:
            _depth = None
            if not args.no_viz_depth:
                _depth = depth_to_frames(vp.replace(".mp4", "_pred_depth.npy"), F, nchunks,
                                         gen[0].shape[:2], mode=args.viz_depth_mode)
                if _depth is not None:
                    _depth = _depth[:k]
            _goal = None
            _gf = info.get("goal_frames")
            if _gf:
                _gcache = {}; _goal = []
                for _j in range(k):
                    _c = min(_j // (F - 1), nchunks - 1)
                    _gi = _gf[_c] if _c < len(_gf) else _gf[-1]
                    if _gi not in _gcache:
                        _gp = os.path.join(seq_dir, "images", f"{int(_gi):06d}.png")
                        _gcache[_gi] = read_img(_gp) if os.path.exists(_gp) else None
                    _goal.append(_gcache[_gi])
                if any(x is None for x in _goal):
                    _goal = None
            _dlab = "Depth (pred, relative)" if args.viz_depth_mode == "si" else "Depth (pred)"
            make_sidebyside(gen[:k], gt[:k], os.path.join(viz_dir, f"{base}_vs_gt.mp4"),
                            depth=_depth, goal=_goal, depth_label=_dlab)
            make_quality_curve(ps, lp, F, os.path.join(viz_dir, f"{base}_quality.png"))
        if args.no_quality:
            _nan = float("nan")
            _dr = (f"roterr={row['roterr']:.3f} transerr={row['transerr']:.3f} cammc={row['cammc']:.3f} "
                   f"rpe_rot={row.get('rpe_rot_deg', _nan):.3f} rpe_tr={row.get('rpe_trans', _nan):.3f}"
                   if "roterr" in row else f"drift_err={str(row.get('drift_err',''))[:50]}")
            print(f"  {base}: {_dr} n={k}", flush=True)
        else:
            print(f"  {base}: psnr={row['psnr_mean']:.2f} (first={row['psnr_first_chunk']:.2f} "
                  f"last={row['psnr_last_chunk']:.2f}) lpips={row['lpips_mean']:.3f} n={k}", flush=True)

    keys = sorted({kk for r in rows for kk in r})
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, restval="")
        w.writeheader(); [w.writerow(r) for r in rows]
    print(f"[done] {len(rows)} rows ({reused} reused) -> {out_csv}")
    if reused:
        print(f"[note] chunk 별 요약표는 새로 계산한 {len(rows) - reused}개 샘플만 집계한다")

    print("\n" + "=" * 48)
    print("  ROLLOUT quality vs chunk (mean over samples)")
    print("=" * 48)
    print(f"  {'chunk':>5} {'psnr':>8} {'lpips':>8} {'frames':>8}")
    for c in sorted(per_chunk_all):
        arr = np.array(per_chunk_all[c])
        print(f"  {c:>5} {np.mean(arr[:,0]):>8.2f} {np.nanmean(arr[:,1]):>8.3f} {len(arr):>8}")
    print("=" * 48)
    print("  psnr 하락 / lpips 상승 시작 chunk = drift 붕괴 지점")

if __name__ == "__main__":
    main()
