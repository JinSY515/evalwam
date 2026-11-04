#!/usr/bin/env python3
"""L_geo 헤드룸 분해. 학습/추론/GPU/VAE 없이 기존 추론 산출물만 읽는다.

  A  GT depth   + GT 카메라    -> 바닥 (와핑/최근접샘플 노이즈)
  B  예측 depth + GT 카메라    -> **R1 (L_geo^GT) 이 최소화하는 양**
  C  예측 depth + 예측 카메라  -> **R2 (L_geo^joint) 이 최소화하는 양**

A, B 는 _pred_poses.npy 가 필요 없다. C 만 필요하고, 없으면 C 를 건너뛴다.

예측 depth 는 SI-disparity 이므로 **윈도 전역 affine** 으로 GT disparity 에 맞춘 뒤
metric 으로 되돌린다. 조합들은 **하나의 고정 공통 마스크**에서만 비교한다 -- 조합마다
유효 픽셀을 따로 잡으면 서로 다른 픽셀 집합의 평균을 비교하게 되어 결론이 뒤집힌다.

읽기 전용. --out_csv 를 줄 때만 CSV 하나. 이름에 `_metrics` 금지 (mean_metrics.py:47 이
디렉터리 입력에서 `*_metrics.csv` 를 glob 한다).
"""
import argparse, csv, json, math, os
import numpy as np

_C3 = np.array([[0, 0, 1], [1, 0, 0], [0, -1, 0]], float)   # data_utils.py:306
NEAR, FAR = 0.5, 200.0
TOL_ABS, TOL_REL = 0.30, 0.05                               # warp_core.py 와 동일


def carla_c2w(v):
    x, y, z, roll, pitch, yaw = [float(t) for t in v[:6]]
    r, p, yw = map(math.radians, (roll, pitch, yaw))
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(yw), math.sin(yw)
    M = np.eye(4)
    M[0, 0] = cy*cp; M[0, 1] = cy*sp*sr - sy*cr; M[0, 2] = cy*sp*cr + sy*sr
    M[1, 0] = sy*cp; M[1, 1] = sy*sp*sr + cy*cr; M[1, 2] = sy*sp*cr - cy*sr
    M[2, 0] = -sp;   M[2, 1] = cp*sr;            M[2, 2] = cp*cr
    M[:3, 3] = (x, y, z)
    return M


def to_work(a, out):
    """최근접 데시메이션. bilinear 은 깊이 불연속에서 없는 중간값을 만든다."""
    if a.shape[0] == out and a.shape[1] == out:
        return a
    H, W = a.shape
    yi = ((np.arange(out) + 0.5) * H / out).astype(int).clip(0, H - 1)
    xi = ((np.arange(out) + 0.5) * W / out).astype(int).clip(0, W - 1)
    return a[yi][:, xi]


def transport(d_t, d_s, c2w_t, c2w_s, intr):
    h, w = d_t.shape
    fx, fy, cx, cy = intr
    vv, uu = np.mgrid[0:h, 0:w]
    U, V = uu + 0.5, vv + 0.5                       # 픽셀 중심
    pc = np.stack([(U - cx)/fx*d_t, (V - cy)/fy*d_t, d_t], -1) @ _C3.T
    T = np.linalg.inv(c2w_s) @ c2w_t
    ps = (pc @ T[:3, :3].T + T[:3, 3]) @ _C3
    Z = ps[..., 2]
    Zs = np.where(np.abs(Z) < 1e-6, 1e-6, Z)
    us = cx + fx*ps[..., 0]/Zs
    vs = cy + fy*ps[..., 1]/Zs
    ok = np.isfinite(us) & np.isfinite(vs) & (Z > NEAR)
    ok &= (us >= 0) & (us <= w) & (vs >= 0) & (vs <= h)
    xi = np.clip(us.astype(int), 0, w - 1)
    yi = np.clip(vs.astype(int), 0, h - 1)
    return Z, d_s[yi, xi], ok


def gt_mask(dg_t, dg_s, c2w_t, c2w_s, intr):
    Z, ds, ok = transport(dg_t, dg_s, c2w_t, c2w_s, intr)
    m = ok & (dg_t > NEAR) & (dg_t < FAR) & (ds > NEAR)
    return m & (np.abs(Z - ds) < np.maximum(TOL_REL*Z, TOL_ABS))


def fit_affine_window(p, g):
    valid = (g > 0) & (g < FAR)
    tgt = np.where(valid, 1.0/np.clip(g, 1e-3, None), 0.0).ravel()
    m = valid.astype(float).ravel()
    pf = p.ravel()
    n = max(m.sum(), 1.0)
    sp = (pf*m).sum(); spp = (pf*pf*m).sum(); st = (tgt*m).sum(); spt = (pf*tgt*m).sum()
    det = max(spp*n - sp*sp, 1e-6)
    return (spt*n - sp*st)/det, (st*spp - sp*spt)/det


def resolve(p, pod, mac):
    return p if os.path.isdir(p) else (mac + p[len(pod):] if p.startswith(pod) else p)


# ---- --matrix -------------------------------------------------------------------------
# A digits-only 3x5 font. The tool stays numpy-only (no cv2 -> no libGL), so the figure
# carries numbers and k-labels; every word belongs in stdout and the caption instead.
_F35 = {"0": "111101101101111", "1": "010110010010111", "2": "111001111100111",
        "3": "111001111001111", "4": "101101111001001", "5": "111100111001111",
        "6": "111100111101111", "7": "111001001001001", "8": "111101111101111",
        "9": "111101111001111", ".": "000000000000010", "k": "100101110101101",
        " ": "000000000000000"}


def _text(img, s, x, y, sc=2, col=(255, 255, 255)):
    for chx in s:
        g = _F35.get(chx, _F35[" "])
        for r in range(5):
            for c in range(3):
                if g[r*3 + c] == "1":
                    img[y + r*sc:y + (r+1)*sc, x + c*sc:x + (c+1)*sc] = col
        x += 4*sc


def _ramp(t):
    t = np.clip(t, 0, 1)
    r = np.clip(2.2*t - 1.1, 0, 1); g = np.clip(1.9*t - 0.35, 0, 1)
    b = np.clip(np.minimum(2.6*t, 1.9 - 1.7*t), 0, 1)
    return (np.stack([r, g, b], -1)*255).astype(np.uint8)


def _png(path, img):
    import struct, zlib
    H, W, _ = img.shape
    raw = b"".join(b"\x00" + img[y].tobytes() for y in range(H))
    def ck(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n"
                + ck(b"IHDR", struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0))
                + ck(b"IDAT", zlib.compress(raw, 6)) + ck(b"IEND", b""))


def _logres(Z, ds, m):
    lg = lambda x: np.log(np.clip(x, 1e-3, None))
    return float(np.abs(lg(Z) - lg(ds))[m].mean())


def _ckpt_name(d):
    p = os.path.normpath(d).split(os.sep)
    for x in reversed(p):
        if x.startswith("checkpoint-"):
            return x.replace("checkpoint-", "CK")
    return p[-1][:10].upper()


def _same_window(d0, d1, metaname):
    """Two runs must cover the same sequences and windows or the panels are not comparable.
    Each run writes its own samples_meta.json, so this has to be checked, not assumed."""
    def key(d):
        p = os.path.join(d, metaname)
        if not os.path.exists(p):
            raise SystemExit(f"[fatal] --compare: {p} not found -- is that a pred_depth dir?")
        m = json.load(open(p))
        return [(m[k]["seq_dir"], int(m[k]["start_frame"])) for k in sorted(m, key=int)]
    a_, b_ = key(d0), key(d1)
    if a_ != b_:
        n = sum(1 for x, y in zip(a_, b_) if x != y)
        raise SystemExit(f"[fatal] --compare window mismatch: {len(a_)} vs {len(b_)} samples, "
                         f"{n} differing. The panels would show different scenes.")


def _one(a, meta, keys, rdir, KS, want_gt):
    """Pairwise matrix for ONE results_dir. Reads rdir, never a.results_dir -- getting that
    wrong makes every panel show the same run, with no error and plausible numbers."""
    N = len(KS)
    accG = np.zeros((N, N)); accP = np.zeros((N, N)); cnt = np.zeros((N, N))
    used = 0
    for k in keys:
        mt = meta[k]
        sd = resolve(mt["seq_dir"], a.pod_root, a.mac_root)
        start = int(mt["start_frame"])
        fd = os.path.join(rdir, f"sample_{int(k):02d}_pred_depth.npy")
        if not (os.path.exists(fd) and os.path.isdir(sd)):
            continue
        pdp = np.load(fd).astype(np.float64)
        T = pdp.shape[0]
        if max(KS) >= T:
            raise SystemExit(f"[fatal] --frames out of range: max {max(KS)} >= {T}")
        gp = np.load(os.path.join(sd, "poses.npy"))
        i0 = np.load(os.path.join(sd, "intrinsics.npy")).astype(float)
        work = a.work or pdp.shape[-1]
        key = [start + 4*i for i in range(T)]
        dgn = [np.load(os.path.join(sd, "depth", f"{i:06d}.npy")).astype(np.float64) for i in key]
        intr = tuple(x*work/float(dgn[0].shape[0]) for x in i0)
        dg = np.stack([to_work(d, work) for d in dgn])
        dpw = np.stack([to_work(pdp[i], work) for i in range(T)])
        s, t = fit_affine_window(dpw, dg)
        dp = 1.0/np.clip(s*dpw + t, 1.0/FAR, 1.0/NEAR)
        cg = [carla_c2w(gp[i]) for i in key]
        for ia, ka in enumerate(KS):
            for ib, kb in enumerate(KS):
                if ia == ib:
                    continue
                base = gt_mask(dg[ka], dg[kb], cg[ka], cg[kb], intr)
                Zg, dsg, okg = transport(dg[ka], dg[kb], cg[ka], cg[kb], intr)
                Zp, dsp, okp = transport(dp[ka], dp[kb], cg[ka], cg[kb], intr)
                m = base & okg & okp          # one common mask, as everywhere else here
                if m.sum() < 64:
                    continue
                accG[ia, ib] += _logres(Zg, dsg, m)
                accP[ia, ib] += _logres(Zp, dsp, m)
                cnt[ia, ib] += 1
        used += 1
    if used == 0:
        raise SystemExit(f"[fatal] --matrix: no usable sample in {rdir}")
    G = accG/np.maximum(cnt, 1); P = accP/np.maximum(cnt, 1)
    return (G if want_gt else None), P, used


def _matrix(a, meta, keys):
    KS = [int(x) for x in a.frames.split(",") if x.strip()]
    N = len(KS)
    dirs = [a.results_dir] + ([d for d in a.compare.split(",") if d.strip()] if a.compare else [])
    for d in dirs[1:]:
        _same_window(dirs[0], d, a.meta)
    labs = ([x for x in a.labels.split(",")] if a.labels else [_ckpt_name(d) for d in dirs])
    if len(labs) != len(dirs):
        raise SystemExit(f"[fatal] --labels {len(labs)} != {len(dirs)} dirs")
    G, Ps = None, []
    for i, d in enumerate(dirs):
        g, p, used = _one(a, meta, keys, d, KS, want_gt=(i == 0))
        if i == 0:
            G = g
        Ps.append(p)
        print(f"[matrix] {labs[i]:>10s}  samples {used}  dir {d}")
    off = ~np.eye(N, dtype=bool)
    def _show(M, nm):
        print(f"\n{nm}   row = from frame, col = to frame   mean |log Z - log d|")
        print("        " + "".join(f"{x:>8d}" for x in KS))
        for i, x in enumerate(KS):
            print(f"  k={x:<4d}" + "".join("       ." if i == j else f"{M[i,j]:8.4f}"
                                           for j in range(N)))
        print(f"  off-diagonal median {np.median(M[off]):.4f}   max {M[off].max():.4f}")
    print(f"[matrix] frames {KS}")
    _show(G, "GT depth (floor)")
    for lab, P in zip(labs, Ps):
        _show(P, lab)
        print(f"  {lab} / GT  (median) = {np.median(P[off])/max(np.median(G[off]),1e-9):.2f}x")
    if not a.out_png:
        return
    # One scale over every prediction panel. Per-panel scaling would make the worse run look
    # the same as the better one -- which is the entire comparison.
    VMAX = float(a.vmax) if a.vmax else float(np.percentile(np.concatenate([p[off] for p in Ps]), 95))
    CELL, PAD = 86, 46
    def panel(M):
        H, W = PAD + N*CELL + 30, PAD + N*CELL
        im = np.zeros((H, W, 3), np.uint8)
        for i in range(N):
            for j in range(N):
                y0, x0 = PAD + i*CELL, PAD + j*CELL
                if i == j:
                    im[y0:y0+CELL-2, x0:x0+CELL-2] = 28
                else:
                    im[y0:y0+CELL-2, x0:x0+CELL-2] = _ramp(np.array(M[i, j]/VMAX))
                    c = (0, 0, 0) if M[i, j]/VMAX > 0.55 else (255, 255, 255)
                    _text(im, f"{M[i,j]:.3f}"[1:], x0+8, y0+CELL//2-8, 2, c)
            _text(im, f"k{KS[i]}", 6, PAD+i*CELL+CELL//2-5, 2)
            _text(im, f"k{KS[i]}", PAD+i*CELL+CELL//2-14, 16, 2)
        _text(im, f"{np.median(M[off]):.4f}", 6, PAD+N*CELL+8, 2)
        return im
    gap = np.full((PAD + N*CELL + 30, 10, 3), 40, np.uint8)
    cols = [panel(G)]
    for P in Ps:
        cols += [gap, panel(P)]
    out = np.concatenate(cols, 1)
    bar = np.zeros((26, out.shape[1], 3), np.uint8)
    bar[6:20, 10:410] = _ramp(np.linspace(0, 1, 400))[None].repeat(14, 0)
    _text(bar, "0.000", 10, 7, 2); _text(bar, f"{VMAX:.3f}", 420, 7, 2)
    _png(a.out_png, np.concatenate([out, bar], 0))
    print(f"  -> {a.out_png}   panels: GT, " + ", ".join(labs) + f"   scale 0 .. {VMAX:.3f}")



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--meta", default="samples_meta.json")
    ap.add_argument("--num_samples", type=int, default=0)
    ap.add_argument("--work", type=int, default=0, help="0 = 예측 depth 네이티브")
    ap.add_argument("--pod_root", default="/root")
    ap.add_argument("--mac_root", default="/Users/mac_al03030290/mnt/root")
    ap.add_argument("--out_csv", default=None)
    ap.add_argument("--matrix", action="store_true",
                    help="N x N pairwise consistency instead of the adjacent-pair scalars. "
                         "Cell (a,b) = mean |log Z_{a->b} - log d_b| over the GT-visible mask, "
                         "for GT depth and for the prediction. Occupancy-style metrics (do two "
                         "frames land in the same voxel) were tried and INVERT -- a blurry cloud "
                         "fills bins more evenly and beat GT twice -- so this stays "
                         "correspondence-based: the same pixel's two estimates.")
    ap.add_argument("--frames", default="4,6,8,10,12,14",
                    help="--matrix only: latent frames to cross-compare.")
    ap.add_argument("--vmax", type=float, default=None,
                    help="--matrix only: pin the colour scale so checkpoints are comparable. "
                         "Without it each run auto-scales and improvement becomes invisible.")
    ap.add_argument("--compare", default=None,
                    help="--matrix only: extra results_dirs (comma separated) drawn as further "
                         "panels. GT is identical across runs so it is drawn ONCE on the left; "
                         "a separate figure per run would put a different GT beside each and "
                         "nothing would line up. All panels share one colour scale.")
    ap.add_argument("--labels", default=None,
                    help="--compare only: comma-separated panel names. Default = checkpoint dir.")
    ap.add_argument("--out_png", default=None,
                    help="--matrix only: write the heatmap. numpy-only PNG writer on purpose -- "
                         "this tool imports no cv2, so it needs no libGL.")
    a = ap.parse_args()
    if a.out_csv and a.out_csv.endswith("_metrics.csv"):
        raise SystemExit("[fatal] --out_csv 에 '_metrics' 금지 (mean_metrics.py:47 glob 충돌)")

    meta = json.load(open(os.path.join(a.results_dir, a.meta)))
    keys = sorted(meta, key=lambda k: int(k))
    # Must precede the matrix hook. The other way round the flag is silently ignored in
    # matrix mode, and a flag that quietly does nothing is worse than one that errors.
    if a.num_samples:
        keys = keys[:a.num_samples]
    if a.matrix:
        _matrix(a, meta, keys)
        return

    rows, skipped, no_pose = [], 0, 0
    print(f"{'sample':11s} {'A 바닥':>9s} {'B GTcam':>9s} {'C predcam':>10s} | "
          f"{'okB%':>6s} {'okC%':>6s} {'mask%':>6s}")
    for k in keys:
        mt = meta[k]
        sd = resolve(mt["seq_dir"], a.pod_root, a.mac_root)
        start = int(mt["start_frame"])
        tag = f"sample_{int(k):02d}"
        fd = os.path.join(a.results_dir, f"{tag}_pred_depth.npy")
        fp = os.path.join(a.results_dir, f"{tag}_pred_poses.npy")
        if not (os.path.exists(fd) and os.path.isdir(sd)):
            skipped += 1
            continue
        has_pose = os.path.exists(fp)
        pdp = np.load(fd).astype(np.float64)
        gp = np.load(os.path.join(sd, "poses.npy"))
        i0 = np.load(os.path.join(sd, "intrinsics.npy")).astype(float)
        T, hh, ww = pdp.shape
        assert hh == ww, f"{tag}: 정사각 아님 {pdp.shape}"
        work = a.work or hh
        key = [start + 4*i for i in range(T)]        # latent k <-> 픽셀 4k
        assert max(key) < len(gp), f"{tag}: poses {len(gp)} <= {max(key)}"
        dgn = [np.load(os.path.join(sd, "depth", f"{i:06d}.npy")).astype(np.float64) for i in key]
        gt_res = dgn[0].shape[0]
        # intrinsics 는 GT depth 네이티브 단위 -> 작업 격자. 512 를 가정하지 않는다.
        intr = tuple(x*work/float(gt_res) for x in i0)
        dg = np.stack([to_work(d, work) for d in dgn])
        dpw = np.stack([to_work(pdp[i], work) for i in range(T)])
        s, t = fit_affine_window(dpw, dg)
        dp = 1.0/np.clip(s*dpw + t, 1.0/FAR, 1.0/NEAR)       # 양쪽 clamp
        cg = [carla_c2w(gp[i]) for i in key]
        cp = None
        if has_pose:
            pp = np.load(fp)
            if pp.shape[0] >= 4*(T - 1) + 1:
                cp = [carla_c2w(pp[4*i]) for i in range(T)]
            else:
                has_pose = False
        if not has_pose:
            no_pose += 1
        r = {x: [] for x in "ABC"}
        okf = {"B": [], "C": []}
        mfr = []
        for i in range(T - 1):
            base = gt_mask(dg[i], dg[i+1], cg[i], cg[i+1], intr)
            if base.sum() < 16:
                continue
            combos = {"A": (dg[i], dg[i+1], cg[i], cg[i+1]),
                      "B": (dp[i], dp[i+1], cg[i], cg[i+1])}
            if cp is not None:
                combos["C"] = (dp[i], dp[i+1], cp[i], cp[i+1])
            res = {n: transport(*v, intr) for n, v in combos.items()}
            common = base
            for n in res:
                common = common & res[n][2]
            if common.sum() < 16:
                continue
            for n, (Z, ds, _) in res.items():
                r[n].append(float((np.abs(Z - ds)/np.maximum(Z, NEAR))[common].mean()))
            for n in ("B", "C"):
                if n in res:
                    okf[n].append(float((base & res[n][2]).sum()/base.sum()))
            mfr.append(float(common.sum()/common.size))
        if not r["A"]:
            skipped += 1
            continue
        row = dict(sample=tag, res=work, affine_s=s, affine_t=t,
                   A=float(np.mean(r["A"])), B=float(np.mean(r["B"])),
                   C=(float(np.mean(r["C"])) if r["C"] else float("nan")),
                   okB=float(np.mean(okf["B"])),
                   okC=(float(np.mean(okf["C"])) if okf["C"] else float("nan")),
                   mask=float(np.mean(mfr)))
        rows.append(row)
        print(f"{tag:11s} {row['A']:9.5f} {row['B']:9.5f} {row['C']:10.5f} | "
              f"{row['okB']*100:5.1f}% {row['okC']*100:5.1f}% {row['mask']*100:5.1f}%")

    if not rows:
        raise SystemExit("[fatal] 유효한 샘플이 0개다")
    A = np.array([q["A"] for q in rows]); B = np.array([q["B"] for q in rows])
    C = np.array([q["C"] for q in rows])
    mA, mB = np.median(A), np.median(B)
    print(f"\n{'='*74}\n  상대 기하 잔차 |Z-d_s|/Z, 인접 latent 쌍, {len(rows)} 샘플 "
          f"({skipped} 건너뜀), {rows[0]['res']}x{rows[0]['res']}\n{'='*74}")
    print(f"  A  GT depth   + GT 카메라    (바닥)         median {mA:.5f}")
    print(f"  B  예측 depth + GT 카메라    **R1 지표**    median {mB:.5f}   = 바닥 {mB/mA:.2f}x")
    if np.isfinite(C).any():
        mC = np.median(C[np.isfinite(C)])
        print(f"  C  예측 depth + 예측 카메라  **R2 지표**    median {mC:.5f}   = 바닥 {mC/mA:.2f}x")
    else:
        print(f"  C  -- _pred_poses.npy 없음 ({no_pose}). invert_raymap.py --stage s3 필요")
    print(f"\n  32^2@8000 기준선: B/A = 4.92x,  C/A = 3.90x")
    if a.out_csv:
        p = a.out_csv if os.path.isabs(a.out_csv) else os.path.join(a.results_dir, a.out_csv)
        with open(p, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
        print(f"\n  [out] {p}")


if __name__ == "__main__":
    main()
