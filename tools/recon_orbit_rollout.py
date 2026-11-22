#!/usr/bin/env python3
"""rollout 산출물로 **긴 궤적** 점구름 orbit 을 만든다 (recon_orbit.py 의 rollout 판).

recon_orbit.py 는 단일 77프레임 윈도(latent 20장, 카메라 이동 6.0 m)를 가정하고 bin 범위가
+-20/22 m 로 박혀 있다. rollout 은 청크 5개 x 77프레임 = 381프레임, 이동 33.8 m, 반경 p95
32.4 m 라 그대로는 심하게 잘린다. 그래서 별도 파일로 둔다 -- recon_orbit.py 는 건드리지 않는다.

입력 (infer_rollout.py 산출):
  rollout_meta.json              start_frame / F / overlap / num_chunks / num_frames
  rollout_NN_pred_depth.npy      (K, T_lat, h, w)  SI-disparity, 청크별
  rollout_NN_pred_poses.npy      (선택) 있으면 예측 카메라 패널까지 4패널

프레임 매핑: 청크 c 의 latent k  <->  픽셀 프레임  start + c*(F-overlap) + 4k.
overlap=1 이면 청크 c 의 latent 19 가 청크 c+1 의 latent 0 과 **같은 프레임**이므로 한 번만 쓴다
(5x20 - 4 = 96 고유 프레임).

depth 스케일: 청크마다 affine 이 독립이지만 **전역 affine 하나**로 GT 에 맞춘다. 청크별로 맞추면
청크 간 일관성을 GT 가 공짜로 주는 꼴이 되어 '최선의 경우' 수치가 된다.
"""
import argparse, json, math, os
import cv2, numpy as np
import imageio.v2 as imageio

_C3 = np.array([[0, 0, 1], [1, 0, 0], [0, -1, 0]], float)
NEAR, FAR = 0.5, 200.0


def c2w(v):
    x, y, z, rr, pp, yy = [float(t) for t in v[:6]]
    r, p, w = map(math.radians, (rr, pp, yy))
    cr, sr = math.cos(r), math.sin(r); cp, sp = math.cos(p), math.sin(p); cy, sy = math.cos(w), math.sin(w)
    M = np.eye(4)
    M[0, 0] = cy * cp; M[0, 1] = cy * sp * sr - sy * cr; M[0, 2] = cy * sp * cr + sy * sr
    M[1, 0] = sy * cp; M[1, 1] = sy * sp * sr + cy * cr; M[1, 2] = sy * sp * cr - cy * sr
    M[2, 0] = -sp;     M[2, 1] = cp * sr;                M[2, 2] = cp * cr
    M[:3, 3] = (x, y, z)
    return M


def dec(a, o):
    H, W = a.shape
    yi = ((np.arange(o) + .5) * H / o).astype(int).clip(0, H - 1)
    xi = ((np.arange(o) + .5) * W / o).astype(int).clip(0, W - 1)
    return a[yi][:, xi]


def imgdec(path, o):
    im = cv2.imread(path, cv2.IMREAD_COLOR)                 # BGR uint8
    if im is None:
        raise SystemExit(f"[fatal] 이미지를 못 읽었다: {path}")
    return np.stack([dec(im[..., c], o) for c in range(3)], -1)


def unproj(d, C, intr):
    h, w = d.shape; fx, fy, cx, cy = intr
    vv, uu = np.mgrid[0:h, 0:w]; U, V = uu + .5, vv + .5
    # errstate: BLAS matmul 커널이 입력/출력이 모두 유한한데도 FP 플래그를 세운다 (numpy 2.0.2
    # 에서 확인: einsum 결과와 바이트 동일하고 출력도 전부 유한). 가짜 경고라 억제한다 --
    # 데이터 이상이 아니다 (GT depth 비유한 0개, 0 이하 0개, poses 전부 유한).
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        p = np.stack([(U - cx) / fx * d, (V - cy) / fy * d, d], -1) @ _C3.T
        return p.reshape(-1, 3) @ C[:3, :3].T + C[:3, 3]


def build(roll_dir, sample, max_depth, stride, want_rgb, mask_pred=False):
    """렌더 이전의 모든 것. cv2/imageio 없이도(want_rgb=False) 돌아가 테스트가 쉽다."""
    meta = json.load(open(os.path.join(roll_dir, "rollout_meta.json")))
    key = str(sample) if str(sample) in meta else f"{sample:02d}"
    if key not in meta:
        raise SystemExit(f"[fatal] rollout_meta.json 에 샘플 {sample} 없음. 있는 키: "
                         f"{sorted(meta)[:5]}...")
    v = meta[key]
    sd = v["seq_dir"]
    st, F, OV, K = int(v["start_frame"]), int(v["F"]), int(v["overlap"]), int(v["num_chunks"])

    p = np.load(os.path.join(roll_dir, f"rollout_{sample:02d}_pred_depth.npy")).astype(np.float64)
    if p.ndim == 5 and p.shape[2] == 1:
        p = p[:, :, 0]
    if p.ndim != 4:
        raise SystemExit(f"[fatal] _pred_depth shape {p.shape} (K,T_lat,h,w) 여야 한다")
    if p.shape[0] != K:
        raise SystemExit(f"[fatal] depth 청크 {p.shape[0]} != meta num_chunks {K}")
    T_lat, W_ = p.shape[1], p.shape[-1]

    # 이음새 프레임 중복 제거
    seen, FR = set(), []
    for c in range(K):
        for k in range(T_lat):
            f = st + c * (F - OV) + 4 * k
            if f in seen:
                continue
            seen.add(f); FR.append((c, k, f))

    gp = np.load(f"{sd}/poses.npy")
    i0 = np.load(f"{sd}/intrinsics.npy").astype(float)
    nd = np.load(f"{sd}/depth/{FR[0][2]:06d}.npy").shape[0]
    intr = tuple(x * W_ / nd for x in i0)
    dg = {f: dec(np.load(f"{sd}/depth/{f:06d}.npy").astype(np.float64), W_) for _, _, f in FR}

    # 전역 affine (모든 청크 공통)
    P = np.concatenate([p[c, k][(dg[f] > NEAR) & (dg[f] < FAR)] for c, k, f in FR])
    T = np.concatenate([(1.0 / np.clip(dg[f], 1e-3, None))[(dg[f] > NEAR) & (dg[f] < FAR)]
                        for c, k, f in FR])
    n = len(P)
    A = np.array([[(P * P).sum(), P.sum()], [P.sum(), n]])
    b = np.array([(P * T).sum(), T.sum()])
    s, t = np.linalg.solve(A + 1e-9 * np.eye(2), b)

    # 예측 포즈 (있으면)
    pp_path = os.path.join(roll_dir, f"rollout_{sample:02d}_pred_poses.npy")
    cp = None
    if os.path.exists(pp_path):
        raw = np.load(pp_path)
        if raw.shape[0] < FR[-1][2] - st + 1:
            print(f"[warn] _pred_poses 길이 {raw.shape[0]} 가 짧다 -> 예측 카메라 패널 생략")
        else:
            cp = {f: c2w(raw[f - st]) for _, _, f in FR}
    cg = {f: c2w(gp[f]) for _, _, f in FR}

    dp = {f: 1.0 / np.clip(s * p[c, k] + t, 1 / FAR, 1 / NEAR) for c, k, f in FR}
    masks = {f: ((dg[f] > NEAR) & (dg[f] < max_depth)).ravel() for _, _, f in FR}
    if mask_pred:   # drop PRED-far pixels from EVERY panel (GT too): same pixel set, no scatter
        n0 = sum(int(m.sum()) for m in masks.values())
        masks = {f: masks[f] & (dp[f] < max_depth).ravel() for _, _, f in FR}
        n1 = sum(int(m.sum()) for m in masks.values())
        print(f"[info] --mask_pred: dropped {100 * (1 - n1 / max(n0, 1)):.2f}% of pixels "
              f"(PRED depth >= {max_depth} m) from all panels")

    combos = [("GT d + GT pose", dg, cg), ("PRED d + GT pose", dp, cg)]
    if cp is not None:
        combos += [("GT d + PRED pose", dg, cp), ("PRED d + PRED pose", dp, cp)]

    clouds = []
    for _, dsrc, psrc in combos:
        clouds.append(np.concatenate([unproj(dsrc[f], psrc[f], intr)[masks[f]][::stride]
                                      for _, _, f in FR]))
    ctr = np.median(clouds[0], 0)                 # GT 구름 기준 (pred outlier 가 시야를 못 흔든다)

    RGB = None
    if want_rgb:
        RGB = np.concatenate([imgdec(f"{sd}/images/{f:06d}.png", W_).reshape(-1, 3)[masks[f]][::stride]
                              for _, _, f in FR]).astype(np.float64)
        if len(RGB) != len(clouds[0]):
            raise SystemExit(f"[fatal] RGB {len(RGB)} != 점 {len(clouds[0])}")

    trav = sum(float(np.linalg.norm(cg[FR[i + 1][2]][:3, 3] - cg[FR[i][2]][:3, 3]))
               for i in range(len(FR) - 1))
    cam_tracks = [np.stack([psrc[f][:3, 3] for _, _, f in FR]) for _, _, psrc in combos]
    # per-frame point counts (same order as the concatenated clouds) -> lets recon_grow_rollout.py slice by frame
    counts = [len(range(0, int(masks[f].sum()), stride)) for _, _, f in FR]
    info = dict(seq_dir=sd, start=st, F=F, overlap=OV, K=K, T_lat=T_lat, W=W_,
                cam_tracks=cam_tracks, frames=[f for _, _, f in FR], counts=counts,
                cam_rots=[np.stack([psrc[f][:3, :3] for _, _, f in FR]) for _, _, psrc in combos],
                n_frames=len(FR), affine=(float(s), float(t)), travel_m=trav,
                has_pred_pose=cp is not None)
    return [c[0] for c in combos], clouds, ctr, RGB, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roll_dir", required=True, help=".../rollout-f77/<tag>/<ckpt>")
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=72)
    ap.add_argument("--elev", type=float, default=22.0)
    ap.add_argument("--max_depth", type=float, default=35.0)
    ap.add_argument("--mask_pred", action="store_true",
                    help="also drop pixels whose PRED depth >= --max_depth, in EVERY panel (GT too) so all panels keep the same pixel set; removes far scatter. Render only.")
    ap.add_argument("--size", type=int, default=400)
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--color", choices=["rgb", "height", "density"], default="rgb")
    ap.add_argument("--view", choices=["orbit", "bev"], default="orbit",
                    help="orbit: 360도 궤도 mp4. bev: 위에서 내려본 1장 PNG + 점유 면적(m^2). "
                         "면적이 '얼마나 탐험했나' 다 -- GT 패널이 기준이고, 예측이 작으면 "
                         "못 본 것, 크면 번져서 없는 데까지 칠한 것이다.")
    ap.add_argument("--traj", action="store_true", default=True,
                    help="BEV 에 그 패널이 쓰는 카메라 경로를 겹쳐 그린다 (기본 켜짐)")
    ap.add_argument("--no_traj", dest="traj", action="store_false")
    ap.add_argument("--area_min_pts", type=int, default=3,
                    help="탐험 면적을 셀 때 요구하는 셀당 최소 점 수. 1 로 두면 '튄 점 하나' 도 "
                         "탐험으로 세어 면적이 부풀려진다 -- 실측: GT 가 35 m 이내인 픽셀의 "
                         "1.03%%가 PRED 에서 35 m 밖으로 날아가고, 원거리는 희소해 점 하나가 "
                         "셀 하나를 차지하므로 점유 셀이 48%% 부푼다. >=5 로 올리면 카메라만 "
                         "틀린 패널은 1.29x -> 1.02x 로 GT 와 같아지고(= outlier 산물), depth "
                         "패널은 1.48x -> 1.36x 로 남는다(= 진짜 번짐).")
    ap.add_argument("--extent", type=float, default=0.0, help="0 = GT 구름 percentile 에서 자동")
    ap.add_argument("--extent_pct", type=float, default=98.0)
    a = ap.parse_args()

    names, clouds, ctr, RGB, info = build(a.roll_dir, a.sample, a.max_depth, a.stride,
                                          a.color == "rgb", mask_pred=a.mask_pred)
    print(f"[info] {info['n_frames']}프레임 ({info['K']}청크 x {info['T_lat']} - 이음새), "
          f"점 {len(clouds[0]):,}/조합, 패널 {len(names)}개")
    print(f"[info] 전역 affine s={info['affine'][0]:.4f} t={info['affine'][1]:.4f}, "
          f"카메라 이동 {info['travel_m']:.1f} m")

    S = a.size + (a.size & 1)                      # libx264/yuv420p 는 짝수 치수를 요구한다
    Q0 = clouds[0] - ctr
    if a.extent > 0:
        EX = a.extent
    else:
        EX = float(np.percentile(np.abs(np.concatenate([Q0[:, 0], Q0[:, 1]])), a.extent_pct))
        EX = max(EX, 5.0)
    print(f"[info] extent +-{EX:.1f} m ({'수동' if a.extent > 0 else f'GT p{a.extent_pct:.0f} 자동'})")
    # square cells: the x range is 1.1x the y range, so give it 1.1x the columns (even for x264)
    SX = int(round(S * 1.1)); SX += SX & 1
    BY = np.linspace(-EX, EX, S + 1); BX = np.linspace(-EX * 1.1, EX * 1.1, SX + 1)

    if a.color == "height":
        ATTR = [P[:, 2] - ctr[2] for P in clouds]
        LO, HI = np.percentile(np.concatenate(ATTR), [2, 98])
        CBL = f"color = height {LO:+.1f} .. {HI:+.1f} m   |   brightness = log density"
    elif a.color == "rgb":
        ATTR = None; LO, HI = 0.0, 255.0
        CBL = (f"color = GT RGB   |   {info['n_frames']} frames, {info['travel_m']:.0f} m travel"
               f"   |   extent +-{EX:.0f} m")
    else:
        ATTR = None; LO, HI = 0.0, 1.0; CBL = "color = log density"

    def _txt(im, s_, xy, sc=0.40):
        cv2.putText(im, s_, xy, cv2.FONT_HERSHEY_SIMPLEX, sc, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(im, s_, xy, cv2.FONT_HERSHEY_SIMPLEX, sc, (255, 255, 255), 1, cv2.LINE_AA)

    # 프레임 밖 비율 -- 클리핑을 숨기지 말고 측정한다
    for nm, P in zip(names, clouds):
        Q = P - ctr
        out = float(((np.abs(Q[:, 0]) > EX) | (np.abs(Q[:, 1]) > EX * 1.1)).mean())
        print(f"[info]   {nm:<20} 프레임 밖 {100 * out:5.1f}%")

    if a.view == "bev":
        # 세계 축을 직접 쓴다. 궤도 기저는 elev=90 에서 cross(fwd,up)=0 으로 퇴화한다
        # (실측: 22deg 0.927 / 89deg 0.017 / 90deg 0.000000 -> rgt /= norm 에서 0 나눗셈).
        xh = np.array([1.0, 0.0, 0.0]); yh = np.array([0.0, 1.0, 0.0])
        cellA = (2 * EX / S) * (2 * EX * 1.1 / SX)
        print(f"[info] BEV 셀 {math.sqrt(cellA) * 100:.1f} cm 변 / {cellA * 1e4:.1f} cm^2")
        row, base_area = [], None
        for k, (nm, P) in enumerate(zip(names, clouds)):
            # CARLA is left-handed: seen from above +y points DOWN, so row ~ +y (not -y).
            # The old -y mirrored the BEV; recon_bev.py has always used +y.
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                Q = P - ctr; yv = Q @ yh; xv = Q @ xh     # 가짜 BLAS FP 플래그 억제
            cnt, _, _ = np.histogram2d(yv, xv, bins=[BY, BX])
            dens = np.log1p(cnt); dens = dens / max(dens.max(), 1e-9)
            if a.color == "rgb":
                cs = np.stack([np.histogram2d(yv, xv, bins=[BY, BX], weights=RGB[:, c])[0]
                               for c in range(3)], -1)
                im = cs / np.maximum(cnt, 1.0)[..., None]
                im = (im * (0.25 + 0.75 * dens ** 0.5)[..., None]).clip(0, 255).astype(np.uint8)
            elif ATTR is None:
                im = cv2.applyColorMap((dens * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
                im[cnt == 0] = 0
            else:
                asum, _, _ = np.histogram2d(yv, xv, bins=[BY, BX], weights=ATTR[k])
                am = asum / np.maximum(cnt, 1.0)
                hn = np.clip((am - LO) / max(HI - LO, 1e-9), 0, 1)
                col = cv2.applyColorMap((hn * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
                im = (col * dens[..., None]).astype(np.uint8)
            occ = int((cnt >= a.area_min_pts).sum()); area = occ * cellA
            occ1 = int((cnt > 0).sum())
            if k == 0:
                base_area = area
            ratio = area / max(base_area, 1e-9)
            if a.traj:                      # 경로 없이는 BEV 를 읽기 어렵다
                TR = info["cam_tracks"][k] - ctr
                px = ((TR @ xh - BX[0]) / (BX[-1] - BX[0]) * SX).astype(np.int32)
                py = (((TR @ yh) - BY[0]) / (BY[-1] - BY[0]) * S).astype(np.int32)
                cv2.polylines(im, [np.stack([px, py], -1).reshape(-1, 1, 2)], False,
                              (0, 255, 255), 2, cv2.LINE_AA)
                cv2.circle(im, (int(px[0]), int(py[0])), 4, (0, 255, 0), -1)
            _txt(im, nm, (6, 20))
            _txt(im, f"{area:.0f} m2  ({ratio:.2f}x GT)  >={a.area_min_pts}pt", (6, 40), 0.36)
            print(f"[info]   {nm:<20} 면적 {area:7.1f} m2 ({ratio:.2f}x GT, >={a.area_min_pts}점)"
                  f"   [>=1점: {occ1 * cellA:7.1f} m2]")
            row.append(im)
        mos = np.concatenate(row, 1)
        bar = np.zeros((24, mos.shape[1], 3), np.uint8)
        _txt(bar, CBL + f"   |   label = explored area (cells with >={a.area_min_pts} pts)",
             (12, 18), 0.42)
        mos = np.concatenate([mos, bar], 0)
        outp = a.out if a.out.lower().endswith(".png") else os.path.splitext(a.out)[0] + ".png"
        cv2.imwrite(outp, mos)
        sz = os.path.getsize(outp)
        print(f"[ok] {outp}  BEV {len(names)}패널, {SX * len(names)}x{S + 24}, {sz / 1e3:.0f} KB")
        return

    el = math.radians(a.elev)
    wr = imageio.get_writer(a.out, fps=a.fps, macro_block_size=1)
    for fi in range(a.frames):
        th = 2 * math.pi * fi / a.frames
        fwd = np.array([math.cos(th) * math.cos(el), math.sin(th) * math.cos(el), math.sin(el)])
        up = np.array([0, 0, 1.0]); rgt = np.cross(fwd, up); rgt /= np.linalg.norm(rgt)
        up2 = np.cross(rgt, fwd)
        row = []
        for k, (nm, P) in enumerate(zip(names, clouds)):
            Q = P - ctr; yv = -(Q @ up2); xv = Q @ rgt
            cnt, _, _ = np.histogram2d(yv, xv, bins=[BY, BX])
            dens = np.log1p(cnt); dens = dens / max(dens.max(), 1e-9)
            if a.color == "rgb":
                cs = np.stack([np.histogram2d(yv, xv, bins=[BY, BX], weights=RGB[:, c])[0]
                               for c in range(3)], -1)
                im = cs / np.maximum(cnt, 1.0)[..., None]
                im = (im * (0.25 + 0.75 * dens ** 0.5)[..., None]).clip(0, 255).astype(np.uint8)
            elif ATTR is None:
                im = cv2.applyColorMap((dens * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
                im[cnt == 0] = 0
            else:
                asum, _, _ = np.histogram2d(yv, xv, bins=[BY, BX], weights=ATTR[k])
                am = asum / np.maximum(cnt, 1.0)
                hn = np.clip((am - LO) / max(HI - LO, 1e-9), 0, 1)
                col = cv2.applyColorMap((hn * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
                im = (col * dens[..., None]).astype(np.uint8)
            _txt(im, nm, (6, 20))
            row.append(im)
        mos = np.concatenate(row, 1)
        bar = np.zeros((24, mos.shape[1], 3), np.uint8)
        _txt(bar, CBL, (12, 18), 0.42)
        mos = np.concatenate([mos, bar], 0)
        if mos.shape[0] & 1:
            mos = np.concatenate([mos, np.zeros((1,) + mos.shape[1:], np.uint8)], 0)
        wr.append_data(cv2.cvtColor(mos, cv2.COLOR_BGR2RGB))
    wr.close()
    sz = os.path.getsize(a.out)
    if sz < 20_000:
        raise SystemExit(f"[fatal] 출력이 {sz} B 뿐 -- 인코더가 프레임을 쓰지 못했다")
    print(f"[ok] {a.out}  {a.frames}프레임, {len(names)}패널, "
          f"{SX * len(names)}x{S + 24}, {sz / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
