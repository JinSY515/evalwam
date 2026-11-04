#!/usr/bin/env python3
"""rollout 의 청크별 _pred_raymap 을 역변환해 **하나의 긴 궤적** 포즈로 이어 붙인다.

invert_raymap.py 는 단일 윈도용이고 `sample_*_action_poses.npy` 로 glob 하므로 rollout
(`rollout_NN_*`, 청크 축이 하나 더 있는 _pred_raymap)에는 그대로 쓸 수 없다. 여기서는 그
파일의 검증된 함수들(load_vae / decode_raymap_rays / invert_rays)을 **그대로 재사용**하고
청크 체이닝만 추가한다.

체이닝: invert_rays 는 `anchor_pose`(6-vector) 에 청크의 **프레임 0** 을 맞춰 절대 포즈를 돌려준다.
overlap=1 이면 청크 c 의 마지막 프레임과 청크 c+1 의 프레임 0 이 **같은 픽셀 프레임**이므로,
청크 c+1 의 앵커로 청크 c 의 **예측** 마지막 포즈를 넣으면 드리프트가 누적된 하나의 궤적이 된다.
길이: 77 + (K-1)*76 = 381  (K=5, F=77, overlap=1)

--anchor gt 는 청크마다 GT 로 재앵커하는 **대조군**이다 (누적 없이 청크별 오차만 본다).
기본값 chain 이 실제로 보려는 것 -- 재앵커하면 드리프트가 사라져 측정 대상이 사라진다.
"""
import argparse, importlib.util, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def _load_ir():
    """invert_raymap.py 를 모듈로 올린다 (모듈 레벨은 경로 상수뿐이고 main() 은 가드됨)."""
    p = os.path.join(HERE, "invert_raymap.py")
    if not os.path.exists(p):
        raise SystemExit(f"[fatal] {p} 없음")
    spec = importlib.util.spec_from_file_location("invert_raymap", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def stitch(lat_all, gt_poses, st, F, OV, K, intr4, orig_wh, moment_scale,
           decode, invert, anchor_mode="chain"):
    """청크별 역변환 + 체이닝. decode/invert 를 인자로 받아 GPU 없이 테스트 가능하게 한다.

    decode(lat_c)          -> (dirs, moments)
    invert(d, m, anchor)   -> (abs (F,6), rel (F,6), diag)
    """
    out, diags, anchor = None, [], np.asarray(gt_poses[st], dtype=np.float64)
    for c in range(K):
        d, m = decode(lat_all[c])
        pa, pr, dg = invert(d, m, anchor)
        pa = np.asarray(pa, dtype=np.float64)
        if pa.shape[0] != F:
            raise SystemExit(f"[fatal] 청크 {c}: 역변환이 {pa.shape[0]}프레임, F={F} 기대")
        out = pa if c == 0 else np.concatenate([out, pa[1:]], 0)   # 이음새 1프레임 중복 제거
        dg = dict(dg); dg["chunk"] = c
        diags.append(dg)
        nxt = st + (c + 1) * (F - OV)
        anchor = pa[-1] if anchor_mode == "chain" else np.asarray(gt_poses[nxt],
                                                                  dtype=np.float64)
    want = F + (K - 1) * (F - OV)
    if out.shape[0] != want:
        raise SystemExit(f"[fatal] 이어 붙인 길이 {out.shape[0]} != 기대 {want}")
    return out, diags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roll_dir", required=True)
    ap.add_argument("--samples", default="all", help="쉼표 구분 또는 all")
    ap.add_argument("--num_samples", type=int, default=0, help="0 = 제한 없음")
    ap.add_argument("--anchor", default="chain", choices=["chain", "gt"],
                    help="chain: 청크 c+1 을 청크 c 의 예측 마지막 포즈에 앵커 (드리프트 누적, "
                         "이게 보려는 것). gt: 청크마다 GT 재앵커 (대조군)")
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--ray_downsample", type=int, default=2)
    ap.add_argument("--raymap_moment_scale", type=float, default=8.0)
    ap.add_argument("--model_name", default="nvidia/Cosmos-Predict2.5-2B")
    ap.add_argument("--revision", default="diffusers/base/post-trained")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_csv", default="rollout_pose_metrics.csv")
    ap.add_argument("--no_save_poses", dest="save_poses", action="store_false", default=True)
    a = ap.parse_args()

    IR = _load_ir()
    import torch

    Hr, Wr = a.height // a.ray_downsample, a.width // a.ray_downsample
    meta = json.load(open(os.path.join(a.roll_dir, "rollout_meta.json")))
    keys = sorted(meta, key=lambda x: int(x)) if a.samples.strip().lower() in ("all", "*", "") \
        else [x.strip() for x in a.samples.split(",") if x.strip()]
    if a.num_samples:
        keys = keys[:a.num_samples]
    print(f"[cfg] ray grid {Hr}x{Wr}, latent {Hr // 8}x{Wr // 8}, "
          f"moment_scale {a.raymap_moment_scale}, anchor={a.anchor}")
    print(f"[cfg] {len(keys)} 샘플: {keys}")

    vae, lmean, lstd = IR.load_vae(a.model_name, a.revision, a.device, a.dtype)
    rows = []
    for key in keys:
        s = int(key)
        tag = f"rollout_{s:02d}"
        rp = os.path.join(a.roll_dir, f"{tag}_pred_raymap.npy")
        if not os.path.exists(rp):
            print(f"[{tag}] skip: no {os.path.basename(rp)}"); continue
        v = meta[key]
        sd = v["seq_dir"]
        st, F, OV, K = int(v["start_frame"]), int(v["F"]), int(v["overlap"]), int(v["num_chunks"])
        NF = int(v["num_frames"])
        lat_all = np.load(rp)
        if lat_all.shape[0] != K:
            print(f"[{tag}] skip: raymap 청크 {lat_all.shape[0]} != meta {K}"); continue
        gp = np.load(f"{sd}/poses.npy").astype(np.float64)
        ip = f"{sd}/intrinsics.npy"
        if not os.path.exists(ip):
            print(f"[{tag}] skip: no intrinsics.npy"); continue
        intr = np.load(ip).astype(np.float64).ravel()[:4].tolist()
        orig_wh = (a.width, a.height)

        def _decode(lat_c):
            lat = torch.from_numpy(np.asarray(lat_c)).to(a.device)
            if lat.ndim == 4:
                lat = lat.unsqueeze(0)
            if not (lat.shape[3] == Hr // 8 and lat.shape[2] == (F - 1) // 4 + 1):
                raise SystemExit(f"[fatal] {tag}: raymap latent {tuple(lat.shape)} 가 "
                                 f"--ray_downsample {a.ray_downsample} / F {F} 와 불일치")
            return IR.decode_raymap_rays(vae, lat, lmean, lstd, a.raymap_moment_scale)

        def _invert(d, m, anchor):
            return IR.invert_rays(d, m, intr, orig_wh, anchor, a.raymap_moment_scale)

        out, diags = stitch(lat_all, gp, st, F, OV, K, intr, orig_wh,
                            a.raymap_moment_scale, _decode, _invert, a.anchor)
        if out.shape[0] != NF:
            print(f"[{tag}] warn: 길이 {out.shape[0]} != meta num_frames {NF}")
        gtw = gp[st:st + out.shape[0]]
        row = {"sample": tag, "n": int(out.shape[0]), "K": K, "anchor": a.anchor}
        row.update(IR.pose_errors(out, gtw))
        row.update(IR.ate_rpe(out, gtw))
        for k in ("proc_resid_deg", "clamp_frac", "cond_G"):
            if k in diags[0]:
                row[k] = float(np.mean([d[k] for d in diags]))
                row[f"{k}_last"] = float(diags[-1][k])
        rows.append(row)
        if a.save_poses:
            np.save(os.path.join(a.roll_dir, f"{tag}_pred_poses.npy"), out)
        print(f"[{tag}] {out.shape[0]}프레임 ({K}청크)  ATE {row['ate_trans_rmse_m']:.4f} m / "
              f"rot {row['ate_rot_mean_deg']:.3f} deg  RPE1 {row['rpe1_trans_m']:.4f} m  "
              f"proc_resid {row.get('proc_resid_deg', float('nan')):.3f} deg")

    if not rows:
        raise SystemExit("[fatal] 처리된 샘플이 없다")
    if a.out_csv:
        import csv
        p = a.out_csv if os.path.isabs(a.out_csv) else os.path.join(a.roll_dir, a.out_csv)
        cols = sorted({k for r in rows for k in r})
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
        print(f"[done] {len(rows)}행 -> {p}")
    g = lambda k: float(np.median([r[k] for r in rows if k in r]))
    print(f"[median] ATE {g('ate_trans_rmse_m'):.4f} m / rot {g('ate_rot_mean_deg'):.3f} deg   "
          f"RPE1 {g('rpe1_trans_m'):.4f} m")


if __name__ == "__main__":
    main()
