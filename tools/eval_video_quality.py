#!/usr/bin/env python3
"""Video quality eval (offline) — same format as eval_camera_control.py.
Per generated sample vs GT (real CARLA) frames: PSNR / SSIM / LPIPS (frame-mean).
GT via out_dir/samples_meta.json {idx:{seq_dir,start_frame,num_frames}} -> images/{start+i:06d}.png."""
import os, sys, argparse, csv, glob, json, re
import numpy as np

def psnr(a, b):
    a=a.astype(np.float64); b=b.astype(np.float64); mse=np.mean((a-b)**2)
    return 99.0 if mse<=1e-10 else float(20*np.log10(255.0)-10*np.log10(mse))

_SSIM=None
def ssim(a,b):
    global _SSIM
    if _SSIM is None:
        try:
            from skimage.metrics import structural_similarity as _s; _SSIM=_s
        except Exception: _SSIM=False
    if _SSIM is False: return float("nan")
    return float(_SSIM(a,b,channel_axis=2,data_range=255))

_LP={"model":None,"torch":None,"tried":False,"dev":"cpu"}
def _lp_init(net):
    if _LP["tried"]: return
    _LP["tried"]=True
    try:
        import torch, lpips
        _LP["torch"]=torch; _LP["dev"]="cuda" if torch.cuda.is_available() else "cpu"
        _LP["model"]=lpips.LPIPS(net=net).to(_LP["dev"]).eval()
    except Exception as e:
        _LP["model"]=None; _LP["err"]=str(e)
def lpips_dist(a,b,net="alex"):
    _lp_init(net)
    if _LP["model"] is None: return float("nan")
    torch=_LP["torch"]
    def t(x): return (torch.from_numpy(x.astype(np.float32)/127.5-1.0).permute(2,0,1).unsqueeze(0).to(_LP["dev"]))
    with torch.no_grad(): return float(_LP["model"](t(a),t(b)).item())

def read_video(p):
    import imageio.v3 as iio; v=iio.imread(p); return [np.asarray(v[i]) for i in range(len(v))]
def read_img(p):
    from PIL import Image; return np.array(Image.open(p).convert("RGB"))
def resize_to(img,hw):
    from PIL import Image; H,W=hw
    return img if img.shape[:2]==(H,W) else np.array(Image.fromarray(img).resize((W,H),Image.BILINEAR))

def selftest():
    rng=np.random.default_rng(0); a=rng.integers(0,256,(64,64,3),dtype=np.uint8)
    assert psnr(a,a)>=80
    b=np.clip(a.astype(int)+rng.integers(-30,30,a.shape),0,255).astype(np.uint8)
    pn=psnr(a,b); assert pn<40
    print(f"[selftest] psnr id>=80 OK, noisy={pn:.2f}<40 OK; ssim={ssim(a,a)} lpips={lpips_dist(a,a)}")
    print("[selftest] PASS")

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--selftest",action="store_true")
    ap.add_argument("--results_dir"); ap.add_argument("--video_glob",default="sample_*.mp4")
    ap.add_argument("--meta",default="samples_meta.json"); ap.add_argument("--images_subdir",default="images")
    ap.add_argument("--out_csv",default="video_quality_metrics.csv"); ap.add_argument("--lpips_net",default="alex")
    ap.add_argument("--sample_re",default=r"sample_\d+(_rev)?\.mp4$",
                    help="Filename filter applied after --video_glob. The glob alone is not enough: a results dir also holds _pred/_raymap/_nomap/_condgt/_vis variants of the same sample. For the worldmap map-ablation pass use r'sample_\\d+(_rev)?_nomap\\.mp4$'.")
    args=ap.parse_args()
    if args.selftest: selftest(); return
    mp=os.path.join(args.results_dir,args.meta); meta=json.load(open(mp)) if os.path.exists(mp) else {}
    if not meta: print(f"[warn] {mp} 없음 — GT 매핑 불가")
    vids=sorted(glob.glob(os.path.join(args.results_dir,"**",args.video_glob),recursive=True))
    vids=[v for v in vids if re.match(args.sample_re, os.path.basename(v))]  # default: base sample_XX.mp4 only (skip _pred/_condgt/_raymap/_nomap/_vis)
    print(f"[info] {len(vids)} videos"); rows=[]
    for vp in vids:
        base=os.path.splitext(os.path.basename(vp))[0]; m=re.search(r"(\d+)",base); row={"sample":base}
        if m is None or str(int(m.group(1))) not in meta: row["error"]="no meta"; rows.append(row); continue
        info=meta[str(int(m.group(1)))]; seq_dir=info["seq_dir"]; start=int(info["start_frame"])
        # frame_stride: GT frame i of this sample is start+i*stride (egowm runs at stride 11,
        # ego_wam at 1). gen_stride/gen_offset subsample the GENERATED video the same way so a
        # dense prediction can be scored on a sparse model's timestamps. Defaults reproduce the
        # old behaviour exactly.
        st=int(info.get("frame_stride",1)); gs=int(info.get("gen_stride",1)); go=int(info.get("gen_offset",0))
        gen=read_video(vp)[go::gs]; img_dir=os.path.join(seq_dir,args.images_subdir); gt=[]
        for i in range(len(gen)):
            fp=os.path.join(img_dir,f"{start+i*st:06d}.png")
            if not os.path.exists(fp): break
            gt.append(read_img(fp))
        if not gt: row["error"]=f"no GT in {img_dir}"; rows.append(row); continue
        k=min(len(gen),len(gt)); hw=gen[0].shape[:2]; ps=[];ss=[];lp=[]
        for i in range(k):
            g=gen[i]; t=resize_to(gt[i],hw); ps.append(psnr(t,g)); ss.append(ssim(t,g)); lp.append(lpips_dist(t,g,args.lpips_net))
        row.update({"n":k,"stride":st,"psnr":float(np.mean(ps)),"ssim":float(np.nanmean(ss)),"lpips":float(np.nanmean(lp))})
        rows.append(row); print(f"  {base}: psnr={row['psnr']:.2f} ssim={row['ssim']:.3f} lpips={row['lpips']:.3f} (n={k})")
    keys=sorted({k for r in rows for k in r})
    with open(args.out_csv,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); [w.writerow(r) for r in rows]
    print(f"[done] {len(rows)} rows -> {args.out_csv}")
    good=[r for r in rows if "psnr" in r]
    if good: print(f"[agg] psnr={np.mean([r['psnr'] for r in good]):.2f} ssim={np.nanmean([r['ssim'] for r in good]):.3f} lpips={np.nanmean([r['lpips'] for r in good]):.3f}")

if __name__=="__main__": main()
