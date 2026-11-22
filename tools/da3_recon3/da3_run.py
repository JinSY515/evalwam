"""Stage 1 (/venv/da3): DA3-LARGE-1.1 depth + camera on frames t=0,4,..,304 of GT / ours / Aether, all on the SAME view
(GT 512 frame rows 85..427 = Aether's 720x480; GT and ours cropped to it), ref_view_strategy="first".
  -> <out>/da3_{gt,ours,aether}.npz  depth (N,h,w), conf, c2w (N,4,4 opencv), K (N,3,3)"""
import sys, numpy as np, torch, imageio.v2 as iio
from PIL import Image
OUT = sys.argv[1]
SD = "/root/data/eval_bench_v2/lookout_turn_rollout_c4_cv_lh/aria/nav/Traderjoes_jan11_s00296"
OURS = "/root/project/egowam/results_syn/goalprog_mode_real6_nomr_hg0_grad0.5/ckpta7500_8750/lookout_rc4_viz/rollout_43.mp4"
AE = "/root/data/aether_rc4_Traderjoes_s00296/aether_video.mp4"
TS = list(range(0, 305, 4)); band = lambda a: np.ascontiguousarray(a[85:427, :, :3])
def vid(p):
    r = iio.get_reader(p); f = [x[..., :3] for x in r]; r.close(); return f
o, ae = vid(OURS), vid(AE)
assert len(o) == 305 and len(ae) == 305
src = {"gt": [band(np.asarray(Image.open(f"{SD}/images/{4 + t:06d}.png").convert("RGB"))) for t in TS],
       "ours": [band(o[t]) for t in TS], "aether": [ae[t] for t in TS]}
from depth_anything_3.api import DepthAnything3
m = DepthAnything3.from_pretrained("depth-anything/DA3-LARGE-1.1").to("cuda").eval()
for k, ims in src.items():
    with torch.no_grad():
        p = m.inference(ims, ref_view_strategy="first")
    ext = np.asarray(p.extrinsics, np.float64); c2w = np.tile(np.eye(4), (len(ext), 1, 1))
    for i, e in enumerate(ext):
        w = np.eye(4); w[:3, :4] = e; c2w[i] = np.linalg.inv(w)
    np.savez(f"{OUT}/da3_{k}.npz", depth=np.asarray(p.depth, np.float32), conf=np.asarray(p.conf, np.float32), c2w=c2w,
             K=np.asarray(p.intrinsics, np.float64), ts=np.array(TS))
    print(k, np.asarray(p.depth).shape, flush=True)
