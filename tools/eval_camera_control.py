#!/usr/bin/env python3
"""Camera controllability eval (offline).
Estimate camera trajectory from a GENERATED video (MASt3R-SfM and/or COLMAP), align to
the TARGET (GT) trajectory, report per-sample: RotErr(deg) | TransErr | CamMC | ADE/FDE.
Monocular -> Sim3-align centers (scale+world+trans) for TransErr/ADE, solve global left&right
rotations (Wl R_gt Wr, absorbs world frame + camera-axis convention, robust to planar) for RotErr.
--selftest validates the metric core (no deps; exact case -> 0)."""
import os, sys, argparse, csv, glob, tempfile, shutil, json, re
import numpy as np

def umeyama_sim3(src, dst, with_scale=True):
    src = np.asarray(src, float); dst = np.asarray(dst, float); n = src.shape[0]
    mu_s, mu_d = src.mean(0), dst.mean(0); Sc, Dc = src - mu_s, dst - mu_d
    Sigma = (Dc.T @ Sc) / n; U, D, Vt = np.linalg.svd(Sigma); S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0: S[2, 2] = -1
    R = U @ S @ Vt
    s = (D * np.diag(S)).sum() / ((Sc**2).sum()/n + 1e-12) if with_scale else 1.0
    return float(s), R, mu_d - s * R @ mu_s

def _proj_so3(M):
    U, _, Vt = np.linalg.svd(M); D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0: D[2, 2] = -1
    return U @ D @ Vt

def geodesic_deg(R1, R2):
    c = (np.trace(R1.T @ R2) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))

def poses_to_RC(poses):
    P = np.asarray(poses, float); return P[:, :3, :3], P[:, :3, 3]

def align_rotations(R_est, R_gt, iters=1000, tol=1e-9):
    N = len(R_est); Wl = np.eye(3); Wr = np.eye(3)
    for _ in range(iters):
        Wl_n = _proj_so3(sum(R_est[i] @ (R_gt[i] @ Wr).T for i in range(N)))
        Wr_n = _proj_so3(sum((Wl_n @ R_gt[i]).T @ R_est[i] for i in range(N)))
        if np.abs(Wl_n-Wl).max() < tol and np.abs(Wr_n-Wr).max() < tol:
            Wl, Wr = Wl_n, Wr_n; break
        Wl, Wr = Wl_n, Wr_n
    return Wl, Wr

def camera_metrics(est_poses, gt_poses):
    R_e, C_e = poses_to_RC(est_poses); R_g, C_g = poses_to_RC(gt_poses); N = len(R_g)
    s, R_al, t_al = umeyama_sim3(C_e, C_g, True)
    C_e_al = (s * (R_al @ C_e.T)).T + t_al
    trans = np.linalg.norm(C_e_al - C_g, axis=1); d_xy = np.linalg.norm(C_e_al[:,:2]-C_g[:,:2], axis=1)
    Wl, Wr = align_rotations(R_e, R_g); R_g_al = np.stack([Wl@R_g[i]@Wr for i in range(N)])
    rot = np.array([geodesic_deg(R_g_al[i], R_e[i]) for i in range(N)])
    dmag = np.array([abs(geodesic_deg(R_e[i-1].T@R_e[i], np.eye(3)) - geodesic_deg(R_g[i-1].T@R_g[i], np.eye(3))) for i in range(1,N)]) if N>1 else np.array([0.0])
    cammc = np.array([np.linalg.norm(np.hstack([R_e[i], C_e_al[i:i+1].T]) - np.hstack([R_g_al[i], C_g[i:i+1].T])) for i in range(N)])
    # RPE (relative pose error, frame-to-frame drift) using the SAME alignments so exact case -> 0.
    # Comparable-in-spirit to WorldCam/GameCraft RPEtrans/RPErot(deg)/RPEcamera (relative, not ATE).
    if N > 1:
        dRe  = [R_e[i].T @ R_e[i+1] for i in range(N-1)]            # est relative rotation
        dRga = [R_g_al[i].T @ R_g_al[i+1] for i in range(N-1)]      # aligned-GT relative rotation
        rpe_rot = np.array([geodesic_deg(dRe[i], dRga[i]) for i in range(N-1)])         # deg
        dCe = np.diff(C_e_al, axis=0); dCg = np.diff(C_g, axis=0)   # world-frame step (sim3-scaled)
        rpe_tr = np.linalg.norm(dCe - dCg, axis=1)                  # meters
        rpe_cam = np.array([np.linalg.norm(np.hstack([dRe[i] - dRga[i], (dCe[i] - dCg[i])[:, None]]))
                            for i in range(N-1)])                   # 3x4 Frobenius
    else:
        rpe_rot = rpe_tr = rpe_cam = np.array([0.0])
    return {"roterr": float(rot.mean()), "roterr_last": float(rot[-1]), "rotstep_err": float(dmag.mean()),
            "transerr": float(trans.mean()), "transerr_last": float(trans[-1]), "cammc": float(cammc.mean()),
            "ade": float(d_xy.mean()), "fde": float(d_xy[-1]), "scale": float(s), "n": int(N),
            "rpe_rot_deg": float(rpe_rot.mean()), "rpe_trans": float(rpe_tr.mean()),
            "rpe_camera": float(rpe_cam.mean())}

def carla_pose_to_mat(x, y, z, roll, pitch, yaw):
    import math; r,p,yw = map(math.radians,(roll,pitch,yaw))
    cr,sr=math.cos(r),math.sin(r); cp,sp=math.cos(p),math.sin(p); cy,sy=math.cos(yw),math.sin(yw)
    M=np.eye(4)
    M[0,0]=cy*cp; M[0,1]=cy*sp*sr-sy*cr; M[0,2]=cy*sp*cr+sy*sr
    M[1,0]=sy*cp; M[1,1]=sy*sp*sr+cy*cr; M[1,2]=sy*sp*cr-cy*sr
    M[2,0]=-sp;   M[2,1]=cp*sr;          M[2,2]=cp*cr
    M[0,3]=x; M[1,3]=y; M[2,3]=z; return M

def load_target_poses(poses_npy, frame_indices):
    p = np.load(poses_npy); return np.stack([carla_pose_to_mat(*p[i]) for i in frame_indices])

def extract_frames(v):
    if os.path.isdir(v):
        from PIL import Image
        fs = sorted(glob.glob(os.path.join(v,"*.png"))+glob.glob(os.path.join(v,"*.jpg")))
        return [np.array(Image.open(f).convert("RGB")) for f in fs]
    import imageio.v3 as iio; vid = iio.imread(v); return [vid[i] for i in range(len(vid))]

def _colmap_cfw_matrix(img):
    """Return 3x4 world->cam [R|t] for a pycolmap Image, across API versions."""
    cfw = getattr(img, "cam_from_world", None)
    if cfw is not None:
        obj = cfw() if callable(cfw) else cfw           # method (older) vs property (newer)
        mm = getattr(obj, "matrix", None)
        if mm is not None:
            M = np.asarray(mm() if callable(mm) else mm)
            if M.shape == (3, 4): return M
            if M.shape == (4, 4): return M[:3, :4]
        R = getattr(obj, "rotation", None); t = getattr(obj, "translation", None)
        if R is not None and t is not None:
            rmm = getattr(R, "matrix", None)
            Rm = np.asarray(rmm() if callable(rmm) else (rmm if rmm is not None else R))
            out = np.zeros((3, 4)); out[:3, :3] = Rm; out[:, 3] = np.asarray(t).ravel(); return out
    if hasattr(img, "projection_matrix"):
        return np.asarray(img.projection_matrix())[:3, :4]
    if hasattr(img, "qvec") and hasattr(img, "tvec"):
        w, x, y, z = np.asarray(img.qvec, float)
        Rm = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
                       [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
                       [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)]])
        out = np.zeros((3, 4)); out[:3, :3] = Rm; out[:, 3] = np.asarray(img.tvec, float).ravel(); return out
    raise RuntimeError("unknown pycolmap Image pose API")

def estimate_colmap(frames, intrinsics=None):
    try: import pycolmap
    except Exception as e: return None, 0, f"pycolmap import failed: {e}"
    tmp = tempfile.mkdtemp(prefix="cc_colmap_")
    try:
        from PIL import Image
        img_dir = os.path.join(tmp,"images"); os.makedirs(img_dir)
        for i,f in enumerate(frames): Image.fromarray(f).save(os.path.join(img_dir,f"{i:04d}.png"))
        db = os.path.join(tmp,"db.db")
        try:
            if intrinsics is not None:
                fx,fy,cx,cy=[float(v) for v in intrinsics[:4]]
                ro=pycolmap.ImageReaderOptions(); ro.camera_model="SIMPLE_PINHOLE"; ro.camera_params=f"{fx},{cx},{cy}"
                pycolmap.extract_features(db, img_dir, camera_mode=pycolmap.CameraMode.SINGLE, reader_options=ro)
            else:
                pycolmap.extract_features(db, img_dir)
        except Exception:
            pycolmap.extract_features(db, img_dir)
        try: pycolmap.match_sequential(db)
        except Exception: pycolmap.match_exhaustive(db)
        try:
            opts=pycolmap.IncrementalPipelineOptions(); opts.min_num_matches=5
            opts.mapper.init_min_num_inliers=15; opts.mapper.abs_pose_min_num_inliers=12; opts.mapper.filter_max_reproj_error=4.0
            if intrinsics is not None:
                opts.ba_refine_focal_length=False; opts.ba_refine_principal_point=False; opts.ba_refine_extra_params=False
            maps=pycolmap.incremental_mapping(db, img_dir, os.path.join(tmp,"sparse"), options=opts)
        except Exception:
            maps=pycolmap.incremental_mapping(db, img_dir, os.path.join(tmp,"sparse"))
        if not maps: return None, 0, "colmap: no reconstruction"
        rec = maps[0]; poses = {}
        for img in rec.images.values():
            idx = int(os.path.splitext(img.name)[0]); T = np.eye(4)
            T[:3,:4] = _colmap_cfw_matrix(img); poses[idx] = np.linalg.inv(T)
        return np.stack([poses.get(i, np.eye(4)) for i in range(len(frames))]), len(poses), None
    except Exception as e: return None, 0, f"colmap error: {e}"
    finally: shutil.rmtree(tmp, ignore_errors=True)

_MAST3R_CACHE = {}
def estimate_mast3r(frames, intrinsics=None):
    # NOTE: verify against installed mast3r API/version on the box.
    try:
        import torch
        from mast3r.model import AsymmetricMASt3R
        from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
        from dust3r.image_pairs import make_pairs
        from dust3r.utils.image import load_images
    except Exception as e: return None, 0, f"mast3r import failed: {e}"
    try:
        from PIL import Image
        tmp = tempfile.mkdtemp(prefix="cc_mast3r_"); paths=[]
        for i,f in enumerate(frames): pth=os.path.join(tmp,f"{i:04d}.png"); Image.fromarray(f).save(pth); paths.append(pth)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _mid = os.environ.get("MAST3R_MODEL","naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric")
        if _MAST3R_CACHE.get("id") != _mid:
            _MAST3R_CACHE["m"] = AsymmetricMASt3R.from_pretrained(_mid).to(dev).eval()
            _MAST3R_CACHE["id"] = _mid
        model = _MAST3R_CACHE["m"]
        imgs = load_images(paths, size=512)
        pairs = make_pairs(imgs, scene_graph="swin", prefilter=None, symmetrize=True)
        scene = sparse_global_alignment(paths, pairs, os.path.join(tmp,"cache"), model, device=dev)
        c2w = scene.get_im_poses().detach().cpu().numpy(); shutil.rmtree(tmp, ignore_errors=True)
        return c2w, len(c2w), None
    except Exception as e: return None, 0, f"mast3r error: {e}"

_DA3_CACHE = {}
def estimate_da3(frames, intrinsics=None):
    # Depth Anything 3: any-view feed-forward pose+depth. We use camera pose only.
    # pred.extrinsics is (N,3,4) opencv w2c -> invert to c2w (eval poses_to_RC expects c2w).
    try:
        import torch
        from depth_anything_3.api import DepthAnything3
    except Exception as e: return None, 0, f"da3 import failed: {e}"
    try:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _mid = os.environ.get("DA3_MODEL", "depth-anything/DA3-LARGE-1.1")
        if _DA3_CACHE.get("id") != _mid:
            _DA3_CACHE["m"] = DepthAnything3.from_pretrained(_mid).to(dev).eval()
            _DA3_CACHE["id"] = _mid
        model = _DA3_CACHE["m"]
        with torch.no_grad():
            # See viz_da3_pointcloud.py:_da3_infer -- "first" anchors DA3 on frame 0 instead of
            # a mid-sequence view. Every CSV written before this change used "saddle_balanced".
            _rv = os.environ.get("DA3_REF_VIEW", "first")
            pred = model.inference([np.asarray(f) for f in frames], ref_view_strategy=_rv)
        ext = pred.extrinsics
        ext = ext.detach().cpu().numpy() if hasattr(ext, "detach") else np.asarray(ext)
        ext = ext.astype(np.float64)                              # (N,3,4) w2c
        c2w = np.zeros((len(ext), 4, 4))
        for i in range(len(ext)):
            w2c = np.eye(4); w2c[:3, :4] = ext[i]; c2w[i] = np.linalg.inv(w2c)  # w2c -> c2w
        return c2w, len(c2w), None
    except Exception as e: return None, 0, f"da3 error: {e}"

ESTIMATORS = {"colmap": estimate_colmap, "mast3r": estimate_mast3r, "da3": estimate_da3}

def selftest():
    rng = np.random.default_rng(0); N=20; gt=[]
    for i in range(N):
        yaw=2.0*i; gt.append(carla_pose_to_mat(0.1*i*np.cos(np.radians(yaw)),0.1*i*np.sin(np.radians(yaw)),0,0,0,yaw))
    gt=np.stack(gt)
    Wr=carla_pose_to_mat(0,0,0,10,5,30)[:3,:3]; C=carla_pose_to_mat(0,0,0,20,-15,90)[:3,:3]; s=2.0; wt=np.array([3.,-1.,2.])
    est=[]
    for i in range(N):
        Rg=gt[i,:3,:3]; Cg=gt[i,:3,3]
        Re=_proj_so3(Wr@Rg@C+0.001*rng.standard_normal((3,3))); Ce=s*(Wr@Cg)+wt+0.002*rng.standard_normal(3)
        T=np.eye(4); T[:3,:3]=Re; T[:3,3]=Ce; est.append(T)
    m=camera_metrics(np.stack(est),gt); print("[selftest] noisy:",{k:round(v,4) for k,v in m.items()})
    assert m["roterr"]<5.0 and m["transerr"]<0.05
    est0=[]
    for i in range(N):
        Rg=gt[i,:3,:3]; Cg=gt[i,:3,3]; T=np.eye(4); T[:3,:3]=_proj_so3(Wr@Rg@C); T[:3,3]=s*(Wr@Cg)+wt; est0.append(T)
    m0=camera_metrics(np.stack(est0),gt); print("[selftest] exact:",{k:round(v,6) for k,v in m0.items()})
    assert m0["roterr"]<1e-3 and m0["transerr"]<1e-3
    assert m0["rpe_rot_deg"]<1e-3 and m0["rpe_trans"]<1e-3
    print("[selftest] PASS")

def _write_csv(rows, path):
    """Rewrite metrics CSV from current rows (union of keys). Called each iteration so
    partial/interrupted runs stay usable and progress is live-monitorable."""
    import csv as _csv
    keys=sorted({k for r in rows for k in r})
    with open(path,"w",newline="") as f:
        w=_csv.DictWriter(f,fieldnames=keys); w.writeheader(); [w.writerow(r) for r in rows]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--selftest",action="store_true")
    ap.add_argument("--results_dir"); ap.add_argument("--estimators",default="mast3r,colmap")
    ap.add_argument("--out_csv",default="camera_control_metrics.csv")
    ap.add_argument("--video_glob",default="sample_*.mp4"); ap.add_argument("--poses_name",default="poses.npy")
    ap.add_argument("--meta",default="samples_meta.json")
    ap.add_argument("--sample_re",default=r"sample_\d+(_rev)?\.mp4$",
                    help="Filename filter applied after --video_glob. The glob alone is not enough: a results dir also holds _pred/_raymap/_nomap/_condgt/_vis variants of the same sample. For the worldmap map-ablation pass use r'sample_\\d+(_rev)?_nomap\\.mp4$'.")
    args=ap.parse_args()
    if args.selftest: selftest(); return
    ests=[e for e in args.estimators.split(",") if e in ESTIMATORS]; rows=[]
    meta_path=os.path.join(args.results_dir,args.meta)
    meta=json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    if not meta: print(f"[warn] {meta_path} not found -> fall back to {args.poses_name} in each video's own dir")
    vids=sorted(glob.glob(os.path.join(args.results_dir,"**",args.video_glob),recursive=True))
    vids=[v for v in vids if re.match(args.sample_re, os.path.basename(v))]  # default: base sample_XX.mp4 only (skip _pred/_condgt/_raymap/_nomap/_vis)
    print(f"[info] {len(vids)} videos, estimators={ests}")
    for vp in vids:
        sdir=os.path.dirname(vp); base=os.path.splitext(os.path.basename(vp))[0]; row={"sample":base,"dir":sdir}
        # GT poses live in the source CARLA seq_dir (via samples_meta.json), indexed by absolute frame id.
        m_id=re.search(r"(\d+)",base)
        info=meta.get(str(int(m_id.group(1)))) if (m_id and meta) else None
        if info is not None:
            seq_dir=info["seq_dir"]; start=int(info.get("start_frame",0)); num=int(info.get("num_frames",0))
            # see eval_video_quality.py: stride-aware GT indexing + generated-video subsampling
            st=int(info.get("frame_stride",1)); gs=int(info.get("gen_stride",1)); go=int(info.get("gen_offset",0))
        else:
            seq_dir=sdir; start=0; num=0; st=1; gs=1; go=0
        pth=os.path.join(seq_dir,args.poses_name)
        if not os.path.exists(pth): row["error"]=f"no {args.poses_name} ({pth})"; rows.append(row); continue
        frames=extract_frames(vp)[go::gs]
        nfr=len(frames)
        if num: nfr=min(nfr,num)
        frames=frames[:nfr]
        fidx=[start+i*st for i in range(nfr)]
        try: gt_poses=load_target_poses(pth,fidx)
        except Exception as e: row["error"]=f"target: {e}"; rows.append(row); continue
        _ip=os.path.join(seq_dir,"intrinsics.npy"); intr=np.load(_ip) if os.path.exists(_ip) else None
        for name in ests:
            poses,n,err=ESTIMATORS[name](frames, intr)
            if poses is None: row[f"{name}_success"]=False; row[f"{name}_err"]=err; continue
            k=min(len(poses),len(gt_poses)); m=camera_metrics(poses[:k],gt_poses[:k])
            row[f"{name}_success"]=True; row[f"{name}_n"]=n
            for mk,mv in m.items(): row[f"{name}_{mk}"]=mv
        rows.append(row); print("  "+base+": "+", ".join(f"{k}={v}" for k,v in row.items() if "roterr" in k))
        _write_csv(rows, args.out_csv)   # incremental: 매 샘플 저장(중단돼도 부분결과 보존)
    _write_csv(rows, args.out_csv)
    print(f"[done] {len(rows)} rows -> {args.out_csv}")

if __name__=="__main__": main()
