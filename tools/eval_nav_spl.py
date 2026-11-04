#!/usr/bin/env python3
"""Navigation SR / SPL (Habitat) for CARLA generated trajectories.
SR = DTG < success_distance (no discrete stop -> horizon-relaxed FINAL pos; min-over-path=OSR separate).
SPL = success * ell / max(ell, p);  p = agent path len (metres).
  window mode: goal = poses[goal_frame][:2] (this window's near goal), ell = GT path start..goal_frame.
  route  mode: goal = goal_loc.npy,          ell = planned_path.npy (needs fixed-goal rollout).
ell and p are computed at the SAME temporal resolution (--action_stride) so SPL is unbiased.
Source (A): pred_actions.npy (inference --pred_action) integrated from GT start pose -> world xy.
"""
import os, sys, json, glob, argparse, csv, math, re
import numpy as np

def integrate_ego_actions(a, start_yaw_deg):
    pos=np.zeros(2); yaw=float(start_yaw_deg); traj=[pos.copy()]
    for dx_fwd,dy_lat,d_yaw in np.asarray(a,float):
        yr=math.radians(yaw)
        pos=pos+np.array([dx_fwd*math.cos(yr)-dy_lat*math.sin(yr), dx_fwd*math.sin(yr)+dy_lat*math.cos(yr)])
        yaw+=d_yaw; traj.append(pos.copy())
    return np.asarray(traj)

def polyline_len(xy):
    xy=np.asarray(xy,float)
    return 0.0 if len(xy)<2 else float(np.linalg.norm(np.diff(xy,axis=0),axis=1).sum())

def sub_idx(start, end_incl, stride):
    idx=list(range(start, end_incl+1, max(1,stride)))
    if not idx or idx[-1]!=end_incl: idx.append(end_incl)
    return idx

def load_meta(d):
    for name in ("samples_meta.json","rollout_meta.json"):
        p=os.path.join(d,name)
        if os.path.exists(p): return json.load(open(p))
    return None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--results_dir",required=True)
    ap.add_argument("--goal_mode",choices=["window","route"],default="window")
    ap.add_argument("--success_distance",type=float,default=1.0)
    ap.add_argument("--testset_json",default=None)
    ap.add_argument("--goal_name",default="goal_loc.npy")
    ap.add_argument("--planned_name",default="planned_path.npy")
    ap.add_argument("--poses_name",default="poses.npy")
    ap.add_argument("--out_csv",default=None)
    ap.add_argument("--gt_sanity",action="store_true")
    ap.add_argument("--action_stride",type=int,default=4,help="VAE temporal factor; subsample GT to action res so ell,p match")
    ap.add_argument("--seed",type=int,default=0)
    ap.add_argument("--goal_offset_min",type=int,default=int(os.environ.get("GOAL_OFFSET_MIN","1")))
    ap.add_argument("--goal_offset_max",type=int,default=int(os.environ.get("GOAL_OFFSET_MAX","5")))
    args=ap.parse_args()
    st=max(1,args.action_stride)
    meta=load_meta(args.results_dir)
    if meta is None: print(f"[err] no meta in {args.results_dir}"); sys.exit(2)
    diff_of={}
    if args.testset_json and os.path.exists(args.testset_json):
        for w in json.load(open(args.testset_json)).get("windows",[]):
            diff_of[os.path.abspath(w["seq_dir"])]=w.get("difficulty")
    pa_files=sorted(glob.glob(os.path.join(args.results_dir,"**","*_pred_actions.npy"),recursive=True))
    if not pa_files: print(f"[err] no *_pred_actions.npy under {args.results_dir}. inference needs --pred_action."); sys.exit(2)
    import random as _rnd
    rows=[]
    for pa in pa_files:
        base=os.path.basename(pa).replace("_pred_actions.npy","")
        m=re.search(r"(\d+)",base); idx_i=int(m.group(1)) if m else None
        info=meta.get(str(idx_i)) if idx_i is not None else None
        if info is None: rows.append({"name":base,"error":"no meta"}); continue
        seq_dir=info["seq_dir"]; start=int(info.get("start_frame",0)); num=int(info.get("num_frames",0))
        try: poses=np.load(os.path.join(seq_dir,args.poses_name))
        except Exception as e: rows.append({"name":base,"error":f"poses:{e}"}); continue
        if start>=len(poses): rows.append({"name":base,"error":"start>=len"}); continue
        sp=poses[start]; start_xy=sp[:2].astype(float); start_yaw=float(sp[5])
        if args.goal_mode=="route":
            try:
                goal=np.load(os.path.join(seq_dir,args.goal_name))[:2].astype(float)
                ell=polyline_len(np.atleast_2d(np.load(os.path.join(seq_dir,args.planned_name)))[:,:2])
            except Exception as e: rows.append({"name":base,"error":f"route gt:{e}"}); continue
        else:
            gf=info.get("goal_frame")
            if gf is None:
                lo=start+num+args.goal_offset_min; hi=min(start+num+args.goal_offset_max,len(poses)-1)
                gf=_rnd.Random(args.seed+idx_i).randint(lo,max(lo,hi))
            gf=int(min(gf,len(poses)-1)); goal=poses[gf,:2].astype(float)
            ell=polyline_len(poses[sub_idx(start,gf,st),:2])          # GT path start->near goal @ action res (endpoint incl)
        def metrics(aw):
            p=polyline_len(aw); d=np.linalg.norm(aw-goal[None,:],axis=1)
            dtg=float(d[-1]); dmin=float(d.min())
            succ=1.0 if dtg<args.success_distance else 0.0
            osr=1.0 if dmin<args.success_distance else 0.0
            spl=succ*(ell/max(ell,p)) if ell>1e-6 else 0.0
            return dict(ell=ell,p=p,dtg=dtg,dtg_min=dmin,success=succ,osr=osr,spl=spl)
        aw=integrate_ego_actions(np.load(pa),start_yaw)+start_xy[None,:]
        r={"name":base,"seq_dir":os.path.abspath(seq_dir),"start":start,"difficulty":diff_of.get(os.path.abspath(seq_dir),"")}
        r.update(metrics(aw))
        if args.gt_sanity:
            end=min(start+max(num,1),len(poses))
            r.update({f"gt_{k}":v for k,v in metrics(poses[sub_idx(start,end-1,st),:2].astype(float)).items()})
        rows.append(r)
    ok=[r for r in rows if "success" in r]
    if not ok: print("[err] no scorable"); [print(" ",r.get("name"),r.get("error")) for r in rows[:5]]; sys.exit(2)
    def agg(sub,pre=""):
        return (len(sub),np.mean([r[pre+"success"] for r in sub]),np.mean([r[pre+"spl"] for r in sub]),np.mean([r[pre+"osr"] for r in sub]))
    print(f"\n==== nav SR/SPL ({args.results_dir}) goal_mode={args.goal_mode} stride={st} ====")
    print(f"  succ_dist={args.success_distance}m | ell={'planned_path' if args.goal_mode=='route' else 'GT->near goal'} | SR=final,OSR=min-over-path")
    n,sr,spl,osr=agg(ok); print(f"  ALL      n={n:4d}  SR={sr:.3f}  SPL={spl:.3f}  OSR={osr:.3f}")
    if any(r.get("difficulty") for r in ok):
        for lvl in ("easy","medium","hard"):
            sub=[r for r in ok if r.get("difficulty")==lvl]
            if sub: n,sr,spl,osr=agg(sub); print(f"  {lvl:8s} n={n:4d}  SR={sr:.3f}  SPL={spl:.3f}  OSR={osr:.3f}")
    if args.gt_sanity and all("gt_success" in r for r in ok):
        n,sr,spl,osr=agg(ok,"gt_"); print(f"  GT-sanity n={n:4d}  SR={sr:.3f}  SPL={spl:.3f}  OSR={osr:.3f}  (should be high)")
    out=args.out_csv or os.path.join(args.results_dir,"nav_spl_metrics.csv")
    keys=sorted({k for r in rows for k in r})
    with open(out,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); [w.writerow(r) for r in rows]
    print(f"  csv -> {out}")

if __name__=="__main__": main()
