#!/usr/bin/env python3
"""Multi-sample 3D reconstruction viewer as a static site (for every sample of a valbench results dir).

  <out>/index.html        viewer (recon_site.html) with a sample picker
  <out>/samples.json      [{i, seq, diff, goal_success, act_roterr, coh_roterr}]
  <out>/data/<i>.bin      PRED / DA3 / GT clouds of sample i (recon_web.pack format, ~9 MB at 7 cm voxels)

Serve it locally (python3 -m http.server --bind 127.0.0.1 <port> -d <out>) and open it through VSCode's port
forwarding or an SSH tunnel; 120 full-density samples (~1 GB) are too large for an Artifact.

  python3 recon_site.py --results_dir <valbench5s_all120> --out_dir <dir> --label "hg0_grad0.5 ckpt4000" --jobs 16
"""
import argparse, csv, importlib.util, json, os, shutil
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
_s = importlib.util.spec_from_file_location("rw", os.path.join(HERE, "recon_web.py")); rw = importlib.util.module_from_spec(_s)
_s.loader.exec_module(rw)


def one(args):
    D, i, out, voxel, max_depth = args
    f = f"{out}/data/{i}.bin"
    if os.path.exists(f):
        return i, "skip"
    try:
        blob = rw.pack(*rw.build(D, i, max_depth=max_depth, voxel_m=voxel))
    except Exception as e:                       # a sample with missing DA3 / depth files: leave it out, say why
        return i, f"FAIL {type(e).__name__}: {e}"
    open(f + ".tmp", "wb").write(blob); os.replace(f + ".tmp", f)
    return i, f"{len(blob) / 1e6:.1f} MB"


def rows(D, name):
    p = f"{D}/{name}"
    return {r["sample"]: r for r in csv.DictReader(open(p))} if os.path.exists(p) else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True); ap.add_argument("--out_dir", required=True)
    ap.add_argument("--label", default=""); ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--voxel", type=float, default=0.07); ap.add_argument("--max_depth", type=float, default=40.0)
    ap.add_argument("--clip", type=float, default=15.0)
    a = ap.parse_args()
    D, O = a.results_dir.rstrip("/"), a.out_dir.rstrip("/")
    os.makedirs(f"{O}/data", exist_ok=True)
    meta = json.load(open(f"{D}/samples_meta.json")); idx = sorted(int(k) for k in meta)
    suite, cam = rows(D, "suite_metrics.csv"), rows(D, "cam77.csv")
    f = lambda d, k: round(float(d[k]), 3) if d.get(k) not in (None, "") else None
    S = []
    for i in idx:
        sd = meta[str(i)]["seq_dir"].split("val_bench5s/")[-1]; s, c = suite.get(f"sample_{i:02d}", {}), cam.get(f"sample_{i:02d}", {})
        S.append({"i": i, "seq": sd, "diff": sd.split("/")[0].replace("id_", ""), "goal_success": f(s, "goal_da3_success"),
                  "act_roterr": f(c, "act_roterr"), "coh_roterr": f(c, "coh_roterr")})
    with ProcessPoolExecutor(a.jobs) as ex:
        for i, msg in ex.map(one, [(D, i, O, a.voxel, a.max_depth) for i in idx]):
            print(f"sample {i:3d}: {msg}", flush=True)
            if msg.startswith("FAIL"):
                S = [x for x in S if x["i"] != i]
    json.dump({"label": a.label, "clip": a.clip, "samples": S}, open(f"{O}/samples.json", "w"))
    shutil.copy(os.path.join(HERE, "recon_site.html"), f"{O}/index.html")
    print(f"{len(S)} samples -> {O}")


if __name__ == "__main__":
    main()
