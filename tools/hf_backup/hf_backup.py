#!/usr/bin/env python3
"""Back up the main / ablation / supplementary / rebuttal results + main and direct checkpoints to private HF repos.

Results -> dataset clwm515/egowam-results as one tar per experiment folder. Members keep their absolute server path
(minus the leading /) so the many absolute symlinks inside the results still resolve after `tar -xf X.tar -C /`.
Checkpoints -> model clwm515/egowam-main as plain files.
Resumable: a shard already in the repo is skipped. Each tar is built, uploaded, then deleted (one at a time on disk).
Run:  /venv/main/bin/python hf_backup.py   (log goes to stdout; launched in tmux hf_backup)
"""
import json, os, subprocess, sys, time
from huggingface_hub import HfApi

MODEL, DATA = "clwm515/egowam-main", "clwm515/egowam-results"
STAGE = "/root/hf_stage"
R = "/root/project/egowam/results_syn"
H = f"{R}/goalprog_mode_real6_nomr_hg0_grad0.5"
AB = f"{R}/abl"
CK = "/root/ckpt/wam"
LO = "/root/data/eval_bench/lookout_val_cv"
LOOKOUT_DATA = sorted(os.path.join(r, d) for r, ds, _ in os.walk(LO) for d in ds if d.startswith("depth_da3")) \
    + ["/root/project/carla/results/lookout_val_cv_lh"]

# (path in repo, [source dirs], what) ; order = small / important first
SHARDS = [
    ("rebuttal/coherence_validation.tar", [f"{R}/coherence_validation"], "/rebuttal: video-action coherence metric validation"),
    ("rebuttal/goal_validation.tar", [f"{R}/goal_validation"], "goal augmentation validation"),
    ("rebuttal/traversal_difficulty.tar", [f"{R}/traversal_difficulty"], "traversal difficulty"),
    ("rebuttal/gtcam_cmp.tar", [f"{R}/gtcam_cmp"], "GT camera comparison"),
    ("rebuttal/attn_wam_report.tar", [f"{R}/attn_wam/report"], "/attn: attention study report"),
    ("rebuttal/teaser.tar", [f"{R}/teaser"], "teaser comparisons"),
    ("rebuttal/vcons_direct_vs_residual.tar", [f"{R}/goalprog_mode_real6_nomr_hg0_grad0.5_direct"], "/vcons/: direct vs residual analysis"),
    ("rebuttal/vcons_resid_viz.tar", [f"{H}/ckpta4000_7500/vcons_resid_viz"], "vcons residual visualisation"),
    ("rebuttal/viewer_v2.tar", [f"{H}/multi/v2"], "viewer tags, LookOut exclusion list, merged sets"),
    ("main/ckpta7500_8750.tar", [f"{H}/ckpta7500_8750"], "MAIN avg 7500+8750: valbench 120, LookOut A/B/C, FVD, DL3DV, rc4"),
    ("supp/camcond.tar", [f"{H}/ckpta7500_8750_camcond"], "mode 0: GT camera as input"),
    ("supp/goalstyle.tar", [f"{H}/ckpta7500_8750_goalstyle"], "mode 3: edited goal appearance"),
    ("supp/cherry_export_cka7500_8750.tar", [f"{H}/cherry_export_cka7500_8750"], "cherry-pick export (main)"),
    ("supp/cherry_export_cka4000_7500.tar", [f"{H}/cherry_export_cka4000_7500"], "cherry-pick export (avg 4000+7500)"),
    ("supp/cherry_export.tar", [f"{H}/cherry_export"], "cherry-pick export (older)"),
    ("supp/aether_rc4_Traderjoes_s00296.tar", ["/root/data/aether_rc4_Traderjoes_s00296"], "Aether comparison (raw chunks + export + DA3)"),
    ("supp/wandering_cmp.tar", ["/root/data/eval_bench/wandering_cmp"], "/cmp: baseline comparison"),
    ("abl/direct2000.tar", [f"{H}/ckptdirect2000"], "direct-matching L_vcons ck 2000"),
    ("abl/I_jointfm_ckpt3000.tar", [f"{AB}/I_jointfm/ckpt3000"], "abl1 action token ck 3000"),
    ("abl/II_nogoalaux_ckpt3000.tar", [f"{AB}/II_nogoalaux/ckpt3000"], "abl2 raymap token ck 3000"),
    ("abl/IV_cont5k_ckpt6000.tar", [f"{AB}/IV_cont5k/ckpt6000"], "abl4 no L_vcons ck 6000"),
    ("abl/RA_raychannel.tar", [f"{AB}/RA_raychannel"], "ray channel ck 1000/1250/1500"),
    ("abl/RB_raytoken.tar", [f"{AB}/RB_raytoken"], "ray token ck 1000/1500"),
    ("abl/III_goalaux.tar", [f"{AB}/III_goalaux"], "ladder III goal aux (valbench)"),
    ("abl/IV_depthpred.tar", [f"{AB}/IV_depthpred"], "ladder IV depth pred (valbench)"),
    ("supp/camcond_goalstyle.tar", [f"{H}/ckpta7500_8750_camcond_goalstyle"], "mode 0 + edited goal (GT camera input, edited goal appearance)"),
    ("supp/cherry_export_3f.tar", [f"{H}/cherry_export_3f"], "cherry-pick export (3-frame strips, 11-13)"),
    ("supp/viewer_cache_cherry.tar", ["/root/project/evalwam/tools/ckpt_viewer/cache/cherry"], "/cherry page cache: rankings (dyn_*.json), grids, routes, chosen route style (route_style.txt); slow to recompute with cherry_compute.py"),
    ("data/lookout_da3_lh.tar", LOOKOUT_DATA, "LookOut DA3 pseudo-GT depth (depth_da3*, not in HF eval_bench) + left-right corrected GT poses (lookout_val_cv_lh)"),
]
# files added / changed after the shard above was made: (path in repo, source dir, since) ; extract over the base tar
DELTAS = [
    ("delta/gtcam_cmp_after1341.tar", f"{R}/gtcam_cmp", "2026-11-22 13:41"),
    ("delta/vcons_direct_vs_residual_after1341.tar", f"{R}/goalprog_mode_real6_nomr_hg0_grad0.5_direct", "2026-11-22 13:41"),
    ("delta/main_ckpta7500_8750_after1342.tar", f"{H}/ckpta7500_8750", "2026-11-22 13:42"),   # RECON (x4), rc4 recon_fig / recon_cherry*, dl3dv
    ("delta/goalstyle_after1343.tar", f"{H}/ckpta7500_8750_goalstyle", "2026-11-22 13:43"),
]
CKPTS = [  # (path in repo, local file)
    *[(f"main/checkpoint-a7500_8750/{f}", f"{CK}/avg_eval/checkpoint-a7500_8750/{f}") for f in ("global_step.txt", "model.safetensors")],
    ("main/avg.py", f"{CK}/avg_eval/avg.py"),
    *[(f"avg/checkpoint-a4000_7500/{f}", f"{CK}/avg_eval/checkpoint-a4000_7500/{f}") for f in ("global_step.txt", "model.safetensors")],
    *[(f"direct/checkpoint-2000/{f}", f"{CK}/goalprog_mode_real6_nomr_hg0_grad0.5_direct/checkpoint-2000/{f}")
      for f in ("global_step.txt", "scheduler.bin", "random_states_0.pkl", "model.safetensors", "optimizer.bin")],
    *[(f"abl2/checkpoint-3000/{f}", f"{CK}/goalprog_appearance_nogoalaux/checkpoint-3000/{f}")
      for f in ("global_step.txt", "scheduler.bin", "random_states_0.pkl", "model.safetensors")],
    *[(f"abl4/checkpoint-1000/{f}", f"{CK}/goalprog_dl3dv_camfix_d10_dpt_cont5k/checkpoint-1000/{f}")
      for f in ("global_step.txt", "scheduler.bin", "random_states_0.pkl", "model.safetensors", "optimizer.bin")],
]
METRIC_EXT = (".json", ".csv", ".md", ".log", ".out", ".txt", ".xlsx", ".sh", ".py")

api = HfApi()


def log(*a):
    print(time.strftime("[%m-%d %H:%M:%S]"), *a, flush=True)


def up(local, repo_path, repo, rtype):
    if api.file_exists(repo, repo_path, repo_type=rtype):
        log("skip (in repo)", repo_path); return False
    for k in range(4):
        try:
            t = time.time(); sz = os.path.getsize(local)
            api.upload_file(path_or_fileobj=local, path_in_repo=repo_path, repo_id=repo, repo_type=rtype,
                            commit_message=f"add {repo_path}")
            dt = time.time() - t
            log(f"uploaded {repo_path} {sz / 1e9:.3f} GB in {dt / 60:.1f} min ({sz / 1e6 / max(dt, 1):.1f} MB/s)"); return True
        except Exception as e:
            log(f"upload failed ({k + 1}/4) {repo_path}: {e!r}"); time.sleep(60 * (k + 1))
    raise SystemExit(f"giving up on {repo_path}")


def tar(out, srcs, members_file=None):
    cmd = ["tar", "-cf", out, "-C", "/"]
    cmd += ["--no-recursion", "-T", members_file] if members_file else [s.lstrip("/") for s in srcs]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode not in (0, 1):   # 1 = "file changed as we read it"
        raise SystemExit(f"tar failed {out}: {r.stderr[-2000:]}")


def main():
    os.makedirs(STAGE, exist_ok=True)
    api.create_repo(MODEL, repo_type="model", private=True, exist_ok=True)
    api.create_repo(DATA, repo_type="dataset", private=True, exist_ok=True)
    manifest = []
    # 1. all metric / log / script files of results_syn (small, as one archive)
    mp = "metrics/results_syn_metrics.tar.gz"
    if not api.file_exists(DATA, mp, repo_type="dataset"):
        lst = f"{STAGE}/metrics.list"
        with open(lst, "w") as f:
            for root, dirs, files in os.walk(R):
                dirs[:] = [d for d in dirs if d not in ("__pycache__", "_fvd_cache")]
                for x in files:
                    p = os.path.join(root, x)
                    if x.endswith(METRIC_EXT) and not os.path.islink(p):
                        f.write(p.lstrip("/") + "\n")
        out = f"{STAGE}/results_syn_metrics.tar.gz"
        r = subprocess.run(["tar", "-czf", out, "-C", "/", "--no-recursion", "-T", lst], capture_output=True, text=True)
        if r.returncode not in (0, 1):
            raise SystemExit(r.stderr[-2000:])
        up(out, mp, DATA, "dataset"); os.remove(out)
    else:
        log("skip (in repo)", mp)
    manifest.append((mp, [R + " (*.json csv md log out txt xlsx sh py)"], "every metric / log / launcher file under results_syn"))
    # 2. checkpoints
    for rp, lp in CKPTS:
        up(lp, rp, MODEL, "model")
    # 3. result shards
    for rp, srcs, what in SHARDS:
        manifest.append((rp, srcs, what))
        if api.file_exists(DATA, rp, repo_type="dataset"):
            log("skip (in repo)", rp); continue
        out = f"{STAGE}/{rp.replace('/', '__')}"
        t = time.time(); tar(out, srcs); log(f"tar {rp} {os.path.getsize(out) / 1e9:.3f} GB in {(time.time() - t) / 60:.1f} min")
        up(out, rp, DATA, "dataset"); os.remove(out)
    # 3b. deltas (only files newer than `since`)
    for rp, src, since in DELTAS:
        manifest.append((rp, [src], f"files changed after {since} (extract after the base tar)"))
        if api.file_exists(DATA, rp, repo_type="dataset"):
            log("skip (in repo)", rp); continue
        lst = f"{STAGE}/delta.list"
        r = subprocess.run(["find", src, "(", "-type", "f", "-o", "-type", "l", ")", "-newermt", since], capture_output=True, text=True)
        open(lst, "w").write("".join(x.lstrip("/") + "\n" for x in r.stdout.splitlines()))
        out = f"{STAGE}/{rp.replace('/', '__')}"
        tar(out, None, lst); log(f"tar {rp} {os.path.getsize(out) / 1e9:.3f} GB ({len(r.stdout.splitlines())} files)")
        up(out, rp, DATA, "dataset"); os.remove(out)
    # 4. READMEs
    rows = "\n".join(f"| `{rp}` | {what} | {'<br>'.join(f'`{s}`' for s in srcs)} |" for rp, srcs, what in manifest)
    readme = f"""# egowam results backup (private)

Server results backed up {time.strftime('%Y-%m-%d')}. Checkpoints are in the model repo `{MODEL}`.

Each tar keeps the absolute server path of its files (without the leading `/`), because the results contain many
absolute symlinks (merged sets such as `*_all51`, `valbench5s/*`). Restore in place:

```bash
huggingface-cli download {DATA} main/ckpta7500_8750.tar --repo-type dataset --local-dir .
tar -xf main/ckpta7500_8750.tar -C /        # -> /root/project/egowam/results_syn/...
```

To restore somewhere else, extract with `-C <dir>`; symlinks then point to `/root/...` and need that prefix
(e.g. bind-mount or symlink `<dir>/root` to `/root`).

`LOCATIONS.md` = what each folder is (main / ablation / supplementary). Pages: evalwam `tools/ckpt_viewer`
(branch `wip-20261122`) reads these paths directly.

| file | content | server source |
|---|---|---|
{rows}
"""
    open(f"{STAGE}/README.md", "w").write(readme)
    api.upload_file(path_or_fileobj=f"{STAGE}/README.md", path_in_repo="README.md", repo_id=DATA, repo_type="dataset",
                    commit_message="README")
    api.upload_file(path_or_fileobj=f"{R}/LOCATIONS.md", path_in_repo="LOCATIONS.md", repo_id=DATA, repo_type="dataset",
                    commit_message="LOCATIONS")
    mreadme = f"""# egowam checkpoints (private)

- `main/checkpoint-a7500_8750/model.safetensors` = MAIN model: uniform weight average of checkpoint-7500 and
  checkpoint-8750 of run `goalprog_mode_real6_nomr_hg0_grad0.5` (`main/avg.py`). Load with the same inference flags as the eval launchers
  (egowam `launchers/goalprog_mode_real6_nomr_hg0_grad0.5_multi/`).
- `avg/checkpoint-a4000_7500/` = uniform average of checkpoint-4000 and checkpoint-7500 of the same run (earlier main
  candidate; used by `vcons_resid_viz`, `cherry_export_cka4000_7500`).
- `direct/checkpoint-2000/` = ablation: L_vcons direct matching (`--vcons_mode direct`), step 2000, with optimizer /
  scheduler / RNG state for resuming (`RESUME=.../checkpoint-2000`).
- `abl2/checkpoint-3000/` = ablation abl2 (raymap token, no goal aux): run `goalprog_appearance_nogoalaux` step 3000
  (no optimizer). Results: `abl/II_nogoalaux_ckpt3000.tar`.
- `abl4/checkpoint-1000/` = ablation abl4 (no L_vcons): run `goalprog_dl3dv_camfix_d10_dpt_cont5k` step 1000
  (= 5000 + 1000 = "ck 6000" in the tables), with optimizer. Results: `abl/IV_cont5k_ckpt6000.tar`.

Results: dataset `{DATA}`. Code: GitHub egowam / evalwam branch `wip-20261122`.
"""
    open(f"{STAGE}/MODEL_README.md", "w").write(mreadme)
    api.upload_file(path_or_fileobj=f"{STAGE}/MODEL_README.md", path_in_repo="README.md", repo_id=MODEL, repo_type="model",
                    commit_message="README")
    log("ALL DONE")


if __name__ == "__main__":
    main()
