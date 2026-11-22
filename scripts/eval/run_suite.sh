#!/bin/bash
# Extended metric suite for ONE valbench5s run (120 windows).
# $1 = flat 120-sample dir (created by merging $2 if missing)
# $2 = per-difficulty parent dir (contains easy/ medium/ hard/) or "-" if $1 already exists
# env: CAM_FIX=opencv|none (raymap convention the run was TRAINED with; none = old)
#      HAS_RAYMAP=1|0  HAS_DEPTH=1|0  GPU (default 2)
set -u
D=$1; SRC=${2:--}; GPU=${GPU:-2}
EV=/root/project/evalwam
MERGE=$EV/tools/merge120.py
export HF_HOME=/workspace/.hf_home
log() { echo "[suite $(date +%H:%M)] $*"; }
if [ ! -f "$D/samples_meta.json" ]; then
  log "merge $SRC -> $D"; python3 -I $MERGE "$SRC" "$D" || exit 1
fi
N=$(python3 -c "import json;print(len(json.load(open('$D/samples_meta.json'))))")
[ "$N" = 120 ] || { log "only $N samples in $D"; exit 1; }
cd $EV
# 1. VQ + DA3 on generated video (+ depth compare) + DA3 camera  (skips whatever is already complete)
PY=/venv/da3/bin/python GPU=$GPU D=$D CHAN=0 DEPTH=$([ "${HAS_DEPTH:-1}" = 1 ] && echo auto || echo 0) \
  bash $EV/scripts/eval/eval_4axis.sh > $D/suite_4axis.log 2>&1 || log "4axis returned nonzero"
# with CHAN=0 and DEPTH=0, eval_4axis skips DA3 on the generated video -> da3 poses missing.
# coherence/goal need them, so run that pass directly (poses only) when absent.
if [ "$(ls $D/viz_da3_fix/*_da3_poses.npy 2>/dev/null | wc -l)" != 120 ]; then
  log "DA3 on generated video (poses)"
  CUDA_VISIBLE_DEVICES=$GPU /venv/da3/bin/python tools/viz_da3_pointcloud.py --results_dir $D --source da3 \
    --out_subdir viz_da3_fix > $D/suite_da3.log 2>&1 || log "da3 pass failed"
fi
# 2. raymap -> poses
if [ "${HAS_RAYMAP:-1}" = 1 ] && [ "$(ls $D/sample_*_pred_poses.npy 2>/dev/null | wc -l)" != 120 ]; then
  if [ "${CAM_FIX:-opencv}" = opencv ]; then export RAYMAP_CAM_FIX=opencv; else unset RAYMAP_CAM_FIX; fi
  log "invert_raymap s3 (CAM_FIX=${RAYMAP_CAM_FIX:-off})"
  CUDA_VISIBLE_DEVICES=$GPU /venv/main/bin/python tools/invert_raymap.py --results_dir $D --stage s3 \
    --ray_downsample 2 --raymap_moment_scale 8.0 > $D/invert_raymap_s3.log 2>&1 || log "invert failed"
  unset RAYMAP_CAM_FIX
fi
# 3. depth compare again so depth_vs_da3_metrics.csv carries delta15 (cheap: reads saved npy)
if [ "${HAS_DEPTH:-1}" = 1 ]; then
  /venv/da3/bin/python tools/eval_depth_vs_da3.py --results_dir $D --da3_subdir viz_da3_fix \
    --out_csv depth_vs_da3_metrics.csv > $D/suite_depth.log 2>&1 || log "depth compare failed"
fi
# 4. pose/depth suite + set-level video metrics
/venv/da3/bin/python tools/eval_suite.py --results_dir $D > $D/suite.log 2>&1 || log "eval_suite failed"
CUDA_VISIBLE_DEVICES=$GPU /venv/metrics/bin/python tools/eval_fvd_musiq.py --results_dir $D > $D/suite_fvd.log 2>&1 || log "fvd/musiq failed"
CUDA_VISIBLE_DEVICES=$GPU HF_HOME=/workspace/.hf_home /venv/metrics/bin/python tools/eval_goal_image.py --results_dir $D > $D/suite_goalimg.log 2>&1 || log "goal image failed"
if [ -x /venv/met3r/bin/python ] && [ ! -f $D/met3r.json ]; then
  CUDA_VISIBLE_DEVICES=$GPU HF_HOME=/workspace/.hf_home /venv/met3r/bin/python tools/eval_met3r.py --results_dir $D > $D/suite_met3r.log 2>&1 || log "met3r failed"
fi
log "done $D"
