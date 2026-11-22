#!/bin/bash
# Extended suite on a LookOut rollout run (51 seqs, 153 frames = 2 chunks; real Aria, no GT depth).
# $1 = run dir under results_syn, e.g. goalprog_dl3dv_camfix_d10_dpt_vcons/ckpt3000
# Uses <run>/lookout_turn_cv + <run>/lookout_turn_cv_h2 -> <run>/lookout_turn_cv_all51
# env GPU (default 0). Every step skips what is already complete.
set -u
RS=/root/project/egowam/results_syn; EV=/root/project/evalwam; GPU=${GPU:-0}
R=$RS/$1; D=$R/lookout_turn_cv_all51
export HF_HOME=/workspace/.hf_home
log() { echo "[lookout-suite $(date +%H:%M)] $*"; }
cd $EV
# 1. raymap -> chained 153-frame poses, per half (writes rollout_NN_pred_poses.npy)
for h in lookout_turn_cv lookout_turn_cv_h2; do
  n=$(ls $R/$h/rollout_??.mp4 | wc -l)
  if [ "$(ls $R/$h/rollout_??_pred_poses.npy 2>/dev/null | wc -l)" != "$n" ]; then
    log "invert_raymap_rollout $h"
    RAYMAP_CAM_FIX=opencv CUDA_VISIBLE_DEVICES=$GPU /venv/main/bin/python tools/invert_raymap_rollout.py \
      --roll_dir $R/$h --ray_downsample 2 --raymap_moment_scale 8.0 > $R/$h/invert_rollout.log 2>&1 || log "invert failed $h"
  fi
done
# 2. flat dir
python3 -I tools/rollout2flat.py $D $R/lookout_turn_cv $R/lookout_turn_cv_h2 || exit 1
N=$(python3 -c "import json;print(len(json.load(open('$D/samples_meta.json'))))")
# 3. PSNR/SSIM/LPIPS vs GT frames
[ "$(python3 -c "import csv;print(sum(1 for _ in csv.DictReader(open('$D/video_quality_metrics.csv'))))" 2>/dev/null)" = "$N" ] || {
  log "video quality"; CUDA_VISIBLE_DEVICES=$GPU /venv/da3/bin/python tools/eval_video_quality.py --results_dir $D \
    --out_csv $D/video_quality_metrics.csv > $D/suite_vq.log 2>&1 || log "vq failed"; }
# 4. DA3 on generated video (poses sim3-fit to GT + depth)
if [ "$(ls $D/viz_da3_fix/*_da3_depth.npy 2>/dev/null | wc -l)" != "$N" ]; then
  log "DA3 on generated video"
  CUDA_VISIBLE_DEVICES=$GPU /venv/da3/bin/python tools/viz_da3_pointcloud.py --results_dir $D --source da3 \
    --out_subdir viz_da3_fix --save_depth > $D/suite_da3.log 2>&1 || log "da3 failed"
fi
# 5. depth coherence (column C only, no GT depth)
/venv/da3/bin/python tools/eval_depth_vs_da3.py --results_dir $D --da3_subdir viz_da3_fix --no_gt_depth --only C \
  --out_csv depth_vs_da3_metrics.csv > $D/suite_depth.log 2>&1 || log "depth failed"
# 6. pose suite + set-level video + goal image + MEt3R
/venv/da3/bin/python tools/eval_suite.py --results_dir $D > $D/suite.log 2>&1 || log "eval_suite failed"
CUDA_VISIBLE_DEVICES=$GPU /venv/metrics/bin/python tools/eval_fvd_musiq.py --results_dir $D > $D/suite_fvd.log 2>&1 || log "fvd failed"
CUDA_VISIBLE_DEVICES=$GPU /venv/metrics/bin/python tools/eval_goal_image.py --results_dir $D > $D/suite_goalimg.log 2>&1 || log "goal image failed"
if [ -x /venv/met3r/bin/python ] && [ ! -f $D/met3r.json ]; then
  CUDA_VISIBLE_DEVICES=$GPU /venv/met3r/bin/python tools/eval_met3r.py --results_dir $D > $D/suite_met3r.log 2>&1 || log "met3r failed"
fi
log "done $D"
