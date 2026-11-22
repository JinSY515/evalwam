#!/bin/bash
# Every 30 min until the ablation chains finish: invert raymaps (III/IV/vcons_mode) -> interim_metrics.py
# Output: /root/project/egowam/results_syn/abl/METRICS.md
cd /root/project/evalwam
export HF_HOME=/workspace/.hf_home CUDA_VISIBLE_DEVICES=${GPU:-0}
inv() {  # $1 dir  $2 cam_fix(opencv|none)
  [ -f "$1/samples_meta.json" ] || return
  if [ "$2" = opencv ]; then export RAYMAP_CAM_FIX=opencv; else unset RAYMAP_CAM_FIX; fi
  /venv/main/bin/python tools/invert_raymap.py --results_dir "$1" --stage s3 --ray_downsample 2 --raymap_moment_scale 8.0 > "$1/invert_raymap_s3.log" 2>&1 || echo "[loop] invert fail $1"
}
while true; do
  for d in easy medium hard; do
    inv /root/project/egowam/results_syn/abl/III_goalaux/ckpt8000/valbench5s/$d none
    inv /root/project/egowam/results_syn/abl/IV_depthpred/ckpt3000/valbench5s/$d opencv
    inv /root/project/egowam/results_syn/goalprog_dl3dv_camfix_d10_dpt_vcons_mode/ckpt500/valbench5s/$d opencv
  done
  /venv/da3/bin/python tools/interim_metrics.py > /root/project/egowam/results_syn/abl/interim_metrics.log 2>&1 || echo "[loop] metrics fail"
  echo "[loop] $(date +%H:%M) updated"
  [ "$(cat /root/project/egowam/results_syn/abl/chain_*.out 2>/dev/null | grep -c '_DONE')" = 3 ] && { echo "[loop] all ablations done"; break; }
  sleep 1800
done
