#!/bin/bash
# valbench5s inference for the ablation ladder. Flags mirror each run's TRAINING launcher.
# $1 abl id (I | III | IV)  $2 difficulty  $3 gpu   -> results_syn/abl/<name>/ckptN/valbench5s/<diff>
#   I   joint fm (compact action token)  abl_interpolate_hist_goal_predaction  ckpt-3000
#       train: abl_interpolate_hist_goal_predaction.sh  (pred_action, no camera, no RAYMAP_CAM_FIX)
#   III (II) + goal aux                  goalprog_appearance_wo_init            ckpt-8000 (only one on HF)
#       train: train_raymap_goalprog_appearance.sh  (raymap pred + goal_state, RAYMAP_CAM_FIX UNSET = old convention)
#   IV  (III) + depth pred               goalprog_dl3dv_camfix_d10_dpt          ckpt-3000
#       train: train_raymap_goalprog_dl3dv_camfix.sh (+ dpt depth, RAYMAP_CAM_FIX=opencv)
set -u
source /venv/main/bin/activate
cd /root/project/egowam/diffusers/examples/cosmos/scripts/inference
export GOAL_OFFSET_MIN=1 GOAL_OFFSET_MAX=5 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset RAYMAP_CAM_FIX
RAY="--raymap_tokens --raymap_moment_scale 8.0 --ray_downsample 2 --pred_raymap --pred_goal_state --goal_dist_scale 5.0"
case "$1" in
  I)   NAME=I_jointfm;    RUN=abl_interpolate_hist_goal_predaction; CK=3000
       FLAGS="--pred_action --act_stats /root/project/egowam/action_stats_goalnav_multiroute.json" ;;
  II)  NAME=II_nogoalaux; RUN=goalprog_appearance_nogoalaux;        CK=${ABL_CK:-3000}   # (III) w/o goal aux; old raymap convention
       FLAGS="--raymap_tokens --raymap_moment_scale 8.0 --ray_downsample 2 --pred_raymap" ;;
  III) NAME=III_goalaux;  RUN=goalprog_appearance_wo_init;          CK=8000
       FLAGS="$RAY" ;;
  IV)  NAME=IV_depthpred; RUN=goalprog_dl3dv_camfix_d10_dpt;        CK=${ABL_CK:-3000}
       FLAGS="$RAY --pred_depth --depth_layers 10 --depth_head dpt --depth_taps 0,3,7,10"
       export RAYMAP_CAM_FIX=opencv ;;
  IVc) NAME=IV_cont5k;   RUN=goalprog_dl3dv_camfix_d10_dpt_cont5k; CK=1000; OUTCK=6000   # d10_dpt ck5000 + 1000 more steps
       FLAGS="$RAY --pred_depth --depth_layers 10 --depth_head dpt --depth_taps 0,3,7,10"
       export RAYMAP_CAM_FIX=opencv ;;
  *) echo "unknown abl id $1"; exit 1 ;;
esac
OUT=/root/project/egowam/results_syn/abl/$NAME/ckpt${OUTCK:-$CK}/valbench5s/$2
MAN=${MAN:-/root/project/egowam/results_syn/plucker_fidelity/e2/valbench5s_40x3_abs.json}
mkdir -p "$OUT"
echo "[abl] $1 $NAME run=$RUN ckpt=$CK diff=$2 gpu=$3 RAYMAP_CAM_FIX=${RAYMAP_CAM_FIX:-<unset>}"
CUDA_VISIBLE_DEVICES=$3 python infer_cosmos_lora.py \
  --model_name nvidia/Cosmos-Predict2.5-2B --revision diffusers/base/post-trained \
  --output_dir_base /root/ckpt/wam/$RUN --checkpoint_path checkpoint-$CK \
  --output_dir "$OUT" \
  --data_dir /root/data/eval_bench/val_bench5s/id_$2 --window_manifest "$MAN" --difficulty $2 \
  --num_frames 77 --height 512 --width 512 --full_ft $FLAGS \
  --hist_cond --hist_cond_n 4 --goal_cond \
  --prompt "First-person egocentric view walking through an urban environment, captured by a head-mounted wearable camera. Smooth motion with natural head movement." \
  --seed 0 --num_inference_steps 36 --fps 16 --use_synthetic_dataset ${EXTRA_ARGS:-}
