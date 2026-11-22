#!/bin/bash
# run_lookout.sh for ablation-ladder checkpoints: the model flags come from env FLAGS instead of the main run's
# hard-coded raymap/goal_state/depth set, and the raymap convention from env CAM_FIX (opencv | none = old, unset).
# $1 run (dir under /root/ckpt/wam)  $2 checkpoint-N  $3 gpu  $4 out_dir
# env: FLAGS (model flags, e.g. "--raymap_tokens --raymap_moment_scale 8.0 --ray_downsample 2 --pred_raymap")
#      CODE (repo root holding diffusers/, default /root/project/egowam; the ray-fusion arms use the worktree
#            /root/project/egowam_rayabl -> its diffusers/src goes first on PYTHONPATH)
#      CAM_FIX (opencv|none)  NUM_SAMPLES(51) NUM_CHUNKS(1)  GOAL_OFF_MIN/MAX(1/5)  MAN  EXTRA_ARGS
source /venv/main/bin/activate
CODE=${CODE:-/root/project/egowam}
[ "$CODE" != /root/project/egowam ] && export PYTHONPATH=$CODE/diffusers/src${PYTHONPATH:+:$PYTHONPATH}
cd $CODE/diffusers/examples/cosmos/scripts/inference
if [ "${CAM_FIX:-opencv}" = opencv ]; then export RAYMAP_CAM_FIX=opencv; else unset RAYMAP_CAM_FIX; fi
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MAN=${MAN:-/root/data/eval_bench/rollout_lookout_turn_cv_abs.json}
mkdir -p "$4"
echo "[lookout-abl] code=$CODE run=$1 $2 gpu=$3 RAYMAP_CAM_FIX=${RAYMAP_CAM_FIX:-<unset>} FLAGS=${FLAGS}"
CUDA_VISIBLE_DEVICES=$3 python infer_rollout.py \
  --model_name nvidia/Cosmos-Predict2.5-2B --revision diffusers/base/post-trained \
  --checkpoint_path "$2" --output_dir_base "/root/ckpt/wam/$1" --output_dir "$4" \
  --data_dir /root/project/carla/results/lookout_val_cv \
  --num_samples ${NUM_SAMPLES:-51} --num_chunks ${NUM_CHUNKS:-1} \
  --num_frames 77 --height 512 --width 512 --full_ft ${FLAGS} \
  --prompt "First-person egocentric view walking through an urban environment, captured by a head-mounted wearable camera. Smooth motion with natural head movement." \
  --seed 0 --num_inference_steps 36 --fps 16 --use_synthetic_dataset \
  --hist_cond --hist_cond_n 4 --goal_cond \
  --goal_offset_min ${GOAL_OFF_MIN:-1} --goal_offset_max ${GOAL_OFF_MAX:-5} \
  --window_manifest "$MAN" ${EXTRA_ARGS:-}
