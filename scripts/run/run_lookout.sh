#!/bin/bash
# LookOut (Aria, real, never in training) rollout bench: 51 seqs x NUM_CHUNKS chunks.
# $1 run (dir under /root/ckpt/wam)  $2 checkpoint-N  $3 gpu  $4 out_dir
# env: HG_FLAG=--hist_goal_cond_mask (maskfix runs)  MODE_FLAG="--mode_embed --mode_idx 2" (mode runs)
#      NUM_SAMPLES(51) NUM_CHUNKS(2) GOAL_OFF_MIN/MAX(1/5 = training)  EXTRA_ARGS
# Settings follow README "Long rollout" (vcons ckpt-3000 on lookout_turn_cv) except the goal
# offset, which README itself says to set to 1-5 to match training.
source /venv/main/bin/activate
cd /root/project/egowam/diffusers/examples/cosmos/scripts/inference
export RAYMAP_CAM_FIX=opencv TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MAN=${MAN:-/root/data/eval_bench/rollout_lookout_turn_cv_abs.json}
mkdir -p "$4"
CUDA_VISIBLE_DEVICES=$3 python infer_rollout.py \
  --model_name nvidia/Cosmos-Predict2.5-2B --revision diffusers/base/post-trained \
  --checkpoint_path "$2" --output_dir_base "/root/ckpt/wam/$1" --output_dir "$4" \
  --data_dir /root/project/carla/results/lookout_val_cv \
  --num_samples ${NUM_SAMPLES:-51} --num_chunks ${NUM_CHUNKS:-2} \
  --num_frames 77 --height 512 --width 512 --full_ft \
  --raymap_tokens --raymap_moment_scale 8.0 --ray_downsample 2 \
  --pred_raymap --pred_goal_state \
  --pred_depth --depth_layers 10 --depth_head dpt --depth_taps 0,3,7,10 \
  --prompt "First-person egocentric view walking through an urban environment, captured by a head-mounted wearable camera. Smooth motion with natural head movement." \
  --seed 0 --num_inference_steps 36 --fps 16 --use_synthetic_dataset \
  --hist_cond --hist_cond_n 4 --goal_cond \
  --goal_offset_min ${GOAL_OFF_MIN:-1} --goal_offset_max ${GOAL_OFF_MAX:-5} \
  --window_manifest "$MAN" ${HG_FLAG:-} ${MODE_FLAG:-} ${EXTRA_ARGS:-}
