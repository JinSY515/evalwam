#!/bin/bash
# $1 difficulty  $2 gpu  $3 out_dir  $4 mode_idx (default 2)  [$5 = "smoke" -> 1 sample, no manifest]
source /venv/main/bin/activate
cd /root/project/egowam/diffusers/examples/cosmos/scripts/inference
export RAYMAP_CAM_FIX=opencv GOAL_OFFSET_MIN=1 GOAL_OFFSET_MAX=5 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MAN=/root/project/egowam/results_syn/plucker_fidelity/e2/valbench5s_40x3_abs.json
if [ "$5" = "smoke" ]; then SEL="--num_samples 1"; else SEL="--window_manifest $MAN"; fi
CUDA_VISIBLE_DEVICES=$2 python infer_cosmos_lora.py \
  --model_name nvidia/Cosmos-Predict2.5-2B --revision diffusers/base/post-trained \
  --output_dir_base /root/ckpt/wam/goalprog_mode_real6_nomr_maskfix --checkpoint_path checkpoint-500 \
  --output_dir $3 \
  --data_dir /root/data/eval_bench/val_bench5s/id_$1 $SEL --difficulty $1 \
  --num_frames 77 --height 512 --width 512 --full_ft \
  --raymap_tokens --raymap_moment_scale 8.0 --ray_downsample 2 \
  --pred_raymap --pred_goal_state --goal_dist_scale 5.0 \
  --pred_depth --depth_layers 10 --depth_head dpt --depth_taps 0,3,7,10 \
  --hist_cond --hist_cond_n 4 --goal_cond \
  --hist_goal_cond_mask --mode_embed --mode_idx ${4:-2} \
  --prompt "First-person egocentric view walking through an urban environment, captured by a head-mounted wearable camera. Smooth motion with natural head movement." \
  --seed 0 --num_inference_steps 36 --fps 16 --use_synthetic_dataset
