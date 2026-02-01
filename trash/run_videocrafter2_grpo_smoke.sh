#!/usr/bin/env bash
set -euo pipefail

cd /home/ubuntu/angel-research/full_sequence_grpo

/home/ubuntu/anaconda3/bin/conda run -n scaling-noise python videocrafter2_grpo.py \
  --prompt "A yellow rubber duck floating on waves at sunset, realistic motion" \
  --ckpt_path scaling-noise/videocrafter_models/base_512_v2/model.ckpt \
  --out_dir videocrafter2_grpo_runs \
  --num_inference_steps 2 \
  --num_grpo_steps 1 \
  --num_rollouts 2 \
  --trainable none

