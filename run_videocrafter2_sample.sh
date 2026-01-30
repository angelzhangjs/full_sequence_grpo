#!/usr/bin/env bash
set -euo pipefail

cd /home/ubuntu/angel-research/full_sequence_grpo

PROMPT="Flocks of birds spiral upwards in synchronized arcs, weaving around the rooftops before scattering into the open sky."
OUT_DIR="videocrafter2_samples/birds_spiral_grpo_$(date +%Y%m%d_%H%M%S)"

# Sampling mode: "ddim" (default) or "fifo" (sliding-window, long-video).
SAMPLING_MODE="${SAMPLING_MODE:-ddim}"

# This runs GRPO and saves:
# - $OUT_DIR/before.mp4
# - $OUT_DIR/intermediate_rollout/rollout_step*_r*.mp4 + logs
# - $OUT_DIR/after.mp4
PYTHONUNBUFFERED=1 /home/ubuntu/anaconda3/bin/conda run -n scaling-noise python -u videocrafter2_grpo_pipeline.py \
  --prompt "$PROMPT" \
  --out_dir "$OUT_DIR" \
  --height 320 \
  --width 512 \
  --num_frames 16 \
  --fps 8 \
  --seed 26 \
  --num_inference_steps 40 \
  --num_grpo_steps 25 \
  --num_rollouts 3 \
  --rollout_noise_scale 0.5 \
  --lr 1e-4 \
  --grad_clip 1.0 \
  --normalize_advantages 1 \
  --trainable temporal \
  --sampling_mode "$SAMPLING_MODE" \
  --fifo_video_length 16 \
  --fifo_new_video_length 64 \
  --fifo_num_partitions 4 \
  --fifo_lookahead_denoising 1 \
  --fifo_train_last_partitions 2

echo "Saved baseline to: $OUT_DIR/before.mp4"
if [[ -f "$OUT_DIR/after.mp4" ]]; then
  echo "Saved GRPO result to: $OUT_DIR/after.mp4"
else
  echo "NOTE: after.mp4 not found at: $OUT_DIR/after.mp4"
  echo "  Check $OUT_DIR/intermediate_rollout/logs/ for why GRPO may have skipped/failed."
fi

