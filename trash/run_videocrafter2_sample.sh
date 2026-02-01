#!/usr/bin/env bash
set -euo pipefail

cd /home/ubuntu/angel-research/full_sequence_grpo

PROMPTS_FILE="${PROMPTS_FILE:-prompt.txt}"
OUT_BASE_DIR="${OUT_BASE_DIR:-videocrafter2_samples}"

# Sampling mode: "ddim" (default) or "fifo" (sliding-window, long-video).
SAMPLING_MODE="${SAMPLING_MODE:-ddim}"

if [[ ! -f "$PROMPTS_FILE" ]]; then
  echo "ERROR: prompts file not found: $PROMPTS_FILE" >&2
  exit 1
fi

RUN_ID="$(date +%Y%m%d_%H%M%S)"
OUT_RUN_DIR="$OUT_BASE_DIR/prompts_grpo_${RUN_ID}"
mkdir -p "$OUT_RUN_DIR"

echo "Prompts file: $PROMPTS_FILE"
echo "Output base:  $OUT_RUN_DIR"
echo "Sampling:     $SAMPLING_MODE"

i=0
while IFS= read -r PROMPT || [[ -n "${PROMPT:-}" ]]; do
  # Skip empty lines and comment lines
  if [[ -z "${PROMPT//[[:space:]]/}" ]]; then
    continue
  fi
  if [[ "${PROMPT}" =~ ^[[:space:]]*# ]]; then
    continue
  fi

  i=$((i+1))
  # Safe-ish slug for folder names (first 60 chars, alnum/._- only)
  SLUG="$(echo "$PROMPT" | tr -cd '[:alnum:].,_ -' | tr ' ' '_' | cut -c1-60)"
  OUT_DIR="$OUT_RUN_DIR/$(printf '%03d' "$i")_${SLUG}"
  mkdir -p "$OUT_DIR"

  echo ""
  echo "=== Prompt $i ==="
  echo "$PROMPT"
  echo "Out: $OUT_DIR"

  # This runs GRPO and saves:
  # - $OUT_DIR/before.mp4
  # - $OUT_DIR/intermediate_rollout/*.mp4 + logs
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
done < "$PROMPTS_FILE"

echo ""
echo "Done. Outputs in: $OUT_RUN_DIR"

