#!/usr/bin/env bash
set -euo pipefail

# Run ScalingNoise FIFO *baseline* (no search) and save an MP4.
#
# Usage:
#   bash run_scalenoise_baseline.sh
#   PROMPT="..." NEW_VIDEO_LENGTH=64 bash run_scalenoise_baseline.sh

ROOT="/home/ubuntu/angel-research/full_sequence_grpo"
cd "$ROOT"

PROMPT="${PROMPT:-Flocks of birds spiral upwards in synchronized arcs, weaving around the rooftops before scattering into the open sky.}"
SEED="${SEED:-26}"
NEW_VIDEO_LENGTH="${NEW_VIDEO_LENGTH:-64}"
VIDEO_LENGTH="${VIDEO_LENGTH:-16}"
NUM_PARTITIONS="${NUM_PARTITIONS:-4}"
FPS="${FPS:-8}"
OUTPUT_FPS="${OUTPUT_FPS:-8}"
ETA="${ETA:-1.0}"
CFG_SCALE="${CFG_SCALE:-12.0}"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-$ROOT/scalingnoise_outputs/baseline_${RUN_ID}}"
PROMPT_FILE="$OUT_DIR/prompt.txt"

mkdir -p "$OUT_DIR"
printf "%s\n" "$PROMPT" > "$PROMPT_FILE"

echo "Prompt:      $PROMPT"
echo "Prompt file: $PROMPT_FILE"
echo "Out dir:     $OUT_DIR"

PYTHONUNBUFFERED=1 /home/ubuntu/anaconda3/bin/conda run -n scaling-noise python -u scaling-noise/scalenoise.py \
  --search_mode baseline \
  --ckpt_path "$ROOT/base_512_v2/model.ckpt" \
  --config "$ROOT/scaling-noise/configs/inference_t2v_512_v2.0.yaml" \
  --prompt_file "$PROMPT_FILE" \
  --output_dir "$OUT_DIR" \
  --seed "$SEED" \
  --video_length "$VIDEO_LENGTH" \
  --num_partitions "$NUM_PARTITIONS" \
  --new_video_length "$NEW_VIDEO_LENGTH" \
  --fps "$FPS" \
  --output_fps "$OUTPUT_FPS" \
  --eta "$ETA" \
  --unconditional_guidance_scale "$CFG_SCALE" \
  --use_mp4

echo "Done. Outputs under: $OUT_DIR"

