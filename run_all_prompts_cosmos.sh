#!/bin/bash
# Batch GRPO for Cosmos-Predict2.5 (unified_grpo native loader, model key e.g. 2B/pre-trained).
# Optional baseline: HuggingFace Diffusers sample -> baseline/baseline.mp4 (see unified_grpo/baseline/cosmos_baseline.sh).
#
# Default prompts: origin_grpo/experiment_prompts.txt (override: PROMPTS_FILE=path/to/other.txt).
#
# Requires: cosmos-predict2.5/ at repo root (same as create_cosmos_adapter).
#
# Local checkpoint override:
#   MODEL_PATH=/path/to/checkpoint COSMOS_EXPERIMENT_NAME=... bash run_all_prompts_cosmos.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
cd "${REPO_ROOT}"

COSMOS_ROOT="${COSMOS_ROOT:-${REPO_ROOT}/cosmos-predict2.5}"
export PYTHONPATH="${REPO_ROOT}:${COSMOS_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONNOUSERSITE=1

SAVE_SNAPSHOTS="${SAVE_SNAPSHOTS:-1}"
VIDEO_SNAPSHOT_SH="${REPO_ROOT}/scripts/video_to_snapshot.sh"
SAVE_KEYFRAME_STRIP="${SAVE_KEYFRAME_STRIP:-1}"
KEYFRAME_STRIP_FRAMES="${KEYFRAME_STRIP_FRAMES:-2}"
KEYFRAME_STRIP_SH="${REPO_ROOT}/scripts/save_prompt_keyframe_strips.sh"
SAVE_DENOISING_STRIP="${SAVE_DENOISING_STRIP:-0}"
DENOISING_SNAPSHOT_STRIDE="${DENOISING_SNAPSHOT_STRIDE:-5}"
SAVE_CHECKPOINTS="${SAVE_CHECKPOINTS:-1}"
# Target **playback** length (seconds) for saved GRPO MP4s (fps = num_frames / duration).
# Baseline Diffusers export is 16 fps (cosmos-predict2.5/scripts/diffusers_inference.py); optional
# ffmpeg retime below stretches baseline.mp4 to this duration as well.
OUTPUT_VIDEO_DURATION_S="${OUTPUT_VIDEO_DURATION_S:-6.0}"
# Diffusers baseline encodes at this FPS; used only to default NUM_FRAMES when unset.
COSMOS_BASELINE_EXPORT_FPS="${COSMOS_BASELINE_EXPORT_FPS:-16}"
# After baseline: ffmpeg retime to OUTPUT_VIDEO_DURATION_S (same idea as run_all_prompts_wan.sh).
COSMOS_BASELINE_RETIME="${COSMOS_BASELINE_RETIME:-1}"

# One prompt per line; repo-relative or absolute.
PROMPTS_FILE="${PROMPTS_FILE:-origin_grpo/experiment_prompts.txt}"

echo "======================================================================"
echo "BATCH GRPO TRAINING - COSMOS-PREDICT2.5 - PROMPTS FROM ${PROMPTS_FILE}"
echo "======================================================================"
echo ""

MODEL_TYPE="${MODEL_TYPE:-cosmos}"
# Registered key (see cosmos_predict2.config) or local checkpoint directory
MODEL_PATH="${MODEL_PATH:-2B/pre-trained}"
HEIGHT="${HEIGHT:-704}"
WIDTH="${WIDTH:-1280}"
# Native GRPO may snap this to the checkpoint’s fixed temporal length (see CosmosPredict25Adapter).
# Default: round(OUTPUT_VIDEO_DURATION_S * COSMOS_BASELINE_EXPORT_FPS), e.g. 6s @ 16fps -> 96.
if [[ -z "${NUM_FRAMES:-}" ]]; then
  NUM_FRAMES="$(python3 -c "d=float('${OUTPUT_VIDEO_DURATION_S}'); f=float('${COSMOS_BASELINE_EXPORT_FPS}'); print(max(1, int(round(d*f))))")"
fi
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
NUM_GRPO_STEPS="${NUM_GRPO_STEPS:-15}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-6}"
LR="${LR:-1e-4}"
SEED="${SEED:-42}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-7.0}"
UNFREEZE_PERCENTAGE="${UNFREEZE_PERCENTAGE:-0.20}"
USE_LORA="${USE_LORA:-1}"
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
COSMOS_LORA_TARGET_MODULES="${COSMOS_LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,output_proj,mlp.layer1,mlp.layer2}"
REWARD_BACKEND="${REWARD_BACKEND:-image_clip}"
REWARD_DEBUG="${REWARD_DEBUG:-1}"
CLIP_NUM_FRAMES="${CLIP_NUM_FRAMES:-0}"
CLIP_AGGREGATION="${CLIP_AGGREGATION:-video_mean_pool}"
ADAPTIVE_PHYSICS_HIDDEN_DIM="${ADAPTIVE_PHYSICS_HIDDEN_DIM:-32}"
PHYSICS_CATEGORY_OVERRIDE="${PHYSICS_CATEGORY_OVERRIDE:-}"
PHYSICS_HANDCRAFTED_W_MOTION="${PHYSICS_HANDCRAFTED_W_MOTION:-0.35}"
PHYSICS_HANDCRAFTED_W_CATEGORY="${PHYSICS_HANDCRAFTED_W_CATEGORY:-0.65}"
RUN_BASELINE="${RUN_BASELINE:-1}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-}"

# Optional: only when MODEL_PATH is a local directory (passed through to run.py)
COSMOS_EXPERIMENT_NAME="${COSMOS_EXPERIMENT_NAME:-}"
COSMOS_CONFIG_FILE="${COSMOS_CONFIG_FILE:-}"

# Baseline (Diffusers) — independent HF id/revision; default matches 2B post-trained
COSMOS_DIFFUSERS_MODEL_ID="${COSMOS_DIFFUSERS_MODEL_ID:-nvidia/Cosmos-Predict2.5-2B}"
COSMOS_DIFFUSERS_REVISION="${COSMOS_DIFFUSERS_REVISION:-diffusers/base/post-trained}"
COSMOS_BASELINE_NUM_STEPS="${COSMOS_BASELINE_NUM_STEPS:-50}"
COSMOS_DEVICE="${COSMOS_DEVICE:-cuda}"
# Skip Diffusers cosmos_guardrail HF downloads (gated repos / PEFT adapter issues). Set 0 if you have full access.
COSMOS_DISABLE_GUARDRAIL="${COSMOS_DISABLE_GUARDRAIL:-1}"

if [[ "$MODEL_TYPE" != "cosmos" ]]; then
  echo "❌ MODEL_TYPE='$MODEL_TYPE' is not supported by this script. Use MODEL_TYPE=cosmos."
  exit 2
fi

if [[ ! -d "$COSMOS_ROOT" ]]; then
  echo "❌ cosmos-predict2.5 not found at: $COSMOS_ROOT" >&2
  exit 1
fi

if [[ "${PROMPTS_FILE}" != /* ]]; then
  PROMPTS_FILE="${REPO_ROOT}/${PROMPTS_FILE}"
fi

if [[ ! -f "$PROMPTS_FILE" ]]; then
  echo "ERROR: Prompt file not found: $PROMPTS_FILE" >&2
  exit 1
fi

PROMPT_FILES=("$PROMPTS_FILE")
TOTAL_FILES=1
TOTAL_PROMPTS_IN_FILE=$(grep -cve '^[[:space:]]*$' "$PROMPTS_FILE" || true)
echo "Using prompt file: $(basename "$PROMPTS_FILE")"
echo "  - prompts: $TOTAL_PROMPTS_IN_FILE"
echo ""

AUTO_YES="${AUTO_YES:-0}"
if [[ "${AUTO_YES}" == "1" || "${AUTO_YES}" == "true" || ! -t 0 ]]; then
  echo "Non-interactive run detected (or AUTO_YES set). Continuing without prompt."
else
  read -p "Continue? (y/n) " -n 1 -r
  echo
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 0
  fi
fi

MODEL_NAME="${MODEL_PATH//\//-}"
BATCH_TIMESTEP=$(date +%Y%m%d_%H%M%S)
BATCH_DIR="./${MODEL_NAME}_grpo_${BATCH_TIMESTEP}"
mkdir -p "$BATCH_DIR"

echo ""
echo "Batch output: $BATCH_DIR"
echo "======================================================================"
echo ""

TOTAL_PROMPTS_PROCESSED=0
FILE_INDEX=0

for PROMPTS_FILE in "${PROMPT_FILES[@]}"; do
  FILE_INDEX=$((FILE_INDEX + 1))
  FILE_BASE=$(basename "$PROMPTS_FILE" .txt)
  TIMESTEP=$(date +%Y%m%d_%H%M%S)
  OUTPUT_TOP_REL="${BATCH_DIR}/${FILE_BASE}_${TIMESTEP}"
  mkdir -p "$OUTPUT_TOP_REL"

  TOTAL_PROMPTS="$(grep -cve '^[[:space:]]*$' "${PROMPTS_FILE}" || true)"
  echo ""
  echo "######################################################################"
  echo "FILE $FILE_INDEX/$TOTAL_FILES: $PROMPTS_FILE -> $OUTPUT_TOP_REL"
  echo "######################################################################"
  echo ""

  LINE_NUM=0
  PROMPT_IDX=0
  while IFS= read -r PROMPT || [ -n "$PROMPT" ]; do
    LINE_NUM=$((LINE_NUM + 1))
    if [[ -z "${PROMPT//[[:space:]]/}" ]]; then
      continue
    fi
    PROMPT_IDX=$((PROMPT_IDX + 1))
    echo ""
    echo "======================================================================"
    echo "Prompt $PROMPT_IDX/$TOTAL_PROMPTS (file: $FILE_BASE)"
    echo "======================================================================"
    echo "$PROMPT"
    echo ""

    PROMPT_SHORT=$(echo "$PROMPT" | head -c 50 | tr -cd '[:alnum:] ' | tr ' ' '-')
    PROMPT_ID="p$(printf '%03d' $PROMPT_IDX)"
    OUTPUT_DIR_REL="${OUTPUT_TOP_REL}/${PROMPT_ID}_${PROMPT_SHORT}"
    OUTPUT_DIR="$(python3 - "$OUTPUT_DIR_REL" <<'PY'
import os, sys
print(os.path.abspath(sys.argv[1]))
PY
)"
    mkdir -p "$OUTPUT_DIR"
    echo "$PROMPT" > "$OUTPUT_DIR/prompt.txt"

    if [[ "${RUN_BASELINE}" == "1" ]]; then
      echo "Step 1/2: Generating baseline (Cosmos Diffusers inference)..."
      mkdir -p "$OUTPUT_DIR/baseline"
      PROMPT="$PROMPT" \
        OUTPUT_DIR="$OUTPUT_DIR/baseline" \
        SEED="$SEED" \
        NUM_FRAMES="$NUM_FRAMES" \
        NEGATIVE_PROMPT="$NEGATIVE_PROMPT" \
        COSMOS_ROOT="$COSMOS_ROOT" \
        COSMOS_DIFFUSERS_MODEL_ID="$COSMOS_DIFFUSERS_MODEL_ID" \
        COSMOS_DIFFUSERS_REVISION="$COSMOS_DIFFUSERS_REVISION" \
        COSMOS_BASELINE_NUM_STEPS="$COSMOS_BASELINE_NUM_STEPS" \
        COSMOS_DEVICE="$COSMOS_DEVICE" \
        COSMOS_DISABLE_GUARDRAIL="$COSMOS_DISABLE_GUARDRAIL" \
        bash "${REPO_ROOT}/unified_grpo/baseline/cosmos_baseline.sh" || {
        echo "⚠️ Baseline failed for prompt $PROMPT_IDX (cosmos_baseline.sh). Continuing to GRPO..."
      }
      if [[ -f "$OUTPUT_DIR/baseline/baseline.mp4" ]] && [[ "${COSMOS_BASELINE_RETIME}" == "1" ]] && command -v ffmpeg >/dev/null 2>&1; then
        in_file="${OUTPUT_DIR}/baseline/baseline.mp4"
        in_dur="$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "${in_file}" 2>/dev/null || true)"
        if [[ -n "${in_dur}" ]]; then
          read -r PTS_SCALE TARGET_FPS < <(python3 - <<PY
dur=float("${in_dur}")
target=float("${OUTPUT_VIDEO_DURATION_S}")
frames=int("${NUM_FRAMES}")
print(target / dur, frames / target)
PY
)
        else
          PTS_SCALE="1.0"
          TARGET_FPS="$(python3 - <<PY
frames=int("${NUM_FRAMES}")
target=float("${OUTPUT_VIDEO_DURATION_S}")
print(frames / target)
PY
)"
        fi
        tmp_out="${OUTPUT_DIR}/baseline/baseline_${OUTPUT_VIDEO_DURATION_S}s.mp4"
        ffmpeg -y -i "${in_file}" -vf "setpts=${PTS_SCALE}*PTS" -r "${TARGET_FPS}" -vsync cfr -an -movflags +faststart "${tmp_out}" \
          && mv -f "${tmp_out}" "${in_file}"
      fi
      if [[ -f "$OUTPUT_DIR/baseline/baseline.mp4" ]]; then
        echo "✅ Baseline saved: $OUTPUT_DIR/baseline/baseline.mp4"
        if [[ "${SAVE_SNAPSHOTS}" == "1" && -f "${VIDEO_SNAPSHOT_SH}" ]]; then
          bash "${VIDEO_SNAPSHOT_SH}" "$OUTPUT_DIR/baseline/baseline.mp4" || true
        fi
      fi
      echo ""
    fi

    echo "Step 2/2: Running GRPO training..."
    cmd=(python "${REPO_ROOT}/unified_grpo/run.py"
      --model-type "$MODEL_TYPE"
      --model-path "$MODEL_PATH"
      --prompt "$PROMPT"
      --negative-prompt "$NEGATIVE_PROMPT"
      --reward-backend "$REWARD_BACKEND"
      --gradient-checkpointing
      --height "$HEIGHT"
      --width "$WIDTH"
      --num-frames "$NUM_FRAMES"
      --guidance-scale "$GUIDANCE_SCALE"
      --num-inference-steps "$NUM_INFERENCE_STEPS"
      --num-grpo-steps "$NUM_GRPO_STEPS"
      --num-rollouts "$NUM_ROLLOUTS"
      --lr "$LR"
      --seed "$SEED"
      --unfreeze-percentage "$UNFREEZE_PERCENTAGE"
      --output-dir "$OUTPUT_DIR/grpo"
      --output-video-duration-s "$OUTPUT_VIDEO_DURATION_S"
      --clip-num-frames "$CLIP_NUM_FRAMES"
      --clip-aggregation "$CLIP_AGGREGATION"
      --adaptive-physics-hidden-dim "$ADAPTIVE_PHYSICS_HIDDEN_DIM"
      --physics-handcrafted-w-motion "$PHYSICS_HANDCRAFTED_W_MOTION"
      --physics-handcrafted-w-category "$PHYSICS_HANDCRAFTED_W_CATEGORY"
    )
    if [[ "${SAVE_CHECKPOINTS}" == "1" ]]; then
      cmd+=(--save-checkpoint-dir "$OUTPUT_DIR/grpo/checkpoint")
    fi
    if [[ "${REWARD_DEBUG}" == "1" || "${REWARD_DEBUG}" == "true" ]]; then
      cmd+=(--reward-debug)
    fi
    if [[ -n "${PHYSICS_CATEGORY_OVERRIDE}" ]]; then
      cmd+=(--physics-category-override "$PHYSICS_CATEGORY_OVERRIDE")
    fi
    if [[ -n "${COSMOS_EXPERIMENT_NAME}" ]]; then
      cmd+=(--cosmos-experiment-name "$COSMOS_EXPERIMENT_NAME")
    fi
    if [[ -n "${COSMOS_CONFIG_FILE}" ]]; then
      cmd+=(--cosmos-config-file "$COSMOS_CONFIG_FILE")
    fi
    if [[ "$USE_LORA" == "1" ]]; then
      cmd+=(--use-lora --lora-rank "$LORA_RANK" --lora-alpha "$LORA_ALPHA"
        --cosmos-lora-target-modules "$COSMOS_LORA_TARGET_MODULES")
    fi
    if [[ "${SAVE_DENOISING_STRIP}" == "1" ]]; then
      cmd+=(
        --save-denoising-strip-png
        --save-denoising-step-snapshots
        --denoising-step-snapshot-stride "$DENOISING_SNAPSHOT_STRIDE"
      )
    fi
    "${cmd[@]}" || {
      echo "⚠️ GRPO failed for prompt $PROMPT_IDX, continuing..."
      continue
    }
    echo "✅ GRPO complete for prompt $PROMPT_IDX"
    if [[ "${SAVE_SNAPSHOTS}" == "1" && -f "${VIDEO_SNAPSHOT_SH}" ]]; then
      for v in "$OUTPUT_DIR/grpo"/*_grpo.mp4; do
        [[ -f "$v" ]] || continue
        bash "${VIDEO_SNAPSHOT_SH}" "$v" || true
      done
    fi
    if [[ "${SAVE_KEYFRAME_STRIP}" == "1" && -f "${KEYFRAME_STRIP_SH}" ]]; then
      bash "${KEYFRAME_STRIP_SH}" "$OUTPUT_DIR" "${KEYFRAME_STRIP_FRAMES}" || true
    fi
    echo ""
    TOTAL_PROMPTS_PROCESSED=$((TOTAL_PROMPTS_PROCESSED + 1))
  done < "$PROMPTS_FILE"
done

echo ""
echo "======================================================================"
echo "BATCH PROCESSING COMPLETE! 🎉"
echo "======================================================================"
echo ""
echo "Processed $TOTAL_PROMPTS_PROCESSED prompts across $TOTAL_FILES file(s)"
echo "Output: $BATCH_DIR/"
echo ""
echo "Structure:"
echo "  <model_key_sanitized>_grpo_YYYYMMDD_HHMMSS/"
echo "  ├── <prompt_file_stem>_YYYYMMDD_HHMMSS/"
echo "  │   ├── p001_*/"
echo "  │   │   ├── baseline/baseline.mp4 (+ *_snapshot.png)  # Diffusers 2B, if RUN_BASELINE=1"
echo "  │   │   ├── grpo/cosmos_grpo.mp4 (+ *_snapshot.png)"
echo "  │   │   ├── baseline_keyframes.png, grpo_keyframes.png (${KEYFRAME_STRIP_FRAMES} frames each, if SAVE_KEYFRAME_STRIP=1)"
echo "  │   │   ├── grpo/checkpoint/"
echo "  │   │   └── prompt.txt"
echo "  │   └── ..."
echo "  └── ..."
echo "======================================================================"
