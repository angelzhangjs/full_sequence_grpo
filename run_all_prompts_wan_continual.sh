#!/bin/bash
# Continual Wan GRPO training across all prompts in a single prompt file.
# Unlike run_all_prompts_wan.sh, this keeps one WAN model instance alive and
# accumulates updates across prompts, then saves one final checkpoint.

set -euo pipefail

cd "$(dirname "$0")"
REPO_ROOT="${PWD}"

export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/Wan2.1:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONNOUSERSITE=1

PROMPTS_FILE="${PROMPTS_FILE:-origin_grpo/newyear_physics_prompts_100.txt}"
MODEL_PATH="${MODEL_PATH:-Wan-AI/Wan2.1-T2V-1.3B}"
WAN_TASK="${WAN_TASK:-t2v-1.3B}"
WAN_SIZE="${WAN_SIZE:-832*480}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
NUM_GRPO_STEPS="${NUM_GRPO_STEPS:-15}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-5}"
LR="${LR:-1e-4}"
SEED="${SEED:-42}"
if [[ "$WAN_SIZE" == *"*"* ]]; then
  WIDTH="${WIDTH:-${WAN_SIZE%%\**}}"
  HEIGHT="${HEIGHT:-${WAN_SIZE#*\*}}"
else
  WIDTH="${WIDTH:-832}"
  HEIGHT="${HEIGHT:-480}"
fi
NUM_FRAMES="${NUM_FRAMES:-33}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-6.0}"
UNFREEZE_PERCENTAGE="${UNFREEZE_PERCENTAGE:-0.20}"
USE_LORA="${USE_LORA:-1}"
LORA_BLOCKS="${LORA_BLOCKS:-}"
LORA_RANK="${LORA_RANK:-4}"
LORA_ALPHA="${LORA_ALPHA:-8}"
REWARD_BACKEND="${REWARD_BACKEND:-image_clip}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-}"
REWARD_DEBUG="${REWARD_DEBUG:-1}"
SAVE_DENOISING_STRIP="${SAVE_DENOISING_STRIP:-1}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"
DENOISING_SNAPSHOT_STRIDE="${DENOISING_SNAPSHOT_STRIDE:-5}"
CLIP_NUM_FRAMES="${CLIP_NUM_FRAMES:-0}"
CLIP_AGGREGATION="${CLIP_AGGREGATION:-video_mean_pool}"
ADAPTIVE_PHYSICS_HIDDEN_DIM="${ADAPTIVE_PHYSICS_HIDDEN_DIM:-32}"
PHYSICS_CATEGORY_OVERRIDE="${PHYSICS_CATEGORY_OVERRIDE:-}"
PHYSICS_HANDCRAFTED_W_MOTION="${PHYSICS_HANDCRAFTED_W_MOTION:-0.35}"
PHYSICS_HANDCRAFTED_W_CATEGORY="${PHYSICS_HANDCRAFTED_W_CATEGORY:-0.65}"

if [[ "${PROMPTS_FILE}" != /* ]]; then
  PROMPTS_FILE="${REPO_ROOT}/${PROMPTS_FILE}"
fi

if [[ ! -f "${PROMPTS_FILE}" ]]; then
  echo "ERROR: Prompt file not found: ${PROMPTS_FILE}" >&2
  exit 1
fi

AUTO_YES="${AUTO_YES:-0}"
if [[ "${AUTO_YES}" != "1" && "${AUTO_YES}" != "true" && -t 0 ]]; then
  read -p "Continue continual WAN training on $(basename "${PROMPTS_FILE}")? (y/n) " -n 1 -r
  echo
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 0
  fi
fi

cmd=(python "${REPO_ROOT}/unified_grpo/run_continual_wan.py"
  --model-path "${MODEL_PATH}"
  --wan-task "${WAN_TASK}"
  --prompt-file "${PROMPTS_FILE}"
  --negative-prompt "${NEGATIVE_PROMPT}"
  --height "${HEIGHT}"
  --width "${WIDTH}"
  --num-frames "${NUM_FRAMES}"
  --guidance-scale "${GUIDANCE_SCALE}"
  --num-inference-steps "${NUM_INFERENCE_STEPS}"
  --num-grpo-steps "${NUM_GRPO_STEPS}"
  --num-rollouts "${NUM_ROLLOUTS}"
  --lr "${LR}"
  --seed "${SEED}"
  --unfreeze-percentage "${UNFREEZE_PERCENTAGE}"
  --sample-shift 5.0
  --sample-solver unipc
  --reward-backend "${REWARD_BACKEND}"
  --clip-num-frames "${CLIP_NUM_FRAMES}"
  --clip-aggregation "${CLIP_AGGREGATION}"
  --adaptive-physics-hidden-dim "${ADAPTIVE_PHYSICS_HIDDEN_DIM}"
  --physics-handcrafted-w-motion "${PHYSICS_HANDCRAFTED_W_MOTION}"
  --physics-handcrafted-w-category "${PHYSICS_HANDCRAFTED_W_CATEGORY}"
)
if [[ "${GRADIENT_CHECKPOINTING}" == "1" || "${GRADIENT_CHECKPOINTING}" == "true" ]]; then
  cmd+=(--gradient-checkpointing)
fi
if [[ -n "${OUTPUT_ROOT}" ]]; then
  cmd+=(--output-root "${OUTPUT_ROOT}")
fi
if [[ "${USE_LORA}" == "1" ]]; then
  cmd+=(--use-lora --lora-blocks "${LORA_BLOCKS}" --lora-rank "${LORA_RANK}" --lora-alpha "${LORA_ALPHA}")
fi
if [[ "${SAVE_DENOISING_STRIP}" == "1" ]]; then
  cmd+=(--save-denoising-strip-png --save-denoising-step-snapshots --denoising-step-snapshot-stride "${DENOISING_SNAPSHOT_STRIDE}")
fi
if [[ "${REWARD_DEBUG}" == "1" || "${REWARD_DEBUG}" == "true" ]]; then
  cmd+=(--reward-debug)
fi
if [[ -n "${PHYSICS_CATEGORY_OVERRIDE}" ]]; then
  cmd+=(--physics-category-override "${PHYSICS_CATEGORY_OVERRIDE}")
fi

printf '%q ' "${cmd[@]}"
echo
"${cmd[@]}"
