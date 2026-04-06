#!/bin/bash
# End-to-end custom PAI-bench-style evaluation for saved GRPO checkpoints.
# Supports CogVideoX, LTX, and Wan generation stubs, then custom VQA generation,
# then VQA evaluation.

set -euo pipefail

cd "$(dirname "$0")"
REPO_ROOT="${PWD}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1

MODEL_TYPE="${MODEL_TYPE:-wan}"   # cogvideox | ltx | wan
CHECKPOINT_DIR="${CHECKPOINT_DIR:-}"
PROMPT_FILE="${PROMPT_FILE:-origin_grpo/newyear_physics_prompts_100.txt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./pai_bench_custom_eval_$(date +%Y%m%d_%H%M%S)}"

# Shared generation knobs
MODEL_PATH="${MODEL_PATH:-}"
NUM_FRAMES="${NUM_FRAMES:-32}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-7.5}"
SEED="${SEED:-42}"
NUM_SEEDS="${NUM_SEEDS:-1}"
SEED_STRIDE="${SEED_STRIDE:-1}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-}"

# CogVideoX / LTX geometry
HEIGHT="${HEIGHT:-480}"
WIDTH="${WIDTH:-720}"
FPS="${FPS:-8}"

# LTX-specific
LTX_PIPELINE_CONFIG="${LTX_PIPELINE_CONFIG:-}"

# Wan-specific
WAN_TASK="${WAN_TASK:-t2v-1.3B}"
WAN_SIZE="${WAN_SIZE:-832*480}"
WAN_FPS="${WAN_FPS:-16}"
SAMPLE_SHIFT="${SAMPLE_SHIFT:-5.0}"
SAMPLE_SOLVER="${SAMPLE_SOLVER:-unipc}"

# VQA evaluator
VQA_MODEL_NAME="${VQA_MODEL_NAME:-Qwen/Qwen2-VL-2B-Instruct}"
VQA_DEVICE="${VQA_DEVICE:-cuda}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"

if [[ -z "${CHECKPOINT_DIR}" ]]; then
  echo "ERROR: CHECKPOINT_DIR is required" >&2
  exit 1
fi

if [[ "${CHECKPOINT_DIR}" != /* ]]; then
  CHECKPOINT_DIR="${REPO_ROOT}/${CHECKPOINT_DIR}"
fi
if [[ "${PROMPT_FILE}" != /* ]]; then
  PROMPT_FILE="${REPO_ROOT}/${PROMPT_FILE}"
fi
if [[ "${OUTPUT_ROOT}" != /* ]]; then
  OUTPUT_ROOT="${REPO_ROOT}/${OUTPUT_ROOT}"
fi

if [[ ! -d "${CHECKPOINT_DIR}" ]]; then
  echo "ERROR: Checkpoint directory not found: ${CHECKPOINT_DIR}" >&2
  exit 1
fi
if [[ ! -f "${PROMPT_FILE}" ]]; then
  echo "ERROR: Prompt file not found: ${PROMPT_FILE}" >&2
  exit 1
fi

GEN_OUTPUT_DIR="${OUTPUT_ROOT}/generated"
VQA_QUESTIONS_DIR="${OUTPUT_ROOT}/custom_vqa"
VQA_RESULTS_DIR="${OUTPUT_ROOT}/vqa_results"

mkdir -p "${GEN_OUTPUT_DIR}" "${VQA_QUESTIONS_DIR}" "${VQA_RESULTS_DIR}"

echo "======================================================================"
echo "CUSTOM PAI-BENCH EVALUATION"
echo "======================================================================"
echo "Model type:      ${MODEL_TYPE}"
echo "Checkpoint dir:  ${CHECKPOINT_DIR}"
echo "Prompt file:     ${PROMPT_FILE}"
echo "Output root:     ${OUTPUT_ROOT}"
echo "VQA model:       ${VQA_MODEL_NAME}"
echo "======================================================================"
echo

case "${MODEL_TYPE}" in
  cogvideox)
    GEN_CMD=(python "${REPO_ROOT}/physical-ai-bench/generation/generate_cogvideox_lora.py"
      --checkpoint-dir "${CHECKPOINT_DIR}"
      --prompt-file "${PROMPT_FILE}"
      --output-dir "${GEN_OUTPUT_DIR}"
      --height "${HEIGHT}"
      --width "${WIDTH}"
      --num-frames "${NUM_FRAMES}"
      --guidance-scale "${GUIDANCE_SCALE}"
      --num-inference-steps "${NUM_INFERENCE_STEPS}"
      --fps "${FPS}"
      --seed "${SEED}"
      --num-seeds "${NUM_SEEDS}"
      --seed-stride "${SEED_STRIDE}"
    )
    [[ -n "${MODEL_PATH}" ]] && GEN_CMD+=(--model-path "${MODEL_PATH}")
    ;;
  ltx)
    GEN_CMD=(python "${REPO_ROOT}/physical-ai-bench/generation/generate_ltx_lora.py"
      --checkpoint-dir "${CHECKPOINT_DIR}"
      --prompt-file "${PROMPT_FILE}"
      --output-dir "${GEN_OUTPUT_DIR}"
      --height "${HEIGHT}"
      --width "${WIDTH}"
      --num-frames "${NUM_FRAMES}"
      --frame-rate "${FPS}"
      --guidance-scale "${GUIDANCE_SCALE}"
      --num-inference-steps "${NUM_INFERENCE_STEPS}"
      --seed "${SEED}"
      --num-seeds "${NUM_SEEDS}"
      --seed-stride "${SEED_STRIDE}"
      --negative-prompt "${NEGATIVE_PROMPT}"
    )
    [[ -n "${MODEL_PATH}" ]] && GEN_CMD+=(--model-path "${MODEL_PATH}")
    [[ -n "${LTX_PIPELINE_CONFIG}" ]] && GEN_CMD+=(--pipeline-config "${LTX_PIPELINE_CONFIG}")
    ;;
  wan)
    GEN_CMD=(python "${REPO_ROOT}/physical-ai-bench/generation/generate_wan_lora.py"
      --checkpoint-dir "${CHECKPOINT_DIR}"
      --prompt-file "${PROMPT_FILE}"
      --output-dir "${GEN_OUTPUT_DIR}"
      --wan-task "${WAN_TASK}"
      --wan-size "${WAN_SIZE}"
      --num-frames "${NUM_FRAMES}"
      --guidance-scale "${GUIDANCE_SCALE}"
      --num-inference-steps "${NUM_INFERENCE_STEPS}"
      --sample-shift "${SAMPLE_SHIFT}"
      --sample-solver "${SAMPLE_SOLVER}"
      --fps "${WAN_FPS}"
      --seed "${SEED}"
      --num-seeds "${NUM_SEEDS}"
      --seed-stride "${SEED_STRIDE}"
      --negative-prompt "${NEGATIVE_PROMPT}"
    )
    [[ -n "${MODEL_PATH}" ]] && GEN_CMD+=(--model-path "${MODEL_PATH}")
    ;;
  *)
    echo "ERROR: Unsupported MODEL_TYPE=${MODEL_TYPE}. Use cogvideox, ltx, or wan." >&2
    exit 1
    ;;
esac

printf '%q ' "${GEN_CMD[@]}"
echo
"${GEN_CMD[@]}"

python "${REPO_ROOT}/physical-ai-bench/generation/generate_custom_vqa.py" \
  --prompt-file "${GEN_OUTPUT_DIR}/prompt_manifest.json" \
  --output-dir "${VQA_QUESTIONS_DIR}"

python "${REPO_ROOT}/physical-ai-bench/generation/evaluate_vqa.py" \
  --prompt_file "${GEN_OUTPUT_DIR}/prompt_manifest.json" \
  --vqa_questions_dir "${VQA_QUESTIONS_DIR}" \
  --video_dir "${GEN_OUTPUT_DIR}/videos" \
  --output_dir "${VQA_RESULTS_DIR}" \
  --model_name "${VQA_MODEL_NAME}" \
  --device "${VQA_DEVICE}" \
  --tensor_parallel_size "${TENSOR_PARALLEL_SIZE}"

echo
echo "======================================================================"
echo "CUSTOM PAI-BENCH EVALUATION COMPLETE"
echo "======================================================================"
echo "Generated videos: ${GEN_OUTPUT_DIR}/videos"
echo "Prompt manifest:  ${GEN_OUTPUT_DIR}/prompt_manifest.json"
echo "Custom VQA:       ${VQA_QUESTIONS_DIR}"
echo "VQA results:      ${VQA_RESULTS_DIR}"
