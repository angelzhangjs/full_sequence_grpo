#!/usr/bin/env bash
set -euo pipefail

cd /home/ubuntu/angel-research/full_sequence_grpo

# One wrapper to run the unified GRPO runner for different backends.
#
# Usage examples:
#
#   # Wan2.1 (requires ckpt dir)
#   BACKEND=wan21 WAN_CKPT_DIR="/abs/path/to/Wan2.1-T2V-1.3B" \
#     PROMPT_FILE="origin_grpo/physical_plausibility.txt" \
#     ./run_unified_grpo.sh
#
#   # Open-Sora (requires Open-Sora config + Open-Sora deps installed)
#   BACKEND=opensora OPENSORA_CONFIG="Open-Sora/configs/diffusion/inference/256px.py" \
#     PROMPT_FILE="origin_grpo/physical_plausibility.txt" \
#     ./run_unified_grpo.sh

: "${BACKEND:=wan21}"                 # wan21 | opensora
: "${PROMPT_FILE:=origin_grpo/physical_plausibility.txt}"
: "${OUT_DIR:=unified_runs}"
: "${SEED:=26}"

# Common GRPO knobs
: "${NUM_INFERENCE_STEPS:=40}"
: "${NUM_GRPO_STEPS:=25}"
: "${NUM_ROLLOUTS:=3}"
: "${ROLLOUT_NOISE_SCALE:=0.5}"
: "${LR:=1e-5}"
: "${GRAD_CLIP:=1.0}"
: "${LOGPROB_SIGMA:=1.0}"

# Common video shape (some backends override via their own args)
: "${HEIGHT:=512}"
: "${WIDTH:=768}"
: "${NUM_FRAMES:=81}"

case "${BACKEND}" in
  wan21)
    : "${WAN_CKPT_DIR:=}"
    : "${WAN_TASK:=t2v-1.3B}"           # t2v-1.3B | t2v-14B | ...
    : "${WAN_SIZE:=832*480}"           # 832*480 | 480*832 | ...
    : "${WAN_SHIFT:=8.0}"
    : "${WAN_SOLVER:=unipc}"           # unipc | dpm++
    : "${WAN_T5_CPU:=1}"
    : "${DEVICE_ID:=0}"
    : "${GUIDANCE_SCALE:=6.0}"

    if [[ -z "${WAN_CKPT_DIR}" ]]; then
      echo "ERROR: WAN_CKPT_DIR is not set."
      exit 2
    fi

    python unified_grpo/run_unified_grpo.py \
      --backend wan21 \
      --prompt_file "${PROMPT_FILE}" \
      --out_dir "${OUT_DIR}" \
      --seed "${SEED}" \
      --num_inference_steps "${NUM_INFERENCE_STEPS}" \
      --num_grpo_steps "${NUM_GRPO_STEPS}" \
      --num_rollouts "${NUM_ROLLOUTS}" \
      --rollout_noise_scale "${ROLLOUT_NOISE_SCALE}" \
      --lr "${LR}" \
      --grad_clip "${GRAD_CLIP}" \
      --logprob_sigma "${LOGPROB_SIGMA}" \
      --num_frames "${NUM_FRAMES}" \
      --wan_ckpt_dir "${WAN_CKPT_DIR}" \
      --wan_task "${WAN_TASK}" \
      --wan_size "${WAN_SIZE}" \
      --wan_shift "${WAN_SHIFT}" \
      --wan_solver "${WAN_SOLVER}" \
      --wan_t5_cpu "${WAN_T5_CPU}" \
      --device_id "${DEVICE_ID}" \
      --guidance_scale "${GUIDANCE_SCALE}"
    ;;

  opensora)
    : "${OPENSORA_ROOT:=Open-Sora}"
    : "${OPENSORA_CONFIG:=}"
    : "${OPENSORA_OFFLOAD:=0}"
    : "${DTYPE:=bf16}"
    : "${GUIDANCE_SCALE:=4.0}"
    : "${OPENSORA_GUIDANCE_IMG:=1.0}"
    : "${OPENSORA_SHIFT:=1}"
    : "${OPENSORA_FLOW_SHIFT:=}"
    : "${OPENSORA_PATCH_SIZE:=2}"
    : "${OPENSORA_CHANNEL:=16}"
    : "${OPENSORA_TEMPORAL_REDUCTION:=1}"
    : "${OPENSORA_IS_CAUSAL_VAE:=0}"

    if [[ -z "${OPENSORA_CONFIG}" ]]; then
      echo "ERROR: OPENSORA_CONFIG is not set."
      exit 2
    fi

    EXTRA=()
    if [[ -n "${OPENSORA_FLOW_SHIFT}" ]]; then
      EXTRA+=(--opensora_flow_shift "${OPENSORA_FLOW_SHIFT}")
    fi

    python unified_grpo/run_unified_grpo.py \
      --backend opensora \
      --prompt_file "${PROMPT_FILE}" \
      --out_dir "${OUT_DIR}" \
      --seed "${SEED}" \
      --height "${HEIGHT}" --width "${WIDTH}" --num_frames "${NUM_FRAMES}" \
      --num_inference_steps "${NUM_INFERENCE_STEPS}" \
      --num_grpo_steps "${NUM_GRPO_STEPS}" \
      --num_rollouts "${NUM_ROLLOUTS}" \
      --rollout_noise_scale "${ROLLOUT_NOISE_SCALE}" \
      --lr "${LR}" \
      --grad_clip "${GRAD_CLIP}" \
      --logprob_sigma "${LOGPROB_SIGMA}" \
      --guidance_scale "${GUIDANCE_SCALE}" \
      --opensora_root "${OPENSORA_ROOT}" \
      --opensora_config "${OPENSORA_CONFIG}" \
      --opensora_offload "${OPENSORA_OFFLOAD}" \
      --dtype "${DTYPE}" \
      --opensora_guidance_img "${OPENSORA_GUIDANCE_IMG}" \
      --opensora_shift "${OPENSORA_SHIFT}" \
      --opensora_patch_size "${OPENSORA_PATCH_SIZE}" \
      --opensora_channel "${OPENSORA_CHANNEL}" \
      --opensora_temporal_reduction "${OPENSORA_TEMPORAL_REDUCTION}" \
      --opensora_is_causal_vae "${OPENSORA_IS_CAUSAL_VAE}" \
      "${EXTRA[@]}"
    ;;

  cogvideox)
    : "${COG_MODEL_PATH:=zai-org/CogVideoX-2b}"
    : "${COG_DTYPE:=bf16}"              # bf16 | fp16
    : "${GUIDANCE_SCALE:=6.0}"

    python unified_grpo/run_unified_grpo.py \
      --backend cogvideox \
      --prompt_file "${PROMPT_FILE}" \
      --out_dir "${OUT_DIR}" \
      --seed "${SEED}" \
      --height "${HEIGHT}" --width "${WIDTH}" --num_frames "${NUM_FRAMES}" \
      --num_inference_steps "${NUM_INFERENCE_STEPS}" \
      --num_grpo_steps "${NUM_GRPO_STEPS}" \
      --num_rollouts "${NUM_ROLLOUTS}" \
      --rollout_noise_scale "${ROLLOUT_NOISE_SCALE}" \
      --lr "${LR}" \
      --grad_clip "${GRAD_CLIP}" \
      --logprob_sigma "${LOGPROB_SIGMA}" \
      --guidance_scale "${GUIDANCE_SCALE}" \
      --cog_model_path "${COG_MODEL_PATH}" \
      --cog_dtype "${COG_DTYPE}"
    ;;

  hunyuan15)
    : "${HUNYUAN_MODEL_BASE:=}"
    : "${HUNYUAN_CPU_OFFLOAD:=0}"
    : "${HUNYUAN_FLOW_SHIFT:=7.0}"
    : "${HUNYUAN_FLOW_SOLVER:=euler}"
    : "${HUNYUAN_FLOW_REVERSE:=0}"
    : "${HUNYUAN_EMBEDDED_CFG_SCALE:=}"
    : "${GUIDANCE_SCALE:=6.0}"

    if [[ -z "${HUNYUAN_MODEL_BASE}" ]]; then
      echo "ERROR: HUNYUAN_MODEL_BASE is not set."
      exit 2
    fi

    EXTRA=()
    if [[ -n "${HUNYUAN_EMBEDDED_CFG_SCALE}" ]]; then
      EXTRA+=(--hunyuan_embedded_cfg_scale "${HUNYUAN_EMBEDDED_CFG_SCALE}")
    fi

    python unified_grpo/run_unified_grpo.py \
      --backend hunyuan15 \
      --prompt_file "${PROMPT_FILE}" \
      --out_dir "${OUT_DIR}" \
      --seed "${SEED}" \
      --height "${HEIGHT}" --width "${WIDTH}" --num_frames "${NUM_FRAMES}" \
      --num_inference_steps "${NUM_INFERENCE_STEPS}" \
      --num_grpo_steps "${NUM_GRPO_STEPS}" \
      --num_rollouts "${NUM_ROLLOUTS}" \
      --rollout_noise_scale "${ROLLOUT_NOISE_SCALE}" \
      --lr "${LR}" \
      --grad_clip "${GRAD_CLIP}" \
      --logprob_sigma "${LOGPROB_SIGMA}" \
      --guidance_scale "${GUIDANCE_SCALE}" \
      --hunyuan_model_base "${HUNYUAN_MODEL_BASE}" \
      --hunyuan_cpu_offload "${HUNYUAN_CPU_OFFLOAD}" \
      --hunyuan_flow_shift "${HUNYUAN_FLOW_SHIFT}" \
      --hunyuan_flow_solver "${HUNYUAN_FLOW_SOLVER}" \
      --hunyuan_flow_reverse "${HUNYUAN_FLOW_REVERSE}" \
      "${EXTRA[@]}"
    ;;

  ltx)
    : "${LTX_PIPELINE_CONFIG:=configs/ltxv-2b-0.9.6-dev.yaml}"
    : "${LTX_CKPT_PATH:=}"
    : "${LTX_FRAME_RATE:=16}"
    : "${GUIDANCE_SCALE:=3.0}"

    EXTRA=()
    if [[ -n "${LTX_CKPT_PATH}" ]]; then
      EXTRA+=(--ltx_ckpt_path "${LTX_CKPT_PATH}")
    fi

    python unified_grpo/run_unified_grpo.py \
      --backend ltx \
      --prompt_file "${PROMPT_FILE}" \
      --out_dir "${OUT_DIR}" \
      --seed "${SEED}" \
      --height "${HEIGHT}" --width "${WIDTH}" --num_frames "${NUM_FRAMES}" \
      --num_inference_steps "${NUM_INFERENCE_STEPS}" \
      --num_grpo_steps "${NUM_GRPO_STEPS}" \
      --num_rollouts "${NUM_ROLLOUTS}" \
      --rollout_noise_scale "${ROLLOUT_NOISE_SCALE}" \
      --lr "${LR}" \
      --grad_clip "${GRAD_CLIP}" \
      --logprob_sigma "${LOGPROB_SIGMA}" \
      --guidance_scale "${GUIDANCE_SCALE}" \
      --ltx_pipeline_config "${LTX_PIPELINE_CONFIG}" \
      --ltx_frame_rate "${LTX_FRAME_RATE}" \
      "${EXTRA[@]}"
    ;;

  *)
    echo "ERROR: Unsupported BACKEND=${BACKEND}. Choose wan21, opensora, cogvideox, hunyuan15, or ltx."
    exit 2
    ;;
esac

