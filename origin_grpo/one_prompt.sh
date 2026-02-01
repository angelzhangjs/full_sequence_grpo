set -euo pipefail

cd /home/ubuntu/angel-research/full_sequence_grpo

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# RUN_ID="$(date +%Y%m%d_%H%M%S)"
# OUT_DIR="grpo_ltx_${RUN_ID}"
# PROMPT_FILE_SRC="/home/ubuntu/angel-research/full_sequence_grpo/prompt.txt"
# PROMPT_FILE="${OUT_DIR}/prompts.txt"

# mkdir -p "${OUT_DIR}"

# # Copy prompts into the run folder for reproducibility.
# cp "${PROMPT_FILE_SRC}" "${PROMPT_FILE}"

# Run ALL prompts (one per line; empty lines and lines starting with '#' are ignored by grpo_modular_pipeline.py)
python origin_grpo/grpo_modular_pipeline.py \
  --mode both \
  --prompt "A wooden cylinder rolls on the ground, slowing down over time due to friction." \
  --output_dir "${OUT_DIR}" \
  --no_timestamp \
  --height 512 --width 768 --num_frames 81 --frame_rate 16 --seed 26 \
  --save_every 1 \
  --num_inference_steps 40 --num_grpo_steps 25 --num_rollouts 5 \
  --lr 2e-4 --attn1_blocks "14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27" --attn2_blocks "24, 25, 26, 27" \
  --rollout_noise_scale 0.5 \
  --negative_prompt "worst quality, inconsistent motion, blurry, jittery, distorted"
