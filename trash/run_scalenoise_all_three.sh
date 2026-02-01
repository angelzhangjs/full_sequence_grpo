cd /home/ubuntu/angel-research/full_sequence_grpo

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN_ID="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="scalingnoise_outputs/compare_${RUN_ID}"
#PROMPT_FILE="${OUT_DIR}/prompts.txt"
PROMPT_FILE_SRC="/home/ubuntu/angel-research/full_sequence_grpo/more_prompt.txt"

mkdir -p "${OUT_DIR}"

# Use all prompts from more_prompt.txt (one prompt per line).
#cp "${PROMPT_FILE_SRC}" "${PROMPT_FILE}"
PROMPT_FILE="${PROMPT_FILE_SRC}"

/home/ubuntu/anaconda3/bin/conda run -n scaling-noise python -u scaling-noise/scalenoise.py \
  --run_modes "baseline,beam,grpo" \
  --use_grpo_actions \
  --grpo_actions_all_partitions 0 \
  --grpo_action_lr 0.15 \
  --grpo_action_entropy_beta 0.01 \
  --grpo_action_normalize_adv 1 \
  --reward_clip_weight 0.6 \
  --reward_dino_weight 0.4 \
  --ckpt_path "/home/ubuntu/angel-research/full_sequence_grpo/base_512_v2/model.ckpt" \
  --config "/home/ubuntu/angel-research/full_sequence_grpo/scaling-noise/configs/inference_t2v_512_v2.0.yaml" \
  --prompt_file "${PROMPT_FILE}" \
  --output_dir "${OUT_DIR}" \
  --seed 26 \
  --video_length 16 \
  --num_partitions 4 \
  --new_video_length 80 \
  --fps 16 \
  --output_fps 16 \
  --eta 1.0 \
  --unconditional_guidance_scale 12.0 \
  --k 6 \
  --m 3 \
  --use_mp4 \
  --grpo_action_log_txt 1

