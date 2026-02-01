cd /home/ubuntu/angel-research/full_sequence_grpo

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN_ID="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="scalingnoise_outputs/grpo_search_${RUN_ID}"
PROMPT_FILE="${OUT_DIR}/prompt.txt"

mkdir -p "${OUT_DIR}"
printf "%s\n" "Flocks of birds spiral upwards in synchronized arcs, weaving around the rooftops before scattering into the open sky." > "${PROMPT_FILE}"

/home/ubuntu/anaconda3/bin/conda run -n scaling-noise python -u scaling-noise/scalenoise.py \
  --search_mode grpo \
  --use_grpo_actions \
  --grpo_action_lr 0.1 \
  --grpo_action_entropy_beta 0.01 \
  --grpo_action_normalize_adv 1 \
  --ckpt_path "/home/ubuntu/angel-research/full_sequence_grpo/base_512_v2/model.ckpt" \
  --config "/home/ubuntu/angel-research/full_sequence_grpo/scaling-noise/configs/inference_t2v_512_v2.0.yaml" \
  --prompt_file "${PROMPT_FILE}" \
  --output_dir "${OUT_DIR}" \
  --seed 2026 \
  --video_length 16 \
  --num_partitions 4 \
  --new_video_length 64 \
  --fps 8 \
  --output_fps 8 \
  --eta 1.0 \
  --unconditional_guidance_scale 12.0 \
  --k 3 \
  --m 1 \
  --use_mp4 \
  --grpo_actions_all_partitions 1
