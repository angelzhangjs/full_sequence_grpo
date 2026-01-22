cd /home/ubuntu/angel-research/full_sequence_grpo

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
GRPO_FROM_START=0 
python baseline_intermediate_videos.py \
  --mode both \
  --prompt "A bright red ball falling and bouncing down a wooden staircase, moving downward and hitting each lower step sequentially, descending to the bottom, vibrant color" \
  --pipeline_config configs/ltxv-2b-0.9.6-dev.yaml \
  --height 512 --width 768 --num_frames 81 --frame_rate 16 --seed 26 \
  --save_every 1 \
  --num_inference_steps 40 --num_grpo_steps 25 --num_rollouts 3 \
  --lr 1e-4 --attn1_blocks "13,14" --attn2_blocks "27" \
  --rollout_noise_scale 0.5 \
  --negative_prompt "worst quality, inconsistent motion, blurry, jittery, distorted" \
  --output_dir baseline_and_grpo_ball_single_prompt_2 