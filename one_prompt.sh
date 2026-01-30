cd /home/ubuntu/angel-research/full_sequence_grpo

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# GRPO_FROM_START=1
python baseline_intermediate_videos.py \
  --mode both \
  --prompt "A wooden cylinder rolls on the ground, slowing down over time due to friction." \
  --height 512 --width 768 --num_frames 81 --frame_rate 16 --seed 26 \
  --save_every 1 \
  --num_inference_steps 40 --num_grpo_steps 25 --num_rollouts 5 \
  --lr 2e-4 --attn1_blocks "14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27" --attn2_blocks "24, 25, 26, 27" \
  --rollout_noise_scale 0.5 \
  --negative_prompt "worst quality, inconsistent motion, blurry, jittery, distorted"
