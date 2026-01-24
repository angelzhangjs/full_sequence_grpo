cd /home/ubuntu/angel-research/full_sequence_grpo

python run_prompts_sequential.py \
  --prompt-file prompt.txt \
  --mode both \
  --pipeline-config configs/ltxv-2b-0.9.6-dev.yaml \
  --height 512 --width 768 --num-frames 81 --frame-rate 16 --seed 26 \
  --save-every 1 \
  --num-inference-steps 40 --num-grpo-steps 25 --num-rollouts 3 \
  --lr 1e-4 --attn1_blocks "14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27" --attn2-blocks "27" \
  --rollout-noise-scale 0.5