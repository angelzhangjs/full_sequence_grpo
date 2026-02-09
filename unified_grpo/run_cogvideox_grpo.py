#!/usr/bin/env python3
"""
Run Unified GRPO Training with CogVideoX
"""

import argparse
import torch
from pathlib import Path
from diffusers import CogVideoXPipeline

from unified_grpo.adapters.cogvideox_adapter import CogVideoXAdapter
from unified_grpo.grpo_core import run_grpo_for_prompt, GRPOConfig


def simple_reward_function(video: torch.Tensor, prompt: str) -> torch.Tensor:
    """
    Simple placeholder reward function
    
    Args:
        video: [B, F, C, H, W] in [0, 1]
        prompt: Text prompt
        
    Returns:
        reward: Scalar tensor
    """
    # Placeholder: random reward
    # Replace with your physics/motion/CLIP rewards!
    
    # Example: Simple variance-based reward
    # (Higher variance = more motion/detail)
    reward = video.var().item()
    
    return torch.tensor(reward, device=video.device)


def main():
    parser = argparse.ArgumentParser(description="CogVideoX Unified GRPO")
    
    # Model
    parser.add_argument(
        "--model_path",
        type=str,
        default="THUDM/CogVideoX-5b",
        help="HuggingFace model path"
    )
    
    # Prompt
    parser.add_argument(
        "--prompt",
        type=str,
        default="A ball bouncing up a staircase",
        help="Text prompt for generation"
    )
    
    # Video params
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--num_frames", type=int, default=49)
    
    # GRPO params
    parser.add_argument("--num_inference_steps", type=int, default=40)
    parser.add_argument("--num_grpo_steps", type=int, default=20)
    parser.add_argument("--num_rollouts", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    
    # Training
    parser.add_argument(
        "--train_blocks",
        type=str,
        default="22,23,24,25,26,27,28,29",  # Last 8 blocks (25%)
        help="Comma-separated block indices to train"
    )
    
    # Output
    parser.add_argument("--output_dir", type=str, default="./grpo_output")
    
    args = parser.parse_args()
    
    print("="*70)
    print("CogVideoX Unified GRPO Training")
    print("="*70)
    print(f"Model: {args.model_path}")
    print(f"Prompt: {args.prompt}")
    print(f"Resolution: {args.width}×{args.height}, {args.num_frames} frames")
    print(f"GRPO steps: {args.num_grpo_steps}")
    print(f"Rollouts: {args.num_rollouts}")
    print(f"Learning rate: {args.lr}")
    print()
    
    # ========================================================================
    # Load Pipeline
    # ========================================================================
    
    print("Loading CogVideoX pipeline...")
    pipeline = CogVideoXPipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
    ).to("cuda")
    
    # Memory optimizations
    pipeline.enable_model_cpu_offload()
    pipeline.vae.enable_slicing()
    pipeline.vae.enable_tiling()
    
    print("✅ Pipeline loaded\n")
    
    # ========================================================================
    # Encode Prompt
    # ========================================================================
    
    print("Encoding prompt...")
    prompt_embeds, negative_prompt_embeds = pipeline.encode_prompt(
        prompt=args.prompt,
        negative_prompt="",
        do_classifier_free_guidance=True,
        device="cuda",
    )
    print("✅ Prompt encoded\n")
    
    # ========================================================================
    # Create Adapter
    # ========================================================================
    
    # Parse train blocks
    train_blocks = [int(x.strip()) for x in args.train_blocks.split(",")]
    
    print(f"Creating adapter...")
    print(f"  Training blocks: {train_blocks}")
    
    adapter = CogVideoXAdapter(
        pipeline=pipeline,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        guidance_scale=6.0,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        train_transformer_blocks=train_blocks,
    )
    
    print(f"✅ Adapter created")
    print(f"  Trainable parameters: {len(adapter.trainable_parameters())}\n")
    
    # ========================================================================
    # GRPO Config
    # ========================================================================
    
    grpo_config = GRPOConfig(
        num_inference_steps=args.num_inference_steps,
        num_grpo_steps=args.num_grpo_steps,
        num_rollouts=args.num_rollouts,
        lr=args.lr,
    )
    
    # ========================================================================
    # Run GRPO
    # ========================================================================
    
    print("="*70)
    print("Running GRPO Training")
    print("="*70)
    
    output_dir = Path(args.output_dir)
    
    metrics = run_grpo_for_prompt(
        adapter=adapter,
        prompt=args.prompt,
        reward_fn=simple_reward_function,
        seed=args.seed,
        out_dir=output_dir,
        cfg=grpo_config,
    )
    
    print("\n" + "="*70)
    print("GRPO Training Complete!")
    print("="*70)
    print("Final metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    
    print(f"\nOutput saved to: {output_dir}")
    print("✅ Done!")


if __name__ == "__main__":
    main()
