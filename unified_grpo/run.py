#!/usr/bin/env python3
"""
Unified GRPO Training Script
Supports multiple video models via --model-type argument
"""

import argparse
import torch
import sys
from pathlib import Path
from datetime import datetime

from unified_grpo.grpo_core import run_grpo_for_prompt, GRPOConfig
from unified_grpo.utils import WriteLogger

# Reward backends are optional-dependency-safe:
# - CLIP+DINO reward lazy-loads CLIP/DINO internally
# - Qwen reward lazy-loads transformers + qwen-vl-utils internally
from unified_grpo.reward_simple_clip_dino import comprehensive_grpo_reward
from unified_grpo.create_adapter import create_cogvideox_adapter, create_ltx_adapter

def create_adapter(args):
    """Create appropriate adapter based on model type"""
    model_type = args.model_type.lower()
    
    if model_type == "cogvideox":
        return create_cogvideox_adapter(args)
    elif model_type == "ltx":
        return create_ltx_adapter(args)
    # # elif model_type == "hunyuan":
    # #     return create_hunyuan_adapter(args)
    # elif model_type == "wan":
    #     return create_wan_adapter(args)
    # elif model_type == "opensora":
    #     return create_opensora_adapter(args)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose from: cogvideox, ltx, wan, opensora")

def main():
    parser = argparse.ArgumentParser(
        description="Unified GRPO Training for Multiple Video Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # CogVideoX:
  python run_unified_grpo.py --model-type cogvideox --model-path THUDM/CogVideoX-5b
  
  # LTX-Video:
  python run_unified_grpo.py --model-type ltx --model-path Lightricks/LTX-Video
  
"""
    )
    
    # ========================================================================
    # Model Selection
    # ========================================================================
    
    parser.add_argument(
        "--model-type",
        type=str,
        required=True,
        choices=["cogvideox", "ltx", "hunyuan", "wan", "opensora"],
        help="Type of video model to use"
    )
    
    parser.add_argument(
        "--model-path",
        type=str,
        help="Model identifier/path. For WAN, this can be a local checkpoint directory OR a HuggingFace model id (auto-downloaded)."
    )

    # # Open-Sora-specific
    # parser.add_argument(
    #     "--opensora-root",
    #     type=str,
    #     default=None,
    #     help="Optional path to Open-Sora repo root (the folder containing `opensora/`). If unset, we auto-discover by searching parent dirs.",
    # )
    # parser.add_argument(
    #     "--opensora-config",
    #     type=str,
    #     default=None,
    #     help="Open-Sora config path (e.g. Open-Sora/configs/opensora-v1-1/inference/sample.py). If not set, we pick a default based on checkpoint type.",
    # )
    
    # ========================================================================
    # Prompt
    # ========================================================================
    
    parser.add_argument(
        "--prompt",
        type=str,
        default="A ball bouncing up a staircase",
        help="Text prompt for generation"
    )

    parser.add_argument(
        "--negative-prompt",
        type=str,
        default="worst quality, inconsistent motion, blurry, jittery, distorted",
        help="Negative prompt for classifier-free guidance (used by models that support it, e.g. LTX/CogVideoX).",
    )

    # ========================================================================
    # Reward configuration
    # ========================================================================

    parser.add_argument(
        "--reward-backend",
        type=str,
        default="clip_dino",
        choices=["clip_dino", "qwen"],
        help="Reward backend for GRPO. 'clip_dino' is fast; 'qwen' is slower but can score physics/temporal plausibility.",
    )
    parser.add_argument(
        "--reward-debug",
        action="store_true",
        default=False,
        help="Print extra reward debug info (can be very noisy).",
    )

    # Qwen reward options (used when --reward-backend qwen)
    parser.add_argument("--qwen-model-id", type=str, default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--qwen-num-frames", type=int, default=8)
    parser.add_argument("--qwen-max-new-tokens", type=int, default=192)
    parser.add_argument("--qwen-temperature", type=float, default=0.0)
    parser.add_argument("--qwen-w-align", type=float, default=0.5)
    parser.add_argument("--qwen-w-physics", type=float, default=0.3)
    # New name + backward-compatible alias
    parser.add_argument("--qwen-w-dynamic-motion", dest="qwen_w_dynamic_motion", type=float, default=0.2)
    parser.add_argument("--qwen-w-temporal", dest="qwen_w_dynamic_motion", type=float, default=0.2, help="Alias for --qwen-w-dynamic-motion")
    
    # ========================================================================
    # Video Parameters
    # ========================================================================
    
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--guidance-scale", type=float, default=6.0)
    
    # ========================================================================
    # GRPO Parameters
    # ========================================================================
    
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=40,
        help="Total denoising steps"
    )
    
    parser.add_argument(
        "--num-grpo-steps",
        type=int,
        default=20,
        help="Last N steps to apply GRPO"
    )
    
    parser.add_argument(
        "--num-rollouts",
        type=int,
        default=3,
        help="Number of rollouts per GRPO step"
    )
    
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="Learning rate"
    )
    
    parser.add_argument("--seed", type=int, default=42)

    # ========================================================================
    # Memory / performance toggles
    # ========================================================================

    parser.add_argument(
        "--compute-dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16"],
        help="Compute dtype for supported models (currently used by LTX). fp16 uses less VRAM but can be less stable.",
    )

    parser.add_argument(
        "--enable-xformers",
        action="store_true",
        default=False,
        help="Enable xFormers memory-efficient attention (can reduce VRAM a lot). Requires xformers installed.",
    )

    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        default=False,
        help="Enable gradient checkpointing on the model transformer to reduce VRAM (slower).",
    )
    
    # ========================================================================
    # Training Configuration
    # ========================================================================
    
    parser.add_argument(
        "--train-blocks",
        type=str,
        default=None,
        help="Comma-separated block indices (e.g., '22,23,24,25,26,27,28,29'). If not specified, uses --unfreeze-percentage."
    )
    
    parser.add_argument(
        "--unfreeze-percentage",
        type=float,
        default=0.25,
        help="Percentage of blocks to unfreeze from the END (0.0-1.0). Default: 0.25 (last 25%% of blocks). Applied across all models. Ignored if --train-blocks is specified."
    )
    
    # LoRA Configuration (recommended for 40GB GPU!)
    parser.add_argument(
        "--use-lora",
        action="store_true",
        default=False,
        help="Use LoRA for parameter-efficient training (recommended for 40GB GPU!)"
    )
    
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=16,
        help="LoRA rank: 8=minimal, 16=balanced, 32=large"
    )
    
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha (typically 2×rank)"
    )
    
    parser.add_argument(
        "--lora-blocks",
        type=str,
        default=None,
        help="LoRA block selection. Examples: 'all' (default), 'last' (last --unfreeze-percentage), or '27,28,29'. If not set, applies to ALL blocks."
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./grpo_output",
        help="Output directory. If left as default (./grpo_output), we create a run-specific dir like `wan_grpo_YYYYMMDD_HHMMSS`."
    )
    
    args = parser.parse_args()
    
    # ========================================================================
    # Set model-specific defaults
    # ========================================================================
    
    if args.model_path is None:
        # Set default model path based on type
        defaults = {
            "cogvideox": "THUDM/CogVideoX-2b",
            "ltx": "Lightricks/LTX-Video", 
            "wan": "Wan-AI/Wan2.1-T2V-1.3B",
            # Open-Sora is resolved inside create_opensora_adapter() (robust NFS-safe resolution).
            "opensora": None,
        }
        if args.model_type != "opensora":
            args.model_path = defaults.get(args.model_type, "THUDM/CogVideoX-2b")
    
    # Auto-select train blocks for *full finetune* mode only.
    # When using LoRA, base weights stay frozen, so auto-selecting `train_blocks` would be confusing and can
    # accidentally bleed into LoRA selection logic.
    if (args.train_blocks is None) and (not bool(getattr(args, "use_lora", False))):
        # Auto-select blocks based on percentage (unified across all models)
        print(f"Auto-selecting blocks: unfreezing last {args.unfreeze_percentage:.1%} of transformer blocks")
        
        # Model-specific total block counts
        model_total_blocks = {
            "cogvideox": 30,
            "ltx": 28,
            "hunyuan": 32,
            "wan": 24,
            "opensora": 28,
        }
        
        total_blocks = model_total_blocks.get(args.model_type, 28)
        num_blocks_to_train = max(1, int(total_blocks * args.unfreeze_percentage))
        start_block = total_blocks - num_blocks_to_train
        
        # Generate block indices
        blocks = list(range(start_block, total_blocks))
        args.train_blocks = ",".join(str(b) for b in blocks)
        
        print(f"  Model: {args.model_type} ({total_blocks} total blocks)")
        print(f"  Unfreezing: {num_blocks_to_train} blocks (last {args.unfreeze_percentage:.0%})")
        print(f"  Block indices: {args.train_blocks}\n")
    
    # ========================================================================
    # Display Configuration
    # ========================================================================
    
    print("="*70)
    print("UNIFIED GRPO TRAINING")
    print("="*70)
    print(f"Model Type: {args.model_type}")
    print(f"Model Path: {args.model_path}")
    print(f"Prompt: {args.prompt}")
    print(f"Resolution: {args.width}×{args.height}, {args.num_frames} frames")
    print(f"Inference steps: {args.num_inference_steps}")
    print(f"GRPO steps: {args.num_grpo_steps}")
    print(f"Rollouts: {args.num_rollouts}")
    print(f"Learning rate: {args.lr}")
    print(f"Training blocks: {args.train_blocks}")
    print()
    
    # ========================================================================
    # Create Adapter
    # ========================================================================
    
    print("Creating adapter...")
    adapter = create_adapter(args)
     
    print(f"✅ {args.model_type.upper()} adapter created")
    print(f"  Trainable parameters: {len(adapter.trainable_parameters())}")
    print()
    
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
    print("RUNNING GRPO")
    print("="*70)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Output directory:
    # - If user leaves default (--output-dir ./grpo_output), create a run-specific folder like `wan_grpo_YYYYMMDD_HHMMSS`
    # - If user explicitly sets --output-dir, respect it (and do NOT force nesting by model-type)
    if args.output_dir in (None, "./grpo_output", "grpo_output"):
        output_dir = Path(f"{args.model_type}_grpo_{timestamp}")
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging to file
    log_file = output_dir / f"training_log_{timestamp}.txt"
    
    print(f"Training log: {log_file}")
    print(f"Output dir: {output_dir}")
    print(f"Prompt: {args.prompt}")
    
    # Redirect stdout to both console and log file
    logger = WriteLogger(str(log_file))
    sys.stdout = logger
    sys.stderr = logger  # Also capture errors
    
    print("="*70)
    print(f"UNIFIED GRPO TRAINING - {args.model_type.upper()}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    print(f"\nPrompt: {args.prompt}")
    print(f"Model: {args.model_type} ({args.model_path})")
    print(f"Resolution: {args.width}×{args.height}, {args.num_frames} frames")
    print(f"GRPO: {args.num_grpo_steps} steps, {args.num_rollouts} rollouts")
    print(f"Learning rate: {args.lr}")
    print(f"Seed: {args.seed}")
    if args.use_lora:
        print(f"LoRA: rank={args.lora_rank}, alpha={args.lora_alpha}")
    else:
        print(f"Traditional unfreezing: blocks {args.train_blocks}")
    print()
    
    # Build reward_fn closure based on CLI.
    def reward_fn(video: torch.Tensor, prompt: str) -> torch.Tensor:
        if str(args.reward_backend).lower() == "qwen":
            from unified_grpo.reward_qwen import QwenRewardConfig, qwen_video_reward

            scores = qwen_video_reward(
                video=video,
                prompt=prompt,
                device=torch.device(video.device),
                cfg=QwenRewardConfig(
                    model_id=str(args.qwen_model_id),
                    num_sampled_frames=int(args.qwen_num_frames),
                    max_new_tokens=int(args.qwen_max_new_tokens),
                    temperature=float(args.qwen_temperature),
                    w_align=float(args.qwen_w_align),
                    w_physics=float(args.qwen_w_physics),
                    w_dynamic_motion=float(args.qwen_w_dynamic_motion),
                ),
            )
            if bool(args.reward_debug):
                print(f"  [QWEN REWARD] {scores}")
            return torch.tensor(float(scores["reward"]), device=video.device)

        # Default: CLIP+DINO
        if bool(args.reward_debug):
            print("\n  [DEBUG] Video properties:")
            print(f"    Shape: {video.shape}")
            print(f"    Dtype: {video.dtype}")
            print(f"    Device: {video.device}")
            print(f"    Range: [{video.min().item():.4f}, {video.max().item():.4f}]")
            print(f"    Mean: {video.mean().item():.4f}")
            print(f"    Non-zero elements: {(video.abs() > 0.001).sum().item()} / {video.numel()}")

        # Adapters return per-frame decoded pixels for reward: [T,3,H,W] in [0,1].
        result = comprehensive_grpo_reward(
            frames=video,
            prompt=prompt,
            device="cuda",
            use_clip=True,
            use_dino=True,
        )
        total_reward = float(result.get("reward", 0.0))
        if bool(args.reward_debug):
            print(f"  [DEBUG] Reward result: {result}")
            print(f"  [DEBUG] Total reward: {total_reward:.6f}\n")
        return torch.tensor(total_reward, device=video.device)

    if str(args.reward_backend).lower() == "qwen":
        print(
            "Using reward function: Qwen2-VL (text alignment + physical plausibility + dynamic motion consistency)\n"
        )
    else:
        print(
            "Using reward function: CLIP alignment (+ optional DINO consistency)\n"
        )
    
    try:
        metrics = run_grpo_for_prompt(
            adapter=adapter,
            prompt=args.prompt,
            reward_fn=reward_fn,
            seed=args.seed,
            out_dir=output_dir,
            cfg=grpo_config,
            model_type=args.model_type,
        )

        print("\n" + "="*70)
        print("TRAINING COMPLETE!")
        print("="*70)
        print(f"Model: {args.model_type}")
        print("Metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

        print(f"\nOutput: {output_dir}")
        print("✅ Done!")
        
    finally:
        # Close logger and restore stdout
        if 'logger' in locals():
            logger.close()
            sys.stdout = logger.terminal
            sys.stderr = sys.__stderr__
        
        print(f"\n✅ Training complete! Log saved to: {log_file}")

if __name__ == "__main__":
    main()
