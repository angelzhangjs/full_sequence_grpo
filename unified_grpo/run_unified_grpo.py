#!/usr/bin/env python3
"""
Unified GRPO Training Script
Supports multiple video models via --model-type argument
"""

import argparse
import torch
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

from unified_grpo.grpo_core import run_grpo_for_prompt, GRPOConfig

# ============================================================================
# Logging Setup - Redirect stdout to both console and file
# ============================================================================

class TeeLogger:
    """Writes to both console and log file simultaneously"""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', buffering=1)  # Line buffered
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.terminal.flush()
        self.log.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()

# Import comprehensive reward function from origin_grpo
import sys
# Use simplified reward function (CLIP + DINO only, no centering/motion)
from unified_grpo.reward_simple_clip_dino import comprehensive_grpo_reward
from diffusers import CogVideoXPipeline
from unified_grpo.adapters.cogvideox_adapter import CogVideoXAdapter
import clip
import torch

# Load once:
clip_model, _ = clip.load("ViT-B/32", device="cuda")
dino_model = torch.hub.load('facebookresearch/dinov2:main', 'dinov2_vitb14')


def reward_function_wrapper(video: torch.Tensor, prompt: str) -> torch.Tensor:
    """
    Wrapper for GRPO reward.

    NOTE: This runner currently uses the simplified reward in
    `unified_grpo/reward_simple_clip_dino.py` (CLIP + DINO consistency).
    """
    # Debug video properties
    print("\n  [DEBUG] Video properties:")
    print(f"    Shape: {video.shape}")
    print(f"    Dtype: {video.dtype}")
    print(f"    Device: {video.device}")
    print(f"    Range: [{video.min().item():.4f}, {video.max().item():.4f}]")
    print(f"    Mean: {video.mean().item():.4f}")
    print(f"    Non-zero elements: {(video.abs() > 0.001).sum().item()} / {video.numel()}")
    
    # comprehensive_grpo_reward returns dict with scores
    result = comprehensive_grpo_reward(
        video=video,
        prompt=prompt,
        device='cuda',
        use_clip=True,
        use_dino=True,
    )
    
    print(f"  [DEBUG] Reward result: {result}")
    
    # Extract total reward (key is 'reward', not 'total_reward')
    total_reward = result.get('reward', 0.0)
    
    print(f"  [DEBUG] Total reward: {total_reward:.6f}\n")
    
    return torch.tensor(total_reward, device=video.device)


def create_cogvideox_adapter(args):
    """Create CogVideoX adapter"""
    from unified_grpo.lora_utils import apply_lora_to_transformer
    
    print(f"Loading CogVideoX pipeline: {args.model_path}")
    pipeline = CogVideoXPipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
    ).to("cuda")
    
    # Ensure all components use consistent dtype
    pipeline.transformer = pipeline.transformer.to(torch.bfloat16)
    pipeline.vae = pipeline.vae.to(torch.bfloat16)
    pipeline.text_encoder = pipeline.text_encoder.to(torch.bfloat16)

    # Important for stable sampling/training: disable dropout-like noise.
    # `.eval()` does NOT disable gradients; it only changes module behavior (e.g. dropout/bn).
    pipeline.transformer.eval()
    pipeline.vae.eval()
    pipeline.text_encoder.eval()
    
    # Apply LoRA if requested (recommended for 40GB GPU!)
    if args.use_lora:
        # Parse lora_blocks if provided
        lora_blocks = None
        if args.lora_blocks:
            lora_blocks = [int(x.strip()) for x in args.lora_blocks.split(",")]
        
        pipeline.transformer, lora_params = apply_lora_to_transformer(
            pipeline.transformer,
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            target_blocks=lora_blocks,
        )
        if lora_blocks:
            print(f"✅ LoRA applied to blocks {lora_blocks} (ultra memory-efficient!)")
        else:
            print("✅ LoRA applied to ALL blocks (memory-efficient training!)")
    else:
        print("⚠️  Training unfrozen blocks (needs 48GB+ VRAM)")
    
    # Memory optimizations
    pipeline.vae.enable_slicing()
    pipeline.vae.enable_tiling()
    
    print("✅ Memory optimizations enabled")
    
    # Encode prompt
    prompt_embeds, negative_embeds = pipeline.encode_prompt(
        prompt=args.prompt,
        negative_prompt="",
        do_classifier_free_guidance=True,
        device="cuda",
    )
    
    # Parse train blocks
    train_blocks = None
    if args.train_blocks:
        train_blocks = [int(x.strip()) for x in args.train_blocks.split(",")]
    
    adapter = CogVideoXAdapter(
        pipeline=pipeline,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_embeds,
        guidance_scale=args.guidance_scale,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        train_transformer_blocks=train_blocks,
    )
    
    return adapter


def create_ltx_adapter(args):
    """Create LTX-Video adapter"""
    from diffusers import LTXVideoPipeline
    from unified_grpo.adapters.ltx_adapter import LTXAdapter
    
    print(f"Loading LTX-Video pipeline: {args.model_path}")
    pipeline = LTXVideoPipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
    ).to("cuda")
    
    pipeline.vae.enable_slicing()
    pipeline.vae.enable_tiling()
    
    # Encode prompt
    prompt_embeds_tuple = pipeline.encode_prompt(
        prompt=args.prompt,
        device="cuda",
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
    )
    prompt_embeds = prompt_embeds_tuple[0]
    negative_embeds = prompt_embeds_tuple[2]
    
    train_blocks = None
    if args.train_blocks:
        train_blocks = [int(x.strip()) for x in args.train_blocks.split(",")]
    
    adapter = LTXAdapter(
        pipeline=pipeline,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_embeds,
        guidance_scale=args.guidance_scale,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        train_transformer_blocks=train_blocks,
    )
    
    return adapter


def create_hunyuan_adapter(args):
    """Create HunyuanVideo adapter"""
    from diffusers import HunyuanVideoPipeline
    from unified_grpo.adapters.hunyuan_adapter import HunyuanAdapter
    
    print(f"Loading HunyuanVideo pipeline: {args.model_path}")
    pipeline = HunyuanVideoPipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
    ).to("cuda")
    
    pipeline.vae.enable_slicing()
    pipeline.vae.enable_tiling()
    
    # Encode prompt
    prompt_embeds = pipeline.encode_prompt(
        prompt=args.prompt,
        device="cuda",
    )
    
    train_blocks = None
    if args.train_blocks:
        train_blocks = [int(x.strip()) for x in args.train_blocks.split(",")]
    
    adapter = HunyuanAdapter(
        pipeline=pipeline,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=None,
        guidance_scale=args.guidance_scale,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        train_transformer_blocks=train_blocks,
    )
    
    return adapter


def create_wan_adapter(args):
    """Create Wan2.1 adapter"""
    import sys
    from pathlib import Path
    
    # Add Wan2.1 to path
    wan_path = Path(__file__).parent.parent.parent / "Wan2.1"
    sys.path.insert(0, str(wan_path))
    
    from wan.text2video import WanT2V
    from unified_grpo.adapters.wan_adapter import WanAdapter
    
    print(f"Loading Wan2.1 pipeline from Wan2.1 folder")
    print("Using custom Wan implementation")
    
    # Load Wan model
    wan_model = WanT2V()  # Will need config path
    
    # Create wrapper that looks like pipeline
    class WanPipelineWrapper:
        def __init__(self, model):
            self.model = model
            self.transformer = model  # For adapter compatibility
            # Add other attributes as needed
    
    pipeline = WanPipelineWrapper(wan_model)
    
    # For now, raise with better instructions
    raise NotImplementedError(
        "Wan adapter requires additional configuration. "
        "See: Wan2.1/generate.py for usage pattern. "
        "Will be completed when Wan pipeline wrapper is ready."
    )


def create_opensora_adapter(args):
    """Create Open-Sora adapter"""
    import sys
    from pathlib import Path

    # Add Open-Sora to path
    opensora_path = Path(__file__).parent.parent.parent / "Open-Sora"
    sys.path.insert(0, str(opensora_path))

    from unified_grpo.adapters.opensora_adapter import OpenSoraAdapter

    try:
        from mmengine.config import Config
        from opensora.utils.sampling import prepare_models
        from opensora.utils.misc import to_torch_dtype
    except Exception as e:
        raise RuntimeError(
            "Open-Sora dependencies not available. "
            "Install Open-Sora requirements and ensure `Open-Sora/` is importable."
        ) from e

    if not args.model_path:
        raise ValueError(
            "For --model-type opensora, please pass --model-path as an Open-Sora config file, "
            "e.g. Open-Sora/configs/diffusion/inference/256px.py"
        )

    cfg = Config.fromfile(args.model_path)
    device = "cuda"
    dtype = to_torch_dtype(getattr(cfg, "dtype", "bf16"))

    print(f"Loading Open-Sora models from config: {args.model_path}")
    model, model_ae, model_t5, model_clip, optional_models = prepare_models(
        cfg, device, dtype, offload_model=bool(getattr(cfg, "offload_model", False))
    )

    class OpenSoraPipelineWrapper:
        def __init__(self):
            self.model = model
            self.ae = model_ae
            self.t5 = model_t5
            self.clip = model_clip
            self.optional_models = optional_models
            self.device = torch.device(device)
            self.dtype = dtype

    pipeline = OpenSoraPipelineWrapper()

    train_blocks = None
    if args.train_blocks:
        train_blocks = [int(x.strip()) for x in args.train_blocks.split(",")]

    adapter = OpenSoraAdapter(
        pipeline=pipeline,
        prompt=args.prompt,
        negative_prompt="",
        guidance_scale=float(args.guidance_scale),
        height=int(args.height),
        width=int(args.width),
        num_frames=int(args.num_frames),
        train_transformer_blocks=train_blocks,
    )
    return adapter


def create_adapter(args):
    """Create appropriate adapter based on model type"""
    
    model_type = args.model_type.lower()
    
    if model_type == "cogvideox":
        return create_cogvideox_adapter(args)
    elif model_type == "ltx":
        return create_ltx_adapter(args)
    elif model_type == "hunyuan":
        return create_hunyuan_adapter(args)
    elif model_type == "wan":
        return create_wan_adapter(args)
    elif model_type == "opensora":
        return create_opensora_adapter(args)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose from: cogvideox, ltx, hunyuan, wan, opensora")


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
  
  # HunyuanVideo:
  python run_unified_grpo.py --model-type hunyuan --model-path Tencent/HunyuanVideo
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
        help="HuggingFace model path or local path"
    )
    
    # ========================================================================
    # Prompt
    # ========================================================================
    
    parser.add_argument(
        "--prompt",
        type=str,
        default="A ball bouncing up a staircase",
        help="Text prompt for generation"
    )
    
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
        help="Comma-separated block indices for LoRA (e.g., '29' or '27,28,29'). If not set, applies to ALL blocks."
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./grpo_output",
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    # ========================================================================
    # Set model-specific defaults
    # ========================================================================
    
    if args.model_path is None:
        # Set default model path based on type
        defaults = {
            "cogvideox": "THUDM/CogVideoX-5b",
            "ltx": "Lightricks/LTX-Video",
            "hunyuan": "tencent/HunyuanVideo",
            "wan": "Wan-AI/Wan2.1-T2V-1.3B",
            "opensora": "hpcaitech/Open-Sora-Plan-v1.1.0",
        }
        args.model_path = defaults.get(args.model_type, "THUDM/CogVideoX-5b")
    
    if args.train_blocks is None:
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
    
    output_dir = Path(args.output_dir) / args.model_type
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_dir / f"training_log_{timestamp}.txt"
    
    print(f"Training log: {log_file}")
    print(f"Output dir: {output_dir}")
    print(f"Prompt: {args.prompt}")
    
    # Redirect stdout to both console and log file
    logger = TeeLogger(str(log_file))
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
    
    print("Using reward function: CLIP alignment + DINO consistency (anti-collapse terms enabled in reward)\n")
    
    try:
        metrics = run_grpo_for_prompt(
            adapter=adapter,
            prompt=args.prompt,
            reward_fn=reward_function_wrapper,
            seed=args.seed,
            out_dir=output_dir,
            cfg=grpo_config,
        )
        
        # ========================================================================
        # Results
        # ========================================================================
        
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
