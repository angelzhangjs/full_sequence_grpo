#!/usr/bin/env python3
"""
Unified GRPO Training Script
Supports multiple video models via --model-type argument
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from functools import partial

import clip
import torch
import torch.distributed as dist
from diffusers import CogVideoXPipeline
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from unified_grpo.adapters.cogvideox_adapter import CogVideoXAdapter
from unified_grpo.grpo_core import GRPOConfig, run_grpo_for_prompt
from unified_grpo.reward_simple_clip_dino import comprehensive_grpo_reward

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

# Load once:
clip_model, _ = clip.load("ViT-B/32", device="cuda")
dino_model = torch.hub.load('facebookresearch/dinov2:main', 'dinov2_vitb14')


def maybe_enable_gradient_checkpointing(module, label: str = "module"):
    """
    Best-effort gradient checkpointing toggle.
    Tries the common Diffusers/HF API, otherwise sets a flag if present.
    """
    if module is None:
        print(f"⚠️  Gradient checkpointing skipped: {label} is None")
        return
    fn = getattr(module, "enable_gradient_checkpointing", None)
    if callable(fn):
        fn()
        print(f"✅ Gradient checkpointing enabled on {label}")
        return
    if hasattr(module, "gradient_checkpointing"):
        setattr(module, "gradient_checkpointing", True)
        print(f"✅ Gradient checkpointing flag set on {label}")
    else:
        print(f"⚠️  Gradient checkpointing not supported on {label}")


def maybe_wrap_fsdp(module, args, label: str = "module"):
    """
    Wrap the given module with FSDP full-shard if requested and dist is initialized.
    Returns the (possibly wrapped) module.
    """
    if not getattr(args, "fsdp", False):
        return module

    if not dist.is_available():
        print(f"⚠️  FSDP requested but torch.distributed not available; skipping for {label}")
        return module

    if not dist.is_initialized():
        # Attempt best-effort init if environment is set (torchrun sets these).
        master_addr = os.environ.get("MASTER_ADDR")
        master_port = os.environ.get("MASTER_PORT")
        world_size = os.environ.get("WORLD_SIZE")
        rank = os.environ.get("RANK")
        if all([master_addr, master_port, world_size, rank]):
            try:
                dist.init_process_group(backend="nccl")
            except Exception as e:
                print(f"⚠️  FSDP requested but failed to init process group: {e}")
                return module
        else:
            print("⚠️  FSDP requested but process group not initialized. Launch with torchrun.")
            return module

    # Ensure we use the local rank device
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    # Auto-wrap transformer blocks if present
    layer_cls = None
    blk = getattr(module, "transformer_blocks", None)
    if blk and len(blk) > 0:
        layer_cls = {type(blk[0])}
    auto_wrap = partial(transformer_auto_wrap_policy, transformer_layer_cls=layer_cls) if layer_cls else None

    # Mixed precision policy aligned with --amp
    mp_policy = None
    if getattr(args, "amp", "none") == "bf16":
        mp_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
    elif getattr(args, "amp", "none") == "fp16":
        mp_policy = MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16,
        )

    try:
        wrapped = FSDP(
            module,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            auto_wrap_policy=auto_wrap,
            mixed_precision=mp_policy,
            device_id=torch.cuda.current_device(),
            use_orig_params=True,
        )
        print(f"✅ FSDP full-shard enabled on {label} (amp={getattr(args, 'amp', 'none')})")
        return wrapped
    except Exception as e:
        print(f"⚠️  Failed to wrap {label} in FSDP: {e}")
        return module


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

    # If FSDP: keep on CPU during load to reduce peak, then wrap and move.
    if args.fsdp:
        if not dist.is_initialized():
            dist.init_process_group("nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)

        pipeline = CogVideoXPipeline.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map=None,
            low_cpu_mem_usage=True,
        )

        tr = pipeline.transformer
        auto_wrap = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={type(tr.transformer_blocks[0])},
        )
        mp_policy = None
        if args.amp == "bf16":
            mp_policy = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            )
        elif args.amp == "fp16":
            mp_policy = MixedPrecision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float16,
                buffer_dtype=torch.float16,
            )

        tr_fsdp = FSDP(
            tr,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            auto_wrap_policy=auto_wrap,
            mixed_precision=mp_policy,
            device_id=torch.cuda.current_device(),
            use_orig_params=True,
        )
        pipeline.transformer = tr_fsdp

        # Move remaining components to GPU with consistent dtype
        pipeline.vae = pipeline.vae.to(device=torch.cuda.current_device(), dtype=torch.bfloat16)
        pipeline.text_encoder = pipeline.text_encoder.to(device=torch.cuda.current_device(), dtype=torch.bfloat16)
    else:
        pipeline = CogVideoXPipeline.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
        ).to("cuda")
        pipeline.transformer = pipeline.transformer.to(torch.bfloat16)
        pipeline.vae = pipeline.vae.to(torch.bfloat16)
        pipeline.text_encoder = pipeline.text_encoder.to(torch.bfloat16)

    if args.gradient_checkpointing:
        maybe_enable_gradient_checkpointing(pipeline.transformer, label="CogVideoX transformer")

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
    
    if args.gradient_checkpointing:
        maybe_enable_gradient_checkpointing(pipeline.transformer, label="LTX transformer")

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


def create_wan_adapter(args):
    """Create Wan2.1 adapter"""
    import sys
    from pathlib import Path
    
    # Add Wan2.1 to path
    wan_path = Path(__file__).parent.parent.parent / "Wan2.1"
    sys.path.insert(0, str(wan_path))
    
    print("Loading Wan2.1 pipeline from Wan2.1 folder")
    print("Using custom Wan implementation")
    
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

    if args.gradient_checkpointing:
        maybe_enable_gradient_checkpointing(getattr(pipeline, "model", None), label="Open-Sora model")

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

    # Mixed precision / AMP
    parser.add_argument(
        "--amp",
        choices=["none", "bf16", "fp16"],
        default="none",
        help="Enable mixed precision for the GRPO step (bf16 or fp16). Default: off."
    )

    parser.add_argument(
        "--fsdp",
        action="store_true",
        default=False,
        help="Enable FSDP full-shard on the transformer (use with torchrun)."
    )

    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        default=False,
        help="Enable gradient checkpointing on the model transformer to reduce activation memory."
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
    # Mixed precision setup
    # ========================================================================
    amp_dtype = None
    if args.amp == "bf16":
        amp_dtype = torch.bfloat16
    elif args.amp == "fp16":
        amp_dtype = torch.float16
    
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
    print(f"Mixed precision: {args.amp}")
    print(f"FSDP: {args.fsdp}")
    print(f"Gradient checkpointing: {args.gradient_checkpointing}")
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
        amp_dtype=amp_dtype,
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
    print(f"Mixed precision: {args.amp}")
    print(f"FSDP: {args.fsdp}")
    print(f"Gradient checkpointing: {args.gradient_checkpointing}")
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
