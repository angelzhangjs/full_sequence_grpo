#!/usr/bin/env python3
"""
Unified GRPO Training Script
Supports multiple video models via --model-type argument
"""

import argparse
import builtins
import json
import os
import torch
import sys
from pathlib import Path
from datetime import datetime
from typing import Any

from unified_grpo.grpo_core import run_grpo_for_prompt, GRPOConfig
from unified_grpo.adapters.base import get_trainable_module
from unified_grpo.lora_utils import has_lora, save_lora_weights
from unified_grpo.utils import WriteLogger

# Reward backends are optional-dependency-safe:
# - CLIP+DINO reward lazy-loads CLIP/DINO internally
# - Qwen reward lazy-loads transformers + qwen-vl-utils internally
from unified_grpo.reward_simple_clip_dino import comprehensive_grpo_reward
from unified_grpo.create_adapter import create_cogvideox_adapter, create_ltx_adapter


def _build_accelerator(args):
    wants_accelerate = bool(
        getattr(args, "use_accelerate", False)
        or getattr(args, "distributed_backend", "none") != "none"
        or getattr(args, "deepspeed_config", None)
        or int(os.environ.get("WORLD_SIZE", "1")) > 1
    )
    if not wants_accelerate:
        return None

    try:
        from accelerate import Accelerator
        from accelerate.utils import DeepSpeedPlugin
    except ImportError as exc:
        raise RuntimeError(
            "Accelerate is required for distributed/DeepSpeed GRPO runs. Install `accelerate` first."
        ) from exc

    deepspeed_plugin = None
    launcher_managed_deepspeed = str(os.environ.get("ACCELERATE_USE_DEEPSPEED", "")).lower() == "true"
    deepspeed_config = getattr(args, "deepspeed_config", None)
    if deepspeed_config:
        deepspeed_path = Path(deepspeed_config).expanduser().resolve()
        if not deepspeed_path.exists():
            raise FileNotFoundError(f"DeepSpeed config file not found: {deepspeed_path}")
        args.deepspeed_config = str(deepspeed_path)
        if not launcher_managed_deepspeed:
            deepspeed_plugin = DeepSpeedPlugin(hf_ds_config=str(deepspeed_path))

    accelerator = Accelerator(
        gradient_accumulation_steps=int(getattr(args, "gradient_accumulation_steps", 1)),
        mixed_precision=str(getattr(args, "mixed_precision", "bf16")),
        deepspeed_plugin=deepspeed_plugin,
    )

    if getattr(args, "distributed_backend", "none") == "deepspeed":
        from accelerate import DistributedType

        if accelerator.distributed_type != DistributedType.DEEPSPEED:
            raise RuntimeError(
                "DeepSpeed backend requested, but Accelerate did not initialize a DeepSpeed engine. "
                "Launch with `accelerate launch` using a DeepSpeed-enabled config, or pass `--deepspeed-config`."
            )

    return accelerator


def _silence_non_main_prints(accelerator, enabled: bool) -> Any:
    original_print = builtins.print
    if accelerator is not None and enabled and not accelerator.is_main_process:
        builtins.print = lambda *args, **kwargs: None  # type: ignore[assignment]
    return original_print


def _restore_print(original_print: Any) -> None:
    builtins.print = original_print

def create_adapter(args):
    """Create appropriate adapter based on model type"""
    model_type = args.model_type.lower()
    
    if model_type == "cogvideox":
        return create_cogvideox_adapter(args)
    elif model_type == "ltx":
        return create_ltx_adapter(args)
    elif model_type == "cosmos":
        from unified_grpo.create_adapter import create_cosmos_adapter
        return create_cosmos_adapter(args)
    # # elif model_type == "hunyuan":
    # #     return create_hunyuan_adapter(args)
    elif model_type == "wan":
        from unified_grpo.create_adapter import create_wan_adapter
        return create_wan_adapter(args)
    # elif model_type == "opensora":
    #     return create_opensora_adapter(args)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose from: cogvideox, ltx, cosmos, wan, opensora")


def _save_model_checkpoint(*, adapter, args, checkpoint_dir: Path, accelerator=None) -> None:
    if accelerator is not None:
        accelerator.wait_for_everyone()
        if not accelerator.is_main_process:
            return

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    target = get_trainable_module(adapter)
    unwrapped_target = accelerator.unwrap_model(target) if accelerator is not None else target

    if bool(getattr(args, "use_lora", False)) and has_lora(unwrapped_target):
        lora_dir = checkpoint_dir / "lora_adapter"
        save_lora_weights(unwrapped_target, str(lora_dir))
        lora_state_path = checkpoint_dir / "lora_state_dict.pt"
        lora_state = {
            name: param.detach().cpu()
            for name, param in unwrapped_target.named_parameters()
            if "lora_" in name.lower()
        }
        if accelerator is not None:
            accelerator.save(lora_state, str(lora_state_path))
        else:
            torch.save(lora_state, lora_state_path)
        print(f"✅ LoRA .pt checkpoint saved to: {lora_state_path}")
        checkpoint_type = "lora"
        checkpoint_path = str(lora_dir)
    else:
        accelerate_state_dir = checkpoint_dir / "accelerate_state"
        if accelerator is not None:
            accelerator.save_state(str(accelerate_state_dir))
            print(f"✅ Accelerator/DeepSpeed state saved to: {accelerate_state_dir}")
        state_path = checkpoint_dir / "model_state_dict.pt"
        state_dict = accelerator.get_state_dict(target) if accelerator is not None else unwrapped_target.state_dict()
        if accelerator is not None:
            accelerator.save(state_dict, str(state_path))
        else:
            torch.save(state_dict, state_path)
        print(f"✅ Model state_dict saved to: {state_path}")
        checkpoint_type = "state_dict"
        checkpoint_path = str(state_path)

    meta = {
        "model_type": str(args.model_type),
        "model_path": str(args.model_path),
        "prompt": str(args.prompt),
        "reward_backend": str(args.reward_backend),
        "use_lora": bool(getattr(args, "use_lora", False)),
        "lora_rank": int(getattr(args, "lora_rank", 0) or 0),
        "lora_alpha": int(getattr(args, "lora_alpha", 0) or 0),
        "lora_blocks": str(getattr(args, "lora_blocks", "")),
        "unfreeze_percentage": float(getattr(args, "unfreeze_percentage", 0.0) or 0.0),
        "height": int(getattr(args, "height", 0) or 0),
        "width": int(getattr(args, "width", 0) or 0),
        "num_frames": int(getattr(args, "num_frames", 0) or 0),
        "guidance_scale": float(getattr(args, "guidance_scale", 0.0) or 0.0),
        "checkpoint_type": checkpoint_type,
        "checkpoint_path": checkpoint_path,
        "checkpoint_pt_path": str(lora_state_path)
        if bool(getattr(args, "use_lora", False)) and has_lora(unwrapped_target)
        else checkpoint_path,
        "timestamp": datetime.now().isoformat(),
    }
    (checkpoint_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"✅ Checkpoint metadata saved to: {checkpoint_dir / 'metadata.json'}")

def main():
    parser = argparse.ArgumentParser(
        description="Unified GRPO Training for Multiple Video Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # CogVideoX:
  python -m unified_grpo.run --model-type cogvideox --model-path THUDM/CogVideoX-5b
  
  # LTX-Video:
  python -m unified_grpo.run --model-type ltx --model-path Lightricks/LTX-Video

  # 8x A100 DeepSpeed ZeRO-3:
  accelerate launch --config_file configs/accelerate_zero3_8xa100.yaml -m unified_grpo.run --model-type ltx --distributed-backend deepspeed --deepspeed-config configs/deepspeed_zero3_8xa100.json
  
"""
    )
    
    # ========================================================================
    # Model Selection
    # ========================================================================
    
    parser.add_argument(
        "--model-type",
        type=str,
        required=True,
        choices=["cogvideox", "ltx", "cosmos", "hunyuan", "wan", "opensora"],
        help="Type of video model to use"
    )
    
    parser.add_argument(
        "--model-path",
        type=str,
        help="Model identifier/path. For WAN, this can be a local checkpoint directory OR a HuggingFace model id (auto-downloaded)."
    )
    parser.add_argument(
        "--wan-t5-cpu",
        action="store_true",
        default=False,
        help="For WAN only: keep the T5 text encoder on CPU. Default is GPU for faster prompt encoding.",
    )
    parser.add_argument(
        "--cosmos-experiment-name",
        type=str,
        default=None,
        help="Optional Cosmos Hydra experiment name when loading from a local checkpoint path.",
    )
    parser.add_argument(
        "--cosmos-config-file",
        type=str,
        default=None,
        help="Optional Cosmos config file path when loading from a local checkpoint path.",
    )
    parser.add_argument(
        "--cosmos-lora-target-modules",
        type=str,
        default="q_proj,k_proj,v_proj,output_proj,mlp.layer1,mlp.layer2",
        help="Comma-separated Cosmos model.net module names to target with LoRA.",
    )
    parser.add_argument(
        "--save-checkpoint-dir",
        type=str,
        default=None,
        help="Optional directory to save final trained weights/checkpoint after this run.",
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
        default="image_clip",
        choices=["image_clip", "xclip", "qwen", "hybrid_video", "physics_handcrafted", "adaptive_physics"],
        help="Reward backend for GRPO. 'image_clip' is fast, 'xclip' uses a true video-text model, 'qwen' scores physics/temporal plausibility, 'hybrid_video' mixes X-CLIP with Qwen physics signals, 'physics_handcrafted' uses category-specific motion heuristics, and 'adaptive_physics' learns to adaptively mix image-CLIP with handcrafted physics terms.",
    )
    parser.add_argument(
        "--reward-debug",
        action="store_true",
        default=False,
        help="Print extra reward debug info (can be very noisy).",
    )
    parser.add_argument(
        "--clip-num-frames",
        type=int,
        default=0,
        help="How many video frames to use for the CLIP reward. Use 0 to score the full decoded video.",
    )
    parser.add_argument(
        "--clip-aggregation",
        type=str,
        default="video_mean_pool",
        choices=["video_mean_pool", "frame_mean"],
        help="CLIP reward aggregation. 'video_mean_pool' pools frame embeddings into one clip embedding before text-video scoring.",
    )
    parser.add_argument("--xclip-model-id", type=str, default="microsoft/xclip-base-patch32")
    parser.add_argument("--xclip-num-frames", type=int, default=8)
    parser.add_argument("--hybrid-adaptive", action="store_true", default=False)
    parser.add_argument("--hybrid-w-xclip", type=float, default=0.6)
    parser.add_argument("--hybrid-w-physics", type=float, default=0.25)
    parser.add_argument("--hybrid-w-dynamic-motion", type=float, default=0.15)
    parser.add_argument("--hybrid-w-xclip-end", type=float, default=0.45)
    parser.add_argument("--hybrid-w-physics-end", type=float, default=0.35)
    parser.add_argument("--hybrid-w-dynamic-motion-end", type=float, default=0.20)
    parser.add_argument("--physics-category-override", type=str, default=None)
    parser.add_argument("--physics-handcrafted-w-motion", type=float, default=0.35)
    parser.add_argument("--physics-handcrafted-w-category", type=float, default=0.65)
    parser.add_argument("--adaptive-physics-hidden-dim", type=int, default=32)

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

    parser.add_argument(
        "--save-denoising-strip-png",
        action="store_true",
        default=False,
        help=(
            "During denoising, periodically VAE-decode latents, take the middle frame, and save one wide "
            "horizontal PNG: denoising_trajectory_strip.png in the output dir."
        ),
    )
    parser.add_argument(
        "--denoising-strip-step-stride",
        type=int,
        default=5,
        help="Capture one denoising-strip panel every N completed denoising steps. Final step is always included.",
    )
    parser.add_argument(
        "--save-denoising-step-snapshots",
        action="store_true",
        default=False,
        help="Save middle-frame PNG snapshots during denoising under output_dir/denoising_steps/.",
    )
    parser.add_argument(
        "--denoising-step-snapshot-stride",
        type=int,
        default=1,
        help="Capture one denoising-step PNG snapshot every N completed denoising steps. Final step is always included.",
    )
    parser.add_argument(
        "--output-video-duration-s",
        type=float,
        default=4.0,
        help="Target duration in seconds for saved baseline/GRPO MP4 outputs when FPS is derived from frame count.",
    )
    parser.add_argument(
        "--denoising-strip-max-thumb-height",
        type=int,
        default=280,
        help="Max panel height (px) before stitching the denoising strip PNG; width scales with aspect ratio.",
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
    parser.add_argument(
        "--use-accelerate",
        action="store_true",
        default=False,
        help="Initialize training through Hugging Face Accelerate even for single-node runs.",
    )
    parser.add_argument(
        "--distributed-backend",
        type=str,
        default="none",
        choices=["none", "deepspeed"],
        help="Distributed backend to require. Use 'deepspeed' for a real ZeRO-3 engine.",
    )
    parser.add_argument(
        "--deepspeed-config",
        type=str,
        default=None,
        help="Path to a DeepSpeed JSON config file. When set, Accelerate will bootstrap a DeepSpeed plugin from this file.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps for Accelerate. The current GRPO implementation requires 1.",
    )
    parser.add_argument(
        "--mixed-precision",
        type=str,
        default="bf16",
        choices=["no", "bf16", "fp16"],
        help="Accelerate mixed precision mode used for distributed training.",
    )
    parser.add_argument(
        "--log-all-ranks",
        action="store_true",
        default=False,
        help="Disable rank-zero-only logging and allow every process to print to stdout.",
    )
    
    args = parser.parse_args()
    args.rank_zero_only_logging = not bool(getattr(args, "log_all_ranks", False))
    delay_accelerator_init = str(getattr(args, "model_type", "")).lower() in {"cogvideox", "ltx"}
    accelerator = None if delay_accelerator_init else _build_accelerator(args)
    original_print = _silence_non_main_prints(accelerator, bool(args.rank_zero_only_logging))
    
    # ========================================================================
    # Set model-specific defaults
    # ========================================================================
    
    if args.model_path is None:
        # Set default model path based on type
        defaults = {
            "cogvideox": "THUDM/CogVideoX-2b",
            "ltx": "Lightricks/LTX-Video", 
            "cosmos": "2B/pre-trained",
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
            "cosmos": 0,
            "hunyuan": 32,
            "wan": 24,
            "opensora": 28,
        }
        
        total_blocks = model_total_blocks.get(args.model_type, 28)
        if total_blocks <= 0:
            print(f"  Model: {args.model_type} (block count unknown)")
            print("  Skipping auto block selection; adapter will decide trainable parameters.\n")
            args.train_blocks = None
        else:
            num_blocks_to_train = max(1, int(total_blocks * args.unfreeze_percentage))
            start_block = total_blocks - num_blocks_to_train
            
            # Generate block indices
            blocks = list(range(start_block, total_blocks))
            args.train_blocks = ",".join(str(b) for b in blocks)
            
            print(f"  Model: {args.model_type} ({total_blocks} total blocks)")
            print(f"  Unfreezing: {num_blocks_to_train} blocks (last {args.unfreeze_percentage:.0%})")
            print(f"  Block indices: {args.train_blocks}\n")

    if int(getattr(args, "gradient_accumulation_steps", 1)) != 1:
        raise ValueError(
            "Unified GRPO currently requires `--gradient-accumulation-steps 1` so each denoising step performs "
            "an immediate policy update."
        )

    if accelerator is not None and getattr(args, "distributed_backend", "none") == "deepspeed":
        if str(args.reward_backend).lower() == "adaptive_physics":
            raise ValueError(
                "DeepSpeed ZeRO-3 is currently implemented only for the main video model fine-tuning path. "
                "`adaptive_physics` adds an auxiliary trainable reward network, which is intentionally deferred."
            )
    
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
    if accelerator is not None:
        print(
            f"Distributed: backend={getattr(args, 'distributed_backend', 'none')} | "
            f"processes={accelerator.num_processes} | mixed_precision={args.mixed_precision}"
        )
        if getattr(args, "deepspeed_config", None):
            print(f"DeepSpeed config: {args.deepspeed_config}")
    if bool(getattr(args, "save_denoising_strip_png", False)):
        print(
            "Denoising strip PNG: ON "
            f"(every {getattr(args, 'denoising_strip_step_stride', 5)} steps, "
            f"max thumb height {getattr(args, 'denoising_strip_max_thumb_height', 280)} px)"
        )
    if bool(getattr(args, "save_denoising_step_snapshots", False)):
        print(
            f"Denoising step snapshots: ON (every {int(getattr(args, 'denoising_step_snapshot_stride', 1))} completed steps)"
        )
    print(f"Training blocks: {args.train_blocks}")
    print()
    
    # ========================================================================
    # Create Adapter
    # ========================================================================
    
    print("Creating adapter...")
    adapter = create_adapter(args)
    if delay_accelerator_init:
        accelerator = _build_accelerator(args)
        original_print = _silence_non_main_prints(accelerator, bool(args.rank_zero_only_logging))
     
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
        detach_advantages=(str(args.reward_backend).lower() != "adaptive_physics"),
        save_denoising_trajectory_strip_png=bool(getattr(args, "save_denoising_strip_png", False)),
        denoising_strip_step_stride=int(getattr(args, "denoising_strip_step_stride", 5)),
        save_denoising_step_snapshots=bool(getattr(args, "save_denoising_step_snapshots", False)),
        denoising_step_snapshot_stride=int(getattr(args, "denoising_step_snapshot_stride", 1)),
        denoising_strip_max_thumb_height=int(getattr(args, "denoising_strip_max_thumb_height", 280)),
        output_video_duration_s=float(getattr(args, "output_video_duration_s", 4.0)),
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
    if accelerator is None or accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
    if accelerator is not None:
        accelerator.wait_for_everyone()
    
    # Setup logging to file
    log_file = output_dir / f"training_log_{timestamp}.txt"
    
    print(f"Training log: {log_file}")
    print(f"Output dir: {output_dir}")
    print(f"Prompt: {args.prompt}")
    
    # Redirect stdout to both console and log file on the main process only.
    logger = None
    if accelerator is None or accelerator.is_main_process:
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
    
    total_hybrid_calls = max(1, int(args.num_grpo_steps) * int(args.num_rollouts))
    hybrid_state = {"call_index": 0}
    reward_aux_params = []
    adaptive_weight_net = None
    if str(args.reward_backend).lower() == "adaptive_physics":
        from unified_grpo.reward_adaptive_physics import AdaptiveRewardWeightNet

        reward_device = accelerator.device if accelerator is not None else adapter.device()
        adaptive_weight_net = AdaptiveRewardWeightNet(hidden_dim=int(args.adaptive_physics_hidden_dim)).to(reward_device)
        reward_aux_params = [p for p in adaptive_weight_net.parameters() if p.requires_grad]

    # Build reward_fn closure based on CLI.
    def reward_fn(video: torch.Tensor, prompt: str) -> torch.Tensor:
        if str(args.reward_backend).lower() == "adaptive_physics":
            from unified_grpo.reward_adaptive_physics import (
                AdaptivePhysicsRewardConfig,
                adaptive_physics_reward,
            )

            progress = float(hybrid_state["call_index"]) / float(max(1, total_hybrid_calls - 1))
            scores = adaptive_physics_reward(
                video=video,
                prompt=prompt,
                device=torch.device(video.device),
                weight_net=adaptive_weight_net,  # type: ignore[arg-type]
                cfg=AdaptivePhysicsRewardConfig(
                    clip_num_sampled_frames=int(args.clip_num_frames),
                    clip_aggregation=str(args.clip_aggregation),
                    category_override=getattr(args, "physics_category_override", None),
                    hidden_dim=int(args.adaptive_physics_hidden_dim),
                ),
                progress=progress,
            )
            hybrid_state["call_index"] += 1
            if bool(args.reward_debug):
                printable = {
                    k: (float(v.item()) if isinstance(v, torch.Tensor) else v)
                    for k, v in scores.items()
                }
                print(
                    "  [ADAPTIVE PHYSICS REWARD]\n"
                    f"    category: {printable['category']}\n"
                    f"    clip_alignment: {printable['clip_alignment']:.4f}\n"
                    f"    generic_motion: {printable['generic_motion']:.4f}\n"
                    f"    category_score: {printable['category_score']:.4f}\n"
                    f"    w_clip: {printable['w_clip']:.4f}\n"
                    f"    w_motion: {printable['w_motion']:.4f}\n"
                    f"    w_category: {printable['w_category']:.4f}\n"
                    f"    progress: {printable['progress']:.4f}\n"
                    f"    reward: {printable['reward']:.4f}"
                )
            return scores["reward"]  # type: ignore[return-value]

        if str(args.reward_backend).lower() == "physics_handcrafted":
            from unified_grpo.reward_physics_handcrafted import (
                HandcraftedPhysicsRewardConfig,
                handcrafted_physics_reward,
            )

            scores = handcrafted_physics_reward(
                video=video,
                prompt=prompt,
                cfg=HandcraftedPhysicsRewardConfig(
                    category_override=getattr(args, "physics_category_override", None),
                    w_motion=float(args.physics_handcrafted_w_motion),
                    w_category=float(args.physics_handcrafted_w_category),
                ),
            )
            if bool(args.reward_debug):
                print(
                    "  [PHYSICS HANDCRAFTED REWARD]\n"
                    f"    category: {scores['category']}\n"
                    f"    generic_motion: {scores['generic_motion']:.4f}\n"
                    f"    category_score: {scores['category_score']:.4f}\n"
                    f"    reward: {scores['reward']:.4f}"
                )
            return torch.tensor(float(scores["reward"]), device=video.device)

        if str(args.reward_backend).lower() == "hybrid_video":
            from unified_grpo.reward_hybrid_video import HybridRewardConfig, hybrid_video_reward
            from unified_grpo.reward_qwen import QwenRewardConfig
            from unified_grpo.reward_xclip import XClipRewardConfig

            progress = None
            if bool(args.hybrid_adaptive):
                progress = float(hybrid_state["call_index"]) / float(max(1, total_hybrid_calls - 1))

            scores = hybrid_video_reward(
                video=video,
                prompt=prompt,
                device=torch.device(video.device),
                xclip_cfg=XClipRewardConfig(
                    model_id=str(args.xclip_model_id),
                    num_sampled_frames=int(args.xclip_num_frames),
                ),
                qwen_cfg=QwenRewardConfig(
                    model_id=str(args.qwen_model_id),
                    num_sampled_frames=int(args.qwen_num_frames),
                    max_new_tokens=int(args.qwen_max_new_tokens),
                    temperature=float(args.qwen_temperature),
                    w_align=float(args.qwen_w_align),
                    w_physics=float(args.qwen_w_physics),
                    w_dynamic_motion=float(args.qwen_w_dynamic_motion),
                ),
                cfg=HybridRewardConfig(
                    w_xclip=float(args.hybrid_w_xclip),
                    w_physics=float(args.hybrid_w_physics),
                    w_dynamic_motion=float(args.hybrid_w_dynamic_motion),
                    w_xclip_end=float(args.hybrid_w_xclip_end),
                    w_physics_end=float(args.hybrid_w_physics_end),
                    w_dynamic_motion_end=float(args.hybrid_w_dynamic_motion_end),
                    adaptive=bool(args.hybrid_adaptive),
                ),
                progress=progress,
            )
            hybrid_state["call_index"] += 1
            if bool(args.reward_debug):
                print(f"  [HYBRID VIDEO REWARD] {scores}")
            return torch.tensor(float(scores["reward"]), device=video.device)

        if str(args.reward_backend).lower() == "xclip":
            from unified_grpo.reward_xclip import XClipRewardConfig, xclip_video_reward

            scores = xclip_video_reward(
                video=video,
                prompt=prompt,
                device=torch.device(video.device),
                cfg=XClipRewardConfig(
                    model_id=str(args.xclip_model_id),
                    num_sampled_frames=int(args.xclip_num_frames),
                ),
            )
            if bool(args.reward_debug):
                print(f"  [XCLIP REWARD] {scores}")
            return torch.tensor(float(scores["reward"]), device=video.device)

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

        # Default: CLIP-only
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
            device=str(video.device),
            use_clip=True,
            clip_num_sampled_frames=int(args.clip_num_frames),
            clip_aggregation=str(args.clip_aggregation),
        )
        total_reward = float(result.get("reward", 0.0))
        if bool(args.reward_debug):
            print(f"  [DEBUG] Reward result: {result}")
            print(f"  [DEBUG] Total reward: {total_reward:.6f}\n")
        return torch.tensor(total_reward, device=video.device)

    if str(args.reward_backend).lower() == "adaptive_physics":
        print(
            "Using reward function: Adaptive physics reward (image-CLIP + generic motion + category score)\n"
        )
    elif str(args.reward_backend).lower() == "physics_handcrafted":
        print(
            "Using reward function: Handcrafted physics reward\n"
        )
    elif str(args.reward_backend).lower() == "hybrid_video":
        print(
            "Using reward function: Hybrid video reward (X-CLIP alignment + Qwen physics/dynamic motion)\n"
        )
    elif str(args.reward_backend).lower() == "xclip":
        print(
            "Using reward function: X-CLIP true video-text alignment\n"
        )
    elif str(args.reward_backend).lower() == "qwen":
        print(
            "Using reward function: Qwen2-VL (text alignment + physical plausibility + dynamic motion consistency)\n"
        )
    else:
        print(
            "Using reward function: CLIP alignment\n"
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
            extra_trainable_parameters=reward_aux_params,
            accelerator=accelerator,
        )

        print("\n" + "="*70)
        print("TRAINING COMPLETE!")
        print("="*70)
        print(f"Model: {args.model_type}")
        print("Metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

        print(f"\nOutput: {output_dir}")
        if args.save_checkpoint_dir:
            _save_model_checkpoint(
                adapter=adapter,
                args=args,
                checkpoint_dir=Path(args.save_checkpoint_dir).expanduser().resolve(),
                accelerator=accelerator,
            )
        print("✅ Done!")
        
    finally:
        # Close logger and restore stdout
        if logger is not None:
            logger.close()
            sys.stdout = logger.terminal
            sys.stderr = sys.__stderr__
        _restore_print(original_print)

        if accelerator is None or accelerator.is_main_process:
            print(f"\n✅ Training complete! Log saved to: {log_file}")

if __name__ == "__main__":
    main()
