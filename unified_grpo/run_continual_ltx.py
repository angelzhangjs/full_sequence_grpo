#!/usr/bin/env python3
"""
Continual GRPO training for LTX across a prompt file.

Unlike the per-prompt batch scripts, this keeps one LTX model instance alive
across all prompts, so learned weights accumulate over the dataset.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import List

import torch

from unified_grpo.create_adapter import create_ltx_adapter
from unified_grpo.grpo_core import GRPOConfig, run_grpo_for_prompt
from unified_grpo.reward_adaptive_physics import (
    AdaptivePhysicsRewardConfig,
    AdaptiveRewardWeightNet,
    adaptive_physics_reward,
)
from unified_grpo.reward_hybrid_video import HybridRewardConfig, hybrid_video_reward
from unified_grpo.reward_physics_handcrafted import HandcraftedPhysicsRewardConfig, handcrafted_physics_reward
from unified_grpo.reward_qwen import QwenRewardConfig, qwen_video_reward
from unified_grpo.reward_simple_clip_dino import comprehensive_grpo_reward
from unified_grpo.reward_xclip import XClipRewardConfig, xclip_video_reward
from unified_grpo.run import _save_model_checkpoint
from unified_grpo.utils import WriteLogger


def _read_prompts(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _safe_slug(text: str, max_chars: int = 50) -> str:
    import re
    s = text[:max_chars]
    s = re.sub(r"[^A-Za-z0-9 ]+", "", s)
    s = re.sub(r"\s+", "-", s.strip())
    return s or "prompt"


def _update_ltx_prompt(adapter, prompt: str, negative_prompt: str) -> None:
    pipeline = adapter.pipeline
    device = adapter.device()
    with torch.no_grad():
        prompt_embeds, prompt_attention_mask, negative_embeds, negative_attention_mask = pipeline.encode_prompt(
            prompt=str(prompt or ""),
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=str(negative_prompt or ""),
        )
    adapter.prompt_embeds = prompt_embeds.detach()
    adapter.negative_prompt_embeds = negative_embeds.detach()
    adapter.prompt_attention_mask = prompt_attention_mask
    adapter.negative_prompt_attention_mask = negative_attention_mask


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Continual GRPO training for LTX over a prompt file")
    p.add_argument("--model-path", type=str, default="Lightricks/LTX-Video")
    p.add_argument("--prompt-file", type=str, required=True)
    p.add_argument("--output-root", type=str, default=None)
    p.add_argument("--negative-prompt", type=str, default="")
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=720)
    p.add_argument("--num-frames", type=int, default=32)
    p.add_argument("--guidance-scale", type=float, default=7.5)
    p.add_argument("--num-inference-steps", type=int, default=50)
    p.add_argument("--num-grpo-steps", type=int, default=15)
    p.add_argument("--num-rollouts", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--unfreeze-percentage", type=float, default=0.20)
    p.add_argument("--use-lora", action="store_true", default=False)
    p.add_argument("--lora-rank", type=int, default=4)
    p.add_argument("--lora-alpha", type=int, default=8)
    p.add_argument("--lora-blocks", type=str, default="last")
    p.add_argument("--train-blocks", type=str, default=None)
    p.add_argument("--reward-backend", type=str, default="image_clip",
                   choices=["image_clip", "xclip", "qwen", "hybrid_video", "physics_handcrafted", "adaptive_physics"])
    p.add_argument("--reward-debug", action="store_true", default=False)
    p.add_argument("--clip-num-frames", type=int, default=0)
    p.add_argument("--clip-aggregation", type=str, default="video_mean_pool",
                   choices=["video_mean_pool", "frame_mean"])
    p.add_argument("--xclip-model-id", type=str, default="microsoft/xclip-base-patch32")
    p.add_argument("--xclip-num-frames", type=int, default=8)
    p.add_argument("--qwen-model-id", type=str, default="Qwen/Qwen2-VL-2B-Instruct")
    p.add_argument("--qwen-num-frames", type=int, default=8)
    p.add_argument("--qwen-max-new-tokens", type=int, default=192)
    p.add_argument("--qwen-temperature", type=float, default=0.0)
    p.add_argument("--qwen-w-align", type=float, default=0.5)
    p.add_argument("--qwen-w-physics", type=float, default=0.3)
    p.add_argument("--qwen-w-dynamic-motion", type=float, default=0.2)
    p.add_argument("--hybrid-adaptive", action="store_true", default=False)
    p.add_argument("--hybrid-w-xclip", type=float, default=0.6)
    p.add_argument("--hybrid-w-physics", type=float, default=0.25)
    p.add_argument("--hybrid-w-dynamic-motion", type=float, default=0.15)
    p.add_argument("--hybrid-w-xclip-end", type=float, default=0.45)
    p.add_argument("--hybrid-w-physics-end", type=float, default=0.35)
    p.add_argument("--hybrid-w-dynamic-motion-end", type=float, default=0.20)
    p.add_argument("--physics-category-override", type=str, default=None)
    p.add_argument("--physics-handcrafted-w-motion", type=float, default=0.35)
    p.add_argument("--physics-handcrafted-w-category", type=float, default=0.65)
    p.add_argument("--adaptive-physics-hidden-dim", type=int, default=32)
    p.add_argument("--gradient-checkpointing", action="store_true", default=False)
    p.add_argument("--save-denoising-strip-png", action="store_true", default=False)
    p.add_argument("--save-denoising-step-snapshots", action="store_true", default=False)
    p.add_argument("--denoising-step-snapshot-stride", type=int, default=1)
    p.add_argument("--denoising-strip-step-stride", type=int, default=5)
    p.add_argument("--denoising-strip-max-thumb-height", type=int, default=280)
    p.add_argument("--output-video-duration-s", type=float, default=4.0)
    p.add_argument("--save-training-trajectory-video", action="store_true", default=False)
    return p


def _build_reward_components(args, adapter):
    total_calls = max(1, int(args.num_grpo_steps) * int(args.num_rollouts) * 1000000)
    state = {"call_index": 0}
    reward_aux_params: List[torch.nn.Parameter] = []
    adaptive_weight_net = None
    if str(args.reward_backend).lower() == "adaptive_physics":
        adaptive_weight_net = AdaptiveRewardWeightNet(hidden_dim=int(args.adaptive_physics_hidden_dim)).to(adapter.device())
        reward_aux_params = [p for p in adaptive_weight_net.parameters() if p.requires_grad]

    def reward_fn(video: torch.Tensor, prompt: str) -> torch.Tensor:
        backend = str(args.reward_backend).lower()
        if backend == "adaptive_physics":
            progress = float(state["call_index"]) / float(max(1, total_calls - 1))
            scores = adaptive_physics_reward(
                video=video,
                prompt=prompt,
                device=torch.device(video.device),
                weight_net=adaptive_weight_net,
                cfg=AdaptivePhysicsRewardConfig(
                    clip_num_sampled_frames=int(args.clip_num_frames),
                    clip_aggregation=str(args.clip_aggregation),
                    category_override=getattr(args, "physics_category_override", None),
                    hidden_dim=int(args.adaptive_physics_hidden_dim),
                ),
                progress=progress,
            )
            state["call_index"] += 1
            if bool(args.reward_debug):
                printable = {k: (float(v.item()) if isinstance(v, torch.Tensor) else v) for k, v in scores.items()}
                print(
                    "  [ADAPTIVE PHYSICS REWARD]\n"
                    f"    category: {printable['category']}\n"
                    f"    clip_alignment: {printable['clip_alignment']:.4f}\n"
                    f"    generic_motion: {printable['generic_motion']:.4f}\n"
                    f"    category_score: {printable['category_score']:.4f}\n"
                    f"    w_clip: {printable['w_clip']:.4f}\n"
                    f"    w_motion: {printable['w_motion']:.4f}\n"
                    f"    w_category: {printable['w_category']:.4f}\n"
                    f"    progress: {printable['progress']:.6f}\n"
                    f"    reward: {printable['reward']:.4f}"
                )
            return scores["reward"]

        if backend == "physics_handcrafted":
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

        if backend == "hybrid_video":
            progress = float(state["call_index"]) / float(max(1, total_calls - 1))
            scores = hybrid_video_reward(
                video=video,
                prompt=prompt,
                device=torch.device(video.device),
                xclip_cfg=XClipRewardConfig(model_id=str(args.xclip_model_id), num_sampled_frames=int(args.xclip_num_frames)),
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
            state["call_index"] += 1
            if bool(args.reward_debug):
                print(f"  [HYBRID VIDEO REWARD] {scores}")
            return torch.tensor(float(scores["reward"]), device=video.device)

        if backend == "xclip":
            scores = xclip_video_reward(
                video=video,
                prompt=prompt,
                device=torch.device(video.device),
                cfg=XClipRewardConfig(model_id=str(args.xclip_model_id), num_sampled_frames=int(args.xclip_num_frames)),
            )
            if bool(args.reward_debug):
                print(f"  [XCLIP REWARD] {scores}")
            return torch.tensor(float(scores["reward"]), device=video.device)

        if backend == "qwen":
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

        result = comprehensive_grpo_reward(
            frames=video,
            prompt=prompt,
            device="cuda",
            use_clip=True,
            clip_num_sampled_frames=int(args.clip_num_frames),
            clip_aggregation=str(args.clip_aggregation),
        )
        return torch.tensor(float(result.get("reward", 0.0)), device=video.device)

    return reward_fn, reward_aux_params


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    prompt_file = Path(args.prompt_file).expanduser().resolve()
    prompts = _read_prompts(prompt_file)
    if not prompts:
        raise ValueError(f"No non-empty prompts found in {prompt_file}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_root is None:
        model_name = str(args.model_path).split("/")[-1].replace("/", "-")
        output_root = Path(f"./{model_name}_continual_grpo_{timestamp}").resolve()
    else:
        output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    master_log = output_root / f"continual_training_log_{timestamp}.txt"
    logger = WriteLogger(str(master_log))
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = logger
    sys.stderr = logger
    try:
        print("=" * 70)
        print("CONTINUAL LTX GRPO TRAINING")
        print("=" * 70)
        print(f"Prompt file: {prompt_file}")
        print(f"Total prompts: {len(prompts)}")
        print(f"Output root: {output_root}")
        print(f"Model path: {args.model_path}")
        print(f"Reward backend: {args.reward_backend}")
        print()

        args.prompt = prompts[0]
        adapter = create_ltx_adapter(args)
        reward_fn, reward_aux_params = _build_reward_components(args, adapter)

        grpo_config = GRPOConfig(
            num_inference_steps=args.num_inference_steps,
            num_grpo_steps=args.num_grpo_steps,
            num_rollouts=args.num_rollouts,
            lr=args.lr,
            detach_advantages=(str(args.reward_backend).lower() != "adaptive_physics"),
            gradient_checkpointing=bool(args.gradient_checkpointing),
            save_denoising_trajectory_strip_png=bool(args.save_denoising_strip_png),
            denoising_strip_step_stride=int(args.denoising_strip_step_stride),
            save_denoising_step_snapshots=bool(args.save_denoising_step_snapshots),
            denoising_step_snapshot_stride=int(args.denoising_step_snapshot_stride),
            denoising_strip_max_thumb_height=int(args.denoising_strip_max_thumb_height),
            output_video_duration_s=float(args.output_video_duration_s),
            save_training_trajectory_video=bool(args.save_training_trajectory_video),
        )

        for idx, prompt in enumerate(prompts, start=1):
            prompt_slug = _safe_slug(prompt)
            prompt_dir = output_root / f"p{idx:03d}_{prompt_slug}"
            prompt_dir.mkdir(parents=True, exist_ok=True)
            (prompt_dir / "prompt.txt").write_text(prompt + "\n")

            print()
            print("=" * 70)
            print(f"Prompt {idx}/{len(prompts)}")
            print("=" * 70)
            print(prompt)
            print(f"Output dir: {prompt_dir / 'grpo'}")
            print()

            _update_ltx_prompt(adapter, prompt, args.negative_prompt)
            metrics = run_grpo_for_prompt(
                adapter=adapter,
                prompt=prompt,
                reward_fn=reward_fn,
                seed=int(args.seed),
                out_dir=prompt_dir / "grpo",
                cfg=grpo_config,
                model_type="ltx",
                extra_trainable_parameters=reward_aux_params,
            )
            print(f"✅ Continual GRPO complete for prompt {idx}")
            print(f"Metrics: {metrics}")

        _save_model_checkpoint(
            adapter=adapter,
            args=args,
            checkpoint_dir=output_root / "final_checkpoint",
        )
        print()
        print("=" * 70)
        print("CONTINUAL TRAINING COMPLETE")
        print("=" * 70)
        print(f"Output root: {output_root}")
        print(f"Master log: {master_log}")
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        logger.close()


if __name__ == "__main__":
    main()
