#!/usr/bin/env python3
"""
Example runner: unified GRPO on LTX-Video (last 25 steps).

This is intentionally minimal; it reuses your existing reward in `reward_functions.py`.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

from ltx_video.ltx_video.inference import load_pipeline_config, create_ltx_video_pipeline
from reward_functions import reward_function, clear_model_cache

from unified_grpo.adapters.ltx_adapter import LTXAdapter
from unified_grpo.grpo_core import GRPOConfig, run_grpo_for_prompt


def main() -> int:
    clear_model_cache()

    config_path = os.getenv("LTX_CONFIG", "configs/ltxv-2b-0.9.5.yaml")
    cfg = load_pipeline_config(config_path)
    ckpt_path = cfg["checkpoint_path"]
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    pipe = create_ltx_video_pipeline(
        ckpt_path=ckpt_path,
        precision="bfloat16",
        text_encoder_model_name_or_path=cfg["text_encoder_model_name_or_path"],
        sampler=cfg.get("sampler"),
        device="cuda",
        enhance_prompt=False,
    )

    prompt = os.getenv("PROMPT") or "A ball bouncing up a staircase, hitting each step sequentially."
    height = int(os.getenv("HEIGHT", "512"))
    width = int(os.getenv("WIDTH", "768"))
    num_frames = int(os.getenv("NUM_FRAMES", "81"))
    seed = int(os.getenv("SEED", "26"))

    # Conditioning (borrowed from your pipeline.py / baseline_intermediate_videos.py patterns)
    # We rely on pipeline internals to produce prompt embeddings.
    prompt_embeds, prompt_attention_mask = pipe.encode_prompt(
        prompt=prompt,
        device=pipe.device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=False,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        lora_scale=None,
    )

    indices_grid = pipe._get_indices_grid(height=height, width=width, num_frames=num_frames, device=pipe.device)

    adapter = LTXAdapter(
        pipeline=pipe,
        prompt_embeds=prompt_embeds,
        prompt_attention_mask=prompt_attention_mask,
        indices_grid=indices_grid,
        height=height,
        width=width,
        num_frames=num_frames,
        trainable_blocks=None,  # set to [14..27] if you want block-only training
    )

    cfg_grpo = GRPOConfig(
        num_inference_steps=int(os.getenv("NUM_INFERENCE_STEPS", "40")),
        num_grpo_steps=int(os.getenv("NUM_GRPO_STEPS", "25")),
        num_rollouts=int(os.getenv("NUM_ROLLOUTS", "3")),
        rollout_noise_scale=float(os.getenv("ROLLOUT_NOISE_SCALE", "0.5")),
        lr=float(os.getenv("LR", "2e-4")),
        grad_clip=float(os.getenv("GRAD_CLIP", "1.0")),
        normalize_advantages=True,
        logprob_sigma=float(os.getenv("LOGPROB_SIGMA", "1.0")),
    )

    out_dir = Path(os.getenv("OUT_DIR", "unified_grpo_ltx_out")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    def _reward_fn(video: torch.Tensor, p: str) -> torch.Tensor:
        # reward_function expects float video tensor and prompt
        if video.dtype == torch.bfloat16:
            video = video.float()
        return reward_function(video, prompt=p, device="cuda")

    stats = run_grpo_for_prompt(adapter=adapter, prompt=prompt, reward_fn=_reward_fn, seed=seed, out_dir=out_dir, cfg=cfg_grpo)
    print("\n[Unified-GRPO] Done:", stats, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

