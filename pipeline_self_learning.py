#!/usr/bin/env python3
"""
Self-learning pipeline that runs the LTX-Video GRPO stack to generate rollout
videos, then ranks them with Gemini (no GRPO advantages). It reuses the
pipeline setup from the main GRPO code but swaps the reward step for Gemini
ranking over saved MP4s.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import imageio
import numpy as np
import torch

from gemini_rewards import score_video_with_gemini
from helper import decode_x0_to_video
from ltx_video.inference import create_ltx_video_pipeline, load_pipeline_config
from ltx_video.models.autoencoders.vae_encode import latent_to_pixel_coords

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate rollout videos with LTX-Video and rank them via Gemini.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default="configs/ltxv-2b-0.9.6-dev-grpo.yaml",
        help="Pipeline config yaml.",
    )
    parser.add_argument(
        "--prompt-file",
        default="prompt.txt",
        help="Text file containing the prompt (first non-empty line used).",
    )
    parser.add_argument(
        "--num-rollouts",
        type=int,
        default=3,
        help="How many rollouts to generate before ranking.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=81,
        help="Number of frames per video.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Video height.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=768,
        help="Video width.",
    )
    parser.add_argument(
        "--frame-rate",
        type=int,
        default=16,
        help="Frames per second for saved MP4.",
    )
    parser.add_argument(
        "--video-glob",
        default="grpo/rollout_*.mp4",
        help="Glob pattern for rollout videos to rank.",
    )
    parser.add_argument(
        "--model-name",
        default="gemini-1.5-flash",
        help="Gemini model to use.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Gemini API key (falls back to GEMINI_API_KEY env var).",
    )
    parser.add_argument(
        "--summary-json",
        default="grpo/gemini_ranking_summary.json",
        help="Where to store per-video scores and the winner.",
    )
    return parser.parse_args()


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _load_prompt(prompt_file: str) -> str:
    if not os.path.exists(prompt_file):
        return "A ball bouncing up a staircase, hitting each step sequentially."
    with open(prompt_file, "r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if text:
                return text
    return "A ball bouncing up a staircase, hitting each step sequentially."


def _prepare_pipeline(config_path: str):
    cfg = load_pipeline_config(config_path)
    ckpt_name = cfg["checkpoint_path"]
    pipeline = create_ltx_video_pipeline(
        ckpt_path=ckpt_name,
        precision="bfloat16",
        text_encoder_model_name_or_path=cfg["text_encoder_model_name_or_path"],
        sampler=cfg.get("sampler"),
        device="cuda",
        enhance_prompt=False,
    )
    if hasattr(pipeline.vae, "enable_tiling"):
        try:
            pipeline.vae.enable_tiling()
            print("✅ VAE tiling enabled")
        except Exception as exc:  # pragma: no cover
            print(f"⚠️  VAE tiling failed: {exc}")
    if hasattr(pipeline.vae, "enable_slicing"):
        pipeline.vae.enable_slicing()
        print("✅ VAE slicing enabled (decodes in chunks)")
    return pipeline


def _encode_prompt(pipeline, prompt: str):
    prompt_embeds, prompt_attention_mask = pipeline.encode_prompt(
        prompt=prompt,
        device="cuda",
        num_images_per_prompt=1,
        do_classifier_free_guidance=False,
    )
    return prompt_embeds, prompt_attention_mask


def _prepare_latents_and_indices(
    pipeline,
    *,
    num_frames: int,
    height: int,
    width: int,
    frame_rate: int,
):
    scheduler = pipeline.scheduler
    scheduler.set_timesteps(20, device="cuda")
    timesteps = scheduler.timesteps

    latents = pipeline.prepare_latents(
        latents=None,
        media_items=None,
        timestep=timesteps[0],
        latent_shape=(
            1,
            pipeline.vae.config.latent_channels,
            num_frames // pipeline.video_scale_factor,
            height // pipeline.vae_scale_factor,
            width // pipeline.vae_scale_factor,
        ),
        dtype=torch.bfloat16,
        device=torch.device("cuda"),
        generator=None,
    )
    latents, latent_coords = pipeline.patchifier.patchify(latents)
    pixel_coords = latent_to_pixel_coords(
        latent_coords,
        pipeline.vae,
        causal_fix=pipeline.transformer.config.causal_temporal_positioning,
    )
    indices_grid = pixel_coords.to(torch.float32)
    indices_grid[:, 0] *= (1.0 / frame_rate)
    return latents, indices_grid, timesteps


def _denoise_single_video(
    pipeline,
    *,
    latents: torch.Tensor,
    indices_grid: torch.Tensor,
    timesteps: torch.Tensor,
    prompt_embeds: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    num_frames: int,
    height: int,
    width: int,
) -> torch.Tensor:
    model = pipeline.transformer
    for t in timesteps:
        with torch.no_grad():
            noise_pred = model(
                latents,
                indices_grid=indices_grid,
                encoder_hidden_states=prompt_embeds,
                encoder_attention_mask=prompt_attention_mask,
                timestep=t,
                return_dict=False,
            )[0]
            latents, x0_est = pipeline.denoising_step(
                latents=latents,
                noise_pred=noise_pred,
                current_timestep=None,
                conditioning_mask=None,
                t=t,
                extra_step_kwargs={},
                stochastic_sampling=True,
                return_x0=True,
            )

    video = decode_x0_to_video(
        x0_est,
        pipeline,
        num_frames=num_frames,
        height=height,
        width=width,
        is_patchified=True,
    )
    return video


def _save_video_to_mp4(video: torch.Tensor, out_path: Path, frame_rate: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    video_np = video[0].float().cpu().numpy()  # [T, C, H, W]
    video_np = np.transpose(video_np, (0, 2, 3, 1))  # [T, H, W, C]
    video_np = (video_np * 255).clip(0, 255).astype(np.uint8)
    writer = imageio.get_writer(
        str(out_path),
        fps=frame_rate,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        macro_block_size=1,
    )
    for frame in video_np:
        writer.append_data(frame)
    writer.close()


# -----------------------------------------------------------------------------
# Gemini ranking
# -----------------------------------------------------------------------------


def _load_videos(glob_pattern: str) -> List[Path]:
    paths = sorted(Path().glob(glob_pattern))
    if not paths:
        raise FileNotFoundError(
            f"No videos matched pattern '{glob_pattern}'. "
            "Generate rollouts first (e.g., grpo/rollout_*.mp4)."
        )
    return paths


def rank_with_gemini(
    video_paths: List[Path],
    *,
    model_name: str,
    api_key: Optional[str],
) -> Tuple[dict, List[dict]]:
    """
    Score each video with Gemini and return the best plus all scores.

    Returns:
        winner: dict with path and scores
        scores: list of dicts for every video
    """
    scores = []
    for path in video_paths:
        score = score_video_with_gemini(
            str(path),
            model_name=model_name,
            api_key=api_key,
        )
        scores.append(
            {
                "path": str(path),
                "motion_dynamics": score["motion_dynamics"],
                "physical_properties": score["physical_properties"],
                "overall": score["overall"],
                "raw_response": score["raw_response"],
            }
        )

    winner = max(scores, key=lambda x: x["overall"])
    return winner, scores


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # 1) Prepare pipeline and prompt
    # ------------------------------------------------------------------
    prompt = _load_prompt(args.prompt_file)
    print(f"Prompt: {prompt}")

    pipeline = _prepare_pipeline(args.config)
    prompt_embeds, prompt_attention_mask = _encode_prompt(pipeline, prompt)
    latents, indices_grid, timesteps = _prepare_latents_and_indices(
        pipeline,
        num_frames=args.frames,
        height=args.height,
        width=args.width,
        frame_rate=args.frame_rate,
    )

    # ------------------------------------------------------------------
    # 2) Generate rollout videos
    # ------------------------------------------------------------------
    rollout_paths: List[Path] = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for r in range(args.num_rollouts):
        print(f"\nGenerating rollout {r+1}/{args.num_rollouts} ...")
        video = _denoise_single_video(
            pipeline,
            latents=latents.clone(),
            indices_grid=indices_grid,
            timesteps=timesteps,
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            num_frames=args.frames,
            height=args.height,
            width=args.width,
        )
        out_path = Path(f"grpo/rollout_{timestamp}_{r}.mp4")
        _save_video_to_mp4(video, out_path, frame_rate=args.frame_rate)
        rollout_paths.append(out_path)
        del video
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 3) Rank rollouts with Gemini
    # ------------------------------------------------------------------
    print(f"\nRanking {len(rollout_paths)} rollouts with Gemini...")
    winner, scores = rank_with_gemini(
        rollout_paths,
        model_name=args.model_name,
        api_key=args.api_key,
    )

    print("\nPer-video scores (higher overall is better):")
    for entry in sorted(scores, key=lambda x: x["overall"], reverse=True):
        print(
            f"  {entry['overall']:5.2f}  "
            f"motion={entry['motion_dynamics']:4.2f}  "
            f"physics={entry['physical_properties']:4.2f}  "
            f"{entry['path']}"
        )

    print("\n🏆 Best rollout:")
    print(json.dumps(winner, indent=2))

    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps({"winner": winner, "scores": scores}, indent=2),
        encoding="utf-8",
    )
    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    main()

