#!/usr/bin/env python3
"""
Save per-timestep intermediate x0 videos for the *baseline* LTX-Video pipeline.

This script follows the same padding + inference call path as:
  ltx_video/ltx_video/inference.py

But it hooks `pipeline.denoising_step(...)` to compute an x0 estimate at each
denoising step and decodes/saves it as an MP4.

Typical use:
  python baseline_intermediate_videos.py \
    --prompt "A bright red ball falling..." \
    --pipeline_config configs/ltxv-2b-0.9.6-dev.yaml \
    --height 512 --width 768 --num_frames 81 --frame_rate 16 --seed 2026 \
    --output_dir baseline_debug_run
"""

from __future__ import annotations

import argparse
import copy
import os
import gc
import sys
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple, List

import numpy as np
import torch
import imageio

from helper import decode_x0_to_video
from reward_functions import reward_function

try:
    # Preferred in this repo layout
    from ltx_video.ltx_video.inference import load_pipeline_config, create_ltx_video_pipeline
except ModuleNotFoundError:
    from ltx_video.inference import load_pipeline_config, create_ltx_video_pipeline  # type: ignore

from huggingface_hub import hf_hub_download

try:
    from ltx_video.models.autoencoders.causal_video_autoencoder import CausalVideoAutoencoder  # type: ignore
    from ltx_video.models.autoencoders.vae_encode import latent_to_pixel_coords  # type: ignore
except ModuleNotFoundError:
    from ltx_video.ltx_video.models.autoencoders.causal_video_autoencoder import (  # type: ignore
        CausalVideoAutoencoder,
    )
    from ltx_video.ltx_video.models.autoencoders.vae_encode import latent_to_pixel_coords  # type: ignore


class TeeLogger:
    """
    Mirror `pipeline.py`'s logger: tee stdout/stderr to a file (line-buffered).
    Captures prints + tracebacks so you can compare GRPO runs deterministically.
    """

    def __init__(self, filename: str):
        self.terminal = sys.stdout
        self.log = open(filename, "w", buffering=1, encoding="utf-8")

    def write(self, message: str):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        try:
            self.log.close()
        except Exception:
            pass


def _get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _calculate_padding(
    source_height: int, source_width: int, target_height: int, target_width: int
) -> Tuple[int, int, int, int]:
    """(left, right, top, bottom) padding amounts."""
    pad_height = target_height - source_height
    pad_width = target_width - source_width
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left
    return (pad_left, pad_right, pad_top, pad_bottom)


def _crop_video_btc_hw(
    video_btc_hw: torch.Tensor,
    *,
    num_frames: int,
    padding: Tuple[int, int, int, int],
) -> torch.Tensor:
    """
    Crop a decoded video in [B, T, C, H, W] using the same logic as inference.py.
    """
    pad_left, pad_right, pad_top, pad_bottom = padding

    # Convert to [B, C, T, H, W] to match inference.py slicing.
    v = video_btc_hw.permute(0, 2, 1, 3, 4)

    pad_bottom = -pad_bottom
    pad_right = -pad_right
    if pad_bottom == 0:
        pad_bottom = v.shape[3]
    if pad_right == 0:
        pad_right = v.shape[4]

    v = v[:, :, :num_frames, pad_top:pad_bottom, pad_left:pad_right]
    return v.permute(0, 2, 1, 3, 4)


def _save_video_to_mp4(video_btc_hw: torch.Tensor, out_path: Path, fps: int) -> None:
    """
    Save [B, T, C, H, W] in [0, 1] as mp4.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    v = video_btc_hw.detach().float().cpu()
    v = torch.nan_to_num(v, nan=0.0, posinf=1.0, neginf=0.0)
    v = v.clamp(0.0, 1.0)
    frames = (v[0].permute(0, 2, 3, 1).numpy() * 255.0).round().astype(np.uint8)  # [T,H,W,C]

    # Use explicit writer settings (matches the repo's other MP4 writers more closely than mimwrite).
    writer = imageio.get_writer(
        str(out_path),
        fps=int(fps),
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        macro_block_size=1,
        output_params=["-movflags", "+faststart", "-profile:v", "baseline"],
    )
    try:
        for fr in frames:
            writer.append_data(fr)
    finally:
        writer.close()

def _clear_gpu_cache(tag: str = "") -> None:
    """
    Best-effort VRAM cleanup to reduce fragmentation between phases (baseline -> GRPO).
    Note: this does NOT unload model weights; it mainly releases cached/reserved blocks.
    """
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
        if tag:
            try:
                free_gb, total_gb = (x / (1024**3) for x in torch.cuda.mem_get_info())
                print(f"🧹 Cleared CUDA cache after {tag}. Free VRAM: {free_gb:.1f} / {total_gb:.1f} GB")
            except Exception:
                print(f"🧹 Cleared CUDA cache after {tag}.")

def _compute_x0_est_like_pipeline_denoising_step(
    *,
    latents: torch.Tensor,
    noise_pred: torch.Tensor,
    current_timestep: torch.Tensor,
    t: float,
) -> torch.Tensor:
    """
    Match `LTXVideoPipeline.denoising_step(..., return_x0=True)` broadcasting logic:
      x0 = latents - scale(t) * noise_pred
    where `scale(t)` is reshaped to broadcast across `latents.ndim`.
    """
    effective_t = t if current_timestep is None else current_timestep
    if not torch.is_tensor(effective_t):
        effective_t = torch.tensor(effective_t, device=latents.device, dtype=latents.dtype)

    # Reshape for broadcasting over latents
    scale_shape = [1] * latents.ndim
    # If timestep is a batch tensor, broadcast along batch dim.
    if effective_t.ndim > 0:
        scale_shape[0] = effective_t.shape[0]
    scale = effective_t.reshape(scale_shape).to(device=latents.device, dtype=latents.dtype)
    return latents - scale * noise_pred


def _parse_int_list(s: str) -> List[int]:
    s = (s or "").strip()
    if not s:
        return []
    parts = [p.strip() for p in s.replace(",", " ").split()]
    out: List[int] = []
    for p in parts:
        if p:
            out.append(int(p))
    return out


def unfreeze_attention_blocks(
    model: torch.nn.Module,
    *,
    attn1_blocks: List[int],
    attn2_blocks: List[int],
    verbose: bool = True,
) -> List[torch.nn.Parameter]:
    """
    Freeze all params, then unfreeze q/k/v/out projections for selected self-attn (attn1)
    and cross-attn (attn2) blocks. Returns unfrozen params list.
    """
    if verbose:
        print("🎯 Unfreezing Strategy:")
        print(f"   Self-Attention (attn1) blocks: {attn1_blocks or '(none)'}")
        print(f"   Cross-Attention (attn2) blocks: {attn2_blocks or '(none)'}\n")

    for p in model.parameters():
        p.requires_grad = False

    attn1_pats = ("attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out")
    attn2_pats = ("attn2.to_q", "attn2.to_k", "attn2.to_v", "attn2.to_out")
    unfrozen: List[torch.nn.Parameter] = []

    for name, p in model.named_parameters():
        typ = None
        for b in attn1_blocks:
            if f"transformer_blocks.{b}." in name and any(x in name for x in attn1_pats) and "attn2" not in name:
                typ = "attn1"
                break
        if typ is None:
            for b in attn2_blocks:
                if f"transformer_blocks.{b}." in name and any(x in name for x in attn2_pats):
                    typ = "attn2"
                    break

        if typ is not None:
            p.requires_grad = True
            unfrozen.append(p)
            if verbose:
                print(f"  Unfreezing [{typ}]: {name} - {tuple(p.shape)}")

    if verbose:
        num_attn1 = sum(1 for n, p in model.named_parameters() if p.requires_grad and "attn1" in n)
        num_attn2 = sum(1 for n, p in model.named_parameters() if p.requires_grad and "attn2" in n)
        print("\n✅ Unfreezing Summary:")
        print(f"   Self-attention (attn1) params: {num_attn1}")
        print(f"   Cross-attention (attn2) params: {num_attn2}")
        print(f"   Total unfrozen parameters: {len(unfrozen)}")
        print(f"   Total param count: {sum(p.numel() for p in unfrozen):,}\n")

    return unfrozen

def run_grpo_for_prompt(
    *,
    pipeline: Any,
    prompt: str,
    out_dir: Path,
    height: int,
    width: int,
    num_frames: int,
    frame_rate: int,
    seed: int,
    num_inference_steps: int,
    num_grpo_steps: int,
    num_rollouts: int,
    lr: float,
    attn1_blocks: List[int],
    attn2_blocks: List[int],
    rollout_noise_scale: float,
    normalize_advantages: bool,
    use_grpo_kl: bool,
    kl_beta: float,
    # Conditioning knobs (match baseline __call__ behavior)
    negative_prompt: str,
    guidance_scale: float,
    stg_scale: float,
    rescaling_scale: float,
    cfg_star_rescale: bool,
    skip_layer_strategy: Any,
    skip_block_list: Any,
) -> None:
    """
    GRPO training loop mirroring pipeline.py, using the baseline pipeline object.
    Saves rollout MP4s per GRPO timestep and a final MP4 from the last updated x0.
    """
    device = pipeline.device if hasattr(pipeline, "device") else torch.device("cuda")
    model = pipeline.transformer

    # Timesteps
    pipeline.scheduler.set_timesteps(int(num_inference_steps), device=device)
    timesteps = pipeline.scheduler.timesteps
    if not (1 <= num_grpo_steps < len(timesteps)):
        raise ValueError(f"num_grpo_steps must be in [1,{len(timesteps)-1}], got {num_grpo_steps}")
    timesteps_for_grpo = timesteps if os.getenv("GRPO_FROM_START", "0") == "1" else timesteps[-num_grpo_steps:]
    num_standard_steps = 0 if os.getenv("GRPO_FROM_START", "0") == "1" else (len(timesteps) - num_grpo_steps)

    print(f"Total timesteps: {len(timesteps)} [{float(timesteps[0]):.4f} → {float(timesteps[-1]):.4f}]")
    print(
        f"GRPO training on: Last {num_grpo_steps} steps "
        f"[{float(timesteps_for_grpo[0]):.4f} → {float(timesteps_for_grpo[-1]):.4f}]"
    )
    print(f"Skipping first {num_standard_steps} steps (rough structure)\n")

    # Encode prompt
    (
        prompt_embeds,
        prompt_attention_mask,
        negative_prompt_embeds,
        negative_prompt_attention_mask,
    ) = pipeline.encode_prompt(
        prompt=prompt,
        device="cuda" if torch.cuda.is_available() else device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
        negative_prompt=negative_prompt,
    )

    # Match `LTXVideoPipeline.__call__`: always build a 3-way batch [neg, pos, pos].
    negative_prompt_embeds = (
        torch.zeros_like(prompt_embeds) if negative_prompt_embeds is None else negative_prompt_embeds
    )
    negative_prompt_attention_mask = (
        torch.zeros_like(prompt_attention_mask)
        if negative_prompt_attention_mask is None
        else negative_prompt_attention_mask
    )
    prompt_embeds_batch = torch.cat([negative_prompt_embeds, prompt_embeds, prompt_embeds], dim=0)
    prompt_attention_mask_batch = torch.cat(
        [negative_prompt_attention_mask, prompt_attention_mask, prompt_attention_mask], dim=0
    )

    # Compute latent shape (match pipeline.py)
    vae_scale_factor = pipeline.vae_scale_factor
    video_scale_factor = pipeline.video_scale_factor
    latent_height = height // vae_scale_factor
    latent_width = width // vae_scale_factor
    latent_frames = num_frames // video_scale_factor
    if isinstance(pipeline.vae, CausalVideoAutoencoder):
        latent_frames += 1
    latent_shape = (
        1,
        pipeline.vae.config.latent_channels,
        latent_frames,
        latent_height,
        latent_width,
    )

    # Reference baseline model (frozen)
    baseline_model = copy.deepcopy(model)
    baseline_model.eval()
    for p in baseline_model.parameters():
        p.requires_grad = False

    # Unfreeze + optimizer
    unfrozen_params = unfreeze_attention_blocks(
        model,
        attn1_blocks=attn1_blocks,
        attn2_blocks=attn2_blocks,
        verbose=True,
    )
    if not unfrozen_params:
        raise ValueError("No parameters were unfrozen. Check --attn1_blocks/--attn2_blocks.")

    # Track initial weights for comparison at the end (match pipeline.py).
    # Store on CPU to avoid doubling VRAM usage during training.
    initial_weights: Dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            initial_weights[name] = param.detach().float().cpu().clone()
    print(f"📊 Tracking {len(initial_weights)} unfrozen parameters\n")

    optimizer = torch.optim.Adam(unfrozen_params, lr=float(lr), betas=(0.9, 0.95), weight_decay=0.01)

    # Init latents (noise) and patchify
    gen = torch.Generator(device=device).manual_seed(int(seed))
    latents = pipeline.prepare_latents(
        latents=None,
        media_items=None,
        timestep=timesteps[0],
        latent_shape=latent_shape,
        dtype=torch.bfloat16,
        device=device,
        generator=gen,
    )
    latents, latent_coords = pipeline.patchifier.patchify(latents)

    pixel_coords = latent_to_pixel_coords(
        latent_coords,
        pipeline.vae,
        causal_fix=pipeline.transformer.config.causal_temporal_positioning,
    )
    indices_grid = pixel_coords.to(torch.float32)
    indices_grid[:, 0] *= (1.0 / frame_rate)

    # Phase 1: standard denoise with baseline_model
    for _, t in enumerate(timesteps[:num_standard_steps]):
        with torch.no_grad():
            # Match baseline conditioning (CFG/STG/rescaling) for phase-1 too.
            do_classifier_free_guidance = float(guidance_scale) > 1.0
            do_spatio_temporal_guidance = float(stg_scale) > 0.0
            do_rescaling = float(rescaling_scale) != 1.0

            num_conds = 1 + (1 if do_classifier_free_guidance else 0) + (1 if do_spatio_temporal_guidance else 0)
            if do_classifier_free_guidance and do_spatio_temporal_guidance:
                indices = slice(0, 3)
            elif do_classifier_free_guidance:
                indices = slice(0, 2)
            elif do_spatio_temporal_guidance:
                indices = slice(1, 3)
            else:
                indices = slice(1, 2)

            # Skip-layer mask for STG
            skip_layer_mask = None
            if do_spatio_temporal_guidance and skip_block_list is not None:
                # YAMLs typically provide a single list[int] (apply to all steps).
                skip_blocks_for_step = skip_block_list
                if isinstance(skip_block_list, list) and len(skip_block_list) > 0 and isinstance(skip_block_list[0], list):
                    skip_blocks_for_step = skip_block_list[min(0, len(skip_block_list) - 1)]
                if hasattr(baseline_model, "create_skip_layer_mask"):
                    skip_layer_mask = baseline_model.create_skip_layer_mask(
                        1, num_conds, num_conds - 1, skip_blocks_for_step
                    )

            batch_fractional_coords = torch.cat([indices_grid] * num_conds)
            latent_model_input = torch.cat([latents] * num_conds) if num_conds > 1 else latents
            latent_model_input = pipeline.scheduler.scale_model_input(latent_model_input, t)

            current_timestep = t
            if not torch.is_tensor(current_timestep):
                current_timestep = torch.tensor([current_timestep], device=latent_model_input.device, dtype=torch.float64)
            elif len(current_timestep.shape) == 0:
                current_timestep = current_timestep[None].to(latent_model_input.device)
            current_timestep = current_timestep.expand(latent_model_input.shape[0]).unsqueeze(-1)

            noise_pred_all = baseline_model(
                latent_model_input.to(baseline_model.dtype),
                indices_grid=batch_fractional_coords,
                encoder_hidden_states=prompt_embeds_batch[indices].to(baseline_model.dtype),
                encoder_attention_mask=prompt_attention_mask_batch[indices],
                timestep=current_timestep,
                skip_layer_mask=skip_layer_mask,
                skip_layer_strategy=skip_layer_strategy,
                return_dict=False,
            )[0]

            # Guidance (CFG + STG) and rescaling, identical to `LTXVideoPipeline.__call__`.
            if do_spatio_temporal_guidance:
                noise_pred_text, noise_pred_text_perturb = noise_pred_all.chunk(num_conds)[-2:]
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred_all.chunk(num_conds)[:2]
                if cfg_star_rescale:
                    positive_flat = noise_pred_text.view(1, -1)
                    negative_flat = noise_pred_uncond.view(1, -1)
                    dot_product = torch.sum(positive_flat * negative_flat, dim=1, keepdim=True)
                    squared_norm = torch.sum(negative_flat**2, dim=1, keepdim=True) + 1e-8
                    alpha = dot_product / squared_norm
                    noise_pred_uncond = alpha * noise_pred_uncond
                noise_pred = noise_pred_uncond + float(guidance_scale) * (noise_pred_text - noise_pred_uncond)
            elif do_spatio_temporal_guidance:
                noise_pred = noise_pred_text
            else:
                noise_pred = noise_pred_all.chunk(num_conds)[0]

            if do_spatio_temporal_guidance:
                noise_pred = noise_pred + float(stg_scale) * (noise_pred_text - noise_pred_text_perturb)
                if do_rescaling and float(stg_scale) > 0.0:
                    noise_pred_text_std = noise_pred_text.view(1, -1).std(dim=1, keepdim=True)
                    noise_pred_std = noise_pred.view(1, -1).std(dim=1, keepdim=True)
                    factor = noise_pred_text_std / noise_pred_std
                    factor = float(rescaling_scale) * factor + (1 - float(rescaling_scale))
                    noise_pred = noise_pred * factor.view(1, 1, 1)

            current_timestep = current_timestep[:1]
            if (baseline_model.config.out_channels // 2) == baseline_model.config.in_channels:
                noise_pred = noise_pred.chunk(2, dim=1)[0]

            latents = pipeline.denoising_step(
                latents=latents,
                noise_pred=noise_pred,
                current_timestep=current_timestep,
                conditioning_mask=None,
                t=t,
                extra_step_kwargs={},
                stochastic_sampling=False,
            )

    rollout_dir = out_dir / "intermediate_rollout"
    (rollout_dir / "logs").mkdir(parents=True, exist_ok=True)
    last_updated_x0_cpu = None

    # Phase 2: GRPO steps
    for step_idx, t in enumerate(timesteps_for_grpo):
        print(f"GRPO Step {step_idx+1:02d}/{len(timesteps_for_grpo)} | t={float(t):.4f}", end="")

        rollout_noise_preds: List[torch.Tensor] = []
        rollout_rewards: List[torch.Tensor] = []
        rollout_paths: List[str] = []

        for r in range(int(num_rollouts)):
            with torch.no_grad():
                rollout_seed = int(seed) + int(step_idx) * 1000 + int(r)
                torch.manual_seed(rollout_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(rollout_seed)

                latents_perturbed = latents
                if r > 0 and float(rollout_noise_scale) > 0.0:
                    latents_perturbed = latents + torch.randn_like(latents) * float(rollout_noise_scale)

                do_classifier_free_guidance = float(guidance_scale) > 1.0
                do_spatio_temporal_guidance = float(stg_scale) > 0.0
                do_rescaling = float(rescaling_scale) != 1.0

                num_conds = 1 + (1 if do_classifier_free_guidance else 0) + (1 if do_spatio_temporal_guidance else 0)
                if do_classifier_free_guidance and do_spatio_temporal_guidance:
                    indices = slice(0, 3)
                elif do_classifier_free_guidance:
                    indices = slice(0, 2)
                elif do_spatio_temporal_guidance:
                    indices = slice(1, 3)
                else:
                    indices = slice(1, 2)

                skip_layer_mask = None
                if do_spatio_temporal_guidance and skip_block_list is not None and hasattr(model, "create_skip_layer_mask"):
                    skip_blocks_for_step = skip_block_list
                    if isinstance(skip_block_list, list) and len(skip_block_list) > 0 and isinstance(skip_block_list[0], list):
                        skip_blocks_for_step = skip_block_list[min(step_idx, len(skip_block_list) - 1)]
                    skip_layer_mask = model.create_skip_layer_mask(1, num_conds, num_conds - 1, skip_blocks_for_step)

                batch_fractional_coords = torch.cat([indices_grid] * num_conds)
                latent_model_input = (
                    torch.cat([latents_perturbed] * num_conds) if num_conds > 1 else latents_perturbed
                )
                latent_model_input = pipeline.scheduler.scale_model_input(latent_model_input, t)

                current_timestep = t
                if not torch.is_tensor(current_timestep):
                    current_timestep = torch.tensor([current_timestep], device=latent_model_input.device, dtype=torch.float64)
                elif len(current_timestep.shape) == 0:
                    current_timestep = current_timestep[None].to(latent_model_input.device)
                current_timestep = current_timestep.expand(latent_model_input.shape[0]).unsqueeze(-1)

                noise_pred_all = model(
                    latent_model_input.to(model.dtype),
                    indices_grid=batch_fractional_coords,
                    encoder_hidden_states=prompt_embeds_batch[indices].to(model.dtype),
                    encoder_attention_mask=prompt_attention_mask_batch[indices],
                    timestep=current_timestep,
                    skip_layer_mask=skip_layer_mask,
                    skip_layer_strategy=skip_layer_strategy,
                    return_dict=False,
                )[0]

                if do_spatio_temporal_guidance:
                    noise_pred_text, noise_pred_text_perturb = noise_pred_all.chunk(num_conds)[-2:]
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred_all.chunk(num_conds)[:2]
                    if cfg_star_rescale:
                        positive_flat = noise_pred_text.view(1, -1)
                        negative_flat = noise_pred_uncond.view(1, -1)
                        dot_product = torch.sum(positive_flat * negative_flat, dim=1, keepdim=True)
                        squared_norm = torch.sum(negative_flat**2, dim=1, keepdim=True) + 1e-8
                        alpha = dot_product / squared_norm
                        noise_pred_uncond = alpha * noise_pred_uncond
                    noise_pred = noise_pred_uncond + float(guidance_scale) * (noise_pred_text - noise_pred_uncond)
                elif do_spatio_temporal_guidance:
                    noise_pred = noise_pred_text
                else:
                    noise_pred = noise_pred_all.chunk(num_conds)[0]

                if do_spatio_temporal_guidance:
                    noise_pred = noise_pred + float(stg_scale) * (noise_pred_text - noise_pred_text_perturb)
                    if do_rescaling and float(stg_scale) > 0.0:
                        noise_pred_text_std = noise_pred_text.view(1, -1).std(dim=1, keepdim=True)
                        noise_pred_std = noise_pred.view(1, -1).std(dim=1, keepdim=True)
                        factor = noise_pred_text_std / noise_pred_std
                        factor = float(rescaling_scale) * factor + (1 - float(rescaling_scale))
                        noise_pred = noise_pred * factor.view(1, 1, 1)

                current_timestep = current_timestep[:1]
                if (model.config.out_channels // 2) == model.config.in_channels:
                    noise_pred = noise_pred.chunk(2, dim=1)[0]

                rollout_noise_preds.append(noise_pred.detach().clone())

                # `LTXVideoPipeline.denoising_step` in this repo does NOT support `return_x0`;
                # compute x0 estimate explicitly and call denoising_step for next_latents only.
                x0_est = _compute_x0_est_like_pipeline_denoising_step(
                    latents=latents_perturbed,
                    noise_pred=noise_pred,
                    current_timestep=current_timestep,
                    t=t,
                )
                _ = pipeline.denoising_step(
                    latents=latents_perturbed,
                    noise_pred=noise_pred,
                    current_timestep=current_timestep,
                    conditioning_mask=None,
                    t=t,
                    extra_step_kwargs={},
                    stochastic_sampling=True,
                )

                video_x0 = decode_x0_to_video(
                    x0_est,
                    pipeline,
                    num_frames=num_frames,
                    height=height,
                    width=width,
                    is_patchified=True,
                )
                out_mp4 = rollout_dir / f"rollout_step{step_idx:03d}_r{r:02d}.mp4"
                _save_video_to_mp4(video_x0, out_mp4, fps=frame_rate)
                rollout_paths.append(str(out_mp4))

                reward = reward_function(video_x0.float(), prompt=prompt)
                reward_t = reward if torch.is_tensor(reward) else torch.tensor(float(reward), device=device)
                rollout_rewards.append(reward_t.to(dtype=torch.float32))

                del video_x0, x0_est
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        rewards = torch.stack(rollout_rewards)
        mean_r = rewards.mean()
        std_r = rewards.std()
        if normalize_advantages and std_r > 1e-8:
            adv = (rewards - mean_r) / (std_r + 1e-4)
        else:
            adv = rewards - mean_r

        with (rollout_dir / "logs" / f"timestep_{step_idx:03d}.txt").open("w") as f:
            f.write(f"timestep_index={step_idx}\n")
            f.write(f"t={float(t):.6f}\n")
            f.write(f"mean_reward={float(mean_r):.6f}\n")
            f.write(f"std_reward={float(std_r):.6f}\n")
            for r in range(int(num_rollouts)):
                f.write(
                    f"r={r} reward={float(rewards[r]):.6f} advantage={float(adv[r]):.6f} mp4={rollout_paths[r]}\n"
                )

        # Step 3: policy gradient update
        optimizer.zero_grad()
        model.zero_grad()

        latents_for_loss = latents.detach().clone().requires_grad_(True)
        # IMPORTANT: for the loss we compute the *guided* noise prediction too (to match baseline conditioning).
        do_classifier_free_guidance = float(guidance_scale) > 1.0
        do_spatio_temporal_guidance = float(stg_scale) > 0.0
        do_rescaling = float(rescaling_scale) != 1.0
        num_conds = 1 + (1 if do_classifier_free_guidance else 0) + (1 if do_spatio_temporal_guidance else 0)
        if do_classifier_free_guidance and do_spatio_temporal_guidance:
            indices = slice(0, 3)
        elif do_classifier_free_guidance:
            indices = slice(0, 2)
        elif do_spatio_temporal_guidance:
            indices = slice(1, 3)
        else:
            indices = slice(1, 2)

        skip_layer_mask = None
        if do_spatio_temporal_guidance and skip_block_list is not None and hasattr(model, "create_skip_layer_mask"):
            skip_blocks_for_step = skip_block_list
            if isinstance(skip_block_list, list) and len(skip_block_list) > 0 and isinstance(skip_block_list[0], list):
                skip_blocks_for_step = skip_block_list[min(step_idx, len(skip_block_list) - 1)]
            skip_layer_mask = model.create_skip_layer_mask(1, num_conds, num_conds - 1, skip_blocks_for_step)

        batch_fractional_coords = torch.cat([indices_grid.detach()] * num_conds)
        latent_model_input = torch.cat([latents_for_loss] * num_conds) if num_conds > 1 else latents_for_loss
        latent_model_input = pipeline.scheduler.scale_model_input(latent_model_input, t)

        current_timestep = t
        if not torch.is_tensor(current_timestep):
            current_timestep = torch.tensor([current_timestep], device=latent_model_input.device, dtype=torch.float64)
        elif len(current_timestep.shape) == 0:
            current_timestep = current_timestep[None].to(latent_model_input.device)
        current_timestep = current_timestep.expand(latent_model_input.shape[0]).unsqueeze(-1)

        noise_pred_all = model(
            latent_model_input.to(model.dtype),
            indices_grid=batch_fractional_coords,
            encoder_hidden_states=prompt_embeds_batch[indices].detach().to(model.dtype),
            encoder_attention_mask=prompt_attention_mask_batch[indices].detach(),
            timestep=current_timestep,
            skip_layer_mask=skip_layer_mask,
            skip_layer_strategy=skip_layer_strategy,
            return_dict=False,
        )[0]

        if do_spatio_temporal_guidance:
            noise_pred_text, noise_pred_text_perturb = noise_pred_all.chunk(num_conds)[-2:]
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred_all.chunk(num_conds)[:2]
            noise_pred_current = noise_pred_uncond + float(guidance_scale) * (noise_pred_text - noise_pred_uncond)
        elif do_spatio_temporal_guidance:
            noise_pred_current = noise_pred_text
        else:
            noise_pred_current = noise_pred_all.chunk(num_conds)[0]

        if do_spatio_temporal_guidance:
            noise_pred_current = noise_pred_current + float(stg_scale) * (noise_pred_text - noise_pred_text_perturb)
            if do_rescaling and float(stg_scale) > 0.0:
                noise_pred_text_std = noise_pred_text.view(1, -1).std(dim=1, keepdim=True)
                noise_pred_std = noise_pred_current.view(1, -1).std(dim=1, keepdim=True)
                factor = noise_pred_text_std / noise_pred_std
                factor = float(rescaling_scale) * factor + (1 - float(rescaling_scale))
                noise_pred_current = noise_pred_current * factor.view(1, 1, 1)

        if (model.config.out_channels // 2) == model.config.in_channels:
            noise_pred_current = noise_pred_current.chunk(2, dim=1)[0]

        log_probs = []
        for r in range(int(num_rollouts)):
            mse = ((rollout_noise_preds[r].detach() - noise_pred_current) ** 2).mean()
            log_probs.append(-mse)
        log_probs_t = torch.stack(log_probs)
        pg_loss = -(log_probs_t * adv).mean()

        kl_loss = torch.tensor(0.0, device=pg_loss.device)
        if use_grpo_kl and float(kl_beta) > 0.0:
            with torch.no_grad():
                # Reference uses same conditioning style (CFG/STG/rescaling) as the trained model.
                do_classifier_free_guidance = float(guidance_scale) > 1.0
                do_spatio_temporal_guidance = float(stg_scale) > 0.0
                do_rescaling = float(rescaling_scale) != 1.0
                num_conds = 1 + (1 if do_classifier_free_guidance else 0) + (1 if do_spatio_temporal_guidance else 0)
                if do_classifier_free_guidance and do_spatio_temporal_guidance:
                    indices = slice(0, 3)
                elif do_classifier_free_guidance:
                    indices = slice(0, 2)
                elif do_spatio_temporal_guidance:
                    indices = slice(1, 3)
                else:
                    indices = slice(1, 2)

                skip_layer_mask = None
                if do_spatio_temporal_guidance and skip_block_list is not None and hasattr(baseline_model, "create_skip_layer_mask"):
                    skip_blocks_for_step = skip_block_list
                    if isinstance(skip_block_list, list) and len(skip_block_list) > 0 and isinstance(skip_block_list[0], list):
                        skip_blocks_for_step = skip_block_list[min(step_idx, len(skip_block_list) - 1)]
                    skip_layer_mask = baseline_model.create_skip_layer_mask(1, num_conds, num_conds - 1, skip_blocks_for_step)

                batch_fractional_coords = torch.cat([indices_grid.detach()] * num_conds)
                latent_model_input = torch.cat([latents.detach()] * num_conds) if num_conds > 1 else latents.detach()
                latent_model_input = pipeline.scheduler.scale_model_input(latent_model_input, t)

                current_timestep = t
                if not torch.is_tensor(current_timestep):
                    current_timestep = torch.tensor([current_timestep], device=latent_model_input.device, dtype=torch.float64)
                elif len(current_timestep.shape) == 0:
                    current_timestep = current_timestep[None].to(latent_model_input.device)
                current_timestep = current_timestep.expand(latent_model_input.shape[0]).unsqueeze(-1)

                noise_pred_all = baseline_model(
                    latent_model_input.to(baseline_model.dtype),
                    indices_grid=batch_fractional_coords,
                    encoder_hidden_states=prompt_embeds_batch[indices].detach().to(baseline_model.dtype),
                    encoder_attention_mask=prompt_attention_mask_batch[indices].detach(),
                    timestep=current_timestep,
                    skip_layer_mask=skip_layer_mask,
                    skip_layer_strategy=skip_layer_strategy,
                    return_dict=False,
                )[0]

                if do_spatio_temporal_guidance:
                    noise_pred_text, noise_pred_text_perturb = noise_pred_all.chunk(num_conds)[-2:]
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred_all.chunk(num_conds)[:2]
                    noise_pred_ref = noise_pred_uncond + float(guidance_scale) * (noise_pred_text - noise_pred_uncond)
                elif do_spatio_temporal_guidance:
                    noise_pred_ref = noise_pred_text
                else:
                    noise_pred_ref = noise_pred_all.chunk(num_conds)[0]

                if do_spatio_temporal_guidance:
                    noise_pred_ref = noise_pred_ref + float(stg_scale) * (noise_pred_text - noise_pred_text_perturb)
                    if do_rescaling and float(stg_scale) > 0.0:
                        noise_pred_text_std = noise_pred_text.view(1, -1).std(dim=1, keepdim=True)
                        noise_pred_std = noise_pred_ref.view(1, -1).std(dim=1, keepdim=True)
                        factor = noise_pred_text_std / noise_pred_std
                        factor = float(rescaling_scale) * factor + (1 - float(rescaling_scale))
                        noise_pred_ref = noise_pred_ref * factor.view(1, 1, 1)

                if (baseline_model.config.out_channels // 2) == baseline_model.config.in_channels:
                    noise_pred_ref = noise_pred_ref.chunk(2, dim=1)[0]
            kl_loss = ((noise_pred_current - noise_pred_ref) ** 2).mean()

        loss = pg_loss + (float(kl_beta) * kl_loss)
        loss.backward()

        # Match pipeline.py logging: grad norms + weight deltas per update.
        grad_norms: List[float] = []
        for param in optimizer.param_groups[0]["params"]:
            if param.grad is not None:
                grad_norms.append(float(param.grad.norm().item()))
        avg_grad_norm = (sum(grad_norms) / len(grad_norms)) if grad_norms else 0.0

        total_grad_norm = torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]["params"], 0.5)

        # Track weights BEFORE update
        weights_before: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                weights_before[name] = param.detach().float().cpu().clone()

        optimizer.step()

        # Track weights AFTER update and compute mean absolute change
        weight_changes: List[float] = []
        for name, param in model.named_parameters():
            if param.requires_grad and name in weights_before:
                after = param.detach().float().cpu()
                change = (after - weights_before[name]).abs().mean().item()
                weight_changes.append(float(change))
        avg_weight_change = (sum(weight_changes) / len(weight_changes)) if weight_changes else 0.0

        print(
            f"  ✅ Weights updated! grad_norm={avg_grad_norm:.6f}, "
            f"total_grad_norm={float(total_grad_norm):.6f}, weight_Δ={avg_weight_change:.6f}"
        )

        # Step 4: advance latents with updated model (deterministic)
        with torch.no_grad():
            # Advance latents using the same guided noise prediction style.
            do_classifier_free_guidance = float(guidance_scale) > 1.0
            do_spatio_temporal_guidance = float(stg_scale) > 0.0
            do_rescaling = float(rescaling_scale) != 1.0
            num_conds = 1 + (1 if do_classifier_free_guidance else 0) + (1 if do_spatio_temporal_guidance else 0)
            if do_classifier_free_guidance and do_spatio_temporal_guidance:
                indices = slice(0, 3)
            elif do_classifier_free_guidance:
                indices = slice(0, 2)
            elif do_spatio_temporal_guidance:
                indices = slice(1, 3)
            else:
                indices = slice(1, 2)

            skip_layer_mask = None
            if do_spatio_temporal_guidance and skip_block_list is not None and hasattr(model, "create_skip_layer_mask"):
                skip_blocks_for_step = skip_block_list
                if isinstance(skip_block_list, list) and len(skip_block_list) > 0 and isinstance(skip_block_list[0], list):
                    skip_blocks_for_step = skip_block_list[min(step_idx, len(skip_block_list) - 1)]
                skip_layer_mask = model.create_skip_layer_mask(1, num_conds, num_conds - 1, skip_blocks_for_step)

            batch_fractional_coords = torch.cat([indices_grid] * num_conds)
            latent_model_input = torch.cat([latents] * num_conds) if num_conds > 1 else latents
            latent_model_input = pipeline.scheduler.scale_model_input(latent_model_input, t)

            current_timestep = t
            if not torch.is_tensor(current_timestep):
                current_timestep = torch.tensor([current_timestep], device=latent_model_input.device, dtype=torch.float64)
            elif len(current_timestep.shape) == 0:
                current_timestep = current_timestep[None].to(latent_model_input.device)
            current_timestep = current_timestep.expand(latent_model_input.shape[0]).unsqueeze(-1)

            noise_pred_all = model(
                latent_model_input.to(model.dtype),
                indices_grid=batch_fractional_coords,
                encoder_hidden_states=prompt_embeds_batch[indices].to(model.dtype),
                encoder_attention_mask=prompt_attention_mask_batch[indices],
                timestep=current_timestep,
                skip_layer_mask=skip_layer_mask,
                skip_layer_strategy=skip_layer_strategy,
                return_dict=False,
            )[0]

            if do_spatio_temporal_guidance:
                noise_pred_text, noise_pred_text_perturb = noise_pred_all.chunk(num_conds)[-2:]
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred_all.chunk(num_conds)[:2]
                noise_pred_u = noise_pred_uncond + float(guidance_scale) * (noise_pred_text - noise_pred_uncond)
            elif do_spatio_temporal_guidance:
                noise_pred_u = noise_pred_text
            else:
                noise_pred_u = noise_pred_all.chunk(num_conds)[0]

            if do_spatio_temporal_guidance:
                noise_pred_u = noise_pred_u + float(stg_scale) * (noise_pred_text - noise_pred_text_perturb)
                if do_rescaling and float(stg_scale) > 0.0:
                    noise_pred_text_std = noise_pred_text.view(1, -1).std(dim=1, keepdim=True)
                    noise_pred_std = noise_pred_u.view(1, -1).std(dim=1, keepdim=True)
                    factor = noise_pred_text_std / noise_pred_std
                    factor = float(rescaling_scale) * factor + (1 - float(rescaling_scale))
                    noise_pred_u = noise_pred_u * factor.view(1, 1, 1)

            current_timestep = current_timestep[:1]
            if (model.config.out_channels // 2) == model.config.in_channels:
                noise_pred_u = noise_pred_u.chunk(2, dim=1)[0]
            # Advance latents; compute x0 estimate explicitly (no return_x0 in this pipeline).
            x0_u = _compute_x0_est_like_pipeline_denoising_step(
                latents=latents,
                noise_pred=noise_pred_u,
                current_timestep=current_timestep,
                t=t,
            )
            next_latents = pipeline.denoising_step(
                latents=latents,
                noise_pred=noise_pred_u,
                current_timestep=current_timestep,
                conditioning_mask=None,
                t=t,
                extra_step_kwargs={},
                stochastic_sampling=False,
            )
            last_updated_x0_cpu = x0_u.detach().float().cpu()
            latents = next_latents.detach().clone()

        print(f" [loss={float(loss):.6f}, meanR={float(mean_r):.4f}] ✅")

        # Best-effort cleanup between steps
        del weights_before
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Final decode from last updated x0
    if last_updated_x0_cpu is not None:
        final_dir = out_dir / "final_videos"
        final_dir.mkdir(parents=True, exist_ok=True)
        x0_cuda = last_updated_x0_cpu.to(device=device, dtype=torch.bfloat16)
        final_video = decode_x0_to_video(
            x0_cuda,
            pipeline,
            num_frames=num_frames,
            height=height,
            width=width,
            is_patchified=True,
        )
        _save_video_to_mp4(final_video, final_dir / "final_video_updated_x0.mp4", fps=frame_rate)

    # End-of-run: report total weight change vs initial (match pipeline.py)
    total_changes: List[float] = []
    print("\n📈 Total weight change vs initial (per param):")
    for name, param in model.named_parameters():
        if param.requires_grad and name in initial_weights:
            cur = param.detach().float().cpu()
            total_change = (cur - initial_weights[name]).abs().mean().item()
            total_changes.append(float(total_change))
            print(f"  {name:60s} Δ={total_change:.8f}")
    avg_total_change = (sum(total_changes) / len(total_changes)) if total_changes else 0.0
    print(f"\n  Average total weight change: {avg_total_change:.8f}\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode",
        type=str,
        default="baseline_intermediates",
        choices=["baseline_intermediates", "grpo_train", "both"],
        help=(
            "baseline_intermediates: save x0 per denoising step (baseline pipeline call). "
            "grpo_train: run GRPO loop like pipeline.py. "
            "both: run baseline_intermediates first, then GRPO, saving under the same output_dir."
        ),
    )
    ap.add_argument("--prompt", type=str, default=None, help="Text prompt.")
    ap.add_argument("--prompt_file", type=str, default=None, help="Optional file of prompts (one per line).")
    ap.add_argument("--pipeline_config", type=str, default="configs/ltxv-2b-0.9.6-dev.yaml")
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--num_frames", type=int, default=81)
    ap.add_argument("--frame_rate", type=int, default=16)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument(
        "--output_dir",
        type=str,
        default="grpo_baseline{timestamp}",
        help=(
            "Output directory. If it contains '{timestamp}', it will be replaced with the current run timestamp. "
            "If it does not contain a timestamp, one will be appended automatically (unless --no_timestamp)."
        ),
    )
    ap.add_argument(
        "--no_timestamp",
        action="store_true",
        help="Do not expand/append timestamps to --output_dir (use the path exactly as provided).",
    )
    ap.add_argument("--save_every", type=int, default=1, help="Save every N denoising steps (1=all).")
    ap.add_argument("--stochastic_sampling", action="store_true", help="Enable stochastic sampling in pipeline.")
    # GRPO args (only used for --mode grpo_train)
    ap.add_argument("--num_inference_steps", type=int, default=40)
    ap.add_argument("--num_grpo_steps", type=int, default=25)
    ap.add_argument("--num_rollouts", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--attn1_blocks", type=str, default="11,12,13,14")
    ap.add_argument("--attn2_blocks", type=str, default="27")
    ap.add_argument("--rollout_noise_scale", type=float, default=0.5)
    ap.add_argument("--normalize_advantages", type=int, default=1)
    ap.add_argument("--use_grpo_kl", type=int, default=0)
    ap.add_argument("--kl_beta", type=float, default=0.0)
    ap.add_argument(
        "--negative_prompt",
        type=str,
        default="worst quality, inconsistent motion, blurry, jittery, distorted",
        help="Negative prompt (baseline default). Use '' to disable.",
    )
    ap.add_argument(
        "--match-grpo-conditioning",
        action="store_true",
        help=(
            "Make the baseline pipeline(...) call use the same conditioning style as the GRPO loop: "
            "guidance_scale=1, stg_scale=0, rescaling_scale=1, negative_prompt=''. "
            "This helps baseline intermediate videos look closer to GRPO rollouts."
        ),
    )
    args = ap.parse_args()

    prompts: list[str] = []
    if args.prompt_file:
        p = Path(args.prompt_file)
        if not p.exists():
            raise FileNotFoundError(f"--prompt_file not found: {p}")
        prompts = [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]
    if args.prompt:
        prompts = [args.prompt] if not prompts else prompts
    if not prompts:
        raise ValueError("Provide --prompt or --prompt_file.")

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir_str = str(args.output_dir)
    if not args.no_timestamp:
        if "{timestamp}" in out_dir_str:
            out_dir_str = out_dir_str.replace("{timestamp}", run_ts)
        else:
            # If the user already provided a timestamp-like substring, don't append another.
            if re.search(r"\d{8}_\d{6}", out_dir_str) is None:
                out_dir_str = f"{out_dir_str}_{run_ts}"
    out_root = Path(out_dir_str)
    out_root.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------
    # Load YAML + resolve checkpoint (local path, CKPT_PATH override, or HF download)
    # ---------------------------------------------------------------------
    config_path = args.pipeline_config
    cfg = load_pipeline_config(config_path)
    ckpt_name = cfg["checkpoint_path"]

    def resolve_checkpoint(name: str) -> str | None:
        """
        Resolve a checkpoint reference.
        - If `name` is an existing local path, return it.
        - Otherwise, try to download from HF hub using the basename (handles YAMLs
          that store absolute paths).
        """
        try:
            p = Path(name)
            if p.exists() and p.is_file():
                print(f"Using local checkpoint: {p}")
                return str(p)

            candidate = p.name  # basename for HF hub download
            print(f"Attempting HF download checkpoint: {candidate} (from {name})")
            return hf_hub_download("Lightricks/LTX-Video", candidate)
        except Exception as e:
            print(f"⚠️  Download failed for {name}: {e}")
            return None

    ckpt_path = os.getenv("CKPT_PATH") or (
        ckpt_name if os.path.isfile(ckpt_name) else resolve_checkpoint(ckpt_name)
    )

    if ckpt_path is None or not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"Could not locate or download the checkpoint: {ckpt_name}\n"
            f"Resolved path: {ckpt_path}\n"
            "Fix options:\n"
            "  - Download the checkpoint file into the path specified by the YAML, or\n"
            "  - Set CKPT_PATH=/path/to/checkpoint.safetensors, or\n"
            "  - Switch --pipeline_config to a YAML that points to an existing checkpoint."
        )

    # Match inference.py padding rules
    height_padded = ((args.height - 1) // 32 + 1) * 32
    width_padded = ((args.width - 1) // 32 + 1) * 32
    num_frames_padded = ((args.num_frames - 2) // 8 + 1) * 8 + 1
    padding = _calculate_padding(args.height, args.width, height_padded, width_padded)

    device = _get_device()
    precision = cfg["precision"]
    text_encoder_model_name_or_path = cfg["text_encoder_model_name_or_path"]
    sampler = cfg.get("sampler", None)

    pipeline = create_ltx_video_pipeline(
        ckpt_path=ckpt_path,
        precision=precision,
        text_encoder_model_name_or_path=text_encoder_model_name_or_path,
        sampler=sampler,
        device=device,
        enhance_prompt=False,
        prompt_enhancer_image_caption_model_name_or_path=cfg.get(
            "prompt_enhancer_image_caption_model_name_or_path"
        ),
        prompt_enhancer_llm_model_name_or_path=cfg.get(
            "prompt_enhancer_llm_model_name_or_path"
        ),
    )

    # Enable VAE tiling/slicing for memory reduction (best-effort).
    if hasattr(pipeline.vae, "enable_tiling"):
        try:
            pipeline.vae.enable_tiling()
            print("✅ VAE tiling enabled")
        except Exception as e:
            print(f"⚠️  VAE tiling failed: {e}")
    if hasattr(pipeline.vae, "enable_slicing"):
        try:
            pipeline.vae.enable_slicing()
            print("✅ VAE slicing enabled (decodes in chunks)")
        except Exception as e:
            print(f"⚠️  VAE slicing failed: {e}")

    print("✅ Pipeline loaded!\n")

    # We will pass through the YAML config kwargs to pipeline(...) below.
    # `inference.py` removes stg_mode before passing; replicate that.
    stg_mode = cfg.get("stg_mode", "attention_values")
    cfg_for_call: Dict[str, Any] = dict(cfg)
    cfg_for_call.pop("stg_mode", None)
    cfg_for_call.pop("checkpoint_path", None)
    cfg_for_call.pop("precision", None)
    cfg_for_call.pop("text_encoder_model_name_or_path", None)
    cfg_for_call.pop("prompt_enhancer_image_caption_model_name_or_path", None)
    cfg_for_call.pop("prompt_enhancer_llm_model_name_or_path", None)
    # Avoid passing duplicate kwargs: we explicitly pass these in the pipeline(...) call below.
    for k in (
        "prompt",
        "negative_prompt",
        "height",
        "width",
        "num_frames",
        "frame_rate",
        "generator",
        "output_type",
        "callback_on_step_end",
        "media_items",
        "conditioning_items",
        "is_video",
        "vae_per_channel_normalize",
        "image_cond_noise_scale",
        "mixed_precision",
        "offload_to_cpu",
        "device",
        "enhance_prompt",
        "stochastic_sampling",
        "skip_layer_strategy",
    ):
        cfg_for_call.pop(k, None)

    # Choose skip_layer_strategy same way inference.py does (default is attention_values).
    try:
        from ltx_video.utils.skip_layer_strategy import SkipLayerStrategy  # type: ignore
    except ModuleNotFoundError:
        from ltx_video.ltx_video.utils.skip_layer_strategy import SkipLayerStrategy  # type: ignore

    stg = stg_mode.lower()
    if stg in ("stg_av", "attention_values"):
        skip_layer_strategy = SkipLayerStrategy.AttentionValues
    elif stg in ("stg_as", "attention_skip"):
        skip_layer_strategy = SkipLayerStrategy.AttentionSkip
    elif stg in ("stg_r", "residual"):
        skip_layer_strategy = SkipLayerStrategy.Residual
    elif stg in ("stg_t", "transformer_block"):
        skip_layer_strategy = SkipLayerStrategy.TransformerBlock
    else:
        raise ValueError(f"Invalid stg_mode: {stg_mode}")

    generator = torch.Generator(device=device).manual_seed(args.seed)

    # Hook denoising_step to capture x0 each step (rebuilt per prompt so the output dir is correct).
    orig_denoising_step = pipeline.denoising_step

    def make_hook(*, save_root: Path) -> Any:
        step_state = {"i": 0}

        def hooked_denoising_step(
            latents: torch.Tensor,
            noise_pred: torch.Tensor,
            current_timestep: torch.Tensor,
            conditioning_mask: torch.Tensor,
            t: float,
            extra_step_kwargs,
            t_eps: float = 1e-6,
            stochastic_sampling: bool = False,
            return_x0: bool = False,
        ):
            i = int(step_state["i"])

            # Compute x0 estimate (same formula used in pipeline_ltx_video.denoising_step when return_x0=True).
            x0_est = _compute_x0_est_like_pipeline_denoising_step(
                latents=latents,
                noise_pred=noise_pred,
                current_timestep=current_timestep,
                t=t,
            )

            if args.save_every > 0 and (i % args.save_every == 0):
                video = decode_x0_to_video(
                    x0_est,
                    pipeline,
                    num_frames=args.num_frames,
                    height=height_padded,
                    width=width_padded,
                    is_patchified=True,
                )
                video = _crop_video_btc_hw(video, num_frames=args.num_frames, padding=padding)

                out_path = save_root / "intermediate_steps" / f"step_{i:03d}_t{float(t):.4f}.mp4"
                _save_video_to_mp4(video, out_path, fps=args.frame_rate)

                del video
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            step_state["i"] = i + 1
            # IMPORTANT: the pipeline's internal denoising loop expects a *Tensor* latents output.
            # Returning a tuple (latents, x0_est) will break `torch.cat([latents] * num_conds)` inside __call__.
            out = orig_denoising_step(
                latents,
                noise_pred,
                current_timestep,
                conditioning_mask,
                t,
                extra_step_kwargs,
                t_eps=t_eps,
                stochastic_sampling=stochastic_sampling,
            )
            if isinstance(out, tuple):
                out = out[0]
            return out

        return hooked_denoising_step

    try:
        for idx, prompt in enumerate(prompts):
            prompt_dir = out_root / f"p{idx:03d}"
            prompt_dir.mkdir(parents=True, exist_ok=True)
            (prompt_dir / "prompt.txt").write_text(prompt + "\n")

            def _run_baseline_intermediates(save_base: Path) -> None:
                # Save intermediates into per-prompt folder
                (save_base / "intermediate_steps").mkdir(parents=True, exist_ok=True)
                # Monkey-patch denoising_step only for the duration of this baseline run.
                # (GRPO mode relies on the real denoising_step behavior; this repo's denoising_step does not return x0.)
                _orig_step = pipeline.denoising_step
                pipeline.denoising_step = make_hook(save_root=save_base)

                # Reset generator per prompt for reproducibility.
                gen_local = torch.Generator(device=device).manual_seed(args.seed)

                sample = {
                    "prompt": prompt,
                    "prompt_attention_mask": None,
                    "negative_prompt": "" if args.match_grpo_conditioning else args.negative_prompt,
                    "negative_prompt_attention_mask": None,
                }

                try:
                    cfg_call = dict(cfg_for_call)
                    extra_conditioning_kwargs: Dict[str, Any] = {}
                    if args.match_grpo_conditioning:
                        # Match GRPO conditioning style: no CFG/STG/rescaling, no negative prompt.
                        # Ensure we don't pass duplicates from YAML.
                        for k in ("guidance_scale", "stg_scale", "rescaling_scale"):
                            cfg_call.pop(k, None)
                        extra_conditioning_kwargs.update(
                            {
                                "guidance_scale": 1.0,
                                "stg_scale": 0.0,
                                "rescaling_scale": 1.0,
                            }
                        )
                    result = pipeline(
                        **cfg_call,
                        skip_layer_strategy=skip_layer_strategy,
                        generator=gen_local,
                        output_type="pt",
                        callback_on_step_end=None,
                        height=height_padded,
                        width=width_padded,
                        num_frames=num_frames_padded,
                        frame_rate=args.frame_rate,
                        **sample,
                        **extra_conditioning_kwargs,
                        media_items=None,
                        conditioning_items=None,
                        is_video=True,
                        vae_per_channel_normalize=True,
                        image_cond_noise_scale=0.0,
                        mixed_precision=(precision == "mixed_precision"),
                        offload_to_cpu=False,
                        device=device,
                        enhance_prompt=False,
                        stochastic_sampling=args.stochastic_sampling,
                    ).images  # [B, C, F, H, W] in [0,1]
                finally:
                    # Always restore, even if the baseline call errors out.
                    pipeline.denoising_step = _orig_step

                # Crop final and save for convenience
                pad_left, pad_right, pad_top, pad_bottom = padding
                pb = -pad_bottom
                pr = -pad_right
                if pb == 0:
                    pb = result.shape[3]
                if pr == 0:
                    pr = result.shape[4]
                result = result[:, :, : args.num_frames, pad_top:pb, pad_left:pr]
                final_video = result.permute(0, 2, 1, 3, 4)  # [B,T,C,H,W]
                _save_video_to_mp4(final_video, save_base / "final.mp4", fps=args.frame_rate)
                # Explicitly drop big tensors and clear cache after baseline phase.
                del result, final_video
                _clear_gpu_cache(tag="baseline")

            def _run_grpo(save_base: Path) -> None:
                # Ensure GRPO uses the original denoising_step implementation.
                pipeline.denoising_step = orig_denoising_step

                # Match pipeline.py: tee all GRPO prints into training_log_*.txt in the GRPO folder.
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                log_path = save_base / f"training_log_{ts}.txt"
                logger = TeeLogger(str(log_path))
                old_stdout, old_stderr = sys.stdout, sys.stderr
                sys.stdout = logger
                sys.stderr = logger
                try:
                    print(f"Training log: {log_path}")
                    print(f"Output dir: {save_base}")
                    print(f"Prompt: {prompt}")
                    print(f"GRPO_FROM_START={os.getenv('GRPO_FROM_START', '0')}")
                    print(
                        f"Reward weights override (REWARD_WEIGHTS_JSON): "
                        f"{os.getenv('REWARD_WEIGHTS_JSON', '<default>')}\n"
                    )

                    run_grpo_for_prompt(
                        pipeline=pipeline,
                        prompt=prompt,
                        out_dir=save_base,
                        height=args.height,
                        width=args.width,
                        num_frames=args.num_frames,
                        frame_rate=args.frame_rate,
                        seed=args.seed,
                        num_inference_steps=args.num_inference_steps,
                        num_grpo_steps=args.num_grpo_steps,
                        num_rollouts=args.num_rollouts,
                        lr=args.lr,
                        attn1_blocks=_parse_int_list(args.attn1_blocks),
                        attn2_blocks=_parse_int_list(args.attn2_blocks),
                        rollout_noise_scale=args.rollout_noise_scale,
                        normalize_advantages=bool(args.normalize_advantages),
                        use_grpo_kl=bool(args.use_grpo_kl),
                        kl_beta=float(args.kl_beta),
                        negative_prompt=args.negative_prompt,
                        guidance_scale=float(cfg.get("guidance_scale", 1.0)),
                        stg_scale=float(cfg.get("stg_scale", 0.0)),
                        rescaling_scale=float(cfg.get("rescaling_scale", 1.0)),
                        cfg_star_rescale=bool(cfg.get("cfg_star_rescale", False)),
                        skip_layer_strategy=skip_layer_strategy,
                        skip_block_list=cfg.get("skip_block_list", None),
                    )
                finally:
                    sys.stdout = old_stdout
                    sys.stderr = old_stderr
                    logger.close()

            if args.mode == "baseline_intermediates":
                # Back-compat: keep old layout directly under prompt_dir
                _run_baseline_intermediates(prompt_dir)
            elif args.mode == "grpo_train":
                # GRPO only, write directly under prompt_dir
                _run_grpo(prompt_dir)
            else:
                # both: keep outputs separate but within the same per-prompt folder
                baseline_dir = prompt_dir / "baseline"
                grpo_dir = prompt_dir / "grpo"
                baseline_dir.mkdir(parents=True, exist_ok=True)
                grpo_dir.mkdir(parents=True, exist_ok=True)
                _run_baseline_intermediates(baseline_dir)
                _clear_gpu_cache(tag="baseline->grpo boundary")
                _run_grpo(grpo_dir)

    finally:
        # restore
        pipeline.denoising_step = orig_denoising_step  # type: ignore[assignment]


if __name__ == "__main__":
    main()

