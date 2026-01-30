#!/usr/bin/env python3
"""
Standalone GRPO runner extracted from `baseline_intermediate_videos.py`.

This module exposes a single public entrypoint:

  run_grpo_for_prompt(...)

so other scripts can import and call GRPO without depending on argparse/global args.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict, List

import imageio
import numpy as np
import torch

from helper import decode_x0_to_video
from reward_functions import reward_function

try:
    from ltx_video.models.autoencoders.causal_video_autoencoder import CausalVideoAutoencoder  # type: ignore
    from ltx_video.models.autoencoders.vae_encode import latent_to_pixel_coords  # type: ignore
except ModuleNotFoundError:
    from ltx_video.ltx_video.models.autoencoders.causal_video_autoencoder import (  # type: ignore
        CausalVideoAutoencoder,
    )
    from ltx_video.ltx_video.models.autoencoders.vae_encode import latent_to_pixel_coords  # type: ignore


def _save_video_to_mp4(video_btc_hw: torch.Tensor, out_path: Path, fps: int) -> None:
    """
    Save [B, T, C, H, W] in [0, 1] as mp4.
    Mirrors `baseline_intermediate_videos._save_video_to_mp4`.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    v = video_btc_hw.detach().float().cpu()
    v = torch.nan_to_num(v, nan=0.0, posinf=1.0, neginf=0.0)
    v = v.clamp(0.0, 1.0)
    frames = (v[0].permute(0, 2, 3, 1).numpy() * 255.0).round().astype(np.uint8)  # [T,H,W,C]

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
        

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="scaling-noise/configs/inference_t2v_512_v2.0.yaml")
    ap.add_argument(
        "--ckpt_path",
        default="base_512_v2/model.ckpt",
        help=(
            "Path to a VideoCrafter2 checkpoint (.ckpt), or a directory containing a .ckpt. "
            "Default prefers the repo-root `base_512_v2/model.ckpt`."
        ),
    )
    ap.add_argument("--prompt", required=True)
    ap.add_argument(
        "--negative_prompt",
        default="",
        help="Optional negative prompt. VideoCrafter2 code here uses OpenCLIP conditioning; negative prompts are not wired like LTX.",
    )
    ap.add_argument("--out_dir", default="videocrafter2_grpo_runs")

    # Video settings (must match the VC2 config's temporal_length unless you know what you're doing).
    ap.add_argument("--height", type=int, default=320)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--num_frames", type=int, default=16)
    ap.add_argument("--fps", type=int, default=8)

    # Diffusion sampling
    ap.add_argument("--num_inference_steps", type=int, default=16)
    ap.add_argument("--ddim_eta", type=float, default=1.0)
    ap.add_argument("--guidance_scale", type=float, default=12.0)
    ap.add_argument("--seed", type=int, default=321)

    # GRPO knobs
    ap.add_argument("--num_grpo_steps", type=int, default=6, help="How many final DDIM steps to GRPO-tune.")
    ap.add_argument("--num_rollouts", type=int, default=4)
    ap.add_argument("--rollout_noise_scale", type=float, default=0.5, help="Latent perturbation scale for rollouts.")
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--normalize_advantages", type=int, default=1)
    ap.add_argument(
        "--trainable",
        default="temporal",
        choices=["temporal", "all", "none"],
        help="Which params to update. 'temporal' = only params with 'temporal' in name.",
    )
    return ap.parse_args()


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

    scale_shape = [1] * latents.ndim
    if effective_t.ndim > 0:
        scale_shape[0] = effective_t.shape[0]
    scale = effective_t.reshape(scale_shape).to(device=latents.device, dtype=latents.dtype)
    return latents - scale * noise_pred


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
    rollout_noise_preds_cpu: bool,
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
    GRPO training loop mirroring `pipeline.py`, using the LTX-Video pipeline object.
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

    # Track initial weights for comparison at the end (match pipeline.py). Store on CPU.
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
            if do_spatio_temporal_guidance and skip_block_list is not None:
                skip_blocks_for_step = skip_block_list
                if (
                    isinstance(skip_block_list, list)
                    and len(skip_block_list) > 0
                    and isinstance(skip_block_list[0], list)
                ):
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
                current_timestep = torch.tensor(
                    [current_timestep], device=latent_model_input.device, dtype=torch.float64
                )
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
        rollout_seeds: List[int] = []

        for r in range(int(num_rollouts)):
            with torch.no_grad():
                rollout_seed = int(seed) + int(step_idx) * 1000 + int(r)
                torch.manual_seed(rollout_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(rollout_seed)
                rollout_seeds.append(int(rollout_seed))
                print(f"\n[GRPO][step={step_idx:03d} r={r:02d}] rollout_seed={rollout_seed}")

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
                if (
                    do_spatio_temporal_guidance
                    and skip_block_list is not None
                    and hasattr(model, "create_skip_layer_mask")
                ):
                    skip_blocks_for_step = skip_block_list
                    if (
                        isinstance(skip_block_list, list)
                        and len(skip_block_list) > 0
                        and isinstance(skip_block_list[0], list)
                    ):
                        skip_blocks_for_step = skip_block_list[min(step_idx, len(skip_block_list) - 1)]
                    skip_layer_mask = model.create_skip_layer_mask(1, num_conds, num_conds - 1, skip_blocks_for_step)

                batch_fractional_coords = torch.cat([indices_grid] * num_conds)
                latent_model_input = (
                    torch.cat([latents_perturbed] * num_conds) if num_conds > 1 else latents_perturbed
                )
                latent_model_input = pipeline.scheduler.scale_model_input(latent_model_input, t)

                current_timestep = t
                if not torch.is_tensor(current_timestep):
                    current_timestep = torch.tensor(
                        [current_timestep], device=latent_model_input.device, dtype=torch.float64
                    )
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

                if rollout_noise_preds_cpu:
                    rollout_noise_preds.append(noise_pred.detach().to("cpu"))
                else:
                    rollout_noise_preds.append(noise_pred.detach().clone())

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
                seed_str = "NA"
                if r < len(rollout_seeds):
                    seed_str = str(int(rollout_seeds[r]))
                f.write(
                    f"r={r} seed={seed_str} reward={float(rewards[r]):.6f} "
                    f"advantage={float(adv[r]):.6f} mp4={rollout_paths[r]}\n"
                )

        # Step 3: policy gradient update
        optimizer.zero_grad()
        model.zero_grad()

        latents_for_loss = latents.detach().clone().requires_grad_(True)
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
            ref = rollout_noise_preds[r]
            if ref.device != noise_pred_current.device:
                ref = ref.to(device=noise_pred_current.device)
            mse = ((ref.detach() - noise_pred_current) ** 2).mean()
            log_probs.append(-mse)
        log_probs_t = torch.stack(log_probs)
        pg_loss = -(log_probs_t * adv).mean()

        kl_loss = torch.tensor(0.0, device=pg_loss.device)
        if use_grpo_kl and float(kl_beta) > 0.0:
            with torch.no_grad():
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
                if (
                    do_spatio_temporal_guidance
                    and skip_block_list is not None
                    and hasattr(baseline_model, "create_skip_layer_mask")
                ):
                    skip_blocks_for_step = skip_block_list
                    if (
                        isinstance(skip_block_list, list)
                        and len(skip_block_list) > 0
                        and isinstance(skip_block_list[0], list)
                    ):
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

        total_grad_norm = torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]["params"], 0.5)

        weights_before: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                weights_before[name] = param.detach().float().cpu().clone()

        optimizer.step()

        weight_changes: List[float] = []
        for name, param in model.named_parameters():
            if param.requires_grad and name in weights_before:
                after = param.detach().float().cpu()
                change = (after - weights_before[name]).abs().mean().item()
                weight_changes.append(float(change))
        avg_weight_change = (sum(weight_changes) / len(weight_changes)) if weight_changes else 0.0

        print(
            f"  ✅ Weights updated! total_grad_norm={float(total_grad_norm):.6f}, weight_Δ={avg_weight_change:.6f}"
        )

        # Step 4: advance latents with updated model (deterministic)
        with torch.no_grad():
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

