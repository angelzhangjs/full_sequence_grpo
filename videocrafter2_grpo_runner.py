#!/usr/bin/env python3
"""
Reusable GRPO runner for VideoCrafter2 (vendored under `scaling-noise/`).

This is the VideoCrafter2 analog of `ltx_grpo_runner.py`: it exposes a single public function:

  run_grpo_for_prompt(...)

which performs:
  - baseline DDIM sampling (before.mp4)
  - GRPO-style updates on the last K DDIM steps using reward_function(video, prompt)
  - another DDIM sampling (after.mp4)

Design notes:
  - VideoCrafter2 sampling uses `DDIMSampler`, whose methods are `@torch.no_grad()`.
    For training we mirror the DDIM step math with autograd enabled.
  - Reward uses this repo's `reward_functions.reward_function` and expects video in [B,T,C,H,W], float.
"""

from __future__ import annotations

import os
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any

import imageio
import numpy as np
import torch

from lvdm.models.samplers.ddim import DDIMSampler
from reward_functions import (
    reward_function,
    subject_consistency,
    clip_text_alignment_reward,
    clip_temporal_alignment_reward,
)  # noqa: E402

def _save_mp4(frames_bcthw: torch.Tensor, out_path: Path, fps: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vid = _to_uint8_video(frames_bcthw)
    imageio.mimsave(str(out_path), list(vid), fps=fps)
    
def _ensure_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script.")
    return torch.device("cuda")

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/inference_t2v_512_v2.0.yaml")
    ap.add_argument(
        "--ckpt_path",
        default="base_512_v2/model.ckpt",
        help="VideoCrafter2 checkpoint (.ckpt) path (or dir containing a .ckpt).",
    )
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--negative_prompt", default="")
    ap.add_argument("--out_dir", default="videocrafter2_grpo_runs")
    ap.add_argument("--height", type=int, default=320)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--num_frames", type=int, default=16)
    ap.add_argument("--fps", type=int, default=8)

    # Sampling
    ap.add_argument("--sampling_mode", default="ddim", choices=["ddim", "fifo"])
    ap.add_argument("--num_inference_steps", type=int, default=16)
    ap.add_argument("--ddim_eta", type=float, default=1.0)
    ap.add_argument("--guidance_scale", type=float, default=12.0)
    ap.add_argument("--seed", type=int, default=321)

    # FIFO knobs (scaling-noise style)
    ap.add_argument("--fifo_video_length", type=int, default=16, help="FIFO window length (f in FIFO paper).")
    ap.add_argument("--fifo_new_video_length", type=int, default=100, help="Final desired output length (N in FIFO paper).")
    ap.add_argument("--fifo_num_partitions", type=int, default=4, help="Number of stagger partitions (n in FIFO paper).")
    ap.add_argument("--fifo_lookahead_denoising", type=int, default=1, help="Use lookahead/overlap (1=yes, 0=no).")
    ap.add_argument(
        "--fifo_train_last_partitions",
        type=int,
        default=2,
        help="GRPO Option B: train on the last K FIFO partitions per outer step (K=1 trains only the last partition).",
    )

    # GRPO knobs
    ap.add_argument("--num_grpo_steps", type=int, default=6)
    ap.add_argument("--num_rollouts", type=int, default=4)
    ap.add_argument("--rollout_noise_scale", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--normalize_advantages", type=int, default=1)
    ap.add_argument("--trainable", default="temporal", choices=["temporal", "all", "none"])

    # Reward knobs
    ap.add_argument("--reward_mode", default="clip_dino", choices=["clip_dino", "subject_consistency", "clip_subject"])
    ap.add_argument("--reward_device", default="cuda")

    return ap.parse_args()

def _to_uint8_video(frames_bcthw: torch.Tensor) -> np.ndarray:
    """
    Input: [B,C,T,H,W] in [-1, 1] (VideoCrafter2 VAE decode convention).
    Output: uint8 [T,H,W,C] in [0,255].
    """
    x = frames_bcthw.detach()
    if x.ndim != 5:
        raise ValueError(f"Expected [B,C,T,H,W], got {tuple(x.shape)}")
    x = x[0]  # [C,T,H,W]
    x = (x / 2.0 + 0.5).clamp(0.0, 1.0)
    x = (x * 255.0).to(torch.uint8)
    x = x.permute(1, 2, 3, 0).contiguous()  # [T,H,W,C]
    return x.cpu().numpy()


def _save_mp4(frames_bcthw: torch.Tensor, out_path: Path, fps: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vid = _to_uint8_video(frames_bcthw)
    try:
        imageio.mimsave(str(out_path), list(vid), fps=int(fps))
    except Exception as e:
        msg = str(e)
        if "Could not find a backend" in msg or "FFMPEG" in msg or "pyav" in msg:
            raise RuntimeError(
                "Failed to write .mp4 via imageio (missing ffmpeg backend).\n"
                "Fix (recommended):\n"
                "  conda run -n scaling-noise pip install \"imageio[ffmpeg]\" imageio-ffmpeg\n"
                "Alternative:\n"
                "  conda install -n scaling-noise -c conda-forge ffmpeg\n"
                f"\nOutput path: {out_path}\n"
            ) from e
        raise


def _make_conditioning(model, prompt: str, fps: int, batch_size: int = 1) -> Dict:
    prompts = [prompt] * batch_size
    text_emb = model.get_learned_conditioning(prompts)
    fps_t = torch.tensor([fps] * batch_size, device=model.device).long()
    return {"c_crossattn": [text_emb], "fps": fps_t}


def _make_unconditional_conditioning(model, cond: Dict, batch_size: int = 1) -> Dict | None:
    if cond is None:
        return None
    prompts = [""] * batch_size
    uc_emb = model.get_learned_conditioning(prompts)
    uc = {k: cond[k] for k in cond.keys()}
    uc.update({"c_crossattn": [uc_emb]})
    return uc


def _select_trainable_params(model, mode: str) -> List[torch.nn.Parameter]:
    for p in model.parameters():
        p.requires_grad = False

    trainable: List[torch.nn.Parameter] = []
    if mode == "none":
        return trainable

    if mode == "all":
        for _, p in model.named_parameters():
            p.requires_grad = True
            trainable.append(p)
        return trainable

    if mode == "temporal":
        # VideoCrafter2 temporal modules are often instances of classes like:
        #   TemporalTransformer, TemporalConvBlock
        # BUT parameter names may *not* include "temporal" (e.g. misspelled "temopral_conv"),
        # so we select by module class name instead of string-matching parameter names.
        seen = set()
        for _, m in model.named_modules():
            cls = m.__class__.__name__.lower()
            if "temporal" not in cls:
                continue
            for p in m.parameters(recurse=True):
                pid = id(p)
                if pid in seen:
                    continue
                seen.add(pid)
                p.requires_grad = True
                trainable.append(p)
        return trainable

    if mode != "none" and len(trainable) == 0:
        print("⚠️ Warning: no parameters matched trainable selection; nothing will be updated.")
    else:
        print(f"Trainable parameters: {len(trainable)}")
    return trainable

def _ddim_step_with_grads(
    *,
    model,
    x: torch.Tensor,  # [B,C,T,H,W]
    t: torch.Tensor,  # [B]
    index: int,  # index into DDIM arrays
    sampler: DDIMSampler,
    cond: Dict,
    uc: Dict | None,
    guidance_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns (x_prev, pred_x0, e_t) with gradients through e_t and pred_x0.
    Mirrors the math in DDIMSampler.p_sample_ddim but keeps autograd enabled.
    """
    device = x.device
    b = x.shape[0]

    # Model prediction (eps / noise pred)
    if uc is None or guidance_scale == 1.0:
        e_t = model.apply_model(x, t, cond, clean_cond=True)
    else:
        e_t_cond = model.apply_model(x, t, cond, clean_cond=True)
        e_t_uncond = model.apply_model(x, t, uc, clean_cond=True)
        e_t = e_t_uncond + guidance_scale * (e_t_cond - e_t_uncond)

    alphas = sampler.ddim_alphas
    alphas_prev = sampler.ddim_alphas_prev
    sqrt_one_minus_alphas = sampler.ddim_sqrt_one_minus_alphas
    sigmas = sampler.ddim_sigmas

    size = (b, 1, 1, 1, 1)
    a_t = torch.full(size, alphas[index], device=device, dtype=x.dtype)
    a_prev = torch.full(size, alphas_prev[index], device=device, dtype=x.dtype)
    sigma_t = torch.full(size, sigmas[index], device=device, dtype=x.dtype)
    sqrt_one_minus_at = torch.full(size, sqrt_one_minus_alphas[index], device=device, dtype=x.dtype)

    pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
    dir_xt = (1.0 - a_prev - sigma_t**2).clamp(min=0.0).sqrt() * e_t
    noise = sigma_t * torch.randn_like(x)
    x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise

    return x_prev, pred_x0, e_t


@torch.no_grad()
def _fifo_prepare_latents_from_x0_video(
    *,
    x0_video_latents_bcthw: torch.Tensor,  # [B,C,video_length,H,W]
    sampler: DDIMSampler,
    video_length: int,
    num_inference_steps: int,
    lookahead_denoising: bool,
) -> torch.Tensor:
    """
    Port of scaling-noise `prepare_latents`, but fully in-memory.
    Builds a long temporal latent buffer where each frame is assigned a different noise level
    (staggered by DDIM alpha) so FIFO can denoise a moving window.
    """
    if x0_video_latents_bcthw.ndim != 5:
        raise ValueError(f"Expected [B,C,T,H,W], got {tuple(x0_video_latents_bcthw.shape)}")
    b, c, t0, h, w = x0_video_latents_bcthw.shape
    if int(t0) != int(video_length):
        raise ValueError(f"x0_video_latents has T={t0}, but video_length={video_length}")

    device = x0_video_latents_bcthw.device
    latents_list: List[torch.Tensor] = []

    # Optional lookahead: prepend half-window with the most-noisy alpha (index 0).
    if bool(lookahead_denoising):
        half = int(video_length) // 2
        alpha0 = sampler.ddim_alphas[0]
        beta0 = 1.0 - alpha0
        for _ in range(half):
            lat = (alpha0**0.5) * x0_video_latents_bcthw[:, :, [0]] + (beta0**0.5) * torch.randn(
                (b, c, 1, h, w), device=device, dtype=x0_video_latents_bcthw.dtype
            )
            latents_list.append(lat)

    # Main staggered buffer length is num_inference_steps (= video_length * num_partitions in FIFO mode).
    for i in range(int(num_inference_steps)):
        alpha = sampler.ddim_alphas[i]
        beta = 1.0 - alpha
        frame_idx = max(0, i - (int(num_inference_steps) - int(video_length)))
        src = x0_video_latents_bcthw[:, :, [int(frame_idx)]]
        lat = (alpha**0.5) * src + (beta**0.5) * torch.randn((b, c, 1, h, w), device=device, dtype=src.dtype)
        latents_list.append(lat)

    return torch.cat(latents_list, dim=2)  # [B,C,Tbuf,H,W]


@torch.no_grad()
def _fifo_shift_latents(latents_bcthw: torch.Tensor) -> torch.Tensor:
    """
    Shift FIFO buffer left by one frame and inject fresh noise at the last frame.
    Port of scaling-noise `shift_latents`.
    """
    latents_bcthw[:, :, :-1] = latents_bcthw[:, :, 1:].clone()
    latents_bcthw[:, :, -1] = torch.randn_like(latents_bcthw[:, :, -1])
    return latents_bcthw


def _fifo_step_with_grads(
    *,
    model,
    sampler: DDIMSampler,
    latents_window_bcthw: torch.Tensor,  # [B,C,video_length,H,W]
    timesteps: np.ndarray,  # [video_length] DDIM timestep values
    indices: np.ndarray,  # [video_length] indices into DDIM arrays
    cond: Dict,
    uc: Dict | None,
    guidance_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    GRAD-enabled version of `DDIMSampler.fifo_onestep`:
    returns (x_prev_window, pred_x0_window, e_t_window) with gradients through model params.
    """
    device = latents_window_bcthw.device
    b = latents_window_bcthw.shape[0]

    ts = torch.tensor(timesteps, device=device, dtype=torch.long)  # [T]
    if uc is None or float(guidance_scale) == 1.0:
        e_t = model.apply_model(latents_window_bcthw, ts, cond, clean_cond=True)
    else:
        e_t_cond = model.apply_model(latents_window_bcthw, ts, cond, clean_cond=True)
        e_t_uncond = model.apply_model(latents_window_bcthw, ts, uc, clean_cond=True)
        e_t = e_t_uncond + float(guidance_scale) * (e_t_cond - e_t_uncond)

    alphas = sampler.ddim_alphas
    alphas_prev = sampler.ddim_alphas_prev
    sqrt_one_minus_alphas = sampler.ddim_sqrt_one_minus_alphas
    sigmas = sampler.ddim_sigmas

    size = (b, 1, 1, 1, 1)
    x_prevs: List[torch.Tensor] = []
    pred_x0s: List[torch.Tensor] = []
    # Per-frame DDIM step (matching `DDIMSampler.ddim_step`).
    for fi, idx in enumerate(indices.tolist()):
        x = latents_window_bcthw[:, :, [fi]]
        et = e_t[:, :, [fi]]
        a_t = torch.full(size, float(alphas[idx]), device=device, dtype=x.dtype)
        a_prev = torch.full(size, float(alphas_prev[idx]), device=device, dtype=x.dtype)
        sigma_t = torch.full(size, float(sigmas[idx]), device=device, dtype=x.dtype)
        sqrt_one_minus_at = torch.full(size, float(sqrt_one_minus_alphas[idx]), device=device, dtype=x.dtype)

        pred_x0 = (x - sqrt_one_minus_at * et) / a_t.sqrt()
        dir_xt = (1.0 - a_prev - sigma_t**2).clamp(min=0.0).sqrt() * et
        noise = sigma_t * torch.randn_like(x)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        x_prevs.append(x_prev)
        pred_x0s.append(pred_x0)

    return torch.cat(x_prevs, dim=2), torch.cat(pred_x0s, dim=2), e_t


@torch.no_grad()
def sample_fifo_decode(
    *,
    model,
    sampler: DDIMSampler,
    cond: Dict,
    uc: Dict | None,
    height: int,
    width: int,
    seed: int,
    guidance_scale: float,
    ddim_eta: float,
    fifo_video_length: int,
    fifo_new_video_length: int,
    fifo_num_partitions: int,
    fifo_lookahead_denoising: bool,
) -> torch.Tensor:
    """
    FIFO baseline sampling (scaling-noise style), returning decoded video [B,C,T,H,W] where T=fifo_new_video_length.
    Implementation is self-contained and relies only on repo-root `lvdm`.
    """
    device = model.device
    latent_h, latent_w = int(height) // 8, int(width) // 8
    base_shape = (4, int(fifo_video_length), latent_h, latent_w)

    # scaling-noise sets num_inference_steps = video_length * num_partitions.
    num_inference_steps = int(fifo_video_length) * int(fifo_num_partitions)
    sampler.make_schedule(ddim_num_steps=num_inference_steps, ddim_eta=float(ddim_eta), verbose=False)

    # 1) Generate the base x0 video latents for the window (standard DDIM).
    x0_video, x0_latents = sample_ddim_decode_and_latents(
        model=model,
        sampler=sampler,
        cond=cond,
        uc=uc,
        shape=base_shape,
        num_steps=num_inference_steps,
        eta=float(ddim_eta),
        guidance_scale=float(guidance_scale),
        seed=int(seed),
    )
    del x0_video  # only need latents

    # 2) Prepare FIFO buffer.
    fifo_latents = _fifo_prepare_latents_from_x0_video(
        x0_video_latents_bcthw=x0_latents,
        sampler=sampler,
        video_length=int(fifo_video_length),
        num_inference_steps=num_inference_steps,
        lookahead_denoising=bool(fifo_lookahead_denoising),
    )

    timesteps = np.array(sampler.ddim_timesteps)  # ascending
    indices = np.arange(num_inference_steps, dtype=np.int64)
    if bool(fifo_lookahead_denoising):
        half = int(fifo_video_length) // 2
        timesteps = np.concatenate([np.full((half,), timesteps[0]), timesteps])
        indices = np.concatenate([np.full((half,), 0, dtype=indices.dtype), indices])

    total_partitions = int(fifo_num_partitions) * (2 if bool(fifo_lookahead_denoising) else 1)
    noise_shape = (1, 4, int(fifo_video_length), latent_h, latent_w)

    frames: List[torch.Tensor] = []
    outer_steps = int(fifo_new_video_length) + num_inference_steps - int(fifo_video_length)
    first_frame_idx = (int(fifo_video_length) // 2) if bool(fifo_lookahead_denoising) else 0

    for _outer in range(int(outer_steps)):
        for rank in reversed(range(total_partitions)):
            start_idx = rank * (int(fifo_video_length) // 2) if bool(fifo_lookahead_denoising) else rank * int(fifo_video_length)
            midpoint_idx = start_idx + int(fifo_video_length) // 2
            end_idx = start_idx + int(fifo_video_length)

            t = timesteps[start_idx:end_idx]
            idx = indices[start_idx:end_idx]

            input_latents = fifo_latents[:, :, start_idx:end_idx].clone()
            output_latents, _ = sampler.fifo_onestep(
                cond=cond,
                shape=noise_shape,
                latents=input_latents,
                timesteps=t,
                indices=idx,
                unconditional_guidance_scale=float(guidance_scale),
                unconditional_conditioning=uc,
                clean_cond=True,
            )
            if bool(fifo_lookahead_denoising):
                fifo_latents[:, :, midpoint_idx:end_idx] = output_latents[:, :, -(int(fifo_video_length) // 2) :]
            else:
                fifo_latents[:, :, start_idx:end_idx] = output_latents

        frame_lat = fifo_latents[:, :, [first_frame_idx]]
        frame = model.decode_first_stage_2DAE(frame_lat)  # [B,C,1,H,W]
        frames.append(frame)
        fifo_latents = _fifo_shift_latents(fifo_latents)

    video = torch.cat(frames, dim=2)  # [B,C,Tout,H,W]
    if video.shape[2] >= int(fifo_new_video_length):
        video = video[:, :, -int(fifo_new_video_length) :]
    return video


@torch.no_grad()
def _ddim_sample_decode(
    *,
    model,
    sampler: DDIMSampler,
    cond: Dict,
    uc: Dict | None,
    shape: Tuple[int, int, int, int],  # [C,T,H,W]
    num_steps: int,
    eta: float,
    guidance_scale: float,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (decoded_video_bcthw, final_latents_bcthw).
    """
    device = model.device
    sampler.make_schedule(ddim_num_steps=num_steps, ddim_eta=eta, verbose=False)

    g = torch.Generator(device=device)
    g.manual_seed(seed)
    x = torch.randn((1,) + shape, device=device, generator=g)

    time_range = np.flip(sampler.ddim_timesteps)
    for i, step_t in enumerate(time_range):
        ts = torch.full((1,), int(step_t), device=device, dtype=torch.long)
        x, _ = sampler.p_sample_ddim(
            x,
            cond,
            ts,
            index=i,
            unconditional_guidance_scale=guidance_scale,
            unconditional_conditioning=uc,
            temperature=1.0,
            noise_dropout=0.0,
            eta=eta,
            clean_cond=True,
            temporal_length=shape[1],
        )

    latents = x
    video = model.decode_first_stage_2DAE(latents)
    return video, latents

@torch.no_grad()
def sample_ddim_decode_and_latents(
    *,
    model,
    sampler: DDIMSampler,
    cond: Dict,
    uc: Dict | None,
    shape: Tuple[int, int, int, int],  # [C,T,H,W]
    num_steps: int,
    eta: float,
    guidance_scale: float,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns (decoded_video_bcthw, final_latents_bcthw)."""
    device = model.device
    sampler.make_schedule(ddim_num_steps=num_steps, ddim_eta=eta, verbose=False)

    g = torch.Generator(device=device)
    g.manual_seed(seed)
    x = torch.randn((1,) + shape, device=device, generator=g)  # [1,C,T,H,W]

    time_range = np.flip(sampler.ddim_timesteps)
    for i, step_t in enumerate(time_range):
        ts = torch.full((1,), int(step_t), device=device, dtype=torch.long)
        x, _ = sampler.p_sample_ddim(
            x,
            cond,
            ts,
            index=i,
            unconditional_guidance_scale=guidance_scale,
            unconditional_conditioning=uc,
            temperature=1.0,
            noise_dropout=0.0,
            eta=eta,
            clean_cond=True,
            temporal_length=shape[1],
        )

    latents = x
    video = model.decode_first_stage_2DAE(latents)
    return video, latents


def run_grpo_for_prompt(
    *,
    model: Any,
    prompt: str,
    out_dir: Path,
    height: int,
    width: int,
    num_frames: int,
    fps: int,
    seed: int,
    num_inference_steps: int,
    num_grpo_steps: int,
    num_rollouts: int,
    lr: float,
    trainable: str,
    rollout_noise_scale: float,
    normalize_advantages: bool,
    grad_clip: float,
    # Sampling knobs
    guidance_scale: float,
    ddim_eta: float,
    sampling_mode: str = "ddim",
    # FIFO knobs (only used when sampling_mode="fifo")
    fifo_video_length: int = 16,
    fifo_new_video_length: int = 100,
    fifo_num_partitions: int = 4,
    fifo_lookahead_denoising: bool = True,
    fifo_train_last_partitions: int = 2,
    # Debug / logging
    save_intermediate_rollouts: bool = False,
    save_rollout_every: int = 1,
    # Reward knobs
    reward_mode: str = "clip_dino",
    reward_device: str = "cuda",
    clip_weight: float = 0.5,
    subject_weight: float = 0.5,
) -> None:
    """
    Run VideoCrafter2 GRPO for a single prompt and save `before.mp4` and `after.mp4` in `out_dir`.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for VideoCrafter2 GRPO.")
    device = model.device if hasattr(model, "device") else torch.device("cuda")

    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))

    mode_sampling = str(sampling_mode).strip().lower()

    # Latent shape: channels=4, spatial compression=8
    latent_h, latent_w = int(height) // 8, int(width) // 8
    if mode_sampling == "fifo":
        latent_shape = (4, int(fifo_video_length), latent_h, latent_w)
    else:
        latent_shape = (4, int(num_frames), latent_h, latent_w)

    cond = _make_conditioning(model, prompt, fps=int(fps), batch_size=1)
    uc = _make_unconditional_conditioning(model, cond, batch_size=1) if float(guidance_scale) != 1.0 else None

    sampler = DDIMSampler(model)

    def _score_x0_video(video_bcthw: torch.Tensor) -> torch.Tensor:
        """
        Score the *decoded x0 video*.
        - video_bcthw: [B,C,T,H,W] (VideoCrafter2 decode; typically in [-1,1])
        Returns: scalar tensor on `reward_device`.
        """
        mode = str(reward_mode).strip().lower()
        if mode in {"clip_dino", "reward_function", "default"}:
            # reward_function expects [B,T,C,H,W] float
            video_btc_hw = video_bcthw.permute(0, 2, 1, 3, 4).contiguous().float()
            return reward_function(video_btc_hw, prompt=prompt, device=reward_device)

        if mode in {"subject_consistency", "dino_consistency"}:
            # Normalize to [C,T,H,W] in [0,1]
            v = video_bcthw[0].detach().float()  # [C,T,H,W]
            if v.min().item() < 0.0:
                v = (v + 1.0) / 2.0
            v = v.clamp(0.0, 1.0)
            frames_tchw = v.permute(1, 0, 2, 3).contiguous()  # [T,C,H,W]
            frames_list = [frames_tchw[i] for i in range(frames_tchw.shape[0])]
            score = float(subject_consistency(frames_list, device=reward_device))
            return torch.tensor(score, device=reward_device, dtype=torch.float32)

        if mode in {"clip_subject", "clip+subject", "clip_subject_consistency"}:
            # Normalize to [C,T,H,W] in [0,1]
            v = video_bcthw[0].detach().float()  # [C,T,H,W]
            if v.min().item() < 0.0:
                v = (v + 1.0) / 2.0
            v = v.clamp(0.0, 1.0)
            # CLIP text alignment (semantic + temporal) computed on [C,T,H,W] video
            clip_a = float(clip_text_alignment_reward(v, prompt=prompt, device=reward_device))
            clip_t = float(clip_temporal_alignment_reward(v, prompt=prompt, device=reward_device))
            text_alignment = 0.5 * clip_a + 0.5 * clip_t

            frames_tchw = v.permute(1, 0, 2, 3).contiguous()
            frames_list = [frames_tchw[i] for i in range(frames_tchw.shape[0])]
            subj = float(subject_consistency(frames_list, device=reward_device))

            w_clip = float(clip_weight)
            w_subj = float(subject_weight)
            denom = (w_clip + w_subj) if (w_clip + w_subj) > 0 else 1.0
            score = (w_clip * text_alignment + w_subj * subj) / denom
            return torch.tensor(score, device=reward_device, dtype=torch.float32)

        raise ValueError(f"Unknown reward_mode={reward_mode!r}. Use 'clip_dino' or 'subject_consistency'.")

    rollout_dir = out_dir / "intermediate_rollout"
    logs_dir = rollout_dir / "logs"
    if save_intermediate_rollouts:
        logs_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # GRPO tuning
    # ------------------------------------------------------------------
    params = _select_trainable_params(model, trainable)
    if len(params) == 0:
        print("Nothing to train; will still save `after.mp4` (no GRPO updates applied).", flush=True)
        # Fall through to the after-sampling block at the end.
        params = []

    opt = torch.optim.AdamW(params, lr=float(lr)) if len(params) > 0 else None

    if mode_sampling == "fifo":
        # -----------------------
        # FIFO GRPO: train on last K FIFO outer steps (frame-generation steps).
        # -----------------------
        fifo_video_length_i = int(fifo_video_length)
        fifo_new_video_length_i = int(fifo_new_video_length)
        fifo_num_partitions_i = int(fifo_num_partitions)
        fifo_lookahead = bool(fifo_lookahead_denoising)

        num_steps = fifo_video_length_i * fifo_num_partitions_i
        sampler.make_schedule(ddim_num_steps=int(num_steps), ddim_eta=float(ddim_eta), verbose=False)

        # Base DDIM to get x0 video latents for the window.
        _, x0_latents = sample_ddim_decode_and_latents(
            model=model,
            sampler=sampler,
            cond=cond,
            uc=uc,
            shape=(4, fifo_video_length_i, latent_h, latent_w),
            num_steps=int(num_steps),
            eta=float(ddim_eta),
            guidance_scale=float(guidance_scale),
            seed=int(seed),
        )

        fifo_latents = _fifo_prepare_latents_from_x0_video(
            x0_video_latents_bcthw=x0_latents,
            sampler=sampler,
            video_length=fifo_video_length_i,
            num_inference_steps=num_steps,
            lookahead_denoising=fifo_lookahead,
        )

        timesteps = np.array(sampler.ddim_timesteps)
        idx_arr = np.arange(num_steps, dtype=np.int64)
        if fifo_lookahead:
            half = fifo_video_length_i // 2
            timesteps = np.concatenate([np.full((half,), timesteps[0]), timesteps])
            idx_arr = np.concatenate([np.full((half,), 0, dtype=idx_arr.dtype), idx_arr])

        total_partitions = fifo_num_partitions_i * (2 if fifo_lookahead else 1)
        last_rank = total_partitions - 1
        train_last_k = int(max(1, min(int(fifo_train_last_partitions), int(total_partitions))))
        train_ranks = set(range(int(total_partitions) - train_last_k, int(total_partitions)))
        outer_steps = fifo_new_video_length_i + num_steps - fifo_video_length_i
        grpo_steps = int(min(int(num_grpo_steps), int(outer_steps)))
        train_outer = set(range(int(outer_steps) - grpo_steps, int(outer_steps)))
        print(
            f"FIFO GRPO will train on FIFO outer indices: {sorted(train_outer)[:5]} ... {sorted(train_outer)[-5:]}\n"
            f"FIFO GRPO Option B: training ranks={sorted(train_ranks)} (last {train_last_k}/{total_partitions} partitions)"
        , flush=True)

        noise_shape = (1, 4, fifo_video_length_i, latent_h, latent_w)
        first_frame_idx = (fifo_video_length_i // 2) if fifo_lookahead else 0

        for outer_i in range(int(outer_steps)):
            if outer_i % 10 == 0:
                print(
                    f"[FIFO] outer={outer_i}/{outer_steps-1} "
                    f"(train={'yes' if outer_i in train_outer else 'no'})",
                    flush=True,
                )
            for rank in reversed(range(total_partitions)):
                start_idx = rank * (fifo_video_length_i // 2) if fifo_lookahead else rank * fifo_video_length_i
                midpoint_idx = start_idx + fifo_video_length_i // 2
                end_idx = start_idx + fifo_video_length_i

                t = timesteps[start_idx:end_idx]
                idx = idx_arr[start_idx:end_idx]

                # Option B: Train on last K partitions for the last GRPO outer steps.
                if (outer_i in train_outer) and (rank in train_ranks):
                    rollout_noise_preds: List[torch.Tensor] = []
                    rollout_rewards: List[torch.Tensor] = []
                    rollout_paths: List[str] = []
                    rollout_seeds: List[int] = []

                    base_window = fifo_latents[:, :, start_idx:end_idx].clone()
                    for r in range(int(num_rollouts)):
                        with torch.no_grad():
                            rollout_seed = int(seed) + int(outer_i) * 1000 + int(r)
                            torch.manual_seed(rollout_seed)
                            if torch.cuda.is_available():
                                torch.cuda.manual_seed_all(rollout_seed)
                            rollout_seeds.append(rollout_seed)

                            if r == 0:
                                x_r = base_window
                            else:
                                x_r = base_window + torch.randn_like(base_window) * float(rollout_noise_scale)

                            _, pred_x0_r, e_t_r = _fifo_step_with_grads(
                                model=model,
                                sampler=sampler,
                                latents_window_bcthw=x_r,
                                timesteps=t,
                                indices=idx,
                                cond=cond,
                                uc=uc,
                                guidance_scale=float(guidance_scale),
                            )
                            rollout_noise_preds.append(e_t_r.detach())
                            video_r = model.decode_first_stage_2DAE(pred_x0_r)  # [B,C,T,H,W]
                            if save_intermediate_rollouts and (outer_i % max(int(save_rollout_every), 1) == 0):
                                out_mp4 = rollout_dir / f"fifo_rollout_outer{outer_i:04d}_r{r:02d}.mp4"
                                _save_mp4(video_r, out_mp4, fps=int(fps))
                                rollout_paths.append(str(out_mp4))
                            rew = _score_x0_video(video_r)
                            rollout_rewards.append(rew.detach().to(dtype=torch.float32))

                    rewards = torch.stack(rollout_rewards)
                    mean_r = rewards.mean()
                    std_r = rewards.std()
                    if bool(normalize_advantages) and float(std_r) > 1e-8:
                        adv = (rewards - mean_r) / (std_r + 1e-4)
                    else:
                        adv = rewards - mean_r

                    if save_intermediate_rollouts and (outer_i % max(int(save_rollout_every), 1) == 0):
                        with (logs_dir / f"fifo_outer_{outer_i:04d}.txt").open("w", encoding="utf-8") as f:
                            f.write(f"fifo_outer={outer_i}\n")
                            f.write(f"rank={rank}\n")
                            f.write(f"train_rank={int(rank in train_ranks)}\n")
                            f.write(f"mean_reward={float(mean_r):.6f}\n")
                            f.write(f"std_reward={float(std_r):.6f}\n")
                            for r in range(int(num_rollouts)):
                                seed_str = str(rollout_seeds[r]) if r < len(rollout_seeds) else "NA"
                                mp4_str = rollout_paths[r] if r < len(rollout_paths) else "NA"
                                f.write(
                                    f"r={r} seed={seed_str} reward={float(rewards[r]):.6f} "
                                    f"advantage={float(adv[r]):.6f} mp4={mp4_str}\n"
                                )

                    # Policy update (proxy logprob via -MSE on e_t window)
                    assert opt is not None
                    opt.zero_grad(set_to_none=True)
                    logps: List[torch.Tensor] = []
                    for r in range(int(num_rollouts)):
                        if r == 0:
                            x_r = base_window
                        else:
                            x_r = base_window + torch.randn_like(base_window) * float(rollout_noise_scale)
                        _, _, e_t_cur = _fifo_step_with_grads(
                            model=model,
                            sampler=sampler,
                            latents_window_bcthw=x_r,
                            timesteps=t,
                            indices=idx,
                            cond=cond,
                            uc=uc,
                            guidance_scale=float(guidance_scale),
                        )
                        mse = torch.mean((e_t_cur - rollout_noise_preds[r].to(e_t_cur.device)) ** 2)
                        logps.append(-mse)

                    logps_t = torch.stack(logps)
                    loss = -(logps_t * adv.to(device=logps_t.device)).mean()
                    loss.backward()
                    if float(grad_clip) and float(grad_clip) > 0:
                        torch.nn.utils.clip_grad_norm_(params, max_norm=float(grad_clip))
                    opt.step()

                    print(
                        f"[FIFO-GRPO] outer={outer_i}/{outer_steps-1} "
                        f"meanR={mean_r.item():.4f} stdR={std_r.item():.4f} loss={loss.item():.6f}"
                    , flush=True)

                # Advance FIFO buffer for this rank using current model (no_grad).
                with torch.no_grad():
                    input_latents = fifo_latents[:, :, start_idx:end_idx].clone()
                    out_latents, _ = sampler.fifo_onestep(
                        cond=cond,
                        shape=noise_shape,
                        latents=input_latents,
                        timesteps=t,
                        indices=idx,
                        unconditional_guidance_scale=float(guidance_scale),
                        unconditional_conditioning=uc,
                        clean_cond=True,
                    )
                    if fifo_lookahead:
                        fifo_latents[:, :, midpoint_idx:end_idx] = out_latents[:, :, -(fifo_video_length_i // 2) :]
                    else:
                        fifo_latents[:, :, start_idx:end_idx] = out_latents

            # Move the window forward by one frame (FIFO)
            _ = model.decode_first_stage_2DAE(fifo_latents[:, :, [first_frame_idx]])  # warm decode; optional
            fifo_latents = _fifo_shift_latents(fifo_latents)

    else:
        # -----------------------
        # DDIM GRPO (original)
        # -----------------------
        sampler.make_schedule(ddim_num_steps=int(num_inference_steps), ddim_eta=float(ddim_eta), verbose=False)
        ddim_timesteps = np.array(sampler.ddim_timesteps)  # ascending; sampling uses flipped
        time_range = list(np.flip(ddim_timesteps))

        num_steps = int(num_inference_steps)
        grpo_steps = int(min(int(num_grpo_steps), num_steps))
        train_indices = set(range(num_steps - grpo_steps, num_steps))
        print(f"GRPO will train on DDIM loop indices: {sorted(train_indices)}")

        # Start from a fixed seed noise for this prompt
        g = torch.Generator(device=device)
        g.manual_seed(int(seed))
        latents = torch.randn((1,) + latent_shape, device=device, generator=g)

        for i, step_t in enumerate(time_range):
            t = torch.full((1,), int(step_t), device=device, dtype=torch.long)

            # Early steps: just advance latents (no training)
            if i not in train_indices:
                with torch.no_grad():
                    latents, _ = sampler.p_sample_ddim(
                        latents,
                        cond,
                        t,
                        index=i,
                        unconditional_guidance_scale=float(guidance_scale),
                        unconditional_conditioning=uc,
                        temperature=1.0,
                        noise_dropout=0.0,
                        eta=float(ddim_eta),
                        clean_cond=True,
                        temporal_length=latent_shape[1],
                    )
                continue

            # -----------------------
            # Rollouts (no_grad): collect (e_t_rollout, reward)
            # -----------------------
            rollout_noise_preds: List[torch.Tensor] = []
            rollout_rewards: List[torch.Tensor] = []
            rollout_paths: List[str] = []
            rollout_seeds: List[int] = []

            for r in range(int(num_rollouts)):
                with torch.no_grad():
                    rollout_seed = int(seed) + int(i) * 1000 + int(r)
                    torch.manual_seed(rollout_seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(rollout_seed)
                    rollout_seeds.append(rollout_seed)

                    if r == 0:
                        x_r = latents
                    else:
                        x_r = latents + torch.randn_like(latents) * float(rollout_noise_scale)

                    _, pred_x0_r, e_t_r = _ddim_step_with_grads(
                        model=model,
                        x=x_r,
                        t=t,
                        index=i,
                        sampler=sampler,
                        cond=cond,
                        uc=uc,
                        guidance_scale=float(guidance_scale),
                    )
                    rollout_noise_preds.append(e_t_r.detach())

                    # Decode pred_x0 (x0 estimate) to pixel video and score it.
                    video_r = model.decode_first_stage_2DAE(pred_x0_r)  # [B,C,T,H,W]

                    if save_intermediate_rollouts and (i % max(int(save_rollout_every), 1) == 0):
                        out_mp4 = rollout_dir / f"rollout_step{i:03d}_r{r:02d}.mp4"
                        _save_mp4(video_r, out_mp4, fps=int(fps))
                        rollout_paths.append(str(out_mp4))

                    rew = _score_x0_video(video_r)
                    rollout_rewards.append(rew.detach().to(dtype=torch.float32))

            rewards = torch.stack(rollout_rewards)  # [K]
            mean_r = rewards.mean()
            std_r = rewards.std()
            if bool(normalize_advantages) and float(std_r) > 1e-8:
                adv = (rewards - mean_r) / (std_r + 1e-4)
            else:
                adv = rewards - mean_r

            if save_intermediate_rollouts and (i % max(int(save_rollout_every), 1) == 0):
                with (logs_dir / f"timestep_{i:03d}.txt").open("w", encoding="utf-8") as f:
                    f.write(f"ddim_index={i}\n")
                    f.write(f"t={int(step_t)}\n")
                    f.write(f"mean_reward={float(mean_r):.6f}\n")
                    f.write(f"std_reward={float(std_r):.6f}\n")
                    for r in range(int(num_rollouts)):
                        seed_str = str(rollout_seeds[r]) if r < len(rollout_seeds) else "NA"
                        mp4_str = rollout_paths[r] if r < len(rollout_paths) else "NA"
                        f.write(
                            f"r={r} seed={seed_str} reward={float(rewards[r]):.6f} "
                            f"advantage={float(adv[r]):.6f} mp4={mp4_str}\n"
                        )

            # -----------------------
            # Policy gradient update (proxy logprob via -MSE)
            # -----------------------
            assert opt is not None
            opt.zero_grad(set_to_none=True)

            logps: List[torch.Tensor] = []
            for r in range(int(num_rollouts)):
                if r == 0:
                    x_r = latents
                else:
                    x_r = latents + torch.randn_like(latents) * float(rollout_noise_scale)

                _, _, e_t_cur = _ddim_step_with_grads(
                    model=model,
                    x=x_r,
                    t=t,
                    index=i,
                    sampler=sampler,
                    cond=cond,
                    uc=uc,
                    guidance_scale=float(guidance_scale),
                )
                mse = torch.mean((e_t_cur - rollout_noise_preds[r].to(e_t_cur.device)) ** 2)
                logps.append(-mse)

            logps_t = torch.stack(logps)  # [K]
            loss = -(logps_t * adv.to(device=logps_t.device)).mean()
            loss.backward()

            if float(grad_clip) and float(grad_clip) > 0:
                torch.nn.utils.clip_grad_norm_(params, max_norm=float(grad_clip))
            opt.step()

            print(
                f"[GRPO] i={i}/{num_steps-1} t={int(step_t)} "
                f"meanR={mean_r.item():.4f} stdR={std_r.item():.4f} loss={loss.item():.6f}"
            )

            # Advance the main latents using current model (unperturbed), so the chain continues.
            with torch.no_grad():
                latents, _ = sampler.p_sample_ddim(
                    latents,
                    cond,
                    t,
                    index=i,
                    unconditional_guidance_scale=float(guidance_scale),
                    unconditional_conditioning=uc,
                    temperature=1.0,
                    noise_dropout=0.0,
                    eta=float(ddim_eta),
                    clean_cond=True,
                    temporal_length=latent_shape[1],
                )

    # ------------------------------------------------------------------
    # Sample after GRPO
    # ------------------------------------------------------------------
    print("Generating sample (after GRPO)...")
    if mode_sampling == "fifo":
        after_video = sample_fifo_decode(
            model=model,
            sampler=sampler,
            cond=cond,
            uc=uc,
            height=int(height),
            width=int(width),
            seed=int(seed),
            guidance_scale=float(guidance_scale),
            ddim_eta=float(ddim_eta),
            fifo_video_length=int(fifo_video_length),
            fifo_new_video_length=int(fifo_new_video_length),
            fifo_num_partitions=int(fifo_num_partitions),
            fifo_lookahead_denoising=bool(fifo_lookahead_denoising),
        )
    else:
        after_video, _ = _ddim_sample_decode(
            model=model,
            sampler=sampler,
            cond=cond,
            uc=uc,
            shape=latent_shape,
            num_steps=int(num_inference_steps),
            eta=float(ddim_eta),
            guidance_scale=float(guidance_scale),
            seed=int(seed),
        )
    _save_mp4(after_video, out_dir / "after.mp4", fps=int(fps))

