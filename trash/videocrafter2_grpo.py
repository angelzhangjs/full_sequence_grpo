#!/usr/bin/env python3
"""
GRPO fine-tuning demo for VideoCrafter2 (from the vendored `scaling-noise/` code).
This script:
  - loads the VideoCrafter2 latent diffusion model + checkpoint
  - runs a baseline DDIM sample for a prompt and saves an MP4
  - runs GRPO-style updates on the last N DDIM steps using per-rollout rewards
  - runs another sample and saves an MP4 for comparison

Notes:
  - This is a lightweight research script (single GPU, single prompt, batch=1).
  - It does NOT implement full VideoCrafter2 training; it only fine-tunes selected parameters
    (by default, temporal modules) using a GRPO-style objective.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import imageio
import numpy as np
import torch
import yaml

try:
    from omegaconf import OmegaConf  # type: ignore
except Exception:  # pragma: no cover
    OmegaConf = None  # type: ignore

# Reuse ScalingNoise / VideoCrafter2 components (vendored in this repo under `scaling-noise/`).
# Note: the folder name contains a hyphen, so we add it to sys.path and import like the original repo.
_ROOT = Path(__file__).resolve().parent
_SCALING_NOISE_ROOT = _ROOT / "scaling-noise"
sys.path.insert(0, str(_SCALING_NOISE_ROOT))

from utils_loc.utils import instantiate_from_config  # type: ignore  # noqa: E402
from lvdm.models.samplers.ddim import DDIMSampler  # type: ignore  # noqa: E402

# Reuse this repo's reward function (CLIP/DINO-based).
from reward_functions import reward_function


def _require_pkg(name: str, install_hint: str) -> None:
    try:
        __import__(name)
    except Exception as e:
        raise RuntimeError(
            f"Missing dependency `{name}` ({e}).\n\n"
            f"Install hint:\n{install_hint}\n\n"
            f"If you installed `scaling-noise/requirements.txt` into the dedicated conda env, run:\n"
            f"  /home/ubuntu/anaconda3/bin/conda run -n scaling-noise python videocrafter2_grpo.py --help\n"
        ) from e


def _load_model_checkpoint(model, ckpt: str, *, strict: bool = True):
    """
    Minimal checkpoint loader copied from ScalingNoise's `load_model_checkpoint`,
    without importing heavy deps from `scripts/evaluation/funcs_search.py`.
    """
    state_dict = torch.load(ckpt, map_location="cpu")
    full_strict = bool(strict)
    if "module" in state_dict:
        new_pl_sd = {}
        for key in state_dict["module"].keys():
            # Strip "module.model." prefix used by some checkpoints.
            new_pl_sd[key[16:]] = state_dict["module"][key]
        model.load_state_dict(new_pl_sd, strict=full_strict)
    else:
        if "state_dict" in list(state_dict.keys()):
            state_dict = state_dict["state_dict"]
        model.load_state_dict(state_dict, strict=full_strict)
    return model


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


def _ensure_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script.")
    return torch.device("cuda")


def _load_videocrafter2_model(cfg_path: str, ckpt_path: str, device: torch.device):
    # VideoCrafter2 configs are written for OmegaConf (attribute access like `unet_config.params.temporal_length`).
    # Use OmegaConf when available; fall back to YAML only if necessary.
    if OmegaConf is not None:
        cfg = OmegaConf.load(cfg_path)
        model_cfg = cfg.get("model", None)
    else:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg_dict = yaml.safe_load(f)
        model_cfg = cfg_dict.get("model", None)
        if model_cfg is not None:
            raise RuntimeError(
                "This VideoCrafter2 config requires OmegaConf for attribute-style access "
                "(e.g. `unet_config.params.*`). Please install `omegaconf` or run in the "
                "`scaling-noise` conda env where it's installed."
            )

    if model_cfg is None:
        raise ValueError(f"Missing top-level 'model' key in config: {cfg_path}")
    model = instantiate_from_config(model_cfg)
    model = model.to(device)

    # Allow passing a directory like `base_512_v2/` and auto-pick the first *.ckpt inside.
    if os.path.isdir(ckpt_path):
        ckpt_dir = ckpt_path
        candidates = sorted(
            [
                os.path.join(ckpt_dir, f)
                for f in os.listdir(ckpt_dir)
                if f.lower().endswith(".ckpt")
            ]
        )
        if len(candidates) == 0:
            raise FileNotFoundError(f"No .ckpt files found in directory: {ckpt_dir}")
        ckpt_path = candidates[0]

    # Backwards-compatible fallback for the legacy path used in the vendored scaling-noise repo.
    if not os.path.exists(ckpt_path):
        legacy = "base_512_v2/model.ckpt"
        if os.path.exists(legacy):
            ckpt_path = legacy
        else:
            raise FileNotFoundError(f"VideoCrafter2 checkpoint not found: {ckpt_path}")
    model = _load_model_checkpoint(model, ckpt_path, strict=True)
    model.eval()
    return model


def _make_conditioning(model, prompt: str, fps: int, batch_size: int = 1) -> Dict:
    prompts = [prompt] * batch_size
    text_emb = model.get_learned_conditioning(prompts)
    fps_t = torch.tensor([fps] * batch_size, device=model.device).long()
    return {"c_crossattn": [text_emb], "fps": fps_t}


def _make_unconditional_conditioning(model, cond: Dict, batch_size: int = 1) -> Dict | None:
    # Mirrors scaling-noise/scripts/evaluation/funcs_search.py logic.
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

    for name, p in model.named_parameters():
        if mode == "all":
            p.requires_grad = True
            trainable.append(p)
        elif mode == "temporal":
            if "temporal" in name.lower():
                p.requires_grad = True
                trainable.append(p)

    # The real denoiser is nested at model.model.diffusion_model; warn if we selected nothing.
    if mode != "none" and len(trainable) == 0:
        print("⚠️ Warning: no parameters matched trainable selection; nothing will be updated.")
    else:
        print(f"Trainable parameters: {len(trainable)}")
    return trainable


def _to_uint8_video(frames_bcthw: torch.Tensor) -> np.ndarray:
    """
    Input: [B,C,T,H,W] in [-1, 1] (typical latent diffusion decode).
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
    imageio.mimsave(str(out_path), list(vid), fps=fps)


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
    This mirrors the math in DDIMSampler.p_sample_ddim but keeps autograd enabled.
    """
    device = x.device
    b = x.shape[0]
    is_video = x.ndim == 5

    # Model prediction (epsilon / noise-pred for eps-parameterization)
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

    if is_video:
        size = (b, 1, 1, 1, 1)
    else:
        size = (b, 1, 1, 1)

    a_t = torch.full(size, alphas[index], device=device, dtype=x.dtype)
    a_prev = torch.full(size, alphas_prev[index], device=device, dtype=x.dtype)
    # `sampler.ddim_sigmas` already includes the configured eta from `make_schedule(ddim_eta=...)`.
    sigma_t = torch.full(size, sigmas[index], device=device, dtype=x.dtype)
    sqrt_one_minus_at = torch.full(size, sqrt_one_minus_alphas[index], device=device, dtype=x.dtype)

    pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
    dir_xt = (1.0 - a_prev - sigma_t**2).clamp(min=0.0).sqrt() * e_t
    noise = sigma_t * torch.randn_like(x)
    x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise

    return x_prev, pred_x0, e_t


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

    # DDIM timesteps go from large->small; use p_sample_ddim for correctness & speed.
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


def main() -> int:
    args = _parse_args()
    device = _ensure_cuda()

    # Preflight check: VideoCrafter2 code depends on PyTorch Lightning (imported by scaling-noise/lvdm/models/ddpm3d.py).
    _require_pkg(
        "pytorch_lightning",
        "If you're using the ltx-grpo env, either switch envs or install:\n"
        "  /home/ubuntu/anaconda3/envs/ltx-grpo/bin/python -m pip install pytorch_lightning==2.1.3\n"
        "Recommended (isolated): use the `scaling-noise` conda env.",
    )

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    model = _load_videocrafter2_model(args.config, args.ckpt_path, device)

    # Sanity: config expects multiples of 16.
    if args.height % 16 != 0 or args.width % 16 != 0:
        raise ValueError("height/width must be multiples of 16 for this VideoCrafter2 config.")
    if args.num_frames != 16:
        print("⚠️ Note: scaling-noise VideoCrafter2 config is usually trained for 16 frames. Proceeding anyway.")

    # Latent shape: channels=4, spatial downsample=8
    latent_h, latent_w = args.height // 8, args.width // 8
    latent_shape = (4, args.num_frames, latent_h, latent_w)

    cond = _make_conditioning(model, args.prompt, fps=args.fps, batch_size=1)
    uc = _make_unconditional_conditioning(model, cond, batch_size=1) if args.guidance_scale != 1.0 else None

    sampler = DDIMSampler(model)

    # ------------------------------------------------------------------
    # Baseline sample (before)
    # ------------------------------------------------------------------
    print("Generating baseline sample (before GRPO)...")
    before_video, _ = _ddim_sample_decode(
        model=model,
        sampler=sampler,
        cond=cond,
        uc=uc,
        shape=latent_shape,
        num_steps=args.num_inference_steps,
        eta=args.ddim_eta,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
    )
    before_path = out_dir / "before.mp4"
    _save_mp4(before_video, before_path, fps=args.fps)
    print(f"Saved: {before_path}")

    # ------------------------------------------------------------------
    # GRPO tuning on last N steps
    # ------------------------------------------------------------------
    trainable = _select_trainable_params(model, args.trainable)
    if len(trainable) == 0:
        print("Nothing to train; exiting after baseline.")
        return 0

    opt = torch.optim.AdamW(trainable, lr=args.lr)

    # Build DDIM schedule once for training.
    sampler.make_schedule(ddim_num_steps=args.num_inference_steps, ddim_eta=args.ddim_eta, verbose=False)

    ddim_timesteps = np.array(sampler.ddim_timesteps)  # ascending in original code; we use flipped for sampling
    # p_sample_ddim iterates over i = 0..S-1 with step_t = flipped(ddim_timesteps).
    # "last N steps" in terms of generation are the final iterations (closest to t=0),
    # i.e. the last N elements of the loop index i.
    num_steps = int(args.num_inference_steps)
    grpo_steps = int(min(args.num_grpo_steps, num_steps))
    train_indices = list(range(num_steps - grpo_steps, num_steps))
    print(f"GRPO will train on DDIM loop indices: {train_indices} (total {len(train_indices)})")

    # Run early steps with no_grad to reach the training region.
    g = torch.Generator(device=device)
    g.manual_seed(args.seed)
    latents = torch.randn((1,) + latent_shape, device=device, generator=g)

    time_range = list(np.flip(ddim_timesteps))

    for i, step_t in enumerate(time_range):
        t = torch.full((1,), int(step_t), device=device, dtype=torch.long)

        if i not in train_indices:
            with torch.no_grad():
                latents, _ = sampler.p_sample_ddim(
                    latents,
                    cond,
                    t,
                    index=i,
                    unconditional_guidance_scale=args.guidance_scale,
                    unconditional_conditioning=uc,
                    temperature=1.0,
                    noise_dropout=0.0,
                    eta=args.ddim_eta,
                    clean_cond=True,
                    temporal_length=latent_shape[1],
                )
            continue

        # -----------------------
        # Rollouts (no_grad)
        # -----------------------
        rollout_noise_preds: List[torch.Tensor] = []
        rollout_rewards: List[torch.Tensor] = []

        for r in range(int(args.num_rollouts)):
            with torch.no_grad():
                if r == 0:
                    x_r = latents
                else:
                    x_r = latents + torch.randn_like(latents) * float(args.rollout_noise_scale)

                # compute rollout e_t (action) and pred_x0 (for reward) using the sampler math
                # using sampler.p_sample_ddim would require extra decoding of pred_x0; we compute directly here:
                # note: this uses model.apply_model internally, but no_grad for rollout collection.
                x_prev_r, pred_x0_r, e_t_r = _ddim_step_with_grads(
                    model=model,
                    x=x_r,
                    t=t,
                    index=i,
                    sampler=sampler,
                    cond=cond,
                    uc=uc,
                    guidance_scale=args.guidance_scale,
                )
                rollout_noise_preds.append(e_t_r.detach())

                # Decode pred_x0 to frames and compute reward.
                video_r = model.decode_first_stage_2DAE(pred_x0_r)
                # reward_function expects [C,T,H,W] or [B,C,T,H,W]; pass [C,T,H,W]
                video_cthw = video_r[0].float()
                rew = reward_function(video_cthw, prompt=args.prompt, device="cuda")
                rollout_rewards.append(rew.detach().to(dtype=torch.float32))

        rewards = torch.stack(rollout_rewards)  # [K]
        if int(args.normalize_advantages) == 1:
            std = rewards.std()
            adv = (rewards - rewards.mean()) / (std + 1e-4) if std > 1e-8 else (rewards - rewards.mean())
        else:
            adv = rewards - rewards.mean()

        # -----------------------
        # Policy gradient update
        # -----------------------
        opt.zero_grad(set_to_none=True)

        # Compute logprob proxy per rollout with grads:
        # logpi_k ~ -MSE(e_t_rollout, e_t_current(x_r))
        logps: List[torch.Tensor] = []
        for r in range(int(args.num_rollouts)):
            if r == 0:
                x_r = latents
            else:
                x_r = latents + torch.randn_like(latents) * float(args.rollout_noise_scale)

            _, _, e_t_cur = _ddim_step_with_grads(
                model=model,
                x=x_r,
                t=t,
                index=i,
                sampler=sampler,
                cond=cond,
                uc=uc,
                guidance_scale=args.guidance_scale,
            )
            mse = torch.mean((e_t_cur - rollout_noise_preds[r].to(e_t_cur.device)) ** 2)
            logps.append(-mse)

        logps_t = torch.stack(logps)  # [K]
        loss = -(logps_t * adv.to(device=logps_t.device)).mean()

        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=float(args.grad_clip))
        opt.step()

        print(
            f"[GRPO] step_idx={i}/{num_steps-1} t={int(step_t)} "
            f"reward_mean={rewards.mean().item():.4f} reward_std={rewards.std().item():.4f} loss={loss.item():.6f}"
        )

        # Advance the main latents using current model (unperturbed).
        with torch.no_grad():
            latents, _ = sampler.p_sample_ddim(
                latents,
                cond,
                t,
                index=i,
                unconditional_guidance_scale=args.guidance_scale,
                unconditional_conditioning=uc,
                temperature=1.0,
                noise_dropout=0.0,
                eta=args.ddim_eta,
                clean_cond=True,
                temporal_length=latent_shape[1],
            )

    # ------------------------------------------------------------------
    # Sample after GRPO
    # ------------------------------------------------------------------
    print("Generating sample (after GRPO)...")
    after_video, _ = _ddim_sample_decode(
        model=model,
        sampler=sampler,
        cond=cond,
        uc=uc,
        shape=latent_shape,
        num_steps=args.num_inference_steps,
        eta=args.ddim_eta,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
    )
    after_path = out_dir / "after.mp4"
    _save_mp4(after_video, after_path, fps=args.fps)
    print(f"Saved: {after_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

