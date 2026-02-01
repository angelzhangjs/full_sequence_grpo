from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch

from unified_grpo.adapters.base import StepContext, VideoGRPOAdapter

RewardFn = Callable[[torch.Tensor, str], torch.Tensor]

@dataclass
class GRPOConfig:
    num_inference_steps: int = 40
    num_grpo_steps: int = 25
    num_rollouts: int = 3
    rollout_noise_scale: float = 0.5
    lr: float = 1e-5
    grad_clip: float = 1.0
    normalize_advantages: bool = True
    logprob_sigma: float = 1.0  # Gaussian-policy surrogate scale for action logp


def _normalize_advantages(rewards: torch.Tensor, *, normalize: bool) -> torch.Tensor:
    mean = rewards.mean()
    std = rewards.std(unbiased=False)
    adv = rewards - mean
    if normalize and float(std) > 1e-8:
        adv = adv / (std + 1e-4)
    return adv


def run_grpo_for_prompt(
    *,
    adapter: VideoGRPOAdapter,
    prompt: str,
    reward_fn: RewardFn,
    seed: int,
    out_dir: Optional[Path] = None,
    cfg: GRPOConfig = GRPOConfig(),
) -> Dict[str, float]:
    """
    Unified GRPO loop over the last N timesteps.

    - Early steps: run normally with no grads.
    - Last `num_grpo_steps`: collect K rollouts (no grads), score, compute advantages,
      then recompute current action with grads and apply GRPO-style policy gradient.

    Returns summary metrics.
    """
    device = adapter.device()
    timesteps = adapter.get_timesteps(num_inference_steps=int(cfg.num_inference_steps))
    if len(timesteps) == 0:
        raise ValueError("Adapter returned empty timesteps.")

    last_start = max(0, len(timesteps) - int(cfg.num_grpo_steps))

    # Prepare trainable params + optimizer.
    params = adapter.trainable_parameters()
    if len(params) == 0:
        raise ValueError(f"Adapter '{adapter.name}' returned 0 trainable parameters.")
    opt = torch.optim.AdamW(params, lr=float(cfg.lr), betas=(0.9, 0.999), weight_decay=0.01)

    latents = adapter.prepare_latents(seed=int(seed)).to(device)

    last_loss = 0.0
    last_mean_r = 0.0
    last_std_r = 0.0

    for i, t in enumerate(timesteps):
        ctx = StepContext(step_index=int(i), t=t)

        # Early steps: deterministic-ish step to reach a reasonable state.
        if i < last_start:
            with torch.no_grad():
                out = adapter.step(
                    latents=latents,
                    ctx=ctx,
                    with_grad=False,
                    rollout_noise_scale=0.0,
                    rollout_index=0,
                )
            latents = out.next_latents.detach()
            continue

        # ---------------------------
        # 1) Collect rollouts (no grad)
        # ---------------------------
        rollout_actions: List[torch.Tensor] = []
        rollout_next_latents: List[torch.Tensor] = []
        rollout_rewards: List[torch.Tensor] = []

        with torch.no_grad():
            for r in range(int(cfg.num_rollouts)):
                out_r = adapter.step(
                    latents=latents,
                    ctx=ctx,
                    with_grad=False,
                    rollout_noise_scale=float(cfg.rollout_noise_scale),
                    rollout_index=int(r),
                )

                # Prefer x0_latents if provided; otherwise reward on next_latents decode.
                to_decode = out_r.x0_latents if out_r.x0_latents is not None else out_r.next_latents
                video = adapter.decode_for_reward(latents_or_x0=to_decode, x0_is_patchified=True)
                rew = reward_fn(video, prompt)

                rollout_actions.append(out_r.action.detach())
                rollout_next_latents.append(out_r.next_latents.detach())
                rollout_rewards.append(rew.detach().float().to(device))

        rewards_t = torch.stack(rollout_rewards)  # [K]
        adv = _normalize_advantages(rewards_t, normalize=bool(cfg.normalize_advantages))

        last_mean_r = float(rewards_t.mean().item())
        last_std_r = float(rewards_t.std(unbiased=False).item())

        # ---------------------------
        # 2) Compute GRPO loss (with grad)
        # ---------------------------
        opt.zero_grad(set_to_none=True)

        out_cur = adapter.step(
            latents=latents,
            ctx=ctx,
            with_grad=True,
            rollout_noise_scale=0.0,
            rollout_index=0,
        )
        action_cur = out_cur.action

        # Gaussian-policy surrogate log-prob: higher if action_cur is close to sampled rollout action.
        # logp(a_r | pi_theta) ~ -||a_r - a_theta||^2 / (2*sigma^2)
        sigma2 = float(cfg.logprob_sigma) ** 2
        logps: List[torch.Tensor] = []
        for a_r in rollout_actions:
            mse = torch.mean((action_cur - a_r) ** 2)
            logps.append(-mse / (2.0 * sigma2))
        logps_t = torch.stack(logps)  # [K]

        loss = -(adv.detach() * logps_t).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, float(cfg.grad_clip))
        opt.step()

        last_loss = float(loss.detach().cpu().item())

        # ---------------------------
        # 3) Advance trajectory using best rollout (stabilizes generation)
        # ---------------------------
        best = int(torch.argmax(rewards_t).item())
        latents = rollout_next_latents[best].detach()

    return {
        "last_loss": float(last_loss),
        "last_mean_reward": float(last_mean_r),
        "last_std_reward": float(last_std_r),
        "num_inference_steps": float(cfg.num_inference_steps),
        "num_grpo_steps": float(cfg.num_grpo_steps),
        "num_rollouts": float(cfg.num_rollouts),
    }

