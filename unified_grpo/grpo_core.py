from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

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
    solver_state: Optional[Any] = None

    last_loss = 0.0
    last_mean_r = 0.0
    last_std_r = 0.0

    for i, t in enumerate(timesteps):
        print(f"\n{'='*50}")
        print(f"Step {i+1}/{len(timesteps)} | Timestep: {t:.4f}")
        print(f"{'='*50}")
        
        ctx = StepContext(step_index=int(i), t=t)

        # Early steps: deterministic-ish step to reach a reasonable state.
        if i < last_start:
            print(f"  Early step (warmup, no GRPO)")

            with torch.no_grad():
                out = adapter.step(
                    latents=latents,
                    ctx=ctx,
                    with_grad=False,
                    rollout_noise_scale=0.0,
                    rollout_index=0,
                    solver_state=solver_state,
                )
            latents = out.next_latents.detach()
            solver_state = out.solver_state
            continue

        # ---------------------------
        # 1) Collect rollouts (no grad)
        # ---------------------------
        print(f"  Generating {cfg.num_rollouts} rollouts...")
        
        rollout_actions: List[torch.Tensor] = []
        rollout_next_latents: List[torch.Tensor] = []
        rollout_solver_states: List[Optional[Any]] = []
        rollout_rewards: List[torch.Tensor] = []

        with torch.no_grad():
            for r in range(int(cfg.num_rollouts)):
                print(f"    Rollout {r+1}/{cfg.num_rollouts}...", end=" ", flush=True)
                out_r = adapter.step(
                    latents=latents,
                    ctx=ctx,
                    with_grad=False,
                    rollout_noise_scale=float(cfg.rollout_noise_scale),
                    rollout_index=int(r),
                    solver_state=solver_state,
                )

                # Prefer x0_latents if provided; otherwise reward on next_latents decode.
                to_decode = out_r.x0_latents if out_r.x0_latents is not None else out_r.next_latents
                video = adapter.decode_for_reward(latents_or_x0=to_decode, x0_is_patchified=True)
                rew = reward_fn(video, prompt)

                rollout_actions.append(out_r.action.detach())
                rollout_next_latents.append(out_r.next_latents.detach())
                rollout_solver_states.append(out_r.solver_state)
                rollout_rewards.append(rew.detach().float().to(device))
                
                print(f"reward={rew.item():.4f}")

        rewards_t = torch.stack(rollout_rewards)  # [K]
        adv = _normalize_advantages(rewards_t, normalize=bool(cfg.normalize_advantages))

        last_mean_r = float(rewards_t.mean().item())
        last_std_r = float(rewards_t.std(unbiased=False).item())
        
        print(f"  Rewards: mean={last_mean_r:.4f}, std={last_std_r:.4f}")

        # ---------------------------
        # 2) Compute GRPO loss (with grad)
        # ---------------------------
        print("\n  Computing GRPO loss and updating model...")
        
        # CRITICAL: Clear all gradients and free old graphs
        opt.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        
        # NUCLEAR OPTION: Recreate latents from scratch to ensure NO graph
        with torch.no_grad():
            # Copy data but create completely new tensor
            fresh_latents = torch.zeros_like(latents)
            fresh_latents.copy_(latents)
        
        # Forward pass with gradients (fresh graph)
        out_cur = adapter.step(
            latents=fresh_latents,
            ctx=ctx,
            with_grad=True,
            rollout_noise_scale=0.0,
            rollout_index=0,
            solver_state=None,
        )
        action_cur = out_cur.action.clone()  # Clone action too

        # Gaussian-policy surrogate log-prob: higher if action_cur is close to sampled rollout action.
        # logp(a_r | pi_theta) ~ -||a_r - a_theta||^2 / (2*sigma^2)
        sigma2 = float(cfg.logprob_sigma) ** 2
        logps: List[torch.Tensor] = []
        for a_r in rollout_actions:
            # Detach rollout actions (from previous graphs)
            mse = torch.mean((action_cur - a_r.detach()) ** 2)
            logps.append(-mse / (2.0 * sigma2))
        logps_t = torch.stack(logps)  # [K] - only connected to action_cur
        
        # Loss with detached advantages (from rewards, no grad needed)
        loss = -(adv.detach() * logps_t).mean()  # logps_t keeps action_cur gradients!
        
        # Backward pass (retain_graph=False to free memory immediately!)
        print(f"  Backward pass...")
        loss.backward(retain_graph=False)
        
        torch.nn.utils.clip_grad_norm_(params, float(cfg.grad_clip))
        opt.step()

        last_loss = float(loss.detach().cpu().item())
        
        # ---------------------------
        # 3) Advance trajectory using best rollout BEFORE deleting anything!
        # ---------------------------
        best = int(torch.argmax(rewards_t).item())
        # CRITICAL: Detach AND clone to break ALL graph connections!
        latents = rollout_next_latents[best].detach().clone()
        solver_state = rollout_solver_states[best]
        
        # NOW delete tensors (after using them!)
        del loss, out_cur, action_cur, logps_t, logps
        del rollout_actions, rollout_next_latents, rollout_solver_states, rollout_rewards
        del rewards_t, adv
        
        # Aggressive memory clearing
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        print(f"  ✅ Model updated | Loss: {last_loss:.4f}")
        print(f"  GPU free: {torch.cuda.mem_get_info()[0] / 1024**3:.1f} GB")

    # Save final video as MP4 (watchable format)
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            final_video = adapter.decode_for_reward(latents_or_x0=latents, x0_is_patchified=True).detach().cpu()
        
        # Convert tensor to MP4 video
        import imageio
        import numpy as np
        
        # Handle tensor format
        video_np = final_video.float().numpy()
        if len(video_np.shape) == 5:  # [B, T, C, H, W]
            video_np = video_np[0]
        if video_np.shape[1] == 3:  # [T, 3, H, W]
            video_np = video_np.transpose(0, 2, 3, 1)  # → [T, H, W, 3]
        
        # Debug normalization
        print(f"  [DEBUG] Video range before norm: [{video_np.min():.3f}, {video_np.max():.3f}]")
        
        # Normalize to [0, 1] first
        if video_np.max() > 10:  # Likely unnormalized (0-255 range)
            print("  Converting from [0, 255] → [0, 1]")
            video_np = video_np / 255.0
        elif video_np.min() < -0.5:  # Likely [-1, 1] range
            print("  Converting from [-1, 1] → [0, 1]")
            video_np = (video_np + 1) / 2
        elif video_np.max() <= 1.1:  # Already [0, 1]
            print("  Already in [0, 1] range")
            pass
        else:  # Unknown range, normalize
            print(f"  Unknown range, normalizing: [{video_np.min()}, {video_np.max()}]")
            video_np = (video_np - video_np.min()) / (video_np.max() - video_np.min() + 1e-8)
        
        # Convert to uint8 [0, 255]
        video_np = (video_np * 255).clip(0, 255).astype(np.uint8)
        print(f"  [DEBUG] Video range after norm: [{video_np.min()}, {video_np.max()}]")
        
        # Save as MP4
        mp4_path = out_dir / "final_grpo.mp4"
        writer = imageio.get_writer(str(mp4_path), fps=8, codec='libx264', quality=8)
        for frame in video_np:
            writer.append_data(frame)
        writer.close()
        
        print(f"\n✅ Final video saved: {mp4_path}")
        print(f"   Frames: {len(video_np)}, Duration: {len(video_np)/8:.2f}s")
        
        # Also save tensor for analysis
        torch.save(final_video, out_dir / "final_video.pt")

    return {
        "last_loss": float(last_loss),
        "last_mean_reward": float(last_mean_r),
        "last_std_reward": float(last_std_r),
        "num_inference_steps": float(cfg.num_inference_steps),
        "num_grpo_steps": float(cfg.num_grpo_steps),
        "num_rollouts": float(cfg.num_rollouts),
    }

