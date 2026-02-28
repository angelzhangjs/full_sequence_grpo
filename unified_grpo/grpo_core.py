from __future__ import annotations, generator_stop
import random as random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple
import torch
from unified_grpo.adapters.base import StepContext, VideoGRPOAdapter
from unified_grpo.utils import save_video_tensor_as_mp4

RewardFn = Callable[[torch.Tensor, str], torch.Tensor]

@dataclass
class GRPOConfig:
    num_inference_steps: int = 50
    num_grpo_steps: int = 15
    num_rollouts: int = 3
    rollout_noise_scale: float = 0.5
    lr: float = 1e-4
    grad_clip: float = 1.0
    normalize_advantages: bool = True
    logprob_sigma: float = 1.0  # Gaussian-policy surrogate scale for action logp

    # ── Rollout diversity controls ────────────────────────────────────────────────
    # Per-rollout guidance scales. If provided, each rollout r uses guidance_scales[r].
    # If None (default), guidance scale is sampled randomly from rollout_guidance_scale_range.
    # Example: [7.5, 3.0, 12.0] for 3 rollouts (standard, loose, tight).
    rollout_guidance_scales: Optional[List[float]] = None

    # Range [min, max] to randomly sample per-rollout guidance scale when
    # rollout_guidance_scales is None. Set to None to disable random guidance scaling.
    # Example: (3.0, 12.0) → each rollout gets a random CFG in [3.0, 12.0].
    rollout_guidance_scale_range: Optional[Tuple[float, float]] = (3.0, 12.0)

    # LTX-only diversity controls (no-op for other adapters).
    # These are implemented by temporarily patching adapter attributes if they exist.
    #
    # - `stg_scale`: LTX "STG" (spatiotemporal guidance) strength.
    rollout_stg_scale_range: Optional[Tuple[float, float]] = None
    # - `rescaling_scale`: rescaling factor used in LTX guidance mixing.
    rollout_rescaling_scale_range: Optional[Tuple[float, float]] = None
    # - `cfg_star_rescale`: optional boolean knob; if prob is set, it will be sampled per-rollout.
    rollout_cfg_star_rescale_prob: Optional[float] = None

    # If True, each rollout uses a unique random seed for its exploration noise,
    # derived from the base training seed via random.Random(seed).
    # This ensures reproducible but diverse rollout noise across runs.
    per_rollout_seed: bool = True

    # ── Memory optimization ───────────────────────────────────────────────────────
    # If True, enable gradient checkpointing on the adapter's transformer before the
    # GRPO training loop. This trades ~30% extra compute for ~50% less activation
    # memory during the backward pass (the grad recompute step).
    gradient_checkpointing: bool = True # default to True for memory optimization

    # ── Output video saving ───────────────────────────────────────────────────────
    # If True, after training finishes we will re-run a full generation pass from
    # scratch (fresh noise) using the *post-training* weights, and save that MP4.
    # This is usually what you want to visually judge “did training improve the model?”
    save_post_training_video_from_scratch: bool = True

    # If True, also save a video decoded from the final latents of the GRPO training
    # trajectory (i.e., the sample obtained by advancing with the best rollout each step).
    # This can be useful for debugging, but is less comparable across runs.
    save_training_trajectory_video: bool = False


def generate_latents_from_scratch(
    *,
    adapter: VideoGRPOAdapter,
    seed: int,
    num_inference_steps: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Run a standard (no-grad) sampling pass from fresh noise using the adapter's
    current model weights.

    This is intentionally implemented using the adapter's `prepare_latents()` + `step()`
    so it works uniformly across CogVideoX / LTX / WAN / OpenSora.
    """
    timesteps = adapter.get_timesteps(num_inference_steps=int(num_inference_steps))
    latents = adapter.prepare_latents(seed=int(seed)).to(device)
    solver_state: Optional[Any] = None

    with torch.no_grad():
        for i, t in enumerate(timesteps):
            step_context = StepContext(step_index=int(i), t=t)
            out = adapter.step(
                latents=latents,
                step_context=step_context,
                with_grad=False,
                solver_state=solver_state,
            )
            latents = out.next_latents.detach()
            solver_state = out.solver_state

    return latents

def normalize_advantages(rewards: torch.Tensor, *, normalize: bool) -> torch.Tensor:
    """ normalize the advantages across all rollouts per timestep, shape [K]
    Args:
        rewards: the rewards tensor, input rewards should be a 1D torch.tensor with shape [K] for K rollouts, e.g. shape[K], typically flaot 32 of the training device. 
        normalize: whether to normalize the advantages
    Returns:
        the normalized advantages across all rollouts per timestep, shape [K]
    """
    mean = rewards.mean()
    std = rewards.std(unbiased=False)
    adv = rewards - mean
    if normalize and float(std) > 1e-8:
        adv = adv / (std + 1e-4)
    return adv

class RolloutResult(NamedTuple):
    rollout_actions_cpu: List[torch.Tensor]      # per-rollout actions stored on CPU fp16
    rollout_rewards: List[torch.Tensor]          # per-rollout reward scalars (gpu float32)
    best_next_latents: torch.Tensor              # next_latents from the highest-reward rollout
    best_solver_state: Optional[Any]             # corresponding solver state
    best_input_latents_cpu: torch.Tensor         # input latents (possibly perturbed) for the best rollout, on CPU
    best_rollout_index: int                      # index of the best rollout (k*)


def collect_rollouts(
    *,
    adapter: VideoGRPOAdapter,
    latents: torch.Tensor,
    step_context: StepContext,
    solver_state: Optional[Any],
    reward_fn: RewardFn,
    prompt: str,
    num_rollouts: int,
    rollout_noise_scale: float,
    rollout_guidance_scales: Optional[List[float]],
    rollout_guidance_scale_range: Optional[Tuple[float, float]],
    rollout_stg_scale_range: Optional[Tuple[float, float]],
    rollout_rescaling_scale_range: Optional[Tuple[float, float]],
    rollout_cfg_star_rescale_prob: Optional[float],
    per_rollout_seed: bool,
    base_seed: int,
    step_index: int,
    device: torch.device,
) -> RolloutResult:
    """
    Collect K rollouts at the current diffusion timestep (no gradients).

    Diversity mechanisms (all combinable):
    1. rollout_noise_scale  – adds random noise to input latents (r > 0).
    2. rollout_guidance_scales – varies classifier-free guidance strength per rollout.
    3. per_rollout_seed – unique RNG seed per rollout (via random.Random(base_seed)),
                          giving reproducible but different noise patterns.

    For each rollout:
      - optionally perturb latents with seeded exploration noise
      - run one adapter.step() to get next_latents / x0_latents / action
      - decode x0 and compute reward
      - store action on CPU (fp16) to keep GPU VRAM low

    Returns a RolloutResult with all collected data.
    """
    rollout_actions_cpu: List[torch.Tensor] = []
    rollout_rewards: List[torch.Tensor] = []
    best_next_latents: Optional[torch.Tensor] = None
    best_solver_state: Optional[Any] = None
    best_reward: Optional[float] = None

    best_input_latents_cpu: Optional[torch.Tensor] = None

    # Seeded: derive a unique but reproducible integer seed per (step, rollout).
    # random.Random(base_seed) gives a deterministic sequence tied to the training seed.
    seed_generator = random.Random(int(base_seed))
    # Advance the RNG by step_index so seeds also vary across GRPO timesteps.
    for _ in range(int(step_index)):
        seed_generator.random()

    with torch.no_grad():
        for rollout_index in range(int(num_rollouts)):

            # ── 1) Per-rollout seed for exploration noise ──────────────────────
            rollout_seed = seed_generator.randint(0, 2**31 - 1)
            rollout_seed_generator = torch.Generator(device=device).manual_seed(rollout_seed) if per_rollout_seed else None
            guidance_scale_info = ""
            adapter_info = ""

            # ── 2) Guidance scale for this rollout ─────────────────────────────
            # Priority: explicit list → random range → keep adapter's default
            orig_guidance_scale = getattr(adapter, "guidance_scale", None)
            if rollout_guidance_scales is not None and rollout_index < len(rollout_guidance_scales):
                # Use the explicit per-rollout list.
                guidance_scale = float(rollout_guidance_scales[rollout_index])
            elif rollout_guidance_scale_range is not None:
                # Sample randomly from [min, max] using the seeded generator for reproducibility.
                guidance_scale_min, guidance_scale_max = float(rollout_guidance_scale_range[0]), float(rollout_guidance_scale_range[1])
                guidance_scale = seed_generator.uniform(guidance_scale_min, guidance_scale_max)
            else:
                guidance_scale = None  # keep adapter's default

            if guidance_scale is not None and orig_guidance_scale is not None:
                adapter.guidance_scale = guidance_scale    
                guidance_scale_info = f" guidance_scale={guidance_scale:.2f}"
            else:
                guidance_scale_info = ""

            # ── 2b) Adapter-specific rollout knobs (optional hook) ──────────────
            # If an adapter defines `apply_rollout_diversity(...)`, let it patch itself.
            # This hook is in-place and does NOT restore previous values.
            apply_rollout_diversity = getattr(adapter, "apply_rollout_diversity", None)
            if callable(apply_rollout_diversity):
                try:
                    adapter_info = apply_rollout_diversity(
                        rng=seed_generator,
                        stg_scale_range=rollout_stg_scale_range,
                        rescaling_scale_range=rollout_rescaling_scale_range,
                        cfg_star_rescale_prob=rollout_cfg_star_rescale_prob,
                    )
                except TypeError:
                    # Adapter hook signature mismatch; treat as absent.
                    adapter_info = ""

            print(
                f"    Rollout {rollout_index+1}/{num_rollouts} [seed={rollout_seed}{guidance_scale_info}{adapter_info}]...",
                end=" ",
                flush=True,
            )

            # ── 3) Latent noise perturbation (seeded) ──────────────────────────
            if rollout_index > 0 and float(rollout_noise_scale) > 0:
                noise = torch.randn(latents.shape, device=device, generator=rollout_seed_generator) if rollout_seed_generator is not None \
                        else torch.randn_like(latents)
                input_latents = latents + float(rollout_noise_scale) * noise
            else:
                input_latents = latents.clone()

            # ── 4) Adapter step ────────────────────────────────────────────────
            rollout_step_output = adapter.step(
                latents=input_latents,
                step_context=step_context,
                with_grad=False,
            
                solver_state=solver_state,
            ) 

            # Restore guidance_scale if we temporarily patched it.
            if orig_guidance_scale is not None:
                adapter.guidance_scale = orig_guidance_scale   # type: ignore[attr-defined]
            # NOTE: adapter-specific rollout knobs are not restored.

            to_decode = rollout_step_output.x0_latents if rollout_step_output.x0_latents is not None else rollout_step_output.next_latents
            ############### need to re-implement here, as we need to decode the x0_latents to video, and then compute the reward
            video = adapter.decode_for_reward(latents_or_x0=to_decode, x0_is_patchified=True)
            reward = reward_fn(video, prompt)
            ############### end of re-implementation

            rollout_actions_cpu.append(rollout_step_output.action.detach().to(dtype=torch.float16, device="cpu"))
            rollout_rewards.append(reward.detach().float().to(device))

            reward_value = float(reward.detach().float().item())
            if best_reward is None or reward_value > best_reward:
                best_reward = reward_value
                best_next_latents = rollout_step_output.next_latents.detach()
                best_solver_state = rollout_step_output.solver_state
                best_rollout_index = rollout_index
                best_input_latents_cpu = input_latents.detach().to(device="cpu", dtype=torch.float32)

            print(f"reward={reward.item():.4f}")

    return RolloutResult(
        rollout_actions_cpu=rollout_actions_cpu,
        rollout_rewards=rollout_rewards,
        best_next_latents=best_next_latents,                        # type: ignore[arg-type]
        best_solver_state=best_solver_state,
        best_input_latents_cpu=best_input_latents_cpu,              # type: ignore[arg-type]
        best_rollout_index=best_rollout_index,
    )


def recompute_action_and_logps(
    *,
    adapter: VideoGRPOAdapter,
    best_input_latents_cpu: torch.Tensor,
    step_context: StepContext,
    solver_state: Optional[Any],
    rollout_actions_cpu: List[torch.Tensor],
    logprob_sigma: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Recompute the BEST rollout's action with gradients, then compute the Gaussian
    surrogate log-prob measuring "how likely is each rollout action a_k to produce
    the best-reward action" under the current model.

    Strategy
    --------
    1. Rerun the model forward pass from the best rollout's exact input latents
       (with grad=True) → gives `action_best`, connected to model weights via autograd.

    2. For each rollout action a_k (stored on CPU, detached):
       Gaussian surrogate:
           logp(a_k | pi_theta) ~ -||action_best - a_k||^2 / (2 * sigma^2)
       Interpretation: high logp when the model's best-reward prediction (action_best)
       is CLOSE to rollout action a_k → a_k is likely under the current policy.

    3. The caller will compute:
           loss = -mean(A_k * logp_k)
       Gradient effect:
         - High advantage A_k > 0: gradient PUSHES action_best TOWARD a_k
           → model weights update to predict more like that high-reward rollout.
         - Low advantage A_k < 0: gradient PUSHES action_best AWAY from a_k
           → model weights update to predict less like that low-reward rollout.

    Returns
    -------
    (action_best, logps_t)
      - action_best: [*] tensor connected to model weights via autograd
      - logps_t: [K] tensor of log-prob surrogates, connected through action_best
    """
    # Move best rollout's input latents to GPU as a fresh leaf (no stale autograd graph).
    with torch.no_grad():
        fresh_latents = torch.zeros(best_input_latents_cpu.shape, device=device, dtype=torch.float32)
        fresh_latents.copy_(best_input_latents_cpu)

    # ── Differentiable forward pass ──────────────────────────────────────────────
    # Use the best rollout's EXACT input latents (with exploration noise already baked in).
    # with_grad=True → action_best carries the autograd graph: model weights → action_best.
    best_rollout_step_output = adapter.step(
        latents=fresh_latents,
        step_context=step_context,
        with_grad=True,
        solver_state=solver_state,
    )
    action_best = best_rollout_step_output.action.clone()   # [*, C, ...], connected to model weights

    # ── Gaussian surrogate log-probs ─────────────────────────────────────────────
    # logp(a_k | pi_theta) ~ -||action_best - a_k||^2 / (2 * sigma^2)
    # High logp ↔ model currently predicts something similar to rollout a_k.
    sigma2 = float(logprob_sigma) ** 2
    logps: List[torch.Tensor] = []
    for a_r_cpu in rollout_actions_cpu:
        # Stream rollout action to GPU one at a time (keeps peak VRAM low).
        a_r = a_r_cpu.to(device=device, dtype=action_best.dtype, non_blocking=True)
        mse = torch.mean((action_best - a_r) ** 2)
        logps.append(-mse / (2.0 * sigma2))

    logps_t = torch.stack(logps)   # [K], connected to action_best → model weights
    return action_best, logps_t


def run_grpo_for_prompt(
    *,
    adapter: VideoGRPOAdapter,
    prompt: str,
    reward_fn: RewardFn,
    seed: int,
    out_dir: Optional[Path] = None,
    cfg: GRPOConfig = GRPOConfig(),
    model_type: None | str = None,
) -> Dict[str, float]:
    """
    Unified GRPO loop over the last N timesteps.

    - Early steps: run normally with no grads.
    - Last `num_grpo_steps`: collect K rollouts (no grads), use reward function to 
    compute rewards based on x0_estimated latents under the current rollout and timestep t, 
    normalize the rewards across all rollouts per timestep, compute advantages scores, 
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
    # AMP GradScaler notes:
    # - GradScaler is primarily for fp16 training and expects to unscale fp16 grads into fp32.
    # - It is NOT needed for bf16, and PyTorch will error on CUDA when trying to unscale bf16 grads:
    #     NotImplementedError: "_amp_foreach_non_finite_check_and_unscale_cuda" not implemented for 'BFloat16'
    # - It also errors if the *parameters* (and thus grads) are fp16 ("Attempting to unscale FP16 gradients.")
    #
    # So we only enable GradScaler when params are fp32 (common AMP setup: fp32 weights + autocast fp16 ops).
    use_scaler = device.type == "cuda" and all(
        getattr(p, "dtype", None) == torch.float32 for p in params
    )
    scaler = torch.amp.GradScaler("cuda", enabled=bool(use_scaler))

    latents = adapter.prepare_latents(seed=int(seed)).to(device)
    solver_state: Optional[Any] = None

    # ── Gradient checkpointing (optional VRAM saver) ──────────────────────────────
    # Enable on the adapter's transformer so intermediate activations are NOT all
    # stored during the differentiable forward pass. Instead PyTorch recomputes them
    # during backward. Tradeoff: ~30% more compute, ~50% less activation VRAM.
    if cfg.gradient_checkpointing:
        model = getattr(adapter, "pipeline", None)
        model = getattr(model, "transformer", None) if model is not None else getattr(adapter, "transformer", None)
        if model is not None and hasattr(model, "enable_gradient_checkpointing"):
            try:
                # Diffusers-style models.
                model.enable_gradient_checkpointing()  # type: ignore[attr-defined]
                print("✅ Gradient checkpointing enabled on transformer.")
            except TypeError:
                # LTX's Transformer3DModel defines `_set_gradient_checkpointing(self, module, value=False)`
                # (no `enable=` kwarg), so diffusers' default `enable_gradient_checkpointing()` path crashes.
                # For LTX, the transformer forward checks `self.gradient_checkpointing` directly.
                if hasattr(model, "gradient_checkpointing"):
                    setattr(model, "gradient_checkpointing", True)
                    print("✅ Gradient checkpointing enabled on transformer (LTX fallback flag).")
                else:
                    print("⚠️  gradient_checkpointing=True but transformer checkpoint API mismatch (skipped).")
        else:
            # Fallback: try model (e.g. WAN uses pipeline.model)
            model = getattr(getattr(adapter, "pipeline", None), "model", None)
            if model is not None and hasattr(model, "enable_gradient_checkpointing"):
                try:
                    model.enable_gradient_checkpointing()
                    print("✅ Gradient checkpointing enabled on model.")
                except TypeError:
                    if hasattr(model, "gradient_checkpointing"):
                        setattr(model, "gradient_checkpointing", True)
                        print("✅ Gradient checkpointing enabled on model (fallback flag).")
                    else:
                        print("⚠️  gradient_checkpointing=True but model checkpoint API mismatch (skipped).")
            else:
                print("⚠️  gradient_checkpointing=True but no compatible module found (skipped).")

    last_loss = 0.0
    last_mean_r = 0.0
    last_std_r = 0.0
    adv: Optional[torch.Tensor] = None

    for i, t in enumerate(timesteps):
        print(f"\n{'='*50}")
        # `t` can be a torch scalar or a 1-element tensor; convert to Python float for formatting.
        try:
            t_val = float(t.item()) if isinstance(t, torch.Tensor) else float(t)
        except Exception:
            t_val = float("nan")
        print(f"Step {i+1}/{len(timesteps)} | Timestep: {t_val:.4f}")
        print(f"{'='*50}")
        
        step_context = StepContext(step_index=int(i), t=t)

        # Early steps: deterministic-ish step to reach a reasonable state.
        if i < last_start:
            print(f"  Early step (warmup, no GRPO)")

            with torch.no_grad():
                out = adapter.step(
                    latents=latents,
                    step_context=step_context,
                    with_grad=False,
               
                    solver_state=solver_state,
                )
            latents = out.next_latents.detach()
            solver_state = out.solver_state
            continue

        # ---------------------------
        # 1) Collect rollouts (no grad)
        # ---------------------------
        print(f"  Generating {cfg.num_rollouts} rollouts...")
        rollout = collect_rollouts(
            adapter=adapter,
            latents=latents,
            step_context=step_context,
            solver_state=solver_state,
            reward_fn=reward_fn,
            prompt=prompt,
            num_rollouts=int(cfg.num_rollouts),
            rollout_noise_scale=float(cfg.rollout_noise_scale),
            rollout_guidance_scales=cfg.rollout_guidance_scales,
            rollout_guidance_scale_range=cfg.rollout_guidance_scale_range,
            rollout_stg_scale_range=cfg.rollout_stg_scale_range,
            rollout_rescaling_scale_range=cfg.rollout_rescaling_scale_range,
            rollout_cfg_star_rescale_prob=cfg.rollout_cfg_star_rescale_prob,
            per_rollout_seed=bool(cfg.per_rollout_seed),
            base_seed=int(seed),
            step_index=int(i),
            device=device,
        )

        rollout_rewards = torch.stack(rollout.rollout_rewards)  # [K]
        adv = normalize_advantages(rollout_rewards, normalize=bool(cfg.normalize_advantages))

        last_mean_r = float(rollout_rewards.mean().item())
        last_std_r = float(rollout_rewards.std(unbiased=False).item())
        
        print(f"  Rewards: mean={last_mean_r:.4f}, std={last_std_r:.4f}")

        # ---------------------------
        # 2) Compute GRPO loss and update model weights
        # ---------------------------
        print(f"\n  Computing GRPO loss (best rollout: #{rollout.best_rollout_index})...")
        opt.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()

        # (a) Recompute action_best with gradients from the best-reward rollout's input latents.
        #     Compute Gaussian log-prob surrogates: logp(a_k|pi_theta) ~ -||action_best - a_k||^2 / (2σ²)
        #     logps_t[k] is HIGH when action_best is similar to rollout k's action.
        action_best, logps_t = recompute_action_and_logps(
            adapter=adapter,
            best_input_latents_cpu=rollout.best_input_latents_cpu,
            step_context=step_context,
            solver_state=solver_state,
            rollout_actions_cpu=rollout.rollout_actions_cpu,
            logprob_sigma=float(cfg.logprob_sigma),
            device=device,
        )

        # (b) GRPO loss: L = -mean_k[ A_k * logp(a_k | pi_theta) ]
        #     Advantages are detached (from rewards, no grad flows through them).
        #     Gradients flow: loss → logps_t → action_best → model weights.
        #
        #     Gradient effect on model weights:
        #       A_k > 0 (high reward): PUSH action_best TOWARD a_k → model predicts more like that rollout
        #       A_k < 0 (low reward):  PUSH action_best AWAY  from a_k → model predicts less like that rollout
        #
        #     Net: optimizer.step() moves model weights toward actions that lead to better rewards.
        loss = -(adv.detach().float() * logps_t.float()).mean()

        # (c) Backward pass + gradient clip + optimizer step
        print(f"  Backward pass...")
        if scaler.is_enabled():
            scaler.scale(loss).backward(retain_graph=False)
            scaler.unscale_(opt)
        else:
            loss.backward(retain_graph=False)

        torch.nn.utils.clip_grad_norm_(params, float(cfg.grad_clip))
        if scaler.is_enabled():
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()   # ← model weights updated here

        last_loss = float(loss.detach().cpu().item())
        
        # ---------------------------
        # 3) Advance trajectory using best rollout BEFORE deleting anything!
        # ---------------------------
        latents = rollout.best_next_latents.detach().clone()
        solver_state = rollout.best_solver_state

        # Release tensors now that we're done with this step.
        del loss, action_best, logps_t, rollout
        # Note: rewards_t and adv are small tensors and are re-assigned each step;
        # do NOT del them here to avoid UnboundLocalError if the next step fails before re-assigning.
        
        # Aggressive memory clearing
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        print(f"  ✅ Model updated | Loss: {last_loss:.4f}")
        print(f"  GPU free: {torch.cuda.mem_get_info()[0] / 1024**3:.1f} GB")

    # Save final video as MP4 (watchable format)
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        base = f"{model_type}_grpo" if model_type else "grpo"

        # (A) Save a clean post-training sample from scratch (recommended)
        if bool(getattr(cfg, "save_post_training_video_from_scratch", True)):
            print("\n🎬 Generating post-training sample from scratch (fresh noise)...")
            final_latents = generate_latents_from_scratch(
                adapter=adapter,
                seed=int(seed),
                num_inference_steps=int(cfg.num_inference_steps),
                device=device,
            )
            with torch.no_grad():
                final_video = adapter.decode_for_reward(latents_or_x0=final_latents, x0_is_patchified=True).detach().cpu()

            mp4_path = out_dir / f"{base}.mp4"
            mp4_path = save_video_tensor_as_mp4(
                video=final_video,
                mp4_path=mp4_path,
                fps=8,
                codec="libx264",
                quality=8,
                verbose=True,
            )
            print(f"\n✅ Final video saved (post-training regen): {mp4_path}")
            torch.save(final_video, out_dir / "final_video.pt")

        # (B) Optionally also save the training-trajectory decode (debug)
        if bool(getattr(cfg, "save_training_trajectory_video", False)):
            print("\n🎞️  Saving training-trajectory sample (decoded from final training latents)...")
            with torch.no_grad():
                traj_video = adapter.decode_for_reward(latents_or_x0=latents, x0_is_patchified=True).detach().cpu()

            traj_mp4_path = out_dir / f"{base}_trajectory.mp4"
            traj_mp4_path = save_video_tensor_as_mp4(
                video=traj_video,
                mp4_path=traj_mp4_path,
                fps=8,
                codec="libx264",
                quality=8,
                verbose=True,
            )
            print(f"\n✅ Trajectory video saved: {traj_mp4_path}")
            torch.save(traj_video, out_dir / "final_video_trajectory.pt")

    return {
        "last_loss": float(last_loss),
        "last_mean_reward": float(last_mean_r),
        "last_std_reward": float(last_std_r),
        "num_inference_steps": float(cfg.num_inference_steps),
        "num_grpo_steps": float(cfg.num_grpo_steps),
        "num_rollouts": float(cfg.num_rollouts),
    }

