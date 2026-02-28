from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from torch.nn.modules import transformer

from unified_grpo.adapters.base import StepContext, StepOutput, VideoGRPOAdapter
from unified_grpo.lora_utils import get_trainable_lora_parameters, has_lora
from unified_grpo.utils import prepare_rotary_emb
@dataclass
class CogVideoXAdapter(VideoGRPOAdapter):
    """
    Adapter for Diffusers CogVideoX pipelines.

    Design goals:
    - Works with an already-constructed diffusers CogVideoX pipeline you pass in.
    - Keeps imports "optional-dependency safe" by typing pipeline as Any.
    - Exposes a unified per-step API for GRPO via scheduler.step(...).

    Expected pipeline attributes (diffusers):
    - pipeline.transformer
    - pipeline.scheduler with set_timesteps(...) and step(...)
    - pipeline.vae with encode/decode and config.scaling_factor

    Notes on tensor shapes:
    - CogVideoX latents are typically shaped [B, F_latent, C, H_latent, W_latent]
      (see `CogVideo/inference/ddim_inversion.py` in this repo).
    - `decode_for_reward` returns pixel video shaped [B, F, C, H, W] in [0, 1].
    """
    pipeline: Any
    # Prompt conditioning (precomputed, to avoid calling pipeline.encode_prompt inside the adapter)
    prompt_embeds: torch.Tensor
    negative_prompt_embeds: Optional[torch.Tensor] = None
    guidance_scale: float = 6.0

    # Generation geometry (pixel-space)
    height: int = 480
    width: int = 720
    num_frames: int = 49

    # Optional knobs
    attention_kwargs: Optional[Dict[str, Any]] = None
    extra_step_kwargs: Optional[Dict[str, Any]] = None

    # Trainable subset selection (indices into transformer.transformer_blocks if present)
    train_transformer_blocks: Optional[List[int]] = None

    # If set, overrides inferred latent temporal length. Otherwise we infer from num_frames and common VAE temporal factor.
    latent_num_frames: Optional[int] = None

    name: str = "cogvideox"

    def device(self) -> torch.device:
        # diffusers pipelines typically expose _execution_device; fall back to cuda.
        dev = getattr(self.pipeline, "_execution_device", None)
        if dev is None:
            dev = getattr(self.pipeline, "device", None)
        return torch.device(dev) if dev is not None else torch.device("cuda")

    def _do_classifier_free_guidance(self) -> bool:
        return bool(self.guidance_scale > 1.0 and self.negative_prompt_embeds is not None)

    def get_timesteps(self, *, num_inference_steps: int) -> list[torch.Tensor]:
        if not hasattr(self.pipeline, "scheduler") or not hasattr(self.pipeline.scheduler, "set_timesteps"):
            raise RuntimeError("CogVideoXAdapter requires pipeline.scheduler.set_timesteps(...)")
        self.pipeline.scheduler.set_timesteps(int(num_inference_steps), device=self.device())
        ts = getattr(self.pipeline.scheduler, "timesteps", None)
        if ts is None:
            raise RuntimeError("pipeline.scheduler has no .timesteps after set_timesteps")
        return [t for t in ts]

    def prepare_latents(self, *, seed: int) -> torch.Tensor:
        """ 
        prepare the initial latents to be used for the first step of the denoising process, which is the input of .step() function. 
        Args:
            seed: the seed for the random number generator
        Returns:
            latents: the initial latents
        """
        g = torch.Generator(device=self.device())
        g.manual_seed(int(seed))

        # Infer latent geometry
        vae_scale_spatial = int(getattr(self.pipeline, "vae_scale_factor_spatial", getattr(self.pipeline, "vae_scale_factor", 8)))
        # Many video VAEs use a temporal factor of 4 (hence 4k+1 frame constraints); fall back to 4.
        vae_scale_temporal = int(getattr(self.pipeline, "vae_scale_factor_temporal", 4))

        latent_h = int(self.height) // max(1, vae_scale_spatial)
        latent_w = int(self.width) // max(1, vae_scale_spatial)
        latent_f = int(self.latent_num_frames) if self.latent_num_frames is not None else ((int(self.num_frames) - 1) // max(1, vae_scale_temporal) + 1)

        # Infer channels from transformer config if possible.
        tr = getattr(self.pipeline, "transformer", None)
        c = 16
        if tr is not None:
            cfg = getattr(tr, "config", None)
            if cfg is not None and hasattr(cfg, "in_channels"):
                c = int(cfg.in_channels)
            elif hasattr(tr, "in_channels"):
                c = int(tr.in_channels)

        latents = torch.randn((1, latent_f, c, latent_h, latent_w), device=self.device(), generator=g, dtype=torch.bfloat16)

        # Match scheduler init scaling if present.
        init_sigma = getattr(self.pipeline.scheduler, "init_noise_sigma", None)
        if init_sigma is not None:
            latents = latents * float(init_sigma)
        return latents

    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        transformer = getattr(self.pipeline, "transformer", None)
        if transformer is None:
            raise RuntimeError("CogVideoXAdapter requires pipeline.transformer")

        # ==========================================================================
        # Training mode A: LoRA/PEFT fine-tuning
        # - If LoRA is attached, we ONLY optimize LoRA params.
        # - We intentionally do NOT unfreeze the base transformer weights, because that
        #   defeats the purpose of parameter-efficient fine-tuning and explodes VRAM.
        # ==========================================================================
        if has_lora(transformer):
            return get_trainable_lora_parameters(transformer, verbose_prefix="  [CogVideoX Adapter]")

        # ==========================================================================
        # Training mode B: Partial fine-tune (only selected blocks)
        # - Freeze everything first, then unfreeze a chosen set of blocks.
        # - This is useful when you are NOT using LoRA but still want to limit VRAM.
        # ==========================================================================
        for p in transformer.parameters():
            p.requires_grad_(False)

        ids = set(int(x) for x in self.train_transformer_blocks)

        # ========================================================
        # C1 vs C2 is entirely determined by how that specific model's transformer is implemented/exposed by the pipeline. 
        # ========================================================
        # Block selection path C1 (preferred):
        # - Some transformer implementations expose a `transformer_blocks` list.
        # - We can reliably unfreeze by iterating blocks by index.
        blocks = getattr(transformer, "transformer_blocks", None)
        if blocks is not None:
            for i, blk in enumerate(blocks):
                req = i in ids
                for p in blk.parameters():
                    p.requires_grad_(req)
            return [p for p in transformer.parameters() if p.requires_grad]

        # Block selection path C2 (fallback):
        # - If there's no `transformer_blocks` list, fall back to name matching.
        # - This is more fragile (depends on naming conventions), but works for many
        #   diffusers-like models.
        for name, p in transformer.named_parameters():
           for bid in ids:
                if f"transformer_blocks.{bid}." in name:
                    p.requires_grad_(True)
        return [p for p in transformer.parameters() if p.requires_grad]

    def step(
        self,
        *,
        latents: torch.Tensor,
        step_context: StepContext,
        with_grad: bool,
        solver_state=None,
    ) -> StepOutput:
        """
        Perform one denoising step using the CogVideoX pipeline.

        Notes:
        - This function has two phases:
          1) Transformer forward pass (text-conditioned) -> `noise_pred`
          2) Scheduler update -> `prev_sample` (next latents) and optional `pred_original_sample` (x0 estimate)
          
    Perform one denoising step using the CogVideoX pipeline.
       Args:
        latents: the latents to denoise
        step_context: the step context
        with_grad: whether to use gradient
        solver_state: the solver state
       Returns:
        StepOutput: the step output
           next_latents: the next latents
           action: the action
           x0_latents: the x0 latents
           solver_state: the solver states
           
        there are two steps in the denoising process:
        1. the transformer forward pass
        2. the scheduler step, which returns the previous latents and the predicted original latents
        """
        # get the transformer
        transformer = getattr(self.pipeline, "transformer", None)
        if transformer is None:
            raise RuntimeError("CogVideoXAdapter requires pipeline.transformer")

        # get the timestep
        t = step_context.t

        # detach the input latents to prevent graph reuse
        latents_input = latents.detach() if not with_grad else latents

        do_cfg = self._do_classifier_free_guidance()
        # detach the prompt embeds (they shouldn't need gradients)
        encoder_hidden_states = self.prompt_embeds.detach()
        if do_cfg:
            encoder_hidden_states = torch.cat([self.negative_prompt_embeds.detach(), self.prompt_embeds.detach()], dim=0)

        latent_model_input = torch.cat([latents_input] * 2, dim=0) if do_cfg else latents_input
        # clone to break any scheduler caching
        latent_model_input = self.pipeline.scheduler.scale_model_input(latent_model_input.clone(), t)

        # broadcast timestep to batch dim and ensure correct dtype, for example, if timestep is a scalar tensor, we need to expand it to a tensor of shape (batch_size, 1)
        timestep = t.expand(latent_model_input.shape[0])
        
        # ensure timestep is proper dtype (some schedulers use float32)
        if timestep.dtype != latent_model_input.dtype:
            timestep = timestep.to(dtype=latent_model_input.dtype)

        # prepare the rotary embeddings (CogVideoX variants that use RoPE)
        image_rotary_emb = prepare_rotary_emb(self, latents=latents_input, device=latent_model_input.device)
        attention_kwargs = self.attention_kwargs

        def _forward() -> torch.Tensor:
            # Ensure all inputs have matching dtypes
            out = transformer(
                hidden_states=latent_model_input.to(torch.bfloat16),
                encoder_hidden_states=encoder_hidden_states.to(torch.bfloat16),
                timestep=timestep.to(torch.bfloat16),
                image_rotary_emb=image_rotary_emb,
                attention_kwargs=attention_kwargs,
                return_dict=False,
            )
            return out[0]

        if with_grad:
            noise_pred = _forward()
        else:
            with torch.no_grad():
                noise_pred = _forward()

        noise_pred = noise_pred.float()

        if do_cfg:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + float(self.guidance_scale) * (noise_pred_text - noise_pred_uncond)

        extra_step_kwargs = self.extra_step_kwargs or {}

        # scheduler transition, simply return value of scheduler.step() for one denoising step. 
        # return DDIMSchedulerOutput object, which contains the following attributes: 
        # - prev_sample: the previous latents
        # - pred_original_sample: the predicted original latents
        step_out = self.pipeline.scheduler.step(noise_pred, t, latents_input, return_dict=True, **extra_step_kwargs)

        next_latents = getattr(step_out, "prev_sample", None)
        if next_latents is None:
            # Some schedulers return tuple.
            next_latents = step_out[0]

        # Optional x0 estimate
        x0_latents = getattr(step_out, "pred_original_sample", None)

        return StepOutput(next_latents=next_latents, action=noise_pred, x0_latents=x0_latents, solver_state=solver_state)

    def decode_for_reward(self, *, latents_or_x0: torch.Tensor, x0_is_patchified: bool) -> torch.Tensor:
        # CogVideoX latents are NOT patchified; ignore x0_is_patchified.
        # Output convention: return per-frame pixel tensor [T, 3, H, W] in [0, 1].
        lat = latents_or_x0

        vae = getattr(self.pipeline, "vae", None)
        if vae is None:
            raise RuntimeError("CogVideoXAdapter requires pipeline.vae for decoding")

        scaling = float(getattr(getattr(vae, "config", None), "scaling_factor", 1.0))
        lat = lat / max(scaling, 1e-8)

        # VAE expects [B, C, F, H, W]
        lat = lat.transpose(1, 2).to(device=vae.device, dtype=getattr(vae, "dtype", torch.float16))

        with torch.no_grad():
            vid = vae.decode(lat, return_dict=False)[0]  # [B, C, F, H, W] in [-1, 1] (usually)
            vid = (vid / 2 + 0.5).clamp(0, 1)
            vid = vid.permute(0, 2, 1, 3, 4).contiguous()  # [B, F, C, H, W]
        # Drop batch for reward backends (they operate on frames)
        return vid[0]

    def extra_log_state(self) -> Dict[str, Any]:
        return {
            "height": int(self.height),
            "width": int(self.width),
            "num_frames": int(self.num_frames),
            "guidance_scale": float(self.guidance_scale),
            "train_transformer_blocks": self.train_transformer_blocks or [],
        }
