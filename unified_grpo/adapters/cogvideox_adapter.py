from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

from unified_grpo.adapters.base import StepContext, StepOutput, VideoGRPOAdapter

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
        tr = getattr(self.pipeline, "transformer", None)
        if tr is None:
            raise RuntimeError("CogVideoXAdapter requires pipeline.transformer")

        # Check if LoRA is applied (PEFT model has special attributes)
        is_lora = hasattr(tr, 'peft_config') or any('lora_' in n for n, _ in tr.named_parameters())
        
        if is_lora:
            # Return only LoRA parameters
            print("  [CogVideoX Adapter] Using LoRA parameters for training")
            lora_params = [p for n, p in tr.named_parameters() if 'lora_' in n.lower() and p.requires_grad]
            if len(lora_params) == 0:
                # Ensure LoRA params are trainable
                for n, p in tr.named_parameters():
                    if 'lora_' in n.lower():
                        p.requires_grad_(True)
                        lora_params.append(p)
            return lora_params

        if not self.train_transformer_blocks:
            for p in tr.parameters():
                p.requires_grad_(True)
            return [p for p in tr.parameters() if p.requires_grad]

        # Freeze everything first.
        for p in tr.parameters():
            p.requires_grad_(False)

        ids = set(int(x) for x in self.train_transformer_blocks)

        # Best case: transformer has transformer_blocks list.
        blocks = getattr(tr, "transformer_blocks", None)
        if blocks is not None:
            for i, blk in enumerate(blocks):
                req = i in ids
                for p in blk.parameters():
                    p.requires_grad_(req)
            return [p for p in tr.parameters() if p.requires_grad]

        # Fallback: unfreeze by name match.
        for name, p in tr.named_parameters():
            for bid in ids:
                if f"transformer_blocks.{bid}." in name:
                    p.requires_grad_(True)
        return [p for p in tr.parameters() if p.requires_grad]

    def _prepare_rotary_emb(self, *, latents: torch.Tensor, device: torch.device) -> Optional[torch.Tensor]:
        # Only needed for some CogVideoX variants.
        tr = getattr(self.pipeline, "transformer", None)
        if tr is None:
            return None
        cfg = getattr(tr, "config", None)
        if cfg is None or not getattr(cfg, "use_rotary_positional_embeddings", False):
            return None

        fn = getattr(self.pipeline, "_prepare_rotary_positional_embeddings", None)
        if fn is None:
            # Not available in some diffusers versions; return None and rely on pipeline defaults.
            return None

        vae_scale_spatial = int(getattr(self.pipeline, "vae_scale_factor_spatial", getattr(self.pipeline, "vae_scale_factor", 8)))
        # Latents are [B, F, C, H, W]
        return fn(
            height=int(latents.size(3) * vae_scale_spatial),
            width=int(latents.size(4) * vae_scale_spatial),
            num_frames=int(latents.size(1)),
            device=device,
        )

    def step(
        self,
        *,
        latents: torch.Tensor,
        ctx: StepContext,
        with_grad: bool,
        rollout_noise_scale: float,
        rollout_index: int,
        solver_state=None,
    ) -> StepOutput:
        
        tr = getattr(self.pipeline, "transformer", None)
        if tr is None:
            raise RuntimeError("CogVideoXAdapter requires pipeline.transformer")

        t = ctx.t

        # CRITICAL: Detach input latents to prevent graph reuse
        lat_in = latents.detach() if not with_grad else latents
        if (not with_grad) and int(rollout_index) > 0 and float(rollout_noise_scale) > 0:
            lat_in = lat_in + float(rollout_noise_scale) * torch.randn_like(lat_in)

        do_cfg = self._do_classifier_free_guidance()
        # Detach prompt embeds (they shouldn't need gradients)
        encoder_hidden_states = self.prompt_embeds.detach()
        if do_cfg:
            encoder_hidden_states = torch.cat([self.negative_prompt_embeds.detach(), self.prompt_embeds.detach()], dim=0)

        latent_model_input = torch.cat([lat_in] * 2, dim=0) if do_cfg else lat_in
        # Clone to break any scheduler caching
        latent_model_input = self.pipeline.scheduler.scale_model_input(latent_model_input.clone(), t)

        # broadcast timestep to batch dim and ensure correct dtype
        timestep = t.expand(latent_model_input.shape[0])
        
        # Ensure timestep is proper dtype (some schedulers use float32)
        if timestep.dtype != latent_model_input.dtype:
            timestep = timestep.to(dtype=latent_model_input.dtype)

        image_rotary_emb = self._prepare_rotary_emb(latents=lat_in, device=latent_model_input.device)
        attention_kwargs = self.attention_kwargs

        def _forward() -> torch.Tensor:
            # Ensure all inputs have matching dtypes
            out = tr(
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

        # scheduler transition
        step_out = self.pipeline.scheduler.step(noise_pred, t, lat_in, return_dict=True, **extra_step_kwargs)

        next_latents = getattr(step_out, "prev_sample", None)
        if next_latents is None:
            # Some schedulers return tuple.
            next_latents = step_out[0]

        # Optional x0 estimate
        x0_latents = getattr(step_out, "pred_original_sample", None)

        return StepOutput(next_latents=next_latents, action=noise_pred, x0_latents=x0_latents, solver_state=solver_state)

    def decode_for_reward(self, *, latents_or_x0: torch.Tensor, x0_is_patchified: bool) -> torch.Tensor:
        # CogVideoX latents are NOT patchified; ignore x0_is_patchified.
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
        return vid

    def extra_log_state(self) -> Dict[str, Any]:
        return {
            "height": int(self.height),
            "width": int(self.width),
            "num_frames": int(self.num_frames),
            "guidance_scale": float(self.guidance_scale),
            "train_transformer_blocks": self.train_transformer_blocks or [],
        }

