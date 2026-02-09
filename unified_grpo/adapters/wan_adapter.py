from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

from unified_grpo.adapters.base import StepContext, StepOutput, VideoGRPOAdapter


@dataclass
class WanAdapter(VideoGRPOAdapter):
    """
    Adapter for Wan2.1 (Kuaishou)
    
    Prediction type: Likely ε-prediction or flow matching
    Note: Wan uses custom implementation, may need adjustments
    """
    
    pipeline: Any
    prompt_embeds: torch.Tensor
    negative_prompt_embeds: Optional[torch.Tensor] = None
    guidance_scale: float = 5.0
    
    height: int = 640
    width: int = 352
    num_frames: int = 33
    
    train_transformer_blocks: Optional[List[int]] = None
    
    name: str = "wan"
    
    def device(self) -> torch.device:
        dev = getattr(self.pipeline, "_execution_device", None)
        return torch.device(dev) if dev is not None else torch.device("cuda")
    
    def get_timesteps(self, *, num_inference_steps: int) -> list[torch.Tensor]:
        # Wan might use different API - try both
        if hasattr(self.pipeline.scheduler, 'set_timesteps'):
            self.pipeline.scheduler.set_timesteps(num_inference_steps, device=self.device())
            return [t for t in self.pipeline.scheduler.timesteps]
        elif hasattr(self.pipeline.scheduler, 'get_timesteps'):
            return [t for t in self.pipeline.scheduler.get_timesteps(num_inference_steps)]
        else:
            raise RuntimeError("Wan scheduler API unknown")
    
    def prepare_latents(self, *, seed: int) -> torch.Tensor:
        g = torch.Generator(device=self.device()).manual_seed(seed)
        
        # Wan latent dimensions (16 channels)
        vae_scale = 8
        temporal_scale = 8
        
        latent_h = self.height // vae_scale
        latent_w = self.width // vae_scale
        latent_f = self.num_frames // temporal_scale
        
        latents = torch.randn(
            (1, 16, latent_f, latent_h, latent_w),
            generator=g,
            device=self.device(),
            dtype=torch.bfloat16
        )
        
        init_sigma = getattr(self.pipeline.scheduler, "init_noise_sigma", 1.0)
        return latents * init_sigma
    
    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        # Wan uses .blocks attribute (from DiT)
        model = getattr(self.pipeline, "transformer", None)
        if model is None:
            model = getattr(self.pipeline, "model", None)
        
        if model is None:
            raise RuntimeError("Wan adapter requires transformer or model")
        
        if not self.train_transformer_blocks:
            for p in model.parameters():
                p.requires_grad_(True)
            return [p for p in model.parameters() if p.requires_grad]
        
        for p in model.parameters():
            p.requires_grad_(False)
        
        ids = set(self.train_transformer_blocks)
        blocks = getattr(model, "blocks", [])  # Wan uses .blocks
        
        for i, blk in enumerate(blocks):
            if i in ids:
                for p in blk.parameters():
                    p.requires_grad_(True)
        
        return [p for p in model.parameters() if p.requires_grad]
    
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
        t = ctx.t
        
        lat_in = latents
        if (not with_grad) and rollout_index > 0 and rollout_noise_scale > 0:
            lat_in = latents + rollout_noise_scale * torch.randn_like(latents)
        
        # Forward (Wan might have different API)
        model = getattr(self.pipeline, "transformer", getattr(self.pipeline, "model", None))
        
        def _forward():
            # Try standard diffusers API first
            try:
                out = model(
                    hidden_states=lat_in.to(torch.bfloat16),
                    encoder_hidden_states=self.prompt_embeds.to(torch.bfloat16),
                    timestep=t,
                    return_dict=False,
                )
                return out[0]
            except:
                # Fallback: Wan custom API
                out = model(
                    latents=lat_in.to(torch.bfloat16),
                    context=self.prompt_embeds.to(torch.bfloat16),
                    timestep=t,
                )
                return out
        
        if with_grad:
            model_output = _forward()
        else:
            with torch.no_grad():
                model_output = _forward()
        
        # Scheduler step
        step_result = self.pipeline.scheduler.step(model_output, t, latents)
        
        # Handle both dict and tuple returns
        if isinstance(step_result, tuple):
            next_latents = step_result[0]
            x0_latents = step_result[1] if len(step_result) > 1 else None
        else:
            next_latents = step_result.prev_sample
            x0_latents = getattr(step_result, 'pred_original_sample', None)
        
        # If x0 not provided, compute manually (ε-prediction assumed)
        if x0_latents is None:
            alphas = self.pipeline.scheduler.alphas_cumprod[int(t)]
            sqrt_alphas = alphas ** 0.5
            sqrt_one_minus = (1 - alphas) ** 0.5
            x0_latents = (latents - sqrt_one_minus * model_output) / sqrt_alphas
        
        return StepOutput(
            next_latents=next_latents,
            action=model_output,
            x0_latents=x0_latents,
            solver_state=solver_state
        )
    
    def decode_for_reward(self, *, latents_or_x0: torch.Tensor, x0_is_patchified: bool) -> torch.Tensor:
        vae = self.pipeline.vae
        scaling = getattr(getattr(vae, "config", None), "scaling_factor", 1.0)
        
        lat = latents_or_x0 / scaling
        lat = lat.to(device=vae.device, dtype=torch.bfloat16)
        
        with torch.no_grad():
            vid = vae.decode(lat).sample
            vid = (vid / 2 + 0.5).clamp(0, 1)
        
        return vid
    
    def extra_log_state(self) -> Dict[str, Any]:
        return {
            "height": self.height,
            "width": self.width,
            "num_frames": self.num_frames,
        }
