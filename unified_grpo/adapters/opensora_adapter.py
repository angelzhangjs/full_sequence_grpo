from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

from unified_grpo.adapters.base import StepContext, StepOutput, VideoGRPOAdapter


@dataclass
class OpenSoraAdapter(VideoGRPOAdapter):
    """
    Adapter for Open-Sora
    
    Prediction type: ε-prediction (like Stable Diffusion)
    Formula: x₀ = (x_t - √(1-ᾱ_t) * ε) / √(ᾱ_t)
    """
    
    pipeline: Any
    prompt_embeds: torch.Tensor
    negative_prompt_embeds: Optional[torch.Tensor] = None
    guidance_scale: float = 5.0
    
    height: int = 512
    width: int = 512
    num_frames: int = 16
    
    train_transformer_blocks: Optional[List[int]] = None
    
    name: str = "opensora"
    
    def device(self) -> torch.device:
        dev = getattr(self.pipeline, "_execution_device", None)
        return torch.device(dev) if dev is not None else torch.device("cuda")
    
    def get_timesteps(self, *, num_inference_steps: int) -> list[torch.Tensor]:
        self.pipeline.scheduler.set_timesteps(num_inference_steps, device=self.device())
        return [t for t in self.pipeline.scheduler.timesteps]
    
    def prepare_latents(self, *, seed: int) -> torch.Tensor:
        g = torch.Generator(device=self.device()).manual_seed(seed)
        
        vae_scale = 8
        temporal_scale = 4
        
        latent_h = self.height // vae_scale
        latent_w = self.width // vae_scale
        latent_f = self.num_frames // temporal_scale
        
        latents = torch.randn(
            (1, 4, latent_f, latent_h, latent_w),
            generator=g,
            device=self.device(),
            dtype=torch.float16
        )
        
        return latents * self.pipeline.scheduler.init_noise_sigma
    
    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        tr = self.pipeline.transformer
        
        if not self.train_transformer_blocks:
            for p in tr.parameters():
                p.requires_grad_(True)
            return [p for p in tr.parameters() if p.requires_grad]
        
        for p in tr.parameters():
            p.requires_grad_(False)
        
        ids = set(self.train_transformer_blocks)
        blocks = getattr(tr, "transformer_blocks", [])
        
        for i, blk in enumerate(blocks):
            if i in ids:
                for p in blk.parameters():
                    p.requires_grad_(True)
        
        return [p for p in tr.parameters() if p.requires_grad]
    
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
        
        # Forward
        def _forward():
            out = self.pipeline.transformer(
                hidden_states=lat_in.to(torch.float16),
                encoder_hidden_states=self.prompt_embeds.to(torch.float16),
                timestep=t,
                return_dict=False,
            )
            return out[0]
        
        if with_grad:
            noise_pred = _forward()
        else:
            with torch.no_grad():
                noise_pred = _forward()
        
        # Scheduler step
        step_out = self.pipeline.scheduler.step(
            noise_pred, t, latents, return_dict=True
        )
        
        next_latents = step_out.prev_sample
        
        # ε-prediction: x₀ = (x_t - √(1-ᾱ_t) * ε) / √(ᾱ_t)
        alphas = self.pipeline.scheduler.alphas_cumprod[int(t)]
        sqrt_alphas = alphas ** 0.5
        sqrt_one_minus = (1 - alphas) ** 0.5
        x0_latents = (latents - sqrt_one_minus * noise_pred) / sqrt_alphas
        
        return StepOutput(
            next_latents=next_latents,
            action=noise_pred,
            x0_latents=x0_latents,
            solver_state=solver_state
        )
    
    def decode_for_reward(self, *, latents_or_x0: torch.Tensor, x0_is_patchified: bool) -> torch.Tensor:
        vae = self.pipeline.vae
        scaling = getattr(vae.config, "scaling_factor", 0.18215)
        
        lat = latents_or_x0 / scaling
        lat = lat.to(device=vae.device, dtype=torch.float16)
        
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
