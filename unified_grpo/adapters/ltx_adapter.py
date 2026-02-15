from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

from unified_grpo.adapters.base import StepContext, StepOutput, VideoGRPOAdapter


@dataclass
class LTXAdapter(VideoGRPOAdapter):
    """
    Adapter for LTX-Video (Lightricks)
    
    Prediction type: v-prediction (Rectified Flow)
    Formula: x₀ = x_t - t * v
    """
    
    pipeline: Any
    prompt_embeds: torch.Tensor
    negative_prompt_embeds: Optional[torch.Tensor] = None
    guidance_scale: float = 4.5
    
    height: int = 512
    width: int = 768
    num_frames: int = 25
    
    train_transformer_blocks: Optional[List[int]] = None
    
    name: str = "ltx"
    
    def device(self) -> torch.device:
        dev = getattr(self.pipeline, "_execution_device", None)
        if dev is None:
            dev = getattr(self.pipeline, "device", None)
        return torch.device(dev) if dev is not None else torch.device("cuda")
    
    def get_timesteps(self, *, num_inference_steps: int) -> list[torch.Tensor]:
        self.pipeline.scheduler.set_timesteps(num_inference_steps, device=self.device())
        return [t for t in self.pipeline.scheduler.timesteps]
    
    def prepare_latents(self, *, seed: int) -> torch.Tensor:
        g = torch.Generator(device=self.device()).manual_seed(seed)
        
        # LTX latent dimensions
        vae_scale = 8
        video_scale = 8
        
        latent_h = self.height // vae_scale
        latent_w = self.width // vae_scale
        latent_f = self.num_frames // video_scale + 1  # +1 for Causal VAE
        
        latents = torch.randn(
            (1, 4, latent_f, latent_h, latent_w),
            generator=g,
            device=self.device(),
            dtype=torch.bfloat16
        )
        
        return latents * self.pipeline.scheduler.init_noise_sigma
    
    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        tr = self.pipeline.transformer
        
        if not self.train_transformer_blocks:
            # Train all
            for p in tr.parameters():
                p.requires_grad_(True)
            return [p for p in tr.parameters() if p.requires_grad]
        
        # Freeze all first
        for p in tr.parameters():
            p.requires_grad_(False)
        
        # Unfreeze specified blocks
        ids = set(self.train_transformer_blocks)
        blocks = tr.transformer_blocks
        
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
        
        # Add noise for exploration
        lat_in = latents
        if (not with_grad) and rollout_index > 0 and rollout_noise_scale > 0:
            lat_in = latents + rollout_noise_scale * torch.randn_like(latents)
        
        # CFG setup
        do_cfg = self.guidance_scale > 1.0 and self.negative_prompt_embeds is not None
        encoder_hidden_states = self.prompt_embeds
        
        if do_cfg:
            encoder_hidden_states = torch.cat([
                self.negative_prompt_embeds,
                self.prompt_embeds
            ], dim=0)
            lat_in = torch.cat([lat_in, lat_in], dim=0)
        
        # Forward pass
        def _forward():
            out = self.pipeline.transformer(
                hidden_states=lat_in.to(torch.bfloat16),
                encoder_hidden_states=encoder_hidden_states.to(torch.bfloat16),
                timestep=t.to(torch.bfloat16),
                return_dict=False,
            )
            return out[0]
        
        if with_grad:
            velocity_pred = _forward()
        else:
            with torch.no_grad():
                velocity_pred = _forward()
        
        # CFG
        if do_cfg:
            v_uncond, v_text = velocity_pred.chunk(2)
            velocity_pred = v_uncond + self.guidance_scale * (v_text - v_uncond)
        
        # Scheduler step
        step_out = self.pipeline.scheduler.step(
            velocity_pred, t, latents, return_dict=True
        )
        
        next_latents = step_out.prev_sample
        
        # Compute x₀ (Rectified Flow formula)
        x0_latents = latents - t * velocity_pred
        
        return StepOutput(
            next_latents=next_latents,
            action=velocity_pred,
            x0_latents=x0_latents,
            solver_state=solver_state
        )
    
    def decode_for_reward(self, *, latents_or_x0: torch.Tensor, x0_is_patchified: bool) -> torch.Tensor:
        vae = self.pipeline.vae
        scaling = vae.config.scaling_factor
        
        lat = latents_or_x0 / scaling
        lat = lat.to(device=vae.device, dtype=torch.bfloat16)
        
        target_shape = (1, 3, self.num_frames, self.height, self.width)
        
        with torch.no_grad():
            vid = vae.decode(lat, target_shape=target_shape, timestep=torch.tensor([0.0])).sample
            vid = (vid / 2 + 0.5).clamp(0, 1)
        
        return vid
    
    def extra_log_state(self) -> Dict[str, Any]:
        return {
            "height": self.height,
            "width": self.width,
            "num_frames": self.num_frames,
            "guidance_scale": self.guidance_scale,
        }
