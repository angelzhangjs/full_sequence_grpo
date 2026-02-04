from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

from unified_grpo.adapters.base import StepContext, StepOutput, VideoGRPOAdapter


@dataclass
class Hunyuan15Adapter(VideoGRPOAdapter):
    """
    Adapter for Tencent HunyuanVideo-1.5 pipeline.

    This adapter is "optional-dependency safe": it only relies on objects you pass in.
    You should pass a fully constructed `pipe` from HunyuanVideo-1.5 where:
      - pipe.transformer exists
      - pipe.scheduler exists and has .step(...)
      - pipe.vae exists for decoding
      - conditioning tensors are precomputed (prompt embeds etc.)
    """

    pipe: Any
    # conditioning prepared from HunyuanVideo pipeline (see their __call__)
    latent_model_input_builder: Any  # callable(latents)->(latent_model_input, t_expand)
    extra_step_kwargs: Dict[str, Any]
    prompt: str
    video_length: int
    height: int
    width: int
    train_double_blocks: Optional[List[int]] = None  # e.g., [14..27]

    name: str = "hunyuanvideo-1.5"

    def device(self) -> torch.device:
        return torch.device(getattr(self.pipe, "execution_device", "cuda"))

    def get_timesteps(self, *, num_inference_steps: int) -> list[torch.Tensor]:
        # Hunyuan pipeline uses retrieve_timesteps(...) in __call__; for adapter we assume scheduler has set_timesteps.
        if hasattr(self.pipe.scheduler, "set_timesteps"):
            self.pipe.scheduler.set_timesteps(int(num_inference_steps), device=self.device())
        ts = getattr(self.pipe.scheduler, "timesteps", None)
        if ts is None:
            raise RuntimeError("Hunyuan scheduler has no .timesteps; adapter requires a schedule.")
        return [t for t in ts]

    def prepare_latents(self, *, seed: int) -> torch.Tensor:
        g = torch.Generator(device=getattr(self.pipe, "noise_init_device", self.device()))
        g.manual_seed(int(seed))
        # Use pipeline helper if present.
        if hasattr(self.pipe, "prepare_latents"):
            latent_target_length, latent_height, latent_width = self.pipe.get_latent_size(
                int(self.video_length), int(self.height), int(self.width)
            )
            num_channels_latents = int(self.pipe.transformer.config.in_channels)
            return self.pipe.prepare_latents(
                1,
                num_channels_latents,
                int(latent_height),
                int(latent_width),
                int(latent_target_length),
                dtype=getattr(self.pipe, "target_dtype", torch.bfloat16),
                device=self.device(),
                generator=g,
            )
        raise RuntimeError("Hunyuan15Adapter requires pipe.prepare_latents.")

    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        tr = self.pipe.transformer
        # default: train all transformer params
        if not self.train_double_blocks:
            for p in tr.parameters():
                p.requires_grad_(True)
            return [p for p in tr.parameters() if p.requires_grad]

        # freeze all
        for p in tr.parameters():
            p.requires_grad_(False)

        if not hasattr(tr, "double_blocks"):
            raise AttributeError("pipe.transformer has no attribute 'double_blocks' (expected by HunyuanVideo-1.5).")

        ids = set(int(x) for x in self.train_double_blocks)
        for i, blk in enumerate(tr.double_blocks):
            req = i in ids
            for p in blk.parameters():
                p.requires_grad_(req)
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
        if (not with_grad) and int(rollout_index) > 0 and float(rollout_noise_scale) > 0:
            lat_in = latents + float(rollout_noise_scale) * torch.randn_like(latents)

        latent_model_input, t_expand = self.latent_model_input_builder(lat_in, t)

        def _forward() -> torch.Tensor:
            out = self.pipe.transformer(
                latent_model_input,
                t_expand,
                return_dict=False,
            )
            return out[0]

        if with_grad:
            noise_pred = _forward()
        else:
            with torch.no_grad():
                noise_pred = _forward()

        # scheduler transition
        next_latents = self.pipe.scheduler.step(
            noise_pred, t, lat_in, **self.extra_step_kwargs, return_dict=False
        )[0]
        return StepOutput(next_latents=next_latents, action=noise_pred, x0_latents=None, solver_state=solver_state)

    def decode_for_reward(self, *, latents_or_x0: torch.Tensor, x0_is_patchified: bool) -> torch.Tensor:
        # Hunyuan VAE decode expects scaled latents; mimic their __call__ decode path.
        lat = latents_or_x0
        if hasattr(self.pipe.vae.config, "shift_factor") and self.pipe.vae.config.shift_factor:
            lat = lat / self.pipe.vae.config.scaling_factor + self.pipe.vae.config.shift_factor
        else:
            lat = lat / self.pipe.vae.config.scaling_factor

        with torch.no_grad():
            vid = self.pipe.vae.decode(lat, return_dict=False)[0]
            vid = (vid / 2 + 0.5).clamp(0, 1)
        return vid

    def extra_log_state(self) -> Dict[str, Any]:
        return {
            "video_length": int(self.video_length),
            "height": int(self.height),
            "width": int(self.width),
            "train_double_blocks": self.train_double_blocks or [],
        }

