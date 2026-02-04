from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

from helper import decode_x0_to_video
from unified_grpo.adapters.base import StepContext, StepOutput, VideoGRPOAdapter


@dataclass
class LTXAdapter(VideoGRPOAdapter):
    """
    Adapter for LTX-Video pipeline in this repo.

    Notes:
    - Uses `pipeline.transformer` as the trainable policy.
    - Uses `pipeline.denoising_step(..., return_x0=True)` if available.
    """

    pipeline: Any
    prompt_embeds: torch.Tensor
    prompt_attention_mask: torch.Tensor
    indices_grid: torch.Tensor
    height: int
    width: int
    num_frames: int
    x0_is_patchified: bool = True
    trainable_blocks: Optional[List[int]] = None  # if set, freeze everything except these attn blocks

    name: str = "ltx-video"

    def device(self) -> torch.device:
        return torch.device(self.pipeline.device) if hasattr(self.pipeline, "device") else torch.device("cuda")

    def get_timesteps(self, *, num_inference_steps: int) -> list[torch.Tensor]:
        self.pipeline.scheduler.set_timesteps(int(num_inference_steps), device=self.device())
        # LTX stores timesteps as a 1D tensor; we want a list of scalar tensors.
        return [t for t in self.pipeline.scheduler.timesteps]

    def prepare_latents(self, *, seed: int) -> torch.Tensor:
        # Use pipeline's latent init for correctness.
        g = torch.Generator(device=self.device())
        g.manual_seed(int(seed))
        # The LTX pipeline has internal utilities for preparing latents; simplest is to call prepare_latents
        # if it exists, otherwise generate noise with the expected shape.
        if hasattr(self.pipeline, "prepare_latents"):
            # prepare_latents signature may vary; fall back to randn if needed.
            try:
                return self.pipeline.prepare_latents(
                    batch_size=1,
                    num_channels_latents=16,  # common for LTX; pipeline will override if needed
                    height=self.height,
                    width=self.width,
                    generator=g,
                    device=self.device(),
                    dtype=torch.bfloat16,
                )
            except Exception:
                pass
        # Fallback: try to infer from transformer config (best-effort).
        c = getattr(getattr(self.pipeline, "transformer", None), "in_channels", 16)
        latent_h = self.height // 8
        latent_w = self.width // 8
        return torch.randn((1, int(c), int(self.num_frames), int(latent_h), int(latent_w)), device=self.device(), generator=g, dtype=torch.bfloat16)

    def _select_trainable_params(self) -> list[torch.nn.Parameter]:
        model = self.pipeline.transformer
        if not self.trainable_blocks:
            for p in model.parameters():
                p.requires_grad_(True)
            return [p for p in model.parameters() if p.requires_grad]

        # Freeze everything first.
        for p in model.parameters():
            p.requires_grad_(False)

        ids = set(int(x) for x in self.trainable_blocks)
        # Best-effort: unfreeze blocks by name match "attn1_blocks.{i}" / "attn2_blocks.{i}" patterns.
        for name, p in model.named_parameters():
            for bid in ids:
                if f"attn1_blocks.{bid}." in name or f"attn2_blocks.{bid}." in name:
                    p.requires_grad_(True)
        return [p for p in model.parameters() if p.requires_grad]

    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        return self._select_trainable_params()

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
        model = self.pipeline.transformer
        t = ctx.t

        # rollout diversity: add noise to latents for r>0
        lat_in = latents
        if (not with_grad) and int(rollout_index) > 0 and float(rollout_noise_scale) > 0:
            lat_in = latents + float(rollout_noise_scale) * torch.randn_like(latents)

        # model forward
        def _forward(x: torch.Tensor) -> torch.Tensor:
            return model(
                x,
                indices_grid=self.indices_grid,
                encoder_hidden_states=self.prompt_embeds,
                encoder_attention_mask=self.prompt_attention_mask,
                timestep=t,
                return_dict=False,
            )[0]

        if with_grad:
            noise_pred = _forward(lat_in)
        else:
            with torch.no_grad():
                noise_pred = _forward(lat_in)

        # solver step + optional x0
        if hasattr(self.pipeline, "denoising_step"):
            # Some LTX versions return (next_latents, x0) when return_x0=True.
            try:
                out = self.pipeline.denoising_step(
                    latents=latents,
                    noise_pred=noise_pred,
                    current_timestep=None,
                    conditioning_mask=None,
                    t=t,
                    extra_step_kwargs={},
                    stochastic_sampling=bool(int(rollout_index) > 0),
                    return_x0=True,
                )
                next_latents, x0_est = out
                return StepOutput(next_latents=next_latents, action=noise_pred, x0_latents=x0_est, solver_state=solver_state)
            except TypeError:
                # If return_x0 isn't supported in this version, fall back to no-x0
                pass

            next_latents = self.pipeline.denoising_step(
                latents=latents,
                noise_pred=noise_pred,
                current_timestep=None,
                conditioning_mask=None,
                t=t,
                extra_step_kwargs={},
                stochastic_sampling=bool(int(rollout_index) > 0),
                return_x0=False,
            )
            return StepOutput(next_latents=next_latents, action=noise_pred, x0_latents=None, solver_state=solver_state)

        raise RuntimeError("LTXAdapter requires pipeline.denoising_step to be available.")

    def decode_for_reward(self, *, latents_or_x0: torch.Tensor, x0_is_patchified: bool) -> torch.Tensor:
        # LTX reward expects [1, T, 3, H, W] in [0,1] (as used in pipeline.py)
        return decode_x0_to_video(
            latents_or_x0,
            self.pipeline,
            num_frames=int(self.num_frames),
            height=int(self.height),
            width=int(self.width),
            is_patchified=bool(x0_is_patchified),
        )

    def extra_log_state(self) -> Dict[str, Any]:
        return {
            "height": int(self.height),
            "width": int(self.width),
            "num_frames": int(self.num_frames),
        }

