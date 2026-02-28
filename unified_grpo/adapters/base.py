from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol

import torch
@dataclass
class StepContext:
    """Context and input for one denoising step."""

    step_index: int # the position in the timestep sequence -- an integer counting which timestep we are currently in.   
    t: torch.Tensor  # scalar timestep token (shape [] or [B]) # the actual diffusion timestep value at this step, a scalar tensor. This can be used to look up the timestep value in the scheduler.

@dataclass
class StepOutput:
    """Output of one denoising step."""

    next_latents: torch.Tensor
    action: torch.Tensor  # model output driving the solver (e.g., noise_pred / velocity)
    x0_latents: Optional[torch.Tensor] = None  # optional x0 estimate (if available)
    solver_state: Optional[Any] = None  # optional per-solver state for multistep schedulers
    
class VideoGRPOAdapter(Protocol):
    """
    Minimal interface needed by the unified GRPO core.
    Each backend (LTX, Hunyuan, etc.) implements this.
    """
    name: str

    def device(self) -> torch.device: ...

    def get_timesteps(self, *, num_inference_steps: int) -> list[torch.Tensor]: ...

    def prepare_latents(self, *, seed: int) -> torch.Tensor: ...

    def step(
        self,
        *,
        latents: torch.Tensor,
        step_context: StepContext,
        with_grad: bool,
        solver_state: Optional[Any] = None,
    ) -> StepOutput: ...
    """ the function call that takes the current latents and performs a single one denoising step, 
        return the StepOuput object. """ 
        
    # For Open-Sora
    def prepare_latents_for_reward(
        self,
        *,
        latents: torch.Tensor,
    ) -> torch.Tensor: ...
    
    
    #### change the decode for reward function here, as we need to decode the x0_latents to video, and then compute the reward
    def decode_for_reward(
        self,
        *,
        latents_or_x0: torch.Tensor,
        x0_is_patchified: bool,
    ) -> torch.Tensor:
        """
        Return decoded pixel-space video tensor suitable for reward models.

        Recommended convention across adapters:
        - Shape: [T, 3, H, W] (per-frame tensor) OR [B, T, 3, H, W]
        - Range: float in [0, 1]

        Reward backends (CLIP/DINO/Qwen) primarily operate on frames, so returning
        a per-frame tensor keeps things simple and avoids extra reshaping.
        """
        ...

    def trainable_parameters(self) -> list[torch.nn.Parameter]: ...

    def extra_log_state(self) -> Dict[str, Any]:
        return {}

