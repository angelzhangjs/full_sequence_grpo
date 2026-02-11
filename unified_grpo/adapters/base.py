from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol

import torch
@dataclass
class StepContext:
    """Context for one denoising step."""

    step_index: int
    t: torch.Tensor  # scalar timestep token (shape [] or [B])

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
        ctx: StepContext,
        with_grad: bool,
        rollout_noise_scale: float,
        rollout_index: int,
        solver_state: Optional[Any] = None,
    ) -> StepOutput: ...

    def decode_for_reward(
        self,
        *,
        latents_or_x0: torch.Tensor,
        x0_is_patchified: bool,
    ) -> torch.Tensor:
        """Return decoded video tensor suitable for reward model. Shape is model-specific."""
        ...

    def trainable_parameters(self) -> list[torch.nn.Parameter]: ...

    def extra_log_state(self) -> Dict[str, Any]:
        return {}

