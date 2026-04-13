from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, Tuple

import torch
import torch.nn as nn
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


def resolve_trainable_module(adapter: VideoGRPOAdapter) -> Tuple[str, nn.Module]:
    """
    Locate the adapter's primary trainable module.

    This centralizes the backend-specific ownership differences:
    - Diffusers pipelines usually expose `pipeline.transformer`
    - WAN exposes `pipeline.model`
    - Cosmos keeps the trainable network at `model.net` and mirrors it on `adapter.transformer`
    """
    candidates = [
        ("pipeline.transformer", getattr(getattr(adapter, "pipeline", None), "transformer", None)),
        ("pipeline.model", getattr(getattr(adapter, "pipeline", None), "model", None)),
        ("model.net", getattr(getattr(adapter, "model", None), "net", None)),
        ("adapter.transformer", getattr(adapter, "transformer", None)),
    ]
    for path, module in candidates:
        if isinstance(module, nn.Module):
            return path, module
    raise RuntimeError(
        f"Could not locate a trainable module for adapter '{getattr(adapter, 'name', type(adapter).__name__)}'."
    )


def get_trainable_module(adapter: VideoGRPOAdapter) -> nn.Module:
    _, module = resolve_trainable_module(adapter)
    return module


def set_trainable_module(adapter: VideoGRPOAdapter, module: nn.Module) -> str:
    """
    Update all known references that point at the adapter's trainable module.

    The return value is the canonical path that originally identified the module.
    """
    resolved_path, _ = resolve_trainable_module(adapter)

    pipeline = getattr(adapter, "pipeline", None)
    if pipeline is not None:
        if getattr(pipeline, "transformer", None) is not None:
            pipeline.transformer = module
        if getattr(pipeline, "model", None) is not None:
            pipeline.model = module

    model = getattr(adapter, "model", None)
    if model is not None and getattr(model, "net", None) is not None:
        model.net = module

    if getattr(adapter, "transformer", None) is not None:
        setattr(adapter, "transformer", module)

    return resolved_path

