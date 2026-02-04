from __future__ import annotations

import copy
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from unified_grpo.adapters.base import StepContext, StepOutput, VideoGRPOAdapter


def _ensure_wan_importable() -> None:
    """
    Make the `wan` python package importable when running from the repo root.

    This repo sometimes contains Wan2.1 as:
    - a folder `Wan2.1/` (where the python package is `Wan2.1/wan/`), OR
    - a submodule at `wan/` (where the python package is `wan/` itself).
    """
    try:
        import wan  # noqa: F401
        return
    except Exception:
        pass

    root = Path(__file__).resolve().parents[2]

    # Case 1: submodule at <root>/wan
    if (root / "wan" / "__init__.py").exists():
        sys.path.insert(0, str(root))
        import wan  # noqa: F401
        return

    # Case 2: upstream repo folder at <root>/Wan2.1, with package at <root>/Wan2.1/wan
    if (root / "Wan2.1" / "wan" / "__init__.py").exists():
        sys.path.insert(0, str(root / "Wan2.1"))
        import wan  # noqa: F401
        return

    raise ImportError(
        "Could not import `wan`. Expected either a `wan/` package at repo root, or `Wan2.1/wan/`."
    )


@dataclass
class Wan21Adapter(VideoGRPOAdapter):
    """
    Adapter for Wan2.1 T2V (1.3B/14B share the same sampling structure).

    Notes:
    - Wan's model output is a *flow/velocity* ("flow_prediction") used by its ODE/solver.
    - Wan solvers (UniPC / multistep DPM++) keep internal multistep state (step index + past model outputs).
      GRPO rollouts call `step()` multiple times at the same timestep, so we must externalize solver state and
      clone it per rollout (otherwise rollouts would corrupt the solver history).
    """

    # A constructed WanT2V instance (from `wan.text2video.WanT2V`)
    wan: Any

    prompt: str
    negative_prompt: str = ""

    width: int = 1280
    height: int = 720
    num_frames: int = 81  # must be 4n+1 in Wan

    # Noise schedule / solver hyperparams (match Wan2.1 defaults)
    shift: float = 5.0
    guide_scale: float = 5.0
    sample_solver: str = "unipc"  # "unipc" or "dpm++" (matches WanT2V.generate)

    # Training control
    train_blocks: Optional[List[int]] = None  # unfreeze WanModel.blocks[i] for i in this list

    name: str = "wan2.1"

    # Internal cached conditioning/schedule state (filled in __post_init__/get_timesteps)
    _context: Optional[List[torch.Tensor]] = None
    _context_null: Optional[List[torch.Tensor]] = None
    _seq_len: Optional[int] = None
    _timesteps: Optional[List[torch.Tensor]] = None  # [S] scalar int64 tensors on device
    _scheduler_template: Optional[Any] = None  # schedule is set; multistep state is fresh

    def __post_init__(self) -> None:
        _ensure_wan_importable()

        # Compute latent shape + seq_len exactly like `WanT2V.generate`.
        F = int(self.num_frames)
        z_dim = int(self.wan.vae.model.z_dim)
        t_lat = (F - 1) // int(self.wan.vae_stride[0]) + 1
        h_lat = int(self.height) // int(self.wan.vae_stride[1])
        w_lat = int(self.width) // int(self.wan.vae_stride[2])

        patch_h = int(self.wan.patch_size[1])
        patch_w = int(self.wan.patch_size[2])
        sp_size = int(getattr(self.wan, "sp_size", 1))
        seq_len = math.ceil((h_lat * w_lat) / (patch_h * patch_w) * t_lat / sp_size) * sp_size
        self._seq_len = int(seq_len)

        # Pre-encode prompt + negative prompt once (same as WanT2V.generate).
        n_prompt = self.negative_prompt if self.negative_prompt != "" else str(self.wan.sample_neg_prompt)

        if not bool(getattr(self.wan, "t5_cpu", False)):
            # Put text encoder on GPU for prompt encoding.
            self.wan.text_encoder.model.to(self.device())
            ctx = self.wan.text_encoder([self.prompt], self.device())
            ctx_null = self.wan.text_encoder([n_prompt], self.device())
        else:
            # Encode on CPU then move tokens to GPU (matches Wan codepath).
            ctx = self.wan.text_encoder([self.prompt], torch.device("cpu"))
            ctx_null = self.wan.text_encoder([n_prompt], torch.device("cpu"))
            ctx = [t.to(self.device()) for t in ctx]
            ctx_null = [t.to(self.device()) for t in ctx_null]

        self._context = ctx
        self._context_null = ctx_null

        # Keep model on device for training (do NOT offload).
        self.wan.model.to(self.device())

    def device(self) -> torch.device:
        # WanT2V stores a `torch.device` in `.device`
        return torch.device(getattr(self.wan, "device"))

    # -------------------------
    # Schedule construction
    # -------------------------
    def get_timesteps(self, *, num_inference_steps: int) -> list[torch.Tensor]:
        """
        Create Wan's solver schedule and return timestep tokens.

        We store a scheduler *template* (schedule set, counters reset). The GRPO core will carry an opaque
        `solver_state` across steps; for rollouts we clone it so each rollout advances independently.
        """
        _ensure_wan_importable()

        steps = int(num_inference_steps)
        solver = str(self.sample_solver).strip().lower()

        if solver == "unipc":
            from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler  # type: ignore

            sched = FlowUniPCMultistepScheduler(
                num_train_timesteps=int(self.wan.num_train_timesteps),
                shift=1,
                use_dynamic_shifting=False,
            )
            sched.set_timesteps(steps, device=self.device(), shift=float(self.shift))
            ts = sched.timesteps

        elif solver in {"dpm++", "dpmpp", "dpm"}:
            from wan.utils.fm_solvers import (  # type: ignore
                FlowDPMSolverMultistepScheduler,
                get_sampling_sigmas,
                retrieve_timesteps,
            )

            sched = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=int(self.wan.num_train_timesteps),
                shift=1,
                use_dynamic_shifting=False,
            )
            sampling_sigmas = get_sampling_sigmas(steps, float(self.shift))
            ts, _ = retrieve_timesteps(sched, device=self.device(), sigmas=sampling_sigmas)

        else:
            raise ValueError(f"Unsupported sample_solver={self.sample_solver!r}. Expected 'unipc' or 'dpm++'.")

        self._scheduler_template = sched
        if isinstance(ts, torch.Tensor):
            t_list = [ts[i] for i in range(int(ts.numel()))]
        else:
            t_list = list(ts)
        self._timesteps = t_list
        return t_list

    # -------------------------
    # Latents init
    # -------------------------
    def prepare_latents(self, *, seed: int) -> torch.Tensor:
        # Match WanT2V.generate latent shape and dtype.
        F = int(self.num_frames)
        target_shape = (
            int(self.wan.vae.model.z_dim),
            (F - 1) // int(self.wan.vae_stride[0]) + 1,
            int(self.height) // int(self.wan.vae_stride[1]),
            int(self.width) // int(self.wan.vae_stride[2]),
        )
        g = torch.Generator(device=self.device())
        g.manual_seed(int(seed))
        return torch.randn(*target_shape, dtype=torch.float32, device=self.device(), generator=g)

    # -------------------------
    # Trainable params selection
    # -------------------------
    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        model = self.wan.model

        # Freeze everything first.
        for p in model.parameters():
            p.requires_grad_(False)

        if not self.train_blocks:
            # Default: train all WanModel parameters.
            for p in model.parameters():
                p.requires_grad_(True)
            return [p for p in model.parameters() if p.requires_grad]

        if not hasattr(model, "blocks"):
            raise AttributeError("Wan model has no attribute `blocks`; cannot select train_blocks.")

        ids = set(int(i) for i in self.train_blocks)
        for i, blk in enumerate(model.blocks):
            req = i in ids
            for p in blk.parameters():
                p.requires_grad_(req)

        # Also unfreeze final layers if they exist (common for DiT-like models).
        for name in ["final_layer", "norm", "proj", "out", "to_out"]:
            if hasattr(model, name):
                for p in getattr(model, name).parameters():
                    p.requires_grad_(True)

        return [p for p in model.parameters() if p.requires_grad]

    # -------------------------
    # One GRPO step
    # -------------------------
    def step(
        self,
        *,
        latents: torch.Tensor,
        ctx: StepContext,
        with_grad: bool,
        rollout_noise_scale: float,
        rollout_index: int,
        solver_state: Optional[Any] = None,
    ) -> StepOutput:
        if self._context is None or self._context_null is None or self._seq_len is None:
            raise RuntimeError("Wan21Adapter conditioning not initialized.")
        if self._scheduler_template is None:
            raise RuntimeError("Call get_timesteps() before stepping (scheduler not initialized).")

        lat_in = latents
        if (not with_grad) and int(rollout_index) > 0 and float(rollout_noise_scale) > 0:
            lat_in = latents + float(rollout_noise_scale) * torch.randn_like(latents)

        # WanModel expects a list of latents (batch of variable-length sequences).
        latent_model_input = [lat_in]
        arg_c = {"context": self._context, "seq_len": int(self._seq_len)}
        arg_null = {"context": self._context_null, "seq_len": int(self._seq_len)}

        # Wan uses `t` as a 1-element tensor.
        t = ctx.t.to(self.device())
        timestep = torch.stack([t]).to(self.device())

        def _forward() -> Tuple[torch.Tensor, torch.Tensor]:
            noise_pred_cond = self.wan.model(latent_model_input, t=timestep, **arg_c)[0]
            noise_pred_uncond = self.wan.model(latent_model_input, t=timestep, **arg_null)[0]
            return noise_pred_cond, noise_pred_uncond

        if with_grad:
            noise_pred_cond, noise_pred_uncond = _forward()
        else:
            with torch.no_grad():
                noise_pred_cond, noise_pred_uncond = _forward()

        # CFG combine (exactly as WanT2V.generate)
        flow = noise_pred_uncond + float(self.guide_scale) * (noise_pred_cond - noise_pred_uncond)

        # Externalized scheduler state:
        # - core carries `solver_state` across steps
        # - each rollout clones `solver_state` so rollouts don't corrupt each other
        base_sched = solver_state if solver_state is not None else self._scheduler_template
        sched = copy.deepcopy(base_sched)

        # Optional x0 prediction for reward (matches fm_solvers flow_prediction conversion)
        x0_pred: Optional[torch.Tensor] = None
        if hasattr(sched, "sigmas"):
            try:
                idx = int(getattr(sched, "step_index")) if getattr(sched, "step_index", None) is not None else int(ctx.step_index)
                sigma_s = sched.sigmas[idx].to(lat_in.device, dtype=torch.float32)
                x0_pred = (lat_in.to(torch.float32) - sigma_s * flow.to(torch.float32)).to(lat_in.dtype)
            except Exception:
                x0_pred = None

        out = sched.step(
            flow.unsqueeze(0),
            t,
            lat_in.unsqueeze(0),
            return_dict=False,
            generator=None,
        )[0]
        next_latents = out.squeeze(0)
        return StepOutput(next_latents=next_latents, action=flow, x0_latents=x0_pred, solver_state=sched)

    # -------------------------
    # Decode for reward
    # -------------------------
    def decode_for_reward(self, *, latents_or_x0: torch.Tensor, x0_is_patchified: bool) -> torch.Tensor:
        # Wan's VAE expects a list of latents [C, T_lat, H_lat, W_lat] and returns a list of videos [3, T, H, W] in [-1,1].
        vid = self.wan.vae.decode([latents_or_x0])[0]  # [3, T, H, W], [-1,1]
        vid = (vid / 2.0 + 0.5).clamp(0, 1)  # [0,1]
        # Return [1, T, 3, H, W] for consistency with other adapters.
        vid = vid.permute(1, 0, 2, 3).unsqueeze(0).contiguous()
        return vid

    def extra_log_state(self) -> Dict[str, Any]:
        return {
            "width": int(self.width),
            "height": int(self.height),
            "num_frames": int(self.num_frames),
            "shift": float(self.shift),
            "guide_scale": float(self.guide_scale),
            "train_blocks": self.train_blocks or [],
        }

