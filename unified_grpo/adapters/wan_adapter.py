from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, List, Optional

import torch
import torch.cuda.amp as amp

from unified_grpo.adapters.base import StepContext, StepOutput, VideoGRPOAdapter


@dataclass
class WanAdapter(VideoGRPOAdapter):
    """
    Adapter for Wan2.1 (Kuaishou)
    
    This adapter is implemented to follow Wan2.1's native sampling loop in
    `Wan2.1/wan/text2video.py` (WanT2V.generate):
    - latents are video-latents shaped [B, C, F_lat, H_lat, W_lat] (we use B=1)
    - model forward signature is `model([latent], t=timestep, context=..., seq_len=...)`
    - scheduler is FlowUniPC / FlowDPM++ from `wan/utils/fm_solvers*.py`
    - VAE decode uses `WanVAE.decode([latent])` which returns pixels in [-1, 1]
    """

    # `pipeline` is expected to be a WanT2V-like object (see Wan2.1/wan/text2video.py)
    # with at least: .device, .model (WanModel), .vae (WanVAE), .vae_stride, .patch_size,
    # .num_train_timesteps, .param_dtype
    pipeline: Any

    # Wan text encoder returns context as a *list* of tensors (see WanT2V.generate).
    prompt_embeds: Any
    negative_prompt_embeds: Optional[Any] = None
    guidance_scale: float = 5.0
    
    # Generation geometry (pixel-space)
    height: int = 720
    width: int = 1280
    num_frames: int = 81

    # Sampling knobs (match Wan2.1 defaults)
    shift: float = 5.0
    sample_solver: str = "unipc"  # "unipc" or "dpm++"

    # Training subset selection
    train_transformer_blocks: Optional[List[int]] = None
    # If set (e.g. 0.2), unfreeze the last N% of transformer blocks.
    # Ignored when `train_transformer_blocks` is provided explicitly.
    unfreeze_percentage: Optional[float] = None

    # Internal scheduler instance created by get_timesteps()
    _scheduler: Optional[Any] = None
    
    name: str = "wan"
    
    def device(self) -> torch.device:
        dev = getattr(self.pipeline, "device", None)
        if dev is None:
            # best-effort fallback
            dev = getattr(getattr(self.pipeline, "model", None), "device", None)
        return torch.device(dev) if dev is not None else torch.device("cuda")
    
    def get_timesteps(self, *, num_inference_steps: int) -> list[torch.Tensor]:
        # Import from Wan2.1 package at runtime (optional dependency safe).
        try:
            from wan.utils.fm_solvers import (
                FlowDPMSolverMultistepScheduler,
                get_sampling_sigmas,
                retrieve_timesteps,
            )
            from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
        except Exception as e:
            raise RuntimeError(
                "WanAdapter requires Wan2.1 on PYTHONPATH. "
                "Expected imports: `wan.utils.fm_solvers*`."
            ) from e

        num_train_timesteps = int(getattr(self.pipeline, "num_train_timesteps", 1000))
        solver = str(self.sample_solver).lower()

        if solver == "unipc":
            sched = FlowUniPCMultistepScheduler(
                num_train_timesteps=num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False,
            )
            sched.set_timesteps(int(num_inference_steps), device=self.device(), shift=float(self.shift))
            timesteps = sched.timesteps
        elif solver in ("dpm++", "dpm", "dpmsolver++"):
            sched = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False,
            )
            sampling_sigmas = get_sampling_sigmas(int(num_inference_steps), float(self.shift))
            timesteps, _ = retrieve_timesteps(sched, device=self.device(), sigmas=sampling_sigmas)
        else:
            raise ValueError(f"Unsupported Wan solver '{self.sample_solver}'. Use 'unipc' or 'dpm++'.")

        self._scheduler = sched
        return [t for t in timesteps]
    
    def prepare_latents(self, *, seed: int) -> torch.Tensor:
        """
        Match WanT2V.generate latent initialization:
        target_shape = (z_dim, (F - 1) // vae_stride_t + 1, H//vae_stride_h, W//vae_stride_w)
        noise is float32 and stored as a single latent tensor (we keep a batch dim).
        """
        g = torch.Generator(device=self.device()).manual_seed(int(seed))

        vae = getattr(self.pipeline, "vae", None)
        if vae is None or not hasattr(vae, "model") or not hasattr(vae.model, "z_dim"):
            raise RuntimeError("WanAdapter expects pipeline.vae.model.z_dim (WanVAE).")

        z_dim = int(vae.model.z_dim)
        vae_stride = getattr(self.pipeline, "vae_stride", (4, 8, 8))
        if len(vae_stride) != 3:
            raise RuntimeError(f"Unexpected pipeline.vae_stride: {vae_stride}")

        F = int(self.num_frames)
        latent_f = (F - 1) // int(vae_stride[0]) + 1
        latent_h = int(self.height) // int(vae_stride[1])
        latent_w = int(self.width) // int(vae_stride[2])

        lat = torch.randn(
            (1, z_dim, latent_f, latent_h, latent_w),
            dtype=torch.float32,
            device=self.device(),
            generator=g,
        )
        return lat
    
    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        # Wan uses `.model` (WanModel) with `.blocks` (transformer blocks).
        model = getattr(self.pipeline, "model", None)
        if model is None:
            raise RuntimeError("WanAdapter expects pipeline.model (WanModel).")

        blocks = getattr(model, "blocks", None)
        if blocks is None:
            raise RuntimeError("WanAdapter expects pipeline.model.blocks to select trainable blocks.")

        # Case 1: explicit block list provided
        if self.train_transformer_blocks:
            ids = set(int(x) for x in self.train_transformer_blocks)
        # Case 2: unfreeze last N% of blocks
        elif self.unfreeze_percentage is not None:
            p = float(self.unfreeze_percentage)
            if not (0.0 < p <= 1.0):
                raise ValueError(f"unfreeze_percentage must be in (0, 1], got {p}")
            total = len(blocks)
            if total == 0:
                raise RuntimeError("WanAdapter: pipeline.model.blocks is empty; cannot unfreeze any blocks.")
            # Use ceil so percentages map more intuitively to counts (avoid rounding down to 0).
            k = max(1, int(math.ceil(total * p)))
            start = max(0, total - k)
            ids = set(range(start, total))
            # Store for logging/debugging
            self.train_transformer_blocks = list(range(start, total))
        # Case 3: train all blocks
        else:
            for p in model.parameters():
                p.requires_grad_(True)
            return [p for p in model.parameters() if p.requires_grad]
        
        # Freeze everything first, then unfreeze selected blocks.
        for p in model.parameters():
            p.requires_grad_(False)
        for i, blk in enumerate(blocks):
            req = i in ids
            for p in blk.parameters():
                p.requires_grad_(req)
        
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
        if self._scheduler is None:
            raise RuntimeError("WanAdapter.step() called before get_timesteps(); scheduler is not initialized.")

        model = getattr(self.pipeline, "model", None)
        if model is None:
            raise RuntimeError("WanAdapter expects pipeline.model (WanModel).")

        # ctx.t is a scalar timestep token (see grpo_core); keep it in Wan2.1's expected form.
        t = ctx.t
        if not torch.is_tensor(t):
            t = torch.tensor(t, device=self.device(), dtype=torch.float32)
        else:
            t = t.to(device=self.device())

        # Exploration noise (only for rollouts, not for gradient step)
        lat_in = latents.detach() if not with_grad else latents
        if (not with_grad) and int(rollout_index) > 0 and float(rollout_noise_scale) > 0:
            lat_in = lat_in + float(rollout_noise_scale) * torch.randn_like(lat_in)
        
        # WanModel expects a *list* of latents, each shaped [C, F, H, W].
        if lat_in.ndim != 5:
            raise ValueError(f"WanAdapter expected latents shape [B,C,F,H,W], got {tuple(lat_in.shape)}")
        x_list = [lat_in[0]]

        # Wan passes t as a stacked tensor of shape [1]
        timestep = torch.stack([t]).to(device=self.device(), dtype=torch.float32)

        # Prompt contexts are lists of tensors (WanT2V.text_encoder output)
        ctx_cond = self.prompt_embeds
        ctx_uncond = self.negative_prompt_embeds if self.negative_prompt_embeds is not None else self.prompt_embeds

        # Use WanT2V seq_len computation if caller provided one; otherwise approximate.
        seq_len = getattr(self, "seq_len", None)
        if seq_len is None:
            patch_size = getattr(self.pipeline, "patch_size", (1, 2, 2))
            # lat_in[0] is [C, F_lat, H_lat, W_lat]
            _, f_lat, h_lat, w_lat = x_list[0].shape
            seq_len = int(
                math.ceil((h_lat * w_lat) / (patch_size[1] * patch_size[2]) * f_lat)
            )

        param_dtype = getattr(self.pipeline, "param_dtype", torch.bfloat16)

        def _forward(context_list: Any) -> torch.Tensor:
            out_list = model(x_list, t=timestep, context=context_list, seq_len=int(seq_len))
            # WanModel returns list[Tensor]; take first element
            return out_list[0]
        
        if with_grad:
            with amp.autocast(dtype=param_dtype):
                noise_pred_cond = _forward(ctx_cond)
                noise_pred_uncond = _forward(ctx_uncond)
        else:
            with torch.no_grad(), amp.autocast(dtype=param_dtype):
                noise_pred_cond = _forward(ctx_cond)
                noise_pred_uncond = _forward(ctx_uncond)
        
        # CFG (matches WanT2V.generate)
        noise_pred = noise_pred_uncond + float(self.guidance_scale) * (noise_pred_cond - noise_pred_uncond)
        
        # Scheduler step (matches WanT2V.generate: step(noise_pred.unsqueeze(0), t, latents[0].unsqueeze(0)))
        step_out = self._scheduler.step(
            noise_pred.unsqueeze(0),
            t,
            lat_in,
            return_dict=False,
        )
        next_latents = step_out[0] if isinstance(step_out, (tuple, list)) else step_out

        return StepOutput(next_latents=next_latents, action=noise_pred, x0_latents=None, solver_state=solver_state)
    
    def decode_for_reward(self, *, latents_or_x0: torch.Tensor, x0_is_patchified: bool) -> torch.Tensor:
        vae = getattr(self.pipeline, "vae", None)
        if vae is None or not hasattr(vae, "decode"):
            raise RuntimeError("WanAdapter expects pipeline.vae.decode (WanVAE).")
        
        lat = latents_or_x0
        if lat.ndim == 5:
            lat0 = lat[0]
        elif lat.ndim == 4:
            lat0 = lat
        else:
            raise ValueError(f"Unexpected latent shape for Wan decode: {tuple(lat.shape)}")

        # WanVAE.decode expects a list of tensors [C, F_lat, H_lat, W_lat]
        with torch.no_grad():
            vids = vae.decode([lat0.to(device=getattr(vae, "device", self.device()), dtype=torch.float32)])
            vid = vids[0]  # [3, F, H, W] in [-1, 1]
            vid = (vid / 2 + 0.5).clamp(0, 1)
            vid = vid.permute(1, 0, 2, 3).contiguous().unsqueeze(0)  # [1, F, 3, H, W]
        
        return vid
    
    def extra_log_state(self) -> Dict[str, Any]:
        return {
            "height": int(self.height),
            "width": int(self.width),
            "num_frames": int(self.num_frames),
            "guidance_scale": float(self.guidance_scale),
            "shift": float(self.shift),
            "sample_solver": str(self.sample_solver),
            "train_transformer_blocks": self.train_transformer_blocks or [],
            "unfreeze_percentage": float(self.unfreeze_percentage) if self.unfreeze_percentage is not None else None,
        }
