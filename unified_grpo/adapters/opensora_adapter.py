from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

from unified_grpo.adapters.base import StepContext, StepOutput, VideoGRPOAdapter


@dataclass
class OpenSoraAdapter(VideoGRPOAdapter):
    """
    Adapter for Open-Sora

    This adapter follows the Open-Sora sampling implementation in:
    - `Open-Sora/opensora/utils/sampling.py`
      update rule (flow-style Euler): `img = img + (t_prev - t_curr) * pred`

    Notes:
    - Open-Sora uses a *packed* latent representation shaped [B, L, D] where
      \(L = T * H' * W'\) and \(D = C * patch_size^2\).
    - The diffusion model forward takes keyword inputs produced by
      `opensora.utils.sampling.prepare(...)`: img/img_ids/txt/txt_ids/y_vec, plus
      timesteps and guidance.
    """

    # Expected pipeline wrapper fields:
    # - model: diffusion model
    # - ae: autoencoder with .decode()
    # - t5, clip: text encoders compatible with opensora.utils.sampling.prepare
    # - device, dtype: execution settings
    pipeline: Any

    # Prompt text for conditioning. (Open-Sora's `prepare()` encodes text using T5/CLIP)
    prompt: str
    negative_prompt: str = ""

    # Sampling configuration
    guidance_scale: float = 4.0  # Open-Sora default in SamplingOption
    patch_size: int = 2
    channel: Optional[int] = None  # if None, inferred from pipeline.model.in_channels
    shift: bool = True
    flow_shift: Optional[float] = None

    # Generation geometry (pixel-space)
    height: int = 512
    width: int = 512
    num_frames: int = 17  # Open-Sora often uses 17-frame chunks
    temporal_reduction: int = 1

    # Trainable subset selection
    train_transformer_blocks: Optional[List[int]] = None
    unfreeze_percentage: Optional[float] = None  # optional "last N%" unfreezing

    # Cached schedule + static conditioning tensors
    _timesteps: Optional[List[float]] = None  # schedule length = num_steps+1
    _static_inp: Optional[Dict[str, torch.Tensor]] = None
    _num_latent_frames: Optional[int] = None

    name: str = "opensora"
    
    def device(self) -> torch.device:
        dev = getattr(self.pipeline, "device", None)
        return torch.device(dev) if dev is not None else torch.device("cuda")
    
    def get_timesteps(self, *, num_inference_steps: int) -> list[torch.Tensor]:
        # Build Open-Sora schedule (num_steps+1) and return step tokens (num_steps).
        try:
            from opensora.utils.sampling import get_schedule
        except Exception as e:
            raise RuntimeError("OpenSoraAdapter requires Open-Sora package importable as `opensora`.") from e

        # We need an estimate of sequence length to compute shifted schedule.
        # Use the latent noise spatial size implied by AE compression (env var in Open-Sora code).
        # We'll compute a conservative seq len based on the packed latent grid.
        h = int(self.height)
        w = int(self.width)
        t = int(self.num_frames) // max(1, int(self.temporal_reduction))
        # This matches `unpack`'s internal grid: ceil(height/D), ceil(width/D), multiplied by patch_size.
        import math, os
        D = int(os.environ.get("AE_SPATIAL_COMPRESSION", 16))
        h_lat = self.patch_size * int(math.ceil(h / D))
        w_lat = self.patch_size * int(math.ceil(w / D))
        image_seq_len = (h_lat * w_lat) // (self.patch_size**2)

        timesteps = get_schedule(
            num_steps=int(num_inference_steps),
            image_seq_len=int(image_seq_len),
            num_frames=int(t),
            shift_alpha=self.flow_shift,
            shift=bool(self.shift),
        )
        # Cache full schedule (includes final zero)
        self._timesteps = [float(x) for x in timesteps]
        return [torch.tensor(float(x), device=self.device()) for x in timesteps[:-1]]
    
    def prepare_latents(self, *, seed: int) -> torch.Tensor:
        """
        Initialize Open-Sora latent noise and build the static conditioning tensors
        via `opensora.utils.sampling.prepare(...)`.
        Returns the *packed* latent `img` shaped [B, L, D].
        """
        try:
            from opensora.utils.sampling import get_noise, prepare
        except Exception as e:
            raise RuntimeError("OpenSoraAdapter requires Open-Sora package importable as `opensora`.") from e

        model = getattr(self.pipeline, "model", None)
        if model is None:
            raise RuntimeError("OpenSoraAdapter expects pipeline.model (diffusion model).")
        ae = getattr(self.pipeline, "ae", None)
        if ae is None:
            raise RuntimeError("OpenSoraAdapter expects pipeline.ae (autoencoder).")
        t5 = getattr(self.pipeline, "t5", None)
        clip = getattr(self.pipeline, "clip", None)
        if t5 is None or clip is None:
            raise RuntimeError("OpenSoraAdapter expects pipeline.t5 and pipeline.clip.")

        dtype = getattr(self.pipeline, "dtype", torch.bfloat16)

        in_ch = int(self.channel) if self.channel is not None else int(getattr(model, "in_channels", 16))
        if in_ch % (self.patch_size**2) != 0:
            raise ValueError(f"model.in_channels={in_ch} must be divisible by patch_size^2={self.patch_size**2}")
        noise_ch = in_ch // (self.patch_size**2)
        num_latent_frames = int(self.num_frames) // max(1, int(self.temporal_reduction))
        self._num_latent_frames = int(num_latent_frames)

        z = get_noise(
            num_samples=1,
            height=int(self.height),
            width=int(self.width),
            num_frames=int(num_latent_frames),
            device=self.device(),
            dtype=dtype,
            seed=int(seed),
            patch_size=int(self.patch_size),
            channel=int(noise_ch),
        )

        # Build static conditioning tensors. `prepare()` packs img and encodes text.
        inp = prepare(
            t5=t5,
            clip=clip,
            img=z,
            prompt=[self.prompt],
            patch_size=int(self.patch_size),
        )
        # Cache everything except `img` (which is the latent state).
        self._static_inp = {k: v for k, v in inp.items() if k != "img"}

        return inp["img"]
    
    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        model = getattr(self.pipeline, "model", None)
        if model is None:
            raise RuntimeError("OpenSoraAdapter expects pipeline.model (diffusion model).")

        blocks = getattr(model, "blocks", None)

        # If no block list and no percentage: train all.
        if not self.train_transformer_blocks and self.unfreeze_percentage is None:
            for p in model.parameters():
                p.requires_grad_(True)
            return [p for p in model.parameters() if p.requires_grad]

        # If blocks are available, allow selecting last N% or explicit list.
        ids: Optional[set[int]] = None
        if self.train_transformer_blocks:
            ids = set(int(x) for x in self.train_transformer_blocks)
        elif self.unfreeze_percentage is not None and blocks is not None:
            import math
            p = float(self.unfreeze_percentage)
            if not (0.0 < p <= 1.0):
                raise ValueError(f"unfreeze_percentage must be in (0,1], got {p}")
            total = len(blocks)
            if total == 0:
                raise RuntimeError("OpenSoraAdapter: model.blocks is empty; cannot unfreeze any blocks.")
            k = max(1, int(math.ceil(total * p)))
            start = max(0, total - k)
            ids = set(range(start, total))
            self.train_transformer_blocks = list(range(start, total))

        for p in model.parameters():
            p.requires_grad_(False)
        if ids is None or blocks is None:
            # Fallback: unfreeze everything if we can't address blocks.
            for p in model.parameters():
                p.requires_grad_(True)
            return [p for p in model.parameters() if p.requires_grad]

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
        if self._timesteps is None:
            raise RuntimeError("OpenSoraAdapter.step() called before get_timesteps(); schedule is not initialized.")
        if self._static_inp is None:
            raise RuntimeError("OpenSoraAdapter.step() called before prepare_latents(); conditioning is not initialized.")

        model = getattr(self.pipeline, "model", None)
        if model is None:
            raise RuntimeError("OpenSoraAdapter expects pipeline.model (diffusion model).")

        # Open-Sora uses float timesteps in [0,1] (schedule from 1->0).
        # Use ctx.step_index to find the next timestep for Euler update.
        i = int(ctx.step_index)
        if i < 0 or i >= len(self._timesteps) - 1:
            raise IndexError(f"step_index {i} out of range for timesteps length {len(self._timesteps)}")
        t_curr = float(self._timesteps[i])
        t_prev = float(self._timesteps[i + 1])

        img = latents.detach() if not with_grad else latents
        if (not with_grad) and int(rollout_index) > 0 and float(rollout_noise_scale) > 0:
            img = img + float(rollout_noise_scale) * torch.randn_like(img)

        # Model expects `timesteps` as vector [B]
        t_vec = torch.full((img.shape[0],), t_curr, device=img.device, dtype=img.dtype)
        guidance_vec = torch.full(
            (img.shape[0],), float(self.guidance_scale), device=img.device, dtype=img.dtype
        )

        def _forward() -> torch.Tensor:
            return model(
                img=img,
                timesteps=t_vec,
                guidance=guidance_vec,
                **self._static_inp,
            )

        if with_grad:
            pred = _forward()
        else:
            with torch.no_grad():
                pred = _forward()

        # Euler update: img_{prev} = img_{curr} + (t_prev - t_curr) * pred
        next_img = img + (t_prev - t_curr) * pred

        return StepOutput(next_latents=next_img, action=pred, x0_latents=None, solver_state=solver_state)
    
    def decode_for_reward(self, *, latents_or_x0: torch.Tensor, x0_is_patchified: bool) -> torch.Tensor:
        if self._num_latent_frames is None:
            raise RuntimeError("OpenSoraAdapter.decode_for_reward() called before prepare_latents().")

        try:
            from opensora.utils.sampling import unpack
        except Exception as e:
            raise RuntimeError("OpenSoraAdapter requires Open-Sora package importable as `opensora`.") from e

        ae = getattr(self.pipeline, "ae", None)
        if ae is None or not hasattr(ae, "decode"):
            raise RuntimeError("OpenSoraAdapter expects pipeline.ae.decode.")

        img = latents_or_x0
        # unpack from [B,L,D] -> [B,C,T,H',W']
        z = unpack(
            img,
            height=int(self.height),
            width=int(self.width),
            num_frames=int(self._num_latent_frames),
            patch_size=int(self.patch_size),
        )
        with torch.no_grad():
            x = ae.decode(z)
        # x: [B, 3, T, H, W] (typically in [-1,1] or [0,1] depending on AE)
        x = x.float()
        if float(x.min()) < -0.1:
            x = (x / 2 + 0.5)
        x = x.clamp(0, 1)
        x = x[:, :, : int(self.num_frames)]  # trim to requested frame count
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # [B, T, C, H, W]
        return x
    
    def extra_log_state(self) -> Dict[str, Any]:
        return {
            "height": int(self.height),
            "width": int(self.width),
            "num_frames": int(self.num_frames),
            "guidance_scale": float(self.guidance_scale),
            "patch_size": int(self.patch_size),
            "temporal_reduction": int(self.temporal_reduction),
            "train_transformer_blocks": self.train_transformer_blocks or [],
            "unfreeze_percentage": float(self.unfreeze_percentage) if self.unfreeze_percentage is not None else None,
        }
