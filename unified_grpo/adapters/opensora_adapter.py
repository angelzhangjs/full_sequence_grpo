from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

from unified_grpo.adapters.base import StepContext, StepOutput, VideoGRPOAdapter


@dataclass
class OpenSoraComponents:
    """Container for Open-Sora components (model/vae/text_encoder/scheduler) built from `Open-Sora/`."""

    model: Any
    vae: Any
    text_encoder: Any
    scheduler: Any
    model_args: Dict[str, torch.Tensor]
    device: torch.device
    dtype: torch.dtype
    num_timesteps: int
    latent_size: tuple[int, int, int]

    @staticmethod
    def from_config(
        *,
        config_path: str,
        model_path: Optional[str],
        device: str,
        dtype: str,
        height: int,
        width: int,
        num_frames: int,
        num_sampling_steps: int,
        guidance_scale: float,
        seed: int,
    ) -> "OpenSoraComponents":
        from mmengine.config import Config

        from opensora.registry import MODELS, SCHEDULERS, build_module
        from opensora.utils.inference_utils import prepare_multi_resolution_info

        cfg = Config.fromfile(config_path)

        if model_path:
            cfg.model["from_pretrained"] = model_path

        # Open-Sora configs sometimes enable apex fused LayerNorm kernels by default.
        # If apex fused LN isn't actually available, Open-Sora will raise:
        #   RuntimeError: FusedLayerNorm not available. Please install apex.
        # For portability, disable it only when the specific fused LN import is missing.
        try:
            from apex.normalization import FusedLayerNorm  # type: ignore  # noqa: F401

            _fused_ln_available = True
        except Exception:
            _fused_ln_available = False
        if (not _fused_ln_available) and isinstance(getattr(cfg, "model", None), dict):
            if bool(cfg.model.get("enable_layernorm_kernel", False)):
                cfg.model["enable_layernorm_kernel"] = False

        # Force explicit image_size/frames to match unified args.
        cfg.image_size = (int(height), int(width))
        cfg.num_frames = int(num_frames)

        # Override scheduler sampling steps + guidance scale.
        if "scheduler" in cfg and isinstance(cfg.scheduler, dict):
            cfg.scheduler["num_sampling_steps"] = int(num_sampling_steps)
            cfg.scheduler["cfg_scale"] = float(guidance_scale)

        dev = torch.device(device)
        dt = torch.bfloat16 if dtype.lower() in ("bf16", "bfloat16") else torch.float16 if dtype.lower() in ("fp16", "float16") else torch.float32

        text_encoder = build_module(cfg.text_encoder, MODELS, device=str(dev))
        vae = build_module(cfg.vae, MODELS).to(dev, dt).eval()

        input_size = (int(cfg.num_frames), *cfg.image_size)
        latent_size = tuple(int(x) for x in vae.get_latent_size(input_size))

        model = (
            build_module(
                cfg.model,
                MODELS,
                input_size=latent_size,
                in_channels=vae.out_channels,
                caption_channels=text_encoder.output_dim,
                model_max_length=text_encoder.model_max_length,
                enable_sequence_parallelism=False,
            )
            .to(dev, dt)
            .eval()
        )
        text_encoder.y_embedder = model.y_embedder

        scheduler = build_module(cfg.scheduler, SCHEDULERS)

        fps = int(getattr(cfg, "fps", 24))
        model_args = prepare_multi_resolution_info(
            getattr(cfg, "multi_resolution", None),
            batch_size=1,
            image_size=cfg.image_size,
            num_frames=int(cfg.num_frames),
            fps=fps,
            device=dev,
            dtype=dt,
        )

        return OpenSoraComponents(
            model=model,
            vae=vae,
            text_encoder=text_encoder,
            scheduler=scheduler,
            model_args=model_args,
            device=dev,
            dtype=dt,
            num_timesteps=int(getattr(scheduler, "num_timesteps", 1000)),
            latent_size=latent_size,
        )


@dataclass
class OpenSoraAdapter(VideoGRPOAdapter):
    """Step-wise GRPO adapter for Open-Sora RFLOW scheduler."""

    components: OpenSoraComponents
    prompt: str
    guidance_scale: float = 7.0
    height: int = 480
    width: int = 720
    num_frames: int = 49

    train_transformer_blocks: Optional[List[int]] = None
    unfreeze_percentage: Optional[float] = None

    _timesteps: Optional[List[torch.Tensor]] = None
    _dts: Optional[List[torch.Tensor]] = None
    _static_model_args: Optional[Dict[str, torch.Tensor]] = None

    name: str = "opensora"

    def device(self) -> torch.device:
        return self.components.device

    def get_timesteps(self, *, num_inference_steps: int) -> list[torch.Tensor]:
        sched = self.components.scheduler
        if not hasattr(sched, "num_timesteps"):
            raise RuntimeError("OpenSoraAdapter currently supports only the RFLOW scheduler.")

        num_timesteps = int(getattr(sched, "num_timesteps", 1000))
        use_discrete = bool(getattr(sched, "use_discrete_timesteps", False))
        use_transform = bool(getattr(sched, "use_timestep_transform", False))

        additional_args = dict(self.components.model_args)

        ts_f = [(1.0 - i / float(num_inference_steps)) * float(num_timesteps) for i in range(num_inference_steps)]
        if use_discrete:
            ts_f = [int(round(t)) for t in ts_f]

        ts = [torch.tensor([t], device=self.device(), dtype=torch.float32) for t in ts_f]
        if use_transform:
            from opensora.schedulers.rf.rectified_flow import timestep_transform

            ts = [timestep_transform(t, additional_args, num_timesteps=num_timesteps) for t in ts]

        dts: list[torch.Tensor] = []
        for i in range(len(ts)):
            dt = ts[i] - ts[i + 1] if i < len(ts) - 1 else ts[i]
            dts.append(dt / float(num_timesteps))

        self._timesteps = ts
        self._dts = dts
        return ts

    def prepare_latents(self, *, seed: int) -> torch.Tensor:
        g = torch.Generator(device=str(self.device()))
        g.manual_seed(int(seed))
        t_lat, h_lat, w_lat = self.components.latent_size
        z = torch.randn(
            1,
            int(self.components.vae.out_channels),
            int(t_lat),
            int(h_lat),
            int(w_lat),
            device=self.device(),
            dtype=self.components.dtype,
            generator=g,
        )

        from opensora.models.text_encoder.t5 import text_preprocessing

        prompt = text_preprocessing(self.prompt)
        te = self.components.text_encoder
        model_args = te.encode([prompt])
        y_null = te.null(1)
        model_args["y"] = torch.cat([model_args["y"], y_null], 0)
        model_args.update(self.components.model_args)
        self._static_model_args = model_args

        return z

    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        model = self.components.model
        blocks = getattr(model, "blocks", None)

        if not self.train_transformer_blocks and self.unfreeze_percentage is None:
            for p in model.parameters():
                p.requires_grad_(True)
            return [p for p in model.parameters() if p.requires_grad]

        ids: Optional[set[int]] = None
        if self.train_transformer_blocks:
            ids = set(int(x) for x in self.train_transformer_blocks)
        elif self.unfreeze_percentage is not None and blocks is not None:
            import math

            p = float(self.unfreeze_percentage)
            total = len(blocks)
            k = max(1, int(math.ceil(total * p)))
            start = max(0, total - k)
            ids = set(range(start, total))
            self.train_transformer_blocks = list(range(start, total))

        for p in model.parameters():
            p.requires_grad_(False)
        if ids is None or blocks is None:
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
        step_context: StepContext,
        with_grad: bool,
        solver_state=None,
    ) -> StepOutput:
        if self._timesteps is None or self._dts is None:
            raise RuntimeError("OpenSoraAdapter.step() called before get_timesteps().")
        if self._static_model_args is None:
            raise RuntimeError("OpenSoraAdapter.step() called before prepare_latents().")

        i = int(step_context.step_index)
        z = latents.detach() if not with_grad else latents

        t = self._timesteps[i]
        dt = self._dts[i]

        z_in = torch.cat([z, z], 0)
        t_in = torch.cat([t, t], 0)

        def _forward():
            return self.components.model(z_in, t_in, **self._static_model_args)

        if with_grad:
            out = _forward()
        else:
            with torch.no_grad():
                out = _forward()

        pred = out.chunk(2, dim=1)[0]
        pred_cond, pred_uncond = pred.chunk(2, dim=0)
        v_pred = pred_uncond + float(self.guidance_scale) * (pred_cond - pred_uncond)

        z_next = z + v_pred * dt[:, None, None, None, None]
        return StepOutput(next_latents=z_next, action=v_pred, x0_latents=None, solver_state=solver_state)

    def decode_for_reward(self, *, latents_or_x0: torch.Tensor, x0_is_patchified: bool) -> torch.Tensor:
        # Decode to pixel-space video and return per-frame tensor [T, 3, H, W] in [0, 1] if possible.
        video = self.components.vae.decode(latents_or_x0.to(self.components.dtype), num_frames=int(self.num_frames))
        # expected [B, C, T, H, W]
        if video.ndim == 5:
            video = video[0]
        if video.ndim != 4:
            raise ValueError(f"Unexpected OpenSora decoded video shape: {tuple(video.shape)}")
        # [C, T, H, W] -> [T, C, H, W]
        if video.shape[0] == 3:
            return video.permute(1, 0, 2, 3).contiguous()
        # If already [T, C, H, W], return as-is.
        if video.shape[1] == 3:
            return video.contiguous()
        raise ValueError(f"Unexpected OpenSora channel layout in decoded video: {tuple(video.shape)}")
