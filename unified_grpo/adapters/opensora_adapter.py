from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

from unified_grpo.adapters.base import StepContext, StepOutput, VideoGRPOAdapter


def _require_opensora() -> Any:
    """
    Import Open-Sora utilities lazily so this adapter file can exist even when Open-Sora deps
    (mmengine/colossalai/etc) aren't installed in the current python env.
    """
    try:
        import opensora  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "OpenSoraAdapter requires the Open-Sora package to be importable. "
            "Make sure you're running in the Open-Sora environment and that `Open-Sora/` is on PYTHONPATH."
        ) from e
    return opensora


@dataclass
class OpenSoraAdapter(VideoGRPOAdapter):
    """
    Minimal Open-Sora adapter for `unified_grpo/grpo_core.py`.

    Open-Sora's core denoising update in `opensora/utils/sampling.py` is:
      img = img + (t_prev - t_curr) * pred

    Where:
    - `img` is the *packed* latent token sequence [B, (T*H*W), D]
    - `pred` is the model output (velocity/flow-like prediction) in the same packed space
    - timesteps are floats in [1, 0] (Open-Sora's "flow time" schedule)

    This adapter implements per-step sampling (one `(t_curr, t_prev)` pair), uses Open-Sora's
    `prepare()` once to build text conditioning tensors, and uses the same CFG composition
    used by `I2VDenoiser` (3-way split).
    """

    # Open-Sora model components (constructed externally, e.g. via opensora.utils.sampling.prepare_models)
    model: torch.nn.Module
    model_ae: torch.nn.Module
    model_t5: torch.nn.Module
    model_clip: torch.nn.Module
    optional_models: Dict[str, torch.nn.Module]

    prompt: str
    negative_prompt: str = ""

    height: int = 512
    width: int = 768
    num_frames: int = 81
    num_inference_steps: int = 50

    # Open-Sora sampling knobs
    guidance: float = 4.0
    guidance_img: float = 1.0  # for pure t2v, keep 1.0 (no image-guidance effect)
    shift: bool = True
    flow_shift: Optional[float] = None
    patch_size: int = 2
    channel: int = 16  # cfg["model"]["in_channels"] in Open-Sora configs
    temporal_reduction: int = 1
    is_causal_vae: bool = False

    # Training control
    train_blocks: Optional[List[int]] = None

    name: str = "opensora"

    # Cached schedule/conditioning
    _timesteps_full: Optional[List[float]] = None  # length = num_steps+1
    _num_frames_lat: Optional[int] = None
    _inp_static: Optional[Dict[str, torch.Tensor]] = None  # excludes "img"
    _cond_token: Optional[torch.Tensor] = None  # packed cond tokens (for I2V-style CFG); shape [B, S, D]

    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _compute_num_frames_lat(self) -> int:
        if self.is_causal_vae:
            # Open-Sora: (num_frames - 1) // temporal_reduction + 1
            return 1 if self.num_frames == 1 else (self.num_frames - 1) // self.temporal_reduction + 1
        # non-causal: num_frames // temporal_reduction
        return 1 if self.num_frames == 1 else self.num_frames // self.temporal_reduction

    def get_timesteps(self, *, num_inference_steps: int) -> list[torch.Tensor]:
        _require_opensora()
        from opensora.utils.sampling import get_schedule  # type: ignore

        self.num_inference_steps = int(num_inference_steps)
        num_frames_lat = int(self._compute_num_frames_lat())
        self._num_frames_lat = num_frames_lat

        # Open-Sora uses AE_SPATIAL_COMPRESSION to determine latent spatial size; default 16.
        D = int(os.environ.get("AE_SPATIAL_COMPRESSION", 16))
        image_seq_len = int(((-(-self.height // D)) * (-(-self.width // D))))  # ceil(h/D)*ceil(w/D)

        ts = get_schedule(
            num_steps=int(self.num_inference_steps),
            image_seq_len=image_seq_len,
            num_frames=int(num_frames_lat),
            shift_alpha=self.flow_shift,
            shift=bool(self.shift),
        )
        # ts includes the extra last step (0). Core `step()` needs t_curr and t_prev, so we store full list.
        self._timesteps_full = list(ts)

        # Return only the "current" timesteps (exclude the last 0), so step_index i maps to (t_curr=ts[i], t_prev=ts[i+1]).
        return [torch.tensor(float(t), device=self.device(), dtype=torch.float32) for t in self._timesteps_full[:-1]]

    def prepare_latents(self, *, seed: int) -> torch.Tensor:
        _require_opensora()
        from opensora.utils.inference import prepare_inference_condition  # type: ignore
        from opensora.utils.sampling import SamplingMethod, SamplingMethodDict, get_noise, pack, prepare  # type: ignore

        if self._timesteps_full is None or self._num_frames_lat is None:
            raise RuntimeError("Call get_timesteps() before prepare_latents().")

        device = self.device()
        dtype = next(self.model.parameters()).dtype

        # 1) Sample initial latent noise z in 5D
        z = get_noise(
            num_samples=1,
            height=int(self.height),
            width=int(self.width),
            num_frames=int(self._num_frames_lat),
            device=device,
            dtype=dtype,
            seed=int(seed),
            patch_size=int(self.patch_size),
            channel=int(self.channel) // (int(self.patch_size) ** 2),
        )

        # 2) Prepare 3-way CFG text list (same behavior as I2VDenoiser.prepare_guidance)
        # Open-Sora uses method I2V for video generation, even when cond_type is t2v.
        denoiser = SamplingMethodDict[SamplingMethod.I2V]
        text_list = [self.prompt]
        neg_list = [self.negative_prompt] if self.negative_prompt != "" else None
        text_list, additional = denoiser.prepare_guidance(
            text=text_list,
            optional_models=self.optional_models,
            device=device,
            dtype=dtype,
            neg=neg_list,
            guidance_img=float(self.guidance_img),
        )

        # 3) Build static model inputs (txt/txt_ids/y_vec/img_ids) using Open-Sora's prepare()
        # NOTE: prepare() will repeat `img` to match the number of prompts (3x) internally.
        inp = prepare(self.model_t5, self.model_clip, z, prompt=text_list, patch_size=int(self.patch_size))
        inp.update(additional)

        # 4) For t2v, Open-Sora still routes through I2V denoiser; condition tensors default to "empty" conditions.
        masks, masked_ref = prepare_inference_condition(z, "t2v", ref_list=[None], causal=bool(self.is_causal_vae))
        cond_5d = torch.cat((masks, masked_ref), dim=1)
        cond_tok = pack(cond_5d, patch_size=int(self.patch_size)).to(device=device, dtype=dtype)
        self._cond_token = cond_tok  # [B, S, D]

        # store everything except "img" (img is the evolving latent state)
        self._inp_static = {k: v for k, v in inp.items() if k != "img"}

        # Return the *base* latent state as packed tokens (B=1), not profiler's 3x batch.
        # `inp["img"]` is repeated to 3x; we keep the first 1/3 as the state.
        img3 = inp["img"]
        base = img3[: img3.shape[0] // 3].contiguous()
        return base

    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        # By default train diffusion model only (not AE/text encoders).
        m = self.model
        for p in m.parameters():
            p.requires_grad_(False)

        if not self.train_blocks:
            for p in m.parameters():
                p.requires_grad_(True)
            return [p for p in m.parameters() if p.requires_grad]

        # Best-effort block unfreezing if model exposes `blocks` like DiT.
        if hasattr(m, "blocks"):
            ids = set(int(i) for i in self.train_blocks)
            for i, blk in enumerate(m.blocks):  # type: ignore[attr-defined]
                req = i in ids
                for p in blk.parameters():
                    p.requires_grad_(req)
        else:
            # Fallback: nothing to unfreeze by block id; unfreeze all.
            for p in m.parameters():
                p.requires_grad_(True)

        return [p for p in m.parameters() if p.requires_grad]

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
        _require_opensora()
        from opensora.utils.sampling import unpack  # type: ignore  # only used for type parity elsewhere

        if self._timesteps_full is None or self._inp_static is None or self._cond_token is None:
            raise RuntimeError("Call get_timesteps() and prepare_latents() before step().")

        i = int(ctx.step_index)
        if i < 0 or i + 1 >= len(self._timesteps_full):
            raise IndexError(f"step_index={i} out of range for schedule length {len(self._timesteps_full)}")

        t_curr = float(self._timesteps_full[i])
        t_prev = float(self._timesteps_full[i + 1])

        lat_in = latents
        if (not with_grad) and int(rollout_index) > 0 and float(rollout_noise_scale) > 0:
            lat_in = latents + float(rollout_noise_scale) * torch.randn_like(latents)

        # Open-Sora I2V denoiser uses a 3-way batch (cond / uncond / uncond_2).
        img = torch.cat([lat_in, lat_in, lat_in], dim=0)

        device = img.device
        dtype = img.dtype

        # Build cond tokens: [B,S,D] -> [3B,S,D] (cond, cond, zeros)
        cond = self._cond_token.to(device=device, dtype=dtype)
        cond3 = torch.cat([cond, cond, torch.zeros_like(cond)], dim=0)

        t_vec = torch.full((img.shape[0],), t_curr, dtype=dtype, device=device)
        guidance_vec = torch.full((img.shape[0],), float(self.guidance), dtype=dtype, device=device)

        def _forward() -> torch.Tensor:
            return self.model(
                img=img,
                cond=cond3,
                timesteps=t_vec,
                guidance=guidance_vec,
                **self._inp_static,
            )

        if with_grad:
            pred_all = _forward()
        else:
            with torch.no_grad():
                pred_all = _forward()

        # 3-way CFG composition, matching `I2VDenoiser.denoise`:
        # pred = uncond_2 + image_gs*(uncond-uncond_2) + text_gs*(cond-uncond)
        cond_pred, uncond_pred, uncond2_pred = pred_all.chunk(3, dim=0)
        text_gs = float(self.guidance)
        image_gs = float(self.guidance_img)
        pred = uncond2_pred + image_gs * (uncond_pred - uncond2_pred) + text_gs * (cond_pred - uncond_pred)

        # Apply same update to all 3 copies (as in Open-Sora), then keep first third as the state.
        pred3 = torch.cat([pred, pred, pred], dim=0)
        img_next = img + (t_prev - t_curr) * pred3
        next_latents = img_next[: img_next.shape[0] // 3].contiguous()

        # Action for GRPO = the guided per-step prediction driving the update (packed token space).
        return StepOutput(
            next_latents=next_latents,
            action=pred.detach() if not with_grad else pred,
            x0_latents=None,
            solver_state=solver_state,  # Open-Sora step here is stateless; keep passthrough for API consistency
        )

    def decode_for_reward(self, *, latents_or_x0: torch.Tensor, x0_is_patchified: bool) -> torch.Tensor:
        _require_opensora()
        from opensora.utils.sampling import unpack  # type: ignore

        if self._num_frames_lat is None:
            raise RuntimeError("Call get_timesteps() before decode_for_reward().")

        x = latents_or_x0
        if x.dim() != 3:
            raise ValueError(f"OpenSoraAdapter expects packed latents [B,S,D], got shape {tuple(x.shape)}")

        # Packed tokens -> 5D latent -> decode
        x5 = unpack(
            x,
            int(self.height),
            int(self.width),
            int(self._num_frames_lat),
            patch_size=int(self.patch_size),
        )
        vid = self.model_ae.decode(x5)  # typically [B, 3, T, H, W]
        vid = vid[:, :, : int(self.num_frames)]

        # Return [B, T, 3, H, W] in [0,1] if decode is [-1,1]; clamp conservatively.
        if vid.min().item() < -0.1:
            vid = (vid / 2.0 + 0.5)
        vid = vid.clamp(0, 1)
        vid = vid.permute(0, 2, 1, 3, 4).contiguous()
        return vid

    def extra_log_state(self) -> Dict[str, Any]:
        return {
            "height": int(self.height),
            "width": int(self.width),
            "num_frames": int(self.num_frames),
            "num_inference_steps": int(self.num_inference_steps),
            "guidance": float(self.guidance),
            "guidance_img": float(self.guidance_img),
            "patch_size": int(self.patch_size),
            "channel": int(self.channel),
            "shift": bool(self.shift),
            "flow_shift": None if self.flow_shift is None else float(self.flow_shift),
            "temporal_reduction": int(self.temporal_reduction),
            "is_causal_vae": bool(self.is_causal_vae),
            "train_blocks": self.train_blocks or [],
        }


# User-facing alias (your wording)
OpenSoraWrapper = OpenSoraAdapter

