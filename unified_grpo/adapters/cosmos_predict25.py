from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

from unified_grpo.adapters.base import StepContext, StepOutput, VideoGRPOAdapter
from unified_grpo.lora_utils import get_trainable_lora_parameters, has_lora


@dataclass
class CosmosPredict25Adapter(VideoGRPOAdapter):
    """
    Adapter for local Cosmos-Predict2.5 Text2World / Video2World models.

    This adapter currently targets the 2B pre-trained Text2World path and exposes
    a one-step denoising interface compatible with the unified GRPO core.
    """

    model: Any
    prompt: str
    negative_prompt: str
    height: int
    width: int
    num_frames: int
    guidance_scale: float = 7.0
    train_transformer_blocks: Optional[list[int]] = None

    name: str = "cosmos"

    def __post_init__(self) -> None:
        # Expose the trainable network via a common attribute so checkpoint saving
        # in unified_grpo/run.py can discover it.
        self.transformer = self.model.net

        self._device = torch.device("cuda")
        self._input_data_key = str(getattr(self.model, "input_data_key", "video"))
        self._caption_key = str(getattr(self.model, "input_caption_key", "ai_caption"))
        self._pixel_num_frames = int(self.model.tokenizer.get_pixel_num_frames(self.model.config.state_t))

        if int(self.num_frames) != self._pixel_num_frames:
            print(
                f"[COSMOS] Requested num_frames={self.num_frames}, but the loaded model expects "
                f"{self._pixel_num_frames} pixel frames. Using {self._pixel_num_frames}."
            )
            self.num_frames = self._pixel_num_frames

        # Text2World uses an all-zero video template plus text conditioning.
        self._video_template = torch.zeros(
            1, 3, int(self.num_frames), int(self.height), int(self.width), dtype=torch.uint8
        )

        self._prompt_embeds = self._encode_text(self.prompt)
        self._negative_prompt_embeds = self._encode_text(self.negative_prompt)

        self._scheduler_reset_state = None

        # Stable eval mode; this does not disable gradients.
        self.model.eval()
        self.model.net.eval()
        try:
            if self.model.text_encoder is not None:
                self.model.text_encoder.eval()
        except Exception:
            pass

    def device(self) -> torch.device:
        return self._device

    def _encode_text(self, text: str) -> torch.Tensor:
        if self.model.text_encoder is not None:
            embeds = self.model.text_encoder.compute_text_embeddings_online(
                data_batch={self._caption_key: [text], "images": None},
                input_caption_key=self._caption_key,
            )
            return embeds.detach()
        from cosmos_predict2._src.predict2.inference.get_t5_emb import get_text_embedding

        return get_text_embedding(text).detach()

    def _make_data_batch(self) -> Dict[str, Any]:
        padding_mask = torch.zeros(1, 1, int(self.height), int(self.width), device=self._device, dtype=torch.bfloat16)
        data_batch: Dict[str, Any] = {
            "dataset_name": "video_data",
            self._input_data_key: self._video_template.clone(),
            "fps": torch.full((1,), 24.0, device=self._device, dtype=torch.bfloat16),
            "padding_mask": padding_mask,
            "num_conditional_frames": 0,
            self._caption_key: [self.prompt],
            "t5_text_embeddings": self._prompt_embeds.clone(),
            "neg_t5_text_embeddings": self._negative_prompt_embeds.clone(),
        }
        return data_batch

    def _clone_state_value(self, value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.detach().clone()
        if isinstance(value, list):
            return [self._clone_state_value(v) for v in value]
        if isinstance(value, tuple):
            return tuple(self._clone_state_value(v) for v in value)
        if isinstance(value, dict):
            return {k: self._clone_state_value(v) for k, v in value.items()}
        return copy.deepcopy(value)

    def _snapshot_scheduler(self) -> Dict[str, Any]:
        scheduler = self.model.sample_scheduler
        return {
            "model_outputs": self._clone_state_value(getattr(scheduler, "model_outputs", None)),
            "timestep_list": self._clone_state_value(getattr(scheduler, "timestep_list", None)),
            "last_sample": self._clone_state_value(getattr(scheduler, "last_sample", None)),
            "lower_order_nums": self._clone_state_value(getattr(scheduler, "lower_order_nums", None)),
            "_step_index": self._clone_state_value(getattr(scheduler, "_step_index", None)),
            "_begin_index": self._clone_state_value(getattr(scheduler, "_begin_index", None)),
            "this_order": self._clone_state_value(getattr(scheduler, "this_order", None)),
        }

    def _restore_scheduler(self, state: Optional[Dict[str, Any]]) -> None:
        scheduler = self.model.sample_scheduler
        src = self._scheduler_reset_state if state is None else state
        if src is None:
            return
        scheduler.model_outputs = self._clone_state_value(src.get("model_outputs"))
        scheduler.timestep_list = self._clone_state_value(src.get("timestep_list"))
        scheduler.last_sample = self._clone_state_value(src.get("last_sample"))
        scheduler.lower_order_nums = self._clone_state_value(src.get("lower_order_nums"))
        scheduler._step_index = self._clone_state_value(src.get("_step_index"))
        scheduler._begin_index = self._clone_state_value(src.get("_begin_index"))
        scheduler.this_order = self._clone_state_value(src.get("this_order"))

    def get_timesteps(self, *, num_inference_steps: int) -> list[torch.Tensor]:
        self.model.sample_scheduler.set_timesteps(
            int(num_inference_steps),
            device=self.device(),
            shift=5.0,
            use_kerras_sigma=self.model.config.use_kerras_sigma_at_inference,
        )
        self._scheduler_reset_state = self._snapshot_scheduler()
        return [t for t in self.model.sample_scheduler.timesteps]

    def prepare_latents(self, *, seed: int) -> torch.Tensor:
        g = torch.Generator(device=self.device())
        g.manual_seed(int(seed))

        latent_h = int(self.height) // int(self.model.tokenizer.spatial_compression_factor)
        latent_w = int(self.width) // int(self.model.tokenizer.spatial_compression_factor)
        latent_t = int(self.model.tokenizer.get_latent_num_frames(int(self.num_frames)))
        state_shape = (int(self.model.config.state_ch), latent_t, latent_h, latent_w)
        latents = torch.randn((1,) + state_shape, device=self.device(), generator=g, dtype=torch.float32)
        return latents

    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        net = self.model.net
        if has_lora(net):
            return get_trainable_lora_parameters(net, verbose_prefix="  [Cosmos Adapter]")

        # Fallback: partial unfreeze by generic block-name heuristics if requested,
        # otherwise train the full network.
        for p in net.parameters():
            p.requires_grad_(False)

        if self.train_transformer_blocks:
            selected = set(int(x) for x in self.train_transformer_blocks)
            matched = 0
            for name, param in net.named_parameters():
                if any(
                    marker in name
                    for marker in (
                        "transformer_blocks.",
                        "blocks.",
                        "layers.",
                        "double_blocks.",
                        "single_blocks.",
                    )
                ):
                    if any(f".{idx}." in name for idx in selected):
                        param.requires_grad_(True)
                        matched += 1
            if matched > 0:
                return [p for p in net.parameters() if p.requires_grad]
            print("[COSMOS] Warning: could not map train_blocks to parameter names; training full net instead.")

        for p in net.parameters():
            p.requires_grad_(True)
        return list(net.parameters())

    def step(
        self,
        *,
        latents: torch.Tensor,
        step_context: StepContext,
        with_grad: bool,
        solver_state: Optional[Any] = None,
    ) -> StepOutput:
        self._restore_scheduler(solver_state)

        initial_noise = latents.detach().clone() if solver_state is None else solver_state.get("initial_noise")
        if initial_noise is None:
            initial_noise = latents.detach().clone()

        data_batch = self._make_data_batch()
        velocity_fn = self.model.get_velocity_fn_from_batch(
            data_batch, guidance=float(self.guidance_scale), is_negative_prompt=True
        )

        timestep = torch.stack([step_context.t]).unsqueeze(0)
        scheduler = self.model.sample_scheduler

        def _forward() -> torch.Tensor:
            velocity_pred = velocity_fn(initial_noise, latents, timestep)
            step_out = scheduler.step(
                velocity_pred,
                step_context.t,
                latents,
                return_dict=True,
            )
            next_latents = step_out.prev_sample
            return velocity_pred, next_latents

        if with_grad:
            velocity_pred, next_latents = _forward()
        else:
            with torch.no_grad():
                velocity_pred, next_latents = _forward()

        next_solver_state = self._snapshot_scheduler()
        next_solver_state["initial_noise"] = initial_noise.detach().clone()

        return StepOutput(
            next_latents=next_latents,
            action=velocity_pred,
            x0_latents=None,
            solver_state=next_solver_state,
        )

    def prepare_latents_for_reward(
        self,
        *,
        latents: torch.Tensor,
    ) -> torch.Tensor:
        return latents

    def decode_for_reward(self, *, latents_or_x0: torch.Tensor, x0_is_patchified: bool) -> torch.Tensor:
        del x0_is_patchified
        latents = latents_or_x0
        if latents.ndim == 4:
            latents = latents.unsqueeze(0)

        with torch.no_grad():
            video = self.model.decode(latents)
            video = ((video + 1.0) / 2.0).clamp(0, 1)  # [B, C, T, H, W]
            video = video.permute(0, 2, 1, 3, 4).contiguous()  # [B, T, C, H, W]
        return video[0]

    def extra_log_state(self) -> Dict[str, Any]:
        return {
            "height": int(self.height),
            "width": int(self.width),
            "num_frames": int(self.num_frames),
            "guidance_scale": float(self.guidance_scale),
            "train_transformer_blocks": self.train_transformer_blocks or [],
            "model_resolution": getattr(self.model.config, "resolution", None),
        }
