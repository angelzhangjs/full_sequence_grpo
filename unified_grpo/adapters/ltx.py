from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from ltx_video.models.autoencoders.vae_encode import latent_to_pixel_coords, vae_decode  # type: ignore
from unified_grpo.adapters.base import StepContext, StepOutput, VideoGRPOAdapter
from unified_grpo.lora_utils import get_trainable_lora_parameters, has_lora

@dataclass
class LTXAdapter(VideoGRPOAdapter):
    """
    Adapter for LTX-Video (Lightricks)
    """
    pipeline: Any
    prompt_embeds: torch.Tensor
    negative_prompt_embeds: Optional[torch.Tensor] = None
    prompt_attention_mask: Optional[torch.Tensor] = None
    negative_prompt_attention_mask: Optional[torch.Tensor] = None
    guidance_scale: float = 4.5
    # LTX "STG" (spatiotemporal guidance) settings (match ltx_video pipeline defaults/config)
    stg_scale: float = 1.0
    rescaling_scale: float = 0.7
    cfg_star_rescale: bool = False
    skip_layer_strategy: Optional[Any] = None
    skip_block_list: Optional[List[int]] = None
    # VAE decode settings (match pipeline config defaults)
    decode_timestep: float = 0.05
    decode_noise_scale: float = 0.025
    height: int = 512
    width: int = 768
    num_frames: int = 25
    train_transformer_blocks: Optional[List[int]] = None
    name: str = "ltx"

    # Cached coordinates/state for the current latent shape (used by LTX transformer calls)
    _indices_grid: Optional[torch.Tensor] = None
    _latent_height: Optional[int] = None
    _latent_width: Optional[int] = None
    _latent_frames: Optional[int] = None
    _frame_rate: float = 30.0
    
    def device(self) -> torch.device:
        dev = getattr(self.pipeline, "_execution_device", None)
        if dev is None:
            dev = getattr(self.pipeline, "device", None)
        return torch.device(dev) if dev is not None else torch.device("cuda")
    
    def apply_rollout_diversity(
        self,
        *,
        rng: random.Random,
        stg_scale_range: Optional[Tuple[float, float]] = None,
        rescaling_scale_range: Optional[Tuple[float, float]] = None,
        cfg_star_rescale_prob: Optional[float] = None,
    ) -> str:
        """
        LTX-only rollout diversity hook.

        This lets GRPO core stay model-agnostic: it can call this method if present,
        without knowing about LTX's STG/rescaling/cfg_star_rescale knobs.

        Returns:
          - info string (for logging)

        Note:
          This function intentionally applies changes *in-place* and does NOT restore
          previous values.
        """

        rollout_diversity_parts: List[str] = []

        if stg_scale_range is not None:
            lo, hi = float(stg_scale_range[0]), float(stg_scale_range[1])
            self.stg_scale = float(rng.uniform(lo, hi))
            rollout_diversity_parts.append(f"stg_scale={self.stg_scale:.2f}")

        if rescaling_scale_range is not None:
            lo, hi = float(rescaling_scale_range[0]), float(rescaling_scale_range[1])
            self.rescaling_scale = float(rng.uniform(lo, hi))
            rollout_diversity_parts.append(f"rescaling_scale={self.rescaling_scale:.2f}")

        if cfg_star_rescale_prob is not None:
            p = max(0.0, min(1.0, float(cfg_star_rescale_prob)))
            self.cfg_star_rescale = bool(rng.random() < p)
            rollout_diversity_parts.append(f"cfg_star_rescale={int(self.cfg_star_rescale)}")

        info = (" " + " ".join(rollout_diversity_parts)) if rollout_diversity_parts else ""
        return info
    
    def get_timesteps(self, *, num_inference_steps: int) -> list[torch.Tensor]:
        self.pipeline.scheduler.set_timesteps(num_inference_steps, device=self.device())
        return [t for t in self.pipeline.scheduler.timesteps]
    
    def prepare_latents(self, *, seed: int) -> torch.Tensor:
        g = torch.Generator(device=self.device()).manual_seed(seed)
        
        # LTX latent dimensions must match the pipeline's VAE scale factors.
        # (Using fixed numbers can explode token counts and VRAM.)
        vae_scale = int(getattr(self.pipeline, "vae_scale_factor", 32))
        video_scale = int(getattr(self.pipeline, "video_scale_factor", 8))

        latent_c = int(getattr(self.pipeline.vae.config, "latent_channels", 128))
        
        latent_h = int(self.height / vae_scale)
        latent_w = int(self.width / vae_scale)
        latent_f = int(self.num_frames / video_scale)

        latents_5d = torch.randn(
            (1, latent_c, latent_f, latent_h, latent_w), # (batch_size, latent_channels, latent_frames, latent_height, latent_width)
            generator=g,
            device=self.device(),
            dtype=torch.bfloat16,
        )
        
        latents_5d = latents_5d * self.pipeline.scheduler.init_noise_sigma
        # Patchify into (b, n, c) tokens + coords, as expected by the Transformer3DModel.
        latents, latent_coords = self.pipeline.patchifier.patchify(latents_5d)
        # Cache indices_grid (pixel coords scaled by 1/frame_rate) for transformer calls.
        pixel_coords = latent_to_pixel_coords(latent_coords, self.pipeline.vae, causal_fix=True)
        indices_grid = pixel_coords.to(torch.float32)
        indices_grid[:, 0] = indices_grid[:, 0] * (1.0 / float(self._frame_rate))

        self._indices_grid = indices_grid
        self._latent_height = latent_h
        self._latent_width = latent_w
        self._latent_frames = latent_f
        return latents
    
    def step(
        self,
        *,
        latents: torch.Tensor,
        step_context: StepContext,
        with_grad: bool,
        solver_state=None,
    ) -> StepOutput:
        
        t = step_context.t
        
        # Add noise for exploration
        latents = latents.detach() if not with_grad else latents
        # Base latents (single batch) used for the scheduler update.
        lat_base = latents
        
        # CFG + STG setup (mirror LTXVideoPipeline / origin_grpo logic)
        do_cfg = self.guidance_scale > 1.0 and self.negative_prompt_embeds is not None
        do_stg = float(self.stg_scale) > 0.0
        do_rescaling = float(self.rescaling_scale) != 1.0

        # Prompt embeddings must be detached leaf tensors (we don't train the text encoder).
        encoder_hidden_states = self.prompt_embeds.detach()
        encoder_attention_mask = self.prompt_attention_mask
        indices_grid = self._indices_grid
        if indices_grid is None:
            raise RuntimeError("LTXAdapter indices_grid cache is empty. Did prepare_latents() run?")

        # Determine how many conditional batches we need:
        # - base: text (1)
        # - + CFG: add uncond (1)
        # - + STG: add "perturbed" text pass (1) controlled by skip_layer_mask/strategy
        num_conds = 1 + (1 if do_cfg else 0) + (1 if do_stg else 0)

        if do_cfg and do_stg:   
            # [uncond, text, text_perturb]
            encoder_hidden_states = torch.cat(
                [self.negative_prompt_embeds.detach(), self.prompt_embeds.detach(), self.prompt_embeds.detach()],
                dim=0,
            )
            if (self.negative_prompt_attention_mask is not None) and (self.prompt_attention_mask is not None):
                encoder_attention_mask = torch.cat(
                    [self.negative_prompt_attention_mask, self.prompt_attention_mask, self.prompt_attention_mask],
                    dim=0,
                )
        elif do_cfg:
            # [uncond, text]
            encoder_hidden_states = torch.cat(
                [self.negative_prompt_embeds.detach(), self.prompt_embeds.detach()],
                dim=0,
            )
            if (self.negative_prompt_attention_mask is not None) and (self.prompt_attention_mask is not None):
                encoder_attention_mask = torch.cat(
                    [self.negative_prompt_attention_mask, self.prompt_attention_mask],
                    dim=0,  
                )
        elif do_stg:
            # [text, text_perturb]
            encoder_hidden_states = torch.cat(
                [self.prompt_embeds.detach(), self.prompt_embeds.detach()],
                dim=0,
            )
            if self.prompt_attention_mask is not None:
                encoder_attention_mask = torch.cat([self.prompt_attention_mask, self.prompt_attention_mask], dim=0)
        # Build transformer input batch:
        # - The transformer expects batch to match encoder_hidden_states and indices_grid.
        # - We repeat the *base* latents for [uncond, text, (optional) text_perturb] passes.
        lat_model_input = lat_base
        indices_grid_model = indices_grid
        if num_conds > 1:
            lat_model_input = torch.cat([lat_base] * num_conds, dim=0)
            indices_grid_model = torch.cat([indices_grid] * num_conds, dim=0)
            
        # Forward pass
        def _forward():
            # Match pipeline behavior: expand timestep to (batch, 1)
            timestep = t
            if not torch.is_tensor(timestep):
                timestep = torch.tensor([timestep], device=self.device())
            elif len(timestep.shape) == 0:
                timestep = timestep[None].to(self.device())
            timestep = timestep.expand(lat_model_input.shape[0]).unsqueeze(-1)

            tr = self.pipeline.transformer
            model_dtype = getattr(tr, "dtype", torch.bfloat16)

            # STG skip-layer mask (this makes the "perturbed" pass differ from the normal text pass)
            skip_layer_mask = None
            if do_stg and (self.skip_block_list is not None) and hasattr(tr, "create_skip_layer_mask"):
                try:
                    # shape: [num_layers, batch_size*num_conds]
                    skip_layer_mask = tr.create_skip_layer_mask(
                        int(lat_base.shape[0]),
                        int(num_conds),
                        int(num_conds - 1),
                        self.skip_block_list,
                    )
                except Exception:
                    skip_layer_mask = None

            kwargs: Dict[str, Any] = dict(
                hidden_states=lat_model_input.to(model_dtype),
                indices_grid=indices_grid_model,
                encoder_hidden_states=encoder_hidden_states.to(model_dtype),
                timestep=timestep,  # keep timestep as float/int (do not cast to bf16)
                return_dict=False,
            )
            if encoder_attention_mask is not None:
                kwargs["encoder_attention_mask"] = encoder_attention_mask
            if skip_layer_mask is not None:
                kwargs["skip_layer_mask"] = skip_layer_mask
            if self.skip_layer_strategy is not None:
                kwargs["skip_layer_strategy"] = self.skip_layer_strategy

            out = tr(**kwargs)
            return out[0]
        
        if with_grad:
            velocity_pred = _forward()
        else:
            with torch.no_grad():
                velocity_pred = _forward()
        if do_cfg and do_stg:
            v_uncond, v_text, v_text_perturb = velocity_pred.chunk(3, dim=0)
            # CFG
            v = v_uncond + float(self.guidance_scale) * (v_text - v_uncond)
            # STG
            v = v + float(self.stg_scale) * (v_text - v_text_perturb)
            # Rescaling (stabilizes STG)
            if do_rescaling and float(self.stg_scale) > 0.0:
                bs = v_text.shape[0]
                v_text_std = v_text.view(bs, -1).std(dim=1, keepdim=True)
                v_std = v.view(bs, -1).std(dim=1, keepdim=True)
                factor = v_text_std / (v_std + 1e-8)
                factor = float(self.rescaling_scale) * factor + (1 - float(self.rescaling_scale))
                v = v * factor.view(bs, 1, 1)
            velocity_pred = v
        elif do_cfg:
            v_uncond, v_text = velocity_pred.chunk(2, dim=0)
            velocity_pred = v_uncond + float(self.guidance_scale) * (v_text - v_uncond)
        elif do_stg:
            v_text, v_text_perturb = velocity_pred.chunk(2, dim=0)
            velocity_pred = v_text + float(self.stg_scale) * (v_text - v_text_perturb)
        
        # Scheduler step
        step_out = self.pipeline.scheduler.step(
            velocity_pred, t, lat_base, return_dict=True
        )
         
        next_latents = step_out.prev_sample
        
        # Compute x₀ (Rectified Flow formula)
        x0_latents = lat_base - t * velocity_pred
        
        return StepOutput(
            next_latents=next_latents,
            action=velocity_pred,
            x0_latents=x0_latents,
            solver_state=solver_state
        )

    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        """
        Return the list of parameters to optimize.

        Rules:
        - If LoRA/PEFT is applied to the transformer, train ONLY LoRA parameters.
        - Else, if `train_transformer_blocks` is provided, train ONLY those blocks.
        - Else, train ALL transformer parameters.
        """
        transformer = getattr(self.pipeline, "transformer", None)
        if transformer is None:
            raise RuntimeError("LTXAdapter requires pipeline.transformer")

        # A) LoRA / PEFT mode: only LoRA params
        if has_lora(transformer):
            return get_trainable_lora_parameters(transformer, verbose_prefix="  [LTX Adapter]")

        # # B) Full fine-tune: all transformer params
        # if not self.train_transformer_blocks:
        #     for p in transformer.parameters(): 
        #         p.requires_grad_(True)
        #     return [p for p in transformer.parameters() if p.requires_grad]

        # C) Partial fine-tune: selected blocks only for full blocks without LoRA.
        for p in transformer.parameters():
            p.requires_grad_(False)

        ids = set(int(x) for x in self.train_transformer_blocks)
        blocks = getattr(transformer, "transformer_blocks", None)
        if blocks is not None:
            for i, blk in enumerate(blocks):
                req = i in ids
                for p in blk.parameters():
                    p.requires_grad_(req)
            return [p for p in transformer.parameters() if p.requires_grad]

        # Fallback: name match
        for name, p in transformer.named_parameters():
            for bid in ids:
                if f"transformer_blocks.{bid}." in name:
                    p.requires_grad_(True)
        return [p for p in transformer.parameters() if p.requires_grad]
        
    def decode_for_reward(self, *, latents_or_x0: torch.Tensor, x0_is_patchified: bool) -> torch.Tensor:
        """ 
        convert the x0_estimate latents to per-frame pixel tensor [T, 3, H, W] in [0, 1]., and pass into the reward functions. 
        """ 
        pipe = self.pipeline
        vae = pipe.vae

        lat = latents_or_x0

        # Unpatchify if we're in token space: [B, N, C] -> [B, C_lat, F_lat, H_lat, W_lat]
        if bool(x0_is_patchified) or lat.dim() == 3:
            if lat.dim() != 3:
                raise ValueError(
                    f"LTXAdapter.decode_for_reward: x0_is_patchified=True but latents have shape {tuple(lat.shape)}"
                )
            if self._latent_height is None or self._latent_width is None:
                raise RuntimeError("LTXAdapter latent shape cache is empty; cannot unpatchify. Did prepare_latents() run?")

            patchifier = getattr(pipe, "patchifier", None)
            if patchifier is None or not hasattr(patchifier, "unpatchify"):
                raise RuntimeError("LTXAdapter requires pipeline.patchifier.unpatchify for decoding patchified latents.")

            tr = getattr(pipe, "transformer", None)
            if tr is None or not hasattr(tr, "in_channels"):
                raise RuntimeError("LTXAdapter requires pipeline.transformer.in_channels to infer latent channels.")

            patch_size = getattr(patchifier, "patch_size", (1, 1, 1))
            out_channels = int(getattr(tr, "in_channels")) // int(math.prod(tuple(patch_size)))

            lat = patchifier.unpatchify(
                latents=lat,
                output_height=int(self._latent_height),
                output_width=int(self._latent_width),
                out_channels=out_channels,
            )

        # At this point, we expect [B, C_lat, F_lat, H_lat, W_lat]
        if lat.dim() != 5:
            raise ValueError(f"LTXAdapter.decode_for_reward: expected 5D latents, got shape {tuple(lat.shape)}")

        # Match LTX pipeline's decode path: optional timestep conditioning with slight noise injection.
        decode_timestep = None
        if bool(getattr(getattr(vae, "decoder", None), "timestep_conditioning", False)):
            decode_timestep = torch.tensor([float(self.decode_timestep)], device=lat.device)
            noise_scale = float(self.decode_noise_scale)
            if noise_scale > 0:
                lat = lat * (1.0 - noise_scale) + torch.randn_like(lat) * noise_scale

        # Decode latents -> pixels (use per-channel normalization like the LTX pipeline).
        vae_dtype = getattr(vae, "dtype", torch.bfloat16)
        vae_device = getattr(vae, "device", self.device())

        with torch.no_grad():
            img = vae_decode(
                lat.to(device=vae_device, dtype=vae_dtype),
                vae,
                is_video=True,
                vae_per_channel_normalize=True,
                timestep=decode_timestep,
            )
            # Postprocess to [0,1] and a stable tensor layout
            if hasattr(pipe, "image_processor"):
                img = pipe.image_processor.postprocess(img, output_type="pt")

        # Normalize output to [B, T, 3, H, W]
        if not isinstance(img, torch.Tensor):
            raise RuntimeError(f"LTXAdapter.decode_for_reward: expected tensor output, got {type(img)}")

        if img.dim() == 5 and img.shape[1] == 3:
            # [B, 3, T, H, W] -> [B, T, 3, H, W]
            vid = img.permute(0, 2, 1, 3, 4).contiguous()
        elif img.dim() == 5 and img.shape[-1] == 3:
            # [B, T, H, W, 3] -> [B, T, 3, H, W]
            vid = img.permute(0, 1, 4, 2, 3).contiguous()
        else:
            raise ValueError(f"LTXAdapter.decode_for_reward: unexpected decoded tensor shape {tuple(img.shape)}")

        # Remove batch and trim/pad to requested num_frames.
        vid = vid[0]  # [T, 3, H, W]
        req_t = int(self.num_frames)
        if vid.size(0) > req_t:
            vid = vid[:req_t]
        elif vid.size(0) < req_t and vid.size(0) > 0:
            last = vid[-1:].expand(req_t - vid.size(0), -1, -1, -1)
            vid = torch.cat([vid, last], dim=0)

        # Ensure float in [0,1]
        vid = vid.float()
        if float(vid.min()) < -0.1:
            # Defensive: some processors return [-1,1]
            vid = (vid + 1.0) / 2.0
        return vid.clamp(0.0, 1.0)