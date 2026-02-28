# from __future__ import annotations

# from dataclasses import dataclass
# import random
# from typing import Tuple
# from typing import Any, Dict, List, Optional

# import torch
# import torch.nn.functional as F

# from unified_grpo.adapters.base import StepContext, StepOutput, VideoGRPOAdapter

# from ltx_video.models.autoencoders.vae_encode import latent_to_pixel_coords, vae_decode  # type: ignore


# @dataclass
# class LTXAdapter(VideoGRPOAdapter):
#     """
#     Adapter for LTX-Video (Lightricks)
    
#     Prediction type: v-prediction (Rectified Flow)
#     Formula: x₀ = x_t - t * v
#     """
    
#     pipeline: Any
#     prompt_embeds: torch.Tensor
#     negative_prompt_embeds: Optional[torch.Tensor] = None
#     prompt_attention_mask: Optional[torch.Tensor] = None
#     negative_prompt_attention_mask: Optional[torch.Tensor] = None
#     guidance_scale: float = 4.5
#     # LTX "STG" (spatiotemporal guidance) settings (match ltx_video pipeline defaults/config)
#     stg_scale: float = 1.0
#     rescaling_scale: float = 0.7
#     cfg_star_rescale: bool = False
#     skip_layer_strategy: Optional[Any] = None
#     skip_block_list: Optional[List[int]] = None

#     # VAE decode settings (match pipeline config defaults)
#     decode_timestep: float = 0.05
#     decode_noise_scale: float = 0.025
    
#     height: int = 512
#     width: int = 768
#     num_frames: int = 25
    
#     train_transformer_blocks: Optional[List[int]] = None
    
#     name: str = "ltx"

#     # Cached coordinates for the current latent shape (used by LTX's RoPE / indices_grid)
#     _indices_grid: Optional[torch.Tensor] = None
#     _latent_height: Optional[int] = None
#     _latent_width: Optional[int] = None
#     _latent_frames: Optional[int] = None
#     _frame_rate: float = 30.0

#     # Requested output shape (what user asked for). LTX may require internal adjustment
#     # (e.g. width divisible by vae_scale_factor, frames = 8k+1). We generate internally at the
#     # adjusted shape, then pad/crop back to requested for reward + saved mp4.
#     _req_height: Optional[int] = None
#     _req_width: Optional[int] = None
#     _req_num_frames: Optional[int] = None
#     _gen_height: Optional[int] = None
#     _gen_width: Optional[int] = None
#     _gen_num_frames: Optional[int] = None
    
#     def device(self) -> torch.device:
#         dev = getattr(self.pipeline, "_execution_device", None)
#         if dev is None:
#             dev = getattr(self.pipeline, "device", None)
#         return torch.device(dev) if dev is not None else torch.device("cuda")

#     def apply_rollout_diversity(
#         self,
#         *,
#         rng: random.Random,
#         stg_scale_range: Optional[Tuple[float, float]] = None,
#         rescaling_scale_range: Optional[Tuple[float, float]] = None,
#         cfg_star_rescale_prob: Optional[float] = None,
#     ) -> str:
#         """
#         LTX-only rollout diversity hook.

#         This lets GRPO core stay model-agnostic: it can call this method if present,
#         without knowing about LTX's STG/rescaling/cfg_star_rescale knobs.

#         Returns:
#           - info string (for logging)

#         Note:
#           This function intentionally applies changes *in-place* and does NOT restore
#           previous values. This maximizes diversity but means the final rollout's
#           values will persist into subsequent steps unless you reset them elsewhere.
#         """
#         info_parts: List[str] = []

#         if stg_scale_range is not None:
#             lo, hi = float(stg_scale_range[0]), float(stg_scale_range[1])
#             self.stg_scale = float(rng.uniform(lo, hi))
#             info_parts.append(f"stg_scale={self.stg_scale:.2f}")

#         if rescaling_scale_range is not None:
#             lo, hi = float(rescaling_scale_range[0]), float(rescaling_scale_range[1])
#             self.rescaling_scale = float(rng.uniform(lo, hi))
#             info_parts.append(f"rescaling_scale={self.rescaling_scale:.2f}")

#         if cfg_star_rescale_prob is not None:
#             p = max(0.0, min(1.0, float(cfg_star_rescale_prob)))
#             self.cfg_star_rescale = bool(rng.random() < p)
#             info_parts.append(f"cfg_star_rescale={int(self.cfg_star_rescale)}")

#         info = (" " + " ".join(info_parts)) if info_parts else ""
#         return info
    
#     def get_timesteps(self, *, num_inference_steps: int) -> list[torch.Tensor]:
#         self.pipeline.scheduler.set_timesteps(num_inference_steps, device=self.device())
#         return [t for t in self.pipeline.scheduler.timesteps]
    
#     def prepare_latents(self, *, seed: int) -> torch.Tensor:
#         g = torch.Generator(device=self.device()).manual_seed(seed)
        
#         # LTX latent dimensions must match the pipeline's VAE scale factors.
#         # (Using fixed numbers can explode token counts and VRAM.)
#         vae_scale = int(getattr(self.pipeline, "vae_scale_factor", 32))
#         video_scale = int(getattr(self.pipeline, "video_scale_factor", 8))

#         latent_c = int(getattr(self.pipeline.vae.config, "latent_channels", 128))
        
#         # Remember what the user asked for.
#         self._req_height = int(self.height)
#         self._req_width = int(self.width)
#         self._req_num_frames = int(self.num_frames)

#         # Internal generation shape must satisfy LTX constraints.
#         # - Spatial: divisible by vae_scale_factor (32)
#         # - Temporal: for video, LTX uses a temporal scale factor of 8 and expects frames of the form 8k+1
#         gen_h = int(self.height) - (int(self.height) % vae_scale)
#         gen_w = int(self.width) - (int(self.width) % vae_scale)
#         if gen_h <= 0 or gen_w <= 0:
#             raise ValueError(f"Invalid (height,width)=({self.height},{self.width}) for vae_scale_factor={vae_scale}")

#         # Match pipeline trimming rule: (num_frames - 1) // scale_factor * scale_factor + 1
#         req_f = int(self.num_frames)
#         gen_f = (max(1, req_f) - 1) // video_scale * video_scale + 1

#         if gen_h != int(self.height) or gen_w != int(self.width) or gen_f != int(self.num_frames):
#             print(
#                 f"⚠️  LTX internal shape adjustment: requested {self.width}×{self.height}, {self.num_frames} frames "
#                 f"→ generating {gen_w}×{gen_h}, {gen_f} frames (will pad/crop back for reward/output)."
#             )

#         self._gen_height = gen_h
#         self._gen_width = gen_w
#         self._gen_num_frames = gen_f

#         latent_h = gen_h // vae_scale
#         latent_w = gen_w // vae_scale
#         latent_f = gen_f // video_scale
#         # +1 for causal VAE (matches pipeline behavior for video)
#         if getattr(self.pipeline.vae, "is_video_supported", False):
#             latent_f += 1

#         model_dtype = getattr(self.pipeline.transformer, "dtype", torch.bfloat16)
        
#         latents_5d = torch.randn(
#             (1, latent_c, latent_f, latent_h, latent_w),
#             generator=g,
#             device=self.device(),
#             dtype=model_dtype
#         )

#         latents_5d = latents_5d * self.pipeline.scheduler.init_noise_sigma

#         # Patchify into (b, n, c) tokens + coords, as expected by the Transformer3DModel.
#         latents, latent_coords = self.pipeline.patchifier.patchify(latents_5d)

#         # Cache indices_grid (pixel coords scaled by 1/frame_rate) for transformer calls.
#         pixel_coords = latent_to_pixel_coords(latent_coords, self.pipeline.vae, causal_fix=True)
#         indices_grid = pixel_coords.to(torch.float32)
#         indices_grid[:, 0] = indices_grid[:, 0] * (1.0 / float(self._frame_rate))

#         self._indices_grid = indices_grid
#         self._latent_height = latent_h
#         self._latent_width = latent_w
#         self._latent_frames = latent_f

#         return latents

#     @staticmethod
#     def _pad_or_center_crop_hw(video_btc_hw: torch.Tensor, *, target_h: int, target_w: int) -> torch.Tensor:
#         """
#         video_btc_hw: [B, T, C, H, W]
#         If H/W smaller -> pad (replicate edges). If larger -> center crop.
#         """
#         b, t, c, h, w = video_btc_hw.shape

#         # Center crop if needed
#         if h > target_h:
#             top = (h - target_h) // 2
#             video_btc_hw = video_btc_hw[:, :, :, top : top + target_h, :]
#             h = target_h
#         if w > target_w:
#             left = (w - target_w) // 2
#             video_btc_hw = video_btc_hw[:, :, :, :, left : left + target_w]
#             w = target_w

#         # Replicate-pad if needed (F.pad works on NCHW, so flatten BT)
#         pad_h = max(0, target_h - h)
#         pad_w = max(0, target_w - w)
#         if pad_h > 0 or pad_w > 0:
#             pad_top = pad_h // 2
#             pad_bottom = pad_h - pad_top
#             pad_left = pad_w // 2
#             pad_right = pad_w - pad_left
#             x = video_btc_hw.reshape(b * t, c, h, w)
#             x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode="replicate")
#             video_btc_hw = x.reshape(b, t, c, target_h, target_w)

#         return video_btc_hw
    
#     def trainable_parameters(self) -> list[torch.nn.Parameter]:
#         tr = self.pipeline.transformer

#         # If LoRA/PEFT is applied, only train LoRA parameters (do NOT unfreeze the base model).
#         is_lora = hasattr(tr, "peft_config") or any("lora_" in n.lower() for n, _ in tr.named_parameters())
#         if is_lora:
#             lora_params = [p for n, p in tr.named_parameters() if "lora_" in n.lower() and p.requires_grad]
#             if len(lora_params) == 0:
#                 # PEFT should already set requires_grad on LoRA params, but be defensive.
#                 for n, p in tr.named_parameters():
#                     if "lora_" in n.lower():
#                         p.requires_grad_(True)
#                         lora_params.append(p)
#             return lora_params
        
#         if not self.train_transformer_blocks:
#             # Train all
#             for p in tr.parameters():
#                 p.requires_grad_(True)
#             return [p for p in tr.parameters() if p.requires_grad]
        
#         # Freeze all first
#         for p in tr.parameters():
#             p.requires_grad_(False)
        
#         # Unfreeze specified blocks
#         ids = set(self.train_transformer_blocks)
#         blocks = tr.transformer_blocks
        
#         for i, blk in enumerate(blocks):
#             if i in ids:
#                 for p in blk.parameters():
#                     p.requires_grad_(True)
        
#         return [p for p in tr.parameters() if p.requires_grad]
    
#     def step(
#         self,
#         *,
#         latents: torch.Tensor,
#         step_context: StepContext,
#         with_grad: bool,
#         solver_state=None,
#     ) -> StepOutput:
#         t = step_context.t
        
#         # Add noise for exploration
#         lat_base = latents.detach() if not with_grad else latents
#         lat_in = lat_base
        
#         # CFG + STG setup (mirror LTXVideoPipeline / origin_grpo logic)
#         do_cfg = self.guidance_scale > 1.0 and self.negative_prompt_embeds is not None
#         do_stg = float(self.stg_scale) > 0.0
#         do_rescaling = float(self.rescaling_scale) != 1.0

#         # Prompt embeddings must be detached leaf tensors (we don't train the text encoder).
#         encoder_hidden_states = self.prompt_embeds.detach()
#         encoder_attention_mask = self.prompt_attention_mask
#         indices_grid = self._indices_grid
#         if indices_grid is None:
#             raise RuntimeError("LTXAdapter indices_grid cache is empty. Did prepare_latents() run?")

#         # Determine how many conditional batches we need:
#         # - base: text (1)
#         # - + CFG: add uncond (1)
#         # - + STG: add "perturbed" text pass (1) controlled by skip_layer_mask/strategy
#         num_conds = 1 + (1 if do_cfg else 0) + (1 if do_stg else 0)

#         if do_cfg and do_stg:
#             # [uncond, text, text_perturb]
#             encoder_hidden_states = torch.cat(
#                 [self.negative_prompt_embeds.detach(), self.prompt_embeds.detach(), self.prompt_embeds.detach()],
#                 dim=0,
#             )
#             if (self.negative_prompt_attention_mask is not None) and (self.prompt_attention_mask is not None):
#                 encoder_attention_mask = torch.cat(
#                     [self.negative_prompt_attention_mask, self.prompt_attention_mask, self.prompt_attention_mask],
#                     dim=0,
#                 )
#         elif do_cfg:
#             # [uncond, text]
#             encoder_hidden_states = torch.cat(
#                 [self.negative_prompt_embeds.detach(), self.prompt_embeds.detach()],
#                 dim=0,
#             )
#             if (self.negative_prompt_attention_mask is not None) and (self.prompt_attention_mask is not None):
#                 encoder_attention_mask = torch.cat(
#                     [self.negative_prompt_attention_mask, self.prompt_attention_mask],
#                     dim=0,
#                 )
#         elif do_stg:
#             # [text, text_perturb]
#             encoder_hidden_states = torch.cat(
#                 [self.prompt_embeds.detach(), self.prompt_embeds.detach()],
#                 dim=0,
#             )
#             if self.prompt_attention_mask is not None:
#                 encoder_attention_mask = torch.cat([self.prompt_attention_mask, self.prompt_attention_mask], dim=0)

#         if num_conds > 1:
#             lat_in = torch.cat([lat_in] * num_conds, dim=0)
#             indices_grid = torch.cat([indices_grid] * num_conds, dim=0)
        
#         # Forward pass
#         def _forward():
#             # Match pipeline behavior: expand timestep to (batch, 1)
#             timestep = t
#             if not torch.is_tensor(timestep):
#                 timestep = torch.tensor([timestep], device=self.device())
#             elif len(timestep.shape) == 0:
#                 timestep = timestep[None].to(self.device())
#             timestep = timestep.expand(lat_in.shape[0]).unsqueeze(-1)

#             tr = self.pipeline.transformer
#             model_dtype = getattr(tr, "dtype", torch.bfloat16)

#             # STG skip-layer mask (this makes the "perturbed" pass differ from the normal text pass)
#             skip_layer_mask = None
#             if do_stg and (self.skip_block_list is not None) and hasattr(tr, "create_skip_layer_mask"):
#                 try:
#                     skip_layer_mask = tr.create_skip_layer_mask(1, num_conds, num_conds - 1, self.skip_block_list)
#                 except Exception:
#                     skip_layer_mask = None

#             kwargs: Dict[str, Any] = dict(
#                 hidden_states=lat_in.to(model_dtype),
#                 indices_grid=indices_grid,
#                 encoder_hidden_states=encoder_hidden_states.to(model_dtype),
#                 timestep=timestep,  # keep timestep as float/int (do not cast to bf16)
#                 return_dict=False,
#             )
#             if encoder_attention_mask is not None:
#                 kwargs["encoder_attention_mask"] = encoder_attention_mask
#             if skip_layer_mask is not None:
#                 kwargs["skip_layer_mask"] = skip_layer_mask
#             if self.skip_layer_strategy is not None:
#                 kwargs["skip_layer_strategy"] = self.skip_layer_strategy

#             out = tr(**kwargs)
#             return out[0]
        
#         if with_grad:
#             velocity_pred = _forward()
#         else:
#             with torch.no_grad():
#                 velocity_pred = _forward()
        
#         # Guidance composition (match pipeline/origin_grpo)
#         if do_cfg and do_stg:
#             v_uncond, v_text, v_text_perturb = velocity_pred.chunk(3, dim=0)
#             # CFG
#             v = v_uncond + float(self.guidance_scale) * (v_text - v_uncond)
#             # STG
#             v = v + float(self.stg_scale) * (v_text - v_text_perturb)
#             # Rescaling (stabilizes STG)
#             if do_rescaling and float(self.stg_scale) > 0.0:
#                 bs = v_text.shape[0]
#                 v_text_std = v_text.view(bs, -1).std(dim=1, keepdim=True)
#                 v_std = v.view(bs, -1).std(dim=1, keepdim=True)
#                 factor = v_text_std / (v_std + 1e-8)
#                 factor = float(self.rescaling_scale) * factor + (1 - float(self.rescaling_scale))
#                 v = v * factor.view(bs, 1, 1)
#             velocity_pred = v
#         elif do_cfg:
#             v_uncond, v_text = velocity_pred.chunk(2, dim=0)
#             velocity_pred = v_uncond + float(self.guidance_scale) * (v_text - v_uncond)
#         elif do_stg:
#             v_text, v_text_perturb = velocity_pred.chunk(2, dim=0)
#             velocity_pred = v_text + float(self.stg_scale) * (v_text - v_text_perturb)
        
#         # Scheduler step
#         step_out = self.pipeline.scheduler.step(
#             velocity_pred, t, lat_base, return_dict=True
#         )
        
#         next_latents = step_out.prev_sample
        
#         # Compute x₀ (Rectified Flow formula)
#         x0_latents = lat_base - t * velocity_pred
        
#         return StepOutput(
#             next_latents=next_latents,
#             action=velocity_pred,
#             x0_latents=x0_latents,
#             solver_state=solver_state
#         )
    
#     def decode_for_reward(self, *, latents_or_x0: torch.Tensor, x0_is_patchified: bool) -> torch.Tensor:
#         """
#         Decode latents to video in [0,1], matching the baseline LTX pipeline / origin_grpo:
#         - use `vae_decode(...)`
#         - then `pipeline.image_processor.postprocess(..., output_type="pt")`
#         Returns: [B, T, C, H, W]
#         """
#         pipe = self.pipeline
#         vae = pipe.vae
#         latent_c = int(getattr(getattr(vae, "config", None), "latent_channels", 128))
#         vae_dtype = getattr(vae, "dtype", torch.bfloat16)

#         # IMPORTANT:
#         # We decode using `vae_decode(..., vae_per_channel_normalize=True)`, which internally
#         # bypasses `vae.config.scaling_factor` and instead uses the VAE's per-channel normalization.
#         # Therefore we must NOT pre-divide by scaling_factor here (doing so washes out the signal and looks blurry).
#         # This matches `origin_grpo/helper.py::decode_x0_to_video`.
#         lat = latents_or_x0

#         # Unpatchify if we're in token space (b, n, c)
#         if lat.dim() == 3:
#             if self._latent_height is None or self._latent_width is None:
#                 raise RuntimeError("LTXAdapter latent shape cache is empty; cannot unpatchify.")
#             lat = pipe.patchifier.unpatchify(
#                 lat,
#                 output_height=int(self._latent_height) * int(pipe.patchifier.patch_size[1]),
#                 output_width=int(self._latent_width) * int(pipe.patchifier.patch_size[2]),
#                 out_channels=latent_c,
#             )

#         decode_timestep = None
#         if getattr(getattr(vae, "decoder", None), "timestep_conditioning", False):
#             # Match LTXVideoPipeline decode path:
#             # - pass decode_timestep (often 0.05)
#             # - optionally add a tiny amount of noise before decoding (decode_noise_scale)
#             decode_timestep = torch.tensor([float(self.decode_timestep)], device=lat.device)
#             noise_scale = float(self.decode_noise_scale)
#             if noise_scale > 0:
#                 noise = torch.randn_like(lat)
#                 lat = lat * (1.0 - noise_scale) + noise * noise_scale

#         with torch.no_grad():
#             img = vae_decode(
#                 lat.to(device=getattr(vae, "device", self.device()), dtype=vae_dtype),
#                 vae,
#                 is_video=True,
#                 vae_per_channel_normalize=True,
#                 timestep=decode_timestep,
#             )
#             if hasattr(pipe, "image_processor"):
#                 img = pipe.image_processor.postprocess(img, output_type="pt")
#             vid = img.permute(0, 2, 1, 3, 4).contiguous()

#         # Trim/pad back to requested output shape for reward and final mp4.
#         req_h = int(self._req_height) if self._req_height is not None else int(self.height)
#         req_w = int(self._req_width) if self._req_width is not None else int(self.width)
#         req_t = int(self._req_num_frames) if self._req_num_frames is not None else int(self.num_frames)

#         # Temporal: LTX often generates 8k+1 frames; user may ask for 32. Trim to requested.
#         if vid.size(1) > req_t:
#             vid = vid[:, :req_t]
#         elif vid.size(1) < req_t:
#             # Rare, but be defensive: replicate last frame
#             last = vid[:, -1:].expand(-1, req_t - vid.size(1), -1, -1, -1)
#             vid = torch.cat([vid, last], dim=1)

#         # Spatial: pad/crop to requested
#         vid = self._pad_or_center_crop_hw(vid, target_h=req_h, target_w=req_w)
#         return vid
    
#     def extra_log_state(self) -> Dict[str, Any]:
#         return {
#             "height": self.height,
#             "width": self.width,
#             "num_frames": self.num_frames,
#             "guidance_scale": self.guidance_scale,
#         }
