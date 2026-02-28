import torch
from pathlib import Path

def create_cogvideox_adapter(args):
    """Create CogVideoX adapter"""
    from unified_grpo.adapters.cogvideox import CogVideoXAdapter
    from unified_grpo.lora_utils import apply_lora_to_transformer
    from unified_grpo.utils import resolve_lora_blocks
    from diffusers.pipelines.cogvideo.pipeline_cogvideox import CogVideoXPipeline
    
    print(f"Loading CogVideoX pipeline: {args.model_path}")
    
    pipeline = CogVideoXPipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
    ).to("cuda")

    # Optional memory optimizations (helpful for avoiding OOM during training)
    transformer = getattr(pipeline, "transformer", None)
    if transformer is None:
        raise RuntimeError("CogVideoX pipeline has no .transformer")

    # Enable gradient checkpointing on the denoiser/transformer (saves VRAM).
    transformer.enable_gradient_checkpointing()
    # IMPORTANT: For CogVideoX, `enable_xformers_memory_efficient_attention()` can swap the model's custom
    # CogVideoX attention processor for a generic xFormers joint processor, which breaks tensor shapes
    # (e.g. "size of tensor a (226) must match ... (10800)").
    # So we intentionally DO NOT enable xFormers for CogVideoX here.

    # Ensure all components use consistent dtype
    pipeline.transformer = transformer.to(torch.bfloat16)
    pipeline.vae = pipeline.vae.to(torch.bfloat16)
    pipeline.text_encoder = pipeline.text_encoder.to(torch.bfloat16)

    # Important for stable sampling/training: disable dropout-like noise.
    # `.eval()` does NOT disable gradients; it only changes module behavior (e.g. dropout/bn).
    pipeline.transformer.eval()
    pipeline.vae.eval()
    pipeline.text_encoder.eval()
    
    # Apply LoRA if requested (recommended for 40GB GPU!)
    if bool(getattr(args, "use_lora", False)):
        # Some PEFT-wrapped models expose `get_base_model()`. We use it only for block counting.
        base_transformer = transformer.get_base_model() if hasattr(transformer, "get_base_model") else transformer
        blocks = getattr(base_transformer, "transformer_blocks", None)
        try:
            total_blocks = len(blocks) if blocks is not None else None  # ModuleList is len()-able
        except Exception:
            total_blocks = None

        # LoRA default: ALL blocks (unless user sets --lora-blocks).
        lora_blocks = resolve_lora_blocks(
            spec=getattr(args, "lora_blocks", None),
            total_blocks=total_blocks,
            unfreeze_pct=float(getattr(args, "unfreeze_percentage", 0.20)),
        )
        
        pipeline.transformer, _ = apply_lora_to_transformer(
            transformer,
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            target_blocks=lora_blocks,
        )
        if lora_blocks:
            print(f"✅ LoRA applied to blocks {lora_blocks} (ultra memory-efficient!)")
        else:
            print("✅ LoRA applied to ALL blocks (memory-efficient training!)")
    else:
        print("⚠️  Training unfrozen blocks (needs 48GB+ VRAM)")
    
    # Memory optimizations
    # Memory optimizations (best-effort; method names differ between diffusers VAEs and LTX's VAE wrapper)
    vae = pipeline.vae
    if hasattr(vae, "enable_slicing"):
        vae.enable_slicing()
    if hasattr(vae, "enable_tiling"):
        vae.enable_tiling()
    # NOTE: LTX's VAE tiling path currently doesn't thread `timestep` through to the decoder
    # when `timestep_conditioning=True`, which can assert during decode. Keep HW tiling off.
    if hasattr(vae, "enable_z_tiling"):
        vae.enable_z_tiling()
    
    print("✅ Memory optimizations enabled")
    
    # Encode prompt WITHOUT tracking gradients (we never train the text encoder here).
    with torch.no_grad():
        prompt_embeds, negative_embeds = pipeline.encode_prompt(
            prompt=str(getattr(args, "prompt", "") or ""),
            negative_prompt=str(getattr(args, "negative_prompt", "") or ""),
            do_classifier_free_guidance=True,
            device="cuda",
        )
    prompt_embeds = prompt_embeds.detach()
    negative_embeds = negative_embeds.detach()
    
    # Parse train blocks
    train_blocks = None
    if args.train_blocks:
        train_blocks = [int(x.strip()) for x in args.train_blocks.split(",")]
    
    adapter = CogVideoXAdapter(
        pipeline=pipeline,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_embeds,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        train_transformer_blocks=train_blocks,
    )
    
    return adapter


def create_ltx_adapter(args):
    """
    Create LTX-Video adapter.

    Why this looks different from diffusers pipelines:
    - LTX-Video is vendored under `ltx_video/` in this repo.
    - For GRPO we build the pipeline via `ltx_video.ltx_video.inference.create_ltx_video_pipeline`,
      which is the supported construction path for this vendored implementation.
    - Checkpoints can be provided as a local file path or a HuggingFace repo id.
    """
    import os

    from huggingface_hub import hf_hub_download  # type: ignore

    from unified_grpo.adapters.ltx import LTXAdapter
    from unified_grpo.lora_utils import apply_lora_to_transformer
    from unified_grpo.utils import resolve_lora_blocks

    # Import the real implementation module. (Avoid top-level `ltx_video/inference.py` wrapper.)
    from ltx_video.ltx_video.inference import create_ltx_video_pipeline, load_pipeline_config  # type: ignore

    repo_root = Path(__file__).resolve().parent.parent  # .../angel-research

    # Default GRPO-aligned pipeline config.
    pipeline_cfg_path = str(repo_root / "ltx_video" / "configs" / "ltxv-2b-0.9.6-dev.yaml")
    pipeline_config = load_pipeline_config(pipeline_cfg_path)

    ckpt_name_or_path = str(pipeline_config["checkpoint_path"])

    # `--model-path` can be:
    # - local checkpoint file, OR
    # - HuggingFace repo id (default: "Lightricks/LTX-Video")
    model_path = getattr(args, "model_path", None) or "Lightricks/LTX-Video"
    if isinstance(model_path, str) and os.path.isfile(model_path):
        ckpt_path = model_path
    else:
        ckpt_path = hf_hub_download(
            repo_id=str(model_path),
            filename=ckpt_name_or_path,
            repo_type="model",
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading LTX-Video checkpoint: {ckpt_path}")

    pipeline = create_ltx_video_pipeline(
        ckpt_path=str(ckpt_path),
        precision=str(pipeline_config.get("precision", "bfloat16")),
        text_encoder_model_name_or_path=str(pipeline_config["text_encoder_model_name_or_path"]),
        sampler=pipeline_config.get("sampler", None),
        device=str(device),
        enhance_prompt=False,
    )

    # LTX guidance / decode knobs (from config; matches origin_grpo defaults).
    stg_scale_cfg = float(pipeline_config.get("stg_scale", 0.0))
    rescaling_scale_cfg = float(pipeline_config.get("rescaling_scale", 1.0))
    cfg_star_rescale_cfg = bool(pipeline_config.get("cfg_star_rescale", False))
    decode_timestep_cfg = float(pipeline_config.get("decode_timestep", 0.05))
    decode_noise_scale_cfg = float(pipeline_config.get("decode_noise_scale", 0.025))

    skip_block_list_cfg = pipeline_config.get("skip_block_list", None)
    if isinstance(skip_block_list_cfg, list):
        skip_block_list_cfg = [int(x) for x in skip_block_list_cfg]
    else:
        skip_block_list_cfg = None

    # Map stg_mode -> SkipLayerStrategy enum
    try:
        from ltx_video.utils.skip_layer_strategy import SkipLayerStrategy  # type: ignore
    except Exception:
        from ltx_video.ltx_video.utils.skip_layer_strategy import SkipLayerStrategy  # type: ignore

    stg_mode = str(pipeline_config.get("stg_mode", "attention_values")).lower()
    if stg_mode in ("stg_av", "attention_values"):
        skip_layer_strategy_cfg = SkipLayerStrategy.AttentionValues
    elif stg_mode in ("stg_as", "attention_skip"):
        skip_layer_strategy_cfg = SkipLayerStrategy.AttentionSkip
    elif stg_mode in ("stg_r", "residual"):
        skip_layer_strategy_cfg = SkipLayerStrategy.Residual
    elif stg_mode in ("stg_t", "transformer_block"):
        skip_layer_strategy_cfg = SkipLayerStrategy.TransformerBlock
    else:
        skip_layer_strategy_cfg = SkipLayerStrategy.AttentionValues

    # Optional LoRA on the LTX transformer.
    if bool(getattr(args, "use_lora", False)):
        blocks = getattr(getattr(pipeline, "transformer", None), "transformer_blocks", None)
        try:
            total_blocks = len(blocks) if blocks is not None else None
        except Exception:
            total_blocks = None
        lora_blocks = resolve_lora_blocks(
            spec=getattr(args, "lora_blocks", None),
            total_blocks=total_blocks,
            unfreeze_pct=float(getattr(args, "unfreeze_percentage", 0.20)),
        )

        pipeline.transformer, _ = apply_lora_to_transformer(
            pipeline.transformer,
            rank=int(getattr(args, "lora_rank", 16)),
            alpha=int(getattr(args, "lora_alpha", 32)),
            target_blocks=lora_blocks,
        )

    # Optional dtype cast (reduces VRAM). Default remains bf16 for stability.
    if str(getattr(args, "compute_dtype", "bf16")).lower() == "fp16" and device.type == "cuda":
        pipeline.transformer = pipeline.transformer.to(dtype=torch.float16)
        pipeline.vae = pipeline.vae.to(dtype=torch.float16)
        pipeline.text_encoder = pipeline.text_encoder.to(dtype=torch.float16)

    # Memory optimizations (best-effort; LTX VAE methods vary by build).
    vae = pipeline.vae
    if hasattr(vae, "enable_slicing"):
        vae.enable_slicing()
    if hasattr(vae, "enable_tiling"):
        vae.enable_tiling()
    if hasattr(vae, "enable_z_tiling") and hasattr(getattr(vae, "encoder", None), "patch_size_t"):
        vae.enable_z_tiling()

    # Freeze text encoder + VAE; we only train transformer weights / LoRA.
    for mod_name in ["text_encoder", "vae"]:
        mod = getattr(pipeline, mod_name, None)
        if mod is None:
            continue
        try:
            for p in mod.parameters():
                p.requires_grad_(False)
        except Exception:
            pass

    # Encode prompt WITHOUT tracking gradients (avoid graph reuse across GRPO steps).
    with torch.no_grad():
        prompt_embeds, prompt_attention_mask, negative_embeds, negative_attention_mask = pipeline.encode_prompt(
            prompt=str(getattr(args, "prompt", "") or ""),
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=str(getattr(args, "negative_prompt", "") or ""),
        )

    # Ensure embeddings are leaf tensors (no autograd history).
    prompt_embeds = prompt_embeds.detach()
    negative_embeds = negative_embeds.detach()

    train_blocks = None
    if getattr(args, "train_blocks", None):
        train_blocks = [int(x.strip()) for x in str(args.train_blocks).split(",") if x.strip()]

    adapter = LTXAdapter(
        pipeline=pipeline,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_embeds,
        prompt_attention_mask=prompt_attention_mask,
        negative_prompt_attention_mask=negative_attention_mask,
        guidance_scale=float(getattr(args, "guidance_scale", 7.5)),
        stg_scale=stg_scale_cfg,
        rescaling_scale=rescaling_scale_cfg,
        cfg_star_rescale=cfg_star_rescale_cfg,
        skip_layer_strategy=skip_layer_strategy_cfg,
        skip_block_list=skip_block_list_cfg,
        decode_timestep=decode_timestep_cfg,
        decode_noise_scale=decode_noise_scale_cfg,
        height=int(getattr(args, "height", 480)),
        width=int(getattr(args, "width", 720)),
        num_frames=int(getattr(args, "num_frames", 32)),
        train_transformer_blocks=train_blocks,
    )

    return adapter