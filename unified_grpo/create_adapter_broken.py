import argparse
import torch
import sys
from pathlib import Path
from datetime import datetime

def create_cogvideox_adapter(args):
    """Create CogVideoX adapter"""
    # Use the active adapter implementation.
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
    #
    transformer = getattr(pipeline, "transformer", None)
    if transformer is None:
        raise RuntimeError("CogVideoX pipeline has no .transformer")
    transformer.enable_gradient_checkpointing()
    # IMPORTANT: For CogVideoX, `enable_xformers_memory_efficient_attention()` can swap the model's custom
    # CogVideoX attention processor for a generic xFormers joint processor, which breaks tensor shapes
    # (e.g. "size of tensor a (226) must match ... (10800)").
    # So we intentionally DO NOT enable xFormers for CogVideoX here.

    # Ensure all components use consistent dtype
    pipeline.transformer = pipeline.transformer.to(torch.bfloat16)
    pipeline.vae = pipeline.vae.to(torch.bfloat16)
    pipeline.text_encoder = pipeline.text_encoder.to(torch.bfloat16)

    # Important for stable sampling/training: disable dropout-like noise.
    # `.eval()` does NOT disable gradients; it only changes module behavior (e.g. dropout/bn).
    pipeline.transformer.eval()
    pipeline.vae.eval()
    pipeline.text_encoder.eval()
    
    # Apply LoRA if requested (recommended for 40GB GPU!)
    if bool(getattr(args, "use_lora", False)):
        base_transformer = transformer.get_base_model() if hasattr(transformer, "get_base_model") else transformer
        blocks = getattr(base_transformer, "transformer_blocks", None)
        try:
            total_blocks = len(blocks) if blocks is not None else None
        except Exception:
            total_blocks = None

        # LoRA default: ALL blocks (unless user sets --lora-blocks).
        lora_blocks = resolve_lora_blocks(
            spec=getattr(args, "lora_blocks", None),
            total_blocks=total_blocks,
            unfreeze_pct=float(getattr(args, "unfreeze_percentage", 0.20)),
        )
        
        pipeline.transformer, lora_params = apply_lora_to_transformer(
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
    
    # Encode prompt
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
        guidance_scale=args.guidance_scale,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        train_transformer_blocks=train_blocks,
    )
    
    return adapter


def create_ltx_adapter(args):
    """Create LTX-Video adapter (vendored `ltx_video/` pipeline)."""
    from unified_grpo.lora_utils import apply_lora_to_transformer
    from unified_grpo.utils import resolve_lora_blocks
    # NOTE:
    # - `diffusers` may or may not expose an LTX pipeline, depending on version.
    # - This repo vendors the official LTX-Video implementation under `ltx_video/`.
    # - Importing `LTXVideoPipeline` (local) and calling `.from_pretrained()` against a HF repo id
    #   can fail with a component-mismatch error (the checkpoint only provides a subset of
    #   components, while the local pipeline expects extra helper modules like patchifier +
    #   prompt enhancer stubs).
    #
    # For GRPO we only need the core pipeline components (transformer/vae/scheduler/text encoder/tokenizer),
    # so we build the pipeline using `ltx_video.inference.create_ltx_video_pipeline`, which is the
    # supported construction path for this vendored implementation.
    import os

    from huggingface_hub import hf_hub_download  # type: ignore

    # Import the real implementation module. (Avoid `ltx_video/inference.py` wrapper which can
    # be picked up as `ltx_video.inference` via namespace-package resolution from repo root.)
    from ltx_video.ltx_video.inference import (  # type: ignore
        create_ltx_video_pipeline,
        load_pipeline_config,
    )
    # Use the active LTX adapter implementation.
    from unified_grpo.adapters.ltx import LTXAdapter
    
    repo_root = Path(__file__).resolve().parent.parent  # .../angel-research
    # Default LTX pipeline config for GRPO runs.
    # User-requested default: 2B 0.9.6 dev checkpoint.
    default_cfg = repo_root / "ltx_video" / "configs" / "ltxv-2b-0.9.6-dev.yaml"
    pipeline_cfg_path = str(default_cfg)

    pipeline_config = load_pipeline_config(pipeline_cfg_path)
    ckpt_name_or_path = pipeline_config["checkpoint_path"]

    # `--model-path` can be:
    # - a local safetensors checkpoint file, OR
    # - a HuggingFace repo id (default: "Lightricks/LTX-Video")
    ckpt_path: str
    model_path = getattr(args, "model_path", None) or "Lightricks/LTX-Video"
    if isinstance(model_path, str) and os.path.isfile(model_path):
        ckpt_path = model_path
    else:
        ckpt_path = hf_hub_download(
            repo_id=str(model_path),
            filename=str(ckpt_name_or_path),
            repo_type="model",
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading LTX-Video checkpoint: {ckpt_path}")

    pipeline = create_ltx_video_pipeline(
        ckpt_path=ckpt_path,
        precision=pipeline_config["precision"],
        text_encoder_model_name_or_path=pipeline_config["text_encoder_model_name_or_path"],
        sampler=pipeline_config.get("sampler", None),
        device=device,
        enhance_prompt=False,
    )

    # Pull LTX pipeline-quality knobs from config (matches origin_grpo defaults)
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

    # Apply LoRA if requested. Default: ALL blocks unless user sets --lora-blocks.
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

    # Optional fp16 cast (reduces VRAM). Default remains bf16 for stability.
    if getattr(args, "compute_dtype", "bf16") == "fp16" and device == "cuda":
        pipeline.transformer = pipeline.transformer.to(dtype=torch.float16)
        pipeline.vae = pipeline.vae.to(dtype=torch.float16)
        pipeline.text_encoder = pipeline.text_encoder.to(dtype=torch.float16)
    
    # Memory optimizations (best-effort; method names differ between diffusers VAEs and LTX's VAE wrapper)
    vae = pipeline.vae
    if hasattr(vae, "enable_slicing"):
        vae.enable_slicing()
    if hasattr(vae, "enable_tiling"):
        vae.enable_tiling()
    # NOTE:
    # - Some LTX VAE builds have a broken z-tiling path that assumes `vae.encoder.patch_size_t` exists.
    # - When it's missing, decode crashes with: AttributeError: 'Encoder' object has no attribute 'patch_size_t'
    # So only enable z-tiling when the expected attribute is present.
    if hasattr(vae, "enable_z_tiling") and hasattr(getattr(vae, "encoder", None), "patch_size_t"):
        vae.enable_z_tiling()
    
    # Freeze non-trained modules (we only train the transformer / LoRA).
    try:
        for p in pipeline.text_encoder.parameters():
            p.requires_grad_(False)
    except Exception:
        pass
    try:
        for p in pipeline.vae.parameters():
            p.requires_grad_(False)
    except Exception:
        pass

    # Encode prompt WITHOUT tracking gradients. Reusing a graph-backed prompt embedding across
    # multiple GRPO steps can trigger "backward through the graph a second time".
    with torch.no_grad():
        prompt_embeds_tuple = pipeline.encode_prompt(
            prompt=str(getattr(args, "prompt", "") or ""),
            device=torch.device(device),
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=str(getattr(args, "negative_prompt", "") or ""),
        )
    # LTX pipeline returns: (prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask)
    prompt_embeds = prompt_embeds_tuple[0]
    prompt_attention_mask = prompt_embeds_tuple[1]
    negative_embeds = prompt_embeds_tuple[2]
    negative_attention_mask = prompt_embeds_tuple[3]

    # Ensure embeddings are leaf tensors (no autograd history).
    prompt_embeds = prompt_embeds.detach()
    negative_embeds = negative_embeds.detach()
    
    train_blocks = None
    if args.train_blocks:
        train_blocks = [int(x.strip()) for x in args.train_blocks.split(",")]
    
    adapter = LTXAdapter(
        pipeline=pipeline,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_embeds,
        prompt_attention_mask=prompt_attention_mask,
        negative_prompt_attention_mask=negative_attention_mask,
        guidance_scale=args.guidance_scale,
        stg_scale=stg_scale_cfg,
        rescaling_scale=rescaling_scale_cfg,
        cfg_star_rescale=cfg_star_rescale_cfg,
        skip_layer_strategy=skip_layer_strategy_cfg,
        skip_block_list=skip_block_list_cfg,
        decode_timestep=decode_timestep_cfg,
        decode_noise_scale=decode_noise_scale_cfg,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        train_transformer_blocks=train_blocks,
    )
    
    return adapter

def create_wan_adapter(args):
    """Create Wan2.1 adapter (WanT2V + WanAdapter)"""
    import sys 
    import math
    import torch
    from pathlib import Path 
    from unified_grpo.adapters.wan_adapter import WanAdapter
    from unified_grpo.lora_utils import apply_lora_to_transformer
    
    # add Wan2.1 to path 
    repo_root = Path(__file__).resolve().parent.parent  # .../angel-research
    wan_path = repo_root / "Wan2.1"
    if not wan_path.exists():
        raise FileNotFoundError(
            f"Expected Wan2.1 checkout at: {str(wan_path)}\n"
            "Please ensure the repo contains `Wan2.1/wan/` or set PYTHONPATH to include the Wan2.1 directory."
        )
    if str(wan_path) not in sys.path:
        sys.path.insert(0, str(wan_path))
    import wan  # type: ignore
    
    # ---------------
    # resolve checkpoint dir + config 
    # ---------------
    ckpt_dir = getattr(args, "model_path", None)
    if not ckpt_dir:
        raise ValueError("Wan requires 'args.model_path' to be a local checkpoint directory")
    ckpt_dir = Path(str(ckpt_dir))
    if not ckpt_dir.exists():
        # Allow passing a HuggingFace repo id (e.g. "Wan-AI/Wan2.1-T2V-1.3B") and auto-download.
        model_id = str(getattr(args, "model_path", "") or "")
        try:
            from huggingface_hub import snapshot_download  # type: ignore
        except Exception as e:
            raise FileNotFoundError(
                f"Wan checkpoint directory does not exist: {str(ckpt_dir)}\n"
                f"Also failed to enable HuggingFace auto-download for model id '{model_id}' because "
                "the dependency `huggingface_hub` is not available.\n\n"
                "Fix options:\n"
                "- Pass a LOCAL checkpoint directory via `--model-path /path/to/Wan2.1-T2V-1.3B`\n"
                "- Or install huggingface_hub: `pip install huggingface_hub`\n"
            ) from e

        print(f"[WAN] Local checkpoint dir not found, downloading from HuggingFace: {model_id}")
        downloaded = snapshot_download(repo_id=model_id)
        ckpt_dir = Path(downloaded)
        if not ckpt_dir.exists():
            raise FileNotFoundError(f"[WAN] snapshot_download returned non-existent path: {str(ckpt_dir)}")
    
    wan_task = getattr(args, "wan_task", None)
    if not wan_task:
        model_path = str(getattr(args, "model_path", "") or "")
        # Default to 1.3B and refuse 14B (too heavy for this GPU)
        if ("14B" in model_path) or model_path.endswith("14B"):
            raise ValueError(
                "Wan 14B is disabled (too heavy for this GPU). "
                "Please use Wan2.1-T2V-1.3B checkpoints."
            )
        wan_task = "t2v-1.3B"

    if str(wan_task) == "t2v-14B":
        raise ValueError("Wan 14B is disabled (too heavy for this GPU). Use `t2v-1.3B`.")
    if wan_task not in wan.configs.WAN_CONFIGS:
        raise ValueError(f"Unknown wan task: '{wan_task}', Valid: {sorted(wan.configs.WAN_CONFIGS.keys())}")
    
    wan_config = wan.configs.WAN_CONFIGS[wan_task]
    
    # --------------
    # Build Wan pipeline 
    # --------------
    device_id = int(getattr(args, "device_id", torch.cuda.current_device() if torch.cuda.is_available() else 0))
    rank = int(getattr(args, "rank", 0))

    # Keep T5 on CPU by default (saves VRAM)
    t5_cpu = bool(getattr(args, "wan_t5_cpu", True))

    pipeline = wan.WanT2V(
        config=wan_config,
        checkpoint_dir=str(ckpt_dir),
        device_id=device_id,
        rank=rank,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=t5_cpu,
    )

    # ----------------------------
    # Optional LoRA on WanModel (pipeline.model.blocks)
    # ----------------------------
    if bool(getattr(args, "use_lora", False)):
        from unified_grpo.utils import resolve_lora_blocks

        model0 = getattr(pipeline, "model", None)
        blocks0 = getattr(model0, "blocks", None) if model0 is not None else None
        total_blocks = len(blocks0) if blocks0 is not None else None
        lora_blocks = resolve_lora_blocks(
            spec=getattr(args, "lora_blocks", None),
            total_blocks=total_blocks,
            unfreeze_pct=float(getattr(args, "unfreeze_percentage", 0.20)),
        )

        # Wan uses linear layers named q/k/v/o (+ i2v has k_img/v_img) under:
        #   blocks.{i}.self_attn.* and blocks.{i}.cross_attn.*
        model = getattr(pipeline, "model", None)
        if model is None:
            raise RuntimeError("WAN pipeline has no .model; cannot apply LoRA.")
        pipeline.model, _ = apply_lora_to_transformer(
            model,
            rank=int(getattr(args, "lora_rank", 16)),
            alpha=int(getattr(args, "lora_alpha", 32)),
            target_modules=["q", "k", "v", "o", "k_img", "v_img"],
            target_blocks=lora_blocks,
        )
    # ----------------------------
    # Prompt embeddings (match WanT2V.generate)
    # ----------------------------

    prompt = str(getattr(args, "prompt", "") or "")
    if not prompt:
        raise ValueError("Wan requires `--prompt` (non-empty).")

    negative_prompt = str(getattr(args, "negative_prompt", "") or "")
    if not negative_prompt:
        negative_prompt = str(getattr(pipeline, "sample_neg_prompt", "") or "")

    # Wan text encoder returns a list of tensors (context list)
    if not pipeline.t5_cpu:
        pipeline.text_encoder.model.to(pipeline.device)
        prompt_embeds = pipeline.text_encoder([prompt], pipeline.device)
        negative_prompt_embeds = pipeline.text_encoder([negative_prompt], pipeline.device)
    else:
        prompt_embeds = pipeline.text_encoder([prompt], torch.device("cpu"))
        negative_prompt_embeds = pipeline.text_encoder([negative_prompt], torch.device("cpu"))
        prompt_embeds = [t.to(pipeline.device) for t in prompt_embeds]
        negative_prompt_embeds = [t.to(pipeline.device) for t in negative_prompt_embeds]

    # ----------------------------
    # Training block selection
    # ----------------------------
    train_blocks = None
    if getattr(args, "train_blocks", None):
        train_blocks = [int(x.strip()) for x in str(args.train_blocks).split(",") if x.strip()]

    adapter = WanAdapter(
        pipeline=pipeline,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        guidance_scale=float(getattr(args, "guidance_scale", 5.0)),
        height=int(getattr(args, "height", 720)),
        width=int(getattr(args, "width", 1280)),
        num_frames=int(getattr(args, "num_frames", 81)),
        shift=float(getattr(args, "sample_shift", 5.0)),
        sample_solver=str(getattr(args, "sample_solver", "unipc")),
        train_transformer_blocks=train_blocks,
        unfreeze_percentage=float(getattr(args, "unfreeze_percentage", 0.20)) if train_blocks is None else None,
    )

    # Exact seq_len (same as WanT2V.generate)
    F = adapter.num_frames
    target_shape = (
        int(pipeline.vae.model.z_dim),
        (F - 1) // int(pipeline.vae_stride[0]) + 1,
        int(adapter.height) // int(pipeline.vae_stride[1]),
        int(adapter.width) // int(pipeline.vae_stride[2]),
    )
    sp_size = int(getattr(pipeline, "sp_size", 1))
    adapter.seq_len = int(
        math.ceil(
            (target_shape[2] * target_shape[3])
            / (pipeline.patch_size[1] * pipeline.patch_size[2])
            * target_shape[1]
            / sp_size
        )
        * sp_size
    )

    return adapter

def create_opensora_adapter(args):
    """Create Open-Sora adapter"""

    # Open-Sora is not provided via diffusers in this repo. We build it from the bundled Open-Sora codebase.
    from unified_grpo.adapters.opensora_adapter import OpenSoraAdapter, OpenSoraComponents

    # Ensure Open-Sora is importable.
    # NFS / symlinked paths can change the parent depth, so find the repo root by searching upwards.
    def _find_opensora_root() -> Path:
        # Optional explicit override
        env_root = str(getattr(args, "opensora_root", "") or "") or str(os.environ.get("OPENSORA_ROOT", "") or "")
        if env_root:
            p = Path(env_root).expanduser().resolve()
            if p.is_dir() and (p / "opensora").is_dir():
                return p
            if p.is_dir() and (p / "Open-Sora" / "opensora").is_dir():
                return (p / "Open-Sora").resolve()
            raise FileNotFoundError(f"OPENSORA_ROOT/--opensora-root points to a non-Open-Sora directory: {p}")

        here = Path(__file__).resolve()
        for parent in [here.parent, *here.parents]:
            cand = parent / "Open-Sora"
            if cand.is_dir() and (cand / "opensora").is_dir():
                return cand
        raise FileNotFoundError(
            "Open-Sora directory not found by searching parents of "
            f"{here}. Expected a sibling folder named 'Open-Sora/'. "
            "Set OPENSORA_ROOT=/path/to/angel-research (or /path/to/Open-Sora) to override."
        )

    import os

    opensora_root = _find_opensora_root()
    if str(opensora_root) not in sys.path:
        sys.path.insert(0, str(opensora_root))

    # Resolve model_path: prefer local ckpts if not provided or if provided path doesn't exist.
    model_path = str(getattr(args, "model_path", "") or "")
    if model_path and not Path(model_path).expanduser().exists():
        print(f"⚠️ Open-Sora model_path does not exist: {model_path}")
        model_path = ""
    if not model_path:
        local_ckpt = opensora_root / "ckpts" / "OpenSora-STDiT-v2-stage3"
        if (local_ckpt / "config.json").is_file() and (local_ckpt / "model.safetensors").is_file():
            model_path = str(local_ckpt)
            print(f"✅ Using local Open-Sora checkpoint: {model_path}")

    # Build components from a config (default to v1-2 inference sample, which uses RFLOW scheduler).
    # You can override with OPENSORA_CONFIG env var.
    opensora_config = str(getattr(args, "opensora_config", "") or "")
    if not opensora_config:
        # If user passed the local v2-stage3 ckpt folder, default to v1-1 inference config (STDiT2),
        # otherwise default to v1-2 (STDiT3).
        mp = model_path
        if "v2-stage3" in mp or "STDiT-v2" in mp or "OpenSora-STDiT-v2" in mp:
            opensora_config = str(opensora_root / "configs/opensora-v1-1/inference/sample.py")
        else:
            opensora_config = str(opensora_root / "configs/opensora-v1-2/inference/sample.py")

    components = OpenSoraComponents.from_config(
        config_path=opensora_config,
        model_path=model_path or None,
        device="cuda",
        dtype=str(getattr(args, "dtype", "") or "bf16"),
        height=int(getattr(args, "height", 480)),
        width=int(getattr(args, "width", 720)),
        num_frames=int(getattr(args, "num_frames", 49)),
        num_sampling_steps=int(getattr(args, "num_inference_steps", 30)),
        guidance_scale=float(getattr(args, "guidance_scale", 7.0)),
        seed=int(getattr(args, "seed", 42)),
    )

    train_blocks = None
    if getattr(args, "train_blocks", None):
        train_blocks = [int(x.strip()) for x in str(args.train_blocks).split(",") if x.strip()]

    adapter = OpenSoraAdapter(
        components=components,
        prompt=str(getattr(args, "prompt", "")),
        guidance_scale=float(getattr(args, "guidance_scale", 7.0)),
        height=int(getattr(args, "height", 480)),
        width=int(getattr(args, "width", 720)),
        num_frames=int(getattr(args, "num_frames", 49)),
        train_transformer_blocks=train_blocks,
        unfreeze_percentage=float(getattr(args, "unfreeze_percentage", 0.20)) if train_blocks is None else None,
    )

    return adapter