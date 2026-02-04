from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Optional

import torch

from unified_grpo.grpo_core import GRPOConfig, run_grpo_for_prompt


def _read_prompts(path: Path) -> list[str]:
    lines: list[str] = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        lines.append(s)
    return lines


def _load_reward_fn(spec: str) -> Callable[[torch.Tensor, str], torch.Tensor]:
    """
    Load a reward function from a spec like:
      "origin_grpo.reward_functions:reward_function_simple"

    The loaded callable can have signature:
      (video_bthwc, prompt) -> Tensor|float
    or
      (video_bthwc, prompt, device=...) -> Tensor|float
    """
    mod_name, fn_name = spec.split(":", 1)
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, fn_name)
    if not callable(fn):
        raise TypeError(f"Reward spec {spec!r} resolved to non-callable: {fn!r}")

    def _wrapped(video: torch.Tensor, prompt: str) -> torch.Tensor:
        sig = inspect.signature(fn)
        kwargs = {}
        if "device" in sig.parameters:
            kwargs["device"] = str(video.device)
        out = fn(video, prompt, **kwargs) if len(sig.parameters) >= 2 else fn(video)
        if isinstance(out, torch.Tensor):
            return out.float().reshape([])
        return torch.tensor(float(out), device=video.device, dtype=torch.float32)

    return _wrapped


def _dummy_reward(video: torch.Tensor, prompt: str) -> torch.Tensor:
    return torch.tensor(0.0, device=video.device, dtype=torch.float32)


def _build_wan21_adapter(args) -> Any:
    # Lazy import; Wan deps are heavy and may not exist in every env.
    from unified_grpo.adapters.wan21_adapter import Wan21Adapter  # noqa

    # Ensure `wan` is importable (handled by adapter too), then create a WanT2V instance.
    try:
        import wan  # type: ignore
        from wan.configs import SIZE_CONFIGS, WAN_CONFIGS  # type: ignore
        from wan.text2video import WanT2V  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("backend=wan21 requires Wan2.1's `wan` package to be importable.") from e

    cfg = WAN_CONFIGS[str(args.wan_task)]
    w, h = SIZE_CONFIGS[str(args.wan_size)]

    wan_t2v = WanT2V(
        cfg,
        checkpoint_dir=str(args.wan_ckpt_dir),
        device_id=int(args.device_id),
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=bool(int(args.wan_t5_cpu)),
    )

    adapter = Wan21Adapter(
        wan=wan_t2v,
        prompt=str(args.prompt_for_adapter),
        negative_prompt=str(args.negative_prompt or ""),
        width=int(w),
        height=int(h),
        num_frames=int(args.num_frames),
        shift=float(args.wan_shift),
        guide_scale=float(args.guidance_scale),
        sample_solver=str(args.wan_solver),
        train_blocks=None,
    )
    return adapter


def _build_opensora_adapter(args) -> Any:
    from unified_grpo.adapters.opensora_adapter import OpenSoraWrapper  # noqa

    # Put Open-Sora on python path if needed.
    opensora_root = Path(args.opensora_root).resolve()
    if str(opensora_root) not in os.sys.path:
        os.sys.path.insert(0, str(opensora_root))

    from mmengine.config import Config  # type: ignore
    from opensora.utils.misc import to_torch_dtype  # type: ignore
    from opensora.utils.sampling import prepare_models  # type: ignore

    cfg = Config.fromfile(str(args.opensora_config))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = to_torch_dtype(getattr(args, "dtype", "bf16"))

    model, model_ae, model_t5, model_clip, optional_models = prepare_models(
        cfg, device, dtype, offload_model=bool(int(args.opensora_offload))
    )

    adapter = OpenSoraWrapper(
        model=model,
        model_ae=model_ae,
        model_t5=model_t5,
        model_clip=model_clip,
        optional_models=optional_models,
        prompt=str(args.prompt_for_adapter),
        negative_prompt=str(args.negative_prompt or ""),
        height=int(args.height),
        width=int(args.width),
        num_frames=int(args.num_frames),
        guidance=float(args.guidance_scale),
        guidance_img=float(args.opensora_guidance_img),
        shift=bool(int(args.opensora_shift)),
        flow_shift=None if args.opensora_flow_shift is None else float(args.opensora_flow_shift),
        patch_size=int(args.opensora_patch_size),
        channel=int(args.opensora_channel),
        temporal_reduction=int(args.opensora_temporal_reduction),
        is_causal_vae=bool(int(args.opensora_is_causal_vae)),
    )
    return adapter


def _build_cogvideox_adapter(args) -> Any:
    from unified_grpo.adapters.cogvideox_adapter import CogVideoXAdapter  # noqa

    try:
        from diffusers import CogVideoXPipeline  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("backend=cogvideox requires `diffusers` installed in the current env.") from e

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.cog_dtype == "bf16" else torch.float16

    pipe = CogVideoXPipeline.from_pretrained(str(args.cog_model_path), torch_dtype=dtype).to(device=device)

    do_cfg = float(args.guidance_scale) > 1.0
    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        str(args.prompt_for_adapter),
        str(args.negative_prompt or ""),
        do_cfg,
        device=device,
    )

    adapter = CogVideoXAdapter(
        pipeline=pipe,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        guidance_scale=float(args.guidance_scale),
        height=int(args.height),
        width=int(args.width),
        num_frames=int(args.num_frames),
        attention_kwargs=None,
        extra_step_kwargs=None,
        train_transformer_blocks=None,
    )
    return adapter


def _build_hunyuan15_adapter(args) -> Any:
    """
    Build Hunyuan pipeline via the official repo helper (HunyuanVideoSampler), then wrap it into Hunyuan15Adapter.
    """
    from unified_grpo.adapters.hunyuan15_adapter import Hunyuan15Adapter  # noqa

    try:
        from hyvideo.inference import HunyuanVideoSampler  # type: ignore
        from hyvideo.diffusion.schedulers import FlowMatchDiscreteScheduler  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("backend=hunyuan15 requires the HunyuanVideo dependencies to be importable.") from e

    import argparse as _argparse
    from pathlib import Path as _Path

    # Construct minimal args namespace using defaults from hyvideo.config + overrides we care about.
    # NOTE: hyvideo.config.parse_args() parses sys.argv, so we avoid it here and build the namespace directly.
    # The sampler expects many fields; we set the commonly required ones for inference.
    ns = _argparse.Namespace()
    # Required by Inference.from_pretrained path
    ns.ulysses_degree = 1
    ns.ring_degree = 1
    ns.use_cpu_offload = bool(int(args.hunyuan_cpu_offload))
    ns.use_fp8 = False
    ns.denoise_type = "flow"

    # Network / precision
    ns.model = str(args.hunyuan_model)
    ns.latent_channels = int(args.hunyuan_latent_channels)
    ns.precision = str(args.hunyuan_precision)
    ns.rope_theta = int(args.hunyuan_rope_theta)

    # Extra models
    ns.vae = str(args.hunyuan_vae)
    ns.vae_precision = str(args.hunyuan_vae_precision)
    ns.vae_tiling = True
    ns.text_encoder = str(args.hunyuan_text_encoder)
    ns.text_encoder_precision = str(args.hunyuan_text_encoder_precision)
    ns.text_states_dim = int(args.hunyuan_text_states_dim)
    ns.text_len = int(args.hunyuan_text_len)
    ns.tokenizer = str(args.hunyuan_tokenizer)
    ns.prompt_template = str(args.hunyuan_prompt_template)
    ns.prompt_template_video = str(args.hunyuan_prompt_template_video)
    ns.hidden_state_skip_layer = int(args.hunyuan_hidden_state_skip_layer)
    ns.apply_final_norm = bool(int(args.hunyuan_apply_final_norm))

    ns.text_encoder_2 = str(args.hunyuan_text_encoder_2)
    ns.text_encoder_precision_2 = str(args.hunyuan_text_encoder_precision_2)
    ns.text_states_dim_2 = int(args.hunyuan_text_states_dim_2)
    ns.tokenizer_2 = str(args.hunyuan_tokenizer_2)
    ns.text_len_2 = int(args.hunyuan_text_len_2)

    # Denoise schedule
    ns.flow_shift = float(args.hunyuan_flow_shift)
    ns.flow_reverse = bool(int(args.hunyuan_flow_reverse))
    ns.flow_solver = str(args.hunyuan_flow_solver)

    # Misc
    ns.disable_autocast = False
    ns.vae_ver = str(args.hunyuan_vae)
    ns.batch_size = 1
    ns.num_videos = 1

    models_root = _Path(str(args.hunyuan_model_base))
    sampler = HunyuanVideoSampler.from_pretrained(models_root, args=ns)
    pipe = sampler.pipeline

    # Set scheduler (matches HunyuanVideoSampler.predict)
    scheduler = FlowMatchDiscreteScheduler(
        shift=float(args.hunyuan_flow_shift),
        reverse=bool(int(args.hunyuan_flow_reverse)),
        solver=str(args.hunyuan_flow_solver),
    )
    pipe.scheduler = scheduler

    # Build rotary freqs and n_tokens (matches sampler.predict)
    freqs_cos, freqs_sin = sampler.get_rotary_pos_embed(int(args.num_frames), int(args.height), int(args.width))
    n_tokens = int(freqs_cos.shape[0])

    # Encode prompt once and cache; adapter will wrap transformer call.
    pipe._guidance_scale = float(args.guidance_scale)
    prompt_embeds, negative_prompt_embeds, prompt_mask, negative_prompt_mask = pipe.encode_prompt(
        prompt=[str(args.prompt_for_adapter)],
        device=pipe._execution_device,
        do_classifier_free_guidance=pipe.do_classifier_free_guidance,
        negative_prompt=[str(args.negative_prompt or "")],
        prompt_embeds=None,
        attention_mask=None,
        negative_prompt_embeds=None,
        negative_attention_mask=None,
        lora_scale=None,
        clip_skip=getattr(pipe, "clip_skip", None),
        text_encoder=pipe.text_encoder,
        data_type="video",
    )

    # Concatenate for CFG (same as pipeline __call__)
    if pipe.do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        if prompt_mask is not None:
            prompt_mask = torch.cat([negative_prompt_mask, prompt_mask])

    embedded_guidance_scale = None if args.hunyuan_embedded_cfg_scale is None else float(args.hunyuan_embedded_cfg_scale)

    # Wrap transformer to accept only (x, t_expand) like our adapter expects.
    orig_tr = pipe.transformer

    def _wrapped_transformer(x, t_expand, return_dict=False):
        guidance_expand = (
            torch.tensor([embedded_guidance_scale] * x.shape[0], device=x.device, dtype=torch.float32).to(x.dtype) * 1000.0
            if embedded_guidance_scale is not None
            else None
        )
        out = orig_tr(
            x,
            t_expand,
            text_states=prompt_embeds,
            text_mask=prompt_mask,
            text_states_2=None,
            freqs_cos=freqs_cos,
            freqs_sin=freqs_sin,
            guidance=guidance_expand,
            return_dict=True,
        )["x"]
        return (out,) if not return_dict else {"x": out}

    pipe.transformer = _wrapped_transformer  # type: ignore

    # Build latent_model_input_builder that applies CFG expansion + scheduler scaling.
    def latent_model_input_builder(lat, t):
        latent_model_input = torch.cat([lat] * 2) if pipe.do_classifier_free_guidance else lat
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
        t_expand = t.repeat(latent_model_input.shape[0])
        return latent_model_input, t_expand

    # Some schedulers require n_tokens on set_timesteps; stash it on the pipe for adapter use.
    setattr(pipe, "_grpo_n_tokens", n_tokens)

    extra_step_kwargs = {}
    adapter = Hunyuan15Adapter(
        pipe=pipe,
        latent_model_input_builder=latent_model_input_builder,
        extra_step_kwargs=extra_step_kwargs,
        prompt=str(args.prompt_for_adapter),
        video_length=int(args.num_frames),
        height=int(args.height),
        width=int(args.width),
        train_double_blocks=None,
    )
    return adapter


def _build_ltx_adapter(args) -> Any:
    from unified_grpo.adapters.ltx_adapter import LTXAdapter  # noqa
    try:
        from ltx_video.inference import load_pipeline_config, create_ltx_video_pipeline  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("backend=ltx requires LTX-Video python deps to be importable (ltx_video).") from e

    cfg = load_pipeline_config(str(args.ltx_pipeline_config))
    ckpt_path = str(args.ltx_ckpt_path) if args.ltx_ckpt_path else str(cfg["checkpoint_path"])
    text_enc = str(cfg["text_encoder_model_name_or_path"])
    precision = str(cfg.get("precision", "bfloat16"))
    sampler = str(cfg.get("sampler", "from_checkpoint"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = create_ltx_video_pipeline(
        ckpt_path=ckpt_path,
        precision=precision,
        text_encoder_model_name_or_path=text_enc,
        sampler=sampler,
        device=device,
        enhance_prompt=False,
    )

    do_cfg = float(args.guidance_scale) > 1.0
    (
        prompt_embeds,
        negative_prompt_embeds,
        prompt_attention_mask,
        negative_prompt_attention_mask,
    ) = pipe.encode_prompt(
        prompt=str(args.prompt_for_adapter),
        do_classifier_free_guidance=do_cfg,
        negative_prompt=str(args.negative_prompt or ""),
        device=torch.device(device),
    )

    # Minimal indices_grid: for LTX, we need patch token coordinates. We approximate by patchifying the initial noise latents once.
    # This is sufficient for unified GRPO training code that uses transformer(inputs=latents_tokens, indices_grid=coords).
    # NOTE: Full LTX fidelity would also require conditioning_mask support; we use pure t2v (no conditioning items).
    latent_h = int(args.height) // int(pipe.vae_scale_factor)
    latent_w = int(args.width) // int(pipe.vae_scale_factor)
    latent_f = int(args.num_frames) // int(pipe.video_scale_factor)
    if hasattr(pipe.vae, "__class__") and pipe.video_scale_factor > 1:
        # causal VAE adds one latent frame
        latent_f += 1
    g = torch.Generator(device=device).manual_seed(int(args.seed))
    init_latents = torch.randn(
        (1, int(pipe.transformer.config.in_channels), latent_f, latent_h, latent_w),
        device=device,
        dtype=torch.bfloat16,
        generator=g,
    )
    lat_tokens, lat_coords = pipe.patchifier.patchify(latents=init_latents)
    from ltx_video.models.autoencoders.vae_encode import latent_to_pixel_coords  # type: ignore

    pixel_coords = latent_to_pixel_coords(lat_coords, pipe.vae, causal_fix=pipe.transformer.config.causal_temporal_positioning)
    indices_grid = pixel_coords.to(torch.float32)
    indices_grid[:, 0] = indices_grid[:, 0] * (1.0 / float(args.ltx_frame_rate))

    adapter = LTXAdapter(
        pipeline=pipe,
        prompt_embeds=prompt_embeds,
        prompt_attention_mask=prompt_attention_mask,
        indices_grid=indices_grid,
        height=int(args.height),
        width=int(args.width),
        num_frames=int(args.num_frames),
        x0_is_patchified=True,
        trainable_blocks=None,
    )
    return adapter


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", type=str, required=True, choices=["wan21", "opensora", "cogvideox", "hunyuan15", "ltx"])

    # prompts
    ap.add_argument("--prompt", type=str, default=None)
    ap.add_argument("--prompt_file", type=str, default=None)
    ap.add_argument("--negative_prompt", type=str, default="")

    # common generation
    ap.add_argument("--seed", type=int, default=26)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--num_frames", type=int, default=81)

    # GRPO
    ap.add_argument("--num_inference_steps", type=int, default=40)
    ap.add_argument("--num_grpo_steps", type=int, default=25)
    ap.add_argument("--num_rollouts", type=int, default=3)
    ap.add_argument("--rollout_noise_scale", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--logprob_sigma", type=float, default=1.0)

    # output
    ap.add_argument("--out_dir", type=str, default="unified_runs")
    ap.add_argument("--run_name", type=str, default=None)

    # reward
    ap.add_argument("--reward", type=str, default="dummy", choices=["dummy", "spec"])
    ap.add_argument("--reward_spec", type=str, default=None, help="module:function")

    # Wan2.1 backend
    ap.add_argument("--wan_ckpt_dir", type=str, default=None)
    ap.add_argument("--wan_task", type=str, default="t2v-1.3B")
    ap.add_argument("--wan_size", type=str, default="832*480")
    ap.add_argument("--wan_shift", type=float, default=8.0)
    ap.add_argument("--wan_solver", type=str, default="unipc", choices=["unipc", "dpm++"])
    ap.add_argument("--wan_t5_cpu", type=int, default=1)
    ap.add_argument("--device_id", type=int, default=0)
    ap.add_argument("--guidance_scale", type=float, default=6.0)

    # Open-Sora backend
    # CogVideoX backend
    ap.add_argument("--cog_model_path", type=str, default=None)
    ap.add_argument("--cog_dtype", type=str, default="bf16", choices=["bf16", "fp16"])

    # Hunyuan backend
    ap.add_argument("--hunyuan_model_base", type=str, default=None)
    ap.add_argument("--hunyuan_cpu_offload", type=int, default=0)
    ap.add_argument("--hunyuan_model", type=str, default="HYVideo-T/2-cfgdistill")
    ap.add_argument("--hunyuan_latent_channels", type=int, default=16)
    ap.add_argument("--hunyuan_precision", type=str, default="bf16")
    ap.add_argument("--hunyuan_rope_theta", type=int, default=256)
    ap.add_argument("--hunyuan_vae", type=str, default="884-16c-hy")
    ap.add_argument("--hunyuan_vae_precision", type=str, default="fp16")
    ap.add_argument("--hunyuan_text_encoder", type=str, default="llm")
    ap.add_argument("--hunyuan_text_encoder_precision", type=str, default="fp16")
    ap.add_argument("--hunyuan_text_states_dim", type=int, default=4096)
    ap.add_argument("--hunyuan_text_len", type=int, default=256)
    ap.add_argument("--hunyuan_tokenizer", type=str, default="llm")
    ap.add_argument("--hunyuan_prompt_template", type=str, default="dit-llm-encode")
    ap.add_argument("--hunyuan_prompt_template_video", type=str, default="dit-llm-encode-video")
    ap.add_argument("--hunyuan_hidden_state_skip_layer", type=int, default=2)
    ap.add_argument("--hunyuan_apply_final_norm", type=int, default=0)
    ap.add_argument("--hunyuan_text_encoder_2", type=str, default="clipL")
    ap.add_argument("--hunyuan_text_encoder_precision_2", type=str, default="fp16")
    ap.add_argument("--hunyuan_text_states_dim_2", type=int, default=768)
    ap.add_argument("--hunyuan_tokenizer_2", type=str, default="clipL")
    ap.add_argument("--hunyuan_text_len_2", type=int, default=77)
    ap.add_argument("--hunyuan_flow_shift", type=float, default=7.0)
    ap.add_argument("--hunyuan_flow_reverse", type=int, default=0)
    ap.add_argument("--hunyuan_flow_solver", type=str, default="euler")
    ap.add_argument("--hunyuan_embedded_cfg_scale", type=float, default=None)

    # LTX backend
    ap.add_argument("--ltx_pipeline_config", type=str, default="configs/ltxv-2b-0.9.6-dev.yaml")
    ap.add_argument("--ltx_ckpt_path", type=str, default=None)
    ap.add_argument("--ltx_frame_rate", type=int, default=16)

    ap.add_argument("--opensora_root", type=str, default="Open-Sora")
    ap.add_argument("--opensora_config", type=str, default=None)
    ap.add_argument("--opensora_offload", type=int, default=0)
    ap.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--opensora_guidance_img", type=float, default=1.0)
    ap.add_argument("--opensora_shift", type=int, default=1)
    ap.add_argument("--opensora_flow_shift", type=float, default=None)
    ap.add_argument("--opensora_patch_size", type=int, default=2)
    ap.add_argument("--opensora_channel", type=int, default=16)
    ap.add_argument("--opensora_temporal_reduction", type=int, default=1)
    ap.add_argument("--opensora_is_causal_vae", type=int, default=0)

    args = ap.parse_args()

    prompts: list[str] = []
    if args.prompt_file:
        prompts = _read_prompts(Path(args.prompt_file))
    elif args.prompt:
        prompts = [str(args.prompt)]
    else:
        raise SystemExit("Provide --prompt or --prompt_file")

    if args.reward == "spec":
        if not args.reward_spec:
            raise SystemExit("--reward spec requires --reward_spec module:function")
        reward_fn = _load_reward_fn(str(args.reward_spec))
    else:
        reward_fn = _dummy_reward

    cfg = GRPOConfig(
        num_inference_steps=int(args.num_inference_steps),
        num_grpo_steps=int(args.num_grpo_steps),
        num_rollouts=int(args.num_rollouts),
        rollout_noise_scale=float(args.rollout_noise_scale),
        lr=float(args.lr),
        grad_clip=float(args.grad_clip),
        logprob_sigma=float(args.logprob_sigma),
    )

    run_id = args.run_name or time.strftime("%Y%m%d_%H%M%S")
    base_out = Path(args.out_dir) / f"{args.backend}_{run_id}"
    base_out.mkdir(parents=True, exist_ok=True)

    all_metrics = []
    for i, prompt in enumerate(prompts):
        prompt_out = base_out / f"p{i:03d}"
        prompt_out.mkdir(parents=True, exist_ok=True)
        (prompt_out / "prompt.txt").write_text(prompt, encoding="utf-8")

        # Build adapter per prompt because some backends cache prompt embeddings internally.
        args.prompt_for_adapter = prompt

        if args.backend == "wan21":
            if not args.wan_ckpt_dir:
                raise SystemExit("--wan_ckpt_dir is required for backend=wan21")
            adapter = _build_wan21_adapter(args)
        elif args.backend == "opensora":
            if not args.opensora_config:
                raise SystemExit("--opensora_config is required for backend=opensora")
            adapter = _build_opensora_adapter(args)
        elif args.backend == "cogvideox":
            if not args.cog_model_path:
                raise SystemExit("--cog_model_path is required for backend=cogvideox")
            adapter = _build_cogvideox_adapter(args)
        elif args.backend == "hunyuan15":
            if not args.hunyuan_model_base:
                raise SystemExit("--hunyuan_model_base is required for backend=hunyuan15")
            adapter = _build_hunyuan15_adapter(args)
        elif args.backend == "ltx":
            if not args.ltx_pipeline_config:
                raise SystemExit("--ltx_pipeline_config is required for backend=ltx")
            adapter = _build_ltx_adapter(args)
        else:
            raise SystemExit(f"Unsupported backend: {args.backend}")

        metrics = run_grpo_for_prompt(
            adapter=adapter,
            prompt=prompt,
            reward_fn=reward_fn,
            seed=int(args.seed),
            out_dir=prompt_out,
            cfg=cfg,
        )
        metrics["prompt_index"] = float(i)
        all_metrics.append(metrics)
        (prompt_out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    (base_out / "all_metrics.json").write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    print(f"[done] wrote: {base_out}")


if __name__ == "__main__":
    main()

