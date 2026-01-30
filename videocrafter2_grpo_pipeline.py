#!/usr/bin/env python3
from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent

# Use the repo-root `lvdm/` package (no scaling-noise path hacks).
from lvdm.models.samplers.ddim import DDIMSampler
try:
    from omegaconf import OmegaConf  # type: ignore  # noqa: E402
except Exception:  # pragma: no cover
    OmegaConf = None  # type: ignore
from videocrafter2_grpo_runner import (
    run_grpo_for_prompt,
    _ensure_cuda,
    _parse_args,
    _save_mp4,
    sample_ddim_decode_and_latents,
    sample_fifo_decode,
)


def _get_obj_from_str(path: str):
    module, cls = path.rsplit(".", 1)
    mod = __import__(module, fromlist=[cls])
    return getattr(mod, cls)


def instantiate_from_config(config):
    """
    Minimal `instantiate_from_config` helper (previously imported from scaling-noise/utils_loc/utils.py).
    """
    if "target" not in config:
        raise KeyError("Expected key `target` to instantiate.")
    return _get_obj_from_str(config["target"])(**config.get("params", dict()))


def load_model_checkpoint(model, ckpt: str, *, strict: bool = True):
    """Minimal checkpoint loader (matches `videocrafter2_grpo._load_model_checkpoint`)."""
    try:
        state_dict = torch.load(ckpt, map_location="cpu")
    except RuntimeError as e:
        msg = str(e)
        if "PytorchStreamReader failed reading zip archive" in msg or "failed finding central directory" in msg:
            raise RuntimeError(
                "Failed to load VideoCrafter2 checkpoint. The file looks corrupted/truncated.\n"
                f"  ckpt: {ckpt}\n\n"
                "Fix: re-download the official checkpoint from Hugging Face and replace the file.\n"
                "  Source: https://huggingface.co/VideoCrafter/VideoCrafter2/resolve/main/model.ckpt\n"
                "  Suggested command:\n"
                "    wget -O base_512_v2/model.ckpt --continue "
                "https://huggingface.co/VideoCrafter/VideoCrafter2/resolve/main/model.ckpt\n\n"
                "Or run:\n"
                "    bash download_videocrafter2_ckpt.sh\n"
            ) from e
        raise
    full_strict = bool(strict)
    if "module" in state_dict:
        new_pl_sd = {}
        for key in state_dict["module"].keys():
            # Strip "module.model." prefix used by some checkpoints.
            new_pl_sd[key[16:]] = state_dict["module"][key]
        model.load_state_dict(new_pl_sd, strict=full_strict)
    else:
        if "state_dict" in list(state_dict.keys()):
            state_dict = state_dict["state_dict"]
        model.load_state_dict(state_dict, strict=full_strict)
    return model


def load_videocrafter2_model_from_config_and_ckpt(cfg_path: str, ckpt_path: str, device: torch.device):
    """
    Load VideoCrafter2 model using:
      OmegaConf.load -> config.pop('model') -> instantiate_from_config -> to(device) -> load ckpt -> eval()
    """
    if OmegaConf is None:
        raise RuntimeError("OmegaConf is required for VideoCrafter2 configs. Install `omegaconf`.")

    config = OmegaConf.load(cfg_path)
    model_config = config.pop("model", OmegaConf.create())
    model = instantiate_from_config(model_config)
    model = model.to(device)

    # Allow passing a directory like `base_512_v2/` and auto-pick the first *.ckpt inside.
    if os.path.isdir(ckpt_path):
        candidates = sorted(
            [os.path.join(ckpt_path, f) for f in os.listdir(ckpt_path) if f.lower().endswith(".ckpt")]
        )
        if len(candidates) == 0:
            raise FileNotFoundError(f"No .ckpt files found in directory: {ckpt_path}")
        ckpt_path = candidates[0]

    assert os.path.exists(ckpt_path), f"Error: checkpoint [{ckpt_path}] Not Found!"
    model = load_model_checkpoint(model, ckpt_path, strict=True)
    model.eval()
    return model


def run_baseline_before_grpo(*, model, args, out_dir: Path) -> None:
    """
    Generate and save the baseline sample (before GRPO) as `before.mp4`.
    This is purely inference-time sampling; GRPO updates happen after this call.
    """
    print("Generating baseline sample (before GRPO)...")

    sampler = DDIMSampler(model)

    cond: Dict = {
        "c_crossattn": [model.get_learned_conditioning([args.prompt])],
        "fps": torch.tensor([int(args.fps)], device=model.device).long(),
    }
    uc: Dict | None = None
    if float(args.guidance_scale) != 1.0:
        uc = {**cond, "c_crossattn": [model.get_learned_conditioning([""])]}

    sampling_mode = str(getattr(args, "sampling_mode", "ddim")).strip().lower()
    if sampling_mode == "fifo":
        before_video = sample_fifo_decode(
            model=model,
            sampler=sampler,
            cond=cond,
            uc=uc,
            height=int(args.height),
            width=int(args.width),
            seed=int(args.seed),
            guidance_scale=float(args.guidance_scale),
            ddim_eta=float(args.ddim_eta),
            fifo_video_length=int(getattr(args, "fifo_video_length", 16)),
            fifo_new_video_length=int(getattr(args, "fifo_new_video_length", int(args.num_frames))),
            fifo_num_partitions=int(getattr(args, "fifo_num_partitions", 4)),
            fifo_lookahead_denoising=bool(int(getattr(args, "fifo_lookahead_denoising", 1))),
        )
    else:
        latent_h, latent_w = int(args.height) // 8, int(args.width) // 8
        latent_shape = (4, int(args.num_frames), latent_h, latent_w)
        before_video, _ = sample_ddim_decode_and_latents(
            model=model,
            sampler=sampler,
            cond=cond,
            uc=uc,
            shape=latent_shape,
            num_steps=int(args.num_inference_steps),
            eta=float(args.ddim_eta),
            guidance_scale=float(args.guidance_scale),
            seed=int(args.seed),
        )

    _save_mp4(before_video, out_dir / "before.mp4", fps=int(args.fps))


def main() -> int:
    args = _parse_args()
    device = _ensure_cuda()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Always prefer the repo-root checkpoint `base_512_v2/model.ckpt`.
    # If the user-provided --ckpt_path exists, we'll use it; otherwise we fall back.
    preferred = str((_ROOT / "base_512_v2" / "model.ckpt").resolve())
    ckpt_path = str(getattr(args, "ckpt_path", "") or "")
    if (not ckpt_path) or (not os.path.exists(ckpt_path)):
        ckpt_path = preferred

    model = load_videocrafter2_model_from_config_and_ckpt(args.config, ckpt_path, device)

    run_baseline_before_grpo(model=model, args=args, out_dir=out_dir)

    # Run GRPO end-to-end (writes before.mp4 and after.mp4 in out_dir).
    run_grpo_for_prompt(
        model=model,
        prompt=args.prompt,
        out_dir=out_dir,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        fps=args.fps,
        seed=args.seed,
        num_inference_steps=args.num_inference_steps,
        num_grpo_steps=args.num_grpo_steps,
        num_rollouts=args.num_rollouts,
        lr=args.lr,
        trainable=args.trainable,
        rollout_noise_scale=args.rollout_noise_scale,
        normalize_advantages=bool(int(args.normalize_advantages)),
        grad_clip=args.grad_clip,
        guidance_scale=args.guidance_scale,
        ddim_eta=args.ddim_eta,
        sampling_mode=str(getattr(args, "sampling_mode", "ddim")),
        fifo_video_length=int(getattr(args, "fifo_video_length", 16)),
        fifo_new_video_length=int(getattr(args, "fifo_new_video_length", 100)),
        fifo_num_partitions=int(getattr(args, "fifo_num_partitions", 4)),
        fifo_lookahead_denoising=bool(int(getattr(args, "fifo_lookahead_denoising", 1))),
        fifo_train_last_partitions=int(getattr(args, "fifo_train_last_partitions", 2)),
        save_intermediate_rollouts=True,
        save_rollout_every=1,
        reward_mode=str(getattr(args, "reward_mode", "clip_dino")),
        reward_device=str(getattr(args, "reward_device", "cuda")),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

