"""
Qwen-based video reward for GRPO.

Focus dimensions (0-10 each, returned by Qwen as JSON):
- text_alignment: how well the video matches the text prompt
- physical_plausibility: whether the motion follows basic physics (gravity/inertia/collisions)
- dynamic_motion_consistency: temporal stability + smooth physically-coherent motion (penalize flicker/jitter/teleport)

We convert these to a single scalar reward in [0, 1] via a weighted average.

This module is designed to be optional-dependency-safe: it only imports Qwen/Transformers
when you actually call the reward function.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch


@dataclass
class QwenRewardConfig:
    model_id: str = "Qwen/Qwen2-VL-2B-Instruct"
    num_sampled_frames: int = 8
    max_new_tokens: int = 192
    temperature: float = 0.0

    w_align: float = 0.5
    w_physics: float = 0.3
    w_dynamic_motion: float = 0.2

    # Resize frames before sending to the VLM (keeps token/image compute down).
    # If None, keep original resolution.
    resize_hw: Optional[Tuple[int, int]] = (336, 336)


_QWEN_MODEL: Any = None
_QWEN_PROCESSOR: Any = None


def _lazy_load_qwen(*, model_id: str, device: torch.device) -> tuple[Any, Any]:
    global _QWEN_MODEL, _QWEN_PROCESSOR
    if _QWEN_MODEL is not None and _QWEN_PROCESSOR is not None:
        return _QWEN_MODEL, _QWEN_PROCESSOR

    try:
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ModuleNotFoundError(
            "Missing dependency for Qwen reward.\n"
            "Install (in your training env):\n"
            "  pip install -U transformers accelerate safetensors pillow\n"
        ) from e

    # Qwen2-VL examples rely on qwen-vl-utils for vision/video preprocessing.
    try:
        import qwen_vl_utils  # type: ignore  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise ModuleNotFoundError(
            "Missing dependency `qwen-vl-utils` for Qwen2-VL message preprocessing.\n"
            "Install:\n"
            "  pip install -U qwen-vl-utils\n"
        ) from e

    # Processor handles chat template + image preprocessing.
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    # Keep it flexible: allow both single-GPU and multi-GPU with device_map.
    # Use torch_dtype="auto" to avoid forcing fp16/bf16 incorrectly.
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto" if device.type == "cuda" else None,
        trust_remote_code=True,
    )
    model.eval()

    _QWEN_MODEL, _QWEN_PROCESSOR = model, processor
    return model, processor


def _video_to_pil_frames(
    video: torch.Tensor,
    *,
    num_frames: int,
    resize_hw: Optional[Tuple[int, int]],
) -> List["Any"]:
    """
    Convert a decoded video tensor to a list of PIL images (chronological order).

    Accepts:
    - [B, T, C, H, W] in [0,1]
    - [T, C, H, W] in [0,1]
    """
    try:
        from PIL import Image  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ModuleNotFoundError(
            "Missing dependency `Pillow` for Qwen reward.\n"
            "Install:\n"
            "  pip install -U pillow\n"
        ) from e

    v = video
    if v.ndim == 5:
        v = v[0]
    if v.ndim != 4:
        raise ValueError(f"Expected video tensor rank 4 or 5, got shape={tuple(video.shape)}")
    # v: [T, C, H, W] or possibly [C, T, H, W]
    if v.shape[0] == 3 and v.shape[1] != 3:
        v = v.permute(1, 0, 2, 3).contiguous()
    if v.shape[1] != 3:
        raise ValueError(f"Expected channel dim=3, got shape={tuple(v.shape)}")

    t = int(v.shape[0])
    k = max(1, int(num_frames))
    if k >= t:
        idx = list(range(t))
    else:
        idx = torch.linspace(0, t - 1, steps=k).round().to(torch.long).tolist()

    frames: List[Any] = []
    v_cpu = v.detach().float().clamp(0, 1).cpu()
    for i in idx:
        frame_chw = v_cpu[int(i)]
        frame_hwc = (frame_chw.permute(1, 2, 0) * 255.0).round().to(torch.uint8).numpy()
        im = Image.fromarray(frame_hwc)
        if resize_hw is not None:
            im = im.resize((int(resize_hw[1]), int(resize_hw[0])))  # PIL uses (W,H)
        frames.append(im)
    return frames


def _maybe_resize_pil_frames(
    frames: Sequence[Any], *, resize_hw: Optional[Tuple[int, int]]
) -> List[Any]:
    """
    Best-effort resize for user-provided PIL frames.
    If resize_hw is None, return frames as a list (no copy).
    """
    if resize_hw is None:
        return list(frames)
    out: List[Any] = []
    for im in frames:
        try:
            out.append(im.resize((int(resize_hw[1]), int(resize_hw[0]))))  # PIL uses (W,H)
        except Exception:
            # If it's not a PIL Image (or resize fails), keep original object.
            out.append(im)
    return out


def _parse_qwen_json(text: str) -> Dict[str, float]:
    """
    Robustly extract a JSON object from model output and parse numeric scores.
    """
    # Try to find the first {...} block.
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise ValueError(f"Qwen output did not contain JSON object. Output was:\n{text}")
    blob = m.group(0)
    # Sometimes models use trailing commas; try a small cleanup.
    blob = re.sub(r",\s*}", "}", blob)
    blob = re.sub(r",\s*]", "]", blob)
    data = json.loads(blob)
    out: Dict[str, float] = {}
    # Backward-compat: older prompt may return "temporal_consistency".
    mapping = {
        "text_alignment": "text_alignment",
        "physical_plausibility": "physical_plausibility",
        "dynamic_motion_consistency": "dynamic_motion_consistency",
    }
    # Accept legacy key if present.
    if "dynamic_motion_consistency" not in data and "temporal_consistency" in data:
        data = dict(data)
        data["dynamic_motion_consistency"] = data["temporal_consistency"]

    for src, dst in mapping.items():
        v = data.get(src, None)
        if v is None:
            raise KeyError(f"Missing key '{src}' in Qwen JSON: {data}")
        out[dst] = float(v)
    return out


@torch.inference_mode()
def qwen_video_reward(
    *,
    video: Optional[torch.Tensor] = None,
    frames: Optional[Sequence[Any]] = None,
    prompt: str,
    device: torch.device,
    cfg: QwenRewardConfig = QwenRewardConfig(),
) -> Dict[str, float]:
    """
    Return dict with component scores and a scalar `reward` in [0, 1].

    Inputs:
    - Prefer passing `frames`: a sequence of PIL Images in chronological order.
      This avoids converting a decoded video tensor -> PIL inside this function.
    - Backward-compatible: you can pass `video` as a decoded pixel tensor
      ([T,3,H,W] or [B,T,3,H,W] in [0,1]) and we will sample+convert to PIL frames.
    """
    model, processor = _lazy_load_qwen(model_id=str(cfg.model_id), device=device)

    if frames is not None:
        # Use user-provided frames (already decoded). Optionally resize for compute cost.
        frames_list = _maybe_resize_pil_frames(frames, resize_hw=cfg.resize_hw)
    else:
        if video is None:
            raise ValueError("qwen_video_reward: must provide either `frames` or `video`.")
        frames_list = _video_to_pil_frames(
            video,
            num_frames=int(cfg.num_sampled_frames),
            resize_hw=cfg.resize_hw,
        )

    # Qwen2-VL uses message objects with mixed image/text parts.
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "You are a strict video judge. You will be given a text prompt and a chronological sequence of frames from a generated video."},
                {"type": "text", "text": f"TEXT PROMPT:\n{prompt}"},
                {"type": "text", "text": "TASK: Rate the video on THREE criteria from 0 to 10 (higher is better):\n"
                                        "1) text_alignment: how well the content matches the prompt\n"
                                        "2) physical_plausibility: whether the motion follows basic physics (gravity/inertia/collisions), penalize teleporting/jitter/impossible dynamics\n"
                                        "3) dynamic_motion_consistency: temporal stability + smooth physically-coherent motion across frames (penalize flicker/identity drift/jitter)\n"
                                        "OUTPUT FORMAT: Output ONLY a JSON object with keys exactly: "
                                        "text_alignment, physical_plausibility, dynamic_motion_consistency. Values must be numbers."},
            ],
        }
    ]

    # Append frames as images (chronological)
    for i, im in enumerate(frames_list):
        messages[0]["content"].append({"type": "image", "image": im})
        messages[0]["content"].append({"type": "text", "text": f"Frame {i+1}/{len(frames_list)} (chronological)."})

    # Qwen2-VL utility to extract image inputs
    from qwen_vl_utils import process_vision_info  # type: ignore

    chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[chat_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    # Move to the model's device (device_map="auto" may shard; processor returns CPU tensors)
    if hasattr(model, "device"):
        inputs = inputs.to(model.device)
    else:
        inputs = inputs.to(device)

    gen = model.generate(
        **inputs,
        max_new_tokens=int(cfg.max_new_tokens),
        do_sample=bool(float(cfg.temperature) > 0),
        temperature=float(cfg.temperature) if float(cfg.temperature) > 0 else None,
    )
    out_ids = gen[0][inputs["input_ids"].shape[-1] :]
    text = processor.decode(out_ids, skip_special_tokens=True)

    scores = _parse_qwen_json(text)
    # Clamp to [0,10] defensively
    ta = max(0.0, min(10.0, float(scores["text_alignment"])))
    ph = max(0.0, min(10.0, float(scores["physical_plausibility"])))
    dm = max(0.0, min(10.0, float(scores["dynamic_motion_consistency"])))

    w_sum = max(1e-8, float(cfg.w_align) + float(cfg.w_physics) + float(cfg.w_dynamic_motion))
    total_0_10 = (float(cfg.w_align) * ta + float(cfg.w_physics) * ph + float(cfg.w_dynamic_motion) * dm) / w_sum
    reward_0_1 = total_0_10 / 10.0

    return {
        "text_alignment": ta,
        "physical_plausibility": ph,
        "dynamic_motion_consistency": dm,
        "reward": float(reward_0_1),
        "raw_total_0_10": float(total_0_10),
        "model_id": str(cfg.model_id),
        "num_sampled_frames": int(cfg.num_sampled_frames),
    }


def _load_mp4_to_video_tensor(path: str) -> torch.Tensor:
    """
    Load MP4 -> float tensor [1, T, 3, H, W] in [0,1] using imageio.
    """
    try:
        import imageio.v3 as iio  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ModuleNotFoundError(
            "Missing dependency `imageio` for loading mp4 in the Qwen reward CLI.\n"
            "Install:\n"
            "  pip install -U imageio imageio-ffmpeg\n"
        ) from e

    arr = iio.imread(path)  # (T,H,W,3) uint8
    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(f"Unexpected video array shape from imageio: {arr.shape}")
    t = torch.from_numpy(arr).to(torch.float32) / 255.0
    t = t.permute(0, 3, 1, 2).contiguous()  # [T,3,H,W]
    return t.unsqueeze(0)  # [1,T,3,H,W]


def main() -> None:  # pragma: no cover
    import argparse

    p = argparse.ArgumentParser(description="Score a video with Qwen reward (alignment/physics/temporal).")
    p.add_argument("--video", type=str, required=True, help="Path to an mp4 file.")
    p.add_argument("--prompt", type=str, required=True, help="Text prompt.")
    p.add_argument("--model-id", type=str, default=QwenRewardConfig.model_id)
    p.add_argument("--num-frames", type=int, default=QwenRewardConfig.num_sampled_frames)
    p.add_argument("--max-new-tokens", type=int, default=QwenRewardConfig.max_new_tokens)
    p.add_argument("--temperature", type=float, default=QwenRewardConfig.temperature)
    p.add_argument("--w-align", type=float, default=QwenRewardConfig.w_align)
    p.add_argument("--w-physics", type=float, default=QwenRewardConfig.w_physics)
    # New name + backward-compatible alias
    p.add_argument("--w-dynamic-motion", dest="w_dynamic_motion", type=float, default=QwenRewardConfig.w_dynamic_motion)
    p.add_argument("--w-temporal", dest="w_dynamic_motion", type=float, default=QwenRewardConfig.w_dynamic_motion, help="Alias for --w-dynamic-motion")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vid = _load_mp4_to_video_tensor(args.video)
    scores = qwen_video_reward(
        video=vid,
        prompt=str(args.prompt),
        device=device,
        cfg=QwenRewardConfig(
            model_id=str(args.model_id),
            num_sampled_frames=int(args.num_frames),
            max_new_tokens=int(args.max_new_tokens),
            temperature=float(args.temperature),
            w_align=float(args.w_align),
            w_physics=float(args.w_physics),
            w_dynamic_motion=float(args.w_dynamic_motion),
        ),
    )
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()

