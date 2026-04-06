"""
X-CLIP based video reward for GRPO.

This uses a true video-text model rather than scoring frames independently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class XClipRewardConfig:
    model_id: str = "microsoft/xclip-base-patch32"
    num_sampled_frames: int = 8


_XCLIP_MODEL: Any = None
_XCLIP_PROCESSOR: Any = None
_XCLIP_MODEL_ID: Optional[str] = None


def _lazy_load_xclip(*, model_id: str, device: torch.device) -> tuple[Any, Any]:
    global _XCLIP_MODEL, _XCLIP_PROCESSOR, _XCLIP_MODEL_ID
    if _XCLIP_MODEL is not None and _XCLIP_PROCESSOR is not None and _XCLIP_MODEL_ID == model_id:
        return _XCLIP_MODEL, _XCLIP_PROCESSOR

    try:
        from transformers import AutoModel, AutoProcessor  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ModuleNotFoundError(
            "Missing dependency for X-CLIP reward.\n"
            "Install (in your training env):\n"
            "  pip install -U transformers accelerate safetensors pillow\n"
        ) from e

    processor = AutoProcessor.from_pretrained(model_id)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModel.from_pretrained(model_id, torch_dtype=dtype)
    model = model.to(device)
    model.eval()

    _XCLIP_MODEL = model
    _XCLIP_PROCESSOR = processor
    _XCLIP_MODEL_ID = model_id
    return model, processor


def _move_batch_to_device(batch: Any, device: torch.device) -> Any:
    out: Dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out


def _prepare_xclip_pixel_values(*, processor: Any, video_np: np.ndarray) -> torch.Tensor:
    """
    Build X-CLIP pixel_values with shape [B, T, C, H, W].

    Some processor/video codepaths return an empty batch for `videos=...`, so we
    explicitly fall back to the underlying video/image processor on a list of frames.
    """
    frames = list(video_np)

    batch = processor(videos=frames, return_tensors="pt")
    pixel_values = batch.get("pixel_values", None) if hasattr(batch, "get") else None

    if pixel_values is None:
        video_processor = getattr(processor, "video_processor", None)
        if video_processor is None:
            raise ValueError("X-CLIP processor did not return pixel_values and has no video_processor fallback.")
        batch = video_processor(images=frames, return_tensors="pt")
        pixel_values = batch.get("pixel_values", None)

    if pixel_values is None:
        raise ValueError("Failed to prepare X-CLIP pixel_values from sampled video frames.")

    # Image processor usually returns [T, C, H, W]; X-CLIP expects [B, T, C, H, W].
    if pixel_values.ndim == 4:
        pixel_values = pixel_values.unsqueeze(0)
    if pixel_values.ndim != 5:
        raise ValueError(f"Unexpected X-CLIP pixel_values shape: {tuple(pixel_values.shape)}")
    return pixel_values


def _sample_video_frames(video: torch.Tensor, *, num_frames: int) -> np.ndarray:
    """
    Convert decoded video tensor to uint8 frames shaped [T, H, W, 3].

    Accepts:
    - [T, C, H, W] in [0,1]
    - [B, T, C, H, W] in [0,1]
    """
    v = video
    if v.ndim == 5:
        v = v[0]
    if v.ndim != 4:
        raise ValueError(f"Expected video tensor rank 4 or 5, got shape={tuple(video.shape)}")
    if v.shape[0] == 3 and v.shape[1] != 3:
        v = v.permute(1, 0, 2, 3).contiguous()
    if v.shape[1] != 3:
        raise ValueError(f"Expected channel dim=3, got shape={tuple(v.shape)}")

    total_frames = int(v.shape[0])
    if total_frames <= 0:
        raise ValueError("Video tensor has zero frames.")

    k = max(1, min(int(num_frames), total_frames))
    if k >= total_frames:
        indices = list(range(total_frames))
    else:
        indices = torch.linspace(0, total_frames - 1, steps=k).round().to(torch.long).tolist()

    v_cpu = v.detach().float().clamp(0.0, 1.0).cpu()
    frames = []
    for idx in indices:
        frame_hwc = (
            v_cpu[int(idx)]
            .permute(1, 2, 0)
            .mul(255.0)
            .round()
            .to(torch.uint8)
            .numpy()
        )
        frames.append(frame_hwc)
    return np.stack(frames, axis=0)


@torch.inference_mode()
def xclip_video_reward(
    *,
    video: torch.Tensor,
    prompt: str,
    device: torch.device,
    cfg: XClipRewardConfig = XClipRewardConfig(),
) -> Dict[str, float]:
    """
    Return dict with a true video-text alignment reward in [0, 1].
    """
    model, processor = _lazy_load_xclip(model_id=str(cfg.model_id), device=device)
    model_device = next(model.parameters()).device

    video_np = _sample_video_frames(video, num_frames=int(cfg.num_sampled_frames))

    text_inputs = processor(text=[prompt], return_tensors="pt", padding=True)
    video_inputs = {"pixel_values": _prepare_xclip_pixel_values(processor=processor, video_np=video_np)}

    text_inputs = _move_batch_to_device(text_inputs, model_device)
    video_inputs = _move_batch_to_device(video_inputs, model_device)

    text_outputs = model.get_text_features(**text_inputs)
    video_outputs = model.get_video_features(**video_inputs)

    print(text_outputs.pooler_output.shape, video_outputs.pooler_output.shape)
    text_features = F.normalize(text_outputs.pooler_output.float(), dim=-1)
    video_features = F.normalize(video_outputs.pooler_output.float(), dim=-1)
    print(text_features.shape, video_features.shape) 

    sim = F.cosine_similarity(text_features, video_features, dim=-1).item()
    reward = max(0.0, float(sim))

    return {
        "xclip_alignment": float(sim),
        "reward": reward,
    }
