#!/usr/bin/env python3
"""
Simplified reward function using CLIP only.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Any, List, Optional, Sequence, Union

# Try to import CLIP
try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("⚠️ CLIP not available")

# Global model cache
clip_model = None

_CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073], dtype=torch.float32)
_CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711], dtype=torch.float32)

def load_clip_model(device='cuda'):
    """Load CLIP model"""
    global clip_model
    if clip_model is None and CLIP_AVAILABLE:
        print("Loading CLIP model (first time)...")
        clip_model, _ = clip.load("ViT-B/32", device=device)
        clip_model.eval()
        clip_model = clip_model.to(dtype=torch.float32)
        print("✓ CLIP model loaded (dtype: torch.float32)")
    return clip_model

def _frame_to_chw_tensor(frame: Any, *, device: str) -> torch.Tensor:
    """
    Convert one frame to a float tensor [3,H,W] in [0,1] on `device`.
    Accepts:
    - torch.Tensor ([3,H,W] or [H,W,3])
    - PIL.Image
    - numpy array ([H,W,3] uint8/float)
    """
    # Torch tensor
    if isinstance(frame, torch.Tensor):
        x = frame.detach()
        if x.ndim == 3 and x.shape[0] == 3:
            pass  # [3,H,W]
        elif x.ndim == 3 and x.shape[-1] == 3:
            x = x.permute(2, 0, 1).contiguous()
        else:
            raise ValueError(f"Unexpected frame tensor shape: {tuple(x.shape)}")
        x = x.float()
        # Handle [-1,1] -> [0,1]
        if float(x.min()) < -0.1:
            x = (x + 1.0) / 2.0
        return x.clamp(0.0, 1.0).to(device)

    # PIL image
    try:
        from PIL import Image  # type: ignore
        is_pil = isinstance(frame, Image.Image)
    except Exception:
        is_pil = False
    if is_pil:
        from torchvision.transforms.functional import to_tensor

        x = to_tensor(frame)  # [3,H,W] in [0,1]
        return x.to(device)

    # numpy array
    if isinstance(frame, np.ndarray):
        x = torch.from_numpy(frame)
        if x.ndim == 3 and x.shape[-1] == 3:
            x = x.permute(2, 0, 1).contiguous()
        elif x.ndim == 3 and x.shape[0] == 3:
            pass
        else:
            raise ValueError(f"Unexpected frame numpy shape: {tuple(frame.shape)}")
        x = x.float()
        if x.max() > 10.0:  # likely uint8 [0,255]
            x = x / 255.0
        return x.clamp(0.0, 1.0).to(device)

    raise TypeError(f"Unsupported frame type: {type(frame)}")


def _sample_frame_indices(total: int, k: int = 8) -> List[int]:
    if total <= 0:
        return []
    kk = max(1, min(int(k), int(total)))
    if kk >= total:
        return list(range(total))
    return torch.linspace(0, total - 1, steps=kk).round().to(torch.long).tolist()


def _normalize_frame_sequence(frames: Union[Sequence[Any], torch.Tensor], *, fn_name: str) -> Sequence[Any]:
    """
    Accept either a python sequence of frames or a tensor shaped [T,3,H,W] / [B,T,3,H,W].
    """
    if isinstance(frames, torch.Tensor):
        v = frames
        if v.ndim == 5:
            v = v[0]
        if v.ndim != 4:
            raise ValueError(f"{fn_name}: expected frames tensor [T,3,H,W] (or [B,T,3,H,W]), got {tuple(frames.shape)}")
        return [v[i] for i in range(int(v.shape[0]))]
    return frames


def _resolve_num_sampled_frames(total_frames: int, num_sampled_frames: Optional[int]) -> List[int]:
    """
    If num_sampled_frames is None or <= 0, use the whole video.
    """
    if num_sampled_frames is None or int(num_sampled_frames) <= 0:
        return list(range(int(total_frames)))
    return _sample_frame_indices(total_frames, k=int(num_sampled_frames))


def _prepare_clip_frame(frame: Any, *, device: str) -> torch.Tensor:
    from torchvision.transforms.functional import resize

    x = _frame_to_chw_tensor(frame, device=device)
    x = resize(x, (224, 224), antialias=True)
    mean = _CLIP_MEAN.to(device=device)
    std = _CLIP_STD.to(device=device)
    return (x - mean[:, None, None]) / std[:, None, None]


@torch.no_grad()
def clip_score(
    *,
    frames: Union[Sequence[Any], torch.Tensor],
    prompt: str,
    device: str = 'cuda',
    num_sampled_frames: Optional[int] = 8,
    aggregation: str = "video_mean_pool",
) -> float:
    """
    CLIP text-video alignment.

    Modes:
    - frame_mean: old behavior, average text-image similarity over sampled frames.
    - video_mean_pool: encode sampled frames, mean-pool the embeddings into one clip
      embedding, then compare text to the pooled clip embedding.
    """
    if not CLIP_AVAILABLE:
        return 0.5

    model = load_clip_model(device)

    frames_seq = _normalize_frame_sequence(frames, fn_name="clip_score")

    T = len(frames_seq)
    sample_indices = _resolve_num_sampled_frames(T, num_sampled_frames)
    if not sample_indices:
        return 0.0

    # Encode text
    text_tokens = clip.tokenize([prompt]).to(device)
    text_features = model.encode_text(text_tokens)
    text_features = F.normalize(text_features, dim=-1)

    # Encode frames, then aggregate at the clip level if requested.
    frame_features = []
    for idx in sample_indices:
        frame = _prepare_clip_frame(frames_seq[int(idx)], device=device)
        image_features = model.encode_image(frame.unsqueeze(0))
        image_features = F.normalize(image_features, dim=-1)
        frame_features.append(image_features)

    image_features = torch.cat(frame_features, dim=0)
    mode = str(aggregation).lower()
    if mode == "frame_mean":
        sims = F.cosine_similarity(text_features, image_features, dim=-1)
        return float(sims.clamp_min(0.0).mean().item())
    if mode != "video_mean_pool":
        raise ValueError(
            f"Unsupported CLIP aggregation mode: {aggregation}. "
            "Choose from: frame_mean, video_mean_pool"
        )

    clip_features = F.normalize(image_features.mean(dim=0, keepdim=True), dim=-1)
    sim = F.cosine_similarity(text_features, clip_features, dim=-1).item()
    return float(max(0.0, sim))

def comprehensive_grpo_reward(*, frames: Union[Sequence[Any], torch.Tensor], prompt: str, device: str = 'cuda', **kwargs):
    """
    Simplified CLIP-only reward.

    Returns dict with component scores.
    """
    scores = {}

    try:
        scores['clip_alignment'] = clip_score(
            frames=frames,
            prompt=prompt,
            device=device,
            num_sampled_frames=kwargs.get("clip_num_sampled_frames", 8),
            aggregation=str(kwargs.get("clip_aggregation", "video_mean_pool")),
        )
        scores['clip_temporal'] = scores['clip_alignment']  # same signal for now
    except Exception as e:
        print(f"⚠️ CLIP error: {e}")
        scores['clip_alignment'] = 0.5
        scores['clip_temporal'] = 0.5

    scores['reward'] = float(scores["clip_alignment"])

    print(f"\n  Reward (CLIP only):")
    print(f"    CLIP: {scores['clip_alignment']:.4f}")
    print(f"  Total: {scores['reward']:.4f}")

    return scores
