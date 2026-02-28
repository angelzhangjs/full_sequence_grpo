#!/usr/bin/env python3
"""
SIMPLIFIED Reward Function - CLIP + DINO ONLY
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
dino_model = None

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

def load_dino_model(device='cuda'):
    """Load DINO model"""
    global dino_model
    if dino_model is None:
        print("Loading DINOv2 model (first time)...")
        dino_model = torch.hub.load(
            'facebookresearch/dinov2:main',
            'dinov2_vitb14'
        ).to(device).eval().to(dtype=torch.float32)
        print("✓ DINOv2 model loaded (dtype: torch.float32)")
    return dino_model


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


@torch.no_grad()
def clip_score(*, frames: Union[Sequence[Any], torch.Tensor], prompt: str, device: str = 'cuda') -> float:
    """CLIP text-frame alignment (frames-only API)."""
    if not CLIP_AVAILABLE:
        return 0.5
    
    model = load_clip_model(device)

    # Support either a python list of frames OR a frame-tensor [T,3,H,W] (or [B,T,3,H,W]).
    frames_seq: Sequence[Any]
    if isinstance(frames, torch.Tensor):
        v = frames
        if v.ndim == 5:
            v = v[0]
        if v.ndim != 4:
            raise ValueError(f"clip_score: expected frames tensor [T,3,H,W] (or [B,T,3,H,W]), got {tuple(frames.shape)}")
        frames_seq = [v[i] for i in range(int(v.shape[0]))]
    else:
        frames_seq = frames

    T = len(frames_seq)
    sample_indices = _sample_frame_indices(T, k=8)
    
    # Encode text
    text_tokens = clip.tokenize([prompt]).to(device)
    text_features = model.encode_text(text_tokens)
    text_features = F.normalize(text_features, dim=-1)
    
    # Encode frames
    scores = []
    from torchvision.transforms.functional import resize
    for idx in sample_indices:
        frame = _frame_to_chw_tensor(frames_seq[int(idx)], device=device)
        
        frame = resize(frame, (224, 224), antialias=True)
        
        # CLIP normalization
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device)
        frame = (frame - mean[:, None, None]) / std[:, None, None]
        
        image_features = model.encode_image(frame.unsqueeze(0))
        image_features = F.normalize(image_features, dim=-1)
        
        sim = F.cosine_similarity(text_features, image_features, dim=-1).item()
        scores.append(max(0, sim))
    
    return float(np.mean(scores))

@torch.no_grad()
def dino_consistency_score(*, frames: Union[Sequence[Any], torch.Tensor], device: str = 'cuda') -> float:
    """DINO feature consistency across frames (frames-only API)."""
    model = load_dino_model(device)

    frames_seq: Sequence[Any]
    if isinstance(frames, torch.Tensor):
        v = frames
        if v.ndim == 5:
            v = v[0]
        if v.ndim != 4:
            raise ValueError(f"dino_consistency_score: expected frames tensor [T,3,H,W] (or [B,T,3,H,W]), got {tuple(frames.shape)}")
        frames_seq = [v[i] for i in range(int(v.shape[0]))]
    else:
        frames_seq = frames

    T = len(frames_seq)
    sample_indices = _sample_frame_indices(T, k=8)
    
    # DINO transform
    from torchvision.transforms.functional import resize
    features = []
    for idx in sample_indices:
        frame = _frame_to_chw_tensor(frames_seq[int(idx)], device=device)
        
        frame = resize(frame, (224, 224), antialias=True)
        
        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406], device=device)
        std = torch.tensor([0.229, 0.224, 0.225], device=device)
        frame = (frame - mean[:, None, None]) / std[:, None, None]
        
        feat = model(frame.unsqueeze(0))
        feat = F.normalize(feat, dim=-1)
        features.append(feat)
    
    # Consistency = similarity between consecutive frames
    similarities = []
    for i in range(len(features) - 1):
        sim = F.cosine_similarity(features[i], features[i+1], dim=-1).item()
        similarities.append(max(0, sim))
    
    return float(np.mean(similarities)) if len(similarities) > 0 else 0.0

def comprehensive_grpo_reward(*, frames: Union[Sequence[Any], torch.Tensor], prompt: str, device: str = 'cuda', **kwargs):
    """
    SIMPLIFIED: CLIP + DINO ONLY
    
    Returns dict with component scores
    """
    scores = {}
    
    # CLIP alignment
    try:
        scores['clip_alignment'] = clip_score(frames=frames, prompt=prompt, device=device)
        scores['clip_temporal'] = scores['clip_alignment']  # same signal for now
    except Exception as e:
        print(f"⚠️ CLIP error: {e}")
        scores['clip_alignment'] = 0.5
        scores['clip_temporal'] = 0.5
    
    # DINO tracking only (removed buggy motion component!)
    try:
        scores['dino_consistency'] = dino_consistency_score(frames=frames, device=device)
    except Exception as e:
        print(f"⚠️ DINO error: {e}")
        scores['dino_consistency'] = 0.5
    
    # Combine components.
    #
    # Important: DINO "consistency" (feature similarity between adjacent frames) is a
    # strong *static bias* if you give it positive weight: the easiest way to increase
    # it is to make frames more similar (less motion).
    #
    # Default weights:
    # - CLIP (text alignment): 0.7
    # - DINO (feature consistency): 0.3
    #
    # Note: giving DINO consistency positive weight can bias toward *less motion*
    # (more similar frames). If you see "static collapse", reduce `w_dino` or
    # replace this term with a motion-aware metric.
    w_clip = float(kwargs.get("w_clip", 0.7))
    w_dino = float(kwargs.get("w_dino", 0.3))
    total_reward = (
        w_clip * float(scores["clip_alignment"]) +
        w_dino * float(scores["dino_consistency"])
    )
    
    scores['reward'] = float(total_reward)
    
    # Clean breakdown
    denom = (abs(w_clip) + abs(w_dino))
    clip_pct = (100.0 * abs(w_clip) / denom) if denom > 0 else 0.0
    dino_pct = (100.0 * abs(w_dino) / denom) if denom > 0 else 0.0
    print(f"\n  Reward (CLIP + DINO):")
    print(f"    CLIP: {scores['clip_alignment']:.4f} (w={w_clip:g}, ~{clip_pct:.0f}%)")
    print(f"    DINO: {scores['dino_consistency']:.4f} (w={w_dino:g}, ~{dino_pct:.0f}%)")
    print(f"  Total: {scores['reward']:.4f}")
    
    return scores
