#!/usr/bin/env python3
"""
Comprehensive Reward Functions for Video Generation GRPO Training

Combines multiple modalities:
  - CLIP: Text-video semantic alignment
  - DINO: Object tracking and consistency
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict
from torchvision.transforms.functional import resize


import io
import os
import cv2
import json
import clip
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms.functional import resize
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, ToPILImage
from omegaconf import OmegaConf

from transformers import AutoModel, AutoProcessor
from transformers.image_utils import load_image


sub_model = None

def subject_consistency(video_list, device):
    global sub_model
    if sub_model == None:
        submodules_list = {'repo_or_dir': 'facebookresearch/dino:main', 'source': 'github', 'model': 'dino_vitb16', 'read_frame': None}
        
        dino_model = torch.hub.load(**submodules_list).to(device)
        sub_model = dino_model
    else:
        dino_model = sub_model
    sim = 0.0
    cnt = 0
    video_sim = 0
    images_list = [dino_transform_image_gpu(video_list[i].to(device), 224, device) for i in range(len(video_list))]

    with torch.no_grad():
        anchor_image = images_list[0].unsqueeze(0)
        anchor_image = anchor_image.to(device)
        anchor_features = dino_model(anchor_image)
        anchor_features = F.normalize(anchor_features, dim=-1, p=2)
   
    
    image_list = images_list[1:]
    for i in range(len(images_list)):
        with torch.no_grad():
            image = images_list[len(images_list)-1].unsqueeze(0)
            image = image.to(device)
            image_features = dino_model(image)
            image_features = F.normalize(image_features, dim=-1, p=2)
            if i == 0:
                sim_pre = max(0.0, F.cosine_similarity(anchor_features, image_features).item())
                cur_sim = sim_pre
                video_sim += cur_sim
            else:
                sim_pre = max(0.0, F.cosine_similarity(former_image_features, image_features).item())
                sim_fir = max(0.0, F.cosine_similarity(anchor_features, image_features).item())
                cur_sim = (sim_pre + sim_fir) / 2
                video_sim += cur_sim
        former_image_features = image_features
    sim_per_images = video_sim / (len(images_list) - 1)
    return sim_per_images
  

def dino_transform_image_gpu(batch_tensor, n_px, device):
    resized_tensor = resize(batch_tensor, (n_px, n_px), antialias=False)
    
    mean = torch.tensor([0.485, 0.456, 0.406], device=device) 
    std = torch.tensor([0.229, 0.224, 0.225], device=device)   

    normalized_tensor = (resized_tensor - mean[:, None, None]) / std[:, None, None]

    return normalized_tensor

# ============================================================================
# Lightweight helpers (no physics models; just basic pixel heuristics)
# ============================================================================

@torch.no_grad()
def _to_01(video_cthw: torch.Tensor) -> torch.Tensor:
    """Convert video to [0,1] range if it looks like [-1,1]."""
    v = video_cthw.float()
    if v.min().item() < 0.0:
        v = (v + 1.0) / 2.0
    return v.clamp(0.0, 1.0)


@torch.no_grad()
def centering_reward(video_cthw: torch.Tensor) -> float:
    """
    Soft centering without freezing the subject:
    - Strongly rewards staying away from edges ("don't leave frame")
    - Only weakly rewards being exactly at the center
    Returns in [0,1].
    """
    try:
        v01 = _to_01(video_cthw)
        C, T, H, W = v01.shape
        if T < 2:
            return 0.5

        intensity = v01.mean(dim=0)  # [T,H,W]
        intensity = intensity.clamp_min(0.0)

        y_coords = torch.arange(H, device=v01.device, dtype=torch.float32).view(1, H, 1)
        x_coords = torch.arange(W, device=v01.device, dtype=torch.float32).view(1, 1, W)

        total = intensity.flatten(1).sum(dim=1) + 1e-6  # [T]
        y_center = (intensity * y_coords).sum(dim=(1, 2)) / total
        x_center = (intensity * x_coords).sum(dim=(1, 2)) / total

        # Edge-safe score (dominant): distance to nearest border
        edge_dist = torch.minimum(
            torch.minimum(x_center, (W - 1) - x_center),
            torch.minimum(y_center, (H - 1) - y_center),
        )  # [T]
        edge_margin = 0.08 * float(min(H, W))
        edge_score = torch.clamp(edge_dist / (edge_margin + 1e-6), 0.0, 1.0).mean().item()

        # Soft center pull (weak): normalized distance to image center
        cx = (W - 1) / 2.0
        cy = (H - 1) / 2.0
        dist = torch.sqrt((x_center - cx) ** 2 + (y_center - cy) ** 2)  # [T]
        center_radius = 0.35 * float(min(H, W))
        center_score = (1.0 - torch.clamp(dist / (center_radius + 1e-6), 0.0, 1.0)).mean().item()

        score = 0.75 * edge_score + 0.25 * center_score
        return float(np.clip(score, 0.0, 1.0))
    except Exception as e:
        print(f"⚠️ centering_reward failed: {e}")
        return 0.5


@torch.no_grad()
def motion_gate(video_cthw: torch.Tensor, low: float = 0.003, high: float = 0.02) -> float:
    """
    Minimum-motion gate in [0,1] based on mean absolute pixel change in [0,1].
    - ~0 for almost-static videos
    - ~1 for clearly moving videos
    """
    try:
        v01 = _to_01(video_cthw)
        if v01.shape[1] < 2:
            return 0.0
        diffs = torch.abs(v01[:, 1:, :, :] - v01[:, :-1, :, :]).mean(dim=(0, 2, 3))  # [T-1]
        mag = float(diffs.mean().item())
        gate = (mag - low) / (high - low + 1e-8)
        return float(np.clip(gate, 0.0, 1.0))
    except Exception as e:
        print(f"⚠️ motion_gate failed: {e}")
        return 0.0

# ============================================================================
# Try to import CLIP (optional dependency)
# ============================================================================
try:
    import clip as openai_clip  # type: ignore
    if hasattr(openai_clip, 'load'):
        clip = openai_clip
        CLIP_AVAILABLE = True
    else:
        import importlib
        import sys
        if 'clip' in sys.modules:
            importlib.reload(sys.modules['clip'])
        import clip  # type: ignore
        CLIP_AVAILABLE = hasattr(clip, 'load')
        if not CLIP_AVAILABLE:
            print("⚠️ CLIP module found but 'load' function missing.")
except ImportError as e:
    CLIP_AVAILABLE = False
    print(f"⚠️ CLIP not installed: {e}")
    print("   To install: pip install git+https://github.com/openai/CLIP.git")
except Exception as e:
    CLIP_AVAILABLE = False
    print(f"⚠️ Error loading CLIP: {e}")

# ============================================================================
# Global Model Cache (Lazy Loading)
# ============================================================================
dino_model = None
clip_model = None
clip_preprocess = None

def clear_model_cache():
    """Clear cached models to force reload with correct dtype"""
    global dino_model, clip_model, clip_preprocess
    dino_model = None
    clip_model = None
    clip_preprocess = None
    print("✓ Model cache cleared")

# ============================================================================
# Model Loading Functions
# ============================================================================

def dino_transform_image_gpu(batch_tensor, n_px, device):
    """Transform image for DINO model (GPU-based)"""
    resized_tensor = resize(batch_tensor, (n_px, n_px), antialias=False)
    
    # ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)
    std = torch.tensor([0.229, 0.224, 0.225], device=device)
    
    normalized_tensor = (resized_tensor - mean[:, None, None]) / std[:, None, None]
    
    return normalized_tensor


def load_dino_model(device='cuda'):
    """Lazy load DINOv2 model"""
    global dino_model
    
    if dino_model is None:
        print("Loading DINOv2 model...")
        try:
            dino_model = torch.hub.load(
                'facebookresearch/dinov2',
                'dinov2_vitb14',
                source='github'
            )
            dino_model.eval()
            # Explicitly convert ALL parameters to float32
            dino_model = dino_model.to(device=device, dtype=torch.float32)
            # Verify conversion
            sample_param_dtype = next(dino_model.parameters()).dtype
            print(f"✓ DINOv2 model loaded (dtype: {sample_param_dtype})")
        except Exception as e:
            print(f"⚠️ DINO loading error: {e}")
            return None
    
    return dino_model


def load_clip_model(device='cuda'):
    """Lazy load CLIP model"""
    global clip_model, clip_preprocess
    
    if not CLIP_AVAILABLE:
        raise ImportError("CLIP is not installed.")
    
    if clip_model is None:
        print("Loading CLIP model...")
        try:
            clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
            clip_model.eval()
            # Explicitly convert ALL parameters to float32
            clip_model = clip_model.to(dtype=torch.float32)
            # Verify conversion worked
            sample_param_dtype = next(clip_model.parameters()).dtype
            if sample_param_dtype != torch.float32:
                print(f"⚠️ CLIP conversion failed, got {sample_param_dtype}")
            print(f"✓ CLIP model loaded (dtype: {sample_param_dtype})")
        except Exception as e:
            print(f"⚠️ CLIP loading error: {e}")
            # Return None to disable CLIP
            return None, None
    
    return clip_model, clip_preprocess


# ============================================================================
# DINO-Based Rewards
# ============================================================================

@torch.no_grad()
def dino_object_presence_reward(video: torch.Tensor, prompt: str = None, device='cuda') -> float:
    """Object presence and saliency using DINO"""
    if video is None or not isinstance(video, torch.Tensor):
        return 0.5
    
    try:
        dino = load_dino_model(device)
        if dino is None:
            return 0.5
    except Exception as e:
        print(f"⚠️ DINO model loading failed: {e}")
        return 0.5
    
    # Normalize format
    if len(video.shape) == 5:
        video = video[0]
    if video.shape[1] == 3 and video.shape[0] > 3:
        video = video.permute(1, 0, 2, 3)
    
    # CRITICAL: Ensure float32
    video = video.float().to(device)
    
    C, T, H, W = video.shape
    mid_frame = video[:, T // 2, :, :]
    
    if mid_frame.min() < 0:
        mid_frame = (mid_frame + 1) / 2
    
    # Ensure float32
    mid_frame = mid_frame.float()
    
    frame_transformed = dino_transform_image_gpu(mid_frame, 224, device)
    frame_batch = frame_transformed.unsqueeze(0)
    features = dino(frame_batch)
    
    feature_magnitude = features.norm().item()
    presence_score = np.clip(feature_magnitude / 10.0, 0, 1)
    
    return float(presence_score)


@torch.no_grad()
def dino_subject_consistency_reward(video: torch.Tensor, prompt: str = None, device='cuda') -> float:
    """Subject consistency using DINO - tracks object identity across frames"""
    if video is None or not isinstance(video, torch.Tensor):
        return 0.5
    
    try:
        dino = load_dino_model(device)
    except Exception as e:
        print(f"⚠️ DINO model loading failed: {e}")
        return 0.5
    
    # Normalize format
    if len(video.shape) == 5:
        video = video[0]
    if video.shape[1] == 3 and video.shape[0] > 3:
        video = video.permute(1, 0, 2, 3)
    
    # CRITICAL: Ensure float32
    video = video.float().to(device)
    
    C, T, H, W = video.shape
    
    # Sample frames
    num_frames_sample = min(8, T)
    frame_indices = torch.linspace(0, T-1, num_frames_sample).long()
    
    # Transform frames
    images_list = []
    for idx in frame_indices:
        frame = video[:, idx, :, :]
        if frame.min() < 0:
            frame = (frame + 1) / 2
        # Ensure float32
        frame = frame.float()
        transformed = dino_transform_image_gpu(frame, 224, device)
        images_list.append(transformed)
    
    # Compute consistency
    video_sim = 0.0
    
    with torch.no_grad():
        anchor_image = images_list[0].unsqueeze(0)
        anchor_features = dino(anchor_image)
        anchor_features = F.normalize(anchor_features, dim=-1, p=2)
    
    former_features = anchor_features
    
    for i in range(1, len(images_list)):
        with torch.no_grad():
            image = images_list[i].unsqueeze(0)
            image_features = dino(image)
            image_features = F.normalize(image_features, dim=-1, p=2)
            
            sim_prev = max(0.0, F.cosine_similarity(former_features, image_features, dim=-1).item())
            sim_anchor = max(0.0, F.cosine_similarity(anchor_features, image_features, dim=-1).item())
            
            cur_sim = 0.6 * sim_prev + 0.4 * sim_anchor
            video_sim += cur_sim
            
            former_features = image_features
    
    sim_per_frame = video_sim / (len(images_list) - 1) if len(images_list) > 1 else 0.5
    
    return float(sim_per_frame)


# ============================================================================
# CLIP-Based Rewards
# ============================================================================

@torch.no_grad()
def clip_text_alignment_reward(video: torch.Tensor, prompt: str, device='cuda') -> float:
    """Text-video alignment using CLIP"""
    if not CLIP_AVAILABLE or prompt is None:
        return 0.5
    
    try:
        clip_model, _ = load_clip_model(device)
        if clip_model is None:
            return 0.5
    except Exception as e:
        print(f"⚠️ CLIP model loading failed: {e}")
        return 0.5
    
    # Normalize format
    if len(video.shape) == 5:
        video = video[0]
    if video.shape[1] == 3 and video.shape[0] > 3:
        video = video.permute(1, 0, 2, 3)
    
    # CRITICAL: Ensure float32 dtype for all operations
    video = video.float().to(device)
    
    C, T, H, W = video.shape
    
    # Sample frames
    num_frames_to_sample = min(8, T)
    frame_indices = torch.linspace(0, T-1, num_frames_to_sample).long()
    
    # Encode text
    text_tokens = clip.tokenize([prompt]).to(device)
    text_features = clip_model.encode_text(text_tokens)
    text_features = F.normalize(text_features, dim=-1)
    
    # Encode frames
    frame_similarities = []
    for idx in frame_indices:
        frame = video[:, idx, :, :]
        
        if frame.min() < 0:
            frame = (frame + 1) / 2
        
        # Ensure float32
        frame = frame.float()
        
        frame_resized = resize(frame, (224, 224), antialias=True)
        
        # CLIP normalization
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device)
        frame_normalized = (frame_resized - mean[:, None, None]) / std[:, None, None]
        
        frame_batch = frame_normalized.unsqueeze(0)
        image_features = clip_model.encode_image(frame_batch)
        image_features = F.normalize(image_features, dim=-1)
        
        similarity = F.cosine_similarity(text_features, image_features, dim=-1).item()
        frame_similarities.append(max(0, similarity))
    
    alignment_score = np.mean(frame_similarities)
    
    return float(alignment_score)


@torch.no_grad()
def clip_temporal_alignment_reward(video: torch.Tensor, prompt: str, device='cuda') -> float:
    """Temporal text alignment using CLIP"""
    if not CLIP_AVAILABLE or prompt is None:
        return 0.5
    
    clip_model, _ = load_clip_model(device)
    
    # Normalize format
    if len(video.shape) == 5:
        video = video[0]
    if video.shape[1] == 3 and video.shape[0] > 3:
        video = video.permute(1, 0, 2, 3)
    
    C, T, H, W = video.shape
    
    # Temporal prompts
    temporal_prompts = [
        f"beginning of {prompt}",
        f"middle of {prompt}",
        f"end of {prompt}",
    ]
    
    text_tokens = clip.tokenize(temporal_prompts).to(device)
    text_features = clip_model.encode_text(text_tokens)
    text_features = F.normalize(text_features, dim=-1)
    
    # Sample frames from temporal regions
    temporal_regions = [T // 6, T // 2, T * 5 // 6]
    
    alignments = []
    for region_idx, frame_t in enumerate(temporal_regions):
        frame = video[:, frame_t, :, :]
        
        if frame.min() < 0:
            frame = (frame + 1) / 2
        
        # Ensure float32
        frame = frame.float()
        
        frame_resized = resize(frame, (224, 224), antialias=True)
        
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device)
        frame_normalized = (frame_resized - mean[:, None, None]) / std[:, None, None]
        
        image_features = clip_model.encode_image(frame_normalized.unsqueeze(0))
        image_features = F.normalize(image_features, dim=-1)
        
        similarity = F.cosine_similarity(
            text_features[region_idx:region_idx+1], image_features, dim=-1
        ).item()
        
        alignments.append(max(0, similarity))
    
    temporal_score = np.mean(alignments)
    
    return float(temporal_score)


# ============================================================================
# Comprehensive Reward Function (CLIP + DINO only)
# ============================================================================

@torch.no_grad()
def comprehensive_grpo_reward(
    video: torch.Tensor,
    prompt: str,
    device: str = 'cuda',
    use_clip: bool = True,
    use_dino: bool = True,
) -> Dict[str, float]:
    """
    Comprehensive reward combining CLIP + DINO only.
    
    Returns dict with all component scores and total reward
    """
    scores = {}
    
    # Helper: keep everything in a consistent format for scoring.
    def _normalize_video(video_tensor: torch.Tensor) -> torch.Tensor:
        # Accept [B,C,T,H,W] or [C,T,H,W] or [T,C,H,W]
        if len(video_tensor.shape) == 5:
            video_tensor = video_tensor[0]
        # Heuristic: if looks like [T,C,H,W], permute to [C,T,H,W]
        if video_tensor.shape[0] > 3 and video_tensor.shape[1] == 3:
            video_tensor = video_tensor.permute(1, 0, 2, 3)
        return video_tensor.float()

    video = _normalize_video(video)
    
    # CLIP Rewards
    if use_clip and prompt and CLIP_AVAILABLE:
        try:
            scores['clip_alignment'] = clip_text_alignment_reward(video, prompt, device)
            scores['clip_temporal'] = clip_temporal_alignment_reward(video, prompt, device)
        except Exception as e:
            print(f"⚠️ CLIP evaluation failed: {e}")
            scores['clip_alignment'] = 0.5
            scores['clip_temporal'] = 0.5
    else:
        if use_clip and not CLIP_AVAILABLE:
            print("⚠️ CLIP requested but not available")
        scores['clip_alignment'] = 0.5
        scores['clip_temporal'] = 0.5
    
    # DINO Rewards
    if use_dino:
        try:
            scores['dino_consistency'] = dino_subject_consistency_reward(video, prompt, device)
            scores['dino_presence'] = dino_object_presence_reward(video, prompt, device)
        except Exception as e:
            print(f"⚠️ DINO evaluation failed: {e}")
            scores['dino_consistency'] = 0.5
            scores['dino_presence'] = 0.5
    else:
        scores['dino_consistency'] = 0.5
        scores['dino_presence'] = 0.5
    
    # Soft centering + minimum-motion gate (prevents "static centered subject" loophole)
    scores['centering'] = centering_reward(video)
    scores['motion_gate'] = motion_gate(video)
    
    # ------------------------------------------------------------------------
    # CLIP + DINO ONLY:
    # - Text alignment: CLIP semantic + temporal matching
    # - Object tracking: DINO consistency + presence
    # Total reward is a weighted mix of the two buckets.
    # ------------------------------------------------------------------------
    text_alignment = 0.5 * scores['clip_alignment'] + 0.5 * scores['clip_temporal']

    object_tracking_raw = 0.5 * scores['dino_consistency'] + 0.5 * scores['dino_presence']
    # Gate DINO reward toward neutral (0.5) when motion is too small.
    object_tracking_gated = 0.5 + scores['motion_gate'] * (object_tracking_raw - 0.5)
    # Mix in a small centering term (mostly "don't leave the frame").
    object_tracking = 0.85 * object_tracking_gated + 0.15 * scores['centering']

    scores['text_alignment'] = float(np.clip(text_alignment, 0.0, 1.0))
    scores['object_tracking'] = float(np.clip(object_tracking, 0.0, 1.0))

    total_reward = 0.6 * scores['text_alignment'] + 0.4 * scores['object_tracking']
    
    scores['reward'] = float(np.clip(total_reward, 0.0, 1.0))
    
    return scores


@torch.no_grad()
def reward_function(
    video: torch.Tensor,
    prompt: str,
    device: str = 'cuda',
) -> torch.Tensor:
    """
    CLIP + DINO REWARD FUNCTION
    
    Returns reward as tensor (compatible with GRPO pipeline)
    
    Component breakdown:
    - Text alignment (60%): CLIP semantic + temporal matching
    - Object tracking (40%): DINO consistency + presence, with:
        - minimum-motion gate (prevents static loophole)
        - small centering term (penalizes leaving frame edges)
    """
    # # Validation
    # if video is None:
    #     print("⚠️ ERROR: Video is None!")
    #     return torch.tensor(0.5, device=device)
    
    # if not isinstance(video, torch.Tensor):
    #     print(f"⚠️ ERROR: Video is not tensor (got {type(video)})")
    #     return torch.tensor(0.5, device=device)
    
    # if video.numel() == 0:
    #     print("⚠️ ERROR: Video tensor is empty!")
    #     return torch.tensor(0.5, device=device)
    
    # Ensure float32 for CLIP/DINO compatibility
    if video.dtype == torch.bfloat16:
        video = video.float()
    
    # Compute comprehensive rewards
    result = comprehensive_grpo_reward(
        video=video,
        prompt=prompt,
        device=device,
        use_clip=True,
        use_dino=True,
    )
    
    # Logging
    print(f"\n  Reward Components (CLIP/DINO):")
    print(f"    Text alignment: {result['text_alignment']:.4f} (60%)  [clip={result['clip_alignment']:.4f}, temporal={result['clip_temporal']:.4f}]")
    print(
        f"    Object tracking: {result['object_tracking']:.4f} (40%)  "
        f"[dino_cons={result['dino_consistency']:.4f}, dino_pres={result['dino_presence']:.4f}, "
        f"center={result.get('centering', 0.5):.4f}, motion_gate={result.get('motion_gate', 0.0):.4f}]"
    )
    print(f"  Total Reward: {result['reward']:.4f}")
    
    return torch.tensor(result['reward'], device=device)


# =============================================================================
# Adaptive reward mixing (for GRPO)
# =============================================================================

from dataclasses import dataclass


@dataclass
class AdaptiveRewardConfig:
    """
    Online reweighting of reward components using EMA of rollout performance.

    Intuition:
    - Track recent mean score for each component (EMA).
    - Increase weight on components that are below their target competence.
    - Smooth weight updates to avoid thrashing.
    """

    # Start weights (will be normalized)
    w_text_alignment: float = 0.6
    w_object_tracking: float = 0.4

    # EMA smoothing for component means
    ema_beta: float = 0.95

    # How fast to move weights toward the proposed new weights (0..1)
    weight_lr: float = 0.2

    # Targets in [0,1]. Higher means "we want this component to get good".
    target_text_alignment: float = 0.75
    target_object_tracking: float = 0.75

    # Softmax temperature for turning gaps into weights (higher => more aggressive)
    gap_temperature: float = 6.0

    # Clamp to keep both signals active
    min_weight: float = 0.1
    max_weight: float = 0.9


class AdaptiveRewardMixer:
    """
    Computes component rewards using `comprehensive_grpo_reward` and mixes them into
    a scalar reward with weights updated online from recent rollouts.
    """

    def __init__(self, config: AdaptiveRewardConfig | None = None):
        self.cfg = config or AdaptiveRewardConfig()

        w_text = float(self.cfg.w_text_alignment)
        w_obj = float(self.cfg.w_object_tracking)
        s = max(w_text + w_obj, 1e-8)
        self.w_text = w_text / s
        self.w_obj = w_obj / s

        # EMA of component means (initialized to neutral)
        self.ema_text = 0.5
        self.ema_obj = 0.5

        # Step counter for logging/debugging
        self.steps = 0

    @torch.no_grad()
    def score_components(self, *, video: torch.Tensor, prompt: str, device: str = "cuda") -> Dict[str, float]:
        return comprehensive_grpo_reward(video=video, prompt=prompt, device=device, use_clip=True, use_dino=True)

    def scalar_from_components(self, components: Dict[str, float], *, device: str = "cuda") -> torch.Tensor:
        text = float(components.get("text_alignment", 0.5))
        obj = float(components.get("object_tracking", 0.5))
        r = self.w_text * text + self.w_obj * obj
        return torch.tensor(float(np.clip(r, 0.0, 1.0)), device=device)

    def update_from_rollouts(self, rollout_components: list[Dict[str, float]]) -> None:
        """
        Update EMA stats and then update weights.

        Call once per GRPO timestep (i.e., after you collect K rollouts).
        """
        if not rollout_components:
            return

        # Mean over rollouts for this timestep
        text_vals = [float(rc.get("text_alignment", 0.5)) for rc in rollout_components]
        obj_vals = [float(rc.get("object_tracking", 0.5)) for rc in rollout_components]
        mean_text = float(np.mean(text_vals))
        mean_obj = float(np.mean(obj_vals))

        b = float(self.cfg.ema_beta)
        self.ema_text = b * self.ema_text + (1.0 - b) * mean_text
        self.ema_obj = b * self.ema_obj + (1.0 - b) * mean_obj

        # "Competence gap" relative to targets; higher gap => higher weight.
        gap_text = float(self.cfg.target_text_alignment) - self.ema_text
        gap_obj = float(self.cfg.target_object_tracking) - self.ema_obj

        # Softmax over gaps (stabilize by subtracting max).
        temp = float(self.cfg.gap_temperature)
        g = np.array([gap_text * temp, gap_obj * temp], dtype=np.float32)
        g = g - float(np.max(g))
        p = np.exp(g)
        p = p / max(float(np.sum(p)), 1e-8)
        w_text_new = float(p[0])
        w_obj_new = float(p[1])

        # Clamp weights so both terms keep contributing.
        w_text_new = float(np.clip(w_text_new, self.cfg.min_weight, self.cfg.max_weight))
        w_obj_new = float(np.clip(w_obj_new, self.cfg.min_weight, self.cfg.max_weight))
        s = max(w_text_new + w_obj_new, 1e-8)
        w_text_new /= s
        w_obj_new /= s

        # Smooth update.
        lr = float(self.cfg.weight_lr)
        self.w_text = (1.0 - lr) * self.w_text + lr * w_text_new
        self.w_obj = (1.0 - lr) * self.w_obj + lr * w_obj_new

        # Re-normalize in case of numeric drift.
        s2 = max(self.w_text + self.w_obj, 1e-8)
        self.w_text /= s2
        self.w_obj /= s2

        self.steps += 1

    def debug_state(self) -> Dict[str, float]:
        return {
            "steps": float(self.steps),
            "w_text_alignment": float(self.w_text),
            "w_object_tracking": float(self.w_obj),
            "ema_text_alignment": float(self.ema_text),
            "ema_object_tracking": float(self.ema_obj),
        }


if __name__ == "__main__":
    print("Comprehensive Reward Functions Module")
    print("\nAvailable functions:")
    print("  - reward_function(): Main GRPO reward (CLIP + DINO, with centering + motion gate)")
    print("  - comprehensive_grpo_reward(): Detailed breakdown")
    print("  - clip_text_alignment_reward(): CLIP alignment")
    print("  - dino_subject_consistency_reward(): DINO tracking")
    print("  - centering_reward(): Soft edge-safe centering term")

