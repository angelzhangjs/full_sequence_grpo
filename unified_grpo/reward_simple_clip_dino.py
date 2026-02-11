#!/usr/bin/env python3
"""
SIMPLIFIED Reward Function - CLIP + DINO ONLY
"""

import torch
import torch.nn.functional as F
import numpy as np

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

@torch.no_grad()
def clip_score(video, prompt, device='cuda'):
    """CLIP text-video alignment"""
    if not CLIP_AVAILABLE:
        return 0.5
    
    model = load_clip_model(device)
    video = video.float()
    
    # Handle format
    if len(video.shape) == 5:
        video = video[0]
    if video.shape[0] == 3:  # [C, T, H, W]
        video = video.permute(1, 0, 2, 3)  # → [T, C, H, W]
    
    T = video.shape[0]
    sample_indices = torch.linspace(0, T-1, min(8, T)).long()
    
    # Encode text
    text_tokens = clip.tokenize([prompt]).to(device)
    text_features = model.encode_text(text_tokens)
    text_features = F.normalize(text_features, dim=-1)
    
    # Encode frames
    scores = []
    from torchvision.transforms.functional import resize
    for idx in sample_indices:
        frame = video[idx]
        if frame.min() < -0.1:
            frame = (frame + 1) / 2
        
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
def dino_consistency_score(video, device='cuda'):
    """DINO object tracking consistency"""
    model = load_dino_model(device)
    video = video.float()
    
    # Handle format
    if len(video.shape) == 5: # [B, C, T, H, W]
        video = video[0]
    if video.shape[0] == 3: # [C, T, H, W]
        video = video.permute(1, 0, 2, 3)
    
    T = video.shape[0] # [T, C, H, W]
    sample_indices = torch.linspace(0, T-1, min(8, T)).long()
    
    # DINO transform
    from torchvision.transforms.functional import resize
    features = []
    for idx in sample_indices:
        frame = video[idx]
        if frame.min() < -0.1:
            frame = (frame + 1) / 2
        
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
    
    return float(np.mean(similarities))

def comprehensive_grpo_reward(video, prompt, device='cuda', **kwargs):
    """
    SIMPLIFIED: CLIP + DINO ONLY
    
    Returns dict with component scores
    """
    scores = {}
    
    # CLIP alignment
    try:
        scores['clip_alignment'] = clip_score(video, prompt, device)
        scores['clip_temporal'] = clip_score(video, prompt, device)  # Same for now
    except Exception as e:
        print(f"⚠️ CLIP error: {e}")
        scores['clip_alignment'] = 0.5
        scores['clip_temporal'] = 0.5
    
    # DINO tracking only (removed buggy motion component!)
    try:
        scores['dino_consistency'] = dino_consistency_score(video, device)
    except Exception as e:
        print(f"⚠️ DINO error: {e}")
        scores['dino_consistency'] = 0.5
    
    # CLEAN: Just CLIP 60% + DINO 40%
    total_reward = (
        1 * scores['clip_alignment'] +     # Text match (main signal!)
        0 * scores['dino_consistency']     # Object tracking
    )
    
    scores['reward'] = float(total_reward)
    
    # Clean breakdown
    print(f"\n  Reward (CLIP + DINO):")
    print(f"    CLIP: {scores['clip_alignment']:.4f} (60%)")
    print(f"    DINO: {scores['dino_consistency']:.4f} (40%)")
    print(f"  Total: {scores['reward']:.4f}")
    
    return scores
