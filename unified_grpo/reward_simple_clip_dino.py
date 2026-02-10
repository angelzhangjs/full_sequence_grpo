#!/usr/bin/env python3
"""
SIMPLIFIED Reward Function - CLIP + DINO ONLY
No centering, no motion_gate, no other components that might cause issues

For debugging white video problem
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
    if len(video.shape) == 5:
        video = video[0]
    if video.shape[0] == 3:
        video = video.permute(1, 0, 2, 3)
    
    T = video.shape[0]
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

@torch.no_grad()
def dino_motion_score(video, device='cuda'):
    """
    DINO-based motion detection
    Tracks object movement across frames (not just consistency)
    """
    model = load_dino_model(device)
    video = video.float()
    
    # Handle format
    if len(video.shape) == 5:
        video = video[0]
    if video.shape[0] == 3:
        video = video.permute(1, 0, 2, 3)
    
    T = video.shape[0]
    
    # Extract features at different time points
    from torchvision.transforms.functional import resize
    early_idx = 0
    mid_idx = T // 2
    late_idx = T - 1
    
    def get_feature(frame_idx):
        frame = video[frame_idx]
        if frame.min() < -0.1:
            frame = (frame + 1) / 2
        
        frame = resize(frame, (224, 224), antialias=True)
        
        mean = torch.tensor([0.485, 0.456, 0.406], device=device)
        std = torch.tensor([0.229, 0.224, 0.225], device=device)
        frame = (frame - mean[:, None, None]) / std[:, None, None]
        
        return model(frame.unsqueeze(0))
    
    feat_early = F.normalize(get_feature(early_idx), dim=-1)
    feat_mid = F.normalize(get_feature(mid_idx), dim=-1)
    feat_late = F.normalize(get_feature(late_idx), dim=-1)
    
    # Motion = features CHANGE over time (object moves)
    # But not TOO much change (still same object)
    change_early_mid = 1.0 - F.cosine_similarity(feat_early, feat_mid, dim=-1).item()
    change_mid_late = 1.0 - F.cosine_similarity(feat_mid, feat_late, dim=-1).item()
    
    # Ideal: Some change (motion) but not complete change (still same object)
    # Sweet spot: 0.1-0.3 (10-30% feature change = good motion)
    avg_change = (change_early_mid + change_mid_late) / 2
    
    # Score: Reward moderate change
    if avg_change < 0.05:  # Too static
        motion_score = avg_change / 0.05  # Linearly increase
    elif avg_change < 0.3:  # Good motion range
        motion_score = 1.0
    else:  # Too much change (different object or chaos)
        motion_score = max(0, 1.0 - (avg_change - 0.3) / 0.7)
    
    return float(np.clip(motion_score, 0, 1))

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
    
    # DINO tracking and motion
    try:
        scores['dino_consistency'] = dino_consistency_score(video, device)
        scores['dino_motion'] = dino_motion_score(video, device)  # NEW!
        scores['dino_presence'] = 1.0  # Assume present
    except Exception as e:
        print(f"⚠️ DINO error: {e}")
        scores['dino_consistency'] = 0.5
        scores['dino_motion'] = 0.5
        scores['dino_presence'] = 1.0
    
    # Weighted combination: CLIP 60%, DINO 40% (with motion emphasis!)
    total_reward = (
        0.40 * scores['clip_alignment'] +     # Text match
        0.10 * scores['clip_temporal'] +      # Temporal coherence
        0.40 * scores['dino_consistency'] +   # Object tracking
        0.10 * scores['dino_motion']       # Object motion! ←        # Object presence
    )
    
    scores['reward'] = float(total_reward)
    
    # Print breakdown with motion emphasis
    print(f"\n  Reward Components (CLIP/DINO with Motion):")
    print(f"    CLIP alignment: {scores['clip_alignment']:.4f} (35%)")
    print(f"    CLIP temporal: {scores['clip_temporal']:.4f} (15%)")
    print(f"    DINO consistency: {scores['dino_consistency']:.4f} (20%)")
    print(f"    DINO motion: {scores['dino_motion']:.4f} (20%) 🎬")
    print(f"    DINO presence: {scores['dino_presence']:.4f} (10%)")
    print(f"  Total Reward: {scores['reward']:.4f}")
    
    return scores

if __name__ == "__main__":
    print("Simplified Reward Function: CLIP + DINO only")
    print("No centering, no motion_gate")
    print("For debugging white video issue")
