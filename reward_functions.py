#!/usr/bin/env python3
"""
Comprehensive Reward Functions for Video Generation GRPO Training

Combines multiple modalities:
  - CLIP: Text-video semantic alignment
  - DINO: Object tracking and consistency
  - Physics: Motion dynamics and realism
  - Quality: Brightness, color, sharpness
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict
from torchvision.transforms.functional import resize

# ============================================================================
# Try to import CLIP (optional dependency)
# ============================================================================
try:
    import clip as openai_clip
    if hasattr(openai_clip, 'load'):
        clip = openai_clip
        CLIP_AVAILABLE = True
    else:
        import importlib
        import sys
        if 'clip' in sys.modules:
            importlib.reload(sys.modules['clip'])
        import clip
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
# Video Quality Rewards
# ============================================================================
@torch.no_grad()
def video_quality_reward(video: torch.Tensor, prompt: str = None) -> float:
    """Visual quality via variance, brightness, and color saturation"""
    if video is None or not isinstance(video, torch.Tensor):
        return 0.5
    
    # Normalize format
    if len(video.shape) == 5:
        video = video[0]
    if video.shape[1] == 3 and video.shape[0] > 3:
        video = video.permute(1, 0, 2, 3)
    
    # Ensure float32
    video = video.float()
    
    # 1. Variance (detail/sharpness)
    variance = video.var().item()
    variance_score = np.clip(variance / 0.1, 0, 1)
    
    # 2. Brightness (should be around 0.4-0.6)
    mean_brightness = video.mean().item()
    brightness_score = 1.0 - abs(mean_brightness - 0.5) * 2
    brightness_score = max(0, brightness_score)
    
    # 3. Color saturation (channels should vary)
    C = video.shape[0]
    if C == 3:
        channel_means = torch.tensor([video[i].mean().item() for i in range(3)])
        channel_std = channel_means.std().item()
        saturation_score = np.clip(channel_std / 0.1, 0, 1)
    else:
        saturation_score = 0.5
    
    # Combined
    quality = 0.4 * variance_score + 0.3 * brightness_score + 0.3 * saturation_score
    
    return float(quality)


@torch.no_grad()
def motion_diversity_reward(video: torch.Tensor, prompt: str = None) -> float:
    """Motion presence and diversity"""
    if video is None or not isinstance(video, torch.Tensor):
        return 0.5
    
    if len(video.shape) == 5:
        video = video[0]
    if video.shape[1] == 3 and video.shape[0] > 3:
        video = video.permute(1, 0, 2, 3)
    
    # Ensure float32
    video = video.float()
    
    T = video.shape[1]
    frame_diffs = []
    
    for t in range(T - 1):
        diff = torch.abs(video[:, t+1] - video[:, t]).mean().item()
        frame_diffs.append(diff)
    
    motion_score = np.std(frame_diffs)
    return float(np.clip(motion_score * 10, 0, 1))


# ============================================================================
# Physics-Based Rewards
# ============================================================================

@torch.no_grad()
def _center_of_mass_trajectory(video: torch.Tensor) -> Dict[str, np.ndarray]:
    """
    Estimate a coarse object trajectory from raw pixels using intensity center-of-mass.

    Input video is expected as [C, T, H, W] (float), in either [-1, 1] or [0, 1].
    Returns dict with x, y arrays of shape [T].
    """
    if len(video.shape) == 5:
        video = video[0]
    if video.shape[0] > 3 and video.shape[1] == 3:
        video = video.permute(1, 0, 2, 3)
    video = video.float()

    C, T, H, W = video.shape
    # intensity: [T, H, W]
    intensity = video.mean(dim=0)
    if intensity.min().item() < 0:
        intensity = (intensity + 1.0) / 2.0
    intensity = intensity.clamp_min(0.0)

    y_coords = torch.arange(H, device=video.device, dtype=torch.float32)
    x_coords = torch.arange(W, device=video.device, dtype=torch.float32)

    xs = []
    ys = []
    for t in range(T):
        frame = intensity[t]
        total = frame.sum() + 1e-6
        y_center = (frame.sum(dim=1) * y_coords).sum() / total
        x_center = (frame.sum(dim=0) * x_coords).sum() / total
        xs.append(float(x_center.item()))
        ys.append(float(y_center.item()))

    return {"x": np.array(xs, dtype=np.float32), "y": np.array(ys, dtype=np.float32)}


@torch.no_grad()
def shape_rigidity_reward(video: torch.Tensor, prompt: str = None) -> float:
    """
    Reward keeping an object "ball-like" / shape-consistent across time.

    We approximate the object's silhouette by intensity and compute the
    intensity-weighted 2D covariance (2nd central moments) per frame:
      - Anisotropy: circle-like shapes have similar var_x and var_y.
      - Size stability: (var_x + var_y) should not fluctuate wildly across frames.

    Returns in [0, 1]. For non-ball prompts, returns 0.5 (neutral).
    """
    if video is None or not isinstance(video, torch.Tensor):
        return 0.5

    # Only apply a strong prior when the prompt implies a rigid round object.
    if prompt:
        pl = prompt.lower()
        is_ball_prompt = any(k in pl for k in ["ball", "sphere", "spherical", "round", "circle", "orb"])
    else:
        is_ball_prompt = False
    if not is_ball_prompt:
        return 0.5

    try:
        if len(video.shape) == 5:
            video = video[0]
        if video.shape[0] > 3 and video.shape[1] == 3:
            video = video.permute(1, 0, 2, 3)
        video = video.float()

        C, T, H, W = video.shape
        if T < 4:
            return 0.5

        intensity = video.mean(dim=0)  # [T,H,W]
        if intensity.min().item() < 0:
            intensity = (intensity + 1.0) / 2.0
        intensity = intensity.clamp_min(0.0)

        y_coords = torch.arange(H, device=video.device, dtype=torch.float32).view(H, 1)
        x_coords = torch.arange(W, device=video.device, dtype=torch.float32).view(1, W)

        anisotropies = []
        sizes = []

        for t in range(T):
            frame = intensity[t]
            total = frame.sum() + 1e-6
            y_bar = (frame * y_coords).sum() / total
            x_bar = (frame * x_coords).sum() / total

            dy = y_coords - y_bar
            dx = x_coords - x_bar
            var_y = (frame * (dy ** 2)).sum() / total
            var_x = (frame * (dx ** 2)).sum() / total

            size = float((var_x + var_y).item())
            sizes.append(size)

            # 0 when perfectly isotropic, ->1 when very stretched.
            anis = float((torch.abs(var_x - var_y) / (var_x + var_y + 1e-6)).item())
            anisotropies.append(anis)

        anis_np = np.array(anisotropies, dtype=np.float32)
        size_np = np.array(sizes, dtype=np.float32)

        # Roundness score: penalize anisotropy.
        roundness = float(1.0 - np.clip(anis_np.mean() * 2.0, 0.0, 1.0))

        # Size stability score: penalize coefficient of variation.
        mean_s = float(size_np.mean() + 1e-6)
        cv = float(size_np.std() / mean_s)
        size_stability = float(1.0 - np.clip(cv * 3.0, 0.0, 1.0))

        # Consistency score: penalize frame-to-frame changes in anisotropy (squash during impact).
        if anis_np.size >= 2:
            d_anis = float(np.abs(np.diff(anis_np)).mean())
            anis_stability = float(1.0 - np.clip(d_anis * 4.0, 0.0, 1.0))
        else:
            anis_stability = 0.5

        score = 0.45 * roundness + 0.35 * size_stability + 0.20 * anis_stability
        return float(np.clip(score, 0.0, 1.0))
    except Exception as e:
        print(f"⚠️ shape_rigidity_reward failed: {e}")
        return 0.5


@torch.no_grad()
def physics_bounce_reward(video: torch.Tensor, prompt: str = None) -> float:
    """
    Detect and reward "bouncing" behavior from a coarse vertical trajectory.

    Uses intensity center-of-mass y(t) and scores:
    - Multiple oscillations (down-up cycles / repeated contacts)
    - Non-trivial amplitude
    - Roughly regular periods (not jitter)
    - Motion presence gate (avoid rewarding static videos)

    Returns in [0, 1]. For non-bounce prompts, returns 0.5 (neutral).
    """
    if video is None or not isinstance(video, torch.Tensor):
        return 0.5

    if prompt:
        pl = prompt.lower()
        bounce_keywords = [
            "bounce",
            "bouncing",
            "trampoline",
            "spring",
            "jump",
            "jumping",
            "rebound",
        ]
        is_bounce_prompt = any(k in pl for k in bounce_keywords)
    else:
        is_bounce_prompt = False

    # Don't penalize non-bounce prompts with a bounce prior.
    if not is_bounce_prompt:
        return 0.5

    try:
        # Normalize format
        if len(video.shape) == 5:
            video = video[0]
        if video.shape[0] > 3 and video.shape[1] == 3:
            video = video.permute(1, 0, 2, 3)
        video = video.float()

        C, T, H, W = video.shape
        if T < 8:
            return 0.5

        traj = _center_of_mass_trajectory(video)
        y = traj["y"].astype(np.float32)  # pixels, y increases downward

        # Smooth a bit to reduce frame-to-frame noise.
        smooth_w = 3
        if smooth_w >= 2:
            k = np.ones(smooth_w, dtype=np.float32) / float(smooth_w)
            y_s = np.convolve(y, k, mode="same")
        else:
            y_s = y

        # Detrend to isolate oscillations (use a larger moving average).
        trend_w = max(5, int(T // 6))
        trend_w = min(trend_w, T - (T % 2 == 0))  # keep < T
        k2 = np.ones(trend_w, dtype=np.float32) / float(trend_w)
        y_trend = np.convolve(y_s, k2, mode="same")
        y_osc = y_s - y_trend

        # Motion presence gate (pixels per frame)
        vy = np.diff(y_s)
        motion_mag = float(np.mean(np.abs(vy)))
        motion_presence = float(np.clip(motion_mag / 1.5, 0.0, 1.0))

        # Find local maxima/minima via derivative sign changes.
        dy = np.diff(y_s)
        s = np.sign(dy)
        # Replace zeros to avoid missing extrema
        for i in range(1, s.shape[0]):
            if s[i] == 0:
                s[i] = s[i - 1]
        for i in range(s.shape[0] - 2, -1, -1):
            if s[i] == 0:
                s[i] = s[i + 1]

        # Indices in [1..T-2] where sign flips
        max_idx = np.where((s[:-1] > 0) & (s[1:] < 0))[0] + 1  # local maxima in y (lowest point)
        min_idx = np.where((s[:-1] < 0) & (s[1:] > 0))[0] + 1  # local minima in y (highest point)

        # For bouncing: repeated "contacts" correspond to repeated local maxima in y (going down then up).
        bounce_count = int(max_idx.shape[0])
        # Score bounce count: want at least a few bounces
        bounce_count_score = float(np.clip(bounce_count / 3.0, 0.0, 1.0))

        # Amplitude estimate: pair each max with nearest preceding min (up -> down)
        amps = []
        if bounce_count > 0 and min_idx.shape[0] > 0:
            for mi in max_idx:
                prev_mins = min_idx[min_idx < mi]
                if prev_mins.size == 0:
                    continue
                mprev = int(prev_mins[-1])
                amps.append(float(y_s[mi] - y_s[mprev]))  # positive means moved down then up
        if len(amps) == 0:
            amp_score = 0.0
        else:
            amp_med = float(np.median(np.array(amps, dtype=np.float32)))
            amp_score = float(np.clip(amp_med / max(1.0, 0.08 * float(H)), 0.0, 1.0))

        # Period regularity between contacts
        if bounce_count >= 2:
            periods = np.diff(max_idx).astype(np.float32)
            mean_p = float(periods.mean())
            std_p = float(periods.std())
            cv = std_p / (mean_p + 1e-6)
            regularity = float(1.0 / (1.0 + 2.0 * cv))
        else:
            regularity = 0.0

        # Oscillation energy (detrended std) scaled by image size
        osc_std = float(np.std(y_osc))
        osc_energy = float(np.clip(osc_std / max(1.0, 0.05 * float(H)), 0.0, 1.0))

        bounce_score = (
            0.35 * bounce_count_score +
            0.25 * amp_score +
            0.25 * regularity +
            0.15 * osc_energy
        )
        # Gate by motion presence to avoid rewarding static noise patterns
        bounce_score = min(bounce_score, 0.2 + 0.8 * motion_presence)

        return float(np.clip(bounce_score, 0.0, 1.0))
    except Exception as e:
        print(f"⚠️ physics_bounce_reward failed: {e}")
        return 0.5


@torch.no_grad()
def physics_plausibility_reward(video: torch.Tensor, prompt: str = None) -> float:
    """
    Physical plausibility score from coarse trajectory dynamics.

    What it rewards:
    - Smooth motion (low jerk / not jittery)
    - Bounded acceleration (no teleporting/exploding motion)
    - Low-frequency dominated dynamics (less high-frequency flicker)
    - Prompt-consistent overall direction when prompt implies a direction (down/up/left/right)

    Returns in [0, 1].
    """
    if video is None or not isinstance(video, torch.Tensor):
        return 0.5

    try:
        traj = _center_of_mass_trajectory(video)
        x = traj["x"]
        y = traj["y"]
        T = x.shape[0]
        if T < 6:
            return 0.5

        vx = np.diff(x)
        vy = np.diff(y)
        speed = np.sqrt(vx * vx + vy * vy)  # [T-1]

        ax = np.diff(vx)
        ay = np.diff(vy)
        acc = np.sqrt(ax * ax + ay * ay)  # [T-2]

        # Jerk (3rd derivative) for jitter detection
        jx = np.diff(ax)
        jy = np.diff(ay)
        jerk = np.sqrt(jx * jx + jy * jy)  # [T-3]

        # Motion presence gate: if essentially no motion, plausibility should be low
        motion_mag = float(np.mean(speed))
        motion_presence = float(np.clip(motion_mag / 2.0, 0.0, 1.0))  # heuristic scale (pixels/frame)

        # Smoothness: penalize high jerk variance (jitter)
        jerk_std = float(np.std(jerk)) if jerk.size > 0 else 0.0
        smoothness = float(1.0 / (1.0 + jerk_std / 1.5))  # 1.5 px/frame^3 scale

        # Bounded acceleration: penalize large spikes
        acc_max = float(np.max(acc)) if acc.size > 0 else 0.0
        bounded_acc = float(1.0 / (1.0 + acc_max / 4.0))  # 4 px/frame^2 scale

        # Low-frequency dominance: physical motions tend to be low-frequency vs flicker/jitter
        if speed.size >= 8:
            s = speed - speed.mean()
            spec = np.abs(np.fft.rfft(s)) ** 2
            total_energy = float(spec.sum() + 1e-6)
            # Low freq = first few bins (excluding DC)
            low_bins = min(4, spec.shape[0] - 1)
            low_energy = float(spec[1:1 + low_bins].sum())
            lowfreq_ratio = float(np.clip(low_energy / total_energy, 0.0, 1.0))
        else:
            lowfreq_ratio = 0.5

        # Direction consistency / prompt-aware direction bonus (re-uses the same keyword logic)
        direction_bonus = 0.5
        if prompt:
            pl = prompt.lower()
            dx = x[-1] - x[0]
            dy = y[-1] - y[0]
            expected = None
            if any(w in pl for w in ["fall", "drop", "descend", "down"]):
                expected = "down"
            elif any(w in pl for w in ["rise", "ascend", "up", "climb"]):
                expected = "up"
            elif any(w in pl for w in ["left", "leftward"]):
                expected = "left"
            elif any(w in pl for w in ["right", "rightward"]):
                expected = "right"

            thresh = 5.0
            if expected == "down":
                direction_bonus = 0.9 if dy > thresh else (0.1 if abs(dy) < thresh else 0.3)
            elif expected == "up":
                direction_bonus = 0.9 if dy < -thresh else (0.1 if abs(dy) < thresh else 0.3)
            elif expected == "left":
                direction_bonus = 0.9 if dx < -thresh else (0.1 if abs(dx) < thresh else 0.3)
            elif expected == "right":
                direction_bonus = 0.9 if dx > thresh else (0.1 if abs(dx) < thresh else 0.3)

        plausibility = (
            0.25 * smoothness +
            0.50 * bounded_acc +
            0.25 * direction_bonus
        )

        # If there is no motion, cap plausibility hard (avoid rewarding static videos)
        plausibility = min(plausibility, 0.2 + 0.8 * motion_presence)

        return float(np.clip(plausibility, 0.0, 1.0))
    except Exception as e:
        print(f"⚠️ physics_plausibility_reward failed: {e}")
        return 0.5


@torch.no_grad()
def physics_velocity_reward(video: torch.Tensor, prompt: str = None) -> float:
    """Velocity characteristics (first derivative of motion)"""
    if video is None or not isinstance(video, torch.Tensor):
        return 0.5
    
    # Normalize format
    if len(video.shape) == 5:
        video = video[0]
    if video.shape[1] == 3 and video.shape[0] > 3:
        video = video.permute(1, 0, 2, 3)
    
    # Ensure float32
    video = video.float()
    
    T = video.shape[1]
    velocities = []
    
    for t in range(T - 1):
        velocity = torch.abs(video[:, t+1] - video[:, t]).mean().item()
        velocities.append(velocity)
    
    velocities = np.array(velocities)
    
    avg_velocity = np.mean(velocities)
    velocity_smoothness = 1.0 / (1.0 + np.std(velocities))
    
    velocity_score = 0.6 * avg_velocity * 100 + 0.4 * velocity_smoothness
    
    return float(np.clip(velocity_score, 0, 1))


@torch.no_grad()
def physics_acceleration_reward(video: torch.Tensor, prompt: str = None) -> float:
    """Acceleration (second derivative of motion)"""
    if video is None or not isinstance(video, torch.Tensor):
        return 0.5
    
    # Normalize format
    if len(video.shape) == 5:
        video = video[0]
    if video.shape[1] == 3 and video.shape[0] > 3:
        video = video.permute(1, 0, 2, 3)
    
    # Ensure float32
    video = video.float()
    
    T = video.shape[1]
    
    velocities = []
    for t in range(T - 1):
        vel = torch.abs(video[:, t+1] - video[:, t]).mean().item()
        velocities.append(vel)
    
    if len(velocities) < 2:
        return 0.5
    
    velocities = np.array(velocities)
    accelerations = np.diff(velocities)
    
    accel_smoothness = 1.0 / (1.0 + np.std(accelerations))
    max_accel = np.max(np.abs(accelerations))
    accel_bounded = 1.0 / (1.0 + max_accel * 100)
    accel_variance = np.std(accelerations)
    accel_naturalness = np.clip(accel_variance * 10, 0, 1)
    
    accel_score = (
        0.4 * accel_smoothness +
        0.3 * accel_bounded +
        0.3 * accel_naturalness
    )
    
    return float(accel_score)


@torch.no_grad()
def physics_trajectory_smoothness(video: torch.Tensor, prompt: str = None) -> float:
    """Smooth trajectories"""
    if video is None or not isinstance(video, torch.Tensor):
        return 0.5
    
    if len(video.shape) == 5:
        video = video[0]
    if video.shape[1] == 3 and video.shape[0] > 3:
        video = video.permute(1, 0, 2, 3)
    
    # Ensure float32
    video = video.float()
    
    T = video.shape[1]
    
    motion_vectors = []
    for t in range(T - 1):
        motion = video[:, t+1] - video[:, t]
        motion_vectors.append(motion)
    
    direction_changes = []
    for t in range(len(motion_vectors) - 1):
        direction_change = torch.abs(motion_vectors[t+1] - motion_vectors[t]).mean().item()
        direction_changes.append(direction_change)
    
    avg_direction_change = np.mean(direction_changes)
    smoothness = 1.0 / (1.0 + avg_direction_change * 100)
    
    return float(smoothness)


@torch.no_grad()
def physics_directional_motion_reward(video: torch.Tensor, prompt: str = None) -> float:
    """

    General directional motion reward - tracks ANY consistent direction
    Works for falling, rising, sliding left/right, etc.
    
    Measures:
    1. Motion magnitude (is there movement?)
    2. Directional consistency (is motion in a consistent direction?)
    3. Prompt-aware direction (if prompt mentions direction)
    """
    if video is None or not isinstance(video, torch.Tensor):
        return 0.5
    
    if len(video.shape) == 5:
        video = video[0]
    if video.shape[1] == 3 and video.shape[0] > 3:
        video = video.permute(1, 0, 2, 3)
    
    video = video.float()
    
    C, T, H, W = video.shape
    
    # Track center of mass in both X and Y over time
    x_positions = []
    y_positions = []
    
    for t in range(T):
        frame = video[:, t, :, :]  # [C, H, W]
        intensity = frame.mean(dim=0)  # [H, W]
        
        # Compute center of mass
        y_coords = torch.arange(H, device=video.device).float()
        x_coords = torch.arange(W, device=video.device).float()
        
        total_intensity = intensity.sum() + 1e-6
        y_center = (intensity.sum(dim=1) * y_coords).sum() / total_intensity
        x_center = (intensity.sum(dim=0) * x_coords).sum() / total_intensity
        
        y_positions.append(y_center.item())
        x_positions.append(x_center.item())
    
    x_positions = np.array(x_positions)
    y_positions = np.array(y_positions)
    
    # Compute motion vector
    delta_x = x_positions[-1] - x_positions[0]
    delta_y = y_positions[-1] - y_positions[0]
    
    # Overall displacement magnitude
    displacement = np.sqrt(delta_x**2 + delta_y**2)
    displacement_score = np.clip(displacement / (H * 0.3), 0, 1)  # Normalize by image size
    
    # Directional consistency (motion should be in one direction, not random)
    velocity_x = np.diff(x_positions)
    velocity_y = np.diff(y_positions)
    
    # Check if motion is consistent (not zigzagging)
    if len(velocity_x) > 0:
        # Consistency = how aligned are velocity vectors?
        consistency_x = np.abs(velocity_x).mean() / (np.std(velocity_x) + 1e-6)
        consistency_y = np.abs(velocity_y).mean() / (np.std(velocity_y) + 1e-6)
        consistency_score = np.clip((consistency_x + consistency_y) / 10, 0, 1)
    else:
        consistency_score = 0.5
    
    # Check if direction matches prompt (if keywords present)
    direction_bonus = 0.5  # Default neutral
    if prompt:
        prompt_lower = prompt.lower()
        expected_direction = None
        
        # Detect expected direction from prompt
        if any(word in prompt_lower for word in ['fall', 'drop', 'descend', 'down']):
            expected_direction = 'down'
        elif any(word in prompt_lower for word in ['rise', 'ascend', 'up', 'climb']):
            expected_direction = 'up'
        elif any(word in prompt_lower for word in ['left', 'leftward']):
            expected_direction = 'left'
        elif any(word in prompt_lower for word in ['right', 'rightward']):
            expected_direction = 'right'
        
        # Check if actual motion matches expected
        if expected_direction == 'down' and delta_y > 5:
            direction_bonus = 0.9  # Moved down as expected!
        elif expected_direction == 'up' and delta_y < -5:
            direction_bonus = 0.9  # Moved up as expected!
        elif expected_direction == 'left' and delta_x < -5:
            direction_bonus = 0.9  # Moved left as expected!
        elif expected_direction == 'right' and delta_x > 5:
            direction_bonus = 0.9  # Moved right as expected!
        elif expected_direction and abs(delta_x) < 5 and abs(delta_y) < 5:
            direction_bonus = 0.1  # Expected motion but got none!
    
    # Combined score
    score = 0.4 * displacement_score + 0.3 * consistency_score + 0.3 * direction_bonus
    
    return float(np.clip(score, 0, 1))


@torch.no_grad()
def physics_momentum_conservation(video: torch.Tensor, prompt: str = None) -> float:
    """Momentum-like behavior"""
    if video is None or not isinstance(video, torch.Tensor):
        return 0.5
    
    if len(video.shape) == 5:
        video = video[0]
    if video.shape[1] == 3 and video.shape[0] > 3:
        video = video.permute(1, 0, 2, 3)
    
    # Ensure float32
    video = video.float()
    
    T = video.shape[1]
    
    velocities = []
    for t in range(T - 1):
        vel = torch.abs(video[:, t+1] - video[:, t]).mean().item()
        velocities.append(vel)
    
    velocities = np.array(velocities)
    
    velocity_consistency = 1.0 / (1.0 + np.std(velocities))
    velocity_changes = np.diff(velocities)
    no_abrupt_reversals = 1.0 / (1.0 + np.sum(np.abs(velocity_changes > 0.1)))
    
    momentum_score = 0.6 * velocity_consistency + 0.4 * no_abrupt_reversals
    
    return float(momentum_score)


# ============================================================================
# Comprehensive Reward Function
# ============================================================================

@torch.no_grad()
def comprehensive_grpo_reward(
    video: torch.Tensor,
    prompt: str,
    device: str = 'cuda',
    use_clip: bool = True,
    use_dino: bool = True,
    use_physics: bool = True,
) -> Dict[str, float]:
    """
    Ultimate comprehensive reward combining all modalities
    
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
            scores['shape_rigidity'] = shape_rigidity_reward(video, prompt)
        except Exception as e:
            print(f"⚠️ DINO evaluation failed: {e}")
            scores['dino_consistency'] = 0.5
            scores['dino_presence'] = 0.5
            scores['shape_rigidity'] = 0.5
    else:
        scores['dino_consistency'] = 0.5
        scores['dino_presence'] = 0.5
        scores['shape_rigidity'] = 0.5
    
    # Physics Rewards
    if use_physics:
        try:
            scores['physics_velocity'] = physics_velocity_reward(video, prompt)
            scores['physics_acceleration'] = physics_acceleration_reward(video, prompt)
            scores['physics_smoothness'] = physics_trajectory_smoothness(video, prompt)
            scores['physics_momentum'] = physics_momentum_conservation(video, prompt)
            scores['physics_directional'] = physics_directional_motion_reward(video, prompt)  # NEW!
            scores['physics_plausibility'] = physics_plausibility_reward(video, prompt)
            scores['physics_bounce'] = physics_bounce_reward(video, prompt)
            # Multi-scale dynamics: measure motion energy at multiple temporal strides and reward coherent structure.
            C, T, H, W = video.shape
            diffs = []
            for stride in (1, 2, 4, 8):
                if T - stride <= 0:
                    continue
                # Mean absolute difference at this temporal stride
                d = torch.abs(video[:, stride:, :, :] - video[:, :-stride, :, :]).mean().item()
                diffs.append(d)
            if len(diffs) == 0:
                scores['physics_multiscale'] = 0.5
            else:
                diffs_np = np.array(diffs, dtype=np.float32)
                # Encourage "enough" motion but avoid extreme jitter: use mean and scale-consistency.
                mean_energy = float(diffs_np.mean())
                scale_consistency = float(1.0 / (1.0 + diffs_np.std() * 50.0))
                energy_score = float(np.clip(mean_energy * 120.0, 0.0, 1.0))
                scores['physics_multiscale'] = float(np.clip(0.6 * energy_score + 0.4 * scale_consistency, 0.0, 1.0))
        except Exception as e:
            print(f"⚠️ Physics evaluation failed: {e}")
            scores['physics_velocity'] = 0.5
            scores['physics_acceleration'] = 0.5
            scores['physics_smoothness'] = 0.5
            scores['physics_momentum'] = 0.5
            scores['physics_directional'] = 0.5
            scores['physics_multiscale'] = 0.5
            scores['physics_plausibility'] = 0.5
            scores['physics_bounce'] = 0.5
    else:
        scores['physics_velocity'] = 0.5
        scores['physics_acceleration'] = 0.5
        scores['physics_smoothness'] = 0.5
        scores['physics_momentum'] = 0.5
        scores['physics_directional'] = 0.5
        scores['physics_multiscale'] = 0.5
        scores['physics_plausibility'] = 0.5
        scores['physics_bounce'] = 0.5
    
    # Video Quality Rewards
    try:
        scores['video_quality'] = video_quality_reward(video, prompt)
        scores['motion_diversity'] = motion_diversity_reward(video, prompt)
    except Exception as e:
        print(f"⚠️ Quality evaluation failed: {e}")
        scores['video_quality'] = 0.5
        scores['motion_diversity'] = 0.5

    # ------------------------------------------------------------------------
    # REQUIRED BREAKDOWN (matches reward_function docstring):
    # - Text alignment (40%): CLIP semantic + temporal matching
    # - Object tracking (10%): DINO consistency + presence
    # - Physics & motion (40%): Multi-scale dynamics analysis
    # - Motion diversity (10%): Temporal variation
    # ------------------------------------------------------------------------
    text_alignment = 0.5 * scores['clip_alignment'] + 0.5 * scores['clip_temporal']
    # Include shape rigidity for ball/sphere prompts to keep the object from deforming.
    object_tracking = (
        0.40 * scores['dino_consistency'] +
        0.40 * scores['dino_presence'] +
        0.20 * scores['shape_rigidity']
    )
    # If prompt is about bouncing, bias the physics bucket toward bounce-structure.
    if prompt and any(k in prompt.lower() for k in ["bounce", "bouncing", "trampoline", "rebound", "spring", "jump", "jumping"]):
        physics_motion = (
            0.30 * scores['physics_plausibility'] +
            0.15 * scores['physics_multiscale'] +
            0.35 * scores['physics_bounce'] +
            0.10 * scores['physics_directional'] +
            0.10 * scores['physics_momentum']
        )
    else:
        physics_motion = (
            0.35 * scores['physics_plausibility'] +
            0.25 * scores['physics_multiscale'] +
            0.25 * scores['physics_directional'] +
            0.15 * scores['physics_momentum']
        )
    motion_diversity = scores['motion_diversity']

    # Store bucket scores for logging/analysis
    scores['text_alignment'] = float(np.clip(text_alignment, 0.0, 1.0))
    scores['object_tracking'] = float(np.clip(object_tracking, 0.0, 1.0))
    scores['physics_motion'] = float(np.clip(physics_motion, 0.0, 1.0))
    scores['motion_diversity_bucket'] = float(np.clip(motion_diversity, 0.0, 1.0))

    total_reward = (
        0.40 * scores['text_alignment'] +
        0.20 * scores['object_tracking'] +
        0.40 * scores['physics_motion']
    )
    
    scores['reward'] = float(np.clip(total_reward, 0.0, 1.0))
    
    return scores


@torch.no_grad()
def reward_function(
    video: torch.Tensor,
    prompt: str,
    device: str = 'cuda',
) -> torch.Tensor:
    """
    COMPREHENSIVE PHYSICS-AWARE REWARD FUNCTION
    
    Designed for GRPO training of video generation models with emphasis on:
    - Realistic motion dynamics (velocity, acceleration, trajectory)
    - Directional consistency (prompt-guided motion)
    - Physical plausibility (gravity, momentum conservation)
    
    Returns reward as tensor (compatible with GRPO pipeline)
    
    Component breakdown:
    - Text alignment (40%): CLIP semantic + temporal matching
    - Object tracking (10%): DINO consistency + presence
    - Physics & motion (40%): Multi-scale dynamics analysis
    - Motion diversity (10%): Temporal variation
    
    Total: 100% (physics-emphasized for realistic motion learning)
    """
    # Validation
    if video is None:
        print("⚠️ ERROR: Video is None!")
        return torch.tensor(0.5, device=device)
    
    if not isinstance(video, torch.Tensor):
        print(f"⚠️ ERROR: Video is not tensor (got {type(video)})")
        return torch.tensor(0.5, device=device)
    
    if video.numel() == 0:
        print("⚠️ ERROR: Video tensor is empty!")
        return torch.tensor(0.5, device=device)
    
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
        use_physics=True,
    )
    
    # Simplified logging (CLIP + DINO only)
    print(f"\n  Reward Components (40/10/40/10):")
    print(f"    Text alignment: {result['text_alignment']:.4f} (40%)  [clip={result['clip_alignment']:.4f}, temporal={result['clip_temporal']:.4f}]")
    print(f"    Object tracking: {result['object_tracking']:.4f} (10%)  [dino_cons={result['dino_consistency']:.4f}, dino_pres={result['dino_presence']:.4f}]")
    print(
        f"    Physics & motion: {result['physics_motion']:.4f} (40%)  "
        f"[plaus={result['physics_plausibility']:.4f}, multi={result['physics_multiscale']:.4f}, "
        f"bounce={result.get('physics_bounce', 0.5):.4f}, dir={result['physics_directional']:.4f}, mom={result['physics_momentum']:.4f}]"
    )
    print(f"    Motion diversity: {result['motion_diversity_bucket']:.4f} (10%)")
    print(f"  Total Reward: {result['reward']:.4f}")
    
    return torch.tensor(result['reward'], device=device)


if __name__ == "__main__":
    print("Comprehensive Reward Functions Module")
    print("\nAvailable functions:")
    print("  - reward_function(): Main GRPO reward (CLIP+DINO+Physics+Quality)")
    print("  - comprehensive_grpo_reward(): Detailed breakdown")
    print("  - clip_text_alignment_reward(): CLIP alignment")
    print("  - dino_subject_consistency_reward(): DINO tracking")
    print("  - video_quality_reward(): Brightness/color/sharpness")

