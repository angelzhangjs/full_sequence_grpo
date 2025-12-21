#!/usr/bin/env python3
"""
Helper functions for LTX-Video Pipeline
Includes video decoding and physics-based reward function using DINO
"""
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from typing import Dict
from ltx_video.models.autoencoders.vae_encode import vae_decode

# ============================================================================
# Video Decoding Helper
# ============================================================================
def decode_x0_to_video(
    x0_est,
    pipeline,
    num_frames: int,
    height: int,
    width: int,
    is_patchified: bool = True,
):
    """
    Decode x0 latent prediction to video tensor.
    
    Args:
        x0_est: Estimated x0 latents (patchified or unpatchified)
        pipeline: LTX-Video pipeline object
        num_frames: Number of video frames
        height: Target video height (in pixel space)
        width: Target video width (in pixel space)
        is_patchified: Whether x0_est is in patchified format
        
    Returns:
        video_tensor: Decoded video as tensor [1, num_frames, 3, height, width]
    """
    with torch.no_grad():
        if is_patchified:
            # Calculate latent space dimensions (VAE downsamples by vae_scale_factor)
            vae_scale_factor = pipeline.vae_scale_factor
            latent_height = height // vae_scale_factor
            latent_width = width // vae_scale_factor
            
            # Unpatchify back to latent shape
            # Note: unpatchify expects latent space dimensions, which it will divide by patch_size
            x0_unpatchified = pipeline.patchifier.unpatchify(
                x0_est,
                output_height=latent_height * pipeline.patchifier.patch_size[1],
                output_width=latent_width * pipeline.patchifier.patch_size[2],
                out_channels=pipeline.vae.config.latent_channels,
            )
        else:
            x0_unpatchified = x0_est
            
        # Prepare timestep for VAE decoder if needed
        if pipeline.vae.decoder.timestep_conditioning:
            # For x0 predictions (clean images), use timestep=0.0
            decode_timestep = torch.tensor([0.0], device=x0_unpatchified.device)
        else:
            decode_timestep = None
            
        # Decode through VAE
        video = vae_decode(
            x0_unpatchified,
            pipeline.vae,
            is_video=True,
            vae_per_channel_normalize=True,
            timestep=decode_timestep,
        )
        # Rearrange from [batch, channels, frames, H, W] to [batch, frames, channels, H, W]
        # This is the format expected by the reward function
        video = video.permute(0, 2, 1, 3, 4)
        
    return video  # Shape: [batch, num_frames, channels, height, width]

# ============================================================================
# DINO Feature Extractor (Lazy Loading)
# ============================================================================
class DINOFeatureExtractor:
    """
    Wrapper for DINOv2 feature extraction.
    Lazy loads the model on first use to avoid loading during import.
    """
    _model = None
    _transform = None
    
    @classmethod
    def get_model(cls):
        """Load DINOv2 model (lazy loading)"""
        if cls._model is None:
            print("Loading DINOv2 model...")
            # Using dinov2_vits14 (small) for faster inference
            # Options: dinov2_vits14, dinov2_vitb14, dinov2_vitl14, dinov2_vitg14
            cls._model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
            cls._model = cls._model.cuda()
            cls._model.eval()
            
            # DINO expects images normalized with ImageNet stats
            cls._transform = transforms.Compose([
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            print("✅ DINOv2 model loaded!")
            
        return cls._model, cls._transform
    
    @classmethod
    def extract_features(cls, frames: torch.Tensor) -> torch.Tensor:
        """
        Extract DINO features from video frames.
        
        Args:
            frames: Video frames [batch, num_frames, 3, H, W] in range [0, 1]
            
        Returns:
            features: DINO features [batch, num_frames, feature_dim]
        """
        model, transform = cls.get_model()
        
        batch_size, num_frames, C, H, W = frames.shape
        
        # Convert to float32 (DINO requires float32, not bfloat16)
        frames = frames.float()
        
        # Reshape to process all frames at once
        frames_flat = frames.view(-1, C, H, W)  # [batch*num_frames, 3, H, W]
        
        # Resize to DINO input size (224x224)
        frames_resized = F.interpolate(
            frames_flat, 
            size=(224, 224), 
            mode='bilinear', 
            align_corners=False
        )
        
        # Normalize for DINO
        frames_normalized = transform(frames_resized)
        
        # Extract features
        with torch.no_grad():
            features = model(frames_normalized)  # [batch*num_frames, feature_dim]
        
        # Reshape back to video format
        feature_dim = features.shape[-1]
        features = features.view(batch_size, num_frames, feature_dim)
        
        return features

# ============================================================================
# Physics Property Extractors
# ============================================================================
def compute_optical_flow(video: torch.Tensor) -> torch.Tensor:
    """
    Compute optical flow between consecutive frames using Farneback method.
    Requires OpenCV (cv2) to be installed.
    
    Args:
        video: Video tensor [1, num_frames, 3, H, W] in range [0, 1]
        
    Returns:
        flow: Optical flow [1, num_frames-1, 2, H, W] (u, v components)
    """
    import cv2
    
    batch_size, num_frames, C, H, W = video.shape
    
    # Convert to numpy and grayscale for OpenCV
    video_np = video[0].cpu().numpy()  # [num_frames, 3, H, W]
    video_np = (video_np * 255).astype(np.uint8)
    video_np = np.transpose(video_np, (0, 2, 3, 1))  # [num_frames, H, W, 3]
    
    flows = []
    for i in range(num_frames - 1):
        frame1_gray = cv2.cvtColor(video_np[i], cv2.COLOR_RGB2GRAY)
        frame2_gray = cv2.cvtColor(video_np[i + 1], cv2.COLOR_RGB2GRAY)
        
        # Compute dense optical flow
        flow = cv2.calcOpticalFlowFarneback(
            frame1_gray, frame2_gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        flows.append(flow)
    
    # Convert back to tensor
    flows = np.stack(flows, axis=0)  # [num_frames-1, H, W, 2]
    flows = torch.from_numpy(flows).permute(0, 3, 1, 2).unsqueeze(0)  # [1, num_frames-1, 2, H, W]
    
    return flows.to(video.device)


def estimate_velocity_acceleration(flow: torch.Tensor, fps: float = 25.0) -> Dict[str, torch.Tensor]:
    """
    Estimate velocity and acceleration from optical flow.
    
    Args:
        flow: Optical flow [1, num_frames-1, 2, H, W]
        fps: Frames per second
        
    Returns:
        dict with velocity and acceleration statistics
    """
    dt = 1.0 / fps
    
    # Compute magnitude of flow (velocity proxy)
    velocity = torch.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2)  # [1, num_frames-1, H, W]
    
    # Mean velocity per frame
    velocity_mean = velocity.mean(dim=[2, 3])  # [1, num_frames-1]
    
    # Acceleration (change in velocity)
    acceleration = torch.diff(velocity_mean, dim=1) / dt  # [1, num_frames-2]
    
    return {
        'velocity': velocity,
        'velocity_mean': velocity_mean,
        'acceleration': acceleration,
        'velocity_magnitude': velocity_mean.mean(),
        'acceleration_magnitude': acceleration.abs().mean(),
    }


def detect_object_motion(dino_features: torch.Tensor) -> Dict[str, float]:
    """
    Analyze object motion consistency using DINO features.
    
    Args:
        dino_features: DINO features [1, num_frames, feature_dim]
        
    Returns:
        dict with motion consistency metrics
    """
    # Compute feature similarity between consecutive frames
    # High similarity = smooth motion, Low similarity = jumpy/inconsistent
    
    features = dino_features[0]  # [num_frames, feature_dim]
    num_frames = features.shape[0]
    
    # Normalize features for cosine similarity
    features_norm = F.normalize(features, p=2, dim=1)
    
    # Compute consecutive frame similarities
    similarities = []
    for i in range(num_frames - 1):
        sim = (features_norm[i] * features_norm[i + 1]).sum()
        similarities.append(sim.item())
    
    similarities = torch.tensor(similarities)
    
    # Smooth motion should have high and consistent similarities
    motion_smoothness = similarities.mean().item()
    motion_consistency = 1.0 - similarities.std().item()  # Lower variance = more consistent
    
    return {
        'motion_smoothness': motion_smoothness,
        'motion_consistency': motion_consistency,
        'frame_similarities': similarities,
    }


def evaluate_gravity_physics(velocity_data: Dict[str, torch.Tensor], 
                             expected_gravity: float = 9.8,
                             tolerance: float = 0.5) -> Dict[str, float]:
    """
    Evaluate if the motion follows expected gravity physics.
    For bouncing ball: should see cyclic acceleration pattern.
    
    Args:
        velocity_data: Output from estimate_velocity_acceleration
        expected_gravity: Expected gravitational acceleration (pixels/s^2 normalized)
        tolerance: Tolerance for physics evaluation
        
    Returns:
        dict with gravity physics scores
    """
    acceleration = velocity_data['acceleration'][0]  # [num_frames-2]
    
    # For bouncing motion, we expect periodic changes in acceleration
    # Normalize acceleration to check for consistent magnitude
    acc_normalized = acceleration / (acceleration.abs().mean() + 1e-6)
    
    # Check for periodicity (bouncing should show periodic pattern)
    # Use FFT to detect dominant frequency
    if len(acc_normalized) > 4:
        fft = torch.fft.rfft(acc_normalized)
        power = torch.abs(fft)
        
        # Strong peak in FFT indicates periodic motion (bouncing)
        periodicity_score = (power.max() / (power.mean() + 1e-6)).item()
    else:
        periodicity_score = 0.0
    
    # Check for acceleration consistency (should not be too erratic)
    acceleration_std = acceleration.std().item()
    acceleration_mean = acceleration.abs().mean().item()
    
    consistency_score = 1.0 / (1.0 + acceleration_std / (acceleration_mean + 1e-6))
    
    return {
        'periodicity_score': min(periodicity_score / 10.0, 1.0),  # Normalize
        'acceleration_consistency': consistency_score,
        'gravity_physics_score': (min(periodicity_score / 10.0, 1.0) + consistency_score) / 2.0,
    }


# ============================================================================
# Main Reward Function
# ============================================================================
def reward_function(video: torch.Tensor, weights: Dict[str, float] = None) -> torch.Tensor:
    """
    Comprehensive reward function combining DINO features and physics properties.
    
    Args:
        video: Generated video tensor [1, num_frames, 3, H, W] in range [0, 1]
        weights: Dictionary of weights for different reward components
        
    Returns:
        reward: Scalar reward value (higher is better)
    """
    if weights is None:
        weights = {
            'motion_smoothness': 0.25,      # DINO-based smooth motion
            'motion_consistency': 0.25,     # DINO-based consistency
            'velocity_realism': 0.15,       # Realistic velocity magnitudes
            'gravity_physics': 0.25,        # Physics-based gravity evaluation
            'flow_consistency': 0.10,       # Optical flow consistency
        }
    
    # ========================================================================
    # 1. Extract DINO Features
    # ========================================================================
    dino_features = DINOFeatureExtractor.extract_features(video)
    motion_metrics = detect_object_motion(dino_features)
    
    # ========================================================================
    # 2. Compute Optical Flow
    # ========================================================================
    try:
        optical_flow = compute_optical_flow(video)
        velocity_data = estimate_velocity_acceleration(optical_flow)
        
        # Flow consistency: penalize sudden large changes in flow
        flow_magnitude = torch.sqrt(optical_flow[:, :, 0]**2 + optical_flow[:, :, 1]**2)
        flow_std = flow_magnitude.std().item()
        flow_mean = flow_magnitude.mean().item()
        flow_consistency_score = 1.0 / (1.0 + flow_std / (flow_mean + 1e-6))
        
        # Velocity realism: penalize unrealistic velocities (too fast/slow)
        velocity_mag = velocity_data['velocity_magnitude'].item()
        # Normalize velocity to reasonable range (0-50 pixels/frame)
        velocity_score = 1.0 - min(abs(velocity_mag - 15.0) / 50.0, 1.0)
        
        # ====================================================================
        # 3. Evaluate Gravity Physics
        # ====================================================================
        physics_metrics = evaluate_gravity_physics(velocity_data)
        gravity_score = physics_metrics['gravity_physics_score']
        
    except Exception as e:
        # Fallback if optical flow fails
        print(f"Warning: Optical flow computation failed: {e}")
        flow_consistency_score = 0.5
        velocity_score = 0.5
        gravity_score = 0.5
    
    # ========================================================================
    # 4. Combine Rewards
    # ========================================================================
    reward_components = {
        'motion_smoothness': motion_metrics['motion_smoothness'],
        'motion_consistency': motion_metrics['motion_consistency'],
        'velocity_realism': velocity_score,
        'gravity_physics': gravity_score,
        'flow_consistency': flow_consistency_score,
    }
    
    # Weighted sum
    total_reward = sum(
        weights[key] * value 
        for key, value in reward_components.items()
    )
    
    # Print detailed breakdown (optional, can be commented out for speed)
    # print(f"\n  Reward Components:")
    # for key, value in reward_components.items():
    #     print(f"    {key}: {value:.4f} (weight: {weights[key]})")
    # print(f"  Total Reward: {total_reward:.4f}")
    
    return torch.tensor(total_reward, device=video.device)

# ============================================================================
# Alternative: Simplified Reward Function (faster, less accurate)
# ============================================================================
def reward_function_simple(video: torch.Tensor) -> torch.Tensor:
    """
    Simplified reward function using only DINO features.
    Faster but less physics-aware.
    
    Args:
        video: Generated video tensor [1, num_frames, 3, H, W]
        
    Returns:
        reward: Scalar reward value
    """
    # Extract DINO features
    dino_features = DINOFeatureExtractor.extract_features(video)
    motion_metrics = detect_object_motion(dino_features)
    
    # Simple reward: average of smoothness and consistency
    reward = (motion_metrics['motion_smoothness'] + motion_metrics['motion_consistency']) / 2.0
    
    return torch.tensor(reward, device=video.device)


if __name__ == "__main__":
    print("Helper module loaded successfully!")
    print("\nAvailable functions:")
    print("  - decode_x0_to_video(): Decode latents to video")
    print("  - reward_function(): Comprehensive physics-based reward")
    print("  - reward_function_simple(): Fast DINO-only reward")

