#!/usr/bin/env python3
"""
Convert .pt tensor file to .mp4 video

Usage:
    python convert_pt_to_mp4.py <input.pt> <output.mp4>
    
Example:
    python convert_pt_to_mp4.py cogvideox_physics_grpo_lora_output/cogvideox/final_video.pt output.mp4
"""

import sys
import torch
import numpy as np
import imageio

def convert_pt_to_mp4(pt_path, mp4_path, fps=8):
    """
    Convert PyTorch tensor to MP4 video
    
    Args:
        pt_path: Path to .pt file
        mp4_path: Path to output .mp4 file
        fps: Frames per second (default: 8 for CogVideoX)
    """
    print(f"Loading {pt_path}...")
    
    # Load tensor
    data = torch.load(pt_path, map_location='cpu')
    
    # Handle different formats
    if isinstance(data, dict):
        # Might be {'video': tensor, 'latents': tensor, etc.}
        if 'video' in data:
            video = data['video']
        elif 'latents' in data:
            print("⚠️ File contains latents (not decoded video)")
            print("   Cannot convert latents without the VAE decoder")
            print("   Please use the model's decode function first")
            return
        else:
            # Try first value
            video = list(data.values())[0]
    else:
        video = data
    
    print(f"Video tensor shape: {video.shape}")
    print(f"Video dtype: {video.dtype}")
    print(f"Video range: [{video.min():.3f}, {video.max():.3f}]")
    
    # Convert bfloat16 to float32 first (numpy doesn't support bfloat16)
    if video.dtype == torch.bfloat16:
        print("  Converting bfloat16 → float32...")
        video = video.float()
    
    # Convert to numpy
    video_np = video.cpu().numpy()
    
    # Handle different tensor formats
    # Possible shapes: [T, C, H, W], [1, T, C, H, W], [B, T, C, H, W]
    if len(video_np.shape) == 5:  # [B, T, C, H, W]
        video_np = video_np[0]  # Take first batch
    
    if len(video_np.shape) == 4:  # [T, C, H, W] or [C, T, H, W]
        # Check if it's [T, C, H, W] or [C, T, H, W]
        if video_np.shape[1] == 3:  # [T, 3, H, W] - correct format
            video_np = video_np.transpose(0, 2, 3, 1)  # → [T, H, W, 3]
        elif video_np.shape[0] == 3:  # [3, T, H, W]
            video_np = video_np.transpose(1, 2, 3, 0)  # → [T, H, W, 3]
        else:
            print("⚠️ Unexpected shape, attempting best guess...")
            video_np = video_np.transpose(0, 2, 3, 1)
    
    # Normalize to [0, 1] if needed
    if video_np.max() > 1.5:
        print("  Normalizing from [0, 255] to [0, 1]")
        video_np = video_np / 255.0
    elif video_np.min() < -0.1:
        print("  Normalizing from [-1, 1] to [0, 1]")
        video_np = (video_np + 1) / 2
    
    # Convert to uint8 [0, 255]
    video_np = (video_np * 255).clip(0, 255).astype(np.uint8)
    
    print(f"Final video shape: {video_np.shape} (T, H, W, C)")
    print(f"Saving to {mp4_path}...")
    
    # Save as MP4
    writer = imageio.get_writer(
        mp4_path,
        fps=fps,
        codec='libx264',
        quality=8,
        pixelformat='yuv420p'
    )
    
    for frame in video_np:
        writer.append_data(frame)
    
    writer.close()
    
    print(f"✅ Saved {len(video_np)} frames to {mp4_path}")
    print(f"   Duration: {len(video_np)/fps:.2f} seconds @ {fps} fps")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_pt_to_mp4.py <input.pt> [output.mp4] [fps]")
        print("Example: python convert_pt_to_mp4.py final_video.pt output.mp4 8")
        sys.exit(1)
    
    pt_file = sys.argv[1]
    mp4_file = sys.argv[2] if len(sys.argv) > 2 else pt_file.replace('.pt', '.mp4')
    fps = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    
    convert_pt_to_mp4(pt_file, mp4_file, fps)
