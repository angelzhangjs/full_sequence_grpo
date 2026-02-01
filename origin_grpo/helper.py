#!/usr/bin/env python3
"""
Helper functions for LTX-Video Pipeline
Includes video decoding and comprehensive CLIP+DINO+Physics reward functions
"""
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms.functional import resize
import numpy as np
from typing import Dict
from ltx_video.models.autoencoders.vae_encode import vae_decode
import clip

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
        # NOTE: Trying vae_per_channel_normalize=False to fix color bias
        video = vae_decode(
            x0_unpatchified,
            pipeline.vae,
            is_video=True,
            vae_per_channel_normalize=True,  # Changed from True - may fix red/dark bias!
            timestep=decode_timestep,
        )
        # Use pipeline's image_processor.postprocess() for consistent normalization
        # This matches exactly what baseline LTX-Video does
        video = pipeline.image_processor.postprocess(video, output_type="pt")
        
        # Rearrange from [batch, channels, frames, H, W] to [batch, frames, channels, H, W]
        # This is the format expected by the reward function
        video = video.permute(0, 2, 1, 3, 4)
        
    return video  # Shape: [batch, num_frames, channels, height, width], range [0, 1]


if __name__ == "__main__":
    print("Helper module loaded successfully!")
    print("\nAvailable functions:")
    print("  - decode_x0_to_video(): Decode latents to video")
    print("  - reward_function(): Comprehensive physics-based reward")
    print("  - reward_function_simple(): Fast DINO-only reward")

