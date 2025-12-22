#!/usr/bin/env python3
"""
Baseline LTX-Video Generation (No GRPO)
Generates video using standard LTX-Video inference for comparison with GRPO-trained model
"""
import torch
from ltx_video.inference import load_pipeline_config, create_ltx_video_pipeline
from huggingface_hub import hf_hub_download
import os
from datetime import datetime
import numpy as np
import imageio

print("="*70)
print("BASELINE LTX-VIDEO GENERATION (NO GRPO)")
print("="*70 + "\n")

# ============================================================================
# Configuration (Match training settings)
# ============================================================================
config_path = "configs/ltxv-2b-0.9.8-distilled.yaml"

# Read prompt from file (in same directory as script)
prompt_file = "prompt.txt"
if os.path.exists(prompt_file):
    with open(prompt_file, 'r') as f:
        # Read first non-empty line as the prompt
        for line in f:
            line = line.strip()
            if line:
                prompt = line
                break
        else:
            prompt = "A ball bouncing up a staircase, hitting each step sequentially."
    print(f"✓ Loaded prompt from {prompt_file}")
else:
    prompt = "A ball bouncing up a staircase, hitting each step sequentially."
    print(f"⚠ {prompt_file} not found, using default prompt")

height = 512
width = 768
num_frames = 81  # 8×10 + 1 (optimal for model), ~5 seconds at 16 fps  
frame_rate = 16
num_inference_steps = 20
guidance_scale = 3.0
seed = 42

print(f"Prompt: '{prompt}'")
print(f"Resolution: {width}×{height}")
print(f"Frames: {num_frames} ({num_frames/frame_rate:.2f}s @ {frame_rate} fps)")
print(f"Inference steps: {num_inference_steps}")
print(f"Guidance scale: {guidance_scale}")
print(f"Seed: {seed}\n")

# ============================================================================
# Load Pipeline
# ============================================================================
print("Loading LTX-Video pipeline...")
cfg = load_pipeline_config(config_path)
ckpt_name = cfg["checkpoint_path"]

if not os.path.isfile(ckpt_name):
    print(f"Downloading {ckpt_name}...")
    ckpt_path = hf_hub_download("Lightricks/LTX-Video", ckpt_name)
else:
    ckpt_path = ckpt_name

pipeline = create_ltx_video_pipeline(
    ckpt_path=ckpt_path,
    precision="bfloat16",
    text_encoder_model_name_or_path=cfg["text_encoder_model_name_or_path"],
    sampler=cfg.get("sampler"),
    device="cuda",
    enhance_prompt=False,
)
print("✅ Pipeline loaded!\n")

# ============================================================================
# Generate Video
# ============================================================================
print("Generating video...")
generator = torch.Generator(device="cuda").manual_seed(seed)

# Use pipeline's built-in __call__ method for standard inference
output = pipeline(
    prompt=prompt,
    height=height,
    width=width,
    num_frames=num_frames,
    frame_rate=frame_rate,  # Required argument
    num_inference_steps=num_inference_steps,
    guidance_scale=guidance_scale,
    generator=generator,
    output_type="pt",  # Return as PyTorch tensor
)

video = output.frames[0]  # Shape: [num_frames, channels, height, width]
print(f"✅ Video generated! Shape: {video.shape}\n")

# ============================================================================
# Save Video
# ============================================================================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_filename = f"baseline_output/baseline_video_{timestamp}.mp4"
os.makedirs("baseline_output", exist_ok=True)

print("Saving video...")
# Convert from [num_frames, channels, height, width] to [num_frames, height, width, channels]
video_np = video.float().cpu().numpy()

# Debug: Check value range
print(f"  [DEBUG] Video stats:")
print(f"    Min: {video_np.min():.4f}, Max: {video_np.max():.4f}, Mean: {video_np.mean():.4f}")
print(f"    Per-channel: R={video_np[:, 0].mean():.4f}, G={video_np[:, 1].mean():.4f}, B={video_np[:, 2].mean():.4f}")

# Transpose to [num_frames, height, width, channels]
video_np = np.transpose(video_np, (0, 2, 3, 1))

# Normalize to [0, 1] if needed (LTX-Video typically outputs in [-1, 1])
if video_np.min() < 0:
    print("  [INFO] Normalizing from [-1, 1] to [0, 1]")
    video_np = (video_np + 1.0) / 2.0
    
# Auto-contrast to improve brightness
vmin, vmax = video_np.min(), video_np.max()
if vmax > vmin:
    video_np = (video_np - vmin) / (vmax - vmin)
    print(f"  [INFO] Applied auto-contrast: [{vmin:.3f}, {vmax:.3f}] → [0, 1]")

# Gamma correction for better brightness
gamma = 0.45
video_np = np.power(video_np, gamma)
print(f"  [INFO] Applied gamma correction: gamma={gamma}")
print(f"  [INFO] Final mean brightness: {video_np.mean():.4f}")

# Convert to uint8
video_np = (video_np * 255).clip(0, 255).astype(np.uint8)

# Save as MP4
writer = imageio.get_writer(
    output_filename,
    fps=frame_rate,
    codec='libx264',
    quality=8,
    pixelformat='yuv420p',
    macro_block_size=1
)
for frame in video_np:
    writer.append_data(frame)
writer.close()

print(f"\n✅ Baseline video saved to: {output_filename}")
print(f"   Resolution: {width}×{height}")
print(f"   Frames: {num_frames}")
print(f"   Duration: {num_frames/frame_rate:.2f}s")
print(f"   FPS: {frame_rate}")
print("\n" + "="*70)
print("BASELINE GENERATION COMPLETE!")
print("")
print("Output saved to: baseline_output/")
print("GRPO outputs in: outputs/")
print("")
print("Compare baseline_output/ with outputs/ to evaluate GRPO improvement")
print("="*70)

