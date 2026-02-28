#!/usr/bin/env python3
# GRPO Training Loop for LTX-Video Pipeline
"""
LTX-Video Pipeline - Complete Working Example
Shows proper denoising loop with x0 prediction and video saving
"""
import torch
from ltx_video.ltx_video.inference import load_pipeline_config, create_ltx_video_pipeline
from ltx_video.models.autoencoders.causal_video_autoencoder import CausalVideoAutoencoder
from ltx_video.models.autoencoders.vae_encode import latent_to_pixel_coords
from huggingface_hub import hf_hub_download
import os
import sys
import copy
from pathlib import Path
from datetime import datetime
import imageio
from helper import decode_x0_to_video
from reward_functions import reward_function, clear_model_cache
from gemini_rewards import score_video_with_gemini

# Clear any cached models to ensure proper dtype loading
clear_model_cache()
import numpy as np
import random
# ============================================================================
# Set Random Seeds for Reproducibility
# ============================================================================
SEED = 2026
# ============================================================================
# Setup Logging to File
# ============================================================================
class WriteLogger:
    """Writes output to both console and file"""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', buffering=1)  # Line buffered
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()
        
    def close(self):
        self.log.close()

# Simple helper to save rollout videos for Gemini ranking
def save_video_to_mp4(video_tensor, out_path, frame_rate: int = 16):
    """
    video_tensor: [1, T, 3, H, W] in [0, 1]
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    video_np = video_tensor[0].float().cpu().numpy()  # [T, 3, H, W]
    video_np = np.transpose(video_np, (0, 2, 3, 1))  # [T, H, W, 3]
    video_np = (video_np * 255).clip(0, 255).astype(np.uint8)
    
    writer = imageio.get_writer(
        str(out_path),
        fps=frame_rate,
        codec='libx264',
        quality=8,
        pixelformat='yuv420p',
        macro_block_size=1,
        output_params=['-movflags', '+faststart', '-profile:v', 'baseline'],
    )
    for frame in video_np:
        writer.append_data(frame)
    writer.close()

# Create log file with timestamp in grpo/ folder
os.makedirs("grpo", exist_ok=True)  # Create grpo folder if it doesn't exist
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"grpo/training_log_{timestamp}.txt"
logger = WriteLogger(log_filename)
sys.stdout = logger
sys.stderr = logger  # Also capture warnings/errors

print("="*70)
print("LTX-VIDEO PIPELINE - DENOISING WITH X0 PREDICTIONS")
print(f"Logging to: {log_filename}")
print("="*70 + "\n")
# ============================================================================
# Load Pipeline
# ============================================================================

# Using 2B model for faster training and lower memory usage
config_path = "configs/ltxv-2b-0.9.6-dev.yaml"  # Using GRPO config (no prompt enhancer)
cfg = load_pipeline_config(config_path)
ckpt_name = cfg["checkpoint_path"]   # load the checkpoint name from the config file

def resolve_checkpoint(name: str) -> str:
    """
    Resolve a checkpoint reference.
    - If `name` is an existing local path, return it.
    - Otherwise, try to download from HF hub using the basename (handles YAMLs
      that store absolute paths).
    """
    try:
        p = Path(name)
        if p.exists() and p.is_file():
            print(f"Using local checkpoint: {p}")
            return str(p)

        candidate = p.name  # basename for HF hub download
        print(f"Attempting HF download checkpoint: {candidate} (from {name})")
        return hf_hub_download("Lightricks/LTX-Video", candidate)
    except Exception as e:
        print(f"⚠️  Download failed for {name}: {e}")
        return None

ckpt_path = os.getenv("CKPT_PATH") or (ckpt_name if os.path.isfile(ckpt_name) else resolve_checkpoint(ckpt_name))

if ckpt_path is None or not os.path.isfile(ckpt_path):
    raise FileNotFoundError(
        f"Could not locate or download the checkpoint: {ckpt_name}\n"
        f"Resolved path: {ckpt_path}\n"
        "Fix options:\n"
        "  - Download the checkpoint file into the path specified by the YAML, or\n"
        "  - Set CKPT_PATH=/path/to/checkpoint.safetensors, or\n"
        "  - Switch config_path to a YAML that points to an existing checkpoint."
    )
pipeline = create_ltx_video_pipeline(
    ckpt_path=ckpt_path,
    precision="bfloat16",
    text_encoder_model_name_or_path=cfg["text_encoder_model_name_or_path"],
    sampler=cfg.get("sampler"),
    device="cuda",
    enhance_prompt=False,
)

# Enable VAE tiling to reduce memory usage during encoding/decoding
if hasattr(pipeline.vae, 'enable_tiling'):
    try:
        pipeline.vae.enable_tiling()
        print("✅ VAE tiling enabled")
    except Exception as e:
        print(f"⚠️  VAE tiling failed: {e}")

# Enable VAE slicing (decode in chunks) - critical for memory
if hasattr(pipeline.vae, 'enable_slicing'):
    pipeline.vae.enable_slicing()
    print("✅ VAE slicing enabled (decodes in chunks)")

print("✅ Pipeline loaded!\n")

# ============================================================================
# Setup
# ============================================================================
model = pipeline.transformer
scheduler = pipeline.scheduler
vae = pipeline.vae

# Toggle Gemini-based ranking for rollouts (one-hot or normalized advantages)
USE_GEMINI_RANKING = True
GEMINI_MODEL_NAME = "models/gemini-2.5-flash"

# Timesteps (increase to 40 for finer denoising; higher cost/memory)
scheduler.set_timesteps(40, device="cuda")
timesteps = scheduler.timesteps

# GRPO only on last N timesteps (fine details and motion)
# Early timesteps (high noise) are just rough structure, don't need GRPO
NUM_GRPO_STEPS = 15  # Increased from 10 for more training
timesteps_for_grpo = timesteps[-NUM_GRPO_STEPS:]  # Last 10 steps
print(f"Total timesteps: {len(timesteps)} steps [{timesteps[0]:.4f} → {timesteps[-1]:.4f}]")
print(f"GRPO training on: Last {NUM_GRPO_STEPS} steps [{timesteps_for_grpo[0]:.4f} → {timesteps_for_grpo[-1]:.4f}]")
print(f"Skipping first {len(timesteps) - NUM_GRPO_STEPS} steps (rough structure only)\n")

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

prompt_embeds_tuple = pipeline.encode_prompt(
    prompt=prompt,
    device="cuda",
    num_images_per_prompt=1,
    do_classifier_free_guidance=False,
)
prompt_embeds = prompt_embeds_tuple[0]
prompt_attention_mask = prompt_embeds_tuple[1]

print(f"Prompt: '{prompt}'")
print(f"prompt_embeds: {prompt_embeds.shape}\n")

# Video parameters
# 5 seconds at 16 fps = 80 frames
height = 512
width = 768
num_frames = 80  # exact 5 seconds at 16 fps
frame_rate = 16

print(f"⚠️  Training will decode {num_frames}-frame videos at each step")
print(f"   Est. VAE decode memory: ~{num_frames * 0.16:.1f} GB per step") 

# Calculate latent dimensions
vae_scale_factor = pipeline.vae_scale_factor
video_scale_factor = pipeline.video_scale_factor

latent_height = height // vae_scale_factor
latent_width = width // vae_scale_factor
latent_frames = num_frames // video_scale_factor

if isinstance(vae, CausalVideoAutoencoder):
    latent_frames += 1

latent_shape = (1, pipeline.vae.config.latent_channels, latent_frames, latent_height, latent_width)

print(f"Latent shape: {latent_shape}\n")

# ============================================================================
# Choose Unfreezing Method: LoRA or Traditional
# ============================================================================
USE_LORA = True  # Set to True to use LoRA (requires: pip install --upgrade transformers>=4.40)

if USE_LORA:
    # ========================================================================
    # LoRA Method - Stable fine-tuning for multiple attention layers
    # ========================================================================
    from lora_config import apply_lora_to_model, get_lora_config_motion_focused
    
    print("\n" + "="*70)
    print("APPLYING LORA FOR STABLE FINE-TUNING")
    print("="*70 + "\n")
    
    # Choose configuration:
    config = get_lora_config_motion_focused()  # Self-attn in 5 blocks (motion/physics)
    
    # Apply LoRA to model
    model, recommended_lr = apply_lora_to_model(model, **config)
    pipeline.transformer = model  # Update pipeline reference
    
    # Create optimizer with LoRA-recommended LR
    optimizer = torch.optim.Adam(
        model.parameters(),  # PEFT handles trainable param filtering
        lr=recommended_lr,
        betas=(0.9, 0.95),
        weight_decay=0.01
    )
    print(f"✅ LoRA initialized with LR={recommended_lr:.2e}\n")

else:
    raise ValueError("LoRA is not used, Traditional method not implemented")

    # # ========================================================================
    # # Traditional Method - Direct unfreezing of attention layers
    # # ========================================================================
    # print("\n" + "="*70)
    # print("TRADITIONAL UNFREEZING (Attention Layers)")
    # print("="*70 + "\n")
    
    # # Freeze all parameters first
    # for param in model.parameters():
    #     param.requires_grad = False
    
    # # Configuration
    # TOTAL_BLOCKS = 28  # LTX-Video has 28 transformer blocks (0-27)
    # ATTN1_NUM_BLOCKS = 4  # Unfreeze self-attention in last 4 blocks
    # ATTN2_TARGET_BLOCK = 27  # Unfreeze cross-attention in last block only
    
    # attn1_start_block = TOTAL_BLOCKS - ATTN1_NUM_BLOCKS  # Block 23
    
    # unfrozen_params = []
    # attn1_count = 0
    # attn2_count = 0
    
    # print(f"🎯 Unfreezing Strategy:")
    # print(f"   Self-Attention (attn1): Blocks {attn1_start_block}-{TOTAL_BLOCKS-1} (last {ATTN1_NUM_BLOCKS} blocks)")
    # print(f"   Cross-Attention (attn2): Block {ATTN2_TARGET_BLOCK} only\n")
    
    # for name, param in model.named_parameters():
    #     should_unfreeze = False
    #     layer_type = None
        
    #     # Strategy 1: Unfreeze self-attention (attn1) in blocks 23-27
    #     for block_idx in range(attn1_start_block, TOTAL_BLOCKS):
    #         if f"transformer_blocks.{block_idx}." in name:
    #             # Look for self-attention patterns (based on actual layer names)
    #             if any(pattern in name for pattern in [
    #                 "attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out"
    #             ]):
    #                 should_unfreeze = True
    #                 layer_type = "attn1"
    #                 attn1_count += 1
    #                 break
        
    #     # Strategy 2: Unfreeze cross-attention (attn2) ONLY in block 27
    #     if f"transformer_blocks.{ATTN2_TARGET_BLOCK}." in name:
    #         if any(pattern in name for pattern in [
    #             "attn2.to_q", "attn2.to_k", "attn2.to_v", "attn2.to_out"
    #         ]):
    #             should_unfreeze = True
    #             layer_type = "attn2"
    #             attn2_count += 1
        
    #     if should_unfreeze:
    #         param.requires_grad = True
    #         unfrozen_params.append(param)
    #         print(f"  Unfreezing [{layer_type:5s}]: {name} - {param.shape}")
    
    # # Safety check - raise error if attention layers not found
    # if len(unfrozen_params) == 0:
    #     print("\n❌ ERROR: No attention layers found!")
    #     print("\n🔍 Available layer names (sample):")
    #     sample_count = 0
    #     for name, _ in model.named_parameters():
    #         if 'block' in name.lower() and sample_count < 10:
    #             print(f"     {name}")
    #             sample_count += 1
        
    #     raise ValueError(
    #         f"No attention parameters were unfrozen!\n"
    #         f"   Looking for: Self-attention (attn1) in blocks {attn1_start_block}-{TOTAL_BLOCKS-1}\n"
    #         f"                Cross-attention (attn2) in block {ATTN2_TARGET_BLOCK}\n"
    #         f"   Check layer naming patterns above and adjust the code."
    #     )
    
    # # Summary
    # print(f"\n✅ Unfreezing Summary:")
    # print(f"   Self-attention (attn1) params: {attn1_count}")
    # print(f"   Cross-attention (attn2) params: {attn2_count}")
    # print(f"   Total unfrozen parameters: {len(unfrozen_params)}")
    # print(f"   Total param count: {sum(p.numel() for p in unfrozen_params):,}")
    
    # # Adjust learning rate based on number of parameters
    # total_params = sum(p.numel() for p in unfrozen_params)
    # if total_params > 10_000_000:  # > 10M params
    #     lr = 1e-5  # VERY gentle LR to preserve baseline quality!
    #     print(f"   Using GENTLE LR to preserve baseline: {lr:.2e}")
    # elif total_params > 1_000_000:  # > 1M params
    #     lr = 1e-4
    #     print(f"   Using moderate LR: {lr:.2e}")
    # else:
    #     lr = 1e-4
    #     print(f"   Using standard LR: {lr:.2e}")
    
    # print()
    
    # optimizer = torch.optim.Adam(
    #     unfrozen_params,
    #     lr=lr,
    #     betas=(0.9, 0.95),
    #     weight_decay=0.01
    # )
    
# ============================================================================
# Initialize Latents and Run Non-GRPO Steps
# ============================================================================
latents = pipeline.prepare_latents(
    latents=None,
    media_items=None,
    timestep=timesteps[0],
    latent_shape=latent_shape,
    dtype=torch.bfloat16,
    device=torch.device("cuda"),
    generator=None,
)

# Patchify latents (now with correct 128 channels!)
latents, latent_coords = pipeline.patchifier.patchify(latents)

# Convert to pixel coords and scale temporal dimension
pixel_coords = latent_to_pixel_coords(
    latent_coords,
    pipeline.vae,
    causal_fix=pipeline.transformer.config.causal_temporal_positioning,
)

# Scale temporal dimension

indices_grid = pixel_coords.to(torch.float32)
indices_grid[:, 0] *= (1.0 / frame_rate)
# ============================================================================
# Create Frozen Baseline Model for GRPO
# ============================================================================
print("Creating frozen baseline model for GRPO comparison...")
baseline_model = copy.deepcopy(model)
baseline_model.eval()  # Set to eval mode
# Freeze all parameters
for param in baseline_model.parameters():
    param.requires_grad = False
print("✅ Baseline model frozen (will not be updated)\n")

# ============================================================================
# Run Standard Denoising for First N Steps (No GRPO)
# ============================================================================
# Run first (20 - 10) = 10 timesteps WITHOUT GRPO (just standard denoising)
# This creates the rough structure before fine-tuning with GRPO
print(f"{'='*70}")
print("PHASE 1: Standard Denoising (Rough Structure)")
print(f"{'='*70}\n")

num_standard_steps = len(timesteps) - NUM_GRPO_STEPS
for i, t in enumerate(timesteps[:num_standard_steps]):
    print(f"Standard step {i+1:02d}/{num_standard_steps} | t={t:.4f}", end="")
    
    with torch.no_grad():
        # Use baseline model for initial steps (more stable)
        noise_pred = baseline_model(
            latents,
            indices_grid=indices_grid,
            encoder_hidden_states=prompt_embeds,
            encoder_attention_mask=prompt_attention_mask,
            timestep=t,
            return_dict=False,
        )[0]
        
        # Standard denoising step (keep False here to avoid dtype issues)
        latents = pipeline.denoising_step(
            latents=latents,
            noise_pred=noise_pred,
            current_timestep=None,
            conditioning_mask=None,
            t=t,
            extra_step_kwargs={},
            stochastic_sampling=False,  
            return_x0=False,  # Don't need x0 for standard steps
        )
        
        # Ensure latents stay in bfloat16 (stochastic sampling might change dtype)
        if latents.dtype != torch.bfloat16:
            latents = latents.to(dtype=torch.bfloat16)
        
    print(" ✓")

print(f"\n✅ Completed {num_standard_steps} standard denoising steps")
print(f"   Latents shape: {latents.shape}")
print(f"\n{'='*70}")
print("PHASE 2: GRPO Fine-Tuning (Last {NUM_GRPO_STEPS} Steps)")
print(f"{'='*70}\n")

# ============================================================================
# GRPO Training steps
# ============================================================================
# Track initial weights for comparison at the end
initial_weights = {}
for name, param in model.named_parameters():
    if param.requires_grad:
        initial_weights[name] = param.data.clone()
print(f"📊 Tracking {len(initial_weights)} unfrozen parameters\n")

num_rollouts = 3 
for i, t in enumerate(timesteps_for_grpo):  # Only iterate over last 10 timesteps
    print(f"GRPO Step {i+1:02d}/{len(timesteps_for_grpo)} | t={t:.4f}", end="")
        
    rollout_noise_preds = []  # Store noise predictions for later gradient computation
    rollout_rewards = []      # Handcrafted rewards
    rollout_video_paths = []  # Saved MP4s for Gemini ranking
        
    for rollout_index in range(num_rollouts):
        #####==========================================================
        # STEP 1:Predict with current model weights to get next_latents and x0_est
        #####==========================================================
        with torch.no_grad():  # No gradients during rollout sampling
            # IMPROVED: Stronger perturbation for GRPO diversity
            if rollout_index > 0:
                noise_scale = 0.5  # Increased to 0.5 for MUCH better diversity and stronger learning signal
                latents_perturbed = latents + torch.randn_like(latents) * noise_scale
            else:
                latents_perturbed = latents
            
            # IMPROVED: Higher temperature for better GRPO diversity  
            temperature = 1.0 + (rollout_index - 1) * 0.08  # [1.0, 1.08, 1.16] - more variation
            
            noise_pred = model(
                latents_perturbed,  # Use perturbed latents
                indices_grid=indices_grid,
                encoder_hidden_states=prompt_embeds,
                encoder_attention_mask=prompt_attention_mask,
                timestep=t,
                return_dict=False,
            )[0]
            
            # Apply minimal temperature to noise prediction
            noise_pred = noise_pred * temperature
            
            # Store noise prediction for later
            rollout_noise_preds.append(noise_pred.clone())
            
            next_latents, x0_est = pipeline.denoising_step(
                latents=latents,  # Use original latents for denoising
                noise_pred=noise_pred,
                current_timestep=None,
                conditioning_mask=None,
                t=t,
                extra_step_kwargs={},
                stochastic_sampling=True,  # ENABLED for diversity (critical for GRPO!)
                return_x0=True,
            )
            
            # Decode x0 for saving and Gemini ranking
            video_x0 = decode_x0_to_video(
                    x0_est,
                    pipeline,
                    #latent_height=latent_height,
                    #latent_width=latent_width,
                    num_frames=num_frames,
                    height=height,
                    width=width,
                    is_patchified=True,
                )
            # Save rollout video for Gemini ranking
            rollout_path = Path("rollout_40_15") / f"rollout_step{i}_r{rollout_index}_{timestamp}.mp4"
            save_video_to_mp4(video_x0, rollout_path, frame_rate=frame_rate)
            rollout_video_paths.append(rollout_path)
            print(f"  Saved rollout video: {rollout_path}")
            # Save latent x0 as numpy for later analysis/ranking if needed
            rollout_x0_path = Path("rollout_40_15_x0") / f"rollout_step{i}_r{rollout_index}_{timestamp}.npy"
            rollout_x0_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(rollout_x0_path, x0_est.detach().float().cpu().numpy())
            # Handcrafted reward
            video_for_reward = video_x0.float() if video_x0.dtype == torch.bfloat16 else video_x0
            reward = reward_function(video_for_reward, prompt=prompt)
            rollout_rewards.append(reward)
            
            ###================================================================== 
            # Clear GPU memory after each rollout to prevent OOM
            del video_x0, x0_est, next_latents
            torch.cuda.empty_cache()
            
    ###==================================================================
    # Compute advantages: Gemini when enabled; otherwise handcrafted
    advantage_rewards = None
    if USE_GEMINI_RANKING:
        try:
            print("\n[Gemini] Scoring rollouts...")
            gemini_scores = []
            for path in rollout_video_paths:
                score = score_video_with_gemini(str(path), model_name=GEMINI_MODEL_NAME)
                gemini_scores.append(score["overall"])
                print(f"  {path.name}: overall={score['overall']:.4f}")
            
            scores_t = torch.tensor(gemini_scores, device=latents.device, dtype=torch.float32)
            mean_g = scores_t.mean()
            std_g = scores_t.std()
            if std_g > 1e-8:
                advantage_rewards = (scores_t - mean_g) / (std_g + 1e-4)
            else:
                advantage_rewards = scores_t - mean_g
            
            best_idx = int(scores_t.argmax().item())
            print(f"[Gemini] Best rollout: idx={best_idx}, score={scores_t[best_idx]:.4f}")
            print(f"Using Gemini-derived advantages (normalized scores).", end="")
        except Exception as gem_err:
            print(f"\n⚠️ Gemini ranking failed ({gem_err}); falling back to handcrafted rewards.")
            advantage_rewards = None
    
    if advantage_rewards is None:
        rewards = torch.stack(rollout_rewards)
        reward_std = rewards.std().item()
        reward_range = (rewards.max() - rewards.min()).item()
        print(f"\n[Reward stats] σ={reward_std:.4f}, range={reward_range:.4f}", end="")
        
        mean_reward = rewards.mean()
        std_reward = rewards.std()
        if std_reward > 1e-8:
            advantage_rewards = (rewards - mean_reward) / (std_reward + 1e-4)
        else:
            advantage_rewards = rewards - mean_reward
        
        print(f"Rollout Statistics:")
        print(f"  Mean reward: {mean_reward.item():.4f}")
        print(f"  Std reward: {std_reward.item():.4f}")
        print(f"\nAdvantages (vs group mean={mean_reward.item():.4f}):")
        for k in range(num_rollouts):
            print(f"  Rollout {k}: reward={rewards[k]:.4f}, advantage={advantage_rewards[k]:.4f}")
    
    # Clear rollout data to free memory before backward pass
    torch.cuda.empty_cache()
    
    #=======================================================================
    # STEP 3: Compute GRPO loss with proper policy gradient
    ###==================================================================
    # Clear all gradients completely before computing loss
    optimizer.zero_grad()
    model.zero_grad()
    
    # Ensure latents has no gradient history from previous timesteps
    latents_for_loss = latents.detach().clone().requires_grad_(True)
    
    # Get current model's noise prediction (with gradients enabled)
    noise_pred_current = model(
        latents_for_loss,
        indices_grid=indices_grid.detach(),
        encoder_hidden_states=prompt_embeds.detach(),
        encoder_attention_mask=prompt_attention_mask.detach(),
        timestep=t,
        return_dict=False,
    )[0]
    
    # GRPO Policy Gradient: Compute log probability for each rollout
    # Treat the noise prediction as a Gaussian policy - lower MSE = higher probability
    log_probs = []
    for rollout_idx in range(num_rollouts):
        # MSE between rollout's noise prediction and current model output
        # Detach the rollout prediction (it was generated with no_grad)
        rollout_noise = rollout_noise_preds[rollout_idx].detach()
        
        # Compute mean squared error as negative log probability
        # MSE measures how different the rollout was from current policy
        mse = ((rollout_noise - noise_pred_current) ** 2).mean()
        
        # Negative MSE as log probability (higher similarity = higher log prob)
        log_prob = -mse
        log_probs.append(log_prob)
    
    # Stack log probabilities and weight by advantages
    log_probs = torch.stack(log_probs)  # [num_rollouts]
    
    # Policy gradient loss: -E[log_prob * advantage]
    # Higher advantage rollouts should have higher log_prob (lower MSE)
    # This pushes the model toward producing outputs similar to high-reward rollouts
    loss = -(log_probs * advantage_rewards).mean()
    
    print(f" [Loss={loss.item():.6f}]", end="")
    
    # Backprop
    loss.backward()
    
    # Track gradient norms BEFORE clipping
    grad_norms = []
    for param in optimizer.param_groups[0]["params"]:
        if param.grad is not None:
            grad_norms.append(param.grad.norm().item())
    avg_grad_norm = sum(grad_norms) / len(grad_norms) if grad_norms else 0
    
    # Clip gradients
    total_grad_norm = torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]["params"], 0.5)
    
    # Track weights BEFORE update
    weights_before = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            weights_before[name] = param.data.clone()

    # Update weights
    optimizer.step()
    
    # Track weights AFTER update and compute change
    weight_changes = []
    for name, param in model.named_parameters():
        if param.requires_grad and name in weights_before:
            change = (param.data - weights_before[name]).abs().mean().item()
            weight_changes.append(change)
    avg_weight_change = sum(weight_changes) / len(weight_changes) if weight_changes else 0
    print(f"  ✅ Weights updated! grad_norm={avg_grad_norm:.6f}, weight_Δ={avg_weight_change:.6f}")
    
    # IMPORTANT: Clear everything after optimizer step
    optimizer.zero_grad()
    model.zero_grad()
    
    # Delete intermediate tensors
    del loss, log_probs, noise_pred_current, latents_for_loss, rollout_noise_preds
    torch.cuda.empty_cache()    
    # ========================================================================
    # Step 4: Recompute with Updated Parameters
    # ========================================================================
    with torch.no_grad():
        noise_pred = model(
            latents,  # First positional argument (not keyword!)
            indices_grid=indices_grid,
            encoder_hidden_states=prompt_embeds,
            encoder_attention_mask=prompt_attention_mask,
            timestep=t,
            return_dict=False,
        )[0] # use the recent UPDATE model weights now 
            
        # update latents with the recent updated model weights for next timestep
        next_latents, x0_est = pipeline.denoising_step(
                latents=latents,
                noise_pred=noise_pred,
                current_timestep=None,
                conditioning_mask=None,
                t=t,
                extra_step_kwargs={},
                return_x0=True
            )         
    
    # Detach from computation graph before next timestep to avoid "backward through graph twice" error
    latents = next_latents.detach().clone()

# ============================================================================
# Training Summary
# ============================================================================
print("\n" + "="*70)
print("TRAINING COMPLETE - WEIGHT CHANGE SUMMARY")
print("="*70)

total_changes = []
for name, param in model.named_parameters():
    if param.requires_grad and name in initial_weights:
        total_change = (param.data - initial_weights[name]).abs().mean().item()
        total_changes.append(total_change)
        print(f"  {name:50s} Δ={total_change:.8f}")

avg_total_change = sum(total_changes) / len(total_changes) if total_changes else 0
print(f"\n  Average total weight change: {avg_total_change:.8f}")
print("="*70 + "\n")

# ============================================================================
# Generate and Save Final Video with Updated Weights
# ============================================================================
print("\n" + "="*70)
print("GENERATING FINAL VIDEO WITH UPDATED WEIGHTS")
print("="*70 + "\n")

# Create outputs directory if it doesn't exist
os.makedirs("outputs", exist_ok=True)

# Generate final video
with torch.no_grad():
    # Use the updated model to generate final prediction
    noise_pred_final = model(
        latents,
        indices_grid=indices_grid,
        encoder_hidden_states=prompt_embeds,
        encoder_attention_mask=prompt_attention_mask,
        timestep=timesteps[-1],  # Use final timestep
        return_dict=False,
    )[0]
    
    # Get final x0 estimate
    _, x0_final = pipeline.denoising_step(
        latents=latents,
        noise_pred=noise_pred_final,
        current_timestep=None,
        conditioning_mask=None,
        t=timesteps[-1],
        extra_step_kwargs={},
        stochastic_sampling=True,  # CRITICAL: Match rollout setting for color!
        return_x0=True
    )
    
    # Decode to video
    print("Decoding final video...")
    print(f"  [DEBUG] x0_final latent stats:")
    print(f"    Shape: {x0_final.shape}")
    print(f"    Min: {x0_final.min():.4f}, Max: {x0_final.max():.4f}, Mean: {x0_final.mean():.4f}")
    print(f"    Std: {x0_final.std():.4f}")
    
    final_video = decode_x0_to_video(
        x0_final,
        pipeline,
        num_frames=num_frames,
        height=height,
        width=width,
        is_patchified=True,
    )
    
    # Save video to file in grpo/ folder
    os.makedirs("grpo", exist_ok=True)  # Create grpo folder if it doesn't exist
    output_filename = f"grpo/final_video_{timestamp}.mp4"
    
    # Convert tensor to numpy for saving
    import numpy as np
    
    # Convert bfloat16 to float32, then to numpy
    video_np = final_video[0].float().cpu().numpy()  # [num_frames, 3, H, W]
    
    print(f"  [DEBUG] Video tensor stats before transpose:")
    print(f"    Shape: {video_np.shape}")
    print(f"    Min: {video_np.min():.4f}, Max: {video_np.max():.4f}, Mean: {video_np.mean():.4f}")
    print(f"    Per-channel: R={video_np[:, 0].mean():.4f}, G={video_np[:, 1].mean():.4f}, B={video_np[:, 2].mean():.4f}")
    
    # Check color saturation
    channel_std = np.std([video_np[:, 0].mean(), video_np[:, 1].mean(), video_np[:, 2].mean()])
    print(f"    Channel std dev: {channel_std:.4f} (low = grayscale)")
    
    video_np = np.transpose(video_np, (0, 2, 3, 1))  # [num_frames, H, W, 3]
    
    # Video is already in [0, 1] range from image_processor.postprocess()
    # No additional normalization needed - this was washing out colors!
    print(f"  [INFO] Video already normalized to [0, 1] by image_processor")
    print(f"    Current range: [{video_np.min():.4f}, {video_np.max():.4f}]")
    
    # Directly convert to uint8 (preserve original color distribution)
    video_np = (video_np * 255).clip(0, 255).astype(np.uint8)
    
    # Final check on uint8 values
    print(f"  [DEBUG] Final uint8 stats:")
    print(f"    Min: {video_np.min()}, Max: {video_np.max()}, Mean: {video_np.mean():.1f}")
    print(f"    Per-channel uint8: R={video_np[:, :, :, 0].mean():.1f}, G={video_np[:, :, :, 1].mean():.1f}, B={video_np[:, :, :, 2].mean():.1f}")
    
    
    # Save as MP4 with high quality settings
    writer = imageio.get_writer(
        output_filename, 
        fps=frame_rate,
        codec='libx264',
        quality=10,  # Maximum quality (1-10 scale)
        pixelformat='yuv420p',  # Standard color format
        macro_block_size=1,
        bitrate='8000k',  # High bitrate for crisp output
        output_params=['-crf', '18']  # Constant Rate Factor: 18 = visually lossless quality
    )
    for frame in video_np:
        writer.append_data(frame)
    writer.close()
    
    print(f"✅ Final video saved to: {output_filename}")
    print(f"   Resolution: {width}×{height}")
    print(f"   Frames: {num_frames}")
    print(f"   Duration: {num_frames/frame_rate:.2f}s")

# ============================================================================
# Close Log File
# ============================================================================
print(f"\n✅ Training complete! Log saved to: {log_filename}")
logger.close()
sys.stdout = logger.terminal
sys.stderr = sys.__stderr__ 
        