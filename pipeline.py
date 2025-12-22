#!/usr/bin/env python3
# GRPO Training Loop for LTX-Video Pipeline
"""
LTX-Video Pipeline - Complete Working Example
Shows proper denoising loop with x0 prediction and video saving
"""
import torch
from ltx_video.inference import load_pipeline_config, create_ltx_video_pipeline
from ltx_video.models.autoencoders.causal_video_autoencoder import CausalVideoAutoencoder
from ltx_video.models.autoencoders.vae_encode import latent_to_pixel_coords
from huggingface_hub import hf_hub_download
import os
import sys
from datetime import datetime
from helper import decode_x0_to_video, reward_function

# ============================================================================
# Setup Logging to File
# ============================================================================
class TeeLogger:
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

# Create log file with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"training_log_{timestamp}.txt"
logger = TeeLogger(log_filename)
sys.stdout = logger
sys.stderr = logger  # Also capture warnings/errors

print("="*70)
print("LTX-VIDEO PIPELINE - DENOISING WITH X0 PREDICTIONS")
print(f"Logging to: {log_filename}")
print("="*70 + "\n")
# ============================================================================
# Load Pipeline
# ============================================================================
config_path = "configs/ltxv-13b-0.9.8-dev-fp8.yaml"
cfg = load_pipeline_config(config_path)
ckpt_name = cfg["checkpoint_path"]   # load the checkpoint name from the config file

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
# Setup
# ============================================================================
model = pipeline.transformer
scheduler = pipeline.scheduler
vae = pipeline.vae

# Timesteps
scheduler.set_timesteps(20, device="cuda")
timesteps = scheduler.timesteps
print(f"Timesteps: {len(timesteps)} steps [{timesteps[0]:.4f} → {timesteps[-1]:.4f}]\n")

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
# Reduced for memory efficiency during GRPO training
height = 512
width = 768
num_frames = 81  # 8×10 + 1 (optimal for model), ~5 seconds at 16 fps
frame_rate = 16

# Calculate latent dimensions
vae_scale_factor = pipeline.vae_scale_factor
video_scale_factor = pipeline.video_scale_factor

latent_height = height // vae_scale_factor
latent_width = width // vae_scale_factor
latent_frames = num_frames // video_scale_factor

if isinstance(vae, CausalVideoAutoencoder):
    latent_frames += 1

# latent_shape = (1, 4, latent_frames, latent_height, latent_width)

latent_shape = (1, pipeline.vae.config.latent_channels, latent_frames, latent_height, latent_width)

print(f"Latent shape: {latent_shape}\n")

# ============================================================================
# unfreeze the model
# ============================================================================
# freeze all parameters in the model first
for param in model.parameters():
    param.requires_grad = False

# unfreeze the parameters of the output projection layer of the transformer 
unfrozen_params = []
for name, param in model.named_parameters():
    if "proj_out" in name:  # Fixed: model uses "proj_out" not "out_proj"
        param.requires_grad = True
        unfrozen_params.append(param)
        print(f"  Unfreezing: {name} - {param.shape}")

# Safety check
if len(unfrozen_params) == 0:
    raise ValueError("No parameters were unfrozen! Check parameter names.")

print(f"\n✅ Total unfrozen parameters: {len(unfrozen_params)}")
print(f"   Unfrozen params count: {sum(p.numel() for p in unfrozen_params):,}\n")

optimizer = torch.optim.Adam(unfrozen_params, # Only these will be updated by the optimizer
                            lr=1e-4, betas=(0.9, 0.95), weight_decay=0.01) # learning rate and betas
# ============================================================================
# Initialize Latents
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
# GRPO Training steps
# ============================================================================
# Track initial weights for comparison at the end
initial_weights = {}
for name, param in model.named_parameters():
    if param.requires_grad:
        initial_weights[name] = param.data.clone()
        
print(f"📊 Tracking {len(initial_weights)} unfrozen parameters\n")

num_rollouts = 3 
for i, t in enumerate(timesteps):
    print(f"Step {i+1:02d}/{len(timesteps)} | t={t:.4f}", end="")
        
    rollout_noise_preds = []  # Store noise predictions for later gradient computation
    rollout_rewards = []
        
    for rollout_index in range(num_rollouts):
        #####==========================================================
        # STEP 1:Predict with current model weights to get next_latents and x0_est
        #####==========================================================
        with torch.no_grad():  # No gradients during rollout sampling
            # Add small noise perturbation to latents for variation between rollouts
            # This is crucial for GRPO to get different rewards
            if rollout_index > 0:  # Keep first rollout deterministic
                noise_scale = 0.02  # Small perturbation
                latents_perturbed = latents + torch.randn_like(latents) * noise_scale
            else:
                latents_perturbed = latents
            
            noise_pred = model(
                latents_perturbed,  # Use perturbed latents
                indices_grid=indices_grid,
                encoder_hidden_states=prompt_embeds,
                encoder_attention_mask=prompt_attention_mask,
                timestep=t,
                return_dict=False,
            )[0] # current model weights
            
            # Store noise prediction for later
            rollout_noise_preds.append(noise_pred.clone())
            
            # Denoise to get x0 estimate (variation comes from perturbed latents above)
            next_latents, x0_est = pipeline.denoising_step(
                latents=latents,  # Use original latents for denoising
                noise_pred=noise_pred,
                current_timestep=None,
                conditioning_mask=None,
                t=t,
                extra_step_kwargs={},
                stochastic_sampling=False,
                return_x0=True,
            )
            
            # Use helper function to get video_x0, and compute reward
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
            reward = reward_function(video_x0)
            rollout_rewards.append(reward)
            
            ###================================================================== 
            # Clear GPU memory after each rollout to prevent OOM
            del video_x0, x0_est, next_latents
            torch.cuda.empty_cache()
    ###==================================================================
    # STEP 2: compute the advantage reward for each rollout
    ###==================================================================
    rewards = torch.stack(rollout_rewards)
    mean_reward = rewards.mean()
    std_reward = rewards.std()
    # Group normalization: compute the advantage reward for each rollout
    advantage_rewards = (rewards - mean_reward) / (std_reward + 1e-4)
    print("\nAdvantages:")
    for k in range(num_rollouts):
        print(f"  Rollout {k}: reward={rewards[k]:.4f}, advantage={advantage_rewards[k]:.4f}")
    
    # Clear rollout data to free memory before backward pass
    del rollout_rewards, rewards, mean_reward, std_reward
    torch.cuda.empty_cache()
    
    #=======================================================================
    # STEP 3: Compute loss with a SINGLE forward pass (model is deterministic)
    ###==================================================================
    # Clear all gradients completely before computing loss
    optimizer.zero_grad()
    model.zero_grad()
    
    # Ensure latents has no gradient history from previous timesteps
    latents_for_loss = latents.detach().clone().requires_grad_(True)
    
    # Since the model is deterministic, one forward pass represents all rollouts
    noise_pred_for_loss = model(
        latents_for_loss,
        indices_grid=indices_grid.detach(),
        encoder_hidden_states=prompt_embeds.detach(),
        encoder_attention_mask=prompt_attention_mask.detach(),
        timestep=t,
        return_dict=False,
    )[0]
    
    log_prob = -0.5 * (noise_pred_for_loss ** 2).sum() / noise_pred_for_loss.numel()
    
    # Weight the log_prob by the mean advantage (representing all rollouts)
    loss = -(log_prob * advantage_rewards.detach().mean())
    # Backprop (gradients flow to unfrozen to_out parameters!)
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
    del loss, log_prob, noise_pred_for_loss, latents_for_loss
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
        return_x0=True
    )
    
    # Decode to video
    print("Decoding final video...")
    final_video = decode_x0_to_video(
        x0_final,
        pipeline,
        num_frames=num_frames,
        height=height,
        width=width,
        is_patchified=True,
    )
    
    # Save video to file
    output_filename = f"outputs/final_video_{timestamp}.mp4"
    
    # Convert tensor to numpy for saving
    import numpy as np
    import imageio
    
    # Convert bfloat16 to float32, then to numpy
    video_np = final_video[0].float().cpu().numpy()  # [num_frames, 3, H, W]
    
    print(f"  [DEBUG] Video tensor stats before transpose:")
    print(f"    Shape: {video_np.shape}")
    print(f"    Min: {video_np.min():.4f}, Max: {video_np.max():.4f}, Mean: {video_np.mean():.4f}")
    print(f"    Per-channel: R={video_np[:, 0].mean():.4f}, G={video_np[:, 1].mean():.4f}, B={video_np[:, 2].mean():.4f}")
    
    video_np = np.transpose(video_np, (0, 2, 3, 1))  # [num_frames, H, W, 3]
    # Video is already in [0, 1] range from decode_x0_to_video
    video_np = (video_np * 255).clip(0, 255).astype(np.uint8)
    
    # Save as MP4 with explicit format and quality settings
    writer = imageio.get_writer(
        output_filename, 
        fps=frame_rate,
        codec='libx264',
        quality=8,  # Higher quality
        pixelformat='yuv420p',  # Standard color format
        macro_block_size=1
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
        