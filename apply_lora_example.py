#!/usr/bin/env python3
"""
Example: How to modify pipeline.py to use LoRA

Replace the unfreezing section (lines ~168-192) with this code.
"""

# ============================================================================
# STEP 1: Choose LoRA or Traditional Unfreezing
# ============================================================================
USE_LORA = True  # Toggle between LoRA and traditional

# ============================================================================
# After loading model in pipeline.py (after line ~97)
# ============================================================================

if USE_LORA:
    # ========================================================================
    # LoRA Method (Recommended for multiple layers)
    # ========================================================================
    from lora_config import apply_lora_to_model, get_lora_config_motion_focused
    
    print("\n" + "="*70)
    print("APPLYING LORA FOR STABLE FINE-TUNING")
    print("="*70 + "\n")
    
    # Choose your configuration
    config = get_lora_config_motion_focused()  # For motion/physics
    # config = get_lora_config_lightweight()   # For fast testing
    # config = get_lora_config_comprehensive()  # For comprehensive training
    
    # Apply LoRA to model
    model, recommended_lr = apply_lora_to_model(model, **config)
    pipeline.transformer = model  # Update pipeline reference
    
    # Create optimizer with LoRA-specific LR
    optimizer = torch.optim.Adam(
        model.parameters(),  # PEFT automatically filters trainable params
        lr=recommended_lr,
        betas=(0.9, 0.95),
        weight_decay=0.01
    )
    
    print(f"✅ LoRA initialized with LR={recommended_lr:.2e}\n")

else:
    # ========================================================================
    # Traditional Method (Current approach)
    # ========================================================================
    print("\n" + "="*70)
    print("TRADITIONAL UNFREEZING (proj_out only)")
    print("="*70 + "\n")
    
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze specific layers
    unfrozen_params = []
    for name, param in model.named_parameters():
        if "proj_out" in name:
            param.requires_grad = True
            unfrozen_params.append(param)
            print(f"  Unfreezing: {name} - {param.shape}")
    
    if len(unfrozen_params) == 0:
        raise ValueError("No parameters were unfrozen!")
    
    print(f"\n✅ Total unfrozen parameters: {len(unfrozen_params)}")
    print(f"   Unfrozen params count: {sum(p.numel() for p in unfrozen_params):,}\n")
    
    # Create optimizer
    optimizer = torch.optim.Adam(
        unfrozen_params,
        lr=1e-4,
        betas=(0.9, 0.95),
        weight_decay=0.01
    )

# ============================================================================
# The rest of pipeline.py continues unchanged!
# ============================================================================
# GRPO training loop, denoising, rewards, etc. all work the same!

