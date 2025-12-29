#!/usr/bin/env python3
"""
LoRA Configuration for GRPO Training
Provides Low-Rank Adaptation for stable fine-tuning of LTX-Video model.

Usage:
    from lora_config import apply_lora_to_model
    
    # After loading your model:
    model = apply_lora_to_model(
        model, 
        target_layers='self_attn',  # or 'cross_attn' or 'both'
        num_blocks=5,               # Last N blocks to apply LoRA
        rank=16                     # LoRA rank
    )
"""

from peft import LoraConfig, TaskType
from peft.tuners.lora import LoraModel
import re
import torch.nn as nn

def apply_lora_to_model(
    model,
    target_layers='self_attn',  # Options: 'self_attn', 'cross_attn', 'both', 'proj_out'
    num_blocks=5,               # Number of last blocks to apply LoRA (0 = all blocks)
    rank=16,                    # LoRA rank (lower = fewer params, higher = more capacity)
    lora_alpha=16,              # LoRA scaling factor (typically same as rank)
    lora_dropout=0.05,          # Dropout for LoRA layers
    learning_rate_scale=0.3,    # Scale down LR for LoRA (0.3 = 30% of original)
):
    """
    Apply LoRA (Low-Rank Adaptation) to specified layers in the model.
    
    Args:
        model: The transformer model to apply LoRA to
        target_layers: Which attention layers to target
            - 'self_attn': Self-attention (attn1) - best for motion/physics
            - 'cross_attn': Cross-attention (attn2) - best for text conditioning
            - 'both': Both self and cross attention
            - 'proj_out': Just output projection (fallback to traditional)
        num_blocks: Number of last blocks to apply LoRA to (0 = all blocks)
        rank: LoRA rank (4-64, typically 8-32)
        lora_alpha: Scaling factor
        lora_dropout: Dropout rate
        learning_rate_scale: Recommended LR scale (return value for reference)
        
    Returns:
        model: Model wrapped with LoRA
        recommended_lr: Suggested learning rate
    """
    
    # Determine target module names based on layer type
    target_modules = []
    
    if target_layers == 'self_attn':
        # Self-attention (attn1) modules in this checkpoint
        target_modules = [
            "attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0",
        ]
        print("🎯 Targeting: Self-Attention (attn1) - Motion & Physics")
        
    elif target_layers == 'cross_attn':
        # Cross-attention (attn2) modules in this checkpoint
        target_modules = [
            "attn2.to_q", "attn2.to_k", "attn2.to_v", "attn2.to_out.0",
        ]
        print("🎯 Targeting: Cross-Attention (attn2) - Text Conditioning")
        
    elif target_layers == 'both':
        # Both self and cross attention
        target_modules = [
            "attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0",
            "attn2.to_q", "attn2.to_k", "attn2.to_v", "attn2.to_out.0",
        ]
        print("🎯 Targeting: Both Self & Cross Attention - Comprehensive")
        
    elif target_layers == 'proj_out':
        # Just output projection (manual unfreezing, not LoRA)
        print("⚠️  'proj_out' selected - LoRA not needed, use manual unfreezing")
        print("   Returning original model without LoRA")
        return model, 1e-4
    
    else:
        raise ValueError(f"Unknown target_layers: {target_layers}. Use 'self_attn', 'cross_attn', 'both', or 'proj_out'")
    
    # Create LoRA configuration
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",  # Don't apply LoRA to biases
        task_type=TaskType.CAUSAL_LM,  # required field; we won't use the LM wrapper
    )

    # Apply LoRA to model using LoraModel to preserve the original forward signature
    print(f"\n📦 Applying LoRA to model...")
    print(f"   Rank: {rank}")
    print(f"   Alpha: {lora_alpha}")
    print(f"   Dropout: {lora_dropout}")
    print(f"   Target modules: {len(target_modules)} patterns")
    
    model = LoraModel(model, lora_config, adapter_name="default")

    # Optionally restrict LoRA to the last `num_blocks` transformer blocks
    if num_blocks > 0:
        block_indices = set()
        for name, _ in model.named_parameters():
            m = re.search(r"transformer_blocks\.(\d+)\.", name)
            if m:
                block_indices.add(int(m.group(1)))

        if block_indices:
            max_idx = max(block_indices)
            allowed = {i for i in block_indices if i >= max_idx - num_blocks + 1}
            kept = skipped = 0
            for name, p in model.named_parameters():
                if "lora_" in name:
                    m = re.search(r"transformer_blocks\.(\d+)\.", name)
                    if m and int(m.group(1)) not in allowed:
                        p.requires_grad = False
                        skipped += 1
                    else:
                        kept += 1

            span = f"{min(allowed)}..{max(allowed)}" if allowed else "n/a"
            print(f"   LoRA block filtering: keeping last {len(allowed)} blocks ({span})")
            print(f"     Trainable LoRA params: {kept}")
            print(f"     Frozen LoRA params:    {skipped}")
        else:
            print("⚠️  num_blocks set but no transformer_blocks.* found; applying LoRA to all matched modules.")
    
    # Print trainable parameters
    print(f"\n✅ LoRA Applied!")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = (trainable / total) * 100 if total else 0
    print(f"   Trainable params: {trainable} / {total} ({pct:.2f}%)")

    # Calculate recommended learning rate
    recommended_lr = 1e-4 * learning_rate_scale
    print(f"\n📚 Recommended learning rate: {recommended_lr:.2e}")
    print(f"   (Original 1e-4 × {learning_rate_scale} scale factor)")
    
    return model, recommended_lr


def save_lora_checkpoint(model, path):
    """Save only the LoRA adapters (not the full model)"""
    model.save_pretrained(path)
    print(f"💾 LoRA checkpoint saved to: {path}")


def load_lora_checkpoint(model, path):
    """Load LoRA adapters onto base model"""
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, path)
    print(f"📂 LoRA checkpoint loaded from: {path}")
    return model


def merge_lora_to_base(model):
    """Merge LoRA weights back into base model (for inference)"""
    model = model.merge_and_unload()
    print("🔄 LoRA merged into base model")
    return model

# ============================================================================
# Example Configurations
# ============================================================================

def get_lora_config_motion_focused():
    """LoRA config optimized for motion and physics (self-attention only)"""
    return {
        'target_layers': 'self_attn',
        'num_blocks': 5,
        'rank': 16,
        'lora_alpha': 16,
        'lora_dropout': 0.05,
        'learning_rate_scale': 0.3,
    }


def get_lora_config_text_focused():
    """LoRA config optimized for text conditioning (cross-attention only)"""
    return {
        'target_layers': 'cross_attn',
        'num_blocks': 1,  # Just last block
        'rank': 32,  # Higher rank for text understanding
        'lora_alpha': 32,
        'lora_dropout': 0.05,
        'learning_rate_scale': 0.5,
    }

def get_lora_config_comprehensive():
    """LoRA config for both motion and text (hybrid approach)"""
    return {
        'target_layers': 'both',
        'num_blocks': 5,  # Last 5 blocks
        'rank': 16,
        'lora_alpha': 16,
        'lora_dropout': 0.1,  # Higher dropout for more params
        'learning_rate_scale': 0.25,  # Lower LR for stability
    }


def get_lora_config_lightweight():
    """Minimal LoRA for testing (fast, low memory)"""
    return {
        'target_layers': 'self_attn',
        'num_blocks': 2,  # Just last 2 blocks
        'rank': 8,  # Lower rank
        'lora_alpha': 8,
        'lora_dropout': 0.05,
        'learning_rate_scale': 0.5,
    }


if __name__ == "__main__":
    print("LoRA Configuration Module")
    print("\nAvailable configurations:")
    print("  1. Motion-Focused (self-attention, 5 blocks, rank=16)")
    print("  2. Text-Focused (cross-attention, 1 block, rank=32)")
    print("  3. Comprehensive (both, 5 blocks, rank=16)")
    print("  4. Lightweight (self-attention, 2 blocks, rank=8)")
    print("\nUsage in pipeline.py:")
    print("  from lora_config import apply_lora_to_model, get_lora_config_motion_focused")
    print("  config = get_lora_config_motion_focused()")
    print("  model, recommended_lr = apply_lora_to_model(model, **config)")

