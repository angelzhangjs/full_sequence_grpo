"""
LoRA Utilities for Unified GRPO Framework
Enables parameter-efficient training with LoRA adapters
"""

import torch
from typing import Optional, List

def _import_peft():
    """
    Import PEFT with a small compatibility shim.

    Some PEFT versions expect `accelerate.utils.memory.clear_device_cache`, which is missing in older accelerate
    releases (e.g. accelerate==0.29.x). We provide a best-effort implementation so the import succeeds.
    """
    try:
        from accelerate.utils import memory as _acc_mem  # type: ignore

        if not hasattr(_acc_mem, "clear_device_cache"):

            def clear_device_cache(*args, **kwargs):  # type: ignore
                # Minimal behavior: clear CUDA cache if available.
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            _acc_mem.clear_device_cache = clear_device_cache  # type: ignore[attr-defined]
    except Exception:
        # If accelerate isn't installed or its internals changed, let PEFT import fail with its original error.
        pass

    from peft import get_peft_model, LoraConfig  # type: ignore

    return get_peft_model, LoraConfig


def apply_lora_to_transformer(
    transformer,
    rank: int = 16,
    alpha: int = 32,
    target_modules: Optional[List[str]] = None,
    lora_dropout: float = 0.05,
    target_blocks: Optional[List[int]] = None,
):
    """
    Apply LoRA adapters to transformer model
    
    Args:
        transformer: The transformer model
        rank: LoRA rank (controls capacity, 8-32 typical)
        alpha: LoRA alpha (controls scaling, typically 2×rank)
        target_modules: Which modules to target (default: attention projections)
        lora_dropout: Dropout for LoRA layers
        target_blocks: Optional list of block indices to apply LoRA to (e.g., [29])
                      If None, applies to ALL blocks
        
    Returns:
        Modified transformer with LoRA
        List of LoRA parameters
    """
    if target_modules is None:
        # Default: Target all attention projections
        # CogVideoX, LTX, Hunyuan use: to_q, to_k, to_v, to_out
        target_modules = ["to_q", "to_k", "to_v", "to_out.0"]
    
    # If specific blocks requested, we'll need to find exact module names.
    # PEFT doesn't support wildcards, so we enumerate matching modules.
    if target_blocks is not None:
        # Get all module names from the transformer
        all_module_names = [name for name, _ in transformer.named_modules()]
        
        # Build list of exact module names matching our pattern
        block_specific_modules = []

        # Family A: diffusers-like models (CogVideoX, LTX, etc): transformer_blocks.{i}.attn1/attn2.to_q...
        has_transformer_blocks = any(n.startswith("transformer_blocks.") for n in all_module_names)
        # Family B: Wan2.1: blocks.{i}.self_attn/cross_attn.q/k/v/o (+ optional k_img/v_img)
        has_wan_blocks = any(n.startswith("blocks.") for n in all_module_names) and any(
            ".self_attn." in n for n in all_module_names
        )

        if has_transformer_blocks:
            for block_idx in target_blocks:
                for module_suffix in target_modules:
                    # Match patterns like: "transformer_blocks.27.attn1.to_q"
                    # CogVideoX has attn1 (self-attn) and attn2 (cross-attn)
                    for attn_type in ["attn1", "attn2"]:
                        pattern = f"transformer_blocks.{block_idx}.{attn_type}.{module_suffix}"
                        if pattern in all_module_names:
                            block_specific_modules.append(pattern)
        elif has_wan_blocks:
            for block_idx in target_blocks:
                for module_suffix in target_modules:
                    for attn_type in ["self_attn", "cross_attn"]:
                        pattern = f"blocks.{block_idx}.{attn_type}.{module_suffix}"
                        if pattern in all_module_names:
                            block_specific_modules.append(pattern)
        
        if len(block_specific_modules) == 0:
            print(f"  ⚠️  Warning: No modules found for blocks {target_blocks}")
            print("  Available transformer_blocks modules:")
            for name in all_module_names:
                if "transformer_blocks" in name and "to_q" in name:
                    print(f"    {name}")
            raise ValueError(f"No modules found matching blocks {target_blocks}")
        
        target_modules = block_specific_modules
    
    print(f"\n{'='*70}")
    print("Applying LoRA Adapters")
    print(f"{'='*70}")
    print(f"  Rank: {rank}")
    print(f"  Alpha: {alpha}")
    if target_blocks is not None:
        print(f"  Target blocks: {target_blocks} (block-specific LoRA)")
    else:
        print("  Target blocks: ALL (full LoRA)")
    print(f"  Target modules: {target_modules}")
    print(f"  Dropout: {lora_dropout}")
    
    get_peft_model, LoraConfig = _import_peft()

    # Create LoRA config
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=None,  # Not for specific task type
    ) 
    # Apply LoRA
    transformer_lora = get_peft_model(transformer, lora_config)

    # IMPORTANT for VRAM:
    # Some PEFT builds will cast activations to the LoRA weights' dtype. If LoRA weights end up in fp32,
    # this can silently upcast large activations to fp32 and cause OOM. Keep LoRA weights in the base dtype.
    try:
        base_dtype = getattr(transformer, "dtype", None)
        if base_dtype is None:
            base_dtype = next(transformer.parameters()).dtype
        transformer_lora.to(dtype=base_dtype)
        print(f"  ✓ Cast PEFT/LoRA modules to base dtype: {base_dtype}")
    except Exception as e:
        print(f"  ⚠️  Warning: failed to cast LoRA modules to base dtype: {e}")
    
    # Print trainable parameters
    transformer_lora.print_trainable_parameters()
    
    # Get LoRA parameters
    lora_params = [p for n, p in transformer_lora.named_parameters() if 'lora_' in n.lower()]
    
    total_lora = sum(p.numel() for p in lora_params)
    total_model = sum(p.numel() for p in transformer_lora.parameters())
    
    print("\n  LoRA Statistics:")
    print(f"    LoRA parameters: {total_lora:,}")
    print(f"    Total parameters: {total_model:,}")
    print(f"    LoRA %: {100*total_lora/total_model:.2f}%")
    if target_blocks:
        print(f"    Applied to blocks: {target_blocks}")
    print(f"{'='*70}\n")
    
    return transformer_lora, lora_params


def has_lora(model) -> bool:
    """
    Best-effort check for whether a model has LoRA/PEFT adapters attached.
    """
    try:
        if hasattr(model, "peft_config"):
            return True
        return any("lora_" in n.lower() for n, _ in model.named_parameters())
    except Exception:
        return False


def get_trainable_lora_parameters(
    model,
    *,
    ensure_requires_grad: bool = True,
    verbose_prefix: Optional[str] = None,
) -> List[torch.nn.Parameter]:
    """
    Return the list of LoRA parameters to optimize for a PEFT/LoRA-wrapped model.

    This consolidates common adapter logic:
    - Prefer params that already have requires_grad=True (what PEFT usually sets up)
    - If none are marked trainable, optionally set requires_grad=True for all LoRA params
      to be defensive across PEFT/model versions.
    """
    prefix = (str(verbose_prefix) + " ") if verbose_prefix else ""

    # First pass: only params already marked trainable.
    lora_params = [p for n, p in model.named_parameters() if "lora_" in n.lower() and p.requires_grad]
    if len(lora_params) > 0:
        if verbose_prefix is not None:
            print(f"{prefix}Using LoRA parameters for training ({len(lora_params)} tensors)")
        return lora_params

    # Second pass: be defensive and force LoRA params trainable.
    if ensure_requires_grad:
        forced: List[torch.nn.Parameter] = []
        for n, p in model.named_parameters():
            if "lora_" in n.lower():
                p.requires_grad_(True)
                forced.append(p)
        if verbose_prefix is not None:
            print(f"{prefix}Using LoRA parameters for training ({len(forced)} tensors; forced requires_grad=True)")
        return forced

    if verbose_prefix is not None:
        print(f"{prefix}LoRA detected but no trainable LoRA params found (requires_grad all False).")
    return []


def get_lora_parameters(model):
    """
    Extract LoRA parameters from a model
    
    Args:
        model: Model with LoRA applied
        
    Returns:
        List of LoRA parameters (for optimizer)
    """
    lora_params = []
    
    for name, param in model.named_parameters():
        # LoRA parameters have 'lora_' in their name
        if 'lora_' in name.lower() and param.requires_grad:
            lora_params.append(param)
    
    return lora_params


def save_lora_weights(model, save_path: str):
    """
    Save only LoRA weights (lightweight checkpoint)
    
    Args:
        model: Model with LoRA
        save_path: Where to save
    """
    # PEFT provides save_pretrained for LoRA
    if hasattr(model, 'save_pretrained'):
        model.save_pretrained(save_path)
        print(f"✅ LoRA weights saved to: {save_path}")
    else:
        # Manual save
        lora_state = {
            name: param.data 
            for name, param in model.named_parameters() 
            if 'lora_' in name.lower()
        }
        torch.save(lora_state, save_path)
        print(f"✅ LoRA weights saved to: {save_path}")


def load_lora_weights(model, load_path: str):
    """
    Load LoRA weights
    
    Args:
        model: Model with LoRA structure
        load_path: Path to LoRA weights
    """
    if hasattr(model, 'load_adapter'):
        model.load_adapter(load_path)
        print(f"✅ LoRA weights loaded from: {load_path}")
    else:
        # Manual load
        lora_state = torch.load(load_path)
        model.load_state_dict(lora_state, strict=False)
        print(f"✅ LoRA weights loaded from: {load_path}")

# ============================================================================
# LoRA Configuration Presets
# ============================================================================

def get_lora_config_preset(preset: str = "balanced") -> dict:
    """
    Get LoRA configuration presets
    
    Presets:
    - minimal: r=8, alpha=16 (smallest, fastest)
    - balanced: r=16, alpha=32 (recommended)
    - large: r=32, alpha=64 (best quality, slower)
    """
    presets = {
        "minimal": {
            "rank": 8,
            "alpha": 16,
            "dropout": 0.05,
        },
        "balanced": {
            "rank": 16,
            "alpha": 32,
            "dropout": 0.05,
        },
        "large": {
            "rank": 32,
            "alpha": 64,
            "dropout": 0.1,
        },
    }
    
    if preset not in presets:
        print(f"⚠️  Unknown preset '{preset}', using 'balanced'")
        preset = "balanced"
    
    return presets[preset]


if __name__ == "__main__":
    print("LoRA Utilities for Unified GRPO")
    print("\nAvailable functions:")
    print("  - apply_lora_to_transformer()")
    print("  - get_lora_parameters()")
    print("  - save_lora_weights()")
    print("  - load_lora_weights()")
    print("\nPresets: minimal, balanced, large")
