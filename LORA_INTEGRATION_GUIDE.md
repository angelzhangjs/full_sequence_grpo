# LoRA Integration Guide for GRPO Training

## 🚀 Quick Start

### Step 1: Install PEFT
```bash
conda activate ltx-grpo
pip install peft
```

### Step 2: Verify Installation
```bash
python -c "from peft import get_peft_model, LoraConfig; print('✅ PEFT installed!')"
```

### Step 3: Modify pipeline.py

Find this section (around line 168-192):
```python
# ============================================================================
# unfreeze the model
# ============================================================================

# freeze all parameters in the model first
for param in model.parameters():
    param.requires_grad = False

# unfreeze the parameters of the output projection layer of the transformer 
unfrozen_params = []
for name, param in model.named_parameters():
    if "proj_out" in name:
        param.requires_grad = True
        unfrozen_params.append(param)
        print(f"  Unfreezing: {name} - {param.shape}")

# ... optimizer creation ...
```

**Replace with:**
```python
# ============================================================================
# Choose Unfreezing Method
# ============================================================================
USE_LORA = False  # Set to True to enable LoRA

if USE_LORA:
    from lora_config import apply_lora_to_model, get_lora_config_motion_focused
    
    config = get_lora_config_motion_focused()  # Choose config here
    model, recommended_lr = apply_lora_to_model(model, **config)
    pipeline.transformer = model
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=recommended_lr,
        betas=(0.9, 0.95),
        weight_decay=0.01
    )
else:
    # Traditional unfreezing (current method)
    for param in model.parameters():
        param.requires_grad = False
    
    unfrozen_params = []
    for name, param in model.named_parameters():
        if "proj_out" in name:
            param.requires_grad = True
            unfrozen_params.append(param)
            print(f"  Unfreezing: {name} - {param.shape}")
    
    print(f"\n✅ Total unfrozen parameters: {len(unfrozen_params)}")
    print(f"   Unfrozen params count: {sum(p.numel() for p in unfrozen_params):,}\n")
    
    optimizer = torch.optim.Adam(
        unfrozen_params,
        lr=1e-4,
        betas=(0.9, 0.95),
        weight_decay=0.01
    )
```

## 📊 Available Configurations

### 1. Motion-Focused (Recommended for your use case)
```python
config = get_lora_config_motion_focused()
# - Self-attention in last 5 blocks
# - Best for: motion, physics, temporal consistency
# - Params: ~1-2M
# - Time: ~25-35 min
```

### 2. Lightweight (Fast testing)
```python
config = get_lora_config_lightweight()
# - Self-attention in last 2 blocks
# - Best for: quick experiments
# - Params: ~400-800K
# - Time: ~18-22 min
```

### 3. Comprehensive (Maximum capability)
```python
config = get_lora_config_comprehensive()
# - Both self & cross attention, 5 blocks
# - Best for: full fine-tuning
# - Params: ~2-3M
# - Time: ~35-50 min
```

### 4. Text-Focused (Prompt conditioning)
```python
config = get_lora_config_text_focused()
# - Cross-attention in last block
# - Best for: better prompt following
# - Params: ~500K-1M
# - Time: ~20-25 min
```

## 🎯 Usage Examples

### Example 1: Switch to LoRA
```python
# In pipeline.py, change line ~168:
USE_LORA = True  # Enable LoRA

# Run training:
python pipeline.py
```

### Example 2: Switch Back to Traditional
```python
# In pipeline.py:
USE_LORA = False  # Use traditional proj_out unfreezing

# Run training:
python pipeline.py
```

### Example 3: Try Different Configs
```python
if USE_LORA:
    # Try lightweight first
    config = get_lora_config_lightweight()
    
    # If results good, try motion-focused
    # config = get_lora_config_motion_focused()
    
    # For best results, try comprehensive
    # config = get_lora_config_comprehensive()
    
    model, recommended_lr = apply_lora_to_model(model, **config)
```

## 💾 Saving/Loading LoRA Checkpoints

### Save LoRA adapters:
```python
from lora_config import save_lora_checkpoint

# After training
save_lora_checkpoint(model, "grpo/lora_checkpoint")
```

### Load LoRA adapters:
```python
from lora_config import load_lora_checkpoint

# Load back
model = load_lora_checkpoint(base_model, "grpo/lora_checkpoint")
```

### Merge LoRA into base model:
```python
from lora_config import merge_lora_to_base

# For inference (faster, single model)
model = merge_lora_to_base(model)
```

## ⚖️ Comparison

| Aspect | Traditional (proj_out) | LoRA (motion-focused) |
|--------|------------------------|------------------------|
| **Params** | 262K | ~1-2M |
| **Layers** | 1 (proj_out) | 20 (attn1 in 5 blocks) |
| **Time** | 15 min | 25-35 min |
| **Stability** | High | Very High |
| **Motion Quality** | Good | Potentially Better |
| **Color Quality** | Working (0.39→0.86) | Similar |
| **Flexibility** | Limited | High |

## 🎯 Recommendation

1. **Finish current training** (proj_out) - see final results
2. **If motion needs improvement** - try LoRA lightweight
3. **If lightweight works** - upgrade to motion-focused
4. **Compare results** - see if LoRA gives better motion/physics

## 📝 Notes

- LoRA doesn't change your GRPO training loop
- All reward functions work the same
- Just changes WHICH parameters are trained
- Easy to switch back and forth (toggle USE_LORA)



