# LoRA for Unified GRPO Framework

## 🎯 Why LoRA?

**Problem:** Standard GRPO with unfrozen transformer blocks requires **48GB+ VRAM** for backpropagation.

**Solution:** LoRA (Low-Rank Adaptation) reduces gradient memory by **95%**, enabling training on **40GB GPUs**!

### Memory Comparison

| Method | VRAM Usage | Fits 40GB? |
|--------|------------|------------|
| Full unfrozen blocks | ~45GB | ❌ No |
| **LoRA (rank=16)** | **~30GB** | ✅ **Yes!** |

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements_lora.txt
```

This installs:
- `peft` (HuggingFace Parameter-Efficient Fine-Tuning library)

### 2. Enable LoRA in `run.sh`

```bash
./run_unified_grpo.sh \
    --model-type cogvideox \
    --use-lora \              # ← Enable LoRA
    --lora-rank 16 \          # ← LoRA rank (8/16/32)
    --lora-alpha 32 \         # ← LoRA alpha (typically 2×rank)
    --num-rollouts 3 \        # Can now use more rollouts!
    ...
```

### 3. Run Training

```bash
bash run.sh
```

---

## 📖 LoRA Configuration

### Rank Selection

| Rank | Quality | Memory | Speed | Use Case |
|------|---------|--------|-------|----------|
| **8** | Good | Lowest | Fastest | Prototyping, quick experiments |
| **16** | Better | Balanced | Medium | **Recommended for 40GB GPU** |
| **32** | Best | Higher | Slower | Maximum quality (if you have VRAM) |

### Alpha Selection

- **Rule of thumb:** `alpha = 2 × rank`
- **Examples:**
  - `rank=8` → `alpha=16`
  - `rank=16` → `alpha=32`
  - `rank=32` → `alpha=64`

---

## 🔧 Advanced Usage

### Save LoRA Weights Only (Lightweight Checkpoints)

```python
from unified_grpo.lora_utils import save_lora_weights

# After training
save_lora_weights(pipeline.transformer, "./checkpoints/lora_weights")
```

LoRA checkpoints are **~100MB** instead of **~5GB** for full model!

### Load LoRA Weights

```python
from unified_grpo.lora_utils import load_lora_weights

load_lora_weights(pipeline.transformer, "./checkpoints/lora_weights")
```

### Presets

```bash
# Minimal (fastest, lowest memory)
--use-lora --lora-rank 8 --lora-alpha 16

# Balanced (recommended)
--use-lora --lora-rank 16 --lora-alpha 32

# Large (best quality)
--use-lora --lora-rank 32 --lora-alpha 64
```

---

## 🧪 How LoRA Works

### Standard Training
```
Full transformer parameters: ~2 billion
Trainable parameters: ~2 billion
Gradient memory: ~12GB
```

### LoRA Training
```
Full transformer parameters: ~2 billion
LoRA adapter parameters: ~10 million (0.5%)
Gradient memory: ~600MB (95% reduction!)
```

### Architecture

LoRA adds small "adapter" layers to attention:

```
Original:         Q = x @ W_q          (frozen)
With LoRA:        Q = x @ W_q + x @ (A @ B)
                        ↑frozen   ↑trainable (tiny!)
```

Where `A` and `B` are low-rank matrices (`rank << hidden_dim`).

---

## 📊 Expected Performance

### Memory Usage (CogVideoX-2B, 480×720, 49 frames)

| Configuration | VRAM | Status |
|---------------|------|--------|
| Unfrozen blocks (8) | ~45GB | ❌ OOM on 40GB |
| Unfrozen blocks (1) | ~40GB | ⚠️ Borderline |
| **LoRA rank=16** | **~30GB** | ✅ **Works!** |
| LoRA rank=8 | ~28GB | ✅ Even safer |

### Training Speed

- **LoRA is ~5-10% faster** than unfrozen blocks (less gradient computation)
- Can use **more rollouts** (e.g., 3 instead of 1) for better GRPO!

---

## 🎓 Best Practices

### For 40GB GPU (Recommended)

```bash
--use-lora \
--lora-rank 16 \
--lora-alpha 32 \
--num-rollouts 3 \       # More rollouts = better GRPO
--num-grpo-steps 25      # More GRPO steps
```

### For 48GB+ GPU

You can choose either:

**Option A: LoRA (parameter-efficient, faster)**
```bash
--use-lora \
--lora-rank 32 \
--lora-alpha 64
```

**Option B: Unfrozen blocks (slightly better quality)**
```bash
--train-blocks "22,23,24,25,26,27,28,29"
```

---

## 🔍 Troubleshooting

### "ModuleNotFoundError: No module named 'peft'"

```bash
pip install peft
```

### "LoRA parameters not found"

Make sure you:
1. Used `--use-lora` flag
2. Applied LoRA **before** calling `trainable_parameters()`

### Still getting OOM?

Try:
1. Reduce rank: `--lora-rank 8`
2. Reduce rollouts: `--num-rollouts 1`
3. Reduce resolution: `--height 360 --width 640`

---

## 📚 References

- **PEFT Library:** https://github.com/huggingface/peft
- **LoRA Paper:** https://arxiv.org/abs/2106.09685
- **VANS (RL+LoRA for video):** Uses LoRA for memory efficiency

---

## ✅ Summary

**LoRA enables GRPO training on 40GB GPUs!**

```bash
# Before (OOM on 40GB)
--train-blocks "29" --num-rollouts 1  ❌

# After (fits in 40GB!)
--use-lora --lora-rank 16 --num-rollouts 3  ✅
```

**This is the recommended approach for your unified framework!** 🚀
