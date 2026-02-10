# ⚡ Quick Start: LoRA for Unified GRPO

## 🎯 Goal
Run GRPO training on your 40GB GPU without OOM errors!

---

## 📦 Step 1: Install PEFT

```bash
cd /home/ghr/angel/full_sequence_grpo

# Option A: Use installer script
bash install_lora.sh

# Option B: Manual install
conda activate cogvideo
pip install peft>=0.7.0
```

---

## 🚀 Step 2: Run Training

```bash
# Just run it!
bash run.sh
```

That's it! The script is already configured with LoRA enabled.

---

## ✅ What to Expect

### Console Output

You should see:

```
======================================================================
Applying LoRA Adapters
======================================================================
  Rank: 16
  Alpha: 32
  Target modules: ['to_q', 'to_k', 'to_v', 'to_out.0']
  Dropout: 0.05

trainable params: 10,485,760 || all params: 2,010,485,760 || trainable%: 0.52

  LoRA Statistics:
    LoRA parameters: 10,485,760
    Total parameters: 2,010,485,760
    LoRA %: 0.52%
======================================================================

✅ LoRA applied (memory-efficient training!)
```

### Memory Usage

- **With LoRA:** ~30GB VRAM ✅
- **Without LoRA:** ~45GB VRAM ❌ (OOM on 40GB)

---

## ⚙️ Configuration Options

### Current Settings (in `run.sh`)

```bash
--use-lora \          # Enable LoRA
--lora-rank 16 \      # Balanced quality/memory
--lora-alpha 32 \     # Standard (2×rank)
--num-rollouts 1 \    # Can increase to 3 if memory allows
--num-grpo-steps 5    # Number of GRPO optimization steps
```

### To Increase Quality (If You Have VRAM Headroom)

```bash
--lora-rank 32 \      # Better quality (uses ~5GB more)
--num-rollouts 3      # Better GRPO (uses ~8GB more)
```

### To Reduce Memory (If Still OOM)

```bash
--lora-rank 8 \       # Minimal (saves ~2GB)
--height 360 \        # Lower resolution (saves ~10GB)
--width 640
```

---

## 🔧 Disable LoRA (48GB+ GPU Required)

Edit `run.sh`:

```bash
# Remove these lines:
#   --use-lora \
#   --lora-rank 16 \
#   --lora-alpha 32 \

# Add this instead:
    --train-blocks "22,23,24,25,26,27,28,29" \
```

---

## 📊 Memory Cheat Sheet

| Config | VRAM | 40GB OK? |
|--------|------|----------|
| LoRA rank=8 | ~28GB | ✅ Safe |
| **LoRA rank=16** | **~30GB** | ✅ **Recommended** |
| LoRA rank=32 | ~35GB | ✅ Should work |
| Unfrozen (1 block) | ~38GB | ⚠️ Risky |
| Unfrozen (8 blocks) | ~45GB | ❌ OOM |

---

## 🐛 Troubleshooting

### "ModuleNotFoundError: No module named 'peft'"

```bash
pip install peft
```

### Still getting OOM with LoRA?

1. **Check rank:**
   ```bash
   grep "lora-rank" run.sh
   # Should show: --lora-rank 16
   ```

2. **Reduce to rank=8:**
   ```bash
   # Edit run.sh, change:
   --lora-rank 8 \
   ```

3. **Check rollouts:**
   ```bash
   grep "num-rollouts" run.sh
   # Should show: --num-rollouts 1
   ```

### LoRA not being applied?

Look for this in output:
```
======================================================================
Applying LoRA Adapters
======================================================================
```

If missing, check that `--use-lora` is in your command.

---

## 📚 More Information

- **Detailed guide:** `cat LORA_GUIDE.md`
- **Implementation details:** `cat IMPLEMENTATION_SUMMARY.md`
- **LoRA code:** `unified_grpo/lora_utils.py`

---

## 🎉 Success Criteria

After running, you should see:

```
Step 46/50 | Timestep: 99.0000
======================================================================
  Generating 1 rollouts...
    Rollout 1/1...
    reward=0.4875
  Computing GRPO loss and updating model...
  ✅ Model updated | Loss: 0.1234
```

**Without OOM errors!** 🎊

---

## 💡 Pro Tips

1. **Start with rank=16** (balanced)
2. **Monitor VRAM** with `nvidia-smi`
3. **Save LoRA weights** (~100MB checkpoints!)
4. **Use more rollouts** when memory allows (better GRPO)

---

## 🚀 You're Ready!

```bash
bash run.sh
```

Happy training! 🎯
