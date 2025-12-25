# LTX-Video GRPO Training for Physics-Based Video Generation

Deep reinforcement learning (GRPO) training for text-to-video diffusion models with physics-aware rewards.

## 🎯 **Overview**

This repository implements **Group Relative Policy Optimization (GRPO)** for training video diffusion models to generate physically realistic videos with proper motion dynamics.

**Key Features:**
- ✅ Per-timestep GRPO training for LTX-Video
- ✅ Comprehensive reward functions (CLIP + DINO + Physics)
- ✅ Baseline comparison to prevent quality degradation
- ✅ Memory-optimized for 80GB GPUs
- ✅ Supports multiple video generation models

---

## 🚀 **Quick Start**

### **Prerequisites**

- Python 3.10+
- CUDA 11.8+
- GPU with 24GB+ VRAM (A100/A6000 recommended)
- 64GB+ system RAM

### **Installation**

```bash
# 1. Create conda environment
conda create -n ltx-grpo python=3.10 -y
conda activate ltx-grpo

# 2. Install PyTorch with CUDA
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu118

# 3. Install dependencies
cd full_sequence_grpo
pip install -r requirements.txt

# 4. Install CLIP
pip install git+https://github.com/openai/CLIP.git

# 5. Install LTX-Video
cd ltx_video
pip install -e .
cd ..
```

### **Run Training**

```bash
# Set your prompt
echo "A bright red ball bouncing down stairs" > prompt.txt

# Run GRPO training
bash pipeline.sh
```

**Outputs:**
- `grpo/final_video_*.mp4` - Trained model output
- `baseline/video_output_*.mp4` - Pretrained baseline
- `grpo/training_log_*.txt` - Training details

---

## 📊 **Project Structure**

```
full_sequence_grpo/
├── pipeline.py                  # Main GRPO training script
├── reward_functions.py          # CLIP+DINO+Physics rewards
├── helper.py                    # Video decoding utilities
├── configs/                     # Model configurations
│   └── ltxv-2b-0.9.6-dev-grpo.yaml
├── grpo/                        # Training outputs
├── baseline/                    # Baseline comparisons
├── requirements.txt             # Package dependencies
└── SETUP_INSTRUCTIONS.md        # Detailed setup guide
```

---

## 🎨 **Reward Functions**

### **Multi-Modal Reward System:**

1. **CLIP (Text-Video Alignment)**
   - Measures semantic alignment with prompt
   - Ensures video matches description

2. **DINO (Object Tracking)**
   - Tracks object consistency across frames
   - Ensures coherent object identity

3. **Physics Dynamics**
   - Velocity and acceleration realism
   - Trajectory smoothness
   - Motion consistency

4. **Video Quality**
   - Brightness and color saturation
   - Sharpness and detail
   - Overall visual quality

---

## 🔬 **Training Algorithm**

### **Per-Timestep GRPO:**

```python
for timestep in last_N_timesteps:
    # 1. Generate baseline (frozen model)
    baseline_video = frozen_model.denoise(latents, timestep)
    baseline_reward = reward_function(baseline_video)
    
    # 2. Generate rollouts (trainable model)
    for rollout in range(3):
        video = model.denoise(latents, timestep)
        reward = reward_function(video)
    
    # 3. Compute advantages
    advantages = (rewards - mean) / std
    
    # 4. Update model
    loss = -advantages * mse_loss
    loss.backward()
    optimizer.step()
```

---

## 📈 **Results**

**Training Configuration:**
- Model: LTX-Video-2B
- Frames: 81 (5 seconds @ 16fps)
- Resolution: 512×768
- GRPO steps: Last 15 timesteps
- Learning rate: 1e-5 (conservative)

**Reward Components:**
- CLIP alignment: 0.24-0.29
- DINO consistency: 0.77-0.97
- Video quality: 0.52-0.68
- Physics scores: 0.55-0.70

---

## 🛠️ **Key Scripts**

| Script | Purpose |
|--------|---------|
| `pipeline.sh` | Complete training pipeline |
| `pipeline.py` | Per-timestep GRPO training |
| `reward_functions.py` | Comprehensive reward functions |
| `helper.py` | Video decoding utilities |

---

## 📝 **Configuration**

Edit `prompt.txt` to change the generation prompt:

```bash
echo "Your custom prompt here" > prompt.txt
```

Adjust training parameters in `pipeline.py`:
- `NUM_GRPO_STEPS`: How many timesteps to train (default: 15)
- `lr`: Learning rate (default: 1e-5)
- `num_rollouts`: Rollouts per timestep (default: 3)
- `num_frames`: Video length (default: 81 frames)

---

## 💾 **Hardware Requirements**

| GPU | Training | Inference |
|-----|----------|-----------|
| **A100 (80GB)** | ✅ Full training (81 frames) | ✅ Any length |
| **A6000 (48GB)** | ⚠️ Reduced frames (17-33) | ✅ Short videos |
| **RTX 4090 (24GB)** | ❌ Too small | ⚠️ Very short only |

**Memory usage:**
- Per-timestep training: ~20-25 GB
- Video decoding: ~10-15 GB per video
- Peak: ~35-40 GB

---

## 📚 **Documentation**

- `SETUP_INSTRUCTIONS.md` - Complete installation guide
- `requirements_annotated.txt` - Annotated dependencies
- Code comments throughout all scripts

---

## 🐛 **Known Issues & Solutions**

### **Issue: CUDA Out of Memory**
**Solution:** Reduce `num_frames` to 17-33 in `pipeline.py`

### **Issue: Videos still blurry**
**Possible causes:**
- Learning rate too small (increase to 1e-4)
- Need more unfrozen layers
- Stochastic sampling diversity issues

### **Issue: GRPO degrading quality**
**Solution:** Use very conservative LR (1e-6) or compare to baseline before updating

---

## 📖 **References**

- LTX-Video: [Lightricks/LTX-Video](https://github.com/Lightricks/LTX-Video)
- GRPO Paper: Group Relative Policy Optimization
- CLIP: [OpenAI/CLIP](https://github.com/openai/CLIP)
- DINOv2: [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2)

---

## 📄 **License**

See individual model licenses:
- LTX-Video: Check Lightricks license
- CLIP: MIT License
- DINOv2: Apache 2.0

---

## 🙏 **Acknowledgments**

- Lightricks team for LTX-Video
- OpenAI for CLIP
- Meta for DINOv2
- Diffusers library maintainers

---

## ⚡ **Quick Commands**

```bash
# Training
bash pipeline.sh

# Just baseline (no training)
cd ltx_video && python run_inference.py --pipeline_config configs/ltxv-2b-0.9.8-distilled-no-enhancer.yaml --prompt "$(cat ../prompt.txt)" --output_path ../baseline

# Check outputs
ls -lh grpo/final_video_*.mp4
ls -lh baseline/video_output_*.mp4
```

---

**For detailed setup instructions, see `SETUP_INSTRUCTIONS.md`**

**For issues and questions, check the training logs in `grpo/training_log_*.txt`**
