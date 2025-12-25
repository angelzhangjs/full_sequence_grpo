# LTX-Video GRPO Training - Complete Setup Guide

## 🎯 **Quick Start (Fresh Environment)**

### **Step 1: Create Conda Environment**

```bash
# Create environment with Python 3.10
conda create -n ltx-grpo python=3.10 -y
conda activate ltx-grpo
```

### **Step 2: Install PyTorch with CUDA**

```bash
# Install PyTorch 2.9.1 with CUDA 11.8 support
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu118
```

### **Step 3: Install Core Dependencies**

```bash
# Install from requirements
pip install -r requirements.txt
```

### **Step 4: Install CLIP from Source**

```bash
# CLIP must be installed from GitHub
pip install git+https://github.com/openai/CLIP.git
```

### **Step 5: Install LTX-Video**

```bash
cd ltx_video
pip install -e .
cd ..
```

### **Step 6: Verify Installation**

```bash
python -c "import torch; print(f'✓ PyTorch: {torch.__version__}'); print(f'✓ CUDA available: {torch.cuda.is_available()}')"
python -c "from ltx_video.inference import create_ltx_video_pipeline; print('✓ LTX-Video OK')"
python -c "import clip; print('✓ CLIP OK')"
python -c "from reward_functions import reward_function; print('✓ Reward functions OK')"
```

---

## 📦 **Key Package Versions**

```
Python: 3.10+
PyTorch: 2.9.1
CUDA: 11.8+
diffusers: 0.36.0
transformers: 4.57.3
clip: 1.0
ltx-video: 0.1.2
```

---

## 💾 **Hardware Requirements**

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **GPU** | 24GB VRAM | 80GB (A100) |
| **RAM** | 32GB | 64GB+ |
| **Storage** | 20GB | 50GB+ |
| **CUDA** | 11.8 | 12.0+ |

---

## 🚀 **Running Training**

```bash
# Activate environment
conda activate ltx-grpo

# Navigate to directory
cd /home/ubuntu/angel-research/full_sequence_grpo

# Run training
bash pipeline.sh
```

---

## 📝 **Files Generated**

After installation and first run:

```
full_sequence_grpo/
├── requirements.txt          # Pip freeze output
├── requirements_annotated.txt # With comments & instructions
├── grpo/                     # Training outputs
│   ├── final_video_*.mp4
│   └── training_log_*.txt
└── baseline/                 # Baseline comparison
    └── video_output_*.mp4
```

---

## 🐛 **Troubleshooting**

### **Issue: CUDA Out of Memory**
```bash
# Reduce frames or resolution in pipeline.py:
num_frames = 17  # Instead of 81
```

### **Issue: CLIP not found**
```bash
pip install git+https://github.com/openai/CLIP.git
```

### **Issue: LTX-Video not found**
```bash
cd ltx_video
pip install -e .
```

### **Issue: Slow download**
```bash
# Set HuggingFace cache location
export HF_HOME=/path/to/large/storage
```

---

## ✅ **Verification Checklist**

- [ ] Python 3.10+ installed
- [ ] CUDA 11.8+ available
- [ ] PyTorch with CUDA support
- [ ] All packages from requirements.txt
- [ ] CLIP installed from GitHub
- [ ] LTX-Video installed (editable mode)
- [ ] GPU has 24GB+ VRAM
- [ ] 50GB+ free storage

---

## 📊 **Package Summary**

Total packages: **65**

**Categories:**
- PyTorch & CUDA: 17 packages
- HuggingFace ecosystem: 8 packages  
- Video processing: 4 packages
- Reward models (CLIP/DINO): 2 packages
- Utilities: 34 packages

**Total install time:** ~10-15 minutes  
**Total download size:** ~15-20 GB


