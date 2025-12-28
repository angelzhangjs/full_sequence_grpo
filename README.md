 # LTX-Video GRPO Training – Complete Setup Guide
 
 GRPO training pipeline for LTX-Video focused on physics-aware, text-aligned video generation.
 
 ## 🎯 Overview
 - Per-timestep GRPO for LTX-Video with CLIP, DINO, and physics rewards
 - Baseline comparison to avoid quality regressions
 - Fits 80GB GPUs; supports shorter configs for 24–48GB cards
 
 ## 🚀 Quick Start (Fresh Environment)
 ### Step 1: Create Conda Environment
 ```bash
 conda create -n ltx-grpo python=3.10 -y
 conda activate ltx-grpo
 ```
 
 ### Step 2: Install PyTorch with CUDA
 ```bash
 pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu118
 ```
 
 ### Step 3: Install Core Dependencies
 ```bash
 pip install -r requirements.txt
 ```
 
 ### Step 4: Install CLIP from Source
 ```bash
 pip install git+https://github.com/openai/CLIP.git
 ```
 
 ### Step 5: Install LTX-Video
 ```bash
 cd ltx_video
 pip install -e .
 cd ..
 ```
 
 ### Step 6: Verify Installation
 ```bash
 python -c "import torch; print(f'✓ PyTorch: {torch.__version__}'); print(f'✓ CUDA available: {torch.cuda.is_available()}')"
 python -c "from ltx_video.inference import create_ltx_video_pipeline; print('✓ LTX-Video OK')"
 python -c "import clip; print('✓ CLIP OK')"
 python -c "from reward_functions import reward_function; print('✓ Reward functions OK')"
 ```
 
 ## 📦 Key Package Versions
 ```
 Python: 3.10+
 PyTorch: 2.9.1
 CUDA: 11.8+
 diffusers: 0.36.0
 transformers: 4.57.3
 clip: 1.0
 ltx-video: 0.1.2
 ```
 
 ## 🚀 Running Training
 ```bash
 conda activate ltx-grpo
 cd /home/ubuntu/angel-research/full_sequence_grpo
 
 # Optional: set a custom prompt
 echo "A bright red ball bouncing down stairs" > prompt.txt
 
 # Launch GRPO training
 bash pipeline.sh
 ```
 
 ## 📝 Configuration Tips
 - Adjust training knobs in `pipeline.py` (`NUM_GRPO_STEPS`, `lr`, `num_rollouts`, `num_frames`).
 - For quicker experiments on smaller GPUs, lower `num_frames` (e.g., 17–33) and resolution.
 
 ## 💾 Hardware Requirements
 | Component | Minimum | Recommended |
 |-----------|---------|-------------|
 | **GPU** | 24GB VRAM | 80GB (A100) |
 | **RAM** | 32GB | 64GB+ |
 | **Storage** | 20GB | 50GB+ |
 | **CUDA** | 11.8 | 12.0+ |
 
 Typical memory usage: ~20–25 GB during GRPO steps; video decoding ~10–15 GB; peak ~35–40 GB.
 
 ## 📝 Files Generated
 ```
 full_sequence_grpo/
 ├── requirements.txt             # Pip requirements
 ├── requirements_annotated.txt   # Notes and guidance
 ├── grpo/                        # Training outputs
 │   ├── final_video_*.mp4
 │   └── training_log_*.txt
 └── baseline/                    # Baseline outputs
     └── video_output_*.mp4
 ```
 
 ## 🐛 Troubleshooting
 - **CUDA OOM:** Reduce `num_frames` or resolution in `pipeline.py`.
 - **CLIP not found:** `pip install git+https://github.com/openai/CLIP.git`
 - **LTX-Video not found:** `cd ltx_video && pip install -e .`
 - **Slow downloads:** `export HF_HOME=/path/to/large/storage`
 - **Quality regressions:** Try lower `lr` (1e-6 to 1e-5) or compare against baseline before updating.
 
 ## ✅ Verification Checklist
 - [ ] Python 3.10+ with CUDA 11.8+
 - [ ] PyTorch with CUDA installed
 - [ ] Dependencies from `requirements.txt`
 - [ ] CLIP installed from GitHub
 - [ ] LTX-Video installed (editable)
 - [ ] GPU ≥24GB VRAM and ≥50GB free storage
 
 ## 📊 Package Summary
 Total packages: **65**
 - PyTorch & CUDA: 17
 - HuggingFace ecosystem: 8
 - Video processing: 4
 - Reward models (CLIP/DINO): 2
 - Utilities: 34
 
 Total install time: ~10–15 minutes; download size: ~15–20 GB.
 
 ## 🙏 Acknowledgments
 - Lightricks team for LTX-Video
 - OpenAI for CLIP
 - Meta for DINOv2
 - Diffusers library maintainers
