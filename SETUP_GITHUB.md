# Push to GitHub Guide

## Step 1: Check Current Status

```bash
cd /home/ubuntu/angel-research/full_sequence_grpo

# Check what files would be committed
git status

# Check for any large files
find . -type f -size +50M -not -path "./.git/*" | head -20
```

## Step 2: Initialize Git (if not already done)

```bash
# Check if git is initialized
if [ ! -d .git ]; then
    git init
    echo "✓ Git initialized"
else
    echo "✓ Git already initialized"
fi
```

## Step 3: Add Remote (Create repo on GitHub first)

1. Go to https://github.com/new
2. Create a new repository (e.g., `ltx-video-grpo`)
3. Don't initialize with README (we have files already)
4. Copy the repository URL

```bash
# Add remote (replace with your repo URL)
git remote add origin https://github.com/YOUR_USERNAME/ltx-video-grpo.git

# Or if using SSH:
git remote add origin git@github.com:YOUR_USERNAME/ltx-video-grpo.git

# Verify remote
git remote -v
```

## Step 4: Stage and Commit Files

```bash
# Add all files (respects .gitignore)
git add .

# Check what will be committed
git status

# Commit
git commit -m "Initial commit: GRPO training for LTX-Video

- Implemented GRPO training loop with physics-based rewards
- Added baseline comparison script
- Includes helper functions for reward computation (DINO features, optical flow, gravity physics)
- Configured for 5-second video generation (81 frames @ 16 fps)
- Memory optimized for H100 80GB GPU
"
```

## Step 5: Push to GitHub

```bash
# Push to main branch
git branch -M main
git push -u origin main
```

## Step 6: Verify Upload

Go to your GitHub repository and verify:
- ✓ Python files uploaded
- ✓ Config files uploaded
- ✗ No model checkpoints (>100MB)
- ✗ No video outputs
- ✗ No training logs

## Optional: Add README

Create a `README.md` with project description:

```bash
cat > README.md << 'EOF'
# LTX-Video GRPO Training

Physics-based reinforcement learning for video generation using Group Relative Policy Optimization (GRPO).

## Features

- **GRPO Training**: Fine-tunes LTX-Video model with physics-based rewards
- **Physics Rewards**: Evaluates gravity, motion smoothness, velocity realism
- **Baseline Comparison**: Generates comparison videos with/without training

## Quick Start

```bash
# Generate baseline
python generate_baseline.py

# Train with GRPO
./run_training.sh
```

## Requirements

- GPU: 80GB+ (H100/A100)
- CUDA: 12.2+
- Python: 3.10+

See `requirements.txt` for dependencies.
EOF

git add README.md
git commit -m "Add README"
git push
```

## Troubleshooting

### Large files rejected

If GitHub rejects files >100MB:

```bash
# Find large files
find . -size +100M -not -path "./.git/*"

# Add to .gitignore
echo "path/to/large/file" >> .gitignore

# Remove from git cache
git rm --cached path/to/large/file

# Commit and push
git add .gitignore
git commit -m "Remove large files"
git push
```

### Use Git LFS for large files

If you need to track large model files:

```bash
# Install Git LFS
git lfs install

# Track model files
git lfs track "*.safetensors"
git lfs track "*.pth"

# Commit .gitattributes
git add .gitattributes
git commit -m "Add Git LFS tracking"
git push
```

## Files Included

- `pipeline.py` - Main GRPO training loop
- `generate_baseline.py` - Baseline video generation
- `helper.py` - Reward functions (DINO, optical flow, physics)
- `run_training.sh` - Training script with memory optimization
- `prompt.txt` - Test prompts
- `configs/` - LTX-Video model configurations

## Files Excluded (.gitignore)

- Model checkpoints (*.pth, *.safetensors)
- Video outputs (outputs/, baseline_output/)
- Training logs
- Cache files
- Large data files (>100MB)

