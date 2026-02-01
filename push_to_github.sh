#!/bin/bash
# Automated GitHub push script

cd /home/ubuntu/angel-research/full_sequence_grpo

echo "=========================================="
echo "GitHub Push Setup"
echo "=========================================="
echo ""

# Check for large files
echo "1. Checking for large files (>100MB)..."
LARGE_FILES=$(find . -type f -size +100M -not -path "./.git/*" 2>/dev/null)
if [ -n "$LARGE_FILES" ]; then
    echo "⚠️  WARNING: Found files >100MB:"
    echo "$LARGE_FILES"
    echo ""
    echo "These are already excluded by .gitignore"
else
    echo "✓ No large files found"
fi
echo ""

# Initialize git if needed
echo "2. Initializing Git..."
if [ ! -d .git ]; then
    git init
    echo "✓ Git initialized"
else
    echo "✓ Git already initialized"
fi
echo ""

# Check remote
echo "3. Checking remote..."
if git remote | grep -q origin; then
    echo "✓ Remote 'origin' exists:"
    git remote get-url origin
else
    echo "⚠️  No remote configured"
    echo ""
    read -p "Enter GitHub repository URL: " REPO_URL
    git remote add origin "$REPO_URL"
    echo "✓ Remote added: $REPO_URL"
fi
echo ""

# Show what will be committed
echo "4. Files to be committed:"
git add .
git status --short
echo ""

# Count files
TOTAL_FILES=$(git ls-files | wc -l)
echo "Total files: $TOTAL_FILES"
echo ""

# Confirm
read -p "Proceed with commit and push? (y/n): " CONFIRM
if [ "$CONFIRM" != "y" ]; then
    echo "Aborted."
    exit 0
fi

# Commit
echo ""
echo "5. Committing files..."
git commit -m "GRPO training implementation for LTX-Video

- GRPO training loop with physics-based rewards
- Baseline comparison script
- Helper functions for reward computation
- Memory optimized for 80GB GPU
- Configured for 5-second videos (81 frames)
"

# Push
echo ""
echo "6. Pushing to GitHub..."
git branch -M main
git push -u origin main

echo ""
echo "=========================================="
echo "✅ Successfully pushed to GitHub!"
echo "=========================================="
echo ""
echo "View your repository at:"
git remote get-url origin | sed 's/\.git$//'

