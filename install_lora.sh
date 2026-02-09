#!/bin/bash
# Install LoRA dependencies for Unified GRPO Framework

set -e

echo "======================================================================="
echo "Installing LoRA Dependencies"
echo "======================================================================="
echo ""

# Check if in conda environment
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "⚠️  Warning: No conda environment detected"
    echo "   Please activate your environment first:"
    echo "   conda activate cogvideo"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "✓ Conda environment: $CONDA_DEFAULT_ENV"
fi

echo ""
echo "Installing PEFT (Parameter-Efficient Fine-Tuning)..."
pip install peft>=0.7.0

echo ""
echo "======================================================================="
echo "✅ Installation Complete!"
echo "======================================================================="
echo ""
echo "Testing installation..."

python -c "
try:
    from peft import get_peft_model, LoraConfig
    print('✅ PEFT imported successfully')
    print('✅ LoRA utilities available')
except ImportError as e:
    print(f'❌ Import failed: {e}')
    exit(1)
"

echo ""
echo "Next steps:"
echo "  1. Run training: bash run.sh"
echo "  2. Read guide: cat LORA_GUIDE.md"
echo "  3. See examples: cat IMPLEMENTATION_SUMMARY.md"
echo ""
echo "🚀 Ready for LoRA training!"
