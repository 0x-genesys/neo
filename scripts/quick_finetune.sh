#!/bin/bash
# Quick Fine-Tuning Script for 117M Transformer with LoRA + CoT
# Automatically prepares data and starts fine-tuning

set -e  # Exit on error

echo "=================================="
echo "🚀 Quick Fine-Tuning Setup"
echo "=================================="
echo ""

# Check if data exists
if [ ! -f "data/cot_train.jsonl" ]; then
    echo "📊 Training data not found. Creating sample dataset..."
    python scripts/prepare_finetuning_data.py all \
        --output-dir data \
        --math-samples 1000 \
        --code-samples 500 \
        --qa-samples 500 \
        --val-ratio 0.1
    echo ""
fi

# Check if pre-trained model exists
if [ ! -f "checkpoints/best_model.pt" ]; then
    echo "⚠️  Pre-trained model not found at checkpoints/best_model.pt"
    echo "   Please train the base model first or update the checkpoint path."
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Detect hardware
echo "🔍 Detecting hardware..."
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected"
    DEVICE="cuda"
elif python -c "import torch; exit(0 if torch.backends.mps.is_available() else 1)" 2>/dev/null; then
    echo "✅ Apple MPS detected"
    DEVICE="mps"
else
    echo "⚠️  No GPU detected, using CPU"
    DEVICE="cpu"
fi
echo ""

# Start fine-tuning
echo "=================================="
echo "🎓 Starting Fine-Tuning"
echo "=================================="
echo ""

if [ "$DEVICE" = "cpu" ]; then
    echo "⚠️  CPU training is slow. Consider using a GPU."
    echo ""
    python src/finetuning/cpu_finetune.py
else
    python src/finetuning/gpu_finetune.py
fi

echo ""
echo "=================================="
echo "✅ Fine-Tuning Complete!"
echo "=================================="
