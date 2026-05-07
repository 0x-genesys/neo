#!/bin/bash
# Test script for chat inference

echo "=================================="
echo "Chat Inference Test Script"
echo "=================================="
echo ""

# Check PyTorch version
echo "1. Checking PyTorch version..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
echo ""

# Check if base model exists
BASE_MODEL="checkpoints/best_model.pt"
if [ ! -f "$BASE_MODEL" ]; then
    echo "⚠️  Base model not found: $BASE_MODEL"
    echo "   Please train the base model first:"
    echo "   python train.py --config config/auto_training_117m_balanced.yaml"
    echo ""
    exit 1
fi

# Check if adapter exists
ADAPTER="finetuned_model_gpu/best_model"
if [ ! -d "$ADAPTER" ]; then
    echo "⚠️  Adapter not found: $ADAPTER"
    echo "   Please run fine-tuning first:"
    echo "   python src/finetuning/gpu_finetune.py"
    echo ""
    exit 1
fi

echo "2. Testing single prompt..."
python src/finetuning/chat_inference.py \
    --base-model "$BASE_MODEL" \
    --adapter "$ADAPTER" \
    --prompt "What is 2+2?" \
    --max-tokens 100

echo ""
echo "=================================="
echo "✅ Test complete!"
echo ""
echo "To start interactive mode:"
echo "python src/finetuning/chat_inference.py \\"
echo "    --base-model $BASE_MODEL \\"
echo "    --adapter $ADAPTER \\"
echo "    --interactive"
echo "=================================="
