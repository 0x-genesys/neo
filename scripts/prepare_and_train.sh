#!/bin/bash

# Prepare and Train Script for Modern 200M Conversational Scholar Model
# This script automates the entire process from dataset preparation to training

set -e  # Exit on error

echo "=============================================================================="
echo "CONVERSATIONAL SCHOLAR DATASET PREPARATION AND TRAINING"
echo "=============================================================================="
echo ""
echo "This script will:"
echo "  1. Install required dependencies"
echo "  2. Prepare 300M token Conversational Scholar dataset"
echo "  3. Verify dataset integrity"
echo "  4. Start training on modern 200M model"
echo ""
echo "Estimated time: 3-5 hours (depending on internet speed and hardware)"
echo ""

# Check if user wants to continue
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

# Step 1: Install dependencies
echo ""
echo "=============================================================================="
echo "STEP 1: Installing dependencies"
echo "=============================================================================="
echo ""

pip install -q tiktoken datasets numpy tqdm pyyaml transformers torch

echo "✅ Dependencies installed"

# Step 2: Prepare dataset
echo ""
echo "=============================================================================="
echo "STEP 2: Preparing Conversational Scholar dataset (300M tokens)"
echo "=============================================================================="
echo ""
echo "This will download and process:"
echo "  - WikiText/factual data (240M training tokens)"
echo "  - UltraChat (60M training tokens + 10M validation tokens)"
echo ""
echo "Expected time: 2-4 hours"
echo ""

python scripts/prepare_balanced_dataset.py \
    --output-dir data/balanced_300m \
    --seed 42

if [ $? -ne 0 ]; then
    echo "❌ Dataset preparation failed!"
    exit 1
fi

echo "✅ Dataset prepared successfully"

# Step 3: Verify dataset
echo ""
echo "=============================================================================="
echo "STEP 3: Verifying dataset integrity"
echo "=============================================================================="
echo ""

python scripts/test_balanced_dataset.py \
    --data-dir data/balanced_300m

if [ $? -ne 0 ]; then
    echo "❌ Dataset verification failed!"
    exit 1
fi

echo "✅ Dataset verified successfully"

# Step 4: Start training
echo ""
echo "=============================================================================="
echo "STEP 4: Starting training"
echo "=============================================================================="
echo ""
echo "Training configuration:"
echo "  Model: modern 200M architecture"
echo "  Dataset: 300M tokens (80% WikiText/factual, 20% UltraChat)"
echo "  Epochs: 8"
echo "  Total tokens: 2.4B"
echo "  Steps: 36,621"
echo ""
echo "Expected time: 10-20 hours (1.5GB GPU) or 1-2 hours (larger GPU)"
echo ""

read -p "Start training now? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Training skipped. To start training later, run:"
    echo "  python train.py --config config/auto_training_200m_modern.yaml"
    exit 0
fi

python train.py --config config/auto_training_200m_modern.yaml

echo ""
echo "=============================================================================="
echo "✅ TRAINING COMPLETE!"
echo "=============================================================================="
echo ""
echo "Next steps:"
echo "  1. Evaluate model: python evaluate.py --checkpoint checkpoints/gpu_training_117m_1.5gb/best_model.pt"
echo "  2. Generate text: python src/inference.py --checkpoint checkpoints/gpu_training_117m_1.5gb/best_model.pt"
echo "  3. View logs: tensorboard --logdir logs/gpu_training_117m_1.5gb"
echo ""
