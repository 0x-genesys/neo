#!/bin/bash

# Prepare and Train Script for 117M Model with Balanced Dataset
# This script automates the entire process from dataset preparation to training

set -e  # Exit on error

echo "=============================================================================="
echo "BALANCED DATASET PREPARATION AND TRAINING"
echo "=============================================================================="
echo ""
echo "This script will:"
echo "  1. Install required dependencies"
echo "  2. Prepare 300M token balanced dataset"
echo "  3. Verify dataset integrity"
echo "  4. Start training on 117M model"
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
echo "STEP 2: Preparing balanced dataset (300M tokens)"
echo "=============================================================================="
echo ""
echo "This will download and process:"
echo "  - WikiText-103 (102M tokens)"
echo "  - UltraChat (150M tokens)"
echo "  - The Stack (48M tokens)"
echo "  - DailyDialog (10M tokens)"
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
echo "  Model: 117M parameters"
echo "  Dataset: 300M tokens (balanced)"
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
    echo "  python train.py --config config/gpu_training_117m_1.5gb.yaml"
    exit 0
fi

python train.py --config config/gpu_training_117m_1.5gb.yaml

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
