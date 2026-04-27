#!/bin/bash

# Quick Start Script for Transformer Training
# This script will train a small model on tiny_shakespeare dataset

set -e

echo "=================================="
echo "🚀 Quick Start: Transformer Training"
echo "=================================="
echo ""

# Activate virtual environment
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
else
    echo "❌ Virtual environment not found!"
    echo "Please run: bash setup.sh"
    exit 1
fi

# Check if dependencies are installed
echo "Checking dependencies..."
python3 -c "import torch; import transformers; import datasets" 2>/dev/null || {
    echo "❌ Dependencies not installed!"
    echo "Please run: bash setup.sh"
    exit 1
}
echo "✅ Dependencies OK"
echo ""

# Test device
echo "Testing device..."
python3 src/device_utils.py
echo ""

# Create a quick start config
echo "Creating quick start configuration..."
cat > config/quick_start.yaml << 'EOF'
# Quick Start Configuration - Very small model for fast testing
model:
  vocab_size: 50257
  d_model: 128        # Very small
  num_heads: 4
  num_layers: 2       # Only 2 layers
  context_length: 128 # Short context
  dropout: 0.1

training:
  batch_size: 8
  gradient_accumulation_steps: 2
  max_epochs: 2       # Just 2 epochs
  max_steps: 500      # Stop after 500 steps
  learning_rate: 5.0e-4
  weight_decay: 0.01
  warmup_steps: 50
  max_grad_norm: 1.0
  eval_interval: 100
  save_interval: 250
  log_interval: 10

data:
  dataset_name: "wikitext"
  dataset_config: "wikitext-2-raw-v1"
  train_split: "train"
  val_split: "validation"
  test_split: "test"
  max_length: 128
  num_workers: 2

tokenizer:
  type: "gpt2"
  vocab_size: 50257

optimizer:
  type: "adamw"
  betas: [0.9, 0.95]
  eps: 1.0e-8

scheduler:
  type: "cosine"
  min_lr: 5.0e-5

system:
  device: "auto"
  mixed_precision: false  # Disabled for compatibility
  compile_model: false
  seed: 42

checkpoint:
  save_dir: "checkpoints/quick_start"
  resume_from: null

logging:
  use_wandb: false
  log_dir: "logs/quick_start"

generation:
  max_new_tokens: 50
  temperature: 0.8
  top_k: 50
  top_p: 0.95
  num_samples: 2
EOF

echo "✅ Configuration created"
echo ""

# Start training
echo "Starting training..."
echo "This will train a small model on WikiText-2 (Wikipedia articles)"
echo "Expected time: 1-2 hours on CPU, 10-20 minutes on GPU"
echo ""
echo "Press Ctrl+C to stop training at any time"
echo ""

python train.py --config config/quick_start.yaml

echo ""
echo "=================================="
echo "✅ Quick Start Complete!"
echo "=================================="
echo ""
echo "Your model has been trained and saved to: checkpoints/quick_start/"
echo ""
echo "To test the model:"
echo "  python src/inference.py --model checkpoints/quick_start/best_model.pt --interactive"
echo ""
echo "To continue training with a larger dataset:"
echo "  1. Edit config/model_config.yaml"
echo "  2. Change dataset_name to 'wikitext' or 'openwebtext'"
echo "  3. Run: python train.py --config config/model_config.yaml"
echo ""
echo "=================================="
