#!/bin/bash
# Emergency OOM Fix Script

echo "🚨 CUDA Out of Memory Fix"
echo "=========================="
echo ""

# Set memory optimization
echo "✅ Setting PYTORCH_CUDA_ALLOC_CONF..."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
echo ""

# Check GPUs
echo "📊 Checking GPU status..."
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader,nounits
echo ""

# Show options
echo "🎯 Choose a solution:"
echo ""
echo "1. Low Memory Config (Recommended)"
echo "   - batch_size: 8, context: 384"
echo "   - Memory: ~8-9GB per GPU"
echo "   - Speed: Moderate"
echo ""
echo "2. Ultra Low Memory Config"
echo "   - batch_size: 4, context: 256"
echo "   - Memory: ~6-7GB per GPU"
echo "   - Speed: Slow but very safe"
echo ""
echo "3. Single GPU (GPU 0 only)"
echo "   - Uses only 1 GPU"
echo "   - Memory: ~8-9GB"
echo "   - Speed: 2x slower"
echo ""

read -p "Enter choice (1-3): " choice

case $choice in
    1)
        echo ""
        echo "🚀 Starting training with Low Memory config..."
        echo "   Config: config/gpu_training_117m_balanced_low_memory.yaml"
        echo ""
        python train.py --config config/gpu_training_117m_balanced_low_memory.yaml
        ;;
    2)
        echo ""
        echo "🚀 Starting training with Ultra Low Memory..."
        echo "   Creating temporary config..."
        
        # Create ultra low memory config
        cat > /tmp/ultra_low_memory.yaml << 'EOF'
model:
  vocab_size: 100277
  d_model: 768
  num_heads: 12
  num_layers: 12
  context_length: 256
  dropout: 0.12
  use_flash_attention: false
  use_gradient_checkpointing: true

training:
  batch_size: 4
  gradient_accumulation_steps: 32
  max_epochs: 8
  max_steps: 36621
  learning_rate: 2.0e-4
  weight_decay: 0.15
  warmup_steps: 1500
  max_grad_norm: 1.0
  dropout: 0.12
  eval_interval: 1000
  save_interval: 1000
  log_interval: 10

data:
  dataset_type: "binary"
  train_file: "data/balanced_300m/train.bin"
  val_file: "data/balanced_300m/val.bin"
  max_length: 256
  num_workers: 4
  pin_memory: true
  huggingface_dataset:
    repo_id: "0x-genesys/mix_wiki_code_chat_data_300M_tokens"
    dataset_name: "balanced_300m"
    auto_download: true

tokenizer:
  type: "tiktoken"
  vocab_size: 100277

optimizer:
  type: "adamw"
  betas: [0.9, 0.95]
  eps: 1.0e-8

scheduler:
  type: "cosine"
  min_lr: 2.0e-5

system:
  device: "cuda"
  mixed_precision: true
  compile_model: false
  seed: 42

checkpoint:
  save_dir: "checkpoints/ultra_low_memory"
  resume_from: null
  save_best_only: false

logging:
  use_wandb: false
  log_dir: "logs/ultra_low_memory"

huggingface_hub:
  enabled: false

generation:
  max_new_tokens: 100
  temperature: 0.8
  top_k: 50
  top_p: 0.95
  num_samples: 3
EOF
        
        python train.py --config /tmp/ultra_low_memory.yaml
        ;;
    3)
        echo ""
        echo "🚀 Starting training with Single GPU..."
        echo "   Using GPU 0 only"
        echo ""
        export CUDA_VISIBLE_DEVICES=0
        python train.py --config config/gpu_training_117m_balanced.yaml
        ;;
    *)
        echo "❌ Invalid choice"
        exit 1
        ;;
esac