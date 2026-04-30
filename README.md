# Neo - Production-Ready Transformer Language Model

A robust, production-ready transformer language model implementation with comprehensive training, inference, and deployment capabilities.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train with pre-processed dataset (auto-downloads)
python train.py --config config/quick_start.yaml

# Resume from HuggingFace Hub
python train.py --config config/production_training.yaml --resume-remote checkpoint.pt

# Run inference
python src/inference.py --model checkpoints/production/best_model.pt --prompt "Once upon a time"

# Run inference in interactive mode with remote model

```

## 📦 Pre-trained Models & Datasets

### Models
- **Repository**: [0x-genesys/neo_weights_checkpoints](https://huggingface.co/0x-genesys/neo_weights_checkpoints)
- **Available checkpoints**: `checkpoint.pt`, `best_model.pt`, `checkpoint_step_*.pt`
- **Model sizes**: 117M parameters (GPT-2 Small architecture)

### Datasets
- **Repository**: [0x-genesys/mix_wiki_code_chat_data_300M_tokens](https://huggingface.co/datasets/0x-genesys/mix_wiki_code_chat_data_300M_tokens)
- **Size**: 300M tokens
- **Composition**:
  - WikiText-103: 102M tokens (34%) - Encyclopedic knowledge
  - UltraChat: 198M tokens (66%) - Conversational AI
  - The Stack: 48M tokens (16%) - Code reasoning

## ✨ Key Features

### 🎯 Training
- **Multi-GPU support** with DataParallel
- **TPU support** with dedicated TPU trainer (Kaggle TPU v3-8, Google Cloud TPU)
- **Cross-hardware checkpoints** - Resume GPU training on TPU and vice versa
- **Mixed precision (FP16/bfloat16)** training
- **Gradient checkpointing** for memory efficiency
- **Automatic dataset download** from HuggingFace Hub
- **Resume from remote checkpoints** on HuggingFace Hub
- **TensorBoard logging** and Weights & Biases integration
- **Automatic checkpoint upload** to HuggingFace Hub

### 🔧 Robustness
- **Multi-environment support**: CUDA, MPS (Apple Silicon), TPU, CPU
- **PyTorch version compatibility**: 2.0, 2.1, 2.2, 2.3, 2.4+
- **Automatic memory optimization** for different GPU sizes
- **Graceful error handling** and recovery
- **Comprehensive test suite**

### 📊 Inference
- **Local model loading**
- **Remote model loading** from HuggingFace Hub
- **Batch inference** support
- **Multiple sampling strategies** (temperature, top-k, top-p)

## 🛠️ Installation

```bash
# Clone repository
git clone https://github.com/yourusername/neo.git
cd neo

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python scripts/fix_environment.py
```

## 📚 Training

### Quick Start Training
```bash
# Small model, fast training (5 minutes)
python train.py --config config/quick_start.yaml
```

### Production Training
```bash
# Full training with balanced dataset
python train.py --config config/production_training.yaml
```

### Auto-Adaptive Training (Recommended)
```bash
# Automatically adapts to available hardware (TPU/GPU/MPS/CPU)
# Seamlessly resume across different hardware
python train.py --config config/auto_training_117m_balanced.yaml

# Resume on different hardware (e.g., GPU → TPU)
python train.py --config config/auto_training_117m_balanced.yaml \
    --resume checkpoints/auto_training_117m_balanced/checkpoint_step_7500.pt \
    --tpu
```

### GPU Training (Multi-GPU)
```bash
# Optimized for 2x T4 GPUs
python train.py --config config/gpu_training_117m_balanced.yaml --multi-gpu

# Low memory config (if OOM)
python train.py --config config/gpu_training_117m_balanced_low_memory.yaml --multi-gpu
```

### TPU Training (Kaggle/Google Cloud)

**Kaggle TPU v3-8** (Recommended - Free!):
```bash
# 1. Enable TPU in Kaggle notebook settings
# 2. Install torch_xla
!bash scripts/setup_kaggle_tpu.sh

# 3. Train on TPU (8 cores, 3x faster than T4 GPU)
!python train.py --config config/auto_training_117m_balanced.yaml --tpu

# 4. Resume from GPU checkpoint on TPU
!python train.py --config config/auto_training_117m_balanced.yaml \
    --resume-remote best_model_step_7500.pt \
    --tpu
```

**Google Cloud TPU**:
```bash
# Install torch_xla
pip install torch_xla

# Train on TPU (8 cores)
python train.py --config config/tpu_training_117m_balanced.yaml --tpu

# Specify number of TPU cores
python train.py --config config/tpu_training_117m_balanced.yaml --tpu --tpu-cores 8
```

**Performance**:
- **3.1x faster** than T4 GPU (2.5 steps/sec vs 0.8 steps/sec)
- **128GB memory** vs 16GB on T4
- **Batch size 128** vs 8 on T4
- **Free on Kaggle** (30 hours/week)

See [KAGGLE_TPU_SUPPORT.md](KAGGLE_TPU_SUPPORT.md) for complete guide.

### Resume Training

**From local checkpoint:**
```bash
python train.py --config config/production_training.yaml --resume checkpoints/production/checkpoint.pt
```

**From HuggingFace Hub:**
```bash
python train.py --config config/production_training.yaml --resume-remote checkpoint.pt
```

**From specific repository:**
```bash
python train.py --config config/production_training.yaml \
    --resume-remote checkpoint_step_4000.pt \
    --model-repo your-username/your-model-repo
```

### Command Line Options

```bash
python train.py \
    --config config/production_training.yaml \  # Config file
    --resume checkpoints/model.pt \             # Resume from local checkpoint
    --resume-remote checkpoint.pt \             # Resume from HuggingFace Hub
    --model-repo 0x-genesys/neo_weights \       # HuggingFace model repo
    --dataset data/custom/train.bin \           # Override dataset
    --batch-size 32 \                           # Override batch size
    --epochs 10 \                               # Override epochs
    --lr 3e-4 \                                 # Override learning rate
    --multi-gpu \                               # Use all GPUs
    --gpu-ids 0,1                               # Use specific GPUs
    --tpu \                                     # Use TPU
    --tpu-cores 8                               # Number of TPU cores
```

## 🔮 Inference

### Local Model
```bash
# Basic inference
python src/inference.py \
    --model checkpoints/production/best_model.pt \
    --prompt "Once upon a time"

# With custom parameters
python src/inference.py \
    --model checkpoints/production/best_model.pt \
    --prompt "The future of AI" \
    --max-length 200 \
    --temperature 0.8 \
    --top-k 50 \
    --top-p 0.95
```

### Remote Model (HuggingFace Hub)
```bash
# Load model from HuggingFace Hub (config auto-loaded from checkpoint)
python src/inference.py \
    --model-remote best_model.pt \
    --prompt "Once upon a time"

# Interactive mode
python src/inference.py \
    --model-remote best_model.pt \
    --interactive

# From specific repository
python src/inference.py \
    --model-remote checkpoint_step_4000.pt \
    --model-repo your-username/your-model-repo \
    --prompt "The future of AI"

# Override generation parameters
python src/inference.py \
    --model-remote best_model.pt \
    --temperature 1.0 \
    --top-k 100 \
    --max-tokens 200 \
    --interactive
```

**Note**: Config is automatically loaded from the checkpoint. You don't need to provide `--config` unless you want to override settings.

### Batch Inference
```bash
# Process multiple prompts
python src/inference.py \
    --model checkpoints/production/best_model.pt \
    --prompts prompts.txt \
    --output results.txt
```

### Interactive Inference
```bash
# From specific repository
python src/inference.py \
    --model-remote checkpoint_step_4000.pt \
    --model-repo your-username/your-model-repo \
    --interactive
```

### Interactive inference with config

```bash
venv/bin/python3 src/inference.py --model-remote best_model_step_2500.pt --interactive --config config/inference.yaml
```

## 📊 Monitoring

### TensorBoard
```bash
tensorboard --logdir logs/
```

### Weights & Biases
```yaml
# In config file
logging:
  use_wandb: true
  wandb_project: "your-project"
  wandb_entity: "your-username"
```

### GPU Monitoring
```bash
# Real-time GPU monitoring
watch -n 1 nvidia-smi

# Memory usage
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

## 🗂️ Project Structure

```
neo/
├── README.md                 # This file
├── ARCHITECTURE.md           # System architecture documentation
├── requirements.txt          # Python dependencies
│
├── config/                   # Training configurations
│   ├── quick_start.yaml      # Fast testing config
│   ├── production_training.yaml
│   ├── gpu_training_117m_balanced.yaml
│   └── gpu_training_117m_balanced_low_memory.yaml
│
├── src/                      # Source code
│   ├── model.py              # Model architecture
│   ├── trainer.py            # Training loop
│   ├── data.py               # Data loading
│   ├── inference.py          # Inference script
│   ├── dataset_downloader.py # Auto-download datasets
│   └── remote_model_loader.py # Load models from HF Hub
│
├── scripts/                  # Utility scripts
│   ├── prepare_balanced_dataset.py
│   ├── test_checkpoint_upload.py
│   ├── setup_dataset_repo.py
│   └── fix_environment.py
│
├── test/                     # Test suite
│   ├── test_setup.py
│   ├── test_installation.py
│   └── test_training.py
│
├── docs/                     # Documentation
│   ├── START_HERE.md         # Getting started guide
│   ├── DATASET_DOWNLOAD_GUIDE.md
│   ├── CHECKPOINT_UPLOAD_GUIDE.md
│   ├── GPU_TRAINING_OPTIMIZATION_GUIDE.md
│   └── MULTI_GPU_MEMORY_FIX.md
│
├── checkpoints/              # Saved model checkpoints
├── logs/                     # Training logs
└── data/                     # Dataset storage
```

## 🎓 Configuration Files

| Config | Model Size | Dataset | GPU Memory | Use Case |
|--------|------------|---------|------------|----------|
| `quick_start.yaml` | 2.36M | WikiText-2 | 2GB | Testing |
| `production_training.yaml` | 16M | WikiText-2 | 4GB | CPU training |
| `auto_training_117m_balanced.yaml` | 117M | 300M tokens | Auto | **Cross-hardware (Recommended)** |
| `gpu_training_117m_balanced.yaml` | 117M | 300M tokens | 14GB | Production GPU |
| `gpu_training_117m_balanced_low_memory.yaml` | 117M | 300M tokens | 8GB | Multi-GPU safe |
| `tpu_training_117m_balanced.yaml` | 117M | 300M tokens | TPU | Google Cloud TPU |

## 🔧 Advanced Features

### Automatic Dataset Download

Datasets are automatically downloaded from HuggingFace Hub when missing:

```yaml
# In config file
data:
  dataset_type: "binary"
  train_file: "data/balanced_300m/train.bin"
  
  huggingface_dataset:
    repo_id: "0x-genesys/mix_wiki_code_chat_data_300M_tokens"
    dataset_name: "balanced_300m"
    auto_download: true
```

### Automatic Checkpoint Upload

Upload checkpoints to HuggingFace Hub during training:

```yaml
# In config file
huggingface_hub:
  enabled: true
  repo_id: "0x-genesys/neo_weights_checkpoints"
  upload_best_only: false
```

### Memory Optimization

For multi-GPU training with limited memory:

```bash
# Set memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Use low memory config
python train.py --config config/gpu_training_117m_balanced_low_memory.yaml
```

## 🐛 Troubleshooting

### Out of Memory (OOM)
```bash
# Use low memory config
python train.py --config config/gpu_training_117m_balanced_low_memory.yaml

# Or use single GPU
export CUDA_VISIBLE_DEVICES=0
python train.py --config config/gpu_training_117m_balanced.yaml
```

### Slow Training
See [GPU_TRAINING_OPTIMIZATION_GUIDE.md](docs/GPU_TRAINING_OPTIMIZATION_GUIDE.md) for optimization strategies.

### Environment Issues
```bash
# Run diagnostics
python scripts/fix_environment.py

# Upgrade PyTorch
bash scripts/upgrade_pytorch.sh
```

## 📖 Documentation

- **[START_HERE.md](docs/START_HERE.md)** - Complete setup guide
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - System architecture
- **[INFERENCE_GUIDE.md](docs/INFERENCE_GUIDE.md)** - Complete inference guide
- **[REMOTE_MODEL_LOADING.md](docs/REMOTE_MODEL_LOADING.md)** - Load models from HuggingFace Hub
- **[DATASET_DOWNLOAD_GUIDE.md](docs/DATASET_DOWNLOAD_GUIDE.md)** - Dataset management
- **[CHECKPOINT_UPLOAD_GUIDE.md](docs/CHECKPOINT_UPLOAD_GUIDE.md)** - Model sharing
- **[GPU_TRAINING_OPTIMIZATION_GUIDE.md](docs/GPU_TRAINING_OPTIMIZATION_GUIDE.md)** - Performance tuning
- **[MULTI_GPU_MEMORY_FIX.md](docs/MULTI_GPU_MEMORY_FIX.md)** - Memory optimization

## 🧪 Testing

```bash
# Run all tests
python -m pytest test/

# Run specific test
python test/test_setup.py

# Check environment
python scripts/fix_environment.py
```

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [PyTorch](https://pytorch.org/)
- Tokenization with [tiktoken](https://github.com/openai/tiktoken)
- Model hosting on [HuggingFace Hub](https://huggingface.co/)

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/neo/issues)
- **Documentation**: [docs/](docs/)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/neo/discussions)

## 🔗 Links

- **Models**: [0x-genesys/neo_weights_checkpoints](https://huggingface.co/0x-genesys/neo_weights_checkpoints)
- **Datasets**: [0x-genesys/mix_wiki_code_chat_data_300M_tokens](https://huggingface.co/datasets/0x-genesys/mix_wiki_code_chat_data_300M_tokens)
- **Documentation**: [docs/](docs/)

---

**Made with ❤️ by the Neo team**