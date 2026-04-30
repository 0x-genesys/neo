# Device Support Summary

## Overview

Neo transformer now supports training and inference on multiple hardware platforms with automatic device detection and optimization.

## Supported Devices

| Device | Status | Mixed Precision | Best For | Priority |
|--------|--------|-----------------|----------|----------|
| **TPU** | ✅ Supported | bfloat16 | Large models, large batches, cloud training | 1 |
| **CUDA** | ✅ Supported | FP16 | General purpose, flexible, widely available | 2 |
| **MPS** | ✅ Supported | Limited | Apple Silicon development, local training | 3 |
| **CPU** | ✅ Supported | No | Testing, small models, fallback | 4 |

## Device Selection

### Automatic Selection

The system automatically selects the best available device:

```bash
# Auto-detect and use best device
python train.py --config config/production_training.yaml
```

**Priority Order**: TPU → CUDA → MPS → CPU

### Manual Selection

Specify device in configuration:

```yaml
system:
  device: "auto"  # or "tpu", "cuda", "mps", "cpu"
```

Or via command line:

```bash
# Use TPU
python train.py --config config/tpu_training_117m_balanced.yaml --tpu

# Use specific GPUs
python train.py --config config/gpu_training_117m_balanced.yaml --multi-gpu --gpu-ids 0,1

# Use CPU
python train.py --config config/production_training.yaml  # Falls back to CPU if no GPU/TPU
```

## Device-Specific Features

### TPU (Tensor Processing Unit)

**Hardware**: Google Cloud TPU v2/v3/v4

**Features**:
- bfloat16 mixed precision
- XLA compiler optimization
- Large batch sizes (128-512)
- 128GB+ memory (8 cores)

**Configuration**: `config/tpu_training_117m_balanced.yaml`

**Setup**:
```bash
# Install torch_xla
pip install torch_xla

# Create TPU VM
gcloud compute tpus tpu-vm create neo-tpu \
  --zone=us-central1-a \
  --accelerator-type=v3-8 \
  --version=tpu-vm-pt-2.0

# Train
python train.py --config config/tpu_training_117m_balanced.yaml --tpu
```

**Best For**:
- Large models (>100M parameters)
- Large batch sizes (>128)
- Long training runs
- Cost-effective scaling

**Documentation**: `docs/TPU_TRAINING_GUIDE.md`

### CUDA (NVIDIA GPUs)

**Hardware**: NVIDIA GPUs (Tesla, RTX, A100, etc.)

**Features**:
- FP16 mixed precision
- Multi-GPU support (DataParallel)
- TF32 on Ampere GPUs
- cuDNN optimization

**Configuration**: `config/gpu_training_117m_balanced.yaml`

**Setup**:
```bash
# Single GPU
python train.py --config config/gpu_training_117m_balanced.yaml

# Multi-GPU
python train.py --config config/gpu_training_117m_balanced.yaml --multi-gpu

# Specific GPUs
python train.py --config config/gpu_training_117m_balanced.yaml --gpu-ids 0,1
```

**Best For**:
- General purpose training
- Flexible batch sizes
- Wide availability
- Development and production

**Documentation**: `docs/GPU_MEMORY_GUIDE.md`, `docs/MULTI_GPU_MEMORY_FIX.md`

### MPS (Apple Silicon)

**Hardware**: Apple M1/M2/M3 chips

**Features**:
- Metal Performance Shaders
- Unified memory architecture
- Good for development
- Limited mixed precision

**Configuration**: `config/production_training.yaml`

**Setup**:
```bash
# Auto-detect MPS
python train.py --config config/production_training.yaml
```

**Best For**:
- Local development on Mac
- Small to medium models
- Testing and prototyping
- Apple Silicon optimization

**Documentation**: `README.md` (MPS section)

### CPU

**Hardware**: Any x86/ARM CPU

**Features**:
- Universal compatibility
- No special setup
- Slower than GPU/TPU
- Good for testing

**Configuration**: Any config file

**Setup**:
```bash
# Auto-fallback to CPU
python train.py --config config/quick_start.yaml
```

**Best For**:
- Testing and debugging
- Small models
- No GPU/TPU available
- CI/CD pipelines

## Configuration Files

| Config | Device | Model Size | Batch Size | Memory | Use Case |
|--------|--------|------------|------------|--------|----------|
| `quick_start.yaml` | CPU/GPU | 2.36M | 8 | 2GB | Quick testing |
| `production_training.yaml` | CPU/GPU | 16M | 16 | 4GB | CPU training |
| `gpu_training_117m_balanced.yaml` | CUDA | 117M | 16 | 14GB | Single GPU |
| `gpu_training_117m_balanced_low_memory.yaml` | CUDA | 117M | 8 | 8GB | Multi-GPU |
| `tpu_training_117m_balanced.yaml` | TPU | 117M | 128 | TPU | Cloud TPU |

## Performance Comparison

### Training Speed (117M model, 512 context)

| Device | Batch Size | Steps/sec | Relative Speed |
|--------|------------|-----------|----------------|
| TPU v3-8 | 128 | ~2.5 | 10x |
| A100 (40GB) | 32 | ~1.8 | 7x |
| V100 (16GB) | 16 | ~1.2 | 5x |
| RTX 3090 | 16 | ~1.0 | 4x |
| M1 Max (MPS) | 8 | ~0.4 | 1.5x |
| CPU (16 cores) | 4 | ~0.25 | 1x |

*Note: Speeds are approximate and depend on model configuration*

### Memory Usage (117M model, 512 context, FP16)

| Device | Batch Size | Memory Used | Notes |
|--------|------------|-------------|-------|
| TPU v3-8 | 128 | ~40GB | 128GB available |
| A100 (40GB) | 32 | ~35GB | With gradient checkpointing |
| V100 (16GB) | 16 | ~14GB | Near capacity |
| RTX 3090 (24GB) | 16 | ~14GB | Comfortable |
| M1 Max (64GB) | 8 | ~8GB | Unified memory |
| CPU | 4 | ~6GB | System RAM |

## Device Detection

### Automatic Detection

```python
from src.device_utils import get_device_info, select_device

# Get device information
info = get_device_info()
print(f"CUDA available: {info['cuda_available']}")
print(f"MPS available: {info['mps_available']}")
print(f"TPU available: {info['tpu_available']}")

# Select best device
device = select_device('auto', verbose=True)
```

### Manual Detection

```bash
# Check CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Check MPS
python -c "import torch; print(f'MPS: {torch.backends.mps.is_available()}')"

# Check TPU
python -c "import torch_xla.core.xla_model as xm; print(f'TPU: {xm.xla_device()}')"
```

## Optimization Strategies

### TPU Optimization

```yaml
training:
  batch_size: 128        # Large batch
  gradient_accumulation_steps: 4
  learning_rate: 3.0e-4  # Higher LR

system:
  mixed_precision: true  # bfloat16
```

### GPU Optimization

```yaml
training:
  batch_size: 16         # Moderate batch
  gradient_accumulation_steps: 8
  learning_rate: 2.0e-4

model:
  use_gradient_checkpointing: true  # Save memory

system:
  mixed_precision: true  # FP16
```

### MPS Optimization

```yaml
training:
  batch_size: 8          # Smaller batch
  gradient_accumulation_steps: 16

system:
  mixed_precision: false  # Disable for stability
```

### CPU Optimization

```yaml
training:
  batch_size: 4          # Small batch
  gradient_accumulation_steps: 32

model:
  d_model: 256           # Smaller model
  num_layers: 4
```

## Troubleshooting

### Device Not Detected

**TPU**:
```bash
# Check torch_xla
pip install torch_xla
python -c "import torch_xla; print(torch_xla.__version__)"
```

**CUDA**:
```bash
# Check CUDA
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

**MPS**:
```bash
# Check MPS
python -c "import torch; print(torch.backends.mps.is_available())"
```

### Out of Memory

**Solution 1**: Reduce batch size
```yaml
training:
  batch_size: 8  # Reduce from 16
```

**Solution 2**: Enable gradient checkpointing
```yaml
model:
  use_gradient_checkpointing: true
```

**Solution 3**: Reduce context length
```yaml
model:
  context_length: 384  # Reduce from 512
```

### Slow Training

**TPU**: Increase batch size (128-512)
**GPU**: Disable gradient checkpointing if memory allows
**MPS**: Reduce num_workers (2-4)
**CPU**: Use smaller model or cloud GPU/TPU

## Migration Guide

### From CPU to GPU

```yaml
# Before (CPU)
system:
  device: "cpu"
training:
  batch_size: 4

# After (GPU)
system:
  device: "cuda"
  mixed_precision: true
training:
  batch_size: 16
```

### From GPU to TPU

```yaml
# Before (GPU)
system:
  device: "cuda"
  mixed_precision: true
training:
  batch_size: 16
  learning_rate: 2.0e-4

# After (TPU)
system:
  device: "tpu"
  mixed_precision: true  # bfloat16
training:
  batch_size: 128
  learning_rate: 3.0e-4  # Scale with batch size
```

### From Single GPU to Multi-GPU

```bash
# Before (single GPU)
python train.py --config config/gpu_training_117m_balanced.yaml

# After (multi-GPU)
python train.py --config config/gpu_training_117m_balanced_low_memory.yaml --multi-gpu
```

## Best Practices

### 1. Start with Auto-Detection

Let the system choose the best device:

```yaml
system:
  device: "auto"
```

### 2. Use Appropriate Batch Sizes

- **TPU**: 128-512
- **GPU**: 16-32
- **MPS**: 8-16
- **CPU**: 4-8

### 3. Enable Mixed Precision

- **TPU**: Always (bfloat16)
- **GPU**: Yes (FP16)
- **MPS**: No (stability issues)
- **CPU**: No (no benefit)

### 4. Monitor Resource Usage

- **TPU**: Cloud Console
- **GPU**: `nvidia-smi`
- **MPS**: Activity Monitor
- **CPU**: `htop` or Task Manager

### 5. Use Checkpointing

Save frequently for all devices:

```yaml
training:
  save_interval: 500
  eval_interval: 500
```

## Documentation

- **TPU**: `docs/TPU_TRAINING_GUIDE.md`
- **GPU**: `docs/GPU_MEMORY_GUIDE.md`
- **Multi-GPU**: `docs/MULTI_GPU_MEMORY_FIX.md`
- **General**: `README.md`, `ARCHITECTURE.md`

## Summary

✅ **4 Devices Supported**: TPU, CUDA, MPS, CPU
✅ **Automatic Detection**: Best device selected automatically
✅ **Device-Specific Optimization**: Tailored for each platform
✅ **Comprehensive Documentation**: Guides for all devices
✅ **Production Ready**: Tested and optimized

**Train on any device with Neo!** 🚀

---

**Last Updated**: 2026-04-30
**Version**: 1.0.0
