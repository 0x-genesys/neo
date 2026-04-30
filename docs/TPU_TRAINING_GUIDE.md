# TPU Training Guide

## Overview

This guide explains how to train Neo transformer models on Google Cloud TPU (Tensor Processing Unit). TPUs provide excellent performance for large-scale training with large batch sizes.

## What is TPU?

**TPU (Tensor Processing Unit)** is Google's custom-developed application-specific integrated circuit (ASIC) designed specifically for machine learning workloads.

### TPU Advantages

1. **High Performance**: Optimized for matrix operations
2. **Large Memory**: 16GB+ HBM per core
3. **Cost Effective**: Better price/performance than GPUs for large models
4. **Scalability**: Easy to scale from 8 to 2048+ cores
5. **bfloat16**: Native support for bfloat16 precision

### TPU Versions

- **TPU v2**: 180 TFLOPS, 64GB HBM (8 cores)
- **TPU v3**: 420 TFLOPS, 128GB HBM (8 cores)
- **TPU v4**: 275 TFLOPS per chip, improved efficiency

## Prerequisites

### 1. Google Cloud Account

Create a Google Cloud account and enable billing:
- Visit: https://cloud.google.com/
- Create project
- Enable Cloud TPU API

### 2. Install gcloud CLI

```bash
# Install gcloud
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# Initialize
gcloud init

# Set project
gcloud config set project YOUR_PROJECT_ID
```

### 3. Install torch_xla

```bash
# Install PyTorch/XLA
pip install torch torch_xla

# Verify installation
python -c "import torch_xla; print(torch_xla.__version__)"
```

## Setup TPU VM

### Create TPU VM

```bash
# Create TPU v3-8 (8 cores)
gcloud compute tpus tpu-vm create neo-tpu \
  --zone=us-central1-a \
  --accelerator-type=v3-8 \
  --version=tpu-vm-pt-2.0

# For TPU v2-8
gcloud compute tpus tpu-vm create neo-tpu \
  --zone=us-central1-a \
  --accelerator-type=v2-8 \
  --version=tpu-vm-pt-2.0
```

### SSH into TPU VM

```bash
gcloud compute tpus tpu-vm ssh neo-tpu --zone=us-central1-a
```

### Install Dependencies

```bash
# Update pip
pip install --upgrade pip

# Install PyTorch/XLA (if not pre-installed)
pip install torch torch_xla

# Clone repository
git clone https://github.com/yourusername/neo.git
cd neo

# Install requirements
pip install -r requirements.txt
```

## Training on TPU

### Quick Start

```bash
# Train with TPU config
python train.py --config config/tpu_training_117m_balanced.yaml --tpu
```

### Configuration

The TPU config (`config/tpu_training_117m_balanced.yaml`) is optimized for TPU:

```yaml
model:
  d_model: 768
  num_layers: 12
  num_heads: 12
  context_length: 512

training:
  batch_size: 128        # Large batch for TPU efficiency
  gradient_accumulation_steps: 4
  learning_rate: 3.0e-4  # Higher LR for larger batch

system:
  device: "tpu"
  mixed_precision: true  # bfloat16 on TPU
```

### Command Line Options

```bash
# Basic TPU training
python train.py --config config/tpu_training_117m_balanced.yaml --tpu

# Specify TPU cores
python train.py --config config/tpu_training_117m_balanced.yaml --tpu --tpu-cores 8

# Override batch size
python train.py --config config/tpu_training_117m_balanced.yaml --tpu --batch-size 256
```

## TPU-Specific Optimizations

### 1. Batch Size

TPUs work best with large batch sizes:

```yaml
training:
  batch_size: 128  # Minimum recommended
  # Or larger: 256, 512
```

**Why?**
- TPUs have high memory bandwidth
- Large batches amortize XLA compilation overhead
- Better utilization of TPU cores

### 2. Mixed Precision (bfloat16)

TPUs natively support bfloat16:

```yaml
system:
  mixed_precision: true  # Uses bfloat16 on TPU
```

**bfloat16 vs float16:**
- bfloat16: Same range as float32, better for training
- float16: Smaller range, may need loss scaling
- TPU: Optimized for bfloat16

### 3. XLA Compilation

XLA (Accelerated Linear Algebra) compiles computation graphs:

```python
# Automatic with torch_xla
import torch_xla.core.xla_model as xm

# Mark step for gradient update
xm.mark_step()
```

**Benefits:**
- Fuses operations for efficiency
- Optimizes memory layout
- Reduces host-device communication

### 4. Data Loading

Use fewer workers for TPU:

```yaml
data:
  num_workers: 4  # TPU works well with fewer workers
  pin_memory: false  # Not needed for TPU
```

## Monitoring TPU Training

### TensorBoard

```bash
# On TPU VM
tensorboard --logdir logs/tpu_training_117m_balanced --host 0.0.0.0 --port 6006

# On local machine (port forward)
gcloud compute tpus tpu-vm ssh neo-tpu --zone=us-central1-a -- -L 6006:localhost:6006

# Open browser
http://localhost:6006
```

### Cloud Console

Monitor TPU utilization in Google Cloud Console:
1. Go to: https://console.cloud.google.com/compute/tpus
2. Select your TPU
3. View metrics: Utilization, Memory, MXU (Matrix Unit)

### Command Line Monitoring

```bash
# Check TPU status
gcloud compute tpus tpu-vm describe neo-tpu --zone=us-central1-a

# SSH and check processes
gcloud compute tpus tpu-vm ssh neo-tpu --zone=us-central1-a
ps aux | grep python
```

## Performance Tuning

### Batch Size Tuning

Start with recommended sizes and increase:

```yaml
# Conservative (safe)
batch_size: 128
gradient_accumulation_steps: 4
# Effective batch: 512

# Aggressive (faster)
batch_size: 256
gradient_accumulation_steps: 2
# Effective batch: 512

# Maximum (if memory allows)
batch_size: 512
gradient_accumulation_steps: 1
# Effective batch: 512
```

### Learning Rate Scaling

Scale learning rate with batch size:

```
LR = base_LR * sqrt(batch_size / base_batch_size)

Example:
base_LR = 2e-4 (batch_size=16)
new_batch_size = 128
new_LR = 2e-4 * sqrt(128/16) = 2e-4 * 2.83 ≈ 5.7e-4

Or use linear scaling:
new_LR = 2e-4 * (128/16) = 1.6e-3
```

### Gradient Accumulation

Reduce gradient accumulation for faster training:

```yaml
# Slower but safer
gradient_accumulation_steps: 8

# Faster
gradient_accumulation_steps: 4

# Fastest (if batch size is large enough)
gradient_accumulation_steps: 1
```

## Troubleshooting

### Issue: torch_xla Import Error

**Error**: `ModuleNotFoundError: No module named 'torch_xla'`

**Solution**:
```bash
# Install torch_xla
pip install torch_xla

# Check PyTorch version compatibility
python -c "import torch; print(torch.__version__)"
pip install torch_xla==2.0  # Match PyTorch version
```

### Issue: TPU Not Detected

**Error**: `TPU requested but not available`

**Solution**:
```bash
# Check TPU status
gcloud compute tpus tpu-vm describe neo-tpu --zone=us-central1-a

# Verify torch_xla can see TPU
python -c "import torch_xla.core.xla_model as xm; print(xm.xla_device())"
```

### Issue: Out of Memory

**Error**: `RuntimeError: XLA out of memory`

**Solution**:
```yaml
# Reduce batch size
training:
  batch_size: 64  # Reduce from 128

# Or enable gradient checkpointing
model:
  use_gradient_checkpointing: true

# Or reduce context length
model:
  context_length: 384  # Reduce from 512
```

### Issue: Slow Training

**Problem**: Training slower than expected

**Solution**:
1. **Increase batch size**: TPUs work best with large batches
   ```yaml
   batch_size: 256  # Increase from 128
   ```

2. **Reduce gradient accumulation**:
   ```yaml
   gradient_accumulation_steps: 2  # Reduce from 4
   ```

3. **Check TPU utilization**: Should be >80%
   - View in Cloud Console
   - If low, increase batch size

4. **Reduce data loading overhead**:
   ```yaml
   num_workers: 2  # Reduce from 4
   ```

### Issue: XLA Compilation Slow

**Problem**: First few steps are very slow

**Solution**:
- This is normal! XLA compiles computation graphs
- First step: 30-60 seconds (compilation)
- Subsequent steps: Fast (compiled graph cached)
- Compilation happens once per unique graph shape

## Cost Optimization

### TPU Pricing (as of 2024)

- **TPU v2-8**: ~$4.50/hour
- **TPU v3-8**: ~$8.00/hour
- **TPU v4-8**: ~$4.00/hour (Spot)

### Cost Saving Tips

1. **Use Preemptible TPUs**: 70% cheaper
   ```bash
   gcloud compute tpus tpu-vm create neo-tpu \
     --zone=us-central1-a \
     --accelerator-type=v3-8 \
     --version=tpu-vm-pt-2.0 \
     --preemptible
   ```

2. **Delete TPU when not in use**:
   ```bash
   gcloud compute tpus tpu-vm delete neo-tpu --zone=us-central1-a
   ```

3. **Use checkpointing**: Resume from checkpoints if preempted
   ```yaml
   checkpoint:
     save_interval: 200  # Save frequently
   ```

4. **Monitor usage**: Set budget alerts in Cloud Console

## Best Practices

### 1. Checkpointing

Save checkpoints frequently:

```yaml
training:
  save_interval: 200  # Every 200 steps
  eval_interval: 200  # Evaluate frequently
```

### 2. Logging

Use TensorBoard for monitoring:

```yaml
logging:
  log_dir: "logs/tpu_training"
  log_interval: 10  # Log every 10 steps
```

### 3. Data Pipeline

Optimize data loading:

```yaml
data:
  num_workers: 4
  pin_memory: false  # Not needed for TPU
  prefetch_factor: 2  # Prefetch batches
```

### 4. Batch Size

Start conservative, then increase:

```
Step 1: batch_size=128, test for OOM
Step 2: batch_size=256, if no OOM
Step 3: batch_size=512, if no OOM
```

### 5. Learning Rate

Scale with batch size:

```yaml
# batch_size=128
learning_rate: 3.0e-4

# batch_size=256
learning_rate: 4.2e-4  # sqrt(2) * 3e-4

# batch_size=512
learning_rate: 6.0e-4  # 2 * 3e-4
```

## Example Training Session

```bash
# 1. Create TPU VM
gcloud compute tpus tpu-vm create neo-tpu \
  --zone=us-central1-a \
  --accelerator-type=v3-8 \
  --version=tpu-vm-pt-2.0

# 2. SSH into TPU VM
gcloud compute tpus tpu-vm ssh neo-tpu --zone=us-central1-a

# 3. Setup environment
git clone https://github.com/yourusername/neo.git
cd neo
pip install -r requirements.txt

# 4. Start training
python train.py --config config/tpu_training_117m_balanced.yaml --tpu

# 5. Monitor (in another terminal)
gcloud compute tpus tpu-vm ssh neo-tpu --zone=us-central1-a -- -L 6006:localhost:6006
# Open http://localhost:6006 in browser

# 6. After training, delete TPU
gcloud compute tpus tpu-vm delete neo-tpu --zone=us-central1-a
```

## Comparison: GPU vs TPU

| Feature | GPU (V100) | TPU v3-8 |
|---------|------------|----------|
| Memory | 16GB | 128GB (8 cores) |
| Performance | ~125 TFLOPS | ~420 TFLOPS |
| Batch Size | 16-32 | 128-512 |
| Cost/hour | ~$2.50 | ~$8.00 |
| Best For | Small-medium models | Large models, large batches |

**When to use TPU:**
- Large models (>100M parameters)
- Large batch sizes (>128)
- Long training runs
- Cost-effective at scale

**When to use GPU:**
- Small models (<100M parameters)
- Small batch sizes (<64)
- Quick experiments
- More flexible for debugging

## Resources

### Documentation
- **PyTorch/XLA**: https://pytorch.org/xla/
- **Google Cloud TPU**: https://cloud.google.com/tpu/docs
- **torch_xla GitHub**: https://github.com/pytorch/xla

### Tutorials
- **PyTorch/XLA Tutorial**: https://pytorch.org/xla/release/2.0/index.html
- **Cloud TPU Quickstart**: https://cloud.google.com/tpu/docs/quickstart

### Community
- **PyTorch Forums**: https://discuss.pytorch.org/
- **Google Cloud Community**: https://www.googlecloudcommunity.com/

## See Also

- [README.md](../README.md) - Main documentation
- [ARCHITECTURE.md](../ARCHITECTURE.md) - System architecture
- [GPU_MEMORY_GUIDE.md](GPU_MEMORY_GUIDE.md) - GPU optimization
- [MULTI_GPU_MEMORY_FIX.md](MULTI_GPU_MEMORY_FIX.md) - Multi-GPU training

---

**Last Updated**: 2026-04-30
**Version**: 1.0.0
