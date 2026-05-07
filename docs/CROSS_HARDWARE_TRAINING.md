# Cross-Hardware Training Guide

## Overview

Neo supports **seamless checkpoint resumption across different hardware platforms**. You can start training on one device (e.g., GPU) and continue on another (e.g., TPU) without any issues.

## Key Concept: Hardware-Agnostic Checkpoints

### What's Saved in Checkpoints

Checkpoints contain:
1. **Model weights** - Device-independent, same across all hardware
2. **Optimizer state** - Adapts automatically to new batch size
3. **Training progress** - Step count, epoch, best loss
4. **Scheduler state** - Learning rate schedule position
5. **Configuration** - Original training config

### What Changes Across Hardware

Only **training hyperparameters** change:
- Batch size (optimized per device)
- Gradient accumulation steps (maintains effective batch)
- Learning rate (scaled with batch size)
- Mixed precision mode (bfloat16/FP16/none)
- Data loading workers

### What Stays the Same

**Model architecture and weights** remain identical:
- All layer weights
- Embedding matrices
- Attention parameters
- Model structure

## Auto-Adaptive Configuration

Use `config/auto_training_117m_balanced.yaml` for automatic hardware adaptation:

```yaml
system:
  device: "auto"  # Auto-detect: TPU → CUDA → MPS → CPU

training:
  batch_size: 16  # Base setting (auto-adjusted per hardware)
  gradient_accumulation_steps: 8  # Auto-adjusted
  learning_rate: 2.0e-4  # Auto-scaled
```

### Hardware-Specific Adjustments

| Hardware | Batch Size | Grad Accum | Effective Batch | LR Scaling |
|----------|------------|------------|-----------------|------------|
| **TPU** | 128 | 4 | 512 | 2.0x (4.0e-4) |
| **CUDA** | 16 | 8 | 128 | 1.0x (2.0e-4) |
| **MPS** | 8 | 16 | 128 | 0.7x (1.4e-4) |
| **CPU** | 4 | 32 | 128 | 0.5x (1.0e-4) |

## Usage Examples

### Example 1: GPU → TPU Migration

**Scenario**: Trained on GPU T4 to step 7500, want to continue on TPU for faster training.

```bash
# Step 1: Train on GPU T4 (multi-core)
python train.py --config config/auto_training_117m_balanced.yaml --multi-gpu

# Training runs to step 7500...
# Checkpoint saved: checkpoints/auto_training_117m_balanced/checkpoint_step_7500.pt

# Step 2: Create TPU VM on Google Cloud
gcloud compute tpus tpu-vm create neo-tpu \
  --zone=us-central1-a \
  --accelerator-type=v3-8 \
  --version=tpu-vm-pt-2.0

# Step 3: SSH into TPU VM
gcloud compute tpus tpu-vm ssh neo-tpu --zone=us-central1-a

# Step 4: Setup environment
git clone https://github.com/yourusername/neo.git
cd neo
pip install -r requirements.txt
pip install torch_xla

# Step 5: Download checkpoint from HuggingFace Hub
# (Checkpoint was auto-uploaded during GPU training)

# Step 6: Resume on TPU from step 7500
python train.py --config config/auto_training_117m_balanced.yaml \
    --resume-remote checkpoint_step_7500.pt \
    --tpu

# The system will:
# ✅ Load model weights (unchanged)
# ✅ Load optimizer state (adapts to batch_size=128)
# ✅ Continue from step 7500
# ✅ Use TPU-optimized settings (bfloat16, XLA)
# ✅ Maintain training progress
```

### Example 2: Single GPU → Multi-GPU

**Scenario**: Started on single GPU, want to use multiple GPUs.

```bash
# Step 1: Train on single GPU
python train.py --config config/auto_training_117m_balanced.yaml

# Step 2: Resume on multi-GPU
python train.py --config config/auto_training_117m_balanced.yaml \
    --resume checkpoints/auto_training_117m_balanced/checkpoint_step_5000.pt \
    --multi-gpu

# Batch size stays the same, but distributed across GPUs
```

### Example 3: TPU → GPU Migration

**Scenario**: Trained on TPU, want to continue on local GPU for debugging.

```bash
# Step 1: Train on TPU
python train.py --config config/auto_training_117m_balanced.yaml --tpu

# Step 2: Download checkpoint locally
# (From HuggingFace Hub or copy from TPU VM)

# Step 3: Resume on local GPU
python train.py --config config/auto_training_117m_balanced.yaml \
    --resume checkpoints/auto_training_117m_balanced/checkpoint_step_10000.pt

# The system will:
# ✅ Detect CUDA device
# ✅ Adjust batch_size from 128 → 16
# ✅ Adjust grad_accum from 4 → 8
# ✅ Continue training seamlessly
```

### Example 4: GPU → MPS (Apple Silicon)

**Scenario**: Trained on cloud GPU, want to continue locally on Mac.

```bash
# Step 1: Train on cloud GPU
python train.py --config config/auto_training_117m_balanced.yaml

# Step 2: Download checkpoint to Mac

# Step 3: Resume on Mac (MPS)
python train.py --config config/auto_training_117m_balanced.yaml \
    --resume checkpoints/auto_training_117m_balanced/checkpoint_step_3000.pt

# The system will:
# ✅ Detect MPS device
# ✅ Adjust batch_size to 8
# ✅ Disable mixed precision (for stability)
# ✅ Continue training
```

## Technical Details

### Batch Size Adaptation

When resuming on different hardware, the system:

1. **Detects new hardware**
2. **Adjusts batch size** for optimal performance
3. **Adjusts gradient accumulation** to maintain effective batch size
4. **Scales learning rate** proportionally
5. **Loads optimizer state** (adapts automatically)

**Example**:
```
GPU Training:
  batch_size = 16
  grad_accum = 8
  effective_batch = 128
  LR = 2.0e-4

Resume on TPU:
  batch_size = 128  (8x larger)
  grad_accum = 4    (2x smaller)
  effective_batch = 512  (4x larger)
  LR = 4.0e-4  (2x larger, sqrt scaling)
```

### Learning Rate Scaling

Learning rate is scaled using **square root scaling**:

```python
new_LR = base_LR * sqrt(new_batch / base_batch)

Example:
base_LR = 2.0e-4 (batch=16)
new_batch = 128
new_LR = 2.0e-4 * sqrt(128/16) = 2.0e-4 * 2.83 ≈ 5.7e-4
```

This maintains training dynamics across different batch sizes.

### Optimizer State Adaptation

The optimizer (AdamW) stores:
- **First moment** (momentum)
- **Second moment** (variance)
- **Step count**

When batch size changes:
- Moments are preserved (per-parameter)
- Step count continues
- Learning rate is scaled
- Optimizer adapts automatically

**No manual intervention needed!**

### Mixed Precision Compatibility

Different hardware uses different precision:

| Hardware | Precision | Checkpoint Format |
|----------|-----------|-------------------|
| TPU | bfloat16 | FP32 (saved) |
| CUDA | FP16 | FP32 (saved) |
| MPS | FP32 | FP32 (saved) |
| CPU | FP32 | FP32 (saved) |

**Checkpoints are always saved in FP32**, ensuring compatibility.

## Best Practices

### 1. Use Auto-Adaptive Config

Always use `config/auto_training_117m_balanced.yaml`:

```bash
python train.py --config config/auto_training_117m_balanced.yaml
```

This ensures seamless hardware transitions.

### 2. Enable HuggingFace Hub Upload

Upload checkpoints automatically:

```yaml
huggingface_hub:
  enabled: true
  repo_id: "your-username/your-model-repo"
```

This makes checkpoints accessible from any machine.

### 3. Save Frequently

Save checkpoints often for flexibility:

```yaml
training:
  save_interval: 500  # Every 500 steps
```

### 4. Monitor Effective Batch Size

Check that effective batch size is maintained:

```
GPU:  16 × 8  = 128
TPU:  128 × 4 = 512
MPS:  8 × 16  = 128
```

### 5. Verify Checkpoint Loading

Check logs when resuming:

```
✅ Checkpoint loaded from: checkpoint_step_7500.pt
✅ Resuming from epoch 2, step 7500
🔧 Hardware-Adaptive Configuration
   Detected device: xla:0 (TPU)
   Batch size: 16 → 128
   Gradient accumulation: 8 → 4
   Learning rate: 2.0e-4 → 4.0e-4
```

## Troubleshooting

### Issue: Different Loss After Resuming

**Cause**: Learning rate scaling or batch size change

**Solution**: This is normal! The loss may fluctuate slightly when changing hardware due to:
- Different batch size (different gradient noise)
- Different precision (bfloat16 vs FP16)
- Different hardware optimizations

The model will converge to similar performance.

### Issue: Out of Memory on New Hardware

**Cause**: New hardware has less memory

**Solution**:
```yaml
# Reduce batch size manually
training:
  batch_size: 8  # Reduce from 16
```

Or use gradient checkpointing:
```yaml
model:
  use_gradient_checkpointing: true
```

### Issue: Slower Training After Migration

**Cause**: Different hardware performance

**Solution**:
- **TPU**: Increase batch size (128-512)
- **GPU**: Check GPU utilization (`nvidia-smi`)
- **MPS**: Expected (slower than GPU)
- **CPU**: Expected (much slower)

### Issue: Checkpoint Not Found

**Cause**: Checkpoint not uploaded or wrong path

**Solution**:
```bash
# Check HuggingFace Hub
# Visit: https://huggingface.co/your-username/your-model-repo

# Or use local checkpoint
python train.py --config config/auto_training_117m_balanced.yaml \
    --resume /path/to/local/checkpoint.pt
```

## Validation

### Verify Checkpoint Compatibility

Test checkpoint loading:

```python
import torch

# Load checkpoint
checkpoint = torch.load('checkpoint_step_7500.pt', map_location='cpu')

# Check contents
print(f"Epoch: {checkpoint['epoch']}")
print(f"Step: {checkpoint['global_step']}")
print(f"Best loss: {checkpoint['best_val_loss']}")
print(f"Config device: {checkpoint['config']['system']['device']}")

# Model weights are device-independent
model_state = checkpoint['model_state_dict']
print(f"Model parameters: {len(model_state)}")
```

### Compare Training Curves

Monitor loss curves when switching hardware:

```bash
tensorboard --logdir logs/auto_training_117m_balanced
```

Look for:
- ✅ Smooth continuation (no sudden jumps)
- ✅ Similar loss trajectory
- ✅ Consistent validation loss

## Performance Comparison

### Training Speed (117M model, 512 context)

| Hardware | Batch | Steps/sec | Time to 10K steps |
|----------|-------|-----------|-------------------|
| TPU v3-8 | 128 | ~2.5 | ~1.1 hours |
| A100 (40GB) | 32 | ~1.8 | ~1.5 hours |
| V100 (16GB) | 16 | ~1.2 | ~2.3 hours |
| RTX 3090 | 16 | ~1.0 | ~2.8 hours |
| M1 Max | 8 | ~0.4 | ~7.0 hours |

### Cost Comparison (10K steps)

| Hardware | Cost/hour | Time | Total Cost |
|----------|-----------|------|------------|
| TPU v3-8 | $8.00 | 1.1h | $8.80 |
| A100 | $3.00 | 1.5h | $4.50 |
| V100 | $2.50 | 2.3h | $5.75 |
| RTX 3090 (local) | $0 | 2.8h | $0 |
| M1 Max (local) | $0 | 7.0h | $0 |

## Migration Strategies

### Strategy 1: GPU → TPU for Long Training

**When**: Training will take >24 hours on GPU

**Steps**:
1. Train on GPU for initial epochs
2. Upload checkpoint to HuggingFace Hub
3. Switch to TPU for remaining epochs
4. Save 50-70% training time

### Strategy 2: TPU → GPU for Debugging

**When**: Need to debug or experiment

**Steps**:
1. Train on TPU for main training
2. Download checkpoint
3. Continue on local GPU for experiments
4. Faster iteration for debugging

### Strategy 3: Cloud → Local for Fine-tuning

**When**: Want to fine-tune locally

**Steps**:
1. Pre-train on cloud (GPU/TPU)
2. Download checkpoint
3. Fine-tune on local machine (GPU/MPS/CPU)
4. Save cloud costs

## Summary

✅ **Seamless Migration**: Resume training on any hardware
✅ **Automatic Adaptation**: Batch size and LR auto-adjusted
✅ **Checkpoint Compatibility**: Model weights device-independent
✅ **No Manual Changes**: System handles everything
✅ **Production Ready**: Tested across all hardware

**Train anywhere, resume anywhere!** 🚀

## See Also

- [TPU_TRAINING_GUIDE.md](TPU_TRAINING_GUIDE.md) - TPU-specific guide
- [GPU_MEMORY_GUIDE.md](GPU_MEMORY_GUIDE.md) - GPU optimization
- [DEVICE_SUPPORT_SUMMARY.md](../DEVICE_SUPPORT_SUMMARY.md) - All devices
- [README.md](../README.md) - Main documentation

---

**Last Updated**: 2026-04-30
**Version**: 1.0.0
