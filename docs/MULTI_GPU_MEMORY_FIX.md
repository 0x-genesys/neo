# Multi-GPU Memory Fix - OOM Solutions

## 🚨 Your Problem

```
CUDA out of memory. Tried to allocate 2.29 GiB.
GPU 0 has 14.56 GiB total, only 313.81 MiB free.
PyTorch allocated: 13.42 GiB
```

**Root cause**: DataParallel (multi-GPU) has significant memory overhead on GPU 0 because it:
1. Replicates the model on each GPU
2. Scatters input data to all GPUs
3. **Gathers all gradients back to GPU 0** ← This causes the spike!
4. Updates model on GPU 0
5. Broadcasts updated model to all GPUs

## 🎯 Immediate Solution

### Option 1: Use Low Memory Config (RECOMMENDED) ✅

```bash
# Set memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Use low memory config
python train.py --config config/gpu_training_117m_balanced_low_memory.yaml
```

**Changes**:
- `batch_size: 16 → 8` (50% less memory)
- `context_length: 512 → 384` (25% less memory)
- `gradient_checkpointing: false → true` (saves 2-3GB)
- `gradient_accumulation: 8 → 16` (maintains effective batch)

**Memory usage**: ~8-9GB per GPU (safe!)

### Option 2: Use Single GPU

```bash
# Use only GPU 0
export CUDA_VISIBLE_DEVICES=0

# Run with original config
python train.py --config config/gpu_training_117m_balanced.yaml
```

**Pros**: No DataParallel overhead, more memory available
**Cons**: 2x slower (only using 1 GPU)

### Option 3: Emergency Minimal Config

Edit your config:
```yaml
model:
  context_length: 256              # Reduce from 512
  use_gradient_checkpointing: true # Enable

training:
  batch_size: 4                    # Reduce from 16
  gradient_accumulation_steps: 32  # Increase from 8

data:
  max_length: 256                  # Match context
```

## 📊 Memory Breakdown - Why You're Running Out

### Single GPU (No DataParallel)
```
Model (FP16):           0.9 GB
Optimizer (FP32):       1.8 GB
Gradients (FP16):       0.9 GB
Activations (b=16):     4.0 GB
Buffer:                 1.0 GB
------------------------
Total:                  8.6 GB ✅ Fits in 14.56 GB
```

### Multi-GPU with DataParallel (Your Case)
```
GPU 0:
  Model (FP16):         0.9 GB
  Optimizer (FP32):     1.8 GB
  Gradients (FP16):     0.9 GB
  Activations (b=16):   4.0 GB
  Gathered gradients:   1.8 GB  ← EXTRA!
  Scatter/gather buf:   1.5 GB  ← EXTRA!
  Buffer:               1.5 GB
  ------------------------
  Total:               12.4 GB

GPU 1:
  Model (FP16):         0.9 GB
  Activations (b=16):   4.0 GB
  Gradients (FP16):     0.9 GB
  Buffer:               1.0 GB
  ------------------------
  Total:                6.8 GB ✅ Fits fine
```

**Problem**: GPU 0 needs ~12-14GB, but you only have 14.56GB total!

## 🔧 Memory Optimization Strategies

### Strategy 1: Enable Gradient Checkpointing ⭐ **BEST**

```yaml
model:
  use_gradient_checkpointing: true
```

**Saves**: 2-3GB per GPU
**Cost**: 20-30% slower training
**Verdict**: Worth it for multi-GPU

### Strategy 2: Reduce Batch Size

```yaml
training:
  batch_size: 8                    # Half of 16
  gradient_accumulation_steps: 16  # Double to maintain effective batch
```

**Saves**: ~2GB per GPU
**Cost**: Slower training (more gradient accumulation)
**Verdict**: Safe and effective

### Strategy 3: Reduce Context Length

```yaml
model:
  context_length: 384  # Or 256

data:
  max_length: 384      # Match context
```

**Saves**: 
- 512 → 384: ~1.5GB
- 512 → 256: ~3GB

**Cost**: Model sees less context
**Verdict**: Good if you don't need long context

### Strategy 4: Use DistributedDataParallel (DDP)

**Requires code changes**, but much more memory efficient than DataParallel.

DDP doesn't gather all gradients to GPU 0 - each GPU updates independently.

### Strategy 5: Set Memory Allocator Config

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

**Saves**: Reduces fragmentation, can free up 500MB-1GB
**Cost**: None
**Verdict**: Always do this!

## 🎯 Recommended Configurations

### For 2x T4 (14.56 GB each)

#### Conservative (Will definitely work)
```yaml
model:
  context_length: 256
  use_gradient_checkpointing: true

training:
  batch_size: 4
  gradient_accumulation_steps: 32
```
**Memory**: ~6-7GB per GPU
**Speed**: Slow but stable

#### Balanced (Recommended)
```yaml
model:
  context_length: 384
  use_gradient_checkpointing: true

training:
  batch_size: 8
  gradient_accumulation_steps: 16
```
**Memory**: ~8-9GB per GPU
**Speed**: Moderate

#### Aggressive (May work)
```yaml
model:
  context_length: 512
  use_gradient_checkpointing: true

training:
  batch_size: 12
  gradient_accumulation_steps: 11
```
**Memory**: ~11-12GB per GPU
**Speed**: Faster

## 🐛 Troubleshooting Steps

### Step 1: Check Current Memory Usage

```bash
# Before training
nvidia-smi

# During training (in another terminal)
watch -n 1 nvidia-smi
```

Look for:
- GPU 0 memory usage
- GPU 1 memory usage
- GPU 0 should be higher (it's the master)

### Step 2: Set Memory Optimization

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### Step 3: Use Low Memory Config

```bash
python train.py --config config/gpu_training_117m_balanced_low_memory.yaml
```

### Step 4: If Still OOM, Reduce Further

Edit config:
```yaml
training:
  batch_size: 4  # Reduce to 4
```

### Step 5: Last Resort - Single GPU

```bash
export CUDA_VISIBLE_DEVICES=0
python train.py --config config/gpu_training_117m_balanced.yaml
```

## 📊 Memory Usage Comparison

| Config | GPU 0 Memory | GPU 1 Memory | Speed | Risk |
|--------|--------------|--------------|-------|------|
| **Original (b=16, ctx=512)** | 13-14GB | 7-8GB | Fast | OOM! |
| **Low Memory (b=8, ctx=384)** | 8-9GB | 5-6GB | Moderate | Safe ✅ |
| **Conservative (b=4, ctx=256)** | 6-7GB | 4-5GB | Slow | Very Safe ✅ |
| **Single GPU (b=16, ctx=512)** | 8-9GB | N/A | Slow | Safe ✅ |

## 💡 Why DataParallel Uses So Much Memory

### The Problem
```python
# DataParallel workflow (simplified)
1. Replicate model on all GPUs          # Each GPU: +0.9GB
2. Scatter batch to all GPUs            # Each GPU: +4GB (activations)
3. Forward pass on each GPU             # Each GPU: compute
4. Backward pass on each GPU            # Each GPU: +0.9GB (gradients)
5. Gather gradients to GPU 0            # GPU 0: +1.8GB EXTRA!
6. Update model on GPU 0                # GPU 0: optimizer step
7. Broadcast updated model to all GPUs  # Each GPU: sync
```

**GPU 0 does extra work**: Gathering gradients + optimizer update

### The Solution
Use DistributedDataParallel (DDP) instead:
```python
# DDP workflow (more efficient)
1. Each GPU has its own model copy
2. Each GPU processes its own batch
3. Each GPU computes gradients
4. Gradients are all-reduced (averaged) across GPUs
5. Each GPU updates its own model
```

**No gathering to GPU 0!** All GPUs share the work equally.

## 🚀 Quick Commands

```bash
# Set memory optimization (always do this)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Use low memory config
python train.py --config config/gpu_training_117m_balanced_low_memory.yaml

# Monitor memory
watch -n 1 nvidia-smi

# If still OOM, use single GPU
export CUDA_VISIBLE_DEVICES=0
python train.py --config config/gpu_training_117m_balanced.yaml
```

## 🎯 Decision Tree

```
Start here:
├─ Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
│
├─ Try: config/gpu_training_117m_balanced_low_memory.yaml
│  ├─ Works? ✅ Great! Continue training
│  └─ OOM? ↓
│
├─ Reduce batch_size to 4 in config
│  ├─ Works? ✅ Continue training (slower)
│  └─ OOM? ↓
│
├─ Reduce context_length to 256 in config
│  ├─ Works? ✅ Continue training
│  └─ OOM? ↓
│
└─ Use single GPU: CUDA_VISIBLE_DEVICES=0
   └─ Will work ✅ (but 2x slower)
```

## 📞 Quick Help

**OOM on GPU 0?** → Use `config/gpu_training_117m_balanced_low_memory.yaml`
**Still OOM?** → Reduce `batch_size` to 4
**Need speed?** → Use single GPU with larger batch
**Need context?** → Keep 512, reduce batch_size

Start with the low memory config - it's designed specifically for your situation! ✅