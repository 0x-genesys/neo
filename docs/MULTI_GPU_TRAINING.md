# Multi-GPU Training Guide

## Your Setup: 2× Tesla T4 GPUs

You have **2× Tesla T4 GPUs** (16GB each = 32GB total). This guide shows you how to use both GPUs for faster training.

---

## Quick Start

### Use Both GPUs (Recommended)
```bash
python train.py --config config/gpu_training_117m.yaml --multi-gpu
```

### Use Specific GPUs
```bash
# Use GPU 0 and 1
python train.py --config config/gpu_training_117m.yaml --gpu-ids 0,1

# Use only GPU 1
python train.py --config config/gpu_training_117m.yaml --gpu-ids 1
```

### Use Single GPU (Default)
```bash
# Uses GPU 0 only
python train.py --config config/gpu_training_117m.yaml
```

---

## How Multi-GPU Works

### DataParallel (What We Use)

**How it works**:
1. Model is copied to each GPU
2. Batch is split across GPUs
3. Each GPU processes its portion
4. Gradients are gathered and averaged
5. Model is updated on primary GPU
6. Updated model is broadcast to all GPUs

**Example with 2 GPUs**:
```
Config batch_size: 16

Single GPU:
  - GPU 0 processes: 16 samples
  - Total: 16 samples per step

Multi-GPU (2 GPUs):
  - GPU 0 processes: 8 samples
  - GPU 1 processes: 8 samples
  - Total: 16 samples per step (same!)
  - Speed: ~1.8x faster (not 2x due to overhead)
```

---

## Benefits of Multi-GPU

### 1. Faster Training ⚡
```
Single GPU (1× T4):
  - Time per step: ~2 seconds
  - 100,000 steps: ~55 hours

Multi-GPU (2× T4):
  - Time per step: ~1.1 seconds (1.8x faster)
  - 100,000 steps: ~30 hours (save 25 hours!)
```

### 2. Larger Batch Sizes 📈
```
Single GPU (16GB):
  - Max batch_size: 16
  - Memory: ~15GB

Multi-GPU (2× 16GB = 32GB):
  - Max batch_size: 32 (or keep 16 and use less memory per GPU)
  - Memory: ~8GB per GPU
```

### 3. More Memory 💾
```
Single GPU:
  - 162M model: batch_size=16 → 15GB (tight!)

Multi-GPU:
  - 162M model: batch_size=16 → 8GB per GPU (comfortable!)
  - Can train larger models (345M, 774M)
```

---

## Usage Examples

### Example 1: Standard Training (2 GPUs)
```bash
python train.py --config config/gpu_training_117m.yaml --multi-gpu
```

**Output**:
```
✅ Multi-GPU training enabled!
   Using GPUs: [0, 1]
   Total GPUs: 2
   GPU 0: Tesla T4 (16.00GB)
   GPU 1: Tesla T4 (16.00GB)

Configuration:
  Batch size: 16
  Effective batch size: 16 (per-GPU: 8)
  
🔧 Wrapping model with DataParallel for 2 GPUs...
✅ Model parallelized across GPUs: [0, 1]
   Primary GPU: 0
   Batch will be split across 2 GPUs
```

### Example 2: Larger Batch Size (2 GPUs)
```bash
# Edit config to increase batch size
# batch_size: 32 (instead of 16)
# gradient_accumulation_steps: 4 (instead of 8)

python train.py --config config/gpu_training_117m.yaml --multi-gpu
```

**Benefits**:
- Effective batch: 32 × 4 = 128 (same as before)
- Per-GPU batch: 16 (same memory as single GPU with batch=16)
- Speed: 1.8x faster!

### Example 3: Train Larger Model (2 GPUs)
```bash
# 345M model (needs ~15GB on single GPU)
python train.py --config config/gpu_training_345m.yaml --multi-gpu
```

**Benefits**:
- 345M model needs ~15GB on single GPU (tight!)
- With 2 GPUs: ~8GB per GPU (comfortable!)

---

## Performance Comparison

### 162M Model (gpu_training_117m.yaml)

| Setup | Batch/GPU | Total Batch | Memory/GPU | Speed | Total Time |
|-------|-----------|-------------|------------|-------|------------|
| **1 GPU** | 16 | 16 | 15GB | 1.0x | 55h |
| **2 GPUs** | 8 | 16 | 8GB | 1.8x | 30h ⚡ |
| **2 GPUs (large batch)** | 16 | 32 | 15GB | 1.8x | 27h ⚡ |

### 345M Model (gpu_training_345m.yaml)

| Setup | Batch/GPU | Total Batch | Memory/GPU | Speed | Total Time |
|-------|-----------|-------------|------------|-------|------------|
| **1 GPU** | 8 | 8 | 15GB | 1.0x | 7d |
| **2 GPUs** | 8 | 16 | 8GB | 1.8x | 4d ⚡ |
| **2 GPUs (large batch)** | 16 | 32 | 15GB | 1.8x | 3.5d ⚡ |

---

## Configuration Recommendations

### For 2× T4 GPUs (16GB each)

#### Option 1: Same Batch Size (Safer)
```yaml
# config/gpu_training_117m.yaml
training:
  batch_size: 16  # Same as single GPU
  gradient_accumulation_steps: 8
  # Effective batch: 16 × 8 = 128
  # Per-GPU: 8 samples
  # Memory: ~8GB per GPU (safe!)
  # Speed: 1.8x faster
```

#### Option 2: Larger Batch Size (Faster)
```yaml
# config/gpu_training_117m.yaml
training:
  batch_size: 32  # 2x larger
  gradient_accumulation_steps: 4  # 2x smaller
  # Effective batch: 32 × 4 = 128 (same convergence)
  # Per-GPU: 16 samples
  # Memory: ~15GB per GPU (tight but works)
  # Speed: 2x faster (less gradient accumulation)
```

#### Option 3: Much Larger Batch (Experimental)
```yaml
# config/gpu_training_117m.yaml
training:
  batch_size: 64  # 4x larger
  gradient_accumulation_steps: 2  # 4x smaller
  # Effective batch: 64 × 2 = 128 (same convergence)
  # Per-GPU: 32 samples
  # Memory: May OOM! Test first
  # Speed: 2.5x faster (minimal gradient accumulation)
```

---

## Monitoring Multi-GPU Training

### Check GPU Usage
```bash
# In another terminal
watch -n 1 nvidia-smi
```

**Expected output**:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.85.12    Driver Version: 525.85.12    CUDA Version: 12.0   |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla T4            Off  | 00000000:00:04.0 Off |                    0 |
| N/A   65C    P0    45W /  70W |   8192MiB / 15360MiB |     95%      Default |
+-------------------------------+----------------------+----------------------+
|   1  Tesla T4            Off  | 00000000:00:05.0 Off |                    0 |
| N/A   64C    P0    44W /  70W |   8192MiB / 15360MiB |     94%      Default |
+-------------------------------+----------------------+----------------------+
```

**What to look for**:
- ✅ Both GPUs show high utilization (90-100%)
- ✅ Both GPUs show similar memory usage
- ✅ Both GPUs show similar temperature
- ⚠️ If GPU 1 shows 0% util → DataParallel not working

---

## Troubleshooting

### Issue 1: Only GPU 0 is Used

**Symptom**:
```
nvidia-smi shows:
  GPU 0: 95% utilization
  GPU 1: 0% utilization
```

**Solution**:
```bash
# Make sure you added --multi-gpu flag
python train.py --config config/gpu_training_117m.yaml --multi-gpu
```

### Issue 2: OOM on Multi-GPU

**Symptom**:
```
CUDA out of memory on GPU 0
```

**Solution**:
```bash
# Reduce batch size
python train.py --config config/gpu_training_117m.yaml --multi-gpu --batch-size 8
```

### Issue 3: Slower with Multi-GPU

**Symptom**:
- Multi-GPU is slower than single GPU

**Possible causes**:
1. Batch size too small (overhead dominates)
2. Model too small (communication overhead)
3. Slow interconnect between GPUs

**Solution**:
```bash
# Increase batch size to amortize overhead
python train.py --config config/gpu_training_117m.yaml --multi-gpu --batch-size 32
```

### Issue 4: Uneven GPU Usage

**Symptom**:
```
GPU 0: 95% utilization, 15GB memory
GPU 1: 60% utilization, 8GB memory
```

**Explanation**:
- This is normal with DataParallel
- GPU 0 is the "primary" GPU (does extra work)
- GPU 1 is a "worker" GPU

**Not a problem**: Training is still faster than single GPU

---

## Advanced: DistributedDataParallel (DDP)

DataParallel is simple but has limitations. For better performance, use DDP:

### Benefits of DDP
- ✅ Better GPU utilization (all GPUs equal)
- ✅ Faster (less communication overhead)
- ✅ Scales to multiple nodes
- ⚠️ More complex setup

### DDP Setup (Future Work)
```bash
# Not implemented yet
# Would require:
# 1. torch.distributed setup
# 2. Multiple processes (one per GPU)
# 3. Gradient synchronization
# 4. Checkpoint handling
```

**For now**: DataParallel is good enough for 2 GPUs!

---

## Recommendations for Your 2× T4 Setup

### For 162M Model (gpu_training_117m.yaml)

✅ **Recommended**:
```bash
python train.py --config config/gpu_training_117m.yaml --multi-gpu
```

**Benefits**:
- Speed: 1.8x faster (30h instead of 55h)
- Memory: 8GB per GPU (safe!)
- Quality: Same as single GPU

### For 345M Model (gpu_training_345m.yaml)

✅ **Recommended**:
```bash
python train.py --config config/gpu_training_345m.yaml --multi-gpu
```

**Benefits**:
- Single GPU: Would need 15GB (tight!)
- Multi-GPU: 8GB per GPU (comfortable!)
- Speed: 1.8x faster

### For 774M Model (gpu_training_774m.yaml)

⚠️ **Possible but tight**:
```bash
python train.py --config config/gpu_training_774m.yaml --multi-gpu --batch-size 1
```

**Reality**:
- 774M model needs ~23GB on single GPU
- With 2 GPUs: ~12GB per GPU (might work with batch=1)
- Better to use A100 40GB

---

## Summary

### Your Situation
- **Hardware**: 2× Tesla T4 (16GB each)
- **Current**: Training uses only 1 GPU
- **Solution**: Add `--multi-gpu` flag

### Quick Commands

**Use both GPUs** (recommended):
```bash
python train.py --config config/gpu_training_117m.yaml --multi-gpu
```

**Use specific GPU**:
```bash
python train.py --config config/gpu_training_117m.yaml --gpu-ids 0
```

**Monitor GPUs**:
```bash
watch -n 1 nvidia-smi
```

### Expected Results

**Before (1 GPU)**:
- Training time: 55 hours
- GPU 0: 95% utilization
- GPU 1: 0% utilization (wasted!)

**After (2 GPUs)**:
- Training time: 30 hours (save 25 hours!)
- GPU 0: 95% utilization
- GPU 1: 94% utilization (both working!)

---

**Use both GPUs and save 25 hours of training time!** 🚀
