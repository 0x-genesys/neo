# GPU Memory Guide - Understanding Your GPU

## Your Situation

You mentioned having a **1.5GB GPU**, but the error message shows:
```
GPU 0 has a total capacity of 14.56 GiB
```

This means you actually have a **~15GB GPU** (14.56GB to be exact).

---

## Understanding the Error

### What Happened
```
CUDA out of memory. Tried to allocate 3.05 GiB.
GPU 0 has a total capacity of 14.56 GiB of which 1.80 GiB is free.
```

**Translation**:
- Total GPU memory: 14.56GB
- Free memory: 1.80GB (only 1.8GB available)
- Tried to allocate: 3.05GB
- Result: OOM (Out Of Memory)

**Why?**
- Your GPU has 14.56GB total
- PyTorch already used 12.76GB
- Only 1.8GB left
- Tried to allocate 3.05GB more
- Not enough space → OOM

---

## GPU Memory Sizes Explained

### Common GPU Memory Sizes

| GPU Model | Memory | Use Case |
|-----------|--------|----------|
| **Mobile/Integrated** | 1-2GB | Light tasks, not for training |
| GTX 1050 Ti | 4GB | Small models only |
| GTX 1060 | 6GB | Small models |
| RTX 2060 | 6GB | Small models |
| RTX 3060 | 12GB | Medium models |
| **RTX 3090** | **24GB** | Large models |
| **Tesla T4** | **16GB** | Cloud/server |
| **V100** | **16-32GB** | Cloud/server |
| **A100** | **40-80GB** | Large-scale training |

### Your GPU: ~15GB (14.56GB)

This is likely:
- **Tesla T4** (16GB) - common in cloud (Kaggle, Colab, AWS)
- **RTX 4070** (12GB) - but shows as ~15GB with overhead
- **Cloud GPU** with 16GB allocation

**Good news**: 15GB is plenty for 162M parameter model with optimization!

---

## What "1.5GB Free" Means

When you see "1.80 GiB is free" in the error:
- This is **available memory** (not total)
- Total: 14.56GB
- Used: 12.76GB (by PyTorch)
- Free: 1.80GB

**Analogy**:
- Your phone has 128GB storage (total)
- You've used 100GB (apps, photos, etc.)
- You have 28GB free
- You try to download 50GB movie → "Not enough space"

---

## Memory Requirements by Model Size

### 16M Parameters (Small Model)
```
Model (FP16):     0.03GB
Optimizer:        0.13GB
Activations:      0.05GB
Total:            ~0.2GB ✅ Fits any GPU
```

### 162M Parameters (Medium Model)
```
Original config (batch=16):
  Model (FP16):     0.32GB
  Optimizer:        1.30GB
  Gradients:        0.32GB
  Activations:      2.50GB
  Overhead:         3.00GB
  Total:           ~7.44GB
  Peak usage:      ~15.5GB ❌ OOM on 15GB!

Optimized config (batch=4):
  Model (FP16):     0.32GB
  Optimizer:        1.30GB
  Gradients:        0.32GB
  Activations:      0.60GB (with checkpointing)
  Overhead:         1.50GB
  Total:           ~4.04GB
  Peak usage:      ~6.5GB ✅ Fits 15GB easily!
```

### Why Optimizer is the Bottleneck

Adam optimizer stores:
- Momentum (4 bytes per param)
- Variance (4 bytes per param)
- Total: 8 bytes per param

For 162M params:
```
162M × 8 bytes = 1.296GB (just optimizer!)
```

This is why you can't fit 162M model in 1.5GB GPU - the optimizer alone needs 1.3GB!

---

## Solutions for Your 15GB GPU

### ✅ Solution 1: Use Optimized Config (RECOMMENDED)
```bash
python train.py --config config/gpu_training_117m_15gb.yaml
```

**Changes**:
- batch_size: 16 → 4 (4x less memory)
- gradient_accumulation: 8 → 32 (maintain quality)
- gradient_checkpointing: enabled
- Memory: 3-4GB (safe!)

### ✅ Solution 2: Use Standard Config with Smaller Batch
Edit `config/gpu_training_117m.yaml`:
```yaml
batch_size: 8  # Instead of 16
gradient_accumulation_steps: 16  # Instead of 8
```

### ✅ Solution 3: Use Smaller Model
```bash
python train.py --config config/gpu_training.yaml
```
- 16M params
- Memory: 0.2GB
- Time: 3 hours
- Quality: Good

---

## If You Actually Have 1.5GB GPU

If you really have a 1.5GB GPU (not 15GB), here's what you can do:

### Reality Check
```
162M model minimum:
  Optimizer alone: 1.30GB
  Model weights:   0.32GB
  Minimum total:   1.62GB
  
Conclusion: CANNOT fit in 1.5GB!
```

### Options for 1.5GB GPU

#### Option 1: Use 16M Model ✅ RECOMMENDED
```bash
python train.py --config config/gpu_training.yaml
```
- Memory: 0.2GB (fits easily!)
- Time: 3 hours
- Quality: Good

#### Option 2: Use CPU Training
```bash
python train.py --config config/production_training.yaml
```
- Memory: Uses RAM instead of GPU
- Time: 10 hours (slower but works)
- Quality: Same as GPU

#### Option 3: Use Smaller Custom Model
Create custom config:
```yaml
model:
  d_model: 384      # Instead of 768
  num_layers: 6     # Instead of 12
  # Result: ~40M params, needs ~0.5GB
```

---

## How to Check Your GPU Memory

### Method 1: nvidia-smi
```bash
nvidia-smi

# Output shows:
# | NVIDIA-SMI 525.85.12    Driver Version: 525.85.12    CUDA Version: 12.0     |
# |-------------------------------+----------------------+----------------------+
# | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
# | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
# |===============================+======================+======================|
# |   0  Tesla T4            Off  | 00000000:00:04.0 Off |                    0 |
# | N/A   45C    P0    28W /  70W |      0MiB / 15360MiB |      0%      Default |
# +-------------------------------+----------------------+----------------------+
#                                           ^^^^^^^^^^^^
#                                           Total: 15360MiB = 15GB
```

### Method 2: PyTorch
```python
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Total memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}GB")
```

### Method 3: During Training
The memory checker in `src/trainer.py` will show:
```
🔍 GPU Memory Check:
   Total memory: 14.56GB
   Estimated usage: 3.20GB
   ✅ Memory estimate looks good (22% of GPU)
```

---

## Memory Optimization Techniques

### 1. Reduce Batch Size
```yaml
batch_size: 4  # Instead of 16
gradient_accumulation_steps: 32  # Maintain effective batch
```
**Saves**: 4x memory on activations

### 2. Enable Gradient Checkpointing
```yaml
use_gradient_checkpointing: true
```
**Saves**: 2-3GB on activations
**Cost**: 20% slower training

### 3. Reduce Context Length
```yaml
context_length: 256  # Instead of 512
```
**Saves**: 2x memory on activations
**Cost**: Shorter context understanding

### 4. Use Mixed Precision
```yaml
mixed_precision: true
```
**Saves**: 50% memory on model weights
**Cost**: None (actually faster!)

### 5. Reduce Workers
```yaml
num_workers: 0  # Instead of 4
```
**Saves**: ~0.5GB on data loading
**Cost**: Slightly slower data loading

---

## Quick Reference

### I have 1.5GB GPU
→ Use `config/gpu_training.yaml` (16M model)

### I have 4-8GB GPU
→ Use `config/gpu_training.yaml` (16M model)

### I have 12-15GB GPU
→ Use `config/gpu_training_117m_15gb.yaml` (162M model, optimized)

### I have 16GB+ GPU
→ Use `config/gpu_training_117m.yaml` (162M model, standard)

### I have 40GB+ GPU
→ Use `config/gpu_training_345m.yaml` or larger

---

## Summary

### Your Situation
- **GPU**: 14.56GB total (likely Tesla T4 or similar)
- **Problem**: Original config used 15.5GB peak → OOM
- **Solution**: Use `gpu_training_117m_15gb.yaml` → 6.5GB peak ✅

### Key Takeaway
**You have a 15GB GPU, not 1.5GB!**

The "1.80 GiB is free" in the error message is **available memory**, not total memory.

### Recommended Action
```bash
# For your 15GB GPU
python train.py --config config/gpu_training_117m_15gb.yaml

# Expected:
# - Memory usage: 3-4GB (safe!)
# - Training time: 55 hours
# - Quality: Excellent
```

---

## Still Confused?

Run this to check your GPU:
```bash
nvidia-smi
```

Look for the line that says:
```
|   0  Tesla T4            Off  | ... |      0MiB / 15360MiB | ...
                                                    ^^^^^^^^
                                                    This is your total GPU memory
```

If it says:
- `15360MiB` → You have 15GB ✅ Use `gpu_training_117m_15gb.yaml`
- `1536MiB` → You have 1.5GB ⚠️ Use `gpu_training.yaml` (16M model)

---

**Bottom line**: Your error message shows 14.56GB total, so you have a ~15GB GPU. Use the optimized config and you'll be fine! 🚀
