# GPU Training Optimization Guide - 2x T4 GPUs

## 🎯 Current Issues

1. **Loss not updating side-by-side** - Multi-GPU logging issue
2. **Training is slow** - Suboptimal configuration for T4 GPUs

## 📊 Current Configuration Analysis

**Your Setup:**
- **GPUs**: 2x T4 (16GB each, 32GB total)
- **Model**: 117M parameters (~450MB)
- **Batch size**: 16 per GPU = 32 total
- **Gradient accumulation**: 8 steps
- **Effective batch**: 256 (32 × 8)
- **Context length**: 512 tokens
- **Mixed precision**: Enabled (FP16)

**Estimated Speed**: ~2-3 seconds per step (slow for T4s)

## 🚀 Optimization Strategies (No Code Changes)

### Strategy 1: Aggressive Batch Size Increase ⚡ **FASTEST**

**Change in config:**
```yaml
training:
  batch_size: 64        # Increase from 16 → 64 (4x)
  gradient_accumulation_steps: 2  # Decrease from 8 → 2
  # Effective batch stays same: 256 (64×2×2 GPUs)
```

**Speed Improvement**: 3-4x faster
**Pros**:
- Much faster training (fewer gradient accumulation steps)
- Better GPU utilization (T4s can handle this)
- Same convergence (effective batch unchanged)
- Better throughput

**Cons**:
- Uses more GPU memory (~12GB per GPU, still safe for T4)
- May need to reduce if OOM occurs

**Caveats**:
- Monitor GPU memory: `nvidia-smi -l 1`
- If OOM, try batch_size: 48 or 32

---

### Strategy 2: Reduce Gradient Accumulation ⚡⚡ **RECOMMENDED**

**Change in config:**
```yaml
training:
  batch_size: 32        # Increase from 16 → 32 (2x)
  gradient_accumulation_steps: 4  # Decrease from 8 → 4 (2x)
  # Effective batch stays same: 256 (32×4×2 GPUs)
```

**Speed Improvement**: 2x faster
**Pros**:
- 2x faster training
- Safer memory usage (~8GB per GPU)
- Same convergence
- Good balance of speed and safety

**Cons**:
- Slightly more memory than current

**Caveats**:
- Very safe for T4 GPUs
- Recommended starting point

---

### Strategy 3: Increase Logging Frequency 📊 **FIX LOSS DISPLAY**

**Change in config:**
```yaml
training:
  log_interval: 10      # Decrease from 100 → 10
  eval_interval: 500    # Decrease from 1000 → 500
```

**Speed Improvement**: None (but better visibility)
**Pros**:
- See loss updates more frequently
- Better monitoring
- Catch issues earlier
- More responsive feedback

**Cons**:
- Slightly more I/O overhead (negligible)
- More log entries

**Caveats**:
- Doesn't speed up training, just improves visibility
- Combine with other strategies

---

### Strategy 4: Reduce Context Length ⚡⚡⚡ **MOST AGGRESSIVE**

**Change in config:**
```yaml
model:
  context_length: 256   # Decrease from 512 → 256

data:
  max_length: 256       # Match context_length
```

**Speed Improvement**: 3-4x faster
**Pros**:
- Much faster training (4x less computation per token)
- Much less memory (can increase batch size further)
- Faster convergence (more steps per epoch)

**Cons**:
- Model can only see 256 tokens at once (vs 512)
- May hurt performance on long-context tasks
- Different model architecture

**Caveats**:
- Changes model capability
- Only do if 256 tokens is sufficient for your use case
- Can't resume from existing checkpoints

---

### Strategy 5: Disable Gradient Checkpointing ⚡ **SPEED BOOST**

**Change in config:**
```yaml
model:
  use_gradient_checkpointing: false  # Disable
```

**Speed Improvement**: 1.5-2x faster
**Pros**:
- Faster backward pass
- Simpler computation graph
- Better GPU utilization

**Cons**:
- Uses more GPU memory (~2-3GB more per GPU)
- May cause OOM if batch size is too large

**Caveats**:
- Only disable if you have memory headroom
- Combine with batch size reduction if needed
- Monitor GPU memory carefully

---

### Strategy 6: Increase num_workers 📦 **DATA LOADING**

**Change in config:**
```yaml
data:
  num_workers: 8        # Increase from 4 → 8
```

**Speed Improvement**: 10-20% faster (if data loading is bottleneck)
**Pros**:
- Faster data loading
- Better CPU utilization
- Reduces GPU idle time

**Cons**:
- More CPU/RAM usage
- May not help if GPU is bottleneck

**Caveats**:
- Only helps if you see GPU utilization < 90%
- Check with `nvidia-smi -l 1`

---

### Strategy 7: Enable Model Compilation 🔥 **PYTORCH 2.0+**

**Change in config:**
```yaml
system:
  compile_model: true   # Enable torch.compile
```

**Speed Improvement**: 1.3-1.5x faster
**Pros**:
- Optimized CUDA kernels
- Better GPU utilization
- No accuracy loss

**Cons**:
- First epoch is slower (compilation time)
- Requires PyTorch 2.0+
- May have compatibility issues

**Caveats**:
- Only works with PyTorch 2.0+
- First run will be slow (compiling)
- May not work with all operations

---

## 🎯 Recommended Optimization Combinations

### Combo 1: Safe & Fast (2-3x speedup) ✅ **RECOMMENDED**

```yaml
training:
  batch_size: 32        # 2x increase
  gradient_accumulation_steps: 4  # 2x decrease
  log_interval: 10      # 10x more frequent

data:
  num_workers: 8        # 2x increase
```

**Expected speed**: 2-3x faster
**Memory usage**: ~8GB per GPU (safe)
**Risk**: Low

---

### Combo 2: Aggressive (4-5x speedup) ⚡

```yaml
model:
  use_gradient_checkpointing: false  # Disable

training:
  batch_size: 64        # 4x increase
  gradient_accumulation_steps: 2  # 4x decrease
  log_interval: 10      # 10x more frequent

data:
  num_workers: 8        # 2x increase

system:
  compile_model: true   # Enable
```

**Expected speed**: 4-5x faster
**Memory usage**: ~12-14GB per GPU
**Risk**: Medium (may OOM)

---

### Combo 3: Maximum Speed (6-8x speedup) 🚀

```yaml
model:
  context_length: 256   # 2x decrease
  use_gradient_checkpointing: false  # Disable

training:
  batch_size: 96        # 6x increase
  gradient_accumulation_steps: 1  # No accumulation
  log_interval: 10      # 10x more frequent

data:
  max_length: 256       # Match context
  num_workers: 8        # 2x increase

system:
  compile_model: true   # Enable
```

**Expected speed**: 6-8x faster
**Memory usage**: ~14-15GB per GPU
**Risk**: High (may OOM, changes model)

---

## 📊 Detailed Comparison Table

| Strategy | Speed Gain | Memory Impact | Risk | Convergence Impact |
|----------|------------|---------------|------|-------------------|
| Batch size 32 | 2x | +2GB | Low | None |
| Batch size 64 | 3-4x | +6GB | Medium | None |
| Batch size 96 | 4-5x | +10GB | High | None |
| Grad accum 4 | 2x | None | None | None |
| Grad accum 2 | 3x | None | None | None |
| Grad accum 1 | 4x | None | None | None |
| Context 256 | 3-4x | -4GB | None | May hurt long context |
| No grad checkpoint | 1.5-2x | +2-3GB | Medium | None |
| num_workers 8 | 1.1-1.2x | +1GB RAM | Low | None |
| compile_model | 1.3-1.5x | None | Low | None |
| log_interval 10 | 0x | Negligible | None | None (visibility only) |

---

## 🔍 Diagnosing Bottlenecks

### Check GPU Utilization
```bash
# Monitor GPU usage
nvidia-smi -l 1

# Look for:
# - GPU Utilization: Should be 90-100%
# - Memory Usage: Should be 70-90% of 16GB
# - Power Usage: Should be near max (70W for T4)
```

**If GPU utilization < 80%**: Data loading is bottleneck
- Increase `num_workers`
- Use faster storage (SSD)
- Reduce data preprocessing

**If GPU utilization > 95%**: GPU is bottleneck (good!)
- Increase batch size
- Reduce gradient accumulation
- Disable gradient checkpointing

### Check Training Speed
```bash
# Time 100 steps
time python train.py --config config/gpu_training_117m_balanced.yaml --max-steps 100

# Calculate steps/second
# Target: 0.5-1.0 steps/sec for 117M model on 2x T4
```

---

## 🎯 Step-by-Step Optimization Process

### Step 1: Baseline (Current)
```yaml
# Current config - measure speed
batch_size: 16
gradient_accumulation_steps: 8
```
**Measure**: Time 100 steps, note GPU utilization

### Step 2: Safe Optimization
```yaml
# Recommended starting point
batch_size: 32
gradient_accumulation_steps: 4
log_interval: 10
num_workers: 8
```
**Measure**: Should be 2x faster, GPU util should increase

### Step 3: Aggressive (if Step 2 works)
```yaml
# Push further
batch_size: 64
gradient_accumulation_steps: 2
use_gradient_checkpointing: false
```
**Measure**: Should be 4x faster, watch for OOM

### Step 4: Maximum (if Step 3 works)
```yaml
# Maximum speed
batch_size: 96
gradient_accumulation_steps: 1
compile_model: true
```
**Measure**: Should be 6x faster, may OOM

---

## 🐛 Troubleshooting

### Out of Memory (OOM)
```
RuntimeError: CUDA out of memory
```

**Solutions** (in order):
1. Reduce `batch_size` by half
2. Increase `gradient_accumulation_steps` by 2x
3. Enable `use_gradient_checkpointing: true`
4. Reduce `context_length` to 256

### Slow Training Despite Changes
```bash
# Check if data loading is bottleneck
nvidia-smi dmon -s u

# If GPU util < 80%, increase num_workers
# If GPU util > 95%, you're GPU-bound (good!)
```

### Loss Not Updating
```yaml
# Increase logging frequency
training:
  log_interval: 10  # Log every 10 steps instead of 100
```

---

## 📈 Expected Performance

### Current (Baseline)
- **Speed**: ~2-3 sec/step
- **Throughput**: ~8,000 tokens/sec
- **Time to 36,621 steps**: ~30-40 hours

### After Safe Optimization (Combo 1)
- **Speed**: ~0.8-1.0 sec/step
- **Throughput**: ~25,000 tokens/sec
- **Time to 36,621 steps**: ~10-12 hours

### After Aggressive Optimization (Combo 2)
- **Speed**: ~0.4-0.5 sec/step
- **Throughput**: ~50,000 tokens/sec
- **Time to 36,621 steps**: ~5-6 hours

### After Maximum Optimization (Combo 3)
- **Speed**: ~0.2-0.3 sec/step
- **Throughput**: ~80,000 tokens/sec
- **Time to 36,621 steps**: ~3-4 hours

---

## 💡 Pro Tips

1. **Start conservative**: Use Combo 1 first, then push further
2. **Monitor memory**: Keep GPU memory at 70-90% utilization
3. **Test with --max-steps 100**: Quick iteration on config changes
4. **Use TensorBoard**: Monitor loss curves for any issues
5. **Save checkpoints frequently**: In case of OOM crashes
6. **Benchmark each change**: Measure before and after

---

## 🎯 Quick Reference: Config Changes

### For 2x T4 GPUs (Recommended)
```yaml
model:
  use_gradient_checkpointing: false  # Disable for speed

training:
  batch_size: 48                     # Increase from 16
  gradient_accumulation_steps: 3     # Decrease from 8
  log_interval: 10                   # Increase from 100

data:
  num_workers: 8                     # Increase from 4
```

**Expected**: 3-4x speedup, ~10 hours total training time

---

## 📞 Need Help?

1. **Check GPU utilization**: `nvidia-smi -l 1`
2. **Monitor memory**: Should be 70-90% of 16GB
3. **Test incrementally**: Change one thing at a time
4. **Measure everything**: Time 100 steps before and after changes

Start with **Combo 1** (Safe & Fast) and work your way up!