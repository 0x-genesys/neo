# OOM Quick Fix - TL;DR

## 🚨 You Got OOM Error

Your GPU 0 ran out of memory (13.42GB used of 14.56GB available).

## 🎯 Immediate Solution (Pick One)

### Option 1: Use Low Memory Config ✅ **RECOMMENDED**

```bash
# Set memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Use low memory config
python train.py --config config/gpu_training_117m_balanced_low_memory.yaml
```

**What it does**:
- Reduces batch_size: 16 → 8
- Reduces context: 512 → 384
- Enables gradient checkpointing
- **Memory**: ~8-9GB per GPU (safe!)

---

### Option 2: Use Emergency Fix Script

```bash
./fix_oom.sh
```

Interactive menu with 3 options:
1. Low memory config (recommended)
2. Ultra low memory (very safe)
3. Single GPU (slower but works)

---

### Option 3: Use Single GPU

```bash
export CUDA_VISIBLE_DEVICES=0
python train.py --config config/gpu_training_117m_balanced.yaml
```

**What it does**:
- Uses only GPU 0
- No DataParallel overhead
- **Memory**: ~8-9GB
- **Speed**: 2x slower

---

## 🔍 Why You Got OOM

**DataParallel Problem**: GPU 0 does extra work
- Gathers gradients from all GPUs
- Updates optimizer
- Broadcasts model back

**Your memory usage**:
- GPU 0: 13.42GB (too much!)
- GPU 1: Probably 7-8GB (fine)

**Solution**: Reduce memory on GPU 0

---

## 📊 Quick Comparison

| Solution | Memory | Speed | Context | Risk |
|----------|--------|-------|---------|------|
| **Low Memory Config** | 8-9GB | Moderate | 384 | Safe ✅ |
| **Single GPU** | 8-9GB | Slow | 512 | Safe ✅ |
| **Ultra Low** | 6-7GB | Very Slow | 256 | Very Safe ✅ |

---

## 💡 Pro Tips

1. **Always set this first**:
   ```bash
   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
   ```

2. **Monitor memory**:
   ```bash
   watch -n 1 nvidia-smi
   ```

3. **Start conservative**: Use low memory config first

4. **If still OOM**: Reduce batch_size to 4 in config

---

## 🎯 Recommended Workflow

```bash
# 1. Set memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 2. Use low memory config
python train.py --config config/gpu_training_117m_balanced_low_memory.yaml

# 3. Monitor in another terminal
watch -n 1 nvidia-smi
```

---

## 🐛 If Still OOM

Edit `config/gpu_training_117m_balanced_low_memory.yaml`:

```yaml
training:
  batch_size: 4  # Change from 8 to 4
  gradient_accumulation_steps: 32  # Change from 16 to 32
```

Or use single GPU:
```bash
export CUDA_VISIBLE_DEVICES=0
```

---

## 📞 Quick Help

**Q: Why GPU 0 uses more memory?**
A: DataParallel gathers all gradients to GPU 0

**Q: Will this slow down training?**
A: Yes, but it will work. Better slow than OOM.

**Q: Can I use original config?**
A: Not with 2 GPUs on T4. Use low memory config or single GPU.

**Q: What's the fastest safe option?**
A: Low memory config with PYTORCH_CUDA_ALLOC_CONF set.

---

## 🚀 Start Here

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python train.py --config config/gpu_training_117m_balanced_low_memory.yaml
```

This will work! ✅