# Quick Optimization Reference - 2x T4 GPUs

## 🚀 TL;DR - Just Tell Me What To Do

### Option 1: Safe & Fast (Recommended) ✅
```bash
python train.py --config config/gpu_training_117m_balanced_fast.yaml
```
- **Speed**: 3-4x faster
- **Time**: ~10 hours (vs ~30-40 hours)
- **Risk**: Low
- **Memory**: ~10-12GB per GPU

### Option 2: Maximum Speed 🚀
```bash
python train.py --config config/gpu_training_117m_balanced_ultra_fast.yaml
```
- **Speed**: 6-8x faster
- **Time**: ~3-5 hours (vs ~30-40 hours)
- **Risk**: Medium (may OOM)
- **Memory**: ~14-15GB per GPU
- **Caveat**: Context length 256 (vs 512)

## 📊 Quick Comparison

| Config | Speed | Time | Memory | Risk | Context |
|--------|-------|------|--------|------|---------|
| **Original** | 1x | 30-40h | 8GB | None | 512 |
| **Fast** | 3-4x | 10h | 12GB | Low | 512 |
| **Ultra Fast** | 6-8x | 5h | 15GB | Medium | 256 |

## 🔧 Manual Optimization (Edit Config Yourself)

### Fix Loss Display Issue
```yaml
training:
  log_interval: 10  # Change from 100
```

### Speed Up Training (Safe)
```yaml
training:
  batch_size: 48                    # Change from 16
  gradient_accumulation_steps: 3   # Change from 8

model:
  use_gradient_checkpointing: false # Change from true

data:
  num_workers: 8                    # Change from 4
```

### Speed Up Training (Aggressive)
```yaml
training:
  batch_size: 96                    # Change from 16
  gradient_accumulation_steps: 1   # Change from 8

model:
  context_length: 256               # Change from 512
  use_gradient_checkpointing: false # Change from true

data:
  max_length: 256                   # Change from 512
  num_workers: 8                    # Change from 4

system:
  compile_model: true               # Change from false
```

## 🎯 Decision Tree

```
Do you need 512 token context?
├─ YES → Use config/gpu_training_117m_balanced_fast.yaml
│         (3-4x faster, ~10 hours)
│
└─ NO → Do you have other GPU processes running?
         ├─ YES → Use config/gpu_training_117m_balanced_fast.yaml
         │         (safer, won't OOM)
         │
         └─ NO → Use config/gpu_training_117m_balanced_ultra_fast.yaml
                   (6-8x faster, ~5 hours)
```

## 🐛 Troubleshooting

### Out of Memory
```bash
# Reduce batch size in config
batch_size: 32  # Instead of 48 or 96
```

### Still Slow
```bash
# Check GPU utilization
nvidia-smi -l 1

# Should see:
# - GPU Util: 90-100%
# - Memory: 70-90% of 16GB
# - Power: Near max (70W for T4)
```

### Loss Not Updating
```yaml
# In config, change:
training:
  log_interval: 10  # Log every 10 steps
```

## 💡 Pro Tips

1. **Start with Fast config** - Safe and proven
2. **Monitor GPU memory** - `nvidia-smi -l 1`
3. **Test with --max-steps 100** - Quick validation
4. **Use TensorBoard** - `tensorboard --logdir logs/`
5. **Save checkpoints** - Training saves every 1000 steps

## 📈 Expected Performance

### Fast Config
```
Step 1/36621 | Loss: 10.234 | Time: 0.8s
Step 10/36621 | Loss: 9.876 | Time: 0.8s
Step 100/36621 | Loss: 7.543 | Time: 0.8s
```

### Ultra Fast Config
```
Step 1/36621 | Loss: 10.234 | Time: 0.3s  (first epoch slower due to compile)
Step 10/36621 | Loss: 9.876 | Time: 0.3s
Step 100/36621 | Loss: 7.543 | Time: 0.2s
```

## 🎯 Recommended Workflow

1. **Start with Fast config**
   ```bash
   python train.py --config config/gpu_training_117m_balanced_fast.yaml --max-steps 100
   ```

2. **Check if it works** (no OOM, good speed)

3. **If successful, run full training**
   ```bash
   python train.py --config config/gpu_training_117m_balanced_fast.yaml
   ```

4. **If you want even faster** (and don't need 512 context)
   ```bash
   python train.py --config config/gpu_training_117m_balanced_ultra_fast.yaml
   ```

## 📞 Quick Help

**Slow training?** → Use Fast or Ultra Fast config
**Loss not showing?** → Set `log_interval: 10`
**Out of memory?** → Reduce `batch_size`
**Need 512 context?** → Use Fast config (not Ultra Fast)

Start here: `config/gpu_training_117m_balanced_fast.yaml` ✅