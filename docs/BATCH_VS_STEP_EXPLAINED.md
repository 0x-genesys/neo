# Batch vs Step - Understanding Training Progress

## The Confusion

You're at **batch 1001** and wondering why validation hasn't run yet, even though `eval_interval: 1000`.

## The Key Difference

### Batch ≠ Step!

With `gradient_accumulation_steps: 8`:
- **8 batches** = **1 step**
- Validation runs every **1000 steps** (not batches)
- 1000 steps = **8000 batches**

## Your Current Progress

```
Batch: 1001 / 84,961
Step:  125  / 36,621  (1001 ÷ 8 = 125)
```

### When Will Validation Run?

**First validation**: Step 1000 = Batch 8000
**Second validation**: Step 2000 = Batch 16000
**Third validation**: Step 3000 = Batch 24000

You're currently at step 125, so you need to wait until step 1000 (batch 8000).

## Why Gradient Accumulation?

### Memory Efficiency

Instead of processing a huge batch at once:
```
❌ Batch size 128 → Requires lots of memory
```

We split it into smaller batches:
```
✅ 8 batches × 16 samples = 128 effective batch size
   Uses 8x less memory!
```

### How It Works

```
Batch 1-8:   Forward + Backward (accumulate gradients)
Step 1:      Optimizer update (apply accumulated gradients)

Batch 9-16:  Forward + Backward (accumulate gradients)
Step 2:      Optimizer update (apply accumulated gradients)

...and so on
```

## Training Timeline

### Per Epoch

```
Total batches per epoch: 84,961
Gradient accumulation:   8
Steps per epoch:         10,620 (84,961 ÷ 8)
```

### Milestones

| Event | Step | Batch | When |
|-------|------|-------|------|
| First log | 100 | 800 | ~5 min |
| Second log | 200 | 1,600 | ~10 min |
| ... | ... | ... | ... |
| **First validation** | **1,000** | **8,000** | **~50 min** |
| First checkpoint | 1,000 | 8,000 | ~50 min |
| Second validation | 2,000 | 16,000 | ~1h 40min |
| Second checkpoint | 2,000 | 16,000 | ~1h 40min |

### Full Training

```
Total steps: 36,621
Total batches: 293,000 (36,621 × 8)
Total epochs: 8

Validations: 36 times (every 1000 steps)
Checkpoints: 36 times (every 1000 steps)
Logs: 366 times (every 100 steps)
```

## Updated Progress Display

The trainer now shows:

### Progress Bar
```
Epoch 0 | Step 125/36621 (next eval: 1000):   1%|█ | 1001/84961 [05:32<10:21:33, 2.27it/s]
│         │                                    │     │
│         │                                    │     └─ Batch progress
│         │                                    └─ Percentage (of batches)
│         └─ Step progress + next evaluation milestone
└─ Current epoch
```

### Console Logs (Every 100 Steps)
```
================================================================================
Step 100/36621 | Loss: 1.234 | LR: 1.30e-04
Next: Log@200 | Eval@1000 | Save@1000
================================================================================
```

This clearly shows:
- Current step
- When next log will appear
- When next validation will run
- When next checkpoint will be saved

### Validation (Every 1000 Steps)
```
================================================================================
🔍 VALIDATION at step 1000
================================================================================
Validation: 100%|██████████| 391/391 [00:45<00:00, 8.64it/s]
Validation loss: 1.123
✅ New best validation loss! Saving best model...
Checkpoint saved: checkpoints/gpu_training_117m_1.5gb_balanced/best_model.pt
================================================================================
```

### Checkpoint (Every 1000 Steps)
```
================================================================================
💾 CHECKPOINT at step 1000
================================================================================
✅ Checkpoint saved
================================================================================
```

## Quick Reference

### Current Config

```yaml
training:
  batch_size: 16
  gradient_accumulation_steps: 8
  log_interval: 100    # Every 100 steps = 800 batches
  eval_interval: 1000  # Every 1000 steps = 8000 batches
  save_interval: 1000  # Every 1000 steps = 8000 batches
```

### Conversion Table

| Steps | Batches | Time (approx) |
|-------|---------|---------------|
| 1 | 8 | ~3 sec |
| 10 | 80 | ~30 sec |
| 100 | 800 | ~5 min |
| 1,000 | 8,000 | ~50 min |
| 10,000 | 80,000 | ~8 hours |
| 36,621 | 293,000 | ~30 hours |

*Based on 2.27 it/s (batches per second)*

## Why This Design?

### 1. Memory Efficiency
- Small batch size (16) fits in limited GPU memory
- Gradient accumulation achieves large effective batch (128)
- Best of both worlds!

### 2. Training Stability
- Larger effective batch = more stable gradients
- Better convergence
- Less noisy updates

### 3. Flexibility
- Can train on small GPUs (1.5GB)
- Can train on large GPUs (15GB) with higher batch size
- Same effective batch size = same training dynamics

## Adjusting Intervals

If you want more frequent validation/checkpointing:

### Option 1: More Frequent (Every 500 Steps)
```yaml
training:
  eval_interval: 500   # Every 500 steps = 4000 batches (~25 min)
  save_interval: 500   # Every 500 steps = 4000 batches (~25 min)
```

### Option 2: Less Frequent (Every 2000 Steps)
```yaml
training:
  eval_interval: 2000  # Every 2000 steps = 16000 batches (~1h 40min)
  save_interval: 2000  # Every 2000 steps = 16000 batches (~1h 40min)
```

### Option 3: Different Intervals
```yaml
training:
  log_interval: 50     # More frequent logs
  eval_interval: 500   # More frequent validation
  save_interval: 1000  # Less frequent checkpoints
```

## Summary

### Key Takeaways

1. **Batch ≠ Step**: 8 batches = 1 step (with gradient_accumulation_steps: 8)
2. **Validation at step 1000**: Not batch 1000, but batch 8000
3. **Current progress**: Batch 1001 = Step 125
4. **Wait time**: ~45 more minutes until first validation

### What You'll See

At your current speed (2.27 it/s):
- **Step 100** (batch 800): First console log - **Already passed!**
- **Step 200** (batch 1,600): Second console log - **~5 minutes from now**
- **Step 1000** (batch 8,000): First validation + checkpoint - **~45 minutes from now**

### Updated Display

The trainer now shows:
- ✅ Current step number
- ✅ Next evaluation milestone
- ✅ Clear console logs every 100 steps
- ✅ Prominent validation messages
- ✅ Prominent checkpoint messages

No more confusion! 🎉
