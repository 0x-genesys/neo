# Kaggle Single-VM TPU Fix

## Problem
The training script was failing with:
```
Expected 8 worker addresses, got 1
```

## Root Cause
The code was using `xmp.spawn()` which is designed for **multi-VM TPU Pods** (distributed training across multiple machines). On Kaggle, you have a **single VM with 8 TPU cores**, not a multi-VM setup.

When you use `xmp.spawn`, PyTorch XLA tries to create a distributed mesh and coordinate 8 separate processes across different VMs. Since you only have one VM, it gets confused and fails.

## Solution
**Bypass `xmp.spawn` for single-VM training.**

For a single VM with 8 TPU cores, you don't need `xmp.spawn`. Instead, use a standard single-process script that uses `ParallelLoader` directly. XLA will handle the parallelism across the 8 cores automatically.

## Key Changes in `src/tpu_trainer.py`

### Before (Multi-Process with xmp.spawn)
```python
def train(self):
    """Start multi-core TPU training with PJRT runtime."""
    xmp.spawn(self._mp_fn, args=(), nprocs=None, start_method='fork')

def _mp_fn(self, rank):
    """Multi-processing function that runs on each TPU core."""
    self.device = xm.xla_device()
    model = xmp.MpModelWrapper(self.model)  # Wrap for multi-process
    model = model.to(self.device)
    # ... training code
```

### After (Single-Process with ParallelLoader)
```python
def train(self):
    """Start single-process TPU training with ParallelLoader."""
    self._train_single_process()

def _train_single_process(self):
    """Single-process training for single-VM TPU."""
    self.device = xm.xla_device()
    model = self.model.to(self.device)  # No MpModelWrapper needed
    
    # ParallelLoader automatically distributes data across 8 cores
    para_loader = pl.ParallelLoader(self.train_loader, [self.device])
    
    for batch in para_loader.per_device_loader(self.device):
        # Standard PyTorch training loop
        # ...
        xm.mark_step()  # Essential for XLA
```

## Why This Works

1. **`xmp.spawn`** is for multi-VM TPU Pods (multiple machines/VMs) that need to coordinate over the network
2. **`ParallelLoader`** is for single machines with multiple chips (your Kaggle VM). It shards the tensors internally without needing to spawn separate process ranks
3. **XLA handles the parallelism** automatically across all 8 cores on your single VM

## What Was Removed

- ❌ `xmp.spawn()` - Not needed for single VM
- ❌ `xmp.MpModelWrapper()` - Not needed for single-process training
- ❌ `DistributedSampler` - Not needed, ParallelLoader handles distribution
- ❌ `xm.reduce_gradients()` - Not needed in single-process mode
- ❌ `xm.mesh_reduce()` - Not needed in single-process mode
- ❌ `xm.master_print()` - Replaced with regular `print()`
- ❌ `xm.is_master_ordinal()` checks - Not needed in single-process mode

## What Was Kept

- ✅ `xm.xla_device()` - Get TPU device
- ✅ `pl.ParallelLoader()` - Distribute data across cores
- ✅ `xm.mark_step()` - Critical XLA optimization barrier
- ✅ `xser.save()` - Memory-optimized checkpointing
- ✅ All training logic, optimizer, scheduler, logging

## Testing

Run your training script on Kaggle:
```bash
python train.py --config config/tpu_training.yaml
```

You should now see:
```
🚀 TPU Trainer Initialized (Single-Process Mode)
📍 Using TPU device: xla:0
📍 World size (cores): 8
📍 Ordinal (rank): 0
```

And training should proceed without the "Expected 8 worker addresses" error.

## Architecture Comparison

### Multi-VM TPU Pod (xmp.spawn)
```
VM 1 (Process 0-3) ←→ Network ←→ VM 2 (Process 4-7)
   ↓                                    ↓
4 TPU cores                        4 TPU cores
```

### Single-VM TPU (ParallelLoader)
```
Single Process
   ↓
ParallelLoader
   ↓
8 TPU cores (automatic distribution)
```

## References

- [PyTorch XLA Documentation](https://pytorch.org/xla/)
- [Kaggle TPU Guide](https://www.kaggle.com/docs/tpu)
- [ParallelLoader API](https://pytorch.org/xla/release/2.0/index.html#torch_xla.distributed.parallel_loader.ParallelLoader)
