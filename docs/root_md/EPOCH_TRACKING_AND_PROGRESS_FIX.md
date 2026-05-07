# Epoch Tracking and Progress Bar Fix

## Issues Fixed

### 1. Incorrect Epoch When Resuming from Checkpoint

**Problem:** When resuming from step 7500, the trainer showed "Epoch 0" instead of the correct epoch (e.g., Epoch 1).

**Root Cause:** The checkpoint didn't have the `epoch` field saved, or it wasn't being calculated from the step number.

**Solution:** Added epoch calculation from global_step when epoch is not in checkpoint:

```python
if 'epoch' in checkpoint:
    self.epoch = checkpoint['epoch']
    print(f"   ✅ Resuming from epoch: {self.epoch}")
else:
    # Calculate epoch from global_step if not in checkpoint
    if self.global_step > 0 and hasattr(self.train_loader, 'dataset'):
        batch_size = self.config['training'].get('batch_size', 16)
        grad_accum = self.config['training'].get('gradient_accumulation_steps', 1)
        dataset_size = len(self.train_loader.dataset)
        steps_per_epoch = (dataset_size // batch_size) // grad_accum
        if steps_per_epoch > 0:
            self.epoch = self.global_step // steps_per_epoch
            print(f"   ✅ Calculated epoch from step: {self.epoch}")
```

**Formula:**
```
steps_per_epoch = (dataset_size / batch_size) / gradient_accumulation_steps
current_epoch = global_step / steps_per_epoch
```

**Example:**
- Dataset: 100,000 samples
- Batch size: 128
- Gradient accumulation: 4
- Steps per epoch: (100,000 / 128) / 4 = 195 steps
- At step 7500: epoch = 7500 / 195 = 38 (approximately)

### 2. No Progress Bar (tqdm)

**Problem:** No visual progress indicator during training, making it hard to track progress through each epoch.

**Solution:** Added tqdm progress bars to both training and validation loops:

#### Training Loop
```python
from tqdm import tqdm
pbar = tqdm(enumerate(train_loader), total=len(train_loader), 
           desc=f"Epoch {self.epoch}", 
           disable=False)

for batch_idx, batch in pbar:
    # ... training code ...
    
    # Update progress bar with metrics
    if hasattr(pbar, 'set_postfix'):
        pbar.set_postfix({
            'loss': f'{avg_loss:.4f}',
            'lr': f'{lr:.2e}',
            'step': self.global_step
        })
```

#### Validation Loop
```python
from tqdm import tqdm
val_iter = tqdm(para_loader.per_device_loader(self.device), 
               desc="Validation", 
               total=len(self.val_loader),
               disable=False)

for batch in val_iter:
    # ... validation code ...
```

## New Output Format

### Before (No Progress Bar)
```
Epoch 0 | Step 7510 | Loss: 2.9139 | LR: 2.00e-04
Epoch 0 | Step 7520 | Loss: 2.8884 | LR: 2.00e-04
Epoch 0 | Step 7530 | Loss: 2.9100 | LR: 2.00e-04
```

### After (With Progress Bar and Correct Epoch)
```
📥 Loading checkpoint from: checkpoints/checkpoint_step_7500.pt
   ✅ Model weights loaded
   ✅ Resuming from step: 7500
   ✅ Calculated epoch from step: 38
   ✅ Checkpoint loaded successfully!
   Continuing training from epoch 38, step 7500

Epoch 38: 100%|████████████| 195/195 [02:30<00:00, 1.29it/s, loss=2.9139, lr=2.00e-04, step=7510]
Epoch 38 | Step 7510 | Loss: 2.9139 | LR: 2.00e-04
Epoch 38 | Step 7520 | Loss: 2.8884 | LR: 2.00e-04

Validation: 100%|████████████| 50/50 [00:15<00:00, 3.21it/s]
📊 Validation Loss: 2.8456

Epoch 39: 45%|████▌     | 88/195 [01:08<01:22, 1.30it/s, loss=2.8832, lr=2.00e-04, step=7540]
```

## Benefits

✅ **Correct epoch tracking** - Shows the actual epoch number when resuming  
✅ **Visual progress** - tqdm shows progress bar with ETA  
✅ **Real-time metrics** - Loss, LR, and step displayed in progress bar  
✅ **Better UX** - Easy to see training progress at a glance  
✅ **Fallback support** - Works even if tqdm is not installed

## Files Modified

- `src/tpu_trainer.py`:
  - `_load_checkpoint_if_needed()`: Added epoch calculation from step
  - `_train_epoch()`: Added tqdm progress bar
  - `_validate()`: Added tqdm progress bar

## Dependencies

The code gracefully handles missing tqdm:
```python
try:
    from tqdm import tqdm
    pbar = tqdm(...)
except ImportError:
    # Fallback to regular enumerate
    pbar = enumerate(train_loader)
```

If tqdm is not installed, install it with:
```bash
pip install tqdm
```
