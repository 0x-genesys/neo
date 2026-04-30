# Training Logging Fixes

## Issues Found and Fixed

### Issue 1: Epoch Restarting ❌

**Problem**: Epoch 0 appeared to restart after completing 100%
```
Epoch 0 | Step 5500/36621: 100%
... (epoch completes)
Epoch 0 | Step 6653/36621:  25%  # ← Epoch 0 again?
```

**Root Cause**: The code had logic to skip batches when resuming mid-epoch (lines 343-344):
```python
start_batch = (self.global_step % (len(self.train_loader) // batches_per_step)) * batches_per_step
pbar = tqdm(self.train_loader, initial=start_batch, ...)
```

This was incorrect because:
1. It tried to set `initial` on the progress bar, which doesn't skip batches
2. The calculation was wrong - it would reset the progress bar mid-training
3. Epochs don't actually restart - it's just the progress bar display that was confusing

**Solution**: Removed the faulty `start_batch` logic and `initial` parameter. The epoch continues normally, and the progress bar now shows the correct state.

### Issue 2: Loss Not Showing in Progress Bar ❌

**Problem**: Loss value was missing or truncated in the progress bar

**Root Cause**: The loss shown in `pbar.set_postfix()` was the **scaled loss** (divided by gradient_accumulation_steps), not the actual loss:
```python
loss = loss / self.config['training']['gradient_accumulation_steps']
# ... later ...
pbar.set_postfix({'loss': f"{loss.item():.4f}"})  # ← Wrong! This is scaled loss
```

**Solution**: Store the actual loss before scaling, then use it in the progress bar:
```python
actual_loss = loss.item()  # Store before scaling
loss = loss / self.config['training']['gradient_accumulation_steps']
# ... later ...
pbar.set_description(f"... | Loss: {actual_loss:.4f} | ...")  # ← Correct!
```

### Issue 3: Progress Bar Confusion ❌

**Problem**: Progress bar showed confusing mixed metrics:
```
Epoch 0 | Step 6653/36621 (next eval: 7000):  25%|▎| 21489/84961
```
- Description: Steps (6653/36621)
- Progress: Batches (21489/84961)
- These are different metrics!

**Root Cause**: The progress bar iterates over batches, but the description talks about steps. With gradient_accumulation_steps > 1, these are different.

**Solution**: Moved step info and loss to the description, making it clear and informative:
```
Epoch 0 | Step 6653/36621 | Loss: 3.2451 | LR: 3.00e-04:  25%|▎| 21489/84961
```

Now it's clear:
- **Description**: Current step, loss, learning rate
- **Progress bar**: Batch progress within epoch

## Code Changes

### Before (Buggy)
```python
def train_epoch(self):
    # Calculate how many batches to skip if resuming mid-epoch
    batches_per_step = self.config['training']['gradient_accumulation_steps']
    start_batch = (self.global_step % (len(self.train_loader) // batches_per_step)) * batches_per_step
    
    pbar = tqdm(
        self.train_loader, 
        desc=f"Epoch {self.epoch}",
        initial=start_batch,  # ← BUG: Doesn't actually skip batches
        total=len(self.train_loader)
    )
    
    for batch_idx, (input_ids, targets) in enumerate(pbar):
        # ...
        loss = loss / self.config['training']['gradient_accumulation_steps']  # Scale loss
        
        # ...
        
        pbar.set_description(
            f"Epoch {self.epoch} | Step {self.global_step}/{max_steps} "
            f"(next eval: {next_eval})"  # ← No loss shown
        )
        
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",  # ← BUG: Scaled loss, not actual loss
            'step': f"{self.global_step}/{max_steps}"
        })
```

### After (Fixed)
```python
def train_epoch(self):
    # No more start_batch calculation - it was buggy
    
    pbar = tqdm(
        self.train_loader, 
        desc=f"Epoch {self.epoch} | Step {self.global_step}/{max_steps}",
        total=len(self.train_loader)
    )
    
    for batch_idx, (input_ids, targets) in enumerate(pbar):
        # ...
        actual_loss = loss.item()  # ← Store actual loss before scaling
        loss = loss / self.config['training']['gradient_accumulation_steps']
        
        # ...
        
        lr = self.optimizer.param_groups[0]['lr']
        
        pbar.set_description(
            f"Epoch {self.epoch} | Step {self.global_step}/{max_steps} | "
            f"Loss: {actual_loss:.4f} | LR: {lr:.2e}"  # ← Shows actual loss and LR
        )
```

## What You'll See Now

### Before (Confusing)
```
Epoch 0 | Step 6653/36621 (next eval: 7000):  25%|▎| 21489/84961 [2:33:04<7:57:5
```
- No loss visible
- Confusing metrics
- Epoch seems to restart

### After (Clear)
```
Epoch 0 | Step 6653/36621 | Loss: 3.2451 | LR: 3.00e-04:  25%|▎| 21489/84961 [2:33:04<7:57:5
```
- ✅ Loss clearly visible
- ✅ Learning rate shown
- ✅ Clear step progress
- ✅ Epoch progresses correctly

## Additional Improvements

### Console Logging
The periodic console logs now also use the actual loss:
```python
print(f"Step {self.global_step}/{max_steps} | "
      f"Loss: {actual_loss:.4f} | "  # ← Actual loss
      f"LR: {lr:.2e}")
```

### TensorBoard Logging
Metrics logged to TensorBoard use the actual loss:
```python
self._log_metrics({
    'train/loss': actual_loss,  # ← Actual loss
    'train/lr': lr,
    'train/step': self.global_step
})
```

## Understanding the Metrics

### Steps vs Batches

With `gradient_accumulation_steps = 8`:
- **1 step** = 8 batches processed
- **Batch progress**: Shows individual batches (0-84961)
- **Step progress**: Shows optimization steps (0-36621)

Example:
```
Epoch 0 | Step 6653/36621 | Loss: 3.2451 | LR: 3.00e-04:  25%|▎| 21489/84961
         ^^^^^^^^^^^^^^^                                        ^^^^^^^^^^^^^^
         Optimization steps                                     Batches processed
         (every 8 batches)                                      (individual batches)
```

### Why Two Metrics?

- **Batches**: Physical data processed (for progress bar)
- **Steps**: Optimizer updates (for learning rate schedule, checkpoints, etc.)

With gradient accumulation, you process multiple batches before updating the model, so steps < batches.

## Verification

To verify the fixes work:

1. **Check loss is visible**:
   ```
   Epoch 0 | Step 100/36621 | Loss: 4.5123 | LR: 1.50e-04: ...
                                    ^^^^^^^
                                    Should be visible!
   ```

2. **Check epoch doesn't restart**:
   - Epoch number should only increment when all batches are processed
   - Step number should continuously increase

3. **Check console logs**:
   ```
   ================================================================================
   Step 1000/36621 | Loss: 3.8765 | LR: 3.00e-04
   Next: Log@1010 | Eval@1500 | Save@2000
   ================================================================================
   ```

## Summary

✅ **Fixed epoch restarting** - Removed buggy batch skipping logic
✅ **Fixed loss display** - Shows actual loss, not scaled loss
✅ **Improved progress bar** - Clear, informative description with loss and LR
✅ **Consistent logging** - All logs use actual loss values

The training loop now provides clear, accurate progress information!
