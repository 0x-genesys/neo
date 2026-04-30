# Model Output Format Fix

## Problem
```
AttributeError: 'tuple' object has no attribute 'view'
```

## Root Cause
The model's `forward` method returns a **tuple** `(logits, loss)`, but the TPU trainer was expecting either:
- A dict with `{'logits': ...}`
- A single tensor (logits)

### Model Forward Signature
```python
def forward(self, idx, targets=None):
    """
    Returns:
        logits: (B, T, vocab_size) if targets is None
        (logits, loss): if targets is provided
    """
    # ... model code ...
    
    loss = None
    if targets is not None:
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-1
        )
    
    return logits, loss  # Always returns tuple!
```

## Solution
Updated the TPU trainer to handle all three output formats:

### Before (Broken)
```python
# Forward pass
outputs = model(inputs)

# Compute loss
if isinstance(outputs, dict):
    logits = outputs['logits']
else:
    logits = outputs  # ❌ Fails when outputs is a tuple!

loss = nn.functional.cross_entropy(
    logits.view(-1, logits.size(-1)),  # ❌ 'tuple' has no attribute 'view'
    targets.view(-1)
)
```

### After (Fixed)
```python
# Forward pass
outputs = model(inputs, targets)

# Handle model output - can be tuple (logits, loss) or dict
if isinstance(outputs, tuple):
    logits, loss = outputs
    # If model computed loss, use it; otherwise compute manually
    if loss is None:
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1)
        )
elif isinstance(outputs, dict):
    logits = outputs['logits']
    loss = nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1)
    )
else:
    # outputs is just logits tensor
    logits = outputs
    loss = nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1)
    )
```

## Key Changes

1. **Pass targets to model**: `model(inputs, targets)` instead of `model(inputs)`
2. **Handle tuple output**: Check if output is tuple and unpack `(logits, loss)`
3. **Use model's loss if available**: If model computed loss, use it directly
4. **Fallback to manual loss**: Compute loss manually if model didn't compute it
5. **Applied to both training and validation**: Fixed in both `_train_epoch` and `_validate`

## Benefits

✅ **Efficient**: Model computes loss once (no redundant computation)  
✅ **Flexible**: Handles tuple, dict, or tensor outputs  
✅ **Correct**: Properly unpacks tuple before accessing attributes  
✅ **Consistent**: Same logic in training and validation loops

## Testing

The training should now proceed without the tuple error:

```bash
python train.py --config config/tpu_training_117m_balanced.yaml
```

Expected output:
```
🚀 Starting TPU training (single-process, multi-core)...
📍 Using TPU device: xla:0
📍 World size (cores): 1
📍 Ordinal (rank): 0

Epoch 0 | Step 10 | Loss: 10.xxxx | LR: x.xxe-xx
Epoch 0 | Step 20 | Loss: 9.xxxx | LR: x.xxe-xx
...
```

## Related Files Modified

- `src/tpu_trainer.py`: Fixed `_train_epoch()` and `_validate()` methods
