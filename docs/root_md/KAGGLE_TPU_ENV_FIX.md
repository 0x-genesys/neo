# Kaggle TPU Environment Variable Fix

## Problem

Training was failing with this error:

```
RuntimeError: TPU initialization failed: Invalid --2a886c8_slice_builder_worker_addresses specified. 
Expected 8 worker addresses, got 1.
```

## Root Cause

The `train.py` file was setting these environment variables:

```python
os.environ['PJRT_DEVICE'] = 'TPU'
os.environ['TPU_PROCESS_ADDRESSES'] = 'local'
os.environ['TPU_NUM_DEVICES'] = '8'
```

**Why this caused the error:**

1. **Kaggle TPU VMs are single-host systems** with 8 cores on one machine
2. Setting `TPU_PROCESS_ADDRESSES='local'` tells the SliceBuilder to expect a single worker
3. But setting `TPU_NUM_DEVICES='8'` tells it to expect 8 separate worker addresses
4. This mismatch causes the SliceBuilder to fail initialization

## Solution

**Remove the environment variables** and let PJRT auto-detect the TPU configuration:

```python
# CRITICAL: DO NOT set PJRT environment variables for Kaggle TPU VMs
# Kaggle TPU VMs are single-host with 8 cores, not multi-host pods
# Setting TPU_PROCESS_ADDRESSES='local' causes SliceBuilder errors
# The PJRT runtime will auto-detect the TPU configuration correctly
```

## Why This Works

1. **PJRT auto-detection**: The PJRT runtime automatically detects:
   - TPU device type (v2, v3, v4)
   - Number of cores (8 for TPU v3-8)
   - Single-host vs multi-host configuration
   - Proper worker addressing

2. **Kaggle TPU VM setup**: Kaggle provides:
   - Pre-configured TPU environment
   - Correct PJRT settings
   - Single-host 8-core configuration

3. **torch_xla handles it**: The `xmp.spawn()` call in TPU trainer:
   - Detects available cores automatically
   - Sets up proper distributed training
   - Manages worker processes correctly

## When You WOULD Need These Variables

These environment variables are useful for:

1. **Multi-host TPU pods**: When you have multiple TPU VMs in a pod
2. **Custom TPU configurations**: Non-standard setups
3. **Debugging**: Forcing specific device configurations

But for **Kaggle TPU VMs** (single-host, 8 cores), they're not needed and cause errors.

## Verification

After the fix, training should start successfully:

```
🚀 TPU Trainer Initialized
================================================================================
TPU cores: 8
torch_xla version: 2.5.0
================================================================================

🚀 Starting TPU training on all available TPU cores...
```

## Related Files Changed

1. **train.py**: Removed environment variable setup
2. **src/tpu_trainer.py**: Added automatic checkpoint cleanup after Hub upload
3. **scripts/cleanup_checkpoints.py**: New script for manual checkpoint cleanup
4. **docs/CHECKPOINT_CLEANUP_GUIDE.md**: Guide for managing disk space

## Testing

To verify the fix works:

```bash
# On Kaggle TPU VM
python train.py \
    --config config/auto_training_117m_balanced.yaml \
    --tpu
```

Should see:
- ✅ TPU initialization succeeds
- ✅ 8 cores detected
- ✅ Training starts across all cores
- ✅ No SliceBuilder errors

## Additional Notes

### Checkpoint Cleanup

The TPU trainer now automatically:
1. Uploads checkpoints to HuggingFace Hub
2. Deletes regular checkpoints after successful upload
3. Keeps best model checkpoints locally
4. Keeps error/interrupted checkpoints for recovery

This saves disk space on Kaggle's limited storage.

### Manual Cleanup

Use the cleanup script if needed:

```bash
python scripts/cleanup_checkpoints.py \
    --checkpoint-dir checkpoints/auto_training_117m_balanced \
    --verify-hub \
    --repo-id 0x-genesys/neo_weights_checkpoints
```

## Summary

- ❌ **Before**: Manual environment variables caused SliceBuilder errors
- ✅ **After**: PJRT auto-detection works correctly
- 💾 **Bonus**: Automatic checkpoint cleanup saves disk space
- 🚀 **Result**: Training works on Kaggle TPU VMs
