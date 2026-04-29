# Training with Binary Dataset

## Overview

The training script (`train.py`) now supports both HuggingFace datasets and pre-tokenized binary datasets. This guide explains how to use the binary dataset format.

## Binary Dataset Format

Binary datasets offer several advantages:
- **Faster loading**: No tokenization overhead
- **Memory efficient**: Memory-mapped file access
- **Deterministic**: Pre-shuffled and consistent
- **Portable**: Easy to share and reproduce

## Configuration

### Binary Dataset Config

```yaml
data:
  dataset_type: "binary"  # Specify binary format
  train_file: "data/balanced_300m/train.bin"
  val_file: "data/balanced_300m/val.bin"
  max_length: 256
  num_workers: 0
  pin_memory: false
```

### HuggingFace Dataset Config (for comparison)

```yaml
data:
  dataset_name: "openwebtext"  # HuggingFace dataset name
  dataset_config: null
  train_split: "train"
  val_split: "train"
  max_length: 256
  num_workers: 0
  pin_memory: false
```

## Training Commands

### Using Binary Dataset

```bash
# Standard training
python train.py --config config/gpu_training_117m_1.5gb.yaml

# Multi-GPU training (2x T4)
python train.py --config config/gpu_training_117m_1.5gb.yaml --multi-gpu

# Resume from checkpoint
python train.py --config config/gpu_training_117m_1.5gb.yaml --resume checkpoints/checkpoint.pt

# Override batch size
python train.py --config config/gpu_training_117m_1.5gb.yaml --batch-size 32
```

### Using HuggingFace Dataset

```bash
# Standard training
python train.py --config config/gpu_training_117m.yaml

# Override dataset
python train.py --config config/gpu_training_117m.yaml --dataset wikitext
```

## Multi-GPU Training

The script automatically detects and uses multiple GPUs when available:

### Automatic Detection

```bash
# Uses all available GPUs
python train.py --config config/gpu_training_117m_1.5gb.yaml --multi-gpu
```

**Output**:
```
✅ Multi-GPU training enabled!
   Using GPUs: [0, 1]
   Total GPUs: 2
   GPU 0: Tesla T4 (15.64GB)
   GPU 1: Tesla T4 (15.64GB)
```

### Specific GPUs

```bash
# Use specific GPUs (e.g., GPU 0 and 2)
python train.py --config config/gpu_training_117m_1.5gb.yaml --gpu-ids 0,2
```

### Effective Batch Size

With multi-GPU training, the effective batch size is multiplied:

```
Single GPU:
  batch_size: 16
  gradient_accumulation: 8
  effective_batch: 128

Multi-GPU (2 GPUs):
  batch_size: 16 per GPU
  gradient_accumulation: 8
  effective_batch: 256 (16 × 2 × 8)
```

**Note**: You may want to adjust learning rate when changing effective batch size.

## Configuration Display

The training script now intelligently displays configuration based on dataset type:

### Binary Dataset

```
================================================================================
Configuration:
================================================================================
Dataset: Binary format
  Train file: data/balanced_300m/train.bin
  Val file: data/balanced_300m/val.bin
Model: 12 layers, 768 dim, 12 heads
Context length: 256
Batch size: 16
Effective batch size: 32 (per-GPU: 16)
Learning rate: 0.0002
Max epochs: 8
Max steps: 36621
================================================================================
```

### HuggingFace Dataset

```
================================================================================
Configuration:
================================================================================
Dataset: openwebtext
Model: 12 layers, 768 dim, 12 heads
Context length: 256
Batch size: 16
Learning rate: 0.0002
Max epochs: 8
Max steps: 36621
================================================================================
```

## Command-Line Overrides

### Dataset Override

```bash
# Override with binary file
python train.py --config config/gpu_training_117m.yaml --dataset data/custom/train.bin

# Override with HuggingFace dataset
python train.py --config config/gpu_training_117m_1.5gb.yaml --dataset wikitext
```

The script automatically detects the type:
- If path ends with `.bin` → Binary dataset
- Otherwise → HuggingFace dataset

### Other Overrides

```bash
# Batch size
python train.py --config config/gpu_training_117m_1.5gb.yaml --batch-size 32

# Learning rate
python train.py --config config/gpu_training_117m_1.5gb.yaml --lr 1.5e-4

# Epochs
python train.py --config config/gpu_training_117m_1.5gb.yaml --epochs 12

# Resume training
python train.py --config config/gpu_training_117m_1.5gb.yaml --resume checkpoints/checkpoint.pt
```

## Troubleshooting

### Issue: KeyError: 'dataset_name'

**Cause**: Using old version of `train.py` with binary dataset config

**Fix**: The script has been updated to handle both formats. Make sure you're using the latest version.

### Issue: Binary file not found

**Error**:
```
FileNotFoundError: Binary file not found: data/balanced_300m/train.bin
```

**Fix**: Prepare the dataset first:
```bash
python scripts/prepare_balanced_dataset.py
```

### Issue: Vocab size mismatch

**Error**:
```
RuntimeError: vocab_size mismatch
```

**Fix**: Ensure tokenizer matches the one used to create binary dataset:
```yaml
tokenizer:
  type: "tiktoken"  # Must match dataset preparation
  vocab_size: 100277
```

### Issue: Multi-GPU not working

**Symptoms**: Only using 1 GPU despite `--multi-gpu` flag

**Checks**:
1. Verify multiple GPUs available: `nvidia-smi`
2. Check CUDA is available: `python -c "import torch; print(torch.cuda.device_count())"`
3. Ensure no other process is using GPUs

**Fix**: The script will automatically fall back to single GPU if multi-GPU is not available.

## Performance Tips

### Single GPU (1.5GB)

```bash
# Optimized for memory
python train.py --config config/gpu_training_117m_1.5gb.yaml
```

**Expected**:
- Steps per second: ~0.5-1.0
- Training time: 10-20 hours
- Memory usage: ~1.4GB

### Multi-GPU (2x T4, 15GB each)

```bash
# Optimized for speed
python train.py --config config/gpu_training_117m_1.5gb.yaml --multi-gpu --batch-size 32
```

**Expected**:
- Steps per second: ~5-10
- Training time: 1-2 hours
- Memory usage: ~8GB per GPU

### Adjusting for Multi-GPU

When using multiple GPUs, consider:

1. **Increase batch size**: More GPUs = can handle larger batches
   ```bash
   --batch-size 32  # or 64 for 2x T4
   ```

2. **Adjust learning rate**: Larger effective batch may need higher LR
   ```bash
   --lr 3.0e-4  # increase from 2.0e-4
   ```

3. **Reduce gradient accumulation**: Less needed with larger batch
   ```yaml
   gradient_accumulation_steps: 4  # reduce from 8
   ```

## Monitoring Training

### TensorBoard

```bash
# Start TensorBoard
tensorboard --logdir logs/gpu_training_117m_1.5gb_balanced

# Open browser to http://localhost:6006
```

### GPU Monitoring

```bash
# Watch GPU usage in real-time
watch -n 1 nvidia-smi

# Or use gpustat (if installed)
watch -n 1 gpustat
```

### Training Logs

The script prints progress every 100 steps:

```
Epoch 1/8 | Step 100/36621 | Loss: 3.45 | LR: 1.3e-4 | Tokens/sec: 32768
Epoch 1/8 | Step 200/36621 | Loss: 3.12 | LR: 1.6e-4 | Tokens/sec: 35840
...
```

## Best Practices

1. **Start with small test**: Test with 1 epoch first
   ```bash
   python train.py --config config/gpu_training_117m_1.5gb.yaml --epochs 1
   ```

2. **Monitor memory**: Watch `nvidia-smi` during first few steps

3. **Save checkpoints**: Checkpoints saved every 1000 steps automatically

4. **Use validation**: Monitor validation loss to detect overfitting

5. **Resume if interrupted**: Use `--resume` to continue from checkpoint

## Example Workflow

```bash
# 1. Prepare dataset (one-time)
python scripts/prepare_balanced_dataset.py

# 2. Test dataset
python scripts/test_balanced_dataset.py

# 3. Start training
python train.py --config config/gpu_training_117m_1.5gb.yaml --multi-gpu

# 4. Monitor progress
tensorboard --logdir logs/gpu_training_117m_1.5gb_balanced

# 5. If interrupted, resume
python train.py --config config/gpu_training_117m_1.5gb.yaml --resume checkpoints/gpu_training_117m_1.5gb_balanced/checkpoint.pt

# 6. Evaluate model
python evaluate.py --checkpoint checkpoints/gpu_training_117m_1.5gb_balanced/best_model.pt

# 7. Generate text
python src/inference.py --checkpoint checkpoints/gpu_training_117m_1.5gb_balanced/best_model.pt
```

## Summary

The training script now seamlessly supports:
- ✅ Binary datasets (pre-tokenized)
- ✅ HuggingFace datasets (on-the-fly tokenization)
- ✅ Multi-GPU training (DataParallel)
- ✅ Command-line overrides
- ✅ Automatic format detection
- ✅ Checkpoint resumption

Ready to train! 🚀
