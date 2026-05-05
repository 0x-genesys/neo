# Quick Commands Reference

## Training on Kaggle TPU

```bash
# Start TPU training
python train.py \
    --config config/auto_training_117m_balanced.yaml \
    --tpu

# Resume from checkpoint
python train.py \
    --config config/auto_training_117m_balanced.yaml \
    --tpu \
    --resume checkpoints/auto_training_117m_balanced/checkpoint_step_5000.pt

# Resume from HuggingFace Hub
python train.py \
    --config config/auto_training_117m_balanced.yaml \
    --tpu \
    --resume-remote checkpoint_step_5000.pt \
    --model-repo 0x-genesys/neo_weights_checkpoints
```

## Checkpoint Management

### Check Disk Usage

```bash
# Total disk usage
df -h

# Checkpoint directory size
du -sh checkpoints/*/

# List checkpoints with sizes
ls -lh checkpoints/auto_training_117m_balanced/*.pt
```

### Cleanup Checkpoints

```bash
# Dry run (see what would be deleted)
python scripts/cleanup_checkpoints.py \
    --checkpoint-dir checkpoints/auto_training_117m_balanced \
    --dry-run

# Delete checkpoints (keeps best models)
python scripts/cleanup_checkpoints.py \
    --checkpoint-dir checkpoints/auto_training_117m_balanced

# Verify on Hub before deleting (safest)
python scripts/cleanup_checkpoints.py \
    --checkpoint-dir checkpoints/auto_training_117m_balanced \
    --verify-hub \
    --repo-id 0x-genesys/neo_weights_checkpoints
```

## Monitoring Training

### Check TPU Status

```bash
# Check if TPU is available
python scripts/check_tpu.py

# Monitor TPU usage (if available)
watch -n 1 'python -c "import torch_xla.core.xla_model as xm; print(xm.get_memory_info(xm.xla_device()))"'
```

### View Logs

```bash
# TensorBoard
tensorboard --logdir logs/

# Follow training output
tail -f training.log
```

## HuggingFace Hub

### Upload Checkpoint

```bash
# Upload specific checkpoint
huggingface-cli upload 0x-genesys/neo_weights_checkpoints \
    checkpoints/auto_training_117m_balanced/best_model_step_10000.pt

# Upload entire directory
huggingface-cli upload 0x-genesys/neo_weights_checkpoints \
    checkpoints/auto_training_117m_balanced/ \
    --repo-type model
```

### Download Checkpoint

```bash
# Download specific file
huggingface-cli download 0x-genesys/neo_weights_checkpoints \
    checkpoint_step_5000.pt \
    --local-dir ./checkpoints/downloaded/

# Download all checkpoints
huggingface-cli download 0x-genesys/neo_weights_checkpoints \
    --repo-type model \
    --local-dir ./checkpoints/downloaded/
```

## Troubleshooting

### TPU Not Detected

```bash
# Check torch_xla installation
python -c "import torch_xla; print(torch_xla.__version__)"

# Check TPU devices
python -c "import torch_xla.core.xla_model as xm; print(xm.get_xla_supported_devices())"

# Reinstall torch_xla if needed
pip install torch_xla -f https://storage.googleapis.com/tpu-pytorch/wheels/tpuvm/torch_xla-2.5.0-cp312-cp312-linux_x86_64.whl
```

### Out of Disk Space

```bash
# Find large files
du -h checkpoints/ | sort -rh | head -20

# Clean up checkpoints
python scripts/cleanup_checkpoints.py \
    --checkpoint-dir checkpoints/auto_training_117m_balanced \
    --verify-hub \
    --repo-id 0x-genesys/neo_weights_checkpoints

# Remove old logs
rm -rf logs/old_runs/
```

### Training Interrupted

```bash
# Resume from interrupted checkpoint
python train.py \
    --config config/auto_training_117m_balanced.yaml \
    --tpu \
    --resume checkpoints/auto_training_117m_balanced/interrupted_checkpoint.pt

# Or from error checkpoint
python train.py \
    --config config/auto_training_117m_balanced.yaml \
    --tpu \
    --resume checkpoints/auto_training_117m_balanced/error_checkpoint.pt
```

## Configuration Adjustments

### Reduce Memory Usage

```bash
# Use smaller batch size
python train.py \
    --config config/auto_training_117m_balanced.yaml \
    --tpu \
    --batch-size 16

# Use gradient accumulation
# Edit config: gradient_accumulation_steps: 4
```

### Adjust Save Frequency

```bash
# Edit config file
vim config/auto_training_117m_balanced.yaml

# Change:
checkpoint:
  save_interval: 2000  # Save every 2000 steps instead of 1000
```

### Change Learning Rate

```bash
# Override from command line
python train.py \
    --config config/auto_training_117m_balanced.yaml \
    --tpu \
    --lr 3e-4
```

## Quick Checks

### Before Training

```bash
# 1. Check disk space
df -h

# 2. Check TPU availability
python scripts/check_tpu.py

# 3. Verify config
cat config/auto_training_117m_balanced.yaml

# 4. Check HuggingFace token
echo $HF_TOKEN
```

### During Training

```bash
# 1. Monitor disk usage
watch -n 60 'df -h'

# 2. Check checkpoint sizes
watch -n 300 'ls -lh checkpoints/auto_training_117m_balanced/*.pt'

# 3. Clean up if needed
python scripts/cleanup_checkpoints.py \
    --checkpoint-dir checkpoints/auto_training_117m_balanced \
    --verify-hub \
    --repo-id 0x-genesys/neo_weights_checkpoints
```

### After Training

```bash
# 1. Upload best model
huggingface-cli upload 0x-genesys/neo_weights_checkpoints \
    checkpoints/auto_training_117m_balanced/best_model_*.pt

# 2. Clean up all regular checkpoints
python scripts/cleanup_checkpoints.py \
    --checkpoint-dir checkpoints/auto_training_117m_balanced \
    --verify-hub \
    --repo-id 0x-genesys/neo_weights_checkpoints

# 3. Verify final disk usage
du -sh checkpoints/auto_training_117m_balanced/
```

## One-Liners

```bash
# Start training with cleanup every hour
while true; do \
    python scripts/cleanup_checkpoints.py \
        --checkpoint-dir checkpoints/auto_training_117m_balanced \
        --verify-hub \
        --repo-id 0x-genesys/neo_weights_checkpoints; \
    sleep 3600; \
done &

# Monitor training progress
watch -n 10 'tail -20 logs/training.log'

# Check latest checkpoint
ls -lt checkpoints/auto_training_117m_balanced/*.pt | head -1

# Total checkpoint size
du -sh checkpoints/auto_training_117m_balanced/
```
