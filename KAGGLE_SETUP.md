# Kaggle Setup Guide

## Quick Start on Kaggle

### 1. Install torch_xla (for TPU)

```bash
# Kaggle doesn't have torch_xla by default
# For now, train on GPU (Kaggle provides T4 or P100)
```

### 2. Verify Dataset Download

The curriculum dataset should auto-download, but you can verify:

```bash
python scripts/verify_curriculum_dataset.py
```

Expected output:
```
================================================================================
Verifying Curriculum Dataset
================================================================================
Directory: data/balanced_300m_curriculum

✅ wikitext_train.bin        (  389.1 MB)
✅ stack_train.bin           (  145.0 MB)
✅ ultrachat_train.bin       (  793.5 MB)
✅ val.bin                   (   38.1 MB)
✅ dataset_stats.json        (    0.0 MB)
================================================================================
✅ All curriculum dataset files present!
```

### 3. Train on Kaggle GPU

```bash
# Use auto-adaptive config (detects GPU automatically)
python train.py --config config/auto_training_117m_balanced.yaml \
    --resume-remote best_model_step_7500.pt
```

## Troubleshooting

### Issue: FileNotFoundError for curriculum files

**Error**:
```
FileNotFoundError: Source file not found: data/balanced_300m_curriculum/wikitext_train.bin
```

**Cause**: Files downloaded to HuggingFace cache but not copied to data directory

**Solution 1**: Run verification script
```bash
python scripts/verify_curriculum_dataset.py
```

**Solution 2**: Manually download
```bash
python -m src.dataset_downloader \
    0x-genesys/mix_wiki_code_chat_data_300M_tokens_curriculum \
    --dataset-name balanced_300m_curriculum
```

**Solution 3**: Check if files are in HF cache
```bash
# Files might be in HuggingFace cache
ls -lh ~/.cache/huggingface/hub/datasets--0x-genesys--mix_wiki_code_chat_data_300M_tokens_curriculum/

# If found, copy them
mkdir -p data/balanced_300m_curriculum
cp ~/.cache/huggingface/hub/datasets--*/snapshots/*/wikitext_train.bin data/balanced_300m_curriculum/
cp ~/.cache/huggingface/hub/datasets--*/snapshots/*/stack_train.bin data/balanced_300m_curriculum/
cp ~/.cache/huggingface/hub/datasets--*/snapshots/*/ultrachat_train.bin data/balanced_300m_curriculum/
cp ~/.cache/huggingface/hub/datasets--*/snapshots/*/val.bin data/balanced_300m_curriculum/
cp ~/.cache/huggingface/hub/datasets--*/snapshots/*/dataset_stats.json data/balanced_300m_curriculum/
```

### Issue: TPU not available on Kaggle

**Error**:
```
⚠️  TPU requested but torch_xla not installed.
```

**Solution**: Kaggle doesn't provide TPU. Use GPU instead:

```bash
# Remove --tpu flag
python train.py --config config/auto_training_117m_balanced.yaml \
    --resume-remote best_model_step_7500.pt
```

The auto-adaptive config will detect Kaggle's GPU (T4 or P100) and optimize accordingly.

### Issue: Out of Memory on Kaggle

**Error**:
```
RuntimeError: CUDA out of memory
```

**Solution**: Kaggle GPUs have limited memory (16GB T4). Use low memory config:

```bash
python train.py --config config/gpu_training_117m_balanced_low_memory.yaml \
    --resume-remote best_model_step_7500.pt
```

Or reduce batch size:
```yaml
# Edit config
training:
  batch_size: 4  # Reduce from 8
  gradient_accumulation_steps: 32  # Increase to maintain effective batch
```

## Kaggle-Specific Settings

### GPU Configuration

Kaggle provides:
- **T4 GPU**: 16GB VRAM (most common)
- **P100 GPU**: 16GB VRAM (older)
- **2x T4**: 32GB total (rare)

### Recommended Config

For Kaggle T4 (16GB):

```yaml
# config/kaggle_training_117m.yaml
training:
  batch_size: 8
  gradient_accumulation_steps: 16
  
model:
  use_gradient_checkpointing: true  # Save memory
```

### Session Limits

- **Time limit**: 12 hours (free tier), 30 hours (paid)
- **Save checkpoints frequently**: Every 500 steps
- **Enable HuggingFace upload**: Auto-upload to resume later

```yaml
training:
  save_interval: 500  # Save every 500 steps

huggingface_hub:
  enabled: true  # Auto-upload checkpoints
  repo_id: "your-username/your-model-repo"
```

## Complete Kaggle Workflow

### 1. Setup

```bash
# Clone repository
git clone https://github.com/0x-genesys/neo.git
cd neo

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify Dataset

```bash
python scripts/verify_curriculum_dataset.py
```

### 3. Start Training

```bash
# Resume from HuggingFace Hub checkpoint
python train.py --config config/auto_training_117m_balanced.yaml \
    --resume-remote best_model_step_7500.pt
```

### 4. Monitor Progress

```bash
# In Kaggle notebook
%load_ext tensorboard
%tensorboard --logdir logs/auto_training_117m_balanced
```

### 5. Save Before Timeout

Kaggle sessions timeout after 12 hours. Checkpoints are auto-saved every 500 steps and uploaded to HuggingFace Hub.

To resume in a new session:
```bash
# Start new Kaggle session
python train.py --config config/auto_training_117m_balanced.yaml \
    --resume-remote checkpoint_step_15000.pt
```

## Performance on Kaggle

### T4 GPU (16GB)

- **Batch size**: 8
- **Gradient accumulation**: 16
- **Effective batch**: 128
- **Speed**: ~0.8 steps/sec
- **Time to 10K steps**: ~3.5 hours

### P100 GPU (16GB)

- **Batch size**: 8
- **Gradient accumulation**: 16
- **Effective batch**: 128
- **Speed**: ~0.6 steps/sec
- **Time to 10K steps**: ~4.6 hours

## Tips for Kaggle

1. **Enable GPU**: Settings → Accelerator → GPU T4 x2
2. **Save frequently**: Checkpoints every 500 steps
3. **Upload to HF Hub**: Enable automatic uploads
4. **Monitor memory**: Use `nvidia-smi` to check usage
5. **Use low memory config**: If OOM, use `gpu_training_117m_balanced_low_memory.yaml`

## See Also

- [README.md](README.md) - Main documentation
- [CROSS_HARDWARE_TRAINING.md](docs/CROSS_HARDWARE_TRAINING.md) - Cross-hardware guide
- [GPU_MEMORY_GUIDE.md](docs/GPU_MEMORY_GUIDE.md) - GPU optimization

---

**Last Updated**: 2026-04-30
