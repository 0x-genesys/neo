# Remote Model Loading Guide

This guide explains how to use the remote model loading feature to resume training or run inference with models stored on HuggingFace Hub.

## Overview

The Neo training system supports loading checkpoints and models directly from HuggingFace Hub without manual download. This enables:

- **Resume training** from checkpoints stored on HuggingFace Hub
- **Run inference** with models from HuggingFace Hub
- **Share models** easily across teams and environments
- **Version control** for model checkpoints

## Model Repository

**Default Repository**: [0x-genesys/neo_weights_checkpoints](https://huggingface.co/0x-genesys/neo_weights_checkpoints)

Available checkpoints:
- `checkpoint.pt` - Latest training checkpoint
- `best_model.pt` - Best performing model
- `checkpoint_step_500.pt` - Checkpoint at step 500
- `checkpoint_step_1000.pt` - Checkpoint at step 1000
- `checkpoint_step_1500.pt` - Checkpoint at step 1500
- ... and more

## Usage

### 1. Resume Training from Remote Checkpoint

#### Basic Usage
```bash
python train.py --config config/production_training.yaml --resume-remote checkpoint.pt
```

#### With Custom Repository
```bash
python train.py \
    --config config/production_training.yaml \
    --resume-remote checkpoint_step_4000.pt \
    --model-repo your-username/your-model-repo
```

#### Multi-GPU Training
```bash
python train.py \
    --config config/gpu_training_117m_balanced.yaml \
    --resume-remote best_model.pt \
    --multi-gpu
```

### 2. Run Inference with Remote Model

#### Basic Usage
```bash
python src/inference.py \
    --model-remote best_model.pt \
    --prompt "Once upon a time"
```

#### With Custom Repository
```bash
python src/inference.py \
    --model-remote checkpoint_step_4000.pt \
    --model-repo your-username/your-model-repo \
    --prompt "The future of AI"
```

#### Interactive Mode
```bash
python src/inference.py \
    --model-remote best_model.pt \
    --interactive
```

#### With Custom Parameters
```bash
python src/inference.py \
    --model-remote best_model.pt \
    --prompt "The future of AI" \
    --max-tokens 200 \
    --temperature 0.8 \
    --top-k 50 \
    --top-p 0.95
```

### 3. Programmatic Usage

#### Load Checkpoint in Python
```python
from src.remote_model_loader import load_remote_checkpoint

# Load from default repository
checkpoint = load_remote_checkpoint(
    filename="checkpoint.pt",
    map_location="cpu"
)

# Load from custom repository
checkpoint = load_remote_checkpoint(
    filename="checkpoint.pt",
    repo_id="your-username/your-model-repo",
    map_location="cuda"
)
```

#### Use RemoteModelLoader Class
```python
from src.remote_model_loader import RemoteModelLoader

# Initialize loader
loader = RemoteModelLoader(repo_id="0x-genesys/neo_weights_checkpoints")

# List available checkpoints
checkpoints = loader.list_available_checkpoints()
print(f"Available: {checkpoints}")

# Download checkpoint
local_path = loader.download_checkpoint("checkpoint.pt")

# Load checkpoint
checkpoint = loader.load_checkpoint("checkpoint.pt", map_location="cpu")
```

## Command Line Arguments

### Training (`train.py`)

| Argument | Description | Example |
|----------|-------------|---------|
| `--resume` | Resume from local checkpoint | `--resume checkpoints/model.pt` |
| `--resume-remote` | Resume from HuggingFace Hub | `--resume-remote checkpoint.pt` |
| `--model-repo` | HuggingFace repository ID | `--model-repo username/repo` |

**Note**: Cannot use both `--resume` and `--resume-remote` at the same time.

### Inference (`src/inference.py`)

| Argument | Description | Example |
|----------|-------------|---------|
| `--model` | Load from local path | `--model checkpoints/model.pt` |
| `--model-remote` | Load from HuggingFace Hub | `--model-remote best_model.pt` |
| `--model-repo` | HuggingFace repository ID | `--model-repo username/repo` |

**Note**: Must provide either `--model` or `--model-remote`, but not both.

## How It Works

### 1. Download Process

When you use `--resume-remote` or `--model-remote`:

1. **Check cache**: First checks if the file is already cached locally
2. **Download**: If not cached, downloads from HuggingFace Hub
3. **Cache**: Stores in `~/.cache/huggingface/hub/` for future use
4. **Load**: Loads the checkpoint into memory

### 2. Caching

Downloaded models are cached locally:
- **Location**: `~/.cache/huggingface/hub/`
- **Reuse**: Subsequent loads use cached version
- **Force download**: Use `force_download=True` in Python API

### 3. Authentication

For private repositories:
```bash
# Login to HuggingFace
huggingface-cli login

# Or set token
export HUGGING_FACE_HUB_TOKEN="your_token_here"
```

## Examples

### Example 1: Quick Start with Remote Model
```bash
# Download and run inference immediately
python src/inference.py \
    --model-remote best_model.pt \
    --prompt "Hello, world!"
```

### Example 2: Resume Training After Interruption
```bash
# Training was interrupted, resume from HuggingFace Hub
python train.py \
    --config config/production_training.yaml \
    --resume-remote checkpoint.pt
```

### Example 3: Test Different Checkpoints
```bash
# Test checkpoint at step 1000
python src/inference.py \
    --model-remote checkpoint_step_1000.pt \
    --prompt "Test prompt"

# Test checkpoint at step 2000
python src/inference.py \
    --model-remote checkpoint_step_2000.pt \
    --prompt "Test prompt"

# Compare results
```

### Example 4: Multi-GPU Training from Remote
```bash
# Resume multi-GPU training from HuggingFace Hub
python train.py \
    --config config/gpu_training_117m_balanced.yaml \
    --resume-remote checkpoint_step_3000.pt \
    --multi-gpu \
    --gpu-ids 0,1
```

## Troubleshooting

### Issue: "Repository not found"

**Solution**: Check repository ID and access permissions
```bash
# Verify repository exists
huggingface-cli repo info 0x-genesys/neo_weights_checkpoints

# Login if private
huggingface-cli login
```

### Issue: "File not found in repository"

**Solution**: List available files
```python
from src.remote_model_loader import RemoteModelLoader

loader = RemoteModelLoader(repo_id="0x-genesys/neo_weights_checkpoints")
checkpoints = loader.list_available_checkpoints()
print(f"Available: {checkpoints}")
```

### Issue: "Download failed"

**Solution**: Check network connection and retry
```bash
# Test connection
ping huggingface.co

# Force re-download
python -c "
from src.remote_model_loader import RemoteModelLoader
loader = RemoteModelLoader()
loader.download_checkpoint('checkpoint.pt', force_download=True)
"
```

### Issue: "Out of disk space"

**Solution**: Clear HuggingFace cache
```bash
# Check cache size
du -sh ~/.cache/huggingface/

# Clear cache
rm -rf ~/.cache/huggingface/hub/

# Or use HuggingFace CLI
huggingface-cli delete-cache
```

## Best Practices

### 1. Use Specific Checkpoints
```bash
# Good: Specific checkpoint
--resume-remote checkpoint_step_4000.pt

# Avoid: Generic name (may change)
--resume-remote checkpoint.pt
```

### 2. Verify Before Long Training
```bash
# Test loading first
python -c "
from src.remote_model_loader import load_remote_checkpoint
checkpoint = load_remote_checkpoint('checkpoint.pt')
print('✅ Checkpoint loaded successfully')
"

# Then start training
python train.py --config config.yaml --resume-remote checkpoint.pt
```

### 3. Use Local for Repeated Access
```bash
# Download once
python -c "
from src.remote_model_loader import get_remote_checkpoint_path
path = get_remote_checkpoint_path('checkpoint.pt')
print(f'Downloaded to: {path}')
"

# Use local path for repeated access
python train.py --config config.yaml --resume /path/to/cached/checkpoint.pt
```

### 4. Monitor Download Progress
The system automatically shows download progress:
```
📥 Downloading checkpoint from HuggingFace Hub...
   Repository: 0x-genesys/neo_weights_checkpoints
   File: checkpoint.pt
✅ Downloaded to: /home/user/.cache/huggingface/hub/...
```

## Integration with Training

### Automatic Upload

Configure automatic checkpoint upload in your config file:

```yaml
huggingface_hub:
  enabled: true
  repo_id: "your-username/your-model-repo"
  upload_best_only: false  # Upload all checkpoints
  upload_interval: 500     # Upload every 500 steps
```

### Workflow

1. **Train locally** with automatic upload enabled
2. **Checkpoints uploaded** to HuggingFace Hub during training
3. **Resume anywhere** using `--resume-remote`
4. **Share with team** via HuggingFace Hub

## Testing

Run the test suite to verify remote loading:

```bash
# Test remote loading functionality
python test/test_remote_loading.py
```

Expected output:
```
✅ List Checkpoints: PASSED
✅ Download Checkpoint: PASSED
✅ Load Checkpoint: PASSED

🎉 All tests passed!
```

## See Also

- [CHECKPOINT_UPLOAD_GUIDE.md](CHECKPOINT_UPLOAD_GUIDE.md) - Upload checkpoints to HuggingFace Hub
- [README.md](../README.md) - Main documentation
- [ARCHITECTURE.md](../ARCHITECTURE.md) - System architecture
- [HuggingFace Hub Documentation](https://huggingface.co/docs/hub/index)
