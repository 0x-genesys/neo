# Checkpoint Upload Guide

Upload your trained model checkpoints to HuggingFace Hub for sharing and deployment.

## 🚀 Quick Start

### Interactive Mode (Recommended)
```bash
# Run interactive upload wizard
python scripts/test_checkpoint_upload.py --test
```

The wizard will:
1. Check your HF authentication
2. Find available checkpoints
3. Let you select which one to upload
4. Guide you through the upload process

### Direct Upload
```bash
# Upload specific checkpoint
python scripts/test_checkpoint_upload.py \
    checkpoints/production/checkpoint.pt \
    your-username/your-model-name
```

## 📋 Prerequisites

### 1. HuggingFace Token
```bash
# Get token from https://huggingface.co/settings/tokens
export HF_TOKEN=your_token_here

# Or add to your shell profile
echo 'export HF_TOKEN=your_token_here' >> ~/.bashrc
source ~/.bashrc
```

### 2. Trained Checkpoint
You need a checkpoint file (`.pt`) from training:
- `checkpoints/production/checkpoint.pt`
- `checkpoints/production/best_model.pt`
- `checkpoints/quick_start/checkpoint.pt`

## 🎯 Usage Examples

### Example 1: Test Upload
```bash
# Interactive mode - easiest
python scripts/test_checkpoint_upload.py --test
```

Output:
```
🧪 Checkpoint Upload Test
============================================================
✅ Authenticated as: your-username

📁 Looking for checkpoints...
✅ Found 2 checkpoint(s):
   1. checkpoints/production/checkpoint.pt (450.2 MB)
   2. checkpoints/production/best_model.pt (450.2 MB)

Select checkpoint (1-2): 1

📋 Checking checkpoint: checkpoints/production/checkpoint.pt
✅ Checkpoint is valid
   File size: 450.2 MB
   Keys: ['model_state_dict', 'optimizer_state_dict', 'epoch', 'global_step']
   Epoch: 5
   Step: 2500
   Best val loss: 3.2456

💡 Suggested repository: your-username/transformer-checkpoint-test
Enter repository ID (or press Enter for default): 

📤 Ready to upload:
   Checkpoint: checkpoints/production/checkpoint.pt
   Repository: your-username/transformer-checkpoint-test

Proceed with upload? (y/N): y

📤 Uploading checkpoint to HuggingFace Hub
============================================================
Local file: checkpoints/production/checkpoint.pt
Repository: your-username/transformer-checkpoint-test
Private: False

📋 Creating/verifying repository...
✅ Repository ready: your-username/transformer-checkpoint-test

📤 Uploading checkpoint.pt...
   Size: 450.2 MB

✅ Upload complete!
🔗 Repository: https://huggingface.co/your-username/transformer-checkpoint-test
🔗 File: https://huggingface.co/your-username/transformer-checkpoint-test/blob/main/checkpoint.pt

💡 To download this checkpoint:
   from huggingface_hub import hf_hub_download
   checkpoint = hf_hub_download(
       repo_id='your-username/transformer-checkpoint-test',
       filename='checkpoint.pt'
   )
```

### Example 2: Direct Upload
```bash
# Upload production checkpoint
python scripts/test_checkpoint_upload.py \
    checkpoints/production/checkpoint.pt \
    your-username/my-transformer-model
```

### Example 3: Private Repository
```bash
# Upload to private repo
python scripts/test_checkpoint_upload.py \
    checkpoints/production/best_model.pt \
    your-username/my-private-model \
    --private
```

### Example 4: Custom Commit Message
```bash
# Upload with custom message
python scripts/test_checkpoint_upload.py \
    checkpoints/production/checkpoint.pt \
    your-username/my-model \
    --message "Checkpoint after 10 epochs, loss=3.24"
```

## 📥 Downloading Checkpoints

After uploading, others can download your checkpoint:

```python
from huggingface_hub import hf_hub_download
import torch

# Download checkpoint
checkpoint_path = hf_hub_download(
    repo_id='your-username/your-model-name',
    filename='checkpoint.pt'
)

# Load checkpoint
checkpoint = torch.load(checkpoint_path, map_location='cpu')

# Use checkpoint
model.load_state_dict(checkpoint['model_state_dict'])
```

## 🔧 Advanced Usage

### Upload Multiple Checkpoints
```bash
# Upload all checkpoints from a directory
for checkpoint in checkpoints/production/*.pt; do
    python scripts/test_checkpoint_upload.py \
        "$checkpoint" \
        your-username/my-model
done
```

### Automated Upload in Training
Add to your training config:

```yaml
huggingface_hub:
  enabled: true
  repo_id: "your-username/your-model"
  upload_best_only: false  # Upload all checkpoints
```

The trainer will automatically upload checkpoints during training.

## 📊 What Gets Uploaded

The checkpoint file (`.pt`) contains:
- Model weights (`model_state_dict`)
- Optimizer state (`optimizer_state_dict`)
- Training metadata (epoch, step, loss)
- Configuration (if saved)

## 🔒 Public vs Private

### Public Repository
- Anyone can download
- Good for sharing models
- Appears in your HF profile

```bash
python scripts/test_checkpoint_upload.py checkpoint.pt username/model
```

### Private Repository
- Only you can access
- Good for work-in-progress
- Not visible publicly

```bash
python scripts/test_checkpoint_upload.py checkpoint.pt username/model --private
```

## 🐛 Troubleshooting

### "Token not found"
```bash
# Set your HuggingFace token
export HF_TOKEN=your_token_here

# Verify it's set
echo $HF_TOKEN
```

### "Checkpoint not found"
```bash
# Check available checkpoints
ls -lh checkpoints/*/

# Train a model first if none exist
python train.py --config config/quick_start.yaml
```

### "Authentication failed"
```bash
# Verify token is valid
python -c "from huggingface_hub import HfApi; print(HfApi().whoami())"

# Get new token from https://huggingface.co/settings/tokens
```

### "Upload failed"
- Check internet connection
- Verify token has write permissions
- Ensure repository name is valid (lowercase, hyphens only)

## 💡 Best Practices

1. **Use descriptive names**: `username/gpt2-wikitext-117m` not `username/model1`
2. **Upload best models**: Use `best_model.pt` for production
3. **Add README**: Create README.md in your HF repo with model details
4. **Version your models**: Use different repos or tags for versions
5. **Test downloads**: Verify uploaded checkpoints can be downloaded

## 🎯 Common Workflows

### Workflow 1: Quick Test
```bash
# 1. Train quick model
python train.py --config config/quick_start.yaml

# 2. Upload checkpoint
python scripts/test_checkpoint_upload.py --test

# 3. Share the HF link with others
```

### Workflow 2: Production Model
```bash
# 1. Train production model
python train.py --config config/production_training.yaml

# 2. Upload best checkpoint
python scripts/test_checkpoint_upload.py \
    checkpoints/production/best_model.pt \
    username/production-model

# 3. Add model card to HF repo
```

### Workflow 3: Continuous Upload
```bash
# Enable automatic upload in config
# Edit config/production_training.yaml:
huggingface_hub:
  enabled: true
  repo_id: "username/my-model"

# Start training - checkpoints upload automatically
python train.py --config config/production_training.yaml
```

## 📚 Related Documentation

- **HuggingFace Hub**: https://huggingface.co/docs/hub
- **Model Cards**: https://huggingface.co/docs/hub/model-cards
- **Tokens**: https://huggingface.co/settings/tokens

## 🎉 You're Ready!

Upload your first checkpoint:
```bash
python scripts/test_checkpoint_upload.py --test
```