# Quick Setup Guide: From Error to Working Training

This guide helps you get from the "Repository Not Found" error to a working training setup in minutes.

## 🚨 Current Issue

You're seeing this error:
```
❌ Error downloading dataset: 404 Client Error
Repository Not Found for url: https://huggingface.co/api/models/0x-genesys/mix_wiki_code_chat_data_300M_tokens
```

**Cause**: The repository `0x-genesys/mix_wiki_code_chat_data_300M_tokens` doesn't exist on HuggingFace Hub yet.

## 🚀 Immediate Solution (2 minutes)

### Option 1: Test with WikiText (Recommended for immediate testing)

```bash
# This works immediately - no setup required
python train.py --config config/gpu_training_117m_wikitext.yaml
```

This uses the standard WikiText-2 dataset that's already available on HuggingFace Hub.

### Option 2: Use Your Local Dataset

If you have a local dataset in `data/balanced_300m/`:

```bash
# Update config to disable auto-download
# Edit config/gpu_training_117m_balanced.yaml and set:
# auto_download: false

python train.py --config config/gpu_training_117m_balanced.yaml
```

## 🔧 Complete Setup (10 minutes)

### Step 1: Check Your Setup

```bash
# Run the interactive setup helper
python scripts/setup_dataset_repo.py
```

This script will:
- Check if you're authenticated with HuggingFace
- Find your existing datasets (local and remote)
- Guide you through the setup process

### Step 2: Set HuggingFace Token (if needed)

```bash
# Get your token from https://huggingface.co/settings/tokens
export HF_TOKEN=your_token_here

# Or add to your shell profile for persistence
echo 'export HF_TOKEN=your_token_here' >> ~/.bashrc
source ~/.bashrc
```

### Step 3: Choose Your Path

#### Path A: Upload Existing Local Dataset

If you have `data/balanced_300m/` locally:

```bash
# Upload to HuggingFace Hub (replace 'your-username' with your actual username)
python scripts/upload_dataset_to_hf.py data/balanced_300m your-username/balanced-300m-tokens

# The script will automatically update your configs
```

#### Path B: Build and Upload New Dataset

```bash
# 1. Build dataset locally (2-6 hours)
python scripts/prepare_balanced_dataset.py

# 2. Upload to HuggingFace Hub
python scripts/upload_dataset_to_hf.py data/balanced_300m your-username/balanced-300m-tokens
```

#### Path C: Use Different Repository

If you already have a dataset repository on HuggingFace:

```bash
# Update configs manually or use the setup script
python scripts/setup_dataset_repo.py
```

## 🧪 Test Your Setup

```bash
# Test the download system
python scripts/test_dataset_download.py

# Test training (short run)
python train.py --config config/gpu_training_117m_balanced.yaml --max-steps 10
```

## 📋 Troubleshooting Checklist

### ✅ Authentication Issues
```bash
# Check if token is set
echo $HF_TOKEN

# Test authentication
python -c "from huggingface_hub import HfApi; print(HfApi().whoami())"
```

### ✅ Repository Issues
```bash
# Check if repository exists
python -c "from huggingface_hub import list_repo_files; print(list_repo_files('your-username/your-repo', repo_type='dataset'))"
```

### ✅ Local Dataset Issues
```bash
# Check local datasets
ls -la data/*/
ls -la data/*/train.bin
```

### ✅ Config Issues
```bash
# Validate config syntax
python -c "import yaml; yaml.safe_load(open('config/gpu_training_117m_balanced.yaml'))"
```

## 🎯 Recommended Workflow

### For Immediate Testing (5 minutes)
1. Use WikiText config: `python train.py --config config/gpu_training_117m_wikitext.yaml`
2. Verify training works on your system
3. Set up custom datasets later

### For Production Setup (30 minutes)
1. Set HuggingFace token: `export HF_TOKEN=...`
2. Run setup helper: `python scripts/setup_dataset_repo.py`
3. Follow interactive prompts
4. Test: `python scripts/test_dataset_download.py`
5. Start training: `python train.py --config config/gpu_training_117m_balanced.yaml`

## 🔄 Migration from Old System

### Old Way (Problematic)
```bash
# This often fails and takes hours
python scripts/prepare_balanced_dataset.py  # 2-6 hours, often fails
python train.py --config some_config.yaml
```

### New Way (Reliable)
```bash
# This works reliably and takes minutes
python scripts/setup_dataset_repo.py       # 5 minutes, interactive
python train.py --config config/gpu_training_117m_balanced.yaml  # Starts immediately
```

## 📞 Getting Help

### Check Status
```bash
# System status
python scripts/test_dataset_download.py

# Config status
python scripts/setup_dataset_repo.py
```

### Common Solutions

**Problem**: "Repository Not Found"
- **Solution**: Run `python scripts/setup_dataset_repo.py` to set up correct repository

**Problem**: "Authentication failed"
- **Solution**: Set HuggingFace token: `export HF_TOKEN=your_token`

**Problem**: "train.bin not found"
- **Solution**: Either upload dataset or use WikiText config for testing

**Problem**: "tqdm creates multiple lines"
- **Solution**: Set `export DISABLE_TQDM=1` or use better terminal

## 🎉 Success Indicators

You'll know it's working when you see:

```bash
✅ Dataset files confirmed:
  Train: data/balanced_300m/train.bin
  Val: data/balanced_300m/val.bin
✅ Model created: 117.23M parameters
✅ Training started...
Step 1/36621 | Loss: 10.234 | LR: 1.33e-06 | Time: 2.1s
```

## 🔗 Quick Links

- **Immediate testing**: `config/gpu_training_117m_wikitext.yaml`
- **Setup helper**: `python scripts/setup_dataset_repo.py`
- **Test system**: `python scripts/test_dataset_download.py`
- **Upload dataset**: `python scripts/upload_dataset_to_hf.py`
- **HuggingFace tokens**: https://huggingface.co/settings/tokens

The key is to start with what works (WikiText config) and then gradually set up your custom datasets!