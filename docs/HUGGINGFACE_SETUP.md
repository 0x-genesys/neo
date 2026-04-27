# 🤗 HuggingFace Setup Guide

Complete guide for setting up HuggingFace Hub access and using datasets with your token.

## 📝 Getting Your HuggingFace Token

### 1. Create a HuggingFace Account

Visit [https://huggingface.co/join](https://huggingface.co/join) and create a free account.

### 2. Generate an Access Token

1. Go to [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Click **"New token"**
3. Give it a name (e.g., "transformer-training")
4. Select token type:
   - **Read**: For downloading public datasets and models
   - **Write**: If you want to upload models (optional)
5. Click **"Generate token"**
6. **Copy the token** (you won't see it again!)

## 🔐 Setting Up Your Token

### Method 1: Using HuggingFace CLI (Recommended)

```bash
# Install HuggingFace CLI (if not already installed)
pip install huggingface-hub

# Login with your token
huggingface-cli login

# Paste your token when prompted
# Choose 'y' to add token as git credential (optional)
```

**Verification**:
```bash
huggingface-cli whoami
```

### Method 2: Using Python

```python
from huggingface_hub import login

# Login with your token
login(token="hf_your_token_here")

# Or login interactively
login()
```

### Method 3: Environment Variable

```bash
# Add to your ~/.bashrc or ~/.zshrc
export HUGGING_FACE_HUB_TOKEN="hf_your_token_here"

# Or set for current session
export HUGGING_FACE_HUB_TOKEN="hf_your_token_here"
```

### Method 4: In Code (Not Recommended for Production)

```python
# In your training script
import os
os.environ['HUGGING_FACE_HUB_TOKEN'] = 'hf_your_token_here'
```

⚠️ **Security Warning**: Never commit tokens to git! Add them to `.gitignore`.

## 📚 Accessing Datasets

### Public Datasets (No Token Required)

Most datasets are public and don't require authentication:

```python
from datasets import load_dataset

# These work without a token
dataset = load_dataset('tiny_shakespeare')
dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')
dataset = load_dataset('openwebtext')
```

### Gated Datasets (Token Required)

Some datasets require accepting terms and using a token:

```python
from datasets import load_dataset

# Login first
from huggingface_hub import login
login(token="your_token")

# Then load gated dataset
dataset = load_dataset('meta-llama/Llama-2-7b')  # Example
```

### Private Datasets

If you have private datasets:

```python
# Must be logged in
dataset = load_dataset('your-username/private-dataset', use_auth_token=True)
```

## 🎯 Using Token in Training

### Option 1: Login Before Training

```bash
# Login once
huggingface-cli login

# Then run training normally
python train.py --config config/model_config.yaml
```

### Option 2: Pass Token in Code

Update `src/data.py`:

```python
def load_data(config):
    # Load tokenizer with token
    tokenizer = AutoTokenizer.from_pretrained(
        config['tokenizer']['type'],
        use_auth_token=True  # Uses cached token
    )
    
    # Load dataset with token
    dataset = load_dataset(
        config['data']['dataset_name'],
        use_auth_token=True
    )
```

### Option 3: Environment Variable

```bash
# Set token
export HUGGING_FACE_HUB_TOKEN="hf_your_token_here"

# Run training
python train.py --config config/model_config.yaml
```

## 📊 Recommended Datasets

### For Quick Testing (No Token Needed)

```python
# Tiny Shakespeare - 1MB
dataset = load_dataset('tiny_shakespeare')

# WikiText-2 - 4MB
dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')
```

### For Serious Training (No Token Needed)

```python
# WikiText-103 - 500MB
dataset = load_dataset('wikitext', 'wikitext-103-raw-v1')

# OpenWebText - 40GB
dataset = load_dataset('openwebtext')

# BookCorpus - 5GB
dataset = load_dataset('bookcorpus')
```

### For Large-Scale Training (May Need Token)

```python
# C4 - 750GB (streaming recommended)
dataset = load_dataset('c4', 'en', streaming=True)

# The Pile - 825GB (streaming recommended)
dataset = load_dataset('EleutherAI/pile', streaming=True)

# RedPajama - 1.2TB (streaming recommended)
dataset = load_dataset('togethercomputer/RedPajama-Data-1T', streaming=True)
```

## 🚀 Quick Start with Your Token

### 1. Setup

```bash
# Install dependencies
pip install huggingface-hub transformers datasets

# Login
huggingface-cli login
# Paste your token
```

### 2. Test Access

```python
# test_hf_access.py
from datasets import load_dataset
from transformers import AutoTokenizer

print("Testing HuggingFace access...")

# Test dataset
print("\n1. Loading dataset...")
dataset = load_dataset('tiny_shakespeare')
print(f"✅ Dataset loaded: {len(dataset['train'])} examples")

# Test tokenizer
print("\n2. Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained('gpt2')
print(f"✅ Tokenizer loaded: {len(tokenizer)} tokens")

print("\n✅ All tests passed! You're ready to train.")
```

Run:
```bash
python test_hf_access.py
```

### 3. Start Training

```bash
# Train on tiny_shakespeare (no token needed)
python train.py --config config/quick_start.yaml

# Train on larger dataset
python train.py --config config/model_config.yaml --dataset wikitext
```

## 🔧 Troubleshooting

### "401 Unauthorized" Error

```bash
# Your token is invalid or expired
# Generate a new token and login again
huggingface-cli login
```

### "403 Forbidden" Error

```bash
# You don't have access to this dataset
# Check if dataset is gated and accept terms on HuggingFace website
```

### "Dataset not found" Error

```bash
# Check dataset name spelling
# Visit https://huggingface.co/datasets to find correct name
```

### Token Not Being Used

```python
# Explicitly pass token
from huggingface_hub import HfFolder

token = HfFolder.get_token()
print(f"Current token: {token[:10]}..." if token else "No token found")

# If no token, login again
from huggingface_hub import login
login()
```

### Cached Datasets

```bash
# HuggingFace caches datasets in ~/.cache/huggingface/

# Clear cache if needed
rm -rf ~/.cache/huggingface/datasets/

# Or use Python
from datasets import load_dataset
dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', download_mode='force_redownload')
```

## 📦 Dataset Streaming (For Large Datasets)

For datasets too large to download:

```python
from datasets import load_dataset

# Stream instead of download
dataset = load_dataset('c4', 'en', streaming=True)

# Iterate without downloading everything
for example in dataset['train']:
    print(example['text'])
    break  # Just show first example
```

Update `src/data.py` for streaming:

```python
def load_data(config):
    # Enable streaming for large datasets
    use_streaming = config['data'].get('streaming', False)
    
    dataset = load_dataset(
        config['data']['dataset_name'],
        streaming=use_streaming
    )
    
    if use_streaming:
        # Take subset for training
        train_dataset = dataset['train'].take(100000)
    else:
        train_dataset = dataset['train']
```

## 🎓 Best Practices

### 1. Security
- ✅ Never commit tokens to git
- ✅ Use environment variables or CLI login
- ✅ Regenerate tokens if exposed
- ✅ Use read-only tokens when possible

### 2. Dataset Selection
- 🚀 Start small: `tiny_shakespeare` → `wikitext-2` → `wikitext-103`
- 📊 Use streaming for datasets > 10GB
- 💾 Check available disk space before downloading
- 🔄 Cache datasets to avoid re-downloading

### 3. Token Management
- 📝 Keep tokens in password manager
- 🔄 Rotate tokens periodically
- 🗑️ Delete unused tokens
- 👥 Use separate tokens for different projects

## 📚 Additional Resources

- [HuggingFace Datasets Documentation](https://huggingface.co/docs/datasets)
- [HuggingFace Hub Documentation](https://huggingface.co/docs/huggingface_hub)
- [Available Datasets](https://huggingface.co/datasets)
- [Dataset Viewer](https://huggingface.co/datasets/viewer)

## 🆘 Getting Help

1. **HuggingFace Forum**: [https://discuss.huggingface.co/](https://discuss.huggingface.co/)
2. **Discord**: [https://discord.gg/hugging-face](https://discord.gg/hugging-face)
3. **GitHub Issues**: For dataset-specific issues

## ✅ Verification Checklist

Before training:
- [ ] HuggingFace account created
- [ ] Access token generated
- [ ] Token configured (CLI, env var, or code)
- [ ] Test script runs successfully
- [ ] Dataset accessible
- [ ] Tokenizer loads correctly

You're now ready to train! 🚀
