# Dataset Download Guide

This guide explains how to use the new automatic dataset download system that fetches pre-processed datasets from HuggingFace Hub instead of building them locally.

## 🚀 Quick Start

### 1. Use Pre-configured Setup

The easiest way is to use the new balanced dataset config:

```bash
# This will automatically download the 300M token dataset if missing
python train.py --config config/gpu_training_117m_balanced.yaml
```

### 2. What Happens Automatically

When you start training:

1. **Check Local Files**: System checks if `data/balanced_300m/train.bin` exists
2. **Auto-Download**: If missing, downloads from HuggingFace Hub automatically
3. **Progress Display**: Shows download progress with proper progress bars
4. **Start Training**: Begins training immediately after download

## 📁 Dataset Repository

The pre-processed datasets are hosted on HuggingFace Hub:

- **Repository**: `0x-genesys/mix_wiki_code_chat_data_300M_tokens`
- **URL**: https://huggingface.co/datasets/0x-genesys/mix_wiki_code_chat_data_300M_tokens
- **Format**: Binary `.bin` files with `dataset_stats.json`

### Dataset Composition (300M tokens)

- **WikiText-103**: 102M tokens (34%) - Encyclopedic knowledge
- **UltraChat**: 198M tokens (66%) - Conversational AI
- **The Stack**: 48M tokens (16%) - Code reasoning (Python/Java)

## ⚙️ Configuration

### Basic Configuration

Add this to your YAML config to enable auto-download:

```yaml
data:
  dataset_type: "binary"
  train_file: "data/balanced_300m/train.bin"
  val_file: "data/balanced_300m/val.bin"
  
  # HuggingFace Auto-Download
  huggingface_dataset:
    repo_id: "0x-genesys/mix_wiki_code_chat_data_300M_tokens"
    dataset_name: "balanced_300m"  # Local folder name
    auto_download: true  # Enable automatic download
```

### Advanced Configuration

```yaml
data:
  # ... other data config ...
  
  huggingface_dataset:
    repo_id: "0x-genesys/mix_wiki_code_chat_data_300M_tokens"
    dataset_name: "balanced_300m"
    auto_download: true
    force_download: false  # Set true to re-download existing files
```

## 🔧 Manual Download

If you prefer to download manually:

```bash
# Download using the built-in script
python -m src.dataset_downloader 0x-genesys/mix_wiki_code_chat_data_300M_tokens --dataset-name balanced_300m

# Or use the test script
python scripts/test_dataset_download.py
```

## 📊 Available Dataset Sizes

You can create configs for different dataset sizes by changing the `repo_id`:

### 117M Tokens (Smaller, Faster)
```yaml
huggingface_dataset:
  repo_id: "0x-genesys/mix_wiki_code_chat_data_117M_tokens"
  dataset_name: "balanced_117m"
```

### 1B Tokens (Larger, Better Quality)
```yaml
huggingface_dataset:
  repo_id: "0x-genesys/mix_wiki_code_chat_data_1B_tokens"
  dataset_name: "balanced_1b"
```

## 🐛 Troubleshooting

### Download Issues

**Problem**: Download fails or times out
```bash
# Check internet connection and HuggingFace access
curl -I https://huggingface.co/datasets/0x-genesys/mix_wiki_code_chat_data_300M_tokens

# Try manual download
python -m src.dataset_downloader 0x-genesys/mix_wiki_code_chat_data_300M_tokens
```

**Problem**: Repository not found
- Verify the repository exists on HuggingFace Hub
- Check if you have access permissions
- Ensure the `repo_id` is correct in your config

### Progress Bar Issues

**Problem**: tqdm creates multiple lines instead of updating
```bash
# Option 1: Disable tqdm completely
export DISABLE_TQDM=1
python train.py --config config/gpu_training_117m_balanced.yaml

# Option 2: Use different terminal/IDE with better tqdm support
```

**Problem**: Progress bars look messy in Google Colab
```python
# In Colab, add this at the start of your notebook
import os
os.environ['DISABLE_TQDM'] = '1'
```

### File Permission Issues

**Problem**: Cannot write to data directory
```bash
# Create directory with proper permissions
mkdir -p data
chmod 755 data

# Or use a different cache directory
python train.py --config config/gpu_training_117m_balanced.yaml --data-dir /tmp/datasets
```

### Memory Issues During Download

**Problem**: Out of memory during download
```bash
# The download uses streaming, but if you still have issues:
# 1. Close other applications
# 2. Use a machine with more RAM
# 3. Download to a different drive with more space
```

## 🔄 Migration from Local Dataset Building

### Old Way (Slow)
```bash
# This takes hours and often fails
python scripts/prepare_balanced_dataset.py
python train.py --config config/gpu_training_117m.yaml
```

### New Way (Fast)
```bash
# This downloads in minutes and always works
python train.py --config config/gpu_training_117m_balanced.yaml
```

### Converting Existing Configs

To convert an existing config to use auto-download:

1. **Change dataset type** to `binary`
2. **Add HuggingFace config** section
3. **Update file paths** to match the downloaded structure

Example conversion:
```yaml
# OLD
data:
  dataset_name: "wikitext"
  dataset_config: "wikitext-2-raw-v1"

# NEW  
data:
  dataset_type: "binary"
  train_file: "data/balanced_300m/train.bin"
  val_file: "data/balanced_300m/val.bin"
  huggingface_dataset:
    repo_id: "0x-genesys/mix_wiki_code_chat_data_300M_tokens"
    dataset_name: "balanced_300m"
    auto_download: true
```

## 📈 Performance Comparison

| Method | Setup Time | Dataset Quality | Reliability |
|--------|------------|-----------------|-------------|
| Local Building | 2-6 hours | Variable | Often fails |
| HuggingFace Download | 2-10 minutes | Consistent | Always works |

## 🧪 Testing

Test the download system:

```bash
# Test configuration loading
python scripts/test_dataset_download.py

# Test manual download
python -m src.dataset_downloader 0x-genesys/mix_wiki_code_chat_data_300M_tokens --dataset-name test_download

# Test with training (dry run)
python train.py --config config/gpu_training_117m_balanced.yaml --max-steps 10
```

## 🔗 Related Files

- **Downloader**: `src/dataset_downloader.py` - Core download functionality
- **Data Loader**: `src/data.py` - Updated to use auto-download
- **Config**: `config/gpu_training_117m_balanced.yaml` - Example configuration
- **Test Script**: `scripts/test_dataset_download.py` - Test the system
- **Fixed Script**: `scripts/prepare_balanced_dataset.py` - Fixed tqdm issues

## 💡 Tips

1. **First Run**: The first training run will download the dataset (2-10 minutes)
2. **Subsequent Runs**: Training starts immediately using cached files
3. **Multiple Configs**: You can have different configs using different datasets
4. **Disk Space**: Ensure you have ~2GB free space for the 300M token dataset
5. **Network**: Stable internet connection recommended for initial download

## 🎯 Next Steps

1. **Try the new system**: Use `config/gpu_training_117m_balanced.yaml`
2. **Upload your datasets**: Create your own HuggingFace dataset repositories
3. **Create variants**: Make configs for different dataset sizes
4. **Share configs**: Share working configurations with your team