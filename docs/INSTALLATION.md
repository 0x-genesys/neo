# Installation Guide

## ⚠️ Important: Python Version Compatibility

**Current Issue**: PyTorch does not yet support Python 3.14.4 (your current version).

### Solution Options

#### Option 1: Use Python 3.11 or 3.12 (Recommended)

```bash
# Install Python 3.11 or 3.12 using Homebrew (Mac)
brew install python@3.11

# Create venv with Python 3.11
python3.11 -m venv venv

# Activate and install dependencies
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

#### Option 2: Use Conda/Miniconda

```bash
# Install Miniconda from https://docs.conda.io/en/latest/miniconda.html

# Create environment with Python 3.11
conda create -n transformer python=3.11
conda activate transformer

# Install PyTorch
conda install pytorch torchvision torchaudio -c pytorch

# Install other dependencies
pip install transformers datasets tokenizers sentencepiece
pip install tqdm pyyaml tensorboard wandb huggingface-hub
pip install numpy pandas scikit-learn matplotlib seaborn
```

#### Option 3: Use pyenv

```bash
# Install pyenv
brew install pyenv

# Install Python 3.11
pyenv install 3.11.7

# Set local Python version
pyenv local 3.11.7

# Create venv
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 📦 Full Installation Steps

### 1. Setup Python Environment

Choose one of the options above to get Python 3.11 or 3.12.

### 2. Create Virtual Environment

```bash
# Using venv
python3.11 -m venv venv
source venv/bin/activate

# Or using conda
conda create -n transformer python=3.11
conda activate transformer
```

### 3. Install PyTorch

**For Mac (Apple Silicon M1/M2/M3)**:
```bash
pip install torch torchvision torchaudio
```

**For Mac (Intel)**:
```bash
pip install torch torchvision torchaudio
```

**For Linux with CUDA**:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**For CPU only**:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 4. Install Other Dependencies

```bash
pip install transformers datasets tokenizers sentencepiece
pip install tqdm pyyaml tensorboard wandb huggingface-hub
pip install numpy pandas scikit-learn matplotlib seaborn
```

Or install all at once:
```bash
pip install -r requirements.txt
```

### 5. Verify Installation

```bash
# Test PyTorch installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"

# Test device detection
python src/device_utils.py
```

## 🖥️ Device-Specific Setup

### Apple Silicon (M1/M2/M3)

```bash
# PyTorch with MPS support
pip install torch torchvision torchaudio

# Verify MPS is available
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

**Expected output**: `MPS available: True`

### NVIDIA GPU (CUDA)

```bash
# Check CUDA version
nvidia-smi

# Install PyTorch with matching CUDA version
# For CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify CUDA is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### CPU Only

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## 🔧 Troubleshooting

### "No module named 'torch'"

```bash
# Make sure you're in the virtual environment
source venv/bin/activate  # or: conda activate transformer

# Reinstall PyTorch
pip install torch torchvision torchaudio
```

### "MPS backend not available"

This means your Mac doesn't support MPS (requires macOS 12.3+ and Metal-compatible GPU).

```bash
# Check macOS version
sw_vers

# If macOS < 12.3, training will use CPU
# Consider upgrading macOS or using cloud GPU
```

### "CUDA out of memory"

```bash
# Reduce batch size in config/model_config.yaml
training:
  batch_size: 8  # or smaller
  gradient_accumulation_steps: 8  # to maintain effective batch size
```

### Import errors

```bash
# Reinstall all dependencies
pip install --upgrade --force-reinstall -r requirements.txt
```

## 📋 Dependency Versions

Minimum versions:
- Python: 3.8+ (3.11 or 3.12 recommended)
- PyTorch: 2.0+
- transformers: 4.30+
- datasets: 2.14+

Check installed versions:
```bash
pip list | grep -E "torch|transformers|datasets"
```

## 🚀 Quick Test

After installation, run:

```bash
# Test device detection
python src/device_utils.py

# Quick training test (5 minutes)
bash quick_start.sh
```

## 📞 Still Having Issues?

1. Check Python version: `python --version`
2. Check PyTorch installation: `python -c "import torch; print(torch.__version__)"`
3. Check device availability: `python src/device_utils.py`
4. See error logs in the terminal
5. Open an issue with:
   - Python version
   - OS and architecture
   - Error message
   - Output of `pip list`

## 🔗 Useful Links

- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
- [HuggingFace Installation](https://huggingface.co/docs/transformers/installation)
- [Conda Documentation](https://docs.conda.io/)
- [pyenv Documentation](https://github.com/pyenv/pyenv)
