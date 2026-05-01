# Environment Fix Guide: PyTorch Compatibility Issues

## 🚨 The Error You're Seeing

```
ImportError: cannot import name 'GradScaler' from 'torch.amp'
```

**Cause**: PyTorch 2.2.2 uses `torch.cuda.amp` but the code was trying to import from `torch.amp` (which is only available in PyTorch 2.4+).

**Status**: ✅ **FIXED** - The code now supports PyTorch 2.0+

## 🚀 Quick Fix (2 minutes)

### Option 1: Upgrade PyTorch (Recommended)

```bash
# Automatic upgrade based on your system
bash scripts/upgrade_pytorch.sh

# Or manually:
pip install --upgrade torch torchvision
```

### Option 2: Use Current PyTorch

The code is now compatible with PyTorch 2.2.2, so you can just run:

```bash
# Check environment
python scripts/fix_environment.py

# Start training
python train.py --config config/quick_start.yaml
```

## 🔍 Diagnostic Steps

### Step 1: Check Your Environment

```bash
python scripts/fix_environment.py
```

This will check:
- Python version
- PyTorch version and availability
- Transformers library
- All dependencies
- Mixed precision support
- Critical imports

### Step 2: Verify PyTorch

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import torch; print(f'MPS: {torch.backends.mps.is_available() if hasattr(torch.backends, \"mps\") else False}')"
```

### Step 3: Test Imports

```bash
python -c "from src.trainer import Trainer; print('✅ Trainer OK')"
python -c "from src.model import create_model; print('✅ Model OK')"
python -c "from src.data import load_data; print('✅ Data OK')"
```

## 🔧 Complete Fix Options

### Option A: Upgrade PyTorch (Best)

```bash
# For Mac (Apple Silicon or Intel)
pip install --upgrade torch torchvision torchaudio

# For Linux with NVIDIA GPU
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For Linux CPU only
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Option B: Fresh Virtual Environment

```bash
# Deactivate current environment
deactivate

# Create new environment
python3 -m venv venv_new
source venv_new/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Verify
python scripts/fix_environment.py
```

### Option C: Keep Current PyTorch

The code is now compatible with PyTorch 2.0+, so you can keep your current version:

```bash
# Just verify it works
python scripts/fix_environment.py

# Start training
python train.py --config config/quick_start.yaml
```

## 🐛 What Was Fixed

### 1. GradScaler Import Compatibility

**Before** (only worked with PyTorch 2.4+):
```python
from torch.amp import autocast, GradScaler
```

**After** (works with PyTorch 2.0+):
```python
try:
    from torch.amp import autocast, GradScaler  # PyTorch 2.4+
except ImportError:
    from torch.cuda.amp import autocast, GradScaler  # PyTorch 2.0-2.3
```

### 2. GradScaler Initialization

**Before**:
```python
self.scaler = GradScaler(device_type)  # Only works in 2.4+
```

**After**:
```python
try:
    self.scaler = GradScaler(device_type)  # Try 2.4+ API
except TypeError:
    self.scaler = GradScaler()  # Fallback to 2.0-2.3 API
```

### 3. Graceful Degradation

If mixed precision isn't available, the code now:
- Detects the issue
- Prints a warning
- Disables mixed precision automatically
- Continues training (slower but works)

## 📊 Version Compatibility Matrix

| PyTorch Version | Status | Notes |
|----------------|--------|-------|
| 2.4+ | ✅ Full support | All features work |
| 2.0-2.3 | ✅ Full support | All features work |
| 1.x | ❌ Not supported | Upgrade required |

## 🧪 Testing Your Fix

### Quick Test
```bash
# Should complete without errors
python scripts/fix_environment.py
```

### Training Test
```bash
# Short training run (10 steps)
python train.py --config config/quick_start.yaml --max-steps 10
```

### Full Test
```bash
# Complete quick start training
python train.py --config config/quick_start.yaml
```

## 🎯 Expected Output After Fix

```bash
$ python scripts/fix_environment.py

🔍 Environment Diagnostic Tool
==================================================
🐍 Checking Python version...
   Python 3.12.x
   ✅ Python version OK

🔥 Checking PyTorch...
   PyTorch version: 2.2.2
   ℹ️  CPU only (no GPU acceleration)
   ✅ PyTorch version OK

🤗 Checking transformers...
   Transformers version: 4.x.x
   ✅ Transformers OK

📦 Checking other dependencies...
   ✅ numpy
   ✅ tqdm
   ✅ pyyaml
   ✅ tensorboard
   ✅ tiktoken
   ✅ datasets
   ✅ huggingface-hub

⚡ Checking mixed precision support...
   ✅ torch.cuda.amp available (PyTorch 2.0-2.3)

🧪 Testing critical imports...
   Testing: src.model
   ✅ src.model OK
   Testing: src.trainer
   ✅ src.trainer OK
   Testing: src.data
   ✅ src.data OK

==================================================
📊 SUMMARY
==================================================
Python Version      : ✅ PASS
PyTorch             : ✅ PASS
Transformers        : ✅ PASS
Other Dependencies  : ✅ PASS
Mixed Precision     : ✅ PASS
Critical Imports    : ✅ PASS

Results: 6/6 checks passed

🎉 All checks passed! Your environment is ready.

You can now run:
  python train.py --config config/quick_start.yaml
```

## 🔗 Quick Commands Reference

```bash
# Check environment
python scripts/fix_environment.py

# Upgrade PyTorch
bash scripts/upgrade_pytorch.sh

# Test training
python train.py --config config/quick_start.yaml --max-steps 10

# Full training
python train.py --config config/quick_start.yaml
```

## 💡 Pro Tips

1. **Always check environment first**: Run `python scripts/fix_environment.py` before training
2. **Use virtual environments**: Keeps dependencies isolated and clean
3. **Upgrade regularly**: `pip install --upgrade torch transformers` for latest features
4. **Test after changes**: Quick 10-step test run to verify everything works

## 🆘 Still Having Issues?

If you still see errors after trying these fixes:

1. **Check Python version**: Must be 3.8+
   ```bash
   python --version
   ```

2. **Verify virtual environment**: Make sure you're in the right venv
   ```bash
   which python
   ```

3. **Clean install**: Remove and recreate virtual environment
   ```bash
   rm -rf venv
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

4. **Check for conflicts**: Look for multiple PyTorch installations
   ```bash
   pip list | grep torch
   ```

The code is now fully compatible with PyTorch 2.0+ and should work out of the box! 🎉