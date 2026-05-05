# Kaggle TPU Troubleshooting Guide

## Issue: "TPU not detected" but TPU is enabled

### Symptoms

You see this error:
```
❌ TPU not detected!

To enable TPU in Kaggle:
1. Go to notebook settings (gear icon)
2. Under 'Accelerator', select 'TPU v3-8'
3. Click 'Save'
4. Restart the notebook
```

But you've already enabled TPU in Kaggle settings!

### Root Cause

The setup script was checking for hardware device files (`/dev/accel0`) that Kaggle's TPU might not expose until torch_xla is loaded.

### Solution

**Option 1: Use Updated Setup Script** (Recommended)

The setup script has been updated to skip the hardware check and verify TPU after installing torch_xla:

```bash
!bash scripts/setup_kaggle_tpu.sh
```

**Option 2: Check TPU Directly**

Use the TPU checker script:

```bash
!python scripts/check_tpu.py
```

This will:
- ✅ Check if torch_xla is installed
- ✅ Check if TPU is available
- ✅ Show TPU device and core count
- ✅ Provide specific error messages

**Option 3: Manual Verification**

Run this in a Kaggle notebook cell:

```python
# Check if torch_xla is installed
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    print(f"✅ torch_xla version: {torch_xla.__version__}")
    
    # Try to get TPU device
    device = xm.xla_device()
    print(f"✅ TPU device: {device}")
    print(f"✅ TPU cores: {xm.xrt_world_size()}")
    print(f"✅ TPU ordinal: {xm.get_ordinal()}")
    
except ImportError:
    print("❌ torch_xla not installed")
    print("Run: !bash scripts/setup_kaggle_tpu.sh")
    
except Exception as e:
    print(f"❌ TPU not available: {e}")
    print("\nMake sure:")
    print("1. TPU v3-8 is selected in notebook settings")
    print("2. Notebook has been restarted after enabling TPU")
    print("3. You have TPU quota available")
```

## Common Issues and Solutions

### Issue 1: TPU Not Actually Enabled

**Symptoms**:
```
❌ TPU not available: No TPU devices found
```

**Solution**:
1. Click the **Settings** icon (gear) in Kaggle notebook
2. Look for **Accelerator** dropdown
3. Make sure **TPU v3-8** is selected (not "None" or "GPU")
4. Click **Save**
5. Wait for notebook to restart
6. Run setup script again

**Verify**:
```bash
!python scripts/check_tpu.py
```

### Issue 2: Notebook Not Restarted

**Symptoms**:
- TPU enabled in settings
- But torch_xla can't find TPU

**Solution**:
1. After enabling TPU, Kaggle should automatically restart
2. If not, manually restart: **Run** → **Restart Session**
3. Re-run all cells including setup

### Issue 3: torch_xla Not Installed

**Symptoms**:
```
❌ torch_xla not installed
ModuleNotFoundError: No module named 'torch_xla'
```

**Solution**:
```bash
# Install using official script
!curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py
!python pytorch-xla-env-setup.py --version nightly --apt-packages libomp5 libopenblas-dev
```

Or use our setup script:
```bash
!bash scripts/setup_kaggle_tpu.sh
```

### Issue 4: TPU Quota Exhausted

**Symptoms**:
```
❌ TPU not available: Quota exceeded
```

**Solution**:
- Kaggle free tier: 30 hours/week TPU
- Check your usage: **Account** → **Usage**
- Wait for quota to reset (weekly)
- Or upgrade to Kaggle Pro for more quota

### Issue 5: Wrong Accelerator Selected

**Symptoms**:
- Setup script says TPU not detected
- But you think TPU is enabled

**Verify**:
1. Go to notebook settings
2. Check **Accelerator** dropdown
3. Should show **TPU v3-8** (not "GPU T4 x2" or "None")
4. If wrong, select **TPU v3-8**
5. Click **Save**
6. Wait for restart

### Issue 6: Old Kaggle Notebook

**Symptoms**:
- TPU option not available in settings
- Only see GPU options

**Solution**:
- Create a new notebook (TPU support added recently)
- Or check if your account has TPU access
- Free tier accounts should have TPU access

## Verification Checklist

Run through this checklist:

### 1. Check Kaggle Environment
```bash
!ls /kaggle
```
**Expected**: Should show `/kaggle/working`, `/kaggle/input`, etc.

### 2. Check Accelerator Setting
- Go to Settings (gear icon)
- Accelerator should show: **TPU v3-8**
- Not "None", not "GPU"

### 3. Check torch_xla Installation
```python
!python -c "import torch_xla; print(torch_xla.__version__)"
```
**Expected**: Should print version like `2.0.0` or `2.1.0`

### 4. Check TPU Device
```python
!python -c "import torch_xla.core.xla_model as xm; print(xm.xla_device())"
```
**Expected**: Should print `xla:0` or similar

### 5. Check TPU Cores
```python
!python -c "import torch_xla.core.xla_model as xm; print(xm.xrt_world_size())"
```
**Expected**: Should print `8` (for TPU v3-8)

### 6. Run Full Check
```bash
!python scripts/check_tpu.py
```
**Expected**: All checks should pass

## Quick Fix Commands

### If torch_xla not installed:
```bash
!bash scripts/setup_kaggle_tpu.sh
```

### If TPU not detected after install:
```bash
# Check TPU status
!python scripts/check_tpu.py

# If still not working, restart notebook:
# Run → Restart Session
# Then re-run setup
```

### If everything fails:
```bash
# 1. Save your work
# 2. Create a new Kaggle notebook
# 3. In settings, select TPU v3-8
# 4. Clone your repo
# 5. Run setup script
!git clone https://github.com/0x-genesys/neo.git
%cd neo
!bash scripts/setup_kaggle_tpu.sh
```

## Expected Output (Success)

When everything works, you should see:

```bash
$ bash scripts/setup_kaggle_tpu.sh

================================================================================
Kaggle TPU Setup
================================================================================

✅ Kaggle environment detected

📝 Note: TPU verification will happen after torch_xla installation

📦 Installing torch_xla using official setup script...
   This may take a few minutes...

[Installation output...]

✅ torch_xla installed successfully

🔍 Verifying torch_xla installation and TPU availability...
✅ torch_xla version: 2.0.0
✅ TPU device: xla:0
✅ TPU cores: 8
✅ TPU ordinal: 0

================================================================================
✅ Kaggle TPU Setup Complete!
================================================================================

You can now train with TPU:
  python train.py --config config/auto_training_117m_balanced.yaml --tpu
```

## Still Having Issues?

### Check Kaggle Status
- Visit: https://www.kaggle.com/status
- Check if TPU service is operational

### Check Your Account
- Visit: https://www.kaggle.com/account
- Check TPU quota usage
- Verify account is in good standing

### Try GPU First
If TPU continues to have issues, you can train on GPU:

```bash
# Use GPU instead (slower but works)
!python train.py --config config/auto_training_117m_balanced.yaml
```

GPU (T4) is 3x slower than TPU but still works fine.

### Contact Support
If nothing works:
1. Check Kaggle forums: https://www.kaggle.com/discussions
2. Report issue with:
   - Output of `!python scripts/check_tpu.py`
   - Screenshot of notebook settings
   - Your Kaggle username

## Summary

✅ **Updated setup script** - No longer checks hardware files  
✅ **New check script** - `scripts/check_tpu.py` for diagnostics  
✅ **Better error messages** - Specific guidance for each issue  
✅ **Verification checklist** - Step-by-step debugging  

**Most common fix**: Make sure TPU v3-8 is selected in settings and notebook has restarted!

---

**Last Updated**: 2026-04-30
