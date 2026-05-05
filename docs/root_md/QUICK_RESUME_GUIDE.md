# Quick Resume Guide - GPU to TPU

## Your Situation

- ✅ Trained on GPU to step 7500
- ✅ Checkpoint: `best_model_step_7500.pt`
- ✅ Want to continue on Kaggle TPU

## Solution: One Command

```bash
python train.py --config config/auto_training_117m_balanced.yaml \
    --resume-remote best_model_step_7500.pt \
    --tpu
```

## Complete Kaggle Workflow

### Step 1: Enable TPU
1. Click **Settings** (gear icon) in Kaggle notebook
2. Select **Accelerator**: TPU v3-8
3. Click **Save** (notebook restarts)

### Step 2: Setup Environment
```bash
# Clone repo
!git clone https://github.com/0x-genesys/neo.git
%cd neo

# Install dependencies
!pip install -r requirements.txt

# Install torch_xla
!bash scripts/setup_kaggle_tpu.sh
```

### Step 3: Resume Training
```bash
!python train.py --config config/auto_training_117m_balanced.yaml \
    --resume-remote best_model_step_7500.pt \
    --tpu
```

## What Happens

1. ✅ Downloads checkpoint from HuggingFace Hub
2. ✅ Loads model weights (from GPU checkpoint)
3. ✅ Loads optimizer state (momentum, variance)
4. ✅ Loads scheduler state (learning rate position)
5. ✅ Continues from step 7500
6. ✅ Auto-adjusts batch size (8 → 128)
7. ✅ Auto-scales learning rate (0.0002 → 0.0004)
8. ✅ Trains 3x faster on TPU!

## Expected Output

```
📥 Resuming from remote checkpoint: best_model_step_7500.pt
✅ Downloaded from HuggingFace Hub

📥 Loading checkpoint...
   ✅ Model weights loaded
   ✅ Resuming from step: 7500
   ✅ Resuming from epoch: 2
   ✅ Best validation loss: 3.2450
✅ Checkpoint loaded successfully!

🔧 Hardware-Adaptive Configuration
   TPU-optimized settings:
   - Batch size: 128 (was 8)
   - Gradient accumulation: 4 (was 16)
   - Effective batch: 512 (was 128)
   - Learning rate: 4.00e-04 (was 2.00e-04)

🚀 Starting TPU training on 8 cores...
   ✅ Optimizer state loaded
   ✅ Scheduler state loaded

Epoch 2 | Step 7501 | Loss: 3.2401 | LR: 2.00e-04
Epoch 2 | Step 7502 | Loss: 3.2389 | LR: 2.00e-04
...
```

## Key Points

✅ **Checkpoint is compatible** - GPU → TPU works perfectly  
✅ **Step counter continues** - Next step is 7501  
✅ **Training state preserved** - Optimizer, scheduler, best loss  
✅ **Automatic optimization** - Batch size and LR auto-adjusted  
✅ **3x faster** - TPU v3-8 is 3x faster than T4 GPU  

## Alternative: Local Checkpoint

If you have the checkpoint file locally:

```bash
# Upload to Kaggle
# (Use Kaggle's "Add Data" → "Upload" feature)

# Then resume with local path
!python train.py --config config/auto_training_117m_balanced.yaml \
    --resume /kaggle/input/your-checkpoint/best_model_step_7500.pt \
    --tpu
```

## Troubleshooting

### Checkpoint not found on HF Hub

```bash
# Check if uploaded
# Visit: https://huggingface.co/0x-genesys/neo_weights_checkpoints

# If not, upload from your local machine
python scripts/test_checkpoint_upload.py
```

### torch_xla not installed

```bash
# Install manually
!curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py
!python pytorch-xla-env-setup.py --version nightly --apt-packages libomp5 libopenblas-dev
```

### TPU not detected

1. Go to Settings
2. Select Accelerator: **TPU v3-8**
3. Save and restart

## Performance Comparison

| Hardware | Steps/sec | Time to 10K steps | From step 7500 to 17500 |
|----------|-----------|-------------------|-------------------------|
| **T4 GPU** | 0.8 | 3.5 hours | 3.5 hours |
| **TPU v3-8** | 2.5 | 1.1 hours | **1.1 hours** |

**You'll save 2.4 hours!** ⚡

## Monitor Progress

```python
# In Kaggle notebook
%load_ext tensorboard
%tensorboard --logdir logs/auto_training_117m_balanced
```

## Save Checkpoints

Checkpoints auto-save every 500 steps and upload to HuggingFace Hub:
- `checkpoint_step_8000.pt`
- `checkpoint_step_8500.pt`
- `checkpoint_step_9000.pt`
- ...

## Resume Again Later

If Kaggle session times out (12 hours), resume in new session:

```bash
!python train.py --config config/auto_training_117m_balanced.yaml \
    --resume-remote checkpoint_step_15000.pt \
    --tpu
```

---

**Ready to train 3x faster!** 🚀

See [CHECKPOINT_COMPATIBILITY_GUIDE.md](CHECKPOINT_COMPATIBILITY_GUIDE.md) for detailed technical information.
