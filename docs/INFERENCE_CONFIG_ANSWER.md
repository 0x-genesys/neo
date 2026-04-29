# Do I Need a Config for Inference?

## Short Answer: **NO!** 🎉

You **do not need** to provide a config file for inference. The model checkpoint already contains all the necessary configuration.

## Why No Config Needed?

When you train a model, the training script saves the complete configuration inside the checkpoint file:

```python
checkpoint = {
    'epoch': self.epoch,
    'global_step': self.global_step,
    'model_state_dict': model_to_save.state_dict(),
    'optimizer_state_dict': self.optimizer.state_dict(),
    'best_val_loss': self.best_val_loss,
    'config': self.config  # ← Config is saved here!
}
```

The inference script automatically extracts this config:

```python
# Load checkpoint
checkpoint = torch.load(model_path, map_location='cpu')

# Get config from checkpoint
self.config = checkpoint.get('config', None)
```

## What's Included in the Checkpoint Config?

The checkpoint contains **everything** needed for inference:

✅ **Model Architecture**
- Number of layers
- Model dimensions
- Number of attention heads
- Context length
- Vocabulary size

✅ **Tokenizer Configuration**
- Tokenizer type (e.g., "tiktoken" or "gpt2")
- Special tokens

✅ **Training Hyperparameters** (for reference)
- Learning rate
- Batch size
- etc.

## Device Auto-Detection

The inference script also **auto-detects** the best available device:

1. **CUDA** (NVIDIA GPU) - if available
2. **MPS** (Apple Silicon GPU) - if available  
3. **CPU** - fallback

So you don't need to configure the device either!

## Usage Examples

### Remote Model (Recommended)
```bash
# Just specify the model - that's it!
python src/inference.py --model-remote best_model.pt --interactive
```

### Local Model
```bash
# Just specify the model path
python src/inference.py --model checkpoints/production/best_model.pt --interactive
```

### With Custom Parameters
```bash
# Override generation parameters via CLI
python src/inference.py \
    --model-remote best_model.pt \
    --temperature 1.0 \
    --top-k 100 \
    --max-tokens 200 \
    --interactive
```

## When Would I Use --config?

You only need `--config` in these rare cases:

### 1. Old Checkpoint Without Config
If you have an old checkpoint that doesn't contain the config:
```bash
python src/inference.py \
    --model old_checkpoint.pt \
    --config config/gpu_training_117m_balanced.yaml \
    --interactive
```

### 2. Override Tokenizer (Advanced)
If you want to use a different tokenizer:
```bash
python src/inference.py \
    --model-remote best_model.pt \
    --config config/custom_inference.yaml \
    --interactive
```

### 3. Documentation/Reproducibility
If you want to document your inference settings:
```bash
python src/inference.py \
    --model-remote best_model.pt \
    --config config/inference.yaml \
    --interactive
```

## Your Specific Case

You asked about a model trained with `gpu_training_117m_balanced.yaml`:

```bash
# ✅ This is all you need!
python src/inference.py --model-remote best_model.pt --interactive
```

The checkpoint already contains:
- ✅ Model architecture (117M parameters)
- ✅ Context length (512 tokens)
- ✅ Tokenizer type (tiktoken cl100k_base)
- ✅ All model hyperparameters

The inference script will:
- ✅ Auto-detect your device (CUDA/MPS/CPU)
- ✅ Load the model with correct architecture
- ✅ Use the correct tokenizer
- ✅ Set up everything automatically

## Optional: Inference Config File

We created an **optional** inference config at `config/inference.yaml` for reference and documentation purposes. You can use it if you want, but it's **not required**:

```bash
# Without config (recommended)
python src/inference.py --model-remote best_model.pt --interactive

# With config (optional, for documentation)
python src/inference.py --model-remote best_model.pt --config config/inference.yaml --interactive
```

The config file is useful for:
- 📝 Documenting your inference settings
- 🎯 Sharing inference configurations with team
- 📚 Reference for parameter ranges and recommendations

## Summary

### What You Need
```bash
python src/inference.py --model-remote best_model.pt --interactive
```

### What You Don't Need
- ❌ Config file (it's in the checkpoint)
- ❌ Device specification (auto-detected)
- ❌ Model architecture details (in the checkpoint)
- ❌ Tokenizer configuration (in the checkpoint)

### What You Can Optionally Override
```bash
python src/inference.py \
    --model-remote best_model.pt \
    --temperature 1.0 \      # Optional: Override temperature
    --top-k 100 \           # Optional: Override top-k
    --max-tokens 200 \      # Optional: Override max tokens
    --device cuda \         # Optional: Force specific device
    --interactive
```

## Quick Reference

| Scenario | Command |
|----------|---------|
| Basic inference | `python src/inference.py --model-remote best_model.pt --interactive` |
| Custom temperature | `python src/inference.py --model-remote best_model.pt --temperature 1.0 --interactive` |
| Force CPU | `python src/inference.py --model-remote best_model.pt --device cpu --interactive` |
| With config (optional) | `python src/inference.py --model-remote best_model.pt --config config/inference.yaml --interactive` |

## More Information

- **[INFERENCE_GUIDE.md](INFERENCE_GUIDE.md)** - Complete inference guide
- **[INFERENCE_QUICK_REFERENCE.md](INFERENCE_QUICK_REFERENCE.md)** - One-page reference
- **[REMOTE_MODEL_LOADING.md](REMOTE_MODEL_LOADING.md)** - Remote model loading guide
- **[README.md](../README.md)** - Main documentation

---

**TL;DR**: No config needed! Just run:
```bash
python src/inference.py --model-remote best_model.pt --interactive
```
