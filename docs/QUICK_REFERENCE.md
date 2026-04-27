# 🚀 Quick Reference Card

## One-Line Commands

```bash
# Setup everything
bash setup.sh && source venv/bin/activate

# Test installation
python test_installation.py

# Quick training (5-15 min)
bash quick_start.sh

# Train on WikiText-103
python train.py --config config/model_config.yaml --dataset wikitext

# Interactive generation
python src/inference.py --model checkpoints/best_model.pt --interactive

# Resume training
python train.py --config config/model_config.yaml --resume checkpoints/checkpoint.pt
```

## File Locations

| What | Where |
|------|-------|
| Main config | `config/model_config.yaml` |
| Checkpoints | `checkpoints/` |
| Logs | `logs/` |
| Model code | `src/model.py` |
| Training code | `src/trainer.py` |
| Inference code | `src/inference.py` |

## Common Config Changes

```yaml
# Smaller model (faster training)
model:
  d_model: 256
  num_layers: 4
  num_heads: 4

# Larger batch size (if you have GPU memory)
training:
  batch_size: 64

# Different dataset
data:
  dataset_name: "openwebtext"

# Force specific device
system:
  device: "mps"  # or "cuda" or "cpu"
```

## Dataset Quick Reference

| Dataset | Size | Training Time | Command |
|---------|------|---------------|---------|
| tiny_shakespeare | 1MB | 5-15 min | `--dataset tiny_shakespeare` |
| wikitext-2 | 4MB | 1-2 hours | `--dataset wikitext` |
| wikitext-103 | 500MB | 6-12 hours | `--dataset wikitext` |
| openwebtext | 40GB | 2-3 days | `--dataset openwebtext` |

## Troubleshooting Quick Fixes

```bash
# Python 3.14 issue
brew install python@3.11
python3.11 -m venv venv

# Out of memory
# Edit config: batch_size: 8, gradient_accumulation_steps: 8

# MPS issues
# Edit config: mixed_precision: false

# Missing packages
pip install -r requirements.txt

# HuggingFace login
huggingface-cli login
```

## Device Detection

```python
# Test device
python -c "from src.device_utils import select_device; select_device('auto')"

# Check PyTorch
python -c "import torch; print(torch.__version__)"

# Check CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Check MPS
python -c "import torch; print(torch.backends.mps.is_available())"
```

## Training Monitoring

```bash
# TensorBoard
tensorboard --logdir logs

# Watch training
tail -f logs/training.log

# Check GPU usage (NVIDIA)
watch -n 1 nvidia-smi

# Check process
ps aux | grep python
```

## Generation Examples

```bash
# Interactive mode
python src/inference.py --model checkpoints/best_model.pt --interactive

# Single prompt
python src/inference.py \
  --model checkpoints/best_model.pt \
  --prompt "Once upon a time" \
  --max-tokens 100

# Multiple samples
python src/inference.py \
  --model checkpoints/best_model.pt \
  --prompt "The future of AI" \
  --num-samples 3 \
  --temperature 0.9
```

## Useful Python Snippets

```python
# Load model
from src.model import create_model
from src.inference import TextGenerator

generator = TextGenerator('checkpoints/best_model.pt')
text = generator.generate("Hello", max_new_tokens=50)

# Check model size
import torch
checkpoint = torch.load('checkpoints/best_model.pt')
params = sum(p.numel() for p in checkpoint['model_state_dict'].values())
print(f"Parameters: {params/1e6:.2f}M")

# Load dataset
from datasets import load_dataset
dataset = load_dataset('tiny_shakespeare')
print(dataset['train'][0])
```

## Environment Variables

```bash
# HuggingFace token
export HUGGING_FACE_HUB_TOKEN="hf_your_token"

# CUDA device
export CUDA_VISIBLE_DEVICES=0

# Disable MPS (force CPU)
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

## Git Commands (if using version control)

```bash
# Initialize
git init
git add .
git commit -m "Initial commit"

# Ignore checkpoints and logs
echo "checkpoints/" >> .gitignore
echo "logs/" >> .gitignore
echo "venv/" >> .gitignore
echo "*.pt" >> .gitignore
```

## Performance Tips

| Situation | Solution |
|-----------|----------|
| Slow training | Use GPU, increase batch size, enable mixed precision |
| Out of memory | Reduce batch size, use gradient accumulation |
| CPU bottleneck | Reduce num_workers, use smaller dataset |
| Slow data loading | Increase num_workers, use SSD |

## Model Size vs Memory

| Model | Parameters | GPU Memory | Training Time |
|-------|-----------|------------|---------------|
| Tiny | 10M | 2GB | 5-15 min |
| Small | 40M | 4GB | 1-6 hours |
| Medium | 120M | 8GB | 1-3 days |
| Large | 350M | 16GB | 1-2 weeks |

## Quick Links

- [Full README](README.md)
- [Installation Guide](INSTALLATION.md)
- [HuggingFace Setup](HUGGINGFACE_SETUP.md)
- [Dataset Guide](DATASETS.md)
- [Production Summary](PRODUCTION_READY_SUMMARY.md)

## Emergency Commands

```bash
# Kill training
pkill -f train.py

# Free GPU memory (NVIDIA)
nvidia-smi --gpu-reset

# Clear cache
rm -rf ~/.cache/huggingface/

# Reinstall everything
rm -rf venv
bash setup.sh
```

## Support

- Test installation: `python test_installation.py`
- Check device: `python src/device_utils.py`
- Read docs: `README.md`, `INSTALLATION.md`
- HuggingFace Forum: https://discuss.huggingface.co/
