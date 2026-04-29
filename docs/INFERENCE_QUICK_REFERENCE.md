# Inference Quick Reference

One-page reference for Neo transformer inference.

## 🚀 Quick Start

```bash
# No config needed! Config is in the checkpoint
python src/inference.py --model-remote best_model.pt --interactive
```

## 📋 Common Commands

### Basic Inference
```bash
# Local model
python src/inference.py --model checkpoints/production/best_model.pt --prompt "Hello"

# Remote model (HuggingFace Hub)
python src/inference.py --model-remote best_model.pt --prompt "Hello"
```

### Interactive Mode
```bash
# Auto-detect device (CUDA > MPS > CPU)
python src/inference.py --model-remote best_model.pt --interactive

# Force specific device
python src/inference.py --model-remote best_model.pt --device cuda --interactive
```

### Custom Parameters
```bash
python src/inference.py \
    --model-remote best_model.pt \
    --temperature 1.0 \
    --top-k 100 \
    --max-tokens 200 \
    --interactive
```

## 🎛️ Generation Parameters

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| `--temperature` | 0.1-2.0 | 0.8 | Randomness (lower=focused, higher=creative) |
| `--top-k` | 1-100+ | 50 | Keep top k tokens (lower=focused, higher=diverse) |
| `--top-p` | 0.0-1.0 | 0.95 | Nucleus sampling (lower=focused, higher=diverse) |
| `--max-tokens` | 1-1000+ | 100 | Maximum tokens to generate |
| `--num-samples` | 1-10+ | 1 | Number of completions to generate |

## 🎯 Use Case Presets

### Creative Writing
```bash
--temperature 1.0 --top-k 100 --top-p 0.95 --max-tokens 300
```

### Code Generation
```bash
--temperature 0.2 --top-k 10 --top-p 0.9 --max-tokens 200
```

### Factual Q&A
```bash
--temperature 0.1 --top-k 5 --top-p 0.85 --max-tokens 100
```

### Balanced (Default)
```bash
--temperature 0.8 --top-k 50 --top-p 0.95 --max-tokens 100
```

## 🖥️ Device Selection

```bash
# Auto-detect (recommended)
python src/inference.py --model-remote best_model.pt --interactive

# Force CUDA (NVIDIA GPU)
python src/inference.py --model-remote best_model.pt --device cuda --interactive

# Force MPS (Apple Silicon)
python src/inference.py --model-remote best_model.pt --device mps --interactive

# Force CPU
python src/inference.py --model-remote best_model.pt --device cpu --interactive
```

## 🔧 Interactive Mode Commands

```bash
Prompt: config                    # Show current settings
Prompt: set temperature 1.0       # Change temperature
Prompt: set max_new_tokens 200    # Change max tokens
Prompt: set top_k 100            # Change top-k
Prompt: set top_p 0.9            # Change top-p
Prompt: quit                     # Exit
```

## 📦 Remote Model Loading

```bash
# From default repo (0x-genesys/neo_weights_checkpoints)
python src/inference.py --model-remote best_model.pt --interactive

# From custom repo
python src/inference.py \
    --model-remote checkpoint.pt \
    --model-repo your-username/your-repo \
    --interactive

# Available checkpoints
python -c "
from src.remote_model_loader import RemoteModelLoader
loader = RemoteModelLoader()
print(loader.list_available_checkpoints())
"
```

## 💾 Memory Usage

| Model Size | Parameters | GPU Memory | CPU Memory |
|------------|------------|------------|------------|
| Quick Start | 2.36M | ~100MB | ~200MB |
| Production | 16M | ~500MB | ~1GB |
| Full (117M) | 117M | ~2-3GB | ~4-5GB |

## 🐛 Troubleshooting

### Out of Memory
```bash
# Use CPU
python src/inference.py --model-remote best_model.pt --device cpu --interactive

# Or reduce max tokens
python src/inference.py --model-remote best_model.pt --max-tokens 50 --interactive
```

### Gibberish Output
```bash
# Lower temperature
python src/inference.py --model-remote best_model.pt --temperature 0.5 --interactive
```

### Too Repetitive
```bash
# Higher temperature and diversity
python src/inference.py --model-remote best_model.pt --temperature 1.0 --top-k 100 --interactive
```

### Slow on CPU
```bash
# Check for GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, MPS: {torch.backends.mps.is_available()}')"

# Use GPU if available
python src/inference.py --model-remote best_model.pt --device cuda --interactive
```

## 🔑 Key Points

✅ **No config needed** - Config is stored in checkpoint
✅ **Auto device detection** - Uses best available device
✅ **Remote loading** - Load from HuggingFace Hub
✅ **Interactive mode** - Adjust settings on the fly
✅ **Flexible parameters** - Customize for your use case

## 📚 Full Documentation

See [INFERENCE_GUIDE.md](INFERENCE_GUIDE.md) for complete documentation.

## 🎓 Examples

### Example 1: Quick Test
```bash
python src/inference.py --model-remote best_model.pt --prompt "Once upon a time"
```

### Example 2: Creative Story
```bash
python src/inference.py \
    --model-remote best_model.pt \
    --temperature 1.0 \
    --top-k 100 \
    --max-tokens 300 \
    --prompt "In a world where magic and technology coexist"
```

### Example 3: Code Completion
```bash
python src/inference.py \
    --model-remote best_model.pt \
    --temperature 0.2 \
    --top-k 10 \
    --prompt "def fibonacci(n):"
```

### Example 4: Interactive Session
```bash
python src/inference.py --model-remote best_model.pt --interactive

Prompt: set temperature 1.0
Prompt: set max_new_tokens 200
Prompt: Tell me a story about a robot
```

### Example 5: Multiple Completions
```bash
python src/inference.py \
    --model-remote best_model.pt \
    --num-samples 3 \
    --prompt "The secret to happiness is"
```

## 🔗 Links

- **Model Repository**: [0x-genesys/neo_weights_checkpoints](https://huggingface.co/0x-genesys/neo_weights_checkpoints)
- **Dataset Repository**: [0x-genesys/mix_wiki_code_chat_data_300M_tokens](https://huggingface.co/datasets/0x-genesys/mix_wiki_code_chat_data_300M_tokens)
- **Full Guide**: [INFERENCE_GUIDE.md](INFERENCE_GUIDE.md)
- **Main README**: [README.md](../README.md)
