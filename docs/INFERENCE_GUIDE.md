# Inference Guide

Complete guide for running inference with Neo transformer models.

## Quick Start

### Basic Inference (No Config Needed!)

The model checkpoint contains all necessary configuration. Just run:

```bash
# Local model
python src/inference.py --model checkpoints/production/best_model.pt --prompt "Hello, world!"

# Remote model from HuggingFace Hub
python src/inference.py --model-remote best_model.pt --prompt "Hello, world!"
```

### Interactive Mode

```bash
# Local model
python src/inference.py --model checkpoints/production/best_model.pt --interactive

# Remote model
python src/inference.py --model-remote best_model.pt --interactive
```

## Do I Need a Config File?

**Short answer: NO!**

The checkpoint file already contains the training configuration, including:
- Model architecture (layers, dimensions, heads)
- Context length
- Tokenizer type
- All model hyperparameters

The inference script automatically extracts this config from the checkpoint.

### When to Use a Config File

You only need `--config` if you want to:
1. **Override generation parameters** (though CLI args are easier)
2. **Document your inference settings** for reproducibility
3. **Use a different tokenizer** (advanced use case)

Example with config:
```bash
python src/inference.py \
    --model-remote best_model.pt \
    --config config/inference.yaml \
    --interactive
```

## Device Selection

The inference script **auto-detects** the best available device:

1. **CUDA** (NVIDIA GPU) - if available
2. **MPS** (Apple Silicon GPU) - if available
3. **CPU** - fallback

### Manual Device Selection

```bash
# Force CPU
python src/inference.py --model-remote best_model.pt --device cpu --interactive

# Force CUDA
python src/inference.py --model-remote best_model.pt --device cuda --interactive

# Force MPS (Apple Silicon)
python src/inference.py --model-remote best_model.pt --device mps --interactive
```

## Generation Parameters

### Command Line Arguments

```bash
python src/inference.py \
    --model-remote best_model.pt \
    --prompt "The future of AI" \
    --max-tokens 200 \        # Maximum tokens to generate
    --temperature 0.8 \       # Sampling temperature (0.1-2.0)
    --top-k 50 \             # Top-k sampling
    --top-p 0.95 \           # Nucleus sampling
    --num-samples 3          # Generate 3 different completions
```

### Interactive Mode Settings

In interactive mode, you can change settings on the fly:

```bash
python src/inference.py --model-remote best_model.pt --interactive

Prompt: config                    # Show current settings
Prompt: set temperature 1.0       # Change temperature
Prompt: set max_new_tokens 200    # Change max tokens
Prompt: set top_k 100            # Change top-k
Prompt: set top_p 0.9            # Change top-p
Prompt: Once upon a time         # Generate text
```

### Parameter Guide

#### Temperature (0.1 - 2.0)
- **0.1-0.3**: Very deterministic, focused (good for factual Q&A)
- **0.5-0.8**: Balanced (default: 0.8)
- **1.0-1.5**: Creative, diverse (good for creative writing)
- **1.5-2.0**: Very random, experimental

#### Top-k (1 - 100+)
- **5-10**: Very focused (good for code generation)
- **20-50**: Balanced (default: 50)
- **100+**: Very diverse (good for brainstorming)

#### Top-p (0.0 - 1.0)
- **0.8-0.9**: Focused (good for factual content)
- **0.9-0.95**: Balanced (default: 0.95)
- **0.95-1.0**: Diverse (good for creative content)

## Use Case Examples

### 1. Creative Writing

```bash
python src/inference.py \
    --model-remote best_model.pt \
    --temperature 1.0 \
    --top-k 100 \
    --top-p 0.95 \
    --max-tokens 300 \
    --prompt "In a world where magic and technology coexist"
```

### 2. Code Generation

```bash
python src/inference.py \
    --model-remote best_model.pt \
    --temperature 0.2 \
    --top-k 10 \
    --top-p 0.9 \
    --max-tokens 200 \
    --prompt "def fibonacci(n):"
```

### 3. Factual Q&A

```bash
python src/inference.py \
    --model-remote best_model.pt \
    --temperature 0.1 \
    --top-k 5 \
    --top-p 0.85 \
    --max-tokens 100 \
    --prompt "What is the capital of France?"
```

### 4. Conversational AI

```bash
python src/inference.py \
    --model-remote best_model.pt \
    --temperature 0.8 \
    --top-k 50 \
    --top-p 0.95 \
    --interactive
```

## Remote Model Loading

### From Default Repository

```bash
# Default repo: 0x-genesys/neo_weights_checkpoints
python src/inference.py --model-remote best_model.pt --interactive
```

### From Custom Repository

```bash
python src/inference.py \
    --model-remote checkpoint.pt \
    --model-repo your-username/your-model-repo \
    --interactive
```

### Available Checkpoints

List available checkpoints:
```python
from src.remote_model_loader import RemoteModelLoader

loader = RemoteModelLoader(repo_id="0x-genesys/neo_weights_checkpoints")
checkpoints = loader.list_available_checkpoints()
print(checkpoints)
```

## Batch Inference

For processing multiple prompts:

```python
from src.inference import TextGenerator

# Initialize generator
generator = TextGenerator(
    model_path="best_model.pt",
    model_repo="0x-genesys/neo_weights_checkpoints"
)

# Batch generate
prompts = [
    "Once upon a time",
    "In the year 2050",
    "The secret to happiness is"
]

results = generator.batch_generate(
    prompts,
    max_new_tokens=100,
    temperature=0.8
)

for prompt, result in zip(prompts, results):
    print(f"Prompt: {prompt}")
    print(f"Result: {result}\n")
```

## Performance Tips

### 1. Use GPU for Faster Inference

```bash
# Auto-detect (recommended)
python src/inference.py --model-remote best_model.pt --interactive

# Force GPU
python src/inference.py --model-remote best_model.pt --device cuda --interactive
```

### 2. Adjust Max Tokens

Shorter sequences are faster:
```bash
# Fast (50 tokens)
python src/inference.py --model-remote best_model.pt --max-tokens 50 --interactive

# Slow (500 tokens)
python src/inference.py --model-remote best_model.pt --max-tokens 500 --interactive
```

### 3. Batch Processing

Process multiple prompts together for better GPU utilization:
```python
# Better GPU utilization
results = generator.batch_generate(prompts, max_new_tokens=100)
```

## Memory Usage

### Model Size vs Memory

| Model Size | Parameters | GPU Memory | CPU Memory |
|------------|------------|------------|------------|
| Quick Start | 2.36M | ~100MB | ~200MB |
| Production | 16M | ~500MB | ~1GB |
| Full (117M) | 117M | ~2-3GB | ~4-5GB |

### Reduce Memory Usage

```bash
# Use CPU (slower but less memory)
python src/inference.py --model-remote best_model.pt --device cpu --interactive

# Reduce max tokens
python src/inference.py --model-remote best_model.pt --max-tokens 50 --interactive
```

## Troubleshooting

### Issue: "Config not found in checkpoint"

**Solution**: The checkpoint is from an old version. Provide a config file:
```bash
python src/inference.py \
    --model checkpoints/old_model.pt \
    --config config/gpu_training_117m_balanced.yaml \
    --interactive
```

### Issue: "Out of memory"

**Solution**: Use CPU or reduce max tokens:
```bash
# Use CPU
python src/inference.py --model-remote best_model.pt --device cpu --interactive

# Or reduce max tokens
python src/inference.py --model-remote best_model.pt --max-tokens 50 --interactive
```

### Issue: "Model generates gibberish"

**Solution**: Adjust temperature and sampling parameters:
```bash
# More focused generation
python src/inference.py \
    --model-remote best_model.pt \
    --temperature 0.5 \
    --top-k 20 \
    --interactive
```

### Issue: "Model is too repetitive"

**Solution**: Increase temperature and diversity:
```bash
# More diverse generation
python src/inference.py \
    --model-remote best_model.pt \
    --temperature 1.0 \
    --top-k 100 \
    --top-p 0.95 \
    --interactive
```

### Issue: "Slow inference on CPU"

**Solution**: This is expected. Use GPU if available:
```bash
# Check available devices
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'MPS available: {torch.backends.mps.is_available()}')
"

# Use GPU
python src/inference.py --model-remote best_model.pt --device cuda --interactive
```

## Advanced Usage

### Custom Tokenizer

```python
from src.inference import TextGenerator
from transformers import AutoTokenizer

# Load with custom tokenizer
generator = TextGenerator(
    model_path="best_model.pt",
    model_repo="0x-genesys/neo_weights_checkpoints"
)

# Replace tokenizer
generator.tokenizer = AutoTokenizer.from_pretrained("gpt2")
```

### Perplexity Calculation

```python
from src.inference import TextGenerator

generator = TextGenerator(
    model_path="best_model.pt",
    model_repo="0x-genesys/neo_weights_checkpoints"
)

text = "The quick brown fox jumps over the lazy dog"
perplexity = generator.get_perplexity(text)
print(f"Perplexity: {perplexity:.2f}")
```

### Stop Tokens

```python
generated = generator.generate(
    prompt="Once upon a time",
    max_new_tokens=200,
    stop_tokens=["\n\n", "THE END"]  # Stop at double newline or "THE END"
)
```

## Best Practices

### 1. Start with Defaults

The default parameters work well for most use cases:
```bash
python src/inference.py --model-remote best_model.pt --interactive
```

### 2. Experiment in Interactive Mode

Test different settings interactively before scripting:
```bash
Prompt: set temperature 0.5
Prompt: set top_k 20
Prompt: Test prompt here
```

### 3. Use Appropriate Parameters for Task

- **Creative writing**: High temperature (1.0), high top-k (100)
- **Code generation**: Low temperature (0.2), low top-k (10)
- **Factual Q&A**: Very low temperature (0.1), very low top-k (5)

### 4. Monitor Memory Usage

```bash
# Check GPU memory
nvidia-smi

# Check CPU memory
top
```

### 5. Cache Models Locally

For repeated use, the model is cached automatically:
```bash
# First run: Downloads model
python src/inference.py --model-remote best_model.pt --interactive

# Subsequent runs: Uses cached model (faster)
python src/inference.py --model-remote best_model.pt --interactive
```

## Summary

### Key Points

✅ **No config needed** - Config is in the checkpoint
✅ **Auto device detection** - Automatically uses best device
✅ **Remote loading** - Load models from HuggingFace Hub
✅ **Interactive mode** - Test and adjust settings on the fly
✅ **Flexible parameters** - Customize generation for your use case

### Quick Commands

```bash
# Basic inference
python src/inference.py --model-remote best_model.pt --prompt "Hello"

# Interactive mode
python src/inference.py --model-remote best_model.pt --interactive

# Custom parameters
python src/inference.py --model-remote best_model.pt --temperature 1.0 --top-k 100 --interactive

# Force device
python src/inference.py --model-remote best_model.pt --device cuda --interactive
```

## See Also

- [README.md](../README.md) - Main documentation
- [REMOTE_MODEL_LOADING.md](REMOTE_MODEL_LOADING.md) - Remote model loading guide
- [ARCHITECTURE.md](../ARCHITECTURE.md) - System architecture
- [GPU_MEMORY_GUIDE.md](GPU_MEMORY_GUIDE.md) - Memory optimization
