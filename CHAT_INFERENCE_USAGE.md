# Chat Inference Usage Guide

Complete guide for using the fine-tuned chat model with LoRA adapter.

## Quick Start (Zero Configuration!)

The simplest way to use chat inference - no arguments needed:

```bash
python src/finetuning/chat_inference.py --interactive
```

This automatically:
- Downloads base model from HuggingFace Hub
- Downloads chat adapter from HuggingFace Hub  
- Auto-detects your device (CUDA/MPS/CPU)
- Starts interactive chat mode

## Default Configuration

By default, the script uses:
- **Repository**: `0x-genesys/neo_weights_checkpoints`
- **Base Model**: `best_model.pt`
- **Adapter**: `finetune/chat_adapter`
- **Device**: Auto-detect (CUDA > MPS > CPU)

## Usage Examples

### 1. Interactive Mode (Remote Models - Default)

```bash
# Simplest - uses all defaults
python src/finetuning/chat_inference.py --interactive

# Explicit remote paths
python src/finetuning/chat_inference.py \
  --base-model-remote best_model.pt \
  --adapter-remote finetune/chat_adapter \
  --interactive
```

### 2. Interactive Mode (Local Models)

```bash
python src/finetuning/chat_inference.py \
  --base-model checkpoints/best_model.pt \
  --adapter finetuned_model_gpu/best_model \
  --interactive
```

### 3. Single Prompt (Remote)

```bash
# Simple question
python src/finetuning/chat_inference.py \
  --prompt "What is 2+2?"

# Math problem with thought process
python src/finetuning/chat_inference.py \
  --prompt "Solve: 5x + 3 = 18" \
  --show-thought

# Coding question
python src/finetuning/chat_inference.py \
  --prompt "Write a Python function to reverse a string" \
  --max-tokens 300
```

### 4. Single Prompt (Local)

```bash
python src/finetuning/chat_inference.py \
  --base-model checkpoints/best_model.pt \
  --adapter finetuned_model_gpu/best_model \
  --prompt "Explain quantum computing"
```

### 5. Custom HuggingFace Repository

```bash
python src/finetuning/chat_inference.py \
  --model-repo username/my-model-repo \
  --base-model-remote my_base_model.pt \
  --adapter-remote finetune/my_adapter \
  --interactive
```

### 6. Custom Generation Parameters

```bash
python src/finetuning/chat_inference.py \
  --temperature 0.9 \
  --top-k 100 \
  --top-p 0.95 \
  --max-tokens 512 \
  --interactive
```

### 7. Force Specific Device

```bash
# Force CUDA
python src/finetuning/chat_inference.py --device cuda --interactive

# Force MPS (Apple Silicon)
python src/finetuning/chat_inference.py --device mps --interactive

# Force CPU
python src/finetuning/chat_inference.py --device cpu --interactive
```

### 8. With Custom Config

```bash
python src/finetuning/chat_inference.py \
  --config config/inference.yaml \
  --interactive
```

## Command-Line Arguments

### Model Loading

| Argument | Default | Description |
|----------|---------|-------------|
| `--base-model` | None | Path to local base model (.pt file) |
| `--base-model-remote` | `best_model.pt` | Remote base model filename from HuggingFace Hub |
| `--adapter` | None | Path to local LoRA adapter directory |
| `--adapter-remote` | `finetune/chat_adapter` | Remote adapter path in HuggingFace Hub |
| `--model-repo` | `0x-genesys/neo_weights_checkpoints` | HuggingFace repository ID |

**Priority**: Local paths (`--base-model`, `--adapter`) take priority over remote paths.

### Configuration

| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | None | Path to model config YAML (optional, auto-detected) |
| `--device` | `auto` | Device: `auto`, `cuda`, `mps`, `cpu` |

### Generation

| Argument | Default | Description |
|----------|---------|-------------|
| `--prompt` | None | Single prompt for generation (non-interactive) |
| `--interactive` | False | Run in interactive chat mode |
| `--max-tokens` | 200 | Maximum tokens to generate |
| `--temperature` | 0.7 | Sampling temperature (0.0-2.0) |
| `--top-k` | 50 | Top-k sampling |
| `--top-p` | 0.9 | Nucleus sampling threshold |
| `--show-thought` | False | Display thought process in output |

## Interactive Mode Commands

When in interactive mode, you can use these commands:

- **Type your message**: Normal chat
- **`quit`** or **`exit`**: End the conversation
- **`clear`**: Clear conversation history
- **`settings`**: View current generation settings
- **`temp <value>`**: Change temperature (e.g., `temp 0.8`)
- **`tokens <value>`**: Change max tokens (e.g., `tokens 300`)
- **`thought on/off`**: Toggle thought process display

## Generation Parameters Guide

### Temperature
- **0.1-0.3**: Very focused, deterministic (good for math/code)
- **0.7-0.8**: Balanced (default, good for general chat)
- **0.9-1.2**: Creative, diverse (good for brainstorming)

### Top-K
- **10-30**: Very focused sampling
- **50**: Balanced (default)
- **100+**: More diverse

### Top-P (Nucleus Sampling)
- **0.8**: Focused
- **0.9**: Balanced (default)
- **0.95**: More diverse

### Max Tokens
- **50-100**: Short answers
- **200**: Default, good for most responses
- **300-512**: Long, detailed responses

## Example Session

```bash
$ python src/finetuning/chat_inference.py --interactive

================================================================================
🤖 Chat Inference with Fine-Tuned LoRA Model
================================================================================

📥 Downloading base model from HuggingFace Hub...
   Repository: 0x-genesys/neo_weights_checkpoints
   File: best_model.pt
✅ Downloaded to: /tmp/hf_cache/...

📥 Downloading adapter from HuggingFace Hub...
   Repository: 0x-genesys/neo_weights_checkpoints
   Path: finetune/chat_adapter
✅ Downloaded to: /tmp/hf_cache/...

📂 Loading base model from: /tmp/hf_cache/...
✅ Base model loaded

📂 Loading LoRA adapter from: /tmp/hf_cache/...
✅ LoRA adapter loaded

🖥️  Device: cuda (NVIDIA GeForce RTX 3090)

📚 Loading tokenizer...
✅ Tokenizer loaded (vocab size: 100277)

================================================================================
💬 Interactive Chat Mode
================================================================================

Type 'quit' or 'exit' to end the conversation.
Type 'clear' to clear conversation history.

You: What is 5 + 7?