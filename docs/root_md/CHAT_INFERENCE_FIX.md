# Chat Inference Fix - PyTorch Version Compatibility

## Issue
Chat inference was failing with PyTorch version incompatibility error:
```
AttributeError: Can't get attribute '_rebuild_device_tensor_from_cpu_tensor'
```

This occurs when a checkpoint saved with PyTorch 2.4+ is loaded with PyTorch 2.2.2.

## Root Cause
- macOS pip only has PyTorch up to 2.2.2
- PyTorch 2.4+ changed internal tensor serialization format
- Checkpoints saved with 2.4+ cannot be loaded with 2.2.2

## Solutions

### Option 1: Convert Checkpoint (Recommended for macOS)

If you have the checkpoint saved with PyTorch 2.4+, convert it to 2.2 compatible format:

```bash
# On the machine with PyTorch 2.4+ (where checkpoint was created)
python scripts/convert_checkpoint_pytorch22.py \
  /path/to/original_checkpoint.pt \
  /path/to/converted_checkpoint.pt
```

Then use the converted checkpoint:
```bash
python src/finetuning/chat_inference.py \
  --base-model /path/to/converted_checkpoint.pt \
  --adapter /path/to/chat_adapter \
  --interactive
```

### Option 2: Use Conda (PyTorch 2.4+ Available)

```bash
# Install conda if not already installed
# Then create environment with PyTorch 2.4+
conda create -n neo python=3.12
conda activate neo
conda install pytorch torchvision torchaudio -c pytorch

# Install other requirements
pip install -r requirements.txt
```

### Option 3: Train with PyTorch 2.2.2

Train the model with your current PyTorch version to ensure compatibility:

```bash
# Train base model
python train.py --config config/auto_training_117m_balanced.yaml

# Fine-tune
python src/finetuning/gpu_finetune.py \
  --model checkpoints/best_model.pt \
  --train-data data/hf_cot/train.jsonl \
  --val-data data/hf_cot/val.jsonl

# Use for inference
python src/finetuning/chat_inference.py \
  --base-model checkpoints/best_model.pt \
  --adapter finetuned_model_gpu/best_model \
  --interactive
```

## Usage

### Interactive Mode (Default)
```bash
python src/finetuning/chat_inference.py \
  --base-model /path/to/base_model.pt \
  --adapter /path/to/chat_adapter \
  --interactive
```

### Single Prompt
```bash
python src/finetuning/chat_inference.py \
  --base-model /path/to/base_model.pt \
  --adapter /path/to/chat_adapter \
  --prompt "What is 2+2?"
```

### Show Thought Process
```bash
python src/finetuning/chat_inference.py \
  --base-model /path/to/base_model.pt \
  --adapter /path/to/chat_adapter \
  --prompt "Solve: 5x + 3 = 18" \
  --show-thought
```

### Load from HuggingFace Hub
```bash
python src/finetuning/chat_inference.py \
  --base-model-remote best_model.pt \
  --adapter-remote chat_adapter \
  --model-repo 0x-genesys/neo_weights_checkpoints \
  --interactive
```

### Custom Generation Parameters
```bash
python src/finetuning/chat_inference.py \
  --base-model /path/to/base_model.pt \
  --adapter /path/to/chat_adapter \
  --temperature 0.7 \
  --top-k 50 \
  --top-p 0.9 \
  --max-tokens 256 \
  --interactive
```

## Device Auto-Detection

The script automatically detects and uses the best available device:

1. **CUDA GPU** (NVIDIA) - if available
2. **MPS** (Apple Silicon) - if available
3. **CPU** - fallback

You can override with `--device`:
```bash
--device cuda    # Force CUDA
--device mps     # Force MPS
--device cpu     # Force CPU
```

## Command-Line Arguments

### Model Loading
- `--base-model`: Path to local base model checkpoint (.pt file)
- `--base-model-remote`: Filename in HuggingFace Hub (e.g., "best_model.pt")
- `--adapter`: Path to local LoRA adapter directory
- `--adapter-remote`: Path in HuggingFace Hub (e.g., "finetune/chat_adapter")
- `--model-repo`: HuggingFace repository ID (default: "0x-genesys/neo_weights_checkpoints")
- `--config`: Path to model config YAML (optional, auto-detected from checkpoint)

### Generation
- `--prompt`: Single prompt to generate response for (non-interactive)
- `--interactive`: Start interactive chat mode (default if no prompt)
- `--max-tokens`: Maximum tokens to generate (default: 256)
- `--temperature`: Sampling temperature (default: 0.8)
- `--top-k`: Top-k sampling (default: 50)
- `--top-p`: Nucleus sampling (default: 0.9)
- `--show-thought`: Display thought process in output

### Device
- `--device`: Device to use: 'auto', 'cuda', 'mps', 'cpu' (default: 'auto')

## Example Session

```bash
$ python src/finetuning/chat_inference.py \
    --base-model checkpoints/best_model.pt \
    --adapter finetuned_model_gpu/best_model \
    --interactive

================================================================================
🤖 Chat Inference with Fine-Tuned LoRA Model
================================================================================

📂 Loading base model from: checkpoints/best_model.pt
✅ Loaded config from checkpoint

🔧 Creating model...
   Architecture: 12 layers, 768 dim
✅ Base model loaded

📂 Loading LoRA adapter from: finetuned_model_gpu/best_model
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