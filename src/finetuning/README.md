# Fine-Tuning Suite for 117M Transformer

Hardware-adaptive fine-tuning suite using **LoRA** (Low-Rank Adaptation) and **Chain-of-Thought** (CoT) alignment to convert the pre-trained 117M parameter Transformer into a general-purpose chatbot that excels at reasoning and handles code without language bleed.

## 🎯 Overview

This fine-tuning suite implements:

- **LoRA (Low-Rank Adaptation)**: Efficient fine-tuning with only ~1% trainable parameters
- **Chain-of-Thought (CoT)**: Step-by-step reasoning with explicit thought processes
- **Hardware Adaptive**: Automatic optimization for CUDA, MPS, TPU, and CPU
- **Mixed Precision**: FP16 training for faster convergence on compatible hardware
- **Frozen Embeddings**: 100k token embedding layer explicitly frozen to preserve pre-training

## 📋 Model Specifications

### Base Model (117M Parameters)
- **Architecture**: Decoder-only Transformer (GPT-style)
- **Parameters**: 117M
- **d_model**: 768
- **Layers**: 12
- **Heads**: 12
- **Vocabulary**: 100,277 (tiktoken)
- **Context Length**: 512 tokens
- **Pre-training Loss**: ~2.1

### LoRA Configuration
- **Rank (r)**: 16
- **Alpha (α)**: 32
- **Target Modules**: All linear layers
  - Attention: `c_attn` (combined QKV), `c_proj` (output projection)
  - MLP: `net.0` (expansion), `net.2` (projection)
- **Dropout**: 0.1
- **Trainable Parameters**: ~1.5M (1.3% of total)

### Training Configuration
- **Optimizer**: AdamW with betas=(0.9, 0.999)
- **Learning Rate**: 2.0e-5 (fixed to prevent catastrophic forgetting)
- **Weight Decay**: 0.05
- **Scheduler**: Cosine with 10% warmup
- **Gradient Clipping**: 1.0
- **Batch Size**: Hardware-adaptive (2-8)
- **Gradient Accumulation**: Hardware-adaptive (4-16)
- **Effective Batch Size**: 32

## 🗂️ Directory Structure

```
src/finetuning/
├── __init__.py           # Package initialization
├── base_trainer.py       # Core LoRA trainer with shared logic
├── data_utils.py         # CoT data formatting and utilities
├── gpu_finetune.py       # GPU script (CUDA/MPS auto-detect)
├── cpu_finetune.py       # CPU-optimized script
└── README.md            # This file
```

## 📊 Data Format

### Chain-of-Thought (CoT) Format

The fine-tuning suite uses a special messaging format with explicit reasoning steps:

#### Special Tokens
- `<|im_start|>`: Start of message
- `<|im_end|>`: End of message
- Roles: `system`, `user`, `thought`, `assistant`

#### Format 1: Messages (Standard)

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant..."},
    {"role": "user", "content": "What is 2+2?"},
    {"role": "thought", "content": "Let me add these numbers: 2 + 2 = 4"},
    {"role": "assistant", "content": "The answer is 4."}
  ]
}
```

#### Format 2: Simplified

```json
{
  "instruction": "What is 2+2?",
  "thought": "Let me add these numbers: 2 + 2 = 4",
  "response": "The answer is 4."
}
```

#### Formatted Output

```
<|im_start|>system
You are a helpful, creative, and clever AI assistant. When a user asks a question, provide a clear and concise answer. If the question involves logic, think through it step-by-step using a 'thought' block. If the user asks for code, provide clean examples in Markdown. Admit if you are unsure of a fact.<|im_end|>
<|im_start|>user
What is 2+2?<|im_end|>
<|im_start|>thought
Let me add these numbers: 2 + 2 = 4<|im_end|>
<|im_start|>assistant
The answer is 4.<|im_end|>
```

### System Prompt

The embedded system prompt mandates step-by-step reasoning:

> "You are a helpful, creative, and clever AI assistant. When a user asks a question, provide a clear and concise answer. If the question involves logic, think through it step-by-step using a 'thought' block. If the user asks for code, provide clean examples in Markdown. Admit if you are unsure of a fact."

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install torch transformers peft tqdm
```

### 2. Prepare Training Data

Create your CoT training data in JSONL format:

```bash
# Create sample data for testing
python -c "from src.finetuning.data_utils import create_sample_cot_data; \
create_sample_cot_data('data/cot_train.jsonl', 1000); \
create_sample_cot_data('data/cot_val.jsonl', 200)"
```

Or prepare your own data:

```python
from src.finetuning.data_utils import validate_cot_format

# Validate your data format
validate_cot_format('data/cot_train.jsonl')
```

### 3. Run Fine-Tuning

#### GPU (CUDA or MPS - Auto-Detect)

```bash
python src/finetuning/gpu_finetune.py
```

The script automatically detects:
- **NVIDIA CUDA**: Uses FP16 mixed precision, batch size 8
- **Apple MPS**: Disables mixed precision, batch size 4

#### CPU (Not Recommended)

```bash
python src/finetuning/cpu_finetune.py
```

⚠️ **Warning**: CPU training is 10-50x slower than GPU. Use for testing only.

### 4. Use Fine-Tuned Model

```python
import torch
from src.model import DecoderOnlyTransformer
from peft import PeftModel
from src.tokenizer_utils import load_tokenizer

# Load base model
model = DecoderOnlyTransformer(
    vocab_size=100277,
    d_model=768,
    num_heads=12,
    num_layers=12,
    context_length=512,
    dropout=0.1,
)

# Load LoRA weights
model = PeftModel.from_pretrained(model, 'finetuned_model_gpu/best_model')
model.eval()

# Load tokenizer
tokenizer = load_tokenizer()

# Generate
prompt = "<|im_start|>user\nWhat is 5 * 7?<|im_end|>\n<|im_start|>thought\n"
input_ids = tokenizer.encode(prompt, return_tensors='pt')

output = model.generate(
    input_ids,
    max_new_tokens=100,
    temperature=0.7,
    top_p=0.9,
)

print(tokenizer.decode(output[0]))
```

## ⚙️ Configuration

### Modify Training Settings

Edit the configuration dictionaries in `gpu_finetune.py` or `cpu_finetune.py`:

```python
# Training configuration
TRAIN_CONFIG = {
    'batch_size': 8,              # Adjust based on GPU memory
    'gradient_accumulation_steps': 4,
    'num_epochs': 3,              # Number of training epochs
    'learning_rate': 2.0e-5,      # Fixed LR (don't change!)
    'weight_decay': 0.05,
    'warmup_ratio': 0.1,
    'max_grad_norm': 1.0,
    'logging_steps': 10,
    'eval_steps': 100,
    'save_steps': 500,
}

# LoRA configuration
LORA_CONFIG = {
    'lora_r': 16,                 # LoRA rank
    'lora_alpha': 32,             # LoRA alpha
    'lora_dropout': 0.1,          # LoRA dropout
}
```

### Hardware-Specific Recommendations

#### NVIDIA GPU (CUDA)
- **Batch Size**: 8-16 (depending on VRAM)
- **Mixed Precision**: Enabled (FP16)
- **Gradient Accumulation**: 2-4
- **Expected Speed**: ~1000 tokens/sec

#### Apple Silicon (MPS)
- **Batch Size**: 4-8
- **Mixed Precision**: Disabled (stability)
- **Gradient Accumulation**: 4-8
- **Expected Speed**: ~300-500 tokens/sec

#### CPU
- **Batch Size**: 2-4
- **Mixed Precision**: Disabled
- **Gradient Accumulation**: 8-16
- **Expected Speed**: ~50-100 tokens/sec

## 📈 Training Monitoring

### Checkpoints

Checkpoints are saved to `finetuned_model_gpu/` or `finetuned_model_cpu/`:

```
finetuned_model_gpu/
├── best_model/              # Best model based on validation loss
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   └── training_state.pt
├── checkpoint_step_500/     # Regular checkpoints
├── checkpoint_step_1000/
└── checkpoint_epoch_1/      # End-of-epoch checkpoints
```

### Logs

Training progress is logged to console with:
- Training loss
- Learning rate
- Validation loss (every `eval_steps`)
- Best model updates

## 🎓 Best Practices

### 1. Data Quality
- **Diverse Examples**: Include math, logic, code, and general knowledge
- **Explicit Reasoning**: Always include `thought` blocks for complex questions
- **Code Formatting**: Use Markdown code blocks for code examples
- **Balanced Dataset**: Mix simple and complex examples

### 2. Training Strategy
- **Start Small**: Test with 100-1000 examples first
- **Monitor Validation**: Watch for overfitting (train loss << val loss)
- **Fixed Learning Rate**: Don't change 2.0e-5 (prevents catastrophic forgetting)
- **Gradual Scaling**: Start with 1 epoch, then increase if needed

### 3. Preventing Language Bleed
- **Consistent Format**: Always use `<|im_start|>` and `<|im_end|>` tokens
- **EOS Token**: Ensure tokenizer has proper EOS token
- **Validation**: Check generated samples during training
- **Temperature**: Use 0.7-0.9 for generation (not too high)

### 4. Memory Optimization
- **Gradient Checkpointing**: Enable in base model for larger batches
- **Gradient Accumulation**: Increase if OOM errors occur
- **Batch Size**: Reduce if GPU memory is insufficient
- **Context Length**: Reduce from 512 to 256 if needed

## 🔧 Troubleshooting

### Out of Memory (OOM)

```python
# Reduce batch size
TRAIN_CONFIG['batch_size'] = 4

# Increase gradient accumulation
TRAIN_CONFIG['gradient_accumulation_steps'] = 8

# Enable gradient checkpointing in model
model = DecoderOnlyTransformer(..., use_gradient_checkpointing=True)
```

### Slow Training

```python
# Increase batch size (if memory allows)
TRAIN_CONFIG['batch_size'] = 16

# Reduce logging frequency
TRAIN_CONFIG['logging_steps'] = 50

# Reduce evaluation frequency
TRAIN_CONFIG['eval_steps'] = 500
```

### Poor Generation Quality

1. **Check Data Format**: Validate with `validate_cot_format()`
2. **Increase Training**: Try more epochs or more data
3. **Adjust Temperature**: Use 0.7-0.9 for generation
4. **Check EOS Token**: Ensure proper tokenizer configuration

### Catastrophic Forgetting

1. **Don't Change LR**: Keep at 2.0e-5
2. **Reduce Epochs**: Try 1-2 epochs first
3. **Smaller LoRA Rank**: Try r=8 instead of r=16
4. **More Diverse Data**: Include general knowledge examples

## 📚 Advanced Usage

### Custom LoRA Configuration

```python
from src.finetuning.base_trainer import LoRAFineTuner

trainer = LoRAFineTuner(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    lora_r=8,           # Smaller rank
    lora_alpha=16,      # Adjust alpha
    lora_dropout=0.05,  # Lower dropout
)
```

### Resume from Checkpoint

```python
trainer = LoRAFineTuner(...)
trainer.load_checkpoint('finetuned_model_gpu/checkpoint_step_1000')
trainer.train()
```

### Merge LoRA Weights

```python
from peft import PeftModel

# Load model with LoRA
model = PeftModel.from_pretrained(base_model, 'finetuned_model_gpu/best_model')

# Merge LoRA weights into base model
model = model.merge_and_unload()

# Save merged model
torch.save(model.state_dict(), 'merged_model.pt')
```

## 📊 Expected Results

### Training Metrics
- **Initial Loss**: ~2.1 (pre-trained)
- **Target Loss**: ~1.5-1.8 (after fine-tuning)
- **Training Time**: 
  - GPU: 2-4 hours (1000 examples, 3 epochs)
  - CPU: 20-40 hours (not recommended)

### Generation Quality
- **Reasoning**: Clear step-by-step thought processes
- **Code**: Clean, well-formatted code examples
- **Accuracy**: Improved on math and logic tasks
- **No Language Bleed**: Proper use of special tokens

## 🤝 Contributing

To add new features or improve the fine-tuning suite:

1. **New Data Formats**: Extend `data_utils.py`
2. **New Hardware**: Add device-specific optimizations to `base_trainer.py`
3. **New Training Strategies**: Modify training loop in `base_trainer.py`

## 📄 License

This fine-tuning suite is part of the 117M Transformer project.

## 🙏 Acknowledgments

- **LoRA**: [Hu et al., 2021](https://arxiv.org/abs/2106.09685)
- **Chain-of-Thought**: [Wei et al., 2022](https://arxiv.org/abs/2201.11903)
- **PEFT Library**: [Hugging Face](https://github.com/huggingface/peft)

---

**Questions?** Check the main project README or open an issue.
