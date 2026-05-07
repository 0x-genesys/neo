# Fine-Tuning Guide: 117M Transformer with LoRA + Chain-of-Thought

Complete guide for fine-tuning the 117M parameter Transformer into a general-purpose chatbot with reasoning capabilities.

## 📋 Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Data Preparation](#data-preparation)
4. [Training](#training)
5. [Evaluation](#evaluation)
6. [Deployment](#deployment)
7. [Troubleshooting](#troubleshooting)
8. [Advanced Topics](#advanced-topics)

## 🎯 Overview

### What is LoRA?

**LoRA (Low-Rank Adaptation)** is a parameter-efficient fine-tuning technique that:
- Freezes the pre-trained model weights
- Adds small trainable rank decomposition matrices
- Reduces trainable parameters by ~99% (117M → 1.5M)
- Maintains model quality while being memory-efficient

### What is Chain-of-Thought (CoT)?

**Chain-of-Thought** is a prompting technique that:
- Encourages step-by-step reasoning
- Makes the model's thinking process explicit
- Improves accuracy on complex tasks
- Reduces hallucinations and errors

### Goals

Transform the pre-trained 117M Transformer into a chatbot that:
- ✅ Excels at reasoning and logic
- ✅ Handles code generation without language bleed
- ✅ Provides clear, step-by-step explanations
- ✅ Admits uncertainty when appropriate

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Key dependencies:
- `peft>=0.5.0` - LoRA implementation
- `transformers>=4.30.0` - Model utilities
- `torch>=2.0.0` - PyTorch

### 2. Prepare Data

Create a complete dataset with math, code, and Q&A examples:

```bash
python scripts/prepare_finetuning_data.py all \
    --output-dir data \
    --math-samples 1000 \
    --code-samples 500 \
    --qa-samples 500 \
    --val-ratio 0.1
```

This creates:
- `data/cot_train.jsonl` - Training data (1800 examples)
- `data/cot_val.jsonl` - Validation data (200 examples)

### 3. Run Fine-Tuning

**Automatic (Recommended)**:
```bash
bash scripts/quick_finetune.sh
```

**Manual**:
```bash
# GPU (CUDA or MPS)
python src/finetuning/gpu_finetune.py

# CPU (not recommended)
python src/finetuning/cpu_finetune.py
```

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

# Generate with CoT
prompt = """<|im_start|>user
What is 15 * 23?<|im_end|>
<|im_start|>thought
"""

input_ids = tokenizer.encode(prompt, return_tensors='pt')
output = model.generate(input_ids, max_new_tokens=100, temperature=0.7)
print(tokenizer.decode(output[0]))
```

## 📊 Data Preparation

### Data Format

The fine-tuning suite uses a special messaging format with Chain-of-Thought reasoning:

#### Special Tokens
- `<|im_start|>` - Start of message
- `<|im_end|>` - End of message
- Roles: `system`, `user`, `thought`, `assistant`

#### Example 1: Math Reasoning

```json
{
  "instruction": "What is 15 * 23?",
  "thought": "Let me multiply these numbers step by step. 15 * 23 = 15 * 20 + 15 * 3 = 300 + 45 = 345",
  "response": "The answer is 345."
}
```

#### Example 2: Code Generation

```json
{
  "instruction": "Write a Python function to check if a number is prime.",
  "thought": "To check if a number is prime, I need to test if it's divisible by any number from 2 to sqrt(n). I'll handle edge cases for numbers less than 2.",
  "response": "```python\ndef is_prime(n):\n    \"\"\"Check if a number is prime.\"\"\"\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True\n```"
}
```

#### Example 3: General Q&A

```json
{
  "instruction": "What causes seasons on Earth?",
  "thought": "Seasons are caused by Earth's axial tilt of about 23.5 degrees. As Earth orbits the Sun, different hemispheres receive more direct sunlight at different times of the year.",
  "response": "Seasons on Earth are caused by the planet's axial tilt of approximately 23.5 degrees. As Earth orbits the Sun, this tilt causes different hemispheres to receive varying amounts of direct sunlight throughout the year, creating seasonal changes."
}
```

### Creating Custom Data

#### Option 1: Use Preparation Script

```bash
# Math reasoning
python scripts/prepare_finetuning_data.py math \
    --output data/math_cot.jsonl \
    --num-samples 1000

# Code generation
python scripts/prepare_finetuning_data.py code \
    --output data/code_cot.jsonl \
    --num-samples 500

# General Q&A
python scripts/prepare_finetuning_data.py qa \
    --output data/qa_cot.jsonl \
    --num-samples 500

# Merge datasets
python scripts/prepare_finetuning_data.py merge \
    --inputs data/math_cot.jsonl data/code_cot.jsonl data/qa_cot.jsonl \
    --output data/cot_all.jsonl

# Split into train/val
python scripts/prepare_finetuning_data.py split \
    --input data/cot_all.jsonl \
    --train data/cot_train.jsonl \
    --val data/cot_val.jsonl \
    --val-ratio 0.1
```

#### Option 2: Manual Creation

Create a JSONL file with one example per line:

```python
import json

examples = [
    {
        "instruction": "Your question here",
        "thought": "Step-by-step reasoning",
        "response": "Final answer"
    },
    # ... more examples
]

with open('data/custom_cot.jsonl', 'w') as f:
    for example in examples:
        f.write(json.dumps(example) + '\n')
```

#### Option 3: Convert Existing Dataset

```python
from src.finetuning.data_utils import validate_cot_format

# Convert your dataset to CoT format
def convert_to_cot(input_path, output_path):
    with open(input_path, 'r') as f_in, open(output_path, 'w') as f_out:
        for line in f_in:
            data = json.loads(line)
            
            # Add thought process
            cot_example = {
                "instruction": data['question'],
                "thought": f"Let me think about this... {data.get('reasoning', '')}",
                "response": data['answer']
            }
            
            f_out.write(json.dumps(cot_example) + '\n')

# Validate format
validate_cot_format('data/custom_cot.jsonl')
```

### Data Quality Guidelines

1. **Diverse Examples**: Include various domains (math, code, science, etc.)
2. **Explicit Reasoning**: Always include `thought` blocks for complex questions
3. **Correct Answers**: Verify all responses are accurate
4. **Balanced Difficulty**: Mix simple and complex examples
5. **Code Formatting**: Use Markdown code blocks for code
6. **Consistent Style**: Maintain consistent formatting across examples

## 🎓 Training

### Hardware Requirements

#### Minimum
- **GPU**: 8GB VRAM (NVIDIA or Apple Silicon)
- **RAM**: 16GB
- **Storage**: 10GB

#### Recommended
- **GPU**: 16GB+ VRAM
- **RAM**: 32GB
- **Storage**: 50GB

#### Supported Hardware
- ✅ NVIDIA GPU (CUDA) - Best performance
- ✅ Apple Silicon (MPS) - Good performance
- ⚠️ CPU - Very slow (testing only)

### Training Configuration

Edit `config/finetuning_config.yaml` or modify the scripts directly:

```yaml
# LoRA Configuration
lora:
  r: 16                    # Rank (lower = fewer parameters)
  alpha: 32                # Scaling factor
  dropout: 0.1             # Dropout rate

# Training Configuration
training:
  batch_size: 8            # Per-device batch size
  gradient_accumulation_steps: 4
  num_epochs: 3
  learning_rate: 2.0e-5    # DO NOT CHANGE
  weight_decay: 0.05
  warmup_ratio: 0.1
```

### Training Process

#### Step 1: Start Training

```bash
# Automatic hardware detection
bash scripts/quick_finetune.sh

# Or manual
python src/finetuning/gpu_finetune.py
```

#### Step 2: Monitor Progress

Training logs show:
- **Training Loss**: Should decrease from ~2.1 to ~1.5-1.8
- **Validation Loss**: Should follow training loss
- **Learning Rate**: Warms up then decays
- **Step/Epoch**: Current progress

Example output:
```
Epoch 1/3 | Step 100/1500 | Loss: 1.85 | LR: 1.8e-05
📊 Step 100 | Val Loss: 1.92
✅ New best model saved! (loss: 1.92)
```

#### Step 3: Checkpoints

Checkpoints are saved to `finetuned_model_gpu/`:
- `best_model/` - Best model based on validation loss
- `checkpoint_step_500/` - Regular checkpoints
- `checkpoint_epoch_1/` - End-of-epoch checkpoints

### Training Time Estimates

| Hardware | Dataset Size | Time per Epoch | Total Time (3 epochs) |
|----------|--------------|----------------|----------------------|
| NVIDIA RTX 3090 | 2000 examples | 30 min | 1.5 hours |
| NVIDIA RTX 4090 | 2000 examples | 20 min | 1 hour |
| Apple M1 Max | 2000 examples | 60 min | 3 hours |
| Apple M2 Ultra | 2000 examples | 40 min | 2 hours |
| CPU (16 cores) | 2000 examples | 8 hours | 24 hours |

### Hyperparameter Tuning

#### Learning Rate
- **Default**: 2.0e-5 (DO NOT CHANGE)
- **Why**: Prevents catastrophic forgetting of pre-training
- **If needed**: Try 1.5e-5 or 2.5e-5 (small adjustments only)

#### LoRA Rank
- **Default**: 16
- **Lower (8)**: Fewer parameters, faster training, may reduce quality
- **Higher (32)**: More parameters, slower training, may improve quality

#### Batch Size
- **CUDA**: 8-16 (depending on VRAM)
- **MPS**: 4-8
- **CPU**: 2-4
- **Rule**: Larger is better, but watch for OOM errors

#### Epochs
- **Default**: 3
- **Fewer (1-2)**: Faster, may underfit
- **More (4-5)**: Slower, may overfit

## 📈 Evaluation

### Validation Metrics

Monitor these during training:
- **Validation Loss**: Should decrease and stabilize
- **Training Loss**: Should be close to validation loss
- **Overfitting**: If train loss << val loss, reduce epochs

### Qualitative Evaluation

Test the model on sample prompts:

```python
test_prompts = [
    "What is 17 * 23?",
    "Write a Python function to reverse a string",
    "Explain photosynthesis",
    "Is 17 prime?",
]

for prompt in test_prompts:
    formatted = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>thought\n"
    # Generate and evaluate
```

### Expected Results

After fine-tuning:
- ✅ Clear step-by-step reasoning
- ✅ Accurate math calculations
- ✅ Clean code generation
- ✅ No language bleed (proper token usage)
- ✅ Admits uncertainty when appropriate

## 🚀 Deployment

### Option 1: Use with LoRA Adapter

```python
from peft import PeftModel

# Load base model
base_model = DecoderOnlyTransformer(...)

# Load LoRA adapter (lightweight)
model = PeftModel.from_pretrained(base_model, 'finetuned_model_gpu/best_model')
```

**Pros**: Small file size (~6MB), easy to swap adapters
**Cons**: Slightly slower inference

### Option 2: Merge LoRA Weights

```python
from peft import PeftModel

# Load with LoRA
model = PeftModel.from_pretrained(base_model, 'finetuned_model_gpu/best_model')

# Merge weights
model = model.merge_and_unload()

# Save merged model
torch.save(model.state_dict(), 'merged_model.pt')
```

**Pros**: Faster inference, single file
**Cons**: Larger file size (~470MB)

### Option 3: Quantization

```python
# Quantize to INT8 for smaller size
import torch.quantization as quantization

model = quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
```

**Pros**: 4x smaller, faster on CPU
**Cons**: Slight quality loss

## 🔧 Troubleshooting

### Out of Memory (OOM)

**Symptoms**: CUDA/MPS out of memory error

**Solutions**:
1. Reduce batch size: `batch_size: 4`
2. Increase gradient accumulation: `gradient_accumulation_steps: 8`
3. Enable gradient checkpointing: `use_gradient_checkpointing: true`
4. Reduce context length: `max_length: 256`

### Slow Training

**Symptoms**: Training takes too long

**Solutions**:
1. Increase batch size (if memory allows)
2. Reduce logging frequency: `logging_steps: 50`
3. Use GPU instead of CPU
4. Enable mixed precision: `use_amp: true`

### Poor Generation Quality

**Symptoms**: Model generates nonsense or doesn't follow CoT format

**Solutions**:
1. Check data format: `validate_cot_format('data/cot_train.jsonl')`
2. Increase training epochs: `num_epochs: 5`
3. Add more diverse examples
4. Check tokenizer has special tokens
5. Verify EOS token is set correctly

### Catastrophic Forgetting

**Symptoms**: Model forgets pre-training knowledge

**Solutions**:
1. Keep learning rate at 2.0e-5
2. Reduce number of epochs
3. Use smaller LoRA rank: `lora_r: 8`
4. Add general knowledge examples to dataset

### Language Bleed

**Symptoms**: Model doesn't use special tokens correctly

**Solutions**:
1. Ensure all examples use `<|im_start|>` and `<|im_end|>`
2. Check tokenizer has special tokens added
3. Verify system prompt is included
4. Use lower temperature for generation: `temperature: 0.7`

## 🎯 Advanced Topics

### Multi-Task Fine-Tuning

Train on multiple tasks simultaneously:

```python
# Create task-specific datasets
math_dataset = CoTDataset('data/math_cot.jsonl', ...)
code_dataset = CoTDataset('data/code_cot.jsonl', ...)
qa_dataset = CoTDataset('data/qa_cot.jsonl', ...)

# Combine with task prefixes
combined_dataset = ConcatDataset([math_dataset, code_dataset, qa_dataset])
```

### Curriculum Learning

Train on progressively harder examples:

```python
# Start with simple examples
trainer = LoRAFineTuner(model, simple_dataset, ...)
trainer.train()

# Continue with complex examples
trainer = LoRAFineTuner(model, complex_dataset, ...)
trainer.train()
```

### Few-Shot Adaptation

Fine-tune on a small number of examples:

```python
# Use very small dataset (10-100 examples)
# Reduce epochs to 1-2
# Use smaller learning rate: 1.0e-5
```

### Domain-Specific Fine-Tuning

Specialize for a specific domain:

```python
# Create domain-specific dataset (e.g., medical, legal, scientific)
# Include domain-specific system prompt
# Train for more epochs (5-10)
```

### Multi-GPU Training

```python
# Use DataParallel or DistributedDataParallel
model = nn.DataParallel(model)

# Or use accelerate
from accelerate import Accelerator
accelerator = Accelerator()
model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
```

## 📚 References

- [LoRA Paper](https://arxiv.org/abs/2106.09685) - Hu et al., 2021
- [Chain-of-Thought Paper](https://arxiv.org/abs/2201.11903) - Wei et al., 2022
- [PEFT Library](https://github.com/huggingface/peft) - Hugging Face
- [Transformers Library](https://github.com/huggingface/transformers) - Hugging Face

## 🤝 Contributing

To improve the fine-tuning suite:
1. Add new data generation templates
2. Implement new training strategies
3. Optimize for new hardware
4. Improve documentation

---

**Questions?** Check the main README or open an issue.
