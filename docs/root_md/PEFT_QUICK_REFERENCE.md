# PEFT Quick Reference Guide

**Quick guide for using PEFT-compliant models**

---

## 🚀 Quick Start

### Fine-Tuning
```bash
# With explicit config (recommended)
python src/finetuning/gpu_finetune.py \
    --config config/auto_training_117m_balanced.yaml \
    --model checkpoints/best_model.pt \
    --train-data data/cot_train.jsonl \
    --val-data data/cot_val.jsonl \
    --epochs 3

# Config auto-loaded from checkpoint
python src/finetuning/gpu_finetune.py \
    --model checkpoints/best_model.pt \
    --train-data data/cot_train.jsonl \
    --val-data data/cot_val.jsonl \
    --epochs 3
```

### Inference
```bash
# Interactive mode with remote models (easiest)
python src/finetuning/chat_inference.py --interactive

# Interactive mode with local models
python src/finetuning/chat_inference.py \
    --base-model checkpoints/best_model.pt \
    --adapter finetuned_model_gpu/best_model \
    --interactive

# Single prompt
python src/finetuning/chat_inference.py \
    --base-model checkpoints/best_model.pt \
    --adapter finetuned_model_gpu/best_model \
    --prompt "What is 2+2?" \
    --show-thought

# With explicit config
python src/finetuning/chat_inference.py \
    --config config/auto_training_117m_balanced.yaml \
    --base-model checkpoints/best_model.pt \
    --adapter finetuned_model_gpu/best_model \
    --interactive
```

---

## 📋 PEFT Compliance Checklist

### Model Requirements
- ✅ `config` attribute with model architecture details
- ✅ `generation_config` attribute with generation parameters
- ✅ `main_input_name = "input_ids"`
- ✅ `forward()` accepts `input_ids` parameter

### Config Requirements
```python
config = SimpleNamespace(
    vocab_size=100277,              # Required
    hidden_size=768,                # Required (d_model)
    num_hidden_layers=12,           # Required (num_layers)
    num_attention_heads=12,         # Required (num_heads)
    max_position_embeddings=512,    # Required (context_length)
    model_type="gpt",               # Required
    is_encoder_decoder=False,       # Required
)
```

---

## 🔧 Common Tasks

### Create PEFT-Compliant Model
```python
from src.model import create_model
import yaml

# Load config
with open('config/auto_training_117m_balanced.yaml') as f:
    config = yaml.safe_load(f)

# Create model (PEFT attributes added automatically)
model = create_model(config)

# Verify
assert hasattr(model, 'config')
assert hasattr(model, 'generation_config')
assert hasattr(model, 'main_input_name')
```

### Apply LoRA
```python
from peft import LoraConfig, get_peft_model, TaskType

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["c_attn", "c_proj", "net.0", "net.2", "lm_head"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

peft_model = get_peft_model(model, lora_config)
```

### Save Adapter
```python
# Save LoRA adapter
peft_model.save_pretrained("my_adapter")

# Files created:
# my_adapter/
# ├── adapter_config.json
# ├── adapter_model.safetensors
# └── README.md
```

### Load Adapter
```python
from peft import PeftModel

# Load base model
base_model = create_model(config)
checkpoint = torch.load('checkpoints/best_model.pt')
base_model.load_state_dict(checkpoint['model_state_dict'])

# Load adapter
model = PeftModel.from_pretrained(base_model, "my_adapter")
```

### Migrate Old Checkpoint
```python
# Load old checkpoint
checkpoint = torch.load('old_checkpoint.pt')

# Create new PEFT-compliant model
model = create_model(config)
model.load_state_dict(checkpoint['model_state_dict'])

# Save with PEFT attributes
torch.save({
    'model_state_dict': model.state_dict(),
    'config': config,
}, 'new_checkpoint.pt')
```

---

## ⚠️ Common Issues

### Issue: "Model must have 'config' attribute"
**Cause**: Using old checkpoint without PEFT attributes

**Solution**: Re-save checkpoint with new model
```python
from src.model import create_model
model = create_model(config)
model.load_state_dict(old_checkpoint['model_state_dict'])
torch.save({'model_state_dict': model.state_dict(), 'config': config}, 'new.pt')
```

### Issue: "AttributeError: 'DecoderOnlyTransformer' object has no attribute 'config'"
**Cause**: Model created before PEFT compliance update

**Solution**: Use `create_model()` factory function
```python
# Wrong
model = DecoderOnlyTransformer(vocab_size, d_model, ...)

# Right
from src.model import create_model
model = create_model(config)
```

### Issue: "Config vocab_size mismatch"
**Cause**: Adapter trained with different vocab size

**Solution**: Ensure base model and adapter use same config
```python
# Check adapter config
import json
with open('adapter/adapter_config.json') as f:
    adapter_config = json.load(f)
    print(f"Adapter vocab_size: {adapter_config.get('vocab_size', 'N/A')}")

# Check base model config
print(f"Base model vocab_size: {model.config.vocab_size}")
```

### Issue: "LoRA weights not being applied"
**Cause**: Using base model's generate() instead of PEFT model's forward()

**Solution**: Use custom generation loop (already implemented in chat_inference.py)
```python
# Wrong
output = base_model.generate(input_ids)

# Right
output = peft_model(input_ids=input_ids)  # Forward pass includes LoRA
```

---

## 📊 Configuration Reference

### 117M Model (Standard)
```yaml
model:
  vocab_size: 100277      # tiktoken cl100k_base
  d_model: 768            # GPT-2 Small
  num_heads: 12           # head_dim = 64
  num_layers: 12          # Good depth
  context_length: 512     # Standard
  dropout: 0.12           # Moderate
```

### LoRA Configuration (Recommended)
```python
lora_config = LoraConfig(
    r=16,                   # Rank
    lora_alpha=32,          # Alpha (2x rank)
    lora_dropout=0.1,       # Dropout
    target_modules=[        # All linear layers
        "c_attn",           # Attention QKV
        "c_proj",           # Attention output
        "net.0",            # MLP first layer
        "net.2",            # MLP second layer
        "lm_head",          # Output projection
    ],
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    init_lora_weights=True,
)
```

### Training Configuration (Recommended)
```python
trainer = LoRAFineTuner(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    batch_size=8,                    # GPU: 8, MPS: 4, CPU: 2
    gradient_accumulation_steps=4,   # Effective batch = 32
    num_epochs=3,
    learning_rate=2.0e-5,            # Fixed (prevents forgetting)
    weight_decay=0.05,
    warmup_ratio=0.1,                # 10% warmup
    max_grad_norm=1.0,
)
```

---

## 🎯 Best Practices

### 1. Always Use Factory Function
```python
# Good
from src.model import create_model
model = create_model(config)

# Bad
model = DecoderOnlyTransformer(vocab_size, d_model, ...)
```

### 2. Save Config with Checkpoint
```python
# Good
torch.save({
    'model_state_dict': model.state_dict(),
    'config': config,  # Include config
}, 'checkpoint.pt')

# Acceptable (config loaded from YAML)
torch.save({
    'model_state_dict': model.state_dict(),
}, 'checkpoint.pt')
```

### 3. Use Explicit Config for Inference
```bash
# Good (explicit config)
python src/finetuning/chat_inference.py \
    --config config/auto_training_117m_balanced.yaml \
    --base-model checkpoints/best_model.pt \
    --adapter finetuned_model/best_model \
    --interactive

# Acceptable (config from checkpoint)
python src/finetuning/chat_inference.py \
    --base-model checkpoints/best_model.pt \
    --adapter finetuned_model/best_model \
    --interactive
```

### 4. Validate Before Fine-Tuning
```python
# Check model has PEFT attributes
assert hasattr(model, 'config')
assert hasattr(model, 'generation_config')
assert hasattr(model, 'main_input_name')

# Check config values
assert model.config.vocab_size == 100277
assert model.config.hidden_size == 768
```

### 5. Use Consistent Vocab Size
```python
# Ensure tokenizer and model match
tokenizer = load_tokenizer()
assert len(tokenizer) == model.config.vocab_size
```

---

## 📚 Additional Resources

- **Full Documentation**: `PEFT_COMPLIANCE_COMPLETE.md`
- **Inference Flow**: `INFERENCE_FLOW_REVIEW.md`
- **Simplification Summary**: `SIMPLIFICATION_COMPLETE.md`
- **Test Script**: `test_peft_compliance.py`

---

## 🎉 Summary

### Key Points
1. ✅ Model is PEFT-compliant from creation
2. ✅ Use `create_model()` factory function
3. ✅ Config included automatically
4. ✅ No manual config creation needed
5. ✅ Clear error messages if incompatible

### Quick Commands
```bash
# Fine-tune
python src/finetuning/gpu_finetune.py \
    --config config/auto_training_117m_balanced.yaml \
    --model checkpoints/best_model.pt \
    --train-data data/cot_train.jsonl \
    --val-data data/cot_val.jsonl \
    --epochs 3

# Inference
python src/finetuning/chat_inference.py --interactive
```

**That's it! The model is PEFT-compliant and ready to use.**

