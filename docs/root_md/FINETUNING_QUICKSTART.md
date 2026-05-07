# Fine-Tuning Quick Start Guide

## 🚀 Get Started in 5 Minutes

This guide will get you fine-tuning the 117M Transformer with LoRA and Chain-of-Thought in just a few commands.

## Prerequisites

- Python 3.8+
- 8GB+ RAM
- GPU recommended (CUDA or Apple Silicon)
- Pre-trained model checkpoint

## Step 1: Install Dependencies (1 minute)

```bash
pip install -r requirements.txt
```

This installs:
- `torch` - PyTorch
- `transformers` - Model utilities
- `peft` - LoRA implementation
- `tqdm` - Progress bars
- Other utilities

## Step 2: Test Setup (1 minute)

```bash
python scripts/test_finetuning_setup.py
```

This verifies:
- ✅ All packages installed
- ✅ Device detection working
- ✅ Model can be loaded
- ✅ Fine-tuning modules work
- ✅ Data creation works

Expected output:
```
🎉 All Tests Passed!
✅ Fine-tuning setup is ready!
```

## Step 3: Prepare Data (2 minutes)

```bash
python scripts/prepare_finetuning_data.py all \
    --output-dir data \
    --math-samples 1000 \
    --code-samples 500 \
    --qa-samples 500 \
    --val-ratio 0.1
```

This creates:
- `data/cot_train.jsonl` - 1800 training examples
- `data/cot_val.jsonl` - 200 validation examples

Data includes:
- Math reasoning (1000 examples)
- Code generation (500 examples)
- General Q&A (500 examples)

## Step 4: Run Fine-Tuning (1-3 hours)

```bash
bash scripts/quick_finetune.sh
```

Or manually:
```bash
# GPU (CUDA or MPS)
python src/finetuning/gpu_finetune.py

# CPU (not recommended)
python src/finetuning/cpu_finetune.py
```

Training progress:
```
Epoch 1/3 | Step 100/450 | Loss: 1.85 | LR: 1.8e-05
📊 Step 100 | Val Loss: 1.92
✅ New best model saved! (loss: 1.92)
```

## Step 5: Use Fine-Tuned Model

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
model = PeftModel.from_pretrained(
    model, 
    'finetuned_model_gpu/best_model'
)
model.eval()

# Load tokenizer
tokenizer = load_tokenizer()

# Generate with Chain-of-Thought
prompt = """<|im_start|>user
What is 17 * 23?<|im_end|>
<|im_start|>thought
"""

input_ids = tokenizer.encode(prompt, return_tensors='pt')
output = model.generate(
    input_ids,
    max_new_tokens=100,
    temperature=0.7,
    top_p=0.9,
)

print(tokenizer.decode(output[0]))
```

Expected output:
```
<|im_start|>user
What is 17 * 23?<|im_end|>
<|im_start|>thought
Let me multiply these numbers step by step.
17 * 23 = 17 * 20 + 17 * 3 = 340 + 51 = 391<|im_end|>
<|im_start|>assistant
The answer is 391.<|im_end|>
```

## 📊 What You Get

### Model Capabilities
- ✅ Step-by-step reasoning
- ✅ Accurate math calculations
- ✅ Clean code generation
- ✅ No language bleed
- ✅ Admits uncertainty

### Files Created
```
finetuned_model_gpu/
├── best_model/              # Best model (use this!)
│   ├── adapter_config.json
│   ├── adapter_model.bin    # LoRA weights (~6MB)
│   └── training_state.pt
├── checkpoint_step_500/     # Regular checkpoints
└── checkpoint_epoch_1/      # Epoch checkpoints
```

### Training Metrics
- **Initial Loss**: ~2.1 (pre-trained)
- **Final Loss**: ~1.5-1.8 (fine-tuned)
- **Trainable Params**: 1.5M (1.3% of 117M)
- **Training Time**: 1-3 hours (GPU)

## 🎯 Configuration

### Quick Adjustments

Edit `src/finetuning/gpu_finetune.py`:

```python
# Reduce batch size if OOM
TRAIN_CONFIG = {
    'batch_size': 4,  # Default: 8
    'gradient_accumulation_steps': 8,  # Default: 4
}

# Train for more epochs
TRAIN_CONFIG = {
    'num_epochs': 5,  # Default: 3
}

# Use different checkpoint
CHECKPOINT_CONFIG = {
    'pretrained_model': 'path/to/your/checkpoint.pt',
}
```

### Hardware-Specific Settings

#### NVIDIA GPU (16GB+)
```python
batch_size: 16
gradient_accumulation_steps: 2
use_amp: True
```

#### NVIDIA GPU (8GB)
```python
batch_size: 4
gradient_accumulation_steps: 8
use_amp: True
```

#### Apple Silicon (M1/M2)
```python
batch_size: 4
gradient_accumulation_steps: 8
use_amp: False
```

## 🔧 Troubleshooting

### Out of Memory
```bash
# Reduce batch size
batch_size: 2
gradient_accumulation_steps: 16
```

### Slow Training
```bash
# Use GPU instead of CPU
# Or increase batch size if memory allows
batch_size: 16
```

### Poor Quality
```bash
# Train for more epochs
num_epochs: 5

# Or add more diverse data
python scripts/prepare_finetuning_data.py all --math-samples 2000
```

## 📚 Next Steps

### Learn More
- Read `docs/FINETUNING_GUIDE.md` for comprehensive guide
- Check `src/finetuning/README.md` for technical details
- Review `config/finetuning_config.yaml` for all options

### Customize
- Create custom data: `scripts/prepare_finetuning_data.py`
- Adjust LoRA config: `lora_r`, `lora_alpha`, `lora_dropout`
- Modify training: `learning_rate`, `num_epochs`, `batch_size`

### Deploy
- Merge LoRA weights for faster inference
- Quantize model for smaller size
- Export to ONNX for production

## 🎉 Success Checklist

- [ ] Dependencies installed
- [ ] Setup test passed
- [ ] Data prepared
- [ ] Training completed
- [ ] Model generates good outputs
- [ ] Checkpoints saved

## 💡 Tips

1. **Start Small**: Test with 100 examples first
2. **Monitor Validation**: Watch for overfitting
3. **Save Checkpoints**: Don't lose progress
4. **Use GPU**: 10-50x faster than CPU
5. **Fixed LR**: Don't change 2.0e-5

## 🆘 Getting Help

- Check `docs/FINETUNING_GUIDE.md` for detailed troubleshooting
- Review error messages carefully
- Verify data format with `validate_cot_format()`
- Test with smaller dataset first

## 📊 Expected Timeline

| Step | Time | Hardware |
|------|------|----------|
| Install | 1 min | Any |
| Test | 1 min | Any |
| Prepare Data | 2 min | Any |
| Training | 1-3 hours | GPU |
| Training | 20-40 hours | CPU |

## 🚀 One-Liner

```bash
pip install -r requirements.txt && \
python scripts/test_finetuning_setup.py && \
python scripts/prepare_finetuning_data.py all && \
bash scripts/quick_finetune.sh
```

---

**Ready to fine-tune!** 🎓

For questions, check the comprehensive guide: `docs/FINETUNING_GUIDE.md`
