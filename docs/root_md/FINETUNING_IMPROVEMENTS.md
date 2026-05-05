# Fine-Tuning Improvements Summary

## Overview
This document summarizes the key improvements made to the fine-tuning pipeline to prevent "lazy" learning and improve model quality.

## 1. Label Masking (Prevent "Lazy" Learning)

### Problem
Previously, the model was trained on **all tokens** including System prompts and User questions. This caused the model to waste its limited 117M parameters learning to reconstruct user input rather than focusing on reasoning and responses.

### Solution
Implemented intelligent label masking in `src/finetuning/data_utils.py`:
- **Masked**: System prompts and User questions (set to `-100`)
- **Trained**: Only Thought processes and Assistant responses

### Impact
- Model focuses its learning capacity on generating quality responses
- Prevents "lazy" behavior where model just echoes user input
- Improves reasoning capability by focusing on Thought tokens

### Code Location
`src/finetuning/data_utils.py` - `CoTDataset.__getitem__()` method

## 2. Expanded LoRA Target Modules

### Problem
Original configuration only targeted attention layers (`c_attn`, `c_proj`, `lm_head`), missing the MLP/FFN layers where much of the reasoning happens.

### Solution
Expanded target modules to include MLP layers:
```python
target_modules = ["c_attn", "c_proj", "net.0", "net.2", "lm_head"]
```

Where:
- `c_attn`: Attention QKV projection
- `c_proj`: Attention output projection
- `net.0`: MLP first linear layer (768 → 3072)
- `net.2`: MLP second linear layer (3072 → 768)
- `lm_head`: Output projection to vocabulary

### Impact
- Increased trainable parameters from ~1.5% to ~2.5% of total
- Better reasoning capability through MLP adaptation
- More expressive model without full fine-tuning cost

### Code Location
`src/finetuning/base_trainer.py` - `_apply_lora()` method

## 3. EOS Token Enforcement

### Problem
Training data might not consistently end with `<|im_end|>` token, causing model to "babble" and not know when to stop generating.

### Solution
Ensured all training examples end with `<|im_end|>`:
- Added explicit check in `_format_conversation()`
- Appends `<|im_end|>` if missing
- Teaches model proper stopping behavior

### Impact
- Model learns when to stop generating
- Prevents infinite generation loops
- Cleaner, more controlled outputs

### Code Location
`src/finetuning/data_utils.py` - `_format_conversation()` method

## 4. Auto-Detection of Ignore Index

### Problem
Base training and fine-tuning use different padding strategies:
- Base training: No padding (continuous sequences)
- Fine-tuning: `-100` padding (HuggingFace standard)

### Solution
Model automatically detects which mode it's in:
```python
if (targets == -100).any():
    ignore_index = -100  # Finetuning mode
else:
    ignore_index = -1    # Base training mode
```

### Impact
- Same model code works for both base training and fine-tuning
- No configuration needed
- Prevents CUDA assertion errors

### Code Location
`src/model.py` - `forward()` method

## Training Configuration

### Current Setup
- **Model**: 117M parameters (162M total with embeddings)
- **LoRA Rank**: 16
- **LoRA Alpha**: 32
- **LoRA Dropout**: 0.1
- **Trainable Parameters**: ~2.5M (1.52% of total)
- **Learning Rate**: 2.0e-5 (fixed, prevents catastrophic forgetting)
- **Batch Size**: 8 (effective 32 with gradient accumulation)
- **Epochs**: 3

### Dataset
- **microsoft/orca-math-word-problems-200k**: 5,000 samples (Logic & Reasoning)
- **databricks/databricks-dolly-15k**: 15,000 samples (General Assistant)
- **sahil2801/CodeAlpaca-20k**: 5,000 samples (Coding & Technical)
- **Total**: ~25,000 samples with 90/10 train/val split

## Results

### Before Improvements
- Model learned to echo user input
- Poor reasoning capability
- Inconsistent stopping behavior

### After Improvements
- Focused learning on responses
- Better reasoning through MLP adaptation
- Clean, controlled generation with proper stopping

## Usage

### Training
```bash
python src/finetuning/gpu_finetune.py \
    --config config/auto_training_117m_balanced.yaml \
    --model-remote final_mode_1_may.pt \
    --upload
```

### Inference
```bash
python src/finetuning/chat_inference.py \
    --base-model checkpoints/best_model.pt \
    --lora-weights finetuned_model_gpu/best_model
```

## Future Improvements

1. **Dynamic Masking**: Experiment with masking ratios
2. **Curriculum Learning**: Gradually increase task difficulty
3. **Multi-Task Training**: Mix different task types
4. **Longer Context**: Increase from 512 to 1024 tokens
5. **Larger LoRA Rank**: Try r=32 or r=64 for more capacity

## References

- PEFT Documentation: https://huggingface.co/docs/peft
- LoRA Paper: https://arxiv.org/abs/2106.09685
- Instruction Tuning Best Practices: https://arxiv.org/abs/2308.10792
