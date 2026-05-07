# Chat Inference - Special Token Issue

## Problem Identified

The debug output shows that `<|im_start|>` and `<|im_end|>` are being **split into individual character tokens** instead of being treated as single special tokens.

```
First 10 tokens: [27, 91, 318, 5011, 91, 29, 9125, 198, 2675, 527]
```

Where:
- `27` = `<`
- `91` = `|`
- `318` = `im`
- `5011` = `_start`
- etc.

## Root Cause

**This is actually CORRECT behavior for tiktoken cl100k_base!**

The `cl100k_base` encoding does NOT have `<|im_start|>` and `<|im_end|>` as special tokens. These ChatML tokens were added later for GPT-3.5-turbo and GPT-4.

The tokens that ARE in cl100k_base:
- `<|endoftext|>` ✅
- `<|fim_prefix|>` ✅
- `<|fim_middle|>` ✅
- `<|fim_suffix|>` ✅
- `<|endofprompt|>` ✅

But NOT:
- `<|im_start|>` ❌
- `<|im_end|>` ❌

## Why Model Isn't Responding

The model was **fine-tuned** with these tokens split into characters. This means:

1. **During fine-tuning**, the training data had `<|im_start|>` split as `['<', '|', 'im', '_start', '|', '>']`
2. **The model learned** to recognize this sequence of character tokens as message boundaries
3. **During inference**, the same splitting happens (which is correct)

**The issue is NOT the tokenization - it's that:**
- The model hasn't been fine-tuned yet, OR
- The fine-tuning didn't converge properly, OR  
- The model needs more training to learn the format

## Solutions

### Option 1: Check Fine-Tuning Status

```bash
# Check if fine-tuning completed
ls -la finetuned_model_gpu/best_model/

# Check training logs for final loss
# Loss should be ~2-3, not 10+
```

### Option 2: Use Model Without Chat Format

If the model isn't fine-tuned for chat, use it for completion instead:

```python
# Direct completion (no chat format)
prompt = "The capital of France is"
output = model.generate(prompt, max_new_tokens=50)
```

### Option 3: Complete Fine-Tuning

The model needs to be fine-tuned on the chat format data:

```bash
# 1. Prepare data (if not done)
python scripts/prepare_finetuning_data.py

# 2. Run fine-tuning
python src/finetuning/gpu_finetune.py \
  --model checkpoints/best_model.pt \
  --train-data data/hf_cot/train.jsonl \
  --val-data data/hf_cot/val.jsonl \
  --epochs 3 \
  --batch-size 8

# 3. Wait for training to complete (loss should drop to ~2-3)

# 4. Then use chat inference
python src/finetuning/chat_inference.py --interactive
```

### Option 4: Use Alternative Format

If you want immediate results without fine-tuning, use a simpler format:

```python
# Simple format the base model might understand
prompt = """Question: What is 2+2?
Answer:"""

# Or
prompt = """User: hi
Assistant:"""
```

## Checking Model State

Run this to check if model is properly fine-tuned:

```bash
# Check adapter files
ls -la finetuned_model_gpu/best_model/

# Should see:
# - adapter_config.json
# - adapter_model.safetensors (or adapter_model.bin)
# - README.md

# Check training state
cat finetuned_model_gpu/best_model/adapter_config.json
```

## Expected Behavior After Fine-Tuning

After successful fine-tuning with loss ~2-3:

```
You: hi

🔍 DEBUG - Raw Generated Text:
<|im_start|>system
You are a helpful assistant...<|im_end|>
<|im_start|>user
hi<|im_end|>
<|im_start|>thought
The user is greeting me. I should respond politely.
<|im_end|>
<|im_start|>assistant
Hello! How can I help you today?<|im_end|>