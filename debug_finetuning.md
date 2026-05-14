# Debugging Finetuning Gibberish/Repetitive Output

## Executive Summary

After analyzing `base_trainer.py`, `data_utils.py`, and `gpu_finetune.py`, I've identified **3 primary causes** of gibberish/repetitive output during finetuning:

1. **Label Masking Bug** - System/User tokens are not being properly masked, causing the model to learn to predict system prompts
2. **Missing `thought` Field Handling** - `null`/missing `thought` values in factual data cause malformed training examples
3. **LoRA Target Modules Missing MLP** - The `lm_head` is commented out in config, limiting adaptation capacity

---

## Root Cause Analysis

### 1. Label Masking Bug in [`data_utils.py`](src/finetuning/data_utils.py:227-279)

**Location:** `CoTDataset.__getitem__()` lines 227-279

**Problem:** The role-based masking logic has a critical flaw:

```python
# Mask system, user. ONLY train on Thought + Assistant.
if 'system' in role_text or 'user' in role_text:
    # Mask this entire block
    labels[i:end_idx+1] = -100
```

**Issue:** When the model generates text, it learns to predict the **next token** after the masked region. Since system/user blocks are fully masked, the model sees:
- Masked: `<|im_start|>system\nYou are a helpful...\n`
- Unmasked: `<|im_start|>thought\nLet me think...\n`
- Unmasked: `<|im_start|>assistant\nResponse...\n`

The model learns that after `<|im_start|>thought\n`, it should generate `<|im_start|>assistant\n`, but it also learns patterns from the **unmasked thought block** which contains the training data's thought patterns.

**Why this causes gibberish:**
- The model sees the same thought patterns repeatedly in training data
- It learns to generate "Let me look into:", "Analyzing the query" etc. as fillers
- Without proper masking, it doesn't learn when to stop generating

**Evidence from code:**
```python
# Line 154 in data_utils.py
thought_content = example.get('thought') or ""
```

When `thought` is `null` (as in your factual data), it becomes an empty string, creating malformed examples like:
```
<|im_start|>user\nQuestion...\n
<|im_start|>thought\n
<|im_start|>assistant\nResponse...\n
```

The empty thought block creates a pattern the model tries to fill with repetitive phrases.

---

### 2. Missing `thought` Field Handling in [`data_utils.py`](src/finetuning/data_utils.py:154)

**Location:** `CoTDataset._format_conversation()` line 154

**Problem:** Your factual data has `null` thought values:
```json
{"instruction": "Which has more electrons?", "thought": null, "response": "neon"}
```

**Current handling:**
```python
thought_content = example.get('thought') or ""
```

This creates:
```
<|im_start|>user\nQuestion...\n
<|im_start|>thought\n
<|im_start|>assistant\nResponse...\n
```

**Why this causes problems:**
1. The empty `<|im_start|>thought\n` block is still present in the training data
2. The model learns to generate content after `<|im_start|>thought\n` but there's no content to learn
3. During inference, the model generates gibberish to fill the empty thought block
4. The model also learns to repeat patterns from the training data's thought field

**Evidence from your data:**
```
$ head -n 5 data/factual/factual_train.jsonl
{"instruction": "Which has more electrons, neon or fluorine?", "thought": null, "response": "neon"}
{"instruction": "The measurement of the extent...", "thought": null, "response": "length"}
```

All your factual data has `null` thoughts, but the formatting code still includes the thought block.

---

### 3. LoRA Target Modules Configuration Issue in [`config/finetuning_config_200m.yaml`](config/finetuning_config_200m.yaml:29)

**Location:** Config file line 29

**Problem:** The `lm_head` is commented out:
```yaml
target_modules:
  - c_attn        # KEEP: Attention QKV (Logic & Grammar)
  - c_proj        # KEEP: Attention Output 
  - gate_proj     # ADD: SwiGLU MLP (Facts & Knowledge)
  - up_proj       # ADD: SwiGLU MLP
  - down_proj     # ADD: SwiGLU MLP
  # - lm_head       # KEEP: Vocabulary output  <-- COMMENTED OUT
```

**Why this causes problems:**
1. The `lm_head` is the final layer that maps hidden states to vocabulary tokens
2. Without LoRA adapters on `lm_head`, the model cannot adapt its output vocabulary
3. The model retains its pre-trained output distribution, which favors common patterns
4. This causes repetitive output as the model falls back to pre-trained patterns

**Evidence from base_trainer.py:**
```python
# Line 225 in base_trainer.py
target_modules = ["c_attn", "c_proj", "net.0", "net.2", "lm_head"]
```

The fallback defaults include `lm_head`, but your config explicitly excludes it.

---

### 4. Label Masking Logic Bug in [`data_utils.py`](src/finetuning/data_utils.py:246-271)

**Location:** Lines 246-271 in the role masking section

**Problem:** The masking loop has a subtle bug:

```python
i = 0
while i < len(labels):
    if input_ids[i] == im_start_id:
        # Find end of message
        end_idx = i + 1
        while end_idx < len(labels) and input_ids[end_idx] != im_end_id:
            end_idx += 1
        
        # Extract role
        if i + 1 < len(input_ids):
            role_check_end = min(i + 20, end_idx)
            role_tokens = input_ids[i+1:role_check_end].tolist()
            role_text = self.tokenizer.decode(role_tokens).lower()
            
            if 'system' in role_text or 'user' in role_text:
                labels[i:end_idx+1] = -100
            # else: keep assistant for training
        
        i = end_idx + 1
    else:
        i += 1
```

**Issue:** The logic only masks system/user, but **does not mask the thought block**. This means:
- System: masked ✓
- User: masked ✓
- Thought: **NOT masked** ✗
- Assistant: **NOT masked** ✓

**Why this causes problems:**
1. The model is trained to predict the thought block content
2. In factual data, thoughts are `null` or generic patterns
3. The model learns to generate repetitive thought patterns
4. During inference, it generates the same thought patterns repeatedly

**The fix should be:**
- Only train on `assistant` response
- Mask `system`, `user`, AND `thought` blocks

---

### 5. Training on Empty/Null Thought Blocks

**Location:** [`data_utils.py`](src/finetuning/data_utils.py:154)

**Problem:** When `thought` is `null`, the formatted conversation becomes:
```
<|im_start|>system\nYou are a helpful...\n
<|im_start|>user\nQuestion...\n
<|im_start|>thought\n
<|im_start|>assistant\nResponse...\n
```

The empty `<|im_start|>thought\n` block creates a pattern where:
1. The model sees `<|im_start|>thought\n` followed by `<|im_start|>assistant\n`
2. It learns to generate nothing (or gibberish) between these tokens
3. During inference, it generates repetitive filler phrases

**Evidence from your data:**
All 15,000 pre-trained tokens likely have proper thought blocks, but your factual data has `null` thoughts.

---

## Recommended Fixes

### Fix 1: Update Label Masking to Only Train on Assistant

**File:** [`src/finetuning/data_utils.py`](src/finetuning/data_utils.py:227-279)

**Change:** Modify the role masking to only train on assistant responses:

```python
# Mask system, user, and thought. ONLY train on Assistant response.
if 'system' in role_text or 'user' in role_text or 'thought' in role_text:
    # Mask this entire block
    labels[i:end_idx+1] = -100
```

### Fix 2: Handle Null Thought Values

**File:** [`src/finetuning/data_utils.py`](src/finetuning/data_utils.py:154)

**Change:** Skip thought block entirely if null/empty:

```python
# User instruction
formatted_parts.append(
    self._format_message(SPECIAL_TOKENS['user'], example['instruction'])
)

# Thought process (only if present)
thought_content = example.get('thought')
if thought_content:  # Only add if not null/empty
    formatted_parts.append(
        self._format_message(SPECIAL_TOKENS['thought'], thought_content)
    )

# Assistant response
formatted_parts.append(
    self._format_message(SPECIAL_TOKENS['assistant'], example['response'])
)
```

### Fix 3: Enable lm_head in LoRA Config

**File:** [`config/finetuning_config_200m.yaml`](config/finetuning_config_200m.yaml:29)

**Change:** Uncomment `lm_head`:

```yaml
target_modules:
  - c_attn        # KEEP: Attention QKV (Logic & Grammar)
  - c_proj        # KEEP: Attention Output 
  - gate_proj     # ADD: SwiGLU MLP (Facts & Knowledge)
  - up_proj       # ADD: SwiGLU MLP
  - down_proj     # ADD: SwiGLU MLP
  - lm_head       # KEEP: Vocabulary output
```

### Fix 4: Add Validation for Null Thought Values

**File:** [`src/finetuning/data_utils.py`](src/finetuning/data_utils.py:154)

**Change:** Add warning for null thoughts:

```python
thought_content = example.get('thought')
if thought_content is None:
    print(f"⚠️  WARNING: Null thought in example {idx}")
    thought_content = ""
```

---

## Debugging Steps

### Step 1: Verify Training Data Format

Run this to check your training data:

```python
from src.finetuning.data_utils import CoTDataset
from src.tokenizer_utils import load_tokenizer

tokenizer = load_tokenizer()
dataset = CoTDataset(
    'data/factual/factual_train.jsonl',
    tokenizer,
    max_length=512
)

# Check first 10 examples
for i in range(10):
    example = dataset.examples[i]
    print(f"\nExample {i}:")
    print(f"  Instruction: {example['instruction'][:50]}...")
    print(f"  Thought: {example.get('thought')}")
    print(f"  Response: {example['response'][:50]}...")
```

### Step 2: Check Label Masking

Add this debug code to `base_trainer.py` line 390:

```python
# Show which tokens are being trained on
non_masked = (labels[0] != -100).sum().item()
masked = (labels[0] == -100).sum().item()
print(f"   Training tokens: {non_masked}, Masked tokens: {masked}")
print(f"   Training ratio: {non_masked / (non_masked + masked):.2%}")
```

### Step 3: Verify LoRA Adapters

After training, verify LoRA adapters are active:

```python
from peft import PeftModel
from src.model import create_model
import yaml

config = yaml.safe_load(open('config/finetuning_config_200m.yaml'))
model = create_model(config)

# Load base model
model.load_state_dict(torch.load('checkpoints/best_model.pt'))

# Load LoRA
model = PeftModel.from_pretrained(model, 'finetuned_model/best_model')

# Check trainable params
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable:,} / {total:,} = {100*trainable/total:.2f}%")
```

Expected: ~0.6% trainable (117M * 0.6% ≈ 700k parameters)

---

## Expected Outcomes After Fixes

1. **No more gibberish:** Only assistant responses are trained, model learns proper response patterns
2. **No more repetition:** Model doesn't learn to fill empty thought blocks
3. **Better factual accuracy:** lm_head LoRA allows vocabulary adaptation

---

## Summary of Most Likely Causes

| Cause | Likelihood | Impact | Fix |
|-------|-----------|--------|-----|
| Null thought values creating empty blocks | **HIGH** | **HIGH** | Fix #2 |
| Label masking not masking thought block | **HIGH** | **HIGH** | Fix #1 |
| lm_head not in LoRA target modules | **MEDIUM** | **MEDIUM** | Fix #3 |
| Role masking logic bug | **MEDIUM** | **MEDIUM** | Fix #4 |

**Priority order:** Fix #2 → Fix #1 → Fix #3 → Fix #4
