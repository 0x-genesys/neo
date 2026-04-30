# Inference Tokenizer Fix

## Issue

When running inference with a model trained using tiktoken, you may encounter:

```
OSError: tiktoken is not a local folder and is not a valid model identifier listed on 'https://huggingface.co/models'
```

## Root Cause

The config stored in the checkpoint has `tokenizer: type: "tiktoken"`, but the inference code was trying to use `AutoTokenizer.from_pretrained("tiktoken")` which doesn't work because:

1. **tiktoken is not a HuggingFace model** - It's a separate tokenizer library by OpenAI
2. **Different API** - tiktoken uses a different API than HuggingFace tokenizers
3. **Wrapper needed** - Training code uses a custom `TiktokenWrapper` class

## Solution

The inference code now properly handles tiktoken by:

1. **Detecting tiktoken** - Checks if tokenizer type is "tiktoken" or contains "cl100k"
2. **Loading tiktoken directly** - Uses `tiktoken.get_encoding("cl100k_base")`
3. **Creating wrapper** - Wraps tiktoken with HuggingFace-compatible interface

## Fixed Code

```python
# Load tokenizer
tokenizer_type = self.config['tokenizer']['type']

# Check if using tiktoken
if tokenizer_type == 'tiktoken' or 'cl100k' in tokenizer_type.lower():
    import tiktoken
    encoding = tiktoken.get_encoding("cl100k_base")
    
    # Create wrapper to match HuggingFace interface
    class TiktokenWrapper:
        def __init__(self, encoding):
            self.encoding = encoding
            self.vocab_size = encoding.n_vocab
            self.eos_token = "<|endoftext|>"
            self.eos_token_id = encoding.encode(self.eos_token)[0]
            self.pad_token = self.eos_token
            self.pad_token_id = self.eos_token_id
        
        def encode(self, text):
            return self.encoding.encode(text)
        
        def decode(self, token_ids, skip_special_tokens=True):
            return self.encoding.decode(token_ids)
        
        def __len__(self):
            return self.vocab_size
    
    self.tokenizer = TiktokenWrapper(encoding)
else:
    # Use HuggingFace tokenizer
    self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_type)
```

## Usage

Now inference works correctly:

```bash
# Local model
python src/inference.py --model checkpoints/production/best_model.pt --interactive

# Remote model
python src/inference.py --model-remote best_model.pt --interactive
```

## Verification

Test the fix:

```bash
# Run tokenizer tests
python test/test_inference_tokenizer.py

# Try inference
python src/inference.py --model checkpoints/production/best_model.pt --prompt "Hello, world!"
```

## Requirements

Make sure tiktoken is installed:

```bash
pip install tiktoken>=0.5.0
```

It's already in `requirements.txt`, so if you installed dependencies, you should have it.

## Supported Tokenizers

The inference code now supports:

### 1. Tiktoken (OpenAI)
```yaml
tokenizer:
  type: "tiktoken"  # or "cl100k_base"
```

Used by models trained with GPT-4 tokenizer.

### 2. HuggingFace Tokenizers
```yaml
tokenizer:
  type: "gpt2"  # or any HuggingFace model
```

Used by models trained with standard HuggingFace tokenizers.

## Technical Details

### TiktokenWrapper Interface

The wrapper provides a HuggingFace-compatible interface:

| Method/Attribute | Description |
|------------------|-------------|
| `encode(text)` | Encode text to token IDs |
| `decode(token_ids)` | Decode token IDs to text |
| `vocab_size` | Vocabulary size |
| `__len__()` | Vocabulary size (via len()) |
| `eos_token` | End-of-sequence token |
| `eos_token_id` | EOS token ID |
| `pad_token` | Padding token |
| `pad_token_id` | Padding token ID |

### Why Tiktoken?

Tiktoken is used because:

1. **Efficiency** - Faster than HuggingFace tokenizers
2. **GPT-4 compatibility** - Same tokenizer as GPT-4
3. **Large vocabulary** - 100K+ tokens
4. **Better code handling** - Optimized for code tokenization

## Troubleshooting

### Issue: "tiktoken is not installed"

**Solution**: Install tiktoken
```bash
pip install tiktoken>=0.5.0
```

### Issue: "Config not found in checkpoint"

**Solution**: Provide config file
```bash
python src/inference.py \
    --model checkpoints/old_model.pt \
    --config config/gpu_training_117m_balanced.yaml \
    --interactive
```

### Issue: "Wrong tokenizer loaded"

**Solution**: Check config in checkpoint
```python
import torch
checkpoint = torch.load("checkpoints/production/best_model.pt", map_location='cpu')
print(checkpoint['config']['tokenizer'])
```

## See Also

- [INFERENCE_GUIDE.md](INFERENCE_GUIDE.md) - Complete inference guide
- [INFERENCE_CONFIG_ANSWER.md](INFERENCE_CONFIG_ANSWER.md) - Config requirements
- [README.md](../README.md) - Main documentation
