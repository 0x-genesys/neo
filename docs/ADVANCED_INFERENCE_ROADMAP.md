# Advanced Inference Features Roadmap

This document outlines planned advanced inference features for the Neo transformer system.

## Current Status

### ✅ Implemented
- Basic text generation
- Temperature, top-k, top-p sampling
- Interactive mode
- Remote model loading
- Auto device detection (CUDA/MPS/CPU)

### ❌ Not Implemented
- KV caching
- Model quantization (GGUF/GPTQ)
- Conversation memory
- Context summarization
- Streaming generation

## Planned Features

### 1. KV Caching 🔥 **HIGH PRIORITY**

**Status**: Not implemented
**Impact**: 5-10x faster generation
**Complexity**: Medium

#### Current Behavior
```python
# Each token generation recomputes ALL previous tokens
for _ in range(max_new_tokens):
    logits, _ = self(idx_cond)  # ← Recomputes everything!
    # Sample next token...
```

#### With KV Caching
```python
# Cache key/value tensors, only compute new token
for _ in range(max_new_tokens):
    logits, cache = self(idx_cond, past_key_values=cache)  # ← Reuses cache!
    # Sample next token...
```

#### Implementation Plan

**Step 1**: Modify `CausalSelfAttention` to support caching
```python
class CausalSelfAttention(nn.Module):
    def forward(self, x, past_key_value=None, use_cache=False):
        # Compute Q, K, V
        q, k, v = self.c_attn(x).split(self.d_model, dim=2)
        
        if past_key_value is not None:
            # Concatenate cached K, V with new K, V
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=1)
            v = torch.cat([past_v, v], dim=1)
        
        # Attention computation...
        
        if use_cache:
            return y, (k, v)  # Return cache
        return y, None
```

**Step 2**: Update `TransformerBlock` to pass cache
```python
class TransformerBlock(nn.Module):
    def forward(self, x, past_key_value=None, use_cache=False):
        attn_out, cache = self.attn(self.ln_1(x), past_key_value, use_cache)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, cache
```

**Step 3**: Update `generate()` to use cache
```python
@torch.no_grad()
def generate(self, idx, max_new_tokens, use_cache=True, ...):
    past_key_values = None
    
    for _ in range(max_new_tokens):
        if use_cache and past_key_values is not None:
            # Only pass the last token
            idx_cond = idx[:, -1:]
        else:
            # Pass full context
            idx_cond = idx[:, -self.context_length:]
        
        logits, past_key_values = self(idx_cond, past_key_values, use_cache)
        # Sample and append...
```

**Benefits**:
- 5-10x faster generation
- Lower memory for long sequences
- Standard in production systems

**Effort**: 2-3 days

---

### 2. Conversation Memory 🔥 **HIGH PRIORITY**

**Status**: Not implemented
**Impact**: Enables chatbot functionality
**Complexity**: Low-Medium

#### Implementation Plan

**Step 1**: Create `ConversationManager` class
```python
class ConversationManager:
    def __init__(self, system_prompt, max_context_length, tokenizer):
        self.system_prompt = system_prompt
        self.max_context_length = max_context_length
        self.tokenizer = tokenizer
        self.history = []  # [(role, content), ...]
    
    def add_message(self, role, content):
        """Add a message to history."""
        self.history.append((role, content))
    
    def build_prompt(self):
        """Build prompt from conversation history."""
        prompt = f"System: {self.system_prompt}\n"
        for role, content in self.history:
            prompt += f"{role}: {content}\n"
        return prompt
    
    def truncate_if_needed(self, new_prompt):
        """Truncate history if context length exceeded."""
        tokens = self.tokenizer.encode(new_prompt)
        if len(tokens) > self.max_context_length:
            # Remove oldest messages until it fits
            while len(tokens) > self.max_context_length and len(self.history) > 1:
                self.history.pop(0)
                new_prompt = self.build_prompt()
                tokens = self.tokenizer.encode(new_prompt)
        return new_prompt
```

**Step 2**: Integrate into `TextGenerator`
```python
class TextGenerator:
    def __init__(self, ...):
        # ...
        self.conversation = None
    
    def start_conversation(self, system_prompt="You are a helpful assistant."):
        """Start a new conversation."""
        self.conversation = ConversationManager(
            system_prompt,
            self.config['model']['context_length'],
            self.tokenizer
        )
    
    def chat(self, user_message):
        """Chat with conversation memory."""
        if self.conversation is None:
            self.start_conversation()
        
        # Add user message
        self.conversation.add_message("User", user_message)
        
        # Build prompt
        prompt = self.conversation.build_prompt()
        prompt = self.conversation.truncate_if_needed(prompt)
        
        # Generate response
        response = self.generate(prompt, ...)
        
        # Extract bot response (remove prompt)
        bot_response = response[len(prompt):]
        
        # Add to history
        self.conversation.add_message("Bot", bot_response)
        
        return bot_response
```

**Step 3**: Add chat mode to CLI
```python
def chat_mode(self):
    """Run interactive chat with conversation memory."""
    print("Chat Mode - Type 'quit' to exit, 'reset' to start new conversation")
    
    self.start_conversation()
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() == 'quit':
            break
        elif user_input.lower() == 'reset':
            self.start_conversation()
            print("Conversation reset!")
            continue
        
        response = self.chat(user_input)
        print(f"Bot: {response}")
```

**Benefits**:
- Chatbot functionality
- Maintains conversation context
- Automatic truncation

**Effort**: 1-2 days

---

### 3. Context Summarization 🟡 **MEDIUM PRIORITY**

**Status**: Not implemented
**Impact**: Better context management
**Complexity**: Medium-High

#### Options

**Option A**: Use external summarization model
```python
from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_context(text, max_length=100):
    summary = summarizer(text, max_length=max_length, min_length=30)
    return summary[0]['summary_text']
```

**Option B**: Use the model itself for summarization
```python
def summarize_with_model(self, text):
    prompt = f"Summarize the following conversation:\n{text}\n\nSummary:"
    summary = self.generate(prompt, max_new_tokens=100)
    return summary
```

**Option C**: Simple truncation (current approach)
```python
def truncate_context(self, text):
    tokens = self.tokenizer.encode(text)
    if len(tokens) > self.max_context_length:
        # Keep most recent tokens
        tokens = tokens[-self.max_context_length:]
    return self.tokenizer.decode(tokens)
```

**Recommendation**: Start with Option C (simple truncation), add Option B later

**Effort**: 1 day (Option C), 2-3 days (Option B)

---

### 4. Model Quantization 🟡 **MEDIUM PRIORITY**

**Status**: Not implemented
**Impact**: 2-4x faster, 50-75% less memory
**Complexity**: High

#### GGUF (CPU/Apple Silicon)

**Requirements**:
- `llama-cpp-python` library
- Model conversion to GGUF format
- Different inference path

**Implementation**:
```python
# Install: pip install llama-cpp-python
from llama_cpp import Llama

class GGUFGenerator:
    def __init__(self, model_path):
        self.model = Llama(
            model_path=model_path,
            n_ctx=512,  # Context length
            n_threads=8,  # CPU threads
            n_gpu_layers=0  # CPU only
        )
    
    def generate(self, prompt, max_tokens=100):
        output = self.model(
            prompt,
            max_tokens=max_tokens,
            temperature=0.8,
            top_k=50,
            top_p=0.95
        )
        return output['choices'][0]['text']
```

**Conversion Process**:
```bash
# 1. Export to ONNX
python scripts/export_to_onnx.py --model checkpoint.pt --output model.onnx

# 2. Convert to GGUF
python scripts/convert_to_gguf.py --input model.onnx --output model.gguf --quantize q4_0
```

#### GPTQ (GPU)

**Requirements**:
- `auto-gptq` library
- CUDA support
- Model calibration data

**Implementation**:
```python
# Install: pip install auto-gptq
from auto_gptq import AutoGPTQForCausalLM

class GPTQGenerator:
    def __init__(self, model_path):
        self.model = AutoGPTQForCausalLM.from_quantized(
            model_path,
            device="cuda:0",
            use_triton=True
        )
    
    def generate(self, prompt, max_tokens=100):
        # Similar to regular generation
        pass
```

**Challenges**:
- Requires model conversion pipeline
- Different inference code paths
- Quality validation needed
- Not all operations supported

**Effort**: 1-2 weeks

---

### 5. Streaming Generation 🟢 **LOW PRIORITY**

**Status**: Not implemented
**Impact**: Better UX for long generations
**Complexity**: Low

**Implementation**:
```python
def generate_stream(self, prompt, max_new_tokens=100):
    """Generate tokens one at a time (streaming)."""
    input_ids = self.tokenizer.encode(prompt)
    input_ids = torch.tensor([input_ids]).to(self.device)
    
    for _ in range(max_new_tokens):
        # Generate next token
        logits, _ = self.model(input_ids)
        next_token = sample_token(logits[:, -1, :])
        
        # Yield token immediately
        yield self.tokenizer.decode([next_token])
        
        # Append for next iteration
        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
```

**Usage**:
```python
for token in generator.generate_stream(prompt):
    print(token, end='', flush=True)
```

**Effort**: 1 day

---

## Implementation Priority

### Phase 1: Core Performance (2-3 weeks)
1. ✅ **KV Caching** - 5-10x speedup
2. ✅ **Conversation Memory** - Chatbot functionality
3. ✅ **Simple Context Truncation** - Basic context management

### Phase 2: Advanced Features (2-3 weeks)
4. **Streaming Generation** - Better UX
5. **Context Summarization** - Intelligent truncation
6. **Batch Inference** - Process multiple prompts

### Phase 3: Optimization (3-4 weeks)
7. **GGUF Quantization** - CPU/Apple Silicon optimization
8. **GPTQ Quantization** - GPU optimization
9. **Model Export** - ONNX/TorchScript

## Configuration

### Updated `config/inference.yaml`

```yaml
# Inference Configuration

# Device Configuration
device:
  auto_detect: true
  prefer_quantized: true  # Use quantized model if available

# Generation Parameters
generation:
  max_new_tokens: 100
  temperature: 0.8
  top_k: 50
  top_p: 0.95
  use_cache: true  # Enable KV caching (5-10x faster)
  streaming: false  # Stream tokens as generated

# Conversation Settings
conversation:
  enabled: false  # Enable conversation memory
  system_prompt: "You are a helpful assistant."
  max_history: 10  # Maximum conversation turns to keep
  truncation_strategy: "simple"  # simple, summarize, or sliding_window

# Context Management
context:
  max_length: 512  # From model config
  truncation_strategy: "simple"  # simple, summarize, or sliding_window
  summarization:
    enabled: false
    model: "self"  # self or external
    max_summary_length: 100

# Quantization (if available)
quantization:
  enabled: false
  format: "auto"  # auto, gguf, gptq, or none
  gguf:
    model_path: null  # Path to .gguf file
    n_threads: 8
    n_gpu_layers: 0  # 0 for CPU, >0 for GPU offload
  gptq:
    model_path: null  # Path to GPTQ model
    bits: 4  # 4 or 8
    group_size: 128

# Performance
performance:
  batch_size: 1
  use_torch_compile: false
  torch_compile_mode: "default"  # default, reduce-overhead, max-autotune
```

## Testing Plan

### KV Caching Tests
```python
def test_kv_caching_correctness():
    """Verify KV caching produces same results."""
    # Generate without cache
    output_no_cache = model.generate(prompt, use_cache=False)
    
    # Generate with cache
    output_with_cache = model.generate(prompt, use_cache=True)
    
    assert torch.allclose(output_no_cache, output_with_cache)

def test_kv_caching_speed():
    """Verify KV caching is faster."""
    import time
    
    # Without cache
    start = time.time()
    model.generate(prompt, max_new_tokens=100, use_cache=False)
    time_no_cache = time.time() - start
    
    # With cache
    start = time.time()
    model.generate(prompt, max_new_tokens=100, use_cache=True)
    time_with_cache = time.time() - start
    
    speedup = time_no_cache / time_with_cache
    print(f"Speedup: {speedup:.2f}x")
    assert speedup > 3.0  # Should be at least 3x faster
```

### Conversation Memory Tests
```python
def test_conversation_memory():
    """Test conversation maintains context."""
    generator.start_conversation()
    
    # First exchange
    response1 = generator.chat("My name is Alice")
    assert "Alice" in response1 or "name" in response1
    
    # Second exchange - should remember name
    response2 = generator.chat("What's my name?")
    assert "Alice" in response2
```

## Documentation Updates

### README.md
- Add KV caching section
- Add conversation mode examples
- Add quantization guide

### ARCHITECTURE.md
- Document KV caching implementation
- Explain conversation memory design
- Add quantization architecture

### New Guides
- `docs/KV_CACHING_GUIDE.md` - KV caching details
- `docs/CONVERSATION_MODE_GUIDE.md` - Chat functionality
- `docs/QUANTIZATION_GUIDE.md` - Model quantization

## Estimated Timeline

| Feature | Effort | Priority | Status |
|---------|--------|----------|--------|
| KV Caching | 2-3 days | HIGH | ❌ Not started |
| Conversation Memory | 1-2 days | HIGH | ❌ Not started |
| Simple Truncation | 1 day | HIGH | ❌ Not started |
| Streaming Generation | 1 day | LOW | ❌ Not started |
| Context Summarization | 2-3 days | MEDIUM | ❌ Not started |
| GGUF Quantization | 1 week | MEDIUM | ❌ Not started |
| GPTQ Quantization | 1 week | MEDIUM | ❌ Not started |

**Total Estimated Time**: 3-4 weeks for Phase 1, 6-8 weeks for all phases

## Next Steps

1. **Immediate**: Implement KV caching (biggest impact)
2. **Short-term**: Add conversation memory
3. **Medium-term**: Add quantization support
4. **Long-term**: Advanced features (streaming, summarization)

## References

- [HuggingFace Transformers KV Cache](https://huggingface.co/docs/transformers/main/en/kv_cache)
- [GGUF Format](https://github.com/ggerganov/llama.cpp)
- [GPTQ Quantization](https://github.com/IST-DASLab/gptq)
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
- [auto-gptq](https://github.com/PanQiWei/AutoGPTQ)
