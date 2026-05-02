# Chat Inference Quick Reference

**Status**: ✅ Ready to Use  
**Last Updated**: May 2, 2026

---

## Quick Start (Zero Configuration)

The simplest way to use chat inference - no arguments needed!

```bash
python src/finetuning/chat_inference.py --interactive
```

This automatically:
- Downloads base model: `best_model.pt`
- Downloads adapter: `finetune/chat_adapter`
- From repository: `0x-genesys/neo_weights_checkpoints`
- Starts interactive chat mode

---

## Common Usage Patterns

### 1. Interactive Chat with Debug

```bash
python src/finetuning/chat_inference.py --interactive --debug
```

Shows detailed information:
- Formatted prompt
- Tokenization details
- Model structure
- Generation process
- Raw output

### 2. Single Prompt

```bash
python src/finetuning/chat_inference.py \
    --prompt "What is 2+2?" \
    --show-thought
```

### 3. Local Models

```bash
python src/finetuning/chat_inference.py \
    --base-model checkpoints/best_model.pt \
    --adapter finetuned_model_gpu/best_model \
    --interactive
```

### 4. Custom Remote Models

```bash
python src/finetuning/chat_inference.py \
    --model-repo username/my-repo \
    --base-model-remote my_model.pt \
    --adapter-remote finetune/my_adapter \
    --interactive
```

### 5. Custom Generation Parameters

```bash
python src/finetuning/chat_inference.py \
    --temperature 0.9 \
    --top-k 100 \
    --top-p 0.95 \
    --max-tokens 512 \
    --interactive
```

---

## Interactive Mode Commands

Once in interactive mode, you can use these commands:

| Command | Description |
|---------|-------------|
| `Type message` | Send message to the model |
| `quit` or `exit` | Exit the program |
| `config` | Show current settings |
| `thought on/off` | Toggle thought display |
| `debug on/off` | Toggle debug mode |
| `set <param> <value>` | Change a setting |

### Example Session

```
🧑 You: hi there

🤖 Assistant: 
💭 [Thought: User is greeting me, I should respond warmly]

Hello! How can I help you today?

🧑 You: thought off
✅ Thought display disabled

🧑 You: what is 5+3?

🤖 Assistant: 
5 + 3 = 8

🧑 You: set temperature 0.9
✅ Updated temperature to 0.9

🧑 You: quit
👋 Goodbye!
```

---

## Command-Line Arguments

### Model Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--base-model` | None | Path to local base model (.pt) |
| `--base-model-remote` | `best_model.pt` | Remote base model filename |
| `--adapter` | None | Path to local adapter directory |
| `--adapter-remote` | `finetune/chat_adapter` | Remote adapter path |
| `--model-repo` | `0x-genesys/neo_weights_checkpoints` | HuggingFace repo ID |

### Generation Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--max-tokens` | 200 | Maximum tokens to generate |
| `--temperature` | 0.7 | Sampling temperature (0.1-2.0) |
| `--top-k` | 50 | Top-k sampling (1-100) |
| `--top-p` | 0.9 | Nucleus sampling (0.0-1.0) |
| `--show-thought` | False | Show thought process |
| `--debug` | False | Enable debug mode |

### Other Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | None | Path to config file |
| `--device` | `auto` | Device (auto/cuda/mps/cpu) |
| `--prompt` | None | Single prompt (non-interactive) |
| `--interactive` | False | Run in interactive mode |

---

## Generation Parameters Explained

### Temperature

Controls randomness in generation:
- **0.1-0.3**: Very focused, deterministic (good for factual answers)
- **0.7**: Balanced (default)
- **1.0-2.0**: More creative, diverse (good for creative writing)

### Top-K

Keeps only the K most likely tokens:
- **10-20**: Very focused
- **50**: Balanced (default)
- **100+**: More diverse

### Top-P (Nucleus Sampling)

Keeps smallest set of tokens with cumulative probability > P:
- **0.8**: Focused
- **0.9**: Balanced (default)
- **0.95**: More diverse

---

## Debug Output Explained

When `--debug` is enabled, you'll see:

### 1. Formatted Prompt
```
🔍 DEBUG - Formatted Prompt:
<|im_start|>system
You are a helpful AI assistant...
<|im_end|>
<|im_start|>user
What is 2+2?
<|im_end|>
<|im_start|>thought
```

### 2. Tokenization
```
🔍 DEBUG - Tokenization:
   Input length: 104 tokens
   First 10 tokens: [27, 91, 318, 5011, 91, 29, 9125, 198, 2675, 527]
   Last 10 tokens: [91, 397, 27, 91, 318, 5011, 91, 29, 61665, 198]
```

### 3. Model Structure
```
🔍 DEBUG - Model Structure:
   Model type: PeftModelForCausalLM
   Has base_model: True
   Base model type: LoraModel
   Inner model type: DecoderOnlyTransformer
```

### 4. Generation Method
```
🔍 DEBUG - Using custom generation loop with LoRA weights applied
```

### 5. Generation Stats
```
🔍 DEBUG - Generation:
   Input length: 104 tokens
   Output length: 156 tokens
   Generated 52 new tokens
```

### 6. Raw Output
```
🔍 DEBUG - Raw Generated Text:
<|im_start|>system...
[full generated text]
```

---

## Troubleshooting

### Issue: "Model not found"

**Solution**: Check model paths or use remote loading:
```bash
python src/finetuning/chat_inference.py \
    --base-model-remote final_mode_1_may.pt \
    --adapter-remote chat_adapter \
    --interactive
```

### Issue: "No response generated"

**Solution**: Enable debug mode to see what's happening:
```bash
python src/finetuning/chat_inference.py --interactive --debug
```

Check:
1. Is the model generating text? (check raw output)
2. Is the parsing failing? (check response parsing section)
3. Try adjusting temperature/top-k/top-p

### Issue: "Checkpoint incompatibility"

**Solution**: PyTorch version mismatch. Either:
1. Upgrade PyTorch: `pip install --upgrade torch`
2. Use checkpoint from same PyTorch version
3. Convert checkpoint: `python scripts/convert_checkpoint_pytorch22.py`

### Issue: "PEFT library not installed"

**Solution**: Install PEFT:
```bash
pip install peft
```

### Issue: "Out of memory"

**Solution**: Reduce batch size or use smaller model:
```bash
python src/finetuning/chat_inference.py \
    --max-tokens 100 \
    --interactive
```

---

## Performance Tips

### 1. Use GPU if Available

The script automatically detects and uses GPU (CUDA/MPS):
```bash
# Force specific device
python src/finetuning/chat_inference.py --device cuda --interactive
```

### 2. Adjust Max Tokens

Shorter responses are faster:
```bash
python src/finetuning/chat_inference.py --max-tokens 100 --interactive
```

### 3. Use Lower Temperature

More deterministic = faster:
```bash
python src/finetuning/chat_inference.py --temperature 0.3 --interactive
```

---

## Example Prompts

### Factual Questions
```
What is the capital of France?
Explain how photosynthesis works.
What are the main causes of climate change?
```

### Math Problems
```
Solve: 5x + 3 = 18
What is 15% of 240?
Calculate the area of a circle with radius 5.
```

### Creative Tasks
```
Write a short poem about the ocean.
Create a story about a robot learning to paint.
Describe a futuristic city in 2100.
```

### Code Questions
```
How do I sort a list in Python?
Explain the difference between let and const in JavaScript.
Write a function to reverse a string.
```

---

## Advanced Usage

### Custom System Prompt

Modify `SYSTEM_PROMPT` in `src/finetuning/base_trainer.py` or pass custom prompt in code.

### Batch Processing

For multiple prompts, use a script:
```python
from src.finetuning.chat_inference import ChatGenerator

generator = ChatGenerator(
    base_model_remote="final_mode_1_may.pt",
    adapter_remote="chat_adapter"
)

prompts = ["What is 2+2?", "Explain AI", "Write a poem"]
for prompt in prompts:
    result = generator.generate(prompt)
    print(f"Q: {prompt}")
    print(f"A: {result['response']}\n")
```

### API Wrapper

Create a simple API:
```python
from flask import Flask, request, jsonify
from src.finetuning.chat_inference import ChatGenerator

app = Flask(__name__)
generator = ChatGenerator(base_model_remote="final_mode_1_may.pt")

@app.route('/chat', methods=['POST'])
def chat():
    prompt = request.json['prompt']
    result = generator.generate(prompt)
    return jsonify(result)

app.run(port=5000)
```

---

## Files Reference

| File | Purpose |
|------|---------|
| `src/finetuning/chat_inference.py` | Main inference script |
| `src/finetuning/base_trainer.py` | Training logic and system prompt |
| `src/finetuning/data_utils.py` | Data formatting and special tokens |
| `src/model.py` | Base transformer model |
| `src/tokenizer_utils.py` | Tokenizer utilities |
| `src/remote_model_loader.py` | HuggingFace Hub download |

---

## Related Documentation

- **CHAT_INFERENCE_LORA_FIX.md**: Technical implementation details
- **CONTEXT_TRANSFER_COMPLETE.md**: Summary of all fixes
- **FINETUNING_QUICKSTART.md**: How to fine-tune your own model
- **CHECKPOINT_COMPATIBILITY_GUIDE.md**: PyTorch version issues

---

**Status**: ✅ Ready for Production Use

For issues or questions, check debug output first, then review the troubleshooting section.
