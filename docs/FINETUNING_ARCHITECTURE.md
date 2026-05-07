# Fine-Tuning Architecture

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Fine-Tuning Pipeline                          │
└─────────────────────────────────────────────────────────────────┘

┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│   Raw Data   │─────▶│ Data Utils   │─────▶│ CoT Dataset  │
│  (JSONL)     │      │ (Formatting) │      │ (Tokenized)  │
└──────────────┘      └──────────────┘      └──────────────┘
                                                     │
                                                     ▼
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│ Pre-trained  │─────▶│ LoRA Adapter │─────▶│ Fine-tuned   │
│ 117M Model   │      │ (1.5M params)│      │    Model     │
└──────────────┘      └──────────────┘      └──────────────┘
                             │
                             ▼
                      ┌──────────────┐
                      │   Training   │
                      │   (AdamW +   │
                      │   Cosine)    │
                      └──────────────┘
```

## 🔧 LoRA Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Transformer Layer                             │
└─────────────────────────────────────────────────────────────────┘

Input (B, T, 768)
      │
      ▼
┌─────────────────────────────────────────────────────────────────┐
│  Layer Norm                                                      │
└─────────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────────┐
│  Attention (with LoRA)                                           │
│                                                                  │
│  ┌──────────────┐     ┌──────────────┐                         │
│  │   c_attn     │────▶│  LoRA (r=16) │                         │
│  │  (frozen)    │     │  (trainable) │                         │
│  │  768 → 2304  │     │  768 → 16    │                         │
│  └──────────────┘     │  16 → 2304   │                         │
│         │             └──────────────┘                         │
│         │                     │                                 │
│         └─────────(+)─────────┘                                 │
│                       │                                         │
│                       ▼                                         │
│              Multi-Head Attention                               │
│                       │                                         │
│                       ▼                                         │
│  ┌──────────────┐     ┌──────────────┐                         │
│  │   c_proj     │────▶│  LoRA (r=16) │                         │
│  │  (frozen)    │     │  (trainable) │                         │
│  │  768 → 768   │     │  768 → 16    │                         │
│  └──────────────┘     │  16 → 768    │                         │
│         │             └──────────────┘                         │
│         │                     │                                 │
│         └─────────(+)─────────┘                                 │
└─────────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────────┐
│  Residual Connection                                             │
└─────────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────────┐
│  Layer Norm                                                      │
└─────────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────────┐
│  MLP (with LoRA)                                                 │
│                                                                  │
│  ┌──────────────┐     ┌──────────────┐                         │
│  │   net.0      │────▶│  LoRA (r=16) │                         │
│  │  (frozen)    │     │  (trainable) │                         │
│  │  768 → 3072  │     │  768 → 16    │                         │
│  └──────────────┘     │  16 → 3072   │                         │
│         │             └──────────────┘                         │
│         │                     │                                 │
│         └─────────(+)─────────┘                                 │
│                       │                                         │
│                       ▼                                         │
│                     GELU                                        │
│                       │                                         │
│                       ▼                                         │
│  ┌──────────────┐     ┌──────────────┐                         │
│  │   net.2      │────▶│  LoRA (r=16) │                         │
│  │  (frozen)    │     │  (trainable) │                         │
│  │  3072 → 768  │     │  3072 → 16   │                         │
│  └──────────────┘     │  16 → 768    │                         │
│         │             └──────────────┘                         │
│         │                     │                                 │
│         └─────────(+)─────────┘                                 │
└─────────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────────┐
│  Residual Connection                                             │
└─────────────────────────────────────────────────────────────────┘
      │
      ▼
Output (B, T, 768)
```

## 📊 Parameter Breakdown

```
┌─────────────────────────────────────────────────────────────────┐
│                    117M Parameter Model                          │
└─────────────────────────────────────────────────────────────────┘

Total Parameters: 117,000,000

┌──────────────────────────┬──────────────┬──────────────┐
│ Component                │ Parameters   │ Status       │
├──────────────────────────┼──────────────┼──────────────┤
│ Token Embedding          │ 77,012,736   │ ❄️  Frozen   │
│ Position Embedding       │    393,216   │ ❄️  Frozen   │
│ Transformer Blocks (12x) │ 38,597,376   │ ❄️  Frozen   │
│ Layer Norm               │      1,536   │ ❄️  Frozen   │
│ LM Head (tied)           │          0   │ ❄️  Frozen   │
├──────────────────────────┼──────────────┼──────────────┤
│ LoRA Adapters            │  1,548,288   │ 🔥 Trainable │
└──────────────────────────┴──────────────┴──────────────┘

Trainable: 1.3% of total parameters
```

## 🔄 Training Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    Training Loop                                 │
└─────────────────────────────────────────────────────────────────┘

For each epoch:
  For each batch:
    
    1. Load Data
       ┌──────────────┐
       │ CoT Dataset  │
       │ (tokenized)  │
       └──────────────┘
              │
              ▼
    2. Forward Pass (with Mixed Precision)
       ┌──────────────┐
       │ Base Model   │──▶ Frozen weights
       │   (frozen)   │
       └──────────────┘
              │
              ▼
       ┌──────────────┐
       │ LoRA Adapter │──▶ Trainable weights
       │ (trainable)  │
       └──────────────┘
              │
              ▼
       ┌──────────────┐
       │   Logits     │
       └──────────────┘
              │
              ▼
       ┌──────────────┐
       │ Cross-Entropy│
       │     Loss     │
       └──────────────┘
              │
              ▼
    3. Backward Pass
       ┌──────────────┐
       │  Gradients   │──▶ Only for LoRA
       └──────────────┘
              │
              ▼
    4. Gradient Accumulation (4 steps)
       ┌──────────────┐
       │ Accumulate   │
       │  Gradients   │
       └──────────────┘
              │
              ▼
    5. Optimizer Step (every 4 batches)
       ┌──────────────┐
       │ Clip Grads   │──▶ max_norm=1.0
       └──────────────┘
              │
              ▼
       ┌──────────────┐
       │ AdamW Update │──▶ LR=2.0e-5
       └──────────────┘
              │
              ▼
       ┌──────────────┐
       │ LR Schedule  │──▶ Cosine decay
       └──────────────┘
              │
              ▼
    6. Validation (every 100 steps)
       ┌──────────────┐
       │ Val Dataset  │
       └──────────────┘
              │
              ▼
       ┌──────────────┐
       │  Val Loss    │
       └──────────────┘
              │
              ▼
       ┌──────────────┐
       │ Save Best    │──▶ If improved
       └──────────────┘
```

## 🎯 Chain-of-Thought Format

```
┌─────────────────────────────────────────────────────────────────┐
│                    Message Format                                │
└─────────────────────────────────────────────────────────────────┘

<|im_start|>system
You are a helpful assistant...<|im_end|>
         │
         ▼
<|im_start|>user
What is 15 * 23?<|im_end|>
         │
         ▼
<|im_start|>thought
Let me multiply step by step...
15 * 23 = 15 * 20 + 15 * 3 = 300 + 45 = 345<|im_end|>
         │
         ▼
<|im_start|>assistant
The answer is 345.<|im_end|>


┌─────────────────────────────────────────────────────────────────┐
│                    Tokenization                                  │
└─────────────────────────────────────────────────────────────────┘

Text ──▶ Tokenizer ──▶ Token IDs ──▶ Model

Special Tokens:
  <|im_start|> ──▶ Token ID: 100257
  <|im_end|>   ──▶ Token ID: 100258
  <|endoftext|>──▶ Token ID: 50256 (EOS)

Padding:
  <|endoftext|>──▶ Used for padding
  Label: -100   ──▶ Ignored in loss
```

## 🔀 Hardware Adaptation

```
┌─────────────────────────────────────────────────────────────────┐
│                    Device Detection                              │
└─────────────────────────────────────────────────────────────────┘

                    ┌──────────────┐
                    │ Auto-Detect  │
                    └──────────────┘
                           │
           ┌───────────────┼───────────────┐
           │               │               │
           ▼               ▼               ▼
    ┌──────────┐    ┌──────────┐    ┌──────────┐
    │   CUDA   │    │   MPS    │    │   CPU    │
    └──────────┘    └──────────┘    └──────────┘
           │               │               │
           ▼               ▼               ▼
    ┌──────────┐    ┌──────────┐    ┌──────────┐
    │ Batch: 8 │    │ Batch: 4 │    │ Batch: 2 │
    │ Accum: 4 │    │ Accum: 8 │    │ Accum:16 │
    │ AMP: Yes │    │ AMP: No  │    │ AMP: No  │
    └──────────┘    └──────────┘    └──────────┘
           │               │               │
           └───────────────┼───────────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │   Training   │
                    └──────────────┘
```

## 📈 Learning Rate Schedule

```
┌─────────────────────────────────────────────────────────────────┐
│                    Cosine Schedule with Warmup                   │
└─────────────────────────────────────────────────────────────────┘

Learning Rate
    │
2.0e-5│                    ╭─────────╮
    │                   ╱             ╲
    │                 ╱                 ╲
    │               ╱                     ╲
    │             ╱                         ╲
    │           ╱                             ╲
    │         ╱                                 ╲
    │       ╱                                     ╲
    │     ╱                                         ╲
    │   ╱                                             ╲
    │ ╱                                                 ╲
2.0e-6│                                                   ╲____
    │
    └────────────────────────────────────────────────────────▶ Steps
    0        10%                                         100%
         (Warmup)              (Cosine Decay)

Phases:
  1. Warmup (0-10%): Linear increase from 0 to 2.0e-5
  2. Decay (10-100%): Cosine decrease from 2.0e-5 to 2.0e-6
```

## 💾 Checkpoint Structure

```
finetuned_model_gpu/
│
├── best_model/
│   ├── adapter_config.json      # LoRA configuration
│   ├── adapter_model.bin        # LoRA weights (~6MB)
│   ├── tokenizer_config.json    # Tokenizer config
│   ├── special_tokens_map.json  # Special tokens
│   └── training_state.pt        # Training state
│
├── checkpoint_step_500/
│   └── (same structure)
│
└── checkpoint_epoch_1/
    └── (same structure)


File Sizes:
  adapter_model.bin:    ~6 MB   (LoRA weights)
  training_state.pt:    ~50 MB  (optimizer state)
  tokenizer files:      ~1 MB   (tokenizer)
  
Total per checkpoint: ~60 MB
```

## 🔄 Inference Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    Inference Flow                                │
└─────────────────────────────────────────────────────────────────┘

User Prompt
    │
    ▼
┌──────────────┐
│  Format CoT  │──▶ Add <|im_start|> tokens
└──────────────┘
    │
    ▼
┌──────────────┐
│  Tokenize    │──▶ Convert to token IDs
└──────────────┘
    │
    ▼
┌──────────────┐
│ Base Model   │──▶ Frozen weights
│  (frozen)    │
└──────────────┘
    │
    ▼
┌──────────────┐
│ LoRA Adapter │──▶ Fine-tuned weights
│ (fine-tuned) │
└──────────────┘
    │
    ▼
┌──────────────┐
│  Generate    │──▶ Autoregressive sampling
└──────────────┘
    │
    ▼
┌──────────────┐
│  Detokenize  │──▶ Convert to text
└──────────────┘
    │
    ▼
Generated Response
```

## 📊 Memory Usage

```
┌─────────────────────────────────────────────────────────────────┐
│                    GPU Memory Breakdown                          │
└─────────────────────────────────────────────────────────────────┘

Component                    Memory (FP16)
─────────────────────────────────────────
Model Parameters (frozen)    ~234 MB
LoRA Parameters              ~3 MB
Optimizer State (AdamW)      ~12 MB
Gradients                    ~3 MB
Activations (batch=8)        ~500 MB
Buffer                       ~250 MB
─────────────────────────────────────────
Total                        ~1 GB

With gradient checkpointing: ~700 MB
```

---

This architecture enables efficient fine-tuning with minimal memory overhead while maintaining model quality through LoRA's low-rank adaptation approach.
