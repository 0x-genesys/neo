# Balanced Logic & Chat Dataset Guide

## Overview

This guide explains how to prepare and use the **300M token "Balanced Logic & Chat"** dataset for training the 117M parameter model on GPUs with 1.5GB memory.

## Dataset Composition

The dataset combines four high-quality sources to create a balanced mix of encyclopedic knowledge, conversational AI, code reasoning, and natural dialogue:

| Source | Tokens | Percentage | Purpose |
|--------|--------|------------|---------|
| **WikiText-103** | 102M | 34% | Encyclopedic knowledge, formal writing |
| **UltraChat** | 150M | 50% | Conversational AI, instruction following |
| **The Stack** | 48M | 16% | Code reasoning (Python + Java) |
| **DailyDialog** | ~10M | ~3% | Natural dialogue patterns |
| **TOTAL** | **300M** | **100%** | Balanced logic & chat |

### Validation Split
- **10M tokens** from DailyDialog conversational subset
- Specifically tracks dialogue loss during training
- Ensures model quality on conversational tasks

## Dataset Format

### Tokenization
- **Tokenizer**: `tiktoken` with `cl100k_base` encoding (GPT-4)
- **Vocabulary size**: 100,277 tokens
- **Advantages**: 
  - Efficient encoding (fewer tokens per text)
  - Better multilingual support
  - Consistent with GPT-4 training

### Conversational Format
All conversational data (UltraChat and DailyDialog) is wrapped in a consistent template:

```
User: <user_message>
Assistant: <assistant_message>
```

This format:
- Clearly delineates roles
- Enables instruction following
- Maintains conversational context

### Binary Format
- **File type**: `.bin` (binary)
- **Data type**: `np.uint32` (4 bytes per token)
- **Storage**: Memory-mapped with `np.memmap`
- **Benefits**:
  - Fast loading (no tokenization overhead)
  - Memory efficient (streaming from disk)
  - Deterministic training (pre-shuffled)

## Training Configuration

### Optimized Parameters

Based on 300M unique tokens and 1.5GB GPU constraints:

```yaml
training:
  # Batch Configuration
  batch_size: 16
  gradient_accumulation_steps: 8
  # Effective batch = 16 × 8 = 128
  
  # Training Duration
  max_epochs: 8
  # Total tokens = 300M × 8 = 2.4B
  
  max_steps: 36621
  # Calculation: 2.4B / (16 × 256 × 8) = 36,621
  # Where: batch_size × context_length × grad_accum = 65,536 tokens/step
  
  # Learning Rate
  learning_rate: 2.0e-4
  # Sweet spot for 124M model with diverse data
  
  weight_decay: 0.15
  # Firm regularization to prevent memorization
  
  warmup_steps: 1500
  # ~4% of total steps for vocab stabilization
  
  dropout: 0.12
  # Lowered from 0.15 due to more unique data
```

### Rationale

#### Why 8 Epochs?
- **Total exposure**: 2.4B tokens (300M × 8)
- **Balance**: Enough for convergence without over-memorization
- **Diversity**: 300M unique tokens prevents overfitting

#### Why 36,621 Steps?
```
Total tokens: 2.4B
Tokens per step: batch_size × context_length × grad_accum
                = 16 × 256 × 8
                = 65,536

Steps = 2,400,000,000 / 65,536 = 36,621
```

#### Why Learning Rate 2.0e-4?
- Optimal for 124M parameter models
- Tested on diverse datasets
- Balances convergence speed and stability

#### Why Weight Decay 0.15?
- **Higher than typical** (usually 0.1)
- Prevents model from "reciting" training data
- Encourages generalization and creativity

#### Why Warmup 1,500 Steps?
- **~4% of total steps** (1,500 / 36,621)
- Critical for stabilizing 100k vocabulary embeddings
- Prevents early training instability

## Preparation Steps

### 1. Install Dependencies

```bash
pip install tiktoken datasets numpy tqdm
```

### 2. Run Dataset Builder

```bash
python scripts/prepare_balanced_dataset.py \
  --output-dir data/balanced_300m \
  --seed 42
```

**Expected runtime**: 2-4 hours (depending on internet speed and CPU)

### 3. Verify Output

The script creates:

```
data/balanced_300m/
├── train.bin           # 300M tokens (~1.2GB)
├── val.bin            # 10M tokens (~40MB)
└── dataset_stats.json # Statistics and metadata
```

### 4. Check Statistics

```bash
cat data/balanced_300m/dataset_stats.json
```

Expected output:
```json
{
  "sources": {
    "wikitext": {"docs": 28475, "tokens": 102000000},
    "ultrachat": {"docs": 75000, "tokens": 150000000},
    "stack": {"docs": 12000, "tokens": 48000000},
    "dailydialog": {"docs": 5000, "tokens": 10000000}
  },
  "totals": {
    "documents": 120475,
    "tokens": 310000000
  },
  "tokenizer": "tiktoken cl100k_base (GPT-4)",
  "vocab_size": 100277
}
```

## Training

### Start Training

```bash
python train.py --config config/gpu_training_117m_1.5gb.yaml
```

### Monitor Progress

The training will:
1. Load binary datasets (fast, no tokenization)
2. Train for 36,621 steps
3. Evaluate every 1,000 steps on conversational validation set
4. Save checkpoints every 1,000 steps

### Expected Timeline

**On 1.5GB GPU (e.g., GTX 1050)**:
- **Steps per second**: ~0.5-1.0
- **Total time**: 10-20 hours
- **Tokens per second**: ~32,768-65,536

**On larger GPU (e.g., RTX 3090)**:
- **Steps per second**: ~5-10
- **Total time**: 1-2 hours
- **Tokens per second**: ~327,680-655,360

## Dataset Quality

### Source Quality

1. **WikiText-103**
   - Curated Wikipedia articles
   - High-quality, factual content
   - Formal writing style

2. **UltraChat**
   - Large-scale conversational dataset
   - Diverse instruction-following examples
   - Multi-turn dialogues

3. **The Stack**
   - Deduplicated code corpus
   - Python and Java (most popular languages)
   - Real-world code patterns

4. **DailyDialog**
   - Natural human conversations
   - Diverse topics and emotions
   - High-quality annotations

### Data Processing

1. **Filtering**
   - Remove empty or very short texts
   - Skip malformed examples
   - Validate token ranges

2. **Formatting**
   - Consistent User/Assistant template
   - Proper conversation structure
   - Clear role delineation

3. **Shuffling**
   - Document-level randomization
   - Prevents source clustering
   - Improves training dynamics

4. **Packing**
   - Efficient binary format
   - Memory-mapped loading
   - No tokenization overhead

## Troubleshooting

### Issue: Dataset download fails

**Solution**: Check internet connection and HuggingFace access
```bash
# Login to HuggingFace (if needed)
huggingface-cli login

# Test connection
python -c "from datasets import load_dataset; load_dataset('wikitext', 'wikitext-2-raw-v1')"
```

### Issue: Out of memory during preparation

**Solution**: The script uses streaming and memory-mapping to minimize memory usage. If still failing:
```bash
# Reduce batch processing (edit script)
# Or use a machine with more RAM (16GB+ recommended)
```

### Issue: Binary files not found during training

**Solution**: Verify paths in config
```yaml
data:
  train_file: "data/balanced_300m/train.bin"  # Check this path
  val_file: "data/balanced_300m/val.bin"      # Check this path
```

### Issue: Vocab size mismatch

**Solution**: Ensure tiktoken is installed and config matches
```bash
pip install tiktoken

# Verify in config
tokenizer:
  type: "tiktoken"
  vocab_size: 100277
```

## Advanced Options

### Custom Token Targets

Edit `scripts/prepare_balanced_dataset.py`:

```python
self.target_tokens = {
    'wikitext': 150_000_000,    # Increase WikiText
    'ultrachat': 100_000_000,   # Decrease UltraChat
    'stack': 50_000_000,        # Increase code
    'dailydialog': 10_000_000,  # Keep same
}
```

### Different Validation Split

To use a different validation source:

```python
# In load_dailydialog() method
if idx % 5 == 0:  # 20% for validation (instead of 10%)
    val_documents.append(tokens)
```

### Add More Sources

To add additional datasets:

```python
def load_custom_dataset(self) -> List[List[int]]:
    """Load custom dataset."""
    dataset = load_dataset('your/dataset')
    documents = []
    
    for example in dataset:
        text = example['text']
        tokens = self.tokenize_text(text)
        documents.append(tokens)
    
    return documents
```

## Performance Tips

### Faster Preparation
1. Use SSD for dataset cache
2. Increase CPU cores (for parallel processing)
3. Use faster internet connection

### Faster Training
1. Increase batch size (if GPU memory allows)
2. Use mixed precision (FP16)
3. Enable gradient checkpointing
4. Use multiple GPUs with DDP

### Better Quality
1. Increase training epochs (8 → 12)
2. Tune learning rate (try 1.5e-4 or 2.5e-4)
3. Adjust weight decay (try 0.1 or 0.2)
4. Add more diverse data sources

## References

- **WikiText**: [Merity et al., 2016](https://arxiv.org/abs/1609.07843)
- **UltraChat**: [Ding et al., 2023](https://github.com/thunlp/UltraChat)
- **The Stack**: [Kocetkov et al., 2022](https://arxiv.org/abs/2211.15533)
- **DailyDialog**: [Li et al., 2017](https://arxiv.org/abs/1710.03957)
- **tiktoken**: [OpenAI](https://github.com/openai/tiktoken)

## Next Steps

After preparing the dataset:

1. ✅ **Verify dataset**: Check `dataset_stats.json`
2. ✅ **Update config**: Ensure paths are correct
3. ✅ **Start training**: `python train.py --config config/gpu_training_117m_1.5gb.yaml`
4. ✅ **Monitor progress**: Watch loss curves and validation metrics
5. ✅ **Evaluate model**: Test on held-out examples
6. ✅ **Fine-tune**: Adjust hyperparameters if needed

Good luck with your training! 🚀
