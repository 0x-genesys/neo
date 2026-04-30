# Curriculum Learning Guide

## Overview

Curriculum learning is a training strategy that progressively changes the dataset distribution across epochs to guide the model's learning. This implementation allows you to strategically shift focus from facts to logic to behavior over the course of training.

## The Strategy

### 8-Epoch Curriculum

| Phase | Epoch | WikiText (Facts) | Stack (Logic) | UltraChat (Behavior) | Objective |
|-------|-------|------------------|---------------|----------------------|-----------|
| **Current** | 1 | 28% | 13% | 59% | Pattern Discovery (detect current leakage) |
| **Foundation** | 2-3 | 70% | 20% | 10% | Knowledge/Logic Hardening |
| **Bridge A** | 4 | 50% | 25% | 25% | Balanced Contextualization |
| **Bridge B** | 5 | 35% | 25% | 40% | Priority Shift |
| **Bridge C** | 6 | 20% | 20% | 60% | Instruction Emergence |
| **Refinement** | 7-8 | 10% | 10% | 80% | Behavior & Formatting (The "Flip") |

### Rationale

1. **Epoch 1 (Current)**: Start with current distribution to establish baseline and detect any data leakage patterns

2. **Epochs 2-3 (Foundation)**: Heavy emphasis on facts (WikiText) and logic (Stack) to build solid knowledge foundation

3. **Epoch 4 (Bridge A)**: Balanced distribution to contextualize learned knowledge

4. **Epoch 5 (Bridge B)**: Shift priority toward conversational behavior while maintaining knowledge

5. **Epoch 6 (Bridge C)**: Further shift toward instruction-following and conversation

6. **Epochs 7-8 (Refinement)**: Heavy emphasis on conversational behavior and output formatting - the "flip" to instruction-tuned model

## Setup

### Quick Start (Recommended)

The curriculum dataset is available on HuggingFace Hub and will be automatically downloaded when you start training:

```bash
python train.py --config config/gpu_training_117m_balanced.yaml
```

The system will:
1. Check if `data/balanced_300m_curriculum/` exists locally
2. If missing, download from HuggingFace Hub: `0x-genesys/mix_wiki_code_chat_data_300M_tokens_curriculum`
3. Load separate source files for dynamic mixing during training

**No manual dataset preparation needed!**

### Manual Setup (Advanced)

If you want to create and upload your own curriculum dataset:

#### Step 1: Prepare Curriculum Dataset

Split the combined dataset into separate source files:

```bash
python scripts/prepare_curriculum_dataset.py \
    --data-dir data/balanced_300m \
    --output-dir data/curriculum_temp
```

This will create:
```
data/curriculum_temp/
├── wikitext_train.bin    # Facts and encyclopedic knowledge
├── stack_train.bin        # Code and logical reasoning
├── ultrachat_train.bin    # Conversational behavior
├── val.bin                # Validation set (unchanged)
├── dataset_stats.json     # Dataset statistics
└── README.md              # Dataset documentation
```

#### Step 2: Upload to HuggingFace Hub

Upload the curriculum dataset to HuggingFace Hub:

```bash
python scripts/upload_curriculum_dataset.py \
    --data-dir data/balanced_300m \
    --repo-id 0x-genesys/mix_wiki_code_chat_data_300M_tokens_curriculum \
    --temp-dir data/curriculum_temp
```

This will:
1. Split the combined dataset into source files
2. Create a comprehensive README
3. Upload all files to HuggingFace Hub
4. Make the dataset available for auto-download

#### Step 3: Configure Training

Update your config to use the uploaded dataset:

```yaml
data:
  train_file: "data/balanced_300m_curriculum/train.bin"
  val_file: "data/balanced_300m_curriculum/val.bin"
  
  huggingface_dataset:
    repo_id: "0x-genesys/mix_wiki_code_chat_data_300M_tokens_curriculum"
    dataset_name: "balanced_300m_curriculum"
    auto_download: true
```

### Configuration

The curriculum is already configured in `config/gpu_training_117m_balanced.yaml`:

```yaml
training:
  max_epochs: 8
  
  curriculum_learning:
    enabled: true
    
    # Distribution per epoch: [wikitext%, stack%, ultrachat%]
    epoch_distributions:
      1:  [28, 13, 59]  # Current: Pattern Discovery
      2:  [70, 20, 10]  # Foundation: Knowledge/Logic Hardening
      3:  [70, 20, 10]  # Foundation: Knowledge/Logic Hardening
      4:  [50, 25, 25]  # Bridge A: Balanced Contextualization
      5:  [35, 25, 40]  # Bridge B: Priority Shift
      6:  [20, 20, 60]  # Bridge C: Instruction Emergence
      7:  [10, 10, 80]  # Refinement: Behavior & Formatting
      8:  [10, 10, 80]  # Refinement: Behavior & Formatting
    
    sources:
      - "wikitext"   # Facts
      - "stack"      # Logic
      - "ultrachat"  # Behavior
```

### Step 3: Update Data Path

The config is already set up for HuggingFace auto-download:

```yaml
data:
  dataset_type: "binary"
  train_file: "data/balanced_300m_curriculum/train.bin"  # Base path
  val_file: "data/balanced_300m_curriculum/val.bin"
  
  huggingface_dataset:
    repo_id: "0x-genesys/mix_wiki_code_chat_data_300M_tokens_curriculum"
    dataset_name: "balanced_300m_curriculum"
    auto_download: true  # Automatically download if missing
```

**Note**: The `train_file` path is used to determine the data directory. The actual source files (`wikitext_train.bin`, `stack_train.bin`, `ultrachat_train.bin`) are loaded based on the `sources` list in the curriculum config.

### Step 4: Start Training

```bash
python train.py --config config/gpu_training_117m_balanced.yaml
```

## What Happens During Training

### Epoch Start

At the beginning of each epoch, you'll see:

```
================================================================================
🎓 CURRICULUM UPDATE - Epoch 2
================================================================================
Phase: Foundation: Knowledge/Logic Hardening

Dataset Distribution:
  wikitext    :  70%
  stack       :  20%
  ultrachat   :  10%
================================================================================

✅ Loaded wikitext: 102,000,024 tokens, 199,219 sequences
✅ Loaded stack: 48,000,366 tokens, 93,751 sequences
✅ Loaded ultrachat: 208,000,134 tokens, 406,251 sequences

📊 Curriculum Distribution:
   wikitext: 70%
   stack: 20%
   ultrachat: 10%
   Total sequences this epoch: 699,221
```

### Training Progress

The progress bar shows normal training:

```
Epoch 2 | Step 5234/36621 | Loss: 3.1234 | LR: 2.00e-04:  45%|████▌     | 15234/33961
```

### Epoch Transition

When moving to the next epoch, the distribution automatically updates:

```
Epoch 2 completed. Average loss: 3.1234

================================================================================
🎓 CURRICULUM UPDATE - Epoch 3
================================================================================
Phase: Foundation: Knowledge/Logic Hardening
...
```

## Customizing the Curriculum

### Modify Distribution

Edit `config/gpu_training_117m_balanced.yaml`:

```yaml
curriculum_learning:
  enabled: true
  epoch_distributions:
    1:  [33, 33, 34]  # Equal distribution
    2:  [80, 10, 10]  # Heavy facts
    3:  [10, 80, 10]  # Heavy logic
    4:  [10, 10, 80]  # Heavy behavior
    # ... add more epochs
```

**Rules**:
- Each distribution must sum to 100
- Percentages are integers
- Epochs are 1-indexed (1, 2, 3, ...)

### Add More Sources

If you have additional data sources:

1. **Prepare the data**:
   ```bash
   # Create source_train.bin file
   ```

2. **Update config**:
   ```yaml
   curriculum_learning:
     sources:
       - "wikitext"
       - "stack"
       - "ultrachat"
       - "newsource"  # Add new source
     
     epoch_distributions:
       1:  [25, 25, 25, 25]  # 4 sources, must sum to 100
   ```

### Disable Curriculum Learning

To train with standard fixed distribution:

```yaml
curriculum_learning:
  enabled: false  # Use standard single-file training
```

## Monitoring

### TensorBoard

Monitor training progress:

```bash
tensorboard --logdir logs/gpu_training_117m_balanced
```

### Loss Curves

You should see:
- **Epoch 1**: Baseline loss
- **Epochs 2-3**: Loss may increase slightly as model adapts to fact-heavy distribution
- **Epochs 4-6**: Loss stabilizes as distribution balances
- **Epochs 7-8**: Loss decreases as model refines behavior

### Checkpoints

Checkpoints are saved with epoch information:
```
checkpoints/gpu_training_117m_balanced/
├── checkpoint_step_500.pt   # Epoch 1
├── checkpoint_step_1000.pt  # Epoch 1
├── checkpoint_step_5000.pt  # Epoch 2
├── best_model.pt            # Best across all epochs
└── final_model.pt           # After epoch 8
```

## Troubleshooting

### Issue: "Source file not found"

**Problem**: Missing `{source}_train.bin` files

**Solution**: Run the preparation script:
```bash
python scripts/prepare_curriculum_dataset.py
```

### Issue: "Distribution must sum to 100"

**Problem**: Percentages don't add up to 100

**Solution**: Check your config:
```yaml
epoch_distributions:
  1:  [28, 13, 59]  # 28 + 13 + 59 = 100 ✅
  2:  [70, 20, 15]  # 70 + 20 + 15 = 105 ❌
```

### Issue: Out of Memory

**Problem**: Curriculum dataset uses more memory

**Solution**: Use low memory config:
```bash
python train.py --config config/gpu_training_117m_balanced_low_memory.yaml
```

Or reduce batch size:
```yaml
training:
  batch_size: 4  # Reduce from 8
  gradient_accumulation_steps: 32  # Increase to maintain effective batch size
```

### Issue: Slow Epoch Start

**Problem**: Loading source files takes time

**Solution**: This is normal. The files are memory-mapped, so loading is a one-time cost per epoch.

## Best Practices

### 1. Start with Baseline

Run epoch 1 with current distribution to establish baseline performance.

### 2. Monitor Validation Loss

Watch validation loss across epochs. If it increases significantly, the curriculum may be too aggressive.

### 3. Save Checkpoints Frequently

```yaml
training:
  save_interval: 500  # Save every 500 steps
```

This allows you to revert if a phase doesn't work well.

### 4. Experiment with Distributions

The provided distribution is a starting point. Experiment to find what works best for your use case.

### 5. Use Validation Set

The validation set uses fixed distribution. This provides consistent evaluation across epochs.

## Advanced: Custom Curriculum Strategies

### Linear Progression

Gradually shift from facts to behavior:

```yaml
epoch_distributions:
  1:  [80, 10, 10]
  2:  [70, 15, 15]
  3:  [60, 20, 20]
  4:  [50, 25, 25]
  5:  [40, 30, 30]
  6:  [30, 30, 40]
  7:  [20, 20, 60]
  8:  [10, 10, 80]
```

### Cyclic Curriculum

Cycle through emphases:

```yaml
epoch_distributions:
  1:  [70, 15, 15]  # Facts
  2:  [15, 70, 15]  # Logic
  3:  [15, 15, 70]  # Behavior
  4:  [70, 15, 15]  # Facts again
  5:  [15, 70, 15]  # Logic again
  6:  [15, 15, 70]  # Behavior again
  7:  [33, 33, 34]  # Balanced
  8:  [33, 33, 34]  # Balanced
```

### Task-Specific

Focus on specific capabilities:

```yaml
# For code-focused model
epoch_distributions:
  1:  [20, 70, 10]  # Heavy code
  2:  [20, 70, 10]
  3:  [30, 50, 20]  # Balance
  4:  [30, 50, 20]

# For chat-focused model
epoch_distributions:
  1:  [10, 10, 80]  # Heavy chat
  2:  [10, 10, 80]
  3:  [20, 20, 60]  # Balance
  4:  [20, 20, 60]
```

## Results and Analysis

### Expected Outcomes

1. **Better Instruction Following**: The progressive shift to behavior data should improve instruction-following capabilities

2. **Stable Knowledge**: Starting with facts ensures the model builds a solid knowledge foundation

3. **Logical Reasoning**: The logic phase helps the model learn structured thinking

4. **Balanced Performance**: The bridge phases prevent catastrophic forgetting

### Evaluation

Compare models trained with and without curriculum learning:

```bash
# Standard training
python train.py --config config/gpu_training_117m_balanced.yaml \
    --curriculum-disabled

# Curriculum training
python train.py --config config/gpu_training_117m_balanced.yaml
```

Evaluate on:
- Factual accuracy (WikiText perplexity)
- Code generation (HumanEval)
- Instruction following (MT-Bench)
- Conversational quality (human evaluation)

## References

- [Curriculum Learning](https://arxiv.org/abs/2101.10382)
- [Data Mixing Strategies](https://arxiv.org/abs/2210.11416)
- [Progressive Training](https://arxiv.org/abs/2203.15556)

## See Also

- [README.md](../README.md) - Main documentation
- [ARCHITECTURE.md](../ARCHITECTURE.md) - System architecture
- [GPU_TRAINING_OPTIMIZATION_GUIDE.md](GPU_TRAINING_OPTIMIZATION_GUIDE.md) - Performance tuning
