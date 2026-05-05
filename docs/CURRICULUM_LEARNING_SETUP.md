# Curriculum Learning Setup Guide

## Quick Start

### Option 1: Use Pre-Built Dataset (Recommended)

The curriculum dataset is already available on HuggingFace Hub and will be automatically downloaded:

```bash
# Just start training - dataset downloads automatically!
python train.py --config config/gpu_training_117m_balanced.yaml
```

**That's it!** The system will:
1. Check if `data/balanced_300m_curriculum/` exists
2. Download from HuggingFace Hub if missing
3. Load separate source files for dynamic mixing

### Option 2: Build and Upload Your Own

If you want to create your own curriculum dataset:

```bash
# Step 1: Prepare curriculum dataset (split into source files)
python scripts/prepare_curriculum_dataset.py \
    --data-dir data/balanced_300m \
    --output-dir data/curriculum_temp

# Step 2: Upload to HuggingFace Hub
python scripts/upload_curriculum_dataset.py \
    --data-dir data/balanced_300m \
    --repo-id 0x-genesys/mix_wiki_code_chat_data_300M_tokens_curriculum \
    --temp-dir data/curriculum_temp

# Step 3: Start training (auto-downloads from your repo)
python train.py --config config/gpu_training_117m_balanced.yaml
```

## What is Curriculum Learning?

Curriculum learning progressively changes the dataset distribution across epochs to guide the model's learning. Instead of training on a fixed mix of data, the model learns in phases:

1. **Pattern Discovery** (Epoch 1): Detect current patterns and establish baseline
2. **Foundation** (Epochs 2-3): Build knowledge and logic foundation
3. **Bridge** (Epochs 4-6): Gradually shift toward conversational behavior
4. **Refinement** (Epochs 7-8): Focus on instruction-following and formatting

## The 8-Epoch Strategy

| Phase | Epoch | WikiText | Stack | UltraChat | Focus |
|-------|-------|----------|-------|-----------|-------|
| Current | 1 | 28% | 13% | 59% | Pattern Discovery |
| Foundation | 2-3 | 70% | 20% | 10% | Knowledge/Logic |
| Bridge A | 4 | 50% | 25% | 25% | Balance |
| Bridge B | 5 | 35% | 25% | 40% | Priority Shift |
| Bridge C | 6 | 20% | 20% | 60% | Instruction Emergence |
| Refinement | 7-8 | 10% | 10% | 80% | Behavior & Formatting |

### Why This Strategy?

- **Start with Current**: Establish baseline and detect any data issues
- **Heavy Facts First**: Build solid knowledge foundation (70% WikiText)
- **Gradual Shift**: Smooth transition prevents catastrophic forgetting
- **The Flip**: Final epochs focus heavily on conversational behavior (80% UltraChat)

## Dataset Structure

### Standard Dataset (Combined)
```
data/balanced_300m/
├── train.bin              # All sources combined
├── val.bin                # Validation set
└── dataset_stats.json     # Statistics
```

### Curriculum Dataset (Separated)
```
data/balanced_300m_curriculum/
├── wikitext_train.bin     # Facts (102M tokens)
├── stack_train.bin        # Logic (48M tokens)
├── ultrachat_train.bin    # Behavior (208M tokens)
├── val.bin                # Validation set
├── dataset_stats.json     # Statistics
└── README.md              # Documentation
```

## Configuration

The curriculum is configured in `config/gpu_training_117m_balanced.yaml`:

```yaml
# Training Configuration
training:
  max_epochs: 8  # 8 epochs for full curriculum
  
  curriculum_learning:
    enabled: true
    
    # Distribution per epoch: [wikitext%, stack%, ultrachat%]
    epoch_distributions:
      1:  [28, 13, 59]  # Current
      2:  [70, 20, 10]  # Foundation
      3:  [70, 20, 10]  # Foundation
      4:  [50, 25, 25]  # Bridge A
      5:  [35, 25, 40]  # Bridge B
      6:  [20, 20, 60]  # Bridge C
      7:  [10, 10, 80]  # Refinement
      8:  [10, 10, 80]  # Refinement
    
    sources:
      - "wikitext"   # Facts
      - "stack"      # Logic
      - "ultrachat"  # Behavior

# Data Configuration
data:
  dataset_type: "binary"
  train_file: "data/balanced_300m_curriculum/train.bin"
  val_file: "data/balanced_300m_curriculum/val.bin"
  
  # HuggingFace Auto-Download
  huggingface_dataset:
    repo_id: "0x-genesys/mix_wiki_code_chat_data_300M_tokens_curriculum"
    dataset_name: "balanced_300m_curriculum"
    auto_download: true
```

## Training Workflow

### 1. Start Training

```bash
python train.py --config config/gpu_training_117m_balanced.yaml
```

### 2. Automatic Download (First Run)

If dataset is missing, you'll see:

```
================================================================================
DOWNLOADING DATASET: balanced_300m_curriculum
================================================================================
Repository: 0x-genesys/mix_wiki_code_chat_data_300M_tokens_curriculum
Local path: data/balanced_300m_curriculum

📋 Checking repository contents...
Found 5 dataset files:
  - wikitext_train.bin
  - stack_train.bin
  - ultrachat_train.bin
  - val.bin
  - dataset_stats.json

📥 Downloading wikitext_train.bin...
✅ Downloaded wikitext_train.bin: 387.9 MB

📥 Downloading stack_train.bin...
✅ Downloaded stack_train.bin: 182.4 MB

📥 Downloading ultrachat_train.bin...
✅ Downloaded ultrachat_train.bin: 790.5 MB

📥 Downloading val.bin...
✅ Downloaded val.bin: 38.1 MB

📥 Downloading dataset_stats.json...
✅ Downloaded dataset_stats.json: 1.2 KB

✅ Dataset download complete!
```

### 3. Epoch Start

At the beginning of each epoch, the distribution updates:

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

### 4. Training Progress

Normal training with progress bar:

```
Epoch 2 | Step 5234/36621 | Loss: 3.1234 | LR: 2.00e-04:  45%|████▌     | 15234/33961
```

### 5. Epoch Transition

When moving to next epoch:

```
Epoch 2 completed. Average loss: 3.1234

================================================================================
🎓 CURRICULUM UPDATE - Epoch 3
================================================================================
Phase: Foundation: Knowledge/Logic Hardening
...
```

## Building Your Own Curriculum Dataset

### Step 1: Prepare Source Files

Split the combined dataset into separate source files:

```bash
python scripts/prepare_curriculum_dataset.py \
    --data-dir data/balanced_300m \
    --output-dir data/curriculum_temp
```

**What it does**:
1. Loads `data/balanced_300m/train.bin` (combined dataset)
2. Reads `dataset_stats.json` to find source boundaries
3. Splits into separate files:
   - `wikitext_train.bin` (102M tokens)
   - `stack_train.bin` (48M tokens)
   - `ultrachat_train.bin` (208M tokens)
4. Copies `val.bin` (unchanged)
5. Creates comprehensive `README.md`

**Output**:
```
data/curriculum_temp/
├── wikitext_train.bin     # 387.9 MB
├── stack_train.bin        # 182.4 MB
├── ultrachat_train.bin    # 790.5 MB
├── val.bin                # 38.1 MB
├── dataset_stats.json     # 1.2 KB
└── README.md              # Documentation
```

### Step 2: Upload to HuggingFace Hub

Upload the curriculum dataset:

```bash
python scripts/upload_curriculum_dataset.py \
    --data-dir data/balanced_300m \
    --repo-id 0x-genesys/mix_wiki_code_chat_data_300M_tokens_curriculum \
    --temp-dir data/curriculum_temp
```

**What it does**:
1. Runs Step 1 (prepare source files)
2. Creates HuggingFace repository (if doesn't exist)
3. Uploads all files:
   - `wikitext_train.bin`
   - `stack_train.bin`
   - `ultrachat_train.bin`
   - `val.bin`
   - `dataset_stats.json`
   - `README.md`
4. Displays dataset URL

**Output**:
```
================================================================================
✅ Upload complete!
================================================================================

Dataset URL: https://huggingface.co/datasets/0x-genesys/mix_wiki_code_chat_data_300M_tokens_curriculum

Files uploaded:
  - wikitext_train.bin
  - stack_train.bin
  - ultrachat_train.bin
  - val.bin
  - dataset_stats.json
  - README.md

To use in training:
  1. Update config:
     huggingface_dataset:
       repo_id: "0x-genesys/mix_wiki_code_chat_data_300M_tokens_curriculum"
       dataset_name: "balanced_300m_curriculum"
  2. Run training:
     python train.py --config config/gpu_training_117m_balanced.yaml
```

### Step 3: Configure and Train

The config is already set up! Just run:

```bash
python train.py --config config/gpu_training_117m_balanced.yaml
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
```

**Rules**:
- Each distribution must sum to 100
- Percentages are integers
- Epochs are 1-indexed (1, 2, 3, ...)

### Disable Curriculum Learning

To use standard fixed distribution:

```yaml
curriculum_learning:
  enabled: false  # Use standard single-file training
```

Or use the standard config:

```bash
# Use standard balanced dataset (no curriculum)
python train.py --config config/gpu_training_117m.yaml
```

## Monitoring

### TensorBoard

Monitor training progress:

```bash
tensorboard --logdir logs/gpu_training_117m_balanced
```

### Expected Loss Curves

- **Epoch 1**: Baseline loss
- **Epochs 2-3**: Loss may increase slightly (adapting to fact-heavy distribution)
- **Epochs 4-6**: Loss stabilizes (balanced distribution)
- **Epochs 7-8**: Loss decreases (refining behavior)

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

### Issue: Dataset Not Found

**Problem**: `FileNotFoundError: Dataset files not found`

**Solution**: Enable auto-download in config:

```yaml
data:
  huggingface_dataset:
    auto_download: true  # Enable this
```

### Issue: Source File Not Found

**Problem**: `FileNotFoundError: wikitext_train.bin not found`

**Solution**: The curriculum dataset hasn't been uploaded. Either:

1. Use the pre-built dataset (recommended):
   ```yaml
   huggingface_dataset:
     repo_id: "0x-genesys/mix_wiki_code_chat_data_300M_tokens_curriculum"
   ```

2. Build and upload your own:
   ```bash
   python scripts/upload_curriculum_dataset.py
   ```

### Issue: Distribution Doesn't Sum to 100

**Problem**: `ValueError: Distribution must sum to 100`

**Solution**: Check your config:

```yaml
epoch_distributions:
  1:  [28, 13, 59]  # 28 + 13 + 59 = 100 ✅
  2:  [70, 20, 15]  # 70 + 20 + 15 = 105 ❌ (should be [70, 20, 10])
```

### Issue: Out of Memory

**Problem**: `RuntimeError: CUDA out of memory`

**Solution**: Reduce batch size:

```yaml
training:
  batch_size: 4  # Reduce from 8
  gradient_accumulation_steps: 32  # Increase to maintain effective batch size
```

Or use low memory config:

```bash
python train.py --config config/gpu_training_117m_balanced_low_memory.yaml
```

## Comparison: Standard vs Curriculum

### Standard Training (Fixed Distribution)

```yaml
# config/gpu_training_117m.yaml
data:
  train_file: "data/balanced_300m/train.bin"  # Combined file
  
  huggingface_dataset:
    repo_id: "0x-genesys/mix_wiki_code_chat_data_300M_tokens"
```

- Single combined file
- Fixed distribution (34% WikiText, 16% Stack, 66% UltraChat)
- Simpler setup
- Faster epoch start

### Curriculum Training (Dynamic Distribution)

```yaml
# config/gpu_training_117m_balanced.yaml
data:
  train_file: "data/balanced_300m_curriculum/train.bin"
  
  huggingface_dataset:
    repo_id: "0x-genesys/mix_wiki_code_chat_data_300M_tokens_curriculum"

training:
  curriculum_learning:
    enabled: true
```

- Separate source files
- Dynamic distribution per epoch
- More complex setup
- Potentially better results

## Files and Scripts

### Scripts

- `scripts/prepare_curriculum_dataset.py` - Split combined dataset into source files
- `scripts/upload_curriculum_dataset.py` - Upload curriculum dataset to HuggingFace Hub

### Source Files

- `src/data.py` - `CurriculumDataset` class for dynamic mixing
- `src/trainer.py` - Curriculum learning logic in `train()` method
- `src/dataset_downloader.py` - Auto-download support for curriculum datasets

### Configuration

- `config/gpu_training_117m_balanced.yaml` - Main curriculum config
- `config/gpu_training_117m_balanced_low_memory.yaml` - Low memory variant

### Documentation

- `docs/CURRICULUM_LEARNING_GUIDE.md` - Comprehensive guide
- `docs/CURRICULUM_LEARNING_SETUP.md` - This file (quick setup)
- `CURRICULUM_LEARNING_READY.md` - Implementation status

## Next Steps

1. **Start Training**:
   ```bash
   python train.py --config config/gpu_training_117m_balanced.yaml
   ```

2. **Monitor Progress**:
   ```bash
   tensorboard --logdir logs/gpu_training_117m_balanced
   ```

3. **Evaluate Results**:
   ```bash
   python src/inference.py --model checkpoints/gpu_training_117m_balanced/best_model.pt --interactive
   ```

4. **Compare with Standard**:
   - Train standard model: `python train.py --config config/gpu_training_117m.yaml`
   - Compare perplexity, instruction-following, and conversational quality

## References

- [Curriculum Learning Paper](https://arxiv.org/abs/2101.10382)
- [Data Mixing Strategies](https://arxiv.org/abs/2210.11416)
- [HuggingFace Hub Documentation](https://huggingface.co/docs/hub/index)

## See Also

- [README.md](../README.md) - Main documentation
- [CURRICULUM_LEARNING_GUIDE.md](CURRICULUM_LEARNING_GUIDE.md) - Detailed guide
- [ARCHITECTURE.md](../ARCHITECTURE.md) - System architecture
- [GPU_MEMORY_GUIDE.md](GPU_MEMORY_GUIDE.md) - Memory optimization
