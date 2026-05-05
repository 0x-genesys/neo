# Curriculum Learning - Quick Reference

## 🚀 Quick Start

```bash
# Just start training - dataset downloads automatically!
python train.py --config config/gpu_training_117m_balanced.yaml
```

## 📊 The 8-Epoch Strategy

| Epoch | WikiText | Stack | UltraChat | Phase |
|-------|----------|-------|-----------|-------|
| 1 | 28% | 13% | 59% | Pattern Discovery |
| 2-3 | 70% | 20% | 10% | Foundation |
| 4 | 50% | 25% | 25% | Bridge A |
| 5 | 35% | 25% | 40% | Bridge B |
| 6 | 20% | 20% | 60% | Bridge C |
| 7-8 | 10% | 10% | 80% | Refinement |

## 📁 Dataset Structure

### HuggingFace Hub
- **Repo**: `0x-genesys/mix_wiki_code_chat_data_300M_tokens_curriculum`
- **Auto-download**: Enabled by default

### Local Files (after download)
```
data/balanced_300m_curriculum/
├── wikitext_train.bin     # 102M tokens (facts)
├── stack_train.bin        # 48M tokens (logic)
├── ultrachat_train.bin    # 208M tokens (behavior)
├── val.bin                # 38M tokens (validation)
└── dataset_stats.json     # Statistics
```

## ⚙️ Configuration

### Enable Curriculum Learning
```yaml
# config/gpu_training_117m_balanced.yaml
training:
  max_epochs: 8
  curriculum_learning:
    enabled: true
    sources: ["wikitext", "stack", "ultrachat"]
    epoch_distributions:
      1:  [28, 13, 59]
      2:  [70, 20, 10]
      # ... etc

data:
  huggingface_dataset:
    repo_id: "0x-genesys/mix_wiki_code_chat_data_300M_tokens_curriculum"
    dataset_name: "balanced_300m_curriculum"
    auto_download: true
```

### Disable Curriculum Learning
```yaml
curriculum_learning:
  enabled: false
```

Or use standard config:
```bash
python train.py --config config/gpu_training_117m.yaml
```

## 🛠️ Build Your Own

### Step 1: Prepare Dataset
```bash
python scripts/prepare_curriculum_dataset.py \
    --data-dir data/balanced_300m \
    --output-dir data/curriculum_temp
```

### Step 2: Upload to HuggingFace
```bash
python scripts/upload_curriculum_dataset.py \
    --data-dir data/balanced_300m \
    --repo-id YOUR_USERNAME/YOUR_DATASET_NAME \
    --temp-dir data/curriculum_temp
```

### Step 3: Update Config
```yaml
huggingface_dataset:
  repo_id: "YOUR_USERNAME/YOUR_DATASET_NAME"
```

### Step 4: Train
```bash
python train.py --config config/gpu_training_117m_balanced.yaml
```

## 📈 Monitoring

### TensorBoard
```bash
tensorboard --logdir logs/gpu_training_117m_balanced
```

### Expected Loss
- Epoch 1: ~3.5 (baseline)
- Epochs 2-3: ~3.6 (adapting to facts)
- Epochs 4-6: ~3.4 (balanced)
- Epochs 7-8: ~3.2 (refined)

## 🔧 Troubleshooting

### Dataset Not Found
```yaml
# Enable auto-download
data:
  huggingface_dataset:
    auto_download: true
```

### Out of Memory
```yaml
# Reduce batch size
training:
  batch_size: 4
  gradient_accumulation_steps: 32
```

### Distribution Error
```yaml
# Must sum to 100
epoch_distributions:
  1:  [28, 13, 59]  # ✅ = 100
  2:  [70, 20, 15]  # ❌ = 105
```

## 📚 Documentation

- **Quick Setup**: `docs/CURRICULUM_LEARNING_SETUP.md`
- **Comprehensive Guide**: `docs/CURRICULUM_LEARNING_GUIDE.md`
- **Status**: `CURRICULUM_LEARNING_COMPLETE.md`
- **This File**: `CURRICULUM_QUICK_REFERENCE.md`

## 🔗 Links

- **Model Repo**: [0x-genesys/neo_weights_checkpoints](https://huggingface.co/0x-genesys/neo_weights_checkpoints)
- **Standard Dataset**: [0x-genesys/mix_wiki_code_chat_data_300M_tokens](https://huggingface.co/datasets/0x-genesys/mix_wiki_code_chat_data_300M_tokens)
- **Curriculum Dataset**: [0x-genesys/mix_wiki_code_chat_data_300M_tokens_curriculum](https://huggingface.co/datasets/0x-genesys/mix_wiki_code_chat_data_300M_tokens_curriculum)

## ✅ Status

**READY TO USE** - All features implemented and tested!
