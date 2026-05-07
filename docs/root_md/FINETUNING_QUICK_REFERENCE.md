# 🚀 Fine-Tuning Quick Reference

## Data Preparation

### HuggingFace Datasets (Recommended)
```bash
# Pull Orca Math, Dolly, CodeAlpaca (25k samples)
python scripts/prepare_finetuning_data.py hf --output-dir data/hf_cot

# With custom max tokens
python scripts/prepare_finetuning_data.py hf --output-dir data/hf_cot --max-tokens 480
```

### Sample Data (Testing)
```bash
# Create sample data (2k samples)
python scripts/prepare_finetuning_data.py all --output-dir data

# Custom sample sizes
python scripts/prepare_finetuning_data.py all \
    --output-dir data \
    --math-samples 1000 \
    --code-samples 500 \
    --qa-samples 500
```

## Fine-Tuning

### Basic Fine-Tuning
```bash
# Local model
python src/finetuning/gpu_finetune.py \
    --model checkpoints/best_model.pt \
    --train-data data/hf_cot/train.jsonl \
    --val-data data/hf_cot/val.jsonl

# Remote model (HuggingFace Hub)
python src/finetuning/gpu_finetune.py \
    --model-remote best_model.pt \
    --train-data data/hf_cot/train.jsonl \
    --val-data data/hf_cot/val.jsonl
```

### Resume Training
```bash
# Resume from local checkpoint
python src/finetuning/gpu_finetune.py \
    --resume finetuned_model_gpu/checkpoint_epoch_1.pt \
    --train-data data/hf_cot/train.jsonl

# Resume from HuggingFace Hub
python src/finetuning/gpu_finetune.py \
    --resume-remote finetuned_checkpoint.pt \
    --model-repo your-username/your-repo \
    --train-data data/hf_cot/train.jsonl
```

### Custom Configuration
```bash
# Full custom configuration
python src/finetuning/gpu_finetune.py \
    --model-remote best_model.pt \
    --model-repo 0x-genesys/neo_weights_checkpoints \
    --train-data data/hf_cot/train.jsonl \
    --val-data data/hf_cot/val.jsonl \
    --output-dir my_finetuned_model \
    --batch-size 16 \
    --epochs 5 \
    --lr 1.5e-5
```

## Command-Line Options

### Model Options
```bash
--model PATH                    # Local pre-trained model
--model-remote FILENAME         # Remote model from HuggingFace Hub
--model-repo REPO_ID            # HuggingFace repository ID
```

### Data Options
```bash
--train-data PATH               # Training data (JSONL)
--val-data PATH                 # Validation data (JSONL)
```

### Training Options
```bash
--output-dir DIR                # Output directory
--batch-size N                  # Batch size (default: 8)
--epochs N                      # Number of epochs (default: 3)
--lr FLOAT                      # Learning rate (default: 2.0e-5)
```

### Resume Options
```bash
--resume PATH                   # Resume from local checkpoint
--resume-remote FILENAME        # Resume from HuggingFace Hub
```

## Common Workflows

### 1. Quick Start (5 minutes)
```bash
# Prepare sample data
python scripts/prepare_finetuning_data.py all

# Fine-tune
python src/finetuning/gpu_finetune.py
```

### 2. Production (Real Datasets)
```bash
# Pull HuggingFace datasets
python scripts/prepare_finetuning_data.py hf --output-dir data/hf_cot

# Fine-tune with remote model
python src/finetuning/gpu_finetune.py \
    --model-remote best_model.pt \
    --train-data data/hf_cot/train.jsonl \
    --val-data data/hf_cot/val.jsonl
```

### 3. Resume After Interruption
```bash
# Resume from last checkpoint
python src/finetuning/gpu_finetune.py \
    --resume finetuned_model_gpu/checkpoint_epoch_2.pt \
    --train-data data/hf_cot/train.jsonl
```

### 4. Custom Repository
```bash
# Use your own model repository
python src/finetuning/gpu_finetune.py \
    --model-remote my_checkpoint.pt \
    --model-repo your-username/your-model-repo \
    --train-data data/hf_cot/train.jsonl
```

## Dataset Information

### HuggingFace Datasets
- **Orca Math**: 5,000 math word problems
- **Dolly**: 15,000 general Q&A samples
- **CodeAlpaca**: 5,000 code generation tasks
- **Total**: ~25,000 samples (after filtering)
- **Split**: 90% train, 10% val

### Sample Datasets
- **Math**: 1,000 arithmetic problems
- **Code**: 500 Python functions
- **Q&A**: 500 general questions
- **Total**: 2,000 samples
- **Split**: 90% train, 10% val

## Tips

### Memory Optimization
```bash
# Reduce batch size if OOM
--batch-size 4

# Increase if memory available
--batch-size 16
```

### Learning Rate
```bash
# Default (recommended)
--lr 2.0e-5

# Lower for stability
--lr 1.0e-5

# Higher for faster convergence
--lr 3.0e-5
```

### Epochs
```bash
# Quick test
--epochs 1

# Recommended
--epochs 3

# Extended training
--epochs 5
```

## Validation

### Check Data Format
```bash
python scripts/prepare_finetuning_data.py validate data/hf_cot/train.jsonl
```

### Test Setup
```bash
python scripts/test_finetuning_setup.py
```

## Output

### Checkpoints
```
finetuned_model_gpu/
├── checkpoint_epoch_1.pt       # After epoch 1
├── checkpoint_epoch_2.pt       # After epoch 2
├── checkpoint_epoch_3.pt       # After epoch 3
├── best_model/                 # Best model (lowest val loss)
│   ├── adapter_config.json
│   └── adapter_model.bin
└── training_state.pt           # Training state
```

### Logs
```
finetuned_model_gpu/
└── logs/
    └── events.out.tfevents.*   # TensorBoard logs
```

## Troubleshooting

### Dataset Download Issues
```bash
# Check internet connection
# Verify dataset names
# Try manual download:
python -c "from datasets import load_dataset; load_dataset('microsoft/orca-math-word-problems-200k')"
```

### OOM (Out of Memory)
```bash
# Reduce batch size
--batch-size 4

# Or use CPU (slower)
python src/finetuning/cpu_finetune.py
```

### Resume Issues
```bash
# Check checkpoint path
ls finetuned_model_gpu/

# Verify checkpoint format
python -c "import torch; print(torch.load('checkpoint.pt').keys())"
```

## Documentation

- **README.md** - Complete guide
- **FINETUNING_ENHANCEMENTS.md** - Detailed enhancements
- **FINETUNING_COMPLETE.md** - Original implementation
- **docs/FINETUNING_GUIDE.md** - Comprehensive guide
- **FINETUNING_QUICK_REFERENCE.md** - This file

---

**🚀 Happy Fine-Tuning!**
