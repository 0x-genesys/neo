# Balanced Dataset Quick Start

## 🚀 One-Command Setup

```bash
bash scripts/prepare_and_train.sh
```

This will automatically:
1. Install dependencies
2. Prepare 300M token dataset
3. Verify integrity
4. Start training

---

## 📋 Manual Setup (3 Steps)

### Step 1: Prepare Dataset (2-4 hours)

```bash
python scripts/prepare_balanced_dataset.py
```

**Output**: `data/balanced_300m/train.bin` (1.2GB) + `val.bin` (40MB)

### Step 2: Verify Dataset

```bash
python scripts/test_balanced_dataset.py
```

**Expected**: All tests pass ✅

### Step 3: Train Model (10-20 hours)

```bash
python train.py --config config/gpu_training_117m_1.5gb.yaml
```

**Output**: Checkpoints in `checkpoints/gpu_training_117m_1.5gb_balanced/`

---

## 📊 Dataset Composition

| Source | Tokens | Purpose |
|--------|--------|---------|
| WikiText-103 | 102M (34%) | Knowledge |
| UltraChat | 150M (50%) | Conversation |
| The Stack | 48M (16%) | Code |
| DailyDialog | 10M (3%) | Dialogue |
| **TOTAL** | **300M** | **Balanced** |

---

## ⚙️ Training Configuration

```yaml
Epochs: 8
Total tokens: 2.4B (300M × 8)
Steps: 36,621
Learning rate: 2.0e-4 → 2.0e-5 (cosine)
Batch size: 16 (effective: 128)
Context length: 256
```

---

## 📁 Key Files

### Scripts
- `scripts/prepare_balanced_dataset.py` - Dataset builder
- `scripts/test_balanced_dataset.py` - Validator
- `scripts/prepare_and_train.sh` - Automation

### Config
- `config/gpu_training_117m_1.5gb.yaml` - Training config

### Docs
- `docs/BALANCED_DATASET_GUIDE.md` - Full guide
- `docs/BALANCED_DATASET_SUMMARY.md` - Implementation details

---

## ⏱️ Time Estimates

| Task | 1.5GB GPU | Larger GPU |
|------|-----------|------------|
| Dataset prep | 2-4 hours | 2-4 hours |
| Training | 10-20 hours | 1-2 hours |
| **Total** | **12-24 hours** | **3-6 hours** |

---

## 🔍 Verify Installation

```bash
# Check dependencies
pip install tiktoken datasets numpy tqdm

# Test tiktoken
python -c "import tiktoken; print(tiktoken.get_encoding('cl100k_base').n_vocab)"
# Expected: 100277

# Test datasets
python -c "from datasets import load_dataset; print('OK')"
# Expected: OK
```

---

## 🐛 Common Issues

### Issue: Dataset download fails
**Fix**: `huggingface-cli login`

### Issue: Out of memory during prep
**Fix**: Use machine with 16GB+ RAM

### Issue: Binary files not found
**Fix**: Check paths in `config/gpu_training_117m_1.5gb.yaml`

### Issue: Vocab size mismatch
**Fix**: `pip install tiktoken`

---

## 📈 Expected Results

### Training Metrics
- Training loss: ~2.5-3.0
- Validation loss: ~2.8-3.2
- Perplexity: ~15-25

### Model Capabilities
- ✅ Encyclopedic knowledge
- ✅ Conversational ability
- ✅ Code understanding
- ✅ Natural dialogue

---

## 💡 Next Steps After Training

```bash
# 1. Evaluate model
python evaluate.py --checkpoint checkpoints/gpu_training_117m_1.5gb_balanced/best_model.pt

# 2. Generate text
python src/inference.py --checkpoint checkpoints/gpu_training_117m_1.5gb_balanced/best_model.pt

# 3. View training logs
tensorboard --logdir logs/gpu_training_117m_1.5gb_balanced
```

---

## 📚 Full Documentation

For detailed information, see:
- **Setup Guide**: `docs/BALANCED_DATASET_GUIDE.md`
- **Implementation**: `docs/BALANCED_DATASET_SUMMARY.md`
- **Training Guide**: `docs/GPU_TRAINING_GUIDE.md`

---

## 🎯 Quick Commands Reference

```bash
# Prepare dataset
python scripts/prepare_balanced_dataset.py --output-dir data/balanced_300m --seed 42

# Test dataset
python scripts/test_balanced_dataset.py --data-dir data/balanced_300m

# Train model
python train.py --config config/gpu_training_117m_1.5gb.yaml

# Resume training
python train.py --config config/gpu_training_117m_1.5gb.yaml --resume checkpoints/gpu_training_117m_1.5gb_balanced/checkpoint.pt

# Monitor GPU
nvidia-smi -l 1

# View logs
tensorboard --logdir logs/gpu_training_117m_1.5gb_balanced
```

---

**Ready to start? Run**: `bash scripts/prepare_and_train.sh` 🚀
