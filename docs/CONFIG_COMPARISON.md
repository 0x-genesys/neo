# Configuration Comparison

## Quick Reference

| Config | Hardware | Time | Steps | Dataset | Loss | Quality | Use Case |
|--------|----------|------|-------|---------|------|---------|----------|
| **quick_start.yaml** | CPU | 1-2h | 500 | wikitext-2 | ~1.0 | Basic | Testing |
| **production_training.yaml** | CPU | 10h | 10K | wikitext-2 | ~0.7 | Okay | CPU training |
| **gpu_training.yaml** | T4 GPU | 3h | 50K | wikitext-103 | ~0.3 | Good | **Recommended** |
| **model_config.yaml** | A100 GPU | 20h | 100K | wikitext-103 | ~0.25 | Excellent | Large scale |

---

## Detailed Comparison

### Quick Start (Testing)
```yaml
# config/quick_start.yaml
model: 16M params (2 layers, 128 dim)
batch_size: 8
max_steps: 500
dataset: wikitext-2 (2M tokens)
device: CPU
time: 1-2 hours
```

**Purpose**: Quick test to verify everything works
**Quality**: Basic, repetitive
**When to use**: Initial setup, debugging

---

### Production Training (CPU)
```yaml
# config/production_training.yaml
model: 16M params (4 layers, 256 dim)
batch_size: 8
max_steps: 10,000
dataset: wikitext-2 (2M tokens)
device: CPU
time: 10 hours
```

**Purpose**: Train on CPU when GPU not available
**Quality**: Okay, some coherence
**When to use**: No GPU access, small experiments

---

### GPU Training (Recommended)
```yaml
# config/gpu_training.yaml
model: 16M params (4 layers, 256 dim)
batch_size: 32
max_steps: 50,000
dataset: wikitext-103 (100M tokens)
device: T4 GPU
time: 2-4 hours
mixed_precision: true
```

**Purpose**: High-quality training on GPU
**Quality**: Good, coherent, diverse
**When to use**: **Default choice for GPU training**

**Key improvements**:
- 4x larger batches (GPU parallelism)
- 5x more steps (better convergence)
- 100x more data (better generalization)
- 2x faster (mixed precision)
- 3x faster overall (vs CPU)

---

### Model Config (Large Scale)
```yaml
# config/model_config.yaml
model: 40M params (6 layers, 512 dim)
batch_size: 32
max_steps: 100,000
dataset: wikitext-103 (100M tokens)
device: A100 GPU
time: 20 hours
```

**Purpose**: Larger model for best quality
**Quality**: Excellent
**When to use**: Production deployment, research

---

## Which Config Should I Use?

### Decision Tree

```
Do you have a GPU?
│
├─ No → production_training.yaml (CPU)
│        Time: 10 hours
│        Quality: Okay
│
└─ Yes → What GPU?
         │
         ├─ T4 / V100 (16GB) → gpu_training.yaml ✅
         │                      Time: 3 hours
         │                      Quality: Good
         │                      **RECOMMENDED**
         │
         ├─ A100 (40GB+) → model_config.yaml
         │                 Time: 20 hours
         │                 Quality: Excellent
         │
         └─ Testing only? → quick_start.yaml
                            Time: 1 hour
                            Quality: Basic
```

---

## Training Steps Explained

### Why Different Step Counts?

**Quick Start (500 steps)**:
- Dataset: wikitext-2 (2M tokens)
- Tokens per step: 8 × 256 × 2 = 4,096
- Total tokens: 500 × 4,096 = 2M tokens
- **Sees dataset once** (1 epoch)
- Purpose: Quick test

**Production CPU (10,000 steps)**:
- Dataset: wikitext-2 (2M tokens)
- Tokens per step: 8 × 256 × 4 = 8,192
- Total tokens: 10,000 × 8,192 = 82M tokens
- **Sees dataset 40 times** (40 epochs)
- Purpose: Learn from limited data

**GPU Training (50,000 steps)**:
- Dataset: wikitext-103 (100M tokens)
- Tokens per step: 32 × 256 × 4 = 32,768
- Total tokens: 50,000 × 32,768 = 1.6B tokens
- **Sees dataset 16 times** (16 epochs)
- Purpose: Learn from diverse data

### Why More Steps on GPU?

1. **Larger dataset** (100M vs 2M tokens)
   - More diverse data to learn from
   - Needs more steps to see all patterns

2. **Faster training** (GPU vs CPU)
   - Same wall-clock time (3 hours)
   - Can afford more steps

3. **Better convergence** (larger batches)
   - Batch size 32 vs 8
   - More stable gradients
   - Can train longer without overfitting

4. **Higher quality target**
   - CPU: Loss ~0.7 (okay)
   - GPU: Loss ~0.3 (good)
   - Needs more steps to reach lower loss

---

## Max Epochs vs Max Steps

### All Configs Have Both
```yaml
max_epochs: 10
max_steps: 50000
```

**Training stops at whichever comes first**

### Why This Design?

1. **Safety**: Prevents infinite training
2. **Flexibility**: Can limit by epochs OR steps
3. **Predictability**: Step-based is more consistent

### What Actually Happens?

**Quick Start**:
- 500 steps ≈ 0.17 epochs
- **Stops at 500 steps** ✅

**Production CPU**:
- 10,000 steps ≈ 3.4 epochs
- **Stops at 10,000 steps** ✅

**GPU Training**:
- 50,000 steps ≈ 15.5 epochs
- **Stops at 10 epochs** ⚠️ (if not changed)
- **Recommendation**: Remove max_epochs limit or set to 20

### Fix for GPU Training

Option 1: Remove epoch limit
```yaml
max_epochs: 999  # Effectively unlimited
max_steps: 50000  # Will stop here
```

Option 2: Calculate epochs needed
```yaml
max_epochs: 20  # Enough for 50K steps
max_steps: 50000
```

---

## GPU Utilization

### Current Configs

| Config | Batch Size | GPU Usage | Speed (steps/sec) |
|--------|------------|-----------|-------------------|
| production (CPU) | 8 | N/A | 0.15 |
| gpu_training | 32 | 60-70% | 2-3 |
| model_config | 32 | 80-90% | 1-2 |

### How to Increase GPU Utilization

**If using <60% GPU**:

1. Increase batch size:
```yaml
batch_size: 48  # or 64
```

2. Increase context length:
```yaml
context_length: 512  # double
```

3. Enable compilation:
```yaml
compile_model: true
```

**Monitor with**:
```bash
nvidia-smi -l 1
```

---

## Cost Comparison

### Cloud GPU Pricing (approximate)

| Config | Hardware | Time | Cost (AWS) | Cost (Colab) |
|--------|----------|------|------------|--------------|
| quick_start | CPU | 1h | $0.10 | Free |
| production | CPU | 10h | $1.00 | Free |
| gpu_training | T4 | 3h | $1.50 | Free |
| model_config | A100 | 20h | $60 | $10 |

**Recommendation**: Use **gpu_training.yaml** on Google Colab (free T4 GPU)

---

## Summary

### For Most Users
**Use `config/gpu_training.yaml`**
- ✅ Best quality/time tradeoff
- ✅ 60-70% GPU utilization
- ✅ 3 hours on T4 GPU
- ✅ Loss ~0.3 (good quality)
- ✅ Free on Google Colab

### For CPU Users
**Use `config/production_training.yaml`**
- ✅ Optimized for CPU
- ✅ 10 hours training
- ✅ Loss ~0.7 (okay quality)

### For Testing
**Use `config/quick_start.yaml`**
- ✅ Quick verification
- ✅ 1 hour training
- ✅ Basic quality

### For Production
**Use `config/model_config.yaml`**
- ✅ Larger model (40M params)
- ✅ Best quality
- ✅ Requires A100 GPU

---

## Quick Start Commands

```bash
# Testing (1 hour, CPU)
python train.py --config config/quick_start.yaml

# CPU Training (10 hours)
python train.py --config config/production_training.yaml

# GPU Training (3 hours) - RECOMMENDED
python train.py --config config/gpu_training.yaml

# Large Scale (20 hours, A100)
python train.py --config config/model_config.yaml
```

---

**For GPU training, use `gpu_training.yaml` - it's optimized for 60-70% GPU utilization and will give you high-quality results in 2-4 hours!** 🚀
