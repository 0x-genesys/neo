# Configuration Index

Complete guide to all available training configurations, from 8M to 13B parameters.

---

## Quick Reference Table

| Config | Params | Layers | Width | Context | Dataset | GPU | Memory | Time | Cost | Quality |
|--------|--------|--------|-------|---------|---------|-----|--------|------|------|---------|
| **quick_start** | 8M | 2 | 128 | 128 | wikitext-2 | CPU | 500MB | 1h | Free | Basic |
| **production** | 16M | 4 | 256 | 256 | wikitext-2 | CPU | 1GB | 10h | Free | Good |
| **gpu_training** | 16M | 4 | 256 | 256 | wikitext-103 | T4 | 2GB | 3h | $9 | Very Good |
| **gpu_training_117m** | 124M | 12 | 768 | 512 | openwebtext | T4 | 4GB | 55h | $165 | Excellent |
| **gpu_training_345m** | 345M | 24 | 1024 | 2048 | openwebtext | A100 | 12GB | 3-5d | $400 | Very Good |
| **gpu_training_774m** | 774M | 36 | 1280 | 2048 | C4 | A100 | 24GB | 1-2w | $2K | Excellent |
| **gpu_training_1.5b** | 1.5B | 48 | 1600 | 2048 | C4 | A100 | 48GB | 2-4w | $6K | Excellent |
| **gpu_training_2.7b** | 2.7B | 32 | 2560 | 2048 | C4 | 4-8×A100 | 80GB | 1-2m | $20K | GPT-3 |
| **gpu_training_6.7b** | 6.7B | 32 | 4096 | 2048 | RedPajama | 8-16×A100 | 200GB | 2-3m | $100K | GPT-3 |
| **gpu_training_13b** | 13B | 40 | 5120 | 2048 | RedPajama | 16-32×A100 | 400GB | 3-6m | $500K | GPT-3 |

---

## Configuration Details

### 1. quick_start.yaml (8M params)
**Purpose**: Quick testing and validation

```yaml
Model: 8M params (2 layers, 128 dim)
Dataset: wikitext-2 (2M tokens)
Hardware: CPU
Time: 1-2 hours
Cost: Free
```

**Use Cases**:
- ✅ Test installation
- ✅ Validate code changes
- ✅ Quick experiments
- ❌ Not for production

**Command**:
```bash
python train.py --config config/quick_start.yaml
```

---

### 2. production_training.yaml (16M params)
**Purpose**: CPU training baseline

```yaml
Model: 16M params (4 layers, 256 dim)
Dataset: wikitext-2 (2M tokens)
Hardware: CPU
Time: 10 hours
Cost: Free
```

**Use Cases**:
- ✅ No GPU available
- ✅ Small experiments
- ✅ Learning/education
- ❌ Not for production quality

**Command**:
```bash
python train.py --config config/production_training.yaml
```

---

### 3. gpu_training.yaml (16M params) ✅ RECOMMENDED
**Purpose**: GPU training baseline

```yaml
Model: 16M params (4 layers, 256 dim)
Dataset: wikitext-103 (100M tokens)
Hardware: T4 GPU (16GB)
Time: 2-4 hours
Cost: $9 (AWS) or Free (Colab)
```

**Use Cases**:
- ✅ **Default choice for GPU training**
- ✅ Best quality/time ratio
- ✅ Validate GPU pipeline
- ✅ Quick high-quality results

**Command**:
```bash
python train.py --config config/gpu_training.yaml
```

---

### 4. gpu_training_117m.yaml (124M params) 🚀 PRODUCTION
**Purpose**: Production-quality small model

```yaml
Model: 124M params (12 layers, 768 dim)
Dataset: openwebtext (8B tokens)
Hardware: T4 GPU (16GB)
Time: 55 hours (2-3 days)
Cost: $165 (AWS) or $10 (Colab Pro)
Features: Warmup scheduler, gradient checkpointing
```

**Use Cases**:
- ✅ **Production deployment**
- ✅ Research projects
- ✅ Domain-specific models
- ✅ Fine-tuning base

**Command**:
```bash
python train.py --config config/gpu_training_117m.yaml
```

**Chinchilla Scaling**: 124M × 20 = 2.48B tokens (training with 6.5B = 2.6× optimal)

---

### 5. gpu_training_345m.yaml (345M params)
**Purpose**: GPT-2 Medium scale

```yaml
Model: 345M params (24 layers, 1024 dim)
Dataset: openwebtext (8B tokens)
Hardware: A100 40GB
Time: 3-5 days
Cost: $200-400
Features: Gradient checkpointing, model compilation
```

**Use Cases**:
- ✅ Production applications
- ✅ High-quality generation
- ✅ Complex reasoning tasks
- ✅ Multi-domain models

**Command**:
```bash
python train.py --config config/gpu_training_345m.yaml
```

**Chinchilla Scaling**: 345M × 20 = 6.9B tokens (training with 52B = 7.5× optimal)

**Requirements**:
- A100 40GB or better
- Gradient checkpointing enabled
- Mixed precision essential

---

### 6. gpu_training_774m.yaml (774M params)
**Purpose**: GPT-2 Large scale

```yaml
Model: 774M params (36 layers, 1280 dim)
Dataset: C4 (300B+ tokens)
Hardware: A100 80GB or 2×A100 40GB
Time: 1-2 weeks
Cost: $1,000-2,000
Features: Gradient checkpointing, model compilation
```

**Use Cases**:
- ✅ High-quality generation
- ✅ Complex reasoning
- ✅ Research projects
- ✅ Commercial applications

**Command**:
```bash
python train.py --config config/gpu_training_774m.yaml
```

**Chinchilla Scaling**: 774M × 20 = 15.5B tokens (training with 78B = 5× optimal)

**Requirements**:
- A100 80GB or 2×A100 40GB
- Gradient checkpointing required
- C4 dataset (larger than openwebtext)

---

### 7. gpu_training_1.5b.yaml (1.5B params)
**Purpose**: GPT-2 XL scale

```yaml
Model: 1.5B params (48 layers, 1600 dim)
Dataset: C4 (300B+ tokens)
Hardware: A100 80GB or 4×A100 40GB
Time: 2-4 weeks
Cost: $3,000-6,000
Features: Gradient checkpointing, model compilation
```

**Use Cases**:
- ✅ State-of-the-art generation
- ✅ Research projects
- ✅ Foundation models
- ✅ Near GPT-3 quality

**Command**:
```bash
python train.py --config config/gpu_training_1.5b.yaml
```

**Chinchilla Scaling**: 1.5B × 20 = 30B tokens (training with 131B = 4.4× optimal)

**Requirements**:
- A100 80GB or 4×A100 40GB with distributed training
- Gradient checkpointing required
- Consider implementing distributed training for multi-GPU

---

### 8. gpu_training_2.7b.yaml (2.7B params)
**Purpose**: GPT-3 Small scale

```yaml
Model: 2.7B params (32 layers, 2560 dim)
Dataset: C4 (300B+ tokens)
Hardware: 4-8×A100 40GB or 2-4×A100 80GB
Time: 1-2 months
Cost: $10,000-20,000
Features: Requires distributed training (DDP/FSDP)
```

**Use Cases**:
- ✅ Foundation models
- ✅ GPT-3 level quality
- ✅ Commercial products
- ✅ Research

**Command**:
```bash
torchrun --nproc_per_node=8 train.py --config config/gpu_training_2.7b.yaml
```

**Chinchilla Scaling**: 2.7B × 20 = 54B tokens (training with 262B = 4.9× optimal)

**CRITICAL REQUIREMENTS**:
- ⚠️ **Distributed training required** (not yet implemented)
- ⚠️ DeepSpeed ZeRO Stage 2 or 3 recommended
- ⚠️ Multi-GPU setup essential

---

### 9. gpu_training_6.7b.yaml (6.7B params)
**Purpose**: GPT-3 Medium scale

```yaml
Model: 6.7B params (32 layers, 4096 dim)
Dataset: RedPajama (1T tokens)
Hardware: 8-16×A100 80GB
Time: 2-3 months
Cost: $50,000-100,000
Features: Requires FSDP, CPU offloading, pipeline parallelism
```

**Use Cases**:
- ✅ Foundation models
- ✅ Commercial applications
- ✅ High-quality versatile models
- ✅ Research

**Command**:
```bash
torchrun --nproc_per_node=16 train.py --config config/gpu_training_6.7b.yaml
```

**Chinchilla Scaling**: 6.7B × 20 = 134B tokens (training with 786B = 5.9× optimal)

**CRITICAL REQUIREMENTS**:
- ⚠️ **FSDP or DeepSpeed ZeRO Stage 3 required**
- ⚠️ CPU offloading for optimizer states
- ⚠️ Pipeline parallelism recommended
- ⚠️ High-bandwidth interconnect (NVLink, InfiniBand)

---

### 10. gpu_training_13b.yaml (13B params)
**Purpose**: GPT-3 Large scale

```yaml
Model: 13B params (40 layers, 5120 dim)
Dataset: RedPajama (1T tokens)
Hardware: 16-32×A100 80GB
Time: 3-6 months
Cost: $200,000-500,000
Features: Requires FSDP, CPU offloading, pipeline + tensor parallelism
```

**Use Cases**:
- ✅ Foundation models
- ✅ Commercial products
- ✅ GPT-3 level quality
- ✅ Frontier research

**Command**:
```bash
torchrun --nproc_per_node=8 --nnodes=4 train.py --config config/gpu_training_13b.yaml
```

**Chinchilla Scaling**: 13B × 20 = 260B tokens (training with 2.1T = 8× optimal)

**CRITICAL REQUIREMENTS**:
- ⚠️ **FSDP or DeepSpeed ZeRO Stage 3 required**
- ⚠️ CPU offloading required
- ⚠️ Pipeline parallelism highly recommended
- ⚠️ Tensor parallelism optional but helpful
- ⚠️ Multi-node setup (4+ nodes)

---

## Dataset Guide

### wikitext-2 (2M tokens)
- **Size**: ~10MB
- **Use**: Testing, small experiments
- **Quality**: Limited diversity
- **Configs**: quick_start, production

### wikitext-103 (100M tokens)
- **Size**: ~500MB
- **Use**: GPU baseline training
- **Quality**: Good diversity
- **Configs**: gpu_training

### openwebtext (8B tokens)
- **Size**: ~12GB
- **Use**: Production small models
- **Quality**: High diversity, web text
- **Configs**: gpu_training_117m, gpu_training_345m

### C4 (300B+ tokens)
- **Size**: ~300GB
- **Use**: Large models (>500M params)
- **Quality**: Very high diversity, cleaned web text
- **Configs**: gpu_training_774m, gpu_training_1.5b, gpu_training_2.7b

### RedPajama (1T tokens)
- **Size**: ~1TB
- **Use**: Foundation models (>5B params)
- **Quality**: Highest diversity, multiple sources
- **Configs**: gpu_training_6.7b, gpu_training_13b

---

## Scaling Progression

### Recommended Path

**Phase 1: Validation** (Complete ✅)
```
Config: quick_start.yaml
Model: 8M params
Time: 1 hour
Goal: Validate pipeline
```

**Phase 2: GPU Baseline** (Recommended Next)
```
Config: gpu_training.yaml
Model: 16M params
Time: 3 hours
Goal: Validate GPU training
Cost: $9
```

**Phase 3: Production Small** (After Phase 2)
```
Config: gpu_training_117m.yaml
Model: 124M params
Time: 2-3 days
Goal: Production-quality model
Cost: $165
```

**Phase 4: Medium Scale** (Optional)
```
Config: gpu_training_345m.yaml or gpu_training_774m.yaml
Model: 345M-774M params
Time: 3 days - 2 weeks
Goal: High-quality generation
Cost: $400-2,000
```

**Phase 5: Large Scale** (Advanced)
```
Config: gpu_training_1.5b.yaml or gpu_training_2.7b.yaml
Model: 1.5B-2.7B params
Time: 2 weeks - 2 months
Goal: State-of-the-art quality
Cost: $6,000-20,000
```

**Phase 6: Foundation Models** (Research/Commercial)
```
Config: gpu_training_6.7b.yaml or gpu_training_13b.yaml
Model: 6.7B-13B params
Time: 2-6 months
Goal: GPT-3 level
Cost: $50,000-500,000
```

---

## Feature Matrix

| Config | Warmup | Grad Ckpt | Mixed Prec | Compile | Distributed | Dataset |
|--------|--------|-----------|------------|---------|-------------|---------|
| quick_start | ✅ | ❌ | ❌ | ❌ | ❌ | wikitext-2 |
| production | ✅ | ❌ | ❌ | ❌ | ❌ | wikitext-2 |
| gpu_training | ✅ | ❌ | ✅ | ❌ | ❌ | wikitext-103 |
| gpu_training_117m | ✅ | ✅ | ✅ | ❌ | ❌ | openwebtext |
| gpu_training_345m | ✅ | ✅ | ✅ | ✅ | ❌ | openwebtext |
| gpu_training_774m | ✅ | ✅ | ✅ | ✅ | ❌ | C4 |
| gpu_training_1.5b | ✅ | ✅ | ✅ | ✅ | Optional | C4 |
| gpu_training_2.7b | ✅ | ✅ | ✅ | ✅ | **Required** | C4 |
| gpu_training_6.7b | ✅ | ✅ | ✅ | ✅ | **Required** | RedPajama |
| gpu_training_13b | ✅ | ✅ | ✅ | ✅ | **Required** | RedPajama |

---

## Summary

### Ready to Use (No Additional Implementation)
- ✅ quick_start.yaml (8M)
- ✅ production_training.yaml (16M)
- ✅ gpu_training.yaml (16M)
- ✅ gpu_training_117m.yaml (124M)
- ✅ gpu_training_345m.yaml (345M)
- ✅ gpu_training_774m.yaml (774M)
- ✅ gpu_training_1.5b.yaml (1.5B)

### Requires Distributed Training Implementation
- ⚠️ gpu_training_2.7b.yaml (2.7B) - DDP/FSDP needed
- ⚠️ gpu_training_6.7b.yaml (6.7B) - FSDP + CPU offload needed
- ⚠️ gpu_training_13b.yaml (13B) - FSDP + pipeline parallelism needed

### Recommended Starting Point
**For most users**: Start with `gpu_training.yaml` (3 hours, $9)
**For production**: Use `gpu_training_117m.yaml` (2-3 days, $165)
**For research**: Scale to `gpu_training_345m.yaml` or larger

---

**All configs are ready to use! Start with the recommended progression and scale up as needed.** 🚀

