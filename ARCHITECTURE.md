

# Neo Transformer - System Architecture

## Overview

Neo is a production-ready transformer language model implementation designed for robustness, scalability, and ease of use. The system supports training from scratch, fine-tuning, and inference with comprehensive multi-environment support.

## System Design Principles

1. **Robustness**: Multi-environment support (CUDA, MPS, TPU, CPU) with graceful degradation
2. **Compatibility**: PyTorch 2.0+ version compatibility with automatic API detection
3. **Scalability**: Multi-GPU and TPU training with memory optimization
4. **Usability**: Automatic dataset download, remote model loading, comprehensive documentation
5. **Production-Ready**: Error handling, checkpointing, monitoring, and deployment support

## Architecture Layers

```
┌─────────────────────────────────────────────────────────────┐
│                     User Interface Layer                     │
│  CLI (train.py, inference.py) │ Scripts │ Configuration     │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                         │
│  Training │ Inference │ Evaluation │ Data Management        │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                      Core Layer                              │
│  Model │ Trainer │ Data Loader │ Tokenizer │ Optimizer     │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                   Infrastructure Layer                       │
│  PyTorch │ CUDA │ HuggingFace Hub │ TensorBoard            │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Model Architecture (`src/model.py`)

**Transformer Implementation**:
- GPT-2 style decoder-only architecture
- Multi-head self-attention with optional flash attention
- Feed-forward networks with GELU activation
- Layer normalization and residual connections
- Positional embeddings (learned)

**Key Features**:
- Configurable model sizes (2.36M to 117M+ parameters)
- Gradient checkpointing for memory efficiency
- Mixed precision (FP16) support
- Multi-GPU support via DataParallel

**Model Sizes**:
```
Quick Start:  d_model=128, layers=2,  heads=4   → 2.36M params
Production:   d_model=256, layers=4,  heads=8   → 16M params
Full:         d_model=768, layers=12, heads=12  → 117M params
```

### 2. Training System (`src/trainer.py`)

**Training Loop**:
- Epoch-based training with step counting
- Gradient accumulation for large effective batch sizes
- Mixed precision training with automatic scaling
- Learning rate scheduling (cosine, linear, constant)
- Gradient clipping for stability

**Multi-GPU Support**:
- DataParallel for multi-GPU training
- Automatic device selection and management
- Memory-optimized gradient gathering
- Balanced workload distribution

**TPU Support**:
- torch_xla integration for Google Cloud TPU
- XLA compiler optimization
- bfloat16 mixed precision
- Large batch size optimization (128-512)

**Checkpointing**:
- Periodic checkpoint saving
- Best model tracking
- Resume from checkpoint (local or remote)
- Automatic upload to HuggingFace Hub

**Monitoring**:
- TensorBoard logging
- Weights & Biases integration
- Real-time metrics (loss, learning rate, throughput)
- GPU memory tracking

### 3. Data Pipeline (`src/data.py`)

**Data Loading**:
- HuggingFace datasets integration
- Binary format for pre-tokenized data
- Memory-mapped files for efficiency
- Automatic dataset download from HuggingFace Hub

**Data Processing**:
- Dynamic batching with padding
- Sequence packing for efficiency
- Multi-worker data loading
- Pin memory for GPU transfer

**Supported Formats**:
- HuggingFace datasets (WikiText, OpenWebText, etc.)
- Binary tokenized files (`.bin`)
- Custom datasets via configuration

### 4. Remote Model Management (`src/remote_model_loader.py`)

**HuggingFace Hub Integration**:
- Download checkpoints from HuggingFace Hub
- List available models in repository
- Automatic caching and version management
- Resume training from remote checkpoints

**Features**:
- Lazy loading (download on demand)
- Progress tracking for large downloads
- Automatic retry on network failures
- Local cache management

### 5. Inference Engine (`src/inference.py`)

**Text Generation**:
- Autoregressive generation
- Multiple sampling strategies (greedy, temperature, top-k, top-p)
- Batch inference support
- Streaming generation

**Model Loading**:
- Local checkpoint loading
- Remote model loading from HuggingFace Hub
- Automatic device selection
- Model compilation for speed (PyTorch 2.0+)

## Data Flow

### Training Flow

```
Configuration → Data Loader → Model → Forward Pass → Loss
     ↓              ↓           ↓          ↓           ↓
  Optimizer ← Backward Pass ← Gradient ← Scaling ← Mixed Precision
     ↓
  Update Weights → Save Checkpoint → Upload to HF Hub
     ↓
  Validation → TensorBoard → Continue/Stop
```

### Inference Flow

```
Checkpoint (Local/Remote) → Load Model → Tokenize Input
                                ↓
                          Generate Tokens → Decode → Output Text
```

### Dataset Flow

```
Config → Check Local → Missing? → Download from HF Hub → Cache
           ↓                              ↓
        Load Binary ← ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┘
           ↓
        Memory Map → DataLoader → Batching → Training
```

## Memory Management

### Memory Optimization Strategies

1. **Gradient Checkpointing**:
   - Trades computation for memory
   - Saves 2-3GB per GPU
   - Recomputes activations during backward pass

2. **Mixed Precision (FP16)**:
   - Reduces memory by ~50%
   - Faster computation on modern GPUs
   - Automatic loss scaling for stability

3. **Gradient Accumulation**:
   - Simulates larger batch sizes
   - Reduces memory per step
   - Maintains training stability

4. **Memory Fragmentation Fix**:
   - `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
   - Reduces fragmentation
   - Frees up 500MB-1GB

### Multi-GPU Memory Distribution

**DataParallel Pattern**:
```
GPU 0 (Master):
  - Model replica
  - Optimizer states
  - Gradients
  - Gathered gradients from all GPUs  ← Extra memory!
  - Activation memory
  Total: ~12-14GB for 117M model

GPU 1+ (Workers):
  - Model replica
  - Gradients
  - Activation memory
  Total: ~7-8GB for 117M model
```

**Memory Breakdown (117M model, batch_size=16, context=512)**:
```
Component                Memory (FP16)    Memory (FP32)
─────────────────────────────────────────────────────────
Model weights            0.9 GB           1.8 GB
Optimizer states         -                1.8 GB
Gradients                0.9 GB           -
Activations (no ckpt)    4.0 GB           -
Activations (with ckpt)  2.0 GB           -
DataParallel overhead    1.5 GB           -
Buffer/fragmentation     1.0 GB           -
─────────────────────────────────────────────────────────
Total (no checkpointing) 8.3 GB           -
Total (with checkpointing) 6.3 GB         -
```

## Resilience Features

### 1. Multi-Environment Support

**Device Detection**:
```python
# Automatic device selection
if torch_xla_available:
    device = xm.xla_device()  # TPU
elif torch.cuda.is_available():
    device = "cuda"  # NVIDIA GPU
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = "mps"  # Apple Silicon
else:
    device = "cpu"
```

**Graceful Degradation**:
- Falls back to CPU if GPU/TPU unavailable
- Disables mixed precision if not supported
- Adjusts batch size based on available memory

### 2. PyTorch Version Compatibility

**Version Detection**:
```python
# GradScaler import (2.0-2.3 vs 2.4+)
try:
    from torch.amp import GradScaler  # 2.4+
except ImportError:
    from torch.cuda.amp import GradScaler  # 2.0-2.3

# autocast API (2.0-2.3 vs 2.4+)
try:
    autocast_context = autocast(device_type=device_type)  # 2.4+
except TypeError:
    autocast_context = autocast()  # 2.0-2.3
```

**Supported Versions**:
- PyTorch 2.0, 2.1, 2.2, 2.3, 2.4+
- Automatic API detection
- Backward and forward compatible

### 3. Error Handling

**Network Operations**:
- Automatic retry with exponential backoff
- Timeout handling
- Graceful failure messages

**Memory Management**:
- OOM detection and recovery
- Automatic batch size reduction
- Memory cleanup on errors

**Checkpointing**:
- Atomic checkpoint saves
- Corruption detection
- Automatic backup

### 4. Monitoring and Debugging

**Logging**:
- Structured logging with levels
- TensorBoard integration
- Weights & Biases support
- File and console output

**Diagnostics**:
- Environment checker (`scripts/fix_environment.py`)
- GPU memory monitoring
- Training speed profiling
- Loss curve analysis

## Configuration System

### Configuration Hierarchy

```
Default Config → User Config → CLI Arguments → Runtime Overrides
```

### Configuration Structure

```yaml
model:                    # Model architecture
  vocab_size: 100277
  d_model: 768
  num_heads: 12
  num_layers: 12
  context_length: 512
  dropout: 0.12
  use_gradient_checkpointing: true

training:                 # Training parameters
  batch_size: 16
  gradient_accumulation_steps: 8
  max_epochs: 8
  learning_rate: 2.0e-4
  weight_decay: 0.15

data:                     # Data configuration
  dataset_type: "binary"
  train_file: "data/balanced_300m/train.bin"
  huggingface_dataset:    # Auto-download config
    repo_id: "0x-genesys/mix_wiki_code_chat_data_300M_tokens"
    auto_download: true

system:                   # System configuration
  device: "cuda"          # Device: cuda, mps, tpu, cpu, or auto
  mixed_precision: true
  compile_model: false

checkpoint:               # Checkpointing
  save_dir: "checkpoints/production"
  resume_from: null

logging:                  # Logging and monitoring
  use_wandb: false
  log_dir: "logs/production"

huggingface_hub:          # HuggingFace Hub integration
  enabled: true
  repo_id: "0x-genesys/neo_weights_checkpoints"
```

## Performance Optimization

### Training Speed Optimization

**Strategies** (in order of impact):
1. **Increase batch size**: 3-4x speedup (if memory allows)
2. **Reduce gradient accumulation**: 2-3x speedup
3. **Reduce context length**: 3-4x speedup (512→256)
4. **Disable gradient checkpointing**: 1.5-2x speedup (uses more memory)
5. **Enable torch.compile**: 1.3-1.5x speedup (PyTorch 2.0+)
6. **Increase num_workers**: 1.1-1.2x speedup (if data loading bottleneck)

**Example Optimization**:
```yaml
# Original (slow but safe)
batch_size: 16
gradient_accumulation_steps: 8
context_length: 512
use_gradient_checkpointing: true

# Optimized (3-4x faster)
batch_size: 48
gradient_accumulation_steps: 3
context_length: 512
use_gradient_checkpointing: false
```

### Memory Optimization

**Strategies** (in order of memory savings):
1. **Enable gradient checkpointing**: Saves 2-3GB
2. **Reduce batch size**: Saves ~2GB per halving
3. **Reduce context length**: Saves ~1.5GB (512→384)
4. **Use FP16 mixed precision**: Saves ~50% memory
5. **Set memory allocator config**: Saves 500MB-1GB

**Example Optimization**:
```yaml
# Original (uses 13-14GB)
batch_size: 16
context_length: 512
use_gradient_checkpointing: false

# Optimized (uses 8-9GB)
batch_size: 8
context_length: 384
use_gradient_checkpointing: true
```

## Deployment

### Model Export

**Checkpoint Format**:
```python
{
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'epoch': epoch,
    'global_step': global_step,
    'best_val_loss': best_val_loss,
    'config': config
}
```

**HuggingFace Hub Upload**:
- Automatic during training
- Manual via `scripts/test_checkpoint_upload.py`
- Versioning with step numbers
- Model cards and documentation

### Inference Deployment

**Options**:
1. **Local inference**: Load checkpoint directly
2. **Remote inference**: Load from HuggingFace Hub
3. **API deployment**: Wrap in FastAPI/Flask
4. **Batch processing**: Process multiple inputs

**Example Deployment**:
```python
from src.inference import TextGenerator

# Load model
generator = TextGenerator(
    model_path="checkpoints/production/best_model.pt",
    device="cuda"
)

# Generate text
output = generator.generate(
    prompt="Once upon a time",
    max_length=100,
    temperature=0.8
)
```

## Testing Strategy

### Test Levels

1. **Unit Tests**: Individual components
   - Model forward/backward pass
   - Data loading and batching
   - Tokenization
   - Device selection

2. **Integration Tests**: Component interactions
   - Training loop
   - Checkpoint save/load
   - Multi-GPU training
   - Remote model loading

3. **System Tests**: End-to-end workflows
   - Full training run
   - Inference pipeline
   - Dataset download
   - Model upload

4. **Environment Tests**: Platform compatibility
   - CUDA, MPS, CPU
   - PyTorch versions
   - Memory constraints
   - Network conditions

### Test Execution

```bash
# Run all tests
python -m pytest test/

# Run specific test
python test/test_setup.py

# Environment diagnostics
python scripts/fix_environment.py
```

## TPU Training Architecture

### Overview

Neo supports **Google Cloud TPU v3-8** training with full PyTorch XLA integration. TPU training provides **3x faster** performance compared to T4 GPU on Kaggle.

### TPU vs Standard Trainer

Neo uses **separate trainers** for TPU and GPU/CPU:

| Trainer | Hardware | File | Use Case |
|---------|----------|------|----------|
| **Standard Trainer** | CUDA, MPS, CPU | `src/trainer.py` | GPU/CPU training |
| **TPU Trainer** | TPU (XLA) | `src/tpu_trainer.py` | TPU training |

**Selection** (automatic in `train.py`):
```python
if use_tpu:
    from src.tpu_trainer import TPUTrainer
    trainer = TPUTrainer(...)
else:
    from src.trainer import Trainer
    trainer = Trainer(...)
```

### TPU Architecture

#### Hardware Configuration

**Kaggle TPU v3-8**:
- **8 TPU cores** (v3 generation)
- **128GB HBM memory** (16GB per core)
- **420 TFLOPS** (bfloat16)
- **8 cores run in parallel** (data parallelism)

**Memory Layout**:
```
┌─────────────────────────────────────────────────────────┐
│                    CPU Memory (Host)                     │
│  Model: 450MB │ Checkpoint: 900MB │ Data: Variable     │
└─────────────────────────────────────────────────────────┘
                         │
                         ↓ (transfer)
┌─────────────────────────────────────────────────────────┐
│                    TPU Memory (Device)                   │
├──────────┬──────────┬──────────┬──────────┬─────────────┤
│ Core 0   │ Core 1   │ Core 2   │ Core 3   │ ...         │
│ 16GB     │ 16GB     │ 16GB     │ 16GB     │ ...         │
│ Model:   │ Model:   │ Model:   │ Model:   │ ...         │
│ 450MB    │ 450MB    │ 450MB    │ 450MB    │ ...         │
│ Batch:   │ Batch:   │ Batch:   │ Batch:   │ ...         │
│ 128      │ 128      │ 128      │ 128      │ ...         │
└──────────┴──────────┴──────────┴──────────┴─────────────┘
```

#### Multi-Core Training Flow

**1. Process Spawning**:
```python
# Main process (CPU)
xmp.spawn(_mp_fn, nprocs=8, start_method='fork')

# Creates 8 child processes:
# Process 0 → TPU Core 0 (rank=0)
# Process 1 → TPU Core 1 (rank=1)
# ...
# Process 7 → TPU Core 7 (rank=7)
```

**2. Model Distribution**:
```python
# Each process (parallel):
device = xm.xla_device()  # Get TPU device for this core
model = xmp.MpModelWrapper(self.model)  # Wrap for multi-core
model = model.to(device)  # Transfer CPU → TPU
```

**3. Data Distribution**:
```python
# Distributed sampler splits data across cores
train_sampler = torch.utils.data.distributed.DistributedSampler(
    dataset,
    num_replicas=8,  # 8 TPU cores
    rank=xm.get_ordinal(),  # 0-7
    shuffle=True
)

# Each core gets different data batches
# Core 0: batches 0, 8, 16, 24, ...
# Core 1: batches 1, 9, 17, 25, ...
# ...
# Core 7: batches 7, 15, 23, 31, ...
```

**4. Parallel Training**:
```python
# Each core (parallel):
for batch in para_loader.per_device_loader(device):
    # Forward pass (independent)
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    
    # Backward pass (independent)
    loss.backward()
    
    # Gradient synchronization (collective)
    xm.reduce_gradients(optimizer)  # All-reduce across cores
    
    # Optimizer step (independent)
    optimizer.step()
    xm.mark_step()  # XLA execution barrier
```

**5. Gradient Synchronization**:
```
Core 0: grad_0 ─┐
Core 1: grad_1 ─┤
Core 2: grad_2 ─┤
Core 3: grad_3 ─┼─→ All-Reduce → avg_grad
Core 4: grad_4 ─┤
Core 5: grad_5 ─┤
Core 6: grad_6 ─┤
Core 7: grad_7 ─┘

Result: All cores get avg_grad = (grad_0 + ... + grad_7) / 8
```

### TPU Loading Flow

#### Checkpoint Resumption (GPU → TPU)

**Timeline** (~40 seconds total):

```
[0-5s]   📥 Download checkpoint from HuggingFace Hub
         ↓
[5-10s]  📥 Load checkpoint on CPU
         checkpoint = torch.load(path, map_location='cpu')
         model.load_state_dict(checkpoint['model_state_dict'])
         ↓
[10-15s] 🚀 Spawn 8 TPU processes
         xmp.spawn(_mp_fn, nprocs=8)
         ↓
[15-25s] 🔄 Transfer model CPU → TPU (parallel across 8 cores)
         model = model.to(xm.xla_device())
         ↓
[25-28s] ✅ Load optimizer/scheduler state
         optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
         ↓
[28-30s] 📊 Setup data loading
         DistributedSampler + ParallelLoader
         ↓
[30-40s] ⚡ XLA compilation (first batch only)
         First forward/backward triggers compilation
         ↓
[40s+]   🚀 Steady state training (2.5 steps/sec)
```

**Memory Flow**:
```
HuggingFace Hub (900MB .pt file)
         ↓ download
CPU Disk Cache (900MB)
         ↓ torch.load(map_location='cpu')
CPU RAM (450MB model weights)
         ↓ model.to(device) × 8 cores (parallel)
TPU Core 0 (450MB)
TPU Core 1 (450MB)
TPU Core 2 (450MB)
TPU Core 3 (450MB)
TPU Core 4 (450MB)
TPU Core 5 (450MB)
TPU Core 6 (450MB)
TPU Core 7 (450MB)
```

**Total TPU Memory**: 450MB × 8 = 3.6GB (2.8% of 128GB)

#### XLA Compilation

**First Batch** (~10-15 seconds):
```python
# First forward pass
outputs = model(inputs)  # XLA traces and compiles graph

# Compilation steps:
# 1. Trace PyTorch operations
# 2. Build XLA HLO (High-Level Optimizer) graph
# 3. Optimize graph (fusion, layout, etc.)
# 4. Compile to TPU machine code
# 5. Execute compiled code
```

**Subsequent Batches** (~0.4 seconds):
```python
# Reuse compiled graph (no recompilation)
outputs = model(inputs)  # Execute cached compiled code
```

**Compilation Cache**:
- Cached per input shape
- Recompilation only if shape changes
- One-time cost amortized over thousands of steps

### PyTorch XLA Patterns

Neo implements all **12 Kaggle TPU patterns**:

| # | Pattern | Implementation | Purpose |
|---|---------|----------------|---------|
| 1 | **Startup Script** | `env-setup.py` | Install torch_xla |
| 2 | **Distributed Function** | `xmp.spawn(_mp_fn, nprocs=8)` | Multi-core spawn |
| 3 | **Model Wrapper** | `MpModelWrapper(model)` | Multi-core model |
| 4 | **Device Setup** | `xm.xla_device()` | Get TPU device |
| 5 | **Data to Device** | `data.to(device)` | Transfer data |
| 6 | **Master Print** | `xm.master_print()` | Print from rank 0 |
| 7 | **Data Loading** | `DistributedSampler` | Split data across cores |
| 8 | **Parallel Loader** | `ParallelLoader` | Efficient TPU loading |
| 9 | **Results Reduction** | `xm.mesh_reduce()` | Aggregate across cores |
| 10 | **Checkpoint Save** | `xser.save()` | Memory-optimized save |
| 11 | **Checkpoint Load** | `torch.load()` | Standard PyTorch load |
| 12 | **Gradient Sync** | `xm.reduce_gradients()` | All-reduce gradients |

### TPU Optimizations

#### Batch Size Optimization

**GPU (T4)**:
- Batch size: 8 (limited by 16GB VRAM)
- Gradient accumulation: 16
- Effective batch: 128

**TPU (v3-8)**:
- Batch size: 128 (16x larger, 128GB HBM)
- Gradient accumulation: 4
- Effective batch: 512 (4x larger)

**Why larger batches on TPU?**
- More memory (128GB vs 16GB)
- Better TPU utilization
- Amortizes communication overhead

#### Learning Rate Scaling

**Square root scaling**:
```python
new_lr = old_lr * sqrt(new_batch / old_batch)
new_lr = 0.0002 * sqrt(128 / 8)
new_lr = 0.0002 * 4
new_lr = 0.0004
```

**Why square root?**
- Balances convergence speed and stability
- Empirically works well for transformers
- Maintains effective learning dynamics

#### Mixed Precision

**TPU uses bfloat16 natively**:
- No explicit AMP (Automatic Mixed Precision) needed
- TPU hardware optimized for bfloat16
- Better numerical stability than float16
- Same memory savings as float16

### Performance Comparison

#### Training Speed

| Hardware | Batch Size | Steps/sec | Time to 10K steps |
|----------|------------|-----------|-------------------|
| **CPU** | 4 | 0.1 | ~27 hours |
| **MPS (M1)** | 8 | 0.3 | ~9 hours |
| **T4 GPU** | 8 | 0.8 | ~3.5 hours |
| **P100 GPU** | 8 | 0.6 | ~4.6 hours |
| **TPU v3-8** | 128 | **2.5** | **~1.1 hours** |

**TPU is 3.1x faster than T4 GPU!**

#### Memory Usage

| Hardware | Model | Optimizer | Activations | Total | Available |
|----------|-------|-----------|-------------|-------|-----------|
| **T4 GPU** | 450MB | 900MB | 2GB | 3.4GB | 16GB (21%) |
| **TPU v3-8** | 3.6GB | 7.2GB | 16GB | 26.8GB | 128GB (21%) |

**TPU uses same percentage but has 8x more memory!**

#### Cost Efficiency (Kaggle Free Tier)

| Hardware | Weekly Quota | Steps/hour | Steps/week |
|----------|--------------|------------|------------|
| **T4 GPU** | 30 hours | 2,880 | 86,400 |
| **TPU v3-8** | 30 hours | 9,000 | **270,000** |

**TPU allows 3.1x more training in same quota!**

### TPU Limitations

#### What Works
- ✅ Standard PyTorch operations
- ✅ Transformer models
- ✅ Data parallelism
- ✅ Gradient accumulation
- ✅ Mixed precision (bfloat16)
- ✅ Checkpoint save/load
- ✅ TensorBoard logging
- ✅ Cross-hardware checkpoints

#### What Doesn't Work
- ❌ Dynamic shapes (requires recompilation)
- ❌ Python control flow in model (use torch ops)
- ❌ In-place operations (use functional)
- ❌ CPU tensors in model (must be on TPU)
- ❌ Interactive debugging (use logging)

#### Workarounds

**Dynamic shapes**:
```python
# Bad: Dynamic sequence length
for i in range(seq_len):  # seq_len varies
    output = model(input[:, :i])

# Good: Fixed sequence length
output = model(input)  # Always same shape
```

**Python control flow**:
```python
# Bad: Python if in model
if x.sum() > 0:
    return x * 2
else:
    return x

# Good: Torch operations
return torch.where(x.sum() > 0, x * 2, x)
```

### TPU Best Practices

#### 1. Use Large Batch Sizes
```yaml
training:
  batch_size: 128  # Or 256, 512
  gradient_accumulation_steps: 4
```

#### 2. Minimize Host-Device Sync
```python
# Bad: Frequent .item() calls
for batch in loader:
    loss = model(batch)
    print(f"Loss: {loss.item()}")  # Sync every batch!

# Good: Batch logging
losses = []
for batch in loader:
    loss = model(batch)
    losses.append(loss)
    if len(losses) == 100:
        avg_loss = torch.stack(losses).mean().item()  # Sync once
        print(f"Avg loss: {avg_loss}")
        losses = []
```

#### 3. Use xm.mark_step()
```python
# After optimizer step
optimizer.step()
xm.mark_step()  # Execute accumulated XLA ops
```

#### 4. Master-Only Operations
```python
# Checkpointing (master only)
if xm.is_master_ordinal():
    save_checkpoint(model, path)

# Logging (master only)
if xm.is_master_ordinal():
    writer.add_scalar('loss', loss, step)
```

#### 5. Use ParallelLoader
```python
# Efficient data loading
para_loader = pl.ParallelLoader(train_loader, [device])
for batch in para_loader.per_device_loader(device):
    # Training code
```

### Debugging TPU Issues

#### Check TPU Availability
```python
import torch_xla.core.xla_model as xm
print(f"TPU device: {xm.xla_device()}")
print(f"TPU cores: {xm.xrt_world_size()}")
```

#### Monitor TPU Utilization
```python
# In training loop
if step % 100 == 0:
    xm.master_print(f"Step {step}")
    # Check if TPU is being used (should see activity)
```

#### Common Issues

**Issue**: "XLA out of memory"
**Solution**: Reduce batch size or enable gradient checkpointing

**Issue**: "Slow first batch"
**Solution**: Normal - XLA compilation (one-time cost)

**Issue**: "Training not using all cores"
**Solution**: Check `xmp.spawn(nprocs=8)` is called

**Issue**: "Loss not syncing across cores"
**Solution**: Use `xm.mesh_reduce()` for validation loss

### TPU Training Checklist

Before training:
- [ ] TPU enabled in Kaggle settings
- [ ] torch_xla installed (`bash scripts/setup_kaggle_tpu.sh`)
- [ ] Checkpoint uploaded to HuggingFace Hub
- [ ] Config uses large batch size (128+)

During training:
- [ ] All 8 cores spawned (check logs)
- [ ] First batch slow (XLA compilation - normal)
- [ ] Subsequent batches fast (~0.4s)
- [ ] Loss decreasing smoothly
- [ ] Checkpoints saving every 500 steps

After training:
- [ ] Checkpoints uploaded to HuggingFace Hub
- [ ] TensorBoard logs available
- [ ] Can resume on GPU if needed

## Security Considerations

### Data Security
- No sensitive data in checkpoints
- Secure HuggingFace Hub authentication
- Local cache encryption (optional)

### Model Security
- Checkpoint integrity verification
- Signed model uploads
- Access control via HuggingFace Hub

### Network Security
- HTTPS for all downloads
- Token-based authentication
- Rate limiting and retry logic

## Future Enhancements

### Planned Features
1. **DistributedDataParallel**: More efficient multi-GPU training
2. **Flash Attention 2**: Faster attention computation
3. **8-bit Optimizers**: Reduced memory usage
4. **Model Quantization**: INT8 inference
5. **ONNX Export**: Cross-platform deployment
6. **Streaming Inference**: Real-time generation
7. **Fine-tuning API**: Easy model adaptation
8. **Evaluation Suite**: Comprehensive benchmarks

### Research Directions
1. **Sparse Attention**: Longer context windows
2. **Mixture of Experts**: Larger models, same compute
3. **Retrieval Augmentation**: Enhanced knowledge
4. **Multi-modal**: Text + images
5. **Reinforcement Learning**: RLHF integration

## References

- **PyTorch Documentation**: https://pytorch.org/docs/
- **HuggingFace Hub**: https://huggingface.co/docs/hub
- **Attention Is All You Need**: https://arxiv.org/abs/1706.03762
- **GPT-2 Paper**: https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf

---

**Last Updated**: 2026-04-29
**Version**: 1.0.0