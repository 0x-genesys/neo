---
name: Neo Transformer Training
description: Expert guidance for training, debugging, and optimizing the Neo transformer language model
version: 1.0.0
keywords: [transformer, training, pytorch, deep-learning, nlp, gpu-optimization, multi-gpu, huggingface]
---

# Neo Transformer Training Expert

You are an expert in the Neo transformer language model training system. You have deep knowledge of:

## Core Capabilities

### Training System
- **Multi-environment support**: CUDA, MPS (Apple Silicon), CPU
- **PyTorch compatibility**: Versions 2.0, 2.1, 2.2, 2.3, 2.4+
- **Multi-GPU training**: DataParallel with memory optimization
- **Mixed precision**: FP16 training with automatic fallback
- **Gradient checkpointing**: Memory-efficient training
- **Automatic dataset download**: From HuggingFace Hub
- **Remote checkpoint loading**: Resume from HuggingFace Hub

### Model Architecture
- **Base**: GPT-2 style transformer
- **Sizes**: 2.36M (quick), 16M (production), 117M (full)
- **Context lengths**: 128, 256, 384, 512 tokens
- **Tokenizer**: tiktoken cl100k_base (GPT-4) or GPT-2

### Datasets
- **Pre-processed**: 300M token balanced dataset
- **Repository**: `0x-genesys/mix_wiki_code_chat_data_300M_tokens`
- **Composition**: WikiText-103 (34%), UltraChat (66%), The Stack (16%)
- **Format**: Binary `.bin` files with memory mapping

### Model Repository
- **Checkpoints**: `0x-genesys/neo_weights_checkpoints`
- **Available models**: checkpoint.pt, best_model.pt, checkpoint_step_*.pt
- **Auto-upload**: During training with HuggingFace Hub integration

## Common Tasks

### Starting Training
```bash
# Quick test (5 minutes)
python train.py --config config/quick_start.yaml

# Production training
python train.py --config config/production_training.yaml

# GPU training with balanced dataset
python train.py --config config/gpu_training_117m_balanced.yaml --multi-gpu

# Resume from HuggingFace Hub
python train.py --config config/production_training.yaml --resume-remote checkpoint.pt
```

### Running Inference
```bash
# Local model
python src/inference.py --model checkpoints/production/best_model.pt --prompt "Once upon a time"

# Remote model from HuggingFace Hub
python src/inference.py --model-remote best_model.pt --prompt "The future of AI"
```

### Troubleshooting

#### Out of Memory (OOM)
1. **Set memory optimization**:
   ```bash
   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
   ```

2. **Use low memory config**:
   ```bash
   python train.py --config config/gpu_training_117m_balanced_low_memory.yaml
   ```

3. **Or use single GPU**:
   ```bash
   export CUDA_VISIBLE_DEVICES=0
   python train.py --config config/gpu_training_117m_balanced.yaml
   ```

#### Slow Training
- Check GPU utilization: `nvidia-smi -l 1`
- Increase batch size if GPU util < 80%
- Reduce gradient accumulation steps
- Disable gradient checkpointing if memory allows
- See `docs/GPU_TRAINING_OPTIMIZATION_GUIDE.md`

#### PyTorch Compatibility Issues
- **GradScaler import error**: Fixed with version-compatible imports
- **autocast device_type error**: Fixed with try-except fallback
- **Lambda pickle error**: Fixed with CollateFnWrapper class
- Run diagnostics: `python scripts/fix_environment.py`

## Key Files

### Configuration
- `config/quick_start.yaml` - Fast testing (2.36M params)
- `config/production_training.yaml` - CPU training (16M params)
- `config/gpu_training_117m_balanced.yaml` - Full training (117M params)
- `config/gpu_training_117m_balanced_low_memory.yaml` - Multi-GPU safe

### Source Code
- `src/model.py` - Model architecture
- `src/trainer.py` - Training loop with multi-GPU support
- `src/data.py` - Data loading with auto-download
- `src/remote_model_loader.py` - HuggingFace Hub integration
- `src/inference.py` - Text generation

### Scripts
- `scripts/fix_environment.py` - Environment diagnostics
- `scripts/test_checkpoint_upload.py` - Upload models to HF Hub
- `scripts/setup_dataset_repo.py` - Dataset repository setup
- `fix_oom.sh` - Emergency OOM fix

### Documentation
- `README.md` - Main documentation
- `ARCHITECTURE.md` - System architecture
- `docs/START_HERE.md` - Getting started guide
- `docs/GPU_TRAINING_OPTIMIZATION_GUIDE.md` - Performance tuning
- `docs/MULTI_GPU_MEMORY_FIX.md` - Memory optimization
- `docs/DATASET_DOWNLOAD_GUIDE.md` - Dataset management
- `docs/CHECKPOINT_UPLOAD_GUIDE.md` - Model sharing

## Resilience Features

### Multi-Environment Support
- **CUDA**: Full GPU acceleration with multi-GPU
- **MPS**: Apple Silicon GPU support
- **CPU**: Fallback for systems without GPU
- **Auto-detection**: Automatic device selection

### PyTorch Version Compatibility
- **2.0-2.3**: Uses `torch.cuda.amp` for mixed precision
- **2.4+**: Uses `torch.amp` with device_type parameter
- **Automatic fallback**: Detects version and uses correct API

### Memory Management
- **Gradient checkpointing**: Saves 2-3GB per GPU
- **Mixed precision**: FP16 reduces memory by 50%
- **Dynamic batch sizing**: Adjust based on available memory
- **Memory fragmentation fix**: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

### Error Recovery
- **Graceful degradation**: Disables features if unavailable
- **Automatic retries**: For network operations
- **Checkpoint saving**: Every N steps for recovery
- **Validation**: Catches issues early

## Best Practices

### Training
1. **Start small**: Use `quick_start.yaml` to verify setup
2. **Monitor GPU**: Use `nvidia-smi -l 1` during training
3. **Set memory optimization**: Always set `PYTORCH_CUDA_ALLOC_CONF`
4. **Use TensorBoard**: Monitor loss curves
5. **Save frequently**: Set reasonable `save_interval`

### Multi-GPU
1. **Check memory**: GPU 0 uses more memory with DataParallel
2. **Use low memory config**: For T4 GPUs (14-16GB)
3. **Monitor both GPUs**: Ensure balanced utilization
4. **Set batch size carefully**: Account for DataParallel overhead

### Optimization
1. **Increase batch size**: If GPU util < 80%
2. **Reduce gradient accumulation**: For faster training
3. **Disable gradient checkpointing**: If memory allows
4. **Use torch.compile**: For PyTorch 2.0+ (1.3-1.5x speedup)

### Debugging
1. **Run diagnostics**: `python scripts/fix_environment.py`
2. **Test with --max-steps 10**: Quick validation
3. **Check logs**: TensorBoard or log files
4. **Verify checkpoints**: Load and inspect saved models

## Quick Reference

### Memory Usage (117M model)
- **Single GPU**: ~8-9GB
- **Multi-GPU (GPU 0)**: ~12-14GB (with DataParallel overhead)
- **Multi-GPU (GPU 1)**: ~7-8GB
- **With gradient checkpointing**: -2-3GB per GPU
- **With FP16**: -50% memory

### Training Speed (117M model, 2x T4)
- **Default config**: ~2-3 sec/step
- **Optimized config**: ~0.8-1.0 sec/step (3-4x faster)
- **Ultra-fast config**: ~0.2-0.3 sec/step (6-8x faster, shorter context)

### Effective Batch Sizes
- `batch_size: 16, grad_accum: 8, 2 GPUs` = 256 effective batch
- `batch_size: 8, grad_accum: 16, 2 GPUs` = 256 effective batch (lower memory)
- `batch_size: 4, grad_accum: 32, 2 GPUs` = 256 effective batch (lowest memory)

## When to Use What

### Config Selection
- **Testing/Debugging**: `quick_start.yaml`
- **CPU Training**: `production_training.yaml`
- **Single GPU**: `gpu_training_117m_balanced.yaml`
- **Multi-GPU (plenty memory)**: `gpu_training_117m_balanced.yaml`
- **Multi-GPU (limited memory)**: `gpu_training_117m_balanced_low_memory.yaml`
- **OOM Issues**: `gpu_training_117m_balanced_low_memory.yaml` or single GPU

### Optimization Strategy
- **Need speed**: Increase batch_size, reduce grad_accum
- **Need memory**: Enable gradient checkpointing, reduce batch_size
- **Need both**: Use low memory config with optimizations
- **Maximum speed**: Reduce context_length, disable checkpointing

## Support Resources

- **Environment issues**: Run `python scripts/fix_environment.py`
- **OOM errors**: See `docs/MULTI_GPU_MEMORY_FIX.md`
- **Slow training**: See `docs/GPU_TRAINING_OPTIMIZATION_GUIDE.md`
- **Dataset setup**: See `docs/DATASET_DOWNLOAD_GUIDE.md`
- **Model sharing**: See `docs/CHECKPOINT_UPLOAD_GUIDE.md`

You should provide expert guidance on training, optimization, debugging, and deployment of the Neo transformer model, always considering the specific environment and constraints of the user.

Always keep skill file updated with any abstract commands needed

Always keep a compressed memory.md file - compress memory aggressively when necessary like fixing a bug ? -> 1 liner with bug fix description and file/impact/tests

keep all new supporting md docs in docs/ folder 

keep all testing to test/ folder