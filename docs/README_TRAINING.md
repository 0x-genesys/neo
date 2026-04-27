# Production-Ready Transformer Training Pipeline

A complete, production-ready implementation for training decoder-only transformer language models (GPT-style) from scratch.

## 🚀 Features

- **Complete Training Pipeline**: End-to-end training with checkpointing, validation, and logging
- **Production Architecture**: Proper model architecture with dropout, layer norm, weight tying
- **Data Loading**: Integrated with HuggingFace datasets - access thousands of datasets
- **Mixed Precision Training**: Automatic mixed precision (AMP) for faster training
- **Gradient Accumulation**: Train with large effective batch sizes on limited hardware
- **Learning Rate Scheduling**: Cosine annealing with warmup
- **Checkpointing**: Save and resume training from checkpoints
- **Logging**: TensorBoard and Weights & Biases integration
- **Text Generation**: Production inference with temperature, top-k, top-p sampling
- **Evaluation**: Perplexity calculation and sample generation

## 📁 Project Structure

```
.
├── config/
│   └── model_config.yaml          # Model and training configuration
├── src/
│   ├── model.py                   # Transformer model architecture
│   ├── data.py                    # Data loading and preprocessing
│   ├── trainer.py                 # Training loop and utilities
│   └── inference.py               # Production inference
├── scripts/
│   └── download_data.py           # Dataset download utility
├── train.py                       # Main training script
├── evaluate.py                    # Model evaluation script
└── requirements.txt               # Python dependencies
```

## 🛠️ Installation

1. **Install dependencies:**
```bash
source venv/bin/activate
pip install -r requirements.txt
```

2. **Login to HuggingFace (optional, for gated datasets):**
```bash
huggingface-cli login
```

## 📊 Available Datasets

### Small (for testing/development)
- **tiny_shakespeare**: ~1MB, Shakespeare's works
- **wikitext-2-raw-v1**: ~2M tokens

### Medium (for experimentation)
- **wikitext-103-raw-v1**: ~100M tokens
- **bookcorpus**: ~11K books

### Large (for serious training)
- **openwebtext**: ~8M documents, ~40GB
- **c4**: Colossal Clean Crawled Corpus
- **EleutherAI/pile**: 825GB diverse text

### Download a dataset:
```bash
python scripts/download_data.py --dataset wikitext --config wikitext-2-raw-v1
```

## 🎯 Quick Start

### 1. Train a Small Model (for testing)

```bash
# Train on tiny_shakespeare (fast, for testing)
python train.py --dataset tiny_shakespeare --batch-size 16 --epochs 5
```

### 2. Train on WikiText-2 (small dataset)

```bash
# Edit config/model_config.yaml to set:
# data:
#   dataset_name: "wikitext"
#   dataset_config: "wikitext-2-raw-v1"

python train.py
```

### 3. Train on WikiText-103 (medium dataset)

```bash
python train.py --dataset wikitext --batch-size 32 --epochs 10
```

### 4. Train on OpenWebText (large dataset)

```bash
# Requires more GPU memory and time
python train.py --dataset openwebtext --batch-size 16 --epochs 3
```

## ⚙️ Configuration

Edit `config/model_config.yaml` to customize:

### Model Architecture
```yaml
model:
  d_model: 512          # Model dimension
  num_heads: 8          # Number of attention heads
  num_layers: 6         # Number of transformer blocks
  context_length: 512   # Maximum sequence length
  dropout: 0.1          # Dropout rate
```

### Training Parameters
```yaml
training:
  batch_size: 32
  gradient_accumulation_steps: 4  # Effective batch = 32 * 4 = 128
  learning_rate: 3.0e-4
  max_epochs: 10
  max_steps: 100000
```

### Small Model (Fast Training)
- d_model: 256, num_heads: 4, num_layers: 4
- ~10M parameters

### Medium Model (Balanced)
- d_model: 512, num_heads: 8, num_layers: 6
- ~40M parameters

### Large Model (High Quality)
- d_model: 768, num_heads: 12, num_layers: 12
- ~120M parameters

## 🏃 Training

### Basic Training
```bash
python train.py
```

### With Custom Config
```bash
python train.py --config config/my_config.yaml
```

### Resume from Checkpoint
```bash
python train.py --resume checkpoints/checkpoint_step_5000.pt
```

### Override Config Parameters
```bash
python train.py --batch-size 64 --lr 1e-4 --epochs 20
```

### Monitor Training

**TensorBoard:**
```bash
tensorboard --logdir logs
```

**Weights & Biases:**
Set `use_wandb: true` in config and run:
```bash
wandb login
python train.py
```

## 🎨 Text Generation (Inference)

### Interactive Mode
```bash
python -m src.inference --model checkpoints/best_model.pt --interactive
```

### Single Prompt
```bash
python -m src.inference \
  --model checkpoints/best_model.pt \
  --prompt "Once upon a time" \
  --max-tokens 100 \
  --temperature 0.8
```

### Python API
```python
from src.inference import TextGenerator

# Load model
generator = TextGenerator('checkpoints/best_model.pt')

# Generate text
text = generator.generate(
    prompt="The future of AI",
    max_new_tokens=100,
    temperature=0.8,
    top_k=50,
    top_p=0.95
)

print(text[0])
```

## 📈 Evaluation

```bash
# Evaluate on test set
python evaluate.py --checkpoint checkpoints/best_model.pt --split test

# Evaluate on validation set
python evaluate.py --checkpoint checkpoints/best_model.pt --split val
```

## 💡 Tips for Better Training

### 1. Start Small
- Use `tiny_shakespeare` or `wikitext-2` for initial experiments
- Verify your pipeline works before scaling up

### 2. Tune Hyperparameters
- Learning rate: 1e-4 to 5e-4 (smaller for larger models)
- Batch size: As large as your GPU allows
- Context length: 256-512 for most tasks

### 3. Monitor Validation Loss
- Should decrease steadily
- If it plateaus, try reducing learning rate
- If it increases, you're overfitting

### 4. Use Gradient Accumulation
- Simulate larger batch sizes on limited hardware
- Effective batch size = batch_size × gradient_accumulation_steps

### 5. Mixed Precision Training
- Enabled by default
- 2-3x faster training with minimal quality loss
- Requires modern GPU (Volta/Turing or newer)

## 🐛 Troubleshooting

### Out of Memory (OOM)
- Reduce `batch_size`
- Reduce `context_length`
- Reduce `d_model` or `num_layers`
- Increase `gradient_accumulation_steps`

### Slow Training
- Enable mixed precision: `mixed_precision: true`
- Increase `batch_size`
- Reduce `num_workers` if CPU bottleneck
- Use `torch.compile()`: `compile_model: true` (PyTorch 2.0+)

### Poor Generation Quality
- Train longer (more epochs/steps)
- Use larger model
- Use more/better training data
- Tune generation parameters (temperature, top_k, top_p)

### NaN Loss
- Reduce learning rate
- Enable gradient clipping (already enabled)
- Check for corrupted data

## 📚 Model Checkpoints

Checkpoints are saved in `checkpoints/` directory:

- `best_model.pt`: Best model based on validation loss
- `final_model.pt`: Model after training completes
- `checkpoint_step_N.pt`: Periodic checkpoints every N steps
- `interrupted_checkpoint.pt`: Saved when training is interrupted

Each checkpoint contains:
- Model weights
- Optimizer state
- Training step/epoch
- Configuration
- Best validation loss

## 🔬 Advanced Features

### Custom Dataset
```python
# In src/data.py, add your dataset loading logic
from datasets import load_dataset

dataset = load_dataset('your_dataset_name')
```

### Custom Tokenizer
```python
# Train your own tokenizer
from tokenizers import Tokenizer, models, trainers

tokenizer = Tokenizer(models.BPE())
# ... train tokenizer on your data
```

### Distributed Training
```bash
# Multi-GPU training (coming soon)
torchrun --nproc_per_node=4 train.py
```

## 📖 References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Language Models are Unsupervised Multitask Learners (GPT-2)](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [HuggingFace Datasets](https://huggingface.co/docs/datasets)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)

## 🤝 Contributing

This is a production-ready template. Feel free to:
- Add new features
- Improve documentation
- Report issues
- Share your trained models

## 📄 License

MIT License - feel free to use for any purpose.

---

**Happy Training! 🚀**
