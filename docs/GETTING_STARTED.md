# Getting Started with Transformer Training

A step-by-step guide to train your first language model.

## 🎯 Goal

By the end of this guide, you'll have:
1. ✅ Trained a small transformer model
2. ✅ Generated text with your model
3. ✅ Understanding of the complete pipeline

**Time required**: 15-30 minutes

---

## Step 1: Environment Setup (5 minutes)

### Activate Virtual Environment
```bash
source venv/bin/activate
```

### Install Dependencies
```bash
pip install torch transformers datasets tokenizers tqdm pyyaml tensorboard
```

**Verify installation:**
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

---

## Step 2: Quick Test (5 minutes)

### Run the Simple Example
```bash
cd examples
python simple_training_example.py
```

This will:
- Download tiny_shakespeare dataset (~1MB)
- Train a tiny model (2 layers, 128 dim)
- Take ~5 minutes on CPU, ~1 minute on GPU
- Save model to `example_checkpoints/`

**What to expect:**
```
Model initialized with 2.45M parameters
Loading data...
Dataset loaded:
  Train: 1000 examples
Starting Training
================================================================================
Epoch 0 | Loss: 4.2341
Epoch 0 | Loss: 3.8765
...
Training Complete!
```

---

## Step 3: Generate Text (2 minutes)

### Interactive Mode
```bash
python -m src.inference \
  --model example_checkpoints/best_model.pt \
  --interactive
```

**Try these prompts:**
- "Once upon a time"
- "To be or not to be"
- "The king said"

**Adjust settings:**
```
set temperature 0.9
set max_new_tokens 50
```

### Single Prompt
```bash
python -m src.inference \
  --model example_checkpoints/best_model.pt \
  --prompt "Once upon a time" \
  --max-tokens 100
```

---

## Step 4: Train on Real Data (30 minutes - 2 hours)

### Option A: WikiText-2 (Small, ~30 minutes)
```bash
python train.py \
  --dataset wikitext \
  --batch-size 32 \
  --epochs 5
```

**Expected results:**
- Training time: ~30 minutes on GPU
- Final perplexity: ~50-80
- Model size: ~40M parameters

### Option B: WikiText-103 (Medium, ~2 hours)
```bash
python train.py \
  --dataset wikitext \
  --batch-size 32 \
  --epochs 3
```

**Expected results:**
- Training time: ~2 hours on GPU
- Final perplexity: ~30-50
- Model size: ~40M parameters

---

## Step 5: Monitor Training

### TensorBoard
```bash
# In a new terminal
tensorboard --logdir logs
```

Open http://localhost:6006 to see:
- Training loss curve
- Learning rate schedule
- Validation metrics

### Watch Progress
Training will show:
```
Epoch 0: 100%|████████| 1250/1250 [05:23<00:00, 3.87it/s, loss=3.4521]

Generated Samples:
================================================================================
Prompt: Once upon a time
Generated: Once upon a time there was a king who ruled over a vast kingdom...
--------------------------------------------------------------------------------
```

---

## Step 6: Evaluate Your Model

```bash
python evaluate.py \
  --checkpoint checkpoints/best_model.pt \
  --split test
```

**Output:**
```
Evaluation Results:
================================================================================
Split: test
Average Loss: 3.2145
Perplexity: 24.87
================================================================================

Sample Generations:
Prompt: The future of artificial intelligence
Generated: The future of artificial intelligence is bright and full of...
```

---

## 🎓 Understanding the Pipeline

### 1. Data Flow
```
HuggingFace Dataset → Tokenizer → DataLoader → Model → Loss → Optimizer
```

### 2. Training Loop
```python
for epoch in epochs:
    for batch in train_loader:
        # Forward pass
        logits, loss = model(input_ids, targets)
        
        # Backward pass
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        # Validate periodically
        if step % eval_interval == 0:
            validate()
```

### 3. Generation
```python
# Start with prompt
input_ids = tokenizer.encode(prompt)

# Generate tokens one by one
for _ in range(max_new_tokens):
    logits = model(input_ids)
    next_token = sample(logits)  # Using temperature, top-k, top-p
    input_ids = append(input_ids, next_token)

# Decode to text
text = tokenizer.decode(input_ids)
```

---

## 🔧 Customization

### Change Model Size

**Small (Fast)**
```yaml
# config/model_config.yaml
model:
  d_model: 256
  num_heads: 4
  num_layers: 4
```

**Medium (Balanced)**
```yaml
model:
  d_model: 512
  num_heads: 8
  num_layers: 6
```

**Large (Quality)**
```yaml
model:
  d_model: 768
  num_heads: 12
  num_layers: 12
```

### Change Dataset

```bash
# Tiny (testing)
python train.py --dataset tiny_shakespeare

# Small (learning)
python train.py --dataset wikitext --config wikitext-2-raw-v1

# Medium (research)
python train.py --dataset wikitext --config wikitext-103-raw-v1

# Large (production)
python train.py --dataset openwebtext
```

### Tune Hyperparameters

```bash
python train.py \
  --batch-size 64 \
  --lr 1e-4 \
  --epochs 10
```

---

## 📊 What to Expect

### Training Metrics

**Good Training:**
- Loss decreases steadily
- Validation loss follows training loss
- Generated samples improve over time

**Overfitting:**
- Training loss decreases
- Validation loss increases
- Generated samples are repetitive

**Underfitting:**
- Both losses plateau high
- Generated samples are incoherent

### Generation Quality

**After 1 epoch:**
- Basic grammar
- Some coherent phrases
- Repetitive patterns

**After 5 epochs:**
- Good grammar
- Coherent sentences
- Some creativity

**After 10+ epochs:**
- Excellent grammar
- Coherent paragraphs
- Creative and diverse

---

## 🐛 Common Issues

### Issue: Out of Memory
**Solution:**
```bash
python train.py --batch-size 8  # Reduce batch size
```

Or edit config:
```yaml
training:
  batch_size: 8
  gradient_accumulation_steps: 8  # Effective batch = 64
```

### Issue: Slow Training
**Solution:**
```yaml
system:
  mixed_precision: true  # Enable AMP
  compile_model: true    # PyTorch 2.0+
```

### Issue: Poor Quality
**Solution:**
1. Train longer (more epochs)
2. Use larger model
3. Use more data
4. Tune generation parameters:
```bash
python -m src.inference \
  --model checkpoints/best_model.pt \
  --prompt "Your prompt" \
  --temperature 0.7 \
  --top-k 40 \
  --top-p 0.9
```

---

## 🎯 Next Steps

### 1. Experiment with Datasets
Try different datasets from [DATASETS.md](DATASETS.md):
- BookCorpus (books)
- OpenWebText (web)
- C4 (massive web corpus)

### 2. Scale Up
Gradually increase:
- Model size (d_model, num_layers)
- Dataset size
- Training time

### 3. Fine-tune
Train on your own data:
```python
# In src/data.py
dataset = load_dataset('your_dataset')
```

### 4. Deploy
Use the inference API in your application:
```python
from src.inference import TextGenerator

generator = TextGenerator('checkpoints/best_model.pt')
text = generator.generate("Your prompt")
```

---

## 📚 Learn More

- **[README_TRAINING.md](README_TRAINING.md)** - Complete training guide
- **[DATASETS.md](DATASETS.md)** - Available datasets
- **[ARCHITECTURE_COMPARISON.md](ARCHITECTURE_COMPARISON.md)** - Implementation details

---

## 🎉 Congratulations!

You've successfully:
- ✅ Set up the environment
- ✅ Trained a transformer model
- ✅ Generated text
- ✅ Understood the pipeline

**Now go train something amazing! 🚀**

---

## 💬 Need Help?

Common questions:

**Q: How long should I train?**
A: Until validation loss stops improving. Usually 5-10 epochs for small datasets.

**Q: What's a good perplexity?**
A: Lower is better. <30 is good, <20 is excellent for WikiText.

**Q: Can I use my own data?**
A: Yes! See [README_TRAINING.md](README_TRAINING.md) for custom datasets.

**Q: How do I resume training?**
A: `python train.py --resume checkpoints/checkpoint_step_5000.pt`

**Q: Can I use multiple GPUs?**
A: Multi-GPU support coming soon. For now, use one GPU.

---

**Happy Training! 🎓**
