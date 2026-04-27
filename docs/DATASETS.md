# Open Source Text Datasets for Training

A comprehensive guide to open-source datasets available for training language models.

## 🎯 Quick Recommendations

| Use Case | Dataset | Size | Best For |
|----------|---------|------|----------|
| **Testing/Development** | tiny_shakespeare | 1MB | Quick experiments |
| **Small Projects** | wikitext-2 | 2M tokens | Learning, prototyping |
| **Medium Projects** | wikitext-103 | 100M tokens | Balanced training |
| **Production** | openwebtext | 40GB | High-quality models |
| **Large Scale** | The Pile | 825GB | State-of-the-art models |

## 📚 Detailed Dataset Information

### 1. Tiny Shakespeare
- **Size**: ~1MB
- **Tokens**: ~1M
- **Content**: Complete works of Shakespeare
- **Use**: Quick testing, debugging pipelines
- **HuggingFace**: `tiny_shakespeare`

```python
from datasets import load_dataset
dataset = load_dataset('tiny_shakespeare')
```

**Pros**: Extremely fast to download and train
**Cons**: Very limited vocabulary and domain

---

### 2. WikiText-2
- **Size**: ~4MB
- **Tokens**: ~2M
- **Content**: Wikipedia articles
- **Use**: Small-scale experiments, education
- **HuggingFace**: `wikitext`, config: `wikitext-2-raw-v1`

```python
dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')
```

**Pros**: Clean, well-formatted text; fast training
**Cons**: Limited size, may not generalize well

---

### 3. WikiText-103
- **Size**: ~500MB
- **Tokens**: ~100M
- **Content**: Wikipedia articles (larger subset)
- **Use**: Medium-scale training, research
- **HuggingFace**: `wikitext`, config: `wikitext-103-raw-v1`

```python
dataset = load_dataset('wikitext', 'wikitext-103-raw-v1')
```

**Pros**: Good balance of size and quality; diverse topics
**Cons**: Still Wikipedia-only (limited domain diversity)

---

### 4. OpenWebText
- **Size**: ~40GB
- **Tokens**: ~8B
- **Content**: Web pages from Reddit URLs
- **Use**: Production models, high-quality training
- **HuggingFace**: `openwebtext`

```python
dataset = load_dataset('openwebtext')
```

**Pros**: Large, diverse, high-quality; similar to GPT-2 training data
**Cons**: Large download; requires significant compute

---

### 5. BookCorpus
- **Size**: ~5GB
- **Tokens**: ~1B
- **Content**: 11,000+ unpublished books
- **Use**: Long-form text generation
- **HuggingFace**: `bookcorpus`

```python
dataset = load_dataset('bookcorpus')
```

**Pros**: Long, coherent narratives; good for story generation
**Cons**: Single domain (fiction); availability issues

---

### 6. C4 (Colossal Clean Crawled Corpus)
- **Size**: ~750GB (en)
- **Tokens**: ~156B
- **Content**: Cleaned Common Crawl data
- **Use**: Large-scale training (T5, etc.)
- **HuggingFace**: `c4`, config: `en`

```python
dataset = load_dataset('c4', 'en', streaming=True)  # Use streaming for large datasets
```

**Pros**: Massive, diverse, cleaned; used by T5
**Cons**: Very large; requires streaming or significant storage

---

### 7. The Pile
- **Size**: 825GB
- **Tokens**: ~300B
- **Content**: 22 diverse high-quality datasets
- **Use**: State-of-the-art model training
- **HuggingFace**: `EleutherAI/pile`

```python
dataset = load_dataset('EleutherAI/pile', streaming=True)
```

**Components**:
- Books (BookCorpus, Books3)
- Academic (ArXiv, PubMed)
- Code (GitHub)
- Web (Common Crawl)
- Dialogue (Ubuntu IRC, EuroParl)
- And more...

**Pros**: Extremely diverse; highest quality; used by GPT-Neo/J
**Cons**: Massive size; requires significant resources

---

### 8. RedPajama
- **Size**: ~1.2TB
- **Tokens**: ~1T
- **Content**: Open reproduction of LLaMA training data
- **Use**: Large-scale open model training
- **HuggingFace**: `togethercomputer/RedPajama-Data-1T`

```python
dataset = load_dataset('togethercomputer/RedPajama-Data-1T', streaming=True)
```

**Pros**: Fully open; diverse sources; high quality
**Cons**: Extremely large; requires significant resources

---

### 9. OSCAR
- **Size**: Varies by language
- **Tokens**: ~6.3T (all languages)
- **Content**: Multilingual web corpus
- **Use**: Multilingual models
- **HuggingFace**: `oscar`, config: language code

```python
dataset = load_dataset('oscar', 'unshuffled_deduplicated_en')
```

**Pros**: Multilingual; large; deduplicated
**Cons**: Variable quality; very large

---

### 10. mC4 (Multilingual C4)
- **Size**: ~10TB
- **Tokens**: Varies by language
- **Content**: Multilingual cleaned Common Crawl
- **Use**: Multilingual model training
- **HuggingFace**: `mc4`, config: language code

```python
dataset = load_dataset('mc4', 'en', streaming=True)
```

**Pros**: Multilingual; cleaned; massive
**Cons**: Extremely large; requires streaming

---

## 🎓 Domain-Specific Datasets

### Code
- **The Stack**: 3TB of permissively licensed code
  ```python
  dataset = load_dataset('bigcode/the-stack', streaming=True)
  ```

- **CodeParrot**: GitHub code dataset
  ```python
  dataset = load_dataset('codeparrot/github-code')
  ```

### Scientific
- **ArXiv**: Scientific papers
  ```python
  dataset = load_dataset('arxiv_dataset')
  ```

- **PubMed**: Medical literature
  ```python
  dataset = load_dataset('pubmed')
  ```

### Dialogue
- **Ubuntu Dialogue Corpus**: Technical support conversations
- **PersonaChat**: Conversational dataset with personas
  ```python
  dataset = load_dataset('bavard/personachat_truecased')
  ```

---

## 💡 Choosing the Right Dataset

### For Learning & Experimentation
1. Start with **tiny_shakespeare** (1 hour training)
2. Move to **wikitext-2** (few hours)
3. Try **wikitext-103** (1-2 days)

### For Research Projects
1. **wikitext-103**: Good baseline
2. **openwebtext**: Better quality
3. **The Pile**: State-of-the-art

### For Production Models
1. **openwebtext**: Minimum for decent quality
2. **C4**: Better diversity
3. **The Pile** or **RedPajama**: Best quality

### For Specific Domains
- **Code**: The Stack, CodeParrot
- **Science**: ArXiv, PubMed
- **Books**: BookCorpus
- **Multilingual**: mC4, OSCAR

---

## 🔧 Using Datasets with HuggingFace Token

Some datasets require authentication:

```bash
# Login to HuggingFace
huggingface-cli login

# Or set token in code
from huggingface_hub import login
login(token="your_token_here")
```

---

## 📊 Dataset Streaming

For large datasets, use streaming to avoid downloading everything:

```python
from datasets import load_dataset

# Stream instead of download
dataset = load_dataset('c4', 'en', streaming=True)

# Iterate without loading all data
for example in dataset['train']:
    print(example['text'])
    break
```

---

## 🎯 Training Time Estimates

Approximate training times on a single V100 GPU:

| Dataset | Size | Tokens | Training Time (1 epoch) |
|---------|------|--------|------------------------|
| tiny_shakespeare | 1MB | 1M | 5 minutes |
| wikitext-2 | 4MB | 2M | 15 minutes |
| wikitext-103 | 500MB | 100M | 6 hours |
| openwebtext | 40GB | 8B | 2-3 days |
| The Pile | 825GB | 300B | 2-3 weeks |

*Times vary based on model size, batch size, and hardware*

---

## 📝 Dataset Quality Considerations

### High Quality
- ✅ The Pile
- ✅ RedPajama
- ✅ OpenWebText
- ✅ C4

### Medium Quality
- ⚠️ WikiText
- ⚠️ BookCorpus
- ⚠️ OSCAR

### Testing Only
- ⚠️ tiny_shakespeare

---

## 🔗 Useful Links

- [HuggingFace Datasets Hub](https://huggingface.co/datasets)
- [The Pile Paper](https://arxiv.org/abs/2101.00027)
- [C4 Documentation](https://www.tensorflow.org/datasets/catalog/c4)
- [RedPajama Project](https://www.together.xyz/blog/redpajama)

---

## 💾 Storage Requirements

Plan your storage based on dataset choice:

- **Development**: 1-10GB (tiny_shakespeare, wikitext)
- **Research**: 50-100GB (openwebtext, wikitext-103)
- **Production**: 500GB-2TB (C4, The Pile, RedPajama)

Use streaming for datasets larger than your available storage!

---

**Remember**: Start small, validate your pipeline, then scale up! 🚀
