"""
Data loading and preprocessing pipeline.
Supports both HuggingFace datasets and binary token files.
"""
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np
from pathlib import Path
import tiktoken
from .dataset_downloader import DatasetDownloader, configure_tqdm_for_datasets

# Configure tqdm for better progress bars
configure_tqdm_for_datasets()


class TextDataset(Dataset):
    """Dataset for language modeling."""
    
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]['text']
        
        # Tokenize with truncation to avoid sequence length warnings
        tokens = self.tokenizer.encode(
            text, 
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length
        )
        
        # Convert to tensor
        tokens = torch.tensor(tokens, dtype=torch.long)
        
        # For language modeling, input is tokens[:-1] and target is tokens[1:]
        # But we'll handle this in collate_fn
        return tokens


class BinaryDataset(Dataset):
    """
    Dataset for pre-tokenized binary files.
    Uses memory-mapped files for efficient loading.
    """
    
    def __init__(self, binary_file: str, max_length: int, vocab_size: int):
        """
        Args:
            binary_file: Path to .bin file containing uint32 tokens
            max_length: Maximum sequence length
            vocab_size: Vocabulary size for validation
        """
        self.binary_file = Path(binary_file)
        self.max_length = max_length
        self.vocab_size = vocab_size
        
        if not self.binary_file.exists():
            raise FileNotFoundError(f"Binary file not found: {self.binary_file}")
        
        # Load as memory-mapped array for efficiency
        self.tokens = np.memmap(self.binary_file, dtype=np.uint32, mode='r')
        self.total_tokens = len(self.tokens)
        
        # Calculate number of sequences
        # Each sequence is max_length tokens
        self.num_sequences = self.total_tokens // max_length
        
        print(f"✅ Loaded binary dataset: {self.binary_file}")
        print(f"   Total tokens: {self.total_tokens:,}")
        print(f"   Sequences: {self.num_sequences:,}")
        print(f"   Max length: {max_length}")
    
    def __len__(self):
        return self.num_sequences
    
    def __getitem__(self, idx):
        """Get a sequence of tokens."""
        start_idx = idx * self.max_length
        end_idx = start_idx + self.max_length
        
        # Get tokens
        tokens = self.tokens[start_idx:end_idx]
        
        # Convert to tensor
        tokens = torch.from_numpy(tokens.astype(np.int64))
        
        return tokens


def collate_fn(batch, max_length):
    """
    Custom collate function for batching.
    Handles variable length sequences and creates input-target pairs.
    """
    # Find max length in batch
    max_len = min(max([len(x) for x in batch]), max_length)
    
    # Pad sequences
    input_ids = []
    for tokens in batch:
        if len(tokens) < max_len:
            # Pad with tokenizer pad token (usually 0)
            padded = torch.cat([
                tokens,
                torch.zeros(max_len - len(tokens), dtype=torch.long)
            ])
        else:
            padded = tokens[:max_len]
        input_ids.append(padded)
    
    input_ids = torch.stack(input_ids)
    
    # Create targets (shifted by 1 for next-token prediction)
    # For language modeling: predict next token
    # Input:  [BOS, token1, token2, token3]
    # Target: [token1, token2, token3, EOS]
    # We shift targets left by 1 position
    targets = input_ids[:, 1:].contiguous()  # Remove first token
    input_ids = input_ids[:, :-1].contiguous()  # Remove last token
    
    # Now input_ids and targets have same shape (B, T-1)
    # input_ids[i] predicts targets[i] (which is input_ids[i+1] from original)
    
    return input_ids, targets


def load_data(config):
    """
    Load and prepare datasets with support for both HuggingFace and binary formats.
    
    Returns:
        train_loader, val_loader, test_loader, tokenizer
    """
    dataset_type = config['data'].get('dataset_type', 'huggingface')
    
    # Load tokenizer
    tokenizer_type = config['tokenizer']['type']
    print(f"Loading tokenizer: {tokenizer_type}")
    
    # Check if using tiktoken
    if tokenizer_type == 'tiktoken' or 'cl100k' in tokenizer_type.lower():
        print("Using tiktoken cl100k_base (GPT-4) tokenizer")
        tokenizer = tiktoken.get_encoding("cl100k_base")
        actual_vocab_size = tokenizer.n_vocab
        print(f"✅ Tiktoken loaded: vocab_size={actual_vocab_size:,}")
        
        # Create a wrapper to match HuggingFace interface
        class TiktokenWrapper:
            def __init__(self, encoding):
                self.encoding = encoding
                self.vocab_size = encoding.n_vocab
                self.eos_token = "<|endoftext|>"
                self.pad_token = "<|endoftext|>"
            
            def encode(self, text, **kwargs):
                return self.encoding.encode(text, allowed_special='all')
            
            def decode(self, tokens, **kwargs):
                return self.encoding.decode(tokens)
            
            def __len__(self):
                return self.vocab_size
        
        tokenizer = TiktokenWrapper(tokenizer)
    else:
        # Use HuggingFace tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_type)
            print(f"✅ Tokenizer loaded successfully: {tokenizer_type}")
        except Exception as e:
            print(f"⚠️  Error loading tokenizer '{tokenizer_type}': {e}")
            print(f"   Falling back to GPT-2 tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained('gpt2')
            tokenizer_type = 'gpt2'
        
        # Set pad token if not exists
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print(f"   Set pad_token = eos_token ({tokenizer.eos_token})")
        
        actual_vocab_size = len(tokenizer)
        print(f"✅ Tokenizer vocabulary size: {actual_vocab_size:,}")
    
    # Update vocab size in config
    config['model']['vocab_size'] = actual_vocab_size
    
    # Warn if vocab size mismatch
    expected_vocab = config['tokenizer'].get('vocab_size', actual_vocab_size)
    if actual_vocab_size != expected_vocab:
        print(f"⚠️  Vocab size mismatch: expected {expected_vocab:,}, got {actual_vocab_size:,}")
        print(f"   Using actual vocab size: {actual_vocab_size:,}")
    
    # Load dataset based on type
    if dataset_type == 'binary':
        return load_binary_data(config, tokenizer)
    else:
        return load_huggingface_data(config, tokenizer)


def load_binary_data(config, tokenizer):
    """Load pre-tokenized binary datasets with automatic download support."""
    print("\n" + "="*80)
    print("Loading Binary Dataset")
    print("="*80)
    
    # Initialize dataset downloader
    downloader = DatasetDownloader()
    
    try:
        # Ensure dataset is available (download if necessary)
        train_file, val_file, stats = downloader.ensure_dataset_available(config)
        
        # Update config with actual file paths
        config['data']['train_file'] = train_file
        if val_file:
            config['data']['val_file'] = val_file
        
        print(f"\n✅ Dataset files confirmed:")
        print(f"  Train: {train_file}")
        if val_file:
            print(f"  Val: {val_file}")
        
    except Exception as e:
        print(f"❌ Dataset availability check failed: {e}")
        print(f"\nFalling back to manual file paths...")
        train_file = config['data'].get('train_file')
        val_file = config['data'].get('val_file')
        stats = {}
    
    max_length = config['data']['max_length']
    vocab_size = config['model']['vocab_size']
    
    if not train_file:
        raise ValueError("train_file must be specified for binary dataset")
    
    # Create datasets
    print("\nCreating PyTorch datasets...")
    train_dataset = BinaryDataset(train_file, max_length, vocab_size)
    
    val_dataset = None
    if val_file and Path(val_file).exists():
        val_dataset = BinaryDataset(val_file, max_length, vocab_size)
    
    # Get data loading parameters
    num_workers = config['data'].get('num_workers', 0)
    pin_memory = config['data'].get('pin_memory', False)
    batch_size = config['training']['batch_size']
    
    print(f"\nDataLoader settings:")
    print(f"  Batch size: {batch_size}")
    print(f"  Num workers: {num_workers}")
    print(f"  Pin memory: {pin_memory}")
    print(f"  Max length: {max_length}")
    
    # Create dataloaders with binary collate function
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn_binary,
        pin_memory=pin_memory
    )
    
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn_binary,
            pin_memory=pin_memory
        )
    
    print(f"\n✅ Data loading complete!")
    print(f"  Train batches: {len(train_loader):,}")
    if val_loader:
        print(f"  Val batches: {len(val_loader):,}")
    
    # Print dataset statistics if available
    if stats:
        print(f"\n📊 Dataset Statistics:")
        totals = stats.get('totals', {})
        if totals:
            print(f"  Total tokens: {totals.get('tokens', 'Unknown'):,}")
            print(f"  Total documents: {totals.get('documents', 'Unknown'):,}")
        
        tokenizer_info = stats.get('tokenizer', 'Unknown')
        print(f"  Tokenizer: {tokenizer_info}")
    
    return train_loader, val_loader, None, tokenizer


def collate_fn_binary(batch):
    """
    Collate function for binary datasets.
    Creates input-target pairs for language modeling.
    """
    # Stack batch
    input_ids = torch.stack(batch)
    
    # Create targets (shifted by 1 for next-token prediction)
    targets = input_ids[:, 1:].contiguous()
    input_ids = input_ids[:, :-1].contiguous()
    
    return input_ids, targets


def load_huggingface_data(config, tokenizer):
    """Load datasets from HuggingFace (original implementation)."""
    print(f"Loading dataset: {config['data']['dataset_name']}")
    
    # Setup cache directory
    cache_dir = Path('datasets')
    cache_dir.mkdir(exist_ok=True)
    print(f"Dataset cache directory: {cache_dir.absolute()}")
    
    # Load tokenizer with error handling
    tokenizer_type = config['tokenizer']['type']
    print(f"Loading tokenizer: {tokenizer_type}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_type)
        print(f"✅ Tokenizer loaded successfully: {tokenizer_type}")
    except Exception as e:
        print(f"⚠️  Error loading tokenizer '{tokenizer_type}': {e}")
        print(f"   Falling back to GPT-2 tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        tokenizer_type = 'gpt2'
    
    # Set pad token if not exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"   Set pad_token = eos_token ({tokenizer.eos_token})")
    
    # Update vocab size in config
    actual_vocab_size = len(tokenizer)
    config['model']['vocab_size'] = actual_vocab_size
    print(f"✅ Tokenizer vocabulary size: {actual_vocab_size:,}")
    
    # Warn if vocab size mismatch
    expected_vocab = config['tokenizer'].get('vocab_size', actual_vocab_size)
    if actual_vocab_size != expected_vocab:
        print(f"⚠️  Vocab size mismatch: expected {expected_vocab:,}, got {actual_vocab_size:,}")
        print(f"   Using actual vocab size: {actual_vocab_size:,}")
    
    # Load dataset from HuggingFace with robust error handling and caching
    dataset = None
    dataset_name = config['data']['dataset_name']
    dataset_config = config['data'].get('dataset_config')
    
    try:
        print(f"Loading dataset: {dataset_name}" + (f" ({dataset_config})" if dataset_config else ""))
        print(f"   Cache will be stored in: {cache_dir / dataset_name}")
        
        if dataset_name == 'wikitext':
            dataset = load_dataset('wikitext', dataset_config or 'wikitext-2-raw-v1', cache_dir=str(cache_dir))
        elif dataset_name == 'openwebtext':
            dataset = load_dataset('openwebtext', cache_dir=str(cache_dir))
        elif dataset_name == 'c4' or dataset_name == 'allenai/c4':
            dataset = load_dataset('allenai/c4', dataset_config or 'en', streaming=False, cache_dir=str(cache_dir))
        elif dataset_name == 'togethercomputer/RedPajama-Data-1T':
            print("⚠️  RedPajama is very large (1TB+). Consider using streaming mode.")
            dataset = load_dataset(dataset_name, dataset_config or 'default', streaming=False, cache_dir=str(cache_dir))
        elif dataset_name == 'bookcorpus':
            dataset = load_dataset('bookcorpus', cache_dir=str(cache_dir))
        elif dataset_name == 'tiny_shakespeare':
            print("⚠️  tiny_shakespeare is deprecated. Using wikitext-2 instead.")
            dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', cache_dir=str(cache_dir))
        else:
            # Try to load any dataset by name
            if dataset_config:
                dataset = load_dataset(dataset_name, dataset_config, cache_dir=str(cache_dir))
            else:
                dataset = load_dataset(dataset_name, cache_dir=str(cache_dir))
        
        print(f"✅ Dataset loaded successfully (cached locally)")
        
    except Exception as e:
        print(f"❌ Error loading dataset '{dataset_name}': {e}")
        print(f"   Falling back to wikitext-2 for testing...")
        try:
            dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', cache_dir=str(cache_dir))
            print(f"✅ Fallback dataset loaded: wikitext-2")
        except Exception as e2:
            print(f"❌ Critical error: Could not load fallback dataset: {e2}")
            raise RuntimeError("Failed to load any dataset. Please check your internet connection and HuggingFace access.")
    
    # Filter out empty texts
    def filter_empty(example):
        return len(example['text'].strip()) > 0
    
    print("Filtering empty examples...")
    dataset = dataset.filter(filter_empty)
    
    # Validate splits exist
    train_split = config['data']['train_split']
    val_split = config['data']['val_split']
    test_split = config['data']['test_split']
    
    if train_split not in dataset:
        print(f"⚠️  Train split '{train_split}' not found in dataset")
        print(f"   Available splits: {list(dataset.keys())}")
        # Use first available split
        train_split = list(dataset.keys())[0]
        print(f"   Using '{train_split}' as train split")
    
    # Calculate approximate token counts
    def estimate_tokens(split_data, tokenizer, sample_size=1000):
        """Estimate total tokens in dataset by sampling."""
        sample_size = min(sample_size, len(split_data))
        total_tokens = 0
        for i in range(sample_size):
            text = split_data[i]['text']
            tokens = tokenizer.encode(text, add_special_tokens=True)
            total_tokens += len(tokens)
        avg_tokens_per_example = total_tokens / sample_size
        estimated_total = int(avg_tokens_per_example * len(split_data))
        return estimated_total, avg_tokens_per_example
    
    print(f"\n✅ Dataset splits:")
    train_tokens, train_avg = estimate_tokens(dataset[train_split], tokenizer)
    print(f"  Train ({train_split}):")
    print(f"    - Examples: {len(dataset[train_split]):,}")
    print(f"    - Estimated tokens: {train_tokens:,} ({train_tokens/1e9:.2f}B)" if train_tokens > 1e9 else f"    - Estimated tokens: {train_tokens:,} ({train_tokens/1e6:.1f}M)")
    print(f"    - Avg tokens/example: {train_avg:.0f}")
    
    if val_split in dataset:
        val_tokens, val_avg = estimate_tokens(dataset[val_split], tokenizer, sample_size=100)
        print(f"  Validation ({val_split}):")
        print(f"    - Examples: {len(dataset[val_split]):,}")
        print(f"    - Estimated tokens: {val_tokens:,} ({val_tokens/1e6:.1f}M)")
    else:
        print(f"  Validation: Not available (will use train split)")
        val_split = train_split
    
    if test_split in dataset:
        test_tokens, test_avg = estimate_tokens(dataset[test_split], tokenizer, sample_size=100)
        print(f"  Test ({test_split}):")
        print(f"    - Examples: {len(dataset[test_split]):,}")
        print(f"    - Estimated tokens: {test_tokens:,} ({test_tokens/1e6:.1f}M)")
    else:
        print(f"  Test: Not available (will use train split)")
        test_split = train_split
    
    # Create datasets
    print("\nCreating PyTorch datasets...")
    train_dataset = TextDataset(
        dataset[train_split],
        tokenizer,
        config['data']['max_length']
    )
    
    val_dataset = None
    if val_split in dataset and val_split != train_split:
        val_dataset = TextDataset(
            dataset[val_split],
            tokenizer,
            config['data']['max_length']
        )
    
    test_dataset = None
    if test_split in dataset and test_split != train_split:
        test_dataset = TextDataset(
            dataset[test_split],
            tokenizer,
            config['data']['max_length']
        )
    
    # Get data loading parameters
    num_workers = config['data'].get('num_workers', 0)
    pin_memory = config['data'].get('pin_memory', False)
    
    print(f"\nDataLoader settings:")
    print(f"  Batch size: {config['training']['batch_size']}")
    print(f"  Num workers: {num_workers}")
    print(f"  Pin memory: {pin_memory}")
    print(f"  Max length: {config['data']['max_length']}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        collate_fn=lambda x: collate_fn(x, config['data']['max_length']),
        pin_memory=pin_memory
    )
    
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=num_workers,
            collate_fn=lambda x: collate_fn(x, config['data']['max_length']),
            pin_memory=pin_memory
        )
    
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=num_workers,
            collate_fn=lambda x: collate_fn(x, config['data']['max_length']),
            pin_memory=pin_memory
        )
    
    print(f"\n✅ Data loading complete!")
    print(f"  Train batches: {len(train_loader):,}")
    if val_loader:
        print(f"  Val batches: {len(val_loader):,}")
    if test_loader:
        print(f"  Test batches: {len(test_loader):,}")
    
    return train_loader, val_loader, test_loader, tokenizer


def get_sample_prompts():
    """Return sample prompts for generation during validation."""
    return [
        "Once upon a time",
        "The future of artificial intelligence",
        "In a world where",
        "The scientist discovered",
        "Deep in the forest"
    ]
