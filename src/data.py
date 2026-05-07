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


class TiktokenWrapper:
    """Wrapper to make tiktoken compatible with HuggingFace tokenizer interface."""
    
    def __init__(self, encoding):
        self.encoding = encoding
        self.vocab_size = encoding.n_vocab
        self.eos_token = "<|endoftext|>"
        self.pad_token = "<|endoftext|>"
        self.eos_token_id = getattr(encoding, "eot_token", None)
        if self.eos_token_id is None:
            self.eos_token_id = encoding.encode_single_token(self.eos_token)
        self.pad_token_id = self.eos_token_id
    
    def encode(self, text, **kwargs):
        return self.encoding.encode(text, allowed_special='all')
    
    def decode(self, tokens, **kwargs):
        return self.encoding.decode(tokens)
    
    def __len__(self):
        return self.vocab_size


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


class CurriculumDataset(Dataset):
    """
    Dataset that mixes multiple sources according to curriculum learning schedule.
    Dynamically adjusts distribution per epoch.
    """
    
    def __init__(self, data_dir: str, sources: list, distribution: list, max_length: int, vocab_size: int):
        """
        Args:
            data_dir: Directory containing source binary files
            sources: List of source names (e.g., ['wikitext', 'ultrachat'])
            distribution: List of percentages for each source (must sum to 100)
            max_length: Maximum sequence length
            vocab_size: Vocabulary size
        """
        self.data_dir = Path(data_dir)
        self.sources = sources
        self.distribution = distribution
        self.max_length = max_length
        self.vocab_size = vocab_size
        
        # Validate distribution
        if abs(sum(distribution) - 100) > 0.1:
            raise ValueError(f"Distribution must sum to 100, got {sum(distribution)}")
        
        # Load all source datasets
        self.source_datasets = {}
        self.source_lengths = {}
        
        for source in sources:
            source_file = self.data_dir / f"{source}_train.bin"
            if not source_file.exists():
                raise FileNotFoundError(f"Source file not found: {source_file}")
            
            # Load as memory-mapped array
            tokens = np.memmap(source_file, dtype=np.uint32, mode='r')
            num_sequences = len(tokens) // max_length
            
            self.source_datasets[source] = tokens
            self.source_lengths[source] = num_sequences
            
            print(f"✅ Loaded {source}: {len(tokens):,} tokens, {num_sequences:,} sequences")
        
        # Calculate total length based on distribution
        if len(distribution) != len(sources):
            raise ValueError(
                f"Curriculum distribution length ({len(distribution)}) must match "
                f"sources length ({len(sources)})"
            )
        self._calculate_epoch_length()
        
        print(f"\n📊 Curriculum Distribution:")
        for source, pct in zip(sources, distribution):
            print(f"   {source}: {pct}%")
        print(f"   Total sequences this epoch: {self.total_sequences:,}")
    
    def _calculate_epoch_length(self):
        """Calculate total sequences for this epoch based on distribution."""
        active_sources = [
            (source, pct / 100.0)
            for source, pct in zip(self.sources, self.distribution)
            if pct > 0
        ]
        if not active_sources:
            raise ValueError("At least one curriculum source must have a non-zero distribution")

        # Pick the largest epoch size that preserves the requested percentages
        # without exhausting any source.
        total_sequences = int(
            min(self.source_lengths[source] / fraction for source, fraction in active_sources)
        )

        self.source_sequence_counts = {}
        for source, pct in zip(self.sources, self.distribution):
            count = int(total_sequences * (pct / 100.0))
            self.source_sequence_counts[source] = min(count, self.source_lengths[source])
        
        self.total_sequences = sum(self.source_sequence_counts.values())
        
        # Create index mapping: global_idx -> (source, source_idx)
        self.index_map = []
        for source in self.sources:
            count = self.source_sequence_counts[source]
            for i in range(count):
                # Use modulo to cycle through source if needed
                source_idx = i % self.source_lengths[source]
                self.index_map.append((source, source_idx))
        
        # Shuffle the index map for better mixing
        np.random.shuffle(self.index_map)
    
    def update_distribution(self, new_distribution: list):
        """Update distribution for new epoch."""
        if len(new_distribution) != len(self.sources):
            raise ValueError(
                f"Curriculum distribution length ({len(new_distribution)}) must match "
                f"sources length ({len(self.sources)})"
            )
        if abs(sum(new_distribution) - 100) > 0.1:
            raise ValueError(f"Distribution must sum to 100, got {sum(new_distribution)}")
        
        self.distribution = new_distribution
        self._calculate_epoch_length()
        
        print(f"\n📊 Updated Curriculum Distribution:")
        for source, pct in zip(self.sources, new_distribution):
            print(f"   {source}: {pct}%")
        print(f"   Total sequences this epoch: {self.total_sequences:,}")
    
    def __len__(self):
        return self.total_sequences
    
    def __getitem__(self, idx):
        """Get a sequence from the appropriate source."""
        if idx >= len(self.index_map):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.index_map)}")
        
        source, source_idx = self.index_map[idx]
        
        # Get tokens from source
        start_idx = source_idx * self.max_length
        end_idx = start_idx + self.max_length
        
        tokens = self.source_datasets[source][start_idx:end_idx]
        tokens = torch.from_numpy(tokens.astype(np.int64))
        
        return tokens


def collate_fn(batch, max_length, pad_token_id=0, ignore_index=-100):
    """
    Custom collate function for batching.
    Handles variable length sequences and creates input-target pairs.
    """
    # Find max length in batch
    max_len = min(max([len(x) for x in batch]), max_length)
    
    # Pad sequences
    input_ids = []
    sequence_lengths = []
    for tokens in batch:
        seq_len = min(len(tokens), max_len)
        if len(tokens) < max_len:
            padded = torch.cat(
                [
                    tokens,
                    torch.full((max_len - len(tokens),), pad_token_id, dtype=torch.long),
                ]
            )
        else:
            padded = tokens[:max_len]
        input_ids.append(padded)
        sequence_lengths.append(seq_len)
    
    input_ids = torch.stack(input_ids)
    
    # Do not shift here. The model handles the causal shift.
    targets = input_ids.clone()

    # Mask padded labels so they do not contribute to loss.
    for i, seq_len in enumerate(sequence_lengths):
        if seq_len < max_len:
            targets[i, seq_len:] = ignore_index
    
    return input_ids, targets


class CollateFnWrapper:
    """Wrapper for collate_fn that can be pickled for multiprocessing."""
    
    def __init__(self, max_length, pad_token_id, ignore_index=-100):
        self.max_length = max_length
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index
    
    def __call__(self, batch):
        return collate_fn(
            batch,
            self.max_length,
            pad_token_id=self.pad_token_id,
            ignore_index=self.ignore_index,
        )


def _load_tokenizer_from_config(config):
    """Load tokenizer once and update config vocab size from actual tokenizer."""
    tokenizer_type = config['tokenizer']['type']
    print(f"Loading tokenizer: {tokenizer_type}")

    if tokenizer_type == 'tiktoken' or 'cl100k' in tokenizer_type.lower():
        print("Using tiktoken cl100k_base (GPT-4) tokenizer")
        encoding = tiktoken.get_encoding("cl100k_base")
        tokenizer = TiktokenWrapper(encoding)
        actual_vocab_size = tokenizer.vocab_size
        print(f"✅ Tiktoken loaded: vocab_size={actual_vocab_size:,}")
    else:
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_type)
            print(f"✅ Tokenizer loaded successfully: {tokenizer_type}")
        except Exception as e:
            print(f"⚠️  Error loading tokenizer '{tokenizer_type}': {e}")
            print("   Falling back to GPT-2 tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained('gpt2')

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print(f"   Set pad_token = eos_token ({tokenizer.eos_token})")

        actual_vocab_size = len(tokenizer)
        print(f"✅ Tokenizer vocabulary size: {actual_vocab_size:,}")

    config['model']['vocab_size'] = actual_vocab_size
    expected_vocab = config['tokenizer'].get('vocab_size', actual_vocab_size)
    if actual_vocab_size != expected_vocab:
        print(f"⚠️  Vocab size mismatch: expected {expected_vocab:,}, got {actual_vocab_size:,}")
        print(f"   Using actual vocab size: {actual_vocab_size:,}")

    return tokenizer


def load_data(config):
    """
    Load and prepare datasets with support for both HuggingFace and binary formats.
    
    Returns:
        train_loader, val_loader, test_loader, tokenizer
    """
    dataset_type = config['data'].get('dataset_type', 'huggingface')
    
    tokenizer = _load_tokenizer_from_config(config)
    
    # Load dataset based on type
    if dataset_type == 'binary':
        return load_binary_data(config, tokenizer)
    else:
        return load_huggingface_data(config, tokenizer)


def load_binary_data(config, tokenizer):
    """Load pre-tokenized binary datasets with automatic download and curriculum learning support."""
    print("\n" + "="*80)
    print("Loading Binary Dataset")
    print("="*80)
    
    # Check if curriculum learning is enabled
    curriculum_config = config['training'].get('curriculum_learning', {})
    curriculum_enabled = curriculum_config.get('enabled', False)
    
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
    
    if curriculum_enabled:
        # Curriculum learning mode - load multiple sources
        print("\n🎓 Curriculum Learning ENABLED")
        
        sources = curriculum_config.get('sources', ['wikitext', 'ultrachat'])
        epoch_distributions = curriculum_config.get('epoch_distributions', {})
        
        # Get distribution for epoch 1 (will be updated by trainer)
        initial_distribution = epoch_distributions.get(1)
        if initial_distribution is None:
            base_pct = 100 // len(sources)
            initial_distribution = [base_pct] * len(sources)
            initial_distribution[-1] += 100 - sum(initial_distribution)
        if len(initial_distribution) != len(sources):
            raise ValueError(
                f"Curriculum distribution length ({len(initial_distribution)}) must match "
                f"sources length ({len(sources)})"
            )
        
        # Get data directory (parent of train_file)
        data_dir = Path(train_file).parent
        
        train_dataset = CurriculumDataset(
            data_dir=str(data_dir),
            sources=sources,
            distribution=initial_distribution,
            max_length=max_length,
            vocab_size=vocab_size
        )
        
        # Store curriculum config in dataset for trainer access
        train_dataset.curriculum_config = curriculum_config
        
    else:
        # Standard mode - single binary file
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
    
    # Do not shift here. The model handles the causal shift.
    targets = input_ids.clone()
    
    return input_ids, targets


def load_huggingface_data(config, tokenizer):
    """Load datasets from HuggingFace."""
    print(f"Loading dataset: {config['data']['dataset_name']}")
    
    # Setup cache directory
    cache_dir = Path('datasets')
    cache_dir.mkdir(exist_ok=True)
    print(f"Dataset cache directory: {cache_dir.absolute()}")
    
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
    max_length = config['data']['max_length']
    
    print(f"\nDataLoader settings:")
    print(f"  Batch size: {config['training']['batch_size']}")
    print(f"  Num workers: {num_workers}")
    print(f"  Pin memory: {pin_memory}")
    print(f"  Max length: {max_length}")
    
    # Create collate function wrapper (can be pickled for multiprocessing)
    pad_token_id = getattr(tokenizer, "pad_token_id", 0)
    collate_wrapper = CollateFnWrapper(max_length, pad_token_id=pad_token_id, ignore_index=-100)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_wrapper,
        pin_memory=pin_memory
    )
    
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_wrapper,
            pin_memory=pin_memory
        )
    
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_wrapper,
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
