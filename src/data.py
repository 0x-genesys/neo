"""
Data loading and preprocessing pipeline.
"""
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np


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
        
        # Tokenize
        tokens = self.tokenizer.encode(text, add_special_tokens=True)
        
        # Truncate or pad
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        
        # Convert to tensor
        tokens = torch.tensor(tokens, dtype=torch.long)
        
        # For language modeling, input is tokens[:-1] and target is tokens[1:]
        # But we'll handle this in collate_fn
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
    Load and prepare datasets.
    
    Returns:
        train_loader, val_loader, test_loader, tokenizer
    """
    print(f"Loading dataset: {config['data']['dataset_name']}")
    
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
    
    # Load dataset from HuggingFace with robust error handling
    dataset = None
    dataset_name = config['data']['dataset_name']
    dataset_config = config['data'].get('dataset_config')
    
    try:
        print(f"Loading dataset: {dataset_name}" + (f" ({dataset_config})" if dataset_config else ""))
        
        if dataset_name == 'wikitext':
            dataset = load_dataset('wikitext', dataset_config or 'wikitext-2-raw-v1')
        elif dataset_name == 'openwebtext':
            dataset = load_dataset('openwebtext')
        elif dataset_name == 'c4' or dataset_name == 'allenai/c4':
            dataset = load_dataset('allenai/c4', dataset_config or 'en', streaming=False)
        elif dataset_name == 'togethercomputer/RedPajama-Data-1T':
            print("⚠️  RedPajama is very large (1TB+). Consider using streaming mode.")
            dataset = load_dataset(dataset_name, dataset_config or 'default', streaming=False)
        elif dataset_name == 'bookcorpus':
            dataset = load_dataset('bookcorpus')
        elif dataset_name == 'tiny_shakespeare':
            print("⚠️  tiny_shakespeare is deprecated. Using wikitext-2 instead.")
            dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')
        else:
            # Try to load any dataset by name
            if dataset_config:
                dataset = load_dataset(dataset_name, dataset_config)
            else:
                dataset = load_dataset(dataset_name)
        
        print(f"✅ Dataset loaded successfully")
        
    except Exception as e:
        print(f"❌ Error loading dataset '{dataset_name}': {e}")
        print(f"   Falling back to wikitext-2 for testing...")
        try:
            dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')
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
    
    print(f"\n✅ Dataset splits:")
    print(f"  Train ({train_split}): {len(dataset[train_split]):,} examples")
    
    if val_split in dataset:
        print(f"  Validation ({val_split}): {len(dataset[val_split]):,} examples")
    else:
        print(f"  Validation: Not available (will use train split)")
        val_split = train_split
    
    if test_split in dataset:
        print(f"  Test ({test_split}): {len(dataset[test_split]):,} examples")
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
