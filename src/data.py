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
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer']['type'])
    
    # Set pad token if not exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Update vocab size in config
    config['model']['vocab_size'] = len(tokenizer)
    print(f"Tokenizer vocabulary size: {len(tokenizer)}")
    
    # Load dataset from HuggingFace
    try:
        if config['data']['dataset_name'] == 'wikitext':
            dataset = load_dataset(
                'wikitext',
                config['data']['dataset_config']
            )
        elif config['data']['dataset_name'] == 'openwebtext':
            dataset = load_dataset('openwebtext')
        elif config['data']['dataset_name'] == 'bookcorpus':
            dataset = load_dataset('bookcorpus')
        elif config['data']['dataset_name'] == 'tiny_shakespeare':
            # tiny_shakespeare is deprecated, use Salesforce/wikitext instead
            print("Note: tiny_shakespeare is deprecated. Using wikitext-2 instead.")
            dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')
        else:
            # Try to load any dataset by name
            dataset = load_dataset(config['data']['dataset_name'])
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Falling back to wikitext-2 for testing...")
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')
    
    # Filter out empty texts
    def filter_empty(example):
        return len(example['text'].strip()) > 0
    
    dataset = dataset.filter(filter_empty)
    
    print(f"Dataset loaded:")
    print(f"  Train: {len(dataset[config['data']['train_split']])} examples")
    if config['data']['val_split'] in dataset:
        print(f"  Validation: {len(dataset[config['data']['val_split']])} examples")
    if config['data']['test_split'] in dataset:
        print(f"  Test: {len(dataset[config['data']['test_split']])} examples")
    
    # Create datasets
    train_dataset = TextDataset(
        dataset[config['data']['train_split']],
        tokenizer,
        config['data']['max_length']
    )
    
    val_dataset = None
    if config['data']['val_split'] in dataset:
        val_dataset = TextDataset(
            dataset[config['data']['val_split']],
            tokenizer,
            config['data']['max_length']
        )
    
    test_dataset = None
    if config['data']['test_split'] in dataset:
        test_dataset = TextDataset(
            dataset[config['data']['test_split']],
            tokenizer,
            config['data']['max_length']
        )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=0,  # Use 0 for Mac to avoid multiprocessing issues
        collate_fn=lambda x: collate_fn(x, config['data']['max_length']),
        pin_memory=False  # Disable for CPU
    )
    
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=0,  # Use 0 for Mac
            collate_fn=lambda x: collate_fn(x, config['data']['max_length']),
            pin_memory=False
        )
    
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=0,  # Use 0 for Mac
            collate_fn=lambda x: collate_fn(x, config['data']['max_length']),
            pin_memory=False
        )
    
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
