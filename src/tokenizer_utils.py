"""
Tokenizer utilities to handle transformers version compatibility.

This module provides helper functions for tokenization that work around
the PyTorch version warning in transformers library.
"""
import torch


def load_tokenizer():
    """
    Load the default tokenizer (tiktoken cl100k_base) with custom special tokens.
    
    Returns:
        Tokenizer wrapper compatible with HuggingFace interface
    """
    try:
        import tiktoken
    except ImportError:
        raise ImportError("tiktoken is required. Install with: pip install tiktoken")
    
    # Load the base encoding
    base = tiktoken.get_encoding("cl100k_base")
    
    # Define our special tokens exactly as specified
    special_tokens = {
        "<|endoftext|>": 100257,
        "<|fim_prefix|>": 100258,
        "<|fim_middle|>": 100259,
        "<|fim_suffix|>": 100260,
        "<|im_start|>": 100264,
        "<|im_end|>": 100265,
        "<|endofprompt|>": 100276,
    }
    
    # Create a new encoding by merging base special tokens with our new ones
    # Use the same name as base for consistency
    encoding = tiktoken.Encoding(
        name=base.name,
        pat_str=base._pat_str,
        mergeable_ranks=base._mergeable_ranks,
        special_tokens={**base._special_tokens, **special_tokens}
    )
    
    # Create a wrapper to match HuggingFace interface
    class TiktokenWrapper:
        def __init__(self, encoding):
            self.encoding = encoding
            self.vocab_size = encoding.n_vocab
            self.eos_token = "<|endoftext|>"
            self.pad_token = "<|endoftext|>"
            
            # Get special token IDs properly
            # These are now guaranteed to exist because we added them above
            self.eos_token_id = encoding.encode_single_token(self.eos_token)
            self.pad_token_id = self.eos_token_id
            self.im_start_id = encoding.encode_single_token("<|im_start|>")
            self.im_end_id = encoding.encode_single_token("<|im_end|>")
            
            print(f"✅ Tokenizer loaded with {len(special_tokens)} custom special tokens.")
            print(f"✅ Chat tokens registered: <|im_start|> (ID: {self.im_start_id}), <|im_end|> (ID: {self.im_end_id})")
        
        def encode(self, text, **kwargs):
            # Always use allowed_special='all' to encode special tokens properly
            return self.encoding.encode(text, allowed_special='all')
        
        def decode(self, tokens, **kwargs):
            return self.encoding.decode(tokens)
        
        def add_special_tokens(self, special_tokens_dict):
            """
            Add special tokens (compatibility method).
            For tiktoken, special tokens are already in vocabulary.
            """
            return 0
        
        def save_pretrained(self, save_directory):
            """
            Save tokenizer (compatibility method).
            """
            pass
        
        def __len__(self):
            return self.vocab_size
    
    return TiktokenWrapper(encoding)


def encode_to_tensor(tokenizer, text, device='cpu'):
    """
    Encode text to PyTorch tensor.
    
    Workaround for transformers library warning about PyTorch version.
    Instead of using tokenizer.encode(text, return_tensors='pt'), we
    manually convert to tensor.
    
    Args:
        tokenizer: HuggingFace tokenizer
        text: Text to encode
        device: Device to place tensor on
        
    Returns:
        torch.Tensor: Encoded text as tensor of shape (1, seq_len)
    """
    # Encode to list of integers
    token_ids = tokenizer.encode(text)
    
    # Convert to tensor
    tensor = torch.tensor([token_ids], dtype=torch.long)
    
    # Move to device
    if device is not None:
        tensor = tensor.to(device)
    
    return tensor


def encode_batch_to_tensor(tokenizer, texts, device='cpu', padding=True, max_length=None):
    """
    Encode multiple texts to PyTorch tensor with optional padding.
    
    Args:
        tokenizer: HuggingFace tokenizer
        texts: List of texts to encode
        device: Device to place tensor on
        padding: Whether to pad sequences to same length
        max_length: Maximum sequence length (truncate if longer)
        
    Returns:
        torch.Tensor: Encoded texts as tensor of shape (batch_size, seq_len)
    """
    # Encode all texts
    token_ids_list = [tokenizer.encode(text) for text in texts]
    
    # Truncate if needed
    if max_length is not None:
        token_ids_list = [ids[:max_length] for ids in token_ids_list]
    
    # Pad if needed
    if padding:
        max_len = max(len(ids) for ids in token_ids_list)
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        
        token_ids_list = [
            ids + [pad_id] * (max_len - len(ids))
            for ids in token_ids_list
        ]
    
    # Convert to tensor
    tensor = torch.tensor(token_ids_list, dtype=torch.long)
    
    # Move to device
    if device is not None:
        tensor = tensor.to(device)
    
    return tensor


def decode_from_tensor(tokenizer, tensor, skip_special_tokens=True):
    """
    Decode PyTorch tensor to text.
    
    Args:
        tokenizer: HuggingFace tokenizer
        tensor: Tensor of token IDs (can be 1D or 2D)
        skip_special_tokens: Whether to skip special tokens in output
        
    Returns:
        str or List[str]: Decoded text(s)
    """
    # Handle 1D tensor (single sequence)
    if tensor.dim() == 1:
        token_ids = tensor.tolist()
        return tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
    
    # Handle 2D tensor (batch)
    elif tensor.dim() == 2:
        texts = []
        for i in range(tensor.size(0)):
            token_ids = tensor[i].tolist()
            text = tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
            texts.append(text)
        return texts
    
    else:
        raise ValueError(f"Expected 1D or 2D tensor, got {tensor.dim()}D")


# Convenience aliases
encode = encode_to_tensor
decode = decode_from_tensor
