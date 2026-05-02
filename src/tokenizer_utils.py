"""
Tokenizer utilities to handle transformers version compatibility.

This module provides helper functions for tokenization that work around
the PyTorch version warning in transformers library.
"""
import torch


def load_tokenizer():
    """
    Load the default tokenizer (tiktoken cl100k_base).
    
    Returns:
        Tokenizer wrapper compatible with HuggingFace interface
    """
    try:
        import tiktoken
    except ImportError:
        raise ImportError("tiktoken is required. Install with: pip install tiktoken")
    
    encoding = tiktoken.get_encoding("cl100k_base")
    
    # Create a wrapper to match HuggingFace interface
    class TiktokenWrapper:
        def __init__(self, encoding):
            self.encoding = encoding
            self.vocab_size = encoding.n_vocab
            self.eos_token = "<|endoftext|>"
            self.pad_token = "<|endoftext|>"
            # Get special token IDs properly
            self.eos_token_id = encoding.encode_single_token(self.eos_token)
            self.pad_token_id = self.eos_token_id
        
        def encode(self, text, **kwargs):
            return self.encoding.encode(text, allowed_special='all')
        
        def decode(self, tokens, **kwargs):
            return self.encoding.decode(tokens)
        
        def add_special_tokens(self, special_tokens_dict):
            """
            Add special tokens (compatibility method).
            For tiktoken, special tokens are already in vocabulary.
            
            Args:
                special_tokens_dict: Dictionary of special tokens
                
            Returns:
                Number of tokens added (always 0 for tiktoken)
            """
            # Tiktoken already has special tokens in vocabulary
            # This is just for compatibility with HuggingFace interface
            return 0
        
        def save_pretrained(self, save_directory):
            """
            Save tokenizer (compatibility method).
            For tiktoken, this is a no-op since it's a fixed vocabulary.
            
            Args:
                save_directory: Directory to save to
            """
            # Tiktoken uses a fixed vocabulary, no need to save
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
