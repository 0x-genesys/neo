"""
Tokenizer utilities to handle transformers version compatibility.

This module provides helper functions for tokenization that work around
the PyTorch version warning in transformers library.
"""
import torch


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
