"""
Production-ready Transformer model with proper architecture.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with dropout."""
    
    def __init__(self, d_model, num_heads, context_length, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.d_model = d_model
        # Each head attends to a slice of the full embedding dimension
        self.head_dim = d_model // num_heads
        
        # Combined QKV projection for efficiency
        # Combined QKV projection for efficiency — one matmul instead of three
        self.c_attn = nn.Linear(d_model, 3 * d_model)
        # Output projection maps concatenated head outputs back to d_model
        self.c_proj = nn.Linear(d_model, d_model)
        
        # Regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)   # applied to attention weights
        self.resid_dropout = nn.Dropout(dropout)  # applied after output projection
        
        # Causal mask
        # Causal mask: lower-triangular matrix ensures each position can only
        # attend to itself and earlier positions (no future token leakage)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(context_length, context_length))
            .view(1, 1, context_length, context_length)
        )

    def forward(self, x):
        B, T, C = x.size()

        # Calculate Q, K, V
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality

        # Project input to queries, keys, and values in a single pass
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.d_model, dim=2)
        
        # Reshape for multi-head attention
        # Reshape from (B, T, C) to (B, num_heads, T, head_dim) for parallel head computation
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        # Scaled dot-product attention: scale by 1/sqrt(head_dim) to keep gradients stable
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # Mask out future positions by setting their logits to -inf before softmax
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        # Softmax normalizes across the key dimension, turning logits into probabilities
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        # Apply attention to values
        # Weighted sum of values using the attention distribution
        y = att @ v
        # Re-assemble all head outputs side by side: (B, num_heads, T, head_dim) -> (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Output projection
        # Final linear projection and residual dropout
        y = self.resid_dropout(self.c_proj(y))
        return y


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""
    
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        # Expand to 4x d_model in the hidden layer, then project back — standard GPT ratio
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """Transformer block with pre-norm architecture and optional gradient checkpointing."""
    
    def __init__(self, d_model, num_heads, context_length, dropout=0.1, use_gradient_checkpointing=False):
        super().__init__()
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.ln_1 = nn.LayerNorm(d_model)  # layer norm before attention
        self.attn = CausalSelfAttention(d_model, num_heads, context_length, dropout)
        self.ln_2 = nn.LayerNorm(d_model)  # layer norm before feed-forward
        self.mlp = FeedForward(d_model, dropout)

    def forward(self, x):
        # Use gradient checkpointing if enabled and in training mode
        if self.training and self.use_gradient_checkpointing:
            return self._forward_with_checkpointing(x)
        else:
            return self._forward(x)
    
    def _forward(self, x):
        """Standard forward pass."""
        # Pre-norm variant: normalize before each sub-layer, then add residual.
        # This is more stable to train than the original post-norm formulation.
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
    
    def _forward_with_checkpointing(self, x):
        """Forward pass with gradient checkpointing to save memory."""
        from torch.utils.checkpoint import checkpoint
        
        # Checkpoint attention block
        x = x + checkpoint(self._attn_forward, x, use_reentrant=False)
        # Checkpoint MLP block
        x = x + checkpoint(self._mlp_forward, x, use_reentrant=False)
        return x
    
    def _attn_forward(self, x):
        """Attention forward for checkpointing."""
        return self.attn(self.ln_1(x))
    
    def _mlp_forward(self, x):
        """MLP forward for checkpointing."""
        return self.mlp(self.ln_2(x))


class DecoderOnlyTransformer(nn.Module):
    """
    Production-ready decoder-only transformer for language modeling.
    Similar to GPT architecture.
    """
    
    def __init__(self, vocab_size, d_model, num_heads, num_layers, 
                 context_length, dropout=0.1, use_gradient_checkpointing=False):
        super().__init__()
        self.context_length = context_length
        self.d_model = d_model
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        # Token and position embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        # Learned absolute position embeddings, one vector per position up to context_length
        self.position_embedding = nn.Embedding(context_length, d_model)
        self.drop = nn.Dropout(dropout)
        
        # Transformer blocks
        # Stack of identical transformer blocks with optional gradient checkpointing
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, context_length, dropout, use_gradient_checkpointing)
            for _ in range(num_layers)
        ])
        
        # Final layer norm and output projection
        # Final layer norm applied before the output projection
        self.ln_f = nn.LayerNorm(d_model)
        # Projects hidden states to vocabulary logits (no bias, following GPT-2)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Weight tying (share weights between token embedding and output projection)
        # Weight tying: share weights between token embedding and output projection.
        # Reduces parameter count and often improves perplexity.
        self.token_embedding.weight = self.lm_head.weight
        
        # Initialize weights
        # Initialize all weights before training
        self.apply(self._init_weights)
        
        # Report number of parameters
        print(f"Model initialized with {self.get_num_params()/1e6:.2f}M parameters")

    def _init_weights(self, module):
        """Initialize weights using GPT-2 style initialization."""
        if isinstance(module, nn.Linear):
            # Small normal distribution keeps activations well-scaled at init
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self, non_embedding=False):
        """Return the number of parameters in the model."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            # Optionally exclude position embeddings, which don't contribute to
            # the model's representational capacity in the same way as other params
            n_params -= self.position_embedding.weight.numel()
        return n_params

    def forward(self, idx=None, input_ids=None, targets=None, attention_mask=None, **kwargs):
        """
        Forward pass.
        
        Args:
            idx: Input token indices (B, T) - our standard parameter name
            input_ids: Alternative name for idx (for PEFT compatibility)
            targets: Target token indices (B, T) for computing loss
            attention_mask: Attention mask (ignored, for PEFT compatibility)
            **kwargs: Additional arguments (ignored, for PEFT compatibility)
            
        Returns:
            logits: (B, T, vocab_size) if targets is None
            (logits, loss): if targets is provided
        """
        # Handle both idx and input_ids for PEFT compatibility
        if input_ids is not None:
            idx = input_ids
        elif idx is None:
            raise ValueError("Either idx or input_ids must be provided")
        
        B, T = idx.size()
        assert T <= self.context_length, f"Sequence length {T} exceeds context length {self.context_length}"
        
        # Token and position embeddings
        # Look up token embeddings and create a position index tensor [0, 1, ..., T-1]
        tok_emb = self.token_embedding(idx)
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.position_embedding(pos)
        
        # Sum token and position embeddings, then apply dropout
        x = self.drop(tok_emb + pos_emb)
        
        # Apply transformer blocks
        # Pass through each transformer block sequentially
        for block in self.blocks:
            x = block(x)
            
        # Final layer norm before projecting to vocabulary
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        # Calculate loss if targets provided
        loss = None
        if targets is not None:
            # Flatten batch and time dimensions for cross-entropy;
            # ignore_index=-1 allows padding positions to be masked out
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )
        
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None):
        """
        Generate new tokens autoregressively.
        
        Args:
            idx: Starting token indices (B, T)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Keep only top k tokens for sampling
            top_p: Nucleus sampling threshold
            
        Returns:
            Generated token indices (B, T + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # Crop context if needed
            # Truncate the context to the last context_length tokens if it has grown too long
            idx_cond = idx if idx.size(1) <= self.context_length else idx[:, -self.context_length:]
            
            # Forward pass
            # Get logits for the last position only — that's the next-token distribution
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature  # divide by temperature before sampling
            
            # Apply top-k filtering
            # Apply top-k filtering: zero out all logits below the k-th highest value
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Apply top-p (nucleus) filtering
            # Apply top-p (nucleus) filtering: keep the smallest set of tokens whose
            # cumulative probability exceeds top_p, discarding the long tail
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift right by one so the token that pushes cumprob over the threshold is kept
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # Scatter the removal mask back to the original (unsorted) logit order
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('Inf')
            
            # Sample from distribution
            # Convert filtered logits to probabilities and draw one sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            # Append the new token to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx
    
    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        """
        Prepare inputs for generation (required by PEFT).
        
        Args:
            input_ids: Input token IDs
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with model inputs
        """
        return {"input_ids": input_ids}



def create_model(config):
    """Factory function to create model from config."""
    use_gradient_checkpointing = config['model'].get('use_gradient_checkpointing', False)
    
    model = DecoderOnlyTransformer(
        vocab_size=config['model']['vocab_size'],
        d_model=config['model']['d_model'],
        num_heads=config['model']['num_heads'],
        num_layers=config['model']['num_layers'],
        context_length=config['model']['context_length'],
        dropout=config['model']['dropout'],
        use_gradient_checkpointing=use_gradient_checkpointing
    )
    
    if use_gradient_checkpointing:
        print("✅ Gradient checkpointing enabled (saves memory, slightly slower)")
    
    return model

