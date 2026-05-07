"""
Production-ready Transformer model with proper architecture.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .generation_utils import apply_repetition_penalty


class RotaryEmbedding(nn.Module):
    """Rotary positional embeddings for attention queries and keys."""

    def __init__(self, head_dim, base=10000):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("RoPE requires an even attention head dimension")

        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len, device, dtype):
        positions = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(positions, self.inv_freq.to(device))
        cos = freqs.cos().view(1, 1, seq_len, -1).to(dtype=dtype)
        sin = freqs.sin().view(1, 1, seq_len, -1).to(dtype=dtype)
        return cos, sin


def apply_rotary_embedding(x, cos, sin):
    """Apply RoPE to tensors shaped (batch, heads, seq_len, head_dim)."""
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    rotated = torch.stack(
        (x_even * cos - x_odd * sin, x_even * sin + x_odd * cos),
        dim=-1,
    )
    return rotated.flatten(-2)


class CausalSelfAttention(nn.Module):
    """Causal self-attention with optional RoPE, GQA, and SDPA flash attention."""
    
    def __init__(
        self,
        d_model,
        num_heads,
        context_length,
        dropout=0.1,
        use_rope=False,
        gqa_num_kv_heads=None,
        use_flash_attention=False,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads
        self.num_kv_heads = gqa_num_kv_heads or num_heads
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by gqa_num_kv_heads")
        self.kv_group_size = self.num_heads // self.num_kv_heads
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.use_flash_attention = use_flash_attention
        self.dropout = dropout
        
        # Combined QKV projection keeps old checkpoint keys compatible when
        # num_kv_heads == num_heads, while allowing smaller K/V projections for GQA.
        self.c_attn = nn.Linear(d_model, d_model + 2 * self.kv_dim)
        self.c_proj = nn.Linear(d_model, d_model)
        
        self.attn_dropout = nn.Dropout(dropout)   # applied to attention weights
        self.resid_dropout = nn.Dropout(dropout)  # applied after output projection
        self.rotary_emb = RotaryEmbedding(self.head_dim) if use_rope else None
        
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(context_length, context_length))
            .view(1, 1, context_length, context_length)
        )

    def forward(self, x):
        B, T, C = x.size()

        qkv = self.c_attn(x)
        q, k, v = qkv.split((self.d_model, self.kv_dim, self.kv_dim), dim=2)
        
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if self.rotary_emb is not None:
            cos, sin = self.rotary_emb(T, device=x.device, dtype=q.dtype)
            q = apply_rotary_embedding(q, cos, sin)
            k = apply_rotary_embedding(k, cos, sin)

        if self.num_kv_heads != self.num_heads:
            k = k.repeat_interleave(self.kv_group_size, dim=1)
            v = v.repeat_interleave(self.kv_group_size, dim=1)

        if self.use_flash_attention:
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class FeedForward(nn.Module):
    """Position-wise feed-forward network with optional SwiGLU."""
    
    def __init__(self, d_model, dropout=0.1, use_swiglu=False):
        super().__init__()
        self.use_swiglu = use_swiglu

        if use_swiglu:
            hidden_dim = int((8 * d_model) / 3)
            self.gate_proj = nn.Linear(d_model, hidden_dim)
            self.up_proj = nn.Linear(d_model, hidden_dim)
            self.down_proj = nn.Linear(hidden_dim, d_model)
            self.dropout = nn.Dropout(dropout)
        else:
            self.net = nn.Sequential(
                nn.Linear(d_model, 4 * d_model),
                nn.GELU(),
                nn.Linear(4 * d_model, d_model),
                nn.Dropout(dropout)
            )
    
    def forward(self, x):
        if self.use_swiglu:
            x = F.silu(self.gate_proj(x)) * self.up_proj(x)
            return self.dropout(self.down_proj(x))
        return self.net(x)


class TransformerBlock(nn.Module):
    """Transformer block with pre-norm architecture and optional gradient checkpointing."""
    
    def __init__(
        self,
        d_model,
        num_heads,
        context_length,
        dropout=0.1,
        use_gradient_checkpointing=False,
        use_swiglu=False,
        use_rope=False,
        gqa_num_kv_heads=None,
        use_flash_attention=False,
    ):
        super().__init__()
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.ln_1 = nn.LayerNorm(d_model)  # layer norm before attention
        self.attn = CausalSelfAttention(
            d_model,
            num_heads,
            context_length,
            dropout,
            use_rope=use_rope,
            gqa_num_kv_heads=gqa_num_kv_heads,
            use_flash_attention=use_flash_attention,
        )
        self.ln_2 = nn.LayerNorm(d_model)  # layer norm before feed-forward
        self.mlp = FeedForward(d_model, dropout, use_swiglu=use_swiglu)

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
    
    PEFT-Compatible: This model includes required attributes for PEFT/LoRA:
    - config: Model configuration (required by PEFT for metadata)
    - generation_config: Generation parameters (required by PEFT generate)
    - main_input_name: Input parameter name (required by some PEFT wrappers)
    """
    
    def __init__(
        self,
        vocab_size,
        d_model,
        num_heads,
        num_layers,
        context_length,
        dropout=0.1,
        use_gradient_checkpointing=False,
        use_swiglu=False,
        use_rope=False,
        gqa_num_kv_heads=None,
        use_flash_attention=False,
    ):
        super().__init__()
        self.context_length = context_length
        self.d_model = d_model
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_rope = use_rope
        self.use_swiglu = use_swiglu
        self.gqa_num_kv_heads = gqa_num_kv_heads or num_heads
        self.use_flash_attention = use_flash_attention
        
        # PEFT Compatibility: Create config object
        # This is required by PEFT for saving/loading adapter metadata
        from transformers import GenerationConfig, PretrainedConfig
        
        # Use PretrainedConfig instead of SimpleNamespace so PEFT can iterate over it
        self.config = PretrainedConfig(
            vocab_size=vocab_size,
            hidden_size=d_model,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            max_position_embeddings=context_length,
            model_type="gpt",  
            is_encoder_decoder=False,
            d_model=d_model,  
            num_heads=num_heads,
            num_layers=num_layers,
            use_swiglu=use_swiglu,
            use_rope=use_rope,
            gqa_num_kv_heads=self.gqa_num_kv_heads,
            use_flash_attention=use_flash_attention,
            _name_or_path="0x-genesys/neo_weights_checkpoints"
        )
        
        # PEFT Compatibility: Generation config for peft.generate() support
        self.generation_config = GenerationConfig(
            max_length=context_length,
            bos_token_id=0,
            eos_token_id=vocab_size - 1,
            pad_token_id=vocab_size - 1,
        )
        
        # PEFT Compatibility: Required by some PEFT wrappers
        self.main_input_name = "input_ids"
        
        # Token and position embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = None if use_rope else nn.Embedding(context_length, d_model)
        self.drop = nn.Dropout(dropout)
        
        # Transformer blocks
        # Stack of identical transformer blocks with optional gradient checkpointing
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model,
                num_heads,
                context_length,
                dropout,
                use_gradient_checkpointing,
                use_swiglu=use_swiglu,
                use_rope=use_rope,
                gqa_num_kv_heads=self.gqa_num_kv_heads,
                use_flash_attention=use_flash_attention,
            )
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
        if non_embedding and self.position_embedding is not None:
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
        
        tok_emb = self.token_embedding(idx)
        if self.use_rope:
            x = self.drop(tok_emb)
        else:
            pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
            pos_emb = self.position_embedding(pos)
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
            if targets.shape != idx.shape:
                raise ValueError(
                    f"Expected targets shape {idx.shape}, got {targets.shape}. "
                    "DataLoader should return unshifted labels; model applies causal shift."
                )

            # Unconditionally apply causal shift: predict token[i+1] from token[i]
            shift_logits = logits[:, :-1, :].contiguous()
            shift_targets = targets[:, 1:].contiguous()
            
            # --- TPU-SAFE PAD NORMALIZATION ---
            # Instead of a Python 'if' statement checking the tensor content, 
            # we use tensor math to map any -1 padding values to -100.
            # This compiles flawlessly on XLA without triggering a graph break.
            shift_targets = torch.where(
                shift_targets == -1, 
                torch.tensor(-100, dtype=shift_targets.dtype, device=shift_targets.device), 
                shift_targets
            )
            
            # Flatten batch and time dimensions for cross-entropy, 
            # using the now-unified -100 ignore_index.
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_targets.view(-1),
                ignore_index=-100
            )
        
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx,
        max_new_tokens,
        temperature=1.0,
        top_k=None,
        top_p=None,
        repetition_penalty=1.2,
        eos_token_id=None,
    ):
        """
        Generate new tokens autoregressively.
        
        Args:
            idx: Starting token indices (B, T)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Keep only top k tokens for sampling
            top_p: Nucleus sampling threshold
            repetition_penalty: Penalty for previously generated tokens (1.0 = disabled)
            eos_token_id: EOS token ID (int) or list of EOS token IDs for early stopping
            
        Returns:
            Generated token indices (B, T + max_new_tokens)
        """
        eos_ids = None
        if eos_token_id is not None:
            if isinstance(eos_token_id, int):
                eos_ids = torch.tensor([eos_token_id], device=idx.device, dtype=idx.dtype)
            else:
                eos_ids = torch.tensor(list(eos_token_id), device=idx.device, dtype=idx.dtype)

        for _ in range(max_new_tokens):
            # Crop context if needed
            # Truncate the context to the last context_length tokens if it has grown too long
            idx_cond = idx if idx.size(1) <= self.context_length else idx[:, -self.context_length:]
            
            # Forward pass
            # Get logits for the last position only — that's the next-token distribution
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature  # divide by temperature before sampling

            # Apply repetition penalty before top-k/top-p filtering
            apply_repetition_penalty(logits, idx, repetition_penalty)
            
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

            # Early stopping if model predicts any EOS token
            if eos_ids is not None:
                hit_eos = (idx_next == eos_ids.view(1, -1)).any(dim=1).any()
                if hit_eos:
                    break
        
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
    model_config = config['model']
    use_gradient_checkpointing = model_config.get('use_gradient_checkpointing', False)
    use_swiglu = model_config.get('use_swiglu', False)
    use_rope = model_config.get('use_rope', False)
    gqa_num_kv_heads = model_config.get('gqa_num_kv_heads')
    use_flash_attention = model_config.get('use_flash_attention', False)
    
    model = DecoderOnlyTransformer(
        vocab_size=model_config['vocab_size'],
        d_model=model_config['d_model'],
        num_heads=model_config['num_heads'],
        num_layers=model_config['num_layers'],
        context_length=model_config['context_length'],
        dropout=model_config['dropout'],
        use_gradient_checkpointing=use_gradient_checkpointing,
        use_swiglu=use_swiglu,
        use_rope=use_rope,
        gqa_num_kv_heads=gqa_num_kv_heads,
        use_flash_attention=use_flash_attention,
    )
    
    if use_gradient_checkpointing:
        print("✅ Gradient checkpointing enabled (saves memory, slightly slower)")
    if use_swiglu:
        print("✅ SwiGLU MLP enabled")
    if use_rope:
        print("✅ RoPE positional embeddings enabled")
    if gqa_num_kv_heads:
        print(f"✅ GQA enabled ({model_config['num_heads']} query heads, {gqa_num_kv_heads} KV heads)")
    if use_flash_attention:
        print("✅ Flash attention enabled via PyTorch SDPA")
    
    return model
