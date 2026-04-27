import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, context_length):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads
        
        # Key, Query, Value projections batched into one linear layer for efficiency
        self.c_attn = nn.Linear(d_model, 3 * d_model)
        self.c_proj = nn.Linear(d_model, d_model)
        
        # Causal mask to ensure attention is only applied to the left (past tokens)
        self.register_buffer("bias", torch.tril(torch.ones(context_length, context_length))
                                     .view(1, 1, context_length, context_length))

    def forward(self, x):
        B, T, C = x.size() # Batch size, Time (sequence length), Channels (d_model)
        
        # Calculate Q, K, V
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.d_model, dim=2)
        
        # Reshape for multi-head attention: (B, num_heads, T, head_dim)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled Dot-Product Attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / (k.size(-1) ** 0.5))
        
        # Apply the causal mask (replace 0s in the mask with -infinity before softmax)
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        
        # Output projection
        y = att @ v # (B, num_heads, T, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, context_length):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, num_heads, context_length)
        self.ln_2 = nn.LayerNorm(d_model)
        # Feed Forward (usually 4x the dimension of the model internally)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )

    def forward(self, x):
        # Notice the residual connections (x + ...)
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class DecoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, context_length):
        super().__init__()
        self.context_length = context_length
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(context_length, d_model)
        
        self.blocks = nn.ModuleList([TransformerBlock(d_model, num_heads, context_length) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False) # Language modeling head

    def forward(self, idx):
        B, T = idx.size()
        
        tok_emb = self.token_embedding(idx) # (B, T, d_model)
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.position_embedding(pos) # (T, d_model)
        
        x = tok_emb + pos_emb
        
        for block in self.blocks:
            x = block(x)
            
        x = self.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        return logits